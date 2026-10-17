// glBlitFramebuffer (GL 3.3 section 4.3.2).
//
// Same-size, same-format colour and depth copies use the blit encoder.
// Everything else renders a full-target triangle whose fragments map their
// window position into the source rectangle (utility shaders wmgl_blit_*).
// Stencil has no shader export path: it is computed on the CPU and copied,
// or written bit by bit with stencil-replace draws into multisampled targets.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/context.h"
#include "core/draw.h"
#include "core/metal/device.h"
#include "core/query.h"
#include "core/surface.h"
#include "core/texture_storage.h"
#include "core/util/log.h"
#include "gl_api.gen.h"

namespace wmgl::api {
namespace {

// The GL format of a framebuffer attachment (null when none).
const FormatInfo* attachedFormat(const Attachment& a) {
    if (a.type == GL_RENDERBUFFER && a.renderbuffer) return a.renderbuffer->format;
    if (a.type == GL_TEXTURE && a.texture) {
        const int face = a.cubeFace ? static_cast<int>(a.cubeFace - GL_TEXTURE_CUBE_MAP_POSITIVE_X) : 0;
        if (a.level < 0 || a.level >= Texture::kMaxLevels) return nullptr;
        return a.texture->image(face, a.level).format;
    }
    return nullptr;
}

// The draw framebuffer's depth or stencil format (the default
// framebuffer's: Surface::depthStencilFormat).
const FormatInfo* drawDepthStencilFormat(Context& ctx, bool stencil) {
    if (const Framebuffer* fb = ctx.state.drawFramebuffer.get()) return attachedFormat(stencil ? fb->stencil : fb->depth);
    Surface* s = ctx.drawSurface();
    if (!s) return nullptr;
    return formatInfo(s->depthStencilFormat());
}

struct BlitParams {
    float dst[4];
    float src[4];
    float srcSize[4];
};

struct Rect {
    int x0, y0, x1, y1;  // half-open, x0 <= x1
    bool empty() const { return x1 <= x0 || y1 <= y0; }
};

Rect normalized(GLint x0, GLint y0, GLint x1, GLint y1) {
    return {std::min(x0, x1), std::min(y0, y1), std::max(x0, x1), std::max(y0, y1)};
}

Rect intersect(Rect a, Rect b) {
    return {std::max(a.x0, b.x0), std::max(a.y0, b.y0), std::min(a.x1, b.x1), std::min(a.y1, b.y1)};
}

// `writeAlpha` false keeps the stored alpha of an RGB format (always one).
id<MTLRenderPipelineState> blitPipeline(NSString* fragment, MTLPixelFormat color, MTLPixelFormat depth,
                                        MTLPixelFormat stencil, uint32_t samples, bool writeAlpha = true) {
    static std::mutex mutex;
    static std::unordered_map<std::string, id<MTLRenderPipelineState>> cache;
    const std::string key = std::string(fragment.UTF8String) + "/" + std::to_string(color) + "/" + std::to_string(depth) +
                            "/" + std::to_string(stencil) + "/" + std::to_string(samples) + (writeAlpha ? "" : "/rgb");
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    id<MTLLibrary> library = Device::get().utilityLibrary();
    MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = [library newFunctionWithName:@"wmgl_clear_vs"];
    desc.fragmentFunction = [library newFunctionWithName:fragment];
    desc.colorAttachments[0].pixelFormat = color;
    if (!writeAlpha) desc.colorAttachments[0].writeMask = MTLColorWriteMaskRed | MTLColorWriteMaskGreen | MTLColorWriteMaskBlue;
    desc.depthAttachmentPixelFormat = depth;
    desc.stencilAttachmentPixelFormat = stencil;
    desc.rasterSampleCount = samples;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [Device::get().mtl() newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!pipeline) WMGL_ERR("blit pipeline %s: %s", fragment.UTF8String, error.localizedDescription.UTF8String);
    cache[key] = pipeline;
    return pipeline;
}

// A single-level 2D (or 2D multisample) texture holding the source slice.
id<MTLTexture> sourceView(Context& ctx, const RenderTargetDesc::Target& t, uint32_t width, uint32_t height) {
    id<MTLTexture> texture = t.texture;
    const bool ms = texture.sampleCount > 1;
    if (texture.textureType == MTLTextureType3D) {
        // A 3D slice cannot be viewed as 2D: copy it out.
        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:texture.pixelFormat
                                                                                         width:width
                                                                                        height:height
                                                                                     mipmapped:NO];
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> copy = [Device::get().mtl() newTextureWithDescriptor:desc];
        [ctx.stream().blit() copyFromTexture:texture sourceSlice:0 sourceLevel:t.level
                                sourceOrigin:MTLOriginMake(0, 0, t.slice) sourceSize:MTLSizeMake(width, height, 1)
                                   toTexture:copy destinationSlice:0 destinationLevel:0
                           destinationOrigin:MTLOriginMake(0, 0, 0)];
        return copy;
    }
    if ((texture.textureType == MTLTextureType2D || texture.textureType == MTLTextureType2DMultisample) && t.level == 0 &&
        texture.mipmapLevelCount == 1)
        return texture;
    return [texture newTextureViewWithPixelFormat:texture.pixelFormat
                                      textureType:ms ? MTLTextureType2DMultisample : MTLTextureType2D
                                           levels:NSMakeRange(t.level, 1)
                                           slices:NSMakeRange(t.slice, 1)];
}

bool sameDirection(GLint a0, GLint a1, GLint b0, GLint b1) { return (a1 - a0) == (b1 - b0); }

// Destination pixels the blit may write: the destination rectangle, clipped
// to the target and (when enabled) the scissor box.
Rect writableRect(Context& ctx, const Rect& dst, uint32_t width, uint32_t height) {
    Rect r = intersect(dst, {0, 0, static_cast<int>(width), static_cast<int>(height)});
    const ContextState& st = ctx.state;
    if ((st.scissorTests & 1u))
        r = intersect(r, {st.scissors[0][0], st.scissors[0][1], st.scissors[0][0] + std::max(0, st.scissors[0][2]),
                          st.scissors[0][1] + std::max(0, st.scissors[0][3])});
    return r;
}

// Copy with the blit encoder when the blit is a 1:1 copy between identical formats.
bool tryCopy(Context& ctx, const RenderTargetDesc::Target& src, uint32_t srcW, uint32_t srcH,
             const RenderTargetDesc::Target& dst, uint32_t dstW, uint32_t dstH, const GLint s[4], const GLint d[4]) {
    if (src.texture.pixelFormat != dst.texture.pixelFormat || src.texture.sampleCount != dst.texture.sampleCount) return false;
    if (!sameDirection(s[0], s[2], d[0], d[2]) || !sameDirection(s[1], s[3], d[1], d[3])) return false;
    if (src.texture == dst.texture && src.level == dst.level && src.slice == dst.slice) return false;
    // Clip the destination, then shift the source by the same amount.
    const Rect dstRect = normalized(d[0], d[1], d[2], d[3]);
    Rect clip = writableRect(ctx, dstRect, dstW, dstH);
    const int dx = std::min(s[0], s[2]) - dstRect.x0, dy = std::min(s[1], s[3]) - dstRect.y0;
    clip = intersect(clip, {-dx, -dy, static_cast<int>(srcW) - dx, static_cast<int>(srcH) - dy});
    if (clip.empty()) return true;
    const bool src3D = src.texture.textureType == MTLTextureType3D, dst3D = dst.texture.textureType == MTLTextureType3D;
    [ctx.stream().blit() copyFromTexture:src.texture sourceSlice:src3D ? 0 : src.slice sourceLevel:src.level
                            sourceOrigin:MTLOriginMake(clip.x0 + dx, clip.y0 + dy, src3D ? src.slice : 0)
                              sourceSize:MTLSizeMake(clip.x1 - clip.x0, clip.y1 - clip.y0, 1)
                               toTexture:dst.texture destinationSlice:dst3D ? 0 : dst.slice destinationLevel:dst.level
                       destinationOrigin:MTLOriginMake(clip.x0, clip.y0, dst3D ? dst.slice : 0)];
    return true;
}

void drawBlit(Context& ctx, const RenderTargetDesc& target, id<MTLRenderPipelineState> pipeline, id<MTLTexture> source,
              const BlitParams& params, const Rect& scissor, bool depth, bool linear) {
    id<MTLRenderCommandEncoder> encoder = ctx.stream().render(target);
    queriesSuspendVisibility(ctx, encoder);  // blits do not count towards occlusion queries
    [encoder setRenderPipelineState:pipeline];
    DepthStencilKey ds;
    std::memset(&ds, 0, sizeof(ds));
    ds.depthCompare = static_cast<uint8_t>(MTLCompareFunctionAlways);
    ds.depthWrite = depth;
    [encoder setDepthStencilState:ctx.share().pipelines().depthStencil(ds)];
    MTLViewport viewport = {0, 0, static_cast<double>(target.width), static_cast<double>(target.height), 0, 1};
    [encoder setViewport:viewport];
    [encoder setScissorRect:{static_cast<NSUInteger>(scissor.x0), static_cast<NSUInteger>(scissor.y0),
                             static_cast<NSUInteger>(scissor.x1 - scissor.x0), static_cast<NSUInteger>(scissor.y1 - scissor.y0)}];
    [encoder setCullMode:MTLCullModeNone];
    [encoder setTriangleFillMode:MTLTriangleFillModeFill];
    [encoder setDepthClipMode:MTLDepthClipModeClamp];
    [encoder setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
    const float zero[4] = {};
    [encoder setVertexBytes:zero length:sizeof(zero) atIndex:0];
    [encoder setFragmentTexture:source atIndex:0];
    SamplerKey key;
    std::memset(&key, 0, sizeof(key));
    key.minFilter = key.magFilter = linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    key.mipFilter = MTLSamplerMipFilterNotMipmapped;
    key.sAddress = key.tAddress = key.rAddress = MTLSamplerAddressModeClampToEdge;
    key.compare = MTLCompareFunctionNever;
    key.maxAnisotropy = 1;
    key.normalized = 1;
    key.lodMax = 1000.0f;
    [encoder setFragmentSamplerState:ctx.share().pipelines().sampler(key) atIndex:0];
    [encoder setFragmentBytes:&params length:sizeof(params) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    ctx.stream().noteDraw();
}

// Resolves a multisampled depth/stencil level to a single-sample texture
// (sample 0 for both aspects, as GL allows any sample for stencil).
RenderTargetDesc::Target resolveDepthStencil(Context& ctx, const RenderTargetDesc::Target& src, uint32_t width,
                                             uint32_t height) {
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:src.texture.pixelFormat
                                                                                     width:width
                                                                                    height:height
                                                                                 mipmapped:NO];
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> resolved = [Device::get().mtl() newTextureWithDescriptor:desc];
    ctx.stream().resolvePendingClear();
    ctx.stream().endEncoder();
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    const bool combined = src.texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
    if (combined || src.texture.pixelFormat != MTLPixelFormatStencil8) {
        pass.depthAttachment.texture = src.texture;
        pass.depthAttachment.resolveTexture = resolved;
        pass.depthAttachment.loadAction = MTLLoadActionLoad;
        pass.depthAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
        pass.depthAttachment.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;
    }
    if (combined || src.texture.pixelFormat == MTLPixelFormatStencil8) {
        pass.stencilAttachment.texture = src.texture;
        pass.stencilAttachment.resolveTexture = resolved;
        pass.stencilAttachment.loadAction = MTLLoadActionLoad;
        pass.stencilAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
        pass.stencilAttachment.stencilResolveFilter = MTLMultisampleStencilResolveFilterSample0;
    }
    [[ctx.stream().commandBuffer() renderCommandEncoderWithDescriptor:pass] endEncoding];
    RenderTargetDesc::Target out;
    out.texture = resolved;
    return out;
}

// Writes per-pixel stencil `values` (0x100: keep) over `clip` of a
// multisampled target: a zeroing pass, then one stencil-replace pass per bit.
void writeStencilMultisample(Context& ctx, const RenderTargetDesc::Target& dst, MTLPixelFormat depthFormat,
                             uint32_t samples, uint32_t dstW, uint32_t dstH, const Rect& clip,
                             const std::vector<uint16_t>& values) {
    const int w = clip.x1 - clip.x0, h = clip.y1 - clip.y0;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Uint
                                                                                    width:static_cast<NSUInteger>(w)
                                                                                   height:static_cast<NSUInteger>(h)
                                                                                mipmapped:NO];
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [Device::get().mtl() newTextureWithDescriptor:desc];
    [texture replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:values.data()
               bytesPerRow:static_cast<NSUInteger>(w) * 2];
    RenderTargetDesc target;
    target.stencil = dst;
    const bool combined = dst.texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
    if (combined) target.depth = dst;
    target.width = dstW;
    target.height = dstH;
    target.samples = samples;
    id<MTLRenderPipelineState> pipeline = blitPipeline(@"wmgl_blit_stencil_bit_fs", MTLPixelFormatInvalid,
                                                       combined ? depthFormat : MTLPixelFormatInvalid,
                                                       dst.texture.pixelFormat, samples);
    if (!pipeline) return;
    id<MTLRenderCommandEncoder> encoder = ctx.stream().render(target);
    queriesSuspendVisibility(ctx, encoder);  // blits do not count towards occlusion queries
    [encoder setRenderPipelineState:pipeline];
    [encoder setViewport:{0, 0, static_cast<double>(dstW), static_cast<double>(dstH), 0, 1}];
    [encoder setScissorRect:{static_cast<NSUInteger>(clip.x0), static_cast<NSUInteger>(clip.y0), static_cast<NSUInteger>(w),
                             static_cast<NSUInteger>(h)}];
    [encoder setCullMode:MTLCullModeNone];
    [encoder setTriangleFillMode:MTLTriangleFillModeFill];
    [encoder setDepthClipMode:MTLDepthClipModeClamp];
    [encoder setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
    const float zero[4] = {};
    [encoder setVertexBytes:zero length:sizeof(zero) atIndex:0];
    [encoder setFragmentTexture:texture atIndex:0];
    for (int pass = 0; pass <= 8; ++pass) {
        DepthStencilKey ds;
        std::memset(&ds, 0, sizeof(ds));
        ds.depthCompare = static_cast<uint8_t>(MTLCompareFunctionAlways);
        ds.stencilEnabled = 1;
        for (DepthStencilKey::Face* f : {&ds.front, &ds.back}) {
            f->compare = static_cast<uint8_t>(MTLCompareFunctionAlways);
            f->fail = f->depthFail = f->pass = static_cast<uint8_t>(MTLStencilOperationReplace);
            f->readMask = 0xff;
            f->writeMask = pass == 0 ? 0xffu : 1u << (pass - 1);
        }
        [encoder setDepthStencilState:ctx.share().pipelines().depthStencil(ds)];
        [encoder setStencilReferenceValue:pass == 0 ? 0u : 0xffu];
        const uint32_t p[4] = {static_cast<uint32_t>(clip.x0), static_cast<uint32_t>(clip.y0),
                               pass == 0 ? 0u : 1u << (pass - 1), 0};
        [encoder setFragmentBytes:p length:sizeof(p) atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    ctx.stream().noteDraw();
}

// Stencil blit through the CPU (nearest filtering only, as GL requires).
void blitStencil(Context& ctx, const RenderTargetDesc::Target& source, uint32_t srcW, uint32_t srcH,
                 const RenderTargetDesc::Target& dst, MTLPixelFormat dstDepthFormat, uint32_t dstSamples, uint32_t dstW,
                 uint32_t dstH, const GLint s[4], const GLint d[4]) {
    const RenderTargetDesc::Target src =
        source.texture.sampleCount > 1 ? resolveDepthStencil(ctx, source, srcW, srcH) : source;
    const Rect clip = writableRect(ctx, normalized(d[0], d[1], d[2], d[3]), dstW, dstH);
    if (clip.empty()) return;
    const bool multisample = dst.texture.sampleCount > 1;
    auto option = [](id<MTLTexture> t) {
        return t.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 ? MTLBlitOptionStencilFromDepthStencil : MTLBlitOptionNone;
    };
    id<MTLDevice> device = Device::get().mtl();
    id<MTLBuffer> in = [device newBufferWithLength:std::max<size_t>(1, static_cast<size_t>(srcW) * srcH)
                                           options:MTLResourceStorageModeShared];
    const bool src3D = src.texture.textureType == MTLTextureType3D;
    [ctx.stream().blit() copyFromTexture:src.texture sourceSlice:src3D ? 0 : src.slice sourceLevel:src.level
                            sourceOrigin:MTLOriginMake(0, 0, src3D ? src.slice : 0) sourceSize:MTLSizeMake(srcW, srcH, 1)
                                toBuffer:in destinationOffset:0 destinationBytesPerRow:srcW
                 destinationBytesPerImage:static_cast<size_t>(srcW) * srcH options:option(src.texture)];
    const int w = clip.x1 - clip.x0, h = clip.y1 - clip.y0;
    id<MTLBuffer> out = [device newBufferWithLength:static_cast<size_t>(w) * h options:MTLResourceStorageModeShared];
    const bool dst3D = dst.texture.textureType == MTLTextureType3D;
    // Keep the destination's existing values where the source is outside its bounds.
    if (!multisample)
        [ctx.stream().blit() copyFromTexture:dst.texture sourceSlice:dst3D ? 0 : dst.slice sourceLevel:dst.level
                                sourceOrigin:MTLOriginMake(clip.x0, clip.y0, dst3D ? dst.slice : 0) sourceSize:MTLSizeMake(w, h, 1)
                                    toBuffer:out destinationOffset:0 destinationBytesPerRow:static_cast<NSUInteger>(w)
                     destinationBytesPerImage:static_cast<size_t>(w) * h options:option(dst.texture)];
    ctx.stream().finish();
    const uint8_t* from = static_cast<const uint8_t*>(in.contents);
    uint8_t* to = static_cast<uint8_t*>(out.contents);
    std::vector<uint16_t> values(multisample ? static_cast<size_t>(w) * h : 0, 0x100);
    const double sx = static_cast<double>(s[2] - s[0]) / (d[2] - d[0]), sy = static_cast<double>(s[3] - s[1]) / (d[3] - d[1]);
    for (int y = clip.y0; y < clip.y1; ++y) {
        for (int x = clip.x0; x < clip.x1; ++x) {
            const int px = static_cast<int>(std::floor(s[0] + (x + 0.5 - d[0]) * sx));
            const int py = static_cast<int>(std::floor(s[1] + (y + 0.5 - d[1]) * sy));
            if (px < 0 || py < 0 || px >= static_cast<int>(srcW) || py >= static_cast<int>(srcH)) continue;
            const size_t at = static_cast<size_t>(y - clip.y0) * w + (x - clip.x0);
            const uint8_t v = from[static_cast<size_t>(py) * srcW + px];
            if (multisample) values[at] = v;
            else to[at] = v;
        }
    }
    if (multisample) {
        writeStencilMultisample(ctx, dst, dstDepthFormat, dstSamples, dstW, dstH, clip, values);
        return;
    }
    [ctx.stream().blit() copyFromBuffer:out sourceOffset:0 sourceBytesPerRow:static_cast<NSUInteger>(w)
                    sourceBytesPerImage:static_cast<size_t>(w) * h sourceSize:MTLSizeMake(w, h, 1) toTexture:dst.texture
                       destinationSlice:dst3D ? 0 : dst.slice destinationLevel:dst.level
                      destinationOrigin:MTLOriginMake(clip.x0, clip.y0, dst3D ? dst.slice : 0) options:option(dst.texture)];
}

NSString* colorShader(const FormatInfo* info, bool ms) {
    if (info && info->isSignedInteger()) return ms ? @"wmgl_blit_int_ms_fs" : @"wmgl_blit_int_fs";
    if (info && info->isInteger()) return ms ? @"wmgl_blit_uint_ms_fs" : @"wmgl_blit_uint_fs";
    return ms ? @"wmgl_blit_float_ms_fs" : @"wmgl_blit_float_fs";
}

}  // namespace

void BlitFramebuffer(Context& ctx, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0,
                     GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter) {
    if (mask & ~(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) return ctx.setError(GL_INVALID_VALUE);
    if (filter != GL_NEAREST && filter != GL_LINEAR) return ctx.setError(GL_INVALID_ENUM);
    if ((mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) && filter == GL_LINEAR)
        return ctx.setError(GL_INVALID_OPERATION);
    if (CheckFramebufferStatus(ctx, GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE ||
        CheckFramebufferStatus(ctx, GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        return ctx.setError(GL_INVALID_FRAMEBUFFER_OPERATION);
    DrawState& draw = ctx.draw();
    RenderTargetDesc dst;
    TargetFormats dstFormats;
    if (!draw.drawTarget(dst, dstFormats)) return;

    // Sources; buffers missing from either framebuffer are silently skipped.
    RenderTargetDesc::Target srcColor, srcDepth, srcStencil;
    const FormatInfo *colorInfo = nullptr, *depthInfo = nullptr, *stencilInfo = nullptr;
    // Each read buffer has its own size (attachments may differ, GL 3.0).
    uint32_t srcW = 0, srcH = 0, srcSamples = 1, depthW = 0, depthH = 0, depthSamples = 1, stencilW = 0, stencilH = 0,
             stencilSamples = 1;
    bool doColor = (mask & GL_COLOR_BUFFER_BIT) && draw.readTarget(GL_COLOR, srcColor, colorInfo, srcW, srcH, srcSamples);
    bool doDepth = (mask & GL_DEPTH_BUFFER_BIT) && dstFormats.depthAttached &&
                   draw.readTarget(GL_DEPTH, srcDepth, depthInfo, depthW, depthH, depthSamples);
    if (doDepth && !doColor) srcSamples = depthSamples;
    bool doStencil = (mask & GL_STENCIL_BUFFER_BIT) && dstFormats.stencilAttached &&
                     draw.readTarget(GL_STENCIL, srcStencil, stencilInfo, stencilW, stencilH, stencilSamples);
    if (doStencil && !doColor && !doDepth) srcSamples = stencilSamples;
    if (doColor) {
        bool anyDst = false;
        for (int i = 0; i < 8; ++i) anyDst |= dst.color[i].texture != nil;
        doColor = anyDst;
    }

    // Validation (GL 3.3 section 4.3.2).
    const bool multisampled = srcSamples > 1 || dst.samples > 1;
    if (multisampled && (std::abs(srcX1 - srcX0) != std::abs(dstX1 - dstX0) || std::abs(srcY1 - srcY0) != std::abs(dstY1 - dstY0)))
        return ctx.setError(GL_INVALID_OPERATION);
    if (srcSamples > 1 && dst.samples > 1 && srcSamples != dst.samples) return ctx.setError(GL_INVALID_OPERATION);
    if (doColor) {
        for (int i = 0; i < 8; ++i) {
            if (!dst.color[i].texture) continue;
            const bool srcInt = colorInfo && colorInfo->isInteger();
            if (srcInt != dstFormats.integer[i]) return ctx.setError(GL_INVALID_OPERATION);
            if (srcInt && filter == GL_LINEAR) return ctx.setError(GL_INVALID_OPERATION);
        }
    }
    if (doDepth && srcDepth.texture.pixelFormat != dstFormats.depth) return ctx.setError(GL_INVALID_OPERATION);
    if (doStencil && srcStencil.texture.pixelFormat != dstFormats.stencil) return ctx.setError(GL_INVALID_OPERATION);
    // The GL formats must match too (DEPTH24_STENCIL8 and DEPTH32F_STENCIL8
    // share Metal's Depth32Float_Stencil8).
    if (doDepth && depthInfo && drawDepthStencilFormat(ctx, false) && depthInfo != drawDepthStencilFormat(ctx, false))
        return ctx.setError(GL_INVALID_OPERATION);
    if (doStencil && stencilInfo && drawDepthStencilFormat(ctx, true) && stencilInfo != drawDepthStencilFormat(ctx, true))
        return ctx.setError(GL_INVALID_OPERATION);
    if (doColor && colorInfo && colorInfo->isInteger()) {
        // Integer colors copy to integer buffers of the same signedness.
        if (const Framebuffer* fb = ctx.state.drawFramebuffer.get())
            for (int i = 0; i < Framebuffer::kMaxColor; ++i)
                if (fb->drawBuffers[i] >= GL_COLOR_ATTACHMENT0 && fb->drawBuffers[i] < GL_COLOR_ATTACHMENT0 + Framebuffer::kMaxColor)
                    if (const FormatInfo* f = attachedFormat(fb->color[fb->drawBuffers[i] - GL_COLOR_ATTACHMENT0]);
                        f && f->isSignedInteger() != colorInfo->isSignedInteger())
                        return ctx.setError(GL_INVALID_OPERATION);
    }
    if (ctx.isEs()) {
        // OpenGL ES 3.2 section 16.2.1: no multisampled draw framebuffer, and
        // a multisampled read framebuffer resolves into its own format.
        if (dst.samples > 1) return ctx.setError(GL_INVALID_OPERATION);
        // ... and a multisampled read framebuffer blits to the same rectangle.
        if (srcSamples > 1 && (srcX0 != dstX0 || srcY0 != dstY0 || srcX1 != dstX1 || srcY1 != dstY1))
            return ctx.setError(GL_INVALID_OPERATION);
        if (srcSamples > 1 && doColor && colorInfo)
            if (const Framebuffer* fb = ctx.state.drawFramebuffer.get())
                for (int i = 0; i < Framebuffer::kMaxColor; ++i)
                    if (fb->drawBuffers[i] >= GL_COLOR_ATTACHMENT0 && fb->drawBuffers[i] < GL_COLOR_ATTACHMENT0 + Framebuffer::kMaxColor)
                        if (const FormatInfo* f = attachedFormat(fb->color[fb->drawBuffers[i] - GL_COLOR_ATTACHMENT0]);
                            f && f->internalFormat != colorInfo->internalFormat)
                            return ctx.setError(GL_INVALID_OPERATION);
        // Section 16.2.1: the source and destination buffers may not be the
        // same image (desktop GL leaves overlapping blits undefined).
        // A view (such as the linear view of sRGB storage) is its texture's
        // image at the view's first level and slice.
        auto same = [](const RenderTargetDesc::Target& a, const RenderTargetDesc::Target& b) {
            if (!a.texture || !b.texture) return false;
            auto storage = [](id<MTLTexture> t) { return t.parentTexture ? t.parentTexture : t; };
            auto level = [](const RenderTargetDesc::Target& t) {
                return t.level + (t.texture.parentTexture ? t.texture.parentRelativeLevel : 0);
            };
            auto slice = [](const RenderTargetDesc::Target& t) {
                return t.slice + (t.texture.parentTexture ? t.texture.parentRelativeSlice : 0);
            };
            return storage(a.texture) == storage(b.texture) && level(a) == level(b) && slice(a) == slice(b);
        };
        if (doColor)
            for (int i = 0; i < 8; ++i)
                if (same(srcColor, dst.color[i])) return ctx.setError(GL_INVALID_OPERATION);
        if ((doDepth && same(srcDepth, dst.depth)) || (doStencil && same(srcStencil, dst.stencil)))
            return ctx.setError(GL_INVALID_OPERATION);
    }
    if (!doColor && !doDepth && !doStencil) return;

    const GLint srcRect[4] = {srcX0, srcY0, srcX1, srcY1};
    const GLint dstRect[4] = {dstX0, dstY0, dstX1, dstY1};
    if (srcX0 == srcX1 || srcY0 == srcY1 || dstX0 == dstX1 || dstY0 == dstY1) return;
    const Rect scissor = writableRect(ctx, normalized(dstX0, dstY0, dstX1, dstY1), dst.width, dst.height);
    if (scissor.empty()) return;
    auto paramsFor = [&](uint32_t width, uint32_t height) {
        return BlitParams{
            {static_cast<float>(dstX0), static_cast<float>(dstY0), static_cast<float>(dstX1), static_cast<float>(dstY1)},
            {static_cast<float>(srcX0), static_cast<float>(srcY0), static_cast<float>(srcX1), static_cast<float>(srcY1)},
            {static_cast<float>(width), static_cast<float>(height), filter == GL_LINEAR ? 1.0f : 0.0f, 0}};
    };
    const BlitParams params = paramsFor(srcW, srcH);

    if (doColor) {
        id<MTLTexture> view = nil;
        for (int i = 0; i < 8; ++i) {
            if (!dst.color[i].texture) continue;
            // An RGB destination keeps its stored alpha of one; a raw copy would overwrite it.
            const bool keepAlpha = dstFormats.alphaIsOne[i] && !(colorInfo && colorInfo->storedAlphaIsOne);
            if (!keepAlpha && tryCopy(ctx, srcColor, srcW, srcH, dst.color[i], dst.width, dst.height, srcRect, dstRect))
                continue;
            if (!view) view = sourceView(ctx, srcColor, srcW, srcH);
            RenderTargetDesc target;
            target.color[0] = dst.color[i];
            target.width = dst.width;
            target.height = dst.height;
            target.samples = dst.samples;
            id<MTLRenderPipelineState> pipeline =
                blitPipeline(colorShader(colorInfo, view.sampleCount > 1), dstFormats.color[i], MTLPixelFormatInvalid,
                             MTLPixelFormatInvalid, dst.samples, !dstFormats.alphaIsOne[i]);
            if (pipeline) drawBlit(ctx, target, pipeline, view, params, scissor, false, filter == GL_LINEAR);
        }
    }
    if (doDepth) {
        // Copying a combined depth/stencil texture would also copy stencil.
        const bool combined = dstFormats.depth == MTLPixelFormatDepth32Float_Stencil8;
        if (combined || !tryCopy(ctx, srcDepth, depthW, depthH, dst.depth, dst.width, dst.height, srcRect, dstRect)) {
            id<MTLTexture> view = sourceView(ctx, srcDepth, depthW, depthH);
            RenderTargetDesc target;
            target.depth = dst.depth;
            if (combined) target.stencil = dst.depth;  // Metal binds combined formats at both points
            target.width = dst.width;
            target.height = dst.height;
            target.samples = dst.samples;
            id<MTLRenderPipelineState> pipeline = blitPipeline(view.sampleCount > 1 ? @"wmgl_blit_depth_ms_fs" : @"wmgl_blit_depth_fs",
                                                               MTLPixelFormatInvalid, dstFormats.depth,
                                                               combined ? dstFormats.depth : MTLPixelFormatInvalid, dst.samples);
            if (pipeline) drawBlit(ctx, target, pipeline, view, paramsFor(depthW, depthH), scissor, true, false);
        }
    }
    // Stencil-only images of one format and sample count copy directly, every
    // sample included (KHR-GL46.texture_stencil8.multisample); otherwise the
    // CPU path, which resolves a multisample source.
    if (doStencil && !(srcStencil.texture.pixelFormat == MTLPixelFormatStencil8 &&
                       tryCopy(ctx, srcStencil, stencilW, stencilH, dst.stencil, dst.width, dst.height, srcRect, dstRect)))
        blitStencil(ctx, srcStencil, stencilW, stencilH, dst.stencil, dstFormats.depth, dst.samples, dst.width, dst.height,
                    srcRect, dstRect);
    // Draws after the blit start a fresh pass with their own state.
    ctx.stream().endEncoder();
    ctx.markDirty(kDirtyAll);
}

namespace {

MTLPixelFormat withoutSrgb(MTLPixelFormat f) {
    switch (f) {
    case MTLPixelFormatRGBA8Unorm_sRGB: return MTLPixelFormatRGBA8Unorm;
    case MTLPixelFormatBGRA8Unorm_sRGB: return MTLPixelFormatBGRA8Unorm;
    case MTLPixelFormatR8Unorm_sRGB: return MTLPixelFormatR8Unorm;
    case MTLPixelFormatRG8Unorm_sRGB: return MTLPixelFormatRG8Unorm;
    default: return f;
    }
}

// The texture with its sRGB encoding read and written as stored values (a
// view of the same format without sRGB, which needs no format-view usage).
id<MTLTexture> storedValues(id<MTLTexture> t) {
    const MTLPixelFormat f = withoutSrgb(t.pixelFormat);
    return f == t.pixelFormat ? t : [t newTextureViewWithPixelFormat:f];
}

bool plainColor(const FormatInfo& info) {
    const GLenum b = info.baseFormat;
    if (b != GL_RED && b != GL_RG && b != GL_RGB && b != GL_RGBA) return false;
    if (info.isInteger() || info.isDepthOrStencil() || info.compressed) return false;
    // RGB stored as RGBA reads alpha one, which is also the stored alpha.
    for (int c = 0; c < 4; ++c)
        if (info.swizzle[c] != static_cast<Swizzle>(c) && !(c == 3 && info.swizzle[c] == SwzOne && info.storedAlphaIsOne))
            return false;
    return true;
}

bool flat2D(id<MTLTexture> t) {
    const MTLTextureType type = t.textureType;
    return t.sampleCount == 1 && type != MTLTextureType1D && type != MTLTextureType1DArray;
}

}  // namespace

bool copyRegionGpu(Context& ctx, const RenderTargetDesc::Target& src, const FormatInfo& srcInfo,
                     const RenderTargetDesc::Target& dst, const FormatInfo& dstInfo, int sx, int sy, int dx, int dy, int w,
                     int h) {
    if (!flat2D(src.texture) || !flat2D(dst.texture) || !(dst.texture.usage & MTLTextureUsageRenderTarget)) return false;
    // Depth into a depth-only image of another format, drawn as depth (a
    // destination with stencil would need its stencil copied too).
    const bool depth = srcInfo.isDepth() && dstInfo.isDepth() && !dstInfo.isStencil();
    if (!depth && (!plainColor(srcInfo) || !plainColor(dstInfo))) return false;
    id<MTLTexture> from = storedValues(src.texture), to = storedValues(dst.texture);
    if (!from || !to) return false;
    const bool src3D = from.textureType == MTLTextureType3D, dst3D = to.textureType == MTLTextureType3D;
    // An RGB destination's stored alpha is one; a raw copy from a format with alpha would change it.
    const bool keepAlpha = dstInfo.storedAlphaIsOne && !srcInfo.storedAlphaIsOne;
    if (from.pixelFormat == to.pixelFormat && !keepAlpha && !depth) {
        [ctx.stream().blit() copyFromTexture:from sourceSlice:src3D ? 0 : src.slice sourceLevel:src.level
                                sourceOrigin:MTLOriginMake(sx, sy, src3D ? src.slice : 0) sourceSize:MTLSizeMake(w, h, 1)
                                   toTexture:to destinationSlice:dst3D ? 0 : dst.slice destinationLevel:dst.level
                           destinationOrigin:MTLOriginMake(dx, dy, dst3D ? dst.slice : 0)];
        return true;
    }
    const uint32_t srcW = std::max<uint32_t>(1, static_cast<uint32_t>(from.width) >> src.level);
    const uint32_t srcH = std::max<uint32_t>(1, static_cast<uint32_t>(from.height) >> src.level);
    id<MTLRenderPipelineState> pipeline =
        depth ? blitPipeline(@"wmgl_blit_depth_fs", MTLPixelFormatInvalid, to.pixelFormat, MTLPixelFormatInvalid, 1)
              : blitPipeline(@"wmgl_blit_float_fs", to.pixelFormat, MTLPixelFormatInvalid, MTLPixelFormatInvalid, 1,
                             !dstInfo.storedAlphaIsOne);
    if (!pipeline) return false;
    id<MTLTexture> view = sourceView(ctx, {from, src.level, src.slice}, srcW, srcH);
    RenderTargetDesc target;
    (depth ? target.depth : target.color[0]) = {to, dst.level, dst.slice};
    target.width = std::max<uint32_t>(1, static_cast<uint32_t>(to.width) >> dst.level);
    target.height = std::max<uint32_t>(1, static_cast<uint32_t>(to.height) >> dst.level);
    const BlitParams params = {
        {static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dx + w), static_cast<float>(dy + h)},
        {static_cast<float>(sx), static_cast<float>(sy), static_cast<float>(sx + w), static_cast<float>(sy + h)},
        {static_cast<float>(srcW), static_cast<float>(srcH), 0, 0}};
    drawBlit(ctx, target, pipeline, view, params, {dx, dy, dx + w, dy + h}, depth, false);
    // Draws after the copy start a fresh pass with their own state.
    ctx.stream().endEncoder();
    ctx.markDirty(kDirtyAll);
    return true;
}

// glGenerateMipmap of a depth (or depth/stencil) texture: each level is
// the nearest downsample of the one above, drawn as depth. Metal's mipmap
// blit takes colour formats only.
void generateDepthMipmaps(Context& ctx, Texture& tex, int base, int last) {
    id<MTLTexture> mtl = tex.mtl;
    const MTLPixelFormat format = mtl.pixelFormat;
    const bool combined = format == MTLPixelFormatDepth32Float_Stencil8;
    const NSUInteger slices = mtl.textureType == MTLTextureTypeCube        ? 6
                              : mtl.textureType == MTLTextureTypeCubeArray ? mtl.arrayLength * 6
                                                                           : std::max<NSUInteger>(1, mtl.arrayLength);
    id<MTLRenderPipelineState> pipeline =
        blitPipeline(@"wmgl_blit_depth_fs", MTLPixelFormatInvalid, format, combined ? format : MTLPixelFormatInvalid, 1);
    if (!pipeline) return;
    for (int level = base + 1; level <= last; ++level) {
        const uint32_t w = std::max<uint32_t>(1, static_cast<uint32_t>(mtl.width) >> level);
        const uint32_t h = std::max<uint32_t>(1, static_cast<uint32_t>(mtl.height) >> level);
        const uint32_t pw = std::max<uint32_t>(1, static_cast<uint32_t>(mtl.width) >> (level - 1));
        const uint32_t ph = std::max<uint32_t>(1, static_cast<uint32_t>(mtl.height) >> (level - 1));
        for (NSUInteger slice = 0; slice < slices; ++slice) {
            id<MTLTexture> source = [mtl newTextureViewWithPixelFormat:format
                                                           textureType:MTLTextureType2D
                                                                levels:NSMakeRange(static_cast<NSUInteger>(level - 1), 1)
                                                                slices:NSMakeRange(slice, 1)];
            RenderTargetDesc target;
            target.depth = {mtl, static_cast<uint32_t>(level), static_cast<uint32_t>(slice)};
            if (combined) target.stencil = target.depth;
            target.width = w;
            target.height = h;
            BlitParams params = {{0, 0, static_cast<float>(w), static_cast<float>(h)},
                                 {0, 0, static_cast<float>(pw), static_cast<float>(ph)},
                                 {static_cast<float>(pw), static_cast<float>(ph), 0, 0}};
            drawBlit(ctx, target, pipeline, source, params, Rect{0, 0, static_cast<int>(w), static_cast<int>(h)}, true, false);
        }
    }
    ctx.stream().endEncoder();
    ctx.markDirty(kDirtyAll);
}

}  // namespace wmgl::api
