#include "core/texture_storage.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "core/context.h"
#include "core/draw.h"
#include "core/limits.h"
#include "core/metal/device.h"
#include "core/metal/enums.h"
#include "core/metal/pipeline_cache.h"
#include "core/metal/pixel_format.h"
#include "core/util/log.h"

namespace wmgl {

GLsizei fullMipCount(GLsizei width, GLsizei height, GLsizei depth) {
    GLsizei size = std::max({width, height, depth, 1});
    GLsizei levels = 1;
    while (size > 1) {
        size >>= 1;
        ++levels;
    }
    return levels;
}

MTLTextureType mtlTextureType(GLenum target, GLsizei samples) {
    switch (target) {
    case GL_TEXTURE_1D: return MTLTextureType2D;  // Metal 1D textures cannot be mipmapped or rendered to
    case GL_TEXTURE_1D_ARRAY: return MTLTextureType2DArray;
    case GL_TEXTURE_2D: case GL_TEXTURE_RECTANGLE: return MTLTextureType2D;
    case GL_TEXTURE_3D: return MTLTextureType3D;
    case GL_TEXTURE_2D_ARRAY: return MTLTextureType2DArray;
    case GL_TEXTURE_CUBE_MAP: return MTLTextureTypeCube;
    case GL_TEXTURE_CUBE_MAP_ARRAY: return MTLTextureTypeCubeArray;
    case GL_TEXTURE_2D_MULTISAMPLE: return samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
    case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: return samples > 1 ? MTLTextureType2DMultisampleArray : MTLTextureType2DArray;
    case GL_TEXTURE_BUFFER: return MTLTextureTypeTextureBuffer;
    default: return MTLTextureType2D;
    }
}

namespace {

// Size of `level` given level 0 dimensions; arrays keep their layer count.
void levelSize(GLenum target, GLsizei w0, GLsizei h0, GLsizei d0, int level, GLsizei& w, GLsizei& h, GLsizei& d) {
    w = std::max(1, w0 >> level);
    const bool oneDArray = target == GL_TEXTURE_1D_ARRAY;
    const bool arrays = target == GL_TEXTURE_2D_ARRAY || target == GL_TEXTURE_CUBE_MAP_ARRAY ||
                        target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
    h = oneDArray ? h0 : std::max(1, h0 >> level);
    d = (arrays || oneDArray || target != GL_TEXTURE_3D) ? d0 : std::max(1, d0 >> level);
}

// Level-0-equivalent dimensions of a texture from its lowest defined level.
bool baseDimensions(const Texture& tex, GLsizei& w0, GLsizei& h0, GLsizei& d0, const FormatInfo*& format,
                    GLsizei& samples) {
    // The chain of TEXTURE_BASE_LEVEL's image when it has one: levels left
    // from earlier definitions that do not fit it are never sampled
    // (a program can redefine a cube map's levels over another size's).
    auto fromLevel = [&](int level) {
        const TextureImage* defined = nullptr;
        for (int face = 0; face < tex.faceCount() && !defined; ++face)
            if (tex.image(face, level).defined()) defined = &tex.image(face, level);
        if (!defined) return false;
        const TextureImage& image = *defined;
        format = image.format;
        samples = image.samples;
        w0 = image.width << level;
        const bool oneDArray = tex.target == GL_TEXTURE_1D_ARRAY;
        h0 = oneDArray || tex.target == GL_TEXTURE_1D ? image.height : image.height << level;
        d0 = tex.target == GL_TEXTURE_3D ? image.depth << level : image.depth;
        return true;
    };
    if (!tex.immutable && tex.baseLevel > 0 && tex.baseLevel < Texture::kMaxLevels && fromLevel(tex.baseLevel)) return true;
    if (!tex.immutable && tex.baseLevel == 0 && fromLevel(0)) return true;
    for (int level = 0; level < Texture::kMaxLevels; ++level) {
        // Any cube face may be specified first.
        const TextureImage* defined = nullptr;
        for (int face = 0; face < tex.faceCount() && !defined; ++face)
            if (tex.image(face, level).defined()) defined = &tex.image(face, level);
        if (!defined) continue;
        const TextureImage& image = *defined;
        format = image.format;
        samples = image.samples;
        w0 = image.width << level;
        const bool oneDArray = tex.target == GL_TEXTURE_1D_ARRAY;
        h0 = oneDArray || tex.target == GL_TEXTURE_1D ? image.height : image.height << level;
        d0 = tex.target == GL_TEXTURE_3D ? image.depth << level : image.depth;
        return true;
    }
    return false;
}

// Clears new storage to `color` in a command buffer of its own, committed
// ahead of the stream's: nothing else references the new texture yet, and
// it keeps any render pass the stream is encoding open.
// - RGB formats are stored with an alpha channel that must read as one
//   (copies and blits move it as stored): (0, 0, 0, 1).
// - Images defined without data start zeroed, as on Windows, where new
//   allocations are zero-filled. Metal leaves private storage undefined, and
//   games blend into textures they never cleared: Foundation's map generator
//   blurs heightmaps with SRC_ALPHA/ONE_MINUS_SRC_ALPHA into fresh R32F
//   textures, and dst * 0 is NaN when the old memory held NaN (issue #12).
void initStorage(Context& ctx, id<MTLTexture> texture, MTLClearColor color) {
    id<MTLCommandBuffer> commandBuffer = [ctx.stream().queue().mtl() commandBuffer];
    const bool volume = texture.textureType == MTLTextureType3D;
    const NSUInteger faces = texture.textureType == MTLTextureTypeCube        ? 6
                             : texture.textureType == MTLTextureTypeCubeArray ? 6 * texture.arrayLength
                                                                               : texture.arrayLength;
    for (NSUInteger level = 0; level < texture.mipmapLevelCount; ++level) {
        // Array and cube layers are cleared by one layered pass per level;
        // 3D depth planes one pass each.
        const NSUInteger count = volume ? std::max<NSUInteger>(1, texture.depth >> level) : 1;
        for (NSUInteger i = 0; i < count; ++i) {
            MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
            pass.colorAttachments[0].texture = texture;
            pass.colorAttachments[0].level = level;
            if (volume) pass.colorAttachments[0].depthPlane = i;
            else if (faces > 1) pass.renderTargetArrayLength = faces;
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].clearColor = color;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            [[commandBuffer renderCommandEncoderWithDescriptor:pass] endEncoding];
        }
    }
    [commandBuffer commit];
}

}  // namespace

namespace {

MTLPixelFormat withoutSrgb(MTLPixelFormat f);

// A texture view's storage (GL 4.3): a Metal view of its root texture's
// levels and layers in the view's format, remade when the root's storage
// changes. Writes through an image unit need write usage on the root.
bool ensureViewStorage(Context& ctx, Texture& tex) {
    Texture& root = *tex.viewOf;
    if (tex.shaderWrite && !root.shaderWrite) {
        root.shaderWrite = true;
        root.storageDirty = true;
    }
    // A view in another format (beyond sRGB) needs format views on the root.
    const TextureImage& viewBase = tex.image(0, 0);
    const TextureImage& rootBase = root.image(0, root.immutable ? 0 : root.baseLevel);
    if (viewBase.format && rootBase.format && !root.formatViews &&
        withoutSrgb(mtlPixelFormat(viewBase.format->storage)) != withoutSrgb(mtlPixelFormat(rootBase.format->storage))) {
        root.formatViews = true;
        root.storageDirty = true;
    }
    if (!ensureTextureStorage(ctx, root)) return false;
    if (tex.mtl && tex.viewSource == root.mtl) {
        tex.storageDirty = false;
        return true;
    }
    const TextureImage& base = tex.image(0, 0);
    if (!base.format) return false;
    const MTLPixelFormat pixelFormat = mtlPixelFormat(base.format->storage);
    if (pixelFormat == MTLPixelFormatInvalid) return false;
    const bool threeD = tex.target == GL_TEXTURE_3D;
    id<MTLTexture> view = [root.mtl newTextureViewWithPixelFormat:pixelFormat
                                                      textureType:mtlTextureType(tex.target, base.samples)
                                                           levels:NSMakeRange(tex.viewMinLevel, tex.viewNumLevels)
                                                           slices:NSMakeRange(threeD ? 0 : tex.viewMinLayer, threeD ? 1 : tex.viewNumLayers)];
    if (!view) {
        WMGL_ERR(Texture, "texture view creation failed (%s)", storageName(base.format->storage));
        return false;
    }
    tex.mtl = view;
    tex.viewSource = root.mtl;
    tex.storage = base.format->storage;
    tex.mtlLevels = static_cast<GLsizei>(tex.viewNumLevels);
    tex.mtlWidth = base.width;
    tex.mtlHeight = base.height;
    tex.mtlDepth = base.depth;
    tex.mtlSamples = base.samples;
    tex.storageDirty = false;
    tex.views.clear();
    return true;
}

}  // namespace

namespace {

// Writes the images' pending contents (TextureImage::pending) that fit the
// storage now; the others stay pending.
void writePendingImages(Context& ctx, Texture& tex) {
    if (tex.target != GL_TEXTURE_2D && tex.target != GL_TEXTURE_CUBE_MAP) return;
    for (int face = 0; face < (tex.target == GL_TEXTURE_CUBE_MAP ? 6 : 1); ++face)
        for (int level = 0; level < tex.mtlLevels && level < Texture::kMaxLevels; ++level) {
            TextureImage& image = tex.image(face, level);
            if (image.pending.empty()) continue;
            const GLsizei w = std::max(1, tex.mtlWidth >> level), h = std::max(1, tex.mtlHeight >> level);
            if (!image.format || image.format->storage != tex.storage || image.width != w || image.height != h) continue;
            const size_t rowBytes = static_cast<size_t>(w) * storageBytes(tex.storage);
            if (image.pending.size() == rowBytes * static_cast<size_t>(h)) {
                id<MTLBuffer> staging = [Device::get().mtl() newBufferWithBytes:image.pending.data() length:image.pending.size()
                                                                         options:MTLResourceStorageModeShared];
                if (staging)
                    [ctx.stream().blit() copyFromBuffer:staging sourceOffset:0 sourceBytesPerRow:rowBytes
                                    sourceBytesPerImage:image.pending.size() sourceSize:MTLSizeMake(w, h, 1)
                                              toTexture:tex.mtl destinationSlice:static_cast<NSUInteger>(face)
                                       destinationLevel:static_cast<NSUInteger>(level) destinationOrigin:MTLOriginMake(0, 0, 0)];
            }
            image.pending.clear();
        }
}

}  // namespace

bool ensureTextureStorage(Context& ctx, Texture& tex) {
    if (tex.viewOf) return ensureViewStorage(ctx, tex);
    if (tex.mtl && !tex.storageDirty) return true;
    GLsizei w0 = 0, h0 = 0, d0 = 0, samples = 0;
    const FormatInfo* format = nullptr;
    if (!baseDimensions(tex, w0, h0, d0, format, samples) || !format) return false;
    if (w0 <= 0 || h0 <= 0 || d0 <= 0) return false;  // zero-sized images have no storage
    const MTLPixelFormat pixelFormat = mtlPixelFormat(format->storage);
    if (pixelFormat == MTLPixelFormatInvalid) return false;
    GLsizei levels = tex.immutable ? tex.immutableLevels
                     : (tex.target == GL_TEXTURE_RECTANGLE || samples > 1) ? 1
                                                                           : fullMipCount(w0, tex.target == GL_TEXTURE_1D || tex.target == GL_TEXTURE_1D_ARRAY ? 1 : h0,
                                                                                          tex.target == GL_TEXTURE_3D ? d0 : 1);
    levels = std::min<GLsizei>(levels, Texture::kMaxLevels);
    // Image units (GL 4.2) write textures from shaders; 32-bit integer
    // formats also take texture atomics.
    const bool atomics = pixelFormat == MTLPixelFormatR32Uint || pixelFormat == MTLPixelFormatR32Sint;
    const MTLTextureUsage writeUsage =
        tex.shaderWrite && !format->compressed ? MTLTextureUsageShaderWrite | (atomics ? MTLTextureUsageShaderAtomic : 0) : 0;
    // Multisample storage cannot be copied into new storage later: it has
    // format views from the start (glClearTexImage and glCopyImageSubData
    // view it in other formats). The packed float formats keep them too:
    // they have several encodings of one value, and compressed storage
    // gives back another encoding than was stored (glCopyImageSubData and
    // readback must see the bits uploaded; KHR-GL46.copy_image).
    const MTLTextureUsage viewUsage =
        tex.formatViews || samples > 1 || format->storage == Storage::RGB9E5Float || format->storage == Storage::RG11B10Float
            ? MTLTextureUsagePixelFormatView
            : 0;
    // Reuse the existing storage if it already has this shape.
    if (tex.mtl && tex.storage == format->storage && tex.mtlWidth == w0 && tex.mtlHeight == h0 && tex.mtlDepth == d0 &&
        tex.mtlSamples == samples && tex.mtlLevels >= levels && (tex.mtl.usage & writeUsage) == writeUsage &&
        (tex.mtl.usage & viewUsage) == viewUsage) {
        tex.storageDirty = false;
        writePendingImages(ctx, tex);
        return true;
    }
    MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
    desc.textureType = mtlTextureType(tex.target, samples);
    desc.pixelFormat = pixelFormat;
    desc.width = static_cast<NSUInteger>(w0);
    const bool cube = tex.target == GL_TEXTURE_CUBE_MAP || tex.target == GL_TEXTURE_CUBE_MAP_ARRAY;
    desc.height = static_cast<NSUInteger>(tex.target == GL_TEXTURE_1D || tex.target == GL_TEXTURE_1D_ARRAY ? 1 : h0);
    desc.depth = static_cast<NSUInteger>(tex.target == GL_TEXTURE_3D ? d0 : 1);
    if (tex.target == GL_TEXTURE_1D_ARRAY) desc.arrayLength = static_cast<NSUInteger>(h0);
    else if (tex.target == GL_TEXTURE_2D_ARRAY || tex.target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
        desc.arrayLength = static_cast<NSUInteger>(d0);
    else if (tex.target == GL_TEXTURE_CUBE_MAP_ARRAY) desc.arrayLength = static_cast<NSUInteger>(d0 / 6);
    desc.mipmapLevelCount = static_cast<NSUInteger>(levels);
    desc.sampleCount = static_cast<NSUInteger>(std::max(1, samples));
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead | viewUsage | writeUsage;
    // Immutable storage can have views of another format, which may be
    // renderable when this one is not (GL_RGB9_E5 as GL_R32UI).
    if ((format->colorRenderable || format->isDepthOrStencil() || format->storedAlphaIsOne || tex.immutable) &&
        !format->compressed)
        desc.usage |= MTLTextureUsageRenderTarget;
    (void)cube;
    id<MTLTexture> texture = [Device::get().mtl() newTextureWithDescriptor:desc];
    if (!texture) {
        WMGL_ERR(Texture, "texture allocation failed (%dx%dx%d %s)", w0, h0, d0, storageName(format->storage));
        return false;
    }
    if (format->storedAlphaIsOne) initStorage(ctx, texture, MTLClearColorMake(0, 0, 0, 1));
    else if (tex.zeroStorage && format->colorRenderable && !format->compressed && !format->isDepthOrStencil())
        initStorage(ctx, texture, MTLClearColorMake(0, 0, 0, 0));
    tex.zeroStorage = false;
    // Keep existing contents of levels that still match.
    if (tex.mtl && tex.storage == format->storage && tex.mtl.textureType == texture.textureType &&
        tex.mtl.sampleCount == texture.sampleCount) {
        id<MTLBlitCommandEncoder> blit = ctx.stream().blit();
        // Slices are faces for cube maps and cube map arrays.
        auto sliceCount = [](id<MTLTexture> t) {
            return t.textureType == MTLTextureTypeCube ? NSUInteger{6}
                   : t.textureType == MTLTextureTypeCubeArray ? t.arrayLength * 6
                                                              : t.arrayLength;
        };
        const NSUInteger slices = sliceCount(tex.mtl);
        for (NSUInteger level = 0; level < std::min<NSUInteger>(tex.mtl.mipmapLevelCount, texture.mipmapLevelCount); ++level) {
            const NSUInteger ow = std::max<NSUInteger>(1, tex.mtl.width >> level), oh = std::max<NSUInteger>(1, tex.mtl.height >> level);
            const NSUInteger nw = std::max<NSUInteger>(1, texture.width >> level), nh = std::max<NSUInteger>(1, texture.height >> level);
            const NSUInteger od = tex.mtl.textureType == MTLTextureType3D ? std::max<NSUInteger>(1, tex.mtl.depth >> level) : 1;
            const NSUInteger nd = texture.textureType == MTLTextureType3D ? std::max<NSUInteger>(1, texture.depth >> level) : 1;
            if (ow != nw || oh != nh || od != nd) continue;
            const NSUInteger newSlices = sliceCount(texture);
            for (NSUInteger slice = 0; slice < std::min(slices, newSlices); ++slice)
                [blit copyFromTexture:tex.mtl sourceSlice:slice sourceLevel:level sourceOrigin:MTLOriginMake(0, 0, 0)
                           sourceSize:MTLSizeMake(ow, oh, od) toTexture:texture destinationSlice:slice
                     destinationLevel:level destinationOrigin:MTLOriginMake(0, 0, 0)];
        }
    }
    tex.mtl = texture;
    tex.storage = format->storage;
    tex.mtlLevels = levels;
    tex.mtlWidth = w0;
    tex.mtlHeight = h0;
    tex.mtlDepth = d0;
    tex.mtlSamples = samples;
    tex.storageDirty = false;
    tex.views.clear();
    if (!tex.label.empty()) texture.label = [NSString stringWithUTF8String:tex.label.c_str()];
    writePendingImages(ctx, tex);
    return true;
}

void storageLevelSize(const Texture& tex, int level, GLsizei& w, GLsizei& h, GLsizei& d) {
    w = std::max(1, tex.mtlWidth >> level);
    const bool oneD = tex.target == GL_TEXTURE_1D || tex.target == GL_TEXTURE_1D_ARRAY;
    h = oneD ? tex.mtlHeight : std::max(1, tex.mtlHeight >> level);
    d = tex.target == GL_TEXTURE_3D ? std::max(1, tex.mtlDepth >> level) : tex.mtlDepth;
    if (tex.target == GL_TEXTURE_1D) h = 1;
}

// True if image (face, level) has storage in tex.mtl that matches its size.
bool imageHasStorage(Context& ctx, Texture& tex, int face, int level) {
    (void)face;
    if (!ensureTextureStorage(ctx, tex) || level >= tex.mtlLevels) return false;
    const TextureImage& image = tex.image(face, level);
    GLsizei w, h, d;
    storageLevelSize(tex, level, w, h, d);
    if (tex.target == GL_TEXTURE_CUBE_MAP) d = 1;
    return image.width == w && image.height == h && image.depth == d && image.format &&
           image.format->storage == tex.storage;
}

namespace {

// GL_RGB32F/I/UI buffer textures: an RGBA32 copy of the texels (alpha 1),
// expanded on the GPU before a draw when the buffer changed. While a render
// or compute encoder is open the previous copy is used.
id<MTLTexture> rgb32BufferView(Context& ctx, Texture& tex, Buffer* buffer) {
    static id<MTLComputePipelineState> pipeline = [] {
        NSError* error = nil;
        id<MTLFunction> function = [Device::get().utilityLibrary() newFunctionWithName:@"wmgl_expand_rgb32"];
        id<MTLComputePipelineState> state = function ? [Device::get().mtl() newComputePipelineStateWithFunction:function error:&error] : nil;
        if (!state) WMGL_ERR(Pipeline, "rgb32 expansion pipeline failed: %s", error.localizedDescription.UTF8String);
        return state;
    }();
    if (!pipeline) return nil;
    const size_t offset = static_cast<size_t>(tex.bufferOffset);
    const size_t available = buffer->mtl.length > offset ? buffer->mtl.length - offset : 0;
    const size_t requested = tex.bufferSize < 0 ? static_cast<size_t>(buffer->size) - offset : static_cast<size_t>(tex.bufferSize);
    const NSUInteger count = static_cast<NSUInteger>(std::min<size_t>(std::min(requested, available) / 12, 1u << 27));
    if (count == 0) return nil;
    const MTLPixelFormat pixelFormat = tex.bufferFormat == GL_RGB32F ? MTLPixelFormatRGBA32Float
                                       : tex.bufferFormat == GL_RGB32I ? MTLPixelFormatRGBA32Sint
                                                                       : MTLPixelFormatRGBA32Uint;
    const NSUInteger rowBytes = count * 16;  // RGBA32: 16-byte texels meet the linear texture alignment
    const bool current = tex.rgb32Shadow && tex.rgb32Shadow.length >= rowBytes && tex.rgb32ShadowSource == buffer->mtl &&
                         tex.rgb32ShadowOffset == tex.bufferOffset && tex.rgb32ShadowVersion == buffer->version + 1;
    const uint64_t key = reinterpret_cast<uint64_t>((__bridge void*)buffer->mtl);
    id<MTLTexture> view = nil;
    for (const Texture::View& v : tex.views)
        if (v.key == key) view = v.view;
    if (current && view && view.width == count) return view;
    if (ctx.stream().encoding()) return view;  // inside a draw or dispatch: the last copy
    if (!tex.rgb32Shadow || tex.rgb32Shadow.length < rowBytes) {
        tex.rgb32Shadow = [Device::get().mtl() newBufferWithLength:rowBytes options:MTLResourceStorageModePrivate];
        view = nil;
    }
    if (!tex.rgb32Shadow) return nil;
    const uint32_t params[2] = {static_cast<uint32_t>(count), tex.bufferFormat == GL_RGB32F ? 0x3f800000u : 1u};
    id<MTLComputeCommandEncoder> encoder = ctx.stream().compute();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer->mtl offset:offset atIndex:0];
    [encoder setBuffer:tex.rgb32Shadow offset:0 atIndex:1];
    [encoder setBytes:params length:sizeof(params) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    ctx.stream().endEncoder();
    buffer->lastUseSerial = ctx.stream().serial();
    tex.rgb32ShadowSource = buffer->mtl;
    tex.rgb32ShadowOffset = tex.bufferOffset;
    tex.rgb32ShadowVersion = buffer->version + 1;
    if (view && view.width == count) return view;
    tex.views.clear();
    MTLTextureDescriptor* desc = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:pixelFormat
                                                                                       width:count
                                                                             resourceOptions:MTLResourceStorageModePrivate
                                                                                       usage:MTLTextureUsageShaderRead];
    view = [tex.rgb32Shadow newTextureWithDescriptor:desc offset:0 bytesPerRow:rowBytes];
    if (view) tex.views.push_back({key, view});
    return view;
}

// A texture view of a buffer texture's data store, rebuilt when the buffer's
// Metal storage changes (BufferData reallocates it).
id<MTLTexture> bufferTextureView(Context& ctx, Texture& tex) {
    Buffer* buffer = tex.buffer.get();
    if (buffer && buffer->mtl &&
        (tex.bufferFormat == GL_RGB32F || tex.bufferFormat == GL_RGB32I || tex.bufferFormat == GL_RGB32UI))
        return rgb32BufferView(ctx, tex, buffer);
    const FormatInfo* format = formatInfo(tex.bufferFormat);
    if (!buffer || !buffer->mtl || !format) return nil;
    const uint64_t key = reinterpret_cast<uint64_t>((__bridge void*)buffer->mtl);
    for (const Texture::View& v : tex.views)
        if (v.key == key) return v.view;
    tex.views.clear();
    const MTLPixelFormat pixelFormat = mtlPixelFormat(format->storage);
    const size_t texel = storageBytes(format->storage);
    const size_t offset = static_cast<size_t>(tex.bufferOffset);
    const size_t available = buffer->mtl.length > offset ? buffer->mtl.length - offset : 0;
    const size_t requested = tex.bufferSize < 0 ? static_cast<size_t>(buffer->size) - offset : static_cast<size_t>(tex.bufferSize);
    const size_t alignment = [Device::get().mtl() minimumLinearTextureAlignmentForPixelFormat:pixelFormat];
    NSUInteger width = static_cast<NSUInteger>(std::min<size_t>(std::min(requested, available) / texel, 1u << 27));
    // bytesPerRow is aligned up; keep it inside the buffer.
    auto rowBytes = [&](NSUInteger w) { return (w * texel + alignment - 1) / alignment * alignment; };
    while (width > 0 && rowBytes(width) > available) --width;
    if (width == 0) return nil;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:pixelFormat
                                                                                       width:width
                                                                             resourceOptions:buffer->mtl.resourceOptions
                                                                                       usage:MTLTextureUsageShaderRead];
    id<MTLTexture> view = [buffer->mtl newTextureWithDescriptor:desc offset:offset bytesPerRow:rowBytes(width)];
    if (view) tex.views.push_back({key, view});
    return view;
}

}  // namespace

bool textureComplete(Context& ctx, Texture& tex, const SamplerParams* params) {
    if (tex.target == GL_TEXTURE_BUFFER) return bufferTextureView(ctx, tex) != nil;
    const SamplerParams& p = params ? *params : tex.sampler;
    // Immutable textures clamp the base and max levels to [0, levels - 1] (GL 4.2 section 3.9.14).
    const int levels = tex.immutable ? std::max<int>(1, tex.immutableLevels) : Texture::kMaxLevels;
    const int base = tex.immutable ? std::clamp(tex.baseLevel, 0, levels - 1) : tex.baseLevel;
    if (base < 0 || base >= Texture::kMaxLevels) return false;
    const TextureImage& baseImage = tex.image(0, base);
    if (!baseImage.defined() || baseImage.width == 0 || baseImage.height == 0 || baseImage.depth == 0) return false;
    // Cube maps: every face square, same size and format.
    if (tex.target == GL_TEXTURE_CUBE_MAP) {
        for (int face = 0; face < 6; ++face) {
            const TextureImage& image = tex.image(face, base);
            if (!image.defined() || image.width != baseImage.width || image.height != baseImage.height ||
                image.internalFormat != baseImage.internalFormat || image.width != image.height)
                return false;
        }
    }
    const bool mipmapped = p.minFilter != GL_NEAREST && p.minFilter != GL_LINEAR &&
                           tex.target != GL_TEXTURE_RECTANGLE && tex.target != GL_TEXTURE_2D_MULTISAMPLE &&
                           tex.target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
    // Multisample textures have no sampler state and are complete once defined.
    const bool multisample = tex.target == GL_TEXTURE_2D_MULTISAMPLE || tex.target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
    if (multisample) return ensureTextureStorage(ctx, tex);
    // Integer formats cannot be filtered; depth formats only with compare mode
    // or nearest filtering in some cases (GL 3.3: depth textures filter linearly).
    // Stencil indices sample like integers (GL 4.6 section 8.17).
    const bool stencilSampled = baseImage.format && baseImage.format->isStencil() &&
                                (!baseImage.format->isDepth() || tex.depthStencilMode == GL_STENCIL_INDEX);
    // Stencil of a depth-stencil texture is sampled through a view in another format.
    if (stencilSampled && baseImage.format->isDepth() && !tex.formatViews) {
        Texture& root = tex.viewOf ? *tex.viewOf : tex;
        tex.formatViews = root.formatViews = true;
        root.storageDirty = tex.storageDirty = true;
    }
    if (baseImage.format && (baseImage.format->isInteger() || stencilSampled) &&
        (p.magFilter != GL_NEAREST || (p.minFilter != GL_NEAREST && p.minFilter != GL_NEAREST_MIPMAP_NEAREST)))
        return false;
    if (mipmapped) {
        const int faces = tex.faceCount();
        // 1D arrays keep their layer count in height; only 3D textures shrink in depth.
        const bool oneD = tex.target == GL_TEXTURE_1D || tex.target == GL_TEXTURE_1D_ARRAY;
        const int maxLevel = std::min({tex.maxLevel, levels - 1, Texture::kMaxLevels - 1,
                                       base + fullMipCount(baseImage.width, oneD ? 1 : baseImage.height,
                                                           tex.target == GL_TEXTURE_3D ? baseImage.depth : 1) - 1});
        for (int level = base; level <= maxLevel; ++level) {
            GLsizei w, h, d;
            levelSize(tex.target, baseImage.width, baseImage.height, baseImage.depth, level - base, w, h, d);
            for (int face = 0; face < faces; ++face) {
                const TextureImage& image = tex.image(face, level);
                if (!image.defined() || image.width != w || image.height != h || image.depth != d ||
                    image.internalFormat != baseImage.internalFormat)
                    return false;
            }
        }
    }
    return ensureTextureStorage(ctx, tex);
}

namespace {

// The format an sRGB format's texels are read as without decoding
// (GL_SKIP_DECODE_EXT); other formats are their own.
MTLPixelFormat withoutSrgb(MTLPixelFormat f) {
    switch (f) {
    case MTLPixelFormatRGBA8Unorm_sRGB: return MTLPixelFormatRGBA8Unorm;
    case MTLPixelFormatBGRA8Unorm_sRGB: return MTLPixelFormatBGRA8Unorm;
    case MTLPixelFormatR8Unorm_sRGB: return MTLPixelFormatR8Unorm;
    case MTLPixelFormatRG8Unorm_sRGB: return MTLPixelFormatRG8Unorm;
    case MTLPixelFormatBC1_RGBA_sRGB: return MTLPixelFormatBC1_RGBA;
    case MTLPixelFormatBC2_RGBA_sRGB: return MTLPixelFormatBC2_RGBA;
    case MTLPixelFormatBC3_RGBA_sRGB: return MTLPixelFormatBC3_RGBA;
    case MTLPixelFormatBC7_RGBAUnorm_sRGB: return MTLPixelFormatBC7_RGBAUnorm;
    case MTLPixelFormatETC2_RGB8_sRGB: return MTLPixelFormatETC2_RGB8;
    case MTLPixelFormatETC2_RGB8A1_sRGB: return MTLPixelFormatETC2_RGB8A1;
    case MTLPixelFormatEAC_RGBA8_sRGB: return MTLPixelFormatEAC_RGBA8;
    default: return f;
    }
}

}  // namespace

id<MTLTexture> samplingView(Context& ctx, Texture& tex, bool skipSrgbDecode) {
    if (tex.target == GL_TEXTURE_BUFFER) return bufferTextureView(ctx, tex);
    const TextureImage& baseImage = tex.baseImage();
    const FormatInfo* format = baseImage.format;
    if (!format || !tex.mtl) return tex.mtl;
    const int base = std::clamp(tex.baseLevel, 0, std::max(0, tex.mtlLevels - 1));
    int top = std::min(tex.maxLevel, tex.mtlLevels - 1);
    // Levels past the last defined one are never sampled when complete.
    while (top > base && !tex.image(0, top).defined()) --top;
    const NSUInteger levelCount = static_cast<NSUInteger>(std::max(1, top - base + 1));
    GLenum swizzle[4] = {tex.swizzle[0], tex.swizzle[1], tex.swizzle[2], tex.swizzle[3]};
    MTLPixelFormat viewFormat = tex.mtl.pixelFormat;
    // Depth/stencil textures sample depth by default; stencil sampling (GL 4.3) uses the stencil view.
    if (viewFormat == MTLPixelFormatDepth32Float_Stencil8 && tex.depthStencilMode == GL_STENCIL_INDEX)
        viewFormat = MTLPixelFormatX32_Stencil8;
    if (skipSrgbDecode) viewFormat = withoutSrgb(viewFormat);
    const uint64_t key = (static_cast<uint64_t>(base) << 56) | (static_cast<uint64_t>(levelCount) << 48) |
                         (static_cast<uint64_t>(viewFormat) << 32) | (static_cast<uint64_t>(swizzle[0] & 0xff) << 24) |
                         (static_cast<uint64_t>(swizzle[1] & 0xff) << 16) | (static_cast<uint64_t>(swizzle[2] & 0xff) << 8) |
                         static_cast<uint64_t>(swizzle[3] & 0xff);
    for (const Texture::View& v : tex.views)
        if (v.key == key) return v.view;
    const NSUInteger slices = tex.mtl.textureType == MTLTextureTypeCube ? 6
                              : tex.mtl.textureType == MTLTextureTypeCubeArray ? tex.mtl.arrayLength * 6
                                                                              : tex.mtl.arrayLength;
    const bool depthFormat = format->isDepthOrStencil();
    MTLTextureSwizzleChannels channels =
        depthFormat ? MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleZero, MTLTextureSwizzleZero,
                                                    MTLTextureSwizzleOne)
                    : composeSwizzle(*format, swizzle);
    if (depthFormat) {
        // Depth reads as (d, 0, 0, 1) and the GL swizzle applies on top.
        FormatInfo depthAsRed = *format;
        depthAsRed.swizzle[0] = SwzR;
        depthAsRed.swizzle[1] = SwzZero;
        depthAsRed.swizzle[2] = SwzZero;
        depthAsRed.swizzle[3] = SwzOne;
        channels = composeSwizzle(depthAsRed, swizzle);
    }
    id<MTLTexture> view = [tex.mtl newTextureViewWithPixelFormat:viewFormat
                                                    textureType:tex.mtl.textureType
                                                         levels:NSMakeRange(static_cast<NSUInteger>(base), levelCount)
                                                         slices:NSMakeRange(0, std::max<NSUInteger>(1, slices))
                                                        swizzle:channels];
    if (!view) view = tex.mtl;
    tex.views.push_back({key, view});
    return view;
}

bool borderEmulation(const Texture& tex, const SamplerParams& p, bool shadow, shader::BorderSlot& out, bool always) {
    int dims = 0;
    switch (tex.target) {
    case GL_TEXTURE_1D: case GL_TEXTURE_1D_ARRAY: dims = 1; break;
    case GL_TEXTURE_2D: case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_RECTANGLE: dims = 2; break;
    case GL_TEXTURE_3D: dims = 3; break;
    default: return false;  // cube maps filter seamlessly; buffers and multisample textures are not filtered
    }
    const GLenum wraps[3] = {p.wrapS, p.wrapT, p.wrapR};
    out = {};
    for (int a = 0; a < dims; ++a) {
        uint32_t mode = 3;
        switch (wraps[a]) {
        case GL_CLAMP_TO_BORDER: mode = 0; out.axes |= 1u << a; break;
        case GL_REPEAT: mode = 1; break;
        case GL_MIRRORED_REPEAT: mode = 2; break;
        case GL_MIRROR_CLAMP_TO_EDGE: mode = 4; break;
        default: break;
        }
        out.wrap |= mode << (4 * a);
    }
    out.always = always;
    if (!out.axes && !always) return false;
    const FormatInfo* format = tex.baseImage().format;
    if (!format) return false;
    const bool stencil = format->isStencil() && (!format->isDepth() || tex.depthStencilMode == GL_STENCIL_INDEX);
    out.kind = stencil || format->type == ComponentType::UnsignedInt ? 'u' : format->type == ComponentType::Int ? 'i' : 'f';
    // The border as the texel of the image's base format (GL 4.6 section
    // 8.14.2, table 8.11), then the texture's swizzle.
    float f[4];
    int32_t n[4];
    for (int k = 0; k < 4; ++k) {
        f[k] = p.borderIsInteger ? static_cast<float>(p.borderColor.i[k]) : p.borderColor.f[k];
        n[k] = p.borderIsInteger ? p.borderColor.i[k] : static_cast<int32_t>(p.borderColor.f[k]);
    }
    int source[4] = {0, 1, 2, 3};  // 0-3: component, 4: zero, 5: one
    switch (format->isDepthOrStencil() ? GL_RED : format->baseFormat) {
    case GL_RED: source[1] = source[2] = 4; source[3] = 5; break;
    case GL_RG: source[2] = 4; source[3] = 5; break;
    case GL_RGB: source[3] = 5; break;
    case GL_ALPHA: source[0] = source[1] = source[2] = 4; break;
    case GL_LUMINANCE: source[1] = source[2] = 0; source[3] = 5; break;
    case GL_LUMINANCE_ALPHA: source[1] = source[2] = 0; break;
    case GL_INTENSITY: source[1] = source[2] = source[3] = 0; break;
    default: break;
    }
    bool identity = format->baseFormat == GL_RGBA || format->baseFormat == GL_BGRA;
    for (int k = 0; k < 4; ++k) {
        int s = 4;
        switch (tex.swizzle[k]) {
        case GL_RED: s = source[0]; break;
        case GL_GREEN: s = source[1]; break;
        case GL_BLUE: s = source[2]; break;
        case GL_ALPHA: s = source[3]; break;
        case GL_ONE: s = 5; break;
        default: break;
        }
        identity = identity && s == k;
        if (out.kind == 'f') {
            const float v = s == 4 ? 0.0f : s == 5 ? 1.0f : f[s];
            std::memcpy(&out.border[k], &v, 4);
        } else {
            const int32_t v = s == 4 ? 0 : s == 5 ? 1 : n[s];
            std::memcpy(&out.border[k], &v, 4);
        }
    }
    // What the Metal sampler gives (samplerState): exact for these RGBA
    // borders, and for shadow lookups whose border depth it has.
    const bool rect = tex.target == GL_TEXTURE_RECTANGLE;
    const float metal = f[3] == 0.0f ? 0.0f : (f[0] + f[1] + f[2]) > 1.5f ? 1.0f : 0.0f;
    if (!rect && out.kind == 'f' && !always) {
        if (shadow && f[0] == metal) return false;
        const bool black = f[0] == 0.0f && f[1] == 0.0f && f[2] == 0.0f, white = f[0] == 1.0f && f[1] == 1.0f && f[2] == 1.0f;
        if (!shadow && identity && ((black && (f[3] == 0.0f || f[3] == 1.0f)) || (white && f[3] == 1.0f))) return false;
    }
    if (!rect && out.kind != 'f' && !shadow && n[0] == 0 && n[1] == 0 && n[2] == 0 && n[3] == 0 && identity && !always)
        return false;
    const auto linear = [](GLenum filter) {
        return filter == GL_LINEAR || filter == GL_LINEAR_MIPMAP_NEAREST || filter == GL_LINEAR_MIPMAP_LINEAR;
    };
    out.magLinear = linear(p.magFilter);
    out.minLinear = linear(p.minFilter);
    out.mip = p.minFilter == GL_NEAREST_MIPMAP_NEAREST || p.minFilter == GL_LINEAR_MIPMAP_NEAREST  ? 1
              : p.minFilter == GL_NEAREST_MIPMAP_LINEAR || p.minFilter == GL_LINEAR_MIPMAP_LINEAR ? 2
                                                                                                  : 0;
    out.rect = rect;
    out.minLod = std::max(0.0f, p.minLod);
    out.maxLod = std::max(out.minLod, p.maxLod);
    out.lodBias = p.lodBias;
    out.compare = shadow && p.compareMode == GL_COMPARE_REF_TO_TEXTURE ? p.compareFunc : 0;
    return true;
}

id<MTLSamplerState> samplerState(Context& ctx, const SamplerParams& p, bool shadow) {
    SamplerKey key;
    std::memset(&key, 0, sizeof(key));
    auto minMag = [](GLenum f) {
        return static_cast<uint8_t>(f == GL_NEAREST || f == GL_NEAREST_MIPMAP_NEAREST || f == GL_NEAREST_MIPMAP_LINEAR
                                        ? MTLSamplerMinMagFilterNearest
                                        : MTLSamplerMinMagFilterLinear);
    };
    key.minFilter = minMag(p.minFilter);
    key.magFilter = minMag(p.magFilter);
    switch (p.minFilter) {
    case GL_NEAREST_MIPMAP_NEAREST: case GL_LINEAR_MIPMAP_NEAREST: key.mipFilter = MTLSamplerMipFilterNearest; break;
    case GL_NEAREST_MIPMAP_LINEAR: case GL_LINEAR_MIPMAP_LINEAR: key.mipFilter = MTLSamplerMipFilterLinear; break;
    default: key.mipFilter = MTLSamplerMipFilterNotMipmapped; break;
    }
    key.sAddress = static_cast<uint8_t>(mtlAddressMode(p.wrapS));
    key.tAddress = static_cast<uint8_t>(mtlAddressMode(p.wrapT));
    key.rAddress = static_cast<uint8_t>(mtlAddressMode(p.wrapR));
    key.compare = static_cast<uint8_t>(shadow ? (p.compareMode == GL_COMPARE_REF_TO_TEXTURE ? mtlCompare(p.compareFunc)
                                                                                           : MTLCompareFunctionAlways)
                                              : MTLCompareFunctionNever);
    // Metal border colours are limited to these three; others are approximated (W12).
    // An integer border colour (glTexParameterI*) is read as integers.
    float c[4];
    for (int i = 0; i < 4; ++i) c[i] = p.borderIsInteger ? static_cast<float>(p.borderColor.i[i]) : p.borderColor.f[i];
    key.border = static_cast<uint8_t>(c[3] == 0.0f ? MTLSamplerBorderColorTransparentBlack
                                      : (c[0] + c[1] + c[2]) > 1.5f ? MTLSamplerBorderColorOpaqueWhite
                                                                    : MTLSamplerBorderColorOpaqueBlack);
    key.maxAnisotropy = static_cast<uint8_t>(std::clamp(p.maxAnisotropy, 1.0f, 16.0f));
    key.normalized = 1;
    key.lodMin = std::max(0.0f, p.minLod);
    key.lodMax = std::max(key.lodMin, p.maxLod);
    // Apple GPUs truncate both the sampler and the shader bias to 1/64;
    // rounding the sampler bias up keeps their sum within 1/64 of GL's.
    key.lodBias = std::ceil(std::clamp(p.lodBias, -limits::kMaxTextureLodBias, limits::kMaxTextureLodBias) * 64.0f) / 64.0f;
    return ctx.share().pipelines().sampler(key);
}

id<MTLSamplerState> samplerState(Context& ctx, Texture& tex, const SamplerParams& params, bool shadow) {
    SamplerParams p = params;
    if (tex.target == GL_TEXTURE_RECTANGLE) {
        // Rectangle textures use unnormalized coordinates and no mipmaps.
        SamplerKey key;
        std::memset(&key, 0, sizeof(key));
        key.minFilter = p.minFilter == GL_NEAREST ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
        key.magFilter = p.magFilter == GL_NEAREST ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
        key.mipFilter = MTLSamplerMipFilterNotMipmapped;
        key.sAddress = key.tAddress = key.rAddress = MTLSamplerAddressModeClampToEdge;
        key.compare = static_cast<uint8_t>(shadow && p.compareMode == GL_COMPARE_REF_TO_TEXTURE ? mtlCompare(p.compareFunc)
                                                                                               : MTLCompareFunctionNever);
        key.maxAnisotropy = 1;
        key.normalized = 0;
        key.lodMax = 0;
        return ctx.share().pipelines().sampler(key);
    }
    // OpenGL ES samples cube maps seamlessly, ignoring their wrap modes
    // (section 8.13.1); Metal's repeat would still pick texels at face edges
    // differently for cube map arrays.
    if (ctx.isEs() && (tex.target == GL_TEXTURE_CUBE_MAP || tex.target == GL_TEXTURE_CUBE_MAP_ARRAY))
        p.wrapS = p.wrapT = p.wrapR = GL_CLAMP_TO_EDGE;
    return samplerState(ctx, p, shadow);
}

id<MTLTexture> dummyTexture(GLenum samplerType) {
    static std::mutex mutex;
    static std::unordered_map<GLenum, id<MTLTexture>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(samplerType);
    if (it != cache.end()) return it->second;
    MTLPixelFormat format = MTLPixelFormatRGBA8Unorm;
    switch (samplerType) {
    case GL_INT_SAMPLER_1D: case GL_INT_SAMPLER_2D: case GL_INT_SAMPLER_3D: case GL_INT_SAMPLER_CUBE:
    case GL_INT_SAMPLER_1D_ARRAY: case GL_INT_SAMPLER_2D_ARRAY: case GL_INT_SAMPLER_2D_RECT: case GL_INT_SAMPLER_BUFFER:
    case GL_INT_SAMPLER_2D_MULTISAMPLE: case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        format = MTLPixelFormatRGBA32Sint;
        break;
    case GL_UNSIGNED_INT_SAMPLER_1D: case GL_UNSIGNED_INT_SAMPLER_2D: case GL_UNSIGNED_INT_SAMPLER_3D:
    case GL_UNSIGNED_INT_SAMPLER_CUBE: case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY: case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
    case GL_UNSIGNED_INT_SAMPLER_2D_RECT: case GL_UNSIGNED_INT_SAMPLER_BUFFER:
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE: case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        format = MTLPixelFormatRGBA32Uint;
        break;
    case GL_SAMPLER_1D_SHADOW: case GL_SAMPLER_2D_SHADOW: case GL_SAMPLER_1D_ARRAY_SHADOW: case GL_SAMPLER_2D_ARRAY_SHADOW:
    case GL_SAMPLER_CUBE_SHADOW: case GL_SAMPLER_2D_RECT_SHADOW:
        format = MTLPixelFormatDepth32Float;
        break;
    default: break;
    }
    MTLTextureType type = MTLTextureType2D;
    switch (samplerType) {
    case GL_SAMPLER_3D: case GL_INT_SAMPLER_3D: case GL_UNSIGNED_INT_SAMPLER_3D: type = MTLTextureType3D; break;
    case GL_SAMPLER_CUBE: case GL_INT_SAMPLER_CUBE: case GL_UNSIGNED_INT_SAMPLER_CUBE: case GL_SAMPLER_CUBE_SHADOW:
        type = MTLTextureTypeCube;
        break;
    case GL_SAMPLER_1D_ARRAY: case GL_SAMPLER_2D_ARRAY: case GL_INT_SAMPLER_1D_ARRAY: case GL_INT_SAMPLER_2D_ARRAY:
    case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY: case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY: case GL_SAMPLER_1D_ARRAY_SHADOW:
    case GL_SAMPLER_2D_ARRAY_SHADOW:
        type = MTLTextureType2DArray;
        break;
    case GL_SAMPLER_2D_MULTISAMPLE: case GL_INT_SAMPLER_2D_MULTISAMPLE: case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
        type = MTLTextureType2DMultisample;
        break;
    case GL_SAMPLER_2D_MULTISAMPLE_ARRAY: case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        type = MTLTextureType2DMultisampleArray;
        break;
    default: break;
    }
    if (samplerType == GL_SAMPLER_BUFFER || samplerType == GL_INT_SAMPLER_BUFFER || samplerType == GL_UNSIGNED_INT_SAMPLER_BUFFER) {
        // Buffer samplers are MSL texture_buffers, which must be backed by a buffer.
        const uint32_t texel[4] = {0, 0, 0, format == MTLPixelFormatRGBA8Unorm ? 0xffu : 1u};
        const uint32_t rgba8 = 0xff000000u;
        const bool unorm = format == MTLPixelFormatRGBA8Unorm;
        id<MTLBuffer> storage = [Device::get().mtl() newBufferWithLength:256 options:MTLResourceStorageModeShared];
        std::memcpy(storage.contents, unorm ? static_cast<const void*>(&rgba8) : texel, unorm ? 4 : 16);
        MTLTextureDescriptor* desc = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:format
                                                                                            width:1
                                                                                  resourceOptions:MTLResourceStorageModeShared
                                                                                            usage:MTLTextureUsageShaderRead];
        id<MTLTexture> texture = [storage newTextureWithDescriptor:desc offset:0 bytesPerRow:256];
        cache[samplerType] = texture;
        return texture;
    }
    MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
    desc.textureType = type;
    desc.pixelFormat = format;
    desc.width = desc.height = 1;
    desc.storageMode = MTLStorageModeShared;
    if (type == MTLTextureType2DMultisample || type == MTLTextureType2DMultisampleArray) {
        desc.sampleCount = 4;
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    }
    if (format == MTLPixelFormatDepth32Float) desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> texture = [Device::get().mtl() newTextureWithDescriptor:desc];
    if (desc.storageMode == MTLStorageModeShared) {
        const NSUInteger slices = type == MTLTextureTypeCube ? 6 : 1;
        uint8_t texel[16] = {};
        if (format == MTLPixelFormatRGBA8Unorm) texel[3] = 255;
        else if (format == MTLPixelFormatRGBA32Sint || format == MTLPixelFormatRGBA32Uint) texel[12] = 1;
        const NSUInteger bpp = format == MTLPixelFormatRGBA8Unorm ? 4 : 16;
        for (NSUInteger slice = 0; slice < slices; ++slice)
            [texture replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1) mipmapLevel:0 slice:slice withBytes:texel
                       bytesPerRow:bpp bytesPerImage:bpp];
    }
    cache[samplerType] = texture;
    return texture;
}

void uploadTexels(Context& ctx, Texture& tex, int face, int level, int x, int y, int z, int width, int height, int depth,
                  const void* data, size_t bytesPerRow, size_t bytesPerImage) {
    if (!tex.mtl || width <= 0 || height <= 0 || depth <= 0) return;
    const bool array = tex.mtl.textureType == MTLTextureType2DArray || tex.mtl.textureType == MTLTextureTypeCubeArray;
    const size_t total = bytesPerImage * static_cast<size_t>(depth);
    if (tex.mtl.pixelFormat == MTLPixelFormatDepth32Float_Stencil8) {
        // Storage texels are (float depth, uint8 stencil, pad); Metal uploads each aspect separately.
        const size_t texels = total / 8;
        const size_t rowTexels = bytesPerRow / 8, imageTexels = bytesPerImage / 8;
        TransientAllocation depthStaging = ctx.draw().transient().allocate(texels * 4, ctx.stream().serial(), 16);
        TransientAllocation stencilStaging = ctx.draw().transient().allocate(texels, ctx.stream().serial(), 16);
        const uint8_t* in = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < texels; ++i) {
            std::memcpy(static_cast<uint8_t*>(depthStaging.data) + i * 4, in + i * 8, 4);
            static_cast<uint8_t*>(stencilStaging.data)[i] = in[i * 8 + 4];
        }
        id<MTLBlitCommandEncoder> blit = ctx.stream().blit();
        for (int layer = 0; layer < depth; ++layer) {
            const NSUInteger slice = static_cast<NSUInteger>(tex.target == GL_TEXTURE_CUBE_MAP ? face : array ? z + layer : 0);
            const size_t first = imageTexels * static_cast<size_t>(layer);
            [blit copyFromBuffer:depthStaging.buffer sourceOffset:depthStaging.offset + first * 4
                   sourceBytesPerRow:rowTexels * 4 sourceBytesPerImage:imageTexels * 4
                          sourceSize:MTLSizeMake(width, height, 1) toTexture:tex.mtl destinationSlice:slice
                    destinationLevel:static_cast<NSUInteger>(level) destinationOrigin:MTLOriginMake(x, y, 0)
                             options:MTLBlitOptionDepthFromDepthStencil];
            [blit copyFromBuffer:stencilStaging.buffer sourceOffset:stencilStaging.offset + first
                   sourceBytesPerRow:rowTexels sourceBytesPerImage:imageTexels
                          sourceSize:MTLSizeMake(width, height, 1) toTexture:tex.mtl destinationSlice:slice
                    destinationLevel:static_cast<NSUInteger>(level) destinationOrigin:MTLOriginMake(x, y, 0)
                             options:MTLBlitOptionStencilFromDepthStencil];
        }
        ++tex.version;
        return;
    }
    TransientAllocation staging = ctx.draw().transient().allocate(total, ctx.stream().serial(), 16);
    if (!staging.data) return;
    std::memcpy(staging.data, data, total);
    uploadStaged(ctx, tex, face, level, x, y, z, width, height, depth, staging, bytesPerRow, bytesPerImage);
}

void uploadStaged(Context& ctx, Texture& tex, int face, int level, int x, int y, int z, int width, int height, int depth,
                  const TransientAllocation& staging, size_t bytesPerRow, size_t bytesPerImage) {
    const bool array = tex.mtl.textureType == MTLTextureType2DArray || tex.mtl.textureType == MTLTextureTypeCubeArray;
    id<MTLBlitCommandEncoder> blit = ctx.stream().blit();
    if (tex.mtl.textureType == MTLTextureType3D) {
        [blit copyFromBuffer:staging.buffer sourceOffset:staging.offset sourceBytesPerRow:bytesPerRow
            sourceBytesPerImage:bytesPerImage sourceSize:MTLSizeMake(width, height, depth) toTexture:tex.mtl
               destinationSlice:0 destinationLevel:static_cast<NSUInteger>(level) destinationOrigin:MTLOriginMake(x, y, z)];
        ++tex.version;
        return;
    }
    // Arrays (and 1D arrays, whose layers are rows of the GL image) upload one slice at a time.
    for (int layer = 0; layer < depth; ++layer) {
        const NSUInteger slice = static_cast<NSUInteger>(tex.target == GL_TEXTURE_CUBE_MAP ? face : array ? z + layer : 0);
        [blit copyFromBuffer:staging.buffer sourceOffset:staging.offset + bytesPerImage * static_cast<size_t>(layer)
               sourceBytesPerRow:bytesPerRow sourceBytesPerImage:bytesPerImage sourceSize:MTLSizeMake(width, height, 1)
                       toTexture:tex.mtl destinationSlice:slice destinationLevel:static_cast<NSUInteger>(level)
               destinationOrigin:MTLOriginMake(x, y, 0)];
    }
    ++tex.version;
}

// ------------------------------------------------------------ image units

namespace {

enum class ImageShape { Plane, Array, Volume, Buffer, Multisample };

// How the shader declares an image type: the MSL texture type it expects
// (1D images are 2D textures and cube images 2D arrays, as the compiler
// emits them) and its component kind.
ImageShape imageShape(GLenum type) {
    switch (type) {
    case GL_IMAGE_1D: case GL_IMAGE_2D: case GL_IMAGE_2D_RECT: case GL_INT_IMAGE_1D: case GL_INT_IMAGE_2D:
    case GL_INT_IMAGE_2D_RECT: case GL_UNSIGNED_INT_IMAGE_1D: case GL_UNSIGNED_INT_IMAGE_2D: case GL_UNSIGNED_INT_IMAGE_2D_RECT:
        return ImageShape::Plane;
    case GL_IMAGE_3D: case GL_INT_IMAGE_3D: case GL_UNSIGNED_INT_IMAGE_3D: return ImageShape::Volume;
    case GL_IMAGE_BUFFER: case GL_INT_IMAGE_BUFFER: case GL_UNSIGNED_INT_IMAGE_BUFFER: return ImageShape::Buffer;
    case GL_IMAGE_2D_MULTISAMPLE: case GL_IMAGE_2D_MULTISAMPLE_ARRAY: case GL_INT_IMAGE_2D_MULTISAMPLE:
    case GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY: case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE:
    case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY:
        return ImageShape::Multisample;
    default: return ImageShape::Array;  // 1D/2D arrays, cubes and cube arrays
    }
}

char imageKind(GLenum type) {
    if (type >= GL_INT_IMAGE_1D && type <= GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY) return 'i';
    if (type >= GL_UNSIGNED_INT_IMAGE_1D && type <= GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY) return 'u';
    return 'f';
}

}  // namespace

id<MTLTexture> dummyImage(GLenum imageType) {
    static std::mutex mutex;
    static std::unordered_map<GLenum, id<MTLTexture>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    if (auto it = cache.find(imageType); it != cache.end()) return it->second;
    const char kind = imageKind(imageType);
    const MTLPixelFormat format = kind == 'i' ? MTLPixelFormatR32Sint : kind == 'u' ? MTLPixelFormatR32Uint : MTLPixelFormatRGBA32Float;
    const MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | (kind == 'f' ? 0 : MTLTextureUsageShaderAtomic);
    id<MTLTexture> texture = nil;
    const ImageShape shape = imageShape(imageType);
    if (shape == ImageShape::Buffer) {
        id<MTLBuffer> storage = [Device::get().mtl() newBufferWithLength:256 options:MTLResourceStorageModePrivate];
        MTLTextureDescriptor* desc = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:format
                                                                                            width:1
                                                                                  resourceOptions:MTLResourceStorageModePrivate
                                                                                            usage:usage];
        texture = [storage newTextureWithDescriptor:desc offset:0 bytesPerRow:256];
    } else {
        MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
        desc.textureType = shape == ImageShape::Volume ? MTLTextureType3D
                           : shape == ImageShape::Array ? MTLTextureType2DArray
                           : shape == ImageShape::Multisample ? MTLTextureType2DMultisample
                                                              : MTLTextureType2D;
        if (shape == ImageShape::Multisample) desc.sampleCount = 4;
        desc.pixelFormat = format;
        desc.width = desc.height = 1;
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = shape == ImageShape::Multisample ? MTLTextureUsageShaderRead : usage;
        texture = [Device::get().mtl() newTextureWithDescriptor:desc];
    }
    cache[imageType] = texture;
    return texture;
}

namespace {

constexpr uint64_t kShadowTag = 0x4000000000000000ull;

id<MTLTexture> findView(const Texture& tex, uint64_t key) {
    for (const Texture::View& v : tex.views)
        if (v.key == key) return v.view;
    return nil;
}

// A shadow texture (ImageShadow) of `type` and `format` for `slices` slices
// (or `planes` depth planes) of tex.mtl's `level`.
id<MTLTexture> shadowTexture(Texture& tex, MTLTextureType type, MTLPixelFormat format, NSUInteger level,
                             NSUInteger firstSlice, NSUInteger slices, NSUInteger firstPlane, NSUInteger planes,
                             ImageShadow* out) {
    const uint64_t key = kShadowTag | (static_cast<uint64_t>(format) << 40) | (static_cast<uint64_t>(type) << 32) |
                         (static_cast<uint64_t>(level & 0xff) << 24) |
                         (static_cast<uint64_t>((firstSlice + firstPlane) & 0xfff) << 12) |
                         static_cast<uint64_t>((slices * planes) & 0xfff);
    const NSUInteger width = std::max<NSUInteger>(1, tex.mtl.width >> level);
    const NSUInteger height = std::max<NSUInteger>(1, tex.mtl.height >> level);
    id<MTLTexture> shadow = findView(tex, key);
    if (!shadow) {
        const bool atomics = format == MTLPixelFormatR32Uint || format == MTLPixelFormatR32Sint;
        MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
        desc.textureType = type;
        desc.pixelFormat = format;
        desc.width = width;
        desc.height = height;
        desc.depth = type == MTLTextureType3D ? planes : 1;
        desc.arrayLength = type == MTLTextureType2DArray ? slices : 1;
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | (atomics ? MTLTextureUsageShaderAtomic : 0);
        shadow = [Device::get().mtl() newTextureWithDescriptor:desc];
        if (!shadow) return nil;
        tex.views.push_back({key, shadow});
    }
    if (out) {
        out->texture = tex.mtl;
        out->shadow = shadow;
        out->level = level;
        out->firstSlice = firstSlice;
        out->slices = slices;
        out->firstPlane = firstPlane;
        out->planes = planes;
        out->width = width;
        out->height = height;
        out->texelBytes = storageBytes(tex.storage);
    }
    return shadow;
}

}  // namespace

void copyImageShadow(Context& ctx, const ImageShadow& s, bool toShadow) {
    // Through a buffer: the shadow's format may differ from the texture's
    // (same texel size), which texture-to-texture copies do not allow.
    const NSUInteger row = s.width * s.texelBytes, image = row * s.height;
    id<MTLBuffer> staging = [Device::get().mtl() newBufferWithLength:std::max<NSUInteger>(1, image * s.planes * s.slices)
                                                            options:MTLResourceStorageModePrivate];
    if (!staging) return;
    id<MTLBlitCommandEncoder> blit = ctx.stream().blit();
    const MTLSize size = MTLSizeMake(s.width, s.height, s.planes);
    for (NSUInteger i = 0; i < s.slices; ++i) {
        const NSUInteger offset = image * s.planes * i;
        id<MTLTexture> from = toShadow ? s.texture : s.shadow, to = toShadow ? s.shadow : s.texture;
        const NSUInteger fromSlice = toShadow ? s.firstSlice + i : i, toSlice = toShadow ? i : s.firstSlice + i;
        const NSUInteger fromLevel = toShadow ? s.level : 0, toLevel = toShadow ? 0 : s.level;
        const MTLOrigin fromOrigin = MTLOriginMake(0, 0, toShadow ? s.firstPlane : 0);
        const MTLOrigin toOrigin = MTLOriginMake(0, 0, toShadow ? 0 : s.firstPlane);
        [blit copyFromTexture:from sourceSlice:fromSlice sourceLevel:fromLevel sourceOrigin:fromOrigin sourceSize:size
                     toBuffer:staging destinationOffset:offset destinationBytesPerRow:row destinationBytesPerImage:image];
        [blit copyFromBuffer:staging sourceOffset:offset sourceBytesPerRow:row sourceBytesPerImage:image sourceSize:size
                   toTexture:to destinationSlice:toSlice destinationLevel:toLevel destinationOrigin:toOrigin];
    }
}

id<MTLTexture> imageView(Context& ctx, const ImageUnit& unit, GLenum imageType, GLenum shaderFormat,
                         ImageShadow* shadow) {
    if (shadow) shadow->shadow = nil;
    Texture* tex = unit.texture.get();
    const FormatInfo* format = formatInfo(shaderFormat ? shaderFormat : unit.format);
    if (!tex || !format) return nil;
    const ImageShape shape = imageShape(imageType);
    const MTLPixelFormat pixelFormat = mtlPixelFormat(format->storage);
    if (pixelFormat == MTLPixelFormatInvalid || shape == ImageShape::Multisample) return nil;
    const uint64_t viewTag = 0x8000000000000000ull;
    if (tex->target == GL_TEXTURE_BUFFER) {
        // A buffer image: a writable texture view of the buffer's data.
        Buffer* buffer = tex->buffer.get();
        if (shape != ImageShape::Buffer || !buffer || !buffer->mtl) return nil;
        const uint64_t key = viewTag | (static_cast<uint64_t>(pixelFormat) << 40) | (reinterpret_cast<uint64_t>((__bridge void*)buffer->mtl) & 0xffffffffffull);
        for (const Texture::View& v : tex->views)
            if (v.key == key) return v.view;
        // Views of the buffer's earlier storage (replaced by glBufferData or
        // an orphaning write) are never used again: dropped, with the
        // storage they keep alive.
        const uint64_t storage = key & 0xffffffffffull;
        std::erase_if(tex->views, [&](const Texture::View& v) { return (v.key & viewTag) && (v.key & 0xffffffffffull) != storage; });
        const size_t texel = storageBytes(format->storage);
        const size_t offset = static_cast<size_t>(tex->bufferOffset);
        const size_t available = buffer->mtl.length > offset ? buffer->mtl.length - offset : 0;
        const size_t requested = tex->bufferSize < 0 ? static_cast<size_t>(buffer->size) - offset : static_cast<size_t>(tex->bufferSize);
        const size_t alignment = [Device::get().mtl() minimumLinearTextureAlignmentForPixelFormat:pixelFormat];
        NSUInteger width = static_cast<NSUInteger>(std::min<size_t>(std::min(requested, available) / texel, 1u << 27));
        auto rowBytes = [&](NSUInteger w) { return (w * texel + alignment - 1) / alignment * alignment; };
        while (width > 0 && rowBytes(width) > available) --width;
        if (width == 0 || offset % alignment) return nil;
        const bool atomics = pixelFormat == MTLPixelFormatR32Uint || pixelFormat == MTLPixelFormatR32Sint;
        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            textureBufferDescriptorWithPixelFormat:pixelFormat
                                             width:width
                                   resourceOptions:buffer->mtl.resourceOptions
                                             usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                                                   (atomics ? MTLTextureUsageShaderAtomic : 0)];
        id<MTLTexture> view = [buffer->mtl newTextureWithDescriptor:desc offset:offset bytesPerRow:rowBytes(width)];
        if (view) tex->views.push_back({key, view});
        buffer->lastUseSerial = buffer->lastWriteSerial = ctx.stream().serial();
        ++buffer->version;
        return view;
    }
    if (!ensureTextureStorage(ctx, *tex) || !tex->mtl || !(tex->mtl.usage & MTLTextureUsageShaderWrite)) return nil;
    if (unit.level >= tex->mtlLevels) return nil;
    // A level that was never defined (a mipmap-incomplete texture bound at
    // another level than its base) is not accessible: loads return 0 and
    // stores are dropped (GL 4.6 section 8.26), whatever its storage holds.
    if (!tex->image(0, unit.level).defined()) return nil;
    // Image formats reinterpret a texture of the same texel size (GL 4.2
    // IMAGE_FORMAT_COMPATIBILITY_BY_SIZE).
    if (storageBytes(format->storage) != storageBytes(tex->storage) || format->compressed) return nil;
    const MTLTextureType type = tex->mtl.textureType;
    const NSUInteger slices = type == MTLTextureTypeCube ? 6 : type == MTLTextureTypeCubeArray ? tex->mtl.arrayLength * 6
                                                                                                 : tex->mtl.arrayLength;
    const bool arrayed = type == MTLTextureType2DArray || type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray;
    MTLTextureType viewType;
    NSUInteger first = 0, count = 1;
    switch (shape) {
    case ImageShape::Plane:
        if (type == MTLTextureType3D) {
            // A slice of a 3D texture: a 2D shadow (ImageShadow).
            const NSUInteger level = static_cast<NSUInteger>(unit.level);
            const NSUInteger plane = unit.layered ? 0 : static_cast<NSUInteger>(unit.layer);
            if (unit.layer < 0 || plane >= std::max<NSUInteger>(1, tex->mtl.depth >> level)) return nil;
            return shadowTexture(*tex, MTLTextureType2D, pixelFormat, level, 0, 1, plane, 1, shadow);
        }
        // One layer of an array, cube or cube array (a layered binding
        // of such a texture to a 2D image reads layer 0).
        if (type != MTLTextureType2D && !arrayed) return nil;
        viewType = MTLTextureType2D;
        first = arrayed && !unit.layered ? static_cast<NSUInteger>(unit.layer) : 0;
        if (first >= std::max<NSUInteger>(1, slices)) return nil;
        break;
    case ImageShape::Array:
        if (!arrayed) return nil;
        viewType = MTLTextureType2DArray;
        first = unit.layered ? 0 : static_cast<NSUInteger>(unit.layer);
        count = unit.layered ? slices : 1;
        if (first + count > slices) return nil;
        break;
    case ImageShape::Volume:
        if (type != MTLTextureType3D) return nil;
        viewType = MTLTextureType3D;
        break;
    default: return nil;
    }
    // Texture atomics on another 32-bit format: an R32 shadow (ImageShadow).
    if ((pixelFormat == MTLPixelFormatR32Uint || pixelFormat == MTLPixelFormatR32Sint) &&
        !(tex->mtl.usage & MTLTextureUsageShaderAtomic)) {
        const NSUInteger level = static_cast<NSUInteger>(unit.level);
        const NSUInteger planes = viewType == MTLTextureType3D ? std::max<NSUInteger>(1, tex->mtl.depth >> level) : 1;
        return shadowTexture(*tex, viewType, pixelFormat, level, first, count, 0, planes, shadow);
    }
    const uint64_t key = viewTag | (static_cast<uint64_t>(pixelFormat) << 40) | (static_cast<uint64_t>(viewType) << 32) |
                         (static_cast<uint64_t>(unit.level & 0xff) << 24) | (static_cast<uint64_t>(first & 0xfff) << 12) |
                         static_cast<uint64_t>(count & 0xfff);
    for (const Texture::View& v : tex->views)
        if (v.key == key) return v.view;
    id<MTLTexture> view = [tex->mtl newTextureViewWithPixelFormat:pixelFormat
                                                      textureType:viewType
                                                           levels:NSMakeRange(static_cast<NSUInteger>(unit.level), 1)
                                                           slices:NSMakeRange(first, count)];
    if (view) tex->views.push_back({key, view});
    return view;
}

}  // namespace wmgl
