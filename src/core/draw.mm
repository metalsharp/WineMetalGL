#include "core/draw.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/compat/arb_programs.h"
#include "core/compat/fixed_function.h"
#include "core/format/vertex_decode.h"
#include "core/shader/glsl_upgrade.h"
#include "core/stats.h"
#include "core/context.h"
#include "core/metal/device.h"
#include "core/metal/enums.h"
#include "core/metal/pipeline_cache.h"
#include "core/metal/pixel_format.h"
#include "core/program.h"
#include "core/query.h"
#include "core/texture_storage.h"
#include "core/util/log.h"
#include "gl_api.gen.h"

namespace wmgl {


DrawState::DrawState(Context& ctx) : ctx_(ctx), transient_(ctx.share().queue) {}

// --------------------------------------------------------- render targets

namespace {

// Vertices one draw converts on the CPU at most (16M: 256 MB of float4s).
constexpr size_t kMaxConvertedVertices = size_t{1} << 24;

// sRGB storage is written (and blitted) raw unless GL_FRAMEBUFFER_SRGB is
// enabled: attachments then use a non-sRGB view of the same texture.
MTLPixelFormat linearFormat(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatRGBA8Unorm_sRGB: return MTLPixelFormatRGBA8Unorm;
    case MTLPixelFormatBGRA8Unorm_sRGB: return MTLPixelFormatBGRA8Unorm;
    default: return MTLPixelFormatInvalid;
    }
}

constexpr uint64_t kLinearRenderViewKey = ~0ull;

bool attachmentTarget(Context& ctx, const Attachment& a, RenderTargetDesc::Target& out, MTLPixelFormat& format,
                      uint32_t& width, uint32_t& height, uint32_t& samples, const FormatInfo*& info) {
    const bool linear = !ctx.state.framebufferSRGB;
    if (a.type == GL_RENDERBUFFER) {
        Renderbuffer* rb = a.renderbuffer.get();
        if (!rb || !rb->mtl) return false;
        out.texture = rb->mtl;
        if (linear && linearFormat(rb->mtl.pixelFormat)) {
            if (!rb->linearView) rb->linearView = [rb->mtl newTextureViewWithPixelFormat:linearFormat(rb->mtl.pixelFormat)];
            out.texture = rb->linearView;
        }
        out.level = 0;
        out.slice = 0;
        width = rb->width;
        height = rb->height;
        samples = std::max(1, rb->samples);
        format = out.texture.pixelFormat;
        info = rb->format;
        return true;
    }
    if (a.type == GL_TEXTURE) {
        Texture* tex = a.texture.get();
        if (!tex || !ensureTextureStorage(ctx, *tex)) return false;
        const int face = a.cubeFace ? static_cast<int>(a.cubeFace - GL_TEXTURE_CUBE_MAP_POSITIVE_X) : 0;
        const TextureImage& image = tex->image(face, a.level);
        // A level defined off the mip chain has no storage of its size.
        if (!image.defined() || !imageHasStorage(ctx, *tex, face, a.level)) return false;
        out.texture = tex->mtl;
        if (linear && linearFormat(tex->mtl.pixelFormat)) {
            id<MTLTexture> view = nil;
            for (const Texture::View& v : tex->views)
                if (v.key == kLinearRenderViewKey) view = v.view;
            if (!view) {
                view = [tex->mtl newTextureViewWithPixelFormat:linearFormat(tex->mtl.pixelFormat)];
                tex->views.push_back({kLinearRenderViewKey, view});
            }
            out.texture = view;
        }
        out.level = static_cast<uint32_t>(a.level);
        out.slice = a.layered ? 0u : tex->target == GL_TEXTURE_CUBE_MAP ? static_cast<uint32_t>(face) : static_cast<uint32_t>(a.layer);
        width = static_cast<uint32_t>(image.width);
        // A 1D array image's height is its layer count; each layer is one row.
        height = tex->target == GL_TEXTURE_1D_ARRAY ? 1u : static_cast<uint32_t>(image.height);
        samples = std::max(1, image.samples);
        format = out.texture.pixelFormat;
        info = image.format;
        return true;
    }
    return false;
}

// Whether an attachment is multisample in GL's terms (samples > 0).
bool multisampleAttachment(const Attachment& a) {
    if (a.type == GL_RENDERBUFFER) return a.renderbuffer && a.renderbuffer->samples > 0;
    const Texture* tex = a.texture.get();
    return a.type == GL_TEXTURE && tex &&
           (tex->target == GL_TEXTURE_2D_MULTISAMPLE || tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
}

// Layers of a layered attachment (glFramebufferTexture on an array, cube or
// 3D texture): the render target array length it gives a pass.
uint32_t attachmentLayers(const Attachment& a) {
    const Texture* tex = a.texture.get();
    if (a.type != GL_TEXTURE || !a.layered || !tex) return 0;
    if (tex->target == GL_TEXTURE_CUBE_MAP) return 6;
    // A 1D array's layers are its image height.
    const TextureImage& image = tex->image(0, a.level);
    return static_cast<uint32_t>(std::max(1, tex->target == GL_TEXTURE_1D_ARRAY ? image.height : image.depth));
}

}  // namespace

namespace {

// Why a framebuffer cannot be drawn to, logged once per framebuffer and
// attachment point: the draws it rejects (INVALID_FRAMEBUFFER_OPERATION)
// otherwise fail without a trace.
void warnUndrawable(const Framebuffer& fb, const char* point, const Attachment* a) {
    char key[64], what[320];
    std::snprintf(key, sizeof(key), "undrawable-%u-%s", fb.name, point);
    if (!a) {
        std::snprintf(what, sizeof(what), "no attachment to draw to (draw buffer 0 %#x)", fb.drawBuffers[0]);
    } else if (a->type == GL_RENDERBUFFER) {
        const Renderbuffer* rb = a->renderbuffer.get();
        std::snprintf(what, sizeof(what), "renderbuffer %u (format %#x, %dx%d, %d samples) has no storage", rb ? rb->name : 0,
                      rb ? rb->internalFormat : 0, rb ? rb->width : 0, rb ? rb->height : 0, rb ? rb->samples : 0);
    } else {
        Texture* tex = a->texture.get();
        const int face = a->cubeFace ? static_cast<int>(a->cubeFace - GL_TEXTURE_CUBE_MAP_POSITIVE_X) : 0;
        const TextureImage* image = tex ? &tex->image(face, a->level) : nullptr;
        std::snprintf(what, sizeof(what),
                      "texture %u (target %#x, level %d, format %#x, %dx%dx%d, %d samples, %s %d levels): %s",
                      tex ? tex->name : 0, tex ? tex->target : 0, a->level, image ? image->internalFormat : 0,
                      image ? image->width : 0, image ? image->height : 0, image ? image->depth : 0,
                      image ? image->samples : 0, tex && tex->immutable ? "immutable" : "mutable",
                      tex ? (tex->immutable ? tex->immutableLevels : 0) : 0,
                      !tex                       ? "no texture"
                      : !tex->mtl                ? "no storage could be made"
                      : !image->defined()        ? "the level is not defined"
                                                 : "the level has no storage");
    }
    logOnce(LogLevel::Warn, LogCategory::Draw, key, "framebuffer %u %s: %s; its draws are dropped", fb.name, point,
            what);
}

}  // namespace

bool DrawState::drawTarget(RenderTargetDesc& target, TargetFormats& formats, bool reportErrors) {
    target = RenderTargetDesc();
    formats = TargetFormats();
    Framebuffer* fb = ctx_.state.drawFramebuffer.get();
    if (!fb) {
        Surface* surface = ctx_.drawSurface();
        if (!surface) return false;
        surface->describe(target);
        const bool srgb = ctx_.state.framebufferSRGB && ctx_.windowSrgb();
        if (!srgb) target.color[0].texture = surface->linearColor();
        formats.color[0] = target.color[0].texture.pixelFormat;
        formats.srgb[0] = surface->srgbCapable() && srgb;
        formats.alphaIsOne[0] = surface->pixelFormat().alpha_bits == 0;
        if (target.depth.texture) formats.depth = target.depth.texture.pixelFormat;
        if (target.stencil.texture) formats.stencil = target.stencil.texture.pixelFormat;
        formats.depthAttached = surface->hasDepth();
        formats.stencilAttached = surface->hasStencil();
        // GL_NONE / GL_FRONT_AND_BACK draw buffer selection on the default framebuffer.
        if (ctx_.state.defaultDrawBuffer == GL_NONE) {
            target.color[0] = {};
            formats.color[0] = MTLPixelFormatInvalid;
        }
        return true;
    }
    uint32_t width = UINT32_MAX, height = UINT32_MAX, samples = 0, layers = UINT32_MAX;
    bool any = false;
    for (int i = 0; i < Framebuffer::kMaxColor; ++i) {
        const GLenum buffer = fb->drawBuffers[i];
        if (buffer == GL_NONE) continue;
        const int index = static_cast<int>(buffer - GL_COLOR_ATTACHMENT0);
        if (index < 0 || index >= Framebuffer::kMaxColor) continue;
        const Attachment& a = fb->color[index];
        if (!a.attached()) continue;
        uint32_t w, h, s;
        MTLPixelFormat format;
        const FormatInfo* info = nullptr;
        if (!attachmentTarget(ctx_, a, target.color[i], format, w, h, s, info)) {
            static const char* const kColor[] = {"color 0", "color 1", "color 2", "color 3",
                                                 "color 4", "color 5", "color 6", "color 7"};
            warnUndrawable(*fb, kColor[index], &a);
            if (reportErrors) ctx_.setError(GL_INVALID_FRAMEBUFFER_OPERATION);
            return false;
        }
        if (const uint32_t n = attachmentLayers(a)) layers = std::min(layers, n);
        formats.color[i] = format;
        formats.alphaIsOne[i] = info && info->storedAlphaIsOne;
        formats.integer[i] = info && info->isInteger();
        formats.srgb[i] = info && info->srgb;
        width = std::min(width, w);
        height = std::min(height, h);
        samples = s;
        any = true;
    }
    for (int which = 0; which < 2; ++which) {
        const Attachment& a = which == 0 ? fb->depth : fb->stencil;
        if (!a.attached()) continue;
        RenderTargetDesc::Target& t = which == 0 ? target.depth : target.stencil;
        uint32_t w, h, s;
        MTLPixelFormat format;
        const FormatInfo* info = nullptr;
        if (!attachmentTarget(ctx_, a, t, format, w, h, s, info)) {
            warnUndrawable(*fb, which == 0 ? "depth" : "stencil", &a);
            if (reportErrors) ctx_.setError(GL_INVALID_FRAMEBUFFER_OPERATION);
            return false;
        }
        if (const uint32_t n = attachmentLayers(a)) layers = std::min(layers, n);
        if (which == 0) formats.depth = format;
        else formats.stencil = format;
        width = std::min(width, w);
        height = std::min(height, h);
        samples = s;
        any = true;
    }
    if (!any && !fb->hasAttachments() && ctx_.version() >= 43 && fb->defaultWidth > 0 && fb->defaultHeight > 0) {
        // GL 4.3: no attachments; rasterization at the framebuffer's default size.
        target.width = static_cast<uint32_t>(fb->defaultWidth);
        target.height = static_cast<uint32_t>(fb->defaultHeight);
        uint32_t samples = 1;
        for (uint32_t s : {2u, 4u, 8u})
            if (static_cast<uint32_t>(fb->defaultSamples) > samples && Device::get().caps().sampleCountSupported[s]) samples = s;
        target.samples = samples;
        target.multisampled = fb->defaultSamples > 0;
        if (fb->defaultLayers > 0) target.layers = static_cast<uint32_t>(fb->defaultLayers);
        return true;
    }
    if (!any) {
        warnUndrawable(*fb, "draw buffers", nullptr);
        if (reportErrors) ctx_.setError(GL_INVALID_FRAMEBUFFER_OPERATION);
        return false;
    }
    formats.depthAttached = target.depth.texture != nil;
    formats.stencilAttached = target.stencil.texture != nil;
    // Combined depth/stencil textures must be bound to both attachment points in Metal.
    if (target.depth.texture && formats.depth == MTLPixelFormatDepth32Float_Stencil8 && !target.stencil.texture) {
        target.stencil = target.depth;
        formats.stencil = formats.depth;
    }
    if (target.stencil.texture && formats.stencil == MTLPixelFormatDepth32Float_Stencil8 && !target.depth.texture) {
        target.depth = target.stencil;
        formats.depth = formats.stencil;
    }
    target.width = width;
    target.height = height;
    target.samples = std::max(1u, samples);
    target.multisampled = target.samples > 1 || multisampleAttachment(fb->depth) || multisampleAttachment(fb->stencil);
    for (const Attachment& a : fb->color) target.multisampled |= multisampleAttachment(a);
    // A layered framebuffer (every attachment is layered, checked by
    // completeness) renders to the layers its attachments have in common.
    if (layers != UINT32_MAX) target.layers = layers;
    return true;
}

id<MTLTexture> DrawState::readColorTexture(uint32_t& level, uint32_t& slice, GLenum& internalFormat, int& width,
                                           int& height) {
    Framebuffer* fb = ctx_.state.readFramebuffer.get();
    level = slice = 0;
    if (!fb) {
        Surface* surface = ctx_.readSurface();
        if (!surface || ctx_.state.defaultReadBuffer == GL_NONE) return nil;
        surface->resolveForRead(ctx_.stream());
        internalFormat = surface->colorFormat(ctx_.windowSrgb());
        width = surface->width();
        height = surface->height();
        return surface->resolvedColor();
    }
    if (fb->readBuffer == GL_NONE) return nil;
    const int index = static_cast<int>(fb->readBuffer - GL_COLOR_ATTACHMENT0);
    if (index < 0 || index >= Framebuffer::kMaxColor || !fb->color[index].attached()) return nil;
    RenderTargetDesc::Target t;
    MTLPixelFormat format;
    uint32_t w, h, s;
    const FormatInfo* info = nullptr;
    if (!attachmentTarget(ctx_, fb->color[index], t, format, w, h, s, info)) return nil;
    level = t.level;
    slice = t.slice;
    internalFormat = info ? info->internalFormat : GL_RGBA8;
    width = static_cast<int>(w);
    height = static_cast<int>(h);
    return t.texture;
}

bool DrawState::readTarget(GLenum buffer, RenderTargetDesc::Target& out, const FormatInfo*& info, uint32_t& width,
                           uint32_t& height, uint32_t& samples) {
    out = RenderTargetDesc::Target();
    info = nullptr;
    Framebuffer* fb = ctx_.state.readFramebuffer.get();
    if (!fb) {
        Surface* surface = ctx_.readSurface();
        if (!surface) return false;
        width = static_cast<uint32_t>(surface->width());
        height = static_cast<uint32_t>(surface->height());
        samples = surface->samples();
        if (buffer == GL_COLOR) {
            if (ctx_.state.defaultReadBuffer == GL_NONE) return false;
            surface->resolveForRead(ctx_.stream());
            out.texture = surface->resolvedColor();
            if ((!ctx_.state.framebufferSRGB || !ctx_.windowSrgb()) && linearFormat(out.texture.pixelFormat))
                out.texture = [out.texture newTextureViewWithPixelFormat:linearFormat(out.texture.pixelFormat)];
            info = formatInfo(surface->colorFormat(ctx_.windowSrgb()));
            return true;
        }
        if ((buffer == GL_DEPTH && !surface->hasDepth()) || (buffer == GL_STENCIL && !surface->hasStencil())) return false;
        out.texture = surface->depthStencil();
        info = formatInfo(surface->depthStencilFormat());
        return out.texture != nil;
    }
    const Attachment* a = nullptr;
    if (buffer == GL_COLOR) {
        if (fb->readBuffer == GL_NONE) return false;
        const int index = static_cast<int>(fb->readBuffer - GL_COLOR_ATTACHMENT0);
        if (index < 0 || index >= Framebuffer::kMaxColor) return false;
        a = &fb->color[index];
    } else {
        a = buffer == GL_DEPTH ? &fb->depth : &fb->stencil;
    }
    if (!a->attached()) return false;
    MTLPixelFormat format;
    return attachmentTarget(ctx_, *a, out, format, width, height, samples, info);
}

// ------------------------------------------------------------- validation

namespace {
GLenum geometryInputFor(GLenum mode);
}  // namespace

bool DrawState::validateDraw(GLenum mode) {
    // GL_SELECT / GL_FEEDBACK render modes draw nothing (no records are produced).
    if (ctx_.state.fixed.renderMode != GL_RENDER) return false;
    switch (mode) {
    case GL_POINTS: case GL_LINES: case GL_LINE_LOOP: case GL_LINE_STRIP: case GL_TRIANGLES: case GL_TRIANGLE_STRIP:
    case GL_TRIANGLE_FAN: case GL_LINES_ADJACENCY: case GL_LINE_STRIP_ADJACENCY: case GL_TRIANGLES_ADJACENCY:
    case GL_TRIANGLE_STRIP_ADJACENCY:
        break;
    case GL_QUADS: case GL_QUAD_STRIP: case GL_POLYGON:
        if (ctx_.isCompat()) break;
        [[fallthrough]];
    case GL_PATCHES:
        if (mode == GL_PATCHES && ctx_.tessellationShaders()) break;
        [[fallthrough]];
    default:
        ctx_.setError(GL_INVALID_ENUM);
        return false;
    }
    Program* program = ctx_.state.program.get();
    if (!program && ctx_.state.programPipeline) {
        // GL_ARB_separate_shader_objects: an invalid pipeline cannot draw.
        if (!api::validateProgramPipeline(*ctx_.state.programPipeline, ctx_.isEs())) {
            ctx_.setError(GL_INVALID_OPERATION);
            return false;
        }
        LinkedProgram* lp = executable();
        if (!lp || !lp->link.stages[static_cast<int>(Stage::Vertex)].present) return false;
    } else if (!program) {
        // OpenGL ES reports an incomplete framebuffer before a missing program.
        if (ctx_.isEs() && api::CheckFramebufferStatus(ctx_, GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            ctx_.setError(GL_INVALID_FRAMEBUFFER_OPERATION);
            return false;
        }
        // Without a program, core profile draws are undefined but not an
        // error; the compatibility profile draws with fixed-function
        // processing (compat/fixed_function.h), or ARB assembly programs,
        // whose enabled stages must have loaded programs.
        if (ctx_.isCompat() && compat::arbProgramsInvalid(ctx_)) {
            ctx_.setError(GL_INVALID_OPERATION);
            return false;
        }
        if (ctx_.isCore() || !executable()) return false;
    } else if (!program->linked) {
        ctx_.setError(GL_INVALID_OPERATION);
        return false;
    }
    // OpenGL ES draws from vertex array 0 (its indirect draws do not: api/draw.mm).
    if (ctx_.isDesktopCore() && ctx_.state.vertexArray == ctx_.defaultVertexArray) {
        ctx_.setError(GL_INVALID_OPERATION);
        return false;
    }
    // KHR_blend_equation_advanced section 3.8.5: an advanced equation draws
    // one buffer, with a fragment shader declaring support for it.
    for (int i = 0; i < ContextState::kMaxDrawBuffers && ctx_.state.blendEnable; ++i) {
        const int advanced = (ctx_.state.blendEnable >> i) & 1 ? advancedBlendIndex(ctx_.state.blend[i].equationRGB) : -1;
        if (advanced < 0) continue;
        const Framebuffer* fb = ctx_.state.drawFramebuffer.get();
        int drawn = 0;
        if (fb)
            for (GLenum b : fb->drawBuffers) drawn += b != GL_NONE;
        else
            drawn = ctx_.state.defaultDrawBuffer == GL_FRONT_AND_BACK ? 2 : 1;
        LinkedProgram* lp = executable();
        const uint32_t support = lp ? lp->link.blendSupport : 0;
        if (drawn > 1 || !(support & ((1u << advanced) | (1u << 15)))) {
            ctx_.setError(GL_INVALID_OPERATION);
            return false;
        }
        break;
    }
    // Mapped buffers may not be used by draws.
    for (const VertexAttrib& a : ctx_.state.vertexArray->attribs)
        if (a.enabled && a.buffer && a.buffer->mappedNonPersistent()) {
            ctx_.setError(GL_INVALID_OPERATION);
            return false;
        }
    // GL_PATCHES draws exactly when a tessellation evaluation shader is
    // active (GL 4.0 section 2.12); tessellation needs a program object.
    LinkedProgram* executableNow = executable();
    const bool tessellated = executableNow && executableNow->hasStage(Stage::TessEval);
    if ((mode == GL_PATCHES) != tessellated) {
        ctx_.setError(GL_INVALID_OPERATION);
        return false;
    }
    // OpenGL ES tessellates only with a control shader too (section 11.1.2.1;
    // desktop GL takes the default levels).
    if (tessellated && ctx_.isEs() && !executableNow->hasStage(Stage::TessControl)) {
        ctx_.setError(GL_INVALID_OPERATION);
        return false;
    }
    if (tessellated && !executableNow->link.tess.active) return false;  // a pipeline without the vertex program's emulation
    // Active transform feedback restricts the draw mode (GL 3.3 section 2.15.2).
    const TransformFeedback& xfb = *ctx_.state.transformFeedback;
    if (xfb.active && !xfb.paused) {
        GLenum family = 0;
        switch (mode) {
        case GL_POINTS: family = GL_POINTS; break;
        case GL_LINES: case GL_LINE_LOOP: case GL_LINE_STRIP: family = GL_LINES; break;
        case GL_TRIANGLES: case GL_TRIANGLE_STRIP: case GL_TRIANGLE_FAN: family = GL_TRIANGLES; break;
        case GL_QUADS: case GL_QUAD_STRIP: case GL_POLYGON: family = GL_TRIANGLES; break;  // compatibility profile
        default: break;
        }
        // With a geometry shader, its output type must match instead; with
        // tessellation, the tessellator's output.
        LinkedProgram* lp = program ? program->linked.get() : nullptr;
        if (tessellated) {
            const shader::TessellationEmulation& t = executableNow->link.tess;
            family = t.pointMode ? GL_POINTS : t.mode == GL_ISOLINES ? GL_LINES : GL_TRIANGLES;
            if (executableNow->hasStage(Stage::Geometry)) {
                const GLenum out = executableNow->link.geometryOutputType;
                family = out == GL_POINTS ? GL_POINTS : out == GL_LINE_STRIP ? GL_LINES : GL_TRIANGLES;
            }
        } else if (lp && lp->link.geometry.active) {
            const GLenum out = lp->link.geometry.outputType;
            family = out == GL_POINTS ? GL_POINTS : out == GL_LINE_STRIP ? GL_LINES : GL_TRIANGLES;
        }
        if (family != xfb.primitiveMode) {
            ctx_.setError(GL_INVALID_OPERATION);
            return false;
        }
    }
    // A geometry shader accepts only draw modes of its input type (GL 3.3
    // section 2.12.1); after tessellation, the tessellator's output type.
    if (LinkedProgram* lp = program ? program->linked.get() : nullptr;
        lp && lp->link.geometry.active && geometryInputFor(mode) != lp->link.geometry.inputType) {
        ctx_.setError(GL_INVALID_OPERATION);
        return false;
    }
    if (tessellated && executableNow->hasStage(Stage::Geometry)) {
        const shader::TessellationEmulation& t = executableNow->link.tess;
        const GLenum produced = t.pointMode ? GL_POINTS : t.mode == GL_ISOLINES ? GL_LINES : GL_TRIANGLES;
        if (executableNow->link.geometryInputType != produced) {
            ctx_.setError(GL_INVALID_OPERATION);
            return false;
        }
    }
    if (LinkedProgram* lp = executable()) uncompressFeedbackTextures(*lp);
    RenderTargetDesc target;
    TargetFormats formats;
    if (!drawTarget(target, formats)) return false;
    commitLargeWork(&target);
    // The draw's first prepare() takes it instead of building it again.
    validatedTarget_ = std::move(target);
    validatedFormats_ = formats;
    validatedFor_ = drawCount_;
    return true;
}

// ---------------------------------------------------------- vertex input

namespace {

struct ConvertedAttribute {
    bool needed = false;
};

}  // namespace

// The array feeding attribute `location`. In the compatibility profile
// generic attribute 0 aliases the conventional vertex array (gl_Vertex).
// ARB vertex programs alias generic attributes with conventional ones as
// NVIDIA does (normal 2, colours 3 and 4, fog coordinate 5, texture
// coordinates 8-15); the GL leaves it implementation-dependent, and programs
// written against NVIDIA drivers (Doom 3's) read vertex.texcoord or
// vertex.normal from generic arrays. A binding reads its alias's array when
// only that one is enabled.
static int arbAttributeAlias(int location) {
    switch (location) {
    case 2: return kConvNormal;
    case 3: return kConvColor;
    case 4: return kConvSecondaryColor;
    case 5: return kConvFogCoord;
    case kConvNormal: return 2;
    case kConvColor: return 3;
    case kConvSecondaryColor: return 4;
    case kConvFogCoord: return 5;
    default: break;
    }
    if (location >= 8 && location < 16) return kConvTexCoord0 + location - 8;
    if (location >= kConvTexCoord0 && location < kConvTexCoord0 + 8) return 8 + location - kConvTexCoord0;
    return -1;
}

const VertexAttrib& DrawState::attribFor(const VertexArray& vao, int location) const {
    if (location == kConvVertex && !vao.attribs[kConvVertex].enabled && vao.attribs[0].enabled) return vao.attribs[0];
    if (arbAliases_ && !vao.attribs[static_cast<size_t>(location)].enabled) {
        const int alias = arbAttributeAlias(location);
        if (alias >= 0 && vao.attribs[static_cast<size_t>(alias)].enabled) return vao.attribs[static_cast<size_t>(alias)];
    }
    return vao.attribs[static_cast<size_t>(location)];
}

namespace {

// How a vertex input reads its attribute's element: the Metal format, the
// byte offset of the first byte read and the bytes read. 64-bit inputs
// (kind 'd') read raw 32-bit words, 16 bytes per location (shader/fp64.h).
struct VertexFetch {
    MTLVertexFormat format = MTLVertexFormatInvalid;
    NSUInteger offset = 0;
    uint32_t bytes = 0;
};

// Metal fetches integer formats only into shader inputs of the same
// signedness; GL reinterprets the bits (an unsigned array read as ivec4).
// 32-bit formats swap signedness bit for bit. Narrower ones would extend
// differently (UChar as Char sign-extends), so they are converted on the CPU.
MTLVertexFormat matchSignedness(MTLVertexFormat format, char kind) {
    static const std::pair<MTLVertexFormat, MTLVertexFormat> words[] = {
        {MTLVertexFormatInt, MTLVertexFormatUInt}, {MTLVertexFormatInt2, MTLVertexFormatUInt2},
        {MTLVertexFormatInt3, MTLVertexFormatUInt3}, {MTLVertexFormatInt4, MTLVertexFormatUInt4}};
    static const MTLVertexFormat signedNarrow[] = {
        MTLVertexFormatChar, MTLVertexFormatChar2, MTLVertexFormatChar3, MTLVertexFormatChar4,
        MTLVertexFormatShort, MTLVertexFormatShort2, MTLVertexFormatShort3, MTLVertexFormatShort4};
    static const MTLVertexFormat unsignedNarrow[] = {
        MTLVertexFormatUChar, MTLVertexFormatUChar2, MTLVertexFormatUChar3, MTLVertexFormatUChar4,
        MTLVertexFormatUShort, MTLVertexFormatUShort2, MTLVertexFormatUShort3, MTLVertexFormatUShort4};
    if (kind != 'i' && kind != 'u') return format;
    for (const auto& [signedFormat, unsignedFormat] : words) {
        if (format == signedFormat || format == unsignedFormat) return kind == 'i' ? signedFormat : unsignedFormat;
    }
    const auto& wrong = kind == 'i' ? unsignedNarrow : signedNarrow;
    if (std::find(std::begin(wrong), std::end(wrong), format) != std::end(wrong)) return MTLVertexFormatInvalid;
    return format;
}

VertexFetch vertexFetch(const shader::StageOutput::VertexInput& input, const VertexAttrib& a) {
    if (input.kind == 'd') {
        const uint32_t element = a.isDouble ? a.elementBytes() : 0,
                       start = 16u * static_cast<uint32_t>(input.part) + static_cast<uint32_t>(input.byteOffset);
        if (element <= start) return {MTLVertexFormatInvalid, static_cast<NSUInteger>(a.offset), 0};
        const uint32_t bytes = std::min({16u, element - start, static_cast<uint32_t>(input.components) * 4u});
        return {bytes == 8 ? MTLVertexFormatUInt2 : MTLVertexFormatUInt4, static_cast<NSUInteger>(a.offset) + start, bytes};
    }
    const MTLVertexFormat format = a.isDouble ? MTLVertexFormatInvalid : mtlVertexFormat(a.type, a.size, a.normalized, a.integer);
    return {a.integer ? matchSignedness(format, input.kind) : format, static_cast<NSUInteger>(a.offset), a.elementBytes()};
}

}  // namespace

size_t DrawState::vertexBufferSlots(LinkedProgram& program) {
    // A vertex function's storage block argument buffer (storage blocks and
    // atomic counter buffers) takes slot 28.
    const size_t vertex = static_cast<size_t>(Stage::Vertex);
    if (!program.link.stages[vertex].storageArgs.empty())
        return shader::kSlotStorageBlocks - shader::kSlotFirstVertexBuffer;
    return shader::kMaxVertexBufferSlots;
}

void DrawState::planVertexInputs(LinkedProgram& program, bool gather, VertexInputPlan& plan) const {
    using Plan = VertexInputPlan;
    const VertexArray& vao = *ctx_.state.vertexArray;
    plan = Plan();
    struct Window {
        size_t group;
        NSUInteger end;  // one past the highest byte the group's attributes read, within one element
    };
    std::vector<Window> windows;
    bool constants = false, sides = false, convertedPerVertex = false;
    std::vector<uint32_t> convertedDivisors;
    auto convert = [&](int location, uint32_t divisor) {
        plan.how[static_cast<size_t>(location)] = Plan::Converted;
        if (!divisor) convertedPerVertex = true;
        else if (std::find(convertedDivisors.begin(), convertedDivisors.end(), divisor) == convertedDivisors.end())
            convertedDivisors.push_back(divisor);
    };
    struct Direct {
        int location;
        const VertexAttrib* a;
        VertexFetch fetch;
    };
    std::vector<Direct> direct;
    for (const shader::StageOutput::VertexInput& input : program.stage(Stage::Vertex).vertexInputs) {
        const int location = input.location;
        if (location < 0 || location >= VertexArray::kAttribSlots) continue;
        if (lineSides_ && location == kLineSide) {
            plan.how[static_cast<size_t>(location)] = Plan::LineSide;
            sides = true;
            continue;
        }
        // The other endpoint of a wide line segment reads the position array in partner order.
        const bool partner = linePartners_ && location == kLinePartner;
        const VertexAttrib& a = attribFor(vao, partner ? static_cast<int>(kConvVertex) : input.source());
        if (!a.enabled) {
            plan.how[static_cast<size_t>(location)] = Plan::Constant;
            constants = true;
            continue;
        }
        const uint32_t stride = static_cast<uint32_t>(a.effectiveStride());
        const VertexFetch fetch = vertexFetch(input, a);
        const bool aligned = (fetch.offset % 4) == 0 && (stride % 4) == 0 && stride > 0;
        const bool instanceGathered = gather && a.divisor && instanceGather_;
        const bool gathered = ((gather || partner) && !a.divisor) || instanceGathered;
        if (a.buffer && a.buffer->mtl && fetch.format != MTLVertexFormatInvalid && aligned && !gathered)
            direct.push_back({location, &a, fetch});
        else
            convert(location, instanceGathered ? 0 : a.divisor);
    }
    // Attributes of one buffer, stride and divisor share a layout when they
    // all fit in one element window, whatever their order.
    std::vector<std::vector<int>> members;
    for (const Direct& d : direct) {
        const int location = d.location;
        const VertexAttrib* a = d.a;
        const uint32_t stride = static_cast<uint32_t>(a->effectiveStride());
        const NSUInteger offset = d.fetch.offset, end = offset + d.fetch.bytes;
        size_t chosen = plan.groups.size();
        for (Window& w : windows) {
            Plan::DirectGroup& g = plan.groups[w.group];
            if (g.buffer != a->buffer.get() || g.stride != stride || g.divisor != a->divisor) continue;
            const NSUInteger base = std::min(g.base, offset), top = std::max(w.end, end);
            if (top - base > stride) continue;
            g.base = base;
            w.end = top;
            chosen = w.group;
            break;
        }
        if (chosen == plan.groups.size()) {
            plan.groups.push_back({a->buffer.get(), offset, stride, a->divisor});
            windows.push_back({chosen, end});
            members.emplace_back();
        }
        members[chosen].push_back(location);
    }
    // Keep within the vertex buffer slots: the last direct groups fall back to conversion.
    auto layouts = [&] {
        return plan.groups.size() + (convertedPerVertex ? 1 : 0) + convertedDivisors.size() + (constants ? 1 : 0) +
               (sides ? 1 : 0);
    };
    const size_t slots = vertexBufferSlots(program);
    while (layouts() > slots && !plan.groups.empty()) {
        for (int location : members.back()) convert(location, plan.groups.back().divisor);
        plan.groups.pop_back();
        members.pop_back();
    }
    for (size_t g = 0; g < plan.groups.size(); ++g)
        for (int location : members[g]) {
            plan.how[static_cast<size_t>(location)] = Plan::Direct;
            plan.group[static_cast<size_t>(location)] = static_cast<int8_t>(g);
        }
}

void DrawState::waitForCpuVertexReads(LinkedProgram& program, bool gather) {
    // Waiting flushes the command buffer, so it must happen before a draw opens its encoder.
    Queue& queue = ctx_.stream().queue();
    const VertexArray& vao = *ctx_.state.vertexArray;
    VertexInputPlan& plan = plannedInputs_;
    planVertexInputs(program, gather, plan);
    // The same draw's bindVertexInputs takes this plan (nothing between them
    // changes vertex state), unless line emulation substitutes inputs.
    plannedFor_ = !lineSides_ && !linePartners_ ? drawCount_ : 0;
    plannedProgram_ = &program;
    plannedGather_ = gather;
    for (const shader::StageOutput::VertexInput& input : program.stage(Stage::Vertex).vertexInputs) {
        if (input.location < 0 || input.location >= VertexArray::kAttribSlots) continue;
        if (plan.how[static_cast<size_t>(input.location)] != VertexInputPlan::Converted) continue;
        const VertexAttrib& a = attribFor(vao, input.source());
        if (a.buffer && a.buffer->lastWriteSerial && !queue.isComplete(a.buffer->lastWriteSerial))
            ctx_.stream().waitFor(a.buffer->lastWriteSerial);
    }
}

bool DrawState::bindVertexInputs(id<MTLRenderCommandEncoder> encoder, PipelineKey& key, LinkedProgram& program,
                                 GLint firstVertex, GLsizei vertexCount, GLsizei instances,
                                 const std::vector<uint32_t>* gather) {
    using Plan = VertexInputPlan;
    VertexArray& vao = *ctx_.state.vertexArray;
    const uint64_t serial = ctx_.stream().serial();
    Plan reused;
    const bool planned = plannedFor_ && plannedFor_ == drawCount_ && plannedProgram_ == &program &&
                         plannedGather_ == (gather != nullptr) && !lineSides_ && !linePartners_ && !instanceGather_;
    if (!planned) planVertexInputs(program, gather != nullptr, reused);
    const Plan& plan = planned ? plannedInputs_ : reused;
    plannedFor_ = 0;
    struct Layout {
        id<MTLBuffer> buffer;
        NSUInteger offset;
        uint32_t stride;
        uint32_t divisor;
        bool constant;
    };
    std::vector<Layout> layouts;
    const size_t slots = vertexBufferSlots(program);
    auto addLayout = [&](id<MTLBuffer> buffer, NSUInteger offset, uint32_t stride, uint32_t divisor, bool constant) -> int {
        if (layouts.size() >= slots) return -1;
        layouts.push_back({buffer, offset, stride, divisor, constant});
        return static_cast<int>(layouts.size() - 1);
    };
    // Direct groups first: layout g is group g. Renumbered vertices (issue())
    // read from the draw's first vertex on.
    for (size_t gi = 0; gi < plan.groups.size(); ++gi) {
        const Plan::DirectGroup& g = plan.groups[gi];
        if (gpuGather_) {
            // The group's windows, gathered per vertex (drawPatchesIndirectGpu).
            if (addLayout(gpuGather_->buffer, gpuGather_->offsets[gi], gpuGather_->strides[gi], 0, false) < 0) return false;
        } else {
            const NSUInteger base = g.base + (g.divisor ? 0 : static_cast<NSUInteger>(vertexRebase_) * g.stride);
            if (addLayout(g.buffer->mtl, base, g.stride, g.divisor, false) < 0) return false;
        }
        g.buffer->lastUseSerial = serial;
    }
    struct Converted {
        int location;
        const VertexAttrib* attrib;
        const std::vector<uint32_t>* order;  // gathered: the vertex (or instance) fetched for each draw vertex
        char kind;
        int part;  // 'd': 16-byte part of the element
        int byteOffset = 0;  // 'd': where in the part (layout(component) packing)
        uint32_t divisor = 0;  // as fetched: 0 for instance attributes gathered per vertex
    };
    std::vector<Converted> converted;
    // Current values, one constant element each, packed into one buffer:
    // the GL attribute and (64-bit inputs) the 16-byte part of its value.
    struct ConstantInput {
        int location, attrib, part;
        char kind;
        int byteOffset = 0;
    };
    std::vector<ConstantInput> constants;
    for (const shader::StageOutput::VertexInput& input : program.stage(Stage::Vertex).vertexInputs) {
        const int location = input.location;
        if (location < 0 || location >= VertexArray::kAttribSlots) continue;
        VertexAttributeKey& out = key.attributes[location];
        const bool partner = linePartners_ && location == kLinePartner;
        const VertexAttrib& a = attribFor(vao, partner ? static_cast<int>(kConvVertex) : input.source());
        switch (plan.how[static_cast<size_t>(location)]) {
        case Plan::LineSide: {
            TransientAllocation alloc = transient_.allocate(lineSides_->size() * 16, serial, 16);
            auto* dst = static_cast<float*>(alloc.data);
            if (!dst) return false;
            for (size_t v = 0; v < lineSides_->size(); ++v) {
                dst[v * 4] = (*lineSides_)[v];
                dst[v * 4 + 1] = dst[v * 4 + 2] = 0;
                dst[v * 4 + 3] = 1;
            }
            const int layout = addLayout(alloc.buffer, alloc.offset, 16, 0, false);
            if (layout < 0) return false;
            out.format = static_cast<uint8_t>(MTLVertexFormatFloat4);
            out.layout = static_cast<uint8_t>(layout);
            out.offset = 0;
            break;
        }
        case Plan::Constant:
            out.format = static_cast<uint8_t>(input.kind == 'i' ? MTLVertexFormatInt4
                                              : input.kind == 'u' || input.kind == 'd' ? MTLVertexFormatUInt4
                                                                                       : MTLVertexFormatFloat4);
            out.offset = static_cast<uint16_t>(constants.size() * 16);
            constants.push_back({location, input.source(), input.part, input.kind, input.byteOffset});
            break;
        case Plan::Direct: {
            const size_t g = static_cast<size_t>(plan.group[static_cast<size_t>(location)]);
            const VertexFetch fetch = vertexFetch(input, a);
            out.format = static_cast<uint8_t>(fetch.format);
            out.layout = static_cast<uint8_t>(g);
            out.offset = static_cast<uint16_t>(fetch.offset - plan.groups[g].base);
            break;
        }
        case Plan::Converted:
            if (a.buffer && !a.buffer->mtl) {
                ctx_.setError(GL_INVALID_OPERATION);
                return false;
            }
            if (gather && a.divisor && instanceGather_)
                converted.push_back({location, &a, instanceGather_, input.kind, input.part, input.byteOffset, 0});
            else
                converted.push_back({location, &a, partner ? linePartners_ : gather, input.kind, input.part, input.byteOffset,
                                     a.divisor});
            if (stats::g_enabled) {
                // Why: client memory, a format Metal cannot fetch (the last such GL type), or alignment.
                if (!a.buffer) {
                    stats::count(&stats::Frame::convertedClient);
                } else if (vertexFetch(input, a).format == MTLVertexFormatInvalid) {
                    stats::count(&stats::Frame::convertedFormat);
                    stats::frame().convertedFormatType = a.type | (a.normalized ? 0x10000u : 0u);
                } else {
                    stats::count(&stats::Frame::convertedOther);
                }
            }
            break;
        case Plan::None: break;
        }
    }
    // Converted attributes: expanded to 4 x 32-bit components, interleaved in
    // one buffer per divisor (GPU writes to their buffers were waited for in
    // waitForCpuVertexReads).
    std::vector<uint32_t> divisors;
    for (const Converted& c : converted)
        if (std::find(divisors.begin(), divisors.end(), c.divisor) == divisors.end()) divisors.push_back(c.divisor);
    for (uint32_t divisor : divisors) {
        std::vector<const Converted*> group;
        for (const Converted& c : converted)
            if (c.divisor == divisor) group.push_back(&c);
        // Per-vertex members are all gathered or none (gathering is per draw).
        const bool gathered = !divisor && group.front()->order;
        uint32_t first = 0, count = 0;
        if (gathered) {
            count = static_cast<uint32_t>(group.front()->order->size());
        } else if (divisor) {
            // One element per `divisor` instances, from the base instance (in 64
            // bits: wined3d's per-instance elements of step rate 0 use UINT_MAX).
            count = baseInstance_ +
                    static_cast<uint32_t>((static_cast<uint64_t>(std::max<GLsizei>(1, instances)) + divisor - 1) / divisor);
        } else {
            first = static_cast<uint32_t>(std::max(0, firstVertex));
            count = static_cast<uint32_t>(std::max(0, vertexCount));
        }
        // Element v of the draw is at v - rebase (vertices renumbered by issue()).
        const uint32_t rebase = gathered || divisor ? 0 : static_cast<uint32_t>(vertexRebase_);
        const size_t elements = static_cast<size_t>(first - rebase) + count;
        if (elements > kMaxConvertedVertices) {
            // An index range this wide is almost certainly garbage indices.
            logOnce(LogLevel::Warn, LogCategory::Draw, "converted-span", "draw skipped: %zu vertices to convert",
                    elements);
            return false;
        }
        const size_t stride = 16 * group.size();
        const size_t bytes = std::max<size_t>(1, elements) * stride;
        auto bindGroup = [&](id<MTLBuffer> buffer, NSUInteger offset) {
            const int layout = addLayout(buffer, offset, static_cast<uint32_t>(stride), divisor, false);
            if (layout < 0) return false;
            for (size_t m = 0; m < group.size(); ++m) {
                VertexAttributeKey& out = key.attributes[group[m]->location];
                out.format = static_cast<uint8_t>(group[m]->kind == 'd' ? MTLVertexFormatUInt4
                                                  : group[m]->attrib->integer
                                                      ? (group[m]->kind == 'u' ? MTLVertexFormatUInt4 : MTLVertexFormatInt4)
                                                      : MTLVertexFormatFloat4);
                out.layout = static_cast<uint8_t>(layout);
                out.offset = static_cast<uint16_t>(m * 16);
            }
            return true;
        };
        // A group read from one buffer object that the application cannot
        // write behind GL's back keeps its conversion until the buffer changes.
        Buffer* cacheable = gathered ? nullptr : group.front()->attrib->buffer.get();
        for (const Converted* c : group)
            if (c->attrib->buffer.get() != cacheable) cacheable = nullptr;
        if (cacheable && (cacheable->mapped || cacheable->pinned)) cacheable = nullptr;
        std::vector<uint64_t> cacheKey;
        if (cacheable) {
            if (cacheable->convertedVersion != cacheable->version) {
                cacheable->converted.clear();
                cacheable->convertedVersion = cacheable->version;
            }
            cacheKey = {divisor, first, count, rebase};
            for (const Converted* c : group) {
                const VertexAttrib& a = *c->attrib;
                cacheKey.push_back(static_cast<uint64_t>(a.offset));
                cacheKey.push_back(static_cast<uint64_t>(a.effectiveStride()));
                cacheKey.push_back(uint64_t{a.type} | uint64_t(static_cast<uint32_t>(a.size) & 0xffff) << 32 |
                                   uint64_t{a.normalized} << 48 | uint64_t{a.integer} << 49 | uint64_t{a.isDouble} << 50);
                cacheKey.push_back(static_cast<uint64_t>(static_cast<uint8_t>(c->kind)) | uint64_t(static_cast<uint32_t>(c->part)) << 8 |
                                   uint64_t(static_cast<uint32_t>(c->byteOffset)) << 32);
            }
            auto it = cacheable->converted.find(cacheKey);
            if (it != cacheable->converted.end()) {
                stats::count(&stats::Frame::convertedReused);
                if (!bindGroup(it->second->mtl, 0)) return false;
                continue;
            }
            if (Buffer::convertedBytes() + bytes > Buffer::kMaxConvertedBytes) cacheable = nullptr;
        }
        stats::count(&stats::Frame::convertedVertices, count);
        TransientAllocation alloc{};
        id<MTLBuffer> kept = nil;
        if (cacheable) {
            kept = [Device::get().mtl() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!kept) cacheable = nullptr;
        }
        if (!kept) alloc = transient_.allocate(bytes, serial, 16);
        auto* dst = static_cast<uint8_t*>(kept ? kept.contents : alloc.data);
        if (!dst) return false;
        for (size_t m = 0; m < group.size(); ++m) {
            const VertexAttrib& a = *group[m]->attrib;
            const uint32_t elementBytes = a.elementBytes(), attribStride = static_cast<uint32_t>(a.effectiveStride());
            // Bytes readable from `source` (a buffer object's size; client arrays are unbounded).
            size_t readable = SIZE_MAX;
            const uint8_t* source = nullptr;
            if (a.buffer) {
                readable = a.buffer->size > a.offset ? static_cast<size_t>(a.buffer->size - a.offset) : 0;
                source = a.buffer->contents() + std::min<size_t>(static_cast<size_t>(a.offset), static_cast<size_t>(a.buffer->size));
            } else {
                source = reinterpret_cast<const uint8_t*>(a.offset);
            }
            if (!source) return false;
            // The common case: every element inside the buffer, read in order.
            const size_t lastAt = count ? static_cast<size_t>(first + count - 1) * attribStride : 0;
            if (!gathered && group[m]->kind != 'd' && !a.integer && count &&
                (readable == SIZE_MAX || (lastAt <= readable && readable - lastAt >= elementBytes)) &&
                decodeVertexFloats(source + static_cast<size_t>(first) * attribStride, attribStride, count, a.type, a.size,
                                   a.normalized, dst + static_cast<size_t>(first - rebase) * stride + m * 16, stride)) {
                if (a.buffer) a.buffer->lastUseSerial = serial;
                continue;
            }
            for (uint32_t v = first; v < first + count; ++v) {
                const uint32_t fetch = gathered ? (*group[m]->order)[v] : v;
                const size_t at = static_cast<size_t>(fetch) * attribStride;
                uint8_t* to = dst + static_cast<size_t>(v - rebase) * stride + m * 16;
                if (readable != SIZE_MAX && (at > readable || readable - at < elementBytes)) {
                    // Past the end of the buffer: GPUs read zeros (robust access), never past the allocation.
                    const int32_t one = 1;
                    const float oneF = 1.0f;
                    std::memset(to, 0, 16);
                    if (group[m]->kind != 'd')
                        std::memcpy(to + 12, a.integer ? static_cast<const void*>(&one) : static_cast<const void*>(&oneF), 4);
                    continue;
                }
                const uint8_t* element = source + at;
                if (group[m]->kind == 'd') {
                    // Raw 64-bit components (shader/fp64.h); anything but an L array reads zeros.
                    std::memset(to, 0, 16);
                    const uint32_t start = 16u * static_cast<uint32_t>(group[m]->part) + static_cast<uint32_t>(group[m]->byteOffset);
                    if (a.isDouble && elementBytes > start) std::memcpy(to, element + start, std::min(16u, elementBytes - start));
                } else if (a.integer) {
                    int32_t values[4];
                    decodeVertexInt(element, a.type, a.size, values);
                    std::memcpy(to, values, 16);
                } else {
                    float values[4];
                    decodeVertexFloat(element, a.type, a.size, a.normalized, values);
                    std::memcpy(to, values, 16);
                }
            }
            if (a.buffer) a.buffer->lastUseSerial = serial;
        }
        if (kept) {
            auto entry = std::make_unique<Buffer::ConvertedVertices>();
            entry->mtl = kept;
            entry->bytes = bytes;
            Buffer::convertedBytes() += bytes;
            cacheable->converted[std::move(cacheKey)] = std::move(entry);
            if (!bindGroup(kept, 0)) return false;
        } else if (!bindGroup(alloc.buffer, alloc.offset)) {
            return false;
        }
    }
    if (!constants.empty()) {
        TransientAllocation alloc = transient_.allocate(constants.size() * 16, serial, 16);
        if (!alloc.data) return false;
        for (size_t c = 0; c < constants.size(); ++c) {
            const ContextState::GenericAttrib& g = ctx_.state.genericAttribs[static_cast<size_t>(constants[c].attrib)];
            uint8_t* dst = static_cast<uint8_t*>(alloc.data) + c * 16;
            if (constants[c].kind == 'd') {
                const size_t start = 16 * static_cast<size_t>(constants[c].part) + static_cast<size_t>(constants[c].byteOffset);
                std::memset(dst, 0, 16);
                if (start < sizeof(g.d)) std::memcpy(dst, reinterpret_cast<const uint8_t*>(g.d) + start, std::min<size_t>(16, sizeof(g.d) - start));
            }
            else std::memcpy(dst, &g.value, 16);
        }
        const int layout = addLayout(alloc.buffer, alloc.offset, static_cast<uint32_t>(constants.size() * 16), 0, true);
        if (layout < 0) return false;
        for (const ConstantInput& c : constants) key.attributes[c.location].layout = static_cast<uint8_t>(layout);
    }
    key.layoutCount = static_cast<uint8_t>(layouts.size());
    for (size_t l = 0; l < layouts.size(); ++l) {
        const Layout& layout = layouts[l];
        key.layouts[l].stride = layout.stride;
        key.layouts[l].stepFunction = static_cast<uint8_t>(layout.constant ? MTLVertexStepFunctionConstant
                                                           : layout.divisor ? MTLVertexStepFunctionPerInstance
                                                                            : MTLVertexStepFunctionPerVertex);
        key.layouts[l].stepRate = layout.constant ? 0 : std::max(1u, layout.divisor);
        [encoder setVertexBuffer:layout.buffer offset:layout.offset atIndex:shader::kSlotFirstVertexBuffer + l];
    }
    return true;
}

Program* DrawState::ownerProgram() {
    return ctx_.state.program ? ctx_.state.program.get() : ffProgram_;
}

Program* DrawState::stageProgram(LinkedProgram& program, int stage) {
    return program.pipeline ? program.stageOwner[stage] : ownerProgram();
}

std::pair<Program*, size_t> DrawState::blockSource(LinkedProgram& program, size_t block) {
    if (!program.pipeline) return {ownerProgram(), block};
    return {program.blockOwner[block].first, static_cast<size_t>(program.blockOwner[block].second)};
}

std::pair<Program*, size_t> DrawState::samplerSource(LinkedProgram& program, size_t index) {
    if (!program.pipeline) return {ownerProgram(), index};
    return {program.samplerOwner[index].first, static_cast<size_t>(program.samplerOwner[index].second)};
}

std::pair<Program*, size_t> DrawState::imageSource(LinkedProgram& program, size_t index) {
    if (!program.pipeline) return {ownerProgram(), index};
    return {program.imageOwner[index].first, static_cast<size_t>(program.imageOwner[index].second)};
}

void DrawState::applyBuiltinUniforms(Program& program) {
    // gl_DepthRange (viewport 0's), gl_NumSamples and the draw parameters (glsl_upgrade.h).
    if (!program.linked) return;
    const shader::LinkResult::BuiltinUniforms& b = program.linked->link.builtinUniforms;
    auto store = [&](int offset, const void* value) {
        if (offset < 0 || static_cast<size_t>(offset) + 4 > program.uniformData.size()) return;
        uint8_t* at = program.uniformData.data() + offset;
        if (std::memcmp(at, value, 4) == 0) return;
        std::memcpy(at, value, 4);
        ++program.uniformVersion;
    };
    if (b.depthRange[0] >= 0) {
        const float n = static_cast<float>(ctx_.state.depthRanges[0][0]), f = static_cast<float>(ctx_.state.depthRanges[0][1]);
        const float values[3] = {n, f, f - n};
        for (int i = 0; i < 3; ++i) store(b.depthRange[i], &values[i]);
    }
    if (b.numSamples >= 0) {
        const int32_t samples = std::max<int32_t>(1, lastKey_.sampleCount);
        store(b.numSamples, &samples);
    }
    // Draw parameters: the command's base vertex (0 for array draws), base instance and draw index.
    const int32_t parameters[3] = {glBaseVertex_, static_cast<int32_t>(baseInstance_), drawId_};
    const int offsets[3] = {b.baseVertex, b.baseInstance, b.drawId};
    for (int i = 0; i < 3; ++i)
        if (offsets[i] >= 0) store(offsets[i], &parameters[i]);
}

GLint DrawState::imageUnit(LinkedProgram& program, size_t index) {
    const auto [owner, ownIndex] = imageSource(program, index);
    return owner && ownIndex < owner->imageUnits.size() ? owner->imageUnits[ownIndex] : 0;
}

id<MTLTexture> DrawState::imageBinding(LinkedProgram& program, size_t index) {
    const shader::ImageSlotInfo& slot = program.link.images[index];
    const GLint unit = imageUnit(program, index);
    id<MTLTexture> view = nil;
    if (unit >= 0 && unit < ContextState::kMaxImageUnits) {
        const ImageUnit& u = ctx_.state.imageUnits[unit];
        view = imageView(ctx_, u, slot.type, slot.format);
        if (view && u.texture) {
            const uint64_t serial = ctx_.stream().serial();
            u.texture->lastUseSerial = serial;
            if (u.access != GL_READ_ONLY) ctx_.stream().noteShaderWrites();
            // Stores to a buffer image change the buffer: CPU reads and later uploads wait for this draw.
            if (Buffer* buffer = u.texture->buffer.get(); buffer && u.access != GL_READ_ONLY) {
                buffer->lastUseSerial = buffer->lastWriteSerial = serial;
                ++buffer->version;
            } else if (buffer) {
                buffer->lastUseSerial = serial;
            }
        }
        if (!view && u.texture) WMGL_TRACE(Draw, "image unit %d: no view for image type 0x%x, format 0x%x", unit,
                                           slot.type, slot.format);
    }
    return view ? view : dummyImage(slot.type);
}

LinkedProgram* DrawState::executable() {
    if (override_) return override_;
    if (Program* p = ctx_.state.program.get()) return p->linked.get();
    if (ProgramPipeline* pp = ctx_.state.programPipeline.get()) {
        current_ = pipelineExecutable(*pp);
        return current_.get();
    }
    if (ctx_.isCore()) return nullptr;
    // Fixed-function processing: the generated program for the state at
    // this draw (chosen once per draw, see beginDraw).
    if (!ffChosen_) {
        // ARB assembly programs (compat/arb_programs.h), with fixed-function
        // code for a stage without one.
        const bool arb = compat::arbProgramsEnabled(ctx_);
        arbAliases_ = arb && compat::arbVertexProgramActive(ctx_);
        ffProgram_ = arb ? compat::arbDrawProgram(ctx_, drawMode_) : compat::fixedFunctionProgram(ctx_, drawMode_);
        ffPointSize_ = drawMode_ == GL_POINTS && !arbAliases_ && compat::fixedFunctionWritesPointSize(ctx_);
        ffWideLines_ = compat::fixedFunctionWideLines(ctx_, drawMode_);
        ffChosen_ = true;
    }
    return ffProgram_ && ffProgram_->linked ? ffProgram_->linked.get() : nullptr;
}

void DrawState::beginDraw(GLenum mode) {
    ++drawCount_;
    drawMode_ = mode;
    glBaseVertex_ = 0;
    ffChosen_ = false;
    ffProgram_ = nullptr;
    arbAliases_ = false;
    ffPointSize_ = false;
    ffWideLines_ = false;
}

void DrawState::bindUniforms(id<MTLRenderCommandEncoder> encoder, LinkedProgram& program, bool known) {
    // `known`: encoderState_ holds what the encoder has bound (encoderStateValid_);
    // driver and default uniforms it already has are not sent again.
    EncoderState& e = encoderState_;
    if (!known) {
        e.driverSet = false;
        e.fragmentDriverSet = false;
        for (auto& u : e.uniforms) u = {};
    }
    if (tessInputs_.active) {
        // The tessellation evaluation vertex function's domain points, patch records and control block.
        static const NSUInteger kSlots[3] = {shader::kSlotTessVertices, shader::kSlotTessPatches, shader::kSlotTessControl};
        for (int i = 0; i < 3; ++i)
            [encoder setVertexBuffer:tessInputs_.buffers[i] offset:tessInputs_.offsets[i] atIndex:kSlots[i]];
    }
    // Driver uniforms read by the generated vertex code (see compiler.cpp addPointSize).
    const struct {
        float pointSize;
        uint32_t programPointSize;
        uint32_t clipFlags;  // see compiler.cpp addPointSize
        uint32_t clipEnables;
    } driver = {std::clamp(ctx_.state.pointSize, 1.0f, 511.0f), ctx_.state.programPointSize || ffPointSize_ ? 1u : 0u,
                (ctx_.state.clipDepthMode == GL_ZERO_TO_ONE ? 1u : 0u) | (ctx_.state.clipOrigin == GL_UPPER_LEFT ? 2u : 0u),
                ctx_.state.clipDistanceEnables};
    static_assert(sizeof(driver) == sizeof(e.driver));
    if (!e.driverSet || std::memcmp(e.driver, &driver, sizeof(driver)) != 0) {
        [encoder setVertexBytes:&driver length:sizeof(driver) atIndex:shader::kSlotDriverUniforms];
        std::memcpy(e.driver, &driver, sizeof(driver));
        e.driverSet = true;
    }
    const uint64_t serial = ctx_.stream().serial();
    // Subroutine selections and built-in uniforms live in the stage programs' default uniforms.
    for (int s : {static_cast<int>(Stage::Vertex), static_cast<int>(Stage::Fragment)})
        if (Program* owner = program.link.stages[s].present ? stageProgram(program, s) : nullptr) {
            applySubroutineSelections(ctx_, *owner);
            applyBuiltinUniforms(*owner);
        }
    static id<MTLBuffer> zero = [Device::get().mtl() newBufferWithLength:65536 options:MTLResourceStorageModePrivate];
    // Default uniforms copied for one stage serve the other stage of the same program.
    const Program* uploadedOwner = nullptr;
    TransientAllocation uploaded{};
    for (int s : {static_cast<int>(Stage::Vertex), static_cast<int>(Stage::Fragment)}) {
        const shader::StageOutput& stage = program.link.stages[s];
        if (!stage.present) continue;
        const bool vertex = s == static_cast<int>(Stage::Vertex);
        if (!vertex && (stage.usesPointCoord || sampleMask_ != ~0u)) {
            // Fragment driver uniforms: the gl_PointCoord flip and the sample mask (compiler.cpp).
            const uint32_t fragmentDriver[2] = {ctx_.state.pointSpriteCoordOrigin == GL_UPPER_LEFT ? 1u : 0u, sampleMask_};
            if (!e.fragmentDriverSet || std::memcmp(e.fragmentDriver, fragmentDriver, sizeof(fragmentDriver)) != 0) {
                [encoder setFragmentBytes:fragmentDriver length:sizeof(fragmentDriver) atIndex:shader::kSlotDriverUniforms];
                std::memcpy(e.fragmentDriver, fragmentDriver, sizeof(fragmentDriver));
                e.fragmentDriverSet = true;
            }
        }
        // Default uniforms of the program that owns the stage.
        Program* owner = stageProgram(program, s);
        const uint32_t size = owner && owner->linked ? (owner->linked->link.defaultBlockSize + 15) & ~15u : 0;
        EncoderState::Uniforms& bound = e.uniforms[vertex ? 0 : 1];
        const EncoderState::Uniforms wanted = {owner, owner && owner->linked ? owner->linked->uid : 0,
                                               owner ? owner->uniformVersion : 0, size};
        if (stage.usesDefaultUniforms && size && bound.owner == wanted.owner && bound.uid == wanted.uid &&
            bound.version == wanted.version && bound.size == wanted.size) {
            // The encoder has these uniforms already.
        } else if (stage.usesDefaultUniforms && size) {
            Program& p = *owner;
            if (p.uniformData.size() < size) p.uniformData.resize(size, 0);
            bound = wanted;
            if (size <= 4096) {
                if (vertex) [encoder setVertexBytes:p.uniformData.data() length:size atIndex:shader::kSlotDefaultUniforms];
                else [encoder setFragmentBytes:p.uniformData.data() length:size atIndex:shader::kSlotDefaultUniforms];
            } else {
                if (uploadedOwner != &p) {
                    uploaded = transient_.allocate(size, serial, 256);
                    std::memcpy(uploaded.data, p.uniformData.data(), size);
                    stats::count(&stats::Frame::uniformBytes, size);
                    uploadedOwner = &p;
                }
                if (vertex) [encoder setVertexBuffer:uploaded.buffer offset:uploaded.offset atIndex:shader::kSlotDefaultUniforms];
                else [encoder setFragmentBuffer:uploaded.buffer offset:uploaded.offset atIndex:shader::kSlotDefaultUniforms];
            }
        }
        for (size_t b = 0; b < program.link.blocks.size(); ++b) {
            const int slot = program.link.blocks[b].stageSlot[s];
            if (slot < 0) continue;
            id<MTLBuffer> buffer = zero;
            NSUInteger offset = 0;
            uniformBlockBuffer(program, b, buffer, offset);
            if (vertex) [encoder setVertexBuffer:buffer offset:offset atIndex:static_cast<NSUInteger>(slot)];
            else [encoder setFragmentBuffer:buffer offset:offset atIndex:static_cast<NSUInteger>(slot)];
        }
        for (size_t b = 0; b < program.link.atomicBuffers.size(); ++b) {
            const int slot = program.link.atomicBuffers[b].stageSlot[s];
            if (slot < 0) continue;
            id<MTLBuffer> buffer = nil;
            NSUInteger offset = 0;
            atomicCounterBuffer(program, b, buffer, offset);
            if (vertex) [encoder setVertexBuffer:buffer offset:offset atIndex:static_cast<NSUInteger>(slot)];
            else [encoder setFragmentBuffer:buffer offset:offset atIndex:static_cast<NSUInteger>(slot)];
        }
        bindStorageBlocks(encoder, program, s);
    }
}

void DrawState::uniformBlockBuffer(LinkedProgram& program, size_t index, __strong id<MTLBuffer>& buffer,
                                   NSUInteger& offset) {
    if (static_cast<int>(index) == program.link.fixedFunctionBlock ||
        (program.pipeline && program.link.blocks[index].name == shader::kFixedFunctionBlockName)) {
        // Lowered legacy built-in uniforms (compat/fixed_function.h).
        uint64_t generation = 0;
        const std::vector<uint8_t>& data = compat::fixedFunctionBlock(ctx_, generation);
        const uint64_t serial = ctx_.stream().serial();
        if (!ffBlock_.buffer || ffBlock_.generation != generation || ffBlock_.serial != serial) {
            TransientAllocation alloc = transient_.allocate(data.size(), serial, 256);
            std::memcpy(alloc.data, data.data(), data.size());
            ffBlock_ = {alloc.buffer, alloc.offset, generation, serial};
        }
        buffer = ffBlock_.buffer;
        offset = ffBlock_.offset;
        return;
    }
    const auto [owner, block] = blockSource(program, index);
    if (!owner) return;
    const Program& p = *owner;
    const GLuint binding = block < p.uniformBlockBindings.size() ? p.uniformBlockBindings[block] : 0;
    if (binding >= ContextState::kMaxUniformBufferBindings) return;
    const IndexedBufferBinding& ib = ctx_.state.uniformBufferBindings[binding];
    if (!ib.buffer || !ib.buffer->mtl) return;
    // A range starting past the end (the buffer shrank after binding) reads as unbound.
    if (ib.offset < 0 || ib.offset >= ib.buffer->size) return;
    buffer = ib.buffer->mtl;
    offset = static_cast<NSUInteger>(ib.offset);
    ib.buffer->lastUseSerial = ctx_.stream().serial();
}

void DrawState::atomicCounterBuffer(LinkedProgram& program, size_t index, __strong id<MTLBuffer>& buffer, NSUInteger& offset) {
    static id<MTLBuffer> scratch = [Device::get().mtl() newBufferWithLength:65536 options:MTLResourceStorageModePrivate];
    buffer = scratch;
    offset = 0;
    const GLuint binding = program.link.atomicBuffers[index].binding;
    if (binding >= static_cast<GLuint>(ContextState::kMaxAtomicCounterBufferBindings)) return;
    const IndexedBufferBinding& ib = ctx_.state.atomicCounterBufferBindings[binding];
    if (!ib.buffer || !ib.buffer->mtl || ib.offset < 0 || ib.offset >= ib.buffer->size) return;
    buffer = ib.buffer->mtl;
    offset = static_cast<NSUInteger>(ib.offset);
    // Counters change the buffer: CPU reads and later uploads wait for this draw.
    ib.buffer->lastUseSerial = ib.buffer->lastWriteSerial = ctx_.stream().serial();
    ctx_.stream().noteShaderWrites();
    ++ib.buffer->version;
}

bool DrawState::storageArguments(const shader::StageOutput& stage, const Program* owner, StorageArguments& out) {
    if (stage.storageArgs.empty()) return false;
    // Unbound elements point at scratch memory (GL leaves their accesses undefined).
    static id<MTLBuffer> scratch = [Device::get().mtl() newBufferWithLength:65536 options:MTLResourceStorageModePrivate];
    const size_t n = stage.storageArgs.size();
    const uint64_t serial = ctx_.stream().serial();
    // Layout (shader/compiler.h): n device pointers, the pointer to the
    // sizes, then the n sizes in bytes.
    TransientAllocation alloc = transient_.allocate((n + 1) * 8 + n * 4, serial, 16);
    uint64_t* pointers = static_cast<uint64_t*>(alloc.data);
    uint32_t* sizes = reinterpret_cast<uint32_t*>(pointers + n + 1);
    out.resources.clear();
    for (size_t i = 0; i < n; ++i) {
        const shader::StageOutput::StorageArg& arg = stage.storageArgs[i];
        id<MTLBuffer> buffer = scratch;
        NSUInteger offset = 0;
        uint32_t size = 0;
        const bool counter = arg.counterBinding >= 0;
        const GLuint binding = counter ? static_cast<GLuint>(arg.counterBinding)
                               : owner && arg.block >= 0 && static_cast<size_t>(arg.block) < owner->storageBlockBindings.size()
                                   ? owner->storageBlockBindings[static_cast<size_t>(arg.block)]
                                   : 0;
        if (binding < static_cast<GLuint>(counter ? ContextState::kMaxAtomicCounterBufferBindings
                                                  : ContextState::kMaxShaderStorageBufferBindings)) {
            const IndexedBufferBinding& ib = counter ? ctx_.state.atomicCounterBufferBindings[binding]
                                                     : ctx_.state.shaderStorageBufferBindings[binding];
            if (ib.buffer && ib.buffer->mtl && ib.offset >= 0 && ib.offset < ib.buffer->size) {
                buffer = ib.buffer->mtl;
                offset = static_cast<NSUInteger>(ib.offset);
                const GLsizeiptr available = ib.buffer->size - ib.offset;
                size = static_cast<uint32_t>(std::min<GLsizeiptr>(ib.size ? std::min(ib.size, available) : available, UINT32_MAX));
                ib.buffer->lastUseSerial = serial;
                if (arg.writes) {
                    // Stores change the buffer: CPU reads and later uploads wait for this work.
                    ib.buffer->lastWriteSerial = serial;
                    ++ib.buffer->version;
                    ctx_.stream().noteShaderWrites();
                }
            }
        }
        pointers[i] = buffer.gpuAddress + offset;
        sizes[i] = size;
        out.resources.emplace_back(buffer, arg.writes);
    }
    pointers[n] = alloc.buffer.gpuAddress + alloc.offset + (n + 1) * 8;
    out.buffer = alloc.buffer;
    out.offset = alloc.offset;
    return true;
}

void DrawState::bindStorageBlocks(id<MTLRenderCommandEncoder> encoder, LinkedProgram& program, int s) {
    StorageArguments args;
    if (!storageArguments(program.link.stages[s], stageProgram(program, s), args)) return;
    const bool vertex = s == static_cast<int>(Stage::Vertex);
    const MTLRenderStages stages = vertex ? MTLRenderStageVertex : MTLRenderStageFragment;
    for (const auto& [buffer, writes] : args.resources)
        [encoder useResource:buffer usage:writes ? MTLResourceUsageRead | MTLResourceUsageWrite : MTLResourceUsageRead stages:stages];
    if (vertex) [encoder setVertexBuffer:args.buffer offset:args.offset atIndex:shader::kSlotStorageBlocks];
    else [encoder setFragmentBuffer:args.buffer offset:args.offset atIndex:shader::kSlotStorageBlocks];
}

void DrawState::bindStorageBlocks(id<MTLComputeCommandEncoder> encoder, const shader::StageOutput& stage, const Program* owner) {
    StorageArguments args;
    if (!storageArguments(stage, owner, args)) return;
    for (const auto& [buffer, writes] : args.resources)
        [encoder useResource:buffer usage:writes ? MTLResourceUsageRead | MTLResourceUsageWrite : MTLResourceUsageRead];
    [encoder setBuffer:args.buffer offset:args.offset atIndex:shader::kSlotStorageBlocks];
}

void DrawState::markSampled(Texture& texture) {
    texture.lastUseSerial = ctx_.stream().serial();
    // A buffer texture reads its buffer: writes to the buffer must wait for this draw.
    if (texture.target == GL_TEXTURE_BUFFER && texture.buffer) texture.buffer->lastUseSerial = texture.lastUseSerial;
}

void DrawState::uncompressFeedbackTextures(LinkedProgram& program) {
    const Framebuffer* fb = ctx_.state.drawFramebuffer.get();
    if (!fb || program.link.samplers.empty()) return;
    auto root = [](Texture* t) { return t && t->viewOf ? t->viewOf.get() : t; };
    for (size_t index = 0; index < program.link.samplers.size(); ++index) {
        const shader::SamplerSlotInfo& slot = program.link.samplers[index];
        const auto [owner, ownIndex] = samplerSource(program, index);
        const GLint unit = owner && ownIndex < owner->samplerUnits.size() ? owner->samplerUnits[ownIndex] : 0;
        if (unit < 0 || unit >= ContextState::kMaxTextureUnits || slot.textureTarget < 0) continue;
        Texture* bound = ctx_.state.textureUnits[unit].bound[slot.textureTarget].get();
        Texture* sampled = root(bound);
        if (!sampled || sampled->formatViews) continue;
        // The levels the draw samples (texture views: all of the root's); an
        // attachment of another level (a mip chain built a level at a time
        // from the one above) is no feedback.
        int first = 0, last = INT32_MAX;
        if (bound == sampled && sampled->mtlLevels > 0) {
            first = std::clamp(sampled->baseLevel, 0, sampled->mtlLevels - 1);
            last = std::max(first, std::min(sampled->maxLevel, sampled->mtlLevels - 1));
        }
        const auto feeds = [&](const Attachment& a) {
            Texture* t = a.texture.get();
            if (root(t) != sampled) return false;
            return t != sampled || (a.level >= first && a.level <= last);
        };
        bool attached = feeds(fb->depth) || feeds(fb->stencil);
        for (const Attachment& a : fb->color) attached = attached || feeds(a);
        if (!attached) continue;
        sampled->formatViews = sampled->storageDirty = true;
        if (sampled->mtl) ensureTextureStorage(ctx_, *sampled);
    }
}

void DrawState::samplerBinding(LinkedProgram& program, size_t index, __strong id<MTLTexture>& view,
                               __strong id<MTLSamplerState>& sampler, Texture** used) {
    if (used) *used = nullptr;
    const shader::SamplerSlotInfo& slot = program.link.samplers[index];
    const auto [owner, ownIndex] = samplerSource(program, index);
    const GLint unit = owner && ownIndex < owner->samplerUnits.size() ? owner->samplerUnits[ownIndex] : 0;
    Texture* texture = nullptr;
    if (unit >= 0 && unit < ContextState::kMaxTextureUnits && slot.textureTarget >= 0)
        texture = ctx_.state.textureUnits[unit].bound[slot.textureTarget].get();
    Sampler* samplerObject = unit >= 0 && unit < ContextState::kMaxTextureUnits
                                 ? ctx_.state.textureUnits[unit].sampler.get()
                                 : nullptr;
    if (texture && textureComplete(ctx_, *texture, samplerObject ? &samplerObject->params : &texture->sampler)) {
        SamplerParams params = samplerObject ? samplerObject->params : texture->sampler;
        // OpenGL ES: texelFetch decodes sRGB whatever GL_TEXTURE_SRGB_DECODE_EXT says.
        view = samplingView(ctx_, *texture, params.srgbDecode == GL_SKIP_DECODE_EXT && (slot.sampled || !ctx_.isEs()));
        params.lodBias += compat::textureUnitLodBias(ctx_, unit);  // glTexEnv(GL_TEXTURE_FILTER_CONTROL)
        sampler = samplerState(ctx_, *texture, params, slot.shadow);
        markSampled(*texture);
        if (used) *used = texture;
    } else {
        view = dummyTexture(slot.type);
        // Shadow lookups of an incomplete texture return 0 (the CTS's
        // incomplete_texture_access): a comparison that never passes.
        SamplerParams params;
        if (slot.shadow) {
            params.compareMode = GL_COMPARE_REF_TO_TEXTURE;
            params.compareFunc = GL_NEVER;
        }
        sampler = samplerState(ctx_, params, slot.shadow);
    }
}

uint16_t DrawState::borderVariant(LinkedProgram& program, int stage) {
    std::vector<shader::BorderSlot> slots;
    for (size_t i = 0; i < program.link.samplers.size(); ++i) {
        const shader::SamplerSlotInfo& slot = program.link.samplers[i];
        const int index = slot.stageTexture[stage];
        if (index < 0 || slot.textureTarget < 0) continue;
        const auto [owner, ownIndex] = samplerSource(program, i);
        const GLint unit = owner && ownIndex < owner->samplerUnits.size() ? owner->samplerUnits[ownIndex] : 0;
        if (unit < 0 || unit >= ContextState::kMaxTextureUnits) continue;
        Texture* texture = ctx_.state.textureUnits[unit].bound[slot.textureTarget].get();
        if (!texture) continue;
        const Sampler* samplerObject = ctx_.state.textureUnits[unit].sampler.get();
        SamplerParams params = samplerObject ? samplerObject->params : texture->sampler;
        params.lodBias += compat::textureUnitLodBias(ctx_, unit);  // as samplerBinding
        // Without sampler LOD bias in Metal (macOS 15), a biased lookup is
        // filtered in the shader.
        const bool bias = params.lodBias != 0.0f && !Device::get().caps().samplerLodBias;
        if (!bias && params.wrapS != GL_CLAMP_TO_BORDER && params.wrapT != GL_CLAMP_TO_BORDER &&
            params.wrapR != GL_CLAMP_TO_BORDER)
            continue;
        shader::BorderSlot border;
        if (!textureComplete(ctx_, *texture, &params) || !borderEmulation(*texture, params, slot.shadow, border, bias)) continue;
        border.texture = static_cast<uint32_t>(index);
        slots.push_back(border);
    }
    return slots.empty() ? 0
                         : shader::borderVariantId(shader::borderDefinitions(slots, stage == static_cast<int>(Stage::Fragment)));
}

void DrawState::prepareSamplerStorage(LinkedProgram& program) {
    // textureComplete builds pending storage with the blit encoder, which
    // would end the render or compute encoder of a draw in progress. Image
    // units (re)allocate writable storage the same way.
    preparedProgram_ = nullptr;
    preparedSamplers_.resize(program.link.samplers.size());
    for (size_t i = 0; i < program.link.samplers.size(); ++i)
        samplerBinding(program, i, preparedSamplers_[i].view, preparedSamplers_[i].sampler, &preparedSamplers_[i].texture);
    preparedProgram_ = &program;
    for (size_t i = 0; i < program.link.images.size(); ++i) imageBinding(program, i);
    prepareImageShadows(program);
}

void DrawState::prepareImageShadows(LinkedProgram& program) {
    // Bindings with a shadow texture (core::ImageShadow): the shadows take
    // their texels before the draw, once per draw (an earlier pass of the
    // draw may have written a shadow already).
    for (size_t i = 0; i < program.link.images.size(); ++i) {
        const GLint unit = imageUnit(program, i);
        if (unit < 0 || unit >= ContextState::kMaxImageUnits) continue;
        const ImageUnit& u = ctx_.state.imageUnits[unit];
        const shader::ImageSlotInfo& slot = program.link.images[i];
        PendingShadow entry;
        if (!imageView(ctx_, u, slot.type, slot.format, &entry.shadow) || !entry.shadow.shadow) continue;
        bool known = false;
        for (PendingShadow& other : imageShadows_)
            if (other.shadow.shadow == entry.shadow.shadow) {
                other.write = other.write || u.access != GL_READ_ONLY;
                known = true;
            }
        if (known) continue;
        entry.write = u.access != GL_READ_ONLY;
        copyImageShadow(ctx_, entry.shadow, true);
        imageShadows_.push_back(entry);
    }
}

void DrawState::finishImageShadows() {
    // The written shadows give their texels back to their textures.
    for (const PendingShadow& entry : imageShadows_)
        if (entry.write) copyImageShadow(ctx_, entry.shadow, false);
    imageShadows_.clear();
}

void DrawState::bindTextures(id<MTLRenderCommandEncoder> encoder, LinkedProgram& program, bool known) {
    EncoderState& e = encoderState_;
    if (!known)
        for (int s = 0; s < 2; ++s)
            for (int i = 0; i < kTrackedTextures; ++i) e.textures[s][i] = nil, e.samplers[s][i] = nil;
    // Each stage's texture and sampler (s: 0 vertex, 1 fragment), set only
    // when they differ from what the encoder has (untracked slots always).
    const auto setTexture = [&](int s, int index, id<MTLTexture> view) {
        if (index < kTrackedTextures) {
            if (known && e.textures[s][index] == view) return;
            e.textures[s][index] = view;
        }
        if (s == 0) [encoder setVertexTexture:view atIndex:static_cast<NSUInteger>(index)];
        else [encoder setFragmentTexture:view atIndex:static_cast<NSUInteger>(index)];
    };
    const auto setSampler = [&](int s, int index, id<MTLSamplerState> sampler) {
        if (index < kTrackedTextures) {
            if (known && e.samplers[s][index] == sampler) return;
            e.samplers[s][index] = sampler;
        }
        if (s == 0) [encoder setVertexSamplerState:sampler atIndex:static_cast<NSUInteger>(index)];
        else [encoder setFragmentSamplerState:sampler atIndex:static_cast<NSUInteger>(index)];
    };
    const bool prepared = preparedProgram_ == &program && preparedSamplers_.size() == program.link.samplers.size();
    for (size_t i = 0; i < program.link.samplers.size(); ++i) {
        const shader::SamplerSlotInfo& slot = program.link.samplers[i];
        id<MTLTexture> view = nil;
        id<MTLSamplerState> sampler = nil;
        if (prepared) {
            view = preparedSamplers_[i].view;
            sampler = preparedSamplers_[i].sampler;
            // The draw may be in a later command buffer than the preparation.
            if (Texture* texture = preparedSamplers_[i].texture) markSampled(*texture);
        } else {
            samplerBinding(program, i, view, sampler);
        }
        for (int s = 0; s < 2; ++s) {
            const int index = slot.stageTexture[s ? static_cast<int>(Stage::Fragment) : static_cast<int>(Stage::Vertex)];
            if (index < 0) continue;
            setTexture(s, index, view);
            setSampler(s, index, sampler);
        }
    }
    for (size_t i = 0; i < program.link.images.size(); ++i) {
        const shader::ImageSlotInfo& slot = program.link.images[i];
        const int vertex = slot.stageTexture[static_cast<int>(Stage::Vertex)];
        const int fragment = slot.stageTexture[static_cast<int>(Stage::Fragment)];
        if (vertex < 0 && fragment < 0) continue;
        id<MTLTexture> view = imageBinding(program, i);
        if (vertex >= 0) setTexture(0, vertex, view);
        if (fragment >= 0) setTexture(1, fragment, view);
    }
}

void DrawState::applyDynamicState(id<MTLRenderCommandEncoder> encoder, const RenderTargetDesc& target) {
    const ContextState& st = ctx_.state;
    // Every viewport of GL 4.1's array: shaders select one with gl_ViewportIndex.
    MTLViewport viewports[ContextState::kMaxViewports];
    for (int i = 0; i < ContextState::kMaxViewports; ++i) {
        viewports[i].originX = st.viewports[i][0];
        viewports[i].originY = st.viewports[i][1];
        viewports[i].width = std::max(0.0f, st.viewports[i][2]);
        viewports[i].height = std::max(0.0f, st.viewports[i][3]);
        viewports[i].znear = st.depthRanges[i][0];
        viewports[i].zfar = st.depthRanges[i][1];
    }
    // Only what changed since this path last set it on the encoder
    // (encoderStateValid_, bindDrawState).
    EncoderState& e = encoderState_;
    const bool all = !encoderStateValid_;
    if (all || std::memcmp(e.viewports, viewports, sizeof(viewports)) != 0) {
        [encoder setViewports:viewports count:ContextState::kMaxViewports];
        std::memcpy(e.viewports, viewports, sizeof(viewports));
    }
    // Cull / winding (the vertex stage flips Y, so GL's CCW is Metal's CW).
    MTLCullMode cull = MTLCullModeNone;
    if (st.cullFace) cull = st.cullFaceMode == GL_FRONT ? MTLCullModeFront : MTLCullModeBack;
    if (all || e.cull != cull) [encoder setCullMode:(e.cull = cull)];
    // With an UPPER_LEFT clip origin (GL_ARB_clip_control) Y is not flipped.
    const bool flipped = st.clipOrigin == GL_LOWER_LEFT;
    const MTLWinding winding = (st.frontFace == GL_CCW) == flipped ? MTLWindingClockwise : MTLWindingCounterClockwise;
    if (all || e.winding != winding) [encoder setFrontFacingWinding:(e.winding = winding)];
    const MTLTriangleFillMode fill = st.polygonMode == GL_LINE ? MTLTriangleFillModeLines : MTLTriangleFillModeFill;
    if (all || e.fill != fill) [encoder setTriangleFillMode:(e.fill = fill)];
    const bool offset = (st.polygonMode == GL_FILL && st.polygonOffsetFill) ||
                        (st.polygonMode == GL_LINE && st.polygonOffsetLine) ||
                        (st.polygonMode == GL_POINT && st.polygonOffsetPoint);
    const float bias[3] = {offset ? st.polygonOffsetUnits : 0.0f, offset ? st.polygonOffsetFactor : 0.0f,
                           offset ? st.polygonOffsetClamp : 0.0f};
    if (all || std::memcmp(e.bias, bias, sizeof(bias)) != 0) {
        [encoder setDepthBias:bias[0] slopeScale:bias[1] clamp:bias[2]];
        std::memcpy(e.bias, bias, sizeof(bias));
    }
    const MTLDepthClipMode clip = st.depthClamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip;
    if (all || e.clip != clip) [encoder setDepthClipMode:(e.clip = clip)];
    // The reference is clamped to the stencil range (GL 3.3 section 4.1.5).
    const uint32_t stencil[2] = {static_cast<uint32_t>(std::clamp(st.stencilFront.ref, 0, 255)),
                                 static_cast<uint32_t>(std::clamp(st.stencilBack.ref, 0, 255))};
    if (all || e.stencil[0] != stencil[0] || e.stencil[1] != stencil[1]) {
        [encoder setStencilFrontReferenceValue:stencil[0] backReferenceValue:stencil[1]];
        e.stencil[0] = stencil[0];
        e.stencil[1] = stencil[1];
    }
    if (all || std::memcmp(e.blend, st.blendColor, sizeof(e.blend)) != 0) {
        [encoder setBlendColorRed:st.blendColor[0] green:st.blendColor[1] blue:st.blendColor[2] alpha:st.blendColor[3]];
        std::memcpy(e.blend, st.blendColor, sizeof(e.blend));
    }
    (void)target;
}

// --------------------------------------------------------------- prepare

id<MTLRenderCommandEncoder> DrawState::prepare(GLenum mode, MTLPrimitiveType& primitive) {
    queriesPrepareVisibility(ctx_);
    RenderTargetDesc target;
    TargetFormats formats;
    if (validatedFor_ && validatedFor_ == drawCount_) {
        target = std::move(validatedTarget_);
        formats = validatedFormats_;
        validatedFor_ = 0;
    } else if (!drawTarget(target, formats)) {
        return nil;
    }
    const ContextState& st = ctx_.state;
    if (st.cullFace && st.cullFaceMode == GL_FRONT_AND_BACK && topologyClass(mode) == MTLPrimitiveTopologyClassTriangle)
        return nil;
    LinkedProgram* program = executable();
    if (!program) return nil;
    // Before the render encoder: completeness checks may build storage.
    const uint16_t borders[2] = {borderVariant(*program, static_cast<int>(Stage::Vertex)),
                                 borderVariant(*program, static_cast<int>(Stage::Fragment))};
    // Scissor rectangles of every viewport index (an empty one clips
    // everything). Without gl_ViewportIndex only index 0 is used, and an
    // empty rectangle there draws nothing.
    for (int i = 0; i < ContextState::kMaxViewports; ++i) {
        MTLScissorRect& scissor = lastScissors_[i];
        scissor = {0, 0, target.width, target.height};
        if (!((st.scissorTests >> i) & 1u)) continue;
        const GLint* box = st.scissors[i];
        const int64_t x0 = std::clamp<int64_t>(box[0], 0, target.width);
        const int64_t y0 = std::clamp<int64_t>(box[1], 0, target.height);
        const int64_t x1 = std::clamp<int64_t>(static_cast<int64_t>(box[0]) + std::max(0, box[2]), 0, target.width);
        const int64_t y1 = std::clamp<int64_t>(static_cast<int64_t>(box[1]) + std::max(0, box[3]), 0, target.height);
        if (x1 <= x0 || y1 <= y0) {
            if (i == 0 && !program->writesViewportIndex()) return nil;
            scissor = {0, 0, 0, 0};
            continue;
        }
        scissor = {static_cast<NSUInteger>(x0), static_cast<NSUInteger>(y0), static_cast<NSUInteger>(x1 - x0),
                   static_cast<NSUInteger>(y1 - y0)};
    }

    id<MTLRenderCommandEncoder> encoder = ctx_.stream().render(target);
    if (ctx_.stream().passGeneration() != lastPassGeneration_) {
        lastPassGeneration_ = ctx_.stream().passGeneration();
    }
    PipelineKey key;
    std::memset(&key, 0, sizeof(key));
    key.programId = program->uid;
    key.sampleCount = static_cast<uint8_t>(target.samples);
    key.alphaToCoverage = st.multisample && target.samples > 1 && st.sampleAlphaToCoverage;
    key.alphaToOne = st.multisample && target.samples > 1 && st.sampleAlphaToOne;
    // GL_SAMPLE_MASK and GL_SAMPLE_COVERAGE: the fragment function writes the
    // mask (a pipeline variant, so other draws keep Metal's fast paths).
    sampleMask_ = ~0u;
    const bool multisampled = target.samples > 1 || target.multisampled;
    if (st.multisample && multisampled && (st.sampleMask || st.sampleCoverage)) {
        uint32_t mask = st.sampleMask ? st.sampleMaskValue : ~0u;
        if (st.sampleCoverage) {
            const uint32_t all = target.samples >= 32 ? ~0u : (1u << target.samples) - 1;
            const long covered = std::lround(st.sampleCoverageValue * static_cast<float>(target.samples));
            uint32_t coverage = covered >= 32 ? ~0u : (1u << covered) - 1;
            if (st.sampleCoverageInvert) coverage = ~coverage & all;
            mask &= coverage;
        }
        if (mask != ~0u && program->fragmentSampleMask()) {
            key.sampleMask = 1;
            sampleMask_ = mask;
        }
    }
    // GL_SAMPLE_SHADING with more than one sample to shade: per-sample execution.
    key.sampleShading = st.multisample && st.sampleShading && target.samples > 1 &&
                        std::ceil(st.minSampleShading * static_cast<float>(target.samples)) > 1.0f;
    key.ignoreSampleMask = !(st.multisample && multisampled) && program->writesSampleMask();
    key.rasterization = !st.rasterizerDiscard;
    // Metal requires the input topology class when the vertex function
    // writes the layer or viewport index; otherwise it stays unspecified.
    key.topologyClass = program->writesLayerOrViewport() ? static_cast<uint8_t>(topologyClass(mode)) : 0;
    // Cull distances: the fragment function drops primitives with a distance
    // negative at every vertex.
    key.cull = static_cast<uint8_t>(program->cullDistances());
    key.borders[0] = borders[0];
    key.borders[1] = key.rasterization ? borders[1] : 0;
    // FRAGMENT_SHADER_INVOCATIONS: a fragment function variant counts them.
    key.countFragments = key.rasterization && queriesCounting(ctx_, kStatFragmentShaderInvocations);
    for (int i = 0; i < 8; ++i) key.colorFormats[i] = static_cast<uint16_t>(formats.color[i]);
    // GL_COLOR_LOGIC_OP replaces blending on unsigned normalized and integer
    // attachments; the fragment function applies the operation (framebuffer
    // fetch). GL_COPY needs no variant.
    if (st.colorLogicOp && st.logicOp != GL_COPY && program->fragmentVariant(false, st.logicOp - GL_CLEAR + 1, key.colorFormats))
        key.logicOp = static_cast<uint8_t>(st.logicOp - GL_CLEAR + 1);
    // KHR_blend_equation_advanced: the fragment function blends draw buffer
    // 0 (the only one drawn: validate()) with the attachment's value.
    const int advanced = (st.blendEnable & 1) && formats.color[0] && !formats.integer[0]
                             ? advancedBlendIndex(st.blend[0].equationRGB)
                             : -1;
    if (advanced >= 0 && program->fragmentVariant(false, static_cast<unsigned>(advanced + 17), key.colorFormats))
        key.logicOp = static_cast<uint8_t>(advanced + 17);
    for (int i = 0; i < 8; ++i) {
        if (!formats.color[i]) continue;
        PipelineKey::Blend& b = key.blend[i];
        const auto& mask = st.colorMask[i];
        b.writeMask = static_cast<uint8_t>((mask[0] ? MTLColorWriteMaskRed : 0) | (mask[1] ? MTLColorWriteMaskGreen : 0) |
                                           (mask[2] ? MTLColorWriteMaskBlue : 0) |
                                           (mask[3] && !formats.alphaIsOne[i] ? MTLColorWriteMaskAlpha : 0));
        b.enabled = (st.blendEnable >> i) & 1 && !formats.integer[i] && !(st.colorLogicOp && logicOpApplies(formats.color[i])) &&
                    !(i == 0 && advanced >= 0);
        if (b.enabled) {
            b.srcRGB = static_cast<uint8_t>(mtlBlendFactor(st.blend[i].srcRGB));
            b.dstRGB = static_cast<uint8_t>(mtlBlendFactor(st.blend[i].dstRGB));
            b.srcAlpha = static_cast<uint8_t>(mtlBlendFactor(st.blend[i].srcAlpha));
            b.dstAlpha = static_cast<uint8_t>(mtlBlendFactor(st.blend[i].dstAlpha));
            b.opRGB = static_cast<uint8_t>(mtlBlendOp(st.blend[i].equationRGB));
            b.opAlpha = static_cast<uint8_t>(mtlBlendOp(st.blend[i].equationAlpha));
        }
    }
    key.depthFormat = static_cast<uint16_t>(formats.depth);
    key.stencilFormat = static_cast<uint16_t>(formats.stencil);
    // Depth textures drawn with colour attachments (depth pre-passes, the
    // main view's depth) keep the full vertex function in their depth-only
    // passes too: the same geometry's depth must match exactly there. Others
    // (shadow maps) may take one that writes only the position.
    key.positionOnly = 0;
    if (target.depth.texture) {
        const void* depth = (__bridge const void*)target.depth.texture;
        bool color = false;
        for (int i = 0; i < 8; ++i) color |= formats.color[i] != 0;
        if (color) depthWithColor_.insert(depth);
        else key.positionOnly = key.rasterization && !depthWithColor_.count(depth);
    }
    (void)primitive;
    lastKey_ = key;
    lastFormats_ = formats;
    return encoder;
}

// ------------------------------------------------------------------ draws

namespace {

MTLPrimitiveType nativePrimitive(GLenum mode, bool& needsIndices) {
    needsIndices = false;
    switch (mode) {
    case GL_POINTS: return MTLPrimitiveTypePoint;
    case GL_LINES: return MTLPrimitiveTypeLine;
    case GL_LINE_STRIP: return MTLPrimitiveTypeLineStrip;
    case GL_TRIANGLES: return MTLPrimitiveTypeTriangle;
    case GL_TRIANGLE_STRIP: return MTLPrimitiveTypeTriangleStrip;
    case GL_LINE_LOOP: needsIndices = true; return MTLPrimitiveTypeLine;
    case GL_TRIANGLE_FAN: case GL_QUADS: case GL_QUAD_STRIP: case GL_POLYGON:
        needsIndices = true;
        return MTLPrimitiveTypeTriangle;
    case GL_LINES_ADJACENCY: case GL_LINE_STRIP_ADJACENCY:  // without a geometry shader: drop adjacency (W16)
        needsIndices = true;
        return MTLPrimitiveTypeLine;
    default:
        needsIndices = true;
        return MTLPrimitiveTypeTriangle;
    }
}

// The utility kernel that rewrites geometry-shader strips as lists in GL's
// last-vertex provoking order.
id<MTLComputePipelineState> stripsProvokingLastKernel() {
    static std::once_flag once;
    static id<MTLComputePipelineState> state = nil;
    std::call_once(once, [] {
        id<MTLFunction> function = [Device::get().utilityLibrary() newFunctionWithName:@"wmgl_strips_provoking_last"];
        NSError* error = nil;
        if (function) state = [Device::get().mtl() newComputePipelineStateWithFunction:function error:&error];
        if (!state) WMGL_ERR(Pipeline, "strip conversion kernel: %s",
                             error ? error.localizedDescription.UTF8String : "missing");
    });
    return state;
}

// The utility kernel that writes Metal tessellation factors from patch records.
id<MTLComputePipelineState> tessFactorsKernel() {
    static std::once_flag once;
    static id<MTLComputePipelineState> state = nil;
    std::call_once(once, [] {
        id<MTLFunction> function = [Device::get().utilityLibrary() newFunctionWithName:@"wmgl_tess_factors"];
        NSError* error = nil;
        if (function) state = [Device::get().mtl() newComputePipelineStateWithFunction:function error:&error];
        if (!state) WMGL_ERR(Pipeline, "tessellation factor kernel: %s",
                             error ? error.localizedDescription.UTF8String : "missing");
    });
    return state;
}

// Lists the primitives of a line or triangle stream with GL's last-vertex
// provoking vertex first: Metal flat-shades from the first vertex. Each
// triangle is rotated, so its winding is unchanged.
void expandProvokingLast(GLenum mode, const std::vector<uint32_t>& v, std::vector<uint32_t>& out) {
    const size_t n = v.size();
    out.clear();
    switch (mode) {
    case GL_LINES:
        for (size_t i = 0; i + 1 < n; i += 2) out.insert(out.end(), {v[i + 1], v[i]});
        return;
    case GL_LINE_STRIP: case GL_LINE_LOOP:
        for (size_t i = 0; i + 1 < n; ++i) out.insert(out.end(), {v[i + 1], v[i]});
        if (mode == GL_LINE_LOOP && n >= 2) out.insert(out.end(), {v[0], v[n - 1]});
        return;
    case GL_TRIANGLES:
        for (size_t i = 0; i + 2 < n; i += 3) out.insert(out.end(), {v[i + 2], v[i], v[i + 1]});
        return;
    case GL_TRIANGLE_STRIP:
        // Triangle i is (i, i+1, i+2) for even i and (i+1, i, i+2) for odd i.
        for (size_t i = 0; i + 2 < n; ++i) {
            if (i % 2 == 0) out.insert(out.end(), {v[i + 2], v[i], v[i + 1]});
            else out.insert(out.end(), {v[i + 2], v[i + 1], v[i]});
        }
        return;
    case GL_TRIANGLE_FAN:
        for (size_t i = 1; i + 1 < n; ++i) out.insert(out.end(), {v[i + 1], v[0], v[i]});
        return;
    case GL_QUADS:
        // Split along the 1-3 diagonal so both triangles end with the quad's last vertex.
        for (size_t i = 0; i + 3 < n; i += 4)
            out.insert(out.end(), {v[i + 3], v[i], v[i + 1], v[i + 3], v[i + 1], v[i + 2]});
        return;
    case GL_QUAD_STRIP:
        // Quad k is (2k, 2k+1, 2k+3, 2k+2) in order; its provoking vertex is 2k+3.
        for (size_t i = 0; i + 3 < n; i += 2)
            out.insert(out.end(), {v[i + 3], v[i], v[i + 1], v[i + 3], v[i + 2], v[i]});
        return;
    // Without a geometry shader the adjacent vertices are dropped; the
    // provoking vertices are those of GL 3.3 table 2.12.
    case GL_LINES_ADJACENCY:
        for (size_t i = 0; i + 3 < n; i += 4) out.insert(out.end(), {v[i + 2], v[i + 1]});
        return;
    case GL_LINE_STRIP_ADJACENCY:
        for (size_t i = 1; i + 2 < n; ++i) out.insert(out.end(), {v[i + 1], v[i]});
        return;
    case GL_TRIANGLES_ADJACENCY:
        for (size_t i = 0; i + 5 < n; i += 6) out.insert(out.end(), {v[i + 4], v[i], v[i + 2]});
        return;
    case GL_TRIANGLE_STRIP_ADJACENCY:
        for (size_t i = 0; i + 5 < n; i += 2) {
            if ((i / 2) % 2 == 0) out.insert(out.end(), {v[i + 4], v[i], v[i + 2]});
            else out.insert(out.end(), {v[i + 4], v[i + 2], v[i]});
        }
        return;
    default:
        return;
    }
}

// Expands a GL primitive stream (vertex numbers in `in`) into a Metal list.
void expandIndices(GLenum mode, const std::vector<uint32_t>& in, std::vector<uint32_t>& out) {
    const size_t n = in.size();
    out.clear();
    switch (mode) {
    case GL_LINE_LOOP:
        if (n < 2) return;
        for (size_t i = 0; i + 1 < n; ++i) { out.push_back(in[i]); out.push_back(in[i + 1]); }
        out.push_back(in[n - 1]);
        out.push_back(in[0]);
        return;
    case GL_TRIANGLE_FAN:
        // First-vertex convention: triangle (0, i, i+1) is provoked by vertex
        // i, listed first (Metal flat-shades from it); rotated, so the
        // winding is unchanged.
        for (size_t i = 1; i + 1 < n; ++i) out.insert(out.end(), {in[i], in[i + 1], in[0]});
        return;
    case GL_POLYGON:
        // A polygon is provoked by its first vertex in both conventions.
        for (size_t i = 1; i + 1 < n; ++i) out.insert(out.end(), {in[0], in[i], in[i + 1]});
        return;
    case GL_QUADS:
        for (size_t i = 0; i + 3 < n; i += 4) {
            out.insert(out.end(), {in[i], in[i + 1], in[i + 2], in[i], in[i + 2], in[i + 3]});
        }
        return;
    case GL_QUAD_STRIP:
        for (size_t i = 0; i + 3 < n; i += 2) {
            out.insert(out.end(), {in[i], in[i + 1], in[i + 3], in[i], in[i + 3], in[i + 2]});
        }
        return;
    case GL_LINES_ADJACENCY:
        for (size_t i = 0; i + 3 < n; i += 4) { out.push_back(in[i + 1]); out.push_back(in[i + 2]); }
        return;
    case GL_LINE_STRIP_ADJACENCY:
        for (size_t i = 1; i + 2 < n; ++i) { out.push_back(in[i]); out.push_back(in[i + 1]); }
        return;
    case GL_TRIANGLES_ADJACENCY:
        for (size_t i = 0; i + 5 < n; i += 6) { out.push_back(in[i]); out.push_back(in[i + 2]); out.push_back(in[i + 4]); }
        return;
    case GL_TRIANGLE_STRIP_ADJACENCY:
        for (size_t i = 0; i + 5 < n; i += 2) {
            if ((i / 2) % 2 == 0) { out.push_back(in[i]); out.push_back(in[i + 2]); out.push_back(in[i + 4]); }
            else { out.push_back(in[i + 2]); out.push_back(in[i]); out.push_back(in[i + 4]); }
        }
        return;
    default:
        out = in;
        return;
    }
}

// The independent-primitive mode a primitive stream is drawn as when listed.
GLenum listMode(GLenum mode) {
    switch (topologyClass(mode)) {
    case MTLPrimitiveTopologyClassPoint: return GL_POINTS;
    case MTLPrimitiveTopologyClassLine: return GL_LINES;
    default: return GL_TRIANGLES;
    }
}

// Appends one restart-free segment as a list of independent primitives, each
// listing its provoking vertex first (`rotate`: the last-vertex convention).
void appendMetalList(GLenum mode, const std::vector<uint32_t>& v, bool rotate, std::vector<uint32_t>& out) {
    std::vector<uint32_t> part;
    const size_t n = v.size();
    if (rotate) {
        expandProvokingLast(mode, v, part);
    } else {
        switch (mode) {
        case GL_POINTS: part = v; break;
        case GL_LINES: part.assign(v.begin(), v.begin() + static_cast<long>(n / 2 * 2)); break;
        case GL_LINE_STRIP:
            for (size_t i = 0; i + 1 < n; ++i) part.insert(part.end(), {v[i], v[i + 1]});
            break;
        case GL_TRIANGLES: part.assign(v.begin(), v.begin() + static_cast<long>(n / 3 * 3)); break;
        case GL_TRIANGLE_STRIP:
            // Triangle i is provoked by vertex i; odd triangles are rotated to list it first.
            for (size_t i = 0; i + 2 < n; ++i) {
                if (i % 2 == 0) part.insert(part.end(), {v[i], v[i + 1], v[i + 2]});
                else part.insert(part.end(), {v[i], v[i + 2], v[i + 1]});
            }
            break;
        default: expandIndices(mode, v, part); break;
        }
    }
    out.insert(out.end(), part.begin(), part.end());
}

}  // namespace

const DrawState::IndexPattern* DrawState::indexPattern(GLenum mode, bool rotate, GLsizei count) {
    constexpr GLsizei kMaxCount = 1 << 20;
    constexpr size_t kMaxPatterns = 256;
    if (count <= 0 || count > kMaxCount) return nullptr;
    const uint64_t key = uint64_t{mode} | (rotate ? 1ull << 16 : 0) | (static_cast<uint64_t>(count) << 17);
    auto it = indexPatterns_.find(key);
    if (it != indexPatterns_.end()) return &it->second;
    std::vector<uint32_t> in(static_cast<size_t>(count)), out;
    for (GLsizei i = 0; i < count; ++i) in[static_cast<size_t>(i)] = static_cast<uint32_t>(i);
    // As issue() rewrites them on the CPU.
    if (rotate) expandProvokingLast(mode, in, out);
    else expandIndices(mode, in, out);
    IndexPattern pattern;
    pattern.count = out.size();
    if (!out.empty()) {
        pattern.buffer = [Device::get().mtl() newBufferWithBytes:out.data() length:out.size() * 4
                                                         options:MTLResourceStorageModeShared];
        if (!pattern.buffer) return nullptr;
    }
    if (indexPatterns_.size() >= kMaxPatterns) indexPatterns_.clear();
    return &indexPatterns_.emplace(key, pattern).first->second;
}

namespace {
// A vertex start or base vertex every Metal device draws from
// (DeviceCaps::drawOffsets16).
bool drawOffsetFits(int64_t v) { return v >= -32768 && v <= 32767; }
}  // namespace

bool DrawState::issue(GLenum mode, GLint first, GLsizei count, GLsizei instances, const uint32_t* indexData,
                      id<MTLBuffer> indexBuffer, NSUInteger indexOffset, MTLIndexType indexType, GLint baseVertex,
                      GLint minVertex, GLsizei vertexSpan, bool listed, const IndirectSource* indirect) {
    queriesCountPrimitives(ctx_, mode, count, instances);
    stats::count(&stats::Frame::draws);
    LinkedProgram* current = executable();
    if (!current) return false;
    {
        stats::Timer timer(&stats::Frame::drawVertexNanos);
        waitForCpuVertexReads(*current, false);
    }
    {
        stats::Timer timer(&stats::Frame::drawSamplersNanos);
        prepareSamplerStorage(*current);
    }
    // GL_POINT polygon mode: Metal only fills or outlines triangles, so the
    // vertices of every triangle are drawn as points. Face culling is not
    // applied to them (FRONT_AND_BACK still draws nothing).
    const bool points = pointPolygons(mode);
    if (points && ctx_.state.cullFace && ctx_.state.cullFaceMode == GL_FRONT_AND_BACK) return true;
    MTLPrimitiveType primitive = MTLPrimitiveTypeTriangle;
    id<MTLRenderCommandEncoder> encoder = nil;
    {
        stats::Timer timer(&stats::Frame::drawPrepareNanos);
        encoder = prepare(points ? GL_POINTS : mode, primitive);
    }
    if (!encoder) return false;
    queriesBeforeDraw(ctx_, encoder, instances);
    LinkedProgram& program = *current;
    PipelineKey key = lastKey_;
    // Vertices are renumbered from the first one used, so converted vertex
    // data starts at the draw's first vertex instead of vertex 0; not when
    // the program reads gl_VertexID, which is the GL number.
    const GLint rebase = minVertex > 0 && !program.readsVertexId() ? minVertex : 0;
    vertexRebase_ = rebase;
    // Instanced attributes need data for every instance they cover.
    bool bound = false;
    {
        stats::Timer timer(&stats::Frame::drawVertexNanos);
        bound = bindVertexInputs(encoder, key, program, minVertex, vertexSpan, instances);
    }
    vertexRebase_ = 0;
    if (!bound) {
        WMGL_WARN(Draw, "too many vertex buffers for one draw");
        return false;
    }
    // Consecutive draws often have the same pipeline: compared before the cache's hash.
    id<MTLRenderPipelineState> pipeline = nil;
    stats::Timer phase(&stats::Frame::drawPipelineNanos);  // then bind, then encode
    if (lastPipeline_ && std::memcmp(&key, &lastPipelineKey_, sizeof(key)) == 0) {
        pipeline = lastPipeline_;
    } else {
        pipeline = ctx_.share().pipelines().render(key, program);
        lastPipelineKey_ = key;
        lastPipeline_ = pipeline;
    }
    if (!pipeline) return false;
    {
        // The pipeline the encoder has (encoderState_ is valid while only
        // this path drew since: bindDrawState's rule) is not set again.
        const uint64_t pass = ctx_.stream().passGeneration(), calls = ctx_.stream().renderCalls();
        const bool valid = encoderState_.pass == pass && (calls == encoderState_.calls || calls == encoderState_.calls + 1);
        if (!valid || encoderState_.pipeline != pipeline) [encoder setRenderPipelineState:pipeline];
        encoderState_.pipeline = pipeline;
    }
    phase.switchTo(&stats::Frame::drawBindNanos);
    bindDrawState(encoder, program);
    if (program.link.xfbVertexCapture) bindCaptureDefaults(encoder, program);
    phase.switchTo(&stats::Frame::drawEncodeNanos);

    bool needsIndices = false;
    MTLPrimitiveType native = nativePrimitive(mode, needsIndices);
    const bool rotate = !listed && !points && rotateProvoking(mode, program);
    if (rotate) native = topologyClass(mode) == MTLPrimitiveTopologyClassLine ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle;
    if (points) native = MTLPrimitiveTypePoint;
    const uint64_t serial = ctx_.stream().serial();
    if (indirect) {
        // The GL command structures are Metal's indirect arguments
        // (MTLDraw[Indexed]PrimitivesIndirectArguments), field for field;
        // a multi-draw's commands share everything bound above.
        stats::count(&stats::Frame::draws, indirect->count - 1);
        for (NSUInteger i = 0; i < indirect->count; ++i) {
            const NSUInteger at = indirect->offset + i * indirect->stride;
            if (indexBuffer)
                [encoder drawIndexedPrimitives:native indexType:indexType indexBuffer:indexBuffer indexBufferOffset:indexOffset
                                indirectBuffer:indirect->buffer indirectBufferOffset:at];
            else
                [encoder drawPrimitives:native indirectBuffer:indirect->buffer indirectBufferOffset:at];
            ctx_.stream().noteDraw();
        }
        return true;
    }
    // Metal on some virtual GPUs (DeviceCaps::drawOffsets16) keeps only 16
    // bits of the vertex start and base vertex: such a draw goes through
    // indices that hold the vertex numbers themselves, with no offset.
    const int64_t arrayStart = int64_t{first} - rebase, indexedBase = int64_t{baseVertex} - rebase;
    const bool limited = Device::get().caps().drawOffsets16;
    std::vector<uint32_t> readIndices;
    if (limited && indexBuffer && !drawOffsetFits(indexedBase) && count > 0) {
        // drawElements has read the element buffer on the CPU already.
        const uint8_t* at = static_cast<const uint8_t*>(indexBuffer.contents) + indexOffset;
        readIndices.resize(static_cast<size_t>(count));
        for (size_t i = 0; i < readIndices.size(); ++i)
            readIndices[i] = indexType == MTLIndexTypeUInt16 ? reinterpret_cast<const uint16_t*>(at)[i]
                                                             : reinterpret_cast<const uint32_t*>(at)[i];
        indexData = readIndices.data();
        indexBuffer = nil;
    }
    const bool offsetIndices = limited && (indexData ? !drawOffsetFits(indexedBase) : !indexBuffer && !drawOffsetFits(arrayStart));
    if (!indexData && !indexBuffer && !points && (needsIndices || rotate) && count > 0 && !offsetIndices) {
        // Array draws of fans, quads, loops or rotated primitives: the
        // rewritten numbering of 0..count-1, kept per mode and count, drawn
        // from the draw's first vertex.
        if (const IndexPattern* pattern = indexPattern(mode, rotate, count)) {
            if (pattern->count)
                [encoder drawIndexedPrimitives:native indexCount:pattern->count indexType:MTLIndexTypeUInt32
                                   indexBuffer:pattern->buffer indexBufferOffset:0
                                 instanceCount:std::max(1, instances) baseVertex:first - rebase
                                  baseInstance:baseInstance_];
            ctx_.stream().noteDraw();
            return true;
        }
    }
    if (needsIndices || rotate || points || (indexData && !indexBuffer) || offsetIndices) {
        stats::count(&stats::Frame::cpuIndexDraws);
        std::vector<uint32_t> in, out;
        in.reserve(static_cast<size_t>(count));
        const int64_t indexBias = offsetIndices ? indexedBase : 0;  // the base vertex, in the indices
        for (GLsizei i = 0; i < count; ++i)
            in.push_back(indexData ? static_cast<uint32_t>(indexData[i] + indexBias) : static_cast<uint32_t>(first + i - rebase));
        if (points) appendMetalList(mode, in, false, out);  // each triangle's vertices, shared ones once per triangle
        else if (rotate) expandProvokingLast(mode, in, out);
        else if (needsIndices) expandIndices(mode, in, out);
        else out.swap(in);
        if (out.empty()) return true;
        TransientAllocation alloc = transient_.allocate(out.size() * 4, serial, 4);
        std::memcpy(alloc.data, out.data(), out.size() * 4);
        [encoder drawIndexedPrimitives:native indexCount:out.size() indexType:MTLIndexTypeUInt32
                           indexBuffer:alloc.buffer indexBufferOffset:alloc.offset instanceCount:std::max(1, instances)
                            baseVertex:indexData && !offsetIndices ? baseVertex - rebase : 0 baseInstance:baseInstance_];
    } else if (indexBuffer) {
        [encoder drawIndexedPrimitives:native indexCount:static_cast<NSUInteger>(count) indexType:indexType
                           indexBuffer:indexBuffer indexBufferOffset:indexOffset instanceCount:std::max(1, instances)
                            baseVertex:baseVertex - rebase baseInstance:baseInstance_];
    } else {
        [encoder drawPrimitives:native vertexStart:static_cast<NSUInteger>(first - rebase)
                    vertexCount:static_cast<NSUInteger>(count) instanceCount:std::max(1, instances)
                   baseInstance:baseInstance_];
    }
    ctx_.stream().noteDraw();
    return true;
}

bool DrawState::noMetalRestartIndex(Buffer& buffer, GLenum type) {
    if (buffer.mapped || buffer.pinned || !buffer.mtl) return false;
    if (buffer.lastWriteSerial && !ctx_.stream().queue().isComplete(buffer.lastWriteSerial)) return false;
    const int wide = type == GL_UNSIGNED_INT ? 1 : 0;
    if (buffer.maxIndexScanVersion[wide] != buffer.version) {
        // In blocks without early exits, which the compiler vectorizes.
        const uint8_t* p = buffer.contents();
        const size_t size = wide ? 4 : 2, n = static_cast<size_t>(buffer.size) / size;
        bool found = false;
        for (size_t block = 0; block < n && !found; block += 4096) {
            const size_t end = std::min(n, block + 4096);
            if (wide) {
                for (size_t i = block; i < end; ++i) {
                    uint32_t v;
                    std::memcpy(&v, p + i * 4, 4);
                    found |= v == 0xffffffffu;
                }
            } else {
                for (size_t i = block; i < end; ++i) {
                    uint16_t v;
                    std::memcpy(&v, p + i * 2, 2);
                    found |= v == 0xffffu;
                }
            }
        }
        buffer.hasMaxIndex[wide] = found;
        buffer.maxIndexScanVersion[wide] = buffer.version;
    }
    return !buffer.hasMaxIndex[wide];
}

stats::IndirectCpu DrawState::indirectCpuReason(GLenum mode, GLenum type, LinkedProgram& program) {
    const ContextState& st = ctx_.state;
    if (mode == GL_PATCHES || program.link.geometry.active || ffWideLines_ || lineSides_ || linePartners_)
        return stats::kIndirectStage;
    if (st.transformFeedback->active) return stats::kIndirectCapture;
    if (queriesNeedDrawSizes(ctx_)) return stats::kIndirectQuery;
    // Drawn as Metal primitives, indices as they are.
    bool needsIndices = false;
    nativePrimitive(mode, needsIndices);
    if (needsIndices || pointPolygons(mode) || rotateProvoking(mode, program)) return stats::kIndirectPrimitive;
    // gl_BaseVertex and gl_BaseInstance are uniforms set from the command
    // (gl_InstanceID and gl_VertexID come from Metal; gl_DrawID is the caller's).
    if (Program* owner = stageProgram(program, static_cast<int>(Stage::Vertex)); owner && owner->linked) {
        const shader::LinkResult::BuiltinUniforms& b = owner->linked->link.builtinUniforms;
        if (b.baseVertex >= 0 || b.baseInstance >= 0) return stats::kIndirectParameters;
    }
    // Every attribute fetched by Metal: conversions cover the vertices drawn.
    VertexInputPlan plan;
    planVertexInputs(program, false, plan);
    for (const shader::StageOutput::VertexInput& input : program.stage(Stage::Vertex).vertexInputs)
        if (input.location >= 0 && input.location < VertexArray::kAttribSlots &&
            plan.how[static_cast<size_t>(input.location)] == VertexInputPlan::Converted)
            return stats::kIndirectConverted;
    if (type == GL_NONE) return stats::kIndirectCpuReasons;
    // Indices Metal reads as GL does: 16 or 32 bits, with no index where one
    // of them restarts and the other does not (noMetalRestartIndex).
    if (type != GL_UNSIGNED_SHORT && type != GL_UNSIGNED_INT) return stats::kIndirectIndexType;
    Buffer* elements = st.vertexArray->elementBuffer.get();
    if (!elements || elements->mappedNonPersistent()) return stats::kIndirectElementsMapped;
    const uint32_t largest = type == GL_UNSIGNED_SHORT ? 0xffffu : 0xffffffffu;
    if (st.primitiveRestart && !st.primitiveRestartFixedIndex && st.primitiveRestartIndex != largest)
        return stats::kIndirectRestart;
    if (noMetalRestartIndex(*elements, type)) return stats::kIndirectCpuReasons;
    if (elements->mapped || elements->pinned || !elements->mtl) return stats::kIndirectElementsMapped;
    return elements->lastWriteSerial && !ctx_.stream().queue().isComplete(elements->lastWriteSerial)
               ? stats::kIndirectElementsWriting
               : stats::kIndirectMaxIndex;
}

void DrawState::noteIndirectCpu(stats::IndirectCpu reason, GLenum mode, GLenum type, LinkedProgram& program) {
    const ContextState& st = ctx_.state;
    char detail[512];
    int n = std::snprintf(detail, sizeof(detail), "program %u (link %llu), mode 0x%x, type 0x%x: ",
                          st.program ? st.program->name : 0, static_cast<unsigned long long>(program.uid), mode, type);
    char* at = detail + std::clamp(n, 0, static_cast<int>(sizeof(detail)) - 1);
    const size_t left = sizeof(detail) - static_cast<size_t>(at - detail);
    std::string cause;  // one description per program and cause
    const Buffer* elements = st.vertexArray->elementBuffer.get();
    switch (reason) {
    case stats::kIndirectStage:
        std::snprintf(at, left, "%s", mode == GL_PATCHES ? "tessellation" : program.link.geometry.active ? "geometry shader" : "wide lines");
        break;
    case stats::kIndirectCapture: std::snprintf(at, left, "transform feedback"); break;
    case stats::kIndirectQuery: std::snprintf(at, left, "an active query counts primitives or samples"); break;
    case stats::kIndirectPrimitive: {
        bool needsIndices = false;
        nativePrimitive(mode, needsIndices);
        std::snprintf(at, left, "%s", needsIndices ? "primitive type" : pointPolygons(mode) ? "point polygon mode"
                                                                      : "flat inputs with the last-vertex convention");
        break;
    }
    case stats::kIndirectParameters: {
        const Program* owner = stageProgram(program, static_cast<int>(Stage::Vertex));
        const shader::LinkResult::BuiltinUniforms& b = owner->linked->link.builtinUniforms;
        std::snprintf(at, left, "reads%s%s", b.baseVertex >= 0 ? " gl_BaseVertex" : "", b.baseInstance >= 0 ? " gl_BaseInstance" : "");
        break;
    }
    case stats::kIndirectConverted: {
        VertexInputPlan plan;
        planVertexInputs(program, false, plan);
        for (const shader::StageOutput::VertexInput& input : program.stage(Stage::Vertex).vertexInputs) {
            if (input.location < 0 || input.location >= VertexArray::kAttribSlots ||
                plan.how[static_cast<size_t>(input.location)] != VertexInputPlan::Converted)
                continue;
            const VertexAttrib& a = attribFor(*st.vertexArray, input.source());
            const char* why = !a.buffer || !a.buffer->mtl                                        ? "client memory"
                              : vertexFetch(input, a).format == MTLVertexFormatInvalid             ? "its format"
                              : a.offset % 4 || a.effectiveStride() % 4 || !a.effectiveStride() ? "alignment"
                                                                                                  : "vertex buffer slots";
            std::snprintf(at, left, "attribute %d converted for %s (type 0x%x, size %d, offset %lld, stride %d)", input.location,
                          why, a.type, a.size, static_cast<long long>(a.offset), static_cast<int>(a.effectiveStride()));
            cause = std::to_string(input.location);
            break;
        }
        break;
    }
    case stats::kIndirectIndexType: std::snprintf(at, left, "8-bit indices"); break;
    case stats::kIndirectRestart: std::snprintf(at, left, "restart index 0x%x", st.primitiveRestartIndex); break;
    case stats::kIndirectElementsMapped: case stats::kIndirectElementsWriting: case stats::kIndirectMaxIndex:
        std::snprintf(at, left, "element buffer %u (size %lld, mapped %d, access 0x%x, pinned %d)", elements ? elements->name : 0,
                      elements ? static_cast<long long>(elements->size) : 0LL, elements && elements->mapped,
                      elements ? elements->mapAccess : 0, elements && elements->pinned);
        cause = std::to_string(elements ? elements->name : 0);
        break;
    default: break;
    }
    stats::indirectCpu(reason, std::to_string(program.uid) + ':' + cause, detail);
}

GLsizei DrawState::drawIndirect(GLenum mode, GLenum type, Buffer& commands, uintptr_t offset, GLsizei count,
                                uintptr_t stride) {
    ImageShadowScope shadows(*this);
    beginDraw(mode);
    if (!validateDraw(mode) || !conditionalRenderAllows(ctx_)) return count;  // as the CPU path would, for each command
    LinkedProgram* current = executable();
    if (!current || !commands.mtl) return 0;
    if (const stats::IndirectCpu reason = indirectCpuReason(mode, type, *current); reason != stats::kIndirectCpuReasons) {
        if (stats::g_enabled) noteIndirectCpu(reason, mode, type, *current);
        return 0;
    }
    // gl_DrawID is a uniform, set for each command.
    if (count > 1)
        if (Program* owner = stageProgram(*current, static_cast<int>(Stage::Vertex));
            owner && owner->linked && owner->linked->link.builtinUniforms.drawId >= 0)
            count = 1;
    stats::count(&stats::Frame::indirectGpu, static_cast<uint64_t>(count));
    if (count > 1) stats::count(&stats::Frame::indirectGpuBatches);
    Buffer* elements = type != GL_NONE ? ctx_.state.vertexArray->elementBuffer.get() : nullptr;
    const IndirectSource source{commands.mtl, static_cast<NSUInteger>(offset), static_cast<NSUInteger>(count),
                                static_cast<NSUInteger>(stride)};
    issue(mode, 0, 0, 1, nullptr, elements ? elements->mtl : nil, 0,
          type == GL_UNSIGNED_SHORT ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32, 0, 0, 0, false, &source);
    const uint64_t serial = ctx_.stream().serial();
    commands.lastUseSerial = serial;
    if (elements) elements->lastUseSerial = serial;
    return count;
}

namespace {
constexpr uint32_t kIdleCommitDraws = 32;
}  // namespace

void DrawState::commitLargeWork(const RenderTargetDesc* target) {
    static const uint32_t draws = [] {
        const char* v = std::getenv("WINEMETALGL_COMMIT_DRAWS");
        return v && *v ? static_cast<uint32_t>(std::strtoul(v, nullptr, 10)) : 256u;
    }();
    const auto micros = [](const char* name, uint64_t fallback) {
        const char* v = std::getenv(name);
        return (v && *v ? std::strtoull(v, nullptr, 10) : fallback) * 1000;
    };
    // A draw that starts a new pass commits the work recorded since the last
    // commit when that is this long ago: the pass ends anyway (no attachment
    // is stored and loaded for it), and the GPU starts on the work while the
    // CPU records the rest of the frame (a frame's many small passes would
    // otherwise reach the GPU in one command buffer at SwapBuffers).
    static const uint64_t passNanos = micros("WINEMETALGL_PASS_COMMIT_US", 1000);
    // After an indirect wait, how long since the last commit before an idle
    // GPU gets the draws recorded meanwhile (0: never). Off by default: with
    // the culling committed at its barrier (CommandStream::memoryBarrier),
    // the GPU idles little after a wait, and every such commit splits the
    // G-buffer pass (Foundation probe: 24.2 -> 21.1 ms a frame without).
    static const uint64_t eagerNanos = micros("WINEMETALGL_EAGER_US", 0);
    CommandStream& stream = ctx_.stream();
    const uint64_t recorded = stream.drawsSinceCommit();
    if (!draws || !recorded) return;
    if (target && passNanos && !stream.continuesPass(*target) && stats::nowNanos() - stream.lastCommitNanos() >= passNanos) {
        stream.flush(true);
        return;
    }
    // In frames where the CPU waited for indirect commands the GPU wrote
    // (Foundation's culling, issue #12), each wait leaves the GPU idle while
    // the CPU records the draws that follow: a few of them are worth
    // committing while it is, instead of at the next wait.
    const bool eager = eagerNanos && stream.indirectWaited();
    if (recorded < (eager ? std::min(draws, kIdleCommitDraws) : draws)) return;
    const bool idle = eager && stream.queue().idle() && stats::nowNanos() - stream.lastCommitNanos() >= eagerNanos;
    if (!idle && recorded < draws) return;
    // A continuing draw ends the pass early only when the GPU is idle.
    if (target && stream.continuesPass(*target) && recorded < uint64_t{draws} * 8 && !idle) return;
    stream.flush(true);
}

bool DrawState::pointPolygons(GLenum mode) const {
    const ContextState& st = ctx_.state;
    return topologyClass(mode) == MTLPrimitiveTopologyClassTriangle && st.polygonMode == GL_POINT &&
           st.polygonModeBack == GL_POINT;
}

bool DrawState::rotateProvoking(GLenum mode, LinkedProgram& program) const {
    if (ctx_.state.provokingVertex != GL_LAST_VERTEX_CONVENTION || !program.stage(Stage::Fragment).flatInputs) return false;
    // Flat inputs the vertex stage computes from per-instance or disabled
    // attributes and values constant over the draw are the same at every
    // vertex of a primitive (Foundation's material ids, issue #12).
    // WINEMETALGL_FLAT_ANALYSIS=0 rotates for every flat input.
    static const bool analysis = [] {
        const char* v = std::getenv("WINEMETALGL_FLAT_ANALYSIS");
        return !v || v[0] != '0';
    }();
    if (analysis && !program.pipeline && !program.link.flatPerVertex) {
        const VertexArray& vao = *ctx_.state.vertexArray;
        bool varies = false;
        for (uint32_t bits = program.link.flatAttribs; bits && !varies; bits &= bits - 1) {
            const VertexAttrib& a = vao.attribs[static_cast<size_t>(__builtin_ctz(bits))];
            varies = a.enabled && a.divisor == 0 && !a.zeroStride;
        }
        if (!varies) return false;
    }
    switch (mode) {
    case GL_LINES: case GL_LINE_STRIP: case GL_LINE_LOOP: case GL_TRIANGLES: case GL_TRIANGLE_STRIP: case GL_TRIANGLE_FAN:
    case GL_QUADS: case GL_QUAD_STRIP:  // a polygon's provoking vertex is its first in both conventions
    case GL_LINES_ADJACENCY: case GL_LINE_STRIP_ADJACENCY: case GL_TRIANGLES_ADJACENCY:
    case GL_TRIANGLE_STRIP_ADJACENCY:  // without a geometry shader (with one, drawGeometry)
        return true;
    default:
        return false;
    }
}

void DrawState::bindDrawState(id<MTLRenderCommandEncoder> encoder, LinkedProgram& program) {
    DepthStencilKey ds;
    std::memset(&ds, 0, sizeof(ds));
    const ContextState& st = ctx_.state;
    const RenderTargetDesc& target = ctx_.stream().currentTarget();
    const bool hasDepth = lastFormats_.depthAttached, hasStencil = lastFormats_.stencilAttached;
    ds.depthCompare = static_cast<uint8_t>(st.depthTest && hasDepth ? mtlCompare(st.depthFunc) : MTLCompareFunctionAlways);
    ds.depthWrite = st.depthTest && hasDepth && st.depthMask;
    ds.stencilEnabled = st.stencilTest && hasStencil;
    if (ds.stencilEnabled) {
        auto face = [](const StencilFace& f, DepthStencilKey::Face& out) {
            out.compare = static_cast<uint8_t>(mtlCompare(f.func));
            out.fail = static_cast<uint8_t>(mtlStencilOp(f.sfail));
            out.depthFail = static_cast<uint8_t>(mtlStencilOp(f.dpfail));
            out.pass = static_cast<uint8_t>(mtlStencilOp(f.dppass));
            out.readMask = f.valueMask & 0xff;
            out.writeMask = f.writeMask & 0xff;
        };
        // Front/back are GL's; the Y flip does not change which faces are front-facing.
        face(st.stencilFront, ds.front);
        face(st.stencilBack, ds.back);
    }
    // Valid while only this draw path used the encoder since it last set state.
    const uint64_t pass = ctx_.stream().passGeneration(), calls = ctx_.stream().renderCalls();
    encoderStateValid_ = encoderState_.pass == pass && (calls == encoderState_.calls || calls == encoderState_.calls + 1);
    encoderState_.pass = pass;
    encoderState_.calls = calls;
    // Draws mostly repeat the last depth/stencil state: compared before the cache's lock and hash.
    id<MTLDepthStencilState> depthStencil = nil;
    if (lastDepthStencil_ && std::memcmp(&ds, &lastDepthStencilKey_, sizeof(ds)) == 0) {
        depthStencil = lastDepthStencil_;
    } else {
        depthStencil = ctx_.share().pipelines().depthStencil(ds);
        lastDepthStencilKey_ = ds;
        lastDepthStencil_ = depthStencil;
    }
    if (!encoderStateValid_ || encoderState_.depthStencil != depthStencil) {
        [encoder setDepthStencilState:depthStencil];
        encoderState_.depthStencil = depthStencil;
    }
    applyDynamicState(encoder, target);
    if (!encoderStateValid_ || std::memcmp(encoderState_.scissors, lastScissors_, sizeof(lastScissors_)) != 0) {
        [encoder setScissorRects:lastScissors_ count:ContextState::kMaxViewports];
        std::memcpy(encoderState_.scissors, lastScissors_, sizeof(lastScissors_));
    }
    bindUniforms(encoder, program, encoderStateValid_);
    bindTextures(encoder, program, encoderStateValid_);
    id<MTLBuffer> counter = nil;
    NSUInteger counterOffset = 0;
    if (lastKey_.countFragments && queriesFragmentCounter(ctx_, counter, counterOffset))
        [encoder setFragmentBuffer:counter offset:counterOffset atIndex:shader::kSlotFragmentInvocations];
}

// ------------------------------------------------------ transform feedback

namespace {

// Assembles a primitive stream into independent primitives in capture order
// (strips and fans as GL 3.3 section 2.15.2 lists their vertices).
std::vector<uint32_t> assembleForCapture(GLenum mode, const std::vector<uint32_t>& v) {
    std::vector<uint32_t> out;
    const size_t n = v.size();
    switch (mode) {
    case GL_POINTS: out = v; break;
    case GL_LINES: out.assign(v.begin(), v.begin() + static_cast<long>(n / 2 * 2)); break;
    case GL_LINE_STRIP:
        for (size_t i = 0; i + 1 < n; ++i) { out.push_back(v[i]); out.push_back(v[i + 1]); }
        break;
    case GL_LINE_LOOP:
        for (size_t i = 0; i + 1 < n; ++i) { out.push_back(v[i]); out.push_back(v[i + 1]); }
        if (n >= 2) { out.push_back(v[n - 1]); out.push_back(v[0]); }
        break;
    case GL_TRIANGLES: out.assign(v.begin(), v.begin() + static_cast<long>(n / 3 * 3)); break;
    case GL_TRIANGLE_STRIP:
        for (size_t i = 0; i + 2 < n; ++i) {
            if (i % 2 == 0) { out.push_back(v[i]); out.push_back(v[i + 1]); }
            else { out.push_back(v[i + 1]); out.push_back(v[i]); }
            out.push_back(v[i + 2]);
        }
        break;
    case GL_TRIANGLE_FAN: case GL_POLYGON:
        for (size_t i = 1; i + 1 < n; ++i) { out.push_back(v[0]); out.push_back(v[i]); out.push_back(v[i + 1]); }
        break;
    case GL_QUADS:  // compatibility profile: each quad as two triangles
        for (size_t i = 0; i + 3 < n; i += 4) out.insert(out.end(), {v[i], v[i + 1], v[i + 2], v[i], v[i + 2], v[i + 3]});
        break;
    case GL_QUAD_STRIP:
        for (size_t i = 0; i + 3 < n; i += 2) out.insert(out.end(), {v[i], v[i + 1], v[i + 3], v[i], v[i + 3], v[i + 2]});
        break;
    default: break;
    }
    return out;
}

// The order that draws a captured list (assembleForCapture) with each
// primitive's provoking vertex (GL 4.6 table 13.2) first, as Metal
// flat-shades: a rotation of the primitive, which keeps its winding. Empty
// when the list is in that order; false when no rotation gives it (quads
// under the last-vertex convention, adjacency).
bool provokingOrder(GLenum mode, bool last, size_t count, std::vector<uint32_t>& order) {
    order.clear();
    size_t per = 3;
    int pattern = 0;  // the provoking vertex's place: 0, 1 or 2; 3: by parity (strips)
    switch (mode) {
    case GL_POINTS: case GL_POLYGON: return true;
    case GL_QUADS: return !last;
    case GL_LINES: case GL_LINE_STRIP: case GL_LINE_LOOP:
        per = 2;
        pattern = last ? 1 : 0;
        break;
    case GL_TRIANGLES: pattern = last ? 2 : 0; break;
    case GL_TRIANGLE_STRIP: pattern = last ? 2 : 3; break;  // first: v[i], second in odd triangles' list
    case GL_TRIANGLE_FAN: pattern = last ? 2 : 1; break;    // (v0, vi, vi+1): first convention vi
    case GL_QUAD_STRIP:
        if (!last) return true;
        pattern = 4;  // (vi, vi+1, vi+3), (vi, vi+3, vi+2): vi+3
        break;
    default: return false;
    }
    if (pattern == 0) return true;
    order.reserve(count);
    for (size_t k = 0; k + per <= count; k += per) {
        const size_t index = k / per;
        const size_t at = pattern == 3 ? index % 2 : pattern == 4 ? (index % 2 ? 1 : 2) : static_cast<size_t>(pattern);
        for (size_t j = 0; j < per; ++j) order.push_back(static_cast<uint32_t>(k + (at + j) % per));
    }
    return true;
}

}  // namespace

void DrawState::bindCaptureDefaults(id<MTLRenderCommandEncoder> encoder, LinkedProgram& program) {
    // Capture code is compiled into the vertex function; outside a capture
    // pass it sees an all-zero control block (capture inactive).
    static id<MTLBuffer> zero = [Device::get().mtl() newBufferWithLength:256 options:MTLResourceStorageModeShared];
    [encoder setVertexBuffer:zero offset:0 atIndex:shader::kSlotXfbControl];
    for (int b = 0; b < program.link.xfbBufferCount; ++b)
        [encoder setVertexBuffer:zero offset:0 atIndex:shader::kSlotFirstXfbBuffer + static_cast<NSUInteger>(b)];
}

bool DrawState::vertexWritesMemory(LinkedProgram& program) {
    const int vertex = static_cast<int>(Stage::Vertex);
    for (const shader::StageOutput::StorageArg& arg : program.link.stages[vertex].storageArgs)
        if (arg.writes) return true;
    for (const shader::ImageSlotInfo& image : program.link.images)
        if (image.stageTexture[vertex] >= 0) return true;
    return false;
}

DrawState::Capture DrawState::captureTransformFeedback(GLenum mode, const std::vector<uint32_t>& vertices, GLsizei instances) {
    TransformFeedback& xfb = *ctx_.state.transformFeedback;
    LinkedProgram* lp = executable();
    if (!xfb.active || xfb.paused || !lp || !lp->link.xfbVertexCapture) return kNotCaptured;
    const shader::LinkResult& link = lp->link;
    const uint32_t perPrimitive = xfb.primitiveMode == GL_POINTS ? 1 : xfb.primitiveMode == GL_LINES ? 2 : 3;
    const std::vector<uint32_t> list = assembleForCapture(mode, vertices);
    if (list.empty()) return kNotCaptured;
    // Whole primitives that fit in every bound buffer.
    const uint64_t requested = (list.size() / perPrimitive) * static_cast<uint64_t>(std::max(1, instances));
    uint64_t primitives = requested;
    for (int b = 0; b < link.xfbBufferCount; ++b) {
        if (!link.xfbBufferUsed(b)) continue;
        const TransformFeedback::Binding& binding = xfb.buffers[b];
        if (!binding.buffer || !binding.buffer->mtl) return kNotCaptured;
        // The bound range, clamped to the buffer's current size (it may have shrunk since binding).
        const int64_t available = std::max<int64_t>(0, binding.buffer->size - binding.offset);
        const int64_t size = binding.size ? std::min<int64_t>(binding.size, available) : available;
        const int64_t remaining = std::max<int64_t>(0, size - xfb.written[b]);
        const uint64_t vertexBytes = static_cast<uint64_t>(link.xfbStride[b]) * 4;
        if (vertexBytes) primitives = std::min<uint64_t>(primitives, static_cast<uint64_t>(remaining) / vertexBytes / perPrimitive);
    }
    if (Query* q = ctx_.state.activeQueries[kQueryXfbPrimitivesWritten].get()) q->result += primitives;
    if (primitives < requested) queriesNoteXfbOverflow(ctx_, 0);
    xfb.primitivesWritten[0] += primitives;
    if (!primitives) return kNotCaptured;

    id<MTLBuffer> buffers[4] = {};
    uint32_t baseWords[4] = {};
    for (int b = 0; b < link.xfbBufferCount; ++b) {
        if (!link.xfbBufferUsed(b)) continue;
        buffers[b] = xfb.buffers[b].buffer->mtl;
        baseWords[b] = static_cast<uint32_t>((xfb.buffers[b].offset + xfb.written[b]) / 4);
    }
    // A vertex stage that writes memory must run once per vertex: the draw
    // rasterizes the captured list (as independent primitives, each with its
    // provoking vertex first) itself, when nothing else rewrites it (polygon
    // points, wide lines, tessellation).
    const GLenum drawn = listMode(mode);
    std::vector<uint32_t> order;
    const bool flat = lp->stage(Stage::Fragment).flatInputs;
    const bool drawHere = !ctx_.state.rasterizerDiscard && vertexWritesMemory(*lp) && !tessInputs_.active && !ffWideLines_ &&
                          !pointPolygons(drawn) &&
                          (!flat || provokingOrder(mode, ctx_.state.provokingVertex == GL_LAST_VERTEX_CONVENTION, list.size(), order));
    bool rasterized = false;
    captureOrder_ = order.empty() ? nullptr : &order;
    const bool ran = runVertexCapture(*lp, list, instances, static_cast<uint32_t>(primitives * perPrimitive), link.xfbBufferCount,
                                      buffers, baseWords, link.xfbStride, drawHere ? drawn : GL_NONE, &rasterized);
    captureOrder_ = nullptr;
    if (!ran) return kNotCaptured;
    const uint64_t serial = ctx_.stream().serial();
    for (int b = 0; b < link.xfbBufferCount; ++b) {
        if (!link.xfbBufferUsed(b)) continue;
        Buffer& buffer = *xfb.buffers[b].buffer;
        xfb.written[b] += static_cast<GLsizeiptr>(primitives * perPrimitive * link.xfbStride[b] * 4);
        buffer.lastUseSerial = buffer.lastWriteSerial = serial;
        ++buffer.version;
    }
    return rasterized ? kCapturedAndDrawn : kCaptured;
}

bool DrawState::runVertexCapture(LinkedProgram& lp, const std::vector<uint32_t>& list, GLsizei instances, uint32_t limit,
                                 int bufferCount, const __strong id<MTLBuffer>* buffers, const uint32_t* baseWords,
                                 const uint32_t* strides, GLenum drawMode, bool* rasterized) {
    waitForCpuVertexReads(lp, true);
    prepareSamplerStorage(lp);
    id<MTLRenderCommandEncoder> encoder = nil;
    PipelineKey key;
    if (drawMode != GL_NONE) {
        // Rasterized with the current state (when it draws anything).
        MTLPrimitiveType primitive = MTLPrimitiveTypeTriangle;
        encoder = prepare(drawMode, primitive);
        if (encoder) queriesBeforeDraw(ctx_, encoder, instances);
        else drawMode = GL_NONE;
        key = lastKey_;
    }
    if (rasterized) *rasterized = drawMode != GL_NONE;
    if (drawMode == GL_NONE) {
        const uint16_t borders = borderVariant(lp, static_cast<int>(Stage::Vertex));
        std::memset(&key, 0, sizeof(key));
        key.programId = lp.uid;
        key.borders[0] = borders;
        key.rasterization = 0;
        key.topologyClass = static_cast<uint8_t>(MTLPrimitiveTopologyClassPoint);
        if (preludeCapture_) {
            // The prelude's pass has no attachments.
            encoder = ctx_.stream().preludeRender();
            key.sampleCount = 1;
        } else {
            RenderTargetDesc target;
            TargetFormats formats;
            if (!drawTarget(target, formats, false)) return false;
            encoder = ctx_.stream().render(target);
            key.sampleCount = static_cast<uint8_t>(target.samples);
            for (int i = 0; i < 8; ++i) key.colorFormats[i] = static_cast<uint16_t>(formats.color[i]);
            key.depthFormat = static_cast<uint16_t>(formats.depth);
            key.stencilFormat = static_cast<uint16_t>(formats.stencil);
        }
    }
    if (!bindVertexInputs(encoder, key, lp, 0, static_cast<GLsizei>(list.size()), instances, &list)) return false;
    id<MTLRenderPipelineState> pipeline = ctx_.share().pipelines().render(key, lp);
    if (!pipeline) return false;
    [encoder setRenderPipelineState:pipeline];
    encoderState_.pipeline = nil;  // issue() sets its own again
    if (drawMode != GL_NONE) {
        bindDrawState(encoder, lp);
    } else {
        bindUniforms(encoder, lp, false);
        bindTextures(encoder, lp, false);
    }
    const uint64_t serial = ctx_.stream().serial();
    // Control block: active, vertex count, slot limit, pad, base[4], stride[4], vertex ids.
    const size_t controlBytes = 12 * 4 + list.size() * 4;
    TransientAllocation control = transient_.allocate(controlBytes, serial, 16);
    auto* words = static_cast<uint32_t*>(control.data);
    std::memset(words, 0, 12 * 4);
    words[0] = 1;
    words[1] = static_cast<uint32_t>(list.size());
    words[2] = limit;
    for (int b = 0; b < bufferCount; ++b) {
        words[4 + b] = baseWords[b];
        words[8 + b] = strides[b];
        static id<MTLBuffer> unused = [Device::get().mtl() newBufferWithLength:16 options:MTLResourceStorageModePrivate];
        [encoder setVertexBuffer:buffers[b] ?: unused offset:0 atIndex:shader::kSlotFirstXfbBuffer + static_cast<NSUInteger>(b)];
    }
    std::memcpy(words + 12, list.data(), list.size() * 4);
    [encoder setVertexBuffer:control.buffer offset:control.offset atIndex:shader::kSlotXfbControl];
    const MTLPrimitiveType primitive = drawMode == GL_LINES       ? MTLPrimitiveTypeLine
                                       : drawMode == GL_TRIANGLES ? MTLPrimitiveTypeTriangle
                                                                  : MTLPrimitiveTypePoint;
    TransientAllocation order;
    if (captureOrder_ && drawMode != GL_NONE) {
        // Each list position once, so the vertex stage still runs once per vertex.
        order = transient_.allocate(captureOrder_->size() * 4, serial, 4);
        if (!order.buffer) return false;
        std::memcpy(order.data, captureOrder_->data(), captureOrder_->size() * 4);
    }
    if (order.buffer)
        [encoder drawIndexedPrimitives:primitive indexCount:captureOrder_->size() indexType:MTLIndexTypeUInt32
                           indexBuffer:order.buffer indexBufferOffset:order.offset
                         instanceCount:static_cast<NSUInteger>(std::max(1, instances)) baseVertex:0 baseInstance:baseInstance_];
    else
        [encoder drawPrimitives:primitive vertexStart:0 vertexCount:list.size()
                  instanceCount:static_cast<NSUInteger>(std::max(1, instances)) baseInstance:baseInstance_];
    // Later draws in this pass may read the captured data (as vertex,
    // uniform or texture buffers): Metal does not order work inside one
    // encoder, so order them after these writes. The prelude's captures are
    // read only by the next command buffer's kernels, and a barrier drains
    // the vertex stage each time.
    if (!preludeCapture_)
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers
                            afterStages:MTLRenderStageVertex
                           beforeStages:MTLRenderStageVertex | MTLRenderStageFragment];
    if (preludeCapture_) {
        // The bindings above went to the prelude's encoder, not the pass in
        // progress: its next draw sets all of its state again.
        encoderState_.pass = 0;
    } else {
        ctx_.stream().noteDraw();
    }
    return true;
}

// -------------------------------------------------------- geometry shaders

namespace {

// Lists the vertices of each geometry shader input primitive in gl_in order
// (GL 3.3 section 2.12.1, table 2.4 for triangle strips with adjacency).
std::vector<uint32_t> assembleGeometryInput(GLenum mode, const std::vector<uint32_t>& v) {
    const size_t n = v.size();
    std::vector<uint32_t> out;
    switch (mode) {
    case GL_LINES_ADJACENCY: out.assign(v.begin(), v.begin() + static_cast<long>(n / 4 * 4)); break;
    case GL_LINE_STRIP_ADJACENCY:
        for (size_t i = 0; i + 3 < n; ++i) out.insert(out.end(), {v[i], v[i + 1], v[i + 2], v[i + 3]});
        break;
    case GL_TRIANGLES_ADJACENCY: out.assign(v.begin(), v.begin() + static_cast<long>(n / 6 * 6)); break;
    case GL_TRIANGLE_STRIP_ADJACENCY: {
        if (n < 6) break;
        const size_t count = (n - 4) / 2;
        auto at = [&](size_t oneBased) { return v[oneBased - 1]; };
        for (size_t i = 0; i < count; ++i) {
            // Triangle vertices a, b, c and the vertices adjacent to edges ab, bc, ca (1-based).
            size_t a, b, c, ab, bc, ca;
            const bool odd = i % 2;
            if (count == 1) { a = 1; b = 3; c = 5; ab = 2; bc = 6; ca = 4; }
            else if (i == 0) { a = 1; b = 3; c = 5; ab = 2; bc = 7; ca = 4; }
            else if (i == count - 1 && odd) { a = 2 * i + 3; b = 2 * i + 1; c = 2 * i + 5; ab = 2 * i - 1; bc = 2 * i + 4; ca = 2 * i + 6; }
            else if (i == count - 1) { a = 2 * i + 1; b = 2 * i + 3; c = 2 * i + 5; ab = 2 * i - 1; bc = 2 * i + 6; ca = 2 * i + 4; }
            else if (odd) { a = 2 * i + 3; b = 2 * i + 1; c = 2 * i + 5; ab = 2 * i - 1; bc = 2 * i + 4; ca = 2 * i + 7; }
            else { a = 2 * i + 1; b = 2 * i + 3; c = 2 * i + 5; ab = 2 * i - 1; bc = 2 * i + 7; ca = 2 * i + 4; }
            out.insert(out.end(), {at(a), at(ab), at(b), at(bc), at(c), at(ca)});
        }
        break;
    }
    default: out = assembleForCapture(mode, v); break;
    }
    return out;
}

// The GS input primitive type a draw mode feeds (0: none).
GLenum geometryInputFor(GLenum mode) {
    switch (mode) {
    case GL_POINTS: return GL_POINTS;
    case GL_LINES: case GL_LINE_STRIP: case GL_LINE_LOOP: return GL_LINES;
    case GL_LINES_ADJACENCY: case GL_LINE_STRIP_ADJACENCY: return GL_LINES_ADJACENCY;
    case GL_TRIANGLES: case GL_TRIANGLE_STRIP: case GL_TRIANGLE_FAN: return GL_TRIANGLES;
    case GL_TRIANGLES_ADJACENCY: case GL_TRIANGLE_STRIP_ADJACENCY: return GL_TRIANGLES_ADJACENCY;
    default: return 0;
    }
}

}  // namespace

void DrawState::bindKernelResources(id<MTLComputeCommandEncoder> encoder, LinkedProgram& program,
                                    bool usesDefaultUniforms, int stage, const shader::StageOutput* kernel) {
    const int cs = stage;
    // The compute program, or (geometry and tessellation kernels) the vertex program.
    Program* owner = stageProgram(program, cs);
    if (!owner) owner = stageProgram(program, static_cast<int>(Stage::Vertex));
    if (!owner || !owner->linked) return;
    Program& p = *owner;
    applySubroutineSelections(ctx_, p);
    applyBuiltinUniforms(p);
    const uint32_t size = (p.linked->link.defaultBlockSize + 15) & ~15u;
    if (usesDefaultUniforms && size) {
        if (p.uniformData.size() < size) p.uniformData.resize(size, 0);
        if (size <= 4096) {
            [encoder setBytes:p.uniformData.data() length:size atIndex:shader::kSlotDefaultUniforms];
        } else {
            TransientAllocation alloc = transient_.allocate(size, ctx_.stream().serial(), 256);
            std::memcpy(alloc.data, p.uniformData.data(), size);
            [encoder setBuffer:alloc.buffer offset:alloc.offset atIndex:shader::kSlotDefaultUniforms];
        }
    }
    static id<MTLBuffer> zero = [Device::get().mtl() newBufferWithLength:65536 options:MTLResourceStorageModePrivate];
    for (size_t b = 0; b < program.link.blocks.size(); ++b) {
        const int slot = program.link.blocks[b].stageSlot[cs];
        if (slot < 0) continue;
        id<MTLBuffer> buffer = zero;
        NSUInteger offset = 0;
        uniformBlockBuffer(program, b, buffer, offset);
        [encoder setBuffer:buffer offset:offset atIndex:static_cast<NSUInteger>(slot)];
    }
    for (size_t i = 0; i < program.link.samplers.size(); ++i) {
        const int index = program.link.samplers[i].stageTexture[cs];
        if (index < 0) continue;
        id<MTLTexture> view = nil;
        id<MTLSamplerState> sampler = nil;
        samplerBinding(program, i, view, sampler);
        [encoder setTexture:view atIndex:static_cast<NSUInteger>(index)];
        [encoder setSamplerState:sampler atIndex:static_cast<NSUInteger>(index)];
    }
    for (size_t b = 0; b < program.link.atomicBuffers.size(); ++b) {
        const int slot = program.link.atomicBuffers[b].stageSlot[cs];
        if (slot < 0) continue;
        id<MTLBuffer> buffer = nil;
        NSUInteger offset = 0;
        atomicCounterBuffer(program, b, buffer, offset);
        [encoder setBuffer:buffer offset:offset atIndex:static_cast<NSUInteger>(slot)];
    }
    // The kernel's storage blocks: a compute program's own, or the geometry
    // or tessellation control emulation kernel's (they index the program's).
    if (kernel) bindStorageBlocks(encoder, *kernel, owner);
    for (size_t i = 0; i < program.link.images.size(); ++i) {
        const int index = program.link.images[i].stageTexture[cs];
        if (index >= 0) [encoder setTexture:imageBinding(program, i) atIndex:static_cast<NSUInteger>(index)];
    }
}

void DrawState::dispatch(GLuint x, GLuint y, GLuint z, Buffer* indirect, GLintptr offset) {
    commitLargeWork(nullptr);
    // Conditional rendering applies to dispatches too (KHR-GL43.compute_shader.conditional-dispatching).
    if (!conditionalRenderAllows(ctx_)) return;
    ImageShadowScope shadows(*this);
    LinkedProgram* current = executable();
    if (!current) return;
    LinkedProgram& lp = *current;
    // The GPU reads the group counts of an indirect dispatch, after the
    // commands before it (serial compute encoder, tracked buffers): only a
    // statistics query needs them on the CPU, after their writes complete.
    const bool countInvocations = queriesCounting(ctx_, kStatComputeShaderInvocations);
    if (indirect && countInvocations && indirect->lastWriteSerial) ctx_.stream().waitFor(indirect->lastWriteSerial);
    prepareSamplerStorage(lp);
    const int cs = static_cast<int>(Stage::Compute);
    const uint16_t borders = borderVariant(lp, cs);
    id<MTLComputePipelineState> pipeline = borders ? lp.kernelWithBorders(lp.stage(Stage::Compute), cs, borders) : lp.computePipeline();
    if (!pipeline) return;
    // A kernel without textures or images touches no attachment of a pending
    // clear (Foundation's culling, between its G-buffer clear and draws).
    id<MTLComputeCommandEncoder> encoder = ctx_.stream().compute(lp.link.samplers.empty() && lp.link.images.empty());
    [encoder setComputePipelineState:pipeline];
    bindKernelResources(encoder, lp, lp.stage(Stage::Compute).usesDefaultUniforms, static_cast<int>(Stage::Compute),
                        &lp.stage(Stage::Compute));
    const uint32_t* local = lp.link.computeLocalSize;
    const MTLSize threads = MTLSizeMake(local[0], local[1], local[2]);
    if (countInvocations) {
        uint32_t groups[3] = {x, y, z};
        if (indirect) std::memcpy(groups, indirect->contents() + offset, sizeof(groups));  // written by the GPU before the wait above
        queriesCountStatistic(ctx_, kStatComputeShaderInvocations,
                              uint64_t{groups[0]} * groups[1] * groups[2] * local[0] * local[1] * local[2]);
    }
    if (indirect) {
        indirect->lastUseSerial = ctx_.stream().serial();
        [encoder dispatchThreadgroupsWithIndirectBuffer:indirect->mtl indirectBufferOffset:static_cast<NSUInteger>(offset)
                                  threadsPerThreadgroup:threads];
    } else {
        [encoder dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:threads];
    }
}

void DrawState::drawWideLines(GLenum mode, const std::vector<uint32_t>& vertices, GLsizei instances) {
    const size_t n = vertices.size();
    std::vector<std::pair<uint32_t, uint32_t>> segments;
    if (mode == GL_LINES) {
        for (size_t i = 0; i + 1 < n; i += 2) segments.emplace_back(vertices[i], vertices[i + 1]);
    } else {
        for (size_t i = 0; i + 1 < n; ++i) segments.emplace_back(vertices[i], vertices[i + 1]);
        if (mode == GL_LINE_LOOP && n >= 2) segments.emplace_back(vertices[n - 1], vertices[0]);
    }
    queriesCountPrimitives(ctx_, mode, static_cast<GLsizei>(n), instances);
    if (segments.empty()) return;
    // Two triangles per segment, each listing the provoking endpoint first
    // (Metal flat-shades from a triangle's first vertex). The program offsets
    // every vertex by half the width to the side given (fixed_function.mm).
    const bool last = ctx_.state.provokingVertex == GL_LAST_VERTEX_CONVENTION;
    std::vector<uint32_t> list, partners;
    std::vector<float> sides;
    for (const auto& [a, b] : segments) {
        const uint32_t p = last ? b : a, q = last ? a : b;
        list.insert(list.end(), {p, p, q, p, q, q});
        partners.insert(partners.end(), {q, q, p, q, p, p});
        sides.insert(sides.end(), {-1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f});
    }
    LinkedProgram* current = executable();
    if (!current) return;
    waitForCpuVertexReads(*current, true);
    prepareSamplerStorage(*current);
    // Lines are not culled and ignore the polygon mode; their quads take the line offset state.
    ContextState& st = ctx_.state;
    const bool cull = st.cullFace, offsetFill = st.polygonOffsetFill;
    const GLenum polygonMode = st.polygonMode, polygonModeBack = st.polygonModeBack;
    st.cullFace = false;
    st.polygonMode = st.polygonModeBack = GL_FILL;
    st.polygonOffsetFill = st.polygonOffsetLine;
    ctx_.markDirty(kDirtyRaster);
    MTLPrimitiveType primitive = MTLPrimitiveTypeTriangle;
    if (id<MTLRenderCommandEncoder> encoder = prepare(GL_TRIANGLES, primitive)) {
        queriesBeforeDraw(ctx_, encoder, instances);
        PipelineKey key = lastKey_;
        linePartners_ = &partners;
        lineSides_ = &sides;
        const bool bound = bindVertexInputs(encoder, key, *current, 0, static_cast<GLsizei>(list.size()), instances, &list);
        linePartners_ = nullptr;
        lineSides_ = nullptr;
        if (id<MTLRenderPipelineState> pipeline = bound ? ctx_.share().pipelines().render(key, *current) : nil) {
            stats::count(&stats::Frame::draws);
            [encoder setRenderPipelineState:pipeline];
            encoderState_.pipeline = nil;  // issue() sets its own again
            bindDrawState(encoder, *current);
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:list.size()
                      instanceCount:static_cast<NSUInteger>(std::max(1, instances)) baseInstance:baseInstance_];
            ctx_.stream().noteDraw();
        }
    }
    st.cullFace = cull;
    st.polygonOffsetFill = offsetFill;
    st.polygonMode = polygonMode;
    st.polygonModeBack = polygonModeBack;
    ctx_.markDirty(kDirtyRaster);
}

bool DrawState::clientVertexArrays(LinkedProgram& program) const {
    const VertexArray& vao = *ctx_.state.vertexArray;
    bool any = false;
    for (const shader::StageOutput::VertexInput& input : program.stage(Stage::Vertex).vertexInputs) {
        if (input.location < 0 || input.location >= VertexArray::kAttribSlots) continue;
        const VertexAttrib& a = attribFor(vao, input.source());
        if (!a.enabled || a.divisor) continue;
        if (a.buffer) return false;
        any = true;
    }
    return any;
}

void DrawState::drawGathered(const std::vector<uint32_t>& list, GLsizei instances) {
    queriesCountPrimitives(ctx_, GL_TRIANGLES, static_cast<GLsizei>(list.size()), instances);
    LinkedProgram* current = executable();
    if (!current || list.empty()) return;
    waitForCpuVertexReads(*current, true);
    prepareSamplerStorage(*current);
    MTLPrimitiveType primitive = MTLPrimitiveTypeTriangle;
    id<MTLRenderCommandEncoder> encoder = prepare(GL_TRIANGLES, primitive);
    if (!encoder) return;
    queriesBeforeDraw(ctx_, encoder, instances);
    PipelineKey key = lastKey_;
    if (!bindVertexInputs(encoder, key, *current, 0, static_cast<GLsizei>(list.size()), instances, &list)) return;
    id<MTLRenderPipelineState> pipeline = ctx_.share().pipelines().render(key, *current);
    if (!pipeline) return;
    stats::count(&stats::Frame::draws);
    [encoder setRenderPipelineState:pipeline];
    encoderState_.pipeline = nil;  // issue() sets its own again
    bindDrawState(encoder, *current);
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:list.size()
              instanceCount:static_cast<NSUInteger>(std::max(1, instances)) baseInstance:baseInstance_];
    ctx_.stream().noteDraw();
}

void DrawState::drawGeometry(GLenum mode, const std::vector<uint32_t>& vertices, GLsizei instances, uint32_t& primitiveBase,
                             bool assembled) {
    LinkedProgram& lp = *executable();
    const shader::GeometryEmulation& g = lp.link.geometry;
    const std::vector<uint32_t> list = assembled ? vertices : assembleGeometryInput(mode, vertices);
    const uint32_t primitives = static_cast<uint32_t>(list.size() / static_cast<size_t>(g.verticesIn));
    if (!primitives) return;
    const uint32_t copies = static_cast<uint32_t>(std::max(1, instances));
    const uint64_t total = static_cast<uint64_t>(primitives) * copies;
    // Kernel threads: one per input primitive and invocation (GS instancing).
    const uint64_t threads = total * static_cast<uint64_t>(std::max(1, g.invocations));
    queriesCountStatistic(ctx_, kStatGeometryShaderInvocations, threads);
    if (g.maxVertices == 0) return;  // layout(max_vertices = 0): nothing is emitted
    const uint32_t maxV = static_cast<uint32_t>(g.maxVertices);
    const uint16_t borders = borderVariant(lp, static_cast<int>(Stage::Compute));  // the kernel's samplers
    id<MTLComputePipelineState> kernel =
        borders ? lp.kernelWithBorders(g.kernel, static_cast<int>(Stage::Geometry), borders) : lp.geometryKernel();
    if (!kernel || !lp.geometryPassthrough()) return;
    // Anything that flushes must happen before the transient allocations below.
    waitForCpuVertexReads(lp, true);
    prepareSamplerStorage(lp);
    queriesPrepareVisibility(ctx_);
    const uint64_t serial = ctx_.stream().serial();

    // 1. Vertex outputs of every input primitive, in primitive order.
    TransientAllocation input = transient_.allocate(total * g.verticesIn * g.inputStride * 4, serial, 16);
    if (!input.buffer) return;  // too large for the device (logged)
    const id<MTLBuffer> inputBuffers[1] = {input.buffer};
    const uint32_t inputBase[1] = {static_cast<uint32_t>(input.offset / 4)};
    const uint32_t inputStride[1] = {g.inputStride};
    if (!runVertexCapture(lp, list, instances, static_cast<uint32_t>(list.size() * copies), 1, inputBuffers, inputBase,
                          inputStride))
        return;

    // 2. The geometry shader: one thread per input primitive.
    const size_t indexCount = threads * 2 * maxV;
    TransientAllocation output = transient_.allocate(threads * maxV * g.outputStride * 4, serial, 16);
    TransientAllocation indices = transient_.allocate(indexCount * 4, serial, 16);
    // The kernel's parameters: a buffer, not setBytes, since the generated
    // kernel declares the slot as device memory (Metal validation rejects
    // read-only bytes there).
    TransientAllocation params = transient_.allocate(16, serial, 16);
    if (!output.buffer || !indices.buffer || !params.buffer) return;
    id<MTLComputeCommandEncoder> compute = ctx_.stream().compute();
    [compute setComputePipelineState:kernel];
    bindKernelResources(compute, lp, g.kernel.usesDefaultUniforms, static_cast<int>(Stage::Compute), &g.kernel);
    const uint32_t control[4] = {static_cast<uint32_t>(total), primitives, primitiveBase, 0};
    [compute setBuffer:input.buffer offset:input.offset atIndex:shader::kSlotFirstXfbBuffer];
    [compute setBuffer:output.buffer offset:output.offset atIndex:shader::kSlotFirstXfbBuffer + 1];
    [compute setBuffer:indices.buffer offset:indices.offset atIndex:shader::kSlotFirstXfbBuffer + 2];
    std::memcpy(params.data, control, sizeof(control));
    [compute setBuffer:params.buffer offset:params.offset atIndex:shader::kSlotFirstXfbBuffer + 3];
    const NSUInteger width = std::min<NSUInteger>(64, kernel.maxTotalThreadsPerThreadgroup);
    [compute dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    primitiveBase += primitives;

    // Flat-shaded strips with GL's last-vertex convention are drawn as lists
    // whose primitives list their provoking vertex first.
    TransientAllocation lastList;
    size_t lastListCount = 0;
    if (g.outputType != GL_POINTS && !ctx_.state.rasterizerDiscard &&
        ctx_.state.provokingVertex == GL_LAST_VERTEX_CONVENTION && lp.stage(Stage::Fragment).flatInputs) {
        if (id<MTLComputePipelineState> rotate = stripsProvokingLastKernel()) {
            const uint32_t perThread = (g.outputType == GL_LINE_STRIP ? 2u : 3u) * maxV;
            lastListCount = static_cast<size_t>(threads) * perThread;
            lastList = transient_.allocate(lastListCount * 4, serial, 16);
            if (!lastList.buffer) return;
            const uint32_t params[4] = {2 * maxV, perThread, g.outputType == GL_LINE_STRIP ? 0u : 1u,
                                        static_cast<uint32_t>(threads)};
            [compute setComputePipelineState:rotate];
            [compute setBuffer:indices.buffer offset:indices.offset atIndex:0];
            [compute setBuffer:lastList.buffer offset:lastList.offset atIndex:1];
            [compute setBytes:params length:sizeof(params) atIndex:2];
            const NSUInteger rotateWidth = std::min<NSUInteger>(64, rotate.maxTotalThreadsPerThreadgroup);
            [compute dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(rotateWidth, 1, 1)];
        }
    }

    // 3. Draw the emitted vertices.
    if (!ctx_.state.rasterizerDiscard) {
        MTLPrimitiveType primitive = MTLPrimitiveTypeTriangle;
        id<MTLRenderCommandEncoder> encoder = prepare(g.outputType, primitive);
        if (encoder) {
            queriesBeforeDraw(ctx_, encoder, instances);
            PipelineKey key = lastKey_;
            key.geometryPass = 1;
            id<MTLRenderPipelineState> pipeline = ctx_.share().pipelines().render(key, lp);
            if (pipeline) {
                [encoder setRenderPipelineState:pipeline];
                encoderState_.pipeline = nil;  // issue() sets its own again
                bindDrawState(encoder, lp);
                [encoder setVertexBuffer:output.buffer offset:output.offset atIndex:shader::kSlotFirstXfbBuffer];
                if (g.outputType == GL_POINTS)
                    [encoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:threads * maxV];
                else if (lastList.buffer)
                    [encoder drawIndexedPrimitives:g.outputType == GL_LINE_STRIP ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle
                                        indexCount:lastListCount
                                         indexType:MTLIndexTypeUInt32
                                       indexBuffer:lastList.buffer
                                 indexBufferOffset:lastList.offset];
                else
                    [encoder drawIndexedPrimitives:g.outputType == GL_LINE_STRIP ? MTLPrimitiveTypeLineStrip
                                                                                  : MTLPrimitiveTypeTriangleStrip
                                        indexCount:indexCount
                                         indexType:MTLIndexTypeUInt32
                                       indexBuffer:indices.buffer
                                 indexBufferOffset:indices.offset];
                ctx_.stream().noteDraw();
            }
        }
    }

    // 4. Primitive queries and transform feedback need the emitted
    // primitives on the CPU, by vertex stream.
    TransformFeedback& xfb = *ctx_.state.transformFeedback;
    const bool capture = xfb.active && !xfb.paused && !lp.link.transformFeedback.empty();
    Query* generated[kMaxVertexStreams] = {};
    Query* writtenQuery[kMaxVertexStreams] = {};
    bool anyQuery = false;
    for (GLuint s = 0; s < static_cast<GLuint>(kMaxVertexStreams); ++s) {
        generated[s] = ctx_.state.activeQueries[queryTargetIndex(ctx_, GL_PRIMITIVES_GENERATED, s)].get();
        writtenQuery[s] = ctx_.state.activeQueries[queryTargetIndex(ctx_, GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, s)].get();
        anyQuery |= generated[s] || writtenQuery[s];
    }
    const bool statistics = queriesCounting(ctx_, kStatGeometryShaderPrimitivesEmitted) ||
                            queriesCounting(ctx_, kStatClippingInputPrimitives) ||
                            queriesCounting(ctx_, kStatClippingOutputPrimitives);
    anyQuery |= statistics;
    if (!capture && !anyQuery) return;
    ctx_.stream().waitFor(serial);
    const auto* slots = static_cast<const uint32_t*>(indices.data);
    const auto* records = static_cast<const uint32_t*>(output.data);
    // Record slots of independent primitives, in order, per stream (only
    // points output has streams other than 0).
    std::vector<uint32_t> emitted[kMaxVertexStreams];
    std::vector<uint32_t> strip;
    for (uint64_t t = 0; t < threads; ++t) {
        for (uint32_t k = 0; k <= 2 * maxV; ++k) {
            const uint32_t slot = k < 2 * maxV ? slots[t * 2 * maxV + k] : 0xffffffffu;
            if (slot != 0xffffffffu) {
                strip.push_back(slot);
                if (g.outputType != GL_POINTS) continue;
            }
            if (strip.empty()) continue;
            if (g.outputType == GL_POINTS) {
                const uint32_t stream = g.streams ? records[static_cast<uint64_t>(strip[0]) * g.outputStride + g.streamOffset] : 0;
                if (stream < static_cast<uint32_t>(kMaxVertexStreams)) emitted[stream].push_back(strip[0]);
            } else {
                const std::vector<uint32_t> assembled =
                    assembleForCapture(g.outputType == GL_LINE_STRIP ? GL_LINE_STRIP : GL_TRIANGLE_STRIP, strip);
                emitted[0].insert(emitted[0].end(), assembled.begin(), assembled.end());
            }
            strip.clear();
        }
    }
    const uint32_t perPrimitive = g.outputType == GL_POINTS ? 1 : g.outputType == GL_LINE_STRIP ? 2 : 3;
    for (int s = 0; s < kMaxVertexStreams; ++s) {
        if (generated[s]) generated[s]->result += emitted[s].size() / perPrimitive;
        queriesCountStatistic(ctx_, kStatGeometryShaderPrimitivesEmitted, emitted[s].size() / perPrimitive);
    }
    // Stream 0 is the one rasterized.
    queriesCountStatistic(ctx_, kStatClippingInputPrimitives, emitted[0].size() / perPrimitive);
    queriesCountStatistic(ctx_, kStatClippingOutputPrimitives, emitted[0].size() / perPrimitive);
    if (!capture) return;
    const shader::LinkResult& link = lp.link;
    // Primitives each stream writes: what fits in every buffer it feeds;
    // a stream without buffers writes none.
    uint64_t written[kMaxVertexStreams] = {};
    bool captured[kMaxVertexStreams] = {};
    for (int s = 0; s < kMaxVertexStreams; ++s) written[s] = emitted[s].size() / perPrimitive;
    for (int b = 0; b < link.xfbBufferCount; ++b) {
        if (!link.xfbBufferUsed(b)) continue;
        const TransformFeedback::Binding& binding = xfb.buffers[b];
        if (!binding.buffer || !binding.buffer->mtl) return;
        const int s = link.xfbStream[b];
        captured[s] = true;
        // The bound range, clamped to the buffer's current size (it may have shrunk since binding).
        const int64_t available = std::max<int64_t>(0, binding.buffer->size - binding.offset);
        const int64_t size = binding.size ? std::min<int64_t>(binding.size, available) : available;
        const int64_t remaining = std::max<int64_t>(0, size - xfb.written[b]);
        const uint64_t vertexBytes = static_cast<uint64_t>(link.xfbStride[b]) * 4;
        if (vertexBytes) written[s] = std::min<uint64_t>(written[s], static_cast<uint64_t>(remaining) / vertexBytes / perPrimitive);
    }
    for (int s = 0; s < kMaxVertexStreams; ++s) {
        if (!captured[s]) written[s] = 0;
        else if (written[s] < emitted[s].size() / perPrimitive) queriesNoteXfbOverflow(ctx_, static_cast<GLuint>(s));
        if (writtenQuery[s]) writtenQuery[s]->result += written[s];
        xfb.primitivesWritten[static_cast<size_t>(s)] += written[s];
    }
    for (const shader::TransformFeedbackVarying& v : link.transformFeedback) {
        if (v.type == GL_NONE) continue;  // gl_SkipComponents / gl_NextBuffer write nothing
        const int s = link.xfbStream[v.buffer];
        const uint64_t vertexCount = written[s] * perPrimitive;
        Buffer& buffer = *xfb.buffers[v.buffer].buffer;
        uint8_t* base = buffer.contents() + xfb.buffers[v.buffer].offset + xfb.written[v.buffer];
        for (uint64_t i = 0; i < vertexCount; ++i) {
            uint32_t* dst = reinterpret_cast<uint32_t*>(base) + i * link.xfbStride[v.buffer] + v.offset;
            if (v.sourceOffset == shader::TransformFeedbackVarying::kNotEmitted)
                std::memset(dst, 0, static_cast<size_t>(v.components) * 4);
            else
                std::memcpy(dst, records + static_cast<uint64_t>(emitted[s][i]) * g.outputStride + v.sourceOffset,
                            static_cast<size_t>(v.components) * 4);
        }
    }
    for (int b = 0; b < link.xfbBufferCount; ++b) {
        if (!link.xfbBufferUsed(b)) continue;
        xfb.written[b] += static_cast<GLsizeiptr>(written[link.xfbStream[b]] * perPrimitive * link.xfbStride[b] * 4);
        ++xfb.buffers[b].buffer->version;
    }
}

// ------------------------------------------------------------ tessellation

bool DrawState::tessPrelude(LinkedProgram& program) {
    // WINEMETALGL_TESS_PRELUDE=0 keeps the capture and control kernel in the
    // command buffer being recorded.
    static const bool enabled = [] {
        const char* v = std::getenv("WINEMETALGL_TESS_PRELUDE");
        return !v || v[0] != '0';
    }();
    if (!enabled) return false;
    // The prelude runs before the command buffer being recorded: the vertex
    // and control stages may read nothing the GPU writes in it, and write
    // nothing but their outputs (images, atomic counters and storage block
    // writes stay in order).
    if (!program.link.images.empty() || !program.link.atomicBuffers.empty()) return false;
    const uint64_t serial = ctx_.stream().serial();
    const auto written = [serial](const Buffer* buffer) { return buffer && buffer->lastWriteSerial == serial; };
    const int stages[2] = {static_cast<int>(Stage::Vertex), static_cast<int>(Stage::TessControl)};
    const VertexArray& va = *ctx_.state.vertexArray;
    for (int a = 0; a < VertexArray::kMaxAttribs; ++a)
        if (va.attribs[static_cast<size_t>(a)].enabled && written(va.attribs[static_cast<size_t>(a)].buffer.get())) return false;
    for (size_t b = 0; b < program.link.blocks.size(); ++b) {
        if (program.link.blocks[b].stageSlot[stages[0]] < 0 && program.link.blocks[b].stageSlot[stages[1]] < 0) continue;
        const auto [owner, block] = blockSource(program, b);
        if (!owner) continue;
        const GLuint binding = block < owner->uniformBlockBindings.size() ? owner->uniformBlockBindings[block] : 0;
        if (binding < ContextState::kMaxUniformBufferBindings && written(ctx_.state.uniformBufferBindings[binding].buffer.get()))
            return false;
    }
    const shader::StageOutput* storage[2] = {&program.stage(Stage::Vertex), &program.link.tess.control};
    for (int s = 0; s < 2; ++s) {
        const Program* owner = stageProgram(program, stages[s]);
        if (!owner) owner = stageProgram(program, stages[0]);
        for (const shader::StageOutput::StorageArg& arg : storage[s]->storageArgs) {
            if (arg.writes || arg.counterBinding >= 0) return false;
            const GLuint binding = owner && arg.block >= 0 && static_cast<size_t>(arg.block) < owner->storageBlockBindings.size()
                                       ? owner->storageBlockBindings[static_cast<size_t>(arg.block)]
                                       : 0;
            if (binding < ContextState::kMaxShaderStorageBufferBindings &&
                written(ctx_.state.shaderStorageBufferBindings[binding].buffer.get()))
                return false;
        }
    }
    // Textures: only buffer textures, whose writes are tracked (render
    // passes in this command buffer may have written any other texture).
    for (size_t i = 0; i < program.link.samplers.size(); ++i) {
        const shader::SamplerSlotInfo& slot = program.link.samplers[i];
        if (slot.stageTexture[stages[0]] < 0 && slot.stageTexture[stages[1]] < 0) continue;
        if (slot.textureTarget != kTexBuffer) return false;
        const auto [owner, ownIndex] = samplerSource(program, i);
        const GLint unit = owner && ownIndex < owner->samplerUnits.size() ? owner->samplerUnits[ownIndex] : 0;
        if (unit < 0 || unit >= ContextState::kMaxTextureUnits) continue;
        const Texture* texture = ctx_.state.textureUnits[unit].bound[kTexBuffer].get();
        if (!texture) continue;
        // RGB32 buffer textures expand their texels in the stream (texture_storage.mm).
        if (texture->bufferFormat == GL_RGB32F || texture->bufferFormat == GL_RGB32I || texture->bufferFormat == GL_RGB32UI)
            return false;
        if (written(texture->buffer.get())) return false;
    }
    return true;
}

void DrawState::drawPatches(const std::vector<uint32_t>& vertices, GLsizei instances, uint32_t primitiveCycle,
                            const std::vector<PatchBase>* bases) {
    LinkedProgram& lp = *executable();
    const shader::TessellationEmulation& t = lp.link.tess;
    std::shared_ptr<LinkedProgram> evaluation = lp.tessEvaluation();
    if (!evaluation) return;
    const uint32_t patchVertices = static_cast<uint32_t>(std::max(1, ctx_.state.patchVertices));
    const size_t patches = vertices.size() / patchVertices;  // an incomplete last patch is ignored
    if (!patches) return;
    const std::vector<uint32_t> list(vertices.begin(), vertices.begin() + static_cast<long>(patches * patchVertices));
    const uint32_t copies = static_cast<uint32_t>(std::max(1, instances));
    const uint64_t total = static_cast<uint64_t>(patches) * copies;  // patch p of instance i is patch i * patches + p
    const uint32_t cycle = primitiveCycle ? primitiveCycle : static_cast<uint32_t>(patches);  // gl_PrimitiveID = patch % cycle
    if (t.hasControl && !lp.tessControlKernel()) return;
    const bool metal = metalTessellation(lp, *evaluation);
    // The capture and control kernel run ahead of the command buffer being
    // recorded when they can: otherwise each tessellated draw ends the
    // render pass in progress for the kernels and starts it again.
    const bool prelude = metal && t.hasControl && tessPrelude(lp);
    if (prelude) stats::count(&stats::Frame::tessPrelude);
    waitForCpuVertexReads(lp, true);
    prepareSamplerStorage(lp);
    queriesPrepareVisibility(ctx_);
    uint64_t serial = ctx_.stream().serial();

    // 1. The vertex outputs of every patch vertex, in patch order.
    TransientAllocation captured = transient_.allocate(total * patchVertices * t.inputStride * 4, serial, 16);
    if (!captured.buffer) return;  // too large for the device (logged)
    {
        const id<MTLBuffer> buffers[1] = {captured.buffer};
        const uint32_t base[1] = {static_cast<uint32_t>(captured.offset / 4)};
        const uint32_t stride[1] = {t.inputStride};
        preludeCapture_ = prelude;
        bool ok = true;
        if (bases && bases->size() > 1 && copies == 1) {
            // Commands with their own base instances: each vertex reads the
            // per-instance attributes of its command's.
            std::vector<uint32_t> instancesOf(list.size());
            for (size_t b = 0; b < bases->size(); ++b) {
                const size_t to = b + 1 < bases->size() ? (*bases)[b + 1].first : list.size();
                for (size_t v = (*bases)[b].first; v < to && v < list.size(); ++v) instancesOf[v] = (*bases)[b].baseInstance;
            }
            instanceGather_ = &instancesOf;
            ok = runVertexCapture(lp, list, 1, static_cast<uint32_t>(list.size()), 1, buffers, base, stride);
            instanceGather_ = nullptr;
        } else {
            ok = runVertexCapture(lp, list, instances, static_cast<uint32_t>(list.size() * copies), 1, buffers, base, stride);
        }
        preludeCapture_ = false;
        if (!ok) return;
    }

    // 2. The control shader, one threadgroup per patch; its levels are read
    // back for the tessellator (the patch records with them).
    std::vector<uint32_t> records;  // control: the patch records, copied out of the transient memory
    const uint32_t recordWords = t.recordWords();
    if (t.hasControl) {
        const uint16_t borders = borderVariant(lp, static_cast<int>(Stage::TessControl));
        id<MTLComputePipelineState> kernel =
            borders ? lp.kernelWithBorders(t.control, static_cast<int>(Stage::TessControl), borders) : lp.tessControlKernel();
        if (!kernel) return;  // the border variant did not compile (logged)
        TransientAllocation output = transient_.allocate(total * recordWords * 4, serial, 16);
        TransientAllocation params = transient_.allocate(16, serial, 16);  // device memory in the kernel, as above
        if (!output.buffer || !params.buffer) return;
        id<MTLComputeCommandEncoder> compute = prelude ? ctx_.stream().preludeCompute() : ctx_.stream().compute();
        [compute setComputePipelineState:kernel];
        bindKernelResources(compute, lp, t.control.usesDefaultUniforms, static_cast<int>(Stage::TessControl), &t.control);
        const uint32_t control[4] = {patchVertices, static_cast<uint32_t>(total), cycle, 0};
        [compute setBuffer:captured.buffer offset:captured.offset atIndex:shader::kSlotFirstXfbBuffer];
        [compute setBuffer:output.buffer offset:output.offset atIndex:shader::kSlotFirstXfbBuffer + 1];
        std::memcpy(params.data, control, sizeof(control));
        [compute setBuffer:params.buffer offset:params.offset atIndex:shader::kSlotFirstXfbBuffer + 2];
        const uint64_t groupPatches = std::max(1u, t.patchesPerGroup);
        [compute dispatchThreadgroups:MTLSizeMake((total + groupPatches - 1) / groupPatches, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(t.verticesOut * groupPatches), 1, 1)];
        if (metal) {
            // Metal's tessellator reads the levels on the GPU: no read-back.
            const uint32_t levelsAt = static_cast<uint32_t>(t.verticesOut) * t.vertexWords;
            TransientAllocation factors = transient_.allocate(total * (t.mode == GL_QUADS ? 12 : 8), serial, 16);
            if (!factors.buffer) return;
            const uint32_t params[4] = {recordWords, levelsAt, static_cast<uint32_t>(total),
                                        (t.mode == GL_QUADS ? 1u : 0u) | (t.spacing == GL_EQUAL ? 2u : 0u)};
            [compute setComputePipelineState:tessFactorsKernel()];
            [compute setBuffer:output.buffer offset:output.offset atIndex:0];
            [compute setBuffer:factors.buffer offset:factors.offset atIndex:1];
            [compute setBytes:params length:sizeof(params) atIndex:2];
            [compute dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            if (drawMetalPatches(lp, *evaluation, output.buffer, output.offset, factors.buffer, factors.offset, total,
                                 cycle, patchVertices))
                return;
        }
        stats::count(&stats::Frame::tessCpu);
        if (prelude) ctx_.stream().waitForPrelude();
        else ctx_.stream().waitFor(serial);
        const auto* words = static_cast<const uint32_t*>(output.data);
        records.assign(words, words + total * recordWords);
        serial = ctx_.stream().serial();
    }

    if (!t.hasControl && metal) {
        // The default levels, the same for every patch (a record stride of 0).
        TransientAllocation levels = transient_.allocate(6 * 4, serial, 16);
        TransientAllocation factors = transient_.allocate(total * (t.mode == GL_QUADS ? 12 : 8), serial, 16);
        id<MTLComputePipelineState> kernel = tessFactorsKernel();
        if (!levels.buffer || !factors.buffer || !kernel) return;
        std::memcpy(levels.data, ctx_.state.patchDefaultOuterLevel, 4 * 4);
        std::memcpy(static_cast<uint8_t*>(levels.data) + 16, ctx_.state.patchDefaultInnerLevel, 2 * 4);
        const uint32_t params[4] = {0, 0, static_cast<uint32_t>(total),
                                    (t.mode == GL_QUADS ? 1u : 0u) | (t.spacing == GL_EQUAL ? 2u : 0u)};
        id<MTLComputeCommandEncoder> compute = ctx_.stream().compute();
        [compute setComputePipelineState:kernel];
        [compute setBuffer:levels.buffer offset:levels.offset atIndex:0];
        [compute setBuffer:factors.buffer offset:factors.offset atIndex:1];
        [compute setBytes:params length:sizeof(params) atIndex:2];
        [compute dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (drawMetalPatches(lp, *evaluation, captured.buffer, captured.offset, factors.buffer, factors.offset, total,
                             cycle, patchVertices))
            return;
    }
    if (!t.hasControl) stats::count(&stats::Frame::tessCpu);

    // 3. Tessellation: domain points (patch, u, v, w) and the primitive list.
    if (!tessellator_) tessellator_ = std::make_unique<tess::Tessellator>();
    const tess::Mode mode{t.mode, t.spacing, t.order == GL_CW, t.pointMode};
    std::vector<uint32_t> points;  // 4 words per domain point
    std::vector<uint32_t> indices;
    tess::Patch patch;
    float lastOuter[4] = {}, lastInner[2] = {};
    bool haveLast = false;
    for (uint64_t p = 0; p < total; ++p) {
        float outer[4], inner[2];
        if (t.hasControl) {
            const uint32_t* levels = records.data() + p * recordWords + static_cast<uint64_t>(t.verticesOut) * t.vertexWords;
            std::memcpy(outer, levels, sizeof(outer));
            std::memcpy(inner, levels + 4, sizeof(inner));
        } else {
            std::memcpy(outer, ctx_.state.patchDefaultOuterLevel, sizeof(outer));
            std::memcpy(inner, ctx_.state.patchDefaultInnerLevel, sizeof(inner));
        }
        // Patches with the same levels tessellate the same way.
        if (!haveLast || std::memcmp(outer, lastOuter, sizeof(outer)) != 0 || std::memcmp(inner, lastInner, sizeof(inner)) != 0) {
            tessellator_->tessellate(mode, outer, inner, patch);
            std::memcpy(lastOuter, outer, sizeof(outer));
            std::memcpy(lastInner, inner, sizeof(inner));
            haveLast = true;
        }
        if (patch.indices.empty()) continue;
        const uint32_t base = static_cast<uint32_t>(points.size() / 4);
        for (size_t i = 0; i < patch.u.size(); ++i) {
            const float u = patch.u[i], v = patch.v[i], w = t.mode == GL_TRIANGLES ? 1.0f - u - v : 0.0f;
            uint32_t word[4] = {static_cast<uint32_t>(p), 0, 0, 0};
            std::memcpy(&word[1], &u, 4);
            std::memcpy(&word[2], &v, 4);
            std::memcpy(&word[3], &w, 4);
            points.insert(points.end(), word, word + 4);
        }
        for (uint32_t index : patch.indices) indices.push_back(base + index);
    }
    const GLenum listMode = t.pointMode ? GL_POINTS : t.mode == GL_ISOLINES ? GL_LINES : GL_TRIANGLES;
    if (t.hasControl) queriesCountStatistic(ctx_, kStatTessControlShaderPatches, total);
    queriesCountStatistic(ctx_, kStatTessEvaluationShaderInvocations, points.size() / 4);
    if (!evaluation->link.geometry.active) {
        const uint64_t primitives = indices.size() / (listMode == GL_POINTS ? 1 : listMode == GL_LINES ? 2 : 3);
        queriesCountStatistic(ctx_, kStatClippingInputPrimitives, primitives);
        queriesCountStatistic(ctx_, kStatClippingOutputPrimitives, primitives);
    }
    if (indices.empty()) return;

    // 4. The evaluation stage over the domain points, drawn (or captured, or
    // fed to the geometry shader) as independent primitives.
    TransientAllocation pointData = transient_.allocate(points.size() * 4, serial, 16);
    std::memcpy(pointData.data, points.data(), points.size() * 4);
    TransientAllocation patchData;
    if (t.hasControl) {
        patchData = transient_.allocate(records.size() * 4, serial, 16);
        std::memcpy(patchData.data, records.data(), records.size() * 4);
    } else {
        patchData = captured;  // read in this command buffer
    }
    struct {
        uint32_t patchVertices, patchesPerInstance, pad[2];
        float outer[4], inner[2], pad2[2];
    } control = {patchVertices, cycle, {0, 0}, {}, {}, {}};
    std::memcpy(control.outer, ctx_.state.patchDefaultOuterLevel, sizeof(control.outer));
    std::memcpy(control.inner, ctx_.state.patchDefaultInnerLevel, sizeof(control.inner));
    TransientAllocation controlData = transient_.allocate(sizeof(control), serial, 16);
    std::memcpy(controlData.data, &control, sizeof(control));
    tessInputs_.active = true;
    tessInputs_.buffers[0] = pointData.buffer;
    tessInputs_.offsets[0] = pointData.offset;
    tessInputs_.buffers[1] = patchData.buffer;
    tessInputs_.offsets[1] = patchData.offset;
    tessInputs_.buffers[2] = controlData.buffer;
    tessInputs_.offsets[2] = controlData.offset;
    override_ = evaluation.get();
    if (evaluation->link.geometry.active) {
        uint32_t primitiveBase = 0;
        drawGeometry(listMode, indices, 1, primitiveBase);
    } else {
        const Capture captured = ctx_.state.transformFeedback->active ? captureTransformFeedback(listMode, indices, 1) : kNotCaptured;
        if (captured == kCaptured && ctx_.state.rasterizerDiscard)
            queriesCountPrimitives(ctx_, listMode, static_cast<GLsizei>(indices.size()), 1);
        else
            issue(listMode, 0, static_cast<GLsizei>(indices.size()), 1, indices.data(), nil, 0, MTLIndexTypeUInt32, 0, 0,
                  static_cast<GLsizei>(points.size() / 4));
    }
    override_ = nullptr;
    tessInputs_ = {};
}

bool DrawState::metalTessellation(LinkedProgram& program, LinkedProgram& evaluation) {
    // WINEMETALGL_METAL_TESSELLATION=0 keeps every draw on the CPU tessellator.
    static const bool enabled = [] {
        const char* v = std::getenv("WINEMETALGL_METAL_TESSELLATION");
        return !v || v[0] != '0';
    }();
    const shader::TessellationEmulation& t = program.link.tess;
    const ContextState& st = ctx_.state;
    const char* why = !enabled                                                    ? nullptr
                      : t.mode == GL_ISOLINES || t.pointMode                       ? "isolines or point mode"
                      : evaluation.link.geometry.active                            ? "geometry shader"
                      : st.transformFeedback->active                               ? "transform feedback"
                      : st.rasterizerDiscard                                       ? "rasterizer discard"
                      : queriesNeedDrawSizes(ctx_)                                 ? "active query"
                      : pointPolygons(GL_TRIANGLES)                                ? "point polygon mode"
                      : !tessFactorsKernel()                                       ? "no factor kernel"
                      : evaluation.metalTessellationFailed                         ? "no Metal tessellation pipeline"
                      : !evaluation.postTessellationVertex(t.mode == GL_QUADS, 0) ? "no post-tessellation function"
                                                                                   : "";
    if (why && *why && stats::g_enabled)
        stats::describeOnce("tess-cpu:" + std::to_string(program.uid) + why, "patches of link %llu tessellated on the CPU: %s",
                            static_cast<unsigned long long>(program.uid), why);
    return why && !*why;
}

bool DrawState::drawMetalPatches(LinkedProgram& program, LinkedProgram& evaluation, id<MTLBuffer> records,
                                 NSUInteger recordsOffset, id<MTLBuffer> factors, NSUInteger factorsOffset, uint64_t patches,
                                 uint32_t patchesPerInstance, uint32_t patchVertices, id<MTLBuffer> indirect,
                                 NSUInteger indirectOffset) {
    const shader::TessellationEmulation& t = program.link.tess;
    const uint64_t serial = ctx_.stream().serial();
    struct {
        uint32_t patchVertices, patchesPerInstance, pad[2];
        float outer[4], inner[2], pad2[2];
    } control = {patchVertices, patchesPerInstance, {0, 0}, {}, {}, {}};
    std::memcpy(control.outer, ctx_.state.patchDefaultOuterLevel, sizeof(control.outer));
    std::memcpy(control.inner, ctx_.state.patchDefaultInnerLevel, sizeof(control.inner));
    TransientAllocation controlData = transient_.allocate(sizeof(control), serial, 16);
    if (!controlData.buffer) return true;
    std::memcpy(controlData.data, &control, sizeof(control));
    // The domain point buffer is unread (its slot is bound to keep the
    // argument valid).
    tessInputs_.active = true;
    tessInputs_.buffers[0] = controlData.buffer;
    tessInputs_.offsets[0] = controlData.offset;
    tessInputs_.buffers[1] = records;
    tessInputs_.offsets[1] = recordsOffset;
    tessInputs_.buffers[2] = controlData.buffer;
    tessInputs_.offsets[2] = controlData.offset;
    override_ = &evaluation;
    {
        stats::Timer timer(&stats::Frame::drawSamplersNanos);
        prepareSamplerStorage(evaluation);
    }
    MTLPrimitiveType primitive = MTLPrimitiveTypeTriangle;
    id<MTLRenderCommandEncoder> encoder = prepare(GL_TRIANGLES, primitive);
    bool drawn = true;
    if (encoder) {
        // GL's vertex order is Metal's winding (the CPU path's D3D
        // tessellator takes the opposite one: tess/tessellator.cpp).
        const uint8_t partition = static_cast<uint8_t>(t.spacing == GL_FRACTIONAL_ODD    ? MTLTessellationPartitionModeFractionalOdd
                                                       : t.spacing == GL_FRACTIONAL_EVEN ? MTLTessellationPartitionModeFractionalEven
                                                                                         : MTLTessellationPartitionModeInteger);
        PipelineKey key = lastKey_;
        key.tessellation = static_cast<uint8_t>(1 | (t.mode == GL_QUADS ? 2 : 0) | partition << 2 | (t.order == GL_CW ? 16 : 0));
        if (stats::g_enabled && rotateProvoking(GL_TRIANGLES, evaluation)) stats::count(&stats::Frame::tessMetalFlat);
        id<MTLRenderPipelineState> pipeline = ctx_.share().pipelines().render(key, evaluation);
        if (!pipeline) {
            evaluation.metalTessellationFailed = true;
            WMGL_WARN(Draw, "no Metal tessellation pipeline for the patches of link %llu: tessellating them on the CPU",
                      static_cast<unsigned long long>(program.uid));
            drawn = false;
        } else {
            stats::count(&stats::Frame::tessMetal);
            stats::count(&stats::Frame::draws);
            [encoder setRenderPipelineState:pipeline];
            encoderState_.pipeline = nil;  // issue() sets its own again
            bindDrawState(encoder, evaluation);
            if (evaluation.link.xfbVertexCapture) bindCaptureDefaults(encoder, evaluation);
            [encoder setTessellationFactorBuffer:factors offset:factorsOffset instanceStride:0];
            const NSUInteger controlPoints = static_cast<NSUInteger>(t.hasControl ? t.verticesOut : static_cast<GLint>(patchVertices));
            if (indirect)
                [encoder drawPatches:controlPoints patchIndexBuffer:nil patchIndexBufferOffset:0 indirectBuffer:indirect
                    indirectBufferOffset:indirectOffset];
            else
                [encoder drawPatches:controlPoints
                           patchStart:0
                           patchCount:static_cast<NSUInteger>(patches)
                     patchIndexBuffer:nil
               patchIndexBufferOffset:0
                        instanceCount:1
                         baseInstance:0];
            ctx_.stream().noteDraw();
        }
    }
    override_ = nullptr;
    tessInputs_ = {};
    return drawn;
}

bool DrawState::canBatchPatchCommands() {
    LinkedProgram* lp = executable();
    if (!lp || !lp->link.tess.active || lp->link.geometry.active || ffWideLines_) return false;
    if (ctx_.state.transformFeedback->active || queriesNeedDrawSizes(ctx_) || queriesCountingStatistics(ctx_)) return false;
    if (Program* owner = stageProgram(*lp, static_cast<int>(Stage::Vertex)); owner && owner->linked) {
        const shader::LinkResult::BuiltinUniforms& b = owner->linked->link.builtinUniforms;
        if (b.drawId >= 0 || b.baseVertex >= 0 || b.baseInstance >= 0) return false;
    }
    return true;
}

void DrawState::drawPatchCommands(GLenum type, const uint32_t (*commands)[5], size_t count) {
    if (!count) return;
    ImageShadowScope shadows(*this);
    beginDraw(GL_PATCHES);
    if (!validateDraw(GL_PATCHES) || !conditionalRenderAllows(ctx_)) return;
    Buffer* elements = ctx_.state.vertexArray->elementBuffer.get();
    if (!elements || !elements->mtl || elements->mappedNonPersistent()) return;
    const size_t size = type == GL_UNSIGNED_BYTE ? 1 : type == GL_UNSIGNED_SHORT ? 2 : 4;
    // The indices are read on the CPU, after GPU writes to the buffer.
    if (elements->lastWriteSerial) ctx_.stream().waitFor(elements->lastWriteSerial);
    const uint8_t* data = elements->contents();
    const uint64_t bytes = static_cast<uint64_t>(std::max<GLsizeiptr>(0, elements->size));
    std::vector<uint32_t> vertices;
    vertices.reserve(count * commands[0][0]);
    // Commands with another base instance than the one before start a capture of their own.
    std::vector<PatchBase> bases;
    for (size_t c = 0; c < count; ++c) {
        const uint64_t first = uint64_t{commands[c][2]} * size, n = commands[c][0];
        if (first + n * size > bytes) continue;  // out-of-range index fetches draw nothing (drawElements)
        if (bases.empty() || bases.back().baseInstance != commands[c][4]) bases.push_back({vertices.size(), commands[c][4]});
        const uint32_t bias = commands[c][3];
        for (uint64_t i = 0; i < n; ++i) {
            const uint8_t* at = data + first + i * size;
            uint32_t v;
            if (size == 1) v = *at;
            else if (size == 2) { uint16_t x; std::memcpy(&x, at, 2); v = x; }
            else std::memcpy(&v, at, 4);
            vertices.push_back(v + bias);
        }
    }
    const uint32_t patchVertices = static_cast<uint32_t>(std::max(1, ctx_.state.patchVertices));
    BaseInstanceScope scope(*this, commands[0][4]);
    if (!bases.empty() && bases.size() > 1) drawPatches(vertices, 1, commands[0][0] / patchVertices, &bases);
    else drawPatches(vertices, 1, commands[0][0] / patchVertices);
}

namespace {

id<MTLComputePipelineState> utilityKernel(const char* name) {
    id<MTLFunction> function = [Device::get().utilityLibrary() newFunctionWithName:@(name)];
    NSError* error = nil;
    id<MTLComputePipelineState> state = function ? [Device::get().mtl() newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!state) WMGL_ERR(Pipeline, "%s kernel: %s", name, error ? error.localizedDescription.UTF8String : "missing");
    return state;
}

id<MTLComputePipelineState> mdiPatchScanKernel() {
    static id<MTLComputePipelineState> state = utilityKernel("wmgl_mdi_patch_scan");
    return state;
}

id<MTLComputePipelineState> mdiPatchExpandKernel() {
    static id<MTLComputePipelineState> state = utilityKernel("wmgl_mdi_patch_expand");
    return state;
}

// The vertices a DrawElementsIndirectCommand lists for patches of
// `patchVertices` (whole patches of each instance), 0 if out of range.
uint64_t mdiPatchVertices(const uint32_t* c, uint64_t elementCount, uint32_t patchVertices) {
    const uint64_t count = c[0], instances = c[1], first = c[2];
    if (!count || !instances || first > elementCount || count > elementCount - first) return 0;
    return count / patchVertices * patchVertices * instances;
}

}  // namespace

bool DrawState::drawPatchesIndirectGpu(GLenum type, Buffer& commands, uintptr_t offset, GLsizei drawcount, uintptr_t stride) {
    static const bool enabled = [] {
        const char* v = std::getenv("WINEMETALGL_TESS_MDI_GPU");
        return !v || v[0] != '0';
    }();
    if (!enabled || drawcount <= 0 || (type != GL_UNSIGNED_INT && type != GL_UNSIGNED_SHORT)) return false;
    if (!commands.mtl || offset % 4 || stride % 4 || commands.mappedNonPersistent()) return false;
    if (!mdiPatchScanKernel() || !mdiPatchExpandKernel() || !tessFactorsKernel()) return false;
    ImageShadowScope shadows(*this);
    beginDraw(GL_PATCHES);
    LinkedProgram* current = executable();
    if (!current || !canBatchPatchCommands()) return false;
    LinkedProgram& lp = *current;
    const shader::TessellationEmulation& t = lp.link.tess;
    if (!t.hasControl || t.verticesOut <= 0 || t.pointMode || t.readsPrimitiveId || t.readsInstanceId) return false;
    std::shared_ptr<LinkedProgram> evaluation = lp.tessEvaluation();
    if (!evaluation || !lp.tessControlKernel() || !metalTessellation(lp, *evaluation)) return false;
    CommandStream& stream = ctx_.stream();
    Buffer* elements = ctx_.state.vertexArray->elementBuffer.get();
    if (!elements || !elements->mtl || elements->mappedNonPersistent()) return false;
    // The prelude runs ahead of the command buffer being recorded: nothing it
    // reads may be written in it (Foundation's culling ends at a barrier,
    // which commits it: CommandStream::memoryBarrier).
    const uint64_t serial = stream.serial();
    if (commands.lastWriteSerial == serial || elements->lastWriteSerial == serial || !tessPrelude(lp)) return false;
    VertexInputPlan plan;
    planVertexInputs(lp, false, plan);
    const VertexArray& vao = *ctx_.state.vertexArray;
    uint32_t windowEnd[13] = {};
    for (const shader::StageOutput::VertexInput& input : lp.stage(Stage::Vertex).vertexInputs) {
        if (input.location < 0 || input.location >= VertexArray::kAttribSlots) continue;
        const VertexInputPlan::How how = plan.how[static_cast<size_t>(input.location)];
        if (how == VertexInputPlan::Direct) {
            const size_t g = static_cast<size_t>(plan.group[static_cast<size_t>(input.location)]);
            const VertexFetch fetch = vertexFetch(input, attribFor(vao, input.source()));
            windowEnd[g] = std::max<uint32_t>(windowEnd[g], static_cast<uint32_t>(fetch.offset - plan.groups[g].base + fetch.bytes));
        } else if (how != VertexInputPlan::Constant && how != VertexInputPlan::None) {
            return false;  // converted on the CPU
        }
    }
    if (plan.groups.size() > 8) return false;
    // Errors (and a draw conditional rendering discards) as the CPU path's first command reports them.
    if (!validateDraw(GL_PATCHES) || !conditionalRenderAllows(ctx_)) return true;

    const uint32_t patchVertices = static_cast<uint32_t>(std::max(1, ctx_.state.patchVertices));
    const uint32_t indexSize = type == GL_UNSIGNED_SHORT ? 2 : 4;
    const uint64_t elementCount = static_cast<uint64_t>(std::max<GLsizeiptr>(0, elements->size)) / indexSize;
    const uint64_t siteKey = (lp.uid * 0x9e3779b97f4a7c15ull) ^ (uint64_t{commands.name} << 40) ^ (uint64_t{offset} << 8) ^
                             static_cast<uint64_t>(drawcount) ^ (uint64_t{patchVertices} << 60);
    if (mdiSites_.size() > 1024) mdiSites_.clear();
    MdiSite& site = mdiSites_[siteKey];
    if (!site.report) {
        site.report = [Device::get().mtl() newBufferWithLength:8 * 4 options:MTLResourceStorageModeShared];
        if (!site.report) return false;
    }
    const auto grow = [&](uint64_t total) {
        // A site sized once keeps a capacity even when its commands drew
        // nothing (all culled): it is never sized on the CPU again.
        if (site.capacity && total <= site.capacity) return;
        // Patches past the capacity are not drawn until a later report grows
        // it (wmgl_mdi_patch_scan), so it holds at least one patch per
        // command: culling that keeps or drops each command's patch
        // (Foundation's terrain, issue #12) then fits from the first frame
        // that shows them, even for a site sized while everything was culled.
        const uint64_t perCommand = std::min<uint64_t>(uint64_t{static_cast<uint32_t>(drawcount)} * patchVertices, 1u << 17);
        const uint64_t wanted = std::max<uint64_t>({256, total * 2, perCommand});
        site.capacity = static_cast<uint32_t>(std::min<uint64_t>(wanted, (1u << 26)) / patchVertices * patchVertices);
    };
    for (int i = 0; i < 8; ++i)
        if (site.serials[i] && stream.queue().isComplete(site.serials[i])) {
            grow(static_cast<const uint32_t*>(site.report.contents)[i]);
            site.serials[i] = 0;
        }
    if (!site.capacity) {
        // The first draw of the site sizes the buffers from the commands
        // (read once on the CPU, as the CPU path that draws it now does).
        if (commands.lastWriteSerial) waitForGpuUnlocked(ctx_, commands.lastWriteSerial);
        if (!commands.mtl || offset + uint64_t{static_cast<uint64_t>(drawcount - 1)} * stride + 20 > static_cast<uint64_t>(commands.size))
            return false;
        uint64_t total = 0;
        for (GLsizei c = 0; c < drawcount; ++c)
            total += mdiPatchVertices(reinterpret_cast<const uint32_t*>(commands.contents() + offset + uint64_t{static_cast<uint64_t>(c)} * stride),
                                      elementCount, patchVertices);
        grow(total);
        return false;
    }
    const uint32_t capacity = site.capacity, patchCapacity = capacity / patchVertices;
    if (!patchCapacity) return false;

    // Buffers, all in this command buffer's transient memory.
    uint32_t windowWords[8] = {}, dstWords[8] = {}, gatheredWords = 0;
    for (size_t g = 0; g < plan.groups.size(); ++g) {
        windowWords[g] = (windowEnd[g] + 3) / 4;
        dstWords[g] = gatheredWords;
        gatheredWords += capacity * windowWords[g];
    }
    const uint32_t recordWords = t.recordWords();
    const bool quads = t.mode == GL_QUADS;
    TransientAllocation starts = transient_.allocate((static_cast<size_t>(drawcount) + 1) * 4, serial, 16);
    TransientAllocation args = transient_.allocate(96, serial, 16);
    TransientAllocation capture = transient_.allocate((12 + size_t{capacity}) * 4, serial, 16);
    TransientAllocation controlParams = transient_.allocate(16, serial, 16);
    TransientAllocation factorParams = transient_.allocate(16, serial, 16);
    // The records and factors share the gathered windows' memory: the
    // capture pass reads the windows (prelude render buffer) before the
    // control kernels write the records (prelude control buffer, committed
    // after it), and tracked resources order the writes after the reads.
    // The capacity holds a patch per command (grow()), so this is most of
    // what a site with many commands takes in a frame.
    const size_t recordsBytes = (size_t{patchCapacity} * recordWords * 4 + 15) / 16 * 16;
    const size_t sharedBytes = std::max<size_t>({4, size_t{gatheredWords} * 4, recordsBytes + size_t{patchCapacity} * (quads ? 12 : 8)});
    TransientAllocation gathered = transient_.allocate(sharedBytes, serial, 16);
    TransientAllocation captured = transient_.allocate(size_t{capacity} * t.inputStride * 4, serial, 16);
    TransientAllocation records = gathered, factors = gathered;
    factors.offset += recordsBytes;
    if (factors.data) factors.data = static_cast<uint8_t*>(factors.data) + recordsBytes;
    if (!starts.buffer || !args.buffer || !capture.buffer || !controlParams.buffer || !factorParams.buffer || !gathered.buffer ||
        !captured.buffer || !records.buffer || !factors.buffer)
        return false;
    {
        auto* w = static_cast<uint32_t*>(capture.data);  // active, vertex count, limit (GPU), pad, base[4], stride[4]
        std::memset(w, 0, 12 * 4);
        w[0] = 1;
        w[4] = static_cast<uint32_t>(captured.offset / 4);
        w[8] = t.inputStride;
        const uint32_t control[4] = {patchVertices, 0, UINT32_MAX, 0};  // patch count from the GPU
        std::memcpy(controlParams.data, control, sizeof(control));
        const uint32_t levelsAt = static_cast<uint32_t>(t.verticesOut) * t.vertexWords;
        const uint32_t factor[4] = {recordWords, levelsAt, 0, (quads ? 1u : 0u) | (t.spacing == GL_EQUAL ? 2u : 0u)};
        std::memcpy(factorParams.data, factor, sizeof(factor));
    }
    struct {
        uint32_t drawCount, strideWords, elementCount, patchVertices, capacity, indexSize, groups, reportSlot;
        uint32_t controlGroupPatches, pad[3];
    } params = {static_cast<uint32_t>(drawcount), static_cast<uint32_t>(stride / 4),
                static_cast<uint32_t>(std::min<uint64_t>(elementCount, UINT32_MAX)), patchVertices, capacity, indexSize,
                static_cast<uint32_t>(plan.groups.size()), site.next, std::max(1u, t.patchesPerGroup), {0, 0, 0}};
    struct Group {
        uint32_t srcBase, srcStride, divisor, windowWords, dstWords, srcLength, pad0, pad1;
    } groups[8] = {};
    for (size_t g = 0; g < plan.groups.size(); ++g) {
        const VertexInputPlan::DirectGroup& d = plan.groups[g];
        groups[g] = {static_cast<uint32_t>(d.base), d.stride, d.divisor, windowWords[g], dstWords[g],
                     static_cast<uint32_t>(std::min<NSUInteger>(d.buffer->mtl.length, UINT32_MAX)), 0, 0};
    }
    site.serials[site.next] = serial;
    site.next = (site.next + 1) % 8;

    // 1. List the patch vertices and gather their attributes.
    id<MTLComputeCommandEncoder> setup = stream.preludeStep(CommandStream::PreludeStep::Scan);
    id<MTLComputePipelineState> scan = mdiPatchScanKernel();
    [setup setComputePipelineState:scan];
    [setup setBuffer:commands.mtl offset:offset atIndex:0];
    [setup setBuffer:starts.buffer offset:starts.offset atIndex:1];
    [setup setBuffer:args.buffer offset:args.offset atIndex:2];
    [setup setBuffer:capture.buffer offset:capture.offset atIndex:3];
    [setup setBuffer:controlParams.buffer offset:controlParams.offset atIndex:4];
    [setup setBuffer:factorParams.buffer offset:factorParams.offset atIndex:5];
    [setup setBytes:&params length:sizeof(params) atIndex:6];
    [setup setBuffer:site.report offset:0 atIndex:7];
    const NSUInteger scanThreads = std::min<NSUInteger>(1024, scan.maxTotalThreadsPerThreadgroup) / 32 * 32;
    [setup dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(scanThreads, 1, 1)];
    setup = stream.preludeStep(CommandStream::PreludeStep::Expand);
    [setup setComputePipelineState:mdiPatchExpandKernel()];
    [setup setBuffer:commands.mtl offset:offset atIndex:0];
    [setup setBuffer:starts.buffer offset:starts.offset atIndex:1];
    [setup setBuffer:capture.buffer offset:capture.offset atIndex:3];
    [setup setBuffer:elements->mtl offset:0 atIndex:2];
    [setup setBuffer:gathered.buffer offset:gathered.offset atIndex:4];
    [setup setBytes:&params length:sizeof(params) atIndex:5];
    [setup setBytes:groups length:sizeof(groups) atIndex:6];
    for (size_t g = 0; g < 8; ++g)
        [setup setBuffer:g < plan.groups.size() ? plan.groups[g].buffer->mtl : elements->mtl offset:0 atIndex:7 + g];
    [setup dispatchThreadgroupsWithIndirectBuffer:args.buffer indirectBufferOffset:args.offset + 48
                            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    commands.lastUseSerial = elements->lastUseSerial = serial;
    for (const VertexInputPlan::DirectGroup& g : plan.groups) g.buffer->lastUseSerial = serial;
    stats::count(&stats::Frame::tessMdiGpu);
    stats::count(&stats::Frame::tessMdiGpuCommands, static_cast<uint64_t>(drawcount));
    stats::count(&stats::Frame::tessPrelude);
    queriesPrepareVisibility(ctx_);
    prepareSamplerStorage(lp);

    // 2. The vertex capture, over the listed vertices.
    {
        PipelineKey key;
        std::memset(&key, 0, sizeof(key));
        key.programId = lp.uid;
        key.borders[0] = borderVariant(lp, static_cast<int>(Stage::Vertex));
        key.rasterization = 0;
        key.topologyClass = static_cast<uint8_t>(MTLPrimitiveTopologyClassPoint);
        key.sampleCount = 1;
        id<MTLRenderCommandEncoder> encoder = stream.preludeRender();
        GpuGather gather;
        gather.buffer = gathered.buffer;
        for (size_t g = 0; g < plan.groups.size(); ++g) {
            gather.offsets[g] = gathered.offset + NSUInteger{dstWords[g]} * 4;
            gather.strides[g] = windowWords[g] * 4;
        }
        plannedFor_ = 0;
        gpuGather_ = &gather;
        const bool bound = bindVertexInputs(encoder, key, lp, 0, 0, 1, nullptr);
        gpuGather_ = nullptr;
        id<MTLRenderPipelineState> pipeline = bound ? ctx_.share().pipelines().render(key, lp) : nil;
        encoderState_.pass = 0;  // the prelude's encoder, not the pass in progress
        if (!pipeline) return true;
        [encoder setRenderPipelineState:pipeline];
        encoderState_.pipeline = nil;  // issue() sets its own again
        bindUniforms(encoder, lp, false);
        bindTextures(encoder, lp, false);
        [encoder setVertexBuffer:captured.buffer offset:0 atIndex:shader::kSlotFirstXfbBuffer];
        [encoder setVertexBuffer:capture.buffer offset:capture.offset atIndex:shader::kSlotXfbControl];
        [encoder drawPrimitives:MTLPrimitiveTypePoint indirectBuffer:args.buffer indirectBufferOffset:args.offset];
        encoderState_.pass = 0;
    }

    // 3. The control kernel (one threadgroup per patch) and the factors.
    {
        const uint16_t borders = borderVariant(lp, static_cast<int>(Stage::TessControl));
        id<MTLComputePipelineState> kernel =
            borders ? lp.kernelWithBorders(t.control, static_cast<int>(Stage::TessControl), borders) : lp.tessControlKernel();
        if (!kernel) return true;
        id<MTLComputeCommandEncoder> compute = stream.preludeStep(CommandStream::PreludeStep::Control);
        [compute setComputePipelineState:kernel];
        bindKernelResources(compute, lp, t.control.usesDefaultUniforms, static_cast<int>(Stage::TessControl), &t.control);
        [compute setBuffer:captured.buffer offset:captured.offset atIndex:shader::kSlotFirstXfbBuffer];
        [compute setBuffer:records.buffer offset:records.offset atIndex:shader::kSlotFirstXfbBuffer + 1];
        [compute setBuffer:controlParams.buffer offset:controlParams.offset atIndex:shader::kSlotFirstXfbBuffer + 2];
        [compute dispatchThreadgroupsWithIndirectBuffer:args.buffer indirectBufferOffset:args.offset + 16
                                  threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(t.verticesOut) * std::max(1u, t.patchesPerGroup), 1, 1)];
        compute = stream.preludeStep(CommandStream::PreludeStep::Factors);
        [compute setComputePipelineState:tessFactorsKernel()];
        [compute setBuffer:records.buffer offset:records.offset atIndex:0];
        [compute setBuffer:factors.buffer offset:factors.offset atIndex:1];
        [compute setBuffer:factorParams.buffer offset:factorParams.offset atIndex:2];
        [compute dispatchThreadgroupsWithIndirectBuffer:args.buffer indirectBufferOffset:args.offset + 32
                                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }

    // 4. The tessellation draw, in the pass in progress.
    drawMetalPatches(lp, *evaluation, records.buffer, records.offset, factors.buffer, factors.offset, patchCapacity, UINT32_MAX,
                     patchVertices, args.buffer, args.offset + 64);
    return true;
}

void DrawState::drawArrays(GLenum mode, GLint first, GLsizei count, GLsizei instances) {
    ImageShadowScope shadows(*this);
    beginDraw(mode);
    if (first < 0 || count < 0 || instances < 0) {
        ctx_.setError(GL_INVALID_VALUE);
        return;
    }
    if (!validateDraw(mode)) return;
    if (count == 0 || instances == 0 || !conditionalRenderAllows(ctx_)) return;
    // Pipeline statistics of the command; tessellation and geometry shaders count their own clipping.
    queriesCountSubmitted(ctx_, mode, count, instances, mode != GL_PATCHES && executable() && !executable()->link.geometry.active);
    if (executable() && ffWideLines_) {
        std::vector<uint32_t> vertices(static_cast<size_t>(count));
        for (GLsizei i = 0; i < count; ++i) vertices[static_cast<size_t>(i)] = static_cast<uint32_t>(first + i);
        drawWideLines(mode, vertices, instances);
        return;
    }
    if (mode == GL_PATCHES) {
        std::vector<uint32_t> vertices(static_cast<size_t>(count));
        for (GLsizei i = 0; i < count; ++i) vertices[static_cast<size_t>(i)] = static_cast<uint32_t>(first + i);
        drawPatches(vertices, instances);
        return;
    }
    if (executable()->link.geometry.active) {
        std::vector<uint32_t> vertices(static_cast<size_t>(count));
        for (GLsizei i = 0; i < count; ++i) vertices[static_cast<size_t>(i)] = static_cast<uint32_t>(first + i);
        uint32_t primitiveBase = 0;
        drawGeometry(mode, vertices, instances, primitiveBase);
        return;
    }
    if (ctx_.state.transformFeedback->active) {
        std::vector<uint32_t> vertices(static_cast<size_t>(count));
        for (GLsizei i = 0; i < count; ++i) vertices[static_cast<size_t>(i)] = static_cast<uint32_t>(first + i);
        // Once the capture ran the vertex stage, a discarding draw has nothing left to do.
        const Capture captured = captureTransformFeedback(mode, vertices, instances);
        if (captured == kCapturedAndDrawn || (captured == kCaptured && ctx_.state.rasterizerDiscard)) {
            queriesCountPrimitives(ctx_, mode, count, instances);
            return;
        }
    }
    issue(mode, first, count, instances, nullptr, nil, 0, MTLIndexTypeUInt16, 0, first, count);
}

namespace {

// The smallest and largest of `count` indices of type T at `p` (any alignment).
template <typename T>
void scanIndexRange(const uint8_t* p, GLsizei count, uint32_t& lo, uint32_t& hi) {
    T low = static_cast<T>(~T(0)), high = 0;
    for (GLsizei i = 0; i < count; ++i) {
        T v;
        std::memcpy(&v, p + static_cast<size_t>(i) * sizeof(T), sizeof(T));
        low = v < low ? v : low;
        high = v > high ? v : high;
    }
    if (count > 0) {
        lo = low;
        hi = high;
    }
}

}  // namespace

void DrawState::drawElements(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instances,
                             GLint baseVertex) {
    ImageShadowScope shadows(*this);
    beginDraw(mode);
    glBaseVertex_ = baseVertex;
    if (count < 0 || instances < 0) {
        ctx_.setError(GL_INVALID_VALUE);
        return;
    }
    if (type != GL_UNSIGNED_BYTE && type != GL_UNSIGNED_SHORT && type != GL_UNSIGNED_INT) {
        ctx_.setError(GL_INVALID_ENUM);
        return;
    }
    if (!validateDraw(mode)) return;
    if (count == 0 || instances == 0 || !conditionalRenderAllows(ctx_)) return;
    Buffer* elementBuffer = ctx_.state.vertexArray->elementBuffer.get();
    if (elementBuffer && elementBuffer->mappedNonPersistent()) {
        ctx_.setError(GL_INVALID_OPERATION);
        return;
    }
    const size_t indexSize = type == GL_UNSIGNED_BYTE ? 1 : type == GL_UNSIGNED_SHORT ? 2 : 4;
    const uint8_t* source;
    const uintptr_t offset = reinterpret_cast<uintptr_t>(indices);
    if (elementBuffer) {
        if (!elementBuffer->mtl || offset + static_cast<size_t>(count) * indexSize > static_cast<size_t>(elementBuffer->size)) {
            // Out-of-range index fetches are undefined; draw nothing rather than read past the buffer.
            return;
        }
        source = elementBuffer->contents() + offset;  // read on the CPU only through cpuIndices()
    } else {
        // Client-memory indices: deprecated in core profiles but accepted, as
        // other desktop drivers do (the KHR-GL33 transform feedback tests use them).
        if (!indices) {
            ctx_.setError(GL_INVALID_OPERATION);
            return;
        }
        source = static_cast<const uint8_t*>(indices);
    }
    const ContextState& st = ctx_.state;
    const uint32_t restartIndex = st.primitiveRestartFixedIndex
                                      ? (type == GL_UNSIGNED_BYTE ? 0xffu : type == GL_UNSIGNED_SHORT ? 0xffffu : 0xffffffffu)
                                      : st.primitiveRestartIndex;
    const bool restart = st.primitiveRestart || st.primitiveRestartFixedIndex;
    // CPU reads of buffer indices wait for GPU writes to the buffer first.
    bool cpuReadable = !elementBuffer;
    auto cpuSource = [&] {
        if (!cpuReadable) {
            if (elementBuffer->lastWriteSerial) ctx_.stream().waitFor(elementBuffer->lastWriteSerial);
            cpuReadable = true;
        }
        return source;
    };
    auto index = [&](const uint8_t* p, GLsizei i) -> uint32_t {
        if (indexSize == 1) return p[i];
        if (indexSize == 2) { uint16_t x; std::memcpy(&x, p + i * 2, 2); return x; }
        uint32_t x;
        std::memcpy(&x, p + i * 4, 4);
        return x;
    };
    // The used vertex range (attribute conversion, renumbering) and restart
    // use: scanned on the CPU, cached per index range of an element buffer
    // until its contents change (not for persistently mapped buffers, which
    // the application writes behind GL's back).
    stats::Timer indexTimer(&stats::Frame::drawIndexNanos);  // the used range: kept, or scanned
    uint32_t minIndex = UINT32_MAX, maxIndex = 0;
    bool hasRestart = false;
    const Buffer::IndexRange range{offset, static_cast<uint64_t>(count), type, restart ? restartIndex : ~0ull};
    const Buffer::IndexScan* cached = nullptr;
    if (elementBuffer && !elementBuffer->mapped) {
        if (elementBuffer->indexScansVersion != elementBuffer->version) {
            elementBuffer->indexScans.clear();
            elementBuffer->indexScansVersion = elementBuffer->version;
        }
        auto it = elementBuffer->indexScans.find(range);
        if (it != elementBuffer->indexScans.end()) cached = &it->second;
    }
    if (cached) {
        minIndex = cached->minIndex;
        maxIndex = cached->maxIndex;
        hasRestart = cached->hasRestart;
    } else {
        const uint8_t* p = cpuSource();
        if (!restart) {
            // Without restart a plain min/max, which the compiler vectorizes.
            if (indexSize == 2) scanIndexRange<uint16_t>(p, count, minIndex, maxIndex);
            else if (indexSize == 4) scanIndexRange<uint32_t>(p, count, minIndex, maxIndex);
            else scanIndexRange<uint8_t>(p, count, minIndex, maxIndex);
        } else {
            for (GLsizei i = 0; i < count; ++i) {
                const uint32_t v = index(p, i);
                if (v == restartIndex) { hasRestart = true; continue; }
                minIndex = std::min(minIndex, v);
                maxIndex = std::max(maxIndex, v);
            }
        }
        if (elementBuffer && !elementBuffer->mapped) {
            auto& scans = elementBuffer->indexScans;
            if (scans.size() >= Buffer::kMaxIndexScans) scans.clear();
            scans[range] = {minIndex, maxIndex, hasRestart};
        }
    }
    indexTimer.switchTo(&stats::Frame::drawOtherNanos);
    if (minIndex == UINT32_MAX) return;  // only restart indices
    // The indices themselves, for the paths that rewrite them on the CPU.
    std::vector<uint32_t> values;
    auto cpuIndices = [&]() -> const std::vector<uint32_t>& {
        if (values.empty()) {
            const uint8_t* p = cpuSource();
            values.resize(static_cast<size_t>(count));
            for (GLsizei i = 0; i < count; ++i) values[static_cast<size_t>(i)] = index(p, i);
        }
        return values;
    };
    // Restart-delimited segments are combined into one list of independent
    // primitives, so every instance draws all of them before the next
    // instance starts (GL draws instance by instance).
    auto forEachSegment = [&](uint32_t bias, auto&& fn) {
        const std::vector<uint32_t>& values = cpuIndices();
        std::vector<uint32_t> segment;
        for (size_t i = 0; i <= values.size(); ++i) {
            if (i < values.size() && !(restart && values[i] == restartIndex)) {
                segment.push_back(values[i] + bias);
                continue;
            }
            if (!segment.empty()) fn(segment);
            segment.clear();
        }
    };
    const uint32_t bias = static_cast<uint32_t>(baseVertex);
    if (queriesCountingStatistics(ctx_)) {
        // Restart indices are not vertices; each run between them counts its primitives.
        const bool clipped = mode != GL_PATCHES && !executable()->link.geometry.active;
        if (hasRestart && mode != GL_PATCHES)
            forEachSegment(0, [&](const std::vector<uint32_t>& segment) {
                queriesCountSubmitted(ctx_, mode, static_cast<GLsizei>(segment.size()), instances, clipped);
            });
        else
            queriesCountSubmitted(ctx_, mode, count, instances, clipped);
    }
    if (ffWideLines_) {
        std::vector<uint32_t> lines;
        forEachSegment(bias, [&](const std::vector<uint32_t>& segment) {
            const std::vector<uint32_t> part = assembleForCapture(mode, segment);
            lines.insert(lines.end(), part.begin(), part.end());
        });
        drawWideLines(GL_LINES, lines, instances);
        return;
    }
    if (mode == GL_PATCHES) {
        // Primitive restart does not apply to patches
        // (PRIMITIVE_RESTART_FOR_PATCHES_SUPPORTED is false).
        std::vector<uint32_t> list = cpuIndices();
        for (uint32_t& v : list) v += bias;
        drawPatches(list, instances);
        return;
    }
    if (executable()->link.geometry.active) {
        // gl_PrimitiveIDIn keeps counting across restarts.
        uint32_t primitiveBase = 0;
        std::vector<uint32_t> assembled;
        forEachSegment(bias, [&](const std::vector<uint32_t>& segment) {
            const std::vector<uint32_t> part = assembleGeometryInput(mode, segment);
            assembled.insert(assembled.end(), part.begin(), part.end());
        });
        drawGeometry(mode, assembled, instances, primitiveBase, true);
        return;
    }
    if (ctx_.state.transformFeedback->active) {
        // gl_VertexID includes basevertex.
        std::vector<uint32_t> captured;
        forEachSegment(bias, [&](const std::vector<uint32_t>& segment) {
            const std::vector<uint32_t> part = assembleForCapture(mode, segment);
            captured.insert(captured.end(), part.begin(), part.end());
        });
        const Capture done = captured.empty() ? kNotCaptured : captureTransformFeedback(listMode(mode), captured, instances);
        if (done == kCapturedAndDrawn || (done == kCaptured && ctx_.state.rasterizerDiscard)) {
            queriesCountPrimitives(ctx_, listMode(mode), static_cast<GLsizei>(captured.size()), instances);
            return;
        }
    }
    const GLint spanFirst = static_cast<GLint>(minIndex) + baseVertex;
    const GLsizei span = static_cast<GLsizei>(std::min<uint64_t>(uint64_t{maxIndex} - minIndex + 1, INT32_MAX));
    bool needsIndices = false;
    nativePrimitive(mode, needsIndices);
    const bool rotate = rotateProvoking(mode, *executable());
    // Apple GPUs shade the 2x2 quads along an edge that two triangles of an
    // indexed draw share (the same vertices) together, so derivatives there
    // mix both triangles' attributes, and a texture lookup's level with them
    // (perspective-correct attributes differ across the edge, as with
    // projected texture coordinates). OpenGL ES draws whose per-vertex
    // attributes are all client memory, converted on the CPU anyway, are
    // drawn without indices instead, which keeps each triangle's quads apart.
    if (ctx_.isEs() && topologyClass(mode) == MTLPrimitiveTopologyClassTriangle && !pointPolygons(mode) &&
        !executable()->readsVertexId() && clientVertexArrays(*executable())) {
        std::vector<uint32_t> list;
        forEachSegment(bias, [&](const std::vector<uint32_t>& segment) { appendMetalList(mode, segment, rotate, list); });
        drawGathered(list, instances);
        return;
    }
    // Metal always restarts at the type's largest index; GL only when asked.
    const uint32_t metalRestart = indexSize == 2 ? 0xffffu : 0xffffffffu;
    const bool drawnAsIs = !hasRestart && !needsIndices && !rotate && indexSize != 1 && offset % indexSize == 0 &&
                           maxIndex != metalRestart;
    if (elementBuffer && !drawnAsIs && !elementBuffer->mapped && !elementBuffer->pinned && !pointPolygons(mode)) {
        // An element buffer range whose indices are rewritten (restart
        // segments, provoking-vertex rotation, fans, quads and loops, byte
        // indices, misaligned offsets) keeps its rewritten list until the
        // buffer changes: draws of an unchanged mesh read it on the GPU.
        if (elementBuffer->indexListsVersion != elementBuffer->version) {
            elementBuffer->indexLists.clear();
            elementBuffer->indexListsVersion = elementBuffer->version;
        }
        std::vector<uint64_t> key = {static_cast<uint64_t>(offset), static_cast<uint64_t>(count), type, mode,
                                     rotate ? 1u : 0u, restart ? uint64_t{restartIndex} : ~0ull};
        auto it = elementBuffer->indexLists.find(key);
        if (it == elementBuffer->indexLists.end()) {
            std::vector<uint32_t> list;
            forEachSegment(0, [&](const std::vector<uint32_t>& segment) { appendMetalList(mode, segment, rotate, list); });
            const size_t bytes = list.size() * 4;
            if (Buffer::convertedBytes() + bytes > Buffer::kMaxConvertedBytes) {
                if (!list.empty())
                    issue(listMode(mode), 0, static_cast<GLsizei>(list.size()), instances, list.data(), nil, 0,
                          MTLIndexTypeUInt32, baseVertex, spanFirst, span, true);
                return;
            }
            auto entry = std::make_unique<Buffer::ConvertedVertices>();
            if (bytes) {
                entry->mtl = [Device::get().mtl() newBufferWithBytes:list.data() length:bytes
                                                             options:MTLResourceStorageModeShared];
                if (!entry->mtl) return;
                entry->bytes = bytes;
                Buffer::convertedBytes() += bytes;
            }
            it = elementBuffer->indexLists.emplace(std::move(key), std::move(entry)).first;
        }
        const Buffer::ConvertedVertices& list = *it->second;
        if (list.bytes)
            issue(listMode(mode), 0, static_cast<GLsizei>(list.bytes / 4), instances, nullptr, list.mtl, 0,
                  MTLIndexTypeUInt32, baseVertex, spanFirst, span, true);
        return;
    }
    if (hasRestart) {
        std::vector<uint32_t> list;
        forEachSegment(0, [&](const std::vector<uint32_t>& segment) { appendMetalList(mode, segment, rotate, list); });
        if (!list.empty())
            issue(listMode(mode), 0, static_cast<GLsizei>(list.size()), instances, list.data(), nil, 0, MTLIndexTypeUInt32,
                  baseVertex, spanFirst, span, true);
        return;
    }
    if (elementBuffer && drawnAsIs && !pointPolygons(mode)) {
        elementBuffer->lastUseSerial = ctx_.stream().serial();
        issue(mode, 0, count, instances, nullptr, elementBuffer->mtl, offset,
              indexSize == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32, baseVertex, spanFirst, span);
        return;
    }
    issue(mode, 0, count, instances, cpuIndices().data(), nil, 0, MTLIndexTypeUInt32, baseVertex, spanFirst, span);
}

// ----------------------------------------------------------------- clears

void DrawState::clear(GLbitfield mask) {
    if (mask & ~(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | (ctx_.isCompat() ? GL_ACCUM_BUFFER_BIT : 0u))) {
        ctx_.setError(GL_INVALID_VALUE);
        return;
    }
    const ContextState& st = ctx_.state;
    if (st.rasterizerDiscard) return;
    RenderTargetDesc target;
    TargetFormats formats;
    if (!drawTarget(target, formats)) return;
    if (!conditionalRenderAllows(ctx_)) return;
    PendingClear clear;
    bool fullMasks = true;
    if (mask & GL_COLOR_BUFFER_BIT) {
        for (int i = 0; i < 8; ++i) {
            // GL leaves Clear of integer buffers undefined; like other
            // drivers, the float clear colour is converted to the integer type.
            if (!target.color[i].texture) continue;
            clear.colorMask |= 1u << i;
            double c[4] = {st.clearColor[0], st.clearColor[1], st.clearColor[2], formats.alphaIsOne[i] ? 1.0 : st.clearColor[3]};
            if (formats.integer[i] && st.clearColorIntegerSet)  // GL_EXT_texture_integer
                for (int k = 0; k < 4; ++k) c[k] = st.clearColorInteger[k];
            clear.color[i] = MTLClearColorMake(c[0], c[1], c[2], c[3]);
            const auto& m = st.colorMask[i];
            if (!m[0] || !m[1] || !m[2] || (!m[3] && !formats.alphaIsOne[i])) fullMasks = false;
        }
    }
    if ((mask & GL_DEPTH_BUFFER_BIT) && formats.depthAttached) {
        clear.depth = true;
        clear.depthValue = std::clamp(st.clearDepth, 0.0, 1.0);
        if (!st.depthMask) fullMasks = false;
    }
    if ((mask & GL_STENCIL_BUFFER_BIT) && formats.stencilAttached) {
        clear.stencil = true;
        clear.stencilValue = static_cast<uint32_t>(st.clearStencil) & 0xff;
        if ((st.stencilFront.writeMask & 0xff) != 0xff) fullMasks = false;
    }
    if (!clear.any()) return;
    bool fullRect = true;
    if ((st.scissorTests & 1u)) {
        fullRect = st.scissors[0][0] <= 0 && st.scissors[0][1] <= 0 &&
                   static_cast<int64_t>(st.scissors[0][0]) + st.scissors[0][2] >= static_cast<int64_t>(target.width) &&
                   static_cast<int64_t>(st.scissors[0][1]) + st.scissors[0][3] >= static_cast<int64_t>(target.height);
    }
    if (fullMasks && fullRect && ctx_.stream().clearWithLoadAction(target, clear)) return;
    clearWithDraw(target, formats, clear);
}

namespace {

// Clear pipelines: the fragment shader's outputs are typed per attachment
// (float, int or uint) to match the pass; values arrive as raw 32-bit words.
// `layered`: one instance per layer of a layered framebuffer.
id<MTLRenderPipelineState> clearPipeline(const TargetFormats& formats, const uint8_t writeMasks[8], uint32_t samples,
                                         const char signature[9], bool layered) {
    static std::mutex mutex;
    static std::unordered_map<std::string, id<MTLRenderPipelineState>> cache;
    std::string key(signature, 8);
    key += layered ? "L" : "-";
    for (int i = 0; i < 8; ++i) key += std::to_string(formats.color[i]) + ":" + std::to_string(writeMasks[i]) + ",";
    key += std::to_string(formats.depth) + "," + std::to_string(formats.stencil) + "," + std::to_string(samples);
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    std::string source = layered ? "#include <metal_stdlib>\nusing namespace metal;\n"
                                   "struct V { float4 position [[position]]; uint layer [[render_target_array_index]]; };\n"
                                   "vertex V clear_vs(uint vid [[vertex_id]], uint iid [[instance_id]], constant float4& depth [[buffer(0)]]) {\n"
                                   "  float2 p = float2((vid << 1) & 2, vid & 2); V out; out.position = float4(p * 2.0 - 1.0, depth.x, 1.0);\n"
                                   "  out.layer = iid; return out; }\n"
                                 : "#include <metal_stdlib>\nusing namespace metal;\n"
                                   "struct V { float4 position [[position]]; };\n"
                                   "vertex V clear_vs(uint vid [[vertex_id]], constant float4& depth [[buffer(0)]]) {\n"
                                   "  float2 p = float2((vid << 1) & 2, vid & 2); V out; out.position = float4(p * 2.0 - 1.0, depth.x, 1.0); return out; }\n";
    std::string fields, body;
    for (int i = 0; i < 8; ++i) {
        if (signature[i] == '-') continue;
        const char* type = signature[i] == 'i' ? "int4" : signature[i] == 'u' ? "uint4" : "float4";
        fields += std::string(type) + " c" + std::to_string(i) + " [[color(" + std::to_string(i) + ")]]; ";
        body += "o.c" + std::to_string(i) + " = as_type<" + type + ">(v[" + std::to_string(i) + "]); ";
    }
    if (fields.empty()) {
        source += "fragment void clear_fs() {}\n";
    } else {
        source += "struct O { " + fields + "};\n"
                  "fragment O clear_fs(constant uint4* v [[buffer(0)]]) { O o; " + body + "return o; }\n";
    }
    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_0;
    id<MTLLibrary> library = [Device::get().mtl() newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                               options:options
                                                                 error:&error];
    if (!library) {
        WMGL_ERR(Pipeline, "clear shader: %s", error.localizedDescription.UTF8String);
        return cache[key] = nil;
    }
    MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = [library newFunctionWithName:@"clear_vs"];
    desc.fragmentFunction = [library newFunctionWithName:@"clear_fs"];
    desc.rasterSampleCount = samples;
    if (layered) desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    for (int i = 0; i < 8; ++i) {
        desc.colorAttachments[i].pixelFormat = formats.color[i];
        desc.colorAttachments[i].writeMask = writeMasks[i];
    }
    desc.depthAttachmentPixelFormat = formats.depth;
    desc.stencilAttachmentPixelFormat = formats.stencil;
    id<MTLRenderPipelineState> pipeline = [Device::get().mtl() newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!pipeline) WMGL_ERR(Pipeline, "clear pipeline: %s", error.localizedDescription.UTF8String);
    return cache[key] = pipeline;
}

}  // namespace

void DrawState::clearWithDraw(const RenderTargetDesc& target, const TargetFormats& formats, const PendingClear& clear) {
    const ContextState& st = ctx_.state;
    MTLScissorRect scissor = {0, 0, target.width, target.height};
    if ((st.scissorTests & 1u)) {
        const int64_t x0 = std::clamp<int64_t>(st.scissors[0][0], 0, target.width);
        const int64_t y0 = std::clamp<int64_t>(st.scissors[0][1], 0, target.height);
        const int64_t x1 = std::clamp<int64_t>(static_cast<int64_t>(st.scissors[0][0]) + std::max(0, st.scissors[0][2]), 0, target.width);
        const int64_t y1 = std::clamp<int64_t>(static_cast<int64_t>(st.scissors[0][1]) + std::max(0, st.scissors[0][3]), 0, target.height);
        if (x1 <= x0 || y1 <= y0) return;
        scissor = {static_cast<NSUInteger>(x0), static_cast<NSUInteger>(y0), static_cast<NSUInteger>(x1 - x0),
                   static_cast<NSUInteger>(y1 - y0)};
    }
    // Per-attachment output type, write mask and raw clear value.
    char signature[9] = "--------";
    uint8_t writeMasks[8] = {};
    uint32_t values[8][4] = {};
    for (int i = 0; i < 8; ++i) {
        if (!formats.color[i]) continue;
        const MTLPixelFormat f = formats.color[i];
        const bool sint = f == MTLPixelFormatR8Sint || f == MTLPixelFormatR16Sint || f == MTLPixelFormatR32Sint ||
                          f == MTLPixelFormatRG8Sint || f == MTLPixelFormatRG16Sint || f == MTLPixelFormatRG32Sint ||
                          f == MTLPixelFormatRGBA8Sint || f == MTLPixelFormatRGBA16Sint || f == MTLPixelFormatRGBA32Sint;
        signature[i] = formats.integer[i] ? (sint ? 'i' : 'u') : 'f';
        const MTLClearColor& c = clear.color[i];
        const double v[4] = {c.red, c.green, c.blue, c.alpha};
        for (int k = 0; k < 4; ++k) {
            if (signature[i] == 'f') {
                const float x = static_cast<float>(v[k]);
                std::memcpy(&values[i][k], &x, 4);
            } else if (signature[i] == 'i') {
                const int32_t x = static_cast<int32_t>(v[k]);
                std::memcpy(&values[i][k], &x, 4);
            } else {
                values[i][k] = static_cast<uint32_t>(v[k]);
            }
        }
        const auto& m = st.colorMask[i];
        const bool cleared = clear.colorMask & (1u << i);
        writeMasks[i] = !cleared ? MTLColorWriteMaskNone
                                 : static_cast<uint8_t>((m[0] ? MTLColorWriteMaskRed : 0) | (m[1] ? MTLColorWriteMaskGreen : 0) |
                                                        (m[2] ? MTLColorWriteMaskBlue : 0) |
                                                        (m[3] && !formats.alphaIsOne[i] ? MTLColorWriteMaskAlpha : 0));
    }
    id<MTLRenderPipelineState> pipeline = clearPipeline(formats, writeMasks, target.samples, signature, target.layers > 0);
    if (!pipeline) return;
    id<MTLRenderCommandEncoder> encoder = ctx_.stream().render(target);
    queriesSuspendVisibility(ctx_, encoder);  // clears do not count towards occlusion queries
    DepthStencilKey ds;
    std::memset(&ds, 0, sizeof(ds));
    ds.depthCompare = static_cast<uint8_t>(MTLCompareFunctionAlways);
    ds.depthWrite = clear.depth && st.depthMask;
    ds.stencilEnabled = clear.stencil;
    if (clear.stencil) {
        for (DepthStencilKey::Face* f : {&ds.front, &ds.back}) {
            f->compare = static_cast<uint8_t>(MTLCompareFunctionAlways);
            f->fail = f->depthFail = f->pass = static_cast<uint8_t>(MTLStencilOperationReplace);
            f->readMask = 0xff;
            f->writeMask = st.stencilFront.writeMask & 0xff;
        }
    }
    [encoder setRenderPipelineState:pipeline];
    encoderState_.pipeline = nil;  // issue() sets its own again
    [encoder setDepthStencilState:ctx_.share().pipelines().depthStencil(ds)];
    MTLViewport viewport = {0, 0, static_cast<double>(target.width), static_cast<double>(target.height), 0, 1};
    [encoder setViewport:viewport];
    [encoder setScissorRect:scissor];
    [encoder setCullMode:MTLCullModeNone];
    [encoder setTriangleFillMode:MTLTriangleFillModeFill];
    [encoder setDepthClipMode:MTLDepthClipModeClamp];
    [encoder setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];  // the encoder may carry a draw's polygon offset
    [encoder setStencilReferenceValue:clear.stencilValue];
    const float depth[4] = {static_cast<float>(clear.depthValue), 0, 0, 0};
    [encoder setVertexBytes:depth length:sizeof(depth) atIndex:0];
    [encoder setFragmentBytes:values length:sizeof(values) atIndex:0];
    // A layered framebuffer clears every layer (GL 4.x section 9.8).
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:std::max(1u, target.layers)];
    ctx_.stream().noteDraw();
}

void DrawState::clearBuffer(GLenum buffer, GLint drawbuffer, const double color[4], double depthValue,
                            GLint stencilValue) {
    const ContextState& st = ctx_.state;
    RenderTargetDesc target;
    TargetFormats formats;
    if (!drawTarget(target, formats)) return;
    if (st.rasterizerDiscard || !conditionalRenderAllows(ctx_)) return;
    PendingClear clear;
    bool fullMasks = true;
    if (buffer == GL_COLOR) {
        if (!target.color[drawbuffer].texture) return;
        clear.colorMask = static_cast<uint8_t>(1u << drawbuffer);
        clear.color[drawbuffer] = MTLClearColorMake(color[0], color[1], color[2], formats.alphaIsOne[drawbuffer] ? 1.0 : color[3]);
        const auto& m = st.colorMask[drawbuffer];
        if (!m[0] || !m[1] || !m[2] || (!m[3] && !formats.alphaIsOne[drawbuffer])) fullMasks = false;
    }
    if ((buffer == GL_DEPTH || buffer == GL_DEPTH_STENCIL) && formats.depthAttached) {
        clear.depth = true;
        clear.depthValue = std::clamp(depthValue, 0.0, 1.0);
        if (!st.depthMask) fullMasks = false;
    }
    if ((buffer == GL_STENCIL || buffer == GL_DEPTH_STENCIL) && formats.stencilAttached) {
        clear.stencil = true;
        clear.stencilValue = static_cast<uint32_t>(stencilValue) & 0xff;
        if ((st.stencilFront.writeMask & 0xff) != 0xff) fullMasks = false;
    }
    if (!clear.any()) return;
    bool fullRect = true;
    if ((st.scissorTests & 1u))
        fullRect = st.scissors[0][0] <= 0 && st.scissors[0][1] <= 0 &&
                   static_cast<int64_t>(st.scissors[0][0]) + st.scissors[0][2] >= static_cast<int64_t>(target.width) &&
                   static_cast<int64_t>(st.scissors[0][1]) + st.scissors[0][3] >= static_cast<int64_t>(target.height);
    if (fullMasks && fullRect && ctx_.stream().clearWithLoadAction(target, clear)) return;
    clearWithDraw(target, formats, clear);
}

}  // namespace wmgl
