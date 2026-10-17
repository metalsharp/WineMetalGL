// Framebuffer and renderbuffer objects, draw/read buffers, ReadPixels
// (GL 3.3 sections 4.2-4.4).

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <set>
#include <cstring>
#include <vector>

#include "core/context.h"
#include "core/stats.h"
#include "core/draw.h"
#include "core/format/es_formats.h"
#include "core/format/pixel_transfer.h"
#include "core/limits.h"
#include "core/metal/device.h"
#include "core/metal/pixel_format.h"
#include "core/texture_storage.h"
#include "core/util/log.h"
#include "core/api/read_format.h"
#include "gl_api.gen.h"

namespace wmgl::api {
namespace {

RefPtr<Framebuffer>* framebufferBinding(Context& ctx, GLenum target) {
    switch (target) {
    case GL_DRAW_FRAMEBUFFER: case GL_FRAMEBUFFER: return &ctx.state.drawFramebuffer;
    case GL_READ_FRAMEBUFFER: return &ctx.state.readFramebuffer;
    default: return nullptr;
    }
}

Attachment* attachmentPoint(Framebuffer& fb, GLenum attachment) {
    if (attachment >= GL_COLOR_ATTACHMENT0 && attachment < GL_COLOR_ATTACHMENT0 + Framebuffer::kMaxColor)
        return &fb.color[attachment - GL_COLOR_ATTACHMENT0];
    if (attachment == GL_DEPTH_ATTACHMENT) return &fb.depth;
    if (attachment == GL_STENCIL_ATTACHMENT) return &fb.stencil;
    return nullptr;
}

const FormatInfo* attachmentFormat(const Attachment& a, GLsizei& width, GLsizei& height, GLsizei& samples, bool& fixedLocations) {
    fixedLocations = true;
    if (a.type == GL_RENDERBUFFER && a.renderbuffer) {
        width = a.renderbuffer->width;
        height = a.renderbuffer->height;
        samples = a.renderbuffer->samples;
        return a.renderbuffer->format;
    }
    if (a.type == GL_TEXTURE && a.texture) {
        const int face = a.cubeFace ? static_cast<int>(a.cubeFace - GL_TEXTURE_CUBE_MAP_POSITIVE_X) : 0;
        if (a.level < 0 || a.level >= Texture::kMaxLevels) return nullptr;
        const TextureImage& image = a.texture->image(face, a.level);
        width = image.width;
        height = a.texture->target == GL_TEXTURE_1D_ARRAY ? 1 : image.height;  // 1D array: height counts layers
        samples = image.samples;
        // Single-sample images (and renderbuffers) have fixed locations.
        fixedLocations = image.samples > 0 ? image.fixedSampleLocations : true;
        return image.format;
    }
    return nullptr;
}

// A cube map's six faces at `level` defined alike (square, one size and format).
bool cubeComplete(const Texture& tex, int level) {
    const TextureImage& first = tex.image(0, level);
    if (!first.defined() || first.width != first.height) return false;
    for (int face = 1; face < 6; ++face) {
        const TextureImage& image = tex.image(face, level);
        if (!image.defined() || image.width != first.width || image.height != first.height ||
            image.internalFormat != first.internalFormat)
            return false;
    }
    return true;
}

// Mipmap complete (OpenGL ES 3.2 section 8.17): the levels from the base to
// the last of the base's chain (or MAX_LEVEL) are defined, each half the
// one before, all of one format; for cube maps, every face.
bool mipmapComplete(const Texture& tex) {
    const int faces = tex.target == GL_TEXTURE_CUBE_MAP ? 6 : 1;
    const int base = tex.baseLevel;
    if (base < 0 || base >= Texture::kMaxLevels) return false;
    const TextureImage& first = tex.image(0, base);
    if (!first.defined()) return false;
    const bool threeD = tex.target == GL_TEXTURE_3D;
    int size = std::max(first.width, first.height);
    if (threeD) size = std::max(size, first.depth);
    int last = base;
    while ((size >> (last - base)) > 1) ++last;
    last = std::min({last, tex.maxLevel, Texture::kMaxLevels - 1});
    for (int level = base; level <= last; ++level)
        for (int face = 0; face < faces; ++face) {
            const TextureImage& image = tex.image(face, level);
            const int shift = level - base;
            if (!image.defined() || image.internalFormat != first.internalFormat ||
                image.width != std::max(1, first.width >> shift) || image.height != std::max(1, first.height >> shift) ||
                (threeD && image.depth != std::max(1, first.depth >> shift)) || (!threeD && image.depth != first.depth))
                return false;
        }
    return true;
}

GLenum completeness(Context& ctx, Framebuffer* fb) {
    if (!fb) return ctx.drawSurface() ? GL_FRAMEBUFFER_COMPLETE : GL_FRAMEBUFFER_UNDEFINED;
    bool any = false;
    GLsizei samples = -1;
    int fixed = -1;
    int layered = -1;
    GLenum layerTarget = 0;
    auto check = [&](const Attachment& a, int kind) -> GLenum {
        if (!a.attached()) return GL_FRAMEBUFFER_COMPLETE;
        GLsizei w = 0, h = 0, s = 0;
        bool fixedLocations = true;
        const FormatInfo* f = attachmentFormat(a, w, h, s, fixedLocations);
        if (!f || w <= 0 || h <= 0) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        if (kind == 0 && !f->colorRenderable) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        // OpenGL ES's color-renderable formats are fewer (no RGB32F, RGB9_E5,
        // luminance or RGB8_SNORM).
        if (kind == 0 && ctx.isEs() && !es::colorRenderable(f->internalFormat)) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        // OES_texture_float's unsized float images are not color-renderable
        // (EXT_color_buffer_float makes the sized formats renderable), nor
        // are EXT_texture_type_2_10_10_10_REV's unsized RGBA images.
        if (kind == 0 && ctx.isEs() && a.type == GL_TEXTURE && a.texture &&
            ((f->isFloat() && f->redBits == 32) || f->internalFormat == GL_RGB10_A2)) {
            const int face = a.cubeFace ? static_cast<int>(a.cubeFace - GL_TEXTURE_CUBE_MAP_POSITIVE_X) : 0;
            const GLenum requested = a.texture->image(face, a.level).requestedFormat;
            if (requested == GL_RGBA || requested == GL_RGB || requested == GL_RED || requested == GL_RG)
                return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        }
        if (kind == 1 && !f->depthRenderable) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        if (kind == 2 && !f->stencilRenderable) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        // OpenGL ES 3.2 section 9.4.1: a cube map attached layered, or at a
        // level other than a mutable texture's base level, is cube complete.
        if (ctx.isEs() && a.type == GL_TEXTURE && a.texture && a.texture->target == GL_TEXTURE_CUBE_MAP &&
            (a.layered || (!a.texture->immutable && a.level != a.texture->baseLevel)) && !cubeComplete(*a.texture, a.level))
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        // ... and any level other than a mutable texture's base level needs
        // the texture mipmap complete.
        if (ctx.isEs() && a.type == GL_TEXTURE && a.texture && !a.texture->immutable && a.level != a.texture->baseLevel &&
            !mipmapComplete(*a.texture))
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        if (a.type == GL_TEXTURE && a.texture) {
            const GLenum t = a.texture->target;
            if ((t == GL_TEXTURE_3D || t == GL_TEXTURE_2D_ARRAY || t == GL_TEXTURE_1D_ARRAY ||
                 t == GL_TEXTURE_2D_MULTISAMPLE_ARRAY || t == GL_TEXTURE_CUBE_MAP_ARRAY) &&
                !a.layered) {
                const TextureImage& image = a.texture->image(0, a.level);
                const GLsizei layers = t == GL_TEXTURE_1D_ARRAY ? image.height : image.depth;
                if (a.layer >= layers) return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
            }
        }
        any = true;
        const GLsizei normalizedSamples = std::max(0, s);
        if (samples < 0) samples = normalizedSamples;
        else if (samples != normalizedSamples) return GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE;
        if (fixed < 0) fixed = fixedLocations;
        else if (fixed != static_cast<int>(fixedLocations)) return GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE;
        const int isLayered = a.layered ? 1 : 0;
        if (layered < 0) {
            layered = isLayered;
            layerTarget = a.texture ? a.texture->target : 0;
        } else if (layered != isLayered) {
            return GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS;
        } else if (isLayered && a.texture && a.texture->target != layerTarget) {
            return GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS;
        }
        return GL_FRAMEBUFFER_COMPLETE;
    };
    for (const Attachment& a : fb->color)
        if (GLenum status = check(a, 0); status != GL_FRAMEBUFFER_COMPLETE) return status;
    if (GLenum status = check(fb->depth, 1); status != GL_FRAMEBUFFER_COMPLETE) return status;
    if (GLenum status = check(fb->stencil, 2); status != GL_FRAMEBUFFER_COMPLETE) return status;
    // GL 4.3: without attachments, the default width and height must be set.
    if (!any && !(ctx.version() >= 43 && fb->defaultWidth > 0 && fb->defaultHeight > 0))
        return GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;
    // GL_ARB_ES2_compatibility (advertised in every context, core in 4.1)
    // removes the INCOMPLETE_DRAW_BUFFER / INCOMPLETE_READ_BUFFER rules:
    // draw and read buffers naming empty attachments are simply ignored.
    // Metal: a combined depth/stencil image must be bound to both points as the same image.
    if (fb->depth.attached() && fb->stencil.attached()) {
        GLsizei w, h, s;
        bool fl;
        const FormatInfo* d = attachmentFormat(fb->depth, w, h, s, fl);
        const FormatInfo* st = attachmentFormat(fb->stencil, w, h, s, fl);
        const bool sameImage = fb->depth.type == fb->stencil.type && fb->depth.texture == fb->stencil.texture &&
                               fb->depth.renderbuffer == fb->stencil.renderbuffer && fb->depth.level == fb->stencil.level &&
                               fb->depth.layer == fb->stencil.layer && fb->depth.cubeFace == fb->stencil.cubeFace;
        if (d && st && (d->isStencil() || st->isDepth()) && !sameImage) return GL_FRAMEBUFFER_UNSUPPORTED;
    }
    return GL_FRAMEBUFFER_COMPLETE;
}

void framebufferChanged(Context& ctx, Framebuffer& fb) {
    ++fb.version;
    ctx.markDirty(kDirtyFramebuffer);
}

// `missing`: the error for a name that is not a texture (glFramebufferTexture
// raises INVALID_VALUE, KHR-GL44.geometry_shader.layered_fbo.fb_texture_invalid_value).
RefPtr<Texture> textureForAttach(Context& ctx, GLuint name, bool& ok, GLenum missing = GL_INVALID_OPERATION) {
    ok = true;
    if (!name) return nullptr;
    ShareLock lock(ctx);
    RefPtr<Texture> tex = ctx.share().textures.getRef(name);
    if (!tex) {
        ctx.setError(missing);
        ok = false;
    }
    return tex;
}

void attachTexture(Context& ctx, GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level,
                   GLint layer, bool layered, bool layerCall) {
    RefPtr<Framebuffer>* binding = framebufferBinding(ctx, target);
    if (!binding) return ctx.setError(GL_INVALID_ENUM);
    Framebuffer* fb = binding->get();
    if (!fb) return ctx.setError(GL_INVALID_OPERATION);
    const bool depthStencil = attachment == GL_DEPTH_STENCIL_ATTACHMENT;
    Attachment* point = depthStencil ? &fb->depth : attachmentPoint(*fb, attachment);
    if (!point) return ctx.setError(attachment >= GL_COLOR_ATTACHMENT0 && attachment <= GL_COLOR_ATTACHMENT0 + 31
                                        ? GL_INVALID_OPERATION
                                        : GL_INVALID_ENUM);
    bool ok;
    const bool framebufferTexture = !textarget && !layerCall;
    RefPtr<Texture> tex = textureForAttach(ctx, texture, ok, framebufferTexture ? GL_INVALID_VALUE : GL_INVALID_OPERATION);
    if (!ok) return;
    Attachment a;
    if (tex) {
        if (tex->target == GL_TEXTURE_BUFFER) return ctx.setError(GL_INVALID_OPERATION);
        if (level < 0 || level >= Texture::kMaxLevels) return ctx.setError(GL_INVALID_VALUE);
        // OpenGL ES 3.2 section 9.2.8: a level past log2 of the largest size.
        if (ctx.isEs() && level > (tex->target == GL_TEXTURE_3D ? 11 : 14)) return ctx.setError(GL_INVALID_VALUE);
        // An immutable texture has only its levels.
        if (tex->immutable && tex->immutableLevels > 0 && level >= tex->immutableLevels) return ctx.setError(GL_INVALID_VALUE);
        if (textarget) {
            const bool cubeFace = textarget >= GL_TEXTURE_CUBE_MAP_POSITIVE_X && textarget <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;
            const GLenum expect = cubeFace ? GL_TEXTURE_CUBE_MAP : textarget;
            if (tex->target != expect) return ctx.setError(GL_INVALID_OPERATION);
            if (cubeFace) a.cubeFace = textarget;
            if ((textarget == GL_TEXTURE_RECTANGLE || textarget == GL_TEXTURE_2D_MULTISAMPLE) && level != 0)
                return ctx.setError(GL_INVALID_VALUE);
        } else if (layerCall) {
            const GLenum t = tex->target;
            // GL 4.5: a cube map's layer is a face.
            const bool cube = t == GL_TEXTURE_CUBE_MAP && ctx.version() >= 45;
            if (t != GL_TEXTURE_3D && t != GL_TEXTURE_2D_ARRAY && t != GL_TEXTURE_1D_ARRAY && t != GL_TEXTURE_2D_MULTISAMPLE_ARRAY &&
                t != GL_TEXTURE_CUBE_MAP_ARRAY && !cube)
                return ctx.setError(GL_INVALID_OPERATION);
            if (layer < 0 || (cube && layer >= 6)) return ctx.setError(GL_INVALID_VALUE);
            // A multisample array has level 0 only.
            if (t == GL_TEXTURE_2D_MULTISAMPLE_ARRAY && level != 0) return ctx.setError(GL_INVALID_VALUE);
            if (cube) a.cubeFace = GL_TEXTURE_CUBE_MAP_POSITIVE_X + static_cast<GLenum>(layer);
            // A layer past the target's limit (MAX_3D_TEXTURE_SIZE or MAX_ARRAY_TEXTURE_LAYERS).
            if (layer >= (t == GL_TEXTURE_3D ? limits::kMax3DTextureSize : limits::kMaxArrayLayers))
                return ctx.setError(GL_INVALID_VALUE);
        }
        a.type = GL_TEXTURE;
        a.texture = tex;
        a.level = level;
        a.layer = layer;
        a.layered = layered && (tex->target == GL_TEXTURE_3D || tex->target == GL_TEXTURE_2D_ARRAY ||
                                tex->target == GL_TEXTURE_CUBE_MAP || tex->target == GL_TEXTURE_1D_ARRAY ||
                                tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY || tex->target == GL_TEXTURE_CUBE_MAP_ARRAY);
    }
    if (depthStencil) {
        fb->depth = a;
        fb->stencil = a;
    } else {
        *point = a;
    }
    framebufferChanged(ctx, *fb);
}

}  // namespace

// ---------------------------------------------------- framebuffer objects

void GenFramebuffers(Context& ctx, GLsizei n, GLuint* framebuffers) {
    if (n < 0) return ctx.setError(GL_INVALID_VALUE);
    if (!framebuffers) return;
    ShareLock lock(ctx);
    ctx.share().framebuffers.generate(n, framebuffers);
}

void DeleteFramebuffers(Context& ctx, GLsizei n, const GLuint* framebuffers) {
    if (n < 0) return ctx.setError(GL_INVALID_VALUE);
    if (!framebuffers) return;
    for (GLsizei i = 0; i < n; ++i) {
        if (!framebuffers[i]) continue;
        RefPtr<Framebuffer> fb;
        {
            ShareLock lock(ctx);
            fb = ctx.share().framebuffers.remove(framebuffers[i]);
        }
        if (!fb) continue;
        if (ctx.state.drawFramebuffer == fb) ctx.state.drawFramebuffer.reset();
        if (ctx.state.readFramebuffer == fb) ctx.state.readFramebuffer.reset();
        ctx.markDirty(kDirtyFramebuffer);
    }
}

GLboolean IsFramebuffer(Context& ctx, GLuint framebuffer) {
    ShareLock lock(ctx);
    return ctx.share().framebuffers.exists(framebuffer) ? GL_TRUE : GL_FALSE;
}

namespace {

// `createUnused`: EXT_framebuffer_object (and compatibility contexts) create
// objects for names GenFramebuffers did not return.
void bindFramebuffer(Context& ctx, GLenum target, GLuint name, bool createUnused) {
    if (target != GL_FRAMEBUFFER && target != GL_DRAW_FRAMEBUFFER && target != GL_READ_FRAMEBUFFER)
        return ctx.setError(GL_INVALID_ENUM);
    RefPtr<Framebuffer> fb;
    if (name) {
        ShareLock lock(ctx);
        NameTable<Framebuffer>& table = ctx.share().framebuffers;
        if (!table.isGenerated(name)) {
            if (!createUnused) return ctx.setError(GL_INVALID_OPERATION);
            table.reserve(name);
        }
        fb = table.getRef(name);
        if (!fb) {
            fb = makeRef<Framebuffer>(name);
            table.set(name, fb);
        }
    }
    if (target != GL_READ_FRAMEBUFFER) ctx.state.drawFramebuffer = fb;
    if (target != GL_DRAW_FRAMEBUFFER) ctx.state.readFramebuffer = fb;
    ctx.markDirty(kDirtyFramebuffer);
}

void bindRenderbuffer(Context& ctx, GLenum target, GLuint name, bool createUnused) {
    if (target != GL_RENDERBUFFER) return ctx.setError(GL_INVALID_ENUM);
    if (!name) {
        ctx.state.renderbuffer.reset();
        return;
    }
    ShareLock lock(ctx);
    RefPtr<Renderbuffer> rb = ctx.share().renderbuffers.getRef(name);
    if (!rb) {
        if (!ctx.share().renderbuffers.isGenerated(name)) {
            if (!createUnused) return ctx.setError(GL_INVALID_OPERATION);
            ctx.share().renderbuffers.reserve(name);
        }
        rb = makeRef<Renderbuffer>(name);
        ctx.share().renderbuffers.set(name, rb);
    }
    ctx.state.renderbuffer = rb;
}

}  // namespace

void BindFramebuffer(Context& ctx, GLenum target, GLuint name) { bindFramebuffer(ctx, target, name, !ctx.isDesktopCore()); }

void BindFramebufferEXT(Context& ctx, GLenum target, GLuint name) { bindFramebuffer(ctx, target, name, true); }

GLenum CheckFramebufferStatus(Context& ctx, GLenum target) {
    RefPtr<Framebuffer>* binding = framebufferBinding(ctx, target);
    if (!binding) {
        ctx.setError(GL_INVALID_ENUM);
        return 0;
    }
    const GLenum status = completeness(ctx, binding->get());
    if (status != GL_FRAMEBUFFER_COMPLETE && logEnabled(LogLevel::Trace, LogCategory::Draw)) {
        GLsizei w = 0, h = 0, s = 0;
        bool fl;
        const Framebuffer* fb = binding->get();
        const FormatInfo* f = fb ? attachmentFormat(fb->color[0], w, h, s, fl) : nullptr;
        WMGL_TRACE(Draw, "framebuffer %u incomplete: 0x%04x (color 0: format 0x%04x %dx%d samples %d)",
                   fb ? fb->name : 0,
                   status, f ? f->internalFormat : 0, w, h, s);
    }
    return status;
}

void FramebufferTexture(Context& ctx, GLenum target, GLenum attachment, GLuint texture, GLint level) {
    attachTexture(ctx, target, attachment, 0, texture, level, 0, true, false);
}

void FramebufferTexture1D(Context& ctx, GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
    if (texture && textarget != GL_TEXTURE_1D) return ctx.setError(GL_INVALID_OPERATION);
    attachTexture(ctx, target, attachment, texture ? textarget : 0, texture, level, 0, false, false);
}

void FramebufferTexture2D(Context& ctx, GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {
    if (texture && textarget != GL_TEXTURE_2D && textarget != GL_TEXTURE_RECTANGLE && textarget != GL_TEXTURE_2D_MULTISAMPLE &&
        !(textarget >= GL_TEXTURE_CUBE_MAP_POSITIVE_X && textarget <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z))
        return ctx.setError(GL_INVALID_ENUM);
    attachTexture(ctx, target, attachment, texture ? textarget : 0, texture, level, 0, false, false);
}

void FramebufferTexture3D(Context& ctx, GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level,
                          GLint zoffset) {
    if (texture && textarget != GL_TEXTURE_3D) return ctx.setError(GL_INVALID_OPERATION);
    if (texture && (zoffset < 0 || zoffset >= limits::kMax3DTextureSize)) return ctx.setError(GL_INVALID_VALUE);
    attachTexture(ctx, target, attachment, texture ? textarget : 0, texture, level, zoffset, false, false);
}

void FramebufferTextureLayer(Context& ctx, GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer) {
    attachTexture(ctx, target, attachment, 0, texture, level, layer, false, true);
}

void FramebufferRenderbuffer(Context& ctx, GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) {
    RefPtr<Framebuffer>* binding = framebufferBinding(ctx, target);
    if (!binding) return ctx.setError(GL_INVALID_ENUM);
    if (renderbuffertarget != GL_RENDERBUFFER) return ctx.setError(GL_INVALID_ENUM);
    Framebuffer* fb = binding->get();
    if (!fb) return ctx.setError(GL_INVALID_OPERATION);
    const bool depthStencil = attachment == GL_DEPTH_STENCIL_ATTACHMENT;
    Attachment* point = depthStencil ? &fb->depth : attachmentPoint(*fb, attachment);
    if (!point) return ctx.setError(GL_INVALID_ENUM);
    Attachment a;
    if (renderbuffer) {
        ShareLock lock(ctx);
        RefPtr<Renderbuffer> rb = ctx.share().renderbuffers.getRef(renderbuffer);
        if (!rb) return ctx.setError(GL_INVALID_OPERATION);
        a.type = GL_RENDERBUFFER;
        a.renderbuffer = rb;
    }
    if (depthStencil) {
        fb->depth = a;
        fb->stencil = a;
    } else {
        *point = a;
    }
    framebufferChanged(ctx, *fb);
}

// GL 4.3 / GL_ARB_framebuffer_no_attachments.
void FramebufferParameteri(Context& ctx, GLenum target, GLenum pname, GLint param) {
    if (ctx.version() < 43 && !ctx.hasExtension("GL_ARB_framebuffer_no_attachments"))
        return ctx.unimplemented("glFramebufferParameteri");
    RefPtr<Framebuffer>* binding = framebufferBinding(ctx, target);
    if (!binding) return ctx.setError(GL_INVALID_ENUM);
    Framebuffer* fb = binding->get();
    if (!fb) return ctx.setError(GL_INVALID_OPERATION);
    switch (pname) {
    case GL_FRAMEBUFFER_DEFAULT_WIDTH: case GL_FRAMEBUFFER_DEFAULT_HEIGHT:
        if (param < 0 || param > limits::kMaxTextureSize) return ctx.setError(GL_INVALID_VALUE);
        (pname == GL_FRAMEBUFFER_DEFAULT_WIDTH ? fb->defaultWidth : fb->defaultHeight) = param;
        break;
    case GL_FRAMEBUFFER_DEFAULT_LAYERS:
        if (param < 0 || param > limits::kMaxArrayLayers) return ctx.setError(GL_INVALID_VALUE);
        fb->defaultLayers = param;
        break;
    case GL_FRAMEBUFFER_DEFAULT_SAMPLES:
        if (param < 0 || param > static_cast<GLint>(Device::get().caps().maxSamples)) return ctx.setError(GL_INVALID_VALUE);
        fb->defaultSamples = param;
        break;
    case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS: fb->defaultFixedSampleLocations = param != 0; break;
    default: return ctx.setError(GL_INVALID_ENUM);
    }
    framebufferChanged(ctx, *fb);
}

void GetFramebufferParameteriv(Context& ctx, GLenum target, GLenum pname, GLint* params) {
    if (ctx.version() < 43 && !ctx.hasExtension("GL_ARB_framebuffer_no_attachments"))
        return ctx.unimplemented("glGetFramebufferParameteriv");
    RefPtr<Framebuffer>* binding = framebufferBinding(ctx, target);
    if (!binding) return ctx.setError(GL_INVALID_ENUM);
    Framebuffer* fb = binding->get();
    // GL 4.5: the framebuffer-dependent state of glGetIntegerv, for the
    // framebuffer bound to `target` (the default framebuffer included).
    if (ctx.version() >= 45 && (pname == GL_DOUBLEBUFFER || pname == GL_STEREO || pname == GL_SAMPLES ||
                                pname == GL_SAMPLE_BUFFERS || pname == GL_IMPLEMENTATION_COLOR_READ_FORMAT ||
                                pname == GL_IMPLEMENTATION_COLOR_READ_TYPE)) {
        if (!params) return;
        if (pname == GL_STEREO) return void(*params = GL_FALSE);
        if (pname == GL_DOUBLEBUFFER) {
            Surface* surface = ctx.drawSurface();
            *params = !fb && surface && surface->pixelFormat().double_buffer ? GL_TRUE : GL_FALSE;
            return;
        }
        RefPtr<Framebuffer> current = *binding;
        RefPtr<Framebuffer> draw = ctx.state.drawFramebuffer, read = ctx.state.readFramebuffer;
        ctx.state.drawFramebuffer = ctx.state.readFramebuffer = current;
        GetIntegerv(ctx, pname, params);
        ctx.state.drawFramebuffer = draw;
        ctx.state.readFramebuffer = read;
        return;
    }
    if (!fb) return ctx.setError(GL_INVALID_OPERATION);
    if (!params) return;
    switch (pname) {
    case GL_FRAMEBUFFER_DEFAULT_WIDTH: *params = fb->defaultWidth; return;
    case GL_FRAMEBUFFER_DEFAULT_HEIGHT: *params = fb->defaultHeight; return;
    case GL_FRAMEBUFFER_DEFAULT_LAYERS: *params = fb->defaultLayers; return;
    case GL_FRAMEBUFFER_DEFAULT_SAMPLES: *params = fb->defaultSamples; return;
    case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS: *params = fb->defaultFixedSampleLocations ? GL_TRUE : GL_FALSE; return;
    default: return ctx.setError(GL_INVALID_ENUM);
    }
}

void GetFramebufferAttachmentParameteriv(Context& ctx, GLenum target, GLenum attachment, GLenum pname, GLint* params) {
    RefPtr<Framebuffer>* binding = framebufferBinding(ctx, target);
    if (!binding) return ctx.setError(GL_INVALID_ENUM);
    if (!params) return;
    Framebuffer* fb = binding->get();
    if (!fb) {
        // Default framebuffer: GL_FRONT_LEFT, GL_BACK_LEFT, GL_DEPTH, GL_STENCIL.
        Surface* surface = target == GL_READ_FRAMEBUFFER ? ctx.readSurface() : ctx.drawSurface();
        const bool color = attachment == GL_FRONT_LEFT || attachment == GL_BACK_LEFT || attachment == GL_FRONT ||
                           attachment == GL_BACK || attachment == GL_LEFT;
        const bool depth = attachment == GL_DEPTH, stencil = attachment == GL_STENCIL;
        // The right buffers exist only in stereo formats, which WineMetalGL has none of.
        const bool right = attachment == GL_FRONT_RIGHT || attachment == GL_BACK_RIGHT || attachment == GL_RIGHT;
        if (!color && !depth && !stencil && !right) return ctx.setError(GL_INVALID_ENUM);
        const bool present = surface && (color || (depth && surface->hasDepth()) || (stencil && surface->hasStencil()));
        switch (pname) {
        case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE: *params = present ? GL_FRAMEBUFFER_DEFAULT : GL_NONE; return;
        case GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE: case GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE:
        case GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE: case GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE:
            if (!present) return ctx.setError(GL_INVALID_OPERATION);
            *params = color ? 8 : 0;
            return;
        case GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE:
            if (!present) return ctx.setError(GL_INVALID_OPERATION);
            *params = depth ? surface->pixelFormat().depth_bits : 0;
            return;
        case GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE:
            if (!present) return ctx.setError(GL_INVALID_OPERATION);
            *params = stencil ? surface->pixelFormat().stencil_bits : 0;
            return;
        case GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE:
            if (!present) return ctx.setError(GL_INVALID_OPERATION);
            *params = color ? GL_UNSIGNED_NORMALIZED
                      : depth ? (surface->pixelFormat().depth_bits == 32 ? GL_FLOAT : GL_UNSIGNED_NORMALIZED)
                              : GL_INDEX;
            return;
        case GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING:
            if (!present) return ctx.setError(GL_INVALID_OPERATION);
            *params = color && surface->srgbCapable() && ctx.windowSrgb() ? GL_SRGB : GL_LINEAR;
            return;
        default: return ctx.setError(present ? GL_INVALID_ENUM : GL_INVALID_OPERATION);
        }
    }
    Attachment* point = attachment == GL_DEPTH_STENCIL_ATTACHMENT ? &fb->depth : attachmentPoint(*fb, attachment);
    if (!point) return ctx.setError(GL_INVALID_ENUM);
    if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
        const bool same = fb->depth.texture == fb->stencil.texture && fb->depth.renderbuffer == fb->stencil.renderbuffer;
        if (!same) return ctx.setError(GL_INVALID_OPERATION);
    }
    const Attachment& a = *point;
    if (pname == GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE) {
        *params = static_cast<GLint>(a.type);
        return;
    }
    if (!a.attached()) {
        if (pname == GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME) { *params = 0; return; }
        return ctx.setError(GL_INVALID_OPERATION);
    }
    GLsizei w, h, s;
    bool fl;
    const FormatInfo* f = attachmentFormat(a, w, h, s, fl);
    switch (pname) {
    case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME:
        *params = static_cast<GLint>(a.type == GL_TEXTURE ? a.texture->name : a.renderbuffer->name);
        return;
    case GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE: *params = f ? f->redBits : 0; return;
    case GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE: *params = f ? f->greenBits : 0; return;
    case GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE: *params = f ? f->blueBits : 0; return;
    case GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE: *params = f ? f->alphaBits : 0; return;
    case GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE: *params = f ? f->depthBits : 0; return;
    case GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE: *params = f ? f->stencilBits : 0; return;
    case GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE:
        if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) return ctx.setError(GL_INVALID_OPERATION);
        if (!f) { *params = GL_NONE; return; }
        if (attachment == GL_STENCIL_ATTACHMENT) { *params = GL_UNSIGNED_INT; return; }
        switch (f->type) {
        case ComponentType::UnsignedNormalized: *params = GL_UNSIGNED_NORMALIZED; return;
        case ComponentType::SignedNormalized: *params = GL_SIGNED_NORMALIZED; return;
        case ComponentType::Float: *params = GL_FLOAT; return;
        case ComponentType::Int: *params = GL_INT; return;
        case ComponentType::UnsignedInt: *params = GL_UNSIGNED_INT; return;
        default: *params = GL_NONE; return;
        }
    case GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING: *params = f && f->srgb ? GL_SRGB : GL_LINEAR; return;
    case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL:
        if (a.type != GL_TEXTURE) break;
        *params = a.level;
        return;
    case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE:
        if (a.type != GL_TEXTURE) break;
        *params = static_cast<GLint>(a.cubeFace);
        return;
    case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER:
        if (a.type != GL_TEXTURE) break;
        *params = a.layer;
        return;
    case GL_FRAMEBUFFER_ATTACHMENT_LAYERED:
        if (a.type != GL_TEXTURE) break;
        *params = a.layered;
        return;
    default: break;
    }
    ctx.setError(a.type == GL_RENDERBUFFER ? GL_INVALID_ENUM : GL_INVALID_ENUM);
}

// --------------------------------------------------------- renderbuffers

void GenRenderbuffers(Context& ctx, GLsizei n, GLuint* renderbuffers) {
    if (n < 0) return ctx.setError(GL_INVALID_VALUE);
    if (!renderbuffers) return;
    ShareLock lock(ctx);
    ctx.share().renderbuffers.generate(n, renderbuffers);
}

void DeleteRenderbuffers(Context& ctx, GLsizei n, const GLuint* renderbuffers) {
    if (n < 0) return ctx.setError(GL_INVALID_VALUE);
    if (!renderbuffers) return;
    for (GLsizei i = 0; i < n; ++i) {
        if (!renderbuffers[i]) continue;
        RefPtr<Renderbuffer> rb;
        {
            ShareLock lock(ctx);
            rb = ctx.share().renderbuffers.remove(renderbuffers[i]);
        }
        if (!rb) continue;
        if (ctx.state.renderbuffer == rb) ctx.state.renderbuffer.reset();
        // Detach from the bound framebuffers of the current context.
        for (Framebuffer* fb : {ctx.state.drawFramebuffer.get(), ctx.state.readFramebuffer.get()}) {
            if (!fb) continue;
            for (Attachment* a : {&fb->depth, &fb->stencil})
                if (a->renderbuffer == rb) *a = Attachment();
            for (Attachment& a : fb->color)
                if (a.renderbuffer == rb) a = Attachment();
            framebufferChanged(ctx, *fb);
        }
    }
}

GLboolean IsRenderbuffer(Context& ctx, GLuint renderbuffer) {
    ShareLock lock(ctx);
    return ctx.share().renderbuffers.exists(renderbuffer) ? GL_TRUE : GL_FALSE;
}

void BindRenderbuffer(Context& ctx, GLenum target, GLuint name) { bindRenderbuffer(ctx, target, name, !ctx.isDesktopCore()); }

void BindRenderbufferEXT(Context& ctx, GLenum target, GLuint name) { bindRenderbuffer(ctx, target, name, true); }

void RenderbufferStorageMultisample(Context& ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width,
                                    GLsizei height) {
    if (target != GL_RENDERBUFFER) return ctx.setError(GL_INVALID_ENUM);
    Renderbuffer* rb = ctx.state.renderbuffer.get();
    if (!rb) return ctx.setError(GL_INVALID_OPERATION);
    // EXT_texture_format_BGRA8888: GL_BGRA_EXT is BGRA8_EXT's unsized name.
    if (ctx.isEs() && internalformat == es::kBgra) internalformat = 0x93A1;  // GL_BGRA8_EXT
    const GLenum resolved = resolveInternalFormat(internalformat, 0, 0);
    const FormatInfo* f = formatInfo(resolved);
    if (!f || f->compressed || (!f->colorRenderable && !f->isDepthOrStencil())) return ctx.setError(GL_INVALID_ENUM);
    if (ctx.isEs() && !es::renderbufferFormat(internalformat)) return ctx.setError(GL_INVALID_ENUM);
    if (width < 0 || height < 0 || samples < 0) return ctx.setError(GL_INVALID_VALUE);
    if (width > 16384 || height > 16384) return ctx.setError(GL_INVALID_VALUE);
    const auto& caps = Device::get().caps();
    // OpenGL ES 3.2 section 9.2.4: more samples than the format supports is INVALID_OPERATION.
    if (samples > static_cast<GLsizei>(caps.maxSamples)) return ctx.setError(ctx.isEs() ? GL_INVALID_OPERATION : GL_INVALID_VALUE);
    if (f->isInteger() && samples > 0) {
        // GL 3.3: MAX_INTEGER_SAMPLES; Metal supports multisampled integer targets.
    }
    // Round the sample count up to one Metal supports. OpenGL ES rounds a
    // request for one sample up too, as multisample textures do (Metal's
    // have two at least): the two then match for framebuffer completeness.
    GLsizei actual = samples;
    if (actual > 1 || (actual == 1 && ctx.isEs())) {
        actual = 2;
        while (actual < samples) actual *= 2;
        while (actual <= 8 && !caps.sampleCountSupported[actual]) actual *= 2;
        actual = std::min<GLsizei>(actual, static_cast<GLsizei>(caps.maxSamples));
    }
    rb->internalFormat = internalformat;
    rb->format = f;
    rb->width = width;
    rb->height = height;
    // One sample is single-sample storage: completeness compares the same
    // value GL_RENDERBUFFER_SAMPLES reports (0).
    rb->samples = actual > 1 ? actual : 0;
    rb->mtl = nil;
    rb->linearView = nil;
    if (width > 0 && height > 0) {
        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:mtlPixelFormat(f->storage)
                                                                                         width:static_cast<NSUInteger>(width)
                                                                                        height:static_cast<NSUInteger>(height)
                                                                                     mipmapped:NO];
        desc.storageMode = MTLStorageModePrivate;
        // Format views (a multisample glCopyImageSubData source) only for multisample
        // storage: they turn off lossless compression (Texture::formatViews).
        desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
                     (actual > 1 ? MTLTextureUsagePixelFormatView : 0);
        if (actual > 1) {
            desc.textureType = MTLTextureType2DMultisample;
            desc.sampleCount = static_cast<NSUInteger>(actual);
        }
        rb->mtl = [Device::get().mtl() newTextureWithDescriptor:desc];
        if (!rb->mtl) return ctx.setError(GL_OUT_OF_MEMORY);
        // RGB formats keep alpha at one in storage.
        if (f->storedAlphaIsOne) {
            PendingClear clear;
            RenderTargetDesc target;
            target.color[0].texture = rb->mtl;
            target.width = static_cast<uint32_t>(width);
            target.height = static_cast<uint32_t>(height);
            target.samples = static_cast<uint32_t>(std::max<GLsizei>(1, actual));
            clear.colorMask = 1;
            clear.color[0] = MTLClearColorMake(0, 0, 0, 1);
            ctx.stream().clearWithLoadAction(target, clear);
        }
    }
    for (Framebuffer* fb : {ctx.state.drawFramebuffer.get(), ctx.state.readFramebuffer.get()})
        if (fb) framebufferChanged(ctx, *fb);
}

void RenderbufferStorage(Context& ctx, GLenum target, GLenum internalformat, GLsizei width, GLsizei height) {
    RenderbufferStorageMultisample(ctx, target, 0, internalformat, width, height);
}

void GetRenderbufferParameteriv(Context& ctx, GLenum target, GLenum pname, GLint* params) {
    if (target != GL_RENDERBUFFER) return ctx.setError(GL_INVALID_ENUM);
    Renderbuffer* rb = ctx.state.renderbuffer.get();
    if (!rb) return ctx.setError(GL_INVALID_OPERATION);
    if (!params) return;
    const FormatInfo* f = rb->format;
    switch (pname) {
    case GL_RENDERBUFFER_WIDTH: *params = rb->width; return;
    case GL_RENDERBUFFER_HEIGHT: *params = rb->height; return;
    case GL_RENDERBUFFER_INTERNAL_FORMAT:
        // A renderbuffer without storage: GL_RGBA, OpenGL ES's GL_RGBA4.
        *params = static_cast<GLint>(ctx.isEs() && !rb->format ? GL_RGBA4 : rb->internalFormat);
        return;
    case GL_RENDERBUFFER_SAMPLES: *params = rb->samples > 1 ? rb->samples : 0; return;
    case GL_RENDERBUFFER_RED_SIZE: *params = f ? f->redBits : 0; return;
    case GL_RENDERBUFFER_GREEN_SIZE: *params = f ? f->greenBits : 0; return;
    case GL_RENDERBUFFER_BLUE_SIZE: *params = f ? f->blueBits : 0; return;
    case GL_RENDERBUFFER_ALPHA_SIZE: *params = f ? f->alphaBits : 0; return;
    case GL_RENDERBUFFER_DEPTH_SIZE: *params = f ? f->depthBits : 0; return;
    case GL_RENDERBUFFER_STENCIL_SIZE: *params = f ? f->stencilBits : 0; return;
    default: return ctx.setError(GL_INVALID_ENUM);
    }
}

// ---------------------------------------------------- draw/read buffers

void DrawBuffers(Context& ctx, GLsizei n, const GLenum* bufs) {
    if (n < 0 || n > ContextState::kMaxDrawBuffers) return ctx.setError(GL_INVALID_VALUE);
    Framebuffer* fb = ctx.state.drawFramebuffer.get();
    const auto isColorAttachment = [](GLenum b) { return b >= GL_COLOR_ATTACHMENT0 && b < GL_COLOR_ATTACHMENT0 + 32; };
    if (!fb) {
        // Default framebuffer (GL 3.3 table 4.4): NONE, FRONT_LEFT, FRONT_RIGHT, BACK_LEFT, BACK_RIGHT.
        // FRONT, BACK, LEFT, RIGHT and FRONT_AND_BACK name several buffers and are only valid for DrawBuffer.
        for (GLsizei i = 0; i < n; ++i) {
            const GLenum b = bufs[i];
            if (b == GL_NONE || b == GL_FRONT_LEFT || b == GL_BACK_LEFT || b == GL_FRONT_RIGHT || b == GL_BACK_RIGHT)
                continue;
            if (n == 1 && (b == GL_BACK || b == GL_FRONT || b == GL_LEFT || b == GL_FRONT_AND_BACK)) continue;
            return ctx.setError(isColorAttachment(b) || b == GL_BACK || b == GL_FRONT || b == GL_LEFT || b == GL_RIGHT ||
                                        b == GL_FRONT_AND_BACK
                                    ? GL_INVALID_OPERATION
                                    : GL_INVALID_ENUM);
        }
        for (GLsizei i = 0; i < n; ++i)
            for (GLsizei j = i + 1; j < n; ++j)
                if (bufs[i] != GL_NONE && bufs[i] == bufs[j]) return ctx.setError(GL_INVALID_OPERATION);
        if (n > 1) return ctx.setError(GL_INVALID_OPERATION);  // one colour buffer in the default framebuffer
        ctx.state.defaultDrawBuffer = n ? bufs[0] : GL_NONE;
        ctx.markDirty(kDirtyFramebuffer);
        return;
    }
    // Framebuffer objects: NONE or any COLOR_ATTACHMENTi in any slot, each at most once.
    for (GLsizei i = 0; i < n; ++i) {
        const GLenum b = bufs[i];
        if (b == GL_NONE) continue;
        if (!isColorAttachment(b))
            return ctx.setError(b == GL_BACK || b == GL_FRONT || b == GL_FRONT_AND_BACK || b == GL_LEFT || b == GL_RIGHT ||
                                        b == GL_FRONT_LEFT || b == GL_BACK_LEFT || b == GL_FRONT_RIGHT || b == GL_BACK_RIGHT
                                    ? GL_INVALID_OPERATION
                                    : GL_INVALID_ENUM);
        if (b >= GL_COLOR_ATTACHMENT0 + Framebuffer::kMaxColor) return ctx.setError(GL_INVALID_OPERATION);
        for (GLsizei j = 0; j < i; ++j)
            if (bufs[j] == b) return ctx.setError(GL_INVALID_OPERATION);
    }
    for (int i = 0; i < Framebuffer::kMaxColor; ++i) fb->drawBuffers[i] = i < n ? bufs[i] : GL_NONE;
    framebufferChanged(ctx, *fb);
}

void DrawBuffer(Context& ctx, GLenum buf) {
    Framebuffer* fb = ctx.state.drawFramebuffer.get();
    if (!fb) return DrawBuffers(ctx, 1, &buf);
    if (buf != GL_NONE && (buf < GL_COLOR_ATTACHMENT0 || buf >= GL_COLOR_ATTACHMENT0 + Framebuffer::kMaxColor))
        return ctx.setError(buf >= GL_COLOR_ATTACHMENT0 && buf < GL_COLOR_ATTACHMENT0 + 32 ? GL_INVALID_OPERATION : GL_INVALID_OPERATION);
    // DrawBuffer on an FBO routes fragment colour 0 to `buf` for every draw buffer slot.
    for (int i = 0; i < Framebuffer::kMaxColor; ++i) fb->drawBuffers[i] = i == 0 ? buf : GL_NONE;
    framebufferChanged(ctx, *fb);
}

void ReadBuffer(Context& ctx, GLenum src) {
    Framebuffer* fb = ctx.state.readFramebuffer.get();
    if (!fb) {
        // OpenGL ES's default framebuffer has only GL_BACK.
        if (ctx.isEs() && src != GL_NONE && src != GL_BACK)
            return ctx.setError(src >= GL_COLOR_ATTACHMENT0 && src < GL_COLOR_ATTACHMENT0 + 32 ? GL_INVALID_OPERATION : GL_INVALID_ENUM);
        if (src != GL_NONE && src != GL_BACK && src != GL_FRONT && src != GL_LEFT && src != GL_BACK_LEFT && src != GL_FRONT_LEFT)
            return ctx.setError(src >= GL_COLOR_ATTACHMENT0 && src < GL_COLOR_ATTACHMENT0 + 32 ? GL_INVALID_OPERATION : GL_INVALID_ENUM);
        ctx.state.defaultReadBuffer = src;
        return;
    }
    if (src != GL_NONE && (src < GL_COLOR_ATTACHMENT0 || src >= GL_COLOR_ATTACHMENT0 + Framebuffer::kMaxColor))
        // (OpenGL ES 3.2 section 16.1.1: INVALID_OPERATION for attachments
        // past MAX_COLOR_ATTACHMENTS; desktop GL 4.6 says INVALID_VALUE.)
        return ctx.setError(src >= GL_COLOR_ATTACHMENT0 && src < GL_COLOR_ATTACHMENT0 + 32
                                ? (ctx.isEs() ? GL_INVALID_OPERATION : GL_INVALID_VALUE)
                            : (src == GL_BACK || (!ctx.isEs() && (src == GL_FRONT || src == GL_LEFT))) ? GL_INVALID_OPERATION
                                                                                                       : GL_INVALID_ENUM);
    fb->readBuffer = src;
    framebufferChanged(ctx, *fb);
}

// ------------------------------------------------------------ ReadPixels

namespace {

// OpenGL ES 3.2 section 16.1.2: RGBA with the read buffer's component type
// (table 16.1 and the extensions' formats), or the implementation's read
// format and type. Returns false with the GL error set.
bool esReadFormat(Context& ctx, GLenum format, GLenum& type) {
    if (format == GL_DEPTH_COMPONENT || format == GL_STENCIL_INDEX || format == GL_DEPTH_STENCIL ||
        (!es::isFormat(format) && format != GL_BGRA) || !es::isType(type))
        return ctx.setError(GL_INVALID_ENUM), false;
    if (type == es::kHalfFloatOes) type = GL_HALF_FLOAT;
    if (completeness(ctx, ctx.state.readFramebuffer.get()) != GL_FRAMEBUFFER_COMPLETE)
        return ctx.setError(GL_INVALID_FRAMEBUFFER_OPERATION), false;
    GLenum implFormat = 0, implType = 0;
    if (implementationReadFormat(ctx, implFormat, implType) && format == implFormat && type == implType) return true;
    uint32_t level, slice;
    GLenum internalFormat = 0;
    int w, h;
    if (!ctx.draw().readColorTexture(level, slice, internalFormat, w, h)) return true;  // no read buffer: below
    if (es::readPixelsCompatible(internalFormat, format, type)) return true;
    // EXT_read_format_bgra
    if (format == GL_BGRA && type == GL_UNSIGNED_BYTE && !formatInfo(internalFormat)->isInteger()) return true;
    return ctx.setError(GL_INVALID_OPERATION), false;
}

}  // namespace

void ReadPixels(Context& ctx, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void* pixels) {
    if (width < 0 || height < 0) return ctx.setError(GL_INVALID_VALUE);
    if (ctx.isEs() && !esReadFormat(ctx, format, type)) return;
    // GL 3.3 section 4.3.2: DEPTH_STENCIL takes only the packed depth/stencil types.
    if (format == GL_DEPTH_STENCIL && type != GL_UNSIGNED_INT_24_8 && type != GL_FLOAT_32_UNSIGNED_INT_24_8_REV)
        return ctx.setError(GL_INVALID_ENUM);
    if (!validTransferFormatType(format, type, ctx.isCompat())) {
        if (groupBytes(format, type) == 0 && !validTransferFormatType(format, GL_UNSIGNED_BYTE, ctx.isCompat()))
            return ctx.setError(GL_INVALID_ENUM);
        return ctx.setError(GL_INVALID_OPERATION);
    }
    Framebuffer* fb = ctx.state.readFramebuffer.get();
    if (completeness(ctx, fb) != GL_FRAMEBUFFER_COMPLETE) return ctx.setError(GL_INVALID_FRAMEBUFFER_OPERATION);
    const bool depth = format == GL_DEPTH_COMPONENT, stencil = format == GL_STENCIL_INDEX, depthStencil = format == GL_DEPTH_STENCIL;
    id<MTLTexture> source = nil;
    uint32_t level = 0, slice = 0;
    GLenum internalFormat = 0;
    int levelWidth = 0, levelHeight = 0;
    if (depth || stencil || depthStencil) {
        const Attachment* a = nullptr;
        if (fb) {
            a = depth ? &fb->depth : &fb->stencil;
            if (depthStencil && !(fb->depth.attached() && fb->stencil.attached())) return ctx.setError(GL_INVALID_OPERATION);
            if (!a->attached()) return ctx.setError(GL_INVALID_OPERATION);
            GLsizei w, h, s;
            bool fl;
            const FormatInfo* f = attachmentFormat(*a, w, h, s, fl);
            if (!f || s > 1) return ctx.setError(GL_INVALID_OPERATION);
            if (a->type == GL_TEXTURE) {
                const int face = a->cubeFace ? static_cast<int>(a->cubeFace - GL_TEXTURE_CUBE_MAP_POSITIVE_X) : 0;
                if (!imageHasStorage(ctx, *a->texture, face, a->level)) return ctx.setError(GL_INVALID_OPERATION);
                source = a->texture->mtl;
                level = static_cast<uint32_t>(a->level);
                slice = a->cubeFace ? a->cubeFace - GL_TEXTURE_CUBE_MAP_POSITIVE_X : static_cast<uint32_t>(a->layer);
            } else {
                source = a->renderbuffer->mtl;
            }
            internalFormat = f->internalFormat;
            levelWidth = w;
            levelHeight = h;
        } else {
            Surface* surface = ctx.readSurface();
            if (!surface || (depth && !surface->hasDepth()) || (stencil && !surface->hasStencil()) ||
                (depthStencil && !(surface->hasDepth() && surface->hasStencil())))
                return ctx.setError(GL_INVALID_OPERATION);
            source = surface->resolvedDepthStencil(ctx.stream());
            internalFormat = surface->depthStencilFormat();
            levelWidth = surface->width();
            levelHeight = surface->height();
        }
    } else {
        if (fb) {
            if (fb->readBuffer == GL_NONE) return ctx.setError(GL_INVALID_OPERATION);
            GLsizei w, h, s;
            bool fl;
            const FormatInfo* f = attachmentFormat(*attachmentPoint(*fb, fb->readBuffer), w, h, s, fl);
            if (f && s > 1) return ctx.setError(GL_INVALID_OPERATION);
            if (f && !transferCompatible(format, type, *f)) return ctx.setError(GL_INVALID_OPERATION);
        } else if (ctx.state.defaultReadBuffer == GL_NONE) {
            return ctx.setError(GL_INVALID_OPERATION);
        } else if (format == GL_RGBA_INTEGER || format == GL_RED_INTEGER || format == GL_RG_INTEGER ||
                   format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER || format == GL_BGR_INTEGER) {
            return ctx.setError(GL_INVALID_OPERATION);
        }
        source = ctx.draw().readColorTexture(level, slice, internalFormat, levelWidth, levelHeight);
    }
    if (!source) return ctx.setError(GL_INVALID_OPERATION);
    const FormatInfo* info = formatInfo(internalFormat);
    if (!info) return ctx.setError(GL_INVALID_OPERATION);

    const PixelStoreParams ps = pixelStoreParams(ctx.state.pack, false);
    Buffer* pbo = ctx.state.pixelPackBuffer.get();
    uint8_t* destination = static_cast<uint8_t*>(pixels);
    // Into a pack buffer, texels already in the requested layout are copied
    // by the GPU in command order, without waiting for it (asynchronous
    // readback, as engines use pack buffers for): a colour region inside the
    // source, in the alignment Metal's copies need.
    if (pbo && pbo->mtl && !pbo->mappedNonPersistent() && !ps.swapBytes && !depth && !stencil && !depthStencil && width > 0 && height > 0 &&
        x >= 0 && y >= 0 && x + width <= levelWidth && y + height <= levelHeight && storedLayout(format, type, *info) &&
        (ctx.state.clampReadColorMode == GL_FALSE || info->type == ComponentType::UnsignedNormalized ||
         (ctx.state.clampReadColorMode == GL_FIXED_ONLY && info->type != ComponentType::SignedNormalized))) {
        const TransferLayout layout = transferLayout(format, type, width, height, 1, ps);
        const size_t texelBytes = storageBytes(info->storage);
        const uintptr_t start = reinterpret_cast<uintptr_t>(pixels) + layout.skipBytes;
        if (start % 4 == 0 && start % texelBytes == 0 && layout.rowStride % texelBytes == 0 && layout.rowStride % 4 == 0 &&
            start <= static_cast<size_t>(pbo->size) && layout.bytesNeeded - layout.skipBytes <= static_cast<size_t>(pbo->size) - start) {
            const bool is3D = source.textureType == MTLTextureType3D;
            [ctx.stream().blit() copyFromTexture:source sourceSlice:is3D ? 0 : slice sourceLevel:level
                                    sourceOrigin:MTLOriginMake(x, y, is3D ? slice : 0)
                                      sourceSize:MTLSizeMake(width, height, 1) toBuffer:pbo->mtl destinationOffset:start
                          destinationBytesPerRow:layout.rowStride destinationBytesPerImage:layout.rowStride * height];
            pbo->lastUseSerial = pbo->lastWriteSerial = ctx.stream().serial();
            ++pbo->version;
            stats::count(&stats::Frame::readsAsync);
            return;
        }
    }
    // The paths below wait for the GPU.
    stats::count(&stats::Frame::readsWaiting);
    if (stats::g_reporting) {
        // Once per format, type and destination: what keeps a read from the asynchronous path.
        static std::mutex mutex;
        static std::set<uint64_t> seen;
        const uint64_t key = uint64_t{format} << 33 | uint64_t{type} << 1 | (pbo ? 1u : 0u);
        std::lock_guard<std::mutex> lock(mutex);
        if (seen.insert(key).second)
            std::fprintf(stderr, "winemetalgl:stats: glReadPixels waits for the GPU: format 0x%x type 0x%x into %s, "
                                 "source storage %d\n",
                         format, type, pbo ? "a pack buffer" : "client memory", static_cast<int>(info->storage));
    }
    if (pbo) {
        const TransferLayout layout = transferLayout(format, type, width, height, 1, ps);
        const uintptr_t offset = reinterpret_cast<uintptr_t>(pixels);
        if (pbo->mappedNonPersistent() || !pbo->mtl || offset > static_cast<size_t>(pbo->size) ||
            layout.bytesNeeded > static_cast<size_t>(pbo->size) - offset)
            return ctx.setError(GL_INVALID_OPERATION);
        ctx.stream().waitFor(pbo->lastUseSerial);
        destination = pbo->contents() + offset;
        ++pbo->version;
    }
    if (!destination || width == 0 || height == 0) return;
    // Clip to the source; pixels outside are left untouched.
    const int x0 = std::max(0, x), y0 = std::max(0, y);
    const int x1 = std::min(levelWidth, x + width), y1 = std::min(levelHeight, y + height);
    if (x1 <= x0 || y1 <= y0) return;
    const int w = x1 - x0, h = y1 - y0;
    // Copy the region to CPU-visible memory.
    Storage storage = info->storage;
    MTLBlitOption option = MTLBlitOptionNone;
    size_t texelBytes = storageBytes(storage);
    FormatInfo readInfo = *info;
    if (storage == Storage::Depth32Float_Stencil8) {
        if (stencil) {
            option = MTLBlitOptionStencilFromDepthStencil;
            readInfo.storage = Storage::Stencil8;
            texelBytes = 1;
        } else if (depth) {
            option = MTLBlitOptionDepthFromDepthStencil;
            readInfo.storage = Storage::Depth32Float;
            texelBytes = 4;
        }
    }
    std::vector<uint8_t> texels;
    auto readRegion = [&](MTLBlitOption opt, size_t bytes) {
        const size_t rowBytes = static_cast<size_t>(w) * bytes;
        id<MTLBuffer> staging = [Device::get().mtl() newBufferWithLength:rowBytes * static_cast<size_t>(h)
                                                                 options:MTLResourceStorageModeShared];
        id<MTLBlitCommandEncoder> blit = ctx.stream().blit();
        // A 3D texture's layer is a depth plane, addressed through the origin.
        const bool is3D = source.textureType == MTLTextureType3D;
        [blit copyFromTexture:source sourceSlice:is3D ? 0 : slice sourceLevel:level
                 sourceOrigin:MTLOriginMake(x0, y0, is3D ? slice : 0)
                   sourceSize:MTLSizeMake(w, h, 1) toBuffer:staging destinationOffset:0 destinationBytesPerRow:rowBytes
        destinationBytesPerImage:rowBytes * static_cast<size_t>(h) options:opt];
        // Without a pack buffer only local memory and the client's are used
        // after the wait, so other contexts may run meanwhile.
        if (pbo) ctx.stream().finish();
        else finishUnlocked(ctx);
        return staging;
    };
    if (depthStencil && storage == Storage::Depth32Float_Stencil8) {
        id<MTLBuffer> d = readRegion(MTLBlitOptionDepthFromDepthStencil, 4);
        id<MTLBuffer> s = readRegion(MTLBlitOptionStencilFromDepthStencil, 1);
        texels.resize(static_cast<size_t>(w) * h * 8);
        for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
            std::memcpy(&texels[i * 8], static_cast<uint8_t*>(d.contents) + i * 4, 4);
            texels[i * 8 + 4] = static_cast<uint8_t*>(s.contents)[i];
        }
    } else {
        id<MTLBuffer> staging = readRegion(option, texelBytes);
        texels.assign(static_cast<uint8_t*>(staging.contents),
                      static_cast<uint8_t*>(staging.contents) + texelBytes * static_cast<size_t>(w) * h);
    }
    PixelStoreParams region = ps;
    region.skipPixels += x0 - x;
    region.skipRows += y0 - y;
    if (region.rowLength == 0) region.rowLength = width;
    const bool clampColor = ctx.state.clampReadColorMode == GL_TRUE ||
                            (ctx.state.clampReadColorMode == GL_FIXED_ONLY &&
                             (info->type == ComponentType::UnsignedNormalized || info->type == ComponentType::SignedNormalized));
    packPixels(texels.data(), readInfo, w, h, 1, format, type, region, clampColor, destination);
}

// ------------------------------------------------------------ sample positions

void GetMultisamplefv(Context& ctx, GLenum pname, GLuint index, GLfloat* val) {
    if (pname != GL_SAMPLE_POSITION) return ctx.setError(GL_INVALID_ENUM);
    RenderTargetDesc target;
    TargetFormats formats;
    if (!ctx.draw().drawTarget(target, formats, false)) {
        if (index >= 1 || ctx.isEs()) return ctx.setError(GL_INVALID_VALUE);
        val[0] = val[1] = 0.5f;
        return;
    }
    const uint32_t samples = std::max(1u, target.samples);
    // OpenGL ES: index must be below SAMPLES, which is 0 without multisampling.
    if (index >= samples || (ctx.isEs() && target.samples <= 1)) return ctx.setError(GL_INVALID_VALUE);
    if (samples == 1) {
        val[0] = val[1] = 0.5f;
        return;
    }
    MTLSamplePosition positions[16];
    [Device::get().mtl() getDefaultSamplePositions:positions count:samples];
    // Metal positions are in a top-left origin pixel; GL's sample y grows upwards,
    // which is the same direction as Metal rows here (GL row 0 = Metal row 0).
    val[0] = positions[index].x;
    val[1] = positions[index].y;
}

}  // namespace wmgl::api
