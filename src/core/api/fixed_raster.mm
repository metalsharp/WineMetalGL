// Raster position and the legacy pixel paths (GL 3.3 compatibility sections
// 2.25, 3.7.5 and 4.3.3): glRasterPos/glWindowPos, glBitmap, glDrawPixels,
// glCopyPixels, glPixelZoom. Pixel rectangles are drawn as window-space
// quads through the normal fragment operations (depth, stencil, blending)
// by small internal programs, from a temporary texture holding the pixels.

#include <algorithm>
#include <cmath>
#include <cstring>

#include "core/compat/compat_api.h"
#include "core/compat/display_list.h"
#include "core/compat/fixed_function.h"
#include "core/compat/fixed_state.h"
#include "core/context.h"
#include "core/draw.h"
#include "core/format/pixel_transfer.h"
#include "core/program.h"
#include "core/util/log.h"
#include "gl_api.gen.h"

namespace wmgl::api {
namespace {

bool compatOnly(Context& ctx) {
    if (ctx.isCompat()) return true;
    ctx.setError(GL_INVALID_OPERATION);
    return false;
}

// ------------------------------------------------------------ raster position

void setRasterPos(Context& ctx, float x, float y, float z, float w) {
    if (!compatOnly(ctx)) return;
    ContextState& s = ctx.state;
    FixedState& f = s.fixed;
    const float object[4] = {x, y, z, w};
    float eye[4], clip[4];
    f.modelview.back().transform(object, eye);
    f.projection.back().transform(eye, clip);
    FixedState::RasterPos& r = f.rasterPos;
    // Clipped against the view volume and the enabled user clip planes.
    bool valid = clip[3] != 0;
    for (int c = 0; c < 3 && valid; ++c) valid = std::fabs(clip[c]) <= std::fabs(clip[3]);
    for (int i = 0; i < FixedState::kClipPlanes && valid; ++i) {
        if (!(s.clipDistanceEnables & (1u << i))) continue;
        const float* p = f.clipPlanes[i];
        valid = p[0] * eye[0] + p[1] * eye[1] + p[2] * eye[2] + p[3] * eye[3] >= 0;
    }
    r.valid = valid;
    if (!valid) return;
    const float ndc[3] = {clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]};
    r.window[0] = static_cast<float>(s.viewports[0][0] + (ndc[0] + 1) * s.viewports[0][2] * 0.5f);
    r.window[1] = static_cast<float>(s.viewports[0][1] + (ndc[1] + 1) * s.viewports[0][3] * 0.5f);
    const double depth = s.clipDepthMode == GL_ZERO_TO_ONE ? ndc[2] : (ndc[2] + 1) * 0.5;
    r.window[2] = static_cast<float>(s.depthRanges[0][0] + depth * (s.depthRanges[0][1] - s.depthRanges[0][0]));
    r.window[3] = clip[3];
    r.distance = std::sqrt(eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]);
    std::memcpy(r.color, s.genericAttribs[kConvColor].value.f, 16);
    std::memcpy(r.texCoord, s.genericAttribs[kConvTexCoord0].value.f, 16);
}

void setWindowPos(Context& ctx, float x, float y, float z) {
    if (!compatOnly(ctx)) return;
    ContextState& s = ctx.state;
    FixedState::RasterPos& r = s.fixed.rasterPos;
    r.window[0] = x;
    r.window[1] = y;
    r.window[2] = static_cast<float>(s.depthRanges[0][0] + std::clamp(z, 0.0f, 1.0f) * (s.depthRanges[0][1] - s.depthRanges[0][0]));
    r.window[3] = 1;
    r.valid = true;
    r.distance = s.fixed.fogCoordSource == GL_FOG_COORD ? s.genericAttribs[kConvFogCoord].value.f[0] : 0.0f;
    std::memcpy(r.color, s.genericAttribs[kConvColor].value.f, 16);
    std::memcpy(r.texCoord, s.genericAttribs[kConvTexCoord0].value.f, 16);
}

template <class T>
float fv(T v) { return static_cast<float>(v); }

// ------------------------------------------------------------ pixel rectangles

enum class Blit { Mask, Color, Depth };

const char* kBlitVertex = R"(#version 330 core
uniform vec4 wmgl_rect;       // window x0, y0, x1, y1
uniform vec4 wmgl_target;     // framebuffer width, height, NDC depth, 0
in vec2 wmgl_corner;
out vec2 wmgl_uv;
void main() {
    vec2 window = mix(wmgl_rect.xy, wmgl_rect.zw, wmgl_corner);
    gl_Position = vec4(window / wmgl_target.xy * 2.0 - 1.0, wmgl_target.z, 1.0);
    wmgl_uv = wmgl_corner;
})";

const char* fragmentSource(Blit kind) {
    switch (kind) {
    case Blit::Mask:  // glBitmap: set bits take the raster colour
        return R"(#version 330 core
uniform sampler2D wmgl_tex0; uniform vec4 wmgl_color; in vec2 wmgl_uv; out vec4 o;
void main() { if (texture(wmgl_tex0, wmgl_uv).r < 0.5) discard; o = wmgl_color; })";
    case Blit::Depth:  // DEPTH_COMPONENT pixels write depth with the raster colour
        return R"(#version 330 core
uniform sampler2D wmgl_tex0; uniform vec4 wmgl_color; in vec2 wmgl_uv; out vec4 o;
void main() { gl_FragDepth = texture(wmgl_tex0, wmgl_uv).r; o = wmgl_color; })";
    default:
        return R"(#version 330 core
uniform sampler2D wmgl_tex0; in vec2 wmgl_uv; out vec4 o;
void main() { o = texture(wmgl_tex0, wmgl_uv); })";
    }
}

Program* blitProgram(Context& ctx, Blit kind) {
    auto& cache = ctx.share().fixedFunctionPrograms;
    const std::string key = std::string("blit:") + static_cast<char>('0' + static_cast<int>(kind));
    auto it = cache.find(key);
    if (it != cache.end()) return it->second.get();
    std::string log;
    RefPtr<Program> program = compat::linkInternalProgram(kBlitVertex, fragmentSource(kind), "wmgl_tex", log);
    if (!program) WMGL_ERR("pixel rectangle program failed to link:\n%s", log.c_str());
    Program* raw = program.get();
    cache.emplace(key, std::move(program));
    return raw;
}

void setUniform(Program& p, const char* name, const float v[4]) {
    for (const shader::UniformInfo& u : p.linked->link.uniforms)
        if (u.blockIndex < 0 && u.name == name && u.offset >= 0 &&
            static_cast<size_t>(u.offset) + 16 <= p.uniformData.size()) {
            std::memcpy(p.uniformData.data() + u.offset, v, 16);
            ++p.uniformVersion;
        }
}

// FNV-1a over `bytes`, continuing from `hash`.
uint64_t hashBytes(uint64_t hash, const void* bytes, size_t size) {
    const auto* p = static_cast<const uint8_t*>(bytes);
    for (size_t i = 0; i < size; ++i) hash = (hash ^ p[i]) * 1099511628211ull;
    return hash;
}

template <class T>
uint64_t hashValue(uint64_t hash, const T& value) { return hashBytes(hash, &value, sizeof(value)); }

// A temporary nearest-filtered 2D texture; `define` specifies its image
// through the regular texture commands on texture unit 0.
template <class Define>
RefPtr<Texture> temporaryTexture(Context& ctx, Define define) {
    RefPtr<Texture> tex = makeRef<Texture>(0, GL_TEXTURE_2D);
    tex->sampler.minFilter = GL_NEAREST;
    tex->sampler.magFilter = GL_NEAREST;
    tex->sampler.wrapS = tex->sampler.wrapT = GL_CLAMP_TO_EDGE;
    tex->maxLevel = 0;
    ContextState& s = ctx.state;
    const GLuint active = s.activeTexture;
    RefPtr<Texture> saved = s.textureUnits[active].bound[kTex2D];
    s.textureUnits[active].bound[kTex2D] = tex;
    define();
    s.textureUnits[active].bound[kTex2D] = saved;
    ctx.markDirty(kDirtyTextures);
    return tex;
}

// A nearest-filtered 2D texture for a pixel rectangle; `define` specifies its
// image through the regular texture commands on texture unit 0. Textures are
// reused by content (`key`), which also keeps uploads (and the render pass
// breaks they cause) out of steady-state frames.
template <class Define>
RefPtr<Texture> cachedTexture(Context& ctx, uint64_t key, Define define) {
    auto& cache = ctx.pixelRectangleCache;
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    if (cache.size() >= 512) cache.clear();
    RefPtr<Texture> tex = temporaryTexture(ctx, define);
    cache.emplace(key, tex);
    return tex;
}

// Draws `tex` as a window-space rectangle from (x0, y0) to (x1, y1) at
// window depth `z`.
void drawRectangle(Context& ctx, Blit kind, const RefPtr<Texture>& tex, float x0, float y0, float x1, float y1, float z,
                   const float color[4]) {
    Program* program = blitProgram(ctx, kind);
    if (!program || !program->linked) return;
    RenderTargetDesc target;
    TargetFormats formats;
    if (!ctx.draw().drawTarget(target, formats)) return;
    ContextState& s = ctx.state;
    const float rect[4] = {x0, y0, x1, y1};
    const float ndcDepth = s.clipDepthMode == GL_ZERO_TO_ONE ? z : z * 2 - 1;
    const float targetSize[4] = {static_cast<float>(target.width), static_cast<float>(target.height), ndcDepth, 0};
    setUniform(*program, "wmgl_rect", rect);
    setUniform(*program, "wmgl_target", targetSize);
    if (color) setUniform(*program, "wmgl_color", color);

    static const float kCorners[8] = {0, 0, 1, 0, 0, 1, 1, 1};
    RefPtr<VertexArray> vao = makeRef<VertexArray>(0);
    VertexAttrib& a = vao->attribs[0];
    a.enabled = true;
    a.size = 2;
    a.type = GL_FLOAT;
    a.offset = reinterpret_cast<GLintptr>(kCorners);

    // Pixel rectangles are not transformed, culled, clipped by the viewport
    // or textured by fixed-function state: draw with the program alone.
    RefPtr<Program> savedProgram = s.program;
    RefPtr<ProgramPipeline> savedPipeline = s.programPipeline;
    RefPtr<VertexArray> savedVao = s.vertexArray;
    TextureUnit savedUnit = s.textureUnits[0];
    GLfloat savedViewport[4];
    std::copy(s.viewports[0], s.viewports[0] + 4, savedViewport);
    const GLdouble savedRange[2] = {s.depthRanges[0][0], s.depthRanges[0][1]};
    const bool savedCull = s.cullFace, savedDiscard = s.rasterizerDiscard;
    const GLenum savedMode = s.polygonMode, savedBack = s.polygonModeBack;
    const uint32_t savedClip = s.clipDistanceEnables;

    s.program = program;
    s.programPipeline.reset();
    s.vertexArray = vao;
    s.textureUnits[0].bound[kTex2D] = tex;
    s.textureUnits[0].sampler.reset();
    s.viewports[0][0] = s.viewports[0][1] = 0;
    s.viewports[0][2] = static_cast<GLint>(target.width);
    s.viewports[0][3] = static_cast<GLint>(target.height);
    s.depthRanges[0][0] = 0;
    s.depthRanges[0][1] = 1;
    s.cullFace = false;
    s.rasterizerDiscard = false;
    s.polygonMode = s.polygonModeBack = GL_FILL;
    s.clipDistanceEnables = 0;
    ctx.markDirty(kDirtyAll);
    ctx.draw().drawArrays(GL_TRIANGLE_STRIP, 0, 4, 1);

    s.program = savedProgram;
    s.programPipeline = savedPipeline;
    s.vertexArray = savedVao;
    s.textureUnits[0] = savedUnit;
    std::copy(savedViewport, savedViewport + 4, s.viewports[0]);
    s.depthRanges[0][0] = savedRange[0];
    s.depthRanges[0][1] = savedRange[1];
    s.cullFace = savedCull;
    s.rasterizerDiscard = savedDiscard;
    s.polygonMode = savedMode;
    s.polygonModeBack = savedBack;
    s.clipDistanceEnables = savedClip;
    ctx.markDirty(kDirtyAll);
}

}  // namespace

namespace {

// Bytes per row of a GL_BITMAP image (GL 3.3 compatibility section 3.7.4).
size_t bitmapRowBytes(const PixelStore& s, GLsizei width) {
    const size_t rowPixels = static_cast<size_t>(s.rowLength > 0 ? s.rowLength : width);
    const size_t align = static_cast<size_t>(std::max(1, s.alignment));
    return ((rowPixels + 7) / 8 + align - 1) / align * align;
}

}  // namespace

size_t bitmapBytes(const PixelStore& s, GLsizei width, GLsizei height) {
    if (width <= 0 || height <= 0) return 0;
    const size_t lastBit = static_cast<size_t>(std::max(0, s.skipPixels)) + static_cast<size_t>(width) - 1;
    return (static_cast<size_t>(std::max(0, s.skipRows)) + static_cast<size_t>(height) - 1) * bitmapRowBytes(s, width) +
           lastBit / 8 + 1;
}

bool bitmapSource(Context& ctx, const GLubyte* bits, GLsizei width, GLsizei height, const uint8_t*& out) {
    out = bits;
    Buffer* pbo = ctx.state.pixelUnpackBuffer.get();
    if (!pbo) return true;
    const uintptr_t offset = reinterpret_cast<uintptr_t>(bits);
    if (pbo->mappedNonPersistent() || !pbo->mtl ||
        offset > static_cast<size_t>(pbo->size) ||
        bitmapBytes(ctx.state.unpack, width, height) > static_cast<size_t>(pbo->size) - offset) {
        ctx.setError(GL_INVALID_OPERATION);
        return false;
    }
    if (pbo->lastWriteSerial) ctx.stream().waitFor(pbo->lastWriteSerial);
    out = pbo->contents() + offset;
    return true;
}

// Unpacks a GL bitmap (1 bit per pixel, GL 3.3 compatibility section 3.7.4)
// to one byte per pixel.
std::vector<uint8_t> unpackBitmap(const PixelStore& s, const uint8_t* bits, GLsizei width, GLsizei height) {
    std::vector<uint8_t> out(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
    if (!bits) return out;
    const size_t rowBytes = bitmapRowBytes(s, width);
    for (GLsizei y = 0; y < height; ++y) {
        const uint8_t* row = bits + (static_cast<size_t>(s.skipRows) + static_cast<size_t>(y)) * rowBytes;
        for (GLsizei x = 0; x < width; ++x) {
            const size_t bit = static_cast<size_t>(s.skipPixels) + static_cast<size_t>(x);
            const uint8_t byte = row[bit / 8];
            const int shift = s.lsbFirst ? static_cast<int>(bit % 8) : 7 - static_cast<int>(bit % 8);
            out[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] = (byte >> shift) & 1 ? 255 : 0;
        }
    }
    return out;
}

void RasterPos2d(Context& ctx, GLdouble x, GLdouble y) { setRasterPos(ctx, fv(x), fv(y), 0, 1); }
void RasterPos2dv(Context& ctx, const GLdouble* v) { setRasterPos(ctx, fv(v[0]), fv(v[1]), 0, 1); }
void RasterPos2f(Context& ctx, GLfloat x, GLfloat y) { setRasterPos(ctx, x, y, 0, 1); }
void RasterPos2fv(Context& ctx, const GLfloat* v) { setRasterPos(ctx, v[0], v[1], 0, 1); }
void RasterPos2i(Context& ctx, GLint x, GLint y) { setRasterPos(ctx, fv(x), fv(y), 0, 1); }
void RasterPos2iv(Context& ctx, const GLint* v) { setRasterPos(ctx, fv(v[0]), fv(v[1]), 0, 1); }
void RasterPos2s(Context& ctx, GLshort x, GLshort y) { setRasterPos(ctx, fv(x), fv(y), 0, 1); }
void RasterPos2sv(Context& ctx, const GLshort* v) { setRasterPos(ctx, fv(v[0]), fv(v[1]), 0, 1); }
void RasterPos3d(Context& ctx, GLdouble x, GLdouble y, GLdouble z) { setRasterPos(ctx, fv(x), fv(y), fv(z), 1); }
void RasterPos3dv(Context& ctx, const GLdouble* v) { setRasterPos(ctx, fv(v[0]), fv(v[1]), fv(v[2]), 1); }
void RasterPos3f(Context& ctx, GLfloat x, GLfloat y, GLfloat z) { setRasterPos(ctx, x, y, z, 1); }
void RasterPos3fv(Context& ctx, const GLfloat* v) { setRasterPos(ctx, v[0], v[1], v[2], 1); }
void RasterPos3i(Context& ctx, GLint x, GLint y, GLint z) { setRasterPos(ctx, fv(x), fv(y), fv(z), 1); }
void RasterPos3iv(Context& ctx, const GLint* v) { setRasterPos(ctx, fv(v[0]), fv(v[1]), fv(v[2]), 1); }
void RasterPos3s(Context& ctx, GLshort x, GLshort y, GLshort z) { setRasterPos(ctx, fv(x), fv(y), fv(z), 1); }
void RasterPos3sv(Context& ctx, const GLshort* v) { setRasterPos(ctx, fv(v[0]), fv(v[1]), fv(v[2]), 1); }
void RasterPos4d(Context& ctx, GLdouble x, GLdouble y, GLdouble z, GLdouble w) { setRasterPos(ctx, fv(x), fv(y), fv(z), fv(w)); }
void RasterPos4dv(Context& ctx, const GLdouble* v) { setRasterPos(ctx, fv(v[0]), fv(v[1]), fv(v[2]), fv(v[3])); }
void RasterPos4f(Context& ctx, GLfloat x, GLfloat y, GLfloat z, GLfloat w) { setRasterPos(ctx, x, y, z, w); }
void RasterPos4fv(Context& ctx, const GLfloat* v) { setRasterPos(ctx, v[0], v[1], v[2], v[3]); }
void RasterPos4i(Context& ctx, GLint x, GLint y, GLint z, GLint w) { setRasterPos(ctx, fv(x), fv(y), fv(z), fv(w)); }
void RasterPos4iv(Context& ctx, const GLint* v) { setRasterPos(ctx, fv(v[0]), fv(v[1]), fv(v[2]), fv(v[3])); }
void RasterPos4s(Context& ctx, GLshort x, GLshort y, GLshort z, GLshort w) { setRasterPos(ctx, fv(x), fv(y), fv(z), fv(w)); }
void RasterPos4sv(Context& ctx, const GLshort* v) { setRasterPos(ctx, fv(v[0]), fv(v[1]), fv(v[2]), fv(v[3])); }

void WindowPos2d(Context& ctx, GLdouble x, GLdouble y) { setWindowPos(ctx, fv(x), fv(y), 0); }
void WindowPos2dv(Context& ctx, const GLdouble* v) { setWindowPos(ctx, fv(v[0]), fv(v[1]), 0); }
void WindowPos2f(Context& ctx, GLfloat x, GLfloat y) { setWindowPos(ctx, x, y, 0); }
void WindowPos2fv(Context& ctx, const GLfloat* v) { setWindowPos(ctx, v[0], v[1], 0); }
void WindowPos2i(Context& ctx, GLint x, GLint y) { setWindowPos(ctx, fv(x), fv(y), 0); }
void WindowPos2iv(Context& ctx, const GLint* v) { setWindowPos(ctx, fv(v[0]), fv(v[1]), 0); }
void WindowPos2s(Context& ctx, GLshort x, GLshort y) { setWindowPos(ctx, fv(x), fv(y), 0); }
void WindowPos2sv(Context& ctx, const GLshort* v) { setWindowPos(ctx, fv(v[0]), fv(v[1]), 0); }
void WindowPos3d(Context& ctx, GLdouble x, GLdouble y, GLdouble z) { setWindowPos(ctx, fv(x), fv(y), fv(z)); }
void WindowPos3dv(Context& ctx, const GLdouble* v) { setWindowPos(ctx, fv(v[0]), fv(v[1]), fv(v[2])); }
void WindowPos3f(Context& ctx, GLfloat x, GLfloat y, GLfloat z) { setWindowPos(ctx, x, y, z); }
void WindowPos3fv(Context& ctx, const GLfloat* v) { setWindowPos(ctx, v[0], v[1], v[2]); }
void WindowPos3i(Context& ctx, GLint x, GLint y, GLint z) { setWindowPos(ctx, fv(x), fv(y), fv(z)); }
void WindowPos3iv(Context& ctx, const GLint* v) { setWindowPos(ctx, fv(v[0]), fv(v[1]), fv(v[2])); }
void WindowPos3s(Context& ctx, GLshort x, GLshort y, GLshort z) { setWindowPos(ctx, fv(x), fv(y), fv(z)); }
void WindowPos3sv(Context& ctx, const GLshort* v) { setWindowPos(ctx, fv(v[0]), fv(v[1]), fv(v[2])); }

void PixelZoom(Context& ctx, GLfloat xfactor, GLfloat yfactor) {
    if (!compatOnly(ctx)) return;
    ctx.state.fixed.pixelZoom[0] = xfactor;
    ctx.state.fixed.pixelZoom[1] = yfactor;
}

// Pixel transfer scale/bias and pixel maps are accepted at their defaults
// only (no application seen in the corpus changes them).
void PixelTransferf(Context& ctx, GLenum pname, GLfloat param) {
    if (!compatOnly(ctx)) return;
    const bool scale = pname == GL_RED_SCALE || pname == GL_GREEN_SCALE || pname == GL_BLUE_SCALE ||
                       pname == GL_ALPHA_SCALE || pname == GL_DEPTH_SCALE;
    if (param != (scale ? 1.0f : 0.0f))
        logOnce(LogLevel::Warn, "glPixelTransfer", "glPixelTransfer(0x%04x, %g) is ignored", pname, static_cast<double>(param));
}
void PixelTransferi(Context& ctx, GLenum pname, GLint param) { PixelTransferf(ctx, pname, static_cast<GLfloat>(param)); }

void Bitmap(Context& ctx, GLsizei width, GLsizei height, GLfloat xorig, GLfloat yorig, GLfloat xmove, GLfloat ymove,
            const GLubyte* bitmap) {
    if (!compatOnly(ctx)) return;
    if (width < 0 || height < 0) return ctx.setError(GL_INVALID_VALUE);
    FixedState::RasterPos& r = ctx.state.fixed.rasterPos;
    if (!r.valid) return;
    if (width > 0 && height > 0) {
        const uint8_t* bits = nullptr;
        if (!bitmapSource(ctx, bitmap, width, height, bits)) return;
        std::vector<uint8_t> mask = unpackBitmap(ctx.state.unpack, bits, width, height);
        uint64_t key = hashValue(hashValue(hashValue(1469598103934665603ull, 'B'), width), height);
        key = hashBytes(key, mask.data(), mask.size());
        RefPtr<Texture> tex = cachedTexture(ctx, key, [&] {
            PixelStore tight;
            tight.alignment = 1;
            list::UnpackScope scope(ctx, tight);
            TexImage2D(ctx, GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, mask.data());
        });
        const float x0 = std::floor(r.window[0] - xorig), y0 = std::floor(r.window[1] - yorig);
        drawRectangle(ctx, Blit::Mask, tex, x0, y0, x0 + static_cast<float>(width), y0 + static_cast<float>(height),
                      r.window[2], r.color);
    }
    r.window[0] += xmove;
    r.window[1] += ymove;
}

void DrawPixels(Context& ctx, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels) {
    if (!compatOnly(ctx)) return;
    if (width < 0 || height < 0) return ctx.setError(GL_INVALID_VALUE);
    const FixedState::RasterPos& r = ctx.state.fixed.rasterPos;
    if (!r.valid || !width || !height) return;
    Blit kind = Blit::Color;
    GLenum internal = type == GL_UNSIGNED_BYTE || type == GL_UNSIGNED_INT_8_8_8_8 || type == GL_UNSIGNED_INT_8_8_8_8_REV
                          ? GL_RGBA8
                          : GL_RGBA32F;
    switch (format) {
    case GL_DEPTH_COMPONENT:
        kind = Blit::Depth;
        internal = GL_DEPTH_COMPONENT32F;
        break;
    case GL_STENCIL_INDEX: case GL_DEPTH_STENCIL: case GL_COLOR_INDEX:
        logOnce(LogLevel::Warn, "glDrawPixels", "glDrawPixels with format 0x%04x is not supported", format);
        return;
    default: break;
    }
    // Key: the parameters, the unpack state and the bytes the image reads.
    const PixelStore& unpack = ctx.state.unpack;
    uint64_t key = hashValue(hashValue(hashValue(1469598103934665603ull, 'D'), width), height);
    key = hashValue(hashValue(hashValue(key, internal), format), type);
    key = hashValue(hashValue(hashValue(hashValue(key, unpack.alignment), unpack.rowLength), unpack.skipPixels), unpack.skipRows);
    key = hashValue(hashValue(key, unpack.swapBytes), unpack.lsbFirst);
    bool cacheable = false;
    {
        const size_t bytes = transferLayout(format, type, width, height, 1, pixelStoreParams(unpack, false)).bytesNeeded;
        const uint8_t* source = static_cast<const uint8_t*>(pixels);
        if (Buffer* pbo = ctx.state.pixelUnpackBuffer.get()) {
            const uintptr_t offset = reinterpret_cast<uintptr_t>(pixels);
            source = nullptr;
            if (pbo->mtl && offset <= static_cast<size_t>(pbo->size) && bytes <= static_cast<size_t>(pbo->size) - offset) {
                if (pbo->lastWriteSerial) ctx.stream().waitFor(pbo->lastWriteSerial);
                source = pbo->contents() + offset;
            }
        }
        if (source && bytes) {
            key = hashBytes(key, source, bytes);
            cacheable = true;
        }
    }
    RefPtr<Texture> tex;
    if (cacheable)
        if (auto it = ctx.pixelRectangleCache.find(key); it != ctx.pixelRectangleCache.end()) tex = it->second;
    if (!tex) {
        const GLenum error = ctx.takeError();
        tex = temporaryTexture(ctx, [&] {
            TexImage2D(ctx, GL_TEXTURE_2D, 0, static_cast<GLint>(internal), width, height, 0, format, type, pixels);
        });
        if (const GLenum texError = ctx.takeError()) {
            ctx.setError(error ? error : texError);  // the format/type errors of glTexImage2D apply
            return;
        }
        if (error) ctx.setError(error);
        if (cacheable) {
            if (ctx.pixelRectangleCache.size() >= 512) ctx.pixelRectangleCache.clear();
            ctx.pixelRectangleCache.emplace(key, tex);
        }
    }
    const float* zoom = ctx.state.fixed.pixelZoom;
    drawRectangle(ctx, kind, tex, r.window[0], r.window[1], r.window[0] + width * zoom[0], r.window[1] + height * zoom[1],
                  r.window[2], r.color);
}

void CopyPixels(Context& ctx, GLint x, GLint y, GLsizei width, GLsizei height, GLenum type) {
    if (!compatOnly(ctx)) return;
    if (width < 0 || height < 0) return ctx.setError(GL_INVALID_VALUE);
    if (type != GL_COLOR && type != GL_DEPTH && type != GL_STENCIL && type != GL_DEPTH_STENCIL)
        return ctx.setError(GL_INVALID_ENUM);
    const FixedState::RasterPos& r = ctx.state.fixed.rasterPos;
    if (!r.valid || !width || !height) return;
    if (type == GL_STENCIL || type == GL_DEPTH_STENCIL) {
        logOnce(LogLevel::Warn, "glCopyPixels", "glCopyPixels of stencil is not supported");
        return;
    }
    const bool depth = type == GL_DEPTH;
    RefPtr<Texture> tex = temporaryTexture(ctx, [&] {
        CopyTexImage2D(ctx, GL_TEXTURE_2D, 0, depth ? GL_DEPTH_COMPONENT32F : GL_RGBA8, x, y, width, height, 0);
    });
    const float* zoom = ctx.state.fixed.pixelZoom;
    drawRectangle(ctx, depth ? Blit::Depth : Blit::Color, tex, r.window[0], r.window[1], r.window[0] + width * zoom[0],
                  r.window[1] + height * zoom[1], r.window[2], r.color);
}

}  // namespace wmgl::api
