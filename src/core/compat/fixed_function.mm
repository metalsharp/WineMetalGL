#include "core/compat/fixed_function.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

#include "core/compat/arb_programs.h"
#include "core/compat/fixed_state.h"
#include "core/context.h"
#include "core/program.h"
#include "core/texture_storage.h"
#include "core/util/log.h"

namespace wmgl::compat {
namespace {

// ------------------------------------------------------------ std140 block

// Appends values with std140 alignment, in the declaration order of
// wmgl_FixedFunction (shader/glsl_upgrade.cpp).
class Std140 {
public:
    std::vector<uint8_t> data;
    void align(size_t a) { data.resize((data.size() + a - 1) / a * a, 0); }
    void scalar(float v) {
        align(4);
        append(&v, 4);
    }
    void vec3(const float* v) {
        align(16);
        append(v, 12);
    }
    void vec4(const float* v) {
        align(16);
        append(v, 16);
    }
    void vec4(float x, float y, float z, float w) {
        const float v[4] = {x, y, z, w};
        vec4(v);
    }
    void mat4(const Mat4& m) {
        for (int c = 0; c < 4; ++c) vec4(m.m + c * 4);
    }
    void mat3(const float m[9]) {  // column major; each column padded to a vec4
        for (int c = 0; c < 3; ++c) {
            vec3(m + c * 3);
            align(16);
        }
    }
    void beginStruct() { align(16); }
    void endStruct() { align(16); }

private:
    void append(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        data.insert(data.end(), b, b + n);
    }
};

void mul4(const float a[4], const float b[4], float out[4]) {
    for (int i = 0; i < 4; ++i) out[i] = a[i] * b[i];
}

// Byte offsets of the matrix section of wmgl_FixedFunction (std140: mat4 =
// 64 bytes, mat3 = 48): modelview, projection, MVP, texture[8], normal,
// their inverses, transposes and inverse transposes. The rest of the block
// (from kTailOffset) is written by packTail.
constexpr size_t kMat4 = 64, kTex = 8 * kMat4;
constexpr size_t kMV = 0, kP = 64, kMVP = 128, kTexM = 192, kNormal = kTexM + kTex;
constexpr size_t kMVInv = kNormal + 48, kPInv = kMVInv + 64, kMVPInv = kPInv + 64, kTexInv = kMVPInv + 64;
constexpr size_t kMVT = kTexInv + kTex, kPT = kMVT + 64, kMVPT = kPT + 64, kTexT = kMVPT + 64;
constexpr size_t kMVIT = kTexT + kTex, kPIT = kMVIT + 64, kMVPIT = kPIT + 64, kTexIT = kMVPIT + 64;
constexpr size_t kTailOffset = kTexIT + kTex;

void putMat4(std::vector<uint8_t>& data, size_t offset, const Mat4& m) { std::memcpy(data.data() + offset, m.m, kMat4); }

// Recomputes the matrix section for the matrices that changed.
void packMatrices(const FixedState& f, Context::FixedFunctionBlockCache& cache) {
    const Mat4& mv = f.modelview.back();
    const Mat4& proj = f.projection.back();
    std::vector<uint8_t>& d = cache.data;
    const bool same = cache.matricesValid;
    if (!same || std::memcmp(mv.m, cache.modelview.m, kMat4) || std::memcmp(proj.m, cache.projection.m, kMat4)) {
        const Mat4 mvp = proj * mv;
        const Mat4 mvInv = mv.inverse(), projInv = proj.inverse(), mvpInv = mvp.inverse();
        putMat4(d, kMV, mv);
        putMat4(d, kP, proj);
        putMat4(d, kMVP, mvp);
        putMat4(d, kMVInv, mvInv);
        putMat4(d, kPInv, projInv);
        putMat4(d, kMVPInv, mvpInv);
        putMat4(d, kMVT, mv.transpose());
        putMat4(d, kPT, proj.transpose());
        putMat4(d, kMVPT, mvp.transpose());
        putMat4(d, kMVIT, mvInv.transpose());
        putMat4(d, kPIT, projInv.transpose());
        putMat4(d, kMVPIT, mvpInv.transpose());
        // Normal matrix: transpose of the inverse of the upper 3x3 of the modelview.
        float normal[12] = {};
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r) normal[c * 4 + r] = mvInv.m[r * 4 + c];
        std::memcpy(d.data() + kNormal, normal, sizeof(normal));
        cache.modelview = mv;
        cache.projection = proj;
        cache.modelviewInverse = mvInv;
    }
    for (int u = 0; u < FixedState::kTextureUnits; ++u) {
        const Mat4& tex = f.texture[static_cast<size_t>(u)].back();
        if (same && !std::memcmp(tex.m, cache.texture[u].m, kMat4)) continue;
        const Mat4 inv = tex.inverse();
        putMat4(d, kTexM + u * kMat4, tex);
        putMat4(d, kTexInv + u * kMat4, inv);
        putMat4(d, kTexT + u * kMat4, tex.transpose());
        putMat4(d, kTexIT + u * kMat4, inv.transpose());
        cache.texture[u] = tex;
    }
    cache.matricesValid = true;
}

// The block from gl_NormalScale on.
std::vector<uint8_t> packTail(const ContextState& st, const Mat4& mvInv) {
    const FixedState& f = st.fixed;
    Std140 b;
    // gl_NormalScale: 1 / length of the third row of the inverse modelview (GL 2.1 section 2.11.3).
    const float row = std::sqrt(mvInv.m[2] * mvInv.m[2] + mvInv.m[6] * mvInv.m[6] + mvInv.m[10] * mvInv.m[10]);
    b.scalar(row > 0 ? 1.0f / row : 1.0f);
    for (const auto& plane : f.clipPlanes) b.vec4(plane);
    b.beginStruct();  // gl_Point
    b.scalar(st.pointSize);
    b.scalar(f.pointSizeMin);
    b.scalar(f.pointSizeMax);
    b.scalar(st.pointFadeThresholdSize);
    for (float a : f.pointDistanceAttenuation) b.scalar(a);
    b.endStruct();
    for (const FixedState::Material& m : f.material) {
        b.beginStruct();
        b.vec4(m.emission);
        b.vec4(m.ambient);
        b.vec4(m.diffuse);
        b.vec4(m.specular);
        b.scalar(m.shininess);
        b.endStruct();
    }
    for (const FixedState::Light& l : f.lights) {
        b.beginStruct();
        b.vec4(l.ambient);
        b.vec4(l.diffuse);
        b.vec4(l.specular);
        b.vec4(l.position);
        // Half vector for an infinite viewer: normalize(VP + (0, 0, 1)).
        float vp[3] = {l.position[0], l.position[1], l.position[2]};
        const float len = std::sqrt(vp[0] * vp[0] + vp[1] * vp[1] + vp[2] * vp[2]);
        if (len > 0)
            for (float& c : vp) c /= len;
        vp[2] += 1;
        const float hl = std::sqrt(vp[0] * vp[0] + vp[1] * vp[1] + vp[2] * vp[2]);
        b.vec4(hl > 0 ? vp[0] / hl : 0, hl > 0 ? vp[1] / hl : 0, hl > 0 ? vp[2] / hl : 1, 0);
        b.vec3(l.spotDirection);
        b.scalar(l.spotExponent);
        b.scalar(l.spotCutoff);
        b.scalar(l.spotCutoff == 180 ? -1.0f : std::cos(l.spotCutoff * 3.14159265358979323846f / 180.0f));
        for (float a : l.attenuation) b.scalar(a);
        b.endStruct();
    }
    b.beginStruct();
    b.vec4(f.lightModelAmbient);
    b.endStruct();
    for (const FixedState::Material& m : f.material) {  // gl_Front/BackLightModelProduct
        b.beginStruct();
        float scene[4];
        mul4(m.ambient, f.lightModelAmbient, scene);
        for (int i = 0; i < 3; ++i) scene[i] += m.emission[i];
        scene[3] = m.diffuse[3];
        b.vec4(scene);
        b.endStruct();
    }
    for (const FixedState::Material& m : f.material)  // gl_Front/BackLightProduct[8]
        for (const FixedState::Light& l : f.lights) {
            float v[4];
            b.beginStruct();
            mul4(m.ambient, l.ambient, v);
            b.vec4(v);
            mul4(m.diffuse, l.diffuse, v);
            b.vec4(v);
            mul4(m.specular, l.specular, v);
            b.vec4(v);
            b.endStruct();
        }
    for (const FixedState::TexEnv& e : f.texEnv) b.vec4(e.color);
    for (int c = 0; c < 4; ++c)
        for (const FixedState::TexGen& g : f.texGen) b.vec4(g.eyePlane[c]);
    for (int c = 0; c < 4; ++c)
        for (const FixedState::TexGen& g : f.texGen) b.vec4(g.objectPlane[c]);
    b.beginStruct();  // gl_Fog
    b.vec4(f.fogColor);
    b.scalar(f.fogDensity);
    b.scalar(f.fogStart);
    b.scalar(f.fogEnd);
    b.scalar(f.fogEnd != f.fogStart ? 1.0f / (f.fogEnd - f.fogStart) : 0.0f);
    b.endStruct();
    // wmgl_AlphaTest: function code (GL_NEVER..GL_ALWAYS -> 1..8, 0 = disabled) and reference.
    b.vec4(f.alphaTest ? static_cast<float>(f.alphaFunc - GL_NEVER + 1) : 0.0f, f.alphaRef, 0, 0);
    // wmgl_LineParams: half the line width and half the viewport size (window pixels per NDC unit).
    b.vec4(std::clamp(st.lineWidth, 1.0f, kMaxLineWidth) * 0.5f, std::max(1.0f, st.viewports[0][2]) * 0.5f,
           std::max(1.0f, st.viewports[0][3]) * 0.5f, 0);
    // wmgl_Stipple: 32 rows of 32 bits.
    for (int i = 0; i < 32; i += 4) {
        float row[4];
        std::memcpy(row, &f.stipple[static_cast<size_t>(i)], 16);
        b.vec4(row);
    }
    b.align(16);
    return b.data;
}

// ------------------------------------------------------------ state key

// Base format classes for the texture environment (GL 3.3 compatibility table 3.24).
enum FormatClass : uint8_t { kFmtRGBA, kFmtRGB, kFmtAlpha, kFmtLuminance, kFmtLuminanceAlpha, kFmtIntensity };
enum TargetCode : uint8_t { kTargetNone, kTarget1D, kTarget2D, kTarget3D, kTargetCube, kTargetRect };

struct UnitKey {
    uint8_t target = kTargetNone, format = kFmtRGBA, shadow = 0, mode = 0;
    uint8_t combineRGB = 0, combineAlpha = 0, scaleRGB = 1, scaleAlpha = 1;
    uint8_t srcRGB[3] = {}, srcAlpha[3] = {}, opRGB[3] = {}, opAlpha[3] = {};
    uint8_t texgen = 0, texgenMode[4] = {};
    uint8_t coordReplace = 0;
    uint8_t coordOut = 0;  // the vertex stage writes ff_tc for an ARB fragment program without a texture enabled
};

struct Key {
    uint8_t lighting = 0, twoSide = 0, localViewer = 0, separateSpecular = 0, normalize = 0, rescale = 0;
    uint8_t colorMaterialMode = 0, colorMaterialFace = 0;  // mode 0 = off
    uint8_t lights = 0, positional = 0, spot = 0;
    uint8_t clipPlanes = 0, fog = 0, fogMode = 0, fogFromCoord = 0, pointSize = 0, flat = 0, colorSum = 0;
    uint8_t wideLines = 0, stipple = 0;
    UnitKey units[FixedState::kTextureUnits];

    std::string bytes() const {
        std::string out;
        auto put = [&](uint8_t v) { out.push_back(static_cast<char>(v)); };
        for (uint8_t v : {lighting, twoSide, localViewer, separateSpecular, normalize, rescale, colorMaterialMode,
                          colorMaterialFace, lights, positional, spot, clipPlanes, fog, fogMode, fogFromCoord, pointSize,
                          flat, colorSum, wideLines, stipple})
            put(v);
        for (const UnitKey& u : units) {
            for (uint8_t v : {u.target, u.format, u.shadow, u.mode, u.combineRGB, u.combineAlpha, u.scaleRGB, u.scaleAlpha,
                              u.texgen, u.coordReplace, u.coordOut})
                put(v);
            for (int i = 0; i < 3; ++i) {
                put(u.srcRGB[i]);
                put(u.srcAlpha[i]);
                put(u.opRGB[i]);
                put(u.opAlpha[i]);
            }
            for (uint8_t m : u.texgenMode) put(m);
        }
        return out;
    }
};

uint8_t formatClass(GLenum baseFormat) {
    switch (baseFormat) {
    case GL_ALPHA: return kFmtAlpha;
    case GL_LUMINANCE: case GL_DEPTH_COMPONENT: case GL_DEPTH_STENCIL: return kFmtLuminance;  // DEPTH_TEXTURE_MODE
    case GL_LUMINANCE_ALPHA: return kFmtLuminanceAlpha;
    case GL_INTENSITY: return kFmtIntensity;
    case GL_RGBA: case GL_BGRA: return kFmtRGBA;
    default: return kFmtRGB;  // RED, RG, RGB
    }
}

uint8_t envCode(GLenum mode) {
    switch (mode) {
    case GL_REPLACE: return 1;
    case GL_MODULATE: return 2;
    case GL_DECAL: return 3;
    case GL_BLEND: return 4;
    case GL_ADD: return 5;
    case GL_COMBINE: return 6;
    default: return 2;
    }
}

uint8_t combineCode(GLenum op) {
    switch (op) {
    case GL_REPLACE: return 1;
    case GL_MODULATE: return 2;
    case GL_ADD: return 3;
    case GL_ADD_SIGNED: return 4;
    case GL_INTERPOLATE: return 5;
    case GL_SUBTRACT: return 6;
    case GL_DOT3_RGB: return 7;
    case GL_DOT3_RGBA: return 8;
    default: return 2;
    }
}

// 0 TEXTURE, 1 CONSTANT, 2 PRIMARY_COLOR, 3 PREVIOUS, 4 + n TEXTUREn.
uint8_t sourceCode(GLenum src) {
    switch (src) {
    case GL_CONSTANT: return 1;
    case GL_PRIMARY_COLOR: return 2;
    case GL_PREVIOUS: return 3;
    case GL_TEXTURE: return 0;
    default: return static_cast<uint8_t>(4 + (src - GL_TEXTURE0));
    }
}

// 0 SRC_COLOR, 1 ONE_MINUS_SRC_COLOR, 2 SRC_ALPHA, 3 ONE_MINUS_SRC_ALPHA.
uint8_t operandCode(GLenum op) {
    switch (op) {
    case GL_ONE_MINUS_SRC_COLOR: return 1;
    case GL_SRC_ALPHA: return 2;
    case GL_ONE_MINUS_SRC_ALPHA: return 3;
    default: return 0;
    }
}

uint8_t texgenCode(GLenum mode) {
    switch (mode) {
    case GL_OBJECT_LINEAR: return 1;
    case GL_EYE_LINEAR: return 2;
    case GL_SPHERE_MAP: return 3;
    case GL_REFLECTION_MAP: return 4;
    case GL_NORMAL_MAP: return 5;
    default: return 2;
    }
}

uint8_t materialCode(GLenum mode) {
    switch (mode) {
    case GL_EMISSION: return 1;
    case GL_AMBIENT: return 2;
    case GL_DIFFUSE: return 3;
    case GL_SPECULAR: return 4;
    default: return 5;  // AMBIENT_AND_DIFFUSE
    }
}

// The texture a fixed-function unit samples: the highest-priority enabled
// target (cube map > 3D > rectangle > 2D > 1D) whose texture is complete.
Texture* unitTexture(Context& ctx, int unit, uint8_t& target) {
    const FixedState& f = ctx.state.fixed;
    const uint8_t enables = f.textureEnables[static_cast<size_t>(unit)];
    struct Choice { uint8_t bit; int index; uint8_t code; };
    const Choice order[] = {{FixedState::kTexCube, kTexCubeMap, kTargetCube}, {FixedState::kTex3D, kTex3D, kTarget3D},
                            {FixedState::kTexRect, kTexRectangle, kTargetRect}, {FixedState::kTex2D, kTex2D, kTarget2D},
                            {FixedState::kTex1D, kTex1D, kTarget1D}};
    for (const Choice& c : order) {
        if (!(enables & c.bit)) continue;
        Texture* tex = ctx.state.textureUnits[static_cast<size_t>(unit)].bound[static_cast<size_t>(c.index)].get();
        Sampler* sampler = ctx.state.textureUnits[static_cast<size_t>(unit)].sampler.get();
        if (!tex || !textureComplete(ctx, *tex, sampler ? &sampler->params : &tex->sampler)) {
            target = kTargetNone;  // an incomplete texture disables the unit
            return nullptr;
        }
        target = c.code;
        return tex;
    }
    target = kTargetNone;
    return nullptr;
}

void unitTexGen(const FixedState& f, int u, UnitKey& uk) {
    const FixedState::TexGen& g = f.texGen[static_cast<size_t>(u)];
    uk.texgen = g.enabled;
    for (int c = 0; c < 4; ++c) uk.texgenMode[c] = (g.enabled & (1u << c)) ? texgenCode(g.mode[c]) : 0;
}

Key buildKey(Context& ctx, GLenum mode) {
    const ContextState& st = ctx.state;
    const ContextState& s = st;
    const FixedState& f = st.fixed;
    Key k;
    if (f.lighting) {
        k.lighting = 1;
        k.twoSide = f.lightModelTwoSide;
        k.localViewer = f.lightModelLocalViewer;
        k.separateSpecular = f.lightModelColorControl == GL_SEPARATE_SPECULAR_COLOR;
        k.normalize = f.normalize;
        k.rescale = f.rescaleNormal && !f.normalize;
        if (f.colorMaterial) {
            k.colorMaterialMode = materialCode(f.colorMaterialMode);
            k.colorMaterialFace = f.colorMaterialFace == GL_FRONT ? 1 : f.colorMaterialFace == GL_BACK ? 2 : 3;
        }
        for (int i = 0; i < FixedState::kLights; ++i) {
            const FixedState::Light& l = f.lights[static_cast<size_t>(i)];
            if (!l.enabled) continue;
            k.lights |= 1u << i;
            if (l.position[3] != 0) k.positional |= 1u << i;
            if (l.position[3] != 0 && l.spotCutoff != 180) k.spot |= 1u << i;
        }
    }
    k.clipPlanes = static_cast<uint8_t>(st.clipDistanceEnables & 0xff);
    if (f.fog) {
        k.fog = 1;
        k.fogMode = f.fogMode == GL_LINEAR ? 0 : f.fogMode == GL_EXP ? 1 : 2;
        k.fogFromCoord = f.fogCoordSource == GL_FOG_COORD;
    }
    if (mode == GL_POINTS) k.pointSize = fixedFunctionWritesPointSize(ctx);
    k.flat = f.shadeModel == GL_FLAT;
    k.colorSum = f.colorSum || f.lighting;
    k.wideLines = fixedFunctionWideLines(ctx, mode);
    if (k.wideLines) k.twoSide = 0;  // lines are front-facing
    const bool polygons = mode != GL_POINTS && mode != GL_LINES && mode != GL_LINE_STRIP && mode != GL_LINE_LOOP &&
                          mode != GL_LINES_ADJACENCY && mode != GL_LINE_STRIP_ADJACENCY;
    k.stipple = f.polygonStipple && polygons && s.polygonMode == GL_FILL;
    for (int u = 0; u < FixedState::kTextureUnits; ++u) {
        UnitKey& uk = k.units[u];
        Texture* tex = unitTexture(ctx, u, uk.target);
        if (!tex) continue;
        const TextureImage& image = tex->baseImage();
        uk.format = image.format ? formatClass(image.format->baseFormat) : kFmtRGBA;
        Sampler* sampler = st.textureUnits[static_cast<size_t>(u)].sampler.get();
        const SamplerParams& params = sampler ? sampler->params : tex->sampler;
        uk.shadow = image.format && image.format->isDepth() && params.compareMode == GL_COMPARE_REF_TO_TEXTURE &&
                    uk.target != kTarget3D && uk.target != kTargetCube;
        const FixedState::TexEnv& e = f.texEnv[static_cast<size_t>(u)];
        uk.mode = envCode(e.mode);
        if (e.mode == GL_COMBINE) {
            uk.combineRGB = combineCode(e.combineRGB);
            uk.combineAlpha = combineCode(e.combineAlpha);
            uk.scaleRGB = static_cast<uint8_t>(e.scaleRGB);
            uk.scaleAlpha = static_cast<uint8_t>(e.scaleAlpha);
            for (int i = 0; i < 3; ++i) {
                uk.srcRGB[i] = sourceCode(e.sourceRGB[i]);
                uk.srcAlpha[i] = sourceCode(e.sourceAlpha[i]);
                uk.opRGB[i] = operandCode(e.operandRGB[i]);
                uk.opAlpha[i] = operandCode(e.operandAlpha[i]);
            }
        }
        unitTexGen(f, u, uk);
        uk.coordReplace = mode == GL_POINTS && f.pointSprite && e.coordReplace;
    }
    return k;
}

// ------------------------------------------------------------ GLSL

std::string str(int v) { return std::to_string(v); }

const char* samplerType(const UnitKey& u) {
    switch (u.target) {
    case kTarget1D: return u.shadow ? "sampler1DShadow" : "sampler1D";
    case kTarget3D: return "sampler3D";
    case kTargetCube: return "samplerCube";
    case kTargetRect: return u.shadow ? "sampler2DRectShadow" : "sampler2DRect";
    default: return u.shadow ? "sampler2DShadow" : "sampler2D";
    }
}

bool needsNormal(const Key& k) {
    if (k.lighting) return true;
    for (const UnitKey& u : k.units)
        for (uint8_t m : u.texgenMode)
            if (m >= 3) return true;
    return false;
}

// Lighting of one face into `color`/`spec` (GL 3.3 compatibility section 2.13.1).
std::string lightFace(const Key& k, bool back) {
    const std::string mat = back ? "gl_BackMaterial" : "gl_FrontMaterial";
    const bool cm = k.colorMaterialMode && (k.colorMaterialFace & (back ? 2 : 1));
    auto prop = [&](const char* name, int code) {
        const bool tracked = cm && (k.colorMaterialMode == code || (k.colorMaterialMode == 5 && (code == 2 || code == 3)));
        return tracked ? std::string("gl_Color") : mat + "." + name;
    };
    std::string s;
    const std::string face = back ? "b" : "f";
    s += "    {\n";
    s += "        vec3 n = " + std::string(back ? "-normal" : "normal") + ";\n";
    s += "        vec4 matEmission = " + prop("emission", 1) + ";\n";
    s += "        vec4 matAmbient = " + prop("ambient", 2) + ";\n";
    s += "        vec4 matDiffuse = " + prop("diffuse", 3) + ";\n";
    s += "        vec4 matSpecular = " + prop("specular", 4) + ";\n";
    s += "        vec3 c = matEmission.rgb + matAmbient.rgb * gl_LightModel.ambient.rgb;\n";
    s += "        vec3 spec = vec3(0.0);\n";
    for (int i = 0; i < FixedState::kLights; ++i) {
        if (!(k.lights & (1u << i))) continue;
        const std::string L = "gl_LightSource[" + str(i) + "]";
        s += "        {\n";
        s += "            float att = 1.0;\n";
        if (k.positional & (1u << i)) {
            s += "            vec3 vp = " + L + ".position.xyz / " + L + ".position.w - eyePos;\n";
            s += "            float d = length(vp);\n";
            s += "            vp = d > 0.0 ? vp / d : vec3(0.0, 0.0, 1.0);\n";
            s += "            att = 1.0 / max(" + L + ".constantAttenuation + " + L + ".linearAttenuation * d + " + L +
                 ".quadraticAttenuation * d * d, 1e-20);\n";
            if (k.spot & (1u << i)) {
                s += "            float sd = dot(-vp, normalize(" + L + ".spotDirection));\n";
                s += "            att *= sd >= " + L + ".spotCosCutoff ? ff_pow(max(sd, 0.0), " + L + ".spotExponent) : 0.0;\n";
            }
        } else {
            s += "            vec3 vp = normalize(" + L + ".position.xyz);\n";
        }
        s += "            float ndl = max(dot(n, vp), 0.0);\n";
        s += k.localViewer ? "            vec3 h = normalize(vp + normalize(-eyePos));\n"
                           : "            vec3 h = normalize(vp + vec3(0.0, 0.0, 1.0));\n";
        s += "            c += att * (" + L + ".ambient.rgb * matAmbient.rgb + ndl * " + L + ".diffuse.rgb * matDiffuse.rgb);\n";
        s += "            if (ndl > 0.0) spec += att * ff_pow(max(dot(n, h), 0.0), " + mat + ".shininess) * " + L +
             ".specular.rgb * matSpecular.rgb;\n";
        s += "        }\n";
    }
    if (k.separateSpecular) {
        s += "        ff_" + std::string(back ? "back" : "front") + " = clamp(vec4(c, matDiffuse.a), 0.0, 1.0);\n";
        s += "        ff_" + std::string(back ? "backSecondary" : "frontSecondary") + " = clamp(vec4(spec, 0.0), 0.0, 1.0);\n";
    } else {
        s += "        ff_" + std::string(back ? "back" : "front") + " = clamp(vec4(c + spec, matDiffuse.a), 0.0, 1.0);\n";
        s += "        ff_" + std::string(back ? "backSecondary" : "frontSecondary") + " = vec4(0.0);\n";
    }
    s += "    }\n";
    (void)face;
    return s;
}

std::string vertexSource(const Key& k) {
    const std::string interp = k.flat ? "flat out" : "out";
    std::string s = "#version 130\n";
    s += interp + " vec4 ff_front;\n" + interp + " vec4 ff_frontSecondary;\n";
    if (k.twoSide && k.lighting) s += interp + " vec4 ff_back;\n" + interp + " vec4 ff_backSecondary;\n";
    for (int u = 0; u < FixedState::kTextureUnits; ++u)
        if ((k.units[u].target || k.units[u].coordOut) && !k.units[u].coordReplace) s += "out vec4 ff_tc" + str(u) + ";\n";
    if (k.fog) s += "out float ff_fogCoord;\n";
    s += "float ff_pow(float x, float y) { return y == 0.0 ? 1.0 : pow(x, y); }\n";
    if (k.wideLines) s += "in vec4 wmgl_LinePartner;\nin float wmgl_LineSide;\n";
    s += "void main() {\n";
    s += "    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n";
    if (k.wideLines) {
        // Offset along the segment's window-space minor axis by half the width.
        s += "    vec4 ff_other = gl_ModelViewProjectionMatrix * wmgl_LinePartner;\n";
        s += "    vec2 ff_delta = (ff_other.xy / ff_other.w - gl_Position.xy / gl_Position.w) * wmgl_LineParams.yz;\n";
        s += "    vec2 ff_offset = abs(ff_delta.x) >= abs(ff_delta.y) ? vec2(0.0, wmgl_LineSide) : vec2(wmgl_LineSide, 0.0);\n";
        s += "    gl_Position.xy += ff_offset * wmgl_LineParams.x / wmgl_LineParams.yz * gl_Position.w;\n";
    }
    s += "    vec4 eye = gl_ModelViewMatrix * gl_Vertex;\n";
    s += "    vec3 eyePos = eye.xyz / eye.w;\n";
    if (needsNormal(k)) {
        s += "    vec3 normal = gl_NormalMatrix * gl_Normal;\n";
        if (k.normalize) s += "    normal = normalize(normal);\n";
        else if (k.rescale) s += "    normal *= gl_NormalScale;\n";
    }
    if (k.lighting) {
        s += lightFace(k, false);
        if (k.twoSide) s += lightFace(k, true);
    } else {
        s += "    ff_front = clamp(gl_Color, 0.0, 1.0);\n";
        s += k.colorSum ? "    ff_frontSecondary = clamp(gl_SecondaryColor, 0.0, 1.0);\n" : "    ff_frontSecondary = vec4(0.0);\n";
    }
    bool sphere = false, reflection = false;
    for (const UnitKey& u : k.units)
        for (uint8_t m : u.texgenMode) {
            sphere |= m == 3;
            reflection |= m == 3 || m == 4;
        }
    if (reflection) s += "    vec3 ff_r = reflect(normalize(eyePos), normalize(normal));\n";
    if (sphere) {
        s += "    float ff_m = 2.0 * sqrt(ff_r.x * ff_r.x + ff_r.y * ff_r.y + (ff_r.z + 1.0) * (ff_r.z + 1.0));\n";
        s += "    vec2 ff_sphere = ff_m > 0.0 ? ff_r.xy / ff_m + 0.5 : vec2(0.5);\n";
    }
    static const char* const kCoord = "STRQ";
    static const char* const kComp = "xyzw";
    for (int u = 0; u < FixedState::kTextureUnits; ++u) {
        const UnitKey& uk = k.units[u];
        if ((!uk.target && !uk.coordOut) || uk.coordReplace) continue;
        s += "    {\n        vec4 tc = gl_MultiTexCoord" + str(u) + ";\n";
        for (int c = 0; c < 4; ++c) {
            const std::string dst = std::string("        tc.") + kComp[c] + " = ";
            switch (uk.texgenMode[c]) {
            case 1: s += dst + "dot(gl_Vertex, gl_ObjectPlane" + kCoord[c] + "[" + str(u) + "]);\n"; break;
            case 2: s += dst + "dot(eye, gl_EyePlane" + kCoord[c] + "[" + str(u) + "]);\n"; break;
            case 3: s += dst + (c < 2 ? std::string("ff_sphere.") + kComp[c] : std::string("0.0")) + ";\n"; break;
            case 4: s += dst + "ff_r." + kComp[c] + ";\n"; break;
            case 5: s += dst + "normal." + kComp[c] + ";\n"; break;
            default: break;
            }
        }
        s += "        ff_tc" + str(u) + " = gl_TextureMatrix[" + str(u) + "] * tc;\n    }\n";
    }
    if (k.fog) s += k.fogFromCoord ? "    ff_fogCoord = gl_FogCoord;\n" : "    ff_fogCoord = abs(eyePos.z);\n";
    for (int i = 0; i < FixedState::kClipPlanes; ++i)
        if (k.clipPlanes & (1u << i)) s += "    gl_ClipDistance[" + str(i) + "] = dot(eye, gl_ClipPlane[" + str(i) + "]);\n";
    if (k.pointSize) {
        s += "    float ff_d = length(eyePos);\n";
        s += "    float ff_size = gl_Point.size * inversesqrt(max(gl_Point.distanceConstantAttenuation + "
             "gl_Point.distanceLinearAttenuation * ff_d + gl_Point.distanceQuadraticAttenuation * ff_d * ff_d, 1e-20));\n";
        s += "    gl_PointSize = clamp(ff_size, max(gl_Point.sizeMin, 1.0), gl_Point.sizeMax);\n";
    }
    s += "}\n";
    return s;
}

std::string sample(const UnitKey& u, int unit) {
    const std::string sampler = "ff_tex" + str(unit);
    const std::string coord = u.coordReplace ? "vec4(gl_PointCoord, 0.0, 1.0)" : "ff_tc" + str(unit);
    if (u.target == kTargetCube) return "texture(" + sampler + ", " + coord + ".xyz)";
    if (u.shadow) return "vec4(vec3(textureProj(" + sampler + ", " + coord + ")), 1.0)";
    return "textureProj(" + sampler + ", " + coord + ")";
}

std::string combineSource(const UnitKey& u, int unit, int i, bool alpha) {
    const uint8_t src = alpha ? u.srcAlpha[i] : u.srcRGB[i];
    std::string v;
    switch (src) {
    case 0: v = "t" + str(unit); break;
    case 1: v = "gl_TextureEnvColor[" + str(unit) + "]"; break;
    case 2: v = "primary"; break;
    case 3: v = "prev"; break;
    default: v = "t" + str(src - 4); break;
    }
    const uint8_t op = alpha ? u.opAlpha[i] : u.opRGB[i];
    if (alpha) return op == 3 || op == 1 ? "(1.0 - " + v + ".a)" : v + ".a";
    switch (op) {
    case 1: return "(vec3(1.0) - " + v + ".rgb)";
    case 2: return "vec3(" + v + ".a)";
    case 3: return "vec3(1.0 - " + v + ".a)";
    default: return v + ".rgb";
    }
}

std::string combineOp(uint8_t op, bool alpha, const std::string& a0, const std::string& a1, const std::string& a2) {
    const std::string half = alpha ? "0.5" : "vec3(0.5)";
    switch (op) {
    case 1: return a0;
    case 3: return a0 + " + " + a1;
    case 4: return a0 + " + " + a1 + " - " + half;
    case 5: return a0 + " * " + a2 + " + " + a1 + " * (" + (alpha ? "1.0" : "vec3(1.0)") + " - " + a2 + ")";
    case 6: return a0 + " - " + a1;
    case 7: case 8: return "vec3(4.0 * dot(" + a0 + " - vec3(0.5), " + a1 + " - vec3(0.5)))";
    default: return a0 + " * " + a1;
    }
}

std::string fragmentSource(const Key& k) {
    const std::string interp = k.flat ? "flat in" : "in";
    std::string s = "#version 130\n";
    s += interp + " vec4 ff_front;\n" + interp + " vec4 ff_frontSecondary;\n";
    const bool twoSide = k.twoSide && k.lighting;
    if (twoSide) s += interp + " vec4 ff_back;\n" + interp + " vec4 ff_backSecondary;\n";
    for (int u = 0; u < FixedState::kTextureUnits; ++u) {
        if (!k.units[u].target) continue;
        if (!k.units[u].coordReplace) s += "in vec4 ff_tc" + str(u) + ";\n";
        s += std::string("uniform ") + samplerType(k.units[u]) + " ff_tex" + str(u) + ";\n";
    }
    if (k.fog) s += "in float ff_fogCoord;\n";
    s += "void main() {\n";
    if (k.stipple) {
        s += "    uvec2 ff_p = uvec2(gl_FragCoord.xy) % 32u;\n";
        s += "    if (((wmgl_Stipple[ff_p.y >> 2u][ff_p.y & 3u] >> ff_p.x) & 1u) == 0u) discard;\n";
    }
    s += twoSide ? "    vec4 primary = gl_FrontFacing ? ff_front : ff_back;\n"
                   "    vec4 secondary = gl_FrontFacing ? ff_frontSecondary : ff_backSecondary;\n"
                 : "    vec4 primary = ff_front;\n    vec4 secondary = ff_frontSecondary;\n";
    for (int u = 0; u < FixedState::kTextureUnits; ++u)
        s += "    vec4 t" + str(u) + " = " + (k.units[u].target ? sample(k.units[u], u) : std::string("vec4(1.0)")) + ";\n";
    s += "    vec4 prev = primary;\n";
    for (int u = 0; u < FixedState::kTextureUnits; ++u) {
        const UnitKey& uk = k.units[u];
        if (!uk.target) continue;
        const std::string t = "t" + str(u), env = "gl_TextureEnvColor[" + str(u) + "]";
        const bool hasColor = uk.format != kFmtAlpha;
        const bool hasAlpha = uk.format == kFmtAlpha || uk.format == kFmtLuminanceAlpha || uk.format == kFmtIntensity ||
                              uk.format == kFmtRGBA;
        const bool intensity = uk.format == kFmtIntensity;
        s += "    // unit " + str(u) + "\n";
        switch (uk.mode) {
        case 1:  // REPLACE
            if (hasColor) s += "    prev.rgb = " + t + ".rgb;\n";
            if (hasAlpha) s += "    prev.a = " + t + ".a;\n";
            break;
        case 3:  // DECAL
            if (uk.format == kFmtRGBA) s += "    prev.rgb = mix(prev.rgb, " + t + ".rgb, " + t + ".a);\n";
            else if (uk.format == kFmtRGB) s += "    prev.rgb = " + t + ".rgb;\n";
            break;
        case 4:  // BLEND
            if (hasColor) s += "    prev.rgb = mix(prev.rgb, " + env + ".rgb, " + t + ".rgb);\n";
            if (intensity) s += "    prev.a = mix(prev.a, " + env + ".a, " + t + ".a);\n";
            else if (hasAlpha) s += "    prev.a *= " + t + ".a;\n";
            break;
        case 5:  // ADD
            if (hasColor) s += "    prev.rgb = min(prev.rgb + " + t + ".rgb, vec3(1.0));\n";
            if (intensity) s += "    prev.a = min(prev.a + " + t + ".a, 1.0);\n";
            else if (hasAlpha) s += "    prev.a *= " + t + ".a;\n";
            break;
        case 6: {  // COMBINE
            const std::string rgb = combineOp(uk.combineRGB, false, combineSource(uk, u, 0, false),
                                              combineSource(uk, u, 1, false), combineSource(uk, u, 2, false));
            s += "    {\n        vec3 rgb = " + rgb + ";\n";
            if (uk.combineRGB == 8) {
                s += "        float a = rgb.r;\n";  // DOT3_RGBA writes alpha too
            } else {
                const std::string a = combineOp(uk.combineAlpha, true, combineSource(uk, u, 0, true),
                                                combineSource(uk, u, 1, true), combineSource(uk, u, 2, true));
                s += "        float a = " + a + ";\n";
            }
            s += "        prev = clamp(vec4(rgb * " + str(uk.scaleRGB) + ".0, a * " + str(uk.scaleAlpha) + ".0), 0.0, 1.0);\n";
            s += "    }\n";
            break;
        }
        default:  // MODULATE
            if (hasColor) s += "    prev.rgb *= " + t + ".rgb;\n";
            if (hasAlpha) s += "    prev.a *= " + t + ".a;\n";
            break;
        }
    }
    if (k.colorSum) s += "    prev.rgb = min(prev.rgb + secondary.rgb, vec3(1.0));\n";
    if (k.fog) {
        switch (k.fogMode) {
        case 0: s += "    float f = (gl_Fog.end - ff_fogCoord) * gl_Fog.scale;\n"; break;
        case 1: s += "    float f = exp(-gl_Fog.density * ff_fogCoord);\n"; break;
        default: s += "    float f = exp(-(gl_Fog.density * ff_fogCoord) * (gl_Fog.density * ff_fogCoord));\n"; break;
        }
        s += "    prev.rgb = mix(gl_Fog.color.rgb, prev.rgb, clamp(f, 0.0, 1.0));\n";
    }
    s += "    gl_FragColor = prev;\n}\n";
    return s;
}

}  // namespace

const std::vector<uint8_t>& fixedFunctionBlock(Context& ctx, uint64_t& generation) {
    static std::atomic<uint64_t> generations{0};
    Context::FixedFunctionBlockCache& cache = ctx.fixedFunctionBlockCache;
    const ContextState& st = ctx.state;
    if (!cache.generation || cache.version != st.fixed.version || cache.pointSize != st.pointSize ||
        cache.fade != st.pointFadeThresholdSize || cache.lineWidth != st.lineWidth ||
        !std::equal(st.viewports[0], st.viewports[0] + 4, cache.viewport)) {
        if (cache.data.size() < kTailOffset) cache.data.assign(kTailOffset, 0);
        packMatrices(st.fixed, cache);
        const std::vector<uint8_t> tail = packTail(st, cache.modelviewInverse);
        cache.data.resize(kTailOffset + tail.size());
        std::memcpy(cache.data.data() + kTailOffset, tail.data(), tail.size());
        cache.version = st.fixed.version;
        cache.pointSize = st.pointSize;
        cache.fade = st.pointFadeThresholdSize;
        cache.lineWidth = st.lineWidth;
        std::copy(st.viewports[0], st.viewports[0] + 4, cache.viewport);
        cache.generation = ++generations;
    }
    generation = cache.generation;
    return cache.data;
}

RefPtr<Program> linkInternalProgram(const std::string& vertex, const std::string& fragment, const char* samplerPrefix,
                                    std::string& log, const std::map<std::string, GLuint>& attribBindings) {
    shader::LinkInput input;
    input.attribBindings = attribBindings;
    input.shaders.push_back({shader::Stage::Vertex, vertex});
    input.shaders.push_back({shader::Stage::Fragment, fragment});
    shader::LinkResult result = shader::link(input);
    if (!result.ok) {
        log = result.log;
        return nullptr;
    }
    RefPtr<Program> p = makeRef<Program>(0);
    auto linked = std::make_shared<LinkedProgram>();
    linked->link = std::move(result);
    linked->uid = nextLinkedProgramId();
    p->uniformData.assign((linked->link.defaultBlockSize + 15) & ~15u, 0);
    p->uniformBlockBindings.assign(linked->link.blocks.size(), 0);
    p->samplerUnits.assign(linked->link.samplers.size(), 0);
    const size_t prefix = std::strlen(samplerPrefix);
    for (const shader::UniformInfo& u : linked->link.uniforms)
        if (u.samplerSlot >= 0 && u.name.compare(0, prefix, samplerPrefix) == 0)
            p->samplerUnits[static_cast<size_t>(u.samplerSlot)] = std::atoi(u.name.c_str() + prefix);
    p->linkStatus = true;
    p->linkAttempted = true;
    p->linked = std::move(linked);
    return p;
}

void fixedFunctionSources(Context& ctx, GLenum mode, std::string& vertex, std::string& fragment) {
    const Key key = buildKey(ctx, mode);
    vertex = vertexSource(key);
    fragment = fragmentSource(key);
}

void fixedFunctionStage(Context& ctx, GLenum mode, bool vertex, const FixedStageNeeds& needs, std::string& key,
                        std::string* source) {
    const FixedState& f = ctx.state.fixed;
    const Key full = buildKey(ctx, mode);
    Key k;  // only the fields of the stage, so the other stage's state does not multiply the variants
    if (vertex) {
        k = full;
        k.stipple = 0;
        k.fogMode = 0;
        for (int u = 0; u < FixedState::kTextureUnits; ++u) {
            UnitKey& uk = k.units[u];
            const UnitKey from = uk;
            uk = UnitKey();
            uk.target = from.target ? 1 : 0;  // the vertex stage only tests for a target
            uk.coordReplace = from.coordReplace;
            if (from.target || (needs.texCoords >> u & 1u)) unitTexGen(f, u, uk);
            uk.coordOut = needs.texCoords >> u & 1u;
        }
        if (needs.fogCoord && !k.fog) {
            k.fog = 1;
            k.fogFromCoord = f.fogCoordSource == GL_FOG_COORD;
        }
        key = "v" + k.bytes();
        if (source) *source = vertexSource(k);
    } else {
        k.flat = full.flat;
        k.fog = full.fog;
        k.fogMode = full.fogMode;
        k.stipple = full.stipple;
        k.colorSum = f.colorSum;  // a vertex program's colour sum follows COLOR_SUM alone
        k.twoSide = k.lighting = needs.twoSide;  // fragmentSource selects back colours for twoSide && lighting
        for (int u = 0; u < FixedState::kTextureUnits; ++u) {
            k.units[u] = full.units[u];
            k.units[u].texgen = 0;
            std::fill(std::begin(k.units[u].texgenMode), std::end(k.units[u].texgenMode), 0);
        }
        key = "f" + k.bytes();
        if (source) *source = fragmentSource(k);
    }
}

bool fixedFunctionTwoSided(Context& ctx, GLenum mode) {
    const Key k = buildKey(ctx, mode);
    return k.twoSide && k.lighting;
}

bool fixedFunctionWideLines(Context& ctx, GLenum mode) {
    const ContextState& s = ctx.state;
    if (!ctx.isCompat() || s.program || s.programPipeline || std::clamp(s.lineWidth, 1.0f, kMaxLineWidth) <= 1.0f) return false;
    if (arbVertexProgramActive(ctx)) return false;  // a vertex program draws lines one pixel wide
    if (mode != GL_LINES && mode != GL_LINE_STRIP && mode != GL_LINE_LOOP) return false;
    return !(s.transformFeedback && s.transformFeedback->active);  // capture records the line, not its quads
}

float textureUnitLodBias(const Context& ctx, GLint unit) {
    if (!ctx.isCompat() || unit < 0 || unit >= FixedState::kTextureUnits) return 0;
    return ctx.state.fixed.texEnv[static_cast<size_t>(unit)].lodBias;
}

bool fixedFunctionWritesPointSize(Context& ctx) {
    // Distance attenuation, or a glPointSize outside [POINT_SIZE_MIN,
    // POINT_SIZE_MAX]: the derived size is always clamped to that range
    // (GL 3.3 compatibility section 3.4).
    const FixedState& f = ctx.state.fixed;
    const float* a = f.pointDistanceAttenuation;
    const float size = ctx.state.pointSize;
    return a[0] != 1 || a[1] != 0 || a[2] != 0 || size < std::max(f.pointSizeMin, 1.0f) || size > f.pointSizeMax;
}

Program* fixedFunctionProgram(Context& ctx, GLenum mode) {
    const Key key = buildKey(ctx, mode);
    const std::string bytes = key.bytes();
    auto& cache = ctx.share().fixedFunctionPrograms;
    auto it = cache.find(bytes);
    if (it != cache.end()) return it->second.get();
    const std::string vs = vertexSource(key), fs = fragmentSource(key);
    std::string log;
    RefPtr<Program> program = linkInternalProgram(vs, fs, "ff_tex", log);
    if (!program) {
        WMGL_ERR(Shader, "fixed-function program failed to link:\n%s\n--- vertex\n%s--- fragment\n%s", log.c_str(),
                 vs.c_str(),
                 fs.c_str());
    } else {
        WMGL_TRACE(Shader, "fixed-function program %zu built", cache.size() + 1);
    }
    Program* raw = program.get();
    cache.emplace(bytes, std::move(program));  // failures are cached too (logged once)
    return raw;
}

}  // namespace wmgl::compat
