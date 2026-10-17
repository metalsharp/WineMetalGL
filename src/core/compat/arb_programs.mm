#include "core/compat/arb_programs.h"

#include <algorithm>
#include <cstring>
#include <map>

#include "core/compat/fixed_function.h"
#include "core/context.h"
#include "core/program.h"
#include "core/util/log.h"

namespace wmgl::compat {
namespace {

const ArbProgramObject* boundConst(const Context& ctx, int i) {
    if (ctx.arb.bound[i]) return ctx.arb.bound[i].get();
    return ctx.arb.defaults[i].get();
}

bool noShaderProgram(const Context& ctx) {
    return ctx.isCompat() && !ctx.state.program && !ctx.state.programPipeline;
}

bool stageEnabled(const Context& ctx, int i) {
    const FixedState& f = ctx.state.fixed;
    return i == 0 ? f.vertexProgram : f.fragmentProgram;
}

bool loaded(const Context& ctx, int i) {
    const ArbProgramObject* p = boundConst(ctx, i);
    return p && p->parsed && p->parsed->ok;
}

// The default-block offset of uniform array `name` ("a[0]") and, in
// `count`, the elements the program kept (at most `count`); -1 if absent.
int uniformOffset(const Program& p, const std::string& name, int stride, int& count) {
    for (const shader::UniformInfo& u : p.linked->link.uniforms)
        if (u.name == name && u.blockIndex < 0 && u.offset >= 0 && (u.size == 1 || u.arrayStride == stride)) {
            count = std::min(count, static_cast<int>(u.size));
            return u.offset;
        }
    count = 0;
    return -1;
}

// Writes `bytes` at `offset` of the default uniforms; bumps the version when they change.
void store(Program& p, int offset, const void* data, size_t bytes) {
    if (offset < 0 || static_cast<size_t>(offset) + bytes > p.uniformData.size()) return;
    uint8_t* at = p.uniformData.data() + offset;
    if (std::memcmp(at, data, bytes) == 0) return;
    std::memcpy(at, data, bytes);
    ++p.uniformVersion;
}

}  // namespace

int arbTargetIndex(GLenum target) {
    return target == GL_VERTEX_PROGRAM_ARB ? 0 : target == GL_FRAGMENT_PROGRAM_ARB ? 1 : -1;
}

ArbProgramObject& arbBound(Context& ctx, int i) {
    std::shared_ptr<ArbProgramObject>& b = ctx.arb.bound[i];
    if (b) return *b;
    std::shared_ptr<ArbProgramObject>& d = ctx.arb.defaults[i];
    if (!d) {
        d = std::make_shared<ArbProgramObject>();
        d->target = i == 0 ? GL_VERTEX_PROGRAM_ARB : GL_FRAGMENT_PROGRAM_ARB;
    }
    return *d;
}

bool arbProgramsEnabled(const Context& ctx) {
    return noShaderProgram(ctx) && (ctx.state.fixed.vertexProgram || ctx.state.fixed.fragmentProgram);
}

bool arbProgramsInvalid(const Context& ctx) {
    if (!arbProgramsEnabled(ctx)) return false;
    for (int i = 0; i < 2; ++i)
        if (stageEnabled(ctx, i) && !loaded(ctx, i)) return true;
    return false;
}

bool arbVertexProgramActive(const Context& ctx) {
    return noShaderProgram(ctx) && ctx.state.fixed.vertexProgram && loaded(ctx, 0);
}

Program* arbDrawProgram(Context& ctx, GLenum mode) {
    const FixedState& f = ctx.state.fixed;
    const ArbProgramObject* stage[2] = {};
    for (int i = 0; i < 2; ++i)
        if (stageEnabled(ctx, i)) {
            stage[i] = &arbBound(ctx, i);
            if (!stage[i]->parsed || !stage[i]->parsed->ok) return nullptr;  // validateDraw reported it
        }
    const arb::Parsed* vp = stage[0] ? stage[0]->parsed.get() : nullptr;
    const arb::Parsed* fp = stage[1] ? stage[1]->parsed.get() : nullptr;

    arb::EmitOptions vo, fo;
    vo.flat = fo.flat = f.shadeModel == GL_FLAT;
    vo.clipPlanes = static_cast<uint8_t>(ctx.state.clipDistanceEnables & 0xff);
    // Back colours: VERTEX_PROGRAM_TWO_SIDE after a vertex program, two-sided lighting after fixed-function.
    fo.twoSide = vp ? f.vertexProgramTwoSide : fixedFunctionTwoSided(ctx, mode);
    if (mode == GL_POINTS && f.pointSprite)
        for (int u = 0; u < FixedState::kTextureUnits; ++u)
            if (f.texEnv[static_cast<size_t>(u)].coordReplace) fo.pointCoordReplace |= 1u << u;

    FixedStageNeeds needs;
    if (fp) {
        needs.texCoords = fp->texCoordRead;
        needs.fogCoord = fp->fogCoordRead || fp->fog;
    }
    needs.twoSide = f.vertexProgramTwoSide;
    std::string key;
    std::string fixedKey;
    if (vp) {
        key += "V" + std::to_string(stage[0]->serial) + ":" + arb::emitKey(*vp, vo);
    } else {
        fixedFunctionStage(ctx, mode, true, needs, fixedKey, nullptr);
        key += fixedKey;
    }
    key += '|';
    if (fp) {
        key += "F" + std::to_string(stage[1]->serial) + ":" + arb::emitKey(*fp, fo);
    } else {
        fixedFunctionStage(ctx, mode, false, needs, fixedKey, nullptr);
        key += fixedKey;
    }

    auto& cache = ctx.share().arbLinkedPrograms;
    auto it = cache.find(key);
    if (it == cache.end()) {
        // Programs reloaded or deleted leave entries behind: start over past a bound.
        if (cache.size() >= 1024) cache.clear();
        std::string vs, fs, unused;
        if (vp) vs = arb::emit(*vp, vo);
        else fixedFunctionStage(ctx, mode, true, needs, unused, &vs);
        if (fp) fs = arb::emit(*fp, fo);
        else fixedFunctionStage(ctx, mode, false, needs, unused, &fs);
        std::map<std::string, GLuint> bindings;
        for (int n = 1; vp && n < arb::Limits::kVertexAttribs; ++n)
            if (vp->genericRead >> n & 1u) bindings["arb_attrib" + std::to_string(n)] = static_cast<GLuint>(n);
        std::string log;
        ArbLinked linked;
        linked.program = linkInternalProgram(vs, fs, "ff_tex", log, bindings);
        if (!linked.program) {
            WMGL_ERR("ARB program failed to link:\n%s\n--- vertex\n%s--- fragment\n%s", log.c_str(), vs.c_str(), fs.c_str());
        } else {
            WMGL_TRACE("ARB program %zu built", cache.size() + 1);
            Program& p = *linked.program;
            for (int i = 0; i < 2; ++i) {
                const arb::Parsed* parsed = i == 0 ? vp : fp;
                if (!parsed) continue;
                linked.parsed[i] = stage[i]->parsed;
                const arb::Kind kind = i == 0 ? arb::Kind::Vertex : arb::Kind::Fragment;
                linked.envCount[i] = parsed->envCount;
                linked.localCount[i] = parsed->localCount;
                if (parsed->envCount)
                    linked.envOffset[i] = uniformOffset(p, std::string(arb::envName(kind, true)) + "[0]", 16, linked.envCount[i]);
                if (parsed->localCount)
                    linked.localOffset[i] =
                        uniformOffset(p, std::string(arb::envName(kind, false)) + "[0]", 16, linked.localCount[i]);
            }
            int matrices = FixedState::kProgramMatrices, inverses = FixedState::kProgramMatrices;
            if ((vp && vp->programMatrices) || (fp && fp->programMatrices))
                linked.matrixOffset = uniformOffset(p, "arb_pm[0]", 64, matrices);
            if ((vp && vp->programMatrixInverses) || (fp && fp->programMatrixInverses))
                linked.inverseOffset = uniformOffset(p, "arb_pmInv[0]", 64, inverses);
            linked.matrixCount = linked.matrixOffset >= 0 ? matrices : 0;
            linked.inverseCount = linked.inverseOffset >= 0 ? inverses : 0;
        }
        it = cache.emplace(key, std::move(linked)).first;  // failures are cached too (logged once)
    }
    ArbLinked& linked = it->second;
    if (!linked.program) return nullptr;
    Program& p = *linked.program;
    if (p.uniformData.size() < p.linked->link.defaultBlockSize) p.uniformData.resize(p.linked->link.defaultBlockSize, 0);
    for (int i = 0; i < 2; ++i) {
        if (!stage[i]) continue;
        store(p, linked.envOffset[i], ctx.arb.env[i].data(), static_cast<size_t>(linked.envCount[i]) * 16);
        store(p, linked.localOffset[i], stage[i]->local.data(), static_cast<size_t>(linked.localCount[i]) * 16);
    }
    if (linked.matrixOffset >= 0 || linked.inverseOffset >= 0) {
        Mat4 m[FixedState::kProgramMatrices], inv[FixedState::kProgramMatrices];
        for (int n = 0; n < FixedState::kProgramMatrices; ++n) {
            m[n] = f.programMatrix[static_cast<size_t>(n)].back();
            if (linked.inverseOffset >= 0) inv[n] = m[n].inverse();
        }
        store(p, linked.matrixOffset, m, static_cast<size_t>(linked.matrixCount) * sizeof(Mat4));
        store(p, linked.inverseOffset, inv, static_cast<size_t>(linked.inverseCount) * sizeof(Mat4));
    }
    return &p;
}

}  // namespace wmgl::compat
