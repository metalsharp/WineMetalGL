// ARB assembly program objects and parameters (GL_ARB_vertex_program and
// GL_ARB_fragment_program, issue #25; GL_EXT_gpu_program_parameters'
// batched setters). Programs are checked when loaded (compat/arb_program.h)
// and drawn through generated GLSL programs (compat/arb_programs.h).

#include <algorithm>
#include <atomic>
#include <cstring>
#include <string>

#include "core/context.h"
#include "core/util/log.h"
#include "gl_api.gen.h"

namespace wmgl {

using compat::ArbProgramObject;
using compat::Vec4;
namespace arb = compat::arb;

namespace {

bool compatOnly(Context& ctx, const char* command) {
    if (ctx.isCompat()) return true;
    ctx.unimplemented(command);  // assembly programs are compatibility profile only
    return false;
}

// The target's index (0 vertex, 1 fragment), or -1 with INVALID_ENUM set.
int targetIndex(Context& ctx, GLenum target) {
    const int i = compat::arbTargetIndex(target);
    if (i < 0) ctx.setError(GL_INVALID_ENUM);
    return i;
}

ArbProgramObject& bound(Context& ctx, int i) { return compat::arbBound(ctx, i); }

template <class T>
void setParameters(Context& ctx, GLenum target, GLuint index, GLsizei count, const T* values, bool env) {
    const int i = targetIndex(ctx, target);
    if (i < 0) return;
    const GLuint limit = env ? arb::Limits::kEnvParameters : arb::Limits::kLocalParameters;
    if (count < 0 || index >= limit || static_cast<GLuint>(count) > limit - index) return ctx.setError(GL_INVALID_VALUE);
    Vec4* dst = env ? ctx.arb.env[i].data() : bound(ctx, i).local.data();
    for (GLsizei c = 0; c < count; ++c)
        for (int k = 0; k < 4; ++k) dst[index + static_cast<GLuint>(c)][static_cast<size_t>(k)] = static_cast<float>(values[c * 4 + k]);
}

template <class T>
void getParameter(Context& ctx, GLenum target, GLuint index, T* out, bool env) {
    const int i = targetIndex(ctx, target);
    if (i < 0) return;
    if (index >= static_cast<GLuint>(env ? arb::Limits::kEnvParameters : arb::Limits::kLocalParameters))
        return ctx.setError(GL_INVALID_VALUE);
    const Vec4& v = env ? ctx.arb.env[i][index] : bound(ctx, i).local[index];
    for (int k = 0; k < 4; ++k) out[k] = static_cast<T>(v[static_cast<size_t>(k)]);
}

}  // namespace

namespace api {

void GenProgramsARB(Context& ctx, GLsizei n, GLuint* programs) {
    if (!compatOnly(ctx, "glGenProgramsARB")) return;
    if (n < 0) return ctx.setError(GL_INVALID_VALUE);
    ShareLock lock(ctx);
    auto& names = ctx.share().arbPrograms;
    GLuint next = 1;
    for (GLsizei k = 0; k < n; ++k) {
        while (names.count(next)) ++next;
        auto object = std::make_shared<ArbProgramObject>();
        object->name = next;  // target 0: not a program object until bound
        names.emplace(next, std::move(object));
        programs[k] = next;
    }
}

void DeleteProgramsARB(Context& ctx, GLsizei n, const GLuint* programs) {
    if (!compatOnly(ctx, "glDeleteProgramsARB")) return;
    if (n < 0) return ctx.setError(GL_INVALID_VALUE);
    ShareLock lock(ctx);
    auto& names = ctx.share().arbPrograms;
    for (GLsizei k = 0; k < n; ++k) {
        if (!programs[k]) continue;
        auto it = names.find(programs[k]);
        if (it == names.end()) continue;
        // A bound program being deleted reverts the binding to the default program.
        for (std::shared_ptr<ArbProgramObject>& b : ctx.arb.bound)
            if (b == it->second) b.reset();
        names.erase(it);
    }
}

GLboolean IsProgramARB(Context& ctx, GLuint program) {
    if (!compatOnly(ctx, "glIsProgramARB")) return GL_FALSE;
    ShareLock lock(ctx);
    auto& names = ctx.share().arbPrograms;
    auto it = names.find(program);
    return it != names.end() && it->second->target ? GL_TRUE : GL_FALSE;
}

void BindProgramARB(Context& ctx, GLenum target, GLuint program) {
    if (!compatOnly(ctx, "glBindProgramARB")) return;
    const int i = targetIndex(ctx, target);
    if (i < 0) return;
    if (!program) {
        ctx.arb.bound[i].reset();
        return;
    }
    ShareLock lock(ctx);
    std::shared_ptr<ArbProgramObject>& object = ctx.share().arbPrograms[program];
    if (!object) {
        object = std::make_shared<ArbProgramObject>();
        object->name = program;
    }
    if (object->target && object->target != target) return ctx.setError(GL_INVALID_OPERATION);
    object->target = target;
    ctx.arb.bound[i] = object;
}

void ProgramStringARB(Context& ctx, GLenum target, GLenum format, GLsizei len, const void* string) {
    if (!compatOnly(ctx, "glProgramStringARB")) return;
    const int i = targetIndex(ctx, target);
    if (i < 0) return;
    if (format != GL_PROGRAM_FORMAT_ASCII_ARB) return ctx.setError(GL_INVALID_ENUM);
    if (len < 0 || (len && !string)) return ctx.setError(GL_INVALID_VALUE);
    const char* text = static_cast<const char*>(string);
    std::shared_ptr<const arb::Parsed> parsed =
        arb::parse(i == 0 ? arb::Kind::Vertex : arb::Kind::Fragment, text, static_cast<size_t>(len));
    if (!parsed->ok) {
        // The program object keeps the program that loaded before.
        ctx.arb.errorPosition = parsed->errorPosition;
        ctx.arb.errorString = parsed->error;
        WMGL_TRACE(Shader, "glProgramStringARB: %s at %d", parsed->error.c_str(), parsed->errorPosition);
        return ctx.setError(GL_INVALID_OPERATION);
    }
    static std::atomic<uint64_t> serials{0};
    ShareLock lock(ctx);
    ArbProgramObject& object = bound(ctx, i);
    object.source.assign(text, static_cast<size_t>(len));
    object.parsed = std::move(parsed);
    object.serial = ++serials;
    ctx.arb.errorPosition = -1;
    ctx.arb.errorString.clear();
}

void GetProgramStringARB(Context& ctx, GLenum target, GLenum pname, void* string) {
    if (!compatOnly(ctx, "glGetProgramStringARB")) return;
    const int i = targetIndex(ctx, target);
    if (i < 0) return;
    if (pname != GL_PROGRAM_STRING_ARB) return ctx.setError(GL_INVALID_ENUM);
    ShareLock lock(ctx);
    const std::string& s = bound(ctx, i).source;
    if (string && !s.empty()) std::memcpy(string, s.data(), s.size());
}

void GetProgramivARB(Context& ctx, GLenum target, GLenum pname, GLint* params) {
    if (!compatOnly(ctx, "glGetProgramivARB")) return;
    const int i = targetIndex(ctx, target);
    if (i < 0) return;
    ShareLock lock(ctx);
    const ArbProgramObject& object = bound(ctx, i);
    const arb::Parsed* p = object.parsed.get();
    const bool vertex = i == 0;
    using L = arb::Limits;
    auto count = [&](int arb::Parsed::*member) { return p ? p->*member : 0; };
    GLint v = 0;
    switch (pname) {
    case GL_PROGRAM_LENGTH_ARB: v = static_cast<GLint>(object.source.size()); break;
    case GL_PROGRAM_FORMAT_ARB: v = GL_PROGRAM_FORMAT_ASCII_ARB; break;
    case GL_PROGRAM_BINDING_ARB: v = static_cast<GLint>(object.name); break;
    case GL_PROGRAM_INSTRUCTIONS_ARB: case GL_PROGRAM_NATIVE_INSTRUCTIONS_ARB: v = count(&arb::Parsed::instructions); break;
    case GL_MAX_PROGRAM_INSTRUCTIONS_ARB: case GL_MAX_PROGRAM_NATIVE_INSTRUCTIONS_ARB: v = L::kInstructions; break;
    case GL_PROGRAM_TEMPORARIES_ARB: case GL_PROGRAM_NATIVE_TEMPORARIES_ARB: v = count(&arb::Parsed::temporaries); break;
    case GL_MAX_PROGRAM_TEMPORARIES_ARB: case GL_MAX_PROGRAM_NATIVE_TEMPORARIES_ARB: v = L::kTemporaries; break;
    case GL_PROGRAM_PARAMETERS_ARB: case GL_PROGRAM_NATIVE_PARAMETERS_ARB: v = count(&arb::Parsed::parameters); break;
    case GL_MAX_PROGRAM_PARAMETERS_ARB: case GL_MAX_PROGRAM_NATIVE_PARAMETERS_ARB: v = L::kParameters; break;
    case GL_PROGRAM_ATTRIBS_ARB: case GL_PROGRAM_NATIVE_ATTRIBS_ARB: v = count(&arb::Parsed::attribs); break;
    case GL_MAX_PROGRAM_ATTRIBS_ARB: case GL_MAX_PROGRAM_NATIVE_ATTRIBS_ARB:
        v = vertex ? L::kVertexAttribs : L::kFragmentAttribs;
        break;
    case GL_MAX_PROGRAM_LOCAL_PARAMETERS_ARB: v = L::kLocalParameters; break;
    case GL_MAX_PROGRAM_ENV_PARAMETERS_ARB: v = L::kEnvParameters; break;
    case GL_PROGRAM_UNDER_NATIVE_LIMITS_ARB: v = 1; break;
    case GL_PROGRAM_ADDRESS_REGISTERS_ARB: case GL_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB:
        if (!vertex) return ctx.setError(GL_INVALID_ENUM);
        v = count(&arb::Parsed::addressRegisters);
        break;
    case GL_MAX_PROGRAM_ADDRESS_REGISTERS_ARB: case GL_MAX_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB:
        if (!vertex) return ctx.setError(GL_INVALID_ENUM);
        v = L::kAddressRegisters;
        break;
    case GL_PROGRAM_ALU_INSTRUCTIONS_ARB: case GL_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB:
        if (vertex) return ctx.setError(GL_INVALID_ENUM);
        v = count(&arb::Parsed::aluInstructions);
        break;
    case GL_PROGRAM_TEX_INSTRUCTIONS_ARB: case GL_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB:
        if (vertex) return ctx.setError(GL_INVALID_ENUM);
        v = count(&arb::Parsed::texInstructions);
        break;
    case GL_PROGRAM_TEX_INDIRECTIONS_ARB: case GL_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB:
        if (vertex) return ctx.setError(GL_INVALID_ENUM);
        // Dependent reads cost nothing extra here: one indirection when the program samples at all.
        v = count(&arb::Parsed::texInstructions) ? 1 : 0;
        break;
    case GL_MAX_PROGRAM_ALU_INSTRUCTIONS_ARB: case GL_MAX_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB:
    case GL_MAX_PROGRAM_TEX_INSTRUCTIONS_ARB: case GL_MAX_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB:
    case GL_MAX_PROGRAM_TEX_INDIRECTIONS_ARB: case GL_MAX_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB:
        if (vertex) return ctx.setError(GL_INVALID_ENUM);
        v = L::kInstructions;
        break;
    default: return ctx.setError(GL_INVALID_ENUM);
    }
    *params = v;
}

void ProgramEnvParameter4fARB(Context& ctx, GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
    if (!compatOnly(ctx, "glProgramEnvParameter4fARB")) return;
    const GLfloat v[4] = {x, y, z, w};
    setParameters(ctx, target, index, 1, v, true);
}
void ProgramEnvParameter4dARB(Context& ctx, GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w) {
    if (!compatOnly(ctx, "glProgramEnvParameter4dARB")) return;
    const GLdouble v[4] = {x, y, z, w};
    setParameters(ctx, target, index, 1, v, true);
}
void ProgramEnvParameter4fvARB(Context& ctx, GLenum target, GLuint index, const GLfloat* params) {
    if (compatOnly(ctx, "glProgramEnvParameter4fvARB")) setParameters(ctx, target, index, 1, params, true);
}
void ProgramEnvParameter4dvARB(Context& ctx, GLenum target, GLuint index, const GLdouble* params) {
    if (compatOnly(ctx, "glProgramEnvParameter4dvARB")) setParameters(ctx, target, index, 1, params, true);
}
void ProgramEnvParameters4fvEXT(Context& ctx, GLenum target, GLuint index, GLsizei count, const GLfloat* params) {
    if (compatOnly(ctx, "glProgramEnvParameters4fvEXT")) setParameters(ctx, target, index, count, params, true);
}
void ProgramLocalParameter4fARB(Context& ctx, GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
    if (!compatOnly(ctx, "glProgramLocalParameter4fARB")) return;
    const GLfloat v[4] = {x, y, z, w};
    ShareLock lock(ctx);
    setParameters(ctx, target, index, 1, v, false);
}
void ProgramLocalParameter4dARB(Context& ctx, GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w) {
    if (!compatOnly(ctx, "glProgramLocalParameter4dARB")) return;
    const GLdouble v[4] = {x, y, z, w};
    ShareLock lock(ctx);
    setParameters(ctx, target, index, 1, v, false);
}
void ProgramLocalParameter4fvARB(Context& ctx, GLenum target, GLuint index, const GLfloat* params) {
    if (!compatOnly(ctx, "glProgramLocalParameter4fvARB")) return;
    ShareLock lock(ctx);
    setParameters(ctx, target, index, 1, params, false);
}
void ProgramLocalParameter4dvARB(Context& ctx, GLenum target, GLuint index, const GLdouble* params) {
    if (!compatOnly(ctx, "glProgramLocalParameter4dvARB")) return;
    ShareLock lock(ctx);
    setParameters(ctx, target, index, 1, params, false);
}
void ProgramLocalParameters4fvEXT(Context& ctx, GLenum target, GLuint index, GLsizei count, const GLfloat* params) {
    if (!compatOnly(ctx, "glProgramLocalParameters4fvEXT")) return;
    ShareLock lock(ctx);
    setParameters(ctx, target, index, count, params, false);
}

void GetProgramEnvParameterfvARB(Context& ctx, GLenum target, GLuint index, GLfloat* params) {
    if (compatOnly(ctx, "glGetProgramEnvParameterfvARB")) getParameter(ctx, target, index, params, true);
}
void GetProgramEnvParameterdvARB(Context& ctx, GLenum target, GLuint index, GLdouble* params) {
    if (compatOnly(ctx, "glGetProgramEnvParameterdvARB")) getParameter(ctx, target, index, params, true);
}
void GetProgramLocalParameterfvARB(Context& ctx, GLenum target, GLuint index, GLfloat* params) {
    if (!compatOnly(ctx, "glGetProgramLocalParameterfvARB")) return;
    ShareLock lock(ctx);
    getParameter(ctx, target, index, params, false);
}
void GetProgramLocalParameterdvARB(Context& ctx, GLenum target, GLuint index, GLdouble* params) {
    if (!compatOnly(ctx, "glGetProgramLocalParameterdvARB")) return;
    ShareLock lock(ctx);
    getParameter(ctx, target, index, params, false);
}

}  // namespace api
}  // namespace wmgl
