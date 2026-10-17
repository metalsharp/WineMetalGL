// Shader and program objects, the program interface, and uniforms
// (GL 3.3 sections 2.11-2.12, 6.1.15).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

#include "core/context.h"
#include "core/limits.h"
#include "core/program.h"
#include "core/shader/spirv_input.h"
#include "core/stats.h"
#include "core/util/log.h"
#include "gl_api.gen.h"

namespace wmgl {

Stage stageFromGL(GLenum type) {
    switch (type) {
    case GL_VERTEX_SHADER: return Stage::Vertex;
    case GL_TESS_CONTROL_SHADER: return Stage::TessControl;
    case GL_TESS_EVALUATION_SHADER: return Stage::TessEval;
    case GL_GEOMETRY_SHADER: return Stage::Geometry;
    case GL_FRAGMENT_SHADER: return Stage::Fragment;
    default: return Stage::Compute;
    }
}

GLenum stageToGL(Stage stage) {
    switch (stage) {
    case Stage::Vertex: return GL_VERTEX_SHADER;
    case Stage::TessControl: return GL_TESS_CONTROL_SHADER;
    case Stage::TessEval: return GL_TESS_EVALUATION_SHADER;
    case Stage::Geometry: return GL_GEOMETRY_SHADER;
    case Stage::Fragment: return GL_FRAGMENT_SHADER;
    default: return GL_COMPUTE_SHADER;
    }
}

namespace api {
namespace {

constexpr GLint kMaxCombinedTextureUnits = limits::kMaxCombinedTextureUnits;

void copyString(const std::string& s, GLsizei bufSize, GLsizei* length, GLchar* out) {
    if (bufSize <= 0 || !out) {
        if (length) *length = 0;
        return;
    }
    const GLsizei n = std::min<GLsizei>(bufSize - 1, static_cast<GLsizei>(s.size()));
    std::memcpy(out, s.data(), static_cast<size_t>(n));
    out[n] = 0;
    if (length) *length = n;
}

bool programBinaries(Context& ctx) { return ctx.version() >= 41 || ctx.hasExtension("GL_ARB_get_program_binary"); }
bool separateShaderObjects(Context& ctx) { return ctx.version() >= 41 || ctx.hasExtension("GL_ARB_separate_shader_objects"); }

// Shader or program lookups with GL's error rules: a name that is neither
// is INVALID_VALUE; a name of the other kind is INVALID_OPERATION.
Shader* shaderOrError(Context& ctx, GLuint name) {
    ShareLock lock(ctx);
    if (Shader* s = ctx.share().shader(name)) return s;
    ctx.setError(ctx.share().program(name) ? GL_INVALID_OPERATION : GL_INVALID_VALUE);
    return nullptr;
}

Program* programOrError(Context& ctx, GLuint name) {
    ShareLock lock(ctx);
    if (Program* p = ctx.share().program(name)) return p;
    ctx.setError(ctx.share().shader(name) ? GL_INVALID_OPERATION : GL_INVALID_VALUE);
    return nullptr;
}

Program* linkedProgramOrError(Context& ctx, GLuint name) {
    Program* p = programOrError(ctx, name);
    if (p && (!p->linkStatus || !p->linked)) {
        ctx.setError(GL_INVALID_OPERATION);
        return nullptr;
    }
    return p;
}

void maybeDeleteShader(Context& ctx, Shader* shader) {
    if (shader->deletePending && shader->attachCount == 0) {
        ShareLock lock(ctx);
        ctx.share().shaders.erase(shader->name);
    }
}

}  // namespace

void maybeDeleteProgram(Context& ctx, Program* program) {
    if (!program->deletePending || program->useCount > 0 || program->stageUses > 0) return;
    for (RefPtr<Shader>& s : program->attached) {
        --s->attachCount;
        maybeDeleteShader(ctx, s.get());
    }
    program->attached.clear();
    ShareLock lock(ctx);
    ctx.share().programs.erase(program->name);
}

namespace {

// --- uniform type information

struct TypeInfo {
    char base;      // 'f', 'i', 'u', 'b', 's' (sampler), 'd'
    int columns;
    int rows;
};

TypeInfo typeInfo(GLenum type) {
    switch (type) {
    case GL_FLOAT: return {'f', 1, 1};
    case GL_FLOAT_VEC2: return {'f', 1, 2};
    case GL_FLOAT_VEC3: return {'f', 1, 3};
    case GL_FLOAT_VEC4: return {'f', 1, 4};
    case GL_INT: return {'i', 1, 1};
    case GL_INT_VEC2: return {'i', 1, 2};
    case GL_INT_VEC3: return {'i', 1, 3};
    case GL_INT_VEC4: return {'i', 1, 4};
    case GL_UNSIGNED_INT: return {'u', 1, 1};
    case GL_UNSIGNED_INT_VEC2: return {'u', 1, 2};
    case GL_UNSIGNED_INT_VEC3: return {'u', 1, 3};
    case GL_UNSIGNED_INT_VEC4: return {'u', 1, 4};
    case GL_BOOL: return {'b', 1, 1};
    case GL_BOOL_VEC2: return {'b', 1, 2};
    case GL_BOOL_VEC3: return {'b', 1, 3};
    case GL_BOOL_VEC4: return {'b', 1, 4};
    case GL_FLOAT_MAT2: return {'f', 2, 2};
    case GL_FLOAT_MAT3: return {'f', 3, 3};
    case GL_FLOAT_MAT4: return {'f', 4, 4};
    case GL_FLOAT_MAT2x3: return {'f', 2, 3};
    case GL_FLOAT_MAT2x4: return {'f', 2, 4};
    case GL_FLOAT_MAT3x2: return {'f', 3, 2};
    case GL_FLOAT_MAT3x4: return {'f', 3, 4};
    case GL_FLOAT_MAT4x2: return {'f', 4, 2};
    case GL_FLOAT_MAT4x3: return {'f', 4, 3};
    case GL_DOUBLE: return {'d', 1, 1};
    case GL_DOUBLE_VEC2: return {'d', 1, 2};
    case GL_DOUBLE_VEC3: return {'d', 1, 3};
    case GL_DOUBLE_VEC4: return {'d', 1, 4};
    case GL_DOUBLE_MAT2: return {'d', 2, 2};
    case GL_DOUBLE_MAT3: return {'d', 3, 3};
    case GL_DOUBLE_MAT4: return {'d', 4, 4};
    case GL_DOUBLE_MAT2x3: return {'d', 2, 3};
    case GL_DOUBLE_MAT2x4: return {'d', 2, 4};
    case GL_DOUBLE_MAT3x2: return {'d', 3, 2};
    case GL_DOUBLE_MAT3x4: return {'d', 3, 4};
    case GL_DOUBLE_MAT4x2: return {'d', 4, 2};
    case GL_DOUBLE_MAT4x3: return {'d', 4, 3};
    default: return {shader::glTypeIsSampler(type) ? 's' : shader::glTypeIsImage(type) ? 'g' : 'f', 1, 1};
    }
}

// Resolves a uniform location to (uniform, element).
const shader::UniformInfo* uniformAt(Program& p, GLint location, int& element) {
    if (!p.linked || location < 0 || static_cast<size_t>(location) >= p.linked->link.locations.size()) return nullptr;
    const auto& [index, e] = p.linked->link.locations[static_cast<size_t>(location)];
    if (index < 0) return nullptr;  // a gap between explicit locations
    element = e;
    return &p.linked->link.uniforms[static_cast<size_t>(index)];
}

// Writes `count` elements of uniform data from `values` (kind f/i/u, `components` per element).
template <class T>
void setUniform(Context& ctx, Program* p, GLint location, GLsizei count, char kind, int components, const T* values) {
    if (!p) return ctx.setError(GL_INVALID_OPERATION);  // checked before location -1 is ignored
    if (location == -1) return;
    if (!p->linked) return ctx.setError(GL_INVALID_OPERATION);
    if (count < 0) return ctx.setError(GL_INVALID_VALUE);
    int element = 0;
    const shader::UniformInfo* u = uniformAt(*p, location, element);
    if (!u) return ctx.setError(GL_INVALID_OPERATION);
    const TypeInfo info = typeInfo(u->type);
    if (info.columns != 1 || info.rows != components) return ctx.setError(GL_INVALID_OPERATION);
    if (count > 1 && u->size == 1) return ctx.setError(GL_INVALID_OPERATION);
    if (info.base == 's' || info.base == 'g') {
        if (kind != 'i' || components != 1) return ctx.setError(GL_INVALID_OPERATION);
    } else if (info.base == 'b') {
        // bools accept f, i and u variants (not d)
        if (kind == 'd') return ctx.setError(GL_INVALID_OPERATION);
    } else if (info.base != kind) {
        return ctx.setError(GL_INVALID_OPERATION);
    }
    const GLsizei n = std::min<GLsizei>(count, u->size - element);
    if (info.base == 'g') {
        // Image uniforms (GL 4.2) hold image unit numbers.
        for (GLsizei i = 0; i < n; ++i)
            if (static_cast<GLint>(values[i]) < 0 || static_cast<GLint>(values[i]) >= limits::kMaxImageUnits)
                return ctx.setError(GL_INVALID_VALUE);
        for (GLsizei i = 0; i < n; ++i) p->imageUnits[static_cast<size_t>(u->imageSlot + element + i)] = static_cast<GLint>(values[i]);
        ctx.markDirty(kDirtyTextures);
        return;
    }
    if (info.base == 's') {
        for (GLsizei i = 0; i < n; ++i) {
            const GLint unit = static_cast<GLint>(values[i]);
            if (unit < 0 || unit >= kMaxCombinedTextureUnits) return ctx.setError(GL_INVALID_VALUE);
        }
        // Games set sampler units every frame to the values they have (Foundation, issue #12).
        bool changed = false;
        for (GLsizei i = 0; i < n; ++i) {
            GLint& unit = p->samplerUnits[static_cast<size_t>(u->samplerSlot + element + i)];
            changed |= unit != static_cast<GLint>(values[i]);
            unit = static_cast<GLint>(values[i]);
        }
        if (changed) ctx.markDirty(kDirtyTextures);
        return;
    }
    const size_t stride = u->arrayStride ? static_cast<size_t>(u->arrayStride) : 16;
    bool changed = false;
    const auto store = [&](uint8_t* dst, const void* value, size_t bytes) {
        if (std::memcmp(dst, value, bytes) == 0) return;
        std::memcpy(dst, value, bytes);
        changed = true;
    };
    for (GLsizei i = 0; i < n; ++i) {
        uint8_t* dst = p->uniformData.data() + u->offset + (element + i) * stride;
        for (int c = 0; c < components; ++c) {
            const T v = values[i * components + c];
            if (info.base == 'd') {
                const double d = static_cast<double>(v);
                store(dst + c * 8, &d, 8);
            } else if (info.base == 'b') {
                const uint32_t b = v != T(0) ? 1u : 0u;
                store(dst + c * 4, &b, 4);
            } else if (info.base == 'f') {
                const float f = static_cast<float>(v);
                store(dst + c * 4, &f, 4);
            } else {
                store(dst + c * 4, &v, 4);
            }
        }
    }
    // An unchanged value leaves the uniforms bound as they are.
    if (!changed) return;
    ++p->uniformVersion;
    ctx.markDirty(kDirtyUniforms);
}

template <class T>
void setUniformMatrix(Context& ctx, Program* p, GLint location, GLsizei count, GLboolean transpose, int columns, int rows,
                      const T* values) {
    constexpr bool isDouble = std::is_same_v<T, GLdouble>;
    constexpr size_t size = isDouble ? 8 : 4;
    if (!p) return ctx.setError(GL_INVALID_OPERATION);  // checked before location -1 is ignored
    if (location == -1) return;
    if (!p->linked) return ctx.setError(GL_INVALID_OPERATION);
    if (count < 0) return ctx.setError(GL_INVALID_VALUE);
    int element = 0;
    const shader::UniformInfo* u = uniformAt(*p, location, element);
    if (!u) return ctx.setError(GL_INVALID_OPERATION);
    const TypeInfo info = typeInfo(u->type);
    if (info.base != (isDouble ? 'd' : 'f') || info.columns != columns || info.rows != rows)
        return ctx.setError(GL_INVALID_OPERATION);
    if (count > 1 && u->size == 1) return ctx.setError(GL_INVALID_OPERATION);
    const GLsizei n = std::min<GLsizei>(count, u->size - element);
    const size_t matrixStride = u->matrixStride ? static_cast<size_t>(u->matrixStride) : 16;
    const size_t arrayStride = u->arrayStride ? static_cast<size_t>(u->arrayStride) : matrixStride * static_cast<size_t>(columns);
    bool changed = false;
    for (GLsizei i = 0; i < n; ++i) {
        const T* m = values + i * columns * rows;
        uint8_t* dst = p->uniformData.data() + u->offset + (element + i) * arrayStride;
        for (int c = 0; c < columns; ++c)
            for (int r = 0; r < rows; ++r) {
                const T v = transpose ? m[r * columns + c] : m[c * rows + r];
                uint8_t* at = dst + (u->rowMajor ? r * matrixStride + c * size : c * matrixStride + r * size);
                if (std::memcmp(at, &v, size) == 0) continue;
                std::memcpy(at, &v, size);
                changed = true;
            }
    }
    if (!changed) return;  // the uniforms bound stay valid
    ++p->uniformVersion;
    ctx.markDirty(kDirtyUniforms);
}

// The target of glUniform*: the current program, else the bound pipeline's
// active program (GL_ARB_separate_shader_objects).
Program* current(Context& ctx) {
    if (Program* p = ctx.state.program.get()) return p;
    ProgramPipeline* pp = ctx.state.programPipeline.get();
    return pp ? pp->activeProgram.get() : nullptr;
}

GLint locationForName(const shader::LinkResult& link, const std::string& name) {
    for (const shader::UniformInfo& u : link.uniforms) {
        if (u.blockIndex >= 0 || u.location < 0) continue;
        if (u.name == name) return u.location;
        if (u.size >= 1 && u.name.size() > 3 && u.name.compare(u.name.size() - 3, 3, "[0]") == 0) {
            const std::string base = u.name.substr(0, u.name.size() - 3);
            if (name == base) return u.location;
            if (name.size() > base.size() + 2 && name.compare(0, base.size(), base) == 0 && name[base.size()] == '[' &&
                name.back() == ']') {
                const std::string digits = name.substr(base.size() + 1, name.size() - base.size() - 2);
                if (digits.empty() || !std::all_of(digits.begin(), digits.end(), ::isdigit)) continue;
                const long index = std::strtol(digits.c_str(), nullptr, 10);
                if (index < u.size) return u.location + static_cast<GLint>(index);
                return -1;
            }
        }
    }
    return -1;
}

}  // namespace

// ------------------------------------------------------------ shaders

GLuint CreateShader(Context& ctx, GLenum type) {
    switch (type) {
    case GL_VERTEX_SHADER: case GL_FRAGMENT_SHADER: break;
    case GL_GEOMETRY_SHADER: if (ctx.version() >= 32) break; [[fallthrough]];
    case GL_TESS_CONTROL_SHADER: case GL_TESS_EVALUATION_SHADER:
        if (type != GL_GEOMETRY_SHADER && ctx.version() >= 40) break;
        [[fallthrough]];
    case GL_COMPUTE_SHADER:
        if (type == GL_COMPUTE_SHADER && ctx.computeShaders()) break;
        [[fallthrough]];
    default: ctx.setError(GL_INVALID_ENUM); return 0;
    }
    ShareLock lock(ctx);
    const GLuint name = ctx.share().createShaderProgramName();
    ctx.share().shaders[name] = makeRef<Shader>(name, type);
    return name;
}

void ShaderSource(Context& ctx, GLuint shader, GLsizei count, const GLchar* const* strings, const GLint* lengths) {
    if (count < 0) return ctx.setError(GL_INVALID_VALUE);
    Shader* s = shaderOrError(ctx, shader);
    if (!s) return;
    std::string source;
    for (GLsizei i = 0; i < count; ++i) {
        if (!strings || !strings[i]) continue;
        if (lengths && lengths[i] >= 0) source.append(strings[i], static_cast<size_t>(lengths[i]));
        else source.append(strings[i]);
    }
    s->source = std::move(source);
    // A source replaces a SPIR-V module (SPIR_V_BINARY becomes FALSE).
    s->spirv = s->specialized = false;
    s->spirvWords.clear();
    s->spirvGlsl.clear();
}

void CompileShader(Context& ctx, GLuint shader) {
    Shader* s = shaderOrError(ctx, shader);
    if (!s) return;
    if (s->spirv) return ctx.setError(GL_INVALID_OPERATION);  // SPIR-V is specialized instead
    // Compiling touches no shared state: other contexts of the share group
    // (a game's render thread while a loader thread compiles) keep running.
    const RefPtr<Shader> keep(s);
    const shader::Stage stage = s->stage;
    const std::string source = s->source;
    shader::CompileResult result;
    std::string desktop;
    {
        ShareUnlock unlocked(ctx);
        if (ctx.isEs()) {
            shader::EsCompileResult es = shader::compileEs(stage, source);
            desktop = std::move(es.desktop);
            result = std::move(es);
        } else {
            result = shader::compile(stage, source);
        }
    }
    s->desktopSource = std::move(desktop);
    s->compiled = true;
    s->compileStatus = result.ok;
    s->infoLog = result.log;
    if (!result.ok) {
        WMGL_INFO("shader %u failed to compile:\n%s", shader, result.log.c_str());
        // WINEMETALGL_DUMP_SHADERS=1 also prints the source.
        if (const char* v = std::getenv("WINEMETALGL_DUMP_SHADERS"); v && v[0] == '1')
            std::fprintf(stderr, "winemetalgl: shader %u source:\n%s\n", shader, s->source.c_str());
    }
}

void DeleteShader(Context& ctx, GLuint shader) {
    if (!shader) return;
    Shader* s = shaderOrError(ctx, shader);
    if (!s) return;
    s->deletePending = true;
    maybeDeleteShader(ctx, s);
}

GLboolean IsShader(Context& ctx, GLuint shader) {
    ShareLock lock(ctx);
    return ctx.share().shader(shader) ? GL_TRUE : GL_FALSE;
}

void GetShaderiv(Context& ctx, GLuint shader, GLenum pname, GLint* params) {
    Shader* s = shaderOrError(ctx, shader);
    if (!s || !params) return;
    switch (pname) {
    case GL_SHADER_TYPE: *params = static_cast<GLint>(s->type); return;
    case GL_DELETE_STATUS: *params = s->deletePending; return;
    case GL_COMPILE_STATUS: *params = s->compileStatus; return;
    case GL_INFO_LOG_LENGTH: *params = s->infoLog.empty() ? 0 : static_cast<GLint>(s->infoLog.size() + 1); return;
    case GL_SHADER_SOURCE_LENGTH: *params = s->source.empty() ? 0 : static_cast<GLint>(s->source.size() + 1); return;
    case GL_SPIR_V_BINARY:
        if (!ctx.glSpirv()) break;
        *params = s->spirv;
        return;
    case GL_COMPLETION_STATUS_ARB:  // compiles finish before glCompileShader returns
        if (!ctx.parallelShaderCompile()) break;
        *params = GL_TRUE;
        return;
    default: break;
    }
    ctx.setError(GL_INVALID_ENUM);
}

// GL 4.6 / GL_ARB_gl_spirv: the shader's module for its stage at
// `pEntryPoint`, with specialization constant pConstantIndex[i] set to
// pConstantValue[i], becomes its compiled form (GLSL, shader/spirv_input.h).
void SpecializeShader(Context& ctx, GLuint shader, const GLchar* pEntryPoint, GLuint numSpecializationConstants,
                      const GLuint* pConstantIndex, const GLuint* pConstantValue) {
    if (!ctx.glSpirv()) return ctx.unimplemented("glSpecializeShader");
    Shader* s = shaderOrError(ctx, shader);
    if (!s) return;
    if (!s->spirv || s->specialized) return ctx.setError(GL_INVALID_OPERATION);
    std::vector<std::pair<uint32_t, uint32_t>> constants;
    for (GLuint i = 0; i < numSpecializationConstants && pConstantIndex && pConstantValue; ++i)
        constants.emplace_back(pConstantIndex[i], pConstantValue[i]);
    const shader::SpirvSpecialization result =
        shader::specializeSpirv(s->spirvWords, s->stage, pEntryPoint ? pEntryPoint : "", constants);
    if (result.status == shader::SpirvSpecialization::kBadEntryPoint || result.status == shader::SpirvSpecialization::kBadConstant)
        return ctx.setError(GL_INVALID_VALUE);
    s->specialized = s->compiled = true;
    s->spirvNamed = result.named;
    if (result.status != shader::SpirvSpecialization::kOk) {
        s->compileStatus = false;
        s->infoLog = result.log;
    } else {
        const shader::CompileResult compiled = shader::compile(s->stage, result.glsl);
        s->compileStatus = compiled.ok;
        s->infoLog = compiled.log;
        s->spirvGlsl = result.glsl;
        s->spirvUniforms = result.uniformLocations;
    }
    if (!s->compileStatus) WMGL_INFO("SPIR-V shader %u failed to specialize:\n%s", shader, s->infoLog.c_str());
    // WINEMETALGL_DUMP_SHADERS=1 prints the GLSL a module became.
    if (const char* v = std::getenv("WINEMETALGL_DUMP_SHADERS"); v && v[0] == '1')
        std::fprintf(stderr, "winemetalgl: SPIR-V shader %u as GLSL:\n%s\n", shader, s->spirvGlsl.c_str());
}

void GetShaderInfoLog(Context& ctx, GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog) {
    if (bufSize < 0) return ctx.setError(GL_INVALID_VALUE);
    Shader* s = shaderOrError(ctx, shader);
    if (s) copyString(s->infoLog, bufSize, length, infoLog);
}

void GetShaderSource(Context& ctx, GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* source) {
    if (bufSize < 0) return ctx.setError(GL_INVALID_VALUE);
    Shader* s = shaderOrError(ctx, shader);
    if (s) copyString(s->source, bufSize, length, source);
}

void ReleaseShaderCompiler(Context& ctx) {}

void MaxShaderCompilerThreadsKHR(Context& ctx, GLuint count) {
    if (!ctx.parallelShaderCompile()) return ctx.unimplemented("glMaxShaderCompilerThreadsKHR");
    ctx.state.maxShaderCompilerThreads = count;
}

// The one shader binary format is SPIR-V (GL 4.6 / GL_ARB_gl_spirv): the
// module goes to every listed shader, to be specialized for its stage.
void ShaderBinary(Context& ctx, GLsizei count, const GLuint* shaders, GLenum binaryFormat, const void* binary,
                  GLsizei length) {
    if (count < 0 || length < 0) return ctx.setError(GL_INVALID_VALUE);
    if (!ctx.glSpirv() || binaryFormat != GL_SHADER_BINARY_FORMAT_SPIR_V) return ctx.setError(GL_INVALID_ENUM);
    std::vector<Shader*> targets;
    for (GLsizei i = 0; i < count && shaders; ++i) {
        Shader* s = shaderOrError(ctx, shaders[i]);
        if (!s) return;
        // At most one shader of each stage.
        for (const Shader* other : targets)
            if (other->type == s->type) return ctx.setError(GL_INVALID_OPERATION);
        targets.push_back(s);
    }
    if (length % 4 || (length && !binary)) return ctx.setError(GL_INVALID_VALUE);
    std::vector<uint32_t> words(static_cast<size_t>(length) / 4);
    if (length) std::memcpy(words.data(), binary, static_cast<size_t>(length));
    if (!shader::spirvHeaderValid(words)) return ctx.setError(GL_INVALID_VALUE);
    for (Shader* s : targets) {
        s->spirv = true;
        s->specialized = s->compiled = s->compileStatus = false;
        s->spirvWords = words;
        s->spirvGlsl.clear();
        s->infoLog.clear();
    }
}

void GetShaderPrecisionFormat(Context& ctx, GLenum shadertype, GLenum precisiontype, GLint* range, GLint* precision) {
    if (shadertype != GL_VERTEX_SHADER && shadertype != GL_FRAGMENT_SHADER) return ctx.setError(GL_INVALID_ENUM);
    const bool isInt = precisiontype == GL_LOW_INT || precisiontype == GL_MEDIUM_INT || precisiontype == GL_HIGH_INT;
    const bool isFloat = precisiontype == GL_LOW_FLOAT || precisiontype == GL_MEDIUM_FLOAT || precisiontype == GL_HIGH_FLOAT;
    if (!isInt && !isFloat) return ctx.setError(GL_INVALID_ENUM);
    if (range) { range[0] = isInt ? 31 : 127; range[1] = isInt ? 30 : 127; }
    if (precision) *precision = isInt ? 0 : 23;
}

// ----------------------------------------------------------- programs

GLuint CreateProgram(Context& ctx) {
    ShareLock lock(ctx);
    const GLuint name = ctx.share().createShaderProgramName();
    ctx.share().programs[name] = makeRef<Program>(name);
    return name;
}

void DeleteProgram(Context& ctx, GLuint program) {
    if (!program) return;
    Program* p = programOrError(ctx, program);
    if (!p) return;
    p->deletePending = true;
    maybeDeleteProgram(ctx, p);
}

GLboolean IsProgram(Context& ctx, GLuint program) {
    ShareLock lock(ctx);
    return ctx.share().program(program) ? GL_TRUE : GL_FALSE;
}

void AttachShader(Context& ctx, GLuint program, GLuint shader) {
    Program* p = programOrError(ctx, program);
    if (!p) return;
    Shader* s = shaderOrError(ctx, shader);
    if (!s) return;
    for (const RefPtr<Shader>& a : p->attached)
        // OpenGL ES: one shader of each stage.
        if (a.get() == s || (ctx.isEs() && a->stage == s->stage)) return ctx.setError(GL_INVALID_OPERATION);
    p->attached.push_back(RefPtr<Shader>(s));
    ++s->attachCount;
}

void DetachShader(Context& ctx, GLuint program, GLuint shader) {
    Program* p = programOrError(ctx, program);
    if (!p) return;
    Shader* s = shaderOrError(ctx, shader);
    if (!s) return;
    auto it = std::find_if(p->attached.begin(), p->attached.end(), [&](const RefPtr<Shader>& a) { return a.get() == s; });
    if (it == p->attached.end()) return ctx.setError(GL_INVALID_OPERATION);
    RefPtr<Shader> keep = *it;
    p->attached.erase(it);
    --s->attachCount;
    maybeDeleteShader(ctx, s);
}

void GetAttachedShaders(Context& ctx, GLuint program, GLsizei maxCount, GLsizei* count, GLuint* shaders) {
    if (maxCount < 0) return ctx.setError(GL_INVALID_VALUE);
    Program* p = programOrError(ctx, program);
    if (!p) return;
    GLsizei n = 0;
    for (const RefPtr<Shader>& s : p->attached) {
        if (n >= maxCount) break;
        if (shaders) shaders[n] = s->name;
        ++n;
    }
    if (count) *count = n;
}

void BindAttribLocation(Context& ctx, GLuint program, GLuint index, const GLchar* name) {
    if (index >= static_cast<GLuint>(VertexArray::kMaxAttribs)) return ctx.setError(GL_INVALID_VALUE);
    Program* p = programOrError(ctx, program);
    if (!p || !name) return;
    if (std::strncmp(name, "gl_", 3) == 0) return ctx.setError(GL_INVALID_OPERATION);
    p->attribBindings[name] = index;
}

void BindFragDataLocationIndexed(Context& ctx, GLuint program, GLuint colorNumber, GLuint index, const GLchar* name) {
    if (index > 1) return ctx.setError(GL_INVALID_VALUE);
    if (colorNumber >= (index ? 1u : static_cast<GLuint>(ContextState::kMaxDrawBuffers))) return ctx.setError(GL_INVALID_VALUE);
    Program* p = programOrError(ctx, program);
    if (!p || !name) return;
    if (std::strncmp(name, "gl_", 3) == 0) return ctx.setError(GL_INVALID_OPERATION);
    p->fragDataBindings[name] = {colorNumber, index};
}

void BindFragDataLocation(Context& ctx, GLuint program, GLuint color, const GLchar* name) {
    BindFragDataLocationIndexed(ctx, program, color, 0, name);
}

namespace {

// Links `p` from `input` (glLinkProgram, glProgramBinary). `log` and `ok`
// carry problems found while gathering the input.
void linkFrom(Context& ctx, Program* p, shader::LinkInput input, std::string log, bool ok) {
    const GLuint program = p->name;
    const RefPtr<Program> keep(p);  // the link runs without the share-group lock
    p->linkAttempted = true;
    if (ok && input.shaders.empty()) {
        log += "error: no shaders attached\n";
        ok = false;
    }
    shader::LinkResult result;
    if (ok) {
        stats::count(&stats::Frame::programLinks);
        stats::Timer timer(&stats::Frame::linkNanos);
        ShareUnlock unlocked(ctx);  // a link reads only `input`, a copy
        result = shader::link(input);
        ok = result.ok;
        log += result.log;
    }
    p->linkStatus = ok;
    p->infoLog = log;
    p->validateStatus = false;
    if (!ok) {
        WMGL_INFO("program %u failed to link:\n%s", program, log.c_str());
        // A failed relink of a program in use (current, or a program
        // pipeline's stage) keeps its old executable in use.
        if (p->useCount == 0 && p->stageUses == 0) p->linked.reset();
        return;
    }
    auto linked = std::make_shared<LinkedProgram>();
    linked->link = std::move(result);
    linked->input = std::move(input);
    linked->uid = nextLinkedProgramId();
    p->uniformData = linked->link.defaultBlockInit;  // uniform initializers, else zero
    p->uniformData.resize((linked->link.defaultBlockSize + 15) & ~15u, 0);
    p->uniformBlockBindings.assign(linked->link.blocks.size(), 0);
    p->storageBlockBindings.assign(linked->link.storageBlocks.size(), 0);
    for (size_t b = 0; b < linked->link.storageBlocks.size(); ++b)
        if (linked->link.storageBlocks[b].explicitBinding >= 0)
            p->storageBlockBindings[b] = static_cast<GLuint>(linked->link.storageBlocks[b].explicitBinding);
    p->samplerUnits.assign(linked->link.samplers.size(), 0);
    p->imageUnits.assign(linked->link.images.size(), 0);
    // layout(binding = N) (GL 4.2) gives the initial bindings and units;
    // array elements take consecutive ones.
    for (size_t b = 0; b < linked->link.blocks.size(); ++b)
        if (linked->link.blocks[b].explicitBinding >= 0) p->uniformBlockBindings[b] = static_cast<GLuint>(linked->link.blocks[b].explicitBinding);
    for (const shader::UniformInfo& u : linked->link.uniforms) {
        if (u.explicitBinding < 0) continue;
        for (GLint e = 0; e < u.size; ++e) {
            if (u.samplerSlot >= 0) p->samplerUnits[static_cast<size_t>(u.samplerSlot + e)] = u.explicitBinding + e;
            if (u.imageSlot >= 0) p->imageUnits[static_cast<size_t>(u.imageSlot + e)] = u.explicitBinding + e;
        }
    }
    p->linked = std::move(linked);
    prewarmFunctions(p->linked);
    ++p->uniformVersion;
    if (ctx.state.program.get() == p) ctx.markDirty(kDirtyProgram | kDirtyUniforms | kDirtyTextures | kDirtyVertexInput);
}

}  // namespace

void LinkProgram(Context& ctx, GLuint program) {
    Program* p = programOrError(ctx, program);
    if (!p) return;
    const TransformFeedback& xfb = *ctx.state.transformFeedback;
    if (xfb.active && xfb.program.get() == p) return ctx.setError(GL_INVALID_OPERATION);
    shader::LinkInput input;
    std::string log;
    bool ok = true;
    std::map<uint32_t, std::string> spirvUniformNames;  // location -> the first stage's name
    for (const RefPtr<Shader>& s : p->attached) {
        if (!s->compileStatus) {
            log += "error: attached shader " + std::to_string(s->name) + (s->spirv ? " is not specialized\n" : " is not compiled\n");
            ok = false;
        }
        // SPIR-V shaders link as the GLSL they were specialized into, and
        // only with SPIR-V shaders.
        if (s->spirv != p->attached.front()->spirv) {
            log += "error: SPIR-V and GLSL shaders cannot be linked together\n";
            ok = false;
        }
        input.spirv |= s->spirv;
        input.spirvUnnamed |= s->spirv && !s->spirvNamed;
        std::string source = s->spirv ? s->spirvGlsl : !s->desktopSource.empty() ? s->desktopSource : s->source;
        if (s->spirv && s->stage != shader::Stage::Compute) {
            // Inputs match the outputs of the closest earlier stage.
            int previous = -1;
            for (const RefPtr<Shader>& other : p->attached)
                if (other->stage < s->stage && other->stage != shader::Stage::Compute)
                    previous = std::max(previous, static_cast<int>(other->stage));
            if (previous >= 0) source = shader::spirvLinkInputs(source, static_cast<shader::Stage>(previous));
        }
        if (s->spirv) {
            source = shader::spirvUnifyUniforms(source, s->spirvUniforms, spirvUniformNames);
            for (const auto& [location, name] : s->spirvUniforms) spirvUniformNames.emplace(location, name);
        }
        input.shaders.push_back({s->stage, std::move(source)});
    }
    input.attribBindings = p->attribBindings;
    input.fragDataBindings = p->fragDataBindings;
    input.transformFeedbackVaryings = p->transformFeedbackVaryings;
    input.transformFeedbackMode = p->transformFeedbackMode;
    input.separable = p->separable;
    input.es = ctx.isEs();
    if (ctx.isEs() && ok && !input.spirv) {
        // OpenGL ES 3.2 section 7.3: a program needs a vertex and a fragment
        // shader (or only a compute shader), unless it is separable.
        bool vertex = false, fragment = false, compute = false, control = false, evaluation = false, geometry = false;
        std::vector<shader::ShaderSource> es;
        for (const RefPtr<Shader>& s : p->attached) {
            vertex |= s->stage == shader::Stage::Vertex;
            fragment |= s->stage == shader::Stage::Fragment;
            compute |= s->stage == shader::Stage::Compute;
            control |= s->stage == shader::Stage::TessControl;
            evaluation |= s->stage == shader::Stage::TessEval;
            geometry |= s->stage == shader::Stage::Geometry;
            es.push_back({s->stage, s->source});
        }
        if (!input.separable && !compute && !(vertex && fragment)) {
            log += "error: an OpenGL ES program needs a vertex and a fragment shader\n";
            ok = false;
        }
        // ... and its tessellation stages come together, unless it is
        // separable (section 7.3).
        if (!input.separable && control != evaluation) {
            log += "error: an OpenGL ES program needs tessellation control and evaluation shaders together\n";
            ok = false;
        }
        // Transform feedback captures the last vertex processing stage, which
        // a tessellation control shader is not (section 11.1.2.1).
        if (control && !evaluation && !geometry && !p->transformFeedbackVaryings.empty()) {
            log += "error: transform feedback cannot capture a tessellation control shader\n";
            ok = false;
        }
        const std::string errors = shader::checkEsLink(es);
        if (!errors.empty()) {
            log += errors;
            ok = false;
        }
        // The interface a program pipeline matches against its other stages.
        const shader::Stage order[] = {shader::Stage::Vertex, shader::Stage::TessControl, shader::Stage::TessEval,
                                       shader::Stage::Geometry, shader::Stage::Fragment};
        const shader::ShaderSource *first = nullptr, *last = nullptr;
        for (shader::Stage stage : order)
            for (const shader::ShaderSource& s : es)
                if (s.stage == stage) {
                    if (!first) first = &s;
                    last = &s;
                }
        if (ok && first) input.esInputInterface = shader::compileEs(first->stage, first->source).inputInterface;
        if (ok && last) input.esOutputInterface = shader::compileEs(last->stage, last->source).outputInterface;
    }
    // Compatibility profile: a program without a vertex shader uses
    // fixed-function vertex processing (transform, colours, texture
    // coordinates and fog coordinate; lighting and texgen need the
    // fixed-function program itself).
    if (ctx.isCompat() && !input.separable) {
        bool vertex = false, fragment = false;
        for (const shader::ShaderSource& s : input.shaders) {
            vertex |= s.stage == shader::Stage::Vertex;
            fragment |= s.stage == shader::Stage::Fragment;
        }
        if (fragment && !vertex)
            input.shaders.push_back({shader::Stage::Vertex, R"(#version 120
void main() {
    gl_Position = ftransform();
    gl_FrontColor = gl_Color;
    gl_FrontSecondaryColor = gl_SecondaryColor;
    gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
    gl_TexCoord[1] = gl_TextureMatrix[1] * gl_MultiTexCoord1;
    gl_TexCoord[2] = gl_TextureMatrix[2] * gl_MultiTexCoord2;
    gl_TexCoord[3] = gl_TextureMatrix[3] * gl_MultiTexCoord3;
    gl_FogFragCoord = abs((gl_ModelViewMatrix * gl_Vertex).z);
}
)"});
    }
    linkFrom(ctx, p, std::move(input), std::move(log), ok);
}

void UseProgram(Context& ctx, GLuint program) {
    const TransformFeedback& xfb = *ctx.state.transformFeedback;
    if (xfb.active && !xfb.paused) return ctx.setError(GL_INVALID_OPERATION);
    RefPtr<Program> next;
    if (program) {
        Program* p = programOrError(ctx, program);
        if (!p) return;
        if (!p->linkStatus) return ctx.setError(GL_INVALID_OPERATION);
        next = RefPtr<Program>(p);
    }
    resetSubroutineSelections(ctx);
    RefPtr<Program> previous = ctx.state.program;
    if (previous == next) return;
    if (next) ++next->useCount;
    ctx.state.program = next;
    if (previous) {
        --previous->useCount;
        maybeDeleteProgram(ctx, previous.get());
    }
    ctx.markDirty(kDirtyProgram | kDirtyUniforms | kDirtyTextures | kDirtyVertexInput);
}

void releaseCurrentProgram(Context& ctx) {
    RefPtr<Program> previous = std::move(ctx.state.program);
    if (!previous) return;
    --previous->useCount;
    maybeDeleteProgram(ctx, previous.get());
}

void ProgramParameteri(Context& ctx, GLuint program, GLenum pname, GLint value) {
    Program* p = programOrError(ctx, program);
    if (!p) return;
    switch (pname) {
    case GL_PROGRAM_BINARY_RETRIEVABLE_HINT:
        if (!programBinaries(ctx)) break;
        if (value != GL_FALSE && value != GL_TRUE) return ctx.setError(GL_INVALID_VALUE);
        p->binaryRetrievableHint = value != GL_FALSE;
        return;
    case GL_PROGRAM_SEPARABLE:
        if (!separateShaderObjects(ctx)) break;
        if (value != GL_FALSE && value != GL_TRUE) return ctx.setError(GL_INVALID_VALUE);
        p->separable = value != GL_FALSE;  // takes effect at the next link
        return;
    default: break;
    }
    ctx.setError(GL_INVALID_ENUM);
}

// GL_ARB_get_program_binary: the one binary format holds what the program
// was linked from (shader sources, attribute/fragment data bindings,
// transform feedback varyings, separability); glProgramBinary relinks it.
// Binaries are portable between WineMetalGL versions because they keep the
// GLSL, not compiled code.
namespace {

constexpr char kBinaryMagic[8] = {'W', 'M', 'G', 'L', 'P', 'B', '0', '1'};

void put32(std::string& out, uint32_t v) { out.append(reinterpret_cast<const char*>(&v), 4); }
void putString(std::string& out, const std::string& s) {
    put32(out, static_cast<uint32_t>(s.size()));
    out += s;
}

std::string serialize(const shader::LinkInput& in) {
    std::string out(kBinaryMagic, sizeof(kBinaryMagic));
    put32(out, static_cast<uint32_t>(in.shaders.size()));
    for (const shader::ShaderSource& s : in.shaders) {
        put32(out, static_cast<uint32_t>(s.stage));
        putString(out, s.source);
    }
    put32(out, static_cast<uint32_t>(in.attribBindings.size()));
    for (const auto& [name, index] : in.attribBindings) {
        putString(out, name);
        put32(out, index);
    }
    put32(out, static_cast<uint32_t>(in.fragDataBindings.size()));
    for (const auto& [name, binding] : in.fragDataBindings) {
        putString(out, name);
        put32(out, binding.first);
        put32(out, binding.second);
    }
    put32(out, static_cast<uint32_t>(in.transformFeedbackVaryings.size()));
    for (const std::string& v : in.transformFeedbackVaryings) putString(out, v);
    put32(out, in.transformFeedbackMode);
    put32(out, in.separable ? 1 : 0);
    // Since 3.0.0: SPIR-V programs (absent from older binaries).
    put32(out, (in.spirv ? 1u : 0u) | (in.spirvUnnamed ? 2u : 0u));
    return out;
}

struct Reader {
    const uint8_t* p;
    size_t left;
    bool ok = true;
    uint32_t u32() {
        uint32_t v = 0;
        if (left < 4) { ok = false; return 0; }
        std::memcpy(&v, p, 4);
        p += 4;
        left -= 4;
        return v;
    }
    std::string str() {
        const uint32_t n = u32();
        if (!ok || n > left) { ok = false; return {}; }
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        left -= n;
        return s;
    }
};

bool deserialize(const void* data, size_t size, shader::LinkInput& in) {
    if (!data || size < sizeof(kBinaryMagic) || std::memcmp(data, kBinaryMagic, sizeof(kBinaryMagic))) return false;
    Reader r{static_cast<const uint8_t*>(data) + sizeof(kBinaryMagic), size - sizeof(kBinaryMagic)};
    const uint32_t shaders = r.u32();
    for (uint32_t i = 0; r.ok && i < shaders; ++i) {
        const uint32_t stage = r.u32();
        if (stage >= static_cast<uint32_t>(shader::kStageCount)) return false;
        in.shaders.push_back({static_cast<shader::Stage>(stage), r.str()});
    }
    const uint32_t attribs = r.u32();
    // Bindings are range-checked as glBindAttribLocation/glBindFragDataLocationIndexed
    // check them: a damaged binary must not index past the linker's tables.
    for (uint32_t i = 0; r.ok && i < attribs; ++i) {
        std::string name = r.str();
        const uint32_t location = r.u32();
        if (location >= static_cast<uint32_t>(limits::kMaxVertexAttribs)) return false;
        in.attribBindings[name] = location;
    }
    const uint32_t outputs = r.u32();
    for (uint32_t i = 0; r.ok && i < outputs; ++i) {
        std::string name = r.str();
        const uint32_t location = r.u32();
        const uint32_t index = r.u32();
        if (location >= static_cast<uint32_t>(limits::kMaxDrawBuffers) || index > 1) return false;
        in.fragDataBindings[name] = {location, index};
    }
    const uint32_t varyings = r.u32();
    for (uint32_t i = 0; r.ok && i < varyings; ++i) in.transformFeedbackVaryings.push_back(r.str());
    in.transformFeedbackMode = r.u32();
    in.separable = r.u32() != 0;
    if (r.ok && r.left >= 4) {
        const uint32_t spirv = r.u32();
        in.spirv = spirv & 1u;
        in.spirvUnnamed = spirv & 2u;
    }
    return r.ok && r.left == 0;
}

}  // namespace

GLsizei programBinaryLength(const Program& p) {
    return p.linkStatus && p.linked ? static_cast<GLsizei>(serialize(p.linked->input).size()) : 0;
}

void GetProgramBinary(Context& ctx, GLuint program, GLsizei bufSize, GLsizei* length, GLenum* binaryFormat, void* binary) {
    Program* p = programOrError(ctx, program);
    if (!p) return;
    if (bufSize < 0) return ctx.setError(GL_INVALID_VALUE);
    if (!p->linkStatus || !p->linked) return ctx.setError(GL_INVALID_OPERATION);
    const std::string blob = serialize(p->linked->input);
    if (blob.size() > static_cast<size_t>(bufSize)) return ctx.setError(GL_INVALID_OPERATION);
    if (binary) std::memcpy(binary, blob.data(), blob.size());
    if (length) *length = static_cast<GLsizei>(blob.size());
    if (binaryFormat) *binaryFormat = kProgramBinaryFormat;
}

void ProgramBinary(Context& ctx, GLuint program, GLenum binaryFormat, const void* binary, GLsizei length) {
    Program* p = programOrError(ctx, program);
    if (!p) return;
    if (binaryFormat != kProgramBinaryFormat) return ctx.setError(GL_INVALID_ENUM);
    const TransformFeedback& xfb = *ctx.state.transformFeedback;
    if (xfb.active && xfb.program.get() == p) return ctx.setError(GL_INVALID_OPERATION);
    shader::LinkInput input;
    if (length < 0 || !deserialize(binary, static_cast<size_t>(length), input)) {
        // An unusable binary is not an error: the program is just not linked.
        p->linkAttempted = true;
        p->linkStatus = false;
        p->infoLog = "error: the program binary is not a WineMetalGL binary\n";
        if (p->useCount == 0 && p->stageUses == 0) p->linked.reset();
        return;
    }
    p->attribBindings = input.attribBindings;
    p->fragDataBindings = input.fragDataBindings;
    p->transformFeedbackVaryings = input.transformFeedbackVaryings;
    p->transformFeedbackMode = input.transformFeedbackMode;
    p->separable = input.separable;
    linkFrom(ctx, p, std::move(input), std::string(), true);
}

void ValidateProgram(Context& ctx, GLuint program) {
    Program* p = programOrError(ctx, program);
    if (!p) return;
    p->validateStatus = p->linkStatus && p->linked;
    if (p->validateStatus) {
        // Samplers of different types may not use the same texture unit.
        const auto& samplers = p->linked->link.samplers;
        for (size_t i = 0; i < samplers.size(); ++i)
            for (size_t j = i + 1; j < samplers.size(); ++j)
                if (p->samplerUnits[i] == p->samplerUnits[j] && samplers[i].textureTarget != samplers[j].textureTarget) {
                    p->validateStatus = false;
                    p->infoLog += "validation: samplers of different types share texture unit " +
                                  std::to_string(p->samplerUnits[i]) + "\n";
                }
    }
}

void GetProgramiv(Context& ctx, GLuint program, GLenum pname, GLint* params) {
    Program* p = programOrError(ctx, program);
    if (!p || !params) return;
    const shader::LinkResult* link = p->linked && p->linkStatus ? &p->linked->link : nullptr;
    auto maxLength = [&](const auto& list, size_t count = SIZE_MAX) {
        GLint n = 0;
        for (size_t i = 0; i < std::min(count, list.size()); ++i)
            n = std::max<GLint>(n, static_cast<GLint>(shader::spirvDisplayName(*link, list[i].name).size() + 1));
        return n;
    };
    switch (pname) {
    case GL_DELETE_STATUS: *params = p->deletePending; return;
    case GL_LINK_STATUS: *params = p->linkStatus; return;
    case GL_COMPLETION_STATUS_ARB:  // links finish before glLinkProgram returns
        if (!ctx.parallelShaderCompile()) break;
        *params = GL_TRUE;
        return;
    case GL_VALIDATE_STATUS: *params = p->validateStatus; return;
    case GL_INFO_LOG_LENGTH: *params = p->infoLog.empty() ? 0 : static_cast<GLint>(p->infoLog.size() + 1); return;
    case GL_ATTACHED_SHADERS: *params = static_cast<GLint>(p->attached.size()); return;
    case GL_ACTIVE_ATTRIBUTES: *params = link ? static_cast<GLint>(link->attribs.size()) : 0; return;
    case GL_ACTIVE_ATTRIBUTE_MAX_LENGTH: *params = link ? maxLength(link->attribs) : 0; return;  // wmgl_X is as long as gl_X + 2
    case GL_ACTIVE_UNIFORMS: *params = link ? static_cast<GLint>(link->visibleUniformCount()) : 0; return;
    case GL_ACTIVE_UNIFORM_MAX_LENGTH: *params = link ? maxLength(link->uniforms, link->visibleUniformCount()) : 0; return;
    case GL_ACTIVE_UNIFORM_BLOCKS: *params = link ? static_cast<GLint>(link->visibleBlockCount()) : 0; return;
    case GL_ACTIVE_ATOMIC_COUNTER_BUFFERS:
        if (!ctx.atomicCounters()) break;
        *params = link ? static_cast<GLint>(link->atomicBuffers.size()) : 0;
        return;
    case GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH: *params = link ? maxLength(link->blocks, link->visibleBlockCount()) : 0; return;
    case GL_TRANSFORM_FEEDBACK_BUFFER_MODE: *params = static_cast<GLint>(p->transformFeedbackMode); return;
    case GL_TRANSFORM_FEEDBACK_VARYINGS: *params = link ? static_cast<GLint>(link->transformFeedback.size()) : 0; return;
    case GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH: *params = link ? maxLength(link->transformFeedback) : 0; return;
    case GL_GEOMETRY_VERTICES_OUT: case GL_GEOMETRY_INPUT_TYPE: case GL_GEOMETRY_OUTPUT_TYPE:
        if (!link || !link->stages[static_cast<int>(shader::Stage::Geometry)].present) return ctx.setError(GL_INVALID_OPERATION);
        *params = pname == GL_GEOMETRY_VERTICES_OUT ? link->geometryVerticesOut
                  : pname == GL_GEOMETRY_INPUT_TYPE ? static_cast<GLint>(link->geometryInputType)
                                                    : static_cast<GLint>(link->geometryOutputType);
        return;
    case GL_TESS_CONTROL_OUTPUT_VERTICES:
        if (!ctx.tessellationShaders()) break;
        if (!link || !link->stages[static_cast<int>(shader::Stage::TessControl)].present) return ctx.setError(GL_INVALID_OPERATION);
        *params = link->tess.verticesOut;
        return;
    case GL_TESS_GEN_MODE: case GL_TESS_GEN_SPACING: case GL_TESS_GEN_VERTEX_ORDER: case GL_TESS_GEN_POINT_MODE:
        if (!ctx.tessellationShaders()) break;
        if (!link || !link->stages[static_cast<int>(shader::Stage::TessEval)].present) return ctx.setError(GL_INVALID_OPERATION);
        *params = pname == GL_TESS_GEN_MODE      ? static_cast<GLint>(link->tess.mode)
                  : pname == GL_TESS_GEN_SPACING ? static_cast<GLint>(link->tess.spacing)
                  : pname == GL_TESS_GEN_VERTEX_ORDER ? static_cast<GLint>(link->tess.order)
                                                      : (link->tess.pointMode ? GL_TRUE : GL_FALSE);
        return;
    case GL_GEOMETRY_SHADER_INVOCATIONS:
        if (ctx.version() < 40 && !ctx.hasExtension("GL_ARB_gpu_shader5")) break;
        if (!link || !link->stages[static_cast<int>(shader::Stage::Geometry)].present) return ctx.setError(GL_INVALID_OPERATION);
        *params = link->geometryInvocations;
        return;
    case GL_COMPUTE_WORK_GROUP_SIZE:
        if (!ctx.computeShaders()) break;
        if (!link || !link->stages[static_cast<int>(shader::Stage::Compute)].present) return ctx.setError(GL_INVALID_OPERATION);
        for (int d = 0; d < 3; ++d) params[d] = static_cast<GLint>(link->computeLocalSize[d]);
        return;
    case GL_PROGRAM_BINARY_RETRIEVABLE_HINT:
        if (programBinaries(ctx)) { *params = p->binaryRetrievableHint; return; }
        break;
    case GL_PROGRAM_BINARY_LENGTH:
        if (programBinaries(ctx)) { *params = programBinaryLength(*p); return; }
        break;
    case GL_PROGRAM_SEPARABLE:
        // A program whose last link failed (or that never linked) reports the
        // initial FALSE, as the Khronos CTS expects of glCreateShaderProgramv
        // with a shader that does not compile (Mesa behaves the same).
        if (separateShaderObjects(ctx)) { *params = p->linkStatus ? p->separable : GL_FALSE; return; }
        break;
    default: break;
    }
    ctx.setError(GL_INVALID_ENUM);
}

void GetProgramInfoLog(Context& ctx, GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog) {
    if (bufSize < 0) return ctx.setError(GL_INVALID_VALUE);
    Program* p = programOrError(ctx, program);
    if (p) copyString(p->infoLog, bufSize, length, infoLog);
}

GLint GetAttribLocation(Context& ctx, GLuint program, const GLchar* name) {
    Program* p = linkedProgramOrError(ctx, program);
    if (!p || !name || std::strncmp(name, "gl_", 3) == 0) return -1;
    const std::string wanted = name;
    for (const shader::AttribInfo& a : p->linked->link.attribs) {
        if (a.conventional) continue;
        if (a.name == wanted) return a.location;
        // Elements of an attribute array: "a[i]" (each element takes the type's locations).
        const std::string base = a.name.size() > 3 && a.name.compare(a.name.size() - 3, 3, "[0]") == 0
                                     ? a.name.substr(0, a.name.size() - 3)
                                     : a.name;
        if ((a.array || a.size > 1) && wanted.size() > base.size() + 2 && wanted.compare(0, base.size(), base) == 0 &&
            wanted[base.size()] == '[' && wanted.back() == ']') {
            const std::string digits = wanted.substr(base.size() + 1, wanted.size() - base.size() - 2);
            if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) continue;
            const long index = std::strtol(digits.c_str(), nullptr, 10);
            if (index < a.size) return a.location + static_cast<GLint>(index) * shader::glTypeLocations(a.type);
        }
    }
    return -1;
}

void GetActiveAttrib(Context& ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLint* size,
                     GLenum* type, GLchar* name) {
    if (bufSize < 0) return ctx.setError(GL_INVALID_VALUE);
    Program* p = programOrError(ctx, program);
    if (!p) return;
    if (!p->linked || !p->linkStatus || index >= p->linked->link.attribs.size()) return ctx.setError(GL_INVALID_VALUE);
    const shader::AttribInfo& a = p->linked->link.attribs[index];
    // Lowered legacy attributes report their built-in name (wmgl_Vertex -> gl_Vertex).
    copyString(shader::spirvDisplayName(p->linked->link, a.conventional ? "gl_" + a.name.substr(5) : a.name), bufSize, length, name);
    if (size) *size = a.size;
    if (type) *type = a.type;
}

GLint GetFragDataLocation(Context& ctx, GLuint program, const GLchar* name) {
    Program* p = linkedProgramOrError(ctx, program);
    if (!p || !name) return -1;
    for (const shader::FragOutputInfo& f : p->linked->link.fragOutputs)
        if (f.name == name) return f.location;
    return -1;
}

GLint GetFragDataIndex(Context& ctx, GLuint program, const GLchar* name) {
    Program* p = linkedProgramOrError(ctx, program);
    if (!p || !name) return -1;
    for (const shader::FragOutputInfo& f : p->linked->link.fragOutputs)
        if (f.name == name) return f.index;
    return -1;
}

GLint GetUniformLocation(Context& ctx, GLuint program, const GLchar* name) {
    Program* p = linkedProgramOrError(ctx, program);
    if (!p || !name || std::strncmp(name, "gl_", 3) == 0) return -1;
    return locationForName(p->linked->link, name);
}

void GetActiveUniform(Context& ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLint* size,
                      GLenum* type, GLchar* name) {
    if (bufSize < 0) return ctx.setError(GL_INVALID_VALUE);
    Program* p = programOrError(ctx, program);
    if (!p) return;
    if (!p->linked || !p->linkStatus || index >= p->linked->link.visibleUniformCount()) return ctx.setError(GL_INVALID_VALUE);
    const shader::UniformInfo& u = p->linked->link.uniforms[index];
    copyString(shader::spirvDisplayName(p->linked->link, u.name), bufSize, length, name);
    if (size) *size = u.size;
    if (type) *type = u.type;
}

void GetActiveUniformName(Context& ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLchar* name) {
    GetActiveUniform(ctx, program, index, bufSize, length, nullptr, nullptr, name);
}

void GetActiveUniformsiv(Context& ctx, GLuint program, GLsizei count, const GLuint* indices, GLenum pname, GLint* params) {
    if (count < 0) return ctx.setError(GL_INVALID_VALUE);
    Program* p = programOrError(ctx, program);
    if (!p || !indices || !params) return;
    const auto& uniforms = p->linked && p->linkStatus ? p->linked->link.uniforms : std::vector<shader::UniformInfo>();
    const size_t visible = p->linked && p->linkStatus ? p->linked->link.visibleUniformCount() : 0;
    for (GLsizei i = 0; i < count; ++i)
        if (indices[i] >= visible) return ctx.setError(GL_INVALID_VALUE);
    for (GLsizei i = 0; i < count; ++i) {
        const shader::UniformInfo& u = uniforms[indices[i]];
        switch (pname) {
        case GL_UNIFORM_TYPE: params[i] = static_cast<GLint>(u.type); break;
        case GL_UNIFORM_SIZE: params[i] = u.size; break;
        case GL_UNIFORM_NAME_LENGTH: params[i] = static_cast<GLint>(shader::spirvDisplayName(p->linked->link, u.name).size() + 1); break;
        case GL_UNIFORM_BLOCK_INDEX: params[i] = u.blockIndex; break;
        // Block members and atomic counters (GL 4.2) have offsets and strides.
        case GL_UNIFORM_OFFSET: params[i] = u.blockIndex >= 0 || u.atomicBufferIndex >= 0 ? u.offset : -1; break;
        case GL_UNIFORM_ARRAY_STRIDE: {
            // Arrays (named "x[0]") report their stride even with one element.
            const bool array = !u.name.empty() && u.name.back() == ']';
            params[i] = u.blockIndex >= 0 ? (array ? u.arrayStride : 0) : u.atomicBufferIndex >= 0 ? (array ? 4 : 0) : -1;
            break;
        }
        case GL_UNIFORM_ATOMIC_COUNTER_BUFFER_INDEX:
            if (!ctx.atomicCounters()) return ctx.setError(GL_INVALID_ENUM);
            params[i] = u.atomicBufferIndex;
            break;
        case GL_UNIFORM_MATRIX_STRIDE: params[i] = u.blockIndex >= 0 ? u.matrixStride : u.atomicBufferIndex >= 0 ? 0 : -1; break;
        case GL_UNIFORM_IS_ROW_MAJOR: params[i] = u.blockIndex >= 0 && u.rowMajor; break;
        default: return ctx.setError(GL_INVALID_ENUM);
        }
    }
}

// GL 4.2 / GL_ARB_shader_atomic_counters.
void GetActiveAtomicCounterBufferiv(Context& ctx, GLuint program, GLuint bufferIndex, GLenum pname, GLint* params) {
    if (!ctx.atomicCounters()) return ctx.unimplemented("glGetActiveAtomicCounterBufferiv");
    Program* p = programOrError(ctx, program);
    if (!p) return;
    const shader::LinkResult* link = p->linked && p->linkStatus ? &p->linked->link : nullptr;
    if (!link || bufferIndex >= link->atomicBuffers.size()) return ctx.setError(GL_INVALID_VALUE);
    const shader::AtomicCounterBufferInfo& b = link->atomicBuffers[bufferIndex];
    auto referenced = [&](shader::Stage stage) { return (b.stageMask >> static_cast<int>(stage)) & 1 ? 1 : 0; };
    switch (pname) {
    case GL_ATOMIC_COUNTER_BUFFER_BINDING: *params = static_cast<GLint>(b.binding); return;
    case GL_ATOMIC_COUNTER_BUFFER_DATA_SIZE: *params = b.dataSize; return;
    case GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTERS: *params = static_cast<GLint>(b.activeUniforms.size()); return;
    case GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTER_INDICES:
        for (size_t i = 0; i < b.activeUniforms.size(); ++i) params[i] = b.activeUniforms[i];
        return;
    case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_VERTEX_SHADER: *params = referenced(shader::Stage::Vertex); return;
    case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_CONTROL_SHADER: *params = referenced(shader::Stage::TessControl); return;
    case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_EVALUATION_SHADER: *params = referenced(shader::Stage::TessEval); return;
    case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_GEOMETRY_SHADER: *params = referenced(shader::Stage::Geometry); return;
    case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_FRAGMENT_SHADER: *params = referenced(shader::Stage::Fragment); return;
    case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_COMPUTE_SHADER:
        if (!ctx.computeShaders()) break;
        *params = referenced(shader::Stage::Compute);
        return;
    default: break;
    }
    ctx.setError(GL_INVALID_ENUM);
}

void GetUniformIndices(Context& ctx, GLuint program, GLsizei count, const GLchar* const* names, GLuint* indices) {
    if (count < 0) return ctx.setError(GL_INVALID_VALUE);
    Program* p = programOrError(ctx, program);
    if (!p || !names || !indices) return;
    const auto* uniforms = p->linked && p->linkStatus ? &p->linked->link.uniforms : nullptr;
    for (GLsizei i = 0; i < count; ++i) {
        indices[i] = GL_INVALID_INDEX;
        if (!uniforms || !names[i]) continue;
        const std::string name = names[i];
        for (size_t u = 0; u < p->linked->link.visibleUniformCount(); ++u) {
            const std::string& n = (*uniforms)[u].name;
            if (n == name || (n.size() > 3 && n.compare(n.size() - 3, 3, "[0]") == 0 && n.substr(0, n.size() - 3) == name)) {
                indices[i] = static_cast<GLuint>(u);
                break;
            }
        }
    }
}

GLuint GetUniformBlockIndex(Context& ctx, GLuint program, const GLchar* name) {
    Program* p = programOrError(ctx, program);
    if (!p || !name || !p->linked || !p->linkStatus) return GL_INVALID_INDEX;
    const auto& blocks = p->linked->link.blocks;
    for (size_t i = 0; i < p->linked->link.visibleBlockCount(); ++i)
        if (blocks[i].name == name) return static_cast<GLuint>(i);
    return GL_INVALID_INDEX;
}

void GetActiveUniformBlockiv(Context& ctx, GLuint program, GLuint blockIndex, GLenum pname, GLint* params) {
    Program* p = programOrError(ctx, program);
    if (!p || !params) return;
    if (!p->linked || !p->linkStatus || blockIndex >= p->linked->link.visibleBlockCount()) return ctx.setError(GL_INVALID_VALUE);
    const shader::UniformBlockInfo& b = p->linked->link.blocks[blockIndex];
    const auto stageBit = [](shader::Stage s) { return 1u << static_cast<int>(s); };
    switch (pname) {
    case GL_UNIFORM_BLOCK_BINDING: *params = static_cast<GLint>(p->uniformBlockBindings[blockIndex]); return;
    case GL_UNIFORM_BLOCK_DATA_SIZE: *params = b.dataSize; return;
    case GL_UNIFORM_BLOCK_NAME_LENGTH: *params = static_cast<GLint>(shader::spirvDisplayName(p->linked->link, b.name).size() + 1); return;
    case GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS: *params = static_cast<GLint>(b.activeUniforms.size()); return;
    case GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES:
        for (size_t i = 0; i < b.activeUniforms.size(); ++i) params[i] = b.activeUniforms[i];
        return;
    case GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER: *params = (b.stageMask & stageBit(shader::Stage::Vertex)) != 0; return;
    case GL_UNIFORM_BLOCK_REFERENCED_BY_GEOMETRY_SHADER: *params = (b.stageMask & stageBit(shader::Stage::Geometry)) != 0; return;
    case GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER: *params = (b.stageMask & stageBit(shader::Stage::Fragment)) != 0; return;
    case GL_UNIFORM_BLOCK_REFERENCED_BY_COMPUTE_SHADER:
        if (!ctx.computeShaders()) break;
        *params = (b.stageMask & stageBit(shader::Stage::Compute)) != 0;
        return;
    case GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_CONTROL_SHADER: case GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_EVALUATION_SHADER:
        if (!ctx.tessellationShaders()) break;
        *params = (b.stageMask & stageBit(pname == GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_CONTROL_SHADER ? shader::Stage::TessControl
                                                                                                      : shader::Stage::TessEval)) != 0;
        return;
    default: break;
    }
    ctx.setError(GL_INVALID_ENUM);
}

void GetActiveUniformBlockName(Context& ctx, GLuint program, GLuint blockIndex, GLsizei bufSize, GLsizei* length,
                               GLchar* name) {
    if (bufSize < 0) return ctx.setError(GL_INVALID_VALUE);
    Program* p = programOrError(ctx, program);
    if (!p) return;
    if (!p->linked || !p->linkStatus || blockIndex >= p->linked->link.visibleBlockCount()) return ctx.setError(GL_INVALID_VALUE);
    copyString(shader::spirvDisplayName(p->linked->link, p->linked->link.blocks[blockIndex].name), bufSize, length, name);
}

void UniformBlockBinding(Context& ctx, GLuint program, GLuint blockIndex, GLuint binding) {
    Program* p = programOrError(ctx, program);
    if (!p) return;
    if (!p->linked || !p->linkStatus || blockIndex >= p->linked->link.visibleBlockCount()) return ctx.setError(GL_INVALID_VALUE);
    if (binding >= static_cast<GLuint>(ContextState::kMaxUniformBufferBindings)) return ctx.setError(GL_INVALID_VALUE);
    p->uniformBlockBindings[blockIndex] = binding;
    ctx.markDirty(kDirtyUniformBuffers);
}

void ShaderStorageBlockBinding(Context& ctx, GLuint program, GLuint blockIndex, GLuint binding) {
    if (!ctx.storageBuffers()) return ctx.unimplemented("glShaderStorageBlockBinding");
    Program* p = programOrError(ctx, program);
    if (!p) return;
    if (!p->linked || !p->linkStatus || blockIndex >= p->linked->link.storageBlocks.size()) return ctx.setError(GL_INVALID_VALUE);
    if (binding >= static_cast<GLuint>(ContextState::kMaxShaderStorageBufferBindings)) return ctx.setError(GL_INVALID_VALUE);
    p->storageBlockBindings[blockIndex] = binding;
}

// ------------------------------------------------------------ uniforms

#define WMGL_UNIFORM_V(Name, T, kind, n)                                                         \
    void Uniform##Name(Context& ctx, GLint location, GLsizei count, const T* value) {           \
        setUniform(ctx, current(ctx), location, count, kind, n, value);                         \
    }                                                                                           \
    void ProgramUniform##Name(Context& ctx, GLuint program, GLint location, GLsizei count, const T* value) { \
        setUniform(ctx, linkedProgramOrError(ctx, program), location, count, kind, n, value);   \
    }
WMGL_UNIFORM_V(1fv, GLfloat, 'f', 1)
WMGL_UNIFORM_V(2fv, GLfloat, 'f', 2)
WMGL_UNIFORM_V(3fv, GLfloat, 'f', 3)
WMGL_UNIFORM_V(4fv, GLfloat, 'f', 4)
WMGL_UNIFORM_V(1iv, GLint, 'i', 1)
WMGL_UNIFORM_V(2iv, GLint, 'i', 2)
WMGL_UNIFORM_V(3iv, GLint, 'i', 3)
WMGL_UNIFORM_V(4iv, GLint, 'i', 4)
WMGL_UNIFORM_V(1uiv, GLuint, 'u', 1)
WMGL_UNIFORM_V(2uiv, GLuint, 'u', 2)
WMGL_UNIFORM_V(3uiv, GLuint, 'u', 3)
WMGL_UNIFORM_V(4uiv, GLuint, 'u', 4)

void Uniform1f(Context& ctx, GLint l, GLfloat x) { Uniform1fv(ctx, l, 1, &x); }
void Uniform2f(Context& ctx, GLint l, GLfloat x, GLfloat y) { const GLfloat v[] = {x, y}; Uniform2fv(ctx, l, 1, v); }
void Uniform3f(Context& ctx, GLint l, GLfloat x, GLfloat y, GLfloat z) { const GLfloat v[] = {x, y, z}; Uniform3fv(ctx, l, 1, v); }
void Uniform4f(Context& ctx, GLint l, GLfloat x, GLfloat y, GLfloat z, GLfloat w) { const GLfloat v[] = {x, y, z, w}; Uniform4fv(ctx, l, 1, v); }
void Uniform1i(Context& ctx, GLint l, GLint x) { Uniform1iv(ctx, l, 1, &x); }
void Uniform2i(Context& ctx, GLint l, GLint x, GLint y) { const GLint v[] = {x, y}; Uniform2iv(ctx, l, 1, v); }
void Uniform3i(Context& ctx, GLint l, GLint x, GLint y, GLint z) { const GLint v[] = {x, y, z}; Uniform3iv(ctx, l, 1, v); }
void Uniform4i(Context& ctx, GLint l, GLint x, GLint y, GLint z, GLint w) { const GLint v[] = {x, y, z, w}; Uniform4iv(ctx, l, 1, v); }
void Uniform1ui(Context& ctx, GLint l, GLuint x) { Uniform1uiv(ctx, l, 1, &x); }
void Uniform2ui(Context& ctx, GLint l, GLuint x, GLuint y) { const GLuint v[] = {x, y}; Uniform2uiv(ctx, l, 1, v); }
void Uniform3ui(Context& ctx, GLint l, GLuint x, GLuint y, GLuint z) { const GLuint v[] = {x, y, z}; Uniform3uiv(ctx, l, 1, v); }
void Uniform4ui(Context& ctx, GLint l, GLuint x, GLuint y, GLuint z, GLuint w) { const GLuint v[] = {x, y, z, w}; Uniform4uiv(ctx, l, 1, v); }

// GL_ARB_gpu_shader_fp64: double uniforms (and their program forms).
WMGL_UNIFORM_V(1dv, GLdouble, 'd', 1)
WMGL_UNIFORM_V(2dv, GLdouble, 'd', 2)
WMGL_UNIFORM_V(3dv, GLdouble, 'd', 3)
WMGL_UNIFORM_V(4dv, GLdouble, 'd', 4)
#undef WMGL_UNIFORM_V
void Uniform1d(Context& ctx, GLint l, GLdouble x) { Uniform1dv(ctx, l, 1, &x); }
void Uniform2d(Context& ctx, GLint l, GLdouble x, GLdouble y) { const GLdouble v[] = {x, y}; Uniform2dv(ctx, l, 1, v); }
void Uniform3d(Context& ctx, GLint l, GLdouble x, GLdouble y, GLdouble z) { const GLdouble v[] = {x, y, z}; Uniform3dv(ctx, l, 1, v); }
void Uniform4d(Context& ctx, GLint l, GLdouble x, GLdouble y, GLdouble z, GLdouble w) {
    const GLdouble v[] = {x, y, z, w};
    Uniform4dv(ctx, l, 1, v);
}
void ProgramUniform1d(Context& ctx, GLuint p, GLint l, GLdouble x) { ProgramUniform1dv(ctx, p, l, 1, &x); }
void ProgramUniform2d(Context& ctx, GLuint p, GLint l, GLdouble x, GLdouble y) { const GLdouble v[] = {x, y}; ProgramUniform2dv(ctx, p, l, 1, v); }
void ProgramUniform3d(Context& ctx, GLuint p, GLint l, GLdouble x, GLdouble y, GLdouble z) {
    const GLdouble v[] = {x, y, z};
    ProgramUniform3dv(ctx, p, l, 1, v);
}
void ProgramUniform4d(Context& ctx, GLuint p, GLint l, GLdouble x, GLdouble y, GLdouble z, GLdouble w) {
    const GLdouble v[] = {x, y, z, w};
    ProgramUniform4dv(ctx, p, l, 1, v);
}

// GL_ARB_separate_shader_objects scalar forms.
void ProgramUniform1f(Context& ctx, GLuint p, GLint l, GLfloat x) { ProgramUniform1fv(ctx, p, l, 1, &x); }
void ProgramUniform2f(Context& ctx, GLuint p, GLint l, GLfloat x, GLfloat y) { const GLfloat v[] = {x, y}; ProgramUniform2fv(ctx, p, l, 1, v); }
void ProgramUniform3f(Context& ctx, GLuint p, GLint l, GLfloat x, GLfloat y, GLfloat z) { const GLfloat v[] = {x, y, z}; ProgramUniform3fv(ctx, p, l, 1, v); }
void ProgramUniform4f(Context& ctx, GLuint p, GLint l, GLfloat x, GLfloat y, GLfloat z, GLfloat w) { const GLfloat v[] = {x, y, z, w}; ProgramUniform4fv(ctx, p, l, 1, v); }
void ProgramUniform1i(Context& ctx, GLuint p, GLint l, GLint x) { ProgramUniform1iv(ctx, p, l, 1, &x); }
void ProgramUniform2i(Context& ctx, GLuint p, GLint l, GLint x, GLint y) { const GLint v[] = {x, y}; ProgramUniform2iv(ctx, p, l, 1, v); }
void ProgramUniform3i(Context& ctx, GLuint p, GLint l, GLint x, GLint y, GLint z) { const GLint v[] = {x, y, z}; ProgramUniform3iv(ctx, p, l, 1, v); }
void ProgramUniform4i(Context& ctx, GLuint p, GLint l, GLint x, GLint y, GLint z, GLint w) { const GLint v[] = {x, y, z, w}; ProgramUniform4iv(ctx, p, l, 1, v); }
void ProgramUniform1ui(Context& ctx, GLuint p, GLint l, GLuint x) { ProgramUniform1uiv(ctx, p, l, 1, &x); }
void ProgramUniform2ui(Context& ctx, GLuint p, GLint l, GLuint x, GLuint y) { const GLuint v[] = {x, y}; ProgramUniform2uiv(ctx, p, l, 1, v); }
void ProgramUniform3ui(Context& ctx, GLuint p, GLint l, GLuint x, GLuint y, GLuint z) { const GLuint v[] = {x, y, z}; ProgramUniform3uiv(ctx, p, l, 1, v); }
void ProgramUniform4ui(Context& ctx, GLuint p, GLint l, GLuint x, GLuint y, GLuint z, GLuint w) { const GLuint v[] = {x, y, z, w}; ProgramUniform4uiv(ctx, p, l, 1, v); }

#define WMGL_UNIFORM_MATRIX(Name, c, r)                                                                       \
    void UniformMatrix##Name##fv(Context& ctx, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value) { \
        setUniformMatrix(ctx, current(ctx), location, count, transpose, c, r, value);                          \
    }                                                                                                          \
    void ProgramUniformMatrix##Name##fv(Context& ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, \
                                        const GLfloat* value) {                                                \
        setUniformMatrix(ctx, linkedProgramOrError(ctx, program), location, count, transpose, c, r, value);    \
    }                                                                                                          \
    void UniformMatrix##Name##dv(Context& ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value) { \
        setUniformMatrix(ctx, current(ctx), location, count, transpose, c, r, value);                          \
    }                                                                                                          \
    void ProgramUniformMatrix##Name##dv(Context& ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, \
                                        const GLdouble* value) {                                               \
        setUniformMatrix(ctx, linkedProgramOrError(ctx, program), location, count, transpose, c, r, value);    \
    }
WMGL_UNIFORM_MATRIX(2, 2, 2)
WMGL_UNIFORM_MATRIX(3, 3, 3)
WMGL_UNIFORM_MATRIX(4, 4, 4)
WMGL_UNIFORM_MATRIX(2x3, 2, 3)
WMGL_UNIFORM_MATRIX(2x4, 2, 4)
WMGL_UNIFORM_MATRIX(3x2, 3, 2)
WMGL_UNIFORM_MATRIX(3x4, 3, 4)
WMGL_UNIFORM_MATRIX(4x2, 4, 2)
WMGL_UNIFORM_MATRIX(4x3, 4, 3)
#undef WMGL_UNIFORM_MATRIX

namespace {

template <class T>
void getUniform(Context& ctx, GLuint program, GLint location, T* params) {
    Program* p = linkedProgramOrError(ctx, program);
    if (!p || !params) return;
    int element = 0;
    const shader::UniformInfo* u = uniformAt(*p, location, element);
    if (!u) return ctx.setError(GL_INVALID_OPERATION);
    const TypeInfo info = typeInfo(u->type);
    if (info.base == 's') {
        *params = static_cast<T>(p->samplerUnits[static_cast<size_t>(u->samplerSlot + element)]);
        return;
    }
    if (info.base == 'g') {
        *params = static_cast<T>(p->imageUnits[static_cast<size_t>(u->imageSlot + element)]);
        return;
    }
    const size_t matrixStride = u->matrixStride ? static_cast<size_t>(u->matrixStride) : 16;
    const size_t stride = u->arrayStride ? static_cast<size_t>(u->arrayStride)
                                         : (info.columns > 1 ? matrixStride * static_cast<size_t>(info.columns) : 16);
    const uint8_t* src = p->uniformData.data() + u->offset + element * stride;
    const size_t size = info.base == 'd' ? 8 : 4;
    int out = 0;
    for (int c = 0; c < info.columns; ++c)
        for (int r = 0; r < info.rows; ++r) {
            const uint8_t* at = info.columns > 1 ? (u->rowMajor ? src + r * matrixStride + c * size : src + c * matrixStride + r * size)
                                                 : src + r * size;
            // Floating-point values read as integers are rounded to the nearest integer (GL 4.6 section 2.2.2).
            if (info.base == 'd' || info.base == 'f') {
                double v;
                if (info.base == 'd') {
                    std::memcpy(&v, at, 8);
                } else {
                    float f;
                    std::memcpy(&f, at, 4);
                    v = f;
                }
                if constexpr (std::is_unsigned_v<T>) params[out++] = v > 0.0 ? static_cast<T>(std::floor(v + 0.5)) : T(0);
                else if constexpr (std::is_integral_v<T>) params[out++] = static_cast<T>(std::floor(v + 0.5));
                else params[out++] = static_cast<T>(v);
            }
            else if (info.base == 'i') { int32_t v; std::memcpy(&v, at, 4); params[out++] = static_cast<T>(v); }
            else { uint32_t v; std::memcpy(&v, at, 4); params[out++] = static_cast<T>(v); }
        }
}

}  // namespace

int uniformComponents(Context& ctx, GLuint program, GLint location) {
    Program* p = ctx.share().program(program);
    if (!p || !p->linked || !p->linkStatus || location < 0) return -1;
    int element = 0;
    const shader::UniformInfo* u = uniformAt(*p, location, element);
    if (!u) return -1;
    const TypeInfo info = typeInfo(u->type);
    return info.base == 's' ? 1 : info.columns * info.rows;
}

void GetUniformfv(Context& ctx, GLuint program, GLint location, GLfloat* params) { getUniform(ctx, program, location, params); }
void GetUniformiv(Context& ctx, GLuint program, GLint location, GLint* params) { getUniform(ctx, program, location, params); }
void GetUniformuiv(Context& ctx, GLuint program, GLint location, GLuint* params) { getUniform(ctx, program, location, params); }
void GetUniformdv(Context& ctx, GLuint program, GLint location, GLdouble* params) { getUniform(ctx, program, location, params); }

// ------------------------------------------------- transform feedback setup

void TransformFeedbackVaryings(Context& ctx, GLuint program, GLsizei count, const GLchar* const* varyings, GLenum bufferMode) {
    if (count < 0) return ctx.setError(GL_INVALID_VALUE);
    if (bufferMode != GL_INTERLEAVED_ATTRIBS && bufferMode != GL_SEPARATE_ATTRIBS) return ctx.setError(GL_INVALID_ENUM);
    if (bufferMode == GL_SEPARATE_ATTRIBS && count > TransformFeedback::kMaxBuffers) return ctx.setError(GL_INVALID_VALUE);
    Program* p = programOrError(ctx, program);
    if (!p) return;
    // GL_ARB_transform_feedback3: gl_NextBuffer and gl_SkipComponents1-4 only
    // interleave, and gl_NextBuffer cannot go past the last buffer.
    if (ctx.version() >= 40 || ctx.hasExtension("GL_ARB_transform_feedback3")) {
        int nextBuffers = 0;
        for (GLsizei i = 0; i < count && varyings; ++i) {
            const char* name = varyings[i] ? varyings[i] : "";
            const bool next = std::strcmp(name, "gl_NextBuffer") == 0;
            const bool skip = std::strncmp(name, "gl_SkipComponents", 17) == 0 && name[17] >= '1' && name[17] <= '4' && !name[18];
            if ((next || skip) && bufferMode != GL_INTERLEAVED_ATTRIBS) return ctx.setError(GL_INVALID_OPERATION);
            nextBuffers += next ? 1 : 0;
        }
        if (nextBuffers >= TransformFeedback::kMaxBuffers) return ctx.setError(GL_INVALID_OPERATION);
    }
    p->transformFeedbackVaryings.clear();
    for (GLsizei i = 0; i < count; ++i) p->transformFeedbackVaryings.emplace_back(varyings && varyings[i] ? varyings[i] : "");
    p->transformFeedbackMode = bufferMode;
}

void GetTransformFeedbackVarying(Context& ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLsizei* size,
                                 GLenum* type, GLchar* name) {
    Program* p = programOrError(ctx, program);
    if (!p) return;
    if (!p->linked || !p->linkStatus || index >= p->linked->link.transformFeedback.size()) return ctx.setError(GL_INVALID_VALUE);
    const auto& v = p->linked->link.transformFeedback[index];
    copyString(shader::spirvDisplayName(p->linked->link, v.name), bufSize, length, name);
    if (size) *size = v.size;
    if (type) *type = v.type;
}

// ------------------------------------------------------------ GL_ARB_shader_objects

// Handles are the shader and program names (one name space, GL 2.0 rules);
// the commands without a core alias map onto the core queries.
void DeleteObjectARB(Context& ctx, GLhandleARB obj) {
    if (!obj) return;
    if (ctx.share().program(obj)) return DeleteProgram(ctx, obj);
    if (ctx.share().shader(obj)) return DeleteShader(ctx, obj);
    ctx.setError(GL_INVALID_VALUE);
}

GLhandleARB GetHandleARB(Context& ctx, GLenum pname) {
    if (pname != GL_PROGRAM_OBJECT_ARB) {
        ctx.setError(GL_INVALID_ENUM);
        return 0;
    }
    return ctx.state.program ? ctx.state.program->name : 0;
}

void GetObjectParameterivARB(Context& ctx, GLhandleARB obj, GLenum pname, GLint* params) {
    const bool program = ctx.share().program(obj) != nullptr;
    Shader* shader = program ? nullptr : ctx.share().shader(obj);
    if (!program && !shader) return ctx.setError(GL_INVALID_VALUE);
    if (pname == GL_OBJECT_TYPE_ARB) {
        *params = program ? GL_PROGRAM_OBJECT_ARB : GL_SHADER_OBJECT_ARB;
        return;
    }
    if (pname == GL_OBJECT_SUBTYPE_ARB) {
        if (program) return ctx.setError(GL_INVALID_OPERATION);
        GetShaderiv(ctx, obj, GL_SHADER_TYPE, params);
        return;
    }
    if (program) GetProgramiv(ctx, obj, pname, params);
    else GetShaderiv(ctx, obj, pname, params);
}

void GetObjectParameterfvARB(Context& ctx, GLhandleARB obj, GLenum pname, GLfloat* params) {
    GLint value[3] = {};
    GetObjectParameterivARB(ctx, obj, pname, value);
    params[0] = static_cast<GLfloat>(value[0]);
}

void GetInfoLogARB(Context& ctx, GLhandleARB obj, GLsizei maxLength, GLsizei* length, GLcharARB* infoLog) {
    if (ctx.share().program(obj)) return GetProgramInfoLog(ctx, obj, maxLength, length, infoLog);
    if (ctx.share().shader(obj)) return GetShaderInfoLog(ctx, obj, maxLength, length, infoLog);
    ctx.setError(GL_INVALID_VALUE);
}

void GetAttachedObjectsARB(Context& ctx, GLhandleARB containerObj, GLsizei maxCount, GLsizei* count, GLhandleARB* obj) {
    GetAttachedShaders(ctx, containerObj, maxCount, count, obj);
}

}  // namespace api
}  // namespace wmgl
