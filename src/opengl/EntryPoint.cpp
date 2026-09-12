/// @file EntryPoint.cpp
/// @brief opengl32.dll shim exports for MetalSharp GL→Metal bridge.
///
/// Exports all commonly-used OpenGL 1.0-2.1 entry points. The shim is thin:
/// each function delegates through OpenGLBridge::getGLProcAddress() to the
/// macOS native OpenGL framework. Phase 4b focuses on correctness over
/// completeness — games that stick to GL 1.x/2.x immediate mode and fixed
/// pipeline will run unmodified.
///
/// Extension functions (GL 3.x/4.x, e.g. glCreateShader, glGenBuffers,
/// glBindBuffer) are resolved at runtime by Wine/wglGetProcAddress from
/// the framework. Full GL 3.x/4.x shader translation via SPIRV-Cross →
/// MSL is scaffolded for Phase 4c; the OpenGLBridge::GLState tracker
/// already captures program/blend/depth/attrib/viewport so future
/// instrumentation can be added without breaking this shim.

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <metalsharp/GLErrorTracker.h>
#include <metalsharp/GLMetalRenderer.h>
#include <metalsharp/GLShaderCache.h>
#include <metalsharp/GLShaderTracker.h>
#include <metalsharp/GLSLCompiler.h>
#include <metalsharp/GLSLVersion.h>
#include <metalsharp/OpenGLBridge.h>
#include <metalsharp/ShaderStage.h>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if __has_include(<spirv_cross_c.h>)
#define METALSHARP_HAS_SPIRV_CROSS 1
#else
#define METALSHARP_HAS_SPIRV_CROSS 0
#endif

namespace {

metalsharp::OpenGLBridge g_glBridge;
metalsharp::GLMetalRenderer g_metalRenderer;
std::once_flag g_glInitFlag;
std::once_flag g_metalInitFlag;
bool g_metalAvailable = false;

struct ExperimentalProgram {
    bool linked = false;
    bool linkSuccess = false;
    std::string infoLog;
    std::unordered_map<std::string, int32_t> uniformLocations;
    std::unordered_map<std::string, int32_t> attributeLocations;
    std::unordered_map<int32_t, std::vector<uint8_t>> uniformValues;
};

std::mutex g_programMutex;
std::unordered_map<uint32_t, ExperimentalProgram> g_programs;
std::mutex g_syncMutex;
std::unordered_set<void*> g_syncs;

struct ExperimentalBuffer {
    uint64_t metalHandle = 0;
    size_t size = 0;
};
std::mutex g_bufferMutex;
std::unordered_map<uint32_t, ExperimentalBuffer> g_buffers;

struct ExperimentalTexture {
    uint64_t metalHandle = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t minFilter = 0x2601; /* GL_LINEAR */
    uint32_t magFilter = 0x2601;
    uint32_t wrapS = 0x2901; /* GL_REPEAT */
    uint32_t wrapT = 0x2901;
    std::vector<uint8_t> pixels;
};
struct ExperimentalFramebuffer {
    uint32_t colorTexture = 0;
};
std::mutex g_resourceMutex;
std::unordered_map<uint32_t, ExperimentalTexture> g_textures;
std::unordered_map<uint32_t, ExperimentalFramebuffer> g_framebuffers;
std::array<uint32_t, metalsharp::kMaxTextureUnits> g_textureUnits{};
std::array<uint32_t, 16> g_imageUnits{};
std::array<uint32_t, metalsharp::kMaxVertexAttribs> g_attributeDivisors{};
std::unordered_map<uint32_t, bool> g_vertexArrays;
struct ExperimentalSampler { uint32_t minFilter = 0x2601, magFilter = 0x2601, wrapS = 0x2901, wrapT = 0x2901; };
std::unordered_map<uint32_t, ExperimentalSampler> g_samplers;
std::array<uint32_t, metalsharp::kMaxTextureUnits> g_samplerUnits{};
uint32_t g_boundStorageBuffer = 0;
uint32_t g_boundIndirectBuffer = 0;
uint32_t g_currentVertexArray = 0;
uint32_t g_activeTextureUnit = 0;

bool metalModeEnabled() {
    const char* value = std::getenv("WINEMETALGL_EXPERIMENTAL");
    return value && !std::strcmp(value, "1");
}

void ensureGLInit() {
    std::call_once(g_glInitFlag, [] { g_glBridge.init(); });
}

bool ensureMetalInit() {
    std::call_once(g_metalInitFlag, [] { g_metalAvailable = g_metalRenderer.init(); });
    return g_metalAvailable;
}

bool isExperimentalProgram(uint32_t program) {
    return metalsharp::GLShaderTracker::instance().hasProgram(program);
}

void discoverShaderInterface(ExperimentalProgram& program, const std::string& source, bool vertexStage) {
    std::istringstream lines(source);
    std::string line;
    while (std::getline(lines, line)) {
        const size_t comment = line.find("//");
        if (comment != std::string::npos) line.resize(comment);
        const size_t uniform = line.find("uniform ");
        if (uniform != std::string::npos) {
            std::istringstream declaration(line.substr(uniform + 8));
            std::string type, name;
            if (declaration >> type >> name) {
                const size_t terminator = name.find_first_of(";=,");
                if (terminator != std::string::npos) name.resize(terminator);
                const size_t array = name.find('[');
                if (array != std::string::npos) name.resize(array);
                if (!name.empty() && !program.uniformLocations.count(name))
                    program.uniformLocations[name] = static_cast<int32_t>(program.uniformLocations.size());
            }
        }
        if (!vertexStage) continue;
        const size_t in = line.find(" in ");
        if (in == std::string::npos && line.rfind("in ", 0) != 0) continue;
        std::string declaration = line.substr(in == std::string::npos ? 3 : in + 4);
        std::istringstream input(declaration);
        std::string type, name;
        if (!(input >> type >> name)) continue;
        const size_t terminator = name.find_first_of(";[=");
        if (terminator != std::string::npos) name.resize(terminator);
        if (name.empty() || program.attributeLocations.count(name)) continue;
        int32_t location = static_cast<int32_t>(program.attributeLocations.size());
        const size_t layout = line.find("location");
        if (layout != std::string::npos) {
            const size_t equals = line.find('=', layout);
            if (equals != std::string::npos) location = std::strtol(line.c_str() + equals + 1, nullptr, 10);
        }
        program.attributeLocations[name] = location;
    }
}

bool beginExperimentalCompute(uint32_t program) {
    if (!ensureMetalInit()) return false;
    std::lock_guard<std::mutex> lock(g_programMutex);
    auto it = g_programs.find(program);
    if (it == g_programs.end() || !it->second.linkSuccess) return false;
    metalsharp::GLShaderState* compute = nullptr;
    for (uint32_t shader : metalsharp::GLShaderTracker::instance().copyAttachedShaders(program)) {
        auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
        if (state && state->stage == metalsharp::ShaderStage::Compute) compute = state;
    }
    if (!compute || !g_metalRenderer.createComputePipeline(*compute)) return false;
    g_metalRenderer.beginComputePass();
    std::lock_guard<std::mutex> resourceLock(g_bufferMutex);
    auto buffer = g_buffers.find(g_boundStorageBuffer);
    if (buffer != g_buffers.end()) g_metalRenderer.bindComputeBuffer(buffer->second.metalHandle, 0);
    std::lock_guard<std::mutex> textureLock(g_resourceMutex);
    for (uint32_t unit = 0; unit < g_imageUnits.size(); ++unit) {
        auto image = g_textures.find(g_imageUnits[unit]);
        if (image != g_textures.end()) g_metalRenderer.bindComputeTexture(image->second.metalHandle, unit);
    }
    return true;
}

bool beginExperimentalDraw(uint32_t program) {
    if (!ensureMetalInit()) return false;
    std::lock_guard<std::mutex> lock(g_programMutex);
    auto programIt = g_programs.find(program);
    if (programIt == g_programs.end() || !programIt->second.linkSuccess) return false;

    metalsharp::GLShaderState* vertex = nullptr;
    metalsharp::GLShaderState* fragment = nullptr;
    for (uint32_t shader : metalsharp::GLShaderTracker::instance().copyAttachedShaders(program)) {
        auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
        if (!state) continue;
        if (state->stage == metalsharp::ShaderStage::Vertex) vertex = state;
        else if (state->stage == metalsharp::ShaderStage::Pixel) fragment = state;
    }
    if (!vertex || !fragment || !g_metalRenderer.createPipeline(*vertex, *fragment, g_glBridge.state())) return false;

    const uint32_t width = g_glBridge.state().viewportWidth > 0
                               ? static_cast<uint32_t>(g_glBridge.state().viewportWidth) : 64;
    const uint32_t height = g_glBridge.state().viewportHeight > 0
                                ? static_cast<uint32_t>(g_glBridge.state().viewportHeight) : 64;
    uint64_t colorTexture = 0;
    uint32_t passWidth = width, passHeight = height;
    {
        std::lock_guard<std::mutex> resourceLock(g_resourceMutex);
        if (g_glBridge.state().boundFramebuffer) {
            auto fbo = g_framebuffers.find(g_glBridge.state().boundFramebuffer);
            if (fbo == g_framebuffers.end() || !fbo->second.colorTexture) return false;
            auto texture = g_textures.find(fbo->second.colorTexture);
            if (texture == g_textures.end()) return false;
            colorTexture = texture->second.metalHandle;
            passWidth = texture->second.width;
            passHeight = texture->second.height;
        }
    }
    g_metalRenderer.setClearColor(g_glBridge.state().clearColor[0], g_glBridge.state().clearColor[1],
                                   g_glBridge.state().clearColor[2], g_glBridge.state().clearColor[3]);
    g_metalRenderer.setClearDepth(g_glBridge.state().clearDepth);
    g_metalRenderer.beginRenderPassToTexture(colorTexture, passWidth, passHeight, true);
    g_metalRenderer.setViewport(0, 0, passWidth, passHeight);
    for (uint32_t unit = 0; unit < g_textureUnits.size(); ++unit) {
        std::lock_guard<std::mutex> resourceLock(g_resourceMutex);
        auto texture = g_textures.find(g_textureUnits[unit]);
        if (texture == g_textures.end()) continue;
        g_metalRenderer.bindTexture(texture->second.metalHandle, unit);
        auto sampler = g_samplers.find(g_samplerUnits[unit]);
        if (sampler != g_samplers.end())
            g_metalRenderer.bindSampler(unit, sampler->second.minFilter, sampler->second.magFilter, sampler->second.wrapS, sampler->second.wrapT);
        else
            g_metalRenderer.bindSampler(unit, texture->second.minFilter, texture->second.magFilter,
                                        texture->second.wrapS, texture->second.wrapT);
    }
    g_metalRenderer.usePipeline();

    /* OpenGL uniform locations are opaque integers. The first Metal ABI
     * reserves buffer(0) for a tightly packed 16-byte slot per queried
     * location; this provides deterministic scalar/vector/matrix updates and
     * keeps the canonical values independent of the legacy GL context. */
    std::vector<uint8_t> uniformData;
    for (const auto& entry : programIt->second.uniformValues) {
        const size_t offset = static_cast<size_t>(entry.first) * 16;
        if (offset + entry.second.size() > uniformData.size()) uniformData.resize(offset + entry.second.size());
        std::memcpy(uniformData.data() + offset, entry.second.data(), entry.second.size());
    }
    if (!uniformData.empty()) g_metalRenderer.updateUniformBuffer(0, uniformData.data(), uniformData.size());
    return true;
}

// Variadic dispatch: forwards to a native GL function resolved by name and
// returns a default-constructed value if the framework is not loaded.
// This is the dispatch core used by every GL_PASSTHROUGH macro.
template <typename Ret, typename... Args> Ret glDispatch(const char* name, Args... args) {
    ensureGLInit();
    auto fn = reinterpret_cast<Ret (*)(Args...)>(g_glBridge.getGLProcAddress(name));
    if (fn) {
        return fn(static_cast<Args>(args)...);
    }
    return Ret();
}

// Per-arity macros. We need separate macros for each arity because variadic
// macros can't "spread" parenthesised argument lists inside another function
// call (e.g. `f((a, b))` is the comma operator, not two arguments). The 0-arg
// variant takes no signature/args and is the simplest case.

#define GL_PASSTHROUGH0(ret, name)                                                                                     \
    extern "C" ret name() {                                                                                            \
        return glDispatch<ret>(#name);                                                                                 \
    }

#define GL_PASSTHROUGH1(ret, name, T1, a1)                                                                             \
    extern "C" ret name(T1 a1) {                                                                                       \
        return glDispatch<ret, T1>(#name, (a1));                                                                       \
    }

#define GL_PASSTHROUGH2(ret, name, T1, a1, T2, a2)                                                                     \
    extern "C" ret name(T1 a1, T2 a2) {                                                                                \
        return glDispatch<ret, T1, T2>(#name, (a1), (a2));                                                             \
    }

#define GL_PASSTHROUGH3(ret, name, T1, a1, T2, a2, T3, a3)                                                             \
    extern "C" ret name(T1 a1, T2 a2, T3 a3) {                                                                         \
        return glDispatch<ret, T1, T2, T3>(#name, (a1), (a2), (a3));                                                   \
    }

#define GL_PASSTHROUGH4(ret, name, T1, a1, T2, a2, T3, a3, T4, a4)                                                     \
    extern "C" ret name(T1 a1, T2 a2, T3 a3, T4 a4) {                                                                  \
        return glDispatch<ret, T1, T2, T3, T4>(#name, (a1), (a2), (a3), (a4));                                         \
    }

#define GL_PASSTHROUGH5(ret, name, T1, a1, T2, a2, T3, a3, T4, a4, T5, a5)                                             \
    extern "C" ret name(T1 a1, T2 a2, T3 a3, T4 a4, T5 a5) {                                                           \
        return glDispatch<ret, T1, T2, T3, T4, T5>(#name, (a1), (a2), (a3), (a4), (a5));                               \
    }

#define GL_PASSTHROUGH6(ret, name, T1, a1, T2, a2, T3, a3, T4, a4, T5, a5, T6, a6)                                     \
    extern "C" ret name(T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6) {                                                    \
        return glDispatch<ret, T1, T2, T3, T4, T5, T6>(#name, (a1), (a2), (a3), (a4), (a5), (a6));                     \
    }

#define GL_PASSTHROUGH7(ret, name, T1, a1, T2, a2, T3, a3, T4, a4, T5, a5, T6, a6, T7, a7)                             \
    extern "C" ret name(T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6, T7 a7) {                                             \
        return glDispatch<ret, T1, T2, T3, T4, T5, T6, T7>(#name, (a1), (a2), (a3), (a4), (a5), (a6), (a7));           \
    }

#define GL_PASSTHROUGH9(ret, name, T1, a1, T2, a2, T3, a3, T4, a4, T5, a5, T6, a6, T7, a7, T8, a8, T9, a9)             \
    extern "C" ret name(T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6, T7 a7, T8 a8, T9 a9) {                               \
        return glDispatch<ret, T1, T2, T3, T4, T5, T6, T7, T8, T9>(#name, (a1), (a2), (a3), (a4), (a5), (a6), (a7),    \
                                                                   (a8), (a9));                                        \
    }

} // namespace

// ---------------------------------------------------------------------------
// Vertex / immediate-mode pipeline (GL 1.0/1.1)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH1(void, glBegin, uint32_t, mode)
GL_PASSTHROUGH0(void, glEnd)

// ---------------------------------------------------------------------------
// Buffers / state
// ---------------------------------------------------------------------------
extern "C" void glClear(uint32_t mask) {
    if (metalModeEnabled() && !g_glBridge.state().currentProgram && ensureMetalInit()) {
        const uint32_t width = g_glBridge.state().viewportWidth > 0 ?
            static_cast<uint32_t>(g_glBridge.state().viewportWidth) : 64;
        const uint32_t height = g_glBridge.state().viewportHeight > 0 ?
            static_cast<uint32_t>(g_glBridge.state().viewportHeight) : 64;
        g_metalRenderer.setClearColor(g_glBridge.state().clearColor[0], g_glBridge.state().clearColor[1],
                                      g_glBridge.state().clearColor[2], g_glBridge.state().clearColor[3]);
        g_metalRenderer.setClearDepth(g_glBridge.state().clearDepth);
        g_metalRenderer.beginRenderPassToTexture(0, width, height, true);
        g_metalRenderer.endRenderPass();
        g_metalRenderer.finish();
        return;
    }
    glDispatch<void, uint32_t>("glClear", mask);
}
extern "C" void glClearColor(float r, float g, float b, float a) {
    glDispatch<void, float, float, float, float>("glClearColor", r, g, b, a);
    g_glBridge.state().clearColor[0] = r;
    g_glBridge.state().clearColor[1] = g;
    g_glBridge.state().clearColor[2] = b;
    g_glBridge.state().clearColor[3] = a;
}
extern "C" void glViewport(int32_t x, int32_t y, int32_t w, int32_t h) {
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(int32_t, int32_t, int32_t, int32_t)>(
        g_glBridge.getGLProcAddress("glViewport"));
    if (fn) {
        fn(x, y, w, h);
    }
    g_glBridge.state().viewportX = x;
    g_glBridge.state().viewportY = y;
    g_glBridge.state().viewportWidth = w;
    g_glBridge.state().viewportHeight = h;
}
extern "C" void glEnable(uint32_t cap) {
    glDispatch<void, uint32_t>("glEnable", cap);
    if (cap == 0x0BE2) g_glBridge.state().blendEnabled = true;
    if (cap == 0x0B71) g_glBridge.state().depthTestEnabled = true;
}
extern "C" void glDisable(uint32_t cap) {
    glDispatch<void, uint32_t>("glDisable", cap);
    if (cap == 0x0BE2) g_glBridge.state().blendEnabled = false;
    if (cap == 0x0B71) g_glBridge.state().depthTestEnabled = false;
}
extern "C" void glBlendFunc(uint32_t sfactor, uint32_t dfactor) {
    glDispatch<void, uint32_t, uint32_t>("glBlendFunc", sfactor, dfactor);
    g_glBridge.state().blendSrcRGB = sfactor;
    g_glBridge.state().blendDstRGB = dfactor;
}
extern "C" void glDepthFunc(uint32_t func) {
    glDispatch<void, uint32_t>("glDepthFunc", func);
    g_glBridge.state().depthFunc = func;
}

// ---------------------------------------------------------------------------
// Buffer objects (GL 1.5)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH2(void, glGenBuffers, int32_t, n, uint32_t*, buffers)

extern "C" void glDeleteBuffers(int32_t n, const uint32_t* buffers) {
    if (buffers) {
        std::lock_guard<std::mutex> lock(g_bufferMutex);
        for (int32_t i = 0; i < n; ++i) g_buffers.erase(buffers[i]);
    }
    glDispatch<void, int32_t, const uint32_t*>("glDeleteBuffers", n, buffers);
}

extern "C" void glBufferData(uint32_t target, int64_t size, const void* data, uint32_t usage) {
    constexpr uint32_t kGL_ARRAY_BUFFER = 0x8892;
    constexpr uint32_t kGL_ELEMENT_ARRAY_BUFFER = 0x8893;
    constexpr uint32_t kGL_SHADER_STORAGE_BUFFER = 0x90D2;
    constexpr uint32_t kGL_DRAW_INDIRECT_BUFFER = 0x8F3F;
    const bool experimental = std::getenv("WINEMETALGL_EXPERIMENTAL") &&
                              std::strcmp(std::getenv("WINEMETALGL_EXPERIMENTAL"), "1") == 0;
    const uint32_t name = target == kGL_ARRAY_BUFFER ? g_glBridge.state().boundArrayBuffer :
                          target == kGL_ELEMENT_ARRAY_BUFFER ? g_glBridge.state().boundElementArrayBuffer :
                          target == kGL_SHADER_STORAGE_BUFFER ? g_boundStorageBuffer :
                          target == kGL_DRAW_INDIRECT_BUFFER ? g_boundIndirectBuffer : 0;
    if (experimental && name && size > 0 && ensureMetalInit()) {
        uint64_t handle = g_metalRenderer.createBuffer(data, static_cast<size_t>(size));
        if (handle) {
            std::lock_guard<std::mutex> lock(g_bufferMutex);
            g_buffers[name] = {handle, static_cast<size_t>(size)};
            return;
        }
        metalsharp::GLErrorTracker::instance().setError(0x0505);
        return;
    }
    glDispatch<void, uint32_t, int64_t, const void*, uint32_t>("glBufferData", target, size, data, usage);
}

extern "C" void glBindBufferBase(uint32_t target, uint32_t index, uint32_t buffer) {
    glDispatch<void, uint32_t, uint32_t, uint32_t>("glBindBufferBase", target, index, buffer);
    if (target == 0x90D2 && index == 0) g_boundStorageBuffer = buffer;
}

extern "C" void glGetBufferSubData(uint32_t target, int64_t offset, int64_t size, void* data) {
    if (metalModeEnabled() && target == 0x90D2 && g_boundStorageBuffer && offset >= 0 && size >= 0) {
        uint64_t handle = 0;
        { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it = g_buffers.find(g_boundStorageBuffer); if (it != g_buffers.end()) handle = it->second.metalHandle; }
        if (handle && g_metalRenderer.readBuffer(handle, static_cast<size_t>(offset), static_cast<size_t>(size), data)) return;
    }
    glDispatch<void, uint32_t, int64_t, int64_t, void*>("glGetBufferSubData", target, offset, size, data);
}

GL_PASSTHROUGH4(void, glBufferSubData, uint32_t, target, int64_t, offset, int64_t, size, const void*, data)
GL_PASSTHROUGH2(void*, glMapBuffer, uint32_t, target, uint32_t, access)
GL_PASSTHROUGH1(unsigned char, glUnmapBuffer, uint32_t, target)
GL_PASSTHROUGH1(unsigned char, glIsBuffer, uint32_t, buffer)

// glBindBuffer is hand-written because it must mirror the binding into
// GLState so subsequent draw calls / VAO setup can observe which buffer
// is currently bound. The native call is still issued so the framework
// context state stays in sync.
extern "C" void glBindBuffer(uint32_t target, uint32_t buffer) {
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t, uint32_t)>(g_glBridge.getGLProcAddress("glBindBuffer"));
    if (fn) {
        fn(target, buffer);
    }
    // Mirror into the state tracker for GL_ARRAY_BUFFER / GL_ELEMENT_ARRAY_BUFFER.
    // Other targets (e.g. GL_PIXEL_PACK_BUFFER, GL_UNIFORM_BUFFER) are ignored
    // here; the bridge only tracks the two targets consumed by draw submission.
    constexpr uint32_t kGL_ARRAY_BUFFER = 0x8892;         // GL_ARRAY_BUFFER
    constexpr uint32_t kGL_ELEMENT_ARRAY_BUFFER = 0x8893; // GL_ELEMENT_ARRAY_BUFFER
    constexpr uint32_t kGL_SHADER_STORAGE_BUFFER = 0x90D2;
    constexpr uint32_t kGL_DRAW_INDIRECT_BUFFER = 0x8F3F;
    if (target == kGL_ARRAY_BUFFER) {
        g_glBridge.state().boundArrayBuffer = buffer;
    } else if (target == kGL_ELEMENT_ARRAY_BUFFER) {
        g_glBridge.state().boundElementArrayBuffer = buffer;
    } else if (target == kGL_SHADER_STORAGE_BUFFER) {
        g_boundStorageBuffer = buffer;
    } else if (target == kGL_DRAW_INDIRECT_BUFFER) {
        g_boundIndirectBuffer = buffer;
    }
}

// ---------------------------------------------------------------------------
// Draw submission
// ---------------------------------------------------------------------------
extern "C" void glDrawArrays(uint32_t mode, int32_t first, int32_t count) {
    const uint32_t program = g_glBridge.state().currentProgram;
    if (isExperimentalProgram(program)) {
        if (!beginExperimentalDraw(program)) {
            metalsharp::GLErrorTracker::instance().setError(0x0502); // GL_INVALID_OPERATION
            return;
        }
        g_metalRenderer.drawArrays(mode, static_cast<uint32_t>(first), static_cast<uint32_t>(count));
        g_metalRenderer.endRenderPass();
        g_metalRenderer.finish();
        return;
    }
    glDispatch<void, uint32_t, int32_t, int32_t>("glDrawArrays", mode, first, count);
}

extern "C" void glDrawElements(uint32_t mode, int32_t count, uint32_t type, const void* indices) {
    const uint32_t program = g_glBridge.state().currentProgram;
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t, int32_t, uint32_t, const void*>("glDrawElements", mode, count, type, indices);
        return;
    }
    uint64_t indexBuffer = 0;
    {
        std::lock_guard<std::mutex> lock(g_bufferMutex);
        auto it = g_buffers.find(g_glBridge.state().boundElementArrayBuffer);
        if (it != g_buffers.end()) indexBuffer = it->second.metalHandle;
    }
    if (!indexBuffer || !beginExperimentalDraw(program)) {
        metalsharp::GLErrorTracker::instance().setError(0x0502);
        return;
    }
    g_metalRenderer.bindIndexBuffer(indexBuffer, 0);
    g_metalRenderer.drawElements(mode, static_cast<uint32_t>(count), type,
                                 reinterpret_cast<size_t>(indices));
    g_metalRenderer.endRenderPass();
    g_metalRenderer.finish();
}

extern "C" void glDrawArraysInstanced(uint32_t mode, int32_t first, int32_t count, int32_t instances) {
    const uint32_t program = g_glBridge.state().currentProgram;
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t, int32_t, int32_t, int32_t>("glDrawArraysInstanced", mode, first, count, instances);
        return;
    }
    if (instances <= 0 || !beginExperimentalDraw(program)) {
        metalsharp::GLErrorTracker::instance().setError(0x0502); return;
    }
    g_metalRenderer.drawArraysInstanced(mode, static_cast<uint32_t>(first), static_cast<uint32_t>(count),
                                        static_cast<uint32_t>(instances));
    g_metalRenderer.endRenderPass();
    g_metalRenderer.finish();
}

extern "C" void glDrawElementsInstanced(uint32_t mode, int32_t count, uint32_t type, const void* indices,
                                         int32_t instances) {
    const uint32_t program = g_glBridge.state().currentProgram;
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t, int32_t, uint32_t, const void*, int32_t>(
            "glDrawElementsInstanced", mode, count, type, indices, instances);
        return;
    }
    uint64_t indexBuffer = 0;
    {
        std::lock_guard<std::mutex> lock(g_bufferMutex);
        auto it = g_buffers.find(g_glBridge.state().boundElementArrayBuffer);
        if (it != g_buffers.end()) indexBuffer = it->second.metalHandle;
    }
    if (instances <= 0 || !indexBuffer || !beginExperimentalDraw(program)) {
        metalsharp::GLErrorTracker::instance().setError(0x0502); return;
    }
    g_metalRenderer.bindIndexBuffer(indexBuffer, 0);
    g_metalRenderer.drawElementsInstanced(mode, static_cast<uint32_t>(count), type,
                                          reinterpret_cast<size_t>(indices), static_cast<uint32_t>(instances));
    g_metalRenderer.endRenderPass();
    g_metalRenderer.finish();
}

extern "C" void glDrawArraysIndirect(uint32_t mode, const void* indirect) {
    const uint32_t program = g_glBridge.state().currentProgram;
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t, const void*>("glDrawArraysIndirect", mode, indirect);
        return;
    }
    struct Command { uint32_t count, instances, first, baseInstance; } command{};
    uint64_t handle = 0;
    { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it = g_buffers.find(g_boundIndirectBuffer); if (it != g_buffers.end()) handle = it->second.metalHandle; }
    if (!handle || !g_metalRenderer.readBuffer(handle, reinterpret_cast<size_t>(indirect), sizeof(command), &command) ||
        !beginExperimentalDraw(program)) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
    if (command.instances > 1) g_metalRenderer.drawArraysInstanced(mode, command.first, command.count, command.instances);
    else g_metalRenderer.drawArrays(mode, command.first, command.count);
    g_metalRenderer.endRenderPass(); g_metalRenderer.finish();
}

extern "C" void glDrawElementsIndirect(uint32_t mode, uint32_t type, const void* indirect) {
    const uint32_t program = g_glBridge.state().currentProgram;
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t, uint32_t, const void*>("glDrawElementsIndirect", mode, type, indirect);
        return;
    }
    struct Command { uint32_t count, instances, firstIndex, baseVertex, baseInstance; } command{};
    uint64_t indirectHandle = 0, indexHandle = 0;
    { std::lock_guard<std::mutex> lock(g_bufferMutex); auto it = g_buffers.find(g_boundIndirectBuffer); if (it != g_buffers.end()) indirectHandle = it->second.metalHandle; auto ix = g_buffers.find(g_glBridge.state().boundElementArrayBuffer); if (ix != g_buffers.end()) indexHandle = ix->second.metalHandle; }
    size_t indexSize = type == 0x1403 ? 2 : type == 0x1405 ? 4 : 0;
    if (!indirectHandle || !indexHandle || !indexSize || !g_metalRenderer.readBuffer(indirectHandle, reinterpret_cast<size_t>(indirect), sizeof(command), &command) ||
        !beginExperimentalDraw(program)) { metalsharp::GLErrorTracker::instance().setError(0x0502); return; }
    g_metalRenderer.bindIndexBuffer(indexHandle, 0);
    size_t indexOffset = static_cast<size_t>(command.firstIndex) * indexSize;
    if (command.instances > 1) g_metalRenderer.drawElementsInstanced(mode, command.count, type, indexOffset, command.instances);
    else g_metalRenderer.drawElements(mode, command.count, type, indexOffset);
    g_metalRenderer.endRenderPass(); g_metalRenderer.finish();
}

extern "C" void glBindImageTexture(uint32_t unit, uint32_t texture, int32_t level, unsigned char layered,
                                    int32_t layer, uint32_t access, uint32_t format) {
    glDispatch<void, uint32_t, uint32_t, int32_t, unsigned char, int32_t, uint32_t, uint32_t>(
        "glBindImageTexture", unit, texture, level, layered, layer, access, format);
    if (metalModeEnabled() && unit < g_imageUnits.size()) g_imageUnits[unit] = texture;
}

extern "C" void glMultiDrawArraysIndirect(uint32_t mode, const void* indirect, int32_t drawcount, int32_t stride) {
    if (!isExperimentalProgram(g_glBridge.state().currentProgram)) {
        glDispatch<void, uint32_t, const void*, int32_t, int32_t>("glMultiDrawArraysIndirect", mode, indirect, drawcount, stride);
        return;
    }
    const size_t commandStride = stride > 0 ? static_cast<size_t>(stride) : 16;
    for (int32_t i = 0; i < drawcount; ++i)
        glDrawArraysIndirect(mode, static_cast<const uint8_t*>(indirect) + i * commandStride);
}

extern "C" void glMultiDrawElementsIndirect(uint32_t mode, uint32_t type, const void* indirect,
                                             int32_t drawcount, int32_t stride) {
    if (!isExperimentalProgram(g_glBridge.state().currentProgram)) {
        glDispatch<void, uint32_t, uint32_t, const void*, int32_t, int32_t>("glMultiDrawElementsIndirect", mode, type, indirect, drawcount, stride);
        return;
    }
    const size_t commandStride = stride > 0 ? static_cast<size_t>(stride) : 20;
    for (int32_t i = 0; i < drawcount; ++i)
        glDrawElementsIndirect(mode, type, static_cast<const uint8_t*>(indirect) + i * commandStride);
}

extern "C" void glMultiDrawArrays(uint32_t mode, const int32_t* first, const int32_t* count, int32_t drawcount) {
    if (!isExperimentalProgram(g_glBridge.state().currentProgram)) {
        glDispatch<void, uint32_t, const int32_t*, const int32_t*, int32_t>("glMultiDrawArrays", mode, first, count, drawcount);
        return;
    }
    for (int32_t i = 0; i < drawcount; ++i) glDrawArrays(mode, first[i], count[i]);
}

extern "C" void glMultiDrawElements(uint32_t mode, const int32_t* count, uint32_t type,
                                     const void* const* indices, int32_t drawcount) {
    if (!isExperimentalProgram(g_glBridge.state().currentProgram)) {
        glDispatch<void, uint32_t, const int32_t*, uint32_t, const void* const*, int32_t>("glMultiDrawElements", mode, count, type, indices, drawcount);
        return;
    }
    for (int32_t i = 0; i < drawcount; ++i) glDrawElements(mode, count[i], type, indices[i]);
}

extern "C" void glGenSamplers(int32_t n, uint32_t* samplers) {
    glDispatch<void, int32_t, uint32_t*>("glGenSamplers", n, samplers);
    if (metalModeEnabled() && samplers) { std::lock_guard<std::mutex> lock(g_resourceMutex); for (int32_t i = 0; i < n; ++i) g_samplers.emplace(samplers[i], ExperimentalSampler{}); }
}
extern "C" void glDeleteSamplers(int32_t n, const uint32_t* samplers) {
    if (samplers) { std::lock_guard<std::mutex> lock(g_resourceMutex); for (int32_t i = 0; i < n; ++i) g_samplers.erase(samplers[i]); }
    glDispatch<void, int32_t, const uint32_t*>("glDeleteSamplers", n, samplers);
}
extern "C" void glBindSampler(uint32_t unit, uint32_t sampler) {
    glDispatch<void, uint32_t, uint32_t>("glBindSampler", unit, sampler);
    if (metalModeEnabled() && unit < g_samplerUnits.size()) g_samplerUnits[unit] = sampler;
}
extern "C" void glSamplerParameteri(uint32_t sampler, uint32_t pname, int32_t param) {
    glDispatch<void, uint32_t, uint32_t, int32_t>("glSamplerParameteri", sampler, pname, param);
    if (!metalModeEnabled()) return;
    std::lock_guard<std::mutex> lock(g_resourceMutex);
    auto it = g_samplers.find(sampler); if (it == g_samplers.end()) return;
    if (pname == 0x2801) it->second.minFilter = param;
    else if (pname == 0x2800) it->second.magFilter = param;
    else if (pname == 0x2802) it->second.wrapS = param;
    else if (pname == 0x2803) it->second.wrapT = param;
}

extern "C" void glDispatchCompute(uint32_t x, uint32_t y, uint32_t z) {
    const uint32_t program = g_glBridge.state().currentProgram;
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t, uint32_t, uint32_t>("glDispatchCompute", x, y, z);
        return;
    }
    if (!beginExperimentalCompute(program)) {
        metalsharp::GLErrorTracker::instance().setError(0x0502); return;
    }
    g_metalRenderer.dispatchCompute(x, y, z);
    g_metalRenderer.finish();
}

extern "C" void glMemoryBarrier(uint32_t barriers) {
    if (metalModeEnabled()) { g_metalRenderer.finish(); return; }
    glDispatch<void, uint32_t>("glMemoryBarrier", barriers);
}

// ---------------------------------------------------------------------------
// Client-side vertex arrays (legacy immediate-mode interop)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH4(void, glVertexPointer, int32_t, size, uint32_t, type, int32_t, stride, const void*, ptr)
GL_PASSTHROUGH4(void, glTexCoordPointer, int32_t, size, uint32_t, type, int32_t, stride, const void*, ptr)
GL_PASSTHROUGH4(void, glColorPointer, int32_t, size, uint32_t, type, int32_t, stride, const void*, ptr)
GL_PASSTHROUGH3(void, glNormalPointer, uint32_t, type, int32_t, stride, const void*, ptr)

// ---------------------------------------------------------------------------
// Vertex attributes (GL 2.0)
// ---------------------------------------------------------------------------

// glEnableVertexAttribArray / glDisableVertexAttribArray are hand-written
// because they must mirror the per-index enable state into GLState so
// subsequent draw calls can observe which attribute streams are active.
// The native call is still issued so the framework context state stays
// in sync with the shim's view.
extern "C" void glEnableVertexAttribArray(uint32_t index) {
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t)>(g_glBridge.getGLProcAddress("glEnableVertexAttribArray"));
    if (fn) {
        fn(index);
    }
    if (index < metalsharp::kMaxVertexAttribs) {
        g_glBridge.state().vertexAttribEnabled[index] = true;
    }
}

extern "C" void glDisableVertexAttribArray(uint32_t index) {
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t)>(g_glBridge.getGLProcAddress("glDisableVertexAttribArray"));
    if (fn) {
        fn(index);
    }
    if (index < metalsharp::kMaxVertexAttribs) {
        g_glBridge.state().vertexAttribEnabled[index] = false;
    }
}

// glVertexAttribPointer is currently a passthrough. Phase 2h marks it as a
// future interception point — Phase 3 will capture each (index, size, type,
// normalized, stride, pointer) tuple in GLShaderTracker so that the Metal
// draw emitter can build an MTLVertexDescriptor at draw time without
// re-deriving the layout from the translated MSL. The capture must observe
// the current GL_ARRAY_BUFFER binding (already tracked in GLState) to
// resolve the actual MTLBuffer for the attribute stream.
//
// For Phase 2, native GL handles the layout; GLSL 1.20 / legacy shaders
// don't need a Metal vertex descriptor because the framework emits them
// directly. The interception only matters once cross-compiled shaders
// start flowing through the Metal backend.
extern "C" void glVertexAttribPointer(uint32_t index, int32_t size, uint32_t type, unsigned char normalized,
                                      int32_t stride, const void* pointer) {
    const bool experimental = std::getenv("WINEMETALGL_EXPERIMENTAL") &&
                              std::strcmp(std::getenv("WINEMETALGL_EXPERIMENTAL"), "1") == 0;
    const uint32_t bufferName = g_glBridge.state().boundArrayBuffer;
    if (experimental && bufferName) {
        uint64_t handle = 0;
        {
            std::lock_guard<std::mutex> lock(g_bufferMutex);
            auto it = g_buffers.find(bufferName);
            if (it != g_buffers.end()) handle = it->second.metalHandle;
        }
        if (handle) {
            g_metalRenderer.setVertexAttribute(index, size, type, normalized != 0,
                                               static_cast<uint32_t>(stride), handle,
                                               reinterpret_cast<size_t>(pointer));
            return;
        }
    }
    glDispatch<void, uint32_t, int32_t, uint32_t, unsigned char, int32_t, const void*>(
        "glVertexAttribPointer", index, size, type, normalized, stride, pointer);
}
GL_PASSTHROUGH2(void, glVertexAttrib1f, uint32_t, index, float, v0)
GL_PASSTHROUGH3(void, glVertexAttrib2f, uint32_t, index, float, v0, float, v1)
GL_PASSTHROUGH4(void, glVertexAttrib3f, uint32_t, index, float, v0, float, v1, float, v2)
GL_PASSTHROUGH5(void, glVertexAttrib4f, uint32_t, index, float, v0, float, v1, float, v2, float, v3)
GL_PASSTHROUGH3(void, glGetVertexAttribiv, uint32_t, index, uint32_t, pname, int32_t*, params)
GL_PASSTHROUGH3(void, glGetVertexAttribPointerv, uint32_t, index, uint32_t, pname, void**, pointer)
extern "C" void glVertexAttribDivisor(uint32_t index, uint32_t divisor) {
    glDispatch<void, uint32_t, uint32_t>("glVertexAttribDivisor", index, divisor);
    if (index < g_attributeDivisors.size()) g_attributeDivisors[index] = divisor;
}

extern "C" void glGenVertexArrays(int32_t n, uint32_t* arrays) {
    glDispatch<void, int32_t, uint32_t*>("glGenVertexArrays", n, arrays);
    if (metalModeEnabled() && arrays) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        for (int32_t i = 0; i < n; ++i) g_vertexArrays[arrays[i]] = true;
    }
}
extern "C" void glDeleteVertexArrays(int32_t n, const uint32_t* arrays) {
    if (arrays) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        for (int32_t i = 0; i < n; ++i) g_vertexArrays.erase(arrays[i]);
    }
    glDispatch<void, int32_t, const uint32_t*>("glDeleteVertexArrays", n, arrays);
}
extern "C" void glBindVertexArray(uint32_t array) {
    glDispatch<void, uint32_t>("glBindVertexArray", array);
    g_currentVertexArray = array;
    if (metalModeEnabled() && array) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        g_vertexArrays[array] = true;
    }
}
extern "C" unsigned char glIsVertexArray(uint32_t array) {
    if (metalModeEnabled()) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        return g_vertexArrays.count(array) ? 1 : 0;
    }
    return glDispatch<unsigned char, uint32_t>("glIsVertexArray", array);
}

// ---------------------------------------------------------------------------
// Shader objects (GL 2.0)
// ---------------------------------------------------------------------------

// glCreateShader is intercepted by the shader tracker so that subsequent
// glShaderSource / glCompileShader calls observe a consistent handle and can
// be routed through the SPIRV-Cross cross-compilation path. When SPIRV-Cross
// is unavailable, the call falls through to the native OpenGL framework.
extern "C" uint32_t glCreateShader(uint32_t shaderType) {
    ensureGLInit();
#if METALSHARP_HAS_SPIRV_CROSS
    // Always allocate through the tracker so that the cross-compile path has
    // a place to park source / SPIR-V / MSL state. A zero return means the
    // tracker rejected the allocation (e.g. out of memory); in that case we
    // fall through to native GL so the caller still gets a usable handle.
    const uint32_t name = metalsharp::GLShaderTracker::instance().createShader(shaderType);
    if (name != 0) {
        return name;
    }
#endif
    auto fn = reinterpret_cast<uint32_t (*)(uint32_t)>(g_glBridge.getGLProcAddress("glCreateShader"));
    if (fn) {
        return fn(shaderType);
    }
    return 0;
}

// glDeleteShader is intercepted so the tracker entry is removed before
// the native handle goes away. Without this hook the tracker would leak
// entries and recycled handles could collide with live shader objects.
extern "C" void glDeleteShader(uint32_t shader) {
    ensureGLInit();
#if METALSHARP_HAS_SPIRV_CROSS
    metalsharp::GLShaderTracker::instance().deleteShader(shader);
#endif
    auto fn = reinterpret_cast<void (*)(uint32_t)>(g_glBridge.getGLProcAddress("glDeleteShader"));
    if (fn) {
        fn(shader);
    }
}

// glShaderSource is intercepted to capture the GLSL source string into the
// shader's tracker entry, parse its #version directive, and decide whether
// the cross-compile path is required. For tracked shaders we do NOT forward
// the source to native GL — the cross-compile path owns these shaders end to
// end. Shaders that are not in the tracker (e.g. created outside the shim)
// fall through to the native driver unchanged.
extern "C" void glShaderSource(uint32_t shader, int32_t count, const char** string, const int32_t* length) {
    ensureGLInit();
#if METALSHARP_HAS_SPIRV_CROSS
    auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
    if (state) {
        // Concatenate the count source strings into a single buffer. Each
        // string may be null-terminated (length == nullptr or length[i] < 0)
        // or have an explicit byte length (length[i] > 0). Either way the
        // result is one canonical source blob keyed by the shader handle.
        std::string source;
        for (int32_t i = 0; i < count; i++) {
            if (!string || !string[i]) {
                continue;
            }
            if (length && length[i] > 0) {
                source.append(string[i], static_cast<size_t>(length[i]));
            } else {
                source.append(string[i]);
            }
        }
        state->source = source;

        // Parse the #version directive and decide whether the cross-compile
        // path is required. Invalid / missing #version is treated as legacy
        // GLSL 1.10 — needsCrossCompile stays false and the compile falls
        // through to native GL.
        metalsharp::parseGLSLVersion(source.c_str(), state->glslVersion);
        state->needsCrossCompile = metalsharp::needsCrossCompile(state->glslVersion);
        return; // Don't forward to native GL — we handle compilation.
    }
#endif
    auto fn = reinterpret_cast<void (*)(uint32_t, int32_t, const char**, const int32_t*)>(
        g_glBridge.getGLProcAddress("glShaderSource"));
    if (fn) {
        fn(shader, count, string, length);
    }
}

// glGetShaderiv is hand-written so that shaders driven through the
// SPIRV-Cross cross-compile path can report their compile status,
// delete status, and info-log length from the tracker's state instead
// of falling through to the native OpenGL framework (which never saw
// the cross-compile and would always report GL_FALSE / length 0).
// Phase 3 will extend this to also report shader type / source length
// for non-tracked shaders using the same shim state.
extern "C" void glGetShaderiv(uint32_t shader, uint32_t pname, int32_t* params) {
    if (!params) {
        return;
    }
#if METALSHARP_HAS_SPIRV_CROSS
    auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
    if (state && state->compiled) {
        switch (pname) {
        case 0x8B81: // GL_COMPILE_STATUS
            *params = state->compileSuccess ? 1 : 0;
            return;
        case 0x8B80: // GL_DELETE_STATUS
            *params = 0;
            return;
        case 0x8B84: // GL_INFO_LOG_LENGTH
            *params = static_cast<int32_t>(state->infoLog.size()) + 1;
            return;
        default:
            break;
        }
    }
#endif
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t, uint32_t, int32_t*)>(g_glBridge.getGLProcAddress("glGetShaderiv"));
    if (fn) {
        fn(shader, pname, params);
    }
}

// glGetShaderInfoLog is hand-written so cross-compiled shaders can
// surface their translate / compile diagnostics to the guest through
// the normal OpenGL info-log query. Native-only shaders fall through
// to the framework unchanged.
extern "C" void glGetShaderInfoLog(uint32_t shader, int32_t bufSize, int32_t* length, char* infoLog) {
#if METALSHARP_HAS_SPIRV_CROSS
    auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
    if (state && state->compiled && !state->infoLog.empty()) {
        int32_t len = static_cast<int32_t>(state->infoLog.size());
        if (len > bufSize - 1) {
            len = bufSize - 1;
        }
        if (len > 0) {
            std::memcpy(infoLog, state->infoLog.data(), static_cast<size_t>(len));
        }
        infoLog[len] = '\0';
        if (length) {
            *length = len;
        }
        return;
    }
#endif
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t, int32_t, int32_t*, char*)>(
        g_glBridge.getGLProcAddress("glGetShaderInfoLog"));
    if (fn) {
        fn(shader, bufSize, length, infoLog);
    }
}
extern "C" unsigned char glIsShader(uint32_t shader) {
    if (metalsharp::GLShaderTracker::instance().getShader(shader)) {
        return 1;
    }
    return glDispatch<unsigned char, uint32_t>("glIsShader", shader);
}

// glCompileShader is intercepted to drive the SPIRV-Cross cross-compilation
// path for tracked shaders whose source needs cross-compilation (GLSL > 1.20
// or any ES flavour). For native-compatible shaders the call still falls
// through to the native OpenGL driver and shaderCompilePending is reset.
//
// Phase 2i adds a per-(source, stage) MSL cache in front of the
// GLSLCompiler round trip. Cache lookup is by hash, so the same source
// string compiles exactly once per process per stage; subsequent compiles
// short-circuit straight to MSL without paying for glslang or SPIRV-Cross.
// The cache is process-wide and best-effort: failures of the cache lookup
// (which shouldn't happen) fall through to the normal compile path.
extern "C" void glCompileShader(uint32_t shader) {
    ensureGLInit();
#if METALSHARP_HAS_SPIRV_CROSS
    auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
    if (state && state->needsCrossCompile && !state->source.empty()) {
        // Cache lookup: if we already translated this exact (source, stage)
        // pair, skip the GLSLCompiler round trip and use the cached MSL.
        // This is safe because the MSL output for a given (source, stage)
        // is fully determined by the inputs — there is no per-compile
        // randomness in glslang or SPIRV-Cross on our code paths.
        const std::string* cached =
            metalsharp::GLShaderCache::instance().lookupMSL(state->source, static_cast<uint32_t>(state->stage));
        if (cached) {
            state->msl = *cached;
            // The cached entry was a successful translation; the SPIR-V
            // blob itself is not cached (we only store MSL), so rebuild
            // it on demand from the cached MSL by leaving spirv empty.
            // Phase 3 will switch to caching SPIR-V too so we can recover
            // the binary blob without re-running glslang.
            state->spirv.clear();
            state->compiled = true;
            state->compileSuccess = true;
            state->infoLog.clear();
            g_glBridge.state().shaderCompilePending = false;
            return;
        }

        std::string errorLog;
        bool ok = metalsharp::GLSLCompiler::compileToSPIRV(state->source.c_str(), state->stage, state->glslVersion,
                                                           state->spirv, errorLog);
        if (ok) {
            ok = metalsharp::GLSLCompiler::translateSPIRVtoMSL(state->spirv, state->stage, state->msl, errorLog);
        }
        state->compiled = true;
        state->compileSuccess = ok;
        state->infoLog = errorLog;
        // Only cache successful translations — failures might succeed on a
        // retry once underlying tools update, and we don't want to lock in
        // a broken result.
        if (ok) {
            metalsharp::GLShaderCache::instance().storeMSL(state->source, static_cast<uint32_t>(state->stage),
                                                           state->msl);
        }
        g_glBridge.state().shaderCompilePending = false;
        return;
    }
#endif
    auto fn = reinterpret_cast<void (*)(uint32_t)>(g_glBridge.getGLProcAddress("glCompileShader"));
    if (fn) {
        fn(shader);
    }
    g_glBridge.state().shaderCompilePending = false;
}

// ---------------------------------------------------------------------------
// Shader program objects (GL 2.0)
// ---------------------------------------------------------------------------
extern "C" uint32_t glCreateProgram() {
    const char* experimental = std::getenv("WINEMETALGL_EXPERIMENTAL");
    if (experimental && std::strcmp(experimental, "1") == 0) {
        const uint32_t program = metalsharp::GLShaderTracker::instance().createProgram();
        if (program) {
            std::lock_guard<std::mutex> lock(g_programMutex);
            g_programs.emplace(program, ExperimentalProgram{});
        }
        return program;
    }
    return glDispatch<uint32_t>("glCreateProgram");
}

extern "C" void glDeleteProgram(uint32_t program) {
    if (isExperimentalProgram(program)) {
        metalsharp::GLShaderTracker::instance().deleteProgram(program);
        std::lock_guard<std::mutex> lock(g_programMutex);
        g_programs.erase(program);
        if (g_glBridge.state().currentProgram == program) {
            g_glBridge.state().currentProgram = 0;
        }
        return;
    }
    glDispatch<void, uint32_t>("glDeleteProgram", program);
}

extern "C" void glAttachShader(uint32_t program, uint32_t shader) {
    if (isExperimentalProgram(program) && metalsharp::GLShaderTracker::instance().getShader(shader)) {
        metalsharp::GLShaderTracker::instance().attachShader(program, shader);
        return;
    }
    glDispatch<void, uint32_t, uint32_t>("glAttachShader", program, shader);
}

extern "C" void glDetachShader(uint32_t program, uint32_t shader) {
    if (isExperimentalProgram(program)) {
        metalsharp::GLShaderTracker::instance().detachShader(program, shader);
        return;
    }
    glDispatch<void, uint32_t, uint32_t>("glDetachShader", program, shader);
}

extern "C" void glLinkProgram(uint32_t program) {
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t>("glLinkProgram", program);
        return;
    }

    metalsharp::GLShaderState* vertex = nullptr;
    metalsharp::GLShaderState* fragment = nullptr;
    metalsharp::GLShaderState* compute = nullptr;
    for (uint32_t shader : metalsharp::GLShaderTracker::instance().copyAttachedShaders(program)) {
        auto* state = metalsharp::GLShaderTracker::instance().getShader(shader);
        if (!state) {
            continue;
        }
        if (state->stage == metalsharp::ShaderStage::Vertex) {
            vertex = state;
        } else if (state->stage == metalsharp::ShaderStage::Pixel) {
            fragment = state;
        } else if (state->stage == metalsharp::ShaderStage::Compute) {
            compute = state;
        }
    }

    ExperimentalProgram result;
    result.linked = true;
    if (compute && !vertex && !fragment) {
        if (!compute->compiled || !compute->compileSuccess) result.infoLog = "MetalSharp: compute shader did not compile";
        else if (!ensureMetalInit()) result.infoLog = "MetalSharp: Metal device initialization failed";
        else result.linkSuccess = true;
    } else if (!vertex || !fragment) {
        result.infoLog = "MetalSharp: a vertex and fragment shader are required";
    } else if (!vertex->compiled || !vertex->compileSuccess || !fragment->compiled || !fragment->compileSuccess) {
        result.infoLog = "MetalSharp: attached shaders did not compile";
    } else if (!ensureMetalInit()) {
        result.infoLog = "MetalSharp: Metal device initialization failed";
    } else {
        discoverShaderInterface(result, vertex->source, true);
        discoverShaderInterface(result, fragment->source, false);
        /* Pipeline creation is deferred until the first draw. OpenGL sets
         * vertex-array state after linking, and Metal requires that state in
         * the pipeline descriptor for shaders with vertex inputs. */
        result.linkSuccess = true;
    }

    std::lock_guard<std::mutex> lock(g_programMutex);
    g_programs[program] = std::move(result);
}

extern "C" void glValidateProgram(uint32_t program) {
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t>("glValidateProgram", program);
    }
}

extern "C" void glGetProgramiv(uint32_t program, uint32_t pname, int32_t* params) {
    if (!params) {
        return;
    }
    if (isExperimentalProgram(program)) {
        std::lock_guard<std::mutex> lock(g_programMutex);
        auto it = g_programs.find(program);
        if (it == g_programs.end()) {
            *params = 0;
            return;
        }
        switch (pname) {
        case 0x8B82: // GL_LINK_STATUS
        case 0x8B83: // GL_VALIDATE_STATUS
            *params = it->second.linkSuccess ? 1 : 0;
            return;
        case 0x8B84: // GL_INFO_LOG_LENGTH
            *params = static_cast<int32_t>(it->second.infoLog.size()) + 1;
            return;
        case 0x8B86: // GL_ACTIVE_UNIFORMS
            *params = static_cast<int32_t>(it->second.uniformLocations.size());
            return;
        case 0x8B89: // GL_ACTIVE_ATTRIBUTES
            *params = static_cast<int32_t>(it->second.attributeLocations.size());
            return;
        default:
            *params = 0;
            return;
        }
    }
    glDispatch<void, uint32_t, uint32_t, int32_t*>("glGetProgramiv", program, pname, params);
}

extern "C" void glGetProgramInfoLog(uint32_t program, int32_t bufSize, int32_t* length, char* infoLog) {
    if (isExperimentalProgram(program)) {
        std::lock_guard<std::mutex> lock(g_programMutex);
        auto it = g_programs.find(program);
        if (it == g_programs.end() || !infoLog || bufSize <= 0) {
            if (length) {
                *length = 0;
            }
            return;
        }
        const int32_t count =
            std::min<int32_t>(bufSize - 1, static_cast<int32_t>(it->second.infoLog.size()));
        if (count > 0) {
            std::memcpy(infoLog, it->second.infoLog.data(), static_cast<size_t>(count));
        }
        infoLog[count] = '\0';
        if (length) {
            *length = count;
        }
        return;
    }
    glDispatch<void, uint32_t, int32_t, int32_t*, char*>("glGetProgramInfoLog", program, bufSize, length, infoLog);
}

extern "C" unsigned char glIsProgram(uint32_t program) {
    if (isExperimentalProgram(program)) {
        return 1;
    }
    return glDispatch<unsigned char, uint32_t>("glIsProgram", program);
}

// glUseProgram is hand-written because it must mirror the active program
// into GLState::currentProgram so subsequent draw calls can observe which
// program is bound. The native call is still issued so the framework
// context state stays in sync with the shim's view.
extern "C" void glUseProgram(uint32_t program) {
    if (isExperimentalProgram(program)) {
        g_glBridge.state().currentProgram = program;
        return;
    }
    if (!program && isExperimentalProgram(g_glBridge.state().currentProgram)) {
        g_glBridge.state().currentProgram = 0;
        return;
    }
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t)>(g_glBridge.getGLProcAddress("glUseProgram"));
    if (fn) {
        fn(program);
    }
    g_glBridge.state().currentProgram = program;
}

// ---------------------------------------------------------------------------
// Uniforms (GL 2.0)
// ---------------------------------------------------------------------------
//
// Phase 2g: glGetUniformLocation and the glUniform* family are currently
// pass-through to the native OpenGL framework. Phase 3 will intercept them
// so that:
//   * glGetUniformLocation returns a Metal buffer-binding index that maps
//     back to the guest's location (a 1:1 mapping is a reasonable Phase 3
//     default since Metal buffer bindings also use small integers).
//   * glUniform* writes are captured by GLShaderTracker and replayed against
//     the Metal argument buffer at glDrawArrays / glDrawElements time.
//
// For Phase 2 we just forward to native GL so that GLSL 1.20 / legacy
// shaders (which never reach the cross-compile path) keep working without
// any uniform-mapping shim.

extern "C" int32_t glGetUniformLocation(uint32_t program, const char* name) {
    if (!isExperimentalProgram(program))
        return glDispatch<int32_t, uint32_t, const char*>("glGetUniformLocation", program, name);
    if (!name || !*name) return -1;
    std::lock_guard<std::mutex> lock(g_programMutex);
    auto it = g_programs.find(program);
    if (it == g_programs.end()) return -1;
    auto found = it->second.uniformLocations.find(name);
    if (found != it->second.uniformLocations.end()) return found->second;
    const int32_t location = static_cast<int32_t>(it->second.uniformLocations.size());
    it->second.uniformLocations.emplace(name, location);
    return location;
}

template <typename T>
void setExperimentalUniform(int32_t location, const T* values, size_t count) {
    const uint32_t program = g_glBridge.state().currentProgram;
    if (!isExperimentalProgram(program) || location < 0 || !values) return;
    std::lock_guard<std::mutex> lock(g_programMutex);
    auto it = g_programs.find(program);
    if (it == g_programs.end()) return;
    it->second.uniformValues[location] = std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(values),
                                                              reinterpret_cast<const uint8_t*>(values) + count);
}

extern "C" void glUniform1f(int32_t location, float v0) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { float v[4] = {v0, 0, 0, 0}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, float>("glUniform1f", location, v0); }
extern "C" void glUniform2f(int32_t location, float v0, float v1) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { float v[4] = {v0, v1, 0, 0}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, float, float>("glUniform2f", location, v0, v1); }
extern "C" void glUniform3f(int32_t location, float v0, float v1, float v2) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { float v[4] = {v0, v1, v2, 0}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, float, float, float>("glUniform3f", location, v0, v1, v2); }
extern "C" void glUniform4f(int32_t location, float v0, float v1, float v2, float v3) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { float v[4] = {v0, v1, v2, v3}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, float, float, float, float>("glUniform4f", location, v0, v1, v2, v3); }
extern "C" void glUniform1i(int32_t location, int32_t v0) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { int32_t v[4] = {v0, 0, 0, 0}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, int32_t>("glUniform1i", location, v0); }
extern "C" void glUniform2i(int32_t location, int32_t v0, int32_t v1) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { int32_t v[4] = {v0, v1, 0, 0}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, int32_t, int32_t>("glUniform2i", location, v0, v1); }
extern "C" void glUniform3i(int32_t location, int32_t v0, int32_t v1, int32_t v2) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { int32_t v[4] = {v0, v1, v2, 0}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, int32_t, int32_t, int32_t>("glUniform3i", location, v0, v1, v2); }
extern "C" void glUniform4i(int32_t location, int32_t v0, int32_t v1, int32_t v2, int32_t v3) { if (isExperimentalProgram(g_glBridge.state().currentProgram)) { int32_t v[4] = {v0, v1, v2, v3}; setExperimentalUniform(location, v, sizeof(v)); } else glDispatch<void, int32_t, int32_t, int32_t, int32_t, int32_t>("glUniform4i", location, v0, v1, v2, v3); }

#define WINEMETALGL_UNIFORM_ARRAY(name, scalar) \
extern "C" void name(int32_t location, int32_t count, const scalar* values) { \
    if (isExperimentalProgram(g_glBridge.state().currentProgram)) { \
        if (count > 0 && values) setExperimentalUniform(location, values, sizeof(scalar) * static_cast<size_t>(count)); \
    } else glDispatch<void, int32_t, int32_t, const scalar*>(#name, location, count, values); \
}
WINEMETALGL_UNIFORM_ARRAY(glUniform1fv, float)
WINEMETALGL_UNIFORM_ARRAY(glUniform2fv, float)
WINEMETALGL_UNIFORM_ARRAY(glUniform3fv, float)
WINEMETALGL_UNIFORM_ARRAY(glUniform4fv, float)
WINEMETALGL_UNIFORM_ARRAY(glUniform1iv, int32_t)
WINEMETALGL_UNIFORM_ARRAY(glUniform2iv, int32_t)
WINEMETALGL_UNIFORM_ARRAY(glUniform3iv, int32_t)
WINEMETALGL_UNIFORM_ARRAY(glUniform4iv, int32_t)
#undef WINEMETALGL_UNIFORM_ARRAY

extern "C" void glUniformMatrix2fv(int32_t location, int32_t count, unsigned char transpose, const float* value) {
    if (isExperimentalProgram(g_glBridge.state().currentProgram)) { if (count > 0 && value) setExperimentalUniform(location, value, sizeof(float) * 4 * static_cast<size_t>(count)); } else glDispatch<void, int32_t, int32_t, unsigned char, const float*>("glUniformMatrix2fv", location, count, transpose, value);
}
extern "C" void glUniformMatrix3fv(int32_t location, int32_t count, unsigned char transpose, const float* value) {
    if (isExperimentalProgram(g_glBridge.state().currentProgram)) { if (count > 0 && value) setExperimentalUniform(location, value, sizeof(float) * 9 * static_cast<size_t>(count)); } else glDispatch<void, int32_t, int32_t, unsigned char, const float*>("glUniformMatrix3fv", location, count, transpose, value);
}
extern "C" void glUniformMatrix4fv(int32_t location, int32_t count, unsigned char transpose, const float* value) {
    if (isExperimentalProgram(g_glBridge.state().currentProgram)) { if (count > 0 && value) setExperimentalUniform(location, value, sizeof(float) * 16 * static_cast<size_t>(count)); } else glDispatch<void, int32_t, int32_t, unsigned char, const float*>("glUniformMatrix4fv", location, count, transpose, value);
}
GL_PASSTHROUGH3(void, glGetUniformfv, uint32_t, program, int32_t, location, float*, params)
GL_PASSTHROUGH3(void, glGetUniformiv, uint32_t, program, int32_t, location, int32_t*, params)
GL_PASSTHROUGH7(void, glGetActiveUniform, uint32_t, program, uint32_t, index, int32_t, bufSize, int32_t*, length,
                int32_t*, size, uint32_t*, type, char*, name)

// ---------------------------------------------------------------------------
// Vertex attribute location queries (GL 2.0)
// ---------------------------------------------------------------------------
extern "C" int32_t glGetAttribLocation(uint32_t program, const char* name) {
    if (!isExperimentalProgram(program))
        return glDispatch<int32_t, uint32_t, const char*>("glGetAttribLocation", program, name);
    if (!name) return -1;
    std::lock_guard<std::mutex> lock(g_programMutex);
    auto it = g_programs.find(program);
    if (it == g_programs.end()) return -1;
    auto found = it->second.attributeLocations.find(name);
    return found == it->second.attributeLocations.end() ? -1 : found->second;
}
extern "C" void glBindAttribLocation(uint32_t program, uint32_t index, const char* name) {
    if (!isExperimentalProgram(program)) {
        glDispatch<void, uint32_t, uint32_t, const char*>("glBindAttribLocation", program, index, name);
        return;
    }
    if (!name) return;
    std::lock_guard<std::mutex> lock(g_programMutex);
    g_programs[program].attributeLocations[name] = static_cast<int32_t>(index);
}
GL_PASSTHROUGH7(void, glGetActiveAttrib, uint32_t, program, uint32_t, index, int32_t, bufSize, int32_t*, length,
                int32_t*, size, uint32_t*, type, char*, name)

// ---------------------------------------------------------------------------
// Rasterization state (GL 1.0)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH1(void, glCullFace, uint32_t, mode)
GL_PASSTHROUGH1(void, glFrontFace, uint32_t, mode)
GL_PASSTHROUGH1(void, glLineWidth, float, width)
GL_PASSTHROUGH1(void, glPointSize, float, size)
GL_PASSTHROUGH2(void, glPolygonMode, uint32_t, face, uint32_t, mode)
GL_PASSTHROUGH2(void, glPolygonOffset, float, factor, float, units)

// ---------------------------------------------------------------------------
// Stencil state (GL 1.0)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH3(void, glStencilFunc, uint32_t, func, int32_t, ref, uint32_t, mask)
GL_PASSTHROUGH4(void, glStencilFuncSeparate, uint32_t, face, uint32_t, func, int32_t, ref, uint32_t, mask)
GL_PASSTHROUGH3(void, glStencilOp, uint32_t, sfail, uint32_t, dpfail, uint32_t, dppass)
GL_PASSTHROUGH4(void, glStencilOpSeparate, uint32_t, face, uint32_t, sfail, uint32_t, dpfail, uint32_t, dppass)
GL_PASSTHROUGH1(void, glStencilMask, uint32_t, mask)
GL_PASSTHROUGH2(void, glStencilMaskSeparate, uint32_t, face, uint32_t, mask)
GL_PASSTHROUGH1(void, glClearStencil, int32_t, s)

// ---------------------------------------------------------------------------
// Color / blend state (GL 1.0-1.4)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH4(void, glColorMask, unsigned char, r, unsigned char, g, unsigned char, b, unsigned char, a)
GL_PASSTHROUGH1(void, glBlendEquation, uint32_t, mode)
GL_PASSTHROUGH2(void, glBlendEquationSeparate, uint32_t, modeRGB, uint32_t, modeAlpha)
GL_PASSTHROUGH4(void, glBlendFuncSeparate, uint32_t, srcRGB, uint32_t, dstRGB, uint32_t, srcAlpha, uint32_t, dstAlpha)
GL_PASSTHROUGH4(void, glBlendColor, float, r, float, g, float, b, float, a)
GL_PASSTHROUGH1(void, glLogicOp, uint32_t, opcode)

// ---------------------------------------------------------------------------
// Depth state (GL 1.0)
// ---------------------------------------------------------------------------
extern "C" void glDepthMask(unsigned char flag) {
    glDispatch<void, unsigned char>("glDepthMask", flag);
    g_glBridge.state().depthWriteEnabled = flag != 0;
}
GL_PASSTHROUGH2(void, glDepthRange, double, nearVal, double, farVal)
extern "C" void glClearDepth(double depth) {
    glDispatch<void, double>("glClearDepth", depth);
    g_glBridge.state().clearDepth = static_cast<float>(depth);
}

// ---------------------------------------------------------------------------
// Pixel storage / transfer (GL 1.0)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH2(void, glPixelStorei, uint32_t, pname, int32_t, param)
GL_PASSTHROUGH2(void, glPixelStoref, uint32_t, pname, float, param)
GL_PASSTHROUGH1(void, glReadBuffer, uint32_t, mode)
GL_PASSTHROUGH1(void, glDrawBuffer, uint32_t, mode)

// ---------------------------------------------------------------------------
// State queries (GL 1.0-1.1)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH2(void, glGetBooleanv, uint32_t, pname, unsigned char*, params)
GL_PASSTHROUGH2(void, glGetFloatv, uint32_t, pname, float*, params)
GL_PASSTHROUGH2(void, glGetDoublev, uint32_t, pname, double*, params)
GL_PASSTHROUGH3(void, glGetTexEnviv, uint32_t, target, uint32_t, pname, int32_t*, params)
GL_PASSTHROUGH3(void, glGetTexEnvfv, uint32_t, target, uint32_t, pname, float*, params)
// glGetError is hand-written: checks the shim error tracker first
// (Phase 5a), then falls through to native GL.
extern "C" uint32_t glGetError() {
    uint32_t err = metalsharp::GLErrorTracker::instance().getError();
    if (err != 0)
        return err;
    // Note: we do NOT forward to native GL's glGetError here because
    // dlsym on the framework handle may resolve to our own shim symbol
    // on macOS, creating infinite recursion. The native GL error state
    // is irrelevant when we're managing our own context via Metal.
    return 0;
}

// Phase 5b — intercept glGetString for GL_EXTENSIONS.
// The old passthrough at the end of the file is kept for non-extension queries.
extern "C" const uint8_t* glGetString_EXTENSIONS_override(uint32_t name) {
    if (name != 0x1F03)
        return nullptr; // not ours
    // Return our bridge extensions directly (no native lookup — avoids
    // dlsym complexity and infinite-recursion risk).
    static const char kExts[] = "GL_ARB_vertex_buffer_object GL_ARB_framebuffer_object "
                                "GL_EXT_framebuffer_object GL_ARB_shader_objects "
                                "GL_ARB_vertex_shader GL_ARB_fragment_shader "
                                "GL_ARB_multitexture METALSHARP_opengl_bridge";
    return reinterpret_cast<const uint8_t*>(kExts);
}

GL_PASSTHROUGH1(unsigned char, glIsEnabled, uint32_t, cap)

// glGetStringi is hand-written following the glGetString pattern.
extern "C" const uint8_t* glGetStringi(uint32_t name, uint32_t index) {
    ensureGLInit();
    auto fn = reinterpret_cast<const uint8_t* (*)(uint32_t, uint32_t)>(g_glBridge.getGLProcAddress("glGetStringi"));
    if (fn) {
        return fn(name, index);
    }
    return reinterpret_cast<const uint8_t*>("");
}

// ---------------------------------------------------------------------------
// Misc commands (GL 1.0)
// ---------------------------------------------------------------------------
extern "C" void glFlush(void) {
    if (metalModeEnabled()) { g_metalRenderer.flush(); return; }
    glDispatch<void>("glFlush");
}
extern "C" void glFinish(void) {
    if (metalModeEnabled()) { g_metalRenderer.finish(); return; }
    glDispatch<void>("glFinish");
}
extern "C" void* glFenceSync(uint32_t condition, uint32_t flags) {
    if (!metalModeEnabled()) return glDispatch<void*, uint32_t, uint32_t>("glFenceSync", condition, flags);
    void* sync = new uint64_t(1);
    std::lock_guard<std::mutex> lock(g_syncMutex); g_syncs.insert(sync); return sync;
}
extern "C" uint32_t glClientWaitSync(void* sync, uint32_t flags, uint64_t timeout) {
    if (!metalModeEnabled()) return glDispatch<uint32_t, void*, uint32_t, uint64_t>("glClientWaitSync", sync, flags, timeout);
    std::lock_guard<std::mutex> lock(g_syncMutex); if (!g_syncs.count(sync)) return 0x0501;
    g_metalRenderer.finish(); return 0x911C; /* GL_CONDITION_SATISFIED */
}
extern "C" void glWaitSync(void* sync, uint32_t flags, uint64_t timeout) {
    if (metalModeEnabled()) { g_metalRenderer.finish(); return; }
    glDispatch<void, void*, uint32_t, uint64_t>("glWaitSync", sync, flags, timeout);
}
extern "C" void glDeleteSync(void* sync) {
    if (metalModeEnabled()) { std::lock_guard<std::mutex> lock(g_syncMutex); if (g_syncs.erase(sync)) delete static_cast<uint64_t*>(sync); return; }
    glDispatch<void, void*>("glDeleteSync", sync);
}
extern "C" unsigned char glIsSync(void* sync) {
    if (metalModeEnabled()) { std::lock_guard<std::mutex> lock(g_syncMutex); return g_syncs.count(sync) != 0; }
    return glDispatch<unsigned char, void*>("glIsSync", sync);
}
GL_PASSTHROUGH2(void, glHint, uint32_t, target, uint32_t, mode)

// ---------------------------------------------------------------------------
// Display lists (GL 1.0)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH1(uint32_t, glGenLists, int32_t, range)
GL_PASSTHROUGH2(void, glNewList, uint32_t, list, uint32_t, mode)
GL_PASSTHROUGH0(void, glEndList)
GL_PASSTHROUGH1(void, glCallList, uint32_t, list)
GL_PASSTHROUGH3(void, glCallLists, int32_t, n, uint32_t, type, const void*, lists)
GL_PASSTHROUGH2(void, glDeleteLists, uint32_t, list, int32_t, range)
GL_PASSTHROUGH1(unsigned char, glIsList, uint32_t, list)

// ---------------------------------------------------------------------------
// Immediate-mode vertex data (GL 1.0)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH2(void, glVertex2f, float, x, float, y)
GL_PASSTHROUGH3(void, glVertex3f, float, x, float, y, float, z)
GL_PASSTHROUGH4(void, glVertex4f, float, x, float, y, float, z, float, w)
GL_PASSTHROUGH3(void, glNormal3f, float, nx, float, ny, float, nz)
GL_PASSTHROUGH1(void, glTexCoord1f, float, s)
GL_PASSTHROUGH2(void, glTexCoord2f, float, s, float, t)
GL_PASSTHROUGH3(void, glTexCoord3f, float, s, float, t, float, r)
GL_PASSTHROUGH4(void, glTexCoord4f, float, s, float, t, float, r, float, q)
GL_PASSTHROUGH3(void, glColor3ub, unsigned char, r, unsigned char, g, unsigned char, b)
GL_PASSTHROUGH4(void, glColor4ub, unsigned char, r, unsigned char, g, unsigned char, b, unsigned char, a)
GL_PASSTHROUGH2(void, glColorMaterial, uint32_t, face, uint32_t, mode)

// ---------------------------------------------------------------------------
// Lighting / material (GL 1.0)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH3(void, glLightfv, uint32_t, light, uint32_t, pname, const float*, params)
GL_PASSTHROUGH2(void, glLightModelfv, uint32_t, pname, const float*, params)
GL_PASSTHROUGH3(void, glMaterialfv, uint32_t, face, uint32_t, pname, const float*, params)
GL_PASSTHROUGH1(void, glShadeModel, uint32_t, mode)

// ---------------------------------------------------------------------------
// Fog (GL 1.0)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH2(void, glFogfv, uint32_t, pname, const float*, params)
GL_PASSTHROUGH2(void, glFogi, uint32_t, pname, int32_t, param)

// ---------------------------------------------------------------------------
// Alpha test (GL 1.0)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH2(void, glAlphaFunc, uint32_t, func, float, ref)

// ---------------------------------------------------------------------------
// Clip planes (GL 1.0)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH2(void, glClipPlane, uint32_t, plane, const double*, equation)

// ---------------------------------------------------------------------------
// Matrix stack (fixed pipeline, GL 1.0)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH1(void, glMatrixMode, uint32_t, mode)
GL_PASSTHROUGH0(void, glLoadIdentity)
GL_PASSTHROUGH6(void, glOrtho, double, l, double, r, double, b, double, t, double, n, double, f)
GL_PASSTHROUGH6(void, glFrustum, double, l, double, r, double, b, double, t, double, n, double, f)
GL_PASSTHROUGH0(void, glPushMatrix)
GL_PASSTHROUGH0(void, glPopMatrix)
GL_PASSTHROUGH3(void, glTranslatef, float, x, float, y, float, z)
GL_PASSTHROUGH4(void, glRotatef, float, angle, float, x, float, y, float, z)
GL_PASSTHROUGH3(void, glScalef, float, x, float, y, float, z)

// ---------------------------------------------------------------------------
// Color (legacy)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH3(void, glColor3f, float, r, float, g, float, b)
GL_PASSTHROUGH4(void, glColor4f, float, r, float, g, float, b, float, a)

// ---------------------------------------------------------------------------
// Texture upload / readback
// ---------------------------------------------------------------------------
extern "C" void glTexImage2D(uint32_t target, int32_t level, int32_t internalFormat, int32_t w, int32_t h,
                              int32_t border, uint32_t format, uint32_t type, const void* data) {
    const uint32_t textureName = g_activeTextureUnit < g_textureUnits.size() ? g_textureUnits[g_activeTextureUnit] : 0;
    if (metalModeEnabled() && target == 0x0DE1 && level == 0 && w > 0 && h > 0 &&
        (format == 0x1908 || format == 0x80E1) && type == 0x1401 && textureName) {
        std::vector<uint8_t> pixels(static_cast<size_t>(w) * static_cast<size_t>(h) * 4u, 0);
        if (data) {
            const auto* source = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < static_cast<size_t>(w) * static_cast<size_t>(h); ++i) {
                if (format == 0x1908) {
                    pixels[i * 4 + 0] = source[i * 4 + 2];
                    pixels[i * 4 + 1] = source[i * 4 + 1];
                    pixels[i * 4 + 2] = source[i * 4 + 0];
                    pixels[i * 4 + 3] = source[i * 4 + 3];
                } else std::memcpy(pixels.data() + i * 4, source + i * 4, 4);
            }
        }
        uint64_t handle = g_metalRenderer.createTexture(static_cast<uint32_t>(w), static_cast<uint32_t>(h), pixels.data());
        if (!handle) { metalsharp::GLErrorTracker::instance().setError(0x0505); return; }
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        auto& texture = g_textures[textureName];
        texture.metalHandle = handle;
        texture.width = static_cast<uint32_t>(w);
        texture.height = static_cast<uint32_t>(h);
        texture.pixels = std::move(pixels);
        return;
    }
    glDispatch<void, uint32_t, int32_t, int32_t, int32_t, int32_t, int32_t, uint32_t, uint32_t, const void*>(
        "glTexImage2D", target, level, internalFormat, w, h, border, format, type, data);
}
extern "C" void glReadPixels(int32_t x, int32_t y, int32_t w, int32_t h, uint32_t format, uint32_t type, void* data) {
    const uint32_t program = g_glBridge.state().currentProgram;
    if (metalModeEnabled() && (isExperimentalProgram(program) || !program)) {
        bool read = false;
        if (x >= 0 && y >= 0 && w > 0 && h > 0 && format == 0x1908 && type == 0x1401) {
            uint64_t textureHandle = 0;
            { std::lock_guard<std::mutex> lock(g_resourceMutex);
              auto fbo = g_framebuffers.find(g_glBridge.state().boundFramebuffer);
              if (fbo != g_framebuffers.end()) { auto texture = g_textures.find(fbo->second.colorTexture); if (texture != g_textures.end()) textureHandle = texture->second.metalHandle; } }
            read = textureHandle ? g_metalRenderer.readTextureRGBA8(textureHandle, static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(w), static_cast<uint32_t>(h), data) :
                g_metalRenderer.readPixelsRGBA8(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(w), static_cast<uint32_t>(h), data);
        }
        if (read) return;
        metalsharp::GLErrorTracker::instance().setError(0x0502);
        return;
    }
    glDispatch<void, int32_t, int32_t, int32_t, int32_t, uint32_t, uint32_t, void*>(
        "glReadPixels", x, y, w, h, format, type, data);
}

// ---------------------------------------------------------------------------
// Texture objects (GL 1.1-1.3)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH2(void, glGenTextures, int32_t, n, uint32_t*, textures)

extern "C" void glDeleteTextures(int32_t n, const uint32_t* textures) {
    if (textures) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        for (int32_t i = 0; i < n; ++i) g_textures.erase(textures[i]);
    }
    glDispatch<void, int32_t, const uint32_t*>("glDeleteTextures", n, textures);
}

extern "C" void glBindTexture(uint32_t target, uint32_t texture) {
    glDispatch<void, uint32_t, uint32_t>("glBindTexture", target, texture);
    if (target == 0x0DE1 && g_activeTextureUnit < g_textureUnits.size()) {
        g_textureUnits[g_activeTextureUnit] = texture;
        g_glBridge.state().boundTexture2D = texture;
    }
}

extern "C" void glActiveTexture(uint32_t texture) {
    glDispatch<void, uint32_t>("glActiveTexture", texture);
    if (texture >= 0x84C0 && texture < 0x84C0 + g_textureUnits.size())
        g_activeTextureUnit = texture - 0x84C0;
}

extern "C" void glTexParameteri(uint32_t target, uint32_t pname, int32_t param) {
    glDispatch<void, uint32_t, uint32_t, int32_t>("glTexParameteri", target, pname, param);
    if (target != 0x0DE1 || g_activeTextureUnit >= g_textureUnits.size()) return;
    std::lock_guard<std::mutex> lock(g_resourceMutex);
    auto it = g_textures.find(g_textureUnits[g_activeTextureUnit]);
    if (it == g_textures.end()) return;
    if (pname == 0x2801) it->second.minFilter = static_cast<uint32_t>(param);
    else if (pname == 0x2800) it->second.magFilter = static_cast<uint32_t>(param);
    else if (pname == 0x2802) it->second.wrapS = static_cast<uint32_t>(param);
    else if (pname == 0x2803) it->second.wrapT = static_cast<uint32_t>(param);
}
GL_PASSTHROUGH3(void, glTexParameterf, uint32_t, target, uint32_t, pname, float, param)
extern "C" void glTexSubImage2D(uint32_t target, int32_t level, int32_t xoffset, int32_t yoffset,
                                int32_t width, int32_t height, uint32_t format, uint32_t type, const void* pixels) {
    const uint32_t textureName = g_activeTextureUnit < g_textureUnits.size() ? g_textureUnits[g_activeTextureUnit] : 0;
    if (metalModeEnabled() && target == 0x0DE1 && level == 0 && width > 0 && height > 0 &&
        format == 0x1908 && type == 0x1401 && pixels && textureName) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        auto it = g_textures.find(textureName);
        if (it != g_textures.end() && xoffset >= 0 && yoffset >= 0 &&
            static_cast<uint32_t>(xoffset + width) <= it->second.width &&
            static_cast<uint32_t>(yoffset + height) <= it->second.height) {
            const auto* source = static_cast<const uint8_t*>(pixels);
            for (int32_t row = 0; row < height; ++row) {
                for (int32_t col = 0; col < width; ++col) {
                    size_t dst = (static_cast<size_t>(yoffset + row) * it->second.width + xoffset + col) * 4;
                    size_t src = (static_cast<size_t>(row) * width + col) * 4;
                    it->second.pixels[dst + 0] = source[src + 2];
                    it->second.pixels[dst + 1] = source[src + 1];
                    it->second.pixels[dst + 2] = source[src + 0];
                    it->second.pixels[dst + 3] = source[src + 3];
                }
            }
            it->second.metalHandle = g_metalRenderer.createTexture(it->second.width, it->second.height,
                                                                     it->second.pixels.data());
            return;
        }
    }
    glDispatch<void, uint32_t, int32_t, int32_t, int32_t, int32_t, int32_t, uint32_t, uint32_t, const void*>(
        "glTexSubImage2D", target, level, xoffset, yoffset, width, height, format, type, pixels);
}
GL_PASSTHROUGH3(void, glTexEnvi, uint32_t, target, uint32_t, pname, int32_t, param)
GL_PASSTHROUGH3(void, glTexEnvf, uint32_t, target, uint32_t, pname, float, param)
GL_PASSTHROUGH3(void, glGetTexParameteriv, uint32_t, target, uint32_t, pname, int32_t*, params)
GL_PASSTHROUGH3(void, glGetTexParameterfv, uint32_t, target, uint32_t, pname, float*, params)
GL_PASSTHROUGH1(unsigned char, glIsTexture, uint32_t, texture)

// glCopyTexImage2D and glCopyTexSubImage2D are hand-written because they
// take 8 arguments each and GL_PASSTHROUGH8 does not exist.
extern "C" void glCopyTexImage2D(uint32_t target, int32_t level, uint32_t internalformat, int32_t x, int32_t y,
                                 int32_t width, int32_t height, int32_t border) {
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t, int32_t, uint32_t, int32_t, int32_t, int32_t, int32_t, int32_t)>(
        g_glBridge.getGLProcAddress("glCopyTexImage2D"));
    if (fn) {
        fn(target, level, internalformat, x, y, width, height, border);
    }
}

extern "C" void glCopyTexSubImage2D(uint32_t target, int32_t level, int32_t xoffset, int32_t yoffset, int32_t x,
                                    int32_t y, int32_t width, int32_t height) {
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t)>(
        g_glBridge.getGLProcAddress("glCopyTexSubImage2D"));
    if (fn) {
        fn(target, level, xoffset, yoffset, x, y, width, height);
    }
}

// ---------------------------------------------------------------------------
// Renderbuffer objects (GL 3.0 / EXT)
// ---------------------------------------------------------------------------
GL_PASSTHROUGH2(void, glGenRenderbuffers, int32_t, n, uint32_t*, renderbuffers)
GL_PASSTHROUGH2(void, glDeleteRenderbuffers, int32_t, n, const uint32_t*, renderbuffers)
GL_PASSTHROUGH2(void, glBindRenderbuffer, uint32_t, target, uint32_t, renderbuffer)
GL_PASSTHROUGH4(void, glRenderbufferStorage, uint32_t, target, uint32_t, internalformat, int32_t, width, int32_t,
                height)
GL_PASSTHROUGH1(unsigned char, glIsRenderbuffer, uint32_t, renderbuffer)

// ---------------------------------------------------------------------------
// Framebuffer objects (GL 3.0 / EXT_framebuffer_object)
// ---------------------------------------------------------------------------
extern "C" void glGenFramebuffers(int32_t n, uint32_t* framebuffers) {
    glDispatch<void, int32_t, uint32_t*>("glGenFramebuffers", n, framebuffers);
    if (metalModeEnabled() && framebuffers) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        for (int32_t i = 0; i < n; ++i) g_framebuffers.emplace(framebuffers[i], ExperimentalFramebuffer{});
    }
}
extern "C" void glDeleteFramebuffers(int32_t n, const uint32_t* framebuffers) {
    if (framebuffers) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        for (int32_t i = 0; i < n; ++i) g_framebuffers.erase(framebuffers[i]);
    }
    glDispatch<void, int32_t, const uint32_t*>("glDeleteFramebuffers", n, framebuffers);
}
extern "C" void glFramebufferTexture2D(uint32_t target, uint32_t attachment, uint32_t textarget,
                                        uint32_t texture, int32_t level) {
    glDispatch<void, uint32_t, uint32_t, uint32_t, uint32_t, int32_t>(
        "glFramebufferTexture2D", target, attachment, textarget, texture, level);
    if (metalModeEnabled() && target == 0x8D40 && attachment == 0x8CE0 && textarget == 0x0DE1) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        g_framebuffers[g_glBridge.state().boundFramebuffer].colorTexture = texture;
    }
}
GL_PASSTHROUGH4(void, glFramebufferRenderbuffer, uint32_t, target, uint32_t, attachment, uint32_t, renderbuffertarget,
                uint32_t, renderbuffer)
extern "C" uint32_t glCheckFramebufferStatus(uint32_t target) {
    if (metalModeEnabled() && target == 0x8D40) {
        std::lock_guard<std::mutex> lock(g_resourceMutex);
        if (!g_glBridge.state().boundFramebuffer) return 0x8CD5;
        auto fbo = g_framebuffers.find(g_glBridge.state().boundFramebuffer);
        if (fbo != g_framebuffers.end() && g_textures.count(fbo->second.colorTexture)) return 0x8CD5;
        return 0x8CD7;
    }
    return glDispatch<uint32_t, uint32_t>("glCheckFramebufferStatus", target);
}
GL_PASSTHROUGH1(unsigned char, glIsFramebuffer, uint32_t, framebuffer)

// glBindFramebuffer is hand-written because it must mirror the binding into
// GLState so subsequent framebuffer attachment calls can observe which
// framebuffer is currently bound. The native call is still issued so the
// framework context state stays in sync with the shim's view.
extern "C" void glBindFramebuffer(uint32_t target, uint32_t framebuffer) {
    ensureGLInit();
    auto fn = reinterpret_cast<void (*)(uint32_t, uint32_t)>(g_glBridge.getGLProcAddress("glBindFramebuffer"));
    if (fn) fn(target, framebuffer);
    if (target == 0x8D40) {
        g_glBridge.state().boundFramebuffer = framebuffer;
        if (metalModeEnabled() && framebuffer) {
            std::lock_guard<std::mutex> lock(g_resourceMutex);
            g_framebuffers.try_emplace(framebuffer, ExperimentalFramebuffer{});
        }
    }
}

// ---------------------------------------------------------------------------
// Info queries (return non-default values — declared by hand instead of
// via GL_PASSTHROUGH).
// glGetString is hand-written for GL_EXTENSIONS passthrough (Phase 5b).
// ---------------------------------------------------------------------------
extern "C" const uint8_t* glGetString(uint32_t name) {
    // Only handle GL_EXTENSIONS via our bridge (Phase 5b).
    // We do NOT forward to native GL's glGetString because dlsym
    // on the framework handle may resolve to our own shim symbol,
    // creating infinite recursion. Non-extension queries return
    // empty strings; real implementations query via Metal API.
    if (name == 0x1F03) { // GL_EXTENSIONS
        return glGetString_EXTENSIONS_override(name);
    }
    return reinterpret_cast<const uint8_t*>("");
}

extern "C" void glGetIntegerv(uint32_t pname, int32_t* params) {
    ensureGLInit();
    if (!params) {
        return;
    }
    auto fn = reinterpret_cast<void (*)(uint32_t, int32_t*)>(g_glBridge.getGLProcAddress("glGetIntegerv"));
    if (fn) {
        fn(pname, params);
    } else {
        // Best-effort default. Return zero so callers don't read garbage.
        *params = 0;
    }
}

/* Private ABI used by the Wine macOS driver.  The driver owns the Cocoa
 * view and creates the CAMetalLayer through its normal view helpers; the
 * sidecar only receives the opaque layer pointer and therefore does not
 * depend on Wine's private Objective-C classes. */
extern "C" void metalsharp_opengl_set_metal_layer(void* layer) {
    if (ensureMetalInit())
        g_metalRenderer.setMetalLayer(layer);
}

extern "C" int metalsharp_opengl_is_drawable_backed(void) {
    return ensureMetalInit() && g_metalRenderer.isDrawableBacked();
}
