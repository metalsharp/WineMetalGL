/// @file OpenGLBridge.mm
/// @brief macOS implementation of the OpenGL bridge.
///
/// Loads /System/Library/Frameworks/OpenGL.framework/OpenGL via dlopen and
/// exposes dlsym-based function lookup to the opengl32 shim. On macOS the
/// legacy GL profile tops out at 2.1 — the bridge records that in GLState
/// so downstream consumers can gate on the available version. GL 3.x/4.x
/// shader translation via SPIRV-Cross is scaffolded for Phase 4c.

#include <cstring>
#include <dlfcn.h>
#include <metalsharp/Logger.h>
#include <metalsharp/OpenGLBridge.h>

namespace metalsharp {

struct OpenGLBridge::Impl {
    void* glFramework = nullptr;
};

OpenGLBridge::OpenGLBridge() : m_impl(new Impl()) {
    // GLState is value-initialized by its default member initializers, but
    // keep an explicit zero here so the contract is clear in the .mm too.
    memset(&m_state, 0, sizeof(m_state));
    m_state.clearColor[3] = 1.0f;
    m_state.clearDepth = 1.0f;
    m_state.depthWriteEnabled = true;
    m_state.depthFunc = 0x0201;
    m_state.blendSrcRGB = m_state.blendSrcAlpha = 1;
    m_state.blendDstRGB = m_state.blendDstAlpha = 0;
    m_state.blendEquationRGB = m_state.blendEquationAlpha = 0x8006;
    m_state.colorMask[0] = m_state.colorMask[1] = m_state.colorMask[2] = m_state.colorMask[3] = true;
    m_state.patchVertices = 3;
    m_state.packAlignment = 4;
    m_state.unpackAlignment = 4;
    m_state.stencilFunc = 0x0207;
    m_state.stencilValueMask = 0xffffffffu;
    m_state.stencilWriteMask = 0xffffffffu;
    m_state.stencilFail = m_state.stencilDepthFail = m_state.stencilDepthPass = 0x1e00;
}

OpenGLBridge::~OpenGLBridge() {
    if (m_impl->glFramework) {
        dlclose(m_impl->glFramework);
        m_impl->glFramework = nullptr;
    }
    delete m_impl;
    m_impl = nullptr;
}

bool OpenGLBridge::init() {
    // Load the system OpenGL framework. RTLD_NOW forces all referenced
    // symbols to resolve immediately; RTLD_LOCAL keeps the framework out
    // of the global namespace so we don't pollute other dylibs.
    m_impl->glFramework = dlopen("/System/Library/Frameworks/OpenGL.framework/OpenGL", RTLD_NOW | RTLD_LOCAL);

    if (!m_impl->glFramework) {
        MS_WARN("OpenGLBridge: system OpenGL framework not available (%s)", dlerror());
        m_hasNativeGL = false;
        m_state.version = GLVersion::None;
        return false;
    }

    m_hasNativeGL = true;
    // macOS native OpenGL supports up to 2.1 on the legacy profile. Core
    // profile 3.2+ exists on some macOS versions but is deprecated and
    // not reliably available; we conservatively report 2.1.
    m_state.version = GLVersion::GL21;
    MS_INFO("OpenGLBridge: initialized with native macOS OpenGL 2.1");
    return true;
}

void* OpenGLBridge::getGLProcAddress(const char* name) {
    if (!m_impl->glFramework || !name) {
        return nullptr;
    }
    // dlsym on the framework handle resolves the symbol if it is exported
    // there. Most GL entry points live in the OpenGL framework itself;
    // symbols not found fall through and return nullptr.
    return dlsym(m_impl->glFramework, name);
}

} // namespace metalsharp