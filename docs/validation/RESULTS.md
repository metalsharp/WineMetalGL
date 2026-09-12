# WineMetalGL 0.2.0 validation

Validation date: 2026-09-12

## Host gates

```text
x86_64 Mach-O sidecar: PASS
macOS deployment target: 15.0
GLSL330 glslang -> SPIR-V -> MSL translation: PASS
WINEMETALGL_NATIVE_TRANSLATION_OK
```

The sidecar and matching Wine Unix driver report exactly one Mach-O
architecture (`x86_64`). No ARM or universal artifact is part of this release.

## Wine gates

`tests/wine/probe-opengl-wow64.sh` was run against:

```text
/Volumes/AverySSD/Crossover-WineForge-macos15/merged-build/install/wine-vulkan-portability-test
```

The runner created one fresh `WINEARCH=wow64` prefix and sequentially ran
x86_64 and i386 Windows fixtures. Both guests passed:

- `opengl32.dll` and WGL loading.
- Window DC, pixel format, context creation, and extension discovery.
- GLSL 1.20 compatibility FBO/readback.
- GLSL 3.30 Metal compile/link/draw/readback.
- GLSL 4.50 Metal compile/link/draw/readback.
- Indexed GL 3.3-style draw with a VBO, IBO, vertex attribute, and uniform.
- CAMetalLayer-backed default-surface presentation through `SwapBuffers`.

Observed markers:

```text
WINEMETALGL_WOW64_X86_64_OK
WINEMETALGL_WOW64_I386_OK
WINEMETALGL_OPENGL32_LOAD_OK
WINEMETALGL_WGL_CONTEXT_OK
WINEMETALGL_METAL_SURFACE_OK
WINEMETALGL_DEFAULT_FBO_PRESENT_OK
WINEMETALGL_READBACK_OK
WINEMETALGL_GLSL330_OK
WINEMETALGL_GLSL450_OK
WINEMETALGL_GL33_RESOURCES_OK
```

The Wine log also reports:

```text
WineMetalGL CAMetalLayer present path
```

which distinguishes the presented drawable from the legacy CGL fallback.

## Scope boundary

This release does not claim complete Khronos OpenGL 4.6 conformance. The
machine-readable support boundary is `docs/api-coverage.json`; geometry,
tessellation, transform feedback, compute/image/SSBO execution, the full
texture/sampler/FBO format matrix, and EGL/GLES remain explicitly unadvertised.
