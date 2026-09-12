# WineMetalGL 1.8.0 validation

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
- GLSL 3.30 Metal compile/link/draw/readback with an experimental 3.3 context string.
- GLSL 4.50 Metal compile/link/draw/readback.
- Tessellation control/evaluation shader compilation through SPIR-V/MSL.
- Basic triangle and quad tessellation evaluation draws with fixed factors.
- Simple triangle geometry pass-through emulation and limited pass-through shader transform-feedback capture.
- Program uniform/attribute interface reflection for Metal-owned programs.
- Indexed GL 3.3-style draw with a VBO, IBO, base-vertex offset, DSA vertex binding, vertex attribute, and uniform, plus direct and indirect multi-draw.
- 2D mipmap generation; 2D, 3D, and 2D-array texture upload, 16-bit normalized/half-float upload/readback, subimage/copy update, packed-pixel/BGRA conversion, image copy, pixel-pack alignment, sampler state, GLSL texture sampling, and 2D/3D texture readback.
- Color-texture plus depth/depth-stencil texture, renderbuffer, and array-layer FBO attachment, completeness, readback, and blit.
- Multisample texture/renderbuffer fallback and readback.
- Instanced drawing with Metal depth/stencil attachments.
- Compute shader dispatch with SSBO writeback on both guests.
- Uniform-buffer object binding and reflection.
- Compute `imageStore` to a 2D RGBA8 texture with FBO readback.
- Arrays/elements indirect draw commands.
- Basic fixed-function immediate-mode triangle/texture rendering, texture replace/modulate, display lists, multi-light diffuse/material lighting, linear/exp fog, blending/scissor/cull state, and fixed transform capture.
- Sync/fence completion behavior.
- CAMetalLayer-backed default-surface presentation through `SwapBuffers`.

Observed markers:

```text
WINEMETALGL_WOW64_X86_64_OK
WINEMETALGL_WOW64_I386_OK
WINEMETALGL_OPENGL32_LOAD_OK
WINEMETALGL_WGL_CONTEXT_OK
WINEMETALGL_WGL_MULTI_CONTEXT_OK
WINEMETALGL_MULTI_DRAW_OK
WINEMETALGL_MULTI_INDIRECT_OK
WINEMETALGL_MULTI_ELEMENTS_INDIRECT_OK
WINEMETALGL_METAL_SURFACE_OK
WINEMETALGL_DEFAULT_FBO_PRESENT_OK
WINEMETALGL_READBACK_OK
WINEMETALGL_GLSL330_OK
WINEMETALGL_GLSL450_OK
WINEMETALGL_GL33_RESOURCES_OK
WINEMETALGL_UNPACK_ALIGNMENT_OK
WINEMETALGL_MIPMAP_OK
WINEMETALGL_GL33_TEXTURE_OK
WINEMETALGL_TEXTURE_PACKED_OK
WINEMETALGL_TEXTURE_USHORT_OK
WINEMETALGL_TEXTURE_HALF_FLOAT_OK
WINEMETALGL_READBACK_PACKED_OK
WINEMETALGL_READBACK_HALF_FLOAT_OK
WINEMETALGL_PACK_ALIGNMENT_OK
WINEMETALGL_TEXTURE_BGRA_OK
WINEMETALGL_COPY_IMAGE_OK
WINEMETALGL_COPY_TEX_OK
WINEMETALGL_TEX_STORAGE_OK
WINEMETALGL_DSA_TEXTURE_OK
WINEMETALGL_FBO_OK
WINEMETALGL_FBO_DEPTH_TEXTURE_OK
WINEMETALGL_FBO_ARRAY_LAYER_OK
WINEMETALGL_BLIT_OK
WINEMETALGL_RENDERBUFFER_OK
WINEMETALGL_MULTISAMPLE_OK
WINEMETALGL_SYNC_OK
WINEMETALGL_QUERY_OK
WINEMETALGL_TESSELLATION_COMPILE_OK
WINEMETALGL_TESSELLATION_DRAW_OK
WINEMETALGL_TESSELLATION_QUAD_OK
WINEMETALGL_GEOMETRY_PASSTHROUGH_OK
WINEMETALGL_UBO_OK
WINEMETALGL_BUFFER_SIZE_OK
WINEMETALGL_TEXTURE3D_OK
WINEMETALGL_TEXTURE_ARRAY_OK
WINEMETALGL_CLEAR_OK
WINEMETALGL_CLEAR_BUFFER_OK
WINEMETALGL_INSTANCED_OK
WINEMETALGL_COMPUTE_OK
WINEMETALGL_IMAGE_OK
WINEMETALGL_INDIRECT_OK
WINEMETALGL_FIXED_OK
WINEMETALGL_FIXED_LIGHTING_OK
WINEMETALGL_FIXED_FOG_OK
WINEMETALGL_BLEND_OK
WINEMETALGL_SCISSOR_OK
WINEMETALGL_CULL_OK
WINEMETALGL_TRANSFORM_FIXED_OK
WINEMETALGL_TRANSFORM_SHADER_OK
WINEMETALGL_FIXED_TEXTURE_OK
WINEMETALGL_FIXED_TEXTURE_REPLACE_OK
```

The explicit `WINEMETALGL_METAL_SURFACE_OK`,
`WINEMETALGL_DEFAULT_FBO_PRESENT_OK`, and `WINEMETALGL_READBACK_OK` markers
prove the drawable-backed path without requiring verbose Wine logging.

## Scope boundary

This release does not claim complete Khronos OpenGL 4.6 conformance. The
machine-readable support boundary is `docs/api-coverage.json`; geometry,
tessellation, transform feedback, fixed-function lighting/matrices/display
lists, image formats beyond the validated 2D RGBA8 path, the full
texture/sampler/FBO format matrix, and EGL/GLES remain explicitly unadvertised.
