# WineMetalGL current validation

Validation date: 2026-09-13

The candidate artifacts from this validation were staged separately from the
published release and were not used to replace it.

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

`tests/wine/probe-opengl-wow64.sh` was run against the freshly rebuilt Wine
11.17 candidate runtime, with the candidate's matching `opengl32.dll`,
`winemac.so`, `win32u.so`, and sidecar staged together.

The runner created one fresh `WINEARCH=wow64` prefix and sequentially ran
x86_64 and i386 Windows fixtures. Both guests passed:

- `opengl32.dll` and WGL loading.
- Window DC, pixel format, legacy and modern context creation, extension discovery, per-context state across concurrent threads, swap interval routing, post-creation display-list sharing, and shared sidecar object visibility.
- GLSL 1.10/1.20 simple vertex/fragment translation, Metal draw/readback, compatibility FBO/readback, and shading-language version reporting.
- GLSL 3.30 Metal compile/link/draw/readback with an experimental 3.3 context string and tracked viewport/clear-state queries.
- Tracked polygon offset, depth-clamp, and clip-control raster state translated to Metal, including target-0 per-target blend/color-mask entry points.
- OpenGL SPIR-V shader-binary ingestion/specialization, MSL translation, linking, and draw/readback.
- Bounded `glReadnPixels` robustness path.
- MetalSharp program-binary serialization/restoration with subsequent draw/readback.
- GLSL 4.50 Metal compile/link/draw/readback, separable-program pipeline lifecycle/draw, stage-uniform binding, and interface-mismatch rejection.
- Tessellation control/evaluation shader compilation through SPIR-V/MSL.
- Basic triangle and quad tessellation evaluation draws with per-edge/inner factor extraction and patch-default-factor state.
- Simple triangle/point geometry pass-through emulation with varying interface handling, including loop-form vertex emission, and limited pass-through shader transform-feedback capture.
- Fixed-function single/multi-texture environment modes (replace, modulate, add, subtract, decal, blend, and basic combine RGB/alpha operations), plus object/eye-linear, sphere-map, and normal-map texture-coordinate generation.
- Program uniform/attribute interface reflection, uniform-array reflection, and program-resource queries for Metal-owned programs.
- Indexed GL 3.3-style draw with a VBO, IBO, base-vertex offset, DSA vertex binding, multi-bind vertex buffers, vertex attribute, and uniform, plus direct and indirect multi-draw.
- PBO pixel-pack readback; 2D mipmap generation; 2D, 3D, and 2D-array texture upload, 16-bit normalized/half-float upload/readback, RGBA8UI/RGBA16F/RGBA32F/sRGB, red/rg/luminance storage, subimage/copy update, packed-pixel/BGRA conversion, image copy, pixel-pack alignment, multi-bind texture/sampler state, internal-format queries, advanced texture/sampler state, swizzle state, and parameter queries, normalized, half-float, RGBA32F, depth, RGBA8UI, and R32UI integer GLSL texture sampling, and 2D/3D texture readback.
- Color-texture plus depth/depth-stencil texture, renderbuffer, and array-layer FBO attachment, depth/depth-texture/stencil/stencil-texture clear readback, completeness, and blit.
- Metal multisample color textures/renderbuffers with matching raster sample count, resolve attachments, readback, and framebuffer blit.
- Cube-map texture creation, sampling, mipmaps, face readback, and x86_64/i386 validation.
- Rectangle-texture creation, unnormalized sampling, GLSL translation, and readback.
- 1D texture creation, subimage updates, GLSL sampling, and readback.
- Instanced drawing with Metal depth/stencil attachments.
- Direct/indirect compute dispatch with ranged SSBO writeback, storage-block reflection, and program-resource interface queries on both guests.
- Transform-feedback capture for gl_Position and simple scalar/vector-varying expressions, including interleaved and separate buffers, indexed draws, and ranged transform-feedback-buffer offsets.
- Uniform-buffer object binding/range offsets, multi-buffer base binding, bounded buffer mapping, immutable buffer-storage allocation and storage-flag reflection, persistent/coherent flags with WoW64 low-address copy/flush emulation, buffer-pointer reflection, and uniform-block name/size reflection.
- Compute `imageStore` to 2D RGBA8, R32UI, R32F, RGBA8UI, and RGBA16F textures with readback, including multi-bind image-unit setup.
- Arrays/elements indirect draw commands, including parameter-buffer-counted multi-draw.
- Basic fixed-function immediate-mode triangle/texture rendering, point-size and line-width state, texture replace/modulate/add/decal, display lists with matrix command replay, multi-light diffuse/material/specular lighting, linear/exp fog, user clip planes, blending/constant-color/scissor/cull state, and fixed transform capture.
- Sync/fence completion behavior and query-result buffer writes.
- CAMetalLayer-backed default-surface presentation through `SwapBuffers`, including drawable resize/present.

Observed markers:

```text
WINEMETALGL_WOW64_X86_64_OK
WINEMETALGL_WOW64_I386_OK
WINEMETALGL_OPENGL32_LOAD_OK
WINEMETALGL_WGL_CONTEXT_OK
WINEMETALGL_WGL_MODERN_CONTEXT_OK
WINEMETALGL_WGL_SWAP_INTERVAL_OK
WINEMETALGL_WGL_MULTI_CONTEXT_OK
WINEMETALGL_WGL_CONTEXT_STATE_OK
WINEMETALGL_WGL_CONTEXT_THREADS_OK
WINEMETALGL_WGL_SHARE_LISTS_OK
WINEMETALGL_WGL_SHARED_OBJECTS_OK
WINEMETALGL_PROGRAM_PIPELINE_OK
WINEMETALGL_PROGRAM_PIPELINE_DRAW_OK
WINEMETALGL_INTERFACE_REJECT_OK
WINEMETALGL_MULTI_DRAW_OK
WINEMETALGL_MULTI_INDIRECT_OK
WINEMETALGL_MULTI_ELEMENTS_INDIRECT_OK
WINEMETALGL_METAL_SURFACE_OK
WINEMETALGL_DEFAULT_FBO_PRESENT_OK
WINEMETALGL_RESIZE_PRESENT_OK
WINEMETALGL_READBACK_OK
WINEMETALGL_GLSL110_OK
WINEMETALGL_GLSL120_OK
WINEMETALGL_GLSL330_OK
WINEMETALGL_SHADING_LANGUAGE_VERSION_OK
WINEMETALGL_STATE_QUERY_OK
WINEMETALGL_ROBUSTNESS_OK
WINEMETALGL_RASTER_STATE_OK
WINEMETALGL_CLIP_CONTROL_OK
WINEMETALGL_GLSL450_OK
WINEMETALGL_GL33_RESOURCES_OK
WINEMETALGL_BUFFER_MAP_RANGE_OK
WINEMETALGL_BUFFER_STORAGE_OK
WINEMETALGL_BUFFER_POINTER_QUERY_OK
WINEMETALGL_VERTEX_ATTRIB_QUERY_OK
WINEMETALGL_MULTI_BIND_VERTEX_BUFFERS_OK
WINEMETALGL_UNIFORM_ARRAY_REFLECTION_OK
WINEMETALGL_PBO_READBACK_OK
WINEMETALGL_UNPACK_ALIGNMENT_OK
WINEMETALGL_INTERNAL_FORMAT_QUERY_OK
WINEMETALGL_MIPMAP_OK
WINEMETALGL_GL33_TEXTURE_OK
WINEMETALGL_INTEGER_TEXTURE_SAMPLE_OK
WINEMETALGL_HALF_FLOAT_TEXTURE_SAMPLE_OK
WINEMETALGL_FLOAT_TEXTURE_SAMPLE_OK
WINEMETALGL_DEPTH_TEXTURE_SAMPLE_OK
WINEMETALGL_R32UI_TEXTURE_SAMPLE_OK
WINEMETALGL_TEXTURE_PACKED_OK
WINEMETALGL_TEXTURE_USHORT_OK
WINEMETALGL_TEXTURE_HALF_FLOAT_OK
WINEMETALGL_TEXTURE_RGBA16F_OK
WINEMETALGL_TEXTURE_RGBA32F_OK
WINEMETALGL_TEXTURE_RGBA8UI_OK
WINEMETALGL_TEXTURE_SRGB_OK
WINEMETALGL_TEXTURE_RED_OK
WINEMETALGL_TEXTURE_RG_OK
WINEMETALGL_TEXTURE_LUMINANCE_OK
WINEMETALGL_READBACK_PACKED_OK
WINEMETALGL_READBACK_HALF_FLOAT_OK
WINEMETALGL_PACK_ALIGNMENT_OK
WINEMETALGL_TEXTURE_BGRA_OK
WINEMETALGL_TEXTURE_SAMPLER_PARAMS_OK
WINEMETALGL_MULTI_BIND_TEXTURES_OK
WINEMETALGL_TEXTURE_SAMPLER_QUERY_OK
WINEMETALGL_TEXTURE_SWIZZLE_OK
WINEMETALGL_COPY_IMAGE_OK
WINEMETALGL_COPY_TEX_OK
WINEMETALGL_TEX_STORAGE_OK
WINEMETALGL_DSA_TEXTURE_OK
WINEMETALGL_TEXTURE_VIEW_OK
WINEMETALGL_SAMPLER_ADVANCED_OK
WINEMETALGL_FBO_OK
WINEMETALGL_FBO_DEPTH_TEXTURE_OK
WINEMETALGL_FBO_DEPTH_READBACK_OK
WINEMETALGL_FBO_DEPTH_TEXTURE_READBACK_OK
WINEMETALGL_FBO_STENCIL_READBACK_OK
WINEMETALGL_FBO_STENCIL_TEXTURE_READBACK_OK
WINEMETALGL_FBO_ARRAY_LAYER_OK
WINEMETALGL_BLIT_OK
WINEMETALGL_RENDERBUFFER_OK
WINEMETALGL_MULTISAMPLE_OK
WINEMETALGL_MULTISAMPLE_STORAGE_OK
WINEMETALGL_SYNC_OK
WINEMETALGL_QUERY_OK
WINEMETALGL_QUERY_BUFFER_OK
WINEMETALGL_TESSELLATION_COMPILE_OK
WINEMETALGL_TESSELLATION_DRAW_OK
WINEMETALGL_TESSELLATION_FACTORS_OK
WINEMETALGL_TESSELLATION_DEFAULT_FACTORS_OK
WINEMETALGL_TESSELLATION_QUAD_OK
WINEMETALGL_GEOMETRY_PASSTHROUGH_OK
WINEMETALGL_GEOMETRY_POINT_PASSTHROUGH_OK
WINEMETALGL_GEOMETRY_VARYING_PASSTHROUGH_OK
WINEMETALGL_UBO_OK
WINEMETALGL_BUFFER_SIZE_OK
WINEMETALGL_UBO_RANGE_OK
WINEMETALGL_MULTI_BIND_BUFFERS_OK
WINEMETALGL_UBO_REFLECTION_OK
WINEMETALGL_TEXTURE3D_OK
WINEMETALGL_TEXTURE_ARRAY_OK
WINEMETALGL_CLEAR_OK
WINEMETALGL_CLEAR_BUFFER_OK
WINEMETALGL_INSTANCED_OK
WINEMETALGL_COMPUTE_OK
WINEMETALGL_SSBO_REFLECTION_OK
WINEMETALGL_RESOURCE_REFLECTION_OK
WINEMETALGL_COMPUTE_RANGE_OK
WINEMETALGL_COMPUTE_INDIRECT_OK
WINEMETALGL_IMAGE_OK
WINEMETALGL_MULTI_BIND_IMAGES_OK
WINEMETALGL_IMAGE_R32UI_OK
WINEMETALGL_IMAGE_R32F_OK
WINEMETALGL_IMAGE_RGBA8UI_OK
WINEMETALGL_IMAGE_RGBA16F_OK
WINEMETALGL_TEXTURE_CUBE_OK
WINEMETALGL_TEXTURE_CUBE_QUERY_OK
WINEMETALGL_TEXTURE_RECTANGLE_OK
WINEMETALGL_TEXTURE_1D_OK
WINEMETALGL_INDIRECT_OK
WINEMETALGL_INDIRECT_COUNT_OK
WINEMETALGL_FIXED_OK
WINEMETALGL_FIXED_LIST_MATRIX_OK
WINEMETALGL_FIXED_LIGHTING_OK
WINEMETALGL_FIXED_SPECULAR_OK
WINEMETALGL_FIXED_FOG_OK
WINEMETALGL_CLIP_PLANE_OK
WINEMETALGL_FIXED_POINT_SIZE_OK
WINEMETALGL_FIXED_LINE_WIDTH_OK
WINEMETALGL_BLEND_OK
WINEMETALGL_PER_TARGET_BLEND_OK
WINEMETALGL_BLEND_CONSTANT_OK
WINEMETALGL_SCISSOR_OK
WINEMETALGL_CULL_OK
WINEMETALGL_TRANSFORM_FIXED_OK
WINEMETALGL_TRANSFORM_SHADER_OK
WINEMETALGL_TRANSFORM_SHADER_ELEMENTS_OK
WINEMETALGL_TRANSFORM_SHADER_RANGE_OK
WINEMETALGL_TRANSFORM_SHADER_VARYING_OK
WINEMETALGL_TRANSFORM_SHADER_SEPARATE_OK
WINEMETALGL_SHADER_BINARY_OK
WINEMETALGL_SHADER_SPECIALIZE_OK
WINEMETALGL_PROGRAM_BINARY_OK
WINEMETALGL_FIXED_TEXTURE_OK
WINEMETALGL_FIXED_TEXTURE_REPLACE_OK
WINEMETALGL_FIXED_TEXTURE_ADD_OK
WINEMETALGL_FIXED_TEXTURE_DECAL_OK
WINEMETALGL_FIXED_TEXTURE_COMBINE_OK
WINEMETALGL_FIXED_TEXTURE_SUBTRACT_OK
WINEMETALGL_FIXED_MULTITEXTURE_OK
WINEMETALGL_FIXED_TEXGEN_OK
```

The explicit `WINEMETALGL_METAL_SURFACE_OK`,
`WINEMETALGL_DEFAULT_FBO_PRESENT_OK`, and `WINEMETALGL_READBACK_OK` markers
prove the drawable-backed path without requiring verbose Wine logging. The
native translation probe also reports `WINEMETALGL_NATIVE_TRANSLATION_OK`.

## Scope boundary

This candidate does not claim complete Khronos OpenGL 4.6 conformance. The
machine-readable support boundary is `docs/api-coverage.json`; geometry,
general tessellation-control/evaluation semantics, general transform feedback,
fixed-function lighting/matrices/display lists, image formats beyond the
validated 2D RGBA8/R32UI/R32F/RGBA16F paths, and the full texture/sampler/FBO
format matrix remain explicitly unadvertised. Fixed-function texture
combine/coordinate generation is limited to the validated modes above.
