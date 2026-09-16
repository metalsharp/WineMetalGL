# WineMetalGL conformance and coverage

This document is the detailed companion to the short project README. It
separates **implemented code**, **validated behavior**, and **unsupported or
unresolved behavior**. A feature is not considered complete merely because an
entry point is exported or forwarded.

## Test environment

Validation uses:

- macOS 15 or newer on an Apple Silicon host; the reference run used an Apple M4.
- x86_64-only Mach-O artifacts.
- A compatible x86_64 Wine build with the supplied OpenGL integration patch; the reference run used WineForge/Wine 11.17.
- Fresh `WINEARCH=wow64` prefixes.
- A locally built VK-GL-CTS `glcts.exe`. Set `CTS_ROOT` to the VK-GL-CTS checkout and use `$CTS_ROOT/build-win64/external/openglcts/modules/glcts.exe`.

The VK-GL-CTS repository is the source of the conformance executable and test definitions; it is not vendored into this adapter. Set its checkout explicitly, for example:

```sh
CTS_ROOT=/path/to/VK-GL-CTS
GLCTS="$CTS_ROOT/build-win64/external/openglcts/modules/glcts.exe"
```

The broad sweep command is:

```sh
"$GLCTS" --deqp-case='KHR-GL33.*' \
  --deqp-terminate-on-device-lost=disable \
  --deqp-log-filename=/tmp/gl33.qpa
```

The broad sweep remains a diagnostic, not a full-conformance claim. The last
complete 9,887-case snapshot recorded 6,441 Pass, 703 NotSupported, 2,734
Fail, and 9 InternalError results. Unsupported results are retained where the runtime 
cannot provide the required behavior. The machine-readable entry-point
boundary is [`api-coverage.json`](api-coverage.json).

## Coverage summary

The percentages below are engineering coverage estimates for the implementation
surface. They are not percentages of all OpenGL commands and are not Khronos
pass rates. GL 3.0 and GL 3.1 also have the targeted CTS measurements shown
below.

| Level | Coverage | Evidence / scope |
|---|---:|---|
| 1.0 | 100% | Wine compatibility path and immediate-mode baseline |
| 1.1 | 100% | Texture objects, client arrays, and legacy object plumbing |
| 1.2 | 95% | Core legacy rendering and pixel transfer subset |
| 1.3 | 95% | Multitexture and texture-coordinate generation subset |
| 1.4 | 90% | Blend, fog, point/line, and legacy state subset |
| 1.5 | 90% | Buffer objects, queries, and synchronization foundations |
| 2.0 | 90% | GLSL 1.10/1.20, shader objects, attributes, and FBO subset |
| 2.1 | 85% | Compatibility GLSL and fixed-function integration |
| 3.0 | 97% | Targeted `KHR-GL30.*`: 856/879 Pass, 23 NotSupported, 0 Fail |
| 3.1 | 96% | Targeted `KHR-GL31.*`: 856/889 Pass, 33 NotSupported |
| 3.2 | 80% | MSAA, expanded FBO, and adjacent core-resource subset |
| 3.3 | 70% |  |
| 4.0 | 45% | Tessellation compilation and limited evaluation draws |
| 4.1 | 35% | Separable programs and program-pipeline lifecycle |
| 4.2 | 25% | Image, atomic, and expanded texture/resource subset |
| 4.3 | 20% | Compute, SSBO, image, and reflection subset |
| 4.4 | 15% | Selected newer resource and robustness entry points |
| 4.5 | 10% | Selected DSA and robustness entry points |
| 4.6 | 0% | Not implemented or claimed |

## OpenGL 1.0–1.5

### Implemented

- Wine legacy OpenGL compatibility entry points.
- Immediate-mode triangles and basic textured rendering.
- Vertex arrays, texture objects, basic texture upload and readback.
- Multitexture, texture environment modes, texture-coordinate generation,
  blending, fog, point size, line width, scissor, culling, and basic queries.
- Display-list matrix command recording and replay.
- Buffer objects, query objects, fences, and basic synchronization.

### Validated

The Wine fixtures cover legacy context creation, immediate-mode rendering,
fixed-function texture modes, multiple lights, fog, display-list matrix replay,
queries, synchronization, and x86_64/i386 WoW64 loading.

### Boundary

Wine's compatibility implementation remains the owner of behavior not listed as
Metal-owned. This project does not claim that every legacy extension or vendor
extension is implemented by the sidecar.

## OpenGL 2.0–2.1

### Implemented

- GLSL 1.10 and 1.20 simple vertex/fragment translation through the
  glslang/SPIR-V/SPIRV-Cross/MSL path.
- Shader compile/link tracking, attributes, uniforms, textures, and FBO
  readback for the validated subset.
- Legacy fixed-function and shader paths can coexist in one Wine context.
- Basic interface and state reflection.

### Validated

The WoW64 fixtures report `WINEMETALGL_GLSL110_OK`,
`WINEMETALGL_GLSL120_OK`, `WINEMETALGL_READBACK_OK`, and the associated WGL,
FBO, texture, and compatibility markers.

### Boundary

The sidecar does not advertise complete OpenGL 2.x extension coverage or every
legacy GLSL corner case.

## OpenGL 3.0

### Implemented

- GLSL 3.30 vertex/fragment translation and Metal pipeline creation.
- VAOs, VBOs, IBOs, indexed and instanced drawing.
- Integer, normalized, half-float, depth, packed-pixel, 1D, 2D, 3D, array,
  rectangle, cube, and selected multisample texture paths.
- FBO color/depth/stencil attachments, clear/readback, blit, invalidate, and
  completeness handling.
- Texture swizzle and sampler state for the implemented formats.
- Transform feedback for the limited scalar/vector and `gl_Position` paths.
- Basic program-resource and uniform/attribute reflection.

### Validation

Targeted evidence recorded for the matching CTS build includes:

- `KHR-GL30.texture_repeat_mode.*`: 162/162 Pass.
- `KHR-GL30.texture_lod_basic.*`: 1/1 Pass.
- `KHR-GL30.texture_lod_bias.*`: 1/1 Pass.
- `KHR-GL30.framebuffer_blit.*`: 3/3 Pass.
- `KHR-GL30.shaders30.*`: 651/651 Pass.
- `KHR-GL30.buffer_objects.*`: 5/5 Pass.
- `KHR-GL30.transform_feedback.*`: 21/21 Pass.

The broader GL 3.0 diagnostic snapshot recorded 856/879 Pass, 23
NotSupported, and no functional failures in that targeted matrix.

### Boundary

Compressed, vendor-specific, and unusual format combinations remain outside
the advertised surface unless listed in the API manifest or validation record.

## OpenGL 3.1

### Implemented

- Uniform buffer objects, ranges, bindings, reflection, and matrix arrays.
- Primitive restart for the tested indexed primitive families.
- Copy-buffer and indirect draw paths.
- Texture-size promotion and signed-normalized/array/3D/multisample subsets.
- PBO pixel-pack readback, map/flush behavior, and bounded robustness paths.

### Validation

The matching targeted run recorded **856/889 Pass**, **33 explicitly
NotSupported**, and no functional failures in the exercised group. The
primitive-restart and texture-size-promotion cases were also run directly.

### Boundary

The 33 NotSupported cases are retained as unsupported declarations rather than
being represented as passing behavior.

## OpenGL 3.2

### Implemented

- Multisample color textures/renderbuffers with device-supported sample counts.
- Resolve attachments, multisample readback, and framebuffer blit/resolve.
- Expanded depth/stencil texture and renderbuffer attachment handling.
- Selected geometry-adjacent and sampler/FBO state needed by later GL 3.x
  paths.

### Boundary

General geometry-shader semantics, the full multisample format matrix, and all
3.2 extensions are not claimed.

## OpenGL 3.3

### Implemented

- GL 3.3-style buffer, VAO, indexed, instanced, base-vertex, indirect, and
  multi-draw paths.
- GLSL array indexing, matrix/vector indexing, array constructors, invalid
  GLSL 3.30 syntax rejection, and shader readback orientation handling for the
  validated CTS cases.
- Uniform arrays, boolean/integer uniforms, uniform-block metadata, packed
  depth/stencil transfer, PBO transfer, byte swapping, pixel-storage modes,
  texture arrays, depth/stencil formats, and framebuffer blits.
- Texture swizzle state and selected scalar/integer render-target formats.
- Explicit lifetime release for tracked Metal buffers and textures on GL
  deletion.

### Validation

The exact 108-case failure ledger is the focused regression suite and currently
passes **108/108 in one invocation**. It is not the entirety of the GL 3.3
work: the full VK-GL-CTS GL 3.3 tree exercised thousands of additional shader,
resource, texture, framebuffer, and API cases. The ledger covers the previously
observed transfer, packed pixel, PBO, depth/stencil, framebuffer-blit,
shader-array, primitive-restart, and readback failures.

A complete `KHR-GL33.*` run was started in a fresh WoW64 prefix. It has not yet
been accepted as a clean conformance run; the recorded complete diagnostic
snapshot is summarized above. Therefore this release does not claim complete
GL 3.3 Khronos conformance despite the passing focused ledger.

### Boundary

General format/target combinations, complete texture swizzle coverage,
full transform-feedback semantics, and broad uniform-block/shader-language
corner cases remain outside the claim unless separately validated.

## OpenGL 4.0

### Implemented

- Tessellation-control shader compilation through the translation toolchain.
- Basic tessellation evaluation shaders, triangle/quad factors, patch defaults,
  and limited draw paths.
- Limited geometry pass-through emulation used by the validated fixtures.

### Boundary

Full tessellation-control/evaluation semantics and general geometry shaders are
not implemented.

## OpenGL 4.1

### Implemented

- Separable shader programs and program-pipeline lifecycle.
- Stage-uniform binding and interface mismatch rejection.
- Selected DSA program and resource operations.

### Boundary

The full 4.1 feature and extension set is not implemented.

## OpenGL 4.2

### Implemented

- Selected image load/store paths, including RGBA8, R32UI, R32F, RGBA8UI,
  and RGBA16F 2D cases.
- Selected texture views and immutable-storage behavior.

### Boundary

Atomic-counter breadth, all image formats, and the full 4.2 extension surface
are not claimed.

## OpenGL 4.3

### Implemented

- Compute dispatch, indirect dispatch, SSBO ranges, image writes, and selected
  resource reflection.
- Multi-bind buffers, textures, samplers, and images for the tested paths.

### Boundary

The complete compute, shader-storage, image, debug, and extension matrix is
not implemented.

## OpenGL 4.4

### Implemented

- Selected persistent/coherent mapping behavior and bounded map/flush paths.
- Additional multi-bind and robustness entry points used by the fixtures.

### Boundary

No complete 4.4 conformance claim is made.

## OpenGL 4.5

### Implemented

- Selected direct-state-access texture/buffer/framebuffer entry points.
- Program/resource queries and bounded robustness helpers used by validation.

### Boundary

The full 4.5 DSA, robustness, and extension surface is not implemented.

## OpenGL 4.6

OpenGL 4.6 is **not implemented or claimed**. The project does not claim a
full Khronos OpenGL 4.6 run.

## Runtime and artifact boundary

- Only x86_64 macOS 15-or-newer host artifacts are produced.
- Windows x86_64 and i386 guests are supported together in a fresh WoW64
  prefix.
- The sidecar, Wine Unix drivers, and guest DLLs are ABI-coupled and must come
  from the same integrated Wine build. WineForge/Wine 11.17 is the tested reference, not a product dependency.
- Legacy Wine OpenGL compatibility remains available when
  `WINEMETALGL_EXPERIMENTAL=0`; Metal-owned paths require
  `WINEMETALGL=1 WINEMETALGL_EXPERIMENTAL=1`.
- Unsupported behavior is intentionally rejected or left to Wine's
  compatibility path; exported symbols are not treated as proof of support.
