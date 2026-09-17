# WineMetalGL validation results

Validation snapshot: 2026-09-16

This is a bounded validation record for the x86_64 macOS 15-or-newer sidecar
and a compatible Wine WoW64 runtime. The reference build used WineForge/Wine
11.17; the adapter is not tied to that distribution. It is not a claim of complete
OpenGL 4.6 or complete Khronos GL 3.3 conformance. See
[`../conformance.md`](../conformance.md) for the feature-level matrix.

## Host and runtime gates

- Host: macOS 15 or newer, reference machine Apple M4.
- Host artifact: one x86_64 Mach-O architecture; deployment target 15.0.
- Guest artifacts: matching x86_64 and i386 `opengl32.dll` files.
- Prefix: fresh `WINEARCH=wow64`.
- Runtime: compatible Wine build; reference runtime WineForge/Wine 11.17.
- GLSL 3.30 translation path: glslang -> SPIR-V -> SPIRV-Cross -> MSL.
- GLSL 4.50 translation is available for the declared experimental subset.

The x86_64 and i386 Wine fixtures cover WGL creation/current-context state,
sharing, swap interval, Metal surface presentation, legacy GLSL, GLSL 3.30,
textures, FBOs, buffers, uniforms, compute/image paths, tessellation
compilation, fixed-function paths, queries, synchronization, and readback.

## Focused GL 3.3 regression ledger

The ledger is `tests/cts/gl33-failure-ledger.txt` and is executed by
`scripts/run-gl33-failure-ledger.sh`. It deliberately runs all listed cases in
one `glcts.exe` invocation and validates every expected QPA result.

```text
ledger cases: 108
Pass:          108
Fail:            0
NotSupported:   0
Missing:        0
```

This is the acceptance result for the previously observed transfer, packed
pixel, PBO, depth/stencil, framebuffer-blit, shader-array, primitive-restart,
and readback regressions.

## Targeted Khronos evidence

Using a locally built `glcts.exe` from a VK-GL-CTS checkout (set the checkout
path explicitly; no machine-specific CTS path is required). The GL 3.3 tree
contains thousands of shader, resource, texture, framebuffer, and API cases:

| Test selection | Result |
|---|---|
| `KHR-GL30.texture_repeat_mode.*` | 162/162 Pass |
| `KHR-GL30.texture_lod_basic.*` | 1/1 Pass |
| `KHR-GL30.texture_lod_bias.*` | 1/1 Pass |
| `KHR-GL30.framebuffer_blit.*` | 3/3 Pass |
| `KHR-GL30.shaders30.*` | 651/651 Pass |
| `KHR-GL30.buffer_objects.*` | 5/5 Pass |
| `KHR-GL30.transform_feedback.*` | 21/21 Pass |
| `KHR-GL31.*` | 856/889 Pass; 33 explicitly NotSupported |
| `KHR-GL33.shaders.indexing.*` | 297/297 Pass |
| `KHR-GL33.shaders30.glsl_constructors.*` | 648/648 Pass |
| GL 3.3 failure ledger | 108/108 Pass |

The last complete broad `KHR-GL33.*` diagnostic snapshot traversed 9,887
cases, including thousands of shader compilations and GL 3.3 resource,
texture, framebuffer, and API cases: 6,441 Pass, 703 NotSupported, 2,734 Fail,
and 9 InternalError. A subsequent run was interrupted before a newer complete
result was available. The 108-case ledger is a focused regression subset, not
a summary of all GL 3.3 work. These results are retained for engineering
follow-up and are not presented as a clean conformance result.

## Bounded GL 4.0 evidence

This is a separate engineering result and does not expand the release claim.
With `WINEMETALGL_GL40_COVERAGE=1`, the exact first 1,000 cases from the
11,863-case `KHR-GL40` caselist were run in one dEQP invocation using a trie
caselist. The result was:

```text
cases:         1000
Pass:            30
Fail:            58
NotSupported:   912
InternalError:    0
```

The 58 failures are in `KHR-GL40.draw_indirect.*`; the unsupported results are
preserved as `NotSupported`. The standalone `KHR-GL40.api.coverage` case also
passed (`1/1`). Reproduce the single-invocation run with
`scripts/run-gl40-bounded.sh`; set `WINEMETALGL_HOST_LIB_DIR` when the runtime
needs an external x86_64 EGL dependency. This bounded run is not a claim of
GL 4.0 conformance, complete GL 3.3 conformance, or GL 4.6 support.

## Implemented feature groups

The validated implementation includes the following declared groups:

- WGL context creation, pixel formats, current-context switching, sharing,
  swap interval, and CAMetalLayer-backed presentation.
- GLSL 1.10/1.20 and GLSL 3.30 vertex/fragment translation, with a GLSL 4.50
  experimental subset.
- VBO/IBO/VAO, indexed/instanced/base-vertex/indirect/multi-draw paths.
- Uniforms, uniform arrays, selected uniform blocks, reflection, program
  binaries, SPIR-V ingestion, and shader specialization.
- 1D/2D/3D/array/rectangle/cube and selected multisample texture paths;
  normalized, half-float, integer, depth/stencil, packed, PBO, swizzle,
  alignment, subimage, copy, and readback handling.
- Color/depth/stencil FBOs, renderbuffers, array layers, clear/readback,
  completeness, blit/resolve, invalidate, and multisample paths.
- Selected compute/SSBO/image-store, transform-feedback, tessellation,
  geometry pass-through, fixed-function, query, and synchronization paths.

## Explicit boundary

The release does not claim complete Khronos OpenGL 4.6 conformance. Geometry
shader semantics, full tessellation-control/evaluation semantics, full
transform feedback, the complete texture/sampler/FBO format matrix, all image
formats, vendor extensions, and unresolved GL 3.3 CTS groups remain outside
the claim. The sidecar intentionally preserves accurate unsupported behavior.
