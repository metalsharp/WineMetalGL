# WineMetalGL

WineMetalGL is a standalone OpenGL-to-Metal layer for Wine on Apple Silicon.
It combines Wine's multi-architecture `opengl32` path with a native
Apple Silicon sidecar that translates modern GLSL through SPIR-V and Metal
Shading Language.

## Status

The complete WineMetalGL acceptance matrix passes: ARM64, ARM64EC, x86_64,
and i386/WoW64 all load and execute in one Wine prefix. The accepted gates
cover DLL loading, WGL context creation, deterministic clear/readback,
GLSL 1.20 rendering, and opt-in GLSL 3.30 and 4.50 translation and Metal
draw/readback. In that precise sense, the project is 100% passing.

This is not a claim of Khronos conformance for every OpenGL 3.x/4.x entry
point. It is a statement that every gate in the project's published,
multi-architecture acceptance matrix passed. See
[`docs/validation/RESULTS.md`](docs/validation/RESULTS.md).

## What works

- One prefix supports Windows ARM64, ARM64EC, x86_64, and i386/WoW64 clients.
- The host-side Wine and Metal libraries are ARM64 Mach-O; Rosetta is not
  required.
- `opengl32.dll` loading and core WGL exports.
- Hidden-window pixel format and WGL context creation.
- OpenGL 2.1 / GLSL 1.20 compatibility rendering through Apple's OpenGL
  framework.
- GLSL 3.30 and 4.50 compilation through glslang to SPIR-V.
- SPIR-V translation to Metal Shading Language through SPIRV-Cross.
- Native Metal pipeline creation, fullscreen-triangle draw, staging-buffer
  readback, and deterministic pixel validation.
- i386 pointer thunks use the packaged canonical guest-memory conversion
  than assuming a low 4-GiB host mapping.
- A runtime switch can disable the sidecar or enable the experimental modern
  Metal path.

## OpenGL support by version

WineMetalGL has two execution paths:

1. The **compatibility path** forwards implemented entry points to Apple's
   native OpenGL framework.
2. The **Metal path** captures modern shaders and selected draw state,
   compiles GLSL with glslang, translates SPIR-V to MSL with SPIRV-Cross,
   and submits the resulting pipeline to Metal.

“Exported” below means WineMetalGL provides the named entry points.
“Translated” means the operation is owned by the Metal path rather than
only forwarded to Apple's OpenGL implementation.

| OpenGL version | Exported or forwarded support | Explicit Metal translation |
| --- | --- | --- |
| **1.0** | Immediate mode (`glBegin`/`glEnd`, vertex, color, normal and texture-coordinate calls); clear, viewport, enable/disable, blending, depth, stencil, rasterization and color masks; matrix stack and transforms; lighting, materials, fog, alpha test and clip planes; pixel-store/read/draw-buffer operations; display lists; state/error/string queries; flush and finish. | Viewport is mirrored into Metal state. RGBA8 readback can be serviced from the active Metal render target. Other 1.0 fixed-function operations use the compatibility path. |
| **1.1** | Texture object creation, deletion, binding, parameters, 2D upload/sub-upload/copy, texture queries and `glIsTexture`; client vertex arrays; `glDrawArrays` and `glDrawElements`. | `glDrawArrays` is translated for a Metal-owned program. Points, lines, line strips, triangles and triangle strips map to corresponding Metal primitive types. `glDrawElements` remains compatibility-path only. |
| **1.2** | Core 1.2 calls already represented by the exported 1.x state, pixel and texture surface continue through the compatibility path. | No 1.2-specific Metal-owned entry points are claimed. |
| **1.3** | Active texture selection and multitexture-era texture state through `glActiveTexture` and the exported texture API. | The renderer has native BGRA8 2D texture creation and fragment-texture binding primitives; the complete OpenGL texture-state bridge is not yet claimed. |
| **1.4** | Blend color, blend equation, separate blend factors, point size, polygon offset and related state exposed by the shim. Multi-draw entry points are not exported. | Blend enable is carried into Metal pipeline creation. The current Metal pipeline uses one/zero blend factors; complete OpenGL blend-factor/equation mapping is not claimed. |
| **1.5** | Buffer objects: generate/delete/bind, data/sub-data, map/unmap and object query. | Array-buffer and element-array-buffer bindings are tracked. The Metal renderer supports shared native buffers and vertex-buffer binding, but the full OpenGL buffer lifecycle is not yet translated. |
| **2.0** | Vertex/fragment shader and program lifecycle; source, compile, attach/detach, link, validate, bind and status/info-log queries; attributes; scalar/vector integer and float uniforms; 2x2, 3x3 and 4x4 matrix uniforms; active uniform/attribute queries; separate stencil and blend operations. | Vertex and fragment GLSL are compiled to SPIR-V, translated to MSL, cached by source/stage, compiled into Metal libraries and linked into a Metal render pipeline. Program status and logs are returned from the translated path. `glDrawArrays` submits that pipeline. General uniform replay is still compatibility-path only. |
| **2.1** | The complete accepted legacy shader gate uses GLSL 1.20 and program rendering through the compatibility path. Non-square matrix entry points are not exported by this release. | No additional 2.1-specific Metal operation beyond the common shader pipeline is claimed. |
| **3.0** | Framebuffer and renderbuffer create/delete/bind/storage/attachment/status/query operations; indexed extension-string query through `glGetStringi`; vertex-attribute divisor export. EXT framebuffer aliases are available through Wine's `wglGetProcAddress` routing. | The Metal renderer creates an offscreen BGRA8 render target, command buffer and render encoder, then performs deterministic staging-buffer readback. General OpenGL FBO attachment-to-Metal render-pass translation is not yet claimed. |
| **3.1** | Compatible shader sources are accepted by glslang, but the complete 3.1 core API—uniform blocks, texture buffers, instanced draw entry points and primitive restart—is not exported as a version-complete surface. | No 3.1-specific draw or resource gate is claimed. |
| **3.2** | `wglCreateContextAttribsARB` can request a macOS OpenGL 3.2 Core context. Geometry-shader source can be parsed to SPIR-V, but geometry is deliberately rejected by the MSL translation stage. | Vertex and fragment stages only for graphics. Geometry and tessellation execution are not supported by the Metal bridge. |
| **3.3** | GLSL 3.30 vertex and fragment programs are accepted. Attribute divisors are exported through the compatibility surface. | **Validated:** GLSL 3.30 → SPIR-V 1.0 → MSL 2.4, Metal pipeline creation, fullscreen-triangle `glDrawArrays`, offscreen BGRA8 render and deterministic RGBA8 readback. |
| **4.0–4.4** | glslang can parse compatible desktop GLSL sources in this range. The complete version-specific APIs for tessellation, indirect draw, image load/store, atomic counters, program pipelines, compute dispatch and shader-storage buffers are not exported as a complete surface. | Vertex/fragment translation uses the common SPIR-V/MSL path. Tessellation and geometry are rejected. Compute can be translated by the compiler library, but is not connected to an OpenGL `glDispatchCompute` execution path. |
| **4.5** | GLSL 4.50 vertex and fragment programs are accepted. Direct-state-access and the rest of the full 4.5 core command surface are not exported. | **Validated:** GLSL 4.50 → SPIR-V 1.0 → MSL 2.4, Metal pipeline creation, fullscreen-triangle draw, offscreen render and deterministic readback. |
| **4.6** | A compatible shader may parse when it uses constructs accepted by the pinned compiler, but WineMetalGL does not advertise or claim a complete OpenGL 4.6 implementation. | No separate 4.6 acceptance gate or SPIR-V 1.6 path is claimed. |

### Shader-stage support

| Stage | GLSL → SPIR-V | SPIR-V → MSL | OpenGL execution |
| --- | --- | --- | --- |
| Vertex | Yes | Yes | Yes, validated |
| Fragment | Yes | Yes | Yes, validated |
| Compute | Yes | Yes | Compiler-library support only; no `glDispatchCompute` bridge |
| Geometry | Yes | Rejected explicitly | No |
| Tessellation control/evaluation | Yes | Rejected explicitly | No |
| Mesh/task | Parser mapping exists | Rejected explicitly | No |
| Ray tracing stages | Parser mapping exists | Rejected explicitly | No |

### WGL support

- `wglCreateContext`, `wglMakeCurrent`, and `wglDeleteContext`
- `wglGetProcAddress`
- `wglCreateContextAttribsARB`
- `wglSwapIntervalEXT`
- single-format `wglChoosePixelFormat` and `wglDescribePixelFormat`
- context sharing at creation time
- post-creation `wglShareLists` deliberately returns false

The accepted graphics matrix proves these paths across every packaged guest
architecture. It is not a Khronos CTS result or a claim that every function
from every listed OpenGL version is implemented.

## Architecture

```text
Windows application (ARM64 / ARM64EC / x86_64 / i386)
  -> Wine PE opengl32.dll
  -> Wine ARM64 opengl32.so Unix thunks
  -> Wine ARM64 winemac.so
  -> metalsharp-opengl.dylib
       -> Apple OpenGL (legacy compatibility path)
       -> glslang -> SPIR-V -> SPIRV-Cross -> MSL -> Metal (modern path)
```

The x86_64 and i386 Windows guests are executed by the packaged FEX/WoW64
provider. They do not introduce x86 Mach-O libraries into the host process.
ARM64EC uses Wine's ARM64/ARM64X host driver surface; it does not need a
separate ARM64EC Mach-O `winemac` binary.

## Build the native sidecar

Requirements:

- Apple Silicon Mac
- Xcode or Xcode Command Line Tools
- CMake 3.24 or newer
- Ninja

glslang and SPIRV-Cross are pinned and included under `vendor/`.

```sh
cmake --preset release
cmake --build --preset release
./scripts/probe-native.sh
```

The resulting library is:

```text
build/release/metalsharp-opengl.dylib
```

It is intentionally ARM64-only. The guest architecture conversion belongs
at the Wine/FEX boundary, not inside the Metal sidecar.

## Integrate with Wine

The exact tested integration is based on Wine 11.12 plus the packaged
ARM64EC, FEX/WoW64, MSync, SDL, and OpenGL work.

For a clean reproduction from the Wine 11.12 tag:

```sh
git switch --create winemetalgl wine-11.12
git am /path/to/WineMetalGL/patches/wine/full-series/*.patch
```

`patches/wine/opengl-only/` contains only the final two OpenGL commits. Use
that smaller series only on a source tree already matching the prerequisite series through
commit `0805c29`; it is not a standalone upstream-Wine patch set.

Configure and build the Wine tree using the same architecture layout as your
runtime. When updating an existing compatible build, rebuild only the
affected targets:

```sh
make -C dlls/opengl32
make -C dlls/winemac.drv
```

Then stage the sidecar beside `winemac.so`:

```sh
./scripts/stage-wine.sh /absolute/path/to/wine-build
```

The `artifacts/` directory contains the exact accepted snapshot. These
binaries are useful for inspection and reproduction, but are ABI-coupled to
the packaged Wine patch series and should not be dropped into an unrelated
Wine build.

## Runtime controls

- Default: `winemac.so` attempts to load
  `@loader_path/metalsharp-opengl.dylib`.
- `WINEMETALGL=0`: disable the Metal sidecar and use the legacy path.
- `WINEMETALGL_EXPERIMENTAL=1`: enable the validated modern
  GLSL/SPIR-V/MSL/Metal draw and readback path.

Example:

```sh
WINEMETALGL_EXPERIMENTAL=1 wine your-program.exe
```

## Tests

`scripts/probe-native.sh` builds and runs the self-contained GLSL translation
probe against the newly built sidecar. The `tests/wine/` fixtures are the C
sources used by the single-prefix architecture matrix. Historical provenance
runners are retained under `tests/` so the accepted result can be audited.

## Repository layout

- `src/`, `include/`: standalone Metal/OpenGL implementation.
- `vendor/`: pinned glslang and SPIRV-Cross sources.
- `patches/wine/`: full reproducible Wine series and OpenGL-only tail.
- `tests/native/`: standalone native translation probe.
- `tests/wine/`: Windows/Wine acceptance fixtures.
- `artifacts/`: accepted host and guest binaries.
- `docs/validation/`: captured WineMetalGL acceptance report.
- `LICENSES/`: third-party license texts.

## Provenance

- WineMetalGL/MetalSharp source: `24c57bc18ef619624ffb4bf73418f818449c5fcd`
- SPIRV-Cross: `bccaa94db814af33d8ef05c153e7c34d8bd4d685`
- glslang: `46ef757e048e760b46601e6e77ae0cb72c97bd2f`
- Wine base: Wine 11.12 (`996020f`)
- Accepted Wine OpenGL integration: `f0a1f2b`

## License

The standalone WineMetalGL source is MIT licensed; see `LICENSE`. Wine
patches remain under Wine's LGPL-2.1 terms. glslang and SPIRV-Cross retain
their upstream licenses in `LICENSES/`.
