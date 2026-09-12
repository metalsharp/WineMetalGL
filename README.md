# WineMetalGL

WineMetalGL is an x86_64-only macOS 15 Metal backend for Wine's OpenGL
surface. Windows x86_64 and i386 guests run together in one `WINEARCH=wow64`
prefix; no ARM or universal artifacts are produced.

## Implemented runtime

- x86_64 Mach-O sidecar, deployment target macOS 15.0.
- Wine `opengl32`/`winemac.drv` loading with an ABI-coupled sidecar.
- WGL context creation, pixel-format selection, current-context switching,
  context sharing at creation, and swap-interval plumbing through Wine.
- A real Wine-window `CAMetalLayer` bridge. Experimental draws acquire a
  `CAMetalDrawable`, render to its texture, present it on the Metal command
  buffer, and read back from that same texture.
- Deterministic off-screen fallback for contexts without a window surface.
- GLSL 3.30 and 4.50 vertex/fragment translation through glslang, SPIR-V,
  SPIRV-Cross, and MSL.
- Metal pipeline creation, viewport/scissor, blend/depth state, vertex
  buffers, indexed and instanced draws with 16/32-bit indices, common
  float/normalized attributes, sampler texture upload/sub-upload, color
  texture/renderbuffer FBOs, uniform slot storage, compute dispatch with
  SSBO/image writeback, uniform-buffer bindings, deterministic clear/readback,
  and synchronized flush/finish.
- Legacy GLSL 1.20 remains on Wine compatibility paths; experimental mode
  provides a Metal-backed immediate-mode triangle path.

The generated support manifest is `docs/api-coverage.json`. It deliberately
separates exported/forwarded entry points from operations owned by Metal and
from features still rejected, so parser acceptance is never reported as API
completion.

## Build

Requirements: macOS 15 or newer, Xcode Command Line Tools, CMake 3.24+, and
Ninja.

```sh
cmake --preset release
cmake --build --preset release -j8
./tests/native/probe-metalsharp-opengl.sh
```

The output is `build/release/metalsharp-opengl.dylib` and must report exactly
one Mach-O architecture: `x86_64`.

## Wine integration

The sidecar is loaded beside the x86_64 Unix `winemac.so`:

```sh
./scripts/stage-wine.sh /absolute/path/to/wine-build
```

The Wine source used for the release is WineForge Wine 11.17 with the
CrossOver integration applied by the MetalSharp build. The driver and sidecar
are one ABI unit; do not copy either into an unrelated Wine build.

Runtime controls:

- `WINEMETALGL=0` disables the sidecar.
- `WINEMETALGL=1` enables sidecar loading.
- `WINEMETALGL_EXPERIMENTAL=1` enables Metal-owned shader/resource draws.

## Wine probes

The end-to-end probe compiles both guest architectures, creates one fresh
WoW64 prefix, and runs the x86_64 and i386 fixtures sequentially:

```sh
./tests/wine/probe-opengl-wow64.sh
```

Override the runtime without changing the script:

```sh
WINEMETALGL_WINE_RUNTIME=/path/to/wine-install \
  ./tests/wine/probe-opengl-wow64.sh
```

Required markers include:

```text
WINEMETALGL_WOW64_X86_64_OK
WINEMETALGL_WOW64_I386_OK
WINEMETALGL_METAL_SURFACE_OK
WINEMETALGL_DEFAULT_FBO_PRESENT_OK
WINEMETALGL_READBACK_OK
WINEMETALGL_GLSL330_OK
WINEMETALGL_GL33_RESOURCES_OK
```

## Repository layout

- `src/`, `include/`: native sidecar and state/resource bridge.
- `vendor/`: pinned glslang and SPIRV-Cross sources.
- `patches/wine/wine-11.17/`: Wine 11.17 integration patches.
- `tests/native/`: host translation tests.
- `tests/wine/`: x86_64/i386 WoW64 fixtures and runners.
- `docs/`: machine-readable support and validation records.
- `artifacts/`: release-stage ABI-coupled binaries.

## License

The standalone source is MIT licensed. Wine integration remains LGPL-2.1;
third-party shader compiler licenses are retained under `LICENSES/`.
