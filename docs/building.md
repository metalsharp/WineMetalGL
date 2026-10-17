# Build it yourself

How to build the core and the Wine module, install them into a Wine, and
check that programs use them. The test suites are in [Test suite](testing.md);
release packages are in [Releasing](release.md).

## Requirements

- macOS 15 or later on Apple silicon, Xcode 16 or later, CMake 3.24 or later,
  Ninja and Python 3.
- Rosetta 2, to run an x86_64 Wine.
- For the Wine module: [llvm-mingw](https://github.com/mstorsjo/llvm-mingw)
  (`x86_64-w64-mingw32-clang`, `i686-w64-mingw32-clang` on `PATH`) and the
  headers of a Wine 11.x (an installation or its source tree).
- For the guest test programs: Homebrew `mingw-w64`
  (`x86_64-w64-mingw32-gcc`, `i686-w64-mingw32-gcc`).

glslang and SPIRV-Cross are vendored at pinned revisions
([`DEPENDENCIES.md`](../DEPENDENCIES.md)), so the build fetches nothing. The
core links only Apple system frameworks and never links OpenGL.framework.

## The core

```sh
./scripts/build.sh            # build/release/libwinemetalgl.dylib (x86_64, macOS 15)
```

That is the core for an x86_64 Wine under Rosetta. `scripts/stage-arm64-core.sh
/path/to/output` builds the arm64 core and stages it with the driver ABI
header; the Wine module is built for x86_64 Wine only, so the arm64 core has
not yet run inside an arm64 Wine.

For development, the core and its tests also build for the machine itself,
without Rosetta:

```sh
cmake -S . -B build/host -DWMGL_HOST_ONLY=ON
cmake --build build/host
build/host/wmgl_host_tests && build/host/wmgl_gpu_tests
```

## The Wine module

WineMetalGL reaches Windows programs through a Wine module of its own, built
outside Wine and installed next to Wine's modules: no Wine source is patched
or rebuilt. It works with any x86_64 Wine 11.x on macOS 15 or later, such as
the WineHQ release or MetalSharp's Wine 11.17; it is tested on stock Wine 11.0,
WineHQ's macOS builds of Wine 11.0 and 11.18, and MetalSharp's Wine 11.17.

| File | Where | What |
|---|---|---|
| `winemetalgl.dll` | `lib/wine/x86_64-windows`, `lib/wine/i386-windows`, and the prefix's `system32` and `syswow64` | WGL and the GL entry points, for 64- and 32-bit programs |
| `winemetalgl.so` | `lib/wine/x86_64-unix` | its unix side, which runs both (32-bit calls through Wine's WoW64 thunks) |
| `libwinemetalgl.dylib` | `lib/wine/x86_64-unix` | the core |
| `opengl32.dll` | the prefix's `system32` and `syswow64` | forwards every export to `winemetalgl.dll` |

```sh
./scripts/build.sh                                      # the core
./scripts/build-wine-module.sh /path/to/wine            # build/wine-module
./scripts/install-wine-module.sh /path/to/wine /path/to/prefix
```

`install-wine-module.sh` adds the module and the core to the Wine installation
(the directory with `bin/wine`, or a Wine app such as WineHQ's `Wine
Stable.app`) without replacing any of Wine's files. Given a prefix, it also puts the
forwarding `opengl32.dll` into it, with a copy of `winemetalgl.dll` for Wines
whose loader does not look in `lib/wine` for it, and sets the DLL override
`opengl32=native`; run it once per prefix (a prefix that does not exist yet
is created). If macOS refuses the copy (`Operation not permitted`), the
terminal needs App Management permission (System Settings, Privacy &
Security). Removing the override (or running
a program with `WINEDLLOVERRIDES=opengl32=b`) gives that prefix Wine's own
OpenGL back.

### CrossOver

CrossOver.app is signed and macOS does not let it be changed (`cp: ...:
Operation not permitted`), so `install-wine-module.sh` refuses it; use
`install-crossover.sh` instead, once per bottle:

```sh
./scripts/install-crossover.sh "My Bottle"                 # a bottle's name, or its directory
./scripts/install-crossover.sh "My Bottle" /path/to/CrossOver.app
```

Nothing goes into CrossOver.app. The module and the core go into
`~/Library/Application Support/WineMetalGL/lib/wine` (`WINEMETALGL_HOME`
chooses another directory), and the bottle's `cxbottle.conf` adds that
directory to its `[Wine] DllPath`, so CrossOver's Wine finds them there. The
bottle gets the forwarding `opengl32.dll` and `winemetalgl.dll` (in
`system32`, and `syswow64` for a 64-bit bottle) and the `opengl32=native`
override. CrossOver's `wine` script ignores `WINEPREFIX` and
`WINEDLLOVERRIDES`; removing the override in the bottle's Wine configuration
(Libraries) gives the bottle CrossOver's OpenGL back, and
`wine --bottle "My Bottle" --dll opengl32=b` does so for one run. It is
tested on CrossOver 26.3, in 64- and 32-bit bottles. CrossOver 27 runs an
arm64 Wine, which the module is not built for yet (issue #13).

What matters:

- The Wine must be an x86_64 build (run under Rosetta) with i386 and x86_64
  guests (`--enable-archs=i386,x86_64`), so one WoW64 prefix runs 32- and
  64-bit programs.
- The module loads the core from its own directory; it reports an error on
  stderr and offers no pixel formats when the core cannot be loaded.
- Windows appear through Wine's macOS driver: the module puts a Metal layer
  over the window's client area, in Wine's Cocoa window.
- Wine's optional features need x86_64 builds of their libraries, such as
  FreeType for fonts. On Apple silicon, Homebrew installs arm64 libraries,
  which an x86_64 Wine cannot load;
  `WINEMETALGL_HOST_LIB_DIR=/path/to/x86_64/libs scripts/stage-host-libs.sh /path/to/wine`
  copies an x86_64 FreeType closure into a Wine installation.

## Checking that a program uses the core

Look at a program's `GL_RENDERER` string: it ends in `(WineMetalGL)`, for
example `Apple M4 (WineMetalGL)`. With `WINEMETALGL_LOG=2` the core prints
`winemetalgl:info: WineMetalGL core on <GPU>` to stderr when it loads. If the
renderer is anything else, the prefix has no `opengl32.dll` from the module or
no `opengl32=native` override, and the program got Wine's own OpenGL.

## Environment

| Variable | Effect |
|---|---|
| `WINEDLLOVERRIDES=opengl32=b` | Use Wine's own OpenGL instead of WineMetalGL (how reference frames are recorded) |
| `WINEMETALGL_LOG=1..3` | Log warnings, information or traces, everywhere or per category (`shader:3,gpu:2`); see [Logging](logging.md). Errors go to stderr and warnings to a log file in `~/Library/Logs/WineMetalGL` by default. Warnings include each GL error the first time a command raises it, with the call's arguments |
| `WINEMETALGL_LOG_FILE=<path>` | Write the log file there (a directory, or a file where `%p` is the process ID); `0` writes none |
| `WINEMETALGL_LOG_MAX_MB=<n>` | Stop the log file at `n` MiB (default 64) |
| `WINEMETALGL_CRASH_HANDLER=0` | Do not write crash reports (they go to the log file, with the recent history, when WineMetalGL's code faults or the process aborts) |
| `WINEMETALGL_TRACE=1` | Log every GL call |
| `WINEMETALGL_STATS=1` | Log per-frame statistics: frame and GL time, draws, passes and why they ended, command buffers, pipeline compiles, CPU conversions, readbacks, indirect draw commands read by the GPU or (after a wait) the CPU, and why the CPU reads them (each reason described once per program), and the memory the Metal device has allocated (growing while a scene stands still means GPU resources are retained); every 60 frames, the costliest commands |
| `WINEMETALGL_BATCH=0` | Make every GL call its own unix call (by default the module batches calls that return nothing and take values, and runs them before the next other call) |
| `WINEMETALGL_COMMIT_DRAWS=<n>` | Commit the recorded work to the GPU once it holds `n` draws (default 256), so the GPU runs a frame's work while the CPU records the rest: where a draw starts a render pass, or within one past `8 × n` draws (a commit ends the pass); 0 commits only when a GL call needs the GPU's results and at SwapBuffers |
| `WINEMETALGL_METAL_TESSELLATION=0` | Tessellate every patch on the CPU, after reading the control stage's levels back, instead of drawing rasterized triangle and quad patches with Metal's tessellator |
| `WINEMETALGL_FLAT_ANALYSIS=0` | Rotate the indices of every draw with flat fragment inputs and the last-vertex convention (reading indirect commands on the CPU), instead of only those whose flat values can differ between a primitive's vertices |
| `WINEMETALGL_SAMPLER_LOD_BIAS=0` | Filter lookups through samplers with a LOD bias in the shader, as on macOS 15 where Metal's samplers have no LOD bias (for testing that path) |
| `WINEMETALGL_VERTEX_OFFSETS16=1` | Draw with vertex starts and base vertices past 16 bits through CPU-written indices, as on macOS virtual machines' GPU (for testing that path) |
| `WINEMETALGL_PREWARM=0` | Compile a program's Metal functions at its first draw instead of in the background after it links |
| `WINEMETALGL_SAFE_MATH=1` | Compile shaders with Metal's safe math instead of relaxed math (relaxed allows optimizations such as reassociation that can change results slightly, and keeps infinities and NaN; safe allows none) |
| `WINEMETALGL_METALFX=0`, `=<factor>` | Turn off, or set the factor of, MetalFX spatial upscaling of window back buffers. Unset, WineMetalGL follows DXMT's setting: off when `DXMT_METALFX_SPATIAL_SWAPCHAIN=0`, else `d3d11.metalSpatialUpscaleFactor` from `DXMT_CONFIG`, `DXMT_CONFIG_FILE` or the Wine installation's `etc/dxmt.conf` (2 when only `DXMT_METALFX_SPATIAL_SWAPCHAIN=1` is set). Factors above 1 (at most 4) present a back buffer through the MetalFX spatial scaler into a drawable that much larger, on GPUs MetalFX supports |
| `WINEMETALGL_WINE_DEBUG=1` | Log what the Wine module does (contexts, surfaces, windows): `WINEMETALGL_LOG=wine:3,pe:3` |
| `WINEMETALGL_VENDOR`, `WINEMETALGL_RENDERER` | Replace the `GL_VENDOR` and `GL_RENDERER` strings, for games that refuse unknown vendors |
