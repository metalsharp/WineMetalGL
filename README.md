# WineMetalGL

WineMetalGL is the standalone OpenGL-to-Metal layer proven by the VKMT
runtime. It combines Wine's multi-architecture `opengl32` path with a native
Apple Silicon sidecar that translates modern GLSL through SPIR-V and Metal
Shading Language.

## Status

The complete VKMT OpenGL acceptance matrix passes: ARM64, ARM64EC, x86_64,
and i386/WoW64 all load and execute in one Wine prefix. The accepted gates
cover DLL loading, WGL context creation, deterministic clear/readback,
GLSL 1.20 rendering, and opt-in GLSL 3.30 and 4.50 translation and Metal
draw/readback. In that precise sense, the project is 100% passing.

This is not a claim of Khronos conformance for every OpenGL 3.x/4.x entry
point. It is a statement that every gate in the project's published,
multi-architecture acceptance matrix passed. See
[`docs/validation/VKMT-RESULTS-20260728.md`](docs/validation/VKMT-RESULTS-20260728.md).

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
- i386 pointer thunks use VKMT's canonical guest-memory conversion rather
  than assuming a low 4-GiB host mapping.
- A runtime switch can disable the sidecar or enable the experimental modern
  Metal path.

The current acceptance suite does not separately claim complete coverage for
indexed drawing, arbitrary vertex layouts, all uniform and texture forms,
visible-window presentation, or the entire GL 3.x/4.x API surface.

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

The x86_64 and i386 Windows guests are executed by VKMT's custom FEX/WoW64
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

The exact tested integration is based on Wine 11.12 plus VKMT's custom
ARM64EC, FEX/WoW64, MSync, SDL, and OpenGL work.

For a clean reproduction from the Wine 11.12 tag:

```sh
git switch --create winemetalgl wine-11.12
git am /path/to/WineMetalGL/patches/wine/full-vkmt-series/*.patch
```

`patches/wine/opengl-only/` contains only the final two OpenGL commits. Use
that smaller series only on a source tree already matching VKMT through
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
- `VKMT_OPENGL_METAL=0`: disable the Metal sidecar and use the legacy path.
- `VKMT_OPENGL_METAL_EXPERIMENTAL=1`: enable the validated modern
  GLSL/SPIR-V/MSL/Metal draw and readback path.

Example:

```sh
VKMT_OPENGL_METAL_EXPERIMENTAL=1 wine your-program.exe
```

## Tests

`scripts/probe-native.sh` builds and runs the self-contained GLSL translation
probe against the newly built sidecar. The `tests/wine/` fixtures are the C
sources used by the VKMT single-prefix matrix.

The two scripts whose names end in `-vkmt.sh` are preserved provenance
runners. They expect the original VKMT directory layout and are included so
the accepted result can be audited rather than paraphrased.

## Repository layout

- `src/`, `include/`: standalone Metal/OpenGL implementation.
- `vendor/`: pinned glslang and SPIRV-Cross sources.
- `patches/wine/`: full reproducible Wine series and OpenGL-only tail.
- `tests/native/`: standalone native translation probe.
- `tests/wine/`: Windows/Wine acceptance fixtures.
- `artifacts/`: accepted host and guest binaries.
- `docs/validation/`: captured VKMT acceptance report.
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
