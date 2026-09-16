# WineMetalGL

WineMetalGL is a small, x86_64-only OpenGL-to-Metal sidecar for Wine on
macOS 15. It uses one matching Wine 11.17 WoW64 runtime for x86_64 and i386
Windows programs.

## Feature-level implementation

These are implementation-coverage estimates, not a claim that every Khronos
case passes. The detailed scope, tests, and unsupported behavior are in
[`docs/conformance.md`](docs/conformance.md).

| OpenGL level | Implementation | Short description |
|---|---:|---|
| 1.0 | 100% | Compatibility entry points and basic immediate mode |
| 1.1 | 100% | Textures, arrays, and legacy objects |
| 1.2 | 95% | Core legacy rendering and transfer paths |
| 1.3 | 95% | Multitexture and texture-coordinate paths |
| 1.4 | 90% | Blending, fog, point/line state, and legacy queries |
| 1.5 | 90% | Buffer objects, queries, and sync foundations |
| 2.0 | 90% | GLSL 1.10/1.20, shaders, FBO, and vertex attributes |
| 2.1 | 85% | Compatibility GLSL and fixed-function integration |
| 3.0 | 97% | Core resources, FBOs, instancing, textures, and GLSL 3.30 |
| 3.1 | 96% | UBOs, primitive restart, texture promotion, and indirect work |
| 3.2 | 80% | MSAA, geometry-adjacent paths, and expanded FBO behavior |
| 3.3 | 70% | GL 3.3 transfer/shader/resource paths; focused ledger is 108/108 |
| 4.0 | 45% | Tessellation compilation and limited draw support |
| 4.1 | 35% | Separable programs and pipeline lifecycle |
| 4.2 | 25% | Image, atomic, and expanded texture/resource subsets |
| 4.3 | 20% | Compute, SSBO, image, and reflection subsets |
| 4.4 | 15% | Limited newer resource and robustness entry points |
| 4.5 | 10% | Selected DSA and robustness entry points |
| 4.6 | 0% | Not implemented or claimed |

## Build

```sh
cmake --preset release
cmake --build --preset release -j8
```

The output is `build/release/metalsharp-opengl.dylib`. It must contain only
one architecture: `x86_64`.

## Wine

Use the sidecar only with the matching Wine 11.17 build. Do not mix its
`opengl32.dll`, Unix OpenGL/macOS drivers, or sidecar with another Wine build.

```sh
./scripts/stage-wine.sh /path/to/wine-build
./scripts/stage-host-libs.sh /path/to/wine-runtime
./scripts/stage-release.sh /path/to/wine-build /path/to/output
```

Enable the Metal-owned path with:

```sh
WINEMETALGL=1 WINEMETALGL_EXPERIMENTAL=1
```

The WoW64 probe is:

```sh
./tests/wine/probe-opengl-wow64.sh
```

## License

The standalone source is MIT licensed. Wine integration is LGPL-2.1;
third-party shader compiler licenses are in `LICENSES/`.
