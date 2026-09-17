# WineMetalGL staged artifacts

These artifacts are ABI-coupled to the integrated Wine build used by the
release probes. The reference build is WineForge/Wine 11.17. They target
x86_64 macOS 15-or-newer hosts and x86_64/i386 Windows
guests in one WoW64 prefix.

- `host-x86_64/metalsharp-opengl.dylib`: native Metal sidecar.
- `wine-driver/opengl32.so`: matching x86_64 Unix OpenGL client bridge.
- `wine-driver/winemac.so`: matching x86_64 Unix macOS driver.
- `wine-driver/win32u.so`: matching OpenGL extension-parser build.
- `guest/x86_64/opengl32.dll`: x86_64 guest OpenGL client.
- `guest/i386/opengl32.dll`: i386 guest OpenGL client.

See `../docs/conformance.md` for the feature-level coverage and validation
boundary. Use `scripts/stage-wine.sh` for a fresh Wine build instead of mixing
these files with an unrelated runtime. These files are inspection/reproduction
artifacts; no installed application runtime is modified by this repository.
