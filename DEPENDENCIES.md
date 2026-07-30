# Pinned dependencies

The package vendors the source required to build without fetching during
configuration:

| Component | Revision | Purpose |
| --- | --- | --- |
| SPIRV-Cross | `bccaa94db814af33d8ef05c153e7c34d8bd4d685` | SPIR-V to GLSL/MSL translation |
| glslang | `46ef757e048e760b46601e6e77ae0cb72c97bd2f` | GLSL parsing and SPIR-V generation |

The built dylib dynamically links only Apple system frameworks and libraries:
Metal, OpenGL, Foundation, QuartzCore, AppKit, CoreFoundation, libc++,
libSystem, and libobjc.

Wine integration requires the Wine patch series in `patches/wine/`. The
prebuilt Wine artifacts are an accepted snapshot, not a substitute for
applying those patches to a compatible source tree.
