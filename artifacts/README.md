# Accepted binary snapshot

These files are the binaries used by the accepted VKMT OpenGL matrix:

- `host-arm64/`: native Apple Silicon sidecar and Wine Unix thunk.
- `guest/aarch64/`, `guest/x86_64/`, `guest/i386/`: Wine PE `opengl32.dll`.
- `wine-driver/`: accepted native ARM64 host driver plus architecture-facing
  Wine driver modules.

ARM64EC shares Wine's ARM64/ARM64X host driver surface, so there is no
separate ARM64EC Mach-O driver.

All host Mach-O libraries in this snapshot must report `arm64`. Do not mix
the snapshot with an ABI-incompatible Wine tree.
