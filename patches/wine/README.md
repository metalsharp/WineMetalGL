# Wine OpenGL integration

This patch is the reference integration for x86_64 Unix Wine with x86_64 and
i386 guests in one WoW64 prefix. It is not a dependency on WineForge, CrossOver,
or MetalSharp. ARM host drivers, ARM guest images, ARM64EC, and universal
output are intentionally outside this adapter's x86_64 scope.

`wine-11.17/0001-winemetalgl-metal-surface-and-extension-parser.patch` records
the main OpenGL-specific delta used by the reference validation build. Apply
`0002-winemetalgl-default-config-and-feature-cap.patch` after it; the second
patch adds the bundled configuration loader and max-feature-level gate.
Wine 11.17 is the provided and tested baseline; other Wine revisions may
require a small rebase when Wine's OpenGL or macOS driver interfaces change.

The bundled `config/winemetalgl.conf` enables `WINEMETALGL=1` and
`WINEMETALGL_EXPERIMENTAL=1` by default and caps experimental OpenGL at 3.3.
`WINEMETALGL_CONFIG_FILE` overrides the file, while explicit environment
variables take precedence. `WINEMETALGL_GL40_COVERAGE=1` remains an opt-in
loader gate for the bounded GL 4.0 API-coverage run.
