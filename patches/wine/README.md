# Wine OpenGL integration

This patch is the reference integration for x86_64 Unix Wine with x86_64 and
i386 guests in one WoW64 prefix. It is not a dependency on WineForge, CrossOver,
or MetalSharp. ARM host drivers, ARM guest images, ARM64EC, and universal
output are intentionally outside this adapter's x86_64 scope.

`wine-11.17/0001-winemetalgl-metal-surface-and-extension-parser.patch` records
the OpenGL-specific delta used by the reference validation build. It includes
the extension parser correction, CAMetalLayer bridge, sidecar core-function
binding, modern WGL lookup gates, and the Metal presentation path. The
`WINEMETALGL_GL40_COVERAGE=1` opt-in additionally enables the extension gates
needed by the bounded GL 4.0 API-coverage loader; it is not part of the GL
3.1-3.3 release evidence.

Apply the patch to the corresponding Wine source revision. Wine 11.17 is the
provided and tested baseline; other Wine revisions may require a small rebase
when Wine's OpenGL or macOS driver interfaces change.
