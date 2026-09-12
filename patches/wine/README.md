# Wine 11.17 integration

The release target is x86_64 Unix Wine 11.17 with x86_64 and i386 guests in a
single WoW64 prefix. ARM host drivers, ARM guest images, ARM64EC, and
universal output are intentionally outside this tree.

The actual release build uses the WineForge 11.17 source at:

```text
/Volumes/AverySSD/Crossover-WineForge-macos15/merged-wine-11.17
```

`wine-11.17/0001-winemetalgl-metal-surface-and-extension-parser.patch` records
the OpenGL-specific delta used by the verified build. It includes the
extension parser correction, CAMetalLayer bridge, sidecar core-function
binding, modern WGL lookup gates, and the Metal presentation path.

The full WineForge/CrossOver source integration is maintained in the separate
build source because it includes non-OpenGL runtime components. Apply this
delta only to the matching WineForge/Wine 11.17 source revision.
