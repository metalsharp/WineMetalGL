# Wine 11.17 integration

The release target is x86_64 Unix Wine 11.17 with x86_64 and i386 guests in a
single WoW64 prefix. ARM host drivers, ARM guest images, ARM64EC, and
universal output are intentionally outside this tree.

The actual release build uses the WineForge 11.17 source at:

```text
/Volumes/AverySSD/Crossover-WineForge-macos15/merged-wine-11.17
```

The local source includes the CrossOver/WineForge integration and the
WineMetalGL changes in `dlls/win32u/opengl.c` and `dlls/winemac.drv/opengl.c`.
Those files are rebuilt and staged together with the sidecar; no installed
runtime is modified by this repository.

`wine-11.17/` contains small, reviewable integration deltas that can be
applied to a matching Wine 11.17 tree. The full WineForge/CrossOver source
integration is maintained in the separate build source because it includes
non-OpenGL runtime components.
