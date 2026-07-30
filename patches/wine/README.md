# Wine patch series

`full-series/` contains 19 ordered `git format-patch` files. Apply it to
the Wine 11.12 tag to reproduce the custom architecture substrate and the
accepted OpenGL integration.

`opengl-only/` contains the final two OpenGL commits:

1. Multi-architecture Metal runtime integration.
2. Experimental Metal readback routing in `winemac`.

The OpenGL-only series assumes the exact prerequisite state through Wine
commit `0805c29`. Use the full series for a fresh tree.
