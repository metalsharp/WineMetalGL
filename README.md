<h1 align="center">WineMetalGL</h1>

<p align="center">
  <a href="https://github.com/metalsharp/WineMetalGL/actions/workflows/metal.yml"><img alt="Metal" src="https://img.shields.io/github/actions/workflow/status/metalsharp/WineMetalGL/metal.yml?branch=main&label=Metal&color=purple"></a>
  <a href="https://github.com/metalsharp/WineMetalGL/actions/workflows/build.yml"><img alt="WineMetalGL build" src="https://img.shields.io/github/actions/workflow/status/metalsharp/WineMetalGL/build.yml?branch=main&label=WineMetalGL&color=lightblue"></a>
  <a href="https://github.com/metalsharp/WineMetalGL/actions/workflows/codeql.yml"><img alt="CodeQL" src="https://img.shields.io/github/actions/workflow/status/metalsharp/WineMetalGL/codeql.yml?branch=main&label=CodeQL&color=white"></a>
</p>

WineMetalGL is an OpenGL implementation on Metal for Wine on macOS 15 or newer. It installs into an unmodified Wine 11.x as a Wine module of its own, and gives Windows programs OpenGL 4.6 contexts, core or compatibility profile, instead of Apple's OpenGL: It runs x86_64 and i386 Windows programs together in a WoW64 prefix. Wine's Direct3D 10 and 11
(wined3d), which draw through OpenGL, get feature level 11_1 on it.

WineMetalGL is in it's early development stages. Please report any issues with an attached log and / or screenshot when problems arrive. 

([Test suite](docs/testing.md#direct3d-through-wined3d))

It also gives OpenGL ES 2.0, 3.0, 3.1 and 3.2 contexts
(`WGL_EXT_create_context_es2_profile`), for programs and engines that render with OpenGL ES on Windows: GLSL ES 1.00 to 3.20, the OpenGL ES format, error and completeness rules, ASTC LDR and ETC2 textures, advanced blending, and the common OpenGL ES 2.0 extensions


The core, `libwinemetalgl.dylib`, has no dependency on Apple's OpenGL. It
compiles GLSL with glslang and SPIRV-Cross to the Metal Shading Language and implements the GL state machine, resources and draws on Metal. The Wine module (`winemetalgl.dll` and `winemetalgl.so`, with a forwarding
`opengl32.dll` in the prefix) carries WGL and every GL call to it through the
driver ABI in `include/winemetalgl/driver.h`. 

Wine itself is not patched or rebuilt ([Build it yourself](docs/building.md#the-wine-module)).

## Proof Of Concept

Dead Cells (Requires OpenGL 3.2+) running under MetalSharp Wine-11.17 Using WineMetalGL;

<img width="1920" height="1082" alt="image" src="https://github.com/user-attachments/assets/dcf456b7-3b7f-48c6-b57e-cfd68794ada2" />

DOOM (2016) on its OpenGL renderer under MetalSharp's Wine 11.17 with the
WineMetalGL module: an OpenGL 4.6 compatibility context on an Apple M4.

<img width="1302" height="737" alt="doom_perf" src="https://github.com/user-attachments/assets/f38603ed-b9e8-4ea9-b256-da4ee1cffd52" />




<!-- BEGIN GENERATED: totals (scripts/docs/generate.py) -->
## Conformance totals

<table>
<tr>
<td valign="top">
<table>
<tr><th align="left">OpenGL</th><th align="right">Conformance %</th><th align="right">Unsupported %</th></tr>
<tr><td>3.0</td><td align="right">99.88%</td><td align="right">2.28%</td></tr>
<tr><td>3.1</td><td align="right">100%</td><td align="right">2.38%</td></tr>
<tr><td>3.2</td><td align="right">100%</td><td align="right">0.69%</td></tr>
<tr><td>3.3</td><td align="right">100%</td><td align="right">1.11%</td></tr>
<tr><td>4.0</td><td align="right">99.99%</td><td align="right">10.11%</td></tr>
<tr><td>4.1</td><td align="right">99.99%</td><td align="right">10.08%</td></tr>
<tr><td>4.2</td><td align="right">99.98%</td><td align="right">9.74%</td></tr>
<tr><td>4.3</td><td align="right">99.97%</td><td align="right">15.60%</td></tr>
<tr><td>4.4</td><td align="right">covered by KHR-GL46</td><td align="right">-</td></tr>
<tr><td>4.5</td><td align="right">covered by KHR-GL46</td><td align="right">-</td></tr>
<tr><td>4.6</td><td align="right">99.98%</td><td align="right">22.33%</td></tr>
</table>
</td>
<td valign="top">
<table>
<tr><th align="left">GLES</th><th align="right">Conformance %</th><th align="right">Unsupported %</th></tr>
<tr><td>2.0</td><td align="right">100%</td><td align="right">0.60%</td></tr>
<tr><td>3.0</td><td align="right">100%</td><td align="right">0.83%</td></tr>
<tr><td>3.1</td><td align="right">99.98%</td><td align="right">1.24%</td></tr>
<tr><td>3.2</td><td align="right">99.95%</td><td align="right">0.64%</td></tr>
</table>
</td>
</tr>
</table>

**Conformance %**: passing cases over applicable cases (cases that are not NotSupported) of the version's Khronos CTS mustpass lists (OpenGL: the worst of its runs; GLES: its lists together). **Unsupported %**: cases the CTS reports NotSupported because WineMetalGL does not offer what they test. Each version's features, what is supported and every run: [Conformance results](docs/conformance.md#conformance-totals).

OpenGL 1.0 to 2.1 are implemented by WineMetalGL itself, as its compatibility profile: the fixed-function pipeline, immediate mode, display lists and legacy GLSL, alongside every 3.x feature. No public Khronos suite measures them; the Wine fixtures, GPU tests and game corpus exercise them, and [Unsupported](docs/unsupported.md#compatibility-profile-opengl-10-to-21) lists what is not implemented.

<!-- END GENERATED: totals -->

## Docs

- [Conformance results](docs/conformance.md): every Khronos CTS run per
  OpenGL version, configuration and runner, with the reason for each failing
  case.
- [Unsupported](docs/unsupported.md): what WineMetalGL does not support and
  why: the gaps in the compatibility profile, what the
  Unsupported % counts, the failing CTS cases, and every case the CTS reports
  NotSupported.
- [Test suite](docs/testing.md): each test suite, how to run it, and roughly
  how long it takes.
- [Build it yourself](docs/building.md): building the core and the Wine
  module, installing them into a Wine 11.x, and checking that a program uses
  them.
- [Performance](docs/performance.md): frame times against Apple's OpenGL.
- [Logging](docs/logging.md): log levels and categories, the log file, and
  crash reports.

Also: the [game corpus](docs/games.md) results, [releasing](docs/release.md), and the [design decisions](docs/decisions/).

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md). Security reports:
[SECURITY.md](SECURITY.md). Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

Copyright (c) 2026 MetalSharp. WineMetalGL is licensed under the GNU Lesser
General Public License, version 3 or (at your option) any later version
(`LGPL-3.0-or-later`): [`LICENSE`](LICENSE), which adds to the GNU General
Public License version 3 in [`LICENSES/GPL-3.0.txt`](LICENSES/GPL-3.0.txt).
Releases up to 2.1.0 were published under the MIT License.

The files of the Wine module adapted from Wine (`src/wine/pe/wgl_formats.c`,
`src/wine/pe/wgl_fonts.c`) are LGPL-2.1-or-later, like Wine
(`LICENSES/Wine-LGPL-2.1.txt`). The vendored glslang and SPIRV-Cross keep
their own licenses (`LICENSES/`).
