# Contributing to WineMetalGL

Thanks for helping. WineMetalGL is an OpenGL implementation on Metal for Wine
on macOS; the [README](README.md) describes what it covers and the
[docs](docs/) how it is built and tested.

## Before you start

- **Bugs and game reports**: open an issue with the matching template, and
  include the `GL_RENDERER` string and the program's log file from
  `~/Library/Logs/WineMetalGL`, from a run with `WINEMETALGL_LOG=2`
  ([Logging](docs/logging.md)).
- **Security issues**: never in a public issue. See [SECURITY.md](SECURITY.md).
- **Larger changes** (a new GL feature, a change to the driver ABI or the Wine
  patches): open an issue first, so the approach is agreed before the work.
- **The MetalSharp app itself** (launching, Steam, bottles): report it to
  [metalsharp/MetalSharp](https://github.com/metalsharp/MetalSharp/issues).

## Building and testing

[Build it yourself](docs/building.md) covers the core and a Wine 11.17 with
the patches; [Test suite](docs/testing.md) lists every suite, how to run it
and how long it takes. Before opening a pull request, run at least:

```sh
cmake -S . -B build/host -DWMGL_HOST_ONLY=ON && cmake --build build/host
build/host/wmgl_host_tests && build/host/wmgl_gpu_tests
./scripts/verify-source.sh
./scripts/check-forbidden-identifiers.sh
```

Changes that reach Wine (WGL, contexts, surfaces, the Wine module in
`src/wine`) also need the Wine fixtures (`tests/wine/probe-opengl-wow64.sh`). Changes to core
rendering or GL state that could move conformance results also need the
affected Khronos CTS lists, in `mustpass.xml` configurations 1 to 4.

## Rules the gates enforce

- **Conformance comes from implementing OpenGL.** Code that recognises Khronos
  CTS tests or shaders, or produces their expected output, is not accepted;
  `scripts/check-forbidden-identifiers.sh` keeps it out.
- **Generated files are not edited by hand.** `docs/conformance.md`,
  `docs/performance.md`, `docs/games.md`, `docs/api-coverage.json` and the
  sections between `GENERATED` markers in `README.md`, `docs/unsupported.md`
  and `docs/testing.md` come from `scripts/docs/generate.py`. Change the
  evidence or the generator and regenerate. Hand-written README content goes
  outside the markers.
- **Evidence describes a commit.** CTS results (`tests/cts/baseline`) and game
  corpus results (`tests/games/*.json`) are recorded from a tree without
  uncommitted changes, and the source gate rejects any that are not.
  Failures the core does not fix need a reason in
  `tests/cts/known-failures.json`; NotSupported cases need one in
  `tests/cts/notsupported.json`.
- **Checksums.** After adding, removing or changing tracked files, run
  `scripts/update-checksums.sh` so `MANIFEST.txt` and `SHA256SUMS` match.
  `README.md` is not checksummed, so it can be edited on GitHub.
- **No Apple OpenGL.** The core must not link OpenGL.framework, and it exports
  only the driver ABI in `include/winemetalgl/driver.h`.
- **No Wine changes.** The Wine module (`src/wine`) builds against the headers
  of an unmodified Wine 11.x and installs next to Wine's own modules; it must
  not need a patched or rebuilt Wine.

## Code

- Match the surrounding code: its naming, structure and comment density. The
  core is C++20 and Objective-C++ (Metal-facing files are `.mm`).
- Comments say why. Where behaviour follows the GL specification, name the
  rule or section.
- Every bug fix comes with a regression test: `tests/host` for the compiler
  and formats, `tests/gpu` for the core on Metal, `tests/wine` for Wine
  integration.

## Commits and pull requests

- Commit messages start with the area (`core:`, `wine:`, `docs:`, `cts:`,
  `release:`) and a short summary; the body explains why the change is needed.
- Describe in the pull request what changed and which tests you ran.
- CI must pass. The maintainer ([@aaf2tbz](https://github.com/aaf2tbz))
  reviews every pull request and merges it with a merge commit, which keeps
  the commits that evidence was recorded at in `main`'s history.

## License

WineMetalGL is LGPL-3.0-or-later; the files of the Wine module adapted from
Wine (`src/wine/pe/wgl_formats.c`, `src/wine/pe/wgl_fonts.c`) are
LGPL-2.1-or-later, like Wine. By contributing you agree that your changes are
licensed under the same terms.
