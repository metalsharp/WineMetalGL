#!/bin/bash
# Validate the x86_64 Unix Wine + x86_64/i386 guest OpenGL lane in one WoW64 prefix.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WINE_RUNTIME=${WINEMETALGL_WINE_RUNTIME:-/Volumes/AverySSD/Crossover-WineForge-macos15/merged-build/install/wine-vulkan-portability-test}
WINE="$WINE_RUNTIME/bin/wine"
WINEBOOT="$WINE_RUNTIME/bin/wineboot"
WINESERVER="$WINE_RUNTIME/bin/wineserver"
CC64=${CC64:-x86_64-w64-mingw32-gcc}
CC32=${CC32:-i686-w64-mingw32-gcc}
RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/winemetalgl-wow64.XXXXXX")
PREFIX="$RUN_ROOT/prefix"
TIMEOUT_SECONDS=${WINEMETALGL_TIMEOUT_SECONDS:-180}
if command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT=(gtimeout --signal=TERM --kill-after=10s "${TIMEOUT_SECONDS}s")
elif command -v timeout >/dev/null 2>&1; then
    TIMEOUT=(timeout --signal=TERM --kill-after=10s "${TIMEOUT_SECONDS}s")
else
    TIMEOUT=()
fi

cleanup() {
    status=$?
    WINEPREFIX="$PREFIX" "$WINESERVER" -k 2>/dev/null || true
    WINEPREFIX="$PREFIX" "$WINESERVER" -w 2>/dev/null || true
    if test "${WINEMETALGL_KEEP_RUN:-0}" = 1; then
        echo "Retained probe run: $RUN_ROOT" >&2
    else
        rm -rf "$RUN_ROOT"
    fi
    exit "$status"
}
trap cleanup EXIT HUP INT TERM

for required in "$WINE" "$WINEBOOT" "$WINESERVER" "$PROJECT_ROOT/build/release/metalsharp-opengl.dylib"; do
    test -e "$required" || { echo "Missing WineMetalGL input: $required" >&2; exit 1; }
done

test "$(/usr/bin/lipo -archs "$PROJECT_ROOT/build/release/metalsharp-opengl.dylib")" = x86_64
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_load_probe.c" -o "$RUN_ROOT/load64.exe" -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_load_probe.c" -o "$RUN_ROOT/load32.exe" -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_runtime.c" -o "$RUN_ROOT/runtime64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_runtime.c" -o "$RUN_ROOT/runtime32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_gl33_metal.c" -o "$RUN_ROOT/gl33-64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_gl33_metal.c" -o "$RUN_ROOT/gl33-32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -DWINEMETALGL_GLSL450 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_gl33_metal.c" -o "$RUN_ROOT/gl45-64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -DWINEMETALGL_GLSL450 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_gl33_metal.c" -o "$RUN_ROOT/gl45-32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_gl33_resources.c" -o "$RUN_ROOT/resources64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_gl33_resources.c" -o "$RUN_ROOT/resources32.exe" -lopengl32 -luser32 -lgdi32

export WINEPREFIX="$PREFIX" WINEARCH=wow64 WINEMETALGL=1 WINEDEBUG=${WINEDEBUG:-+wgl}
"${TIMEOUT[@]}" "$WINE" wineboot -u >"$RUN_ROOT/wineboot.log" 2>&1

run_marker() {
    local exe=$1 marker=$2
    "${TIMEOUT[@]}" "$WINE" "$exe" "Z:$marker" >"$marker.stdout" 2>&1
}
run_shader() {
    local exe=$1
    WINEMETALGL_EXPERIMENTAL=1 "${TIMEOUT[@]}" "$WINE" "$exe" >"$exe.stdout" 2>&1
}

run_marker "$RUN_ROOT/load64.exe" "$RUN_ROOT/load64.marker"
run_marker "$RUN_ROOT/load32.exe" "$RUN_ROOT/load32.marker"
run_marker "$RUN_ROOT/runtime64.exe" "$RUN_ROOT/runtime64.marker"
run_marker "$RUN_ROOT/runtime32.exe" "$RUN_ROOT/runtime32.marker"
grep -q OPENGL_RUNTIME_ALL_OK "$RUN_ROOT/runtime64.marker"
grep -q OPENGL_RUNTIME_ALL_OK "$RUN_ROOT/runtime32.marker"
run_shader "$RUN_ROOT/gl33-64.exe"
run_shader "$RUN_ROOT/gl33-32.exe"
run_shader "$RUN_ROOT/gl45-64.exe"
run_shader "$RUN_ROOT/gl45-32.exe"
run_shader "$RUN_ROOT/resources64.exe"
run_shader "$RUN_ROOT/resources32.exe"
grep -q OPENGL_GL330_METAL_DRAW_READBACK_OK "$RUN_ROOT/gl33-64.exe.stdout"
grep -q OPENGL_GL330_METAL_DRAW_READBACK_OK "$RUN_ROOT/gl33-32.exe.stdout"
grep -q OPENGL_GL450_METAL_DRAW_READBACK_OK "$RUN_ROOT/gl45-64.exe.stdout"
grep -q OPENGL_GL450_METAL_DRAW_READBACK_OK "$RUN_ROOT/gl45-32.exe.stdout"
grep -q 'WineMetalGL CAMetalLayer present path' "$RUN_ROOT/gl33-64.exe.stdout"
grep -q 'WineMetalGL CAMetalLayer present path' "$RUN_ROOT/gl33-32.exe.stdout"
grep -q WINEMETALGL_GL33_RESOURCES_OK "$RUN_ROOT/resources64.exe.stdout"
grep -q WINEMETALGL_GL33_RESOURCES_OK "$RUN_ROOT/resources32.exe.stdout"

echo WINEMETALGL_WOW64_X86_64_OK
echo WINEMETALGL_WOW64_I386_OK
echo WINEMETALGL_OPENGL32_LOAD_OK
echo WINEMETALGL_WGL_CONTEXT_OK
echo WINEMETALGL_METAL_SURFACE_OK
echo WINEMETALGL_DEFAULT_FBO_PRESENT_OK
echo WINEMETALGL_READBACK_OK
echo WINEMETALGL_GLSL330_OK
echo WINEMETALGL_GLSL450_OK
echo WINEMETALGL_GL33_RESOURCES_OK
