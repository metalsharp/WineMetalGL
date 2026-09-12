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
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_gl33_texture.c" -o "$RUN_ROOT/texture64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_gl33_texture.c" -o "$RUN_ROOT/texture32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_gl33_fbo.c" -o "$RUN_ROOT/fbo64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_gl33_fbo.c" -o "$RUN_ROOT/fbo32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_renderbuffer.c" -o "$RUN_ROOT/renderbuffer64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_renderbuffer.c" -o "$RUN_ROOT/renderbuffer32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_multisample.c" -o "$RUN_ROOT/multisample64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_multisample.c" -o "$RUN_ROOT/multisample32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_sync.c" -o "$RUN_ROOT/sync64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_sync.c" -o "$RUN_ROOT/sync32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_ubo.c" -o "$RUN_ROOT/ubo64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_ubo.c" -o "$RUN_ROOT/ubo32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_texture3d.c" -o "$RUN_ROOT/texture3d64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_texture3d.c" -o "$RUN_ROOT/texture3d32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_fixed_texture.c" -o "$RUN_ROOT/fixed-texture64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_fixed_texture.c" -o "$RUN_ROOT/fixed-texture32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_clear_metal.c" -o "$RUN_ROOT/clear64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_clear_metal.c" -o "$RUN_ROOT/clear32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_gl33_instanced.c" -o "$RUN_ROOT/instanced64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_gl33_instanced.c" -o "$RUN_ROOT/instanced32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_compute.c" -o "$RUN_ROOT/compute64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_compute.c" -o "$RUN_ROOT/compute32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_image.c" -o "$RUN_ROOT/image64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_image.c" -o "$RUN_ROOT/image32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_indirect.c" -o "$RUN_ROOT/indirect64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_indirect.c" -o "$RUN_ROOT/indirect32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_fixed.c" -o "$RUN_ROOT/fixed64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_fixed.c" -o "$RUN_ROOT/fixed32.exe" -lopengl32 -luser32 -lgdi32

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
run_shader "$RUN_ROOT/texture64.exe"
run_shader "$RUN_ROOT/texture32.exe"
run_shader "$RUN_ROOT/fbo64.exe"
run_shader "$RUN_ROOT/fbo32.exe"
run_shader "$RUN_ROOT/renderbuffer64.exe"
run_shader "$RUN_ROOT/renderbuffer32.exe"
run_shader "$RUN_ROOT/multisample64.exe"
run_shader "$RUN_ROOT/multisample32.exe"
run_shader "$RUN_ROOT/sync64.exe"
run_shader "$RUN_ROOT/sync32.exe"
run_shader "$RUN_ROOT/ubo64.exe"
run_shader "$RUN_ROOT/ubo32.exe"
run_shader "$RUN_ROOT/texture3d64.exe"
run_shader "$RUN_ROOT/texture3d32.exe"
run_shader "$RUN_ROOT/fixed-texture64.exe"
run_shader "$RUN_ROOT/fixed-texture32.exe"
run_shader "$RUN_ROOT/clear64.exe"
run_shader "$RUN_ROOT/clear32.exe"
run_shader "$RUN_ROOT/instanced64.exe"
run_shader "$RUN_ROOT/instanced32.exe"
run_shader "$RUN_ROOT/compute64.exe"
run_shader "$RUN_ROOT/compute32.exe"
run_shader "$RUN_ROOT/image64.exe"
run_shader "$RUN_ROOT/image32.exe"
run_shader "$RUN_ROOT/indirect64.exe"
run_shader "$RUN_ROOT/indirect32.exe"
run_shader "$RUN_ROOT/fixed64.exe"
run_shader "$RUN_ROOT/fixed32.exe"
grep -q OPENGL_GL330_METAL_DRAW_READBACK_OK "$RUN_ROOT/gl33-64.exe.stdout"
grep -q OPENGL_GL330_METAL_DRAW_READBACK_OK "$RUN_ROOT/gl33-32.exe.stdout"
grep -q 'OPENGL_EXPERIMENTAL_VERSION_3.3 WineMetalGL' "$RUN_ROOT/gl33-64.exe.stdout"
grep -q 'OPENGL_EXPERIMENTAL_VERSION_3.3 WineMetalGL' "$RUN_ROOT/gl33-32.exe.stdout"
grep -q OPENGL_GL450_METAL_DRAW_READBACK_OK "$RUN_ROOT/gl45-64.exe.stdout"
grep -q OPENGL_GL450_METAL_DRAW_READBACK_OK "$RUN_ROOT/gl45-32.exe.stdout"
grep -q 'WineMetalGL CAMetalLayer present path' "$RUN_ROOT/gl33-64.exe.stdout"
grep -q 'WineMetalGL CAMetalLayer present path' "$RUN_ROOT/gl33-32.exe.stdout"
grep -q WINEMETALGL_GL33_RESOURCES_OK "$RUN_ROOT/resources64.exe.stdout"
grep -q WINEMETALGL_GL33_RESOURCES_OK "$RUN_ROOT/resources32.exe.stdout"
grep -q WINEMETALGL_GL33_TEXTURE_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_GL33_TEXTURE_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_FBO_OK "$RUN_ROOT/fbo64.exe.stdout"
grep -q WINEMETALGL_FBO_OK "$RUN_ROOT/fbo32.exe.stdout"
grep -q WINEMETALGL_BLIT_OK "$RUN_ROOT/fbo64.exe.stdout"
grep -q WINEMETALGL_BLIT_OK "$RUN_ROOT/fbo32.exe.stdout"
grep -q WINEMETALGL_RENDERBUFFER_OK "$RUN_ROOT/renderbuffer64.exe.stdout"
grep -q WINEMETALGL_RENDERBUFFER_OK "$RUN_ROOT/renderbuffer32.exe.stdout"
grep -q WINEMETALGL_MULTISAMPLE_OK "$RUN_ROOT/multisample64.exe.stdout"
grep -q WINEMETALGL_MULTISAMPLE_OK "$RUN_ROOT/multisample32.exe.stdout"
grep -q WINEMETALGL_SYNC_OK "$RUN_ROOT/sync64.exe.stdout"
grep -q WINEMETALGL_SYNC_OK "$RUN_ROOT/sync32.exe.stdout"
grep -q WINEMETALGL_UBO_OK "$RUN_ROOT/ubo64.exe.stdout"
grep -q WINEMETALGL_UBO_OK "$RUN_ROOT/ubo32.exe.stdout"
grep -q WINEMETALGL_TEXTURE3D_OK "$RUN_ROOT/texture3d64.exe.stdout"
grep -q WINEMETALGL_TEXTURE3D_OK "$RUN_ROOT/texture3d32.exe.stdout"
grep -q WINEMETALGL_FIXED_TEXTURE_OK "$RUN_ROOT/fixed-texture64.exe.stdout"
grep -q WINEMETALGL_FIXED_TEXTURE_OK "$RUN_ROOT/fixed-texture32.exe.stdout"
grep -q WINEMETALGL_CLEAR_OK "$RUN_ROOT/clear64.exe.stdout"
grep -q WINEMETALGL_CLEAR_OK "$RUN_ROOT/clear32.exe.stdout"
grep -q WINEMETALGL_INSTANCED_OK "$RUN_ROOT/instanced64.exe.stdout"
grep -q WINEMETALGL_INSTANCED_OK "$RUN_ROOT/instanced32.exe.stdout"
grep -q WINEMETALGL_COMPUTE_OK "$RUN_ROOT/compute64.exe.stdout"
grep -q WINEMETALGL_COMPUTE_OK "$RUN_ROOT/compute32.exe.stdout"
grep -q WINEMETALGL_IMAGE_OK "$RUN_ROOT/image64.exe.stdout"
grep -q WINEMETALGL_IMAGE_OK "$RUN_ROOT/image32.exe.stdout"
grep -q WINEMETALGL_INDIRECT_OK "$RUN_ROOT/indirect64.exe.stdout"
grep -q WINEMETALGL_INDIRECT_OK "$RUN_ROOT/indirect32.exe.stdout"
grep -q WINEMETALGL_FIXED_OK "$RUN_ROOT/fixed64.exe.stdout"
grep -q WINEMETALGL_FIXED_OK "$RUN_ROOT/fixed32.exe.stdout"

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
echo WINEMETALGL_GL33_TEXTURE_OK
echo WINEMETALGL_FBO_OK
echo WINEMETALGL_BLIT_OK
echo WINEMETALGL_RENDERBUFFER_OK
echo WINEMETALGL_MULTISAMPLE_OK
echo WINEMETALGL_SYNC_OK
echo WINEMETALGL_UBO_OK
echo WINEMETALGL_TEXTURE3D_OK
echo WINEMETALGL_FIXED_TEXTURE_OK
echo WINEMETALGL_CLEAR_OK
echo WINEMETALGL_INSTANCED_OK
echo WINEMETALGL_COMPUTE_OK
echo WINEMETALGL_IMAGE_OK
echo WINEMETALGL_INDIRECT_OK
echo WINEMETALGL_FIXED_OK
