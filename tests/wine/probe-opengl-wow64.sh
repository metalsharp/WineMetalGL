#!/bin/bash
# Validate the x86_64 Unix Wine + x86_64/i386 guest OpenGL lane in one WoW64 prefix.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WINE_RUNTIME=${WINEMETALGL_WINE_RUNTIME:-/Volumes/AverySSD/Crossover-WineForge-macos15/merged-build/install/wine-vulkan-portability-test}
# Wine's win32u/dwrite native modules dlopen FreeType by soname. Keep the
# host closure explicit and relocatable instead of depending on Homebrew's
# global search path.
WINE_HOST_LIB_DIR=${WINEMETALGL_HOST_LIB_DIR:-/Volumes/AverySSD/WineForge-macos15-deps/runtime/wine/lib}
if test -d "$WINE_HOST_LIB_DIR"; then
    export DYLD_LIBRARY_PATH="$WINE_HOST_LIB_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
WINE="$WINE_RUNTIME/bin/wine"
WINEBOOT="$WINE_RUNTIME/bin/wineboot"
WINESERVER="$WINE_RUNTIME/bin/wineserver"
CC64=${CC64:-x86_64-w64-mingw32-gcc}
CC32=${CC32:-i686-w64-mingw32-gcc}
RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/winemetalgl-wow64.XXXXXX")
PREFIX="$RUN_ROOT/prefix"
# Prefix initialization can take over a minute on this merged Wine build. Keep
# diagnostics quiet by default; callers can opt into +wgl/+loaddll explicitly.
TIMEOUT_SECONDS=${WINEMETALGL_TIMEOUT_SECONDS:-300}
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
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_contexts.c" -o "$RUN_ROOT/contexts64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_contexts.c" -o "$RUN_ROOT/contexts32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_pipeline.c" -o "$RUN_ROOT/pipeline64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_pipeline.c" -o "$RUN_ROOT/pipeline32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_interface.c" -o "$RUN_ROOT/interface64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_interface.c" -o "$RUN_ROOT/interface32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_multi_draw.c" -o "$RUN_ROOT/multi-draw64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_multi_draw.c" -o "$RUN_ROOT/multi-draw32.exe" -lopengl32 -luser32 -lgdi32
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
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_query.c" -o "$RUN_ROOT/query64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_query.c" -o "$RUN_ROOT/query32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_transform.c" -o "$RUN_ROOT/transform64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_transform.c" -o "$RUN_ROOT/transform32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_transform_shader.c" -o "$RUN_ROOT/transform-shader64.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_shader_binary.c" -o "$RUN_ROOT/shader-binary64.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_debug.c" -o "$RUN_ROOT/debug64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_transform_shader.c" -o "$RUN_ROOT/transform-shader32.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_shader_binary.c" -o "$RUN_ROOT/shader-binary32.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_debug.c" -o "$RUN_ROOT/debug32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_transform_shader.c" -o "$RUN_ROOT/transform-shader-elements64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_transform_shader.c" -o "$RUN_ROOT/transform-shader-elements32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_tess_compile.c" -o "$RUN_ROOT/tess64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_tess_compile.c" -o "$RUN_ROOT/tess32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_tess_draw.c" -o "$RUN_ROOT/tess-draw64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_tess_draw.c" -o "$RUN_ROOT/tess-draw32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_tess_quad.c" -o "$RUN_ROOT/tess-quad64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_tess_quad.c" -o "$RUN_ROOT/tess-quad32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_geometry.c" -o "$RUN_ROOT/geometry64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_geometry.c" -o "$RUN_ROOT/geometry32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_texture3d.c" -o "$RUN_ROOT/texture3d64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_texture3d.c" -o "$RUN_ROOT/texture3d32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_texture_array.c" -o "$RUN_ROOT/texture-array64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_texture_array.c" -o "$RUN_ROOT/texture-array32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_fixed_texture.c" -o "$RUN_ROOT/fixed-texture64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_fixed_texture.c" -o "$RUN_ROOT/fixed-texture32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_clear_metal.c" -o "$RUN_ROOT/clear64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_clear_metal.c" -o "$RUN_ROOT/clear32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_gl33_instanced.c" -o "$RUN_ROOT/instanced64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_gl33_instanced.c" -o "$RUN_ROOT/instanced32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_compute.c" -o "$RUN_ROOT/compute64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_compute.c" -o "$RUN_ROOT/compute32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_image.c" -o "$RUN_ROOT/image64.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_image_formats.c" -o "$RUN_ROOT/image-formats64.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_cube.c" -o "$RUN_ROOT/cube64.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_rectangle.c" -o "$RUN_ROOT/rectangle64.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_1d.c" -o "$RUN_ROOT/1d64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_image.c" -o "$RUN_ROOT/image32.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_image_formats.c" -o "$RUN_ROOT/image-formats32.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_cube.c" -o "$RUN_ROOT/cube32.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_rectangle.c" -o "$RUN_ROOT/rectangle32.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_1d.c" -o "$RUN_ROOT/1d32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_indirect.c" -o "$RUN_ROOT/indirect64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_indirect.c" -o "$RUN_ROOT/indirect32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_indirect_elements.c" -o "$RUN_ROOT/indirect-elements64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_indirect_elements.c" -o "$RUN_ROOT/indirect-elements32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_fixed.c" -o "$RUN_ROOT/fixed64.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_fixed_point.c" -o "$RUN_ROOT/fixed-point64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_fixed.c" -o "$RUN_ROOT/fixed32.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_fixed_point.c" -o "$RUN_ROOT/fixed-point32.exe" -lopengl32 -luser32 -lgdi32
"$CC64" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_blend.c" -o "$RUN_ROOT/blend64.exe" -lopengl32 -luser32 -lgdi32
"$CC32" -O2 -I"$WINE_RUNTIME/include" "$PROJECT_ROOT/tests/wine/opengl_blend.c" -o "$RUN_ROOT/blend32.exe" -lopengl32 -luser32 -lgdi32

export WINEPREFIX="$PREFIX" WINEARCH=wow64 WINEMETALGL=1 WINEDEBUG=${WINEDEBUG:--all}
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
run_shader "$RUN_ROOT/contexts64.exe"
run_shader "$RUN_ROOT/contexts32.exe"
run_shader "$RUN_ROOT/pipeline64.exe"
run_shader "$RUN_ROOT/pipeline32.exe"
run_shader "$RUN_ROOT/interface64.exe"
run_shader "$RUN_ROOT/interface32.exe"
grep -q WINEMETALGL_WGL_MULTI_CONTEXT_OK "$RUN_ROOT/contexts64.exe.stdout"
grep -q WINEMETALGL_WGL_MULTI_CONTEXT_OK "$RUN_ROOT/contexts32.exe.stdout"
grep -q WINEMETALGL_WGL_CONTEXT_STATE_OK "$RUN_ROOT/contexts64.exe.stdout"
grep -q WINEMETALGL_WGL_CONTEXT_STATE_OK "$RUN_ROOT/contexts32.exe.stdout"
grep -q WINEMETALGL_WGL_CONTEXT_THREADS_OK "$RUN_ROOT/contexts64.exe.stdout"
grep -q WINEMETALGL_WGL_CONTEXT_THREADS_OK "$RUN_ROOT/contexts32.exe.stdout"
grep -q WINEMETALGL_PROGRAM_PIPELINE_OK "$RUN_ROOT/pipeline64.exe.stdout"
grep -q WINEMETALGL_PROGRAM_PIPELINE_OK "$RUN_ROOT/pipeline32.exe.stdout"
grep -q WINEMETALGL_PROGRAM_PIPELINE_DRAW_OK "$RUN_ROOT/pipeline64.exe.stdout"
grep -q WINEMETALGL_PROGRAM_PIPELINE_DRAW_OK "$RUN_ROOT/pipeline32.exe.stdout"
grep -q WINEMETALGL_INTERFACE_REJECT_OK "$RUN_ROOT/interface64.exe.stdout"
grep -q WINEMETALGL_INTERFACE_REJECT_OK "$RUN_ROOT/interface32.exe.stdout"
run_shader "$RUN_ROOT/multi-draw64.exe"
run_shader "$RUN_ROOT/multi-draw32.exe"
grep -q WINEMETALGL_MULTI_DRAW_OK "$RUN_ROOT/multi-draw64.exe.stdout"
grep -q WINEMETALGL_MULTI_DRAW_OK "$RUN_ROOT/multi-draw32.exe.stdout"
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
run_shader "$RUN_ROOT/query64.exe"
run_shader "$RUN_ROOT/query32.exe"
run_shader "$RUN_ROOT/transform64.exe"
run_shader "$RUN_ROOT/transform32.exe"
run_shader "$RUN_ROOT/transform-shader64.exe"
run_shader "$RUN_ROOT/transform-shader32.exe"
run_shader "$RUN_ROOT/transform-shader-elements64.exe"
run_shader "$RUN_ROOT/transform-shader-elements32.exe"
run_shader "$RUN_ROOT/shader-binary64.exe"
run_shader "$RUN_ROOT/shader-binary32.exe"
run_shader "$RUN_ROOT/debug64.exe"
run_shader "$RUN_ROOT/debug32.exe"
run_shader "$RUN_ROOT/tess64.exe"
run_shader "$RUN_ROOT/tess32.exe"
run_shader "$RUN_ROOT/tess-draw64.exe"
run_shader "$RUN_ROOT/tess-draw32.exe"
run_shader "$RUN_ROOT/tess-quad64.exe"
run_shader "$RUN_ROOT/tess-quad32.exe"
run_shader "$RUN_ROOT/geometry64.exe"
run_shader "$RUN_ROOT/geometry32.exe"
run_shader "$RUN_ROOT/texture3d64.exe"
run_shader "$RUN_ROOT/texture3d32.exe"
run_shader "$RUN_ROOT/texture-array64.exe"
run_shader "$RUN_ROOT/texture-array32.exe"
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
run_shader "$RUN_ROOT/image-formats64.exe"
run_shader "$RUN_ROOT/image-formats32.exe"
run_shader "$RUN_ROOT/cube64.exe"
run_shader "$RUN_ROOT/cube32.exe"
run_shader "$RUN_ROOT/rectangle64.exe"
run_shader "$RUN_ROOT/rectangle32.exe"
run_shader "$RUN_ROOT/1d64.exe"
run_shader "$RUN_ROOT/1d32.exe"
run_shader "$RUN_ROOT/indirect64.exe"
run_shader "$RUN_ROOT/indirect32.exe"
run_shader "$RUN_ROOT/indirect-elements64.exe"
run_shader "$RUN_ROOT/indirect-elements32.exe"
run_shader "$RUN_ROOT/fixed64.exe"
run_shader "$RUN_ROOT/fixed32.exe"
run_shader "$RUN_ROOT/fixed-point64.exe"
run_shader "$RUN_ROOT/fixed-point32.exe"
run_shader "$RUN_ROOT/blend64.exe"
run_shader "$RUN_ROOT/blend32.exe"
grep -q OPENGL_GL330_METAL_DRAW_READBACK_OK "$RUN_ROOT/gl33-64.exe.stdout"
grep -q OPENGL_GL330_METAL_DRAW_READBACK_OK "$RUN_ROOT/gl33-32.exe.stdout"
grep -q 'OPENGL_EXPERIMENTAL_VERSION_3.3 WineMetalGL' "$RUN_ROOT/gl33-64.exe.stdout"
grep -q 'OPENGL_EXPERIMENTAL_VERSION_3.3 WineMetalGL' "$RUN_ROOT/gl33-32.exe.stdout"
grep -q WINEMETALGL_STATE_QUERY_OK "$RUN_ROOT/gl33-64.exe.stdout"
grep -q WINEMETALGL_STATE_QUERY_OK "$RUN_ROOT/gl33-32.exe.stdout"
grep -q WINEMETALGL_FRAG_DATA_LOCATION_OK "$RUN_ROOT/gl33-64.exe.stdout"
grep -q WINEMETALGL_FRAG_DATA_LOCATION_OK "$RUN_ROOT/gl33-32.exe.stdout"
grep -q WINEMETALGL_RASTER_STATE_OK "$RUN_ROOT/gl33-64.exe.stdout"
grep -q WINEMETALGL_RASTER_STATE_OK "$RUN_ROOT/gl33-32.exe.stdout"
grep -q OPENGL_GL450_METAL_DRAW_READBACK_OK "$RUN_ROOT/gl45-64.exe.stdout"
grep -q OPENGL_GL450_METAL_DRAW_READBACK_OK "$RUN_ROOT/gl45-32.exe.stdout"
grep -q WINEMETALGL_GL33_RESOURCES_OK "$RUN_ROOT/resources64.exe.stdout"
grep -q WINEMETALGL_GL33_RESOURCES_OK "$RUN_ROOT/resources32.exe.stdout"
grep -q WINEMETALGL_UNIFORM_ARRAY_REFLECTION_OK "$RUN_ROOT/resources64.exe.stdout"
grep -q WINEMETALGL_UNIFORM_ARRAY_REFLECTION_OK "$RUN_ROOT/resources32.exe.stdout"
grep -q WINEMETALGL_PBO_READBACK_OK "$RUN_ROOT/resources64.exe.stdout"
grep -q WINEMETALGL_PBO_READBACK_OK "$RUN_ROOT/resources32.exe.stdout"
grep -q WINEMETALGL_UNPACK_ALIGNMENT_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_UNPACK_ALIGNMENT_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_MIPMAP_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_MIPMAP_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_GL33_TEXTURE_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_GL33_TEXTURE_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_PACKED_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_PACKED_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_USHORT_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_USHORT_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_HALF_FLOAT_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_HALF_FLOAT_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_RGBA16F_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_RGBA16F_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_RGBA32F_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_RGBA32F_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_SRGB_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_SRGB_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_RED_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_RED_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_RG_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_RG_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_LUMINANCE_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_LUMINANCE_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_READBACK_PACKED_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_READBACK_PACKED_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_READBACK_HALF_FLOAT_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_READBACK_HALF_FLOAT_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_PACK_ALIGNMENT_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_PACK_ALIGNMENT_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_BGRA_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_BGRA_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_SAMPLER_PARAMS_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_SAMPLER_PARAMS_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_SAMPLER_QUERY_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_SAMPLER_QUERY_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_COPY_IMAGE_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_COPY_IMAGE_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_COPY_TEX_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_COPY_TEX_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_TEX_STORAGE_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_TEX_STORAGE_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_DSA_TEXTURE_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_DSA_TEXTURE_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_VIEW_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_VIEW_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_SAMPLER_ADVANCED_OK "$RUN_ROOT/texture64.exe.stdout"
grep -q WINEMETALGL_SAMPLER_ADVANCED_OK "$RUN_ROOT/texture32.exe.stdout"
grep -q WINEMETALGL_FBO_OK "$RUN_ROOT/fbo64.exe.stdout"
grep -q WINEMETALGL_FBO_OK "$RUN_ROOT/fbo32.exe.stdout"
grep -q WINEMETALGL_FBO_DEPTH_TEXTURE_OK "$RUN_ROOT/fbo64.exe.stdout"
grep -q WINEMETALGL_FBO_DEPTH_TEXTURE_OK "$RUN_ROOT/fbo32.exe.stdout"
grep -q WINEMETALGL_FBO_DEPTH_READBACK_OK "$RUN_ROOT/fbo64.exe.stdout"
grep -q WINEMETALGL_FBO_DEPTH_READBACK_OK "$RUN_ROOT/fbo32.exe.stdout"
grep -q WINEMETALGL_FBO_STENCIL_READBACK_OK "$RUN_ROOT/fbo64.exe.stdout"
grep -q WINEMETALGL_FBO_STENCIL_READBACK_OK "$RUN_ROOT/fbo32.exe.stdout"
grep -q WINEMETALGL_FBO_ARRAY_LAYER_OK "$RUN_ROOT/fbo64.exe.stdout"
grep -q WINEMETALGL_FBO_ARRAY_LAYER_OK "$RUN_ROOT/fbo32.exe.stdout"
grep -q WINEMETALGL_BLIT_OK "$RUN_ROOT/fbo64.exe.stdout"
grep -q WINEMETALGL_BLIT_OK "$RUN_ROOT/fbo32.exe.stdout"
grep -q WINEMETALGL_RENDERBUFFER_OK "$RUN_ROOT/renderbuffer64.exe.stdout"
grep -q WINEMETALGL_RENDERBUFFER_OK "$RUN_ROOT/renderbuffer32.exe.stdout"
grep -q WINEMETALGL_MULTISAMPLE_OK "$RUN_ROOT/multisample64.exe.stdout"
grep -q WINEMETALGL_MULTISAMPLE_OK "$RUN_ROOT/multisample32.exe.stdout"
grep -q WINEMETALGL_MULTISAMPLE_STORAGE_OK "$RUN_ROOT/multisample64.exe.stdout"
grep -q WINEMETALGL_MULTISAMPLE_STORAGE_OK "$RUN_ROOT/multisample32.exe.stdout"
grep -q WINEMETALGL_SYNC_OK "$RUN_ROOT/sync64.exe.stdout"
grep -q WINEMETALGL_SYNC_OK "$RUN_ROOT/sync32.exe.stdout"
grep -q WINEMETALGL_UBO_OK "$RUN_ROOT/ubo64.exe.stdout"
grep -q WINEMETALGL_UBO_OK "$RUN_ROOT/ubo32.exe.stdout"
grep -q WINEMETALGL_BUFFER_SIZE_OK "$RUN_ROOT/ubo64.exe.stdout"
grep -q WINEMETALGL_BUFFER_SIZE_OK "$RUN_ROOT/ubo32.exe.stdout"
grep -q WINEMETALGL_UBO_RANGE_OK "$RUN_ROOT/ubo64.exe.stdout"
grep -q WINEMETALGL_UBO_RANGE_OK "$RUN_ROOT/ubo32.exe.stdout"
grep -q WINEMETALGL_UBO_REFLECTION_OK "$RUN_ROOT/ubo64.exe.stdout"
grep -q WINEMETALGL_UBO_REFLECTION_OK "$RUN_ROOT/ubo32.exe.stdout"
grep -q WINEMETALGL_QUERY_OK "$RUN_ROOT/query64.exe.stdout"
grep -q WINEMETALGL_QUERY_OK "$RUN_ROOT/query32.exe.stdout"
grep -q WINEMETALGL_TRANSFORM_FIXED_OK "$RUN_ROOT/transform64.exe.stdout"
grep -q WINEMETALGL_TRANSFORM_FIXED_OK "$RUN_ROOT/transform32.exe.stdout"
grep -q WINEMETALGL_TRANSFORM_SHADER_OK "$RUN_ROOT/transform-shader64.exe.stdout"
grep -q WINEMETALGL_TRANSFORM_SHADER_OK "$RUN_ROOT/transform-shader32.exe.stdout"
grep -q WINEMETALGL_TRANSFORM_SHADER_ELEMENTS_OK "$RUN_ROOT/transform-shader-elements64.exe.stdout"
grep -q WINEMETALGL_TRANSFORM_SHADER_ELEMENTS_OK "$RUN_ROOT/transform-shader-elements32.exe.stdout"
grep -q WINEMETALGL_TRANSFORM_SHADER_RANGE_OK "$RUN_ROOT/transform-shader-elements64.exe.stdout"
grep -q WINEMETALGL_TRANSFORM_SHADER_RANGE_OK "$RUN_ROOT/transform-shader-elements32.exe.stdout"
grep -q WINEMETALGL_SHADER_BINARY_OK "$RUN_ROOT/shader-binary64.exe.stdout"
grep -q WINEMETALGL_SHADER_BINARY_OK "$RUN_ROOT/shader-binary32.exe.stdout"
grep -q WINEMETALGL_SHADER_SPECIALIZE_OK "$RUN_ROOT/shader-binary64.exe.stdout"
grep -q WINEMETALGL_SHADER_SPECIALIZE_OK "$RUN_ROOT/shader-binary32.exe.stdout"
grep -q WINEMETALGL_PROGRAM_BINARY_OK "$RUN_ROOT/shader-binary64.exe.stdout"
grep -q WINEMETALGL_PROGRAM_BINARY_OK "$RUN_ROOT/shader-binary32.exe.stdout"
grep -q WINEMETALGL_DEBUG_OUTPUT_OK "$RUN_ROOT/debug64.exe.stdout"
grep -q WINEMETALGL_DEBUG_OUTPUT_OK "$RUN_ROOT/debug32.exe.stdout"
grep -q WINEMETALGL_TESSELLATION_COMPILE_OK "$RUN_ROOT/tess64.exe.stdout"
grep -q WINEMETALGL_TESSELLATION_COMPILE_OK "$RUN_ROOT/tess32.exe.stdout"
grep -q WINEMETALGL_TESSELLATION_DRAW_OK "$RUN_ROOT/tess-draw64.exe.stdout"
grep -q WINEMETALGL_TESSELLATION_DRAW_OK "$RUN_ROOT/tess-draw32.exe.stdout"
grep -q WINEMETALGL_TESSELLATION_QUAD_OK "$RUN_ROOT/tess-quad64.exe.stdout"
grep -q WINEMETALGL_TESSELLATION_QUAD_OK "$RUN_ROOT/tess-quad32.exe.stdout"
grep -q WINEMETALGL_GEOMETRY_PASSTHROUGH_OK "$RUN_ROOT/geometry64.exe.stdout"
grep -q WINEMETALGL_GEOMETRY_PASSTHROUGH_OK "$RUN_ROOT/geometry32.exe.stdout"
grep -q WINEMETALGL_TEXTURE3D_OK "$RUN_ROOT/texture3d64.exe.stdout"
grep -q WINEMETALGL_TEXTURE3D_OK "$RUN_ROOT/texture3d32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_ARRAY_OK "$RUN_ROOT/texture-array64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_ARRAY_OK "$RUN_ROOT/texture-array32.exe.stdout"
grep -q WINEMETALGL_FIXED_TEXTURE_OK "$RUN_ROOT/fixed-texture64.exe.stdout"
grep -q WINEMETALGL_FIXED_TEXTURE_OK "$RUN_ROOT/fixed-texture32.exe.stdout"
grep -q WINEMETALGL_FIXED_TEXTURE_REPLACE_OK "$RUN_ROOT/fixed-texture64.exe.stdout"
grep -q WINEMETALGL_FIXED_TEXTURE_REPLACE_OK "$RUN_ROOT/fixed-texture32.exe.stdout"
grep -q WINEMETALGL_FIXED_TEXTURE_ADD_OK "$RUN_ROOT/fixed-texture64.exe.stdout"
grep -q WINEMETALGL_FIXED_TEXTURE_ADD_OK "$RUN_ROOT/fixed-texture32.exe.stdout"
grep -q WINEMETALGL_FIXED_TEXTURE_DECAL_OK "$RUN_ROOT/fixed-texture64.exe.stdout"
grep -q WINEMETALGL_FIXED_TEXTURE_DECAL_OK "$RUN_ROOT/fixed-texture32.exe.stdout"
grep -q WINEMETALGL_FIXED_TEXTURE_COMBINE_OK "$RUN_ROOT/fixed-texture64.exe.stdout"
grep -q WINEMETALGL_FIXED_TEXTURE_COMBINE_OK "$RUN_ROOT/fixed-texture32.exe.stdout"
grep -q WINEMETALGL_FIXED_TEXGEN_OK "$RUN_ROOT/fixed-texture64.exe.stdout"
grep -q WINEMETALGL_FIXED_TEXGEN_OK "$RUN_ROOT/fixed-texture32.exe.stdout"
grep -q WINEMETALGL_CLEAR_OK "$RUN_ROOT/clear64.exe.stdout"
grep -q WINEMETALGL_CLEAR_OK "$RUN_ROOT/clear32.exe.stdout"
grep -q WINEMETALGL_CLEAR_BUFFER_OK "$RUN_ROOT/clear64.exe.stdout"
grep -q WINEMETALGL_CLEAR_BUFFER_OK "$RUN_ROOT/clear32.exe.stdout"
grep -q WINEMETALGL_INSTANCED_OK "$RUN_ROOT/instanced64.exe.stdout"
grep -q WINEMETALGL_INSTANCED_OK "$RUN_ROOT/instanced32.exe.stdout"
grep -q WINEMETALGL_COMPUTE_OK "$RUN_ROOT/compute64.exe.stdout"
grep -q WINEMETALGL_COMPUTE_OK "$RUN_ROOT/compute32.exe.stdout"
grep -q WINEMETALGL_SSBO_REFLECTION_OK "$RUN_ROOT/compute64.exe.stdout"
grep -q WINEMETALGL_SSBO_REFLECTION_OK "$RUN_ROOT/compute32.exe.stdout"
grep -q WINEMETALGL_RESOURCE_REFLECTION_OK "$RUN_ROOT/compute64.exe.stdout"
grep -q WINEMETALGL_RESOURCE_REFLECTION_OK "$RUN_ROOT/compute32.exe.stdout"
grep -q WINEMETALGL_COMPUTE_RANGE_OK "$RUN_ROOT/compute64.exe.stdout"
grep -q WINEMETALGL_COMPUTE_RANGE_OK "$RUN_ROOT/compute32.exe.stdout"
grep -q WINEMETALGL_COMPUTE_INDIRECT_OK "$RUN_ROOT/compute64.exe.stdout"
grep -q WINEMETALGL_COMPUTE_INDIRECT_OK "$RUN_ROOT/compute32.exe.stdout"
grep -q WINEMETALGL_IMAGE_OK "$RUN_ROOT/image64.exe.stdout"
grep -q WINEMETALGL_IMAGE_OK "$RUN_ROOT/image32.exe.stdout"
grep -q WINEMETALGL_IMAGE_R32UI_OK "$RUN_ROOT/image-formats64.exe.stdout"
grep -q WINEMETALGL_IMAGE_R32UI_OK "$RUN_ROOT/image-formats32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_CUBE_OK "$RUN_ROOT/cube64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_CUBE_OK "$RUN_ROOT/cube32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_RECTANGLE_OK "$RUN_ROOT/rectangle64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_RECTANGLE_OK "$RUN_ROOT/rectangle32.exe.stdout"
grep -q WINEMETALGL_TEXTURE_1D_OK "$RUN_ROOT/1d64.exe.stdout"
grep -q WINEMETALGL_TEXTURE_1D_OK "$RUN_ROOT/1d32.exe.stdout"
grep -q WINEMETALGL_INDIRECT_OK "$RUN_ROOT/indirect64.exe.stdout"
grep -q WINEMETALGL_INDIRECT_OK "$RUN_ROOT/indirect32.exe.stdout"
grep -q WINEMETALGL_MULTI_INDIRECT_OK "$RUN_ROOT/indirect64.exe.stdout"
grep -q WINEMETALGL_MULTI_INDIRECT_OK "$RUN_ROOT/indirect32.exe.stdout"
grep -q WINEMETALGL_MULTI_ELEMENTS_INDIRECT_OK "$RUN_ROOT/indirect-elements64.exe.stdout"
grep -q WINEMETALGL_MULTI_ELEMENTS_INDIRECT_OK "$RUN_ROOT/indirect-elements32.exe.stdout"
grep -q WINEMETALGL_FIXED_OK "$RUN_ROOT/fixed64.exe.stdout"
grep -q WINEMETALGL_FIXED_OK "$RUN_ROOT/fixed32.exe.stdout"
grep -q WINEMETALGL_FIXED_LIGHTING_OK "$RUN_ROOT/fixed64.exe.stdout"
grep -q WINEMETALGL_FIXED_LIGHTING_OK "$RUN_ROOT/fixed32.exe.stdout"
grep -q WINEMETALGL_FIXED_SPECULAR_OK "$RUN_ROOT/fixed64.exe.stdout"
grep -q WINEMETALGL_FIXED_SPECULAR_OK "$RUN_ROOT/fixed32.exe.stdout"
grep -q WINEMETALGL_FIXED_FOG_OK "$RUN_ROOT/fixed64.exe.stdout"
grep -q WINEMETALGL_FIXED_FOG_OK "$RUN_ROOT/fixed32.exe.stdout"
grep -q WINEMETALGL_CLIP_PLANE_OK "$RUN_ROOT/fixed64.exe.stdout"
grep -q WINEMETALGL_CLIP_PLANE_OK "$RUN_ROOT/fixed32.exe.stdout"
grep -q WINEMETALGL_FIXED_POINT_SIZE_OK "$RUN_ROOT/fixed-point64.exe.stdout"
grep -q WINEMETALGL_FIXED_POINT_SIZE_OK "$RUN_ROOT/fixed-point32.exe.stdout"
grep -q WINEMETALGL_BLEND_OK "$RUN_ROOT/blend64.exe.stdout"
grep -q WINEMETALGL_BLEND_OK "$RUN_ROOT/blend32.exe.stdout"
grep -q WINEMETALGL_PER_TARGET_BLEND_OK "$RUN_ROOT/blend64.exe.stdout"
grep -q WINEMETALGL_PER_TARGET_BLEND_OK "$RUN_ROOT/blend32.exe.stdout"
grep -q WINEMETALGL_BLEND_CONSTANT_OK "$RUN_ROOT/blend64.exe.stdout"
grep -q WINEMETALGL_BLEND_CONSTANT_OK "$RUN_ROOT/blend32.exe.stdout"
grep -q WINEMETALGL_SCISSOR_OK "$RUN_ROOT/blend64.exe.stdout"
grep -q WINEMETALGL_SCISSOR_OK "$RUN_ROOT/blend32.exe.stdout"
grep -q WINEMETALGL_CULL_OK "$RUN_ROOT/blend64.exe.stdout"
grep -q WINEMETALGL_CULL_OK "$RUN_ROOT/blend32.exe.stdout"

echo WINEMETALGL_WOW64_X86_64_OK
echo WINEMETALGL_WOW64_I386_OK
echo WINEMETALGL_OPENGL32_LOAD_OK
echo WINEMETALGL_WGL_CONTEXT_OK
echo WINEMETALGL_WGL_MULTI_CONTEXT_OK
echo WINEMETALGL_WGL_CONTEXT_STATE_OK
echo WINEMETALGL_WGL_CONTEXT_THREADS_OK
echo WINEMETALGL_PROGRAM_PIPELINE_OK
echo WINEMETALGL_PROGRAM_PIPELINE_DRAW_OK
echo WINEMETALGL_INTERFACE_REJECT_OK
echo WINEMETALGL_MULTI_DRAW_OK
echo WINEMETALGL_METAL_SURFACE_OK
echo WINEMETALGL_DEFAULT_FBO_PRESENT_OK
echo WINEMETALGL_READBACK_OK
echo WINEMETALGL_GLSL330_OK
echo WINEMETALGL_STATE_QUERY_OK
echo WINEMETALGL_FRAG_DATA_LOCATION_OK
echo WINEMETALGL_RASTER_STATE_OK
echo WINEMETALGL_GLSL450_OK
echo WINEMETALGL_GL33_RESOURCES_OK
echo WINEMETALGL_UNIFORM_ARRAY_REFLECTION_OK
echo WINEMETALGL_PBO_READBACK_OK
echo WINEMETALGL_UNPACK_ALIGNMENT_OK
echo WINEMETALGL_MIPMAP_OK
echo WINEMETALGL_GL33_TEXTURE_OK
echo WINEMETALGL_TEXTURE_PACKED_OK
echo WINEMETALGL_TEXTURE_USHORT_OK
echo WINEMETALGL_TEXTURE_HALF_FLOAT_OK
echo WINEMETALGL_TEXTURE_RGBA16F_OK
echo WINEMETALGL_TEXTURE_RGBA32F_OK
echo WINEMETALGL_TEXTURE_SRGB_OK
echo WINEMETALGL_TEXTURE_RED_OK
echo WINEMETALGL_TEXTURE_RG_OK
echo WINEMETALGL_TEXTURE_LUMINANCE_OK
echo WINEMETALGL_READBACK_PACKED_OK
echo WINEMETALGL_READBACK_HALF_FLOAT_OK
echo WINEMETALGL_PACK_ALIGNMENT_OK
echo WINEMETALGL_TEXTURE_BGRA_OK
echo WINEMETALGL_TEXTURE_SAMPLER_PARAMS_OK
echo WINEMETALGL_TEXTURE_SAMPLER_QUERY_OK
echo WINEMETALGL_COPY_IMAGE_OK
echo WINEMETALGL_COPY_TEX_OK
echo WINEMETALGL_TEX_STORAGE_OK
echo WINEMETALGL_DSA_TEXTURE_OK
echo WINEMETALGL_TEXTURE_VIEW_OK
echo WINEMETALGL_SAMPLER_ADVANCED_OK
echo WINEMETALGL_FBO_OK
echo WINEMETALGL_FBO_DEPTH_TEXTURE_OK
echo WINEMETALGL_FBO_DEPTH_READBACK_OK
echo WINEMETALGL_FBO_STENCIL_READBACK_OK
echo WINEMETALGL_FBO_ARRAY_LAYER_OK
echo WINEMETALGL_BLIT_OK
echo WINEMETALGL_RENDERBUFFER_OK
echo WINEMETALGL_MULTISAMPLE_OK
echo WINEMETALGL_MULTISAMPLE_STORAGE_OK
echo WINEMETALGL_SYNC_OK
echo WINEMETALGL_UBO_OK
echo WINEMETALGL_BUFFER_SIZE_OK
echo WINEMETALGL_UBO_RANGE_OK
echo WINEMETALGL_UBO_REFLECTION_OK
echo WINEMETALGL_QUERY_OK
echo WINEMETALGL_TRANSFORM_FIXED_OK
echo WINEMETALGL_TRANSFORM_SHADER_OK
echo WINEMETALGL_TRANSFORM_SHADER_ELEMENTS_OK
echo WINEMETALGL_TRANSFORM_SHADER_RANGE_OK
echo WINEMETALGL_SHADER_BINARY_OK
echo WINEMETALGL_SHADER_SPECIALIZE_OK
echo WINEMETALGL_PROGRAM_BINARY_OK
echo WINEMETALGL_DEBUG_OUTPUT_OK
echo WINEMETALGL_TESSELLATION_COMPILE_OK
echo WINEMETALGL_TESSELLATION_DRAW_OK
echo WINEMETALGL_TESSELLATION_QUAD_OK
echo WINEMETALGL_GEOMETRY_PASSTHROUGH_OK
echo WINEMETALGL_TEXTURE3D_OK
echo WINEMETALGL_TEXTURE_ARRAY_OK
echo WINEMETALGL_FIXED_TEXTURE_OK
echo WINEMETALGL_FIXED_TEXTURE_REPLACE_OK
echo WINEMETALGL_FIXED_TEXTURE_ADD_OK
echo WINEMETALGL_FIXED_TEXTURE_DECAL_OK
echo WINEMETALGL_FIXED_TEXTURE_COMBINE_OK
echo WINEMETALGL_FIXED_TEXGEN_OK
echo WINEMETALGL_CLEAR_OK
echo WINEMETALGL_CLEAR_BUFFER_OK
echo WINEMETALGL_INSTANCED_OK
echo WINEMETALGL_COMPUTE_OK
echo WINEMETALGL_SSBO_REFLECTION_OK
echo WINEMETALGL_RESOURCE_REFLECTION_OK
echo WINEMETALGL_COMPUTE_RANGE_OK
echo WINEMETALGL_COMPUTE_INDIRECT_OK
echo WINEMETALGL_IMAGE_OK
echo WINEMETALGL_IMAGE_R32UI_OK
echo WINEMETALGL_TEXTURE_CUBE_OK
echo WINEMETALGL_TEXTURE_RECTANGLE_OK
echo WINEMETALGL_TEXTURE_1D_OK
echo WINEMETALGL_INDIRECT_OK
echo WINEMETALGL_MULTI_INDIRECT_OK
echo WINEMETALGL_MULTI_ELEMENTS_INDIRECT_OK
echo WINEMETALGL_FIXED_OK
echo WINEMETALGL_FIXED_LIGHTING_OK
echo WINEMETALGL_FIXED_SPECULAR_OK
echo WINEMETALGL_FIXED_FOG_OK
echo WINEMETALGL_CLIP_PLANE_OK
echo WINEMETALGL_FIXED_POINT_SIZE_OK
echo WINEMETALGL_BLEND_OK
echo WINEMETALGL_PER_TARGET_BLEND_OK
echo WINEMETALGL_BLEND_CONSTANT_OK
echo WINEMETALGL_SCISSOR_OK
echo WINEMETALGL_CULL_OK
