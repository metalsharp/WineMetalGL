#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
library="$project_root/build/release/metalsharp-opengl.dylib"

if test ! -f "$library"; then
    "$project_root/scripts/build.sh"
fi

probe_root=$(mktemp -d "${TMPDIR:-/tmp}/winemetalgl-probe.XXXXXX")
cleanup()
{
    find "$probe_root" -depth -delete
}
trap cleanup EXIT HUP INT TERM

clang -arch arm64 \
    "$project_root/tests/native/glsl_translation_probe.c" \
    -o "$probe_root/glsl_translation_probe"

"$probe_root/glsl_translation_probe" "$library"
