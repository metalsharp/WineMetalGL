#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
library=${1:-$project_root/build/release/metalsharp-opengl.dylib}
probe_root=$(mktemp -d "${TMPDIR:-/tmp}/winemetalgl-native.XXXXXX")
cleanup() { rm -rf "$probe_root"; }
trap cleanup EXIT HUP INT TERM

if test ! -f "$library"; then
    cmake --preset release >/dev/null
    cmake --build --preset release -j"${JOBS:-8}" >/dev/null
fi
test "$(/usr/bin/lipo -archs "$library")" = x86_64
clang -arch x86_64 "$project_root/tests/native/glsl_translation_probe.c" -o "$probe_root/probe"
test "$(/usr/bin/lipo -archs "$probe_root/probe")" = x86_64
"$probe_root/probe" "$library"
printf '%s\n' WINEMETALGL_NATIVE_TRANSLATION_OK
