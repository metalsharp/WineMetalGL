#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

cmake --preset release -S "$project_root"
cmake --build --preset release

library="$project_root/build/release/metalsharp-opengl.dylib"
test -f "$library"
test "$(lipo -archs "$library")" = "x86_64"

printf '%s\n' "Built x86_64 sidecar: $library"
