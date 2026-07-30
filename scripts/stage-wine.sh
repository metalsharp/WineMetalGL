#!/bin/sh
set -eu

if test "$#" -ne 1; then
    printf '%s\n' "usage: $0 /absolute/path/to/wine-build" >&2
    exit 2
fi

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
wine_build=$1
source_library="$project_root/build/release/metalsharp-opengl.dylib"
destination_dir="$wine_build/dlls/winemac.drv"
destination="$destination_dir/metalsharp-opengl.dylib"

test -d "$destination_dir"
test -f "$source_library"
test "$(lipo -archs "$source_library")" = "arm64"

install -m 755 "$source_library" "$destination"
codesign --force --sign - "$destination"

printf '%s\n' "Staged: $destination"
