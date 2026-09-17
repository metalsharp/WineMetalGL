#!/bin/sh
set -eu

if test "$#" -ne 1; then
    printf '%s\n' "usage: $0 /absolute/path/to/wine-build" >&2
    exit 2
fi

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
wine_build=$1
source_library="$project_root/build/release/metalsharp-opengl.dylib"
config_file="$project_root/config/winemetalgl.conf"
destination_dir="$wine_build/dlls/winemac.drv"
destination="$destination_dir/metalsharp-opengl.dylib"
config_destination="$destination_dir/winemetalgl.conf"

test -d "$destination_dir"
test -f "$source_library"
test -f "$config_file"
test "$(lipo -archs "$source_library")" = "x86_64"
test "$(otool -l "$source_library" | awk '/LC_BUILD_VERSION/{found=1} found && /minos/{print $2; exit}')" = "15.0"

install -m 755 "$source_library" "$destination"
install -m 644 "$config_file" "$config_destination"
codesign --force --sign - "$destination"

printf '%s\n' "Staged: $destination"
printf '%s\n' "Staged: $config_destination"
