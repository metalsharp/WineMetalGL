#!/bin/sh
set -eu

if test "$#" -ne 1; then
    printf '%s\n' "usage: $0 /absolute/path/to/staged-release" >&2
    exit 2
fi

stage=$1
sidecar="$stage/artifacts/host-x86_64/metalsharp-opengl.dylib"

for path in \
    "$sidecar" \
    "$stage/artifacts/guest/i386/opengl32.dll" \
    "$stage/artifacts/guest/x86_64/opengl32.dll" \
    "$stage/artifacts/wine-driver/winemac.so" \
    "$stage/artifacts/wine-driver/win32u.so" \
    "$stage/artifacts/wine-driver/opengl32.so" \
    "$stage/api-coverage.json" \
    "$stage/SHA256SUMS"; do
    test -f "$path" || { printf 'missing staged artifact: %s\n' "$path" >&2; exit 1; }
done

for path in \
    "$sidecar" \
    "$stage/artifacts/wine-driver/winemac.so" \
    "$stage/artifacts/wine-driver/win32u.so" \
    "$stage/artifacts/wine-driver/opengl32.so"; do
    arch=$(lipo -archs "$path")
    test "$arch" = x86_64 || { printf 'unexpected Mach-O architecture %s: %s\n' "$arch" "$path" >&2; exit 1; }
done

minos=$(otool -l "$sidecar" | awk '/LC_BUILD_VERSION/{found=1} found && /minos/{print $2; exit}')
test "$minos" = 15.0 || { printf 'unexpected sidecar deployment target: %s\n' "$minos" >&2; exit 1; }

file "$stage/artifacts/guest/i386/opengl32.dll" | grep -q 'Intel 80386'
file "$stage/artifacts/guest/x86_64/opengl32.dll" | grep -q 'x86-64'

python3 -m json.tool "$stage/api-coverage.json" >/dev/null
(
    cd "$stage"
    shasum -a 256 -c SHA256SUMS >/dev/null
)

if grep -R -i -n 'VKMT' "$stage/README.md" "$stage/api-coverage.json" "$stage/RESULTS.md" 2>/dev/null; then
    printf '%s\n' 'staged standalone metadata contains VKMT-specific naming' >&2
    exit 1
fi

printf '%s\n' 'WINEMETALGL_RELEASE_AUDIT_OK'
printf 'stage=%s\n' "$stage"
printf 'sidecar_arch=%s\n' "$(lipo -archs "$sidecar")"
printf 'sidecar_minos=%s\n' "$minos"
