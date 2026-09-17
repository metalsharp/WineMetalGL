#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_root"

git diff --check
python3 - <<'PY'
from pathlib import Path
import subprocess
manifest = set(Path('MANIFEST.txt').read_text().splitlines())
tracked = subprocess.check_output(['git', 'ls-files'], text=True).splitlines()
missing = [p for p in tracked if p not in manifest and p not in ('.gitignore', 'MANIFEST.txt') and not (p.startswith('tests/cts/gl33-shards/failures-') or p == 'tests/cts/gl33-shards/unrun-full-tail.txt')]
if missing:
    raise SystemExit('manifest missing: ' + ', '.join(missing))
PY
python3 -m json.tool docs/api-coverage.json >/dev/null
shasum -a 256 -c SHA256SUMS >/dev/null
if grep -i -n 'VKMT' scripts/build.sh scripts/probe-native.sh scripts/stage-host-libs.sh scripts/stage-release.sh scripts/stage-wine.sh README.md CMakeLists.txt || grep -R -i -n 'VKMT' tests; then
    printf '%s\n' 'standalone source contains VKMT-specific naming' >&2
    exit 1
fi

if test -f build/release/metalsharp-opengl.dylib; then
    test "$(lipo -archs build/release/metalsharp-opengl.dylib)" = x86_64
    test "$(otool -l build/release/metalsharp-opengl.dylib | awk '/LC_BUILD_VERSION/{found=1} found && /minos/{print $2; exit}')" = 15.0
fi

wine_source=${WINEMETALGL_WINE_SOURCE:-}
if test -n "$wine_source"; then
    patch_file=${WINEMETALGL_WINE_PATCH:-}
    patch_dir=$project_root/patches/wine/wine-11.17
    check_dir=$(mktemp -d "${TMPDIR:-/tmp}/winemetalgl-patch-check.XXXXXX")
    cleanup() { git -C "$wine_source" worktree remove --force "$check_dir" >/dev/null 2>&1 || true; }
    trap cleanup EXIT INT TERM
    git -C "$wine_source" worktree add --detach "$check_dir" HEAD >/dev/null 2>&1
    if test -n "$patch_file"; then
        git -C "$check_dir" apply --check "$patch_file"
    else
        git -C "$check_dir" apply "$patch_dir/0001-winemetalgl-metal-surface-and-extension-parser.patch"
        git -C "$check_dir" apply --check "$patch_dir/0002-winemetalgl-default-config-and-feature-cap.patch"
    fi
fi

printf '%s\n' 'WINEMETALGL_SOURCE_AUDIT_OK'
