#!/bin/sh
set -eu

if test "$#" -ne 4; then
    printf '%s\n' "usage: $0 /absolute/path/to/wine-runtime /absolute/path/to/glcts.exe /absolute/path/to/caselist.trie /absolute/path/to/output.qpa" >&2
    exit 2
fi

runtime=$1
cts=$2
caselist=$3
output=$4

case "$runtime" in /*) ;; *) printf '%s\n' 'runtime must be an absolute path' >&2; exit 2 ;; esac
case "$cts" in /*) ;; *) printf '%s\n' 'glcts executable must be an absolute path' >&2; exit 2 ;; esac
case "$caselist" in /*) ;; *) printf '%s\n' 'caselist must be an absolute path' >&2; exit 2 ;; esac
case "$output" in /*) ;; *) printf '%s\n' 'output must be an absolute path' >&2; exit 2 ;; esac

test -x "$runtime/bin/wine" || { printf 'missing Wine executable: %s\n' "$runtime/bin/wine" >&2; exit 1; }
test -f "$cts" || { printf 'missing CTS executable: %s\n' "$cts" >&2; exit 1; }
test -f "$caselist" || { printf 'missing trie caselist: %s\n' "$caselist" >&2; exit 1; }
test ! -e "$output" || { printf 'output already exists: %s\n' "$output" >&2; exit 1; }

prefix=$(mktemp -d "${TMPDIR:-/tmp}/winemetalgl-gl40-prefix.XXXXXX")
trap 'rm -rf "$prefix" 2>/dev/null || true' EXIT INT TERM HUP
cts_dir=$(CDPATH= cd -- "$(dirname -- "$cts")" && pwd)
host_lib_dir=${WINEMETALGL_HOST_LIB_DIR:-$runtime/lib}
export WINEPREFIX="$prefix" WINEARCH=wow64 WINEMETALGL=1
export WINEMETALGL_EXPERIMENTAL=1 WINEMETALGL_MAX_FEATURE_LEVEL=4.0 WINEMETALGL_GL40_COVERAGE=1
export WINEDEBUG="${WINEDEBUG:--all}"
export PATH="$runtime/bin:$PATH"
export DYLD_LIBRARY_PATH="$host_lib_dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

"$runtime/bin/wine" wineboot -u >/dev/null 2>&1
cd "$cts_dir"
set +e
"$runtime/bin/wine" "$cts" \
    --deqp-caselist-file="$caselist" \
    --deqp-terminate-on-device-lost=disable \
    --deqp-log-filename="$output"
run_status=$?
set -e

python3 - "$output" <<'PY'
import collections
import re
import sys

qpa = sys.argv[1]
text = open(qpa, "rb").read().decode("latin1")
results = {}
for match in re.finditer(r'<TestCaseResult CasePath="([^"]+)"[\s\S]*?</TestCaseResult>', text):
    status = re.search(r'<Result StatusCode="(\w+)"', match.group(0))
    if status:
        results[match.group(1)] = status.group(1)
counts = collections.Counter(results.values())
print("gl40_bounded_cases=%d pass=%d fail=%d not_supported=%d internal_error=%d" % (
    len(results), counts["Pass"], counts["Fail"], counts["NotSupported"], counts["InternalError"]))
if len(results) != 1000:
    print("expected exactly 1000 results from the single trie invocation", file=sys.stderr)
    raise SystemExit(1)
PY
parse_status=$?
set -e
if test "$run_status" -ne 0 || test "$parse_status" -ne 0; then
    printf 'bounded GL 4.0 run completed with failures; QPA retained: %s\n' "$output" >&2
    exit 1
fi
printf 'bounded GL 4.0 run completed: %s\n' "$output"
