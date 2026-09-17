#!/bin/sh
set -eu

if test "$#" -ne 4; then
    printf '%s\n' "usage: $0 /absolute/path/to/wine-runtime /absolute/path/to/glcts.exe /absolute/path/to/shard.txt /absolute/path/to/output.qpa" >&2
    exit 2
fi

runtime=$1
cts=$2
shard=$3
output=$4

case "$runtime" in /*) ;; *) printf '%s\n' 'runtime must be an absolute path' >&2; exit 2 ;; esac
case "$cts" in /*) ;; *) printf '%s\n' 'glcts executable must be an absolute path' >&2; exit 2 ;; esac
case "$shard" in /*) ;; *) printf '%s\n' 'shard file must be an absolute path' >&2; exit 2 ;; esac
case "$output" in /*) ;; *) printf '%s\n' 'output must be an absolute path' >&2; exit 2 ;; esac

test -x "$runtime/bin/wine" || { printf 'missing Wine executable: %s\n' "$runtime/bin/wine" >&2; exit 1; }
test -f "$cts" || { printf 'missing CTS executable: %s\n' "$cts" >&2; exit 1; }
test -f "$shard" || { printf 'missing shard file: %s\n' "$shard" >&2; exit 1; }
test ! -e "$output" || { printf 'output already exists: %s\n' "$output" >&2; exit 1; }

patterns=$(awk 'NF && $1 !~ /^#/ { printf "%s%s", separator, $1; separator="," }' "$shard")
test -n "$patterns" || { printf 'shard is empty: %s\n' "$shard" >&2; exit 1; }

prefix=$(mktemp -d "${TMPDIR:-/tmp}/winemetalgl-gl33-shard-prefix.XXXXXX")
trap 'rm -rf "$prefix" 2>/dev/null || true' EXIT INT TERM HUP
cts_dir=$(CDPATH= cd -- "$(dirname -- "$cts")" && pwd)
export WINEPREFIX="$prefix" WINEARCH=wow64 WINEMETALGL=1 WINEMETALGL_EXPERIMENTAL=1
export WINEDEBUG="${WINEDEBUG:--all}"
export PATH="$runtime/bin:$PATH"
export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-$runtime/lib}"

"$runtime/bin/wine" wineboot -u >/dev/null 2>&1
cd "$cts_dir"
set +e
"$runtime/bin/wine" "$cts" \
    --deqp-case="$patterns" \
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
print("shard_cases=%d pass=%d fail=%d not_supported=%d internal_error=%d" % (
    len(results), counts["Pass"], counts["Fail"], counts["NotSupported"], counts["InternalError"]))
if not results or counts["Fail"] or counts["InternalError"]:
    failures = [(case, status) for case, status in sorted(results.items()) if status not in ("Pass", "NotSupported")]
    for case, status in failures[:20]: print("not_passed=%s [%s]" % (case, status))
    if len(failures) > 20: print("not_passed_remaining=%d" % (len(failures) - 20))
    raise SystemExit(1)
PY
validation_status=$?
if test "$run_status" -ne 0 || test "$validation_status" -ne 0; then
    printf 'GL 3.3 shard failed; QPA retained: %s\n' "$output" >&2
    exit 1
fi
printf 'GL 3.3 shard passed: %s\n' "$shard"
