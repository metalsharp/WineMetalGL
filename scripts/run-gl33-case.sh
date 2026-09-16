#!/bin/sh
set -eu

if test "$#" -lt 3 || test "$#" -gt 4; then
    printf '%s\n' "usage: $0 /absolute/path/to/wine-runtime /absolute/path/to/glcts.exe CASE [output.qpa]" >&2
    exit 2
fi

runtime=$1
cts=$2
case_name=$3
output=${4:-}

case "$runtime" in /*) ;; *) printf '%s\n' 'runtime must be an absolute path' >&2; exit 2 ;; esac
case "$cts" in /*) ;; *) printf '%s\n' 'glcts executable must be an absolute path' >&2; exit 2 ;; esac
case "$case_name" in KHR-GL33.*) ;; *) printf '%s\n' 'case must be a KHR-GL33 case path' >&2; exit 2 ;; esac

test -x "$runtime/bin/wine" || { printf 'missing Wine executable: %s\n' "$runtime/bin/wine" >&2; exit 1; }
test -f "$cts" || { printf 'missing CTS executable: %s\n' "$cts" >&2; exit 1; }

if test -n "$output"; then
    case "$output" in /*) ;; *) printf '%s\n' 'output must be an absolute path' >&2; exit 2 ;; esac
    test ! -e "$output" || { printf 'output already exists: %s\n' "$output" >&2; exit 1; }
    keep_output=1
else
    output=$(mktemp "${TMPDIR:-/tmp}/winemetalgl-gl33-case.XXXXXX.qpa")
    keep_output=0
fi

prefix=$(mktemp -d "${TMPDIR:-/tmp}/winemetalgl-gl33-case-prefix.XXXXXX")
cleanup() {
    status=$?
    rm -rf "$prefix" 2>/dev/null || true
    if test "$keep_output" -eq 0 && test "$status" -eq 0; then rm -f "$output"; fi
    exit "$status"
}
trap cleanup EXIT INT TERM HUP

cts_dir=$(CDPATH= cd -- "$(dirname -- "$cts")" && pwd)
export WINEPREFIX="$prefix" WINEARCH=wow64 WINEMETALGL=1 WINEMETALGL_EXPERIMENTAL=1
export WINEDEBUG="${WINEDEBUG:--all}"
export PATH="$runtime/bin:$PATH"
export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-$runtime/lib}"

"$runtime/bin/wine" wineboot -u >/dev/null 2>&1
cd "$cts_dir"
set +e
"$runtime/bin/wine" "$cts" \
    --deqp-case="$case_name" \
    --deqp-terminate-on-device-lost=disable \
    --deqp-log-filename="$output"
run_status=$?
set -e

python3 - "$case_name" "$output" <<'PY'
import re
import sys

case_name, qpa = sys.argv[1:]
text = open(qpa, "rb").read().decode("latin1")
match = re.search(r'<TestCaseResult CasePath="' + re.escape(case_name) + r'"[\s\S]*?</TestCaseResult>', text)
if not match:
    print("missing_case=%s" % case_name)
    raise SystemExit(1)
status = re.search(r'<Result StatusCode="(\w+)"', match.group(0))
actual = status.group(1) if status else "MissingStatus"
print("case=%s status=%s" % (case_name, actual))
if actual != "Pass":
    raise SystemExit(1)
PY
validation_status=$?
if test "$run_status" -ne 0 || test "$validation_status" -ne 0; then
    printf 'GL 3.3 case failed; QPA retained: %s\n' "$output" >&2
    exit 1
fi
printf 'GL 3.3 case passed: %s\n' "$case_name"
