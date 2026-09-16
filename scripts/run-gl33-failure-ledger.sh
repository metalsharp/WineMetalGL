#!/bin/sh
set -eu

if test "$#" -ne 3; then
    printf '%s\n' "usage: $0 /absolute/path/to/wine-runtime /absolute/path/to/glcts.exe /absolute/path/to/output.qpa" >&2
    exit 2
fi

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
runtime=$1
cts=$2
output=$3
ledger="$project_root/tests/cts/gl33-failure-ledger.txt"

case "$runtime" in /*) ;; *) printf '%s\n' 'runtime must be an absolute path' >&2; exit 2 ;; esac
case "$cts" in /*) ;; *) printf '%s\n' 'glcts executable must be an absolute path' >&2; exit 2 ;; esac
case "$output" in /*) ;; *) printf '%s\n' 'output must be an absolute path' >&2; exit 2 ;; esac

test -x "$runtime/bin/wine" || { printf 'missing Wine executable: %s\n' "$runtime/bin/wine" >&2; exit 1; }
test -f "$cts" || { printf 'missing CTS executable: %s\n' "$cts" >&2; exit 1; }
test -f "$ledger" || { printf 'missing failure ledger: %s\n' "$ledger" >&2; exit 1; }

test ! -e "$output" || { printf 'output already exists: %s\n' "$output" >&2; exit 1; }

prefix=$(mktemp -d "${TMPDIR:-/tmp}/winemetalgl-gl33-ledger.XXXXXX")
trap 'rm -rf "$prefix" 2>/dev/null || true' EXIT INT TERM
patterns=$(paste -sd, "$ledger")
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

python3 - "$ledger" "$output" <<'PY'
import re
import sys

ledger, qpa = sys.argv[1:]
expected = [line.strip() for line in open(ledger, encoding="utf-8") if line.strip()]
text = open(qpa, "rb").read().decode("latin1")
results = {}
for match in re.finditer(r'<TestCaseResult CasePath="([^"]+)"[\s\S]*?</TestCaseResult>', text):
    status = re.search(r'<Result StatusCode="(\w+)"', match.group(0))
    if status:
        results[match.group(1)] = status.group(1)
missing = [case for case in expected if case not in results]
not_passed = [(case, results.get(case)) for case in expected if results.get(case) != "Pass"]
print("ledger_cases=%d observed=%d pass=%d failed=%d missing=%d" % (
    len(expected), len(results), sum(results.get(case) == "Pass" for case in expected),
    sum(results.get(case) == "Fail" for case in expected), len(missing)))
if missing or not_passed:
    if missing:
        print("missing:\n" + "\n".join(missing))
    if not_passed:
        print("not_passed:\n" + "\n".join("%s [%s]" % item for item in not_passed))
    sys.exit(1)
PY
validation_status=$?
if test "$run_status" -ne 0 || test "$validation_status" -ne 0; then exit 1; fi
printf 'GL 3.3 failure ledger passed: %s\n' "$output"
