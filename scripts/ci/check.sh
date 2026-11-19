#!/usr/bin/env bash
# CI checks, one subcommand per CI step so each shows up separately.
# CI-provider neutral; also runs locally from the repo root:
#
#   scripts/ci/check.sh models   no CAD file is tracked outside tests/ci_models/
#   scripts/ci/check.sh pins     every installed package is pinned in constraints-ci.txt
#   scripts/ci/check.sh syntax   byte-compile all first-party packages
#   scripts/ci/check.sh types    mypy over the mypy.ini allowlist
#   scripts/ci/check.sh tests    full unittest regression suite (must exit 0)
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PYTHON:-python}"

case "${1:-}" in
models)
    # CAD is ignored everywhere except the CI test models, but `git add -f`
    # gets past .gitignore. Fail on any tracked STL/STEP outside
    # tests/ci_models/, and on anything but .stl inside it.
    bad=$(git ls-files -z | tr '\0' '\n' \
        | grep -iE '\.(stl|step|stp)$' \
        | grep -vE '^tests/ci_models/[^/]+\.stl$' || true)
    if [ -n "$bad" ]; then
        echo "CAD files are only allowed as CI test models in tests/ci_models/*.stl:"
        echo "$bad" | sed 's/^/  /'
        echo "Untrack them: git rm --cached <file>"
        exit 1
    fi
    echo "CAD files tracked: only tests/ci_models/*.stl."
    ;;
pins)
    # A requirement added without regenerating constraints-ci.txt installs at
    # whatever version is newest that day. Fail instead, so the pins stay the
    # single source of versions.
    "$PY" - <<'EOF'
import re, subprocess, sys
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
pins = {}
for line in open("constraints-ci.txt"):
    line = line.split("#", 1)[0].strip()
    if "==" in line:
        name, ver = line.split("==", 1)
        pins[norm(name)] = ver
out = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze",
                      "--exclude-editable", "--disable-pip-version-check"],
                     check=True, capture_output=True, text=True).stdout
ignore = {"pip", "setuptools", "wheel"}
bad = []
for line in out.splitlines():
    name, _, ver = line.partition("==")
    n = norm(name)
    if n in ignore:
        continue
    if n not in pins:
        bad.append(f"{n}=={ver} is installed but not pinned")
    elif pins[n] != ver:
        bad.append(f"{n}=={ver} is installed but pinned to {pins[n]}")
if bad:
    print("constraints-ci.txt is out of date:")
    print("\n".join(f"  {b}" for b in sorted(bad)))
    print("Regenerate: venv/bin/python scripts/ci/update_constraints.py")
    sys.exit(1)
print(f"All installed packages match constraints-ci.txt ({len(pins)} pins).")
EOF
    ;;
syntax)
    "$PY" -m compileall -q config core physics ui web tests tools scripts
    ;;
types)
    # mypy.ini `files=` is an explicit allowlist; only those files are checked.
    "$PY" -m mypy --config-file mypy.ini
    ;;
tests)
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
    export MPLBACKEND="${MPLBACKEND:-Agg}"
    # -X faulthandler prints a Python traceback when a Numba kernel or a Qt
    # extension segfaults, instead of a bare exit 139. A heap-corrupting
    # kernel once made discovery exit 139 and hid eight failures for weeks.
    # The outer timeout sends SIGABRT before the CI job limit, which makes
    # faulthandler dump every thread's stack for a hung test.
    # Same discovery form as AGENT.md (`discover tests`): tests import as
    # top-level `test_*` modules, which tests/helpers/module_stubs.py expects.
    cmd=("$PY" -X faulthandler -m unittest discover tests --durations 25)
    if command -v timeout >/dev/null 2>&1; then
        timeout --signal=ABRT "${CI_TEST_TIMEOUT:-25m}" "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
    ;;
*)
    echo "usage: $0 {models|pins|syntax|types|tests}" >&2
    exit 2
    ;;
esac
