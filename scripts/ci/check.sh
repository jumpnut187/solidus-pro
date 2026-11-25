set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PYTHON:-python}"

case "${1:-}" in
models)
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
    "$PY" -m mypy --config-file mypy.ini
    ;;
tests)
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
    export MPLBACKEND="${MPLBACKEND:-Agg}"
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
