set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "${1:-}" != "--python" ]]; then
    SUDO=""
    if [[ "$(id -u)" != "0" ]]; then SUDO="sudo"; fi
    $SUDO apt-get update -q
    $SUDO apt-get install -y -q --no-install-recommends \
        libgl1 libglx-mesa0 libglu1-mesa libegl1 \
        libxkbcommon0 libxkbcommon-x11-0 libfontconfig1 libdbus-1-3 \
        libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0
fi

python --version
python -m pip install --disable-pip-version-check \
    -r requirements-ci.txt -c constraints-ci.txt
