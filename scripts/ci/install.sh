#!/usr/bin/env bash
# CI install: system libraries for Qt/VTK, then the pinned Python set.
#
# CI-provider neutral. GitHub Actions calls it from .github/workflows/tests.yml;
# a GitLab job can call the same script. Assumes a Debian/Ubuntu image with
# glibc >= 2.34 (PySide6 6.11 ships manylinux_2_34 wheels only) and the
# target Python already on PATH as `python`.
#
#   scripts/ci/install.sh            system libraries + Python packages
#   scripts/ci/install.sh --python   Python packages only
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "${1:-}" != "--python" ]]; then
    SUDO=""
    if [[ "$(id -u)" != "0" ]]; then SUDO="sudo"; fi
    # Qt (offscreen platform) and VTK need GL/EGL and the xcb/xkb runtime
    # libraries even without a display.
    $SUDO apt-get update -q
    $SUDO apt-get install -y -q --no-install-recommends \
        libgl1 libglx-mesa0 libglu1-mesa libegl1 \
        libxkbcommon0 libxkbcommon-x11-0 libfontconfig1 libdbus-1-3 \
        libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0
fi

python --version
# torch/coremltools/onnx (requirements-ml.txt) are deliberately not installed;
# the deep-model tests skip when torch is absent.
python -m pip install --disable-pip-version-check \
    -r requirements-ci.txt -c constraints-ci.txt
