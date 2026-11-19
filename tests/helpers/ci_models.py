"""Paths to the CI test models in ``tests/ci_models/``.

These STL files exist only so the regression suite can run on a clean CI
checkout. They are the one place in the repo where CAD files are tracked
(``.gitignore`` ignores every other ``*.stl``/``*.step``/``*.stp``, and
``scripts/ci/check.sh models`` fails if one is committed anywhere else).
"""

from __future__ import annotations

import os

CI_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ci_models")


def ci_model(name: str) -> str:
    """Absolute path of a CI test model, e.g. ``ci_model("block_2inch.stl")``."""
    return os.path.join(CI_MODELS_DIR, name)
