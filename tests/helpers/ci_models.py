from __future__ import annotations

import os

CI_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ci_models")

def ci_model(name: str) -> str:
    return os.path.join(CI_MODELS_DIR, name)
