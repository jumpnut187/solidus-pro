from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict

RESULT_HELP_PATH = Path(__file__).with_name("result_help.json")

@lru_cache(maxsize=1)
def load_result_help() -> Dict[str, str]:
    with RESULT_HELP_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(mode): str(text) for mode, text in data.items()}

def result_help_text(mode_id: str) -> str:
    return load_result_help().get(mode_id, "")
