"""Help text for every result type, shared by the desktop and web apps.

``core/result_help.json`` is the single source: one short paragraph per result
mode id (what the field shows, its units, how to read high values, how it is
computed, and its default threshold). The desktop reads it through this module;
the web imports the same file at build time
(``web_frontend/src/lib/viewState.ts`` ``RESULT_MODE_HELP``). Mode ids are the
web's ``RESULT_MODE_IDS`` and the desktop's ``_current_vis_mode`` values.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict

RESULT_HELP_PATH = Path(__file__).with_name("result_help.json")


@lru_cache(maxsize=1)
def load_result_help() -> Dict[str, str]:
    """Mode id -> help text, read once from ``result_help.json``."""
    with RESULT_HELP_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(mode): str(text) for mode, text in data.items()}


def result_help_text(mode_id: str) -> str:
    """Help text for ``mode_id``; empty string for an unknown mode."""
    return load_result_help().get(mode_id, "")
