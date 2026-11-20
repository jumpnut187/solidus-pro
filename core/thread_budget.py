from __future__ import annotations

import os
from typing import Optional

_process_budget: Optional[int] = None

def set_process_thread_budget(budget: Optional[int]) -> None:
    global _process_budget
    _process_budget = budget

def get_process_thread_budget() -> Optional[int]:
    return _process_budget

def resolve_thread_budget() -> Optional[int]:
    if _process_budget is not None:
        return _process_budget

    raw_value = os.getenv("CASTSIM_THREAD_BUDGET", "").strip()
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value > 0 else None
