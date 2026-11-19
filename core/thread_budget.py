"""
Per-process CPU thread budget for concurrent simulation workers.

The web queue manager divides available CPU cores evenly among concurrent
simulations (fair-share). Each simulation worker process applies its share
through this module so that every Numba/kernel thread-count consumer in the
process honors the same cap. Desktop and single-sim workloads are unaffected
unless a budget is explicitly set.
"""

from __future__ import annotations

import os
from typing import Optional

_process_budget: Optional[int] = None


def set_process_thread_budget(budget: Optional[int]) -> None:
    """Set the process-local thread budget override (None clears it)."""
    global _process_budget
    _process_budget = budget


def get_process_thread_budget() -> Optional[int]:
    """Return the process-local thread budget override, if any."""
    return _process_budget


def resolve_thread_budget() -> Optional[int]:
    """Resolve the effective thread budget for this process.

    Precedence:
    1. Process-local override (set by web simulation workers).
    2. ``CASTSIM_THREAD_BUDGET`` environment variable (inherited by
       worker pools spawned by the process).
    3. ``None`` — no cap; use the default hardware-derived counts.
    """
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
