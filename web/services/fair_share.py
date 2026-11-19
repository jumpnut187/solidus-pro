"""
Fair-share CPU and memory policy for concurrent web simulations.

Pure functions here are unit-testable without spawning processes. The web
queue manager uses them to divide CPU cores evenly among running simulations
and to refuse (queue) new simulations when RAM headroom is insufficient.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from config.hardware_config import hardware
from core.thread_budget import set_process_thread_budget


def resolve_usable_cores(logical_cores: int, reserved_cores: int) -> int:
    """Total logical cores minus those reserved for the OS/web server."""
    return max(1, max(1, logical_cores) - max(0, reserved_cores))


def compute_thread_share(usable_cores: int, running: int) -> int:
    """Divide usable cores evenly among running simulations."""
    return max(1, max(1, usable_cores) // max(1, running))


def can_start_simulation(
    *,
    running_count: int,
    max_concurrent: int,
    available_mb: float,
    active_memory_mb: float,
    reserve_mb: float,
    min_sim_memory_mb: float,
) -> bool:
    """Return True when a new simulation may start right now.

    The concurrency cap is lowered when RAM headroom cannot support another
    simulation, but a machine with no running simulations always allows one.
    """
    if running_count >= max_concurrent:
        return False
    if running_count == 0:
        return True
    headroom_mb = available_mb - max(0.0, active_memory_mb) - max(0.0, reserve_mb)
    return headroom_mb >= max(0.0, min_sim_memory_mb)


def make_thread_budget_provider(
    thread_budget: Any,
) -> Optional[Callable[[], Optional[int]]]:
    """Build a budget provider reading a shared multiprocessing value.

    The returned provider applies budget changes to this process (Numba
    thread pools, hardware cap, and the ``CASTSIM_THREAD_BUDGET`` env var
    inherited by spawned worker pools) and returns the effective budget.
    """
    if thread_budget is None:
        return None

    _UNSET = object()
    last_applied: Dict[str, Any] = {"value": _UNSET}

    def provider() -> Optional[int]:
        shared = int(thread_budget.value)
        budget = shared if shared > 0 else None
        if budget == last_applied["value"]:
            return budget
        last_applied["value"] = budget
        set_process_thread_budget(budget)
        hardware.set_thread_cap(budget)
        if budget is None:
            os.environ.pop("CASTSIM_THREAD_BUDGET", None)
        else:
            os.environ["CASTSIM_THREAD_BUDGET"] = str(budget)
        return budget

    return provider
