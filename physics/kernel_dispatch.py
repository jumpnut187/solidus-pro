from __future__ import annotations

import os
import types
from typing import Any

try:
    from numba import get_num_threads, njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

SERIAL_KERNEL_BUDGET_NS = 60_000.0
SERIAL_KERNEL_BUDGET_PER_THREAD_NS = 2_500.0

FUSED_KERNELS = os.environ.get("CASTSIM_FUSED_KERNELS", "1").strip().lower() not in ("0", "false", "no", "off")

def fused_kernels_enabled() -> bool:
    return bool(FUSED_KERNELS and NUMBA_AVAILABLE)

def runs_serial(n_cells: int, cost_ns_per_cell: float) -> bool:
    if SERIAL_KERNEL_BUDGET_NS <= 0.0:
        return False
    threads = get_num_threads()
    if threads <= 1:
        return True
    budget = SERIAL_KERNEL_BUDGET_NS + SERIAL_KERNEL_BUDGET_PER_THREAD_NS * (threads - 2)
    return n_cells * cost_ns_per_cell <= budget

def serial_twin(kernel: Any) -> Any:
    if kernel.targetoptions.get("fastmath"):
        raise ValueError(f"{kernel.py_func.__name__} is compiled with fastmath; its serial twin may differ in the last bit")
    py_func = kernel.py_func
    twin = types.FunctionType(
        py_func.__code__,
        py_func.__globals__,
        py_func.__name__ + "_serial",
        py_func.__defaults__,
        py_func.__closure__,
    )
    twin.__qualname__ = py_func.__qualname__ + "_serial_numpy_errors"
    twin.__module__ = py_func.__module__
    twin.__doc__ = py_func.__doc__
    twin.__kwdefaults__ = py_func.__kwdefaults__
    return njit(cache=True, error_model="numpy")(twin)
