"""Serial twins of the Numba parallel kernels, and the rule for choosing one.

A Numba parallel region has a fixed launch cost: the worker threads are woken
and joined whatever the array length. On the workqueue threading layer that is
0.07 ms (2 threads) to 0.11 ms (17 threads) per launch, measured on Apple
silicon. A cheap element-wise kernel does 13,000 cells in 0.02 ms on the
calling thread and in 0.11 ms through the pool, and a thermal step launches
about 16 kernels, so on small meshes the launches cost more than the work.

``serial_twin`` compiles the same Python function a second time without
``parallel=True``. ``runs_serial`` picks the twin when the whole serial run is
cheaper than one launch.

The twin gives the same bits as the parallel kernel only for kernels compiled
WITHOUT ``fastmath``: each cell is then plain IEEE arithmetic with no
reassociation, so the result cannot depend on how the loop is split or
vectorised. A ``fastmath`` kernel has no such guarantee and must not be
twinned.

Division by zero needs one more setting. Inside a ``prange`` loop Numba does
not raise: ``x / 0.0`` gives inf or NaN. A plain ``njit`` function raises
``ZeroDivisionError`` instead (``SolidificationModel.update(T, 0.0)`` divides
by ``dt``). The twins are compiled with ``error_model="numpy"``, which gives
the loop the parallel kernel's inf/NaN. The one difference left is a division
written OUTSIDE the loop: the parallel kernel raises there and the twin does
not. The twinned kernels guard those divisors or take them from callers that
do. ``tests/test_kernel_dispatch.py`` compares every twin with its parallel
kernel, division by zero inside the loop included.
"""

from __future__ import annotations

import os
import types
from typing import Any

try:
    from numba import get_num_threads, njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where numba is absent
    NUMBA_AVAILABLE = False

#: Launch cost of a parallel region (ns): the part every launch pays, and the
#: part each thread beyond two adds. Measured 65,000 at 2 threads, 70,000 at 4,
#: 81,000 at 8 and 107,000 at 17; both numbers are set a little below the fit,
#: so the twin is only taken where it wins. A budget of 0 turns the twins off.
SERIAL_KERNEL_BUDGET_NS = 60_000.0
SERIAL_KERNEL_BUDGET_PER_THREAD_NS = 2_500.0


#: The fused step kernels: work that the bit-for-bit path does in several
#: kernels and NumPy passes, done in one. They sum the same terms in a
#: different order and are compiled without ``fastmath``, so a run matches the
#: bit-for-bit path to rounding (temperatures to about 1e-11 degC) and not bit
#: for bit. ``CASTSIM_FUSED_KERNELS=0`` runs the bit-for-bit path instead,
#: which stays in the code as the reference
#: (``tests/test_fused_kernels.py`` compares the two).
FUSED_KERNELS = os.environ.get("CASTSIM_FUSED_KERNELS", "1").strip().lower() not in ("0", "false", "no", "off")


def fused_kernels_enabled() -> bool:
    """True when the fused step kernels are in use (Numba present and not switched off)."""
    return bool(FUSED_KERNELS and NUMBA_AVAILABLE)


def runs_serial(n_cells: int, cost_ns_per_cell: float) -> bool:
    """True when ``n_cells`` at ``cost_ns_per_cell`` is cheaper on the calling thread.

    The serial run is compared with the launch alone, as if the pool then did
    the work in no time, so a kernel sent to the calling thread never takes
    longer there. With one Numba thread the pool has nothing to offer.
    """
    if SERIAL_KERNEL_BUDGET_NS <= 0.0:
        return False
    threads = get_num_threads()
    if threads <= 1:
        return True
    budget = SERIAL_KERNEL_BUDGET_NS + SERIAL_KERNEL_BUDGET_PER_THREAD_NS * (threads - 2)
    return n_cells * cost_ns_per_cell <= budget


def serial_twin(kernel: Any) -> Any:
    """``kernel`` compiled again without the thread pool (``prange`` becomes ``range``).

    Only for kernels compiled without ``fastmath`` (see the module docstring);
    a ``fastmath`` kernel is refused. The twin is built from a copy of the
    function under its own name, so its on-disk cache entry cannot collide
    with the parallel kernel's.
    """
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
    # The on-disk cache is keyed by this name, not by the compile options.
    twin.__qualname__ = py_func.__qualname__ + "_serial_numpy_errors"
    twin.__module__ = py_func.__module__
    twin.__doc__ = py_func.__doc__
    twin.__kwdefaults__ = py_func.__kwdefaults__
    return njit(cache=True, error_model="numpy")(twin)
