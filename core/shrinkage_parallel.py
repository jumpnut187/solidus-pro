"""Accelerated shrinkage helpers.

This module contains Numba kernels used by the shrinkage pipeline.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def _compute_shrinkage_chunk(
    base_risk: np.ndarray,
    chill_reduction: np.ndarray,
    riser_feeding_reduction: np.ndarray,
    feeding_penalty: np.ndarray,
    out: np.ndarray,
) -> None:
    """Compute final shrinkage risk for a chunk of voxels.

    Args:
        base_risk: Base risk values for voxels in the chunk.
        chill_reduction: Chill multipliers (0.15-1.0).
        riser_feeding_reduction: Riser feeding multipliers (0.10-1.0).
        feeding_penalty: Feeding penalty multipliers (typically 1.0-2.0).
        out: Output array to write final risk values into.
    """
    n_voxels = len(base_risk)
    for i in prange(n_voxels):
        value = (
            base_risk[i]
            * chill_reduction[i]
            * riser_feeding_reduction[i]
            * feeding_penalty[i]
        )
        if value < 0.0:
            out[i] = 0.0
        elif value > 1.0:
            out[i] = 1.0
        else:
            out[i] = value


def compute_shrinkage_chunk(
    base_risk: np.ndarray,
    chill_reduction: np.ndarray,
    riser_feeding_reduction: np.ndarray,
    feeding_penalty: np.ndarray,
) -> np.ndarray:
    """Numba-backed wrapper for chunk shrinkage computation."""
    base = np.ascontiguousarray(base_risk, dtype=np.float32)
    chill = np.ascontiguousarray(chill_reduction, dtype=np.float32)
    riser_feed = np.ascontiguousarray(riser_feeding_reduction, dtype=np.float32)
    penalty = np.ascontiguousarray(feeding_penalty, dtype=np.float32)
    out = np.empty(len(base), dtype=np.float32)
    _compute_shrinkage_chunk(base, chill, riser_feed, penalty, out)
    return out
