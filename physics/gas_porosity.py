"""
Gas Porosity Analysis — Post-processing module for Solidus Pro.

Computes a gas porosity risk score (0-1) per voxel based on mold gas
generation and entrapment.  Three separable visualization layers are
produced:

  gas_generation   — raw SGI field (source + exponential decay)
  gas_entrapment   — E field (solidification-time-based)
  gas_porosity_risk— combined R_gas field

All Numba-JIT kernels use parallel=True for O(N) performance.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numba import jit, prange

# ---------------------------------------------------------------------------
# Surface-type weighting for gas generation sources
# ---------------------------------------------------------------------------
SURFACE_TYPE_WEIGHTS: dict = {
    "core": 1.0,
    "mold_wall": 0.6,
    "open_riser": 0.2,
    "chill": 0.0,
}

# Mapping from part-type integer to surface-type key string
# 0 = cast -> mold_wall,  1 = riser -> open_riser,
# 2 = chill -> chill, 3 = riser_post -> open_riser,
# 4 = core -> core
PART_TYPE_SURFACE_MAP: dict = {
    0: "mold_wall",
    1: "open_riser",
    2: "chill",
    3: "open_riser",
    4: "core",
}

# Binder decomposition gas yields (cm³ of gas per gram of binder at STP)
BINDER_GAS_YIELDS: dict = {
    "phenolic_urethane": 12.0,
    "furan": 10.0,
    "phenolic_ester": 8.0,
    "sodium_silicate": 2.0,
    "clay_green_sand": 3.0,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_surface_gas_generation(
    surface_indices: np.ndarray,
    voxel_part_types: Optional[np.ndarray],
    local_solidification_times: np.ndarray,
    *,
    humidity: float = 0.5,
    binder_type: str = "phenolic_urethane",
    binder_content: float = 1.5,
    core_venting_effectiveness: float = 0.5,
) -> np.ndarray:
    """
    Compute SGI₀ (surface gas generation index) for each surface voxel.

    SGI₀ integrates binder decomposition + moisture gas over the local
    solidification time, weighted by surface type.

    Parameters
    ----------
    surface_indices : (S,) int array
        Indices of surface voxels in the full mesh.
    voxel_part_types : (N,) int array or None
        Part-type label per voxel (0=cast, 1=riser, 2=chill, 3=riser_post, 4=core).
        If None, all surfaces are treated as mold_wall.
    local_solidification_times : (N,) float array
        Local solidification time (liquidus -> solidus) per voxel in seconds.
    humidity : float
        Ambient relative humidity 0-1 (default 0.5).
    binder_type : str
        Binder chemistry key (see BINDER_GAS_YIELDS).
    binder_content : float
        Weight-percent binder in the sand mix (default 1.5 %).
    core_venting_effectiveness : float
        0-1 factor for how well core gas can vent (1=perfect venting, 0=no venting,
        effective gas kept low when venting is good).

    Returns
    -------
    sgi0 : (S,) float32 array
        Per-surface-voxel SGI₀ values (0-1 range).
    """
    n_voxels = len(local_solidification_times)
    if voxel_part_types is None:
        voxel_part_types = np.zeros(n_voxels, dtype=np.int32)
    surface_types = _classify_surface_types(
        surface_indices, np.asarray(voxel_part_types, dtype=np.int32)
    )

    gas_yield = BINDER_GAS_YIELDS.get(binder_type, 10.0)
    # base gas generation rate: binder decomposition + moisture
    # moisture contribution: humidity * 2.0 cm³/g equivalent
    base_rate = binder_content * gas_yield + humidity * 2.0

    # normalize to a 0-1 target using reference: ~20 cm³/g yields max SGI₀ ≈ 1.0
    reference_yield = 20.0
    source_scale = min(base_rate / reference_yield, 2.0)

    surface_st = local_solidification_times[surface_indices]
    max_st = float(np.max(surface_st)) if surface_st.size > 0 else 1.0
    if max_st <= 0:
        max_st = 1.0

    time_factor = np.clip(surface_st / max_st, 0.0, 1.0).astype(np.float32)

    sgi0 = np.zeros(len(surface_indices), dtype=np.float32)
    for idx, stype in enumerate(surface_types):
        weight = SURFACE_TYPE_WEIGHTS.get(stype, 0.6)
        sgi0[idx] = source_scale * weight * time_factor[idx]

    if stype == "core":
        sgi0[sgi0 > 0] = np.maximum(
            sgi0[surface_types == "core"] * (1.0 - core_venting_effectiveness), 0.0
        )
        # Re-apply: core venting reduces effective gas retained
        core_mask = surface_types == "core"
        if np.any(core_mask):
            sgi0_core = source_scale * 1.0 * time_factor[core_mask]
            sgi0_core *= (1.0 - core_venting_effectiveness)
            sgi0[core_mask] = sgi0_core.astype(np.float32)

    return np.clip(sgi0, 0.0, 1.0).astype(np.float32)


def _classify_surface_types(
    surface_indices: np.ndarray,
    voxel_part_types: np.ndarray,
) -> np.ndarray:
    """Map part-type integers to surface-type strings."""
    types = np.empty(len(surface_indices), dtype=object)
    for i, si in enumerate(surface_indices):
        pt = int(voxel_part_types[si])
        types[i] = PART_TYPE_SURFACE_MAP.get(pt, "mold_wall")
    return types


def propagate_gas_to_interior(
    voxel_centers: np.ndarray,
    surface_indices: np.ndarray,
    surface_sgi0: np.ndarray,
    *,
    gas_penetration_depth: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate surface gas generation to all voxels via exponential distance decay.

    Uses scipy's KDTree for O(N log S) nearest-surface queries.

    Parameters
    ----------
    voxel_centers : (N, 3) float array
        Coordinates of every voxel center (inches).
    surface_indices : (S,) int array
        Indices of surface voxels.
    surface_sgi0 : (S,) float32 array
        Per-surface-voxel SGI₀ values.
    gas_penetration_depth : float
        λ — characteristic penetration depth (inches, default 3.0 in ≈ 76 mm).

    Returns
    -------
    gas_generation : (N,) float32 array
        SGI(d) = SGI₀ · e^(-d/λ)  for every voxel.
    nearest_surface_map : (N,) int32 array
        Index into surface_indices of the nearest surface voxel for each voxel.
    """
    from scipy.spatial import cKDTree

    surface_centers = np.asarray(voxel_centers[surface_indices], dtype=np.float64)
    surface_sgi0 = np.asarray(surface_sgi0, dtype=np.float32)

    tree = cKDTree(surface_centers)
    distances, nearest_idx = tree.query(
        np.asarray(voxel_centers, dtype=np.float64),
        k=1,
    )

    distances = np.asarray(distances, dtype=np.float32)
    nearest_idx = np.asarray(nearest_idx, dtype=np.int32)
    nearest_sgi0 = surface_sgi0[nearest_idx]

    gas_generation = _propagate_sgi_numba(
        distances, nearest_sgi0, np.float32(gas_penetration_depth)
    )
    return gas_generation, nearest_idx


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _propagate_sgi_numba(
    distances: np.ndarray,
    nearest_sgi0: np.ndarray,
    penetration_depth: float,
) -> np.ndarray:
    n = distances.shape[0]
    sgi = np.zeros(n, dtype=np.float32)
    inv_lambda = 1.0 / max(penetration_depth, 1e-6)
    for i in prange(n):
        sgi[i] = nearest_sgi0[i] * np.exp(-distances[i] * inv_lambda)
    return sgi


def compute_combined_risk(
    gas_generation: np.ndarray,
    local_solidification_time: np.ndarray,
    niyama_risk: Optional[np.ndarray],
    alloy_gas_sensitivity: float = 0.5,
    gas_entrapment_exponent: float = 2.0,
    niyama_coupling_alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute entrapment factor, Niyama coupling, and final combined risk.

    Returns
    -------
    gas_entrapment : (N,) float32    E(i) = (local_st / max_st)^n
    niyama_coupling : (N,) float32   F_niyama(i)
    gas_porosity_risk : (N,) float32 R_gas(i)
    """
    n = gas_generation.shape[0]
    if niyama_risk is None or len(niyama_risk) != n:
        niyama_risk = np.full(n, 0.0, dtype=np.float32)
    else:
        niyama_risk = np.asarray(niyama_risk, dtype=np.float32)

    return _compute_combined_risk_numba(
        np.asarray(gas_generation, dtype=np.float32),
        np.asarray(local_solidification_time, dtype=np.float32),
        niyama_risk,
        np.float32(alloy_gas_sensitivity),
        np.float32(gas_entrapment_exponent),
        np.float32(niyama_coupling_alpha),
    )


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _compute_combined_risk_numba(
    gas_generation: np.ndarray,
    local_solidification_time: np.ndarray,
    niyama_risk: np.ndarray,
    alloy_gas_sensitivity: np.float32,
    gas_entrapment_exponent: np.float32,
    niyama_coupling_alpha: np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = gas_generation.shape[0]
    entrapment = np.zeros(n, dtype=np.float32)
    niyama_coupling = np.ones(n, dtype=np.float32)
    risk = np.zeros(n, dtype=np.float32)

    max_st = np.max(local_solidification_time)
    if max_st <= 0.0:
        max_st = 1.0

    for i in prange(n):
        st = max(local_solidification_time[i], 0.0)
        entrapment[i] = (st / max_st) ** gas_entrapment_exponent

        n_norm = max(0.0, min(1.0, 1.0 - niyama_risk[i]))
        f_niyama = 1.0 + niyama_coupling_alpha * (1.0 - n_norm)
        niyama_coupling[i] = min(f_niyama, 2.0)

        risk[i] = (
            gas_generation[i]
            * entrapment[i]
            * alloy_gas_sensitivity
            * niyama_coupling[i]
        )
        if risk[i] < 0.0:
            risk[i] = 0.0
        if risk[i] > 1.0:
            risk[i] = 1.0

    return entrapment, niyama_coupling, risk


def compute_hotspots(
    gas_porosity_risk: np.ndarray,
    threshold: float = 0.65,
) -> np.ndarray:
    """Threshold the continuous risk field to produce discrete hotspot markers."""
    return np.asarray(gas_porosity_risk >= threshold, dtype=bool)
