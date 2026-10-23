"""Defect-region extraction for shrink/porosity pockets.

The desktop Shrink and Microporosity results are per-voxel scalar fields
(``SolidificationEngine.get_shrink_display_intensity`` and
``get_microporosity_display_intensity``). For the
inspector feature we need to recover the connected defect pocket
under the active threshold so we can show the user the volume of
the pocket they clicked on.

This module operates on the voxel mesh's neighbor graph so the
pocket boundaries follow the same canonical adjacency that the
solidification kernels use. The mesh stores up to 26 neighbors per
voxel with the canonical layout documented in
``core/voxel_mesh.py:_canonicalize_neighbor_order``:

    0-5   face neighbors:   +X, -X, +Y, -Y, +Z, -Z
    6-17  edge neighbors:   combinations of two axes
    18-25 corner neighbors: combinations of three axes

6-connectivity uses slots 0-5 (face-adjacent only) so two defect
volumes that touch only at a corner remain separate pockets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


MIXED_DEFECT_TOLERANCE = 0.05


@dataclass(frozen=True)
class DefectPocket:
    """A connected shrink/porosity pocket above the active threshold.

    Attributes:
        id: Monotonically increasing pocket identifier (per call).
        voxel_indices: Global voxel indices belonging to the pocket.
        anchor_voxel_index: The voxel the user clicked on.
        anchor_position: World position of the anchor voxel.
        centroid: Mean world position of the pocket voxels.
        bounds: ``((x_min, x_max), (y_min, y_max), (z_min, z_max))``.
        geometric_volume_in3: Sum of per-voxel volumes in cubic inches.
        voxel_count: Number of voxels in the pocket.
        max_shrinkage_risk: Maximum shrinkage probability in the pocket.
        max_porosity_risk: Maximum porosity probability in the pocket.
        mean_display_risk: Mean display intensity across the pocket.
        dominant_defect_type: ``"shrinkage"``, ``"porosity"``, or ``"mixed"``.
    """

    id: int
    voxel_indices: np.ndarray
    anchor_voxel_index: int
    anchor_position: Tuple[float, float, float]
    centroid: Tuple[float, float, float]
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    geometric_volume_in3: float
    voxel_count: int
    max_shrinkage_risk: float
    max_porosity_risk: float
    mean_display_risk: float
    dominant_defect_type: str


def compute_merged_shrink_display_intensity(
    shrinkage_probability: Optional[np.ndarray],
    porosity_probability: Optional[np.ndarray],
    voxel_part_types: Optional[np.ndarray],
    chill_part_type: int = 2,
) -> np.ndarray:
    """Build the merged max(shrink, microporosity) intensity.

    Mirrors ``SolidificationEngine.get_shrinkage_display_intensity`` (the field
    the old "Shrinkage/Porosity" view showed). The desktop inspector now takes
    the Shrink or Microporosity display array directly.
    CHILL and CORE voxels are forced to zero so mold material doesn't pollute
    the merged field.
    """
    if shrinkage_probability is None and porosity_probability is None:
        return np.zeros(0, dtype=np.float32)

    if shrinkage_probability is None:
        shrink_arr = np.zeros(
            int(np.asarray(porosity_probability).shape[0]),
            dtype=np.float32,
        )
    else:
        shrink_arr = np.asarray(shrinkage_probability, dtype=np.float32)

    if porosity_probability is None:
        poro_arr = np.zeros(shrink_arr.shape, dtype=np.float32)
    else:
        poro_arr = np.asarray(porosity_probability, dtype=np.float32)

    if shrink_arr.shape != poro_arr.shape:
        n = min(shrink_arr.shape[0], poro_arr.shape[0])
        shrink_arr = shrink_arr[:n]
        poro_arr = poro_arr[:n]

    display = np.maximum(shrink_arr, poro_arr).astype(np.float32, copy=True)

    if voxel_part_types is not None and len(voxel_part_types) == display.shape[0]:
        part_types = np.asarray(voxel_part_types, dtype=np.int32)
        chill_mask = part_types == int(chill_part_type)
        if np.any(chill_mask):
            display[chill_mask] = 0.0

    return np.clip(display, 0.0, 1.0).astype(np.float32, copy=False)


def _resolve_connectivity_slot_count(connectivity: int) -> int:
    """Return the neighbor slot count for the requested connectivity."""
    if connectivity == 6:
        return 6
    if connectivity == 18:
        return 18
    if connectivity == 26:
        return 26
    raise ValueError(
        f"connectivity must be 6, 18, or 26 (got {connectivity!r})"
    )


def extract_pocket_at(
    voxel_index: int,
    display_intensity: np.ndarray,
    voxel_neighbors: np.ndarray,
    voxel_centers: np.ndarray,
    voxel_volumes: np.ndarray,
    shrinkage_risk: Optional[np.ndarray],
    porosity_risk: Optional[np.ndarray],
    threshold: float,
    connectivity: int = 6,
    pocket_id: int = 0,
) -> Optional[DefectPocket]:
    """Extract the connected shrink/porosity pocket that contains ``voxel_index``.

    Args:
        voxel_index: Global voxel index that the user clicked on.
        display_intensity: Per-voxel merged shrink/porosity intensity.
        voxel_neighbors: ``(N, 26)`` int32 neighbor table (see module docstring).
        voxel_centers: ``(N, 3)`` float array of voxel center positions.
        voxel_volumes: ``(N,)`` float array of per-voxel volumes in cubic inches.
        shrinkage_risk: Optional per-voxel shrinkage probability (cast to float32).
        porosity_risk: Optional per-voxel porosity probability (cast to float32).
        threshold: Minimum display intensity to retain a voxel in the pocket.
        connectivity: ``6`` (face), ``18`` (face+edge), or ``26`` (face+edge+corner).
        pocket_id: Identifier stamped onto the returned ``DefectPocket``.

    Returns:
        The connected pocket, or ``None`` if the anchor voxel is below
        the threshold or the arrays are inconsistent.
    """
    display_intensity = np.asarray(display_intensity, dtype=np.float32)
    voxel_centers = np.asarray(voxel_centers, dtype=np.float64)
    voxel_volumes = np.asarray(voxel_volumes, dtype=np.float32)
    voxel_neighbors = np.asarray(voxel_neighbors, dtype=np.int32)

    n_voxels = display_intensity.shape[0]
    if voxel_centers.shape[0] != n_voxels or voxel_volumes.shape[0] != n_voxels:
        return None
    if voxel_neighbors.ndim != 2 or voxel_neighbors.shape[1] < 6:
        return None
    if voxel_neighbors.shape[0] != n_voxels:
        return None

    if voxel_index < 0 or voxel_index >= n_voxels:
        return None
    if not np.isfinite(display_intensity[voxel_index]):
        return None
    if float(display_intensity[voxel_index]) < float(threshold):
        return None

    slot_count = _resolve_connectivity_slot_count(connectivity)

    visited = np.zeros(n_voxels, dtype=bool)
    visited[voxel_index] = True
    queue: list[int] = [int(voxel_index)]
    head = 0

    while head < len(queue):
        i = queue[head]
        head += 1
        row = voxel_neighbors[i]
        for slot in range(slot_count):
            n = int(row[slot])
            if n < 0 or n >= n_voxels or visited[n]:
                continue
            value = display_intensity[n]
            if not np.isfinite(value) or float(value) < float(threshold):
                continue
            visited[n] = True
            queue.append(n)

    indices = np.asarray(queue, dtype=np.int32)
    pts = voxel_centers[indices]
    volumes = voxel_volumes[indices]

    geometric_volume_in3 = float(volumes.sum())

    shrink_vals = (
        np.asarray(shrinkage_risk, dtype=np.float32)[indices]
        if shrinkage_risk is not None
        else np.zeros(indices.shape[0], dtype=np.float32)
    )
    poro_vals = (
        np.asarray(porosity_risk, dtype=np.float32)[indices]
        if porosity_risk is not None
        else np.zeros(indices.shape[0], dtype=np.float32)
    )
    display_vals = display_intensity[indices]

    max_shrink = float(np.max(shrink_vals)) if shrink_vals.size else 0.0
    max_poro = float(np.max(poro_vals)) if poro_vals.size else 0.0
    mean_display = float(np.mean(display_vals)) if display_vals.size else 0.0

    if max_shrink <= 0.0 and max_poro <= 0.0:
        dominant = "mixed"
    elif abs(max_shrink - max_poro) <= MIXED_DEFECT_TOLERANCE:
        dominant = "mixed"
    elif max_shrink > max_poro:
        dominant = "shrinkage"
    else:
        dominant = "porosity"

    centroid = (
        float(pts[:, 0].mean()),
        float(pts[:, 1].mean()),
        float(pts[:, 2].mean()),
    )
    bounds = (
        (float(pts[:, 0].min()), float(pts[:, 0].max())),
        (float(pts[:, 1].min()), float(pts[:, 1].max())),
        (float(pts[:, 2].min()), float(pts[:, 2].max())),
    )
    anchor_position = (
        float(voxel_centers[voxel_index, 0]),
        float(voxel_centers[voxel_index, 1]),
        float(voxel_centers[voxel_index, 2]),
    )

    return DefectPocket(
        id=int(pocket_id),
        voxel_indices=indices,
        anchor_voxel_index=int(voxel_index),
        anchor_position=anchor_position,
        centroid=centroid,
        bounds=bounds,
        geometric_volume_in3=geometric_volume_in3,
        voxel_count=int(indices.shape[0]),
        max_shrinkage_risk=max_shrink,
        max_porosity_risk=max_poro,
        mean_display_risk=mean_display,
        dominant_defect_type=dominant,
    )


def format_pocket_volume_label(pocket: DefectPocket) -> str:
    """Return the leader text for a defect pocket."""
    return f"Shrink pocket: {pocket.geometric_volume_in3:.3f} in^3"
