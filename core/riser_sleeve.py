"""Which riser voxels stand in bare sand or in a no-heat-loss sleeve (``Part.riser_sleeve``).

Every riser part carries a sleeve choice, but the solver sees voxels, and a
voxel does not remember its part. :func:`riser_sleeve_masks` maps the choice
back onto the voxel mesh. Both thermal setups read the result
(``voxel_mesh.voxel_riser_bare``, ``voxel_mesh.voxel_riser_adiabatic``): the
explicit-mold coupling and the semi-infinite-sand boundary give a bare riser
the casting's metal-to-sand contact, a no-heat-loss riser no contact with the
sand at all, and every other riser the insulating sleeve's leak.

A riser is one connected component of riser voxels, and a component cools one
way. When it spans parts with different settings (two risers that touch), the
voxels inside each riser part vote and the choice with the most voxels wins. A
tie, and voxels inside no riser part (cut-cell slivers on the skin), count as
insulating, the default.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from core.part_manager import (
    PART_TYPE_RISER,
    RISER_SLEEVE_ADIABATIC,
    RISER_SLEEVE_CHOICES,
    RISER_SLEEVE_INSULATING,
    RISER_SLEEVE_NONE,
)


def riser_sleeve_masks(
    centers: np.ndarray,
    part_types: np.ndarray,
    neighbors: np.ndarray,
    riser_parts: Sequence[Tuple[trimesh.Trimesh, str]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Per-voxel bool masks ``(bare, adiabatic)`` of the riser components set to those sleeves.

    Args:
        centers: (N, 3) voxel centres, inches.
        part_types: (N,) voxel part codes.
        neighbors: (N, K) neighbor indices, -1 where there is none.
        riser_parts: ``(mesh, riser_sleeve)`` per riser part, as
            ``PartManager.get_riser_sleeve_parts`` returns them.

    Returns:
        ``(bare, adiabatic)``. ``bare`` is True on the voxels of every riser
        component set to no sleeve, ``adiabatic`` on those set to the
        no-heat-loss sleeve. Either is None when no riser component takes that
        choice (nothing to change).
    """
    special = (RISER_SLEEVE_NONE, RISER_SLEEVE_ADIABATIC)
    if not any(sleeve in special for _, sleeve in riser_parts):
        return None, None
    types = np.asarray(part_types)
    risers = np.flatnonzero(types == PART_TYPE_RISER)
    if len(risers) == 0:
        return None, None

    points = np.asarray(centers, dtype=np.float64)[risers]
    votes = {choice: np.zeros(len(risers), dtype=np.int64) for choice in RISER_SLEEVE_CHOICES}
    for mesh, sleeve in riser_parts:
        lo, hi = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
        near = np.flatnonzero(np.all((points >= lo) & (points <= hi), axis=1))
        if len(near) == 0:
            continue
        try:
            inside = near[np.asarray(mesh.contains(points[near]), dtype=bool)]
        except Exception:  # a mesh the ray test cannot answer for casts no vote
            continue
        votes[sleeve if sleeve in votes else RISER_SLEEVE_INSULATING][inside] += 1

    local = np.full(len(types), -1, dtype=np.int64)
    local[risers] = np.arange(len(risers))
    nbr = np.asarray(neighbors)[risers]
    rows, slots = np.nonzero(nbr >= 0)
    cols = local[nbr[rows, slots]]
    keep = cols >= 0
    graph = coo_matrix(
        (np.ones(int(keep.sum()), dtype=np.int8), (rows[keep], cols[keep])),
        shape=(len(risers), len(risers)),
    )
    n_components, component = connected_components(graph, directed=False)
    totals = {
        choice: np.bincount(component, weights=votes[choice], minlength=n_components)
        for choice in RISER_SLEEVE_CHOICES
    }

    masks = []
    for choice in special:
        # Strictly the most votes: a tie falls back to insulating.
        wins = np.ones(n_components, dtype=bool)
        for other in RISER_SLEEVE_CHOICES:
            if other != choice:
                wins &= totals[choice] > totals[other]
        mask = np.zeros(len(types), dtype=bool)
        mask[risers] = wins[component]
        masks.append(mask if np.any(mask) else None)
    return masks[0], masks[1]


def bare_riser_mask(
    centers: np.ndarray,
    part_types: np.ndarray,
    neighbors: np.ndarray,
    riser_parts: Sequence[Tuple[trimesh.Trimesh, str]],
) -> Optional[np.ndarray]:
    """Per-voxel bool, True on the voxels of every riser component that is bare.

    The first mask of :func:`riser_sleeve_masks`. None when no riser component
    is bare.
    """
    return riser_sleeve_masks(centers, part_types, neighbors, riser_parts)[0]
