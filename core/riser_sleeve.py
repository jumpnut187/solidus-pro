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
        except Exception:
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
    return riser_sleeve_masks(centers, part_types, neighbors, riser_parts)[0]
