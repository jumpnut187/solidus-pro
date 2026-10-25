"""Which riser voxels sit in no sleeve (``Part.riser_sleeve == "none"``).

Every riser part carries a sleeve choice, but the solver sees voxels, and a
voxel does not remember its part. :func:`bare_riser_mask` maps the choice back
onto the voxel mesh. Both thermal setups read the result
(``voxel_mesh.voxel_riser_bare``): the explicit-mold coupling and the
semi-infinite-sand boundary give a bare riser the casting's metal-to-sand
contact instead of the sleeve's.

A riser is one connected component of riser voxels, and a component cools one
way. When it spans parts with different settings (two risers that touch), the
voxels inside each riser part vote and the majority by voxel count wins. A tie,
and voxels inside no riser part (cut-cell slivers on the skin), count as
insulating, the default.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from core.part_manager import PART_TYPE_RISER, RISER_SLEEVE_NONE


def bare_riser_mask(
    centers: np.ndarray,
    part_types: np.ndarray,
    neighbors: np.ndarray,
    riser_parts: Sequence[Tuple[trimesh.Trimesh, str]],
) -> Optional[np.ndarray]:
    """Per-voxel bool, True on the voxels of every riser component that is bare.

    Args:
        centers: (N, 3) voxel centres, inches.
        part_types: (N,) voxel part codes.
        neighbors: (N, K) neighbor indices, -1 where there is none.
        riser_parts: ``(mesh, riser_sleeve)`` per riser part, as
            ``PartManager.get_riser_sleeve_parts`` returns them.

    Returns:
        The mask, or None when no riser part is bare (nothing to change).
    """
    if not any(sleeve == RISER_SLEEVE_NONE for _, sleeve in riser_parts):
        return None
    types = np.asarray(part_types)
    risers = np.flatnonzero(types == PART_TYPE_RISER)
    if len(risers) == 0:
        return None

    points = np.asarray(centers, dtype=np.float64)[risers]
    bare_votes = np.zeros(len(risers), dtype=np.int64)
    sleeved_votes = np.zeros(len(risers), dtype=np.int64)
    for mesh, sleeve in riser_parts:
        lo, hi = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
        near = np.flatnonzero(np.all((points >= lo) & (points <= hi), axis=1))
        if len(near) == 0:
            continue
        try:
            inside = near[np.asarray(mesh.contains(points[near]), dtype=bool)]
        except Exception:  # a mesh the ray test cannot answer for casts no vote
            continue
        votes = bare_votes if sleeve == RISER_SLEEVE_NONE else sleeved_votes
        votes[inside] += 1

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
    bare_total = np.bincount(component, weights=bare_votes, minlength=n_components)
    sleeved_total = np.bincount(component, weights=sleeved_votes, minlength=n_components)
    bare_component = bare_total > sleeved_total

    mask = np.zeros(len(types), dtype=bool)
    mask[risers] = bare_component[component]
    return mask if np.any(mask) else None
