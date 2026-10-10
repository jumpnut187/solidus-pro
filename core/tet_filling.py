"""Mold filling on a tet run (contract C5 in docs/tet_migration/DESIGN.md).

The filling LBM needs a uniform lattice. On a tet run the lattice comes from
:class:`core.tet_lattice.TetLatticeSampler` at pitch
``base_voxel_size * settings.mesh.tet_lattice_spacing_factor``. The existing
:func:`core.filling_stage.maybe_run_filling` runs on that lattice unchanged,
and every per-cell field it returns is mapped onto the tet nodes, so the
engine sees ``(N,)`` arrays in node order.

FILTER tets are removed from the tet mesh (contract C1), so the sampled
lattice has no filter cells and the fill would stop at the filter as if it
were a wall. :func:`build_filling_lattice` rasterizes each FILTER part onto
the same lattice and appends those cells. They exist only for the fill and
are never mapped to nodes.

Node mapping: each node takes the value of the nearest lattice cell centre of
the same class (metal or non-metal). Restricting by class stops a metal node
on a chill or mold interface from inheriting the mold temperature of an
adjacent non-metal cell.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from core.dense_grid import LatticeTooLargeError
from core.filling_stage import FillFrame, FillingResult, maybe_run_filling
from core.part_manager import PART_TYPE_FILTER, is_metal
from core.tet_lattice import LatticeView, TetLatticeSampler
from core.tet_mesh import winding_number_contains

logger = logging.getLogger(__name__)

MessageCallback = Callable[[str], None]

#: A node farther than this many lattice pitches from its mapped cell centre
#: is counted in ``FillingLattice.far_node_count``. A cell's centre is at most
#: sqrt(3)/2 pitch from any point of the cell.
FAR_NODE_PITCHES = float(np.sqrt(3.0))


class FillingLatticeMesh:
    """Lattice cells the filling stage solves on: the sampled view plus rasterized FILTER cells.

    Exposes only the members :mod:`core.filling_stage` reads
    (``voxel_centers``, ``voxel_sizes``, ``voxel_volumes``,
    ``voxel_part_types``, ``get_voxel_count``). Cells ``0..view_cell_count-1``
    are the view's cells in view order. The rest are FILTER cells.
    """

    mesh_type = "lattice"

    def __init__(self, view: LatticeView, filter_centers: np.ndarray) -> None:
        spacing = float(view.base_voxel_size)
        filter_centers = np.asarray(filter_centers, dtype=np.float32).reshape(-1, 3)
        self.view = view
        self.view_cell_count = view.get_voxel_count()
        self.filter_cell_count = int(filter_centers.shape[0])
        self.base_voxel_size = spacing
        self.voxel_centers = np.vstack([view.voxel_centers, filter_centers]).astype(np.float32)
        self.voxel_part_types = np.concatenate(
            [
                np.asarray(view.voxel_part_types, dtype=np.int8),
                np.full(self.filter_cell_count, PART_TYPE_FILTER, dtype=np.int8),
            ]
        )
        count = self.voxel_centers.shape[0]
        self.voxel_sizes = np.full(count, spacing, dtype=np.float32)
        self.voxel_volumes = np.full(count, spacing**3, dtype=np.float32)

    def get_voxel_count(self) -> int:
        """Total cells the fill solves on, filter cells included."""
        return int(self.voxel_centers.shape[0])


@dataclass(frozen=True)
class FillingLattice:
    """The lattice a tet run fills on, and how its cells map back to nodes."""

    sampler: TetLatticeSampler
    view: LatticeView
    #: What ``maybe_run_filling`` receives: ``view`` itself, or a
    #: :class:`FillingLatticeMesh` when FILTER cells were added.
    filling_mesh: Any
    filter_cell_count: int
    #: (N,) index into the view's cells for every tet node.
    node_cells: np.ndarray
    #: Nodes whose mapped cell centre is farther than ``FAR_NODE_PITCHES``
    #: pitches, e.g. a metal part thinner than the pitch with no cell of its own.
    far_node_count: int = 0

    @property
    def filling_cell_count(self) -> int:
        return int(self.filling_mesh.get_voxel_count())


def filter_lattice_centers(
    sampler: TetLatticeSampler, filter_meshes: Sequence[trimesh.Trimesh]
) -> np.ndarray:
    """(F, 3) float64 centres of lattice cells inside a FILTER part and outside the tet mesh.

    Candidates are the sampler's cell centres (``origin + spacing * ijk``)
    inside each filter's bounding box, so every returned centre sits exactly
    on the sampler lattice. Containment uses the voxel path's
    ``_robust_contains_check_impl`` with the same half-pitch proximity
    tolerance ``AdaptiveVoxelMesh.assign_part_types`` uses. Open filter
    surfaces use ``core.tet_mesh.winding_number_contains`` with the same
    tolerance, because ray tests are unreliable on open surfaces. Cells
    already inside the tet mesh keep their metal or chill code. Filter parts
    outside the lattice box are clipped to it.
    """
    from core.voxel_mesh import _robust_contains_check_impl

    if not filter_meshes:
        return np.zeros((0, 3), dtype=np.float64)
    nx, ny, nz = sampler.dims
    dims_arr = np.array(sampler.dims, dtype=np.int64)
    spacing = sampler.spacing
    origin = sampler.origin
    taken = np.zeros(sampler.cell_count, dtype=bool)
    taken[sampler.cell_flat] = True
    found = []
    for mesh in filter_meshes:
        lo, hi = np.asarray(mesh.bounds, dtype=np.float64)
        ijk_lo = np.clip(np.ceil((lo - origin) / spacing - 1e-9), 0, dims_arr - 1).astype(np.int64)
        ijk_hi = np.clip(np.floor((hi - origin) / spacing + 1e-9), 0, dims_arr - 1).astype(np.int64)
        if np.any(ijk_hi < ijk_lo):
            continue
        axes = [np.arange(ijk_lo[a], ijk_hi[a] + 1) for a in range(3)]
        iz, iy, ix = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        ijk = np.column_stack([ix.ravel(), iy.ravel(), iz.ravel()])
        flat = ijk[:, 0] + nx * (ijk[:, 1] + ny * ijk[:, 2])
        free = ~taken[flat]
        if not np.any(free):
            continue
        ijk, flat = ijk[free], flat[free]
        centers = origin + spacing * ijk.astype(np.float64)
        if mesh.is_watertight:
            inside = np.asarray(
                _robust_contains_check_impl(mesh, centers, 0.5 * spacing), dtype=bool
            )
        else:
            inside = winding_number_contains(mesh, centers, near_surface_tol=0.5 * spacing)
        taken[flat[inside]] = True
        found.append(centers[inside])
    if not found:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(found)


def node_lattice_cells(
    node_xyz: np.ndarray,
    node_part_types: np.ndarray,
    view: LatticeView,
    *,
    return_distance: bool = False,
) -> Any:
    """(N,) int64: for each node, the nearest view cell centre of the same class (metal or not).

    Falls back to the nearest cell of any class when the view has no cell of
    the node's class. With ``return_distance`` also returns the (N,) float64
    node-to-centre distances in inches.
    """
    nodes = np.asarray(node_xyz, dtype=np.float64)
    node_metal = is_metal(np.asarray(node_part_types, dtype=np.int32))
    centers = np.asarray(view.voxel_centers, dtype=np.float64)
    if centers.shape[0] == 0:
        raise ValueError("the filling lattice has no inside cells")
    cell_metal = is_metal(np.asarray(view.voxel_part_types, dtype=np.int32))
    result = np.empty(nodes.shape[0], dtype=np.int64)
    distances = np.empty(nodes.shape[0], dtype=np.float64)
    for metal in (True, False):
        node_ids = np.flatnonzero(node_metal == metal)
        if node_ids.size == 0:
            continue
        cell_ids = np.flatnonzero(cell_metal == metal)
        if cell_ids.size == 0:
            cell_ids = np.arange(centers.shape[0])
        distance, nearest = cKDTree(centers[cell_ids]).query(nodes[node_ids], workers=-1)
        result[node_ids] = cell_ids[np.asarray(nearest, dtype=np.int64)]
        distances[node_ids] = distance
    if return_distance:
        return result, distances
    return result


def build_filling_lattice(
    tet_mesh: Any,
    spacing_in: float,
    *,
    filter_meshes: Sequence[trimesh.Trimesh] = (),
) -> FillingLattice:
    """Sample the tet mesh onto a lattice of pitch ``spacing_in`` and add FILTER cells.

    Raises :class:`core.dense_grid.LatticeTooLargeError` when the lattice
    exceeds ``MAX_LATTICE_CELLS``.
    """
    sampler = TetLatticeSampler(
        tet_mesh.node_xyz, tet_mesh.tets, tet_mesh.tet_part_types, spacing_in
    )
    view = sampler.lattice_view()
    if view.get_voxel_count() == 0:
        raise ValueError(
            f"no lattice cell centre at pitch {spacing_in:g} in falls inside the tet mesh"
        )
    filter_centers = filter_lattice_centers(sampler, filter_meshes)
    filling_mesh: Any = view
    if filter_centers.shape[0] > 0:
        filling_mesh = FillingLatticeMesh(view, filter_centers)
    node_cells, node_distance = node_lattice_cells(
        tet_mesh.node_xyz, tet_mesh.voxel_part_types, view, return_distance=True
    )
    return FillingLattice(
        sampler=sampler,
        view=view,
        filling_mesh=filling_mesh,
        filter_cell_count=int(filter_centers.shape[0]),
        node_cells=node_cells,
        far_node_count=int(np.count_nonzero(node_distance > FAR_NODE_PITCHES * float(spacing_in))),
    )


def map_filling_result_to_nodes(
    result: FillingResult, node_cells: np.ndarray, filling_cell_count: int
) -> FillingResult:
    """Copy of ``result`` with every per-cell field gathered onto nodes.

    Per-cell fields are ``fill_time_s``, ``temperature_c``,
    ``entrapped_gas``, ``cold_shut`` and each frame's ``fill_fraction`` and
    ``speed_m_s``. A field whose length is not ``filling_cell_count`` (an
    empty default) is left as is. Scalar counts stay in lattice cells.
    """
    cells = np.asarray(node_cells, dtype=np.int64)

    def gather(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        if arr.ndim != 1 or arr.shape[0] != filling_cell_count:
            return arr
        return arr[cells]

    frames = tuple(
        FillFrame(
            time_s=frame.time_s,
            fill_fraction=gather(frame.fill_fraction),
            speed_m_s=gather(frame.speed_m_s),
        )
        for frame in result.frames
    )
    return replace(
        result,
        fill_time_s=gather(result.fill_time_s),
        temperature_c=gather(result.temperature_c),
        entrapped_gas=gather(result.entrapped_gas),
        cold_shut=gather(result.cold_shut),
        frames=frames,
    )


def maybe_run_tet_filling(
    tet_mesh: Any,
    alloy: Any,
    analysis_options: Any,
    inputs: Any,
    *,
    spacing_in: float,
    parts_with_types: Sequence[Tuple[trimesh.Trimesh, int]] = (),
    cancel_callback: Optional[Callable[[], bool]] = None,
    progress_message_callback: Optional[MessageCallback] = None,
    detailed_progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Optional[FillingResult]:
    """Tet-run counterpart of :func:`core.filling_stage.maybe_run_filling`.

    Same gate (filling enabled and a pour location set). Returns a
    :class:`FillingResult` whose per-cell fields have length N nodes, or
    ``None`` when the stage did not run. The lattice is built only when the
    gate is open.
    """

    def emit(message: str) -> None:
        if progress_message_callback is not None:
            progress_message_callback(message)

    if not getattr(analysis_options, "filling", False) or not getattr(inputs, "pour_location", None):
        # Gate only: maybe_run_filling returns before reading the mesh, and
        # emits the same "no pour location" notice the voxel path does.
        return maybe_run_filling(
            tet_mesh,
            alloy,
            analysis_options,
            inputs,
            cancel_callback=cancel_callback,
            progress_message_callback=progress_message_callback,
            detailed_progress_callback=detailed_progress_callback,
        )

    filter_meshes = [mesh for mesh, code in parts_with_types if int(code) == PART_TYPE_FILTER]
    try:
        lattice = build_filling_lattice(tet_mesh, spacing_in, filter_meshes=filter_meshes)
    except (LatticeTooLargeError, ValueError) as exc:
        emit(f"Filling stage skipped: {exc}")
        return None
    emit(
        f"Filling: lattice {lattice.sampler.dims[0]} x {lattice.sampler.dims[1]} x "
        f"{lattice.sampler.dims[2]} at {spacing_in:g} in, "
        f"{lattice.view.get_voxel_count():,} cells sampled from the tet mesh"
    )
    if lattice.far_node_count:
        far_message = (
            f"Filling: {lattice.far_node_count:,} tet nodes are more than "
            f"{FAR_NODE_PITCHES * spacing_in:.3g} in from the nearest lattice cell of their "
            f"class and copy that cell's fill values (a part thinner than the {spacing_in:g} in "
            f"pitch has no cell of its own); lower tet_lattice_spacing_factor to resolve it"
        )
        logger.warning(far_message)
        emit(far_message)
    if filter_meshes:
        emit(
            f"Filling: rasterized {lattice.filter_cell_count:,} FILTER cells from "
            f"{len(filter_meshes)} filter part(s) with the voxel-path containment check "
            f"(filter tets are not in the thermal mesh)"
        )

    result = maybe_run_filling(
        lattice.filling_mesh,
        alloy,
        analysis_options,
        inputs,
        cancel_callback=cancel_callback,
        progress_message_callback=progress_message_callback,
        detailed_progress_callback=detailed_progress_callback,
    )
    if result is None:
        return None
    return map_filling_result_to_nodes(result, lattice.node_cells, lattice.filling_cell_count)
