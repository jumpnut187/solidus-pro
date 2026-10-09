"""
Barycentric sampling of tet-mesh nodal fields onto a uniform lattice.

The tet solver stores one value per mesh node. Mold filling, the ML models and
both result renderers work on a uniform lattice. :class:`TetLatticeSampler`
locates every lattice cell centre in the tet mesh once and then maps any P1
nodal field onto the lattice by barycentric interpolation, which reproduces a
linear field exactly.

Layout matches :mod:`core.dense_grid`: ``origin`` is the centre of cell
``(0, 0, 0)`` in inches, cells are ``spacing`` inches on a side and flat
indices are x-fastest, ``ix + nx * (iy + ny * iz)``. The occupancy mask uses
the ``web/fields.py`` encoding: part code + 1 inside, 0 outside.

:class:`LatticeView` duck-types ``AdaptiveVoxelMesh`` for lattice consumers
(filling, ML). Its ``voxel_neighbors`` use the canonical 26-slot layout of
``AdaptiveVoxelMesh._canonicalize_neighbor_order``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from core.dense_grid import MAX_LATTICE_CELLS, LatticeTooLargeError

#: Candidate (tet, cell) pairs tested per chunk during point location. Each
#: candidate costs roughly 150 bytes of transient arrays, so this bounds the
#: location step at about 150 MB regardless of mesh or lattice size.
CANDIDATE_CHUNK = 1 << 20

#: Slack, in cells, added to each tet's bounding box before rounding to cell
#: ranges, so a centre exactly on a tet face is never dropped by rounding.
_RANGE_SLACK = 1e-7


def _canonical_slot_offsets() -> np.ndarray:
    """(26, 3) int64 ijk offsets in the slot order of ``_canonicalize_neighbor_order``.

    Slots 0-5 are +X, -X, +Y, -Y, +Z, -Z. Slots 6-17 are edges in plane
    groups XY, XZ, YZ, each ordered (+,+), (+,-), (-,+), (-,-). Slots 18-25
    are corners indexed by ``(sx<0)*4 | (sy<0)*2 | (sz<0)``.
    """
    offsets = np.zeros((26, 3), dtype=np.int64)
    offsets[0:6] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    for group, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        for code in range(4):
            slot = 6 + 4 * group + code
            offsets[slot, a] = -1 if code & 2 else 1
            offsets[slot, b] = -1 if code & 1 else 1
    for code in range(8):
        offsets[18 + code] = [
            -1 if code & 4 else 1,
            -1 if code & 2 else 1,
            -1 if code & 1 else 1,
        ]
    return offsets


CANONICAL_SLOT_OFFSETS = _canonical_slot_offsets()


class TetLatticeSampler:
    """Locate uniform lattice cell centres in a tet mesh and sample nodal fields.

    Args:
        node_xyz_in: (N, 3) node coordinates, inches.
        tets: (T, 4) node indices per tet, either orientation.
        tet_part_types: (T,) part code per tet.
        spacing_in: lattice pitch, inches.
        inside_tol: a centre is inside a tet when every barycentric weight is
            ``>= -inside_tol``. A centre inside several tets (shared face,
            edge or node) belongs to the lowest tet index.

    The lattice is centred on the mesh bounding box with
    ``dims = ceil(extent / spacing)`` per axis, so the cells tile the box and
    no cell centre sits on a box face.
    """

    def __init__(
        self,
        node_xyz_in: np.ndarray,
        tets: np.ndarray,
        tet_part_types: np.ndarray,
        spacing_in: float,
        *,
        inside_tol: float = 1e-9,
    ) -> None:
        nodes = np.ascontiguousarray(node_xyz_in, dtype=np.float64)
        tet_arr = np.ascontiguousarray(tets, dtype=np.int64)
        part_types = np.asarray(tet_part_types)
        spacing = float(spacing_in)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] == 0:
            raise ValueError(f"node_xyz_in must be (N, 3) with N > 0, got {nodes.shape}")
        if not np.all(np.isfinite(nodes)):
            raise ValueError("node_xyz_in contains non-finite coordinates")
        if tet_arr.ndim != 2 or tet_arr.shape[1] != 4:
            raise ValueError(f"tets must be (T, 4), got {tet_arr.shape}")
        if part_types.shape != (tet_arr.shape[0],):
            raise ValueError(
                f"tet_part_types must be ({tet_arr.shape[0]},), got {part_types.shape}"
            )
        if tet_arr.size and (tet_arr.min() < 0 or tet_arr.max() >= nodes.shape[0]):
            raise ValueError("tets reference node indices outside node_xyz_in")
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError(f"spacing_in must be a positive finite length, got {spacing_in}")

        lo = nodes.min(axis=0)
        hi = nodes.max(axis=0)
        dims_arr = np.maximum(1, np.ceil((hi - lo) / spacing - 1e-9)).astype(np.int64)
        dims = (int(dims_arr[0]), int(dims_arr[1]), int(dims_arr[2]))
        if dims[0] * dims[1] * dims[2] > MAX_LATTICE_CELLS:
            raise LatticeTooLargeError(dims, spacing)

        self.dims: Tuple[int, int, int] = dims
        self.spacing: float = spacing
        self.origin: np.ndarray = 0.5 * (lo + hi) - 0.5 * (dims_arr - 1) * spacing
        self.bounds: np.ndarray = np.stack([lo, hi], axis=1)
        self.inside_tol: float = float(inside_tol)
        self._node_count = nodes.shape[0]

        flat, tet_index, bary = self._locate(nodes, tet_arr)
        self.cell_flat: np.ndarray = flat
        self.cell_tet: np.ndarray = tet_index.astype(np.int32)
        self.cell_bary: np.ndarray = bary
        nx, ny, _ = dims
        ijk = np.stack([flat % nx, (flat // nx) % ny, flat // (nx * ny)], axis=1)
        self.cell_centers: np.ndarray = self.origin + spacing * ijk.astype(np.float64)
        self.cell_part_types: np.ndarray = part_types[tet_index].astype(np.int8)
        self._cell_nodes = tet_arr[tet_index].astype(np.int32)  # (M, 4)
        self.nearest_node: np.ndarray = self._cell_nodes[
            np.arange(flat.size), np.argmax(bary, axis=1)
        ].astype(np.int32)
        self._view: Optional[LatticeView] = None

    @property
    def cell_count(self) -> int:
        """Total lattice cells, ``nx * ny * nz``."""
        return self.dims[0] * self.dims[1] * self.dims[2]

    def _locate(
        self, nodes: np.ndarray, tets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the lowest-index containing tet of every inside cell centre.

        Returns ``(flat, tet, bary)`` sorted by ascending flat index. Candidate
        cells come from each tet's bounding box. The global candidate list is
        walked in chunks of :data:`CANDIDATE_CHUNK` in tet order, so a cell is
        claimed by the first tet that contains it and memory stays bounded
        even when one tet spans most of the lattice.
        """
        empty = (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 4), dtype=np.float64),
        )
        if tets.shape[0] == 0:
            return empty

        nx, ny, nz = self.dims
        dims_arr = np.array(self.dims, dtype=np.int64)
        spacing = self.spacing
        origin = self.origin
        tol = self.inside_tol

        corners = nodes[tets]  # (T, 4, 3)
        tmin = (corners.min(axis=1) - origin) / spacing
        tmax = (corners.max(axis=1) - origin) / spacing
        ijk_lo = np.clip(np.ceil(tmin - _RANGE_SLACK), 0, dims_arr).astype(np.int64)
        ijk_hi = np.clip(np.floor(tmax + _RANGE_SLACK), -1, dims_arr - 1).astype(np.int64)
        extent = np.maximum(ijk_hi - ijk_lo + 1, 0)
        del tmin, tmax

        v0 = corners[:, 0, :]
        e1 = corners[:, 1, :] - v0
        e2 = corners[:, 2, :] - v0
        e3 = corners[:, 3, :] - v0
        del corners
        det = np.einsum("ij,ij->i", e1, np.cross(e2, e3))
        scale = np.maximum(
            np.max(np.abs(np.stack([e1, e2, e3], axis=1)), axis=(1, 2)), np.finfo(float).tiny
        )
        valid = np.abs(det) > 1e-12 * scale**3
        counts = np.where(valid, extent.prod(axis=1), 0)

        active = np.flatnonzero(counts > 0)
        if active.size == 0:
            return empty
        counts = counts[active]
        ijk_lo = ijk_lo[active]
        ni = extent[active, 0]
        nj = extent[active, 1]
        v0 = v0[active]
        # Rows of the inverse of [e1 e2 e3]: lambda_{1..3} = inv @ (p - v0).
        det = det[active]
        e1, e2, e3 = e1[active], e2[active], e3[active]
        inv = np.stack([np.cross(e2, e3), np.cross(e3, e1), np.cross(e1, e2)], axis=1)
        inv /= det[:, None, None]
        del e1, e2, e3, det

        ends = np.cumsum(counts)
        starts = ends - counts
        total = int(ends[-1])
        assigned = np.zeros(nx * ny * nz, dtype=bool)
        found_flat: List[np.ndarray] = []
        found_tet: List[np.ndarray] = []
        found_bary: List[np.ndarray] = []

        for g0 in range(0, total, CANDIDATE_CHUNK):
            g1 = min(g0 + CANDIDATE_CHUNK, total)
            a0 = int(np.searchsorted(ends, g0, side="right"))
            a1 = int(np.searchsorted(starts, g1, side="left"))
            seg = np.minimum(ends[a0:a1], g1) - np.maximum(starts[a0:a1], g0)
            loc = np.repeat(np.arange(a0, a1, dtype=np.int64), seg)
            k = np.arange(g0, g1, dtype=np.int64) - starts[loc]
            row = k // ni[loc]
            ix = ijk_lo[loc, 0] + k % ni[loc]
            iy = ijk_lo[loc, 1] + row % nj[loc]
            iz = ijk_lo[loc, 2] + row // nj[loc]
            flat = ix + nx * (iy + ny * iz)
            free = ~assigned[flat]
            if not np.any(free):
                continue
            loc, flat = loc[free], flat[free]
            ix, iy, iz = ix[free], iy[free], iz[free]

            base = v0[loc]
            dx = origin[0] + spacing * ix - base[:, 0]
            dy = origin[1] + spacing * iy - base[:, 1]
            dz = origin[2] + spacing * iz - base[:, 2]
            m = inv[loc]
            lam = np.empty((loc.size, 4), dtype=np.float64)
            lam[:, 1] = m[:, 0, 0] * dx + m[:, 0, 1] * dy + m[:, 0, 2] * dz
            lam[:, 2] = m[:, 1, 0] * dx + m[:, 1, 1] * dy + m[:, 1, 2] * dz
            lam[:, 3] = m[:, 2, 0] * dx + m[:, 2, 1] * dy + m[:, 2, 2] * dz
            lam[:, 0] = 1.0 - lam[:, 1] - lam[:, 2] - lam[:, 3]
            inside = np.all(lam >= -tol, axis=1)
            if not np.any(inside):
                continue
            flat_in = flat[inside]
            # Candidates are in ascending tet order, so the first occurrence of
            # each cell is its lowest containing tet.
            _, first = np.unique(flat_in, return_index=True)
            flat_in = flat_in[first]
            assigned[flat_in] = True
            found_flat.append(flat_in)
            found_tet.append(active[loc[inside][first]])
            found_bary.append(lam[inside][first])

        if not found_flat:
            return empty
        flat_all = np.concatenate(found_flat)
        order = np.argsort(flat_all, kind="stable")
        return (
            flat_all[order],
            np.concatenate(found_tet)[order],
            np.concatenate(found_bary)[order],
        )

    def sample(self, node_values: np.ndarray) -> np.ndarray:
        """Barycentric interpolation of nodal values at inside cell centres.

        ``(N,)`` returns ``(M,)``. ``(F, N)`` returns ``(F, M)``. float64,
        same units as the input.
        """
        values = np.asarray(node_values, dtype=np.float64)
        if values.shape[-1] != self._node_count or values.ndim not in (1, 2):
            raise ValueError(
                f"node_values must be ({self._node_count},) or (F, {self._node_count}), "
                f"got {values.shape}"
            )
        out = values[..., self._cell_nodes[:, 0]] * self.cell_bary[:, 0]
        for corner in range(1, 4):
            out += values[..., self._cell_nodes[:, corner]] * self.cell_bary[:, corner]
        return out

    def dense(self, node_values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """Flat x-fastest float32 grid: sampled values inside, ``fill`` outside.

        ``(N,)`` returns ``(nx*ny*nz,)``. ``(F, N)`` returns ``(F, nx*ny*nz)``.
        """
        sampled = self.sample(node_values)
        grid = np.full(sampled.shape[:-1] + (self.cell_count,), fill, dtype=np.float32)
        grid[..., self.cell_flat] = sampled
        return grid

    def mask_codes(self) -> np.ndarray:
        """Flat x-fastest uint8 grid: part code + 1 inside, 0 outside."""
        mask = np.zeros(self.cell_count, dtype=np.uint8)
        codes = np.clip(self.cell_part_types.astype(np.int16) + 1, 0, 255)
        mask[self.cell_flat] = codes.astype(np.uint8)
        return mask

    def lattice_view(self) -> "LatticeView":
        """Voxel-mesh duck type over the inside cells. Built once, then cached."""
        if self._view is None:
            self._view = LatticeView(self)
        return self._view


class LatticeView:
    """Inside lattice cells exposed with the ``AdaptiveVoxelMesh`` attribute names.

    Cell ``i`` of the view is ``sampler.cell_flat[i]``. Lengths are inches.

    ``voxel_moduli`` is the distance from each cell centre to the nearest
    surface cell centre, 0 on surface cells. That is the voxel path's
    definition (``AdaptiveVoxelMesh`` queries a KD-tree of surface voxel
    centres), computed here with an exact Euclidean distance transform.
    """

    mesh_type = "lattice"

    def __init__(self, sampler: TetLatticeSampler) -> None:
        self.sampler = sampler
        spacing = sampler.spacing
        count = sampler.cell_flat.size
        nx, ny, nz = sampler.dims

        self.base_voxel_size: float = spacing
        self.bounds: np.ndarray = sampler.bounds.copy()
        self.grid_shape: Tuple[int, int, int] = sampler.dims
        self.voxel_centers: np.ndarray = sampler.cell_centers.astype(np.float32)
        self.voxel_sizes: np.ndarray = np.full(count, spacing, dtype=np.float32)
        self.voxel_volumes: np.ndarray = np.full(count, spacing**3, dtype=np.float32)
        self.voxel_part_types: np.ndarray = sampler.cell_part_types.copy()

        lookup = np.full(sampler.cell_count, -1, dtype=np.int32)
        lookup[sampler.cell_flat] = np.arange(count, dtype=np.int32)
        flat = sampler.cell_flat
        ijk = np.stack([flat % nx, (flat // nx) % ny, flat // (nx * ny)], axis=1)
        dims_arr = np.array(sampler.dims, dtype=np.int64)
        neighbors = np.full((count, 26), -1, dtype=np.int32)
        for slot, offset in enumerate(CANONICAL_SLOT_OFFSETS):
            nijk = ijk + offset
            in_box = np.all((nijk >= 0) & (nijk < dims_arr), axis=1)
            nflat = nijk[in_box, 0] + nx * (nijk[in_box, 1] + ny * nijk[in_box, 2])
            neighbors[in_box, slot] = lookup[nflat]
        self.voxel_neighbors: np.ndarray = neighbors

        exposed = np.count_nonzero(neighbors[:, :6] < 0, axis=1)
        self.surface_voxels: np.ndarray = exposed > 0
        self.voxel_surface_areas: np.ndarray = (exposed * spacing**2).astype(np.float32)
        self.voxel_moduli: np.ndarray = self._surface_distance()

    def _surface_distance(self) -> np.ndarray:
        """(M,) float32 inches: distance from each cell centre to the nearest surface cell centre."""
        sampler = self.sampler
        count = sampler.cell_flat.size
        if count == 0 or not np.any(self.surface_voxels):
            return np.zeros(count, dtype=np.float32)
        nx, ny, nz = sampler.dims
        not_surface = np.ones(sampler.cell_count, dtype=bool)
        not_surface[sampler.cell_flat[self.surface_voxels]] = False
        distance = ndimage.distance_transform_edt(
            not_surface.reshape((nz, ny, nx)), sampling=sampler.spacing
        )
        return np.asarray(distance, dtype=np.float32).reshape(-1)[sampler.cell_flat]

    def get_voxel_count(self) -> int:
        """Number of inside cells."""
        return int(self.voxel_centers.shape[0])

    def get_surface_voxel_indices(self) -> np.ndarray:
        """Indices of inside cells with at least one face neighbour outside."""
        return np.flatnonzero(self.surface_voxels)

    def get_voxels_by_type(self, part_type: int) -> np.ndarray:
        """Indices of inside cells with the given part code."""
        return np.flatnonzero(self.voxel_part_types == part_type)
