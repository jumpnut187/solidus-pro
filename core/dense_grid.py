"""
Uniform dense lattice built from the engine's sparse voxel result.

The engine hands both front ends a *sparse* result: occupied voxel centers
plus one value per voxel per mode. Anything smoother than drawing a cube per
voxel -- painting the field on the casting's own surface, contouring a
threshold envelope, cutting a section cap -- needs the values on a dense,
uniform lattice. This module is that conversion, shared by the web sidecar
(``web/fields.py``) and the desktop result renderer (``ui/result_surface.py``).

Layout contract, relied on by both clients:

1. The lattice is uniform: every cell is ``spacing`` on a side, so ``dims``,
   ``origin`` and ``spacing`` fully describe it.
2. Cells are indexed x-fastest: ``index = ix + nx * (iy + ny * iz)``. Reshape
   a flat grid as ``(nz, ny, nx)`` for numpy, or hand it to VTK in Fortran
   order for a ``vtkImageData`` of dimensions ``(nx, ny, nz)``.
3. Scalar grids are *hole-filled*: a cell outside the metal carries the value
   of the nearest cell inside it (see :func:`scatter_and_fill`). Leaving them
   at zero makes trilinear sampling fade every surface toward the bottom of
   the colormap -- the blue halo of naive volume painting. The occupancy mask,
   not the scalar grid, says where metal actually is.

``origin`` is the world position of the **center** of cell ``(0, 0, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

#: Largest lattice this module will build, in cells.
#:
#: The lattice is derived from the extent of the voxel centers, so a run whose
#: voxels lie on a long diagonal at a fine voxel size implies a box far larger
#: than the metal in it. 64 million cells is about 64 MB for the mask and
#: 256 MB of transient EDT indices -- past any real casting on this hardware,
#: and short of the runs that cannot work at all.
MAX_LATTICE_CELLS = 64_000_000


class LatticeTooLargeError(ValueError):
    """The derived lattice is larger than this module will build.

    Carries the numbers the message needs so a front end can turn it into
    something a user can act on, rather than a stack trace.
    """

    def __init__(self, dims: Tuple[int, int, int], spacing: float) -> None:
        self.dims = dims
        self.spacing = spacing
        self.cell_count = int(dims[0]) * int(dims[1]) * int(dims[2])
        super().__init__(
            f"Result lattice would be {dims[0]} x {dims[1]} x {dims[2]} "
            f"({self.cell_count:,} cells) at {spacing:g} in; "
            f"the limit is {MAX_LATTICE_CELLS:,}."
        )


@dataclass(frozen=True)
class Lattice:
    """The uniform voxel lattice a run was solved on."""

    dims: Tuple[int, int, int]
    origin: Tuple[float, float, float]
    spacing: float

    @property
    def cell_count(self) -> int:
        nx, ny, nz = self.dims
        return int(nx) * int(ny) * int(nz)

    def to_volume(self, flat: np.ndarray) -> np.ndarray:
        """View a flat x-fastest grid as ``[ix, iy, iz]``."""
        nx, ny, nz = self.dims
        return np.asarray(flat).reshape((nz, ny, nx)).transpose(2, 1, 0)

    def world_to_cell(self, points: np.ndarray) -> np.ndarray:
        """Flat cell index of the cell whose center is nearest each world point."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        nx, ny, nz = self.dims
        ijk = np.rint((pts - np.asarray(self.origin)) / self.spacing).astype(np.int64)
        ix = np.clip(ijk[:, 0], 0, nx - 1)
        iy = np.clip(ijk[:, 1], 0, ny - 1)
        iz = np.clip(ijk[:, 2], 0, nz - 1)
        return ix + nx * (iy + ny * iz)


def derive_lattice(centers: np.ndarray, sizes: Optional[np.ndarray] = None) -> Lattice:
    """
    Recover the uniform lattice the sparse centers were sampled from.

    Raises :class:`LatticeTooLargeError` when the extent and the voxel size
    imply more than :data:`MAX_LATTICE_CELLS`. The check is on the cell count
    computed from the dimensions, before anything is allocated, because the
    failure this guards against is an allocation.

    The spacing is taken from the per-voxel size when it is uniform (it always
    is for a run at ``base_voxel_size``) and from the smallest inter-plane gap
    otherwise, and ``dims`` comes from the extent rather than the number of
    unique values -- an axis can have an entirely empty plane in the middle of
    the casting, and counting occupied planes would silently shear the grid.
    """
    if centers.size == 0:
        return Lattice(dims=(1, 1, 1), origin=(0.0, 0.0, 0.0), spacing=1.0)

    spacing = 0.0
    if sizes is not None and sizes.size:
        finite = sizes[np.isfinite(sizes) & (sizes > 0.0)]
        if finite.size:
            spacing = float(np.median(finite))

    if spacing <= 0.0:
        gaps: List[float] = []
        for axis in range(3):
            unique = np.unique(np.round(centers[:, axis].astype(np.float64), 6))
            if unique.size >= 2:
                diffs = np.diff(unique)
                positive = diffs[diffs > 1e-9]
                if positive.size:
                    gaps.append(float(np.min(positive)))
        spacing = float(np.min(gaps)) if gaps else 1.0

    origin: List[float] = []
    dims: List[int] = []
    for axis in range(3):
        column = centers[:, axis].astype(np.float64)
        low = float(np.min(column))
        high = float(np.max(column))
        origin.append(low)
        dims.append(int(round((high - low) / spacing)) + 1)

    shape = (dims[0], dims[1], dims[2])
    if dims[0] * dims[1] * dims[2] > MAX_LATTICE_CELLS:
        raise LatticeTooLargeError(shape, float(spacing))

    return Lattice(
        dims=shape,
        origin=(origin[0], origin[1], origin[2]),
        spacing=float(spacing),
    )


def flat_indices(centers: np.ndarray, lattice: Lattice) -> np.ndarray:
    """Map voxel centers onto flat, x-fastest cell indices."""
    nx, ny, nz = lattice.dims
    ox, oy, oz = lattice.origin
    spacing = lattice.spacing
    ix = np.clip(np.rint((centers[:, 0].astype(np.float64) - ox) / spacing), 0, nx - 1).astype(np.int64)
    iy = np.clip(np.rint((centers[:, 1].astype(np.float64) - oy) / spacing), 0, ny - 1).astype(np.int64)
    iz = np.clip(np.rint((centers[:, 2].astype(np.float64) - oz) / spacing), 0, nz - 1).astype(np.int64)
    return ix + nx * (iy + ny * iz)


def nearest_known_map(known: np.ndarray, lattice: Lattice) -> np.ndarray:
    """
    For every cell, the flat index of the nearest cell where ``known`` is true.

    ``known`` is a flat, x-fastest boolean grid. The Euclidean distance
    transform is the expensive part of building a dense grid, so the map
    derived from the occupancy mask is computed once and reused for every
    scalar field; only a mode defined on a strict subset of the metal pays
    for a second one.
    """
    nx, ny, nz = lattice.dims
    volume = np.asarray(known, dtype=bool).reshape((nz, ny, nx))
    unknown = ~volume
    if not np.any(unknown):
        return np.arange(lattice.cell_count, dtype=np.int64)
    if np.all(unknown):
        return np.zeros(lattice.cell_count, dtype=np.int64)

    # distance_transform_edt measures distance to the nearest ZERO of its
    # input, so the "unknown" grid is what goes in: the feature it walks
    # towards is a known cell.
    indices = ndimage.distance_transform_edt(unknown, return_distances=False, return_indices=True)
    kz, ky, kx = indices  # each (nz, ny, nx)
    flat = kx.astype(np.int64) + nx * (ky.astype(np.int64) + ny * kz.astype(np.int64))
    return np.asarray(flat, dtype=np.int64).reshape(-1)


def scatter_and_fill(
    values: np.ndarray,
    cell_indices: np.ndarray,
    sample_mask: np.ndarray,
    nearest: np.ndarray,
    lattice: Lattice,
) -> np.ndarray:
    """
    Scatter sparse per-voxel values onto the dense lattice, then fill the rest.

    ``sample_mask`` selects which voxels carry a usable value (a misrun voxel
    in ``fill_time`` does not). Cells with no value take the value of the
    nearest cell that has one -- via ``nearest``, which is computed once
    against the occupancy mask, plus a second, mode-specific pass in the rarer
    case where a mode is undefined on some *occupied* cells.
    """
    cell_count = lattice.cell_count
    dense = np.zeros(cell_count, dtype=np.float32)
    known = np.zeros(cell_count, dtype=bool)
    if not np.any(sample_mask):
        return dense

    dense[cell_indices[sample_mask]] = values[sample_mask]
    known[cell_indices[sample_mask]] = True

    filled: np.ndarray = dense[nearest]
    if np.all(known[nearest]):
        return filled

    mode_nearest = nearest_known_map(known, lattice)
    return np.asarray(dense[mode_nearest], dtype=np.float32)
