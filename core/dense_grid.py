from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

MAX_LATTICE_CELLS = 64_000_000

class LatticeTooLargeError(ValueError):

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

    dims: Tuple[int, int, int]
    origin: Tuple[float, float, float]
    spacing: float

    @property
    def cell_count(self) -> int:
        nx, ny, nz = self.dims
        return int(nx) * int(ny) * int(nz)

    def to_volume(self, flat: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.dims
        return np.asarray(flat).reshape((nz, ny, nx)).transpose(2, 1, 0)

    def world_to_cell(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        nx, ny, nz = self.dims
        ijk = np.rint((pts - np.asarray(self.origin)) / self.spacing).astype(np.int64)
        ix = np.clip(ijk[:, 0], 0, nx - 1)
        iy = np.clip(ijk[:, 1], 0, ny - 1)
        iz = np.clip(ijk[:, 2], 0, nz - 1)
        return ix + nx * (iy + ny * iz)

def derive_lattice(centers: np.ndarray, sizes: Optional[np.ndarray] = None) -> Lattice:
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
    nx, ny, nz = lattice.dims
    ox, oy, oz = lattice.origin
    spacing = lattice.spacing
    ix = np.clip(np.rint((centers[:, 0].astype(np.float64) - ox) / spacing), 0, nx - 1).astype(np.int64)
    iy = np.clip(np.rint((centers[:, 1].astype(np.float64) - oy) / spacing), 0, ny - 1).astype(np.int64)
    iz = np.clip(np.rint((centers[:, 2].astype(np.float64) - oz) / spacing), 0, nz - 1).astype(np.int64)
    return ix + nx * (iy + ny * iz)

def nearest_known_map(known: np.ndarray, lattice: Lattice) -> np.ndarray:
    nx, ny, nz = lattice.dims
    volume = np.asarray(known, dtype=bool).reshape((nz, ny, nx))
    unknown = ~volume
    if not np.any(unknown):
        return np.arange(lattice.cell_count, dtype=np.int64)
    if np.all(unknown):
        return np.zeros(lattice.cell_count, dtype=np.int64)

    indices = ndimage.distance_transform_edt(unknown, return_distances=False, return_indices=True)
    kz, ky, kx = indices
    flat = kx.astype(np.int64) + nx * (ky.astype(np.int64) + ny * kz.astype(np.int64))
    return np.asarray(flat, dtype=np.int64).reshape(-1)

def scatter_and_fill(
    values: np.ndarray,
    cell_indices: np.ndarray,
    sample_mask: np.ndarray,
    nearest: np.ndarray,
    lattice: Lattice,
) -> np.ndarray:
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
