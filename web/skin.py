"""
Signed distance from the result lattice to the casting's *real* surface.

The viewport draws two pictures of the same metal and they do not agree on
where it ends. The casting's skin is the STL, drawn as triangles. Everything
derived from the result -- the occupancy mask, the painted field, the contoured
liquid front -- lives on the voxel lattice, and the mesher marks a voxel as
metal whenever its cube *touches* the part. On a coarse lattice that puts the
voxel body up to half a voxel outside the STL all the way round: on the run
this was written against, one voxel center in four sat outside the surface it
was supposedly inside, by as much as 0.26 in at 0.38 in voxels.

A liquid front clipped against that mask is therefore free to stand up to a
half voxel proud of the skin, and it did: thin molten sheets draped down the
outside of the riser, curled petals floating over its dome, dark red shards at
the plate corners. Every one of those was a genuine piece of the front, in
cells the mask calls metal, poking through a skin that does not reach them.

This module gives the renderer the missing fact: for every lattice cell, how
far its center is from the nearest part surface and on which side. Sampled
trilinearly, the zero crossing of that grid *is* the STL to within a small
fraction of a voxel -- exactly on flat faces, a few hundredths of an inch
short at a fillet, slightly rounded at a sharp corner -- so a fragment of the
front can be thrown away the moment it leaves the casting, and the seam where
the front meets the skin closes.

The grid is a byte per cell in the lattice's own layout, so it travels and
binds exactly like the mask. Distances are in voxels, clamped to
``+-SKIN_RANGE_VOXELS`` and mapped onto ``0..255`` with the surface at 127.5:
below half is inside, above is outside. Two voxels of range is plenty; the
only thing anyone reads out of this grid is where it crosses zero.

Cost is kept to the band of cells within two voxels of the mask boundary,
which is the only place the zero crossing can be. The unsigned distance there
is one closest-point query per part, which is cheap. The *sign* is what is
expensive -- an inside test is a ray cast -- so it is resolved by ray only for
cells close enough to a surface for the mask to be wrong about them, and taken
from the mask everywhere else: a voxel the mesher left empty is outside, and a
metal voxel further than half a diagonal from every surface is inside.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import hashlib
import json
import logging
import os
import tempfile
import threading

import numpy as np
from scipy import ndimage

from web.fields import Lattice

logger = logging.getLogger(__name__)

#: Half-width of the distance range the byte grid encodes, in voxels.
SKIN_RANGE_VOXELS = 2.0

#: Cells on either side of the mask boundary that get a measured distance, in
#: Chebyshev steps. The STL is never more than about 0.87 voxels (half a cube
#: diagonal) from the mask boundary, so two is one voxel of margin for the
#: interpolation.
_BAND_CELLS = 2

#: A cell closer than this to a surface, in voxels, may be on either side of
#: it whatever the mask says; half a cube diagonal is the furthest a voxel the
#: mesher counted as metal can reach past the part.
_AMBIGUOUS_VOXELS = float(np.sqrt(3.0) / 2.0)

#: Refuse to build a grid whose band is larger than this. Past it the
#: closest-point pass alone runs to minutes, and the viewport has a fallback.
MAX_SKIN_BAND_CELLS = 2_000_000

SKIN_VERSION = 1


@dataclass
class SkinGrid:
    """The quantized signed-distance grid for one run and one set of parts."""

    values: np.ndarray  # uint8, flat, x-fastest
    range_voxels: float
    #: Identifies the lattice and the parts this was built from.
    signature: str

    def tobytes(self) -> bytes:
        return np.ascontiguousarray(self.values, dtype=np.uint8).tobytes()


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def quantize(distance_voxels: np.ndarray, range_voxels: float = SKIN_RANGE_VOXELS) -> np.ndarray:
    """Map signed distances in voxels onto bytes, the surface at 127.5."""
    scaled = np.clip(distance_voxels / range_voxels, -1.0, 1.0)
    return np.rint((scaled + 1.0) * 0.5 * 255.0).astype(np.uint8)


def dequantize(values: np.ndarray, range_voxels: float = SKIN_RANGE_VOXELS) -> np.ndarray:
    """The inverse of :func:`quantize`, for tests and diagnostics."""
    return (values.astype(np.float32) / 255.0 * 2.0 - 1.0) * range_voxels


def build_skin_grid(lattice: Lattice, mask: np.ndarray, meshes: Sequence[Any]) -> Optional[np.ndarray]:
    """
    Signed distance to the union of ``meshes`` at every lattice cell, as bytes.

    ``meshes`` are ``trimesh.Trimesh`` objects already in world coordinates.
    Returns ``None`` when there is nothing to measure against or the band is
    too large to measure in reasonable time.
    """
    meshes = [mesh for mesh in meshes if mesh is not None and len(getattr(mesh, "faces", ())) > 0]
    if not meshes:
        return None

    nx, ny, nz = lattice.dims
    spacing = float(lattice.spacing)
    metal = np.asarray(mask, dtype=np.uint8).reshape((nz, ny, nx)) != 0

    # Deep inside and far outside need no measurement; the sign alone is the
    # answer, and the magnitude is off the end of the range anyway.
    distance = np.where(metal, -SKIN_RANGE_VOXELS, SKIN_RANGE_VOXELS).astype(np.float32)

    # The full 3 x 3 x 3 neighbourhood, not the six-connected one: trilinear
    # sampling reads all eight corners of a cell, so a cell diagonal to the
    # last metal voxel takes part in the interpolation right at the surface.
    # Left unmeasured at the range limit, it would drag the zero crossing
    # inward at every convex corner.
    structure = ndimage.generate_binary_structure(3, 3)
    grown = ndimage.binary_dilation(metal, structure=structure, iterations=_BAND_CELLS)
    shrunk = ndimage.binary_erosion(metal, structure=structure, iterations=_BAND_CELLS)
    band = grown & ~shrunk
    band_count = int(np.count_nonzero(band))
    if band_count == 0:
        return quantize(distance.reshape(-1))
    if band_count > MAX_SKIN_BAND_CELLS:
        logger.info("Skin grid skipped: %d band cells exceeds %d", band_count, MAX_SKIN_BAND_CELLS)
        return None

    bz, by, bx = np.nonzero(band)
    origin = np.asarray(lattice.origin, dtype=np.float64)
    points = origin + np.stack([bx, by, bz], axis=1).astype(np.float64) * spacing

    unsigned = np.full(band_count, np.inf, dtype=np.float64)
    # A second opinion on the sign, from the geometry alone: which side of its
    # nearest triangle each point is on. Wrong only where the nearest point is
    # an edge or a corner, which is exactly where the ray test below is spent.
    facing_inside = np.zeros(band_count, dtype=bool)
    for mesh in meshes:
        closest, part_distance, triangle = _closest_point(mesh, points)
        np.minimum(unsigned, part_distance, out=unsigned)
        normals = np.asarray(mesh.face_normals)[np.asarray(triangle, dtype=np.int64)]
        facing_inside |= np.einsum("ij,ij->i", points - closest, normals) < 0.0
    unsigned_voxels = unsigned / spacing

    inside = metal[bz, by, bx].copy()
    # The mask is trusted where it can be right: a cell further from every
    # surface than a voxel the mesher counted as metal can reach. Where the
    # geometry disagrees with it anyway -- a part moved since the run, an STL
    # swapped under the same name -- the ray decides, so the grid follows the
    # parts as they are drawn rather than as they were solved.
    doubtful = (unsigned_voxels <= _AMBIGUOUS_VOXELS) | (facing_inside != inside)
    if np.any(doubtful):
        resolved = _contains_any(meshes, points[doubtful])
        if resolved is not None:
            inside[doubtful] = resolved

    signed = np.where(inside, -unsigned_voxels, unsigned_voxels)
    distance[bz, by, bx] = np.clip(signed, -SKIN_RANGE_VOXELS, SKIN_RANGE_VOXELS)
    return quantize(distance.reshape(-1))


def _closest_point(mesh: Any, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import trimesh  # Deferred: this module is imported by the app factory.

    return trimesh.proximity.closest_point(mesh, points)


def _contains_any(meshes: Sequence[Any], points: np.ndarray) -> Optional[np.ndarray]:
    """
    Whether each point is inside at least one mesh, or ``None`` if untestable.

    Per part rather than on the concatenated soup: two parts that overlap --
    a riser seated into the plate it feeds -- would count a point inside both
    as crossing two surfaces, and ray parity would call it outside.
    """
    inside = np.zeros(len(points), dtype=bool)
    for mesh in meshes:
        try:
            result = np.asarray(mesh.contains(points), dtype=bool)
        except Exception as exc:  # noqa: BLE001 -- any failure means "unknown"
            logger.info("Skin grid: inside test failed (%s); keeping the mask's sign", exc)
            return None
        if result.shape != inside.shape:
            return None
        inside |= result
    return inside


# --------------------------------------------------------------------------
# Parts -> meshes, and the signature that says when they changed
# --------------------------------------------------------------------------


def parts_signature(lattice: Lattice, parts: Iterable[Dict[str, Any]]) -> Optional[str]:
    """
    A fingerprint of everything the grid depends on.

    The lattice, and for each part the file (path, size, mtime) and its
    placement. A part moved after the run, or an STL replaced under the same
    name, changes the fingerprint and the grid is rebuilt rather than trusted.
    Returns ``None`` when no part has a file on disk to measure.
    """
    entries: List[Any] = []
    for part in parts:
        filepath = str(part.get("filepath") or "")
        if not filepath:
            continue
        try:
            stat = os.stat(filepath)
        except OSError:
            continue
        entries.append(
            [
                filepath,
                int(stat.st_size),
                int(stat.st_mtime_ns),
                [float(part.get(f"position_{axis}", 0.0) or 0.0) for axis in "xyz"],
                [float(part.get(f"rotation_{axis}", 0.0) or 0.0) for axis in "xyz"],
                float(part.get("scale", 1.0) or 1.0),
            ]
        )
    if not entries:
        return None
    entries.sort(key=lambda entry: entry[0])
    payload = {
        "version": SKIN_VERSION,
        "range": SKIN_RANGE_VOXELS,
        "dims": [int(v) for v in lattice.dims],
        "origin": [float(v) for v in lattice.origin],
        "spacing": float(lattice.spacing),
        "parts": entries,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def part_meshes(parts: Sequence[Dict[str, Any]]) -> List[Any]:
    """
    The parts as world-space meshes, placed exactly as the run placed them.

    Goes through the same ``PartManager`` the simulation used, so scale,
    rotation order and translation match the voxelizer's rather than a second
    implementation of the same transform.
    """
    from web.payloads import build_part_manager  # Deferred: heavy import chain.

    present = [part for part in parts if part.get("filepath") and os.path.exists(str(part["filepath"]))]
    if not present:
        return []
    try:
        manager = build_part_manager(list(present))
    except Exception as exc:  # noqa: BLE001 -- a broken STL must not 500 the grid
        logger.info("Skin grid: parts could not be loaded (%s)", exc)
        return []
    meshes: List[Any] = []
    for part in manager.get_all_parts():
        try:
            meshes.append(part.get_transformed_mesh())
        except Exception as exc:  # noqa: BLE001
            logger.info("Skin grid: part %s could not be transformed (%s)", part.name, exc)
    return meshes


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


def skin_path(artifact_path: Path | str) -> Path:
    """``.../<id>.json`` -> ``.../<id>.skin.npz``."""
    path = Path(artifact_path)
    return path.with_name(path.stem + ".skin.npz")


def _save(path: Path, grid: SkinGrid) -> None:
    payload = {
        "version": np.asarray([SKIN_VERSION], dtype=np.int32),
        "range": np.asarray([grid.range_voxels], dtype=np.float64),
        "signature": np.frombuffer(grid.signature.encode("utf-8"), dtype=np.uint8),
        "values": np.asarray(grid.values, dtype=np.uint8),
    }
    # Write-then-rename, like the grid sidecar: a reader must never see half
    # a file, and two processes racing must not interleave their writes.
    handle, temp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _load(path: Path, signature: str) -> Optional[SkinGrid]:
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"][0]) != SKIN_VERSION:
                return None
            stored = bytes(data["signature"]).decode("utf-8")
            if stored != signature:
                return None
            return SkinGrid(
                values=np.asarray(data["values"], dtype=np.uint8),
                range_voxels=float(data["range"][0]),
                signature=stored,
            )
    except Exception:
        # A cache entry that cannot be read is not a cache entry.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


_MEMORY_MAX = 8
_memory: "OrderedDict[Tuple[str, str], SkinGrid]" = OrderedDict()
_lock = threading.Lock()


def clear_cache() -> None:
    with _lock:
        _memory.clear()


def load_skin_grid(
    project_id: str,
    lattice: Lattice,
    mask: np.ndarray,
    parts: Sequence[Dict[str, Any]],
    artifact_path: Optional[Path | str],
) -> Optional[SkinGrid]:
    """
    The skin grid for a run: from memory, from disk, or built and cached.

    ``None`` when it cannot be built -- no part files, an unloadable STL, a
    lattice too large -- in which case the viewport falls back to clipping
    against the mask, which is what it always did.
    """
    signature = parts_signature(lattice, parts)
    if signature is None:
        return None

    key = (project_id, signature)
    with _lock:
        cached = _memory.get(key)
        if cached is not None:
            _memory.move_to_end(key)
            return cached

    disk = skin_path(artifact_path) if artifact_path is not None else None
    grid: Optional[SkinGrid] = None
    if disk is not None and disk.exists():
        grid = _load(disk, signature)

    if grid is None:
        values = build_skin_grid(lattice, mask, part_meshes(parts))
        if values is None:
            return None
        grid = SkinGrid(values=values, range_voxels=SKIN_RANGE_VOXELS, signature=signature)
        if disk is not None:
            try:
                _save(disk, grid)
            except OSError:  # pragma: no cover - read-only artifact directory
                pass

    with _lock:
        _memory[key] = grid
        _memory.move_to_end(key)
        while len(_memory) > _MEMORY_MAX:
            _memory.popitem(last=False)
    return grid
