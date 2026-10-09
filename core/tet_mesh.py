"""Conforming tetrahedral volume mesh of a casting assembly (contract C1).

``build_tet_mesh`` meshes every part in one fTetWild CSG union, so parts that
touch share nodes on their interface. CORE and FILTER tets are removed after
meshing; the faces they expose become boundary faces with a contact code.

``TetVolumeMesh`` duck-types ``AdaptiveVoxelMesh``: the ``voxel_*`` arrays
exist with node semantics (one entry per tet node, in node order). All lengths
are inches.

pytetwild imports pyvista/vtk at package import, so it is imported inside
``build_tet_mesh`` only.
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import sys
import tempfile
import threading
from contextlib import contextmanager
from typing import Callable, Iterator, Sequence

import numpy as np
import trimesh
from numba import njit, prange
from scipy.spatial import cKDTree

from config.exceptions import MeshGenerationError
from core.part_manager import (
    METAL_PART_CODES,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_FILTER,
    is_metal,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

#: Held for the whole stdout-silencing context, native fTetWild call included.
#: The context swaps process-wide state (fd 1 and ``sys.stdout``), so two
#: threads inside it at once restore each other's devnull. fTetWild's own
#: thread safety is unknown, so serializing the native call is wanted anyway.
_NATIVE_MESH_LOCK = threading.Lock()

#: Point x triangle pairs above which ``winding_number_contains`` falls back
#: to the voxel path's containment check. Measured 1.8e9 pairs/s on 18
#: cores, so the cap is about 6 s there.
WINDING_NUMBER_MAX_PAIRS = 10_000_000_000

#: Part codes whose tets are removed from the thermal domain after meshing.
REMOVED_PART_CODES: tuple[int, ...] = (PART_TYPE_CORE, PART_TYPE_FILTER)

#: Boundary faces (nearest by centroid) tested per node for ``voxel_moduli``.
_MODULUS_CANDIDATES = 16

#: Tets whose smallest dihedral angle is below this are counted as slivers.
SLIVER_DIHEDRAL_DEG = 5.0

# Outward faces of a positively oriented tet (v0, v1, v2, v3). Face k is the
# face opposite vertex k.
_TET_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int64)
_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
# Face pairs (k, l) sharing each edge: faces opposite the two vertices not on it.
_EDGE_FACE_PAIRS = np.array([[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]], dtype=np.int64)


class TetVolumeMesh:
    """Tet mesh of the thermal domain with ``AdaptiveVoxelMesh``-style node arrays.

    Construct with ``build_tet_mesh``. The constructor takes an already
    extracted domain: positively oriented ``tets`` over ``node_xyz`` with no
    orphan nodes, and outward ``boundary_faces``.
    """

    mesh_type = "tet"

    def __init__(
        self,
        node_xyz: np.ndarray,
        tets: np.ndarray,
        tet_part_types: np.ndarray,
        boundary_faces: np.ndarray,
        boundary_face_part_types: np.ndarray,
        boundary_face_contact_codes: np.ndarray,
        *,
        base_voxel_size: float,
        metal_part_bounds: Sequence[np.ndarray] = (),
    ) -> None:
        self.node_xyz = np.ascontiguousarray(node_xyz, dtype=np.float64)
        self.tets = np.ascontiguousarray(tets, dtype=np.int32)
        self.tet_part_types = np.ascontiguousarray(tet_part_types, dtype=np.int8)
        self.boundary_faces = np.ascontiguousarray(boundary_faces, dtype=np.int32).reshape(-1, 3)
        self.boundary_face_part_types = np.ascontiguousarray(boundary_face_part_types, dtype=np.int8)
        self.boundary_face_contact_codes = np.ascontiguousarray(
            boundary_face_contact_codes, dtype=np.int8
        )
        n_nodes = len(self.node_xyz)
        if n_nodes == 0 or len(self.tets) == 0:
            raise MeshGenerationError("tet mesh is empty", voxel_size=base_voxel_size)

        self.tet_volumes = _tet_signed_volumes(self.node_xyz, self.tets)
        if not np.all(self.tet_volumes > 0.0):
            raise MeshGenerationError(
                f"{int(np.sum(self.tet_volumes <= 0.0))} tets are not positively oriented",
                voxel_size=base_voxel_size,
            )
        self.boundary_face_areas = _triangle_areas(self.node_xyz, self.boundary_faces)

        self.base_voxel_size = float(base_voxel_size)
        self.original_voxel_size = float(base_voxel_size)
        self.voxel_size_was_adjusted = False
        self.adjustment_message = ""
        self.grid_shape = None
        self.bounds = np.column_stack([self.node_xyz.min(axis=0), self.node_xyz.max(axis=0)])
        self.metal_part_bounds = [np.asarray(b, dtype=np.float64) for b in metal_part_bounds]

        tet_nodes = self.tets.reshape(-1)
        self.voxel_centers = self.node_xyz.astype(np.float32)
        self.voxel_volumes = np.bincount(
            tet_nodes, weights=np.repeat(self.tet_volumes / 4.0, 4), minlength=n_nodes
        ).astype(np.float32)
        self.voxel_surface_areas = np.bincount(
            self.boundary_faces.reshape(-1),
            weights=np.repeat(self.boundary_face_areas / 3.0, 3),
            minlength=n_nodes,
        ).astype(np.float32)
        self.surface_voxels = self.voxel_surface_areas > 0

        edges = _unique_edges(self.tets)
        self.adj_indptr, self.adj_indices = _edge_csr(edges, n_nodes)
        self.voxel_neighbors = _padded_neighbors(self.adj_indptr, self.adj_indices)
        edge_lengths = np.linalg.norm(self.node_xyz[edges[:, 0]] - self.node_xyz[edges[:, 1]], axis=1)
        self.mean_edge_length = float(edge_lengths.mean())
        degree = np.diff(self.adj_indptr)
        edge_length_sum = np.bincount(
            edges.reshape(-1), weights=np.repeat(edge_lengths, 2), minlength=n_nodes
        )
        self.voxel_sizes = (edge_length_sum / np.maximum(degree, 1)).astype(np.float32)

        self.voxel_part_types = _node_part_types(self.tets, self.tet_part_types, n_nodes)
        self.voxel_moduli = _distance_to_boundary(
            self.node_xyz, self.boundary_faces, self.surface_voxels
        ).astype(np.float32)
        self.voxel_states = np.zeros(n_nodes, dtype=np.float32)
        self._material_surface_indices: np.ndarray | None = None

    # AdaptiveVoxelMesh mirror ------------------------------------------------

    def get_voxel_count(self) -> int:
        """Number of nodes (the DOF count)."""
        return len(self.node_xyz)

    def get_surface_voxel_indices(self) -> np.ndarray:
        """Indices of nodes on a boundary face."""
        return np.flatnonzero(self.surface_voxels)

    def get_voxels_by_type(self, part_type: int) -> np.ndarray:
        """Indices of nodes whose node part code equals ``part_type``."""
        return np.flatnonzero(self.voxel_part_types == part_type)

    def get_material_surface_voxel_indices(self) -> np.ndarray:
        """Boundary nodes plus metal nodes shared with a CHILL tet.

        Nodes on faces with contact code CORE are boundary nodes already, so
        they are included through the geometric surface.
        """
        if self._material_surface_indices is not None:
            return self._material_surface_indices
        mask = self.surface_voxels.copy()
        chill_tets = self.tet_part_types == PART_TYPE_CHILL
        if np.any(chill_tets):
            touches_chill = np.zeros(len(self.node_xyz), dtype=bool)
            touches_chill[self.tets[chill_tets].reshape(-1)] = True
            mask |= touches_chill & is_metal(self.voxel_part_types)
        self._material_surface_indices = np.flatnonzero(mask)
        return self._material_surface_indices

    def get_memory_stats(self) -> dict:
        """Memory statistics with the keys ``MemoryManager.get_memory_stats`` returns."""
        import psutil

        arrays = [v for v in vars(self).values() if isinstance(v, np.ndarray)]
        ram_mb = sum(a.nbytes for a in arrays) / (1024**2)
        vm = psutil.virtual_memory()
        return {
            "ram_mb": ram_mb,
            "disk_mb": 0.0,
            "total_mb": ram_mb,
            "available_gb": vm.available / (1024**3),
            "total_system_gb": vm.total / (1024**3),
            "array_count": len(arrays),
            "mode": "ram",
        }

    def print_memory_stats(self) -> None:
        """Print ``get_memory_stats``."""
        stats = self.get_memory_stats()
        print(
            f"Tet mesh memory: RAM={stats['ram_mb']:.1f}MB, arrays={stats['array_count']}, "
            f"available={stats['available_gb']:.1f}GB"
        )

    def cleanup(self) -> None:
        """No-op. Every array is in RAM, there are no memmap files to remove."""

    # Tet-only ----------------------------------------------------------------

    def quality_summary(self) -> dict:
        """Dihedral-angle quality (degrees) and element counts."""
        dihedral = _dihedral_angles_deg(self.node_xyz, self.tets)
        per_tet_min = dihedral.min(axis=1)
        return {
            "min_dihedral_deg": float(per_tet_min.min()),
            "median_dihedral_deg": float(np.median(dihedral)),
            "sliver_count": int(np.sum(per_tet_min < SLIVER_DIHEDRAL_DEG)),
            "sliver_threshold_deg": SLIVER_DIHEDRAL_DEG,
            "node_count": int(len(self.node_xyz)),
            "tet_count": int(len(self.tets)),
            "boundary_face_count": int(len(self.boundary_faces)),
        }


def build_tet_mesh(
    parts: Sequence[tuple[trimesh.Trimesh, int]],
    edge_length_in: float,
    *,
    epsilon_rel: float = 1e-3,
    stop_energy: float = 10.0,
    num_threads: int = 0,
    progress_callback: ProgressCallback | None = None,
) -> TetVolumeMesh:
    """Mesh ``parts`` (world coords, inches, part code) into one conforming tet mesh.

    ``edge_length_in`` is the target tet edge length in inches. ``epsilon_rel``
    is the fTetWild envelope relative to the assembly bounding-box diagonal.
    Raises ``MeshGenerationError`` on failure, empty output or non-finite
    coordinates.
    """
    def report(fraction: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(fraction, message)

    meshes = [m for m, _ in parts]
    codes = np.array([int(c) for _, c in parts], dtype=np.int64)
    if len(meshes) == 0:
        raise MeshGenerationError("no parts to mesh", voxel_size=edge_length_in)
    if not (np.isfinite(edge_length_in) and edge_length_in > 0.0):
        raise MeshGenerationError(f"invalid edge length {edge_length_in!r}", voxel_size=edge_length_in)
    for mesh in meshes:
        if len(mesh.faces) == 0 or not np.all(np.isfinite(mesh.vertices)):
            raise MeshGenerationError("part mesh is empty or has non-finite vertices", voxel_size=edge_length_in)

    report(0.0, "Tetrahedralizing parts")
    try:
        vertices, tets, markers = _tetrahedralize(
            meshes, edge_length_in, epsilon_rel, stop_energy, num_threads
        )
    except MeshGenerationError:
        raise
    except Exception as exc:  # native failures surface as RuntimeError/ValueError
        raise MeshGenerationError(f"fTetWild failed: {exc}", voxel_size=edge_length_in) from exc
    if len(tets) == 0 or len(vertices) == 0:
        raise MeshGenerationError("fTetWild returned an empty mesh", voxel_size=edge_length_in)
    if not np.all(np.isfinite(vertices)):
        raise MeshGenerationError("fTetWild returned non-finite coordinates", voxel_size=edge_length_in)
    if markers.min() < 1 or markers.max() > len(meshes):
        raise MeshGenerationError(
            f"fTetWild markers {markers.min()}..{markers.max()} outside 1..{len(meshes)}",
            voxel_size=edge_length_in,
        )

    report(0.7, "Assigning part codes")
    tets = _orient_positive(vertices, tets)
    tet_codes = _resolve_overlap_codes(vertices, tets, markers - 1, meshes, codes)

    report(0.8, "Building node arrays")
    domain = _extract_thermal_domain(vertices, tets, tet_codes)
    node_xyz, kept_tets, kept_codes, boundary_faces, owner_codes, contact_codes = domain
    if len(kept_tets) == 0:
        raise MeshGenerationError(
            "no tets remain after removing CORE/FILTER parts", voxel_size=edge_length_in
        )
    metal_part_bounds = [
        np.array(m.bounds, dtype=np.float64)
        for m, c in zip(meshes, codes)
        if int(c) in METAL_PART_CODES
    ]
    mesh = TetVolumeMesh(
        node_xyz,
        kept_tets,
        kept_codes,
        boundary_faces,
        owner_codes,
        contact_codes,
        base_voxel_size=edge_length_in,
        metal_part_bounds=metal_part_bounds,
    )
    report(1.0, f"Tet mesh ready: {mesh.get_voxel_count()} nodes, {len(mesh.tets)} tets")
    return mesh


# Meshing ----------------------------------------------------------------------


def _tetrahedralize(
    meshes: Sequence[trimesh.Trimesh],
    edge_length_in: float,
    epsilon_rel: float,
    stop_energy: float,
    num_threads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run fTetWild. Returns vertices (in), tets, and 1-based leaf markers.

    One part uses ``pytetwild.tetrahedralize`` with an absolute edge length.
    Several parts use one left-deep CSG union; marker k is the k-th leaf,
    which is the k-th input part.
    """
    import pytetwild
    from pytetwild import PyfTetWildWrapper

    if len(meshes) == 1:
        mesh = meshes[0]
        with _silence_c_stdout():
            vertices, tets = pytetwild.tetrahedralize(
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int32),
                edge_length_abs=edge_length_in,
                epsilon=epsilon_rel,
                stop_energy=stop_energy,
                coarsen=False,
                num_threads=num_threads,
            )
        tets = np.asarray(tets, dtype=np.int64)
        return np.asarray(vertices, dtype=np.float64), tets, np.ones(len(tets), dtype=np.int64)

    all_vertices = np.vstack([np.asarray(m.vertices, dtype=np.float64) for m in meshes])
    diagonal = float(np.linalg.norm(all_vertices.max(axis=0) - all_vertices.min(axis=0)))
    edge_length_r = min(edge_length_in / diagonal, 1.0)
    with tempfile.TemporaryDirectory(prefix="castsim_tet_") as tmp:
        tree: str | dict = ""
        for index, mesh in enumerate(meshes):
            path = os.path.join(tmp, f"part_{index}.obj")
            mesh.export(path, file_type="obj")
            tree = path if index == 0 else {"operation": "union", "left": tree, "right": path}
        csg_path = os.path.join(tmp, "csg.json")
        with open(csg_path, "w", encoding="utf-8") as handle:
            json.dump(tree, handle)
        with _silence_c_stdout():
            vertices, tets, markers = PyfTetWildWrapper.tetrahedralize_csg(
                csg_path, epsilon_rel, edge_length_r, stop_energy, False, num_threads, 6, False
            )
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(tets, dtype=np.int64),
        np.asarray(markers, dtype=np.int64),
    )


@contextmanager
def _silence_c_stdout() -> Iterator[None]:
    """Send fd 1 and ``sys.stdout`` to os.devnull; fTetWild prints regardless of its log level.

    The CSG wrapper forwards ``std::cout`` to Python's ``sys.stdout``, so the
    fd redirect alone does not silence it. Holds ``_NATIVE_MESH_LOCK`` for the
    whole context. ``sys.stdout`` and fd 1 are restored even when opening a
    descriptor, the body or the C stdio flush fails.
    """
    with _NATIVE_MESH_LOCK:
        saved_stdout = sys.stdout
        saved_fd: int | None = None
        devnull_fd: int | None = None
        python_devnull = None
        try:
            try:
                saved_stdout.flush()
            except (AttributeError, OSError, ValueError):
                pass  # None (pythonw) or a closed stream: nothing to flush
            saved_fd = os.dup(1)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            python_devnull = open(os.devnull, "w", encoding="utf-8")
            os.dup2(devnull_fd, 1)
            sys.stdout = python_devnull
            yield
        finally:
            sys.stdout = saved_stdout
            # Drain C stdio buffers while fd 1 still points at devnull.
            _flush_c_stdio()
            if saved_fd is not None:
                os.dup2(saved_fd, 1)
                os.close(saved_fd)
            if devnull_fd is not None:
                os.close(devnull_fd)
            if python_devnull is not None:
                python_devnull.close()


def _flush_c_stdio() -> None:
    """``fflush(NULL)`` in the C runtime, skipped when the runtime or its ``fflush`` is unavailable."""
    try:
        libc = _libc()
        fflush = getattr(libc, "fflush", None) if libc is not None else None
        if fflush is not None:
            fflush(None)
    except Exception:
        pass


def _libc() -> ctypes.CDLL | None:
    """The C runtime that owns the process's stdio, or None.

    ``ctypes.CDLL(None)`` raises TypeError on Windows, where the CRT is
    ``ucrtbase`` (MSVC 2015+) or the legacy ``msvcrt``.
    """
    names: tuple[str | None, ...] = ("ucrtbase", "msvcrt") if sys.platform == "win32" else (None,)
    for name in names:
        try:
            return ctypes.CDLL(name)
        except Exception:
            continue
    return None


# Topology ---------------------------------------------------------------------


def _tet_signed_volumes(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Signed tet volumes (in^3), positive for positive orientation."""
    v0 = vertices[tets[:, 0]]
    cross = np.cross(vertices[tets[:, 1]] - v0, vertices[tets[:, 2]] - v0)
    return np.einsum("ij,ij->i", cross, vertices[tets[:, 3]] - v0) / 6.0


def _orient_positive(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Swap vertices 2 and 3 of every negatively oriented tet."""
    tets = np.array(tets, dtype=np.int64, copy=True)
    negative = _tet_signed_volumes(vertices, tets) < 0.0
    tets[negative, 2:4] = tets[negative][:, [3, 2]]
    return tets


def _resolve_overlap_codes(
    vertices: np.ndarray,
    tets: np.ndarray,
    leaf_index: np.ndarray,
    meshes: Sequence[trimesh.Trimesh],
    codes: np.ndarray,
) -> np.ndarray:
    """Part code per tet: marker code, raised to the highest code of any part containing the centroid.

    Only parts whose bounding box overlaps the marker part's box, and only
    centroids inside the testing part's box with a lower current code, are
    tested. Watertight parts use ``Trimesh.contains``. Open parts use the
    generalized winding number, the rule fTetWild meshes them by.
    """
    tet_codes = codes[leaf_index].copy()
    centroids = vertices[tets].mean(axis=1)
    boxes = np.array([m.bounds for m in meshes], dtype=np.float64)  # (P, 2, 3)
    for part, mesh in enumerate(meshes):
        lo, hi = boxes[part]
        overlaps = np.all((boxes[:, 0] <= hi) & (boxes[:, 1] >= lo), axis=1)
        overlaps[part] = False
        candidates = np.flatnonzero(
            overlaps[leaf_index]
            & (tet_codes < codes[part])
            & np.all((centroids >= lo) & (centroids <= hi), axis=1)
        )
        if len(candidates) == 0:
            continue
        if mesh.is_watertight:
            inside = mesh.contains(centroids[candidates])
        else:
            inside = winding_number_contains(mesh, centroids[candidates])
        tet_codes[candidates[inside]] = codes[part]
    return tet_codes


def winding_number_contains(
    mesh: trimesh.Trimesh, points: np.ndarray, *, near_surface_tol: float = 0.0
) -> np.ndarray:
    """(P,) bool: generalized winding number of ``mesh`` at each point above 0.5.

    Valid for open and non-manifold surfaces, where ``Trimesh.contains`` ray
    tests are not. Points within ``near_surface_tol`` (inches) of the surface
    also count as inside. Above ``WINDING_NUMBER_MAX_PAIRS`` point x triangle
    pairs it logs a warning and returns
    ``core.voxel_mesh._robust_contains_check_impl(mesh, points, near_surface_tol)``.
    """
    from core.voxel_mesh import _near_surface_mask, _robust_contains_check_impl

    query = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    if len(query) == 0 or len(mesh.faces) == 0:
        return np.zeros(len(query), dtype=bool)
    pairs = len(query) * len(mesh.faces)
    if pairs > WINDING_NUMBER_MAX_PAIRS:
        logger.warning(
            "winding-number containment skipped: %d points x %d faces exceeds %d pairs; "
            "using the ray-test containment check, which is unreliable on open surfaces",
            len(query),
            len(mesh.faces),
            WINDING_NUMBER_MAX_PAIRS,
        )
        return np.asarray(_robust_contains_check_impl(mesh, query, near_surface_tol), dtype=bool)
    triangles = np.ascontiguousarray(mesh.triangles, dtype=np.float64)
    inside = _winding_numbers(triangles, query) > 0.5
    if near_surface_tol > 0.0:
        inside |= np.asarray(_near_surface_mask(mesh, query, near_surface_tol), dtype=bool)
    return inside


@njit(parallel=True, cache=True)
def _winding_numbers(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(P,) generalized winding numbers: sum of signed triangle solid angles / 4 pi.

    Solid angle by Van Oosterom and Strackee (1983).
    """
    n_points = points.shape[0]
    n_faces = triangles.shape[0]
    result = np.empty(n_points, dtype=np.float64)
    for i in prange(n_points):
        px = points[i, 0]
        py = points[i, 1]
        pz = points[i, 2]
        total = 0.0
        for f in range(n_faces):
            ax = triangles[f, 0, 0] - px
            ay = triangles[f, 0, 1] - py
            az = triangles[f, 0, 2] - pz
            bx = triangles[f, 1, 0] - px
            by = triangles[f, 1, 1] - py
            bz = triangles[f, 1, 2] - pz
            cx = triangles[f, 2, 0] - px
            cy = triangles[f, 2, 1] - py
            cz = triangles[f, 2, 2] - pz
            la = math.sqrt(ax * ax + ay * ay + az * az)
            lb = math.sqrt(bx * bx + by * by + bz * bz)
            lc = math.sqrt(cx * cx + cy * cy + cz * cz)
            det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
            den = (
                la * lb * lc
                + (ax * bx + ay * by + az * bz) * lc
                + (bx * cx + by * cy + bz * cz) * la
                + (cx * ax + cy * ay + cz * az) * lb
            )
            total += 2.0 * math.atan2(det, den)
        result[i] = total / (4.0 * math.pi)
    return result


def _face_partners(tets: np.ndarray) -> np.ndarray:
    """For each of the 4T tet faces (row-major, face k opposite vertex k), the matching face index or -1."""
    faces = np.sort(tets[:, _TET_FACES].reshape(-1, 3), axis=1)
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    sorted_faces = faces[order]
    same = np.all(sorted_faces[1:] == sorted_faces[:-1], axis=1)
    first = order[:-1][same]
    second = order[1:][same]
    partner = np.full(len(faces), -1, dtype=np.int64)
    partner[first] = second
    partner[second] = first
    return partner


def _extract_thermal_domain(
    vertices: np.ndarray, tets: np.ndarray, tet_codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove CORE/FILTER tets, re-index nodes, and find outward boundary faces.

    Returns node_xyz, tets, tet codes, boundary faces, per-face owner codes,
    and per-face contact codes (code of the removed tet across the face,
    else -1).
    """
    keep = ~np.isin(tet_codes, REMOVED_PART_CODES)
    partner = _face_partners(tets)
    face_tet = np.arange(partner.size) // 4
    partner_tet = np.where(partner >= 0, partner // 4, -1)
    kept_face = keep[face_tet]
    partner_kept = np.where(partner_tet >= 0, keep[np.maximum(partner_tet, 0)], False)
    boundary = np.flatnonzero(kept_face & ~partner_kept)
    contact = np.where(
        partner_tet[boundary] >= 0, tet_codes[np.maximum(partner_tet[boundary], 0)], -1
    ).astype(np.int8)
    boundary_faces = tets[face_tet[boundary][:, None], _TET_FACES[boundary % 4]]

    kept_tets = tets[keep]
    used = np.unique(kept_tets)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return (
        vertices[used],
        remap[kept_tets].astype(np.int32),
        tet_codes[keep].astype(np.int8),
        remap[boundary_faces].astype(np.int32),
        tet_codes[face_tet[boundary]].astype(np.int8),
        contact,
    )


def _unique_edges(tets: np.ndarray) -> np.ndarray:
    """Unique undirected edges (E, 2) int64 with column 0 < column 1."""
    edges = np.sort(tets.astype(np.int64)[:, _TET_EDGES].reshape(-1, 2), axis=1)
    n = int(tets.max()) + 1
    keys = np.unique(edges[:, 0] * n + edges[:, 1])
    return np.column_stack([keys // n, keys % n])


def _edge_csr(edges: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric CSR adjacency (int64 indptr, int32 indices, rows ascending)."""
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.lexsort((cols, rows))
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n_nodes), out=indptr[1:])
    return indptr, cols[order].astype(np.int32)


def _padded_neighbors(indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """(N, K) int32 neighbour table padded with -1, K = max degree."""
    degree = np.diff(indptr)
    n_nodes = len(degree)
    table = np.full((n_nodes, int(degree.max(initial=0))), -1, dtype=np.int32)
    rows = np.repeat(np.arange(n_nodes), degree)
    slots = np.arange(len(indices)) - indptr[rows]
    table[rows, slots] = indices
    return table


def _node_part_types(tets: np.ndarray, tet_codes: np.ndarray, n_nodes: int) -> np.ndarray:
    """Node code: max metal code of incident metal tets, else max code of incident tets."""
    nodes = tets.reshape(-1)
    codes = np.repeat(tet_codes.astype(np.int16), 4)
    metal = np.repeat(is_metal(tet_codes), 4)
    any_max = np.full(n_nodes, -1, dtype=np.int16)
    np.maximum.at(any_max, nodes, codes)
    metal_max = np.full(n_nodes, -1, dtype=np.int16)
    np.maximum.at(metal_max, nodes[metal], codes[metal])
    return np.where(metal_max >= 0, metal_max, any_max).astype(np.int8)


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Triangle areas (in^2)."""
    if len(faces) == 0:
        return np.zeros(0, dtype=np.float64)
    tri = vertices[faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def _distance_to_boundary(
    points: np.ndarray, faces: np.ndarray, on_boundary: np.ndarray
) -> np.ndarray:
    """Distance (in) from each point to the nearest boundary triangle, 0 on the boundary.

    Exact point-triangle distance to the ``_MODULUS_CANDIDATES`` faces with
    the nearest centroids. If the true nearest face is not among them, the
    result overestimates by at most the largest centroid-to-vertex distance
    of a boundary face (about 0.6 h). A provably exact certificate needed
    ~30x the work on a 2 M tet mesh for no measured change.
    """
    result = np.zeros(len(points), dtype=np.float64)
    pending = np.flatnonzero(~on_boundary)
    if len(pending) == 0 or len(faces) == 0:
        return result
    triangles = points[faces]
    tree = cKDTree(triangles.mean(axis=1))
    k = min(_MODULUS_CANDIDATES, len(faces))
    chunk = 500_000
    for start in range(0, len(pending), chunk):
        ids = pending[start : start + chunk]
        query = points[ids]
        _, candidates = tree.query(query, k=k, workers=-1)
        candidates = candidates.reshape(len(ids), k)
        best = np.full(len(ids), np.inf)
        for column in range(k):
            closest = trimesh.triangles.closest_point(triangles[candidates[:, column]], query)
            np.minimum(best, np.linalg.norm(closest - query, axis=1), out=best)
        result[ids] = best
    return result


def _dihedral_angles_deg(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """(T, 6) interior dihedral angles in degrees."""
    tri = vertices[tets[:, _TET_FACES]]  # (T, 4, 3, 3)
    normals = np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0])
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    cos = -np.einsum(
        "tij,tij->ti", normals[:, _EDGE_FACE_PAIRS[:, 0]], normals[:, _EDGE_FACE_PAIRS[:, 1]]
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
