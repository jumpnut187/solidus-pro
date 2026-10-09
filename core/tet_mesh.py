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
import os
import sys
import tempfile
from contextlib import contextmanager
from typing import Callable, Iterator, Sequence

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from config.exceptions import MeshGenerationError
from core.part_manager import (
    METAL_PART_CODES,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_FILTER,
    is_metal,
)

ProgressCallback = Callable[[float, str], None]

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
    fd redirect alone does not silence it.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = sys.stdout
    libc = _libc()
    try:
        with open(os.devnull, "w", encoding="utf-8") as python_devnull:
            os.dup2(devnull, 1)
            sys.stdout = python_devnull
            try:
                yield
            finally:
                if libc is not None:
                    libc.fflush(None)  # drain C stdio buffers into devnull, not the restored fd
                sys.stdout = saved_stdout
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def _libc() -> ctypes.CDLL | None:
    try:
        return ctypes.CDLL(None)
    except OSError:
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
    tested with ``Trimesh.contains``.
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
        inside = mesh.contains(centroids[candidates])
        tet_codes[candidates[inside]] = codes[part]
    return tet_codes


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
