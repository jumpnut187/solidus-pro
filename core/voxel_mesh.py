import itertools
import numpy as np
import os
from typing import Tuple, List, Dict, Optional, Union, Sequence
import trimesh
from core.part_manager import (
    CODE_TO_PART_TYPE,
    METAL_PART_CODES,
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_FILTER,
    PART_TYPE_GATE,
    PART_TYPE_POUR_CUP,
    PART_TYPE_RISER,
    PART_TYPE_RISER_POST,
    PART_TYPE_RUNNER,
    PART_TYPE_SPRUE,
    is_mold,
)
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

_INTERFACE_BRIDGE_PAIRS = (
    (PART_TYPE_CAST, PART_TYPE_RISER),
    (PART_TYPE_CAST, PART_TYPE_CHILL),
    (PART_TYPE_CAST, PART_TYPE_RISER_POST),
    (PART_TYPE_RISER_POST, PART_TYPE_RISER),
    (PART_TYPE_POUR_CUP, PART_TYPE_SPRUE),
    (PART_TYPE_SPRUE, PART_TYPE_RUNNER),
    (PART_TYPE_SPRUE, PART_TYPE_FILTER),
    (PART_TYPE_SPRUE, PART_TYPE_GATE),
    (PART_TYPE_RUNNER, PART_TYPE_FILTER),
    (PART_TYPE_RUNNER, PART_TYPE_GATE),
    (PART_TYPE_FILTER, PART_TYPE_GATE),
    (PART_TYPE_GATE, PART_TYPE_CAST),
    (PART_TYPE_GATE, PART_TYPE_RISER),
)

_INTERFACE_BRIDGE_CODES = tuple(
    sorted({code for pair in _INTERFACE_BRIDGE_PAIRS for code in pair})
)
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
from core.acceleration import (
    compute_modulus_parallel
)
from core.memory_manager import create_managed_array, get_memory_manager
from core.memory_limits import (
    validate_mesh_parameters,
    auto_adjust_voxel_size,
    MemoryMonitor,
    MemorySeverity,
    get_max_voxels_for_system,
    estimate_voxel_count,
    estimate_memory_usage_mb,
)
from config.exceptions import BoundsError, InvalidMeshError, MeshGenerationError
from config.hardware_config import hardware

try:
    from core.error_logger import logger
except ImportError:
    from core.error_logger_dummy import logger

try:
    from core.gpu_compute import gpu_compute, compute_distances_accelerated
    GPU_AVAILABLE = gpu_compute.is_gpu_available
except ImportError:
    GPU_AVAILABLE = False

try:
    from core.octree import Octree, SpatialIndex
    OCTREE_AVAILABLE = True
except ImportError:
    OCTREE_AVAILABLE = False

RTREE_AVAILABLE = True
try:
    from rtree import index
except ImportError:
    RTREE_AVAILABLE = False

_PART_ASSIGNMENT_WORKER_MESH: Optional[Tuple[str, trimesh.Trimesh]] = None

_PROXIMITY_CHUNK_POINTS = 512

_WORKER_BASE_BYTES = 256 * 1024 ** 2

def _cut_cells_enabled() -> bool:
    try:
        from config.settings import get_settings

        return bool(getattr(get_settings().mesh, "voxel_cut_cells", False))
    except Exception:
        return False

def _available_memory_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return int(getattr(hardware, "available_memory", 0) or 0)

def _load_part_assignment_mesh(part_key: str, mesh_path: str) -> trimesh.Trimesh:
    global _PART_ASSIGNMENT_WORKER_MESH
    cached = _PART_ASSIGNMENT_WORKER_MESH
    if cached is not None and cached[0] == part_key:
        return cached[1]
    _PART_ASSIGNMENT_WORKER_MESH = None
    with np.load(mesh_path) as payload:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(payload["vertices"]),
            faces=np.asarray(payload["faces"]),
            process=False,
        )
    _PART_ASSIGNMENT_WORKER_MESH = (part_key, mesh)
    return mesh

def _near_surface_mask(mesh: trimesh.Trimesh, points: np.ndarray, tolerance: float) -> np.ndarray:
    n_points = len(points)
    result = np.zeros(n_points, dtype=bool)
    if n_points == 0 or tolerance <= 0:
        return result
    tree = mesh.triangles_tree
    lo = points - tolerance
    hi = points + tolerance
    pair_points: List[int] = []
    pair_tris: List[int] = []
    for i in range(n_points):
        hits = list(tree.intersection((lo[i, 0], lo[i, 1], lo[i, 2], hi[i, 0], hi[i, 1], hi[i, 2])))
        if hits:
            pair_points.extend([i] * len(hits))
            pair_tris.extend(hits)
    if not pair_points:
        return result
    pp = np.asarray(pair_points, dtype=np.int64)
    pt = np.asarray(pair_tris, dtype=np.int64)
    closest = trimesh.triangles.closest_point(mesh.triangles[pt], points[pp])
    dist = np.linalg.norm(closest - points[pp], axis=1)
    result[np.unique(pp[dist < tolerance])] = True
    return result

def _robust_contains_check_impl(mesh: trimesh.Trimesh, points: np.ndarray, tolerance: float) -> np.ndarray:
    n_points = len(points)
    inside_mask = np.zeros(n_points, dtype=bool)

    try:
        inside_mask = mesh.contains(points)
        if np.sum(inside_mask) == 0:
            logger.debug(f"mesh.contains() returned all False for {n_points} points, trying fallbacks")
        else:
            try:
                outside_idx = np.flatnonzero(~inside_mask)
                for start in range(0, len(outside_idx), _PROXIMITY_CHUNK_POINTS):
                    chunk = outside_idx[start:start + _PROXIMITY_CHUNK_POINTS]
                    inside_mask[chunk[_near_surface_mask(mesh, points[chunk], tolerance)]] = True
            except Exception:
                pass
            return inside_mask
    except Exception as e:
        logger.warning(f"mesh.contains() failed: {e}, trying fallbacks")

    try:
        from trimesh.proximity import signed_distance
        signed_dists = signed_distance(mesh, points)
        inside_mask = signed_dists <= tolerance
        if np.sum(inside_mask) > 0:
            return inside_mask
    except Exception as e:
        logger.debug(f"signed_distance fallback failed: {e}")

    try:
        inside_mask = np.zeros(n_points, dtype=bool)
        for start in range(0, n_points, _PROXIMITY_CHUNK_POINTS):
            stop = start + _PROXIMITY_CHUNK_POINTS
            inside_mask[start:stop] = _near_surface_mask(mesh, points[start:stop], tolerance * 1.5)
        if np.sum(inside_mask) > 0:
            return inside_mask
    except Exception as e:
        logger.debug(f"proximity fallback failed: {e}")

    try:
        bbox_min = mesh.bounds[0] - tolerance * 0.5
        bbox_max = mesh.bounds[1] + tolerance * 0.5
        inside_mask = (
            (points[:, 0] >= bbox_min[0]) & (points[:, 0] <= bbox_max[0]) &
            (points[:, 1] >= bbox_min[1]) & (points[:, 1] <= bbox_max[1]) &
            (points[:, 2] >= bbox_min[2]) & (points[:, 2] <= bbox_max[2])
        )
        logger.warning(f"Using bounding box containment fallback - {np.sum(inside_mask)} points")
        return inside_mask
    except Exception as e:
        logger.error(f"All containment strategies failed: {e}")
        return np.zeros(n_points, dtype=bool)

VOXELIZE_BATCH_SUBFACES = 2_000_000

def voxelize_surface(
    mesh: trimesh.Trimesh,
    pitch: float,
    max_iter: int = 10,
    edge_factor: float = 2.0,
    batch_subfaces: int = VOXELIZE_BATCH_SUBFACES,
    workers: int = 1,
):
    from trimesh import remesh
    from trimesh.voxel import base as voxel_base
    from trimesh.voxel import encoding as voxel_encoding
    from trimesh import transformations as voxel_transform

    pitch = float(pitch)
    max_edge = pitch / edge_factor
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    corners = np.round(vertices / pitch).astype(np.int64)
    origin_index = corners.min(axis=0)
    shape = tuple(int(n) for n in corners.max(axis=0) - origin_index + 1)
    occupied = np.zeros(shape, dtype=bool)
    local = corners - origin_index
    occupied[local[:, 0], local[:, 1], local[:, 2]] = True
    del corners, local

    if len(faces):
        tri = vertices[faces]
        longest = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
        del tri
        long_faces = np.flatnonzero(longest > max_edge * (1.0 - 1e-9))
        faces = faces[long_faces]
        levels = np.ceil(np.log2(np.maximum(longest[long_faces] / max_edge, 1.0))).astype(np.int64)
        del longest, long_faces
        weight = np.power(4.0, np.minimum(np.maximum(levels, 1), max_iter))
        cumulative = np.cumsum(weight)
        bounds = []
        start = 0
        while start < len(faces):
            budget = (cumulative[start - 1] if start else 0.0) + float(batch_subfaces)
            stop = max(int(np.searchsorted(cumulative, budget, side="right")), start + 1)
            bounds.append((start, stop))
            start = stop

        def new_cells(span):
            batch = faces[span[0]:span[1]]
            used, compact = np.unique(batch, return_inverse=True)
            sub_vertices, _sub_faces = remesh.subdivide_to_size(
                vertices[used], compact.reshape(batch.shape), max_edge=max_edge, max_iter=max_iter
            )
            return np.round(sub_vertices[len(used):] / pitch).astype(np.int64) - origin_index

        def mark(cells):
            occupied[cells[:, 0], cells[:, 1], cells[:, 2]] = True

        if workers > 1 and len(bounds) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=int(workers)) as pool:
                for cells in pool.map(new_cells, bounds):
                    mark(cells)
        else:
            for span in bounds:
                mark(new_cells(span))

    occupied_index = np.argwhere(occupied).astype(np.int64)
    return voxel_base.VoxelGrid(
        voxel_encoding.SparseBinaryEncoding(occupied_index),
        transform=voxel_transform.scale_and_translate(scale=pitch, translate=origin_index * pitch),
    )

_POCKET_SAMPLES = 64

_POCKET_KEEP_FRACTION = 0.25

def _unfill_sealed_pockets(
    surface: np.ndarray,
    filled: np.ndarray,
    cell_points,
    part_meshes: Sequence[trimesh.Trimesh],
) -> Tuple[np.ndarray, int]:
    from scipy import ndimage

    meshes = [mesh for mesh in part_meshes if mesh is not None and len(mesh.faces) > 0]
    added = filled & ~surface
    if not meshes or not added.any():
        return filled, 0

    labels, count = ndimage.label(added)
    if count == 0:
        return filled, 0

    watertight = [mesh for mesh in meshes if mesh.is_watertight]
    open_bounds = [np.asarray(mesh.bounds, dtype=np.float64) for mesh in meshes if not mesh.is_watertight]

    regions = ndimage.find_objects(labels)
    sample_index: List[np.ndarray] = []
    sample_owner: List[np.ndarray] = []
    for label, region in enumerate(regions, start=1):
        if region is None:
            continue
        local = np.argwhere(labels[region] == label)
        if len(local) > _POCKET_SAMPLES:
            local = local[np.linspace(0, len(local) - 1, _POCKET_SAMPLES).astype(np.int64)]
        offset = np.array([axis.start for axis in region], dtype=np.int64)
        sample_index.append(local + offset)
        sample_owner.append(np.full(len(local), label, dtype=np.int64))
    indices = np.concatenate(sample_index)
    owners = np.concatenate(sample_owner)
    points = np.asarray(cell_points(indices), dtype=np.float64)

    inside = np.zeros(len(points), dtype=bool)
    for mesh in watertight:
        lo, hi = np.asarray(mesh.bounds, dtype=np.float64)
        candidates = np.flatnonzero(~inside & np.all((points >= lo) & (points <= hi), axis=1))
        if len(candidates):
            inside[candidates] = np.asarray(mesh.contains(points[candidates]), dtype=bool)
    for lo, hi in open_bounds:
        inside |= np.all((points >= lo) & (points <= hi), axis=1)

    totals = np.bincount(owners, minlength=count + 1)
    hits = np.bincount(owners, weights=inside, minlength=count + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(totals > 0, hits / np.maximum(totals, 1), 1.0)
    pockets = np.flatnonzero((totals > 0) & (share < _POCKET_KEEP_FRACTION))
    if not len(pockets):
        return filled, 0
    remove = np.isin(labels, pockets)
    removed = int(np.count_nonzero(remove))
    return filled & ~remove, removed

def _part_assignment_contains_worker(part_key: str, mesh_path: str,
                                     chunk_idx: int, chunk_indices: np.ndarray,
                                     chunk_points: np.ndarray, tolerance: float) -> Tuple[int, np.ndarray]:
    mesh = _load_part_assignment_mesh(part_key, mesh_path)
    inside_mask = _robust_contains_check_impl(mesh, chunk_points, tolerance)
    return chunk_idx, chunk_indices[inside_mask]

def _direction_slot(sx: int, sy: int, sz: int) -> int:
    if sx == 0 and sy == 0 and sz == 0:
        return -1
    if sx != 0 and sy == 0 and sz == 0:
        return 0 if sx > 0 else 1
    if sx == 0 and sy != 0 and sz == 0:
        return 2 if sy > 0 else 3
    if sx == 0 and sy == 0 and sz != 0:
        return 4 if sz > 0 else 5
    if sx != 0 and sy != 0 and sz == 0:
        return 6 + ((0 if sx > 0 else 2) | (0 if sy > 0 else 1))
    if sx != 0 and sy == 0 and sz != 0:
        return 10 + ((0 if sx > 0 else 2) | (0 if sz > 0 else 1))
    if sx == 0 and sy != 0 and sz != 0:
        return 14 + ((0 if sy > 0 else 2) | (0 if sz > 0 else 1))
    return 18 + ((0 if sx > 0 else 4) | (0 if sy > 0 else 2) | (0 if sz > 0 else 1))

_DIRECTION_SLOTS = np.array(
    [[[_direction_slot(sx, sy, sz) for sz in (-1, 0, 1)] for sy in (-1, 0, 1)] for sx in (-1, 0, 1)],
    dtype=np.int64,
)

def _build_kdtree(points, **kwargs):
    kwargs.setdefault("balanced_tree", False)
    return cKDTree(points, **kwargs)

class _MidpointKDTree(cKDTree):

    def __init__(self, data, *args, **kwargs):
        kwargs.setdefault("balanced_tree", False)
        super().__init__(data, *args, **kwargs)

def _use_midpoint_kdtree_in_trimesh_proximity() -> bool:
    try:
        from trimesh import proximity as trimesh_proximity
    except Exception as exc:
        logger.debug(f"trimesh.proximity unavailable, kd-tree not patched: {exc}")
        return False

    existing = getattr(trimesh_proximity, "cKDTree", None)
    if existing is _MidpointKDTree:
        return True
    if not isinstance(existing, type) or not issubclass(existing, cKDTree):
        return False

    trimesh_proximity.cKDTree = _MidpointKDTree
    return True

_use_midpoint_kdtree_in_trimesh_proximity()

class AdaptiveVoxelMesh:

    AUTO_PART_ASSIGNMENT_MIN_CANDIDATES = 10_000
    AUTO_PART_ASSIGNMENT_MEMORY_PER_WORKER_BYTES = int(1.5 * 1024 ** 3)
    PART_ASSIGNMENT_SUBMIT_MULTIPLIER = 2
    PART_ASSIGNMENT_BATCH_FACE_REFERENCE = 50_000
    PART_ASSIGNMENT_MIN_BATCH = 256
    PART_ASSIGNMENT_MEMORY_RESERVE_FRACTION = 0.20
    PART_ASSIGNMENT_MEMORY_RESERVE_MIN_BYTES = 2 * 1024 ** 3
    PART_ASSIGNMENT_PROBE_SAFETY = 1.5
    PART_ASSIGNMENT_MIN_PARALLEL_SECONDS = 3.0

    @staticmethod
    def preview_voxel_count(
        trimesh_obj: trimesh.Trimesh,
        voxel_size: float,
        refinement_level: int = 1
    ) -> dict:
        from core.memory_limits import (
            estimate_memory_usage_mb,
            get_max_voxels_for_system,
            BYTES_PER_VOXEL
        )

        if trimesh_obj is None:
            return {
                'estimated_voxels': 0,
                'estimated_memory_mb': 0.0,
                'estimated_time_seconds': 0.0,
                'fill_ratio': 0.0,
                'bounds': None,
                'grid_shape': (0, 0, 0),
                'can_proceed': False,
                'warning': None,
                'error': 'No mesh provided',
            }

        if voxel_size is None or voxel_size <= 0:
            voxel_size = 0.1
        voxel_size = max(0.01, min(voxel_size, 2.0))

        refinement_level = max(1, min(4, int(refinement_level) if refinement_level else 1))

        try:
            bounds = trimesh_obj.bounds
            if bounds is None or len(bounds) != 2:
                raise InvalidMeshError("Mesh bounds are missing or malformed")

            x_range = max(0.001, bounds[1][0] - bounds[0][0])
            y_range = max(0.001, bounds[1][1] - bounds[0][1])
            z_range = max(0.001, bounds[1][2] - bounds[0][2])
        except Exception as e:
            return {
                'estimated_voxels': 0,
                'estimated_memory_mb': 0.0,
                'estimated_time_seconds': 0.0,
                'fill_ratio': 0.0,
                'bounds': None,
                'grid_shape': (0, 0, 0),
                'can_proceed': False,
                'warning': None,
                'error': f'Failed to get mesh bounds: {e}',
            }

        bounds_tuple = (
            (float(bounds[0][0]), float(bounds[1][0])),
            (float(bounds[0][1]), float(bounds[1][1])),
            (float(bounds[0][2]), float(bounds[1][2])),
        )

        bbox_volume = x_range * y_range * z_range
        mesh_volume = float(trimesh_obj.volume) if trimesh_obj.volume > 0 else bbox_volume * 0.4
        fill_ratio = max(0.1, min(mesh_volume / bbox_volume, 0.95)) if bbox_volume > 0 else 0.4

        padding = voxel_size * 0.5
        nx = int(np.ceil((x_range + 2 * padding) / voxel_size))
        ny = int(np.ceil((y_range + 2 * padding) / voxel_size))
        nz = int(np.ceil((z_range + 2 * padding) / voxel_size))

        total_grid = nx * ny * nz

        estimated_voxels = int(total_grid * fill_ratio)

        refinement_factor = 1.0 + (refinement_level - 1) * 0.3
        estimated_voxels = int(estimated_voxels * refinement_factor)

        try:
            if total_grid < 500_000:
                quick_voxels = voxelize_surface(trimesh_obj, voxel_size)
                if quick_voxels is not None:
                    actual_count = len(quick_voxels.points) if hasattr(quick_voxels, 'points') else 0
                    if actual_count > 0:
                        estimated_voxels = int(actual_count * refinement_factor)
        except Exception:
            pass

        estimated_memory_mb = estimate_memory_usage_mb(estimated_voxels)

        estimated_time = estimated_voxels * 0.00009

        max_voxels = get_max_voxels_for_system()
        can_proceed = estimated_voxels <= max_voxels

        warning = None
        if estimated_voxels > max_voxels:
            warning = f"Exceeds memory limit ({estimated_voxels:,} > {max_voxels:,}). Increase voxel size."
        elif estimated_voxels > max_voxels * 0.8:
            warning = f"Near memory limit. Consider larger voxel size."
        elif estimated_time > 60:
            warning = f"Long simulation expected (~{estimated_time/60:.1f} min)"

        return {
            'estimated_voxels': estimated_voxels,
            'estimated_memory_mb': estimated_memory_mb,
            'estimated_time_seconds': estimated_time,
            'fill_ratio': fill_ratio,
            'bounds': bounds_tuple,
            'grid_shape': (nx, ny, nz),
            'can_proceed': can_proceed,
            'warning': warning,
            'error': None,
        }

    def __init__(self, base_voxel_size: float = 0.1, max_refinement_level: int = 3,
                 memory_mode: str = "auto", use_octree: bool = False,
                 use_chunked_voxelization: bool = False,
                 part_assignment_mode: str = "auto",
                 part_assignment_workers: int = 0,
                 part_assignment_batch_size: int = 2500,
                 voxel_cut_cells: Optional[bool] = None):
        self.base_voxel_size = base_voxel_size
        self.original_voxel_size = base_voxel_size
        self.max_refinement_level = max_refinement_level
        self.memory_mode = memory_mode
        self.use_octree = use_octree and OCTREE_AVAILABLE
        self.use_chunked_voxelization = use_chunked_voxelization
        mode = str(part_assignment_mode).lower()
        self.part_assignment_mode = mode if mode in {"auto", "parallel", "legacy"} else "auto"
        self.part_assignment_workers = max(0, int(part_assignment_workers))
        self.part_assignment_batch_size = max(256, int(part_assignment_batch_size))
        self._part_assignment_pool: Optional[ProcessPoolExecutor] = None
        self._part_assignment_scratch_dir: Optional[str] = None
        self._part_assignment_pool_size: int = 0
        self.voxel_size_was_adjusted = False
        self.adjustment_message = ""

        self.voxels = None
        self.voxel_centers = None
        self.voxel_sizes = None
        self.voxel_volumes = None
        self.voxel_surface_areas = None
        self.voxel_moduli = None
        self.voxel_neighbors = None
        self.voxel_states = None
        self.surface_voxels = None
        self._material_surface_indices = None
        self._chill_surface_indices = None
        self.metal_part_bounds = None
        self._voxel_cut_cells_requested = voxel_cut_cells
        self.cut_cells = False
        self.voxel_metal_fractions = None
        self._cut_cell_part_counts = None
        self._cut_cell_part_meshes = None
        self.cut_cell_stats = None
        self.mold_layout = None
        self.voxel_riser_bare = None
        self.voxel_riser_adiabatic = None

        self._spatial_index = None

        self.voxel_part_types = None

        self.bounds = None
        self.grid_shape = None
        self.memory_manager = get_memory_manager(memory_mode)
        try:
            _mem_abort = float(os.environ.get("CASTSIM_MEM_ABORT_PCT", "90.0"))
        except (TypeError, ValueError):
            _mem_abort = 90.0
        self.memory_monitor = MemoryMonitor(abort_threshold_percent=_mem_abort)

    def generate(self, trimesh_obj: trimesh.Trimesh,
                bounds: np.ndarray,
                show_progress: bool = True,
                progress_callback=None,
                auto_adjust: bool = True,
                part_meshes: Optional[Sequence[trimesh.Trimesh]] = None) -> bool:
        if show_progress:
            print("\n" + "="*60)
            print("Generating Adaptive Voxel Mesh (Multi-core)")
            print("="*60)

        self.bounds = bounds

        bounds_tuple = (
            (float(bounds[0][0]), float(bounds[0][1])),
            (float(bounds[1][0]), float(bounds[1][1])),
            (float(bounds[2][0]), float(bounds[2][1])),
        )

        if show_progress:
            print("\nStep 0: Validating memory requirements...")

        bbox_volume = np.prod([b[1] - b[0] for b in bounds_tuple])
        mesh_volume = float(trimesh_obj.volume) if trimesh_obj.volume > 0 else bbox_volume * 0.5
        fill_ratio = max(0.1, min(mesh_volume / bbox_volume, 0.9)) if bbox_volume > 0 else 0.5

        if auto_adjust:
            adjusted_size, was_adjusted, adjustment_msg = auto_adjust_voxel_size(
                bounds_tuple, self.base_voxel_size, fill_ratio
            )
            if was_adjusted:
                self.voxel_size_was_adjusted = True
                self.original_voxel_size = self.base_voxel_size
                self.base_voxel_size = adjusted_size
                self.adjustment_message = adjustment_msg
                if show_progress:
                    print(f"  ⚠️  {adjustment_msg}")

        validation = validate_mesh_parameters(bounds_tuple, self.base_voxel_size, fill_ratio)

        if show_progress:
            print(f"  Estimated voxels: ~{validation.estimated_voxels:,}")
            print(f"  Max allowed: {validation.max_allowed_voxels:,}")
            print(f"  Estimated memory: ~{validation.estimated_memory_mb:.0f} MB")
            print(f"  Available memory: ~{validation.available_memory_mb:.0f} MB")
            print(f"  Status: {validation.severity.value.upper()}")

        if not validation.can_proceed:
            logger.error(f"Voxel mesh generation blocked: {validation.message}")
            print(f"\n❌ CANNOT PROCEED: {validation.message}")
            if validation.suggestion:
                print(f"   Suggestion: {validation.suggestion}")
            return False

        if validation.severity == MemorySeverity.WARNING:
            print(f"  ⚠️  {validation.message}")

        self.memory_monitor.start()

        if show_progress:
            print("\nStep 1: Creating initial uniform grid...")

        initial_voxels = self._create_uniform_grid(trimesh_obj, progress_callback, part_meshes=part_meshes)

        actual_voxel_count = len(initial_voxels)
        if actual_voxel_count > validation.max_allowed_voxels:
            actual_memory_mb = estimate_memory_usage_mb(actual_voxel_count)
            logger.error(
                "Voxel mesh generation blocked after voxelization: "
                f"actual voxel count {actual_voxel_count:,} exceeds safe limit "
                f"{validation.max_allowed_voxels:,}"
            )
            print(
                f"\n❌ CANNOT PROCEED: Voxelization produced {actual_voxel_count:,} voxels, "
                f"exceeding the safe limit of {validation.max_allowed_voxels:,}."
            )
            print(f"   Estimated memory for actual voxel count: ~{actual_memory_mb:.0f} MB")
            print("   Suggestion: Increase voxel size and rerun the simulation.")
            return False

        can_continue, mem_status = self.memory_monitor.check()
        if not can_continue:
            logger.error(f"Memory limit exceeded during voxel creation: {mem_status}")
            logger.log_memory_warning(self.memory_monitor.get_used_mb(),
                                      self.memory_monitor.get_limit_mb(),
                                      len(initial_voxels))
            print(f"\n❌ {mem_status}")
            return False

        if show_progress:
            print(f"  Initial voxels: {len(initial_voxels):,}")
            print(f"  Memory status: {mem_status}")

        if show_progress:
            print("Step 2: Skipping refinement (using uniform mesh for memory efficiency)...")

        refined_voxels = initial_voxels

        cut_geometry = None
        requested_cut_cells = getattr(self, "_voxel_cut_cells_requested", None)
        use_cut_cells = _cut_cells_enabled() if requested_cut_cells is None else bool(requested_cut_cells)
        if use_cut_cells and len(refined_voxels) > 0:
            if progress_callback:
                progress_callback(0, 100, 'cut_cells')
            cut_geometry = self._compute_cut_cells(refined_voxels, trimesh_obj, part_meshes)
            refined_voxels = cut_geometry.centers
            if progress_callback:
                progress_callback(100, 100, 'cut_cells')

        if show_progress:
            print("Step 3: Computing voxel properties (parallel)...")

        self._calculate_voxel_properties(refined_voxels, trimesh_obj)
        if cut_geometry is not None:
            self._apply_cut_cell_geometry(cut_geometry)

        can_continue, mem_status = self.memory_monitor.check()
        if not can_continue:
            logger.error(f"Memory limit exceeded during property calculation: {mem_status}")
            print(f"\n❌ {mem_status}")
            return False

        if show_progress:
            print("Step 4: Building neighbor connectivity...")
            print(f"  Memory status: {mem_status}")

        self._build_neighbor_graph()

        can_continue, mem_status = self.memory_monitor.check()
        if not can_continue:
            logger.error(f"Memory limit exceeded during neighbor graph build: {mem_status}")
            print(f"\n❌ {mem_status}")
            return False

        if show_progress:
            print("Step 5: Identifying surface voxels...")

        self._identify_surface_voxels()

        if show_progress:
            mem_stats = self.memory_monitor.get_stats()
            print(f"\n✓ Voxel mesh generated:")
            print(f"  Total voxels: {len(self.voxel_centers):,}")
            print(f"  Surface voxels: {np.sum(self.surface_voxels):,}")
            print(f"  Memory usage: {self._estimate_memory_mb():.1f} MB")
            print(f"  Peak RAM: {mem_stats['peak_used_mb']:.0f} MB ({mem_stats['used_percent']:.0f}%)")
            if self.voxel_size_was_adjusted:
                logger.warning(f"Voxel size auto-adjusted from {self.original_voxel_size:.3f}\" to {self.base_voxel_size:.3f}\"")
                print(f"  ⚠️  Voxel size was auto-adjusted from {self.original_voxel_size:.3f}\" to {self.base_voxel_size:.3f}\"")
            print("="*60 + "\n")

        logger.info(f"Voxel mesh generated: {len(self.voxel_centers):,} voxels, {np.sum(self.surface_voxels):,} surface, {self._estimate_memory_mb():.1f}MB")
        logger.log_performance("Voxel mesh generation", 0, len(self.voxel_centers))
        return True

    def _compute_cut_cells(
        self,
        voxels: Union[List[Dict], np.ndarray],
        trimesh_obj: trimesh.Trimesh,
        part_meshes: Optional[Sequence[trimesh.Trimesh]],
    ):
        from core.cut_cells import compute_cut_cell_geometry

        if isinstance(voxels, np.ndarray):
            centers = np.asarray(voxels, dtype=np.float64)
        else:
            centers = np.array([v['center'] for v in voxels], dtype=np.float64)
        meshes = list(part_meshes) if part_meshes else [trimesh_obj]
        geometry = compute_cut_cell_geometry(centers, float(self.base_voxel_size), meshes)
        self._cut_cell_part_meshes = meshes
        print(
            f"  Cut cells: {geometry.n_input:,} voxelized, {geometry.n_duplicates:,} duplicates, {geometry.n_added:,} added, "
            f"{geometry.n_dropped:,} dropped, {int(np.count_nonzero(geometry.sampled)):,} sampled "
            f"({geometry.timings.get('total_s', 0.0):.1f} s)"
        )
        return geometry

    def _apply_cut_cell_geometry(self, geometry) -> None:
        n_voxels = len(self.voxel_centers)
        self.voxel_metal_fractions = create_managed_array(
            n_voxels, dtype=np.float32, name="voxel_metal_fractions", manager=self.memory_manager
        )
        self.voxel_metal_fractions[:] = geometry.metal_fractions
        sizes = np.asarray(self.voxel_sizes, dtype=np.float64)
        self.voxel_volumes[:] = (geometry.metal_fractions.astype(np.float64) * sizes ** 3).astype(np.float32)
        self.voxel_surface_areas[:] = geometry.surface_areas
        self._cut_cell_part_counts = geometry.part_counts
        self.cut_cells = True
        self.cut_cell_stats = {
            "input_voxels": geometry.n_input,
            "duplicate_voxels": geometry.n_duplicates,
            "added_voxels": geometry.n_added,
            "dropped_voxels": geometry.n_dropped,
            "sampled_voxels": int(np.count_nonzero(geometry.sampled)),
            "metal_volume_in3": float(np.sum(self.voxel_volumes, dtype=np.float64)),
            "surface_area_in2": float(np.sum(geometry.surface_areas, dtype=np.float64)),
            **{key: float(value) for key, value in geometry.timings.items()},
        }

    def _apply_cut_cell_part_types(self, parts_data: list) -> None:
        from core.cut_cells import dominant_part_codes

        counts = self._cut_cell_part_counts
        meshes = self._cut_cell_part_meshes
        if not self.cut_cells or counts is None or not meshes or self.voxel_part_types is None:
            return
        code_by_mesh = {id(mesh): int(code) for mesh, code in parts_data if mesh is not None}
        codes = [code_by_mesh.get(id(mesh)) for mesh in meshes]
        if any(code is None for code in codes):
            logger.warning("Cut-cell part types skipped: voxelized meshes differ from the part list")
            return
        dominant = dominant_part_codes(counts, codes)
        sampled = dominant >= 0
        self.voxel_part_types[sampled] = dominant[sampled].astype(np.int8)

    def _create_uniform_grid(self, trimesh_obj: trimesh.Trimesh, progress_callback=None,
                             part_meshes: Optional[Sequence[trimesh.Trimesh]] = None) -> Union[List[Dict], np.ndarray]:
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        z_min, z_max = self.bounds[2]

        padding = self.base_voxel_size * 0.5
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        z_min -= padding
        z_max += padding

        nx = int(np.ceil((x_max - x_min) / self.base_voxel_size))
        ny = int(np.ceil((y_max - y_min) / self.base_voxel_size))
        nz = int(np.ceil((z_max - z_min) / self.base_voxel_size))

        self.grid_shape = (nx, ny, nz)
        total_candidates = nx * ny * nz

        print(f"  Grid: {nx} x {ny} x {nz} = {total_candidates:,} candidate voxels")

        estimated_mb = (total_candidates * 32) / (1024 * 1024)
        if estimated_mb > 500:
            print(f"  WARNING: Large mesh will use ~{estimated_mb:.0f}MB. Consider larger voxel size.")

        use_chunked = self.use_chunked_voxelization and (
            total_candidates > 100_000 or estimated_mb > 500
        )

        if use_chunked:
            print(f"  Using CHUNKED parallel voxelization for large model...")
            return self._create_uniform_grid_chunked(
                trimesh_obj, x_min, x_max, y_min, y_max, z_min, z_max,
                nx, ny, nz, progress_callback
            )

        print(f"  Using optimized trimesh voxelization...")

        try:
            voxel_grid = voxelize_surface(trimesh_obj, self.base_voxel_size)
            surface = np.array(voxel_grid.matrix, dtype=bool) if part_meshes else None

            voxel_grid = voxel_grid.fill()

            if surface is not None:
                occupancy, removed = _unfill_sealed_pockets(
                    surface,
                    np.asarray(voxel_grid.matrix, dtype=bool),
                    voxel_grid.indices_to_points,
                    part_meshes,
                )
                if removed:
                    print(f"  Left {removed:,} voxels empty: pockets sealed between parts but inside none")
                    logger.info(f"Voxelization: {removed:,} filled voxels were inside no part and were removed")
                voxel_points = np.asarray(
                    voxel_grid.indices_to_points(np.argwhere(occupancy)), dtype=np.float32
                )
            else:
                voxel_points = np.asarray(voxel_grid.points, dtype=np.float32)

            if progress_callback:
                progress_callback(50, 100, 'voxelization')

            print(f"  Trimesh voxelization found {len(voxel_points):,} voxels")

            if progress_callback:
                total_points = max(len(voxel_points), 1)
                for processed in range(10000, len(voxel_points) + 10000, 10000):
                    current = min(processed, len(voxel_points))
                    progress = 50 + int(50 * current / total_points)
                    progress_callback(progress, 100, 'voxelization')

            if progress_callback:
                progress_callback(100, 100, 'voxelization')

            print(f"  Total voxels found: {len(voxel_points):,}")
            print(f"  Converting to memory-managed arrays...")

            return voxel_points

        except Exception as e:
            logger.warning(f"Trimesh voxelization failed, using chunked fallback: {e}")
            print(f"  Trimesh voxelization failed ({e}), using chunked fallback...")
            return self._create_uniform_grid_chunked(
                trimesh_obj, x_min, x_max, y_min, y_max, z_min, z_max,
                nx, ny, nz, progress_callback
            )

    def _create_uniform_grid_chunked(self, trimesh_obj: trimesh.Trimesh,
                                     x_min: float, x_max: float,
                                     y_min: float, y_max: float,
                                     z_min: float, z_max: float,
                                     nx: int, ny: int, nz: int,
                                     progress_callback=None) -> List[Dict]:
        if not RTREE_AVAILABLE:
            raise RuntimeError(
                "rtree dependency is required for chunked voxelization via "
                "trimesh.contains(). Install with: pip install rtree"
            )

        n_workers = hardware.get_optimal_thread_count('compute')

        available_mb = self.memory_manager.get_available_memory_gb() * 1024
        voxels_per_slice = nx * ny
        bytes_per_voxel = 32
        mb_per_slice = (voxels_per_slice * bytes_per_voxel) / (1024 * 1024)

        target_mb = available_mb * 0.3
        slices_per_batch = max(1, int(target_mb / mb_per_slice))
        slices_per_batch = min(slices_per_batch, nz)

        print(f"  Chunked voxelization parameters:")
        print(f"    Z-slices: {nz}")
        print(f"    Slices per batch: {slices_per_batch}")
        print(f"    Workers: {n_workers}")
        print(f"    Est. memory per batch: {slices_per_batch * mb_per_slice:.1f} MB")

        x_coords = np.linspace(x_min + self.base_voxel_size/2,
                               x_max - self.base_voxel_size/2, nx)
        y_coords = np.linspace(y_min + self.base_voxel_size/2,
                               y_max - self.base_voxel_size/2, ny)
        z_coords = np.linspace(z_min + self.base_voxel_size/2,
                               z_max - self.base_voxel_size/2, nz)

        all_voxels = []
        processed_slices = 0

        for batch_start in range(0, nz, slices_per_batch):
            batch_end = min(batch_start + slices_per_batch, nz)
            batch_z_coords = z_coords[batch_start:batch_end]

            batch_voxels, batch_failed = self._process_z_slice_batch_parallel(
                trimesh_obj, x_coords, y_coords, batch_z_coords,
                n_workers
            )

            if batch_failed == len(batch_z_coords):
                raise MeshGenerationError(
                    f"All {batch_failed} slices in batch [{batch_start+1}-{batch_end}] "
                    "failed during containment checks. This may indicate missing rtree "
                    "or a mesh geometry issue. Install rtree: pip install rtree",
                    voxel_size=float(self.base_voxel_size),
                )

            if len(all_voxels) == 0 and processed_slices == 0 and batch_failed > 0:
                if not RTREE_AVAILABLE:
                    raise RuntimeError(
                        f"Containment check failed for all {batch_failed} slices. "
                        f"trimesh.contains() requires rtree. Install: pip install rtree"
                    )

            all_voxels.extend(batch_voxels)
            processed_slices += len(batch_z_coords)

            progress_pct = int(processed_slices / nz * 100)
            print(f"    Processed Z-slices {batch_start+1}-{batch_end}/{nz} "
                  f"({progress_pct}%), found {len(all_voxels):,} voxels so far")

            if progress_callback:
                progress_callback(progress_pct, 100, 'voxelization')

            import gc
            gc.collect()

        print(f"  Total voxels found: {len(all_voxels):,}")
        return all_voxels

    def _process_z_slice_batch_parallel(self, trimesh_obj: trimesh.Trimesh,
                                        x_coords: np.ndarray,
                                        y_coords: np.ndarray,
                                        z_coords: np.ndarray,
                                        n_workers: int) -> Tuple[List[Dict], int]:
        if len(z_coords) <= 2:
            voxels = []
            failed = 0
            for z_val in z_coords:
                try:
                    slice_voxels = self._process_single_z_slice(
                        trimesh_obj, x_coords, y_coords, z_val
                    )
                    voxels.extend(slice_voxels)
                except Exception:
                    failed += 1
            return voxels, failed

        z_chunks = np.array_split(z_coords, min(n_workers, len(z_coords)))

        all_voxels = []
        failed_slices = 0

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for chunk in z_chunks:
                if len(chunk) > 0:
                    future = executor.submit(
                        self._process_z_slice_chunk,
                        trimesh_obj, x_coords, y_coords, chunk
                    )
                    futures[future] = len(chunk)

            for future in as_completed(futures):
                n_slices_in_chunk = futures[future]
                try:
                    chunk_voxels, chunk_failed = future.result()
                    all_voxels.extend(chunk_voxels)
                    failed_slices += chunk_failed
                except Exception as e:
                    failed_slices += n_slices_in_chunk
                    logger.warning(f"Chunk processing error: {e}")
                    print(f"    Warning: Chunk processing error: {e}")

        return all_voxels, failed_slices

    def _process_z_slice_chunk(self, trimesh_obj: trimesh.Trimesh,
                                 x_coords: np.ndarray,
                                 y_coords: np.ndarray,
                                 z_coords: np.ndarray) -> Tuple[List[Dict], int]:
        voxels = []
        failed = 0
        for z_val in z_coords:
            try:
                slice_voxels = self._process_single_z_slice(
                    trimesh_obj, x_coords, y_coords, z_val
                )
                voxels.extend(slice_voxels)
            except Exception:
                failed += 1
        return voxels, failed

    def _process_single_z_slice(self, trimesh_obj: trimesh.Trimesh,
                                x_coords: np.ndarray,
                                y_coords: np.ndarray,
                                z_val: float) -> List[Dict]:
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
        zz = np.full_like(xx, z_val)
        slice_centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        if GPU_AVAILABLE:
            inside = gpu_compute.compute_voxel_containment_gpu(
                slice_centers.astype(np.float32),
                trimesh_obj.vertices.astype(np.float32),
                trimesh_obj.faces
            )
            if inside is None:
                if not RTREE_AVAILABLE:
                    raise RuntimeError(
                        "rtree is required for trimesh.contains() in chunked voxelization. "
                        "Install with: pip install rtree"
                    )
                inside = trimesh_obj.contains(slice_centers)
        else:
            if not RTREE_AVAILABLE:
                raise RuntimeError(
                    "rtree is required for trimesh.contains() in chunked voxelization. "
                    "Install with: pip install rtree"
                )
            inside = trimesh_obj.contains(slice_centers)

        voxels = []
        for center, is_inside in zip(slice_centers, inside):
            if is_inside:
                voxels.append({
                    'center': center.astype(np.float32),
                    'size': self.base_voxel_size,
                    'level': 0
                })

        return voxels

    def _create_uniform_grid_fallback(self, trimesh_obj: trimesh.Trimesh, progress_callback=None) -> List[Dict]:
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        z_min, z_max = self.bounds[2]

        padding = self.base_voxel_size * 0.5
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        z_min -= padding
        z_max += padding

        nx, ny, nz = self.grid_shape

        voxels = []
        x_coords = np.linspace(x_min, x_max, nx)
        y_coords = np.linspace(y_min, y_max, ny)
        z_coords = np.linspace(z_min, z_max, nz)

        print(f"  Checking occupancy in chunks (fallback mode)...")

        for iz in range(nz):
            z_val = z_coords[iz]

            xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
            zz = np.full_like(xx, z_val)
            slice_centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

            if not RTREE_AVAILABLE:
                raise RuntimeError(
                    "rtree is required for trimesh.contains() in fallback voxelization. "
                    "Install with: pip install rtree"
                )
            inside = trimesh_obj.contains(slice_centers)

            for center, is_inside in zip(slice_centers, inside):
                if is_inside:
                    voxels.append({
                        'center': center.astype(np.float32),
                        'size': self.base_voxel_size,
                        'level': 0
                    })

            del xx, yy, zz, slice_centers, inside

            if (iz + 1) % 10 == 0 or iz == nz - 1:
                print(f"    Z-slice {iz+1}/{nz}, found {len(voxels):,} voxels so far")
                if progress_callback:
                    progress_callback(iz + 1, nz, 'voxelization')

        print(f"  Total voxels found: {len(voxels):,}")
        print(f"  Converting to memory-managed arrays...")

        return voxels

    def _adaptive_refinement(self, voxels: List[Dict],
                           trimesh_obj: trimesh.Trimesh) -> List[Dict]:
        refined = []

        vertex_tree = _build_kdtree(trimesh_obj.vertices)

        for voxel in voxels:
            center = voxel['center']
            size = voxel['size']
            level = voxel['level']

            dist, _ = vertex_tree.query(center)

            if dist < size * 2 and level < self.max_refinement_level:
                sub_size = size / 2
                offsets = [
                    [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
                ]

                for offset in offsets:
                    sub_center = center + np.array(offset) * sub_size / 2

                    if trimesh_obj.contains([sub_center])[0]:
                        refined.append({
                            'center': sub_center,
                            'size': sub_size,
                            'level': level + 1,
                            'index': len(refined)
                        })
            else:
                refined.append(voxel)

        return refined

    def _calculate_voxel_properties(self, voxels: Union[List[Dict], np.ndarray],
                                   trimesh_obj: trimesh.Trimesh):
        n_voxels = len(voxels)

        print(f"\n  Allocating memory for {n_voxels:,} voxels...")
        print(f"  Memory mode: {self.memory_mode.upper()}")

        self.voxel_centers = create_managed_array(
            (n_voxels, 3), dtype=np.float32, name="voxel_centers", manager=self.memory_manager
        )
        self.voxel_sizes = create_managed_array(
            n_voxels, dtype=np.float32, name="voxel_sizes", manager=self.memory_manager
        )

        print(f"  Transferring voxel data to managed arrays...")
        chunk_size = 10000
        voxel_size_value = np.float32(self.base_voxel_size)
        voxels_is_array = isinstance(voxels, np.ndarray)
        for start_idx in range(0, n_voxels, chunk_size):
            end_idx = min(start_idx + chunk_size, n_voxels)
            if voxels_is_array:
                self.voxel_centers[start_idx:end_idx] = np.asarray(voxels[start_idx:end_idx], dtype=np.float32)
                self.voxel_sizes[start_idx:end_idx] = voxel_size_value
            else:
                for i in range(start_idx, end_idx):
                    self.voxel_centers[i] = voxels[i]['center']
                    self.voxel_sizes[i] = voxels[i]['size']

            if (end_idx - start_idx) == chunk_size or end_idx == n_voxels:
                print(f"    Transferred {end_idx:,}/{n_voxels:,} voxels...")

        print(f"  Freeing temporary voxel list...")
        del voxels
        import gc
        gc.collect()

        self.voxel_volumes = create_managed_array(
            n_voxels, dtype=np.float32, name="voxel_volumes", manager=self.memory_manager
        )
        np.multiply(self.voxel_sizes, self.voxel_sizes, out=self.voxel_volumes)
        np.multiply(self.voxel_volumes, self.voxel_sizes, out=self.voxel_volumes)

        self.voxel_surface_areas = create_managed_array(
            n_voxels, dtype=np.float32, name="voxel_surface_areas", manager=self.memory_manager
        )

        voxel_tree = _build_kdtree(self.voxel_centers)

        print(f"  Computing surface areas for {n_voxels:,} voxels...")

        voxel_centers_array = np.asarray(self.voxel_centers)
        voxel_sizes_array = np.asarray(self.voxel_sizes)

        directions_base = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=np.float32)

        surface_indices = []

        BATCH_SIZE = 10000
        n_batches = (n_voxels + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, n_voxels)
            batch_size = end - start

            batch_centers = voxel_centers_array[start:end]
            batch_sizes = voxel_sizes_array[start:end]

            exposed_counts = np.zeros(batch_size, dtype=np.int32)

            direction_pairs = [
                (np.array([1, 0, 0], dtype=np.float32),
                 np.array([-1, 0, 0], dtype=np.float32)),
                (np.array([0, 1, 0], dtype=np.float32),
                 np.array([0, -1, 0], dtype=np.float32)),
                (np.array([0, 0, 1], dtype=np.float32),
                 np.array([0, 0, -1], dtype=np.float32)),
            ]

            for dir_pos, dir_neg in direction_pairs:
                pos_positions = batch_centers + dir_pos * batch_sizes[:, np.newaxis]
                neg_positions = batch_centers + dir_neg * batch_sizes[:, np.newaxis]
                stacked = np.concatenate([pos_positions, neg_positions], axis=0)
                distances, _ = voxel_tree.query(stacked, k=1)
                d_pos = distances[:batch_size]
                d_neg = distances[batch_size:]
                exposed_counts += (d_pos > batch_sizes * 0.5).astype(np.int32)
                exposed_counts += (d_neg > batch_sizes * 0.5).astype(np.int32)

            self.voxel_surface_areas[start:end] = exposed_counts * (batch_sizes ** 2)

            surface_mask = exposed_counts > 0
            batch_surface_indices = np.where(surface_mask)[0] + start
            surface_indices.extend(batch_surface_indices.tolist())

            if (batch_idx + 1) % 5 == 0 or end == n_voxels:
                print(f"    Processed {end:,}/{n_voxels:,} voxels...", flush=True)

        print("  Computing geometric modulus (distance to surface)...")
        self.voxel_moduli = create_managed_array(
            n_voxels, dtype=np.float32, name="voxel_moduli", manager=self.memory_manager
        )

        if len(surface_indices) > 0:
            surface_centers = self.voxel_centers[surface_indices]
            surface_tree = _build_kdtree(surface_centers)

            print(f"    Querying distances for {n_voxels:,} voxels to {len(surface_indices):,} surface voxels...")
            all_centers = np.asarray(self.voxel_centers)
            distances, _ = surface_tree.query(all_centers, workers=-1)

            self.voxel_moduli[:] = distances.astype(np.float32)

            surface_set = set(surface_indices)
            for i in surface_indices:
                self.voxel_moduli[i] = 0.0
        else:
            temp_moduli = compute_modulus_parallel(
                self.voxel_volumes,
                self.voxel_surface_areas
            )
            self.voxel_moduli[:] = temp_moduli

        self.voxel_states = create_managed_array(
            n_voxels, dtype=np.float32, name="voxel_states", manager=self.memory_manager
        )

        print(f"  Modulus range: {np.min(self.voxel_moduli):.4f} - {np.max(self.voxel_moduli):.4f} inches")
        print(f"  Mean modulus: {np.mean(self.voxel_moduli):.4f} inches")

    def _build_neighbor_graph(self):
        n_voxels = len(self.voxel_centers)

        print(f"  Building neighbor graph for {n_voxels:,} voxels...")

        self.voxel_neighbors = create_managed_array(
            (n_voxels, 26), dtype=np.int32, name="voxel_neighbors", manager=self.memory_manager
        )
        self.voxel_neighbors[:] = -1

        import gc
        gc.collect()

        search_radius = self.base_voxel_size * 2.0

        if self.use_octree and n_voxels > 10000:
            print(f"  Building OCTREE spatial index for {n_voxels:,} voxels...")

            bounds_min = np.min(self.voxel_centers, axis=0)
            bounds_max = np.max(self.voxel_centers, axis=0)

            self._spatial_index = SpatialIndex(
                np.asarray(self.voxel_centers),
                use_octree=True
            )

            print(f"  Finding neighbors using octree (radius={search_radius:.3f}\")...")
            self.voxel_neighbors[:] = self._canonicalize_batch(
                self._spatial_index.get_neighbors_batch(
                    search_radius, max_neighbors=26
                )
            )

            print(f"  Octree neighbor search complete")

        else:
            print(f"  Building KD-tree for neighbor search...")
            tree = _build_kdtree(self.voxel_centers)

            voxel_centers_array = np.asarray(self.voxel_centers)
            voxel_sizes_array = np.asarray(self.voxel_sizes)

            print(f"  Finding neighbors (2.0x radius for interface connectivity)...")
            BATCH_SIZE = 50000
            n_batches = (n_voxels + BATCH_SIZE - 1) // BATCH_SIZE

            for batch_idx in range(n_batches):
                start = batch_idx * BATCH_SIZE
                end = min(start + BATCH_SIZE, n_voxels)

                radii = voxel_sizes_array[start:end] * 2.0
                candidate_lists = tree.query_ball_point(voxel_centers_array[start:end], radii)
                counts = np.fromiter(
                    (len(candidates) for candidates in candidate_lists),
                    dtype=np.int64,
                    count=end - start,
                )
                total = int(counts.sum())
                if total > 0:
                    cand = np.fromiter(
                        itertools.chain.from_iterable(candidate_lists), dtype=np.int64, count=total
                    )
                    own = np.repeat(np.arange(start, end, dtype=np.int64), counts)
                    self.voxel_neighbors[start:end] = self._canonicalize_pairs(own, cand, start, end)

                if (batch_idx + 1) % 10 == 0 or end == n_voxels:
                    print(f"    Processed {end:,}/{n_voxels:,} voxels...", flush=True)

    def _canonicalize_pairs(
        self,
        own: np.ndarray,
        cand: np.ndarray,
        start: int,
        end: int,
    ) -> np.ndarray:
        rows = np.full((end - start, 26), -1, dtype=np.int32)
        keep = cand != own
        own = own[keep]
        cand = cand[keep]
        if own.size == 0:
            return rows

        centers_arr = np.asarray(self.voxel_centers)
        diff = centers_arr[cand].astype(np.float64) - centers_arr[own].astype(np.float64)
        tolerance = max(self.base_voxel_size * 0.25, 1e-6)
        signs = np.where(diff > tolerance, 1, np.where(diff < -tolerance, -1, 0))
        slot = _DIRECTION_SLOTS[signs[:, 0] + 1, signs[:, 1] + 1, signs[:, 2] + 1]
        dist = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2]

        placed = slot >= 0
        if not np.any(placed):
            return rows
        key = (own[placed] - start) * 26 + slot[placed]
        cand = cand[placed]
        dist = dist[placed]
        order = np.lexsort((cand, dist, key))
        key = key[order]
        first = np.ones(key.size, dtype=bool)
        first[1:] = key[1:] != key[:-1]
        rows.reshape(-1)[key[first]] = cand[order][first].astype(np.int32)
        return rows

    def _canonicalize_neighbor_order(
        self,
        voxel_idx: int,
        center: np.ndarray,
        neighbor_indices: Sequence[int],
    ) -> np.ndarray:
        canonical = np.full(26, -1, dtype=np.int32)
        best_dist = np.full(26, np.inf, dtype=np.float64)
        if not neighbor_indices:
            return canonical

        centers_arr = np.asarray(self.voxel_centers)
        own_center = np.asarray(center, dtype=np.float64)
        tolerance = max(self.base_voxel_size * 0.25, 1e-6)

        for nidx in neighbor_indices:
            if nidx == voxel_idx:
                continue
            diff = centers_arr[nidx].astype(np.float64) - own_center
            sx = 1 if diff[0] > tolerance else (-1 if diff[0] < -tolerance else 0)
            sy = 1 if diff[1] > tolerance else (-1 if diff[1] < -tolerance else 0)
            sz = 1 if diff[2] > tolerance else (-1 if diff[2] < -tolerance else 0)
            if sx == 0 and sy == 0 and sz == 0:
                continue

            if sx != 0 and sy == 0 and sz == 0:
                slot = 0 if sx > 0 else 1
            elif sx == 0 and sy != 0 and sz == 0:
                slot = 2 if sy > 0 else 3
            elif sx == 0 and sy == 0 and sz != 0:
                slot = 4 if sz > 0 else 5
            elif sx != 0 and sy != 0 and sz == 0:
                slot = 6 + ((0 if sx > 0 else 2) | (0 if sy > 0 else 1))
            elif sx != 0 and sy == 0 and sz != 0:
                slot = 10 + ((0 if sx > 0 else 2) | (0 if sz > 0 else 1))
            elif sx == 0 and sy != 0 and sz != 0:
                slot = 14 + ((0 if sy > 0 else 2) | (0 if sz > 0 else 1))
            else:
                slot = 18 + ((0 if sx > 0 else 4) | (0 if sy > 0 else 2) | (0 if sz > 0 else 1))

            dist = float(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            if dist < best_dist[slot]:
                canonical[slot] = np.int32(nidx)
                best_dist[slot] = dist

        return canonical

    def _canonicalize_batch(self, raw_neighbors: np.ndarray) -> np.ndarray:
        n_voxels = raw_neighbors.shape[0]
        canonical = np.full((n_voxels, 26), -1, dtype=np.int32)
        centers_arr = np.asarray(self.voxel_centers)
        tolerance = max(self.base_voxel_size * 0.25, 1e-6)

        for i in range(n_voxels):
            row = raw_neighbors[i]
            own_center = centers_arr[i].astype(np.float64)
            best_dist = np.full(26, np.inf, dtype=np.float64)
            for nidx in row:
                if nidx < 0 or nidx == i:
                    continue
                diff = centers_arr[nidx].astype(np.float64) - own_center
                sx = 1 if diff[0] > tolerance else (-1 if diff[0] < -tolerance else 0)
                sy = 1 if diff[1] > tolerance else (-1 if diff[1] < -tolerance else 0)
                sz = 1 if diff[2] > tolerance else (-1 if diff[2] < -tolerance else 0)
                if sx == 0 and sy == 0 and sz == 0:
                    continue

                if sx != 0 and sy == 0 and sz == 0:
                    slot = 0 if sx > 0 else 1
                elif sx == 0 and sy != 0 and sz == 0:
                    slot = 2 if sy > 0 else 3
                elif sx == 0 and sy == 0 and sz != 0:
                    slot = 4 if sz > 0 else 5
                elif sx != 0 and sy != 0 and sz == 0:
                    slot = 6 + ((0 if sx > 0 else 2) | (0 if sy > 0 else 1))
                elif sx != 0 and sy == 0 and sz != 0:
                    slot = 10 + ((0 if sx > 0 else 2) | (0 if sz > 0 else 1))
                elif sx == 0 and sy != 0 and sz != 0:
                    slot = 14 + ((0 if sy > 0 else 2) | (0 if sz > 0 else 1))
                else:
                    slot = 18 + ((0 if sx > 0 else 4) | (0 if sy > 0 else 2) | (0 if sz > 0 else 1))

                dist = float(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
                if dist < best_dist[slot]:
                    canonical[i, slot] = np.int32(nidx)
                    best_dist[slot] = dist
        return canonical

    def _identify_surface_voxels(self):
        n_voxels = len(self.voxel_centers)
        self.surface_voxels = create_managed_array(
            n_voxels, dtype=np.bool_, name="surface_voxels", manager=self.memory_manager
        )

        surface_threshold = 0.5

        for i in range(n_voxels):
            if self.voxel_surface_areas[i] > 0:
                self.surface_voxels[i] = True

    def assign_part_types(self, parts_data: list):
        if self.voxel_centers is None:
            print("Warning: No voxels to assign part types to")
            return

        n_voxels = len(self.voxel_centers)
        self._material_surface_indices = None
        self._chill_surface_indices = None

        self.voxel_part_types = create_managed_array(
            n_voxels, dtype=np.int8, name="voxel_part_types", manager=self.memory_manager
        )
        self.voxel_part_types[:] = 0

        if not parts_data:
            print("  No part type data provided - all voxels set to CAST")
            return

        self.metal_part_bounds = [
            np.array(m.bounds, dtype=np.float64)
            for m, t in parts_data
            if m is not None and t in METAL_PART_CODES
        ]

        parts_by_code: Dict[int, list] = {}
        for mesh_obj, code in parts_data:
            if mesh_obj is not None:
                parts_by_code.setdefault(int(code), []).append((mesh_obj, int(code)))

        cast_parts = parts_by_code.get(PART_TYPE_CAST, [])
        all_parts = [
            part
            for code in sorted(parts_by_code)
            if code != PART_TYPE_CAST
            for part in parts_by_code[code]
        ]

        if len(cast_parts) <= 1 and not all_parts:
            print(f"  Single CAST part - all {n_voxels:,} voxels assigned to CAST (fast path)")
            return

        def _type_name(code: int) -> str:
            part_type = CODE_TO_PART_TYPE.get(code)
            return part_type.name if part_type is not None else f"UNKNOWN({code})"

        print(f"\n  Assigning part types to {n_voxels:,} voxels...")
        print(
            "    Parts: "
            + ", ".join(
                f"{len(parts_by_code[code])} {_type_name(code)}"
                for code in sorted(parts_by_code)
            )
        )

        voxel_centers_array = np.asarray(self.voxel_centers)

        total_parts = len(all_parts)

        proximity_tolerance = self.base_voxel_size * 0.5

        try:
            self._assign_part_types_loop(
                all_parts=all_parts,
                voxel_centers_array=voxel_centers_array,
                total_parts=total_parts,
                proximity_tolerance=proximity_tolerance,
                type_name_fn=_type_name,
            )
        finally:
            self._shutdown_part_assignment_pool()

        self._claim_unowned_boundary_voxels(cast_parts, all_parts)
        self._apply_cut_cell_part_types(parts_data)

        counts = {
            code: int(np.sum(self.voxel_part_types == code))
            for code in sorted(set(parts_by_code) | {PART_TYPE_CAST})
        }
        print(
            "    Part types assigned: "
            + ", ".join(f"{_type_name(code)}={count:,}" for code, count in counts.items())
        )

        for code, parts in sorted(parts_by_code.items()):
            if code == PART_TYPE_CAST or counts.get(code, 0) > 0:
                continue
            name = _type_name(code)
            print(f"    ⚠️  WARNING: {len(parts)} {name} parts loaded but 0 voxels assigned!")
            print(f"       This may indicate containment check issues. Check STL mesh quality.")
            if code == PART_TYPE_CHILL:
                print(f"       Chills must be watertight meshes for containment detection.")
            logger.warning(
                f"{name} parts loaded ({len(parts)}) but no voxels assigned - "
                f"containment check may have failed"
            )

        if any(count > 0 for code, count in counts.items() if code != PART_TYPE_CAST):
            self._bridge_part_interfaces()

        self._material_surface_indices = None
        self._chill_surface_indices = None

    _BOUNDARY_CLAIM_MAX_CANDIDATES = 250_000

    def _claim_unowned_boundary_voxels(self, cast_parts: list, other_parts: list) -> None:
        if not other_parts or self.voxel_part_types is None:
            return
        centers = np.asarray(self.voxel_centers, dtype=np.float64)
        sizes = np.asarray(self.voxel_sizes, dtype=np.float64).reshape(-1)
        if sizes.size == 1:
            sizes = np.full(len(centers), float(sizes[0]))
        reach = 0.5 * np.sqrt(3.0) * sizes
        max_reach = float(reach.max()) if reach.size else 0.0

        others = [(m, c) for m, c in other_parts if m is not None and m.is_watertight]
        if not others:
            return
        is_cast = np.asarray(self.voxel_part_types) == PART_TYPE_CAST
        near_other = np.zeros(len(centers), dtype=bool)
        for mesh, _code in others:
            lo = mesh.bounds[0] - max_reach
            hi = mesh.bounds[1] + max_reach
            near_other |= np.all((centers >= lo) & (centers <= hi), axis=1)
        candidates = np.flatnonzero(is_cast & near_other)
        if len(candidates) == 0:
            return
        if len(candidates) > self._BOUNDARY_CLAIM_MAX_CANDIDATES:
            print(f"    Boundary claim skipped: {len(candidates):,} candidates")
            return
        pts = centers[candidates]
        cand_reach = reach[candidates]

        keep = np.zeros(len(candidates), dtype=bool)
        for mesh, _code in cast_parts:
            if mesh is None:
                continue
            if not mesh.is_watertight:
                keep[:] = True
                break
            lo = mesh.bounds[0] - max_reach
            hi = mesh.bounds[1] + max_reach
            near = ~keep & np.all((pts >= lo) & (pts <= hi), axis=1)
            if np.any(near):
                sd = trimesh.proximity.signed_distance(mesh, pts[near])
                keep[np.flatnonzero(near)[sd >= -cand_reach[near]]] = True
        todo = np.flatnonzero(~keep)
        if len(todo) == 0:
            return

        best_sd = np.full(len(todo), -np.inf)
        best_code = np.full(len(todo), -1, dtype=np.int64)
        for mesh, code in others:
            lo = mesh.bounds[0] - max_reach
            hi = mesh.bounds[1] + max_reach
            near = np.all((pts[todo] >= lo) & (pts[todo] <= hi), axis=1)
            if not np.any(near):
                continue
            sd = np.full(len(todo), -np.inf)
            sd[near] = trimesh.proximity.signed_distance(mesh, pts[todo][near])
            reaches = sd >= -cand_reach[todo]
            better = reaches & ((sd > best_sd) | ((sd == best_sd) & (code > best_code)))
            best_sd[better] = sd[better]
            best_code[better] = code
        claimed = best_code >= 0
        if np.any(claimed):
            self.voxel_part_types[candidates[todo[claimed]]] = best_code[claimed].astype(np.int8)
            print(f"    Boundary cells claimed from CAST default: {int(claimed.sum()):,}")

    def _assign_part_types_loop(
        self,
        *,
        all_parts: list,
        voxel_centers_array: np.ndarray,
        total_parts: int,
        proximity_tolerance: float,
        type_name_fn,
    ) -> None:
        _type_name = type_name_fn
        for idx, (mesh, part_type) in enumerate(all_parts):
            type_name = _type_name(part_type)
            print(f"    [{idx+1}/{total_parts}] Checking {type_name} part...", end=" ", flush=True)

            bbox_min = mesh.bounds[0] - proximity_tolerance
            bbox_max = mesh.bounds[1] + proximity_tolerance

            in_bbox = (
                (voxel_centers_array[:, 0] >= bbox_min[0]) &
                (voxel_centers_array[:, 0] <= bbox_max[0]) &
                (voxel_centers_array[:, 1] >= bbox_min[1]) &
                (voxel_centers_array[:, 1] <= bbox_max[1]) &
                (voxel_centers_array[:, 2] >= bbox_min[2]) &
                (voxel_centers_array[:, 2] <= bbox_max[2])
            )

            candidate_indices = np.where(in_bbox)[0]
            n_candidates = len(candidate_indices)

            if n_candidates == 0:
                print(f"0 voxels (bbox filter: 0 candidates)")
                continue

            BATCH_SIZE = self._effective_batch_size(len(mesh.faces))
            inside_global_indices = []

            if n_candidates > BATCH_SIZE:
                print(f"({n_candidates:,} candidates, {len(mesh.faces):,} faces, batch {BATCH_SIZE})", flush=True)
                use_parallel = self._should_parallelize_part_assignment(n_candidates, batch_size=BATCH_SIZE)
                if use_parallel:
                    inside_global_indices = self._run_part_assignment_parallel(
                        mesh=mesh,
                        candidate_indices=candidate_indices,
                        voxel_centers_array=voxel_centers_array,
                        part_idx=idx,
                        total_parts=total_parts,
                        type_name=type_name,
                        tolerance=proximity_tolerance,
                        batch_size=BATCH_SIZE,
                    )
                else:
                    inside_global_indices = self._run_part_assignment_sequential(
                        mesh=mesh,
                        candidate_indices=candidate_indices,
                        voxel_centers_array=voxel_centers_array,
                        part_idx=idx,
                        total_parts=total_parts,
                        type_name=type_name,
                        tolerance=proximity_tolerance,
                        batch_size=BATCH_SIZE,
                    )

                inside_global_indices = np.array(inside_global_indices) if inside_global_indices else np.array([], dtype=int)
                print(f"\r    [{idx+1}/{total_parts}] Checking {type_name} part... {len(inside_global_indices):,} voxels ({n_candidates:,} candidates)   ")
            else:
                try:
                    candidate_centers = voxel_centers_array[candidate_indices]

                    inside_mask = self._robust_contains_check(mesh, candidate_centers, proximity_tolerance)

                    inside_global_indices = candidate_indices[inside_mask]
                    print(f"{len(inside_global_indices):,} voxels ({n_candidates:,} candidates)")
                except Exception as e:
                    logger.error(f"Part type assignment error: {e}")
                    print(f"error: {e}")
                    inside_global_indices = np.array([], dtype=int)

            if len(inside_global_indices) > 0:
                current_types = self.voxel_part_types[inside_global_indices]
                override_mask = part_type >= current_types
                if np.any(override_mask):
                    self.voxel_part_types[inside_global_indices[override_mask]] = part_type

    def _bridge_part_interfaces(self):
        if self.voxel_part_types is None or self.voxel_neighbors is None:
            return

        voxel_centers_array = np.asarray(self.voxel_centers)
        voxel_sizes_array = np.asarray(self.voxel_sizes)

        bridges_added = 0

        indices_by_code = {
            code: np.where(self.voxel_part_types == code)[0]
            for code in _INTERFACE_BRIDGE_CODES
        }
        for code_a, code_b in _INTERFACE_BRIDGE_PAIRS:
            side_a = indices_by_code.get(code_a)
            side_b = indices_by_code.get(code_b)
            if side_a is None or side_b is None or len(side_a) == 0 or len(side_b) == 0:
                continue
            label = f"{CODE_TO_PART_TYPE[code_a].name}-{CODE_TO_PART_TYPE[code_b].name}"
            bridges_added += self._bridge_two_types(
                side_a, side_b, voxel_centers_array, voxel_sizes_array, label
            )

        if bridges_added > 0:
            print(f"    Interface bridges added: {bridges_added:,} new neighbor connections")

    def _bridge_two_types(self, type1_indices: np.ndarray, type2_indices: np.ndarray,
                          voxel_centers: np.ndarray, voxel_sizes: np.ndarray,
                          interface_name: str) -> int:
        type2_centers = voxel_centers[type2_indices]
        type2_tree = _build_kdtree(type2_centers)

        bridges_added = 0
        interface_search_radius = self.base_voxel_size * 3.0

        for idx1 in type1_indices:
            center1 = voxel_centers[idx1]

            nearby_local = type2_tree.query_ball_point(center1, interface_search_radius)

            if len(nearby_local) == 0:
                continue

            nearby_global = type2_indices[nearby_local]

            current_neighbors = set(self.voxel_neighbors[idx1])
            current_neighbors.discard(-1)

            for nearby_idx in nearby_global:
                if nearby_idx not in current_neighbors:
                    for slot in range(26):
                        if self.voxel_neighbors[idx1, slot] == -1:
                            self.voxel_neighbors[idx1, slot] = nearby_idx
                            bridges_added += 1
                            break

                    for slot in range(26):
                        if self.voxel_neighbors[nearby_idx, slot] == -1:
                            self.voxel_neighbors[nearby_idx, slot] = idx1
                            break
                        elif self.voxel_neighbors[nearby_idx, slot] == idx1:
                            break

        if bridges_added > 0:
            print(f"    {interface_name} interface: {bridges_added:,} connections")

        return bridges_added

    def _robust_contains_check(self, mesh: trimesh.Trimesh, points: np.ndarray,
                               tolerance: float) -> np.ndarray:
        return _robust_contains_check_impl(mesh, points, tolerance)

    def _get_part_assignment_workers(self) -> int:
        if self.part_assignment_workers > 0:
            return max(1, self.part_assignment_workers)
        cpu_budget = hardware.get_optimal_thread_count('compute')
        mem_budget = max(
            1,
            int(_available_memory_bytes() // self.AUTO_PART_ASSIGNMENT_MEMORY_PER_WORKER_BYTES),
        )
        return max(1, min(cpu_budget, mem_budget))

    def _effective_batch_size(self, n_faces: int) -> int:
        configured = self.part_assignment_batch_size
        reference = self.PART_ASSIGNMENT_BATCH_FACE_REFERENCE
        if n_faces <= reference:
            return configured
        scaled = int(configured * reference / max(1, n_faces))
        return max(self.PART_ASSIGNMENT_MIN_BATCH, min(configured, scaled))

    def _memory_reserve_bytes(self) -> int:
        try:
            import psutil
            total = int(psutil.virtual_memory().total)
        except Exception:
            total = int(getattr(hardware, "total_memory", 0) or 0)
        return max(
            self.PART_ASSIGNMENT_MEMORY_RESERVE_MIN_BYTES,
            int(total * self.PART_ASSIGNMENT_MEMORY_RESERVE_FRACTION),
        )

    def _memory_bounded_concurrency(self, per_worker_bytes: int, ceiling: int) -> int:
        headroom = _available_memory_bytes() - self._memory_reserve_bytes()
        if headroom <= 0:
            return 0
        return max(0, min(ceiling, int(headroom // max(1, per_worker_bytes))))

    @staticmethod
    def _measure_peak_rss(fn):
        try:
            import psutil
            proc = psutil.Process()
        except Exception:
            return fn(), 0
        baseline = proc.memory_info().rss
        peak = [baseline]
        stop = threading.Event()

        def _sample() -> None:
            while not stop.is_set():
                try:
                    peak[0] = max(peak[0], proc.memory_info().rss)
                except Exception:
                    break
                stop.wait(0.005)

        sampler = threading.Thread(target=_sample, daemon=True)
        sampler.start()
        try:
            result = fn()
        finally:
            stop.set()
            sampler.join(timeout=1.0)
        try:
            peak[0] = max(peak[0], proc.memory_info().rss)
        except Exception:
            pass
        return result, max(0, peak[0] - baseline)

    def _get_part_assignment_pool(self) -> ProcessPoolExecutor:
        if self._part_assignment_pool is None:
            workers = self._get_part_assignment_workers()
            _spawn_ctx = multiprocessing.get_context('spawn')
            self._part_assignment_pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=_spawn_ctx,
            )
            self._part_assignment_pool_size = workers
        return self._part_assignment_pool

    def _part_assignment_scratch(self) -> str:
        if self._part_assignment_scratch_dir is None:
            self._part_assignment_scratch_dir = tempfile.mkdtemp(prefix="castsim_parts_")
        return self._part_assignment_scratch_dir

    def _shutdown_part_assignment_pool(self) -> None:
        pool = self._part_assignment_pool
        self._part_assignment_pool = None
        self._part_assignment_pool_size = 0
        if pool is not None:
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except Exception as e:
                logger.debug(f"Part assignment pool shutdown error: {e}")
        scratch = self._part_assignment_scratch_dir
        self._part_assignment_scratch_dir = None
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)

    def _should_parallelize_part_assignment(self, n_candidates: int, batch_size: Optional[int] = None) -> bool:
        if self.part_assignment_mode == "legacy":
            return False
        workers = self._get_part_assignment_workers()
        if workers <= 1:
            return False
        if self.part_assignment_mode == "parallel":
            return True
        batch = batch_size if batch_size is not None else self.part_assignment_batch_size
        return n_candidates >= max(
            self.AUTO_PART_ASSIGNMENT_MIN_CANDIDATES,
            batch * workers * self.PART_ASSIGNMENT_SUBMIT_MULTIPLIER,
        )

    def _run_part_assignment_sequential(
        self,
        *,
        mesh: trimesh.Trimesh,
        candidate_indices: np.ndarray,
        voxel_centers_array: np.ndarray,
        part_idx: int,
        total_parts: int,
        type_name: str,
        tolerance: float,
        batch_size: Optional[int] = None,
        start_batch: int = 0,
        initial_inside: Optional[List[np.ndarray]] = None,
    ) -> List[int]:
        batch_size = batch_size or self.part_assignment_batch_size
        found: List[np.ndarray] = list(initial_inside or [])
        inside_count = sum(len(a) for a in found)
        n_candidates = len(candidate_indices)
        n_batches = (n_candidates + batch_size - 1) // batch_size

        for batch_idx in range(start_batch, n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_candidates)
            batch_indices = candidate_indices[start:end]
            batch_centers = voxel_centers_array[batch_indices]

            try:
                inside_mask = self._robust_contains_check(mesh, batch_centers, tolerance)
                batch_inside = batch_indices[inside_mask]
                found.append(batch_inside)
                inside_count += len(batch_inside)
            except Exception as e:
                logger.warning(f"Part type batch processing error: {e}")
                print(f" [batch error: {e}]", end="", flush=True)

            pct = int((batch_idx + 1) / n_batches * 100)
            print(
                f"\r    [{part_idx+1}/{total_parts}] Checking {type_name} part... {pct}% ({inside_count:,} inside)",
                end="",
                flush=True,
            )

        if not found:
            return []
        return list(np.concatenate(found))

    def _run_part_assignment_parallel(
        self,
        *,
        mesh: trimesh.Trimesh,
        candidate_indices: np.ndarray,
        voxel_centers_array: np.ndarray,
        part_idx: int,
        total_parts: int,
        type_name: str,
        tolerance: float,
        batch_size: Optional[int] = None,
    ) -> List[int]:
        batch_size = batch_size or self.part_assignment_batch_size
        n_candidates = len(candidate_indices)
        n_batches = (n_candidates + batch_size - 1) // batch_size

        def _batch_slice(batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
            start = batch_idx * batch_size
            end = min(start + batch_size, n_candidates)
            batch_indices = candidate_indices[start:end]
            return batch_indices, voxel_centers_array[batch_indices]

        def _sequential_from(start_batch: int, found: Dict[int, np.ndarray]) -> List[int]:
            ordered = [found[i] for i in sorted(found)]
            return self._run_part_assignment_sequential(
                mesh=mesh,
                candidate_indices=candidate_indices,
                voxel_centers_array=voxel_centers_array,
                part_idx=part_idx,
                total_parts=total_parts,
                type_name=type_name,
                tolerance=tolerance,
                batch_size=batch_size,
                start_batch=start_batch,
                initial_inside=ordered,
            )

        found_by_batch: Dict[int, np.ndarray] = {}
        try:
            probe_indices, probe_centers = _batch_slice(0)
            probe_mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False
            )
            probe_t0 = time.perf_counter()
            probe_mask, probe_peak = self._measure_peak_rss(
                lambda: self._robust_contains_check(probe_mesh, probe_centers, tolerance)
            )
            probe_seconds = time.perf_counter() - probe_t0
            found_by_batch[0] = probe_indices[probe_mask]
            if n_batches == 1:
                del probe_mesh
                return list(found_by_batch[0])

            steady_seconds = probe_seconds
            next_batch_idx = 1
            if n_batches > 2:
                b1_indices, b1_centers = _batch_slice(1)
                t1 = time.perf_counter()
                b1_mask = self._robust_contains_check(probe_mesh, b1_centers, tolerance)
                steady_seconds = time.perf_counter() - t1
                found_by_batch[1] = b1_indices[b1_mask]
                next_batch_idx = 2
            del probe_mesh

            sequential_estimate = steady_seconds * (n_batches - next_batch_idx)
            if (
                self.part_assignment_mode == "auto"
                and self._part_assignment_pool is None
                and sequential_estimate < self.PART_ASSIGNMENT_MIN_PARALLEL_SECONDS
            ):
                logger.info(
                    f"Part assignment: {n_batches} batches at ~{steady_seconds*1000:.0f} ms each, "
                    f"~{sequential_estimate:.1f}s sequential estimate; not worth a worker pool"
                )
                return _sequential_from(next_batch_idx, found_by_batch)

            per_worker = _WORKER_BASE_BYTES + int(probe_peak * self.PART_ASSIGNMENT_PROBE_SAFETY)

            pool = self._get_part_assignment_pool()
            concurrency = self._memory_bounded_concurrency(per_worker, self._part_assignment_pool_size)
            if concurrency < 2:
                logger.info(
                    f"Part assignment: batch peak {probe_peak/2**20:.0f} MB, "
                    f"headroom allows {concurrency} worker(s); running sequentially"
                )
                return _sequential_from(next_batch_idx, found_by_batch)

            logger.info(
                f"Part assignment: {n_batches} batches of {batch_size}, batch peak "
                f"{probe_peak/2**20:.0f} MB, ~{per_worker/2**20:.0f} MB/worker, "
                f"{concurrency}/{self._part_assignment_pool_size} workers"
            )

            mesh_path = os.path.join(self._part_assignment_scratch(), f"part_{part_idx}.npz")
            np.savez(mesh_path, vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces))
            part_key = f"{id(mesh)}:{part_idx}"
            reserve = self._memory_reserve_bytes()

            futures: Dict[object, int] = {}
            inside_count = sum(len(v) for v in found_by_batch.values())
            completed = len(found_by_batch)
            degraded = False

            def submit_batch(batch_idx: int) -> None:
                batch_indices, batch_centers = _batch_slice(batch_idx)
                future = pool.submit(
                    _part_assignment_contains_worker,
                    part_key,
                    mesh_path,
                    batch_idx,
                    batch_indices,
                    batch_centers,
                    tolerance,
                )
                futures[future] = batch_idx

            def can_submit() -> bool:
                return _available_memory_bytes() > reserve

            while next_batch_idx < n_batches and len(futures) < concurrency:
                if not can_submit():
                    degraded = True
                    break
                submit_batch(next_batch_idx)
                next_batch_idx += 1

            while futures:
                done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future, None)
                    chunk_idx, inside_chunk = future.result()
                    found_by_batch[chunk_idx] = inside_chunk
                    inside_count += len(inside_chunk)
                    completed += 1
                    pct = int(completed / n_batches * 100)
                    print(
                        f"\r    [{part_idx+1}/{total_parts}] Checking {type_name} part... {pct}% ({inside_count:,} inside)",
                        end="",
                        flush=True,
                    )

                    if next_batch_idx < n_batches and not degraded:
                        if can_submit():
                            submit_batch(next_batch_idx)
                            next_batch_idx += 1
                        else:
                            degraded = True

            if degraded and next_batch_idx < n_batches:
                logger.warning(
                    f"Part assignment: available RAM fell below the "
                    f"{reserve/2**30:.1f} GB reserve; finishing "
                    f"{n_batches - next_batch_idx} batches sequentially"
                )
                print(f"\n      Memory guard: finishing part sequentially", flush=True)
                return _sequential_from(next_batch_idx, found_by_batch)

            ordered = [found_by_batch[i] for i in sorted(found_by_batch)]
            if not ordered:
                return []
            return list(np.concatenate(ordered))
        except Exception as e:
            logger.error(f"Parallel part assignment failed; falling back to sequential: {e}")
            print(f"\n      Parallel fallback: {e}")
            self._shutdown_part_assignment_pool()
            return self._run_part_assignment_sequential(
                mesh=mesh,
                candidate_indices=candidate_indices,
                voxel_centers_array=voxel_centers_array,
                part_idx=part_idx,
                total_parts=total_parts,
                type_name=type_name,
                tolerance=tolerance,
                batch_size=batch_size,
            )

    def get_voxels_by_type(self, part_type: int) -> np.ndarray:
        if self.voxel_part_types is None:
            return np.array([], dtype=int)
        return np.where(self.voxel_part_types == part_type)[0]

    def _estimate_memory_mb(self) -> float:
        if self.voxel_centers is None:
            return 0.0

        total_bytes = (
            self.voxel_centers.nbytes +
            self.voxel_sizes.nbytes +
            self.voxel_volumes.nbytes +
            self.voxel_surface_areas.nbytes +
            self.voxel_moduli.nbytes +
            self.voxel_neighbors.nbytes +
            self.voxel_states.nbytes +
            self.surface_voxels.nbytes
        )

        if self.voxel_part_types is not None:
            total_bytes += self.voxel_part_types.nbytes

        return total_bytes / (1024 ** 2)

    def get_memory_stats(self) -> dict:
        return self.memory_manager.get_memory_stats()

    def print_memory_stats(self):
        self.memory_manager.print_stats()

    def get_voxel_count(self) -> int:
        return len(self.voxel_centers) if self.voxel_centers is not None else 0

    def get_surface_voxel_indices(self) -> np.ndarray:
        if self.surface_voxels is None:
            return np.array([], dtype=int)
        return np.where(self.surface_voxels)[0]

    def get_material_surface_voxel_indices(self, include_cores: bool = True) -> np.ndarray:
        cached = self._material_surface_indices if include_cores else self._chill_surface_indices
        if cached is not None:
            return cached

        geometric = self.get_surface_voxel_indices()
        part_types = self.voxel_part_types
        neighbors = self.voxel_neighbors
        if part_types is None or neighbors is None or self.voxel_centers is None:
            return self._cache_material_surface(geometric, include_cores)

        part_types = np.asarray(part_types)
        if not np.any(is_mold(part_types)):
            return self._cache_material_surface(geometric, include_cores)

        n_voxels = len(self.voxel_centers)
        mask = np.zeros(n_voxels, dtype=bool)
        mask[geometric] = True

        chunk = 100_000
        for start in range(0, n_voxels, chunk):
            end = min(start + chunk, n_voxels)
            nbr = neighbors[start:end]
            valid = nbr >= 0
            nbr_types = np.where(valid, part_types[np.maximum(nbr, 0)], -1)
            backing = (nbr_types == 2) | (nbr_types == 4) if include_cores else nbr_types == 2
            mold_adjacent = np.any(backing, axis=1)
            mask[start:end] |= mold_adjacent & (part_types[start:end] == 0)

        return self._cache_material_surface(np.flatnonzero(mask), include_cores)

    def _cache_material_surface(self, indices: np.ndarray, include_cores: bool) -> np.ndarray:
        if include_cores:
            self._material_surface_indices = indices
        else:
            self._chill_surface_indices = indices
        return indices

    def cleanup(self):
        self.memory_manager.cleanup()
