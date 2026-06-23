"""
Adaptive Voxel Mesh Generator
Optimized for M3 multi-core processing
Adapts voxel size based on geometry complexity
Uses memory-mapped arrays for large models to prevent RAM exhaustion

CRITICAL: Includes hard memory limits to prevent RAM/swap exhaustion

Performance Optimizations:
- Chunked voxelization: Process large models in Z-slices with parallel workers
- Octree spatial queries: Optional octree for faster neighbor searches
- GPU acceleration: Uses GPU when available for distance computations
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import trimesh
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import multiprocessing
from core.acceleration import (
    accelerator, 
    compute_distances_parallel,
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

# Import error logger
try:
    from core.error_logger import logger
except ImportError:
    class DummyLogger:
        def debug(self, msg, **kwargs): pass
        def info(self, msg, **kwargs): pass
        def warning(self, msg, **kwargs): pass
        def error(self, msg, **kwargs): pass
        def exception(self, msg, **kwargs): pass
        def log_memory_warning(self, *args, **kwargs): pass
        def log_performance(self, *args, **kwargs): pass
    logger = DummyLogger()

# Optional imports for enhanced performance
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
    from rtree import index as rtree_index
    import rtree
except ImportError:
    RTREE_AVAILABLE = False


_PART_ASSIGNMENT_WORKER_MESH = None


def _init_part_assignment_worker(mesh_payload: Dict[str, np.ndarray]) -> None:
    """Initialize worker-local mesh cache for part assignment containment checks."""
    global _PART_ASSIGNMENT_WORKER_MESH
    _PART_ASSIGNMENT_WORKER_MESH = trimesh.Trimesh(
        vertices=np.asarray(mesh_payload["vertices"]),
        faces=np.asarray(mesh_payload["faces"]),
        process=False,
    )


def _robust_contains_check_impl(mesh: trimesh.Trimesh, points: np.ndarray, tolerance: float) -> np.ndarray:
    """Shared containment logic used by both main and worker processes."""
    n_points = len(points)
    inside_mask = np.zeros(n_points, dtype=bool)

    # Strategy 1: Try mesh.contains() first (most reliable when it works)
    try:
        inside_mask = mesh.contains(points)
        # Sanity check: if contains() returns all False, it might have failed
        if np.sum(inside_mask) == 0:
            logger.debug(f"mesh.contains() returned all False for {n_points} points, trying fallbacks")
        else:
            # Also check proximity for points near the surface (touching assemblies)
            try:
                if hasattr(mesh, 'nearest'):
                    _closest_pts, distances, _ = mesh.nearest.on_surface(points)
                    near_surface = distances < tolerance
                    inside_mask = inside_mask | near_surface
            except Exception:
                pass  # Proximity check failed, use contains() result
            return inside_mask
    except Exception as e:
        logger.warning(f"mesh.contains() failed: {e}, trying fallbacks")

    # Strategy 2: Try signed distance (more reliable for watertight meshes)
    # Negative distance = inside, positive = outside
    try:
        from trimesh.proximity import signed_distance
        signed_dists = signed_distance(mesh, points)
        # Inside if signed distance is negative (or within tolerance)
        inside_mask = signed_dists <= tolerance
        if np.sum(inside_mask) > 0:
            return inside_mask
    except Exception as e:
        logger.debug(f"signed_distance fallback failed: {e}")

    # Strategy 3: Proximity-based containment (for non-watertight meshes)
    # If a point is very close to the mesh surface, consider it "inside"
    try:
        if hasattr(mesh, 'nearest'):
            _closest_pts, distances, _ = mesh.nearest.on_surface(points)
            # Consider points within 1.5x tolerance as "inside"
            # This is more generous for RISER/CHILL parts that may touch the casting
            inside_mask = distances < (tolerance * 1.5)
            if np.sum(inside_mask) > 0:
                return inside_mask
    except Exception as e:
        logger.debug(f"proximity fallback failed: {e}")

    # Strategy 4: Simple bounding box containment (last resort)
    # This is least accurate but always works
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


def _part_assignment_contains_worker(chunk_idx: int, chunk_indices: np.ndarray,
                                     chunk_points: np.ndarray, tolerance: float) -> Tuple[int, np.ndarray]:
    """Worker task for part-assignment containment checks."""
    if _PART_ASSIGNMENT_WORKER_MESH is None:
        raise RuntimeError("Part assignment worker mesh was not initialized")
    inside_mask = _robust_contains_check_impl(_PART_ASSIGNMENT_WORKER_MESH, chunk_points, tolerance)
    return chunk_idx, chunk_indices[inside_mask]


class AdaptiveVoxelMesh:
    """
    Creates adaptive voxel mesh for casting simulation
    - Fine voxels for intricate sections
    - Coarse voxels for solid sections
    - ENFORCES memory limits to prevent system crashes
    - Tracks part types (CAST, RISER, CHILL) for multi-part simulations
    
    Performance Features:
    - Chunked voxelization for models exceeding RAM
    - Octree-based spatial queries for large meshes
    - GPU acceleration when available
    """

    AUTO_PART_ASSIGNMENT_MIN_CANDIDATES = 10_000
    AUTO_PART_ASSIGNMENT_MEMORY_PER_WORKER_BYTES = int(1.5 * 1024 ** 3)
    PART_ASSIGNMENT_SUBMIT_MULTIPLIER = 2
    
    @staticmethod
    def preview_voxel_count(
        trimesh_obj: trimesh.Trimesh,
        voxel_size: float,
        refinement_level: int = 1
    ) -> dict:
        """
        Fast preview of expected voxel count without full mesh generation.
        
        Uses trimesh's fast voxelization to get accurate count estimate.
        This is much faster than full generate() and suitable for UI feedback.
        
        Args:
            trimesh_obj: Input mesh (must be valid trimesh.Trimesh)
            voxel_size: Base voxel size in inches (must be > 0)
            refinement_level: Refinement level 1-4 (affects final count estimate)
        
        Returns:
            dict with keys:
                - estimated_voxels: int
                - estimated_memory_mb: float
                - estimated_time_seconds: float
                - fill_ratio: float
                - bounds: tuple
                - grid_shape: tuple (nx, ny, nz)
                - can_proceed: bool
                - warning: str or None
                - error: str or None (set if an error occurred)
        """
        from core.memory_limits import (
            estimate_memory_usage_mb, 
            get_max_voxels_for_system,
            BYTES_PER_VOXEL
        )
        
        # Validate inputs
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
        
        # Validate voxel size
        if voxel_size is None or voxel_size <= 0:
            voxel_size = 0.1  # Default fallback
        voxel_size = max(0.01, min(voxel_size, 2.0))  # Clamp to reasonable range
        
        # Validate refinement level
        refinement_level = max(1, min(4, int(refinement_level) if refinement_level else 1))
        
        try:
            # Get bounds
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
        
        # Calculate bounding box volume and mesh volume
        bbox_volume = x_range * y_range * z_range
        mesh_volume = float(trimesh_obj.volume) if trimesh_obj.volume > 0 else bbox_volume * 0.4
        fill_ratio = max(0.1, min(mesh_volume / bbox_volume, 0.95)) if bbox_volume > 0 else 0.4
        
        # Calculate grid dimensions
        padding = voxel_size * 0.5
        nx = int(np.ceil((x_range + 2 * padding) / voxel_size))
        ny = int(np.ceil((y_range + 2 * padding) / voxel_size))
        nz = int(np.ceil((z_range + 2 * padding) / voxel_size))
        
        total_grid = nx * ny * nz
        
        # Estimate filled voxels using fill ratio
        estimated_voxels = int(total_grid * fill_ratio)
        
        # Account for refinement (each level can add ~30% more voxels on average)
        refinement_factor = 1.0 + (refinement_level - 1) * 0.3
        estimated_voxels = int(estimated_voxels * refinement_factor)
        
        # For more accuracy on small meshes, try quick voxelization
        try:
            if total_grid < 500_000:  # Only for reasonable sizes
                quick_voxels = trimesh_obj.voxelized(pitch=voxel_size)
                if quick_voxels is not None:
                    actual_count = len(quick_voxels.points) if hasattr(quick_voxels, 'points') else 0
                    if actual_count > 0:
                        estimated_voxels = int(actual_count * refinement_factor)
        except Exception:
            pass  # Fall back to estimate
        
        # Memory and time estimates
        estimated_memory_mb = estimate_memory_usage_mb(estimated_voxels)
        
        # Time estimate based on empirical data from mesh_runs.csv
        # ~0.00005s per voxel for mesh gen, ~0.00004s for simulation
        estimated_time = estimated_voxels * 0.00009
        
        # Check if can proceed
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
                 part_assignment_batch_size: int = 2500):
        """
        Args:
            base_voxel_size: Base voxel size in inches
            max_refinement_level: Maximum refinement levels (each level halves voxel size)
            memory_mode: "ram" (force RAM), "disk" (force disk), or "auto" (choose based on available memory)
            use_octree: Use octree for spatial queries (faster for very large meshes >500K voxels)
            use_chunked_voxelization: Use chunked voxelization for models exceeding RAM
            part_assignment_mode: "auto", "parallel", or "legacy"
            part_assignment_workers: Worker count for parallel part assignment (0 = auto)
            part_assignment_batch_size: Candidate points per containment batch
        """
        self.base_voxel_size = base_voxel_size
        self.original_voxel_size = base_voxel_size  # Track if we had to adjust
        self.max_refinement_level = max_refinement_level
        self.memory_mode = memory_mode
        self.use_octree = use_octree and OCTREE_AVAILABLE
        self.use_chunked_voxelization = use_chunked_voxelization
        mode = str(part_assignment_mode).lower()
        self.part_assignment_mode = mode if mode in {"auto", "parallel", "legacy"} else "auto"
        self.part_assignment_workers = max(0, int(part_assignment_workers))
        self.part_assignment_batch_size = max(256, int(part_assignment_batch_size))
        self.voxel_size_was_adjusted = False
        self.adjustment_message = ""
        
        self.voxels = None
        self.voxel_centers = None
        self.voxel_sizes = None
        self.voxel_volumes = None
        self.voxel_surface_areas = None
        self.voxel_moduli = None
        self.voxel_neighbors = None
        # Heuristic solidification-progress field. Downstream analyses often use
        # it as a fraction-solid proxy, but it is not a full thermodynamic phase
        # solution.
        self.voxel_states = None
        self.surface_voxels = None
        
        # Spatial index for fast queries
        self._spatial_index = None
        
        # Part type tracking: 0=CAST, 1=RISER, 2=CHILL
        self.voxel_part_types = None
        
        self.bounds = None
        self.grid_shape = None
        self.memory_manager = get_memory_manager(memory_mode)
        self.memory_monitor = MemoryMonitor(abort_threshold_percent=90.0)
        
    def generate(self, trimesh_obj: trimesh.Trimesh, 
                bounds: np.ndarray,
                show_progress: bool = True,
                progress_callback=None,
                auto_adjust: bool = True) -> bool:
        """
        Generate adaptive voxel mesh from trimesh object
        
        Args:
            trimesh_obj: Input mesh
            bounds: Mesh bounds [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            show_progress: Show progress information
            progress_callback: Optional callback(current, total, stage) for progress updates
            auto_adjust: Automatically adjust voxel size if it would exceed memory limits
        
        Returns:
            True if successful
        """
        if show_progress:
            print("\n" + "="*60)
            print("Generating Adaptive Voxel Mesh (Multi-core)")
            print("="*60)
        
        self.bounds = bounds
        
        # Convert bounds to tuple format for validation
        bounds_tuple = (
            (float(bounds[0][0]), float(bounds[0][1])),
            (float(bounds[1][0]), float(bounds[1][1])),
            (float(bounds[2][0]), float(bounds[2][1])),
        )
        
        # CRITICAL: Pre-validate memory usage before proceeding
        if show_progress:
            print("\nStep 0: Validating memory requirements...")
        
        # Estimate fill ratio from mesh
        bbox_volume = np.prod([b[1] - b[0] for b in bounds_tuple])
        mesh_volume = float(trimesh_obj.volume) if trimesh_obj.volume > 0 else bbox_volume * 0.5
        fill_ratio = max(0.1, min(mesh_volume / bbox_volume, 0.9)) if bbox_volume > 0 else 0.5
        
        # Auto-adjust voxel size if needed
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
        
        # Validate the (possibly adjusted) parameters
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
        
        # Start memory monitoring
        self.memory_monitor.start()
        
        # Step 1: Create initial uniform grid
        if show_progress:
            print("\nStep 1: Creating initial uniform grid...")
        
        initial_voxels = self._create_uniform_grid(trimesh_obj, progress_callback)

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
        
        # Check memory after voxel creation
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
        
        # Step 2: Skip refinement for memory efficiency
        if show_progress:
            print("Step 2: Skipping refinement (using uniform mesh for memory efficiency)...")
        
        refined_voxels = initial_voxels  # No refinement to save memory
        
        # Step 3: Calculate voxel properties (parallel)
        if show_progress:
            print("Step 3: Computing voxel properties (parallel)...")
        
        self._calculate_voxel_properties(refined_voxels, trimesh_obj)
        
        # Check memory after property calculation
        can_continue, mem_status = self.memory_monitor.check()
        if not can_continue:
            logger.error(f"Memory limit exceeded during property calculation: {mem_status}")
            print(f"\n❌ {mem_status}")
            return False
        
        # Step 4: Build neighbor connectivity (parallel)
        if show_progress:
            print("Step 4: Building neighbor connectivity...")
            print(f"  Memory status: {mem_status}")
        
        self._build_neighbor_graph()
        
        # Final memory check
        can_continue, mem_status = self.memory_monitor.check()
        if not can_continue:
            logger.error(f"Memory limit exceeded during neighbor graph build: {mem_status}")
            print(f"\n❌ {mem_status}")
            return False
        
        # Step 5: Identify surface voxels
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
        logger.log_performance("Voxel mesh generation", 0, len(self.voxel_centers))  # Duration tracked elsewhere
        return True
    
    def _create_uniform_grid(self, trimesh_obj: trimesh.Trimesh, progress_callback=None) -> Union[List[Dict], np.ndarray]:
        """Create initial uniform voxel grid - optimized using trimesh voxelization"""
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        z_min, z_max = self.bounds[2]
        
        # Minimal padding to reduce memory
        padding = self.base_voxel_size * 0.5
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        z_min -= padding
        z_max += padding
        
        # Calculate grid dimensions for reference
        nx = int(np.ceil((x_max - x_min) / self.base_voxel_size))
        ny = int(np.ceil((y_max - y_min) / self.base_voxel_size))
        nz = int(np.ceil((z_max - z_min) / self.base_voxel_size))
        
        self.grid_shape = (nx, ny, nz)
        total_candidates = nx * ny * nz
        
        print(f"  Grid: {nx} x {ny} x {nz} = {total_candidates:,} candidate voxels")
        
        # Memory check: warn if this will create too many voxels
        estimated_mb = (total_candidates * 32) / (1024 * 1024)  # 32 bytes per voxel estimate
        if estimated_mb > 500:
            print(f"  WARNING: Large mesh will use ~{estimated_mb:.0f}MB. Consider larger voxel size.")
        
        # Determine whether to use chunked voxelization
        # Use chunked approach for models with > 100K voxels OR > 500 MB estimated memory
        use_chunked = self.use_chunked_voxelization and (
            total_candidates > 100_000 or estimated_mb > 500
        )
        
        if use_chunked:
            print(f"  Using CHUNKED parallel voxelization for large model...")
            return self._create_uniform_grid_chunked(
                trimesh_obj, x_min, x_max, y_min, y_max, z_min, z_max,
                nx, ny, nz, progress_callback
            )
        
        # Standard voxelization for smaller models
        print(f"  Using optimized trimesh voxelization...")
        
        try:
            # trimesh.voxelized uses efficient ray-casting internally
            voxel_grid = trimesh_obj.voxelized(pitch=self.base_voxel_size)
            
            # Fill hollow regions (important for castings)
            voxel_grid = voxel_grid.fill()
            
            # Get voxel centers
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
            # Fallback to chunked method if trimesh voxelization fails
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
        """
        Chunked parallel voxelization for models that exceed available RAM.

        Processes mesh in Z-slices using parallel workers, then merges chunks
        with minimal memory overhead. This enables processing of arbitrarily
        large models by trading time for memory.

        Performance characteristics:
        - Memory usage: O(n_voxels_per_slice) instead of O(total_voxels)
        - Parallel speedup: Near-linear with CPU cores
        - GPU acceleration: Used when available for containment checks
        """
        if not RTREE_AVAILABLE:
            raise RuntimeError(
                "rtree dependency is required for chunked voxelization via "
                "trimesh.contains(). Install with: pip install rtree"
            )

        n_workers = hardware.get_optimal_thread_count('compute')

        # Determine optimal chunk size based on available memory
        available_mb = self.memory_manager.get_available_memory_gb() * 1024
        voxels_per_slice = nx * ny
        bytes_per_voxel = 32  # Approximate
        mb_per_slice = (voxels_per_slice * bytes_per_voxel) / (1024 * 1024)

        # Target using ~30% of available memory per batch of slices
        target_mb = available_mb * 0.3
        slices_per_batch = max(1, int(target_mb / mb_per_slice))
        slices_per_batch = min(slices_per_batch, nz)  # Don't exceed total slices

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

        # Process in batches of Z-slices
        for batch_start in range(0, nz, slices_per_batch):
            batch_end = min(batch_start + slices_per_batch, nz)
            batch_z_coords = z_coords[batch_start:batch_end]

            # Process slices in this batch in parallel
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

            # Progress update
            progress_pct = int(processed_slices / nz * 100)
            print(f"    Processed Z-slices {batch_start+1}-{batch_end}/{nz} "
                  f"({progress_pct}%), found {len(all_voxels):,} voxels so far")

            if progress_callback:
                progress_callback(progress_pct, 100, 'voxelization')

            # Memory cleanup between batches
            import gc
            gc.collect()
        
        print(f"  Total voxels found: {len(all_voxels):,}")
        return all_voxels
    
    def _process_z_slice_batch_parallel(self, trimesh_obj: trimesh.Trimesh,
                                        x_coords: np.ndarray,
                                        y_coords: np.ndarray,
                                        z_coords: np.ndarray,
                                        n_workers: int) -> Tuple[List[Dict], int]:
        """
        Process a batch of Z-slices in parallel.

        Each worker processes one or more Z-slices and returns voxels found.

        Returns:
            Tuple of (voxels list, failed slice count)
        """
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

        # Parallel processing for larger batches
        z_chunks = np.array_split(z_coords, min(n_workers, len(z_coords)))

        all_voxels = []
        failed_slices = 0

        # Use ThreadPoolExecutor (safer than ProcessPoolExecutor with trimesh)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for chunk in z_chunks:
                if len(chunk) > 0:
                    future = executor.submit(
                        self._process_z_slice_chunk,
                        trimesh_obj, x_coords, y_coords, chunk
                    )
                    futures[future] = len(chunk)

            # Collect results
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
        """Process a chunk of Z-slices (called by parallel workers).

        Returns:
            Tuple of (voxels list, failed slice count)
        """
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
        """
        Process a single Z-slice to find occupied voxels.
        
        Uses GPU acceleration when available for containment checks.
        """
        # Create 2D grid for this Z slice
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
        zz = np.full_like(xx, z_val)
        slice_centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        
        # Check containment (GPU-accelerated if available)
        if GPU_AVAILABLE:
            # Try GPU-accelerated containment
            inside = gpu_compute.compute_voxel_containment_gpu(
                slice_centers.astype(np.float32),
                trimesh_obj.vertices.astype(np.float32),
                trimesh_obj.faces
            )
            if inside is None:
                # GPU fallback returned None, use CPU
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
        
        # Collect occupied voxels
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
        """Fallback: Create uniform grid using slice-by-slice contains() - slower but reliable"""
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
        
        # Process Z slices to reduce memory
        for iz in range(nz):
            z_val = z_coords[iz]
            
            # Create 2D grid for this Z slice
            xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
            zz = np.full_like(xx, z_val)
            slice_centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
            
            # Check this slice
            if not RTREE_AVAILABLE:
                raise RuntimeError(
                    "rtree is required for trimesh.contains() in fallback voxelization. "
                    "Install with: pip install rtree"
                )
            inside = trimesh_obj.contains(slice_centers)
            
            # Add occupied voxels
            for center, is_inside in zip(slice_centers, inside):
                if is_inside:
                    voxels.append({
                        'center': center.astype(np.float32),
                        'size': self.base_voxel_size,
                        'level': 0
                    })
            
            # Free memory after each slice
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
        """
        Refine voxels based on geometry complexity
        Uses parallel processing for feature detection
        """
        refined = []
        
        # Build KD-tree for mesh vertices
        vertex_tree = cKDTree(trimesh_obj.vertices)
        
        for voxel in voxels:
            center = voxel['center']
            size = voxel['size']
            level = voxel['level']
            
            # Check if refinement needed
            # Criteria: proximity to surface, curvature, thickness variation
            
            # Distance to nearest surface vertex
            dist, _ = vertex_tree.query(center)
            
            # If close to surface and not at max level, refine
            if dist < size * 2 and level < self.max_refinement_level:
                # Split into 8 sub-voxels
                sub_size = size / 2
                offsets = [
                    [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
                ]
                
                for offset in offsets:
                    sub_center = center + np.array(offset) * sub_size / 2
                    
                    # Check if sub-voxel is inside
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
        """
        Calculate voxel properties in parallel
        - Volume
        - Surface area
        - Geometric modulus
        Uses memory-mapped arrays for large datasets
        """
        n_voxels = len(voxels)
        
        print(f"\n  Allocating memory for {n_voxels:,} voxels...")
        print(f"  Memory mode: {self.memory_mode.upper()}")
        
        # Use memory-managed arrays (auto-switches to disk if needed)
        self.voxel_centers = create_managed_array(
            (n_voxels, 3), dtype=np.float32, name="voxel_centers", manager=self.memory_manager
        )
        self.voxel_sizes = create_managed_array(
            n_voxels, dtype=np.float32, name="voxel_sizes", manager=self.memory_manager
        )
        
        # Fill arrays from voxel list in chunks to reduce memory pressure
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
        
        # Free the voxels list immediately
        print(f"  Freeing temporary voxel list...")
        del voxels
        import gc
        gc.collect()
        
        # Calculate volumes
        self.voxel_volumes = create_managed_array(
            n_voxels, dtype=np.float32, name="voxel_volumes", manager=self.memory_manager
        )
        np.multiply(self.voxel_sizes, self.voxel_sizes, out=self.voxel_volumes)
        np.multiply(self.voxel_volumes, self.voxel_sizes, out=self.voxel_volumes)
        
        # Estimate surface areas (6 faces, but account for internal/external)
        # External voxels have more exposed surface
        self.voxel_surface_areas = create_managed_array(
            n_voxels, dtype=np.float32, name="voxel_surface_areas", manager=self.memory_manager
        )
        
        # Build KD-tree for voxel centers
        voxel_tree = cKDTree(self.voxel_centers)
        
        # Identify surface voxels - use VECTORIZED batch queries for speed
        print(f"  Computing surface areas for {n_voxels:,} voxels...")
        
        voxel_centers_array = np.asarray(self.voxel_centers)
        voxel_sizes_array = np.asarray(self.voxel_sizes)
        
        # Direction vectors for 6-connectivity
        directions_base = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=np.float32)
        
        surface_indices = []
        
        # Process in batches with vectorized queries per direction
        BATCH_SIZE = 10000
        n_batches = (n_voxels + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, n_voxels)
            batch_size = end - start
            
            batch_centers = voxel_centers_array[start:end]
            batch_sizes = voxel_sizes_array[start:end]
            
            # Count exposed faces for each voxel in batch
            exposed_counts = np.zeros(batch_size, dtype=np.int32)
            
            for direction in directions_base:
                # Compute neighbor positions for all voxels in batch
                neighbor_positions = batch_centers + direction * batch_sizes[:, np.newaxis]
                
                # Query all neighbor positions at once (vectorized)
                distances, _ = voxel_tree.query(neighbor_positions, k=1)
                
                # Check if no neighbor (exposed face)
                exposed = distances > batch_sizes * 0.5
                exposed_counts += exposed.astype(np.int32)
            
            # Calculate surface areas
            self.voxel_surface_areas[start:end] = exposed_counts * (batch_sizes ** 2)
            
            # Track surface voxels
            surface_mask = exposed_counts > 0
            batch_surface_indices = np.where(surface_mask)[0] + start
            surface_indices.extend(batch_surface_indices.tolist())
            
            if (batch_idx + 1) % 5 == 0 or end == n_voxels:
                print(f"    Processed {end:,}/{n_voxels:,} voxels...", flush=True)
        
        # Calculate geometric modulus using distance to surface
        # This gives true section thickness for casting analysis
        print("  Computing geometric modulus (distance to surface)...")
        self.voxel_moduli = create_managed_array(
            n_voxels, dtype=np.float32, name="voxel_moduli", manager=self.memory_manager
        )
        
        if len(surface_indices) > 0:
            surface_centers = self.voxel_centers[surface_indices]
            surface_tree = cKDTree(surface_centers)
            
            # OPTIMIZED: Query all voxels at once (vectorized) instead of one at a time
            print(f"    Querying distances for {n_voxels:,} voxels to {len(surface_indices):,} surface voxels...")
            all_centers = np.asarray(self.voxel_centers)
            distances, _ = surface_tree.query(all_centers, workers=-1)  # Use all CPU cores
            
            # Copy distances to moduli array
            self.voxel_moduli[:] = distances.astype(np.float32)
            
            # Surface voxels have 0 modulus (they ARE the surface)
            surface_set = set(surface_indices)
            for i in surface_indices:
                self.voxel_moduli[i] = 0.0
        else:
            # Fallback if no surface detected
            temp_moduli = compute_modulus_parallel(
                self.voxel_volumes, 
                self.voxel_surface_areas
            )
            self.voxel_moduli[:] = temp_moduli
        
        # Initialize solidification states
        self.voxel_states = create_managed_array(
            n_voxels, dtype=np.float32, name="voxel_states", manager=self.memory_manager
        )
        
        # Debug output
        print(f"  Modulus range: {np.min(self.voxel_moduli):.4f} - {np.max(self.voxel_moduli):.4f} inches")
        print(f"  Mean modulus: {np.mean(self.voxel_moduli):.4f} inches")
    
    def _build_neighbor_graph(self):
        """
        Build neighbor connectivity graph
        Each voxel has up to 26 neighbors (3D)
        
        Uses 2.0x voxel size as search radius to ensure connectivity
        between voxels at part interfaces (CAST-RISER, CAST-CHILL, etc.)
        
        Performance optimization: Uses octree for O(log n) queries when
        available, falls back to KD-tree otherwise.
        """
        n_voxels = len(self.voxel_centers)
        
        print(f"  Building neighbor graph for {n_voxels:,} voxels...")
        
        # Allocate neighbor array first
        self.voxel_neighbors = create_managed_array(
            (n_voxels, 26), dtype=np.int32, name="voxel_neighbors", manager=self.memory_manager
        )
        self.voxel_neighbors[:] = -1  # Initialize with -1
        
        # Free memory before building tree
        import gc
        gc.collect()
        
        # Determine search radius
        search_radius = self.base_voxel_size * 2.0  # Extended for interface connectivity
        
        # Choose spatial index based on configuration and size
        if self.use_octree and n_voxels > 10000:
            print(f"  Building OCTREE spatial index for {n_voxels:,} voxels...")
            
            # Build octree
            bounds_min = np.min(self.voxel_centers, axis=0)
            bounds_max = np.max(self.voxel_centers, axis=0)
            
            self._spatial_index = SpatialIndex(
                np.asarray(self.voxel_centers), 
                use_octree=True
            )
            
            # Use batch neighbor query
            print(f"  Finding neighbors using octree (radius={search_radius:.3f}\")...")
            self.voxel_neighbors[:] = self._spatial_index.get_neighbors_batch(
                search_radius, max_neighbors=26
            )
            
            print(f"  Octree neighbor search complete")
            
        else:
            # Standard KD-tree approach
            print(f"  Building KD-tree for neighbor search...")
            tree = cKDTree(self.voxel_centers)
            
            # Pre-compute arrays
            voxel_centers_array = np.asarray(self.voxel_centers)
            voxel_sizes_array = np.asarray(self.voxel_sizes)
            
            print(f"  Finding neighbors (2.0x radius for interface connectivity)...")
            BATCH_SIZE = 5000
            n_batches = (n_voxels + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_idx in range(n_batches):
                start = batch_idx * BATCH_SIZE
                end = min(start + BATCH_SIZE, n_voxels)
                
                for i in range(start, end):
                    center = voxel_centers_array[i]
                    size = voxel_sizes_array[i]
                    # Increased from 1.5x to 2.0x to ensure part interface connectivity
                    local_search_radius = size * 2.0
                    
                    neighbor_indices = tree.query_ball_point(center, local_search_radius)
                    neighbor_indices = [idx for idx in neighbor_indices if idx != i]
                    
                    # Store up to 26 neighbors
                    count = min(len(neighbor_indices), 26)
                    if count > 0:
                        self.voxel_neighbors[i, :count] = neighbor_indices[:count]
                
                if (batch_idx + 1) % 10 == 0 or end == n_voxels:
                    print(f"    Processed {end:,}/{n_voxels:,} voxels...", flush=True)
    
    def _identify_surface_voxels(self):
        """Identify voxels on or near the surface"""
        n_voxels = len(self.voxel_centers)
        self.surface_voxels = create_managed_array(
            n_voxels, dtype=np.bool_, name="surface_voxels", manager=self.memory_manager
        )
        
        # Surface voxels have higher surface area relative to volume
        surface_threshold = 0.5  # Arbitrary threshold
        
        for i in range(n_voxels):
            # Check if voxel has exposed surfaces
            if self.voxel_surface_areas[i] > 0:
                self.surface_voxels[i] = True
    
    def assign_part_types(self, parts_data: list):
        """
        Assign part types to voxels based on which part mesh they belong to.
        
        Args:
            parts_data: List of tuples (trimesh_obj, part_type_int) where:
                       - trimesh_obj: The part's transformed trimesh
                       - part_type_int: 0=CAST, 1=RISER, 2=CHILL, 3=RISER_POST
        
        This must be called after generate() completes.
        """
        if self.voxel_centers is None:
            print("Warning: No voxels to assign part types to")
            return
        
        n_voxels = len(self.voxel_centers)
        
        # Initialize all voxels as CAST (0) by default
        self.voxel_part_types = create_managed_array(
            n_voxels, dtype=np.int8, name="voxel_part_types", manager=self.memory_manager
        )
        self.voxel_part_types[:] = 0  # Default to CAST
        
        if not parts_data:
            print("  No part type data provided - all voxels set to CAST")
            return
        
        # Count parts by type
        cast_parts = [(m, t) for m, t in parts_data if t == 0 and m is not None]
        riser_parts = [(m, t) for m, t in parts_data if t == 1 and m is not None]
        chill_parts = [(m, t) for m, t in parts_data if t == 2 and m is not None]
        riser_post_parts = [(m, t) for m, t in parts_data if t == 3 and m is not None]
        
        # FAST PATH: If only one CAST part and no risers/chills/posts, skip containment check
        # All voxels were already determined to be inside during voxelization
        if len(cast_parts) <= 1 and len(riser_parts) == 0 and len(chill_parts) == 0 and len(riser_post_parts) == 0:
            print(f"  Single CAST part - all {n_voxels:,} voxels assigned to CAST (fast path)")
            return
        
        print(f"\n  Assigning part types to {n_voxels:,} voxels...")
        print(f"    Parts: {len(cast_parts)} CAST, {len(riser_parts)} RISER, {len(chill_parts)} CHILL, {len(riser_post_parts)} RISER_POST")
        
        # Only check containment for RISER, CHILL, and RISER_POST parts (they override CAST)
        # CAST voxels are already the default, no need to re-check
        voxel_centers_array = np.asarray(self.voxel_centers)
        
        # Process parts with bounding box pre-filtering for speed
        all_parts = riser_parts + chill_parts + riser_post_parts
        total_parts = len(all_parts)
        
        # Tolerance for proximity-based containment (for touching assemblies)
        # Use half voxel size to catch voxels at touching surfaces
        proximity_tolerance = self.base_voxel_size * 0.5
        
        for idx, (mesh, part_type) in enumerate(all_parts):
            type_name = {1: "RISER", 2: "CHILL", 3: "RISER_POST"}.get(part_type, "UNKNOWN")
            print(f"    [{idx+1}/{total_parts}] Checking {type_name} part...", end=" ", flush=True)
            
            # OPTIMIZATION: Use bounding box to pre-filter voxels
            # Expand bbox slightly to catch voxels at touching surfaces
            bbox_min = mesh.bounds[0] - proximity_tolerance
            bbox_max = mesh.bounds[1] + proximity_tolerance
            
            # Find voxels within expanded bounding box (fast NumPy operation)
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
            
            # Process containment check with batched progress feedback.
            # In parallel mode we use worker processes (not threads) for safety.
            BATCH_SIZE = self.part_assignment_batch_size
            inside_global_indices = []
            
            if n_candidates > BATCH_SIZE:
                print(f"({n_candidates:,} candidates, {len(mesh.faces):,} faces)", flush=True)
                use_parallel = self._should_parallelize_part_assignment(n_candidates)
                if use_parallel:
                    inside_global_indices = self._run_part_assignment_parallel(
                        mesh=mesh,
                        candidate_indices=candidate_indices,
                        voxel_centers_array=voxel_centers_array,
                        part_idx=idx,
                        total_parts=total_parts,
                        type_name=type_name,
                        tolerance=proximity_tolerance,
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
                    )

                inside_global_indices = np.array(inside_global_indices) if inside_global_indices else np.array([], dtype=int)
                print(f"\r    [{idx+1}/{total_parts}] Checking {type_name} part... {len(inside_global_indices):,} voxels ({n_candidates:,} candidates)   ")
            else:
                # Small batch - process all at once
                try:
                    candidate_centers = voxel_centers_array[candidate_indices]
                    
                    # Primary check: strict containment with cross-platform fallbacks
                    inside_mask = self._robust_contains_check(mesh, candidate_centers, proximity_tolerance)
                    
                    inside_global_indices = candidate_indices[inside_mask]
                    print(f"{len(inside_global_indices):,} voxels ({n_candidates:,} candidates)")
                except Exception as e:
                    logger.error(f"Part type assignment error: {e}")
                    print(f"error: {e}")
                    inside_global_indices = np.array([], dtype=int)
            
            # Assign part types (RISER/CHILL/RISER_POST override CAST)
            if len(inside_global_indices) > 0:
                current_types = self.voxel_part_types[inside_global_indices]
                override_mask = part_type >= current_types
                if np.any(override_mask):
                    self.voxel_part_types[inside_global_indices[override_mask]] = part_type
        
        # Count voxels by type
        cast_count = np.sum(self.voxel_part_types == 0)
        riser_count = np.sum(self.voxel_part_types == 1)
        chill_count = np.sum(self.voxel_part_types == 2)
        riser_post_count = np.sum(self.voxel_part_types == 3)
        print(f"    Part types assigned: CAST={cast_count:,}, RISER={riser_count:,}, CHILL={chill_count:,}, RISER_POST={riser_post_count:,}")
        
        # Warn if expected parts have no voxels (likely containment check failure)
        if len(riser_parts) > 0 and riser_count == 0:
            print(f"    ⚠️  WARNING: {len(riser_parts)} RISER parts loaded but 0 voxels assigned!")
            print(f"       This may indicate containment check issues. Check STL mesh quality.")
            logger.warning(f"RISER parts loaded ({len(riser_parts)}) but no voxels assigned - containment check may have failed")
        
        if len(chill_parts) > 0 and chill_count == 0:
            print(f"    ⚠️  WARNING: {len(chill_parts)} CHILL parts loaded but 0 voxels assigned!")
            print(f"       This may indicate containment check issues. Check STL mesh quality.")
            print(f"       Chills must be watertight meshes for containment detection.")
            logger.warning(f"CHILL parts loaded ({len(chill_parts)}) but no voxels assigned - containment check may have failed")
        
        if len(riser_post_parts) > 0 and riser_post_count == 0:
            print(f"    ⚠️  WARNING: {len(riser_post_parts)} RISER_POST parts loaded but 0 voxels assigned!")
            logger.warning(f"RISER_POST parts loaded ({len(riser_post_parts)}) but no voxels assigned")
        
        # Bridge interfaces between different part types
        if riser_count > 0 or chill_count > 0 or riser_post_count > 0:
            self._bridge_part_interfaces()
    
    def _bridge_part_interfaces(self):
        """
        Ensure thermal connectivity between different part types at their interfaces.
        
        When separate STL files are used for CAST, RISER, and CHILL parts, there may be
        small gaps between them that prevent voxels from being neighbors. This method
        finds voxels at part type boundaries and ensures they're connected as neighbors.
        
        This is critical for:
        - RISER feeding the CAST (risers need to thermally connect to casting)
        - CHILL accelerating solidification (chills need to contact the casting)
        """
        if self.voxel_part_types is None or self.voxel_neighbors is None:
            return
        
        n_voxels = len(self.voxel_centers)
        voxel_centers_array = np.asarray(self.voxel_centers)
        voxel_sizes_array = np.asarray(self.voxel_sizes)
        
        # Find voxels of each type
        cast_indices = np.where(self.voxel_part_types == 0)[0]
        riser_indices = np.where(self.voxel_part_types == 1)[0]
        chill_indices = np.where(self.voxel_part_types == 2)[0]
        riser_post_indices = np.where(self.voxel_part_types == 3)[0]
        
        bridges_added = 0
        
        # Bridge CAST-RISER interfaces
        if len(cast_indices) > 0 and len(riser_indices) > 0:
            bridges_added += self._bridge_two_types(
                cast_indices, riser_indices, voxel_centers_array, voxel_sizes_array, "CAST-RISER"
            )
        
        # Bridge CAST-CHILL interfaces
        if len(cast_indices) > 0 and len(chill_indices) > 0:
            bridges_added += self._bridge_two_types(
                cast_indices, chill_indices, voxel_centers_array, voxel_sizes_array, "CAST-CHILL"
            )
        
        # Bridge CAST-RISER_POST interfaces (post connects casting to riser)
        if len(cast_indices) > 0 and len(riser_post_indices) > 0:
            bridges_added += self._bridge_two_types(
                cast_indices, riser_post_indices, voxel_centers_array, voxel_sizes_array, "CAST-RISER_POST"
            )
        
        # Bridge RISER_POST-RISER interfaces (post connects to the riser body)
        if len(riser_post_indices) > 0 and len(riser_indices) > 0:
            bridges_added += self._bridge_two_types(
                riser_post_indices, riser_indices, voxel_centers_array, voxel_sizes_array, "RISER_POST-RISER"
            )
        
        if bridges_added > 0:
            print(f"    Interface bridges added: {bridges_added:,} new neighbor connections")
    
    def _bridge_two_types(self, type1_indices: np.ndarray, type2_indices: np.ndarray,
                          voxel_centers: np.ndarray, voxel_sizes: np.ndarray,
                          interface_name: str) -> int:
        """
        Bridge neighbors between two part types.
        
        Uses a larger search radius (3x voxel size) to find voxels at interfaces
        that should be thermally connected but might have small gaps.
        
        Returns:
            Number of new neighbor connections added
        """
        # Build KD-tree for type2 voxels
        type2_centers = voxel_centers[type2_indices]
        type2_tree = cKDTree(type2_centers)
        
        bridges_added = 0
        interface_search_radius = self.base_voxel_size * 3.0  # Extended radius for interfaces
        
        # For each type1 voxel, check if there are nearby type2 voxels not yet connected
        for idx1 in type1_indices:
            center1 = voxel_centers[idx1]
            
            # Find type2 voxels within interface radius
            nearby_local = type2_tree.query_ball_point(center1, interface_search_radius)
            
            if len(nearby_local) == 0:
                continue
            
            # Convert local indices back to global indices
            nearby_global = type2_indices[nearby_local]
            
            # Check current neighbors of this voxel
            current_neighbors = set(self.voxel_neighbors[idx1])
            current_neighbors.discard(-1)
            
            # Find first empty neighbor slot
            for nearby_idx in nearby_global:
                if nearby_idx not in current_neighbors:
                    # Add as neighbor in first empty slot
                    for slot in range(26):
                        if self.voxel_neighbors[idx1, slot] == -1:
                            self.voxel_neighbors[idx1, slot] = nearby_idx
                            bridges_added += 1
                            break
                    
                    # Also add reverse connection
                    for slot in range(26):
                        if self.voxel_neighbors[nearby_idx, slot] == -1:
                            self.voxel_neighbors[nearby_idx, slot] = idx1
                            break
                        elif self.voxel_neighbors[nearby_idx, slot] == idx1:
                            break  # Already connected
        
        if bridges_added > 0:
            print(f"    {interface_name} interface: {bridges_added:,} connections")
        
        return bridges_added
    
    def _robust_contains_check(self, mesh: trimesh.Trimesh, points: np.ndarray, 
                               tolerance: float) -> np.ndarray:
        """
        Cross-platform robust containment check with multiple fallback strategies.
        
        On Windows with certain Python/trimesh versions, mesh.contains() can fail
        or return incorrect results. This method tries multiple approaches:
        
        1. Primary: mesh.contains() (fast, uses ray casting)
        2. Fallback 1: Signed distance (negative = inside)
        3. Fallback 2: Proximity to surface (for non-watertight meshes)
        
        Args:
            mesh: The trimesh object to check containment in
            points: Array of (N, 3) points to check
            tolerance: Distance tolerance for proximity-based containment
        
        Returns:
            Boolean array of length N indicating containment
        """
        return _robust_contains_check_impl(mesh, points, tolerance)

    def _get_part_assignment_workers(self) -> int:
        """Resolve worker count for part assignment.

        Worker count is bounded by both the CPU budget (physical cores - 1)
        and a memory budget (assume ~1.5 GB of working set per worker for
        the trimesh mesh payload and numpy scratch). The lower of the two
        wins, so 8 GB machines do not over-spawn workers and start
        swapping.
        """
        if self.part_assignment_workers > 0:
            return max(1, self.part_assignment_workers)
        cpu_budget = hardware.get_optimal_thread_count('compute')
        mem_budget = max(
            1,
            int(hardware.available_memory // self.AUTO_PART_ASSIGNMENT_MEMORY_PER_WORKER_BYTES),
        )
        return max(1, min(cpu_budget, mem_budget))

    def _should_parallelize_part_assignment(self, n_candidates: int) -> bool:
        """Decide if this part should use process-based parallel containment checks."""
        if self.part_assignment_mode == "legacy":
            return False
        workers = self._get_part_assignment_workers()
        if workers <= 1:
            return False
        if self.part_assignment_mode == "parallel":
            return True
        # auto mode: avoid process overhead and worker data copies for medium jobs
        return n_candidates >= max(
            self.AUTO_PART_ASSIGNMENT_MIN_CANDIDATES,
            self.part_assignment_batch_size * workers * self.PART_ASSIGNMENT_SUBMIT_MULTIPLIER,
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
    ) -> List[int]:
        """Legacy sequential batched containment processing."""
        inside_global_indices: List[int] = []
        n_candidates = len(candidate_indices)
        n_batches = (n_candidates + self.part_assignment_batch_size - 1) // self.part_assignment_batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * self.part_assignment_batch_size
            end = min(start + self.part_assignment_batch_size, n_candidates)
            batch_indices = candidate_indices[start:end]
            batch_centers = voxel_centers_array[batch_indices]

            try:
                inside_mask = self._robust_contains_check(mesh, batch_centers, tolerance)
                batch_inside = batch_indices[inside_mask]
                inside_global_indices.extend(batch_inside)
            except Exception as e:
                logger.warning(f"Part type batch processing error: {e}")
                print(f" [batch error: {e}]", end="", flush=True)

            pct = int((batch_idx + 1) / n_batches * 100)
            print(
                f"\r    [{part_idx+1}/{total_parts}] Checking {type_name} part... {pct}% ({len(inside_global_indices):,} inside)",
                end="",
                flush=True,
            )

        return inside_global_indices

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
    ) -> List[int]:
        """Process part assignment containment checks in parallel worker processes."""
        try:
            workers = self._get_part_assignment_workers()
            n_candidates = len(candidate_indices)
            n_batches = (n_candidates + self.part_assignment_batch_size - 1) // self.part_assignment_batch_size
            mesh_payload = {
                "vertices": np.asarray(mesh.vertices),
                "faces": np.asarray(mesh.faces),
            }

            found_by_batch: Dict[int, np.ndarray] = {}
            inside_count = 0
            completed = 0

            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_part_assignment_worker,
                initargs=(mesh_payload,),
            ) as executor:
                futures = {}
                next_batch_idx = 0
                max_pending = max(1, workers * self.PART_ASSIGNMENT_SUBMIT_MULTIPLIER)

                def submit_batch(batch_idx: int) -> None:
                    start = batch_idx * self.part_assignment_batch_size
                    end = min(start + self.part_assignment_batch_size, n_candidates)
                    batch_indices = candidate_indices[start:end]
                    batch_centers = voxel_centers_array[batch_indices]
                    future = executor.submit(
                        _part_assignment_contains_worker,
                        batch_idx,
                        batch_indices,
                        batch_centers,
                        tolerance,
                    )
                    futures[future] = batch_idx

                while next_batch_idx < n_batches and len(futures) < max_pending:
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

                        if next_batch_idx < n_batches:
                            submit_batch(next_batch_idx)
                            next_batch_idx += 1

            ordered = [found_by_batch[i] for i in sorted(found_by_batch)]
            if not ordered:
                return []
            return list(np.concatenate(ordered))
        except Exception as e:
            logger.warning(f"Parallel part assignment failed; falling back to sequential: {e}")
            print(f"\n      Parallel fallback: {e}")
            return self._run_part_assignment_sequential(
                mesh=mesh,
                candidate_indices=candidate_indices,
                voxel_centers_array=voxel_centers_array,
                part_idx=part_idx,
                total_parts=total_parts,
                type_name=type_name,
                tolerance=tolerance,
            )
    
    def get_voxels_by_type(self, part_type: int) -> np.ndarray:
        """
        Get indices of voxels belonging to a specific part type.
        
        Args:
            part_type: 0=CAST, 1=RISER, 2=CHILL
        
        Returns:
            Array of voxel indices
        """
        if self.voxel_part_types is None:
            return np.array([], dtype=int)
        return np.where(self.voxel_part_types == part_type)[0]
    
    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage in MB (includes both RAM and disk-backed arrays)"""
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
        
        # Add part types if present
        if self.voxel_part_types is not None:
            total_bytes += self.voxel_part_types.nbytes
        
        return total_bytes / (1024 ** 2)
    
    def get_memory_stats(self) -> dict:
        """Get detailed memory statistics"""
        return self.memory_manager.get_memory_stats()
    
    def print_memory_stats(self):
        """Print memory statistics"""
        self.memory_manager.print_stats()
    
    def get_voxel_count(self) -> int:
        """Get total voxel count"""
        return len(self.voxel_centers) if self.voxel_centers is not None else 0
    
    def get_surface_voxel_indices(self) -> np.ndarray:
        """Get indices of surface voxels"""
        if self.surface_voxels is None:
            return np.array([], dtype=int)
        return np.where(self.surface_voxels)[0]
    
    def cleanup(self):
        """Clean up memory-mapped files"""
        self.memory_manager.cleanup()
