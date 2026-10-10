"""
Hardware Acceleration Layer
Provides unified interface for CPU acceleration (Intel, AMD, Apple Silicon)
Uses physical cores only to minimize memory usage
"""

import numpy as np
from numba import jit, prange, set_num_threads
from joblib import Parallel, delayed
import sys
import multiprocessing
from typing import Callable, Any, List
from config.hardware_config import hardware

try:
    from core.error_logger import logger
except ImportError:
    from core.error_logger_dummy import logger

# Configure Numba to use physical cores only (no hyperthreading)
optimal_threads = hardware.get_optimal_thread_count('compute')
set_num_threads(optimal_threads)
if multiprocessing.current_process().name == "MainProcess":
    print(f"Numba configured to use {optimal_threads} threads (physical cores only)")


class AccelerationManager:
    """Manages compute acceleration across different hardware (CPU-focused)"""
    
    def __init__(self):
        self.gpu_available = False  # GPU disabled for cross-platform compatibility
        self.neural_engine_available = False  # Neural Engine not used for numerical simulation
        self.opencl_context = None
        
        # GPU acceleration is disabled by default for stability and cross-platform support
        # Uncomment below if you want to try GPU acceleration (requires pyopencl)
        # if hardware.should_use_gpu():
        #     self._init_gpu()
    
    def _init_gpu(self):
        """
        Initialize GPU compute context (optional)
        
        Note: GPU acceleration via OpenCL is experimental and may cause stability issues.
        The CPU-based Numba JIT implementation is fast enough for most use cases.
        """
        try:
            import pyopencl as cl
            
            platforms = cl.get_platforms()
            if not platforms:
                print("GPU: No OpenCL platforms found, using CPU only")
                self.gpu_available = False
                return
            
            # Try to find the best GPU device
            gpu_device = None
            for platform in platforms:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    gpu_device = devices[0]
                    self.opencl_context = cl.Context(devices=[gpu_device])
                    print(f"GPU initialized: {gpu_device.name} on {platform.name}")
                    self.gpu_available = True
                    return
            
            print("GPU: No GPU devices found, using CPU only")
            self.gpu_available = False
            
        except ImportError:
            logger.debug("PyOpenCL not installed, using CPU only")
            print("GPU: PyOpenCL not installed, using CPU only")
            print("  To enable GPU: pip install pyopencl")
            self.gpu_available = False
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}")
            print(f"GPU: Initialization failed ({e}), using CPU only")
            self.gpu_available = False
    
    def parallel_map(self, func: Callable, data: List[Any], 
                    n_jobs: int = None, backend: str = 'loky') -> List[Any]:
        """
        Execute function in parallel across CPU cores
        
        Args:
            func: Function to execute
            data: List of inputs
            n_jobs: Number of parallel jobs (None = auto)
            backend: 'loky' (process), 'threading' (thread), or 'multiprocessing'
        
        Returns:
            List of results
        """
        if n_jobs is None:
            n_jobs = hardware.get_optimal_thread_count('compute')
        
        return Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(func)(item) for item in data
        )
    
    def parallel_compute(self, func: Callable, *args, n_jobs: int = None, **kwargs):
        """Execute parallel computation with progress tracking"""
        if n_jobs is None:
            n_jobs = hardware.get_optimal_thread_count('compute')
        
        return func(*args, n_jobs=n_jobs, **kwargs)
    
    def gpu_compute_modulus(self, volumes: np.ndarray, areas: np.ndarray) -> np.ndarray:
        """
        Compute modulus (GPU would be used if available, but falls back to CPU)
        
        The CPU implementation using Numba JIT is highly optimized and runs on
        physical cores only to minimize memory usage. Performance is excellent
        even without GPU acceleration.
        """
        # Always use CPU implementation (GPU disabled for stability/compatibility)
        return compute_modulus_parallel(volumes, areas)


# JIT-compiled functions for performance-critical operations
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_modulus_parallel(volumes: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """
    Compute geometric modulus for voxels (parallel)
    Modulus = Volume / Surface Area (in inches)
    """
    n_voxels = volumes.shape[0]
    moduli = np.empty(n_voxels, dtype=np.float32)

    for i in prange(n_voxels):
        if areas[i] > 0:
            moduli[i] = volumes[i] / areas[i]
        else:
            moduli[i] = 0.0

    return moduli


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def update_solidification_with_part_types_parallel(
    voxel_states: np.ndarray,
    neighbors: np.ndarray,
    moduli: np.ndarray,
    heights: np.ndarray,
    chill_boost: np.ndarray,
    riser_slowdown: np.ndarray,
    time_step: float,
    material_factor: float = 1.0
) -> np.ndarray:
    """
    Update solidification front with chill/riser/material effects (JIT-optimized)
    
    CRITICAL: Voxels with low riser_slowdown (< 0.5) are in the "feeding zone"
    and should only solidify when their riser connection path has solidified.
    This prevents isolated liquid pools.
    
    Args:
        voxel_states: Current solidification state (0=liquid, 1=solid)
        neighbors: Neighbor connectivity matrix (n_voxels x 26)
        moduli: Geometric modulus for each voxel
        heights: Normalized vertical position (0=bottom, 1=top)
        chill_boost: Multiplier for chill acceleration (>1 = faster)
        riser_slowdown: Multiplier for riser exothermic effect (<1 = slower)
        time_step: Current time step
        material_factor: Material-specific solidification rate multiplier
    
    Returns:
        Updated voxel states
    """
    n_voxels = voxel_states.shape[0]
    new_states = voxel_states.copy()
    
    # Threshold for "in feeding zone" - voxels with low slowdown are fed by risers
    feeding_zone_threshold = 0.6
    
    for i in prange(n_voxels):
        if voxel_states[i] < 1.0:  # Not fully solid
            # Check if this voxel is in the riser feeding zone
            in_feeding_zone = riser_slowdown[i] < feeding_zone_threshold
            
            # Check neighbors for solid contact
            has_solid_neighbor = False
            has_liquid_feeding_neighbor = False
            
            for j in range(neighbors.shape[1]):
                neighbor_idx = neighbors[i, j]
                if neighbor_idx >= 0:
                    if voxel_states[neighbor_idx] >= 1.0:
                        has_solid_neighbor = True
                    elif voxel_states[neighbor_idx] < 1.0 and riser_slowdown[neighbor_idx] < feeding_zone_threshold:
                        # This neighbor is liquid AND in the feeding zone
                        has_liquid_feeding_neighbor = True
            
            # CRITICAL LOGIC for feeding connectivity:
            # If this voxel is in the feeding zone AND has liquid neighbors also in feeding zone,
            # it stays liquid longer (directional solidification toward riser)
            if in_feeding_zone and has_liquid_feeding_neighbor:
                # Suppress solidification - connected to riser through liquid
                # Use moderate reduction to maintain feeding path while still solidifying
                effective_slowdown = riser_slowdown[i] * 0.6  # 40% reduction (was 70%)
            else:
                effective_slowdown = riser_slowdown[i]
            
            if has_solid_neighbor:
                # Base solidification rate based on modulus
                base_rate = 0.02
                solidification_rate = base_rate / (1.0 + moduli[i] * moduli[i] * 2.0)
                
                # Vertical factor: bottom solidifies faster
                vertical_factor = 1.0 + (1.0 - heights[i]) * 0.3
                
                # Apply material thermal properties factor
                thermal_factor = material_factor
                
                # Apply chill boost (faster) and riser slowdown (slower)
                part_type_factor = chill_boost[i] * effective_slowdown
                
                # Update state with all factors
                delta = solidification_rate * vertical_factor * thermal_factor * part_type_factor * time_step
                new_states[i] = min(voxel_states[i] + delta, 1.0)
    
    return new_states


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def update_solidification_with_feeding_connectivity(
    voxel_states: np.ndarray,
    neighbors: np.ndarray,
    moduli: np.ndarray,
    heights: np.ndarray,
    chill_boost: np.ndarray,
    riser_slowdown: np.ndarray,
    part_types: np.ndarray,
    time_step: float,
    material_factor: float = 1.0,
    feeding_zone_vertical_bias: float = 0.5
) -> np.ndarray:
    """
    Update solidification with RISER connectivity-based feeding zone detection.

    CRITICAL FIX: This function traces feeding connectivity through part_types,
    ensuring that voxels connected to RISER voxels (part_type=1) through liquid
    neighbors maintain feeding zone status. This prevents "islands" of molten
    metal at riser post junctions from solidifying prematurely.

    Feeding Zone Rules:
    1. All RISER voxels (part_type=1) are in the feeding zone
    2. Any liquid CAST voxel with a liquid RISER neighbor is in the feeding zone
    3. Any liquid CAST voxel with a liquid feeding-zone neighbor propagates status

    This implements neighbor-propagated directional solidification toward risers.

    Args:
        voxel_states: Current solidification state (0=liquid, 1=solid)
        neighbors: Neighbor connectivity matrix (n_voxels x 26)
        moduli: Geometric modulus for each voxel
        heights: Normalized vertical position (0=bottom, 1=top)
        chill_boost: Multiplier for chill acceleration (>1 = faster)
        riser_slowdown: Multiplier for riser exothermic effect (<1 = slower)
        part_types: Part type for each voxel (0=CAST, 1=RISER, 2=CHILL)
        time_step: Current time step
        material_factor: Material-specific solidification rate multiplier
        feeding_zone_vertical_bias: Strength of +Z directional-solidification bias
            inside the riser feeding zone. Effective vertical_factor = 1.0 +
            (1.0 - height) * feeding_zone_vertical_bias.

    Returns:
        Updated voxel states
    """
    n_voxels = voxel_states.shape[0]
    new_states = voxel_states.copy()
    
    # Part type constants
    CAST = 0
    RISER = 1
    CHILL = 2
    
    # Feeding zone threshold based on riser_slowdown values
    # Voxels with low slowdown are close to risers
    feeding_zone_threshold = 0.6
    
    for i in prange(n_voxels):
        if voxel_states[i] < 1.0:  # Not fully solid
            # STEP 1: Determine if this voxel is in the feeding zone
            # 
            # A voxel is in the feeding zone if:
            # - It IS a RISER voxel (part_type == 1), OR
            # - It has a liquid RISER neighbor (direct connection), OR
            # - It has a liquid neighbor that's in the feeding zone (riser_slowdown based), OR
            # - It has low riser_slowdown (distance-based fallback)
            
            is_riser_voxel = part_types[i] == RISER
            has_liquid_riser_neighbor = False
            has_liquid_feeding_neighbor = False
            has_solid_neighbor = False
            
            # Check all neighbors
            for j in range(neighbors.shape[1]):
                neighbor_idx = neighbors[i, j]
                if neighbor_idx >= 0:
                    neighbor_state = voxel_states[neighbor_idx]
                    neighbor_part_type = part_types[neighbor_idx]
                    neighbor_slowdown = riser_slowdown[neighbor_idx]
                    
                    if neighbor_state >= 1.0:
                        # Solid neighbor - triggers solidification
                        has_solid_neighbor = True
                    else:
                        # Liquid neighbor - check for feeding connectivity
                        if neighbor_part_type == RISER:
                            # CRITICAL: Direct connection to liquid riser
                            has_liquid_riser_neighbor = True
                        elif neighbor_slowdown < feeding_zone_threshold:
                            # Neighbor is in feeding zone (propagated connectivity)
                            has_liquid_feeding_neighbor = True
            
            # Determine effective feeding zone status
            in_feeding_zone = (
                is_riser_voxel or 
                has_liquid_riser_neighbor or 
                (riser_slowdown[i] < feeding_zone_threshold and has_liquid_feeding_neighbor)
            )
            
            # STEP 2: Calculate effective slowdown based on feeding connectivity
            if in_feeding_zone:
                if is_riser_voxel:
                    # Riser voxels themselves use full exothermic effect
                    effective_slowdown = riser_slowdown[i]
                elif has_liquid_riser_neighbor:
                    # CRITICAL: Direct riser connection - strong feeding suppression
                    # These are the voxels at the CAST-RISER interface (riser posts)
                    # They should solidify LAST, feeding into the riser
                    effective_slowdown = riser_slowdown[i] * 0.4  # 60% reduction
                else:
                    # Propagated feeding zone - moderate suppression
                    effective_slowdown = riser_slowdown[i] * 0.6  # 40% reduction
            else:
                # Not in feeding zone - normal solidification
                effective_slowdown = riser_slowdown[i]
            
            # STEP 3: Apply solidification if has solid neighbor
            if has_solid_neighbor:
                # Base solidification rate based on modulus
                base_rate = 0.02
                solidification_rate = base_rate / (1.0 + moduli[i] * moduli[i] * 2.0)
                
                # Vertical factor: bottom solidifies faster (directional solidification)
                # ENHANCED: Feeding zone voxels have stronger upward bias
                if in_feeding_zone:
                    # In feeding zone: even stronger upward bias (solidifies from bottom up)
                    vertical_factor = 1.0 + (1.0 - heights[i]) * feeding_zone_vertical_bias
                else:
                    # Normal: moderate upward bias
                    vertical_factor = 1.0 + (1.0 - heights[i]) * 0.3
                
                # Apply material thermal properties factor
                thermal_factor = material_factor
                
                # Apply chill boost (faster) and effective slowdown (slower)
                part_type_factor = chill_boost[i] * effective_slowdown
                
                # Update state with all factors
                delta = solidification_rate * vertical_factor * thermal_factor * part_type_factor * time_step
                new_states[i] = min(voxel_states[i] + delta, 1.0)
    
    return new_states


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_min_distances_parallel(
    points: np.ndarray,
    reference_points: np.ndarray
) -> np.ndarray:
    """
    Compute minimum distance from each point to any reference point (JIT-optimized)
    
    This is much faster than Python loops for distance calculations.
    
    Args:
        points: Array of points (n_points x 3)
        reference_points: Array of reference points (n_ref x 3)
    
    Returns:
        Array of minimum distances (n_points,)
    """
    n_points = points.shape[0]
    n_ref = reference_points.shape[0]
    min_distances = np.empty(n_points, dtype=np.float32)
    
    for i in prange(n_points):
        min_dist = np.float32(1e30)
        px, py, pz = points[i, 0], points[i, 1], points[i, 2]
        
        for j in range(n_ref):
            dx = px - reference_points[j, 0]
            dy = py - reference_points[j, 1]
            dz = pz - reference_points[j, 2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < min_dist:
                min_dist = dist
        
        min_distances[i] = min_dist
    
    return min_distances


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_chill_internal_thickness_parallel(
    chill_centers: np.ndarray,
    chill_surface_centers: np.ndarray
) -> np.ndarray:
    """
    Compute spatially-varying thickness of chill part using point-based EDT.
    
    Thickness at point P = 2 * distance to nearest chill surface point.
    
    Args:
        chill_centers: Centers of all chill voxels
        chill_surface_centers: Centers of chill voxels on the boundary
        
    Returns:
        Array of thickness values for each chill voxel
    """
    n_chills = chill_centers.shape[0]
    n_surface = chill_surface_centers.shape[0]
    thickness = np.zeros(n_chills, dtype=np.float32)
    
    if n_surface == 0:
        return thickness
    
    for i in prange(n_chills):
        px, py, pz = chill_centers[i, 0], chill_centers[i, 1], chill_centers[i, 2]
        min_dist_sq = np.float32(1e30)
        
        for j in range(n_surface):
            dx = px - chill_surface_centers[j, 0]
            dy = py - chill_surface_centers[j, 1]
            dz = pz - chill_surface_centers[j, 2]
            dist_sq = dx*dx + dy*dy + dz*dz
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        
        thickness[i] = 2.0 * np.sqrt(min_dist_sq)
        
    return thickness


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_dynamic_chill_boost_parallel(
    voxel_centers: np.ndarray,
    chill_centers: np.ndarray,
    chill_thicknesses: np.ndarray,
    voxel_states: np.ndarray,
    part_types: np.ndarray,
    current_time: float,
    max_chill_boost: float,
    air_gap_time_constant: float = 120.0,
    saturation_threshold: float = 0.8
) -> np.ndarray:
    """
    Compute dynamic chill boost with Air Gap and Saturation effects.
    
    Air Gap: Boost decays exponentially over time as metal shrinks away.
    Saturation: Boost decays as local casting metal solidifies (chill heats up).
    Spacial: Influence radius depends on local chill thickness.
    
    Args:
        voxel_centers: All voxel centers (n_voxels x 3)
        chill_centers: Chill voxel centers (n_chills x 3)
        chill_thicknesses: Spherically-symmetric thickness at each chill voxel
        voxel_states: Solidification state (0=liquid, 1=solid)
        part_types: 0=CAST, 1=RISER, 2=CHILL
        current_time: Time in seconds
        max_chill_boost: Initial maximum boost factor (e.g., 3.0)
        air_gap_time_constant: Characteristic time for air gap formation (s)
        saturation_threshold: Progress at which chill is considered saturated
        
    Returns:
        Dynamic boost multipliers
    """
    n_voxels = voxel_centers.shape[0]
    n_chills = chill_centers.shape[0]
    chill_boost = np.ones(n_voxels, dtype=np.float32)
    
    if n_chills == 0:
        return chill_boost
        
    # Time-based decay (Air Gap Factor)
    # Starts at 1.0 at t=0, decays toward 0.0 (representing 50% loss of effectiveness)
    # Boost formula: 1.0 + (max - 1.0) * exp(-t/tau)
    time_decay = np.exp(-current_time / air_gap_time_constant)
    
    for i in prange(n_voxels):
        # Only CAST and RISER voxels receive boost
        if part_types[i] == 2:
            continue
            
        px, py, pz = voxel_centers[i, 0], voxel_centers[i, 1], voxel_centers[i, 2]
        
        # Find nearest chill and its properties
        min_dist_sq = np.float32(1e30)
        nearest_chill_idx = -1
        
        for j in range(n_chills):
            dx = px - chill_centers[j, 0]
            dy = py - chill_centers[j, 1]
            dz = pz - chill_centers[j, 2]
            dist_sq = dx*dx + dy*dy + dz*dz
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_chill_idx = j
        
        if nearest_chill_idx >= 0:
            min_dist = np.sqrt(min_dist_sq)
            local_chill_thickness = chill_thicknesses[nearest_chill_idx]
            
            # Use local chill thickness as the influence radius
            # (Rule of thumb: depth of effect = thickness of chill)
            influence_radius = max(local_chill_thickness, 0.1) # Min 0.1"
            
            if min_dist < influence_radius:
                # 1. Spacial Falloff (Linear)
                spacial_influence = 1.0 - (min_dist / influence_radius)
                
                # 2. Saturation Factor
                # As this voxel solidifies, the chill heats up and becomes less effective.
                # If voxel_state is 0.0 (liquid), saturation_factor is 1.0.
                # As it approach 1.0 (solid), it drops.
                saturation_factor = max(1.0 - (voxel_states[i] / saturation_threshold), 0.0)
                
                # Combine all factors
                # We want the effect to never drop below 1.5x (baseline graphite effectiveness even after gap)
                # or actually, we can let it drop to 1.0 if it's completely solidified.
                effective_max_boost = 1.0 + (max_chill_boost - 1.0) * time_decay * saturation_factor
                
                chill_boost[i] = 1.0 + (effective_max_boost - 1.0) * spacial_influence
                
    return chill_boost


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_riser_influence_parallel(
    riser_slowdown: np.ndarray,
    cast_indices: np.ndarray,
    distances: np.ndarray,
    riser_influence_radius: float,
    exothermic_factor: float,
    min_influence_factor: float
) -> int:
    """
    Compute riser influence on cast voxels with smooth falloff (JIT-optimized).
    
    This replaces the slow Python loop for applying riser influence to casting voxels.
    Modifies riser_slowdown in-place for affected voxels.
    
    Args:
        riser_slowdown: Array to modify in-place (n_voxels,)
        cast_indices: Indices of CAST voxels
        distances: Pre-computed distances from each cast voxel to nearest riser
        riser_influence_radius: Maximum distance of riser feeding influence (inches)
        exothermic_factor: Slowdown factor at riser contact
        min_influence_factor: Minimum slowdown at edge of influence radius
        
    Returns:
        Count of influenced voxels
    """
    n_cast = cast_indices.shape[0]
    influenced_count = 0
    inner_radius = riser_influence_radius * 0.4
    
    for i in prange(n_cast):
        dist = distances[i]
        cast_idx = cast_indices[i]
        
        if dist < riser_influence_radius:
            # Quadratic falloff - strong effect near riser
            normalized_dist = dist / riser_influence_radius
            influence = (1.0 - normalized_dist) ** 2
            
            # Blend from min_influence_factor to exothermic_factor
            slowdown = min_influence_factor + (exothermic_factor - min_influence_factor) * influence
            
            # Moderate additional effect within inner 40% of feeding distance
            # This creates directional solidification toward risers
            if dist < inner_radius:
                close_blend = 1.0 - (dist / inner_radius)
                # Moderate blend toward riser behavior (40% blend)
                slowdown = slowdown * (1.0 - close_blend * 0.4) + exothermic_factor * (close_blend * 0.4)
            
            # Only update if this makes it slower (min of current and new)
            if slowdown < riser_slowdown[cast_idx]:
                riser_slowdown[cast_idx] = slowdown
            
            influenced_count += 1
    
    return influenced_count


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_mold_air_gap_effect_parallel(
    voxel_states: np.ndarray,
    feeding_distances: np.ndarray,
    surface_mask: np.ndarray,
    current_time: float,
    air_gap_time_constant: float = 60.0,
    surface_influence_depth: float = 0.5,
    max_slowdown: float = 0.75
) -> np.ndarray:
    """
    Compute mold air gap effect on all surface and near-surface voxels.
    
    Physical Model:
    As metal solidifies, it shrinks and pulls away from the sand mold wall,
    creating a microscopic air gap. Since air is a great insulator, this
    GAP SLOWS heat extraction from the casting.
    
    Effect characteristics:
    - Only affects surface and near-surface voxels (within surface_influence_depth)
    - Effect increases with TIME (gap grows as more metal solidifies)
    - Effect increases with local SOLIDIFICATION state (gap proportional to shrinkage)
    - Returns slowdown multipliers (<1 means slower solidification)
    
    Args:
        voxel_states: Current solidification state (0=liquid, 1=solid)
        feeding_distances: Distance from surface for each voxel (inches)
        surface_mask: Boolean mask of surface voxels
        current_time: Simulation time in seconds
        air_gap_time_constant: Time constant for gap formation (seconds)
        surface_influence_depth: How deep from surface the effect extends (inches)
        max_slowdown: Maximum slowdown factor (0.75 = 25% slower at full effect)
        
    Returns:
        Array of slowdown multipliers (0.75-1.0)
    """
    n_voxels = voxel_states.shape[0]
    air_gap_slowdown = np.ones(n_voxels, dtype=np.float32)
    
    # Time-based gap formation factor
    # Gap starts forming immediately and grows over time
    # At t=0, time_factor=0. At t=time_constant, time_factor=0.63. At t=3*time_constant, time_factor=0.95
    time_factor = 1.0 - np.exp(-current_time / air_gap_time_constant)
    
    if time_factor < 0.01:
        # No significant gap yet
        return air_gap_slowdown
    
    for i in prange(n_voxels):
        # Only affects voxels near the surface
        dist_from_surface = feeding_distances[i]
        
        if dist_from_surface < surface_influence_depth:
            # Spatial factor: stronger effect at surface, decreasing with depth
            # At surface (dist=0): spatial_factor = 1.0
            # At depth = surface_influence_depth: spatial_factor = 0.0
            spatial_factor = 1.0 - (dist_from_surface / surface_influence_depth)
            
            # Solidification factor: gap proportional to local solidification
            # Only solid metal can shrink and form a gap
            # At liquid (state=0): solidification_factor = 0.0
            # At solid (state=1): solidification_factor = 1.0
            solidification_factor = voxel_states[i]
            
            # Combined effect
            # gap_effect ranges from 0 (no gap) to 1 (full gap)
            gap_effect = time_factor * spatial_factor * solidification_factor
            
            # Convert to slowdown multiplier
            # max_slowdown = 0.75 means 25% slower at full gap effect
            air_gap_slowdown[i] = 1.0 - (1.0 - max_slowdown) * gap_effect
    
    return air_gap_slowdown


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_modulus_variance_parallel(
    voxel_moduli: np.ndarray,
    neighbor_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute modulus variance among neighbors for each voxel (JIT-optimized).
    
    High variance indicates section changes where hot tears form preferentially.
    
    Args:
        voxel_moduli: Geometric modulus at each voxel
        neighbor_matrix: Neighbor indices for each voxel (n_voxels x max_neighbors)
    
    Returns:
        Array of modulus variance values
    """
    n_voxels = voxel_moduli.shape[0]
    max_neighbors = neighbor_matrix.shape[1]
    modulus_variance = np.zeros(n_voxels, dtype=np.float32)
    
    for i in prange(n_voxels):
        # Collect neighbor moduli
        neighbor_count = 0
        sum_moduli = 0.0
        sum_sq_moduli = 0.0
        
        for j in range(max_neighbors):
            neighbor_idx = neighbor_matrix[i, j]
            if 0 <= neighbor_idx < n_voxels:
                m = voxel_moduli[neighbor_idx]
                sum_moduli += m
                sum_sq_moduli += m * m
                neighbor_count += 1
        
        if neighbor_count > 1:
            mean = sum_moduli / neighbor_count
            variance = (sum_sq_moduli / neighbor_count) - (mean * mean)
            modulus_variance[i] = np.sqrt(max(variance, 0.0))  # std dev
    
    return modulus_variance


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_constraint_strain_parallel(
    solidification_times: np.ndarray,
    voxel_moduli: np.ndarray,
    neighbor_matrix: np.ndarray,
    thermal_expansion: float,
    temp_range: float
) -> np.ndarray:
    """
    Compute mechanical constraint strain from already-solidified neighbors (JIT-optimized).

    Physical model: When thin sections solidify first, they become rigid and contract.
    Adjacent mushy zones (still solidifying) are mechanically constrained by the solid
    sections and experience tensile strain as the solid contracts but the mush cannot flow.

    Hot tears localize at the most-restraining neighbor, not at the average restraint.
    We therefore return the *worst-case* (max) constraint over all early-solidifying
    neighbors, matching the experimental observation that a single adjacent
    already-solidified thin section can nucleate a hot tear (Braccini et al., 2000).

    Strain accumulation = α × ΔT × (solid_neighbor_contraction / mushy_zone_compliance)

    Args:
        solidification_times: Time when each voxel solidified (seconds)
        voxel_moduli: Geometric modulus (proxy for section thickness) at each voxel
        neighbor_matrix: Neighbor indices for each voxel (n_voxels x max_neighbors)
        thermal_expansion: Thermal expansion coefficient (1/K)
        temp_range: Temperature range used for the strain (°C) — pass the BTR
                    temperature range, not the full freezing range, for physical
                    accuracy (Phase 1.2 fix)

    Returns:
        Array of constraint strain values (dimensionless)
    """
    n_voxels = solidification_times.shape[0]
    max_neighbors = neighbor_matrix.shape[1]
    constraint_strain = np.zeros(n_voxels, dtype=np.float32)

    # Find max solidification time for normalization
    max_time = 0.0
    for i in range(n_voxels):
        if solidification_times[i] > max_time:
            max_time = solidification_times[i]

    if max_time <= 0:
        return constraint_strain

    for i in prange(n_voxels):
        my_time = solidification_times[i]
        my_modulus = voxel_moduli[i]

        # Skip if this voxel solidified first (no constraint from neighbors)
        if my_time <= 0:
            continue

        # Track the worst-case (max) constraint from any early-solidifying neighbor.
        # Hot tears localize at the most-restraining boundary, not at the average.
        max_strain_seen = 0.0
        solid_neighbor_count = 0

        for j in range(max_neighbors):
            neighbor_idx = neighbor_matrix[i, j]
            if 0 <= neighbor_idx < n_voxels:
                neighbor_time = solidification_times[neighbor_idx]
                neighbor_modulus = voxel_moduli[neighbor_idx]

                # Only count neighbors that solidified BEFORE this voxel
                if neighbor_time < my_time and neighbor_time > 0:
                    # Time differential: how much earlier did neighbor solidify
                    time_diff = my_time - neighbor_time
                    time_factor = time_diff / max_time

                    # Modulus ratio: thin sections (low modulus) create more constraint
                    # on thick neighbors (high modulus) because thin solidifies first
                    # and contracts while thick is still mushy
                    if my_modulus > 0:
                        modulus_ratio = neighbor_modulus / my_modulus
                        # Constraint is higher when thin (low modulus) neighbor
                        # constrains thick (high modulus) voxel
                        thickness_factor = 1.0 / (1.0 + modulus_ratio)
                    else:
                        thickness_factor = 0.5

                    # Accumulated contraction strain from this neighbor
                    # ε = α × ΔT × time_factor × thickness_factor
                    neighbor_strain = thermal_expansion * temp_range * time_factor * thickness_factor
                    if neighbor_strain > max_strain_seen:
                        max_strain_seen = neighbor_strain
                    solid_neighbor_count += 1

        if solid_neighbor_count > 0:
            # Worst-case (max) constraint — localizes hot tears correctly
            constraint_strain[i] = max_strain_seen

    return constraint_strain


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_solid_path_strain_parallel(
    solidification_times: np.ndarray,
    voxel_centers: np.ndarray,
    neighbor_matrix: np.ndarray,
    thermal_expansion: float,
    temp_range: float,
    max_path_length: float
) -> np.ndarray:
    """
    Compute accumulated strain along solid paths to mushy zones (JIT-optimized).
    
    Physical model: As thin sections solidify and cool, they contract along their
    length. This contraction is transmitted through connected solid paths to
    adjacent mushy zones. The strain accumulates with path length:
    
    ε_accumulated = α × ΔT × L_solid_path
    
    This captures the "pulling apart" effect where a long thin section that
    solidified early pulls on the mushy zone at its junction with a thick section.
    
    Args:
        solidification_times: Time when each voxel solidified (seconds)
        voxel_centers: 3D coordinates of each voxel center
        neighbor_matrix: Neighbor indices for each voxel (n_voxels x max_neighbors)
        thermal_expansion: Thermal expansion coefficient (1/K)
        temp_range: Temperature range used for the strain (°C) — pass the BTR
                    temperature range, not the full freezing range, for physical
                    accuracy (Phase 1.2 fix)
        max_path_length: Maximum path length to trace (inches)

    Returns:
        Array of path strain values (dimensionless)
    """
    n_voxels = solidification_times.shape[0]
    max_neighbors = neighbor_matrix.shape[1]
    path_strain = np.zeros(n_voxels, dtype=np.float32)
    
    # Find median solidification time to identify "mushy" voxels
    # (those in the middle of solidification sequence)
    max_time = 0.0
    for i in range(n_voxels):
        if solidification_times[i] > max_time:
            max_time = solidification_times[i]
    
    if max_time <= 0:
        return path_strain
    
    # Threshold: voxels that solidified in the early 40% are "solid" when
    # constraining later voxels
    early_solid_threshold = max_time * 0.4
    
    for i in prange(n_voxels):
        my_time = solidification_times[i]
        
        # Only calculate for voxels that solidified in middle-to-late range
        # (these are the ones being constrained by early-solidifying neighbors)
        if my_time < early_solid_threshold:
            continue
        
        # Trace solid paths from this voxel
        # For each neighbor that solidified earlier, measure path length
        max_solid_path = 0.0
        
        for j in range(max_neighbors):
            neighbor_idx = neighbor_matrix[i, j]
            if 0 <= neighbor_idx < n_voxels:
                neighbor_time = solidification_times[neighbor_idx]
                
                # If neighbor solidified significantly earlier, trace the path
                if neighbor_time < my_time * 0.8 and neighbor_time > 0:
                    # Simple path length: distance to this earlier-solid neighbor
                    dx = voxel_centers[i, 0] - voxel_centers[neighbor_idx, 0]
                    dy = voxel_centers[i, 1] - voxel_centers[neighbor_idx, 1]
                    dz = voxel_centers[i, 2] - voxel_centers[neighbor_idx, 2]
                    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    # Weight by how much earlier the neighbor solidified
                    time_weight = (my_time - neighbor_time) / max_time
                    effective_path = dist * (1.0 + time_weight * 2.0)
                    
                    if effective_path > max_solid_path:
                        max_solid_path = effective_path
        
        # Cap at max_path_length
        if max_solid_path > max_path_length:
            max_solid_path = max_path_length
        
        # Accumulated strain along the solid path
        # ε = α × ΔT × L_path (normalized by max_path_length)
        path_strain[i] = thermal_expansion * temp_range * (max_solid_path / max_path_length)

    return path_strain


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_isolation_score_parallel(
    solid_times: np.ndarray,
    neighbors: np.ndarray,
    part_types: np.ndarray,
) -> np.ndarray:
    """
    Compute isolation score for each voxel (JIT-optimized).

    For each voxel, count the fraction of neighbors that solidified before it.
    High values mean the voxel was surrounded by solid material when it froze,
    indicating a likely shrinkage defect candidate.

    Chill (2) and core (4) neighbours are skipped entirely, the same way an
    absent neighbour at a mold wall is. They are solid from t=0, so counting
    them in the denominator alone would dilute the score; counting them as
    "solidified before" would instead flag every chilled face as isolated.
    Neither is right - a chill was never a feed path.

    Args:
        solid_times: Solidification time per voxel (0 = never solidified)
        neighbors: Neighbor connectivity matrix (n_voxels x max_neighbors)
        part_types: Part type per voxel (0=CAST, 1=RISER, 2=CHILL, 3=RISER_POST, 4=CORE)

    Returns:
        Array of isolation scores in [0, 1].
    """
    n_voxels = solid_times.shape[0]
    isolation_score = np.zeros(n_voxels, dtype=np.float32)
    max_neighbors = neighbors.shape[1]

    for i in prange(n_voxels):
        my_solid_time = solid_times[i]
        if my_solid_time <= 0.0:
            continue

        neighbors_before = 0
        total_neighbors = 0

        for j in range(max_neighbors):
            neighbor_idx = neighbors[i, j]
            if neighbor_idx >= 0:
                neighbor_type = part_types[neighbor_idx]
                if neighbor_type == 2 or neighbor_type == 4:
                    continue
                total_neighbors += 1
                neighbor_time = solid_times[neighbor_idx]
                if neighbor_time > 0.0 and neighbor_time < my_solid_time:
                    neighbors_before += 1

        if total_neighbors > 0:
            isolation_score[i] = neighbors_before / total_neighbors

    return isolation_score


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_isolation_score_volume_weighted(
    solid_times: np.ndarray,
    neighbors: np.ndarray,
    part_types: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Tet-mesh isolation score: volume fraction of neighbours frozen first.

    Same rule as :func:`compute_isolation_score_parallel` (chill and core
    neighbours skipped, absent slots skipped), but each neighbour counts with
    its lumped node volume instead of 1. Tet node degree and neighbour size
    vary (unlike the uniform 26-cell stencil), so an unweighted count lets a
    cluster of small neighbours dominate. With uniform volumes the result is
    identical to the count fraction. Range stays [0, 1].
    """
    n_points = solid_times.shape[0]
    isolation_score = np.zeros(n_points, dtype=np.float32)
    width = neighbors.shape[1]

    for i in prange(n_points):
        my_solid_time = solid_times[i]
        if my_solid_time <= 0.0:
            continue

        volume_before = 0.0
        volume_total = 0.0
        for j in range(width):
            neighbor_idx = neighbors[i, j]
            if neighbor_idx < 0:
                continue
            neighbor_type = part_types[neighbor_idx]
            if neighbor_type == 2 or neighbor_type == 4:
                continue
            weight = volumes[neighbor_idx]
            volume_total += weight
            neighbor_time = solid_times[neighbor_idx]
            if neighbor_time > 0.0 and neighbor_time < my_solid_time:
                volume_before += weight

        if volume_total > 0.0:
            isolation_score[i] = volume_before / volume_total

    return isolation_score


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_solid_path_strain_local(
    solidification_times: np.ndarray,
    point_centers: np.ndarray,
    neighbor_matrix: np.ndarray,
    thermal_expansion: float,
    temp_range: float,
    max_path_lengths: np.ndarray,
) -> np.ndarray:
    """Tet-mesh variant of :func:`compute_solid_path_strain_parallel`.

    Identical rule, but the path-length cap is per point
    (``max_path_lengths[i]``, inches) so it follows the local tet edge length
    instead of one lattice pitch for the whole mesh.
    """
    n_points = solidification_times.shape[0]
    width = neighbor_matrix.shape[1]
    path_strain = np.zeros(n_points, dtype=np.float32)

    max_time = 0.0
    for i in range(n_points):
        if solidification_times[i] > max_time:
            max_time = solidification_times[i]
    if max_time <= 0:
        return path_strain

    early_solid_threshold = max_time * 0.4
    for i in prange(n_points):
        my_time = solidification_times[i]
        if my_time < early_solid_threshold:
            continue
        max_solid_path = 0.0
        for j in range(width):
            neighbor_idx = neighbor_matrix[i, j]
            if 0 <= neighbor_idx < n_points:
                neighbor_time = solidification_times[neighbor_idx]
                if neighbor_time < my_time * 0.8 and neighbor_time > 0:
                    dx = point_centers[i, 0] - point_centers[neighbor_idx, 0]
                    dy = point_centers[i, 1] - point_centers[neighbor_idx, 1]
                    dz = point_centers[i, 2] - point_centers[neighbor_idx, 2]
                    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                    time_weight = (my_time - neighbor_time) / max_time
                    effective_path = dist * (1.0 + time_weight * 2.0)
                    if effective_path > max_solid_path:
                        max_solid_path = effective_path
        cap = max_path_lengths[i]
        if cap <= 0.0:
            continue
        if max_solid_path > cap:
            max_solid_path = cap
        path_strain[i] = thermal_expansion * temp_range * (max_solid_path / cap)

    return path_strain


# Global acceleration manager instance
accelerator = AccelerationManager()
