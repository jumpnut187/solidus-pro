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

optimal_threads = hardware.get_optimal_thread_count('compute')
set_num_threads(optimal_threads)
if multiprocessing.current_process().name == "MainProcess":
    print(f"Numba configured to use {optimal_threads} threads (physical cores only)")

class AccelerationManager:

    def __init__(self):
        self.gpu_available = False
        self.neural_engine_available = False
        self.opencl_context = None

    def _init_gpu(self):
        try:
            import pyopencl as cl

            platforms = cl.get_platforms()
            if not platforms:
                print("GPU: No OpenCL platforms found, using CPU only")
                self.gpu_available = False
                return

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
        if n_jobs is None:
            n_jobs = hardware.get_optimal_thread_count('compute')

        return Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(func)(item) for item in data
        )

    def parallel_compute(self, func: Callable, *args, n_jobs: int = None, **kwargs):
        if n_jobs is None:
            n_jobs = hardware.get_optimal_thread_count('compute')

        return func(*args, n_jobs=n_jobs, **kwargs)

    def gpu_compute_modulus(self, volumes: np.ndarray, areas: np.ndarray) -> np.ndarray:
        return compute_modulus_parallel(volumes, areas)

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_modulus_parallel(volumes: np.ndarray, areas: np.ndarray) -> np.ndarray:
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
    n_voxels = voxel_states.shape[0]
    new_states = voxel_states.copy()

    feeding_zone_threshold = 0.6

    for i in prange(n_voxels):
        if voxel_states[i] < 1.0:
            in_feeding_zone = riser_slowdown[i] < feeding_zone_threshold

            has_solid_neighbor = False
            has_liquid_feeding_neighbor = False

            for j in range(neighbors.shape[1]):
                neighbor_idx = neighbors[i, j]
                if neighbor_idx >= 0:
                    if voxel_states[neighbor_idx] >= 1.0:
                        has_solid_neighbor = True
                    elif voxel_states[neighbor_idx] < 1.0 and riser_slowdown[neighbor_idx] < feeding_zone_threshold:
                        has_liquid_feeding_neighbor = True

            if in_feeding_zone and has_liquid_feeding_neighbor:
                effective_slowdown = riser_slowdown[i] * 0.6
            else:
                effective_slowdown = riser_slowdown[i]

            if has_solid_neighbor:
                base_rate = 0.02
                solidification_rate = base_rate / (1.0 + moduli[i] * moduli[i] * 2.0)

                vertical_factor = 1.0 + (1.0 - heights[i]) * 0.3

                thermal_factor = material_factor

                part_type_factor = chill_boost[i] * effective_slowdown

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
    n_voxels = voxel_states.shape[0]
    new_states = voxel_states.copy()

    CAST = 0
    RISER = 1
    CHILL = 2

    feeding_zone_threshold = 0.6

    for i in prange(n_voxels):
        if voxel_states[i] < 1.0:

            is_riser_voxel = part_types[i] == RISER
            has_liquid_riser_neighbor = False
            has_liquid_feeding_neighbor = False
            has_solid_neighbor = False

            for j in range(neighbors.shape[1]):
                neighbor_idx = neighbors[i, j]
                if neighbor_idx >= 0:
                    neighbor_state = voxel_states[neighbor_idx]
                    neighbor_part_type = part_types[neighbor_idx]
                    neighbor_slowdown = riser_slowdown[neighbor_idx]

                    if neighbor_state >= 1.0:
                        has_solid_neighbor = True
                    else:
                        if neighbor_part_type == RISER:
                            has_liquid_riser_neighbor = True
                        elif neighbor_slowdown < feeding_zone_threshold:
                            has_liquid_feeding_neighbor = True

            in_feeding_zone = (
                is_riser_voxel or
                has_liquid_riser_neighbor or
                (riser_slowdown[i] < feeding_zone_threshold and has_liquid_feeding_neighbor)
            )

            if in_feeding_zone:
                if is_riser_voxel:
                    effective_slowdown = riser_slowdown[i]
                elif has_liquid_riser_neighbor:
                    effective_slowdown = riser_slowdown[i] * 0.4
                else:
                    effective_slowdown = riser_slowdown[i] * 0.6
            else:
                effective_slowdown = riser_slowdown[i]

            if has_solid_neighbor:
                base_rate = 0.02
                solidification_rate = base_rate / (1.0 + moduli[i] * moduli[i] * 2.0)

                if in_feeding_zone:
                    vertical_factor = 1.0 + (1.0 - heights[i]) * feeding_zone_vertical_bias
                else:
                    vertical_factor = 1.0 + (1.0 - heights[i]) * 0.3

                thermal_factor = material_factor

                part_type_factor = chill_boost[i] * effective_slowdown

                delta = solidification_rate * vertical_factor * thermal_factor * part_type_factor * time_step
                new_states[i] = min(voxel_states[i] + delta, 1.0)

    return new_states

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_min_distances_parallel(
    points: np.ndarray,
    reference_points: np.ndarray
) -> np.ndarray:
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
    n_voxels = voxel_centers.shape[0]
    n_chills = chill_centers.shape[0]
    chill_boost = np.ones(n_voxels, dtype=np.float32)

    if n_chills == 0:
        return chill_boost

    time_decay = np.exp(-current_time / air_gap_time_constant)

    for i in prange(n_voxels):
        if part_types[i] == 2:
            continue

        px, py, pz = voxel_centers[i, 0], voxel_centers[i, 1], voxel_centers[i, 2]

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

            influence_radius = max(local_chill_thickness, 0.1)

            if min_dist < influence_radius:
                spacial_influence = 1.0 - (min_dist / influence_radius)

                saturation_factor = max(1.0 - (voxel_states[i] / saturation_threshold), 0.0)

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
    n_cast = cast_indices.shape[0]
    influenced_count = 0
    inner_radius = riser_influence_radius * 0.4

    for i in prange(n_cast):
        dist = distances[i]
        cast_idx = cast_indices[i]

        if dist < riser_influence_radius:
            normalized_dist = dist / riser_influence_radius
            influence = (1.0 - normalized_dist) ** 2

            slowdown = min_influence_factor + (exothermic_factor - min_influence_factor) * influence

            if dist < inner_radius:
                close_blend = 1.0 - (dist / inner_radius)
                slowdown = slowdown * (1.0 - close_blend * 0.4) + exothermic_factor * (close_blend * 0.4)

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
    n_voxels = voxel_states.shape[0]
    air_gap_slowdown = np.ones(n_voxels, dtype=np.float32)

    time_factor = 1.0 - np.exp(-current_time / air_gap_time_constant)

    if time_factor < 0.01:
        return air_gap_slowdown

    for i in prange(n_voxels):
        dist_from_surface = feeding_distances[i]

        if dist_from_surface < surface_influence_depth:
            spatial_factor = 1.0 - (dist_from_surface / surface_influence_depth)

            solidification_factor = voxel_states[i]

            gap_effect = time_factor * spatial_factor * solidification_factor

            air_gap_slowdown[i] = 1.0 - (1.0 - max_slowdown) * gap_effect

    return air_gap_slowdown

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_modulus_variance_parallel(
    voxel_moduli: np.ndarray,
    neighbor_matrix: np.ndarray
) -> np.ndarray:
    n_voxels = voxel_moduli.shape[0]
    max_neighbors = neighbor_matrix.shape[1]
    modulus_variance = np.zeros(n_voxels, dtype=np.float32)

    for i in prange(n_voxels):
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
            modulus_variance[i] = np.sqrt(max(variance, 0.0))

    return modulus_variance

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_constraint_strain_parallel(
    solidification_times: np.ndarray,
    voxel_moduli: np.ndarray,
    neighbor_matrix: np.ndarray,
    thermal_expansion: float,
    temp_range: float
) -> np.ndarray:
    n_voxels = solidification_times.shape[0]
    max_neighbors = neighbor_matrix.shape[1]
    constraint_strain = np.zeros(n_voxels, dtype=np.float32)

    max_time = 0.0
    for i in range(n_voxels):
        if solidification_times[i] > max_time:
            max_time = solidification_times[i]

    if max_time <= 0:
        return constraint_strain

    for i in prange(n_voxels):
        my_time = solidification_times[i]
        my_modulus = voxel_moduli[i]

        if my_time <= 0:
            continue

        max_strain_seen = 0.0
        solid_neighbor_count = 0

        for j in range(max_neighbors):
            neighbor_idx = neighbor_matrix[i, j]
            if 0 <= neighbor_idx < n_voxels:
                neighbor_time = solidification_times[neighbor_idx]
                neighbor_modulus = voxel_moduli[neighbor_idx]

                if neighbor_time < my_time and neighbor_time > 0:
                    time_diff = my_time - neighbor_time
                    time_factor = time_diff / max_time

                    if my_modulus > 0:
                        modulus_ratio = neighbor_modulus / my_modulus
                        thickness_factor = 1.0 / (1.0 + modulus_ratio)
                    else:
                        thickness_factor = 0.5

                    neighbor_strain = thermal_expansion * temp_range * time_factor * thickness_factor
                    if neighbor_strain > max_strain_seen:
                        max_strain_seen = neighbor_strain
                    solid_neighbor_count += 1

        if solid_neighbor_count > 0:
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
    n_voxels = solidification_times.shape[0]
    max_neighbors = neighbor_matrix.shape[1]
    path_strain = np.zeros(n_voxels, dtype=np.float32)

    max_time = 0.0
    for i in range(n_voxels):
        if solidification_times[i] > max_time:
            max_time = solidification_times[i]

    if max_time <= 0:
        return path_strain

    early_solid_threshold = max_time * 0.4

    for i in prange(n_voxels):
        my_time = solidification_times[i]

        if my_time < early_solid_threshold:
            continue

        max_solid_path = 0.0

        for j in range(max_neighbors):
            neighbor_idx = neighbor_matrix[i, j]
            if 0 <= neighbor_idx < n_voxels:
                neighbor_time = solidification_times[neighbor_idx]

                if neighbor_time < my_time * 0.8 and neighbor_time > 0:
                    dx = voxel_centers[i, 0] - voxel_centers[neighbor_idx, 0]
                    dy = voxel_centers[i, 1] - voxel_centers[neighbor_idx, 1]
                    dz = voxel_centers[i, 2] - voxel_centers[neighbor_idx, 2]
                    dist = np.sqrt(dx*dx + dy*dy + dz*dz)

                    time_weight = (my_time - neighbor_time) / max_time
                    effective_path = dist * (1.0 + time_weight * 2.0)

                    if effective_path > max_solid_path:
                        max_solid_path = effective_path

        if max_solid_path > max_path_length:
            max_solid_path = max_path_length

        path_strain[i] = thermal_expansion * temp_range * (max_solid_path / max_path_length)

    return path_strain

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_isolation_score_parallel(
    solid_times: np.ndarray,
    neighbors: np.ndarray,
    part_types: np.ndarray,
) -> np.ndarray:
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

accelerator = AccelerationManager()
