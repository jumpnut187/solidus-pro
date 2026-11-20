import numpy as np
from numba import jit, prange
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

try:
    from core.materials import AlloyProperties
except ImportError:
    AlloyProperties = None

try:
    from core.memory_manager import create_managed_array
except ImportError:
    create_managed_array = np.zeros

@dataclass
class StressAnalysisResult:
    thermal_strain: np.ndarray
    constraint_strain: np.ndarray
    effective_stress: np.ndarray
    distortion_magnitude: np.ndarray
    distortion_vectors: np.ndarray

    high_stress_zones: np.ndarray
    crack_susceptibility: np.ndarray

    max_stress: float
    avg_stress: float
    max_distortion: float
    high_stress_fraction: float

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_thermal_strain_field(
    solidification_times: np.ndarray,
    max_solidification_time: float,
    thermal_expansion: float,
    delta_T: float
) -> np.ndarray:
    n_voxels = len(solidification_times)
    strain = np.zeros(n_voxels, dtype=np.float64)

    for i in prange(n_voxels):
        if max_solidification_time > 0:
            progress = solidification_times[i] / max_solidification_time
        else:
            progress = 0.5

        cooling_factor = 1.0 - 0.5 * progress

        strain[i] = thermal_expansion * delta_T * cooling_factor

    return strain

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_constraint_strain_field(
    voxel_centers: np.ndarray,
    neighbors: np.ndarray,
    solidification_times: np.ndarray,
    thermal_expansion: float,
    delta_T: float,
    max_time: float
) -> Tuple[np.ndarray, np.ndarray]:
    n_voxels = len(voxel_centers)
    max_neighbors = neighbors.shape[1]

    constraint_strain = np.zeros(n_voxels, dtype=np.float64)
    distortion_vectors = np.zeros((n_voxels, 3), dtype=np.float64)

    for i in prange(n_voxels):
        my_time = solidification_times[i]
        my_progress = my_time / max_time if max_time > 0 else 0.5

        total_strain = 0.0
        total_dx = 0.0
        total_dy = 0.0
        total_dz = 0.0
        neighbor_count = 0

        for j in range(max_neighbors):
            neighbor_idx = neighbors[i, j]
            if neighbor_idx < 0:
                continue

            neighbor_time = solidification_times[neighbor_idx]
            neighbor_progress = neighbor_time / max_time if max_time > 0 else 0.5

            time_diff = abs(my_progress - neighbor_progress)

            diff_strain = thermal_expansion * delta_T * time_diff
            total_strain += diff_strain

            dx = voxel_centers[i, 0] - voxel_centers[neighbor_idx, 0]
            dy = voxel_centers[i, 1] - voxel_centers[neighbor_idx, 1]
            dz = voxel_centers[i, 2] - voxel_centers[neighbor_idx, 2]

            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            if dist > 1e-6:
                weight = diff_strain
                total_dx += weight * dx / dist
                total_dy += weight * dy / dist
                total_dz += weight * dz / dist

            neighbor_count += 1

        if neighbor_count > 0:
            constraint_strain[i] = total_strain / neighbor_count
            distortion_vectors[i, 0] = total_dx
            distortion_vectors[i, 1] = total_dy
            distortion_vectors[i, 2] = total_dz

    return constraint_strain, distortion_vectors

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_effective_stress(
    thermal_strain: np.ndarray,
    constraint_strain: np.ndarray,
    elastic_modulus: float,
    fraction_solid: np.ndarray
) -> np.ndarray:
    n_voxels = len(thermal_strain)
    stress = np.zeros(n_voxels, dtype=np.float64)

    for i in prange(n_voxels):
        fs = fraction_solid[i]

        if fs >= 0.9:
            E_eff = elastic_modulus
        elif fs <= 0.1:
            E_eff = 0.0
        else:
            if fs > 0.5:
                modulus_factor = ((fs - 0.5) / 0.4) ** 2
            else:
                modulus_factor = 0.0
            E_eff = elastic_modulus * modulus_factor

        total_strain = thermal_strain[i] + constraint_strain[i]

        stress[i] = E_eff * total_strain

    return stress

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_distortion_magnitude(
    distortion_vectors: np.ndarray,
    constraint_strain: np.ndarray,
    voxel_size: float
) -> np.ndarray:
    n_voxels = len(constraint_strain)
    magnitude = np.zeros(n_voxels, dtype=np.float64)

    for i in prange(n_voxels):
        vec_mag = np.sqrt(
            distortion_vectors[i, 0]**2 +
            distortion_vectors[i, 1]**2 +
            distortion_vectors[i, 2]**2
        )

        magnitude[i] = voxel_size * vec_mag

    return magnitude

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_crack_susceptibility(
    effective_stress: np.ndarray,
    constraint_strain: np.ndarray,
    fraction_solid: np.ndarray,
    yield_strength: float,
    critical_strain: float = 0.02
) -> np.ndarray:
    n_voxels = len(effective_stress)
    susceptibility = np.zeros(n_voxels, dtype=np.float64)

    for i in prange(n_voxels):
        fs = fraction_solid[i]

        stress_factor = min(1.0, effective_stress[i] / yield_strength) if yield_strength > 0 else 0

        strain_factor = min(1.0, constraint_strain[i] / critical_strain) if critical_strain > 0 else 0

        if 0.85 <= fs <= 0.95:
            btr_factor = 1.0
        elif 0.7 <= fs < 0.85:
            btr_factor = (fs - 0.7) / 0.15
        elif 0.95 < fs <= 1.0:
            btr_factor = (1.0 - fs) / 0.05
        else:
            btr_factor = 0.0

        susceptibility[i] = max(
            stress_factor * btr_factor,
            strain_factor * btr_factor
        )

    return susceptibility

class StressAnalyzer:

    def __init__(self,
                 voxel_mesh,
                 alloy: Optional['AlloyProperties'] = None):
        self.mesh = voxel_mesh
        self.alloy = alloy

        if alloy is not None:
            self.thermal_expansion = 12e-6
            self.elastic_modulus = 200000.0
            self.yield_strength = 250.0
            self.delta_T = alloy.pouring_temp - 25.0
        else:
            self.thermal_expansion = 12e-6
            self.elastic_modulus = 200000.0
            self.yield_strength = 250.0
            self.delta_T = 1500.0

        self.result: Optional[StressAnalysisResult] = None

    def analyze(self,
                solidification_times: np.ndarray,
                fraction_solid: np.ndarray,
                show_progress: bool = True) -> StressAnalysisResult:
        n_voxels = self.mesh.get_voxel_count()

        if show_progress:
            print("\n" + "="*60)
            print("Stress and Distortion Analysis")
            print("="*60)
            print(f"  Voxels: {n_voxels:,}")
            print(f"  ΔT: {self.delta_T:.0f}°C")
            print(f"  α: {self.thermal_expansion:.2e} 1/°C")

        max_time = np.max(solidification_times)

        if show_progress:
            print("  Computing thermal strain field...")

        thermal_strain = compute_thermal_strain_field(
            np.ascontiguousarray(solidification_times, dtype=np.float64),
            max_time,
            self.thermal_expansion,
            self.delta_T
        )

        if show_progress:
            print("  Computing constraint strain field...")

        voxel_centers = np.ascontiguousarray(self.mesh.voxel_centers, dtype=np.float64)
        neighbors = np.ascontiguousarray(self.mesh.voxel_neighbors, dtype=np.int32)

        constraint_strain, distortion_vectors = compute_constraint_strain_field(
            voxel_centers,
            neighbors,
            np.ascontiguousarray(solidification_times, dtype=np.float64),
            self.thermal_expansion,
            self.delta_T,
            max_time
        )

        if show_progress:
            print("  Computing effective stress...")

        effective_stress = compute_effective_stress(
            thermal_strain,
            constraint_strain,
            self.elastic_modulus,
            np.ascontiguousarray(fraction_solid, dtype=np.float64)
        )

        if show_progress:
            print("  Computing distortion magnitude...")

        distortion_magnitude = compute_distortion_magnitude(
            distortion_vectors,
            constraint_strain,
            self.mesh.base_voxel_size
        )

        if show_progress:
            print("  Computing crack susceptibility...")

        crack_susceptibility = compute_crack_susceptibility(
            effective_stress,
            constraint_strain,
            np.ascontiguousarray(fraction_solid, dtype=np.float64),
            self.yield_strength
        )

        stress_threshold = 0.5 * self.yield_strength
        high_stress_zones = effective_stress > stress_threshold

        max_stress = np.max(effective_stress)
        max_distortion = np.max(distortion_magnitude)
        avg_stress = np.mean(effective_stress)
        high_stress_fraction = np.sum(high_stress_zones) / n_voxels

        self.result = StressAnalysisResult(
            thermal_strain=thermal_strain.astype(np.float32),
            constraint_strain=constraint_strain.astype(np.float32),
            effective_stress=effective_stress.astype(np.float32),
            distortion_magnitude=distortion_magnitude.astype(np.float32),
            distortion_vectors=distortion_vectors.astype(np.float32),
            high_stress_zones=high_stress_zones,
            crack_susceptibility=crack_susceptibility.astype(np.float32),
            max_stress=max_stress,
            avg_stress=avg_stress,
            max_distortion=max_distortion,
            high_stress_fraction=high_stress_fraction
        )

        if show_progress:
            print(f"\n  Results:")
            print(f"    Max stress: {max_stress:.1f} MPa")
            print(f"    Avg stress: {avg_stress:.1f} MPa")
            print(f"    Max distortion: {max_distortion*1000:.3f} mils")
            print(f"    High-stress fraction: {high_stress_fraction*100:.1f}%")
            print("="*60 + "\n")

        return self.result

    def get_distortion_arrows(self,
                               scale_factor: float = 100.0,
                               sample_fraction: float = 0.1,
                               min_magnitude: float = 0.0001) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self.result is None:
            raise RuntimeError("Call analyze() first")

        arrows = []
        voxel_centers = np.asarray(self.mesh.voxel_centers)

        valid_mask = self.result.distortion_magnitude > min_magnitude
        valid_indices = np.where(valid_mask)[0]

        n_sample = max(10, int(len(valid_indices) * sample_fraction))
        if len(valid_indices) > n_sample:
            sample_indices = np.random.choice(valid_indices, n_sample, replace=False)
        else:
            sample_indices = valid_indices

        for idx in sample_indices:
            start = voxel_centers[idx]
            direction = self.result.distortion_vectors[idx] * scale_factor

            if np.linalg.norm(direction) > 1e-6:
                arrows.append((start, direction))

        return arrows

    def get_statistics(self) -> Dict:
        if self.result is None:
            return {}

        return {
            'max_stress_mpa': self.result.max_stress,
            'avg_stress_mpa': self.result.avg_stress,
            'max_distortion_inches': self.result.max_distortion,
            'max_distortion_mils': self.result.max_distortion * 1000,
            'high_stress_fraction': self.result.high_stress_fraction,
            'thermal_expansion': self.thermal_expansion,
            'elastic_modulus_mpa': self.elastic_modulus
        }

if __name__ == '__main__':
    print("Stress and Distortion Analysis Module")
    print("="*60)
    print("This module provides thermal stress and distortion analysis.")
    print("\nKey features:")
    print("  - Thermal strain from cooling")
    print("  - Constraint strain from differential solidification")
    print("  - Effective stress calculation")
    print("  - Distortion magnitude and direction")
    print("  - Crack susceptibility mapping")
