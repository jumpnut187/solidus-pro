"""
Stress and Distortion Analysis Module
Predicts residual stresses and distortion from differential thermal contraction.

MAGMASOFT-style capabilities:
- Thermal strain from cooling
- Constraint strain between regions cooling at different rates
- Distortion vectors
- Residual stress estimation

Physics:
    Thermal strain: ε_th = α * ΔT
    Constraint strain: ε_c = α * (T_neighbor - T_self)
    Stress: σ = E * ε (with effective modulus in mushy zone)

Key Outputs:
- Distortion magnitude map (0-1 normalized)
- Distortion vectors (for visualization)
- Residual stress magnitude
- High-stress zones (crack susceptibility)

Note: This is a SIMPLIFIED thermo-mechanical model. For precise stress analysis,
use dedicated FEA software. This provides qualitative guidance for:
- Identifying high-stress regions
- Comparing design alternatives
- Predicting distortion tendencies

Author: GitHub Copilot
Date: December 2025
"""

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
    """Results from stress and distortion analysis"""
    # Per-voxel stress results
    thermal_strain: np.ndarray           # Thermal strain magnitude
    constraint_strain: np.ndarray        # Constraint strain from neighbors
    effective_stress: np.ndarray         # Effective stress (MPa)
    distortion_magnitude: np.ndarray     # Distortion magnitude (inches)
    distortion_vectors: np.ndarray       # Distortion direction vectors (Nx3)
    
    # Derived results
    high_stress_zones: np.ndarray        # Boolean mask of high-stress regions
    crack_susceptibility: np.ndarray     # 0-1 crack risk score
    
    # Statistics
    max_stress: float                    # Maximum stress (MPa)
    avg_stress: float                    # Average stress (MPa)
    max_distortion: float                # Maximum distortion (inches)
    high_stress_fraction: float          # Fraction of voxels with high stress


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_thermal_strain_field(
    solidification_times: np.ndarray,
    max_solidification_time: float,
    thermal_expansion: float,
    delta_T: float
) -> np.ndarray:
    """
    Compute thermal strain at each voxel based on cooling sequence.

    This is a heuristic fallback for the geometry-driven engine. Without a
    transient temperature history, earlier-solidifying regions are assumed to
    have experienced more post-solidification cooling and therefore more
    contraction. The weighting is intentionally bounded to 50-100% of the full
    ``α * ΔT`` strain so the field remains comparative rather than pretending
    to be a resolved thermo-mechanical solution.
    
    Args:
        solidification_times: Time when each voxel solidified
        max_solidification_time: Maximum solidification time
        thermal_expansion: Thermal expansion coefficient (1/°C)
        delta_T: Total temperature change (°C)
        
    Returns:
        Thermal strain at each voxel
    """
    n_voxels = len(solidification_times)
    strain = np.zeros(n_voxels, dtype=np.float64)
    
    for i in prange(n_voxels):
        # Normalized solidification progress (0 = first, 1 = last)
        if max_solidification_time > 0:
            progress = solidification_times[i] / max_solidification_time
        else:
            progress = 0.5
        
        # Earlier solidification = more cooling = more strain
        # Later solidification = less cooling since initial pour
        cooling_factor = 1.0 - 0.5 * progress  # Range: 0.5 to 1.0
        
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
    """
    Compute constraint strain from differential cooling with neighbors.
    
    When adjacent voxels solidify at different times, they want to contract by
    different amounts. This creates constraint stresses. The solver uses
    normalized time differences and averages the neighbor contribution so noisy
    local timing outliers do not accumulate unbounded strain.
    
    Args:
        voxel_centers: Position of each voxel
        neighbors: Neighbor connectivity array
        solidification_times: When each voxel solidified
        thermal_expansion: Thermal expansion coefficient
        delta_T: Total temperature change
        max_time: Maximum solidification time
        
    Returns:
        Tuple of (constraint_strain, distortion_vectors)
    """
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
            
            # Time difference = differential cooling
            time_diff = abs(my_progress - neighbor_progress)
            
            # Differential strain
            diff_strain = thermal_expansion * delta_T * time_diff
            total_strain += diff_strain
            
            # Direction of constraint (from neighbor to self)
            dx = voxel_centers[i, 0] - voxel_centers[neighbor_idx, 0]
            dy = voxel_centers[i, 1] - voxel_centers[neighbor_idx, 1]
            dz = voxel_centers[i, 2] - voxel_centers[neighbor_idx, 2]
            
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            if dist > 1e-6:
                # Weight by differential strain and normalize
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
    """
    Compute effective stress from strain fields.
    
    Uses reduced modulus in mushy zone (can't support full stress).
    
    Args:
        thermal_strain: Thermal strain array
        constraint_strain: Constraint strain array
        elastic_modulus: Young's modulus (MPa)
        fraction_solid: Fraction solid at each voxel
        
    Returns:
        Effective stress (MPa)
    """
    n_voxels = len(thermal_strain)
    stress = np.zeros(n_voxels, dtype=np.float64)
    
    for i in prange(n_voxels):
        fs = fraction_solid[i]
        
        # Effective modulus drops significantly in mushy zone
        # Solid: full modulus
        # Liquid: zero modulus
        # Mushy: gradual transition
        if fs >= 0.9:
            E_eff = elastic_modulus
        elif fs <= 0.1:
            E_eff = 0.0
        else:
            # Nonlinear transition in mushy zone
            # Stress-bearing capacity develops above ~50% solid
            if fs > 0.5:
                modulus_factor = ((fs - 0.5) / 0.4) ** 2
            else:
                modulus_factor = 0.0
            E_eff = elastic_modulus * modulus_factor
        
        # Total effective strain
        total_strain = thermal_strain[i] + constraint_strain[i]
        
        # Stress (convert to MPa)
        stress[i] = E_eff * total_strain
    
    return stress


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_distortion_magnitude(
    distortion_vectors: np.ndarray,
    constraint_strain: np.ndarray,
    voxel_size: float
) -> np.ndarray:
    """
    Compute distortion magnitude at each voxel.
    
    Distortion = weighted displacement tendency * length scale
    
    Args:
        distortion_vectors: Strain-weighted displacement-tendency vectors
        constraint_strain: Constraint strain values (kept for API compatibility)
        voxel_size: Voxel size (inches)
        
    Returns:
        Distortion magnitude (inches)
    """
    n_voxels = len(constraint_strain)
    magnitude = np.zeros(n_voxels, dtype=np.float64)
    
    for i in prange(n_voxels):
        # Vector magnitude
        vec_mag = np.sqrt(
            distortion_vectors[i, 0]**2 +
            distortion_vectors[i, 1]**2 +
            distortion_vectors[i, 2]**2
        )
        
        # distortion_vectors already carry the strain weighting from
        # compute_constraint_strain_field, so applying constraint_strain again
        # would square the strain term.
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
    """
    Compute crack susceptibility based on stress and strain.
    
    High susceptibility when:
    - Stress approaches yield strength
    - High strain rate in mushy zone
    - BTR (brittle temperature range) conditions
    
    Args:
        effective_stress: Effective stress (MPa)
        constraint_strain: Constraint strain
        fraction_solid: Fraction solid
        yield_strength: Material yield strength (MPa)
        critical_strain: Critical strain for cracking
        
    Returns:
        Crack susceptibility (0-1)
    """
    n_voxels = len(effective_stress)
    susceptibility = np.zeros(n_voxels, dtype=np.float64)
    
    for i in prange(n_voxels):
        fs = fraction_solid[i]
        
        # Stress factor (ratio to yield)
        stress_factor = min(1.0, effective_stress[i] / yield_strength) if yield_strength > 0 else 0
        
        # Strain factor
        strain_factor = min(1.0, constraint_strain[i] / critical_strain) if critical_strain > 0 else 0
        
        # BTR factor (highest susceptibility in mushy zone 0.85-0.95 solid)
        if 0.85 <= fs <= 0.95:
            btr_factor = 1.0
        elif 0.7 <= fs < 0.85:
            btr_factor = (fs - 0.7) / 0.15
        elif 0.95 < fs <= 1.0:
            btr_factor = (1.0 - fs) / 0.05
        else:
            btr_factor = 0.0
        
        # Combined susceptibility
        # High stress OR high strain in BTR = high risk
        susceptibility[i] = max(
            stress_factor * btr_factor,
            strain_factor * btr_factor
        )
    
    return susceptibility


class StressAnalyzer:
    """
    Analyzes thermal stress and distortion in solidifying castings.
    
    This simplified thermo-mechanical model provides:
    - Thermal strain from cooling
    - Constraint strain from differential solidification
    - Estimated residual stress magnitude
    - Distortion vectors for visualization
    - Crack susceptibility mapping
    
    Limitations:
    - Does not account for mold constraint
    - Simplified elastic model (no plasticity)
    - No relaxation/creep effects
    
    For precise stress analysis, use dedicated FEA tools.
    """
    
    def __init__(self,
                 voxel_mesh,  # AdaptiveVoxelMesh
                 alloy: Optional['AlloyProperties'] = None):
        """
        Initialize stress analyzer.
        
        Args:
            voxel_mesh: AdaptiveVoxelMesh with neighbor connectivity
            alloy: AlloyProperties for material parameters
        """
        self.mesh = voxel_mesh
        self.alloy = alloy
        
        # Material properties (defaults for carbon steel)
        if alloy is not None:
            self.thermal_expansion = 12e-6  # 1/°C, typical for steel
            self.elastic_modulus = 200000.0  # MPa
            self.yield_strength = 250.0  # MPa
            self.delta_T = alloy.pouring_temp - 25.0  # Total cooling range
        else:
            self.thermal_expansion = 12e-6
            self.elastic_modulus = 200000.0
            self.yield_strength = 250.0
            self.delta_T = 1500.0
        
        # Results
        self.result: Optional[StressAnalysisResult] = None
    
    def analyze(self,
                solidification_times: np.ndarray,
                fraction_solid: np.ndarray,
                show_progress: bool = True) -> StressAnalysisResult:
        """
        Perform stress and distortion analysis.
        
        Args:
            solidification_times: When each voxel solidified (seconds)
            fraction_solid: Final fraction solid at each voxel
            show_progress: Print progress information
            
        Returns:
            StressAnalysisResult with all computed fields
        """
        n_voxels = self.mesh.get_voxel_count()
        
        if show_progress:
            print("\n" + "="*60)
            print("Stress and Distortion Analysis")
            print("="*60)
            print(f"  Voxels: {n_voxels:,}")
            print(f"  ΔT: {self.delta_T:.0f}°C")
            print(f"  α: {self.thermal_expansion:.2e} 1/°C")
        
        max_time = np.max(solidification_times)
        
        # Step 1: Compute thermal strain field
        if show_progress:
            print("  Computing thermal strain field...")
        
        thermal_strain = compute_thermal_strain_field(
            np.ascontiguousarray(solidification_times, dtype=np.float64),
            max_time,
            self.thermal_expansion,
            self.delta_T
        )
        
        # Step 2: Compute constraint strain and distortion vectors
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
        
        # Step 3: Compute effective stress
        if show_progress:
            print("  Computing effective stress...")
        
        effective_stress = compute_effective_stress(
            thermal_strain,
            constraint_strain,
            self.elastic_modulus,
            np.ascontiguousarray(fraction_solid, dtype=np.float64)
        )
        
        # Step 4: Compute distortion magnitude
        if show_progress:
            print("  Computing distortion magnitude...")
        
        distortion_magnitude = compute_distortion_magnitude(
            distortion_vectors,
            constraint_strain,
            self.mesh.base_voxel_size
        )
        
        # Step 5: Compute crack susceptibility
        if show_progress:
            print("  Computing crack susceptibility...")
        
        crack_susceptibility = compute_crack_susceptibility(
            effective_stress,
            constraint_strain,
            np.ascontiguousarray(fraction_solid, dtype=np.float64),
            self.yield_strength
        )
        
        # Identify high-stress zones
        stress_threshold = 0.5 * self.yield_strength  # 50% of yield
        high_stress_zones = effective_stress > stress_threshold
        
        # Calculate statistics
        max_stress = np.max(effective_stress)
        avg_stress = np.mean(effective_stress)
        max_distortion = np.max(distortion_magnitude)
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
        """
        Get arrow vectors for visualizing distortion directions.
        
        Args:
            scale_factor: Scale distortion for visibility
            sample_fraction: Fraction of voxels to show
            min_magnitude: Minimum distortion to show arrow
            
        Returns:
            List of (start_point, scaled_direction) tuples
        """
        if self.result is None:
            raise RuntimeError("Call analyze() first")
        
        arrows = []
        voxel_centers = np.asarray(self.mesh.voxel_centers)
        
        # Filter by magnitude
        valid_mask = self.result.distortion_magnitude > min_magnitude
        valid_indices = np.where(valid_mask)[0]
        
        # Sample
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
        """Get stress analysis statistics"""
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


def create_stress_analyzer(mesh, alloy=None) -> StressAnalyzer:
    """
    Convenience function to create StressAnalyzer.
    
    Args:
        mesh: AdaptiveVoxelMesh
        alloy: AlloyProperties (optional)
        
    Returns:
        Configured StressAnalyzer
    """
    return StressAnalyzer(mesh, alloy)


# Test function
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
