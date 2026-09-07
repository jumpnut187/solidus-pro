"""
Microstructure Prediction Module
Predicts secondary dendrite arm spacing (SDAS) and grain structure from cooling rate.

industry-standard capability:
- Predict SDAS from local cooling rate
- Correlate microstructure with mechanical properties
- Visualize microstructure distribution

Physics:
    SDAS correlation: λ₂ = A * Ṫ^(-n)
    
    where:
        λ₂ = secondary dendrite arm spacing (μm)
        Ṫ = cooling rate (°C/s)
        A, n = alloy-specific constants

Mechanical property correlations:
    - Tensile strength increases with finer SDAS
    - Ductility generally improves with finer SDAS
    - Finer SDAS = lower permeability (harder to feed)

Key Outputs:
- SDAS map (μm)
- Predicted mechanical properties map
- Grain size estimation

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


@dataclass
class MicrostructureConstants:
    """
    Alloy-specific constants for SDAS prediction.
    
    λ₂ = A * Ṫ^(-n)
    
    Literature values for common alloys.
    """
    name: str
    A: float      # Pre-exponential constant (μm·(°C/s)^n)
    n: float      # Cooling rate exponent (typically 0.3-0.5)
    
    # Mechanical property correlations
    # σ_UTS = σ_0 + k_UTS / √λ₂  (Hall-Petch type relationship)
    sigma_0: float      # Base strength (MPa)
    k_uts: float        # Strengthening coefficient
    
    # Elongation correlation
    # ε = ε_max * (1 - c * λ₂)
    epsilon_max: float  # Maximum elongation (%)
    c_elong: float      # SDAS reduction coefficient


# Database of SDAS constants by alloy type
SDAS_CONSTANTS: Dict[str, MicrostructureConstants] = {
    'WCB': MicrostructureConstants(
        name='WCB Carbon Steel',
        A=150.0,
        n=0.40,
        sigma_0=350.0,
        k_uts=250.0,
        epsilon_max=25.0,
        c_elong=0.003
    ),
    'CF8': MicrostructureConstants(
        name='CF8 Stainless Steel',
        A=120.0,
        n=0.38,
        sigma_0=400.0,
        k_uts=200.0,
        epsilon_max=35.0,
        c_elong=0.002
    ),
    'C958': MicrostructureConstants(
        name='C958 Aluminum Bronze',
        A=180.0,
        n=0.42,
        sigma_0=250.0,
        k_uts=150.0,
        epsilon_max=15.0,
        c_elong=0.004
    ),
    'CA6NM': MicrostructureConstants(
        name='CA6NM Cast Steel',
        A=145.0,
        n=0.39,
        sigma_0=365.0,
        k_uts=225.0,
        epsilon_max=14.0,
        c_elong=0.0028
    ),
    'A356': MicrostructureConstants(
        name='A356 Aluminum',
        A=40.0,
        n=0.33,
        sigma_0=150.0,
        k_uts=100.0,
        epsilon_max=8.0,
        c_elong=0.005
    ),
    'GrayIron': MicrostructureConstants(
        name='Gray Cast Iron',
        A=200.0,
        n=0.35,
        sigma_0=200.0,
        k_uts=80.0,
        epsilon_max=2.0,
        c_elong=0.002
    ),
    'DuctileIron': MicrostructureConstants(
        name='Ductile Cast Iron',
        A=100.0,
        n=0.37,
        sigma_0=350.0,
        k_uts=200.0,
        epsilon_max=18.0,
        c_elong=0.003
    )
}


def get_sdas_constants(alloy_name: str) -> MicrostructureConstants:
    """
    Get SDAS constants for an alloy.
    
    Args:
        alloy_name: Alloy name (e.g., 'WCB', 'CF8', 'C958', 'CA6NM')
        
    Returns:
        MicrostructureConstants for the alloy
    """
    # Try exact match first
    if alloy_name in SDAS_CONSTANTS:
        return SDAS_CONSTANTS[alloy_name]
    
    # Try partial match
    for key in SDAS_CONSTANTS:
        if key.lower() in alloy_name.lower():
            return SDAS_CONSTANTS[key]
    
    # Default to WCB (carbon steel)
    return SDAS_CONSTANTS['WCB']


@dataclass
class MicrostructureResult:
    """Results from microstructure prediction"""
    # Primary outputs
    sdas: np.ndarray                     # Secondary dendrite arm spacing (μm)
    predicted_uts: np.ndarray            # Predicted ultimate tensile strength (MPa)
    predicted_elongation: np.ndarray     # Predicted elongation (%)
    
    # Derived
    fine_structure_zones: np.ndarray     # Boolean mask of fine microstructure regions
    coarse_structure_zones: np.ndarray   # Boolean mask of coarse microstructure regions
    
    # Statistics
    min_sdas: float                      # Minimum SDAS (μm) - finest structure
    max_sdas: float                      # Maximum SDAS (μm) - coarsest structure
    avg_sdas: float                      # Average SDAS (μm)
    avg_uts: float                       # Average UTS (MPa)


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_sdas_field(
    cooling_rates: np.ndarray,
    A: float,
    n: float,
    min_cooling_rate: float = 0.01
) -> np.ndarray:
    """
    Compute SDAS at each voxel from cooling rate.
    
    λ₂ = A * Ṫ^(-n)
    
    Args:
        cooling_rates: Local cooling rate (°C/s)
        A: Pre-exponential constant
        n: Cooling rate exponent
        min_cooling_rate: Minimum cooling rate to avoid division issues
        
    Returns:
        SDAS array (μm)
    """
    n_voxels = len(cooling_rates)
    sdas = np.zeros(n_voxels, dtype=np.float64)
    
    for i in prange(n_voxels):
        cr = max(cooling_rates[i], min_cooling_rate)
        
        # SDAS = A * cooling_rate^(-n)
        sdas[i] = A * (cr ** (-n))
        
        # Clamp to reasonable range
        sdas[i] = max(10.0, min(500.0, sdas[i]))  # 10-500 μm
    
    return sdas


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_mechanical_properties(
    sdas: np.ndarray,
    sigma_0: float,
    k_uts: float,
    epsilon_max: float,
    c_elong: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute predicted mechanical properties from SDAS.
    
    Uses Hall-Petch type relationships:
        σ_UTS = σ_0 + k / √λ₂
        ε = ε_max * (1 - c * λ₂)
    
    Args:
        sdas: Secondary dendrite arm spacing (μm)
        sigma_0: Base strength (MPa)
        k_uts: Strengthening coefficient
        epsilon_max: Maximum elongation (%)
        c_elong: SDAS reduction coefficient
        
    Returns:
        Tuple of (UTS array, elongation array)
    """
    n_voxels = len(sdas)
    uts = np.zeros(n_voxels, dtype=np.float64)
    elongation = np.zeros(n_voxels, dtype=np.float64)
    
    for i in prange(n_voxels):
        # UTS: Hall-Petch type
        sqrt_sdas = np.sqrt(sdas[i])
        uts[i] = sigma_0 + k_uts / sqrt_sdas
        
        # Elongation: decreases with coarser SDAS
        elong = epsilon_max * (1.0 - c_elong * sdas[i])
        elongation[i] = max(0.5, elong)  # Minimum 0.5%
    
    return uts, elongation


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def estimate_cooling_rate_from_solidification(
    solidification_times: np.ndarray,
    neighbors: np.ndarray,
    liquidus_temp: float,
    solidus_temp: float,
    voxel_size: float
) -> np.ndarray:
    """
    Estimate local cooling rate from solidification times.
    
    Cooling rate ≈ (T_liquidus - T_solidus) / solidification_time
    
    Uses a bounded neighbor-time refinement as an engineering estimate when no
    transient thermal field is available. This is intentionally a surrogate,
    not a replacement for a temperature-driven cooling-rate solve.
    
    Args:
        solidification_times: When each voxel solidified (seconds)
        neighbors: Neighbor connectivity
        liquidus_temp: Liquidus temperature (°C)
        solidus_temp: Solidus temperature (°C)
        voxel_size: Voxel size (inches)
        
    Returns:
        Estimated cooling rate (°C/s)
    """
    n_voxels = len(solidification_times)
    max_neighbors = neighbors.shape[1]
    
    cooling_rates = np.zeros(n_voxels, dtype=np.float64)
    freezing_range = liquidus_temp - solidus_temp
    
    for i in prange(n_voxels):
        my_time = solidification_times[i]
        
        if my_time <= 0:
            cooling_rates[i] = 10.0  # Default for instant solidification
            continue
        
        # Base estimate from solidification time
        # Faster solidification = higher cooling rate
        base_rate = freezing_range / my_time
        
        # Refine using neighbor gradient
        # Thermal gradient ≈ ΔT/Δx where ΔT is proportional to time difference
        max_time_diff = 0.0
        neighbor_count = 0
        
        for j in range(max_neighbors):
            neighbor_idx = neighbors[i, j]
            if neighbor_idx < 0:
                continue
            
            neighbor_time = solidification_times[neighbor_idx]
            time_diff = abs(my_time - neighbor_time)
            max_time_diff = max(max_time_diff, time_diff)
            neighbor_count += 1
        
        # Higher time gradient suggests a steeper local thermal field, but keep
        # the refinement bounded so isolated timing noise cannot dominate the
        # result.
        if neighbor_count > 0 and max_time_diff > 0:
            gradient_factor = min(1.5, 1.0 + (max_time_diff / my_time) * 0.5)
            cooling_rates[i] = base_rate * gradient_factor
        else:
            cooling_rates[i] = base_rate
        
        # Clamp to reasonable range
        cooling_rates[i] = max(0.01, min(100.0, cooling_rates[i]))
    
    return cooling_rates


class MicrostructurePredictor:
    """
    Predicts microstructure (SDAS) and mechanical properties from cooling rate.
    
    Key features:
    - SDAS prediction using alloy-specific correlations
    - Mechanical property estimation (UTS, elongation)
    - Fine/coarse structure zone identification
    
    Usage:
        predictor = MicrostructurePredictor(mesh, alloy)
        result = predictor.predict(solidification_times, cooling_rates)
        
        # Or estimate cooling rates from solidification times
        result = predictor.predict_from_solidification(solidification_times)
    """
    
    def __init__(self,
                 voxel_mesh,  # AdaptiveVoxelMesh
                 alloy: Optional['AlloyProperties'] = None):
        """
        Initialize microstructure predictor.
        
        Args:
            voxel_mesh: AdaptiveVoxelMesh
            alloy: AlloyProperties for material parameters
        """
        self.mesh = voxel_mesh
        self.alloy = alloy
        
        # Get SDAS constants for alloy
        alloy_name = alloy.name.split()[0] if alloy else 'WCB'
        self.constants = get_sdas_constants(alloy_name)
        
        # Temperature range
        if alloy is not None:
            self.liquidus_temp = alloy.liquidus_temp
            self.solidus_temp = alloy.solidus_temp
        else:
            self.liquidus_temp = 1500.0
            self.solidus_temp = 1450.0
        
        # Results
        self.result: Optional[MicrostructureResult] = None
    
    def predict(self,
                cooling_rates: np.ndarray,
                show_progress: bool = True) -> MicrostructureResult:
        """
        Predict microstructure from cooling rate field.
        
        Args:
            cooling_rates: Local cooling rate at each voxel (°C/s)
            show_progress: Print progress information
            
        Returns:
            MicrostructureResult with SDAS and property predictions
        """
        n_voxels = self.mesh.get_voxel_count()
        
        if show_progress:
            print("\n" + "="*60)
            print("Microstructure Prediction")
            print("="*60)
            print(f"  Alloy: {self.constants.name}")
            print(f"  SDAS constants: A={self.constants.A:.1f}, n={self.constants.n:.2f}")
        
        # Step 1: Compute SDAS field
        if show_progress:
            print("  Computing SDAS field...")
        
        sdas = compute_sdas_field(
            np.ascontiguousarray(cooling_rates, dtype=np.float64),
            self.constants.A,
            self.constants.n
        )
        
        # Step 2: Compute mechanical properties
        if show_progress:
            print("  Predicting mechanical properties...")
        
        uts, elongation = compute_mechanical_properties(
            sdas,
            self.constants.sigma_0,
            self.constants.k_uts,
            self.constants.epsilon_max,
            self.constants.c_elong
        )
        
        # Step 3: Identify fine/coarse structure zones
        avg_sdas = np.mean(sdas)
        fine_threshold = avg_sdas * 0.7  # 30% below average
        coarse_threshold = avg_sdas * 1.3  # 30% above average
        
        fine_zones = sdas < fine_threshold
        coarse_zones = sdas > coarse_threshold
        
        # Statistics
        min_sdas = np.min(sdas)
        max_sdas = np.max(sdas)
        avg_uts = np.mean(uts)
        
        self.result = MicrostructureResult(
            sdas=sdas.astype(np.float32),
            predicted_uts=uts.astype(np.float32),
            predicted_elongation=elongation.astype(np.float32),
            fine_structure_zones=fine_zones,
            coarse_structure_zones=coarse_zones,
            min_sdas=min_sdas,
            max_sdas=max_sdas,
            avg_sdas=avg_sdas,
            avg_uts=avg_uts
        )
        
        if show_progress:
            print(f"\n  Results:")
            print(f"    SDAS range: {min_sdas:.0f} - {max_sdas:.0f} μm")
            print(f"    Average SDAS: {avg_sdas:.0f} μm")
            print(f"    Average UTS: {avg_uts:.0f} MPa")
            print(f"    Fine structure: {np.sum(fine_zones)/n_voxels*100:.1f}%")
            print(f"    Coarse structure: {np.sum(coarse_zones)/n_voxels*100:.1f}%")
            print("="*60 + "\n")
        
        return self.result
    
    def predict_from_solidification(self,
                                     solidification_times: np.ndarray,
                                     show_progress: bool = True) -> MicrostructureResult:
        """
        Predict microstructure by first estimating cooling rates.
        
        Args:
            solidification_times: When each voxel solidified (seconds)
            show_progress: Print progress information
            
        Returns:
            MicrostructureResult with SDAS and property predictions
        """
        n_voxels = self.mesh.get_voxel_count()
        
        if show_progress:
            print("\n  Estimating cooling rates from solidification sequence...")
        
        neighbors = np.ascontiguousarray(self.mesh.voxel_neighbors, dtype=np.int32)
        
        cooling_rates = estimate_cooling_rate_from_solidification(
            np.ascontiguousarray(solidification_times, dtype=np.float64),
            neighbors,
            self.liquidus_temp,
            self.solidus_temp,
            self.mesh.base_voxel_size
        )
        
        if show_progress:
            print(f"    Cooling rate range: {np.min(cooling_rates):.2f} - {np.max(cooling_rates):.2f} °C/s")
        
        return self.predict(cooling_rates, show_progress)
    
    def get_property_variation_report(self) -> str:
        """
        Generate a report on property variation across the casting.
        
        Returns:
            Text report of property variation
        """
        if self.result is None:
            return "No results available. Run predict() first."
        
        lines = [
            "="*60,
            "MICROSTRUCTURE AND PROPERTY REPORT",
            "="*60,
            "",
            f"Alloy: {self.constants.name}",
            "",
            "SDAS (Secondary Dendrite Arm Spacing):",
            f"  Minimum: {self.result.min_sdas:.0f} μm (finest, surface/chill regions)",
            f"  Maximum: {self.result.max_sdas:.0f} μm (coarsest, thermal center)",
            f"  Average: {self.result.avg_sdas:.0f} μm",
            f"  Range: {self.result.max_sdas/self.result.min_sdas:.1f}x variation",
            "",
            "Predicted Mechanical Properties:",
            f"  UTS (avg): {self.result.avg_uts:.0f} MPa",
            f"  UTS (range): {np.min(self.result.predicted_uts):.0f} - {np.max(self.result.predicted_uts):.0f} MPa",
            f"  Elongation (avg): {np.mean(self.result.predicted_elongation):.1f}%",
            "",
            "Property Implications:",
            f"  - Fine structure regions ({np.sum(self.result.fine_structure_zones)} voxels): Higher strength",
            f"  - Coarse structure regions ({np.sum(self.result.coarse_structure_zones)} voxels): Lower strength",
            "",
            "Design Recommendations:",
            "  - Add chills to coarse regions for finer SDAS",
            "  - Expect lower fatigue life in coarse regions",
            "  - Section thickness affects cooling rate and SDAS",
            "="*60
        ]
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict:
        """Get microstructure statistics"""
        if self.result is None:
            return {}
        
        return {
            'min_sdas_um': self.result.min_sdas,
            'max_sdas_um': self.result.max_sdas,
            'avg_sdas_um': self.result.avg_sdas,
            'avg_uts_mpa': self.result.avg_uts,
            'min_uts_mpa': float(np.min(self.result.predicted_uts)),
            'max_uts_mpa': float(np.max(self.result.predicted_uts)),
            'fine_structure_fraction': float(np.mean(self.result.fine_structure_zones)),
            'coarse_structure_fraction': float(np.mean(self.result.coarse_structure_zones)),
            'alloy_constants': {
                'A': self.constants.A,
                'n': self.constants.n,
                'name': self.constants.name
            }
        }


# Test function
if __name__ == '__main__':
    print("Microstructure Prediction Module")
    print("="*60)
    print("This module predicts SDAS and mechanical properties from cooling rate.")
    print("\nAvailable alloy constants:")
    for key, const in SDAS_CONSTANTS.items():
        print(f"  {key}: A={const.A:.0f}, n={const.n:.2f} ({const.name})")
    
    print("\nKey features:")
    print("  - SDAS = A * cooling_rate^(-n)")
    print("  - UTS correlation with SDAS (Hall-Petch)")
    print("  - Fine/coarse structure zone identification")
