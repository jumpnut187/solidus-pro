"""
Niyama Criterion Module for Casting Simulation

Implements the Niyama criterion for micro-shrinkage prediction:
    Ny = G / √(dT/dt)
    
where:
    G = temperature gradient (°C/m or K/m)
    dT/dt = cooling rate (°C/s or K/s)

Critical threshold: Ny < 1.0 typically indicates porosity risk

Physical Basis:
    - High thermal gradient promotes directional solidification
    - Low cooling rate allows time for interdendritic feeding
    - Low Niyama (low G, high cooling rate) = poor feeding = porosity

Usage:
    from physics.niyama import NiyamaModel
    
    model = NiyamaModel(mesh, alloy, voxel_size)
    niyama_values = model.calculate(temperatures, prev_temperatures, dt)
    porosity_risk = model.get_porosity_risk()

Status:
    This module is retained as a reference/experimental thermal-field path.
    The application's authoritative user-visible Niyama outputs are generated
    in ``core/solidification.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional, Protocol, Tuple
import numpy as np
from numpy.typing import NDArray


# Type aliases
FloatArray = NDArray[Any]
IntArray = NDArray[Any]
BoolArray = NDArray[Any]


class PorosityRisk(Enum):
    """Porosity risk levels based on Niyama criterion"""
    LOW = auto()      # Ny > 3.0
    MODERATE = auto() # 1.0 < Ny <= 3.0
    HIGH = auto()     # 0.5 < Ny <= 1.0
    SEVERE = auto()   # Ny <= 0.5


@dataclass
class NiyamaResult:
    """
    Container for Niyama analysis results.
    
    Attributes:
        values: Niyama criterion values for each voxel
        thermal_gradient: Temperature gradient magnitude
        cooling_rate: Cooling rate (positive = cooling)
        risk_levels: Porosity risk classification
        critical_indices: Voxels with Ny below critical threshold
    """
    values: FloatArray
    thermal_gradient: FloatArray
    cooling_rate: FloatArray
    risk_levels: List[PorosityRisk]
    critical_indices: IntArray
    
    @property
    def min(self) -> float:
        """Minimum Niyama value (highest risk)"""
        valid = self.values[np.isfinite(self.values)]
        return float(np.min(valid)) if len(valid) > 0 else float('inf')
    
    @property
    def mean(self) -> float:
        """Mean Niyama value"""
        valid = self.values[np.isfinite(self.values)]
        return float(np.mean(valid)) if len(valid) > 0 else float('inf')
    
    @property
    def porosity_fraction(self) -> float:
        """Fraction of voxels with high porosity risk"""
        return len(self.critical_indices) / len(self.values)


@dataclass
class NiyamaThresholds:
    """
    Threshold values for Niyama criterion interpretation.
    
    Default values are for aluminum alloys.
    Steel typically uses Ny_critical ~ 1.0 (√K·s)/m
    Aluminum typically uses Ny_critical ~ 0.5-1.0 (√K·s)/m
    """
    critical: float = 1.0      # Below this = high porosity risk
    severe: float = 0.5        # Below this = severe porosity
    moderate: float = 3.0      # Below this = moderate risk
    safe: float = 5.0          # Above this = generally safe


class MaterialNiyamaProperties(Protocol):
    """Protocol for material properties needed for Niyama"""
    liquidus_temp: float
    solidus_temp: float


class NiyamaModel:
    """
    Niyama criterion model for micro-shrinkage prediction.
    
    The Niyama criterion predicts micro-porosity by relating:
    - Thermal gradient G: drives directional solidification
    - Cooling rate Ṫ: affects interdendritic feeding time
    
    High Niyama values indicate good feeding conditions.
    Low values indicate porosity risk.
    
    Features:
    - Gradient calculation using finite differences
    - Cooling rate calculation from temperature history
    - Risk classification based on thresholds
    - Mushy zone filtering (criterion only meaningful in mushy zone)
    
    Example:
        model = NiyamaModel(n_voxels, neighbors, voxel_size, alloy)
        
        # In simulation loop:
        niyama = model.calculate(T_current, T_previous, dt)
        result = model.get_result()
        
        for idx in result.critical_indices:
            print(f"Porosity risk at voxel {idx}: Ny = {result.values[idx]:.2f}")
    """
    
    def __init__(
        self,
        n_voxels: int,
        neighbors: IntArray,
        voxel_size: float,
        alloy: MaterialNiyamaProperties,
        thresholds: Optional[NiyamaThresholds] = None
    ):
        """
        Initialize Niyama model.
        
        Args:
            n_voxels: Number of voxels
            neighbors: Neighbor connectivity (n_voxels x max_neighbors)
            voxel_size: Voxel size in meters
            alloy: Material properties
            thresholds: Custom threshold values
        """
        self.n_voxels = n_voxels
        self.neighbors = neighbors
        self.voxel_size = voxel_size
        self.alloy = alloy
        self.thresholds = thresholds or NiyamaThresholds()
        
        # Cache temperature range
        self.T_liq = alloy.liquidus_temp
        self.T_sol = alloy.solidus_temp
        
        # Results
        self._niyama: Optional[FloatArray] = None
        self._gradient: Optional[FloatArray] = None
        self._cooling_rate: Optional[FloatArray] = None
        self._result: Optional[NiyamaResult] = None
        
        # Rolling average for smoother cooling rate
        self._cooling_rate_history: List[FloatArray] = []
        self._history_length = 5
    
    @property
    def niyama_values(self) -> Optional[FloatArray]:
        """Current Niyama values"""
        return self._niyama
    
    @property
    def result(self) -> Optional[NiyamaResult]:
        """Latest analysis result"""
        return self._result
    
    def calculate(
        self,
        temperatures: FloatArray,
        prev_temperatures: FloatArray,
        dt: float,
        fraction_solid: Optional[FloatArray] = None
    ) -> FloatArray:
        """
        Calculate Niyama criterion.
        
        Ny = G / √(|dT/dt|)
        
        Args:
            temperatures: Current temperature field
            prev_temperatures: Previous temperature field
            dt: Time step
            fraction_solid: Optional fraction solid for mushy zone filtering
            
        Returns:
            Niyama values for each voxel
        """
        # Calculate thermal gradient
        self._gradient = self._calculate_gradient(temperatures)
        
        # Calculate cooling rate (positive = cooling)
        cooling_rate = (prev_temperatures - temperatures) / dt
        
        # Rolling average for stability
        self._cooling_rate_history.append(cooling_rate)
        if len(self._cooling_rate_history) > self._history_length:
            self._cooling_rate_history.pop(0)
        
        self._cooling_rate = np.mean(self._cooling_rate_history, axis=0)
        
        # Niyama criterion
        # Use absolute value of cooling rate (can be negative during recalescence)
        cooling_rate_abs = np.abs(self._cooling_rate)
        
        # Avoid division by zero
        cooling_rate_safe = np.maximum(cooling_rate_abs, 1e-6)
        
        self._niyama = self._gradient / np.sqrt(cooling_rate_safe)
        
        # Set to infinity where not applicable
        # (liquid phase, solid phase, very low cooling rate)
        if fraction_solid is not None:
            # Only calculate in mushy zone (roughly 0.1 < fs < 0.9)
            not_mushy = (fraction_solid < 0.1) | (fraction_solid > 0.9)
            self._niyama[not_mushy] = np.inf
        else:
            # Use temperature to estimate mushy zone
            not_mushy = (temperatures > self.T_liq) | (temperatures < self.T_sol)
            self._niyama[not_mushy] = np.inf
        
        # Where gradient is ~0 (uniform temperature)
        self._niyama[self._gradient < 1e-3] = np.inf
        
        # Build result
        self._result = self._build_result()
        
        return self._niyama
    
    def _calculate_gradient(self, temperatures: FloatArray) -> FloatArray:
        """
        Calculate temperature gradient magnitude.
        
        Uses central difference with neighbors:
        |∇T| ≈ √(Σ((T_neighbor - T)²) / (n * dx²))
        """
        gradient = np.zeros(self.n_voxels, dtype=np.float64)
        
        for i in range(self.n_voxels):
            grad_sq_sum = 0.0
            count = 0
            
            for j in range(self.neighbors.shape[1]):
                neighbor = self.neighbors[i, j]
                if neighbor >= 0:
                    dT = temperatures[neighbor] - temperatures[i]
                    grad_sq_sum += (dT / self.voxel_size) ** 2
                    count += 1
            
            if count > 0:
                # RMS gradient
                gradient[i] = np.sqrt(grad_sq_sum / count)
        
        return gradient
    
    def _build_result(self) -> NiyamaResult:
        """Build result object from current state"""
        # Classify risk levels
        risk_levels = []
        for ny in self._niyama:
            if not np.isfinite(ny):
                risk_levels.append(PorosityRisk.LOW)  # Not applicable
            elif ny <= self.thresholds.severe:
                risk_levels.append(PorosityRisk.SEVERE)
            elif ny <= self.thresholds.critical:
                risk_levels.append(PorosityRisk.HIGH)
            elif ny <= self.thresholds.moderate:
                risk_levels.append(PorosityRisk.MODERATE)
            else:
                risk_levels.append(PorosityRisk.LOW)
        
        # Find critical voxels
        valid_niyama = np.where(np.isfinite(self._niyama), self._niyama, np.inf)
        critical = valid_niyama <= self.thresholds.critical
        critical_indices = np.where(critical)[0].astype(np.int32)
        
        return NiyamaResult(
            values=self._niyama.copy(),
            thermal_gradient=self._gradient.copy(),
            cooling_rate=self._cooling_rate.copy(),
            risk_levels=risk_levels,
            critical_indices=critical_indices
        )
    
    def get_porosity_probability(self) -> FloatArray:
        """
        Convert Niyama values to porosity probability.
        
        Uses empirical correlation:
        P = exp(-α * Ny) for Ny < Ny_critical
        P = 0 for Ny >= Ny_critical
        
        Returns:
            Probability of porosity formation (0-1)
        """
        if self._niyama is None:
            return np.zeros(self.n_voxels)
        
        # Empirical decay constant
        alpha = 2.0
        
        prob = np.zeros(self.n_voxels, dtype=np.float64)
        
        for i in range(self.n_voxels):
            ny = self._niyama[i]
            if np.isfinite(ny) and ny < self.thresholds.critical:
                prob[i] = np.exp(-alpha * ny)
        
        return prob
    
    def get_expected_porosity_volume_fraction(
        self,
        volumetric_shrinkage: float = 0.04
    ) -> FloatArray:
        """
        Estimate porosity volume fraction at each voxel.
        
        Based on probability and volumetric shrinkage.
        
        Args:
            volumetric_shrinkage: Alloy's volumetric shrinkage (typically 3-7%)
            
        Returns:
            Expected porosity volume fraction
        """
        probability = self.get_porosity_probability()
        
        # Maximum porosity is limited by shrinkage
        # Actual porosity depends on local feeding conditions
        return probability * volumetric_shrinkage * 0.5  # Conservative estimate


class ThermalGradientCalculator:
    """
    Utility class for thermal gradient calculations.
    
    Provides various gradient calculation methods.
    """
    
    @staticmethod
    def central_difference(
        temperatures: FloatArray,
        neighbors: IntArray,
        dx: float
    ) -> FloatArray:
        """
        Calculate gradient using central differences.
        """
        n = len(temperatures)
        gradient = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            grad_sq = 0.0
            count = 0
            
            for j in range(neighbors.shape[1]):
                neighbor = neighbors[i, j]
                if neighbor >= 0:
                    dT = temperatures[neighbor] - temperatures[i]
                    grad_sq += (dT / dx) ** 2
                    count += 1
            
            if count > 0:
                gradient[i] = np.sqrt(grad_sq / count)
        
        return gradient
    
    @staticmethod
    def directional_gradient(
        temperatures: FloatArray,
        neighbors: IntArray,
        neighbor_directions: FloatArray,
        dx: float
    ) -> Tuple[FloatArray, FloatArray]:
        """
        Calculate gradient magnitude and direction.
        
        Args:
            temperatures: Temperature field
            neighbors: Neighbor indices
            neighbor_directions: Unit vectors to neighbors (n x max_neighbors x 3)
            dx: Grid spacing
            
        Returns:
            Tuple of (gradient magnitude, gradient direction vectors)
        """
        n = len(temperatures)
        gradient_mag = np.zeros(n, dtype=np.float64)
        gradient_dir = np.zeros((n, 3), dtype=np.float64)
        
        for i in range(n):
            grad_vec = np.zeros(3)
            
            for j in range(neighbors.shape[1]):
                neighbor = neighbors[i, j]
                if neighbor >= 0:
                    dT = temperatures[neighbor] - temperatures[i]
                    grad_vec += dT * neighbor_directions[i, j] / dx
            
            gradient_mag[i] = np.linalg.norm(grad_vec)
            if gradient_mag[i] > 1e-10:
                gradient_dir[i] = grad_vec / gradient_mag[i]
        
        return gradient_mag, gradient_dir
    
    @staticmethod
    def smooth_gradient(
        temperatures: FloatArray,
        neighbors: IntArray,
        dx: float,
        smoothing_iterations: int = 2
    ) -> FloatArray:
        """
        Calculate smoothed gradient (reduces numerical noise).
        
        Applies Laplacian smoothing to temperature field first.
        """
        # Smooth temperatures
        T_smooth = temperatures.copy()
        
        for _ in range(smoothing_iterations):
            T_new = T_smooth.copy()
            for i in range(len(temperatures)):
                neighbor_sum = 0.0
                count = 0
                for j in range(neighbors.shape[1]):
                    neighbor = neighbors[i, j]
                    if neighbor >= 0:
                        neighbor_sum += T_smooth[neighbor]
                        count += 1
                if count > 0:
                    T_new[i] = 0.5 * T_smooth[i] + 0.5 * neighbor_sum / count
            T_smooth = T_new
        
        # Calculate gradient on smoothed field
        return ThermalGradientCalculator.central_difference(T_smooth, neighbors, dx)


class NiyamaCriterion:
    """
    Static methods for Niyama criterion calculations.
    
    Can be used without instantiating full model.
    """
    
    @staticmethod
    def calculate(
        gradient: float,
        cooling_rate: float,
        min_cooling_rate: float = 1e-6
    ) -> float:
        """
        Calculate single Niyama value.
        
        Ny = G / √(|Ṫ|)
        
        Args:
            gradient: Temperature gradient (K/m)
            cooling_rate: Cooling rate (K/s)
            min_cooling_rate: Minimum cooling rate to avoid infinity
            
        Returns:
            Niyama criterion value (K^0.5 · s^0.5 / m)
        """
        cooling_abs = max(abs(cooling_rate), min_cooling_rate)
        return gradient / np.sqrt(cooling_abs)
    
    @staticmethod
    def is_porosity_risk(
        niyama: float,
        threshold: float = 1.0
    ) -> bool:
        """Check if Niyama value indicates porosity risk"""
        return np.isfinite(niyama) and niyama < threshold
    
    @staticmethod
    def get_risk_level(
        niyama: float,
        thresholds: Optional[NiyamaThresholds] = None
    ) -> PorosityRisk:
        """Get risk classification for Niyama value"""
        thresh = thresholds or NiyamaThresholds()
        
        if not np.isfinite(niyama):
            return PorosityRisk.LOW
        elif niyama <= thresh.severe:
            return PorosityRisk.SEVERE
        elif niyama <= thresh.critical:
            return PorosityRisk.HIGH
        elif niyama <= thresh.moderate:
            return PorosityRisk.MODERATE
        else:
            return PorosityRisk.LOW
    
    @staticmethod
    def optimal_cooling_rate(
        gradient: float,
        target_niyama: float = 2.0
    ) -> float:
        """
        Calculate optimal cooling rate for target Niyama.
        
        Ṫ = (G / Ny)²
        
        Args:
            gradient: Temperature gradient
            target_niyama: Desired Niyama value
            
        Returns:
            Required cooling rate
        """
        return (gradient / target_niyama) ** 2


# ============================================================================
# Numba-accelerated functions
# ============================================================================

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def calculate_gradient_parallel(
        temperatures: FloatArray,
        neighbors: IntArray,
        dx: float
    ) -> FloatArray:
        """
        Calculate thermal gradient in parallel using Numba.
        """
        n = len(temperatures)
        gradient = np.zeros(n, dtype=np.float64)
        
        for i in prange(n):
            grad_sq_sum = 0.0
            count = 0
            
            for j in range(neighbors.shape[1]):
                neighbor = neighbors[i, j]
                if neighbor >= 0:
                    dT = temperatures[neighbor] - temperatures[i]
                    grad_sq_sum += (dT / dx) ** 2
                    count += 1
            
            if count > 0:
                gradient[i] = np.sqrt(grad_sq_sum / count)
        
        return gradient
    
    @njit(parallel=True, cache=True)
    def calculate_niyama_parallel(
        gradient: FloatArray,
        cooling_rate: FloatArray,
        fraction_solid: FloatArray,
        fs_min: float = 0.1,
        fs_max: float = 0.9,
        min_cooling_rate: float = 1e-6
    ) -> FloatArray:
        """
        Calculate Niyama criterion in parallel.
        
        Only computes for voxels in mushy zone.
        """
        n = len(gradient)
        niyama = np.full(n, np.inf, dtype=np.float64)
        
        for i in prange(n):
            fs = fraction_solid[i]
            
            # Only in mushy zone
            if fs < fs_min or fs > fs_max:
                continue
            
            # Skip if gradient too low
            if gradient[i] < 1e-3:
                continue
            
            # Calculate Niyama
            cr = max(abs(cooling_rate[i]), min_cooling_rate)
            niyama[i] = gradient[i] / np.sqrt(cr)
        
        return niyama
    
    @njit(parallel=True, cache=True)
    def find_critical_voxels_parallel(
        niyama: FloatArray,
        threshold: float
    ) -> IntArray:
        """
        Find indices of voxels below Niyama threshold (parallel).
        """
        n = len(niyama)
        
        # Count critical voxels
        count = 0
        for i in range(n):
            if niyama[i] <= threshold and np.isfinite(niyama[i]):
                count += 1
        
        # Fill result array
        result = np.empty(count, dtype=np.int32)
        idx = 0
        for i in range(n):
            if niyama[i] <= threshold and np.isfinite(niyama[i]):
                result[idx] = i
                idx += 1
        
        return result
