"""
Heat Transfer Module for Casting Solidification Simulation

Implements heat conduction and convection physics:
- Fourier's law of heat conduction
- Finite difference discretization
- Boundary conditions (mold interface, ambient)
- Temperature-dependent material properties

Mathematical Model:
    ∂T/∂t = α∇²T + Q/ρcₚ
    
where:
    T = temperature
    α = thermal diffusivity (k/ρcₚ)
    Q = heat source/sink (e.g., latent heat)
    ρ = density
    cₚ = specific heat

Usage:
    from physics.heat_transfer import HeatTransferModel, BoundaryType
    
    model = HeatTransferModel(mesh, alloy)
    model.add_boundary(surface_indices, BoundaryType.CONVECTION, h=10, T_inf=25)
    model.step(dt)
    temperatures = model.get_temperatures()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Union
import numpy as np
from numpy.typing import NDArray


# Type aliases
FloatArray = NDArray[Any]
IntArray = NDArray[Any]
BoolArray = NDArray[Any]


class BoundaryType(Enum):
    """Types of boundary conditions"""
    FIXED_TEMPERATURE = auto()  # Dirichlet: T = T_boundary
    HEAT_FLUX = auto()          # Neumann: -k∂T/∂n = q
    CONVECTION = auto()         # Robin: -k∂T/∂n = h(T - T_inf)
    ADIABATIC = auto()          # Neumann: ∂T/∂n = 0
    INTERFACE = auto()          # Mold-metal interface with IHTC


@dataclass
class BoundaryCondition:
    """
    Specification of a boundary condition.
    
    Attributes:
        type: Type of boundary condition
        indices: Voxel indices where this condition applies
        value: Condition-specific value (temperature, heat flux, etc.)
        htc: Heat transfer coefficient for convection/interface
        T_infinity: Ambient temperature for convection
        time_function: Optional time-dependent modifier
    """
    type: BoundaryType
    indices: IntArray
    value: float = 0.0
    htc: float = 0.0  # W/(m²·K)
    T_infinity: float = 25.0  # °C
    time_function: Optional[Callable[[float], float]] = None
    
    def get_effective_value(self, time: float) -> float:
        """Get value at given time (applies time function if present)"""
        if self.time_function is not None:
            return self.value * self.time_function(time)
        return self.value


@dataclass  
class TemperatureField:
    """
    Container for temperature field data.
    
    Provides convenient access to temperature statistics and gradients.
    """
    values: FloatArray
    time: float = 0.0
    
    @property
    def min(self) -> float:
        return float(np.min(self.values))
    
    @property
    def max(self) -> float:
        return float(np.max(self.values))
    
    @property
    def mean(self) -> float:
        return float(np.mean(self.values))
    
    @property
    def std(self) -> float:
        return float(np.std(self.values))
    
    def get_gradient_magnitude(self, neighbors: IntArray, voxel_size: float) -> FloatArray:
        """
        Calculate temperature gradient magnitude at each voxel.
        
        Uses central difference approximation.
        """
        if NUMBA_AVAILABLE:
            return compute_gradient_magnitude_parallel(
                np.asarray(self.values, dtype=np.float64),
                np.asarray(neighbors, dtype=np.int32),
                float(voxel_size),
            )

        n_voxels = len(self.values)
        gradients = np.zeros(n_voxels, dtype=np.float32)
        for i in range(n_voxels):
            grad_sq = 0.0
            count = 0
            for j in range(neighbors.shape[1]):
                neighbor_idx = neighbors[i, j]
                if neighbor_idx >= 0:
                    dT = self.values[neighbor_idx] - self.values[i]
                    grad_sq += (dT / voxel_size) ** 2
                    count += 1
            if count > 0:
                gradients[i] = np.sqrt(grad_sq / count)
        return gradients


class MaterialThermalProperties(Protocol):
    """Protocol for material thermal properties"""
    thermal_conductivity: float
    specific_heat: float
    density_liquid: float
    density_solid: float
    liquidus_temp: float
    solidus_temp: float
    latent_heat: float
    
    @property
    def thermal_diffusivity(self) -> float: ...


class HeatTransferModel:
    """
    Heat transfer model using finite differences.
    
    Solves the heat equation:
        ∂T/∂t = α∇²T + Q/(ρcₚ)
    
    Using explicit Euler method with adaptive time stepping.
    
    Features:
    - Multiple boundary condition types
    - Temperature-dependent properties (optional)
    - Latent heat release during solidification
    - Vectorized NumPy operations for performance
    
    Example:
        model = HeatTransferModel(mesh, alloy)
        model.set_initial_temperature(alloy.pouring_temp)
        model.add_boundary(
            surface_indices, 
            BoundaryType.CONVECTION,
            htc=500,  # W/(m²·K)
            T_infinity=200  # Mold temperature
        )
        
        while not model.is_solidified():
            model.step(dt)
            print(f"T_max = {model.temperature_field.max:.1f}°C")
    """
    
    def __init__(
        self,
        n_voxels: int,
        voxel_size: float,
        neighbors: IntArray,
        alloy: MaterialThermalProperties,
        use_temperature_dependent: bool = False
    ):
        """
        Initialize heat transfer model.
        
        Args:
            n_voxels: Number of voxels in mesh
            voxel_size: Voxel size in meters
            neighbors: Neighbor connectivity array (n_voxels x max_neighbors)
            alloy: Material thermal properties
            use_temperature_dependent: Use temperature-dependent k, cp
        """
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.neighbors = neighbors
        self.alloy = alloy
        self.use_temperature_dependent = use_temperature_dependent
        
        # Temperature field
        self._temperatures = np.zeros(n_voxels, dtype=np.float64)
        self._temperatures_new = np.zeros(n_voxels, dtype=np.float64)
        
        # Time tracking
        self.time = 0.0
        
        # Boundary conditions
        self._boundaries: List[BoundaryCondition] = []
        
        # Heat source (latent heat during solidification)
        self._heat_source = np.zeros(n_voxels, dtype=np.float64)
        
        # Precompute constants
        self._alpha = alloy.thermal_diffusivity  # m²/s
        self._dx2 = voxel_size ** 2
        self._numba_parallel_eligible = bool(NUMBA_AVAILABLE and not use_temperature_dependent)
        self._last_step_used_numba = False
        
        # Count neighbors for each voxel
        self._neighbor_counts = np.sum(neighbors >= 0, axis=1).astype(np.float64)

    @property
    def numba_parallel_eligible(self) -> bool:
        """Return True when the model can use Numba parallel kernels."""
        return self._numba_parallel_eligible

    @property
    def last_step_used_numba(self) -> bool:
        """Return whether the most recent step used Numba parallel kernels."""
        return bool(self._last_step_used_numba)
    
    @property
    def temperatures(self) -> FloatArray:
        """Current temperature field"""
        return self._temperatures
    
    @property
    def temperature_field(self) -> TemperatureField:
        """Get temperature field object"""
        return TemperatureField(self._temperatures.copy(), self.time)
    
    def set_initial_temperature(self, temperature: Union[float, FloatArray]) -> None:
        """
        Set initial temperature field.
        
        Args:
            temperature: Uniform temperature or per-voxel array
        """
        if isinstance(temperature, (int, float)):
            self._temperatures.fill(temperature)
        else:
            self._temperatures[:] = temperature
    
    def add_boundary(
        self,
        indices: IntArray,
        boundary_type: BoundaryType,
        value: float = 0.0,
        htc: float = 0.0,
        T_infinity: float = 25.0,
        time_function: Optional[Callable[[float], float]] = None
    ) -> None:
        """
        Add a boundary condition.
        
        Args:
            indices: Voxel indices where condition applies
            boundary_type: Type of boundary condition
            value: Temperature (FIXED) or heat flux (HEAT_FLUX)
            htc: Heat transfer coefficient for CONVECTION/INTERFACE
            T_infinity: Ambient temperature for CONVECTION
            time_function: Optional time-dependent multiplier
        """
        bc = BoundaryCondition(
            type=boundary_type,
            indices=np.asarray(indices, dtype=np.int32),
            value=value,
            htc=htc,
            T_infinity=T_infinity,
            time_function=time_function
        )
        self._boundaries.append(bc)
    
    def set_heat_source(self, source: FloatArray) -> None:
        """
        Set volumetric heat source term (W/m³).
        
        Typically used for latent heat release during solidification.
        """
        self._heat_source[:] = source
    
    def get_stable_timestep(self) -> float:
        """
        Calculate maximum stable time step for explicit scheme.
        
        Stability criterion (3D):
            dt < dx² / (6α)
        """
        return 0.9 * self._dx2 / (6.0 * self._alpha)

    def get_gradient_magnitude(self, neighbors: IntArray, voxel_size: float) -> FloatArray:
        """Return gradient magnitude from the current thermal field without copying."""
        neighbor_arr = np.asarray(neighbors, dtype=np.int32)
        if NUMBA_AVAILABLE:
            return compute_gradient_magnitude_parallel(
                np.asarray(self._temperatures, dtype=np.float64),
                neighbor_arr,
                float(voxel_size),
            )

        values = np.asarray(self._temperatures, dtype=np.float64)
        n_voxels = len(values)
        gradients = np.zeros(n_voxels, dtype=np.float32)
        for i in range(n_voxels):
            grad_sq = 0.0
            count = 0
            for j in range(neighbor_arr.shape[1]):
                neighbor_idx = neighbor_arr[i, j]
                if neighbor_idx >= 0:
                    dT = values[neighbor_idx] - values[i]
                    grad_sq += (dT / voxel_size) ** 2
                    count += 1
            if count > 0:
                gradients[i] = np.sqrt(grad_sq / count)
        return gradients
    
    def step(self, dt: float) -> None:
        """
        Advance temperature field by one time step.
        
        Uses explicit finite difference method.
        
        Args:
            dt: Time step in seconds
        """
        # Check stability
        dt_stable = self.get_stable_timestep()
        if dt > dt_stable:
            # Subdivide step
            n_substeps = int(np.ceil(dt / dt_stable))
            dt_sub = dt / n_substeps
            for _ in range(n_substeps):
                self._step_internal(dt_sub)
        else:
            self._step_internal(dt)
        
        self.time += dt
    
    def _step_internal(self, dt: float) -> None:
        """Internal time stepping (assumes dt is stable)"""
        T = self._temperatures
        T_new = self._temperatures_new
        
        # Get thermal diffusivity (potentially temperature-dependent)
        if self.use_temperature_dependent:
            alpha = self._get_temperature_dependent_diffusivity(T)
        else:
            alpha = self._alpha
        use_numba_kernels = bool(
            self._numba_parallel_eligible
            and not isinstance(alpha, np.ndarray)
        )
        self._last_step_used_numba = use_numba_kernels

        if use_numba_kernels:
            laplacian = compute_laplacian_parallel(
                T,
                self.neighbors,
                self._neighbor_counts,
                self._dx2,
            )
        else:
            # Compute Laplacian using neighbor stencil (vectorized)
            # ∇²T ≈ Σ(T_neighbor - T_i) / (n * dx²)
            laplacian = np.zeros(self.n_voxels, dtype=np.float64)

            for j in range(self.neighbors.shape[1]):
                neighbor_indices = self.neighbors[:, j]
                valid = neighbor_indices >= 0

                # T_neighbor - T for valid neighbors
                dT = np.where(
                    valid,
                    T[np.maximum(neighbor_indices, 0)] - T,
                    0.0
                )
                laplacian += dT

            # Normalize by neighbor count and dx²
            laplacian = np.where(
                self._neighbor_counts > 0,
                laplacian / (self._neighbor_counts * self._dx2),
                0.0
            )
        
        # Heat source term: Q / (ρ * cp)
        rho_cp = self.alloy.density_liquid * self.alloy.specific_heat
        source_term = self._heat_source / rho_cp
        
        # Update: T_new = T + dt * (α∇²T + Q/ρcp)
        if use_numba_kernels:
            T_new[:] = update_temperatures_parallel(
                T,
                laplacian,
                source_term,
                float(alpha),
                dt,
            )
        elif isinstance(alpha, np.ndarray):
            T_new[:] = T + dt * (alpha * laplacian + source_term)
        else:
            T_new[:] = T + dt * (alpha * laplacian + source_term)
        
        # Apply boundary conditions
        self._apply_boundaries(T_new, dt)
        
        # Swap arrays
        self._temperatures, self._temperatures_new = self._temperatures_new, self._temperatures
    
    def _apply_boundaries(self, T: FloatArray, dt: float) -> None:
        """Apply all boundary conditions"""
        for bc in self._boundaries:
            if bc.type == BoundaryType.FIXED_TEMPERATURE:
                # Dirichlet: T = T_boundary
                T[bc.indices] = bc.get_effective_value(self.time)
                
            elif bc.type == BoundaryType.CONVECTION:
                # Convective boundary: -k∂T/∂n = h(T - T_inf)
                # Implemented as source term
                h = bc.htc
                T_inf = bc.T_infinity
                k = self.alloy.thermal_conductivity
                
                # Heat loss rate: h * (T - T_inf) / (dx/2) ~ boundary flux
                # Apply as temperature change
                T_boundary = T[bc.indices]
                flux = h * (T_boundary - T_inf)  # W/m²
                dT = -flux * dt / (self.alloy.density_liquid * 
                                   self.alloy.specific_heat * 
                                   self.voxel_size / 2)
                T[bc.indices] += dT
                
            elif bc.type == BoundaryType.INTERFACE:
                # Mold-metal interface with IHTC
                # Similar to convection but may have time-dependent htc
                h = bc.htc * (bc.time_function(self.time) if bc.time_function else 1.0)
                T_mold = bc.T_infinity
                
                T_boundary = T[bc.indices]
                flux = h * (T_boundary - T_mold)
                dT = -flux * dt / (self.alloy.density_liquid * 
                                   self.alloy.specific_heat * 
                                   self.voxel_size / 2)
                T[bc.indices] += dT
                
            elif bc.type == BoundaryType.ADIABATIC:
                # No-flux boundary: ∂T/∂n = 0
                # Nothing to do (handled by finite difference stencil)
                pass
    
    def _get_temperature_dependent_diffusivity(self, T: FloatArray) -> FloatArray:
        """
        Get temperature-dependent thermal diffusivity.
        
        Simple linear interpolation between solid and liquid values.
        """
        T_solidus = self.alloy.solidus_temp
        T_liquidus = self.alloy.liquidus_temp
        
        # Fraction liquid
        f_liquid = np.clip(
            (T - T_solidus) / (T_liquidus - T_solidus),
            0.0, 1.0
        )
        
        # Assume liquid diffusivity is lower
        alpha_solid = self._alpha
        alpha_liquid = self._alpha * 0.7
        
        return alpha_solid * (1 - f_liquid) + alpha_liquid * f_liquid
    
    def is_solidified(self, threshold_temp: Optional[float] = None) -> bool:
        """
        Check if all voxels are below solidus temperature.
        
        Args:
            threshold_temp: Temperature threshold (default: solidus)
        """
        threshold = threshold_temp or self.alloy.solidus_temp
        return np.all(self._temperatures <= threshold)
    
    def get_cooling_rate(self, prev_temperatures: FloatArray, dt: float) -> FloatArray:
        """
        Calculate cooling rate at each voxel.
        
        Args:
            prev_temperatures: Temperature field from previous step
            dt: Time step
            
        Returns:
            Cooling rate in °C/s
        """
        return (prev_temperatures - self._temperatures) / dt


# ============================================================================
# Numba-accelerated kernels (optional)
# ============================================================================

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def compute_laplacian_parallel(
        T: FloatArray,
        neighbors: IntArray,
        neighbor_counts: FloatArray,
        dx2: float
    ) -> FloatArray:
        """
        Compute discrete Laplacian in parallel.
        
        Uses Numba for acceleration.
        """
        n = len(T)
        laplacian = np.zeros(n, dtype=np.float64)
        
        for i in prange(n):
            sum_dT = 0.0
            for j in range(neighbors.shape[1]):
                neighbor_idx = neighbors[i, j]
                if neighbor_idx >= 0:
                    sum_dT += T[neighbor_idx] - T[i]
            
            if neighbor_counts[i] > 0:
                laplacian[i] = sum_dT / (neighbor_counts[i] * dx2)
        
        return laplacian
    
    @njit(parallel=True, cache=True)
    def update_temperatures_parallel(
        T: FloatArray,
        laplacian: FloatArray,
        source: FloatArray,
        alpha: float,
        dt: float
    ) -> FloatArray:
        """
        Update temperature field in parallel.
        """
        n = len(T)
        T_new = np.empty(n, dtype=np.float64)
        
        for i in prange(n):
            T_new[i] = T[i] + dt * (alpha * laplacian[i] + source[i])
        
        return T_new

    @njit(parallel=True, cache=True)
    def compute_gradient_magnitude_parallel(
        values: FloatArray,
        neighbors: IntArray,
        voxel_size: float,
    ) -> FloatArray:
        """Compute temperature gradient magnitude in parallel."""
        n = len(values)
        gradients = np.zeros(n, dtype=np.float32)
        inv_h = 1.0 / max(voxel_size, 1e-12)

        for i in prange(n):
            grad_sq = 0.0
            count = 0
            base = values[i]
            for j in range(neighbors.shape[1]):
                neighbor_idx = neighbors[i, j]
                if neighbor_idx >= 0:
                    dT = (values[neighbor_idx] - base) * inv_h
                    grad_sq += dT * dT
                    count += 1
            if count > 0:
                gradients[i] = np.sqrt(grad_sq / count)

        return gradients
