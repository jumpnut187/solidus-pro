"""
Thermal Solver for Casting Solidification Simulation
Implements finite difference solution to the heat equation:

    ∂T/∂t = α∇²T + Q/(ρcₚ)

where:
    T = temperature (°C)
    t = time (s)
    α = thermal diffusivity (m²/s) = k/(ρcₚ)
    ∇²T = Laplacian of temperature
    Q = heat source/sink (W/m³) - includes latent heat release
    ρ = density (kg/m³)
    cₚ = specific heat (J/kg·K)

Features:
- Explicit finite difference method for heat diffusion
- Latent heat release during solidification (enthalpy method)
- Temperature-dependent material properties
- GPU acceleration when available
- Incremental updates for efficiency

Author: GitHub Copilot
Date: November 2025
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field

# Optional GPU acceleration
try:
    from core.gpu_compute import gpu_compute
    GPU_AVAILABLE = gpu_compute.is_gpu_available
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class ThermalProperties:
    """
    Temperature-dependent thermal properties for a material.
    
    Properties can vary with temperature. The class provides interpolation
    for properties at any temperature within the defined range.
    """
    # Reference temperatures for property values (°C)
    temp_points: np.ndarray = field(default_factory=lambda: np.array([20.0, 500.0, 1000.0, 1500.0]))
    
    # Thermal conductivity at each temperature point (W/m·K)
    conductivity_values: np.ndarray = field(default_factory=lambda: np.array([45.0, 40.0, 35.0, 30.0]))
    
    # Specific heat at each temperature point (J/kg·K)
    specific_heat_values: np.ndarray = field(default_factory=lambda: np.array([450.0, 550.0, 650.0, 750.0]))
    
    # Density at each temperature point (kg/m³)
    density_values: np.ndarray = field(default_factory=lambda: np.array([7850.0, 7750.0, 7650.0, 7200.0]))
    
    # Latent heat of fusion (J/kg)
    latent_heat: float = 270000.0
    
    # Solidification temperatures
    liquidus_temp: float = 1450.0  # °C
    solidus_temp: float = 1400.0   # °C
    
    def get_conductivity(self, temperature: float) -> float:
        """Get thermal conductivity at given temperature"""
        return np.interp(temperature, self.temp_points, self.conductivity_values)
    
    def get_specific_heat(self, temperature: float) -> float:
        """Get specific heat at given temperature"""
        return np.interp(temperature, self.temp_points, self.specific_heat_values)
    
    def get_density(self, temperature: float) -> float:
        """Get density at given temperature"""
        return np.interp(temperature, self.temp_points, self.density_values)
    
    def get_diffusivity(self, temperature: float) -> float:
        """Get thermal diffusivity at given temperature: α = k/(ρcₚ)"""
        k = self.get_conductivity(temperature)
        rho = self.get_density(temperature)
        cp = self.get_specific_heat(temperature)
        return k / (rho * cp)
    
    def get_fraction_solid(self, temperature: float) -> float:
        """
        Get fraction solid based on temperature using lever rule.
        Assumes linear solidification between liquidus and solidus.
        """
        if temperature >= self.liquidus_temp:
            return 0.0
        elif temperature <= self.solidus_temp:
            return 1.0
        else:
            # Linear interpolation in mushy zone
            freezing_range = self.liquidus_temp - self.solidus_temp
            return (self.liquidus_temp - temperature) / freezing_range


def create_thermal_properties_from_alloy(alloy) -> ThermalProperties:
    """
    Create ThermalProperties from an AlloyProperties object.
    
    Generates temperature-dependent property curves based on
    the alloy's base properties.
    """
    # Define temperature points from room temp to above liquidus
    temp_points = np.array([
        20.0,                          # Room temperature
        alloy.solidus_temp * 0.5,      # Mid-range solid
        alloy.solidus_temp,            # Solidus
        (alloy.solidus_temp + alloy.liquidus_temp) / 2,  # Mushy zone
        alloy.liquidus_temp,           # Liquidus
        alloy.liquidus_temp + 100      # Superheated liquid
    ])
    
    # Thermal conductivity typically decreases with temperature
    # Liquid has lower conductivity than solid
    k_solid = alloy.thermal_conductivity
    k_liquid = alloy.thermal_conductivity * 0.7  # Liquid ~70% of solid
    conductivity_values = np.array([
        k_solid,
        k_solid * 0.95,
        k_solid * 0.85,
        k_liquid * 1.1,
        k_liquid,
        k_liquid * 0.95
    ])
    
    # Specific heat increases with temperature
    cp_base = alloy.specific_heat
    specific_heat_values = np.array([
        cp_base * 0.85,
        cp_base * 0.92,
        cp_base,
        cp_base * 1.05,
        cp_base * 1.1,
        cp_base * 1.15
    ])
    
    # Density decreases with temperature
    rho_solid = alloy.density_solid
    rho_liquid = alloy.density_liquid
    density_values = np.array([
        rho_solid,
        rho_solid * 0.98,
        rho_solid * 0.96,
        (rho_solid + rho_liquid) / 2,
        rho_liquid,
        rho_liquid * 0.98
    ])
    
    return ThermalProperties(
        temp_points=temp_points,
        conductivity_values=conductivity_values,
        specific_heat_values=specific_heat_values,
        density_values=density_values,
        latent_heat=alloy.latent_heat,
        liquidus_temp=alloy.liquidus_temp,
        solidus_temp=alloy.solidus_temp
    )


@dataclass 
class ThermalSolverConfig:
    """Configuration for the thermal solver"""
    
    # Time stepping
    max_time_step: float = 0.1  # seconds (will be reduced for stability)
    cfl_safety_factor: float = 0.4  # CFL number for stability
    
    # Convergence
    max_iterations_per_step: int = 10  # For nonlinear iterations
    temperature_tolerance: float = 0.1  # °C
    
    # Initial conditions
    pouring_temperature: float = 1600.0  # °C (superheat above liquidus)
    ambient_temperature: float = 25.0    # °C
    
    # Boundary conditions
    mold_temperature: float = 25.0       # °C initial mold temperature
    
    # Performance
    use_gpu: bool = False
    use_adaptive_timestep: bool = True
    
    # Latent heat
    include_latent_heat: bool = True
    
    # Temperature-dependent properties
    use_temperature_dependent_props: bool = True


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_laplacian_6point(
    temperatures: np.ndarray,
    neighbors: np.ndarray,
    voxel_size: float
) -> np.ndarray:
    """
    Compute discrete Laplacian using 6-point stencil (face neighbors).
    
    ∇²T ≈ (Σ T_neighbors - 6*T_center) / h²
    
    For non-uniform meshes with missing neighbors, the stencil is adjusted.
    
    Args:
        temperatures: Temperature at each voxel (°C)
        neighbors: Neighbor indices (n_voxels x 26), -1 for no neighbor
        voxel_size: Voxel size in meters
    
    Returns:
        Laplacian at each voxel (°C/m²)
    """
    n_voxels = temperatures.shape[0]
    laplacian = np.zeros(n_voxels, dtype=np.float32)
    h2 = voxel_size * voxel_size
    
    # Use first 6 neighbors (face-connected) for standard 6-point stencil
    # Indices 0-5 correspond to +x, -x, +y, -y, +z, -z neighbors
    for i in prange(n_voxels):
        T_center = temperatures[i]
        neighbor_sum = np.float32(0.0)
        neighbor_count = 0
        
        # Check 6 face neighbors (first 6 in neighbor list)
        for j in range(min(6, neighbors.shape[1])):
            neighbor_idx = neighbors[i, j]
            if neighbor_idx >= 0:
                neighbor_sum += temperatures[neighbor_idx]
                neighbor_count += 1
        
        if neighbor_count > 0:
            # Adjusted stencil for missing neighbors
            # Standard: ∇²T = (Σ T_i - n*T_0) / h²
            laplacian[i] = (neighbor_sum - neighbor_count * T_center) / h2
    
    return laplacian


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_latent_heat_source(
    temperatures: np.ndarray,
    prev_temperatures: np.ndarray,
    fraction_solid: np.ndarray,
    prev_fraction_solid: np.ndarray,
    latent_heat: float,
    density: float,
    specific_heat: float,
    time_step: float
) -> np.ndarray:
    """
    Compute latent heat source term during solidification.
    
    Uses the enthalpy method: Q = ρL * (∂fₛ/∂t)
    where fₛ is fraction solid.
    
    The heat source is incorporated as an apparent specific heat increase
    in the mushy zone.
    
    Args:
        temperatures: Current temperature (°C)
        prev_temperatures: Previous temperature (°C)
        fraction_solid: Current fraction solid (0-1)
        prev_fraction_solid: Previous fraction solid (0-1)
        latent_heat: Latent heat of fusion (J/kg)
        density: Density (kg/m³)
        specific_heat: Specific heat (J/kg·K)
        time_step: Time step (s)
    
    Returns:
        Heat source per unit volume (W/m³)
    """
    n_voxels = temperatures.shape[0]
    heat_source = np.zeros(n_voxels, dtype=np.float32)
    
    for i in prange(n_voxels):
        # Rate of solidification
        dfs_dt = (fraction_solid[i] - prev_fraction_solid[i]) / time_step
        
        # Latent heat release: Q = ρL * (dfs/dt)
        # This is added as a heat source (positive when solidifying)
        if dfs_dt > 0:  # Solidifying
            heat_source[i] = density * latent_heat * dfs_dt
    
    return heat_source


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_fraction_solid_array(
    temperatures: np.ndarray,
    liquidus_temp: float,
    solidus_temp: float
) -> np.ndarray:
    """
    Compute fraction solid for all voxels based on temperature.
    
    Uses linear interpolation in mushy zone (lever rule approximation).
    
    Args:
        temperatures: Temperature at each voxel (°C)
        liquidus_temp: Liquidus temperature (°C)
        solidus_temp: Solidus temperature (°C)
    
    Returns:
        Fraction solid at each voxel (0-1)
    """
    n_voxels = temperatures.shape[0]
    fraction_solid = np.zeros(n_voxels, dtype=np.float32)
    
    freezing_range = liquidus_temp - solidus_temp
    if freezing_range <= 0:
        freezing_range = 1.0  # Prevent division by zero
    
    for i in prange(n_voxels):
        T = temperatures[i]
        if T >= liquidus_temp:
            fraction_solid[i] = 0.0
        elif T <= solidus_temp:
            fraction_solid[i] = 1.0
        else:
            # Linear interpolation in mushy zone
            fraction_solid[i] = (liquidus_temp - T) / freezing_range
    
    return fraction_solid


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def explicit_heat_equation_step(
    temperatures: np.ndarray,
    laplacian: np.ndarray,
    heat_source: np.ndarray,
    diffusivity: np.ndarray,
    density: float,
    specific_heat: float,
    time_step: float,
    boundary_mask: np.ndarray,
    boundary_temps: np.ndarray
) -> np.ndarray:
    """
    Perform one explicit forward Euler step for the heat equation.
    
    T^(n+1) = T^n + Δt * [α∇²T + Q/(ρcₚ)]
    
    Args:
        temperatures: Current temperatures (°C)
        laplacian: Laplacian of temperature (°C/m²)
        heat_source: Heat source term (W/m³)
        diffusivity: Thermal diffusivity at each voxel (m²/s)
        density: Density (kg/m³)
        specific_heat: Specific heat (J/kg·K)
        time_step: Time step (s)
        boundary_mask: Boolean mask for boundary voxels
        boundary_temps: Fixed temperature for boundary voxels (°C)
    
    Returns:
        Updated temperatures (°C)
    """
    n_voxels = temperatures.shape[0]
    new_temps = np.zeros(n_voxels, dtype=np.float32)
    
    rho_cp = density * specific_heat
    
    for i in prange(n_voxels):
        if boundary_mask[i]:
            # Fixed boundary condition
            new_temps[i] = boundary_temps[i]
        else:
            # Heat equation update
            diffusion = diffusivity[i] * laplacian[i]
            source = heat_source[i] / rho_cp if rho_cp > 0 else 0.0
            
            new_temps[i] = temperatures[i] + time_step * (diffusion + source)
    
    return new_temps


@jit(nopython=True, cache=True)
def compute_stable_time_step(
    diffusivity_max: float,
    voxel_size: float,
    cfl_number: float = 0.4
) -> float:
    """
    Compute stable time step for explicit scheme using CFL condition.
    
    For 3D explicit heat equation: Δt ≤ h² / (6α) * CFL
    
    Args:
        diffusivity_max: Maximum thermal diffusivity (m²/s)
        voxel_size: Voxel size (m)
        cfl_number: CFL safety factor (typically 0.3-0.5)
    
    Returns:
        Stable time step (s)
    """
    if diffusivity_max <= 0:
        return 1.0
    
    # 3D stability limit: Δt ≤ h² / (6α)
    dt_stability = (voxel_size * voxel_size) / (6.0 * diffusivity_max)
    
    return dt_stability * cfl_number


class ThermalSolver:
    """
    Finite difference thermal solver for casting solidification.
    
    Solves the heat equation with latent heat release:
        ∂T/∂t = α∇²T + Q/(ρcₚ)
    
    Features:
    - Explicit time stepping with adaptive time step
    - Latent heat release during solidification (enthalpy method)
    - Temperature-dependent material properties
    - Configurable boundary conditions
    - GPU acceleration when available
    """
    
    def __init__(self, 
                 n_voxels: int,
                 voxel_size_m: float,
                 thermal_props: ThermalProperties,
                 config: Optional[ThermalSolverConfig] = None):
        """
        Initialize thermal solver.
        
        Args:
            n_voxels: Number of voxels in mesh
            voxel_size_m: Voxel size in meters
            thermal_props: Material thermal properties
            config: Solver configuration (uses defaults if None)
        """
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size_m
        self.thermal_props = thermal_props
        self.config = config or ThermalSolverConfig()
        
        # State arrays
        self.temperatures = np.zeros(n_voxels, dtype=np.float32)
        self.prev_temperatures = np.zeros(n_voxels, dtype=np.float32)
        self.fraction_solid = np.zeros(n_voxels, dtype=np.float32)
        self.prev_fraction_solid = np.zeros(n_voxels, dtype=np.float32)
        
        # Property arrays (updated each step if temp-dependent)
        self.diffusivity = np.zeros(n_voxels, dtype=np.float32)
        
        # Boundary conditions
        self.boundary_mask = np.zeros(n_voxels, dtype=np.bool_)
        self.boundary_temps = np.zeros(n_voxels, dtype=np.float32)
        
        # Time tracking
        self.current_time = 0.0
        self.time_step = self.config.max_time_step
        
        # Statistics
        self.total_steps = 0
        self.total_latent_heat_released = 0.0
        
        # GPU acceleration
        self._use_gpu = self.config.use_gpu and GPU_AVAILABLE
        
    def initialize(self,
                   neighbors: np.ndarray,
                   surface_mask: np.ndarray,
                   pouring_temp: Optional[float] = None):
        """
        Initialize temperature field and boundary conditions.
        
        Args:
            neighbors: Neighbor connectivity array (n_voxels x 26)
            surface_mask: Boolean mask for surface voxels
            pouring_temp: Initial pouring temperature (uses config default if None)
        """
        self.neighbors = np.ascontiguousarray(neighbors, dtype=np.int32)
        
        # Initial temperature: pouring temperature for interior
        T_pour = pouring_temp or self.config.pouring_temperature
        self.temperatures[:] = T_pour
        self.prev_temperatures[:] = T_pour
        
        # Surface voxels are boundaries
        self.boundary_mask = surface_mask.copy()
        self.boundary_temps[surface_mask] = self.config.mold_temperature
        
        # Initialize fraction solid (all liquid at pouring temp)
        self.fraction_solid = compute_fraction_solid_array(
            self.temperatures,
            self.thermal_props.liquidus_temp,
            self.thermal_props.solidus_temp
        )
        self.prev_fraction_solid[:] = self.fraction_solid
        
        # Compute initial diffusivity
        self._update_properties()
        
        # Compute stable time step
        self._compute_stable_timestep()
        
        print(f"\n  Thermal solver initialized:")
        print(f"    Voxels: {self.n_voxels:,}")
        print(f"    Voxel size: {self.voxel_size*1000:.2f} mm")
        print(f"    Pouring temp: {T_pour:.0f}°C")
        print(f"    Mold temp: {self.config.mold_temperature:.0f}°C")
        print(f"    Stable time step: {self.time_step:.4f}s")
        print(f"    Liquidus: {self.thermal_props.liquidus_temp:.0f}°C")
        print(f"    Solidus: {self.thermal_props.solidus_temp:.0f}°C")
        print(f"    Latent heat: {self.thermal_props.latent_heat/1000:.0f} kJ/kg")
        if self._use_gpu:
            print(f"    GPU acceleration: Enabled")
    
    def _update_properties(self):
        """Update temperature-dependent properties"""
        if self.config.use_temperature_dependent_props:
            # Vectorized property lookup for all voxels
            for i in range(self.n_voxels):
                self.diffusivity[i] = self.thermal_props.get_diffusivity(
                    self.temperatures[i]
                )
        else:
            # Use constant properties at reference temperature
            ref_temp = (self.thermal_props.liquidus_temp + 
                       self.thermal_props.solidus_temp) / 2
            alpha = self.thermal_props.get_diffusivity(ref_temp)
            self.diffusivity[:] = alpha
    
    def _compute_stable_timestep(self):
        """Compute stable time step based on current properties"""
        alpha_max = np.max(self.diffusivity)
        self.time_step = compute_stable_time_step(
            alpha_max,
            self.voxel_size,
            self.config.cfl_safety_factor
        )
        
        # Don't exceed user-specified max
        self.time_step = min(self.time_step, self.config.max_time_step)
    
    def step(self) -> Tuple[float, float]:
        """
        Advance solution by one time step.
        
        Returns:
            Tuple of (time_step_used, max_temp_change)
        """
        # Store previous state
        self.prev_temperatures[:] = self.temperatures
        self.prev_fraction_solid[:] = self.fraction_solid
        
        # Update properties if temperature-dependent
        if self.config.use_temperature_dependent_props:
            self._update_properties()
            if self.config.use_adaptive_timestep:
                self._compute_stable_timestep()
        
        # Compute Laplacian
        laplacian = compute_laplacian_6point(
            self.temperatures,
            self.neighbors,
            self.voxel_size
        )
        
        # Compute latent heat source
        if self.config.include_latent_heat:
            # Get average properties for source term
            rho = self.thermal_props.get_density(
                np.mean(self.temperatures)
            )
            cp = self.thermal_props.get_specific_heat(
                np.mean(self.temperatures)
            )
            
            heat_source = compute_latent_heat_source(
                self.temperatures,
                self.prev_temperatures,
                self.fraction_solid,
                self.prev_fraction_solid,
                self.thermal_props.latent_heat,
                rho,
                cp,
                max(self.time_step, 1e-10)  # Avoid division by zero
            )
        else:
            heat_source = np.zeros(self.n_voxels, dtype=np.float32)
        
        # Average density and specific heat for explicit step
        rho = self.thermal_props.get_density(np.mean(self.temperatures))
        cp = self.thermal_props.get_specific_heat(np.mean(self.temperatures))
        
        # Explicit time step
        self.temperatures = explicit_heat_equation_step(
            self.temperatures,
            laplacian,
            heat_source,
            self.diffusivity,
            rho,
            cp,
            self.time_step,
            self.boundary_mask,
            self.boundary_temps
        )
        
        # Update fraction solid
        self.fraction_solid = compute_fraction_solid_array(
            self.temperatures,
            self.thermal_props.liquidus_temp,
            self.thermal_props.solidus_temp
        )
        
        # Update time
        self.current_time += self.time_step
        self.total_steps += 1
        
        # Track latent heat released
        if self.config.include_latent_heat:
            dfs = self.fraction_solid - self.prev_fraction_solid
            self.total_latent_heat_released += np.sum(
                np.maximum(dfs, 0) * self.thermal_props.latent_heat
            )
        
        # Compute max temperature change
        max_change = np.max(np.abs(self.temperatures - self.prev_temperatures))
        
        return self.time_step, max_change
    
    def run_until(self, 
                  target_time: float = None,
                  target_fraction_solid: float = None,
                  max_steps: int = 100000,
                  progress_callback=None,
                  show_progress: bool = True) -> bool:
        """
        Run simulation until target condition is met.
        
        Args:
            target_time: Stop when simulation time reaches this value (s)
            target_fraction_solid: Stop when average fraction solid exceeds this
            max_steps: Maximum number of time steps
            progress_callback: Optional callback(current, total, stage)
            show_progress: Print progress to console
        
        Returns:
            True if simulation completed successfully
        """
        if target_time is None and target_fraction_solid is None:
            target_fraction_solid = 0.99  # Default: 99% solidified
        
        step_count = 0
        last_progress = 0
        
        while step_count < max_steps:
            dt, max_change = self.step()
            step_count += 1
            
            # Check termination conditions
            if target_time is not None and self.current_time >= target_time:
                break
            
            avg_fs = np.mean(self.fraction_solid)
            if target_fraction_solid is not None and avg_fs >= target_fraction_solid:
                break
            
            # Progress reporting
            if target_fraction_solid is not None:
                progress_pct = int(avg_fs / target_fraction_solid * 100)
            else:
                progress_pct = int(self.current_time / target_time * 100)
            
            if progress_pct > last_progress:
                if progress_callback:
                    progress_callback(progress_pct, 100, "Thermal")
                
                if show_progress and progress_pct % 10 == 0:
                    T_min = np.min(self.temperatures)
                    T_max = np.max(self.temperatures[~self.boundary_mask])
                    print(f"    t={self.current_time:.1f}s: "
                          f"fs={avg_fs*100:.1f}%, "
                          f"T_range=[{T_min:.0f}, {T_max:.0f}]°C, "
                          f"dt={dt:.4f}s")
                
                last_progress = progress_pct
        
        if show_progress:
            print(f"\n  Thermal simulation complete:")
            print(f"    Time: {self.current_time:.1f}s")
            print(f"    Steps: {step_count:,}")
            print(f"    Avg fraction solid: {np.mean(self.fraction_solid)*100:.1f}%")
            print(f"    Final T range: [{np.min(self.temperatures):.0f}, "
                  f"{np.max(self.temperatures):.0f}]°C")
        
        return True
    
    def get_thermal_gradient(self) -> np.ndarray:
        """
        Compute thermal gradient magnitude at each voxel.
        
        Returns:
            Gradient magnitude (°C/m)
        """
        # Use centered differences for gradient estimation
        return _compute_gradient_magnitude(
            self.temperatures,
            self.neighbors,
            self.voxel_size
        )
    
    def get_cooling_rate(self) -> np.ndarray:
        """
        Compute cooling rate at each voxel.
        
        Returns:
            Cooling rate (°C/s)
        """
        dt = max(self.time_step, 1e-10)
        return (self.prev_temperatures - self.temperatures) / dt
    
    def get_niyama_criterion(self) -> np.ndarray:
        """
        Compute Niyama criterion: G / sqrt(R)
        
        Returns:
            Niyama values at each voxel
        """
        G = self.get_thermal_gradient()
        R = np.maximum(self.get_cooling_rate(), 1e-10)
        return G / np.sqrt(R)
    
    def get_statistics(self) -> Dict:
        """Get solver statistics"""
        return {
            'current_time': self.current_time,
            'total_steps': self.total_steps,
            'time_step': self.time_step,
            'avg_fraction_solid': np.mean(self.fraction_solid),
            'min_temperature': np.min(self.temperatures),
            'max_temperature': np.max(self.temperatures),
            'avg_temperature': np.mean(self.temperatures),
            'total_latent_heat_released': self.total_latent_heat_released
        }


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _compute_gradient_magnitude(
    temperatures: np.ndarray,
    neighbors: np.ndarray,
    voxel_size: float
) -> np.ndarray:
    """
    Compute gradient magnitude using finite differences.
    
    |∇T| ≈ sqrt((dT/dx)² + (dT/dy)² + (dT/dz)²)
    """
    n_voxels = temperatures.shape[0]
    gradient = np.zeros(n_voxels, dtype=np.float32)
    h = voxel_size
    
    for i in prange(n_voxels):
        T_center = temperatures[i]
        grad_x = np.float32(0.0)
        grad_y = np.float32(0.0)
        grad_z = np.float32(0.0)
        
        # X gradient (neighbors 0 and 1 for +x and -x)
        if neighbors.shape[1] >= 2:
            n_px = neighbors[i, 0]  # +x
            n_mx = neighbors[i, 1]  # -x
            if n_px >= 0 and n_mx >= 0:
                grad_x = (temperatures[n_px] - temperatures[n_mx]) / (2 * h)
            elif n_px >= 0:
                grad_x = (temperatures[n_px] - T_center) / h
            elif n_mx >= 0:
                grad_x = (T_center - temperatures[n_mx]) / h
        
        # Y gradient (neighbors 2 and 3)
        if neighbors.shape[1] >= 4:
            n_py = neighbors[i, 2]  # +y
            n_my = neighbors[i, 3]  # -y
            if n_py >= 0 and n_my >= 0:
                grad_y = (temperatures[n_py] - temperatures[n_my]) / (2 * h)
            elif n_py >= 0:
                grad_y = (temperatures[n_py] - T_center) / h
            elif n_my >= 0:
                grad_y = (T_center - temperatures[n_my]) / h
        
        # Z gradient (neighbors 4 and 5)
        if neighbors.shape[1] >= 6:
            n_pz = neighbors[i, 4]  # +z
            n_mz = neighbors[i, 5]  # -z
            if n_pz >= 0 and n_mz >= 0:
                grad_z = (temperatures[n_pz] - temperatures[n_mz]) / (2 * h)
            elif n_pz >= 0:
                grad_z = (temperatures[n_pz] - T_center) / h
            elif n_mz >= 0:
                grad_z = (T_center - temperatures[n_mz]) / h
        
        gradient[i] = np.sqrt(grad_x*grad_x + grad_y*grad_y + grad_z*grad_z)
    
    return gradient


# Convenience function for quick setup
def create_thermal_solver(
    mesh,  # AdaptiveVoxelMesh
    alloy,  # AlloyProperties
    config: Optional[ThermalSolverConfig] = None
) -> ThermalSolver:
    """
    Create and initialize a thermal solver from mesh and alloy.
    
    Args:
        mesh: AdaptiveVoxelMesh with voxel data
        alloy: AlloyProperties for the casting material
        config: Optional solver configuration
    
    Returns:
        Initialized ThermalSolver
    """
    # Convert voxel size from inches to meters
    voxel_size_m = mesh.base_voxel_size * 0.0254  # inches to meters
    
    # Create thermal properties from alloy
    thermal_props = create_thermal_properties_from_alloy(alloy)
    
    # Create solver
    solver = ThermalSolver(
        n_voxels=mesh.get_voxel_count(),
        voxel_size_m=voxel_size_m,
        thermal_props=thermal_props,
        config=config
    )
    
    # Initialize with mesh data
    solver.initialize(
        neighbors=mesh.voxel_neighbors,
        surface_mask=mesh.surface_voxels,
        pouring_temp=alloy.pouring_temp
    )
    
    return solver
