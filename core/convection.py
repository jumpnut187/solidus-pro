"""
Convection Model for Liquid Metal in Casting
Implements simplified natural convection model for thermal homogenization.

Physics Background:
Natural convection occurs in liquid metal due to density differences
caused by temperature gradients. Hot liquid rises, cold liquid sinks,
creating convective cells that enhance heat transfer.

Key Equations:
    Rayleigh Number: Ra = gβΔTL³/(να)
    Nusselt Number:  Nu = f(Ra, Pr)
    
    where:
        g = gravitational acceleration (9.81 m/s²)
        β = thermal expansion coefficient (1/K)
        ΔT = temperature difference (K)
        L = characteristic length (m)
        ν = kinematic viscosity (m²/s)
        α = thermal diffusivity (m²/s)
        Pr = Prandtl number = ν/α

The convection model:
1. Calculates effective thermal conductivity enhancement
2. Models buoyancy-driven temperature redistribution  
3. Applies to liquid (mushy zone < 50% solid) regions only
4. Decays as solidification progresses

This is a SIMPLIFIED model that captures the key physics without
full CFD simulation. It's appropriate for:
- Capturing convection-enhanced heat transfer
- Modeling temperature homogenization in liquid pools
- Predicting convection's effect on solidification patterns

For full fluid flow prediction, use dedicated CFD software.

Author: GitHub Copilot
Date: November 2025
"""

import numpy as np
from numba import jit, prange
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


# Physical constants
GRAVITY = 9.81  # m/s²


@dataclass
class LiquidMetalProperties:
    """
    Properties of liquid metal for convection calculations.
    
    These properties affect natural convection strength and behavior.
    """
    # Thermal expansion coefficient (1/K)
    # Typical range for molten metals: 1e-4 to 2e-4
    thermal_expansion: float = 1.2e-4
    
    # Dynamic viscosity (Pa·s = kg/(m·s))
    # Molten steel: ~0.005-0.007 Pa·s
    dynamic_viscosity: float = 0.006
    
    # Density (kg/m³)
    density: float = 7200.0
    
    # Thermal diffusivity (m²/s)
    thermal_diffusivity: float = 7.5e-6
    
    # Thermal conductivity (W/m·K)
    thermal_conductivity: float = 35.0
    
    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity: ν = μ/ρ (m²/s)"""
        return self.dynamic_viscosity / self.density
    
    @property
    def prandtl_number(self) -> float:
        """Prandtl number: Pr = ν/α"""
        return self.kinematic_viscosity / self.thermal_diffusivity


def create_liquid_properties_from_alloy(alloy) -> LiquidMetalProperties:
    """
    Create liquid metal properties from AlloyProperties object.
    
    Estimates liquid-phase properties based on alloy characteristics.
    """
    # Typical thermal expansion for molten metals
    # Steel: ~1.2e-4, Stainless: ~1.5e-4, Bronze: ~1.0e-4
    if 'steel' in alloy.name.lower() or 'wcb' in alloy.name.lower():
        beta = 1.2e-4
        mu = 0.006  # Pa·s
    elif 'stainless' in alloy.name.lower() or 'cf8' in alloy.name.lower():
        beta = 1.5e-4
        mu = 0.007  # Higher viscosity for stainless
    else:  # Bronze and other alloys
        beta = 1.0e-4
        mu = 0.004  # Lower viscosity
    
    return LiquidMetalProperties(
        thermal_expansion=beta,
        dynamic_viscosity=mu,
        density=alloy.density_liquid,
        thermal_diffusivity=alloy.thermal_diffusivity,
        thermal_conductivity=alloy.thermal_conductivity * 0.7  # Liquid ~70% of solid
    )


@jit(nopython=True, cache=True)
def compute_rayleigh_number(
    delta_T: float,
    characteristic_length: float,
    thermal_expansion: float,
    kinematic_viscosity: float,
    thermal_diffusivity: float
) -> float:
    """
    Compute Rayleigh number for natural convection.
    
    Ra = gβΔTL³ / (να)
    
    Args:
        delta_T: Temperature difference (K)
        characteristic_length: Length scale (m)
        thermal_expansion: Thermal expansion coefficient (1/K)
        kinematic_viscosity: Kinematic viscosity (m²/s)
        thermal_diffusivity: Thermal diffusivity (m²/s)
    
    Returns:
        Rayleigh number (dimensionless)
    """
    if kinematic_viscosity <= 0 or thermal_diffusivity <= 0:
        return 0.0
    
    L3 = characteristic_length ** 3
    Ra = (GRAVITY * thermal_expansion * abs(delta_T) * L3) / (
        kinematic_viscosity * thermal_diffusivity
    )
    
    return Ra


@jit(nopython=True, cache=True)
def compute_nusselt_number(rayleigh: float, prandtl: float) -> float:
    """
    Compute Nusselt number for natural convection.
    
    Uses Churchill-Chu correlation for vertical surfaces/enclosures:
    Nu = 0.68 + 0.67*Ra^(1/4) / [1 + (0.492/Pr)^(9/16)]^(4/9)
    
    For Ra < 10^9 (laminar regime).
    
    Args:
        rayleigh: Rayleigh number
        prandtl: Prandtl number
    
    Returns:
        Nusselt number (dimensionless)
    """
    if rayleigh < 1e3:
        # Very weak convection - conduction dominated
        return 1.0
    
    # Churchill-Chu correlation
    Ra_quarter = rayleigh ** 0.25
    Pr_term = (1.0 + (0.492 / prandtl) ** (9.0/16.0)) ** (4.0/9.0)
    
    Nu = 0.68 + 0.67 * Ra_quarter / Pr_term
    
    # Clamp to reasonable range
    return max(1.0, min(Nu, 100.0))


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_effective_conductivity(
    temperatures: np.ndarray,
    fraction_solid: np.ndarray,
    neighbors: np.ndarray,
    voxel_size: float,
    base_conductivity: float,
    thermal_expansion: float,
    kinematic_viscosity: float,
    thermal_diffusivity: float,
    prandtl: float,
    liquid_threshold: float = 0.5
) -> np.ndarray:
    """
    Compute effective thermal conductivity including convection enhancement.
    
    In liquid regions, convection enhances heat transfer:
        k_eff = k_base * Nu
    
    The enhancement factor (Nusselt number) depends on local temperature
    gradients and convection strength.
    
    Args:
        temperatures: Temperature at each voxel (°C)
        fraction_solid: Fraction solid at each voxel (0-1)
        neighbors: Neighbor connectivity
        voxel_size: Voxel size (m)
        base_conductivity: Base thermal conductivity (W/m·K)
        thermal_expansion: Thermal expansion coefficient (1/K)
        kinematic_viscosity: Kinematic viscosity (m²/s)
        thermal_diffusivity: Thermal diffusivity (m²/s)
        prandtl: Prandtl number
        liquid_threshold: Fraction solid below which convection occurs
    
    Returns:
        Effective conductivity at each voxel (W/m·K)
    """
    n_voxels = temperatures.shape[0]
    k_eff = np.full(n_voxels, base_conductivity, dtype=np.float32)
    
    for i in prange(n_voxels):
        # Only apply convection in liquid/mushy regions
        if fraction_solid[i] > liquid_threshold:
            continue
        
        # Compute local temperature difference with neighbors
        T_center = temperatures[i]
        T_max_neighbor = T_center
        T_min_neighbor = T_center
        
        for j in range(neighbors.shape[1]):
            neighbor_idx = neighbors[i, j]
            if neighbor_idx >= 0:
                T_neighbor = temperatures[neighbor_idx]
                if T_neighbor > T_max_neighbor:
                    T_max_neighbor = T_neighbor
                if T_neighbor < T_min_neighbor:
                    T_min_neighbor = T_neighbor
        
        delta_T = T_max_neighbor - T_min_neighbor
        
        if delta_T < 1.0:  # Negligible gradient
            continue
        
        # Characteristic length: use voxel size or local liquid pool size
        L = voxel_size * 2  # Approximate local length scale
        
        # Compute Rayleigh and Nusselt numbers
        Ra = compute_rayleigh_number(
            delta_T, L, thermal_expansion, 
            kinematic_viscosity, thermal_diffusivity
        )
        Nu = compute_nusselt_number(Ra, prandtl)
        
        # Apply convection factor, modulated by liquid fraction
        # Full convection only in fully liquid; decays in mushy zone
        liquid_factor = 1.0 - (fraction_solid[i] / liquid_threshold)
        k_eff[i] = base_conductivity * (1.0 + (Nu - 1.0) * liquid_factor)
    
    return k_eff


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def apply_buoyancy_mixing(
    temperatures: np.ndarray,
    fraction_solid: np.ndarray,
    neighbors: np.ndarray,
    heights: np.ndarray,
    mixing_coefficient: float,
    time_step: float,
    liquid_threshold: float = 0.5
) -> np.ndarray:
    """
    Apply simplified buoyancy-driven mixing in liquid regions.
    
    Models the tendency for hot liquid to rise and cold to sink,
    homogenizing temperature in connected liquid pools.
    
    This is a diffusion-like approximation of convective mixing:
    - Hot liquid below cold liquid: promotes mixing
    - Cold liquid below hot liquid: stable stratification
    
    Args:
        temperatures: Current temperatures (°C)
        fraction_solid: Fraction solid (0-1)
        neighbors: Neighbor connectivity
        heights: Normalized height of each voxel (0-1)
        mixing_coefficient: Mixing strength (0-1)
        time_step: Time step (s)
        liquid_threshold: Fraction solid below which mixing occurs
    
    Returns:
        Updated temperatures after mixing (°C)
    """
    n_voxels = temperatures.shape[0]
    new_temps = temperatures.copy()
    
    for i in prange(n_voxels):
        # Only mix in liquid regions
        if fraction_solid[i] > liquid_threshold:
            continue
        
        T_center = temperatures[i]
        h_center = heights[i]
        
        # Find liquid neighbors and their temperatures
        total_mixing = np.float32(0.0)
        count = 0
        
        for j in range(neighbors.shape[1]):
            neighbor_idx = neighbors[i, j]
            if neighbor_idx < 0:
                continue
            
            # Skip solid neighbors
            if fraction_solid[neighbor_idx] > liquid_threshold:
                continue
            
            T_neighbor = temperatures[neighbor_idx]
            h_neighbor = heights[neighbor_idx]
            
            # Temperature and height differences
            delta_T = T_neighbor - T_center
            delta_h = h_neighbor - h_center
            
            # Buoyancy-driven mixing:
            # - Hot below cold (delta_T > 0 and delta_h > 0): strong mixing
            # - Cold below hot (delta_T < 0 and delta_h < 0): strong mixing
            # - Stable stratification: weak mixing
            
            if delta_T * delta_h > 0:  # Unstable stratification
                mixing_factor = abs(delta_T) * 0.1  # Strong mixing
            else:  # Stable or horizontal
                mixing_factor = abs(delta_T) * 0.01  # Weak mixing
            
            total_mixing += mixing_factor * delta_T
            count += 1
        
        if count > 0:
            # Apply mixing proportional to liquid fraction
            liquid_factor = 1.0 - (fraction_solid[i] / liquid_threshold)
            adjustment = mixing_coefficient * liquid_factor * total_mixing / count
            new_temps[i] = T_center + adjustment * time_step
    
    return new_temps


class ConvectionModel:
    """
    Simplified natural convection model for liquid metal.
    
    Features:
    - Effective conductivity enhancement in liquid regions
    - Buoyancy-driven temperature mixing
    - Decays smoothly in mushy zone
    - Deactivates in solid regions
    
    Usage:
        model = ConvectionModel(n_voxels, liquid_props)
        
        # Each time step:
        k_eff = model.get_effective_conductivity(temperatures, fraction_solid, ...)
        temperatures = model.apply_mixing(temperatures, fraction_solid, ...)
    """
    
    def __init__(self,
                 n_voxels: int,
                 liquid_props: Optional[LiquidMetalProperties] = None,
                 liquid_threshold: float = 0.5,
                 mixing_coefficient: float = 0.1):
        """
        Initialize convection model.
        
        Args:
            n_voxels: Number of voxels
            liquid_props: Liquid metal properties
            liquid_threshold: Fraction solid below which convection active
            mixing_coefficient: Strength of buoyancy mixing (0-1)
        """
        self.n_voxels = n_voxels
        self.props = liquid_props or LiquidMetalProperties()
        self.liquid_threshold = liquid_threshold
        self.mixing_coefficient = mixing_coefficient
        
        # Pre-compute constant properties
        self.prandtl = self.props.prandtl_number
        
        # Statistics
        self.avg_nusselt = 1.0
        self.liquid_fraction = 1.0
        
    def initialize(self):
        """Initialize convection model (call after mesh setup)"""
        print(f"\n  Convection model initialized:")
        print(f"    Prandtl number: {self.prandtl:.3f}")
        print(f"    Thermal expansion: {self.props.thermal_expansion:.2e} 1/K")
        print(f"    Kinematic viscosity: {self.props.kinematic_viscosity:.2e} m²/s")
        print(f"    Liquid threshold: {self.liquid_threshold*100:.0f}% solid")
        print(f"    Mixing coefficient: {self.mixing_coefficient:.2f}")
    
    def get_effective_conductivity(self,
                                   temperatures: np.ndarray,
                                   fraction_solid: np.ndarray,
                                   neighbors: np.ndarray,
                                   voxel_size: float) -> np.ndarray:
        """
        Compute effective thermal conductivity with convection.
        
        Args:
            temperatures: Temperature at each voxel (°C)
            fraction_solid: Fraction solid (0-1)
            neighbors: Neighbor connectivity
            voxel_size: Voxel size (m)
        
        Returns:
            Effective conductivity array (W/m·K)
        """
        k_eff = compute_effective_conductivity(
            np.ascontiguousarray(temperatures, dtype=np.float32),
            np.ascontiguousarray(fraction_solid, dtype=np.float32),
            np.ascontiguousarray(neighbors, dtype=np.int32),
            voxel_size,
            self.props.thermal_conductivity,
            self.props.thermal_expansion,
            self.props.kinematic_viscosity,
            self.props.thermal_diffusivity,
            self.prandtl,
            self.liquid_threshold
        )
        
        # Update statistics
        liquid_mask = fraction_solid < self.liquid_threshold
        if np.any(liquid_mask):
            self.avg_nusselt = np.mean(k_eff[liquid_mask]) / self.props.thermal_conductivity
        self.liquid_fraction = np.mean(1.0 - fraction_solid)
        
        return k_eff
    
    def apply_mixing(self,
                     temperatures: np.ndarray,
                     fraction_solid: np.ndarray,
                     neighbors: np.ndarray,
                     heights: np.ndarray,
                     time_step: float) -> np.ndarray:
        """
        Apply buoyancy-driven mixing in liquid regions.
        
        Args:
            temperatures: Current temperatures (°C)
            fraction_solid: Fraction solid (0-1)
            neighbors: Neighbor connectivity
            heights: Normalized height (0-1) of each voxel
            time_step: Time step (s)
        
        Returns:
            Updated temperatures (°C)
        """
        return apply_buoyancy_mixing(
            np.ascontiguousarray(temperatures, dtype=np.float32),
            np.ascontiguousarray(fraction_solid, dtype=np.float32),
            np.ascontiguousarray(neighbors, dtype=np.int32),
            np.ascontiguousarray(heights, dtype=np.float32),
            self.mixing_coefficient,
            time_step,
            self.liquid_threshold
        )
    
    def get_convection_activity(self,
                                fraction_solid: np.ndarray) -> np.ndarray:
        """
        Get map of where convection is active (for visualization).
        
        Returns values 0-1 indicating convection strength.
        """
        activity = np.zeros(len(fraction_solid), dtype=np.float32)
        liquid_mask = fraction_solid < self.liquid_threshold
        
        if np.any(liquid_mask):
            # Activity proportional to liquid fraction
            activity[liquid_mask] = 1.0 - (fraction_solid[liquid_mask] / self.liquid_threshold)
        
        return activity
    
    def get_statistics(self) -> Dict:
        """Get convection model statistics"""
        return {
            'average_nusselt': self.avg_nusselt,
            'liquid_fraction': self.liquid_fraction,
            'convection_active': self.liquid_fraction > 0.01,
            'prandtl_number': self.prandtl,
            'thermal_expansion': self.props.thermal_expansion,
            'kinematic_viscosity': self.props.kinematic_viscosity
        }


@jit(nopython=True, cache=True)
def estimate_convection_time_scale(
    characteristic_length: float,
    thermal_expansion: float,
    delta_T: float,
    kinematic_viscosity: float
) -> float:
    """
    Estimate convection time scale for adaptive time stepping.
    
    Convection time scale: τ = L / U
    where U ~ sqrt(gβΔTL) is the buoyancy velocity scale.
    
    Args:
        characteristic_length: Length scale (m)
        thermal_expansion: Thermal expansion coefficient (1/K)
        delta_T: Temperature difference (K)
        kinematic_viscosity: Kinematic viscosity (m²/s)
    
    Returns:
        Convection time scale (s)
    """
    if delta_T < 1e-6:
        return 1e6  # Very large (no convection)
    
    # Buoyancy velocity scale
    U = np.sqrt(GRAVITY * thermal_expansion * abs(delta_T) * characteristic_length)
    
    if U < 1e-10:
        return 1e6
    
    # Convection time scale
    tau = characteristic_length / U
    
    return max(tau, 0.001)  # At least 1ms


# Convenience function for quick setup
def create_convection_model(
    mesh,  # AdaptiveVoxelMesh
    alloy = None,  # AlloyProperties
    liquid_threshold: float = None,  # Use alloy's critical_fraction_solid if not specified
    mixing_coefficient: float = 0.1
) -> ConvectionModel:
    """
    Create and initialize convection model from mesh and alloy.
    
    Args:
        mesh: AdaptiveVoxelMesh
        alloy: AlloyProperties (for liquid metal properties)
        liquid_threshold: Fraction solid threshold for convection.
                          If None, uses alloy.critical_fraction_solid
        mixing_coefficient: Mixing strength
    
    Returns:
        Initialized ConvectionModel
    """
    if alloy is not None:
        liquid_props = create_liquid_properties_from_alloy(alloy)
        # Use alloy's critical fraction solid as liquid threshold if not specified
        if liquid_threshold is None:
            liquid_threshold = alloy.critical_fraction_solid
    else:
        liquid_props = LiquidMetalProperties()
        if liquid_threshold is None:
            liquid_threshold = 0.5  # Default fallback
    
    model = ConvectionModel(
        n_voxels=mesh.get_voxel_count(),
        liquid_props=liquid_props,
        liquid_threshold=liquid_threshold,
        mixing_coefficient=mixing_coefficient
    )
    
    model.initialize()
    
    return model


def print_convection_info(liquid_props: LiquidMetalProperties):
    """Print liquid metal convection properties"""
    print(f"\n{'='*60}")
    print(f"Liquid Metal Convection Properties")
    print(f"{'='*60}")
    print(f"\nFluid Properties:")
    print(f"  Density:           {liquid_props.density:.0f} kg/m³")
    print(f"  Dynamic viscosity: {liquid_props.dynamic_viscosity:.4f} Pa·s")
    print(f"  Kinematic visc:    {liquid_props.kinematic_viscosity:.2e} m²/s")
    print(f"\nThermal Properties:")
    print(f"  Conductivity:      {liquid_props.thermal_conductivity:.1f} W/(m·K)")
    print(f"  Diffusivity:       {liquid_props.thermal_diffusivity:.2e} m²/s")
    print(f"  Expansion coeff:   {liquid_props.thermal_expansion:.2e} 1/K")
    print(f"\nDimensionless Numbers:")
    print(f"  Prandtl number:    {liquid_props.prandtl_number:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Demo with default properties
    props = LiquidMetalProperties()
    print_convection_info(props)
    
    # Calculate example Rayleigh and Nusselt numbers
    delta_T = 50.0  # °C temperature difference
    L = 0.05  # 5 cm characteristic length
    
    Ra = compute_rayleigh_number(
        delta_T, L,
        props.thermal_expansion,
        props.kinematic_viscosity,
        props.thermal_diffusivity
    )
    Nu = compute_nusselt_number(Ra, props.prandtl_number)
    
    print(f"Example calculation (ΔT={delta_T}°C, L={L*100}cm):")
    print(f"  Rayleigh number: {Ra:.2e}")
    print(f"  Nusselt number:  {Nu:.2f}")
    print(f"  Effective k/k₀:  {Nu:.2f}x")
