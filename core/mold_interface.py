"""
Mold-Metal Interface Heat Transfer Module
Models heat transfer at the casting-mold interface including:

1. Interfacial Heat Transfer Coefficient (IHTC)
   - Varies with time and temperature
   - Accounts for contact quality changes

2. Mold Thermal Properties
   - Sand molds (silica, olivine, zircon)
   - Metal molds (steel, copper, cast iron)
   - Ceramic molds (investment casting)

3. Air Gap Formation
   - Gap develops during solidification due to shrinkage
   - Dramatically reduces heat transfer
   - Modeled as thermal resistance

Physics:
    q = h * (T_casting - T_mold)
    
    where:
        q = heat flux (W/m²)
        h = interfacial heat transfer coefficient (W/m²·K)
        T_casting = casting surface temperature (°C)
        T_mold = mold surface temperature (°C)

The IHTC varies with:
    - Time (contact quality degrades)
    - Temperature (thermal expansion)
    - Air gap formation (shrinkage)
    - Mold type (sand vs metal vs ceramic)

Author: GitHub Copilot
Date: November 2025
"""

import numpy as np
from numba import jit, prange
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MoldType(str, Enum):
    """Types of casting molds"""
    SAND_SILICA = "sand_silica"        # Traditional silica sand
    SAND_OLIVINE = "sand_olivine"      # Olivine sand (better properties)
    SAND_ZIRCON = "sand_zircon"        # Zircon sand (high-end)
    METAL_STEEL = "metal_steel"        # Steel permanent mold
    METAL_COPPER = "metal_copper"      # Copper mold (high conductivity)
    METAL_CAST_IRON = "metal_cast_iron"  # Cast iron mold
    CERAMIC_SHELL = "ceramic_shell"    # Investment casting shell
    CERAMIC_PLASTER = "ceramic_plaster"  # Plaster mold


class BinderType(str, Enum):
    """Binder chemistries used in sand mold/core production"""
    PHENOLIC_URETHANE = "phenolic_urethane"
    FURAN = "furan"
    PHENOLIC_ESTER = "phenolic_ester"
    SODIUM_SILICATE = "sodium_silicate"
    CLAY_GREEN_SAND = "clay_green_sand"


@dataclass
class MoldProperties:
    """
    Thermal properties for mold materials.
    
    Properties affect heat extraction rate from casting.
    Higher conductivity = faster cooling.
    """
    name: str
    mold_type: MoldType
    
    # Thermal properties
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float         # J/(kg·K)
    density: float               # kg/m³
    
    # Initial IHTC parameters
    initial_ihtc: float          # W/(m²·K) at perfect contact
    ihtc_decay_rate: float       # Rate of IHTC reduction (1/s)
    min_ihtc: float              # Minimum IHTC after air gap (W/(m²·K))
    
    # Air gap parameters  
    gap_onset_temp_fraction: float  # Fraction of solidus where gap starts
    max_gap_thickness: float     # Maximum air gap (mm)
    gap_growth_rate: float       # Gap growth rate (mm/s)
    
    # Initial temperature
    initial_temperature: float   # °C
    
    # Sand burn-in properties (for sand molds)
    # Permeability: how easily gas/metal can penetrate the mold (AFS units)
    # Lower = finer sand, less penetration, better surface finish
    # Typical ranges: 25-50 (fine), 50-100 (medium), 100-200 (coarse)
    permeability: float = 80.0  # AFS permeability number
    
    # Grain Fineness Number (GFN/AFS number)
    # Higher = finer sand, better surface but lower permeability
    # Typical ranges: 30-50 (coarse), 50-70 (medium), 70-100 (fine), 100+ (very fine)
    grain_fineness: float = 55.0  # AFS grain fineness number
    
    # Gas porosity analysis properties
    binder_type: BinderType = BinderType.PHENOLIC_URETHANE
    binder_content: float = 1.5       # Weight-percent binder in sand mix
    binder_gas_yield: float = 12.0    # cm³ gas per gram binder at STP
    core_venting_effectiveness: float = 0.5  # 0-1, higher = better venting
    
    @property
    def thermal_diffusivity(self) -> float:
        """Thermal diffusivity: α = k/(ρcₚ)"""
        return self.thermal_conductivity / (self.density * self.specific_heat)


# Mold material database
MOLD_DATABASE: Dict[MoldType, MoldProperties] = {
    MoldType.SAND_SILICA: MoldProperties(
        name="Silica Sand Mold",
        mold_type=MoldType.SAND_SILICA,
        thermal_conductivity=0.5,    # W/(m·K) - very low
        specific_heat=1130,          # J/(kg·K)
        density=1600,                # kg/m³
        initial_ihtc=500,            # W/(m²·K) - typical for sand
        ihtc_decay_rate=0.01,        # Slow decay
        min_ihtc=100,                # Minimum with air gap
        gap_onset_temp_fraction=0.7, # Gap starts at 70% of solidus
        max_gap_thickness=0.5,       # mm
        gap_growth_rate=0.01,        # mm/s
        initial_temperature=25,      # °C
        permeability=90.0,           # Medium-high permeability (coarse)
        grain_fineness=50.0          # Medium-coarse grain
    ),
    
    MoldType.SAND_OLIVINE: MoldProperties(
        name="Olivine Sand Mold",
        mold_type=MoldType.SAND_OLIVINE,
        thermal_conductivity=0.6,
        specific_heat=1050,
        density=1750,
        initial_ihtc=600,
        ihtc_decay_rate=0.008,
        min_ihtc=120,
        gap_onset_temp_fraction=0.65,
        max_gap_thickness=0.4,
        gap_growth_rate=0.008,
        initial_temperature=25,
        permeability=70.0,           # Medium permeability
        grain_fineness=60.0          # Medium grain
    ),
    
    MoldType.SAND_ZIRCON: MoldProperties(
        name="Zircon Sand Mold",
        mold_type=MoldType.SAND_ZIRCON,
        thermal_conductivity=1.0,
        specific_heat=620,
        density=2800,
        initial_ihtc=800,
        ihtc_decay_rate=0.005,
        min_ihtc=150,
        gap_onset_temp_fraction=0.6,
        max_gap_thickness=0.3,
        gap_growth_rate=0.005,
        initial_temperature=25,
        permeability=40.0,           # Low permeability (fine)
        grain_fineness=85.0          # Fine grain - better surface finish
    ),
    
    MoldType.METAL_STEEL: MoldProperties(
        name="Steel Permanent Mold",
        mold_type=MoldType.METAL_STEEL,
        thermal_conductivity=40,     # Much higher than sand
        specific_heat=500,
        density=7850,
        initial_ihtc=2000,           # High initial contact
        ihtc_decay_rate=0.02,        # Faster decay
        min_ihtc=400,                # Still good with gap
        gap_onset_temp_fraction=0.9, # Gap starts earlier (more shrinkage)
        max_gap_thickness=0.2,
        gap_growth_rate=0.02,
        initial_temperature=150,     # Typically preheated
        permeability=0.0,            # Metal mold - no permeability
        grain_fineness=0.0           # Metal mold - no grain
    ),
    
    MoldType.METAL_COPPER: MoldProperties(
        name="Copper Mold",
        mold_type=MoldType.METAL_COPPER,
        thermal_conductivity=380,    # Very high conductivity
        specific_heat=385,
        density=8960,
        initial_ihtc=5000,           # Excellent initial contact
        ihtc_decay_rate=0.03,
        min_ihtc=800,
        gap_onset_temp_fraction=0.95,
        max_gap_thickness=0.15,
        gap_growth_rate=0.03,
        initial_temperature=100,
        permeability=0.0,            # Metal mold - no permeability
        grain_fineness=0.0           # Metal mold - no grain
    ),
    
    MoldType.METAL_CAST_IRON: MoldProperties(
        name="Cast Iron Mold",
        mold_type=MoldType.METAL_CAST_IRON,
        thermal_conductivity=50,
        specific_heat=460,
        density=7200,
        initial_ihtc=1800,
        ihtc_decay_rate=0.015,
        min_ihtc=350,
        gap_onset_temp_fraction=0.85,
        max_gap_thickness=0.25,
        gap_growth_rate=0.015,
        initial_temperature=200,
        permeability=0.0,            # Metal mold - no permeability
        grain_fineness=0.0           # Metal mold - no grain
    ),
    
    MoldType.CERAMIC_SHELL: MoldProperties(
        name="Ceramic Shell (Investment)",
        mold_type=MoldType.CERAMIC_SHELL,
        thermal_conductivity=0.8,
        specific_heat=950,
        density=2200,
        initial_ihtc=400,            # Lower than sand due to shell nature
        ihtc_decay_rate=0.005,       # Slow decay
        min_ihtc=80,
        gap_onset_temp_fraction=0.75,
        max_gap_thickness=0.3,
        gap_growth_rate=0.01,
        initial_temperature=25,
        permeability=25.0,           # Very low permeability
        grain_fineness=100.0         # Very fine - excellent surface finish
    ),
    
    MoldType.CERAMIC_PLASTER: MoldProperties(
        name="Plaster Mold",
        mold_type=MoldType.CERAMIC_PLASTER,
        thermal_conductivity=0.35,   # Lower than sand
        specific_heat=840,
        density=1200,
        initial_ihtc=300,
        ihtc_decay_rate=0.008,
        min_ihtc=60,
        gap_onset_temp_fraction=0.7,
        max_gap_thickness=0.4,
        gap_growth_rate=0.01,
        initial_temperature=25,
        permeability=30.0,           # Low permeability
        grain_fineness=90.0          # Fine grain
    )
}


def get_mold_properties(mold_type: MoldType) -> MoldProperties:
    """Get properties for a mold type"""
    if mold_type not in MOLD_DATABASE:
        raise KeyError(f"Mold type '{mold_type}' not found")
    return MOLD_DATABASE[mold_type]


def get_all_mold_types() -> list:
    """Get list of all available mold types"""
    return list(MOLD_DATABASE.keys())


@dataclass
class InterfaceState:
    """
    Tracks the state of mold-metal interface for each surface voxel.
    
    State includes:
    - Current air gap thickness
    - Current IHTC
    - Time since solidification started at this location
    - Mold surface temperature
    """
    air_gap: np.ndarray           # Gap thickness (m) per surface voxel
    ihtc: np.ndarray              # Current IHTC (W/m²·K)
    contact_time: np.ndarray      # Time since first solid contact (s)
    mold_surface_temp: np.ndarray # Mold surface temperature (°C)
    is_solidified: np.ndarray     # Whether surface has begun solidifying


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_ihtc_with_air_gap(
    casting_temps: np.ndarray,
    contact_times: np.ndarray,
    air_gaps: np.ndarray,
    is_solidified: np.ndarray,
    initial_ihtc: float,
    ihtc_decay_rate: float,
    min_ihtc: float,
    solidus_temp: float,
    gap_onset_fraction: float,
    gap_growth_rate: float,
    max_gap: float,
    time_step: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute IHTC and update air gap for all surface voxels.
    
    Physics:
    1. IHTC decays exponentially with time after solidification
    2. Air gap grows once surface temperature drops below threshold
    3. Air gap reduces effective IHTC (parallel thermal resistances)
    
    Args:
        casting_temps: Surface casting temperatures (°C)
        contact_times: Time since solidification started (s)
        air_gaps: Current air gap thickness (m)
        is_solidified: Whether surface has started solidifying
        initial_ihtc: Initial IHTC at perfect contact (W/m²·K)
        ihtc_decay_rate: Exponential decay rate (1/s)
        min_ihtc: Minimum IHTC (W/m²·K)
        solidus_temp: Solidus temperature (°C)
        gap_onset_fraction: Temperature fraction where gap starts
        gap_growth_rate: Air gap growth rate (m/s)
        max_gap: Maximum air gap (m)
        time_step: Time step (s)
    
    Returns:
        Tuple of (updated_ihtc, updated_air_gaps)
    """
    n_voxels = casting_temps.shape[0]
    updated_ihtc = np.zeros(n_voxels, dtype=np.float32)
    updated_gaps = np.zeros(n_voxels, dtype=np.float32)
    
    gap_onset_temp = solidus_temp * gap_onset_fraction
    
    # Air thermal conductivity (W/m·K)
    k_air = 0.026
    
    for i in prange(n_voxels):
        T = casting_temps[i]
        t_contact = contact_times[i]
        current_gap = air_gaps[i]
        
        # Check if solidification has started at this voxel
        if T < solidus_temp and not is_solidified[i]:
            is_solidified[i] = True
        
        # Base IHTC with time decay
        if is_solidified[i]:
            # Exponential decay: h = h0 * exp(-λt) + h_min
            h_base = (initial_ihtc - min_ihtc) * np.exp(-ihtc_decay_rate * t_contact) + min_ihtc
        else:
            h_base = initial_ihtc
        
        # Air gap evolution
        if T < gap_onset_temp and is_solidified[i]:
            # Gap grows proportional to subcooling below onset temperature
            subcooling = (gap_onset_temp - T) / gap_onset_temp
            gap_growth = gap_growth_rate * subcooling * time_step
            new_gap = min(current_gap + gap_growth, max_gap)
        else:
            new_gap = current_gap
        
        updated_gaps[i] = new_gap
        
        # Combined IHTC with air gap resistance
        # 1/h_total = 1/h_contact + gap/k_air
        if new_gap > 1e-6:  # Significant gap
            R_contact = 1.0 / h_base
            R_gap = new_gap / k_air
            h_total = 1.0 / (R_contact + R_gap)
            updated_ihtc[i] = max(h_total, min_ihtc)
        else:
            updated_ihtc[i] = h_base
    
    return updated_ihtc, updated_gaps


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_mold_surface_temperature(
    mold_temps: np.ndarray,
    casting_temps: np.ndarray,
    ihtc: np.ndarray,
    mold_conductivity: float,
    mold_thickness: float,
    mold_back_temp: float,
    time_step: float,
    mold_density: float,
    mold_specific_heat: float
) -> np.ndarray:
    """
    Update mold surface temperature based on heat flux.
    
    Simple 1D model: mold surface heats up from casting, cools to back.
    
    Energy balance at mold surface:
        ρcₚ * dT/dt = h*(T_cast - T_mold) - k*(T_mold - T_back)/L
    
    Args:
        mold_temps: Current mold surface temperatures (°C)
        casting_temps: Casting surface temperatures (°C)
        ihtc: Interfacial heat transfer coefficients (W/m²·K)
        mold_conductivity: Mold thermal conductivity (W/m·K)
        mold_thickness: Effective mold thickness for 1D model (m)
        mold_back_temp: Temperature at mold back surface (°C)
        time_step: Time step (s)
        mold_density: Mold density (kg/m³)
        mold_specific_heat: Mold specific heat (J/kg·K)
    
    Returns:
        Updated mold surface temperatures (°C)
    """
    n_voxels = mold_temps.shape[0]
    new_mold_temps = np.zeros(n_voxels, dtype=np.float32)
    
    # Thermal mass per unit area
    thermal_mass = mold_density * mold_specific_heat * mold_thickness
    
    for i in prange(n_voxels):
        T_mold = mold_temps[i]
        T_cast = casting_temps[i]
        h = ihtc[i]
        
        # Heat flux from casting to mold surface
        q_in = h * (T_cast - T_mold)
        
        # Heat flux from mold surface to back (conduction)
        q_out = mold_conductivity * (T_mold - mold_back_temp) / mold_thickness
        
        # Net heat flux and temperature change
        q_net = q_in - q_out
        dT = q_net * time_step / thermal_mass
        
        new_mold_temps[i] = T_mold + dT
    
    return new_mold_temps


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_boundary_heat_flux(
    casting_temps: np.ndarray,
    mold_temps: np.ndarray,
    ihtc: np.ndarray
) -> np.ndarray:
    """
    Compute heat flux at casting-mold interface.
    
    q = h * (T_casting - T_mold)
    
    Args:
        casting_temps: Casting surface temperatures (°C)
        mold_temps: Mold surface temperatures (°C)
        ihtc: Heat transfer coefficients (W/m²·K)
    
    Returns:
        Heat flux at each surface voxel (W/m²)
    """
    n_voxels = casting_temps.shape[0]
    heat_flux = np.zeros(n_voxels, dtype=np.float32)
    
    for i in prange(n_voxels):
        heat_flux[i] = ihtc[i] * (casting_temps[i] - mold_temps[i])
    
    return heat_flux


class MoldInterfaceModel:
    """
    Models heat transfer at the casting-mold interface.
    
    Features:
    - Time-varying IHTC
    - Air gap formation
    - Mold surface temperature tracking
    - Different mold types (sand, metal, ceramic)
    
    Usage:
        model = MoldInterfaceModel(n_surface_voxels, mold_type)
        model.initialize(surface_temps, solidus_temp)
        
        # Each time step:
        heat_flux = model.update(surface_temps, time_step)
    """
    
    def __init__(self, 
                 n_surface_voxels: int,
                 mold_type: MoldType = MoldType.SAND_SILICA,
                 mold_props: Optional[MoldProperties] = None):
        """
        Initialize mold interface model.
        
        Args:
            n_surface_voxels: Number of surface voxels
            mold_type: Type of mold material
            mold_props: Custom mold properties (overrides mold_type)
        """
        self.n_surface = n_surface_voxels
        
        if mold_props is not None:
            self.mold_props = mold_props
        else:
            self.mold_props = get_mold_properties(mold_type)
        
        # Interface state
        self.state = InterfaceState(
            air_gap=np.zeros(n_surface_voxels, dtype=np.float32),
            ihtc=np.full(n_surface_voxels, 
                        self.mold_props.initial_ihtc, 
                        dtype=np.float32),
            contact_time=np.zeros(n_surface_voxels, dtype=np.float32),
            mold_surface_temp=np.full(n_surface_voxels,
                                     self.mold_props.initial_temperature,
                                     dtype=np.float32),
            is_solidified=np.zeros(n_surface_voxels, dtype=np.bool_)
        )
        
        # Mold 1D model parameters
        self.mold_thickness = 0.05  # 5cm effective thickness
        self.mold_back_temp = self.mold_props.initial_temperature
        
        # Solidification parameters (set during initialize)
        self.solidus_temp = 1400.0
        
        # Statistics
        self.total_heat_extracted = 0.0
        
    def initialize(self, 
                   surface_temps: np.ndarray,
                   solidus_temp: float,
                   liquidus_temp: float):
        """
        Initialize interface state.
        
        Args:
            surface_temps: Initial surface temperatures (°C)
            solidus_temp: Solidus temperature (°C)
            liquidus_temp: Liquidus temperature (°C)
        """
        self.solidus_temp = solidus_temp
        self.liquidus_temp = liquidus_temp
        
        # Reset state
        self.state.air_gap[:] = 0.0
        self.state.ihtc[:] = self.mold_props.initial_ihtc
        self.state.contact_time[:] = 0.0
        self.state.mold_surface_temp[:] = self.mold_props.initial_temperature
        self.state.is_solidified[:] = False
        
        print(f"\n  Mold interface model initialized:")
        print(f"    Mold type: {self.mold_props.name}")
        print(f"    Initial IHTC: {self.mold_props.initial_ihtc:.0f} W/m²·K")
        print(f"    Mold conductivity: {self.mold_props.thermal_conductivity:.2f} W/m·K")
        print(f"    Mold initial temp: {self.mold_props.initial_temperature:.0f}°C")
        print(f"    Surface voxels: {self.n_surface:,}")
    
    def update(self, 
               surface_temps: np.ndarray,
               time_step: float) -> np.ndarray:
        """
        Update interface state and compute boundary heat flux.
        
        Args:
            surface_temps: Current casting surface temperatures (°C)
            time_step: Time step (s)
        
        Returns:
            Heat flux at each surface voxel (W/m²)
        """
        # Update contact time for solidified voxels
        self.state.contact_time[self.state.is_solidified] += time_step
        
        # Update IHTC and air gap
        self.state.ihtc, self.state.air_gap = compute_ihtc_with_air_gap(
            surface_temps,
            self.state.contact_time,
            self.state.air_gap,
            self.state.is_solidified,
            self.mold_props.initial_ihtc,
            self.mold_props.ihtc_decay_rate,
            self.mold_props.min_ihtc,
            self.solidus_temp,
            self.mold_props.gap_onset_temp_fraction,
            self.mold_props.gap_growth_rate / 1000,  # mm to m
            self.mold_props.max_gap_thickness / 1000,  # mm to m
            time_step
        )
        
        # Update mold surface temperature
        self.state.mold_surface_temp = compute_mold_surface_temperature(
            self.state.mold_surface_temp,
            surface_temps,
            self.state.ihtc,
            self.mold_props.thermal_conductivity,
            self.mold_thickness,
            self.mold_back_temp,
            time_step,
            self.mold_props.density,
            self.mold_props.specific_heat
        )
        
        # Compute heat flux
        heat_flux = compute_boundary_heat_flux(
            surface_temps,
            self.state.mold_surface_temp,
            self.state.ihtc
        )
        
        # Track total heat extracted
        self.total_heat_extracted += np.sum(np.abs(heat_flux)) * time_step
        
        return heat_flux
    
    def get_boundary_temperature(self) -> np.ndarray:
        """Get mold surface temperature for thermal solver boundary condition"""
        return self.state.mold_surface_temp.copy()
    
    def get_ihtc(self) -> np.ndarray:
        """Get current IHTC values"""
        return self.state.ihtc.copy()
    
    def get_air_gap(self) -> np.ndarray:
        """Get current air gap thickness (m)"""
        return self.state.air_gap.copy()
    
    def get_statistics(self) -> Dict:
        """Get interface statistics"""
        return {
            'avg_ihtc': np.mean(self.state.ihtc),
            'min_ihtc': np.min(self.state.ihtc),
            'max_ihtc': np.max(self.state.ihtc),
            'avg_air_gap_um': np.mean(self.state.air_gap) * 1e6,  # m to μm
            'max_air_gap_um': np.max(self.state.air_gap) * 1e6,
            'avg_mold_surface_temp': np.mean(self.state.mold_surface_temp),
            'max_mold_surface_temp': np.max(self.state.mold_surface_temp),
            'solidified_fraction': np.mean(self.state.is_solidified),
            'total_heat_extracted_mj': self.total_heat_extracted / 1e6
        }


def print_mold_info(mold_type: MoldType):
    """Print detailed information about a mold type"""
    props = get_mold_properties(mold_type)
    
    print(f"\n{'='*60}")
    print(f"Mold: {props.name}")
    print(f"{'='*60}")
    print(f"\nThermal Properties:")
    print(f"  Conductivity:    {props.thermal_conductivity:.2f} W/(m·K)")
    print(f"  Specific Heat:   {props.specific_heat:.0f} J/(kg·K)")
    print(f"  Density:         {props.density:.0f} kg/m³")
    print(f"  Diffusivity:     {props.thermal_diffusivity:.2e} m²/s")
    print(f"\nInterface Parameters:")
    print(f"  Initial IHTC:    {props.initial_ihtc:.0f} W/(m²·K)")
    print(f"  IHTC Decay Rate: {props.ihtc_decay_rate:.4f} 1/s")
    print(f"  Minimum IHTC:    {props.min_ihtc:.0f} W/(m²·K)")
    print(f"\nAir Gap Parameters:")
    print(f"  Gap Onset:       {props.gap_onset_temp_fraction*100:.0f}% of solidus")
    print(f"  Max Gap:         {props.max_gap_thickness:.2f} mm")
    print(f"  Growth Rate:     {props.gap_growth_rate:.3f} mm/s")
    print(f"\nInitial Temperature: {props.initial_temperature:.0f}°C")
    print(f"{'='*60}\n")


# Convenience functions for quick setup
def create_mold_interface(
    mesh,  # AdaptiveVoxelMesh
    mold_type: MoldType = MoldType.SAND_SILICA,
    alloy=None  # AlloyProperties
) -> MoldInterfaceModel:
    """
    Create and initialize mold interface model from mesh.
    
    Args:
        mesh: AdaptiveVoxelMesh with surface voxel data
        mold_type: Type of mold
        alloy: Alloy properties (for solidification temperatures)
    
    Returns:
        Initialized MoldInterfaceModel
    """
    surface_indices = mesh.get_surface_voxel_indices()
    n_surface = len(surface_indices)
    
    model = MoldInterfaceModel(n_surface, mold_type)
    
    if alloy is not None:
        # Get initial surface temperatures (pouring temp)
        surface_temps = np.full(n_surface, alloy.pouring_temp, dtype=np.float32)
        model.initialize(surface_temps, alloy.solidus_temp, alloy.liquidus_temp)
    
    return model


if __name__ == '__main__':
    # Display all mold types
    for mold_type in get_all_mold_types():
        print_mold_info(mold_type)
