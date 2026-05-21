"""
Material Properties for Casting Alloys
WCB (Steel), CF8 (Stainless Steel), C958 (Aluminum Bronze),
C964 (CuNi), MONEL400 (Monel 400), IN625 (Inconel 625)

Extended properties for thermal simulation:
- Temperature-dependent conductivity and specific heat
- Convection parameters (viscosity, thermal expansion)
- Niyama critical thresholds
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class AlloyProperties:
    """
    Properties for casting simulation.
    
    Includes both basic properties and extended thermal/convection data.
    """
    name: str
    liquidus_temp: float  # °C
    solidus_temp: float   # °C
    pouring_temp: float   # °C
    density_liquid: float  # kg/m³
    density_solid: float   # kg/m³
    thermal_conductivity: float  # W/(m·K) at reference temperature
    specific_heat: float   # J/(kg·K) at reference temperature
    latent_heat: float     # J/kg
    linear_shrinkage: float  # % (volumetric shrinkage = 3x linear)
    feeding_distance_multiplier: float  # Multiplier of section thickness
    
    # Extended thermal properties (for temperature-dependent calculations)
    conductivity_liquid: Optional[float] = None  # W/(m·K) in liquid state
    specific_heat_liquid: Optional[float] = None  # J/(kg·K) in liquid state
    
    # Convection properties (for natural convection modeling)
    dynamic_viscosity: float = 0.006  # Pa·s (liquid state)
    thermal_expansion: float = 1.2e-4  # 1/K (liquid state)
    viscosity_temp_sensitivity: float = 1.2  # Exponential factor over freezing range
    
    # Niyama criterion threshold (defect risk below this value)
    niyama_threshold: float = 1.0
    
    # Hot Tear Susceptibility (HTS) coefficient
    # Higher value = more susceptible to hot tearing
    # Based on: freezing range, ductility at BTR, shrinkage strain
    # WCB ~0.30 (moderate - narrow freezing range)
    # CF8 ~0.65 (high - wide freezing range, low ductility)
    # C958 ~0.20 (low - excellent ductility, good feeding)
    hot_tear_susceptibility: float = 0.30
    
    # Solid-state thermal expansion for mechanical strain calculation (1/K)
    # Different from liquid thermal expansion used for convection
    solid_thermal_expansion: float = 1.2e-5
    
    # Critical strain for BTR ductility (dimensionless)
    # Strain above this in the BTR leads to hot tear formation
    btr_critical_strain: float = 0.008
    
    # Healing factor for hot tear risk reduction via feeding (0-1)
    # Higher values mean better feeding reduces hot tear risk more
    hot_tear_healing_factor: float = 0.55
    
    # Brittle Temperature Range (BTR) fraction solid bounds
    # Range where alloy has minimum ductility (semi-solid "mush")
    # Hot tears form in this range when tensile strain exceeds ductility
    # btr_start: fraction solid where ductility drops (typically 0.85-0.92)
    # btr_end: fraction solid where alloy regains ductility (typically 0.97-0.99)
    btr_fraction_solid_start: float = 0.90
    btr_fraction_solid_end: float = 0.98
    
    # Critical fraction solid: point where interdendritic feeding stops
    # Below this threshold, liquid can flow freely through mushy zone
    # Above this threshold, dendrites block flow (coherency point)
    # WCB ~0.50-0.55, CF8 ~0.45-0.50, C958 ~0.60-0.65 (better fluidity)
    critical_fraction_solid: float = 0.50
    
    # Secondary dendrite arm spacing (SDAS) in micrometers
    # Affects interdendritic permeability: smaller SDAS = lower permeability
    # Typical values: 50-200 μm depending on cooling rate
    dendrite_arm_spacing: float = 100.0  # μm
    sdas_reference_cooling_rate: float = 1.0  # °C/s reference for SDAS scaling
    sdas_cooling_rate_exponent: float = -0.33  # SDAS ~ cooling_rate^n
    
    # Sand burn-in susceptibility coefficient (0-1)
    # Higher values = more likely to penetrate sand mold
    # Factors: superheat tolerance, fluidity, surface tension
    # WCB ~0.45 (moderate - high superheat, good fluidity)
    # CF8 ~0.35 (lower - higher viscosity, lower superheat)
    # C958 ~0.55 (higher - excellent fluidity, wide superheat range)
    # C964 ~0.48 (high fluidity from copper-nickel composition)
    # MONEL400 ~0.40 (moderate cast fluidity and permeability)
    # IN625 ~0.37 (moderate)
    sand_burn_in_susceptibility: float = 0.40
    
    def __post_init__(self):
        """Set default liquid properties if not specified"""
        if self.conductivity_liquid is None:
            # Liquid typically ~70% of solid conductivity
            self.conductivity_liquid = self.thermal_conductivity * 0.7
        if self.specific_heat_liquid is None:
            # Liquid typically ~10% higher specific heat
            self.specific_heat_liquid = self.specific_heat * 1.1
    
    @property
    def thermal_diffusivity(self) -> float:
        """Calculate thermal diffusivity: k / (ρ * cp)"""
        return self.thermal_conductivity / (self.density_liquid * self.specific_heat)
    
    @property
    def thermal_diffusivity_liquid(self) -> float:
        """Thermal diffusivity in liquid state"""
        return self.conductivity_liquid / (self.density_liquid * self.specific_heat_liquid)
    
    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity: ν = μ/ρ (m²/s)"""
        return self.dynamic_viscosity / self.density_liquid
    
    @property
    def prandtl_number(self) -> float:
        """Prandtl number: Pr = ν/α"""
        return self.kinematic_viscosity / self.thermal_diffusivity_liquid
    
    @property
    def freezing_range(self) -> float:
        """Temperature range between liquidus and solidus (°C)"""
        return self.liquidus_temp - self.solidus_temp
    
    @property
    def volumetric_shrinkage(self) -> float:
        """Volumetric shrinkage percentage (volume loss during solidification)"""
        # Volume decreases as density increases: V_solid/V_liquid = ρ_liquid/ρ_solid
        # Shrinkage = (V_liquid - V_solid)/V_liquid = 1 - (ρ_liquid/ρ_solid)
        return (1 - (self.density_liquid / self.density_solid)) * 100
    
    @property
    def density_change_ratio(self) -> float:
        """Density increase ratio (solid/liquid)"""
        return self.density_solid / self.density_liquid
    
    def get_feeding_distance(self, thickness: float) -> float:
        """
        Calculate feeding distance based on section thickness
        
        Args:
            thickness: Section thickness in inches
        
        Returns:
            Feeding distance in inches
        """
        return thickness * self.feeding_distance_multiplier

    def get_dynamic_viscosity(
        self,
        temperature: Optional[float] = None,
        fraction_solid: Optional[float] = None,
    ) -> float:
        """
        Estimate dynamic viscosity in liquid/mushy regions.

        The base viscosity is specified near liquid conditions. Viscosity
        increases as the melt cools toward solidus and rises further in mush.
        """
        viscosity = float(self.dynamic_viscosity)
        freeze_range = max(float(self.freezing_range), 1.0)

        if temperature is not None:
            undercooling = max(float(self.liquidus_temp) - float(temperature), 0.0)
            temp_factor = np.exp(self.viscosity_temp_sensitivity * (undercooling / freeze_range))
            viscosity *= float(temp_factor)

        if fraction_solid is not None:
            fs = float(max(0.0, min(1.0, fraction_solid)))
            if fs >= self.critical_fraction_solid:
                overshoot = (fs - self.critical_fraction_solid) / max(
                    1.0 - self.critical_fraction_solid,
                    1e-6,
                )
                viscosity *= float(np.exp(3.0 * overshoot))

        return float(np.clip(viscosity, self.dynamic_viscosity * 0.5, self.dynamic_viscosity * 250.0))

    def get_sdas_from_cooling_rate(self, cooling_rate: Optional[float]) -> float:
        """
        Estimate SDAS (μm) from local cooling rate.

        Uses a power-law around the alloy's baseline SDAS calibration.
        """
        if cooling_rate is None:
            return float(self.dendrite_arm_spacing)
        rate = max(float(cooling_rate), 1e-4)
        ref_rate = max(float(self.sdas_reference_cooling_rate), 1e-4)
        sdas = float(self.dendrite_arm_spacing) * ((rate / ref_rate) ** float(self.sdas_cooling_rate_exponent))
        return float(np.clip(sdas, 10.0, 500.0))

    def get_permeability(
        self,
        fraction_solid: float,
        *,
        sdas_um: Optional[float] = None,
        cooling_rate: Optional[float] = None,
    ) -> float:
        """
        Calculate interdendritic permeability using Kozeny-Carman equation.
        
        Permeability determines how easily liquid metal can flow through
        the mushy zone (dendritic network) to feed shrinkage.
        
        K = (λ² * (1 - fs)³) / (180 * fs²)
        
        where:
            λ = secondary dendrite arm spacing (SDAS) in meters
            fs = fraction solid (0-1)
        
        When fs > critical_fraction_solid, permeability drops to near-zero
        (coherency point - dendrites touch and block flow).
        
        Args:
            fraction_solid: Fraction solid (0-1)
        
        Returns:
            Permeability in m² (Darcy's law units)
        """
        # Convert SDAS from μm to m
        if sdas_um is None:
            sdas_um = self.get_sdas_from_cooling_rate(cooling_rate)
        lambda_m = float(sdas_um) * 1e-6
        
        # Clamp fraction solid to valid range
        fs = max(0.001, min(fraction_solid, 0.999))
        
        # Check if above critical fraction solid (coherency point)
        if fs >= self.critical_fraction_solid:
            # Above coherency: permeability drops exponentially
            # Not exactly zero but very low (prevents numerical issues)
            overshoot = (fs - self.critical_fraction_solid) / (1.0 - self.critical_fraction_solid)
            decay_factor = np.exp(-10 * overshoot)  # Rapid decay
            fs_effective = self.critical_fraction_solid
        else:
            decay_factor = 1.0
            fs_effective = fs
        
        # Kozeny-Carman equation for permeability
        liquid_fraction = 1.0 - fs_effective
        
        # Avoid division by zero
        if fs_effective < 0.001:
            return 1e-8  # Very high permeability (essentially pure liquid)
        
        K = (lambda_m**2 * liquid_fraction**3) / (180 * fs_effective**2)
        
        return K * decay_factor
    
    def get_pressure_drop(
        self,
        fraction_solid: float,
        flow_velocity: float,
        viscosity: float,
        path_length: float,
        *,
        sdas_um: Optional[float] = None,
        cooling_rate: Optional[float] = None,
    ) -> float:
        """
        Calculate interdendritic pressure drop using Darcy's Law.
        
        ΔP = (μ * L * v) / K
        
        This is the pressure required to push liquid metal through
        the mushy zone (dendritic network).
        
        Args:
            fraction_solid: Fraction solid (0-1)
            flow_velocity: Feed metal velocity (m/s)
            viscosity: Dynamic viscosity (Pa·s)
            path_length: Length of mushy zone path (m)
        
        Returns:
            Pressure drop in Pa
        """
        K = self.get_permeability(
            fraction_solid,
            sdas_um=sdas_um,
            cooling_rate=cooling_rate,
        )
        
        # Prevent division by zero (very low permeability = infinite pressure drop)
        if K < 1e-20:
            return 1e10  # Very high pressure drop (feeding impossible)
        
        delta_P = (viscosity * path_length * flow_velocity) / K
        
        return delta_P


# Material database
ALLOY_DATABASE: Dict[str, AlloyProperties] = {
    'WCB': AlloyProperties(
        name='WCB (Carbon Steel Casting)',
        liquidus_temp=1460,  # 2660°F
        solidus_temp=1399,   # 2550°F
        pouring_temp=1621,   # 2950°F
        density_liquid=7200,
        density_solid=7850,
        thermal_conductivity=45,
        specific_heat=600,
        latent_heat=272000,
        linear_shrinkage=2.0,  # 2% linear, ~6% volumetric
        feeding_distance_multiplier=6.0,
        # Extended properties
        conductivity_liquid=32,  # W/(m·K)
        specific_heat_liquid=660,  # J/(kg·K)
        dynamic_viscosity=0.006,  # Pa·s
        thermal_expansion=1.2e-4,  # 1/K
        niyama_threshold=1.0,  # Standard steel threshold
        # Hot tear properties - carbon steel is moderate susceptibility
        hot_tear_susceptibility=0.30,  # Moderate - narrow freezing range helps
        solid_thermal_expansion=1.2e-5,  # 1/K solid-state thermal expansion
        btr_critical_strain=0.008,  # Critical strain for BTR ductility
        hot_tear_healing_factor=0.55,  # Healing factor for feeding relief
        btr_fraction_solid_start=0.90,  # BTR starts at 90% solid
        btr_fraction_solid_end=0.98,    # BTR ends at 98% solid
        # Feeding/permeability properties
        critical_fraction_solid=0.52,  # Coherency point for carbon steel
        dendrite_arm_spacing=80.0,  # μm - typical for sand casting cooling rates
        # Sand burn-in - moderate susceptibility (high superheat but good surface tension)
        sand_burn_in_susceptibility=0.45
    ),
    
    'CF8': AlloyProperties(
        name='CF8 (Stainless Steel 304)',
        liquidus_temp=1450,  # 2642°F
        solidus_temp=1400,   # 2552°F
        pouring_temp=1621,   # 2950°F
        density_liquid=7400,
        density_solid=8000,
        thermal_conductivity=16.2,
        specific_heat=500,
        latent_heat=260000,
        linear_shrinkage=2.5,  # 2.5% linear, ~7.5% volumetric
        feeding_distance_multiplier=5.5,
        # Extended properties
        conductivity_liquid=12,  # W/(m·K) - lower than carbon steel
        specific_heat_liquid=550,
        dynamic_viscosity=0.007,  # Pa·s - slightly higher viscosity
        thermal_expansion=1.5e-4,  # 1/K - higher expansion
        niyama_threshold=0.7,  # Lower threshold due to poor conductivity
        # Hot tear properties - stainless is HIGHLY susceptible
        hot_tear_susceptibility=0.65,  # High - wide freezing range, low ductility at BTR
        solid_thermal_expansion=1.7e-5,  # 1/K solid-state thermal expansion
        btr_critical_strain=0.004,  # Lower critical strain - more susceptible
        hot_tear_healing_factor=0.35,  # Poor healing - lower feeding effectiveness
        btr_fraction_solid_start=0.85,  # Earlier onset due to alloy segregation
        btr_fraction_solid_end=0.99,    # Extended BTR due to wide freezing range
        # Feeding/permeability properties - stainless has narrower feeding window
        critical_fraction_solid=0.48,  # Lower coherency due to wide freezing range
        dendrite_arm_spacing=60.0,  # μm - finer structure due to alloying
        # Sand burn-in - lower susceptibility (higher viscosity, lower superheat tolerance)
        sand_burn_in_susceptibility=0.35
    ),
    
    'C958': AlloyProperties(
        name='C958 (Aluminum Bronze)',
        liquidus_temp=1063,  # 1945°F
        solidus_temp=1043,   # 1910°F
        pouring_temp=1288,   # 2350°F
        density_liquid=7100,  # ~7% less than solid
        density_solid=7640,   # ASTM B148 C95800 standard
        thermal_conductivity=36,  # W/(m·K) per technical sources
        specific_heat=440,    # J/(kg·K) standard value
        latent_heat=205000,
        linear_shrinkage=1.5,  # 1.5% linear, ~4.5% volumetric
        feeding_distance_multiplier=8.0,
        # Extended properties
        conductivity_liquid=28,  # W/(m·K)
        specific_heat_liquid=480,
        dynamic_viscosity=0.004,  # Pa·s - lower viscosity (better fluidity)
        thermal_expansion=1.0e-4,  # 1/K
        niyama_threshold=1.2,  # Higher threshold (better feeding)
        # Hot tear properties - bronze has LOW susceptibility
        hot_tear_susceptibility=0.20,  # Low - excellent ductility, narrow freezing range
        solid_thermal_expansion=1.8e-5,  # 1/K solid-state thermal expansion
        btr_critical_strain=0.012,  # Higher critical strain - more ductile
        hot_tear_healing_factor=0.75,  # Excellent healing due to good feeding
        btr_fraction_solid_start=0.92,  # Later onset - more ductile at high fs
        btr_fraction_solid_end=0.97,    # Shorter BTR - regains ductility quickly
        # Feeding/permeability properties - bronze has excellent feeding
        critical_fraction_solid=0.62,  # Higher coherency - better fluidity
        dendrite_arm_spacing=120.0,  # μm - coarser structure, better permeability
        # Sand burn-in - higher susceptibility (excellent fluidity, wide superheat range)
        sand_burn_in_susceptibility=0.55
    ),

    'C964': AlloyProperties(
        name='C964 (CuNi)',
        liquidus_temp=1175,  # 2147°F
        solidus_temp=1140,   # 2084°F
        pouring_temp=1280,   # 2336°F
        density_liquid=8700,
        density_solid=8860,
        thermal_conductivity=22,  # W/(m·K)
        specific_heat=500,  # J/(kg·K)
        latent_heat=186000,
        linear_shrinkage=1.8,  # 1.8% linear, ~5.4% volumetric
        feeding_distance_multiplier=7.8,
        # Extended properties
        conductivity_liquid=16,  # W/(m·K)
        specific_heat_liquid=555,  # J/(kg·K)
        dynamic_viscosity=0.0048,  # Pa·s - good cast fluidity
        thermal_expansion=1.3e-4,  # 1/K
        niyama_threshold=1.1,  # Good feeding and lower defect susceptibility
        # Hot tear properties - copper-nickel has moderate susceptibility
        hot_tear_susceptibility=0.28,  # Moderate susceptibility
        solid_thermal_expansion=1.6e-5,  # 1/K solid-state thermal expansion
        btr_critical_strain=0.007,  # Moderate critical strain
        hot_tear_healing_factor=0.50,  # Moderate healing effectiveness
        btr_fraction_solid_start=0.89,  # Later onset due to good ductility
        btr_fraction_solid_end=0.97,    # Regains ductility more quickly
        # Feeding/permeability properties
        critical_fraction_solid=0.56,  # Moderate coherency
        dendrite_arm_spacing=95.0,  # μm
        # Sand burn-in - good fluidity and superheat range
        sand_burn_in_susceptibility=0.48
    ),

    'CA6NM': AlloyProperties(
        name='CA6NM (Cast Steel)',
        liquidus_temp=1440,  # 2624°F
        solidus_temp=1385,   # 2525°F
        pouring_temp=1560,   # 2840°F
        density_liquid=7600,
        density_solid=7900,
        thermal_conductivity=28,  # W/(m·K)
        specific_heat=485,
        latent_heat=252000,
        linear_shrinkage=2.1,  # 2.1% linear, ~6.3% volumetric
        feeding_distance_multiplier=5.9,
        # Extended properties
        conductivity_liquid=20,  # W/(m·K)
        specific_heat_liquid=530,
        dynamic_viscosity=0.0064,  # Pa·s
        thermal_expansion=1.35e-4,  # 1/K
        niyama_threshold=0.95,
        # Hot tear properties - medium susceptibility
        hot_tear_susceptibility=0.38,
        solid_thermal_expansion=1.3e-5,  # 1/K solid-state thermal expansion
        btr_critical_strain=0.006,  # Moderate critical strain
        hot_tear_healing_factor=0.45,  # Moderate healing effectiveness
        btr_fraction_solid_start=0.87,
        btr_fraction_solid_end=0.98,
        # Feeding/permeability properties
        critical_fraction_solid=0.54,
        dendrite_arm_spacing=78.0,  # μm
        # Sand burn-in - moderate susceptibility
        sand_burn_in_susceptibility=0.43
    ),

    'MONEL400': AlloyProperties(
        name='MONEL400 (Monel 400 Nickel-Copper)',
        liquidus_temp=1370,  # 2498°F
        solidus_temp=1320,   # 2408°F
        pouring_temp=1490,   # 2714°F
        density_liquid=8100,
        density_solid=8350,
        thermal_conductivity=19,  # W/(m·K)
        specific_heat=480,  # J/(kg·K)
        latent_heat=250000,
        linear_shrinkage=1.9,  # 1.9% linear, ~5.7% volumetric
        feeding_distance_multiplier=6.4,
        # Extended properties
        conductivity_liquid=14,  # W/(m·K)
        specific_heat_liquid=530,  # J/(kg·K)
        dynamic_viscosity=0.0063,  # Pa·s
        thermal_expansion=1.45e-4,  # 1/K
        niyama_threshold=0.85,  # Moderate threshold
        # Hot tear properties - nickel alloys are moderate for hot tear
        hot_tear_susceptibility=0.42,
        solid_thermal_expansion=1.4e-5,  # 1/K solid-state thermal expansion
        btr_critical_strain=0.005,  # Lower critical strain for Ni-based
        hot_tear_healing_factor=0.40,  # Moderate healing effectiveness
        btr_fraction_solid_start=0.86,  # Wider/earlier BTR than mild steel
        btr_fraction_solid_end=0.98,
        # Feeding/permeability properties
        critical_fraction_solid=0.53,  # Moderate coherency point
        dendrite_arm_spacing=85.0,  # μm
        # Sand burn-in - moderate susceptibility
        sand_burn_in_susceptibility=0.39
    ),

    'IN625': AlloyProperties(
        name='IN625 (Inconel 625)',
        liquidus_temp=1330,  # 2426°F
        solidus_temp=1260,   # 2300°F
        pouring_temp=1420,   # 2588°F
        density_liquid=8250,
        density_solid=8480,
        thermal_conductivity=13,  # W/(m·K)
        specific_heat=435,  # J/(kg·K)
        latent_heat=255000,
        linear_shrinkage=2.0,  # 2.0% linear, ~6.0% volumetric
        feeding_distance_multiplier=5.8,
        # Extended properties
        conductivity_liquid=10,  # W/(m·K)
        specific_heat_liquid=485,  # J/(kg·K)
        dynamic_viscosity=0.0068,  # Pa·s
        thermal_expansion=1.5e-4,  # 1/K
        niyama_threshold=0.9,  # Conservative threshold
        # Hot tear properties - high-alloy materials often higher susceptibility
        hot_tear_susceptibility=0.58,
        solid_thermal_expansion=1.5e-5,  # 1/K solid-state thermal expansion
        btr_critical_strain=0.004,  # Low critical strain - high susceptibility
        hot_tear_healing_factor=0.30,  # Poor healing due to restricted feeding
        btr_fraction_solid_start=0.84,  # Earlier onset with wider BTR
        btr_fraction_solid_end=0.99,
        # Feeding/permeability properties
        critical_fraction_solid=0.50,  # Lower coherency due to restricted feeding
        dendrite_arm_spacing=70.0,  # μm
        # Sand burn-in - controlled by viscosity and oxidation behavior
        sand_burn_in_susceptibility=0.37
    )
}


def get_alloy(name: str) -> AlloyProperties:
    """
    Get alloy properties by name
    
    Args:
        name: Alloy name ('WCB', 'CF8', 'C958', 'C964', 'CA6NM', 'MONEL400', 'IN625')
    
    Returns:
        AlloyProperties object
    
    Raises:
        KeyError: If alloy not found
    """
    if name not in ALLOY_DATABASE:
        available = ', '.join(ALLOY_DATABASE.keys())
        raise KeyError(f"Alloy '{name}' not found. Available: {available}")
    
    return ALLOY_DATABASE[name]


def get_all_alloy_names() -> list:
    """Get list of all available alloy names"""
    return list(ALLOY_DATABASE.keys())


def print_alloy_info(name: str):
    """Print detailed information about an alloy"""
    alloy = get_alloy(name)
    
    print(f"\n{'='*60}")
    print(f"Alloy: {alloy.name}")
    print(f"{'='*60}")
    print(f"Temperature Range:")
    print(f"  Liquidus:        {alloy.liquidus_temp}°C")
    print(f"  Solidus:         {alloy.solidus_temp}°C")
    print(f"  Freezing range:  {alloy.freezing_range}°C")
    print(f"  Pouring:         {alloy.pouring_temp}°C")
    print(f"\nDensity:")
    print(f"  Liquid:          {alloy.density_liquid} kg/m³")
    print(f"  Solid:           {alloy.density_solid} kg/m³")
    print(f"  Density Ratio:   {alloy.density_change_ratio:.3f}")
    print(f"\nThermal Properties (Solid):")
    print(f"  Conductivity:    {alloy.thermal_conductivity} W/(m·K)")
    print(f"  Specific Heat:   {alloy.specific_heat} J/(kg·K)")
    print(f"  Diffusivity:     {alloy.thermal_diffusivity:.6f} m²/s")
    print(f"\nThermal Properties (Liquid):")
    print(f"  Conductivity:    {alloy.conductivity_liquid} W/(m·K)")
    print(f"  Specific Heat:   {alloy.specific_heat_liquid} J/(kg·K)")
    print(f"  Diffusivity:     {alloy.thermal_diffusivity_liquid:.6f} m²/s")
    print(f"\nConvection Properties:")
    print(f"  Dyn. Viscosity:  {alloy.dynamic_viscosity:.4f} Pa·s")
    print(f"  Kin. Viscosity:  {alloy.kinematic_viscosity:.2e} m²/s")
    print(f"  Therm. Expansion:{alloy.thermal_expansion:.2e} 1/K")
    print(f"  Prandtl Number:  {alloy.prandtl_number:.4f}")
    print(f"\nLatent Heat:       {alloy.latent_heat} J/kg")
    print(f"\nShrinkage:")
    print(f"  Linear:          {alloy.linear_shrinkage}%")
    print(f"  Volumetric:      {alloy.volumetric_shrinkage:.2f}%")
    print(f"\nFeeding:")
    print(f"  Distance Factor: {alloy.feeding_distance_multiplier}x thickness")
    print(f"  Critical fs:     {alloy.critical_fraction_solid:.2f} (coherency point)")
    print(f"  SDAS:            {alloy.dendrite_arm_spacing:.0f} μm")
    print(f"\nPermeability (at 50% solid):")
    perm_at_50 = alloy.get_permeability(0.5)
    print(f"  K(fs=0.5):       {perm_at_50:.2e} m²")
    print(f"\nNiyama Threshold:  {alloy.niyama_threshold}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Display all alloys
    for alloy_name in get_all_alloy_names():
        print_alloy_info(alloy_name)
