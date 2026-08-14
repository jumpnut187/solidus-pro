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

    # Strain-rate sensitivity of BTR critical strain (dimensionless, default 0 = no
    # sensitivity). Hot tearing is a kinetic phenomenon: at higher strain rates
    # the effective BTR ductility drops because the mush has less time to
    # accommodate. Reference: Magnin et al. 1996; Novikov 1966.
    # epsilon_c_eff = epsilon_c * (1 - sensitivity * log10(1 + strain_rate / ref))
    # Typical values: steels ~0.10, Ni-base ~0.15, Al-bronze ~0.08
    btr_critical_strain_rate_sensitivity: float = 0.0

    # Reference strain rate for the BTR critical strain sensitivity (1/s).
    # 1e-3 /s is a reasonable "static" reference (corresponds to foundry cooling
    # in the minutes-timescale). Above this, the effective critical strain drops.
    btr_critical_strain_ref_rate: float = 1.0e-3

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

    # Cooling-rate exponent for BTR bound adjustment (default 0 = no adjustment).
    # Back diffusion in the solid reduces the effective BTR width (Liu and Kou
    # 2015, Acta Mater 100:359). At faster cooling, less back diffusion
    # occurs, the BTR widens, and the alloy becomes more susceptible.
    # The exponent is POSITIVE: btr_width ~ (cooling_rate / ref) ** exponent.
    # Typical values: steels 0.05, Ni-base 0.10, Al-alloys 0.08
    btr_cooling_rate_exponent: float = 0.0

    # Reference cooling rate for BTR cooling-rate adjustment (°C/s).
    # 1.0 °C/s is a reasonable foundry reference (sand casting of a small block).
    btr_btr_cooling_rate_ref: float = 1.0

    # Scheil solidification parameters (approximate binary-system values; see
    # each alloy entry for the solvent metal and partition coefficient).
    # melting_point_solvent: melting point of the pure solvent metal (°C).
    # partition_coefficient: solute partition coefficient k = Cs/Cl.
    # eutectic_temperature: eutectic reaction temperature (°C); None = none.
    # eutectic_fraction: fraction solid at which the eutectic reaction starts.
    melting_point_solvent: Optional[float] = None  # °C; None = surrogate fallback
    partition_coefficient: float = 0.17  # Cs/Cl
    eutectic_temperature: Optional[float] = None  # °C
    eutectic_fraction: float = 0.0  # 0 = no eutectic

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
    
    # Gas porosity analysis
    gas_sensitivity: float = 0.5  # 0-1, alloy-specific multiplier for gas risk
    
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
        """Density-derived volume loss from liquid volume, as a percentage."""
        # Volume decreases as density increases: V_solid/V_liquid = ρ_liquid/ρ_solid
        # Shrinkage = (V_liquid - V_solid)/V_liquid = 1 - (ρ_liquid/ρ_solid)
        # ``linear_shrinkage`` is catalog metadata; this density relation is the
        # authoritative value used by the solver for volumetric compensation.
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

    def vectorized_dynamic_viscosity(
        self,
        temperatures: np.ndarray,
        fraction_solids: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized dynamic viscosity over arrays of voxels.

        Mirrors :meth:`get_dynamic_viscosity` element-wise so callers
        can avoid Python-level loops over potentially tens of thousands
        of voxels.
        """
        temps = np.asarray(temperatures, dtype=np.float64)
        fss = np.asarray(fraction_solids, dtype=np.float64)
        viscosity = float(self.dynamic_viscosity)
        freeze_range = max(float(self.freezing_range), 1.0)

        result = np.full(temps.shape, viscosity, dtype=np.float64)

        undercooling = np.maximum(float(self.liquidus_temp) - temps, 0.0)
        temp_factor = np.exp(
            float(self.viscosity_temp_sensitivity) * (undercooling / freeze_range)
        )
        result = result * temp_factor

        fs_clamped = np.clip(fss, 0.0, 1.0)
        above_critical = fs_clamped >= float(self.critical_fraction_solid)
        if np.any(above_critical):
            denom = max(1.0 - float(self.critical_fraction_solid), 1e-6)
            overshoot = (fs_clamped - float(self.critical_fraction_solid)) / denom
            mush_factor = np.where(above_critical, np.exp(3.0 * overshoot), 1.0)
            result = result * mush_factor

        low = float(self.dynamic_viscosity) * 0.5
        high = float(self.dynamic_viscosity) * 250.0
        return np.clip(result, low, high)

    def vectorized_sdas(
        self,
        cooling_rates: Optional[np.ndarray],
    ) -> np.ndarray:
        """Vectorized SDAS calculation from cooling-rate arrays."""
        if cooling_rates is None:
            return np.full(1, float(self.dendrite_arm_spacing), dtype=np.float64)

        rates = np.asarray(cooling_rates, dtype=np.float64)
        clamped = np.maximum(rates, 1e-4)
        ref = max(float(self.sdas_reference_cooling_rate), 1e-4)
        sdas = float(self.dendrite_arm_spacing) * (
            (clamped / ref) ** float(self.sdas_cooling_rate_exponent)
        )
        return np.clip(sdas, 10.0, 500.0)

    def vectorized_permeability(
        self,
        fraction_solids: np.ndarray,
        sdas_um: np.ndarray,
    ) -> np.ndarray:
        """Vectorized Kozeny-Carman permeability."""
        fs = np.clip(np.asarray(fraction_solids, dtype=np.float64), 0.001, 0.999)
        sdas_arr = np.asarray(sdas_um, dtype=np.float64)
        lambda_m = sdas_arr * 1e-6

        decay = np.ones_like(fs)
        above = fs >= float(self.critical_fraction_solid)
        if np.any(above):
            denom = max(1.0 - float(self.critical_fraction_solid), 1e-6)
            overshoot = (fs - float(self.critical_fraction_solid)) / denom
            decay = np.where(above, np.exp(-10.0 * overshoot), 1.0)
            fs_eff = np.where(above, float(self.critical_fraction_solid), fs)
        else:
            fs_eff = fs

        liquid_fraction = 1.0 - fs_eff
        fs_eff_safe = np.where(fs_eff < 0.001, 0.001, fs_eff)
        K = (lambda_m ** 2 * liquid_fraction ** 3) / (180.0 * fs_eff_safe ** 2)
        K = np.where(fs_eff < 0.001, 1e-8, K * decay)
        return K

    def vectorized_pressure_drop(
        self,
        fraction_solids: np.ndarray,
        flow_velocity: float,
        viscosities: np.ndarray,
        path_lengths_m: np.ndarray,
        sdas_um: np.ndarray,
        feeder_neck_factors: np.ndarray,
    ) -> np.ndarray:
        """Vectorized Darcy pressure drop with feeder-neck resistance."""
        K = self.vectorized_permeability(fraction_solids, sdas_um)
        delta_P = (viscosities * path_lengths_m * float(flow_velocity)) / np.maximum(K, 1e-20)
        delta_P = np.where(K < 1e-20, 1e10, delta_P)
        return delta_P * feeder_neck_factors


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
        btr_critical_strain_rate_sensitivity=0.10,  # Mild rate-dependence
        hot_tear_healing_factor=0.55,  # Healing factor for feeding relief
        btr_fraction_solid_start=0.90,  # BTR starts at 90% solid
        btr_fraction_solid_end=0.98,    # BTR ends at 98% solid
        btr_cooling_rate_exponent=0.05,  # Mild BTR-widening with cooling rate
        # Scheil (binary approx.): pure Fe solvent, C in austenite k ~0.13-0.2
        melting_point_solvent=1538,  # °C pure Fe
        partition_coefficient=0.17,  # C in austenite ~0.13-0.2
        # Feeding/permeability properties
        critical_fraction_solid=0.52,  # Coherency point for carbon steel
        dendrite_arm_spacing=80.0,  # μm - typical for sand casting cooling rates
        # Sand burn-in - moderate susceptibility (high superheat but good surface tension)
        sand_burn_in_susceptibility=0.45,
        # Gas porosity - moderate sensitivity (steel is intermediate)
        gas_sensitivity=0.50
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
        btr_critical_strain_rate_sensitivity=0.15,  # Strong rate-dependence (austenitic)
        hot_tear_healing_factor=0.35,  # Poor healing - lower feeding effectiveness
        btr_fraction_solid_start=0.85,  # Earlier onset due to alloy segregation
        btr_fraction_solid_end=0.99,    # Extended BTR due to wide freezing range
        btr_cooling_rate_exponent=0.08,  # Moderate BTR-widening (segregation-driven)
        # Scheil (binary approx.): pure Fe solvent, austenitic Ni/Cr system
        melting_point_solvent=1538,  # °C pure Fe
        partition_coefficient=0.7,  # alloying elements partition weakly
        # Feeding/permeability properties - stainless has narrower feeding window
        critical_fraction_solid=0.48,  # Lower coherency due to wide freezing range
        dendrite_arm_spacing=60.0,  # μm - finer structure due to alloying
        # Sand burn-in - lower susceptibility (higher viscosity, lower superheat tolerance)
        sand_burn_in_susceptibility=0.35,
        # Gas porosity - CF8M ~0.60 (stainless more sensitive)
        gas_sensitivity=0.60
    ),
    
    'C958': AlloyProperties(
        name='C958 (Aluminum Bronze)',
        # ---------------------------------------------------------------
        # Sources (7-source cross-check, C95800 / CuAl10Fe5Ni5-C / CC333G):
        #   CDA / copper.org (cast):    liquidus 1060 °C, solidus 1043 °C
        #   Concast C95800 spec sheet:  1060 °C, 1043 °C
        #   Cast Technologies PDF:      1060 °C, 1045 °C
        #   St. Paul Foundry datasheet: 1910-1940 °F (~1043-1060 °C)
        # Adopted: liquidus 1060 °C, solidus 1043 °C (CDA / Concast).
        # ---------------------------------------------------------------
        liquidus_temp=1060,  # 1940°F — CDA liquidus (was 1063)
        solidus_temp=1043,   # 1910°F — CDA solidus (was 1043, unchanged)
        pouring_temp=1255,   # 2300°F — St. Paul Foundry pour for light castings
        # Density — CDA copper.org, AZoM, MatMake, Cast Technologies all give
        #   7.64 g/cm³ (7640 kg/m³) at 20 °C for the solid casting.
        #   C964 liquid density is typically 5-7 % below solid for Cu-Al bronzes
        #   (rule-of-mixtures + thermal expansion); adopt 7.0 %.
        density_liquid=7100,
        density_solid=7640,
        # Thermal conductivity (20 °C reference). All 5 sources agree exactly:
        #   CDA copper.org: 20.8 Btu/(hr·ft·°F) → 36.0 W/(m·K)
        #   Cast Technologies: 36.0 W/m-K
        #   AZoM C95800 datasheet: 36.0 W/mK
        #   MatMake C95800: 36 W/m·K
        #   Concast C95800 spec: 36 W/m·K at 20 °C
        thermal_conductivity=36,    # (unchanged, already correct)
        specific_heat=440,         # J/(kg·K) — matches all 4 sources (unchanged)
        # Latent heat of fusion — MakeItFrom lists 230 J/g for C95800
        # specifically (the MakeItFrom AISI-316 vs C95800 comparison page).
        # Rule-of-mixtures: ~0.79·Cu(205) + 0.09·Al(397) + 0.045·Ni(290)
        #   + 0.04·Fe(247) + 0.012·Mn(267) ≈ 230-240 kJ/kg total. Al
        # contribution raises this above pure Cu; published value is 230 kJ/kg.
        # Previous 205,000 J/kg was effectively pure-copper value, missing the
        # Al contribution entirely. Adopt MakeItFrom catalog value.
        latent_heat=230000,
        # CDA "Patternmakers shrinkage" for C95800 is 3/16 in/ft = 1.5625 % linear;
        # Cast Technologies "Casting shrinkage allowance is 1.6 %". App's 1.5 %
        # linear / 4.5 % volumetric is correct.
        linear_shrinkage=1.5,
        feeding_distance_multiplier=8.0,
        # Extended properties — liquid Cu-Al bronze retains ~ 0.78 of solid k
        # and gains ~ 1.09× in cp at the liquidus.
        conductivity_liquid=28,    # W/(m·K)
        specific_heat_liquid=480,
        dynamic_viscosity=0.004,  # Pa·s — consistent with NAB literature
        thermal_expansion=1.0e-4,
        # Niyama — NAB has excellent feeding and high thermal conductivity,
        # so defect-tolerant threshold is high.
        niyama_threshold=1.2,
        # Hot tear properties — NAB has LOW hot-tear susceptibility (narrow
        # ~ 17 °C freezing range; wide α + κ phase separation suppresses
        # severe BTR; CuAl intermetallic contributes to ductility).
        hot_tear_susceptibility=0.20,
        solid_thermal_expansion=1.8e-5,  # CTE ~ 16-17 µm/m·K (CDA, AZoM)
        btr_critical_strain=0.012,
        btr_critical_strain_rate_sensitivity=0.08,  # Mild rate-dependence
        hot_tear_healing_factor=0.75,
        btr_fraction_solid_start=0.92,
        btr_fraction_solid_end=0.97,
        btr_cooling_rate_exponent=0.05,  # Mild BTR-widening (good ductility)
        # Scheil (binary approx.): pure Cu solvent, Al partitions in bronze
        melting_point_solvent=1084.6,  # °C pure Cu
        partition_coefficient=0.7,  # Al in Cu-rich alpha
        # Feeding/permeability — NAB has excellent feeding; CDA "Fluidity:
        # Medium" but "Shrinkage in Solidification: HIGH" — feeding range
        # extends further than most bronzes before coherency.
        critical_fraction_solid=0.62,
        dendrite_arm_spacing=120.0,
        # Sand burn-in — excellent fluidity raises burn-in risk.
        sand_burn_in_susceptibility=0.55,
        # Gas porosity — CDA: "Gassing: MEDIUM" (H pickup from mould moisture;
        # the Al content forms a protective Al2O3 skin that limits penetration).
        gas_sensitivity=0.55
    ),

    'C964': AlloyProperties(
        name='C964 (CuNi)',
        # ---------------------------------------------------------------
        # Sources (8-source cross-check):
        #   NSWCCD-61-TR-2024/14         (DSC cast): liquidus 1232, solidus 1180
        #   CDA / copper.org (cast):      liquidus 1232, solidus 1166
        #   ZOLLERN GN 30 (cast spec):    1170 - 1240
        #   Cast Technologies (sand-cast): 1170 - 1240
        #   AZoM (sand-cast):              melting 1170 - 1240
        # Adopted: liquidus 1232 °C, solidus 1180 °C (DSC measurement).
        # ---------------------------------------------------------------
        liquidus_temp=1232,  # 2250°F — DSC cast value (was 1175)
        solidus_temp=1180,   # 2156°F — DSC cast value (was 1140)
        pouring_temp=1280,   # 2336°F (~48 °C superheat above liquidus)
        # Density — weighted solid ~ 8940 kg/m³ (CDA, AZoM, MatMake), liquid
        # ~ 1.5-2 % lower (densification on solidification of Cu-Ni FCC).
        density_liquid=8770,
        density_solid=8940,
        # Thermal conductivity (20 °C reference). C96400 sand-cast bar values:
        #   AZoM 29.0, Cast Technologies 29.0, ZOLLERN 29.0, MatMake (C71500) 29.0,
        #   copper.org (Btu conversion) 28.4, MakeItFrom 28.
        # NSWCCD Table B-4 reports 6.15 W/(m·K) but the units in that PDF table
        # are internally inconsistent (rho*cp*alpha yields ~750 W/(m·K) using
        # their numbers); we adopt the literature consensus of 29 W/(m·K).
        thermal_conductivity=29,  # W/(m·K) — was 22
        specific_heat=376,       # J/(kg·K) — AZoM 375, MatMake 380, CDA 377 (was 500)
        # Latent heat of fusion. MakeItFrom and MatMake both list 230-240 kJ/kg
        # (~70% Cu × 205 + 30% Ni × 290 = 230.5 kJ/kg by rule-of-mixtures).
        # NSWCCD Table B-2's 48.2 J/g is the integrated area of a single DSC
        # sub-peak (the PDF itself notes "two peaks"); total latent heat
        # is ~ 240 kJ/kg per industry data.
        latent_heat=240000,
        linear_shrinkage=1.8,  # CDA "Patternmakers shrinkage" 7/32 in/ft = 1.786%
        feeding_distance_multiplier=7.8,  # CDA "Fluidity" is HIGH — large multiplier
        # Extended properties — liquid Cu-Ni retains ~ 70% of solid conductivity
        # and gains ~ 10% in specific heat near liquidus.
        conductivity_liquid=20,   # W/(m·K) (≈ 0.7 of solid) — was 16
        specific_heat_liquid=414, # J/(kg·K) (≈ 1.1 of solid) — was 555
        dynamic_viscosity=0.0048,  # Pa·s — consistent with liquid Cu alloys
        thermal_expansion=1.3e-4,  # 1/K (liquid)
        niyama_threshold=1.1,  # Cu-Ni has good feeding; CDA "Fluidity: HIGH"
        # Hot tear properties — copper-nickel has low-to-moderate susceptibility
        # (CDA "Drossing: Medium", "Section size: Low"). The app default (0.28)
        # sits between C958 (0.20) and CA6NM (0.38), matching CDA's profile.
        hot_tear_susceptibility=0.28,
        solid_thermal_expansion=1.6e-5,  # 1/K — mean CTE 16 µm/m·K (AZoM, CDA)
        btr_critical_strain=0.007,
        btr_critical_strain_rate_sensitivity=0.10,  # Moderate rate-dependence
        hot_tear_healing_factor=0.50,
        btr_fraction_solid_start=0.89,
        btr_fraction_solid_end=0.97,
        btr_cooling_rate_exponent=0.06,  # Mild BTR-widening
        # Scheil (binary approx.): pure Cu solvent; Cu-Ni is isomorphous and
        # does not partition, so k = 1 reduces the Scheil path to the lever rule.
        melting_point_solvent=1084.6,  # °C pure Cu
        partition_coefficient=1.0,  # isomorphous Cu-Ni, no partitioning
        critical_fraction_solid=0.56,
        dendrite_arm_spacing=95.0,  # µm — typical 70/30 Cu-Ni sand-cast
        # CDA "Sand burn-in" susceptibility falls between bronze (high) and
        # stainless (low); C964 "Fluidity" rating is HIGH.
        sand_burn_in_susceptibility=0.48,
        # CDA "Gassing: High" for Cu-Ni (hydrogen pickup from moisture is the
        # primary gas-defect mode for 70/30 Cu-Ni in marine service).
        gas_sensitivity=0.60
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
        btr_critical_strain_rate_sensitivity=0.10,  # Mild rate-dependence
        hot_tear_healing_factor=0.45,  # Moderate healing effectiveness
        btr_fraction_solid_start=0.87,
        btr_fraction_solid_end=0.98,
        btr_cooling_rate_exponent=0.05,  # Mild BTR-widening
        # Scheil (binary approx.): pure Fe solvent, C in austenite k ~0.13-0.2
        melting_point_solvent=1538,  # °C pure Fe
        partition_coefficient=0.17,  # C in austenite ~0.13-0.2
        # Feeding/permeability properties
        critical_fraction_solid=0.54,
        dendrite_arm_spacing=78.0,  # μm
        # Sand burn-in - moderate susceptibility
        sand_burn_in_susceptibility=0.43,
        # Gas porosity - CA6NM ~0.75 (cast steel alloy is sensitive)
        gas_sensitivity=0.75
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
        btr_critical_strain_rate_sensitivity=0.12,  # Strong rate-dependence
        hot_tear_healing_factor=0.40,  # Moderate healing effectiveness
        btr_fraction_solid_start=0.86,  # Wider/earlier BTR than mild steel
        btr_fraction_solid_end=0.98,
        btr_cooling_rate_exponent=0.10,  # Strong BTR-widening (Ni-base alloys)
        # Scheil (binary approx.): pure Ni solvent, Cu partitions moderately
        melting_point_solvent=1455,  # °C pure Ni
        partition_coefficient=0.3,  # Cu in Ni-rich matrix
        # Feeding/permeability properties
        critical_fraction_solid=0.53,  # Moderate coherency point
        dendrite_arm_spacing=85.0,  # μm
        # Sand burn-in - moderate susceptibility
        sand_burn_in_susceptibility=0.39,
        # Gas porosity - Monel 400 has moderate gas sensitivity
        gas_sensitivity=0.55
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
        btr_critical_strain_rate_sensitivity=0.15,  # Strong rate-dependence
        hot_tear_healing_factor=0.30,  # Poor healing due to restricted feeding
        btr_fraction_solid_start=0.84,  # Earlier onset with wider BTR
        btr_fraction_solid_end=0.99,
        btr_cooling_rate_exponent=0.10,  # Strong BTR-widening (Ni-base)
        # Scheil (binary approx.): pure Ni solvent, Nb/Mo partition in 625
        melting_point_solvent=1455,  # °C pure Ni
        partition_coefficient=0.5,  # refractory solutes in Ni matrix
        # Feeding/permeability properties
        critical_fraction_solid=0.50,  # Lower coherency due to restricted feeding
        dendrite_arm_spacing=70.0,  # μm
        # Sand burn-in - controlled by viscosity and oxidation behavior
        sand_burn_in_susceptibility=0.37,
        # Gas porosity - Inconel 625 has lower sensitivity
        gas_sensitivity=0.45
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
