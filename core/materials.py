from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

@dataclass
class AlloyProperties:
    name: str
    liquidus_temp: float
    solidus_temp: float
    pouring_temp: float
    density_liquid: float
    density_solid: float
    thermal_conductivity: float
    specific_heat: float
    latent_heat: float
    linear_shrinkage: float
    feeding_distance_multiplier: float

    conductivity_liquid: Optional[float] = None
    specific_heat_liquid: Optional[float] = None

    dynamic_viscosity: float = 0.006
    thermal_expansion: float = 1.2e-4
    viscosity_temp_sensitivity: float = 1.2

    surface_tension: float = 1.8

    niyama_threshold: float = 1.0

    hot_tear_susceptibility: float = 0.30

    solid_thermal_expansion: float = 1.2e-5

    btr_critical_strain: float = 0.008

    btr_critical_strain_rate_sensitivity: float = 0.0

    btr_critical_strain_ref_rate: float = 1.0e-3

    hot_tear_healing_factor: float = 0.55

    btr_fraction_solid_start: float = 0.90
    btr_fraction_solid_end: float = 0.98

    btr_cooling_rate_exponent: float = 0.0

    btr_btr_cooling_rate_ref: float = 1.0

    melting_point_solvent: Optional[float] = None
    partition_coefficient: float = 0.17
    eutectic_temperature: Optional[float] = None
    eutectic_fraction: float = 0.0

    critical_fraction_solid: float = 0.50

    dendrite_arm_spacing: float = 100.0
    sdas_reference_cooling_rate: float = 1.0
    sdas_cooling_rate_exponent: float = -0.33

    sand_burn_in_susceptibility: float = 0.40

    gas_sensitivity: float = 0.5

    solidification_shrinkage: float = 3.0

    def __post_init__(self):
        if self.conductivity_liquid is None:
            self.conductivity_liquid = self.thermal_conductivity * 0.7
        if self.specific_heat_liquid is None:
            self.specific_heat_liquid = self.specific_heat * 1.1

    @property
    def thermal_diffusivity(self) -> float:
        return self.thermal_conductivity / (self.density_liquid * self.specific_heat)

    @property
    def thermal_diffusivity_liquid(self) -> float:
        return self.conductivity_liquid / (self.density_liquid * self.specific_heat_liquid)

    @property
    def kinematic_viscosity(self) -> float:
        return self.dynamic_viscosity / self.density_liquid

    @property
    def prandtl_number(self) -> float:
        return self.kinematic_viscosity / self.thermal_diffusivity_liquid

    @property
    def freezing_range(self) -> float:
        return self.liquidus_temp - self.solidus_temp

    @property
    def volumetric_shrinkage(self) -> float:
        return (1 - (self.density_liquid / self.density_solid)) * 100

    @property
    def feeding_shrinkage(self) -> float:
        superheat = max(float(self.pouring_temp) - float(self.liquidus_temp), 0.0)
        return float(self.solidification_shrinkage) + 100.0 * float(self.thermal_expansion) * superheat

    @property
    def density_change_ratio(self) -> float:
        return self.density_solid / self.density_liquid

    def get_feeding_distance(self, thickness: float) -> float:
        return thickness * self.feeding_distance_multiplier

    def get_dynamic_viscosity(
        self,
        temperature: Optional[float] = None,
        fraction_solid: Optional[float] = None,
    ) -> float:
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
        if sdas_um is None:
            sdas_um = self.get_sdas_from_cooling_rate(cooling_rate)
        lambda_m = float(sdas_um) * 1e-6

        fs = max(0.001, min(fraction_solid, 0.999))

        if fs >= self.critical_fraction_solid:
            overshoot = (fs - self.critical_fraction_solid) / (1.0 - self.critical_fraction_solid)
            decay_factor = np.exp(-10 * overshoot)
            fs_effective = self.critical_fraction_solid
        else:
            decay_factor = 1.0
            fs_effective = fs

        liquid_fraction = 1.0 - fs_effective

        if fs_effective < 0.001:
            return 1e-8

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
        K = self.get_permeability(
            fraction_solid,
            sdas_um=sdas_um,
            cooling_rate=cooling_rate,
        )

        if K < 1e-20:
            return 1e10

        delta_P = (viscosity * path_length * flow_velocity) / K

        return delta_P

    def vectorized_dynamic_viscosity(
        self,
        temperatures: np.ndarray,
        fraction_solids: np.ndarray,
    ) -> np.ndarray:
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
        K = self.vectorized_permeability(fraction_solids, sdas_um)
        delta_P = (viscosities * path_lengths_m * float(flow_velocity)) / np.maximum(K, 1e-20)
        delta_P = np.where(K < 1e-20, 1e10, delta_P)
        return delta_P * feeder_neck_factors

ALLOY_DATABASE: Dict[str, AlloyProperties] = {
    'WCB': AlloyProperties(
        name='WCB (Carbon Steel Casting)',
        liquidus_temp=1460,
        solidus_temp=1399,
        pouring_temp=1621,
        density_liquid=7200,
        density_solid=7850,
        thermal_conductivity=45,
        specific_heat=600,
        latent_heat=272000,
        linear_shrinkage=2.0,
        feeding_distance_multiplier=6.0,
        conductivity_liquid=32,
        specific_heat_liquid=660,
        dynamic_viscosity=0.006,
        thermal_expansion=1.2e-4,
        surface_tension=1.85,
        niyama_threshold=1.0,
        hot_tear_susceptibility=0.30,
        solid_thermal_expansion=1.2e-5,
        btr_critical_strain=0.008,
        btr_critical_strain_rate_sensitivity=0.10,
        hot_tear_healing_factor=0.55,
        btr_fraction_solid_start=0.90,
        btr_fraction_solid_end=0.98,
        btr_cooling_rate_exponent=0.05,
        melting_point_solvent=1538,
        partition_coefficient=0.17,
        critical_fraction_solid=0.52,
        dendrite_arm_spacing=80.0,
        sand_burn_in_susceptibility=0.45,
        gas_sensitivity=0.50
    ),

    'CF8': AlloyProperties(
        name='CF8 (Stainless Steel 304)',
        liquidus_temp=1450,
        solidus_temp=1400,
        pouring_temp=1621,
        density_liquid=7400,
        density_solid=8000,
        thermal_conductivity=16.2,
        specific_heat=500,
        latent_heat=260000,
        linear_shrinkage=2.5,
        feeding_distance_multiplier=5.5,
        conductivity_liquid=12,
        specific_heat_liquid=550,
        dynamic_viscosity=0.007,
        thermal_expansion=1.5e-4,
        surface_tension=1.80,
        niyama_threshold=0.7,
        hot_tear_susceptibility=0.65,
        solid_thermal_expansion=1.7e-5,
        btr_critical_strain=0.004,
        btr_critical_strain_rate_sensitivity=0.15,
        hot_tear_healing_factor=0.35,
        btr_fraction_solid_start=0.85,
        btr_fraction_solid_end=0.99,
        btr_cooling_rate_exponent=0.08,
        melting_point_solvent=1538,
        partition_coefficient=0.7,
        critical_fraction_solid=0.48,
        dendrite_arm_spacing=60.0,
        sand_burn_in_susceptibility=0.35,
        gas_sensitivity=0.60
    ),

    'C958': AlloyProperties(
        name='C958 (Aluminum Bronze)',
        liquidus_temp=1060,
        solidus_temp=1043,
        pouring_temp=1255,
        density_liquid=7100,
        density_solid=7640,
        thermal_conductivity=36,
        specific_heat=440,
        latent_heat=230000,
        linear_shrinkage=1.5,
        solidification_shrinkage=5.0,
        feeding_distance_multiplier=8.0,
        conductivity_liquid=28,
        specific_heat_liquid=480,
        dynamic_viscosity=0.004,
        thermal_expansion=1.0e-4,
        surface_tension=1.2,
        niyama_threshold=1.2,
        hot_tear_susceptibility=0.20,
        solid_thermal_expansion=1.8e-5,
        btr_critical_strain=0.012,
        btr_critical_strain_rate_sensitivity=0.08,
        hot_tear_healing_factor=0.75,
        btr_fraction_solid_start=0.92,
        btr_fraction_solid_end=0.97,
        btr_cooling_rate_exponent=0.05,
        melting_point_solvent=1084.6,
        partition_coefficient=0.7,
        critical_fraction_solid=0.62,
        dendrite_arm_spacing=120.0,
        sand_burn_in_susceptibility=0.55,
        gas_sensitivity=0.55
    ),

    'C964': AlloyProperties(
        name='C964 (CuNi)',
        liquidus_temp=1232,
        solidus_temp=1180,
        pouring_temp=1280,
        density_liquid=8770,
        density_solid=8940,
        thermal_conductivity=29,
        specific_heat=376,
        latent_heat=240000,
        solidification_shrinkage=5.0,
        linear_shrinkage=1.8,
        feeding_distance_multiplier=7.8,
        conductivity_liquid=20,
        specific_heat_liquid=414,
        dynamic_viscosity=0.0048,
        thermal_expansion=1.3e-4,
        surface_tension=1.2,
        niyama_threshold=1.1,
        hot_tear_susceptibility=0.28,
        solid_thermal_expansion=1.6e-5,
        btr_critical_strain=0.007,
        btr_critical_strain_rate_sensitivity=0.10,
        hot_tear_healing_factor=0.50,
        btr_fraction_solid_start=0.89,
        btr_fraction_solid_end=0.97,
        btr_cooling_rate_exponent=0.06,
        melting_point_solvent=1084.6,
        partition_coefficient=1.0,
        critical_fraction_solid=0.56,
        dendrite_arm_spacing=95.0,
        sand_burn_in_susceptibility=0.48,
        gas_sensitivity=0.60
    ),

    'CA6NM': AlloyProperties(
        name='CA6NM (Cast Steel)',
        liquidus_temp=1440,
        solidus_temp=1385,
        pouring_temp=1560,
        density_liquid=7600,
        density_solid=7900,
        thermal_conductivity=28,
        specific_heat=485,
        latent_heat=252000,
        linear_shrinkage=2.1,
        feeding_distance_multiplier=5.9,
        conductivity_liquid=20,
        specific_heat_liquid=530,
        dynamic_viscosity=0.0064,
        thermal_expansion=1.35e-4,
        surface_tension=1.85,
        niyama_threshold=0.95,
        hot_tear_susceptibility=0.38,
        solid_thermal_expansion=1.3e-5,
        btr_critical_strain=0.006,
        btr_critical_strain_rate_sensitivity=0.10,
        hot_tear_healing_factor=0.45,
        btr_fraction_solid_start=0.87,
        btr_fraction_solid_end=0.98,
        btr_cooling_rate_exponent=0.05,
        melting_point_solvent=1538,
        partition_coefficient=0.17,
        critical_fraction_solid=0.54,
        dendrite_arm_spacing=78.0,
        sand_burn_in_susceptibility=0.43,
        gas_sensitivity=0.75
    ),

    'MONEL400': AlloyProperties(
        name='MONEL400 (Monel 400 Nickel-Copper)',
        liquidus_temp=1370,
        solidus_temp=1320,
        pouring_temp=1490,
        density_liquid=8100,
        density_solid=8350,
        thermal_conductivity=19,
        specific_heat=480,
        latent_heat=250000,
        linear_shrinkage=1.9,
        feeding_distance_multiplier=6.4,
        conductivity_liquid=14,
        specific_heat_liquid=530,
        dynamic_viscosity=0.0063,
        thermal_expansion=1.45e-4,
        surface_tension=1.75,
        niyama_threshold=0.85,
        hot_tear_susceptibility=0.42,
        solid_thermal_expansion=1.4e-5,
        btr_critical_strain=0.005,
        btr_critical_strain_rate_sensitivity=0.12,
        hot_tear_healing_factor=0.40,
        btr_fraction_solid_start=0.86,
        btr_fraction_solid_end=0.98,
        btr_cooling_rate_exponent=0.10,
        melting_point_solvent=1455,
        partition_coefficient=0.3,
        critical_fraction_solid=0.53,
        dendrite_arm_spacing=85.0,
        sand_burn_in_susceptibility=0.39,
        gas_sensitivity=0.55
    ),

    'IN625': AlloyProperties(
        name='IN625 (Inconel 625)',
        liquidus_temp=1330,
        solidus_temp=1260,
        pouring_temp=1420,
        density_liquid=8250,
        density_solid=8480,
        thermal_conductivity=13,
        specific_heat=435,
        latent_heat=255000,
        linear_shrinkage=2.0,
        feeding_distance_multiplier=5.8,
        conductivity_liquid=10,
        specific_heat_liquid=485,
        dynamic_viscosity=0.0068,
        thermal_expansion=1.5e-4,
        surface_tension=1.75,
        niyama_threshold=0.9,
        hot_tear_susceptibility=0.58,
        solid_thermal_expansion=1.5e-5,
        btr_critical_strain=0.004,
        btr_critical_strain_rate_sensitivity=0.15,
        hot_tear_healing_factor=0.30,
        btr_fraction_solid_start=0.84,
        btr_fraction_solid_end=0.99,
        btr_cooling_rate_exponent=0.10,
        melting_point_solvent=1455,
        partition_coefficient=0.5,
        critical_fraction_solid=0.50,
        dendrite_arm_spacing=70.0,
        sand_burn_in_susceptibility=0.37,
        gas_sensitivity=0.45
    )
}

def get_alloy(name: str) -> AlloyProperties:
    if name not in ALLOY_DATABASE:
        available = ', '.join(ALLOY_DATABASE.keys())
        raise KeyError(f"Alloy '{name}' not found. Available: {available}")

    return ALLOY_DATABASE[name]

def get_all_alloy_names() -> list:
    return list(ALLOY_DATABASE.keys())

def print_alloy_info(name: str):
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
    for alloy_name in get_all_alloy_names():
        print_alloy_info(alloy_name)
