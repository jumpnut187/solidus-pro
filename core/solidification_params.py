from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

DEFAULT_MAX_SIMULATION_TIME_S = 57600.0

SHRINK_SCALAR_MIN = 0.25
SHRINK_SCALAR_MAX = 4.0
SHRINK_SCALAR_DEFAULT = 1.0

def _clamp_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if numeric != numeric:
        numeric = default
    return max(minimum, min(maximum, numeric))

def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = default
    return max(minimum, min(maximum, numeric))

@dataclass
class FeedingParams:
    base_feeding_distance: float = 3.0
    min_influence_factor: float = 0.6
    riser_max_reduction: float = 0.10
    riser_min_reduction: float = 1.0
    post_contact_reduction: float = 0.35
    disconnected_neck_floor: float = 0.70
    shrink_scalar: float = SHRINK_SCALAR_DEFAULT
    feeding_zone_vertical_bias: float = 0.0

    def __post_init__(self) -> None:
        self.shrink_scalar = _clamp_float(
            self.shrink_scalar,
            SHRINK_SCALAR_MIN,
            SHRINK_SCALAR_MAX,
            SHRINK_SCALAR_DEFAULT,
        )
        try:
            bias = float(self.feeding_zone_vertical_bias)
        except (TypeError, ValueError):
            bias = 0.0
        if bias != bias:
            bias = 0.0
        self.feeding_zone_vertical_bias = max(0.0, min(2.0, bias))

@dataclass
class ChillParams:
    max_chill_boost: float = 3.0
    shrinkage_radius_multiplier: float = 1.5
    shrinkage_max_reduction: float = 0.15
    shrinkage_min_reduction: float = 1.0
    thickness_fallback_voxels: float = 5.0

@dataclass
class ExothermicParams:
    ignition_delay: float = 10.0
    burn_duration: float = 120.0
    insulation_only_factor: float = 0.7
    full_exothermic_factor: float = 0.5
    decay_time_constant: float = 60.0

@dataclass
class AirGapParams:
    time_constant: float = 60.0
    surface_influence_depth: float = 0.5
    max_slowdown: float = 0.75
    htc_floor_fraction: float = 0.35

    def __post_init__(self) -> None:
        self.htc_floor_fraction = _clamp_float(
            self.htc_floor_fraction,
            0.05,
            1.0,
            0.35,
        )

@dataclass
class ThermalBoundaryParams:
    mold_temperature: float = 21.0
    chill_temperature: float = 21.0
    htc_cast_interface: float = 500.0
    htc_riser_post: float = 350.0
    htc_riser: float = 220.0
    htc_gating: float = 400.0
    chill_ambient_htc: float = 500.0
    chill_htc_floor_fraction: float = 0.7
    mold_thermal_conductivity: float = 0.6
    mold_thermal_diffusivity: float = 3.3e-7
    core_thermal_conductivity: float = 0.6
    core_density: float = 1600.0
    core_specific_heat: float = 1140.0
    htc_core_contact: float = 1000.0
    htc_core_print: float = 300.0
    core_print_htc_floor_fraction: float = 0.1

    def __post_init__(self) -> None:
        self.chill_ambient_htc = _clamp_float(
            self.chill_ambient_htc,
            1.0,
            10000.0,
            500.0,
        )
        self.chill_htc_floor_fraction = _clamp_float(
            self.chill_htc_floor_fraction,
            0.05,
            1.0,
            0.7,
        )
        self.mold_thermal_conductivity = _clamp_float(
            self.mold_thermal_conductivity,
            0.05,
            500.0,
            0.6,
        )
        self.mold_thermal_diffusivity = _clamp_float(
            self.mold_thermal_diffusivity,
            1e-9,
            1e-3,
            3.3e-7,
        )
        self.core_thermal_conductivity = _clamp_float(self.core_thermal_conductivity, 0.05, 500.0, 0.6)
        self.core_density = _clamp_float(self.core_density, 100.0, 20000.0, 1600.0)
        self.core_specific_heat = _clamp_float(self.core_specific_heat, 50.0, 5000.0, 1140.0)
        self.htc_core_contact = _clamp_float(self.htc_core_contact, 1.0, 1e5, 1000.0)
        self.htc_core_print = _clamp_float(self.htc_core_print, 1.0, 1e4, 300.0)
        self.core_print_htc_floor_fraction = _clamp_float(self.core_print_htc_floor_fraction, 0.0, 1.0, 0.1)

CORE_RULE_DRAW = 0
CORE_RULE_ENCLOSED = 1

SAND_MATERIAL_CUSTOM = 0

RISER_SLEEVE_EXPONENT = 0.68

MOLD_GRADING_MAX_LEVELS = 5

@dataclass
class MoldParams:
    auto_mold: int = 1
    auto_cores: int = 1
    core_rule_code: int = CORE_RULE_DRAW
    wall_thickness_in: float = 2.0
    closed_riser_cover_in: float = 2.0
    mold_material_code: int = 2
    core_material_code: int = SAND_MATERIAL_CUSTOM
    htc_cast_mold: float = 1000.0
    interface_htc_curve: int = 1
    htc_chill_mold: float = 1500.0
    riser_sleeve_htc: float = 61.7
    ambient_temperature: float = 21.0
    air_htc: float = 10.0
    sand_emissivity: float = 0.9
    metal_emissivity: float = 0.8
    max_mold_cells: int = 40_000_000
    mold_grading_levels: int = 3
    mold_fine_band_cells: int = 4

    def __post_init__(self) -> None:
        self.auto_mold = _clamp_int(self.auto_mold, 0, 1, 1)
        self.auto_cores = _clamp_int(self.auto_cores, 0, 1, 1)
        self.core_rule_code = _clamp_int(self.core_rule_code, 0, 1, CORE_RULE_DRAW)
        self.wall_thickness_in = _clamp_float(self.wall_thickness_in, 0.0, 48.0, 2.0)
        self.closed_riser_cover_in = _clamp_float(self.closed_riser_cover_in, 0.0, 24.0, 2.0)
        self.mold_material_code = _clamp_int(self.mold_material_code, 0, 255, 2)
        self.core_material_code = _clamp_int(self.core_material_code, 0, 255, SAND_MATERIAL_CUSTOM)
        self.htc_cast_mold = _clamp_float(self.htc_cast_mold, 1.0, 1e5, 1000.0)
        self.interface_htc_curve = _clamp_int(self.interface_htc_curve, 0, 1, 1)
        self.htc_chill_mold = _clamp_float(self.htc_chill_mold, 1.0, 1e5, 1500.0)
        self.riser_sleeve_htc = _clamp_float(self.riser_sleeve_htc, 1.0, 1e5, 61.7)
        self.ambient_temperature = _clamp_float(self.ambient_temperature, -40.0, 400.0, 21.0)
        self.air_htc = _clamp_float(self.air_htc, 0.0, 1000.0, 10.0)
        self.sand_emissivity = _clamp_float(self.sand_emissivity, 0.0, 1.0, 0.9)
        self.metal_emissivity = _clamp_float(self.metal_emissivity, 0.0, 1.0, 0.8)
        self.max_mold_cells = _clamp_int(self.max_mold_cells, 1_000, 500_000_000, 40_000_000)
        self.mold_grading_levels = _clamp_int(self.mold_grading_levels, 0, MOLD_GRADING_MAX_LEVELS, 3)
        self.mold_fine_band_cells = _clamp_int(self.mold_fine_band_cells, 2, 64, 4)

@dataclass
class TimingParams:
    chvorinov_mold_constant: float = 280.0
    safety_factor_with_risers: float = 2.5
    safety_factor_no_risers: float = 1.5
    min_simulation_time: float = 600.0
    max_simulation_time: float = DEFAULT_MAX_SIMULATION_TIME_S
    sand_saturation_time: float = 30.0
    max_hot_sand_slowdown: float = 0.6

@dataclass
class FilterParams:
    porosity: float = 0.85
    pore_diameter_mm: float = 2.54
    ergun_prefactor: float = 2.2
    forchheimer_c_f: float = 0.55
    contact_angle_deg: float = 120.0
    window_fraction: float = 0.45
    rated_velocity_warn_factor: float = 1.5
    head_fraction_warn: float = 0.25
    priming_enabled: int = 1
    drag_enabled: int = 1

    def __post_init__(self) -> None:
        self.porosity = _clamp_float(self.porosity, 0.50, 0.95, 0.85)
        self.pore_diameter_mm = _clamp_float(self.pore_diameter_mm, 0.5, 8.0, 2.54)
        self.ergun_prefactor = _clamp_float(self.ergun_prefactor, 0.5, 10.0, 2.2)
        self.forchheimer_c_f = _clamp_float(self.forchheimer_c_f, 0.1, 2.0, 0.55)
        self.contact_angle_deg = _clamp_float(self.contact_angle_deg, 90.0, 160.0, 120.0)
        self.window_fraction = _clamp_float(self.window_fraction, 0.2, 0.8, 0.45)

@dataclass
class FillingParams:
    pour_rate_lb_s: float = 0.0
    pour_time_s: float = 0.0
    pour_temperature_c: float = 0.0
    initial_head_in: float = 4.0
    fill_grid_coarsening: int = 2
    choke_auto_resolution: int = 1
    max_fill_time_s: float = 60.0
    smagorinsky_cs: float = 0.16
    lattice_velocity_max: float = 0.05
    wall_htc_multiplier: float = 1.0
    inlet_mode_code: int = 0
    pour_mode_code: int = 0
    vent_risers: int = 1
    cold_shut_arrival_gap_s: float = 0.5
    cold_shut_temp_margin_c: float = 25.0
    entrapment_check_interval: int = 50
    filter: FilterParams = field(default_factory=FilterParams)

    def __post_init__(self) -> None:
        self.fill_grid_coarsening = _clamp_int(self.fill_grid_coarsening, 1, 8, 2)
        self.choke_auto_resolution = _clamp_int(self.choke_auto_resolution, 0, 1, 1)
        self.lattice_velocity_max = _clamp_float(self.lattice_velocity_max, 0.005, 0.2, 0.05)
        self.max_fill_time_s = _clamp_float(self.max_fill_time_s, 1.0, 600.0, 60.0)
        self.cold_shut_arrival_gap_s = _clamp_float(self.cold_shut_arrival_gap_s, 0.01, 10.0, 0.5)
        self.cold_shut_temp_margin_c = _clamp_float(self.cold_shut_temp_margin_c, 0.0, 200.0, 25.0)
        self.entrapment_check_interval = _clamp_int(self.entrapment_check_interval, 1, 1000, 50)

LEGACY_DEFAULT_POUR_RATE_LB_S = 5.0

def migrate_legacy_pour_rate(params: Dict[str, Any]) -> Dict[str, Any]:
    migrated = dict(params)
    filling = migrated.get('filling')
    if not isinstance(filling, dict):
        return migrated
    try:
        rate = float(filling['pour_rate_lb_s'])
    except (KeyError, TypeError, ValueError):
        return migrated
    if rate != LEGACY_DEFAULT_POUR_RATE_LB_S:
        return migrated
    migrated['filling'] = {**filling, 'pour_rate_lb_s': 0.0}
    return migrated

@dataclass
class HotTearParams:
    mold_restraint_index: float = 0.5
    metal_mold_friction_index: float = 0.5
    core_restraint_index: float = 0.0
    low_melting_impurity_index: float = 0.0
    hot_tear_zone_threshold: float = 0.0
    healing_relief_strength: float = 0.65
    surface_restraint_depth_inches: float = 0.5
    solid_path_search_depth: int = 8

@dataclass
class SolidificationParameters:
    feeding: FeedingParams = field(default_factory=FeedingParams)
    chill: ChillParams = field(default_factory=ChillParams)
    exothermic: ExothermicParams = field(default_factory=ExothermicParams)
    air_gap: AirGapParams = field(default_factory=AirGapParams)
    thermal_boundary: ThermalBoundaryParams = field(default_factory=ThermalBoundaryParams)
    timing: TimingParams = field(default_factory=TimingParams)
    hot_tear: HotTearParams = field(default_factory=HotTearParams)
    filling: FillingParams = field(default_factory=FillingParams)
    mold: MoldParams = field(default_factory=MoldParams)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> SolidificationParameters:
        if data is None:
            return cls()

        def merge_into_dataclass(obj: Any, updates: Dict[str, Any]) -> None:
            for raw_key, value in updates.items():
                key = raw_key.lower() if isinstance(raw_key, str) else raw_key
                if not hasattr(obj, key):
                    continue
                current = getattr(obj, key)
                if isinstance(value, dict):
                    if hasattr(current, '__dataclass_fields__'):
                        merge_into_dataclass(current, value)
                    else:
                        setattr(obj, key, value)
                else:
                    setattr(obj, key, value)

        instance = cls()
        merge_into_dataclass(instance, data)
        instance._clamp_values()
        return instance

    def _clamp_values(self) -> None:
        self.feeding.shrink_scalar = _clamp_float(
            self.feeding.shrink_scalar,
            SHRINK_SCALAR_MIN,
            SHRINK_SCALAR_MAX,
            SHRINK_SCALAR_DEFAULT,
        )
        try:
            bias = float(self.feeding.feeding_zone_vertical_bias)
        except (TypeError, ValueError):
            bias = 0.0
        if bias != bias:
            bias = 0.0
        self.feeding.feeding_zone_vertical_bias = max(0.0, min(2.0, bias))
        self.air_gap.htc_floor_fraction = _clamp_float(
            self.air_gap.htc_floor_fraction,
            0.05,
            1.0,
            0.35,
        )
        self.thermal_boundary.chill_ambient_htc = _clamp_float(
            self.thermal_boundary.chill_ambient_htc,
            1.0,
            10000.0,
            500.0,
        )
        self.thermal_boundary.chill_htc_floor_fraction = _clamp_float(
            self.thermal_boundary.chill_htc_floor_fraction,
            0.05,
            1.0,
            0.7,
        )
        self.thermal_boundary.mold_thermal_conductivity = _clamp_float(
            self.thermal_boundary.mold_thermal_conductivity,
            0.05,
            500.0,
            0.6,
        )
        self.thermal_boundary.mold_thermal_diffusivity = _clamp_float(
            self.thermal_boundary.mold_thermal_diffusivity,
            1e-9,
            1e-3,
            3.3e-7,
        )
        tb = self.thermal_boundary
        tb.core_thermal_conductivity = _clamp_float(tb.core_thermal_conductivity, 0.05, 500.0, 0.6)
        tb.core_density = _clamp_float(tb.core_density, 100.0, 20000.0, 1600.0)
        tb.core_specific_heat = _clamp_float(tb.core_specific_heat, 50.0, 5000.0, 1140.0)
        tb.htc_core_contact = _clamp_float(tb.htc_core_contact, 1.0, 1e5, 1000.0)
        tb.htc_core_print = _clamp_float(tb.htc_core_print, 1.0, 1e4, 300.0)
        tb.core_print_htc_floor_fraction = _clamp_float(tb.core_print_htc_floor_fraction, 0.0, 1.0, 0.1)
        self.filling.fill_grid_coarsening = _clamp_int(
            self.filling.fill_grid_coarsening, 1, 8, 2,
        )
        self.filling.choke_auto_resolution = _clamp_int(
            self.filling.choke_auto_resolution, 0, 1, 1,
        )
        self.filling.lattice_velocity_max = _clamp_float(
            self.filling.lattice_velocity_max, 0.005, 0.2, 0.05,
        )
        self.filling.max_fill_time_s = _clamp_float(
            self.filling.max_fill_time_s, 1.0, 600.0, 60.0,
        )
        self.filling.cold_shut_arrival_gap_s = _clamp_float(
            self.filling.cold_shut_arrival_gap_s, 0.01, 10.0, 0.5,
        )
        self.filling.cold_shut_temp_margin_c = _clamp_float(
            self.filling.cold_shut_temp_margin_c, 0.0, 200.0, 25.0,
        )
        self.filling.entrapment_check_interval = _clamp_int(
            self.filling.entrapment_check_interval, 1, 1000, 50,
        )
        self.filling.filter.porosity = _clamp_float(
            self.filling.filter.porosity, 0.50, 0.95, 0.85,
        )
        self.filling.filter.pore_diameter_mm = _clamp_float(
            self.filling.filter.pore_diameter_mm, 0.5, 8.0, 2.54,
        )
        self.filling.filter.ergun_prefactor = _clamp_float(
            self.filling.filter.ergun_prefactor, 0.5, 10.0, 2.2,
        )
        self.filling.filter.forchheimer_c_f = _clamp_float(
            self.filling.filter.forchheimer_c_f, 0.1, 2.0, 0.55,
        )
        self.filling.filter.contact_angle_deg = _clamp_float(
            self.filling.filter.contact_angle_deg, 90.0, 160.0, 120.0,
        )
        self.filling.filter.window_fraction = _clamp_float(
            self.filling.filter.window_fraction, 0.2, 0.8, 0.45,
        )
        self.mold.__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        self._clamp_values()
        return {
            'feeding': {
                'base_feeding_distance': self.feeding.base_feeding_distance,
                'min_influence_factor': self.feeding.min_influence_factor,
                'riser_max_reduction': self.feeding.riser_max_reduction,
                'riser_min_reduction': self.feeding.riser_min_reduction,
                'post_contact_reduction': self.feeding.post_contact_reduction,
                'disconnected_neck_floor': self.feeding.disconnected_neck_floor,
                'shrink_scalar': self.feeding.shrink_scalar,
                'feeding_zone_vertical_bias': self.feeding.feeding_zone_vertical_bias,
            },
            'chill': {
                'max_chill_boost': self.chill.max_chill_boost,
                'shrinkage_radius_multiplier': self.chill.shrinkage_radius_multiplier,
                'shrinkage_max_reduction': self.chill.shrinkage_max_reduction,
                'shrinkage_min_reduction': self.chill.shrinkage_min_reduction,
                'thickness_fallback_voxels': self.chill.thickness_fallback_voxels,
            },
            'exothermic': {
                'ignition_delay': self.exothermic.ignition_delay,
                'burn_duration': self.exothermic.burn_duration,
                'insulation_only_factor': self.exothermic.insulation_only_factor,
                'full_exothermic_factor': self.exothermic.full_exothermic_factor,
                'decay_time_constant': self.exothermic.decay_time_constant,
            },
            'air_gap': {
                'time_constant': self.air_gap.time_constant,
                'surface_influence_depth': self.air_gap.surface_influence_depth,
                'max_slowdown': self.air_gap.max_slowdown,
                'htc_floor_fraction': self.air_gap.htc_floor_fraction,
            },
            'thermal_boundary': {
                'mold_temperature': self.thermal_boundary.mold_temperature,
                'chill_temperature': self.thermal_boundary.chill_temperature,
                'htc_cast_interface': self.thermal_boundary.htc_cast_interface,
                'htc_riser_post': self.thermal_boundary.htc_riser_post,
                'htc_riser': self.thermal_boundary.htc_riser,
                'htc_gating': self.thermal_boundary.htc_gating,
                'chill_ambient_htc': self.thermal_boundary.chill_ambient_htc,
                'chill_htc_floor_fraction': self.thermal_boundary.chill_htc_floor_fraction,
                'mold_thermal_conductivity': self.thermal_boundary.mold_thermal_conductivity,
                'mold_thermal_diffusivity': self.thermal_boundary.mold_thermal_diffusivity,
                'core_thermal_conductivity': self.thermal_boundary.core_thermal_conductivity,
                'core_density': self.thermal_boundary.core_density,
                'core_specific_heat': self.thermal_boundary.core_specific_heat,
                'htc_core_contact': self.thermal_boundary.htc_core_contact,
                'htc_core_print': self.thermal_boundary.htc_core_print,
                'core_print_htc_floor_fraction': self.thermal_boundary.core_print_htc_floor_fraction,
            },
            'timing': {
                'chvorinov_mold_constant': self.timing.chvorinov_mold_constant,
                'safety_factor_with_risers': self.timing.safety_factor_with_risers,
                'safety_factor_no_risers': self.timing.safety_factor_no_risers,
                'min_simulation_time': self.timing.min_simulation_time,
                'max_simulation_time': self.timing.max_simulation_time,
                'sand_saturation_time': self.timing.sand_saturation_time,
                'max_hot_sand_slowdown': self.timing.max_hot_sand_slowdown,
            },
            'hot_tear': {
                'mold_restraint_index': self.hot_tear.mold_restraint_index,
                'metal_mold_friction_index': self.hot_tear.metal_mold_friction_index,
                'core_restraint_index': self.hot_tear.core_restraint_index,
                'low_melting_impurity_index': self.hot_tear.low_melting_impurity_index,
                'hot_tear_zone_threshold': self.hot_tear.hot_tear_zone_threshold,
                'healing_relief_strength': self.hot_tear.healing_relief_strength,
                'surface_restraint_depth_inches': self.hot_tear.surface_restraint_depth_inches,
                'solid_path_search_depth': self.hot_tear.solid_path_search_depth,
            },
            'filling': {
                'pour_rate_lb_s': self.filling.pour_rate_lb_s,
                'pour_time_s': self.filling.pour_time_s,
                'pour_temperature_c': self.filling.pour_temperature_c,
                'initial_head_in': self.filling.initial_head_in,
                'fill_grid_coarsening': self.filling.fill_grid_coarsening,
                'choke_auto_resolution': self.filling.choke_auto_resolution,
                'max_fill_time_s': self.filling.max_fill_time_s,
                'smagorinsky_cs': self.filling.smagorinsky_cs,
                'lattice_velocity_max': self.filling.lattice_velocity_max,
                'wall_htc_multiplier': self.filling.wall_htc_multiplier,
                'inlet_mode_code': self.filling.inlet_mode_code,
                'pour_mode_code': self.filling.pour_mode_code,
                'vent_risers': self.filling.vent_risers,
                'cold_shut_arrival_gap_s': self.filling.cold_shut_arrival_gap_s,
                'cold_shut_temp_margin_c': self.filling.cold_shut_temp_margin_c,
                'entrapment_check_interval': self.filling.entrapment_check_interval,
                'filter': {
                    'porosity': self.filling.filter.porosity,
                    'pore_diameter_mm': self.filling.filter.pore_diameter_mm,
                    'ergun_prefactor': self.filling.filter.ergun_prefactor,
                    'forchheimer_c_f': self.filling.filter.forchheimer_c_f,
                    'contact_angle_deg': self.filling.filter.contact_angle_deg,
                    'window_fraction': self.filling.filter.window_fraction,
                    'rated_velocity_warn_factor': self.filling.filter.rated_velocity_warn_factor,
                    'head_fraction_warn': self.filling.filter.head_fraction_warn,
                    'priming_enabled': self.filling.filter.priming_enabled,
                    'drag_enabled': self.filling.filter.drag_enabled,
                },
            },
            'mold': {
                'auto_mold': self.mold.auto_mold,
                'auto_cores': self.mold.auto_cores,
                'core_rule_code': self.mold.core_rule_code,
                'wall_thickness_in': self.mold.wall_thickness_in,
                'closed_riser_cover_in': self.mold.closed_riser_cover_in,
                'mold_material_code': self.mold.mold_material_code,
                'core_material_code': self.mold.core_material_code,
                'htc_cast_mold': self.mold.htc_cast_mold,
                'interface_htc_curve': self.mold.interface_htc_curve,
                'htc_chill_mold': self.mold.htc_chill_mold,
                'riser_sleeve_htc': self.mold.riser_sleeve_htc,
                'ambient_temperature': self.mold.ambient_temperature,
                'air_htc': self.mold.air_htc,
                'sand_emissivity': self.mold.sand_emissivity,
                'metal_emissivity': self.mold.metal_emissivity,
                'max_mold_cells': self.mold.max_mold_cells,
                'mold_grading_levels': self.mold.mold_grading_levels,
                'mold_fine_band_cells': self.mold.mold_fine_band_cells,
            },
        }
