"""
SolidificationParameters: Tunable parameters for the solidification simulation.

This module exposes ~30 previously hardcoded constants as a structured,
serializable dataclass hierarchy so they can be configured via:
  - YAML config files (castsim.yaml)
  - Environment variables (CASTSIM_SOLIDIFICATION_*)
  - UI settings modal
  - Direct Python instantiation

The parameters are organized into subgroups:
  - FeedingParams      : Riser feeding distance and shrinkage reduction near risers
  - ChillParams         : Chill thermal boost and shrinkage reduction near chills
  - ExothermicParams    : Exothermic riser sleeve timing and effectiveness
  - AirGapParams        : Mold air-gap development and surface influence
  - ThermalBoundaryParams: Mold/chill temperatures and HTC values
  - TimingParams        : Chvorinov mold constant and simulation time bounds
  - HotTearParams       : Hot tear risk analysis parameters
  - FillingParams       : Mold-filling stage parameters (nests FilterParams)

Usage:
    from core.solidification_params import SolidificationParameters

    # Defaults
    params = SolidificationParameters()

    # From dict (e.g. loaded from YAML or modal)
    params = SolidificationParameters.from_dict({
        "feeding": {"base_feeding_distance": 8.0},
        "chill": {"shrinkage_max_reduction": 0.10},
    })

    # Access
    fp = params.feeding
    print(fp.base_feeding_distance)   # 8.0 (overridden)
    print(fp.min_influence_factor)    # 0.6 (default)

    # Serialize to dict for storage
    d = params.to_dict()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


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
    """Parameters for riser feeding distance and shrinkage reduction near risers."""
    base_feeding_distance: float = 6.0
    min_influence_factor: float = 0.6
    riser_max_reduction: float = 0.10
    riser_min_reduction: float = 1.0
    post_contact_reduction: float = 0.35
    disconnected_neck_floor: float = 0.70
    shrink_scalar: float = SHRINK_SCALAR_DEFAULT
    feeding_zone_vertical_bias: float = 0.5

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
            bias = 0.5
        if bias != bias:
            bias = 0.5
        self.feeding_zone_vertical_bias = max(0.0, min(2.0, bias))


@dataclass
class ChillParams:
    """Parameters for chill thermal boost and shrinkage reduction near chills."""
    max_chill_boost: float = 3.0
    shrinkage_radius_multiplier: float = 1.5
    shrinkage_max_reduction: float = 0.15
    shrinkage_min_reduction: float = 1.0
    thickness_fallback_voxels: float = 5.0


@dataclass
class ExothermicParams:
    """Parameters for exothermic riser sleeve timing and effectiveness."""
    ignition_delay: float = 10.0
    burn_duration: float = 120.0
    insulation_only_factor: float = 0.7
    full_exothermic_factor: float = 0.5
    decay_time_constant: float = 60.0


@dataclass
class AirGapParams:
    """Parameters for mold air-gap development and surface influence."""
    time_constant: float = 60.0
    surface_influence_depth: float = 0.5
    max_slowdown: float = 0.75
    # Fraction of initial interface HTC remaining after the air gap fully forms.
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
    """Parameters for mold/chill temperatures and heat-transfer coefficients."""
    mold_temperature: float = 21.0
    chill_temperature: float = 21.0
    htc_cast_interface: float = 500.0
    htc_riser_post: float = 350.0
    htc_riser: float = 220.0
    # W/(m^2*K) for gating-system surfaces (pour cup, sprue, runner, gate) and
    # ceramic filters. They sit in the same sand as the casting, so the value
    # belongs between the riser (220, insulated sleeve) and the casting (500,
    # bare sand contact). Without an entry here these surfaces get no boundary
    # condition at all and integrate adiabatically -- a silent failure, since
    # an adiabatic sprue simply never gives up heat.
    htc_gating: float = 400.0
    # W/(m^2*K) heat transfer from chill surfaces into the surrounding mold
    # sand (finite-heat-sink chill model). Same order as the casting-to-sand
    # contact conductance: the chill back face is buried in the same sand.
    chill_ambient_htc: float = 500.0
    # Late-time HTC floor for chill surfaces. The chill is rigid and does not
    # shrink away from the sand the way the casting does, so it keeps a much
    # higher late-time contact conductance than the air-gapped casting faces.
    chill_htc_floor_fraction: float = 0.7
    # Semi-infinite mold heat-absorption model (silica-sand defaults): the
    # effective interface HTC decays as the mold heats up,
    # h_pen(t) = k / sqrt(pi * alpha * t).
    mold_thermal_conductivity: float = 0.6     # W/(m·K)
    mold_thermal_diffusivity: float = 3.3e-7   # m²/s (k/(rho*cp))

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


@dataclass
class TimingParams:
    """Parameters for Chvorinov simulation time and safety factors."""
    chvorinov_mold_constant: float = 280.0
    safety_factor_with_risers: float = 2.5
    safety_factor_no_risers: float = 1.5
    min_simulation_time: float = 600.0
    max_simulation_time: float = 14400.0
    sand_saturation_time: float = 30.0
    max_hot_sand_slowdown: float = 0.6


@dataclass
class FilterParams:
    """Parameters for the ceramic pour filter (foam/extruded) used during
    mold filling.

    priming_enabled and drag_enabled are boolean on/off switches, but every
    leaf in this parameter tree is rendered by the desktop settings dialog
    (ui/main_window.py FIELD_DEFS) and the web settings drawer
    (SolidificationDrawer.tsx), and both of those UIs only know how to render
    a numeric slider/spinbox -- neither has a checkbox or dropdown widget.
    A bool (or str/enum) leaf here would simply not appear in either UI. So
    these are encoded as plain ints with a documented 0/1 meaning instead of
    as bools, matching every other leaf in this tree (see also
    FillingParams' docstring, which has the same constraint for its own
    mode-code fields).
    """
    porosity: float = 0.85
    pore_diameter_mm: float = 2.54          # 10 ppi
    ergun_prefactor: float = 2.2
    forchheimer_c_f: float = 0.55
    contact_angle_deg: float = 120.0
    window_fraction: float = 0.45
    rated_velocity_warn_factor: float = 1.5
    head_fraction_warn: float = 0.25
    priming_enabled: int = 1                # 0/1 bool-as-int; see class docstring
    drag_enabled: int = 1                   # 0/1 bool-as-int; see class docstring

    def __post_init__(self) -> None:
        self.porosity = _clamp_float(self.porosity, 0.50, 0.95, 0.85)
        self.pore_diameter_mm = _clamp_float(self.pore_diameter_mm, 0.5, 8.0, 2.54)
        self.ergun_prefactor = _clamp_float(self.ergun_prefactor, 0.5, 10.0, 2.2)
        self.forchheimer_c_f = _clamp_float(self.forchheimer_c_f, 0.1, 2.0, 0.55)
        self.contact_angle_deg = _clamp_float(self.contact_angle_deg, 90.0, 160.0, 120.0)
        self.window_fraction = _clamp_float(self.window_fraction, 0.2, 0.8, 0.45)


@dataclass
class FillingParams:
    """Parameters for the (not-yet-implemented) mold-filling stage.

    inlet_mode_code (0=head, 1=velocity) and pour_mode_code (0=lip,
    1=bottom) are small closed sets of named modes, and vent_risers is a
    boolean on/off switch. As with FilterParams above, both settings UIs
    only render numeric fields, so these are ints with a documented
    encoding rather than an enum/str or bool -- introducing either would
    make the field invisible in both UIs.
    """
    # 0 = derive from the square-root rule, t = sqrt(W): a casting of W pounds
    # pours in sqrt(W) seconds, so the rate is W/sqrt(W) = sqrt(W) lb/s. The
    # old fixed 5.0 was independent of the casting, which is only right at
    # about 25 lb. See core.filling_stage.recommended_pour_rate_lb_s.
    pour_rate_lb_s: float = 0.0
    pour_time_s: float = 0.0                # 0 = derive from rate and volume
    pour_temperature_c: float = 0.0         # 0 = use alloy.pouring_temp
    initial_head_in: float = 4.0
    fill_grid_coarsening: int = 2
    max_fill_time_s: float = 60.0
    smagorinsky_cs: float = 0.16
    lattice_velocity_max: float = 0.05
    wall_htc_multiplier: float = 1.0
    inlet_mode_code: int = 0                # 0 = head, 1 = velocity
    pour_mode_code: int = 0                 # 0 = lip, 1 = bottom
    vent_risers: int = 1                    # 0/1 bool-as-int; see class docstring
    cold_shut_arrival_gap_s: float = 0.5
    cold_shut_temp_margin_c: float = 25.0
    entrapment_check_interval: int = 50
    filter: FilterParams = field(default_factory=FilterParams)

    def __post_init__(self) -> None:
        self.fill_grid_coarsening = _clamp_int(self.fill_grid_coarsening, 1, 8, 2)
        self.lattice_velocity_max = _clamp_float(self.lattice_velocity_max, 0.005, 0.2, 0.05)
        self.max_fill_time_s = _clamp_float(self.max_fill_time_s, 1.0, 600.0, 60.0)
        self.cold_shut_arrival_gap_s = _clamp_float(self.cold_shut_arrival_gap_s, 0.01, 10.0, 0.5)
        self.cold_shut_temp_margin_c = _clamp_float(self.cold_shut_temp_margin_c, 0.0, 200.0, 25.0)
        self.entrapment_check_interval = _clamp_int(self.entrapment_check_interval, 1, 1000, 50)


@dataclass
class HotTearParams:
    """Parameters for hot tear risk analysis."""
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
    """
    Top-level container for all tunable solidification simulation parameters.

    Attributes:
        feeding: Riser feeding distance and shrinkage reduction parameters.
        chill: Chill thermal boost and shrinkage reduction parameters.
        exothermic: Exothermic sleeve timing and effectiveness parameters.
        air_gap: Mold air-gap development parameters.
        thermal_boundary: Mold/chill temperatures and HTC values.
        timing: Chvorinov time calculation and simulation time bounds.
        hot_tear: Hot tear risk analysis parameters.
        filling: Mold-filling stage parameters (pour rate/time/head, LBM
            solver knobs, and the nested filter media parameters).
    """
    feeding: FeedingParams = field(default_factory=FeedingParams)
    chill: ChillParams = field(default_factory=ChillParams)
    exothermic: ExothermicParams = field(default_factory=ExothermicParams)
    air_gap: AirGapParams = field(default_factory=AirGapParams)
    thermal_boundary: ThermalBoundaryParams = field(default_factory=ThermalBoundaryParams)
    timing: TimingParams = field(default_factory=TimingParams)
    hot_tear: HotTearParams = field(default_factory=HotTearParams)
    filling: FillingParams = field(default_factory=FillingParams)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> SolidificationParameters:
        """
        Create a SolidificationParameters instance from a dict, deep-merging
        with defaults. Partial dicts are valid — only specified keys override.

        Args:
            data: Nested dict matching the structure of to_dict(), or None.

        Returns:
            SolidificationParameters instance with values merged in.
        """
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
            bias = 0.5
        if bias != bias:
            bias = 0.5
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
        self.filling.fill_grid_coarsening = _clamp_int(
            self.filling.fill_grid_coarsening, 1, 8, 2,
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a nested dict suitable for YAML storage or JSON.

        Returns:
            Dict matching the from_dict() expected input format.
        """
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
        }
