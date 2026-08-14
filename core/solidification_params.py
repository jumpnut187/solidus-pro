"""
SolidificationParameters: Tunable parameters for the solidification simulation.

This module exposes ~30 previously hardcoded constants as a structured,
serializable dataclass hierarchy so they can be configured via:
  - YAML config files (castsim.yaml)
  - Environment variables (CASTSIM_SOLIDIFICATION_*)
  - UI settings modal
  - Direct Python instantiation

The parameters are organized into 6 subgroups:
  - FeedingParams      : Riser feeding distance and shrinkage reduction near risers
  - ChillParams         : Chill thermal boost and shrinkage reduction near chills
  - ExothermicParams    : Exothermic riser sleeve timing and effectiveness
  - AirGapParams        : Mold air-gap development and surface influence
  - ThermalBoundaryParams: Mold/chill temperatures and HTC values
  - TimingParams        : Chvorinov mold constant and simulation time bounds

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
    """
    feeding: FeedingParams = field(default_factory=FeedingParams)
    chill: ChillParams = field(default_factory=ChillParams)
    exothermic: ExothermicParams = field(default_factory=ExothermicParams)
    air_gap: AirGapParams = field(default_factory=AirGapParams)
    thermal_boundary: ThermalBoundaryParams = field(default_factory=ThermalBoundaryParams)
    timing: TimingParams = field(default_factory=TimingParams)
    hot_tear: HotTearParams = field(default_factory=HotTearParams)

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
        }
