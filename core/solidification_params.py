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
  - MoldParams          : Explicit sand mold, automatic mold box and cores, sands
  - MoldParams          : Explicit sand mold, automatic mold box and cores, sands

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


#: Hard cap on simulated time, seconds (16 h). Runs stop when the casting is
#: solid; the cap only bounds a run that would not finish. 4 h was enough on
#: the semi-infinite sand boundary, but in an explicit sand mold heavy
#: sections take longer: a bronze 13.5 in box under a 10.5 x 15.8 in sleeved
#: riser was still freezing at 4 h (2678 s on the boundary), as Chvorinov
#: with real sand predicts (~13 000 s).
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
    """Parameters for riser feeding distance and shrinkage reduction near risers."""
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
    # Sand core body (bonded silica sand). Core voxels are a conducting sand
    # volume, so a core absorbs heat early and saturates when metal encloses
    # it, then insulates the metal around it (a late hot spot). The defaults
    # give alpha = 0.6 / (1600 * 1140) = 3.3e-7 m²/s, the mold sand above.
    core_thermal_conductivity: float = 0.6     # W/(m·K)
    core_density: float = 1600.0               # kg/m³
    core_specific_heat: float = 1140.0         # J/(kg·K), mean over 20-1500 °C
    # W/(m^2*K) contact conductance between metal and core. The casting
    # shrinks onto a core rather than away from it, so this keeps no air-gap
    # decay; the sand's own heating supplies the late-time drop.
    htc_core_contact: float = 1000.0
    # W/(m^2*K) core print (core sand seated in mold sand) to the mold, with
    # a low late-time floor so the sand penetration decay governs.
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


#: Automatic core rules (``MoldParams.core_rule_code``).
#:
#: 0, ``draw`` (foundry rule, default): a pattern is drawn straight up out of
#: the drag and straight down out of the cope, so sand that has the assembly
#: both above and below it cannot be formed by the mold and needs a core --
#: enclosed cavities, horizontal holes and undercuts.
#: 1, ``enclosed`` (MAGMASOFT's automatic core): sand the outside of the mold
#: cannot reach (a flood fill from the mold's outer faces) is core.
CORE_RULE_DRAW = 0
CORE_RULE_ENCLOSED = 1

#: Sand material code 0: the constant sand of ThermalBoundaryParams instead
#: of a core.mold_materials database sand -- core_thermal_conductivity,
#: core_density and core_specific_heat for cores, mold_thermal_conductivity
#: and mold_thermal_diffusivity for the mold.
SAND_MATERIAL_CUSTOM = 0

#: Exponent of the riser modulus in MoldParams.riser_sleeve_htc.
RISER_SLEEVE_EXPONENT = 0.68

#: Largest MoldParams.mold_grading_levels (cells up to 32 voxels per side).
MOLD_GRADING_MAX_LEVELS = 5


@dataclass
class MoldParams:
    """The sand mold and cores (see core/mold_setup.py, docs/mold_info.txt).

    By default the mold is generated (``auto_mold``) and cores are found
    (``auto_cores``) automatically. Supplied geometry overrides both: a
    loaded mold part replaces the automatic box, and loaded core parts
    replace the automatic cores. With ``auto_mold`` off and no mold part the
    mold is the semi-infinite sand boundary of ThermalBoundaryParams. An
    explicit mold puts the mold in the thermal solve as sand cells around
    the assembly, MAGMASOFT style: metal and chills exchange heat with them
    through interface coefficients, and the outer mold faces and open riser
    tops radiate and convect to the air.

    Coefficients follow the MAGMASOFT practice used in steel validation work
    (Beckermann group, SFSA): ``htc_cast_mold`` above the solidus falling
    along the measured steel-sand air-gap curve (``interface_htc_curve``),
    sand to chill ``htc_chill_mold``. A riser in a sleeve exchanges no heat
    with the sand (``riser_sleeve_adiabatic``), and a sand core meets the mold
    through htc_core_contact.

    Flags and choices are ints (0/1 flags, documented codes), like the other
    groups: both settings UIs render numeric leaves only.
    """
    # 0/1: explicit sand mold box generated around the assembly when no mold
    # part is loaded.
    auto_mold: int = 1
    # 0/1: core sand added where the mold cannot form the cavity, when no
    # core part is loaded.
    auto_cores: int = 1
    core_rule_code: int = CORE_RULE_DRAW
    # Sand between the assembly and the outer mold faces (sides, bottom, and
    # top when no riser is open). 0 = automatic: three heat-penetration
    # depths at the freeze time, which by Chvorinov is 3 x the half-thickness
    # of the heaviest section, between 0.75 and 8 in
    # (core.mold_setup.auto_wall_thickness). Sand beyond that never warms
    # before the casting is solid; on the graded mold (mold_grading_levels)
    # it is solved on coarse cells and costs little.
    # MAGMASOFT steel validation molds carry about 4 in.
    wall_thickness_in: float = 0.0
    # Sand over a blind (closed) riser. Open risers and pour cups stay open
    # to the air at the top of the mold.
    closed_riser_cover_in: float = 2.0
    # core.mold_materials.SAND_MATERIALS codes (0: the constant custom sand).
    mold_material_code: int = 2
    core_material_code: int = SAND_MATERIAL_CUSTOM
    # W/(m^2 K) metal (casting, gating) to mold sand while the metal surface
    # is above its solidus; below it the air-gap curve scales it down.
    htc_cast_mold: float = 1000.0
    # 0/1: follow the steel-sand air-gap curve below the solidus, or hold
    # htc_cast_mold constant (MAGMASOFT "C1000" style).
    interface_htc_curve: int = 1
    # W/(m^2 K) chill to mold sand.
    htc_chill_mold: float = 1500.0
    # 0/1: an insulating sleeve cuts its riser off from the sand (owner,
    # 2026-11-11: the mold and the cores must not change how fast a riser
    # freezes). 1: a sleeved riser passes no heat to the mold sand or to a sand
    # core, on the explicit mold and on the sand boundary alike. It cools
    # through the metal it feeds and, when its top is open in an explicit mold,
    # to the air. 0: the sleeve leaks into the mold at riser_sleeve_htc below
    # (htc_riser on the sand boundary), and a core touching the riser takes
    # heat through htc_core_contact. A riser set to no sleeve
    # (Part.riser_sleeve "none") stands in bare sand either way.
    riser_sleeve_adiabatic: int = 1
    # Used only with riser_sleeve_adiabatic = 0. Insulating riser sleeve
    # against the mold, per riser: h = riser_sleeve_htc
    # * M^-RISER_SLEEVE_EXPONENT, M the riser's modulus in inches. A sleeve wall
    # thickens with the riser, so its resistance grows with size; a constant
    # contact coefficient gave modulus extension factors of 1.31 on a 2 x 3 in
    # riser but 1.15 at 6 x 9 in. 61.7 holds MEF at 1.30-1.31 from M = 0.375 to
    # 1.125 in (the 1.3 core/surrogate/design_rules.py sizes risers for;
    # measured insulating sleeves 1.07-1.28, Williams et al. 2016).
    riser_sleeve_htc: float = 61.7
    # Air side: convective h (W/(m^2 K)), the air temperature, and the
    # emissivities of the mold sand and of an open metal surface (oxidized
    # steel under slag).
    ambient_temperature: float = 21.0
    air_htc: float = 10.0
    sand_emissivity: float = 0.9
    metal_emissivity: float = 0.8
    # Cap on lattice cells in the mold box (voxel-size cells, before grading:
    # the dense layout arrays are lattice sized), so a fine voxel size with a
    # large mold fails with a message instead of exhausting memory.
    max_mold_cells: int = 40_000_000
    # Graded mold (core.mold_setup.grade_mold): sand away from the assembly is
    # solved on cells up to 2^mold_grading_levels voxels per side, block by
    # block, stepping 2:1. 0 = the uniform lattice at the voxel size, as
    # before (bit for bit). The heat penetrates only a few cm of sand before
    # the casting is solid, so the outer mold needs no voxel-size cells.
    mold_grading_levels: int = 3
    # Layers of voxel-size sand kept around every mesh cell (metal, chills,
    # cores) and gap; at least 2 so every metal-sand and core-sand pair
    # meets a voxel-size cell.
    mold_fine_band_cells: int = 4

    def __post_init__(self) -> None:
        self.auto_mold = _clamp_int(self.auto_mold, 0, 1, 1)
        self.auto_cores = _clamp_int(self.auto_cores, 0, 1, 1)
        self.core_rule_code = _clamp_int(self.core_rule_code, 0, 1, CORE_RULE_DRAW)
        self.wall_thickness_in = _clamp_float(self.wall_thickness_in, 0.0, 48.0, 0.0)
        self.closed_riser_cover_in = _clamp_float(self.closed_riser_cover_in, 0.0, 24.0, 2.0)
        self.mold_material_code = _clamp_int(self.mold_material_code, 0, 255, 2)
        self.core_material_code = _clamp_int(self.core_material_code, 0, 255, SAND_MATERIAL_CUSTOM)
        self.htc_cast_mold = _clamp_float(self.htc_cast_mold, 1.0, 1e5, 1000.0)
        self.interface_htc_curve = _clamp_int(self.interface_htc_curve, 0, 1, 1)
        self.htc_chill_mold = _clamp_float(self.htc_chill_mold, 1.0, 1e5, 1500.0)
        self.riser_sleeve_adiabatic = _clamp_int(self.riser_sleeve_adiabatic, 0, 1, 1)
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
    """Parameters for Chvorinov simulation time and safety factors."""
    chvorinov_mold_constant: float = 280.0
    safety_factor_with_risers: float = 2.5
    safety_factor_no_risers: float = 1.5
    min_simulation_time: float = 600.0
    max_simulation_time: float = DEFAULT_MAX_SIMULATION_TIME_S
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
    # 0/1 bool-as-int; see class docstring. When on (the default), the filling
    # stage measures the gating channels on the grid
    # ``fill_grid_coarsening`` asks for and, if any of them comes out at or
    # below the two cells a channel needs before it has an interior, drops the
    # coarsening a step at a time until they clear it or it reaches 1. Turn it
    # off to pin the coarsening exactly as set -- the under-resolution warning
    # still fires, so an under-resolved fill is never silent either way.
    # See core.filling_stage.choose_fill_coarsening.
    choke_auto_resolution: int = 1
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
        self.choke_auto_resolution = _clamp_int(self.choke_auto_resolution, 0, 1, 1)
        self.lattice_velocity_max = _clamp_float(self.lattice_velocity_max, 0.005, 0.2, 0.05)
        self.max_fill_time_s = _clamp_float(self.max_fill_time_s, 1.0, 600.0, 60.0)
        self.cold_shut_arrival_gap_s = _clamp_float(self.cold_shut_arrival_gap_s, 0.01, 10.0, 0.5)
        self.cold_shut_temp_margin_c = _clamp_float(self.cold_shut_temp_margin_c, 0.0, 200.0, 25.0)
        self.entrapment_check_interval = _clamp_int(self.entrapment_check_interval, 1, 1000, 50)


#: The pour rate FillingParams used to default to, before 0 ("derive from the
#: square-root rule") replaced it. See migrate_legacy_pour_rate.
LEGACY_DEFAULT_POUR_RATE_LB_S = 5.0


def migrate_legacy_pour_rate(params: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite a stored pour rate of exactly 5 lb/s to 0, meaning "derive".

    5.0 was the fixed default every project inherited, and it is the right
    rate only for a casting of about 25 lb: a W-pound casting pours in
    sqrt(W) seconds, so the rate is sqrt(W) lb/s -- 20 lb/s for a 400 lb
    casting, 2 lb/s for a 4 lb one. Almost every stored 5.0 is therefore the
    old default rather than a measured decision, and leaving it alone pours
    every existing project at a rate chosen for none of them.

    A deliberate 5.0 cannot be told apart from an inherited one, which is why
    this is applied per *file version* rather than on every deserialization:
    ui.project_state runs it only on projects saved before the migration
    existed, and saving stamps the current schema version, so a user who
    genuinely wants 5 lb/s can set it, save, and have it survive. Migrating on
    every load would make that one value impossible to hold.

    Args:
        params: A serialized SolidificationParameters dict (to_dict shape).
            Not mutated.

    Returns:
        A copy, with filling.pour_rate_lb_s rewritten where it applies.
    """
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
        mold: Sand mold and core setup (explicit mold, automatic mold box and
            cores, mold and core sands).
    """
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
                'riser_sleeve_adiabatic': self.mold.riser_sleeve_adiabatic,
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
