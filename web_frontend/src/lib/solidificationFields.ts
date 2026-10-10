/**
 * The 45 solidification parameters, mirroring
 * `SolidificationParamsDialog.FIELD_DEFS` in `ui/main_window.py` field for
 * field: same labels, defaults, ranges, steps, decimals, units and help text.
 *
 * Two structural quirks of the desktop dialog are reproduced deliberately:
 *
 * - `INERT_FIELD_KEYS` are serialized and round-trip through saved projects but
 *   no solver reads them. They are hidden rather than deleted, because a
 *   control that promises to change the answer and does not is worse than no
 *   control at all.
 * - `FILTER_KEYS` share the Filling tab visually but live one level deeper in
 *   the persisted structure, at `filling.filter.*`.
 */

export type ParamGroup =
  | 'feeding'
  | 'chill'
  | 'exothermic'
  | 'air_gap'
  | 'thermal_boundary'
  | 'timing'
  | 'hot_tear'
  | 'filling';

export type ParamDef = {
  group: ParamGroup;
  key: string;
  label: string;
  default: number;
  min: number;
  max: number;
  step: number;
  decimals: number;
  unit: string;
  help: string;
};

export type ParamTab = {
  id: ParamGroup;
  label: string;
};

export const PARAM_TABS: ParamTab[] = [
  { id: 'feeding', label: 'Feeding' },
  { id: 'chill', label: 'Chills' },
  { id: 'exothermic', label: 'Exothermic' },
  { id: 'air_gap', label: 'Air Gap' },
  { id: 'thermal_boundary', label: 'Thermal' },
  { id: 'timing', label: 'Timing' },
  { id: 'hot_tear', label: 'Hot Tear' },
  { id: 'filling', label: 'Filling' }
];

/** Rendered as a slider paired with a numeric entry, not a bare number box. */
export const SLIDER_KEYS: ReadonlySet<string> = new Set(['feeding_zone_vertical_bias']);

/** Serialized but unread by any solver — hidden from the UI. */
export const INERT_KEYS: ReadonlySet<string> = new Set([
  'pour_time_s',
  'rated_velocity_warn_factor',
  'head_fraction_warn',
  'priming_enabled',
  'drag_enabled'
]);

/** Filling-tab fields that persist under `filling.filter.*`. */
export const FILTER_KEYS: ReadonlySet<string> = new Set([
  'porosity',
  'pore_diameter_mm',
  'ergun_prefactor',
  'forchheimer_c_f',
  'contact_angle_deg',
  'window_fraction',
  'rated_velocity_warn_factor',
  'head_fraction_warn',
  'priming_enabled',
  'drag_enabled'
]);

/** A 0/1 field is a checkbox, not a spin box that accepts 0.5. */
export function isBooleanParam(def: ParamDef): boolean {
  return def.min === 0 && def.max === 1 && def.step === 1 && def.decimals === 0;
}

export const PARAM_DEFS: ParamDef[] = [
  // --- Feeding -------------------------------------------------------------
  {
    group: 'feeding',
    key: 'base_feeding_distance',
    label: 'Base Feeding Distance',
    default: 6.0,
    min: 2.0,
    max: 24.0,
    step: 0.1,
    decimals: 2,
    unit: 'inches',
    help: 'Base riser influence radius. Final radius = this x (alloy feeding_multiplier / 3.0).'
  },
  {
    group: 'feeding',
    key: 'min_influence_factor',
    label: 'Min Influence Factor',
    default: 0.6,
    min: 0.01,
    max: 1.0,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'At edge of riser influence, solidification is slowed to this fraction of normal rate.'
  },
  {
    group: 'feeding',
    key: 'riser_max_reduction',
    label: 'Riser Max Shrinkage Reduction',
    default: 0.1,
    min: 0.01,
    max: 0.5,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'At riser contact, only this fraction of calculated shrinkage remains.'
  },
  {
    group: 'feeding',
    key: 'riser_min_reduction',
    label: 'Riser Min Shrinkage Reduction',
    default: 1.0,
    min: 0.5,
    max: 1.0,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'At edge of feeding distance, full shrinkage (no reduction).'
  },
  {
    group: 'feeding',
    key: 'post_contact_reduction',
    label: 'Feeder Neck Contact Reduction',
    default: 0.35,
    min: 0.1,
    max: 0.8,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'Shrinkage reduction at feeder neck contact with casting.'
  },
  {
    group: 'feeding',
    key: 'disconnected_neck_floor',
    label: 'Disconnected Neck Floor',
    default: 0.7,
    min: 0.3,
    max: 1.0,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'Minimum shrinkage reduction for feeder necks not connected to a riser system.'
  },
  {
    group: 'feeding',
    key: 'shrink_scalar',
    label: 'Shrink Scalar',
    default: 1.0,
    min: 0.25,
    max: 4.0,
    step: 0.05,
    decimals: 2,
    unit: '',
    help: 'Global calibration of the Shrink and Microporosity results. Above 1 reduces displayed risk; below 1 increases it.'
  },
  {
    group: 'feeding',
    key: 'feeding_zone_vertical_bias',
    label: 'Feeding-Zone +Z Bias',
    default: 0.5,
    min: 0.0,
    max: 2.0,
    step: 0.05,
    decimals: 2,
    unit: '',
    help:
      'Strength of +Z directional solidification inside the riser feeding zone. Higher hardens directional freezing toward risers. 0 disables the feeding-zone extra (a constant 0.3 baseline bias still applies to the whole cast). No-op when no riser voxels are present.'
  },
  // --- Chills --------------------------------------------------------------
  {
    group: 'chill',
    key: 'max_chill_boost',
    label: 'Max Chill Boost',
    default: 3.0,
    min: 1.0,
    max: 10.0,
    step: 0.1,
    decimals: 1,
    unit: 'multiplier',
    help: 'Maximum solidification acceleration near chills.'
  },
  {
    group: 'chill',
    key: 'shrinkage_radius_multiplier',
    label: 'Shrinkage Radius Multiplier',
    default: 1.5,
    min: 0.5,
    max: 5.0,
    step: 0.1,
    decimals: 2,
    unit: 'x chill thickness',
    help: 'Shrinkage reduction zone = chill thickness x this value.'
  },
  {
    group: 'chill',
    key: 'shrinkage_max_reduction',
    label: 'Chill Max Shrinkage Reduction',
    default: 0.15,
    min: 0.01,
    max: 0.5,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'At chill contact, only this fraction of calculated shrinkage remains.'
  },
  {
    group: 'chill',
    key: 'shrinkage_min_reduction',
    label: 'Chill Min Shrinkage Reduction',
    default: 1.0,
    min: 0.5,
    max: 1.0,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'At edge of chill influence, full shrinkage (no reduction).'
  },
  {
    group: 'chill',
    key: 'thickness_fallback_voxels',
    label: 'Chill Thickness Fallback',
    default: 5.0,
    min: 1.0,
    max: 20.0,
    step: 1.0,
    decimals: 0,
    unit: 'voxels',
    help: 'When no explicit chill thickness is available, estimate as this x voxel_size.'
  },
  // --- Exothermic ----------------------------------------------------------
  {
    group: 'exothermic',
    key: 'ignition_delay',
    label: 'Ignition Delay',
    default: 10.0,
    min: 0.0,
    max: 60.0,
    step: 1.0,
    decimals: 0,
    unit: 'seconds',
    help: 'Seconds for an exothermic sleeve to reach ignition temperature.'
  },
  {
    group: 'exothermic',
    key: 'burn_duration',
    label: 'Burn Duration',
    default: 120.0,
    min: 30.0,
    max: 600.0,
    step: 10.0,
    decimals: 0,
    unit: 'seconds',
    help: 'Duration of the active exothermic reaction.'
  },
  {
    group: 'exothermic',
    key: 'insulation_only_factor',
    label: 'Insulation-Only Factor',
    default: 0.7,
    min: 0.1,
    max: 1.0,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'Post-burnout: solidification rate as a fraction of normal (0.7 = 1.4x slower).'
  },
  {
    group: 'exothermic',
    key: 'full_exothermic_factor',
    label: 'Full Exothermic Factor',
    default: 0.5,
    min: 0.1,
    max: 0.9,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'Active burn: solidification rate as a fraction of normal (0.5 = 2x slower).'
  },
  {
    group: 'exothermic',
    key: 'decay_time_constant',
    label: 'Decay Time Constant',
    default: 60.0,
    min: 10.0,
    max: 300.0,
    step: 5.0,
    decimals: 0,
    unit: 'seconds',
    help: 'Time constant for exothermic effect decay after burnout.'
  },
  // --- Air gap -------------------------------------------------------------
  {
    group: 'air_gap',
    key: 'time_constant',
    label: 'Air Gap Time Constant',
    default: 60.0,
    min: 10.0,
    max: 600.0,
    step: 10.0,
    decimals: 0,
    unit: 'seconds',
    help: 'Time over which the air gap develops.'
  },
  {
    group: 'air_gap',
    key: 'surface_influence_depth',
    label: 'Surface Influence Depth',
    default: 0.5,
    min: 0.1,
    max: 3.0,
    step: 0.1,
    decimals: 2,
    unit: 'inches',
    help: 'How far the air-gap effect extends from the casting surface.'
  },
  {
    group: 'air_gap',
    key: 'max_slowdown',
    label: 'Max Air Gap Slowdown',
    default: 0.75,
    min: 0.1,
    max: 0.95,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'Maximum slowdown at the surface due to the air gap (0.75 = 25% slower).'
  },
  // --- Thermal boundary ----------------------------------------------------
  {
    group: 'thermal_boundary',
    key: 'mold_temperature',
    label: 'Mold Temperature',
    default: 21.0,
    min: 20.0,
    max: 500.0,
    step: 5.0,
    decimals: 0,
    unit: 'degC',
    help: 'Mold surface boundary condition temperature.'
  },
  {
    group: 'thermal_boundary',
    key: 'chill_temperature',
    label: 'Chill Temperature',
    default: 21.0,
    min: 0.0,
    max: 100.0,
    step: 5.0,
    decimals: 0,
    unit: 'degC',
    help: 'Chill fixed temperature boundary condition.'
  },
  {
    group: 'thermal_boundary',
    key: 'htc_cast_interface',
    label: 'HTC Cast Interface',
    default: 500.0,
    min: 100.0,
    max: 2000.0,
    step: 50.0,
    decimals: 0,
    unit: 'W/m2K',
    help: 'Heat-transfer coefficient at the cast-mold interface.'
  },
  {
    group: 'thermal_boundary',
    key: 'htc_riser_post',
    label: 'HTC Riser Post',
    default: 350.0,
    min: 100.0,
    max: 2000.0,
    step: 50.0,
    decimals: 0,
    unit: 'W/m2K',
    help: 'Heat-transfer coefficient at the riser-post-mold interface.'
  },
  {
    group: 'thermal_boundary',
    key: 'htc_riser',
    label: 'HTC Riser',
    default: 220.0,
    min: 100.0,
    max: 2000.0,
    step: 50.0,
    decimals: 0,
    unit: 'W/m2K',
    help: 'Heat-transfer coefficient at the riser-mold interface.'
  },
  {
    group: 'thermal_boundary',
    key: 'htc_gating',
    label: 'HTC Gating',
    default: 400.0,
    min: 100.0,
    max: 2000.0,
    step: 50.0,
    decimals: 0,
    unit: 'W/m2K',
    help: 'Heat-transfer coefficient at the gating/filter-mold interface (sprue, runner, gate, pour cup, filter).'
  },
  // --- Timing --------------------------------------------------------------
  {
    group: 'timing',
    key: 'chvorinov_mold_constant',
    label: 'Chvorinov Mold Constant',
    default: 280.0,
    min: 50.0,
    max: 1000.0,
    step: 10.0,
    decimals: 0,
    unit: 's/in2',
    help: 'Chvorinov mold constant B (t = B*M^2). About 280 s/in2 for WCB in sand.'
  },
  {
    group: 'timing',
    key: 'safety_factor_with_risers',
    label: 'Safety Factor (with Risers)',
    default: 2.5,
    min: 1.0,
    max: 10.0,
    step: 0.1,
    decimals: 2,
    unit: 'multiplier',
    help: 'Time safety factor when risers are present.'
  },
  {
    group: 'timing',
    key: 'safety_factor_no_risers',
    label: 'Safety Factor (no Risers)',
    default: 1.5,
    min: 1.0,
    max: 10.0,
    step: 0.1,
    decimals: 2,
    unit: 'multiplier',
    help: 'Time safety factor when no risers are present.'
  },
  {
    group: 'timing',
    key: 'min_simulation_time',
    label: 'Min Simulation Time',
    default: 120.0,
    min: 10.0,
    max: 600.0,
    step: 10.0,
    decimals: 0,
    unit: 'seconds',
    help: 'Minimum simulation time for any casting.'
  },
  {
    group: 'timing',
    key: 'max_simulation_time',
    label: 'Max Simulation Time',
    default: 14400.0,
    min: 600.0,
    max: 86400.0,
    step: 600.0,
    decimals: 0,
    unit: 'seconds',
    help: 'Maximum simulation time cap (thermal-coupled mode). 14400 s = 4 hours.'
  },
  {
    group: 'timing',
    key: 'sand_saturation_time',
    label: 'Sand Saturation Time',
    default: 30.0,
    min: 5.0,
    max: 300.0,
    step: 5.0,
    decimals: 0,
    unit: 'seconds',
    help: 'Time constant for sand heat saturation around riser posts.'
  },
  {
    group: 'timing',
    key: 'max_hot_sand_slowdown',
    label: 'Max Hot Sand Slowdown',
    default: 0.6,
    min: 0.1,
    max: 0.95,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'Maximum slowdown from heat-saturated sand (0.6 = 1.7x slower).'
  },
  // --- Hot tear ------------------------------------------------------------
  {
    group: 'hot_tear',
    key: 'mold_restraint_index',
    label: 'Mold Restraint Index',
    default: 0.5,
    min: 0.0,
    max: 1.0,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'Mold restraint level at the casting surface (0.0 = free, 1.0 = fully restrained).'
  },
  {
    group: 'hot_tear',
    key: 'metal_mold_friction_index',
    label: 'Metal/Mold Friction',
    default: 0.5,
    min: 0.0,
    max: 1.0,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'Friction coefficient at the metal-mold interface.'
  },
  {
    group: 'hot_tear',
    key: 'core_restraint_index',
    label: 'Core Restraint Index',
    default: 0.0,
    min: 0.0,
    max: 1.0,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'Core restraint level at core voxels (0.0 = no restraint, 1.0 = fully restrained).'
  },
  {
    group: 'hot_tear',
    key: 'low_melting_impurity_index',
    label: 'Impurity Index',
    default: 0.0,
    min: 0.0,
    max: 1.0,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'Low-melting impurity level affecting hot tear susceptibility.'
  },
  {
    group: 'hot_tear',
    key: 'hot_tear_zone_threshold',
    label: 'Hot Tear Threshold',
    default: 0.0,
    min: 0.0,
    max: 1.0,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'Threshold for hot tear zone detection (0.0 = adaptive, based on alloy).'
  },
  {
    group: 'hot_tear',
    key: 'healing_relief_strength',
    label: 'Healing Relief Strength',
    default: 0.65,
    min: 0.0,
    max: 1.0,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'Feeding healing effectiveness at reducing hot tear risk.'
  },
  {
    group: 'hot_tear',
    key: 'surface_restraint_depth_inches',
    label: 'Surface Restraint Depth',
    default: 0.5,
    min: 0.1,
    max: 3.0,
    step: 0.1,
    decimals: 2,
    unit: 'inches',
    help: 'Depth from the surface where mold restraint applies.'
  },
  {
    group: 'hot_tear',
    key: 'solid_path_search_depth',
    label: 'Solid Path Search Depth',
    default: 8,
    min: 2,
    max: 20,
    step: 1,
    decimals: 0,
    unit: 'voxels',
    help: 'Search depth for the solidification path strain calculation.'
  },
  // --- Filling -------------------------------------------------------------
  {
    group: 'filling',
    key: 'pour_rate_lb_s',
    label: 'Pour Rate',
    default: 0.0,
    min: 0.0,
    max: 50.0,
    step: 0.5,
    decimals: 2,
    unit: 'lb/s',
    help:
      'Metal pour rate into the mold. 0 = derive from the square-root rule: a casting of W lb pours in sqrt(W) seconds. Only used when Inlet Mode is Velocity.'
  },
  {
    group: 'filling',
    key: 'pour_time_s',
    label: 'Pour Time',
    default: 0.0,
    min: 0.0,
    max: 120.0,
    step: 1.0,
    decimals: 0,
    unit: 'seconds',
    help: 'Fixed pour duration. 0 = derive from pour rate and casting volume.'
  },
  {
    group: 'filling',
    key: 'pour_temperature_c',
    label: 'Pour Temperature',
    default: 0.0,
    min: 0.0,
    max: 1700.0,
    step: 10.0,
    decimals: 0,
    unit: 'degC',
    help: "Metal temperature at pour. 0 = use the alloy's default pouring temperature."
  },
  {
    group: 'filling',
    key: 'initial_head_in',
    label: 'Initial Head',
    default: 4.0,
    min: 0.5,
    max: 24.0,
    step: 0.5,
    decimals: 2,
    unit: 'inches',
    help: 'Initial metallostatic head height at the start of the pour.'
  },
  {
    group: 'filling',
    key: 'fill_grid_coarsening',
    label: 'Fill Grid Coarsening',
    default: 2,
    min: 1,
    max: 8,
    step: 1,
    decimals: 0,
    unit: 'x voxel',
    help: 'Coarsening factor applied to the solidification grid for the filling-stage lattice.'
  },
  {
    group: 'filling',
    key: 'choke_auto_resolution',
    label: 'Auto Refine Under-Resolved Chokes',
    default: 1,
    min: 0,
    max: 1,
    step: 1,
    decimals: 0,
    unit: '',
    help:
      'Steps Fill Grid Coarsening down toward 1 until every gating channel is more than two cells thick. A channel two cells or thinner has no interior and seals shut. Off pins the coarsening exactly as set; the under-resolution warning fires either way.'
  },
  {
    group: 'filling',
    key: 'max_fill_time_s',
    label: 'Max Fill Time',
    default: 60.0,
    min: 1.0,
    max: 600.0,
    step: 5.0,
    decimals: 0,
    unit: 'seconds',
    help: 'Upper bound on simulated fill time before the filling stage is cut off.'
  },
  {
    group: 'filling',
    key: 'smagorinsky_cs',
    label: 'Smagorinsky Cs',
    default: 0.16,
    min: 0.05,
    max: 0.3,
    step: 0.01,
    decimals: 2,
    unit: '',
    help:
      'Smagorinsky constant for the lattice-Boltzmann turbulence sub-grid model. 0.16 is standard; raising it damps the fill front harder, lowering it resolves more detail with less stability margin.'
  },
  {
    group: 'filling',
    key: 'lattice_velocity_max',
    label: 'Lattice Velocity Max',
    default: 0.05,
    min: 0.005,
    max: 0.2,
    step: 0.005,
    decimals: 3,
    unit: 'lattice units',
    help: 'Maximum stable lattice velocity for the filling-stage solver.'
  },
  {
    group: 'filling',
    key: 'wall_htc_multiplier',
    label: 'Wall HTC Multiplier',
    default: 1.0,
    min: 0.1,
    max: 5.0,
    step: 0.1,
    decimals: 2,
    unit: 'multiplier',
    help: 'Multiplier applied to mold-wall heat-transfer coefficients during filling.'
  },
  {
    group: 'filling',
    key: 'inlet_mode_code',
    label: 'Inlet Mode (0 = Head, 1 = Velocity)',
    default: 0,
    min: 0,
    max: 1,
    step: 1,
    decimals: 0,
    unit: '',
    help: 'Inlet boundary mode: 0 = head-driven, 1 = prescribed velocity.'
  },
  {
    group: 'filling',
    key: 'pour_mode_code',
    label: 'Pour Mode (0 = Lip, 1 = Bottom)',
    default: 0,
    min: 0,
    max: 1,
    step: 1,
    decimals: 0,
    unit: '',
    help: 'Pour mode: 0 = lip (top) pour, 1 = bottom pour.'
  },
  {
    group: 'filling',
    key: 'vent_risers',
    label: 'Vent Risers Enabled',
    default: 1,
    min: 0,
    max: 1,
    step: 1,
    decimals: 0,
    unit: '',
    help: 'Whether open risers act as air vents during filling.'
  },
  {
    group: 'filling',
    key: 'cold_shut_arrival_gap_s',
    label: 'Cold Shut Arrival Gap',
    default: 0.5,
    min: 0.01,
    max: 10.0,
    step: 0.05,
    decimals: 2,
    unit: 'seconds',
    help: 'Minimum arrival-time difference between converging fronts to count as a cold-shut candidate.'
  },
  {
    group: 'filling',
    key: 'cold_shut_temp_margin_c',
    label: 'Cold Shut Temp Margin',
    default: 25.0,
    min: 0.0,
    max: 200.0,
    step: 5.0,
    decimals: 0,
    unit: 'degC',
    help: 'Degrees above the liquidus still counted as cold enough to risk a cold shut.'
  },
  {
    group: 'filling',
    key: 'entrapment_check_interval',
    label: 'Entrapment Check Interval',
    default: 50,
    min: 1,
    max: 1000,
    step: 10,
    decimals: 0,
    unit: 'steps',
    help: 'Lattice steps between air-entrapment connected-component checks during the fill.'
  },
  // Filter sub-group (persists under filling.filter.*)
  {
    group: 'filling',
    key: 'porosity',
    label: 'Filter Porosity',
    default: 0.85,
    min: 0.5,
    max: 0.95,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'Ceramic filter media porosity (void fraction).'
  },
  {
    group: 'filling',
    key: 'pore_diameter_mm',
    label: 'Filter Pore Diameter',
    default: 2.54,
    min: 0.5,
    max: 8.0,
    step: 0.1,
    decimals: 2,
    unit: 'mm',
    help: 'Ceramic filter pore diameter (2.54 mm is about 10 ppi).'
  },
  {
    group: 'filling',
    key: 'ergun_prefactor',
    label: 'Filter Ergun Prefactor',
    default: 2.2,
    min: 0.5,
    max: 10.0,
    step: 0.1,
    decimals: 2,
    unit: '',
    help: 'Ergun-equation viscous-drag prefactor for flow through the filter.'
  },
  {
    group: 'filling',
    key: 'forchheimer_c_f',
    label: 'Filter Forchheimer C_f',
    default: 0.55,
    min: 0.1,
    max: 2.0,
    step: 0.05,
    decimals: 2,
    unit: '',
    help: 'Forchheimer-term inertial-drag coefficient for flow through the filter.'
  },
  {
    group: 'filling',
    key: 'contact_angle_deg',
    label: 'Filter Contact Angle',
    default: 120.0,
    min: 90.0,
    max: 160.0,
    step: 1.0,
    decimals: 0,
    unit: 'degrees',
    help: 'Metal/filter-media wetting contact angle used for priming resistance.'
  },
  {
    group: 'filling',
    key: 'window_fraction',
    label: 'Filter Window Fraction',
    default: 0.45,
    min: 0.2,
    max: 0.8,
    step: 0.01,
    decimals: 2,
    unit: 'fraction',
    help: 'Open-window fraction of the filter footprint available to flow.'
  },
  {
    group: 'filling',
    key: 'rated_velocity_warn_factor',
    label: 'Filter Rated Velocity Warn Factor',
    default: 1.5,
    min: 0.5,
    max: 5.0,
    step: 0.1,
    decimals: 2,
    unit: 'multiplier',
    help: 'Warn when filter approach velocity exceeds this multiple of the rated velocity.'
  },
  {
    group: 'filling',
    key: 'head_fraction_warn',
    label: 'Filter Head Fraction Warn',
    default: 0.25,
    min: 0.05,
    max: 1.0,
    step: 0.05,
    decimals: 2,
    unit: 'fraction',
    help: 'Warn when the head consumed across the filter exceeds this fraction of available head.'
  },
  {
    group: 'filling',
    key: 'priming_enabled',
    label: 'Filter Priming Enabled',
    default: 1,
    min: 0,
    max: 1,
    step: 1,
    decimals: 0,
    unit: '',
    help: 'Whether contact-angle driven filter priming resistance is modeled.'
  },
  {
    group: 'filling',
    key: 'drag_enabled',
    label: 'Filter Drag Enabled',
    default: 1,
    min: 0,
    max: 1,
    step: 1,
    decimals: 0,
    unit: '',
    help: 'Whether filter flow drag (Ergun/Forchheimer) is modeled.'
  }
];

/** Visible (non-inert) fields for one tab, filter fields last. */
export function paramsForTab(tab: ParamGroup): { main: ParamDef[]; filter: ParamDef[] } {
  const visible = PARAM_DEFS.filter((def) => def.group === tab && !INERT_KEYS.has(def.key));
  return {
    main: visible.filter((def) => !(tab === 'filling' && FILTER_KEYS.has(def.key))),
    filter: tab === 'filling' ? visible.filter((def) => FILTER_KEYS.has(def.key)) : []
  };
}
