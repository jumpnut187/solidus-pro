/**
 * Presentation for `GridMeta.stats`.
 *
 * Which keys appear depends on which analyses a run enabled, and a statistic
 * the run did not produce is *absent* rather than zero -- "no hot tears" and
 * "hot tearing was not analysed" are different answers, and the panel must not
 * flatten them into the same 0.
 *
 * So this cannot be a fixed list. The server names statistics to a convention
 * (`core/simulation_pipeline._add_summary_statistics`): flat scalars, a unit
 * suffix wherever the unit is not implied (`_s`, `_in3`, `_in`, `_mpa`, `_um`,
 * `_percentage`), `_count` for whole voxels and `_fraction` for 0..1. Units and
 * precision are derived from that suffix, so a statistic added upstream shows
 * up correctly without this file being touched. The table below only overrides
 * the handful whose derived wording reads badly, or that deserve a tooltip.
 */
import { formatCount, formatNumber } from './format';

export type StatDisplay = {
  key: string;
  label: string;
  value: string;
  /** Longer explanation for a tooltip, when the number needs one. */
  hint?: string;
};

type Unit = { suffix: string; unit: string; decimals: number };

/** Longest suffix first: `_max_in3` must not be read as `_in`. */
const UNIT_SUFFIXES: Unit[] = [
  { suffix: '_in3', unit: 'in³', decimals: 3 },
  { suffix: '_mpa', unit: 'MPa', decimals: 1 },
  { suffix: '_um', unit: 'µm', decimals: 1 },
  { suffix: '_percentage', unit: '%', decimals: 2 },
  { suffix: '_fraction', unit: '', decimals: 3 },
  { suffix: '_in', unit: 'in', decimals: 3 },
  { suffix: '_s', unit: 's', decimals: 1 }
];

const COUNT_SUFFIXES = ['_count', '_voxel_count', '_snapshots'];

/** Overrides for names whose derived label or precision is not good enough. */
const OVERRIDES: Record<string, { label?: string; hint?: string; decimals?: number }> = {
  alloy: { label: 'Alloy' },
  voxel_count: { label: 'Voxels', hint: 'Total metal and mold voxels in the solved mesh.' },
  total_volume: { label: 'Total volume' },
  simulation_time_s: { label: 'Simulated time' },
  chvorinov_time_max_s: {
    label: 'Chvorinov time (max)',
    hint: 't = B x M²: the bulk solidification estimate for the slowest section.'
  },
  chvorinov_time_mean_s: { label: 'Chvorinov time (mean)' },
  defect_volume_in3: {
    label: 'Defect volume (probability-weighted)',
    hint: "Each voxel above the engine's default threshold counts its volume times its defect probability."
  },
  defect_volume_max_in3: {
    label: 'Defect volume (upper bound)',
    hint: 'The same voxels counted at full volume, as if every one were entirely defective.'
  },
  // Not "Defect volume" again: the in³ figures above already own that label,
  // and two rows with the same name and different numbers read as a bug.
  defect_volume_percentage: { label: 'Defect share of casting' },
  defect_voxel_count: { label: 'Defect voxels' },
  defect_threshold: { label: 'Threshold used', decimals: 2 },
  casting_volume_in3: { label: 'Casting volume' },
  shrinkage_volume_in3: { label: 'Shrinkage volume' },
  // Two counts arrive for shrinkage, for hot spots and for Niyama, and they
  // differ by an order of magnitude because they are counted against different
  // thresholds: one on the displayed field, one by the solver itself. Labelling
  // both "voxels at risk" would put two contradictory numbers side by side.
  shrinkage_voxel_count: {
    label: 'Voxels flagged by the solver',
    hint: "The solver's own shrinkage threshold, which is stricter than the display one."
  },
  shrinkage_percentage: { label: 'Share of voxels flagged' },
  shrinkage_max: { label: 'Max risk' },
  shrinkage_mean: { label: 'Mean risk' },
  shrinkage_affected_count: {
    label: 'Voxels showing any risk',
    hint: 'Every voxel with a non-trivial shrinkage value in the displayed field.'
  },
  hotspot_voxel_count: {
    label: 'Voxels flagged by the solver',
    hint: "The solver's own hot spot threshold."
  },
  hotspot_count: {
    label: 'Voxels above display intensity',
    hint: 'Counted on the displayed hot spot field, at a higher bar than the solver uses.'
  },
  hotspot_max: { label: 'Max intensity' },
  hotspot_mean: { label: 'Mean intensity' },
  niyama_defect_count: {
    label: 'Defect voxels',
    hint: "Below the alloy's Niyama threshold, as the solver counts them."
  },
  niyama_affected_count: { label: 'Voxels showing any risk' },
  niyama_max_risk: { label: 'Max risk' },
  niyama_mean_risk: { label: 'Mean risk' },
  niyama_metadata_threshold: { label: 'Threshold', decimals: 2 },
  niyama_metadata_threshold_status: { label: 'Threshold source' },
  niyama_metadata_source: { label: 'Computed from' },
  niyama_metadata_units: { label: 'Criterion units' },
  niyama_metadata_gradient_units: { label: 'Gradient units' },
  niyama_metadata_cooling_rate_units: { label: 'Cooling-rate units' },
  niyama_metadata_eval_fraction_solid: { label: 'Evaluated at fraction solid', decimals: 2 },
  niyama_metadata_eval_temperature_c: { label: 'Evaluated at', decimals: 1 },
  hot_tear_voxel_count: { label: 'Hot tear voxels' },
  sand_burn_in_voxel_count: { label: 'Sand burn-in voxels' },
  gas_porosity_voxel_count: { label: 'Gas porosity voxels' },
  feeding_blocked_count: { label: 'Blocked voxels' },
  fed_fraction: { label: 'Fed fraction', hint: 'Share of the casting a riser can still reach.' },
  feed_modulus_min_in: { label: 'Min modulus' },
  feed_modulus_max_in: { label: 'Max modulus' },
  feed_modulus_mean_in: { label: 'Mean modulus' },
  max_stress_mpa: { label: 'Max stress' },
  avg_sdas_um: { label: 'Mean SDAS' },
  fill_max_time_s: { label: 'Fill time' },
  fill_filled_voxel_count: { label: 'Filled voxels' },
  fill_unfilled_voxel_count: { label: 'Unfilled voxels' },
  fraction_liquid_snapshots: { label: 'Liquid-fraction frames' },
  shrink_scalar: {
    label: 'Shrink scalar',
    decimals: 2,
    hint: 'Global calibration applied to the displayed shrinkage risk.'
  },
  shrink_scalar_formula: { label: 'Shrink scalar formula' }
};

/** Section headings, matched against the start of a key, longest prefix first. */
const GROUPS: { prefix: string; label: string }[] = [
  { prefix: 'niyama_metadata_', label: 'Niyama method' },
  { prefix: 'niyama_', label: 'Niyama' },
  { prefix: 'shrinkage_', label: 'Shrinkage' },
  { prefix: 'shrink_scalar', label: 'Shrinkage' },
  { prefix: 'defect_', label: 'Shrinkage' },
  { prefix: 'casting_volume', label: 'Overview' },
  { prefix: 'hotspot_', label: 'Hot spots' },
  { prefix: 'hot_tear', label: 'Hot tear' },
  { prefix: 'feed_modulus_', label: 'Feeding' },
  { prefix: 'feeding_', label: 'Feeding' },
  { prefix: 'fed_', label: 'Feeding' },
  { prefix: 'sand_burn_in', label: 'Sand burn-in' },
  { prefix: 'gas_', label: 'Gas' },
  { prefix: 'max_stress', label: 'Stress' },
  { prefix: 'thermal_stress', label: 'Stress' },
  { prefix: 'distortion', label: 'Stress' },
  { prefix: 'crack', label: 'Stress' },
  { prefix: 'sdas', label: 'Microstructure' },
  { prefix: 'avg_sdas', label: 'Microstructure' },
  { prefix: 'predicted_uts', label: 'Microstructure' },
  { prefix: 'fill_', label: 'Filling' },
  { prefix: 'misrun', label: 'Filling' },
  { prefix: 'cold_shut', label: 'Filling' },
  { prefix: 'fraction_liquid', label: 'Solidification' },
  { prefix: 'chvorinov', label: 'Solidification' },
  { prefix: 'simulation_time', label: 'Solidification' }
];

const GROUP_ORDER = [
  'Overview',
  'Solidification',
  'Shrinkage',
  'Feeding',
  'Hot spots',
  'Niyama',
  'Niyama method',
  'Hot tear',
  'Gas',
  'Sand burn-in',
  'Filling',
  'Stress',
  'Microstructure',
  'Other'
];

function humanize(key: string): string {
  const words = key.replace(/_/g, ' ').trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** Strip the unit suffix from a key so the label is not "Fill max time s". */
function describe(key: string): { label: string; unit: string; decimals: number; count: boolean } {
  const override = OVERRIDES[key];
  const count = COUNT_SUFFIXES.some((suffix) => key.endsWith(suffix));
  if (count) {
    return {
      label: override?.label ?? humanize(key.replace(/_(voxel_)?count$/, '').replace(/_snapshots$/, '')),
      unit: '',
      decimals: 0,
      count: true
    };
  }
  const match = UNIT_SUFFIXES.find((entry) => key.endsWith(entry.suffix));
  const stem = match ? key.slice(0, -match.suffix.length) : key;
  return {
    label: override?.label ?? humanize(stem),
    unit: match?.unit ?? '',
    decimals: override?.decimals ?? match?.decimals ?? 3,
    count: false
  };
}

function renderValue(value: number | string, spec: ReturnType<typeof describe>): string {
  if (typeof value === 'string') return value;
  if (!Number.isFinite(value)) return '--';
  if (spec.count) return formatCount(value);
  const text = formatNumber(value, spec.decimals);
  return spec.unit ? `${text} ${spec.unit}` : text;
}

function groupFor(key: string): string {
  const match = GROUPS.find((group) => key.startsWith(group.prefix));
  return match ? match.label : 'Overview';
}

export type StatGroup = { label: string; rows: StatDisplay[] };

/**
 * Turn the raw stats bag into labelled, grouped, ordered rows.
 *
 * `voxelCount`, `castVoxelCount` and `totalVolumeIn3` come from `GridMeta`
 * itself rather than from `stats`, so they are folded in here -- a statistics
 * panel that omitted the cast voxel count would be missing the number people
 * check first. Each is included only when the viewport actually reported it;
 * none of them is defaulted to zero.
 */
export function buildStatGroups(
  stats: Record<string, number | string>,
  extras: { voxelCount: number; castVoxelCount: number; totalVolumeIn3: number; spacing: number }
): StatGroup[] {
  const byGroup = new Map<string, StatDisplay[]>();
  const push = (group: string, row: StatDisplay) => {
    const rows = byGroup.get(group);
    if (rows) rows.push(row);
    else byGroup.set(group, [row]);
  };

  if (extras.voxelCount > 0) {
    push('Overview', { key: '_voxels', label: 'Voxels', value: formatCount(extras.voxelCount) });
  }
  if (extras.castVoxelCount > 0) {
    push('Overview', {
      key: '_cast_voxels',
      label: 'Cast voxels',
      value: formatCount(extras.castVoxelCount),
      hint: 'Voxels belonging to cast, riser and riser-post bodies.'
    });
  }
  if (extras.totalVolumeIn3 > 0) {
    push('Overview', {
      key: '_volume',
      label: 'Metal volume',
      value: `${formatNumber(extras.totalVolumeIn3, 1)} in³`
    });
  }
  if (extras.spacing > 0) {
    push('Overview', {
      key: '_spacing',
      label: 'Voxel size',
      value: `${formatNumber(extras.spacing, 3)} in`
    });
  }

  for (const [key, value] of Object.entries(stats)) {
    // Already shown from GridMeta's own fields, in better units.
    if (key === 'voxel_count' || key === 'total_volume') continue;
    const spec = describe(key);
    push(groupFor(key), {
      key,
      label: spec.label,
      value: renderValue(value, spec),
      hint: OVERRIDES[key]?.hint
    });
  }

  return [...byGroup.entries()]
    .map(([label, rows]) => ({ label, rows }))
    .sort((a, b) => {
      const ai = GROUP_ORDER.indexOf(a.label);
      const bi = GROUP_ORDER.indexOf(b.label);
      return (ai === -1 ? GROUP_ORDER.length : ai) - (bi === -1 ? GROUP_ORDER.length : bi);
    });
}
