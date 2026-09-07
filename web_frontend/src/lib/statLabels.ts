/**
 * Presentation for `GridMeta.stats`.
 *
 * The server flattens whatever the run produced into a bag of scalars, and
 * which keys appear depends on which analyses were enabled — so this cannot be
 * a fixed list. Known keys get a real label, a unit and a sensible precision;
 * anything else is humanised rather than dropped, because a statistic the
 * engine bothered to compute should never vanish just because this table has
 * not caught up with it.
 */
import { formatCount, formatNumber } from './format';

export type StatDisplay = {
  key: string;
  label: string;
  value: string;
  /** Longer explanation for a tooltip, when the number needs one. */
  hint?: string;
};

type StatSpec = {
  label: string;
  unit?: string;
  decimals?: number;
  /** Render as a whole number with thousands separators. */
  count?: boolean;
  hint?: string;
};

/** Explicit presentation for the keys the engine is known to emit. */
const SPECS: Record<string, StatSpec> = {
  voxel_count: { label: 'Voxels', count: true, hint: 'Total metal and mold voxels in the solved mesh.' },
  total_volume: { label: 'Total volume', unit: 'in³', decimals: 1 },
  chvorinov_time: { label: 'Chvorinov time', unit: 's', decimals: 1, hint: 't = B x M², the bulk solidification estimate.' },
  fraction_liquid_snapshots: { label: 'Liquid-fraction frames', count: true },
  shrink_scalar: { label: 'Shrink scalar', decimals: 2, hint: 'Global shrinkage calibration applied to the displayed risk.' },
  shrink_scalar_formula: { label: 'Shrink scalar formula' },

  shrinkage_max: { label: 'Max risk', decimals: 3 },
  shrinkage_mean: { label: 'Mean risk', decimals: 3 },
  shrinkage_affected_count: { label: 'Voxels at risk', count: true },
  shrinkage_volume: { label: 'Shrinkage volume', unit: 'in³', decimals: 3 },
  shrinkage_percentage: { label: 'Shrinkage', unit: '%', decimals: 2 },
  defect_volume_estimate: { label: 'Estimated defect volume', unit: 'in³', decimals: 3 },

  hotspot_max: { label: 'Max intensity', decimals: 3 },
  hotspot_mean: { label: 'Mean intensity', decimals: 3 },
  hotspot_count: { label: 'Hot spot voxels', count: true },

  niyama_max_risk: { label: 'Max risk', decimals: 3 },
  niyama_mean_risk: { label: 'Mean risk', decimals: 3 },
  niyama_affected_count: { label: 'Voxels at risk', count: true },
  niyama_defect_count: { label: 'Defect voxels', count: true },
  niyama_metadata_threshold: { label: 'Threshold', decimals: 2 },
  niyama_metadata_threshold_status: { label: 'Threshold source' },
  niyama_metadata_source: { label: 'Computed from' },
  niyama_metadata_units: { label: 'Criterion units' },
  niyama_metadata_gradient_units: { label: 'Gradient units' },
  niyama_metadata_cooling_rate_units: { label: 'Cooling-rate units' },
  niyama_metadata_eval_fraction_solid: { label: 'Evaluated at fraction solid', decimals: 2 },
  niyama_metadata_eval_temperature_c: { label: 'Evaluated at', unit: '°C', decimals: 1 },

  feed_modulus_min_in: { label: 'Min modulus', unit: 'in', decimals: 3 },
  feed_modulus_max_in: { label: 'Max modulus', unit: 'in', decimals: 3 },
  feed_modulus_mean_in: { label: 'Mean modulus', unit: 'in', decimals: 3 },
  fed_fraction: { label: 'Fed fraction', decimals: 3 },
  feeding_blocked_count: { label: 'Blocked voxels', count: true },

  hot_tear_count: { label: 'Hot tear voxels', count: true },
  sand_burn_in_count: { label: 'Sand burn-in voxels', count: true },
  gas_porosity_count: { label: 'Gas porosity voxels', count: true },

  max_stress_mpa: { label: 'Max stress', unit: 'MPa', decimals: 1 },
  avg_sdas_um: { label: 'Mean SDAS', unit: 'µm', decimals: 1 },

  misrun_count: { label: 'Misrun voxels', count: true },
  filled_voxels: { label: 'Filled voxels', count: true },
  unfilled_voxels: { label: 'Unfilled voxels', count: true }
};

/** Section headings, matched against the start of a key, longest prefix first. */
const GROUPS: { prefix: string; label: string }[] = [
  { prefix: 'niyama_metadata_', label: 'Niyama method' },
  { prefix: 'niyama_', label: 'Niyama' },
  { prefix: 'shrinkage_', label: 'Shrinkage' },
  { prefix: 'shrink_scalar', label: 'Shrinkage' },
  { prefix: 'defect_volume', label: 'Shrinkage' },
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
  { prefix: 'fill', label: 'Filling' },
  { prefix: 'misrun', label: 'Filling' },
  { prefix: 'cold_shut', label: 'Filling' },
  { prefix: 'fraction_liquid', label: 'Solidification' },
  { prefix: 'chvorinov', label: 'Solidification' }
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

function groupFor(key: string): string {
  const match = GROUPS.find((group) => key.startsWith(group.prefix));
  return match ? match.label : 'Overview';
}

function renderValue(value: number | string, spec: StatSpec | undefined): string {
  if (typeof value === 'string') return value;
  if (!Number.isFinite(value)) return '--';
  if (spec?.count) return formatCount(value);
  const decimals = spec?.decimals ?? (Number.isInteger(value) && Math.abs(value) >= 1000 ? 0 : 3);
  const text = spec?.count ? formatCount(value) : formatNumber(value, decimals);
  return spec?.unit ? `${text} ${spec.unit}` : text;
}

export type StatGroup = { label: string; rows: StatDisplay[] };

/**
 * Turn the raw stats bag into labelled, grouped, ordered rows.
 *
 * `voxelCount`, `castVoxelCount` and `totalVolumeIn3` come from `GridMeta`
 * itself rather than from `stats`, so they are folded in here — a statistics
 * panel that omitted the cast voxel count would be missing the number people
 * check first.
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
    const spec = SPECS[key];
    push(groupFor(key), {
      key,
      label: spec?.label ?? humanize(key),
      value: renderValue(value, spec),
      hint: spec?.hint
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
