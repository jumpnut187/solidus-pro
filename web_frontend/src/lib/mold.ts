import type { SandInfo, SolidificationParams } from './api';
import { formatCount } from './format';

export type MoldParams = {
  auto_mold: number;
  auto_cores: number;
  core_rule_code: number;
  wall_thickness_in: number;
  closed_riser_cover_in: number;
  mold_material_code: number;
  core_material_code: number;
  htc_cast_mold: number;
  interface_htc_curve: number;
  htc_chill_mold: number;
  riser_sleeve_htc: number;
  ambient_temperature: number;
  air_htc: number;
  sand_emissivity: number;
  metal_emissivity: number;
  max_mold_cells: number;
  mold_grading_levels: number;
  mold_fine_band_cells: number;
};

export const MOLD_DEFAULTS: MoldParams = {
  auto_mold: 1,
  auto_cores: 1,
  core_rule_code: 0,
  wall_thickness_in: 2,
  closed_riser_cover_in: 2.0,
  mold_material_code: 2,
  core_material_code: 0,
  htc_cast_mold: 1000.0,
  interface_htc_curve: 1,
  htc_chill_mold: 1500.0,
  riser_sleeve_htc: 61.7,
  ambient_temperature: 21.0,
  air_htc: 10.0,
  sand_emissivity: 0.9,
  metal_emissivity: 0.8,
  max_mold_cells: 40_000_000,
  mold_grading_levels: 3,
  mold_fine_band_cells: 4
};

export const FALLBACK_SANDS: SandInfo[] = [
  { code: 0, label: 'Custom (constant properties)', uses: '', notes: '', bulk_density: null },
  { code: 1, label: 'Green sand (silica, bentonite, 3.5 % water)', uses: '', notes: '', bulk_density: 1500 },
  { code: 2, label: 'Silica sand, chemically bonded (phenolic urethane / furan no-bake)', uses: '', notes: '', bulk_density: 1488 },
  { code: 3, label: 'Silica core sand, fine (furan / phenolic urethane cold box)', uses: '', notes: '', bulk_density: 1565 },
  { code: 4, label: 'Ceramic sand (sintered mullite beads, Cerabeads type)', uses: '', notes: '', bulk_density: 1690 },
  { code: 5, label: 'Chromite sand', uses: '', notes: '', bulk_density: 2750 },
  { code: 6, label: 'Zircon sand', uses: '', notes: '', bulk_density: 2950 },
  { code: 7, label: 'Olivine sand', uses: '', notes: '', bulk_density: 1850 },
  { code: 8, label: 'Shell sand (resin-coated silica)', uses: '', notes: '', bulk_density: 1357 },
  { code: 9, label: 'Silica sand, dry / unbonded', uses: '', notes: '', bulk_density: 1500 },
  { code: 10, label: 'Bauxite ceramic sand (high-alumina beads)', uses: '', notes: '', bulk_density: 1900 },
  { code: 11, label: 'Alumina-silicate sand (andalusite, Kerphalite type)', uses: '', notes: '', bulk_density: 1650 },
  { code: 12, label: 'Silicon carbide sand', uses: '', notes: '', bulk_density: 2000 }
];

export function readMold(params: SolidificationParams): MoldParams {
  const stored = (params.mold ?? {}) as Record<string, unknown>;
  const out = { ...MOLD_DEFAULTS };
  for (const key of Object.keys(MOLD_DEFAULTS) as (keyof MoldParams)[]) {
    const value = stored[key];
    if (typeof value === 'number' && Number.isFinite(value)) out[key] = value;
  }
  return out;
}

export function writeMold(params: SolidificationParams, mold: MoldParams): SolidificationParams {
  return { ...params, mold: { ...mold } };
}

export function moldPreviewWanted(mold: MoldParams): boolean {
  return Boolean(mold.auto_mold) || Boolean(mold.auto_cores);
}

export function moldPreviewNote(
  summary: { pitch_in: number; mold_cells: number; auto_core_cells: number; graded_mold_cells?: number },
  autoCores: boolean
): string {
  const graded = summary.graded_mold_cells;
  const sand =
    graded !== undefined && graded < summary.mold_cells
      ? `${formatCount(graded)} mold sand cells (graded from ${formatCount(summary.mold_cells)})`
      : `${formatCount(summary.mold_cells)} mold sand cells`;
  return (
    sand +
    (autoCores ? `, ${formatCount(summary.auto_core_cells)} auto-core cells` : '') +
    ` at ${summary.pitch_in.toFixed(3)} in (preview).`
  );
}
