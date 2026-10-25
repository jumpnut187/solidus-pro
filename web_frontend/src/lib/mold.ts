/**
 * The `mold` group of `solidification_params`, mirroring `MoldParams` in
 * `core/solidification_params.py`. Flags and choices are ints, like every other
 * group: 0/1 flags, `core_rule_code` 0 = draw direction, 1 = enclosed, and sand
 * codes from `core/mold_materials.py` (0 = the constant custom sand).
 */
import type { SandInfo, SolidificationParams } from './api';

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
};

export const MOLD_DEFAULTS: MoldParams = {
  auto_mold: 1,
  auto_cores: 1,
  core_rule_code: 0,
  wall_thickness_in: 0,
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
  max_mold_cells: 40_000_000
};

/** Shown until `/api/sands` answers (or when offline): the codes the server knows. */
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

/** True when the run (and the preview) build sand: the automatic mold or automatic cores. */
export function moldPreviewWanted(mold: MoldParams): boolean {
  return Boolean(mold.auto_mold) || Boolean(mold.auto_cores);
}
