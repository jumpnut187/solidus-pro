import type { AnalysisOptions, PerformanceOptions, ProjectSettings, SolidificationParams } from './api';
import { FILTER_KEYS, PARAM_DEFS } from './solidificationFields';

/**
 * Analysis toggles, matching `ui/dialogs/analysis_options_dialog.py`.
 * `gas_porosity` was missing from the web client entirely.
 */
export const analysisDefaults: AnalysisOptions = {
  shrinkage: true,
  hotspot: true,
  niyama: false,
  hot_tear: false,
  sand_burn_in: false,
  gas_porosity: false,
  fraction_liquid: true,
  feeding_path: false,
  stress_analysis: false,
  microstructure: false,
  // Mold-filling (CFD) analysis. Gated in the UI on a pour location existing
  // and on the backend by core.filling_stage.maybe_run_filling.
  filling: false,
  sand_restraint_index: 0.5
};

/**
 * Performance options. `use_gpu` is read by
 * `web/services/simulation_queue.py` and had no web control before; the
 * desktop defaults it on from `config.settings.use_gpu`.
 */
export const performanceDefaults: PerformanceOptions = {
  use_gpu: true,
  use_incremental: false,
  use_chunked_voxelization: false,
  part_assignment_mode: 'auto',
  part_assignment_workers: 0,
  part_assignment_batch_size: 2500,
  max_voxels_for_visualization: 90000
};

/**
 * Solidification defaults derived from the single field table rather than
 * repeated by hand — the old copy had drifted from the desktop dialog and the
 * two disagreed on several defaults.
 */
export function buildSolidificationDefaults(): SolidificationParams {
  const params: SolidificationParams = {};
  for (const def of PARAM_DEFS) {
    const group = (params[def.group] ??= {});
    if (def.group === 'filling' && FILTER_KEYS.has(def.key)) {
      const filter = (group.filter ??= {}) as Record<string, number>;
      filter[def.key] = def.default;
    } else {
      group[def.key] = def.default;
    }
  }
  return params;
}

export const solidificationDefaults: SolidificationParams = buildSolidificationDefaults();

export const meshDefaults = {
  voxelSize: 0.1,
  voxelSizeMin: 0.02,
  voxelSizeMax: 0.5,
  voxelSizeStep: 0.02,
  refinementMin: 1,
  refinementMax: 4,
  refinement: 3
};

export function defaultSettings(): ProjectSettings {
  return {
    name: 'Untitled Casting',
    alloy: 'WCB',
    voxel_size: meshDefaults.voxelSize,
    refinement_level: meshDefaults.refinement,
    memory_mode: 'auto',
    analysis_options: { ...analysisDefaults },
    performance_options: { ...performanceDefaults },
    solidification_params: buildSolidificationDefaults(),
    pour_location: null
  };
}

/** Niyama threshold hints, matching `AnalysisOptionsDialog.update_niyama_info`. */
const NIYAMA_HINTS: Record<string, number> = { WCB: 1.0, CF8: 0.7, C958: 1.2 };

export function niyamaHint(alloy: string): string {
  const threshold = NIYAMA_HINTS[alloy] ?? 1.0;
  return `Niyama threshold for ${alloy}: below ${threshold.toFixed(1)} indicates risk.`;
}

const SETTINGS_STORAGE_KEY = 'solidus.settings.v2';

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Merge stored settings over the defaults, keeping only keys the defaults
 * declare. A key that has since been removed from the app cannot resurrect
 * itself out of a stale localStorage blob.
 */
export function deepMerge<T>(base: T, override: unknown): T {
  if (!isPlainObject(base) || !isPlainObject(override)) {
    return override === undefined ? base : (override as T);
  }
  const result: Record<string, unknown> = { ...base };
  for (const key of Object.keys(base)) {
    if (key in override) {
      result[key] = deepMerge((base as Record<string, unknown>)[key], (override as Record<string, unknown>)[key]);
    }
  }
  return result as T;
}

export function loadStoredSettings(): ProjectSettings {
  if (typeof window === 'undefined') return defaultSettings();
  try {
    const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return defaultSettings();
    return deepMerge(defaultSettings(), JSON.parse(raw) as unknown);
  } catch {
    return defaultSettings();
  }
}

export function saveStoredSettings(settings: ProjectSettings): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    /* storage unavailable (private mode / quota) - ignore */
  }
}
