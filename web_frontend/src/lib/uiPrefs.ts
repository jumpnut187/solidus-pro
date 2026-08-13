export type SectionState = {
  enabled: boolean;
  mode: 'single' | 'slab';
  axis: 'x' | 'y' | 'z';
  position: number;
  thickness: number;
  invert: boolean;
};

export const defaultUiPrefs = {
  resultMode: 'fraction_liquid',
  threshold: 30,
  timePercent: 0,
  flSpeed: 1,
  modelOpacity: 34,
  showModel: true,
  showResults: true,
  section: {
    enabled: false,
    mode: 'single' as 'single' | 'slab',
    axis: 'z' as 'x' | 'y' | 'z',
    position: 50,
    thickness: 10,
    invert: false,
  } as SectionState,
  ortho: false,
};

export type UiPrefs = typeof defaultUiPrefs;

const UI_STORAGE_KEY = 'solidus.ui.v1';

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function deepMerge<T>(base: T, override: unknown): T {
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

export function loadUiPrefs(): UiPrefs {
  if (typeof window === 'undefined') return defaultUiPrefs;
  try {
    const raw = window.localStorage.getItem(UI_STORAGE_KEY);
    if (!raw) return defaultUiPrefs;
    return deepMerge(defaultUiPrefs, JSON.parse(raw)) as UiPrefs;
  } catch {
    return defaultUiPrefs;
  }
}

export function saveUiPrefs(prefs: UiPrefs): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(UI_STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    /* storage unavailable - ignore */
  }
}

export type ModeControl = {
  legendTitle: string;
  thresholdLabel: string;
  defaultThreshold: number;
};

export const MODE_CONTROL: Record<string, ModeControl> = {
  shrinkage: { legendTitle: 'Shrinkage / Porosity Risk', thresholdLabel: 'X-Ray Level %', defaultThreshold: 30 },
  feed_modulus: { legendTitle: 'Feed Modulus (in)', thresholdLabel: 'Feed Modulus Range %', defaultThreshold: 20 },
  hotspot: { legendTitle: 'Hot Spot Intensity', thresholdLabel: 'Min Intensity %', defaultThreshold: 50 },
  niyama_raw: { legendTitle: 'Niyama (raw)', thresholdLabel: 'Min Value %', defaultThreshold: 10 },
  niyama: { legendTitle: 'Niyama Defect Risk', thresholdLabel: 'Min Defect Risk %', defaultThreshold: 30 },
  combined_feed_risk: { legendTitle: 'Combined Feed Risk', thresholdLabel: 'Min Value %', defaultThreshold: 10 },
  fraction_liquid: { legendTitle: 'Fraction Liquid', thresholdLabel: '', defaultThreshold: 30 },
  hot_tear: { legendTitle: 'Hot Tear Risk', thresholdLabel: 'Min Risk %', defaultThreshold: 30 },
  sand_burn_in: { legendTitle: 'Sand Burn-In', thresholdLabel: 'Min Risk %', defaultThreshold: 30 },
  feeding_quality: { legendTitle: 'Blocked Feeding', thresholdLabel: 'Min Value %', defaultThreshold: 10 },
  thermal_stress: { legendTitle: 'Thermal Stress', thresholdLabel: 'Min Value %', defaultThreshold: 10 },
  distortion: { legendTitle: 'Distortion', thresholdLabel: 'Min Value %', defaultThreshold: 10 },
  crack_susceptibility: { legendTitle: 'Crack Susceptibility', thresholdLabel: 'Min Risk %', defaultThreshold: 10 },
  sdas: { legendTitle: 'SDAS', thresholdLabel: 'Min Value %', defaultThreshold: 10 },
  predicted_uts: { legendTitle: 'Low UTS', thresholdLabel: 'Min Value %', defaultThreshold: 10 },
};
