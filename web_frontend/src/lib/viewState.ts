/**
 * Shared contract between the application shell (panels, modals, top bar) and
 * the 3D results viewport. Both sides import from this file only; neither
 * side reaches into the other's internals.
 *
 * Ownership: authored by the integration lead. The viewport owner may add
 * fields here ADDITIVELY (new optional props / new state fields with a default
 * in `defaultResultViewState`). The shell owner consumes them.
 */
import type { PourLocationPayload, ProjectPart } from './api';

// ---------------------------------------------------------------------------
// Result modes
// ---------------------------------------------------------------------------

export const RESULT_MODE_IDS = [
  'shrinkage',
  'porosity',
  'feed_modulus',
  'hotspot',
  'solidification_time',
  'niyama_raw',
  'niyama',
  'combined_feed_risk',
  'fraction_liquid',
  'fill_time',
  'fill_animation',
  'hot_tear',
  'sand_burn_in',
  'gas_generation',
  'gas_entrapment',
  'gas_porosity',
  'feeding_quality',
  'thermal_stress',
  'distortion',
  'crack_susceptibility',
  'sdas',
  'predicted_uts'
] as const;

export type ResultModeId = (typeof RESULT_MODE_IDS)[number];

export const RESULT_MODE_LABELS: Record<ResultModeId, string> = {
  shrinkage: 'Shrinkage / Porosity',
  porosity: 'Porosity',
  feed_modulus: 'Feed Modulus',
  hotspot: 'Hot Spots',
  solidification_time: 'Solidification Time',
  niyama_raw: 'Niyama (raw)',
  niyama: 'Niyama Risk',
  combined_feed_risk: 'Combined Feed Risk',
  fraction_liquid: 'Fraction Liquid',
  fill_time: 'Fill Time',
  fill_animation: 'Fill Animation',
  hot_tear: 'Hot Tear Risk',
  sand_burn_in: 'Sand Burn-In',
  gas_generation: 'Gas Generation',
  gas_entrapment: 'Gas Entrapment',
  gas_porosity: 'Gas Porosity Risk',
  feeding_quality: 'Blocked Feeding',
  thermal_stress: 'Thermal Stress',
  distortion: 'Distortion',
  crack_susceptibility: 'Crack Susceptibility',
  sdas: 'SDAS (Microstructure)',
  predicted_uts: 'Predicted UTS'
};

/** Display unit for the legend; empty string means a dimensionless 0..1 risk field. */
export const RESULT_MODE_UNITS: Partial<Record<ResultModeId, string>> = {
  feed_modulus: 'in',
  solidification_time: 's',
  fill_time: 's',
  thermal_stress: 'MPa',
  distortion: 'in',
  sdas: 'µm',
  predicted_uts: 'MPa'
};

/** Modes driven by the time scrubber instead of a threshold slider. */
export const TIME_MODES: readonly ResultModeId[] = ['fraction_liquid', 'fill_time', 'fill_animation'];

export function isTimeMode(mode: ResultModeId): boolean {
  return TIME_MODES.includes(mode);
}

/** Recommended envelope threshold (percent) per mode, mirroring the desktop defaults. */
export const DEFAULT_THRESHOLD: Record<ResultModeId, number> = {
  shrinkage: 30,
  porosity: 30,
  feed_modulus: 1,
  hotspot: 30,
  solidification_time: 50,
  niyama_raw: 10,
  niyama: 30,
  combined_feed_risk: 10,
  fraction_liquid: 50,
  fill_time: 0,
  fill_animation: 0,
  hot_tear: 10,
  sand_burn_in: 20,
  gas_generation: 10,
  gas_entrapment: 10,
  gas_porosity: 10,
  feeding_quality: 20,
  thermal_stress: 20,
  distortion: 20,
  crack_susceptibility: 20,
  sdas: 20,
  predicted_uts: 20
};

// ---------------------------------------------------------------------------
// Colormaps
// ---------------------------------------------------------------------------

export const COLORMAP_IDS = [
  'rainbow',
  'red_gradient',
  'viridis',
  'plasma',
  'coolwarm',
  'hot_cold',
  'defect_industry',
  'niyama',
  'feedmod'
] as const;

export type ColormapId = (typeof COLORMAP_IDS)[number];

// ---------------------------------------------------------------------------
// View state (owned by the shell, rendered by the viewport)
// ---------------------------------------------------------------------------

export type SectionAxisState = {
  enabled: boolean;
  /** 0..100 along the assembly's bounding box on this axis. */
  position: number;
  /** Keep the other side of the plane instead. */
  flip: boolean;
};

export type SectionState = {
  x: SectionAxisState;
  y: SectionAxisState;
  z: SectionAxisState;
  /** Paint the interior field on the cut faces. */
  caps: boolean;
};

/**
 * surface  = the field painted smoothly on the casting surfaces (and cut caps)
 * envelope = a smooth isosurface enclosing everything above the threshold
 * both     = translucent painted surface plus the envelope inside it
 */
export type ResultStyle = 'surface' | 'envelope' | 'both';

export type ResultViewState = {
  mode: ResultModeId;
  colormap: ColormapId;
  /** null = auto from the field range. */
  scaleMin: number | null;
  scaleMax: number | null;
  /** 0..100, envelope threshold for non-time modes. */
  threshold: number;
  /** 0..100 for time modes. */
  timePercent: number;
  playing: boolean;
  /** 0.5, 1, 2, 4 */
  speed: number;
  style: ResultStyle;
  showModel: boolean;
  showResults: boolean;
  /** 0..100. 100 is fully opaque. */
  modelOpacity: number;
  showLegend: boolean;
  /** Restrict painted results to CAST bodies (risers, chills, cores stay neutral). */
  castOnly: boolean;
  section: SectionState;
};

const defaultAxis: SectionAxisState = { enabled: false, position: 50, flip: false };

export const defaultResultViewState: ResultViewState = {
  mode: 'fraction_liquid',
  colormap: 'rainbow',
  scaleMin: null,
  scaleMax: null,
  threshold: DEFAULT_THRESHOLD.shrinkage,
  timePercent: 0,
  playing: false,
  speed: 1,
  style: 'surface',
  showModel: true,
  showResults: true,
  modelOpacity: 100,
  showLegend: true,
  castOnly: false,
  section: { x: { ...defaultAxis }, y: { ...defaultAxis }, z: { ...defaultAxis }, caps: true }
};

// ---------------------------------------------------------------------------
// Viewport interaction
// ---------------------------------------------------------------------------

export type CameraPreset = 'top' | 'bottom' | 'front' | 'back' | 'left' | 'right' | 'iso' | 'fit';

/**
 * select  = click selects a part (default)
 * probe   = click reports the field value under the cursor
 * measure = two clicks report a distance
 * pour    = click sets the pour (metal-entry) location
 */
export type ViewportTool = 'select' | 'probe' | 'measure' | 'pour';

export type ProbeReading = {
  point: [number, number, number];
  mode: ResultModeId;
  value: number | null;
  unit: string;
  partType: string | null;
};

export type MeasureResult = {
  start: [number, number, number];
  end: [number, number, number];
  distanceIn: number;
};

export type PourPick = {
  point: [number, number, number];
  normal: [number, number, number] | null;
  partId: string | null;
};

/**
 * One entry of the mask code table.
 *
 * The dense mask grid stores `part-type code + 1`, because 0 has to mean "no
 * metal in this cell". `code` is the byte as stored; `partTypeCode` is the
 * engine's own code.
 */
export type GridPartType = {
  code: number;
  partTypeCode: number;
  /** `cast`, `riser`, `chill`, ... matching the engine's PartType values. */
  id: string;
  label: string;
  /** Linear RGB, 0..1. */
  color: [number, number, number];
  voxelCount: number;
};

/** What the viewport learned about the loaded results; the shell uses it to populate menus and stats. */
export type GridMeta = {
  dims: [number, number, number];
  origin: [number, number, number];
  /** Uniform voxel edge length in inches. */
  spacing: number;
  /** Modes that actually have data for this run. */
  modes: ResultModeId[];
  ranges: Partial<Record<ResultModeId, { min: number; max: number; mean: number }>>;
  frames: {
    fraction_liquid?: { count: number; times_s: number[] };
    fill?: { count: number; times_s: number[] };
  };
  voxelCount: number;
  castVoxelCount: number;
  totalVolumeIn3: number;
  /** Summary statistics passthrough (defect volume, hotspot count, Chvorinov time, ...). */
  stats: Record<string, number | string>;
  /** True when the grids had to be built from a strided (downsampled) result, so counts under-report. */
  downsampled?: boolean;
  /** Modes whose field is a single non-zero value everywhere; selectable, with a degenerate legend. */
  constantModes?: ResultModeId[];

  // --- Added by the viewport; optional so the shell can ignore them. ---

  /**
   * Legend unit per mode as the server reports it. Prefer this over the
   * static `RESULT_MODE_UNITS` table when it is present: it is the unit the
   * values in `ranges` are actually in.
   */
  units?: Partial<Record<ResultModeId, string>>;
  /** Mask code table for this run, one entry per part type present. */
  partTypes?: GridPartType[];
  /**
   * Modes the run produced but that carry no signal at all (a constant-zero
   * field). Absent from `modes`, listed here so the shell can say "no data"
   * rather than silently omitting a menu entry the user expected.
   */
  emptyModes?: ResultModeId[];
  /** `dims[0] * dims[1] * dims[2]`; the number of cells in each dense grid. */
  cellCount?: number;
};

/** Imperative operations the shell can invoke through a ref. */
export type ViewportHandle = {
  screenshot(options?: { width?: number; height?: number }): Promise<Blob>;
  setCamera(preset: CameraPreset): void;
  clearMeasurements(): void;
  /** Volume in cubic inches enclosed by the current envelope threshold, or null if unavailable. */
  defectVolumeIn3(): number | null;
};

export type ViewportProps = {
  projectId: string | null;
  parts: ProjectPart[];
  selectedPartId: string | null;
  onSelectPart: (partId: string | null) => void;

  /** True once the server reports completed results for `projectId`. */
  resultsAvailable: boolean;
  /** Bump to force a reload of results (for example after a new run completes). */
  resultsVersion: number;

  view: ResultViewState;
  /** The viewport may request state changes (playback advancing, auto scale bounds). */
  onViewChange: (patch: Partial<ResultViewState>) => void;

  ortho: boolean;
  tool: ViewportTool;
  onProbe: (reading: ProbeReading | null) => void;
  onMeasure: (result: MeasureResult) => void;

  pourLocation: PourLocationPayload | null;
  onPourPicked: (pick: PourPick) => void;

  onGridMeta: (meta: GridMeta | null) => void;
  /** Short loading / progress messages, or null when idle. */
  onStatus: (message: string | null) => void;
  onError: (message: string) => void;

  className?: string;
};

export type ColorBarProps = {
  view: ResultViewState;
  meta: GridMeta | null;
  orientation?: 'vertical' | 'horizontal';
  /** Present when the bar is interactive (dragging the end labels edits the scale bounds). */
  onScaleChange?: (min: number | null, max: number | null) => void;
};
