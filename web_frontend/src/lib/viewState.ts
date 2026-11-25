import type { PourLocationPayload, ProjectPart } from './api';
import resultHelp from '../../../core/result_help.json';
import type { ModelingEvent, ModelingViewState, Vec3 } from './modeling/types';

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
  shrinkage: 'Shrink',
  porosity: 'Microporosity',
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

export const RESULT_MODE_HELP: Record<ResultModeId, string> = resultHelp;

export const RESULT_MODE_UNITS: Partial<Record<ResultModeId, string>> = {
  feed_modulus: 'in',
  solidification_time: 's',
  fill_time: 's',
  thermal_stress: 'MPa',
  distortion: 'in',
  sdas: 'µm',
  predicted_uts: 'MPa'
};

export const TIME_MODES: readonly ResultModeId[] = ['fraction_liquid', 'fill_time', 'fill_animation'];

export const SWEEP_MODES: readonly ResultModeId[] = ['solidification_time'];

export function isSweepMode(mode: ResultModeId): boolean {
  return SWEEP_MODES.includes(mode);
}

export function isTimeMode(mode: ResultModeId): boolean {
  return TIME_MODES.includes(mode);
}

export const DEFAULT_THRESHOLD: Record<ResultModeId, number> = {
  shrinkage: 70,
  porosity: 70,
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

export const COLORMAP_IDS = [
  'rainbow',
  'molten',
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

export type SectionAxisState = {
  enabled: boolean;
  position: number;
  flip: boolean;
};

export type SectionState = {
  x: SectionAxisState;
  y: SectionAxisState;
  z: SectionAxisState;
  caps: boolean;
};

export type ResultStyle = 'surface' | 'envelope' | 'both';

export type ResultViewState = {
  mode: ResultModeId;
  colormap: ColormapId;
  scaleMin: number | null;
  scaleMax: number | null;
  threshold: number;
  timePercent: number;
  playing: boolean;
  speed: number;
  style: ResultStyle;
  showModel: boolean;
  showResults: boolean;
  modelOpacity: number;
  showLegend: boolean;
  castOnly: boolean;
  showPartingLine: boolean;
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
  style: 'envelope',
  showModel: true,
  showResults: true,
  modelOpacity: 100,
  showLegend: true,
  castOnly: false,
  showPartingLine: false,
  section: { x: { ...defaultAxis }, y: { ...defaultAxis }, z: { ...defaultAxis }, caps: true }
};

export type CameraPreset = 'top' | 'bottom' | 'front' | 'back' | 'left' | 'right' | 'iso' | 'fit';

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

export type OrientPickMode = 'faceDown' | 'faceUp' | 'pickHeight';

export type OrientPick = {
  mode: OrientPickMode;
  partId: string | null;
  point: [number, number, number];
  normal: [number, number, number] | null;
};

export type OrientViewState = {
  active: boolean;
  pick: OrientPickMode | null;
  parting: { z: number; visible: boolean } | null;
};

export type GridPartType = {
  code: number;
  partTypeCode: number;
  id: string;
  label: string;
  color: [number, number, number];
  voxelCount: number;
};

export type GridMeta = {
  dims: [number, number, number];
  origin: [number, number, number];
  spacing: number;
  modes: ResultModeId[];
  ranges: Partial<Record<ResultModeId, { min: number; max: number; mean: number }>>;
  frames: {
    fraction_liquid?: { count: number; times_s: number[] };
    fill?: { count: number; times_s: number[] };
  };
  voxelCount: number;
  castVoxelCount: number;
  totalVolumeIn3: number;
  stats: Record<string, number | string>;
  downsampled?: boolean;
  constantModes?: ResultModeId[];

  units?: Partial<Record<ResultModeId, string>>;
  partTypes?: GridPartType[];
  emptyModes?: ResultModeId[];
  cellCount?: number;
};

export type ViewportHandle = {
  screenshot(options?: { width?: number; height?: number }): Promise<Blob>;
  setCamera(preset: CameraPreset): void;
  clearMeasurements(): void;
  defectVolumeIn3(): number | null;
  normalTo(frame: { normal: Vec3; v: Vec3; focus: Vec3 }): void;
};

export type ViewportProps = {
  projectId: string | null;
  parts: ProjectPart[];
  selectedPartId: string | null;
  onSelectPart: (partId: string | null) => void;

  resultsAvailable: boolean;
  resultsVersion: number;

  view: ResultViewState;
  onViewChange: (patch: Partial<ResultViewState>) => void;

  ortho: boolean;
  tool: ViewportTool;
  onProbe: (reading: ProbeReading | null) => void;
  onMeasure: (result: MeasureResult) => void;

  pourLocation: PourLocationPayload | null;
  onPourPicked: (pick: PourPick) => void;

  onGridMeta: (meta: GridMeta | null) => void;
  modeling?: ModelingViewState | null;
  onModelingEvent?: (event: ModelingEvent) => void;
  orient?: OrientViewState | null;
  onOrientPick?: (pick: OrientPick) => void;
  onStatus: (message: string | null) => void;
  onError: (message: string) => void;

  className?: string;
};

export type ColorBarProps = {
  view: ResultViewState;
  meta: GridMeta | null;
  orientation?: 'vertical' | 'horizontal';
  onScaleChange?: (min: number | null, max: number | null) => void;
};
