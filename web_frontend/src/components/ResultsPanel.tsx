import { ChevronLeft, ChevronRight, Crosshair, FlipHorizontal, MousePointer2, Ruler, Target } from 'lucide-react';

import { Group } from './SetupPanel';
import { COLORMAPS, gradientCss } from '../lib/colormaps';
import { formatNumber } from '../lib/format';
import {
  COLORMAP_IDS,
  DEFAULT_THRESHOLD,
  RESULT_MODE_LABELS,
  RESULT_MODE_UNITS,
  isTimeMode,
  type CameraPreset,
  type ColormapId,
  type GridMeta,
  type ResultModeId,
  type ResultStyle,
  type ResultViewState,
  type SectionAxisState,
  type SectionState,
  type ViewportTool
} from '../lib/viewState';

export type ResultsPanelProps = {
  collapsed: boolean;
  onToggleCollapsed: () => void;
  view: ResultViewState;
  onViewChange: (patch: Partial<ResultViewState>) => void;
  meta: GridMeta | null;
  resultsAvailable: boolean;
  ortho: boolean;
  onOrthoChange: (ortho: boolean) => void;
  tool: ViewportTool;
  onToolChange: (tool: ViewportTool) => void;
  onCamera: (preset: CameraPreset) => void;
};

/** Result modes grouped the way a foundry engineer looks for them. */
const MODE_GROUPS: { label: string; modes: ResultModeId[] }[] = [
  {
    label: 'Solidification',
    modes: ['fraction_liquid', 'solidification_time', 'hotspot', 'feed_modulus']
  },
  {
    label: 'Feeding and Defects',
    modes: ['shrinkage', 'porosity', 'niyama', 'niyama_raw', 'combined_feed_risk', 'feeding_quality', 'hot_tear']
  },
  {
    label: 'Filling',
    modes: ['fill_time', 'fill_animation', 'gas_generation', 'gas_entrapment', 'gas_porosity', 'sand_burn_in']
  },
  { label: 'Stress', modes: ['thermal_stress', 'distortion', 'crack_susceptibility'] },
  { label: 'Microstructure', modes: ['sdas', 'predicted_uts'] }
];

const CAMERA_PRESETS: { id: CameraPreset; label: string }[] = [
  { id: 'iso', label: 'Iso' },
  { id: 'top', label: 'Top' },
  { id: 'bottom', label: 'Bottom' },
  { id: 'front', label: 'Front' },
  { id: 'back', label: 'Back' },
  { id: 'left', label: 'Left' },
  { id: 'right', label: 'Right' },
  { id: 'fit', label: 'Fit' }
];

const TOOLS: { id: ViewportTool; label: string; Icon: typeof MousePointer2 }[] = [
  { id: 'select', label: 'Select', Icon: MousePointer2 },
  { id: 'probe', label: 'Probe', Icon: Target },
  { id: 'measure', label: 'Measure', Icon: Ruler },
  { id: 'pour', label: 'Pour', Icon: Crosshair }
];

export function ResultsPanel(props: ResultsPanelProps) {
  const {
    collapsed,
    onToggleCollapsed,
    view,
    onViewChange,
    meta,
    resultsAvailable,
    ortho,
    onOrthoChange,
    tool,
    onToolChange,
    onCamera
  } = props;

  if (collapsed) {
    return (
      <aside className="results-panel is-collapsed" aria-label="Results (collapsed)">
        <button
          type="button"
          className="btn btn-ghost btn-icon"
          onClick={onToggleCollapsed}
          aria-label="Expand results panel"
          title="Expand results panel"
        >
          <ChevronLeft size={15} />
        </button>
        <span className="results-collapsed-label">Results</span>
      </aside>
    );
  }

  return (
    <aside className="results-panel" aria-label="Results">
      <header className="panel-head">
        <h2 className="panel-title">Results</h2>
        <div className="spacer" />
        <button
          type="button"
          className="btn btn-ghost btn-icon"
          onClick={onToggleCollapsed}
          aria-label="Collapse results panel"
          title="Collapse results panel"
        >
          <ChevronRight size={15} />
        </button>
      </header>

      <div className="panel-body">
        {/* Field, threshold, scale and section all describe a result that does
            not exist yet, so they stay out of the way until a run completes.
            Everything below acts on the model itself and is useful straight
            away. */}
        {resultsAvailable ? (
          <FieldControls view={view} onViewChange={onViewChange} meta={meta} />
        ) : (
          <p className="results-empty">
            No results yet. Run a simulation and the field, threshold, scale and section controls appear here.
            The display, camera and tool controls below act on the model now.
          </p>
        )}

        <Group title="Display">
          <label className="check">
            <input
              type="checkbox"
              checked={view.showModel}
              onChange={(event) => onViewChange({ showModel: event.target.checked })}
            />
            <span className="check-body">
              <span>Show model</span>
            </span>
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={view.showResults}
              onChange={(event) => onViewChange({ showResults: event.target.checked })}
            />
            <span className="check-body">
              <span>Show results</span>
            </span>
          </label>
          <label className="field">
            <span className="field-label">
              Model opacity
              <span className="field-unit">{view.modelOpacity}%</span>
            </span>
            <input
              type="range"
              className="range"
              min={0}
              max={100}
              step={1}
              value={view.modelOpacity}
              onChange={(event) => onViewChange({ modelOpacity: Number(event.target.value) })}
            />
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={view.castOnly}
              onChange={(event) => onViewChange({ castOnly: event.target.checked })}
            />
            <span className="check-body">
              <span>Cast bodies only</span>
              <span className="check-note">Leave risers, chills and cores unpainted.</span>
            </span>
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={view.showLegend}
              onChange={(event) => onViewChange({ showLegend: event.target.checked })}
            />
            <span className="check-body">
              <span>Show color bar</span>
            </span>
          </label>
          <label className="check">
            <input type="checkbox" checked={ortho} onChange={(event) => onOrthoChange(event.target.checked)} />
            <span className="check-body">
              <span>Orthographic projection</span>
            </span>
          </label>
        </Group>

        <Group title="Camera">
          <div className="camera-grid">
            {CAMERA_PRESETS.map((preset) => (
              <button key={preset.id} type="button" className="btn btn-sm" onClick={() => onCamera(preset.id)}>
                {preset.label}
              </button>
            ))}
          </div>
        </Group>

        <Group title="Tools">
          <div className="tool-grid">
            {TOOLS.map(({ id, label, Icon }) => (
              <button
                key={id}
                type="button"
                className="tool-btn"
                aria-pressed={tool === id}
                onClick={() => onToolChange(tool === id ? 'select' : id)}
              >
                <Icon size={14} />
                {label}
              </button>
            ))}
          </div>
        </Group>
      </div>
    </aside>
  );
}

/** The controls that only mean something once a run has produced fields. */
function FieldControls({
  view,
  onViewChange,
  meta
}: {
  view: ResultViewState;
  onViewChange: (patch: Partial<ResultViewState>) => void;
  meta: GridMeta | null;
}) {
  const available = new Set(meta?.modes ?? []);
  const range = meta?.ranges[view.mode];
  const unit = RESULT_MODE_UNITS[view.mode] ?? '';
  const autoScale = view.scaleMin === null && view.scaleMax === null;
  const patchSection = (patch: Partial<SectionState>) =>
    onViewChange({ section: { ...view.section, ...patch } });

  return (
    <>
      <Group title="Field">
        <label className="field">
          <span className="visually-hidden">Result type</span>
          <select
            className="select"
            value={view.mode}
            onChange={(event) => {
              const mode = event.target.value as ResultModeId;
              onViewChange({
                mode,
                threshold: DEFAULT_THRESHOLD[mode],
                scaleMin: null,
                scaleMax: null,
                playing: false,
                timePercent: 0
              });
            }}
          >
            {MODE_GROUPS.map((group) => (
              <optgroup key={group.label} label={group.label}>
                {group.modes.map((mode) => (
                  <option key={mode} value={mode} disabled={!available.has(mode)}>
                    {RESULT_MODE_LABELS[mode]}
                    {available.has(mode) ? '' : ' — not in this run'}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>

        <div className="colormap-row">
          <span
            className="colormap-preview"
            style={{ background: gradientCss(view.colormap) }}
            aria-hidden="true"
          />
          <select
            className="select"
            value={view.colormap}
            aria-label="Colormap"
            onChange={(event) => onViewChange({ colormap: event.target.value as ColormapId })}
          >
            {COLORMAP_IDS.map((id) => (
              <option key={id} value={id}>
                {COLORMAPS[id].name}
              </option>
            ))}
          </select>
        </div>

        <div className="segmented segmented-block" role="group" aria-label="Result style">
          {(['surface', 'envelope', 'both'] as ResultStyle[]).map((style) => (
            <button
              key={style}
              type="button"
              aria-pressed={view.style === style}
              onClick={() => onViewChange({ style })}
            >
              {style === 'surface' ? 'Surface' : style === 'envelope' ? 'Envelope' : 'Both'}
            </button>
          ))}
        </div>
      </Group>

      {!isTimeMode(view.mode) ? (
        <Group title="Threshold">
          <div className="field">
            <span className="field-label">
              Envelope threshold
              <span className="field-unit">{view.threshold.toFixed(0)}%</span>
            </span>
            <div className="range-marked">
              <input
                type="range"
                className="range"
                min={1}
                max={100}
                step={1}
                value={view.threshold}
                aria-label="Envelope threshold"
                onChange={(event) => onViewChange({ threshold: Number(event.target.value) })}
              />
              <span
                className="range-tick"
                style={{ left: `${DEFAULT_THRESHOLD[view.mode]}%` }}
                title={`Recommended: ${DEFAULT_THRESHOLD[view.mode]}%`}
              />
            </div>
            <span className="field-hint">
              Recommended for {RESULT_MODE_LABELS[view.mode]}: {DEFAULT_THRESHOLD[view.mode]}%.
            </span>
          </div>
        </Group>
      ) : null}

      <Group title="Scale">
        <div className="scale-row">
          <label className="field">
            <span className="field-label">Min</span>
            <input
              type="number"
              className="input input-num"
              value={view.scaleMin ?? (range ? Number(range.min.toFixed(4)) : 0)}
              disabled={autoScale}
              onChange={(event) => onViewChange({ scaleMin: Number(event.target.value) })}
            />
          </label>
          <label className="field">
            <span className="field-label">Max</span>
            <input
              type="number"
              className="input input-num"
              value={view.scaleMax ?? (range ? Number(range.max.toFixed(4)) : 1)}
              disabled={autoScale}
              onChange={(event) => onViewChange({ scaleMax: Number(event.target.value) })}
            />
          </label>
          <label className="check" title="Take the scale bounds from the loaded field">
            <input
              type="checkbox"
              checked={autoScale}
              onChange={(event) =>
                onViewChange(
                  event.target.checked
                    ? { scaleMin: null, scaleMax: null }
                    : {
                        scaleMin: range ? Number(range.min.toFixed(4)) : 0,
                        scaleMax: range ? Number(range.max.toFixed(4)) : 1
                      }
                )
              }
            />
            <span className="check-body">
              <span>Auto</span>
            </span>
          </label>
        </div>
        {range ? (
          <span className="field-hint mono">
            Field range {formatNumber(range.min, 3)} to {formatNumber(range.max, 3)}
            {unit ? ` ${unit}` : ''} (mean {formatNumber(range.mean, 3)})
          </span>
        ) : null}
      </Group>

      <Group title="Section">
        {(['x', 'y', 'z'] as const).map((axis) => (
          <AxisRow
            key={axis}
            axis={axis}
            state={view.section[axis]}
            onChange={(patch) => patchSection({ [axis]: { ...view.section[axis], ...patch } })}
          />
        ))}
        <label className="check">
          <input
            type="checkbox"
            checked={view.section.caps}
            onChange={(event) => patchSection({ caps: event.target.checked })}
          />
          <span className="check-body">
            <span>Cap cut faces</span>
            <span className="check-note">Paint the interior field on the cut instead of leaving it hollow.</span>
          </span>
        </label>
      </Group>
    </>
  );
}

function AxisRow({
  axis,
  state,
  onChange
}: {
  axis: 'x' | 'y' | 'z';
  state: SectionAxisState;
  onChange: (patch: Partial<SectionAxisState>) => void;
}) {
  return (
    <div className="axis-row">
      <label className="check">
        <input
          type="checkbox"
          checked={state.enabled}
          onChange={(event) => onChange({ enabled: event.target.checked })}
        />
        <span className="check-body">
          <span>{axis.toUpperCase()}</span>
        </span>
      </label>
      <input
        type="range"
        className="range"
        min={0}
        max={100}
        step={1}
        value={state.position}
        disabled={!state.enabled}
        aria-label={`Section ${axis.toUpperCase()} position`}
        onChange={(event) => onChange({ position: Number(event.target.value) })}
      />
      <button
        type="button"
        className="btn btn-sm btn-icon"
        aria-pressed={state.flip}
        disabled={!state.enabled}
        title={`Keep the ${state.flip ? 'far' : 'near'} side of the ${axis.toUpperCase()} cut`}
        aria-label={`Flip ${axis.toUpperCase()} cut direction`}
        onClick={() => onChange({ flip: !state.flip })}
        style={
          state.flip && state.enabled
            ? { color: 'var(--accent-hover)', borderColor: 'var(--accent-border)' }
            : undefined
        }
      >
        <FlipHorizontal size={13} />
      </button>
    </div>
  );
}
