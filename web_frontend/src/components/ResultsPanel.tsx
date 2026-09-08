import { ChevronLeft, ChevronRight, FlipHorizontal, Maximize2, Ruler } from 'lucide-react';

import { CollapsibleGroup, Group } from './SetupPanel';
import { COLORMAPS, gradientCss } from '../lib/colormaps';
import { formatCount, formatNumber } from '../lib/format';
import { buildStatGroups } from '../lib/statLabels';
import {
  COLORMAP_IDS,
  DEFAULT_THRESHOLD,
  RESULT_MODE_LABELS,
  RESULT_MODE_UNITS,
  isTimeMode,
  type ColormapId,
  type GridMeta,
  type MeasureResult,
  type ResultModeId,
  type ResultStyle,
  type ResultViewState,
  type SectionAxisState,
  type SectionState
} from '../lib/viewState';
import type { ResultSectionId } from '../lib/uiPrefs';

export type ResultsPanelProps = {
  collapsed: boolean;
  onToggleCollapsed: () => void;
  sectionsCollapsed: Record<ResultSectionId, boolean>;
  onSectionToggle: (id: ResultSectionId, collapsed: boolean) => void;
  view: ResultViewState;
  onViewChange: (patch: Partial<ResultViewState>) => void;
  meta: GridMeta | null;
  resultsAvailable: boolean;
  measurement: MeasureResult | null;
  onClearMeasurements: () => void;
  /** Volume above the current threshold, from the viewport handle. */
  defectVolumeIn3: number | null;
  onRecomputeDefectVolume: () => void;
  onOpenStatistics: () => void;
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

export function ResultsPanel(props: ResultsPanelProps) {
  const {
    collapsed,
    onToggleCollapsed,
    sectionsCollapsed,
    onSectionToggle,
    view,
    onViewChange,
    meta,
    resultsAvailable,
    measurement,
    onClearMeasurements,
    defectVolumeIn3,
    onRecomputeDefectVolume,
    onOpenStatistics
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

  const available = new Set(meta?.modes ?? []);
  const emptyModes = new Set(meta?.emptyModes ?? []);
  const range = meta?.ranges[view.mode];
  const unit = meta?.units?.[view.mode] ?? RESULT_MODE_UNITS[view.mode] ?? '';
  const timeMode = isTimeMode(view.mode);
  const autoScale = view.scaleMin === null && view.scaleMax === null;
  const enabledAxes = (['x', 'y', 'z'] as const).filter((axis) => view.section[axis].enabled);
  const section = (id: ResultSectionId) => ({
    id,
    collapsed: sectionsCollapsed[id],
    onToggle: (next: boolean) => onSectionToggle(id, next)
  });
  const patchSection = (patch: Partial<SectionState>) =>
    onViewChange({ section: { ...view.section, ...patch } });

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
        {!resultsAvailable ? (
          <p className="results-empty">
            No results yet. Run a simulation and the field, threshold, scale and section controls appear here.
            Tools, views and projection are on the toolbar above the canvas.
          </p>
        ) : (
          <>
            <CollapsibleGroup title="Field" summary={RESULT_MODE_LABELS[view.mode]} {...section('field')}>
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
                          {available.has(mode) ? '' : emptyModes.has(mode) ? ' — no signal' : ' — not in this run'}
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
            </CollapsibleGroup>

            {!timeMode ? (
              <CollapsibleGroup
                title="Threshold"
                summary={`${view.threshold.toFixed(0)}%`}
                {...section('threshold')}
              >
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
                    {view.style === 'surface'
                      ? ' In Surface style it sets the defect volume below; switch to Envelope or Both to see it as a shape.'
                      : ' The envelope encloses everything at or above this value.'}
                  </span>
                </div>

                <div className="row">
                  <button type="button" className="btn btn-sm" onClick={onRecomputeDefectVolume}>
                    Calculate defect volume
                  </button>
                  {defectVolumeIn3 !== null ? (
                    <span className="chip chip-accent mono">{formatNumber(defectVolumeIn3, 3)} in³</span>
                  ) : null}
                </div>
                <span className="field-hint">
                  Volume of metal above the threshold, counted on the lattice — independent of the camera and
                  of what is hidden.
                </span>
              </CollapsibleGroup>
            ) : null}

            <CollapsibleGroup
              title="Scale"
              summary={autoScale ? 'Auto' : `${formatNumber(view.scaleMin ?? 0, 2)} to ${formatNumber(view.scaleMax ?? 1, 2)}`}
              {...section('scale')}
            >
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
            </CollapsibleGroup>

            <CollapsibleGroup
              title="Section"
              summary={enabledAxes.length ? enabledAxes.join(', ').toUpperCase() : 'Off'}
              {...section('section')}
            >
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
                  <span className="check-note">
                    Paint the interior field on the cut instead of leaving it hollow.
                  </span>
                </span>
              </label>
            </CollapsibleGroup>
          </>
        )}

        <CollapsibleGroup
          title="Display"
          summary={`${view.modelOpacity}% opacity`}
          {...section('display')}
        >
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
              aria-label="Model opacity"
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
        </CollapsibleGroup>

        <Group title="Measurement">
          {measurement ? (
            <div className="property-sheet">
              <div className="property-row">
                <span className="property-label">Distance</span>
                <span className="property-value">{formatNumber(measurement.distanceIn, 4)} in</span>
              </div>
              <div className="property-row">
                <span className="property-label">From</span>
                <span className="property-value">
                  {measurement.start.map((value) => value.toFixed(2)).join(', ')}
                </span>
              </div>
              <div className="property-row">
                <span className="property-label">To</span>
                <span className="property-value">
                  {measurement.end.map((value) => value.toFixed(2)).join(', ')}
                </span>
              </div>
            </div>
          ) : (
            <p className="field-hint">
              <Ruler size={11} style={{ verticalAlign: '-1px', marginRight: 4 }} />
              Pick the Measure tool on the canvas toolbar, then click two points on the model.
            </p>
          )}
          <button type="button" className="btn btn-sm" onClick={onClearMeasurements} disabled={!measurement}>
            Clear measurements
          </button>
        </Group>

        {resultsAvailable ? (
          <CollapsibleGroup
            title="Statistics"
            summary={meta ? `${formatCount(meta.voxelCount)} voxels` : ''}
            {...section('statistics')}
          >
            <StatisticsSummary meta={meta} />
            <button type="button" className="btn btn-sm btn-block" onClick={onOpenStatistics}>
              <Maximize2 size={12} />
              All statistics
            </button>
          </CollapsibleGroup>
        ) : null}
      </div>
    </aside>
  );
}

/** The first handful of statistics, enough to judge a run without a dialog. */
function StatisticsSummary({ meta }: { meta: GridMeta | null }) {
  if (!meta) return <p className="field-hint">No statistics reported for this run.</p>;
  const groups = buildStatGroups(meta.stats, {
    voxelCount: meta.voxelCount,
    castVoxelCount: meta.castVoxelCount,
    totalVolumeIn3: meta.totalVolumeIn3,
    spacing: meta.spacing
  });
  const rows = groups.flatMap((group) => group.rows).slice(0, 6);
  if (!rows.length) return <p className="field-hint">No statistics reported for this run.</p>;
  return (
    <div className="property-sheet">
      {rows.map((row) => (
        <div className="property-row" key={row.key}>
          <span className="property-label" title={row.hint}>
            {row.label}
          </span>
          <span className="property-value">{row.value}</span>
        </div>
      ))}
    </div>
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
