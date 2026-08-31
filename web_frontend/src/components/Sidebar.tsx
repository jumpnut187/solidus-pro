import { Camera, Pause, Play } from 'lucide-react';

import type { ProjectPart, ProjectSettings, ResultFieldsPayload, ResultPayload } from '../lib/api';
import { resultModes } from '../lib/defaults';
import { MODE_CONTROL, type UiPrefs } from '../lib/uiPrefs';

export type DrawerId = 'parts' | 'material' | 'mesh' | 'analysis' | 'performance' | 'solidification';

type Bounds = { min: { x: number; y: number; z: number }; max: { x: number; y: number; z: number } };

type Props = {
  parts: ProjectPart[];
  settings: ProjectSettings;
  resultsReady: boolean;
  resultFields: ResultFieldsPayload | null;
  results: ResultPayload['data'] | null;
  ui: UiPrefs;
  patchUi: (patch: Partial<UiPrefs>) => void;
  isRunning: boolean;
  progress: number;
  status: string;
  statusWarning: boolean;
  activeResultCount: number | null;
  isPlayingTimeline: boolean;
  setIsPlayingTimeline: (playing: boolean) => void;
  onRun: () => void;
  onStop: () => void;
  onOpenDrawer: (drawer: DrawerId) => void;
  setCameraCommand: (command: string | null) => void;
};

function combinedBounds(parts: ProjectPart[]): Bounds | null {
  if (!parts.length) return null;
  const bounds: Bounds = {
    min: { x: Infinity, y: Infinity, z: Infinity },
    max: { x: -Infinity, y: -Infinity, z: -Infinity },
  };
  for (const part of parts) {
    const b = part.meshData?.bounds;
    if (!b) continue;
    bounds.min.x = Math.min(bounds.min.x, b.x[0]);
    bounds.min.y = Math.min(bounds.min.y, b.y[0]);
    bounds.min.z = Math.min(bounds.min.z, b.z[0]);
    bounds.max.x = Math.max(bounds.max.x, b.x[1]);
    bounds.max.y = Math.max(bounds.max.y, b.y[1]);
    bounds.max.z = Math.max(bounds.max.z, b.z[1]);
  }
  if (!Number.isFinite(bounds.min.x)) return null;
  return bounds;
}

export function Sidebar({
  parts,
  settings,
  resultsReady,
  resultFields,
  results,
  ui,
  patchUi,
  isRunning,
  progress,
  status,
  statusWarning,
  activeResultCount,
  isPlayingTimeline,
  setIsPlayingTimeline,
  onRun,
  onStop,
  onOpenDrawer,
  setCameraCommand,
}: Props) {
  const { section, flSpeed, resultMode, threshold, timePercent } = ui;
  const hasParts = parts.length > 0;
  const control = MODE_CONTROL[resultMode];
  const isFractionLiquid = resultMode === 'fraction_liquid';
  const isFillTime = resultMode === 'fill_time';
  // Both modes reuse the same timePercent scrubber instead of the threshold
  // slider -- fraction_liquid sweeps a solidification snapshot, fill_time
  // sweeps mold-filling arrival time.
  const isTimeSliced = isFractionLiquid || isFillTime;
  const bounds = combinedBounds(parts);

  function sectionPositionLabel(): string {
    if (!bounds) return '';
    const extent = bounds.max[section.axis] - bounds.min[section.axis];
    const fraction = section.position / 100;
    const value = bounds.min[section.axis] + extent * fraction;
    return `${section.position}% (${value.toFixed(2)}")`;
  }

  const perf = settings.performance_options ?? {};
  const performanceSummary =
    `Incremental: ${perf.use_incremental ? 'On' : 'Off'} | ` +
    `Chunked: ${perf.use_chunked_voxelization ? 'On' : 'Off'} | Mode: ${String(perf.part_assignment_mode ?? 'auto')}`;

  const sp = settings.solidification_params;
  const solidificationSummary =
    `Shrink scalar: ${Number(sp?.feeding?.shrink_scalar ?? 1).toFixed(2)} | ` +
    `Restraint: ${Number(sp?.hot_tear?.mold_restraint_index ?? 0.5).toFixed(2)}`;

  const analysisSummary = (() => {
    const ao = settings.analysis_options ?? {};
    const map: Array<[string, string]> = [
      ['shrinkage', 'Shrink'],
      ['hotspot', 'Hot Spots'],
      ['niyama', 'Niyama'],
      ['fraction_liquid', 'Frac. Liq.'],
      ['hot_tear', 'Hot Tear'],
      ['sand_burn_in', 'Burn-In'],
      ['feeding_path', 'Feeding'],
      ['stress_analysis', 'Stress'],
      ['microstructure', 'SDAS'],
    ];
    const enabled = map
      .filter(([key]) => Boolean(ao[key as keyof typeof ao]))
      .map(([, label]) => label);
    if (!enabled.length) return 'None enabled';
    return enabled.slice(0, 3).join(', ') + (enabled.length > 3 ? ` +${enabled.length - 3}` : '');
  })();

  return (
    <aside className="sidebar">
      <header className="sidebar-header">
        <p className="sidebar-title">Solidus Pro</p>
        <p className="sidebar-subtitle">SOLIDIFICATION SIMULATOR</p>
      </header>
      <div className="sidebar-scroll">
        <section className="group">
          <div className="group-title">Settings</div>
          <div className="group-body">
            <button className="setting-btn" onClick={() => onOpenDrawer('parts')}>
              <span className="setting-icon">🔧</span> Manage Parts…
            </button>
            <p className="setting-summary">{hasParts ? `${parts.length} part(s) loaded` : 'No parts loaded'}</p>

            <button className="setting-btn" onClick={() => onOpenDrawer('material')}>
              <span className="setting-icon">⚙</span> Material Properties…
            </button>
            <p className="setting-summary">Alloy: {settings.alloy}</p>

            <button className="setting-btn" onClick={() => onOpenDrawer('mesh')}>
              <span className="setting-icon">▦</span> Mesh Parameters…
            </button>
            <p className="setting-summary">
              Voxel: {settings.voxel_size.toFixed(3)}", Ref: {settings.refinement_level}
            </p>

            <button className="setting-btn" onClick={() => onOpenDrawer('analysis')}>
              <span className="setting-icon">📊</span> Analysis Options…
            </button>
            <p className="setting-summary">Analyses: {analysisSummary}</p>

            <button className="setting-btn" onClick={() => onOpenDrawer('performance')}>
              <span className="setting-icon">⚡</span> Performance Options…
            </button>
            <p className="setting-summary">{performanceSummary}</p>

            <button className="setting-btn" onClick={() => onOpenDrawer('solidification')}>
              <span className="setting-icon">✦</span> Solidification Params…
            </button>
            <p className="setting-summary">{solidificationSummary}</p>
          </div>
        </section>

        <section className="group">
          <div className="group-title">Simulation</div>
          <div className="group-body">
            <button className="run-button" disabled={!hasParts || isRunning} onClick={onRun}>
              ▶ Run Simulation
            </button>
            <button className="stop-button" disabled={!isRunning} onClick={onStop}>
              ■ Stop Simulation
            </button>
            <div className="progress-row">
              <div className="progress">
                <span style={{ width: `${progress}%` }} />
              </div>
              <span className="progress-text">{progress}%</span>
            </div>
            <p className={statusWarning ? 'status status-warning' : 'status'}>{status}</p>
          </div>
        </section>

        <section className="group">
          <div className="group-title">Visualization</div>
          <div className="group-body">
            <div className="vis-row">
              <label className="check">
                <input
                  type="checkbox"
                  checked={ui.showModel}
                  onChange={(e) => patchUi({ showModel: e.target.checked })}
                />
                Show Model
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={ui.showResults}
                  onChange={(e) => patchUi({ showResults: e.target.checked })}
                />
                Show Results
              </label>
            </div>

            <div className="vis-row">
              <span>Result Type:</span>
              <select
                className="vis-select"
                value={resultMode}
                disabled={!resultsReady}
                onChange={(e) => {
                  const next = e.target.value;
                  patchUi({
                    resultMode: next,
                    threshold: MODE_CONTROL[next]?.defaultThreshold ?? threshold,
                  });
                }}
              >
                {resultModes.map((mode) => (
                  <option
                    key={mode.id}
                    value={mode.id}
                    disabled={resultFields ? !resultFields.modes.includes(mode.id) : false}
                  >
                    {mode.label}
                  </option>
                ))}
              </select>
            </div>

            {!isTimeSliced && control && (
              <div className="threshold-row">
                <label>{control.thresholdLabel}:</label>
                <input
                  type="range"
                  min="1"
                  max="100"
                  value={threshold}
                  disabled={!resultsReady}
                  onChange={(e) => patchUi({ threshold: Number(e.target.value) })}
                />
                <span className="threshold-value">{threshold}%</span>
              </div>
            )}

            {isTimeSliced && (
              <div className="timeline">
                <div className="timeline-labels">
                  <span>{isFillTime ? 'Empty' : 'Liquid'}</span>
                  <span>{isFillTime ? 'Filled' : 'Solid'}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={timePercent}
                  disabled={!resultsReady}
                  onChange={(e) => {
                    patchUi({ timePercent: Number(e.target.value) });
                  }}
                />
                <div className="timeline-controls">
                  <button
                    className="mini-button"
                    disabled={!resultsReady}
                    onClick={() => {
                      setIsPlayingTimeline(false);
                      patchUi({ timePercent: 0 });
                    }}
                  >
                    {isFillTime ? 'Start of Pour' : 'Liquidus'}
                  </button>
                  {/* Fill Time has no autoplay wired up (App.tsx's playback
                      timer is gated to fraction_liquid) -- dragging the
                      scrubber is the animation for this mode. Showing a
                      Play button here would just do nothing. */}
                  {isFractionLiquid && (
                    <button
                      className="mini-button"
                      disabled={!resultsReady}
                      onClick={() => {
                        const playing = !isPlayingTimeline;
                        setIsPlayingTimeline(playing);
                        patchUi({ timePercent: playing && timePercent >= 100 ? 0 : timePercent });
                      }}
                    >
                      {isPlayingTimeline ? <Pause size={13} /> : <Play size={13} />}
                      {isPlayingTimeline ? 'Pause' : 'Play'}
                    </button>
                  )}
                  {isFractionLiquid && (
                    <label>
                      Speed
                      <select
                        value={flSpeed}
                        disabled={!resultsReady}
                        onChange={(e) => patchUi({ flSpeed: Number(e.target.value) })}
                      >
                        <option value={0.5}>0.5x</option>
                        <option value={1}>1x</option>
                        <option value={2}>2x</option>
                        <option value={4}>4x</option>
                      </select>
                    </label>
                  )}
                  <span className="timeline-time">{timePercent}%</span>
                </div>
              </div>
            )}

            <div className="vis-row">
              <span>Model Opacity: {ui.modelOpacity}%</span>
            </div>
            <input
              type="range"
              min="8"
              max="80"
              value={ui.modelOpacity}
              onChange={(e) => patchUi({ modelOpacity: Number(e.target.value) })}
            />

            <div className="stats-row">
              <span>
                Voxels <strong>{typeof results?.voxel_count === 'number' ? results.voxel_count.toLocaleString() : '--'}</strong>
              </span>
              <span>
                Volume <strong>{typeof results?.total_volume === 'number' ? `${results.total_volume.toFixed(2)} in3` : '--'}</strong>
              </span>
              <span>
                Active <strong>{activeResultCount?.toLocaleString() ?? '--'}</strong>
              </span>
            </div>
          </div>
        </section>

        <section className="group">
          <div className="group-title">Section View</div>
          <div className="group-body">
            <label className="check">
              <input
                type="checkbox"
                checked={section.enabled}
                onChange={(e) => patchUi({ section: { ...section, enabled: e.target.checked } })}
              />
              Enable Section
            </label>
            <fieldset className="fieldset-body" disabled={!hasParts || !section.enabled}>
              <div className="radio-row">
                <span>Mode:</span>
                <label>
                  <input
                    type="radio"
                    name="section-mode"
                    checked={section.mode === 'single'}
                    onChange={() => patchUi({ section: { ...section, mode: 'single' } })}
                  />
                  Single Plane
                </label>
                <label>
                  <input
                    type="radio"
                    name="section-mode"
                    checked={section.mode === 'slab'}
                    onChange={() => patchUi({ section: { ...section, mode: 'slab' } })}
                  />
                  Slab
                </label>
              </div>
              <div className="vis-row">
                <span>Axis:</span>
                <select
                  className="vis-select"
                  value={section.axis}
                  onChange={(e) =>
                    patchUi({ section: { ...section, axis: e.target.value as 'x' | 'y' | 'z' } })
                  }
                >
                  <option value="x">X</option>
                  <option value="y">Y</option>
                  <option value="z">Z</option>
                </select>
              </div>
              <div className="slider-row">
                <span>Position:</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={section.position}
                  onChange={(e) =>
                    patchUi({ section: { ...section, position: Number(e.target.value) } })
                  }
                />
                <span className="slider-value">{sectionPositionLabel()}</span>
              </div>
              {section.mode === 'slab' && (
                <div className="slider-row">
                  <span>Thickness:</span>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={section.thickness}
                    onChange={(e) =>
                      patchUi({ section: { ...section, thickness: Number(e.target.value) } })
                    }
                  />
                  <span className="slider-value">{section.thickness}%</span>
                </div>
              )}
              <label className="check">
                <input
                  type="checkbox"
                  checked={section.invert}
                  onChange={(e) =>
                    patchUi({ section: { ...section, invert: e.target.checked } })
                  }
                />
                Invert Cut
              </label>
            </fieldset>
          </div>
        </section>

        <section className="group">
          <div className="group-title">Camera Views</div>
          <div className="group-body">
            <div className="camera-grid">
              <button className="camera-btn col-2" onClick={() => setCameraCommand('top')}>
                Top
              </button>
              <button className="camera-btn col-1" onClick={() => setCameraCommand('left')}>
                Left
              </button>
              <button className="camera-btn col-3" onClick={() => setCameraCommand('right')}>
                Right
              </button>
              <button className="camera-btn col-2" onClick={() => setCameraCommand('front')}>
                Front
              </button>
              <button className="camera-btn col-2" onClick={() => setCameraCommand('bottom')}>
                Bottom
              </button>
              <button className="camera-btn col-2" onClick={() => setCameraCommand('iso')}>
                <Camera size={12} /> Iso
              </button>
            </div>
            <label className="check">
              <input
                type="checkbox"
                checked={ui.ortho}
                onChange={(e) => patchUi({ ortho: e.target.checked })}
              />
              Orthographic Projection
            </label>
          </div>
        </section>
      </div>
    </aside>
  );
}
