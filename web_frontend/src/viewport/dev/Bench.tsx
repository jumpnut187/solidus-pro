/**
 * Standalone bench for the results viewport.
 *
 * Served at `/dev.html`. It owns a `ResultViewState` and a control strip and
 * nothing else, which lets the renderer be exercised, screenshotted and
 * reviewed without waiting on the application shell -- and without either
 * workstream editing the other's `App.tsx`.
 *
 * Query parameters: `?project=<id>` picks a run, `?mode=`, `?time=`,
 * `?colormap=`, `?opacity=`, `?ui=0` preset the view. The headless screenshot
 * harness drives it entirely through those.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { ProjectPart } from '../../lib/api';
import {
  COLORMAP_IDS,
  RESULT_MODE_LABELS,
  defaultResultViewState,
  isTimeMode,
  type CameraPreset,
  type ColormapId,
  type GridMeta,
  type ResultModeId,
  type ResultViewState,
  type ViewportHandle
} from '../../lib/viewState';
import { ColorBar } from '../ColorBar';
import { ResultsViewport } from '../ResultsViewport';
import { parseBinaryStl } from './stl';
import './bench.css';

type ProjectSummary = { id: string; name: string; status: string };

const params = new URLSearchParams(window.location.search);
const showUi = params.get('ui') !== '0';

function initialView(): ResultViewState {
  const view = { ...defaultResultViewState };
  const mode = params.get('mode');
  if (mode) view.mode = mode as ResultModeId;
  const time = params.get('time');
  if (time) view.timePercent = Number.parseFloat(time);
  const colormap = params.get('colormap');
  if (colormap) view.colormap = colormap as ColormapId;
  const opacity = params.get('opacity');
  if (opacity) view.modelOpacity = Number.parseFloat(opacity);
  return view;
}

async function loadParts(projectId: string): Promise<ProjectPart[]> {
  const response = await fetch(`/api/projects/${projectId}`);
  if (!response.ok) throw new Error(`Project ${projectId}: ${response.status}`);
  const body = (await response.json()) as { parts: ProjectPart[] };
  const parts = body.parts ?? [];

  // Part rows carry a server-side path, not geometry. Fetch each STL through
  // the uploads route and rebuild the triangles the viewport expects.
  return await Promise.all(
    parts.map(async (part) => {
      const filename = part.filepath.split('/').pop() ?? '';
      try {
        const stl = await fetch(`/api/uploads/${encodeURIComponent(filename)}`);
        if (!stl.ok) return part;
        return { ...part, meshData: parseBinaryStl(await stl.arrayBuffer()) };
      } catch {
        return part;
      }
    })
  );
}

export function Bench() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [projectId, setProjectId] = useState<string | null>(params.get('project'));
  const [parts, setParts] = useState<ProjectPart[]>([]);
  const [selectedPartId, setSelectedPartId] = useState<string | null>(null);
  const [meta, setMeta] = useState<GridMeta | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<ResultViewState>(initialView);
  const [ortho, setOrtho] = useState(false);
  const handleRef = useRef<ViewportHandle | null>(null);

  useEffect(() => {
    fetch('/api/projects')
      .then((response) => response.json())
      .then((rows: ProjectSummary[]) => {
        setProjects(rows);
        if (!projectId) {
          const completed = rows.find((row) => row.status === 'completed');
          if (completed) setProjectId(completed.id);
        }
      })
      .catch((problem) => setError(String(problem)));
    // Runs once; the project id is seeded from the URL or from this response.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!projectId) return;
    setParts([]);
    loadParts(projectId).then(setParts).catch((problem) => setError(String(problem)));
  }, [projectId]);

  const patchView = useCallback((patch: Partial<ResultViewState>) => {
    setView((current) => ({ ...current, ...patch }));
  }, []);

  const availableModes = useMemo(() => meta?.modes ?? [], [meta]);

  // The bench exists to exercise the viewport, and `ViewportHandle` is half
  // of what the shell will drive it with. Publishing the ref lets the
  // screenshot harness call screenshot(), setCamera() and defectVolumeIn3()
  // the way the shell will.
  useEffect(() => {
    const globals = window as unknown as { __viewport?: ViewportHandle | null };
    globals.__viewport = handleRef.current;
    return () => {
      globals.__viewport = null;
    };
  });

  return (
    <div className="bench">
      {showUi ? (
        <header className="bench-bar">
          <select value={projectId ?? ''} onChange={(event) => setProjectId(event.target.value)}>
            {projects.map((project) => (
              <option key={project.id} value={project.id}>
                {project.name} [{project.status}]
              </option>
            ))}
          </select>

          <select
            value={view.mode}
            onChange={(event) => patchView({ mode: event.target.value as ResultModeId })}
          >
            {availableModes.map((mode) => (
              <option key={mode} value={mode}>
                {RESULT_MODE_LABELS[mode]}
              </option>
            ))}
          </select>

          <select
            value={view.colormap}
            onChange={(event) => patchView({ colormap: event.target.value as ColormapId })}
          >
            {COLORMAP_IDS.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>

          <label>
            Opacity {Math.round(view.modelOpacity)}%
            <input
              type="range"
              min={0}
              max={100}
              value={view.modelOpacity}
              onChange={(event) => patchView({ modelOpacity: Number(event.target.value) })}
            />
          </label>

          {isTimeMode(view.mode) ? (
            <>
              <button type="button" onClick={() => patchView({ playing: !view.playing })}>
                {view.playing ? 'Pause' : 'Play'}
              </button>
              <label>
                Time {view.timePercent.toFixed(0)}%
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={0.5}
                  value={view.timePercent}
                  onChange={(event) => patchView({ timePercent: Number(event.target.value) })}
                />
              </label>
              <select
                value={view.speed}
                onChange={(event) => patchView({ speed: Number(event.target.value) })}
              >
                {[0.5, 1, 2, 4].map((speed) => (
                  <option key={speed} value={speed}>
                    {speed}x
                  </option>
                ))}
              </select>
            </>
          ) : null}

          <span className="bench-spacer" />

          {(['iso', 'top', 'front', 'right', 'fit'] as CameraPreset[]).map((preset) => (
            <button key={preset} type="button" onClick={() => handleRef.current?.setCamera(preset)}>
              {preset}
            </button>
          ))}
          <label className="bench-check">
            <input type="checkbox" checked={ortho} onChange={(event) => setOrtho(event.target.checked)} />
            ortho
          </label>
          <label className="bench-check">
            <input
              type="checkbox"
              checked={view.castOnly}
              onChange={(event) => patchView({ castOnly: event.target.checked })}
            />
            cast only
          </label>
          <label className="bench-check">
            <input
              type="checkbox"
              checked={view.showResults}
              onChange={(event) => patchView({ showResults: event.target.checked })}
            />
            results
          </label>
        </header>
      ) : null}

      <main className="bench-stage">
        <ResultsViewport
          ref={handleRef}
          projectId={projectId}
          parts={parts}
          selectedPartId={selectedPartId}
          onSelectPart={setSelectedPartId}
          resultsAvailable={Boolean(projectId)}
          resultsVersion={0}
          view={view}
          onViewChange={patchView}
          ortho={ortho}
          tool="select"
          onProbe={() => undefined}
          onMeasure={() => undefined}
          pourLocation={null}
          onPourPicked={() => undefined}
          onGridMeta={setMeta}
          onStatus={setStatus}
          onError={setError}
        />

        {view.showLegend ? (
          <div className="bench-legend">
            <ColorBar
              view={view}
              meta={meta}
              orientation="vertical"
              onScaleChange={(min, max) => patchView({ scaleMin: min, scaleMax: max })}
            />
          </div>
        ) : null}

        {/* The screenshot harness waits for this to disappear. */}
        {status ? (
          <p className="bench-status" data-bench-status="busy">
            {status}
          </p>
        ) : (
          <span hidden data-bench-status="idle" />
        )}
        {error ? <p className="bench-error">{error}</p> : null}
      </main>
    </div>
  );
}
