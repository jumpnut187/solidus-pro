import { useEffect, useRef, useState, type MutableRefObject, type ReactNode } from 'react';

import {
  DownsampledBanner,
  StaleResultsBanner,
  DropVeil,
  EmptyState,
  ProbeReadout,
  StatsChip,
  Toasts
} from './CanvasOverlays';
import { ModelingToolbar } from './ModelingToolbar';
import { ViewportToolbar } from './ViewportToolbar';
import type { ModelingController } from '../hooks/useModeling';
import { ColorBar } from '../viewport/ColorBar';
import { ResultsViewport } from '../viewport/ResultsViewport';
import type { PourLocationPayload, ProjectPart } from '../lib/api';
import type {
  CameraPreset,
  GridMeta,
  MeasureResult,
  OrientPick,
  OrientViewState,
  PourPick,
  ProbeReading,
  ResultViewState,
  ViewportHandle,
  ViewportTool
} from '../lib/viewState';
import type { Toast } from '../hooks/useToasts';

export type CanvasStageProps = {
  viewportRef: MutableRefObject<ViewportHandle | null>;
  projectId: string | null;
  parts: ProjectPart[];
  selectedPartId: string | null;
  onSelectPart: (partId: string | null) => void;
  resultsAvailable: boolean;
  resultsVersion: number;
  view: ResultViewState;
  onViewChange: (patch: Partial<ResultViewState>) => void;
  /** The parametric modeling session; its toolbar replaces the viewport's while one is open. */
  modeling: ModelingController;
  ortho: boolean;
  tool: ViewportTool;
  probe: ProbeReading | null;
  onProbe: (reading: ProbeReading | null) => void;
  onMeasure: (result: MeasureResult) => void;
  pourLocation: PourLocationPayload | null;
  onPourPicked: (pick: PourPick) => void;
  gridMeta: GridMeta | null;
  onGridMeta: (meta: GridMeta | null) => void;
  status: string | null;
  onStatus: (message: string | null) => void;
  onError: (message: string) => void;
  onAddFiles: (files: File[]) => void;
  /** A project or upload is in flight, so an empty canvas is not yet "empty". */
  busy: boolean;
  hasParts: boolean;
  onOrthoChange: (ortho: boolean) => void;
  onToolChange: (tool: ViewportTool) => void;
  onCamera: (preset: CameraPreset) => void;
  onClearMeasurements: () => void;
  measurement: MeasureResult | null;
  toasts: Toast[];
  onDismissToast: (id: number) => void;
  /** The loaded run was stored strided; warn until the user dismisses it. */
  showDownsampledWarning: boolean;
  onDismissDownsampledWarning: () => void;
  /** What changed since the loaded run, or null when the results match the parts. */
  staleResults: string | null;
  onRunAgain: () => void;
  running: boolean;
  onRequestAddParts: () => void;
  /** Hands the shell a way to open this stage's file picker from a shortcut. */
  registerAddParts: (open: () => void) => void;
  /** The job log drawer, anchored to the foot of the canvas. */
  log?: ReactNode;
  /** The Orient & Parting Line tool's view state, or null when unused. */
  orient: OrientViewState | null;
  onOrientPick: (pick: OrientPick) => void;
  /** One-click prompt shown over the canvas, e.g. to orient a fresh casting; null for none. */
  prompt?: ReactNode;
};

/**
 * The canvas well: the viewport plus everything painted over it.
 *
 * With no parts loaded the whole canvas is a drop target, not just a small zone
 * inside a panel — dropping an STL onto the empty 3D view is what people try
 * first.
 */
export function CanvasStage(props: CanvasStageProps) {
  const {
    viewportRef,
    projectId,
    parts,
    selectedPartId,
    onSelectPart,
    resultsAvailable,
    resultsVersion,
    view,
    onViewChange,
    modeling,
    ortho,
    tool,
    probe,
    onProbe,
    onMeasure,
    pourLocation,
    onPourPicked,
    gridMeta,
    onGridMeta,
    status,
    onStatus,
    onError,
    onAddFiles,
    busy,
    hasParts,
    onOrthoChange,
    onToolChange,
    onCamera,
    onClearMeasurements,
    measurement,
    toasts,
    onDismissToast,
    showDownsampledWarning,
    onDismissDownsampledWarning,
    staleResults,
    onRunAgain,
    running,
    onRequestAddParts,
    registerAddParts,
    log,
    orient,
    onOrientPick,
    prompt
  } = props;

  const [dragDepth, setDragDepth] = useState(0);
  const fileRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    registerAddParts(() => fileRef.current?.click());
  }, [registerAddParts]);

  return (
    <main
      className="stage"
      onDragEnter={(event) => {
        if (!event.dataTransfer.types.includes('Files')) return;
        event.preventDefault();
        setDragDepth((depth) => depth + 1);
      }}
      onDragOver={(event) => {
        if (event.dataTransfer.types.includes('Files')) event.preventDefault();
      }}
      onDragLeave={() => setDragDepth((depth) => Math.max(0, depth - 1))}
      onDrop={(event) => {
        if (!event.dataTransfer.files.length) return;
        event.preventDefault();
        setDragDepth(0);
        onAddFiles(Array.from(event.dataTransfer.files));
      }}
    >
      {/* The canvas and everything painted over it. The log drawer sits
          outside this box so opening it shortens the canvas instead of
          covering the stats chip and the orientation gizmo. */}
      {/* `has-stats` tells the stylesheet that the bottom-centre band is taken
          by the stats chip, so the toast column lifts clear of it. */}
      <div className={gridMeta && resultsAvailable ? 'stage-main has-stats' : 'stage-main'}>
      <div className="stage-canvas">
        <ResultsViewport
          ref={viewportRef}
          projectId={projectId}
          parts={parts}
          selectedPartId={selectedPartId}
          onSelectPart={onSelectPart}
          resultsAvailable={resultsAvailable}
          resultsVersion={resultsVersion}
          view={view}
          onViewChange={onViewChange}
          ortho={ortho}
          tool={tool}
          onProbe={onProbe}
          onMeasure={onMeasure}
          pourLocation={pourLocation}
          onPourPicked={onPourPicked}
          onGridMeta={onGridMeta}
          modeling={modeling.viewState}
          onModelingEvent={modeling.handleEvent}
          orient={orient}
          onOrientPick={onOrientPick}
          onStatus={onStatus}
          onError={onError}
        />
      </div>

      {/* One column for everything that appears under the toolbar, so the
          banner and the loading pill stack instead of landing on each other or
          on the toolbar. It stops short of the ColorBar's gutter on the right;
          see `.stage-top-stack`. */}
      {showDownsampledWarning || status || staleResults !== null || prompt ? (
        <div className="stage-top-stack">
          {prompt ?? null}
          {staleResults !== null ? (
            <StaleResultsBanner changes={staleResults} onRun={onRunAgain} running={running} />
          ) : null}
          {showDownsampledWarning ? <DownsampledBanner onDismiss={onDismissDownsampledWarning} /> : null}
          {status ? (
            <div className="stage-status" aria-live="polite">
              {status}
            </div>
          ) : null}
        </div>
      ) : null}

      {parts.length === 0 && !busy && !modeling.active ? <EmptyState onAddParts={onRequestAddParts} /> : null}
      {dragDepth > 0 ? <DropVeil /> : null}

      {modeling.active ? (
        <ModelingToolbar modeling={modeling} onCamera={onCamera} />
      ) : (
        <ViewportToolbar
          tool={tool}
          onToolChange={onToolChange}
          ortho={ortho}
          onOrthoChange={onOrthoChange}
          onCamera={onCamera}
          onClearMeasurements={onClearMeasurements}
          hasParts={hasParts}
        />
      )}

      {view.showLegend && resultsAvailable && !modeling.active ? (
        <div className="overlay-colorbar">
          <ColorBar view={view} meta={gridMeta} orientation="vertical" />
        </div>
      ) : null}

      {probe || measurement ? (
        <ProbeReadout
          reading={probe}
          measurement={measurement}
          onClear={() => {
            onProbe(null);
            onClearMeasurements();
          }}
        />
      ) : null}
      {gridMeta && resultsAvailable && !modeling.active ? <StatsChip meta={gridMeta} /> : null}

      <Toasts toasts={toasts} onDismiss={onDismissToast} />
      </div>

      {log}

      <input
        ref={fileRef}
        type="file"
        accept=".stl,.step,.stp"
        multiple
        className="visually-hidden"
        onChange={(event) => {
          onAddFiles(Array.from(event.target.files ?? []));
          event.target.value = '';
        }}
      />
    </main>
  );
}
