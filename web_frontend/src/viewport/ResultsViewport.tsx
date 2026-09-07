/**
 * PLACEHOLDER ADAPTER. Implements the `ViewportProps` contract on top of the
 * legacy `BabylonViewport` so the application shell can be built and run
 * before the new renderer lands. The viewport owner replaces this file
 * (and everything under `src/viewport/`) with the real implementation; the
 * props and the imperative handle must stay as declared in `lib/viewState.ts`.
 *
 * The component fills its nearest positioned ancestor.
 */
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';

import { BabylonViewport } from '../components/BabylonViewport';
import { api, type ResultFieldsPayload } from '../lib/api';
import type { SectionState as LegacySectionState } from '../lib/uiPrefs';
import type { CameraPreset, GridMeta, ResultModeId, SectionState, ViewportHandle, ViewportProps } from '../lib/viewState';

const LEGACY_CAMERA: Record<CameraPreset, string> = {
  top: 'top',
  bottom: 'bottom',
  front: 'front',
  back: 'front',
  left: 'left',
  right: 'right',
  iso: 'iso',
  fit: 'home'
};

function legacySection(section: SectionState): LegacySectionState {
  const axis = (['z', 'y', 'x'] as const).find((a) => section[a].enabled);
  if (!axis) {
    return { enabled: false, mode: 'single', axis: 'z', position: 50, thickness: 10, invert: false };
  }
  return {
    enabled: true,
    mode: 'single',
    axis,
    position: section[axis].position,
    thickness: 10,
    invert: section[axis].flip
  };
}

function metaFromFields(payload: ResultFieldsPayload): GridMeta {
  const ranges: GridMeta['ranges'] = {};
  for (const [mode, field] of Object.entries(payload.fields)) {
    ranges[mode as ResultModeId] = { min: field.min, max: field.max, mean: field.mean };
  }
  const frameCount = payload.fraction_liquid?.frames.length ?? 0;
  const maxTime = payload.solidification.max_time || 0;
  const times = (payload.fraction_liquid?.frame_percents ?? []).map((p) => (p / 100) * maxTime);
  return {
    dims: [0, 0, 0],
    origin: [0, 0, 0],
    spacing: payload.sizes[0] ?? 0.1,
    modes: payload.modes as ResultModeId[],
    ranges,
    frames: frameCount ? { fraction_liquid: { count: frameCount, times_s: times } } : {},
    voxelCount: payload.source_count,
    castVoxelCount: payload.source_count,
    totalVolumeIn3: 0,
    stats: {}
  };
}

export const ResultsViewport = forwardRef<ViewportHandle, ViewportProps>(function ResultsViewport(props, ref) {
  const {
    projectId,
    parts,
    selectedPartId,
    onSelectPart,
    resultsAvailable,
    resultsVersion,
    view,
    ortho,
    tool,
    pourLocation,
    onPourPicked,
    className
  } = props;
  const [fields, setFields] = useState<ResultFieldsPayload | null>(null);
  const [cameraCommand, setCameraCommand] = useState<string | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const callbacks = useRef(props);
  callbacks.current = props;

  useEffect(() => {
    if (!projectId || !resultsAvailable) {
      setFields(null);
      callbacks.current.onGridMeta(null);
      return;
    }
    let cancelled = false;
    callbacks.current.onStatus('Loading results...');
    api
      .resultFields(projectId, { maxPoints: 120000 })
      .then((payload) => {
        if (cancelled) return;
        setFields(payload);
        callbacks.current.onGridMeta(metaFromFields(payload));
        callbacks.current.onStatus(null);
      })
      .catch((error) => {
        if (cancelled) return;
        callbacks.current.onStatus(null);
        callbacks.current.onError(error instanceof Error ? error.message : 'Failed to load results');
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, resultsAvailable, resultsVersion]);

  useImperativeHandle(
    ref,
    () => ({
      screenshot: () =>
        new Promise<Blob>((resolve, reject) => {
          const canvas = wrapRef.current?.querySelector<HTMLCanvasElement>('canvas.viewport-canvas');
          if (!canvas) {
            reject(new Error('Viewport canvas is not ready'));
            return;
          }
          canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('Screenshot failed'))), 'image/png');
        }),
      setCamera: (preset) => setCameraCommand(LEGACY_CAMERA[preset] ?? 'home'),
      clearMeasurements: () => undefined,
      defectVolumeIn3: () => null
    }),
    []
  );

  return (
    <div
      ref={wrapRef}
      className={className}
      style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', minHeight: 0 }}
    >
      <BabylonViewport
        parts={parts}
        selectedPartId={selectedPartId}
        resultFields={fields}
        resultMode={view.mode}
        threshold={view.threshold}
        timePercent={view.timePercent}
        modelOpacity={Math.min(0.8, Math.max(0.08, view.modelOpacity / 100))}
        showModel={view.showModel}
        showResults={view.showResults}
        cameraCommand={cameraCommand}
        onCameraHandled={() => setCameraCommand(null)}
        ortho={ortho}
        section={legacySection(view.section)}
        onPartSelected={onSelectPart}
        pourPickMode={tool === 'pour'}
        pourLocation={pourLocation}
        onPourLocationPicked={onPourPicked}
      />
    </div>
  );
});
