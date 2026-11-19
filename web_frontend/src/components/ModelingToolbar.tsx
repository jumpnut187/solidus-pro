import { ArrowDownUp, Box, Check, Focus, Maximize2, PenLine, Square, X } from 'lucide-react';

import type { ModelingController } from '../hooks/useModeling';
import { referencePlaneProblem, sketchProblem, STANDARD_PLANES } from '../lib/modeling/types';
import type { CameraPreset } from '../lib/viewState';
import { SKETCH_TOOLS } from './sketchTools';

/**
 * The CommandManager over the canvas while a body is being modeled: the
 * sketch tools within a mouse flick of the sketch, and the confirm/cancel
 * pair SOLIDWORKS keeps in the corner of the graphics area.
 */
export function ModelingToolbar({
  modeling,
  onCamera
}: {
  modeling: ModelingController;
  onCamera: (preset: CameraPreset) => void;
}) {
  const session = modeling.session;
  const planeSession = modeling.planeSession;
  const measuring = modeling.measurePicks !== null;
  if (!session && !planeSession && !measuring) return null;

  const stageLabel = measuring
    ? 'Measure — click vertices, edges or faces'
    : planeSession
      ? planeSession.mode === 'mid'
        ? 'Plane — select two parallel planes or faces'
        : 'Plane — select a plane or a face'
      : session?.stage === 'plane'
        ? session.purpose === 'faceExtrude'
          ? 'Extrude Face — select a flat face'
          : 'Select a plane or a face'
        : session?.stage === 'sketch'
          ? 'Sketch'
          : session?.feature.kind === 'cut'
            ? 'Cut-Extrude'
            : 'Boss-Extrude';

  const planeSource =
    planeSession && planeSession.mode === 'offset' && planeSession.refs[0]
      ? { kind: 'offset' as const, base: planeSession.refs[0], distance: planeSession.distance, flip: planeSession.flip }
      : planeSession && planeSession.refs.length === 2
        ? { kind: 'mid' as const, a: planeSession.refs[0], b: planeSession.refs[1] }
        : null;

  return (
    <>
      <div className="viewport-toolbar modeling-toolbar" role="toolbar" aria-label="Modeling tools">
        <span className="modeling-stage">{stageLabel}</span>
        {session?.stage === 'plane' && session.purpose === 'sketch' ? (
          <>
            <span className="vtool-sep" aria-hidden="true" />
            {STANDARD_PLANES.map((plane) => (
              <button
                key={plane.id}
                type="button"
                className="vtool vtool-text"
                title={`Sketch on the ${plane.label} plane (${plane.hint})`}
                onClick={() => modeling.choosePlane({ kind: 'standard', id: plane.id, offset: 0 })}
              >
                <Square size={13} />
                {plane.label}
              </button>
            ))}
          </>
        ) : null}
        {session?.stage === 'sketch' ? (
          <>
            <span className="vtool-sep" aria-hidden="true" />
            {SKETCH_TOOLS.map(({ id, label, hint, Icon }) => (
              <button
                key={id}
                type="button"
                className="vtool"
                aria-pressed={session.tool === id}
                aria-label={label}
                title={`${label}: ${hint}`}
                onClick={() => modeling.patchSession({ tool: id })}
              >
                <Icon size={15} />
              </button>
            ))}
          </>
        ) : null}
        {session?.stage === 'feature' ? (
          <>
            <span className="vtool-sep" aria-hidden="true" />
            <button type="button" className="vtool vtool-text" title="Back to the sketch" onClick={modeling.backToSketch}>
              <PenLine size={14} />
              Edit Sketch
            </button>
            <button
              type="button"
              className="vtool"
              aria-pressed={session.feature.reverse}
              aria-label="Reverse direction"
              title="Reverse direction"
              onClick={() => modeling.patchFeature({ reverse: !session.feature.reverse })}
            >
              <ArrowDownUp size={15} />
            </button>
          </>
        ) : null}
        <span className="vtool-sep" aria-hidden="true" />
        {session && session.stage !== 'plane' ? (
          <button type="button" className="vtool" aria-label="Normal To" title="Normal To the sketch plane" onClick={modeling.normalTo}>
            <Focus size={15} />
          </button>
        ) : null}
        <button type="button" className="vtool" aria-label="Isometric" title="Isometric view" onClick={() => onCamera('iso')}>
          <Box size={15} />
        </button>
        <button type="button" className="vtool" aria-label="Fit to view" title="Fit to view" onClick={() => onCamera('fit')}>
          <Maximize2 size={15} />
        </button>
      </div>

      <div className="confirm-corner" role="group" aria-label="Confirm">
        {session?.stage === 'sketch' ? (
          <button
            type="button"
            className="confirm-ok"
            title={session.feature.kind === 'cut' ? 'Exit sketch and cut' : 'Exit sketch and extrude'}
            aria-label="Exit sketch"
            disabled={Boolean(sketchProblem(session.feature.sketch.entities))}
            onClick={modeling.exitSketch}
          >
            <Check size={18} />
          </button>
        ) : null}
        {session?.stage === 'feature' ? (
          <button
            type="button"
            className="confirm-ok"
            title="OK"
            aria-label="OK"
            disabled={modeling.preview.status !== 'ok' || modeling.committing}
            onClick={() => void modeling.commit()}
          >
            <Check size={18} />
          </button>
        ) : null}
        {planeSession ? (
          <button
            type="button"
            className="confirm-ok"
            title="OK"
            aria-label="OK"
            disabled={Boolean(referencePlaneProblem(planeSource))}
            onClick={() => void modeling.commitPlane()}
          >
            <Check size={18} />
          </button>
        ) : null}
        <button type="button" className="confirm-cancel" title={measuring ? 'Close' : 'Cancel'} aria-label="Cancel" onClick={modeling.cancel}>
          <X size={18} />
        </button>
      </div>
    </>
  );
}
