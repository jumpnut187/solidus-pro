import { useState } from 'react';
import { Video } from 'lucide-react';

import { Modal } from './Modal';
import { animationSupported } from '../../lib/animationExport';

export type AnimationModalProps = {
  busy: boolean;
  progress: number;
  error: string | null;
  frameCount: number;
  onRecord: (options: { fps: number; frames: number }) => void;
  onCancel: () => void;
  onClose: () => void;
};

export function AnimationModal({
  busy,
  progress,
  error,
  frameCount,
  onRecord,
  onCancel,
  onClose
}: AnimationModalProps) {
  const [fps, setFps] = useState(10);
  const [frames, setFrames] = useState(Math.max(2, Math.min(240, frameCount || 60)));
  const supported = animationSupported();
  const seconds = frames / Math.max(1, fps);

  return (
    <Modal
      title="Export Animation"
      subtitle="Records the time sweep straight off the canvas"
      size="sm"
      onClose={busy ? onCancel : onClose}
      footer={
        <>
          <div className="spacer" />
          <button type="button" className="btn" onClick={busy ? onCancel : onClose}>
            {busy ? 'Stop' : 'Cancel'}
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy || !supported}
            onClick={() => onRecord({ fps, frames })}
          >
            <Video size={13} />
            {busy ? `Recording ${(progress * 100).toFixed(0)}%` : 'Record'}
          </button>
        </>
      }
    >
      {!supported ? (
        <p className="status-note status-note-warning">
          This browser cannot record a canvas. Chrome, Edge and Firefox can.
        </p>
      ) : null}

      <div className="grid-2">
        <label className="field">
          <span className="field-label">
            Frame rate <span className="field-unit">1 to 60 fps</span>
          </span>
          <input
            type="number"
            className="input input-num"
            min={1}
            max={60}
            step={1}
            value={fps}
            disabled={busy}
            onChange={(event) => setFps(Math.max(1, Math.min(60, Number(event.target.value) || 10)))}
          />
        </label>
        <label className="field">
          <span className="field-label">
            Steps <span className="field-unit">2 to 240</span>
          </span>
          <input
            type="number"
            className="input input-num"
            min={2}
            max={240}
            step={1}
            value={frames}
            disabled={busy}
            onChange={(event) => setFrames(Math.max(2, Math.min(240, Number(event.target.value) || 60)))}
          />
        </label>
      </div>

      <p className="field-hint" style={{ marginTop: 'var(--sp-3)' }}>
        {frames} steps at {fps} fps is about {seconds.toFixed(1)} seconds of video. Recording plays the
        sweep in real time, so leave the window in the foreground until it finishes. Output is WebM.
        {frameCount > 0 ? ` This run stored ${frameCount} frames.` : ''}
      </p>

      {busy ? (
        <div className="job-bar" style={{ marginTop: 'var(--sp-4)', flexBasis: 'auto', width: '100%' }}>
          <div className="job-bar-fill" style={{ width: `${progress * 100}%` }} />
        </div>
      ) : null}

      {error ? (
        <p className="status-note status-note-warning" style={{ marginTop: 'var(--sp-4)' }}>
          {error}
        </p>
      ) : null}
    </Modal>
  );
}
