import { Pause, Play, Repeat, SkipBack, SkipForward } from 'lucide-react';

import { formatSeconds } from '../lib/format';
import { isTimeMode, type GridMeta, type ResultViewState } from '../lib/viewState';

export type TransportBarProps = {
  view: ResultViewState;
  onViewChange: (patch: Partial<ResultViewState>) => void;
  meta: GridMeta | null;
  /** No results means nothing to scrub; the bar gives its height to the canvas. */
  resultsAvailable: boolean;
  loop: boolean;
  onLoopChange: (loop: boolean) => void;
};

const SPEEDS = [0.5, 1, 2, 4];

/** Frame timings for whichever time series the current mode plays back. */
function frameTimes(meta: GridMeta | null, view: ResultViewState): number[] {
  if (!meta) return [];
  if (view.mode === 'fraction_liquid') return meta.frames.fraction_liquid?.times_s ?? [];
  return meta.frames.fill?.times_s ?? meta.frames.fraction_liquid?.times_s ?? [];
}

/**
 * Playback for the time-resolved modes.
 *
 * The percentage is the source of truth (it is what `ResultViewState` carries
 * and what the viewport samples); seconds and the frame index are derived from
 * `GridMeta.frames`, so the readout says "12.4 s of 31.7 s, frame 38 of 93"
 * instead of a bare slider position that means nothing to a foundry engineer.
 */
export function TransportBar({
  view,
  onViewChange,
  meta,
  resultsAvailable,
  loop,
  onLoopChange
}: TransportBarProps) {
  if (!resultsAvailable || !isTimeMode(view.mode)) return null;

  const times = frameTimes(meta, view);
  const frameCount = times.length;
  const maxTime = frameCount ? times[frameCount - 1] : 0;
  const percent = Math.min(100, Math.max(0, view.timePercent));
  const currentSeconds = maxTime * (percent / 100);
  const frameIndex = frameCount ? Math.min(frameCount - 1, Math.round((percent / 100) * (frameCount - 1))) : 0;
  const stepPercent = frameCount > 1 ? 100 / (frameCount - 1) : 5;

  const step = (direction: -1 | 1) => {
    const next = percent + direction * stepPercent;
    onViewChange({ timePercent: Math.min(100, Math.max(0, next)), playing: false });
  };

  return (
    <div className="transport" role="group" aria-label="Playback">
      <div className="transport-controls">
        <button
          type="button"
          className="btn btn-icon"
          onClick={() => step(-1)}
          aria-label="Step back one frame"
          title="Step back"
        >
          <SkipBack size={13} />
        </button>
        <button
          type="button"
          className="btn btn-icon btn-primary"
          onClick={() => onViewChange({ playing: !view.playing })}
          aria-label={view.playing ? 'Pause' : 'Play'}
          title={view.playing ? 'Pause (Space)' : 'Play (Space)'}
        >
          {view.playing ? <Pause size={13} /> : <Play size={13} />}
        </button>
        <button
          type="button"
          className="btn btn-icon"
          onClick={() => step(1)}
          aria-label="Step forward one frame"
          title="Step forward"
        >
          <SkipForward size={13} />
        </button>
      </div>

      <div className="transport-scrub">
        <input
          type="range"
          className="range"
          min={0}
          max={100}
          step={0.1}
          value={percent}
          aria-label="Playback position"
          onChange={(event) => onViewChange({ timePercent: Number(event.target.value) })}
        />
      </div>

      <span className="transport-time">
        {frameCount ? `${formatSeconds(currentSeconds)} / ${formatSeconds(maxTime)}` : `${percent.toFixed(0)}%`}
      </span>

      <span className="chip" title="Frame index within the stored time series">
        {frameCount ? `Frame ${frameIndex + 1} / ${frameCount}` : `${percent.toFixed(0)}%`}
      </span>

      <div className="segmented" role="group" aria-label="Playback speed">
        {SPEEDS.map((speed) => (
          <button
            key={speed}
            type="button"
            aria-pressed={view.speed === speed}
            onClick={() => onViewChange({ speed })}
          >
            {speed}x
          </button>
        ))}
      </div>

      <button
        type="button"
        className="btn btn-icon"
        aria-pressed={loop}
        onClick={() => onLoopChange(!loop)}
        aria-label="Loop playback"
        title={loop ? 'Looping' : 'Play once'}
        style={loop ? { color: 'var(--accent-hover)', borderColor: 'var(--accent-border)' } : undefined}
      >
        <Repeat size={13} />
      </button>
    </div>
  );
}
