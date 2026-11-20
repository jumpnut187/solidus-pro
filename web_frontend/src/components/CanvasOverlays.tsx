import { AlertTriangle, Upload, X } from 'lucide-react';

import { formatCount, formatNumber } from '../lib/format';
import { shortcutFor } from '../lib/shortcuts';
import { RESULT_MODE_LABELS, type GridMeta, type MeasureResult, type ProbeReading } from '../lib/viewState';
import type { Toast } from '../hooks/useToasts';

export function StatsChip({ meta }: { meta: GridMeta | null }) {
  if (!meta || meta.voxelCount <= 0) return null;
  const showCast = meta.castVoxelCount > 0 && meta.castVoxelCount !== meta.voxelCount;
  return (
    <div className="stats-chip" aria-label="Model statistics">
      <div className="stat">
        <span className="stat-label">Voxels</span>
        <span className="stat-value">{formatCount(meta.voxelCount)}</span>
      </div>
      {showCast ? (
        <div className="stat">
          <span className="stat-label">Cast voxels</span>
          <span className="stat-value">{formatCount(meta.castVoxelCount)}</span>
        </div>
      ) : null}
      {meta.totalVolumeIn3 > 0 ? (
        <div className="stat">
          <span className="stat-label">Volume</span>
          <span className="stat-value">{formatNumber(meta.totalVolumeIn3, 1)} in³</span>
        </div>
      ) : null}
      {meta.spacing > 0 ? (
        <div className="stat">
          <span className="stat-label">Voxel</span>
          <span className="stat-value">{formatNumber(meta.spacing, 3)} in</span>
        </div>
      ) : null}
    </div>
  );
}

export function ProbeReadout({
  reading,
  measurement,
  onClear
}: {
  reading: ProbeReading | null;
  measurement: MeasureResult | null;
  onClear: () => void;
}) {
  return (
    <div className="probe-readout" aria-live="polite">
      <div className="row">
        <strong style={{ fontSize: 'var(--fs-sm)' }}>
          {reading ? RESULT_MODE_LABELS[reading.mode] : 'Measurement'}
        </strong>
        <div className="spacer" />
        <button
          type="button"
          className="btn btn-ghost btn-icon"
          onClick={onClear}
          aria-label="Clear probe and measurements"
          title="Clear probe and measurements"
        >
          <X size={12} />
        </button>
      </div>
      <dl>
        {reading ? (
          <>
            <dt>Value</dt>
            <dd>
              {reading.value === null ? '--' : formatNumber(reading.value, 4)}
              {reading.unit ? ` ${reading.unit}` : ''}
            </dd>
            <dt>Point</dt>
            <dd>{reading.point.map((value) => value.toFixed(2)).join(', ')}</dd>
            {reading.partType ? (
              <>
                <dt>Body</dt>
                <dd>{reading.partType}</dd>
              </>
            ) : null}
          </>
        ) : null}
        {measurement ? (
          <>
            <dt>Distance</dt>
            <dd>{formatNumber(measurement.distanceIn, 4)} in</dd>
          </>
        ) : null}
      </dl>
    </div>
  );
}

export function DownsampledBanner({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div className="canvas-banner" role="status">
      <AlertTriangle size={14} aria-hidden="true" className="canvas-banner-icon" />
      <span className="canvas-banner-text">
        This result was stored thinned, so the painted field and the voxel counts below come from a reduced
        lattice rather than the whole casting. Re-running this project writes a full-resolution result;
        raising <b>Voxels kept in the stored result</b> in Performance first also keeps the stored copy whole.
      </span>
      <button
        type="button"
        className="btn btn-ghost btn-icon"
        onClick={onDismiss}
        aria-label="Dismiss the reduced-resolution warning"
        title="Dismiss"
      >
        <X size={12} />
      </button>
    </div>
  );
}

export function StaleResultsBanner({
  changes,
  onRun,
  running
}: {
  changes: string;
  onRun: () => void;
  running: boolean;
}) {
  return (
    <div className="canvas-banner" role="status">
      <AlertTriangle size={14} aria-hidden="true" className="canvas-banner-icon" />
      <span className="canvas-banner-text">
        <b>Results are out of date.</b> {changes ? `${changes} since this run, so` : 'The parts changed since this run, so'}{' '}
        the field shown is for a model that no longer exists. Run again to see results for these parts.
      </span>
      <button type="button" className="btn btn-sm" onClick={onRun} disabled={running}>
        {running ? 'Running…' : 'Run'}
      </button>
    </div>
  );
}

export function EmptyState({ onAddParts }: { onAddParts: () => void }) {
  return (
    <div className="empty-state">
      <span className="empty-icon" aria-hidden="true">
        <Upload size={26} />
      </span>
      <h2 className="empty-title">Start with the casting</h2>
      <p className="empty-copy">
        Drop STL or STEP files anywhere on this canvas, or use the Parts panel. Part types are detected from
        the file name — a body called <span className="mono">riser_1</span> becomes a riser, one called{' '}
        <span className="mono">downsprue</span> becomes a sprue — and you can change any of them afterwards.
      </p>
      <div className="empty-hints">
        <button type="button" className="btn btn-primary" onClick={onAddParts}>
          Add parts
        </button>
        {}
        <span className="chip">{shortcutFor('Add CAD parts')}</span>
        <span className="chip">{shortcutFor('Run simulation')} to run</span>
      </div>
    </div>
  );
}

export function DropVeil() {
  return (
    <div className="drop-veil" aria-hidden="true">
      <Upload size={28} />
      Drop CAD files to add them
    </div>
  );
}

export function Toasts({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
  if (!toasts.length) return null;
  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((toast) => (
        <div className={`toast toast-${toast.kind}`} key={toast.id}>
          <span className="toast-text">{toast.message}</span>
          <button
            type="button"
            className="btn btn-ghost btn-icon"
            onClick={() => onDismiss(toast.id)}
            aria-label="Dismiss notification"
          >
            <X size={12} />
          </button>
        </div>
      ))}
    </div>
  );
}
