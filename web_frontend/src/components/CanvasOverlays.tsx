import { AlertTriangle, Upload, X } from 'lucide-react';

import { formatCount, formatNumber } from '../lib/format';
import { shortcutFor } from '../lib/shortcuts';
import { RESULT_MODE_LABELS, type GridMeta, type MeasureResult, type ProbeReading } from '../lib/viewState';
import type { Toast } from '../hooks/useToasts';

/**
 * Voxel and volume counts for the loaded run.
 *
 * A statistic the viewport did not report is omitted rather than shown as
 * zero: "0.0 in³" next to forty thousand voxels reads as a broken model, not
 * as a missing number.
 */
export function StatsChip({ meta }: { meta: GridMeta | null }) {
  if (!meta || meta.voxelCount <= 0) return null;
  const showCast = meta.castVoxelCount > 0 && meta.castVoxelCount !== meta.voxelCount;
  // On a tet run the counts are mesh nodes and the spacing is the display-lattice pitch.
  const tet = meta.stats.mesh_type === 'tet';
  return (
    <div className="stats-chip" aria-label="Model statistics">
      <div className="stat">
        <span className="stat-label">{tet ? 'Nodes' : 'Voxels'}</span>
        <span className="stat-value">{formatCount(meta.voxelCount)}</span>
      </div>
      {showCast ? (
        <div className="stat">
          <span className="stat-label">{tet ? 'Cast nodes' : 'Cast voxels'}</span>
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
          <span className="stat-label">{tet ? 'Lattice' : 'Voxel'}</span>
          <span className="stat-value">{formatNumber(meta.spacing, 3)} in</span>
        </div>
      ) : null}
    </div>
  );
}

/**
 * The probe value and the last measurement, in one card.
 *
 * They share a corner because they are the same kind of answer — "what is it
 * here?" — and because two floating cards fighting for the top-left is worse
 * than one that grows a section.
 */
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

/**
 * Warning that the results on screen came from a thinned lattice.
 *
 * A run past the stored-result voxel cap keeps only every step-th voxel in the
 * JSON, and this run has no full-resolution sidecar to fall back on, so the
 * grid was rebuilt from the thinned copy: the field the viewport paints and
 * every count derived from it under-report. That is not an error and must not
 * stop anyone working, but it is exactly the kind of thing a person quotes off
 * a screenshot without knowing — so it says so on the canvas, once per run,
 * and gets out of the way when dismissed.
 *
 * Re-running is the fix on its own: a completed run writes the full-resolution
 * grid beside its artifact regardless of the cap. Raising the cap is worth
 * mentioning second because it is what keeps the stored result itself whole,
 * which is what riser design reads.
 */
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

/**
 * The parts have changed since the results on screen were solved.
 *
 * The field is kept, because it is still the last answer anyone has, but it
 * describes a model that no longer exists: a deleted riser is still in it,
 * feeding the casting from where nothing is drawn. Not dismissible -- it goes
 * away when the parts match the run again or a new run finishes.
 */
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
        {/* From the shortcut table, so the chips name the modifier this
            platform actually uses and cannot drift from the handler. */}
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
