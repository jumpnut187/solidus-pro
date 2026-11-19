/** Number and date formatting shared by the panels, overlays and modals. */

/** A measured value: fixed decimals, thousands separators, no exponent soup. */
export function formatNumber(value: number, decimals = 2): string {
  if (!Number.isFinite(value)) return '--';
  const abs = Math.abs(value);
  if (abs !== 0 && (abs >= 1e6 || abs < 1e-4)) return value.toExponential(2);
  return value.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}

export function formatCount(value: number): string {
  if (!Number.isFinite(value)) return '--';
  return Math.round(value).toLocaleString();
}

/**
 * Simulated time, for every clock readout in the shell.
 *
 * Always seconds, to one decimal, with thousands separators. The transport
 * used to read `0.0 s / 15:09.6` while the legend beside it read
 * `0 s of 910 s`: two styles and two roundings of the same instant. Seconds
 * win because everything this time is compared against is in seconds -- the
 * legend's ticks, a probed solidification time, the Chvorinov statistics --
 * and a readout in another unit cannot be read against them at a glance.
 * Rounding before formatting keeps a value just above zero from being shown
 * in exponent form by `formatNumber`.
 */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return '--';
  return `${formatNumber(Math.round(seconds * 10) / 10, 1)} s`;
}

/**
 * Seconds at a 0..1 playback position, the way the viewport seeks.
 *
 * `FrameStack.seek` walks the recorded times linearly from the first frame to
 * the last, so the clock has to start from the first frame's time as well;
 * scaling the last time alone disagrees with the picture whenever a run does
 * not start its record at zero.
 */
export function playbackSeconds(progress: number, startSeconds: number, endSeconds: number): number {
  const clamped = Math.min(Math.max(progress, 0), 1);
  return startSeconds + clamped * (endSeconds - startSeconds);
}

/**
 * A SQLite `CURRENT_TIMESTAMP` string is UTC without a zone marker; append one
 * so it is not read as local time and shown hours off.
 */
export function parseServerDate(value: string | null | undefined): Date | null {
  if (!value) return null;
  const normalized = /[zZ]|[+-]\d{2}:?\d{2}$/.test(value) ? value : `${value.replace(' ', 'T')}Z`;
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatRelativeTime(value: string | null | undefined): string {
  const date = parseServerDate(value);
  if (!date) return '--';
  const seconds = (Date.now() - date.getTime()) / 1000;
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)} d ago`;
  return date.toLocaleDateString();
}

export function formatDateTime(value: string | null | undefined): string {
  const date = parseServerDate(value);
  return date ? date.toLocaleString() : '--';
}

/** Clamp and round a numeric field to its declared step and decimals. */
export function clampToStep(value: number, min: number, max: number, decimals: number): number {
  if (!Number.isFinite(value)) return min;
  const clamped = Math.min(max, Math.max(min, value));
  const factor = 10 ** decimals;
  return Math.round(clamped * factor) / factor;
}
