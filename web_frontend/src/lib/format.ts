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

/** Seconds as `m:ss.s`, or `s.s` under a minute — how a pour is actually read. */
export function formatSeconds(seconds: number): string {
  if (!Number.isFinite(seconds)) return '--';
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest < 10 ? '0' : ''}${rest.toFixed(1)}`;
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
