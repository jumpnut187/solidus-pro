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

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return '--';
  return `${formatNumber(Math.round(seconds * 10) / 10, 1)} s`;
}

export function playbackSeconds(progress: number, startSeconds: number, endSeconds: number): number {
  const clamped = Math.min(Math.max(progress, 0), 1);
  return startSeconds + clamped * (endSeconds - startSeconds);
}

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

export function clampToStep(value: number, min: number, max: number, decimals: number): number {
  if (!Number.isFinite(value)) return min;
  const clamped = Math.min(max, Math.max(min, value));
  const factor = 10 ** decimals;
  return Math.round(clamped * factor) / factor;
}
