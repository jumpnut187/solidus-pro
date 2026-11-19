/** Number formatting shared by the legend and the readouts around it. */

/**
 * Format a value for a tick.
 *
 * Fields here span from 0..1 risks to 731-second solidification times to
 * micron SDAS, so a fixed precision is wrong for at least one of them. The
 * rule is "about three significant figures, and never scientific notation
 * where a plain number fits", because a legend is read at a glance.
 */
export function formatFieldValue(value: number): string {
  if (!Number.isFinite(value)) return '--';
  const magnitude = Math.abs(value);
  if (magnitude === 0) return '0';
  if (magnitude >= 100000 || magnitude < 0.001) return value.toExponential(1);
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 10) return value.toFixed(1);
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(3);
}
