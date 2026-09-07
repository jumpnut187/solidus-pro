/**
 * PLACEHOLDER. Minimal legend that honours the `ColorBarProps` contract. The
 * viewport owner replaces this with the full color bar (numeric ticks, units,
 * editable scale bounds, vertical orientation).
 */
import { gradientCss } from '../lib/colormaps';
import { RESULT_MODE_LABELS, RESULT_MODE_UNITS, type ColorBarProps } from '../lib/viewState';

function formatValue(value: number): string {
  if (!Number.isFinite(value)) return '--';
  const abs = Math.abs(value);
  if (abs >= 10000 || (abs > 0 && abs < 0.01)) return value.toExponential(1);
  if (abs >= 100) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toFixed(2);
}

export function ColorBar({ view, meta, orientation = 'horizontal' }: ColorBarProps) {
  const range = meta?.ranges[view.mode];
  const unit = RESULT_MODE_UNITS[view.mode] ?? '';
  let minLabel: string;
  let maxLabel: string;
  if (view.mode === 'fraction_liquid') {
    minLabel = 'Solid';
    maxLabel = 'Liquid';
  } else if (view.mode === 'fill_time' || view.mode === 'fill_animation') {
    minLabel = 'First filled';
    maxLabel = 'Last filled';
  } else {
    const min = view.scaleMin ?? range?.min ?? 0;
    const max = view.scaleMax ?? range?.max ?? 1;
    minLabel = `${formatValue(min)}${unit ? ` ${unit}` : ''}`;
    maxLabel = `${formatValue(max)}${unit ? ` ${unit}` : ''}`;
  }
  const vertical = orientation === 'vertical';
  return (
    <div className={vertical ? 'colorbar colorbar-vertical' : 'colorbar'} aria-label="Result color scale">
      <p className="colorbar-title">{RESULT_MODE_LABELS[view.mode]}</p>
      <div
        className="colorbar-ramp"
        style={{ background: gradientCss(view.colormap, vertical ? 'to top' : 'to right') }}
      />
      <div className="colorbar-scale">
        <span>{minLabel}</span>
        <span>{maxLabel}</span>
      </div>
    </div>
  );
}
