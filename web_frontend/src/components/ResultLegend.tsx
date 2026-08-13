import { MODE_CONTROL } from '../lib/uiPrefs';
import { gradientCssForMode } from '../lib/colorMaps';
import type { ResultFieldsPayload } from '../lib/api';

type Props = {
  resultMode: string;
  resultFields: ResultFieldsPayload | null;
  visible: boolean;
};

function formatValue(value: number): string {
  if (!Number.isFinite(value)) return '--';
  const abs = Math.abs(value);
  if (abs >= 10000 || (abs > 0 && abs < 0.01)) return value.toExponential(1);
  if (abs >= 100) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toFixed(2);
}

export function ResultLegend({ resultMode, resultFields, visible }: Props) {
  if (!visible || !resultFields || resultFields.empty) return null;

  const control = MODE_CONTROL[resultMode];
  if (!control) return null;

  let minLabel = '--';
  let maxLabel = '--';
  if (resultMode === 'fraction_liquid') {
    minLabel = 'Solid';
    maxLabel = 'Liquid';
  } else {
    const field = resultFields.fields[resultMode];
    if (field) {
      minLabel = formatValue(field.min);
      maxLabel = formatValue(field.max);
    }
  }

  return (
    <div className="legend-chip">
      <p className="legend-title">{control.legendTitle}</p>
      <div
        className="legend-bar"
        style={{ ['--legend-stops' as string]: gradientCssForMode(resultMode) }}
      />
      <div className="legend-scale">
        <span>{minLabel}</span>
        <span>{maxLabel}</span>
      </div>
    </div>
  );
}
