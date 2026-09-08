/**
 * The result legend.
 *
 * A color ramp is only as useful as its numbers: "red" means nothing until it
 * means 2.01 inches of feed modulus or 731 seconds to solidify. So the bar
 * carries real ticks in field units, the unit itself, and the mode's name,
 * and its end labels are editable when the shell passes `onScaleChange` --
 * clamping the scale is how you get contrast out of a field where one hot
 * spot pins the top of the range.
 *
 * Vertical by default: it sits beside a 3D view whose width is worth more
 * than its height, and vertical is what an operator reading a casting report
 * expects.
 */
import { useEffect, useState } from 'react';

import './viewport.css';

import { gradientCss } from '../lib/colormaps';
import { formatFieldValue } from './formatValue';
import {
  RESULT_MODE_LABELS,
  RESULT_MODE_UNITS,
  isTimeMode,
  type ColorBarProps,
  type ResultModeId
} from '../lib/viewState';

const TICK_COUNT = 6;

/** End labels for the modes where the number matters less than the meaning. */
const SEMANTIC_ENDS: Partial<Record<ResultModeId, [string, string]>> = {
  fraction_liquid: ['Solid', 'Liquid'],
  fill_animation: ['Empty', 'Full'],
  fill_time: ['First filled', 'Last filled']
};

type EditableEndProps = {
  value: number;
  auto: boolean;
  unit: string;
  onCommit: (value: number | null) => void;
  label: string;
};

function EditableEnd({ value, auto, unit, onCommit, label }: EditableEndProps) {
  const [draft, setDraft] = useState<string | null>(null);
  const display = `${formatFieldValue(value)}${unit ? ` ${unit}` : ''}`;

  useEffect(() => {
    setDraft(null);
  }, [value]);

  if (draft === null) {
    return (
      <button
        type="button"
        className={auto ? 'vp-colorbar-end vp-colorbar-end-auto' : 'vp-colorbar-end'}
        onClick={() => setDraft(String(value))}
        title={auto ? `${label} (auto from the data; click to set)` : `${label} (click to edit)`}
      >
        {display}
      </button>
    );
  }

  const commit = () => {
    const trimmed = draft.trim();
    // An emptied box means "go back to the data's own range", which is the
    // only way out of a scale the user has pinned.
    if (trimmed === '') {
      onCommit(null);
    } else {
      const parsed = Number.parseFloat(trimmed);
      if (Number.isFinite(parsed)) onCommit(parsed);
    }
    setDraft(null);
  };

  return (
    <input
      className="vp-colorbar-end-input"
      value={draft}
      autoFocus
      aria-label={label}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === 'Enter') commit();
        if (event.key === 'Escape') setDraft(null);
      }}
    />
  );
}

export function ColorBar({ view, meta, orientation = 'vertical', onScaleChange }: ColorBarProps) {
  const vertical = orientation !== 'horizontal';
  const mode = view.mode;
  const range = meta?.ranges[mode];
  // The server's unit table wins outright when it is present, including when
  // it says a mode has *no* unit. `RESULT_MODE_UNITS` is a static fallback for
  // before a run has loaded, and it still lists a unit for fill animation,
  // whose values are a 0..1 fraction and not the seconds its frames are
  // stamped with.
  const unit = meta?.units ? (meta.units[mode] ?? '') : (RESULT_MODE_UNITS[mode] ?? '');

  const autoMin = range?.min ?? 0;
  const autoMax = range?.max ?? 1;
  const min = view.scaleMin ?? autoMin;
  const max = view.scaleMax ?? autoMax;
  const span = max - min;

  const semantic = SEMANTIC_ENDS[mode];
  const timeMode = isTimeMode(mode);

  // Ticks are listed low to high and reversed for the vertical bar, so the
  // top of the ramp is the top of the list either way.
  const ticks = Array.from({ length: TICK_COUNT }, (_, index) => {
    const fraction = index / (TICK_COUNT - 1);
    return { fraction, value: min + span * fraction };
  });
  const ordered = vertical ? [...ticks].reverse() : ticks;

  return (
    <div
      className={vertical ? 'vp-colorbar vp-colorbar-vertical' : 'vp-colorbar vp-colorbar-horizontal'}
      aria-label={`${RESULT_MODE_LABELS[mode]} color scale`}
    >
      <p className="vp-colorbar-title">{RESULT_MODE_LABELS[mode]}</p>
      {unit ? <p className="vp-colorbar-unit">{unit}</p> : null}

      <div className="vp-colorbar-body">
        <div
          className="vp-colorbar-ramp"
          style={{ background: gradientCss(view.colormap, vertical ? 'to top' : 'to right') }}
        />
        <ol className="vp-colorbar-ticks">
          {ordered.map((tick) => (
            <li key={tick.fraction} className="vp-colorbar-tick">
              <span className="vp-colorbar-tick-mark" aria-hidden="true" />
              <span className="vp-colorbar-tick-label">{formatFieldValue(tick.value)}</span>
            </li>
          ))}
        </ol>
      </div>

      <div className="vp-colorbar-ends">
        {semantic ? (
          <>
            <span className="vp-colorbar-end vp-colorbar-end-static">{semantic[vertical ? 1 : 0]}</span>
            <span className="vp-colorbar-end vp-colorbar-end-static">{semantic[vertical ? 0 : 1]}</span>
          </>
        ) : onScaleChange ? (
          <>
            <EditableEnd
              label="Scale maximum"
              value={max}
              auto={view.scaleMax === null}
              unit={unit}
              onCommit={(next) => onScaleChange(view.scaleMin, next)}
            />
            <EditableEnd
              label="Scale minimum"
              value={min}
              auto={view.scaleMin === null}
              unit={unit}
              onCommit={(next) => onScaleChange(next, view.scaleMax)}
            />
          </>
        ) : (
          <>
            <span className="vp-colorbar-end vp-colorbar-end-static">{formatFieldValue(max)}</span>
            <span className="vp-colorbar-end vp-colorbar-end-static">{formatFieldValue(min)}</span>
          </>
        )}
      </div>

      {timeMode && meta ? <TimeReadout view={view} meta={meta} /> : null}
    </div>
  );
}

/** Seconds at the current scrubber position, for the time modes. */
function TimeReadout({ view, meta }: Pick<ColorBarProps, 'view'> & { meta: NonNullable<ColorBarProps['meta']> }) {
  // Only the modes that really are a stack of frames get a clock. Fill time
  // is listed as a time mode because the shell's transport bar owns it, but
  // it is one static map whose *colour* is the time; borrowing the fill
  // animation's clock for it put a running second count next to a picture
  // that never changes.
  const frames =
    view.mode === 'fraction_liquid'
      ? meta.frames.fraction_liquid
      : view.mode === 'fill_animation'
        ? meta.frames.fill
        : undefined;
  const times = frames?.times_s ?? [];
  if (times.length < 2) return null;
  const seconds = times[0] + (view.timePercent / 100) * (times[times.length - 1] - times[0]);
  return (
    <p className="vp-colorbar-time">
      {formatFieldValue(seconds)} s <span className="vp-colorbar-time-total">of {formatFieldValue(times[times.length - 1])} s</span>
    </p>
  );
}
