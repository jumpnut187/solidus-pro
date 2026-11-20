import { useEffect, useState } from 'react';

import './viewport.css';

import { gradientCss } from '../lib/colormaps';
import { formatDuration, playbackSeconds } from '../lib/format';
import { formatFieldValue } from './formatValue';
import {
  RESULT_MODE_LABELS,
  RESULT_MODE_UNITS,
  isSweepMode,
  isTimeMode,
  type ColorBarProps,
  type ResultModeId
} from '../lib/viewState';

const TICK_COUNT = 6;

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
  const unit = meta?.units ? (meta.units[mode] ?? '') : (RESULT_MODE_UNITS[mode] ?? '');

  const autoMin = range?.min ?? 0;
  const autoMax = range?.max ?? 1;
  const min = view.scaleMin ?? autoMin;
  const max = view.scaleMax ?? autoMax;
  const span = max - min;

  const semantic = SEMANTIC_ENDS[mode];
  const timeMode = isTimeMode(mode) || isSweepMode(mode);

  const ticks =
    span > 0
      ? Array.from({ length: TICK_COUNT }, (_, index) => {
          const fraction = index / (TICK_COUNT - 1);
          return { fraction, value: min + span * fraction };
        })
      : [{ fraction: 0.5, value: min }];
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
        <ol className={ticks.length === 1 ? 'vp-colorbar-ticks vp-colorbar-ticks-single' : 'vp-colorbar-ticks'}>
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

function TimeReadout({ view, meta }: Pick<ColorBarProps, 'view'> & { meta: NonNullable<ColorBarProps['meta']> }) {
  let start: number;
  let end: number;
  if (isSweepMode(view.mode)) {
    const range = meta.ranges[view.mode];
    if (!range || !(range.max > 0)) return null;
    start = 0;
    end = range.max;
  } else {
    const frames =
      view.mode === 'fraction_liquid'
        ? meta.frames.fraction_liquid
        : view.mode === 'fill_animation'
          ? meta.frames.fill
          : undefined;
    const times = frames?.times_s ?? [];
    if (times.length < 2) return null;
    start = times[0];
    end = times[times.length - 1];
  }
  const seconds = playbackSeconds(view.timePercent / 100, start, end);
  return (
    <p className="vp-colorbar-time">
      {formatDuration(seconds)} <span className="vp-colorbar-time-total">of {formatDuration(end)}</span>
    </p>
  );
}
