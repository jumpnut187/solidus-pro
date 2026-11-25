import { useState } from 'react';
import { RotateCcw } from 'lucide-react';

import { Modal } from './Modal';
import { clampToStep } from '../../lib/format';
import {
  PARAM_TABS,
  isBooleanParam,
  paramsForTab,
  SLIDER_KEYS,
  type ParamDef,
  type ParamGroup
} from '../../lib/solidificationFields';
import { changedCount, readParam, resetGroup, writeParam } from '../../lib/solidificationValues';
import type { SolidificationParams } from '../../lib/api';

export type SolidificationModalProps = {
  params: SolidificationParams;
  onChange: (params: SolidificationParams) => void;
  onClose: () => void;
};

export function SolidificationModal({ params, onChange, onClose }: SolidificationModalProps) {
  const [tab, setTab] = useState<ParamGroup>('feeding');
  const { main, filter } = paramsForTab(tab);
  const changed = changedCount(params, tab);

  const set = (def: ParamDef, value: number) => onChange(writeParam(params, def, value));

  return (
    <Modal
      title="Solidification Parameters"
      subtitle="Changes apply immediately and are saved with the project."
      size="lg"
      onClose={onClose}
      toolbar={
        <div className="tabs" role="tablist" aria-label="Parameter groups">
          {PARAM_TABS.map((entry) => (
            <button
              key={entry.id}
              type="button"
              role="tab"
              className="tab"
              aria-selected={tab === entry.id}
              onClick={() => setTab(entry.id)}
            >
              {entry.label}
            </button>
          ))}
        </div>
      }
      footer={
        <>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => onChange(resetGroup(params, tab))}
            disabled={changed === 0}
          >
            <RotateCcw size={12} />
            Reset {PARAM_TABS.find((entry) => entry.id === tab)?.label}
          </button>
          <span className="text-faint" style={{ fontSize: 'var(--fs-sm)' }}>
            {changed === 0 ? 'All defaults' : `${changed} changed from default`}
          </span>
          <div className="spacer" />
          <button type="button" className="btn btn-primary" onClick={onClose}>
            Done
          </button>
        </>
      }
    >
      <div className="param-list" role="tabpanel">
        {main.map((def) => (
          <ParamRow key={def.key} def={def} value={readParam(params, def)} onChange={(value) => set(def, value)} />
        ))}
        {filter.length ? (
          <div className="param-subgroup">
            <div className="group-head" style={{ marginBottom: 'var(--sp-3)' }}>
              <span className="group-title">Ceramic filter</span>
              <span className="group-rule" />
            </div>
            <div className="param-list">
              {filter.map((def) => (
                <ParamRow
                  key={def.key}
                  def={def}
                  value={readParam(params, def)}
                  onChange={(value) => set(def, value)}
                />
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </Modal>
  );
}

function ParamRow({
  def,
  value,
  onChange
}: {
  def: ParamDef;
  value: number;
  onChange: (value: number) => void;
}) {
  const isSlider = SLIDER_KEYS.has(def.key);
  const isBoolean = isBooleanParam(def);
  const meta = isBoolean
    ? def.unit
    : def.unit
      ? `${def.unit} · ${def.min} to ${def.max}`
      : `${def.min} to ${def.max}`;

  const label = (
    <span className="param-label">
      <span className="param-name" title={def.help}>
        {def.label}
      </span>
      {meta ? <span className="param-unit">{meta}</span> : null}
    </span>
  );

  if (isBoolean) {
    return (
      <div className="param-row">
        {label}
        <label className="check" style={{ justifySelf: 'end' }} title={def.help}>
          <input
            type="checkbox"
            checked={value >= 0.5}
            onChange={(event) => onChange(event.target.checked ? 1 : 0)}
          />
          <span className="check-body">
            <span>{value >= 0.5 ? 'On' : 'Off'}</span>
          </span>
        </label>
      </div>
    );
  }

  const numberInput = (
    <input
      type="number"
      className="input input-num"
      min={def.min}
      max={def.max}
      step={def.step}
      value={value}
      aria-label={def.label}
      title={def.help}
      onChange={(event) => {
        const parsed = Number(event.target.value);
        if (Number.isNaN(parsed)) return;
        onChange(parsed);
      }}
      onBlur={(event) => onChange(clampToStep(Number(event.target.value), def.min, def.max, def.decimals))}
    />
  );

  if (isSlider) {
    return (
      <div className="param-row is-slider">
        {label}
        <input
          type="range"
          className="range"
          min={def.min}
          max={def.max}
          step={def.step}
          value={value}
          aria-label={`${def.label} slider`}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        {numberInput}
      </div>
    );
  }

  return (
    <div className="param-row">
      {label}
      {numberInput}
    </div>
  );
}
