import { useEffect, useState } from 'react';
import { ArrowDownToLine, Crosshair, RotateCcw, Undo2 } from 'lucide-react';

import { Group, SetupPanel } from '../SetupPanel';
import { HEIGHT_STEPS, ROTATION_STEPS, suggestedPartingSide, type OrientController } from '../../hooks/useOrient';
import { boundsSize, type WorldAxis } from '../../lib/orientation';
import { partTypeLabel } from '../../lib/partTypes';

const AXES: { axis: WorldAxis; color: string }[] = [
  { axis: 'x', color: '#e5534b' },
  { axis: 'y', color: '#57ab5a' },
  { axis: 'z', color: '#539bf5' }
];

const PICK_HINT = {
  faceDown: 'Click a face of the part to turn it face down (Esc to cancel).',
  faceUp: 'Click a face of the part to turn it face up (Esc to cancel).',
  pickHeight: 'Click a point on the model to put the parting line at its height (Esc to cancel).'
} as const;

function fmt(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined || !Number.isFinite(value) ? '--' : value.toFixed(digits);
}

function NumberField({
  label,
  value,
  onCommit,
  step,
  disabled
}: {
  label: string;
  value: number | null;
  onCommit: (value: number) => void;
  step: number;
  disabled?: boolean;
}) {
  const shown = value === null ? '' : String(Math.round(value * 1e4) / 1e4);
  const [draft, setDraft] = useState(shown);
  useEffect(() => setDraft(shown), [shown]);
  const commit = () => {
    const parsed = Number(draft);
    if (draft.trim() === '' || !Number.isFinite(parsed)) {
      setDraft(shown);
      return;
    }
    if (parsed !== value) onCommit(parsed);
  };
  return (
    <input
      type="number"
      className="input input-num"
      aria-label={label}
      title={label}
      step={step}
      value={draft}
      disabled={disabled}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === 'Enter') commit();
      }}
    />
  );
}

function StepPicker({
  label,
  steps,
  value,
  unit,
  onChange
}: {
  label: string;
  steps: readonly number[];
  value: number;
  unit: string;
  onChange: (value: number) => void;
}) {
  return (
    <div className="segmented segmented-block" role="group" aria-label={label}>
      {steps.map((step) => (
        <button
          key={step}
          type="button"
          aria-pressed={value === step}
          aria-label={`${label} ${step}${unit === '°' ? ' degrees' : ' in'}`}
          onClick={() => onChange(step)}
        >
          {step}
          {unit}
        </button>
      ))}
    </div>
  );
}

export function OrientPanel({ orient, onClose }: { orient: OrientController; onClose: () => void }) {
  const {
    target,
    targetBounds,
    casting,
    partingLine,
    presets,
    depths,
    split,
    measuring,
    infoError,
    pick
  } = orient;
  const size = targetBounds ? boundsSize(targetBounds) : null;
  const isCasting = Boolean(target && casting && target.id === casting.id);
  const suggestedSide = target && !isCasting ? suggestedPartingSide(target.part_type) : null;
  const disabled = !target;

  return (
    <SetupPanel title="Orient & Parting Line" onClose={onClose}>
      <div className="orient-target">
        <div className="orient-target-text">
          <span className="field-label">Target</span>
          <span className="orient-target-name" aria-label="Orient target">
            {target ? `${target.name} (${partTypeLabel(target.part_type)})` : 'No part'}
          </span>
        </div>
        <button
          type="button"
          className="btn btn-sm"
          onClick={orient.undo}
          disabled={!orient.undoLabel}
          aria-label="Undo"
          title={orient.undoLabel ? `Undo: ${orient.undoLabel}` : 'Nothing to undo'}
        >
          <Undo2 size={12} />
          Undo
        </button>
      </div>
      <p className="field-hint">
        Acts on the selected part, or on the casting when nothing is selected. Click a part in the view to orient it
        instead.
      </p>
      <div className="orient-readout" aria-label="Target size">
        <span>Size W × D × H</span>
        <span className="mono">
          {size ? `${fmt(size[0])} × ${fmt(size[1])} × ${fmt(size[2])} in` : '--'}
        </span>
      </div>

      {pick ? (
        <div className="status-note status-note-info orient-pick-note" role="status">
          <span>{PICK_HINT[pick]}</span>
          <button type="button" className="btn btn-sm" onClick={orient.cancelPick} aria-label="Cancel pick">
            Cancel
          </button>
        </div>
      ) : null}

      <Group title="Rotate">
        <StepPicker
          label="Rotation step"
          steps={ROTATION_STEPS}
          value={orient.rotationStep}
          unit="°"
          onChange={orient.setRotationStep}
        />
        <div className="orient-axis-rows">
          {AXES.map(({ axis, color }) => {
            const name = axis.toUpperCase();
            return (
              <div className="orient-axis-row" key={axis}>
                <span className="orient-axis-name" style={{ color }}>
                  {name}
                </span>
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={disabled}
                  aria-label={`Rotate ${name} minus`}
                  title={`Turn ${orient.rotationStep}° clockwise about the world ${name} axis, seen from +${name}, about the part's centre`}
                  onClick={() => orient.rotate(axis, -1)}
                >
                  − {orient.rotationStep}°
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={disabled}
                  aria-label={`Rotate ${name} plus`}
                  title={`Turn ${orient.rotationStep}° counter-clockwise about the world ${name} axis, seen from +${name}, about the part's centre`}
                  onClick={() => orient.rotate(axis, 1)}
                >
                  + {orient.rotationStep}°
                </button>
              </div>
            );
          })}
        </div>
        <label className="check">
          <input
            type="checkbox"
            checked={orient.keepOnFloor}
            onChange={(event) => orient.setKeepOnFloor(event.target.checked)}
            aria-label="Keep on floor"
          />
          <span className="check-body">
            <span>Keep on floor</span>
            <span className="check-note">After a turn, the part's lowest point stays at the same height.</span>
          </span>
        </label>
        <div className="row row-wrap">
          <button
            type="button"
            className={pick === 'faceDown' ? 'btn btn-sm btn-primary' : 'btn btn-sm'}
            aria-pressed={pick === 'faceDown'}
            disabled={disabled}
            onClick={() => orient.armPick('faceDown')}
            title="Then click a face in the view: it is turned to point straight down, onto the floor"
          >
            <Crosshair size={12} />
            Face down
          </button>
          <button
            type="button"
            className={pick === 'faceUp' ? 'btn btn-sm btn-primary' : 'btn btn-sm'}
            aria-pressed={pick === 'faceUp'}
            disabled={disabled}
            onClick={() => orient.armPick('faceUp')}
            title="Then click a face in the view: it is turned to point straight up"
          >
            <Crosshair size={12} />
            Face up
          </button>
          <button
            type="button"
            className="btn btn-sm"
            disabled={disabled}
            onClick={orient.resetOrientation}
            title="Turn the part back to how it was imported (rotation 0, 0, 0), about its centre"
          >
            <RotateCcw size={12} />
            Reset orientation
          </button>
        </div>
        <details className="orient-advanced">
          <summary>Exact rotation (advanced)</summary>
          <p className="field-hint">
            Static X-Y-Z Euler angles in degrees, as the simulation applies them (X first, then Y, then Z, about the
            world axes). Applies on Enter.
          </p>
          <div className="transform-grid">
            {AXES.map(({ axis }) => {
              const name = axis.toUpperCase();
              const values = target ? [target.rotation_x, target.rotation_y, target.rotation_z] : null;
              const index = { x: 0, y: 1, z: 2 }[axis];
              return (
                <label className="transform-axis-field" key={axis}>
                  <span className="transform-axis">{name}</span>
                  <NumberField
                    label={`Exact rotation ${name} (degrees)`}
                    value={values ? values[index] : null}
                    step={1}
                    disabled={disabled}
                    onCommit={(value) => {
                      if (!values) return;
                      const next: [number, number, number] = [values[0], values[1], values[2]];
                      next[index] = value;
                      orient.setExactRotation(next);
                    }}
                  />
                </label>
              );
            })}
          </div>
        </details>
      </Group>

      <Group title="Height">
        <StepPicker
          label="Height step"
          steps={HEIGHT_STEPS}
          value={orient.heightStep}
          unit=" in"
          onChange={orient.setHeightStep}
        />
        <div className="row">
          <button
            type="button"
            className="btn btn-sm"
            disabled={disabled}
            aria-label="Lower part"
            title={`Move the part down ${orient.heightStep} in`}
            onClick={() => orient.nudgeHeight(-1)}
          >
            ▼ Lower
          </button>
          <button
            type="button"
            className="btn btn-sm"
            disabled={disabled}
            aria-label="Raise part"
            title={`Move the part up ${orient.heightStep} in`}
            onClick={() => orient.nudgeHeight(1)}
          >
            ▲ Raise
          </button>
          <button
            type="button"
            className="btn btn-sm"
            disabled={disabled}
            onClick={orient.dropToFloor}
            title="Put the part's lowest point on the floor, Z = 0"
          >
            <ArrowDownToLine size={12} />
            Drop to floor
          </button>
        </div>
        <label className="field">
          <span className="field-label">
            Bottom Z <span className="field-unit">inches, applies on Enter</span>
          </span>
          <NumberField
            label="Bottom Z (in)"
            value={targetBounds ? targetBounds[0][2] : null}
            step={orient.heightStep}
            disabled={disabled}
            onCommit={orient.setBottomZ}
          />
        </label>
        {isCasting && partingLine?.locked_to_casting ? (
          <p className="field-hint">The parting line is locked to the casting and moves with it.</p>
        ) : null}
      </Group>

      <Group title="Parting line">
        {!partingLine ? (
          <div className="status-note">
            No parting line yet. Risers and gating are placed against it.
            <div className="row" style={{ marginTop: 6 }}>
              <button
                type="button"
                className="btn btn-sm btn-primary"
                disabled={!casting}
                onClick={() => orient.setPartingVisible(true)}
                title="Put the parting line at the casting's widest horizontal section and show it"
              >
                Set parting line
              </button>
            </div>
          </div>
        ) : null}
        <label className="check">
          <input
            type="checkbox"
            checked={Boolean(partingLine?.visible)}
            disabled={!casting}
            onChange={(event) => orient.setPartingVisible(event.target.checked)}
            aria-label="Show parting line"
          />
          <span className="check-body">
            <span>Show parting line</span>
            <span className="check-note">A translucent plane at the parting height: cope above, drag below.</span>
          </span>
        </label>
        <label className="field">
          <span className="field-label">
            Parting line Z <span className="field-unit">inches, applies on Enter</span>
          </span>
          <NumberField
            label="Parting line Z (in)"
            value={partingLine?.z ?? null}
            step={orient.heightStep}
            disabled={!casting}
            onCommit={(value) => orient.setPartingZ('Set parting Z', value)}
          />
        </label>
        <div className="row">
          <button
            type="button"
            className="btn btn-sm"
            disabled={!partingLine}
            aria-label="Lower parting line"
            title={`Move the parting line down ${orient.heightStep} in`}
            onClick={() => orient.nudgeParting(-1)}
          >
            ▼ Lower
          </button>
          <button
            type="button"
            className="btn btn-sm"
            disabled={!partingLine}
            aria-label="Raise parting line"
            title={`Move the parting line up ${orient.heightStep} in`}
            onClick={() => orient.nudgeParting(1)}
          >
            ▲ Raise
          </button>
          <button
            type="button"
            className={pick === 'pickHeight' ? 'btn btn-sm btn-primary' : 'btn btn-sm'}
            aria-pressed={pick === 'pickHeight'}
            disabled={!casting}
            onClick={() => orient.armPick('pickHeight')}
            title="Then click a point on the model: the parting line moves to its height"
          >
            <Crosshair size={12} />
            Pick height
          </button>
        </div>
        <div className="field">
          <span className="field-label">Presets</span>
          <div className="orient-presets">
            <button
              type="button"
              className="btn btn-sm"
              disabled={presets.widest === null}
              onClick={() => presets.widest !== null && orient.setPartingZ('Widest section', presets.widest)}
              title={
                presets.widest === null
                  ? measuring
                    ? 'Measuring the casting...'
                    : 'Needs the casting measured by the server'
                  : `Top face of the casting's widest horizontal section, Z = ${fmt(presets.widest)} in`
              }
            >
              Widest section
            </button>
            <button
              type="button"
              className="btn btn-sm"
              disabled={presets.top === null}
              onClick={() => presets.top !== null && orient.setPartingZ('Top of casting', presets.top)}
              title="At the casting's top: the whole casting in the drag"
            >
              Top of casting
            </button>
            <button
              type="button"
              className="btn btn-sm"
              disabled={presets.bottom === null}
              onClick={() => presets.bottom !== null && orient.setPartingZ('Bottom of casting', presets.bottom)}
              title="At the casting's bottom: the whole casting in the cope"
            >
              Bottom of casting
            </button>
            <button
              type="button"
              className="btn btn-sm"
              disabled={presets.center === null}
              onClick={() => presets.center !== null && orient.setPartingZ('Center', presets.center)}
              title="Half way up the casting"
            >
              Center
            </button>
          </div>
        </div>
        <div className="orient-split" aria-label="Cope and drag split">
          <div className="orient-readout">
            <span>Cope depth</span>
            <span className="mono">{depths ? `${fmt(depths.cope)} in` : '--'}</span>
          </div>
          <div className="orient-readout">
            <span>Drag depth</span>
            <span className="mono">{depths ? `${fmt(depths.drag)} in` : '--'}</span>
          </div>
          <div className="orient-readout">
            <span>Cope volume</span>
            <span className="mono">{split ? `${fmt(split.cope_fraction * 100, 0)} %` : partingLine && measuring ? '...' : '--'}</span>
          </div>
          <div className="orient-readout">
            <span>Drag volume</span>
            <span className="mono">{split ? `${fmt(split.drag_fraction * 100, 0)} %` : partingLine && measuring ? '...' : '--'}</span>
          </div>
        </div>
        {infoError ? <p className="field-hint">Volume split unavailable: {infoError}</p> : null}
        <label className="check">
          <input
            type="checkbox"
            checked={partingLine?.locked_to_casting ?? true}
            disabled={!partingLine}
            onChange={(event) => orient.setPartingLocked(event.target.checked)}
            aria-label="Lock to casting"
          />
          <span className="check-body">
            <span>Lock to casting</span>
            <span className="check-note">Raising or lowering the casting carries the parting line with it.</span>
          </span>
        </label>

        {target && !isCasting ? (
          <div className="field">
            <span className="field-label">Place {target.name} on the parting line</span>
            <div className="row row-wrap">
              <button
                type="button"
                className={suggestedSide === 'above' ? 'btn btn-sm btn-primary' : 'btn btn-sm'}
                disabled={!partingLine}
                onClick={() => orient.snapToParting('above')}
                title="Move the part so its bottom sits on the parting line (cope side)"
              >
                Seat on parting line
              </button>
              <button
                type="button"
                className={suggestedSide === 'below' ? 'btn btn-sm btn-primary' : 'btn btn-sm'}
                disabled={!partingLine}
                onClick={() => orient.snapToParting('below')}
                title="Move the part so its top meets the parting line from below (drag side)"
              >
                Hang from parting line
              </button>
            </div>
            {suggestedSide ? (
              <p className="field-hint">
                Suggested for a {partTypeLabel(target.part_type).toLowerCase()}:{' '}
                {suggestedSide === 'above' ? 'seat it on the parting line (cope side).' : 'hang it from the parting line (drag side).'}
              </p>
            ) : null}
          </div>
        ) : null}
      </Group>
    </SetupPanel>
  );
}
