import { Group, SetupPanel } from '../SetupPanel';
import type { MoldPreview, SandInfo } from '../../lib/api';
import type { MoldParams } from '../../lib/mold';

export type MoldPanelProps = {
  mold: MoldParams;
  sands: SandInfo[];
  preview: MoldPreview | null;
  previewError: string | null;
  hasParts: boolean;
  onChange: (mold: MoldParams) => void;
  onReset: () => void;
  onClose: () => void;
};

function SandSelect({
  value,
  sands,
  label,
  onChange
}: {
  value: number;
  sands: SandInfo[];
  label: string;
  onChange: (code: number) => void;
}) {
  const selected = sands.find((sand) => sand.code === value);
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <select className="select" value={value} onChange={(event) => onChange(Number(event.target.value))}>
        {sands.map((sand) => (
          <option key={sand.code} value={sand.code}>
            {sand.label}
          </option>
        ))}
      </select>
      {selected ? <span className="field-hint">{selected.uses}</span> : null}
    </label>
  );
}

function NumberField({
  label,
  unit,
  value,
  min,
  max,
  step,
  hint,
  disabled,
  onChange
}: {
  label: string;
  unit: string;
  value: number;
  min: number;
  max: number;
  step: number;
  hint?: string;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field">
      <span className="field-label">
        {label}
        <span className="field-unit">{unit}</span>
      </span>
      <input
        type="number"
        className="input"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next)) onChange(Math.min(max, Math.max(min, next)));
        }}
      />
      {hint ? <span className="field-hint">{hint}</span> : null}
    </label>
  );
}

/**
 * The sand mold and cores (`MoldParams`, `core/mold_setup.py`).
 *
 * Mold bodies draw purple and cores green, translucent, while setting up; both
 * are hidden while a result is on screen.
 */
export function MoldPanel({ mold, sands, preview, previewError, hasParts, onChange, onReset, onClose }: MoldPanelProps) {
  const set = (patch: Partial<MoldParams>) => onChange({ ...mold, ...patch });
  const summary = preview?.summary ?? null;
  let note = 'Without a mold, the mold is the semi-infinite sand boundary of the Thermal parameters.';
  if (previewError) note = previewError;
  else if (summary)
    note =
      `${summary.mold_cells.toLocaleString()} mold sand cells` +
      (mold.auto_cores ? `, ${summary.auto_core_cells.toLocaleString()} auto-core cells` : '') +
      ` at ${summary.pitch_in.toFixed(3)} in (preview).`;
  else if (!hasParts) note = 'Add parts to see the mold and cores.';

  return (
    <SetupPanel title="Mold & Cores" onClose={onClose} onReset={onReset} resetLabel="Reset mold and cores">
      <Group title="Mold">
        <label className="check">
          <input
            type="checkbox"
            checked={Boolean(mold.auto_mold)}
            onChange={(event) => set({ auto_mold: event.target.checked ? 1 : 0 })}
          />
          <span className="check-body">
            <span>Generate the sand mold automatically</span>
            <span className="check-note">
              The mold becomes sand cells around the assembly in the thermal solve, MAGMASOFT style. A part typed
              Mold (flask, cope, drag) is used instead when loaded.
            </span>
          </span>
        </label>
        <NumberField
          label="Wall thickness"
          unit={mold.wall_thickness_in > 0 ? 'in' : summary?.wall_in ? `auto: ${summary.wall_in.toFixed(2)} in` : 'auto'}
          value={mold.wall_thickness_in}
          min={0}
          max={48}
          step={0.5}
          disabled={!mold.auto_mold}
          hint="Sand between the assembly and the mold's outer faces. 0 = automatic: 3 x the half-thickness of the heaviest section (0.75-8 in), three heat-penetration depths at the freeze time."
          onChange={(value) => set({ wall_thickness_in: value })}
        />
        <NumberField
          label="Closed riser cover"
          unit="in"
          value={mold.closed_riser_cover_in}
          min={0}
          max={24}
          step={0.5}
          disabled={!mold.auto_mold}
          hint="Sand over a blind riser. Open risers and pour cups stay open to the air at the top of the mold; mark a riser open in the Parts panel."
          onChange={(value) => set({ closed_riser_cover_in: value })}
        />
        <SandSelect
          label="Mold sand"
          value={mold.mold_material_code}
          sands={sands}
          onChange={(code) => set({ mold_material_code: code })}
        />
      </Group>

      <Group title="Cores">
        <label className="check">
          <input
            type="checkbox"
            checked={Boolean(mold.auto_cores)}
            onChange={(event) => set({ auto_cores: event.target.checked ? 1 : 0 })}
          />
          <span className="check-body">
            <span>Find cores automatically</span>
            <span className="check-note">
              Core sand where the mold cannot form the cavity. Loaded core parts override this: with any core part,
              no cores are added.
            </span>
          </span>
        </label>
        <label className="field">
          <span className="field-label">Rule</span>
          <select
            className="select"
            value={mold.core_rule_code}
            disabled={!mold.auto_cores}
            onChange={(event) => set({ core_rule_code: Number(event.target.value) })}
          >
            <option value={0}>Draw direction: undercuts, holes and cavities</option>
            <option value={1}>Enclosed cavities only (MAGMASOFT)</option>
          </select>
          <span className="field-hint">
            A pattern drawn vertically out of cope and drag cannot form sand that has the assembly both above and
            below it.
          </span>
        </label>
        <SandSelect
          label="Core sand"
          value={mold.core_material_code}
          sands={sands}
          onChange={(code) => set({ core_material_code: code })}
        />
      </Group>

      <Group title="Interfaces">
        <NumberField
          label="Metal to sand"
          unit="W/m²K"
          value={mold.htc_cast_mold}
          min={1}
          max={100000}
          step={100}
          hint="Above the solidus. MAGMASOFT's default for sand is C1000."
          onChange={(value) => set({ htc_cast_mold: value })}
        />
        <label className="check">
          <input
            type="checkbox"
            checked={Boolean(mold.interface_htc_curve)}
            onChange={(event) => set({ interface_htc_curve: event.target.checked ? 1 : 0 })}
          />
          <span className="check-body">
            <span>Air gap below the solidus</span>
            <span className="check-note">The steel-sand curve Williams, Hardin and Beckermann fitted in MAGMASOFT.</span>
          </span>
        </label>
        <NumberField
          label="Chill to sand"
          unit="W/m²K"
          value={mold.htc_chill_mold}
          min={1}
          max={100000}
          step={100}
          onChange={(value) => set({ htc_chill_mold: value })}
        />
        <NumberField
          label="Air temperature"
          unit="°C"
          value={mold.ambient_temperature}
          min={-40}
          max={400}
          step={1}
          onChange={(value) => set({ ambient_temperature: value })}
        />
      </Group>

      <p className={previewError ? 'status-note status-note-warning' : 'status-note'}>{note}</p>
      <p className="status-note">
        Mold draws purple and cores green, translucent, while setting up. Both are hidden while a result is on
        screen.
      </p>
    </SetupPanel>
  );
}
