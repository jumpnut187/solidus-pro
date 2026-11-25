import { Group, SetupPanel } from '../SetupPanel';
import { niyamaHint } from '../../lib/defaults';
import type { AnalysisOptions } from '../../lib/api';

export type AnalysisPanelProps = {
  options: AnalysisOptions;
  alloy: string;
  hasPourLocation: boolean;
  onToggle: (key: string, value: boolean) => void;
  onNumber: (key: string, value: number) => void;
  onReset: () => void;
  onClose: () => void;
};

type Toggle = { key: string; label: string; note: string };

const CORE_TOGGLES: Toggle[] = [
  {
    key: 'shrinkage',
    label: 'Shrink and microporosity',
    note: 'Shrink: macro-shrinkage risk from feeding distance, modulus and solidification sequence. Microporosity: dispersed porosity risk from Niyama, feed-path blockage and Darcy margin.'
  },
  {
    key: 'hotspot',
    label: 'Hot spots (last to freeze)',
    note: 'Regions that solidify last and are isolated from feed metal.'
  },
  {
    key: 'niyama',
    label: 'Niyama criterion',
    note: 'Ny = G / sqrt(R). Low values indicate centerline shrinkage risk.'
  },
  {
    key: 'hot_tear',
    label: 'Hot tear risk',
    note: 'Tearing in the brittle temperature range at corners and section changes.'
  },
  {
    key: 'sand_burn_in',
    label: 'Sand burn-in risk',
    note: 'Metal penetration into mold pores, driven by superheat, head and contact time.'
  },
  {
    key: 'gas_porosity',
    label: 'Gas porosity risk',
    note: 'Mold gas generation and entrapment in thick sections. Adds three result modes.'
  },
  {
    key: 'fraction_liquid',
    label: 'Fraction liquid (solidification path)',
    note: 'Time-resolved liquid fraction, played back on the transport bar.'
  }
];

const ADVANCED_TOGGLES: Toggle[] = [
  {
    key: 'feeding_path',
    label: 'Feeding path analysis',
    note: 'Traces liquid feed paths from risers and finds regions that cannot be fed.'
  },
  {
    key: 'stress_analysis',
    label: 'Thermal stress and distortion',
    note: 'Residual stress, distortion vectors and crack susceptibility. Simplified model.'
  },
  {
    key: 'microstructure',
    label: 'Microstructure (SDAS) prediction',
    note: 'Secondary dendrite arm spacing and predicted UTS from cooling rate.'
  }
];

export function AnalysisPanel({
  options,
  alloy,
  hasPourLocation,
  onToggle,
  onNumber,
  onReset,
  onClose
}: AnalysisPanelProps) {
  const restraint = Number(options.sand_restraint_index ?? 0.5);
  return (
    <SetupPanel title="Analysis" onClose={onClose} onReset={onReset}>
      <Group title="Analyses to run">
        {CORE_TOGGLES.map((toggle) => (
          <label className="check" key={toggle.key}>
            <input
              type="checkbox"
              checked={Boolean(options[toggle.key])}
              onChange={(event) => onToggle(toggle.key, event.target.checked)}
            />
            <span className="check-body">
              <span>{toggle.label}</span>
              <span className="check-note">{toggle.note}</span>
            </span>
          </label>
        ))}

        <label className="check" title={hasPourLocation ? undefined : 'Set a pour location on the model first.'}>
          <input
            type="checkbox"
            checked={Boolean(options.filling)}
            disabled={!hasPourLocation}
            onChange={(event) => onToggle('filling', event.target.checked)}
          />
          <span className="check-body">
            <span>Mold filling (CFD)</span>
            <span className="check-note">
              {hasPourLocation
                ? 'Simulates the fill from the pour location: front advance, air entrapment, fill time and cold shuts.'
                : 'Needs a pour location. Set one in the Parts panel, then this becomes available.'}
            </span>
          </span>
        </label>
      </Group>

      <Group title="Advanced">
        {ADVANCED_TOGGLES.map((toggle) => (
          <label className="check" key={toggle.key}>
            <input
              type="checkbox"
              checked={Boolean(options[toggle.key])}
              onChange={(event) => onToggle(toggle.key, event.target.checked)}
            />
            <span className="check-body">
              <span>{toggle.label}</span>
              <span className="check-note">{toggle.note}</span>
            </span>
          </label>
        ))}
      </Group>

      <Group title="Mold behaviour">
        <label className="field">
          <span className="field-label">
            Sand restraint index
            <span className="field-unit">{restraint.toFixed(2)}</span>
          </span>
          <input
            type="range"
            className="range"
            min={0}
            max={1}
            step={0.05}
            value={restraint}
            aria-label="Sand restraint index"
            onChange={(event) => onNumber('sand_restraint_index', Number(event.target.value))}
          />
          <span className="field-hint">
            0.0 is a highly collapsible mold, 1.0 a very rigid one. Raises near-surface restraint in the hot
            tear and crack calculations. Neutral default is 0.50.
          </span>
        </label>
      </Group>

      <p className="status-note status-note-info">{niyamaHint(alloy)}</p>
      <p className="field-hint">
        Selecting only the analyses you need reduces computation time and memory.
      </p>
    </SetupPanel>
  );
}
