import type { ReactNode } from 'react';
import { ChartColumn } from 'lucide-react';

import { ConfigDrawer } from '../ConfigDrawer';
import type { ProjectSettings } from '../../lib/api';

type Props = {
  settings: ProjectSettings;
  setAnalysis: (key: string, value: boolean | number) => void;
  onClose: () => void;
};

const ANALYSIS_ROWS: Array<{ key: string; label: string; section: 'core' | 'advanced' }> = [
  { key: 'shrinkage', label: 'Shrinkage / Porosity', section: 'core' },
  { key: 'hotspot', label: 'Hot Spots (Last to Freeze)', section: 'core' },
  { key: 'niyama', label: 'Niyama Criterion', section: 'core' },
  { key: 'fraction_liquid', label: 'Fraction Liquid (Solidification Path)', section: 'core' },
  { key: 'hot_tear', label: 'Hot Tear Risk', section: 'core' },
  { key: 'sand_burn_in', label: 'Sand Burn-In', section: 'core' },
  { key: 'feeding_path', label: 'Feeding Path Analysis', section: 'advanced' },
  { key: 'stress_analysis', label: 'Thermal Stress & Distortion', section: 'advanced' },
  { key: 'microstructure', label: 'Microstructure (SDAS) Prediction', section: 'advanced' },
];

function Check({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return <label className="check"><input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} /> {label}</label>;
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <strong style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6, letterSpacing: '0.5px' }}>
      {children}
    </strong>
  );
}

export function AnalysisDrawer({ settings, setAnalysis, onClose }: Props) {
  return (
    <ConfigDrawer title="Analysis Options" icon={<ChartColumn size={16} />} onClose={onClose}>
      <SectionLabel>Core Analysis:</SectionLabel>
      {ANALYSIS_ROWS.filter((row) => row.section === 'core').map((row) => (
        <Check
          key={row.key}
          label={row.label}
          checked={Boolean(settings.analysis_options[row.key as keyof typeof settings.analysis_options])}
          onChange={(v) => setAnalysis(row.key, v)}
        />
      ))}
      <SectionLabel>Advanced Analysis:</SectionLabel>
      {ANALYSIS_ROWS.filter((row) => row.section === 'advanced').map((row) => (
        <Check
          key={row.key}
          label={row.label}
          checked={Boolean(settings.analysis_options[row.key as keyof typeof settings.analysis_options])}
          onChange={(v) => setAnalysis(row.key, v)}
        />
      ))}
      <label>
        Sand Restraint Index {Number(settings.analysis_options.sand_restraint_index).toFixed(2)}
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={Number(settings.analysis_options.sand_restraint_index)}
          onChange={(e) => setAnalysis('sand_restraint_index', Number(e.target.value))}
        />
      </label>
      <p className="muted">Select only the analyses you need to reduce computation time and memory usage.</p>
    </ConfigDrawer>
  );
}
