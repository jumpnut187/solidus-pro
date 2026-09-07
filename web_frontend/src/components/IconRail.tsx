import { Box, FlaskConical, FolderOpen, Gauge, Grid3x3, Layers, Thermometer } from 'lucide-react';
import type { ComponentType } from 'react';

import type { PanelId } from '../lib/uiPrefs';

type RailEntry = {
  id: PanelId;
  label: string;
  title: string;
  Icon: ComponentType<{ size?: number }>;
};

const ENTRIES: RailEntry[] = [
  { id: 'parts', label: 'Parts', title: 'Parts and assembly', Icon: Box },
  { id: 'material', label: 'Alloy', title: 'Material properties', Icon: FlaskConical },
  { id: 'mesh', label: 'Mesh', title: 'Mesh parameters', Icon: Grid3x3 },
  { id: 'analysis', label: 'Analysis', title: 'Analysis options', Icon: Layers },
  { id: 'performance', label: 'Perf', title: 'Performance options', Icon: Gauge },
  { id: 'solidification', label: 'Params', title: 'Solidification parameters', Icon: Thermometer },
  { id: 'projects', label: 'Projects', title: 'Saved projects', Icon: FolderOpen }
];

/**
 * The left rail. Exactly one setup panel is open at a time and clicking the
 * active icon closes it, so the canvas is never competing with two stacked
 * panels the way the old 300px sidebar did.
 */
export function IconRail({
  active,
  onSelect,
  partCount
}: {
  active: PanelId | null;
  onSelect: (id: PanelId | null) => void;
  partCount: number;
}) {
  return (
    <nav className="rail" aria-label="Setup panels">
      {ENTRIES.map(({ id, label, title, Icon }) => (
        <button
          key={id}
          type="button"
          className="rail-btn"
          title={title}
          aria-pressed={active === id}
          onClick={() => onSelect(active === id ? null : id)}
        >
          <Icon size={17} />
          <span className="rail-btn-label">{label}</span>
          {id === 'parts' && partCount > 0 ? <span className="rail-count">{partCount}</span> : null}
        </button>
      ))}
    </nav>
  );
}
