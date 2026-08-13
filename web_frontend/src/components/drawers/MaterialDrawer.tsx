import type { Dispatch, SetStateAction } from 'react';
import { FlaskConical } from 'lucide-react';

import { ConfigDrawer } from '../ConfigDrawer';
import type { MaterialInfo } from '../../lib/api';
import type { ProjectSettings } from '../../lib/api';

type Props = {
  settings: ProjectSettings;
  setSettings: Dispatch<SetStateAction<ProjectSettings>>;
  materials: MaterialInfo[];
  material: MaterialInfo | undefined;
  onClose: () => void;
};

export function MaterialDrawer({ settings, setSettings, materials, material, onClose }: Props) {
  return (
    <ConfigDrawer title="Material Properties" icon={<FlaskConical size={16} />} onClose={onClose}>
      <label>
        Alloy
        <select value={settings.alloy} onChange={(e) => setSettings({ ...settings, alloy: e.target.value })}>
          {materials.length === 0 && <option value={settings.alloy}>{settings.alloy} (loading…)</option>}
          {materials.map((item) => (
            <option key={item.name} value={item.name}>
              {item.name} - {item.full_name}
            </option>
          ))}
        </select>
      </label>
      <div className="stats-card">
        <span>Liquidus <strong>{material?.liquidus_temp ?? '--'} C</strong></span>
        <span>Solidus <strong>{material?.solidus_temp ?? '--'} C</strong></span>
        <span>Shrinkage <strong>{material?.volumetric_shrinkage ?? '--'}%</strong></span>
        <span>Niyama <strong>{material?.niyama_threshold ?? '--'}</strong></span>
      </div>
      <p className="muted">
        Liquidus, solidus, and shrinkage values are shown for reference. Material properties drive the
        solidification model for the selected alloy.
      </p>
    </ConfigDrawer>
  );
}
