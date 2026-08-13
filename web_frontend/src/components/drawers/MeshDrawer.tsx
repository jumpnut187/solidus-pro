import type { Dispatch, SetStateAction } from 'react';
import { Wand2 } from 'lucide-react';

import { ConfigDrawer } from '../ConfigDrawer';
import type { ProjectSettings } from '../../lib/api';

type Props = {
  settings: ProjectSettings;
  setSettings: Dispatch<SetStateAction<ProjectSettings>>;
  projectId: string | null;
  meshEstimate: string;
  mlStatus: string;
  updateMeshPreview: () => Promise<void>;
  applyMlMesh: () => Promise<void>;
  onClose: () => void;
};

export function MeshDrawer({
  settings,
  setSettings,
  projectId,
  meshEstimate,
  mlStatus,
  updateMeshPreview,
  applyMlMesh,
  onClose,
}: Props) {
  return (
    <ConfigDrawer title="Mesh Parameters" icon={<span style={{ fontSize: 15 }}>▦</span>} onClose={onClose}>
      <label>
        Base Voxel Size (inches)
        <input
          type="number"
          min="0.04"
          step="0.05"
          value={settings.voxel_size}
          onChange={(e) => setSettings({ ...settings, voxel_size: Number(e.target.value) })}
        />
      </label>
      <label>
        Refinement
        <select
          value={settings.refinement_level}
          onChange={(e) => setSettings({ ...settings, refinement_level: Number(e.target.value) })}
        >
          <option value={1}>1 - Coarse</option>
          <option value={2}>2 - Medium</option>
          <option value={3}>3 - Fine</option>
          <option value={4}>4 - Very fine</option>
        </select>
      </label>
      <label>
        Memory Mode
        <select
          value={settings.memory_mode}
          onChange={(e) => setSettings({ ...settings, memory_mode: e.target.value })}
        >
          <option value="auto">Auto (Recommended)</option>
          <option value="ram">Force RAM</option>
          <option value="disk">Force Disk</option>
        </select>
      </label>
      <button className="ghost-button" onClick={() => void updateMeshPreview()} disabled={!projectId}>
        Estimate Mesh
      </button>
      <p className="muted">{meshEstimate}</p>
      <button className="ghost-button" onClick={() => void applyMlMesh()} disabled={!projectId}>
        <Wand2 size={14} /> Apply Machine Learning Mesh
      </button>
      <p className="muted">{mlStatus}</p>
      <p className="muted">Larger voxel sizes run faster with more memory headroom; refinement adds detail near the mold walls.</p>
    </ConfigDrawer>
  );
}
