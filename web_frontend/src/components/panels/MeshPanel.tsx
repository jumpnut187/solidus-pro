import { Group, SetupPanel } from '../SetupPanel';
import { meshDefaults } from '../../lib/defaults';
import type { ProjectSettings } from '../../lib/api';

export type MeshStatus = { text: string; tone: 'info' | 'warning' | 'success' } | null;

export type MeshPanelProps = {
  settings: ProjectSettings;
  onPatch: (patch: Partial<ProjectSettings>) => void;
  onEstimate: () => void;
  onApplyMlMesh: () => void;
  estimate: MeshStatus;
  mlStatus: MeshStatus;
  busy: boolean;
  hasParts: boolean;
  onReset: () => void;
  onClose: () => void;
};

function note(status: MeshStatus, fallback: string) {
  if (!status) return <p className="status-note">{fallback}</p>;
  return <p className={`status-note status-note-${status.tone}`}>{status.text}</p>;
}

export function MeshPanel({
  settings,
  onPatch,
  onEstimate,
  onApplyMlMesh,
  estimate,
  mlStatus,
  busy,
  hasParts,
  onReset,
  onClose
}: MeshPanelProps) {
  return (
    <SetupPanel title="Mesh" onClose={onClose} onReset={onReset}>
      <Group title="Resolution">
        <label className="field">
          <span className="field-label">
            Base voxel size
            <span className="field-unit">{settings.voxel_size.toFixed(3)} in</span>
          </span>
          <input
            type="range"
            className="range"
            min={meshDefaults.voxelSizeMin}
            max={meshDefaults.voxelSizeMax}
            step={meshDefaults.voxelSizeStep}
            value={settings.voxel_size}
            aria-label="Base voxel size"
            onChange={(event) => onPatch({ voxel_size: Number(event.target.value) })}
          />
          <span className="field-hint">
            Smaller voxels resolve thin sections and gating channels but cost memory and time. Halving the
            size multiplies the voxel count by roughly eight.
          </span>
        </label>

        <label className="field">
          <span className="field-label">
            Refinement level
            <span className="field-unit">{settings.refinement_level}</span>
          </span>
          <input
            type="range"
            className="range"
            min={meshDefaults.refinementMin}
            max={meshDefaults.refinementMax}
            step={1}
            value={settings.refinement_level}
            aria-label="Refinement level"
            onChange={(event) => onPatch({ refinement_level: Number(event.target.value) })}
          />
          <span className="field-hint">How aggressively the mesh subdivides around surfaces and thin walls.</span>
        </label>

        <label className="field">
          <span className="field-label">Memory mode</span>
          <select
            className="select"
            value={settings.memory_mode}
            onChange={(event) => onPatch({ memory_mode: event.target.value })}
          >
            <option value="auto">Auto</option>
            <option value="ram">RAM (fastest)</option>
            <option value="disk">Disk (largest models)</option>
          </select>
        </label>
      </Group>

      <Group title="Estimate">
        <button type="button" className="btn btn-block" onClick={onEstimate} disabled={!hasParts || busy}>
          Estimate mesh
        </button>
        {note(estimate, hasParts ? 'No estimate yet.' : 'Add a part to estimate the mesh.')}
      </Group>

      <Group title="Machine-learned mesh">
        <button type="button" className="btn btn-block" onClick={onApplyMlMesh} disabled={!hasParts || busy}>
          Apply ML mesh
        </button>
        {note(
          mlStatus,
          hasParts
            ? 'Suggests a voxel size and refinement level from the geometry.'
            : 'Add a part to request a recommendation.'
        )}
      </Group>
    </SetupPanel>
  );
}
