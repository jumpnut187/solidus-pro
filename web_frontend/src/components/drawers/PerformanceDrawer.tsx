import type { Dispatch, SetStateAction } from 'react';
import { Gauge } from 'lucide-react';

import { ConfigDrawer } from '../ConfigDrawer';
import type { ProjectSettings } from '../../lib/api';

type Props = {
  settings: ProjectSettings;
  setSettings: Dispatch<SetStateAction<ProjectSettings>>;
  onClose: () => void;
};

export function PerformanceDrawer({ settings, setSettings, onClose }: Props) {
  return (
    <ConfigDrawer title="Performance Options" icon={<Gauge size={16} />} onClose={onClose}>
      <label className="check">
        <input
          type="checkbox"
          checked={Boolean(settings.performance_options?.use_incremental)}
          onChange={(e) =>
            setSettings({
              ...settings,
              performance_options: { ...settings.performance_options, use_incremental: e.target.checked },
            })
          }
        />
        Incremental Solidification
      </label>
      <label className="check">
        <input
          type="checkbox"
          checked={Boolean(settings.performance_options?.use_chunked_voxelization)}
          onChange={(e) =>
            setSettings({
              ...settings,
              performance_options: { ...settings.performance_options, use_chunked_voxelization: e.target.checked },
            })
          }
        />
        Chunked Voxelization
      </label>
      <label>
        Part Assignment Mode
        <select
          value={String(settings.performance_options?.part_assignment_mode ?? 'auto')}
          onChange={(e) =>
            setSettings({
              ...settings,
              performance_options: { ...settings.performance_options, part_assignment_mode: e.target.value },
            })
          }
        >
          <option value="auto">Auto</option>
          <option value="parallel">Parallel</option>
          <option value="legacy">Legacy</option>
        </select>
      </label>
      <label>
        Assignment Workers
        <input
          type="number"
          min={0}
          step={1}
          value={Number(settings.performance_options?.part_assignment_workers ?? 0)}
          onChange={(e) =>
            setSettings({
              ...settings,
              performance_options: { ...settings.performance_options, part_assignment_workers: Number(e.target.value) },
            })
          }
        />
      </label>
      <label>
        Assignment Batch Size
        <input
          type="number"
          min={100}
          step={100}
          value={Number(settings.performance_options?.part_assignment_batch_size ?? 2500)}
          onChange={(e) =>
            setSettings({
              ...settings,
              performance_options: { ...settings.performance_options, part_assignment_batch_size: Number(e.target.value) },
            })
          }
        />
      </label>
      <label>
        Max Visualization Voxels
        <input
          type="number"
          min={10000}
          step={10000}
          value={Number(settings.performance_options?.max_voxels_for_visualization ?? 90000)}
          onChange={(e) =>
            setSettings({
              ...settings,
              performance_options: {
                ...settings.performance_options,
                max_voxels_for_visualization: Number(e.target.value),
              },
            })
          }
        />
      </label>
    </ConfigDrawer>
  );
}
