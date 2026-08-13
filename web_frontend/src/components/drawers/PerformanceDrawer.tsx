import type { Dispatch, SetStateAction } from 'react';
import { Gauge } from 'lucide-react';

import { ConfigDrawer } from '../ConfigDrawer';
import type { ProjectSettings } from '../../lib/api';

type Props = {
  settings: ProjectSettings;
  setSettings: Dispatch<SetStateAction<ProjectSettings>>;
  onClose: () => void;
};

const MODE_HINTS: Record<string, string> = {
  auto: 'Auto (recommended): Lets the program pick the fastest method for the size of your assembly.',
  parallel:
    'Parallel: Labels voxels using multiple processes at once. Fastest for large assemblies with many parts.',
  legacy: 'Legacy: The original single-process method. Slower, but useful when troubleshooting problems.'
};

export function PerformanceDrawer({ settings, setSettings, onClose }: Props) {
  const perf = settings.performance_options ?? {};
  const mode = String(perf.part_assignment_mode ?? 'auto');

  return (
    <ConfigDrawer title="Performance Options" icon={<Gauge size={16} />} onClose={onClose}>
      <div className="perf-field">
        <label className="check">
          <input
            type="checkbox"
            checked={Boolean(perf.use_incremental)}
            onChange={(e) =>
              setSettings({
                ...settings,
                performance_options: { ...perf, use_incremental: e.target.checked },
              })
            }
          />
          Incremental Solidification
        </label>
        <p className="form-hint">
          Speeds up the solidification calculation by only updating voxels near the freezing
          boundary, instead of the whole model, at every time step. Recommended for large models;
          results are essentially unchanged.
        </p>
      </div>

      <div className="perf-field">
        <label className="check">
          <input
            type="checkbox"
            checked={Boolean(perf.use_chunked_voxelization)}
            onChange={(e) =>
              setSettings({
                ...settings,
                performance_options: { ...perf, use_chunked_voxelization: e.target.checked },
              })
            }
          />
          Chunked Voxelization
        </label>
        <p className="form-hint">
          Builds the voxel grid in smaller chunks to reduce peak memory use. Turn this on for very
          large models that risk running out of RAM; it may add a little time to the meshing stage.
        </p>
      </div>

      <div className="perf-field">
        <label>
          Part Assignment Mode
          <select
            value={mode}
            onChange={(e) =>
              setSettings({
                ...settings,
                performance_options: { ...perf, part_assignment_mode: e.target.value },
              })
            }
          >
            <option value="auto">Auto</option>
            <option value="parallel">Parallel</option>
            <option value="legacy">Legacy</option>
          </select>
        </label>
        <p className="form-hint">
          Controls how each voxel is identified as Cast, Riser, Riser Post, Chill, or Core before
          the physics runs. {MODE_HINTS[mode] ?? MODE_HINTS.auto}
        </p>
      </div>

      <div className="perf-field">
        <label>
          Assignment Workers
          <input
            type="number"
            min={0}
            step={1}
            value={Number(perf.part_assignment_workers ?? 0)}
            onChange={(e) =>
              setSettings({
                ...settings,
                performance_options: { ...perf, part_assignment_workers: Number(e.target.value) },
              })
            }
          />
        </label>
        <p className="form-hint">
          How many processes are used to label parts. 0 = automatically use the available CPU
          cores. Raising this can speed up large assemblies; lowering it frees cores for other work.
        </p>
      </div>

      <div className="perf-field">
        <label>
          Assignment Batch Size
          <input
            type="number"
            min={100}
            step={100}
            value={Number(perf.part_assignment_batch_size ?? 2500)}
            onChange={(e) =>
              setSettings({
                ...settings,
                performance_options: { ...perf, part_assignment_batch_size: Number(e.target.value) },
              })
            }
          />
        </label>
        <p className="form-hint">
          The number of voxel points each process handles at one time. Larger values reduce
          overhead but use more memory; smaller values are gentler on memory. Leave at the default
          unless you hit memory limits.
        </p>
      </div>

      <div className="perf-field">
        <label>
          Max Visualization Voxels
          <input
            type="number"
            min={10000}
            step={10000}
            value={Number(perf.max_voxels_for_visualization ?? 90000)}
            onChange={(e) =>
              setSettings({
                ...settings,
                performance_options: {
                  ...perf,
                  max_voxels_for_visualization: Number(e.target.value),
                },
              })
            }
          />
        </label>
        <p className="form-hint">
          The most voxels the 3D viewer will display. Lower = snappier browser and faster result
          loading; higher = more detailed results on screen. This only affects the display, not the
          simulation itself.
        </p>
      </div>
    </ConfigDrawer>
  );
}
