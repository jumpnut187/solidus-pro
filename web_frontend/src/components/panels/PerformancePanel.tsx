import { Group, SetupPanel } from '../SetupPanel';
import { formatCount } from '../../lib/format';
import type { HardwareInfo, PerformanceOptions } from '../../lib/api';

export type PerformancePanelProps = {
  options: PerformanceOptions;
  hardware: HardwareInfo | null;
  onSet: (key: string, value: boolean | number | string) => void;
  onReset: () => void;
  onClose: () => void;
};

export function PerformancePanel({ options, hardware, onSet, onReset, onClose }: PerformancePanelProps) {
  const workers = Number(options.part_assignment_workers ?? 0);
  const batchSize = Number(options.part_assignment_batch_size ?? 2500);
  const maxVoxels = Number(options.max_voxels_for_visualization ?? 90000);

  return (
    <SetupPanel title="Performance" onClose={onClose} onReset={onReset}>
      <Group title="Acceleration">
        {hardware ? (
          <p className={hardware.gpu_available ? 'status-note status-note-success' : 'status-note'}>
            {hardware.gpu_available
              ? `GPU acceleration available (${hardware.gpu_backend}).`
              : 'No GPU backend detected. Runs fall back to the CPU solver.'}
          </p>
        ) : null}

        <label className="check">
          <input
            type="checkbox"
            checked={Boolean(options.use_gpu)}
            disabled={hardware ? !hardware.gpu_available : false}
            onChange={(event) => onSet('use_gpu', event.target.checked)}
          />
          <span className="check-body">
            <span>Use GPU</span>
            <span className="check-note">
              Offloads distance fields and the solidification update to the detected GPU backend.
            </span>
          </span>
        </label>

        <label className="check">
          <input
            type="checkbox"
            checked={Boolean(options.use_incremental)}
            onChange={(event) => onSet('use_incremental', event.target.checked)}
          />
          <span className="check-body">
            <span>Incremental solidification</span>
            <span className="check-note">Reuses the previous solve where the geometry did not change.</span>
          </span>
        </label>

        <label className="check">
          <input
            type="checkbox"
            checked={Boolean(options.use_chunked_voxelization)}
            onChange={(event) => onSet('use_chunked_voxelization', event.target.checked)}
          />
          <span className="check-body">
            <span>Chunked voxelization</span>
            <span className="check-note">
              Voxelizes in blocks so a very large assembly does not need the whole grid resident at once.
            </span>
          </span>
        </label>
      </Group>

      <Group title="Part assignment">
        <label className="field">
          <span className="field-label">Mode</span>
          <select
            className="select"
            value={String(options.part_assignment_mode ?? 'parallel')}
            onChange={(event) => onSet('part_assignment_mode', event.target.value)}
          >
            <option value="auto">Auto</option>
            <option value="parallel">Parallel</option>
            <option value="serial">Serial</option>
          </select>
        </label>

        <div className="grid-2">
          <label className="field">
            <span className="field-label">
              Workers <span className="field-unit">0 = auto</span>
            </span>
            <input
              type="number"
              className="input input-num"
              min={0}
              max={64}
              step={1}
              value={workers}
              onChange={(event) => onSet('part_assignment_workers', Math.max(0, Number(event.target.value) || 0))}
            />
          </label>
          <label className="field">
            <span className="field-label">
              Batch size <span className="field-unit">voxels</span>
            </span>
            <input
              type="number"
              className="input input-num"
              min={100}
              max={100000}
              step={100}
              value={batchSize}
              onChange={(event) =>
                onSet('part_assignment_batch_size', Math.max(100, Number(event.target.value) || 2500))
              }
            />
          </label>
        </div>
      </Group>

      <Group title="Stored result detail">
        <label className="field">
          <span className="field-label">
            Voxels kept in the stored result
            <span className="field-unit">{formatCount(maxVoxels)}</span>
          </span>
          <input
            type="range"
            className="range"
            min={20000}
            max={500000}
            step={10000}
            value={maxVoxels}
            aria-label="Voxels kept in the stored result"
            onChange={(event) => onSet('max_voxels_for_visualization', Number(event.target.value))}
          />
          <span className="field-hint">
            A run with more voxels than this is stored strided down to it. The 3D view is not affected — it
            reads the full-resolution grid — but riser design works from the stored result and refuses a
            downsampled one, so raise this before a run you intend to design risers from.
          </span>
        </label>
      </Group>

      <Group title="Hardware">
        {hardware ? (
          <dl className="kv-grid">
            <dt>CPU</dt>
            <dd>{hardware.cpu_brand}</dd>
            <dt>Cores</dt>
            <dd>
              {hardware.physical_cores ?? '--'} physical / {hardware.logical_cores ?? '--'} logical
            </dd>
            {hardware.core_tiers?.length ? (
              <>
                <dt>Tiers</dt>
                <dd>{hardware.core_tiers.map((tier) => `${tier.cores} ${tier.name}`).join(' / ')}</dd>
              </>
            ) : null}
            <dt>Memory</dt>
            <dd>
              {hardware.available_memory_gb ?? '--'} GB free of {hardware.total_memory_gb ?? '--'} GB
            </dd>
            <dt>GPU backend</dt>
            <dd>{hardware.gpu_backend}</dd>
            <dt>Numba threads</dt>
            <dd>{hardware.numba_threads ?? '--'}</dd>
          </dl>
        ) : (
          <p className="status-note">Reading hardware information...</p>
        )}
      </Group>

      <p className="field-hint">
        Keep the defaults unless you are working with very large models (over five million voxels).
      </p>
    </SetupPanel>
  );
}
