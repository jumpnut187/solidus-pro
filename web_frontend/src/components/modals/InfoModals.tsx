import { Modal } from './Modal';
import { formatNumber } from '../../lib/format';
import type { AlloyGuideRow, HardwareInfo } from '../../lib/api';

export function HelpGuideModal({ alloys, onClose }: { alloys: AlloyGuideRow[]; onClose: () => void }) {
  return (
    <Modal
      title="Help Guide"
      subtitle="Workflow and the alloy reference table"
      size="lg"
      onClose={onClose}
      footer={
        <>
          <div className="spacer" />
          <button type="button" className="btn btn-primary" onClick={onClose}>
            Close
          </button>
        </>
      }
    >
      <div className="prose">
        <div>
          <h3>Running a simulation</h3>
          <ul>
            <li>Add the casting, then its risers, chills, cores and gating system in the Parts panel.</li>
            <li>Confirm each part's type — it decides whether a body is scored for defects or only conducts heat.</li>
            <li>Pick the alloy, then set the mesh. Estimate the mesh before committing to a fine voxel size.</li>
            <li>Choose the analyses you need. Mold filling additionally needs a pour location.</li>
            <li>Run. Results appear in the right-hand panel; time-based modes get the transport bar.</li>
          </ul>
        </div>

        <div>
          <h3>Reading the results</h3>
          <ul>
            <li>Surface paints the field on the casting; Envelope draws an isosurface around everything above the threshold.</li>
            <li>The threshold slider carries a marker at the recommended value for the current mode.</li>
            <li>Section cuts on X, Y and Z show the interior; enable caps to paint the field on the cut face.</li>
          </ul>
        </div>

        <div>
          <h3>Alloys</h3>
          <div style={{ overflowX: 'auto' }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Alloy</th>
                  <th>Liquidus</th>
                  <th>Solidus</th>
                  <th>Range</th>
                  <th>Pour</th>
                  <th>Shrink</th>
                  <th>Feed x</th>
                  <th>Niyama</th>
                </tr>
              </thead>
              <tbody>
                {alloys.map((alloy) => (
                  <tr key={alloy.name}>
                    <td title={alloy.full_name}>
                      <strong>{alloy.name}</strong>
                    </td>
                    <td className="mono">{formatNumber(alloy.liquidus_temp, 0)} °C</td>
                    <td className="mono">{formatNumber(alloy.solidus_temp, 0)} °C</td>
                    <td className="mono">{formatNumber(alloy.freezing_range, 0)} °C</td>
                    <td className="mono">{formatNumber(alloy.pouring_temp, 0)} °C</td>
                    <td className="mono">{formatNumber(alloy.volumetric_shrinkage, 2)} %</td>
                    <td className="mono">{formatNumber(alloy.feeding_multiplier, 1)}</td>
                    <td className="mono">{formatNumber(alloy.niyama_threshold, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {alloys.length === 0 ? <p className="status-note">Loading alloy data...</p> : null}
        </div>
      </div>
    </Modal>
  );
}

export function HardwareInfoModal({ hardware, onClose }: { hardware: HardwareInfo | null; onClose: () => void }) {
  return (
    <Modal
      title="Hardware Info"
      subtitle="What the solver detected on this machine"
      size="sm"
      onClose={onClose}
      footer={
        <>
          <div className="spacer" />
          <button type="button" className="btn btn-primary" onClick={onClose}>
            Close
          </button>
        </>
      }
    >
      {hardware ? (
        <dl className="kv-grid">
          <dt>CPU</dt>
          <dd>{hardware.cpu_brand}</dd>
          <dt>Vendor</dt>
          <dd>{hardware.cpu_vendor}</dd>
          <dt>Physical cores</dt>
          <dd>{hardware.physical_cores ?? '--'}</dd>
          <dt>Logical cores</dt>
          <dd>{hardware.logical_cores ?? '--'}</dd>
          {hardware.performance_cores !== null ? (
            <>
              <dt>Core split</dt>
              <dd>
                {hardware.performance_cores} P / {hardware.efficiency_cores ?? '--'} E
              </dd>
            </>
          ) : null}
          <dt>Memory</dt>
          <dd>
            {hardware.available_memory_gb ?? '--'} GB free / {hardware.total_memory_gb ?? '--'} GB
          </dd>
          <dt>Numba threads</dt>
          <dd>{hardware.numba_threads ?? '--'}</dd>
          <dt>GPU backend</dt>
          <dd>{hardware.gpu_backend}</dd>
          <dt>Neural engine</dt>
          <dd>{hardware.neural_engine_available ? 'available' : 'not available'}</dd>
        </dl>
      ) : (
        <p className="status-note">Reading hardware information...</p>
      )}
      {hardware?.detection_error ? (
        <p className="status-note status-note-warning" style={{ marginTop: 'var(--sp-4)' }}>
          Detection was incomplete: {hardware.detection_error}
        </p>
      ) : null}
    </Modal>
  );
}

export function AboutModal({ onClose }: { onClose: () => void }) {
  return (
    <Modal
      title="About Solidus Pro"
      size="sm"
      onClose={onClose}
      footer={
        <>
          <div className="spacer" />
          <button type="button" className="btn btn-primary" onClick={onClose}>
            Close
          </button>
        </>
      }
    >
      <div className="prose">
        <p>
          Solidus Pro is a casting solidification and mold-filling simulator: voxel thermal solve, feeding and
          defect prediction, and a lattice-Boltzmann filling stage, with results rendered the way
          industry-standard casting simulators render them.
        </p>
        <dl className="kv-grid">
          <dt>Version</dt>
          <dd>2.0</dd>
          <dt>Interface</dt>
          <dd>Web</dd>
        </dl>
      </div>
    </Modal>
  );
}
