import { Modal } from './Modal';
// Single source of truth for the version: the manifest that is already
// bumped on release, rather than a constant that quietly goes stale.
import { version as appVersion } from '../../../package.json';
import { formatNumber } from '../../lib/format';
import { SHORTCUT_GROUPS } from '../../lib/shortcuts';
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
            <li>The envelope is an isosurface drawn around everything above the threshold; the casting itself stays a neutral ghost behind it.</li>
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
          {hardware.core_tiers?.length ? (
            <>
              <dt>Core tiers</dt>
              <dd>
                {hardware.core_tiers.map((tier) => `${tier.cores} ${tier.name}`).join(' / ')}
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

export function ShortcutsModal({ onClose }: { onClose: () => void }) {
  return (
    <Modal
      title="Keyboard Shortcuts"
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
      <div className="property-sheet">
        {SHORTCUT_GROUPS.map((group) => (
          <div key={group.group}>
            <p className="property-section-title">{group.group}</p>
            {group.items.map((item) => (
              <div className="property-row" key={item.keys}>
                <span className="property-label">{item.description}</span>
                <span className="property-value">{item.keys}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
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
          <dd>{appVersion}</dd>
          <dt>Interface</dt>
          <dd>Web</dd>
        </dl>
      </div>
    </Modal>
  );
}
