import { Modal } from './Modal';
import { buildStatGroups } from '../../lib/statLabels';
import type { GridMeta } from '../../lib/viewState';

export function StatisticsModal({ meta, onClose }: { meta: GridMeta | null; onClose: () => void }) {
  const groups = meta
    ? buildStatGroups(meta.stats, {
        voxelCount: meta.voxelCount,
        castVoxelCount: meta.castVoxelCount,
        totalVolumeIn3: meta.totalVolumeIn3,
        spacing: meta.spacing
      })
    : [];

  return (
    <Modal
      title="Run Statistics"
      subtitle={meta ? `${meta.dims.join(' x ')} lattice at ${meta.spacing.toFixed(3)} in` : undefined}
      size="md"
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
      {groups.length === 0 ? (
        <p className="status-note">No statistics yet. Run a simulation first.</p>
      ) : (
        <div className="property-sheet">
          {groups.map((group) => (
            <div key={group.label}>
              <p className="property-section-title">{group.label}</p>
              {group.rows.map((row) => (
                <div className="property-row" key={row.key}>
                  <span className="property-label" title={row.hint}>
                    {row.label}
                  </span>
                  <span className="property-value">{row.value}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {meta?.partTypes?.length ? (
        <div style={{ marginTop: 'var(--sp-5)' }}>
          <p className="property-section-title" style={{ borderRadius: 'var(--radius) var(--radius) 0 0' }}>
            Voxels by part type
          </p>
          <div className="property-sheet" style={{ borderTopLeftRadius: 0, borderTopRightRadius: 0 }}>
            {meta.partTypes.map((partType) => (
              <div className="property-row" key={partType.id}>
                <span className="property-label">
                  <span
                    className="swatch"
                    style={{
                      background: `rgb(${partType.color
                        .map((channel) => Math.round(channel * 255))
                        .join(', ')})`,
                      marginRight: 'var(--sp-3)'
                    }}
                    aria-hidden="true"
                  />
                  {partType.label}
                </span>
                <span className="property-value">{partType.voxelCount.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </Modal>
  );
}
