import { useState } from 'react';
import { Sparkles } from 'lucide-react';

import { Modal } from './Modal';
import { formatCount, formatNumber } from '../../lib/format';
import type { RiserDesignResult, RiserRecommendation } from '../../lib/api';

export type RiserDesignModalProps = {
  result: RiserDesignResult | null;
  analyzing: boolean;
  error: string | null;
  onAnalyze: (params: { max_risers: number; modulus_factor: number; feeding_constant: number }) => void;
  onApply: (risers: RiserRecommendation[]) => void;
  applying: boolean;
  onClose: () => void;
};

/**
 * Automatic riser design, matching the desktop tool.
 *
 * The engine proposes; the user disposes — every recommendation is a checkbox,
 * because the modulus method is a starting point and a foundry engineer will
 * always want to drop the one sitting over a core print.
 */
export function RiserDesignModal({
  result,
  analyzing,
  error,
  onAnalyze,
  onApply,
  applying,
  onClose
}: RiserDesignModalProps) {
  const [maxRisers, setMaxRisers] = useState(5);
  const [modulusFactor, setModulusFactor] = useState(1.2);
  const [feedingConstant, setFeedingConstant] = useState(4.5);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [detailIndex, setDetailIndex] = useState<number | null>(null);

  const recommendations = result?.recommendations ?? [];
  const detail = recommendations.find((item) => item.index === detailIndex) ?? recommendations[0] ?? null;

  const analyze = () => {
    setSelected(new Set());
    setDetailIndex(null);
    onAnalyze({
      max_risers: maxRisers,
      modulus_factor: modulusFactor,
      feeding_constant: feedingConstant
    });
  };

  const toggle = (index: number) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const chosen = recommendations.filter((item) => selected.has(item.index));

  return (
    <Modal
      title="Automatic Riser Design"
      subtitle="Heuvers circle method over the solved geometry"
      size="lg"
      onClose={onClose}
      footer={
        <>
          <span className="text-faint" style={{ fontSize: 'var(--fs-sm)' }}>
            {chosen.length ? `${chosen.length} selected` : 'Select the risers to add'}
          </span>
          <div className="spacer" />
          <button type="button" className="btn" onClick={onClose}>
            Close
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!chosen.length || applying}
            onClick={() => onApply(chosen)}
          >
            {applying ? 'Adding...' : `Add ${chosen.length || ''} selected riser${chosen.length === 1 ? '' : 's'}`}
          </button>
        </>
      }
    >
      <div className="group" style={{ marginBottom: 'var(--sp-5)' }}>
        <div className="grid-3">
          <label className="field">
            <span className="field-label">
              Max risers <span className="field-unit">1 to 10</span>
            </span>
            <input
              type="number"
              className="input input-num"
              min={1}
              max={10}
              step={1}
              value={maxRisers}
              onChange={(event) => setMaxRisers(Math.max(1, Math.min(10, Number(event.target.value) || 1)))}
            />
          </label>
          <label className="field">
            <span className="field-label">
              Modulus factor <span className="field-unit">1.0 to 2.0</span>
            </span>
            <input
              type="number"
              className="input input-num"
              min={1}
              max={2}
              step={0.1}
              value={modulusFactor}
              onChange={(event) => setModulusFactor(Number(event.target.value) || 1.2)}
            />
          </label>
          <label className="field">
            <span className="field-label">
              Feeding constant <span className="field-unit">2.0 to 6.0</span>
            </span>
            <input
              type="number"
              className="input input-num"
              min={2}
              max={6}
              step={0.5}
              value={feedingConstant}
              onChange={(event) => setFeedingConstant(Number(event.target.value) || 4.5)}
            />
          </label>
        </div>
        <span className="field-hint">
          The riser must solidify last: its modulus is sized at the hot spot's modulus times this factor. The
          feeding constant sets how far one riser reaches — about 4.5 for steel, 3.0 for aluminium.
        </span>
        <button type="button" className="btn btn-primary" onClick={analyze} disabled={analyzing}>
          <Sparkles size={13} />
          {analyzing ? 'Analyzing geometry...' : 'Analyze geometry'}
        </button>
      </div>

      {error ? <p className="status-note status-note-warning">{error}</p> : null}

      {analyzing ? (
        <p className="status-note status-note-info">
          Walking the solved lattice for hot spots. On a large model this takes a few seconds.
        </p>
      ) : null}

      {result && !analyzing ? (
        <>
          <div className="status-note status-note-info">
            Coverage: {(result.coverage.covered_fraction * 100).toFixed(1)}% of the casting fed by{' '}
            {result.coverage.n_risers} riser{result.coverage.n_risers === 1 ? '' : 's'} ·{' '}
            {formatNumber(result.coverage.total_riser_volume_in3, 1)} in³ of riser metal ·{' '}
            {formatCount(result.coverage.uncovered_voxels)} voxels unfed
            {result.merged_duplicates > 0
              ? ` · ${result.merged_duplicates} coincident riser${
                  result.merged_duplicates === 1 ? '' : 's'
                } merged`
              : ''}
          </div>

          {recommendations.length === 0 ? (
            <p className="status-note" style={{ marginTop: 'var(--sp-4)' }}>
              No riser could be placed on this geometry. Try a lower modulus factor.
            </p>
          ) : (
            <div className="riser-layout">
              <div className="parts-table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th aria-label="Selected" />
                      <th>Riser</th>
                      <th className="cell-tight">Ø x H</th>
                      <th className="cell-tight">Fed</th>
                      <th className="cell-tight">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recommendations.map((item) => (
                      <tr
                        key={item.index}
                        aria-selected={detail?.index === item.index}
                        onMouseDown={() => setDetailIndex(item.index)}
                      >
                        <td>
                          <input
                            type="checkbox"
                            checked={selected.has(item.index)}
                            aria-label={`Add riser ${item.index + 1}`}
                            onChange={() => toggle(item.index)}
                          />
                        </td>
                        <td>
                          Riser {item.index + 1}
                          <span className="text-faint mono" style={{ marginLeft: 6 }}>
                            ({item.position.map((value) => value.toFixed(1)).join(', ')})
                          </span>
                        </td>
                        <td className="cell-tight mono">
                          {formatNumber(item.diameter_in, 2)} x {formatNumber(item.height_in, 2)} in
                        </td>
                        <td className="cell-tight mono">{formatCount(item.voxels_fed)}</td>
                        <td className="cell-tight mono">{(item.confidence * 100).toFixed(0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {detail ? (
                <div className="riser-detail">
                  <div className="property-sheet">
                    <p className="property-section-title">Riser {detail.index + 1}</p>
                    <Row label="Type" value={detail.riser_type} />
                    <Row label="Shape" value={detail.shape} />
                    <Row
                      label="Position"
                      value={`${detail.position.map((value) => value.toFixed(3)).join(', ')} in`}
                    />
                    <Row label="Diameter" value={`${formatNumber(detail.diameter_in, 3)} in`} />
                    <Row label="Height" value={`${formatNumber(detail.height_in, 3)} in`} />
                    <Row label="Volume" value={`${formatNumber(detail.volume_in3, 3)} in³`} />
                    <Row label="Modulus" value={`${formatNumber(detail.modulus_in, 4)} in`} />
                    <Row label="Voxels fed" value={formatCount(detail.voxels_fed)} />
                    <Row label="Confidence" value={`${(detail.confidence * 100).toFixed(0)}%`} />
                  </div>
                  <pre className="riser-rationale">{detail.rationale}</pre>
                </div>
              ) : null}
            </div>
          )}
        </>
      ) : null}
    </Modal>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="property-row">
      <span className="property-label">{label}</span>
      <span className="property-value">{value}</span>
    </div>
  );
}
