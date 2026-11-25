import { useState } from 'react';
import { FileDown } from 'lucide-react';

import { Modal } from './Modal';
import type { ReportFormat, ReportSectionToggles } from '../../lib/api';

export type ReportModalProps = {
  formats: ReportFormat[];
  busy: boolean;
  error: string | null;
  canCaptureView: boolean;
  onGenerate: (format: string, sections: ReportSectionToggles) => void;
  onClose: () => void;
};

const SECTION_LABELS: { key: keyof ReportSectionToggles; label: string; note: string }[] = [
  { key: 'settings', label: 'Simulation setup', note: 'Alloy, mesh, analyses run and the pour location.' },
  { key: 'parts', label: 'Parts table', note: 'Every body with its type and transform.' },
  { key: 'statistics', label: 'Summary statistics', note: 'The scalar results the run produced.' },
  { key: 'screenshot', label: 'Result view', note: 'A capture of the viewport as it looks now.' }
];

export function ReportModal({
  formats,
  busy,
  error,
  canCaptureView,
  onGenerate,
  onClose
}: ReportModalProps) {
  const firstAvailable = formats.find((format) => format.available)?.id ?? 'html';
  const [format, setFormat] = useState<string>(firstAvailable);
  const [sections, setSections] = useState<ReportSectionToggles>({
    settings: true,
    parts: true,
    statistics: true,
    screenshot: true
  });

  const selected = formats.find((item) => item.id === format);
  const unavailable = formats.filter((item) => !item.available);

  return (
    <Modal
      title="Generate Report"
      subtitle="Pick what goes in and how it comes out"
      size="sm"
      onClose={onClose}
      footer={
        <>
          <div className="spacer" />
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy || !selected?.available}
            onClick={() => onGenerate(format, sections)}
          >
            <FileDown size={13} />
            {busy ? 'Generating...' : 'Download'}
          </button>
        </>
      }
    >
      <label className="field">
        <span className="field-label">Format</span>
        <select className="select" value={format} onChange={(event) => setFormat(event.target.value)}>
          {formats.map((item) => (
            <option key={item.id} value={item.id} disabled={!item.available}>
              {item.label}
              {item.available ? '' : ' — unavailable'}
            </option>
          ))}
        </select>
      </label>

      <div className="group" style={{ marginTop: 'var(--sp-5)' }}>
        <div className="group-head">
          <span className="group-title">Sections</span>
          <span className="group-rule" />
        </div>
        {SECTION_LABELS.map((entry) => {
          const disabled = entry.key === 'screenshot' && !canCaptureView;
          return (
            <label className="check" key={entry.key}>
              <input
                type="checkbox"
                checked={sections[entry.key] && !disabled}
                disabled={disabled}
                onChange={(event) =>
                  setSections((current) => ({ ...current, [entry.key]: event.target.checked }))
                }
              />
              <span className="check-body">
                <span>{entry.label}</span>
                <span className="check-note">
                  {disabled ? 'Needs a loaded model in the viewport.' : entry.note}
                </span>
              </span>
            </label>
          );
        })}
      </div>

      {unavailable.map((item) => (
        <p className="status-note status-note-warning" key={item.id} style={{ marginTop: 'var(--sp-4)' }}>
          {item.label}: {item.reason}
        </p>
      ))}

      {error ? (
        <p className="status-note status-note-warning" style={{ marginTop: 'var(--sp-4)' }}>
          {error}
        </p>
      ) : null}
    </Modal>
  );
}
