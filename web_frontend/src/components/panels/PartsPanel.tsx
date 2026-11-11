import { useRef, useState } from 'react';
import { Copy, Crosshair, Eye, EyeOff, Trash2, Upload } from 'lucide-react';

import { Group, SetupPanel } from '../SetupPanel';
import { PART_TYPES, partTypeColor, supportsOpenTop } from '../../lib/partTypes';
import type { PartType, PourLocationPayload, ProjectPart, RiserSleeve } from '../../lib/api';
import type { ViewportTool } from '../../lib/viewState';

export type PartsPanelProps = {
  parts: ProjectPart[];
  selectedPartId: string | null;
  onSelectPart: (partId: string | null) => void;
  onAddFiles: (files: File[]) => void;
  onUpdatePart: (partId: string, patch: Partial<ProjectPart>) => void;
  onRemovePart: (partId: string) => void;
  onDuplicatePart: (partId: string) => void;
  onClearAll: () => void;
  onArrangeGrid: () => void;
  onArrangeLine: () => void;
  onClose: () => void;
  pourLocation: PourLocationPayload | null;
  tool: ViewportTool;
  onSetTool: (tool: ViewportTool) => void;
  onClearPourLocation: () => void;
  uploading: boolean;
};

const ACCEPT = '.stl,.step,.stp';

export function PartsPanel(props: PartsPanelProps) {
  const {
    parts,
    selectedPartId,
    onSelectPart,
    onAddFiles,
    onUpdatePart,
    onRemovePart,
    onDuplicatePart,
    onClearAll,
    onArrangeGrid,
    onArrangeLine,
    onClose,
    pourLocation,
    tool,
    onSetTool,
    onClearPourLocation,
    uploading
  } = props;
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const selected = parts.find((part) => part.id === selectedPartId) ?? null;

  return (
    <SetupPanel title="Parts" onClose={onClose}>
      <div
        className={dragOver ? 'dropzone is-over' : 'dropzone'}
        onClick={() => fileRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragOver(false);
          onAddFiles(Array.from(event.dataTransfer.files));
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            fileRef.current?.click();
          }
        }}
      >
        <Upload size={17} />
        <span>{uploading ? 'Uploading...' : 'Drop CAD files or click to browse'}</span>
        <span className="dropzone-hint">STL, STEP or STP. A multi-body STEP becomes one part per body.</span>
        <input
          ref={fileRef}
          type="file"
          accept={ACCEPT}
          multiple
          className="visually-hidden"
          onChange={(event) => {
            onAddFiles(Array.from(event.target.files ?? []));
            event.target.value = '';
          }}
        />
      </div>

      <Group
        title={`Assembly (${parts.length})`}
        action={
          parts.length ? (
            <button type="button" className="btn btn-ghost btn-sm" onClick={onClearAll}>
              Clear all
            </button>
          ) : null
        }
      >
        {parts.length === 0 ? (
          <p className="status-note">
            No parts yet. Add the casting first, then risers, chills and the gating system. Part types are
            detected from the file name and can be changed here.
          </p>
        ) : (
          <div className="parts-table-wrap">
            <table className="table parts-table">
              <thead>
                <tr>
                  <th aria-label="Type colour" />
                  <th>Name</th>
                  <th className="part-type-cell">Type</th>
                  <th className="part-actions-cell" aria-label="Actions" />
                </tr>
              </thead>
              <tbody>
                {parts.map((part) => (
                  <PartRow
                    key={part.id}
                    part={part}
                    selected={part.id === selectedPartId}
                    onSelect={() => onSelectPart(part.id)}
                    onUpdate={(patch) => onUpdatePart(part.id, patch)}
                    onRemove={() => onRemovePart(part.id)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Group>

      {parts.length > 0 ? (
        <Group title="Arrange">
          <div className="row row-wrap">
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => selected && onDuplicatePart(selected.id)}
              disabled={!selected}
              title="Copy the selected part, offset 2 inches in X"
            >
              <Copy size={12} />
              Duplicate
            </button>
            <button type="button" className="btn btn-sm" onClick={onArrangeGrid}>
              Arrange grid
            </button>
            <button type="button" className="btn btn-sm" onClick={onArrangeLine}>
              Arrange line
            </button>
          </div>
        </Group>
      ) : null}

      {selected ? <PartTransform part={selected} onUpdate={(patch) => onUpdatePart(selected.id, patch)} /> : null}

      <Group title="Pour location">
        <p className="field-hint">
          The mold-filling analysis needs to know where metal enters the mold. Pick the point on the assembly
          where the stream lands.
        </p>
        {pourLocation ? (
          <div className="status-note status-note-success">
            Set at{' '}
            <span className="mono">
              {pourLocation.point_in.map((value) => value.toFixed(2)).join(', ')}
            </span>{' '}
            in
          </div>
        ) : (
          <div className="status-note">Not set. Mold filling stays unavailable until one is picked.</div>
        )}
        <div className="row">
          <button
            type="button"
            className={tool === 'pour' ? 'btn btn-sm btn-primary' : 'btn btn-sm'}
            onClick={() => onSetTool(tool === 'pour' ? 'select' : 'pour')}
            disabled={!parts.length}
          >
            <Crosshair size={12} />
            {tool === 'pour' ? 'Click the model...' : 'Set pour location'}
          </button>
          <button type="button" className="btn btn-sm" onClick={onClearPourLocation} disabled={!pourLocation}>
            Clear
          </button>
        </div>
      </Group>
    </SetupPanel>
  );
}

function PartRow({
  part,
  selected,
  onSelect,
  onUpdate,
  onRemove
}: {
  part: ProjectPart;
  selected: boolean;
  onSelect: () => void;
  onUpdate: (patch: Partial<ProjectPart>) => void;
  onRemove: () => void;
}) {
  return (
    <>
      <tr
        aria-selected={selected}
        className={part.visible ? undefined : 'part-row-hidden'}
        onMouseDown={onSelect}
      >
        <td>
          <span className="swatch" style={{ background: partTypeColor(part.part_type) }} aria-hidden="true" />
        </td>
        <td>
          <input
            className="part-name-input"
            value={part.name}
            aria-label={`Name of ${part.name}`}
            title={part.name}
            onChange={(event) => onUpdate({ name: event.target.value })}
          />
        </td>
        <td className="part-type-cell">
          <select
            className="select part-type-select"
            value={part.part_type}
            aria-label={`Type of ${part.name}`}
            onChange={(event) => onUpdate({ part_type: event.target.value as PartType })}
          >
            {PART_TYPES.map((type) => (
              <option key={type.id} value={type.id} title={type.hint}>
                {type.label}
              </option>
            ))}
          </select>
        </td>
        <td className="part-actions-cell">
          <button
            type="button"
            className="btn btn-ghost btn-icon"
            title={part.visible ? 'Hide part' : 'Show part'}
            aria-label={part.visible ? `Hide ${part.name}` : `Show ${part.name}`}
            onClick={() => onUpdate({ visible: !part.visible })}
          >
            {part.visible ? <Eye size={13} /> : <EyeOff size={13} />}
          </button>
          <button
            type="button"
            className="btn btn-ghost btn-icon"
            title="Remove part"
            aria-label={`Remove ${part.name}`}
            onClick={onRemove}
          >
            <Trash2 size={13} />
          </button>
        </td>
      </tr>
      {supportsOpenTop(part.part_type) ? (
        <tr aria-selected={selected} className={part.visible ? undefined : 'part-row-hidden'}>
          <td />
          <td colSpan={3} style={{ paddingTop: 0 }}>
            <label className="check">
              <input
                type="checkbox"
                checked={part.is_open_top}
                onChange={(event) => onUpdate({ is_open_top: event.target.checked })}
              />
              <span className="check-body">
                <span>Open top</span>
                <span className="check-note">Open to the air at the top of the mold and vents during filling. A blind riser gets the Mold panel's sand cover.</span>
              </span>
            </label>
            <label className="field">
              <span className="field-label">Sleeve</span>
              <select
                className="select"
                value={part.riser_sleeve}
                aria-label={`Sleeve of ${part.name}`}
                onChange={(event) => onUpdate({ riser_sleeve: event.target.value as RiserSleeve })}
              >
                <option value="insulating">Insulating</option>
                <option value="none">None (bare sand)</option>
              </select>
              <span className="field-hint">An insulating sleeve cuts the riser off from the mold and core sand: it cools only through the metal it feeds, and through an open top. A bare sand riser cools through the same sand contact as the casting.</span>
            </label>
          </td>
        </tr>
      ) : null}
    </>
  );
}

function PartTransform({ part, onUpdate }: { part: ProjectPart; onUpdate: (patch: Partial<ProjectPart>) => void }) {
  const axes = ['x', 'y', 'z'] as const;
  return (
    <Group title={`Transform — ${part.name}`}>
      <div className="field">
        <span className="field-label">
          Position <span className="field-unit">inches</span>
        </span>
        <div className="transform-grid">
          {axes.map((axis) => (
            <label className="transform-axis-field" key={`p-${axis}`}>
              <span className="transform-axis">{axis.toUpperCase()}</span>
              <input
                type="number"
                className="input input-num"
                step={0.5}
                min={-1000}
                max={1000}
                value={part[`position_${axis}` as const]}
                aria-label={`Position ${axis.toUpperCase()}`}
                onChange={(event) => onUpdate({ [`position_${axis}`]: Number(event.target.value) })}
              />
            </label>
          ))}
        </div>
      </div>

      <div className="field">
        <span className="field-label">
          Rotation <span className="field-unit">degrees</span>
        </span>
        <div className="transform-grid">
          {axes.map((axis) => (
            <label className="transform-axis-field" key={`r-${axis}`}>
              <span className="transform-axis">{axis.toUpperCase()}</span>
              <input
                type="number"
                className="input input-num"
                step={5}
                min={-360}
                max={360}
                value={part[`rotation_${axis}` as const]}
                aria-label={`Rotation ${axis.toUpperCase()}`}
                onChange={(event) => onUpdate({ [`rotation_${axis}`]: Number(event.target.value) })}
              />
            </label>
          ))}
        </div>
      </div>

      <label className="field">
        <span className="field-label">
          Scale <span className="field-unit">multiplier</span>
        </span>
        <input
          type="number"
          className="input input-num"
          step={0.05}
          min={0.01}
          max={100}
          value={part.scale}
          onChange={(event) => onUpdate({ scale: Number(event.target.value) || 1 })}
        />
      </label>
    </Group>
  );
}
