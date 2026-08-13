import { useRef } from 'react';
import { Box, Copy, Layers3, Plus, Trash2, Upload, X } from 'lucide-react';

import { ConfigDrawer } from '../ConfigDrawer';
import type { ProjectPart } from '../../lib/api';

const TYPE_OPTIONS: { value: ProjectPart['part_type']; label: string }[] = [
  { value: 'cast', label: 'Cast' },
  { value: 'riser', label: 'Riser' },
  { value: 'riser_post', label: 'Riser Post' },
  { value: 'chill', label: 'Chill' },
  { value: 'core', label: 'Core' }
];

type Props = {
  parts: ProjectPart[];
  selectedPartId: string | null;
  onSelectPart: (id: string | null) => void;
  onUpdatePart: (patch: Partial<ProjectPart>) => void;
  onAddParts: (files: FileList | File[]) => void;
  onRemovePart: (id: string) => void;
  onClearAll: () => void;
  onDuplicatePart: (id: string) => void;
  onClose: () => void;
};

export function PartsDrawer({
  parts,
  selectedPartId,
  onSelectPart,
  onUpdatePart,
  onAddParts,
  onRemovePart,
  onClearAll,
  onDuplicatePart,
  onClose
}: Props) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const selectedPart = parts.find((p) => p.id === selectedPartId) ?? null;
  const totalVolume = parts.reduce((sum, p) => sum + (p.meshData?.volume ?? 0), 0);

  return (
    <ConfigDrawer title="Parts Manager" icon={<Layers3 size={16} />} onClose={onClose}>
      <div
        className="drop-zone"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          onAddParts(event.dataTransfer.files);
        }}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInputRef.current?.click();
          }
        }}
      >
        <Upload size={20} />
        <span>Drop STL/STEP files here, or click to browse</span>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".stl,.step,.stp"
          style={{ display: 'none' }}
          onChange={(event) => {
            if (event.target.files) onAddParts(event.target.files);
            event.target.value = '';
          }}
        />
      </div>

      {parts.length === 0 ? (
        <p className="parts-meta">No parts loaded. Upload one or more STL/STEP files to begin.</p>
      ) : (
        <>
          <div className="parts-table">
            <div className="parts-table-header">
              <span className="parts-header-name">Part Name</span>
              <span className="parts-header-type">Part Type</span>
              <span className="parts-header-open-top">Open Top</span>
              <span className="parts-header-spacer" />
            </div>
            {parts.map((part) => (
              <div
                key={part.id}
                className={part.id === selectedPartId ? 'parts-row parts-row-selected' : 'parts-row'}
                onClick={() => onSelectPart(part.id)}
              >
                <span className="parts-name" title={part.name}>
                  {part.name}
                </span>
                <select
                  value={part.part_type}
                  title={
                    'Cast: Actual casting - analyzed for shrinkage/porosity\n' +
                    'Riser: Feeder with exothermic effect - slows cooling\n' +
                    'Riser Post: Riser neck/post connecting riser to casting\n' +
                    'Chill: External chill - accelerates cooling\n' +
                    'Core: Sand core - excluded from defect analysis'
                  }
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) =>
                    onUpdatePart({ id: part.id, part_type: e.target.value as ProjectPart['part_type'] })
                  }
                >
                  {TYPE_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
                <input
                  type="checkbox"
                  title={
                    'Open Top Riser: Atmospheric pressure pushes metal down for better feeding.\n' +
                    'Blind Riser (unchecked): Enclosed, may form vacuum reducing feeding by ~20%.'
                  }
                  checked={part.is_open_top}
                  disabled={part.part_type !== 'riser'}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => onUpdatePart({ id: part.id, is_open_top: e.target.checked })}
                />
                <button
                  type="button"
                  className="ghost-button"
                  title="Remove part"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemovePart(part.id);
                  }}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        </>
      )}

      {selectedPart && (
        <>
          <div className="group-title">Selected Part</div>
          <div className="drawer-form-grid">
            <label className="form-span-2">
              Name
              <input
                value={selectedPart.name}
                onChange={(e) => onUpdatePart({ id: selectedPart.id, name: e.target.value })}
              />
            </label>
            <label>
              Scale
              <input
                type="number"
                min={0.05}
                step={0.05}
                value={selectedPart.scale}
                onChange={(e) => onUpdatePart({ id: selectedPart.id, scale: Number(e.target.value) })}
              />
            </label>
            <label>
              X
              <input
                type="number"
                step={0.1}
                value={selectedPart.position_x}
                onChange={(e) => onUpdatePart({ id: selectedPart.id, position_x: Number(e.target.value) })}
              />
            </label>
            <label>
              Y
              <input
                type="number"
                step={0.1}
                value={selectedPart.position_y}
                onChange={(e) => onUpdatePart({ id: selectedPart.id, position_y: Number(e.target.value) })}
              />
            </label>
            <label>
              Z
              <input
                type="number"
                step={0.1}
                value={selectedPart.position_z}
                onChange={(e) => onUpdatePart({ id: selectedPart.id, position_z: Number(e.target.value) })}
              />
            </label>
            <label>
              Rx
              <input
                type="number"
                step={1}
                value={selectedPart.rotation_x}
                onChange={(e) => onUpdatePart({ id: selectedPart.id, rotation_x: Number(e.target.value) })}
              />
            </label>
            <label>
              Ry
              <input
                type="number"
                step={1}
                value={selectedPart.rotation_y}
                onChange={(e) => onUpdatePart({ id: selectedPart.id, rotation_y: Number(e.target.value) })}
              />
            </label>
            <label>
              Rz
              <input
                type="number"
                step={1}
                value={selectedPart.rotation_z}
                onChange={(e) => onUpdatePart({ id: selectedPart.id, rotation_z: Number(e.target.value) })}
              />
            </label>
            <label className="form-span-2 check">
              <input
                type="checkbox"
                checked={selectedPart.visible}
                onChange={(e) => onUpdatePart({ id: selectedPart.id, visible: e.target.checked })}
              />
              Visible
            </label>
          </div>
        </>
      )}

      <div className="group-title">Assembly Tools</div>
      <div className="drawer-tool-row">
        <button className="ghost-button" onClick={() => fileInputRef.current?.click()}>
          <Plus size={14} /> Add Part
        </button>
        <button className="ghost-button" disabled={!selectedPartId} onClick={() => onDuplicatePart(selectedPartId!)}>
          <Copy size={14} /> Duplicate
        </button>
        <button className="ghost-button" disabled={!parts.length} onClick={onClearAll}>
          <X size={14} /> Clear All
        </button>
      </div>

      <div className="drawer-footer-meta">
        <span>{parts.length} part{parts.length === 1 ? '' : 's'}</span>
        <span>Volume {totalVolume.toFixed(2)} in³</span>
      </div>
      <p className="parts-meta">
        {parts.length === 0 && <Box size={20} style={{ display: 'block', margin: '0 auto 6px', opacity: 0.6 }} />}
        Click a part to edit its position, rotation, and scale.
      </p>
    </ConfigDrawer>
  );
}
