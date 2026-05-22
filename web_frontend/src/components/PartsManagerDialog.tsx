import { useRef } from 'react';
import {
  Box,
  Copy,
  Grid3X3,
  Layers3,
  ListStart,
  Plus,
  Trash2,
  X
} from 'lucide-react';
import type { ProjectPart } from '../lib/api';
import { DraggableDialog } from './DraggableDialog';

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
  onAddParts: (files: FileList) => void;
  onRemovePart: (id: string) => void;
  onClearAll: () => void;
  onDuplicatePart: (id: string) => void;
  onArrangeGrid: () => void;
  onArrangeLine: () => void;
  onClose: () => void;
};

export function PartsManagerDialog({
  parts,
  selectedPartId,
  onSelectPart,
  onUpdatePart,
  onAddParts,
  onRemovePart,
  onClearAll,
  onDuplicatePart,
  onArrangeGrid,
  onArrangeLine,
  onClose
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const selectedPart = parts.find((p) => p.id === selectedPartId) ?? null;

  const totalVolume = parts.reduce((sum, p) => sum + (p.meshData?.volume ?? 0), 0);

  function handleAddClick() {
    fileInputRef.current?.click();
  }

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const files = event.target.files;
    if (files && files.length) {
      onAddParts(files);
    }
    event.target.value = '';
  }

  function handleRemoveClick() {
    if (selectedPartId) {
      onRemovePart(selectedPartId);
    }
  }

  function handleDuplicateClick() {
    if (selectedPartId) {
      onDuplicatePart(selectedPartId);
    }
  }

  return (
    <DraggableDialog title="Parts Manager" initialX={120} initialY={90} onClose={onClose}>
      <div className="dialog-toolbar">
        <button className="dialog-toolbtn" onClick={handleAddClick}>
          <Plus size={14} /> Add Part
        </button>
        <button className="dialog-toolbtn" disabled={!selectedPartId} onClick={handleRemoveClick}>
          <Trash2 size={14} /> Remove
        </button>
        <button className="dialog-toolbtn" disabled={!parts.length} onClick={onClearAll}>
          <X size={14} /> Clear All
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".stl"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
      </div>

      <div className="parts-table-wrap">
        <table className="parts-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Open Top</th>
              <th style={{ width: 40 }} />
            </tr>
          </thead>
          <tbody>
            {parts.length === 0 && (
              <tr>
                <td colSpan={4} className="parts-empty">
                  <Box size={28} />
                  <span>No parts loaded. Click "Add Part" to import STL files.</span>
                </td>
              </tr>
            )}
            {parts.map((part) => {
              const isSelected = part.id === selectedPartId;
              return (
                <tr
                  key={part.id}
                  className={isSelected ? 'selected' : ''}
                  onClick={() => onSelectPart(part.id)}
                >
                  <td className="part-name-cell">
                    <Layers3 size={13} />
                    <span>{part.name}</span>
                  </td>
                  <td>
                    <select
                      value={part.part_type}
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
                  </td>
                  <td style={{ textAlign: 'center' }}>
                    <input
                      type="checkbox"
                      checked={part.is_open_top}
                      disabled={part.part_type !== 'riser'}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => onUpdatePart({ id: part.id, is_open_top: e.target.checked })}
                    />
                  </td>
                  <td>
                    <button
                      className="dialog-iconbtn"
                      title="Remove part"
                      onClick={(e) => {
                        e.stopPropagation();
                        onRemovePart(part.id);
                      }}
                    >
                      <Trash2 size={13} />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {selectedPart && (
        <div className="dialog-section">
          <strong className="dialog-section-title">Selected Part</strong>
          <div className="dialog-form">
            <label>
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
                onChange={(e) =>
                  onUpdatePart({ id: selectedPart.id, position_x: Number(e.target.value) })
                }
              />
            </label>
            <label>
              Y
              <input
                type="number"
                step={0.1}
                value={selectedPart.position_y}
                onChange={(e) =>
                  onUpdatePart({ id: selectedPart.id, position_y: Number(e.target.value) })
                }
              />
            </label>
            <label>
              Z
              <input
                type="number"
                step={0.1}
                value={selectedPart.position_z}
                onChange={(e) =>
                  onUpdatePart({ id: selectedPart.id, position_z: Number(e.target.value) })
                }
              />
            </label>
            <label>
              Rx
              <input
                type="number"
                step={1}
                value={selectedPart.rotation_x}
                onChange={(e) =>
                  onUpdatePart({ id: selectedPart.id, rotation_x: Number(e.target.value) })
                }
              />
            </label>
            <label>
              Ry
              <input
                type="number"
                step={1}
                value={selectedPart.rotation_y}
                onChange={(e) =>
                  onUpdatePart({ id: selectedPart.id, rotation_y: Number(e.target.value) })
                }
              />
            </label>
            <label>
              Rz
              <input
                type="number"
                step={1}
                value={selectedPart.rotation_z}
                onChange={(e) =>
                  onUpdatePart({ id: selectedPart.id, rotation_z: Number(e.target.value) })
                }
              />
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={selectedPart.visible}
                onChange={(e) => onUpdatePart({ id: selectedPart.id, visible: e.target.checked })}
              />
              Visible
            </label>
          </div>
        </div>
      )}

      <div className="dialog-toolbar">
        <button className="dialog-toolbtn" disabled={!selectedPartId} onClick={handleDuplicateClick}>
          <Copy size={14} /> Duplicate
        </button>
        <button className="dialog-toolbtn" disabled={parts.length < 2} onClick={onArrangeGrid}>
          <Grid3X3 size={14} /> Arrange Grid
        </button>
        <button className="dialog-toolbtn" disabled={parts.length < 2} onClick={onArrangeLine}>
          <ListStart size={14} /> Arrange Line
        </button>
      </div>

      <div className="dialog-footer">
        <span>
          {parts.length} part{parts.length === 1 ? '' : 's'}
        </span>
        <span>Volume {totalVolume.toFixed(2)} in³</span>
      </div>
    </DraggableDialog>
  );
}
