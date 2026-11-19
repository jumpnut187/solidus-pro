import { useEffect, useRef, useState } from 'react';
import {
  Box,
  Crosshair,
  Eraser,
  Maximize2,
  MousePointer2,
  Ruler,
  ScanEye,
  Target
} from 'lucide-react';

import type { CameraPreset, ViewportTool } from '../lib/viewState';

export type ViewportToolbarProps = {
  tool: ViewportTool;
  onToolChange: (tool: ViewportTool) => void;
  ortho: boolean;
  onOrthoChange: (ortho: boolean) => void;
  onCamera: (preset: CameraPreset) => void;
  onClearMeasurements: () => void;
  /** Disabled until an assembly exists — there is nothing to look at. */
  hasParts: boolean;
};

const TOOLS: { id: ViewportTool; label: string; hint: string; Icon: typeof MousePointer2 }[] = [
  { id: 'select', label: 'Select', hint: 'Select a part', Icon: MousePointer2 },
  { id: 'probe', label: 'Probe', hint: 'Read the field value under the cursor', Icon: Target },
  { id: 'measure', label: 'Measure', hint: 'Measure between two points', Icon: Ruler },
  { id: 'pour', label: 'Pour', hint: 'Set the pour (metal entry) location', Icon: Crosshair }
];

const VIEWS: { id: CameraPreset; label: string }[] = [
  { id: 'iso', label: 'Isometric' },
  { id: 'top', label: 'Top' },
  { id: 'bottom', label: 'Bottom' },
  { id: 'front', label: 'Front' },
  { id: 'back', label: 'Back' },
  { id: 'left', label: 'Left' },
  { id: 'right', label: 'Right' }
];

/**
 * The CAD-style floating toolbar over the canvas.
 *
 * Tools, views and projection used to live at the bottom of the results panel,
 * which put them below the fold at 1366x768 — you had to scroll a settings
 * panel to reach Fit. They belong on the canvas they act on, within a mouse
 * flick of the model.
 */
export function ViewportToolbar({
  tool,
  onToolChange,
  ortho,
  onOrthoChange,
  onCamera,
  onClearMeasurements,
  hasParts
}: ViewportToolbarProps) {
  const [viewsOpen, setViewsOpen] = useState(false);
  const viewsRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!viewsOpen) return;
    function onPointerDown(event: MouseEvent) {
      if (!viewsRef.current?.contains(event.target as Node)) setViewsOpen(false);
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        event.stopPropagation();
        setViewsOpen(false);
      }
    }
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKeyDown, true);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown, true);
    };
  }, [viewsOpen]);

  return (
    <div className="viewport-toolbar" role="toolbar" aria-label="Viewport tools">
      {TOOLS.map(({ id, label, hint, Icon }) => (
        <button
          key={id}
          type="button"
          className="vtool"
          aria-pressed={tool === id}
          aria-label={label}
          title={`${label} — ${hint}`}
          disabled={!hasParts}
          onClick={() => onToolChange(tool === id ? 'select' : id)}
        >
          <Icon size={15} />
        </button>
      ))}

      <span className="vtool-sep" aria-hidden="true" />

      <button
        type="button"
        className="vtool"
        aria-label="Clear measurements"
        title="Clear measurements"
        disabled={!hasParts}
        onClick={onClearMeasurements}
      >
        <Eraser size={15} />
      </button>

      <span className="vtool-sep" aria-hidden="true" />

      <button
        type="button"
        className="vtool"
        aria-label="Fit to view"
        title="Fit to view (Home)"
        disabled={!hasParts}
        onClick={() => onCamera('fit')}
      >
        <Maximize2 size={15} />
      </button>

      <div className="vtool-menu-anchor" ref={viewsRef}>
        <button
          type="button"
          className="vtool vtool-wide"
          aria-haspopup="menu"
          aria-expanded={viewsOpen}
          aria-label="Standard views"
          title="Standard views"
          disabled={!hasParts}
          onClick={() => setViewsOpen((open) => !open)}
        >
          <Box size={15} />
          <span className="vtool-caret" aria-hidden="true" />
        </button>
        {viewsOpen ? (
          <div className="vtool-menu" role="menu">
            {VIEWS.map((view) => (
              <button
                key={view.id}
                type="button"
                role="menuitem"
                className="menu-item"
                onClick={() => {
                  setViewsOpen(false);
                  onCamera(view.id);
                }}
              >
                {view.label}
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <span className="vtool-sep" aria-hidden="true" />

      <button
        type="button"
        className="vtool"
        aria-pressed={ortho}
        aria-label="Orthographic projection"
        title={ortho ? 'Orthographic projection (on)' : 'Perspective projection'}
        onClick={() => onOrthoChange(!ortho)}
      >
        <ScanEye size={15} />
      </button>
    </div>
  );
}
