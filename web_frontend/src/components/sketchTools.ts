import {
  Circle,
  CircleDashed,
  Copy,
  Hexagon,
  MousePointer2,
  Pill,
  RectangleHorizontal,
  RotateCw,
  SeparatorHorizontal,
  Slash,
  Spline,
  SquareDashed
} from 'lucide-react';
import type { ComponentType } from 'react';

import type { SketchTool } from '../lib/modeling/types';

/** The sketch tools, shared by the property manager and the canvas command bar. */
export const SKETCH_TOOLS: { id: SketchTool; label: string; hint: string; Icon: ComponentType<{ size?: number }> }[] = [
  { id: 'select', label: 'Select', hint: 'Select, drag or delete; drag a line end to move it (Esc)', Icon: MousePointer2 },
  { id: 'line', label: 'Line', hint: 'Click points to chain lines; it stops when the contour closes. Double-click or Enter to stop (L)', Icon: Slash },
  { id: 'centerline', label: 'Centerline', hint: 'A construction line: drawn dashed, never part of a profile', Icon: SeparatorHorizontal },
  { id: 'rectangle', label: 'Corner Rectangle', hint: 'Click two opposite corners (R)', Icon: RectangleHorizontal },
  { id: 'centerRectangle', label: 'Center Rectangle', hint: 'Click the centre, then a corner', Icon: SquareDashed },
  { id: 'circle', label: 'Circle', hint: 'Click the centre, then a point on the circle (C)', Icon: Circle },
  { id: 'arc3', label: '3 Point Arc', hint: 'Click the start, the end, then a point the arc passes through (A)', Icon: Spline },
  { id: 'centerArc', label: 'Centerpoint Arc', hint: 'Click the centre, the start, then sweep to the end', Icon: RotateCw },
  { id: 'polygon', label: 'Polygon', hint: 'Click the centre, then a vertex', Icon: Hexagon },
  { id: 'slot', label: 'Slot', hint: 'Click both arc centres, then the width', Icon: Pill },
  { id: 'convert', label: 'Convert Entities', hint: "Click a face to copy its outline into the sketch", Icon: Copy }
];

export const CONSTRUCTION_ICON = CircleDashed;
