import type { ProjectPart } from './api';

export type Axis = 'x' | 'y' | 'z';

type PositionPatch = { id: string; position_x: number; position_y: number; position_z: number };

function boundsOf(part: ProjectPart): { min: [number, number, number]; size: [number, number, number] } | null {
  const bounds = part.meshData?.bounds;
  if (!bounds) return null;
  return {
    min: [bounds.x[0], bounds.y[0], bounds.z[0]],
    size: [bounds.x[1] - bounds.x[0], bounds.y[1] - bounds.y[0], bounds.z[1] - bounds.z[0]]
  };
}

export function arrangeGrid(parts: ProjectPart[], spacing: number): PositionPatch[] {
  const measured = parts.map((part) => ({ part, box: boundsOf(part) })).filter((entry) => entry.box !== null);
  if (!measured.length) return [];

  const cols = Math.ceil(Math.sqrt(measured.length));
  const cellX = Math.max(...measured.map((entry) => entry.box!.size[0])) + spacing;
  const cellY = Math.max(...measured.map((entry) => entry.box!.size[1])) + spacing;

  return measured.map((entry, index) => {
    const { min, size } = entry.box!;
    const row = Math.floor(index / cols);
    const col = index % cols;
    return {
      id: entry.part.id,
      position_x: col * cellX + (cellX - size[0]) / 2 - min[0],
      position_y: row * cellY + (cellY - size[1]) / 2 - min[1],
      position_z: -min[2]
    };
  });
}

export function arrangeLine(parts: ProjectPart[], axis: Axis, spacing: number): PositionPatch[] {
  const index = { x: 0, y: 1, z: 2 }[axis];
  let cursor = 0;
  const patches: PositionPatch[] = [];

  for (const part of parts) {
    const box = boundsOf(part);
    if (!box) continue;
    const position: [number, number, number] = [-box.min[0], -box.min[1], -box.min[2]];
    position[index] = cursor - box.min[index];
    patches.push({
      id: part.id,
      position_x: position[0],
      position_y: position[1],
      position_z: position[2]
    });
    cursor += box.size[index] + spacing;
  }
  return patches;
}

export const DUPLICATE_OFFSET: [number, number, number] = [2, 0, 0];
