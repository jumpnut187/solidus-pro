import { cross3, dot3, normalize3, roundTo, scale3, sub3, type Vec2, type Vec3 } from './vec';

export type StandardPlaneId = 'top' | 'front' | 'right';

export type SketchPlane =
  | { kind: 'standard'; id: StandardPlaneId; offset: number; refId?: string }
  | { kind: 'custom'; normal: Vec3; offset: number; label: string; refId?: string };

export const STANDARD_PLANES: { id: StandardPlaneId; label: string; hint: string }[] = [
  { id: 'top', label: 'Top', hint: 'XY plane, normal +Z' },
  { id: 'front', label: 'Front', hint: 'XZ plane, normal -Y' },
  { id: 'right', label: 'Right', hint: 'YZ plane, normal +X' }
];

export type PlaneFrame = { origin: Vec3; normal: Vec3; u: Vec3; v: Vec3 };

const STANDARD_FRAMES: Record<StandardPlaneId, { normal: Vec3; u: Vec3; v: Vec3 }> = {
  top: { normal: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] },
  front: { normal: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
  right: { normal: [1, 0, 0], u: [0, 1, 0], v: [0, 0, 1] }
};

export function planeNormal(plane: SketchPlane): Vec3 {
  return plane.kind === 'standard' ? STANDARD_FRAMES[plane.id].normal : normalize3(plane.normal);
}

export function planeFrame(plane: SketchPlane): PlaneFrame {
  if (plane.kind === 'standard') {
    const frame = STANDARD_FRAMES[plane.id];
    return { ...frame, origin: scale3(frame.normal, plane.offset) };
  }
  const normal = normalize3(plane.normal);
  const { u, v } = axesForNormal(normal);
  return { origin: scale3(normal, plane.offset), normal, u, v };
}

export function axesForNormal(normal: Vec3): { u: Vec3; v: Vec3 } {
  if (Math.abs(normal[2]) > 0.995) {
    const u: Vec3 = [1, 0, 0];
    return { u, v: normalize3(cross3(normal, u)) };
  }
  const up: Vec3 = [0, 0, 1];
  const v = normalize3(sub3(up, scale3(normal, dot3(up, normal))));
  return { u: normalize3(cross3(v, normal)), v };
}

function snapNormal(normal: Vec3): Vec3 {
  const n = normalize3(normal);
  for (let axis = 0; axis < 3; axis += 1) {
    if (Math.abs(n[axis]) > Math.cos((1 * Math.PI) / 180)) {
      const snapped: Vec3 = [0, 0, 0];
      snapped[axis] = Math.sign(n[axis]) || 1;
      return snapped;
    }
  }
  return n;
}

export function planeThroughPoint(point: Vec3, normal: Vec3, label = 'Face'): SketchPlane {
  const n = snapNormal(normal);
  return { kind: 'custom', normal: n, offset: roundTo(dot3(n, point), 1e-4), label };
}

export function planeLabel(plane: SketchPlane): string {
  if (plane.kind === 'custom') return plane.label;
  const base = STANDARD_PLANES.find((entry) => entry.id === plane.id)?.label ?? plane.id;
  if (Math.abs(plane.offset) < 1e-9) return `${base} Plane`;
  return `${base} Plane ${plane.offset > 0 ? '+' : '−'}${Math.abs(plane.offset).toFixed(3)} in`;
}

export function sketchToWorld(frame: PlaneFrame, point: Vec2, lift = 0): Vec3 {
  return [
    frame.origin[0] + frame.u[0] * point[0] + frame.v[0] * point[1] + frame.normal[0] * lift,
    frame.origin[1] + frame.u[1] * point[0] + frame.v[1] * point[1] + frame.normal[1] * lift,
    frame.origin[2] + frame.u[2] * point[0] + frame.v[2] * point[1] + frame.normal[2] * lift
  ];
}

export function worldToSketch(frame: PlaneFrame, point: Vec3): Vec2 {
  const d = sub3(point, frame.origin);
  return [dot3(d, frame.u), dot3(d, frame.v)];
}

export function offsetPlane(plane: SketchPlane, distance: number, label?: string): SketchPlane {
  if (plane.kind === 'standard') return { kind: 'standard', id: plane.id, offset: plane.offset + distance };
  return { kind: 'custom', normal: plane.normal, offset: plane.offset + distance, label: label ?? plane.label };
}

export type ReferencePlaneSource =
  | { kind: 'offset'; base: SketchPlane; distance: number; flip: boolean }
  | { kind: 'mid'; a: SketchPlane; b: SketchPlane };

export type ReferencePlane = { id: string; name: string; source: ReferencePlaneSource };

export function parallelOffset(a: SketchPlane, b: SketchPlane): number | null {
  const na = planeNormal(a);
  const nb = planeNormal(b);
  const cos = dot3(na, nb);
  if (Math.abs(Math.abs(cos) - 1) > 1e-4) return null;
  return cos > 0 ? b.offset : -b.offset;
}

export function resolveReferencePlane(ref: ReferencePlane): SketchPlane | null {
  const source = ref.source;
  if (source.kind === 'offset') {
    const distance = source.flip ? -source.distance : source.distance;
    const moved = offsetPlane(stripRef(source.base), distance, ref.name);
    return moved.kind === 'standard' ? { ...moved, refId: ref.id } : { ...moved, label: ref.name, refId: ref.id };
  }
  const bOffset = parallelOffset(source.a, source.b);
  if (bOffset === null) return null;
  const mid = (source.a.offset + bOffset) / 2;
  const a = stripRef(source.a);
  if (a.kind === 'standard') return { kind: 'standard', id: a.id, offset: mid, refId: ref.id };
  return { kind: 'custom', normal: a.normal, offset: mid, label: ref.name, refId: ref.id };
}

function stripRef(plane: SketchPlane): SketchPlane {
  const copy = { ...plane };
  delete copy.refId;
  return copy;
}

export function referencePlanePreview(source: ReferencePlaneSource, count: number, name: string): SketchPlane[] {
  if (source.kind === 'mid') {
    const plane = resolveReferencePlane({ id: 'preview', name, source });
    return plane ? [plane] : [];
  }
  const planes: SketchPlane[] = [];
  for (let k = 1; k <= Math.max(1, Math.round(count)); k += 1) {
    const plane = resolveReferencePlane({
      id: `preview-${k}`,
      name,
      source: { ...source, distance: source.distance * k }
    });
    if (plane) planes.push(plane);
  }
  return planes;
}

export function referencePlaneProblem(source: ReferencePlaneSource | null): string | null {
  if (!source) return 'Select a plane or a face to offset from.';
  if (source.kind === 'offset') {
    return Number.isFinite(source.distance) ? null : 'Enter an offset distance.';
  }
  return parallelOffset(source.a, source.b) === null ? 'A mid plane needs two parallel planes or faces.' : null;
}

export function nextPlaneName(planes: ReferencePlane[]): string {
  const used = new Set(planes.map((plane) => plane.name));
  for (let n = 1; ; n += 1) {
    const name = `Plane${n}`;
    if (!used.has(name)) return name;
  }
}

export function parseReferencePlanes(value: unknown): ReferencePlane[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is ReferencePlane => {
    if (!item || typeof item !== 'object') return false;
    const candidate = item as Partial<ReferencePlane>;
    return typeof candidate.id === 'string' && typeof candidate.name === 'string' && typeof candidate.source === 'object';
  });
}

let planeCounter = 0;
export function newPlaneId(): string {
  planeCounter += 1;
  return `p${Date.now().toString(36)}${planeCounter.toString(36)}`;
}
