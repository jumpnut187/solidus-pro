import type { ProjectPart } from './api';

export type Vec3 = [number, number, number];
export type Mat3 = [Vec3, Vec3, Vec3];
export type WorldAxis = 'x' | 'y' | 'z';
export type WorldBounds = [Vec3, Vec3];
export type PartingSide = 'above' | 'below';

export type PartPlacement = {
  rotation: Vec3;
  position: Vec3;
  scale: number;
};

const AXES: Record<WorldAxis, Vec3> = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
const EULER_EPSILON = 2.220446049250313e-16 * 4;
const DEG = Math.PI / 180;

export function cleanAngle(value: number): number {
  const shifted = value + 180;
  let angle = (((shifted % 360) + 360) % 360) - 180;
  if (angle <= -180 + 1e-9) angle = 180;
  const rounded = Math.round(angle * 1e6) / 1e6;
  return rounded === 0 ? 0 : rounded;
}

export function eulerToMatrix(rotationDeg: Vec3): Mat3 {
  const [ax, ay, az] = rotationDeg.map((value) => value * DEG);
  const cx = Math.cos(ax);
  const sx = Math.sin(ax);
  const cy = Math.cos(ay);
  const sy = Math.sin(ay);
  const cz = Math.cos(az);
  const sz = Math.sin(az);
  return [
    [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
    [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
    [-sy, cy * sx, cy * cx]
  ];
}

export function matrixToEuler(m: Mat3): Vec3 {
  const cy = Math.hypot(m[0][0], m[1][0]);
  let ax: number;
  let ay: number;
  let az: number;
  if (cy > EULER_EPSILON) {
    ax = Math.atan2(m[2][1], m[2][2]);
    ay = Math.atan2(-m[2][0], cy);
    az = Math.atan2(m[1][0], m[0][0]);
  } else {
    ax = Math.atan2(-m[1][2], m[1][1]);
    ay = Math.atan2(-m[2][0], cy);
    az = 0;
  }
  return [cleanAngle(ax / DEG), cleanAngle(ay / DEG), cleanAngle(az / DEG)];
}

export function multiply(a: Mat3, b: Mat3): Mat3 {
  const out: Mat3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  return out;
}

export function transpose(m: Mat3): Mat3 {
  return [
    [m[0][0], m[1][0], m[2][0]],
    [m[0][1], m[1][1], m[2][1]],
    [m[0][2], m[1][2], m[2][2]]
  ];
}

export function apply(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]
  ];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function norm(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function scaled(v: Vec3, s: number): Vec3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

export function axisRotationMatrix(axis: WorldAxis | Vec3, angleDeg: number): Mat3 {
  const raw = typeof axis === 'string' ? AXES[axis] : axis;
  const d = scaled(raw, 1 / norm(raw));
  const angle = angleDeg * DEG;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1 - c;
  return [
    [c + d[0] * d[0] * t, d[0] * d[1] * t - d[2] * s, d[0] * d[2] * t + d[1] * s],
    [d[1] * d[0] * t + d[2] * s, c + d[1] * d[1] * t, d[1] * d[2] * t - d[0] * s],
    [d[2] * d[0] * t - d[1] * s, d[2] * d[1] * t + d[0] * s, c + d[2] * d[2] * t]
  ];
}

export function pivotPosition(position: Vec3, delta: Mat3, pivot: Vec3): Vec3 {
  const turned = apply(delta, [position[0] - pivot[0], position[1] - pivot[1], position[2] - pivot[2]]);
  return [turned[0] + pivot[0], turned[1] + pivot[1], turned[2] + pivot[2]];
}

export function applyDelta(rotationDeg: Vec3, position: Vec3, delta: Mat3, pivot: Vec3): { rotation: Vec3; position: Vec3 } {
  return {
    rotation: matrixToEuler(multiply(delta, eulerToMatrix(rotationDeg))),
    position: pivotPosition(position, delta, pivot)
  };
}

export function rotateAboutWorldAxis(
  rotationDeg: Vec3,
  position: Vec3,
  axis: WorldAxis | Vec3,
  angleDeg: number,
  pivot: Vec3
): { rotation: Vec3; position: Vec3 } {
  return applyDelta(rotationDeg, position, axisRotationMatrix(axis, angleDeg), pivot);
}

export function alignVectorMatrix(src: Vec3, dst: Vec3): Mat3 {
  const a = scaled(src, 1 / norm(src));
  const b = scaled(dst, 1 / norm(dst));
  const axis = cross(a, b);
  const dot = Math.min(1, Math.max(-1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]));
  if (norm(axis) < 1e-9) {
    if (dot > 0) {
      return [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ];
    }
    let perp = cross(a, [1, 0, 0]);
    if (norm(perp) < 1e-6) perp = cross(a, [0, 1, 0]);
    return axisRotationMatrix(scaled(perp, 1 / norm(perp)), 180);
  }
  const angle = Math.atan2(norm(axis), dot) / DEG;
  return axisRotationMatrix(scaled(axis, 1 / norm(axis)), angle);
}

export function faceToDirection(
  rotationDeg: Vec3,
  position: Vec3,
  worldNormal: Vec3,
  pivot: Vec3,
  target: Vec3 = [0, 0, -1]
): { rotation: Vec3; position: Vec3 } {
  return applyDelta(rotationDeg, position, alignVectorMatrix(worldNormal, target), pivot);
}

export function worldBounds(vertices: ArrayLike<number>, placement: PartPlacement): WorldBounds {
  const r = eulerToMatrix(placement.rotation);
  const s = placement.scale || 1;
  const [px, py, pz] = placement.position;
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i + 2 < vertices.length; i += 3) {
    const x = vertices[i] * s;
    const y = vertices[i + 1] * s;
    const z = vertices[i + 2] * s;
    const wx = r[0][0] * x + r[0][1] * y + r[0][2] * z + px;
    const wy = r[1][0] * x + r[1][1] * y + r[1][2] * z + py;
    const wz = r[2][0] * x + r[2][1] * y + r[2][2] * z + pz;
    if (wx < min[0]) min[0] = wx;
    if (wy < min[1]) min[1] = wy;
    if (wz < min[2]) min[2] = wz;
    if (wx > max[0]) max[0] = wx;
    if (wy > max[1]) max[1] = wy;
    if (wz > max[2]) max[2] = wz;
  }
  return [min, max];
}

export function boundsCenter(bounds: WorldBounds): Vec3 {
  return [
    (bounds[0][0] + bounds[1][0]) / 2,
    (bounds[0][1] + bounds[1][1]) / 2,
    (bounds[0][2] + bounds[1][2]) / 2
  ];
}

export function boundsSize(bounds: WorldBounds): Vec3 {
  return [bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1], bounds[1][2] - bounds[0][2]];
}

export function dropToFloorDz(bounds: WorldBounds, floorZ = 0): number {
  return floorZ - bounds[0][2];
}

export function snapToPartingDz(bounds: WorldBounds, partingZ: number, side: PartingSide = 'above'): number {
  return side === 'below' ? partingZ - bounds[1][2] : partingZ - bounds[0][2];
}

export function snapAngle(angleDeg: number, step = 15): number {
  if (step <= 0) return angleDeg;
  return cleanAngle(Math.round(angleDeg / step) * step);
}

export function matrixToQuaternion(m: Mat3): [number, number, number, number] {
  const trace = m[0][0] + m[1][1] + m[2][2];
  let x: number;
  let y: number;
  let z: number;
  let w: number;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    w = 0.25 / s;
    x = (m[2][1] - m[1][2]) * s;
    y = (m[0][2] - m[2][0]) * s;
    z = (m[1][0] - m[0][1]) * s;
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    const s = 2 * Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]);
    w = (m[2][1] - m[1][2]) / s;
    x = 0.25 * s;
    y = (m[0][1] + m[1][0]) / s;
    z = (m[0][2] + m[2][0]) / s;
  } else if (m[1][1] > m[2][2]) {
    const s = 2 * Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]);
    w = (m[0][2] - m[2][0]) / s;
    x = (m[0][1] + m[1][0]) / s;
    y = 0.25 * s;
    z = (m[1][2] + m[2][1]) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]);
    w = (m[1][0] - m[0][1]) / s;
    x = (m[0][2] + m[2][0]) / s;
    y = (m[1][2] + m[2][1]) / s;
    z = 0.25 * s;
  }
  const length = Math.hypot(x, y, z, w);
  return [x / length, y / length, z / length, w / length];
}

export function partQuaternion(rotationDeg: Vec3): [number, number, number, number] {
  return matrixToQuaternion(eulerToMatrix(rotationDeg));
}

export function placementOf(part: ProjectPart): PartPlacement {
  return {
    rotation: [part.rotation_x, part.rotation_y, part.rotation_z],
    position: [part.position_x, part.position_y, part.position_z],
    scale: part.scale || 1
  };
}

export function partWorldBounds(part: ProjectPart): WorldBounds | null {
  const vertices = part.meshData?.vertices;
  if (!vertices?.length) return null;
  return worldBounds(vertices, placementOf(part));
}

export function partBoxWorldBounds(part: ProjectPart): WorldBounds | null {
  const box = part.meshData?.bounds;
  if (!box) return null;
  const corners: number[] = [];
  for (const x of box.x) for (const y of box.y) for (const z of box.z) corners.push(x, y, z);
  return worldBounds(corners, placementOf(part));
}
