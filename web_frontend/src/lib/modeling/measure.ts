/**
 * The Measure tool's arithmetic (SOLIDWORKS' Tools > Evaluate > Measure).
 *
 * The viewport turns a click into a `MeasurePick` -- a vertex, an edge (a
 * straight line or a circular arc, recovered from the tessellation), a face
 * (flat or curved), or a bare point -- and this file says what to report for
 * one pick or a pair. The result also names the two points the viewport
 * draws its dimension line between.
 */
import {
  add3,
  angleBetween3,
  cross3,
  distance3,
  dot3,
  length3,
  lerp3,
  normalize3,
  scale3,
  sub3,
  type Vec3
} from './vec';

export type MeasurePick =
  | { kind: 'vertex'; point: Vec3; partName: string | null }
  | { kind: 'point'; point: Vec3; normal: Vec3; partName: string | null }
  | {
      kind: 'face';
      point: Vec3;
      normal: Vec3;
      /** False for a facet of a curved face, whose plane is only a tangent plane. */
      planar: boolean;
      area: number;
      centroid: Vec3;
      /** World-space triangles for the highlight, nine floats each. */
      triangles: Float32Array;
      partName: string | null;
    }
  | { kind: 'line'; a: Vec3; b: Vec3; points: Vec3[]; partName: string | null }
  | {
      kind: 'circle';
      center: Vec3;
      axis: Vec3;
      radius: number;
      /** True when the edge closes on itself (a full circle). */
      full: boolean;
      length: number;
      points: Vec3[];
      partName: string | null;
    };

export type MeasureRow = { label: string; value: string };

export type MeasureResult = {
  title: string;
  rows: MeasureRow[];
  /** Where the viewport draws its dimension line, and the text on it. */
  line: { from: Vec3; to: Vec3; label: string } | null;
};

export function pickLabel(pick: MeasurePick): string {
  const on = pick.partName ? ` of ${pick.partName}` : '';
  switch (pick.kind) {
    case 'vertex':
      return `Vertex${on}`;
    case 'point':
      return `Point${on}`;
    case 'face':
      return `${pick.planar ? 'Face' : 'Curved face'}${on}`;
    case 'line':
      return `Edge${on}`;
    case 'circle':
      return `${pick.full ? 'Circular edge' : 'Arc edge'}${on}`;
  }
}

const inch = (value: number) => `${value.toFixed(4)} in`;
const deg = (value: number) => `${value.toFixed(2)}°`;
/** Three places, with a rounded-away negative zero shown as 0. */
const coord = (value: number) => (Math.abs(value) < 5e-4 ? 0 : value).toFixed(3);
const vec = (value: Vec3) => `${coord(value[0])}, ${coord(value[1])}, ${coord(value[2])}`;

/** The point a pick stands for when it has to be one point. */
function anchor(pick: MeasurePick): Vec3 {
  switch (pick.kind) {
    case 'vertex':
    case 'point':
      return pick.point;
    case 'face':
      return pick.planar ? pick.centroid : pick.point;
    case 'line':
      return lerp3(pick.a, pick.b, 0.5);
    case 'circle':
      return pick.center;
  }
}

function closestOnSegment(p: Vec3, a: Vec3, b: Vec3): Vec3 {
  const ab = sub3(b, a);
  const lengthSq = dot3(ab, ab);
  const t = lengthSq > 0 ? Math.min(Math.max(dot3(sub3(p, a), ab) / lengthSq, 0), 1) : 0;
  return add3(a, scale3(ab, t));
}

/** Closest points between two segments (clamped), by alternating projection. */
function closestBetweenSegments(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3): [Vec3, Vec3] {
  let p = lerp3(a0, a1, 0.5);
  let q = closestOnSegment(p, b0, b1);
  for (let i = 0; i < 32; i += 1) {
    p = closestOnSegment(q, a0, a1);
    q = closestOnSegment(p, b0, b1);
  }
  return [p, q];
}

function deltaRows(a: Vec3, b: Vec3): MeasureRow[] {
  return [
    { label: 'dX', value: inch(Math.abs(b[0] - a[0])) },
    { label: 'dY', value: inch(Math.abs(b[1] - a[1])) },
    { label: 'dZ', value: inch(Math.abs(b[2] - a[2])) }
  ];
}

function single(pick: MeasurePick): MeasureResult {
  switch (pick.kind) {
    case 'vertex':
    case 'point':
      return { title: pickLabel(pick), rows: [{ label: 'X, Y, Z', value: vec(pick.point) }], line: null };
    case 'face':
      return {
        title: pickLabel(pick),
        rows: pick.planar
          ? [
              { label: 'Area', value: `${pick.area.toFixed(4)} in²` },
              { label: 'Normal', value: vec(pick.normal) },
              { label: 'Centroid', value: vec(pick.centroid) }
            ]
          : [
              { label: 'Point', value: vec(pick.point) },
              { label: 'Normal there', value: vec(pick.normal) }
            ],
        line: null
      };
    case 'line': {
      const length = distance3(pick.a, pick.b);
      return {
        title: pickLabel(pick),
        rows: [{ label: 'Length', value: inch(length) }, ...deltaRows(pick.a, pick.b)],
        line: { from: pick.a, to: pick.b, label: inch(length) }
      };
    }
    case 'circle': {
      const rows: MeasureRow[] = [
        { label: 'Diameter', value: inch(pick.radius * 2) },
        { label: 'Radius', value: inch(pick.radius) },
        { label: 'Center', value: vec(pick.center) },
        { label: pick.full ? 'Circumference' : 'Arc length', value: inch(pick.length) }
      ];
      const rim = pick.points[0] ?? pick.center;
      return { title: pickLabel(pick), rows, line: { from: pick.center, to: rim, label: `Ø${(pick.radius * 2).toFixed(3)}` } };
    }
  }
}

function pair(a: MeasurePick, b: MeasurePick): MeasureResult {
  const title = `${pickLabel(a)} → ${pickLabel(b)}`;

  // Two flat faces: parallel ones have a normal distance, others an angle.
  if (a.kind === 'face' && b.kind === 'face' && a.planar && b.planar) {
    const angle = angleBetween3(a.normal, b.normal);
    const parallel = angle < 0.05 || angle > 179.95;
    if (parallel) {
      const gap = Math.abs(dot3(a.normal, sub3(b.centroid, a.centroid)));
      const from = b.centroid;
      const to = sub3(from, scale3(a.normal, dot3(a.normal, sub3(b.centroid, a.centroid))));
      return {
        title,
        rows: [
          { label: 'Normal distance', value: inch(gap) },
          { label: 'Center distance', value: inch(distance3(a.centroid, b.centroid)) },
          { label: 'Angle', value: deg(angle > 90 ? 180 - angle : angle) }
        ],
        line: { from: to, to: from, label: inch(gap) }
      };
    }
    return {
      title,
      rows: [
        { label: 'Angle', value: deg(180 - angle) },
        { label: 'Center distance', value: inch(distance3(a.centroid, b.centroid)) }
      ],
      line: { from: a.centroid, to: b.centroid, label: deg(180 - angle) }
    };
  }

  // A point (or vertex, or circle centre) and a flat face: the normal distance.
  const asPoint = (pick: MeasurePick): Vec3 | null =>
    pick.kind === 'vertex' || pick.kind === 'point' ? pick.point : pick.kind === 'circle' ? pick.center : null;
  for (const [p, f] of [
    [a, b],
    [b, a]
  ] as const) {
    const point = asPoint(p);
    if (point && f.kind === 'face' && f.planar) {
      const signed = dot3(f.normal, sub3(point, f.centroid));
      const foot = sub3(point, scale3(f.normal, signed));
      return {
        title,
        rows: [{ label: 'Normal distance', value: inch(Math.abs(signed)) }, ...deltaRows(point, foot)],
        line: { from: point, to: foot, label: inch(Math.abs(signed)) }
      };
    }
  }

  // Two straight edges: angle between them and the closest distance.
  if (a.kind === 'line' && b.kind === 'line') {
    const angle = angleBetween3(sub3(a.b, a.a), sub3(b.b, b.a));
    const [p, q] = closestBetweenSegments(a.a, a.b, b.a, b.b);
    const acute = angle > 90 ? 180 - angle : angle;
    const parallel = acute < 0.05;
    const rows: MeasureRow[] = [
      { label: 'Angle', value: deg(acute) },
      { label: parallel ? 'Perpendicular distance' : 'Minimum distance', value: inch(distance3(p, q)) }
    ];
    return { title, rows, line: { from: p, to: q, label: parallel ? inch(distance3(p, q)) : deg(acute) } };
  }

  // A straight edge and a flat face: angle, and the distance if parallel.
  for (const [l, f] of [
    [a, b],
    [b, a]
  ] as const) {
    if (l.kind === 'line' && f.kind === 'face' && f.planar) {
      const direction = normalize3(sub3(l.b, l.a));
      const angle = 90 - angleBetween3(direction, f.normal);
      const mid = lerp3(l.a, l.b, 0.5);
      const signed = dot3(f.normal, sub3(mid, f.centroid));
      const foot = sub3(mid, scale3(f.normal, signed));
      const rows: MeasureRow[] = [{ label: 'Angle to face', value: deg(Math.abs(angle)) }];
      if (Math.abs(angle) < 0.05) rows.push({ label: 'Normal distance', value: inch(Math.abs(signed)) });
      return { title, rows, line: { from: mid, to: foot, label: Math.abs(angle) < 0.05 ? inch(Math.abs(signed)) : deg(Math.abs(angle)) } };
    }
  }

  // A point and a straight edge: the perpendicular distance.
  for (const [p, l] of [
    [a, b],
    [b, a]
  ] as const) {
    const point = asPoint(p);
    if (point && l.kind === 'line') {
      const foot = closestOnSegment(point, l.a, l.b);
      return {
        title,
        rows: [{ label: 'Distance', value: inch(distance3(point, foot)) }, ...deltaRows(point, foot)],
        line: { from: point, to: foot, label: inch(distance3(point, foot)) }
      };
    }
  }

  // Two circles: the distance between centres, and whether they share an axis.
  if (a.kind === 'circle' && b.kind === 'circle') {
    const between = distance3(a.center, b.center);
    const coaxial =
      angleBetween3(a.axis, b.axis) % 180 < 0.1 &&
      length3(cross3(normalize3(a.axis), sub3(b.center, a.center))) < 1e-3;
    const rows: MeasureRow[] = [
      { label: 'Center distance', value: inch(between) },
      ...deltaRows(a.center, b.center),
      { label: 'Diameters', value: `${(a.radius * 2).toFixed(4)} / ${(b.radius * 2).toFixed(4)} in` }
    ];
    if (coaxial) rows.push({ label: 'Coaxial', value: 'yes' });
    return { title, rows, line: { from: a.center, to: b.center, label: inch(between) } };
  }

  // Anything else: the distance between the points the picks stand for.
  const from = anchor(a);
  const to = anchor(b);
  return {
    title,
    rows: [{ label: 'Distance', value: inch(distance3(from, to)) }, ...deltaRows(from, to)],
    line: { from, to, label: inch(distance3(from, to)) }
  };
}

export function measure(picks: MeasurePick[]): MeasureResult | null {
  if (picks.length === 0) return null;
  if (picks.length === 1) return single(picks[0]);
  return pair(picks[0], picks[1]);
}
