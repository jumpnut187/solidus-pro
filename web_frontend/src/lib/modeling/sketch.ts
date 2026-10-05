/**
 * Sketch entities, the profiles they enclose, and their driving dimensions.
 *
 * Two kinds of entity, as in SOLIDWORKS:
 *
 * - Closed shapes (rectangle, circle, polygon, slot, closed polyline) are
 *   each a profile on their own, defined by their driving dimensions, so a
 *   change to a dimension cannot leave them under- or over-defined.
 * - Open segments (line, arc) are joined into profiles wherever their ends
 *   meet. The loops are found by walking the graph of segments face by face
 *   (see `sketchProfiles`), so a rectangle drawn as four lines, a slot drawn
 *   as two lines and two arcs, or a rectangle split in two by a line all give
 *   the regions you would expect.
 *
 * Any entity can be marked For Construction. Construction geometry is drawn
 * dashed and never forms a profile; a centerline is a construction line.
 *
 * Coordinates are sketch-plane (u, v) inches; angles are degrees.
 */
import { distance2, type Vec2 } from './vec';

type Construction = { construction?: boolean };

export type SketchEntity =
  | ({ id: string; type: 'rectangle'; cx: number; cy: number; w: number; h: number; angle: number } & Construction)
  | ({ id: string; type: 'circle'; cx: number; cy: number; d: number } & Construction)
  | ({ id: string; type: 'polygon'; cx: number; cy: number; d: number; sides: number; angle: number } & Construction)
  | ({ id: string; type: 'slot'; cx: number; cy: number; length: number; width: number; angle: number } & Construction)
  | ({ id: string; type: 'polyline'; points: Vec2[] } & Construction)
  /**
   * `relation` is the Horizontal or Vertical relation inferred while drawing:
   * it keeps the line so when a joined end or a dimension moves it.
   */
  | ({ id: string; type: 'line'; x1: number; y1: number; x2: number; y2: number; relation?: 'horizontal' | 'vertical' } & Construction)
  /** Counter-clockwise from `start` to `end` (degrees) about the centre. */
  | ({ id: string; type: 'arc'; cx: number; cy: number; r: number; start: number; end: number } & Construction);

export type SketchEntityType = SketchEntity['type'];

export const ENTITY_LABELS: Record<SketchEntityType, string> = {
  rectangle: 'Rectangle',
  circle: 'Circle',
  polygon: 'Polygon',
  slot: 'Slot',
  polyline: 'Polyline',
  line: 'Line',
  arc: 'Arc'
};

export function entityLabel(entity: SketchEntity): string {
  if (entity.type === 'line' && entity.construction) return 'Centerline';
  return ENTITY_LABELS[entity.type] + (entity.construction ? ' (construction)' : '');
}

/** Sketch tools: what a click on the sketch plane does. */
export type SketchTool =
  | 'select'
  | 'line'
  | 'centerline'
  | 'rectangle'
  | 'centerRectangle'
  | 'circle'
  | 'arc3'
  | 'centerArc'
  | 'polygon'
  | 'slot'
  | 'convert';

/** The longest chord allowed on a tessellated arc, in inches. */
const ARC_CHORD_IN = 0.06;
/** Endpoints closer than this are the same point. */
export const COINCIDENT_TOL = 1e-5;

export function arcSegments(radius: number, sweepRad = Math.PI * 2): number {
  const full = Math.ceil((2 * Math.PI * Math.max(radius, 0)) / ARC_CHORD_IN);
  const fullCircle = sweepRad >= Math.PI * 2 - 1e-9;
  const segments = Math.round((full * sweepRad) / (Math.PI * 2));
  const floor = fullCircle ? 48 : Math.max(4, Math.ceil((48 * sweepRad) / (Math.PI * 2)));
  return Math.min(Math.max(segments, floor), fullCircle ? 256 : 192);
}

export function rotate2(point: Vec2, center: Vec2, angleDeg: number): Vec2 {
  const a = (angleDeg * Math.PI) / 180;
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [center[0] + point[0] * c - point[1] * s, center[1] + point[0] * s + point[1] * c];
}

// ---------------------------------------------------------------------------
// Arcs
// ---------------------------------------------------------------------------

/** Sweep of an arc in degrees, in (0, 360]. */
export function arcSweep(start: number, end: number): number {
  let sweep = (end - start) % 360;
  if (sweep <= 1e-9) sweep += 360;
  return sweep;
}

export function arcPoint(cx: number, cy: number, r: number, deg: number): Vec2 {
  const a = (deg * Math.PI) / 180;
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
}

export function arcEnds(entity: Extract<SketchEntity, { type: 'arc' }>): [Vec2, Vec2] {
  return [arcPoint(entity.cx, entity.cy, entity.r, entity.start), arcPoint(entity.cx, entity.cy, entity.r, entity.end)];
}

function angleOf(center: Vec2, point: Vec2): number {
  return (Math.atan2(point[1] - center[1], point[0] - center[0]) * 180) / Math.PI;
}

/** The circle through three points, or null when they are collinear. */
export function circleThrough(a: Vec2, b: Vec2, c: Vec2): { center: Vec2; r: number } | null {
  const d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
  if (Math.abs(d) < 1e-12) return null;
  const a2 = a[0] * a[0] + a[1] * a[1];
  const b2 = b[0] * b[0] + b[1] * b[1];
  const c2 = c[0] * c[0] + c[1] * c[1];
  const center: Vec2 = [
    (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d,
    (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d
  ];
  return { center, r: distance2(center, a) };
}

/** SOLIDWORKS' 3 Point Arc: from `start` to `end`, bulging through `through`. */
export function arcThrough(start: Vec2, end: Vec2, through: Vec2, id: string): SketchEntity | null {
  const circle = circleThrough(start, end, through);
  if (!circle || circle.r < 1e-4) return null;
  const a0 = angleOf(circle.center, start);
  const a1 = angleOf(circle.center, end);
  const at = angleOf(circle.center, through);
  // Counter-clockwise from start to end passes `through`? Otherwise the arc
  // runs the other way, which is counter-clockwise from end to start.
  const ccw = arcSweep(a0, at) < arcSweep(a0, a1);
  return {
    id,
    type: 'arc',
    cx: circle.center[0],
    cy: circle.center[1],
    r: circle.r,
    start: ccw ? a0 : a1,
    end: ccw ? a1 : a0
  };
}

/** SOLIDWORKS' Centerpoint Arc: radius from `start`, sweep to the angle of `end`. */
export function arcFromCenter(center: Vec2, start: Vec2, end: Vec2, clockwise: boolean, id: string): SketchEntity | null {
  const r = distance2(center, start);
  if (r < 1e-4 || distance2(center, end) < 1e-9) return null;
  const a0 = angleOf(center, start);
  const a1 = angleOf(center, end);
  if (Math.abs(arcSweep(a0, a1) % 360) < 1e-6) return null;
  return { id, type: 'arc', cx: center[0], cy: center[1], r, start: clockwise ? a1 : a0, end: clockwise ? a0 : a1 };
}

/**
 * The arc with the same sweep through two moved endpoints, for an arc whose
 * end was dragged along with the line it is joined to. Keeping the sweep
 * keeps the arc's shape; the radius follows the chord.
 */
function arcThroughEnds(arc: Extract<SketchEntity, { type: 'arc' }>, p1: Vec2, p2: Vec2): SketchEntity {
  const sweep = (arcSweep(arc.start, arc.end) * Math.PI) / 180;
  const chord = distance2(p1, p2);
  if (chord < 1e-9 || sweep >= Math.PI * 2 - 1e-9) return arc;
  const r = chord / (2 * Math.sin(sweep / 2));
  const mid: Vec2 = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
  const dir: Vec2 = [(p2[0] - p1[0]) / chord, (p2[1] - p1[1]) / chord];
  const left: Vec2 = [-dir[1], dir[0]];
  const h = r * Math.cos(sweep / 2);
  const center: Vec2 = [mid[0] + left[0] * h, mid[1] + left[1] * h];
  return { ...arc, cx: center[0], cy: center[1], r: Math.abs(r), start: angleOf(center, p1), end: angleOf(center, p2) };
}

// ---------------------------------------------------------------------------
// Geometry of an entity
// ---------------------------------------------------------------------------

/**
 * The points of an entity, in order. A closed shape comes back
 * counter-clockwise without repeating its first point; an open one (line,
 * arc) from its first end to its second. What is drawn is exactly what is
 * built.
 */
export function entityPath(entity: SketchEntity): { points: Vec2[]; closed: boolean } {
  switch (entity.type) {
    case 'rectangle': {
      const hw = entity.w / 2;
      const hh = entity.h / 2;
      const c: Vec2 = [entity.cx, entity.cy];
      return {
        closed: true,
        points: [
          rotate2([-hw, -hh], c, entity.angle),
          rotate2([hw, -hh], c, entity.angle),
          rotate2([hw, hh], c, entity.angle),
          rotate2([-hw, hh], c, entity.angle)
        ]
      };
    }
    case 'circle': {
      const r = entity.d / 2;
      const n = arcSegments(r);
      const points: Vec2[] = [];
      for (let i = 0; i < n; i += 1) {
        const a = (i / n) * Math.PI * 2;
        points.push([entity.cx + r * Math.cos(a), entity.cy + r * Math.sin(a)]);
      }
      return { closed: true, points };
    }
    case 'polygon': {
      const r = entity.d / 2;
      const sides = Math.max(3, Math.round(entity.sides));
      const points: Vec2[] = [];
      for (let i = 0; i < sides; i += 1) {
        const a = (entity.angle * Math.PI) / 180 + (i / sides) * Math.PI * 2;
        points.push([entity.cx + r * Math.cos(a), entity.cy + r * Math.sin(a)]);
      }
      return { closed: true, points };
    }
    case 'slot': {
      const r = entity.width / 2;
      const half = entity.length / 2;
      const n = arcSegments(r, Math.PI);
      const local: Vec2[] = [];
      for (let i = 0; i <= n; i += 1) {
        const a = -Math.PI / 2 + (i / n) * Math.PI;
        local.push([half + r * Math.cos(a), r * Math.sin(a)]);
      }
      for (let i = 0; i <= n; i += 1) {
        const a = Math.PI / 2 + (i / n) * Math.PI;
        local.push([-half + r * Math.cos(a), r * Math.sin(a)]);
      }
      const c: Vec2 = [entity.cx, entity.cy];
      return { closed: true, points: local.map((point) => rotate2(point, c, entity.angle)) };
    }
    case 'polyline': {
      const points = entity.points.map((point) => [point[0], point[1]] as Vec2);
      return { closed: true, points: signedArea(points) < 0 ? points.reverse() : points };
    }
    case 'line':
      return { closed: false, points: [[entity.x1, entity.y1], [entity.x2, entity.y2]] };
    case 'arc': {
      const sweep = arcSweep(entity.start, entity.end);
      const n = arcSegments(entity.r, (sweep * Math.PI) / 180);
      const points: Vec2[] = [];
      for (let i = 0; i <= n; i += 1) points.push(arcPoint(entity.cx, entity.cy, entity.r, entity.start + (sweep * i) / n));
      // Pin the ends exactly, so they weld with whatever they were snapped to.
      const [p0, p1] = arcEnds(entity);
      points[0] = p0;
      points[points.length - 1] = p1;
      return { closed: false, points };
    }
  }
}

/** The outline of an entity: `entityPath(entity).points`. */
export function entityOutline(entity: SketchEntity): Vec2[] {
  return entityPath(entity).points;
}

export function signedArea(points: Vec2[]): number {
  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    area += a[0] * b[1] - b[0] * a[1];
  }
  return area / 2;
}

/** The ends of an open entity, or null for a closed one. */
export function entityEnds(entity: SketchEntity): [Vec2, Vec2] | null {
  if (entity.type === 'line') return [[entity.x1, entity.y1], [entity.x2, entity.y2]];
  if (entity.type === 'arc') return arcEnds(entity);
  return null;
}

/** Points worth snapping to: centres, corners, ends and midpoints. */
export function entityKeyPoints(entity: SketchEntity): Vec2[] {
  switch (entity.type) {
    case 'rectangle':
    case 'polygon':
      return [[entity.cx, entity.cy], ...entityOutline(entity)];
    case 'circle': {
      const r = entity.d / 2;
      return [
        [entity.cx, entity.cy],
        [entity.cx + r, entity.cy],
        [entity.cx - r, entity.cy],
        [entity.cx, entity.cy + r],
        [entity.cx, entity.cy - r]
      ];
    }
    case 'slot': {
      const half = entity.length / 2;
      const c: Vec2 = [entity.cx, entity.cy];
      return [c, rotate2([half, 0], c, entity.angle), rotate2([-half, 0], c, entity.angle)];
    }
    case 'polyline':
      return entity.points;
    case 'line':
      return [
        [entity.x1, entity.y1],
        [entity.x2, entity.y2],
        [(entity.x1 + entity.x2) / 2, (entity.y1 + entity.y2) / 2]
      ];
    case 'arc': {
      const [a, b] = arcEnds(entity);
      const mid = arcPoint(entity.cx, entity.cy, entity.r, entity.start + arcSweep(entity.start, entity.end) / 2);
      return [a, b, mid, [entity.cx, entity.cy]];
    }
  }
}

/** Why an entity cannot be part of a profile, or null if it can. */
export function entityProblem(entity: SketchEntity): string | null {
  const positive = (value: number) => Number.isFinite(value) && value > 1e-4;
  switch (entity.type) {
    case 'rectangle':
      return positive(entity.w) && positive(entity.h) ? null : 'Width and height must be above zero.';
    case 'circle':
      return positive(entity.d) ? null : 'Diameter must be above zero.';
    case 'polygon':
      if (!positive(entity.d)) return 'Diameter must be above zero.';
      return entity.sides >= 3 && entity.sides <= 64 ? null : 'A polygon needs 3 to 64 sides.';
    case 'slot':
      return positive(entity.width) && Number.isFinite(entity.length) && entity.length >= 0
        ? null
        : 'Slot width must be above zero.';
    case 'polyline':
      if (entity.points.length < 3) return 'A polyline needs at least three points.';
      return Math.abs(signedArea(entity.points)) > 1e-6 ? null : 'The polyline encloses no area.';
    case 'line':
      return Math.hypot(entity.x2 - entity.x1, entity.y2 - entity.y1) > 1e-4 ? null : 'A line must have length.';
    case 'arc':
      return positive(entity.r) ? null : 'Radius must be above zero.';
  }
}

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------

export type SketchProfiles = {
  /** Closed contours, each counter-clockwise; combined even-odd into regions. */
  loops: Vec2[][];
  /** Ends of segments that meet nothing: an open contour. */
  openEnds: Vec2[];
};

/**
 * The closed contours a sketch encloses.
 *
 * Closed shapes are contours as they stand. Lines and arcs become a planar
 * graph: ends within `COINCIDENT_TOL` are one vertex, and segments hanging
 * off the graph by a free end are pruned (their free ends are the open
 * contour SOLIDWORKS marks). What remains is walked face by face -- at each
 * vertex leave by the segment next clockwise from the one you arrived on,
 * which keeps the face on your left -- and every face that comes out
 * counter-clockwise is a bounded region. Each region is a contour; the
 * clockwise face of each component is its outside and is dropped.
 */
export function sketchProfiles(entities: SketchEntity[]): SketchProfiles {
  const loops: Vec2[][] = [];
  const segments: Vec2[][] = [];
  for (const entity of entities) {
    if (entity.construction || entityProblem(entity)) continue;
    const path = entityPath(entity);
    if (path.closed) loops.push(path.points);
    else segments.push(path.points);
  }

  // An end landing on the middle of another segment (a T junction, a line
  // drawn across to an edge) splits that segment there, so the regions on
  // either side of it come out as faces of their own.
  splitAtEnds(segments);

  // Weld ends into vertices.
  const vertices: Vec2[] = [];
  const vertexOf = (point: Vec2): number => {
    for (let i = 0; i < vertices.length; i += 1) {
      if (distance2(vertices[i], point) <= COINCIDENT_TOL) return i;
    }
    vertices.push(point);
    return vertices.length - 1;
  };
  type Edge = { a: number; b: number; points: Vec2[]; alive: boolean };
  const edges: Edge[] = [];
  for (const points of segments) {
    const a = vertexOf(points[0]);
    const b = vertexOf(points[points.length - 1]);
    if (a === b && points.length <= 2) continue;
    edges.push({ a, b, points, alive: true });
  }

  const degree = new Array(vertices.length).fill(0);
  for (const edge of edges) {
    degree[edge.a] += 1;
    degree[edge.b] += 1;
  }
  const openEnds = vertices.filter((_, index) => degree[index] === 1);

  // Prune dangling segments until every live vertex has degree >= 2.
  let pruned = true;
  while (pruned) {
    pruned = false;
    for (const edge of edges) {
      if (!edge.alive) continue;
      if (degree[edge.a] === 1 || degree[edge.b] === 1) {
        edge.alive = false;
        degree[edge.a] -= 1;
        degree[edge.b] -= 1;
        pruned = true;
      }
    }
  }

  // Half-edges, and the outgoing ones at each vertex sorted by angle.
  type Half = { edge: number; from: number; to: number; points: Vec2[]; angle: number; used: boolean };
  const halves: Half[] = [];
  const outgoing: number[][] = vertices.map(() => []);
  edges.forEach((edge, index) => {
    if (!edge.alive) return;
    const forward = edge.points;
    const backward = [...edge.points].reverse();
    for (const [from, to, points] of [
      [edge.a, edge.b, forward],
      [edge.b, edge.a, backward]
    ] as const) {
      const p0 = points[0];
      const p1 = points[1];
      halves.push({ edge: index, from, to, points, angle: Math.atan2(p1[1] - p0[1], p1[0] - p0[0]), used: false });
      outgoing[from].push(halves.length - 1);
    }
  });
  for (const list of outgoing) list.sort((x, y) => halves[x].angle - halves[y].angle);

  const twinOf = (index: number) => (index % 2 === 0 ? index + 1 : index - 1);
  const nextOf = (index: number): number => {
    const twin = twinOf(index);
    const list = outgoing[halves[index].to];
    const at = list.indexOf(twin);
    // Next clockwise from the twin = previous in counter-clockwise order.
    return list[(at - 1 + list.length) % list.length];
  };

  for (let start = 0; start < halves.length; start += 1) {
    if (halves[start].used) continue;
    const face: Vec2[] = [];
    let current = start;
    let guard = 0;
    while (!halves[current].used && guard < halves.length + 1) {
      halves[current].used = true;
      const points = halves[current].points;
      for (let i = 0; i < points.length - 1; i += 1) face.push(points[i]);
      current = nextOf(current);
      guard += 1;
    }
    if (current === start && face.length >= 3 && signedArea(face) > 1e-9) loops.push(face);
  }

  return { loops, openEnds };
}

/** Split, in place, every segment polyline at the ends of the others lying on it. */
function splitAtEnds(segments: Vec2[][]): void {
  // Each split removes one end from the middle of a segment, so this ends;
  // the cap only guards against a pathological sketch.
  for (let guard = 0; guard < 2000; guard += 1) {
    const ends = segments.flatMap((points) => [points[0], points[points.length - 1]]);
    let split = false;
    for (let s = 0; s < segments.length && !split; s += 1) {
      const points = segments[s];
      for (const end of ends) {
        if (distance2(end, points[0]) <= COINCIDENT_TOL || distance2(end, points[points.length - 1]) <= COINCIDENT_TOL) continue;
        for (let i = 0; i < points.length - 1; i += 1) {
          const a = points[i];
          const b = points[i + 1];
          const dx = b[0] - a[0];
          const dy = b[1] - a[1];
          const lengthSq = dx * dx + dy * dy;
          if (lengthSq < 1e-24) continue;
          const t = ((end[0] - a[0]) * dx + (end[1] - a[1]) * dy) / lengthSq;
          if (t <= 0 || t >= 1) continue;
          const foot: Vec2 = [a[0] + t * dx, a[1] + t * dy];
          if (distance2(foot, end) > COINCIDENT_TOL) continue;
          const first = [...points.slice(0, i + 1), end];
          const second = [end, ...points.slice(i + 1)];
          segments.splice(s, 1, first, second);
          split = true;
          break;
        }
        if (split) break;
      }
    }
    if (!split) return;
  }
}

/** The sketch's problem, or null when it encloses at least one region and has no open contour. */
export function sketchProblem(entities: SketchEntity[]): string | null {
  const real = entities.filter((entity) => !entity.construction);
  if (!real.length) return 'Draw at least one closed profile.';
  for (const entity of real) {
    const problem = entityProblem(entity);
    if (problem) return `${ENTITY_LABELS[entity.type]}: ${problem}`;
  }
  const { loops, openEnds } = sketchProfiles(entities);
  if (openEnds.length) {
    const count = openEnds.length;
    return `The sketch has an open contour (${count} open end${count === 1 ? '' : 's'}, marked in red). Close it, or mark the stray lines For Construction.`;
  }
  if (!loops.length) return 'The sketch encloses no area.';
  return null;
}

// ---------------------------------------------------------------------------
// Editing with coincident ends
// ---------------------------------------------------------------------------

function withEnds(entity: SketchEntity, ends: [Vec2, Vec2]): SketchEntity {
  if (entity.type === 'line') return { ...entity, x1: ends[0][0], y1: ends[0][1], x2: ends[1][0], y2: ends[1][1] };
  if (entity.type === 'arc') return arcThroughEnds(entity, ends[0], ends[1]);
  return entity;
}

/**
 * Replace one entity and carry along everything joined to the ends it
 * moved, the way coincident and horizontal/vertical relations behave in a
 * sketch: drag the corner of a rectangle drawn with four lines and both
 * lines at the corner follow, and a vertical side stays vertical by moving
 * its far end too -- which in turn moves the line joined there.
 */
export function replaceEntity(entities: SketchEntity[], next: SketchEntity): SketchEntity[] {
  // An edit that breaks the line's own relation removes the relation.
  if (next.type === 'line' && next.relation) {
    const holds = next.relation === 'horizontal' ? Math.abs(next.y2 - next.y1) < 1e-9 : Math.abs(next.x2 - next.x1) < 1e-9;
    if (!holds) {
      const { relation: _dropped, ...rest } = next;
      void _dropped;
      next = rest;
    }
  }
  const result = new Map(entities.map((entity) => [entity.id, entity]));
  const previous = result.get(next.id);
  result.set(next.id, next);
  const queue: Array<{ from: Vec2; to: Vec2; source: string }> = [];
  const pushMoves = (before: SketchEntity | undefined, after: SketchEntity) => {
    const a = before ? entityEnds(before) : null;
    const b = entityEnds(after);
    if (!a || !b) return;
    for (let k = 0; k < 2; k += 1) {
      if (distance2(a[k], b[k]) > 1e-12) queue.push({ from: a[k], to: b[k], source: after.id });
    }
  };
  pushMoves(previous, next);
  const touched = new Set([next.id]);
  for (let guard = 0; queue.length && guard < 4 * entities.length + 8; guard += 1) {
    const move = queue.shift() as { from: Vec2; to: Vec2; source: string };
    for (const [id, entity] of result) {
      if (id === move.source || touched.has(id)) continue;
      const ends = entityEnds(entity);
      if (!ends) continue;
      const k = ends.findIndex((end) => distance2(end, move.from) <= COINCIDENT_TOL);
      if (k < 0) continue;
      const moved: [Vec2, Vec2] = [ends[0], ends[1]];
      moved[k] = move.to;
      // Keep the line's relation by moving its far end along with it.
      if (entity.type === 'line' && entity.relation) {
        const far = 1 - k;
        moved[far] = entity.relation === 'horizontal' ? [moved[far][0], move.to[1]] : [move.to[0], moved[far][1]];
      }
      const updated = withEnds(entity, moved);
      touched.add(id);
      result.set(id, updated);
      pushMoves(entity, updated);
    }
  }
  return entities.map((entity) => result.get(entity.id) as SketchEntity);
}

/** Move one end of a line or arc (index 0 or 1) and everything joined there. */
export function moveEnd(entities: SketchEntity[], id: string, end: 0 | 1, to: Vec2): SketchEntity[] {
  const entity = entities.find((item) => item.id === id);
  const ends = entity ? entityEnds(entity) : null;
  if (!entity || !ends) return entities;
  const next: [Vec2, Vec2] = end === 0 ? [to, ends[1]] : [ends[0], to];
  return replaceEntity(entities, withEnds(entity, next));
}

export function translateEntity(entity: SketchEntity, dx: number, dy: number): SketchEntity {
  switch (entity.type) {
    case 'polyline':
      return { ...entity, points: entity.points.map((point) => [point[0] + dx, point[1] + dy] as Vec2) };
    case 'line':
      return { ...entity, x1: entity.x1 + dx, y1: entity.y1 + dy, x2: entity.x2 + dx, y2: entity.y2 + dy };
    default:
      return { ...entity, cx: entity.cx + dx, cy: entity.cy + dy };
  }
}

/** The entities joined to `id` through shared ends, `id` included. */
export function connectedIds(entities: SketchEntity[], id: string): Set<string> {
  const found = new Set([id]);
  let grew = true;
  while (grew) {
    grew = false;
    for (const entity of entities) {
      if (found.has(entity.id)) continue;
      const ends = entityEnds(entity);
      if (!ends) continue;
      const touches = entities.some((other) => {
        if (!found.has(other.id)) return false;
        const otherEnds = entityEnds(other);
        return otherEnds !== null && ends.some((end) => otherEnds.some((o) => distance2(o, end) <= COINCIDENT_TOL));
      });
      if (touches) {
        found.add(entity.id);
        grew = true;
      }
    }
  }
  return found;
}

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

export type DimensionKey = 'cx' | 'cy' | 'w' | 'h' | 'd' | 'length' | 'width' | 'angle' | 'sides' | 'r' | 'lineLength';

/**
 * A driving dimension as the viewport draws it: a value, where its label sits
 * on the sketch, and the span it measures. Editing the label edits the
 * entity, which is the whole of what makes the sketch parametric.
 */
export type DimensionSpec = {
  entityId: string;
  key: DimensionKey;
  value: number;
  prefix: string;
  anchor: Vec2;
  /** The measured span, drawn as a thin dimension line. */
  line?: [Vec2, Vec2];
};

/**
 * Size dimensions for every entity, and position dimensions (to the sketch
 * origin) for the selected one. Showing every entity's position at once
 * buries a busy sketch in labels; the selected one is the one being placed.
 */
export function sketchDimensions(entities: SketchEntity[], selectedId: string | null, gap: number): DimensionSpec[] {
  const dims: DimensionSpec[] = [];
  for (const entity of entities) {
    if (entity.construction && entity.id !== selectedId) continue;
    switch (entity.type) {
      case 'rectangle': {
        const hw = entity.w / 2;
        const hh = entity.h / 2;
        const c: Vec2 = [entity.cx, entity.cy];
        dims.push({
          entityId: entity.id,
          key: 'w',
          value: entity.w,
          prefix: '',
          anchor: rotate2([0, -hh - gap], c, entity.angle),
          line: [rotate2([-hw, -hh - gap], c, entity.angle), rotate2([hw, -hh - gap], c, entity.angle)]
        });
        dims.push({
          entityId: entity.id,
          key: 'h',
          value: entity.h,
          prefix: '',
          anchor: rotate2([hw + gap, 0], c, entity.angle),
          line: [rotate2([hw + gap, -hh], c, entity.angle), rotate2([hw + gap, hh], c, entity.angle)]
        });
        break;
      }
      case 'circle': {
        const r = entity.d / 2;
        const a = Math.PI / 4;
        const rim: Vec2 = [entity.cx + r * Math.cos(a), entity.cy + r * Math.sin(a)];
        const out: Vec2 = [entity.cx + (r + gap) * Math.cos(a), entity.cy + (r + gap) * Math.sin(a)];
        dims.push({ entityId: entity.id, key: 'd', value: entity.d, prefix: 'Ø', anchor: out, line: [[entity.cx, entity.cy], rim] });
        break;
      }
      case 'polygon': {
        const r = entity.d / 2;
        const a = (entity.angle * Math.PI) / 180;
        const out: Vec2 = [entity.cx + (r + gap) * Math.cos(a), entity.cy + (r + gap) * Math.sin(a)];
        dims.push({
          entityId: entity.id,
          key: 'd',
          value: entity.d,
          prefix: 'Ø',
          anchor: out,
          line: [[entity.cx, entity.cy], [entity.cx + r * Math.cos(a), entity.cy + r * Math.sin(a)]]
        });
        break;
      }
      case 'slot': {
        const c: Vec2 = [entity.cx, entity.cy];
        const half = entity.length / 2;
        const r = entity.width / 2;
        dims.push({
          entityId: entity.id,
          key: 'length',
          value: entity.length,
          prefix: '',
          anchor: rotate2([0, -r - gap], c, entity.angle),
          line: [rotate2([-half, -r - gap], c, entity.angle), rotate2([half, -r - gap], c, entity.angle)]
        });
        dims.push({
          entityId: entity.id,
          key: 'width',
          value: entity.width,
          prefix: '',
          anchor: rotate2([half + r + gap, 0], c, entity.angle),
          line: [rotate2([half + r + gap, -r], c, entity.angle), rotate2([half + r + gap, r], c, entity.angle)]
        });
        break;
      }
      case 'line': {
        const length = Math.hypot(entity.x2 - entity.x1, entity.y2 - entity.y1);
        if (length < 1e-9) break;
        // Offset to the left of the line's direction, the side its label reads from.
        const nx = -(entity.y2 - entity.y1) / length;
        const ny = (entity.x2 - entity.x1) / length;
        const mid: Vec2 = [(entity.x1 + entity.x2) / 2 + nx * gap, (entity.y1 + entity.y2) / 2 + ny * gap];
        dims.push({ entityId: entity.id, key: 'lineLength', value: length, prefix: '', anchor: mid });
        break;
      }
      case 'arc': {
        const mid = arcPoint(entity.cx, entity.cy, entity.r + gap, entity.start + arcSweep(entity.start, entity.end) / 2);
        const rim = arcPoint(entity.cx, entity.cy, entity.r, entity.start + arcSweep(entity.start, entity.end) / 2);
        dims.push({ entityId: entity.id, key: 'r', value: entity.r, prefix: 'R', anchor: mid, line: [[entity.cx, entity.cy], rim] });
        break;
      }
      case 'polyline':
        break;
    }
    if (entity.id === selectedId && 'cx' in entity && entity.type !== 'arc') {
      dims.push({
        entityId: entity.id,
        key: 'cx',
        value: entity.cx,
        prefix: 'X ',
        anchor: [entity.cx / 2, entity.cy],
        line: [[0, entity.cy], [entity.cx, entity.cy]]
      });
      dims.push({
        entityId: entity.id,
        key: 'cy',
        value: entity.cy,
        prefix: 'Y ',
        anchor: [entity.cx, entity.cy / 2],
        line: [[entity.cx, 0], [entity.cx, entity.cy]]
      });
    }
  }
  return dims;
}

/** Apply an edited dimension, refusing values that would break the entity. */
export function withDimension(entity: SketchEntity, key: DimensionKey, value: number): SketchEntity | null {
  if (!Number.isFinite(value) || entity.type === 'polyline') return null;
  if (key === 'lineLength') {
    if (entity.type !== 'line' || !(value > 1e-4)) return null;
    const length = Math.hypot(entity.x2 - entity.x1, entity.y2 - entity.y1);
    if (length < 1e-12) return null;
    const scale = value / length;
    // The first end stays put; the second moves along the line.
    return { ...entity, x2: entity.x1 + (entity.x2 - entity.x1) * scale, y2: entity.y1 + (entity.y2 - entity.y1) * scale };
  }
  if ((key === 'w' || key === 'h' || key === 'd' || key === 'width' || key === 'r') && value <= 1e-4) return null;
  if (key === 'length' && value < 0) return null;
  if (key === 'sides' && (value < 3 || value > 64)) return null;
  if (!(key in entity)) return null;
  return { ...entity, [key]: key === 'sides' ? Math.round(value) : value } as SketchEntity;
}

let entityCounter = 0;
export function newEntityId(): string {
  entityCounter += 1;
  return `e${Date.now().toString(36)}${entityCounter.toString(36)}`;
}

/**
 * Drop points that lie on the straight line between their neighbours, for a
 * contour traced off a tessellated face: a straight edge of an STL face is
 * usually several collinear triangle edges.
 */
export function simplifyLoop(points: Vec2[], tolerance: number): Vec2[] {
  if (points.length <= 3) return points;
  let current = points;
  let changed = true;
  while (changed && current.length > 3) {
    changed = false;
    const kept: Vec2[] = [];
    for (let i = 0; i < current.length; i += 1) {
      const prev = kept.length ? kept[kept.length - 1] : current[(i - 1 + current.length) % current.length];
      const point = current[i];
      const next = current[(i + 1) % current.length];
      const span = distance2(prev, next);
      const cross = Math.abs((point[0] - prev[0]) * (next[1] - prev[1]) - (point[1] - prev[1]) * (next[0] - prev[0]));
      const dot = (point[0] - prev[0]) * (next[0] - prev[0]) + (point[1] - prev[1]) * (next[1] - prev[1]);
      const between = dot > 0 && dot < span * span;
      if (span > 0 && cross / span <= tolerance && between) {
        changed = true;
        continue;
      }
      if (distance2(prev, point) <= tolerance && kept.length) {
        changed = true;
        continue;
      }
      kept.push(point);
    }
    if (kept.length >= 3) current = kept;
    else break;
  }
  return current;
}
