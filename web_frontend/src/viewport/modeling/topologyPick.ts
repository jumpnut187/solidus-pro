/**
 * What the Measure tool's click is on: a vertex, an edge, or a face.
 *
 * An STL carries none of these, so they are recovered from the tessellation
 * the viewport already has: vertices are triangle corners, edges are the
 * feature edges drawn for "shaded with edges" (chained end to end until the
 * chain turns sharply or branches, then classified as a straight line or a
 * circular arc), and faces are the flat regions `planarFace` finds. Priority
 * follows SOLIDWORKS: a vertex within a few pixels beats an edge, an edge
 * beats the face under it.
 */
import type { Camera } from '@babylonjs/core/Cameras/camera';
import { Matrix, Vector3 } from '@babylonjs/core/Maths/math.vector';
import type { Viewport } from '@babylonjs/core/Maths/math.viewport';
import type { AbstractMesh } from '@babylonjs/core/Meshes/abstractMesh';
import type { LinesMesh } from '@babylonjs/core/Meshes/linesMesh';
import type { Scene } from '@babylonjs/core/scene';

import type { MeasurePick } from '../../lib/modeling/measure';
import type { Vec3 } from '../../lib/modeling/vec';
import type { PlanarFace } from './planarFace';

const VERTEX_RADIUS_PX = 9;
const EDGE_RADIUS_PX = 6;
/** A chain continues through a vertex only if it turns less than this. */
const CHAIN_TURN_DEG = 40;

type EdgeGraph = {
  /** World-space segment ends, six floats per segment. */
  segments: Float32Array;
  /** Welded vertex index of each segment end, two per segment. */
  ends: Uint32Array;
  /** Segments at each welded vertex (CSR). */
  offsets: Uint32Array;
  incident: Uint32Array;
  points: Vec3[];
};

const graphCache = new WeakMap<object, { matrixKey: string; graph: EdgeGraph }>();

function edgeGraph(edges: LinesMesh): EdgeGraph | null {
  const local = edges.getVerticesData('position');
  if (!local || local.length < 6) return null;
  const world = edges.getWorldMatrix();
  const matrixKey = world.asArray().join(',');
  const cached = graphCache.get(local);
  if (cached && cached.matrixKey === matrixKey) return cached.graph;

  const segments = new Float32Array(local.length);
  const scratch = new Vector3();
  for (let i = 0; i < local.length; i += 3) {
    Vector3.TransformCoordinatesFromFloatsToRef(local[i], local[i + 1], local[i + 2], world, scratch);
    segments[i] = scratch.x;
    segments[i + 1] = scratch.y;
    segments[i + 2] = scratch.z;
  }
  const keyIndex = new Map<string, number>();
  const points: Vec3[] = [];
  const segmentCount = segments.length / 6;
  const ends = new Uint32Array(segmentCount * 2);
  for (let s = 0; s < segmentCount * 2; s += 1) {
    const key = `${local[3 * s]},${local[3 * s + 1]},${local[3 * s + 2]}`;
    let index = keyIndex.get(key);
    if (index === undefined) {
      index = points.length;
      keyIndex.set(key, index);
      points.push([segments[3 * s], segments[3 * s + 1], segments[3 * s + 2]]);
    }
    ends[s] = index;
  }
  const offsets = new Uint32Array(points.length + 1);
  for (let s = 0; s < ends.length; s += 1) offsets[ends[s] + 1] += 1;
  for (let v = 0; v < points.length; v += 1) offsets[v + 1] += offsets[v];
  const fill = offsets.slice(0, points.length);
  const incident = new Uint32Array(ends.length);
  for (let s = 0; s < ends.length; s += 1) {
    incident[fill[ends[s]]] = s >> 1;
    fill[ends[s]] += 1;
  }
  const graph = { segments, ends, offsets, incident, points };
  graphCache.set(local, { matrixKey, graph });
  return graph;
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function unit(a: Vec3): Vec3 {
  const length = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / length, a[1] / length, a[2] / length];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Ordered points of the chain of feature edges through segment `seed`. */
function chainThrough(graph: EdgeGraph, seed: number): { points: Vec3[]; closed: boolean } {
  const turnCos = Math.cos((CHAIN_TURN_DEG * Math.PI) / 180);
  const used = new Set<number>([seed]);
  const walk = (fromVertex: number, viaSegment: number): number[] => {
    const vertices: number[] = [];
    let vertex = fromVertex;
    let segment = viaSegment;
    for (let guard = 0; guard < graph.ends.length; guard += 1) {
      const start = graph.offsets[vertex];
      const end = graph.offsets[vertex + 1];
      if (end - start !== 2) break;
      const other = graph.incident[start] === segment ? graph.incident[start + 1] : graph.incident[start];
      if (used.has(other)) break;
      const previousVertex = graph.ends[2 * segment] === vertex ? graph.ends[2 * segment + 1] : graph.ends[2 * segment];
      const nextVertex = graph.ends[2 * other] === vertex ? graph.ends[2 * other + 1] : graph.ends[2 * other];
      const incoming = unit(sub(graph.points[vertex], graph.points[previousVertex]));
      const outgoing = unit(sub(graph.points[nextVertex], graph.points[vertex]));
      if (dot(incoming, outgoing) < turnCos) break;
      used.add(other);
      vertices.push(nextVertex);
      vertex = nextVertex;
      segment = other;
    }
    return vertices;
  };
  const a = graph.ends[2 * seed];
  const b = graph.ends[2 * seed + 1];
  const forward = walk(b, seed);
  const closed = forward.length > 0 && forward[forward.length - 1] === a;
  const backward = closed ? [] : walk(a, seed);
  const order = [...backward.reverse(), a, b, ...(closed ? forward.slice(0, -1) : forward)];
  return { points: order.map((index) => graph.points[index]), closed };
}

function isCorner(graph: EdgeGraph, point: Vec3): boolean {
  const index = graph.points.findIndex((candidate) => Math.hypot(...sub(candidate, point)) < 1e-6);
  if (index < 0) return false;
  const start = graph.offsets[index];
  const end = graph.offsets[index + 1];
  if (end - start !== 2) return true;
  const other = (segment: number) =>
    graph.ends[2 * segment] === index ? graph.ends[2 * segment + 1] : graph.ends[2 * segment];
  const a = unit(sub(graph.points[index], graph.points[other(graph.incident[start])]));
  const b = unit(sub(graph.points[other(graph.incident[start + 1])], graph.points[index]));
  return dot(a, b) < Math.cos((CHAIN_TURN_DEG * Math.PI) / 180);
}

/** The circle through three points in space, or null if they are collinear. */
function circle3(a: Vec3, b: Vec3, c: Vec3): { center: Vec3; radius: number; axis: Vec3 } | null {
  const ab = sub(b, a);
  const ac = sub(c, a);
  const n = cross(ab, ac);
  const nn = dot(n, n);
  if (nn < 1e-18) return null;
  const term1 = cross(n, ab).map((value) => value * dot(ac, ac)) as Vec3;
  const term2 = cross(ac, n).map((value) => value * dot(ab, ab)) as Vec3;
  const center: Vec3 = [
    a[0] + (term1[0] + term2[0]) / (2 * nn),
    a[1] + (term1[1] + term2[1]) / (2 * nn),
    a[2] + (term1[2] + term2[2]) / (2 * nn)
  ];
  return { center, radius: Math.hypot(...sub(a, center)), axis: unit(n) };
}

function classifyChain(points: Vec3[], closed: boolean, partName: string | null): MeasurePick | null {
  if (points.length < 2) return null;
  const length = points.reduce((sum, point, i) => (i === 0 ? 0 : sum + Math.hypot(...sub(point, points[i - 1]))), 0) +
    (closed ? Math.hypot(...sub(points[0], points[points.length - 1])) : 0);
  const first = points[0];
  const last = points[points.length - 1];
  const span = Math.hypot(...sub(last, first));
  if (!closed && span > 0) {
    const direction = unit(sub(last, first));
    const straight = points.every((point) => {
      const d = sub(point, first);
      const along = dot(d, direction);
      return Math.hypot(d[0] - direction[0] * along, d[1] - direction[1] * along, d[2] - direction[2] * along) <= span * 1e-3;
    });
    if (straight) return { kind: 'line', a: first, b: last, points, partName };
  }
  if (points.length >= 3) {
    const third = Math.floor(points.length / 3);
    const circle = circle3(points[0], points[third], points[Math.min(2 * third, points.length - 1)]);
    if (circle && circle.radius > 0) {
      const round = points.every((point) => Math.abs(Math.hypot(...sub(point, circle.center)) - circle.radius) <= circle.radius * 0.02);
      if (round) {
        return {
          kind: 'circle',
          center: circle.center,
          axis: circle.axis,
          radius: circle.radius,
          full: closed,
          length,
          points,
          partName
        };
      }
    }
  }
  return null;
}

export type TopologyContext = {
  scene: Scene;
  camera: Camera;
  /** Pointer, in `scene.pointerX` space. */
  x: number;
  y: number;
  mesh: AbstractMesh;
  faceId: number;
  hitPoint: Vector3;
  hitNormal: Vector3;
  edges: LinesMesh | null;
  face: PlanarFace | null;
  partName: string | null;
  /** Longest edge of the assembly, for depth tolerances. */
  assemblySize: number;
};

function projector(scene: Scene, camera: Camera) {
  const engine = scene.getEngine();
  const viewport = camera.viewport.toGlobal(engine.getRenderWidth(), engine.getRenderHeight()) as Viewport;
  const transform = scene.getTransformMatrix();
  const scale = engine.getHardwareScalingLevel();
  const identity = Matrix.Identity();
  return (p: Vec3) => {
    const projected = Vector3.Project(new Vector3(p[0], p[1], p[2]), identity, transform, viewport);
    return { x: projected.x * scale, y: projected.y * scale };
  };
}

export function pickTopology(context: TopologyContext): MeasurePick {
  const { scene, camera, x, y, mesh, faceId, partName } = context;
  const project = projector(scene, camera);
  const hit: Vec3 = [context.hitPoint.x, context.hitPoint.y, context.hitPoint.z];

  // Vertex: a corner of the triangle under the pointer.
  const indices = mesh.getIndices();
  const positions = mesh.getVerticesData('position');
  if (indices && positions && faceId >= 0) {
    const world = mesh.getWorldMatrix();
    let best: { point: Vec3; distance: number } | null = null;
    for (let k = 0; k < 3; k += 1) {
      const v = indices[3 * faceId + k] * 3;
      const p = Vector3.TransformCoordinates(new Vector3(positions[v], positions[v + 1], positions[v + 2]), world);
      const point: Vec3 = [p.x, p.y, p.z];
      const screen = project(point);
      const distance = Math.hypot(screen.x - x, screen.y - y);
      if (distance <= VERTEX_RADIUS_PX && (!best || distance < best.distance)) best = { point, distance };
    }
    // Only a real corner of the model: a point where feature edges branch or
    // turn sharply. A vertex inside a flat face's fan of triangles, or one of
    // the points a circle is tessellated into, is not one.
    if (best && context.edges) {
      const graph = edgeGraph(context.edges);
      if (graph && isCorner(graph, best.point)) return { kind: 'vertex', point: best.point, partName };
    }
  }

  // Edge: the nearest feature-edge segment on screen, near the hit in depth.
  if (context.edges) {
    const graph = edgeGraph(context.edges);
    if (graph) {
      const depthTolerance = Math.max(context.assemblySize * 0.02, 0.05);
      let best: { segment: number; distance: number } | null = null;
      const s = graph.segments;
      for (let i = 0; i < s.length / 6; i += 1) {
        const a: Vec3 = [s[6 * i], s[6 * i + 1], s[6 * i + 2]];
        const b: Vec3 = [s[6 * i + 3], s[6 * i + 4], s[6 * i + 5]];
        const pa = project(a);
        const pb = project(b);
        const dx = pb.x - pa.x;
        const dy = pb.y - pa.y;
        const lengthSq = dx * dx + dy * dy;
        const t = lengthSq > 0 ? Math.min(Math.max(((x - pa.x) * dx + (y - pa.y) * dy) / lengthSq, 0), 1) : 0;
        const distance = Math.hypot(x - (pa.x + t * dx), y - (pa.y + t * dy));
        if (distance > EDGE_RADIUS_PX || (best && distance >= best.distance)) continue;
        const onSegment: Vec3 = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
        // Hidden edges behind the face under the pointer are not candidates.
        if (Math.hypot(...sub(onSegment, hit)) > depthTolerance) continue;
        best = { segment: i, distance };
      }
      if (best) {
        const chain = chainThrough(graph, best.segment);
        const classified = classifyChain(chain.points, chain.closed, partName);
        if (classified) return classified;
      }
    }
  }

  const face = context.face;
  if (face && face.planar) {
    return {
      kind: 'face',
      point: hit,
      normal: face.normal,
      planar: true,
      area: face.area,
      centroid: face.centroid,
      triangles: face.world,
      partName
    };
  }
  const normal: Vec3 = [context.hitNormal.x, context.hitNormal.y, context.hitNormal.z];
  return { kind: 'point', point: hit, normal, partName };
}
