/**
 * The flat face of a tessellated part around one triangle.
 *
 * An STL has no faces, only triangles, so "the face you clicked" has to be
 * recovered: every triangle reachable from the picked one across shared
 * vertices that lies in the same plane (normal within half a degree, and on
 * the plane within a small tolerance). That is the region a CAD viewer lights
 * up when you hover a face, and the region the sketch plane will sit on. A
 * curved face has no such region; the result is then the picked triangle
 * alone, and the plane is the tangent plane there.
 */
import type { AbstractMesh } from '@babylonjs/core/Meshes/abstractMesh';
import { Vector3 } from '@babylonjs/core/Maths/math.vector';

const NORMAL_COS = Math.cos((0.5 * Math.PI) / 180);
const MAX_TRIANGLES = 50_000;

/**
 * Vertex-to-triangle adjacency over WELDED vertices: the crease-angle split
 * duplicates every vertex on a sharp edge, and two copies of one point have
 * to count as the same vertex or a face would stop at its own diagonal.
 */
type Adjacency = { weld: Uint32Array; offsets: Uint32Array; triangles: Uint32Array };
const adjacencyCache = new WeakMap<object, Adjacency>();

function buildAdjacency(positions: ArrayLike<number>, indices: ArrayLike<number>): Adjacency {
  const vertexCount = positions.length / 3;
  const weld = new Uint32Array(vertexCount);
  const seen = new Map<string, number>();
  for (let v = 0; v < vertexCount; v += 1) {
    const key = `${positions[3 * v]},${positions[3 * v + 1]},${positions[3 * v + 2]}`;
    const existing = seen.get(key);
    weld[v] = existing ?? v;
    if (existing === undefined) seen.set(key, v);
  }
  const triangleCount = indices.length / 3;
  const offsets = new Uint32Array(vertexCount + 1);
  for (let i = 0; i < indices.length; i += 1) offsets[weld[indices[i]] + 1] += 1;
  for (let v = 0; v < vertexCount; v += 1) offsets[v + 1] += offsets[v];
  const fill = offsets.slice(0, vertexCount);
  const triangles = new Uint32Array(triangleCount * 3);
  for (let t = 0; t < triangleCount; t += 1) {
    for (let k = 0; k < 3; k += 1) {
      const v = weld[indices[3 * t + k]];
      triangles[fill[v]] = t;
      fill[v] += 1;
    }
  }
  return { weld, offsets, triangles };
}

export type PlanarFace = {
  /** Triangle indices of the face, for testing whether a later pick is on it. */
  ids: Set<number>;
  /** World-space triangles, nine floats each, for drawing the highlight. */
  world: Float32Array;
  /**
   * False when the region is a facet of a curved surface rather than a flat
   * face: one or two triangles (a cylinder's facet is a quad) with a smoothly
   * turning neighbour across an edge.
   */
  planar: boolean;
  /** World-space area, square inches. */
  area: number;
  /** Area-weighted centre, world space. */
  centroid: [number, number, number];
  /** Outward unit normal, world space. */
  normal: [number, number, number];
};

/** Faces meeting within this angle are a smooth surface, not an edge. */
const SMOOTH_COS = Math.cos((20 * Math.PI) / 180);

/** The planar face containing triangle `faceId` of `mesh`, or null if there is no such triangle. */
export function planarFace(mesh: AbstractMesh, faceId: number): PlanarFace | null {
  const positions = mesh.getVerticesData('position');
  const indices = mesh.getIndices();
  if (!positions || !indices || faceId < 0 || faceId * 3 + 2 >= indices.length) return null;

  let adjacency = adjacencyCache.get(positions);
  if (!adjacency) {
    adjacency = buildAdjacency(positions, indices);
    adjacencyCache.set(positions, adjacency);
  }
  const { weld, offsets, triangles } = adjacency;

  const point = (v: number) => new Vector3(positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]);
  const unitNormal = (t: number): Vector3 | null => {
    const a = point(indices[3 * t]);
    const n = Vector3.Cross(point(indices[3 * t + 1]).subtract(a), point(indices[3 * t + 2]).subtract(a));
    const length = n.length();
    return length > 0 ? n.scaleInPlace(1 / length) : null;
  };

  const seedNormal = unitNormal(faceId);
  if (!seedNormal) return null;
  const offset = Vector3.Dot(seedNormal, point(indices[3 * faceId]));
  // Scaled to the part, so a large casting's float noise does not split a face.
  const extent = mesh.getBoundingInfo().boundingBox.extendSize.length();
  const planeTolerance = Math.max(extent * 1e-4, 1e-5);

  const visited = new Uint8Array(indices.length / 3);
  const stack = [faceId];
  visited[faceId] = 1;
  const region: number[] = [];
  let smoothNeighbour = false;
  while (stack.length && region.length < MAX_TRIANGLES) {
    const t = stack.pop() as number;
    region.push(t);
    for (let k = 0; k < 3; k += 1) {
      const v = weld[indices[3 * t + k]];
      for (let p = offsets[v]; p < offsets[v + 1]; p += 1) {
        const other = triangles[p];
        if (visited[other]) continue;
        const n = unitNormal(other);
        const cos = n ? Vector3.Dot(n, seedNormal) : -1;
        const coplanar =
          n !== null &&
          cos >= NORMAL_COS &&
          Math.abs(Vector3.Dot(seedNormal, point(indices[3 * other])) - offset) <= planeTolerance;
        if (!coplanar) {
          if (cos >= SMOOTH_COS) smoothNeighbour = true;
          continue;
        }
        visited[other] = 1;
        stack.push(other);
      }
    }
  }

  const world = mesh.getWorldMatrix();
  const out = new Float32Array(region.length * 9);
  const scratch = new Vector3();
  region.forEach((t, i) => {
    for (let k = 0; k < 3; k += 1) {
      Vector3.TransformCoordinatesToRef(point(indices[3 * t + k]), world, scratch);
      out[9 * i + 3 * k] = scratch.x;
      out[9 * i + 3 * k + 1] = scratch.y;
      out[9 * i + 3 * k + 2] = scratch.z;
    }
  });

  let area = 0;
  const centroid: [number, number, number] = [0, 0, 0];
  const normalSum: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < out.length; i += 9) {
    const ux = out[i + 3] - out[i];
    const uy = out[i + 4] - out[i + 1];
    const uz = out[i + 5] - out[i + 2];
    const vx = out[i + 6] - out[i];
    const vy = out[i + 7] - out[i + 1];
    const vz = out[i + 8] - out[i + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    const a = Math.hypot(nx, ny, nz) / 2;
    area += a;
    normalSum[0] += nx;
    normalSum[1] += ny;
    normalSum[2] += nz;
    for (let axis = 0; axis < 3; axis += 1) centroid[axis] += (a * (out[i + axis] + out[i + 3 + axis] + out[i + 6 + axis])) / 3;
  }
  if (area > 0) for (let axis = 0; axis < 3; axis += 1) centroid[axis] /= area;
  const nLength = Math.hypot(...normalSum) || 1;
  return {
    ids: new Set(region),
    world: out,
    planar: !(region.length <= 2 && smoothNeighbour),
    area,
    centroid,
    normal: [normalSum[0] / nLength, normalSum[1] / nLength, normalSum[2] / nLength]
  };
}

/**
 * The closed loops bounding a set of triangles (nine floats each), as
 * world-space points. Edges used by one triangle of the set are its boundary;
 * they are chained head to tail in the triangles' own winding, so the outer
 * loop of a face runs counter-clockwise seen from its outside and the loops
 * around its holes clockwise -- exactly what an even-odd profile wants.
 */
export function faceBoundaryLoops(triangles: Float32Array): Array<Array<[number, number, number]>> {
  const keyOf = (i: number) => `${triangles[i]},${triangles[i + 1]},${triangles[i + 2]}`;
  const pointOf = new Map<string, [number, number, number]>();
  const directed = new Map<string, number>();
  const edgeKey = (a: string, b: string) => `${a}|${b}`;
  for (let i = 0; i < triangles.length; i += 9) {
    const keys = [keyOf(i), keyOf(i + 3), keyOf(i + 6)];
    for (let k = 0; k < 3; k += 1) {
      pointOf.set(keys[k], [triangles[i + 3 * k], triangles[i + 3 * k + 1], triangles[i + 3 * k + 2]]);
      const key = edgeKey(keys[k], keys[(k + 1) % 3]);
      directed.set(key, (directed.get(key) ?? 0) + 1);
    }
  }
  // A boundary edge is one whose reverse is not in the set.
  const next = new Map<string, string[]>();
  for (const key of directed.keys()) {
    const [a, b] = key.split('|');
    if (directed.has(edgeKey(b, a))) continue;
    const list = next.get(a);
    if (list) list.push(b);
    else next.set(a, [b]);
  }
  const loops: Array<Array<[number, number, number]>> = [];
  for (const start of [...next.keys()]) {
    while ((next.get(start)?.length ?? 0) > 0) {
      const loop: Array<[number, number, number]> = [];
      let at = start;
      let guard = 0;
      while (guard < directed.size + 1) {
        const outs = next.get(at);
        if (!outs?.length) break;
        const to = outs.pop() as string;
        loop.push(pointOf.get(at) as [number, number, number]);
        at = to;
        guard += 1;
        if (at === start) break;
      }
      if (at === start && loop.length >= 3) loops.push(loop);
    }
  }
  return loops;
}
