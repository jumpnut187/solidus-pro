export const CAD_CREASE_ANGLE_DEG = 30;

export type CreasedGeometry = {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  edgeSegments: Float32Array;
};

type ArrayLike3 = ArrayLike<number>;

export function buildCreasedGeometry(
  positions: ArrayLike3,
  indices: ArrayLike3,
  creaseAngleDeg: number = CAD_CREASE_ANGLE_DEG
): CreasedGeometry {
  const vertexCount = Math.floor(positions.length / 3);
  const faceCount = Math.floor(indices.length / 3);
  const creaseCos = Math.cos((creaseAngleDeg * Math.PI) / 180);

  const faceUnit = new Float64Array(faceCount * 3);
  const faceArea = new Float64Array(faceCount);
  for (let f = 0; f < faceCount; f += 1) {
    const a = indices[3 * f] * 3;
    const b = indices[3 * f + 1] * 3;
    const c = indices[3 * f + 2] * 3;
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    const length = Math.hypot(nx, ny, nz);
    faceArea[f] = length;
    if (length > 0) {
      faceUnit[3 * f] = nx / length;
      faceUnit[3 * f + 1] = ny / length;
      faceUnit[3 * f + 2] = nz / length;
    }
  }

  const degree = new Uint32Array(vertexCount + 1);
  for (let i = 0; i < faceCount * 3; i += 1) degree[indices[i] + 1] += 1;
  for (let v = 0; v < vertexCount; v += 1) degree[v + 1] += degree[v];
  const fill = degree.slice(0, vertexCount);
  const vertexFaces = new Uint32Array(faceCount * 3);
  for (let f = 0; f < faceCount; f += 1) {
    for (let k = 0; k < 3; k += 1) {
      const v = indices[3 * f + k];
      vertexFaces[fill[v]] = f;
      fill[v] += 1;
    }
  }

  const outPositions: number[] = [];
  const outNormals: number[] = [];
  const outIndices = new Uint32Array(faceCount * 3);
  const emitted: Array<number[] | undefined> = new Array(vertexCount);

  for (let f = 0; f < faceCount; f += 1) {
    const fx = faceUnit[3 * f];
    const fy = faceUnit[3 * f + 1];
    const fz = faceUnit[3 * f + 2];
    for (let k = 0; k < 3; k += 1) {
      const v = indices[3 * f + k];
      let sx = 0;
      let sy = 0;
      let sz = 0;
      for (let p = degree[v]; p < degree[v + 1]; p += 1) {
        const g = vertexFaces[p];
        const gx = faceUnit[3 * g];
        const gy = faceUnit[3 * g + 1];
        const gz = faceUnit[3 * g + 2];
        if (fx * gx + fy * gy + fz * gz >= creaseCos) {
          const w = faceArea[g];
          sx += gx * w;
          sy += gy * w;
          sz += gz * w;
        }
      }
      let length = Math.hypot(sx, sy, sz);
      if (length === 0) {
        sx = fx;
        sy = fy;
        sz = fz;
        length = Math.hypot(sx, sy, sz) || 1;
      }
      sx /= length;
      sy /= length;
      sz /= length;

      let target = -1;
      const candidates = emitted[v];
      if (candidates) {
        for (const candidate of candidates) {
          const o = candidate * 3;
          if (outNormals[o] * sx + outNormals[o + 1] * sy + outNormals[o + 2] * sz > 0.9999) {
            target = candidate;
            break;
          }
        }
      }
      if (target < 0) {
        target = outPositions.length / 3;
        outPositions.push(positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]);
        outNormals.push(sx, sy, sz);
        if (candidates) candidates.push(target);
        else emitted[v] = [target];
      }
      outIndices[3 * f + k] = target;
    }
  }

  const firstFace = new Map<number, number>();
  const shared = new Map<number, number>();
  const segments: number[] = [];
  const pushSegment = (a: number, b: number) => {
    segments.push(
      positions[3 * a], positions[3 * a + 1], positions[3 * a + 2],
      positions[3 * b], positions[3 * b + 1], positions[3 * b + 2]
    );
  };
  for (let f = 0; f < faceCount; f += 1) {
    for (let k = 0; k < 3; k += 1) {
      const a = indices[3 * f + k];
      const b = indices[3 * f + ((k + 1) % 3)];
      const lo = a < b ? a : b;
      const hi = a < b ? b : a;
      const key = lo * vertexCount + hi;
      const other = firstFace.get(key);
      if (other === undefined) {
        firstFace.set(key, f);
      } else {
        shared.set(key, (shared.get(key) ?? 1) + 1);
        if ((shared.get(key) ?? 0) === 2) {
          const dot =
            faceUnit[3 * f] * faceUnit[3 * other] +
            faceUnit[3 * f + 1] * faceUnit[3 * other + 1] +
            faceUnit[3 * f + 2] * faceUnit[3 * other + 2];
          if (dot < creaseCos) pushSegment(lo, hi);
        }
      }
    }
  }
  for (const [key, count] of shared) {
    if (count > 2) pushSegment(Math.floor(key / vertexCount), key % vertexCount);
  }
  for (const key of firstFace.keys()) {
    if (!shared.has(key)) pushSegment(Math.floor(key / vertexCount), key % vertexCount);
  }

  return {
    positions: new Float32Array(outPositions),
    normals: new Float32Array(outNormals),
    indices: outIndices,
    edgeSegments: new Float32Array(segments)
  };
}
