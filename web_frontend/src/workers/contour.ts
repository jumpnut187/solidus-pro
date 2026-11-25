import type { ContourInit, ContourMessage, ContourRequest, ContourResult } from './contourTypes';

const INSIDE_SAMPLE_CELLS = 2;

const CORNERS: ReadonlyArray<readonly [number, number, number]> = [
  [0, 0, 0],
  [1, 0, 0],
  [0, 1, 0],
  [1, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [0, 1, 1],
  [1, 1, 1]
];

const EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [2, 3],
  [4, 5],
  [6, 7],
  [0, 2],
  [1, 3],
  [4, 6],
  [5, 7],
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7]
];

type State = {
  dims: [number, number, number];
  origin: [number, number, number];
  spacing: number;
  mask: Uint8Array;
  cellCount: number;
  field: Float32Array;
  blurValue: Float32Array;
  blurWeight: Float32Array;
  blurValueSwap: Float32Array;
  blurWeightSwap: Float32Array;
  nearest: Map<string, Int32Array>;
};

let state: State | null = null;

self.onmessage = (event: MessageEvent<ContourMessage>) => {
  const message = event.data;
  if (message.kind === 'init') {
    state = initialize(message);
    return;
  }
  if (message.kind === 'contour') {
    const result = contour(message);
    if (result) {
      (self as unknown as Worker).postMessage(result, [
        result.positions.buffer,
        result.normals.buffer,
        result.indices.buffer,
        result.scalars.buffer
      ]);
    }
  }
};

function initialize(message: ContourInit): State {
  const [nx, ny, nz] = message.dims;
  const cells = nx * ny * nz;
  return {
    dims: message.dims,
    origin: message.origin,
    spacing: message.spacing,
    mask: message.mask,
    cellCount: cells,
    field: new Float32Array(cells),
    blurValue: new Float32Array(cells),
    blurWeight: new Float32Array(cells),
    blurValueSwap: new Float32Array(cells),
    blurWeightSwap: new Float32Array(cells),
    nearest: new Map()
  };
}

function nearestUsableMap(current: State, castOnly: boolean): Int32Array | null {
  const key = castOnly ? 'cast' : 'all';
  const cached = current.nearest.get(key);
  if (cached) return cached;

  const [nx, ny, nz] = current.dims;
  const { mask, cellCount } = current;
  const nearest = new Int32Array(cellCount).fill(-1);
  const queue = new Int32Array(cellCount);
  let head = 0;
  let tail = 0;

  for (let index = 0; index < cellCount; index += 1) {
    const code = mask[index];
    if (code !== 0 && (!castOnly || code === CAST_CODE)) {
      nearest[index] = index;
      queue[tail] = index;
      tail += 1;
    }
  }
  if (tail === 0) return null;

  const strideY = nx;
  const strideZ = nx * ny;
  while (head < tail) {
    const index = queue[head];
    head += 1;
    const ix = index % nx;
    const iy = ((index / nx) | 0) % ny;
    const iz = (index / strideZ) | 0;
    const source = nearest[index];

    if (ix > 0) tail = visit(nearest, queue, tail, index - 1, source);
    if (ix < nx - 1) tail = visit(nearest, queue, tail, index + 1, source);
    if (iy > 0) tail = visit(nearest, queue, tail, index - strideY, source);
    if (iy < ny - 1) tail = visit(nearest, queue, tail, index + strideY, source);
    if (iz > 0) tail = visit(nearest, queue, tail, index - strideZ, source);
    if (iz < nz - 1) tail = visit(nearest, queue, tail, index + strideZ, source);
  }

  current.nearest.set(key, nearest);
  return nearest;
}

function visit(
  nearest: Int32Array,
  queue: Int32Array,
  tail: number,
  neighbour: number,
  source: number
): number {
  if (nearest[neighbour] !== -1) return tail;
  nearest[neighbour] = source;
  queue[tail] = neighbour;
  return tail + 1;
}

const CAST_CODE = 1;

function buildField(current: State, request: ContourRequest): boolean {
  const { cellCount, mask, field, blurValue, blurWeight } = current;
  const a = request.frameA;
  const b = request.frameB;
  const blend = Math.min(Math.max(request.blend, 0), 1);

  for (let index = 0; index < cellCount; index += 1) {
    const code = mask[index];
    const usable = code !== 0 && (!request.castOnly || code === CAST_CODE);
    blurWeight[index] = usable ? 1 : 0;
    blurValue[index] = usable ? (a[index] * (1 - blend) + b[index] * blend) / 255 : 0;
  }

  blurInsideMask(current, Math.min(Math.max(request.blurStrength ?? 1, 0), 1));

  const nearest = nearestUsableMap(current, request.castOnly);
  if (!nearest) return false;
  for (let index = 0; index < cellCount; index += 1) {
    field[index] = blurValue[nearest[index]];
  }
  return true;
}

function blurInsideMask(current: State, strength: number): void {
  const [nx, ny, nz] = current.dims;
  if (strength <= 0) return;
  const raw = current.blurValue.slice();
  let value = current.blurValue;
  let weight = current.blurWeight;
  let valueOut = current.blurValueSwap;
  let weightOut = current.blurWeightSwap;

  const strides = [1, nx, nx * ny];
  const limits = [nx, ny, nz];

  for (let axis = 0; axis < 3; axis += 1) {
    const stride = strides[axis];
    const limit = limits[axis];
    if (limit < 3) {
      continue;
    }
    for (let iz = 0; iz < nz; iz += 1) {
      for (let iy = 0; iy < ny; iy += 1) {
        for (let ix = 0; ix < nx; ix += 1) {
          const index = ix + nx * (iy + ny * iz);
          const position = axis === 0 ? ix : axis === 1 ? iy : iz;
          const low = position > 0 ? index - stride : index;
          const high = position < limit - 1 ? index + stride : index;
          valueOut[index] = value[low] + 2 * value[index] + value[high];
          weightOut[index] = weight[low] + 2 * weight[index] + weight[high];
        }
      }
    }
    [value, valueOut] = [valueOut, value];
    [weight, weightOut] = [weightOut, weight];
  }

  const cellCount = current.cellCount;
  for (let index = 0; index < cellCount; index += 1) {
    const total = weight[index];
    const blurred = total > 0 ? value[index] / total : 0;
    current.blurValue[index] = raw[index] + (blurred - raw[index]) * strength;
  }
}

function emptyResult(id: number, started: number): ContourResult {
  return {
    kind: 'contour',
    id,
    positions: new Float32Array(0),
    normals: new Float32Array(0),
    indices: new Uint32Array(0),
    scalars: new Float32Array(0),
    vertexCount: 0,
    triangleCount: 0,
    elapsedMs: performance.now() - started
  };
}

function contour(request: ContourRequest): ContourResult | null {
  const current = state;
  if (!current) return null;
  const started = performance.now();
  if (!buildField(current, request)) return emptyResult(request.id, started);

  const [nx, ny, nz] = current.dims;
  const { field, spacing, origin } = current;
  const iso = request.iso;
  const sign = request.invert ? -1 : 1;

  const sampleAt = (ix: number, iy: number, iz: number): number =>
    field[ix + nx * (iy + ny * iz)];
  const insideAt = (value: number): boolean => sign * (value - iso) >= 0;

  const cellVertex = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const positions: number[] = [];
  const normals: number[] = [];
  const scalars: number[] = [];

  const usableCell = (ix: number, iy: number, iz: number): boolean => {
    if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) return false;
    const code = current.mask[ix + nx * (iy + ny * iz)];
    return code !== 0 && (!request.castOnly || code === CAST_CODE);
  };

  const usableAt = (fx: number, fy: number, fz: number): boolean =>
    usableCell(Math.round(fx), Math.round(fy), Math.round(fz));

  const sampleLattice = (fx: number, fy: number, fz: number): number => {
    const cx = Math.min(Math.max(fx, 0), nx - 1);
    const cy = Math.min(Math.max(fy, 0), ny - 1);
    const cz = Math.min(Math.max(fz, 0), nz - 1);
    const x0 = Math.floor(cx);
    const y0 = Math.floor(cy);
    const z0 = Math.floor(cz);
    const x1 = Math.min(x0 + 1, nx - 1);
    const y1 = Math.min(y0 + 1, ny - 1);
    const z1 = Math.min(z0 + 1, nz - 1);
    const tx = cx - x0;
    const ty = cy - y0;
    const tz = cz - z0;
    const at = (ix: number, iy: number, iz: number) => field[ix + nx * (iy + ny * iz)];
    const c00 = at(x0, y0, z0) + (at(x1, y0, z0) - at(x0, y0, z0)) * tx;
    const c10 = at(x0, y1, z0) + (at(x1, y1, z0) - at(x0, y1, z0)) * tx;
    const c01 = at(x0, y0, z1) + (at(x1, y0, z1) - at(x0, y0, z1)) * tx;
    const c11 = at(x0, y1, z1) + (at(x1, y1, z1) - at(x0, y1, z1)) * tx;
    const c0 = c00 + (c10 - c00) * ty;
    const c1 = c01 + (c11 - c01) * ty;
    return c0 + (c1 - c0) * tz;
  };

  const cellIndex = (ix: number, iy: number, iz: number): number =>
    ix + (nx - 1) * (iy + (ny - 1) * iz);

  const corner = new Float64Array(8);

  for (let iz = 0; iz < nz - 1; iz += 1) {
    for (let iy = 0; iy < ny - 1; iy += 1) {
      for (let ix = 0; ix < nx - 1; ix += 1) {
        let metalCorners = 0;
        for (let c = 0; c < 8; c += 1) {
          const [dx, dy, dz] = CORNERS[c];
          if (usableCell(ix + dx, iy + dy, iz + dz)) metalCorners += 1;
        }
        if (metalCorners === 0) continue;

        let insideCount = 0;
        for (let c = 0; c < 8; c += 1) {
          const [dx, dy, dz] = CORNERS[c];
          const value = sampleAt(ix + dx, iy + dy, iz + dz);
          corner[c] = value;
          if (insideAt(value)) insideCount += 1;
        }
        if (insideCount === 0 || insideCount === 8) continue;

        let sumX = 0;
        let sumY = 0;
        let sumZ = 0;
        let crossings = 0;
        for (let e = 0; e < 12; e += 1) {
          const [c0, c1] = EDGES[e];
          const v0 = corner[c0];
          const v1 = corner[c1];
          if (insideAt(v0) === insideAt(v1)) continue;
          const denominator = v1 - v0;
          const t = Math.abs(denominator) < 1e-9 ? 0.5 : (iso - v0) / denominator;
          const clamped = Math.min(Math.max(t, 0), 1);
          const p0 = CORNERS[c0];
          const p1 = CORNERS[c1];
          sumX += p0[0] + (p1[0] - p0[0]) * clamped;
          sumY += p0[1] + (p1[1] - p0[1]) * clamped;
          sumZ += p0[2] + (p1[2] - p0[2]) * clamped;
          crossings += 1;
        }
        if (crossings === 0) continue;

        const localX = sumX / crossings;
        const localY = sumY / crossings;
        const localZ = sumZ / crossings;

        cellVertex[cellIndex(ix, iy, iz)] = positions.length / 3;
        positions.push(
          Math.fround(origin[0] + (ix + localX) * spacing),
          Math.fround(origin[1] + (iy + localY) * spacing),
          Math.fround(origin[2] + (iz + localZ) * spacing)
        );

        const gx =
          sampleAt(Math.min(ix + 1, nx - 1), iy, iz) - sampleAt(Math.max(ix - 1, 0), iy, iz);
        const gy =
          sampleAt(ix, Math.min(iy + 1, ny - 1), iz) - sampleAt(ix, Math.max(iy - 1, 0), iz);
        const gz =
          sampleAt(ix, iy, Math.min(iz + 1, nz - 1)) - sampleAt(ix, iy, Math.max(iz - 1, 0));
        const scale = -sign;
        const length = Math.hypot(gx, gy, gz) || 1;
        const nxv = (gx * scale) / length;
        const nyv = (gy * scale) / length;
        const nzv = (gz * scale) / length;
        normals.push(nxv, nyv, nzv);

        let depth = INSIDE_SAMPLE_CELLS;
        while (
          depth > 0 &&
          !usableAt(ix + localX - nxv * depth, iy + localY - nyv * depth, iz + localZ - nzv * depth)
        ) {
          depth -= 0.5;
        }
        scalars.push(
          sampleLattice(
            ix + localX - nxv * depth,
            iy + localY - nyv * depth,
            iz + localZ - nzv * depth
          )
        );
      }
    }
  }

  const indices: number[] = [];
  const quad = (a: number, b: number, c: number, d: number, flip: boolean) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (flip) {
      indices.push(a, c, b, a, d, c);
    } else {
      indices.push(a, b, c, a, c, d);
    }
  };

  for (let iz = 0; iz < nz - 1; iz += 1) {
    for (let iy = 0; iy < ny - 1; iy += 1) {
      for (let ix = 0; ix < nx - 1; ix += 1) {
        const here = insideAt(sampleAt(ix, iy, iz));

        if (iy > 0 && iz > 0 && here !== insideAt(sampleAt(ix + 1, iy, iz))) {
          quad(
            cellVertex[cellIndex(ix, iy - 1, iz - 1)],
            cellVertex[cellIndex(ix, iy, iz - 1)],
            cellVertex[cellIndex(ix, iy, iz)],
            cellVertex[cellIndex(ix, iy - 1, iz)],
            here
          );
        }
        if (ix > 0 && iz > 0 && here !== insideAt(sampleAt(ix, iy + 1, iz))) {
          quad(
            cellVertex[cellIndex(ix - 1, iy, iz - 1)],
            cellVertex[cellIndex(ix, iy, iz - 1)],
            cellVertex[cellIndex(ix, iy, iz)],
            cellVertex[cellIndex(ix - 1, iy, iz)],
            !here
          );
        }
        if (ix > 0 && iy > 0 && here !== insideAt(sampleAt(ix, iy, iz + 1))) {
          quad(
            cellVertex[cellIndex(ix - 1, iy - 1, iz)],
            cellVertex[cellIndex(ix, iy - 1, iz)],
            cellVertex[cellIndex(ix, iy, iz)],
            cellVertex[cellIndex(ix - 1, iy, iz)],
            here
          );
        }
      }
    }
  }

  return {
    kind: 'contour',
    id: request.id,
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices),
    scalars: new Float32Array(scalars),
    vertexCount: positions.length / 3,
    triangleCount: indices.length / 3,
    elapsedMs: performance.now() - started
  };
}
