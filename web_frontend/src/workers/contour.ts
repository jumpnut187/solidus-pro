/**
 * Isosurface extraction for the time modes, off the main thread.
 *
 * The solidification front moves every frame, so its surface cannot be
 * fetched: at 93 frames and four playback speeds the server would be asked
 * for a new contour dozens of times a second. It is extracted here instead,
 * from the frame stack the client already holds, in a worker so that a 50 ms
 * contour never lands in the middle of a rendered frame.
 *
 * The algorithm is naive surface nets rather than marching cubes. Both walk
 * the same sign changes; the difference is that surface nets places one
 * vertex per straddling cell at the average of its edge crossings and joins
 * neighbouring cells with quads, instead of emitting a triangle soup from a
 * 256-case table. That buys three things worth having here: the output is
 * naturally smooth, which is what a liquid pool should look like without a
 * separate smoothing pass; vertices are shared, so a front over this lattice
 * is thousands of vertices rather than tens of thousands; and there is no
 * lookup table to get subtly wrong.
 *
 * Normals come from the central-difference gradient of the field at each
 * vertex, not from face normals, so the surface shades smoothly even where
 * the lattice is coarse relative to the curvature.
 */
import type { ContourInit, ContourMessage, ContourRequest, ContourResult } from './contourTypes';

/** Corner offsets of a cell, in the canonical order used by the edge table. */
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

/** The twelve edges of a cell as corner index pairs. */
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
  /** Scratch field, rebuilt per request. */
  field: Float32Array;
  /** Scratch buffers for the masked blur; allocated once, reused per request. */
  blurValue: Float32Array;
  blurWeight: Float32Array;
  blurValueSwap: Float32Array;
  blurWeightSwap: Float32Array;
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
        result.indices.buffer
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
    blurWeightSwap: new Float32Array(cells)
  };
}

/**
 * Build the scalar field for this request: two frames blended, then masked.
 *
 * Masking is not optional. The stored grids are hole-filled so that painting
 * the casting's *surface* never fades to zero at the boundary, which means
 * every cell in the bounding box carries a plausible value. Contouring that
 * unmasked would grow the liquid pool out through the mold.
 */
function buildField(current: State, request: ContourRequest): void {
  const { cellCount, mask, field, blurValue, blurWeight } = current;
  const a = request.frameA;
  const b = request.frameB;
  const blend = Math.min(Math.max(request.blend, 0), 1);
  const castCode = 1;

  for (let index = 0; index < cellCount; index += 1) {
    const code = mask[index];
    const usable = code !== 0 && (!request.castOnly || code === castCode);
    blurWeight[index] = usable ? 1 : 0;
    blurValue[index] = usable ? (a[index] * (1 - blend) + b[index] * blend) / 255 : 0;
  }

  blurInsideMask(current);

  // Cells outside the metal are pushed clear of any iso level so the surface
  // closes against them rather than passing through. This happens *after* the
  // blur, so the front still ends exactly at the metal boundary.
  const outside = request.invert ? 2 : -1;
  for (let index = 0; index < cellCount; index += 1) {
    const code = mask[index];
    const usable = code !== 0 && (!request.castOnly || code === castCode);
    field[index] = usable ? blurValue[index] : outside;
  }
}

/**
 * A 1-2-1 blur that only ever averages cells inside the metal.
 *
 * A mold-filling frame is not a smooth field: it is close to 1 in filled
 * metal and close to 0 ahead of the front, and individual cells dip below
 * half for entirely physical reasons -- a cell straddling the free surface,
 * residual from the lattice solve, the pore fraction inside a filter.
 * Contouring that raw at 0.5 turns every one of them into a hole, and the
 * filled region comes out looking shot through rather than solid.
 *
 * The blur is normalized: value and weight are blurred separately and then
 * divided, so a cell at the wall averages only with the metal beside it. A
 * plain blur would mix in the zeros of the mold and pull the front back off
 * every wall, eroding exactly the surface being drawn.
 */
function blurInsideMask(current: State): void {
  const [nx, ny, nz] = current.dims;
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
      // Nothing to average along an axis one or two cells deep.
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
    current.blurValue[index] = total > 0 ? value[index] / total : 0;
  }
}

function contour(request: ContourRequest): ContourResult | null {
  const current = state;
  if (!current) return null;
  const started = performance.now();
  buildField(current, request);

  const [nx, ny, nz] = current.dims;
  const { field, spacing, origin } = current;
  const iso = request.iso;
  const sign = request.invert ? -1 : 1;

  const sampleAt = (ix: number, iy: number, iz: number): number =>
    field[ix + nx * (iy + ny * iz)];
  const insideAt = (value: number): boolean => sign * (value - iso) >= 0;

  // One vertex per straddling cell; -1 marks a cell that produced none.
  const cellVertex = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const positions: number[] = [];
  const normals: number[] = [];

  const cellIndex = (ix: number, iy: number, iz: number): number =>
    ix + (nx - 1) * (iy + (ny - 1) * iz);

  const corner = new Float64Array(8);

  for (let iz = 0; iz < nz - 1; iz += 1) {
    for (let iy = 0; iy < ny - 1; iy += 1) {
      for (let ix = 0; ix < nx - 1; ix += 1) {
        let insideCount = 0;
        for (let c = 0; c < 8; c += 1) {
          const [dx, dy, dz] = CORNERS[c];
          const value = sampleAt(ix + dx, iy + dy, iz + dz);
          corner[c] = value;
          if (insideAt(value)) insideCount += 1;
        }
        // All in or all out: no surface passes through this cell.
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
          origin[0] + (ix + localX) * spacing,
          origin[1] + (iy + localY) * spacing,
          origin[2] + (iz + localZ) * spacing
        );

        // Central-difference gradient, one cell out, clamped at the border.
        const gx =
          sampleAt(Math.min(ix + 1, nx - 1), iy, iz) - sampleAt(Math.max(ix - 1, 0), iy, iz);
        const gy =
          sampleAt(ix, Math.min(iy + 1, ny - 1), iz) - sampleAt(ix, Math.max(iy - 1, 0), iz);
        const gz =
          sampleAt(ix, iy, Math.min(iz + 1, nz - 1)) - sampleAt(ix, iy, Math.max(iz - 1, 0));
        // The gradient points uphill; the outward normal points away from the
        // solid side, so it is the negated gradient for a "high is inside"
        // field and the gradient itself for an inverted one.
        const scale = -sign;
        const length = Math.hypot(gx, gy, gz) || 1;
        normals.push((gx * scale) / length, (gy * scale) / length, (gz * scale) / length);
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

  // One quad per sign-changing lattice edge, joining the four cells around it.
  for (let iz = 1; iz < nz - 1; iz += 1) {
    for (let iy = 1; iy < ny - 1; iy += 1) {
      for (let ix = 1; ix < nx - 1; ix += 1) {
        const here = insideAt(sampleAt(ix, iy, iz));

        if (ix < nx - 1 && here !== insideAt(sampleAt(ix + 1, iy, iz))) {
          quad(
            cellVertex[cellIndex(ix, iy - 1, iz - 1)],
            cellVertex[cellIndex(ix, iy, iz - 1)],
            cellVertex[cellIndex(ix, iy, iz)],
            cellVertex[cellIndex(ix, iy - 1, iz)],
            here
          );
        }
        if (iy < ny - 1 && here !== insideAt(sampleAt(ix, iy + 1, iz))) {
          quad(
            cellVertex[cellIndex(ix - 1, iy, iz - 1)],
            cellVertex[cellIndex(ix, iy, iz - 1)],
            cellVertex[cellIndex(ix, iy, iz)],
            cellVertex[cellIndex(ix - 1, iy, iz)],
            !here
          );
        }
        if (iz < nz - 1 && here !== insideAt(sampleAt(ix, iy, iz + 1))) {
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
    vertexCount: positions.length / 3,
    triangleCount: indices.length / 3,
    elapsedMs: performance.now() - started
  };
}
