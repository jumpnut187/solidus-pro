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

/**
 * How far inside the surface the colour scalar is sampled, in lattice cells.
 *
 * Under about one cell the sample is still on the iso and the front comes back
 * flat. Much over three and a thin pool samples clean through itself and out
 * the far side, which paints its middle cold. Two is a fair reading of "just
 * behind this surface" at every lattice this renderer sees.
 */
const INSIDE_SAMPLE_CELLS = 2;

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
  /**
   * For every cell, the nearest cell that is usable metal. Built once per run
   * from the mask, which does not change; keyed by whether the caller is
   * restricting to cast bodies, because that changes what "usable" means.
   */
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

/**
 * Nearest usable metal cell for every cell in the lattice.
 *
 * Multi-source breadth-first search out of the metal, six-connected, so the
 * result is exact in the lattice's own step metric and costs one pass over the
 * grid. It is built once per run because the mask never changes -- doing it
 * per request would put a full grid traversal in the middle of playback.
 *
 * The map exists so that the field handed to the contourer has no artificial
 * background; see `buildField`.
 */
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
  // A run with no cast bodies at all under `castOnly`: there is nothing to
  // spread, and the caller falls back to leaving the field empty.
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

/** Mask byte for cast metal: the engine's CAST code (0) plus one. */
const CAST_CODE = 1;

/**
 * Build the scalar field for this request: two frames blended, then extended.
 *
 * The mask still decides which cells carry real data -- the stored grids are
 * hole-filled so that painting the casting's *surface* never fades to zero at
 * the boundary, so every cell in the bounding box carries a plausible value
 * and contouring that unmasked would grow the pool out through the mold.
 *
 * What the mask must *not* do is put a wall of some sentinel value around the
 * metal. That is what this used to do, and it is what made the front look like
 * a shell. With the mold pinned below the iso, fully liquid metal put the 0.5
 * level set exactly on the metal boundary -- so at t = 0 the contourer
 * faithfully produced a second copy of the casting's own surface, sitting a
 * depth-bias away from the painted shell and z-fighting into stripes along the
 * riser wall, with a sawtooth rim wherever the lattice cut the plate at an
 * angle. Every vertex of it also sampled its colour out in the mold, which is
 * where the stray yellow and blue came from.
 *
 * Instead, cells outside the metal take the value of the nearest cell inside
 * it. The field is then continuous across the boundary, so there is no
 * crossing there and no surface: the front exists only where liquid meets
 * solid *within* the metal, which is the only place a front physically is.
 * Where liquid runs to the mold wall the surface simply ends, which is why the
 * envelope material is two-sided. Anything the extension does produce out
 * beyond the casting is discarded per fragment against the same mask.
 */
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
function blurInsideMask(current: State, strength: number): void {
  const [nx, ny, nz] = current.dims;
  if (strength <= 0) return;
  // Kept so the blurred result can be mixed back over the raw values; a
  // partial blur is what stops the 0.5 crossing drifting out to the wall.
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

  // One vertex per straddling cell; -1 marks a cell that produced none.
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

  /** Trilinear sample in lattice coordinates, clamped at the border. */
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
        // A cell with no metal in it at all cannot hold a front.
        //
        // The field no longer has a wall around the casting, which is what
        // stopped the front copying the skin -- but it does mean the extension
        // out in the mold carries real values, and where the extension of a
        // frozen region meets the extension of a liquid one, out in the air
        // beside the riser, it crosses the iso. That crossing is not a front;
        // it is an artefact of the extension, and it was the molten tongues
        // running down the outside of the riser wall.
        //
        // The test is deliberately "any corner" rather than "every corner".
        // Every corner also removes the tongues, but it takes the front out of
        // the whole boundary cell layer with it, and where the frozen shell is
        // thinner than a cell -- the fillet at the riser base, at 0.26 in --
        // that left the liquid body open. The vertex itself is checked instead,
        // below, which is the guarantee that actually matters.
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

        // The vertex must land in metal. This is the promise the renderer
        // relies on: no part of the front hangs off the casting, so nothing
        // has to be trimmed away afterwards to hide a drip.
        //
        // Tested on the *stored* coordinates, rounded to float32 exactly as
        // the output buffer will hold them. A vertex sitting on a cell
        // boundary otherwise answers this question one way here and the other
        // way to anything that reads the buffer back.
        const worldX = Math.fround(origin[0] + (ix + localX) * spacing);
        const worldY = Math.fround(origin[1] + (iy + localY) * spacing);
        const worldZ = Math.fround(origin[2] + (iz + localZ) * spacing);
        if (
          !usableCell(
            Math.round((worldX - origin[0]) / spacing),
            Math.round((worldY - origin[1]) / spacing),
            Math.round((worldZ - origin[2]) / spacing)
          )
        ) {
          continue;
        }

        cellVertex[cellIndex(ix, iy, iz)] = positions.length / 3;
        positions.push(worldX, worldY, worldZ);

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
        const nxv = (gx * scale) / length;
        const nyv = (gy * scale) / length;
        const nzv = (gz * scale) / length;
        normals.push(nxv, nyv, nzv);

        // Step back along the outward normal -- into the region the surface
        // encloses -- and record what the field is there. See
        // `ContourResult.scalars`.
        //
        // The step is shortened until it lands in metal. A front that runs
        // along the wall of the casting has an inward normal pointing straight
        // out of the part, so a fixed two-cell step walks into the mold and
        // reads the extended field there; that is what put stray colours on
        // the vertices at the rim. Falling back through one cell to none keeps
        // every sample on a value the metal actually holds.
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
    scalars: new Float32Array(scalars),
    vertexCount: positions.length / 3,
    triangleCount: indices.length / 3,
    elapsedMs: performance.now() - started
  };
}
