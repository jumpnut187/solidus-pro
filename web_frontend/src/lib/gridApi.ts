/**
 * Typed client for the dense field grids the viewport renders.
 *
 * Metadata arrives as JSON; grids and frame stacks arrive as raw typed
 * arrays. The layout is x-fastest -- `index = ix + nx * (iy + ny * iz)` --
 * which is exactly a WebGL 3D texture of width `nx`, so the bytes go from
 * `fetch` into `RawTexture3D` with no repacking.
 *
 * Every fetcher takes an `AbortSignal`. Loading results is the one place in
 * this application where a user can outrun the network -- clicking through
 * result modes queues a 100 KB request each -- and an unaborted response
 * arriving late would paint the previous mode over the current one.
 */
import type { GridMeta, ResultModeId } from './viewState';

export class GridRequestError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'GridRequestError';
    this.status = status;
  }
}

/** A scalar grid plus the range needed to turn its bytes back into field units. */
export type FieldGridData = {
  mode: ResultModeId;
  /** One byte per cell, x-fastest. */
  data: Uint8Array;
  dims: [number, number, number];
  /** `value = min + (byte / 255) * (max - min)`. */
  min: number;
  max: number;
  unit: string;
};

/** A window of consecutive time frames, each one a full grid. */
export type FrameStackData = {
  mode: ResultModeId;
  /** `count * nx * ny * nz` bytes, frame-major. */
  data: Uint8Array;
  dims: [number, number, number];
  start: number;
  count: number;
  total: number;
  min: number;
  max: number;
};

function projectBase(projectId: string): string {
  return `/api/projects/${encodeURIComponent(projectId)}/grid`;
}

async function failure(response: Response): Promise<GridRequestError> {
  let message = `Request failed: ${response.status}`;
  try {
    const body = await response.json();
    if (body && typeof body.error === 'string') message = body.error;
  } catch {
    // A binary endpoint that failed may not have produced JSON at all.
  }
  return new GridRequestError(message, response.status);
}

function parseDims(header: string | null): [number, number, number] {
  const parts = (header ?? '').split(',').map((value) => Number.parseInt(value, 10));
  if (parts.length !== 3 || parts.some((value) => !Number.isFinite(value) || value <= 0)) {
    return [0, 0, 0];
  }
  return [parts[0], parts[1], parts[2]];
}

function parseNumber(header: string | null, fallback: number): number {
  const value = Number.parseFloat(header ?? '');
  return Number.isFinite(value) ? value : fallback;
}

async function fetchBinary(url: string, signal?: AbortSignal): Promise<Response> {
  const response = await fetch(url, { signal, headers: { Accept: 'application/octet-stream' } });
  if (!response.ok) throw await failure(response);
  return response;
}

/**
 * Dimensions, placement, available modes, ranges, frame times and statistics.
 *
 * Always the first call: everything else needs the dims to make sense of the
 * bytes, and the shell needs the mode list to build its menu.
 */
export async function fetchGridMeta(projectId: string, signal?: AbortSignal): Promise<GridMeta> {
  const response = await fetch(projectBase(projectId), { signal, headers: { Accept: 'application/json' } });
  if (!response.ok) throw await failure(response);
  return (await response.json()) as GridMeta;
}

/** The occupancy grid: 0 outside the metal, otherwise a part-type code + 1. */
export async function fetchMaskGrid(
  projectId: string,
  signal?: AbortSignal
): Promise<{ data: Uint8Array; dims: [number, number, number] }> {
  const response = await fetchBinary(`${projectBase(projectId)}/mask`, signal);
  const buffer = await response.arrayBuffer();
  return { data: new Uint8Array(buffer), dims: parseDims(response.headers.get('X-Grid-Dims')) };
}

/** One dense scalar grid, quantized to bytes. */
export async function fetchFieldGrid(
  projectId: string,
  mode: ResultModeId,
  signal?: AbortSignal
): Promise<FieldGridData> {
  const response = await fetchBinary(`${projectBase(projectId)}/${encodeURIComponent(mode)}`, signal);
  const buffer = await response.arrayBuffer();
  return {
    mode,
    data: new Uint8Array(buffer),
    dims: parseDims(response.headers.get('X-Grid-Dims')),
    min: parseNumber(response.headers.get('X-Field-Min'), 0),
    max: parseNumber(response.headers.get('X-Field-Max'), 1),
    unit: response.headers.get('X-Field-Unit') ?? ''
  };
}

/** The same grid in field units, for callers that want floats rather than bytes. */
export async function fetchFieldGridFloat(
  projectId: string,
  mode: ResultModeId,
  signal?: AbortSignal
): Promise<{ data: Float32Array; dims: [number, number, number]; min: number; max: number }> {
  const response = await fetchBinary(
    `${projectBase(projectId)}/${encodeURIComponent(mode)}?dtype=f32`,
    signal
  );
  const buffer = await response.arrayBuffer();
  return {
    data: new Float32Array(buffer),
    dims: parseDims(response.headers.get('X-Grid-Dims')),
    min: parseNumber(response.headers.get('X-Field-Min'), 0),
    max: parseNumber(response.headers.get('X-Field-Max'), 1)
  };
}

/**
 * A window of time frames.
 *
 * `count` defaults to the whole stack. A 93-frame run of a 41,000-voxel
 * casting is 9.5 MB raw and about 1.7 MB over a gzipped connection, so
 * fetching the lot in one request is cheaper than paging it.
 */
export async function fetchFrames(
  projectId: string,
  mode: ResultModeId,
  options: { start?: number; count?: number; signal?: AbortSignal } = {}
): Promise<FrameStackData> {
  const params = new URLSearchParams();
  if (options.start !== undefined) params.set('start', String(options.start));
  if (options.count !== undefined) params.set('count', String(options.count));
  const query = params.toString();
  const url = `${projectBase(projectId)}/${encodeURIComponent(mode)}/frames${query ? `?${query}` : ''}`;

  const response = await fetchBinary(url, options.signal);
  const buffer = await response.arrayBuffer();
  return {
    mode,
    data: new Uint8Array(buffer),
    dims: parseDims(response.headers.get('X-Grid-Dims')),
    start: Math.trunc(parseNumber(response.headers.get('X-Frame-Start'), 0)),
    count: Math.trunc(parseNumber(response.headers.get('X-Frame-Count'), 0)),
    total: Math.trunc(parseNumber(response.headers.get('X-Frame-Total'), 0)),
    min: parseNumber(response.headers.get('X-Field-Min'), 0),
    max: parseNumber(response.headers.get('X-Field-Max'), 1)
  };
}

/** A threshold isosurface, as the buffers a vertex buffer wants. */
export type EnvelopeData = {
  mode: ResultModeId;
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  /** Per-vertex field value in field units, when the server sampled it. */
  scalars: Float32Array | null;
  vertexCount: number;
  triangleCount: number;
  threshold: number;
  min: number;
  max: number;
  unit: string;
};

export type EnvelopeOptions = {
  /** Iso level in **field units**, not percent. */
  threshold: number;
  castOnly?: boolean;
  /** Skip the per-vertex sampling pass when the colour is not needed. */
  scalars?: boolean;
  /** Frame index, for the animated modes. */
  frame?: number;
  signal?: AbortSignal;
};

/**
 * Fetch the threshold isosurface for a mode.
 *
 * The body is one little-endian block -- positions, normals, indices, then
 * optional scalars -- with the counts in headers. Each section is a whole
 * number of 4-byte words, so the typed-array views below are windows onto the
 * response buffer rather than copies of it.
 */
export async function fetchEnvelope(
  projectId: string,
  mode: ResultModeId,
  options: EnvelopeOptions
): Promise<EnvelopeData> {
  const params = new URLSearchParams({ threshold: String(options.threshold) });
  if (options.castOnly) params.set('cast_only', '1');
  if (options.scalars === false) params.set('scalars', '0');
  if (options.frame !== undefined) params.set('frame', String(options.frame));

  const response = await fetchBinary(
    `${projectBase(projectId)}/${encodeURIComponent(mode)}/isosurface?${params}`,
    options.signal
  );
  const buffer = await response.arrayBuffer();

  const vertexCount = Math.trunc(parseNumber(response.headers.get('X-Vertex-Count'), 0));
  const indexCount = Math.trunc(parseNumber(response.headers.get('X-Index-Count'), 0));
  const hasScalar = response.headers.get('X-Has-Scalar') === '1';

  let offset = 0;
  const positions = new Float32Array(buffer, offset, vertexCount * 3);
  offset += vertexCount * 3 * 4;
  const normals = new Float32Array(buffer, offset, vertexCount * 3);
  offset += vertexCount * 3 * 4;
  const indices = new Uint32Array(buffer, offset, indexCount);
  offset += indexCount * 4;
  const scalars = hasScalar ? new Float32Array(buffer, offset, vertexCount) : null;

  return {
    mode,
    positions,
    normals,
    indices,
    scalars,
    vertexCount,
    triangleCount: Math.trunc(indexCount / 3),
    threshold: parseNumber(response.headers.get('X-Threshold'), options.threshold),
    min: parseNumber(response.headers.get('X-Field-Min'), 0),
    max: parseNumber(response.headers.get('X-Field-Max'), 1),
    unit: response.headers.get('X-Field-Unit') ?? ''
  };
}

/** True when a rejection is the caller's own abort rather than a real failure. */
export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}
