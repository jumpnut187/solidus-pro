/** Message types shared by the contouring worker and its client. */

export type ContourInit = {
  kind: 'init';
  dims: [number, number, number];
  origin: [number, number, number];
  spacing: number;
  /** Occupancy grid: 0 outside the metal, else part-type code + 1. */
  mask: Uint8Array;
};

/**
 * One contour job.
 *
 * The two frames travel with the request rather than living in the worker. A
 * stack can be tens of megabytes -- more than the frame budget allows to be
 * resident at once, which is why large runs stream windows of it -- so the
 * worker is handed exactly the two frames the front sits between and keeps
 * nothing. At about a hundred kilobytes a frame for a typical lattice the
 * copy costs far less than the contour it feeds.
 */
export type ContourRequest = {
  kind: 'contour';
  /** Correlates the reply with the request; stale replies are dropped. */
  id: number;
  frameA: Uint8Array;
  frameB: Uint8Array;
  /** Weight of the second frame, 0..1. */
  blend: number;
  /** Iso level in the frames' own normalized 0..1 units. */
  iso: number;
  /** Restrict the surface to cast bodies. */
  castOnly: boolean;
  /** True when the region of interest is where the field is *below* the iso. */
  invert: boolean;
};

export type ContourResult = {
  kind: 'contour';
  id: number;
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  /**
   * Per-vertex field value sampled a short way *inside* the surface.
   *
   * Sampling on the surface itself is useless for colour: a level set sits at
   * the iso value by construction, so every vertex would come back 0.5 and the
   * front would be one flat tone. A step inward reports how deep the liquid
   * runs behind that point -- near 1 where the vertex caps a real pool, barely
   * over the iso where it caps a sliver about to freeze -- which is what makes
   * the pool read yellow in its body and orange-red around the solidifying
   * rim.
   */
  scalars: Float32Array;
  vertexCount: number;
  triangleCount: number;
  /** Milliseconds spent contouring, for the performance budget. */
  elapsedMs: number;
};

export type ContourMessage = ContourInit | ContourRequest;
