/** Messages between the shell and the modeling worker (`workers/modeling.ts`). */
import type { Box3, CutPreview } from '../lib/modeling/kernel';
import type { CutFeature, ExtrudeFeature, Sketch, SketchRegion } from '../lib/modeling/types';

/** A part's triangles, sent once per geometry-and-placement key and cached by the worker. */
export type BodyPayload = {
  key: string;
  id: string;
  name: string;
  positions: Float32Array;
  indices: Uint32Array;
  /** Column-major world matrix (the engine's part transform). */
  matrix: number[];
};

/** Bodies for a request that works against the cached assembly. */
type CachedBodies = {
  /** Bodies the worker has not been sent yet. */
  bodies: BodyPayload[];
  /** The bodies the request works against, by key. */
  targetKeys: string[];
  /** Every key still in use; the worker frees the rest. */
  liveKeys: string[];
};

export type ModelingBuildRequest = CachedBodies & {
  kind: 'build';
  id: number;
  feature: ExtrudeFeature;
};

/** The material a cut would remove from the bodies in its scope. */
export type ModelingCutPreviewRequest = CachedBodies & {
  kind: 'cutPreview';
  id: number;
  cut: CutFeature;
};

/** Cut one body, sent whole in its own frame (not cached: cuts are committed, not previewed). */
export type ModelingCutRequest = {
  kind: 'cut';
  id: number;
  cuts: CutFeature[];
  target: { name: string; positions: Float32Array; indices: Uint32Array; matrix: number[] };
};

/** The regions a sketch's contours divide the plane into, for picking. */
export type ModelingRegionsRequest = { kind: 'regions'; id: number; sketch: Sketch };

export type ModelingRequest = ModelingBuildRequest | ModelingCutPreviewRequest | ModelingCutRequest | ModelingRegionsRequest;

type Failure = {
  id: number;
  ok: false;
  error: string;
  /** Keys the worker had no body for, so the client can resend them. */
  missingKeys?: string[];
};

export type BuildOutputMessage = {
  positions: Float32Array;
  indices: Uint32Array;
  volume: number;
  triangles: number;
  bbox: Box3;
  warnings: string[];
};

export type ModelingResult =
  | ({ kind: 'build'; id: number; ok: true; elapsedMs: number } & BuildOutputMessage)
  | ({ kind: 'cutPreview'; id: number; ok: true; elapsedMs: number } & CutPreview)
  | ({ kind: 'cut'; id: number; ok: true; elapsedMs: number; changed: true } & BuildOutputMessage)
  | { kind: 'cut'; id: number; ok: true; elapsedMs: number; changed: false; warnings: string[] }
  | { kind: 'regions'; id: number; ok: true; elapsedMs: number; regions: SketchRegion[] }
  | (Failure & { kind: ModelingRequest['kind'] });
