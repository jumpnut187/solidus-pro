import type { Box3, CutPreview } from '../lib/modeling/kernel';
import type { CutFeature, ExtrudeFeature, Sketch, SketchRegion } from '../lib/modeling/types';

export type BodyPayload = {
  key: string;
  id: string;
  name: string;
  positions: Float32Array;
  indices: Uint32Array;
  matrix: number[];
};

type CachedBodies = {
  bodies: BodyPayload[];
  targetKeys: string[];
  liveKeys: string[];
};

export type ModelingBuildRequest = CachedBodies & {
  kind: 'build';
  id: number;
  feature: ExtrudeFeature;
};

export type ModelingCutPreviewRequest = CachedBodies & {
  kind: 'cutPreview';
  id: number;
  cut: CutFeature;
};

export type ModelingCutRequest = {
  kind: 'cut';
  id: number;
  cuts: CutFeature[];
  target: { name: string; positions: Float32Array; indices: Uint32Array; matrix: number[] };
};

export type ModelingRegionsRequest = { kind: 'regions'; id: number; sketch: Sketch };

export type ModelingRequest = ModelingBuildRequest | ModelingCutPreviewRequest | ModelingCutRequest | ModelingRegionsRequest;

type Failure = {
  id: number;
  ok: false;
  error: string;
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
