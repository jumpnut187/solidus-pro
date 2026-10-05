/** Messages between the shell and the modeling worker (`workers/modeling.ts`). */
import type { Box3 } from '../lib/modeling/kernel';
import type { ExtrudeFeature } from '../lib/modeling/types';

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

export type ModelingBuildRequest = {
  kind: 'build';
  id: number;
  feature: ExtrudeFeature;
  /** Bodies the worker has not been sent yet. */
  bodies: BodyPayload[];
  /** The bodies this feature may stop at or trim against, by key. */
  targetKeys: string[];
  /** Every key still in use; the worker frees the rest. */
  liveKeys: string[];
};

export type ModelingBuildResult =
  | {
      kind: 'build';
      id: number;
      ok: true;
      positions: Float32Array;
      indices: Uint32Array;
      volume: number;
      triangles: number;
      bbox: Box3;
      warnings: string[];
      elapsedMs: number;
    }
  | {
      kind: 'build';
      id: number;
      ok: false;
      error: string;
      /** Keys the worker had no body for, so the client can resend them. */
      missingKeys?: string[];
    };
