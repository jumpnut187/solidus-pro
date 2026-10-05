/**
 * Main-thread half of the modeling worker.
 *
 * Sends each part's triangles to the worker once (keyed by geometry and
 * placement, so moving a part or replacing its file re-sends it) and turns
 * build replies back into promises.
 */
import type { ProjectPart } from '../api';
import type { BodyPayload, ModelingBuildRequest, ModelingBuildResult } from '../../workers/modelingTypes';
import { partWorldMatrix, type Box3 } from './kernel';
import type { ExtrudeFeature } from './types';

export type BuiltBody = {
  positions: Float32Array;
  indices: Uint32Array;
  volume: number;
  triangles: number;
  bbox: Box3;
  warnings: string[];
  elapsedMs: number;
};

/** A feature the kernel refused; the message is for the user. */
export class ModelingError extends Error {}

export function partBodyKey(part: ProjectPart): string {
  const mesh = part.meshData;
  return [
    part.id,
    part.filepath,
    mesh?.vertex_count ?? 0,
    mesh?.face_count ?? 0,
    part.position_x,
    part.position_y,
    part.position_z,
    part.rotation_x,
    part.rotation_y,
    part.rotation_z,
    part.scale
  ].join('|');
}

type Pending = { resolve: (body: BuiltBody) => void; reject: (error: Error) => void };

export class ModelingClient {
  private readonly _worker: Worker;
  private _nextId = 1;
  private readonly _pending = new Map<number, Pending>();
  /** Keys whose triangles the worker already holds. */
  private readonly _sent = new Set<string>();
  private _disposed = false;

  constructor() {
    this._worker = new Worker(new URL('../../workers/modeling.ts', import.meta.url), { type: 'module' });
    this._worker.onmessage = (event: MessageEvent<ModelingBuildResult>) => {
      const reply = event.data;
      const pending = this._pending.get(reply.id);
      if (!pending) return;
      this._pending.delete(reply.id);
      if (reply.ok) {
        const { kind: _kind, id: _id, ok: _ok, ...body } = reply;
        void _kind;
        void _id;
        void _ok;
        pending.resolve(body);
      } else {
        const error = new ModelingError(reply.error);
        (error as ModelingError & { missingKeys?: string[] }).missingKeys = reply.missingKeys;
        pending.reject(error);
      }
    };
    this._worker.onerror = (event: ErrorEvent) => {
      const error = new ModelingError(event.message || 'The modeling worker stopped.');
      for (const pending of this._pending.values()) pending.reject(error);
      this._pending.clear();
      this._sent.clear();
    };
  }

  /**
   * Build `feature` against `targets`. `assembly` is every part currently
   * loaded, which tells the worker which cached bodies it may free.
   */
  async build(feature: ExtrudeFeature, targets: ProjectPart[], assembly: ProjectPart[]): Promise<BuiltBody> {
    try {
      return await this._request(feature, targets, assembly);
    } catch (error) {
      const missing = (error as { missingKeys?: string[] }).missingKeys;
      if (!missing?.length) throw error;
      // The worker lost its cache (a reload of the worker): send again, once.
      for (const key of missing) this._sent.delete(key);
      return this._request(feature, targets, assembly);
    }
  }

  dispose(): void {
    this._disposed = true;
    this._worker.terminate();
    for (const pending of this._pending.values()) pending.reject(new ModelingError('Modeling stopped.'));
    this._pending.clear();
  }

  private _request(feature: ExtrudeFeature, targets: ProjectPart[], assembly: ProjectPart[]): Promise<BuiltBody> {
    if (this._disposed) return Promise.reject(new ModelingError('Modeling stopped.'));
    const bodies: BodyPayload[] = [];
    const targetKeys: string[] = [];
    for (const part of targets) {
      const key = partBodyKey(part);
      targetKeys.push(key);
      if (this._sent.has(key) || !part.meshData) continue;
      bodies.push({
        key,
        id: part.id,
        name: part.name,
        positions: new Float32Array(part.meshData.vertices),
        indices: new Uint32Array(part.meshData.faces),
        matrix: partWorldMatrix(
          [part.position_x, part.position_y, part.position_z],
          [part.rotation_x, part.rotation_y, part.rotation_z],
          part.scale
        )
      });
      this._sent.add(key);
    }
    const liveKeys = assembly.filter((part) => part.meshData).map(partBodyKey);
    const live = new Set(liveKeys);
    for (const key of [...this._sent]) if (!live.has(key)) this._sent.delete(key);

    const id = this._nextId;
    this._nextId += 1;
    const request: ModelingBuildRequest = { kind: 'build', id, feature, bodies, targetKeys, liveKeys };
    return new Promise<BuiltBody>((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this._worker.postMessage(
        request,
        bodies.flatMap((body) => [body.positions.buffer, body.indices.buffer])
      );
    });
  }
}
