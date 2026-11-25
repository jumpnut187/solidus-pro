import type { ProjectPart } from '../api';
import type { BodyPayload, ModelingRequest, ModelingResult } from '../../workers/modelingTypes';
import { partWorldMatrix, type Box3, type CutPreview } from './kernel';
import type { CutFeature, ExtrudeFeature, Sketch, SketchRegion } from './types';

export type BuiltBody = {
  positions: Float32Array;
  indices: Uint32Array;
  volume: number;
  triangles: number;
  bbox: Box3;
  warnings: string[];
  elapsedMs: number;
};

export type CutBody = ({ changed: true } & BuiltBody) | { changed: false; warnings: string[] };

export type CutPreviewBody = CutPreview & { elapsedMs: number };

export type CutSource = { name: string; positions: Float32Array; indices: Uint32Array; matrix: number[] };

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

export function partMatrix(part: ProjectPart): number[] {
  return partWorldMatrix(
    [part.position_x, part.position_y, part.position_z],
    [part.rotation_x, part.rotation_y, part.rotation_z],
    part.scale
  );
}

type Pending = { resolve: (reply: ModelingResult) => void; reject: (error: Error) => void };

type WithoutId<T> = T extends unknown ? Omit<T, 'id'> : never;

export class ModelingClient {
  private readonly _worker: Worker;
  private _nextId = 1;
  private readonly _pending = new Map<number, Pending>();
  private readonly _sent = new Set<string>();
  private _disposed = false;

  constructor() {
    this._worker = new Worker(new URL('../../workers/modeling.ts', import.meta.url), { type: 'module' });
    this._worker.onmessage = (event: MessageEvent<ModelingResult>) => {
      const reply = event.data;
      const pending = this._pending.get(reply.id);
      if (!pending) return;
      this._pending.delete(reply.id);
      if (reply.ok) {
        pending.resolve(reply);
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

  async build(feature: ExtrudeFeature, targets: ProjectPart[], assembly: ProjectPart[]): Promise<BuiltBody> {
    const reply = await this._cached((bodies) => ({ kind: 'build', feature, ...bodies }), targets, assembly);
    if (reply.kind !== 'build' || !reply.ok) throw new ModelingError('Unexpected reply from the modeling worker.');
    const { kind: _kind, id: _id, ok: _ok, ...body } = reply;
    void _kind;
    void _id;
    void _ok;
    return body;
  }

  async previewCut(cut: CutFeature, targets: ProjectPart[], assembly: ProjectPart[]): Promise<CutPreviewBody> {
    const reply = await this._cached((bodies) => ({ kind: 'cutPreview', cut, ...bodies }), targets, assembly);
    if (reply.kind !== 'cutPreview' || !reply.ok) throw new ModelingError('Unexpected reply from the modeling worker.');
    const { kind: _kind, id: _id, ok: _ok, ...body } = reply;
    void _kind;
    void _id;
    void _ok;
    return body;
  }

  async cut(source: CutSource, cuts: CutFeature[]): Promise<CutBody> {
    const target = { ...source, positions: source.positions.slice(), indices: source.indices.slice() };
    const reply = await this._send({ kind: 'cut', cuts, target }, [target.positions.buffer, target.indices.buffer]);
    if (reply.kind !== 'cut' || !reply.ok) throw new ModelingError('Unexpected reply from the modeling worker.');
    if (!reply.changed) return { changed: false, warnings: reply.warnings };
    const { kind: _kind, id: _id, ok: _ok, ...body } = reply;
    void _kind;
    void _id;
    void _ok;
    return body;
  }

  async regions(sketch: Sketch): Promise<SketchRegion[]> {
    const reply = await this._send({ kind: 'regions', sketch }, []);
    if (reply.kind !== 'regions' || !reply.ok) throw new ModelingError('Unexpected reply from the modeling worker.');
    return reply.regions;
  }

  dispose(): void {
    this._disposed = true;
    this._worker.terminate();
    for (const pending of this._pending.values()) pending.reject(new ModelingError('Modeling stopped.'));
    this._pending.clear();
  }

  private async _cached(
    make: (bodies: { bodies: BodyPayload[]; targetKeys: string[]; liveKeys: string[] }) => WithoutId<ModelingRequest>,
    targets: ProjectPart[],
    assembly: ProjectPart[]
  ): Promise<ModelingResult> {
    const attempt = () => {
      const bodies = this._bodies(targets, assembly);
      return this._send(
        make(bodies),
        bodies.bodies.flatMap((body) => [body.positions.buffer, body.indices.buffer])
      );
    };
    try {
      return await attempt();
    } catch (error) {
      const missing = (error as { missingKeys?: string[] }).missingKeys;
      if (!missing?.length) throw error;
      for (const key of missing) this._sent.delete(key);
      return attempt();
    }
  }

  private _bodies(targets: ProjectPart[], assembly: ProjectPart[]) {
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
        matrix: partMatrix(part)
      });
      this._sent.add(key);
    }
    const liveKeys = [...assembly, ...targets].filter((part) => part.meshData).map(partBodyKey);
    const live = new Set(liveKeys);
    for (const key of [...this._sent]) if (!live.has(key)) this._sent.delete(key);
    return { bodies, targetKeys, liveKeys };
  }

  private _send(request: WithoutId<ModelingRequest>, transfer: Transferable[]): Promise<ModelingResult> {
    if (this._disposed) return Promise.reject(new ModelingError('Modeling stopped.'));
    const id = this._nextId;
    this._nextId += 1;
    return new Promise<ModelingResult>((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this._worker.postMessage({ ...request, id }, transfer);
    });
  }
}
