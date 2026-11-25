import type { ContourInit, ContourRequest, ContourResult } from '../workers/contourTypes';

export type ContourGeometry = {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  scalars: Float32Array;
  vertexCount: number;
  triangleCount: number;
  elapsedMs: number;
};

export type ContourParams = Omit<ContourRequest, 'kind' | 'id'>;

export class ContourClient {
  private readonly _worker: Worker;
  private _nextId = 1;
  private _inFlight = 0;
  private _pending: ContourParams | null = null;
  private _onResult: (geometry: ContourGeometry) => void;
  private _disposed = false;
  lastError: string | null = null;
  readonly stats = { requested: 0, delivered: 0, lastVertices: 0 };

  constructor(onResult: (geometry: ContourGeometry) => void) {
    this._onResult = onResult;
    this._worker = new Worker(new URL('../workers/contour.ts', import.meta.url), {
      type: 'module'
    });
    this._worker.onerror = (event: ErrorEvent) => {
      this.lastError = event.message || 'contour worker failed';
    };
    this._worker.onmessageerror = () => {
      this.lastError = 'contour worker could not decode a message';
    };
    this._worker.onmessage = (event: MessageEvent<ContourResult>) => {
      const message = event.data;
      if (message.kind !== 'contour') return;
      const superseded = message.id !== this._inFlight;
      this._inFlight = 0;
      if (!superseded && !this._disposed) {
        this.stats.delivered += 1;
        this.stats.lastVertices = message.vertexCount;
        this._onResult({
          positions: message.positions,
          normals: message.normals,
          indices: message.indices,
          scalars: message.scalars,
          vertexCount: message.vertexCount,
          triangleCount: message.triangleCount,
          elapsedMs: message.elapsedMs
        });
      }
      this._drain();
    };
  }

  load(init: Omit<ContourInit, 'kind'>): void {
    this._worker.postMessage({ kind: 'init', ...init } satisfies ContourInit);
  }

  request(params: ContourParams): void {
    this._pending = params;
    this._drain();
  }

  private _drain(): void {
    if (this._disposed || this._inFlight !== 0 || !this._pending) return;
    const params = this._pending;
    this._pending = null;
    const id = this._nextId++;
    this._inFlight = id;
    this.stats.requested += 1;
    const frameA = params.frameA.slice();
    const frameB = params.frameB.slice();
    this._worker.postMessage({ kind: 'contour', id, ...params, frameA, frameB } satisfies ContourRequest, [
      frameA.buffer,
      frameB.buffer
    ]);
  }

  dispose(): void {
    this._disposed = true;
    this._pending = null;
    this._worker.terminate();
  }
}
