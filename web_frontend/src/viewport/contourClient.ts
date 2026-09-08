/**
 * Main-thread half of the contouring worker.
 *
 * Coalescing is the whole point. Scrubbing time or playing back asks for a
 * new front far faster than one can be built, so this keeps exactly one
 * request in flight and one pending, drops everything in between, and ignores
 * replies whose id is no longer the newest. The result is a front that always
 * catches up to where the scrubber actually is instead of crawling through a
 * backlog.
 */
import type { ContourInit, ContourRequest, ContourResult } from '../workers/contourTypes';

export type ContourGeometry = {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
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
  /** Last worker failure, for diagnostics. */
  lastError: string | null = null;
  /** Requests issued and results accepted, for diagnostics. */
  readonly stats = { requested: 0, delivered: 0, lastVertices: 0 };

  constructor(onResult: (geometry: ContourGeometry) => void) {
    this._onResult = onResult;
    this._worker = new Worker(new URL('../workers/contour.ts', import.meta.url), {
      type: 'module'
    });
    // A worker that fails silently is a front that never appears; surface it.
    this._worker.onerror = (event: ErrorEvent) => {
      this.lastError = event.message || 'contour worker failed';
    };
    this._worker.onmessageerror = () => {
      this.lastError = 'contour worker could not decode a message';
    };
    this._worker.onmessage = (event: MessageEvent<ContourResult>) => {
      const message = event.data;
      if (message.kind !== 'contour') return;
      // A reply for a request that has already been superseded describes a
      // moment the scrubber has left.
      const superseded = message.id !== this._inFlight;
      this._inFlight = 0;
      if (!superseded && !this._disposed) {
        this.stats.delivered += 1;
        this.stats.lastVertices = message.vertexCount;
        this._onResult({
          positions: message.positions,
          normals: message.normals,
          indices: message.indices,
          vertexCount: message.vertexCount,
          triangleCount: message.triangleCount,
          elapsedMs: message.elapsedMs
        });
      }
      this._drain();
    };
  }

  /** Hand the worker the lattice and the occupancy mask, once per run. */
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
    // The frames are copied and the copies transferred: the originals stay on
    // this side for probing and for the GPU upload, and a transfer of the
    // originals would detach them.
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
