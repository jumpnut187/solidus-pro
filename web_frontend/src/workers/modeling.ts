/// <reference lib="webworker" />
/**
 * The modeling worker: Manifold, the bodies it extrudes against, and builds.
 *
 * Booleans against a casting of a few hundred thousand triangles take tens to
 * hundreds of milliseconds, and the extrude preview reruns on every dimension
 * edit; on the main thread that is a frozen viewport. Welding a part's STL
 * into a Manifold is the expensive part and the same for every build, so each
 * body is welded once, kept here under its geometry-and-placement key, and
 * freed when the client stops listing it as live.
 */
import Module from 'manifold-3d';
import type { ManifoldToplevel, Mat4 } from 'manifold-3d';
import wasmUrl from 'manifold-3d/manifold.wasm?url';

import { bodyFromMesh, buildExtrude, FeatureBuildError, type KernelBody } from '../lib/modeling/kernel';
import type { ModelingBuildRequest, ModelingBuildResult } from './modelingTypes';

const scope = self as unknown as DedicatedWorkerGlobalScope;

let wasmPromise: Promise<ManifoldToplevel> | null = null;
function manifold(): Promise<ManifoldToplevel> {
  if (!wasmPromise) {
    wasmPromise = Module({ locateFile: () => wasmUrl }).then((wasm) => {
      wasm.setup();
      return wasm;
    });
  }
  return wasmPromise;
}

const bodies = new Map<string, KernelBody>();

function freeBody(body: KernelBody): void {
  try {
    body.manifold?.delete();
  } catch {
    /* already freed */
  }
}

// Builds run one at a time, in arrival order: each awaits the one before it.
let queue: Promise<void> = Promise.resolve();

scope.onmessage = (event: MessageEvent<ModelingBuildRequest>) => {
  const request = event.data;
  if (request.kind !== 'build') return;
  queue = queue.then(() => handle(request)).catch(() => undefined);
};

async function handle(request: ModelingBuildRequest): Promise<void> {
  const started = performance.now();
  let reply: ModelingBuildResult;
  try {
    const wasm = await manifold();
    for (const payload of request.bodies) {
      const previous = bodies.get(payload.key);
      if (previous) freeBody(previous);
      bodies.set(
        payload.key,
        bodyFromMesh(wasm, payload.id, payload.name, payload.positions, payload.indices, payload.matrix as Mat4)
      );
    }
    const live = new Set(request.liveKeys);
    for (const [key, body] of bodies) {
      if (!live.has(key)) {
        freeBody(body);
        bodies.delete(key);
      }
    }
    const missing = request.targetKeys.filter((key) => !bodies.has(key));
    if (missing.length) {
      reply = { kind: 'build', id: request.id, ok: false, error: 'Body geometry is not loaded.', missingKeys: missing };
    } else {
      const targets = request.targetKeys.map((key) => bodies.get(key) as KernelBody);
      const output = buildExtrude(wasm, request.feature, targets);
      reply = {
        kind: 'build',
        id: request.id,
        ok: true,
        ...output,
        elapsedMs: performance.now() - started
      };
      scope.postMessage(reply, [output.positions.buffer, output.indices.buffer]);
      return;
    }
  } catch (error) {
    const message =
      error instanceof FeatureBuildError
        ? error.message
        : `The modeling worker failed: ${error instanceof Error ? error.message : String(error)}`;
    reply = { kind: 'build', id: request.id, ok: false, error: message };
  }
  scope.postMessage(reply);
}
