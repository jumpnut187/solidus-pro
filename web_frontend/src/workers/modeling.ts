/// <reference lib="webworker" />
/**
 * The modeling worker: Manifold, the bodies it extrudes against and cuts, and builds.
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

import {
  applyCuts,
  bodyFromMesh,
  buildExtrude,
  FeatureBuildError,
  previewCut,
  sketchRegions,
  type KernelBody
} from '../lib/modeling/kernel';
import type { ModelingRequest, ModelingResult } from './modelingTypes';

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

// Requests run one at a time, in arrival order: each awaits the one before it.
let queue: Promise<void> = Promise.resolve();

scope.onmessage = (event: MessageEvent<ModelingRequest>) => {
  const request = event.data;
  if (!['build', 'cutPreview', 'cut', 'regions'].includes(request.kind)) return;
  queue = queue.then(() => handle(request)).catch(() => undefined);
};

/** Take the bodies a request carries into the cache and free the ones no longer live. */
function syncBodies(
  wasm: ManifoldToplevel,
  request: Extract<ModelingRequest, { kind: 'build' | 'cutPreview' }>
): { targets: KernelBody[]; missing: string[] } {
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
  if (missing.length) return { targets: [], missing };
  return { targets: request.targetKeys.map((key) => bodies.get(key) as KernelBody), missing };
}

async function handle(request: ModelingRequest): Promise<void> {
  const started = performance.now();
  const elapsed = () => performance.now() - started;
  let reply: ModelingResult;
  try {
    const wasm = await manifold();
    if (request.kind === 'regions') {
      const regions = sketchRegions(wasm, request.sketch);
      reply = { kind: 'regions', id: request.id, ok: true, elapsedMs: elapsed(), regions };
      scope.postMessage(reply, regions.map((region) => region.triangles.buffer));
      return;
    }
    if (request.kind === 'cut') {
      const { target } = request;
      const output = applyCuts(wasm, { ...target, matrix: target.matrix as Mat4 }, request.cuts);
      if (output.changed) {
        reply = { kind: 'cut', id: request.id, ok: true, elapsedMs: elapsed(), ...output };
        scope.postMessage(reply, [output.positions.buffer, output.indices.buffer]);
      } else {
        reply = { kind: 'cut', id: request.id, ok: true, elapsedMs: elapsed(), changed: false, warnings: output.warnings };
        scope.postMessage(reply);
      }
      return;
    }
    const { targets, missing } = syncBodies(wasm, request);
    if (missing.length) {
      reply = {
        kind: request.kind,
        id: request.id,
        ok: false,
        error: 'Body geometry is not loaded.',
        missingKeys: missing
      };
    } else if (request.kind === 'cutPreview') {
      const output = previewCut(wasm, request.cut, targets);
      reply = { kind: 'cutPreview', id: request.id, ok: true, elapsedMs: elapsed(), ...output };
      scope.postMessage(reply, [output.positions.buffer, output.indices.buffer]);
      return;
    } else {
      const output = buildExtrude(wasm, request.feature, targets);
      reply = { kind: 'build', id: request.id, ok: true, elapsedMs: elapsed(), ...output };
      scope.postMessage(reply, [output.positions.buffer, output.indices.buffer]);
      return;
    }
  } catch (error) {
    const message =
      error instanceof FeatureBuildError
        ? error.message
        : `The modeling worker failed: ${error instanceof Error ? error.message : String(error)}`;
    reply = { kind: request.kind, id: request.id, ok: false, error: message };
  }
  scope.postMessage(reply);
}
