/**
 * Everything a loaded run owns on the GPU, and the loading that fills it.
 *
 * Kept out of the React component so that the component is about *when*
 * things load and this is about *what* gets loaded. The lifetime rule is
 * simple: one `RunResources` per project id, disposed whole when the project
 * or the results version changes. Individual fields are fetched lazily the
 * first time their mode is selected and then cached, so switching back and
 * forth between two modes costs one request each, not one per switch.
 */
import type { RawTexture, RawTexture3D, Scene } from '@babylonjs/core';

import { fetchFieldGrid, fetchFrames, fetchGridMeta, fetchMaskGrid } from '../lib/gridApi';
import type { ColormapId, GridMeta, ResultModeId } from '../lib/viewState';
import { createColormapTexture, createFieldVolume, createMaskVolume, FrameStackTextures } from './fieldTextures';
import type { Dims } from './fieldTextures';

/** Modes served as a stack of time frames rather than as one static grid. */
export const FRAME_MODES: readonly ResultModeId[] = ['fraction_liquid', 'fill_animation'];

export function isFrameMode(mode: ResultModeId): boolean {
  return FRAME_MODES.includes(mode);
}

/** Which `GridMeta.frames` key backs a given mode. */
function frameKey(mode: ResultModeId): 'fraction_liquid' | 'fill' | null {
  if (mode === 'fraction_liquid') return 'fraction_liquid';
  if (mode === 'fill_animation') return 'fill';
  return null;
}

export type LoadedField = {
  texture: RawTexture3D;
  /** The bytes, kept for probing and volume measurement without a GPU readback. */
  data: Uint8Array;
  min: number;
  max: number;
};

export class RunResources {
  readonly projectId: string;
  readonly meta: GridMeta;
  readonly dims: Dims;
  readonly maskTexture: RawTexture3D;
  readonly mask: Uint8Array;

  private readonly _scene: Scene;
  private readonly _fields = new Map<ResultModeId, LoadedField>();
  private readonly _stacks = new Map<ResultModeId, FrameStackTextures>();
  private readonly _pending = new Map<ResultModeId, Promise<void>>();
  private _lut: { id: ColormapId; texture: RawTexture } | null = null;
  private _disposed = false;

  private constructor(
    scene: Scene,
    projectId: string,
    meta: GridMeta,
    mask: Uint8Array,
    maskTexture: RawTexture3D
  ) {
    this._scene = scene;
    this.projectId = projectId;
    this.meta = meta;
    this.dims = [meta.dims[0], meta.dims[1], meta.dims[2]];
    this.mask = mask;
    this.maskTexture = maskTexture;
  }

  static async load(
    scene: Scene,
    projectId: string,
    signal: AbortSignal
  ): Promise<RunResources> {
    const meta = await fetchGridMeta(projectId, signal);
    const mask = await fetchMaskGrid(projectId, signal);
    const dims: Dims = [meta.dims[0], meta.dims[1], meta.dims[2]];
    const maskTexture = createMaskVolume(scene, mask.data, dims);
    return new RunResources(scene, projectId, meta, mask.data, maskTexture);
  }

  get disposed(): boolean {
    return this._disposed;
  }

  field(mode: ResultModeId): LoadedField | null {
    return this._fields.get(mode) ?? null;
  }

  stack(mode: ResultModeId): FrameStackTextures | null {
    return this._stacks.get(mode) ?? null;
  }

  /** True when the mode is already resident and can be painted this frame. */
  has(mode: ResultModeId): boolean {
    return isFrameMode(mode) ? this._stacks.has(mode) : this._fields.has(mode);
  }

  /**
   * Fetch a mode if it is not already loaded.
   *
   * Concurrent requests for the same mode share one promise: clicking through
   * a result menu quickly must not start four downloads of the same grid.
   */
  ensure(mode: ResultModeId, signal: AbortSignal): Promise<void> {
    if (this._disposed || this.has(mode)) return Promise.resolve();
    const existing = this._pending.get(mode);
    if (existing) return existing;

    const load = this._load(mode, signal).finally(() => {
      this._pending.delete(mode);
    });
    this._pending.set(mode, load);
    return load;
  }

  private async _load(mode: ResultModeId, signal: AbortSignal): Promise<void> {
    if (isFrameMode(mode)) {
      const key = frameKey(mode);
      const info = key ? this.meta.frames[key] : undefined;
      if (!info || info.count <= 0) return;
      const frames = await fetchFrames(this.projectId, mode, { signal });
      if (this._disposed || frames.count <= 0) return;
      this._stacks.set(
        mode,
        new FrameStackTextures(this._scene, frames.data, this.dims, frames.count, info.times_s)
      );
      return;
    }

    const grid = await fetchFieldGrid(this.projectId, mode, signal);
    if (this._disposed) return;
    this._fields.set(mode, {
      texture: createFieldVolume(this._scene, grid.data, this.dims, `field-${mode}`),
      data: grid.data,
      min: grid.min,
      max: grid.max
    });
  }

  /** The colormap ramp, rebuilt only when the chosen map actually changes. */
  lut(colormap: ColormapId): RawTexture {
    if (this._lut && this._lut.id === colormap) return this._lut.texture;
    this._lut?.texture.dispose();
    const texture = createColormapTexture(this._scene, colormap);
    this._lut = { id: colormap, texture };
    return texture;
  }

  /** Flat cell index for a world point, or -1 when it is off the lattice. */
  cellIndexAt(x: number, y: number, z: number): number {
    const [nx, ny, nz] = this.dims;
    const { origin, spacing } = this.meta;
    if (!(spacing > 0)) return -1;
    const ix = Math.round((x - origin[0]) / spacing);
    const iy = Math.round((y - origin[1]) / spacing);
    const iz = Math.round((z - origin[2]) / spacing);
    if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) return -1;
    return ix + nx * (iy + ny * iz);
  }

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this.maskTexture.dispose();
    for (const field of this._fields.values()) field.texture.dispose();
    for (const stack of this._stacks.values()) stack.dispose();
    this._fields.clear();
    this._stacks.clear();
    this._lut?.texture.dispose();
    this._lut = null;
  }
}
