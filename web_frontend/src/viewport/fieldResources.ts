import type { RawTexture } from '@babylonjs/core/Materials/Textures/rawTexture';
import type { RawTexture3D } from '@babylonjs/core/Materials/Textures/rawTexture3D';
import type { Scene } from '@babylonjs/core/scene';

import { fetchFieldGrid, fetchFrames, fetchGridMeta, fetchMaskGrid, fetchSkinGrid } from '../lib/gridApi';
import type { ColormapId, GridMeta, ResultModeId } from '../lib/viewState';
import {
  createColormapTexture,
  createFieldVolume,
  createMaskVolume,
  createSkinVolume,
  FrameStackTextures
} from './fieldTextures';
import type { Dims, FrameSource } from './fieldTextures';

const FRAME_MEMORY_BUDGET_BYTES = 32 * 1024 * 1024;

const FRAME_WINDOW = 16;

const FRAME_WINDOW_CACHE = 4;

export const FRAME_MODES: readonly ResultModeId[] = ['fraction_liquid', 'fill_animation'];

export function isFrameMode(mode: ResultModeId): boolean {
  return FRAME_MODES.includes(mode);
}

function frameKey(mode: ResultModeId): 'fraction_liquid' | 'fill' | null {
  if (mode === 'fraction_liquid') return 'fraction_liquid';
  if (mode === 'fill_animation') return 'fill';
  return null;
}

export type LoadedField = {
  texture: RawTexture3D;
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
  readonly skinTexture: RawTexture3D | null;

  private readonly _scene: Scene;
  private readonly _fields = new Map<ResultModeId, LoadedField>();
  private readonly _stacks = new Map<ResultModeId, FrameStackTextures>();
  private readonly _pending = new Map<ResultModeId, Promise<void>>();
  private _lut: { id: ColormapId; texture: RawTexture } | null = null;
  private _disposed = false;
  private _onFramesLoaded: () => void = () => undefined;

  private constructor(
    scene: Scene,
    projectId: string,
    meta: GridMeta,
    mask: Uint8Array,
    maskTexture: RawTexture3D,
    skinTexture: RawTexture3D | null
  ) {
    this._scene = scene;
    this.projectId = projectId;
    this.meta = meta;
    this.dims = [meta.dims[0], meta.dims[1], meta.dims[2]];
    this.mask = mask;
    this.maskTexture = maskTexture;
    this.skinTexture = skinTexture;
  }

  static async load(
    scene: Scene,
    projectId: string,
    signal: AbortSignal
  ): Promise<RunResources> {
    const meta = await fetchGridMeta(projectId, signal);
    const [mask, skin] = await Promise.all([fetchMaskGrid(projectId, signal), fetchSkinGrid(projectId, signal)]);
    const dims: Dims = [meta.dims[0], meta.dims[1], meta.dims[2]];
    const maskTexture = createMaskVolume(scene, mask.data, dims);
    const skinTexture =
      skin && skin.data.length === dims[0] * dims[1] * dims[2] ? createSkinVolume(scene, skin.data, dims) : null;
    return new RunResources(scene, projectId, meta, mask.data, maskTexture, skinTexture);
  }

  onFramesLoaded(callback: () => void): void {
    this._onFramesLoaded = callback;
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

  has(mode: ResultModeId): boolean {
    return isFrameMode(mode) ? this._stacks.has(mode) : this._fields.has(mode);
  }

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
      const cellCount = this.dims[0] * this.dims[1] * this.dims[2];
      const totalBytes = cellCount * info.count;

      let source: FrameSource;
      if (totalBytes <= FRAME_MEMORY_BUDGET_BYTES) {
        const frames = await fetchFrames(this.projectId, mode, { signal });
        if (this._disposed || frames.count <= 0) return;
        source = new EagerFrameSource(frames.data, frames.count, info.times_s, cellCount);
      } else {
        source = new StreamingFrameSource(
          this.projectId,
          mode,
          info.count,
          info.times_s,
          cellCount,
          () => this._onFramesLoaded()
        );
        await (source as StreamingFrameSource).ensure(0, signal);
        if (this._disposed) return;
      }
      this._stacks.set(mode, new FrameStackTextures(this._scene, source, this.dims));
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

  lut(colormap: ColormapId): RawTexture {
    if (this._lut && this._lut.id === colormap) return this._lut.texture;
    this._lut?.texture.dispose();
    const texture = createColormapTexture(this._scene, colormap);
    this._lut = { id: colormap, texture };
    return texture;
  }

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
    this.skinTexture?.dispose();
    for (const field of this._fields.values()) field.texture.dispose();
    for (const stack of this._stacks.values()) stack.dispose();
    this._fields.clear();
    this._stacks.clear();
    this._lut?.texture.dispose();
    this._lut = null;
  }
}

class EagerFrameSource implements FrameSource {
  constructor(
    private readonly _data: Uint8Array,
    readonly count: number,
    readonly timesS: number[],
    private readonly _cellCount: number
  ) {}

  frame(index: number): Uint8Array | null {
    if (index < 0 || index >= this.count) return null;
    const start = index * this._cellCount;
    return this._data.subarray(start, start + this._cellCount);
  }

  prefetch(): void {
  }

  whenResident(index: number): Promise<boolean> {
    return Promise.resolve(index >= 0 && index < this.count);
  }
}

class StreamingFrameSource implements FrameSource {
  private readonly _windows = new Map<number, Uint8Array>();
  private readonly _inFlight = new Map<number, Promise<void>>();

  constructor(
    private readonly _projectId: string,
    private readonly _mode: ResultModeId,
    readonly count: number,
    readonly timesS: number[],
    private readonly _cellCount: number,
    private readonly _onWindow: () => void = () => undefined
  ) {}

  private _windowOf(index: number): number {
    return Math.floor(index / FRAME_WINDOW);
  }

  frame(index: number): Uint8Array | null {
    if (index < 0 || index >= this.count) return null;
    const windowIndex = this._windowOf(index);
    const window = this._windows.get(windowIndex);
    if (!window) {
      this.prefetch(index);
      return null;
    }
    this._windows.delete(windowIndex);
    this._windows.set(windowIndex, window);

    const offset = (index - windowIndex * FRAME_WINDOW) * this._cellCount;
    if (offset + this._cellCount > window.length) return null;
    return window.subarray(offset, offset + this._cellCount);
  }

  prefetch(index: number): void {
    if (index < 0 || index >= this.count) return;
    void this.ensure(index);
  }

  async whenResident(index: number): Promise<boolean> {
    if (index < 0 || index >= this.count) return false;
    if (this._windows.has(this._windowOf(index))) return true;
    await this.ensure(index);
    return this._windows.has(this._windowOf(index));
  }

  ensure(index: number, signal?: AbortSignal): Promise<void> {
    const windowIndex = this._windowOf(index);
    if (this._windows.has(windowIndex)) return Promise.resolve();
    const existing = this._inFlight.get(windowIndex);
    if (existing) return existing;

    const load = this._fetchWindow(windowIndex, signal).finally(() => {
      this._inFlight.delete(windowIndex);
    });
    this._inFlight.set(windowIndex, load);
    return load;
  }

  private async _fetchWindow(windowIndex: number, signal?: AbortSignal): Promise<void> {
    try {
      const start = windowIndex * FRAME_WINDOW;
      const result = await fetchFrames(this._projectId, this._mode, {
        start,
        count: Math.min(FRAME_WINDOW, this.count - start),
        signal
      });
      if (result.count > 0) {
        this._windows.set(windowIndex, result.data);
        while (this._windows.size > FRAME_WINDOW_CACHE) {
          const oldest = this._windows.keys().next().value;
          if (oldest === undefined) break;
          this._windows.delete(oldest);
        }
        this._onWindow();
      }
    } catch {
    }
  }
}
