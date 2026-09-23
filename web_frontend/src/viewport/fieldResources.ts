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

/**
 * How much of a frame stack may sit in memory at once.
 *
 * Below this the whole stack is fetched in one request and held; above it,
 * windows are streamed and the least recently used ones dropped. 32 MB is
 * roughly a 300-frame run of the reference lattice, or a 30-frame run of a
 * million-cell one -- past that the browser tab starts paying for memory it
 * looks at two frames at a time.
 */
const FRAME_MEMORY_BUDGET_BYTES = 32 * 1024 * 1024;

/** Frames per streamed window. Large enough that playback rarely waits. */
const FRAME_WINDOW = 16;

/** Windows kept resident, so scrubbing back a little never refetches. */
const FRAME_WINDOW_CACHE = 4;

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
  /**
   * Signed distance to the parts' real surface, or null if the server could
   * not measure it.
   *
   * The mask is the mesher's opinion of where the metal is, and the mesher
   * counts a voxel as metal when its cube so much as touches the part. On a
   * coarse lattice that leaves the voxel body up to half a voxel outside the
   * STL, and anything clipped against the mask alone -- the liquid front
   * above all -- stands that far proud of the skin. This grid is what the
   * front is clipped against instead: it crosses one half exactly on the
   * surface the casting is drawn with.
   */
  readonly skinTexture: RawTexture3D | null;

  private readonly _scene: Scene;
  private readonly _fields = new Map<ResultModeId, LoadedField>();
  private readonly _stacks = new Map<ResultModeId, FrameStackTextures>();
  private readonly _pending = new Map<ResultModeId, Promise<void>>();
  private _lut: { id: ColormapId; texture: RawTexture } | null = null;
  private _disposed = false;
  /** Called when a streamed window arrives, so the view can repaint with it. */
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
    // A skin grid of the wrong size would bind, sample garbage, and clip the
    // front to noise; better none at all.
    const skinTexture =
      skin && skin.data.length === dims[0] * dims[1] * dims[2] ? createSkinVolume(scene, skin.data, dims) : null;
    return new RunResources(scene, projectId, meta, mask.data, maskTexture, skinTexture);
  }

  /** Register the repaint hook for streamed frame arrivals. */
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
        // Block only on the first window, so the first painted frame is real.
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
    this.skinTexture?.dispose();
    for (const field of this._fields.values()) field.texture.dispose();
    for (const stack of this._stacks.values()) stack.dispose();
    this._fields.clear();
    this._stacks.clear();
    this._lut?.texture.dispose();
    this._lut = null;
  }
}

/** The whole stack in memory: one request, every frame instantly available. */
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
    // Everything is already here.
  }

  whenResident(index: number): Promise<boolean> {
    return Promise.resolve(index >= 0 && index < this.count);
  }
}

/**
 * Windows of the stack, fetched as the scrubber reaches them.
 *
 * For a million-cell lattice a single frame is a megabyte and a hundred of
 * them will not sit in a tab alongside the grids and the geometry. The
 * endpoint already slices by `start` and `count`, so the client keeps a few
 * windows and drops the least recently used -- which for a scrubber that
 * mostly moves forwards means the window behind it.
 *
 * A frame that is not resident is reported as null rather than awaited: the
 * renderer keeps showing the last frame it had while the fetch lands, which
 * is a moment of staleness instead of a stall.
 */
class StreamingFrameSource implements FrameSource {
  private readonly _windows = new Map<number, Uint8Array>();
  /** In-flight fetches, so a second caller awaits the first rather than racing it. */
  private readonly _inFlight = new Map<number, Promise<void>>();

  constructor(
    private readonly _projectId: string,
    private readonly _mode: ResultModeId,
    readonly count: number,
    readonly timesS: number[],
    private readonly _cellCount: number,
    /** Called after a window lands, so the view can repaint with it. */
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
    // Refresh recency: a Map preserves insertion order, so re-inserting is
    // what makes this an LRU.
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
    // Awaits the in-flight fetch when there is one, rather than reporting
    // "not here" and leaving the caller with nothing to wait for.
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
      // A failed window is retried the next time the scrubber asks for it.
    }
  }
}
