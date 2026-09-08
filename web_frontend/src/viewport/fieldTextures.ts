/**
 * GPU residency for the dense result grids.
 *
 * One `RawTexture3D` per loaded field, one for the mask, one 256x1 lookup
 * table for the colormap. The important detail is the filtering: the *field*
 * is sampled linearly so the painted color varies continuously across a
 * voxel, while the *mask* is sampled with nearest so the boundary between
 * metal and air stays a hard edge. Filtering the mask would produce a
 * half-voxel band of half-painted fragments all the way around the casting.
 */
import { Constants, RawTexture, RawTexture3D, Texture } from '@babylonjs/core';
import type { Scene } from '@babylonjs/core';

import { colormapLut } from '../lib/colormaps';
import type { ColormapId } from '../lib/viewState';
import { clampVolume } from './fieldPaintPlugin';

export type Dims = [number, number, number];

function createVolume(
  scene: Scene,
  data: Uint8Array,
  dims: Dims,
  samplingMode: number,
  name: string
): RawTexture3D {
  const [nx, ny, nz] = dims;
  const texture = new RawTexture3D(
    data,
    nx,
    ny,
    nz,
    Constants.TEXTUREFORMAT_R,
    scene,
    false,
    false,
    samplingMode,
    Constants.TEXTURETYPE_UNSIGNED_BYTE
  );
  texture.name = name;
  clampVolume(texture);
  return texture;
}

/** The scalar field: linear filtering, so the paint is smooth. */
export function createFieldVolume(scene: Scene, data: Uint8Array, dims: Dims, name = 'field'): RawTexture3D {
  return createVolume(scene, data, dims, Texture.BILINEAR_SAMPLINGMODE, name);
}

/** The occupancy mask: nearest, so "is this metal?" is never half-true. */
export function createMaskVolume(scene: Scene, data: Uint8Array, dims: Dims): RawTexture3D {
  return createVolume(scene, data, dims, Texture.NEAREST_SAMPLINGMODE, 'field-mask');
}

/** A 256x1 RGBA ramp of the chosen colormap, sampled by normalized value. */
export function createColormapTexture(scene: Scene, colormap: ColormapId): RawTexture {
  const texture = RawTexture.CreateRGBATexture(
    colormapLut(colormap, 256),
    256,
    1,
    scene,
    false,
    false,
    Texture.BILINEAR_SAMPLINGMODE
  );
  texture.name = `colormap-${colormap}`;
  texture.wrapU = Texture.CLAMP_ADDRESSMODE;
  texture.wrapV = Texture.CLAMP_ADDRESSMODE;
  return texture;
}

/**
 * Where a frame's bytes come from.
 *
 * A small run keeps its whole stack in memory; a large one keeps a few
 * windows and fetches the rest as the scrubber reaches them. Both answer the
 * same question -- "give me frame n, if you have it" -- so nothing above here
 * has to know which kind it is holding.
 */
export type FrameSource = {
  readonly count: number;
  readonly timesS: number[];
  /** The frame's bytes, or null when it is not resident yet. */
  frame(index: number): Uint8Array | null;
  /** Ask for a frame to be made resident; safe to call every scrubber tick. */
  prefetch(index: number): void;
  /**
   * Resolve once the frame is resident, or false if it never will be.
   *
   * The contouring path needs the actual bytes rather than "not yet": a front
   * that silently skips the frames it does not have simply never appears on a
   * streamed stack.
   */
  whenResident(index: number): Promise<boolean>;
};

/**
 * A frame stack on the GPU, two frames at a time.
 *
 * The whole stack never goes to the GPU: the shader looks at exactly two
 * frames -- the ones the current time sits between -- and uploading ninety
 * more would be tens of megabytes of VRAM for data nothing reads. Re-fetching
 * per frame instead would make playback hostage to the network, which is why
 * the *source* owns the bytes and this owns only the two textures.
 */
export class FrameStackTextures {
  private readonly _source: FrameSource;
  private readonly _textureA: RawTexture3D;
  private readonly _textureB: RawTexture3D;
  private _indexA = -1;
  private _indexB = -1;

  constructor(scene: Scene, source: FrameSource, dims: Dims) {
    this._source = source;
    const blank = new Uint8Array(dims[0] * dims[1] * dims[2]);
    const first = source.frame(0);
    this._textureA = createFieldVolume(scene, first ?? blank, dims, 'frame-a');
    this._textureB = createFieldVolume(scene, first ?? blank, dims, 'frame-b');
    this._indexA = first ? 0 : -1;
    this._indexB = this._indexA;
  }

  get count(): number {
    return this._source.count;
  }

  get timesS(): number[] {
    return this._source.timesS;
  }

  get textureA(): RawTexture3D {
    return this._textureA;
  }

  get textureB(): RawTexture3D {
    return this._textureB;
  }

  /** The two frame indices currently on the GPU. */
  get boundFrames(): [number, number] {
    return [this._indexA, this._indexB];
  }

  /**
   * Point the two textures at the frames bracketing `t` (0..1) and return the
   * lerp weight between them.
   *
   * Frames are spaced by *simulated* time, not evenly: a solidification run
   * takes 93 snapshots that crowd together early and spread out late. Walking
   * the recorded times rather than scaling an index is what makes the
   * scrubber's seconds readout agree with what is on screen.
   */
  seek(t: number): number {
    const count = this.count;
    if (count <= 1) {
      this._bind(0, 0);
      return 0;
    }
    const times = this.timesS;
    const usable = times.length === count && times[times.length - 1] > times[0];
    const progress = Math.min(Math.max(t, 0), 1);

    if (!usable) {
      const position = progress * (count - 1);
      const lower = Math.min(Math.floor(position), count - 2);
      this._bind(lower, lower + 1);
      return position - lower;
    }

    const target = times[0] + progress * (times[times.length - 1] - times[0]);
    let upper = 1;
    while (upper < count - 1 && times[upper] < target) upper += 1;
    const lower = upper - 1;
    const span = times[upper] - times[lower];
    this._bind(lower, upper);
    return span > 1e-9 ? Math.min(Math.max((target - times[lower]) / span, 0), 1) : 0;
  }

  /** The frame indices bracketing a 0..1 position, and the weight between them. */
  bracket(t: number): { lower: number; upper: number; blend: number } {
    const count = this.count;
    if (count <= 1) return { lower: 0, upper: 0, blend: 0 };
    const times = this.timesS;
    const progress = Math.min(Math.max(t, 0), 1);
    if (times.length !== count || times[times.length - 1] <= times[0]) {
      const position = progress * (count - 1);
      const lower = Math.min(Math.floor(position), count - 2);
      return { lower, upper: lower + 1, blend: position - lower };
    }
    const target = times[0] + progress * (times[times.length - 1] - times[0]);
    let upper = 1;
    while (upper < count - 1 && times[upper] < target) upper += 1;
    const lower = upper - 1;
    const span = times[upper] - times[lower];
    return {
      lower,
      upper,
      blend: span > 1e-9 ? Math.min(Math.max((target - times[lower]) / span, 0), 1) : 0
    };
  }

  /** The frame index nearest a 0..1 position, for reading values on the CPU. */
  nearestIndex(t: number): number {
    const count = this.count;
    if (count <= 1) return 0;
    const times = this.timesS;
    const progress = Math.min(Math.max(t, 0), 1);
    if (times.length !== count || times[times.length - 1] <= times[0]) {
      return Math.round(progress * (count - 1));
    }
    const target = times[0] + progress * (times[times.length - 1] - times[0]);
    let best = 0;
    let bestDistance = Infinity;
    for (let index = 0; index < count; index += 1) {
      const distance = Math.abs(times[index] - target);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = index;
      }
    }
    return best;
  }

  /** Raw byte for one cell in one frame, without a GPU readback. */
  sample(frameIndex: number, cellIndex: number): number {
    const frame = this._source.frame(frameIndex);
    return frame ? (frame[cellIndex] ?? 0) : 0;
  }

  /** The bytes of a frame, for the contouring worker. */
  frameData(frameIndex: number): Uint8Array | null {
    return this._source.frame(frameIndex);
  }

  /** Resolve once a frame's bytes are in memory. */
  whenResident(frameIndex: number): Promise<boolean> {
    return this._source.whenResident(frameIndex);
  }

  /** Seconds at a 0..1 scrubber position. */
  timeAt(t: number): number {
    const times = this.timesS;
    if (!times.length) return 0;
    const progress = Math.min(Math.max(t, 0), 1);
    return times[0] + progress * (times[times.length - 1] - times[0]);
  }

  private _bind(indexA: number, indexB: number): void {
    // A frame that is not resident yet leaves the previous one on the GPU.
    // Showing the last known state for a moment is what "without stalls" has
    // to mean while bytes are still in flight; the alternative is a hole.
    if (indexA !== this._indexA) {
      const frame = this._source.frame(indexA);
      if (frame) {
        this._textureA.update(frame);
        this._indexA = indexA;
      } else {
        this._source.prefetch(indexA);
      }
    }
    if (indexB !== this._indexB) {
      const frame = this._source.frame(indexB);
      if (frame) {
        this._textureB.update(frame);
        this._indexB = indexB;
      } else {
        this._source.prefetch(indexB);
      }
    }
    // One step ahead, so playback does not stall at a window boundary.
    this._source.prefetch(Math.min(indexB + 1, this.count - 1));
  }

  dispose(): void {
    this._textureA.dispose();
    this._textureB.dispose();
  }
}

