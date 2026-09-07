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
 * A frame stack held on the CPU, with individual frames uploaded on demand.
 *
 * The whole stack arrives in one response (93 frames of a 41,000-voxel run is
 * about 1.7 MB gzipped), but only the two frames bracketing the current time
 * are ever on the GPU. Uploading all 93 would be 10 MB of VRAM for data the
 * shader looks at two frames at a time, and re-fetching per frame would make
 * playback hostage to the network.
 */
export class FrameStackTextures {
  private readonly _data: Uint8Array;
  private readonly _cellCount: number;
  private readonly _textureA: RawTexture3D;
  private readonly _textureB: RawTexture3D;
  private _indexA = -1;
  private _indexB = -1;

  readonly count: number;
  readonly timesS: number[];

  constructor(scene: Scene, data: Uint8Array, dims: Dims, count: number, timesS: number[]) {
    this._data = data;
    this._cellCount = dims[0] * dims[1] * dims[2];
    this.count = count;
    this.timesS = timesS;
    this._textureA = createFieldVolume(scene, this._frame(0), dims, 'frame-a');
    this._textureB = createFieldVolume(scene, this._frame(0), dims, 'frame-b');
    this._indexA = 0;
    this._indexB = 0;
  }

  private _frame(index: number): Uint8Array {
    const clamped = Math.min(Math.max(index, 0), Math.max(this.count - 1, 0));
    const start = clamped * this._cellCount;
    return this._data.subarray(start, start + this._cellCount);
  }

  get textureA(): RawTexture3D {
    return this._textureA;
  }

  get textureB(): RawTexture3D {
    return this._textureB;
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
    if (this.count <= 1) {
      this._bind(0, 0);
      return 0;
    }
    const times = this.timesS;
    const usable = times.length === this.count && times[times.length - 1] > times[0];
    const progress = Math.min(Math.max(t, 0), 1);

    if (!usable) {
      const position = progress * (this.count - 1);
      const lower = Math.min(Math.floor(position), this.count - 2);
      this._bind(lower, lower + 1);
      return position - lower;
    }

    const target = times[0] + progress * (times[times.length - 1] - times[0]);
    let upper = 1;
    while (upper < this.count - 1 && times[upper] < target) upper += 1;
    const lower = upper - 1;
    const span = times[upper] - times[lower];
    this._bind(lower, upper);
    return span > 1e-9 ? Math.min(Math.max((target - times[lower]) / span, 0), 1) : 0;
  }

  /** The frame index nearest a 0..1 position, for reading values on the CPU. */
  nearestIndex(t: number): number {
    if (this.count <= 1) return 0;
    const times = this.timesS;
    const progress = Math.min(Math.max(t, 0), 1);
    if (times.length !== this.count || times[times.length - 1] <= times[0]) {
      return Math.round(progress * (this.count - 1));
    }
    const target = times[0] + progress * (times[times.length - 1] - times[0]);
    let best = 0;
    let bestDistance = Infinity;
    for (let index = 0; index < this.count; index += 1) {
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
    const clampedFrame = Math.min(Math.max(frameIndex, 0), Math.max(this.count - 1, 0));
    return this._data[clampedFrame * this._cellCount + cellIndex] ?? 0;
  }

  /** Seconds at a 0..1 scrubber position. */
  timeAt(t: number): number {
    const times = this.timesS;
    if (!times.length) return 0;
    const progress = Math.min(Math.max(t, 0), 1);
    return times[0] + progress * (times[times.length - 1] - times[0]);
  }

  private _bind(indexA: number, indexB: number): void {
    if (indexA !== this._indexA) {
      this._textureA.update(this._frame(indexA));
      this._indexA = indexA;
    }
    if (indexB !== this._indexB) {
      this._textureB.update(this._frame(indexB));
      this._indexB = indexB;
    }
  }

  dispose(): void {
    this._textureA.dispose();
    this._textureB.dispose();
  }
}
