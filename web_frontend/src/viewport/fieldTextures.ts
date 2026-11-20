import { Constants } from '@babylonjs/core/Engines/constants';
import { RawTexture } from '@babylonjs/core/Materials/Textures/rawTexture';
import { RawTexture3D } from '@babylonjs/core/Materials/Textures/rawTexture3D';
import { Texture } from '@babylonjs/core/Materials/Textures/texture';
import type { Scene } from '@babylonjs/core/scene';

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

export function createFieldVolume(scene: Scene, data: Uint8Array, dims: Dims, name = 'field'): RawTexture3D {
  return createVolume(scene, data, dims, Texture.BILINEAR_SAMPLINGMODE, name);
}

export function createMaskVolume(scene: Scene, data: Uint8Array, dims: Dims): RawTexture3D {
  return createVolume(scene, data, dims, Texture.NEAREST_SAMPLINGMODE, 'field-mask');
}

export function createSkinVolume(scene: Scene, data: Uint8Array, dims: Dims): RawTexture3D {
  return createVolume(scene, data, dims, Texture.BILINEAR_SAMPLINGMODE, 'field-skin');
}

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

export type FrameSource = {
  readonly count: number;
  readonly timesS: number[];
  frame(index: number): Uint8Array | null;
  prefetch(index: number): void;
  whenResident(index: number): Promise<boolean>;
};

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

  get boundFrames(): [number, number] {
    return [this._indexA, this._indexB];
  }

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

  sample(frameIndex: number, cellIndex: number): number {
    const frame = this._source.frame(frameIndex);
    return frame ? (frame[cellIndex] ?? 0) : 0;
  }

  frameData(frameIndex: number): Uint8Array | null {
    return this._source.frame(frameIndex);
  }

  whenResident(frameIndex: number): Promise<boolean> {
    return this._source.whenResident(frameIndex);
  }

  timeAt(t: number): number {
    const times = this.timesS;
    if (!times.length) return 0;
    const progress = Math.min(Math.max(t, 0), 1);
    return times[0] + progress * (times[times.length - 1] - times[0]);
  }

  private _bind(indexA: number, indexB: number): void {
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
    this._source.prefetch(Math.min(indexB + 1, this.count - 1));
  }

  dispose(): void {
    this._textureA.dispose();
    this._textureB.dispose();
  }
}
