import { VertexBuffer } from '@babylonjs/core/Buffers/buffer';
import { PBRMaterial } from '@babylonjs/core/Materials/PBR/pbrMaterial';
import { Color3 } from '@babylonjs/core/Maths/math.color';
import { Mesh } from '@babylonjs/core/Meshes/mesh';
import { VertexData } from '@babylonjs/core/Meshes/mesh.vertexData';
import type { Scene } from '@babylonjs/core/scene';

import { sampleColormap } from '../lib/colormaps';
import { FieldPaintPluginMaterial } from './fieldPaintPlugin';
import { MoltenGlowPluginMaterial } from './moltenGlowPlugin';
import type { ColormapId } from '../lib/viewState';

export type EnvelopeGeometry = {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  scalars: Float32Array | null;
};

export type EnvelopeStyle = {
  colormap: ColormapId;
  scaleMin: number;
  scaleMax: number;
  flatColor: Color3;
  alpha: number;
  metallic: number;
  roughness: number;
  glow: number;
};

export class EnvelopeMesh {
  private readonly _mesh: Mesh;
  private readonly _material: PBRMaterial;
  private readonly _glow: MoltenGlowPluginMaterial;
  private readonly _clip: FieldPaintPluginMaterial;
  private _scalars: Float32Array | null = null;
  private _vertexCount = 0;

  constructor(
    scene: Scene,
    name: string,
    createClip: (material: PBRMaterial) => FieldPaintPluginMaterial
  ) {
    this._mesh = new Mesh(name, scene);
    this._mesh.isPickable = false;
    this._mesh.setEnabled(false);
    this._material = new PBRMaterial(`${name}-mat`, scene);
    this._material.backFaceCulling = false;
    this._material.twoSidedLighting = true;
    this._material.albedoColor = Color3.White();
    this._material.zOffset = 0;
    this._glow = new MoltenGlowPluginMaterial(this._material);
    this._clip = createClip(this._material);
    this._clip.state.strength = 0;
    this._mesh.useVertexColors = true;
    this._mesh.material = this._material;
  }

  get clip(): FieldPaintPluginMaterial {
    return this._clip;
  }

  get mesh(): Mesh {
    return this._mesh;
  }

  get material(): PBRMaterial {
    return this._material;
  }

  get vertexCount(): number {
    return this._vertexCount;
  }

  setGeometry(geometry: EnvelopeGeometry | null): void {
    if (!geometry || geometry.indices.length === 0) {
      this._vertexCount = 0;
      this._scalars = null;
      this._mesh.setEnabled(false);
      return;
    }

    const vertexData = new VertexData();
    vertexData.positions = geometry.positions;
    vertexData.normals = geometry.normals;
    vertexData.indices = geometry.indices as unknown as Int32Array;
    vertexData.applyToMesh(this._mesh, true);

    this._vertexCount = geometry.positions.length / 3;
    this._scalars = geometry.scalars;
    this._mesh.setEnabled(true);
  }

  setStyle(style: EnvelopeStyle): void {
    this._material.alpha = style.alpha;
    this._material.transparencyMode =
      style.alpha < 0.999 ? PBRMaterial.PBRMATERIAL_ALPHABLEND : PBRMaterial.PBRMATERIAL_OPAQUE;
    this._material.needDepthPrePass = style.alpha < 0.999;
    this._material.metallic = style.metallic;
    this._material.roughness = style.roughness;
    this._glow.glowStrength = style.glow;

    if (this._vertexCount === 0) return;

    const colors = new Float32Array(this._vertexCount * 4);
    const span = style.scaleMax - style.scaleMin;
    const lut = buildLut(style.colormap);

    for (let index = 0; index < this._vertexCount; index += 1) {
      let r = style.flatColor.r;
      let g = style.flatColor.g;
      let b = style.flatColor.b;
      if (this._scalars) {
        const t = span > 1e-9 ? (this._scalars[index] - style.scaleMin) / span : 0;
        const slot = Math.min(255, Math.max(0, Math.round(t * 255))) * 3;
        r = lut[slot];
        g = lut[slot + 1];
        b = lut[slot + 2];
      }
      const base = index * 4;
      colors[base] = r;
      colors[base + 1] = g;
      colors[base + 2] = b;
      colors[base + 3] = 1;
    }
    this._mesh.setVerticesData(VertexBuffer.ColorKind, colors, true);
  }

  setVisible(visible: boolean): void {
    this._mesh.setEnabled(visible && this._vertexCount > 0);
  }

  dispose(): void {
    this._mesh.dispose();
    this._material.dispose();
  }
}

const LUT_CACHE = new Map<string, Float32Array>();

function buildLut(colormap: ColormapId): Float32Array {
  const cached = LUT_CACHE.get(colormap);
  if (cached) return cached;
  const lut = new Float32Array(256 * 3);
  for (let index = 0; index < 256; index += 1) {
    const [r, g, b] = sampleColormap(colormap, index / 255);
    lut[index * 3] = toLinear(r / 255);
    lut[index * 3 + 1] = toLinear(g / 255);
    lut[index * 3 + 2] = toLinear(b / 255);
  }
  LUT_CACHE.set(colormap, lut);
  return lut;
}

function toLinear(channel: number): number {
  return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
}
