/**
 * The isosurface mesh: defect envelopes and moving solidification fronts.
 *
 * Both come from the same place conceptually -- a level set of a field -- and
 * so share one mesh and one material here. The only difference is who
 * computed the triangles: the server for a static threshold, the contouring
 * worker for a moving front.
 *
 * Colour is baked into a vertex colour buffer rather than sampled from the 3D
 * texture the painted surface uses. An envelope is a *level set*: almost every
 * vertex sits at the same field value by construction, so per-fragment
 * sampling would spend the whole budget re-deriving a number the vertex
 * already carries. Baking also makes the "both" style honest -- the envelope
 * is coloured by the value it actually encloses, sampled server-side, while
 * the surface around it is coloured per fragment.
 */
import { Color3, Mesh, PBRMaterial, VertexBuffer, VertexData } from '@babylonjs/core';
import type { Scene } from '@babylonjs/core';

import { sampleColormap } from '../lib/colormaps';
import type { ColormapId } from '../lib/viewState';

export type EnvelopeGeometry = {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  /** Per-vertex field value, or null to use a single flat colour. */
  scalars: Float32Array | null;
};

export type EnvelopeStyle = {
  colormap: ColormapId;
  /** Colour scale in field units. */
  scaleMin: number;
  scaleMax: number;
  /** Used when there are no per-vertex scalars. */
  flatColor: Color3;
  alpha: number;
  metallic: number;
  roughness: number;
};

/**
 * A level-set surface in the scene.
 *
 * Geometry and colour are updated independently: changing the colormap or the
 * scale rewrites a colour buffer and leaves the triangles alone, which is what
 * makes dragging the scale bounds feel instant even on a large envelope.
 */
export class EnvelopeMesh {
  private readonly _mesh: Mesh;
  private readonly _material: PBRMaterial;
  private _scalars: Float32Array | null = null;
  private _vertexCount = 0;

  constructor(scene: Scene, name: string) {
    this._mesh = new Mesh(name, scene);
    this._mesh.isPickable = false;
    this._mesh.setEnabled(false);
    // Level sets are closed surfaces, but a front clipped by a section plane
    // or by the lattice edge is not, and a one-sided surface there shows as a
    // hole straight through the liquid.
    this._material = new PBRMaterial(`${name}-mat`, scene);
    this._material.backFaceCulling = false;
    this._material.twoSidedLighting = true;
    // PBR picks up VERTEXCOLOR from the mesh, not from a material flag, and
    // multiplies the albedo by it -- so the albedo has to be white for the
    // colour buffer to survive unaltered.
    this._material.albedoColor = Color3.White();
    this._mesh.useVertexColors = true;
    this._mesh.material = this._material;
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
    // Babylon's index buffer takes a plain number array or an Int32Array; a
    // Uint32Array is accepted and used directly when 32-bit indices are
    // supported, which they are on every WebGL 2 context.
    vertexData.indices = geometry.indices as unknown as Int32Array;
    vertexData.applyToMesh(this._mesh, true);

    this._vertexCount = geometry.positions.length / 3;
    this._scalars = geometry.scalars;
    this._mesh.setEnabled(true);
  }

  /** Rewrite the colour buffer without touching the triangles. */
  setStyle(style: EnvelopeStyle): void {
    this._material.alpha = style.alpha;
    this._material.transparencyMode =
      style.alpha < 0.999 ? PBRMaterial.PBRMATERIAL_ALPHABLEND : PBRMaterial.PBRMATERIAL_OPAQUE;
    this._material.needDepthPrePass = style.alpha < 0.999;
    this._material.metallic = style.metallic;
    this._material.roughness = style.roughness;

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

/**
 * A 256-entry linear-space ramp.
 *
 * `sampleColormap` returns 0..255 sRGB-ish values; Babylon's vertex colours
 * are linear, and handing it sRGB numbers directly washes the middle of every
 * ramp out. The conversion here is why a painted surface and an envelope of
 * the same value read as the same colour.
 */
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
