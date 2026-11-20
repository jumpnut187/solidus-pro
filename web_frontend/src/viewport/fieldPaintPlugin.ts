import { Constants } from '@babylonjs/core/Engines/constants';
import { MaterialPluginBase } from '@babylonjs/core/Materials/materialPluginBase';
import { RawTexture } from '@babylonjs/core/Materials/Textures/rawTexture';
import { RawTexture3D } from '@babylonjs/core/Materials/Textures/rawTexture3D';
import { Texture } from '@babylonjs/core/Materials/Textures/texture';
import { Vector3 } from '@babylonjs/core/Maths/math.vector';
import type { AbstractEngine } from '@babylonjs/core/Engines/abstractEngine';
import type { Material } from '@babylonjs/core/Materials/material';
import type { UniformBuffer } from '@babylonjs/core/Materials/uniformBuffer';
import type { SubMesh } from '@babylonjs/core/Meshes/subMesh';
import type { Scene } from '@babylonjs/core/scene';

export const CAST_MASK_CODE = 1;

const NO_SWEEP = 3.0e30;

const FIELD_SAMPLERS = [
  'fieldSamplerA',
  'fieldSamplerB',
  'fieldMaskSampler',
  'fieldSkinSampler',
  'fieldLutSampler'
];

const FIELD_UNIFORMS = [
  'fieldTexelStep',
  'fieldTexelBias',
  'fieldQuant',
  'fieldScale',
  'fieldBlend',
  'fieldStrength',
  'fieldCastOnly',
  'fieldDiscardOutside',
  'fieldDiscardUnpaintable',
  'fieldSkinClip',
  'fieldSweepTime',
  'fieldLiquidBand',
  'fieldLiquidColor',
  'fieldLiquidGlow',
  'fieldLiquidBody',
  'fieldSolidAlpha',
  'fieldSkinDepth',
  'fieldGhostSolid'
];

export type FieldPaintState = {
  origin: Vector3;
  spacing: number;
  dims: [number, number, number];
  quantMin: number;
  quantMax: number;
  scaleMin: number;
  scaleMax: number;
  frameBlend: number;
  strength: number;
  castOnly: boolean;
  discardOutside: boolean;
  discardUnpaintable: boolean;
  skinClip: boolean;
  sweepTime: number | null;
  liquidBand: number;
  liquidColor: [number, number, number];
  liquidGlow: number;
  liquidBody: boolean;
  solidAlpha: number;
  skinDepth: number;
  ghostSolid: boolean;
};

export const defaultFieldPaintState = (): FieldPaintState => ({
  origin: Vector3.Zero(),
  spacing: 1,
  dims: [1, 1, 1],
  quantMin: 0,
  quantMax: 1,
  scaleMin: 0,
  scaleMax: 1,
  frameBlend: 0,
  strength: 0,
  castOnly: false,
  discardOutside: false,
  discardUnpaintable: false,
  skinClip: false,
  sweepTime: null,
  liquidBand: 0,
  liquidColor: [1, 0.6, 0.15],
  liquidGlow: 0,
  liquidBody: false,
  solidAlpha: 1,
  skinDepth: 0,
  ghostSolid: false
});

export function createPlaceholderVolume(scene: Scene): RawTexture3D {
  const texture = new RawTexture3D(
    new Uint8Array([0]),
    1,
    1,
    1,
    Constants.TEXTUREFORMAT_R,
    scene,
    false,
    false,
    Texture.NEAREST_SAMPLINGMODE,
    Constants.TEXTURETYPE_UNSIGNED_BYTE
  );
  texture.name = 'field-placeholder';
  clampVolume(texture);
  return texture;
}

export function createPlaceholderLut(scene: Scene): RawTexture {
  const texture = RawTexture.CreateRGBATexture(
    new Uint8Array([0, 0, 0, 255]),
    1,
    1,
    scene,
    false,
    false,
    Texture.NEAREST_SAMPLINGMODE
  );
  texture.name = 'field-lut-placeholder';
  texture.wrapU = Texture.CLAMP_ADDRESSMODE;
  texture.wrapV = Texture.CLAMP_ADDRESSMODE;
  return texture;
}

export function clampVolume(texture: RawTexture3D): void {
  texture.wrapU = Texture.CLAMP_ADDRESSMODE;
  texture.wrapV = Texture.CLAMP_ADDRESSMODE;
  texture.wrapR = Texture.CLAMP_ADDRESSMODE;
}

export class FieldPaintPluginMaterial extends MaterialPluginBase {
  state: FieldPaintState = defaultFieldPaintState();

  fieldTextureA: RawTexture3D | null = null;
  fieldTextureB: RawTexture3D | null = null;
  maskTexture: RawTexture3D | null = null;
  skinTexture: RawTexture3D | null = null;
  lutTexture: RawTexture | null = null;

  private readonly _placeholderVolume: RawTexture3D;
  private readonly _placeholderLut: RawTexture;
  private readonly _texelToWorld = new Vector3();

  constructor(material: Material, placeholderVolume: RawTexture3D, placeholderLut: RawTexture) {
    super(material, 'CastSimFieldPaint', 200, { CASTSIM_FIELD_PAINT: true });
    this._placeholderVolume = placeholderVolume;
    this._placeholderLut = placeholderLut;
    this._enable(true);
  }

  getClassName(): string {
    return 'FieldPaintPluginMaterial';
  }

  getSamplers(samplers: string[]): void {
    samplers.push(...FIELD_SAMPLERS);
  }

  getUniforms() {
    return { externalUniforms: [...FIELD_UNIFORMS] };
  }

  getCustomCode(shaderType: string): { [pointName: string]: string } | null {
    if (shaderType !== 'fragment') return null;
    return {
      CUSTOM_FRAGMENT_DEFINITIONS: `

uniform highp sampler3D fieldSamplerA;
uniform highp sampler3D fieldSamplerB;
uniform highp sampler3D fieldMaskSampler;
uniform highp sampler3D fieldSkinSampler;
uniform sampler2D fieldLutSampler;
uniform vec3 fieldTexelStep;
uniform vec3 fieldTexelBias;
uniform vec2 fieldQuant;
uniform vec2 fieldScale;
uniform float fieldBlend;
uniform float fieldStrength;
uniform float fieldCastOnly;
uniform float fieldDiscardOutside;
uniform float fieldDiscardUnpaintable;
uniform float fieldSkinClip;
uniform float fieldSweepTime;
uniform float fieldLiquidBand;
uniform vec3 fieldLiquidColor;
uniform float fieldLiquidGlow;
uniform float fieldLiquidBody;
uniform float fieldSolidAlpha;
uniform float fieldSkinDepth;
uniform float fieldGhostSolid;

`,
      CUSTOM_FRAGMENT_UPDATE_ALPHA: `

  float castSimLiquid = 0.0;
  if (
    fieldStrength > 0.0 ||
    fieldDiscardOutside > 0.5 ||
    fieldDiscardUnpaintable > 0.5 ||
    fieldSkinClip > 0.5
  ) {
    vec3 fieldUvw = vPositionW * fieldTexelStep + fieldTexelBias;
    if (fieldSkinClip > 0.5 && texture(fieldSkinSampler, fieldUvw).r > 0.5) {
      discard;
    }
    float maskCode = texture(fieldMaskSampler, fieldUvw).r * 255.0;
    bool insideMetal = maskCode > 0.5;
    bool paintable = insideMetal && (fieldCastOnly < 0.5 || abs(maskCode - 1.0) < 0.5);
    if (fieldDiscardOutside > 0.5 && !insideMetal) {
      discard;
    }
    if (fieldDiscardUnpaintable > 0.5 && !paintable) {
      discard;
    }
    if (paintable) {
      float normalized = mix(
        texture(fieldSamplerA, fieldUvw).r,
        texture(fieldSamplerB, fieldUvw).r,
        fieldBlend
      );
      float value = fieldQuant.x + normalized * (fieldQuant.y - fieldQuant.x);
      float liquidValue = value;
#ifdef NORMAL
      if (fieldLiquidBody > 0.5 && fieldSkinDepth > 0.0) {
        vec3 innerUvw =
          (vPositionW - normalize(vNormalW) * fieldSkinDepth) * fieldTexelStep + fieldTexelBias;
        if (texture(fieldMaskSampler, innerUvw).r * 255.0 > 0.5) {
          float innerNormalized = mix(
            texture(fieldSamplerA, innerUvw).r,
            texture(fieldSamplerB, innerUvw).r,
            fieldBlend
          );
          float innerValue = fieldQuant.x + innerNormalized * (fieldQuant.y - fieldQuant.x);
          liquidValue = max(liquidValue, innerValue);
        }
      }
#endif
      float liquid = fieldLiquidBand > 0.0
        ? smoothstep(fieldSweepTime - fieldLiquidBand, fieldSweepTime, liquidValue)
        : (liquidValue > fieldSweepTime ? 1.0 : 0.0);
#ifdef DEPTHPREPASS
      if (fieldLiquidBody > 0.5 && liquid < 0.5) {
        discard;
      }
#endif
      vec3 frozenAlbedo = surfaceAlbedo;
      if (fieldGhostSolid < 0.5) {
        float span = fieldScale.y - fieldScale.x;
        float t = span > 1e-9 ? clamp((value - fieldScale.x) / span, 0.0, 1.0) : 0.0;
        frozenAlbedo = mix(surfaceAlbedo, texture(fieldLutSampler, vec2(t, 0.5)).rgb, fieldStrength);
      }
      surfaceAlbedo = mix(frozenAlbedo, mix(surfaceAlbedo, fieldLiquidColor, fieldStrength), liquid);
      castSimLiquid = fieldStrength * liquid;
      if (fieldLiquidBody > 0.5) {
        alpha = mix(fieldSolidAlpha, 1.0, liquid);
      }
    }
  }

`,
      CUSTOM_FRAGMENT_BEFORE_FINALCOLORCOMPOSITION: `
  finalEmissive += fieldLiquidColor * (fieldLiquidGlow * castSimLiquid);
`
    };
  }

  bindForSubMesh(
    _uniformBuffer: UniformBuffer,
    _scene: Scene,
    _engine: AbstractEngine,
    subMesh: SubMesh
  ): void {
    const effect = subMesh.effect;
    if (!effect) return;

    const state = this.state;
    const [nx, ny, nz] = state.dims;
    const spacing = state.spacing > 1e-9 ? state.spacing : 1;

    this._texelToWorld.set(1 / (spacing * nx), 1 / (spacing * ny), 1 / (spacing * nz));
    effect.setFloat3('fieldTexelStep', this._texelToWorld.x, this._texelToWorld.y, this._texelToWorld.z);
    effect.setFloat3(
      'fieldTexelBias',
      0.5 / nx - state.origin.x * this._texelToWorld.x,
      0.5 / ny - state.origin.y * this._texelToWorld.y,
      0.5 / nz - state.origin.z * this._texelToWorld.z
    );
    effect.setFloat2('fieldQuant', state.quantMin, state.quantMax);
    effect.setFloat2('fieldScale', state.scaleMin, state.scaleMax);
    effect.setFloat('fieldBlend', state.frameBlend);
    effect.setFloat('fieldStrength', state.strength);
    effect.setFloat('fieldCastOnly', state.castOnly ? 1 : 0);
    effect.setFloat('fieldDiscardOutside', state.discardOutside ? 1 : 0);
    effect.setFloat('fieldDiscardUnpaintable', state.discardUnpaintable ? 1 : 0);
    effect.setFloat('fieldSkinClip', state.skinClip ? 1 : 0);
    effect.setFloat('fieldSweepTime', state.sweepTime ?? NO_SWEEP);
    effect.setFloat('fieldLiquidBand', state.liquidBand);
    effect.setFloat3('fieldLiquidColor', state.liquidColor[0], state.liquidColor[1], state.liquidColor[2]);
    effect.setFloat('fieldLiquidGlow', state.liquidGlow);
    effect.setFloat('fieldLiquidBody', state.liquidBody ? 1 : 0);
    effect.setFloat('fieldSolidAlpha', state.solidAlpha);
    effect.setFloat('fieldSkinDepth', state.skinDepth);
    effect.setFloat('fieldGhostSolid', state.ghostSolid ? 1 : 0);

    effect.setTexture('fieldSamplerA', this.fieldTextureA ?? this._placeholderVolume);
    effect.setTexture('fieldSamplerB', this.fieldTextureB ?? this.fieldTextureA ?? this._placeholderVolume);
    effect.setTexture('fieldMaskSampler', this.maskTexture ?? this._placeholderVolume);
    effect.setTexture('fieldSkinSampler', this.skinTexture ?? this._placeholderVolume);
    effect.setTexture('fieldLutSampler', this.lutTexture ?? this._placeholderLut);
  }
}
