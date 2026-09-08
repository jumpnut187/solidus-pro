/**
 * Paints a result field onto a lit PBR surface.
 *
 * The result is a property of the *metal*, not of the mesh: a fragment's color
 * comes from where it sits in the casting, not from a UV map. So instead of
 * baking colors into vertices -- which quantizes the field to the tessellation
 * of the STL and has to be rebuilt on every mode, scale or time change -- the
 * field lives in a 3D texture and each fragment samples it at its own world
 * position. Changing mode is a texture swap; scrubbing time is a uniform.
 *
 * It is a `MaterialPluginBase` rather than a `ShaderMaterial` for one reason:
 * the painted surface still has to be *lit*. A `ShaderMaterial` would mean
 * reimplementing image-based lighting, tone mapping and alpha blending to get
 * back what `PBRMaterial` already does, and the casting would read as a flat
 * cutout against an environment-lit scene. Injecting into the PBR fragment
 * shader keeps every one of those and replaces only the albedo.
 *
 * Injection point: `CUSTOM_FRAGMENT_UPDATE_ALPHA`. Despite the name, that
 * marker sits immediately after `surfaceAlbedo` and `alpha` are declared and
 * before any lighting runs, which is exactly where the albedo has to change.
 * (`CUSTOM_FRAGMENT_UPDATE_ALBEDO` does not exist in the PBR shader; it is a
 * standard-material marker.)
 */
import {
  Constants,
  MaterialPluginBase,
  RawTexture,
  RawTexture3D,
  Texture,
  Vector3
} from '@babylonjs/core';
import type { AbstractEngine, Material, Scene, SubMesh, UniformBuffer } from '@babylonjs/core';

/** Mask byte for cast metal: the engine's CAST code (0) plus one. */
export const CAST_MASK_CODE = 1;

/** Sweep cut-off standing for "not sweeping"; see `FieldPaintState.sweepTime`. */
const NO_SWEEP = 3.0e30;

const FIELD_SAMPLERS = ['fieldSamplerA', 'fieldSamplerB', 'fieldMaskSampler', 'fieldLutSampler'];

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
  'fieldSweepTime',
  'fieldLiquidColor',
  'fieldLiquidGlow',
  'fieldLiquidBody',
  'fieldSolidAlpha',
  'fieldSkinDepth'
];

export type FieldPaintState = {
  /** World position of the center of cell (0, 0, 0). */
  origin: Vector3;
  /** Uniform voxel edge length, in inches. */
  spacing: number;
  /** Lattice dimensions. */
  dims: [number, number, number];
  /** Range the stored bytes decode to: `value = min + byte / 255 * (max - min)`. */
  quantMin: number;
  quantMax: number;
  /** Legend range the colormap spans, in field units. */
  scaleMin: number;
  scaleMax: number;
  /** Lerp weight between the two bound frame textures, 0..1. */
  frameBlend: number;
  /** 0 hides the field entirely and leaves the part's own color. */
  strength: number;
  /** Restrict painting to cast bodies. */
  castOnly: boolean;
  /**
   * Discard fragments that fall outside the metal instead of leaving them
   * the part's own colour.
   *
   * This is what turns a plain quad into a section cap: the cut face is not a
   * shape anyone modelled, it is wherever the plane happens to intersect the
   * metal, and the occupancy grid is the only thing that knows where that is.
   */
  discardOutside: boolean;
  /**
   * Also discard fragments on metal the current view does not paint.
   *
   * `discardOutside` alone keeps a section cap honest: with `castOnly` the
   * riser still has metal in the cut, so the cap shows it in the part's own
   * colour rather than punching a hole. The contoured front is the opposite
   * case -- it is a surface derived from a field that was masked to cast
   * bodies, so any of it that lands on a riser is extrapolation and must go.
   */
  discardUnpaintable: boolean;
  /**
   * Sweep cut-off in field units, or null for no sweep.
   *
   * Set only for the sweep modes, where the field's values are themselves
   * times: a cell whose value is at or below this has already happened and is
   * painted normally, and a cell above it has not happened yet and is drawn as
   * liquid metal. One uniform turns a static map of solidification time into a
   * playable freezing sequence, with no second grid and no fetch.
   */
  sweepTime: number | null;
  /** Colour for cells the sweep has not reached, in linear space. */
  liquidColor: [number, number, number];
  /** How much of that colour those cells emit, matching the front's glow. */
  liquidGlow: number;
  /**
   * Draw the still-liquid metal as a closed solid body.
   *
   * The interior front only exists where liquid meets *solid*. Where liquid
   * runs to the casting's own skin there is no front to draw, so a ghosted
   * shell left the pool open to the background there: dark cut-outs around the
   * riser base and at the end of the plate, and at t = 0 -- when the whole
   * casting is liquid and there is no front at all -- an empty glass box.
   *
   * With this on, the skin itself closes the body: opaque wherever the metal
   * under it is still liquid, ghosted at `solidAlpha` wherever it has frozen.
   * The liquid region is then watertight without a second surface anywhere
   * near the wall, which is what kept the old sentinel-wall version z-fighting.
   */
  liquidBody: boolean;
  /** Opacity of the skin over metal that has already solidified. */
  solidAlpha: number;
  /**
   * How far below the surface the skin asks "is this still liquid?", in world
   * units. Normally one voxel.
   *
   * The front is built only from cells made entirely of metal, so it can never
   * reach into the outermost cell layer. If the skin tested the field at the
   * surface itself, then wherever the frozen shell is thinner than one cell
   * the skin would already read solid -- and go ghost -- while the front still
   * could not be built there. Neither would cover it and the body would open
   * up, which is what left a jagged dark crescent around the riser fillet.
   * Testing one cell down hands that outermost layer to the skin, so the two
   * meet exactly.
   */
  skinDepth: number;
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
  sweepTime: null,
  liquidColor: [1, 0.6, 0.15],
  liquidGlow: 0,
  liquidBody: false,
  solidAlpha: 1,
  skinDepth: 0
});

/**
 * A 1x1x1 stand-in bound whenever a real grid is not loaded.
 *
 * The samplers are declared unconditionally so that loading results never
 * recompiles the shader -- a recompile mid-session shows up as a visible
 * hitch, and a sampler left unbound is undefined behaviour on some drivers.
 * `strength` is what actually turns painting on and off.
 */
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

/**
 * Clamp on all three axes.
 *
 * A casting's surface sits half a voxel outside the outermost voxel *center*,
 * so surface fragments legitimately sample just past the edge of the volume.
 * Clamping makes that read the boundary cell; wrapping would fold the far side
 * of the casting onto the near one.
 */
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
    // `externalUniforms` registers the names so the Effect resolves their
    // locations; the declarations themselves are emitted into
    // CUSTOM_FRAGMENT_DEFINITIONS below rather than through the `ubo` or
    // `fragment` entries. Those two land in ADDITIONAL_UBO_DECLARATION and
    // ADDITIONAL_FRAGMENT_DECLARATION respectively, and the PBR shader
    // includes exactly one of those blocks depending on whether the engine
    // has uniform buffers -- which Babylon switches off for a blocklist of
    // drivers, SwiftShader among them. Declaring into either block alone
    // compiles on some machines and fails on others.
    // CUSTOM_FRAGMENT_DEFINITIONS is present in both.
    return { externalUniforms: [...FIELD_UNIFORMS] };
  }

  getCustomCode(shaderType: string): { [pointName: string]: string } | null {
    if (shaderType !== 'fragment') return null;
    return {
      CUSTOM_FRAGMENT_DEFINITIONS: `
uniform highp sampler3D fieldSamplerA;
uniform highp sampler3D fieldSamplerB;
uniform highp sampler3D fieldMaskSampler;
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
// "No sweep" is carried as a cut-off above every possible field value rather
// than as a second enable uniform: the test below is value > fieldSweepTime,
// so a huge cut-off leaves every cell painted exactly as it was.
uniform float fieldSweepTime;
uniform vec3 fieldLiquidColor;
uniform float fieldLiquidGlow;
uniform float fieldLiquidBody;
uniform float fieldSolidAlpha;
uniform float fieldSkinDepth;
`,
      // Runs after `vec3 surfaceAlbedo` and `float alpha` exist and before
      // any lighting, so the painted color is shaded like a real surface
      // rather than pasted on top of one.
      CUSTOM_FRAGMENT_UPDATE_ALPHA: `
  // Declared outside the branch so the emissive injection further down the
  // shader can still see it; both markers sit in the same main().
  float castSimLiquid = 0.0;
  if (fieldStrength > 0.0 || fieldDiscardOutside > 0.5 || fieldDiscardUnpaintable > 0.5) {
    vec3 fieldUvw = vPositionW * fieldTexelStep + fieldTexelBias;
    float maskCode = texture(fieldMaskSampler, fieldUvw).r * 255.0;
    bool insideMetal = maskCode > 0.5;
    bool paintable = insideMetal && (fieldCastOnly < 0.5 || abs(maskCode - 1.0) < 0.5);
    // A cap only exists where the plane actually crosses metal, and a
    // contoured front only exists over metal the field was built from.
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
      // Above the cut-off the metal here has not frozen yet. For the
      // solidification sweep the cut-off is the scrubber's own time; for a
      // fraction-liquid or fill frame it is the same 0.5 the front is
      // contoured at, so skin and front agree on where the liquid ends.
      bool stillLiquid = value > fieldSweepTime;
#ifdef NORMAL
      if (fieldLiquidBody > 0.5 && fieldSkinDepth > 0.0) {
        // See FieldPaintState.skinDepth. Only the liquid *test* moves; the
        // colour still comes from the value at the surface.
        vec3 innerUvw =
          (vPositionW - normalize(vNormalW) * fieldSkinDepth) * fieldTexelStep + fieldTexelBias;
        if (texture(fieldMaskSampler, innerUvw).r * 255.0 > 0.5) {
          float innerNormalized = mix(
            texture(fieldSamplerA, innerUvw).r,
            texture(fieldSamplerB, innerUvw).r,
            fieldBlend
          );
          float innerValue = fieldQuant.x + innerNormalized * (fieldQuant.y - fieldQuant.x);
          stillLiquid = stillLiquid || innerValue > fieldSweepTime;
        }
      }
#endif

#ifdef DEPTHPREPASS
      // Only the opaque part of the skin may claim the depth buffer. The
      // ghost over frozen metal must not, or it would hide the very front it
      // is there to show; the liquid part must, or the far wall of the pool
      // draws over the near one and the body reads as a bag of triangles.
      if (fieldLiquidBody > 0.5 && !stillLiquid) {
        discard;
      }
#endif

      vec3 painted;
      if (stillLiquid) {
        // Drawn as metal rather than as a value: one colour for the whole
        // molten region, whatever colormap the rest of the casting is using.
        painted = fieldLiquidColor;
        castSimLiquid = fieldStrength;
      } else {
        float span = fieldScale.y - fieldScale.x;
        float t = span > 1e-9 ? clamp((value - fieldScale.x) / span, 0.0, 1.0) : 0.0;
        painted = texture(fieldLutSampler, vec2(t, 0.5)).rgb;
      }
      surfaceAlbedo = mix(surfaceAlbedo, painted, fieldStrength);
      if (fieldLiquidBody > 0.5) {
        alpha = stillLiquid ? 1.0 : fieldSolidAlpha;
      }
    }
  }
`,
      // Liquid metal emits; solid metal reflects. Without this the not-yet-
      // frozen region is the right colour but the wrong material, and the
      // casting reads as painted rather than as hot. Matches the glow the
      // contoured front carries, so the two agree where they meet.
      CUSTOM_FRAGMENT_BEFORE_FINALCOLORCOMPOSITION: `
  finalEmissive += fieldLiquidColor * (fieldLiquidGlow * castSimLiquid);
`
    };
  }

  /**
   * Write the uniforms straight onto the effect.
   *
   * Not through the `UniformBuffer` argument: these uniforms are declared
   * outside the material's uniform block (see `getUniforms`), so a buffer
   * write would land somewhere the shader never reads.
   */
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

    // World -> texture coordinate, folded into a multiply and an add so the
    // fragment shader does not repeat the division per pixel:
    //   uvw = ((p - origin) / spacing + 0.5) / dims
    // The +0.5 lands cell centers on texel centers; without it every sample
    // is half a voxel off and the field slides against the geometry.
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
    // NO_SWEEP is above any field value the engine produces, so the shader's
    // `value > fieldSweepTime` is false everywhere and the paint is unchanged.
    effect.setFloat('fieldSweepTime', state.sweepTime ?? NO_SWEEP);
    effect.setFloat3('fieldLiquidColor', state.liquidColor[0], state.liquidColor[1], state.liquidColor[2]);
    effect.setFloat('fieldLiquidGlow', state.liquidGlow);
    effect.setFloat('fieldLiquidBody', state.liquidBody ? 1 : 0);
    effect.setFloat('fieldSolidAlpha', state.solidAlpha);
    effect.setFloat('fieldSkinDepth', state.skinDepth);

    effect.setTexture('fieldSamplerA', this.fieldTextureA ?? this._placeholderVolume);
    effect.setTexture('fieldSamplerB', this.fieldTextureB ?? this.fieldTextureA ?? this._placeholderVolume);
    effect.setTexture('fieldMaskSampler', this.maskTexture ?? this._placeholderVolume);
    effect.setTexture('fieldLutSampler', this.lutTexture ?? this._placeholderLut);
  }
}
