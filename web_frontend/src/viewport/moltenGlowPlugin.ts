/**
 * Makes the liquid front look like metal that is glowing rather than metal
 * that is being lit.
 *
 * The front already carries the molten ramp in its vertex colours, which PBR
 * multiplies into the albedo. Albedo alone is not enough: albedo is what a
 * surface *reflects*, so a pool of steel at pouring temperature painted purely
 * in albedo goes dark wherever the environment map is dark, and reads as
 * polished orange plastic. What it needs is emission -- light the surface
 * produces itself, independent of the scene.
 *
 * `emissiveColor` on the material would give that, but only as one flat colour
 * for the whole front, which is exactly the variation worth keeping: the body
 * of the pool should glow harder than the crust freezing at its rim. So this
 * plugin adds the vertex colour into `finalEmissive`, scaled by one uniform.
 * The same buffer therefore drives both the reflected and the emitted colour,
 * and they cannot drift apart.
 *
 * Injection point: `CUSTOM_FRAGMENT_BEFORE_FINALCOLORCOMPOSITION`. That marker
 * sits after `pbrBlockFinalUnlitComponents` has declared `finalEmissive` and
 * before `pbrBlockFinalColorComposition` consumes it, which is the one place
 * in the PBR fragment shader where the emissive term can still be changed.
 *
 * Guarded on `VERTEXCOLOR` because the same material is used for server-built
 * defect envelopes, which are drawn from a flat colour and have no colour
 * buffer bound. With `glowStrength` at 0 -- which is what every non-molten
 * caller leaves it at -- this compiles to a no-op and the envelope shades
 * exactly as it did before.
 */
import { MaterialPluginBase } from '@babylonjs/core/Materials/materialPluginBase';
import type { AbstractEngine } from '@babylonjs/core/Engines/abstractEngine';
import type { Material } from '@babylonjs/core/Materials/material';
import type { UniformBuffer } from '@babylonjs/core/Materials/uniformBuffer';
import type { SubMesh } from '@babylonjs/core/Meshes/subMesh';
import type { Scene } from '@babylonjs/core/scene';

const GLOW_UNIFORMS = ['moltenGlowStrength'];

export class MoltenGlowPluginMaterial extends MaterialPluginBase {
  /** 0 disables the effect entirely; around 0.6 reads as liquid metal. */
  glowStrength = 0;

  constructor(material: Material) {
    super(material, 'CastSimMoltenGlow', 210, { CASTSIM_MOLTEN_GLOW: true });
    this._enable(true);
  }

  getClassName(): string {
    return 'MoltenGlowPluginMaterial';
  }

  getUniforms() {
    // Declared in CUSTOM_FRAGMENT_DEFINITIONS rather than through `ubo` or
    // `fragment`, for the reason spelled out in `fieldPaintPlugin`: those two
    // land in blocks the PBR shader includes only one of, depending on whether
    // the driver is on Babylon's uniform-buffer blocklist.
    return { externalUniforms: [...GLOW_UNIFORMS] };
  }

  getCustomCode(shaderType: string): { [pointName: string]: string } | null {
    if (shaderType !== 'fragment') return null;
    return {
      CUSTOM_FRAGMENT_DEFINITIONS: `
uniform float moltenGlowStrength;
`,
      CUSTOM_FRAGMENT_BEFORE_FINALCOLORCOMPOSITION: `
#ifdef VERTEXCOLOR
  finalEmissive += vColor.rgb * moltenGlowStrength;
#endif
`
    };
  }

  bindForSubMesh(
    _uniformBuffer: UniformBuffer,
    _scene: Scene,
    _engine: AbstractEngine,
    subMesh: SubMesh
  ): void {
    subMesh.effect?.setFloat('moltenGlowStrength', this.glowStrength);
  }
}
