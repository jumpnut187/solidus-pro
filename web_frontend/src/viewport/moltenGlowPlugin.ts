import { MaterialPluginBase } from '@babylonjs/core/Materials/materialPluginBase';
import type { AbstractEngine } from '@babylonjs/core/Engines/abstractEngine';
import type { Material } from '@babylonjs/core/Materials/material';
import type { UniformBuffer } from '@babylonjs/core/Materials/uniformBuffer';
import type { SubMesh } from '@babylonjs/core/Meshes/subMesh';
import type { Scene } from '@babylonjs/core/scene';

const GLOW_UNIFORMS = ['moltenGlowStrength'];

export class MoltenGlowPluginMaterial extends MaterialPluginBase {
  glowStrength = 0;

  constructor(material: Material) {
    super(material, 'CastSimMoltenGlow', 210, { CASTSIM_MOLTEN_GLOW: true });
    this._enable(true);
  }

  getClassName(): string {
    return 'MoltenGlowPluginMaterial';
  }

  getUniforms() {
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
