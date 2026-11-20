import { StandardMaterial } from '@babylonjs/core/Materials/standardMaterial';
import { DynamicTexture } from '@babylonjs/core/Materials/Textures/dynamicTexture';
import { Color3 } from '@babylonjs/core/Maths/math.color';
import { Mesh } from '@babylonjs/core/Meshes/mesh';
import { MeshBuilder } from '@babylonjs/core/Meshes/meshBuilder';
import type { Scene } from '@babylonjs/core/scene';

export function createPourMarker(scene: Scene, size: number): Mesh {
  const diameter = Math.max(size * 0.035, 0.15);
  const marker = MeshBuilder.CreateSphere('pour-marker', { diameter, segments: 16 }, scene);
  marker.isPickable = false;
  marker.renderingGroupId = 2;

  const material = new StandardMaterial('pour-marker-mat', scene);
  material.diffuseColor = new Color3(1, 0, 1);
  material.emissiveColor = new Color3(0.9, 0.05, 0.9);
  material.specularColor = Color3.Black();
  material.disableLighting = true;
  marker.material = material;

  const label = MeshBuilder.CreatePlane('pour-marker-label', { size: diameter * 2.6 }, scene);
  label.isPickable = false;
  label.renderingGroupId = 2;
  label.billboardMode = Mesh.BILLBOARDMODE_ALL;
  label.parent = marker;
  label.position.z = diameter * 1.4;

  const texture = new DynamicTexture('pour-marker-label-texture', { width: 256, height: 128 }, scene, false);
  texture.hasAlpha = true;
  texture.drawText('POUR', null, 84, 'bold 44px Helvetica, Arial, sans-serif', '#ff21ff', 'transparent', true, true);

  const labelMaterial = new StandardMaterial('pour-marker-label-mat', scene);
  labelMaterial.diffuseTexture = texture;
  labelMaterial.opacityTexture = texture;
  labelMaterial.emissiveColor = Color3.White();
  labelMaterial.disableLighting = true;
  labelMaterial.backFaceCulling = false;
  label.material = labelMaterial;

  return marker;
}
