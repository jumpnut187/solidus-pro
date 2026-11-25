import { GreasedLineMeshMaterialType } from '@babylonjs/core/Materials/GreasedLine/greasedLineMaterialInterfaces';
import type { GreasedLineSimpleMaterial } from '@babylonjs/core/Materials/GreasedLine/greasedLineSimpleMaterial';
import type { Color3 } from '@babylonjs/core/Maths/math.color';
import { Vector2, type Vector3 } from '@babylonjs/core/Maths/math.vector';
import { CreateGreasedLine } from '@babylonjs/core/Meshes/Builders/greasedLineBuilder';
import type { Mesh } from '@babylonjs/core/Meshes/mesh';
import type { Scene } from '@babylonjs/core/scene';

function devicePixels(scene: Scene, cssPixels: number): number {
  return (2 * cssPixels) / scene.getEngine().getHardwareScalingLevel();
}

export function createFatLines(
  name: string,
  lines: Vector3[][],
  color: Color3,
  cssWidth: number,
  scene: Scene
): Mesh | null {
  const points = lines.filter((line) => line.length >= 2).map((line) => line.flatMap((p) => [p.x, p.y, p.z]));
  if (!points.length) return null;
  const engine = scene.getEngine();
  const mesh = CreateGreasedLine(
    name,
    { points },
    {
      materialType: GreasedLineMeshMaterialType.MATERIAL_TYPE_SIMPLE,
      sizeAttenuation: true,
      width: devicePixels(scene, cssWidth),
      color,
      resolution: new Vector2(engine.getRenderWidth(), engine.getRenderHeight())
    },
    scene
  ) as Mesh;
  mesh.isPickable = false;
  return mesh;
}

export function syncFatLineResolution(mesh: Mesh | null, scene: Scene): void {
  const material = mesh?.material as GreasedLineSimpleMaterial | null | undefined;
  if (!material || !('resolution' in material)) return;
  const engine = scene.getEngine();
  const width = engine.getRenderWidth();
  const height = engine.getRenderHeight();
  if (material.resolution.x !== width || material.resolution.y !== height) {
    material.resolution = new Vector2(width, height);
  }
}
