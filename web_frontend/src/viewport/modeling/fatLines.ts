/**
 * Screen-space lines a few pixels wide, for the sketch.
 *
 * WebGL draws `LINES` one device pixel wide whatever is asked for, which on a
 * 2x display is half a CSS pixel: right for the floor grid and the feature
 * edges, too faint for the profile being drawn, which is the thing the user
 * is looking at. Babylon's GreasedLine extrudes each segment into a
 * camera-facing ribbon of constant pixel width instead.
 */
import { GreasedLineMeshMaterialType } from '@babylonjs/core/Materials/GreasedLine/greasedLineMaterialInterfaces';
import type { GreasedLineSimpleMaterial } from '@babylonjs/core/Materials/GreasedLine/greasedLineSimpleMaterial';
import type { Color3 } from '@babylonjs/core/Maths/math.color';
import { Vector2, type Vector3 } from '@babylonjs/core/Maths/math.vector';
import { CreateGreasedLine } from '@babylonjs/core/Meshes/Builders/greasedLineBuilder';
import type { Mesh } from '@babylonjs/core/Meshes/mesh';
import type { Scene } from '@babylonjs/core/scene';

/**
 * The `width` to pass for a line this many CSS pixels wide.
 *
 * With size attenuation GreasedLine offsets each side by width/4 device
 * pixels (half the width, through a projection it then divides back out
 * against a resolution measured in full NDC units), so the drawn line is
 * width/2 device pixels: double it, and convert CSS to device pixels.
 */
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
      // In GreasedLine's terms "size attenuation" is what makes the width a
      // constant number of pixels on screen.
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

/** Keep a line's pixel width right after the canvas or the pixel ratio changes. */
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
