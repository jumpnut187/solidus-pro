/**
 * Engine, scene, lighting and the shop floor.
 *
 * Z-up, right-handed, to match the engine's own coordinates: the voxel
 * centers, the STL geometry and the pour location are all in inches with Z
 * vertical, and converting at the render boundary would mean converting back
 * for every pick and every probe.
 *
 * Lighting is image-based. A casting is a specular metal object, and what
 * makes one read as metal is the environment reflected in it -- three
 * directional lights give a plastic look no amount of roughness tuning fixes.
 * A single soft key light is kept on top of the environment only to hold the
 * silhouette apart from the background.
 */
import {
  ArcRotateCamera,
  Color3,
  Color4,
  CubeTexture,
  DefaultRenderingPipeline,
  DirectionalLight,
  Engine,
  ImageProcessingConfiguration,
  Mesh,
  MeshBuilder,
  Scene,
  Vector3
} from '@babylonjs/core';

export const ENVIRONMENT_URL = '/environments/studio.env';

export type ViewportScene = {
  engine: Engine;
  scene: Scene;
  camera: ArcRotateCamera;
  pipeline: DefaultRenderingPipeline;
  /** Rebuild the floor grid to suit a model of this size, centered here. */
  layoutFloor: (center: Vector3, extent: number, floorZ: number) => void;
  dispose: () => void;
};

const BACKGROUND = new Color4(0.055, 0.062, 0.074, 1);

export function createViewportScene(canvas: HTMLCanvasElement): ViewportScene {
  const engine = new Engine(canvas, true, {
    antialias: true,
    stencil: true,
    // The shell captures PNG screenshots straight off the canvas, which reads
    // the drawing buffer after the frame is presented.
    preserveDrawingBuffer: true,
    powerPreference: 'high-performance'
  });

  const scene = new Scene(engine);
  scene.clearColor = BACKGROUND;
  scene.useRightHandedSystem = true;
  scene.imageProcessingConfiguration.toneMappingEnabled = true;
  scene.imageProcessingConfiguration.toneMappingType = ImageProcessingConfiguration.TONEMAPPING_ACES;
  scene.imageProcessingConfiguration.exposure = 1.15;
  scene.imageProcessingConfiguration.contrast = 1.1;

  const environment = CubeTexture.CreateFromPrefilteredData(ENVIRONMENT_URL, scene);
  environment.name = 'studio-environment';
  // The environment is a light source, not scenery: it lights the casting but
  // the background stays the flat dark ground the UI is designed against.
  scene.environmentTexture = environment;
  scene.environmentIntensity = 1.0;

  const key = new DirectionalLight('key', new Vector3(-0.4, 0.45, -0.8), scene);
  key.intensity = 1.1;
  key.diffuse = new Color3(1, 0.98, 0.94);
  key.specular = new Color3(0.7, 0.7, 0.7);

  const camera = new ArcRotateCamera('camera', -Math.PI / 4, Math.PI / 3, 40, Vector3.Zero(), scene);
  camera.upVector = new Vector3(0, 0, 1);
  camera.lowerRadiusLimit = 0.05;
  camera.upperRadiusLimit = 8000;
  camera.wheelDeltaPercentage = 0.05;
  camera.panningSensibility = 120;
  camera.minZ = 0.01;
  camera.maxZ = 40000;
  camera.attachControl(canvas, true);

  const pipeline = new DefaultRenderingPipeline('viewport-pipeline', true, scene, [camera]);
  pipeline.fxaaEnabled = true;
  pipeline.samples = 4;
  pipeline.imageProcessingEnabled = true;

  let floor: Mesh | null = null;
  const layoutFloor = (center: Vector3, extent: number, floorZ: number) => {
    floor?.dispose();
    floor = buildFloorGrid(scene, center, extent, floorZ);
  };
  layoutFloor(Vector3.Zero(), 20, 0);

  return {
    engine,
    scene,
    camera,
    pipeline,
    layoutFloor,
    dispose: () => {
      floor?.dispose();
      pipeline.dispose();
      scene.dispose();
      engine.dispose();
    }
  };
}

/**
 * A grid of lines in the XY plane, fading out from the model.
 *
 * Drawn as a line system rather than a textured ground so that it needs no
 * material library, costs one draw call, and can carry a per-vertex alpha
 * ramp -- which is what stops the floor from reading as a hard-edged card
 * floating under the casting.
 */
function buildFloorGrid(scene: Scene, center: Vector3, extent: number, floorZ: number): Mesh {
  const spacing = niceStep(extent / 10);
  const halfSpan = Math.ceil((extent * 1.6) / spacing) * spacing;
  const cx = Math.round(center.x / spacing) * spacing;
  const cy = Math.round(center.y / spacing) * spacing;

  const lines: Vector3[][] = [];
  const colors: Color4[][] = [];
  const base = new Color3(0.32, 0.38, 0.45);

  for (let offset = -halfSpan; offset <= halfSpan + 1e-6; offset += spacing) {
    const fade = 1 - Math.abs(offset) / (halfSpan + spacing);
    const major = Math.abs(offset) < 1e-6;
    const alpha = Math.max(0, fade * fade) * (major ? 0.55 : 0.28);
    const color = new Color4(base.r, base.g, base.b, alpha);

    lines.push([
      new Vector3(cx + offset, cy - halfSpan, floorZ),
      new Vector3(cx + offset, cy + halfSpan, floorZ)
    ]);
    colors.push([color, color]);
    lines.push([
      new Vector3(cx - halfSpan, cy + offset, floorZ),
      new Vector3(cx + halfSpan, cy + offset, floorZ)
    ]);
    colors.push([color, color]);
  }

  const mesh = MeshBuilder.CreateLineSystem('shop-floor', { lines, colors, useVertexAlpha: true }, scene);
  mesh.isPickable = false;
  mesh.alwaysSelectAsActiveMesh = true;
  return mesh;
}

/** Round a spacing to 1, 2 or 5 times a power of ten, the way a ruler is marked. */
function niceStep(raw: number): number {
  if (!Number.isFinite(raw) || raw <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / magnitude;
  if (normalized <= 1.5) return magnitude;
  if (normalized <= 3.5) return 2 * magnitude;
  if (normalized <= 7.5) return 5 * magnitude;
  return 10 * magnitude;
}

/**
 * The world axes, drawn as three short lines at the origin.
 *
 * Separate from the corner gizmo: this one is in the scene and shows where
 * the model sits relative to the machine origin, which matters when a part
 * has been arranged away from it.
 */
export function createOriginAxes(scene: Scene, length: number): Mesh {
  const axes: Array<{ direction: Vector3; color: Color4 }> = [
    { direction: new Vector3(1, 0, 0), color: new Color4(0.95, 0.28, 0.24, 0.85) },
    { direction: new Vector3(0, 1, 0), color: new Color4(0.28, 0.85, 0.35, 0.85) },
    { direction: new Vector3(0, 0, 1), color: new Color4(0.3, 0.52, 1, 0.85) }
  ];
  const mesh = MeshBuilder.CreateLineSystem(
    'origin-axes',
    {
      lines: axes.map(({ direction }) => [Vector3.Zero(), direction.scale(length)]),
      colors: axes.map(({ color }) => [color, color]),
      useVertexAlpha: true
    },
    scene
  );
  mesh.isPickable = false;
  return mesh;
}
