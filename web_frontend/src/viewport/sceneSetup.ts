import { ArcRotateCamera } from '@babylonjs/core/Cameras/arcRotateCamera';
import { Engine } from '@babylonjs/core/Engines/engine';
import { DirectionalLight } from '@babylonjs/core/Lights/directionalLight';
import { HemisphericLight } from '@babylonjs/core/Lights/hemisphericLight';
import { ImageProcessingConfiguration } from '@babylonjs/core/Materials/imageProcessingConfiguration';
import { CubeTexture } from '@babylonjs/core/Materials/Textures/cubeTexture';
import { Color3, Color4 } from '@babylonjs/core/Maths/math.color';
import { Vector3 } from '@babylonjs/core/Maths/math.vector';
import { MeshBuilder } from '@babylonjs/core/Meshes/meshBuilder';
import { DefaultRenderingPipeline } from '@babylonjs/core/PostProcesses/RenderPipeline/Pipelines/defaultRenderingPipeline';
import { SSAO2RenderingPipeline } from '@babylonjs/core/PostProcesses/RenderPipeline/Pipelines/ssao2RenderingPipeline';
import { Scene } from '@babylonjs/core/scene';
import type { ArcRotateCameraKeyboardMoveInput } from '@babylonjs/core/Cameras/Inputs/arcRotateCameraKeyboardMoveInput';
import type { Mesh } from '@babylonjs/core/Meshes/mesh';
import '@babylonjs/core/Rendering/prePassRendererSceneComponent';

export const ENVIRONMENT_URL = '/environments/studio.env';

export type ViewportScene = {
  engine: Engine;
  scene: Scene;
  camera: ArcRotateCamera;
  pipeline: DefaultRenderingPipeline;
  layoutFloor: (center: Vector3, extent: number, floorZ: number) => void;
  dispose: () => void;
};

const BACKGROUND = new Color4(0.055, 0.062, 0.074, 1);

export const CAD_ENVIRONMENT_INTENSITY = 0.3;
const HEADLIGHT_ELEVATION = 0.35;
const HEADLIGHT_SIDE = 0.2;

const FLOOR_EXTENT_FACTOR = 1.25;
const FLOOR_MAJOR_ALPHA = 0.3;
const FLOOR_MINOR_ALPHA = 0.14;

export function createViewportScene(canvas: HTMLCanvasElement): ViewportScene {
  const engine = new Engine(canvas, true, {
    antialias: true,
    stencil: true,
    preserveDrawingBuffer: true,
    powerPreference: 'high-performance'
  });
  const stopTrackingDeviceRatio = trackDeviceRatio(engine);

  const scene = new Scene(engine);
  scene.clearColor = BACKGROUND;
  scene.useRightHandedSystem = true;
  scene.imageProcessingConfiguration.toneMappingEnabled = true;
  scene.imageProcessingConfiguration.toneMappingType = ImageProcessingConfiguration.TONEMAPPING_ACES;
  scene.imageProcessingConfiguration.exposure = 1.0;
  scene.imageProcessingConfiguration.contrast = 1.0;

  const environment = CubeTexture.CreateFromPrefilteredData(ENVIRONMENT_URL, scene);
  environment.name = 'studio-environment';
  scene.environmentTexture = environment;
  scene.environmentIntensity = CAD_ENVIRONMENT_INTENSITY;

  const fill = new HemisphericLight('fill', new Vector3(0.15, -0.3, 1), scene);
  fill.intensity = 0.55;
  fill.diffuse = new Color3(0.95, 0.96, 1);
  fill.groundColor = new Color3(0.36, 0.38, 0.42);
  fill.specular = Color3.Black();

  const key = new DirectionalLight('key', new Vector3(-0.4, 0.45, -0.8), scene);
  key.intensity = 1.35;
  key.diffuse = new Color3(1, 1, 1);
  key.specular = new Color3(0.18, 0.18, 0.18);

  const camera = new ArcRotateCamera('camera', -Math.PI / 4, Math.PI / 3, 40, Vector3.Zero(), scene);
  camera.upVector = new Vector3(0, 0, 1);
  camera.lowerRadiusLimit = 0.05;
  camera.upperRadiusLimit = 8000;
  camera.inertia = 0.7;
  const inertiaCompensation = (1 - 0.9) / (1 - camera.inertia);
  camera.angularSensibilityX = 1000 * inertiaCompensation;
  camera.angularSensibilityY = 1000 * inertiaCompensation;
  camera.wheelDeltaPercentage = 0.05 / inertiaCompensation;
  const keyboard = camera.inputs.attached.keyboard as ArcRotateCameraKeyboardMoveInput | undefined;
  if (keyboard) {
    keyboard.angularSpeed = 0.01 / inertiaCompensation;
    keyboard.zoomingSensibility = 25 * inertiaCompensation;
    keyboard.panningSensibility = 50 / 10;
  }
  camera.zoomToMouseLocation = true;
  camera.panningInertia = 0;
  camera.minZ = 0.01;
  camera.maxZ = 40000;
  camera.attachControl(canvas, false);

  const refuseGesture = (event: Event) => event.preventDefault();
  const gestureEvents = ['gesturestart', 'gesturechange', 'gestureend'] as const;
  for (const name of gestureEvents) canvas.addEventListener(name, refuseGesture, { passive: false });

  const aimHeadlight = () => {
    const forward = camera.getForwardRay(1).direction;
    const right = Vector3.Cross(forward, camera.upVector).normalize();
    const camUp = Vector3.Cross(right, forward).normalize();
    key.direction = forward
      .add(camUp.scale(-HEADLIGHT_ELEVATION))
      .add(right.scale(HEADLIGHT_SIDE))
      .normalize();
  };
  scene.onBeforeRenderObservable.add(aimHeadlight);

  const pipeline = new DefaultRenderingPipeline('viewport-pipeline', true, scene, [camera]);
  pipeline.fxaaEnabled = true;
  pipeline.samples = 4;
  pipeline.imageProcessingEnabled = true;

  pipeline.bloomEnabled = false;
  pipeline.bloomThreshold = 0.82;
  pipeline.bloomWeight = 0.35;
  pipeline.bloomKernel = 48;
  pipeline.bloomScale = 0.5;

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
      stopTrackingDeviceRatio();
      for (const name of gestureEvents) canvas.removeEventListener(name, refuseGesture);
      environment.dispose();
      floor?.dispose();
      pipeline.dispose();
      scene.dispose();
      engine.dispose();
    }
  };
}

const MAX_DEVICE_RATIO = 2;

export function viewportDeviceRatio(): number {
  return Math.min(window.devicePixelRatio || 1, MAX_DEVICE_RATIO);
}

function trackDeviceRatio(engine: Engine): () => void {
  const apply = () => {
    const level = 1 / viewportDeviceRatio();
    if (engine.getHardwareScalingLevel() !== level) engine.setHardwareScalingLevel(level);
  };
  apply();
  const observer = engine.onBeginFrameObservable.add(apply);
  return () => engine.onBeginFrameObservable.remove(observer);
}

function buildFloorGrid(scene: Scene, center: Vector3, extent: number, floorZ: number): Mesh {
  const spacing = niceStep(extent / 8);
  const halfSpan = Math.ceil((extent * FLOOR_EXTENT_FACTOR) / spacing) * spacing;
  const cx = Math.round(center.x / spacing) * spacing;
  const cy = Math.round(center.y / spacing) * spacing;

  const lines: Vector3[][] = [];
  const colors: Color4[][] = [];
  const base = new Color3(0.36, 0.42, 0.5);

  const segments = 12;
  for (let offset = -halfSpan; offset <= halfSpan + 1e-6; offset += spacing) {
    const major = Math.abs(offset) < 1e-6;
    for (let axis = 0; axis < 2; axis += 1) {
      const points: Vector3[] = [];
      const vertexColors: Color4[] = [];
      for (let step = 0; step <= segments; step += 1) {
        const along = -halfSpan + (2 * halfSpan * step) / segments;
        const radial = Math.hypot(along, offset) / halfSpan;
        const fade = Math.max(0, 1 - radial * radial);
        const alpha = fade * (major ? FLOOR_MAJOR_ALPHA : FLOOR_MINOR_ALPHA);
        points.push(
          axis === 0
            ? new Vector3(cx + offset, cy + along, floorZ)
            : new Vector3(cx + along, cy + offset, floorZ)
        );
        vertexColors.push(new Color4(base.r, base.g, base.b, alpha));
      }
      lines.push(points);
      colors.push(vertexColors);
    }
  }

  const mesh = MeshBuilder.CreateLineSystem('shop-floor', { lines, colors, useVertexAlpha: true }, scene);
  mesh.isPickable = false;
  mesh.alwaysSelectAsActiveMesh = true;
  return mesh;
}

function niceStep(raw: number): number {
  if (!Number.isFinite(raw) || raw <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / magnitude;
  if (normalized <= 1.5) return magnitude;
  if (normalized <= 3.5) return 2 * magnitude;
  if (normalized <= 7.5) return 5 * magnitude;
  return 10 * magnitude;
}

export function createOriginAxes(scene: Scene): Mesh {
  const axes: Array<{ direction: Vector3; color: Color4 }> = [
    { direction: new Vector3(1, 0, 0), color: new Color4(0.95, 0.28, 0.24, 0.7) },
    { direction: new Vector3(0, 1, 0), color: new Color4(0.28, 0.85, 0.35, 0.7) },
    { direction: new Vector3(0, 0, 1), color: new Color4(0.3, 0.52, 1, 0.7) }
  ];
  const mesh = MeshBuilder.CreateLineSystem(
    'origin-axes',
    {
      lines: axes.map(({ direction }) => [Vector3.Zero(), direction]),
      colors: axes.map(({ color }) => [color, color]),
      useVertexAlpha: true
    },
    scene
  );
  mesh.isPickable = false;
  return mesh;
}

export function enableAmbientOcclusion(
  scene: Scene,
  camera: ArcRotateCamera
): SSAO2RenderingPipeline | null {
  if (!SSAO2RenderingPipeline.IsSupported) return null;
  const pipeline = new SSAO2RenderingPipeline(
    'viewport-ssao',
    scene,
    { ssaoRatio: 0.75, blurRatio: 1 },
    [camera]
  );
  pipeline.samples = 12;
  pipeline.radius = 1.1;
  pipeline.totalStrength = 0.8;
  pipeline.expensiveBlur = false;
  pipeline.maxZ = 400;
  return pipeline;
}
