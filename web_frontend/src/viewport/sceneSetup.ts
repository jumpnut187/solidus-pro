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
// SSAO2 renders its depth and normals through the pre-pass renderer and calls
// `scene.enablePrePassRenderer()`, which only exists once this component is
// registered. The pipeline module registers the geometry-buffer component
// but not this one.
import '@babylonjs/core/Rendering/prePassRendererSceneComponent';

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

/** How far past the model the floor reaches, as a multiple of its longest edge. */
const FLOOR_EXTENT_FACTOR = 1.25;
const FLOOR_MAJOR_ALPHA = 0.3;
const FLOOR_MINOR_ALPHA = 0.14;

export function createViewportScene(canvas: HTMLCanvasElement): ViewportScene {
  const engine = new Engine(canvas, true, {
    antialias: true,
    stencil: true,
    // The shell captures PNG screenshots straight off the canvas, which reads
    // the drawing buffer after the frame is presented.
    preserveDrawingBuffer: true,
    powerPreference: 'high-performance'
  });
  const stopTrackingDeviceRatio = trackDeviceRatio(engine);

  const scene = new Scene(engine);
  scene.clearColor = BACKGROUND;
  scene.useRightHandedSystem = true;
  scene.imageProcessingConfiguration.toneMappingEnabled = true;
  scene.imageProcessingConfiguration.toneMappingType = ImageProcessingConfiguration.TONEMAPPING_ACES;
  scene.imageProcessingConfiguration.exposure = 1.15;
  scene.imageProcessingConfiguration.contrast = 1.1;

  // The environment is a light source, not scenery: it lights the casting but
  // the background stays the flat dark ground the UI is designed against.
  const environment = CubeTexture.CreateFromPrefilteredData(ENVIRONMENT_URL, scene);
  environment.name = 'studio-environment';
  scene.environmentTexture = environment;
  scene.environmentIntensity = 1.0;

  // Analytic lights as well as the environment, and not only for taste.
  // A metallic PBR surface has almost no diffuse term, so with the
  // environment missing it renders *black* -- and the environment can go
  // missing for entirely ordinary reasons: a slow connection, a 404 on the
  // asset, or the RGBD decode losing its scene to a fast unmount. A casting
  // that disappears while a 200 KB file loads is not an acceptable failure
  // mode for the one view the whole application exists to show, so the fill
  // below guarantees a lit object and the environment refines it.
  const fill = new HemisphericLight('fill', new Vector3(0.15, -0.3, 1), scene);
  fill.intensity = 0.65;
  fill.diffuse = new Color3(0.95, 0.96, 1);
  fill.groundColor = new Color3(0.28, 0.3, 0.34);

  const key = new DirectionalLight('key', new Vector3(-0.4, 0.45, -0.8), scene);
  key.intensity = 1.1;
  key.diffuse = new Color3(1, 0.98, 0.94);
  key.specular = new Color3(0.7, 0.7, 0.7);

  const camera = new ArcRotateCamera('camera', -Math.PI / 4, Math.PI / 3, 40, Vector3.Zero(), scene);
  camera.upVector = new Vector3(0, 0, 1);
  camera.lowerRadiusLimit = 0.05;
  camera.upperRadiusLimit = 8000;
  // Orbit and wheel zoom share this one inertia. At Babylon's default 0.9 an
  // orbit kept turning for another quarter of the drag after the mouse was
  // released (23 degrees after a 91 degree drag), so the view settled
  // somewhere the user had not put it. 0.7 keeps a short, smooth settle
  // without the overshoot.
  camera.inertia = 0.7;
  // Every input's step is summed over the inertia's decay, 1 / (1 - inertia)
  // times over, so dropping it from 0.9 to 0.7 would have cut every orbit,
  // wheel tick and arrow key to a third of its travel. Each is scaled back by
  // that factor so the controls cover the same ground as before and only the
  // coast after release is shorter.
  const inertiaCompensation = (1 - 0.9) / (1 - camera.inertia);
  camera.angularSensibilityX = 1000 * inertiaCompensation;
  camera.angularSensibilityY = 1000 * inertiaCompensation;
  camera.wheelDeltaPercentage = 0.05 / inertiaCompensation;
  const keyboard = camera.inputs.attached.keyboard as ArcRotateCameraKeyboardMoveInput | undefined;
  if (keyboard) {
    keyboard.angularSpeed = 0.01 / inertiaCompensation;
    keyboard.zoomingSensibility = 25 * inertiaCompensation;
    // Keyboard panning had the camera's 0.9 panning inertia behind it, ten
    // times over, and now has none.
    keyboard.panningSensibility = 50 / 10;
  }
  // The wheel zooms toward the point under the cursor rather than toward the
  // middle of the screen: zooming in on a riser should not slide it off to
  // the side. See `CameraController` for the hit plane this relies on.
  camera.zoomToMouseLocation = true;
  // Panning stops when the mouse stops. With inertia the model slid on after
  // release, so it never ended up under the cursor that dragged it. The
  // sensibility itself depends on the zoom and is kept current by
  // `CameraController`.
  camera.panningInertia = 0;
  camera.minZ = 0.01;
  camera.maxZ = 40000;
  // The camera owns the events it acts on. Attached with `noPreventDefault`
  // it zoomed the model *and* let the same wheel event scroll the page, and a
  // trackpad pinch -- which Chrome and Firefox deliver as a ctrl+wheel --
  // zoomed the whole application along with the casting. Claiming the events
  // also has the canvas take focus on pointerdown, which is what gives the
  // arrow keys to the camera rather than to whatever was focused before.
  camera.attachControl(canvas, false);

  // Safari reports a pinch as its own non-standard gesture events rather than
  // as a ctrl+wheel, and zooms the page on them unless they are refused. No
  // camera input listens for them, so refusing them only keeps the page still;
  // the wheel and a two-finger scroll zoom the model as before.
  const refuseGesture = (event: Event) => event.preventDefault();
  const gestureEvents = ['gesturestart', 'gesturechange', 'gestureend'] as const;
  for (const name of gestureEvents) canvas.addEventListener(name, refuseGesture, { passive: false });

  const pipeline = new DefaultRenderingPipeline('viewport-pipeline', true, scene, [camera]);
  pipeline.fxaaEnabled = true;
  pipeline.samples = 4;
  pipeline.imageProcessingEnabled = true;

  // Bloom is configured here but left switched off; the viewport turns it on
  // only while a molten front is on screen. The threshold is high on purpose:
  // it should catch the emitting front and nothing else, so a painted casting
  // and the floor grid are untouched no matter which colormap is chosen. The
  // weight is small for the same reason -- this is heat haze around a pool of
  // metal, not a lens flare.
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

/**
 * The most device pixels rendered per CSS pixel.
 *
 * At 1 a Retina display drew the casting at half its resolution and let the
 * browser stretch it, which is what made every edge soft. 2 is what the
 * displays this runs on actually have; a 3x phone panel would pay 2.25 times
 * the fill of 2x for a difference nobody sees on a casting at arm's length.
 */
const MAX_DEVICE_RATIO = 2;

/**
 * The pixel density the viewport renders at: the display's, capped. Exported
 * so the 2D overlays drawn beside the canvas stay as sharp as it.
 */
export function viewportDeviceRatio(): number {
  return Math.min(window.devicePixelRatio || 1, MAX_DEVICE_RATIO);
}

/**
 * Render at the display's pixel density, capped, and follow it when it changes.
 *
 * Done here rather than with the engine's own `adaptToDeviceRatio`, which
 * applies its cap only at construction: on a later change it multiplies the
 * old scaling level by the ratio of old to new density, so moving the window
 * from a 2x screen to a 3x one would render at 3x regardless. The density
 * changes when the window moves between monitors and when the page is
 * zoomed, and moving monitors does not resize the canvas, so the container's
 * resize observer never hears of it.
 *
 * Checked at the start of every frame rather than through a `resolution`
 * media query. The query is the usual way, but it matches one exact density
 * and has to be re-armed on every change, and Chrome's device emulation --
 * what the screenshot harness drives -- changes the density without firing
 * it at all. One property read per frame is cheaper than being wrong.
 *
 * Pointer coordinates are unaffected: Babylon keeps `scene.pointerX` in CSS
 * pixels and divides by the scaling level when it builds a picking ray.
 *
 * Returns the function that stops tracking.
 */
function trackDeviceRatio(engine: Engine): () => void {
  const apply = () => {
    const level = 1 / viewportDeviceRatio();
    // Resizes the drawing buffer, and every post-process with it.
    if (engine.getHardwareScalingLevel() !== level) engine.setHardwareScalingLevel(level);
  };
  apply();
  const observer = engine.onBeginFrameObservable.add(apply);
  return () => engine.onBeginFrameObservable.remove(observer);
}

/**
 * A grid of lines in the XY plane, fading out from the model.
 *
 * The floor is there to give the casting a ground to sit on and a sense of
 * scale; it is not the subject. The first version reached to the horizon at
 * a uniform weight and competed with the result for attention, so it is now
 * held to about two and a half times the model, faded quadratically toward
 * its edge, and drawn thin enough that the far lines dissolve rather than
 * stop.
 *
 * Drawn as a line system rather than a textured ground so that it needs no
 * material library, costs one draw call, and can carry a per-vertex alpha
 * ramp -- which is what stops it reading as a hard-edged card under the part.
 */
function buildFloorGrid(scene: Scene, center: Vector3, extent: number, floorZ: number): Mesh {
  const spacing = niceStep(extent / 8);
  const halfSpan = Math.ceil((extent * FLOOR_EXTENT_FACTOR) / spacing) * spacing;
  const cx = Math.round(center.x / spacing) * spacing;
  const cy = Math.round(center.y / spacing) * spacing;

  const lines: Vector3[][] = [];
  const colors: Color4[][] = [];
  const base = new Color3(0.36, 0.42, 0.5);

  // Each line is split at the centre so its alpha can fall off along its own
  // length as well as with its distance from the model. A single segment can
  // only fade as a whole, which leaves a bright cross through the origin.
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
 * Unit length, so the caller scales them to the model rather than guessing.
 * They exist to show where the machine origin sits relative to a part that
 * has been arranged away from it, which is a question about a few inches --
 * axes long enough to cross the whole floor answered it by shouting.
 */
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

/**
 * Screen-space ambient occlusion.
 *
 * The cheapest way to make a casting read as a solid object: it darkens the
 * crevice where a riser meets the plate, which is exactly where the eye looks
 * for the geometry. It is a full-resolution depth pass plus a blur on top of
 * the existing chain, so whether to run it is a measurement, not a taste --
 * see the caller.
 */
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
