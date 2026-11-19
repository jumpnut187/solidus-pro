/**
 * Camera presets, animated transitions, orthographic mode and framing.
 *
 * Two rules drive the design:
 *
 * 1. A preset is a *move*, not a jump. Snapping the camera between views
 *    loses the viewer's sense of which way round the casting is; a short
 *    eased transition on alpha, beta, radius and target keeps it.
 * 2. Framing happens only when the thing being framed changes. The previous
 *    renderer re-fit the camera whenever opacity or a section plane moved,
 *    so nudging a slider threw away whatever view the user had set up. Here
 *    `fit` is called explicitly, and the component decides when.
 */
import { Camera } from '@babylonjs/core/Cameras/camera';
import { PointerEventTypes } from '@babylonjs/core/Events/pointerEvents';
import { Vector3 } from '@babylonjs/core/Maths/math.vector';
import type { ArcRotateCamera } from '@babylonjs/core/Cameras/arcRotateCamera';
import type { Engine } from '@babylonjs/core/Engines/engine';
import type { Scene } from '@babylonjs/core/scene';

import type { CameraPreset } from '../lib/viewState';

export type Bounds = {
  min: Vector3;
  max: Vector3;
  center: Vector3;
  /** Longest edge of the bounding box, never below a floor value. */
  size: number;
};

type Pose = {
  alpha: number;
  beta: number;
  radius: number;
  target: Vector3;
};

const TRANSITION_MS = 380;

/** Straight down the axis, minus a hair, because beta exactly 0 is a gimbal singularity. */
const POLE = 0.02;

/**
 * Preset orientations in (alpha, beta).
 *
 * With `upVector` on +Z, beta is measured from +Z: 0 looks down from above,
 * PI looks up from below, PI/2 is horizontal.
 *
 * The views are the desktop viewer's (`ui/visualizer.py`, `set_camera_view`)
 * so the same casting reads the same way round in both: Front looks from +Y
 * toward -Y, which is the reverse of the common CAD convention, and matching
 * the desktop was the deliberate choice. Top looks down from +Z with +X to the
 * right and +Y up the screen, and Bottom looks up from -Z with +X still to the
 * right and -Y up. Near a pole alpha no longer turns the view but spins it, so
 * those two need alpha +PI/2; at -PI/2 both came out rotated 180 degrees from
 * the desktop's, with +X pointing left.
 */
const PRESETS: Record<Exclude<CameraPreset, 'fit'>, { alpha: number; beta: number }> = {
  top: { alpha: Math.PI / 2, beta: POLE },
  bottom: { alpha: Math.PI / 2, beta: Math.PI - POLE },
  front: { alpha: -Math.PI / 2, beta: Math.PI / 2 },
  back: { alpha: Math.PI / 2, beta: Math.PI / 2 },
  right: { alpha: 0, beta: Math.PI / 2 },
  left: { alpha: Math.PI, beta: Math.PI / 2 },
  iso: { alpha: -Math.PI / 4, beta: Math.PI / 3 }
};

export class CameraController {
  private readonly _camera: ArcRotateCamera;
  private readonly _engine: Engine;
  private readonly _scene: Scene;
  private _bounds: Bounds | null = null;
  private _ortho = false;
  private _transition: { from: Pose; to: Pose; startedAt: number } | null = null;

  constructor(camera: ArcRotateCamera, engine: Engine, scene: Scene) {
    this._camera = camera;
    this._engine = engine;
    this._scene = scene;
    this._installOrthoStandoff();
    this._camera.onViewMatrixChangedObservable.add(() => this._syncOrthoZoom());
    this._scene.onBeforeRenderObservable.add(() => {
      this._step();
      this._syncPanning();
    });
    // Before the camera's own wheel handler, which the pre-pointer observable
    // always runs ahead of.
    this._scene.onPrePointerObservable.add(() => this._refreshZoomHitPlane(), PointerEventTypes.POINTERWHEEL);
  }

  get camera(): ArcRotateCamera {
    return this._camera;
  }

  setBounds(bounds: Bounds | null): void {
    this._bounds = bounds;
    // The ortho standoff is measured from the bounds, and nothing about the
    // camera's own state changes to tell Babylon the view is stale.
    if (this._ortho) this._camera.getViewMatrix(true);
  }

  setOrtho(ortho: boolean): void {
    if (this._ortho === ortho) return;
    this._ortho = ortho;
    this._camera.mode = ortho ? Camera.ORTHOGRAPHIC_CAMERA : Camera.PERSPECTIVE_CAMERA;
    // Switching projection moves the physical eye (see `_orthoStandoff`)
    // without touching alpha, beta, radius or target -- the only things
    // Babylon checks before reusing its cached view -- so the view is rebuilt
    // here. That also re-sizes the ortho frustum, through the observer.
    this._camera.getViewMatrix(true);
  }

  /** Re-fit the ortho frustum to the canvas's aspect after it changes size. */
  refreshProjection(): void {
    if (!this._ortho) return;
    this._syncOrthoZoom();
  }

  goTo(preset: CameraPreset): void {
    if (preset === 'fit') {
      this.fit();
      return;
    }
    const orientation = PRESETS[preset];
    const bounds = this._bounds;
    this._animateTo({
      alpha: this._shortestAlpha(orientation.alpha),
      beta: orientation.beta,
      radius: this._camera.radius,
      target: bounds ? bounds.center.clone() : this._camera.target.clone()
    });
  }

  /**
   * Look along a direction: the camera on the `normal` side of `target`,
   * with `up` up the screen. This is SOLIDWORKS' "Normal To" for a sketch
   * plane. The radius is kept, so the zoom the user had is kept.
   *
   * The angles are found by placing the camera and letting Babylon derive
   * them, which accounts for the Z-up frame; near the poles alpha only spins
   * the view, so there it is chosen to put `up` up the screen.
   */
  lookAlong(normal: Vector3, up: Vector3, target: Vector3): void {
    const n = normal.normalizeToNew();
    const saved = {
      alpha: this._camera.alpha,
      beta: this._camera.beta,
      radius: this._camera.radius,
      target: this._camera.target.clone()
    };
    const radius = Math.max(this._camera.radius, this._minRadius());
    this._camera.setTarget(target, false, false, true);
    this._camera.setPosition(target.add(n.scale(radius)));
    let alpha = this._camera.alpha;
    let beta = Math.min(Math.max(this._camera.beta, POLE), Math.PI - POLE);
    if (Math.abs(n.z) > 0.999) {
      // Looking straight down (or up) the Z axis the screen's up is set by
      // alpha alone. At beta near 0 the camera sits a hair off the pole along
      // (cos a, -sin a) and the screen's up points the other way; near PI it
      // points the same way. (Top's preset, alpha = PI/2, puts +Y up and
      // Bottom's puts -Y up, as the presets above document.)
      alpha = n.z > 0 ? Math.atan2(up.y, -up.x) : Math.atan2(-up.y, up.x);
      beta = n.z > 0 ? POLE : Math.PI - POLE;
    }
    this._apply(saved);
    this._animateTo({ alpha: this._shortestAlpha(alpha), beta, radius, target: target.clone() });
  }

  /** Frame the whole assembly, keeping the current orientation. */
  fit(options: { immediate?: boolean } = {}): void {
    const bounds = this._bounds;
    if (!bounds) return;
    const pose: Pose = {
      alpha: this._camera.alpha,
      beta: this._camera.beta,
      radius: this._radiusFor(bounds),
      target: bounds.center.clone()
    };
    if (options.immediate) {
      this._apply(pose);
      this._transition = null;
      this.refreshProjection();
    } else {
      this._animateTo(pose);
    }
  }

  /** Orbit around a picked point without changing the viewing direction. */
  focusOn(point: Vector3): void {
    this._animateTo({
      alpha: this._camera.alpha,
      beta: this._camera.beta,
      // Pull in a little: a double-click on a feature means "look at this",
      // and staying at assembly distance answers a question nobody asked.
      radius: Math.max(this._camera.radius * 0.55, this._minRadius()),
      target: point.clone()
    });
  }

  private _minRadius(): number {
    return Math.max((this._bounds?.size ?? 1) * 0.12, 0.5);
  }

  private _radiusFor(bounds: Bounds): number {
    // 2.15 x the longest edge leaves the assembly comfortably inside a
    // 1366x768 viewport at the default 45 degree field of view, including
    // when the diagonal rather than an edge faces the camera.
    return Math.max(bounds.size * 2.15, 1);
  }

  /**
   * Pick the equivalent alpha nearest the current one.
   *
   * Alpha is unbounded -- orbiting a few times leaves it at 14 radians -- so
   * animating to a literal -PI/2 would unwind every one of those turns.
   */
  private _shortestAlpha(target: number): number {
    const current = this._camera.alpha;
    const twoPi = Math.PI * 2;
    let candidate = target + Math.round((current - target) / twoPi) * twoPi;
    if (Math.abs(candidate - current) > Math.PI) {
      candidate += candidate > current ? -twoPi : twoPi;
    }
    return candidate;
  }

  private _animateTo(to: Pose): void {
    this._transition = {
      from: {
        alpha: this._camera.alpha,
        beta: this._camera.beta,
        radius: this._camera.radius,
        target: this._camera.target.clone()
      },
      to,
      startedAt: performance.now()
    };
  }

  private _step(): void {
    const transition = this._transition;
    if (!transition) return;
    const elapsed = performance.now() - transition.startedAt;
    const t = Math.min(elapsed / TRANSITION_MS, 1);
    const eased = t * t * (3 - 2 * t);
    const { from, to } = transition;

    this._apply({
      alpha: from.alpha + (to.alpha - from.alpha) * eased,
      beta: from.beta + (to.beta - from.beta) * eased,
      radius: from.radius + (to.radius - from.radius) * eased,
      target: Vector3.Lerp(from.target, to.target, eased)
    });

    if (t >= 1) {
      this._transition = null;
      this.refreshProjection();
    }
  }

  private _apply(pose: Pose): void {
    // Target first, and without letting Babylon rebuild the angles. By
    // default `setTarget` re-derives alpha, beta and radius from the camera's
    // last computed position, which threw away the pose just written whenever
    // the target moved: once a pan or a zoom toward the cursor had shifted the
    // target off the assembly centre, a preset or a fit ran its whole
    // transition without turning the camera at all.
    this._camera.setTarget(pose.target, false, false, true);
    this._camera.alpha = pose.alpha;
    this._camera.beta = pose.beta;
    this._camera.radius = pose.radius;
  }

  /**
   * Make one pixel of pan drag move the model one pixel on screen.
   *
   * Babylon pans by `pixels / panningSensibility` world units, so a fixed
   * sensibility is only right at one zoom: the old constant moved the model
   * 1.2 pixels per pixel at the default framing and 4.8 at a quarter of that
   * distance. The sensibility is the screen's pixels per world unit at the
   * target's depth, which is the depth the orbit pivots on and the one plane
   * that pans exactly one-to-one in perspective. Recomputed every frame
   * because the zoom, the projection and the canvas size can all change it.
   *
   * Measured in CSS pixels, because that is what the pointer offsets are in:
   * the render size is divided by the hardware scaling level, so rendering at
   * the device pixel ratio does not make the pan run slower.
   */
  private _syncPanning(): void {
    const camera = this._camera;
    const cssWidth = this._engine.getRenderWidth(true) * this._engine.getHardwareScalingLevel() * camera.viewport.width;
    const cssHeight =
      this._engine.getRenderHeight(true) * this._engine.getHardwareScalingLevel() * camera.viewport.height;
    if (!(cssWidth > 0) || !(cssHeight > 0)) return;

    let pixelsPerUnit: number;
    if (camera.mode === Camera.ORTHOGRAPHIC_CAMERA && camera.orthoTop !== null && camera.orthoBottom !== null) {
      pixelsPerUnit = cssHeight / (camera.orthoTop - camera.orthoBottom);
    } else {
      const visible = 2 * camera.radius * Math.tan(camera.fov / 2);
      pixelsPerUnit =
        camera.fovMode === Camera.FOVMODE_HORIZONTAL_FIXED ? cssWidth / visible : cssHeight / visible;
    }
    // Zero would switch panning off altogether, and a degenerate frustum is
    // no reason to do that.
    if (Number.isFinite(pixelsPerUnit) && pixelsPerUnit > 0) camera.panningSensibility = pixelsPerUnit;
  }

  /**
   * Re-derive the plane that zoom-to-cursor zooms onto before each wheel event.
   *
   * Babylon picks the zoom point where the cursor's ray meets a plane through
   * the target, facing the camera, but it only refreshes that plane while the
   * camera is moving under inertia. A preset, a fit or a pan moves the camera
   * without inertia and leaves the plane where it was. Zoom at Top, switch to
   * Front, and the cursor's ray meets the old horizontal plane at the camera
   * itself: the next tick shrank the radius without moving the camera, which
   * left the view where it was and the orbit pivot somewhere new. Refreshing
   * it here costs one subtraction.
   *
   * `_updateHitPlane` is private to Babylon's mouse-wheel input, hence the
   * guarded call rather than a typed one.
   */
  private _refreshZoomHitPlane(): void {
    const wheel = this._camera.inputs.attached.mousewheel as unknown as { _updateHitPlane?: () => void } | undefined;
    if (typeof wheel?._updateHitPlane === 'function') wheel._updateHitPlane();
  }

  /**
   * In orthographic mode, keep the eye behind the assembly however far in
   * the user zooms.
   *
   * The radius stays the zoom control in both projections: the wheel, the
   * pinch, the keyboard, zoom-to-cursor and the radius limits all act on it
   * and need no ortho path of their own. But an orthographic eye sitting at
   * that radius walked into the part as the view zoomed in -- at radius 2.9
   * on the ring casting the near wall was behind the near plane and simply
   * gone. So in ortho the view is built from the standoff distance whenever
   * the radius is shorter, and the radius goes back to what the inputs set.
   * The eye's distance changes nothing an orthographic image shows; only the
   * frustum does, and `_syncOrthoZoom` takes that from the radius.
   *
   * Building the view from the pushed-back eye, rather than moving the near
   * plane behind a close one, is what keeps `camera.position` true. PBR takes
   * each fragment's view direction from it, and an eye inside the casting
   * lit everything behind it from the wrong side.
   *
   * `_getViewMatrix` is Babylon's internal hook for this, hence the override
   * on the instance rather than a subclass the scene setup would have to
   * know about.
   */
  private _installOrthoStandoff(): void {
    const camera = this._camera;
    const computeView = camera._getViewMatrix.bind(camera);
    camera._getViewMatrix = () => {
      const standoff = this._orthoStandoff();
      const radius = camera.radius;
      if (standoff === null || radius >= standoff) return computeView();
      camera.radius = standoff;
      try {
        return computeView();
      } finally {
        camera.radius = radius;
      }
    };
  }

  /**
   * The shortest eye distance from which nothing on the stage is behind the
   * near plane, or null when there is no need for one.
   *
   * The fit distance keeps the assembly and its shop floor in front of a
   * perspective eye, so it does for an orthographic one too. It is measured
   * from the assembly centre, and the target can be panned or zoomed toward
   * the cursor well away from it, so the target's own distance from the
   * centre is added on top.
   */
  private _orthoStandoff(): number | null {
    const bounds = this._bounds;
    if (!bounds || this._camera.mode !== Camera.ORTHOGRAPHIC_CAMERA) return null;
    const offset = Vector3.Distance(this._camera.target, bounds.center);
    return offset + this._radiusFor(bounds) + this._camera.minZ;
  }

  /**
   * Size the orthographic frustum from the radius.
   *
   * An orthographic camera ignores its distance, so the radius the inputs
   * zoom has to reach the frustum some other way. Its half-height is the
   * height a perspective camera at the same radius sees at the target:
   * switching projection keeps the scale of the plane the orbit pivots on,
   * `fit` frames the same way in both, and because the frustum follows the
   * radius alone, a preset move, a double-click focus or a resize no longer
   * snaps a zoomed-in ortho view back out to the whole assembly, as the old
   * size-based frustum did whenever it was re-derived.
   *
   * Zoom-to-cursor needs nothing extra either. It pans the target toward the
   * cursor by the same fraction it shrinks the radius, and a frustum that
   * shrinks by that fraction too leaves the point under the cursor in place.
   */
  private _syncOrthoZoom(): void {
    const camera = this._camera;
    if (camera.mode !== Camera.ORTHOGRAPHIC_CAMERA) return;
    const aspect = this._engine.getAspectRatio(camera);
    const extent = camera.radius * Math.tan(camera.fov / 2);
    const horizontal = camera.fovMode === Camera.FOVMODE_HORIZONTAL_FIXED;
    const halfHeight = horizontal ? extent / aspect : extent;
    const halfWidth = horizontal ? extent : extent * aspect;
    camera.orthoTop = halfHeight;
    camera.orthoBottom = -halfHeight;
    camera.orthoRight = halfWidth;
    camera.orthoLeft = -halfWidth;
  }
}

/** Union of several axis-aligned boxes, or null when there are none. */
export function unionBounds(
  boxes: Array<{ x: [number, number]; y: [number, number]; z: [number, number] }>
): Bounds | null {
  if (!boxes.length) return null;
  const min = new Vector3(
    Math.min(...boxes.map((b) => b.x[0])),
    Math.min(...boxes.map((b) => b.y[0])),
    Math.min(...boxes.map((b) => b.z[0]))
  );
  const max = new Vector3(
    Math.max(...boxes.map((b) => b.x[1])),
    Math.max(...boxes.map((b) => b.y[1])),
    Math.max(...boxes.map((b) => b.z[1]))
  );
  const span = max.subtract(min);
  return {
    min,
    max,
    center: min.add(max).scale(0.5),
    size: Math.max(span.x, span.y, span.z, 1)
  };
}
