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
import { Camera, Vector3 } from '@babylonjs/core';
import type { ArcRotateCamera, Engine, Scene } from '@babylonjs/core';

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
 */
const PRESETS: Record<Exclude<CameraPreset, 'fit'>, { alpha: number; beta: number }> = {
  top: { alpha: -Math.PI / 2, beta: POLE },
  bottom: { alpha: -Math.PI / 2, beta: Math.PI - POLE },
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
  /** Half-height of the ortho frustum at the radius it was set up for. */
  private _orthoBase: { extent: number; radius: number } | null = null;
  private _transition: { from: Pose; to: Pose; startedAt: number } | null = null;

  constructor(camera: ArcRotateCamera, engine: Engine, scene: Scene) {
    this._camera = camera;
    this._engine = engine;
    this._scene = scene;
    this._camera.onViewMatrixChangedObservable.add(() => this._syncOrthoZoom());
    this._scene.onBeforeRenderObservable.add(() => this._step());
  }

  get camera(): ArcRotateCamera {
    return this._camera;
  }

  setBounds(bounds: Bounds | null): void {
    this._bounds = bounds;
  }

  setOrtho(ortho: boolean): void {
    if (this._ortho === ortho) return;
    this._ortho = ortho;
    if (!ortho) {
      this._camera.mode = Camera.PERSPECTIVE_CAMERA;
      this._orthoBase = null;
      return;
    }
    this._camera.mode = Camera.ORTHOGRAPHIC_CAMERA;
    this._resetOrthoBase();
  }

  /** Re-derive the ortho frustum after the model or the canvas changes size. */
  refreshProjection(): void {
    if (!this._ortho) return;
    this._resetOrthoBase();
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
    this._camera.alpha = pose.alpha;
    this._camera.beta = pose.beta;
    this._camera.radius = pose.radius;
    this._camera.setTarget(pose.target);
  }

  private _resetOrthoBase(): void {
    const bounds = this._bounds;
    const extent = ((bounds?.size ?? 10) * 1.25) / 2;
    this._orthoBase = { extent, radius: Math.max(this._camera.radius, 1e-6) };
    this._syncOrthoZoom();
  }

  /**
   * Keep the wheel zooming in orthographic mode.
   *
   * An orthographic camera ignores its radius, so without this the wheel
   * changes a number nothing reads and the view appears frozen. Scaling the
   * frustum by the radius makes the same gesture do the same thing in both
   * projections.
   */
  private _syncOrthoZoom(): void {
    const base = this._orthoBase;
    if (!base || this._camera.mode !== Camera.ORTHOGRAPHIC_CAMERA) return;
    const extent = base.extent * (this._camera.radius / base.radius);
    const aspect = this._engine.getAspectRatio(this._camera);
    if (aspect > 1) {
      this._camera.orthoTop = extent;
      this._camera.orthoBottom = -extent;
      this._camera.orthoRight = extent * aspect;
      this._camera.orthoLeft = -extent * aspect;
    } else {
      this._camera.orthoRight = extent;
      this._camera.orthoLeft = -extent;
      this._camera.orthoTop = extent / aspect;
      this._camera.orthoBottom = -extent / aspect;
    }
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
