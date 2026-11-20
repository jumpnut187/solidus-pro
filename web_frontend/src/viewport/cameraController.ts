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
  size: number;
};

type Pose = {
  alpha: number;
  beta: number;
  radius: number;
  target: Vector3;
};

const TRANSITION_MS = 380;

const POLE = 0.02;

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
    this._scene.onPrePointerObservable.add(() => this._refreshZoomHitPlane(), PointerEventTypes.POINTERWHEEL);
  }

  get camera(): ArcRotateCamera {
    return this._camera;
  }

  setBounds(bounds: Bounds | null): void {
    this._bounds = bounds;
    if (this._ortho) this._camera.getViewMatrix(true);
  }

  setOrtho(ortho: boolean): void {
    if (this._ortho === ortho) return;
    this._ortho = ortho;
    this._camera.mode = ortho ? Camera.ORTHOGRAPHIC_CAMERA : Camera.PERSPECTIVE_CAMERA;
    this._camera.getViewMatrix(true);
  }

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
      alpha = n.z > 0 ? Math.atan2(up.y, -up.x) : Math.atan2(-up.y, up.x);
      beta = n.z > 0 ? POLE : Math.PI - POLE;
    }
    this._apply(saved);
    this._animateTo({ alpha: this._shortestAlpha(alpha), beta, radius, target: target.clone() });
  }

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

  focusOn(point: Vector3): void {
    this._animateTo({
      alpha: this._camera.alpha,
      beta: this._camera.beta,
      radius: Math.max(this._camera.radius * 0.55, this._minRadius()),
      target: point.clone()
    });
  }

  private _minRadius(): number {
    return Math.max((this._bounds?.size ?? 1) * 0.12, 0.5);
  }

  private _radiusFor(bounds: Bounds): number {
    return Math.max(bounds.size * 2.15, 1);
  }

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
    this._camera.setTarget(pose.target, false, false, true);
    this._camera.alpha = pose.alpha;
    this._camera.beta = pose.beta;
    this._camera.radius = pose.radius;
  }

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
    if (Number.isFinite(pixelsPerUnit) && pixelsPerUnit > 0) camera.panningSensibility = pixelsPerUnit;
  }

  private _refreshZoomHitPlane(): void {
    const wheel = this._camera.inputs.attached.mousewheel as unknown as { _updateHitPlane?: () => void } | undefined;
    if (typeof wheel?._updateHitPlane === 'function') wheel._updateHitPlane();
  }

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

  private _orthoStandoff(): number | null {
    const bounds = this._bounds;
    if (!bounds || this._camera.mode !== Camera.ORTHOGRAPHIC_CAMERA) return null;
    const offset = Vector3.Distance(this._camera.target, bounds.center);
    return offset + this._radiusFor(bounds) + this._camera.minZ;
  }

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
