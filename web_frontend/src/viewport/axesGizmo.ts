/**
 * The corner orientation gizmo.
 *
 * Drawn on a small 2D canvas rather than as scene geometry: it has to stay
 * the same size regardless of zoom, sit on top of everything, and never be
 * pickable or clipped by a section plane. Projecting three unit vectors
 * through the camera's view matrix by hand is a dozen lines and avoids all
 * three problems.
 */
import { Vector3 } from '@babylonjs/core/Maths/math.vector';
import type { ArcRotateCamera } from '@babylonjs/core/Cameras/arcRotateCamera';

import { viewportDeviceRatio } from './sceneSetup';

/** The gizmo's size in CSS pixels when the stylesheet has not laid it out yet. */
const FALLBACK_CSS_SIZE = 96;

const AXES: Array<{ direction: Vector3; color: string; label: string }> = [
  { direction: new Vector3(1, 0, 0), color: '#ff5b52', label: 'X' },
  { direction: new Vector3(0, 1, 0), color: '#4ade5c', label: 'Y' },
  { direction: new Vector3(0, 0, 1), color: '#5b8cff', label: 'Z' }
];

export function drawAxesGizmo(canvas: HTMLCanvasElement, camera: ArcRotateCamera): void {
  const context = canvas.getContext('2d');
  if (!context) return;

  // Drawn in CSS pixels onto a backing store at the viewport's own density,
  // so the gizmo is as sharp as the casting beside it. A backing store the
  // size of the CSS box was stretched 2x on a Retina display and blurred.
  // Sized from the laid-out content box, which is what the backing store is
  // stretched over; the stylesheet's border sits outside it.
  const size = canvas.clientWidth || FALLBACK_CSS_SIZE;
  const ratio = viewportDeviceRatio();
  const backing = Math.round(size * ratio);
  if (canvas.width !== backing || canvas.height !== backing) {
    canvas.width = backing;
    canvas.height = backing;
  }
  context.setTransform(backing / size, 0, 0, backing / size, 0, 0);

  const center = size / 2;
  const radius = size * 0.32;
  context.clearRect(0, 0, size, size);

  const view = camera.getViewMatrix();
  const projected = AXES.map((axis) => {
    const inView = Vector3.TransformNormal(axis.direction, view);
    return { ...axis, x: inView.x, y: inView.y, depth: inView.z };
  });

  // Painter's algorithm: an axis pointing away from the viewer is drawn
  // first so the ones in front overlap it, which is the only depth cue a
  // flat gizmo has.
  projected.sort((a, b) => a.depth - b.depth);

  for (const axis of projected) {
    const endX = center + axis.x * radius;
    // Canvas Y grows downward; view space Y grows up.
    const endY = center - axis.y * radius;
    const behind = axis.depth < 0;

    context.globalAlpha = behind ? 0.45 : 1;
    context.strokeStyle = axis.color;
    context.lineWidth = 2;
    context.lineCap = 'round';
    context.beginPath();
    context.moveTo(center, center);
    context.lineTo(endX, endY);
    context.stroke();

    context.fillStyle = axis.color;
    context.beginPath();
    context.arc(endX, endY, 7, 0, Math.PI * 2);
    context.fill();

    context.globalAlpha = 1;
    context.fillStyle = behind ? 'rgba(10, 12, 16, 0.85)' : '#0a0c10';
    context.font = 'bold 9px ui-sans-serif, system-ui, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(axis.label, endX, endY + 0.5);
  }

  context.globalAlpha = 1;
}
