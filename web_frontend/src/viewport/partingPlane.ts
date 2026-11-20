import { StandardMaterial } from '@babylonjs/core/Materials/standardMaterial';
import { DynamicTexture } from '@babylonjs/core/Materials/Textures/dynamicTexture';
import { Color3 } from '@babylonjs/core/Maths/math.color';
import { Vector3 } from '@babylonjs/core/Maths/math.vector';
import { LinesMesh } from '@babylonjs/core/Meshes/linesMesh';
import { Mesh } from '@babylonjs/core/Meshes/mesh';
import { MeshBuilder } from '@babylonjs/core/Meshes/meshBuilder';
import { VertexData } from '@babylonjs/core/Meshes/mesh.vertexData';
import type { Scene } from '@babylonjs/core/scene';

import type { Bounds } from './cameraController';

const SHEET_COLOR = new Color3(1.0, 0.62, 0.18);
const SHEET_ALPHA = 0.16;
const RIM_COLOR = new Color3(1.0, 0.7, 0.25);
const CUT_COLOR = new Color3(1.0, 0.45, 0.05);
const SHEET_MARGIN = 0.25;

const LABEL_WIDTH = 1024;
const LABEL_HEIGHT = 128;

type Label = { plane: Mesh; texture: DynamicTexture; material: StandardMaterial; text: string };

function createLabel(scene: Scene, name: string, aspect: number): Label {
  const plane = MeshBuilder.CreatePlane(name, { width: aspect, height: 1 }, scene);
  plane.isPickable = false;
  plane.billboardMode = Mesh.BILLBOARDMODE_ALL;
  plane.renderingGroupId = 2;
  const texture = new DynamicTexture(
    `${name}-texture`,
    { width: Math.round(LABEL_HEIGHT * aspect), height: LABEL_HEIGHT },
    scene,
    false
  );
  texture.hasAlpha = true;
  const material = new StandardMaterial(`${name}-mat`, scene);
  material.diffuseTexture = texture;
  material.opacityTexture = texture;
  material.emissiveColor = Color3.White();
  material.disableLighting = true;
  material.backFaceCulling = false;
  material.disableDepthWrite = true;
  plane.material = material;
  return { plane, texture, material, text: '' };
}

function drawLabel(label: Label, text: string, color: string): void {
  if (label.text === text) return;
  label.text = text;
  const size = label.texture.getSize();
  const context = label.texture.getContext();
  context.clearRect(0, 0, size.width, size.height);
  context.fillStyle = 'rgba(8, 11, 16, 0.8)';
  context.fillRect(4, 16, size.width - 8, size.height - 32);
  label.texture.update();
  label.texture.drawText(text, null, 86, 'bold 56px Helvetica, Arial, sans-serif', color, null, true, true);
}

function disposeLabel(label: Label): void {
  label.plane.dispose();
  label.texture.dispose();
  label.material.dispose();
}

export function planeCutSegments(mesh: Mesh, z: number): number[] {
  const positions = mesh.getVerticesData('position');
  const indices = mesh.getIndices();
  if (!positions || !indices) return [];
  const world = mesh.computeWorldMatrix(true).m;
  const count = positions.length / 3;
  const wp = new Float64Array(positions.length);
  for (let i = 0; i < count; i += 1) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const zz = positions[i * 3 + 2];
    wp[i * 3] = x * world[0] + y * world[4] + zz * world[8] + world[12];
    wp[i * 3 + 1] = x * world[1] + y * world[5] + zz * world[9] + world[13];
    wp[i * 3 + 2] = x * world[2] + y * world[6] + zz * world[10] + world[14];
  }
  const out: number[] = [];
  const crossing: number[] = [];
  for (let t = 0; t + 2 < indices.length; t += 3) {
    crossing.length = 0;
    for (let e = 0; e < 3; e += 1) {
      const a = indices[t + e];
      const b = indices[t + ((e + 1) % 3)];
      const za = wp[a * 3 + 2] - z;
      const zb = wp[b * 3 + 2] - z;
      if ((za > 0) === (zb > 0)) continue;
      const s = za / (za - zb);
      crossing.push(
        wp[a * 3] + (wp[b * 3] - wp[a * 3]) * s,
        wp[a * 3 + 1] + (wp[b * 3 + 1] - wp[a * 3 + 1]) * s,
        z
      );
    }
    if (crossing.length === 6) out.push(...crossing);
  }
  return out;
}

export type PartingPlaneState = { z: number; visible: boolean };

export class PartingPlane {
  private readonly _scene: Scene;
  private _sheet: Mesh | null = null;
  private _sheetMaterial: StandardMaterial | null = null;
  private _rim: LinesMesh | null = null;
  private _cut: LinesMesh | null = null;
  private _label: Label | null = null;
  private _cope: Label | null = null;
  private _drag: Label | null = null;
  private _anchors: Vector3[] = [];
  private _labelHeight = 1;
  private _anchorIndex = -1;
  private readonly _placeLabels = () => this._followCamera();

  constructor(scene: Scene) {
    this._scene = scene;
    scene.onBeforeRenderObservable.add(this._placeLabels);
  }

  private _followCamera(): void {
    const camera = this._scene.activeCamera;
    if (!camera || !this._label || !this._label.plane.isEnabled() || !this._anchors.length) return;
    const eye = camera.globalPosition;
    let best = 0;
    let bestDistance = Infinity;
    this._anchors.forEach((corner, index) => {
      const distance = Vector3.DistanceSquared(corner, eye);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = index;
      }
    });
    if (best === this._anchorIndex) return;
    this._anchorIndex = best;
    const corner = this._anchors[best];
    const h = this._labelHeight;
    this._label.plane.position.set(corner.x, corner.y, corner.z + h * 0.7);
    this._cope?.plane.position.set(corner.x, corner.y, corner.z + h * 1.7);
    this._drag?.plane.position.set(corner.x, corner.y, corner.z - h * 0.7);
  }

  update(state: PartingPlaneState | null, bounds: Bounds | null, castings: Mesh[]): void {
    if (!state || !state.visible || !bounds) {
      this._setEnabled(false);
      return;
    }
    const scene = this._scene;
    const z = state.z;
    const spanX = bounds.max.x - bounds.min.x;
    const spanY = bounds.max.y - bounds.min.y;
    const margin = bounds.size * SHEET_MARGIN;
    const width = Math.max(spanX + 2 * margin, bounds.size * 0.5);
    const depth = Math.max(spanY + 2 * margin, bounds.size * 0.5);
    const cx = (bounds.min.x + bounds.max.x) / 2;
    const cy = (bounds.min.y + bounds.max.y) / 2;

    if (!this._sheet) {
      this._sheet = MeshBuilder.CreatePlane('parting-plane-sheet', { size: 1, sideOrientation: Mesh.DOUBLESIDE }, scene);
      this._sheet.isPickable = false;
      const material = new StandardMaterial('parting-plane-mat', scene);
      material.diffuseColor = SHEET_COLOR;
      material.emissiveColor = SHEET_COLOR.scale(0.6);
      material.specularColor = Color3.Black();
      material.alpha = SHEET_ALPHA;
      material.backFaceCulling = false;
      material.disableDepthWrite = true;
      this._sheet.material = material;
      this._sheetMaterial = material;
    }
    this._sheet.scaling.set(width, depth, 1);
    this._sheet.position.set(cx, cy, z);

    this._rim?.dispose();
    const x0 = cx - width / 2;
    const x1 = cx + width / 2;
    const y0 = cy - depth / 2;
    const y1 = cy + depth / 2;
    this._rim = MeshBuilder.CreateLines(
      'parting-plane-rim',
      {
        points: [
          new Vector3(x0, y0, z),
          new Vector3(x1, y0, z),
          new Vector3(x1, y1, z),
          new Vector3(x0, y1, z),
          new Vector3(x0, y0, z)
        ]
      },
      scene
    );
    this._rim.color = RIM_COLOR;
    this._rim.isPickable = false;

    this._cut?.dispose();
    this._cut = null;
    const segments: number[] = [];
    for (const mesh of castings) {
      if (!mesh.isEnabled()) continue;
      const cut = planeCutSegments(mesh, z);
      for (let i = 0; i < cut.length; i += 1) segments.push(cut[i]);
    }
    if (segments.length) {
      const cutMesh = new LinesMesh('parting-plane-cut', scene);
      const lineIndices = new Uint32Array(segments.length / 3);
      for (let i = 0; i < lineIndices.length; i += 1) lineIndices[i] = i;
      const data = new VertexData();
      data.positions = segments;
      data.indices = lineIndices;
      data.applyToMesh(cutMesh);
      cutMesh.color = CUT_COLOR;
      cutMesh.isPickable = false;
      this._cut = cutMesh;
    }

    const labelHeight = Math.max(bounds.size * 0.05, 0.2);
    if (!this._label) this._label = createLabel(scene, 'parting-plane-label', LABEL_WIDTH / LABEL_HEIGHT);
    if (!this._cope) this._cope = createLabel(scene, 'parting-plane-cope', 3);
    if (!this._drag) this._drag = createLabel(scene, 'parting-plane-drag', 3);
    drawLabel(this._label, `Parting line  Z = ${z.toFixed(2)} in`, '#ffb547');
    drawLabel(this._cope, 'COPE', '#ffd08a');
    drawLabel(this._drag, 'DRAG', '#9fd3ff');
    this._label.plane.scaling.setAll(labelHeight);
    this._cope.plane.scaling.setAll(labelHeight * 0.8);
    this._drag.plane.scaling.setAll(labelHeight * 0.8);
    this._labelHeight = labelHeight;
    this._anchors = [
      new Vector3(bounds.min.x, bounds.min.y, z),
      new Vector3(bounds.max.x, bounds.min.y, z),
      new Vector3(bounds.max.x, bounds.max.y, z),
      new Vector3(bounds.min.x, bounds.max.y, z)
    ];
    this._anchorIndex = -1;

    this._setEnabled(true);
  }

  private _setEnabled(enabled: boolean): void {
    this._sheet?.setEnabled(enabled);
    this._rim?.setEnabled(enabled);
    this._cut?.setEnabled(enabled);
    this._label?.plane.setEnabled(enabled);
    this._cope?.plane.setEnabled(enabled);
    this._drag?.plane.setEnabled(enabled);
  }

  dispose(): void {
    this._scene.onBeforeRenderObservable.removeCallback(this._placeLabels);
    this._sheet?.dispose();
    this._sheetMaterial?.dispose();
    this._rim?.dispose();
    this._cut?.dispose();
    if (this._label) disposeLabel(this._label);
    if (this._cope) disposeLabel(this._cope);
    if (this._drag) disposeLabel(this._drag);
    this._sheet = null;
    this._sheetMaterial = null;
    this._rim = null;
    this._cut = null;
    this._label = null;
    this._cope = null;
    this._drag = null;
  }
}
