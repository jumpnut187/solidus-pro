import type { ArcRotateCamera } from '@babylonjs/core/Cameras/arcRotateCamera';
import type { Camera } from '@babylonjs/core/Cameras/camera';
import { Constants } from '@babylonjs/core/Engines/constants';
import { PointerEventTypes, type PointerInfo, type PointerInfoPre } from '@babylonjs/core/Events/pointerEvents';
import { Material } from '@babylonjs/core/Materials/material';
import { PBRMaterial } from '@babylonjs/core/Materials/PBR/pbrMaterial';
import { StandardMaterial } from '@babylonjs/core/Materials/standardMaterial';
import { Color3, Color4 } from '@babylonjs/core/Maths/math.color';
import { Matrix, Vector3 } from '@babylonjs/core/Maths/math.vector';
import type { Viewport } from '@babylonjs/core/Maths/math.viewport';
import type { LinesMesh } from '@babylonjs/core/Meshes/linesMesh';
import { Mesh } from '@babylonjs/core/Meshes/mesh';
import { VertexData } from '@babylonjs/core/Meshes/mesh.vertexData';
import { MeshBuilder } from '@babylonjs/core/Meshes/meshBuilder';
import type { Observer } from '@babylonjs/core/Misc/observable';
import type { Scene } from '@babylonjs/core/scene';

import type { MeasurePick } from '../../lib/modeling/measure';
import {
  arcFromCenter,
  arcThrough,
  connectedIds,
  COINCIDENT_TOL,
  distance2,
  entityEnds,
  entityKeyPoints,
  entityOutline,
  entityPath,
  moveEnd,
  newEntityId,
  planeFrame,
  planeLabel,
  planeThroughPoint,
  pointInContours,
  sketchDimensions,
  sketchProfiles,
  sketchToWorld,
  STANDARD_PLANES,
  translateEntity,
  withDimension,
  worldToSketch,
  type DimensionSpec,
  type FaceOutline,
  type ModelingEvent,
  type ModelingViewState,
  type PlaneFrame,
  type SketchEntity,
  type SketchPlane,
  type SketchTool,
  type Vec2,
  type Vec3
} from '../../lib/modeling/types';
import { buildCreasedGeometry } from '../creaseGeometry';
import { createFatLines, syncFatLineResolution } from './fatLines';
import { faceBoundaryLoops, type PlanarFace } from './planarFace';

export type SurfacePick = {
  point: Vector3;
  normal: Vector3;
  distance: number;
  partName: string | null;
  face: PlanarFace | null;
};

export type AssemblyBounds = { center: Vector3; size: number } | null;

const SNAP_RADIUS_PX = 10;
const SELECT_RADIUS_PX = 7;
const CLICK_DRAG_THRESHOLD_PX = 4;
const HV_INFERENCE_DEG = 4;
const GRID_STEPS = [1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 5, 10, 20, 50];
const GRID_TARGET_PX = 22;

const SKETCH_LINE = new Color3(0.36, 0.66, 1);
const SELECTED_LINE = new Color3(1, 0.6, 0.15);
const CONSTRUCTION_LINE = new Color3(0.6, 0.68, 0.78);
const RUBBER_LINE = new Color3(0.62, 0.84, 1);
const OPEN_END = new Color3(1, 0.25, 0.25);
const DIM_COLOR = new Color4(0.85, 0.88, 0.92, 0.75);
const GRID_MINOR = new Color4(0.5, 0.6, 0.75, 0.12);
const GRID_MAJOR = new Color4(0.55, 0.65, 0.8, 0.26);
const AXIS_U = new Color4(0.95, 0.35, 0.3, 0.9);
const AXIS_V = new Color4(0.35, 0.85, 0.4, 0.9);
const PLANE_COLOR = new Color3(0.32, 0.55, 0.95);
const REF_PLANE_COLOR = new Color3(0.55, 0.45, 0.95);
const PREVIEW_PLANE_COLOR = new Color3(1, 0.62, 0.2);
const CHOSEN_PLANE_COLOR = new Color3(0.3, 0.85, 0.55);
const PREVIEW_OK = new Color3(0.98, 0.78, 0.25);
const PREVIEW_BAD = new Color3(0.95, 0.3, 0.3);
const PREVIEW_CUT = new Color3(0.9, 0.22, 0.42);
const ARROW_LINE = new Color3(1, 0.78, 0.25);
const MEASURE_LINE = new Color3(1, 0.62, 0.15);
const MEASURE_HOVER = new Color3(0.45, 0.8, 1);
const REGION_SELECTED = new Color3(0.3, 0.62, 1);
const REGION_OPEN = new Color3(0.7, 0.76, 0.86);

type Draft =
  | { tool: 'rectangle' | 'centerRectangle' | 'circle' | 'polygon'; first: Vec2 }
  | { tool: 'slot'; first: Vec2; second: Vec2 | null }
  | { tool: 'line' | 'centerline'; start: Vec2; chainStart: Vec2 }
  | { tool: 'arc3'; first: Vec2; second: Vec2 | null }
  | { tool: 'centerArc'; center: Vec2; start: Vec2 | null; sweep: number; lastAngle: number };

type SnapKind = 'point' | 'grid' | 'free' | 'horizontal' | 'vertical';
type Snap = { point: Vec2; kind: SnapKind };

type Label = { element: HTMLElement; anchor: Vec3; spec?: DimensionSpec };

type PlaneQuad = { key: string; plane: SketchPlane; frame: PlaneFrame; center: Vec2; half: number; mesh: Mesh };

type Drag =
  | { kind: 'entities'; ids: Set<string>; start: Vec2; original: SketchEntity[]; moved: boolean }
  | { kind: 'end'; id: string; end: 0 | 1; original: SketchEntity[]; moved: boolean };

export class SketchOverlay {
  private readonly _scene: Scene;
  private readonly _camera: ArcRotateCamera;
  private readonly _canvas: HTMLCanvasElement;
  private readonly _layer: HTMLDivElement;
  private readonly _pickSurface: (x: number, y: number) => SurfacePick | null;
  private readonly _pickMeasure: (x: number, y: number) => MeasurePick | null;
  private readonly _emit: (event: ModelingEvent) => void;

  private _state: ModelingViewState | null = null;
  private _bounds: AssemblyBounds = null;
  private _frame: PlaneFrame | null = null;

  private _planeQuads: PlaneQuad[] = [];
  private _decor: Mesh[] = [];
  private _hoverPlane: string | null = null;
  private _faceHint: Mesh | null = null;
  private _hintSource: Float32Array | null = null;

  private _sketchPlane: Mesh | null = null;
  private _grid: LinesMesh | null = null;
  private _gridStep = 0;
  private _axes: LinesMesh | null = null;
  private _entities: Mesh[] = [];
  private _dimLines: LinesMesh | null = null;
  private _rubber: Mesh | null = null;
  private _snapMarker: Mesh | null = null;

  private _regionMeshes: Mesh[] = [];
  private _regionLines: LinesMesh | null = null;
  private _hoverRegion = -1;
  private _preview: Mesh | null = null;
  private _previewEdges: LinesMesh | null = null;
  private _arrow: Mesh | null = null;

  private _measureMeshes: Mesh[] = [];
  private _measureHover: Mesh[] = [];
  private _hoverMeasureKey = '';

  private readonly _planeMaterial: StandardMaterial;
  private readonly _hintMaterial: StandardMaterial;
  private readonly _measureFaceMaterial: StandardMaterial;
  private readonly _previewMaterial: PBRMaterial;
  private readonly _cutPreviewMaterial: PBRMaterial;

  private _labels: Label[] = [];
  private _dimLabels: Label[] = [];
  private _measureLabels: Label[] = [];
  private _editing: { spec: DimensionSpec; input: HTMLInputElement } | null = null;
  private readonly _readout: HTMLDivElement;

  private _draft: Draft | null = null;
  private _cursor: Snap | null = null;
  private _pointerDown: { x: number; y: number; button: number; orbit: boolean; startedDraft: boolean } | null = null;
  private _numeric: { box: HTMLDivElement; input: HTMLInputElement } | null = null;
  private _depthHandle: { origin: Vector3; dir: Vector3; base: Vector3; tip: Vector3; midplane: boolean } | null = null;
  private _depthDrag: { last: number } | null = null;
  private _box: { x0: number; y0: number; additive: boolean; element: HTMLDivElement | null } | null = null;
  private _depthLabel: Label | null = null;
  private _drag: Drag | null = null;
  private _dirtyRubber = false;
  private _dirtyLabels = true;
  private _dirtyHover = false;
  private readonly _pointerObserver: Observer<PointerInfo>;
  private readonly _prePointerObserver: Observer<PointerInfoPre>;
  private readonly _renderObserver: Observer<Scene>;
  private readonly _viewObserver: Observer<Camera>;
  private readonly _onKey: (event: KeyboardEvent) => void;

  constructor(options: {
    scene: Scene;
    camera: ArcRotateCamera;
    canvas: HTMLCanvasElement;
    layer: HTMLDivElement;
    pickSurface: (x: number, y: number) => SurfacePick | null;
    pickMeasure: (x: number, y: number) => MeasurePick | null;
    emit: (event: ModelingEvent) => void;
  }) {
    this._scene = options.scene;
    this._camera = options.camera;
    this._canvas = options.canvas;
    this._layer = options.layer;
    this._pickSurface = options.pickSurface;
    this._pickMeasure = options.pickMeasure;
    this._emit = options.emit;

    this._planeMaterial = flatMaterial('sketch-plane-mat', PLANE_COLOR, 0.14, this._scene);
    this._hintMaterial = flatMaterial('sketch-hint-mat', new Color3(0.25, 0.6, 1), 0.55, this._scene);
    this._measureFaceMaterial = flatMaterial('measure-face-mat', MEASURE_LINE, 0.45, this._scene);
    for (const material of [this._hintMaterial, this._measureFaceMaterial]) {
      material.zOffset = -2;
      material.zOffsetUnits = -8;
      material.disableDepthWrite = true;
    }

    this._previewMaterial = new PBRMaterial('sketch-preview-mat', this._scene);
    this._previewMaterial.albedoColor = PREVIEW_OK;
    this._previewMaterial.metallic = 0;
    this._previewMaterial.roughness = 0.6;
    this._previewMaterial.alpha = 0.62;
    this._previewMaterial.backFaceCulling = true;
    this._previewMaterial.sideOrientation = Material.CounterClockWiseSideOrientation;
    this._previewMaterial.transparencyMode = PBRMaterial.PBRMATERIAL_ALPHABLEND;
    this._previewMaterial.needDepthPrePass = true;

    this._cutPreviewMaterial = new PBRMaterial('sketch-cut-preview-mat', this._scene);
    this._cutPreviewMaterial.albedoColor = PREVIEW_CUT;
    this._cutPreviewMaterial.metallic = 0;
    this._cutPreviewMaterial.roughness = 0.6;
    this._cutPreviewMaterial.alpha = 0.55;
    this._cutPreviewMaterial.backFaceCulling = true;
    this._cutPreviewMaterial.sideOrientation = Material.CounterClockWiseSideOrientation;
    this._cutPreviewMaterial.transparencyMode = PBRMaterial.PBRMATERIAL_ALPHABLEND;
    this._cutPreviewMaterial.depthFunction = Constants.ALWAYS;
    this._cutPreviewMaterial.disableDepthWrite = true;

    this._readout = document.createElement('div');
    this._readout.className = 'sketch-readout';
    this._readout.hidden = true;
    this._layer.appendChild(this._readout);

    this._pointerObserver = this._scene.onPointerObservable.add((info) => this._onPointer(info));
    this._prePointerObserver = this._scene.onPrePointerObservable.add((info) => {
      const event = info.event as PointerEvent;
      this._syncCameraButtons(event.button === 0 && event.shiftKey);
    }, PointerEventTypes.POINTERDOWN);
    this._renderObserver = this._scene.onBeforeRenderObservable.add(() => this._beforeRender());
    this._viewObserver = this._camera.onViewMatrixChangedObservable.add(() => {
      this._dirtyLabels = true;
      if (this._state?.stage === 'plane' || this._state?.stage === 'measure') this._dirtyHover = true;
    });
    this._onKey = (event) => this._handleKey(event);
    window.addEventListener('keydown', this._onKey);
  }

  setBounds(bounds: AssemblyBounds): void {
    this._bounds = bounds;
    if (this._state) this._rebuildAll();
  }

  setState(state: ModelingViewState | null): void {
    const previous = this._state;
    this._state = state;
    this._syncCameraButtons(false);
    if (!state) {
      this._draft = null;
      this._cursor = null;
      this._clearAll();
      this._canvas.style.cursor = '';
      return;
    }
    const planeChanged =
      !previous ||
      previous.stage !== state.stage ||
      previous.purpose !== state.purpose ||
      JSON.stringify(previous.plane) !== JSON.stringify(state.plane);
    if (previous?.tool !== state.tool || state.stage !== 'sketch') {
      this._draft = null;
      this._closeNumeric();
    }
    if (planeChanged) {
      this._rebuildAll();
      return;
    }
    if (
      state.stage === 'plane' &&
      (previous.referencePlanes !== state.referencePlanes ||
        previous.planePreview !== state.planePreview ||
        previous.chosenPlanes !== state.chosenPlanes)
    ) {
      this._clearPlaneStage();
      this._buildPlaneStage();
    }
    if (previous.entities !== state.entities || previous.selectedEntityIds !== state.selectedEntityIds) {
      this._rebuildEntities();
    }
    if (
      previous.preview !== state.preview ||
      previous.direction !== state.direction ||
      previous.extrudeDepth?.value !== state.extrudeDepth?.value ||
      previous.extrudeDepth?.midplane !== state.extrudeDepth?.midplane
    ) {
      this._rebuildPreview();
    }
    if (previous.measure !== state.measure) this._rebuildMeasure();
    if (previous.regions !== state.regions) this._rebuildRegions();
    this._dirtyRubber = true;
  }

  dispose(): void {
    this._scene.onPointerObservable.remove(this._pointerObserver);
    this._scene.onPrePointerObservable.remove(this._prePointerObserver);
    this._state = null;
    this._syncCameraButtons(false);
    this._scene.onBeforeRenderObservable.remove(this._renderObserver);
    this._camera.onViewMatrixChangedObservable.remove(this._viewObserver);
    window.removeEventListener('keydown', this._onKey);
    this._clearAll();
    this._planeMaterial.dispose();
    this._hintMaterial.dispose();
    this._measureFaceMaterial.dispose();
    this._previewMaterial.dispose();
    this._cutPreviewMaterial.dispose();
    this._readout.remove();
  }

  private _syncCameraButtons(orbitWithLeft: boolean): void {
    const pointers = this._camera.inputs.attached.pointers as unknown as { buttons: number[] } | undefined;
    if (!pointers) return;
    pointers.buttons = this._state?.stage === 'sketch' && !orbitWithLeft ? [1, 2] : [0, 1, 2];
  }

  private _clearPlaneStage(): void {
    for (const quad of this._planeQuads) quad.mesh.dispose(false, true);
    this._planeQuads = [];
    for (const mesh of this._decor) mesh.dispose(false, true);
    this._decor = [];
    this._hoverPlane = null;
    this._clearFaceHint();
    for (const label of this._labels) label.element.remove();
    this._labels = [];
  }

  private _clearFaceHint(): void {
    this._faceHint?.dispose();
    this._faceHint = null;
    this._hintSource = null;
  }

  private _clearAll(): void {
    this._clearPlaneStage();
    this._sketchPlane?.dispose(false, true);
    this._sketchPlane = null;
    this._grid?.dispose();
    this._grid = null;
    this._gridStep = 0;
    this._axes?.dispose();
    this._axes = null;
    for (const mesh of this._entities) mesh.dispose();
    this._entities = [];
    this._dimLines?.dispose();
    this._dimLines = null;
    this._rubber?.dispose();
    this._rubber = null;
    this._snapMarker?.dispose();
    this._snapMarker = null;
    this._preview?.dispose();
    this._preview = null;
    this._previewEdges?.dispose();
    this._previewEdges = null;
    this._arrow?.dispose();
    this._arrow = null;
    this._clearDepthLabel();
    this._depthHandle = null;
    this._closeNumeric();
    this._clearRegions();
    this._box?.element?.remove();
    this._box = null;
    this._clearMeasure();
    this._clearMeasureHover();
    this._clearDimLabels();
    this._readout.hidden = true;
  }

  private _clearDepthLabel(): void {
    this._depthLabel?.element.remove();
    this._depthLabel = null;
  }

  private _clearDimLabels(): void {
    this._editing = null;
    for (const label of this._dimLabels) label.element.remove();
    this._dimLabels = [];
  }

  private _rebuildAll(): void {
    this._clearAll();
    const state = this._state;
    if (!state) return;
    this._frame = state.plane ? planeFrame(state.plane) : null;
    if (state.stage === 'plane') {
      this._buildPlaneStage();
      return;
    }
    if (state.stage === 'measure') {
      this._rebuildMeasure();
      return;
    }
    this._buildSketchPlane();
    this._rebuildEntities();
    this._rebuildRegions();
    this._rebuildPreview();
    this._dirtyRubber = true;
    this._dirtyLabels = true;
  }

  private _extent(): number {
    return Math.max((this._bounds?.size ?? 10) * 0.75, 4);
  }

  private _gridCenter(frame: PlaneFrame): Vec2 {
    const center = this._bounds?.center;
    if (!center) return [0, 0];
    return worldToSketch(frame, [center.x, center.y, center.z]);
  }

  private _quad(name: string, frame: PlaneFrame, center: Vec2, half: number, material: StandardMaterial): Mesh {
    const corners: Vec2[] = [
      [center[0] - half, center[1] - half],
      [center[0] + half, center[1] - half],
      [center[0] + half, center[1] + half],
      [center[0] - half, center[1] + half]
    ];
    const positions = corners.flatMap((corner) => sketchToWorld(frame, corner));
    const mesh = new Mesh(name, this._scene);
    const data = new VertexData();
    data.positions = positions;
    data.indices = [0, 1, 2, 0, 2, 3];
    data.normals = [...frame.normal, ...frame.normal, ...frame.normal, ...frame.normal];
    data.applyToMesh(mesh);
    mesh.material = material;
    mesh.isPickable = false;
    return mesh;
  }

  private _outline(name: string, frame: PlaneFrame, center: Vec2, half: number, color: Color3, alpha: number, parent?: Mesh): LinesMesh {
    const corners: Vec2[] = [
      [center[0] - half, center[1] - half],
      [center[0] + half, center[1] - half],
      [center[0] + half, center[1] + half],
      [center[0] - half, center[1] + half],
      [center[0] - half, center[1] - half]
    ];
    const outline = MeshBuilder.CreateLines(
      name,
      { points: corners.map((corner) => toVector(sketchToWorld(frame, corner))) },
      this._scene
    );
    outline.color = color;
    outline.alpha = alpha;
    outline.isPickable = false;
    if (parent) outline.parent = parent;
    return outline;
  }

  private _planeLabel(text: string, anchor: Vec3, className = 'sketch-plane-label'): void {
    const label = document.createElement('div');
    label.className = className;
    label.textContent = text;
    this._layer.appendChild(label);
    this._labels.push({ element: label, anchor });
  }

  private _buildPlaneStage(): void {
    const state = this._state;
    if (!state) return;
    const half = this._extent();
    const planes: Array<{ key: string; plane: SketchPlane; name: string; color: Color3; size: number }> = [
      ...STANDARD_PLANES.map((plane) => ({
        key: `datum-${plane.id}`,
        plane: { kind: 'standard', id: plane.id, offset: 0 } as SketchPlane,
        name: `${plane.label} Plane`,
        color: PLANE_COLOR,
        size: half
      })),
      ...state.referencePlanes.map((ref) => ({
        key: `ref-${ref.id}`,
        plane: ref.plane,
        name: ref.name,
        color: REF_PLANE_COLOR,
        size: half * 0.8
      }))
    ];
    const chosenKeys = new Set(state.chosenPlanes.map((plane) => planeKey(plane)));
    for (const entry of planes) {
      const frame = planeFrame(entry.plane);
      const center = this._gridCenter(frame);
      const chosen = chosenKeys.has(planeKey(entry.plane));
      const color = chosen ? CHOSEN_PLANE_COLOR : entry.color;
      const material = flatMaterial(`sketch-plane-${entry.key}-mat`, color, chosen ? 0.3 : 0.12, this._scene);
      const mesh = this._quad(`sketch-plane-${entry.key}`, frame, center, entry.size, material);
      this._outline(`sketch-plane-edge-${entry.key}`, frame, center, entry.size, color, 0.7, mesh);
      this._planeQuads.push({ key: entry.key, plane: entry.plane, frame, center, half: entry.size, mesh });
      this._planeLabel(entry.name, sketchToWorld(frame, [center[0] - entry.size, center[1] + entry.size]));
    }
    for (const plane of state.chosenPlanes) {
      if (planes.some((entry) => planeKey(entry.plane) === planeKey(plane))) continue;
      const frame = planeFrame(plane);
      const center = this._gridCenter(frame);
      const material = flatMaterial('sketch-plane-chosen-mat', CHOSEN_PLANE_COLOR, 0.25, this._scene);
      const mesh = this._quad('sketch-plane-chosen', frame, center, half * 0.6, material);
      this._outline('sketch-plane-chosen-edge', frame, center, half * 0.6, CHOSEN_PLANE_COLOR, 0.9, mesh);
      this._decor.push(mesh);
      this._planeLabel(planeLabel(plane), sketchToWorld(frame, [center[0] - half * 0.6, center[1] + half * 0.6]));
    }
    state.planePreview.forEach((plane, index) => {
      const frame = planeFrame(plane);
      const center = this._gridCenter(frame);
      const size = half * 0.7;
      const material = flatMaterial(`sketch-plane-preview-${index}-mat`, PREVIEW_PLANE_COLOR, 0.18, this._scene);
      const mesh = this._quad(`sketch-plane-preview-${index}`, frame, center, size, material);
      this._outline(`sketch-plane-preview-edge-${index}`, frame, center, size, PREVIEW_PLANE_COLOR, 1, mesh);
      this._decor.push(mesh);
      const name = state.planePreview.length > 1 ? `New plane ${index + 1}` : 'New plane';
      this._planeLabel(`${name} · ${formatInches(plane.offset)} in`, sketchToWorld(frame, [center[0] - size, center[1] + size]), 'sketch-plane-label is-preview');
    });
    this._dirtyLabels = true;
  }

  private _buildSketchPlane(): void {
    const frame = this._frame;
    if (!frame) return;
    const center = this._gridCenter(frame);
    const half = this._extent();
    const material = flatMaterial('sketch-active-plane-mat', PLANE_COLOR, 0.05, this._scene);
    this._sketchPlane = this._quad('sketch-active-plane', frame, center, half, material);
    const axisLength = Math.max(half * 0.12, 0.5);
    this._axes = MeshBuilder.CreateLineSystem(
      'sketch-axes',
      {
        lines: [
          [toVector(sketchToWorld(frame, [0, 0])), toVector(sketchToWorld(frame, [axisLength, 0]))],
          [toVector(sketchToWorld(frame, [0, 0])), toVector(sketchToWorld(frame, [0, axisLength]))]
        ],
        colors: [
          [AXIS_U, AXIS_U],
          [AXIS_V, AXIS_V]
        ],
        useVertexAlpha: true
      },
      this._scene
    );
    this._axes.renderingGroupId = 1;
    this._axes.isPickable = false;
    this._rebuildGrid(true);
  }

  private _pixelsPerUnit(): number {
    const frame = this._frame;
    if (!frame) return 1;
    const a = this._project(frame.origin);
    const b = this._project(sketchToWorld(frame, [1, 0]));
    const c = this._project(sketchToWorld(frame, [0, 1]));
    if (!a || !b || !c) return 1;
    return Math.max(Math.hypot(b.x - a.x, b.y - a.y), Math.hypot(c.x - a.x, c.y - a.y), 1e-6);
  }

  private _rebuildGrid(force = false): void {
    const frame = this._frame;
    const state = this._state;
    if (!frame || !state || (state.stage !== 'sketch' && state.stage !== 'feature')) return;
    const ppu = this._pixelsPerUnit();
    const step = GRID_STEPS.find((candidate) => candidate * ppu >= GRID_TARGET_PX) ?? GRID_STEPS[GRID_STEPS.length - 1];
    if (!force && step === this._gridStep) return;
    this._gridStep = step;
    this._grid?.dispose();
    const center = this._gridCenter(frame);
    const half = Math.min(this._extent(), step * 160);
    const majorEvery = step < 1 ? Math.round(1 / step) : 5;
    const lines: Vector3[][] = [];
    const colors: Color4[][] = [];
    const startU = Math.floor((center[0] - half) / step);
    const endU = Math.ceil((center[0] + half) / step);
    const startV = Math.floor((center[1] - half) / step);
    const endV = Math.ceil((center[1] + half) / step);
    for (let i = startU; i <= endU; i += 1) {
      const u = i * step;
      const color = i % majorEvery === 0 ? GRID_MAJOR : GRID_MINOR;
      lines.push([toVector(sketchToWorld(frame, [u, startV * step])), toVector(sketchToWorld(frame, [u, endV * step]))]);
      colors.push([color, color]);
    }
    for (let j = startV; j <= endV; j += 1) {
      const v = j * step;
      const color = j % majorEvery === 0 ? GRID_MAJOR : GRID_MINOR;
      lines.push([toVector(sketchToWorld(frame, [startU * step, v])), toVector(sketchToWorld(frame, [endU * step, v]))]);
      colors.push([color, color]);
    }
    this._grid = MeshBuilder.CreateLineSystem('sketch-grid', { lines, colors, useVertexAlpha: true }, this._scene);
    this._grid.isPickable = false;
  }

  private _rebuildEntities(): void {
    const state = this._state;
    const frame = this._frame;
    for (const mesh of this._entities) mesh.dispose();
    this._entities = [];
    this._dimLines?.dispose();
    this._dimLines = null;
    this._clearDimLabels();
    if (!state || !frame || (state.stage !== 'sketch' && state.stage !== 'feature')) return;

    const ppu = this._pixelsPerUnit();
    const toWorld = (points: Vec2[]) => points.map((point) => toVector(sketchToWorld(frame, point)));
    const groups: Record<'normal' | 'selected' | 'construction', Vector3[][]> = { normal: [], selected: [], construction: [] };
    for (const entity of state.entities) {
      const path = entityPath(entity);
      if (path.points.length < 2) continue;
      const points = path.closed ? [...path.points, path.points[0]] : path.points;
      const selected = state.selectedEntityIds.includes(entity.id);
      if (entity.construction) {
        for (const dash of dashes(points, 7 / ppu, 5 / ppu)) (selected ? groups.selected : groups.construction).push(toWorld(dash));
      } else {
        (selected ? groups.selected : groups.normal).push(toWorld(points));
      }
      if ('cx' in entity && entity.type !== 'rectangle') {
        const mark = 4 / ppu;
        const c: Vec2 = [entity.cx, entity.cy];
        const target = selected ? groups.selected : entity.construction ? groups.construction : groups.normal;
        target.push(toWorld([[c[0] - mark, c[1]], [c[0] + mark, c[1]]]));
        target.push(toWorld([[c[0], c[1] - mark], [c[0], c[1] + mark]]));
      }
    }
    const colors = { normal: SKETCH_LINE, selected: SELECTED_LINE, construction: CONSTRUCTION_LINE };
    const thinLines: Vector3[][] = [];
    const thinColors: Color4[][] = [];
    for (const [key, lines] of Object.entries(groups) as Array<[keyof typeof colors, Vector3[][]]>) {
      const color = Color4.FromColor3(colors[key], 1);
      for (const line of lines) {
        thinLines.push(line);
        thinColors.push(line.map(() => color));
      }
      const mesh = createFatLines(`sketch-entities-${key}`, lines, colors[key], key === 'construction' ? 1.5 : 2.25, this._scene);
      if (!mesh) continue;
      mesh.renderingGroupId = 1;
      this._entities.push(mesh);
    }
    if (thinLines.length) {
      const thin = MeshBuilder.CreateLineSystem('sketch-entities-thin', { lines: thinLines, colors: thinColors, useVertexAlpha: true }, this._scene);
      thin.renderingGroupId = 1;
      thin.isPickable = false;
      this._entities.push(thin);
    }

    const openEnds = sketchProfiles(state.entities).openEnds;
    if (openEnds.length) {
      const size = 4 / ppu;
      const squares = openEnds.map((p) =>
        toWorld([
          [p[0] - size, p[1] - size],
          [p[0] + size, p[1] - size],
          [p[0] + size, p[1] + size],
          [p[0] - size, p[1] + size],
          [p[0] - size, p[1] - size]
        ])
      );
      const mesh = createFatLines('sketch-open-ends', squares, OPEN_END, 2.5, this._scene);
      if (mesh) {
        mesh.renderingGroupId = 1;
        this._entities.push(mesh);
      }
    }

    if (state.stage !== 'sketch') {
      this._dirtyLabels = true;
      return;
    }
    const gap = Math.max(16 / ppu, 0.05);
    const dims = sketchDimensions(state.entities, this._primarySelection(), gap);
    const dimLines: Vector3[][] = [];
    const dimColors: Color4[][] = [];
    for (const spec of dims) {
      if (spec.line) {
        dimLines.push(toWorld(spec.line));
        dimColors.push([DIM_COLOR, DIM_COLOR]);
      }
      const element = document.createElement('button');
      element.type = 'button';
      element.className =
        spec.key === 'cx'
          ? 'sketch-dim sketch-dim-position sketch-dim-x'
          : spec.key === 'cy'
            ? 'sketch-dim sketch-dim-position sketch-dim-y'
            : 'sketch-dim';
      element.textContent = `${spec.prefix}${formatInches(spec.value)}`;
      element.title = 'Click to edit this dimension';
      element.addEventListener('pointerdown', (event) => event.stopPropagation());
      element.addEventListener('click', (event) => {
        event.stopPropagation();
        this._beginEdit(spec, element);
      });
      this._layer.appendChild(element);
      this._dimLabels.push({ element, anchor: sketchToWorld(frame, spec.anchor), spec });
    }
    if (dimLines.length) {
      this._dimLines = MeshBuilder.CreateLineSystem('sketch-dim-lines', { lines: dimLines, colors: dimColors, useVertexAlpha: true }, this._scene);
      this._dimLines.renderingGroupId = 1;
      this._dimLines.isPickable = false;
    }
    this._dirtyLabels = true;
  }

  private _beginEdit(spec: DimensionSpec, element: HTMLElement, typed?: string): void {
    if (!this._state) return;
    const input = document.createElement('input');
    input.className = 'sketch-dim-input';
    input.type = 'text';
    input.inputMode = 'decimal';
    input.value = typed ?? formatInches(spec.value);
    input.style.left = element.style.left;
    input.style.top = element.style.top;
    element.replaceWith(input);
    const label = this._dimLabels.find((entry) => entry.element === element);
    if (label) label.element = input;
    this._editing = { spec, input };
    input.focus();
    if (typed === undefined) input.select();
    else input.setSelectionRange(input.value.length, input.value.length);
    let done = false;
    const finish = (commit: boolean) => {
      if (done) return;
      done = true;
      this._editing = null;
      const entity = this._state?.entities.find((item) => item.id === spec.entityId);
      if (commit && entity) {
        const value = evaluateLength(input.value);
        const next = value === null ? null : withDimension(entity, spec.key, value);
        if (next) {
          this._emit({ type: 'entityChanged', entity: next });
          return;
        }
      }
      this._rebuildEntities();
    };
    input.addEventListener('keydown', (event) => {
      event.stopPropagation();
      if (event.key === 'Enter') finish(true);
      else if (event.key === 'Escape') finish(false);
    });
    input.addEventListener('blur', () => finish(true));
    input.addEventListener('pointerdown', (event) => event.stopPropagation());
  }

  private _rebuildPreview(): void {
    const state = this._state;
    this._preview?.dispose();
    this._preview = null;
    this._previewEdges?.dispose();
    this._previewEdges = null;
    this._arrow?.dispose();
    this._arrow = null;
    this._clearDepthLabel();
    this._depthHandle = null;
    if (!state || state.stage !== 'feature') return;

    const preview = state.preview;
    if (preview && preview.positions.length && preview.indices.length) {
      const creased = buildCreasedGeometry(preview.positions, preview.indices);
      const mesh = new Mesh('sketch-preview', this._scene);
      mesh.sideOrientation = Material.CounterClockWiseSideOrientation;
      const data = new VertexData();
      data.positions = creased.positions;
      data.indices = creased.indices;
      data.normals = creased.normals;
      data.applyToMesh(mesh);
      if (preview.cut && preview.ok) {
        mesh.material = this._cutPreviewMaterial;
      } else {
        this._previewMaterial.albedoColor = preview.ok ? PREVIEW_OK : PREVIEW_BAD;
        mesh.material = this._previewMaterial;
      }
      mesh.isPickable = false;
      this._preview = mesh;
      if (creased.edgeSegments.length) {
        const lines: Vector3[][] = [];
        const e = creased.edgeSegments;
        for (let i = 0; i < e.length; i += 6) {
          lines.push([new Vector3(e[i], e[i + 1], e[i + 2]), new Vector3(e[i + 3], e[i + 4], e[i + 5])]);
        }
        this._previewEdges = MeshBuilder.CreateLineSystem('sketch-preview-edges', { lines }, this._scene);
        this._previewEdges.color = preview.cut ? new Color3(0.45, 0.05, 0.15) : new Color3(0.35, 0.25, 0.05);
        this._previewEdges.isPickable = false;
        if (preview.cut) this._previewEdges.renderingGroupId = 1;
      }
    }

    const frame = this._frame;
    const direction = state.direction;
    if (frame && direction && state.entities.length) {
      const centroid = sketchCentroid(state.entities);
      const origin = toVector(sketchToWorld(frame, centroid));
      const dir = new Vector3(direction[0], direction[1], direction[2]).normalize();
      const depth = state.extrudeDepth;
      const endOffset = depth ? (depth.midplane ? depth.value / 2 : depth.value) : 0;
      const base = origin.add(dir.scale(endOffset));
      const length = Math.max(this._extent() * 0.25, 1);
      const tip = base.add(dir.scale(length));
      const side = Vector3.Cross(dir, new Vector3(frame.u[0], frame.u[1], frame.u[2])).normalize();
      const across = side.lengthSquared() > 0.5 ? side : new Vector3(frame.v[0], frame.v[1], frame.v[2]);
      const head = length * 0.18;
      const back = tip.subtract(dir.scale(head));
      this._arrow = createFatLines(
        'sketch-direction',
        [
          [base, tip],
          [back.add(across.scale(head * 0.5)), tip, back.subtract(across.scale(head * 0.5))]
        ],
        this._depthDrag ? SELECTED_LINE : ARROW_LINE,
        this._depthDrag ? 4 : 3,
        this._scene
      );
      if (this._arrow) this._arrow.renderingGroupId = 1;
      if (depth) {
        this._depthHandle = { origin, dir, base, tip, midplane: depth.midplane };
        const element = document.createElement('button');
        element.type = 'button';
        element.className = 'sketch-dim sketch-depth';
        element.textContent = `${depth.midplane ? 'Total ' : ''}${formatInches(depth.value)} in`;
        element.title = 'Drag the arrow to set the depth, or click to type it';
        element.addEventListener('pointerdown', (event) => event.stopPropagation());
        element.addEventListener('click', (event) => {
          event.stopPropagation();
          this._editDepth(element, depth.value);
        });
        this._layer.appendChild(element);
        this._depthLabel = { element, anchor: [tip.x, tip.y, tip.z] };
      }
    }
    this._dirtyLabels = true;
  }

  private _clearRegions(): void {
    for (const mesh of this._regionMeshes) mesh.dispose(false, true);
    this._regionMeshes = [];
    this._regionLines?.dispose();
    this._regionLines = null;
    this._hoverRegion = -1;
  }

  private _rebuildRegions(): void {
    this._clearRegions();
    const state = this._state;
    const frame = this._frame;
    if (!state || !frame || state.stage !== 'feature' || !state.regions) return;
    const outlines: Vector3[][] = [];
    const colors: Color4[][] = [];
    state.regions.forEach((region, index) => {
      const positions: number[] = [];
      for (let i = 0; i < region.triangles.length; i += 2) {
        positions.push(...sketchToWorld(frame, [region.triangles[i], region.triangles[i + 1]]));
      }
      const mesh = new Mesh(`sketch-region-${index}`, this._scene);
      const data = new VertexData();
      data.positions = positions;
      data.indices = Array.from({ length: positions.length / 3 }, (_, i) => i);
      data.applyToMesh(mesh);
      const material = flatMaterial(`sketch-region-${index}-mat`, region.selected ? REGION_SELECTED : REGION_OPEN, 0, this._scene);
      material.disableDepthWrite = true;
      material.zOffset = -2;
      material.zOffsetUnits = -8;
      mesh.material = material;
      mesh.isPickable = false;
      this._regionMeshes.push(mesh);
      const color = region.selected ? Color4.FromColor3(REGION_SELECTED, 0.9) : Color4.FromColor3(REGION_OPEN, 0.55);
      for (const contour of region.contours) {
        const line = [...contour, contour[0]].map((point) => toVector(sketchToWorld(frame, point)));
        outlines.push(line);
        colors.push(line.map(() => color));
      }
    });
    if (outlines.length) {
      this._regionLines = MeshBuilder.CreateLineSystem('sketch-region-lines', { lines: outlines, colors, useVertexAlpha: true }, this._scene);
      this._regionLines.renderingGroupId = 1;
      this._regionLines.isPickable = false;
    }
    this._paintRegions();
  }

  private _paintRegions(): void {
    const regions = this._state?.regions ?? [];
    this._regionMeshes.forEach((mesh, index) => {
      const material = mesh.material as StandardMaterial;
      const selected = regions[index]?.selected;
      const hover = index === this._hoverRegion;
      material.alpha = selected ? (hover ? 0.5 : 0.34) : hover ? 0.22 : 0.05;
    });
  }

  private _regionAtPointer(): number {
    const regions = this._state?.regions;
    if (!regions?.length) return -1;
    const raw = this._pointerOnPlane();
    if (!raw) return -1;
    return regions.findIndex((region) => pointInContours(region.contours, raw));
  }

  private _editDepth(element: HTMLElement, value: number): void {
    const input = document.createElement('input');
    input.className = 'sketch-dim-input';
    input.type = 'text';
    input.inputMode = 'decimal';
    input.value = formatInches(value);
    input.style.left = element.style.left;
    input.style.top = element.style.top;
    element.replaceWith(input);
    if (this._depthLabel) this._depthLabel.element = input;
    input.focus();
    input.select();
    let done = false;
    const finish = (commit: boolean) => {
      if (done) return;
      done = true;
      const next = commit ? evaluateLength(input.value) : null;
      if (next !== null && next > 1e-4 && Math.abs(next - value) > 1e-9) {
        this._emit({ type: 'depthChanged', depth: next });
      } else {
        this._rebuildPreview();
      }
    };
    input.addEventListener('keydown', (event) => {
      event.stopPropagation();
      if (event.key === 'Enter') finish(true);
      else if (event.key === 'Escape') finish(false);
    });
    input.addEventListener('blur', () => finish(true));
    input.addEventListener('pointerdown', (event) => event.stopPropagation());
  }

  private _onDepthHandle(): boolean {
    const handle = this._depthHandle;
    if (!handle) return false;
    const a = this._project([handle.base.x, handle.base.y, handle.base.z]);
    const b = this._project([handle.tip.x, handle.tip.y, handle.tip.z]);
    if (!a || !b) return false;
    if (Math.hypot(b.x - a.x, b.y - a.y) < 14) return false;
    const p: Vec2 = [this._scene.pointerX, this._scene.pointerY];
    return segmentDistance(p, [a.x, a.y], [b.x, b.y]) <= 10;
  }

  private _depthAtPointer(free: boolean): number | null {
    const handle = this._depthHandle;
    if (!handle) return null;
    const { origin, direction } = this._ray();
    const w0 = handle.origin.subtract(origin);
    const a = Vector3.Dot(handle.dir, handle.dir);
    const b = Vector3.Dot(handle.dir, direction);
    const c = Vector3.Dot(direction, direction);
    const d = Vector3.Dot(handle.dir, w0);
    const e = Vector3.Dot(direction, w0);
    const denom = a * c - b * b;
    if (Math.abs(denom) < 1e-9) return null;
    const t = (b * e - c * d) / denom;
    let depth = handle.midplane ? 2 * t : t;
    const step = (this._gridStep || 0.125) / 4;
    if (!free) depth = Math.round(depth / step) * step;
    return Math.max(depth, free ? 1e-3 : step);
  }

  private _clearMeasure(): void {
    for (const mesh of this._measureMeshes) mesh.dispose();
    this._measureMeshes = [];
    for (const label of this._measureLabels) label.element.remove();
    this._measureLabels = [];
  }

  private _clearMeasureHover(): void {
    for (const mesh of this._measureHover) mesh.dispose();
    this._measureHover = [];
    this._hoverMeasureKey = '';
  }

  private _drawPick(pick: MeasurePick, color: Color3, into: Mesh[], labels: Label[] | null, material: StandardMaterial): void {
    if (pick.kind === 'face') {
      into.push(this._triangleMesh('measure-face', pick.triangles, pick.normal, material));
    } else if (pick.kind === 'line' || pick.kind === 'circle') {
      const points = pick.points.map((p) => new Vector3(p[0], p[1], p[2]));
      if (pick.kind === 'circle' && pick.full) points.push(points[0]);
      const mesh = createFatLines('measure-edge', [points], color, 3, this._scene);
      if (mesh) {
        mesh.renderingGroupId = 1;
        into.push(mesh);
      }
    }
    if (labels && (pick.kind === 'vertex' || pick.kind === 'point' || pick.kind === 'circle')) {
      const dot = document.createElement('div');
      dot.className = 'measure-dot';
      this._layer.appendChild(dot);
      labels.push({ element: dot, anchor: pick.kind === 'circle' ? pick.center : pick.point });
    }
  }

  private _triangleMesh(name: string, triangles: Float32Array, normal: Vec3, material: StandardMaterial): Mesh {
    const lift = Math.max((this._bounds?.size ?? 10) * 2e-4, 1e-4);
    const positions = new Float32Array(triangles.length);
    for (let i = 0; i < triangles.length; i += 3) {
      positions[i] = triangles[i] + normal[0] * lift;
      positions[i + 1] = triangles[i + 1] + normal[1] * lift;
      positions[i + 2] = triangles[i + 2] + normal[2] * lift;
    }
    const mesh = new Mesh(name, this._scene);
    const data = new VertexData();
    data.positions = positions;
    data.indices = Array.from({ length: positions.length / 3 }, (_, i) => i);
    data.applyToMesh(mesh);
    mesh.material = material;
    mesh.isPickable = false;
    return mesh;
  }

  private _rebuildMeasure(): void {
    this._clearMeasure();
    const measure = this._state?.measure;
    if (!measure) return;
    for (const pick of measure.picks) this._drawPick(pick, MEASURE_LINE, this._measureMeshes, this._measureLabels, this._measureFaceMaterial);
    const line = measure.result?.line;
    if (line) {
      const from = new Vector3(...line.from);
      const to = new Vector3(...line.to);
      if (Vector3.Distance(from, to) > 1e-9) {
        const mesh = createFatLines('measure-dimension', [[from, to]], MEASURE_LINE, 2, this._scene);
        if (mesh) {
          mesh.renderingGroupId = 1;
          this._measureMeshes.push(mesh);
        }
      }
      const label = document.createElement('div');
      label.className = 'measure-label';
      label.textContent = line.label;
      this._layer.appendChild(label);
      this._measureLabels.push({
        element: label,
        anchor: [(line.from[0] + line.to[0]) / 2, (line.from[1] + line.to[1]) / 2, (line.from[2] + line.to[2]) / 2]
      });
    }
    this._dirtyLabels = true;
  }

  private _hoverMeasure(): void {
    const pick = this._pickMeasure(this._scene.pointerX, this._scene.pointerY);
    const key = pick ? measureKey(pick) : '';
    this._canvas.style.cursor = pick ? 'pointer' : 'default';
    if (key === this._hoverMeasureKey) return;
    this._clearMeasureHover();
    this._hoverMeasureKey = key;
    if (pick) this._drawPick(pick, MEASURE_HOVER, this._measureHover, null, this._hintMaterial);
  }

  private _beforeRender(): void {
    const state = this._state;
    if (!state) return;
    for (const mesh of [...this._entities, this._rubber, this._snapMarker, this._arrow, ...this._measureMeshes, ...this._measureHover]) {
      syncFatLineResolution(mesh, this._scene);
    }
    if (this._dirtyLabels) {
      this._dirtyLabels = false;
      if (state.stage === 'sketch' || state.stage === 'feature') {
        const previousStep = this._gridStep;
        this._rebuildGrid();
        if (previousStep !== this._gridStep && !this._editing) this._rebuildEntities();
      }
      this._placeLabels();
    }
    if (this._dirtyRubber) {
      this._dirtyRubber = false;
      this._drawRubberBand();
    }
    if (this._dirtyHover && !this._pointerDown) {
      this._dirtyHover = false;
      if (state.stage === 'plane' || (state.stage === 'sketch' && state.tool === 'convert')) this._hoverPlanes();
      else if (state.stage === 'measure') this._hoverMeasure();
    }
  }

  private _project(point: Vec3): { x: number; y: number } | null {
    const engine = this._scene.getEngine();
    const projected = Vector3.Project(
      new Vector3(point[0], point[1], point[2]),
      Matrix.Identity(),
      this._scene.getTransformMatrix(),
      this._camera.viewport.toGlobal(engine.getRenderWidth(), engine.getRenderHeight()) as Viewport
    );
    if (projected.z < 0 || projected.z > 1) return null;
    const scale = engine.getHardwareScalingLevel();
    return { x: projected.x * scale, y: projected.y * scale };
  }

  private _placeLabels(): void {
    const depth = this._depthLabel ? [this._depthLabel] : [];
    for (const label of [...this._labels, ...this._dimLabels, ...this._measureLabels, ...depth]) {
      const at = this._project(label.anchor);
      if (!at) {
        label.element.style.display = 'none';
        continue;
      }
      label.element.style.display = '';
      label.element.style.left = `${at.x}px`;
      label.element.style.top = `${at.y}px`;
    }
  }

  private _ray(): { origin: Vector3; direction: Vector3 } {
    const ray = this._scene.createPickingRay(this._scene.pointerX, this._scene.pointerY, Matrix.Identity(), this._camera);
    return { origin: ray.origin, direction: ray.direction };
  }

  private _pointerOnPlane(frame: PlaneFrame | null = this._frame): Vec2 | null {
    if (!frame) return null;
    const { origin, direction } = this._ray();
    const n = new Vector3(frame.normal[0], frame.normal[1], frame.normal[2]);
    const denom = Vector3.Dot(direction, n);
    if (Math.abs(denom) < 1e-9) return null;
    const o = new Vector3(frame.origin[0], frame.origin[1], frame.origin[2]);
    const t = Vector3.Dot(o.subtract(origin), n) / denom;
    const hit = origin.add(direction.scale(t));
    return worldToSketch(frame, [hit.x, hit.y, hit.z]);
  }

  private _snap(raw: Vec2, free: boolean): Snap {
    const state = this._state;
    if (!state || free || !state.snap) return { point: raw, kind: 'free' };
    const radius = SNAP_RADIUS_PX / this._pixelsPerUnit();
    let best: Vec2 | null = null;
    let bestDistance = radius;
    const candidates: Vec2[] = [[0, 0]];
    const dragging = this._drag?.kind === 'entities' ? this._drag.ids : this._drag?.kind === 'end' ? new Set([this._drag.id]) : null;
    for (const entity of state.entities) {
      if (dragging?.has(entity.id)) continue;
      candidates.push(...entityKeyPoints(entity));
    }
    const draft = this._draft;
    if (draft) {
      if ('first' in draft) candidates.push(draft.first);
      if ('second' in draft && draft.second) candidates.push(draft.second);
      if ('chainStart' in draft) candidates.push(draft.chainStart);
      if (draft.tool === 'centerArc') candidates.push(draft.center);
    }
    for (const candidate of candidates) {
      const distance = distance2(candidate, raw);
      if (distance < bestDistance) {
        best = candidate;
        bestDistance = distance;
      }
    }
    if (best) return { point: [best[0], best[1]], kind: 'point' };

    if (draft && (draft.tool === 'line' || draft.tool === 'centerline')) {
      const dx = raw[0] - draft.start[0];
      const dy = raw[1] - draft.start[1];
      const angle = Math.abs((Math.atan2(dy, dx) * 180) / Math.PI);
      const step = this._gridStep || 0.125;
      if (Math.min(angle, 180 - angle) < HV_INFERENCE_DEG) {
        return { point: [Math.round(raw[0] / step) * step, draft.start[1]], kind: 'horizontal' };
      }
      if (Math.abs(angle - 90) < HV_INFERENCE_DEG) {
        return { point: [draft.start[0], Math.round(raw[1] / step) * step], kind: 'vertical' };
      }
    }
    const step = this._gridStep || 0.125;
    return { point: [Math.round(raw[0] / step) * step, Math.round(raw[1] / step) * step], kind: 'grid' };
  }

  private _onPointer(info: PointerInfo): void {
    const state = this._state;
    if (!state || this._editing) return;
    const event = info.event as PointerEvent;
    const scene = this._scene;
    switch (info.type) {
      case PointerEventTypes.POINTERMOVE:
        this._onMove(event);
        break;
      case PointerEventTypes.POINTERDOWN: {
        const orbit = state.stage === 'sketch' && event.button === 0 && event.shiftKey;
        const down = { x: scene.pointerX, y: scene.pointerY, button: event.button, orbit, startedDraft: false };
        this._pointerDown = down;
        if (event.button !== 0 || orbit) break;
        if (state.stage === 'feature' && this._onDepthHandle()) {
          this._camera.detachControl();
          this._depthDrag = { last: state.extrudeDepth?.value ?? 0 };
          this._rebuildPreview();
          break;
        }
        this._maybeStartDrag(event);
        if (state.stage === 'sketch' && isDrawingTool(state.tool) && !this._draft) {
          this._closeNumeric();
          this._onClick(event);
          down.startedDraft = this._draft !== null;
        }
        break;
      }
      case PointerEventTypes.POINTERUP: {
        const down = this._pointerDown;
        this._pointerDown = null;
        this._dirtyHover = true;
        if (this._depthDrag) {
          this._depthDrag = null;
          this._camera.attachControl(this._canvas, false);
          this._rebuildPreview();
          break;
        }
        if (this._drag) {
          this._endDrag();
          break;
        }
        if (this._box && this._endBox()) break;
        if (event.button !== 0 || !down || down.button !== 0 || down.orbit) break;
        const moved = Math.hypot(scene.pointerX - down.x, scene.pointerY - down.y) > CLICK_DRAG_THRESHOLD_PX;
        if (state.stage === 'sketch' && isDrawingTool(state.tool)) {
          if (down.startedDraft && !moved) break;
          const dragged = down.startedDraft && moved;
          this._onClick(event);
          if (dragged && (state.tool === 'line' || state.tool === 'centerline')) {
            this._draft = null;
            this._dirtyRubber = true;
          }
          break;
        }
        if (moved) break;
        this._onClick(event);
        break;
      }
      case PointerEventTypes.POINTERDOUBLETAP:
        if (state.stage === 'sketch' && (this._draft?.tool === 'line' || this._draft?.tool === 'centerline')) {
          this._draft = null;
          this._dirtyRubber = true;
        }
        break;
      default:
        break;
    }
  }

  private _onMove(event: PointerEvent): void {
    const state = this._state;
    if (!state) return;
    if (state.stage === 'feature') {
      if (this._depthDrag) {
        const depth = this._depthAtPointer(event.altKey);
        if (depth !== null && Math.abs(depth - this._depthDrag.last) > 1e-9) {
          this._depthDrag.last = depth;
          this._emit({ type: 'depthChanged', depth });
        }
        return;
      }
      if (!this._pointerDown) {
        const onHandle = this._onDepthHandle();
        const region = onHandle ? -1 : this._regionAtPointer();
        if (region !== this._hoverRegion) {
          this._hoverRegion = region;
          this._paintRegions();
        }
        this._canvas.style.cursor = onHandle ? 'grab' : region >= 0 ? 'pointer' : '';
      }
      return;
    }
    if (state.stage === 'plane' || state.stage === 'measure' || (state.stage === 'sketch' && state.tool === 'convert')) {
      this._dirtyHover = true;
      if (state.stage !== 'sketch') return;
    }
    if (state.stage !== 'sketch') {
      this._canvas.style.cursor = '';
      return;
    }
    const raw = this._pointerOnPlane();
    if (!raw) return;
    this._cursor = this._snap(raw, event.altKey);
    const draft = this._draft;
    if (draft?.tool === 'centerArc' && draft.start) {
      const angle = Math.atan2(this._cursor.point[1] - draft.center[1], this._cursor.point[0] - draft.center[0]);
      let delta = angle - draft.lastAngle;
      if (delta > Math.PI) delta -= Math.PI * 2;
      if (delta < -Math.PI) delta += Math.PI * 2;
      draft.sweep += delta;
      draft.lastAngle = angle;
    }
    if (this._drag) {
      this._drag.moved = true;
      this._previewDrag();
    }
    if (this._box) this._updateBox();
    if (state.tool === 'select') {
      const end = this._endAt(raw);
      this._canvas.style.cursor = end ? 'crosshair' : this._entityAt(raw) ? 'move' : 'default';
    } else if (state.tool !== 'convert') {
      this._canvas.style.cursor = 'crosshair';
    }
    this._updateReadout();
    this._dirtyRubber = true;
  }

  private _updateReadout(): void {
    const cursor = this._cursor;
    if (!cursor) return;
    let text = `X ${formatInches(cursor.point[0])}   Y ${formatInches(cursor.point[1])}`;
    const draft = this._draft;
    const from =
      draft && 'start' in draft && draft.start
        ? draft.start
        : draft && 'first' in draft
          ? draft.first
          : draft?.tool === 'centerArc'
            ? draft.center
            : null;
    if (from) {
      const dx = cursor.point[0] - from[0];
      const dy = cursor.point[1] - from[1];
      text += `   L ${formatInches(Math.hypot(dx, dy))}   ∠ ${((Math.atan2(dy, dx) * 180) / Math.PI).toFixed(1)}°`;
    }
    if (cursor.kind === 'horizontal') text += '   ─ Horizontal';
    if (cursor.kind === 'vertical') text += '   │ Vertical';
    if (this._numericPrompt() && !this._numeric) text += '   · type a size';
    this._readout.hidden = false;
    this._readout.textContent = `${text}  in`;
  }

  private _planeTarget(): { surface: SurfacePick } | { quad: PlaneQuad } | null {
    const surface = this._pickSurface(this._scene.pointerX, this._scene.pointerY);
    if (surface) return { surface };
    if (this._state?.stage !== 'plane') return null;
    const quad = this._planeUnderPointer();
    return quad ? { quad } : null;
  }

  private _hoverPlanes(): void {
    const target = this._planeTarget();
    const surface = target && 'surface' in target ? target.surface : null;
    const hovered = target && 'quad' in target ? target.quad.key : null;
    if (hovered !== this._hoverPlane) {
      this._hoverPlane = hovered;
      for (const quad of this._planeQuads) {
        const material = quad.mesh.material as StandardMaterial;
        material.alpha = quad.key === hovered ? 0.32 : 0.12;
      }
    }
    const triangles = surface?.face?.world ?? null;
    if (triangles !== this._hintSource) {
      this._clearFaceHint();
      this._hintSource = triangles;
      if (triangles && surface) {
        this._faceHint = this._triangleMesh('sketch-face-hint', triangles, [surface.normal.x, surface.normal.y, surface.normal.z], this._hintMaterial);
      }
    }
    this._canvas.style.cursor = hovered || surface ? 'pointer' : 'default';
  }

  private _planeUnderPointer(): PlaneQuad | null {
    const { origin, direction } = this._ray();
    let best: { quad: PlaneQuad; distance: number } | null = null;
    for (const quad of this._planeQuads) {
      const n = new Vector3(...quad.frame.normal);
      const denom = Vector3.Dot(direction, n);
      if (Math.abs(denom) < 1e-9) continue;
      const o = new Vector3(...quad.frame.origin);
      const t = Vector3.Dot(o.subtract(origin), n) / denom;
      if (t <= 0) continue;
      const hit = origin.add(direction.scale(t));
      const [u, v] = worldToSketch(quad.frame, [hit.x, hit.y, hit.z]);
      if (Math.abs(u - quad.center[0]) > quad.half || Math.abs(v - quad.center[1]) > quad.half) continue;
      if (!best || t < best.distance) best = { quad, distance: t };
    }
    return best?.quad ?? null;
  }

  private _faceOutline(surface: SurfacePick): FaceOutline | null {
    const face = surface.face;
    if (!face) return null;
    return { loops: faceBoundaryLoops(face.world), planar: face.planar };
  }

  private _onClick(event: PointerEvent): void {
    const state = this._state;
    if (!state) return;
    if (state.stage === 'plane') {
      const target = this._planeTarget();
      if (target && 'surface' in target) {
        const surface = target.surface;
        const label = surface.partName ? `Face of ${surface.partName}` : 'Face';
        const normal = surface.face?.planar ? surface.face.normal : ([surface.normal.x, surface.normal.y, surface.normal.z] as Vec3);
        this._emit({
          type: 'planePicked',
          plane: planeThroughPoint([surface.point.x, surface.point.y, surface.point.z], normal, label),
          face: this._faceOutline(surface)
        });
      } else if (target) {
        this._emit({ type: 'planePicked', plane: target.quad.plane, face: null });
      }
      return;
    }
    if (state.stage === 'measure') {
      this._emit({ type: 'measurePicked', pick: this._pickMeasure(this._scene.pointerX, this._scene.pointerY) });
      return;
    }
    if (state.stage === 'feature') {
      const raw = state.regions ? this._pointerOnPlane() : null;
      if (raw && this._regionAtPointer() >= 0) this._emit({ type: 'regionToggled', point: raw });
      return;
    }
    if (state.stage !== 'sketch') return;

    if (state.tool === 'convert') {
      const surface = this._pickSurface(this._scene.pointerX, this._scene.pointerY);
      const outline = surface ? this._faceOutline(surface) : null;
      if (outline) this._emit({ type: 'convertFace', face: outline });
      return;
    }

    const raw = this._pointerOnPlane();
    if (!raw) return;
    const snap = this._snap(raw, event.altKey);
    const point = snap.point;

    if (state.tool === 'select') {
      const hit = this._entityAt(raw);
      if (addsToSelection(event)) {
        if (hit) {
          const ids = state.selectedEntityIds;
          this._emit({ type: 'entitySelected', ids: ids.includes(hit.id) ? ids.filter((id) => id !== hit.id) : [...ids, hit.id] });
        }
      } else {
        this._emit({ type: 'entitySelected', ids: hit ? [hit.id] : [] });
      }
      return;
    }
    this._applyPoint(point, snap.kind);
  }

  private _applyPoint(point: Vec2, snapKind: SnapKind): void {
    const state = this._state;
    if (!state || state.stage !== 'sketch') return;
    const draft = this._draft;
    switch (state.tool) {
      case 'line':
      case 'centerline': {
        if (!draft || (draft.tool !== 'line' && draft.tool !== 'centerline')) {
          this._draft = { tool: state.tool, start: point, chainStart: point };
          break;
        }
        if (distance2(point, draft.start) < 1e-9) break;
        const construction = state.tool === 'centerline';
        const exactly =
          Math.abs(point[1] - draft.start[1]) < 1e-9 ? 'horizontal' : Math.abs(point[0] - draft.start[0]) < 1e-9 ? 'vertical' : null;
        this._emit({
          type: 'entityCreated',
          entity: {
            id: newEntityId(),
            type: 'line',
            x1: draft.start[0],
            y1: draft.start[1],
            x2: point[0],
            y2: point[1],
            ...(construction ? { construction } : {}),
            ...(exactly && snapKind !== 'free' ? { relation: exactly } : {})
          }
        });
        const closes = distance2(point, draft.chainStart) <= COINCIDENT_TOL || (snapKind === 'point' && this._isEnd(point));
        this._draft = closes || construction ? null : { ...draft, start: point };
        break;
      }
      case 'arc3': {
        if (!draft || draft.tool !== 'arc3') {
          this._draft = { tool: 'arc3', first: point, second: null };
        } else if (!draft.second) {
          if (distance2(point, draft.first) > 1e-6) draft.second = point;
        } else {
          const arc = arcThrough(draft.first, draft.second, point, newEntityId());
          this._draft = null;
          if (arc) this._emit({ type: 'entityCreated', entity: arc });
        }
        break;
      }
      case 'centerArc': {
        if (!draft || draft.tool !== 'centerArc') {
          this._draft = { tool: 'centerArc', center: point, start: null, sweep: 0, lastAngle: 0 };
        } else if (!draft.start) {
          if (distance2(point, draft.center) > 1e-6) {
            draft.start = point;
            draft.sweep = 0;
            draft.lastAngle = Math.atan2(point[1] - draft.center[1], point[0] - draft.center[0]);
          }
        } else {
          const arc = arcFromCenter(draft.center, draft.start, point, draft.sweep < 0, newEntityId());
          this._draft = null;
          if (arc) this._emit({ type: 'entityCreated', entity: arc });
        }
        break;
      }
      case 'rectangle':
      case 'centerRectangle':
      case 'circle':
      case 'polygon': {
        if (!draft || !('first' in draft) || draft.tool !== state.tool) {
          this._draft = { tool: state.tool, first: point };
          break;
        }
        const entity = this._shapeFrom(state.tool, draft.first, point);
        this._draft = null;
        if (entity) this._emit({ type: 'entityCreated', entity });
        break;
      }
      case 'slot': {
        if (!draft || draft.tool !== 'slot') {
          this._draft = { tool: 'slot', first: point, second: null };
        } else if (!draft.second) {
          if (distance2(point, draft.first) > 1e-6) draft.second = point;
        } else {
          const entity = slotFrom(draft.first, draft.second, point);
          this._draft = null;
          if (entity) this._emit({ type: 'entityCreated', entity });
        }
        break;
      }
      default:
        break;
    }
    this._dirtyRubber = true;
  }

  private _isEnd(point: Vec2): boolean {
    return (this._state?.entities ?? []).some((entity) => {
      const ends = entityEnds(entity);
      return ends !== null && ends.some((end) => distance2(end, point) <= COINCIDENT_TOL);
    });
  }

  private _shapeFrom(tool: 'rectangle' | 'centerRectangle' | 'circle' | 'polygon', a: Vec2, b: Vec2): SketchEntity | null {
    const state = this._state;
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    switch (tool) {
      case 'rectangle': {
        const w = Math.abs(dx);
        const h = Math.abs(dy);
        if (w < 1e-4 || h < 1e-4) return null;
        return { id: newEntityId(), type: 'rectangle', cx: (a[0] + b[0]) / 2, cy: (a[1] + b[1]) / 2, w, h, angle: 0 };
      }
      case 'centerRectangle': {
        const w = Math.abs(dx) * 2;
        const h = Math.abs(dy) * 2;
        if (w < 1e-4 || h < 1e-4) return null;
        return { id: newEntityId(), type: 'rectangle', cx: a[0], cy: a[1], w, h, angle: 0 };
      }
      case 'circle': {
        const r = Math.hypot(dx, dy);
        if (r < 1e-4) return null;
        return { id: newEntityId(), type: 'circle', cx: a[0], cy: a[1], d: r * 2 };
      }
      case 'polygon': {
        const r = Math.hypot(dx, dy);
        if (r < 1e-4) return null;
        return {
          id: newEntityId(),
          type: 'polygon',
          cx: a[0],
          cy: a[1],
          d: r * 2,
          sides: state?.polygonSides ?? 6,
          angle: (Math.atan2(dy, dx) * 180) / Math.PI
        };
      }
    }
  }

  private _entityAt(point: Vec2, outlineOnly = false): SketchEntity | null {
    const state = this._state;
    if (!state) return null;
    const tolerance = SELECT_RADIUS_PX / this._pixelsPerUnit();
    let best: SketchEntity | null = null;
    let bestDistance = tolerance;
    let inside: { entity: SketchEntity; area: number } | null = null;
    for (const entity of state.entities) {
      const path = entityPath(entity);
      const count = path.closed ? path.points.length : path.points.length - 1;
      for (let i = 0; i < count; i += 1) {
        const distance = segmentDistance(point, path.points[i], path.points[(i + 1) % path.points.length]);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = entity;
        }
      }
      if (!outlineOnly && path.closed && pointInPolygon(point, path.points)) {
        const area = Math.abs(polygonArea(path.points));
        if (!inside || area < inside.area) inside = { entity, area };
      }
    }
    return best ?? inside?.entity ?? null;
  }

  private _endAt(point: Vec2): { id: string; end: 0 | 1 } | null {
    const state = this._state;
    if (!state) return null;
    const tolerance = SELECT_RADIUS_PX / this._pixelsPerUnit();
    for (const entity of state.entities) {
      const ends = entityEnds(entity);
      if (!ends) continue;
      for (const k of [0, 1] as const) {
        if (distance2(ends[k], point) <= tolerance) return { id: entity.id, end: k };
      }
    }
    return null;
  }

  private _maybeStartDrag(event: PointerEvent): void {
    const state = this._state;
    if (!state || state.stage !== 'sketch' || state.tool !== 'select') return;
    const raw = this._pointerOnPlane();
    if (!raw) return;
    const end = this._endAt(raw);
    if (end) {
      this._camera.detachControl();
      this._drag = { kind: 'end', id: end.id, end: end.end, original: state.entities, moved: false };
      return;
    }
    const entity = this._entityAt(raw, true);
    if (!entity) {
      this._box = { x0: this._scene.pointerX, y0: this._scene.pointerY, additive: addsToSelection(event), element: null };
      return;
    }
    this._camera.detachControl();
    const selected = state.selectedEntityIds;
    const moving = selected.includes(entity.id) ? selected : addsToSelection(event) ? [...selected, entity.id] : [entity.id];
    const ids = new Set<string>();
    for (const id of moving) {
      const item = state.entities.find((candidate) => candidate.id === id);
      if (!item) continue;
      for (const joined of entityEnds(item) ? connectedIds(state.entities, id) : [id]) ids.add(joined);
    }
    this._drag = { kind: 'entities', ids, start: this._snap(raw, false).point, original: state.entities, moved: false };
    if (moving !== selected) this._emit({ type: 'entitySelected', ids: moving });
  }

  private _primarySelection(): string | null {
    const ids = this._state?.selectedEntityIds ?? [];
    return ids.length === 1 ? ids[0] : null;
  }

  private _updateBox(): void {
    const box = this._box;
    if (!box) return;
    const x = this._scene.pointerX;
    const y = this._scene.pointerY;
    if (!box.element) {
      if (Math.hypot(x - box.x0, y - box.y0) <= CLICK_DRAG_THRESHOLD_PX) return;
      box.element = document.createElement('div');
      box.element.className = 'sketch-box';
      this._layer.appendChild(box.element);
    }
    box.element.classList.toggle('is-crossing', x < box.x0);
    box.element.style.left = `${Math.min(x, box.x0)}px`;
    box.element.style.top = `${Math.min(y, box.y0)}px`;
    box.element.style.width = `${Math.abs(x - box.x0)}px`;
    box.element.style.height = `${Math.abs(y - box.y0)}px`;
  }

  private _endBox(): boolean {
    const box = this._box;
    this._box = null;
    if (!box) return false;
    if (!box.element) return false;
    box.element.remove();
    const state = this._state;
    const frame = this._frame;
    if (!state || !frame) return true;
    const x1 = this._scene.pointerX;
    const y1 = this._scene.pointerY;
    const rect = { left: Math.min(box.x0, x1), right: Math.max(box.x0, x1), top: Math.min(box.y0, y1), bottom: Math.max(box.y0, y1) };
    const crossing = x1 < box.x0;
    const inside = (p: { x: number; y: number }) => p.x >= rect.left && p.x <= rect.right && p.y >= rect.top && p.y <= rect.bottom;
    const hits: string[] = [];
    for (const entity of state.entities) {
      const path = entityPath(entity);
      const points = (path.closed ? closed(path.points) : path.points).map((point) => this._project(sketchToWorld(frame, point)));
      if (points.some((point) => !point)) continue;
      const screen = points as Array<{ x: number; y: number }>;
      const hit = crossing
        ? screen.some(inside) || screen.some((p, i) => i > 0 && segmentCrossesRect(screen[i - 1], p, rect))
        : screen.every(inside);
      if (hit) hits.push(entity.id);
    }
    const ids = box.additive ? [...state.selectedEntityIds, ...hits.filter((id) => !state.selectedEntityIds.includes(id))] : hits;
    this._emit({ type: 'entitySelected', ids });
    return true;
  }

  private _draggedEntities(): SketchEntity[] | null {
    const drag = this._drag;
    const cursor = this._cursor;
    if (!drag || !cursor) return null;
    if (drag.kind === 'end') return moveEnd(drag.original, drag.id, drag.end, cursor.point);
    const dx = cursor.point[0] - drag.start[0];
    const dy = cursor.point[1] - drag.start[1];
    return drag.original.map((entity) => (drag.ids.has(entity.id) ? translateEntity(entity, dx, dy) : entity));
  }

  private _previewDrag(): void {
    this._dirtyRubber = true;
  }

  private _endDrag(): void {
    const drag = this._drag;
    const next = this._draggedEntities();
    this._drag = null;
    this._camera.attachControl(this._canvas, false);
    this._dirtyRubber = true;
    if (!drag || !drag.moved || !next) return;
    this._emit({ type: 'entitiesChanged', entities: next });
  }

  private _drawRubberBand(): void {
    this._rubber?.dispose();
    this._rubber = null;
    this._snapMarker?.dispose();
    this._snapMarker = null;
    const state = this._state;
    const frame = this._frame;
    const cursor = this._cursor;
    if (!state || !frame || state.stage !== 'sketch' || !cursor) return;

    const outlines: Vec2[][] = [];
    const draft = this._draft;
    const at = cursor.point;
    if (this._drag?.moved) {
      const next = this._draggedEntities();
      const ids = this._drag.kind === 'end' ? null : this._drag.ids;
      for (const entity of next ?? []) {
        if (ids && !ids.has(entity.id)) continue;
        if (!ids) {
          const before = this._drag.original.find((item) => item.id === entity.id);
          if (before === entity) continue;
        }
        const path = entityPath(entity);
        outlines.push(path.closed ? closed(path.points) : path.points);
      }
    } else if (draft) {
      switch (draft.tool) {
        case 'line':
        case 'centerline':
          outlines.push([draft.start, at]);
          break;
        case 'slot':
          if (!draft.second) outlines.push([draft.first, at]);
          else {
            const slot = slotFrom(draft.first, draft.second, at);
            if (slot) outlines.push(closed(entityOutline(slot)));
          }
          break;
        case 'arc3':
          if (!draft.second) outlines.push([draft.first, at]);
          else {
            const arc = arcThrough(draft.first, draft.second, at, 'rubber');
            if (arc) outlines.push(entityOutline(arc));
            else outlines.push([draft.first, draft.second]);
          }
          break;
        case 'centerArc':
          if (!draft.start) outlines.push([draft.center, at]);
          else {
            const arc = arcFromCenter(draft.center, draft.start, at, draft.sweep < 0, 'rubber');
            if (arc) outlines.push(entityOutline(arc));
            outlines.push([draft.center, at]);
          }
          break;
        default: {
          const shape = this._shapeFrom(draft.tool, draft.first, at);
          if (shape) outlines.push(closed(entityOutline(shape)));
          if (draft.tool === 'circle' || draft.tool === 'polygon') outlines.push([draft.first, at]);
        }
      }
    }
    if (outlines.length) {
      const lines = outlines.map((outline) => outline.map((point) => toVector(sketchToWorld(frame, point))));
      this._rubber = createFatLines('sketch-rubber', lines, RUBBER_LINE, 1.75, this._scene);
      if (this._rubber) this._rubber.renderingGroupId = 1;
    }

    if (state.tool !== 'select' && state.tool !== 'convert') {
      const size = 5 / this._pixelsPerUnit();
      const color = cursor.kind === 'grid' || cursor.kind === 'free' ? RUBBER_LINE : SELECTED_LINE;
      const marker: Vec2[][] =
        cursor.kind === 'point'
          ? [closed([[at[0] - size, at[1] - size], [at[0] + size, at[1] - size], [at[0] + size, at[1] + size], [at[0] - size, at[1] + size]])]
          : [
              [[at[0] - size, at[1]], [at[0] + size, at[1]]],
              [[at[0], at[1] - size], [at[0], at[1] + size]]
            ];
      const lines = marker.map((line) => line.map((point) => toVector(sketchToWorld(frame, point))));
      this._snapMarker = createFatLines('sketch-snap', lines, color, 1.75, this._scene);
      if (this._snapMarker) this._snapMarker.renderingGroupId = 1;
    }
  }

  private _numericPrompt(): string | null {
    switch (this._draft?.tool) {
      case 'line':
      case 'centerline':
        return 'Length';
      case 'circle':
      case 'polygon':
        return 'Ø';
      case 'rectangle':
      case 'centerRectangle':
        return 'W, H';
      default:
        return null;
    }
  }

  private _openNumeric(typed: string): void {
    const prompt = this._numericPrompt();
    const cursor = this._cursor;
    const frame = this._frame;
    if (!prompt || !cursor || !frame) return;
    this._closeNumeric();
    const box = document.createElement('div');
    box.className = 'sketch-numeric';
    const label = document.createElement('span');
    label.className = 'sketch-numeric-label';
    label.textContent = prompt;
    const input = document.createElement('input');
    input.className = 'sketch-numeric-input';
    input.type = 'text';
    input.inputMode = 'decimal';
    input.value = typed;
    input.placeholder = prompt === 'W, H' ? 'width, height' : 'inches';
    box.append(label, input);
    const at = this._project(sketchToWorld(frame, cursor.point));
    box.style.left = `${(at?.x ?? 0) + 18}px`;
    box.style.top = `${(at?.y ?? 0) + 18}px`;
    this._layer.appendChild(box);
    this._numeric = { box, input };
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
    input.addEventListener('keydown', (event) => {
      event.stopPropagation();
      if (event.key === 'Enter') {
        event.preventDefault();
        this._commitNumeric();
      } else if (event.key === 'Escape') {
        event.preventDefault();
        this._closeNumeric();
      } else if (event.key === 'Tab' && prompt === 'W, H') {
        event.preventDefault();
        if (!input.value.includes(',')) input.value = `${input.value.trim()}, `;
      }
    });
    input.addEventListener('input', () => box.classList.remove('is-invalid'));
    input.addEventListener('blur', () => {
      if (this._numeric?.input === input) this._closeNumeric();
    });
    input.addEventListener('pointerdown', (event) => event.stopPropagation());
  }

  private _closeNumeric(): void {
    const numeric = this._numeric;
    if (!numeric) return;
    this._numeric = null;
    numeric.box.remove();
  }

  private _commitNumeric(): void {
    const numeric = this._numeric;
    const draft = this._draft;
    if (!numeric || !draft) {
      this._closeNumeric();
      return;
    }
    const parts = numeric.input.value
      .split(/\s*[,x×]\s*/)
      .filter((part) => part.trim().length > 0)
      .map((part) => evaluateLength(part));
    if (!parts.length || parts.some((value) => value === null || !(value > 1e-4))) {
      numeric.box.classList.add('is-invalid');
      return;
    }
    const [a, b = a] = parts as number[];
    const at = this._cursor?.point ?? null;
    const toward = (from: Vec2): Vec2 => {
      const dx = (at?.[0] ?? from[0] + 1) - from[0];
      const dy = (at?.[1] ?? from[1]) - from[1];
      const length = Math.hypot(dx, dy);
      return length > 1e-9 ? [dx / length, dy / length] : [1, 0];
    };
    const sign = (from: number, to: number | undefined) => (to !== undefined && to < from ? -1 : 1);
    let point: Vec2 | null = null;
    switch (draft.tool) {
      case 'line':
      case 'centerline': {
        const u = toward(draft.start);
        point = [draft.start[0] + u[0] * a, draft.start[1] + u[1] * a];
        break;
      }
      case 'circle':
      case 'polygon': {
        const u = toward(draft.first);
        point = [draft.first[0] + (u[0] * a) / 2, draft.first[1] + (u[1] * a) / 2];
        break;
      }
      case 'rectangle':
      case 'centerRectangle': {
        const half = draft.tool === 'centerRectangle' ? 0.5 : 1;
        point = [
          draft.first[0] + sign(draft.first[0], at?.[0]) * a * half,
          draft.first[1] + sign(draft.first[1], at?.[1]) * b * half
        ];
        break;
      }
      default:
        break;
    }
    this._closeNumeric();
    if (!point) return;
    const kind = this._cursor?.kind;
    this._applyPoint(point, kind === 'horizontal' || kind === 'vertical' ? kind : 'free');
  }

  private _handleKey(event: KeyboardEvent): void {
    const state = this._state;
    if (!state) return;
    const target = event.target as HTMLElement | null;
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable)) {
      return;
    }
    if (state.stage === 'measure' && event.key === 'Escape') {
      this._emit({ type: 'measurePicked', pick: null });
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    if (state.stage !== 'sketch') return;
    if (event.key === 'Escape') {
      if (this._draft) {
        this._draft = null;
        this._dirtyRubber = true;
      } else if (state.tool !== 'select') {
        this._emit({ type: 'toolChanged', tool: 'select' });
      } else if (state.selectedEntityIds.length) {
        this._emit({ type: 'entitySelected', ids: [] });
      }
      event.preventDefault();
      event.stopPropagation();
    } else if (event.key === 'Enter' && (this._draft?.tool === 'line' || this._draft?.tool === 'centerline')) {
      this._draft = null;
      this._dirtyRubber = true;
      event.preventDefault();
    } else if ((event.key === 'Delete' || event.key === 'Backspace') && state.selectedEntityIds.length && !this._draft) {
      this._emit({ type: 'entitiesDeleted', ids: state.selectedEntityIds });
      event.preventDefault();
    } else if (event.key.toLowerCase() === 'a' && (event.metaKey || event.ctrlKey) && state.tool === 'select') {
      this._emit({ type: 'entitySelected', ids: state.entities.map((entity) => entity.id) });
      event.preventDefault();
    } else if (/^[0-9.]$/.test(event.key) && !event.metaKey && !event.ctrlKey && !event.altKey) {
      if (this._numericPrompt()) {
        this._openNumeric(event.key);
        event.preventDefault();
      } else if (!this._draft && this._primarySelection()) {
        const primary = this._primarySelection();
        const own = this._dimLabels.filter((entry) => entry.spec?.entityId === primary);
        const label = own.find((entry) => entry.spec?.key !== 'cx' && entry.spec?.key !== 'cy') ?? own[0];
        if (label?.spec) {
          this._beginEdit(label.spec, label.element, event.key);
          event.preventDefault();
        }
      }
    } else if (!event.metaKey && !event.ctrlKey && !event.altKey) {
      const tool = SHORTCUTS[event.key.toLowerCase()];
      if (tool) {
        this._emit({ type: 'toolChanged', tool });
        event.preventDefault();
      }
    }
  }
}

function addsToSelection(event: PointerEvent): boolean {
  return event.ctrlKey || event.metaKey;
}

function segmentCrossesRect(
  a: { x: number; y: number },
  b: { x: number; y: number },
  rect: { left: number; right: number; top: number; bottom: number }
): boolean {
  let t0 = 0;
  let t1 = 1;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const edges: Array<[number, number]> = [
    [-dx, a.x - rect.left],
    [dx, rect.right - a.x],
    [-dy, a.y - rect.top],
    [dy, rect.bottom - a.y]
  ];
  for (const [p, q] of edges) {
    if (Math.abs(p) < 1e-12) {
      if (q < 0) return false;
      continue;
    }
    const t = q / p;
    if (p < 0) t0 = Math.max(t0, t);
    else t1 = Math.min(t1, t);
    if (t0 > t1) return false;
  }
  return true;
}

function isDrawingTool(tool: SketchTool): boolean {
  return tool !== 'select' && tool !== 'convert';
}

const SHORTCUTS: Record<string, SketchTool> = {
  l: 'line',
  r: 'rectangle',
  c: 'circle',
  a: 'arc3'
};

function flatMaterial(name: string, color: Color3, alpha: number, scene: Scene): StandardMaterial {
  const material = new StandardMaterial(name, scene);
  material.disableLighting = true;
  material.emissiveColor = color;
  material.alpha = alpha;
  material.backFaceCulling = false;
  return material;
}

function planeKey(plane: SketchPlane): string {
  const normal = plane.kind === 'standard' ? plane.id : plane.normal.map((value) => value.toFixed(5)).join(',');
  return `${normal}@${plane.offset.toFixed(5)}`;
}

function measureKey(pick: MeasurePick): string {
  switch (pick.kind) {
    case 'vertex':
    case 'point':
      return `${pick.kind}:${pick.point.map((v) => v.toFixed(4)).join(',')}`;
    case 'face':
      return `face:${pick.centroid.map((v) => v.toFixed(4)).join(',')}:${pick.area.toFixed(4)}`;
    case 'line':
      return `line:${[...pick.a, ...pick.b].map((v) => v.toFixed(4)).join(',')}`;
    case 'circle':
      return `circle:${pick.center.map((v) => v.toFixed(4)).join(',')}:${pick.radius.toFixed(4)}`;
  }
}

function toVector(point: Vec3): Vector3 {
  return new Vector3(point[0], point[1], point[2]);
}

function closed(points: Vec2[]): Vec2[] {
  return points.length ? [...points, points[0]] : points;
}

function dashes(points: Vec2[], on: number, off: number): Vec2[][] {
  const out: Vec2[][] = [];
  let current: Vec2[] = [points[0]];
  let drawing = true;
  let left = on;
  for (let i = 1; i < points.length; i += 1) {
    let a = points[i - 1];
    const b = points[i];
    let segment = distance2(a, b);
    while (segment > 1e-12) {
      const step = Math.min(left, segment);
      const t = step / segment;
      const p: Vec2 = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
      if (drawing) current.push(p);
      left -= step;
      segment -= step;
      a = p;
      if (left <= 1e-12) {
        if (drawing && current.length >= 2) out.push(current);
        drawing = !drawing;
        left = drawing ? on : off;
        current = [p];
      }
    }
  }
  if (drawing && current.length >= 2) out.push(current);
  return out;
}

export function formatInches(value: number): string {
  const fixed = value.toFixed(3);
  return fixed.endsWith('0') ? fixed.slice(0, -1) : fixed;
}

export function evaluateLength(text: string): number | null {
  let source = text.trim().toLowerCase();
  let factor = 1;
  const unit = /(mm|cm|in|")\s*$/.exec(source);
  if (unit) {
    factor = unit[1] === 'mm' ? 1 / 25.4 : unit[1] === 'cm' ? 1 / 2.54 : 1;
    source = source.slice(0, unit.index).trim();
  }
  const value = parseArithmetic(source);
  return value !== null && Number.isFinite(value) ? value * factor : null;
}

function parseArithmetic(source: string): number | null {
  const tokens = source.match(/\d*\.?\d+(?:e[-+]?\d+)?|[-+*/()]|\S/g) ?? [];
  let at = 0;
  const peek = () => tokens[at];
  const expression = (): number | null => {
    let left = term();
    while (left !== null && (peek() === '+' || peek() === '-')) {
      const op = tokens[at++];
      const right = term();
      if (right === null) return null;
      left = op === '+' ? left + right : left - right;
    }
    return left;
  };
  const term = (): number | null => {
    let left = factor();
    while (left !== null && (peek() === '*' || peek() === '/')) {
      const op = tokens[at++];
      const right = factor();
      if (right === null) return null;
      left = op === '*' ? left * right : left / right;
    }
    return left;
  };
  const factor = (): number | null => {
    const token = tokens[at++];
    if (token === undefined) return null;
    if (token === '-') {
      const value = factor();
      return value === null ? null : -value;
    }
    if (token === '+') return factor();
    if (token === '(') {
      const value = expression();
      if (tokens[at++] !== ')') return null;
      return value;
    }
    const number = Number(token);
    return Number.isFinite(number) ? number : null;
  };
  const result = expression();
  return at === tokens.length ? result : null;
}

function slotFrom(a: Vec2, b: Vec2, c: Vec2): SketchEntity | null {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const length = Math.hypot(dx, dy);
  if (length < 1e-6) return null;
  const offset = Math.abs((c[0] - a[0]) * dy - (c[1] - a[1]) * dx) / length;
  const width = offset * 2;
  if (width < 1e-4) return null;
  return {
    id: newEntityId(),
    type: 'slot',
    cx: (a[0] + b[0]) / 2,
    cy: (a[1] + b[1]) / 2,
    length,
    width,
    angle: (Math.atan2(dy, dx) * 180) / Math.PI
  };
}

function segmentDistance(p: Vec2, a: Vec2, b: Vec2): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSq = dx * dx + dy * dy;
  const t = lengthSq > 0 ? Math.min(Math.max(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / lengthSq, 0), 1) : 0;
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

function pointInPolygon(p: Vec2, polygon: Vec2[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i];
    const b = polygon[j];
    if (a[1] > p[1] !== b[1] > p[1] && p[0] < ((b[0] - a[0]) * (p[1] - a[1])) / (b[1] - a[1]) + a[0]) {
      inside = !inside;
    }
  }
  return inside;
}

function polygonArea(points: Vec2[]): number {
  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    area += a[0] * b[1] - b[0] * a[1];
  }
  return area / 2;
}

function sketchCentroid(entities: SketchEntity[]): Vec2 {
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const entity of entities) {
    if (entity.construction) continue;
    for (const point of entityOutline(entity)) {
      sx += point[0];
      sy += point[1];
      n += 1;
    }
  }
  return n ? [sx / n, sy / n] : [0, 0];
}
