import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState
} from 'react';
import { PointerEventTypes } from '@babylonjs/core/Events/pointerEvents';
import { Color3 } from '@babylonjs/core/Maths/math.color';
import { Plane } from '@babylonjs/core/Maths/math.plane';
import { Vector3 } from '@babylonjs/core/Maths/math.vector';
import { Tools } from '@babylonjs/core/Misc/tools';
import type { PickingInfo } from '@babylonjs/core/Collisions/pickingInfo';
import type { PBRMaterial } from '@babylonjs/core/Materials/PBR/pbrMaterial';
import type { AbstractMesh } from '@babylonjs/core/Meshes/abstractMesh';
import type { Mesh } from '@babylonjs/core/Meshes/mesh';
import type { Scene } from '@babylonjs/core/scene';
import '@babylonjs/core/Misc/screenshotTools';

import { sampleColormap } from '../lib/colormaps';
import { fetchEnvelope, isAbortError, type EnvelopeData } from '../lib/gridApi';
import {
  RESULT_MODE_IDS,
  RESULT_MODE_UNITS,
  isSweepMode,
  isTimeMode,
  type GridMeta,
  type ResultModeId,
  type ResultViewState,
  type ViewportHandle,
  type ViewportProps
} from '../lib/viewState';
import type { ProjectPart } from '../lib/api';
import { partBoxWorldBounds } from '../lib/orientation';
import { PART_TYPES } from '../lib/partTypes';
import { Annotations } from './annotations';
import { drawAxesGizmo } from './axesGizmo';
import { CameraController, unionBounds, type Bounds } from './cameraController';
import { ContourClient, type ContourGeometry } from './contourClient';
import { EnvelopeMesh } from './envelopeMesh';
import { RunResources, isFrameMode } from './fieldResources';
import {
  createPlaceholderLut,
  createPlaceholderVolume,
  FieldPaintPluginMaterial
} from './fieldPaintPlugin';
import { createPourMarker } from './markers';
import { applyPartAppearance, setPartClipPlanes, syncPartMeshes, type PartEntry } from './partMeshes';
import { PartingPlane } from './partingPlane';
import { planarFace, type PlanarFace } from './modeling/planarFace';
import { pickTopology } from './modeling/topologyPick';
import type { MeasurePick } from '../lib/modeling/measure';
import { SketchOverlay, type SurfacePick } from './modeling/sketchOverlay';
import { pickVisibleAll } from './picking';
import { SECTION_AXES, SectionCaps, isSectionCap, type SectionAxis } from './sectionCaps';
import {
  createOriginAxes,
  createViewportScene,
  enableAmbientOcclusion,
  type ViewportScene
} from './sceneSetup';

const CLICK_DRAG_THRESHOLD_PX = 4;

const DOUBLE_CLICK_MS = 300;
const DOUBLE_CLICK_RADIUS_PX = 6;

const MIN_MEASURE_DISTANCE_IN = 1e-4;

const MEASURE_SECOND_POINT_HINT = 'Click the second point (Esc to cancel)';

const HOVER_PICK_DUTY = 0.25;

function cursorFor(tool: ViewportProps['tool'] | 'orientPick', overPart: boolean, overSurface: boolean): string {
  if (tool === 'select') return overPart ? 'pointer' : 'default';
  return overSurface ? 'crosshair' : 'default';
}

const PLAYBACK_PERCENT_PER_SECOND = 12;

const ENVELOPE_DEBOUNCE_MS = 150;

const ENVELOPE_SHELL_OPACITY = 0.35;
const BOTH_SHELL_OPACITY = 0.35;

const MOLTEN_GLOW = 0.62;

const FRONT_BLUR_STRENGTH = 0.3;

const SWEEP_FRONT_SPAN = 0.15;

const LIQUID_SKIN_BAND = 0.12;

const LIQUID_COLOR: [number, number, number] = (() => {
  const [r, g, b] = sampleColormap('molten', 1);
  const toLinear = (channel: number) =>
    channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  return [toLinear(r / 255), toLinear(g / 255), toLinear(b / 255)];
})();

const AO_MINIMUM_FPS = 55;

const METAL_COLOR = new Color3(0.78, 0.8, 0.83);

type SceneRefs = {
  viewport: ViewportScene;
  camera: CameraController;
  placeholderVolume: ReturnType<typeof createPlaceholderVolume>;
  placeholderLut: ReturnType<typeof createPlaceholderLut>;
  originAxes: Mesh;
  caps: SectionCaps;
  envelope: EnvelopeMesh;
  annotations: Annotations;
  sketch: SketchOverlay;
  parting: PartingPlane;
};

function geometrySignature(parts: ProjectPart[]): string {
  return parts
    .map(
      (part) =>
        `${part.id}:${part.filepath}:${part.meshData?.vertex_count ?? 0}:` +
        `${part.position_x},${part.position_y},${part.position_z}:` +
        `${part.rotation_x},${part.rotation_y},${part.rotation_z}:${part.scale}`
    )
    .join('|');
}

function membershipSignature(parts: ProjectPart[]): string {
  return parts
    .map((part) => `${part.id}:${part.filepath}:${part.meshData?.vertex_count ?? 0}`)
    .join('|');
}

function floorZFor(bounds: Bounds, orientActive: boolean): number {
  return orientActive ? 0 : bounds.min.z - bounds.size * 0.02;
}

function triangleWorldNormal(mesh: AbstractMesh, faceId: number): Vector3 | null {
  const indices = mesh.getIndices();
  const positions = mesh.getVerticesData('position');
  if (!indices || !positions || faceId < 0) return null;
  const corner = (k: number) => {
    const index = indices[faceId * 3 + k] * 3;
    return new Vector3(positions[index], positions[index + 1], positions[index + 2]);
  };
  const a = corner(0);
  const local = Vector3.Cross(corner(1).subtract(a), corner(2).subtract(a));
  if (local.lengthSquared() === 0) return null;
  return Vector3.TransformNormal(local, mesh.getWorldMatrix()).normalize();
}

function partBounds(parts: ProjectPart[]): Bounds | null {
  const boxes = parts
    .map((part) => partBoxWorldBounds(part))
    .filter((box): box is NonNullable<typeof box> => Boolean(box))
    .map(([min, max]) => ({
      x: [min[0], max[0]] as [number, number],
      y: [min[1], max[1]] as [number, number],
      z: [min[2], max[2]] as [number, number]
    }));
  return unionBounds(boxes);
}

function gridBounds(meta: GridMeta | null): Bounds | null {
  if (!meta) return null;
  const [nx, ny, nz] = meta.dims;
  const spacing = meta.spacing;
  const min = new Vector3(
    meta.origin[0] - spacing * 0.5,
    meta.origin[1] - spacing * 0.5,
    meta.origin[2] - spacing * 0.5
  );
  const max = new Vector3(
    meta.origin[0] + spacing * (nx - 0.5),
    meta.origin[1] + spacing * (ny - 0.5),
    meta.origin[2] + spacing * (nz - 0.5)
  );
  const span = max.subtract(min);
  return {
    min,
    max,
    center: min.add(max).scale(0.5),
    size: Math.max(span.x, span.y, span.z, 1)
  };
}

function sectionPlanes(
  section: ViewportProps['view']['section'],
  bounds: Bounds | null
): Record<SectionAxis, Plane | null> {
  const planes: Record<SectionAxis, Plane | null> = { x: null, y: null, z: null };
  if (!bounds) return planes;

  SECTION_AXES.forEach((axis, index) => {
    const state = section[axis];
    if (!state.enabled) return;
    const min = bounds.min.asArray()[index];
    const max = bounds.max.asArray()[index];
    const position = min + (max - min) * (Math.min(Math.max(state.position, 0), 100) / 100);
    const normal = new Vector3(axis === 'x' ? 1 : 0, axis === 'y' ? 1 : 0, axis === 'z' ? 1 : 0);
    const direction = state.flip ? normal.scale(-1) : normal;
    planes[axis] = Plane.FromPositionAndNormal(
      new Vector3(axis === 'x' ? position : 0, axis === 'y' ? position : 0, axis === 'z' ? position : 0),
      direction
    );
  });
  return planes;
}

function thresholdInFieldUnits(view: ResultViewState, meta: GridMeta | null): number {
  const range = meta?.ranges[view.mode];
  const min = range?.min ?? 0;
  const max = range?.max ?? 1;
  return min + (Math.min(Math.max(view.threshold, 0), 100) / 100) * (max - min);
}

function wantsEnvelope(style: ResultViewState['style']): boolean {
  return style === 'envelope' || style === 'both';
}

export const ResultsViewport = forwardRef<ViewportHandle, ViewportProps>(function ResultsViewport(props, ref) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const gizmoRef = useRef<HTMLCanvasElement | null>(null);
  const sketchLayerRef = useRef<HTMLDivElement | null>(null);

  const [resourcesEpoch, setResourcesEpoch] = useState(0);

  const sceneRef = useRef<SceneRefs | null>(null);
  const partEntriesRef = useRef<Map<string, PartEntry>>(new Map());
  const resourcesRef = useRef<RunResources | null>(null);
  const boundsRef = useRef<Bounds | null>(null);
  const pourMarkerRef = useRef<Mesh | null>(null);
  const measurePendingRef = useRef<Vector3 | null>(null);
  const measureHintRef = useRef(false);
  const refreshHoverRef = useRef<(() => void) | null>(null);
  const contourRef = useRef<ContourClient | null>(null);
  const teardownRef = useRef<{ timer: number } | null>(null);
  const envelopeCacheRef = useRef<Map<string, EnvelopeData>>(new Map());
  const envelopeKeyRef = useRef<string | null>(null);

  const propsRef = useRef(props);
  propsRef.current = props;

  const clearMeasureHint = useCallback(() => {
    if (!measureHintRef.current) return;
    measureHintRef.current = false;
    propsRef.current.onStatus(null);
  }, []);

  const { projectId, resultsAvailable, resultsVersion, view, ortho, parts, selectedPartId, tool } = props;
  const modelingActive = Boolean(props.modeling);

  const paint = useCallback(() => {
    const refs = sceneRef.current;
    if (!refs) return;
    const current = propsRef.current;
    const state = current.view;
    const resources = resourcesRef.current;
    const entries = partEntriesRef.current;

    const mode = state.mode;
    const stack = resources && isFrameMode(mode) ? resources.stack(mode) : null;
    const field = resources && !isFrameMode(mode) ? resources.field(mode) : null;
    const ready = Boolean(resources) && (Boolean(stack) || Boolean(field));

    let blend = 0;
    if (stack) blend = stack.seek(state.timePercent / 100);

    const range = resources?.meta.ranges[mode];
    const quantMin = stack ? 0 : (field?.min ?? range?.min ?? 0);
    const quantMax = stack ? 1 : (field?.max ?? range?.max ?? 1);
    const scaleMin = state.scaleMin ?? quantMin;
    const scaleMax = state.scaleMax ?? quantMax;

    const planes = sectionPlanes(state.section, boundsRef.current);
    const showEnvelope = wantsEnvelope(state.style) && state.showResults;

    const sweeping = isSweepMode(mode) && ready;

    const liquidCutoff = sweeping
      ? (state.timePercent / 100) * quantMax
      : isFrameMode(mode) && ready && wantsEnvelope(state.style)
        ? 0.5
        : null;

    const liquidBody = liquidCutoff !== null && wantsEnvelope(state.style) && state.showResults;

    const quantSpan = quantMax - quantMin;
    const sweepIso =
      sweeping && liquidCutoff !== null && quantSpan > 1e-9
        ? Math.min(1, Math.max(0, (liquidCutoff - quantMin) / quantSpan))
        : 0;

    const ghostShell = state.style === 'envelope';
    const paintSurface =
      state.showResults && (state.style !== 'envelope' || isFrameMode(state.mode) || sweeping);

    const applyPaint = (plugin: FieldPaintPluginMaterial, strength: number) => {
      if (resources && ready) {
        const meta = resources.meta;
        plugin.state.origin.set(meta.origin[0], meta.origin[1], meta.origin[2]);
        plugin.state.spacing = meta.spacing;
        plugin.state.dims = resources.dims;
        plugin.state.quantMin = quantMin;
        plugin.state.quantMax = quantMax;
        plugin.state.scaleMin = scaleMin;
        plugin.state.scaleMax = scaleMax;
        plugin.state.frameBlend = blend;
        plugin.state.castOnly = state.castOnly;
        plugin.state.strength = strength;
        plugin.state.sweepTime = liquidCutoff;
        plugin.state.liquidBand = liquidBody ? LIQUID_SKIN_BAND * quantSpan : 0;
        plugin.state.liquidColor = LIQUID_COLOR;
        plugin.state.liquidGlow = MOLTEN_GLOW;
        plugin.state.liquidBody = liquidBody;
        plugin.state.solidAlpha = ENVELOPE_SHELL_OPACITY;
        plugin.state.skinDepth = meta.spacing;
        plugin.state.ghostSolid = ghostShell;
        plugin.state.skinClip = false;
        plugin.skinTexture = resources.skinTexture;
        plugin.fieldTextureA = stack ? stack.textureA : (field?.texture ?? null);
        plugin.fieldTextureB = stack ? stack.textureB : (field?.texture ?? null);
        plugin.maskTexture = resources.maskTexture;
        plugin.lutTexture = resources.lut(state.colormap);
      } else {
        plugin.state.strength = 0;
        plugin.state.sweepTime = null;
        plugin.state.liquidBand = 0;
        plugin.state.liquidBody = false;
        plugin.state.ghostSolid = false;
        plugin.state.skinClip = false;
      }
    };

    for (const entry of entries.values()) {
      applyPaint(entry.plugin, paintSurface ? 1 : 0);
      setPartClipPlanes(entry, planes.x, planes.y, planes.z);
    }

    refs.caps.layout(state.section, state.section.caps, boundsRef.current, planes);
    refs.caps.orient(state.section);
    for (const cap of refs.caps.entries()) {
      applyPaint(cap.plugin, state.showResults && ready ? 1 : 0);
      cap.plugin.state.discardOutside = true;
      cap.plugin.state.liquidBody = false;
      cap.plugin.state.ghostSolid = false;
    }

    refs.envelope.material.clipPlane = planes.x;
    refs.envelope.material.clipPlane2 = planes.y;
    refs.envelope.material.clipPlane3 = planes.z;
    const molten = isFrameMode(mode) || sweeping;

    applyPaint(refs.envelope.clip, 0);
    refs.envelope.clip.state.liquidBody = false;
    refs.envelope.clip.state.ghostSolid = false;
    refs.envelope.clip.state.discardOutside = molten;
    refs.envelope.clip.state.discardUnpaintable = molten && state.castOnly;
    refs.envelope.clip.state.skinClip = resources?.skinTexture != null;
    refs.envelope.setStyle({
      colormap: molten ? 'molten' : state.colormap,
      scaleMin: molten ? (sweeping ? sweepIso : 0) : scaleMin,
      scaleMax: molten ? (sweeping ? Math.min(1, sweepIso + SWEEP_FRONT_SPAN) : 1) : scaleMax,
      flatColor: METAL_COLOR,
      alpha: 1,
      metallic: molten ? 0.1 : 0.0,
      roughness: molten ? 0.55 : 0.65,
      glow: molten ? MOLTEN_GLOW : 0
    });
    refs.viewport.pipeline.bloomEnabled = molten && showEnvelope;
    refs.envelope.setVisible(showEnvelope);
  }, []);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    if (teardownRef.current) {
      window.clearTimeout(teardownRef.current.timer);
      teardownRef.current = null;
      return;
    }

    const viewport = createViewportScene(canvas);
    const camera = new CameraController(viewport.camera, viewport.engine, viewport.scene);
    const placeholderVolume = createPlaceholderVolume(viewport.scene);
    const placeholderLut = createPlaceholderLut(viewport.scene);
    const originAxes = createOriginAxes(viewport.scene);
    const makePlugin = (material: PBRMaterial) =>
      new FieldPaintPluginMaterial(material, placeholderVolume, placeholderLut);
    const caps = new SectionCaps(viewport.scene, makePlugin);
    const envelope = new EnvelopeMesh(viewport.scene, 'result-envelope', makePlugin);
    const annotations = new Annotations(viewport.scene, 10);
    const parting = new PartingPlane(viewport.scene);
    let lastFace: { mesh: AbstractMesh; face: PlanarFace } | null = null;
    const pickSurface = (x: number, y: number): SurfacePick | null => {
      const [hit] = pickVisibleAll(viewport.scene, x, y, { predicate: isPartMesh });
      if (!hit?.pickedPoint || !hit.pickedMesh) return null;
      const mesh = hit.pickedMesh;
      if (!lastFace || lastFace.mesh !== mesh || !lastFace.face.ids.has(hit.faceId)) {
        const face = planarFace(mesh, hit.faceId);
        lastFace = face ? { mesh, face } : null;
      }
      const indices = mesh.getIndices();
      const positions = mesh.getVerticesData('position');
      if (!indices || !positions || hit.faceId < 0) return null;
      const corner = (k: number) => {
        const index = indices[hit.faceId * 3 + k] * 3;
        return new Vector3(positions[index], positions[index + 1], positions[index + 2]);
      };
      const a = corner(0);
      const local = Vector3.Cross(corner(1).subtract(a), corner(2).subtract(a));
      const normal = Vector3.TransformNormal(local, mesh.getWorldMatrix()).normalize();
      const partId = partIdOf(mesh);
      const partName = propsRef.current.parts.find((part) => part.id === partId)?.name ?? null;
      return {
        point: hit.pickedPoint.clone(),
        normal,
        distance: hit.distance,
        partName,
        face: lastFace?.face ?? null
      };
    };
    const pickMeasure = (x: number, y: number): MeasurePick | null => {
      const surface = pickSurface(x, y);
      const [hit] = pickVisibleAll(viewport.scene, x, y, { predicate: isPartMesh });
      if (!surface || !hit?.pickedMesh) return null;
      const partId = partIdOf(hit.pickedMesh);
      return pickTopology({
        scene: viewport.scene,
        camera: viewport.camera,
        x,
        y,
        mesh: hit.pickedMesh,
        faceId: hit.faceId,
        hitPoint: surface.point,
        hitNormal: surface.normal,
        edges: partId ? (partEntriesRef.current.get(partId)?.edges ?? null) : null,
        face: surface.face,
        partName: surface.partName,
        assemblySize: boundsRef.current?.size ?? 10
      });
    };
    const sketchLayer = sketchLayerRef.current;
    const sketch = new SketchOverlay({
      scene: viewport.scene,
      camera: viewport.camera,
      canvas,
      layer: sketchLayer ?? container.appendChild(document.createElement('div')),
      pickSurface,
      pickMeasure,
      emit: (event) => propsRef.current.onModelingEvent?.(event)
    });
    sceneRef.current = {
      viewport,
      camera,
      placeholderVolume,
      placeholderLut,
      originAxes,
      caps,
      envelope,
      annotations,
      sketch,
      parting
    };

    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      (window as unknown as { __viewportDebug?: unknown }).__viewportDebug = {
        engine: viewport.engine,
        scene: viewport.scene,
        parts: partEntriesRef.current,
        contour: () => contourRef.current,
        resources: () => resourcesRef.current,
        pick: (x: number, y: number) => pickModel(viewport.scene, x, y)
      };
    }

    if (viewport.engine.webGLVersion < 2) {
      propsRef.current.onError(
        'This browser reports WebGL 1 only. Result fields need WebGL 2 and will not be painted.'
      );
    }

    let pointerDownAt: { x: number; y: number; button: number } | null = null;
    let lastClick: { x: number; y: number; time: number } | null = null;
    let doubleClicked = false;
    const pointerObserver = viewport.scene.onPointerObservable.add((info) => {
      const scene = viewport.scene;
      const button = info.event.button;
      if (info.type === PointerEventTypes.POINTERDOWN) {
        pointerDownAt = { x: scene.pointerX, y: scene.pointerY, button };
        return;
      }
      if (info.type === PointerEventTypes.POINTERDOUBLETAP) {
        const wasDoubleClick = doubleClicked;
        doubleClicked = false;
        if (button !== 0 || !wasDoubleClick || propsRef.current.tool !== 'select') return;
        if (propsRef.current.orient?.pick) return;
        if (propsRef.current.modeling) return;
        const pick = pickModel(scene);
        if (pick) camera.focusOn(pick.point);
        return;
      }
      if (info.type !== PointerEventTypes.POINTERUP) return;

      const downAt = pointerDownAt;
      pointerDownAt = null;
      doubleClicked = false;
      if (button !== 0 || (downAt && downAt.button !== 0)) return;
      if (downAt) {
        const travelled = Math.hypot(scene.pointerX - downAt.x, scene.pointerY - downAt.y);
        if (travelled > CLICK_DRAG_THRESHOLD_PX) return;
      }

      const now = performance.now();
      const previous = lastClick;
      if (
        previous &&
        now - previous.time < DOUBLE_CLICK_MS &&
        Math.hypot(scene.pointerX - previous.x, scene.pointerY - previous.y) <= DOUBLE_CLICK_RADIUS_PX
      ) {
        lastClick = null;
        doubleClicked = true;
        return;
      }
      lastClick = { x: scene.pointerX, y: scene.pointerY, time: now };
      handleClick();
    });

    const capDrawnAt = (point: Vector3): boolean => {
      const resources = resourcesRef.current;
      if (!resources) return false;
      const cell = resources.cellIndexAt(point.x, point.y, point.z);
      return cell >= 0 && resources.mask[cell] !== 0;
    };

    const pickModel = (
      scene: Scene,
      x = scene.pointerX,
      y = scene.pointerY
    ): { hit: PickingInfo; point: Vector3; partId: string | null; onCap: boolean } | null => {
      const [hit] = pickVisibleAll(scene, x, y, {
        predicate: (mesh) => isPartMesh(mesh) || isSectionCap(mesh),
        accept: (candidate) =>
          !isSectionCap(candidate.pickedMesh) ||
          (candidate.pickedPoint !== null && capDrawnAt(candidate.pickedPoint))
      });
      if (!hit?.pickedPoint) return null;
      if (!isSectionCap(hit.pickedMesh)) {
        return { hit, point: hit.pickedPoint, partId: partIdOf(hit.pickedMesh), onCap: false };
      }
      const owner = pickVisibleAll(scene, x, y, {
        predicate: isPartMesh,
        cullBackFaces: false
      }).find((candidate) => candidate.distance >= hit.distance);
      return { hit, point: hit.pickedPoint, partId: partIdOf(owner?.pickedMesh), onCap: true };
    };

    const handleClick = () => {
      const scene = viewport.scene;
      const current = propsRef.current;
      if (current.modeling) return;
      const pick = pickModel(scene);
      const partId = pick?.partId ?? null;

      const orientMode = current.orient?.pick ?? null;
      if (orientMode) {
        if (!pick) return;
        const pickedMesh = pick.hit.pickedMesh;
        const normal = pick.onCap || !pickedMesh ? null : triangleWorldNormal(pickedMesh, pick.hit.faceId);
        current.onOrientPick?.({
          mode: orientMode,
          partId,
          point: [pick.point.x, pick.point.y, pick.point.z],
          normal: normal ? [normal.x, normal.y, normal.z] : null
        });
        return;
      }

      if (current.tool === 'pour') {
        if (!pick) return;
        const normal = pick.hit.getNormal(true, true);
        current.onPourPicked({
          point: [pick.point.x, pick.point.y, pick.point.z],
          normal: normal ? [normal.x, normal.y, normal.z] : null,
          partId
        });
        return;
      }

      if (current.tool === 'probe') {
        if (!pick) {
          sceneRef.current?.annotations.clearProbe();
          current.onProbe(null);
          return;
        }
        const reading = probeAt(pick.point, partId);
        const unit = reading.unit ? ` ${reading.unit}` : '';
        sceneRef.current?.annotations.showProbe(
          pick.point,
          reading.value === null ? 'no data' : `${formatReading(reading.value)}${unit}`
        );
        current.onProbe(reading);
        return;
      }

      if (current.tool === 'measure') {
        if (!pick) return;
        const start = measurePendingRef.current;
        if (!start) {
          measurePendingRef.current = pick.point.clone();
          sceneRef.current?.annotations.showMeasureStart(pick.point);
          measureHintRef.current = true;
          current.onStatus(MEASURE_SECOND_POINT_HINT);
          return;
        }
        const end = pick.point.clone();
        const distanceIn = Vector3.Distance(start, end);
        if (distanceIn < MIN_MEASURE_DISTANCE_IN) return;
        measurePendingRef.current = null;
        if (measureHintRef.current) {
          measureHintRef.current = false;
          current.onStatus(null);
        }
        sceneRef.current?.annotations.showMeasurement(start, end, distanceIn);
        current.onMeasure({
          start: [start.x, start.y, start.z],
          end: [end.x, end.y, end.z],
          distanceIn
        });
        return;
      }

      current.onSelectPart(partId);
    };

    const probeAt = (point: Vector3, partId: string | null) => {
      const current = propsRef.current;
      const resources = resourcesRef.current;
      const mode = current.view.mode;
      const unit = resources?.meta.units?.[mode] ?? RESULT_MODE_UNITS[mode] ?? '';
      const reading = {
        point: [point.x, point.y, point.z] as [number, number, number],
        mode,
        value: null as number | null,
        unit,
        partType: partId ? (current.parts.find((part) => part.id === partId)?.part_type ?? null) : null
      };
      if (!resources) return reading;

      const cell = resources.cellIndexAt(point.x, point.y, point.z);
      if (cell < 0 || resources.mask[cell] === 0) return reading;
      if (reading.partType === null) {
        const maskType = resources.meta.partTypes?.find((entry) => entry.code === resources.mask[cell])?.id;
        reading.partType = PART_TYPES.find((type) => type.id === maskType)?.id ?? null;
      }

      if (isFrameMode(mode)) {
        const stack = resources.stack(mode);
        if (!stack) return reading;
        const byte = stack.sample(stack.nearestIndex(current.view.timePercent / 100), cell);
        reading.value = byte / 255;
        return reading;
      }
      const field = resources.field(mode);
      if (!field) return reading;
      reading.value = field.min + (field.data[cell] / 255) * (field.max - field.min);
      return reading;
    };

    const drawGizmo = () => {
      if (gizmoRef.current) drawAxesGizmo(gizmoRef.current, viewport.camera);
    };
    viewport.scene.onBeforeRenderObservable.add(drawGizmo);

    viewport.scene.doNotHandleCursors = true;
    let hoverAt: { x: number; y: number } | null = null;
    let hoverStale = false;
    let buttonsDown = false;
    let shownCursor = '';
    let nextHoverPickAt = 0;
    let hoverPicks = 0;
    const hoverObserver = viewport.scene.onPointerObservable.add((info) => {
      const scene = viewport.scene;
      if (info.type === PointerEventTypes.POINTERMOVE) {
        buttonsDown = (info.event as PointerEvent).buttons !== 0;
        hoverAt = { x: scene.pointerX, y: scene.pointerY };
        hoverStale = true;
      } else if (info.type === PointerEventTypes.POINTERDOWN) {
        buttonsDown = true;
      } else if (info.type === PointerEventTypes.POINTERUP) {
        buttonsDown = (info.event as PointerEvent).buttons !== 0;
        hoverStale = true;
      } else if (info.type === PointerEventTypes.POINTERWHEEL) {
        hoverStale = true;
      }
    });
    const viewObserver = viewport.camera.onViewMatrixChangedObservable.add(() => {
      hoverStale = true;
    });
    const leaveCanvas = () => {
      hoverAt = null;
      hoverStale = false;
    };
    canvas.addEventListener('pointerleave', leaveCanvas);
    refreshHoverRef.current = () => {
      hoverStale = true;
    };
    const updateHover = () => {
      if (!hoverStale || buttonsDown || !hoverAt) return;
      if (propsRef.current.modeling) {
        hoverStale = false;
        shownCursor = '';
        return;
      }
      const startedAt = performance.now();
      if (startedAt < nextHoverPickAt) return;
      hoverStale = false;
      hoverPicks += 1;
      const pick = pickModel(viewport.scene, hoverAt.x, hoverAt.y);
      nextHoverPickAt = startedAt + (performance.now() - startedAt) / HOVER_PICK_DUTY;
      const cursor = cursorFor(
        propsRef.current.orient?.pick ? 'orientPick' : propsRef.current.tool,
        Boolean(pick?.partId),
        pick !== null
      );
      if (cursor === shownCursor) return;
      shownCursor = cursor;
      canvas.style.cursor = cursor;
    };
    viewport.scene.onBeforeRenderObservable.add(updateHover);
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      const debug = (window as unknown as { __viewportDebug?: Record<string, unknown> }).__viewportDebug;
      if (debug) debug.hoverPicks = () => hoverPicks;
    }

    viewport.engine.runRenderLoop(() => viewport.scene.render());

    const aoTimer = window.setTimeout(() => {
      if (viewport.scene.isDisposed) return;
      if (viewport.engine.getFps() >= AO_MINIMUM_FPS) {
        enableAmbientOcclusion(viewport.scene, viewport.camera);
      }
    }, 1500);

    const resizeObserver = new ResizeObserver(() => {
      viewport.engine.resize();
      camera.refreshProjection();
    });
    resizeObserver.observe(container);

    const entries = partEntriesRef.current;
    const envelopeCache = envelopeCacheRef.current;

    const teardown = () => {
      window.clearTimeout(aoTimer);
      resizeObserver.disconnect();
      viewport.scene.onPointerObservable.remove(pointerObserver);
      viewport.scene.onPointerObservable.remove(hoverObserver);
      viewport.camera.onViewMatrixChangedObservable.remove(viewObserver);
      viewport.scene.onBeforeRenderObservable.removeCallback(drawGizmo);
      viewport.scene.onBeforeRenderObservable.removeCallback(updateHover);
      canvas.removeEventListener('pointerleave', leaveCanvas);
      refreshHoverRef.current = null;
      contourRef.current?.dispose();
      contourRef.current = null;
      resourcesRef.current?.dispose();
      resourcesRef.current = null;
      envelopeCache.clear();
      entries.clear();
      pourMarkerRef.current = null;
      annotations.dispose();
      parting.dispose();
      sketch.dispose();
      caps.dispose();
      envelope.dispose();
      viewport.dispose();
      sceneRef.current = null;
    };

    return () => {
      teardownRef.current = { timer: window.setTimeout(teardown, 0) };
    };
  }, []);

  const signature = geometrySignature(parts);
  const membership = membershipSignature(parts);
  const framedMembershipRef = useRef<string | null>(null);
  const orientActive = Boolean(props.orient?.active);
  useEffect(() => {
    const refs = sceneRef.current;
    if (!refs) return;

    syncPartMeshes(
      refs.viewport.scene,
      propsRef.current.parts,
      partEntriesRef.current,
      (material) =>
        new FieldPaintPluginMaterial(material, refs.placeholderVolume, refs.placeholderLut)
    );

    const bounds =
      partBounds(propsRef.current.parts) ?? gridBounds(resourcesRef.current?.meta ?? null);
    boundsRef.current = bounds;
    refs.camera.setBounds(bounds);
    refs.sketch.setBounds(bounds ? { center: bounds.center, size: bounds.size } : null);
    if (bounds) {
      const orienting = Boolean(propsRef.current.orient?.active);
      refs.viewport.layoutFloor(bounds.center, bounds.size, floorZFor(bounds, orienting));
      refs.originAxes.scaling.setAll(Math.max(bounds.size * 0.1, 0.25));
      refs.originAxes.setEnabled(true);
      refs.annotations.setAssemblySize(bounds.size);
      const placementOnly = framedMembershipRef.current === membership;
      if (!(orienting && placementOnly)) refs.camera.fit({ immediate: true });
      framedMembershipRef.current = membership;
    } else {
      refs.viewport.layoutFloor(Vector3.Zero(), 12, 0);
      refs.originAxes.setEnabled(false);
    }
    paint();
  }, [signature, membership, paint]);

  useEffect(() => {
    const refs = sceneRef.current;
    const bounds = boundsRef.current;
    if (!refs || !bounds) return;
    refs.viewport.layoutFloor(bounds.center, bounds.size, floorZFor(bounds, orientActive));
  }, [orientActive]);

  useEffect(() => {
    refreshHoverRef.current?.();
  }, [props.orient?.pick]);

  useEffect(() => {
    if (!sceneRef.current) return;
    const style = view.style;
    const requested = Math.min(Math.max(view.modelOpacity, 0), 100) / 100;
    const resultsShowing = view.showResults && resultsAvailable;
    const cap = !resultsShowing
      ? 1
      : style === 'envelope'
        ? ENVELOPE_SHELL_OPACITY
        : style === 'both'
          ? BOTH_SHELL_OPACITY
          : 1;
    applyPartAppearance(parts, partEntriesRef.current, {
      selectedPartId: modelingActive ? null : selectedPartId,
      opacity: Math.min(requested, cap),
      showModel: view.showModel,
      assemblySize: boundsRef.current?.size ?? 1,
      neutralGhost: resultsShowing && style === 'envelope',
      resultsShowing,
      depthPrePass: !wantsEnvelope(style)
    });
    paint();
  }, [
    parts,
    selectedPartId,
    view.modelOpacity,
    view.showModel,
    view.style,
    view.mode,
    view.showResults,
    resultsAvailable,
    modelingActive,
    paint
  ]);

  const partingZ = props.orient?.parting?.z ?? null;
  const partingResultsHidden = !view.showPartingLine && !props.orient?.active;
  const partingVisible = (props.orient?.parting?.visible ?? false) && !partingResultsHidden;
  const castingKey = parts
    .filter((part) => part.part_type === 'cast' && part.visible !== false)
    .map((part) => part.id)
    .join('|');
  useEffect(() => {
    const refs = sceneRef.current;
    if (!refs) return;
    const castings: Mesh[] = [];
    for (const id of castingKey ? castingKey.split('|') : []) {
      const entry = partEntriesRef.current.get(id);
      if (entry) castings.push(entry.mesh);
    }
    refs.parting.update(
      partingZ === null ? null : { z: partingZ, visible: partingVisible },
      boundsRef.current,
      castings
    );
  }, [partingZ, partingVisible, castingKey, signature, view.showModel]);

  useEffect(() => {
    sceneRef.current?.sketch.setState(props.modeling ?? null);
    refreshHoverRef.current?.();
  }, [props.modeling]);

  useEffect(() => {
    const refs = sceneRef.current;
    if (!refs) return;
    measurePendingRef.current = null;
    refs.annotations.clearAll();
    clearMeasureHint();
    refreshHoverRef.current?.();
  }, [tool, clearMeasureHint]);

  useEffect(() => {
    const refs = sceneRef.current;
    if (!refs) return;
    const pour = props.pourLocation;
    if (!pour) {
      pourMarkerRef.current?.dispose();
      pourMarkerRef.current = null;
      return;
    }
    if (!pourMarkerRef.current) {
      pourMarkerRef.current = createPourMarker(refs.viewport.scene, boundsRef.current?.size ?? 10);
    }
    const [x, y, z] = pour.point_in;
    pourMarkerRef.current.position.set(x, y, z);
  }, [props.pourLocation]);

  useEffect(() => {
    const refs = sceneRef.current;
    if (!refs) return;

    resourcesRef.current?.dispose();
    resourcesRef.current = null;
    contourRef.current?.dispose();
    contourRef.current = null;
    envelopeCacheRef.current.clear();
    envelopeKeyRef.current = null;
    refs.envelope.setGeometry(null);
    setResourcesEpoch((epoch) => epoch + 1);
    paint();

    if (!projectId || !resultsAvailable) {
      propsRef.current.onGridMeta(null);
      return;
    }

    const controller = new AbortController();
    propsRef.current.onStatus('Loading result grid...');
    RunResources.load(refs.viewport.scene, projectId, controller.signal)
      .then((resources) => {
        if (controller.signal.aborted) {
          resources.dispose();
          return;
        }
        resourcesRef.current = resources;
        resources.onFramesLoaded(() => setResourcesEpoch((epoch) => epoch + 1));

        const contour = new ContourClient((geometry: ContourGeometry) => {
          sceneRef.current?.envelope.setGeometry({
            positions: geometry.positions,
            normals: geometry.normals,
            indices: geometry.indices,
            scalars: geometry.scalars
          });
          paint();
        });
        contour.load({
          dims: resources.dims,
          origin: [resources.meta.origin[0], resources.meta.origin[1], resources.meta.origin[2]],
          spacing: resources.meta.spacing,
          mask: resources.mask
        });
        contourRef.current = contour;

        if (!boundsRef.current) {
          const fallback = gridBounds(resources.meta);
          boundsRef.current = fallback;
          refs.camera.setBounds(fallback);
          if (fallback) {
            refs.viewport.layoutFloor(
              fallback.center,
              fallback.size,
              fallback.min.z - fallback.size * 0.02
            );
            refs.annotations.setAssemblySize(fallback.size);
            refs.camera.fit({ immediate: true });
          }
        }

        propsRef.current.onGridMeta(withReportedModes(resources.meta));
        propsRef.current.onStatus(null);
        setResourcesEpoch((epoch) => epoch + 1);
        paint();
      })
      .catch((error) => {
        if (isAbortError(error) || controller.signal.aborted) return;
        propsRef.current.onStatus(null);
        propsRef.current.onGridMeta(null);
        propsRef.current.onError(error instanceof Error ? error.message : 'Failed to load results');
      });

    return () => controller.abort();
  }, [projectId, resultsAvailable, resultsVersion, paint]);

  useEffect(() => {
    const resources = resourcesRef.current;
    if (!resources || resources.has(view.mode)) {
      paint();
      return;
    }
    const controller = new AbortController();
    propsRef.current.onStatus('Loading field...');
    resources
      .ensure(view.mode, controller.signal)
      .then(() => {
        if (controller.signal.aborted) return;
        propsRef.current.onStatus(null);
        setResourcesEpoch((epoch) => epoch + 1);
        paint();
      })
      .catch((error) => {
        if (isAbortError(error) || controller.signal.aborted) return;
        propsRef.current.onStatus(null);
        propsRef.current.onError(error instanceof Error ? error.message : 'Failed to load field');
      });
    return () => controller.abort();
  }, [view.mode, resourcesEpoch, paint]);

  const staticEnvelopeKey =
    !isTimeMode(view.mode) && !isSweepMode(view.mode) && wantsEnvelope(view.style)
      ? `${projectId}|${view.mode}|${view.threshold}|${view.castOnly}|${resourcesEpoch}`
      : null;

  useEffect(() => {
    const refs = sceneRef.current;
    if (!refs || staticEnvelopeKey === null) return;
    const resources = resourcesRef.current;
    if (!resources || !projectId) return;

    envelopeKeyRef.current = staticEnvelopeKey;

    const cached = envelopeCacheRef.current.get(staticEnvelopeKey);
    if (cached) {
      applyEnvelope(refs.envelope, cached);
      paint();
      return;
    }

    const controller = new AbortController();
    const current = propsRef.current.view;
    const threshold = thresholdInFieldUnits(current, resources.meta);
    const timer = window.setTimeout(() => {
      propsRef.current.onStatus('Building envelope...');
      fetchEnvelope(projectId, current.mode, {
        threshold,
        castOnly: current.castOnly,
        signal: controller.signal
      })
        .then((envelope) => {
          if (controller.signal.aborted) return;
          envelopeCacheRef.current.set(staticEnvelopeKey, envelope);
          if (envelopeCacheRef.current.size > 12) {
            const oldest = envelopeCacheRef.current.keys().next().value;
            if (oldest !== undefined) envelopeCacheRef.current.delete(oldest);
          }
          if (envelopeKeyRef.current !== staticEnvelopeKey) return;
          const scene = sceneRef.current;
          if (!scene) return;
          applyEnvelope(scene.envelope, envelope);
          propsRef.current.onStatus(null);
          paint();
        })
        .catch((error) => {
          if (isAbortError(error) || controller.signal.aborted) return;
          propsRef.current.onStatus(null);
          propsRef.current.onError(
            error instanceof Error ? error.message : 'Failed to build the envelope'
          );
        });
    }, ENVELOPE_DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [staticEnvelopeKey, projectId, paint]);

  useEffect(() => {
    const refs = sceneRef.current;
    const resources = resourcesRef.current;
    const contour = contourRef.current;
    if (!refs || !resources || !contour) return;
    if (!isTimeMode(view.mode) || !wantsEnvelope(view.style)) return;

    const stack = resources.stack(view.mode);
    if (!stack) return;

    const { lower, upper, blend } = stack.bracket(view.timePercent / 100);
    let cancelled = false;

    void Promise.all([stack.whenResident(lower), stack.whenResident(upper)]).then(([a, b]) => {
      if (cancelled || !a || !b) return;
      const frameA = stack.frameData(lower);
      const frameB = stack.frameData(upper);
      if (!frameA || !frameB) return;
      contour.request({
        frameA,
        frameB,
        blend,
        iso: 0.5,
        castOnly: view.castOnly,
        invert: false,
        blurStrength: FRONT_BLUR_STRENGTH
      });
    });

    return () => {
      cancelled = true;
    };
  }, [view.mode, view.style, view.timePercent, view.castOnly, resourcesEpoch]);

  useEffect(() => {
    const refs = sceneRef.current;
    const resources = resourcesRef.current;
    const contour = contourRef.current;
    if (!refs || !resources || !contour) return;
    if (!isSweepMode(view.mode) || !wantsEnvelope(view.style)) return;

    const field = resources.field(view.mode);
    if (!field) return;

    const span = field.max - field.min;
    const sweepTime = (view.timePercent / 100) * field.max;
    const iso = span > 1e-9 ? Math.min(1, Math.max(0, (sweepTime - field.min) / span)) : 0;

    contour.request({
      frameA: field.data,
      frameB: field.data,
      blend: 0,
      iso,
      castOnly: view.castOnly,
      invert: false,
      blurStrength: FRONT_BLUR_STRENGTH
    });
  }, [view.mode, view.style, view.timePercent, view.castOnly, resourcesEpoch]);

  useEffect(() => {
    const refs = sceneRef.current;
    if (!refs) return;
    if (!wantsEnvelope(view.style)) {
      refs.envelope.setVisible(false);
      return;
    }
    if (isTimeMode(view.mode) || isSweepMode(view.mode)) {
      envelopeKeyRef.current = null;
    }
  }, [view.style, view.mode]);

  useEffect(() => {
    paint();
  }, [
    view.colormap,
    view.scaleMin,
    view.scaleMax,
    view.timePercent,
    view.showResults,
    view.castOnly,
    view.style,
    view.threshold,
    view.section,
    resourcesEpoch,
    paint
  ]);

  useEffect(() => {
    sceneRef.current?.camera.setOrtho(ortho);
  }, [ortho]);

  useEffect(() => {
    if (!view.playing || !isTimeMode(view.mode)) return;
    let frame = 0;
    let last = performance.now();

    const tick = (now: number) => {
      const deltaSeconds = Math.min((now - last) / 1000, 0.25);
      last = now;
      const current = propsRef.current;
      const step = deltaSeconds * PLAYBACK_PERCENT_PER_SECOND * (current.view.speed || 1);
      const next = current.view.timePercent + step;
      current.onViewChange({ timePercent: next >= 100 ? next % 100 : next });
      frame = requestAnimationFrame(tick);
    };

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [view.playing, view.speed, view.mode]);

  useImperativeHandle(
    ref,
    (): ViewportHandle => ({
      screenshot: async (options) => {
        const refs = sceneRef.current;
        if (!refs) throw new Error('Viewport is not ready');
        if (options?.width || options?.height) {
          const dataUrl = await Tools.CreateScreenshotUsingRenderTargetAsync(
            refs.viewport.engine,
            refs.viewport.camera,
            { width: options.width ?? 1920, height: options.height ?? 1080 },
            'image/png',
            4,
            true
          );
          return await (await fetch(dataUrl)).blob();
        }
        const canvas = canvasRef.current;
        if (!canvas) throw new Error('Viewport canvas is not ready');
        refs.viewport.scene.render();
        return await new Promise<Blob>((resolve, reject) => {
          canvas.toBlob(
            (blob) => (blob ? resolve(blob) : reject(new Error('Screenshot failed'))),
            'image/png'
          );
        });
      },
      setCamera: (preset) => sceneRef.current?.camera.goTo(preset),
      normalTo: (frame) => {
        const refs = sceneRef.current;
        if (!refs) return;
        refs.camera.lookAlong(
          new Vector3(frame.normal[0], frame.normal[1], frame.normal[2]),
          new Vector3(frame.v[0], frame.v[1], frame.v[2]),
          new Vector3(frame.focus[0], frame.focus[1], frame.focus[2])
        );
      },
      clearMeasurements: () => {
        measurePendingRef.current = null;
        clearMeasureHint();
        sceneRef.current?.annotations.clearMeasurements();
      },
      defectVolumeIn3: () => defectVolume()
    }),
    [clearMeasureHint]
  );

  const defectVolume = (): number | null => {
    const resources = resourcesRef.current;
    const state = propsRef.current.view;
    if (!resources || isFrameMode(state.mode)) return null;
    const field = resources.field(state.mode);
    if (!field) return null;

    const cutoff = thresholdInFieldUnits(state, resources.meta);
    const span = field.max - field.min;
    if (!(span > 0)) return 0;
    const cutoffByte = ((cutoff - field.min) / span) * 255;

    let cells = 0;
    const mask = resources.mask;
    const castOnly = state.castOnly;
    for (let index = 0; index < mask.length; index += 1) {
      const code = mask[index];
      if (code === 0) continue;
      if (castOnly && code !== 1) continue;
      if (field.data[index] >= cutoffByte) cells += 1;
    }
    return cells * resources.meta.spacing ** 3;
  };

  return (
    <div
      ref={containerRef}
      className={props.className}
      style={{ position: 'absolute', inset: 0, overflow: 'hidden' }}
    >
      <canvas
        ref={canvasRef}
        className="viewport-canvas"
        style={{ width: '100%', height: '100%', display: 'block', outline: 'none', touchAction: 'none' }}
      />
      {}
      <div ref={sketchLayerRef} className="sketch-layer" />
      <canvas ref={gizmoRef} className="viewport-gizmo" width={96} height={96} aria-hidden="true" />
    </div>
  );
});

const PART_NAME_PREFIX = 'part-';

function isPartMesh(mesh: AbstractMesh): boolean {
  return mesh.name.startsWith(PART_NAME_PREFIX);
}

function partIdOf(mesh: AbstractMesh | null | undefined): string | null {
  return mesh && isPartMesh(mesh) ? mesh.name.slice(PART_NAME_PREFIX.length) : null;
}

function applyEnvelope(mesh: EnvelopeMesh, envelope: EnvelopeData): void {
  mesh.setGeometry({
    positions: envelope.positions,
    normals: envelope.normals,
    indices: envelope.indices,
    scalars: envelope.scalars
  });
}

function formatReading(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return '0';
  if (magnitude >= 1000 || magnitude < 0.001) return value.toExponential(1);
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 10) return value.toFixed(1);
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

function withReportedModes(meta: GridMeta): GridMeta {
  const known = new Set<string>(RESULT_MODE_IDS);
  const filter = (modes: readonly ResultModeId[] | undefined): ResultModeId[] =>
    (modes ?? []).filter((mode) => known.has(mode));
  return { ...meta, modes: filter(meta.modes), emptyModes: filter(meta.emptyModes) };
}
