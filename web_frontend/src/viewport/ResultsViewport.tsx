/**
 * The 3D results viewport.
 *
 * Renders the casting as a solid, environment-lit object and paints the
 * result field onto it continuously, by sampling a 3D texture of the dense
 * grid at each fragment's world position. There are no voxel cubes anywhere
 * in this file — the result is a colour on the casting's own surface, on the
 * capped face of a section cut, and on the isosurface that encloses whatever
 * is above the threshold.
 *
 * Three ways of showing the same field, chosen by `view.style`:
 *
 *   surface   the field painted on the casting's outside
 *   envelope  a smooth isosurface of the threshold, inside a ghosted shell
 *   both      the painted surface at partial opacity with the envelope inside
 *
 * The isosurface comes from the server for a static threshold and from a
 * worker for a moving one — a solidification front changes every frame, and
 * a round trip per frame is not a thing playback can afford.
 *
 * Implements `ViewportProps` / `ViewportHandle` from `lib/viewState` exactly;
 * the shell treats this component as a black box.
 */
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState
} from 'react';
import { Color3, PointerEventTypes, Plane, Tools, Vector3 } from '@babylonjs/core';
import type { Mesh, PBRMaterial } from '@babylonjs/core';

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
import { applyPartAppearance, syncPartMeshes, type PartEntry } from './partMeshes';
import { SECTION_AXES, SectionCaps, type SectionAxis } from './sectionCaps';
import {
  createOriginAxes,
  createViewportScene,
  enableAmbientOcclusion,
  type ViewportScene
} from './sceneSetup';

/** A pointerup further than this from its pointerdown was a camera drag, not a click. */
const CLICK_DRAG_THRESHOLD_PX = 4;

/** Playback advances the scrubber this far per second at 1x speed. */
const PLAYBACK_PERCENT_PER_SECOND = 12;

/**
 * How long the threshold has to stop moving before an envelope is fetched.
 *
 * A threshold slider emits a change per pixel of travel; contouring is tens
 * of milliseconds server-side plus the transfer. 150 ms is long enough that a
 * sweep across the slider costs one request and short enough that letting go
 * feels immediate.
 */
const ENVELOPE_DEBOUNCE_MS = 150;

/**
 * Opacity the casting drops to so an envelope inside it can be seen.
 *
 * A defect envelope is a small blob somewhere inside the metal, so the shell
 * has to get out of its way almost entirely. A liquid front is not: it fills
 * the casting, and at 0.22 with no paint on it the shell disappeared -- which
 * is what "no results in the riser" was. A riser is mostly *outside* the front
 * for most of a run, so with the shell ghosted there was nothing there to see
 * at all. For the time modes the shell stays at the same 0.35 that 'both' uses
 * and keeps its paint, so the riser still carries the field.
 */
const ENVELOPE_SHELL_OPACITY = 0.22;
const BOTH_SHELL_OPACITY = 0.35;
const TIME_ENVELOPE_SHELL_OPACITY = 0.35;

/**
 * How much of its own colour the liquid front emits.
 *
 * Measured by eye against the tone-mapped scene rather than derived: below
 * about 0.4 the front still reads as lit metal, and past about 0.9 the ACES
 * curve clips the yellow end to white and the pool loses the shape the ramp is
 * there to show.
 */
const MOLTEN_GLOW = 0.62;

/**
 * How much of the contourer's masked blur to keep.
 *
 * A full blur was needed while the front had to look solid on its own; it also
 * pushed the 0.5 crossing out to the casting wall, where on a coarse lattice
 * it draped as molten tongues down the outside of the riser. The skin now
 * closes the liquid body at the wall, so the front only has to be smooth
 * *inside* the metal, and a light mix does that without the drape.
 */
const FRONT_BLUR_STRENGTH = 0.3;

/**
 * Colour for metal the sweep has not reached, in linear space.
 *
 * The liquid end of the molten ramp, converted the same way the envelope's
 * vertex colours are, so a still-liquid cell on the painted casting and the
 * front that bounds it are the same shade of hot.
 */
const LIQUID_COLOR: [number, number, number] = (() => {
  const [r, g, b] = sampleColormap('molten', 1);
  const toLinear = (channel: number) =>
    channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  return [toLinear(r / 255), toLinear(g / 255), toLinear(b / 255)];
})();

/** Sustained frame rate a machine must hold before it is offered SSAO. */
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
};

/**
 * What has to change before the camera is allowed to re-frame the model.
 *
 * Deliberately excludes opacity, section planes, selection and every result
 * setting: re-fitting on those was the previous renderer's most disruptive
 * behaviour, because nudging an opacity slider threw away the view the user
 * had just set up.
 */
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

function partBounds(parts: ProjectPart[]): Bounds | null {
  const boxes = parts
    .map((part) => part.meshData?.bounds)
    .filter((box): box is NonNullable<typeof box> => Boolean(box));
  return unionBounds(boxes);
}

/**
 * The lattice's own extent, as a fallback for framing.
 *
 * Results can be present while the part geometry is not -- a project reopened
 * from the database has parts whose triangles have not been fetched yet, and
 * the envelope and the section caps are drawn from the grid rather than from
 * the meshes. Framing the lattice means the result is still on screen instead
 * of somewhere off in the dark while the geometry catches up.
 */
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

/**
 * Section planes, as material clip planes rather than scene clip planes.
 *
 * Scene-wide clipping would take the shop floor and the origin axes with it,
 * and the floor is the one thing that should stay put while the casting is
 * cut open.
 */
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
    // Babylon discards fragments on the +normal side of the plane, so the
    // default normal points up-axis and the low side of the cut survives.
    // Flip keeps the other half.
    const direction = state.flip ? normal.scale(-1) : normal;
    planes[axis] = Plane.FromPositionAndNormal(
      new Vector3(axis === 'x' ? position : 0, axis === 'y' ? position : 0, axis === 'z' ? position : 0),
      direction
    );
  });
  return planes;
}

/** Threshold percent to field units, against the field's own range. */
function thresholdInFieldUnits(view: ResultViewState, meta: GridMeta | null): number {
  const range = meta?.ranges[view.mode];
  const min = range?.min ?? 0;
  const max = range?.max ?? 1;
  return min + (Math.min(Math.max(view.threshold, 0), 100) / 100) * (max - min);
}

/** Does this style want an isosurface at all? */
function wantsEnvelope(style: ResultViewState['style']): boolean {
  return style === 'envelope' || style === 'both';
}


export const ResultsViewport = forwardRef<ViewportHandle, ViewportProps>(function ResultsViewport(props, ref) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const gizmoRef = useRef<HTMLCanvasElement | null>(null);

  // Bumped whenever a new `RunResources` becomes current. The mode-loading
  // effect keys off it: the grid arrives from a promise, long after the
  // effect that started it settled, and without this the first mode of a run
  // would never be fetched at all.
  const [resourcesEpoch, setResourcesEpoch] = useState(0);

  const sceneRef = useRef<SceneRefs | null>(null);
  const partEntriesRef = useRef<Map<string, PartEntry>>(new Map());
  const resourcesRef = useRef<RunResources | null>(null);
  const boundsRef = useRef<Bounds | null>(null);
  const pourMarkerRef = useRef<Mesh | null>(null);
  const measurePendingRef = useRef<Vector3 | null>(null);
  const contourRef = useRef<ContourClient | null>(null);
  /** A scene teardown waiting to see whether this is an unmount or a remount. */
  const teardownRef = useRef<{ timer: number } | null>(null);
  /** Envelopes already fetched, keyed by everything that changes the geometry. */
  const envelopeCacheRef = useRef<Map<string, EnvelopeData>>(new Map());
  const envelopeKeyRef = useRef<string | null>(null);

  // Every Babylon callback runs outside React's render cycle, so it reads the
  // latest props through this ref rather than closing over a stale render.
  const propsRef = useRef(props);
  propsRef.current = props;

  const { projectId, resultsAvailable, resultsVersion, view, ortho, parts, selectedPartId, tool } = props;

  // ---------------------------------------------------------------------
  // Painting
  // ---------------------------------------------------------------------

  /**
   * Push the whole view state onto the GPU.
   *
   * One function rather than an effect per property: mode, colormap, scale,
   * time, style, section and cast-only all feed the same uniforms and the
   * same handful of meshes, and splitting them would mean each one having to
   * know what the others had already written.
   */
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

    // A sweep mode reads its own values as times. `sweepTime` is the cut-off
    // the shader compares each cell against, taken from 0 rather than from the
    // field's minimum so that t = 0 is "nothing has frozen yet" rather than
    // "the first cell is about to". At 100 percent it sits on the maximum, so
    // every cell is at or below it and the view is the static map exactly.
    const sweeping = isSweepMode(mode) && ready;

    /**
     * The value above which the metal at a point is still liquid.
     *
     * For a sweep it is the scrubber's own time. For a frame mode it is the
     * same 0.5 the front is contoured at -- but only in the styles that draw a
     * front, because in `surface` the whole point is the field's own colours
     * and recolouring half of it as metal would throw them away.
     */
    const liquidCutoff = sweeping
      ? (state.timePercent / 100) * quantMax
      : isFrameMode(mode) && ready && wantsEnvelope(state.style)
        ? 0.5
        : null;

    /**
     * Whether the skin closes the liquid region into a solid body.
     *
     * Only where a front is drawn: in `surface` the casting is opaque anyway
     * and there is nothing to close.
     */
    const liquidBody = liquidCutoff !== null && wantsEnvelope(state.style) && state.showResults;

    // In 'envelope' the shell is a ghost: it locates the defect in the part
    // but carries no colour of its own, so nothing competes with the surface
    // that does. In 'both' the shell keeps its paint and steps back.
    //
    // The time modes are the exception, and the reason is the riser. A front
    // is not a small marker inside the metal, it is a boundary that sweeps
    // through it, and a riser spends most of a run on the far side of that
    // boundary. Stripping the paint there left the riser blank -- reported as
    // "no results in the riser" -- when in fact it had a perfectly good
    // fraction-liquid value the whole time. So a front gets a painted shell to
    // sit inside, not an empty one.
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
        plugin.state.liquidColor = LIQUID_COLOR;
        plugin.state.liquidGlow = MOLTEN_GLOW;
        plugin.state.liquidBody = liquidBody;
        plugin.state.solidAlpha = TIME_ENVELOPE_SHELL_OPACITY;
        // One voxel: the layer the front is not allowed to enter.
        plugin.state.skinDepth = meta.spacing;
        plugin.fieldTextureA = stack ? stack.textureA : (field?.texture ?? null);
        plugin.fieldTextureB = stack ? stack.textureB : (field?.texture ?? null);
        plugin.maskTexture = resources.maskTexture;
        plugin.lutTexture = resources.lut(state.colormap);
      } else {
        plugin.state.strength = 0;
        plugin.state.sweepTime = null;
        plugin.state.liquidBody = false;
      }
    };

    for (const entry of entries.values()) {
      applyPaint(entry.plugin, paintSurface ? 1 : 0);
      entry.material.clipPlane = planes.x;
      entry.material.clipPlane2 = planes.y;
      entry.material.clipPlane3 = planes.z;
    }

    // Caps are painted whatever the style: the point of cutting a casting
    // open is to see the field on the inside, and a hollow shell in
    // 'envelope' would show the envelope through a hole rather than through
    // a cut.
    refs.caps.layout(state.section, state.section.caps, boundsRef.current, planes);
    refs.caps.orient(state.section);
    for (const cap of refs.caps.entries()) {
      applyPaint(cap.plugin, state.showResults && ready ? 1 : 0);
      cap.plugin.state.discardOutside = true;
      // A cut face is a solid slab of metal whichever side of the front it is
      // on: it shows the field on the inside, so it is never ghosted and never
      // needs closing.
      cap.plugin.state.liquidBody = false;
    }

    // The envelope is clipped by the same planes, so a section cuts through
    // it rather than leaving it floating in front of the cut.
    refs.envelope.material.clipPlane = planes.x;
    refs.envelope.material.clipPlane2 = planes.y;
    refs.envelope.material.clipPlane3 = planes.z;
    // A liquid front and a defect envelope are the same geometry with opposite
    // jobs. The envelope marks a region and should look like a marker, so it
    // stays an ordinary lit surface. The front is a picture of the metal
    // itself, and metal at pouring temperature emits light rather than
    // reflecting it -- a reflective front borrows the environment map and
    // comes out looking like polished steel whatever colour it is painted.
    //
    // A frame mode's front is the liquid pool; a sweep mode's is the region
    // that has not frozen yet. Same surface, same material.
    const molten = isFrameMode(mode) || sweeping;

    // The front is a level set of the metal, so it is discarded wherever the
    // occupancy grid says there is no metal to be a front of. See
    // `EnvelopeMesh.clip`: the contourer stopped walling the metal off with a
    // sentinel value, which is what removed the false shell at t = 0, and this
    // is what keeps the surface it does produce inside the casting.
    applyPaint(refs.envelope.clip, 0);
    refs.envelope.clip.state.liquidBody = false;
    refs.envelope.clip.state.discardOutside = molten;
    refs.envelope.clip.state.discardUnpaintable = molten && state.castOnly;
    refs.envelope.setStyle({
      // A molten front is coloured by the metal it encloses, so it takes the
      // molten map whatever the casting around it is painted with.
      colormap: molten ? 'molten' : state.colormap,
      // A front is contoured out of a normalized 0..1 grid, so its per-vertex
      // scalars are in those units and must be ramped over them, not over
      // whatever the legend has been clamped to. This is why the pool keeps
      // its full molten range even when the scale is pinned.
      scaleMin: molten ? 0 : scaleMin,
      scaleMax: molten ? 1 : scaleMax,
      // Only reached when a surface arrives without per-vertex scalars.
      flatColor: METAL_COLOR,
      alpha: 1,
      metallic: molten ? 0.1 : 0.35,
      roughness: molten ? 0.55 : 0.5,
      glow: molten ? MOLTEN_GLOW : 0
    });
    // Bloom is what carries the glow past the edge of the mesh into the
    // surrounding air. It is switched with the front rather than left on: on
    // every other mode there is nothing on screen bright enough to want it,
    // and it would only cost a blur pass.
    refs.viewport.pipeline.bloomEnabled = molten && showEnvelope;
    refs.envelope.setVisible(showEnvelope);
  }, []);

  // ---------------------------------------------------------------------
  // Scene lifetime
  // ---------------------------------------------------------------------

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    // A mount that immediately follows an unmount is a remount, not a new
    // viewport: React's strict mode does exactly that in development, and the
    // shell does it whenever the canvas is re-parented. Rebuilding for it
    // means tearing down a WebGL context and standing another one up, and the
    // environment map's asynchronous decode does not survive losing its scene
    // half way through -- it leaves the texture holding raw encoded data,
    // which as an image-based light is black, and a metallic casting lit only
    // by that disappears. Cancelling the pending teardown instead keeps one
    // context for the life of the component.
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
    sceneRef.current = {
      viewport,
      camera,
      placeholderVolume,
      placeholderLut,
      originAxes,
      caps,
      envelope,
      annotations
    };

    // Local-only handle for the screenshot harness and for diagnosing a
    // render that does not look like the numbers say it should. Gated on the
    // hostname rather than a build flag, matching how the rest of this
    // codebase marks development-only paths.
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      (window as unknown as { __viewportDebug?: unknown }).__viewportDebug = {
        engine: viewport.engine,
        scene: viewport.scene,
        parts: partEntriesRef.current,
        contour: () => contourRef.current,
        resources: () => resourcesRef.current
      };
    }

    if (viewport.engine.webGLVersion < 2) {
      propsRef.current.onError(
        'This browser reports WebGL 1 only. Result fields need WebGL 2 and will not be painted.'
      );
    }

    let pointerDownAt: { x: number; y: number } | null = null;
    const pointerObserver = viewport.scene.onPointerObservable.add((info) => {
      const scene = viewport.scene;
      if (info.type === PointerEventTypes.POINTERDOWN) {
        pointerDownAt = { x: scene.pointerX, y: scene.pointerY };
        return;
      }
      if (info.type === PointerEventTypes.POINTERDOUBLETAP) {
        const pick = scene.pick(scene.pointerX, scene.pointerY, (mesh) => mesh.name.startsWith('part-'));
        if (pick?.hit && pick.pickedPoint) camera.focusOn(pick.pickedPoint);
        return;
      }
      if (info.type !== PointerEventTypes.POINTERUP) return;

      const downAt = pointerDownAt;
      pointerDownAt = null;
      if (downAt) {
        const travelled = Math.hypot(scene.pointerX - downAt.x, scene.pointerY - downAt.y);
        // An orbit begins and ends in this same handler; only a pointerup
        // close to its pointerdown is a click. This is what keeps the probe
        // and measure tools out of the way of the camera.
        if (travelled > CLICK_DRAG_THRESHOLD_PX) return;
      }
      handleClick();
    });

    const handleClick = () => {
      const scene = viewport.scene;
      const current = propsRef.current;
      const pick = scene.pick(scene.pointerX, scene.pointerY, (mesh) => mesh.name.startsWith('part-'));
      const partId = pick?.pickedMesh?.name.startsWith('part-')
        ? pick.pickedMesh.name.slice('part-'.length)
        : null;

      if (current.tool === 'pour') {
        if (!pick?.hit || !pick.pickedPoint) return;
        const normal = pick.getNormal(true, true);
        current.onPourPicked({
          point: [pick.pickedPoint.x, pick.pickedPoint.y, pick.pickedPoint.z],
          normal: normal ? [normal.x, normal.y, normal.z] : null,
          partId
        });
        return;
      }

      if (current.tool === 'probe') {
        if (!pick?.hit || !pick.pickedPoint) {
          sceneRef.current?.annotations.clearProbe();
          current.onProbe(null);
          return;
        }
        const reading = probeAt(pick.pickedPoint, partId);
        const unit = reading.unit ? ` ${reading.unit}` : '';
        sceneRef.current?.annotations.showProbe(
          pick.pickedPoint,
          reading.value === null ? 'no data' : `${formatReading(reading.value)}${unit}`
        );
        current.onProbe(reading);
        return;
      }

      if (current.tool === 'measure') {
        if (!pick?.hit || !pick.pickedPoint) return;
        const start = measurePendingRef.current;
        if (!start) {
          measurePendingRef.current = pick.pickedPoint.clone();
          sceneRef.current?.annotations.showMeasureStart(pick.pickedPoint);
          return;
        }
        const end = pick.pickedPoint.clone();
        measurePendingRef.current = null;
        const distanceIn = Vector3.Distance(start, end);
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

    viewport.engine.runRenderLoop(() => viewport.scene.render());

    // Ambient occlusion, gated on a measurement rather than a guess. Taken
    // after a second and a half of real rendering, because the same browser
    // is fast on a GPU and unusable on a software rasterizer: measured here,
    // Chrome on Metal holds 119-124 fps with SSAO2 on while orbiting this
    // 41,593-voxel casting, and the SwiftShader rasterizer the headless
    // harness uses manages 41 fps without it. One of those can afford a
    // second full-resolution depth pass and the other cannot.
    const aoTimer = window.setTimeout(() => {
      if (viewport.scene.isDisposed) return;
      if (viewport.engine.getFps() >= AO_MINIMUM_FPS) {
        enableAmbientOcclusion(viewport.scene, viewport.camera);
      }
    }, 1500);

    // The shell resizes the canvas by opening and closing panels, which never
    // fires a window resize. Watching the container is the only way to see it.
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
      viewport.scene.onBeforeRenderObservable.removeCallback(drawGizmo);
      contourRef.current?.dispose();
      contourRef.current = null;
      resourcesRef.current?.dispose();
      resourcesRef.current = null;
      envelopeCache.clear();
      entries.clear();
      pourMarkerRef.current = null;
      annotations.dispose();
      caps.dispose();
      envelope.dispose();
      viewport.dispose();
      sceneRef.current = null;
    };

    return () => {
      // Deferred by one task, so a remount can cancel it. A real unmount has
      // nothing to cancel it and the scene goes away on the next tick.
      teardownRef.current = { timer: window.setTimeout(teardown, 0) };
    };
  }, []);

  // ---------------------------------------------------------------------
  // Parts
  // ---------------------------------------------------------------------

  const signature = geometrySignature(parts);
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
    if (bounds) {
      refs.viewport.layoutFloor(bounds.center, bounds.size, bounds.min.z - bounds.size * 0.02);
      // Short enough to say where the origin is without drawing the eye.
      refs.originAxes.scaling.setAll(Math.max(bounds.size * 0.1, 0.25));
      refs.originAxes.setEnabled(true);
      refs.annotations.setAssemblySize(bounds.size);
      // Only here. The set of parts or their placement changed, so the last
      // framing is no longer about the thing on screen.
      refs.camera.fit({ immediate: true });
    } else {
      // An empty stage gets a quiet floor and no axes to explain.
      refs.viewport.layoutFloor(Vector3.Zero(), 12, 0);
      refs.originAxes.setEnabled(false);
    }
    paint();
    // `signature` is the dependency on purpose: it changes exactly when the
    // geometry does, and not when a slider moves.
  }, [signature, paint]);

  useEffect(() => {
    if (!sceneRef.current) return;
    const style = view.style;
    const requested = Math.min(Math.max(view.modelOpacity, 0), 100) / 100;
    // An envelope has to be visible through the casting that contains it --
    // but a liquid front only has to be visible *through* it, not instead of
    // it. See ENVELOPE_SHELL_OPACITY.
    // A sweep's envelope is a front like any other, so it gets the readable
    // shell rather than the defect envelope's near-invisible one.
    const front = isFrameMode(view.mode) || isSweepMode(view.mode);
    const cap =
      style === 'envelope'
        ? front
          ? TIME_ENVELOPE_SHELL_OPACITY
          : ENVELOPE_SHELL_OPACITY
        : style === 'both'
          ? BOTH_SHELL_OPACITY
          : 1;
    applyPartAppearance(parts, partEntriesRef.current, {
      selectedPartId,
      opacity: Math.min(requested, cap),
      showModel: view.showModel,
      assemblySize: boundsRef.current?.size ?? 1,
      // A ghost shell must not claim the depth buffer ahead of the envelope
      // it is supposed to be showing.
      depthPrePass: !wantsEnvelope(style)
    });
    paint();
    // `view.mode` is a dependency because the opacity cap now depends on it:
    // a time mode keeps a readable shell where a defect mode ghosts it.
  }, [
    parts,
    selectedPartId,
    view.modelOpacity,
    view.showModel,
    view.style,
    view.mode,
    view.showResults,
    paint
  ]);

  // Markers belong to the tool that made them.
  useEffect(() => {
    const refs = sceneRef.current;
    if (!refs) return;
    measurePendingRef.current = null;
    refs.annotations.clearAll();
  }, [tool]);

  // ---------------------------------------------------------------------
  // Pour marker
  // ---------------------------------------------------------------------

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

  // ---------------------------------------------------------------------
  // Results
  // ---------------------------------------------------------------------

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
        // A streamed window arriving is new data for the painted surface and
        // for the contoured front alike; both key off the epoch.
        resources.onFramesLoaded(() => setResourcesEpoch((epoch) => epoch + 1));

        // The contouring worker only ever needs the lattice and the mask;
        // frames travel with each request.
        const contour = new ContourClient((geometry: ContourGeometry) => {
          sceneRef.current?.envelope.setGeometry({
            positions: geometry.positions,
            normals: geometry.normals,
            indices: geometry.indices,
            // Not the value *on* the front -- that is the iso everywhere -- but
            // the value just behind it, which is what makes the pool read hot
            // in its body and cooling at its rim.
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
          // Nothing framed the view yet, because no part carried geometry.
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

  // Fetch whichever mode is selected, then paint it.
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

  // ---------------------------------------------------------------------
  // Envelopes
  // ---------------------------------------------------------------------

  // A sweep mode is excluded alongside the frame modes: its envelope is the
  // still-liquid region at the scrubber's position, which the worker contours
  // per position, not a fixed threshold the server can build once.
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
    // Debounced, so dragging the threshold slider is one request rather than
    // one per pixel of travel.
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
          // A handful of thresholds is plenty; each is a few hundred KB.
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

  // A moving front is contoured here rather than fetched: at four playback
  // speeds over 93 frames a round trip per frame is not affordable.
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

    // Wait for the two frames rather than skipping them. On a stack small
    // enough to hold whole this resolves immediately; on a streamed one it is
    // the difference between a front that appears and one that never does.
    void Promise.all([stack.whenResident(lower), stack.whenResident(upper)]).then(([a, b]) => {
      if (cancelled || !a || !b) return;
      const frameA = stack.frameData(lower);
      const frameB = stack.frameData(upper);
      if (!frameA || !frameB) return;
      contour.request({
        frameA,
        frameB,
        blend,
        // Half solid, half liquid: the front itself.
        iso: 0.5,
        castOnly: view.castOnly,
        // Fraction liquid is high where the metal is still molten, so the
        // region of interest is above the iso; a fill fraction behaves the
        // same way, and nothing else reaches here.
        invert: false,
        blurStrength: FRONT_BLUR_STRENGTH
      });
    });

    return () => {
      cancelled = true;
    };
  }, [view.mode, view.style, view.timePercent, view.castOnly, resourcesEpoch]);

  // The sweep modes contour the same way, from one static grid instead of a
  // pair of frames: the still-liquid region at time t is simply the part of
  // the solidification-time field that lies above t. Passing the one grid as
  // both frames with a zero blend reuses the whole path -- masked blur,
  // nearest-inside extension, surface nets, per-vertex colour -- for free.
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
    // The grid's bytes decode as `min + byte / 255 * (max - min)`, so the iso
    // in those units is where the cut-off sits within the field's own range.
    // Clamped at zero because a cut-off below the field's minimum would put
    // the iso underneath the extended background as well, and the contourer
    // would enclose the bounding box rather than the casting.
    const iso = span > 1e-9 ? Math.min(1, Math.max(0, (sweepTime - field.min) / span)) : 0;

    contour.request({
      frameA: field.data,
      frameB: field.data,
      blend: 0,
      iso,
      castOnly: view.castOnly,
      // Still liquid means "has not solidified yet", which is where the
      // solidification time is *above* the cut-off.
      invert: false,
      blurStrength: FRONT_BLUR_STRENGTH
    });
  }, [view.mode, view.style, view.timePercent, view.castOnly, resourcesEpoch]);

  // Clear a stale front when the style or mode stops wanting one.
  useEffect(() => {
    const refs = sceneRef.current;
    if (!refs) return;
    if (!wantsEnvelope(view.style)) {
      refs.envelope.setVisible(false);
      return;
    }
    if (isTimeMode(view.mode) || isSweepMode(view.mode)) {
      // The worker effects above own the geometry from here.
      envelopeKeyRef.current = null;
    }
  }, [view.style, view.mode]);

  // Everything else the paint depends on. Cheap: uniform writes, no uploads.
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

  // ---------------------------------------------------------------------
  // Playback
  // ---------------------------------------------------------------------

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
      // Loop rather than stop: a solidification animation is read by watching
      // the same front advance several times over.
      current.onViewChange({ timePercent: next >= 100 ? next % 100 : next });
      frame = requestAnimationFrame(tick);
    };

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [view.playing, view.speed, view.mode]);

  // ---------------------------------------------------------------------
  // Imperative surface
  // ---------------------------------------------------------------------

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
        // Render once more first: the drawing buffer is only guaranteed to
        // hold the current frame immediately after a render.
        refs.viewport.scene.render();
        return await new Promise<Blob>((resolve, reject) => {
          canvas.toBlob(
            (blob) => (blob ? resolve(blob) : reject(new Error('Screenshot failed'))),
            'image/png'
          );
        });
      },
      setCamera: (preset) => sceneRef.current?.camera.goTo(preset),
      clearMeasurements: () => {
        measurePendingRef.current = null;
        sceneRef.current?.annotations.clearMeasurements();
      },
      defectVolumeIn3: () => defectVolume()
    }),
    []
  );

  /**
   * Volume of metal above the current threshold, in cubic inches.
   *
   * Counted on the dense grid rather than on rendered geometry, so it is the
   * same number whatever the camera is doing and whatever is hidden. A time
   * mode has no threshold, so it has no defect volume.
   */
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
      {/*
        Placement is the host's, not this component's. The shell puts the
        gizmo above its log strip so the drawer never covers it; hard-coding
        `right`/`bottom` here overrode that and dropped it underneath. The
        class is the hook; `bench.css` positions it in the standalone bench
        and `shell.css` in the application.
      */}
      <canvas ref={gizmoRef} className="viewport-gizmo" width={96} height={96} aria-hidden="true" />
    </div>
  );
});

function applyEnvelope(mesh: EnvelopeMesh, envelope: EnvelopeData): void {
  mesh.setGeometry({
    positions: envelope.positions,
    normals: envelope.normals,
    indices: envelope.indices,
    scalars: envelope.scalars
  });
}

/** Three significant figures, which is what a probe callout has room for. */
function formatReading(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return '0';
  if (magnitude >= 1000 || magnitude < 0.001) return value.toExponential(1);
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 10) return value.toFixed(1);
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

/**
 * Keep only modes the client has an id for.
 *
 * The server catalogues a few engine fields the shell's menu has no entry for
 * (`feeding_path_lengths` and friends); passing them through would add a row
 * to the result menu that nothing on this side knows how to label or select.
 */
function withReportedModes(meta: GridMeta): GridMeta {
  const known = new Set<string>(RESULT_MODE_IDS);
  const filter = (modes: readonly ResultModeId[] | undefined): ResultModeId[] =>
    (modes ?? []).filter((mode) => known.has(mode));
  return { ...meta, modes: filter(meta.modes), emptyModes: filter(meta.emptyModes) };
}
