/**
 * The 3D results viewport.
 *
 * Renders the casting as a solid, environment-lit object and paints the
 * result field onto it continuously, by sampling a 3D texture of the dense
 * grid at each fragment's world position. There are no voxel cubes anywhere
 * in this file, and no result geometry at all -- the result is a color on the
 * casting's own surface, which is how commercial casting simulators present
 * it and what makes a 41,000-voxel field readable at a glance.
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
import { PointerEventTypes, Plane, Tools, Vector3 } from '@babylonjs/core';
import type { Mesh } from '@babylonjs/core';

import { isAbortError } from '../lib/gridApi';
import {
  RESULT_MODE_IDS,
  RESULT_MODE_UNITS,
  isTimeMode,
  type GridMeta,
  type ResultModeId,
  type ViewportHandle,
  type ViewportProps
} from '../lib/viewState';
import type { ProjectPart } from '../lib/api';
import { drawAxesGizmo } from './axesGizmo';
import { CameraController, unionBounds, type Bounds } from './cameraController';
import { RunResources, isFrameMode } from './fieldResources';
import {
  createPlaceholderLut,
  createPlaceholderVolume,
  FieldPaintPluginMaterial
} from './fieldPaintPlugin';
import { createPourMarker } from './markers';
import { applyPartAppearance, syncPartMeshes, type PartEntry } from './partMeshes';
import { createOriginAxes, createViewportScene, type ViewportScene } from './sceneSetup';

/** A pointerup further than this from its pointerdown was a camera drag, not a click. */
const CLICK_DRAG_THRESHOLD_PX = 4;

/** Playback advances the scrubber this far per second at 1x speed. */
const PLAYBACK_PERCENT_PER_SECOND = 12;

type SceneRefs = {
  viewport: ViewportScene;
  camera: CameraController;
  placeholderVolume: ReturnType<typeof createPlaceholderVolume>;
  placeholderLut: ReturnType<typeof createPlaceholderLut>;
  originAxes: Mesh;
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
 * Section planes, as material clip planes rather than scene clip planes.
 *
 * Scene-wide clipping would take the shop floor and the origin axes with it,
 * and the floor is the one thing that should stay put while the casting is
 * cut open. M1 cuts; capping the exposed faces with the interior field is M2.
 */
function sectionPlanes(section: ViewportProps['view']['section'], bounds: Bounds | null): Array<Plane | null> {
  if (!bounds) return [null, null, null];
  const axes = ['x', 'y', 'z'] as const;
  return axes.map((axis, index) => {
    const state = section[axis];
    if (!state.enabled) return null;
    const min = bounds.min.asArray()[index];
    const max = bounds.max.asArray()[index];
    const position = min + (max - min) * (Math.min(Math.max(state.position, 0), 100) / 100);
    const normal = new Vector3(axis === 'x' ? 1 : 0, axis === 'y' ? 1 : 0, axis === 'z' ? 1 : 0);
    // Babylon discards fragments on the +normal side of the plane, so the
    // default normal points up-axis and the low side of the cut survives.
    // Flip keeps the other half.
    const direction = state.flip ? normal.scale(-1) : normal;
    return Plane.FromPositionAndNormal(
      new Vector3(
        axis === 'x' ? position : 0,
        axis === 'y' ? position : 0,
        axis === 'z' ? position : 0
      ),
      direction
    );
  });
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

  // Every Babylon callback runs outside React's render cycle, so it reads the
  // latest props through this ref rather than closing over a stale render.
  const propsRef = useRef(props);
  propsRef.current = props;

  const { projectId, resultsAvailable, resultsVersion, view, ortho, parts, selectedPartId } = props;

  // ---------------------------------------------------------------------
  // Painting
  // ---------------------------------------------------------------------

  /**
   * Push the whole view state onto the GPU.
   *
   * One function rather than an effect per property: mode, colormap, scale,
   * time and cast-only all feed the same uniforms, and splitting them would
   * mean each one having to know what the others had already written.
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

    for (const entry of entries.values()) {
      const plugin = entry.plugin;
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
        plugin.state.strength = state.showResults ? 1 : 0;
        plugin.fieldTextureA = stack ? stack.textureA : (field?.texture ?? null);
        plugin.fieldTextureB = stack ? stack.textureB : (field?.texture ?? null);
        plugin.maskTexture = resources.maskTexture;
        plugin.lutTexture = resources.lut(state.colormap);
      } else {
        plugin.state.strength = 0;
      }

      entry.material.clipPlane = planes[0];
      entry.material.clipPlane2 = planes[1];
      entry.material.clipPlane3 = planes[2];
    }
  }, []);

  // ---------------------------------------------------------------------
  // Scene lifetime
  // ---------------------------------------------------------------------

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const viewport = createViewportScene(canvas);
    const camera = new CameraController(viewport.camera, viewport.engine, viewport.scene);
    const placeholderVolume = createPlaceholderVolume(viewport.scene);
    const placeholderLut = createPlaceholderLut(viewport.scene);
    const originAxes = createOriginAxes(viewport.scene, 4);
    sceneRef.current = { viewport, camera, placeholderVolume, placeholderLut, originAxes };

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
        // close to its pointerdown is a click.
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
        current.onProbe(pick?.hit && pick.pickedPoint ? probeAt(pick.pickedPoint, partId) : null);
        return;
      }

      if (current.tool === 'measure') {
        if (!pick?.hit || !pick.pickedPoint) return;
        const start = measurePendingRef.current;
        if (!start) {
          measurePendingRef.current = pick.pickedPoint.clone();
          return;
        }
        const end = pick.pickedPoint;
        measurePendingRef.current = null;
        current.onMeasure({
          start: [start.x, start.y, start.z],
          end: [end.x, end.y, end.z],
          distanceIn: Vector3.Distance(start, end)
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

    // The shell resizes the canvas by opening and closing panels, which never
    // fires a window resize. Watching the container is the only way to see it.
    const resizeObserver = new ResizeObserver(() => {
      viewport.engine.resize();
      camera.refreshProjection();
    });
    resizeObserver.observe(container);

    const entries = partEntriesRef.current;
    return () => {
      resizeObserver.disconnect();
      viewport.scene.onPointerObservable.remove(pointerObserver);
      viewport.scene.onBeforeRenderObservable.removeCallback(drawGizmo);
      resourcesRef.current?.dispose();
      resourcesRef.current = null;
      entries.clear();
      pourMarkerRef.current = null;
      viewport.dispose();
      sceneRef.current = null;
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

    const bounds = partBounds(propsRef.current.parts);
    boundsRef.current = bounds;
    refs.camera.setBounds(bounds);
    if (bounds) {
      refs.viewport.layoutFloor(bounds.center, bounds.size, bounds.min.z - bounds.size * 0.02);
      refs.originAxes.scaling.setAll(Math.max(bounds.size * 0.12, 0.5));
      // Only here. The set of parts or their placement changed, so the last
      // framing is no longer about the thing on screen.
      refs.camera.fit({ immediate: true });
    }
    paint();
    // `signature` is the dependency on purpose: it changes exactly when the
    // geometry does, and not when a slider moves.
  }, [signature, paint]);

  useEffect(() => {
    if (!sceneRef.current) return;
    applyPartAppearance(parts, partEntriesRef.current, {
      selectedPartId,
      opacity: Math.min(Math.max(view.modelOpacity, 0), 100) / 100,
      showModel: view.showModel,
      assemblySize: boundsRef.current?.size ?? 1
    });
    paint();
  }, [parts, selectedPartId, view.modelOpacity, view.showModel, paint]);

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
        paint();
      })
      .catch((error) => {
        if (isAbortError(error) || controller.signal.aborted) return;
        propsRef.current.onStatus(null);
        propsRef.current.onError(error instanceof Error ? error.message : 'Failed to load field');
      });
    return () => controller.abort();
  }, [view.mode, resourcesEpoch, paint]);

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
    view.section,
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

    const scaleMin = state.scaleMin ?? field.min;
    const scaleMax = state.scaleMax ?? field.max;
    const cutoff = scaleMin + (Math.min(Math.max(state.threshold, 0), 100) / 100) * (scaleMax - scaleMin);
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
      <canvas
        ref={gizmoRef}
        className="viewport-gizmo"
        width={96}
        height={96}
        aria-hidden="true"
        style={{ position: 'absolute', right: 12, bottom: 12, width: 96, height: 96, pointerEvents: 'none' }}
      />
    </div>
  );
});

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
