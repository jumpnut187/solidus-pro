/**
 * The modeling session and the history of modeled bodies behind it.
 *
 * Three commands, each SOLIDWORKS':
 *
 * - Extrude: choose a plane (a datum plane, a reference plane or a face),
 *   sketch closed profiles on it, exit the sketch into the Extrude property
 *   manager, watch the preview as the end condition and depth change, OK.
 *   Extrude Face is the same with the clicked face's outline as the sketch.
 *   The body is stored as a part of the chosen type (riser, chill, ...) with
 *   the feature that made it, so editing it later replays the same steps.
 * - Plane: a reference plane offset from a plane or face, or midway between
 *   two. Sketches on it follow it when its distance changes.
 * - Measure: pick vertices, edges and faces and read distances and angles.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type MutableRefObject } from 'react';

import { api, type PartType, type ProjectPart } from '../lib/api';
import { measure, type MeasurePick } from '../lib/modeling/measure';
import { ModelingClient, type BuiltBody } from '../lib/modeling/modelingClient';
import {
  defaultExtrude,
  entityOutline,
  featureOf,
  featureProblem,
  featureTargets,
  newEntityId,
  newPlaneId,
  nextFeatureSeq,
  nextPlaneName,
  planeFrame,
  referencePlanePreview,
  referencePlaneProblem,
  replaceEntity,
  resolveReferencePlane,
  signedArea,
  simplifyLoop,
  sketchProblem,
  sketchToWorld,
  worldToSketch,
  type ExtrudeFeature,
  type FaceOutline,
  type FeatureBody,
  type ModelingEvent,
  type ModelingViewState,
  type PickPurpose,
  type ReferencePlane,
  type ReferencePlaneSource,
  type SketchEntity,
  type SketchPlane,
  type SketchTool,
  type Vec2,
  type Vec3
} from '../lib/modeling/types';
import { partTypeLabel } from '../lib/partTypes';
import type { ViewportHandle } from '../lib/viewState';

export type ExtrudeStage = 'plane' | 'sketch' | 'feature';

export type ModelingSession = {
  stage: ExtrudeStage;
  /** What a click in the plane stage chooses: a sketch plane, or a face to extrude as it is. */
  purpose: Extract<PickPurpose, 'sketch' | 'faceExtrude'>;
  feature: ExtrudeFeature;
  body: FeatureBody & { autoName: boolean };
  /** The part being edited, or null for a new body. */
  editingPartId: string | null;
  tool: SketchTool;
  polygonSides: number;
  selectedEntityId: string | null;
  snap: boolean;
};

export type PlaneSession = {
  mode: 'offset' | 'mid';
  /** The chosen references: one for an offset plane, two for a mid plane. */
  refs: SketchPlane[];
  distance: number;
  flip: boolean;
  count: number;
  name: string;
  /** The reference plane being edited, or null for a new one. */
  editingId: string | null;
};

export type PreviewState =
  | { status: 'idle' }
  | { status: 'building' }
  | { status: 'ok'; body: BuiltBody; key: string }
  | { status: 'error'; error: string };

type Notify = {
  succeed: (message: string) => void;
  fail: (error: unknown, context?: string) => void;
  warn: (message: string) => void;
};

export type UseModelingOptions = {
  projectId: string | null;
  parts: ProjectPart[];
  setParts: (update: (current: ProjectPart[]) => ProjectPart[]) => void;
  setSelectedPartId: (id: string | null) => void;
  ensureProject: () => Promise<string>;
  viewportRef: MutableRefObject<ViewportHandle | null>;
  referencePlanes: ReferencePlane[];
  setReferencePlanes: (planes: ReferencePlane[]) => void;
  notify: Notify;
};

const PREVIEW_DEBOUNCE_MS = 140;

function autoName(partType: PartType, parts: ProjectPart[]): string {
  const label = partTypeLabel(partType);
  const used = new Set(parts.map((part) => part.name));
  for (let n = 1; ; n += 1) {
    const name = `${label} ${n}`;
    if (!used.has(name)) return name;
  }
}

function featureKey(feature: ExtrudeFeature, targets: ProjectPart[]): string {
  return JSON.stringify(feature) + '|' + targets.map((part) => `${part.id}:${part.filepath}`).join(',');
}

/** The world centre of all part meshes, for centring a sketch that has nothing in it yet. */
function assemblyCenter(parts: ProjectPart[]): Vec3 | null {
  let min: Vec3 | null = null;
  let max: Vec3 | null = null;
  for (const part of parts) {
    const bounds = part.meshData?.bounds;
    if (!bounds) continue;
    const lo: Vec3 = [bounds.x[0] + part.position_x, bounds.y[0] + part.position_y, bounds.z[0] + part.position_z];
    const hi: Vec3 = [bounds.x[1] + part.position_x, bounds.y[1] + part.position_y, bounds.z[1] + part.position_z];
    min = min ? [Math.min(min[0], lo[0]), Math.min(min[1], lo[1]), Math.min(min[2], lo[2])] : lo;
    max = max ? [Math.max(max[0], hi[0]), Math.max(max[1], hi[1]), Math.max(max[2], hi[2])] : hi;
  }
  return min && max ? [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2] : null;
}

/**
 * A face outline as closed sketch polylines on `plane`: each world loop
 * pressed onto the plane and cleared of the collinear points the
 * tessellation put on its straight edges. Orientation is left to the
 * profile's even-odd rule, so holes stay holes.
 */
export function outlineToEntities(face: FaceOutline, plane: SketchPlane): SketchEntity[] {
  const frame = planeFrame(plane);
  return face.loops
    .map((loop) => simplifyLoop(loop.map((point) => worldToSketch(frame, point)), 1e-4))
    .filter((points) => points.length >= 3 && Math.abs(signedArea(points)) > 1e-6)
    .map((points) => ({ id: newEntityId(), type: 'polyline' as const, points }));
}

export function useModeling(options: UseModelingOptions) {
  const { projectId, parts, setParts, setSelectedPartId, ensureProject, viewportRef, referencePlanes, setReferencePlanes, notify } =
    options;
  const [session, setSession] = useState<ModelingSession | null>(null);
  const [planeSession, setPlaneSession] = useState<PlaneSession | null>(null);
  const [measurePicks, setMeasurePicks] = useState<MeasurePick[] | null>(null);
  const [preview, setPreview] = useState<PreviewState>({ status: 'idle' });
  const [committing, setCommitting] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);

  const clientRef = useRef<ModelingClient | null>(null);
  const client = useCallback(() => {
    if (!clientRef.current) clientRef.current = new ModelingClient();
    return clientRef.current;
  }, []);
  useEffect(
    () => () => {
      clientRef.current?.dispose();
      clientRef.current = null;
    },
    []
  );

  const partsRef = useRef(parts);
  partsRef.current = parts;
  const sessionRef = useRef(session);
  sessionRef.current = session;
  const planeSessionRef = useRef(planeSession);
  planeSessionRef.current = planeSession;
  const notifyRef = useRef(notify);
  notifyRef.current = notify;
  const referencePlanesRef = useRef(referencePlanes);
  referencePlanesRef.current = referencePlanes;

  /** Reference planes as planes, carrying their ids. */
  const resolvedReferences = useMemo(
    () =>
      referencePlanes
        .map((ref) => {
          const plane = resolveReferencePlane(ref);
          return plane ? { id: ref.id, name: ref.name, plane } : null;
        })
        .filter((entry): entry is { id: string; name: string; plane: SketchPlane } => entry !== null),
    [referencePlanes]
  );

  // --- Camera --------------------------------------------------------------

  const lookAtSketch = useCallback(
    (plane: SketchPlane, entities: SketchEntity[]) => {
      const frame = planeFrame(plane);
      let focus2: Vec2 = [0, 0];
      const outline = entities.flatMap((entity) => entityOutline(entity));
      if (outline.length) {
        focus2 = [
          outline.reduce((sum, point) => sum + point[0], 0) / outline.length,
          outline.reduce((sum, point) => sum + point[1], 0) / outline.length
        ];
      } else {
        const center = assemblyCenter(partsRef.current);
        if (center) focus2 = worldToSketch(frame, center);
      }
      viewportRef.current?.normalTo({ normal: frame.normal, v: frame.v, focus: sketchToWorld(frame, focus2) });
    },
    [viewportRef]
  );

  // --- Session lifecycle ---------------------------------------------------

  const closeAll = useCallback(() => {
    setSession(null);
    setPlaneSession(null);
    setMeasurePicks(null);
    setPreview({ status: 'idle' });
  }, []);

  const startExtrude = useCallback(
    (plane?: SketchPlane, purpose: 'sketch' | 'faceExtrude' = 'sketch') => {
      closeAll();
      const current = partsRef.current;
      const seq = nextFeatureSeq(current);
      const feature = defaultExtrude(seq, plane ?? { kind: 'standard', id: 'top', offset: 0 });
      // A face extruded as it is is almost always a chill on that face; a
      // sketched boss is almost always a riser.
      const partType: PartType = purpose === 'faceExtrude' ? 'chill' : 'riser';
      setSession({
        stage: plane ? 'sketch' : 'plane',
        purpose,
        feature,
        body: { name: autoName(partType, current), partType, isOpenTop: false, autoName: true },
        editingPartId: null,
        tool: 'line',
        polygonSides: 6,
        selectedEntityId: null,
        snap: true
      });
      if (plane) lookAtSketch(plane, []);
    },
    [closeAll, lookAtSketch]
  );

  const editFeature = useCallback(
    (partId: string, stage: 'sketch' | 'feature') => {
      const part = partsRef.current.find((item) => item.id === partId);
      const feature = part ? featureOf(part) : null;
      if (!part || !feature) return;
      closeAll();
      const copy: ExtrudeFeature = JSON.parse(JSON.stringify(feature));
      setSession({
        stage,
        purpose: 'sketch',
        feature: copy,
        body: { name: part.name, partType: part.part_type, isOpenTop: part.is_open_top, autoName: false },
        editingPartId: part.id,
        tool: 'select',
        polygonSides: 6,
        selectedEntityId: null,
        snap: true
      });
      if (stage === 'sketch') lookAtSketch(copy.sketch.plane, copy.sketch.entities);
    },
    [closeAll, lookAtSketch]
  );

  const cancel = closeAll;

  const choosePlane = useCallback(
    (plane: SketchPlane, face: FaceOutline | null = null) => {
      const current = sessionRef.current;
      if (!current) return;
      if (current.purpose === 'faceExtrude') {
        if (!face) {
          notifyRef.current.warn('Click a flat face of the model to extrude it.');
          return;
        }
        if (!face.planar) {
          notifyRef.current.warn('That face is curved. Sketch on a plane over it and extrude Up To Next to follow it.');
          return;
        }
        const entities = outlineToEntities(face, plane);
        if (!entities.length) {
          notifyRef.current.warn('That face has no outline to extrude.');
          return;
        }
        setSession({
          ...current,
          stage: 'feature',
          feature: { ...current.feature, sketch: { plane, entities } }
        });
        return;
      }
      setSession({
        ...current,
        stage: 'sketch',
        feature: { ...current.feature, sketch: { ...current.feature.sketch, plane } }
      });
      lookAtSketch(plane, current.feature.sketch.entities);
    },
    [lookAtSketch]
  );

  const setPlaneOffset = useCallback((offset: number) => {
    if (!Number.isFinite(offset)) return;
    setSession((current) => {
      if (!current) return current;
      const plane = { ...current.feature.sketch.plane, offset };
      // Moved by hand, the plane is no longer the reference plane it came from.
      delete plane.refId;
      return { ...current, feature: { ...current.feature, sketch: { ...current.feature.sketch, plane } } };
    });
  }, []);

  const normalTo = useCallback(() => {
    if (session) lookAtSketch(session.feature.sketch.plane, session.feature.sketch.entities);
  }, [lookAtSketch, session]);

  const patchSession = useCallback(
    (patch: Partial<Pick<ModelingSession, 'tool' | 'polygonSides' | 'snap' | 'selectedEntityId'>>) => {
      setSession((current) => (current ? { ...current, ...patch } : current));
    },
    []
  );

  const setEntities = useCallback((update: (entities: SketchEntity[]) => SketchEntity[]) => {
    setSession((current) =>
      current
        ? {
            ...current,
            feature: {
              ...current.feature,
              sketch: { ...current.feature.sketch, entities: update(current.feature.sketch.entities) }
            }
          }
        : current
    );
  }, []);

  const updateEntity = useCallback(
    (entity: SketchEntity) => setEntities((entities) => replaceEntity(entities, entity)),
    [setEntities]
  );

  const deleteEntity = useCallback(
    (id: string) => {
      setEntities((entities) => entities.filter((item) => item.id !== id));
      setSession((current) => (current && current.selectedEntityId === id ? { ...current, selectedEntityId: null } : current));
    },
    [setEntities]
  );

  // --- Reference planes ----------------------------------------------------

  const startPlane = useCallback(
    (editId?: string) => {
      closeAll();
      const existing = editId ? referencePlanesRef.current.find((plane) => plane.id === editId) : null;
      if (existing) {
        const source = existing.source;
        setPlaneSession(
          source.kind === 'offset'
            ? { mode: 'offset', refs: [source.base], distance: source.distance, flip: source.flip, count: 1, name: existing.name, editingId: existing.id }
            : { mode: 'mid', refs: [source.a, source.b], distance: 1, flip: false, count: 1, name: existing.name, editingId: existing.id }
        );
        return;
      }
      setPlaneSession({
        mode: 'offset',
        refs: [],
        distance: 1,
        flip: false,
        count: 1,
        name: nextPlaneName(referencePlanesRef.current),
        editingId: null
      });
    },
    [closeAll]
  );

  const patchPlane = useCallback((patch: Partial<PlaneSession>) => {
    setPlaneSession((current) => {
      if (!current) return current;
      const next = { ...current, ...patch };
      if (patch.mode && patch.mode !== current.mode) next.refs = patch.mode === 'offset' ? current.refs.slice(0, 1) : current.refs;
      return next;
    });
  }, []);

  const planeSource = useCallback((plane: PlaneSession | null): ReferencePlaneSource | null => {
    if (!plane) return null;
    if (plane.mode === 'offset') {
      return plane.refs[0] ? { kind: 'offset', base: plane.refs[0], distance: plane.distance, flip: plane.flip } : null;
    }
    return plane.refs.length === 2 ? { kind: 'mid', a: plane.refs[0], b: plane.refs[1] } : null;
  }, []);

  const savePlanes = useCallback(
    async (planes: ReferencePlane[]) => {
      const previous = referencePlanesRef.current;
      setReferencePlanes(planes);
      try {
        const target = projectId ?? (await ensureProject());
        await api.saveReferencePlanes(target, planes);
      } catch (error) {
        setReferencePlanes(previous);
        throw error;
      }
    },
    [ensureProject, projectId, setReferencePlanes]
  );

  // --- Measure -------------------------------------------------------------

  const startMeasure = useCallback(() => {
    closeAll();
    setMeasurePicks([]);
  }, [closeAll]);

  const clearMeasure = useCallback(() => setMeasurePicks((current) => (current ? [] : current)), []);

  // --- Events from the viewport --------------------------------------------

  const handleEvent = useCallback(
    (event: ModelingEvent) => {
      switch (event.type) {
        case 'planePicked': {
          const planeEdit = planeSessionRef.current;
          if (planeEdit) {
            setPlaneSession({
              ...planeEdit,
              refs: planeEdit.mode === 'offset' ? [event.plane] : [...planeEdit.refs, event.plane].slice(-2)
            });
            return;
          }
          choosePlane(event.plane, event.face);
          break;
        }
        case 'entityCreated':
          setEntities((entities) => [...entities, event.entity]);
          patchSession({ selectedEntityId: event.entity.id });
          break;
        case 'entitySelected':
          patchSession({ selectedEntityId: event.id });
          break;
        case 'entityChanged':
          updateEntity(event.entity);
          break;
        case 'entitiesChanged':
          setEntities(() => event.entities);
          break;
        case 'entityDeleted':
          deleteEntity(event.id);
          break;
        case 'toolChanged':
          patchSession({ tool: event.tool });
          break;
        case 'convertFace': {
          const current = sessionRef.current;
          if (!current) return;
          const added = outlineToEntities(event.face, current.feature.sketch.plane);
          if (!added.length) {
            notifyRef.current.warn('That face has no outline to convert.');
            return;
          }
          setEntities((entities) => [...entities, ...added]);
          notifyRef.current.succeed(`Converted ${added.length} contour${added.length === 1 ? '' : 's'} from the face.`);
          break;
        }
        case 'measurePicked':
          setMeasurePicks((current) => {
            if (!current) return current;
            if (!event.pick) return [];
            return current.length >= 2 ? [event.pick] : [...current, event.pick];
          });
          break;
      }
    },
    [choosePlane, deleteEntity, patchSession, setEntities, updateEntity]
  );

  const exitSketch = useCallback(() => {
    if (!session) return;
    const problem = sketchProblem(session.feature.sketch.entities);
    if (problem) {
      notifyRef.current.warn(problem);
      return;
    }
    setSession({ ...session, stage: 'feature', selectedEntityId: null });
  }, [session]);

  const backToSketch = useCallback(() => {
    if (!session) return;
    setSession({ ...session, stage: 'sketch', tool: 'select' });
    lookAtSketch(session.feature.sketch.plane, session.feature.sketch.entities);
  }, [lookAtSketch, session]);

  const patchFeature = useCallback((patch: Partial<ExtrudeFeature>) => {
    setSession((current) => (current ? { ...current, feature: { ...current.feature, ...patch } } : current));
  }, []);

  const patchBody = useCallback((patch: Partial<FeatureBody>) => {
    setSession((current) => {
      if (!current) return current;
      const body = { ...current.body, ...patch };
      if ('name' in patch) body.autoName = false;
      if (patch.partType && current.body.autoName) body.name = autoName(patch.partType, partsRef.current);
      if (body.partType !== 'riser') body.isOpenTop = false;
      return { ...current, body };
    });
  }, []);

  // --- Preview -------------------------------------------------------------

  const previewFeature = session?.stage === 'feature' ? session.feature : null;
  const previewOwner = session?.editingPartId ?? null;
  useEffect(() => {
    if (!previewFeature) return;
    const problem = featureProblem(previewFeature);
    if (problem) {
      setPreview({ status: 'error', error: problem });
      return;
    }
    const targets = featureTargets(parts, previewFeature, previewOwner);
    const key = featureKey(previewFeature, targets);
    let cancelled = false;
    setPreview((current) => (current.status === 'ok' && current.key === key ? current : { status: 'building' }));
    const timer = window.setTimeout(() => {
      client()
        .build(previewFeature, targets, parts)
        .then((body) => {
          if (!cancelled) setPreview({ status: 'ok', body, key });
        })
        .catch((error: unknown) => {
          if (!cancelled) setPreview({ status: 'error', error: error instanceof Error ? error.message : String(error) });
        });
    }, PREVIEW_DEBOUNCE_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [previewFeature, previewOwner, parts, client]);

  // --- Storing bodies ------------------------------------------------------

  const store = useCallback(
    async (target: string, feature: ExtrudeFeature, body: FeatureBody, built: BuiltBody, partId: string | null, visible = true) => {
      const request = {
        name: body.name,
        part_type: body.partType,
        is_open_top: body.isOpenTop,
        visible,
        feature,
        mesh: { vertices: Array.from(built.positions), faces: Array.from(built.indices) }
      };
      if (partId) {
        const { part } = await api.updateFeatureBody(target, partId, request);
        setParts((current) => current.map((item) => (item.id === partId ? part : item)));
        return part;
      }
      const { part } = await api.createFeatureBody(target, request);
      setParts((current) => [...current, part]);
      return part;
    },
    [setParts]
  );

  /**
   * Rebuild every modeled body after `afterSeq`, in history order, each
   * against the assembly as the ones before it left it. This is what keeps a
   * chill extruded Up To Next sitting on a riser that was just made taller.
   */
  const rebuildFrom = useCallback(
    async (target: string, afterSeq: number, assembly: ProjectPart[]): Promise<number> => {
      let current = assembly;
      const queue = current
        .filter((part) => (featureOf(part)?.seq ?? -1) > afterSeq)
        .sort((a, b) => (featureOf(a)?.seq ?? 0) - (featureOf(b)?.seq ?? 0));
      let failed = 0;
      for (const part of queue) {
        const feature = featureOf(part);
        if (!feature) continue;
        try {
          const built = await client().build(feature, featureTargets(current, feature, part.id), current);
          const stored = await store(
            target,
            feature,
            { name: part.name, partType: part.part_type, isOpenTop: part.is_open_top },
            built,
            part.id,
            part.visible
          );
          current = current.map((item) => (item.id === part.id ? stored : item));
        } catch (error) {
          failed += 1;
          notifyRef.current.fail(error, `Could not rebuild "${part.name}"`);
        }
      }
      return failed;
    },
    [client, store]
  );

  const commit = useCallback(async () => {
    if (!session || committing) return;
    const problem = featureProblem(session.feature);
    if (problem) {
      notifyRef.current.warn(problem);
      return;
    }
    setCommitting(true);
    try {
      const assembly = partsRef.current;
      const targets = featureTargets(assembly, session.feature, session.editingPartId);
      const key = featureKey(session.feature, targets);
      const built =
        preview.status === 'ok' && preview.key === key ? preview.body : await client().build(session.feature, targets, assembly);
      const target = projectId ?? (await ensureProject());
      const stored = await store(target, session.feature, session.body, built, session.editingPartId);
      setSelectedPartId(stored.id);
      const verb = session.editingPartId ? 'Updated' : 'Created';
      notifyRef.current.succeed(`${verb} ${partTypeLabel(stored.part_type).toLowerCase()} "${stored.name}".`);
      for (const warning of built.warnings) notifyRef.current.warn(warning);
      closeAll();
      if (session.editingPartId) {
        const after = assembly.map((part) => (part.id === stored.id ? stored : part));
        await rebuildFrom(target, session.feature.seq, after);
      }
    } catch (error) {
      notifyRef.current.fail(error, 'Could not build the body');
    } finally {
      setCommitting(false);
    }
  }, [client, closeAll, committing, ensureProject, preview, projectId, rebuildFrom, session, setSelectedPartId, store]);

  const rebuildAll = useCallback(async () => {
    if (!projectId || rebuilding) return;
    setRebuilding(true);
    try {
      const failed = await rebuildFrom(projectId, 0, partsRef.current);
      if (!failed) notifyRef.current.succeed('Rebuilt every modeled body.');
    } finally {
      setRebuilding(false);
    }
  }, [projectId, rebuildFrom, rebuilding]);

  const commitPlane = useCallback(async () => {
    const plane = planeSessionRef.current;
    const source = planeSource(plane);
    const problem = referencePlaneProblem(source);
    if (!plane || !source || problem) {
      if (problem) notifyRef.current.warn(problem);
      return;
    }
    const existing = referencePlanesRef.current;
    let next: ReferencePlane[];
    const made: ReferencePlane[] = [];
    if (plane.editingId) {
      const edited: ReferencePlane = { id: plane.editingId, name: plane.name.trim() || 'Plane', source };
      next = existing.map((item) => (item.id === plane.editingId ? edited : item));
      made.push(edited);
    } else {
      const count = source.kind === 'offset' ? Math.max(1, Math.round(plane.count)) : 1;
      next = [...existing];
      for (let k = 1; k <= count; k += 1) {
        const name = k === 1 ? plane.name.trim() || nextPlaneName(next) : nextPlaneName(next);
        const ref: ReferencePlane = {
          id: newPlaneId(),
          name,
          source: source.kind === 'offset' ? { ...source, distance: source.distance * k } : source
        };
        next.push(ref);
        made.push(ref);
      }
    }
    try {
      await savePlanes(next);
      notifyRef.current.succeed(
        plane.editingId ? `Updated ${made[0].name}.` : `Created ${made.map((ref) => ref.name).join(', ')}.`
      );
      closeAll();
      if (plane.editingId) {
        // Sketches on the plane move with it, and their bodies are rebuilt.
        const moved = resolveReferencePlane(made[0]);
        if (!moved || !projectId) return;
        let earliest = Infinity;
        const assembly = partsRef.current.map((part) => {
          const feature = featureOf(part);
          if (!feature || feature.sketch.plane.refId !== plane.editingId) return part;
          earliest = Math.min(earliest, feature.seq);
          return { ...part, feature: { ...feature, sketch: { ...feature.sketch, plane: moved } } };
        });
        if (Number.isFinite(earliest)) await rebuildFrom(projectId, earliest - 1, assembly);
      }
    } catch (error) {
      notifyRef.current.fail(error, 'Could not save the plane');
    }
  }, [closeAll, planeSource, projectId, rebuildFrom, savePlanes]);

  const deletePlane = useCallback(
    async (id: string) => {
      try {
        await savePlanes(referencePlanesRef.current.filter((plane) => plane.id !== id));
      } catch (error) {
        notifyRef.current.fail(error, 'Could not delete the plane');
      }
    },
    [savePlanes]
  );

  // --- What the viewport draws ---------------------------------------------

  const measureResult = useMemo(() => (measurePicks ? measure(measurePicks) : null), [measurePicks]);
  const planePreview = useMemo(() => {
    const source = planeSource(planeSession);
    if (!planeSession || !source || referencePlaneProblem(source)) return [];
    return referencePlanePreview(source, planeSession.editingId ? 1 : planeSession.count, planeSession.name);
  }, [planeSession, planeSource]);

  const viewState = useMemo<ModelingViewState | null>(() => {
    const base = {
      plane: null,
      entities: [],
      tool: 'select' as SketchTool,
      polygonSides: 6,
      selectedEntityId: null,
      snap: true,
      preview: null,
      direction: null,
      referencePlanes: resolvedReferences,
      planePreview: [],
      chosenPlanes: [],
      measure: null
    };
    if (measurePicks) {
      return { ...base, stage: 'measure', purpose: 'sketch', measure: { picks: measurePicks, result: measureResult } };
    }
    if (planeSession) {
      // A plane being edited is drawn as its preview, not as itself.
      const references = planeSession.editingId
        ? resolvedReferences.filter((ref) => ref.id !== planeSession.editingId)
        : resolvedReferences;
      return {
        ...base,
        stage: 'plane',
        purpose: 'referencePlane',
        referencePlanes: references,
        planePreview,
        chosenPlanes: planeSession.refs
      };
    }
    if (!session) return null;
    const frame = planeFrame(session.feature.sketch.plane);
    const sign = session.feature.reverse ? -1 : 1;
    const direction: Vec3 = [frame.normal[0] * sign, frame.normal[1] * sign, frame.normal[2] * sign];
    return {
      ...base,
      stage: session.stage,
      purpose: session.purpose,
      plane: session.stage === 'plane' ? null : session.feature.sketch.plane,
      entities: session.feature.sketch.entities,
      tool: session.tool,
      polygonSides: session.polygonSides,
      selectedEntityId: session.selectedEntityId,
      snap: session.snap,
      preview:
        session.stage === 'feature' && preview.status === 'ok'
          ? { positions: preview.body.positions, indices: preview.body.indices, ok: true }
          : null,
      direction: session.stage === 'feature' ? direction : null
    };
  }, [measurePicks, measureResult, planePreview, planeSession, preview, resolvedReferences, session]);

  const active = session !== null || planeSession !== null || measurePicks !== null;

  return {
    session,
    planeSession,
    measurePicks,
    measureResult,
    planePreview,
    preview,
    committing,
    rebuilding,
    viewState,
    active,
    referencePlanes,
    resolvedReferences,
    startExtrude,
    editFeature,
    cancel,
    choosePlane,
    setPlaneOffset,
    normalTo,
    patchSession,
    updateEntity,
    deleteEntity,
    handleEvent,
    exitSketch,
    backToSketch,
    patchFeature,
    patchBody,
    commit,
    rebuildAll,
    startPlane,
    patchPlane,
    commitPlane,
    deletePlane,
    startMeasure,
    clearMeasure
  };
}

export type ModelingController = ReturnType<typeof useModeling>;
