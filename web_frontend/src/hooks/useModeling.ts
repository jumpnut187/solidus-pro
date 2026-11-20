import { useCallback, useEffect, useMemo, useRef, useState, type MutableRefObject } from 'react';

import { api, type MeshData, type PartType, type ProjectPart } from '../lib/api';
import { partWorldMatrix, type Box3 } from '../lib/modeling/kernel';
import { measure, type MeasurePick } from '../lib/modeling/measure';
import { ModelingClient, partMatrix, type BuiltBody, type CutBody } from '../lib/modeling/modelingClient';
import {
  bodySeq,
  cutCandidates,
  cutsOnPart,
  defaultCut,
  defaultExtrude,
  entityOutline,
  featureOf,
  featureProblem,
  featureTargets,
  newEntityId,
  newPlaneId,
  nextCutName,
  nextFeatureSeq,
  nextPlaneName,
  planeFrame,
  referencePlanePreview,
  referencePlaneProblem,
  pointInContours,
  regionSelected,
  replaceEntity,
  resolveReferencePlane,
  signedArea,
  simplifyLoop,
  sketchProblem,
  sketchToWorld,
  worldToSketch,
  type CutFeature,
  type ExtrudeFeature,
  type FaceOutline,
  type FeatureBody,
  type ModelFeature,
  type ModelingEvent,
  type ModelingViewState,
  type PickPurpose,
  type ReferencePlane,
  type ReferencePlaneSource,
  type SketchEntity,
  type SketchPlane,
  type SketchRegion,
  type SketchTool,
  type Vec2,
  type Vec3
} from '../lib/modeling/types';
import { partTypeLabel } from '../lib/partTypes';
import type { ViewportHandle } from '../lib/viewState';

export type ExtrudeStage = 'plane' | 'sketch' | 'feature';

export type ModelingSession = {
  stage: ExtrudeStage;
  purpose: Extract<PickPurpose, 'sketch' | 'faceExtrude'>;
  feature: ModelFeature;
  body: FeatureBody & { autoName: boolean };
  editingPartId: string | null;
  editingCutId: string | null;
  autoDirection?: boolean;
  tool: SketchTool;
  polygonSides: number;
  selectedEntityIds: string[];
  snap: boolean;
};

export type PlaneSession = {
  mode: 'offset' | 'mid';
  refs: SketchPlane[];
  distance: number;
  flip: boolean;
  count: number;
  name: string;
  editingId: string | null;
};

export type PreviewState =
  | { status: 'idle' }
  | { status: 'building' }
  | { status: 'ok'; body: BuiltBody; key: string; removed?: Record<string, number> }
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
  cutFeatures: CutFeature[];
  setCutFeatures: (cuts: CutFeature[]) => void;
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

function featureKey(feature: ModelFeature, targets: ProjectPart[]): string {
  return (
    JSON.stringify(feature) +
    '|' +
    targets.map((part) => `${part.id}:${part.filepath}:${part.base_filepath ?? ''}`).join(',')
  );
}

function positionsBox(positions: Float32Array): Box3 {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], positions[i + axis]);
      max[axis] = Math.max(max[axis], positions[i + axis]);
    }
  }
  return { min, max };
}

function meshPayload(body: { positions: Float32Array; indices: Uint32Array }) {
  return { vertices: Array.from(body.positions), faces: Array.from(body.indices) };
}

function assemblyCenter(parts: ProjectPart[]): Vec3 | null {
  let min: Vec3 | null = null;
  let max: Vec3 | null = null;
  for (const part of parts) {
    const box = worldBounds(part);
    if (!box) continue;
    const lo: Vec3 = box.min;
    const hi: Vec3 = box.max;
    min = min ? [Math.min(min[0], lo[0]), Math.min(min[1], lo[1]), Math.min(min[2], lo[2])] : lo;
    max = max ? [Math.max(max[0], hi[0]), Math.max(max[1], hi[1]), Math.max(max[2], hi[2])] : hi;
  }
  return min && max ? [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2] : null;
}

export function outlineToEntities(face: FaceOutline, plane: SketchPlane): SketchEntity[] {
  const frame = planeFrame(plane);
  return face.loops
    .map((loop) => simplifyLoop(loop.map((point) => worldToSketch(frame, point)), 1e-4))
    .filter((points) => points.length >= 3 && Math.abs(signedArea(points)) > 1e-6)
    .map((points) => ({ id: newEntityId(), type: 'polyline' as const, points }));
}

function worldBounds(part: ProjectPart): Box3 | null {
  const bounds = part.meshData?.bounds;
  if (!bounds) return null;
  const m = partWorldMatrix(
    [part.position_x, part.position_y, part.position_z],
    [part.rotation_x, part.rotation_y, part.rotation_z],
    part.scale
  );
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const x of bounds.x) {
    for (const y of bounds.y) {
      for (const z of bounds.z) {
        const p = [m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14]];
        for (let axis = 0; axis < 3; axis += 1) {
          min[axis] = Math.min(min[axis], p[axis]);
          max[axis] = Math.max(max[axis], p[axis]);
        }
      }
    }
  }
  return { min, max };
}

function cutDirection(
  parts: ProjectPart[],
  plane: SketchPlane
): Pick<CutFeature, 'reverse' | 'endCondition'> | null {
  const frame = planeFrame(plane);
  let ahead = -Infinity;
  let behind = Infinity;
  for (const part of parts) {
    const box = worldBounds(part);
    if (!box) continue;
    for (const x of [box.min[0], box.max[0]]) {
      for (const y of [box.min[1], box.max[1]]) {
        for (const z of [box.min[2], box.max[2]]) {
          const s =
            (x - frame.origin[0]) * frame.normal[0] +
            (y - frame.origin[1]) * frame.normal[1] +
            (z - frame.origin[2]) * frame.normal[2];
          ahead = Math.max(ahead, s);
          behind = Math.min(behind, s);
        }
      }
    }
  }
  if (!Number.isFinite(ahead)) return null;
  const tolerance = 1e-4 * Math.max(ahead - behind, 1);
  if (ahead > tolerance && behind < -tolerance) return { reverse: true, endCondition: 'throughAllBoth' };
  return { reverse: ahead <= tolerance, endCondition: 'throughAll' };
}

function boundsTouch(a: ProjectPart, b: ProjectPart, pad = 1e-3): boolean {
  const boxA = worldBounds(a);
  const boxB = worldBounds(b);
  if (!boxA || !boxB) return true;
  for (let axis = 0; axis < 3; axis += 1) {
    if (boxA.max[axis] + pad < boxB.min[axis] || boxB.max[axis] + pad < boxA.min[axis]) return false;
  }
  return true;
}

export function useModeling(options: UseModelingOptions) {
  const {
    projectId,
    parts,
    setParts,
    setSelectedPartId,
    ensureProject,
    viewportRef,
    referencePlanes,
    setReferencePlanes,
    cutFeatures,
    setCutFeatures,
    notify
  } = options;
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
  const cutsRef = useRef(cutFeatures);
  cutsRef.current = cutFeatures;

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
      const seq = nextFeatureSeq(current, cutsRef.current);
      const feature = defaultExtrude(seq, plane ?? { kind: 'standard', id: 'top', offset: 0 });
      const partType: PartType = purpose === 'faceExtrude' ? 'chill' : 'riser';
      setSession({
        stage: plane ? 'sketch' : 'plane',
        purpose,
        feature,
        body: { name: autoName(partType, current), partType, isOpenTop: false, autoName: true },
        editingPartId: null,
        editingCutId: null,
        tool: 'line',
        polygonSides: 6,
        selectedEntityIds: [],
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
        editingCutId: null,
        tool: 'select',
        polygonSides: 6,
        selectedEntityIds: [],
        snap: true
      });
      if (stage === 'sketch') lookAtSketch(copy.sketch.plane, copy.sketch.entities);
    },
    [closeAll, lookAtSketch]
  );

  const startCut = useCallback(
    (plane?: SketchPlane) => {
      closeAll();
      const current = partsRef.current;
      const cuts = cutsRef.current;
      const feature = defaultCut(nextFeatureSeq(current, cuts), plane ?? { kind: 'standard', id: 'top', offset: 0 }, nextCutName(cuts));
      feature.partIds = cutCandidates(current, feature)
        .filter((part) => part.visible !== false)
        .map((part) => part.id);
      setSession({
        stage: plane ? 'sketch' : 'plane',
        purpose: 'sketch',
        feature,
        body: { name: feature.name, partType: 'cast', isOpenTop: false, autoName: false },
        editingPartId: null,
        editingCutId: null,
        autoDirection: true,
        tool: 'line',
        polygonSides: 6,
        selectedEntityIds: [],
        snap: true
      });
      if (plane) lookAtSketch(plane, []);
    },
    [closeAll, lookAtSketch]
  );

  const editCut = useCallback(
    (cutId: string, stage: 'sketch' | 'feature') => {
      const cut = cutsRef.current.find((item) => item.id === cutId);
      if (!cut) return;
      closeAll();
      const copy: CutFeature = JSON.parse(JSON.stringify(cut));
      setSession({
        stage,
        purpose: 'sketch',
        feature: copy,
        body: { name: cut.name, partType: 'cast', isOpenTop: false, autoName: false },
        editingPartId: null,
        editingCutId: cut.id,
        tool: 'select',
        polygonSides: 6,
        selectedEntityIds: [],
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
      delete plane.refId;
      return { ...current, feature: { ...current.feature, sketch: { ...current.feature.sketch, plane } } };
    });
  }, []);

  const normalTo = useCallback(() => {
    if (session) lookAtSketch(session.feature.sketch.plane, session.feature.sketch.entities);
  }, [lookAtSketch, session]);

  const patchSession = useCallback(
    (patch: Partial<Pick<ModelingSession, 'tool' | 'polygonSides' | 'snap' | 'selectedEntityIds'>>) => {
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

  const deleteEntities = useCallback(
    (ids: string[]) => {
      const gone = new Set(ids);
      setEntities((entities) => entities.filter((item) => !gone.has(item.id)));
      setSession((current) =>
        current ? { ...current, selectedEntityIds: current.selectedEntityIds.filter((id) => !gone.has(id)) } : current
      );
    },
    [setEntities]
  );

  const deleteEntity = useCallback((id: string) => deleteEntities([id]), [deleteEntities]);

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

  const startMeasure = useCallback(() => {
    closeAll();
    setMeasurePicks([]);
  }, [closeAll]);

  const clearMeasure = useCallback(() => setMeasurePicks((current) => (current ? [] : current)), []);

  const [regionState, setRegionState] = useState<{ key: string; regions: SketchRegion[] } | null>(null);
  const regionStateRef = useRef(regionState);
  regionStateRef.current = regionState;
  const regionSketch = session?.stage === 'feature' ? session.feature.sketch : null;
  const regionKey = regionSketch ? JSON.stringify(regionSketch.entities) : null;
  useEffect(() => {
    if (!regionSketch || !regionKey) return;
    if (regionStateRef.current?.key === regionKey) return;
    let cancelled = false;
    client()
      .regions(regionSketch)
      .then((regions) => {
        if (cancelled) return;
        setRegionState({ key: regionKey, regions });
        setSession((current) => {
          const seeds = current?.feature.regions;
          if (!current || !seeds) return current;
          const kept = seeds.filter((seed) => regions.some((region) => pointInContours(region.contours, seed)));
          if (kept.length === seeds.length) return current;
          return { ...current, feature: { ...current.feature, regions: kept.length ? kept : null } as ModelFeature };
        });
      })
      .catch(() => {
        if (!cancelled) setRegionState({ key: regionKey, regions: [] });
      });
    return () => {
      cancelled = true;
    };
  }, [regionKey, client]);
  const currentRegions = regionState && regionState.key === regionKey ? regionState.regions : null;

  const toggleRegion = useCallback((point: Vec2) => {
    const regions = regionStateRef.current?.regions;
    setSession((current) => {
      if (!current || current.stage !== 'feature' || !regions?.length) return current;
      const hit = regions.find((region) => pointInContours(region.contours, point));
      if (!hit) return current;
      const selected = regions.filter((region) => regionSelected(region, current.feature.regions));
      const next = selected.includes(hit) ? selected.filter((region) => region !== hit) : [...selected, hit];
      return { ...current, feature: { ...current.feature, regions: next.map((region) => region.seed) } as ModelFeature };
    });
  }, []);

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
          patchSession({ selectedEntityIds: [event.entity.id] });
          break;
        case 'entitySelected':
          patchSession({ selectedEntityIds: event.ids });
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
        case 'entitiesDeleted':
          deleteEntities(event.ids);
          break;
        case 'regionToggled':
          toggleRegion(event.point);
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
        case 'depthChanged':
          if (Number.isFinite(event.depth) && event.depth > 1e-4) {
            setSession((current) =>
              current ? { ...current, feature: { ...current.feature, depth: event.depth } as ModelFeature } : current
            );
          }
          break;
        case 'measurePicked':
          setMeasurePicks((current) => {
            if (!current) return current;
            if (!event.pick) return [];
            return current.length >= 2 ? [event.pick] : [...current, event.pick];
          });
          break;
      }
    },
    [choosePlane, deleteEntities, deleteEntity, patchSession, setEntities, toggleRegion, updateEntity]
  );

  const exitSketch = useCallback(() => {
    if (!session) return;
    const problem = sketchProblem(session.feature.sketch.entities);
    if (problem) {
      notifyRef.current.warn(problem);
      return;
    }
    let feature = session.feature;
    if (feature.kind === 'cut' && session.autoDirection) {
      const scope = partsRef.current.filter((part) => (feature as CutFeature).partIds.includes(part.id));
      const direction = cutDirection(scope, feature.sketch.plane);
      if (direction) feature = { ...feature, ...direction };
    }
    setSession({ ...session, feature, stage: 'feature', selectedEntityIds: [], autoDirection: false });
  }, [session]);

  const backToSketch = useCallback(() => {
    if (!session) return;
    setSession({ ...session, stage: 'sketch', tool: 'select' });
    lookAtSketch(session.feature.sketch.plane, session.feature.sketch.entities);
  }, [lookAtSketch, session]);

  const patchFeature = useCallback((patch: Partial<ExtrudeFeature> | Partial<CutFeature>) => {
    setSession((current) =>
      current ? { ...current, feature: { ...current.feature, ...patch, kind: current.feature.kind } as ModelFeature } : current
    );
  }, []);

  const resetRegions = useCallback(() => {
    setSession((current) => (current ? { ...current, feature: { ...current.feature, regions: null } as ModelFeature } : current));
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

  const baseMeshes = useRef(new Map<string, Promise<MeshData>>());
  useEffect(() => baseMeshes.current.clear(), [projectId]);

  const baseOf = useCallback(async (target: string, part: ProjectPart): Promise<ProjectPart> => {
    const basePath = part.base_filepath;
    if (!basePath) return part;
    let pending = baseMeshes.current.get(basePath);
    if (!pending) {
      pending = api.partBaseMesh(target, part.id);
      baseMeshes.current.set(basePath, pending);
      pending.catch(() => baseMeshes.current.delete(basePath));
    }
    return { ...part, filepath: basePath, meshData: await pending };
  }, []);

  const previewFeature = session?.stage === 'feature' ? session.feature : null;
  const previewOwner = session?.editingPartId ?? null;
  useEffect(() => {
    if (!previewFeature) return;
    const feature = previewFeature;
    const problem = featureProblem(feature);
    if (problem) {
      setPreview({ status: 'error', error: problem });
      return;
    }
    const targets =
      feature.kind === 'cut'
        ? cutCandidates(parts, feature).filter((part) => feature.partIds.includes(part.id))
        : featureTargets(parts, feature, previewOwner);
    const key = featureKey(feature, targets);
    let cancelled = false;
    setPreview((current) => (current.status === 'ok' && current.key === key ? current : { status: 'building' }));
    const run = async (): Promise<PreviewState> => {
      if (feature.kind !== 'cut') return { status: 'ok', body: await client().build(feature, targets, parts), key };
      const bases = projectId ? await Promise.all(targets.map((part) => baseOf(projectId, part))) : targets;
      const cut = await client().previewCut(feature, bases, [...parts, ...bases]);
      return {
        status: 'ok',
        key,
        removed: cut.removed,
        body: {
          positions: cut.positions,
          indices: cut.indices,
          volume: cut.volume,
          triangles: cut.indices.length / 3,
          bbox: positionsBox(cut.positions),
          warnings: cut.warnings,
          elapsedMs: cut.elapsedMs
        }
      };
    };
    const timer = window.setTimeout(() => {
      run()
        .then((state) => {
          if (!cancelled) setPreview(state);
        })
        .catch((error: unknown) => {
          if (!cancelled) setPreview({ status: 'error', error: error instanceof Error ? error.message : String(error) });
        });
    }, PREVIEW_DEBOUNCE_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [previewFeature, previewOwner, parts, client, projectId, baseOf]);

  const store = useCallback(
    async (
      target: string,
      feature: ExtrudeFeature,
      body: FeatureBody,
      built: BuiltBody,
      partId: string | null,
      visible = true,
      cut: CutBody | null = null
    ) => {
      const request = {
        name: body.name,
        part_type: body.partType,
        is_open_top: body.isOpenTop,
        visible,
        feature,
        mesh: meshPayload(built),
        cut_mesh: cut?.changed ? meshPayload(cut) : null
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

  const cutBuilt = useCallback(
    async (part: ProjectPart, built: BuiltBody, cuts: CutFeature[]): Promise<CutBody | null> => {
      const applicable = cutsOnPart(cuts, part);
      if (!applicable.length) return null;
      const out = await client().cut(
        { name: part.name, positions: built.positions, indices: built.indices, matrix: partMatrix(part) },
        applicable
      );
      for (const warning of out.warnings) notifyRef.current.warn(warning);
      return out.changed ? out : null;
    },
    [client]
  );

  const recut = useCallback(
    async (target: string, part: ProjectPart, cuts: CutFeature[]): Promise<ProjectPart> => {
      const applicable = cutsOnPart(cuts, part);
      let result: CutBody | null = null;
      if (applicable.length) {
        const base = await baseOf(target, part);
        if (!base.meshData) throw new Error(`"${part.name}" has no geometry loaded yet.`);
        result = await client().cut(
          {
            name: part.name,
            positions: new Float32Array(base.meshData.vertices),
            indices: new Uint32Array(base.meshData.faces),
            matrix: partMatrix(part)
          },
          applicable
        );
        for (const warning of result.warnings) notifyRef.current.warn(warning);
      }
      if (!result?.changed && !part.base_filepath) return part;
      const { part: stored } = await api.storeCutBody(target, part.id, result?.changed ? meshPayload(result) : null);
      setParts((current) => current.map((item) => (item.id === part.id ? stored : item)));
      return stored;
    },
    [baseOf, client, setParts]
  );

  const saveCuts = useCallback(
    async (target: string, cuts: CutFeature[]) => {
      const previous = cutsRef.current;
      cutsRef.current = cuts;
      setCutFeatures(cuts);
      try {
        await api.saveCutFeatures(target, cuts);
      } catch (error) {
        cutsRef.current = previous;
        setCutFeatures(previous);
        throw error;
      }
    },
    [setCutFeatures]
  );

  const recutBodies = useCallback(
    async (target: string, partIds: Set<string>, cuts: CutFeature[], assembly: ProjectPart[]) => {
      let current = assembly;
      let failed = 0;
      for (const part of assembly.filter((item) => partIds.has(item.id))) {
        try {
          const stored = await recut(target, part, cuts);
          current = current.map((item) => (item.id === part.id ? stored : item));
        } catch (error) {
          failed += 1;
          notifyRef.current.fail(error, `Could not cut "${part.name}"`);
        }
      }
      return { assembly: current, failed };
    },
    [recut]
  );

  const rebuildFrom = useCallback(
    async (target: string, afterSeq: number, assembly: ProjectPart[], cuts: CutFeature[] = cutsRef.current): Promise<number> => {
      let current = assembly;
      const queue = current
        .filter((part) => (featureOf(part)?.seq ?? -1) > afterSeq)
        .sort((a, b) => (featureOf(a)?.seq ?? 0) - (featureOf(b)?.seq ?? 0));
      let failed = 0;
      for (const part of queue) {
        const feature = featureOf(part);
        if (!feature) continue;
        try {
          if (feature.endCondition === 'upToBody' && !current.some((item) => item.id === feature.targetPartId)) {
            throw new Error('It extrudes up to a body that no longer exists. Edit it and choose another body.');
          }
          const built = await client().build(feature, featureTargets(current, feature, part.id), current);
          const cut = await cutBuilt(part, built, cuts);
          const stored = await store(
            target,
            feature,
            { name: part.name, partType: part.part_type, isOpenTop: part.is_open_top },
            built,
            part.id,
            part.visible,
            cut
          );
          current = current.map((item) => (item.id === part.id ? stored : item));
        } catch (error) {
          failed += 1;
          notifyRef.current.fail(error, `Could not rebuild "${part.name}"`);
        }
      }
      return failed;
    },
    [client, cutBuilt, store]
  );

  const commitCut = useCallback(
    async (cut: CutFeature, editingCutId: string | null) => {
      const cuts = cutsRef.current;
      const previous = editingCutId ? (cuts.find((item) => item.id === editingCutId) ?? null) : null;
      const next = previous ? cuts.map((item) => (item.id === previous.id ? cut : item)) : [...cuts, cut];
      const target = projectId ?? (await ensureProject());
      await saveCuts(target, next);
      const touched = new Set([...cut.partIds, ...(previous?.partIds ?? [])]);
      closeAll();
      setRebuilding(true);
      try {
        const { assembly, failed } = await recutBodies(target, touched, next, partsRef.current);
        if (!failed) notifyRef.current.succeed(`${previous ? 'Updated' : 'Created'} ${cut.name}.`);
        if (previous) await rebuildFrom(target, cut.seq, assembly, next);
      } finally {
        setRebuilding(false);
      }
    },
    [closeAll, ensureProject, projectId, rebuildFrom, recutBodies, saveCuts]
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
      if (session.feature.kind === 'cut') {
        await commitCut(session.feature, session.editingCutId);
        return;
      }
      const feature = session.feature;
      const assembly = partsRef.current;
      const targets = featureTargets(assembly, feature, session.editingPartId);
      const key = featureKey(feature, targets);
      const built =
        preview.status === 'ok' && preview.key === key ? preview.body : await client().build(feature, targets, assembly);
      const target = projectId ?? (await ensureProject());
      const editing = session.editingPartId ? assembly.find((part) => part.id === session.editingPartId) : undefined;
      const cut = editing ? await cutBuilt(editing, built, cutsRef.current) : null;
      const stored = await store(target, feature, session.body, built, session.editingPartId, true, cut);
      setSelectedPartId(stored.id);
      const verb = session.editingPartId ? 'Updated' : 'Created';
      notifyRef.current.succeed(`${verb} ${partTypeLabel(stored.part_type).toLowerCase()} "${stored.name}".`);
      for (const warning of built.warnings) notifyRef.current.warn(warning);
      closeAll();
      if (session.editingPartId) {
        const after = assembly.map((part) => (part.id === stored.id ? stored : part));
        await rebuildFrom(target, feature.seq, after);
      }
    } catch (error) {
      notifyRef.current.fail(error, session.feature.kind === 'cut' ? 'Could not cut the bodies' : 'Could not build the body');
    } finally {
      setCommitting(false);
    }
  }, [client, closeAll, commitCut, committing, cutBuilt, ensureProject, preview, projectId, rebuildFrom, session, setSelectedPartId, store]);

  const deleteCut = useCallback(
    async (cutId: string) => {
      const cut = cutsRef.current.find((item) => item.id === cutId);
      if (!cut || !projectId || rebuilding) return;
      setRebuilding(true);
      try {
        const next = cutsRef.current.filter((item) => item.id !== cutId);
        await saveCuts(projectId, next);
        const { assembly, failed } = await recutBodies(projectId, new Set(cut.partIds), next, partsRef.current);
        const rebuildFailed = await rebuildFrom(projectId, cut.seq, assembly, next);
        if (!failed && !rebuildFailed) notifyRef.current.succeed(`Deleted ${cut.name}.`);
      } catch (error) {
        notifyRef.current.fail(error, `Could not delete ${cut.name}`);
      } finally {
        setRebuilding(false);
      }
    },
    [projectId, rebuildFrom, rebuilding, recutBodies, saveCuts]
  );

  const rebuildAll = useCallback(async () => {
    if (!projectId || rebuilding) return;
    setRebuilding(true);
    try {
      const cuts = cutsRef.current;
      const uploaded = partsRef.current.filter(
        (part) => !featureOf(part) && (part.base_filepath || cutsOnPart(cuts, part).length)
      );
      const { assembly, failed } = await recutBodies(projectId, new Set(uploaded.map((part) => part.id)), cuts, partsRef.current);
      const rebuildFailed = await rebuildFrom(projectId, 0, assembly, cuts);
      if (!failed && !rebuildFailed) notifyRef.current.succeed('Rebuilt every modeled body and cut.');
    } finally {
      setRebuilding(false);
    }
  }, [projectId, rebuildFrom, rebuilding, recutBodies]);

  const removalQueue = useRef<Promise<void>>(Promise.resolve());
  const removedIds = useRef(new Set<string>());

  const rebuildDependents = useCallback(
    (partId: string) => {
      const before = partsRef.current;
      const removed = before.find((part) => part.id === partId);
      if (!removed || !projectId) return;
      removedIds.current.add(partId);
      const dependents = before.filter((part) => {
        const feature = featureOf(part);
        return (
          part.id !== partId &&
          feature !== null &&
          featureTargets(before, feature, part.id).some((target) => target.id === partId) &&
          (feature.endCondition === 'upToBody' || boundsTouch(part, removed))
        );
      });
      if (!dependents.length) return;
      const earliest = Math.min(...dependents.map((part) => featureOf(part)?.seq ?? Infinity));
      const target = projectId;
      removalQueue.current = removalQueue.current.then(async () => {
        setRebuilding(true);
        try {
          const assembly = partsRef.current.filter((part) => !removedIds.current.has(part.id));
          const failed = await rebuildFrom(target, earliest - 1, assembly);
          if (!failed) {
            const names = dependents.map((part) => `"${part.name}"`).join(', ');
            notifyRef.current.succeed(`Rebuilt ${names} without "${removed.name}".`);
          }
        } finally {
          setRebuilding(false);
        }
      });
    },
    [projectId, rebuildFrom]
  );

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
        const moved = resolveReferencePlane(made[0]);
        if (!moved || !projectId) return;
        let earliest = Infinity;
        let assembly = partsRef.current.map((part) => {
          const feature = featureOf(part);
          if (!feature || feature.sketch.plane.refId !== plane.editingId) return part;
          earliest = Math.min(earliest, feature.seq);
          return { ...part, feature: { ...feature, sketch: { ...feature.sketch, plane: moved } } };
        });
        const cuts = cutsRef.current;
        const movedCuts = cuts.filter((cut) => cut.sketch.plane.refId === plane.editingId);
        let nextCuts = cuts;
        if (movedCuts.length) {
          nextCuts = cuts.map((cut) =>
            cut.sketch.plane.refId === plane.editingId ? { ...cut, sketch: { ...cut.sketch, plane: moved } } : cut
          );
          await saveCuts(projectId, nextCuts);
          const reached = new Set(movedCuts.flatMap((cut) => cut.partIds));
          const uploaded = new Set(assembly.filter((part) => reached.has(part.id) && !featureOf(part)).map((part) => part.id));
          assembly = (await recutBodies(projectId, uploaded, nextCuts, assembly)).assembly;
          for (const cut of movedCuts) earliest = Math.min(earliest, cut.seq);
          const firstModeled = Math.min(
            ...assembly.filter((part) => reached.has(part.id) && featureOf(part)).map((part) => bodySeq(part))
          );
          earliest = Math.min(earliest, firstModeled);
        }
        if (Number.isFinite(earliest)) await rebuildFrom(projectId, earliest - 1, assembly, nextCuts);
      }
    } catch (error) {
      notifyRef.current.fail(error, 'Could not save the plane');
    }
  }, [closeAll, planeSource, projectId, rebuildFrom, recutBodies, saveCuts, savePlanes]);

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
      selectedEntityIds: [],
      snap: true,
      preview: null,
      direction: null,
      extrudeDepth: null,
      regions: null,
      referencePlanes: resolvedReferences,
      planePreview: [],
      chosenPlanes: [],
      measure: null
    };
    if (measurePicks) {
      return { ...base, stage: 'measure', purpose: 'sketch', measure: { picks: measurePicks, result: measureResult } };
    }
    if (planeSession) {
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
      selectedEntityIds: session.selectedEntityIds,
      snap: session.snap,
      preview:
        session.stage === 'feature' && preview.status === 'ok'
          ? {
              positions: preview.body.positions,
              indices: preview.body.indices,
              ok: true,
              cut: session.feature.kind === 'cut'
            }
          : null,
      direction: session.stage === 'feature' ? direction : null,
      regions:
        session.stage === 'feature' && currentRegions && (currentRegions.length > 1 || session.feature.regions)
          ? currentRegions.map((region) => ({ ...region, selected: regionSelected(region, session.feature.regions) }))
          : null,
      extrudeDepth:
        session.stage === 'feature' &&
        (session.feature.endCondition === 'blind' || session.feature.endCondition === 'midplane')
          ? { value: session.feature.depth, midplane: session.feature.endCondition === 'midplane' }
          : null
    };
  }, [currentRegions, measurePicks, measureResult, planePreview, planeSession, preview, resolvedReferences, session]);

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
    cutFeatures,
    startExtrude,
    editFeature,
    startCut,
    editCut,
    deleteCut,
    cancel,
    choosePlane,
    setPlaneOffset,
    normalTo,
    patchSession,
    updateEntity,
    deleteEntity,
    deleteEntities,
    handleEvent,
    exitSketch,
    backToSketch,
    patchFeature,
    patchBody,
    resetRegions,
    regions: currentRegions,
    commit,
    rebuildAll,
    rebuildDependents,
    startPlane,
    patchPlane,
    commitPlane,
    deletePlane,
    startMeasure,
    clearMeasure
  };
}

export type ModelingController = ReturnType<typeof useModeling>;
