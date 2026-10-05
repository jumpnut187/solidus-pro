/**
 * Parametric modeling: features, and the contract between the modeling
 * session and the viewport.
 *
 * A modeled body is one feature: a sketch of closed profiles on a plane, and
 * an extrude of that sketch with a SOLIDWORKS-style end condition. The
 * definition is stored on the part row (see `web/routes_modeling.py`) and the
 * triangles it produces are stored as the part's STL, which is all the
 * simulation engine ever reads. Editing a dimension rebuilds the body from
 * this definition, which is what makes it parametric.
 *
 * Everything here is in inches and in the engine's Z-up world frame, and it
 * is pure: no Babylon, no Manifold, so the kernel, the worker, the viewport
 * overlay and the panels can all import it. Planes live in `planes.ts`,
 * sketch geometry in `sketch.ts`, the Measure tool in `measure.ts`; this
 * module re-exports them so callers have one import.
 */
import type { PartType, ProjectPart } from '../api';
import type { MeasurePick, MeasureResult } from './measure';
import type { SketchPlane } from './planes';
import type { SketchRegion } from './regions';
import { sketchProblem, type SketchEntity, type SketchTool } from './sketch';
import type { Vec2, Vec3 } from './vec';

export * from './planes';
export * from './regions';
export * from './sketch';
export * from './vec';

export type Sketch = { plane: SketchPlane; entities: SketchEntity[] };

// ---------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------

/**
 * SOLIDWORKS extrude end conditions.
 *
 * - blind:      a fixed depth from the sketch plane.
 * - midplane:   a fixed depth split evenly either side of the sketch plane.
 * - upToNext:   up to the first surface that meets the profile; the end face
 *               conforms to that surface, which is how a riser or a chill is
 *               made to sit exactly on a curved casting.
 * - upToBody:   the same, but only the chosen body stops the extrude.
 * - throughAll: through everything in the extrude direction.
 */
export type EndCondition = 'blind' | 'midplane' | 'upToNext' | 'upToBody' | 'throughAll';

export const END_CONDITIONS: { id: EndCondition; label: string; hint: string }[] = [
  { id: 'blind', label: 'Blind', hint: 'A fixed depth from the sketch plane.' },
  { id: 'upToNext', label: 'Up To Next', hint: 'Stops at the next surface and takes its shape.' },
  { id: 'upToBody', label: 'Up To Body', hint: 'Stops at the chosen body and takes its shape.' },
  { id: 'midplane', label: 'Mid Plane', hint: 'The depth split evenly either side of the sketch.' },
  { id: 'throughAll', label: 'Through All', hint: 'Through every body in the extrude direction.' }
];

export type ExtrudeFeature = {
  version: 1;
  kind: 'extrude';
  /** History order. A feature only sees bodies made before it (and every uploaded body). */
  seq: number;
  sketch: Sketch;
  endCondition: EndCondition;
  /** Inches; used by blind and mid plane. */
  depth: number;
  /** Extrude against the sketch normal instead of along it. */
  reverse: boolean;
  /** The body an `upToBody` extrude stops at. */
  targetPartId?: string | null;
  /** Taper angle in degrees, 0 for straight walls. */
  draftDeg: number;
  /** Taper outward (the body widens away from the sketch) instead of inward. */
  draftOutward: boolean;
  /** Cut the body back wherever it overlaps an earlier body. */
  trimOverlap: boolean;
  /**
   * The sketch regions to extrude, named by a point inside each (see
   * `regions.ts`); null or absent for every region the even-odd rule fills.
   */
  regions?: Vec2[] | null;
};

/**
 * SOLIDWORKS Cut-Extrude end conditions. A cut removes material, so the
 * conditions that stop on a surface have nothing to offer it; Through All in
 * both directions cuts a sketch on a mid plane clean through a casting.
 */
export type CutEndCondition = 'blind' | 'midplane' | 'throughAll' | 'throughAllBoth';

export const CUT_END_CONDITIONS: { id: CutEndCondition; label: string; hint: string }[] = [
  { id: 'blind', label: 'Blind', hint: 'A fixed depth from the sketch plane.' },
  { id: 'throughAll', label: 'Through All', hint: 'Through every selected body in the cut direction.' },
  { id: 'throughAllBoth', label: 'Through All - Both', hint: 'Through every selected body on both sides of the sketch.' },
  { id: 'midplane', label: 'Mid Plane', hint: 'The depth split evenly either side of the sketch.' }
];

/**
 * An extrude cut: a sketch swept into a tool solid that is subtracted from
 * the bodies in its feature scope.
 *
 * A cut is not a body. Its definition lives in the project's `ui_state`
 * (`cut_features`), and what it does lives in the STL of every body it cut:
 * the server keeps each cut body's pre-cut STL as its base, so editing or
 * deleting the cut rebuilds the body from that base and restores material.
 * A body is cut only by cuts that come after it in the history.
 */
export type CutFeature = {
  version: 1;
  kind: 'cut';
  id: string;
  name: string;
  seq: number;
  sketch: Sketch;
  endCondition: CutEndCondition;
  /** Inches; used by blind and mid plane. */
  depth: number;
  /** Cut against the sketch normal instead of along it. */
  reverse: boolean;
  draftDeg: number;
  /** Taper outward (the cut widens away from the sketch) instead of inward. */
  draftOutward: boolean;
  /** Remove the material outside the profile instead of inside it. */
  flipSide: boolean;
  /** The feature scope: the bodies this cut removes material from. */
  partIds: string[];
  /** The sketch regions to cut, as for an extrude; null for the even-odd default. */
  regions?: Vec2[] | null;
};

/** Anything in the history: a body-making extrude or a cut. */
export type ModelFeature = ExtrudeFeature | CutFeature;

export type FeatureBody = {
  name: string;
  partType: PartType;
  isOpenTop: boolean;
};

export function defaultExtrude(seq: number, plane: SketchPlane): ExtrudeFeature {
  return {
    version: 1,
    kind: 'extrude',
    seq,
    sketch: { plane, entities: [] },
    endCondition: 'blind',
    depth: 1,
    reverse: false,
    targetPartId: null,
    draftDeg: 0,
    draftOutward: false,
    trimOverlap: true
  };
}

export function defaultCut(seq: number, plane: SketchPlane, name: string): CutFeature {
  return {
    version: 1,
    kind: 'cut',
    id: newCutId(),
    name,
    seq,
    sketch: { plane, entities: [] },
    endCondition: 'throughAll',
    depth: 1,
    reverse: true,
    draftDeg: 0,
    draftOutward: false,
    flipSide: false,
    partIds: []
  };
}

let cutCounter = 0;
export function newCutId(): string {
  cutCounter += 1;
  return `cut${Date.now().toString(36)}${cutCounter.toString(36)}`;
}

export function nextCutName(cuts: CutFeature[]): string {
  const used = new Set(cuts.map((cut) => cut.name));
  for (let n = 1; ; n += 1) {
    const name = `Cut-Extrude${n}`;
    if (!used.has(name)) return name;
  }
}

/** The cut features stored in a project's `ui_state`, dropping anything malformed. */
export function parseCutFeatures(value: unknown): CutFeature[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is CutFeature => {
    if (!item || typeof item !== 'object') return false;
    const candidate = item as Partial<CutFeature>;
    return (
      candidate.kind === 'cut' &&
      typeof candidate.id === 'string' &&
      typeof candidate.seq === 'number' &&
      Array.isArray(candidate.partIds) &&
      Boolean(candidate.sketch && Array.isArray(candidate.sketch.entities))
    );
  });
}

/** Where a body sits in the history: its extrude's seq, or 0 for an uploaded body. */
export function bodySeq(part: ProjectPart): number {
  return featureOf(part)?.seq ?? 0;
}

/** The cuts that remove material from `part`, in history order. */
export function cutsOnPart(cuts: CutFeature[], part: ProjectPart): CutFeature[] {
  const seq = bodySeq(part);
  return cuts.filter((cut) => cut.seq > seq && cut.partIds.includes(part.id)).sort((a, b) => a.seq - b.seq);
}

/** The bodies a cut may have in its scope: every body that comes before it in the history. */
export function cutCandidates(parts: ProjectPart[], cut: CutFeature): ProjectPart[] {
  return parts.filter((part) => part.meshData && bodySeq(part) < cut.seq);
}

/** Why a feature cannot be built as it stands, or null. */
export function featureProblem(feature: ModelFeature): string | null {
  const sketch = sketchProblem(feature.sketch.entities);
  if (sketch) return sketch;
  if (feature.regions && !feature.regions.length) return 'Select at least one region of the sketch.';
  if ((feature.endCondition === 'blind' || feature.endCondition === 'midplane') && !(feature.depth > 1e-4)) {
    return 'Depth must be above zero.';
  }
  if (feature.kind === 'cut') {
    if (!feature.partIds.length) return 'Select the bodies to cut.';
  } else if (feature.endCondition === 'upToBody' && !feature.targetPartId) {
    return 'Choose the body to extrude up to.';
  }
  if (!(feature.draftDeg >= 0 && feature.draftDeg < 60)) return 'Draft must be between 0 and 60 degrees.';
  return null;
}

/** The feature definition stored on a part, if the part was modeled here. */
export function featureOf(part: ProjectPart): ExtrudeFeature | null {
  const feature = part.feature;
  if (!feature || typeof feature !== 'object') return null;
  return feature.kind === 'extrude' && feature.sketch ? feature : null;
}

export function nextFeatureSeq(parts: ProjectPart[], cuts: CutFeature[] = []): number {
  let seq = 0;
  for (const part of parts) seq = Math.max(seq, featureOf(part)?.seq ?? 0);
  for (const cut of cuts) seq = Math.max(seq, cut.seq);
  return seq + 1;
}

/**
 * The bodies a feature may stop at or be trimmed by: every uploaded body, and
 * every modeled body that comes earlier in the history. Hidden bodies are not
 * simulated, so they do not stop an extrude either.
 */
export function featureTargets(parts: ProjectPart[], feature: ExtrudeFeature, ownPartId: string | null): ProjectPart[] {
  if (feature.endCondition === 'upToBody') {
    return parts.filter((part) => part.id === feature.targetPartId && part.id !== ownPartId && part.meshData);
  }
  return parts.filter((part) => {
    if (part.id === ownPartId || !part.meshData || part.visible === false) return false;
    const other = featureOf(part);
    return !other || other.seq < feature.seq;
  });
}

// ---------------------------------------------------------------------------
// The viewport contract for modeling
// ---------------------------------------------------------------------------

/**
 * Which step the session is on. `plane` waits for a plane or a face to be
 * clicked (what for is the `purpose`); `sketch` and `feature` are the two
 * halves of an extrude; `measure` is the Measure tool.
 */
export type ModelingStage = 'plane' | 'sketch' | 'feature' | 'measure';

/** What a click in the `plane` stage is choosing. */
export type PickPurpose = 'sketch' | 'faceExtrude' | 'referencePlane';

/** A face's outline in world space, as clicked: the loops bounding it. */
export type FaceOutline = {
  loops: Vec3[][];
  /** False when the clicked face is a facet of a curved surface. */
  planar: boolean;
};

export type ModelingViewState = {
  stage: ModelingStage;
  purpose: PickPurpose;
  plane: SketchPlane | null;
  entities: SketchEntity[];
  tool: SketchTool;
  polygonSides: number;
  /** The selected sketch entities; the last one is the one its dimensions show for. */
  selectedEntityIds: string[];
  snap: boolean;
  /**
   * The extrude being previewed, in world space; `ok` false draws it as an
   * error. For a cut it is the material the cut removes, drawn as `cut`.
   */
  preview: { positions: Float32Array; indices: Uint32Array; ok: boolean; cut?: boolean } | null;
  /** The direction the extrude grows in, drawn as an arrow in the feature stage. */
  direction: Vec3 | null;
  /**
   * The depth the feature stage's arrow sets by dragging, for end conditions
   * that have one (blind and mid plane); null when the depth is not a number
   * the user chooses.
   */
  extrudeDepth: { value: number; midplane: boolean } | null;
  /** The sketch's regions in the feature stage, and which the feature uses. */
  regions: Array<SketchRegion & { selected: boolean }> | null;
  /** The project's reference planes, drawn and pickable in the plane stage. */
  referencePlanes: { id: string; name: string; plane: SketchPlane }[];
  /** Planes the Plane property manager would create, drawn as a preview. */
  planePreview: SketchPlane[];
  /** Planes already chosen as references, highlighted. */
  chosenPlanes: SketchPlane[];
  /** The Measure tool's selections and what they measure. */
  measure: { picks: MeasurePick[]; result: MeasureResult | null } | null;
};

export type ModelingEvent =
  | { type: 'planePicked'; plane: SketchPlane; face: FaceOutline | null }
  | { type: 'entityCreated'; entity: SketchEntity }
  | { type: 'entitySelected'; ids: string[] }
  | { type: 'entityChanged'; entity: SketchEntity }
  | { type: 'entitiesChanged'; entities: SketchEntity[] }
  | { type: 'entityDeleted'; id: string }
  | { type: 'entitiesDeleted'; ids: string[] }
  | { type: 'regionToggled'; point: Vec2 }
  | { type: 'toolChanged'; tool: SketchTool }
  | { type: 'convertFace'; face: FaceOutline }
  | { type: 'measurePicked'; pick: MeasurePick | null }
  | { type: 'depthChanged'; depth: number };

