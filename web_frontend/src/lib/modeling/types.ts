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
import { sketchProblem, type SketchEntity, type SketchTool } from './sketch';
import type { Vec3 } from './vec';

export * from './planes';
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
};

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

/** Why a feature cannot be built as it stands, or null. */
export function featureProblem(feature: ExtrudeFeature): string | null {
  const sketch = sketchProblem(feature.sketch.entities);
  if (sketch) return sketch;
  if ((feature.endCondition === 'blind' || feature.endCondition === 'midplane') && !(feature.depth > 1e-4)) {
    return 'Depth must be above zero.';
  }
  if (feature.endCondition === 'upToBody' && !feature.targetPartId) return 'Choose the body to extrude up to.';
  if (!(feature.draftDeg >= 0 && feature.draftDeg < 60)) return 'Draft must be between 0 and 60 degrees.';
  return null;
}

/** The feature definition stored on a part, if the part was modeled here. */
export function featureOf(part: ProjectPart): ExtrudeFeature | null {
  const feature = part.feature;
  if (!feature || typeof feature !== 'object') return null;
  return feature.kind === 'extrude' && feature.sketch ? feature : null;
}

export function nextFeatureSeq(parts: ProjectPart[]): number {
  let seq = 0;
  for (const part of parts) seq = Math.max(seq, featureOf(part)?.seq ?? 0);
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
  selectedEntityId: string | null;
  snap: boolean;
  /** The extrude being previewed, in world space; `ok` false draws it as an error. */
  preview: { positions: Float32Array; indices: Uint32Array; ok: boolean } | null;
  /** The direction the extrude grows in, drawn as an arrow in the feature stage. */
  direction: Vec3 | null;
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
  | { type: 'entitySelected'; id: string | null }
  | { type: 'entityChanged'; entity: SketchEntity }
  | { type: 'entitiesChanged'; entities: SketchEntity[] }
  | { type: 'entityDeleted'; id: string }
  | { type: 'toolChanged'; tool: SketchTool }
  | { type: 'convertFace'; face: FaceOutline }
  | { type: 'measurePicked'; pick: MeasurePick | null };

