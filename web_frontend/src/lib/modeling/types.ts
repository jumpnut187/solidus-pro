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
  seq: number;
  sketch: Sketch;
  endCondition: EndCondition;
  depth: number;
  reverse: boolean;
  targetPartId?: string | null;
  draftDeg: number;
  draftOutward: boolean;
  trimOverlap: boolean;
  regions?: Vec2[] | null;
};

export type CutEndCondition = 'blind' | 'midplane' | 'throughAll' | 'throughAllBoth';

export const CUT_END_CONDITIONS: { id: CutEndCondition; label: string; hint: string }[] = [
  { id: 'blind', label: 'Blind', hint: 'A fixed depth from the sketch plane.' },
  { id: 'throughAll', label: 'Through All', hint: 'Through every selected body in the cut direction.' },
  { id: 'throughAllBoth', label: 'Through All - Both', hint: 'Through every selected body on both sides of the sketch.' },
  { id: 'midplane', label: 'Mid Plane', hint: 'The depth split evenly either side of the sketch.' }
];

export type CutFeature = {
  version: 1;
  kind: 'cut';
  id: string;
  name: string;
  seq: number;
  sketch: Sketch;
  endCondition: CutEndCondition;
  depth: number;
  reverse: boolean;
  draftDeg: number;
  draftOutward: boolean;
  flipSide: boolean;
  partIds: string[];
  regions?: Vec2[] | null;
};

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

export function bodySeq(part: ProjectPart): number {
  return featureOf(part)?.seq ?? 0;
}

export function cutsOnPart(cuts: CutFeature[], part: ProjectPart): CutFeature[] {
  const seq = bodySeq(part);
  return cuts.filter((cut) => cut.seq > seq && cut.partIds.includes(part.id)).sort((a, b) => a.seq - b.seq);
}

export function cutCandidates(parts: ProjectPart[], cut: CutFeature): ProjectPart[] {
  return parts.filter((part) => part.meshData && bodySeq(part) < cut.seq);
}

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

export type ModelingStage = 'plane' | 'sketch' | 'feature' | 'measure';

export type PickPurpose = 'sketch' | 'faceExtrude' | 'referencePlane';

export type FaceOutline = {
  loops: Vec3[][];
  planar: boolean;
};

export type ModelingViewState = {
  stage: ModelingStage;
  purpose: PickPurpose;
  plane: SketchPlane | null;
  entities: SketchEntity[];
  tool: SketchTool;
  polygonSides: number;
  selectedEntityIds: string[];
  snap: boolean;
  preview: { positions: Float32Array; indices: Uint32Array; ok: boolean; cut?: boolean } | null;
  direction: Vec3 | null;
  extrudeDepth: { value: number; midplane: boolean } | null;
  regions: Array<SketchRegion & { selected: boolean }> | null;
  referencePlanes: { id: string; name: string; plane: SketchPlane }[];
  planePreview: SketchPlane[];
  chosenPlanes: SketchPlane[];
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
