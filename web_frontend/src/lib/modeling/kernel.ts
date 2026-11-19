/**
 * The geometry kernel behind modeled bodies: sketch profiles, extrudes and
 * their end conditions, on Manifold.
 *
 * Manifold is a mesh-boolean kernel, and that is the right kind for this
 * application: every body it has to work against is an uploaded STL, a
 * triangle mesh with no B-rep behind it, and what it has to hand the
 * simulation is a watertight triangle mesh. A B-rep kernel would have to turn
 * a 200,000-triangle casting into 200,000 planar faces before it could cut
 * anything. Manifold's booleans are exact and guarantee a manifold result.
 *
 * "Up To Next" is the operation the foundry needs most -- a riser or a chill
 * that sits exactly on a curved casting -- and on a mesh kernel it is three
 * steps. Extrude the profile far past everything in the extrude direction,
 * subtract the bodies, and keep only the pieces that start at the sketch
 * plane: each of those ran from the plane until it met the first surface in
 * its way, and its end face IS that surface. As in SOLIDWORKS, the surface
 * has to cover the whole profile; a piece that runs all the way to the far
 * end found nothing to stop it, and the feature fails rather than build a
 * body that goes nowhere.
 *
 * Pure apart from the Manifold module it is given, so it runs in the worker
 * and in tests alike. Every WASM object it makes is freed before it returns.
 */
import type { CrossSection, Manifold, ManifoldToplevel, Mat4, Vec2 as MVec2 } from 'manifold-3d';

import {
  featureProblem,
  interiorPoint,
  planeFrame,
  pointInContours,
  sketchProblem,
  sketchProfiles,
  type CutFeature,
  type ExtrudeFeature,
  type Sketch,
  type SketchRegion,
  type Vec2,
  type Vec3
} from './types';

export type Box3 = { min: Vec3; max: Vec3 };

/** A body the kernel can stop at or trim against, already in world space. */
export type KernelBody = {
  id: string;
  name: string;
  /** Null when the mesh is not a closed solid; `error` says why. */
  manifold: Manifold | null;
  error: string | null;
  bbox: Box3 | null;
};

export type BuildOutput = {
  positions: Float32Array;
  indices: Uint32Array;
  volume: number;
  triangles: number;
  bbox: Box3;
  warnings: string[];
};

/** A feature that cannot be built as specified. The message is for the user. */
export class FeatureBuildError extends Error {}

// ---------------------------------------------------------------------------
// Matrices (column-major, as Manifold takes them)
// ---------------------------------------------------------------------------

function frameMatrix(u: Vec3, v: Vec3, w: Vec3, origin: Vec3): Mat4 {
  return [u[0], u[1], u[2], 0, v[0], v[1], v[2], 0, w[0], w[1], w[2], 0, origin[0], origin[1], origin[2], 1];
}

/** Inverse of an orthonormal frame matrix (mirrored or not). */
function inverseFrame(u: Vec3, v: Vec3, w: Vec3, origin: Vec3): Mat4 {
  const tx = -(u[0] * origin[0] + u[1] * origin[1] + u[2] * origin[2]);
  const ty = -(v[0] * origin[0] + v[1] * origin[1] + v[2] * origin[2]);
  const tz = -(w[0] * origin[0] + w[1] * origin[1] + w[2] * origin[2]);
  return [u[0], v[0], w[0], 0, u[1], v[1], w[1], 0, u[2], v[2], w[2], 0, tx, ty, tz, 1];
}

/**
 * The engine's part transform as a matrix: scale, then rotate about X, Y and
 * Z in that order (`trimesh.transformations.euler_matrix(..., 'sxyz')`), then
 * translate -- `core/part_manager.Part._apply_transform`. The modeled body
 * has to meet the casting where the simulation will put the casting, not
 * where some other convention would.
 */
export function partWorldMatrix(
  position: Vec3,
  rotationDeg: Vec3,
  scale: number
): Mat4 {
  const [ax, ay, az] = rotationDeg.map((value) => (value * Math.PI) / 180);
  const cx = Math.cos(ax);
  const sx = Math.sin(ax);
  const cy = Math.cos(ay);
  const sy = Math.sin(ay);
  const cz = Math.cos(az);
  const sz = Math.sin(az);
  // R = Rz * Ry * Rx, row-major entries rij.
  const r00 = cz * cy;
  const r01 = cz * sy * sx - sz * cx;
  const r02 = cz * sy * cx + sz * sx;
  const r10 = sz * cy;
  const r11 = sz * sy * sx + cz * cx;
  const r12 = sz * sy * cx - cz * sx;
  const r20 = -sy;
  const r21 = cy * sx;
  const r22 = cy * cx;
  const s = scale || 1;
  return [
    r00 * s, r10 * s, r20 * s, 0,
    r01 * s, r11 * s, r21 * s, 0,
    r02 * s, r12 * s, r22 * s, 0,
    position[0], position[1], position[2], 1
  ];
}

// ---------------------------------------------------------------------------
// Bodies
// ---------------------------------------------------------------------------

/**
 * A Manifold from a part's triangles, in world space.
 *
 * STL stores every triangle's corners separately; `merge` welds them back
 * into a closed surface. A mesh that still has open edges after that is not
 * a solid, and nothing can be "up to" its surface, so it comes back with an
 * error instead of a Manifold.
 */
export function bodyFromMesh(
  wasm: ManifoldToplevel,
  id: string,
  name: string,
  positions: Float32Array,
  indices: Uint32Array,
  matrix: Mat4
): KernelBody {
  try {
    const mesh = new wasm.Mesh({ numProp: 3, vertProperties: positions, triVerts: indices });
    mesh.merge();
    const local = new wasm.Manifold(mesh);
    const world = local.transform(matrix);
    local.delete();
    if (world.isEmpty()) {
      world.delete();
      return { id, name, manifold: null, error: `"${name}" has no volume.`, bbox: null };
    }
    const box = world.boundingBox();
    return {
      id,
      name,
      manifold: world,
      error: null,
      bbox: { min: [...box.min] as Vec3, max: [...box.max] as Vec3 }
    };
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    return {
      id,
      name,
      manifold: null,
      error: `"${name}" is not a closed solid (${reason}), so it cannot bound an extrude.`,
      bbox: meshBounds(positions, matrix)
    };
  }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/** Tracks WASM objects so a build frees everything it made, however it exits. */
class Scope {
  private readonly _items: Array<{ delete(): void }> = [];
  keep<T extends { delete(): void }>(item: T): T {
    this._items.push(item);
    return item;
  }
  release(): void {
    for (const item of this._items.splice(0)) {
      try {
        item.delete();
      } catch {
        /* already freed */
      }
    }
  }
}

function boxesOverlap(a: Box3, b: Box3, pad: number): boolean {
  for (let axis = 0; axis < 3; axis += 1) {
    if (a.max[axis] + pad < b.min[axis] || b.max[axis] + pad < a.min[axis]) return false;
  }
  return true;
}

function boxCorners(box: Box3): Vec3[] {
  const corners: Vec3[] = [];
  for (const x of [box.min[0], box.max[0]]) {
    for (const y of [box.min[1], box.max[1]]) {
      for (const z of [box.min[2], box.max[2]]) corners.push([x, y, z]);
    }
  }
  return corners;
}

function toBox3(box: { min: ArrayLike<number>; max: ArrayLike<number> }): Box3 {
  return { min: [box.min[0], box.min[1], box.min[2]], max: [box.max[0], box.max[1], box.max[2]] };
}

/** Regions smaller than this (in²) are boolean noise, not something drawn. */
const REGION_AREA_EPS = 1e-6;

type RegionAtom = { section: CrossSection; contours: Vec2[][]; count: number };

/**
 * The sketch's regions: every piece its closed contours divide the plane
 * into. Two overlapping circles are three regions -- two crescents and the
 * lens between them -- each of which can be extruded on its own. `count` is
 * how many contours a region lies inside; the even-odd rule fills the ones
 * where it is odd.
 *
 * Built by splitting: start from the union of all contours, and split every
 * piece so far by each contour in turn into the part inside it and the part
 * outside.
 */
function regionAtoms(wasm: ManifoldToplevel, scope: Scope, sketch: Sketch): RegionAtom[] {
  const { CrossSection } = wasm;
  const loops = sketchProfiles(sketch.entities).loops;
  if (!loops.length) return [];
  const sections = loops.map((loop) =>
    scope.keep(new CrossSection([loop.map((point) => [point[0], point[1]] as MVec2)], 'EvenOdd'))
  );
  const pieces = (section: CrossSection) =>
    section
      .decompose()
      .map((piece) => scope.keep(piece))
      .filter((piece) => piece.area() > REGION_AREA_EPS);
  let atoms: Array<{ section: CrossSection; count: number }> = pieces(scope.keep(CrossSection.union(sections))).map(
    (section) => ({ section, count: 0 })
  );
  for (const contour of sections) {
    const next: typeof atoms = [];
    for (const atom of atoms) {
      for (const section of pieces(scope.keep(atom.section.intersect(contour)))) next.push({ section, count: atom.count + 1 });
      for (const section of pieces(scope.keep(atom.section.subtract(contour)))) next.push({ section, count: atom.count });
    }
    atoms = next;
  }
  return atoms.map((atom) => ({
    ...atom,
    contours: atom.section.toPolygons().map((polygon) => polygon.map((point) => [point[0], point[1]] as Vec2))
  }));
}

/** The sketch's regions, for drawing and picking them (see `regions.ts`). */
export function sketchRegions(wasm: ManifoldToplevel, sketch: Sketch): SketchRegion[] {
  const scope = new Scope();
  try {
    return regionAtoms(wasm, scope, sketch).map((atom) => {
      const flat = atom.contours.flat();
      const triangles = wasm.triangulate(atom.contours.map((contour) => contour.map((p) => [p[0], p[1]] as MVec2)), 1e-9);
      const xy = new Float32Array(triangles.length * 6);
      triangles.forEach((triangle, t) => {
        for (let k = 0; k < 3; k += 1) {
          const point = flat[triangle[k]];
          xy[t * 6 + k * 2] = point[0];
          xy[t * 6 + k * 2 + 1] = point[1];
        }
      });
      return {
        contours: atom.contours,
        seed: interiorPoint(atom.contours),
        inDefault: atom.count % 2 === 1,
        area: atom.section.area(),
        triangles: xy
      };
    });
  } finally {
    scope.release();
  }
}

/**
 * The profile a feature sweeps. With no regions picked, every closed contour
 * goes in together under the even-odd rule, the way a sketch's contours
 * become regions: a circle inside a rectangle is a hole in it, which is how a
 * ring chill or a sleeve is drawn. With regions picked, it is the union of
 * the regions the feature's seed points fall in.
 */
function sketchSection(
  wasm: ManifoldToplevel,
  scope: Scope,
  feature: { sketch: Sketch; regions?: Vec2[] | null }
): CrossSection {
  let profile: CrossSection;
  if (feature.regions) {
    const seeds = feature.regions;
    const chosen = regionAtoms(wasm, scope, feature.sketch).filter((atom) =>
      seeds.some((seed) => pointInContours(atom.contours, seed))
    );
    if (!chosen.length) {
      throw new FeatureBuildError('None of the picked regions is in the sketch any more. Click regions in the view to pick them again.');
    }
    profile = scope.keep(wasm.CrossSection.union(chosen.map((atom) => atom.section)));
  } else {
    const contours: MVec2[][] = sketchProfiles(feature.sketch.entities).loops.map((loop) =>
      loop.map((point) => [point[0], point[1]] as MVec2)
    );
    profile = scope.keep(new wasm.CrossSection(contours, 'EvenOdd'));
  }
  if (profile.isEmpty() || profile.area() < 1e-6) {
    throw new FeatureBuildError('The sketch encloses no area.');
  }
  return profile;
}

/**
 * Build an extrude feature.
 *
 * `targets` are the bodies this feature may stop at or trim against (see
 * `featureTargets`); they are not modified and stay owned by the caller.
 * Throws `FeatureBuildError` with a user-facing message when the feature
 * cannot be built.
 */
export function buildExtrude(wasm: ManifoldToplevel, feature: ExtrudeFeature, targets: KernelBody[]): BuildOutput {
  const problem = featureProblem(feature);
  if (problem) throw new FeatureBuildError(problem);

  const { Manifold } = wasm;
  const scope = new Scope();
  const warnings: string[] = [];
  try {
    const frame = planeFrame(feature.sketch.plane);
    const w: Vec3 = feature.reverse
      ? [-frame.normal[0], -frame.normal[1], -frame.normal[2]]
      : frame.normal;
    const toWorld = frameMatrix(frame.u, frame.v, w, frame.origin);
    const toLocal = inverseFrame(frame.u, frame.v, w, frame.origin);

    const profile = sketchSection(wasm, scope, feature);

    const usable = targets.filter((body) => body.manifold || body.error);
    const extent = extentAlong(usable, frame.origin, w);
    const size = sceneSize(usable, profile);

    let start = 0;
    let length: number;
    switch (feature.endCondition) {
      case 'blind':
        length = feature.depth;
        break;
      case 'midplane':
        length = feature.depth;
        start = -feature.depth / 2;
        break;
      case 'throughAll':
      case 'upToNext':
      case 'upToBody': {
        if (extent === null || extent <= 1e-6) {
          throw new FeatureBuildError(
            feature.endCondition === 'upToBody'
              ? 'The chosen body is not in the extrude direction. Reverse the direction.'
              : 'There is no body in the extrude direction. Reverse the direction.'
          );
        }
        // Past the far side of everything, so a piece that reaches the end
        // is one that met nothing on the way.
        length = extent + Math.max(size * 0.05, 0.1);
        break;
      }
    }

    let solid = extrudeProfile(wasm, scope, profile, length, feature);
    if (start !== 0) solid = scope.keep(solid.translate([0, 0, start]));
    const prismBox = toBox3(solid.boundingBox());
    const tolerance = Math.max(size * 1e-5, 1e-5);

    if (feature.endCondition === 'upToNext' || feature.endCondition === 'upToBody') {
      const blockers = localBodies(scope, usable, toLocal, toWorld, prismBox, tolerance);
      if (!blockers.length) {
        throw new FeatureBuildError(
          feature.endCondition === 'upToBody'
            ? 'The profile does not meet the chosen body in this direction.'
            : 'The profile meets no surface in this direction. Reverse the direction or move the sketch.'
        );
      }
      const cut = scope.keep(solid.subtract(scope.keep(Manifold.union(blockers))));
      const pieces = cut.decompose().map((piece) => scope.keep(piece));
      const fromPlane = pieces.filter((piece) => piece.boundingBox().min[2] <= tolerance * 10);
      if (!fromPlane.length) {
        throw new FeatureBuildError(
          'The sketch plane starts inside a body. Offset the plane or reverse the direction.'
        );
      }
      const farEnd = solid.boundingBox().max[2];
      const runaway = fromPlane.some((piece) => piece.boundingBox().max[2] >= farEnd - tolerance * 10);
      if (runaway) {
        throw new FeatureBuildError(
          'Part of the profile never meets a surface in this direction. Up To Next needs a surface ' +
            'that covers the whole profile: move the sketch over the body or reverse the direction.'
        );
      }
      solid = scope.keep(Manifold.compose(fromPlane));
    }

    if (feature.trimOverlap) {
      const box = toBox3(solid.boundingBox());
      const overlapping = localBodies(scope, usable, toLocal, toWorld, box, tolerance);
      if (overlapping.length) {
        const before = solid.volume();
        const trimmed = scope.keep(solid.subtract(scope.keep(Manifold.union(overlapping))));
        if (trimmed.isEmpty()) {
          throw new FeatureBuildError('The body lies entirely inside other bodies.');
        }
        if (trimmed.volume() < before - Math.max(before * 1e-6, 1e-9)) {
          warnings.push('Trimmed where it overlapped other bodies.');
        }
        solid = trimmed;
      }
    }

    const raw = scope.keep(solid.transform(toWorld));
    if (raw.isEmpty()) throw new FeatureBuildError('The extrude produced no solid.');
    if (raw.status() !== 'NoError') {
      throw new FeatureBuildError(`The geometry kernel rejected the result (${raw.status()}).`);
    }
    const { solid: world, pieces } = withoutSlivers(wasm, scope, raw, Math.max(size * 1e-5, 1e-5));
    if (pieces > 1) warnings.push(`The body is in ${pieces} separate pieces.`);

    const { positions, indices } = watertightMesh(wasm, world, size);
    return {
      positions,
      indices,
      volume: world.volume(),
      triangles: indices.length / 3,
      bbox: toBox3(world.boundingBox()),
      warnings
    };
  } catch (error) {
    if (error instanceof FeatureBuildError) throw error;
    const reason = error instanceof Error ? error.message : String(error);
    throw new FeatureBuildError(`The geometry kernel failed: ${reason}`);
  } finally {
    scope.release();
  }
}

// ---------------------------------------------------------------------------
// Extrude cut
// ---------------------------------------------------------------------------
//
// A cut is the extrude's tool solid subtracted from the bodies in its scope.
// It is applied to each body in that body's own frame -- the frame its STL is
// stored in -- with the tool carried there, so every face the cut does not
// touch comes back exactly as it was uploaded.

/** A body to cut, in its own (stored) frame, with the matrix that places it in the world. */
export type CutTarget = {
  name: string;
  positions: Float32Array;
  indices: Uint32Array;
  matrix: Mat4;
};

/** The cut body in its own frame, or `changed: false` when no cut reached it. */
export type CutOutput = ({ changed: true } & BuildOutput) | { changed: false; warnings: string[] };

export type CutPreview = {
  /** The material the cut removes, in world space. */
  positions: Float32Array;
  indices: Uint32Array;
  /** Volume removed from each body, by body id; bodies it misses are absent. */
  removed: Record<string, number>;
  volume: number;
  warnings: string[];
};

/** Column-major `a * b`. */
function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c += 1) {
    for (let r = 0; r < 4; r += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) sum += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = sum;
    }
  }
  return out as unknown as Mat4;
}

/** Inverse of an affine matrix (the part transform: rotation, uniform scale, translation). */
function inverseAffine(m: Mat4): Mat4 {
  const [a, b, c] = [m[0], m[4], m[8]];
  const [d, e, f] = [m[1], m[5], m[9]];
  const [g, h, i] = [m[2], m[6], m[10]];
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!(Math.abs(det) > 1e-12)) throw new FeatureBuildError('The body has a zero scale and cannot be cut.');
  const s = 1 / det;
  // Rows of the inverse 3x3.
  const r0 = [A * s, -(b * i - c * h) * s, (b * f - c * e) * s];
  const r1 = [B * s, (a * i - c * g) * s, -(a * f - c * d) * s];
  const r2 = [C * s, -(a * h - b * g) * s, (a * e - b * d) * s];
  const [tx, ty, tz] = [m[12], m[13], m[14]];
  return [
    r0[0], r1[0], r2[0], 0,
    r0[1], r1[1], r2[1], 0,
    r0[2], r1[2], r2[2], 0,
    -(r0[0] * tx + r0[1] * ty + r0[2] * tz),
    -(r1[0] * tx + r1[1] * ty + r1[2] * tz),
    -(r2[0] * tx + r2[1] * ty + r2[2] * tz),
    1
  ] as unknown as Mat4;
}

const IDENTITY: Mat4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

type CutFrame = { toWorld: Mat4; toLocal: Mat4; profile: CrossSection };

function cutFrame(wasm: ManifoldToplevel, scope: Scope, cut: CutFeature): CutFrame {
  const problem = sketchProblem(cut.sketch.entities);
  if (problem) throw new FeatureBuildError(`${cut.name}: ${problem}`);
  if ((cut.endCondition === 'blind' || cut.endCondition === 'midplane') && !(cut.depth > 1e-4)) {
    throw new FeatureBuildError(`${cut.name}: depth must be above zero.`);
  }
  if (!(cut.draftDeg >= 0 && cut.draftDeg < 60)) {
    throw new FeatureBuildError(`${cut.name}: draft must be between 0 and 60 degrees.`);
  }
  const frame = planeFrame(cut.sketch.plane);
  const w: Vec3 = cut.reverse ? [-frame.normal[0], -frame.normal[1], -frame.normal[2]] : frame.normal;
  return {
    toWorld: frameMatrix(frame.u, frame.v, w, frame.origin),
    toLocal: inverseFrame(frame.u, frame.v, w, frame.origin),
    profile: sketchSection(wasm, scope, cut)
  };
}

/**
 * The solid a cut removes, in the cut's frame (sketch plane at z = 0, cutting
 * toward +z), long enough to get through `reach` -- a box in that frame around
 * the body being cut. Null when the cut cannot reach the body: Through All
 * from a sketch that lies past it.
 *
 * Both-sided conditions are built as two halves that each start at the
 * sketch plane, so a draft tapers away from the sketch in both directions.
 * Flip Side to Cut is a block around the body less the profile's tool.
 */
function cutTool(
  wasm: ManifoldToplevel,
  scope: Scope,
  cut: CutFeature,
  profile: CrossSection,
  reach: Box3,
  size: number
): Manifold | null {
  const { Manifold } = wasm;
  const margin = Math.max(size * 0.05, 0.1);
  let back = 0;
  let ahead: number;
  switch (cut.endCondition) {
    case 'blind':
      ahead = cut.depth;
      break;
    case 'midplane':
      ahead = cut.depth / 2;
      back = cut.depth / 2;
      break;
    case 'throughAll':
      ahead = reach.max[2] + margin;
      break;
    case 'throughAllBoth':
      ahead = Math.max(reach.max[2], 0) + margin;
      back = Math.max(-reach.min[2], 0) + margin;
      break;
  }
  if (ahead <= 1e-9 && back <= 1e-9) return null;

  const halves: Manifold[] = [];
  if (ahead > 1e-9) halves.push(extrudeProfile(wasm, scope, profile, ahead, cut));
  if (back > 1e-9) {
    const half = extrudeProfile(wasm, scope, profile, back, cut);
    halves.push(scope.keep(half.scale([1, 1, -1])));
  }
  const prism = halves.length === 1 ? halves[0] : scope.keep(Manifold.union(halves));
  if (!cut.flipSide) return prism;

  const bounds = profile.bounds();
  const lo: [number, number] = [Math.min(reach.min[0], bounds.min[0]) - margin, Math.min(reach.min[1], bounds.min[1]) - margin];
  const hi: [number, number] = [Math.max(reach.max[0], bounds.max[0]) + margin, Math.max(reach.max[1], bounds.max[1]) + margin];
  // The prism's own ends, so the block is exactly as deep as the cut.
  const prismBox = prism.boundingBox();
  const block = scope.keep(
    Manifold.cube([hi[0] - lo[0], hi[1] - lo[1], prismBox.max[2] - prismBox.min[2]]).translate([lo[0], lo[1], prismBox.min[2]])
  );
  return scope.keep(block.subtract(prism));
}

function boxSize(box: Box3): number {
  return Math.max(box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2], 1);
}

/**
 * Cut one body by `cuts`, in history order.
 *
 * `target` is the body before any of these cuts (its base), in its own frame;
 * the result is in that frame too, ready to store as the body's STL. A cut
 * that leaves nothing of the body is refused rather than stored as an empty
 * part.
 */
export function applyCuts(wasm: ManifoldToplevel, target: CutTarget, cuts: CutFeature[]): CutOutput {
  const scope = new Scope();
  const warnings: string[] = [];
  try {
    const body = bodyFromMesh(wasm, 'target', target.name, target.positions, target.indices, IDENTITY);
    if (!body.manifold || !body.bbox) {
      throw new FeatureBuildError(`"${target.name}" is not a closed solid, so it cannot be cut.`);
    }
    let solid = scope.keep(body.manifold);
    const worldBox = transformBox(body.bbox, target.matrix);
    const toBody = inverseAffine(target.matrix);
    const size = boxSize(worldBox);
    let changed = false;

    for (const cut of [...cuts].sort((a, b) => a.seq - b.seq)) {
      const { toWorld, toLocal, profile } = cutFrame(wasm, scope, cut);
      const tool = cutTool(wasm, scope, cut, profile, transformBox(worldBox, toLocal), size);
      if (!tool) continue;
      const toolHere = scope.keep(tool.transform(multiply(toBody, toWorld)));
      if (!boxesOverlap(toBox3(toolHere.boundingBox()), toBox3(solid.boundingBox()), 0)) continue;
      const before = solid.volume();
      const next = scope.keep(solid.subtract(toolHere));
      if (next.isEmpty() || next.volume() <= Math.max(before * 1e-9, 1e-12)) {
        throw new FeatureBuildError(`${cut.name} removes all of "${target.name}".`);
      }
      if (before - next.volume() > Math.max(before * 1e-9, 1e-9)) {
        changed = true;
        solid = next;
      }
    }
    if (!changed) return { changed: false, warnings };

    if (solid.status() !== 'NoError') {
      throw new FeatureBuildError(`The geometry kernel rejected the cut (${solid.status()}).`);
    }
    const localSize = boxSize(toBox3(solid.boundingBox()));
    const { solid: kept, pieces, voids } = withoutSlivers(wasm, scope, solid, Math.max(localSize * 1e-5, 1e-5));
    if (pieces > 1) warnings.push(`"${target.name}" is in ${pieces} separate pieces after the cut.`);
    if (voids) {
      warnings.push(
        `The cut leaves ${voids === 1 ? 'an enclosed void' : `${voids} enclosed voids`} inside "${target.name}"; ` +
          'nothing can form it without a core.'
      );
    }
    const { positions, indices } = watertightMesh(wasm, kept, localSize);
    return {
      changed: true,
      positions,
      indices,
      volume: kept.volume(),
      triangles: indices.length / 3,
      bbox: toBox3(kept.boundingBox()),
      warnings
    };
  } catch (error) {
    if (error instanceof FeatureBuildError) throw error;
    const reason = error instanceof Error ? error.message : String(error);
    throw new FeatureBuildError(`The geometry kernel failed: ${reason}`);
  } finally {
    scope.release();
  }
}

/**
 * What `cut` removes from each of `bodies` (world space, the bodies in its
 * scope), for the preview: the removed material as one mesh and its volume
 * per body. Throws when the cut removes nothing or all of a body.
 */
export function previewCut(wasm: ManifoldToplevel, cut: CutFeature, bodies: KernelBody[]): CutPreview {
  const problem = featureProblem(cut);
  if (problem) throw new FeatureBuildError(problem);
  const scope = new Scope();
  const warnings: string[] = [];
  try {
    const { toWorld, toLocal, profile } = cutFrame(wasm, scope, cut);
    const pieces: Manifold[] = [];
    const removed: Record<string, number> = {};
    let total = 0;
    for (const body of bodies) {
      if (!body.bbox) continue;
      if (!body.manifold) throw new FeatureBuildError(body.error ?? `"${body.name}" cannot be cut.`);
      const tool = cutTool(wasm, scope, cut, profile, transformBox(body.bbox, toLocal), boxSize(body.bbox));
      if (!tool) continue;
      const toolWorld = scope.keep(tool.transform(toWorld));
      if (!boxesOverlap(toBox3(toolWorld.boundingBox()), body.bbox, 0)) continue;
      const piece = scope.keep(body.manifold.intersect(toolWorld));
      const volume = piece.volume();
      const bodyVolume = body.manifold.volume();
      if (volume <= Math.max(bodyVolume * 1e-9, 1e-9)) continue;
      if (volume >= bodyVolume * (1 - 1e-9)) {
        throw new FeatureBuildError(`The cut removes all of "${body.name}". Take it out of the feature scope.`);
      }
      removed[body.id] = volume;
      total += volume;
      pieces.push(piece);
    }
    if (!pieces.length) {
      throw new FeatureBuildError(
        'The cut removes no material from the selected bodies. Reverse the direction, move the sketch or select other bodies.'
      );
    }
    const shown = pieces.length === 1 ? pieces[0] : scope.keep(wasm.Manifold.compose(pieces));
    const mesh = shown.getMesh();
    const stride = mesh.numProp;
    const count = mesh.vertProperties.length / stride;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      for (let k = 0; k < 3; k += 1) positions[3 * i + k] = mesh.vertProperties[stride * i + k];
    }
    return { positions, indices: new Uint32Array(mesh.triVerts), removed, volume: total, warnings };
  } catch (error) {
    if (error instanceof FeatureBuildError) throw error;
    const reason = error instanceof Error ? error.message : String(error);
    throw new FeatureBuildError(`The geometry kernel failed: ${reason}`);
  } finally {
    scope.release();
  }
}

// ---------------------------------------------------------------------------
// Output that stays closed through STL storage
// ---------------------------------------------------------------------------
//
// Two bodies that touch -- a riser modeled onto the casting -- do not touch
// exactly once stored: the riser's STL holds float32 copies of points on the
// casting's surface, up to 2e-7 in off it, so the seam between them is a
// field of sub-micron gaps and overlaps. An extrude that stops on both bodies
// fills every space they leave open, the seam included, and comes out with
// sheets of no thickness and vertices that coincide. Manifold represents that
// as a valid solid. An STL cannot: it has no topology, every reader (trimesh,
// the voxelizer, this worker) rebuilds it by welding equal positions, and the
// weld turns those sheets into edges shared by four triangles and the body
// into a surface no voxelizer can call inside or outside.
//
// So the output is finished in three steps: pieces that are only such a sheet
// are dropped, every face the body took from another body is pushed a hair
// into that body, and what is left is welded the way storage will weld it and
// refused if it does not come out closed. The push opens every sheet and pinch
// into a solid of real (if tiny) thickness, and it leaves the stored body
// overlapping its neighbours instead of meeting them at a seam, so the next
// feature drawn across them finds no seam to fill.

/**
 * `solid` without the pieces thinner than `minThickness`, the number of
 * solid pieces left, and the number of enclosed voids. Thickness is 2|V|/A: a
 * plate's thickness, a rod's radius. A sheet that leaked into a seam is about
 * 1e-7 in thick; nothing modeled for a casting is under a thousandth of an
 * inch. A void (a cut that stops inside a body) is a separate shell of
 * negative volume; it is kept, unless it is itself a sliver.
 */
function withoutSlivers(
  wasm: ManifoldToplevel,
  scope: Scope,
  solid: Manifold,
  minThickness: number
): { solid: Manifold; pieces: number; voids: number } {
  const pieces = solid.decompose().map((piece) => scope.keep(piece));
  const kept = pieces.filter((piece) => {
    const area = piece.surfaceArea();
    return area > 0 && (2 * Math.abs(piece.volume())) / area >= minThickness;
  });
  const solids = kept.filter((piece) => piece.volume() > 0).length;
  if (!solids) throw new FeatureBuildError('The extrude produced no solid.');
  const voids = kept.length - solids;
  if (kept.length === pieces.length) return { solid, pieces: solids, voids };
  return { solid: scope.keep(wasm.Manifold.compose(kept)), pieces: solids, voids };
}

/**
 * The solid's triangles, made to survive storage, verified by doing what
 * storage does to them.
 *
 * Faces Manifold marks as taken from a subtracted body move along their
 * outward normal, into that body, by eight float32 steps at the largest
 * coordinate or two millionths of the scene, whichever is more -- 2.4e-5 in
 * on a 12 in casting, far below any voxel. Storage rounds by half a step, so
 * the stored body still overlaps its neighbour. Two vertices still within a
 * few float32 steps of each other -- a pinch the push did not reach, or an
 * edge too short for float32 or for trimesh's 8-decimal weld -- are moved
 * apart (see `separate`). A mesh that would still weld into anything but a
 * closed solid is refused rather than stored.
 */
function watertightMesh(
  wasm: ManifoldToplevel,
  solid: Manifold,
  size: number
): { positions: Float32Array; indices: Uint32Array } {
  const mesh = solid.getMesh();
  const stride = mesh.numProp;
  const count = mesh.vertProperties.length / stride;
  const positions = new Float32Array(count * 3);
  let reach = 0;
  for (let i = 0; i < count; i += 1) {
    for (let k = 0; k < 3; k += 1) {
      const value = mesh.vertProperties[stride * i + k];
      positions[3 * i + k] = value;
      reach = Math.max(reach, Math.abs(value));
    }
  }
  const indices = new Uint32Array(mesh.triVerts);
  pushContactFaces(positions, indices, mesh.runIndex, mesh.runFlags, Math.max(size * 2e-6, reach * 2 ** -20, 2e-6));
  const close = Math.max(reach * 2 ** -21, 1e-7);
  const edges = edgeUses(indices, count);
  for (let pass = 0; pass < 4; pass += 1) {
    const pairs = closePairs(positions, close);
    if (!pairs.length) break;
    separate(positions, indices, pairs, edges, 2 * close * (pass + 1));
  }
  const problem = weldProblem(wasm, positions, indices, edges);
  if (problem) {
    throw new FeatureBuildError(
      `The result is not a closed solid once stored (${problem}). Move the sketch slightly or ` +
        'change the end condition so the body does not end on a seam between two bodies.'
    );
  }
  return { positions, indices };
}

/**
 * Move every vertex of a face taken from a subtracted body (a run Manifold
 * flags as backside) `distance` along the area-weighted normal of those faces
 * around it -- out of this body, into the one it was cut against. A vertex
 * where faces of opposite sides meet (the edge of a sheet) has no such normal
 * and stays.
 */
function pushContactFaces(
  positions: Float32Array,
  indices: Uint32Array,
  runIndex: Uint32Array | undefined,
  runFlags: Uint8Array | undefined,
  distance: number
): void {
  if (!runIndex || !runFlags || !runFlags.length) return;
  const count = positions.length / 3;
  const normals = new Float64Array(count * 3);
  const areas = new Float64Array(count);
  for (let run = 0; run < runFlags.length; run += 1) {
    if (!(runFlags[run] & 1)) continue;
    for (let t = runIndex[run]; t < runIndex[run + 1]; t += 3) {
      const a = indices[t];
      const b = indices[t + 1];
      const c = indices[t + 2];
      const ux = positions[3 * b] - positions[3 * a];
      const uy = positions[3 * b + 1] - positions[3 * a + 1];
      const uz = positions[3 * b + 2] - positions[3 * a + 2];
      const vx = positions[3 * c] - positions[3 * a];
      const vy = positions[3 * c + 1] - positions[3 * a + 1];
      const vz = positions[3 * c + 2] - positions[3 * a + 2];
      const nx = uy * vz - uz * vy;
      const ny = uz * vx - ux * vz;
      const nz = ux * vy - uy * vx;
      const area = Math.hypot(nx, ny, nz);
      for (const vertex of [a, b, c]) {
        normals[3 * vertex] += nx;
        normals[3 * vertex + 1] += ny;
        normals[3 * vertex + 2] += nz;
        areas[vertex] += area;
      }
    }
  }
  for (let vertex = 0; vertex < count; vertex += 1) {
    const length = Math.hypot(normals[3 * vertex], normals[3 * vertex + 1], normals[3 * vertex + 2]);
    if (!(length > 0.25 * areas[vertex])) continue;
    const scale = distance / length;
    for (let k = 0; k < 3; k += 1) {
      positions[3 * vertex + k] = Math.fround(positions[3 * vertex + k] + normals[3 * vertex + k] * scale);
    }
  }
}

/** Hash of a grid cell. Collisions only cost a distance test. */
function cellHash(cx: number, cy: number, cz: number): number {
  return Math.imul(cx | 0, 73856093) ^ Math.imul(cy | 0, 19349663) ^ Math.imul(cz | 0, 83492791);
}

/** How many triangles use each edge, keyed `min * count + max`. */
function edgeUses(indices: Uint32Array, count: number): Map<number, number> {
  const uses = new Map<number, number>();
  for (let t = 0; t < indices.length; t += 3) {
    for (let corner = 0; corner < 3; corner += 1) {
      const a = indices[t + corner];
      const b = indices[t + ((corner + 1) % 3)];
      const key = Math.min(a, b) * count + Math.max(a, b);
      uses.set(key, (uses.get(key) ?? 0) + 1);
    }
  }
  return uses;
}

/** Every pair of vertices closer than `gap`, as `[i, j]` with `i < j`. */
function closePairs(positions: Float32Array, gap: number): Array<[number, number]> {
  const cells = new Map<number, number[]>();
  const count = positions.length / 3;
  for (let i = 0; i < count; i += 1) {
    const key = cellHash(
      Math.floor(positions[3 * i] / gap),
      Math.floor(positions[3 * i + 1] / gap),
      Math.floor(positions[3 * i + 2] / gap)
    );
    const list = cells.get(key);
    if (list) list.push(i);
    else cells.set(key, [i]);
  }
  const pairs: Array<[number, number]> = [];
  const seen = new Set<number>();
  const gap2 = gap * gap;
  for (let i = 0; i < count; i += 1) {
    const x = positions[3 * i];
    const y = positions[3 * i + 1];
    const z = positions[3 * i + 2];
    const cx = Math.floor(x / gap);
    const cy = Math.floor(y / gap);
    const cz = Math.floor(z / gap);
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dz = -1; dz <= 1; dz += 1) {
          const list = cells.get(cellHash(cx + dx, cy + dy, cz + dz));
          if (!list) continue;
          for (const j of list) {
            // Hash collisions can list a cell twice; count each pair once.
            if (j <= i || seen.has(i * count + j)) continue;
            const ex = positions[3 * j] - x;
            const ey = positions[3 * j + 1] - y;
            const ez = positions[3 * j + 2] - z;
            if (ex * ex + ey * ey + ez * ez < gap2) {
              seen.add(i * count + j);
              pairs.push([i, j]);
            }
          }
        }
      }
    }
  }
  return pairs;
}

/**
 * Move the vertices of each close pair apart by about `step`.
 *
 * Two ends of one edge move away from each other along it, which lengthens
 * the edge without leaving the surface. Two vertices with no edge between
 * them are a pinch: each moves toward the centre of its own ring of
 * neighbours, which lies on its own side of the pinch.
 */
function separate(
  positions: Float32Array,
  indices: Uint32Array,
  pairs: Array<[number, number]>,
  edges: Map<number, number>,
  step: number
): void {
  const count = positions.length / 3;
  const moves = new Map<number, [number, number, number]>();
  const add = (vertex: number, dx: number, dy: number, dz: number) => {
    const move = moves.get(vertex) ?? [0, 0, 0];
    moves.set(vertex, [move[0] + dx, move[1] + dy, move[2] + dz]);
  };
  const pinched = new Set<number>();
  for (const [i, j] of pairs) {
    const ex = positions[3 * j] - positions[3 * i];
    const ey = positions[3 * j + 1] - positions[3 * i + 1];
    const ez = positions[3 * j + 2] - positions[3 * i + 2];
    const length = Math.hypot(ex, ey, ez);
    if (edges.has(i * count + j) && length > 0) {
      const s = step / (2 * length);
      add(i, -ex * s, -ey * s, -ez * s);
      add(j, ex * s, ey * s, ez * s);
    } else {
      pinched.add(i);
      pinched.add(j);
    }
  }
  if (pinched.size) {
    const sums = new Map<number, [number, number, number, number]>();
    for (let t = 0; t < indices.length; t += 3) {
      for (let corner = 0; corner < 3; corner += 1) {
        const vertex = indices[t + corner];
        if (!pinched.has(vertex)) continue;
        const sum = sums.get(vertex) ?? [0, 0, 0, 0];
        for (const other of [indices[t + ((corner + 1) % 3)], indices[t + ((corner + 2) % 3)]]) {
          sum[0] += positions[3 * other];
          sum[1] += positions[3 * other + 1];
          sum[2] += positions[3 * other + 2];
          sum[3] += 1;
        }
        sums.set(vertex, sum);
      }
    }
    for (const [vertex, [sx, sy, sz, n]] of sums) {
      const dx = sx / n - positions[3 * vertex];
      const dy = sy / n - positions[3 * vertex + 1];
      const dz = sz / n - positions[3 * vertex + 2];
      const length = Math.hypot(dx, dy, dz);
      if (length < 1e-12) continue;
      // Never past halfway to the ring centre, so a small ring is not folded.
      const s = Math.min(step, length / 2) / length;
      add(vertex, dx * s, dy * s, dz * s);
    }
  }
  for (const [vertex, [dx, dy, dz]] of moves) {
    positions[3 * vertex] = Math.fround(positions[3 * vertex] + dx);
    positions[3 * vertex + 1] = Math.fround(positions[3 * vertex + 1] + dy);
    positions[3 * vertex + 2] = Math.fround(positions[3 * vertex + 2] + dz);
  }
}

/**
 * Why the triangles would not survive storage, or null: what the server, the
 * STL file and the next build all do to them -- weld equal positions -- must
 * leave every edge on exactly two triangles, and the result must load back
 * into Manifold as a solid.
 */
function weldProblem(
  wasm: ManifoldToplevel,
  positions: Float32Array,
  indices: Uint32Array,
  edges: Map<number, number>
): string | null {
  // trimesh welds positions equal to 8 decimals: vertices 4e-8 apart differ
  // by more than that in some coordinate, and STL keeps any two distinct
  // float32 positions apart.
  const crowded = closePairs(positions, 4e-8).length;
  if (crowded) return `${crowded} pairs of vertices share a position`;
  for (let t = 0; t < indices.length; t += 3) {
    const [a, b, c] = [indices[t], indices[t + 1], indices[t + 2]];
    if (a === b || b === c || a === c) return 'a triangle has a repeated corner';
  }
  let open = 0;
  for (const uses of edges.values()) if (uses !== 2) open += 1;
  if (open) return `${open} edges are not shared by exactly two triangles`;
  try {
    const mesh = new wasm.Mesh({ numProp: 3, vertProperties: positions, triVerts: indices });
    mesh.merge();
    const reloaded = new wasm.Manifold(mesh);
    const status = reloaded.status();
    reloaded.delete();
    if (status !== 'NoError') return status;
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
  return null;
}

/**
 * The profile swept along +Z for `length`, with the draft applied.
 *
 * A draft is a taper of `length * tan(angle)` on every side. Each separate
 * region of the profile is scaled about its own centre so that its width and
 * its height each change by twice that, which is an exact draft for the
 * circles and rectangles risers and chills are drawn with and a close one for
 * anything else. An inward draft that would close the profile before the end
 * stops at the apex instead.
 */
function extrudeProfile(
  wasm: ManifoldToplevel,
  scope: Scope,
  profile: CrossSection,
  length: number,
  feature: Pick<ExtrudeFeature, 'draftDeg' | 'draftOutward'>
): Manifold {
  const { Manifold } = wasm;
  const draft = Math.tan((Math.max(feature.draftDeg, 0) * Math.PI) / 180);
  if (draft <= 1e-9) return scope.keep(profile.extrude(length));

  const sign = feature.draftOutward ? 1 : -1;
  const regions = profile.decompose().map((region) => scope.keep(region));
  const solids = regions.map((region) => {
    const bounds = region.bounds();
    const cx = (bounds.min[0] + bounds.max[0]) / 2;
    const cy = (bounds.min[1] + bounds.max[1]) / 2;
    const width = Math.max(bounds.max[0] - bounds.min[0], 1e-9);
    const height = Math.max(bounds.max[1] - bounds.min[1], 1e-9);
    let run = length;
    if (sign < 0) {
      // Where the narrower side would reach zero.
      const apex = Math.min(width, height) / (2 * draft);
      run = Math.min(length, apex * 0.999);
    }
    const sx = Math.max((width + sign * 2 * run * draft) / width, 0);
    const sy = Math.max((height + sign * 2 * run * draft) / height, 0);
    const centred = scope.keep(region.translate([-cx, -cy]));
    const tapered = scope.keep(centred.extrude(run, 0, 0, [sx, sy]));
    return scope.keep(tapered.translate([cx, cy, 0]));
  });
  return scope.keep(solids.length === 1 ? solids[0] : Manifold.union(solids));
}

/**
 * The bodies whose bounding boxes reach `localBox` (given in the feature's
 * local frame), carried into that frame. The test is made in world space on
 * the cached boxes, so a body nowhere near the feature costs nothing. A body
 * that is not a closed solid is an error only if it is in the way.
 */
function localBodies(
  scope: Scope,
  bodies: KernelBody[],
  toLocal: Mat4,
  toWorld: Mat4,
  localBox: Box3,
  pad: number
): Manifold[] {
  const worldBox = transformBox(localBox, toWorld);
  const local: Manifold[] = [];
  for (const body of bodies) {
    if (!body.bbox || !boxesOverlap(body.bbox, worldBox, pad)) continue;
    if (!body.manifold) {
      throw new FeatureBuildError(body.error ?? `"${body.name}" cannot bound an extrude.`);
    }
    local.push(scope.keep(body.manifold.transform(toLocal)));
  }
  return local;
}

function transformBox(box: Box3, m: Mat4): Box3 {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const [x, y, z] of boxCorners(box)) {
    const p: Vec3 = [
      m[0] * x + m[4] * y + m[8] * z + m[12],
      m[1] * x + m[5] * y + m[9] * z + m[13],
      m[2] * x + m[6] * y + m[10] * z + m[14]
    ];
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], p[axis]);
      max[axis] = Math.max(max[axis], p[axis]);
    }
  }
  return { min, max };
}

/** World-space bounds of a mesh's vertices under a part transform. */
function meshBounds(positions: Float32Array, m: Mat4): Box3 | null {
  if (positions.length < 3) return null;
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    const p: Vec3 = [
      m[0] * x + m[4] * y + m[8] * z + m[12],
      m[1] * x + m[5] * y + m[9] * z + m[13],
      m[2] * x + m[6] * y + m[10] * z + m[14]
    ];
    for (let axis = 0; axis < 3; axis += 1) {
      if (p[axis] < min[axis]) min[axis] = p[axis];
      if (p[axis] > max[axis]) max[axis] = p[axis];
    }
  }
  return { min, max };
}

/** How far the bodies reach along `direction` from `origin`, or null with no bodies. */
function extentAlong(bodies: KernelBody[], origin: Vec3, direction: Vec3): number | null {
  let extent: number | null = null;
  for (const body of bodies) {
    if (!body.bbox) continue;
    for (const corner of boxCorners(body.bbox)) {
      const s =
        (corner[0] - origin[0]) * direction[0] +
        (corner[1] - origin[1]) * direction[1] +
        (corner[2] - origin[2]) * direction[2];
      extent = extent === null ? s : Math.max(extent, s);
    }
  }
  return extent;
}

/** A length scale for tolerances: the largest of the bodies and the profile. */
function sceneSize(bodies: KernelBody[], profile: CrossSection): number {
  const bounds = profile.bounds();
  let size = Math.max(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1]);
  for (const body of bodies) {
    if (!body.bbox) continue;
    for (let axis = 0; axis < 3; axis += 1) size = Math.max(size, body.bbox.max[axis] - body.bbox.min[axis]);
  }
  return Math.max(size, 1);
}
