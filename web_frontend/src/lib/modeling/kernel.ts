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
  planeFrame,
  sketchProfiles,
  type ExtrudeFeature,
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

  const { CrossSection, Manifold } = wasm;
  const scope = new Scope();
  const warnings: string[] = [];
  try {
    const frame = planeFrame(feature.sketch.plane);
    const w: Vec3 = feature.reverse
      ? [-frame.normal[0], -frame.normal[1], -frame.normal[2]]
      : frame.normal;
    const toWorld = frameMatrix(frame.u, frame.v, w, frame.origin);
    const toLocal = inverseFrame(frame.u, frame.v, w, frame.origin);

    // Every closed contour goes in together under the even-odd rule, the
    // way a sketch's contours become regions: a circle inside a rectangle is
    // a hole in it, which is how a ring chill or a sleeve is drawn.
    const contours: MVec2[][] = sketchProfiles(feature.sketch.entities).loops.map((loop) =>
      loop.map((point) => [point[0], point[1]] as MVec2)
    );
    const profile = scope.keep(new CrossSection(contours, 'EvenOdd'));
    if (profile.isEmpty() || profile.area() < 1e-6) {
      throw new FeatureBuildError('The sketch encloses no area.');
    }

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

    const world = scope.keep(solid.transform(toWorld));
    if (world.isEmpty()) throw new FeatureBuildError('The extrude produced no solid.');
    if (world.status() !== 'NoError') {
      throw new FeatureBuildError(`The geometry kernel rejected the result (${world.status()}).`);
    }
    const pieces = world.decompose();
    if (pieces.length > 1) warnings.push(`The body is in ${pieces.length} separate pieces.`);
    for (const piece of pieces) piece.delete();

    const mesh = world.getMesh();
    const stride = mesh.numProp;
    const vertexCount = mesh.vertProperties.length / stride;
    const positions = new Float32Array(vertexCount * 3);
    for (let i = 0; i < vertexCount; i += 1) {
      positions[3 * i] = mesh.vertProperties[stride * i];
      positions[3 * i + 1] = mesh.vertProperties[stride * i + 1];
      positions[3 * i + 2] = mesh.vertProperties[stride * i + 2];
    }
    const indices = new Uint32Array(mesh.triVerts);
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
  feature: ExtrudeFeature
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
