/**
 * Picking that agrees with what is on screen.
 *
 * `scene.pick` tests triangles, and a triangle is still a triangle after a
 * clip plane has discarded every fragment of it. With a section open that
 * made half of all clicks land on the cut-away half of the casting -- a probe
 * reading, a measurement end or a pour point on metal the user could not see.
 * Nothing in Babylon's ray code reads a material's clip planes, so this module
 * does, and asks of every candidate hit the same questions the fragment shader
 * asks of every fragment.
 *
 * Two of those questions are answered here: the clip planes, and back-face
 * culling. The second only matters once a section is open. Uncut, the ray
 * meets a closed casting's near and far walls in the same silhouette either
 * way; cut open, one of the two walls is the one drawn and the other is
 * culled, and a pick has to land on the one that is on screen.
 *
 * Anything decided by a texture -- a section cap discarding wherever the
 * occupancy grid holds no metal -- is not geometry and cannot be answered
 * here, so callers pass `accept` for it.
 */
import { Material } from '@babylonjs/core/Materials/material';
import { Matrix, Vector3 } from '@babylonjs/core/Maths/math.vector';
import { Mesh } from '@babylonjs/core/Meshes/mesh';
import type { Camera } from '@babylonjs/core/Cameras/camera';
import type { PickingInfo } from '@babylonjs/core/Collisions/pickingInfo';
import type { Ray } from '@babylonjs/core/Culling/ray';
import type { Plane } from '@babylonjs/core/Maths/math.plane';
import type { AbstractMesh } from '@babylonjs/core/Meshes/abstractMesh';
import type { Scene } from '@babylonjs/core/scene';
import type { Nullable } from '@babylonjs/core/types';
// Adds `scene.createPickingRay`, `scene.pick` and the rest of picking to the
// Scene prototype. The input manager's own pointer picking needs it too.
import '@babylonjs/core/Culling/ray';

export type PickVisibleOptions = {
  /** Which meshes may be hit at all. Disabled and hidden meshes are excluded regardless. */
  predicate: (mesh: AbstractMesh) => boolean;
  /**
   * A last say over a hit the geometry allows, for visibility decided in a
   * shader. A rejected hit is skipped and the next one behind it considered.
   */
  accept?: (hit: PickingInfo) => boolean;
  /** Defaults to the scene's active camera, as `scene.pick` does. */
  camera?: Camera | null;
  /**
   * Skip faces the renderer culls. On by default, because that is what
   * "visible" means; off to find the wall of a body from inside it, which is
   * how the part behind a section cap is found.
   */
  cullBackFaces?: boolean;
};

/**
 * The clip planes Babylon will actually apply to this mesh.
 *
 * Read from the material with the scene as fallback, slot by slot, exactly as
 * `BindClipPlane` resolves them. Reading them here rather than being handed a
 * list is what lets a section cap work unchanged: it deliberately carries the
 * other two axes' planes and not its own, because it lies on its own plane
 * where the sign of the clip distance is rounding.
 */
export function clipPlanesOf(mesh: AbstractMesh, scene: Scene): Plane[] {
  const material = mesh.material;
  const slots: Array<Nullable<Plane> | undefined> = [
    material?.clipPlane ?? scene.clipPlane,
    material?.clipPlane2 ?? scene.clipPlane2,
    material?.clipPlane3 ?? scene.clipPlane3,
    material?.clipPlane4 ?? scene.clipPlane4,
    material?.clipPlane5 ?? scene.clipPlane5,
    material?.clipPlane6 ?? scene.clipPlane6
  ];
  return slots.filter((plane): plane is Plane => Boolean(plane));
}

/**
 * Whether a world point survives every plane. The shader discards where
 * `dot(plane, point) > 0`, so a point exactly on a plane is kept.
 */
export function isOnKeptSide(point: Vector3, planes: readonly Plane[]): boolean {
  return planes.every((plane) => plane.dotCoordinate(point) <= 0);
}

/**
 * The nearest hit a user could actually see under the pointer, or null.
 *
 * `x` and `y` are in the same render-pixel space as `scene.pointerX`.
 */
export function pickVisible(
  scene: Scene,
  x: number,
  y: number,
  options: PickVisibleOptions
): PickingInfo | null {
  return pickVisibleAll(scene, x, y, options)[0] ?? null;
}

/**
 * Every mesh's nearest visible hit under the pointer, nearest first.
 *
 * One hit per mesh, as `scene.multiPick` gives, but each already filtered for
 * clip planes and culling -- so a mesh cut open still reports the part of it
 * that is left. Callers that need to know what lies *behind* the first hit
 * (the part a section cap is cutting, say) read further down the list.
 */
export function pickVisibleAll(
  scene: Scene,
  x: number,
  y: number,
  options: PickVisibleOptions
): PickingInfo[] {
  const camera = options.camera ?? scene.activeCamera;
  if (!camera) return [];
  const ray = scene.createPickingRay(x, y, Matrix.Identity(), camera);

  const hits: PickingInfo[] = [];
  for (const mesh of scene.meshes) {
    if (!mesh.isEnabled() || !mesh.isVisible || mesh.visibility <= 0) continue;
    if (!options.predicate(mesh)) continue;
    const hit = intersectVisible(ray, mesh, scene, options.cullBackFaces ?? true);
    if (!hit?.hit || !hit.pickedPoint) continue;
    if (options.accept && !options.accept(hit)) continue;
    hits.push(hit);
  }
  hits.sort((a, b) => a.distance - b.distance);
  return hits;
}

/**
 * The nearest triangle of one mesh that the renderer would draw at this ray.
 *
 * Triangles are rejected in Babylon's own `trianglePredicate`, so the search
 * carries on past a clipped or culled face to the next one along the ray
 * rather than giving up on the whole mesh. The triangle positions it sees are
 * in mesh-local space, so the planes are carried into that space once per
 * mesh instead of every vertex being carried out to the world.
 */
function intersectVisible(
  ray: Ray,
  mesh: AbstractMesh,
  scene: Scene,
  cullBackFaces: boolean
): PickingInfo | null {
  const world = mesh.computeWorldMatrix(true);
  const localPlanes = clipPlanesOf(mesh, scene).map((plane) => toLocalPlane(plane, world));
  // An uncut mesh is picked exactly as `scene.pick` would pick it, so a view
  // with no section open behaves as it always has.
  if (!localPlanes.length) return ray.intersectsMesh(mesh, false);
  const cullSign = cullBackFaces ? backFaceSign(mesh, scene, world) : 0;

  return ray.intersectsMesh(mesh, false, (p0, p1, p2, localRay) => {
    if (cullSign !== 0) {
      p1.subtractToRef(p0, EDGE_A);
      p2.subtractToRef(p0, EDGE_B);
      Vector3.CrossToRef(EDGE_A, EDGE_B, FACE_NORMAL);
      if (Vector3.Dot(FACE_NORMAL, localRay.direction) * cullSign > 0) return false;
    }

    // A whole triangle on one side needs no intersection point; only one
    // straddling a plane does, and those are a thin band along the cut.
    let straddles = false;
    for (const plane of localPlanes) {
      const d0 = plane.dot(p0);
      const d1 = plane.dot(p1);
      const d2 = plane.dot(p2);
      if (d0 > 0 && d1 > 0 && d2 > 0) return false;
      if (d0 > 0 || d1 > 0 || d2 > 0) straddles = true;
    }
    if (!straddles) return true;

    const crossing = localRay.intersectsTriangle(p0, p1, p2);
    if (!crossing) return false;
    const point = localRay.origin.add(localRay.direction.scale(crossing.distance));
    return localPlanes.every((plane) => plane.dot(point) <= 0);
  });
}

// Scratch space for the per-triangle culling test, which runs for every
// candidate triangle of every cut mesh on every pick.
const EDGE_A = new Vector3();
const EDGE_B = new Vector3();
const FACE_NORMAL = new Vector3();

/** A world-space plane as coefficients against mesh-local points. */
type LocalPlane = { dot: (point: Vector3) => number };

/**
 * Carry a world plane into mesh-local coordinates.
 *
 * For a local point `p` the world point is `p * W` (Babylon's row-vector
 * convention), so `plane . (p * W)` is `p . (W * plane)`. Only the sign is
 * ever read, so a scaled mesh needs no renormalization.
 */
function toLocalPlane(plane: Plane, world: Matrix): LocalPlane {
  const m = world.m;
  const { x: a, y: b, z: c } = plane.normal;
  const d = plane.d;
  const la = m[0] * a + m[1] * b + m[2] * c + m[3] * d;
  const lb = m[4] * a + m[5] * b + m[6] * c + m[7] * d;
  const lc = m[8] * a + m[9] * b + m[10] * c + m[11] * d;
  const ld = m[12] * a + m[13] * b + m[14] * c + m[15] * d;
  return { dot: (point) => la * point.x + lb * point.y + lc * point.z + ld };
}

/**
 * Which way a culled triangle's winding faces the ray, or 0 for no culling.
 *
 * A triangle is culled when `cross(p1 - p0, p2 - p0) . direction * sign > 0`.
 * Derived from the same state Babylon hands to `setState` -- the material's
 * or else the mesh's side orientation, flipped by a mirroring world matrix --
 * so the answer follows the renderer whatever it is configured to do. The one
 * fixed point is measured rather than derived: in this right-handed scene,
 * with clockwise orientation, an unmirrored mesh and back faces culled, the
 * faces that disappear are the ones with `cross . direction < 0`, found by
 * rendering the parts opaque with each side culled in turn and comparing
 * against both sides drawn. Handedness, orientation, a mirroring matrix and
 * `cullBackFaces` each swap the culled side, so each flips the sign.
 *
 * Babylon only draws both sides when `backFaceCulling` is off; a separate
 * culling pass merely orders them, so it does not enter into it.
 */
function backFaceSign(mesh: AbstractMesh, scene: Scene, world: Matrix): number {
  const material = mesh.material;
  if (!material || !material.backFaceCulling) return 0;
  let sign = -1;
  if (!scene.useRightHandedSystem) sign = -sign;
  const orientation =
    material.sideOrientation ?? (mesh instanceof Mesh ? mesh.sideOrientation : Material.ClockWiseSideOrientation);
  if (orientation === Material.CounterClockWiseSideOrientation) sign = -sign;
  if (world.determinant() < 0) sign = -sign;
  if (!material.cullBackFaces) sign = -sign;
  return sign;
}
