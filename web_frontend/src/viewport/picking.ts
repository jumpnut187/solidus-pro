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
import '@babylonjs/core/Culling/ray';

export type PickVisibleOptions = {
  predicate: (mesh: AbstractMesh) => boolean;
  accept?: (hit: PickingInfo) => boolean;
  camera?: Camera | null;
  cullBackFaces?: boolean;
};

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

export function isOnKeptSide(point: Vector3, planes: readonly Plane[]): boolean {
  return planes.every((plane) => plane.dotCoordinate(point) <= 0);
}

export function pickVisible(
  scene: Scene,
  x: number,
  y: number,
  options: PickVisibleOptions
): PickingInfo | null {
  return pickVisibleAll(scene, x, y, options)[0] ?? null;
}

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

function intersectVisible(
  ray: Ray,
  mesh: AbstractMesh,
  scene: Scene,
  cullBackFaces: boolean
): PickingInfo | null {
  const world = mesh.computeWorldMatrix(true);
  const localPlanes = clipPlanesOf(mesh, scene).map((plane) => toLocalPlane(plane, world));
  if (!localPlanes.length) return ray.intersectsMesh(mesh, false);
  const cullSign = cullBackFaces ? backFaceSign(mesh, scene, world) : 0;

  return ray.intersectsMesh(mesh, false, (p0, p1, p2, localRay) => {
    if (cullSign !== 0) {
      p1.subtractToRef(p0, EDGE_A);
      p2.subtractToRef(p0, EDGE_B);
      Vector3.CrossToRef(EDGE_A, EDGE_B, FACE_NORMAL);
      if (Vector3.Dot(FACE_NORMAL, localRay.direction) * cullSign > 0) return false;
    }

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

const EDGE_A = new Vector3();
const EDGE_B = new Vector3();
const FACE_NORMAL = new Vector3();

type LocalPlane = { dot: (point: Vector3) => number };

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
