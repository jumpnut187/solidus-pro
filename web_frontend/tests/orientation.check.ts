/**
 * Checks the browser's orientation math and part placement against the engine.
 *
 *   npm run check:orientation
 *
 * `orientation.fixture.json` holds cases computed by `core/orientation.py`
 * (written by `python -m tests.test_web_orientation --write-fixture`, and
 * checked to be current by that test). Every function of
 * `src/lib/orientation.ts` must reproduce them, and a Babylon node placed by
 * `applyPartTransform` must put the fixture's vertices exactly where
 * `core.orientation.world_bounds` puts them.
 */
import { NullEngine } from '@babylonjs/core/Engines/nullEngine';
import { Vector3 } from '@babylonjs/core/Maths/math.vector';
import { TransformNode } from '@babylonjs/core/Meshes/transformNode';
import { Scene } from '@babylonjs/core/scene';

import {
  cleanAngle,
  dropToFloorDz,
  eulerToMatrix,
  faceToDirection,
  matrixToEuler,
  rotateAboutWorldAxis,
  snapAngle,
  snapToPartingDz,
  worldBounds,
  type Mat3,
  type Vec3,
  type WorldAxis,
  type WorldBounds
} from '../src/lib/orientation';
import { applyPartTransform } from '../src/viewport/partTransform';
import fixture from './orientation.fixture.json';

let failures = 0;
function check(name: string, condition: boolean, detail = ''): void {
  if (condition) console.log(`  ok   ${name}`);
  else {
    failures += 1;
    console.log(`  FAIL ${name} ${detail}`);
  }
}

function close(a: unknown, b: unknown, tol = 1e-6): boolean {
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => close(x, b[i], tol));
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) <= tol;
  return a === b;
}

/** Euler triples are equal when they name the same angles, 180 and -180 included. */
function sameAngles(a: Vec3, b: number[]): boolean {
  return a.every((value, i) => Math.abs(cleanAngle(value - b[i])) <= 1e-5);
}

console.log('euler_to_matrix');
for (const item of fixture.euler_to_matrix) {
  const got = eulerToMatrix(item.rotation as Vec3);
  check(`R(${item.rotation})`, close(got, item.matrix, 1e-12), JSON.stringify(got));
}

console.log('matrix_to_euler');
for (const item of fixture.matrix_to_euler) {
  const got = matrixToEuler(item.matrix as Mat3);
  check(`euler(${item.euler})`, close(got, item.euler, 1e-5), JSON.stringify(got));
}

console.log('rotate_about_world_axis');
for (const item of fixture.rotate_about_world_axis) {
  const got = rotateAboutWorldAxis(
    item.rotation as Vec3,
    item.position as Vec3,
    item.axis as WorldAxis | Vec3,
    item.angle,
    item.pivot as Vec3
  );
  const [rotation, position] = item.result;
  check(
    `${item.axis} ${item.angle} from ${item.rotation}`,
    sameAngles(got.rotation, rotation) && close(got.position, position, 1e-9),
    JSON.stringify(got)
  );
}

console.log('face_to_direction');
for (const item of fixture.face_to_direction) {
  const got = faceToDirection(
    item.rotation as Vec3,
    item.position as Vec3,
    item.normal as Vec3,
    item.pivot as Vec3,
    item.target as Vec3
  );
  const [rotation, position] = item.result;
  // The fixture's own matrix is the reference: two Euler triples can name
  // the same turn, so compare what they do rather than how they are spelled.
  const same = close(eulerToMatrix(got.rotation), eulerToMatrix(rotation as Vec3), 1e-6);
  check(`normal ${item.normal} -> ${item.target}`, same && close(got.position, position, 1e-9), JSON.stringify(got));
}

console.log('world_bounds (lib/orientation.ts)');
for (const item of fixture.world_bounds) {
  const got = worldBounds(fixture.vertices, {
    rotation: item.rotation as Vec3,
    position: item.position as Vec3,
    scale: item.scale
  });
  check(`rot ${item.rotation} pos ${item.position} s ${item.scale}`, close(got, item.bounds, 1e-9), JSON.stringify(got));
}

console.log('world_bounds (Babylon node placed by applyPartTransform)');
{
  const engine = new NullEngine();
  const scene = new Scene(engine);
  scene.useRightHandedSystem = true;
  const node = new TransformNode('part', scene);
  for (const item of fixture.world_bounds) {
    applyPartTransform(node, {
      rotation_x: item.rotation[0],
      rotation_y: item.rotation[1],
      rotation_z: item.rotation[2],
      position_x: item.position[0],
      position_y: item.position[1],
      position_z: item.position[2],
      scale: item.scale
    });
    const world = node.computeWorldMatrix(true);
    const min: Vec3 = [Infinity, Infinity, Infinity];
    const max: Vec3 = [-Infinity, -Infinity, -Infinity];
    let vertexError = 0;
    const r = eulerToMatrix(item.rotation as Vec3);
    for (let i = 0; i < fixture.vertices.length; i += 3) {
      const local = new Vector3(fixture.vertices[i], fixture.vertices[i + 1], fixture.vertices[i + 2]);
      const p = Vector3.TransformCoordinates(local, world);
      const xyz: Vec3 = [p.x, p.y, p.z];
      for (let axis = 0; axis < 3; axis += 1) {
        min[axis] = Math.min(min[axis], xyz[axis]);
        max[axis] = Math.max(max[axis], xyz[axis]);
        // Vertex by vertex too: equal boxes could still hide a mirrored part.
        const expected =
          item.scale * (r[axis][0] * local.x + r[axis][1] * local.y + r[axis][2] * local.z) + item.position[axis];
        vertexError = Math.max(vertexError, Math.abs(expected - xyz[axis]));
      }
    }
    const bounds: WorldBounds = [min, max];
    check(
      `babylon rot ${item.rotation} pos ${item.position} s ${item.scale}`,
      close(bounds, item.bounds, 1e-5) && vertexError < 1e-5,
      `${JSON.stringify(bounds)} vertex error ${vertexError}`
    );
  }
  // The old `mesh.rotation` path, for the record: it must NOT agree on a
  // compound rotation, or this check is not testing anything.
  const legacy = new TransformNode('legacy', scene);
  legacy.rotation.set((30 * Math.PI) / 180, (-20 * Math.PI) / 180, (75 * Math.PI) / 180);
  legacy.position.set(1, 2, 3);
  const legacyWorld = legacy.computeWorldMatrix(true);
  let legacyMinZ = Infinity;
  for (let i = 0; i < fixture.vertices.length; i += 3) {
    const p = Vector3.TransformCoordinates(
      new Vector3(fixture.vertices[i], fixture.vertices[i + 1], fixture.vertices[i + 2]),
      legacyWorld
    );
    legacyMinZ = Math.min(legacyMinZ, p.z);
  }
  check(
    'euler mesh.rotation differs from the engine (the bug this fixes)',
    Math.abs(legacyMinZ - fixture.world_bounds[0].bounds[0][2]) > 1e-3,
    String(legacyMinZ)
  );
  scene.dispose();
  engine.dispose();
}

console.log('drop / snap / angle');
for (const item of fixture.drop_to_floor_dz) {
  const got = dropToFloorDz(item.bounds as WorldBounds, item.floor_z);
  check(`drop ${JSON.stringify(item.bounds)} to ${item.floor_z}`, close(got, item.dz, 1e-12), String(got));
}
for (const item of fixture.snap_to_parting_dz) {
  const got = snapToPartingDz(item.bounds as WorldBounds, item.parting_z, item.side as 'above' | 'below');
  check(`snap ${item.side} ${item.parting_z}`, close(got, item.dz, 1e-12), String(got));
}
for (const item of fixture.snap_angle) {
  const got = snapAngle(item.angle, item.step);
  check(`snap_angle(${item.angle}, ${item.step})`, close(got, item.result, 1e-9), String(got));
}

if (failures) {
  console.log(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log('\nall orientation checks passed');
