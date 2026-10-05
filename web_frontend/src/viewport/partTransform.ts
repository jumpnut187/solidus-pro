/**
 * A part's placement on a Babylon node, exactly as the engine places it.
 *
 * The engine turns a part with static-XYZ Euler angles, `R = Rz @ Ry @ Rx`
 * (`core.part_manager.Part._apply_transform`). Babylon's `node.rotation` is a
 * yaw-pitch-roll triple applied in a different order, so any rotation about
 * more than one axis drew the part somewhere other than where it was
 * simulated. The rotation is set as a quaternion of the engine's own matrix
 * instead; `tests/orientation.check.ts` compares the resulting world vertices
 * with `core.orientation.world_bounds`.
 */
import { Quaternion } from '@babylonjs/core/Maths/math.vector';
import type { TransformNode } from '@babylonjs/core/Meshes/transformNode';

import type { ProjectPart } from '../lib/api';
import { partQuaternion } from '../lib/orientation';

export type PartPlacementFields = Pick<
  ProjectPart,
  'position_x' | 'position_y' | 'position_z' | 'rotation_x' | 'rotation_y' | 'rotation_z' | 'scale'
>;

export function applyPartTransform(node: TransformNode, part: PartPlacementFields): void {
  node.position.set(part.position_x, part.position_y, part.position_z);
  const [x, y, z, w] = partQuaternion([part.rotation_x, part.rotation_y, part.rotation_z]);
  if (node.rotationQuaternion) node.rotationQuaternion.set(x, y, z, w);
  else node.rotationQuaternion = new Quaternion(x, y, z, w);
  const scale = part.scale || 1;
  node.scaling.set(scale, scale, scale);
}
