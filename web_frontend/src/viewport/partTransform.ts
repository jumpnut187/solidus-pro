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
