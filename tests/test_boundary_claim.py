from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import trimesh

from core.part_manager import PART_TYPE_CAST, PART_TYPE_CHILL, PART_TYPE_RISER
from core.voxel_mesh import AdaptiveVoxelMesh

def _mesh(centers, size=0.25):
    n = len(centers)
    return SimpleNamespace(
        voxel_centers=np.asarray(centers, dtype=np.float64),
        voxel_sizes=np.full(n, size),
        voxel_part_types=np.zeros(n, dtype=np.int8),
        _BOUNDARY_CLAIM_MAX_CANDIDATES=AdaptiveVoxelMesh._BOUNDARY_CLAIM_MAX_CANDIDATES,
    )

class BoundaryClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cast = trimesh.creation.box(extents=[6.0, 6.0, 2.0])
        self.cast.apply_translation([0.0, 0.0, 1.0])
        self.riser = trimesh.creation.cylinder(radius=2.0, height=4.0, sections=64)
        self.riser.apply_translation([0.0, 0.0, 4.0])

    def _claim(self, mesh, others):
        AdaptiveVoxelMesh._claim_unowned_boundary_voxels(
            mesh, [(self.cast, PART_TYPE_CAST)], others
        )

    def test_cell_just_outside_riser_wall_becomes_riser(self) -> None:
        mesh = _mesh([[2.15, 0.0, 4.0]])
        self._claim(mesh, [(self.riser, PART_TYPE_RISER)])
        self.assertEqual(int(mesh.voxel_part_types[0]), PART_TYPE_RISER)

    def test_cell_reaching_the_casting_stays_cast(self) -> None:
        mesh = _mesh([[1.0, 0.0, 1.9], [2.9, 0.0, 1.0]])
        self._claim(mesh, [(self.riser, PART_TYPE_RISER)])
        self.assertEqual(mesh.voxel_part_types.tolist(), [PART_TYPE_CAST, PART_TYPE_CAST])

    def test_cell_far_from_every_part_stays_cast(self) -> None:
        mesh = _mesh([[0.0, 0.0, 9.0]])
        self._claim(mesh, [(self.riser, PART_TYPE_RISER)])
        self.assertEqual(int(mesh.voxel_part_types[0]), PART_TYPE_CAST)

    def test_higher_reaching_part_wins(self) -> None:
        chill = trimesh.creation.box(extents=[1.0, 6.0, 4.0])
        chill.apply_translation([2.8, 0.0, 4.0])
        mesh = _mesh([[2.6, 0.0, 4.0]])
        self._claim(mesh, [(self.riser, PART_TYPE_RISER), (chill, PART_TYPE_CHILL)])
        self.assertEqual(int(mesh.voxel_part_types[0]), PART_TYPE_CHILL)

if __name__ == "__main__":
    unittest.main()
