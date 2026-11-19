"""
Regression tests for voxel neighbor canonicalization.

The 2x search radius used for part-interface bridging can return voxels at
2 spacings along a direction in addition to the true adjacent voxel. The
canonical slot assignment must keep the NEAREST candidate per direction
slot: a 2-spacing voxel in a face slot makes the finite-difference stencil
treat a 2*dx distance as dx (4x diffusivity error, direction-dependent).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from core.voxel_mesh import AdaptiveVoxelMesh


class NeighborCanonicalizationTests(unittest.TestCase):
    def _mesh_like(self) -> SimpleNamespace:
        """Minimal object exposing what _canonicalize_neighbor_order uses."""
        centers = np.array(
            [
                [0.0, 0.0, 0.0],   # 0 self
                [0.1, 0.0, 0.0],   # 1 +X near
                [0.2, 0.0, 0.0],   # 2 +X far
                [-0.1, 0.0, 0.0],  # 3 -X
                [0.0, 0.1, 0.0],   # 4 +Y
                [0.1, 0.1, 0.0],   # 5 +X+Y near
                [0.2, 0.2, 0.0],   # 6 +X+Y far
                [0.1, 0.1, 0.1],   # 7 +X+Y+Z near
                [0.2, 0.2, 0.2],   # 8 +X+Y+Z far
            ],
            dtype=np.float32,
        )
        return SimpleNamespace(voxel_centers=centers, base_voxel_size=0.1)

    def test_face_slot_keeps_nearest_candidate(self) -> None:
        mesh = self._mesh_like()
        center = np.asarray(mesh.voxel_centers[0])
        # Far +X candidate listed FIRST (KD-tree order is arbitrary).
        neighbor_indices = [2, 1, 3]
        canonical = AdaptiveVoxelMesh._canonicalize_neighbor_order(
            mesh, 0, center, neighbor_indices
        )
        self.assertEqual(int(canonical[0]), 1, "+X slot must hold the adjacent voxel")
        self.assertEqual(int(canonical[1]), 3, "-X slot")

    def test_edge_and_corner_slots_keep_nearest_candidate(self) -> None:
        mesh = self._mesh_like()
        center = np.asarray(mesh.voxel_centers[0])
        neighbor_indices = [6, 5, 8, 7]
        canonical = AdaptiveVoxelMesh._canonicalize_neighbor_order(
            mesh, 0, center, neighbor_indices
        )
        self.assertEqual(int(canonical[6]), 5, "+X+Y edge slot must hold the adjacent voxel")
        self.assertEqual(int(canonical[18]), 7, "+X+Y+Z corner slot must hold the adjacent voxel")

    def test_batch_canonicalization_matches_single(self) -> None:
        mesh = self._mesh_like()
        raw = np.full((1, 4), -1, dtype=np.int32)
        raw[0, :] = [2, 1, 3, -1]
        canonical = AdaptiveVoxelMesh._canonicalize_batch(mesh, raw)
        self.assertEqual(int(canonical[0, 0]), 1)
        self.assertEqual(int(canonical[0, 1]), 3)


if __name__ == "__main__":
    unittest.main()
