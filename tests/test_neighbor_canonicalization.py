from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from core.voxel_mesh import AdaptiveVoxelMesh

class NeighborCanonicalizationTests(unittest.TestCase):
    def _mesh_like(self) -> SimpleNamespace:
        centers = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [-0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
                [0.2, 0.2, 0.0],
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],
            ],
            dtype=np.float32,
        )
        return SimpleNamespace(voxel_centers=centers, base_voxel_size=0.1)

    def test_face_slot_keeps_nearest_candidate(self) -> None:
        mesh = self._mesh_like()
        center = np.asarray(mesh.voxel_centers[0])
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

class VectorizedCanonicalizationTests(unittest.TestCase):

    @staticmethod
    def _lattice_with_gaps(shape: tuple[int, int, int], pitch: float, keep: float, seed: int) -> SimpleNamespace:
        rng = np.random.default_rng(seed)
        grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).reshape(-1, 3)
        grid = grid[rng.random(len(grid)) < keep]
        rng.shuffle(grid)
        centers = (grid * pitch + np.array([0.37, -1.2, 5.5])).astype(np.float32)
        return SimpleNamespace(voxel_centers=centers, base_voxel_size=pitch)

    def _reference(self, mesh: SimpleNamespace, tree, radius: float) -> np.ndarray:
        centers = np.asarray(mesh.voxel_centers)
        rows = np.full((len(centers), 26), -1, dtype=np.int32)
        for i in range(len(centers)):
            found = [j for j in tree.query_ball_point(centers[i], radius) if j != i]
            if found:
                rows[i] = AdaptiveVoxelMesh._canonicalize_neighbor_order(mesh, i, centers[i], found)
        return rows

    def _vectorized(self, mesh: SimpleNamespace, tree, radius: float, batch: int) -> np.ndarray:
        centers = np.asarray(mesh.voxel_centers)
        rows = np.full((len(centers), 26), -1, dtype=np.int32)
        for start in range(0, len(centers), batch):
            end = min(start + batch, len(centers))
            lists = tree.query_ball_point(centers[start:end], radius)
            counts = np.array([len(found) for found in lists], dtype=np.int64)
            cand = np.array([j for found in lists for j in found], dtype=np.int64)
            own = np.repeat(np.arange(start, end, dtype=np.int64), counts)
            rows[start:end] = AdaptiveVoxelMesh._canonicalize_pairs(mesh, own, cand, start, end)
        return rows

    def test_matches_reference_on_lattices_with_gaps(self) -> None:
        from core.voxel_mesh import _build_kdtree

        for shape, pitch, keep, seed in (
            ((9, 8, 7), 0.15, 1.0, 0),
            ((12, 9, 6), 0.30, 0.7, 1),
            ((10, 10, 10), 0.10, 0.4, 2),
        ):
            mesh = self._lattice_with_gaps(shape, pitch, keep, seed)
            tree = _build_kdtree(mesh.voxel_centers)
            radius = np.float32(pitch) * 2.0
            reference = self._reference(mesh, tree, radius)
            for batch in (len(mesh.voxel_centers), 97):
                with self.subTest(shape=shape, keep=keep, batch=batch):
                    np.testing.assert_array_equal(self._vectorized(mesh, tree, radius, batch), reference)
            self.assertTrue(np.any(reference >= 0))

    def test_slot_table_matches_reference(self) -> None:
        from core.voxel_mesh import _DIRECTION_SLOTS

        signs = [(sx, sy, sz) for sx in (-1, 0, 1) for sy in (-1, 0, 1) for sz in (-1, 0, 1) if (sx, sy, sz) != (0, 0, 0)]
        centers = np.array([(0.0, 0.0, 0.0)] + [tuple(0.1 * s for s in sign) for sign in signs], dtype=np.float32)
        mesh = SimpleNamespace(voxel_centers=centers, base_voxel_size=0.1)
        reference = AdaptiveVoxelMesh._canonicalize_neighbor_order(
            mesh, 0, centers[0], list(range(1, len(centers)))
        )
        for index, (sx, sy, sz) in enumerate(signs, start=1):
            slot = int(_DIRECTION_SLOTS[sx + 1, sy + 1, sz + 1])
            self.assertEqual(int(reference[slot]), index)
        self.assertEqual(int(_DIRECTION_SLOTS[1, 1, 1]), -1)
        self.assertEqual(sorted(int(s) for s in _DIRECTION_SLOTS.reshape(-1) if s >= 0), list(range(26)))

if __name__ == "__main__":
    unittest.main()
