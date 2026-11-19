from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from core.voxel_mesh import AdaptiveVoxelMesh


def _grid_centers() -> np.ndarray:
    axis = np.linspace(-0.9, 0.9, 19, dtype=np.float32)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(np.float32)


def _parts_data() -> list[tuple[trimesh.Trimesh, int]]:
    cast = trimesh.creation.box(extents=[2.2, 2.2, 2.2])

    riser = trimesh.creation.box(extents=[1.2, 1.0, 1.0])
    riser.apply_translation([0.45, 0.0, 0.0])

    chill = trimesh.creation.box(extents=[0.8, 0.8, 0.8])
    chill.apply_translation([0.0, 0.5, 0.0])

    post = trimesh.creation.box(extents=[0.6, 0.6, 0.6])
    post.apply_translation([0.0, 0.0, 0.45])

    return [(cast, 0), (riser, 1), (chill, 2), (post, 3)]


def _run_assignment(mode: str, workers: int = 2) -> np.ndarray:
    mesh = AdaptiveVoxelMesh(
        base_voxel_size=0.1,
        part_assignment_mode=mode,
        part_assignment_workers=workers,
        part_assignment_batch_size=64,
    )
    mesh.voxel_centers = _grid_centers()
    mesh.assign_part_types(_parts_data())
    assert mesh.voxel_part_types is not None
    return np.asarray(mesh.voxel_part_types).copy()


class PartAssignmentParallelTests(unittest.TestCase):
    def test_single_cast_uses_fast_path(self) -> None:
        mesh = AdaptiveVoxelMesh(base_voxel_size=0.1)
        mesh.voxel_centers = _grid_centers()
        cast = trimesh.creation.box(extents=[2.2, 2.2, 2.2])

        with patch.object(mesh, "_robust_contains_check", side_effect=AssertionError("fast path skipped")):
            mesh.assign_part_types([(cast, 0)])

        assert mesh.voxel_part_types is not None
        self.assertTrue(np.all(np.asarray(mesh.voxel_part_types) == 0))

    def test_auto_mode_does_not_parallelize_small_candidate_sets(self) -> None:
        mesh = AdaptiveVoxelMesh(
            base_voxel_size=0.1,
            part_assignment_mode="auto",
            part_assignment_workers=2,
            part_assignment_batch_size=2_500,
        )

        self.assertFalse(mesh._should_parallelize_part_assignment(5_000))

    def test_auto_mode_worker_count_is_memory_bounded(self) -> None:
        import unittest.mock as mock
        import core.voxel_mesh as voxel_mesh_module

        mesh = AdaptiveVoxelMesh(
            base_voxel_size=0.1,
            part_assignment_mode="auto",
            part_assignment_workers=0,
            part_assignment_batch_size=2_500,
        )
        per_worker = AdaptiveVoxelMesh.AUTO_PART_ASSIGNMENT_MEMORY_PER_WORKER_BYTES

        # Sizing must read live RAM, not the import-time hardware snapshot.
        with mock.patch.object(voxel_mesh_module, "_available_memory_bytes", return_value=per_worker * 2), \
             mock.patch.object(voxel_mesh_module.hardware, "get_optimal_thread_count", return_value=8):
            self.assertEqual(mesh._get_part_assignment_workers(), 2)

        with mock.patch.object(voxel_mesh_module, "_available_memory_bytes", return_value=per_worker * 16), \
             mock.patch.object(voxel_mesh_module.hardware, "get_optimal_thread_count", return_value=8):
            self.assertEqual(mesh._get_part_assignment_workers(), 8)

    def test_batch_size_shrinks_with_face_count(self) -> None:
        mesh = AdaptiveVoxelMesh(base_voxel_size=0.1, part_assignment_batch_size=2_500)
        ref = AdaptiveVoxelMesh.PART_ASSIGNMENT_BATCH_FACE_REFERENCE

        self.assertEqual(mesh._effective_batch_size(ref), 2_500)
        self.assertEqual(mesh._effective_batch_size(ref * 2), 1_250)
        self.assertEqual(mesh._effective_batch_size(500_000), AdaptiveVoxelMesh.PART_ASSIGNMENT_MIN_BATCH)

    def test_concurrency_is_bounded_by_measured_cost_and_reserve(self) -> None:
        import unittest.mock as mock
        import core.voxel_mesh as voxel_mesh_module

        mesh = AdaptiveVoxelMesh(base_voxel_size=0.1)
        gb = 1024 ** 3
        with mock.patch.object(mesh, "_memory_reserve_bytes", return_value=4 * gb):
            with mock.patch.object(voxel_mesh_module, "_available_memory_bytes", return_value=16 * gb):
                # (16 - 4) / 6 = 2 workers at a 6 GB measured cost
                self.assertEqual(mesh._memory_bounded_concurrency(6 * gb, ceiling=17), 2)
                # cheap batches are capped by the pool ceiling
                self.assertEqual(mesh._memory_bounded_concurrency(100 * 2**20, ceiling=17), 17)
            with mock.patch.object(voxel_mesh_module, "_available_memory_bytes", return_value=3 * gb):
                # inside the reserve: nothing may run in workers
                self.assertEqual(mesh._memory_bounded_concurrency(100 * 2**20, ceiling=17), 0)

    def test_parallel_falls_back_to_sequential_when_headroom_is_gone(self) -> None:
        expected_types = _run_assignment("legacy", workers=1)

        with patch("core.voxel_mesh._available_memory_bytes", return_value=0), \
             patch("core.voxel_mesh.ProcessPoolExecutor", side_effect=AssertionError("pool must not be created")):
            # No pool is ever built: _get_part_assignment_workers() sees no RAM and
            # returns 1, so _should_parallelize_part_assignment() is False.
            types = _run_assignment("parallel", workers=0)

        np.testing.assert_array_equal(types, expected_types)

    def test_memory_guard_degrades_mid_part_to_sequential(self) -> None:
        import core.voxel_mesh as voxel_mesh_module
        expected_types = _run_assignment("legacy", workers=1)

        # Plenty of RAM for sizing, then the guard sees the reserve breached on
        # every submission check: the first batch is probed in-process, the
        # pool is created, and the rest must finish sequentially with the same
        # answer.
        readings = iter([64 * 1024 ** 3, 64 * 1024 ** 3])

        def _fake_available() -> int:
            return next(readings, 0)

        with patch.object(voxel_mesh_module, "_available_memory_bytes", side_effect=_fake_available):
            types = _run_assignment("parallel", workers=2)

        np.testing.assert_array_equal(types, expected_types)

    def test_pool_is_shared_across_parts_and_torn_down(self) -> None:
        import core.voxel_mesh as voxel_mesh_module
        real_executor = voxel_mesh_module.ProcessPoolExecutor
        created = []

        def _counting_executor(*args, **kwargs):
            pool = real_executor(*args, **kwargs)
            created.append(pool)
            return pool

        mesh = AdaptiveVoxelMesh(
            base_voxel_size=0.1,
            part_assignment_mode="parallel",
            part_assignment_workers=2,
            part_assignment_batch_size=64,
        )
        mesh.voxel_centers = _grid_centers()
        with patch.object(voxel_mesh_module, "ProcessPoolExecutor", side_effect=_counting_executor):
            mesh.assign_part_types(_parts_data())

        # Three non-CAST parts, one pool.
        self.assertEqual(len(created), 1)
        self.assertIsNone(mesh._part_assignment_pool)
        self.assertIsNone(mesh._part_assignment_scratch_dir)

    def test_auto_mode_parallelizes_medium_candidate_sets(self) -> None:
        mesh = AdaptiveVoxelMesh(
            base_voxel_size=0.1,
            part_assignment_mode="auto",
            part_assignment_workers=2,
            part_assignment_batch_size=2_500,
        )

        self.assertTrue(mesh._should_parallelize_part_assignment(25_000))

    def test_explicit_parallel_mode_still_parallelizes(self) -> None:
        mesh = AdaptiveVoxelMesh(
            base_voxel_size=0.1,
            part_assignment_mode="parallel",
            part_assignment_workers=2,
            part_assignment_batch_size=2_500,
        )

        self.assertTrue(mesh._should_parallelize_part_assignment(25_000))

    def test_parallel_mode_matches_legacy_output(self) -> None:
        legacy_types = _run_assignment("legacy", workers=1)
        parallel_types = _run_assignment("parallel", workers=2)

        np.testing.assert_array_equal(parallel_types, legacy_types)

    def test_parallel_failures_fallback_to_legacy(self) -> None:
        expected_types = _run_assignment("legacy", workers=1)

        with patch("core.voxel_mesh.ProcessPoolExecutor", side_effect=RuntimeError("pool unavailable")):
            fallback_types = _run_assignment("parallel", workers=2)

        np.testing.assert_array_equal(fallback_types, expected_types)


if __name__ == "__main__":
    unittest.main()
