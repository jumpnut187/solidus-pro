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

        with mock.patch.object(voxel_mesh_module.hardware, "available_memory", per_worker * 2), \
             mock.patch.object(voxel_mesh_module.hardware, "get_optimal_thread_count", return_value=8):
            self.assertEqual(mesh._get_part_assignment_workers(), 2)

        with mock.patch.object(voxel_mesh_module.hardware, "available_memory", per_worker * 16), \
             mock.patch.object(voxel_mesh_module.hardware, "get_optimal_thread_count", return_value=8):
            self.assertEqual(mesh._get_part_assignment_workers(), 8)

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
