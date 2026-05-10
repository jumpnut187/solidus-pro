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
