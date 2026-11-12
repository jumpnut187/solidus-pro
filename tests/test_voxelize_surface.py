"""``voxel_mesh.voxelize_surface``: trimesh's ``voxelized(pitch)`` with the faces subdivided in batches.

The batched form exists to bound the mesh phase's transient (trimesh subdivides the whole mesh at once and holds
every subdivided vertex). It must give the same cells: the same surface matrix, transform and filled matrix as
``mesh.voxelized(pitch)``, at the default batch size and at batches small enough to split every face apart. If a
trimesh upgrade changes how it subdivides, this test is what says so.
"""

from __future__ import annotations

import copy
import glob
import os
import unittest

import numpy as np
import trimesh

from tests.helpers.ci_models import CI_MODELS_DIR
from core.voxel_mesh import voxelize_surface

PITCHES = (0.3, 0.2, 0.1, 0.05)


def _same(reference, other) -> bool:
    if not np.array_equal(np.asarray(reference.matrix), np.asarray(other.matrix)):
        return False
    if not np.array_equal(reference.transform, other.transform):
        return False
    # fill() works in place: compare filled copies.
    return np.array_equal(
        np.asarray(copy.deepcopy(reference).fill().matrix), np.asarray(copy.deepcopy(other).fill().matrix)
    )


class VoxelizeSurfaceIdentityTests(unittest.TestCase):
    def test_the_ci_models_voxelize_to_the_same_cells(self) -> None:
        paths = sorted(glob.glob(os.path.join(CI_MODELS_DIR, "*.stl")))
        self.assertGreaterEqual(len(paths), 7)
        for path in paths:
            mesh = trimesh.load(path)
            for pitch in PITCHES:
                with self.subTest(model=os.path.basename(path), pitch=pitch):
                    reference = mesh.voxelized(pitch=pitch)
                    self.assertTrue(_same(reference, voxelize_surface(mesh, pitch)))
                    # One face (or a few) per batch: every shared edge is split in two batches.
                    self.assertTrue(_same(reference, voxelize_surface(mesh, pitch, batch_subfaces=1000)))

    def test_an_unreferenced_vertex_still_marks_its_cell(self) -> None:
        """trimesh rounds every vertex, referenced by a face or not; so does the batched form."""
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        mesh = trimesh.Trimesh(vertices=np.vstack([mesh.vertices, [[3.0, 0.0, 0.0]]]), faces=mesh.faces, process=False)
        reference = mesh.voxelized(pitch=0.2)
        self.assertTrue(_same(reference, voxelize_surface(mesh, 0.2)))

    def test_a_mesh_of_two_bodies_far_apart(self) -> None:
        a = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        b = trimesh.creation.box(extents=(0.5, 2.0, 0.3))
        b.apply_translation((4.0, -1.0, 2.0))
        mesh = trimesh.util.concatenate([a, b])
        for pitch in (0.15, 0.07):
            self.assertTrue(_same(mesh.voxelized(pitch=pitch), voxelize_surface(mesh, pitch, batch_subfaces=500)))


if __name__ == "__main__":
    unittest.main()
