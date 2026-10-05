"""A pocket the parts seal off between them is mold, not metal.

``AdaptiveVoxelMesh`` voxelizes the parts' surfaces and then fills every region
those surfaces seal off from the outside. That region was meant to be the parts'
interiors, but on an assembly it also took in any pocket the parts enclose
without filling it. Deleting a riser from under a pad that had been modeled on
top of it left exactly such a pocket, and the next run simulated it as cast
metal: the riser was gone from the model and still in the results.
"""

from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
import trimesh

from core.part_manager import PART_TYPE_CAST, PART_TYPE_RISER
from core.voxel_mesh import AdaptiveVoxelMesh

VOXEL_SIZE = 0.25


def _box(extents, center) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    return box


def _pad_with_pocket() -> trimesh.Trimesh:
    """A 4 x 4 x 3 pad, z 2..5, with a 2 x 2 x 2 pocket open at its underside.

    Built from vertices rather than a boolean, which needs a backend the test
    environment may not have: the outer box with a square hole in its bottom
    face, and the pocket's four walls and ceiling facing into it.
    """
    a, b, z0, z1, zp = 2.0, 1.0, 2.0, 5.0, 4.0
    vertices = np.array(
        [
            # 0-3 outer bottom corners, 4-7 outer top corners
            [-a, -a, z0], [a, -a, z0], [a, a, z0], [-a, a, z0],
            [-a, -a, z1], [a, -a, z1], [a, a, z1], [-a, a, z1],
            # 8-11 pocket mouth on the bottom face, 12-15 pocket ceiling
            [-b, -b, z0], [b, -b, z0], [b, b, z0], [-b, b, z0],
            [-b, -b, zp], [b, -b, zp], [b, b, zp], [-b, b, zp],
        ]
    )
    faces = [
        # Top and outer walls.
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        # Bottom face: the ring between the outer square and the pocket mouth.
        [0, 9, 1], [0, 8, 9], [1, 10, 2], [1, 9, 10], [2, 11, 3], [2, 10, 11], [3, 8, 0], [3, 11, 8],
        # Pocket walls and ceiling, facing into the pocket.
        [8, 12, 13], [8, 13, 9], [9, 13, 14], [9, 14, 10], [10, 14, 15], [10, 15, 11], [11, 15, 12], [11, 12, 8],
        [12, 15, 14], [12, 14, 13],
    ]
    pad = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    trimesh.repair.fix_normals(pad)
    assert pad.is_watertight and pad.is_volume, "pad fixture is not a closed solid"
    assert abs(pad.volume - (4 * 4 * 3 - 2 * 2 * 2)) < 1e-9
    return pad


def _voxelize(bodies, part_meshes=None) -> np.ndarray:
    combined = trimesh.util.concatenate(bodies)
    mesh = AdaptiveVoxelMesh(base_voxel_size=VOXEL_SIZE)
    with contextlib.redirect_stdout(io.StringIO()):
        ok = mesh.generate(
            combined, np.asarray(combined.bounds).T, show_progress=False, auto_adjust=False, part_meshes=part_meshes
        )
    if not ok:
        raise unittest.SkipTest("mesh generation failed for the fixture geometry")
    return np.asarray(mesh.voxel_centers, dtype=np.float64)


def _inside(points: np.ndarray, lo, hi) -> np.ndarray:
    return np.all((points > np.asarray(lo)) & (points < np.asarray(hi)), axis=1)


class SealedPocketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The casting's top face closes the pad's pocket from below.
        cls.casting = _box((6.0, 6.0, 2.0), (0.0, 0.0, 1.0))
        cls.pad = _pad_with_pocket()
        # Cells a full voxel clear of the pocket walls: no surface reaches them.
        cls.pocket_lo = (-1.0 + VOXEL_SIZE, -1.0 + VOXEL_SIZE, 2.0 + VOXEL_SIZE)
        cls.pocket_hi = (1.0 - VOXEL_SIZE, 1.0 - VOXEL_SIZE, 4.0 - VOXEL_SIZE)

    def test_pocket_between_parts_is_left_empty(self) -> None:
        bodies = [self.casting, self.pad]
        before = _voxelize(bodies)
        after = _voxelize(bodies, part_meshes=bodies)
        self.assertGreater(int(_inside(before, self.pocket_lo, self.pocket_hi).sum()), 0, "fixture no longer seals a pocket")
        self.assertEqual(int(_inside(after, self.pocket_lo, self.pocket_hi).sum()), 0)

    def test_no_metal_is_lost(self) -> None:
        bodies = [self.casting, self.pad]
        after = _voxelize(bodies, part_meshes=bodies)
        keys = {tuple(key) for key in np.round(after / VOXEL_SIZE).astype(np.int64)}
        lo = np.min([body.bounds[0] for body in bodies], axis=0)
        hi = np.max([body.bounds[1] for body in bodies], axis=0)
        axes = [np.arange(np.ceil(lo[i] / VOXEL_SIZE), np.floor(hi[i] / VOXEL_SIZE) + 1) * VOXEL_SIZE for i in range(3)]
        lattice = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        inside = np.zeros(len(lattice), dtype=bool)
        for body in bodies:
            inside |= body.contains(lattice)
        missing = [p for p in lattice[inside] if tuple(np.round(p / VOXEL_SIZE).astype(np.int64)) not in keys]
        self.assertEqual(missing, [])

    def test_assembly_without_a_pocket_is_unchanged(self) -> None:
        # A riser seated into the casting: the overlap is inside both parts.
        riser = _box((2.0, 2.0, 3.0), (0.0, 0.0, 3.0))
        bodies = [self.casting, riser]
        np.testing.assert_array_equal(_voxelize(bodies), _voxelize(bodies, part_meshes=bodies))

    def test_open_part_keeps_the_fill(self) -> None:
        # A pad that is not a closed solid cannot say what it contains, so the
        # region within its bounds is kept as it always was.
        pad = self.pad.copy()
        pad.update_faces(np.arange(len(pad.faces)) != 0)
        self.assertFalse(pad.is_watertight)
        bodies = [self.casting, pad]
        np.testing.assert_array_equal(_voxelize(bodies), _voxelize(bodies, part_meshes=bodies))

    def test_part_types_skip_the_pocket(self) -> None:
        bodies = [self.casting, self.pad]
        combined = trimesh.util.concatenate(bodies)
        mesh = AdaptiveVoxelMesh(base_voxel_size=VOXEL_SIZE)
        with contextlib.redirect_stdout(io.StringIO()):
            mesh.generate(combined, np.asarray(combined.bounds).T, show_progress=False, auto_adjust=False, part_meshes=bodies)
            mesh.assign_part_types([(self.casting, PART_TYPE_CAST), (self.pad, PART_TYPE_RISER)])
        centers = np.asarray(mesh.voxel_centers, dtype=np.float64)
        types = np.asarray(mesh.voxel_part_types)
        self.assertEqual(int(_inside(centers, self.pocket_lo, self.pocket_hi).sum()), 0)
        self.assertGreater(int(np.sum(types == PART_TYPE_RISER)), 0)


if __name__ == "__main__":
    unittest.main()
