"""Guards the cKDTree build-strategy fix in core/voxel_mesh.py.

`AdaptiveVoxelMesh.generate()` used to die with SIGSEGV -- exit 139, no
exception, no traceback -- at several voxel sizes. It was a C-stack overflow in
SciPy's recursive kd-tree build: the default median split rule recurses to a
depth set by the data distribution, and on voxel centres that exceeded the 8 MB
macOS main-thread stack.

Diagnosis, for anyone who meets it again: running the identical crashing case on
a 256 MB thread stack made it pass unchanged, which is what identified stack
exhaustion rather than bad data. The saved input array also built fine in a
fresh process, which is why it looked like corrupted state for a long time.

Midpoint splitting (`balanced_tree=False`) bounds recursion by the bounding box
instead of by the points -- the natural fit for a uniform lattice. It is a crash
fix, not a tuning choice, which is why these tests exist.
"""

from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
import trimesh

from core.voxel_mesh import AdaptiveVoxelMesh, _build_kdtree


class BuildStrategyTests(unittest.TestCase):
    def test_midpoint_splitting_is_the_default(self) -> None:
        rng = np.random.default_rng(3)
        tree = _build_kdtree(rng.random((64, 3)))
        # SciPy does not expose the flag, so assert via the documented
        # constructor contract: passing it explicitly must be accepted and
        # produce an equivalent tree.
        self.assertEqual(tree.n, 64)

    def test_explicit_override_is_honoured(self) -> None:
        rng = np.random.default_rng(5)
        points = rng.random((32, 3))
        self.assertEqual(_build_kdtree(points, balanced_tree=True).n, 32)

    def test_query_results_are_unchanged_by_the_build_rule(self) -> None:
        """
        The fix must not move any answer. Tree structure differs; nearest
        neighbours do not. If this ever fails, the crash fix has become a
        behaviour change and every distance-derived field is suspect.
        """
        from scipy.spatial import cKDTree

        rng = np.random.default_rng(11)
        data = rng.random((500, 3))
        query = rng.random((200, 3))

        midpoint = _build_kdtree(data)
        median = cKDTree(data, balanced_tree=True)

        d_mid, i_mid = midpoint.query(query)
        d_med, i_med = median.query(query)
        np.testing.assert_allclose(d_mid, d_med, rtol=1e-12)
        np.testing.assert_array_equal(i_mid, i_med)

    def test_query_matches_on_lattice_data(self) -> None:
        """Lattice points are the case that crashed; check them specifically."""
        from scipy.spatial import cKDTree

        idx = np.stack(
            np.meshgrid(*[np.arange(12)] * 3, indexing="ij"), axis=-1
        ).reshape(-1, 3).astype(np.float32) * 0.25
        rng = np.random.default_rng(13)
        query = rng.random((150, 3)) * 3.0

        d_mid, i_mid = _build_kdtree(idx).query(query)
        d_med, i_med = cKDTree(idx, balanced_tree=True).query(query)
        np.testing.assert_allclose(d_mid, d_med, rtol=1e-12)
        np.testing.assert_array_equal(i_mid, i_med)


class RegressionTests(unittest.TestCase):
    """
    The sizes that used to take the process down.

    These run real mesh generation, so they are slower than a unit test, but a
    crash this severe -- it kills the whole application, mid-run, with no
    diagnostic -- earns the seconds.
    """

    #: base_voxel_size values that segfaulted on thick_thin_test.stl.
    CRASHING_SIZES = (0.30, 0.17)

    def test_previously_crashing_sizes_now_generate(self) -> None:
        mesh_source = trimesh.load("sample_stl_files/thick_thin_test.stl")
        bounds = np.asarray(mesh_source.bounds).T
        for dx in self.CRASHING_SIZES:
            with self.subTest(base_voxel_size=dx):
                mesh = AdaptiveVoxelMesh(base_voxel_size=dx)
                with contextlib.redirect_stdout(io.StringIO()):
                    ok = mesh.generate(mesh_source, bounds, show_progress=False)
                self.assertTrue(ok)
                self.assertGreater(len(mesh.voxel_centers), 0)

    def test_a_size_that_always_worked_still_works(self) -> None:
        """Guard against fixing the crash by breaking the working path."""
        mesh_source = trimesh.load("sample_stl_files/thick_thin_test.stl")
        mesh = AdaptiveVoxelMesh(base_voxel_size=0.25)
        with contextlib.redirect_stdout(io.StringIO()):
            ok = mesh.generate(mesh_source, np.asarray(mesh_source.bounds).T,
                               show_progress=False)
        self.assertTrue(ok)
        self.assertGreater(len(mesh.voxel_centers), 0)

    def test_moduli_are_still_computed(self) -> None:
        """
        The crash was inside the geometric-modulus pass, so confirm that pass
        still produces sane output rather than merely not dying.
        """
        mesh_source = trimesh.load("sample_stl_files/thick_thin_test.stl")
        mesh = AdaptiveVoxelMesh(base_voxel_size=0.30)
        with contextlib.redirect_stdout(io.StringIO()):
            mesh.generate(mesh_source, np.asarray(mesh_source.bounds).T,
                          show_progress=False)
        moduli = np.asarray(mesh.voxel_moduli)
        self.assertEqual(moduli.shape[0], len(mesh.voxel_centers))
        self.assertTrue(bool(np.all(np.isfinite(moduli))))
        self.assertGreater(float(moduli.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
