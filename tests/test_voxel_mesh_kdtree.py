from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from tests.helpers.ci_models import ci_model
from core.voxel_mesh import (
    AdaptiveVoxelMesh,
    _build_kdtree,
    _MidpointKDTree,
    _robust_contains_check_impl,
    _use_midpoint_kdtree_in_trimesh_proximity,
)

class BuildStrategyTests(unittest.TestCase):
    def test_midpoint_splitting_is_the_default(self) -> None:
        rng = np.random.default_rng(3)
        tree = _build_kdtree(rng.random((64, 3)))
        self.assertEqual(tree.n, 64)

    def test_explicit_override_is_honoured(self) -> None:
        rng = np.random.default_rng(5)
        points = rng.random((32, 3))
        self.assertEqual(_build_kdtree(points, balanced_tree=True).n, 32)

    def test_query_results_are_unchanged_by_the_build_rule(self) -> None:
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

    CRASHING_SIZES = (0.30, 0.17)

    def test_previously_crashing_sizes_now_generate(self) -> None:
        mesh_source = trimesh.load(ci_model("thick_thin_test.stl"))
        bounds = np.asarray(mesh_source.bounds).T
        for dx in self.CRASHING_SIZES:
            with self.subTest(base_voxel_size=dx):
                mesh = AdaptiveVoxelMesh(base_voxel_size=dx)
                with contextlib.redirect_stdout(io.StringIO()):
                    ok = mesh.generate(mesh_source, bounds, show_progress=False)
                self.assertTrue(ok)
                self.assertGreater(len(mesh.voxel_centers), 0)

    def test_a_size_that_always_worked_still_works(self) -> None:
        mesh_source = trimesh.load(ci_model("thick_thin_test.stl"))
        mesh = AdaptiveVoxelMesh(base_voxel_size=0.25)
        with contextlib.redirect_stdout(io.StringIO()):
            ok = mesh.generate(mesh_source, np.asarray(mesh_source.bounds).T,
                               show_progress=False)
        self.assertTrue(ok)
        self.assertGreater(len(mesh.voxel_centers), 0)

    def test_moduli_are_still_computed(self) -> None:
        mesh_source = trimesh.load(ci_model("thick_thin_test.stl"))
        mesh = AdaptiveVoxelMesh(base_voxel_size=0.30)
        with contextlib.redirect_stdout(io.StringIO()):
            mesh.generate(mesh_source, np.asarray(mesh_source.bounds).T,
                          show_progress=False)
        moduli = np.asarray(mesh.voxel_moduli)
        self.assertEqual(moduli.shape[0], len(mesh.voxel_centers))
        self.assertTrue(bool(np.all(np.isfinite(moduli))))
        self.assertGreater(float(moduli.max()), 0.0)

def _tree_shape(tree):
    shape = []
    stack = [(tree.tree, 0)]
    while stack:
        node, depth = stack.pop()
        shape.append((depth, getattr(node, "split_dim", -1),
                      round(float(getattr(node, "split", 0.0)), 12)))
        for child in ("greater", "lesser"):
            found = getattr(node, child, None)
            if found is not None:
                stack.append((found, depth + 1))
    return shape

def _triangle_soup_box(subdivisions: int = 3) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=(2.0, 1.0, 3.0))
    for _ in range(subdivisions):
        box = box.subdivide()
    exploded = np.asarray(box.vertices, dtype=np.float64)[box.faces].reshape(-1, 3)
    return trimesh.Trimesh(
        vertices=exploded,
        faces=np.arange(len(exploded), dtype=np.int64).reshape(-1, 3),
        process=False,
    )

class TrimeshProximityTests(unittest.TestCase):

    def setUp(self) -> None:
        self.mesh = _triangle_soup_box()
        lo, hi = self.mesh.bounds
        rng = np.random.default_rng(17)
        self.points = rng.uniform(lo - 0.2, hi + 0.2, size=(200, 3))

    def test_the_fixture_really_is_duplicate_heavy(self) -> None:
        vertices = np.asarray(self.mesh.vertices)
        unique = len(np.unique(vertices, axis=0))
        self.assertLess(unique * 2, len(vertices))

    def test_the_patch_is_installed_by_importing_the_module(self) -> None:
        import trimesh.proximity

        self.assertIs(trimesh.proximity.cKDTree, _MidpointKDTree)

    def test_reapplying_the_patch_is_a_no_op(self) -> None:
        self.assertTrue(_use_midpoint_kdtree_in_trimesh_proximity())
        self.assertTrue(_use_midpoint_kdtree_in_trimesh_proximity())

    @staticmethod
    def _capture_proximity_trees(run):
        import trimesh.proximity

        captured = []
        installed = trimesh.proximity.cKDTree

        class _Spy(installed):
            def __init__(self, data, *args, **kwargs):
                captured.append(np.array(data, dtype=np.float64, copy=True))
                super().__init__(data, *args, **kwargs)

        trimesh.proximity.cKDTree = _Spy
        try:
            result = run()
        finally:
            trimesh.proximity.cKDTree = installed
        return result, captured, installed

    def test_the_primary_containment_path_builds_no_proximity_tree(self) -> None:
        inside, captured, _ = self._capture_proximity_trees(
            lambda: _robust_contains_check_impl(self.mesh, self.points, 0.05)
        )
        self.assertEqual(inside.shape, (len(self.points),))
        self.assertEqual(captured, [], "primary containment path built a trimesh.proximity kd-tree")

    def test_the_containment_fallback_builds_midpoint_trees(self) -> None:
        from unittest.mock import patch

        with patch.object(type(self.mesh), "contains", side_effect=RuntimeError("forced")):
            inside, captured, installed = self._capture_proximity_trees(
                lambda: _robust_contains_check_impl(self.mesh, self.points, 0.05)
            )

        self.assertTrue(
            captured,
            "the fallback built no kd-tree, so this test is watching the wrong "
            "code -- check that the signed_distance fallback still runs when "
            "mesh.contains raises",
        )
        self.assertEqual(inside.shape, (len(self.points),))

        disagreed = False
        for data in captured:
            midpoint = _tree_shape(cKDTree(data, balanced_tree=False))
            median = _tree_shape(cKDTree(data, balanced_tree=True))
            self.assertEqual(_tree_shape(installed(data)), midpoint)
            disagreed = disagreed or midpoint != median
        self.assertTrue(
            disagreed,
            "the two split rules produced identical trees on this fixture, so "
            "the assertion above proves nothing -- change the fixture",
        )

    def test_the_query_still_answers_correctly(self) -> None:
        lo, hi = self.mesh.bounds
        centre = (lo + hi) / 2.0
        far = hi + 5.0
        probes = np.vstack([centre, far])
        inside = _robust_contains_check_impl(self.mesh, probes, 0.05)
        self.assertTrue(bool(inside[0]))
        self.assertFalse(bool(inside[1]))

if __name__ == "__main__":
    unittest.main()
