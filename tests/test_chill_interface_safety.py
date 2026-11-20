from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial import cKDTree

from core.solidification import SOLVER_MODE_SCHEDULE_LEGACY
from core.solidification_loop import _least_squares_gradients_c_per_mm
from tests.test_simulation_scenarios import (
    _cast_with_chill_and_riser_manager,
    _cast_with_chill_manager,
    _run_pipeline_with_patched_memory,
)

CAST = 0
CHILL = 2

class NiyamaGradientStencilTests(unittest.TestCase):

    def test_chill_neighbor_dominates_the_gradient(self) -> None:
        centers = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        temperatures = np.array([200.0, 1450.0, 1455.0], dtype=np.float64)
        neighbors = np.array(
            [[1, -1, -1], [0, 2, -1], [1, -1, -1]],
            dtype=np.int32,
        )
        cast_mask = np.array([False, True, True])

        gradients, _ = _least_squares_gradients_c_per_mm(
            centers_in=centers,
            temperatures=temperatures,
            neighbors=neighbors,
            target_indices=np.array([1], dtype=np.int64),
            cast_mask=cast_mask,
        )

        cast_only_gradient = 5.0 / 25.4
        self.assertGreater(float(gradients[0]), 10.0 * cast_only_gradient)

    def test_evaluation_is_still_restricted_to_cast_voxels(self) -> None:
        centers = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        temperatures = np.array([200.0, 1450.0, 1455.0], dtype=np.float64)
        neighbors = np.array(
            [[1, -1, -1], [0, 2, -1], [1, -1, -1]],
            dtype=np.int32,
        )
        cast_mask = np.array([False, True, True])

        gradients, quality = _least_squares_gradients_c_per_mm(
            centers_in=centers,
            temperatures=temperatures,
            neighbors=neighbors,
            target_indices=np.array([0], dtype=np.int64),
            cast_mask=cast_mask,
        )

        self.assertEqual(float(gradients[0]), 0.0)
        self.assertEqual(int(quality[0]), 0)

    def test_one_sided_stencil_reports_the_sampled_slope(self) -> None:
        centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        temperatures = np.array([1450.0, 200.0], dtype=np.float64)
        neighbors = np.array([[1, -1], [0, -1]], dtype=np.int32)

        gradients, quality = _least_squares_gradients_c_per_mm(
            centers_in=centers,
            temperatures=temperatures,
            neighbors=neighbors,
            target_indices=np.array([0], dtype=np.int64),
            cast_mask=np.array([True, True]),
        )

        self.assertAlmostEqual(float(gradients[0]), 1250.0 / 25.4, places=3)
        self.assertEqual(int(quality[0]), 1)

def _index_chill_bands(cls) -> None:
    part_types = np.asarray(cls.engine.mesh.voxel_part_types)
    cls.part_types = part_types

    chill_indices = np.flatnonzero(part_types == CHILL)
    assert len(chill_indices) > 0, "fixture produced no chill voxels"
    cast_indices = np.flatnonzero(part_types == CAST)

    centers = np.asarray(cls.engine.mesh.voxel_centers)
    voxel_size = float(np.max(cls.engine.mesh.voxel_sizes))
    distances, _ = cKDTree(centers[chill_indices]).query(centers[cast_indices], k=1)

    cls.chill_adjacent = cast_indices[distances <= voxel_size * 1.05]
    assert len(cls.chill_adjacent) > 0, "no cast voxels adjacent to the chill"
    cls.far_side = cast_indices[distances >= np.percentile(distances, 75)]

class ChillInterfaceRiskTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = _run_pipeline_with_patched_memory(_cast_with_chill_manager())
        _index_chill_bands(cls)

    def test_material_surface_includes_chill_backed_cast_voxels(self) -> None:
        material_surface = set(
            self.engine.mesh.get_material_surface_voxel_indices().tolist()
        )
        missing = [i for i in self.chill_adjacent.tolist() if i not in material_surface]
        self.assertEqual(missing, [], "chill-backed cast voxels missing from material surface")

    def test_material_surface_is_a_superset_of_the_geometric_surface(self) -> None:
        geometric = set(self.engine.mesh.get_surface_voxel_indices().tolist())
        material = set(self.engine.mesh.get_material_surface_voxel_indices().tolist())
        self.assertTrue(geometric.issubset(material))

    def test_risk_moduli_do_not_count_the_chill_as_wall_thickness(self) -> None:
        raw = np.asarray(self.engine.mesh.voxel_moduli, dtype=np.float32)
        risk = self.engine._get_risk_moduli()
        self.assertTrue(np.all(risk <= raw + 1e-6))
        self.assertLess(float(np.max(risk[self.chill_adjacent])), 0.3)

    def test_no_niyama_risk_at_the_chill_face(self) -> None:
        np.testing.assert_array_equal(
            np.asarray(self.engine.niyama_risk)[self.chill_adjacent], 0.0
        )
        np.testing.assert_array_equal(
            np.asarray(self.engine.combined_feed_risk)[self.chill_adjacent], 0.0
        )

    def test_no_visible_shrink_or_porosity_at_the_chill_face(self) -> None:
        shrink = np.asarray(self.engine.shrinkage_probability)[self.chill_adjacent]
        porosity = np.asarray(self.engine.porosity_probability)[self.chill_adjacent]
        self.assertLess(float(np.max(shrink)), 0.30)
        self.assertLess(float(np.max(porosity)), 0.30)

    def test_feed_paths_are_not_reported_blocked_at_the_chill_face(self) -> None:
        blocked = getattr(self.engine, "feeding_blocked", None)
        if blocked is None:
            self.skipTest("feeding_blocked not populated for this run")
        self.assertFalse(bool(np.any(np.asarray(blocked, dtype=bool)[self.chill_adjacent])))

    def test_chill_side_is_safer_than_the_far_side(self) -> None:
        shrink = np.asarray(self.engine.shrinkage_probability)
        self.assertLessEqual(
            float(np.mean(shrink[self.chill_adjacent])),
            float(np.mean(shrink[self.far_side])),
        )

    def test_display_intensity_is_clear_around_the_chill(self) -> None:
        display = np.asarray(self.engine.get_shrinkage_display_intensity())
        np.testing.assert_array_equal(display[self.part_types == CHILL], 0.0)
        self.assertLess(float(np.max(display[self.chill_adjacent])), 0.30)

class FeedingHistoryScoringTests(unittest.TestCase):

    @staticmethod
    def _line_analyzer():
        from types import SimpleNamespace

        from core.feeding_path import FeedingPathAnalyzer

        n = 4
        centers = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            dtype=np.float64,
        )
        neighbors = np.full((n, 6), -1, dtype=np.int32)
        for i in range(n - 1):
            neighbors[i, 0] = i + 1
            neighbors[i + 1, 1] = i
        mesh = SimpleNamespace(
            voxel_centers=centers,
            voxel_neighbors=neighbors,
            get_voxel_count=lambda: n,
        )
        return FeedingPathAnalyzer(mesh, alloy=None)

    def _analyze(self, history, reference_times):
        analyzer = self._line_analyzer()
        times = [0.0, 10.0, 20.0]
        return analyzer.analyze(
            np.array([0], dtype=np.int64),
            np.asarray(history[-1], dtype=np.float64),
            show_progress=False,
            history=[np.asarray(h, dtype=np.float64) for h in history],
            history_times=times,
            reference_times=np.asarray(reference_times, dtype=np.float64),
        )

    def test_fast_freeze_past_coherency_is_not_blocked(self) -> None:
        history = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.3, 0.95],
            [0.0, 0.4, 0.4, 0.99],
        ]
        result = self._analyze(history, reference_times=[20.0, 20.0, 20.0, 10.0])

        self.assertFalse(bool(result.blocked_voxels[3]))
        self.assertGreaterEqual(float(result.history_open_fraction[3]), 0.99)

    def test_genuine_isolated_pocket_is_still_blocked(self) -> None:
        history = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.97, 0.3],
            [0.0, 0.4, 0.99, 0.4],
        ]
        result = self._analyze(history, reference_times=[20.0, 20.0, 10.0, 20.0])

        self.assertTrue(bool(result.blocked_voxels[3]))
        self.assertLess(float(result.history_open_fraction[3]), 0.5)

class OrphanMetalPocketTests(unittest.TestCase):

    @staticmethod
    def _fake_engine(part_types, neighbors, centers=None, metal_part_bounds=None):
        from core.solidification import SolidificationEngine

        engine = SolidificationEngine.__new__(SolidificationEngine)
        engine.mesh = type("M", (), {})()
        engine.mesh.voxel_part_types = np.asarray(part_types, dtype=np.int8)
        engine.mesh.voxel_neighbors = np.asarray(neighbors, dtype=np.int32)
        if centers is None:
            centers = np.zeros((len(part_types), 3), dtype=np.float32)
        engine.mesh.voxel_centers = np.asarray(centers, dtype=np.float64)
        engine.mesh.voxel_sizes = np.full(len(part_types), 1.0, dtype=np.float32)
        if metal_part_bounds is not None:
            engine.mesh.metal_part_bounds = [
                np.asarray(b, dtype=np.float64) for b in metal_part_bounds
            ]
        return engine

    def test_enclosed_pocket_is_orphaned(self) -> None:
        part_types = [0, 0, 0, 2, 0]
        neighbors = [
            [1, -1, -1],
            [0, 2, -1],
            [1, 3, -1],
            [2, 4, -1],
            [3, -1, -1],
        ]
        engine = self._fake_engine(part_types, neighbors)

        orphan = engine._get_orphan_metal_mask()

        np.testing.assert_array_equal(orphan, [False, False, False, False, True])

    def test_display_intensity_zeroes_orphans(self) -> None:
        part_types = [0, 0, 0, 2, 0]
        neighbors = [
            [1, -1, -1],
            [0, 2, -1],
            [1, 3, -1],
            [2, 4, -1],
            [3, -1, -1],
        ]
        engine = self._fake_engine(part_types, neighbors)
        engine.shrinkage_probability = np.array([0.1, 0.2, 0.1, 0.0, 0.9], dtype=np.float32)
        engine.porosity_probability = np.array([0.0, 0.0, 0.0, 0.0, 0.8], dtype=np.float32)

        display = engine.get_shrinkage_display_intensity()

        self.assertEqual(float(display[4]), 0.0)
        self.assertAlmostEqual(float(display[1]), 0.2, places=5)

    def test_riser_connected_through_casting_is_kept(self) -> None:
        part_types = [0, 0, 0, 1]
        neighbors = [
            [1, -1],
            [0, 2],
            [1, 3],
            [2, -1],
        ]
        engine = self._fake_engine(part_types, neighbors)

        orphan = engine._get_orphan_metal_mask()

        self.assertFalse(bool(np.any(orphan)))

    def test_outside_metal_boxes_is_flagged_even_when_connected(self) -> None:
        part_types = [0, 0, 0, 0, 0]
        neighbors = [
            [1, -1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, -1],
        ]
        centers = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [10, 0, 0]]
        bounds = [[[-0.5, -0.5, -0.5], [3.5, 0.5, 0.5]]]
        engine = self._fake_engine(part_types, neighbors, centers=centers,
                                   metal_part_bounds=bounds)

        orphan = engine._get_orphan_metal_mask()

        np.testing.assert_array_equal(orphan, [False, False, False, False, True])

    def test_chill_smothered_voxel_is_flagged_even_when_connected(self) -> None:
        part_types = [0, 0] + [2] * 8
        nbr_row_1 = [0] + list(range(2, 10))
        neighbors = [
            [1] + [-1] * 8,
            nbr_row_1,
        ] + [[1] + [-1] * 8 for _ in range(8)]
        engine = self._fake_engine(part_types, neighbors)

        orphan = engine._get_orphan_metal_mask()

        self.assertFalse(bool(orphan[0]))
        self.assertTrue(bool(orphan[1]))

    def test_flat_chill_face_voxel_is_not_smothered(self) -> None:
        part_types = [0] + [0] * 17 + [2] * 9
        neighbors = [list(range(1, 27))] + [[0] + [-1] * 25 for _ in range(26)]
        engine = self._fake_engine(part_types, neighbors)

        orphan = engine._get_orphan_metal_mask()

        self.assertFalse(bool(orphan[0]))

    def test_fraction_liquid_view_hides_orphans_and_chills(self) -> None:
        part_types = [0, 0, 0, 2, 0]
        neighbors = [
            [1, -1, -1],
            [0, 2, -1],
            [1, 3, -1],
            [2, 4, -1],
            [3, -1, -1],
        ]
        engine = self._fake_engine(part_types, neighbors)
        raw = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        masked = engine._mask_non_cast_fraction_liquid(raw)

        self.assertEqual(float(masked[3]), 0.0)
        self.assertEqual(float(masked[4]), 0.0)
        self.assertEqual(float(masked[0]), 1.0)

class ChillInterfaceWithRiserTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = _run_pipeline_with_patched_memory(_cast_with_chill_and_riser_manager())
        _index_chill_bands(cls)

    def test_trace_is_connected_to_the_casting(self) -> None:
        result = self.engine.feeding_path_result
        self.assertIsNotNone(result)
        reached_cast = (result.path_length >= 0.0) & (self.part_types == CAST)
        cast_total = int(np.sum(self.part_types == CAST))
        self.assertGreater(int(np.sum(reached_cast)), cast_total // 2)

    def test_chill_face_is_not_reported_feed_blocked(self) -> None:
        blocked = np.asarray(self.engine.feeding_blocked, dtype=bool)
        self.assertFalse(bool(np.any(blocked[self.chill_adjacent])))

    def test_chill_face_open_fraction_is_full(self) -> None:
        open_fraction = np.asarray(self.engine.feeding_open_fraction)
        self.assertGreaterEqual(float(np.min(open_fraction[self.chill_adjacent])), 0.99)

    def test_no_visible_risk_at_the_chill_face(self) -> None:
        shrink = np.asarray(self.engine.shrinkage_probability)[self.chill_adjacent]
        porosity = np.asarray(self.engine.porosity_probability)[self.chill_adjacent]
        self.assertLess(float(np.max(shrink)), 0.30)
        self.assertLess(float(np.max(porosity)), 0.30)

    def test_blocked_floor_does_not_fire_at_the_chill_face(self) -> None:
        display = np.asarray(self.engine.get_shrinkage_display_intensity())
        self.assertLess(float(np.max(display[self.chill_adjacent])), 0.35)

class ChillInterfaceRiskLegacySolverTests(ChillInterfaceRiskTests):

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = _run_pipeline_with_patched_memory(
            _cast_with_chill_manager(),
            extra_analysis_options={"solver_mode": SOLVER_MODE_SCHEDULE_LEGACY},
        )
        _index_chill_bands(cls)

if __name__ == "__main__":
    unittest.main()
