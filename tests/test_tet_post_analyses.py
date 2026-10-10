"""Post-analyses on tet nodes (wave 3, docs/tet_migration/wave3_post.md).

The casting + riser + chill assembly of ``tests/test_tet_end_to_end.py`` runs
with every post-analysis enabled on a tet mesh (edge 0.2 in) and on a voxel
mesh (0.1 in). The tet results must agree qualitatively with the voxel ones.
fTetWild is not bit-deterministic, so every comparison uses a tolerance.

Small array-level tests pin the tet branches that need no mesher: weighted
isolation, edge-length graph distance, the orphan "smothered" degree rule,
pocket connectivity, riser design adapter and incremental front buffers.
"""

from __future__ import annotations

import sys
import time
import types
import unittest
from unittest import mock

import numpy as np

from config.settings import Settings
from core import simulation_pipeline as sp
from core import tet_thermal
from core.acceleration import (
    compute_isolation_score_parallel,
    compute_isolation_score_volume_weighted,
)
from core.defect_regions import extract_pocket_at
from core.incremental_solidification import expand_front
from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_RISER,
    is_metal,
)
from core.riser_design import RiserDesigner, RiserDesignParameters, RiserRecommendation
from core.solidification import SolidificationEngine, _tet_hop_length_in
from tests.test_tet_end_to_end import _assembly

TET_EDGE_IN = 0.2
VOXEL_SIZE_IN = 0.1
RISER_RADIUS_IN = 0.6
RISER_BASE_Z_IN = 1.0  # bottom of the riser cylinder = top of the casting

ALL_ANALYSES = {
    "shrinkage": True,
    "hotspot": True,
    "niyama": True,
    "hot_tear": True,
    "sand_burn_in": True,
    "fraction_liquid": True,
    "feeding_path": True,
    "stress_analysis": True,
    "microstructure": True,
    "gas_porosity": True,
}


def _run_all_analyses(mesh_type: str, size_in: float):
    inputs = sp.PipelineInputs("WCB", size_in, 0, mesh_type=mesh_type, analysis_options=dict(ALL_ANALYSES))
    settings = Settings()
    settings.mesh.tet_edge_length_factor = 1.0  # size_in is the tet edge length (TET_EDGE_IN)
    started = time.perf_counter()
    with mock.patch.object(sp, "get_settings", return_value=settings), mock.patch.object(
        tet_thermal, "get_settings", return_value=settings
    ):
        result = sp.run_pipeline(inputs, part_manager=_assembly())
    print(f"\n  {mesh_type} all analyses: {time.perf_counter() - started:.1f} s", file=sys.stderr)
    return result


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(values[order][np.searchsorted(cumulative, q / 100.0)])


def _cast_peak_centroid(result, values) -> np.ndarray:
    """Value-weighted centroid (in) of CAST points at >= 50 % of the CAST maximum."""
    mesh = result.voxel_mesh
    part_types = np.asarray(mesh.voxel_part_types, dtype=np.int32)
    centers = np.asarray(mesh.voxel_centers, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    cast = np.flatnonzero(part_types == PART_TYPE_CAST)
    peak = cast[values[cast] >= 0.5 * np.max(values[cast])]
    weights = values[peak]
    return (centers[peak] * weights[:, None]).sum(axis=0) / weights.sum()


class TetPostAnalysesAgreeWithVoxel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tet = _run_all_analyses("tet", TET_EDGE_IN)
        cls.voxel = _run_all_analyses("voxel", VOXEL_SIZE_IN)

    def _metal(self, result) -> np.ndarray:
        return is_metal(np.asarray(result.voxel_mesh.voxel_part_types, dtype=np.int32))

    def test_no_nan_in_exported_arrays_on_metal_nodes(self) -> None:
        mesh = self.tet.voxel_mesh
        engine = self.tet.engine
        n = mesh.get_voxel_count()
        metal = self._metal(self.tet)
        arrays = {k: v for k, v in engine.export_results().items() if isinstance(v, np.ndarray)}
        arrays["stress.distortion_vectors"] = engine.stress_result.distortion_vectors
        arrays["stress.constraint_strain"] = engine.stress_result.constraint_strain
        arrays["feeding.path_length"] = engine.feeding_path_result.path_length
        arrays["feeding.history_open_fraction"] = engine.feeding_path_result.history_open_fraction
        arrays["shrinkage_display"] = engine.get_shrinkage_display_intensity()
        arrays["isolation"] = engine._calculate_isolation_score()
        self.assertIn("shrinkage_risk", arrays)
        self.assertIn("sdas", arrays)
        self.assertIn("gas_porosity_risk", arrays)
        for name, value in arrays.items():
            value = np.asarray(value)
            if value.ndim == 0 or value.shape[0] != n or value.dtype.kind != "f":
                continue
            self.assertFalse(np.any(np.isnan(value[metal])), f"{name} has NaN on metal nodes")
        for key in ("shrinkage_percentage", "shrinkage_volume", "max_stress_mpa", "avg_sdas_um", "fed_fraction"):
            self.assertTrue(np.isfinite(engine.export_results()[key]), key)

    def test_hot_spot_and_shrinkage_peaks_sit_under_the_riser_neck_on_both_meshes(self) -> None:
        for label, values in (
            ("hotspot", lambda r: r.engine.hotspot_intensity),
            ("shrinkage", lambda r: r.engine.get_shrinkage_display_intensity()),
        ):
            tet_peak = _cast_peak_centroid(self.tet, values(self.tet))
            voxel_peak = _cast_peak_centroid(self.voxel, values(self.voxel))
            print(f"\n  {label} peak: tet {tet_peak.round(2)} voxel {voxel_peak.round(2)}", file=sys.stderr)
            for peak in (tet_peak, voxel_peak):
                self.assertLess(float(np.hypot(peak[0], peak[1])), RISER_RADIUS_IN, label)
                self.assertGreater(peak[2], 0.4, label)
            self.assertLess(float(np.linalg.norm(tet_peak - voxel_peak)), 0.5, label)

    def test_feeding_paths_reach_the_riser(self) -> None:
        for result in (self.tet, self.voxel):
            part_types = np.asarray(result.voxel_mesh.voxel_part_types, dtype=np.int32)
            feeding = result.engine.feeding_path_result
            source = np.asarray(feeding.feeding_source)
            fed = source >= 0
            np.testing.assert_array_equal(part_types[source[fed]], PART_TYPE_RISER)
            self.assertGreater(feeding.fed_fraction, 0.8)
            cast = part_types == PART_TYPE_CAST
            self.assertGreater(float(np.mean(fed[cast])), 0.8)
        # Tet fed_fraction is a node-volume fraction, so it matches the voxel one.
        self.assertLess(abs(self.tet.engine.feeding_path_result.fed_fraction - self.voxel.engine.feeding_path_result.fed_fraction), 0.1)

    def _axis_distances(self, result):
        mesh = result.voxel_mesh
        engine = result.engine
        part_types = np.asarray(mesh.voxel_part_types, dtype=np.int32)
        centers = np.asarray(mesh.voxel_centers, dtype=np.float64)
        active = np.isin(part_types, (0, 1, 3))
        hops = engine._compute_graph_distance_to_riser(active)
        hop_in = _tet_hop_length_in(mesh) if result.mesh_type == "tet" else float(mesh.base_voxel_size)
        axis = np.flatnonzero((part_types == PART_TYPE_CAST) & (np.hypot(centers[:, 0], centers[:, 1]) < 0.15))
        axis = axis[np.argsort(centers[axis, 2])]
        path_length = np.asarray(engine.feeding_path_result.path_length, dtype=np.float64)[axis]
        return centers[axis, 2], hops[axis] * hop_in, path_length

    def test_distance_to_riser_is_monotone_along_the_axis_and_in_inches(self) -> None:
        for result, tolerance in ((self.tet, TET_EDGE_IN), (self.voxel, VOXEL_SIZE_IN)):
            z, graph_in, path_in = self._axis_distances(result)
            self.assertGreaterEqual(len(z), 4)
            expected = RISER_BASE_Z_IN - z
            for label, distance in (("graph", graph_in), ("feeding path", path_in)):
                self.assertTrue(np.all(np.isfinite(distance)), label)
                np.testing.assert_allclose(distance, expected, atol=tolerance + 0.15 * expected.max(), err_msg=label)
                # Monotone: farther below the riser never reads closer (up to one edge of slack).
                lower = z[:, None] < z[None, :] - tolerance
                self.assertTrue(np.all((distance[:, None] >= distance[None, :] - 1e-6)[lower]), label)
            print(f"\n  {result.mesh_type} axis z {z.round(2)} graph {graph_in.round(2)} in", file=sys.stderr)

    def test_isolation_scores_cover_the_same_range(self) -> None:
        stats = {}
        for name, result in (("tet", self.tet), ("voxel", self.voxel)):
            metal = self._metal(result)
            volumes = np.asarray(result.voxel_mesh.voxel_volumes, dtype=np.float64)[metal]
            score = result.engine._calculate_isolation_score()[metal]
            self.assertGreaterEqual(float(np.min(score)), 0.0)
            self.assertLessEqual(float(np.max(score)), 1.0)
            stats[name] = (
                _weighted_percentile(score, volumes, 50),
                _weighted_percentile(score, volumes, 95),
                float(np.max(score)),
            )
        print(f"\n  isolation p50/p95/max: tet {np.round(stats['tet'], 3)} voxel {np.round(stats['voxel'], 3)}", file=sys.stderr)
        self.assertLess(abs(stats["tet"][0] - stats["voxel"][0]), 0.15)
        self.assertLess(abs(stats["tet"][1] - stats["voxel"][1]), 0.15)
        self.assertGreater(stats["tet"][2], 0.4)
        self.assertGreater(stats["voxel"][2], 0.4)

    def test_shrinkage_volume_fraction_within_a_factor_of_two(self) -> None:
        tet = self.tet.engine.export_results()["shrinkage_percentage"]
        voxel = self.voxel.engine.export_results()["shrinkage_percentage"]
        print(f"\n  shrinkage %: tet {tet:.2f} voxel {voxel:.2f}", file=sys.stderr)
        self.assertGreater(tet, 0.0)
        self.assertGreater(voxel, 0.0)
        self.assertLess(max(tet, voxel) / min(tet, voxel), 2.0)
        # Tet percentage is the affected node volume over the total node volume.
        engine = self.tet.engine
        volumes = np.asarray(self.tet.voxel_mesh.voxel_volumes, dtype=np.float64)
        affected = (engine.shrinkage_probability > 0.2) | (engine.porosity_probability > 0.2)
        self.assertAlmostEqual(tet, 100.0 * volumes[affected].sum() / volumes.sum(), places=4)

    def test_riser_pipe_and_defect_volumes_use_node_volumes(self) -> None:
        engine = self.tet.engine
        volumes = np.asarray(self.tet.voxel_mesh.voxel_volumes, dtype=np.float64)
        pipe = engine.riser_pipe_report
        self.assertEqual(len(pipe), 1)
        metal = self._metal(self.tet)
        expected_demand = volumes[metal].sum() * float(engine.alloy.feeding_shrinkage) / 100.0
        # Demand covers the riser pool (riser + the casting it feeds), at most all metal.
        self.assertGreater(pipe[0]["demand_in3"], 0.5 * expected_demand)
        self.assertLessEqual(pipe[0]["demand_in3"], expected_demand * 1.001)
        estimate = engine.get_defect_volume_estimate()
        self.assertAlmostEqual(estimate["casting_volume_in3"], float(volumes.sum()), places=3)

    def test_orphan_rule_flags_at_most_the_flush_chill_edge_on_tets(self) -> None:
        # The tet union is conforming, so no gap-fill metal exists. The degree
        # fraction of the smothered rule still flags nodes on the line where
        # the chill side is flush with the casting edge (half chill, half metal
        # neighbours) - the voxel rule flags the same edge (13 voxels here).
        mesh = self.tet.voxel_mesh
        orphan = self.tet.engine._get_orphan_metal_mask()
        part_types = np.asarray(mesh.voxel_part_types, dtype=np.int32)
        volumes = np.asarray(mesh.voxel_volumes, dtype=np.float64)
        metal = self._metal(self.tet)
        self.assertLess(volumes[orphan].sum(), 0.01 * volumes[metal].sum())
        for node in np.flatnonzero(orphan):
            row = np.asarray(mesh.voxel_neighbors)[node]
            self.assertIn(PART_TYPE_CHILL, part_types[row[row >= 0]])

    def test_defect_pocket_on_tet_uses_full_rows_and_node_volumes(self) -> None:
        mesh = self.tet.voxel_mesh
        engine = self.tet.engine
        display = engine.get_shrinkage_display_intensity()
        anchor = int(np.argmax(display))
        pocket = extract_pocket_at(
            voxel_index=anchor,
            display_intensity=display,
            voxel_neighbors=mesh.voxel_neighbors,
            voxel_centers=mesh.voxel_centers,
            voxel_volumes=mesh.voxel_volumes,
            shrinkage_risk=engine.shrinkage_probability,
            porosity_risk=engine.porosity_probability,
            threshold=0.3,
            mesh_type="tet",
        )
        self.assertIsNotNone(pocket)
        volumes = np.asarray(mesh.voxel_volumes, dtype=np.float64)
        self.assertAlmostEqual(pocket.geometric_volume_in3, float(volumes[pocket.voxel_indices].sum()), places=4)
        # Connected over the full adjacency: every member above threshold, and no
        # above-threshold neighbour of a member left out.
        members = set(pocket.voxel_indices.tolist())
        rows = np.asarray(mesh.voxel_neighbors)[pocket.voxel_indices]
        neighbours = rows[rows >= 0]
        outside = [n for n in neighbours.tolist() if n not in members]
        self.assertTrue(all(display[n] < 0.3 for n in outside))

    def test_riser_designer_accepts_tet_volumes_and_moves_along_edges(self) -> None:
        mesh = self.tet.voxel_mesh
        cast = np.flatnonzero(np.asarray(mesh.voxel_part_types) == PART_TYPE_CAST)
        index_of = {int(g): i for i, g in enumerate(cast)}
        neighbors = np.asarray(mesh.voxel_neighbors)[cast]
        local = np.where(neighbors >= 0, [[index_of.get(int(n), -1) for n in row] for row in neighbors], -1)
        volumes = np.asarray(mesh.voxel_volumes, dtype=np.float64)[cast]
        designer = RiserDesigner(
            voxel_positions=np.asarray(mesh.voxel_centers, dtype=np.float64)[cast],
            voxel_size=float(mesh.base_voxel_size),
            neighbors=local,
            surface_mask=np.asarray(mesh.surface_voxels)[cast],
            alloy=self.tet.engine.alloy,
            params=RiserDesignParameters(min_diameter=0.5, max_diameter=12.0),
            voxel_volumes=volumes,
            mesh_type="tet",
        )
        recommendations = designer.design_risers(max_risers=1)
        self.assertEqual(len(recommendations), 1)
        hotspot = designer.hotspots[0]
        self.assertAlmostEqual(hotspot.volume, float(volumes[hotspot.indices].sum()), places=6)
        moves = designer._candidate_attachment_moves(recommendations[0])
        row = local[recommendations[0].voxel_index]
        self.assertTrue(moves)
        self.assertTrue(all(index in row for index, _ in moves))
        designer.optimize_positions(iterations=1)


class TetBranchUnitTests(unittest.TestCase):
    def test_volume_weighted_isolation_equals_count_with_uniform_volumes(self) -> None:
        rng = np.random.default_rng(3)
        times = rng.uniform(1.0, 10.0, 50).astype(np.float32)
        neighbors = rng.integers(-1, 50, size=(50, 12)).astype(np.int32)
        part_types = np.zeros(50, dtype=np.int32)
        part_types[:5] = PART_TYPE_CHILL
        count = compute_isolation_score_parallel(times, neighbors, part_types)
        weighted = compute_isolation_score_volume_weighted(times, neighbors, part_types, np.full(50, 0.3))
        np.testing.assert_allclose(weighted, count, atol=1e-6)

    def test_volume_weighted_isolation_counts_big_neighbours_more(self) -> None:
        times = np.array([5.0, 1.0, 9.0], dtype=np.float32)
        neighbors = np.array([[1, 2], [0, -1], [0, -1]], dtype=np.int32)
        weighted = compute_isolation_score_volume_weighted(
            times, neighbors, np.zeros(3, dtype=np.int32), np.array([1.0, 3.0, 1.0])
        )
        self.assertAlmostEqual(float(weighted[0]), 0.75, places=6)

    def _fake_tet_engine(self, centers, neighbors, part_types, *, mean_edge=1.0):
        mesh = types.SimpleNamespace(
            mesh_type="tet",
            voxel_centers=np.asarray(centers, dtype=np.float32),
            voxel_neighbors=np.asarray(neighbors, dtype=np.int32),
            voxel_part_types=np.asarray(part_types, dtype=np.int8),
            voxel_sizes=np.full(len(centers), mean_edge, dtype=np.float32),
            base_voxel_size=mean_edge,
            mean_edge_length=mean_edge,
            metal_part_bounds=[],
        )
        engine = SolidificationEngine.__new__(SolidificationEngine)
        engine.mesh = mesh
        return engine

    def test_tet_graph_distance_is_edge_length_over_mean_edge(self) -> None:
        # Chain riser(0) - 1 - 2 - 3 with edges of 0.5, 1.0 and 2.0 in.
        centers = [[0, 0, 0], [0, 0, -0.5], [0, 0, -1.5], [0, 0, -3.5]]
        neighbors = [[1, -1], [0, 2], [1, 3], [2, -1]]
        engine = self._fake_tet_engine(centers, neighbors, [PART_TYPE_RISER, 0, 0, 0], mean_edge=0.5)
        hops = engine._compute_graph_distance_to_riser(np.ones(4, dtype=bool))
        np.testing.assert_allclose(hops * 0.5, [0.0, 0.5, 1.5, 3.5], atol=1e-6)
        # A non-traversable node cuts the path.
        blocked = engine._compute_graph_distance_to_riser(np.array([True, True, False, True]))
        self.assertTrue(np.isinf(blocked[3]))

    def test_tet_smothered_rule_uses_a_fraction_of_node_degree(self) -> None:
        # Node 0 is metal with 3 chill neighbours and 2 metal ones (degree 5):
        # 3 >= 8/26 * 5 and 3 >= 2, so it is smothered. On a 26-slot voxel row
        # 3 chill neighbours would be far below the absolute 8.
        centers = np.zeros((6, 3))
        centers[:, 0] = np.arange(6)
        neighbors = [
            [1, 2, 3, 4, 5],
            [0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1],
            [0, 5, -1, -1, -1],
            [0, 4, -1, -1, -1],
        ]
        part_types = [0, PART_TYPE_CHILL, PART_TYPE_CHILL, PART_TYPE_CHILL, 0, 0]
        engine = self._fake_tet_engine(centers, neighbors, part_types)
        orphan = engine._get_orphan_metal_mask()
        self.assertTrue(orphan[0])
        # 2 chill against 3 metal neighbours: fewer chill than metal, not smothered.
        part_types[3] = 0
        engine = self._fake_tet_engine(centers, neighbors, part_types)
        self.assertFalse(engine._get_orphan_metal_mask()[0])

    def test_pocket_tet_mode_ignores_connectivity_and_uses_full_rows(self) -> None:
        # Node 0's only link to node 1 sits in slot 7. With voxel semantics and
        # connectivity=6 the slot is never read.
        neighbors = np.full((2, 8), -1, dtype=np.int32)
        neighbors[0, 7] = 1
        neighbors[1, 0] = 0
        args = dict(
            voxel_index=0,
            display_intensity=np.array([0.9, 0.8], dtype=np.float32),
            voxel_neighbors=neighbors,
            voxel_centers=np.zeros((2, 3)),
            voxel_volumes=np.array([0.25, 0.5]),
            shrinkage_risk=None,
            porosity_risk=None,
            threshold=0.5,
            connectivity=6,
        )
        self.assertEqual(extract_pocket_at(**args).voxel_count, 1)
        pocket = extract_pocket_at(**args, mesh_type="tet")
        self.assertEqual(pocket.voxel_count, 2)
        self.assertAlmostEqual(pocket.geometric_volume_in3, 0.75)

    def test_expand_front_buffer_follows_table_width(self) -> None:
        # One newly solid node with 30 liquid neighbours (K = 30 > 26).
        n = 31
        neighbors = np.full((n, 30), -1, dtype=np.int32)
        neighbors[0] = np.arange(1, 31)
        states = np.zeros(n, dtype=np.float32)
        states[0] = 1.0
        front, count = expand_front(
            np.zeros(0, dtype=np.int32), 0, np.array([0], dtype=np.int32), 1, states, neighbors
        )
        self.assertEqual(count, 30)
        np.testing.assert_array_equal(np.sort(front), np.arange(1, 31))

    def test_riser_designer_tet_without_volumes_keeps_size_cubed(self) -> None:
        positions = np.array([[0, 0, 0], [0, 0, 0.2], [0, 0, 0.4], [0, 0, 0.6]], dtype=np.float64)
        surface = np.array([True, False, False, True])
        common = dict(
            voxel_positions=positions,
            voxel_size=0.2,
            neighbors=np.array([[1, -1], [0, 2], [1, 3], [2, -1]]),
            surface_mask=surface,
            alloy=None,
            mesh_type="tet",
        )
        designer = RiserDesigner(**common)
        hotspot = designer._heuvers.find_hotspots(n_hotspots=1)[0]
        self.assertAlmostEqual(hotspot.volume, len(hotspot.indices) * 0.2 ** 3)
        volumes = np.array([0.1, 0.2, 0.3, 0.4])
        designer = RiserDesigner(**common, voxel_volumes=volumes)
        hotspot = designer._heuvers.find_hotspots(n_hotspots=1)[0]
        self.assertAlmostEqual(hotspot.volume, float(volumes[hotspot.indices].sum()))
        with self.assertRaises(ValueError):
            RiserDesigner(**common, voxel_volumes=np.ones(3))

    # review_integration.md P1-2: coverage, move acceptance and hot-spot
    # priority weight tet nodes by volume.

    def _coverage_designer(self, voxel_volumes, mesh_type="tet") -> RiserDesigner:
        # Ten small nodes at x = 0.0..0.9 (5 is L), M at 1.5 (start), R at 2.5,
        # one big node B at 3.0. M's edges reach L and R.
        x = np.concatenate([np.arange(10) * 0.1, [1.5, 2.5, 3.0]])
        positions = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
        neighbors = np.full((13, 2), -1, dtype=np.int32)
        neighbors[10] = [5, 11]
        designer = RiserDesigner(
            voxel_positions=positions,
            voxel_size=0.1,
            neighbors=neighbors,
            surface_mask=np.ones(13, dtype=bool),
            alloy=None,
            voxel_volumes=voxel_volumes,
            mesh_type=mesh_type,
        )
        designer._feeding_distance = lambda diameter: 1.05
        designer._recommendations = [
            RiserRecommendation(
                position=tuple(positions[10]),
                voxel_index=10,
                diameter=1.0,
                feeding_zone=designer._calculate_feeding_zone(10, 1.0),
            )
        ]
        return designer

    def test_riser_coverage_is_a_node_volume_share_on_tets(self) -> None:
        volumes = np.array([0.01] * 12 + [5.0])
        designer = self._coverage_designer(volumes)
        covered = designer._recommendations[0].feeding_zone  # x in [0.5, 2.5]: 5 small, M, R
        self.assertEqual(len(covered), 7)
        coverage = designer.analyze_coverage()
        self.assertAlmostEqual(coverage["covered_fraction"], 0.07 / volumes.sum(), places=9)
        self.assertEqual(coverage["covered_voxels"], 7)
        # Voxel meshes keep the count share.
        voxel = self._coverage_designer(None, mesh_type="voxel")
        self.assertAlmostEqual(voxel.analyze_coverage()["covered_fraction"], 7 / 13)

    def test_optimize_positions_moves_toward_more_volume_not_more_nodes(self) -> None:
        # L covers 11 small nodes (0.11 in3); R covers 3 nodes incl. B (5.02 in3).
        designer = self._coverage_designer(np.array([0.01] * 12 + [5.0]))
        self.assertEqual(designer.optimize_positions(iterations=1)[0].voxel_index, 11)
        # Without node volumes the count share still picks L.
        designer = self._coverage_designer(None)
        self.assertEqual(designer.optimize_positions(iterations=1)[0].voxel_index, 5)

    def test_hot_spot_priority_weights_cluster_volume_on_tets(self) -> None:
        positions = np.array([[0, 0, 0], [0, 0, 0.2], [0, 0, 0.4], [0, 0, 0.6]], dtype=np.float64)
        common = dict(
            voxel_positions=positions,
            voxel_size=0.2,
            neighbors=np.array([[1, -1], [0, 2], [1, 3], [2, -1]]),
            surface_mask=np.array([True, False, False, True]),
            alloy=None,
            mesh_type="tet",
        )
        volumes = np.array([0.1, 0.2, 0.3, 0.4])
        heuvers = RiserDesigner(**common, voxel_volumes=volumes)._heuvers
        hotspot = heuvers.find_hotspots(n_hotspots=1)[0]
        distance = heuvers.calculate_distance_to_surface()[hotspot.center_index]
        self.assertAlmostEqual(hotspot.priority, distance * float(volumes[hotspot.indices].sum()))
        heuvers = RiserDesigner(**common)._heuvers
        hotspot = heuvers.find_hotspots(n_hotspots=1)[0]
        self.assertAlmostEqual(hotspot.priority, distance * len(hotspot.indices))

    # review_integration.md P2-1: gas porosity sees the removed cores on tets.

    def _gas_porosity_engine(self, contact_codes) -> SolidificationEngine:
        # Two tets sharing face (1, 2, 3); every node is on the boundary.
        centers = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32)
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 4], [1, 4, 3], [2, 3, 4]], dtype=np.int32)
        mesh = types.SimpleNamespace(
            mesh_type="tet",
            voxel_centers=centers,
            voxel_part_types=np.zeros(5, dtype=np.int8),
            boundary_faces=faces,
            boundary_face_contact_codes=np.asarray(contact_codes, dtype=np.int8),
            get_surface_voxel_indices=lambda: np.arange(5),
        )
        engine = SolidificationEngine.__new__(SolidificationEngine)
        engine.mesh = mesh
        engine.alloy = types.SimpleNamespace(gas_sensitivity=0.5)
        engine.local_solidification_time = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        engine.niyama_risk = None
        return engine

    def _gas_generation(self, engine, venting: float) -> np.ndarray:
        from physics import gas_porosity

        original = gas_porosity.compute_surface_gas_generation

        def with_venting(*args, **kwargs):
            kwargs["core_venting_effectiveness"] = venting
            return original(*args, **kwargs)

        with mock.patch.object(gas_porosity, "compute_surface_gas_generation", side_effect=with_venting), \
                mock.patch("builtins.print"):
            engine._calculate_gas_porosity()
        return np.asarray(engine.gas_generation, dtype=np.float64)

    def test_core_venting_changes_the_tet_gas_result(self) -> None:
        # Face (1, 2, 4) touches a removed CORE: nodes 1, 2, 4 are core surface.
        engine = self._gas_porosity_engine([-1, -1, -1, 4, -1, -1])
        unvented = self._gas_generation(engine, 0.0)
        vented = self._gas_generation(engine, 1.0)
        self.assertFalse(np.allclose(unvented, vented))
        np.testing.assert_array_equal(engine.mesh.voxel_part_types, 0)  # the mesh array is not modified
        # Without a core face venting has nothing to act on.
        engine = self._gas_porosity_engine([-1] * 6)
        np.testing.assert_allclose(self._gas_generation(engine, 0.0), self._gas_generation(engine, 1.0))


if __name__ == "__main__":
    unittest.main()
