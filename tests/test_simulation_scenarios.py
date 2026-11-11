from __future__ import annotations

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.helpers.ci_models import ci_model
from core.memory_limits import MemorySeverity, MemoryValidationResult
from core.materials import get_alloy
from core.part_manager import PartManager, PartType
from core.solidification import AnalysisOptions, SOLVER_MODE_SCHEDULE_LEGACY, SolidificationEngine
from core.simulation_pipeline import PipelineInputs, build_part_manager_from_parts, run_pipeline


def _ok_validation(*_args, **_kwargs) -> MemoryValidationResult:
    return MemoryValidationResult(
        can_proceed=True,
        severity=MemorySeverity.OK,
        estimated_voxels=1_000,
        max_allowed_voxels=1_000_000,
        estimated_memory_mb=1.0,
        available_memory_mb=10_000.0,
        message="ok",
    )


def _memory_monitor_check(_self: object) -> tuple[bool, str]:
    return True, "ok"


def _run_pipeline_with_patched_memory(
    part_manager: PartManager,
    *,
    alloy_name: str = "WCB",
    hot_tear: bool = False,
    stress_analysis: bool = False,
    sand_restraint_index: float = 0.5,
    extra_analysis_options: dict | None = None,
    voxel_cut_cells: bool | None = None,
    solidification_params: dict | None = None,
) -> object:
    """Run the pipeline with the voxel memory checks patched out.

    ``voxel_cut_cells`` None follows ``settings.mesh.voxel_cut_cells``.
    """
    analysis_options = {
        "shrinkage": True,
        "hotspot": True,
        "niyama": True,
        "fraction_liquid": True,
        "hot_tear": hot_tear,
        "stress_analysis": stress_analysis,
        "sand_restraint_index": sand_restraint_index,
    }
    if extra_analysis_options:
        analysis_options.update(extra_analysis_options)

    inputs = PipelineInputs(
        alloy_name=alloy_name,
        voxel_size=0.5,
        refinement_level=0,
        memory_mode="conservative",
        analysis_options=analysis_options,
        use_gpu=False,
        use_incremental=False,
        show_progress=False,
        max_voxels_for_visualization=5_000,
        voxel_cut_cells=voxel_cut_cells,
        solidification_params=solidification_params,
    )

    with ExitStack() as stack:
        stack.enter_context(
            patch("core.voxel_mesh.auto_adjust_voxel_size", return_value=(0.5, False, ""))
        )
        stack.enter_context(
            patch("core.voxel_mesh.validate_mesh_parameters", side_effect=_ok_validation)
        )
        stack.enter_context(patch("core.voxel_mesh.MemoryMonitor.start", lambda self: None))
        stack.enter_context(
            patch("core.voxel_mesh.MemoryMonitor.check", _memory_monitor_check)
        )
        return run_pipeline(inputs, part_manager=part_manager).engine


def _single_cast_manager(filepath: str) -> PartManager:
    manager = PartManager()
    part_id = manager.load_part(filepath, name="cast")
    assert part_id is not None
    part = manager.get_part(part_id)
    assert part is not None
    part.set_part_type(PartType.CAST)
    part.set_position(0.0, 0.0, 0.0)
    return manager


def _cast_with_riser_manager() -> PartManager:
    manager = PartManager()
    cast_id = manager.load_part(ci_model("small_block.stl"), name="cast")
    riser_id = manager.load_part(ci_model("cylinder.stl"), name="riser")
    assert cast_id is not None and riser_id is not None

    cast = manager.get_part(cast_id)
    riser = manager.get_part(riser_id)
    assert cast is not None and riser is not None

    cast.set_part_type(PartType.CAST)
    riser.set_part_type(PartType.RISER)
    cast.set_position(0.0, 0.0, 0.0)
    riser.set_position(0.0, 0.0, 0.5)
    return manager


def _cast_with_chill_manager() -> PartManager:
    manager = PartManager()
    cast_id = manager.load_part(ci_model("block_2inch.stl"), name="cast")
    chill_id = manager.load_part(ci_model("small_block.stl"), name="chill")
    assert cast_id is not None and chill_id is not None

    cast = manager.get_part(cast_id)
    chill = manager.get_part(chill_id)
    assert cast is not None and chill is not None

    cast.set_part_type(PartType.CAST)
    chill.set_part_type(PartType.CHILL)
    cast.set_position(0.0, 0.0, 0.0)
    chill.set_position(1.5, 0.0, 0.0)
    return manager


def _cast_with_chill_and_riser_manager() -> PartManager:
    """Casting with a chill on one face and a riser in full contact on top.

    The riser makes the feeding-path trace run, which is what exposes any
    feeding-history mis-scoring of the fast-freezing metal against the chill.
    """
    manager = PartManager()
    cast_id = manager.load_part(ci_model("block_2inch.stl"), name="cast")
    chill_id = manager.load_part(ci_model("small_block.stl"), name="chill")
    riser_id = manager.load_part(ci_model("cylinder.stl"), name="riser")
    assert cast_id is not None and chill_id is not None and riser_id is not None

    cast = manager.get_part(cast_id)
    chill = manager.get_part(chill_id)
    riser = manager.get_part(riser_id)
    assert cast is not None and chill is not None and riser is not None

    cast.set_part_type(PartType.CAST)
    chill.set_part_type(PartType.CHILL)
    riser.set_part_type(PartType.RISER)
    cast.set_position(0.0, 0.0, 0.0)
    chill.set_position(1.5, 0.0, 0.0)   # flush against the +x face
    riser.set_position(0.0, 0.0, 1.0)   # base flush on the top face
    return manager


def _cast_with_two_risers_manager() -> PartManager:
    manager = PartManager()
    cast_id = manager.load_part(ci_model("block_2inch.stl"), name="cast")
    riser_a_id = manager.load_part(ci_model("small_block.stl"), name="riser_a")
    riser_b_id = manager.load_part(ci_model("small_block.stl"), name="riser_b")
    assert cast_id is not None and riser_a_id is not None and riser_b_id is not None

    cast = manager.get_part(cast_id)
    riser_a = manager.get_part(riser_a_id)
    riser_b = manager.get_part(riser_b_id)
    assert cast is not None and riser_a is not None and riser_b is not None

    cast.set_part_type(PartType.CAST)
    riser_a.set_part_type(PartType.RISER)
    riser_b.set_part_type(PartType.RISER)
    cast.set_position(0.0, 0.0, 0.0)
    riser_a.set_position(-0.75, 0.0, 1.0)
    riser_b.set_position(0.75, 0.0, 1.0)
    return manager


def _cast_with_riser_post_manager() -> PartManager:
    manager = PartManager()
    cast_id = manager.load_part(ci_model("block_2inch.stl"), name="cast")
    post_id = manager.load_part(ci_model("small_block.stl"), name="riser_post")
    riser_id = manager.load_part(ci_model("cylinder.stl"), name="riser")
    assert cast_id is not None and post_id is not None and riser_id is not None

    cast = manager.get_part(cast_id)
    post = manager.get_part(post_id)
    riser = manager.get_part(riser_id)
    assert cast is not None and post is not None and riser is not None

    cast.set_part_type(PartType.CAST)
    post.set_part_type(PartType.RISER_POST)
    riser.set_part_type(PartType.RISER)
    cast.set_position(0.0, 0.0, 0.0)
    post.set_position(0.0, 0.0, 1.0)
    riser.set_position(0.0, 0.0, 2.0)
    return manager


class ScenarioRegressionTests(unittest.TestCase):
    def test_saved_riser_post_payload_is_normalized_on_load(self) -> None:
        manager = build_part_manager_from_parts([
            {
                "name": "Riser Post 1",
                "part_type": "riser",
                "filepath": ci_model("small_block.stl"),
            }
        ])

        parts = manager.get_all_parts()
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].part_type, PartType.RISER_POST)

    def test_surface_cast_voxels_start_liquid_in_fraction_history(self) -> None:
        engine = _run_pipeline_with_patched_memory(
            _single_cast_manager(ci_model("small_block.stl"))
        )

        self.assertTrue(engine.fraction_liquid_history)
        first_snapshot = np.asarray(engine.fraction_liquid_history[0], dtype=np.float32)
        surface_indices = engine.mesh.get_surface_voxel_indices()
        self.assertGreater(len(surface_indices), 0)
        self.assertTrue(np.all(first_snapshot[surface_indices] < 1.0))

    def test_full_contact_chill_boosts_adjacent_cast_face(self) -> None:
        centers = []
        part_types = []
        for z_value, part_type in ((0.0, 2), (1.0, 0), (2.0, 0)):
            for x_value in range(3):
                for y_value in range(3):
                    centers.append([float(x_value), float(y_value), z_value])
                    part_types.append(part_type)

        centers_array = np.asarray(centers, dtype=np.float32)
        part_types_array = np.asarray(part_types, dtype=np.int32)
        mesh = SimpleNamespace(
            voxel_centers=centers_array,
            voxel_part_types=part_types_array,
            surface_voxels=part_types_array == 2,
            voxel_states=np.zeros(len(centers_array), dtype=np.float32),
            voxel_neighbors=np.full((len(centers_array), 26), -1, dtype=np.int32),
            base_voxel_size=1.0,
        )
        engine = SolidificationEngine(mesh, get_alloy("WCB"), AnalysisOptions())

        chill_boost = engine._calculate_chill_influence()

        contact_face = (part_types_array == 0) & np.isclose(centers_array[:, 2], 1.0)
        upper_face = (part_types_array == 0) & np.isclose(centers_array[:, 2], 2.0)

        self.assertTrue(np.all(chill_boost[contact_face] > 1.0))
        self.assertTrue(np.allclose(chill_boost[upper_face], 1.0))

    def test_schedule_progresses_away_from_full_contact_chill(self) -> None:
        centers = []
        part_types = []
        for z_value, part_type in ((0.0, 2), (1.0, 0), (2.0, 0), (3.0, 0), (4.0, 1), (5.0, 1)):
            for x_value in range(3):
                for y_value in range(3):
                    if part_type == 1 and (x_value, y_value) != (1, 1):
                        continue
                    centers.append([float(x_value), float(y_value), z_value])
                    part_types.append(part_type)

        centers_array = np.asarray(centers, dtype=np.float32)
        part_types_array = np.asarray(part_types, dtype=np.int32)
        n_voxels = len(centers_array)
        neighbors = np.full((n_voxels, 26), -1, dtype=np.int32)
        for idx, center in enumerate(centers_array):
            distances = np.linalg.norm(centers_array - center, axis=1)
            adjacent = [
                neighbor_idx
                for neighbor_idx, distance in sorted(
                    enumerate(distances),
                    key=lambda item: item[1],
                )
                if neighbor_idx != idx and distance <= 1.01
            ]
            neighbors[idx, : len(adjacent[:26])] = adjacent[:26]

        mesh = SimpleNamespace(
            voxel_centers=centers_array,
            voxel_part_types=part_types_array,
            surface_voxels=np.ones(n_voxels, dtype=bool),
            voxel_states=np.zeros(n_voxels, dtype=np.float32),
            voxel_neighbors=neighbors,
            voxel_moduli=np.where(part_types_array == 2, 0.0, 1.0).astype(np.float32),
            base_voxel_size=1.0,
        )
        options = AnalysisOptions(
            shrinkage=False,
            hotspot=False,
            fraction_liquid=True,
            solver_mode=SOLVER_MODE_SCHEDULE_LEGACY,
        )
        engine = SolidificationEngine(mesh, get_alloy("WCB"), options)
        engine.max_time = 1000.0
        engine.solidification_start_times = np.zeros(n_voxels, dtype=np.float32)
        engine.solidification_end_times = np.zeros(n_voxels, dtype=np.float32)
        engine.local_solidification_time = np.zeros(n_voxels, dtype=np.float32)
        engine.solidification_times = np.zeros(n_voxels, dtype=np.float32)

        chill_boost = engine._calculate_chill_influence()
        engine._initialize_solidification_schedule(chill_boost, np.ones(n_voxels, dtype=np.float32))

        lower_cast = (part_types_array == 0) & np.isclose(centers_array[:, 2], 1.0)
        middle_cast = (part_types_array == 0) & np.isclose(centers_array[:, 2], 2.0)
        upper_cast = (part_types_array == 0) & np.isclose(centers_array[:, 2], 3.0)
        riser = part_types_array == 1

        self.assertLess(
            float(np.max(engine.solidification_end_times[lower_cast])),
            float(np.min(engine.solidification_end_times[middle_cast])),
        )
        self.assertLess(
            float(np.max(engine.solidification_end_times[middle_cast])),
            float(np.min(engine.solidification_end_times[upper_cast])),
        )
        self.assertLess(
            float(np.max(engine.solidification_end_times[upper_cast])),
            float(np.min(engine.solidification_end_times[riser])),
        )

    def test_thicker_sections_solidify_slower_than_thin_sections(self) -> None:
        thin_engine = _run_pipeline_with_patched_memory(
            _single_cast_manager(ci_model("small_block.stl"))
        )
        thick_engine = _run_pipeline_with_patched_memory(
            _single_cast_manager(ci_model("block_2inch.stl"))
        )

        self.assertGreater(
            float(np.max(thick_engine.solidification_times)),
            float(np.max(thin_engine.solidification_times)),
        )

    def test_riser_reduces_cast_shrinkage_and_keeps_pipe_behavior_in_riser(self) -> None:
        # Plain voxels: the shrinkage thresholds (> 0.15 counts) were calibrated on old-voxel geometry at 0.5 in.
        baseline_engine = _run_pipeline_with_patched_memory(
            _single_cast_manager(ci_model("small_block.stl")), voxel_cut_cells=False
        )
        riser_engine = _run_pipeline_with_patched_memory(_cast_with_riser_manager(), voxel_cut_cells=False)

        cast_mask = riser_engine.mesh.voxel_part_types == 0
        riser_mask = riser_engine.mesh.voxel_part_types == 1

        self.assertTrue(np.any(cast_mask))
        self.assertTrue(np.any(riser_mask))
        self.assertLess(
            float(np.mean(riser_engine.shrinkage_probability[cast_mask])),
            float(np.mean(baseline_engine.shrinkage_probability)),
        )
        self.assertGreater(
            float(np.mean(riser_engine.shrinkage_probability[riser_mask])),
            float(np.mean(riser_engine.shrinkage_probability[cast_mask])),
        )
        self.assertIsNotNone(riser_engine.feeding_path_lengths)
        valid_cast_paths = cast_mask & (riser_engine.feeding_path_lengths >= 0.0)
        self.assertTrue(np.any(valid_cast_paths))
        # Re-baselined 2026-08-14 for the thermal-coupled physics overhaul:
        # with same-step latent heat the mush freezes within a couple of steps,
        # so the history-aware tracer keeps only the riser-adjacent cast voxels
        # open (feeding_path_lengths == surface_distances there) instead of
        # routing paths that exceed the straight-line surface distance. The
        # surviving guarantee is that the traced path still reaches into the
        # casting from the riser.
        self.assertGreater(
            float(np.max(riser_engine.feeding_path_lengths[valid_cast_paths])),
            0.0,
        )
        # A sleeved riser passes no heat to the sand, so this 2 x 3 in riser on a 1 in block is still
        # partly liquid when the run stops: no time (NaN) means it froze after everything else.
        riser_times = np.asarray(riser_engine.solidification_times, dtype=np.float64)[riser_mask]
        self.assertGreaterEqual(
            float(np.max(np.where(np.isnan(riser_times), np.inf, riser_times))),
            float(np.max(riser_engine.solidification_times[cast_mask])),
        )
        self.assertGreaterEqual(
            int(np.count_nonzero(riser_engine.shrinkage_probability[riser_mask] > 0.15)),
            8,
        )
        self.assertEqual(
            int(np.count_nonzero(riser_engine.shrinkage_probability[cast_mask] > 0.15)),
            0,
        )

    def test_riser_post_behaves_like_feeder_neck_not_primary_defect_sink(self) -> None:
        # Plain voxels: the fixture post stops 0.5 in below the riser; only 0.5 in voxels close that gap.
        baseline_engine = _run_pipeline_with_patched_memory(
            _single_cast_manager(ci_model("block_2inch.stl")), voxel_cut_cells=False
        )
        feeder_engine = _run_pipeline_with_patched_memory(_cast_with_riser_post_manager(), voxel_cut_cells=False)

        cast_mask = feeder_engine.mesh.voxel_part_types == 0
        riser_mask = feeder_engine.mesh.voxel_part_types == 1
        post_mask = feeder_engine.mesh.voxel_part_types == 3

        self.assertTrue(np.any(cast_mask))
        self.assertTrue(np.any(riser_mask))
        self.assertTrue(np.any(post_mask))

        self.assertGreaterEqual(
            float(np.max(feeder_engine.solidification_times[riser_mask])),
            float(np.max(feeder_engine.solidification_times[post_mask])),
        )
        self.assertLess(
            float(np.mean(feeder_engine.shrinkage_probability[post_mask])),
            float(np.mean(feeder_engine.shrinkage_probability[riser_mask])),
        )
        # Re-baselined 2026-08-14 (2nd pass) for the conservative-flux +
        # implicit-enthalpy physics: the riser body now holds the porosity
        # (post_porosity_to_riser_ratio ~0.47) because the feeder neck stays
        # feedable, so the post is NOT the primary defect sink. The riser
        # pipe carries the largest SHRINKAGE (asserted above).
        self.assertLess(
            float(np.mean(feeder_engine.porosity_probability[post_mask])),
            float(np.mean(feeder_engine.porosity_probability[riser_mask])),
        )
        self.assertLess(
            float(np.mean(feeder_engine.shrinkage_probability[cast_mask])),
            float(np.mean(baseline_engine.shrinkage_probability)),
        )

    def test_two_risers_reduce_cast_shrinkage_more_than_single_riser(self) -> None:
        # Plain voxels: at CAD size the 1 in risers freeze before the 2 in casting (cut cells: 130 s vs 166 s).
        baseline_engine = _run_pipeline_with_patched_memory(
            _single_cast_manager(ci_model("block_2inch.stl")), voxel_cut_cells=False
        )
        two_riser_engine = _run_pipeline_with_patched_memory(_cast_with_two_risers_manager(), voxel_cut_cells=False)

        two_cast_mask = two_riser_engine.mesh.voxel_part_types == 0
        two_riser_mask = two_riser_engine.mesh.voxel_part_types == 1

        self.assertLess(
            float(np.mean(two_riser_engine.shrinkage_probability[two_cast_mask])),
            float(np.mean(baseline_engine.shrinkage_probability)),
        )
        self.assertGreater(
            float(np.max(two_riser_engine.shrinkage_probability[two_riser_mask])),
            0.0,
        )
        self.assertIsNotNone(two_riser_engine.feeding_path_lengths)
        valid_cast_paths = two_cast_mask & (two_riser_engine.feeding_path_lengths >= 0.0)
        self.assertTrue(np.any(valid_cast_paths))

    def test_chill_reduces_nearby_shrinkage_and_solidification_time(self) -> None:
        baseline_engine = _run_pipeline_with_patched_memory(
            _single_cast_manager(ci_model("block_2inch.stl"))
        )
        chill_engine = _run_pipeline_with_patched_memory(_cast_with_chill_manager())

        baseline_cast = baseline_engine.mesh.voxel_part_types == 0
        chill_cast = chill_engine.mesh.voxel_part_types == 0
        baseline_mid_side = baseline_cast & np.isclose(baseline_engine.mesh.voxel_centers[:, 0], 0.5)
        chill_mid_side = chill_cast & np.isclose(chill_engine.mesh.voxel_centers[:, 0], 0.5)

        self.assertTrue(np.any(baseline_mid_side))
        self.assertTrue(np.any(chill_mid_side))
        self.assertLess(
            float(np.mean(chill_engine.shrinkage_probability[chill_mid_side])),
            float(np.mean(baseline_engine.shrinkage_probability[baseline_mid_side])),
        )
        self.assertLess(
            float(np.mean(chill_engine.porosity_probability[chill_mid_side])),
            float(np.mean(baseline_engine.porosity_probability[baseline_mid_side])),
        )
        self.assertLess(
            float(np.mean(chill_engine.solidification_times[chill_mid_side])),
            float(np.mean(baseline_engine.solidification_times[baseline_mid_side])),
        )

    def test_chill_voxels_are_excluded_from_shrinkage_porosity_risk(self) -> None:
        chill_engine = _run_pipeline_with_patched_memory(_cast_with_chill_manager())
        chill_mask = chill_engine.mesh.voxel_part_types == 2

        self.assertTrue(np.any(chill_mask))
        self.assertTrue(
            np.allclose(chill_engine.shrinkage_probability[chill_mask], 0.0),
            msg="Chill voxels should not accumulate shrinkage risk",
        )
        self.assertTrue(
            np.allclose(chill_engine.porosity_probability[chill_mask], 0.0),
            msg="Chill voxels should not accumulate porosity risk",
        )

    def test_directional_solidification_improves_niyama_near_chill(self) -> None:
        baseline_engine = _run_pipeline_with_patched_memory(
            _single_cast_manager(ci_model("block_2inch.stl"))
        )
        chill_engine = _run_pipeline_with_patched_memory(_cast_with_chill_manager())

        baseline_cast = baseline_engine.mesh.voxel_part_types == 0
        chill_cast = chill_engine.mesh.voxel_part_types == 0
        baseline_mid_side = baseline_cast & np.isclose(baseline_engine.mesh.voxel_centers[:, 0], 0.5)
        chill_mid_side = chill_cast & np.isclose(chill_engine.mesh.voxel_centers[:, 0], 0.5)

        self.assertTrue(np.any(baseline_mid_side))
        self.assertTrue(np.any(chill_mid_side))
        self.assertLess(
            float(np.mean(chill_engine.niyama_severity[chill_mid_side])),
            float(np.mean(baseline_engine.niyama_severity[baseline_mid_side])),
        )

    def test_nominal_run_output_sanity(self) -> None:
        engine = _run_pipeline_with_patched_memory(_single_cast_manager(ci_model("block_2inch.stl")))

        self.assertTrue(np.all(engine.shrinkage_probability >= 0.0))
        self.assertTrue(np.all(engine.shrinkage_probability <= 1.0))
        self.assertTrue(np.all(engine.porosity_probability >= 0.0))
        self.assertTrue(np.all(engine.porosity_probability <= 1.0))
        self.assertTrue(np.all(engine.solidification_times > 0.0))
        self.assertTrue(np.all(engine.solidification_times <= engine.time + 1e-6))

        snapshots = np.asarray(engine.fraction_liquid_history, dtype=np.float32)
        self.assertGreaterEqual(len(snapshots), 2)
        self.assertTrue(np.all(np.diff(snapshots, axis=0) >= -1e-6))

        feeding = engine._calculate_feeding_effectiveness()
        self.assertGreater(float(np.mean(feeding)), 0.5)
        self.assertGreater(np.count_nonzero(feeding > 0.9), 0)

    def test_small_block_has_single_interior_hotspot(self) -> None:
        engine = _run_pipeline_with_patched_memory(
            _single_cast_manager(ci_model("small_block.stl"))
        )

        hotspot_mask = engine.hotspot_intensity > 0.95
        hotspot_indices = np.flatnonzero(hotspot_mask)

        self.assertEqual(len(hotspot_indices), 1)
        center = engine.mesh.voxel_centers[hotspot_indices[0]]
        self.assertTrue(np.allclose(center, np.zeros(3, dtype=np.float32)))

    def test_higher_sand_restraint_increases_hot_tear_risk(self) -> None:
        low_restraint_engine = _run_pipeline_with_patched_memory(
            _single_cast_manager(ci_model("thick_thin_test.stl")),
            hot_tear=True,
            sand_restraint_index=0.1,
        )
        high_restraint_engine = _run_pipeline_with_patched_memory(
            _single_cast_manager(ci_model("thick_thin_test.stl")),
            hot_tear=True,
            sand_restraint_index=0.9,
        )

        self.assertGreater(
            float(np.max(high_restraint_engine.hot_tear_risk)),
            float(np.max(low_restraint_engine.hot_tear_risk)),
        )
        self.assertGreaterEqual(
            int(np.count_nonzero(high_restraint_engine.hot_tear_zones)),
            int(np.count_nonzero(low_restraint_engine.hot_tear_zones)),
        )

    def test_default_sand_restraint_matches_explicit_neutral(self) -> None:
        default_engine = _run_pipeline_with_patched_memory(
            _single_cast_manager(ci_model("thick_thin_test.stl")),
            hot_tear=True,
        )
        explicit_neutral_engine = _run_pipeline_with_patched_memory(
            _single_cast_manager(ci_model("thick_thin_test.stl")),
            hot_tear=True,
            sand_restraint_index=0.5,
        )

        np.testing.assert_allclose(
            default_engine.hot_tear_risk,
            explicit_neutral_engine.hot_tear_risk,
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            default_engine.hot_tear_zones,
            explicit_neutral_engine.hot_tear_zones,
        )

    def test_sand_restraint_scales_crack_without_changing_stress_fields(self) -> None:
        class _FakeStressAnalyzer:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def analyze(self, *_args, **_kwargs):
                return SimpleNamespace(
                    effective_stress=np.array([10.0, 10.0, 10.0], dtype=np.float32),
                    distortion_magnitude=np.array([0.2, 0.2, 0.2], dtype=np.float32),
                    crack_susceptibility=np.array([0.5, 0.5, 0.5], dtype=np.float32),
                    max_stress=10.0,
                    max_distortion=0.2,
                )

        mesh = SimpleNamespace(
            voxel_centers=np.array(
                [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            voxel_states=np.ones(3, dtype=np.float32),
            base_voxel_size=0.1,
        )

        low_engine = SolidificationEngine(
            mesh,
            get_alloy("WCB"),
            AnalysisOptions(stress_analysis=True, sand_restraint_index=0.1),
        )
        high_engine = SolidificationEngine(
            mesh,
            get_alloy("WCB"),
            AnalysisOptions(stress_analysis=True, sand_restraint_index=0.9),
        )
        low_engine.solidification_times = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        high_engine.solidification_times = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        surface_distances = np.array([0.0, 0.25, 1.0], dtype=np.float32)
        low_engine.surface_distances = surface_distances
        high_engine.surface_distances = surface_distances
        low_engine.feeding_distances = surface_distances
        high_engine.feeding_distances = surface_distances

        with patch("core.solidification.StressAnalyzer", _FakeStressAnalyzer):
            low_engine._calculate_stress_analysis()
            high_engine._calculate_stress_analysis()

        np.testing.assert_allclose(low_engine.effective_stress, high_engine.effective_stress)
        np.testing.assert_allclose(low_engine.distortion_magnitude, high_engine.distortion_magnitude)
        self.assertGreater(high_engine.crack_susceptibility[0], low_engine.crack_susceptibility[0])
        self.assertAlmostEqual(
            float(high_engine.crack_susceptibility[2]),
            float(low_engine.crack_susceptibility[2]),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
