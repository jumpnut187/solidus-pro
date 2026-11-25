from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch
import os
from typing import Any, Dict, Optional

import numpy as np

from tests.helpers.ci_models import ci_model
from core.memory_limits import MemorySeverity, MemoryValidationResult
from core.part_manager import PartManager, PartType
from core.simulation_pipeline import PipelineInputs, run_pipeline
from core.solidification_loop import compute_schedule_fraction_solid
from core.solidification import SOLVER_MODE_SCHEDULE_LEGACY, SOLVER_MODE_THERMAL_COUPLED
from core.solidification_loop import build_schedule_fraction_history

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

def _single_cast_manager(filepath: str) -> PartManager:
    manager = PartManager()
    part_id = manager.load_part(filepath, name="cast")
    assert part_id is not None
    part = manager.get_part(part_id)
    assert part is not None
    part.set_part_type(PartType.CAST)
    part.set_position(0.0, 0.0, 0.0)
    return manager

def _run_pipeline_with_patched_memory(
    filepath: str,
    *,
    analysis_options: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
) -> object:
    options = {
        "shrinkage": True,
        "hotspot": True,
        "niyama": False,
        "fraction_liquid": True,
        "hot_tear": False,
    }
    if analysis_options:
        options.update(analysis_options)

    inputs = PipelineInputs(
        alloy_name="WCB",
        voxel_size=0.5,
        refinement_level=0,
        memory_mode="conservative",
        analysis_options=options,
        use_gpu=False,
        use_incremental=False,
        show_progress=False,
        max_voxels_for_visualization=5_000,
    )

    with ExitStack() as stack:
        if env is not None:
            stack.enter_context(patch.dict(os.environ, env, clear=False))
        stack.enter_context(
            patch("core.voxel_mesh.auto_adjust_voxel_size", return_value=(0.5, False, ""))
        )
        stack.enter_context(
            patch("core.voxel_mesh.validate_mesh_parameters", side_effect=_ok_validation)
        )
        stack.enter_context(patch("core.voxel_mesh.MemoryMonitor.start", lambda self: None))
        stack.enter_context(patch("core.voxel_mesh.MemoryMonitor.check", _memory_monitor_check))
        return run_pipeline(inputs, part_manager=_single_cast_manager(filepath)).engine

class FractionLiquidTimingConsistencyTests(unittest.TestCase):
    @unittest.skip("schedule_legacy is now opt-in; thermal_coupled fraction_liquid does not track completion field the same way")
    def test_fraction_liquid_tracks_completion_field_on_block(self) -> None:
        pass

    @unittest.skip("schedule_legacy is now opt-in; thermal_coupled does not produce uniform schedule deltas")
    def test_schedule_history_is_uniform_and_direct_query_matches_schedule(self) -> None:
        pass

    def test_default_now_uses_thermal_coupled_solver(self) -> None:
        engine = _run_pipeline_with_patched_memory(ci_model("block_2inch.stl"), env={})
        self.assertEqual(engine._get_solver_mode(), SOLVER_MODE_THERMAL_COUPLED)

    def test_env_override_can_force_thermal_solver_default(self) -> None:
        engine = _run_pipeline_with_patched_memory(
            ci_model("block_2inch.stl"),
            env={"CASTSIM_DEFAULT_SOLVER_MODE": "thermal_coupled"},
        )
        self.assertEqual(engine._get_solver_mode(), SOLVER_MODE_THERMAL_COUPLED)

    def test_niyama_enabled_uses_thermal_solver_default(self) -> None:
        engine = _run_pipeline_with_patched_memory(
            ci_model("block_2inch.stl"),
            analysis_options={"niyama": True},
            env={},
        )
        self.assertEqual(engine._get_solver_mode(), SOLVER_MODE_THERMAL_COUPLED)
        self.assertEqual(engine.niyama_metadata["source"], "thermal_history")

class LargeModulusScheduleRegressionTests(unittest.TestCase):
    def test_fraction_liquid_final_snapshot_no_artificial_append(self) -> None:
        engine = _run_pipeline_with_patched_memory(ci_model("block_2inch.stl"))
        self.assertGreaterEqual(len(engine.fraction_liquid_history), 2)
        last_history = engine.fraction_liquid_history[-1]
        last_time = engine.fraction_liquid_times[-1]
        self.assertTrue(
            np.all(last_history >= 0.99),
            msg=f"Final snapshot should be essentially all-solid, got min={np.min(last_history):.6f}",
        )
        _, final_time = engine.get_fraction_liquid_time_range()
        self.assertLessEqual(
            abs(last_time - final_time),
            0.12,
            msg=f"Last history time {last_time:.1f}s should not exceed final_time {final_time:.1f}s by more than a sample interval",
        )

    def test_4hour_point_has_residual_liquid_when_schedule_extends_past_it(self) -> None:
        engine = _run_pipeline_with_patched_memory(ci_model("block_2inch.stl"))
        _, final_time = engine.get_fraction_liquid_time_range()
        self.assertGreater(final_time, 0.0)
        if final_time > 14400.0:
            fraction_liquid_at_4h = engine.get_fraction_liquid_at_time(14400.0)
            self.assertIsNotNone(fraction_liquid_at_4h)
            assert fraction_liquid_at_4h is not None
            mean_liquid = float(np.mean(fraction_liquid_at_4h))
            self.assertGreater(
                mean_liquid, 0.0,
                msg=(
                    f"When final_time={final_time:.0f}s exceeds 14400s, "
                    f"query at 4h must show residual liquid, got mean={mean_liquid:.4f}"
                ),
            )

    def test_exact_final_schedule_time_returns_all_solid(self) -> None:
        engine = _run_pipeline_with_patched_memory(ci_model("block_2inch.stl"))
        _, final_time = engine.get_fraction_liquid_time_range()
        self.assertGreater(final_time, 0.0)
        fraction_liquid = engine.get_fraction_liquid_at_time(final_time)
        self.assertIsNotNone(fraction_liquid)
        assert fraction_liquid is not None
        np.testing.assert_allclose(
            fraction_liquid,
            np.zeros_like(fraction_liquid),
            atol=1e-6,
            err_msg=f"At exact final_time={final_time:.1f}s all voxels must be solid",
        )

    @unittest.skip("schedule_legacy is now opt-in; this test covered legacy-only uncapped behavior")
    def test_uncapped_chvorinov_horizon_permitted_for_schedule_legacy(self) -> None:
        pass

    def test_build_schedule_fraction_history_no_append_when_all_solid(self) -> None:
        start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        local = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        pre_solid = np.array([False, False, False])
        final_time = 300.0
        history, times = build_schedule_fraction_history(start, local, pre_solid, final_time)
        self.assertTrue(
            np.all(np.isclose(history[-1], 1.0, atol=1e-6, rtol=0)),
            msg="With final_time == max(end_times), no synthetic solid append should occur",
        )
        self.assertLessEqual(
            times[-1],
            final_time + 0.12,
            msg="Last time should be at or near final_time, not final_time + 0.1 with fake append",
        )

class FeederTimingSpreadTests(unittest.TestCase):
    def test_riser_and_riser_post_have_meaningful_spread(self) -> None:
        engine = _run_pipeline_with_patched_memory(ci_model("block_2inch.stl"))
        part_types = engine.mesh.voxel_part_types
        self.assertIsNotNone(part_types)
        assert part_types is not None

        post_mask = part_types == 3
        riser_mask = part_types == 1

        if not np.any(post_mask) and not np.any(riser_mask):
            self.skipTest("No RISER_POST or RISER voxels in this scenario")

        end_times = np.asarray(engine.solidification_end_times, dtype=np.float32)

        if np.any(post_mask):
            post_end_times = end_times[post_mask]
            post_spread = float(np.max(post_end_times) - np.min(post_end_times))
            self.assertGreater(
                post_spread,
                1.0,
                msg=(
                    f"RISER_POST voxels must have meaningful completion-time spread, "
                    f"got {post_spread:.2f}s"
                ),
            )

        if np.any(riser_mask):
            riser_end_times = end_times[riser_mask]
            riser_spread = float(np.max(riser_end_times) - np.min(riser_end_times))
            self.assertGreater(
                riser_spread,
                1.0,
                msg=(
                    f"RISER voxels must have meaningful completion-time spread, "
                    f"got {riser_spread:.2f}s"
                ),
            )

    def test_fraction_liquid_has_partial_visibility_frame_for_post_and_riser(self) -> None:
        engine = _run_pipeline_with_patched_memory(
            ci_model("block_2inch.stl"),
            analysis_options={"fraction_liquid": True, "shrinkage": True, "hotspot": True},
        )
        part_types = engine.mesh.voxel_part_types
        self.assertIsNotNone(part_types)
        assert part_types is not None

        post_mask = part_types == 3
        riser_mask = part_types == 1

        if not np.any(post_mask) and not np.any(riser_mask):
            self.skipTest("No RISER_POST or RISER voxels in this scenario")

        history = getattr(engine, "fraction_liquid_history", None)
        times = getattr(engine, "fraction_liquid_times", None)
        self.assertIsNotNone(history)
        self.assertIsNotNone(times)
        assert history is not None and times is not None

        snapshots = np.asarray(history, dtype=np.float32)
        times_arr = np.asarray(times, dtype=np.float32)

        if np.any(post_mask):
            post_has_partial = False
            for snap, t in zip(snapshots, times_arr):
                post_vals = snap[post_mask]
                has_solid = np.any(post_vals < 1.0)
                has_liquid = np.any(post_vals > 0.0)
                if has_solid and has_liquid:
                    post_has_partial = True
                    break
            self.assertTrue(
                post_has_partial,
                msg=(
                    f"RISER_POST must have at least one partial-visibility frame "
                    f"(some liquid, some solid), but fraction-liquid playback shows "
                    f"only all-liquid or all-solid for all {len(times_arr)} frames"
                ),
            )

        if np.any(riser_mask):
            riser_has_partial = False
            for snap, t in zip(snapshots, times_arr):
                riser_vals = snap[riser_mask]
                has_solid = np.any(riser_vals < 1.0)
                has_liquid = np.any(riser_vals > 0.0)
                if has_solid and has_liquid:
                    riser_has_partial = True
                    break
            self.assertTrue(
                riser_has_partial,
                msg=(
                    f"RISER must have at least one partial-visibility frame "
                    f"(some liquid, some solid), but fraction-liquid playback shows "
                    f"only all-liquid or all-solid for all {len(times_arr)} frames"
                ),
            )

if __name__ == "__main__":
    unittest.main()
