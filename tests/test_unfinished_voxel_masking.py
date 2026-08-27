from __future__ import annotations

import unittest

import numpy as np

from core.acceleration import compute_isolation_score_parallel
from tests.test_simulation_scenarios import (
    _run_pipeline_with_patched_memory,
    _single_cast_manager,
)


def _build_engine_with_nan_voxels() -> object:
    """
    Run the standard tiny cast pipeline and mark interior cast voxels as
    unfinished by writing np.nan into ``solidification_times`` (the thermal
    loop's representation of "still liquid when the max_time cap was hit").
    """
    engine = _run_pipeline_with_patched_memory(
        _single_cast_manager("sample_stl_files/small_block.stl")
    )
    times = np.asarray(engine.solidification_times, dtype=np.float32).copy()
    part_types = np.asarray(engine.mesh.voxel_part_types, dtype=np.int32)
    cast_indices = np.flatnonzero(part_types == 0)
    surface_distances = np.asarray(engine.surface_distances, dtype=np.float32)

    interior = cast_indices[surface_distances[cast_indices] > 0.0]
    if len(interior) == 0:
        interior = cast_indices
    # Prefer the deepest (last-to-freeze) interior voxels, at most two.
    interior = interior[np.argsort(surface_distances[interior])[-2:]]
    times[interior] = np.nan
    engine.solidification_times = times
    engine._unfinished_voxels = np.asarray(interior, dtype=np.int64)
    return engine


def _sanitized_times(engine: object) -> np.ndarray:
    """Reference sanitize: NaN -> max_time (last-to-freeze substitution)."""
    times = np.asarray(engine.solidification_times, dtype=np.float32)
    return np.where(np.isnan(times), float(engine.max_time), times)


class UnfinishedVoxelMaskingTests(unittest.TestCase):
    def test_isolation_score_masks_nan_as_last_to_freeze(self) -> None:
        engine = _build_engine_with_nan_voxels()
        nan_mask = np.isnan(engine.solidification_times)
        self.assertTrue(np.any(nan_mask))

        result = engine._calculate_isolation_score()

        self.assertTrue(np.all(np.isfinite(result)))

        # Reference: compute with NaN replaced by max_time first.
        times = _sanitized_times(engine)
        neighbors = np.ascontiguousarray(engine.mesh.voxel_neighbors, dtype=np.int32)
        reference = compute_isolation_score_parallel(
            np.ascontiguousarray(times, dtype=np.float32),
            neighbors,
            np.zeros(len(times), dtype=np.int32),
        )
        surface_distances = engine._get_cast_surface_distances()
        surface_factor = surface_distances / np.max(surface_distances + 0.1)
        reference = reference * (0.5 + 0.5 * surface_factor)
        reference = np.clip(reference, 0.0, 1.0).astype(np.float32)

        np.testing.assert_allclose(result, reference)
        # Unfinished voxels solidify last -> surrounded by earlier-solid neighbors.
        self.assertTrue(np.all(result[nan_mask] > 0.0))

    def test_shrinkage_base_risk_time_factor_masks_nan(self) -> None:
        engine = _build_engine_with_nan_voxels()
        valid_indices = engine._unfinished_voxels.astype(np.int64)
        n_voxels = len(engine.mesh.voxel_centers)
        normalized_heights = engine._get_normalized_heights()
        local_section_size = engine._estimate_local_section_size()
        riser_mask = np.zeros(n_voxels, dtype=bool)
        riser_post_mask = np.zeros(n_voxels, dtype=bool)

        (
            _base_risk,
            _heights,
            _moduli,
            _thicknesses,
            _times,
            _is_riser,
            _is_riser_post,
            _height_factor,
            _interior_factor,
            _thickness_factor,
            time_factor,
        ) = engine._compute_shrinkage_base_risk(
            valid_indices,
            normalized_heights,
            local_section_size,
            riser_mask,
            riser_post_mask,
        )

        # Reference time_factor: NaN replaced with max_time first.
        times = _sanitized_times(engine)[valid_indices]
        reference_factor = np.where(times > 0, np.minimum(times / 800.0, 1.0), 0.0)

        np.testing.assert_allclose(time_factor, reference_factor)
        self.assertTrue(np.all(np.isfinite(time_factor)))

    def test_hotspots_mask_nan_as_last_to_freeze(self) -> None:
        engine = _build_engine_with_nan_voxels()
        engine._calculate_hotspots()

        self.assertTrue(np.all(np.isfinite(engine.hotspot_intensity)))
        nan_mask = np.isnan(engine.solidification_times)
        # Unfinished voxels are last-to-freeze -> hot spot candidates.
        self.assertTrue(np.all(engine.hotspot_intensity[nan_mask] > 0.0))

    def test_niyama_schedule_legacy_masks_nan(self) -> None:
        engine = _build_engine_with_nan_voxels()
        # Force the schedule_legacy approximate branch regardless of solver mode.
        engine._thermal_niyama_values_at_eval = None
        engine._thermal_niyama_sampled = None
        engine._calculate_niyama()

        self.assertTrue(np.all(np.isfinite(engine.cooling_rates)))
        self.assertTrue(np.all(np.isfinite(engine.thermal_gradients)))
        self.assertTrue(np.all(np.isfinite(engine.niyama_values)))
        self.assertTrue(np.all(np.isfinite(engine.niyama_severity)))

        nan_mask = np.isnan(engine.solidification_times)
        temp_range = engine.alloy.liquidus_temp - engine.alloy.solidus_temp
        expected_rate = temp_range * (1.0 / float(engine.max_time)) + 0.01
        np.testing.assert_allclose(engine.cooling_rates[nan_mask], expected_rate, rtol=1e-5)

    def test_statistics_exclude_unfinished_from_mean_time(self) -> None:
        engine = _build_engine_with_nan_voxels()
        stats = engine._generate_statistics()

        self.assertIn("unfinished_voxel_count", stats)
        part_types = np.asarray(engine.mesh.voxel_part_types, dtype=np.int32)
        cast_mask = part_types == 0
        nan_mask = np.isnan(engine.solidification_times)
        expected_unfinished = int(np.sum(nan_mask & cast_mask))
        self.assertEqual(stats["unfinished_voxel_count"], expected_unfinished)

        times = np.asarray(engine.solidification_times, dtype=np.float32)
        finished = np.isfinite(times) & (times > 0)
        self.assertAlmostEqual(
            stats["mean_solidification_time"],
            float(np.mean(times[finished])),
            places=6,
        )
        self.assertAlmostEqual(
            stats["max_solidification_time"],
            float(np.max(times[finished])),
            places=6,
        )
        self.assertTrue(np.isfinite(stats["mean_solidification_time"]))
        self.assertTrue(np.isfinite(stats["max_solidification_time"]))

    def test_porosity_and_defect_volume_remain_finite(self) -> None:
        engine = _build_engine_with_nan_voxels()
        engine._thermal_niyama_values_at_eval = None
        engine._thermal_niyama_sampled = None
        engine._calculate_niyama()
        engine._calculate_porosity()

        self.assertTrue(np.all(np.isfinite(engine.shrinkage_risk)))
        self.assertTrue(np.all(np.isfinite(engine.porosity_risk)))
        self.assertTrue(np.all(np.isfinite(engine.porosity_probability)))

        estimate = engine.get_defect_volume_estimate()
        for key, value in estimate.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.assertTrue(np.isfinite(value), msg=f"{key} is not finite: {value}")


if __name__ == "__main__":
    unittest.main()
