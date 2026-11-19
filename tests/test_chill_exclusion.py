"""
Regression tests: chills must never show shrinkage or fraction liquid.

Chills are external cooling devices, pre-solid by definition. The thermal
model pins their fraction solid to 1.0 and their solidification times to 0,
but two post-processing edge cases could leak through:
1. ``_calculate_shrinkage`` adds feeding-blocked voxels to the shrinkage
   candidate set without excluding chills (chills flagged blocked by the
   feeding tracer received the 0.35+ blocked floor).
2. ``get_fraction_liquid_at_time`` trusted the chill bookkeeping
   (local_solidification_time == 0) instead of hard-masking chill voxels.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.test_porosity_benchmark_case_pack import (
    _run_pipeline_with_patched_memory,
    _cast_with_chill_manager,
)


class ChillExclusionTests(unittest.TestCase):
    def _engine(self) -> object:
        return _run_pipeline_with_patched_memory(_cast_with_chill_manager())

    def test_chill_voxels_have_zero_shrinkage_and_porosity(self) -> None:
        engine = self._engine()
        part_types = np.asarray(engine.mesh.voxel_part_types, dtype=np.int32)
        chill_mask = part_types == 2
        self.assertGreater(int(chill_mask.sum()), 0, "fixture must contain chill voxels")

        shrink = np.asarray(engine.shrinkage_probability, dtype=float)
        poro = np.asarray(engine.porosity_probability, dtype=float)
        self.assertEqual(float(shrink[chill_mask].max()), 0.0)
        self.assertEqual(float(poro[chill_mask].max()), 0.0)
        # The raw (pre-scalar) risk arrays must be clean too.
        raw_shrink = np.asarray(engine.shrinkage_risk, dtype=float)
        self.assertEqual(float(raw_shrink[chill_mask].max()), 0.0)

    def test_chill_voxels_have_zero_fraction_liquid_through_ui_path(self) -> None:
        engine = self._engine()
        part_types = np.asarray(engine.mesh.voxel_part_types, dtype=np.int32)
        chill_mask = part_types == 2
        _, max_time = engine.get_fraction_liquid_time_range()

        for t in (0.0, max_time * 0.25, max_time * 0.5, max_time * 0.9, max_time):
            fraction_liquid = engine.get_fraction_liquid_at_time(float(t))
            self.assertIsNotNone(fraction_liquid)
            fl = np.asarray(fraction_liquid, dtype=float)
            self.assertEqual(
                float(fl[chill_mask].max()),
                0.0,
                msg=f"chill fraction liquid must be 0 at t={t:.1f}s",
            )

    def test_chill_voxels_are_solid_in_every_history_snapshot(self) -> None:
        engine = self._engine()
        part_types = np.asarray(engine.mesh.voxel_part_types, dtype=np.int32)
        chill_mask = part_types == 2
        history = engine.fraction_liquid_history
        self.assertGreaterEqual(len(history), 2)
        for frame in history:
            fs = np.asarray(frame, dtype=float)
            self.assertTrue(np.all(fs[chill_mask] >= 0.999), "chill fs must stay 1.0 in every frame")

    def test_fraction_liquid_masking_survives_broken_chill_bookkeeping(self) -> None:
        """Even if a chill voxel's schedule bookkeeping is corrupted (e.g. a
        positive local solidification time from an edge case), the UI query
        must still report zero fraction liquid for it."""
        engine = self._engine()
        part_types = np.asarray(engine.mesh.voxel_part_types, dtype=np.int32)
        chill_mask = part_types == 2
        chill_idx = np.flatnonzero(chill_mask)[:2]

        # Corrupt the bookkeeping: pretend the chill voxels have a normal
        # solidification schedule (start 5s, local 30s, end 35s).
        engine.solidification_start_times[chill_idx] = 5.0
        engine.local_solidification_time[chill_idx] = 30.0
        engine.solidification_end_times[chill_idx] = 35.0

        fraction_liquid = engine.get_fraction_liquid_at_time(20.0)
        self.assertIsNotNone(fraction_liquid)
        fl = np.asarray(fraction_liquid, dtype=float)
        self.assertEqual(float(fl[chill_idx].max()), 0.0, "chill masking must be unconditional")


if __name__ == "__main__":
    unittest.main()
