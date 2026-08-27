"""Regression tests for sand burn-in risk, in particular chill proximity.

``_calculate_sand_burn_in`` reduces burn-in risk near chills, because metal
against a chill solidifies before it can penetrate the sand. That reduction had
never actually executed: the block read ``chill_burn_in_radius`` eight lines
before assigning it, and a broad ``except Exception`` downgraded the resulting
NameError to a printed warning.

The spatial assertion below is what distinguishes the two states. With the
reduction disabled the chill still perturbs burn-in indirectly through
solidification time, so near-chill risk is only ~7 % below the far side
(ratio ~0.93). With it active the reduction is ~35 % (ratio ~0.65).
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.test_simulation_scenarios import (
    _cast_with_chill_manager,
    _run_pipeline_with_patched_memory,
)


class SandBurnInChillProximityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = _run_pipeline_with_patched_memory(
            _cast_with_chill_manager(),
            extra_analysis_options={"sand_burn_in": True},
        )

    def _cast_surface_split(self):
        """Split cast surface voxels into near-chill and far-side halves.

        ``_cast_with_chill_manager`` puts the chill at +1.5 on X, so the
        higher-X half of the cast surface is the chill-facing side.
        """
        engine = self.engine
        part_types = np.asarray(engine.mesh.voxel_part_types, dtype=np.int32)
        surface = np.asarray(engine.mesh.surface_voxels, dtype=bool)
        centers = np.asarray(engine.mesh.voxel_centers, dtype=np.float32)

        cast_surface = (part_types == 0) & surface
        self.assertTrue(np.any(cast_surface), "no cast surface voxels to compare")

        x = centers[:, 0]
        midpoint = float(np.median(x[cast_surface]))
        near = cast_surface & (x >= midpoint)
        far = cast_surface & (x < midpoint)
        self.assertTrue(np.any(near) and np.any(far), "split produced an empty half")
        return near, far

    def test_sand_burn_in_risk_is_produced(self) -> None:
        risk = getattr(self.engine, "sand_burn_in_risk", None)
        self.assertIsNotNone(risk, "sand_burn_in analysis produced no risk field")
        self.assertEqual(len(np.asarray(risk)), len(self.engine.mesh.voxel_centers))

    def test_chill_proximity_reduces_burn_in_risk(self) -> None:
        """Surfaces facing the chill must carry materially less burn-in risk.

        Guards the use-before-assignment of chill_burn_in_radius: with that bug
        present this ratio is ~0.93 and the assertion fails.
        """
        near, far = self._cast_surface_split()
        risk = np.asarray(self.engine.sand_burn_in_risk, dtype=np.float32)

        near_mean = float(np.mean(risk[near]))
        far_mean = float(np.mean(risk[far]))
        self.assertGreater(far_mean, 0.0, "far-side burn-in risk is zero; nothing to compare")

        self.assertLess(
            near_mean / far_mean,
            0.85,
            msg=(
                "chill proximity did not reduce sand burn-in risk "
                f"(near={near_mean:.5f}, far={far_mean:.5f}, "
                f"ratio={near_mean / far_mean:.3f}); the chill-proximity block "
                "is likely raising and being swallowed again"
            ),
        )

    def test_chill_proximity_block_does_not_swallow_programming_errors(self) -> None:
        """A NameError in the chill block must propagate, not become a warning."""
        import inspect

        import core.solidification as solidification

        # Checked structurally rather than by provoking a real failure: the
        # point is that the broad handler must not sit in front of NameError.
        src = inspect.getsource(
            solidification.SolidificationEngine._calculate_sand_burn_in
        )
        self.assertIn(
            "except NameError:",
            src,
            "the chill-proximity handler no longer re-raises NameError, so a "
            "use-before-assignment bug there would be silently swallowed again",
        )


if __name__ == "__main__":
    unittest.main()
