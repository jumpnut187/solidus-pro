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
        import inspect

        import core.solidification as solidification

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
