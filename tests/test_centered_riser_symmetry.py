"""Regression test: centered plate+riser assembly must produce a symmetric
last-to-solidify pattern in the riser.

The CI models (tests/ci_models/) hold a 15.25 x 6.0 x 2.5 in plate (cast) with a
4 x 4 x 6 in cylindrical riser centred on top. This is geometrically
symmetric about (X=0, Y=0). Until the canonical neighbour-ordering fix,
the late-stage surviving liquid riser voxels drifted toward the (+X, +Y)
quadrant, producing a visibly off-centre column in the desktop app.

This test runs the full pipeline at voxel_size=0.30 in (the same config
that previously showed the bias) and asserts the centroid of the last
remaining riser liquid voxels is within a small tolerance of (0, 0).
"""
from __future__ import annotations

import os
import unittest
from typing import Tuple

import numpy as np

from tests.helpers.ci_models import CI_MODELS_DIR
from core.part_manager import build_part_manager
from core.simulation_pipeline import PipelineInputs, run_pipeline


TEST_MODEL_DIR = CI_MODELS_DIR
CAST_PATH = os.path.join(TEST_MODEL_DIR, "cast.stl")
RISER_PATH = os.path.join(TEST_MODEL_DIR, "riser.stl")


def _build_pipeline_inputs() -> PipelineInputs:
    return PipelineInputs(
        alloy_name="WCB",
        voxel_size=0.30,
        refinement_level=0,
        analysis_options={
            "shrinkage": True,
            "hotspot": True,
            "fraction_liquid": True,
            "solver_mode": "thermal_coupled",
        },
        show_progress=False,
    )


def _run_simulation() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.isfile(CAST_PATH) or not os.path.isfile(RISER_PATH):
        raise unittest.SkipTest(
            f"CI model STL files not present at {TEST_MODEL_DIR}"
        )

    pm = build_part_manager([
        {"filepath": CAST_PATH, "part_type": "cast"},
        {"filepath": RISER_PATH, "part_type": "riser"},
    ])
    inputs = _build_pipeline_inputs()
    result = run_pipeline(inputs, part_manager=pm)
    mesh = result.voxel_mesh
    engine = result.engine
    centers = np.asarray(mesh.voxel_centers)
    part_types = np.asarray(mesh.voxel_part_types)
    times = np.asarray(engine.solidification_times)
    return centers, part_types, times


class TestCenteredRiserSymmetry(unittest.TestCase):
    """Verify the centred-riser test case produces a symmetric solidification
    time distribution along X and Y.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.centers, cls.part_types, cls.times = _run_simulation()

    def _riser_voxels(self) -> Tuple[np.ndarray, np.ndarray]:
        riser = self.part_types == 1
        return self.centers[riser], self.times[riser]

    def test_geometry_is_symmetric(self) -> None:
        """Riser voxel positions must average to (0, 0) within one cell."""
        rc, _ = self._riser_voxels()
        self.assertAlmostEqual(float(rc[:, 0].mean()), 0.0, delta=0.05)
        self.assertAlmostEqual(float(rc[:, 1].mean()), 0.0, delta=0.05)
        self.assertAlmostEqual(float(rc[:, 0].std()), float(rc[:, 1].std()), delta=0.05)

    def test_last_solidifying_riser_voxels_are_centered(self) -> None:
        """The top-K longest-time riser voxels must be centred at (0, 0).

        Before the canonical neighbour-order fix this centroid was around
        (+0.6, +0.6), with zero voxels surviving in the (-X) or (-Y)
        quadrants. After the fix the bias drops below a single voxel
        cell (0.30 in) in every slice.
        """
        rc, rt = self._riser_voxels()
        # Take the top 5% longest-time riser voxels
        k = max(1, int(len(rc) * 0.05))
        order = np.argsort(rt)[::-1][:k]
        top_centers = rc[order]

        cx = float(top_centers[:, 0].mean())
        cy = float(top_centers[:, 1].mean())

        # Tolerance: a single voxel cell.
        self.assertAlmostEqual(cx, 0.0, delta=0.30,
                               msg=f"Last-N riser centroid X={cx:+.3f} should be near 0")
        self.assertAlmostEqual(cy, 0.0, delta=0.30,
                               msg=f"Last-N riser centroid Y={cy:+.3f} should be near 0")

    def test_quadrant_balance_of_last_solidifying_riser(self) -> None:
        """Both +X and -X must contribute to the long-time tail; same for Y.

        Before the fix, the last-N riser voxels were 100% in the
        (+X, +Y) quadrant. After the fix the contributions come from
        both sides.
        """
        rc, rt = self._riser_voxels()
        k = max(1, int(len(rc) * 0.05))
        order = np.argsort(rt)[::-1][:k]
        top = rc[order]

        x_pos = int((top[:, 0] > 0).sum())
        x_neg = int((top[:, 0] < 0).sum())
        y_pos = int((top[:, 1] > 0).sum())
        y_neg = int((top[:, 1] < 0).sum())

        # Both halves must contribute at least 10% of the long-time tail.
        # Anything stricter would still allow one-sided bias up to a
        # voxel of asymmetry.
        threshold = max(1, int(k * 0.10))
        self.assertGreater(x_pos, threshold,
                           f"Only {x_pos}/{k} last-riser voxels at +X")
        self.assertGreater(x_neg, threshold,
                           f"Only {x_neg}/{k} last-riser voxels at -X")
        self.assertGreater(y_pos, threshold,
                           f"Only {y_pos}/{k} last-riser voxels at +Y")
        self.assertGreater(y_neg, threshold,
                           f"Only {y_neg}/{k} last-riser voxels at -Y")


if __name__ == "__main__":
    unittest.main()