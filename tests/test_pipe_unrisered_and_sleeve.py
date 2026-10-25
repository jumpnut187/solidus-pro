"""Pipe in metal no riser feeds, and the per-riser sleeve choice.

Uses the Test_Model plate (15.25 x 6 x 2.5 in) with the 4 in diameter, 6 in
tall cylinder on top (z 2.5 to 8.5), WCB, voxel_size=0.30 in.

- The cylinder typed CAST: nothing is a riser, but the liquid still drains by
  gravity, so a funnel forms at the top of the last-to-freeze region (deeper on
  the axis than at the wall) and is reported apart from the riser report.
- The cylinder typed RISER with ``riser_sleeve`` "none" (bare sand) against the
  default "insulating": the bare riser freezes sooner and its pipe reaches
  further down, into the casting.
- A riser run with default settings reproduces the numbers the code gave before
  unrisered pools existed.
"""
from __future__ import annotations

import os
import unittest
from typing import Any, Dict, Optional

import numpy as np

from core.part_manager import build_part_manager
from core.simulation_pipeline import PipelineInputs, run_pipeline


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TEST_MODEL_DIR = os.path.join(REPO_ROOT, "Test_Model")
CAST_PATH = os.path.join(TEST_MODEL_DIR, "cast.stl")
RISER_PATH = os.path.join(TEST_MODEL_DIR, "riser.stl")

CYLINDER_BOTTOM_Z = 2.5
CYLINDER_TOP_Z = 8.5
CYLINDER_RADIUS = 2.0

#: ``riser_pipe_report[0]`` of the riser run with default settings, taken on the
#: code before unrisered pools were added (voxel_size 0.30, auto mold on).
OLD_RISER_REPORT = {
    "demand_in3": 14.216858530080579,
    "pipe_in3": 14.215963226140072,
    "pipe_in_riser_in3": 14.153116177504351,
    "pipe_in_casting_in3": 0.06284704863571945,
    "pipe_bottom_above_riser_base_in": 0.0,
}

BOUNDARY_ONLY = {"mold": {"auto_mold": 0, "auto_cores": 0}}


_RUNS: Dict[str, Dict[str, Any]] = {}


def _run(
    cylinder_type: str,
    riser_sleeve: Optional[str] = None,
    solidification_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One pipeline run, kept for the module so each case is simulated once."""
    key = repr((cylinder_type, riser_sleeve, solidification_params))
    if key not in _RUNS:
        _RUNS[key] = _simulate(cylinder_type, riser_sleeve, solidification_params)
    return _RUNS[key]


def _simulate(
    cylinder_type: str,
    riser_sleeve: Optional[str],
    solidification_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not os.path.isfile(CAST_PATH) or not os.path.isfile(RISER_PATH):
        raise unittest.SkipTest(f"Test_Model STL files not present at {TEST_MODEL_DIR}")
    cylinder: Dict[str, Any] = {
        "filepath": RISER_PATH,
        "part_type": cylinder_type,
        "is_open_top": True,
    }
    if riser_sleeve is not None:
        cylinder["riser_sleeve"] = riser_sleeve
    pm = build_part_manager([{"filepath": CAST_PATH, "part_type": "cast"}, cylinder])
    inputs = PipelineInputs(
        alloy_name="WCB",
        voxel_size=0.30,
        refinement_level=0,
        solidification_params=solidification_params or {},
        analysis_options={
            "shrinkage": True,
            "hotspot": True,
            "fraction_liquid": True,
            "solver_mode": "thermal_coupled",
        },
        show_progress=False,
    )
    result = run_pipeline(inputs, part_manager=pm)
    engine = result.engine
    centers = np.asarray(result.voxel_mesh.voxel_centers, dtype=np.float64)
    return {
        "centers": centers,
        "part_types": np.asarray(result.voxel_mesh.voxel_part_types),
        "times": np.asarray(engine.solidification_times, dtype=np.float64),
        "pipe": np.asarray(engine.riser_pipe_fraction, dtype=np.float64),
        "risk": np.asarray(engine.shrinkage_risk, dtype=np.float64),
        "riser_report": list(engine.riser_pipe_report),
        "casting_report": list(engine.casting_pipe_report),
        "summary": dict(result.result_data),
    }


def _radius(centers: np.ndarray) -> np.ndarray:
    return np.hypot(centers[:, 0], centers[:, 1])


def _in_cylinder(centers: np.ndarray) -> np.ndarray:
    return (_radius(centers) < CYLINDER_RADIUS) & (centers[:, 2] > CYLINDER_BOTTOM_Z)


def _lowest_piped_z(run: Dict[str, Any], select: np.ndarray) -> float:
    piped = select & (run["pipe"] > 0.5)
    return float(run["centers"][piped, 2].min()) if np.any(piped) else float("inf")


class UnriseredPipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _run("cast")

    def test_pipe_forms_with_no_riser_and_is_reported_apart(self) -> None:
        self.assertEqual(self.sim["riser_report"], [])
        self.assertEqual(len(self.sim["casting_report"]), 1)
        pipe_volume = sum(pool["pipe_in3"] for pool in self.sim["casting_report"])
        self.assertGreater(pipe_volume, 1.0)
        self.assertAlmostEqual(self.sim["summary"]["casting_pipe_volume_in3"], pipe_volume, places=9)
        self.assertEqual(self.sim["summary"]["casting_pipe_pool_count"], 1)
        self.assertNotIn("riser_pipe_volume_in3", self.sim["summary"])

    def test_pipe_lies_in_the_cylinder_and_reaches_the_risk_field(self) -> None:
        centers = self.sim["centers"]
        piped = self.sim["pipe"] > 0.5
        self.assertTrue(np.any(piped))
        self.assertTrue(np.all(centers[piped, 2] > CYLINDER_BOTTOM_Z))
        self.assertTrue(np.all(self.sim["risk"][piped] >= self.sim["pipe"][piped] - 1e-6))

    def test_funnel_is_deeper_on_the_axis_than_at_the_wall(self) -> None:
        centers = self.sim["centers"]
        radius = _radius(centers)
        inside = _in_cylinder(centers)
        top = float(centers[inside, 2].max())
        axis = _lowest_piped_z(self.sim, inside & (radius < 0.45))
        wall = _lowest_piped_z(self.sim, inside & (radius > 1.5))
        axis_depth = top - axis if np.isfinite(axis) else 0.0
        wall_depth = top - wall if np.isfinite(wall) else 0.0
        self.assertGreater(axis_depth, 2.0)
        self.assertGreater(axis_depth, wall_depth + 1.0)


class RiserDefaultUnchangedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _run("riser")

    def test_riser_pipe_report_matches_the_code_before_unrisered_pools(self) -> None:
        self.assertEqual(len(self.sim["riser_report"]), 1)
        report = self.sim["riser_report"][0]
        for key, expected in OLD_RISER_REPORT.items():
            self.assertAlmostEqual(report[key], expected, delta=0.02, msg=key)

    def test_riser_run_has_no_unrisered_pool(self) -> None:
        self.assertEqual(self.sim["casting_report"], [])
        self.assertNotIn("casting_pipe_volume_in3", self.sim["summary"])


class RiserSleeveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.insulating = _run("riser", "insulating")
        cls.bare = _run("riser", "none")

    def _riser_freeze_time(self, run: Dict[str, Any]) -> float:
        return float(run["times"][run["part_types"] == 1].max())

    def test_bare_riser_freezes_sooner_and_pipes_deeper(self) -> None:
        self.assertLess(self._riser_freeze_time(self.bare), 0.95 * self._riser_freeze_time(self.insulating))
        self.assertGreater(
            self.bare["riser_report"][0]["pipe_in_casting_in3"],
            self.insulating["riser_report"][0]["pipe_in_casting_in3"],
        )
        everything = np.ones(len(self.bare["pipe"]), dtype=bool)
        self.assertLess(
            _lowest_piped_z(self.bare, everything), _lowest_piped_z(self.insulating, everything)
        )

    def test_default_sleeve_is_insulating(self) -> None:
        default = _run("riser")
        np.testing.assert_array_equal(default["times"], self.insulating["times"])
        self.assertEqual(default["riser_report"], self.insulating["riser_report"])


class RiserSleeveBoundaryPathTests(unittest.TestCase):
    """Without an explicit mold the bare riser takes the casting's interface HTC."""

    def test_bare_riser_freezes_sooner_on_the_sand_boundary(self) -> None:
        insulating = _run("riser", "insulating", BOUNDARY_ONLY)
        bare = _run("riser", "none", BOUNDARY_ONLY)
        riser = insulating["part_types"] == 1
        self.assertLess(float(bare["times"][riser].max()), 0.97 * float(insulating["times"][riser].max()))


if __name__ == "__main__":
    unittest.main()
