"""Pipe in metal no riser feeds, and the per-riser sleeve choice.

Uses the CI model plate (tests/ci_models/cast.stl, 15.25 x 6 x 2.5 in) with
the 4 in diameter, 6 in tall cylinder on top (z 2.5 to 8.5), WCB. The pipe
cases run at voxel_size=0.30 in; the sleeve comparisons only need the sign of
the difference and run at 0.50 in to keep CI fast.

- The cylinder typed CAST: nothing is a riser, but the liquid still drains by
  gravity, so a funnel forms at the top of the last-to-freeze region (deeper on
  the axis than at the wall) and is reported apart from the riser report.
- The cylinder typed RISER with ``riser_sleeve`` "none" (bare sand) against the
  default "insulating": the bare riser freezes sooner and its pipe reaches
  further down, into the casting.
- A riser run on the default insulating sleeve, which leaks into the sand,
  reproduces the numbers the code gave before unrisered pools existed.
- The pipe shape follows the sleeve. In the insulating sleeve a skin grows on
  the riser wall and the pipe is a funnel on the axis. With ``riser_sleeve``
  "adiabatic", which passes no heat to the sand, the riser freezes after the
  plate, holds the whole pipe, and the pipe is a level drop across the bore.
"""
from __future__ import annotations

import os
import unittest
from typing import Any, Dict, Optional

import numpy as np

from tests.helpers.ci_models import CI_MODELS_DIR
from core.part_manager import build_part_manager
from core.simulation_pipeline import PipelineInputs, run_pipeline


TEST_MODEL_DIR = CI_MODELS_DIR
CAST_PATH = os.path.join(TEST_MODEL_DIR, "cast.stl")
RISER_PATH = os.path.join(TEST_MODEL_DIR, "riser.stl")

CYLINDER_BOTTOM_Z = 2.5
CYLINDER_TOP_Z = 8.5
CYLINDER_RADIUS = 2.0

#: ``riser_pipe_report[0]`` of the riser run on the insulating (leaking) sleeve,
#: taken on the code before unrisered pools were added (voxel_size 0.30, auto mold on).
OLD_RISER_REPORT = {
    "demand_in3": 14.216858530080579,
    "pipe_in3": 14.215963226140072,
    "pipe_in_riser_in3": 14.153116177504351,
    "pipe_in_casting_in3": 0.06284704863571945,
    "pipe_bottom_above_riser_base_in": 0.0,
}

BOUNDARY_ONLY = {"mold": {"auto_mold": 0, "auto_cores": 0}}

PIPE_VOXEL_SIZE = 0.30
SLEEVE_VOXEL_SIZE = 0.50


_RUNS: Dict[str, Dict[str, Any]] = {}


def _run(
    cylinder_type: str,
    riser_sleeve: Optional[str] = None,
    solidification_params: Optional[Dict[str, Any]] = None,
    voxel_size: float = PIPE_VOXEL_SIZE,
    open_top: bool = True,
) -> Dict[str, Any]:
    """One pipeline run, kept for the module so each case is simulated once."""
    key = repr((cylinder_type, riser_sleeve, solidification_params, voxel_size, open_top))
    if key not in _RUNS:
        _RUNS[key] = _simulate(cylinder_type, riser_sleeve, solidification_params, voxel_size, open_top)
    return _RUNS[key]


def _simulate(
    cylinder_type: str,
    riser_sleeve: Optional[str],
    solidification_params: Optional[Dict[str, Any]],
    voxel_size: float,
    open_top: bool = True,
) -> Dict[str, Any]:
    if not os.path.isfile(CAST_PATH) or not os.path.isfile(RISER_PATH):
        raise unittest.SkipTest(f"CI model STL files not present at {TEST_MODEL_DIR}")
    cylinder: Dict[str, Any] = {
        "filepath": RISER_PATH,
        "part_type": cylinder_type,
        "is_open_top": open_top,
    }
    if riser_sleeve is not None:
        cylinder["riser_sleeve"] = riser_sleeve
    pm = build_part_manager([{"filepath": CAST_PATH, "part_type": "cast"}, cylinder])
    inputs = PipelineInputs(
        alloy_name="WCB",
        voxel_size=voxel_size,
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
    """No sleeve given means insulating (tests/test_riser_sleeve.py), so the
    default run stands in for the insulated riser."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.insulating = _run("riser", voxel_size=SLEEVE_VOXEL_SIZE)
        cls.bare = _run("riser", "none", voxel_size=SLEEVE_VOXEL_SIZE)

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

    def test_no_heat_loss_riser_freezes_after_the_plate_and_holds_the_pipe(self) -> None:
        run = _run("riser", "adiabatic", voxel_size=SLEEVE_VOXEL_SIZE)
        riser = run["part_types"] == 1
        self.assertGreater(float(run["times"][riser].max()), float(run["times"][~riser].max()))
        self.assertLess(run["riser_report"][0]["pipe_in_casting_in3"], 0.01)
        # The insulating sleeve leaks, so the same riser freezes sooner.
        self.assertLess(float(self.insulating["times"][riser].max()), float(run["times"][riser].max()))


def _drained_radius_by_layer(run: Dict[str, Any]) -> Dict[float, float]:
    """Equivalent radius of the drained metal in each riser layer, top first (voxel-count weighted)."""
    riser = run["part_types"] == 1
    z = np.round(run["centers"][:, 2], 3)
    out: Dict[float, float] = {}
    for level in sorted(np.unique(z[riser]).tolist(), reverse=True):
        layer = riser & (z == level)
        out[level] = CYLINDER_RADIUS * float(np.sqrt(run["pipe"][layer].mean()))
    return out


class PipeShapeFollowsTheSleeveTests(unittest.TestCase):
    """A funnel in the insulating sleeve, a level drop in the no-heat-loss sleeve.

    Closed top, so no crust forms and the pipe opens at the top of the riser.
    """

    def test_insulating_sleeve_gives_a_funnel_on_the_axis(self) -> None:
        run = _run("riser", voxel_size=SLEEVE_VOXEL_SIZE, open_top=False)
        radii = [r for r in _drained_radius_by_layer(run).values() if r > 0.1]
        self.assertGreaterEqual(len(radii), 5)
        # The full bore at the top, narrowing with depth.
        self.assertGreater(radii[0], 0.9 * CYLINDER_RADIUS)
        self.assertLess(radii[3], 0.5 * radii[0])
        self.assertLess(max(radii[3:]), 0.5 * radii[0])
        # Below the mouth the drained metal sits on the axis inside a sound wall.
        riser = run["part_types"] == 1
        below_mouth = riser & (run["centers"][:, 2] < CYLINDER_TOP_Z - 1.5)
        piped = below_mouth & (run["pipe"] > 0.5)
        sound = below_mouth & (run["pipe"] <= 0.5)
        self.assertGreater(int(piped.sum()), 5)
        radius = _radius(run["centers"])
        self.assertLess(float(radius[piped].mean()), 0.5 * float(radius[sound].mean()))
        self.assertLess(float(radius[piped].max()), CYLINDER_RADIUS - 0.5)
        self.assertLess(run["riser_report"][0]["pipe_in_casting_in3"], 0.01)

    def test_no_heat_loss_sleeve_gives_a_level_drop(self) -> None:
        run = _run("riser", "adiabatic", voxel_size=SLEEVE_VOXEL_SIZE, open_top=False)
        radii = [r for r in _drained_radius_by_layer(run).values() if r > 0.0]
        self.assertGreater(len(radii), 1)
        # Every drained layer but the last is drained across the whole bore.
        for r in radii[:-1]:
            self.assertGreater(r, 0.95 * CYLINDER_RADIUS)
        # Shallower than the funnel the insulating sleeve gives the same riser.
        funnel = _run("riser", voxel_size=SLEEVE_VOXEL_SIZE, open_top=False)
        everything = np.ones(len(run["pipe"]), dtype=bool)
        self.assertGreater(_lowest_piped_z(run, everything), _lowest_piped_z(funnel, everything) + 1.0)


class RiserSleeveBoundaryPathTests(unittest.TestCase):
    """Without an explicit mold the bare riser takes the casting's interface HTC."""

    def test_bare_riser_freezes_sooner_on_the_sand_boundary(self) -> None:
        insulating = _run("riser", None, BOUNDARY_ONLY, SLEEVE_VOXEL_SIZE)
        bare = _run("riser", "none", BOUNDARY_ONLY, SLEEVE_VOXEL_SIZE)
        riser = insulating["part_types"] == 1
        self.assertLess(float(bare["times"][riser].max()), 0.97 * float(insulating["times"][riser].max()))


if __name__ == "__main__":
    unittest.main()
