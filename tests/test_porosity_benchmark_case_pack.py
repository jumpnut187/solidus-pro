from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from tests.helpers.ci_models import ci_model
from tests.test_simulation_scenarios import (
    _cast_with_chill_manager,
    _cast_with_riser_post_manager,
    _cast_with_two_risers_manager,
    _run_pipeline_with_patched_memory,
    _single_cast_manager,
)

def _build_single_block_2in():
    return _single_cast_manager(ci_model("block_2inch.stl"))

CASE_BUILDERS: Dict[str, Callable[[], object]] = {
    "single_block_2in": _build_single_block_2in,
    "cast_with_riser_post": _cast_with_riser_post_manager,
    "cast_with_chill": _cast_with_chill_manager,
    "cast_with_two_risers": _cast_with_two_risers_manager,
}

def _collect_case_metrics(engine: object) -> Dict[str, float]:
    part_types = np.asarray(engine.mesh.voxel_part_types, dtype=np.int32)
    cast_mask = part_types == 0
    riser_mask = part_types == 1
    post_mask = part_types == 3

    shrink = np.asarray(engine.shrinkage_probability, dtype=np.float32)
    porosity = np.asarray(engine.porosity_probability, dtype=np.float32)
    solid_time = np.asarray(engine.solidification_times, dtype=np.float32)
    centers = np.asarray(engine.mesh.voxel_centers, dtype=np.float32)
    side_mask = cast_mask & np.isclose(centers[:, 0], 0.5)

    metrics: Dict[str, float] = {
        "cast_shrinkage_mean": float(np.mean(shrink[cast_mask])) if np.any(cast_mask) else 0.0,
        "cast_porosity_mean": float(np.mean(porosity[cast_mask])) if np.any(cast_mask) else 0.0,
        "riser_shrinkage_mean": float(np.mean(shrink[riser_mask])) if np.any(riser_mask) else 0.0,
        "riser_porosity_mean": float(np.mean(porosity[riser_mask])) if np.any(riser_mask) else 0.0,
        "post_porosity_mean": float(np.mean(porosity[post_mask])) if np.any(post_mask) else 0.0,
        "side_porosity_mean": float(np.mean(porosity[side_mask])) if np.any(side_mask) else 0.0,
        "side_solidification_time_mean": (
            float(np.mean(solid_time[side_mask])) if np.any(side_mask) else 0.0
        ),
    }
    metrics["post_porosity_to_riser_ratio"] = metrics["post_porosity_mean"] / max(
        metrics["riser_porosity_mean"],
        1e-6,
    )
    return metrics

class PorosityBenchmarkCasePackTests(unittest.TestCase):
    def _load_case_pack(self) -> dict:
        return json.loads(Path("tests/data/porosity_benchmark_case_pack.json").read_text(encoding="utf-8"))

    def _load_reference_targets(self) -> dict:
        return json.loads(Path("tests/data/porosity_reference_targets.json").read_text(encoding="utf-8"))

    def _run_case_metrics(self, case_pack: dict) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        for case in case_pack["cases"]:
            builder_name = case["builder"]
            self.assertIn(builder_name, CASE_BUILDERS)
            engine = _run_pipeline_with_patched_memory(
                CASE_BUILDERS[builder_name](),
                voxel_cut_cells=False,
                solidification_params={"mold": {"auto_mold": 0, "auto_cores": 0}},
            )
            metrics[case["id"]] = _collect_case_metrics(engine)
        return metrics

    def test_case_pack_acceptance_thresholds(self) -> None:
        case_pack = self._load_case_pack()
        metrics = self._run_case_metrics(case_pack)

        for case in case_pack["cases"]:
            case_id = case["id"]
            acceptance = case["acceptance"]
            case_metrics = metrics[case_id]
            compare_to = case.get("compare_to")
            baseline_metrics = metrics.get(compare_to) if compare_to else None

            if "cast_shrinkage_mean_max" in acceptance:
                self.assertLessEqual(
                    case_metrics["cast_shrinkage_mean"],
                    float(acceptance["cast_shrinkage_mean_max"]),
                    msg=f"{case_id}: cast_shrinkage_mean above threshold",
                )
            if "cast_porosity_mean_max" in acceptance:
                self.assertLessEqual(
                    case_metrics["cast_porosity_mean"],
                    float(acceptance["cast_porosity_mean_max"]),
                    msg=f"{case_id}: cast_porosity_mean above threshold",
                )
            if "cast_shrinkage_mean_ratio_max" in acceptance:
                self.assertIsNotNone(baseline_metrics)
                ratio = case_metrics["cast_shrinkage_mean"] / max(
                    baseline_metrics["cast_shrinkage_mean"],
                    1e-6,
                )
                self.assertLessEqual(
                    ratio,
                    float(acceptance["cast_shrinkage_mean_ratio_max"]),
                    msg=f"{case_id}: cast_shrinkage_mean ratio above threshold",
                )
            if "post_porosity_to_riser_ratio_max" in acceptance:
                self.assertLessEqual(
                    case_metrics["post_porosity_to_riser_ratio"],
                    float(acceptance["post_porosity_to_riser_ratio_max"]),
                    msg=f"{case_id}: post_porosity_to_riser_ratio above threshold",
                )
            if "side_porosity_ratio_max" in acceptance:
                self.assertIsNotNone(baseline_metrics)
                ratio = case_metrics["side_porosity_mean"] / max(
                    baseline_metrics["side_porosity_mean"],
                    1e-6,
                )
                self.assertLessEqual(
                    ratio,
                    float(acceptance["side_porosity_ratio_max"]),
                    msg=f"{case_id}: side_porosity ratio above threshold",
                )
            if "side_solidification_time_ratio_max" in acceptance:
                self.assertIsNotNone(baseline_metrics)
                ratio = case_metrics["side_solidification_time_mean"] / max(
                    baseline_metrics["side_solidification_time_mean"],
                    1e-6,
                )
                self.assertLessEqual(
                    ratio,
                    float(acceptance["side_solidification_time_ratio_max"]),
                    msg=f"{case_id}: side_solidification_time ratio above threshold",
                )

    def test_case_pack_reference_targets(self) -> None:
        case_pack = self._load_case_pack()
        reference_targets = self._load_reference_targets()
        metrics = self._run_case_metrics(case_pack)

        case_targets = reference_targets.get("cases", {})
        weighted_error_sum = 0.0
        weighted_metric_count = 0

        for case in case_pack["cases"]:
            case_id = case["id"]
            reference_id = case.get("reference_id")
            self.assertIsNotNone(reference_id, msg=f"{case_id}: missing reference_id")
            self.assertIn(reference_id, case_targets, msg=f"{case_id}: unknown reference_id {reference_id}")

            case_metrics = metrics[case_id]
            target_block = case_targets[reference_id]
            target_metrics = target_block["metrics"]
            tolerance = target_block["tolerance"]

            for metric_name, target_value in target_metrics.items():
                self.assertIn(metric_name, case_metrics, msg=f"{case_id}: metric {metric_name} missing")
                self.assertIn(metric_name, tolerance, msg=f"{case_id}: tolerance for {metric_name} missing")
                predicted = float(case_metrics[metric_name])
                target = float(target_value)
                tol = float(tolerance[metric_name])
                error = abs(predicted - target)
                with self.subTest(case=case_id, metric=metric_name):
                    self.assertLessEqual(
                        error,
                        tol,
                        msg=(
                            f"{case_id}: {metric_name} outside tolerance "
                            f"(pred={predicted:.6f}, target={target:.6f}, tol={tol:.6f})"
                        ),
                    )
                weighted_error_sum += error / max(abs(target), 1e-3)
                weighted_metric_count += 1

        mean_relative_error = weighted_error_sum / max(weighted_metric_count, 1)
        self.assertLessEqual(
            mean_relative_error,
            0.55,
            msg=f"Mean relative error too high: {mean_relative_error:.3f}",
        )

if __name__ == "__main__":
    unittest.main()
