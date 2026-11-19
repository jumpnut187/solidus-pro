from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from core.solidification_params import SolidificationParameters
from core.solidification_loop import run_time_integration


class _DummyLogger:
    def info(self, _message: str, context=None) -> None:
        self.context = context


class _DummyEngine:
    def __init__(self) -> None:
        self.alloy = SimpleNamespace(
            thermal_diffusivity=45 / (7200 * 600),
            liquidus_temp=1510.0,
            solidus_temp=1449.0,
            name="WCB",
        )
        self.mesh = SimpleNamespace(
            voxel_states=np.array([1.0, 0.992], dtype=np.float32),
            voxel_neighbors=np.array([[-1], [0]], dtype=np.int32),
            voxel_moduli=np.array([0.0, 1.0], dtype=np.float32),
            voxel_centers=np.zeros((2, 3), dtype=np.float32),
        )
        self.analysis_options = SimpleNamespace(
            fraction_liquid=True,
            solidification_params=SolidificationParameters(),
        )
        self.fraction_liquid_history = []
        self.fraction_liquid_times = []
        self.solidification_times = np.zeros(2, dtype=np.float32)
        self.max_time = 1.0
        self.time_step = 1.0
        self.time = 0.0
        self._use_incremental = False
        self._incremental_engine = None
        self._incremental_engine_cls = None
        self._use_gpu = False
        self._gpu_device_name = ""
        self._air_gap_slowdown = np.ones(2, dtype=np.float32)
        self._chill_centers = None
        self._logger = _DummyLogger()

    def _get_normalized_heights(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    def _update_solidification_with_part_types(
        self,
        chill_boost: np.ndarray,
        effective_slowdown: np.ndarray,
        material_factor: float,
    ) -> np.ndarray:
        del chill_boost, effective_slowdown, material_factor
        return np.array([1.0, 0.996], dtype=np.float32)


class FractionLiquidFinalSnapshotTests(unittest.TestCase):
    def test_appends_fully_solid_snapshot_when_run_ends_with_residual_liquid(self) -> None:
        engine = _DummyEngine()

        run_time_integration(
            engine,
            chill_boost=np.ones(2, dtype=np.float32),
            riser_slowdown=np.ones(2, dtype=np.float32),
            show_progress=False,
            progress_callback=None,
        )

        self.assertGreaterEqual(len(engine.fraction_liquid_history), 3)
        np.testing.assert_allclose(engine.fraction_liquid_history[-1], np.ones(2, dtype=np.float32))
        self.assertAlmostEqual(engine.fraction_liquid_times[-1], engine.time + 0.1)


if __name__ == "__main__":
    unittest.main()
