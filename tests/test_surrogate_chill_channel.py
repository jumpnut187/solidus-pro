from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from core.materials import get_alloy
from core.part_manager import PART_TYPE_CAST, PART_TYPE_CHILL
from core.surrogate.conditioning import build_conditioning_vector
from core.surrogate.dataset import (
    CHILL_BOOST_CHANNEL,
    capture_run,
    load_dataset,
    validate_record,
)
from core.surrogate.fields import _LOG6, PART_TYPE_DIVISOR, encode_channels
from core.surrogate.model import NumpyDemoModel, _decode_chill_boost
from tests.test_simulation_scenarios import (
    _cast_with_chill_manager,
    _run_pipeline_with_patched_memory,
)

def _fake_engine(n: int = 300, chill_fraction: float = 0.0):
    rng = np.random.default_rng(0)
    part_types = np.full(n, PART_TYPE_CAST, dtype=np.int32)
    n_chill = int(round(n * chill_fraction))
    if n_chill:
        part_types[:n_chill] = PART_TYPE_CHILL

    mesh = SimpleNamespace(
        voxel_centers=rng.uniform(-5, 5, (n, 3)),
        voxel_moduli=np.full(n, 0.5, dtype=np.float32),
        voxel_part_types=part_types,
    )
    engine = SimpleNamespace(
        alloy=get_alloy("WCB"),
        mesh=mesh,
        surface_distances=np.full(n, 2.0, dtype=np.float32),
        solidification_times=np.full(n, 100.0, dtype=np.float32),
        solidification_start_times=np.full(n, 50.0, dtype=np.float32),
        local_solidification_time=np.full(n, 50.0, dtype=np.float32),
    )

    def heights():
        z = mesh.voxel_centers[:, 2]
        return (z - z.min()) / (z.max() - z.min() + 1e-9)

    engine._get_normalized_heights = heights
    engine._calculate_riser_influence = lambda: np.ones(n, dtype=np.float32)
    return engine

class ChillDecodeTests(unittest.TestCase):
    def test_decode_inverts_encoder_across_the_boost_range(self):
        for boost in (1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
            encoded = np.log1p(np.clip(boost, 1.0, 5.0) - 1.0) / _LOG6
            decoded = float(_decode_chill_boost(np.asarray([encoded]), _LOG6)[0])
            self.assertAlmostEqual(decoded, boost, places=5, msg=f"boost={boost}")

    def test_demo_model_speeds_up_with_chill(self):
        model = NumpyDemoModel()
        engine = _fake_engine()
        n = len(engine.mesh.voxel_centers)
        slow = np.ones(n, dtype=np.float32)
        cond = build_conditioning_vector(get_alloy("WCB"), {"max_dim": 10.0, "voxel_size": 0.1})

        no_chill, _ = encode_channels(engine, np.ones(n, dtype=np.float32), slow, resolution=16)
        with_chill, _ = encode_channels(engine, np.full(n, 3.0, dtype=np.float32), slow, resolution=16)

        end_plain = model.predict(no_chill, cond)[1]
        end_chill = model.predict(with_chill, cond)[1]
        active = no_chill[0] > 0.5
        ratio = end_chill[active] / end_plain[active]
        self.assertTrue(np.allclose(ratio, 1.0 / 3.0, rtol=1e-3), f"ratio range {ratio.min()}..{ratio.max()}")

    def test_demo_model_responds_to_boosts_below_two(self):
        model = NumpyDemoModel()
        engine = _fake_engine()
        n = len(engine.mesh.voxel_centers)
        slow = np.ones(n, dtype=np.float32)
        cond = build_conditioning_vector(get_alloy("WCB"), {"max_dim": 10.0, "voxel_size": 0.1})
        plain, _ = encode_channels(engine, np.ones(n, dtype=np.float32), slow, resolution=16)
        mild, _ = encode_channels(engine, np.full(n, 1.5, dtype=np.float32), slow, resolution=16)
        active = plain[0] > 0.5
        end_plain = model.predict(plain, cond)[1][active]
        end_mild = model.predict(mild, cond)[1][active]
        self.assertTrue(np.all(end_mild < end_plain), "boost 1.5 was decoded as no chill")

class CaptureUsesInitialBoostTests(unittest.TestCase):
    def test_capture_prefers_initial_boost_over_decayed_array(self):
        engine = _fake_engine(chill_fraction=0.1)
        n = len(engine.mesh.voxel_centers)
        engine._chill_boost = np.ones(n, dtype=np.float32)
        engine._calculate_chill_influence = lambda: np.ones(n, dtype=np.float32)
        engine.compute_initial_chill_influence = lambda: np.full(n, 3.0, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            path = capture_run(engine, voxel_size=0.5, resolution=8, out_dir=tmp)
            self.assertIsNotNone(path)
            records = load_dataset(tmp)
            self.assertEqual(len(records), 1)
            channels = records[0]["channels"]
            active = channels[0].reshape(-1) > 0.5
            boost = channels[CHILL_BOOST_CHANNEL].reshape(-1)[active]
            self.assertTrue(np.any(boost > 0.1), "chill_boost channel captured as zero")

class ValidateRecordTests(unittest.TestCase):
    def _record(self, boost_value: float, chill_fraction: float):
        engine = _fake_engine(chill_fraction=chill_fraction)
        n = len(engine.mesh.voxel_centers)
        channels, _ = encode_channels(
            engine,
            np.full(n, boost_value, dtype=np.float32),
            np.ones(n, dtype=np.float32),
            resolution=8,
        )
        end = np.where(channels[0] > 0.5, 100.0, 0.0).astype(np.float32)
        return {
            "channels": channels,
            "end": end,
            "start": end * 0.5,
            "local": end * 0.5,
        }

    def test_rejects_chill_geometry_with_zero_boost_channel(self):
        record = self._record(boost_value=1.0, chill_fraction=0.1)
        self.assertEqual(validate_record(record), "chill_present_but_zero_boost")

    def test_accepts_chill_geometry_with_boost_signal(self):
        record = self._record(boost_value=2.5, chill_fraction=0.1)
        self.assertIsNone(validate_record(record))

    def test_accepts_no_chill_geometry_with_flat_boost(self):
        record = self._record(boost_value=1.0, chill_fraction=0.0)
        self.assertIsNone(validate_record(record))

class EngineInitialBoostTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = _run_pipeline_with_patched_memory(_cast_with_chill_manager())

    def test_end_of_run_boost_is_flat(self):
        saved = self.engine._chill_boost
        try:
            self.assertGreater(float(self.engine.time), 60.0)
            decayed = np.asarray(self.engine._calculate_chill_influence(), dtype=np.float32)
        finally:
            self.engine._chill_boost = saved
        self.assertLess(float(decayed.max()), 1.05)

    def test_initial_boost_is_strong_next_to_the_chill(self):
        boost = self.engine.compute_initial_chill_influence()
        part_types = np.asarray(self.engine.mesh.voxel_part_types)
        cast = part_types == PART_TYPE_CAST
        max_boost = float(self.engine.analysis_options.solidification_params.chill.max_chill_boost)
        self.assertGreater(float(boost[cast].max()), 1.5)
        self.assertLessEqual(float(boost.max()), max_boost + 1e-4)
        self.assertTrue(np.all(boost[part_types == PART_TYPE_CHILL] == 1.0))

    def test_initial_boost_does_not_mutate_engine_state(self):
        time_before = float(self.engine.time)
        states_before = np.array(self.engine.mesh.voxel_states, copy=True)
        boost_before = self.engine._chill_boost
        boost_copy = None if boost_before is None else np.array(boost_before, copy=True)
        self.engine.compute_initial_chill_influence()
        self.assertEqual(float(self.engine.time), time_before)
        self.assertTrue(np.array_equal(self.engine.mesh.voxel_states, states_before))
        if boost_copy is None:
            self.assertIsNone(self.engine._chill_boost)
        else:
            self.assertTrue(np.array_equal(self.engine._chill_boost, boost_copy))

    def test_capture_from_solved_engine_carries_chill_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = capture_run(self.engine, voxel_size=0.5, resolution=16, out_dir=tmp)
            self.assertIsNotNone(path)
            records = load_dataset(tmp)
            self.assertEqual(len(records), 1, "record rejected by validate_record")
            channels = records[0]["channels"]
            occupied = channels[0].reshape(-1) > 0.5
            chill = np.abs(channels[1].reshape(-1) - PART_TYPE_CHILL / PART_TYPE_DIVISOR) < 0.05
            boost = channels[CHILL_BOOST_CHANNEL].reshape(-1)[occupied & ~chill]
            self.assertGreater(float(boost.max()), 0.1)

if __name__ == "__main__":
    unittest.main()
