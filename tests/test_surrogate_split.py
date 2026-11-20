from __future__ import annotations

import unittest

import numpy as np

from core.surrogate.split import (
    geometry_group_key,
    grouped_train_val_indices,
    split_records,
)

def _record(seed: int, resolution: int = 8) -> dict:
    rng = np.random.default_rng(seed)
    channels = np.zeros((7, resolution, resolution, resolution), dtype=np.float32)
    channels[0] = (rng.uniform(size=(resolution,) * 3) > 0.7).astype(np.float32)
    channels[3] = rng.uniform(size=(resolution,) * 3).astype(np.float32)
    return {"channels": channels, "alloy_name": f"alloy{seed}"}

class GroupKeyTests(unittest.TestCase):
    def test_same_geometry_different_channels_share_key(self):
        a = _record(1)
        b = _record(1)
        b["channels"][3] += 0.1
        b["alloy_name"] = "other"
        self.assertEqual(geometry_group_key(a), geometry_group_key(b))

    def test_different_geometry_differs(self):
        self.assertNotEqual(geometry_group_key(_record(1)), geometry_group_key(_record(2)))

class GroupedSplitTests(unittest.TestCase):
    def test_no_group_straddles_split(self):
        records = []
        for g in range(6):
            for a in range(4):
                r = _record(g)
                r["alloy_name"] = f"alloy{a}"
                records.append(r)
        train, val = grouped_train_val_indices(records, val_fraction=0.2, seed=0)
        self.assertEqual(sorted(train + val), list(range(24)))
        train_keys = {geometry_group_key(records[i]) for i in train}
        val_keys = {geometry_group_key(records[i]) for i in val}
        self.assertFalse(train_keys & val_keys)
        self.assertGreaterEqual(len(val), 4)
        self.assertGreater(len(train), len(val))

    def test_single_group_goes_entirely_to_train(self):
        records = [_record(3) for _ in range(5)]
        train, val = grouped_train_val_indices(records)
        self.assertEqual(train, list(range(5)))
        self.assertEqual(val, [])

    def test_deterministic_by_seed(self):
        records = [_record(g) for g in range(10)]
        a = grouped_train_val_indices(records, seed=7)
        b = grouped_train_val_indices(records, seed=7)
        self.assertEqual(a, b)

    def test_split_records_wrapper(self):
        records = [_record(g) for g in range(10)]
        train, val = split_records(records, val_fraction=0.3)
        self.assertEqual(len(train) + len(val), 10)
        self.assertTrue(val)

    def test_empty(self):
        self.assertEqual(grouped_train_val_indices([]), ([], []))

if __name__ == "__main__":
    unittest.main()
