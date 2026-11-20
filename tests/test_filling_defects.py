from __future__ import annotations

import unittest

import numpy as np

from physics.filling import (
    FLAG_FLUID,
    FLAG_GAS,
    FLAG_INTERFACE,
    FLAG_OBSTACLE,
    NO_TEMPERATURE,
    find_cold_shuts,
    find_entrapped_gas,
)
from core.filling_stage import (
    EMPTY_CELL,
    _open_air_mask,
    _entrapment_warning_if_no_vent,
    _static_vent_mask,
)
from core.part_manager import PART_TYPE_CAST, PART_TYPE_RISER

def _blank_flags(shape) -> np.ndarray:
    return np.full(shape, FLAG_OBSTACLE, dtype=np.uint8)

class TestFindEntrappedGas(unittest.TestCase):
    def test_sealed_pocket_is_entrapped(self):
        flags = np.full((5, 5, 5), FLAG_FLUID, dtype=np.uint8)
        flags[2, 2, 2] = FLAG_GAS
        vent_mask = np.zeros_like(flags, dtype=bool)

        entrapped = find_entrapped_gas(flags, vent_mask)

        self.assertTrue(entrapped[2, 2, 2])
        self.assertEqual(int(entrapped.sum()), 1)

    def test_sealed_pocket_with_one_vent_cell_is_not_entrapped(self):
        flags = np.full((5, 5, 5), FLAG_FLUID, dtype=np.uint8)
        flags[2, 2, 1] = FLAG_GAS
        flags[2, 2, 2] = FLAG_GAS
        vent_mask = np.zeros_like(flags, dtype=bool)
        vent_mask[2, 2, 1] = True

        entrapped = find_entrapped_gas(flags, vent_mask)

        self.assertFalse(entrapped[2, 2, 1])
        self.assertFalse(entrapped[2, 2, 2])
        self.assertEqual(int(entrapped.sum()), 0)

    def test_two_pockets_labelled_independently(self):
        flags = np.full((5, 5, 9), FLAG_FLUID, dtype=np.uint8)
        flags[2, 2, 2] = FLAG_GAS
        flags[2, 2, 6] = FLAG_GAS
        vent_mask = np.zeros_like(flags, dtype=bool)
        vent_mask[2, 2, 6] = True

        entrapped = find_entrapped_gas(flags, vent_mask)

        self.assertTrue(entrapped[2, 2, 2])
        self.assertFalse(entrapped[2, 2, 6])
        self.assertEqual(int(entrapped.sum()), 1)

    def test_empty_vent_mask_flags_everything_and_warns(self):
        flags = np.full((5, 5, 5), FLAG_FLUID, dtype=np.uint8)
        flags[1, 1, 1] = FLAG_GAS
        flags[3, 3, 3] = FLAG_GAS
        vent_mask = np.zeros_like(flags, dtype=bool)

        entrapped = find_entrapped_gas(flags, vent_mask)
        self.assertTrue(entrapped[1, 1, 1])
        self.assertTrue(entrapped[3, 3, 3])
        self.assertEqual(int(entrapped.sum()), 2)

        message = _entrapment_warning_if_no_vent(vent_mask)
        self.assertIsNotNone(message)
        self.assertIn("vent", message.lower())

        vent_mask_nonempty = np.zeros_like(flags, dtype=bool)
        vent_mask_nonempty[0, 0, 0] = True
        self.assertIsNone(_entrapment_warning_if_no_vent(vent_mask_nonempty))

    def test_volume_is_counted_correctly(self):
        flags = np.full((6, 6, 6), FLAG_FLUID, dtype=np.uint8)
        flags[2, 2, 1:4] = FLAG_GAS
        vent_mask = np.zeros_like(flags, dtype=bool)

        entrapped = find_entrapped_gas(flags, vent_mask)

        self.assertEqual(int(entrapped.sum()), 3)

    def test_peak_entrapped_is_at_least_final(self):
        shape = (6, 6, 6)
        vent_mask = np.zeros(shape, dtype=bool)

        mid_fill = np.full(shape, FLAG_FLUID, dtype=np.uint8)
        mid_fill[2, 2, 1:5] = FLAG_GAS
        mid_count = int(find_entrapped_gas(mid_fill, vent_mask).sum())

        final = np.full(shape, FLAG_FLUID, dtype=np.uint8)
        final[2, 2, 1] = FLAG_GAS
        final_count = int(find_entrapped_gas(final, vent_mask).sum())

        peak = max(mid_count, final_count)
        self.assertEqual(mid_count, 4)
        self.assertEqual(final_count, 1)
        self.assertGreaterEqual(peak, final_count)

    def test_no_gas_at_all_is_never_entrapped(self):
        flags = np.full((4, 4, 4), FLAG_FLUID, dtype=np.uint8)
        vent_mask = np.zeros_like(flags, dtype=bool)
        entrapped = find_entrapped_gas(flags, vent_mask)
        self.assertEqual(int(entrapped.sum()), 0)

class TestVentMaskConstruction(unittest.TestCase):

    def test_open_riser_top_is_a_vent(self):
        dense = np.full((3, 3, 3), EMPTY_CELL, dtype=np.int8)
        dense[1, 1, :] = PART_TYPE_RISER
        inlet = np.zeros((0, 3), dtype=np.int64)

        vent = _static_vent_mask(dense, inlet, vent_risers_enabled=True)

        self.assertTrue(vent[1, 1, 2])

    def test_riser_top_ignored_when_covered(self):
        dense = np.full((3, 3, 3), EMPTY_CELL, dtype=np.int8)
        dense[1, 1, 0] = PART_TYPE_RISER
        dense[1, 1, 1] = PART_TYPE_CAST
        inlet = np.zeros((0, 3), dtype=np.int64)

        vent = _static_vent_mask(dense, inlet, vent_risers_enabled=True)

        self.assertFalse(vent[1, 1, 0])

    def test_vent_risers_disabled_gives_no_riser_vent(self):
        dense = np.full((3, 3, 3), EMPTY_CELL, dtype=np.int8)
        dense[1, 1, :] = PART_TYPE_RISER
        inlet = np.zeros((0, 3), dtype=np.int64)

        vent = _static_vent_mask(dense, inlet, vent_risers_enabled=False)

        self.assertFalse(np.any(vent))

    def test_inlet_cells_are_always_vents(self):
        dense = np.full((3, 3, 3), PART_TYPE_CAST, dtype=np.int8)
        inlet = np.array([[0, 0, 0]], dtype=np.int64)

        vent = _static_vent_mask(dense, inlet, vent_risers_enabled=False)

        self.assertTrue(vent[0, 0, 0])
        self.assertEqual(int(vent.sum()), 1)

    def test_open_air_mask_is_the_top_face_and_nothing_else(self):
        shape = (3, 4, 5)
        open_air = _open_air_mask(shape)

        self.assertTrue(open_air[0, 0, -1])
        self.assertTrue(open_air[-1, -1, -1])
        self.assertTrue(open_air[1, 2, -1])
        self.assertEqual(int(open_air.sum()), shape[0] * shape[1])
        self.assertFalse(open_air[0, 0, 0])
        self.assertFalse(open_air[0, 2, 3])
        self.assertFalse(open_air[1, 0, 3])
        self.assertFalse(open_air[1, 1, 1])

    def test_full_vent_pipeline_frees_a_pocket_touching_an_open_riser_top(self):
        shape = (5, 5, 5)
        dense = np.full(shape, EMPTY_CELL, dtype=np.int8)
        dense[2, 2, :] = PART_TYPE_RISER

        flags = np.full(shape, FLAG_FLUID, dtype=np.uint8)
        flags[2, 2, 4] = FLAG_GAS
        flags[1, 1, 1] = FLAG_GAS

        inlet = np.zeros((0, 3), dtype=np.int64)
        static_vent = _static_vent_mask(dense, inlet, vent_risers_enabled=True)
        vent_mask = static_vent | (_open_air_mask(shape) & (flags == FLAG_GAS))

        entrapped = find_entrapped_gas(flags, vent_mask)

        self.assertFalse(entrapped[2, 2, 4])
        self.assertTrue(entrapped[1, 1, 1])

    def test_a_pocket_on_the_floor_is_entrapped_not_vented(self):
        shape = (5, 5, 5)
        dense = np.full(shape, EMPTY_CELL, dtype=np.int8)
        dense[:, :, :] = PART_TYPE_CAST

        flags = np.full(shape, FLAG_FLUID, dtype=np.uint8)
        flags[2, 2, 0] = FLAG_GAS

        inlet = np.zeros((0, 3), dtype=np.int64)
        static_vent = _static_vent_mask(dense, inlet, vent_risers_enabled=True)

        shipped = static_vent | (_open_air_mask(shape) & (flags == FLAG_GAS))
        self.assertEqual(int(find_entrapped_gas(flags, shipped).sum()), 1)
        self.assertTrue(find_entrapped_gas(flags, shipped)[2, 2, 0])

        six_face = np.zeros(shape, dtype=bool)
        six_face[[0, -1], :, :] = True
        six_face[:, [0, -1], :] = True
        six_face[:, :, [0, -1]] = True
        stale = static_vent | (six_face & (flags == FLAG_GAS))
        self.assertEqual(int(find_entrapped_gas(flags, stale).sum()), 0)

    def test_a_pocket_on_a_side_face_is_entrapped_too(self):
        shape = (5, 5, 5)
        dense = np.full(shape, PART_TYPE_CAST, dtype=np.int8)
        flags = np.full(shape, FLAG_FLUID, dtype=np.uint8)
        flags[0, 2, 2] = FLAG_GAS

        inlet = np.zeros((0, 3), dtype=np.int64)
        static_vent = _static_vent_mask(dense, inlet, vent_risers_enabled=True)
        vent_mask = static_vent | (_open_air_mask(shape) & (flags == FLAG_GAS))

        self.assertTrue(find_entrapped_gas(flags, vent_mask)[0, 2, 2])

LIQUIDUS_C = 600.0
MARGIN_C = 25.0
GAP_S = 0.5
HOT_C = 700.0
COLD_C = 400.0

def _corridor(length: int):
    shape = (length + 2, 3, 3)
    flags = _blank_flags(shape)
    flags[1 : length + 1, 1, 1] = FLAG_FLUID
    fill_time = np.full(shape, -1.0, dtype=np.float32)
    temperature = np.full(shape, NO_TEMPERATURE, dtype=np.float32)
    x = np.arange(1, length + 1)
    return flags, fill_time, temperature, x

class TestFindColdShuts(unittest.TestCase):
    def test_hot_fronts_meeting_late_is_not_flagged(self):
        flags, fill_time, temperature, x = _corridor(11)
        mid = x[len(x) // 2]
        values = np.minimum(x - x.min(), x.max() - x).astype(np.float64)
        fill_time[x, 1, 1] = values
        temperature[x, 1, 1] = HOT_C

        result = find_cold_shuts(
            fill_time, temperature, flags,
            liquidus_c=LIQUIDUS_C, arrival_gap_s=GAP_S,
            temperature_margin_c=MARGIN_C,
        )

        self.assertEqual(int(result.sum()), 0)
        self.assertFalse(result[mid, 1, 1])

    def test_cold_metal_arriving_smoothly_is_not_flagged(self):
        flags, fill_time, temperature, x = _corridor(11)
        fill_time[x, 1, 1] = (x - x.min()).astype(np.float64)
        temperature[x, 1, 1] = COLD_C

        result = find_cold_shuts(
            fill_time, temperature, flags,
            liquidus_c=LIQUIDUS_C, arrival_gap_s=GAP_S,
            temperature_margin_c=MARGIN_C,
        )

        self.assertEqual(int(result.sum()), 0)

    def test_advancing_front_false_positive_is_avoided(self):
        flags, fill_time, temperature, x = _corridor(20)
        dt_per_cell = 2.0
        fill_time[x, 1, 1] = (x - x.min()).astype(np.float64) * dt_per_cell
        temperature[x, 1, 1] = COLD_C

        wet = flags == FLAG_FLUID
        naive_flagged = np.zeros(flags.shape, dtype=bool)
        for shift in (1, -1):
            nbr_time = np.roll(fill_time, shift=shift, axis=0)
            nbr_wet = np.roll(wet, shift=shift, axis=0)
            differs = wet & nbr_wet & (np.abs(fill_time - nbr_time) > GAP_S)
            naive_flagged |= differs
        naive_flagged &= temperature <= (LIQUIDUS_C + MARGIN_C)
        self.assertGreater(
            int(naive_flagged.sum()), 0,
            "test setup did not reproduce the false positive it means to guard",
        )

        result = find_cold_shuts(
            fill_time, temperature, flags,
            liquidus_c=LIQUIDUS_C, arrival_gap_s=GAP_S,
            temperature_margin_c=MARGIN_C,
        )
        self.assertEqual(
            int(result.sum()), 0,
            "a single slow front must not read as a cold shut across its length",
        )

    def test_cold_fronts_meeting_is_flagged(self):
        flags, fill_time, temperature, x = _corridor(11)
        mid = x[len(x) // 2]
        values = np.minimum(x - x.min(), x.max() - x).astype(np.float64)
        fill_time[x, 1, 1] = values
        temperature[x, 1, 1] = COLD_C

        result = find_cold_shuts(
            fill_time, temperature, flags,
            liquidus_c=LIQUIDUS_C, arrival_gap_s=GAP_S,
            temperature_margin_c=MARGIN_C,
        )

        self.assertTrue(result[mid, 1, 1])
        self.assertGreater(int(result.sum()), 0)

    def test_sentinels_never_produce_a_false_positive(self):
        flags, fill_time, temperature, x = _corridor(11)
        mid = x[len(x) // 2]
        values = np.minimum(x - x.min(), x.max() - x).astype(np.float64)
        fill_time[x, 1, 1] = values
        temperature[x, 1, 1] = COLD_C

        fill_time[mid, 1, 1] = -1.0
        result = find_cold_shuts(
            fill_time, temperature, flags,
            liquidus_c=LIQUIDUS_C, arrival_gap_s=GAP_S,
            temperature_margin_c=MARGIN_C,
        )
        self.assertFalse(result[mid, 1, 1])

        fill_time[mid, 1, 1] = values[len(x) // 2]
        temperature[mid, 1, 1] = NO_TEMPERATURE
        result = find_cold_shuts(
            fill_time, temperature, flags,
            liquidus_c=LIQUIDUS_C, arrival_gap_s=GAP_S,
            temperature_margin_c=MARGIN_C,
        )
        self.assertFalse(result[mid, 1, 1])

        never_filled = (flags != FLAG_FLUID) & (flags != FLAG_INTERFACE)
        self.assertTrue(np.all(fill_time[never_filled] < 0))
        self.assertTrue(np.all(np.isnan(temperature[never_filled])))
        result = find_cold_shuts(
            fill_time, temperature, flags,
            liquidus_c=LIQUIDUS_C, arrival_gap_s=GAP_S,
            temperature_margin_c=MARGIN_C,
        )
        self.assertFalse(np.any(result[never_filled]))

    def test_no_wet_cells_returns_all_false(self):
        shape = (4, 4, 4)
        flags = _blank_flags(shape)
        fill_time = np.full(shape, -1.0, dtype=np.float32)
        temperature = np.full(shape, NO_TEMPERATURE, dtype=np.float32)

        result = find_cold_shuts(
            fill_time, temperature, flags,
            liquidus_c=LIQUIDUS_C, arrival_gap_s=GAP_S,
            temperature_margin_c=MARGIN_C,
        )
        self.assertEqual(int(result.sum()), 0)

if __name__ == "__main__":
    unittest.main()
