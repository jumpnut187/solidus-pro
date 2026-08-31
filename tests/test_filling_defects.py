"""Tests for the F3 defect indicators: air entrapment and cold shuts.

All synthetic -- no mesh, no solver run. ``physics.filling.find_entrapped_gas``
and ``find_cold_shuts`` are pure functions of dense arrays, which is exactly
what lets them be exercised directly against hand-built lattices that isolate
one behaviour at a time (see the module docstrings for why: ``physics/`` may
not depend on a mesh at all).
"""

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
    _domain_boundary_mask,
    _entrapment_warning_if_no_vent,
    _static_vent_mask,
)
from core.part_manager import PART_TYPE_CAST, PART_TYPE_RISER


def _blank_flags(shape) -> np.ndarray:
    """An all-obstacle lattice -- the safe default to carve cells out of."""
    return np.full(shape, FLAG_OBSTACLE, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Air entrapment
# ---------------------------------------------------------------------------


class TestFindEntrappedGas(unittest.TestCase):
    def test_sealed_pocket_is_entrapped(self):
        # A single gas cell with fluid on all six faces has nowhere to go.
        flags = np.full((5, 5, 5), FLAG_FLUID, dtype=np.uint8)
        flags[2, 2, 2] = FLAG_GAS
        vent_mask = np.zeros_like(flags, dtype=bool)

        entrapped = find_entrapped_gas(flags, vent_mask)

        self.assertTrue(entrapped[2, 2, 2])
        self.assertEqual(int(entrapped.sum()), 1)

    def test_sealed_pocket_with_one_vent_cell_is_not_entrapped(self):
        # A two-cell pocket, fully enclosed by fluid except that one of its
        # own cells is (per the caller's vent_mask) open to atmosphere. The
        # whole connected component escapes, not just the vented cell.
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
        # Two isolated single-cell pockets, far enough apart to be separate
        # 6-connected components. Venting one must not affect the other.
        flags = np.full((5, 5, 9), FLAG_FLUID, dtype=np.uint8)
        flags[2, 2, 2] = FLAG_GAS  # pocket A
        flags[2, 2, 6] = FLAG_GAS  # pocket B
        vent_mask = np.zeros_like(flags, dtype=bool)
        vent_mask[2, 2, 6] = True  # only B is vented

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

        # A non-empty vent mask must not warn, even if it vents nothing in
        # particular right now.
        vent_mask_nonempty = np.zeros_like(flags, dtype=bool)
        vent_mask_nonempty[0, 0, 0] = True
        self.assertIsNone(_entrapment_warning_if_no_vent(vent_mask_nonempty))

    def test_volume_is_counted_correctly(self):
        # A known 1x1x3 sealed pocket must count exactly 3 cells, not the
        # component count or the bounding-box volume.
        flags = np.full((6, 6, 6), FLAG_FLUID, dtype=np.uint8)
        flags[2, 2, 1:4] = FLAG_GAS
        vent_mask = np.zeros_like(flags, dtype=bool)

        entrapped = find_entrapped_gas(flags, vent_mask)

        self.assertEqual(int(entrapped.sum()), 3)

    def test_peak_entrapped_is_at_least_final(self):
        # Two snapshots in time of the same pocket: mid-fill it is 4 cells
        # and unvented; by the end the metal has closed off all but 1 cell
        # of it (still unvented). Tracking a running max across snapshots,
        # as core.filling_stage does across periodic checks, must never let
        # the final value exceed the peak.
        shape = (6, 6, 6)
        vent_mask = np.zeros(shape, dtype=bool)

        mid_fill = np.full(shape, FLAG_FLUID, dtype=np.uint8)
        mid_fill[2, 2, 1:5] = FLAG_GAS  # 4 cells
        mid_count = int(find_entrapped_gas(mid_fill, vent_mask).sum())

        final = np.full(shape, FLAG_FLUID, dtype=np.uint8)
        final[2, 2, 1] = FLAG_GAS  # 1 cell left
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
    """core.filling_stage's static half of the vent mask (riser tops, inlet)."""

    def test_open_riser_top_is_a_vent(self):
        # A riser cell at the very top of the domain (nothing modelled above
        # it) is open to atmosphere regardless of what currently occupies it.
        dense = np.full((3, 3, 3), EMPTY_CELL, dtype=np.int8)
        dense[1, 1, :] = PART_TYPE_RISER
        inlet = np.zeros((0, 3), dtype=np.int64)

        vent = _static_vent_mask(dense, inlet, vent_risers_enabled=True)

        self.assertTrue(vent[1, 1, 2])  # top of the domain -- nothing above

    def test_riser_top_ignored_when_covered(self):
        # A riser cell with more geometry directly above it is not the open
        # top face and must not vent.
        dense = np.full((3, 3, 3), EMPTY_CELL, dtype=np.int8)
        dense[1, 1, 0] = PART_TYPE_RISER
        dense[1, 1, 1] = PART_TYPE_CAST  # covers it
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

    def test_domain_boundary_mask_covers_only_the_edges(self):
        shape = (3, 4, 5)
        boundary = _domain_boundary_mask(shape)

        self.assertTrue(boundary[0, 0, 0])
        self.assertTrue(boundary[-1, -1, -1])
        self.assertTrue(boundary[0, 2, 3])  # x boundary, interior y/z
        self.assertFalse(boundary[1, 1, 1])  # strictly interior

    def test_full_vent_pipeline_frees_a_pocket_touching_an_open_riser_top(self):
        # Wires _static_vent_mask + _domain_boundary_mask together with
        # find_entrapped_gas, the way core.filling_stage assembles them each
        # periodic check.
        shape = (5, 5, 5)
        dense = np.full(shape, EMPTY_CELL, dtype=np.int8)
        dense[2, 2, :] = PART_TYPE_RISER  # riser column, open at the top

        flags = np.full(shape, FLAG_FLUID, dtype=np.uint8)
        flags[2, 2, 4] = FLAG_GAS  # gas sitting in the open riser top
        flags[1, 1, 1] = FLAG_GAS  # a separate, fully enclosed pocket

        inlet = np.zeros((0, 3), dtype=np.int64)
        static_vent = _static_vent_mask(dense, inlet, vent_risers_enabled=True)
        boundary = _domain_boundary_mask(shape)
        vent_mask = static_vent | (boundary & (flags == FLAG_GAS))

        entrapped = find_entrapped_gas(flags, vent_mask)

        self.assertFalse(entrapped[2, 2, 4])  # vented through the riser top
        self.assertTrue(entrapped[1, 1, 1])   # no path out


# ---------------------------------------------------------------------------
# Cold shuts
# ---------------------------------------------------------------------------


LIQUIDUS_C = 600.0
MARGIN_C = 25.0
GAP_S = 0.5
HOT_C = 700.0
COLD_C = 400.0


def _corridor(length: int):
    """
    A 1-cell-wide fluid corridor along x, padded with obstacle on every
    side (including x) so np.roll's wraparound never contaminates a result
    with a bogus "neighbour" from the far end of the array.

    Returns (flags, fill_time, temperature, x_indices), where x_indices are
    the wet cell x-coordinates (1..length), and fill_time/temperature start
    at the sentinels everywhere.
    """
    shape = (length + 2, 3, 3)
    flags = _blank_flags(shape)
    flags[1 : length + 1, 1, 1] = FLAG_FLUID
    fill_time = np.full(shape, -1.0, dtype=np.float32)
    temperature = np.full(shape, NO_TEMPERATURE, dtype=np.float32)
    x = np.arange(1, length + 1)
    return flags, fill_time, temperature, x


class TestFindColdShuts(unittest.TestCase):
    def test_hot_fronts_meeting_late_is_not_flagged(self):
        # Two fronts converge (a "tent" arrival-time profile peaking in the
        # middle -- both neighbours of the peak arrived clearly earlier),
        # but the metal is still hot. Meeting alone is not a cold shut.
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
        # A single front sweeping the whole corridor: arrival time is
        # monotonic, so at most one face direction per cell is "older".
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
        """
        Documents the false positive the design doc's own prose warns
        about, and shows this implementation's fix for it.

        The literal wording in docs/filling_implementation.md section 6.4
        ("a wet neighbour whose arrival time differs by more than
        arrival_gap_s") is checked here directly against a slow single
        front -- one cell every 2 seconds, well past a 0.5 s gap -- and
        shown to misfire along the whole channel. find_cold_shuts requires
        at least two such directions rather than one, which the same
        single front never produces, and correctly reports nothing.
        """
        flags, fill_time, temperature, x = _corridor(20)
        dt_per_cell = 2.0  # >> arrival_gap_s
        fill_time[x, 1, 1] = (x - x.min()).astype(np.float64) * dt_per_cell
        temperature[x, 1, 1] = COLD_C  # cold, so only "meeting" can save it

        # The naive, literal criterion: >=1 wet neighbour differs by more
        # than the gap. This is what the design doc's prose describes.
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
        # Start from the flagged "meeting and cold" case, then corrupt the
        # peak cell's own fields with the sentinels -- as if a bug briefly
        # left a wet cell without a proper arrival time or temperature.
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

        fill_time[mid, 1, 1] = values[len(x) // 2]  # restore
        temperature[mid, 1, 1] = NO_TEMPERATURE
        result = find_cold_shuts(
            fill_time, temperature, flags,
            liquidus_c=LIQUIDUS_C, arrival_gap_s=GAP_S,
            temperature_margin_c=MARGIN_C,
        )
        self.assertFalse(result[mid, 1, 1])

        # Cells that were always gas/obstacle carry the sentinels by
        # construction (never touched above) and must never be flagged.
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
