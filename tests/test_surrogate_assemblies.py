"""Rigged training assemblies (core/surrogate/assemblies.py) and their generate_dataset path."""

from __future__ import annotations

import tempfile
import unittest
from collections import Counter

import numpy as np

from core.part_manager import PART_TYPE_CAST, PART_TYPE_CHILL, PART_TYPE_CORE, PART_TYPE_RISER
from core.surrogate.assemblies import ASSEMBLY_FAMILIES, CORED_FAMILIES, iter_assemblies
from core.surrogate.dataset import load_dataset
from core.surrogate.fields import CHANNELS, PART_TYPE_DIVISOR
from core.surrogate.training import alloys_for_geometry, generate_dataset


class AssemblyGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = list(iter_assemblies(2 * len(ASSEMBLY_FAMILIES), seed=3))

    def test_deterministic_and_prefix_stable(self) -> None:
        short = [s.name for s in iter_assemblies(5, seed=3)]
        self.assertEqual(short, [s.name for s in self.specs[:5]])

    def test_every_assembly_has_one_casting_and_one_riser_on_metal(self) -> None:
        for spec in self.specs:
            codes = Counter(code for _, code in spec.parts)
            self.assertEqual(codes[PART_TYPE_CAST], 1, spec.name)
            self.assertEqual(codes[PART_TYPE_RISER], 1, spec.name)
            cast = spec.parts[0][0]
            riser = next(m for m, c in spec.parts if c == PART_TYPE_RISER)
            lo, hi = riser.bounds
            base = [[0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2] + 0.05]]
            self.assertTrue(bool(cast.contains(base)[0]), f"{spec.name}: riser base is not on metal")

    def test_cored_families_carry_a_core_within_the_casting(self) -> None:
        for spec in self.specs:
            has_core = any(c == PART_TYPE_CORE for _, c in spec.parts)
            self.assertEqual(has_core, spec.family.removeprefix("rig_") in CORED_FAMILIES, spec.name)
            if not has_core:
                continue
            # Bounding boxes, not contains(): compound castings are
            # overlapping shells, where ray parity flips inside the overlap.
            cast = spec.parts[0][0]
            core = next(m for m, c in spec.parts if c == PART_TYPE_CORE)
            # Through cores have prints standing out of the casting, so the
            # core only has to reach into it on every axis.
            overlaps = np.all(core.bounds[0] < cast.bounds[1]) and np.all(core.bounds[1] > cast.bounds[0])
            self.assertTrue(bool(overlaps), f"{spec.name}: core does not reach into the casting")

    def test_chills_touch_the_casting(self) -> None:
        chilled = [s for s in self.specs if any(c == PART_TYPE_CHILL for _, c in s.parts)]
        self.assertGreater(len(chilled), 0)
        for spec in chilled:
            cast = spec.parts[0][0]
            chill = next(m for m, c in spec.parts if c == PART_TYPE_CHILL)
            gap = np.maximum(cast.bounds[0] - chill.bounds[1], chill.bounds[0] - cast.bounds[1])
            self.assertLessEqual(float(np.max(gap)), 1e-6, spec.name)

    def test_alloy_subset_is_seeded_by_name(self) -> None:
        alloys = ["A", "B", "C", "D", "E"]
        picks = alloys_for_geometry("rig_box_001", alloys, 2)
        self.assertEqual(picks, alloys_for_geometry("rig_box_001", alloys, 2))
        self.assertEqual(len(picks), 2)
        self.assertEqual(alloys_for_geometry("rig_box_001", alloys, None), alloys)


class AssemblyDatasetTests(unittest.TestCase):
    def test_generate_dataset_runs_an_assembly_as_a_multi_part_job(self) -> None:
        spec = next(iter_assemblies(1, seed=0, families=["cored_cylinder"]))
        with tempfile.TemporaryDirectory() as out_dir:
            generated = generate_dataset(
                ["WCB"], out_dir, grid=32, geometries=[spec], target_voxels=4_000, max_voxels=8_000,
            )
            self.assertEqual(generated, 1)
            record = load_dataset(out_dir)[0]
        channels = np.asarray(record["channels"])
        occupied = channels[0] > 0.5
        codes = np.rint(channels[CHANNELS.index("part_type")][occupied] * PART_TYPE_DIVISOR).astype(int)
        self.assertIn(PART_TYPE_CORE, codes.tolist())
        self.assertIn(PART_TYPE_RISER, codes.tolist())
        end = np.asarray(record["end"])[occupied]
        # A cell is labelled by its majority part, so a core-labelled cell on
        # the core face can still average in some metal end time.
        self.assertEqual(float(np.median(end[codes == PART_TYPE_CORE])), 0.0)
        self.assertGreater(float(end[codes == PART_TYPE_CAST].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
