"""Rigged training assemblies (core/surrogate/assemblies.py, design_rules.py) and their generate_dataset path."""

from __future__ import annotations

import tempfile
import unittest
from collections import Counter

import numpy as np
import trimesh

from core.part_manager import PART_TYPE_CAST, PART_TYPE_CHILL, PART_TYPE_CORE, PART_TYPE_RISER
from core.surrogate import design_rules
from core.surrogate.assemblies import ASSEMBLY_FAMILIES, COMPLEX_FAMILIES, CORED_FAMILIES, iter_assemblies
from core.surrogate.dataset import load_dataset
from core.surrogate.fields import CHANNELS, PART_TYPE_DIVISOR
from core.surrogate.training import alloys_for_geometry, generate_dataset


class AssemblyGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = list(iter_assemblies(len(ASSEMBLY_FAMILIES), seed=3))
        cls.complex = list(iter_assemblies(len(COMPLEX_FAMILIES), seed=3, families=COMPLEX_FAMILIES))

    def test_deterministic_and_prefix_stable(self) -> None:
        short = [s.name for s in iter_assemblies(3, seed=3)]
        self.assertEqual(short, [s.name for s in self.specs[:3]])

    def test_every_assembly_passes_the_design_rules(self) -> None:
        for spec in self.specs + self.complex:
            report = design_rules.check(spec.parts)
            self.assertTrue(report.ok, f"{spec.name}: {report.violations}")
            self.assertGreaterEqual(report.metrics["fed_fraction"], design_rules.FED_FRACTION_REQUIRED - 0.01)

    def test_one_casting_and_at_least_one_riser(self) -> None:
        for spec in self.specs + self.complex:
            codes = Counter(code for _, code in spec.parts)
            self.assertEqual(codes[PART_TYPE_CAST], 1, spec.name)
            self.assertGreaterEqual(codes[PART_TYPE_RISER], 1, spec.name)

    def test_cored_families_carry_cores(self) -> None:
        for spec in self.specs + self.complex:
            family = spec.family.removeprefix("rig_")
            if family in CORED_FAMILIES or family in ("complex_cored_housing", "complex_flanged_hub"):
                self.assertTrue(any(c == PART_TYPE_CORE for _, c in spec.parts), spec.name)

    def test_chills_sit_flush_on_the_casting(self) -> None:
        for spec in self.specs + self.complex:
            cast = spec.parts[0][0]
            for chill in (m for m, c in spec.parts if c == PART_TYPE_CHILL):
                gap = np.maximum(cast.bounds[0] - chill.bounds[1], chill.bounds[0] - cast.bounds[1])
                self.assertLessEqual(float(np.max(gap)), 0.02, spec.name)

    def test_alloy_subset_is_seeded_by_name(self) -> None:
        alloys = ["A", "B", "C", "D", "E"]
        picks = alloys_for_geometry("rig_box_001", alloys, 2)
        self.assertEqual(picks, alloys_for_geometry("rig_box_001", alloys, 2))
        self.assertEqual(len(picks), 2)
        self.assertEqual(alloys_for_geometry("rig_box_001", alloys, None), alloys)


class DesignRuleTests(unittest.TestCase):
    """``design_rules.check`` rejects the classic bad designs."""

    def setUp(self) -> None:
        self.cast = trimesh.creation.box(extents=(4.0, 4.0, 2.0))
        self.cast.apply_translation((0.0, 0.0, 1.0))

    def _riser(self, diameter: float) -> trimesh.Trimesh:
        h = 1.5 * diameter
        mesh = trimesh.creation.cylinder(radius=diameter / 2, height=h, sections=48)
        mesh.apply_translation((0.0, 0.0, 2.0 - design_rules.RISER_SINK_IN + h / 2))
        return mesh

    def test_a_correctly_sized_riser_passes(self) -> None:
        report = design_rules.check([(self.cast, PART_TYPE_CAST), (self._riser(3.0), PART_TYPE_RISER)])
        self.assertTrue(report.ok, report.violations)

    def test_an_undersized_riser_fails(self) -> None:
        report = design_rules.check([(self.cast, PART_TYPE_CAST), (self._riser(1.0), PART_TYPE_RISER)])
        self.assertTrue(any("modulus" in v for v in report.violations), report.violations)

    def test_no_riser_fails(self) -> None:
        self.assertIn("no riser", design_rules.check([(self.cast, PART_TYPE_CAST)]).violations)

    def test_an_enclosed_core_fails(self) -> None:
        core = trimesh.creation.box(extents=(1.0, 1.0, 0.8))
        core.apply_translation((0.0, 0.0, 1.0))
        report = design_rules.check(
            [(self.cast, PART_TYPE_CAST), (core, PART_TYPE_CORE), (self._riser(3.0), PART_TYPE_RISER)]
        )
        self.assertTrue(any("enclosed" in v for v in report.violations), report.violations)

    def test_a_floating_chill_fails(self) -> None:
        chill = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        chill.apply_translation((0.0, 0.0, -3.0))
        report = design_rules.check(
            [(self.cast, PART_TYPE_CAST), (chill, PART_TYPE_CHILL), (self._riser(3.0), PART_TYPE_RISER)]
        )
        self.assertTrue(any("does not touch" in v for v in report.violations), report.violations)

    def test_feeding_distance_follows_the_steel_rule(self) -> None:
        self.assertAlmostEqual(design_rules.feeding_distance(1.0), 4.5)
        self.assertAlmostEqual(design_rules.feeding_distance(4.0), 12.0)


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
