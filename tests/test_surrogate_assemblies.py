"""Rigged training assemblies (core/surrogate/assemblies.py, design_rules.py) and their generate_dataset path."""

from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from unittest import mock

import numpy as np
import trimesh
from scipy import ndimage

from core.part_manager import PART_TYPE_CAST, PART_TYPE_CHILL, PART_TYPE_CORE, PART_TYPE_RISER
from core.surrogate import design_rules
from core.surrogate.assemblies import (
    ASSEMBLY_FAMILIES,
    COMPLEX_FAMILIES,
    CORED_FAMILIES,
    HEAVY_FAMILIES,
    HEAVY_SECTION_IN,
    iter_assemblies,
)
from core.surrogate.dataset import load_dataset
from core.surrogate.fields import CHANNELS, PART_TYPE_DIVISOR
from core.surrogate.training import alloys_for_geometry, generate_dataset


class AssemblyGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = list(iter_assemblies(len(ASSEMBLY_FAMILIES), seed=3))
        cls.complex = list(iter_assemblies(len(COMPLEX_FAMILIES), seed=3, families=COMPLEX_FAMILIES))
        cls.heavy = list(iter_assemblies(len(HEAVY_FAMILIES), seed=3, families=HEAVY_FAMILIES))

    def test_deterministic_and_prefix_stable(self) -> None:
        short = [s.name for s in iter_assemblies(3, seed=3)]
        self.assertEqual(short, [s.name for s in self.specs[:3]])

    def test_every_assembly_passes_the_design_rules(self) -> None:
        for spec in self.specs + self.complex + self.heavy:
            report = design_rules.check(spec.parts)
            self.assertTrue(report.ok, f"{spec.name}: {report.violations}")
            self.assertGreaterEqual(report.metrics["fed_fraction"], design_rules.FED_FRACTION_REQUIRED - 0.01)

    def test_one_casting_and_at_least_one_riser(self) -> None:
        for spec in self.specs + self.complex + self.heavy:
            codes = Counter(code for _, code in spec.parts)
            self.assertEqual(codes[PART_TYPE_CAST], 1, spec.name)
            self.assertGreaterEqual(codes[PART_TYPE_RISER], 1, spec.name)

    def test_cored_families_carry_cores(self) -> None:
        for spec in self.specs + self.complex + self.heavy:
            family = spec.family.removeprefix("rig_")
            if family in CORED_FAMILIES or family.endswith(("_cored_housing", "_flanged_hub")):
                self.assertTrue(any(c == PART_TYPE_CORE for _, c in spec.parts), spec.name)

    def test_chills_sit_flush_on_the_casting(self) -> None:
        for spec in self.specs + self.complex + self.heavy:
            cast = spec.parts[0][0]
            for chill in (m for m, c in spec.parts if c == PART_TYPE_CHILL):
                gap = np.maximum(cast.bounds[0] - chill.bounds[1], chill.bounds[0] - cast.bounds[1])
                self.assertLessEqual(float(np.max(gap)), 0.02, spec.name)

    def test_heavy_assemblies_carry_a_heavy_section(self) -> None:
        self.assertTrue(self.heavy)
        for spec in self.heavy:
            raster = design_rules.rasterize(spec.parts)
            # A core inside a section saturates and freezes with it, so it counts as section.
            section = np.isin(raster.labels, (design_rules.METAL, design_rules.CORE))
            thickness = 2.0 * float(ndimage.distance_transform_edt(section).max()) * raster.pitch
            self.assertGreaterEqual(spec.max_dim, 18.0, spec.name)
            self.assertGreaterEqual(thickness, 0.8 * HEAVY_SECTION_IN[0], spec.name)

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

    def test_core_faces_do_not_count_as_cooling(self) -> None:
        # A through core leaves 1 in walls; a riser that would feed them if the
        # core faces cooled at half rate is too small once they do not.
        core = trimesh.creation.box(extents=(2.0, 6.0, 1.0))
        core.apply_translation((0.0, 0.0, 1.0))
        parts = [(self.cast, PART_TYPE_CAST), (core, PART_TYPE_CORE), (self._riser(1.8), PART_TYPE_RISER)]
        report = design_rules.check(parts)
        self.assertTrue(any("modulus" in v for v in report.violations), report.violations)
        with mock.patch.object(design_rules, "CORE_FACE_WEIGHT", 0.5):
            half = design_rules.check(parts)
        self.assertFalse(any("modulus" in v for v in half.violations), half.violations)

    def test_a_riser_is_sized_for_the_heaviest_section_not_the_average(self) -> None:
        # A 4 in boss on a 1 in plate: averaged with the plate, a 3 in riser
        # is twice what the zone needs; the boss alone needs more.
        plate = trimesh.creation.box(extents=(16.0, 16.0, 1.0))
        plate.apply_translation((0.0, 0.0, 0.5))
        boss = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
        boss.apply_translation((0.0, 0.0, 2.9))
        cast = trimesh.util.concatenate([plate, boss])
        riser = self._riser(3.0).apply_translation((0.0, 0.0, 2.9))
        report = design_rules.check([(cast, PART_TYPE_CAST), (riser, PART_TYPE_RISER)])
        self.assertTrue(any("modulus" in v for v in report.violations), report.violations)
        with mock.patch.object(design_rules, "HEAVY_SECTION_FRACTION", np.inf):
            average = design_rules.check([(cast, PART_TYPE_CAST), (riser, PART_TYPE_RISER)])
        self.assertTrue(average.ok, average.violations)

    def test_a_riser_may_stand_over_a_narrow_bore_only(self) -> None:
        block = trimesh.creation.box(extents=(8.0, 8.0, 4.0))
        block.apply_translation((0.0, 0.0, 2.0))
        for bore, bridged in ((1.5, True), (4.0, False)):
            core = trimesh.creation.cylinder(radius=bore / 2, height=7.0, sections=48)
            core.apply_translation((0.0, 0.0, 2.0))
            raster = design_rules.rasterize([(block, PART_TYPE_CAST), (core, PART_TYPE_CORE)])
            top, label = design_rules._columns(raster)
            i, j = np.round(-raster.origin[:2] / raster.pitch).astype(int)
            self.assertEqual(label[i, j], design_rules.BORE)
            limit = design_rules._site_limit(raster, top, label, i, j)
            if bridged:
                self.assertGreaterEqual(limit, 7.5, bore)
            else:
                self.assertEqual(limit, 0.0, bore)

    def test_metal_freezing_after_the_last_riser_is_unfed(self) -> None:
        parts = np.array([PART_TYPE_CAST, PART_TYPE_CAST, PART_TYPE_RISER, PART_TYPE_CORE])
        self.assertIsNone(design_rules.unfed_metal(parts, np.array([10.0, 20.0, 30.0, 0.0])))
        reason = design_rules.unfed_metal(parts, np.array([10.0, 40.0, 30.0, 0.0]))
        self.assertIn("50.00% of the casting", reason)
        self.assertIsNone(design_rules.unfed_metal(parts[:2], np.array([10.0, 40.0])))

    def test_feeding_distance_follows_the_steel_rule(self) -> None:
        self.assertAlmostEqual(design_rules.feeding_distance(1.0), 4.5)
        self.assertAlmostEqual(design_rules.feeding_distance(4.0), 12.0)


class AssemblyDatasetTests(unittest.TestCase):
    def test_generate_dataset_runs_an_assembly_as_a_multi_part_job(self) -> None:
        # Seed 1: seed 0's rigging misses the soundness check by 0.4 % of the
        # casting in the explicit sand mold, and the run is rejected.
        spec = next(iter_assemblies(1, seed=1, families=["cored_cylinder"]))
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
