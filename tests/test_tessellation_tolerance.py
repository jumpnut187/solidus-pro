"""Tests for tying STEP tessellation density to the voxel size.

STEP bodies arrive as B-reps and have to be tessellated. cascadio's default
deflection is 0.01 mm, which turns a plain cylindrical riser into 18,004
triangles -- detail no casting mesh can represent, and which containment
testing then ray-casts against for every candidate voxel. Measured on
``Test_Model/Filling_Test``: 84 % of mesh-build time went there, and three
over-tessellated bodies accounted for 13 of those 16.5 seconds.

Tying the deflection to the voxel size makes the CAD exactly as detailed as
the grid can resolve. End to end on that assembly at a 0.20 in voxel:
16.2 s -> 4.4 s, 54,558 faces -> 5,968, with 0.18 % of voxels changing part.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from core.part_manager import (
    MAX_TESSELLATION_TOLERANCE_MM,
    MIN_TESSELLATION_TOLERANCE_MM,
    TESSELLATION_VOXELS_PER_DEFLECTION,
    PartManager,
    PartType,
    tessellation_tolerance_mm,
)

STEP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Test_Model",
    "Filling_Test",
)
RISER_STEP = os.path.join(STEP_DIR, "Feeding Distance Test - Riser.step")


class ToleranceRuleTests(unittest.TestCase):
    def test_tolerance_is_a_fixed_fraction_of_the_voxel(self) -> None:
        # 0.20 in = 5.08 mm; at 1/50 that is 0.1016 mm.
        self.assertAlmostEqual(tessellation_tolerance_mm(0.20), 0.1016, places=6)

    def test_it_scales_with_the_voxel_size(self) -> None:
        fine = tessellation_tolerance_mm(0.10)
        coarse = tessellation_tolerance_mm(0.20)
        self.assertAlmostEqual(coarse / fine, 2.0, places=6)

    def test_the_ratio_is_the_declared_one(self) -> None:
        """Guards against the constant and the arithmetic drifting apart."""
        voxel_in = 0.30
        self.assertAlmostEqual(
            tessellation_tolerance_mm(voxel_in),
            voxel_in * 25.4 / TESSELLATION_VOXELS_PER_DEFLECTION,
            places=9,
        )

    def test_a_fine_grid_never_asks_for_more_triangles_than_the_old_default(
        self,
    ) -> None:
        """The floor is cascadio's own default, so this can only ever help."""
        self.assertEqual(
            tessellation_tolerance_mm(0.001), MIN_TESSELLATION_TOLERANCE_MM
        )

    def test_a_coarse_grid_is_capped(self) -> None:
        """
        Past the ceiling, chord error on a small feature stops being a
        rounding difference and starts changing the shape the solver sees.
        """
        self.assertEqual(
            tessellation_tolerance_mm(100.0), MAX_TESSELLATION_TOLERANCE_MM
        )

    def test_no_voxel_size_leaves_the_loader_alone(self) -> None:
        self.assertIsNone(tessellation_tolerance_mm(None))

    def test_nonsense_voxel_sizes_leave_the_loader_alone(self) -> None:
        for value in (0.0, -1.0, float("nan"), float("inf"), "big", None):
            with self.subTest(value=value):
                self.assertIsNone(tessellation_tolerance_mm(value))


@unittest.skipUnless(os.path.exists(RISER_STEP), "STEP fixture not available")
class StepLoadingTests(unittest.TestCase):
    """The rule has to actually reach cascadio, not just compute a number."""

    def _load(self, voxel_size_in):
        pm = PartManager()
        ids = pm.load_assembly_parts(RISER_STEP, voxel_size_in=voxel_size_in)
        self.assertTrue(ids, "riser STEP failed to load")
        return pm, pm.parts[ids[0]]

    def test_a_coarser_voxel_gives_fewer_triangles(self) -> None:
        _, fine = self._load(0.05)
        _, coarse = self._load(0.40)
        self.assertLess(
            len(coarse.trimesh_obj.faces),
            len(fine.trimesh_obj.faces),
            "tolerance did not reach the tessellator",
        )

    def test_the_default_is_the_expensive_one(self) -> None:
        """Passing no voxel size must not silently change existing behaviour."""
        _, tied = self._load(0.20)
        pm = PartManager()
        ids = pm.load_assembly_parts(RISER_STEP)
        untied = pm.parts[ids[0]]
        self.assertGreater(
            len(untied.trimesh_obj.faces), len(tied.trimesh_obj.faces)
        )

    def test_geometry_survives_the_coarser_tessellation(self) -> None:
        """
        The riser is 3 x 3 x 5 inches. Tessellation may not move that.
        Checked against the known part size, not against another run of the
        same code.
        """
        _, part = self._load(0.20)
        extents = np.sort(part.trimesh_obj.extents)
        np.testing.assert_allclose(extents, [3.0, 3.0, 5.0], rtol=0.01)

    def test_volume_is_preserved_within_a_fraction_of_a_percent(self) -> None:
        _, fine = self._load(0.02)   # clamped to the 0.01 mm floor
        _, coarse = self._load(0.20)
        rel = abs(coarse.trimesh_obj.volume - fine.trimesh_obj.volume)
        rel /= abs(fine.trimesh_obj.volume)
        self.assertLess(rel, 0.005, f"volume moved by {rel:.3%}")

    def test_provenance_is_recorded_for_retessellation(self) -> None:
        _, part = self._load(0.20)
        self.assertIsNotNone(part.body_name)
        self.assertIsNotNone(part.body_index)
        self.assertAlmostEqual(part.tessellation_tolerance_mm, 0.1016, places=6)


@unittest.skipUnless(os.path.exists(RISER_STEP), "STEP fixture not available")
class RetessellationTests(unittest.TestCase):
    """
    CAD is loaded when the file is opened; the voxel size is chosen later.
    Without re-tessellation, whichever value happened to be current at open
    time would decide the geometry forever.
    """

    def _manager(self, voxel_size_in):
        pm = PartManager()
        ids = pm.load_assembly_parts(RISER_STEP, voxel_size_in=voxel_size_in)
        return pm, pm.parts[ids[0]]

    def test_choosing_a_finer_grid_refines_the_geometry(self) -> None:
        pm, part = self._manager(0.40)
        before = len(part.trimesh_obj.faces)
        changed = pm.retessellate_for_voxel_size(0.05)
        self.assertEqual(changed, 1)
        self.assertGreater(
            len(part.trimesh_obj.faces),
            before,
            "a finer voxel size did not refine the CAD",
        )

    def test_choosing_a_coarser_grid_keeps_the_finer_geometry(self) -> None:
        """Going coarser would discard fidelity to save time already spent."""
        pm, part = self._manager(0.05)
        before = len(part.trimesh_obj.faces)
        changed = pm.retessellate_for_voxel_size(0.40)
        self.assertEqual(changed, 0)
        self.assertEqual(len(part.trimesh_obj.faces), before)

    def test_part_identity_survives(self) -> None:
        """id, name, assigned type and transform must all be preserved."""
        pm, part = self._manager(0.40)
        part.set_part_type(PartType.RISER)
        part.position = np.array([1.0, 2.0, 3.0])
        part_id, name = part.id, part.name

        pm.retessellate_for_voxel_size(0.05)

        self.assertIs(pm.parts[part_id], part)
        self.assertEqual(part.name, name)
        self.assertIs(part.part_type, PartType.RISER)
        np.testing.assert_allclose(part.position, [1.0, 2.0, 3.0])

    def test_the_transform_cache_is_invalidated(self) -> None:
        """
        A stale cached transformed mesh would hand the solver the old
        triangles while the Part reports the new ones.
        """
        pm, part = self._manager(0.40)
        part.position = np.array([1.0, 0.0, 0.0])
        stale = part.get_transformed_mesh()
        stale_faces = len(stale.faces)

        pm.retessellate_for_voxel_size(0.05)
        fresh = part.get_transformed_mesh()

        self.assertNotEqual(len(fresh.faces), stale_faces)
        self.assertEqual(len(fresh.faces), len(part.trimesh_obj.faces))

    def test_geometry_still_matches_the_real_part(self) -> None:
        pm, part = self._manager(0.40)
        pm.retessellate_for_voxel_size(0.05)
        extents = np.sort(part.get_transformed_mesh().extents)
        np.testing.assert_allclose(extents, [3.0, 3.0, 5.0], rtol=0.01)

    def test_no_voxel_size_is_a_no_op(self) -> None:
        pm, part = self._manager(0.40)
        before = len(part.trimesh_obj.faces)
        self.assertEqual(pm.retessellate_for_voxel_size(None), 0)
        self.assertEqual(len(part.trimesh_obj.faces), before)

    def test_a_missing_source_file_is_survived(self) -> None:
        """Losing the CAD must not lose the part."""
        pm, part = self._manager(0.40)
        part.filepath = os.path.join(STEP_DIR, "does-not-exist.step")
        before = len(part.trimesh_obj.faces)
        self.assertEqual(pm.retessellate_for_voxel_size(0.05), 0)
        self.assertEqual(len(part.trimesh_obj.faces), before)


if __name__ == "__main__":
    unittest.main()
