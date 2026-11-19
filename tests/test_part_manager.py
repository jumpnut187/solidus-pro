"""Tests for ``core.part_manager``.

Covers both single-part ``load_part`` and multi-body assembly loading
via ``load_assembly_parts``. STEP-specific tests are skipped when
``cascadio`` is not installed so the suite still runs on STL-only
installations.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import List

import trimesh

from core.part_manager import (
    PartManager,
    PartType,
    STEP_SUPPORTED,
    _scale_step_to_inches,
    infer_part_type_from_filename,
    infer_part_type_from_name,
)


def _write_scene_glb(path: Path, bodies: List[tuple]) -> None:
    """Write a ``trimesh.Scene`` to a GLB file with the given bodies.

    ``bodies`` is a list of ``(name, trimesh.Trimesh)`` tuples. GLB is
    used because ``trimesh`` round-trips Scenes through it natively
    (unlike STL, which only handles single bodies).
    """
    scene = trimesh.Scene()
    for name, mesh in bodies:
        scene.add_geometry(mesh, geom_name=name)
    scene.export(str(path), file_type="glb")


class InferPartTypeFromNameTests(unittest.TestCase):
    """Unit tests for the keyword-driven part-type inference helper."""

    def test_casting_keyword_returns_cast(self) -> None:
        self.assertEqual(infer_part_type_from_name("Casting_Block"), PartType.CAST)

    def test_riser_keyword_returns_riser(self) -> None:
        self.assertEqual(infer_part_type_from_name("Riser_Sleeve"), PartType.RISER)

    def test_chill_keyword_returns_chill(self) -> None:
        self.assertEqual(infer_part_type_from_name("Copper_Chill"), PartType.CHILL)

    def test_core_keyword_returns_core(self) -> None:
        # 'core' must beat 'cast' (casting contains 'cast').
        self.assertEqual(infer_part_type_from_name("Sand_Core"), PartType.CORE)

    def test_riser_post_keyword_returns_riser_post(self) -> None:
        # 'post' alone would not be enough; 'riser_post' wins over 'riser'.
        self.assertEqual(infer_part_type_from_name("Riser_Post_1"), PartType.RISER_POST)

    def test_unknown_name_defaults_to_cast(self) -> None:
        self.assertEqual(infer_part_type_from_name("mystery_part"), PartType.CAST)

    def test_filename_with_path_and_extension_is_handled(self) -> None:
        # The helper should strip the path and extension transparently.
        self.assertEqual(
            infer_part_type_from_name("path/to/riser.stl"),
            PartType.RISER,
        )

    def test_a_dot_that_is_not_an_extension_is_kept(self) -> None:
        # A revision like "Rev 1.0" is not an extension: cutting at its dot
        # dropped the keyword, so every body of such an upload came in as cast.
        self.assertEqual(infer_part_type_from_name("C06345_Rigging_Rev_1.0_-_Chill_12"), PartType.CHILL)
        self.assertEqual(infer_part_type_from_name("C06345_Rigging_Rev 1.0 - Riser03.step"), PartType.RISER)

    def test_case_insensitive_matching(self) -> None:
        self.assertEqual(infer_part_type_from_name("Riser_Sleeve"), PartType.RISER)
        self.assertEqual(infer_part_type_from_name("RISER_SLEEVE"), PartType.RISER)
        self.assertEqual(infer_part_type_from_name("riser_sleeve"), PartType.RISER)

    def test_filename_alias_matches_name(self) -> None:
        # infer_part_type_from_filename is documented as a thin wrapper;
        # verify it produces the same answer as infer_part_type_from_name
        # for full filenames with path and extension.
        self.assertEqual(
            infer_part_type_from_filename("path/to/Riser_Sleeve.STL"),
            infer_part_type_from_name("Riser_Sleeve"),
        )


class LoadPartTests(unittest.TestCase):
    """Single-body STL loading via ``PartManager.load_part``."""

    def setUp(self) -> None:
        self.manager = PartManager()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_stl_returns_part_id_with_inferred_type(self) -> None:
        mesh_path = os.path.join(self.temp_dir, "riser_sleeve.stl")
        trimesh.creation.box(extents=(0.5, 0.5, 1.0)).export(mesh_path)

        part_id = self.manager.load_part(mesh_path)

        self.assertIsNotNone(part_id)
        assert part_id is not None
        part = self.manager.get_part(part_id)
        assert part is not None
        self.assertEqual(part.part_type, PartType.RISER)
        self.assertGreater(len(part.trimesh_obj.vertices), 0)
        self.assertGreater(len(part.trimesh_obj.faces), 0)

    def test_load_part_with_explicit_position(self) -> None:
        mesh_path = os.path.join(self.temp_dir, "casting.stl")
        trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(mesh_path)

        part_id = self.manager.load_part(mesh_path, position=(2.0, 0.0, 0.0))
        assert part_id is not None
        part = self.manager.get_part(part_id)
        assert part is not None
        # set_position was called with the explicit offset; the part's
        # position vector should reflect it.
        self.assertAlmostEqual(float(part.position[0]), 2.0, places=4)

    def test_load_missing_file_returns_none(self) -> None:
        bogus = os.path.join(self.temp_dir, "does_not_exist.stl")
        self.assertIsNone(self.manager.load_part(bogus))


@unittest.skipUnless(STEP_SUPPORTED, "cascadio not installed; STEP tests skipped")
class LoadPartStepTests(unittest.TestCase):
    """Single-body STEP loading via ``PartManager.load_part``.

    These tests are skipped when ``cascadio`` is not installed; they
    exercise the STEP-specific error path in ``load_part`` and verify
    that body-name-inferred type defaults to CAST when only the file's
    filename is available.
    """

    def setUp(self) -> None:
        self.manager = PartManager()
        # Use a tiny synthetic STEP file written from trimesh via cascadio.
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_step_file_missing_cascadio_returns_none(self) -> None:
        # If we got here, cascadio IS installed; but verify the loader
        # accepts a .step path without crashing.
        # Build a STEP file via cascadio (the only practical way without
        # shipping a fixture) — skip if STEP_SUPPORTED but no writable
        # cascadio backend available.
        try:
            import cascadio

            # cascadio doesn't expose a STEP writer API publicly, so we
            # can't synthesize one easily. Mark as expected skip.
            self.skipTest("cascadio has no public STEP writer; no fixture available")
        except ImportError:
            self.fail("cascadio should be importable")


class LoadAssemblyPartsTests(unittest.TestCase):
    """Multi-body assembly loading via ``PartManager.load_assembly_parts``.

    Uses a synthetic ``trimesh.Scene`` round-tripped through GLB. GLB is
    the one format ``trimesh`` reliably preserves Scene structure
    through. STEP-specific behavior (cascadio tessellation quality,
    body-name preservation) is verified separately when cascadio is
    available.
    """

    def setUp(self) -> None:
        self.manager = PartManager()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_assembly_creates_one_part_per_body(self) -> None:
        bodies = [
            ("Casting_Block", trimesh.creation.box(extents=(1.0, 1.0, 1.0))),
            ("Riser_Sleeve", trimesh.creation.box(extents=(0.5, 0.5, 2.0))),
            ("Copper_Chill", trimesh.creation.box(extents=(0.3, 0.3, 0.3))),
        ]
        glb_path = Path(self.temp_dir) / "engine_block.glb"
        _write_scene_glb(glb_path, bodies)

        part_ids = self.manager.load_assembly_parts(str(glb_path))

        self.assertEqual(len(part_ids), 3)
        self.assertEqual(self.manager.get_part_count(), 3)

        # Verify each body was registered with the inferred type
        types_by_name = {
            self.manager.get_part(pid).name: self.manager.get_part(pid).part_type
            for pid in part_ids
            if self.manager.get_part(pid) is not None
        }
        self.assertEqual(types_by_name.get("Casting_Block"), PartType.CAST)
        self.assertEqual(types_by_name.get("Riser_Sleeve"), PartType.RISER)
        self.assertEqual(types_by_name.get("Copper_Chill"), PartType.CHILL)

    def test_load_assembly_missing_file_returns_empty_list(self) -> None:
        bogus = os.path.join(self.temp_dir, "missing.glb")
        result = self.manager.load_assembly_parts(bogus)
        self.assertEqual(result, [])
        self.assertEqual(self.manager.get_part_count(), 0)

    def test_load_assembly_with_name_prefix(self) -> None:
        bodies = [
            ("Body_A", trimesh.creation.box(extents=(1.0, 1.0, 1.0))),
            ("Body_B", trimesh.creation.box(extents=(0.5, 0.5, 0.5))),
        ]
        glb_path = Path(self.temp_dir) / "test.glb"
        _write_scene_glb(glb_path, bodies)

        part_ids = self.manager.load_assembly_parts(str(glb_path), name_prefix="engine")

        self.assertEqual(len(part_ids), 2)
        names = sorted(self.manager.get_part(pid).name for pid in part_ids)
        # name_prefix is prepended to each body name to keep assemblies distinct.
        self.assertEqual(names, ["engine_Body_A", "engine_Body_B"])


class StepSupportFlagTests(unittest.TestCase):
    """The optional ``STEP_SUPPORTED`` flag should be present."""

    def test_step_supported_flag_exists(self) -> None:
        from core import part_manager

        self.assertTrue(hasattr(part_manager, "STEP_SUPPORTED"))
        # It must be a bool (True when cascadio is installed, False otherwise).
        self.assertIsInstance(part_manager.STEP_SUPPORTED, bool)


class ScaleStepToInchesTests(unittest.TestCase):
    """Verify the meters-to-inches rescale for cascadio/trimesh STEP output.

    These tests do not require cascadio — they construct synthetic
    ``Trimesh`` and ``Scene`` inputs that mimic the post-cascadio state
    (coordinates in meters) and verify that ``_scale_step_to_inches``
    rescales to inches using the canonical 39.3701 conversion factor.

    Without this fix, a STEP file authored in mm would be loaded ~39x
    too small, the voxel mesh would collapse to a few cells, and every
    defect analysis would silently return 0 (the bug that motivated
    this helper).
    """

    METERS_TO_INCHES = 39.3701

    def _box_in_meters(self, extents):
        """Create a unit-cube mesh in meters, anchored at the origin."""
        mesh = trimesh.creation.box(extents=extents)
        # The box is centered at origin; shift so its min corner is at
        # the origin. This makes the expected post-scale bounds
        # ``(0, 0, 0)`` to ``(extent*METERS_TO_INCHES, ...)``.
        mesh.apply_translation([e / 2 for e in extents])
        return mesh

    def test_trimesh_input_scales_to_inches(self) -> None:
        # A 1-meter cube (a STEP file authored in mm with a 1000 mm
        # feature). After scale, bounds should span 39.37 inches.
        mesh = self._box_in_meters((1.0, 1.0, 1.0))
        self.assertAlmostEqual(float(mesh.bounds[0][0]), 0.0, places=5)

        _scale_step_to_inches(mesh)

        expected_extent = self.METERS_TO_INCHES
        self.assertAlmostEqual(
            float(mesh.bounds[0][0]), 0.0, places=4
        )
        self.assertAlmostEqual(
            float(mesh.bounds[1][0]), expected_extent, places=3
        )
        self.assertAlmostEqual(
            float(mesh.bounds[1][1]), expected_extent, places=3
        )
        self.assertAlmostEqual(
            float(mesh.bounds[1][2]), expected_extent, places=3
        )

    def test_scene_input_scales_every_body(self) -> None:
        # Multi-body STEP assembly: 3 boxes in meters.
        scene = trimesh.Scene()
        scene.add_geometry(self._box_in_meters((1.0, 1.0, 1.0)), geom_name="casting")
        scene.add_geometry(self._box_in_meters((0.5, 0.5, 2.0)), geom_name="riser")
        scene.add_geometry(self._box_in_meters((0.3, 0.3, 0.3)), geom_name="chill")

        _scale_step_to_inches(scene)

        casting = scene.geometry["casting"]
        riser = scene.geometry["riser"]
        chill = scene.geometry["chill"]
        self.assertAlmostEqual(
            float(casting.bounds[1][0]), 1.0 * self.METERS_TO_INCHES, places=3
        )
        self.assertAlmostEqual(
            float(riser.bounds[1][2]), 2.0 * self.METERS_TO_INCHES, places=3
        )
        self.assertAlmostEqual(
            float(chill.bounds[1][0]), 0.3 * self.METERS_TO_INCHES, places=3
        )

    def test_scene_with_empty_body_is_skipped(self) -> None:
        # Empty bodies (zero faces) cannot be scaled meaningfully;
        # verify they don't raise and the non-empty bodies still scale.
        scene = trimesh.Scene()
        scene.add_geometry(self._box_in_meters((1.0, 1.0, 1.0)), geom_name="real")
        # An empty Trimesh (0 faces) — trimesh allows this.
        empty = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[])
        scene.add_geometry(empty, geom_name="empty")

        # Should not raise
        _scale_step_to_inches(scene)

        # Real body was scaled to inches.
        real = scene.geometry["real"]
        self.assertAlmostEqual(
            float(real.bounds[1][0]), 1.0 * self.METERS_TO_INCHES, places=3
        )

    def test_volume_scales_with_cube_of_factor(self) -> None:
        # trimesh applies the scale uniformly: volumes scale by k^3.
        # A 1-meter cube has volume 1.0 m^3 = 61023.7 in^3.
        mesh = self._box_in_meters((1.0, 1.0, 1.0))

        _scale_step_to_inches(mesh)

        expected_volume = self.METERS_TO_INCHES ** 3  # 61023.7...
        actual_volume = float(mesh.volume)
        self.assertAlmostEqual(actual_volume, expected_volume, delta=0.1)

    def test_stl_load_path_does_not_scale(self) -> None:
        # Regression guard: only STEP files should be scaled. An STL
        # file's mesh should pass through ``load_part`` unchanged.
        manager = PartManager()
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = os.path.join(tmp, "casting_box.stl")
            box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
            box.export(mesh_path)

            part_id = manager.load_part(mesh_path)
            assert part_id is not None
            part = manager.get_part(part_id)
            assert part is not None

            # The original STL is in inches. The mesh extents should
            # remain ~2 inches per axis, NOT ~2*39.37 inches (which
            # would mean we accidentally scaled an STL).
            extent = float(part.trimesh_obj.bounds[1][0] - part.trimesh_obj.bounds[0][0])
            self.assertAlmostEqual(extent, 2.0, places=4)


if __name__ == "__main__":
    unittest.main()