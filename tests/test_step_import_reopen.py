"""Regression tests for STEP import: project reopen, unit scaling, naming.

Covers:
- a multi-body STEP project reopens as distinct bodies, not N copies of body 1;
- a legacy project (no recorded body) reopens by saved order;
- gating part types survive reopen;
- a single-body STEP is named and typed from its file stem;
- STEP length units: inch-authored files and an mm re-declaration of one.

The multi-body file is built by merging the single-body ``Test_Model``
STEP files into one exchange file, which cascadio reads as five bodies.
"""

from __future__ import annotations

import glob
import json
import os
import re
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh

from core import part_manager as part_manager_module
from core.part_manager import (
    PartManager,
    PartType,
    STEP_SUPPORTED,
    _scale_step_to_inches,
    part_type_from_value,
    step_declared_length_unit_inches,
    tessellation_tolerance_mm,
)
from ui.project_state import build_project_state, load_project_state

STEP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Test_Model",
    "Filling_Test",
)
CASTING_STEP = os.path.join(STEP_DIR, "Feeding Distance Test - Casting.step")
SPRUE_STEP = os.path.join(STEP_DIR, "Feeding Distance Test - Down Sprue.step")
ASSEMBLY_SOURCES = [
    os.path.join(STEP_DIR, f"Feeding Distance Test - {name}.step")
    for name in ("Casting", "Down Sprue", "Gate", "Riser", "Runner")
]
HAVE_STEP_FIXTURES = STEP_SUPPORTED and all(os.path.exists(p) for p in ASSEMBLY_SOURCES)

VOXEL_IN = 0.20

_INCH_UNIT_BLOCK = re.compile(
    r"\(\s*CONVERSION_BASED_UNIT\('INCH',#\d+\)\s*LENGTH_UNIT\(\)\s*NAMED_UNIT\(#\d+\)\s*\)"
)
_MM_UNIT_BLOCK = "(\nLENGTH_UNIT()\nNAMED_UNIT(*)\nSI_UNIT(.MILLI.,.METRE.)\n)"


def merge_step_files(sources, destination):
    """Concatenate single-body STEP files into one multi-body STEP file.

    Entity ids of each later file are offset past the previous ones; every
    file keeps its own product and unit context.
    """
    header = None
    data_sections = []
    offset = 0
    for source in sources:
        with open(source, "r", encoding="utf-8") as handle:
            text = handle.read()
        head, rest = text.split("DATA;", 1)
        data = rest.rsplit("ENDSEC;", 1)[0]
        if header is None:
            header = head
        ids = [int(value) for value in re.findall(r"#(\d+)", data)]
        data_sections.append(
            re.sub(r"#(\d+)", lambda m, o=offset: f"#{int(m.group(1)) + o}", data)
        )
        offset += max(ids) + 1
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(header + "DATA;" + "".join(data_sections) + "ENDSEC;\nEND-ISO-10303-21;\n")


def write_mm_variant(source, destination):
    """Re-declare an inch STEP file as millimetres; coordinates unchanged."""
    with open(source, "r", encoding="utf-8") as handle:
        text = handle.read()
    replaced, count = _INCH_UNIT_BLOCK.subn(_MM_UNIT_BLOCK, text)
    if count != 1:
        raise AssertionError(f"expected one inch unit block, found {count}")
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(replaced)


def save_and_reload(manager):
    """Round-trip a PartManager through the desktop project format."""
    payload = build_project_state(
        manager,
        alloy_name="WCB",
        voxel_size=VOXEL_IN,
        refinement_level=3,
        memory_mode="auto",
        analysis_options={},
        performance_options={},
        mesh_setting_origin="manual",
        current_vis_mode="shrinkage",
        ui_state={},
    ).to_dict()
    return json.loads(json.dumps(payload))


def warning_messages(mock_warning):
    return [str(call.args[0]) for call in mock_warning.call_args_list]


@unittest.skipUnless(HAVE_STEP_FIXTURES, "cascadio or STEP fixtures not available")
class MultiBodyProjectReopenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.mkdtemp()
        cls.assembly_path = os.path.join(cls.temp_dir, "Gating Assembly.step")
        merge_step_files(ASSEMBLY_SOURCES, cls.assembly_path)

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _loaded_assembly(self):
        manager = PartManager()
        ids = manager.load_assembly_parts(self.assembly_path, voxel_size_in=VOXEL_IN)
        self.assertEqual(len(ids), 5, "merged fixture should load as five bodies")
        return manager

    def _customise(self, manager):
        """Give every part a distinct, non-default saved state."""
        for offset, part in enumerate(manager.get_all_parts()):
            part.set_position(1.0 + offset, -2.0 * offset, 0.5 * offset)
            part.set_rotation(0.0, 0.0, 10.0 * offset)
            part.visible = offset != 2
        riser = next(p for p in manager.get_all_parts() if p.body_name == "Riser")
        riser.is_open_top = True
        gate = next(p for p in manager.get_all_parts() if p.body_name == "Gate")
        gate.name = "Ingate renamed"

    def test_reopen_yields_distinct_bodies_with_saved_state(self) -> None:
        original = self._loaded_assembly()
        self._customise(original)
        saved = load_project_state(save_and_reload(original)).to_dict()

        reopened = PartManager()
        with mock.patch("trimesh.load", wraps=trimesh.load) as load_spy:
            ids = reopened.restore_parts(saved["parts"], voxel_size_in=VOXEL_IN)
        self.assertTrue(all(ids))
        self.assertEqual(load_spy.call_count, 1, "STEP file must be loaded once per file")

        before = original.get_all_parts()
        after = reopened.get_all_parts()
        self.assertEqual(len(after), 5)
        self.assertEqual([p.body_name for p in after], [p.body_name for p in before])
        self.assertEqual(len({p.body_name for p in after}), 5)
        for old, new in zip(before, after):
            with self.subTest(body=old.body_name):
                self.assertEqual(new.name, old.name)
                self.assertIs(new.part_type, old.part_type)
                self.assertEqual(new.body_index, old.body_index)
                np.testing.assert_allclose(new.position, old.position)
                np.testing.assert_allclose(new.rotation, old.rotation)
                self.assertEqual(new.visible, old.visible)
                self.assertEqual(new.is_open_top, old.is_open_top)
                np.testing.assert_allclose(
                    np.sort(new.trimesh_obj.extents),
                    np.sort(old.trimesh_obj.extents),
                    rtol=1e-6,
                )
                self.assertAlmostEqual(
                    new.tessellation_tolerance_mm, tessellation_tolerance_mm(VOXEL_IN)
                )

        by_body = {p.body_name: p for p in after}
        self.assertIs(by_body["Down Sprue"].part_type, PartType.SPRUE)
        self.assertIs(by_body["Gate"].part_type, PartType.GATE)
        self.assertIs(by_body["Runner"].part_type, PartType.RUNNER)
        self.assertIs(by_body["Riser"].part_type, PartType.RISER)
        self.assertTrue(by_body["Riser"].is_open_top)
        self.assertEqual(by_body["Gate"].name, "Ingate renamed")

    def test_reopened_parts_still_retessellate(self) -> None:
        original = self._loaded_assembly()
        saved = load_project_state(save_and_reload(original)).to_dict()
        reopened = PartManager()
        reopened.restore_parts(saved["parts"], voxel_size_in=0.40)
        self.assertEqual(reopened.retessellate_for_voxel_size(0.05), 5)
        for part in reopened.get_all_parts():
            self.assertAlmostEqual(
                part.tessellation_tolerance_mm, tessellation_tolerance_mm(0.05)
            )

    def test_legacy_project_without_body_names_reopens_by_order(self) -> None:
        original = self._loaded_assembly()
        payload = save_and_reload(original)
        for part in payload["parts"]:
            del part["body_name"]
            del part["body_index"]
        saved = load_project_state(payload).to_dict()
        self.assertTrue(all(p["body_name"] is None for p in saved["parts"]))

        reopened = PartManager()
        with mock.patch.object(part_manager_module.logger, "warning") as warn:
            ids = reopened.restore_parts(saved["parts"], voxel_size_in=VOXEL_IN)
        self.assertTrue(all(ids))
        self.assertEqual(
            [p.body_name for p in reopened.get_all_parts()],
            [p.body_name for p in original.get_all_parts()],
        )
        self.assertTrue(
            any("saved order" in message for message in warning_messages(warn)),
            "legacy reopen must warn that bodies were matched by order",
        )

    def test_unmatched_body_is_reported_not_substituted(self) -> None:
        original = self._loaded_assembly()
        saved = load_project_state(save_and_reload(original)).to_dict()
        saved["parts"][0]["body_name"] = "No Such Body"
        saved["parts"][0]["body_index"] = 99
        reopened = PartManager()
        ids = reopened.restore_parts(saved["parts"], voxel_size_in=VOXEL_IN)
        self.assertIsNone(ids[0])
        self.assertTrue(all(ids[1:]))

    def test_duplicated_body_reopens_as_the_same_body(self) -> None:
        """Review P1: a duplicate used to reopen as another body, or vanish."""
        original = self._loaded_assembly()
        riser = next(p for p in original.get_all_parts() if p.body_name == "Riser")
        riser.is_open_top = True
        copy_id = original.duplicate_part(riser.id)
        copy = original.get_part(copy_id)
        self.assertEqual(copy.body_name, "Riser")
        self.assertEqual(copy.body_index, riser.body_index)
        self.assertIs(copy.part_type, PartType.RISER)
        self.assertTrue(copy.is_open_top)
        self.assertIsNot(copy.trimesh_obj, riser.trimesh_obj)

        # Also remove a body so saved positions no longer line up with the
        # scene -- the case that loaded the wrong geometry under the copy's name.
        gate = next(p for p in original.get_all_parts() if p.body_name == "Gate")
        original.remove_part(gate.id)

        saved = load_project_state(save_and_reload(original)).to_dict()
        reopened = PartManager()
        ids = reopened.restore_parts(saved["parts"], voxel_size_in=VOXEL_IN)
        self.assertTrue(all(ids))
        reopened_copy = reopened.get_part(ids[-1])
        self.assertEqual(reopened_copy.name, copy.name)
        self.assertEqual(reopened_copy.body_name, "Riser")
        np.testing.assert_allclose(
            np.sort(reopened_copy.trimesh_obj.extents),
            np.sort(riser.trimesh_obj.extents),
            rtol=1e-6,
        )

    def test_mixed_group_ref_without_body_gets_first_body(self) -> None:
        """A body-less ref among recorded ones is not matched by position."""
        original = self._loaded_assembly()
        saved = load_project_state(save_and_reload(original)).to_dict()
        extra = dict(saved["parts"][0])
        extra["name"] = "Old duplicate"
        extra["body_name"] = None
        extra["body_index"] = None
        parts = saved["parts"] + [extra]

        reopened = PartManager()
        with mock.patch.object(part_manager_module.logger, "warning") as warn:
            ids = reopened.restore_parts(parts, voxel_size_in=VOXEL_IN)
        self.assertTrue(all(ids), "a sixth ref must not be dropped for running past the body count")
        first_body = original.get_all_parts()[0].body_name
        self.assertEqual(reopened.get_part(ids[-1]).body_name, first_body)
        self.assertTrue(any("first body" in m for m in warning_messages(warn)))

    def test_renamed_body_falls_back_to_index_with_warning(self) -> None:
        original = self._loaded_assembly()
        saved = load_project_state(save_and_reload(original)).to_dict()
        saved["parts"][1]["body_name"] = "Renamed In CAD"
        reopened = PartManager()
        with mock.patch.object(part_manager_module.logger, "warning") as warn:
            ids = reopened.restore_parts(saved["parts"], voxel_size_in=VOXEL_IN)
        self.assertTrue(all(ids))
        self.assertTrue(
            any("Renamed In CAD" in m and "by position" in m for m in warning_messages(warn))
        )

    def test_load_part_on_assembly_does_not_false_warn_units(self) -> None:
        """Review P2: load_part keeps body 1 only; the whole-file check must not run."""
        manager = PartManager()
        with mock.patch.object(part_manager_module.logger, "warning") as warn:
            self.assertIsNotNone(manager.load_part(self.assembly_path))
        self.assertFalse(any("unit check failed" in m for m in warning_messages(warn)))


class GatingTypeReopenTests(unittest.TestCase):
    def test_gating_part_types_survive_reopen(self) -> None:
        gating = [
            PartType.POUR_CUP,
            PartType.SPRUE,
            PartType.RUNNER,
            PartType.FILTER,
            PartType.GATE,
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PartManager()
            for index, part_type in enumerate(gating):
                path = os.path.join(temp_dir, f"block_{index}.stl")
                trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(path)
                part = manager.get_part(manager.load_part(path))
                part.set_part_type(part_type)

            saved = load_project_state(save_and_reload(manager)).to_dict()
            self.assertEqual(
                [p["part_type"] for p in saved["parts"]], [t.value for t in gating]
            )

            reopened = PartManager()
            ids = reopened.restore_parts(saved["parts"])
            self.assertTrue(all(ids))
            self.assertEqual([p.part_type for p in reopened.get_all_parts()], gating)

    def test_unknown_part_type_falls_back_to_cast_with_warning(self) -> None:
        with mock.patch.object(part_manager_module.logger, "warning") as warn:
            self.assertIs(part_type_from_value("teapot"), PartType.CAST)
        self.assertTrue(warn.called)

    def test_every_part_type_value_round_trips(self) -> None:
        for part_type in PartType:
            self.assertIs(part_type_from_value(part_type.value), part_type)

    def test_stl_reload_keeps_saved_name_and_transform(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "casting.stl")
            trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(path)
            manager = PartManager()
            part = manager.get_part(manager.load_part(path, name="My Casting"))
            part.set_position(1.0, 2.0, 3.0)
            part.set_scale(1.5)
            part.color = (0.1, 0.2, 0.3)
            saved = load_project_state(save_and_reload(manager)).to_dict()
            self.assertIsNone(saved["parts"][0]["body_name"])

            reopened = PartManager()
            reopened.restore_parts(saved["parts"])
            restored = reopened.get_all_parts()[0]
            self.assertEqual(restored.name, "My Casting")
            self.assertIsNone(restored.body_name)
            np.testing.assert_allclose(restored.position, [1.0, 2.0, 3.0])
            self.assertEqual(restored.scale, 1.5)
            self.assertEqual(restored.color, (0.1, 0.2, 0.3))


@unittest.skipUnless(HAVE_STEP_FIXTURES, "cascadio or STEP fixtures not available")
class SingleBodyStepNamingTests(unittest.TestCase):
    def test_single_body_is_named_and_typed_from_file_stem(self) -> None:
        manager = PartManager()
        ids = manager.load_assembly_parts(SPRUE_STEP, voxel_size_in=VOXEL_IN)
        self.assertEqual(len(ids), 1)
        part = manager.get_part(ids[0])
        self.assertEqual(part.name, "Feeding Distance Test - Down Sprue")
        self.assertIs(part.part_type, PartType.SPRUE)
        # Provenance keeps the real CAD body name for re-tessellation.
        self.assertEqual(part.body_name, "Down Sprue")
        self.assertEqual(part.body_index, 0)

    def test_single_body_name_matches_load_part(self) -> None:
        via_assembly = PartManager()
        via_part = PartManager()
        assembly_part = via_assembly.get_part(
            via_assembly.load_assembly_parts(CASTING_STEP, voxel_size_in=VOXEL_IN)[0]
        )
        single_part = via_part.get_part(via_part.load_part(CASTING_STEP))
        self.assertEqual(assembly_part.name, single_part.name)
        self.assertIs(assembly_part.part_type, single_part.part_type)


class DeclaredLengthUnitTests(unittest.TestCase):
    def _units(self, unit_entities):
        return step_declared_length_unit_inches(
            "#1=(GEOMETRIC_REPRESENTATION_CONTEXT(3)"
            "GLOBAL_UNIT_ASSIGNED_CONTEXT((#2))"
            "REPRESENTATION_CONTEXT('',''));\n" + unit_entities
        )

    def test_si_millimetre(self) -> None:
        self.assertAlmostEqual(
            self._units("#2=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));"),
            1.0 / 25.4,
        )

    def test_si_metre_without_prefix(self) -> None:
        self.assertAlmostEqual(
            self._units("#2=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.));"),
            1000.0 / 25.4,
        )

    def test_conversion_based_foot(self) -> None:
        self.assertAlmostEqual(
            self._units(
                "#2=(CONVERSION_BASED_UNIT('FOOT',#3)LENGTH_UNIT()NAMED_UNIT(#4));\n"
                "#3=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(304.8),#5);\n"
                "#5=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));"
            ),
            12.0,
        )

    def test_unnamed_conversion_follows_its_measure(self) -> None:
        self.assertAlmostEqual(
            self._units(
                "#2=(CONVERSION_BASED_UNIT('BARLEYCORN',#3)LENGTH_UNIT()NAMED_UNIT(#4));\n"
                "#3=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(8.4666),#5);\n"
                "#5=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));"
            ),
            8.4666 / 25.4,
        )

    def test_missing_unit_is_none(self) -> None:
        self.assertIsNone(step_declared_length_unit_inches("#1=CARTESIAN_POINT('',(0.,0.,0.));"))

    @unittest.skipUnless(os.path.exists(CASTING_STEP), "STEP fixture not available")
    def test_inch_fixture_reads_as_inch_not_its_millimetre_base(self) -> None:
        """An inch file also carries SI_UNIT(.MILLI.,.METRE.) as the inch's base."""
        with open(CASTING_STEP, "r", encoding="utf-8") as handle:
            self.assertAlmostEqual(step_declared_length_unit_inches(handle.read()), 1.0)


@unittest.skipUnless(HAVE_STEP_FIXTURES, "cascadio or STEP fixtures not available")
class StepUnitScalingTests(unittest.TestCase):
    """The casting fixture is a 12 x 6 x 1 box in its declared unit."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.mm_path = os.path.join(self.temp_dir, "Casting mm.step")
        write_mm_variant(CASTING_STEP, self.mm_path)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _extents(self, path):
        manager = PartManager()
        with mock.patch.object(part_manager_module.logger, "warning") as warn:
            ids = manager.load_assembly_parts(path, voxel_size_in=VOXEL_IN)
        self.assertEqual(len(ids), 1)
        unit_warnings = [m for m in warning_messages(warn) if "unit" in m.lower()]
        self.assertEqual(unit_warnings, [], "correct geometry must not warn")
        return np.sort(manager.get_part(ids[0]).trimesh_obj.extents)

    def test_inch_file_loads_at_declared_size(self) -> None:
        np.testing.assert_allclose(self._extents(CASTING_STEP), [1.0, 6.0, 12.0], rtol=1e-4)

    def test_mm_variant_loads_25_4_times_smaller(self) -> None:
        np.testing.assert_allclose(
            self._extents(self.mm_path), np.array([1.0, 6.0, 12.0]) / 25.4, rtol=1e-4
        )

    def test_every_inch_fixture_passes_the_unit_check(self) -> None:
        for path in sorted(glob.glob(os.path.join(STEP_DIR, "*.step"))):
            with self.subTest(path=os.path.basename(path)):
                self._extents(path)

    def test_mis_scaled_geometry_warns(self) -> None:
        """Geometry 25.4x too small -- an inch file read as mm -- is flagged."""
        scene = trimesh.load(CASTING_STEP, force="scene")
        for body in scene.geometry.values():
            body.apply_scale(1.0 / 25.4)
        with mock.patch.object(part_manager_module.logger, "warning") as warn:
            _scale_step_to_inches(scene, CASTING_STEP)
        self.assertTrue(any("unit check failed" in m for m in warning_messages(warn)))

    def test_mis_scaled_large_geometry_warns(self) -> None:
        """Geometry 25.4x too large -- an mm file read as inches -- is flagged."""
        scene = trimesh.load(self.mm_path, force="scene")
        for body in scene.geometry.values():
            body.apply_scale(25.4)
        with mock.patch.object(part_manager_module.logger, "warning") as warn:
            _scale_step_to_inches(scene, self.mm_path)
        self.assertTrue(any("unit check failed" in m for m in warning_messages(warn)))

    def test_unreadable_unit_keeps_scale_and_warns(self) -> None:
        with open(CASTING_STEP, "r", encoding="utf-8") as handle:
            text = handle.read()
        stripped = os.path.join(self.temp_dir, "no unit.step")
        with open(stripped, "w", encoding="utf-8") as handle:
            handle.write(_INCH_UNIT_BLOCK.sub("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.))", text))
        mesh = trimesh.creation.box(extents=(0.3048, 0.1524, 0.0254))
        with mock.patch.object(part_manager_module.logger, "warning") as warn:
            _scale_step_to_inches(mesh, stripped)
        np.testing.assert_allclose(np.sort(mesh.extents), [1.0, 6.0, 12.0], rtol=1e-4)
        self.assertTrue(
            any("declared length unit" in m for m in warning_messages(warn))
        )

    def test_unit_check_skips_files_over_size_limit(self) -> None:
        scene = trimesh.load(CASTING_STEP, force="scene")
        for body in scene.geometry.values():
            body.apply_scale(1.0 / 25.4)  # would fail the check if it ran
        with mock.patch.object(part_manager_module, "STEP_SCALE_CHECK_MAX_BYTES", 10), \
                mock.patch("builtins.open", side_effect=AssertionError("file was read")), \
                mock.patch.object(part_manager_module.logger, "warning") as warn:
            _scale_step_to_inches(scene, CASTING_STEP)
        self.assertEqual(warning_messages(warn), [])

    def test_no_filepath_keeps_legacy_positional_call(self) -> None:
        """web/payloads.py calls _scale_step_to_inches(geometry) positionally."""
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        _scale_step_to_inches(mesh)
        np.testing.assert_allclose(mesh.extents, [39.3701] * 3, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
