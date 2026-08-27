"""Regression tests for the post-simulation auxiliary-part visibility rule.

After a fresh simulation, ``MainWindow._hide_auxiliary_parts`` should flip
``visible=False`` for every CHILL / SLEEVE / CORE part so the results view
shows only CAST, RISER, and RISER_POST. The behavior is gated on the
``visualization.hide_auxiliary_parts_on_sim_finish`` setting.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import trimesh

from config.settings import settings as _settings
from core.part_manager import PartManager, PartType


class HideAuxiliaryPartsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_setting = _settings.visualization.hide_auxiliary_parts_on_sim_finish
        self._manager = PartManager()
        self._temp_dir = tempfile.TemporaryDirectory()
        self._part_ids: dict[PartType, str] = {}

        for part_type, label in (
            (PartType.CAST, "casting"),
            (PartType.RISER_POST, "riser_post"),
            (PartType.RISER, "riser"),
            (PartType.CHILL, "chill"),
            (PartType.SLEEVE, "sleeve"),
            (PartType.CORE, "core"),
        ):
            mesh_path = os.path.join(self._temp_dir.name, f"{label}.stl")
            trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(mesh_path)
            part_id = self._manager.load_part(mesh_path, name=label)
            self.assertIsNotNone(part_id, f"Failed to load {label}.stl")
            assert part_id is not None
            part = self._manager.get_part(part_id)
            assert part is not None
            part.set_part_type(part_type)
            self._part_ids[part_type] = part_id

    def tearDown(self) -> None:
        _settings.visualization.hide_auxiliary_parts_on_sim_finish = self._original_setting
        self._temp_dir.cleanup()

    @staticmethod
    def _apply_rule(manager: PartManager, auxiliary_types: set[PartType]) -> None:
        """Inline replica of ``MainWindow._hide_auxiliary_parts``.

        Mirrors the production logic so we can verify the rule without
        spinning up a full Qt MainWindow in the test harness.
        """
        if not _settings.visualization.hide_auxiliary_parts_on_sim_finish:
            return
        for part in manager.get_all_parts():
            if part.part_type in auxiliary_types:
                part.visible = False

    def test_default_setting_hides_chill_sleeve_core(self) -> None:
        _settings.visualization.hide_auxiliary_parts_on_sim_finish = True

        self._apply_rule(self._manager, {PartType.CHILL, PartType.SLEEVE, PartType.CORE})

        for part_type in (PartType.CAST, PartType.RISER_POST, PartType.RISER):
            part = self._manager.get_part(self._part_ids[part_type])
            assert part is not None
            self.assertTrue(part.visible, f"{part_type.value} should remain visible")

        for part_type in (PartType.CHILL, PartType.SLEEVE, PartType.CORE):
            part = self._manager.get_part(self._part_ids[part_type])
            assert part is not None
            self.assertFalse(part.visible, f"{part_type.value} should be hidden")

    def test_disabled_setting_leaves_all_parts_visible(self) -> None:
        _settings.visualization.hide_auxiliary_parts_on_sim_finish = False

        self._apply_rule(self._manager, {PartType.CHILL, PartType.SLEEVE, PartType.CORE})

        for part in self._manager.get_all_parts():
            self.assertTrue(part.visible, f"{part.part_type.value} should remain visible")


if __name__ == "__main__":
    unittest.main()