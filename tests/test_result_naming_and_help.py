"""Shrink / Microporosity result naming, back-compat, and result help coverage.

"Shrink" (mode ``shrinkage``) is shrinkage_probability with chills at 0, and
"Microporosity" (mode ``porosity``) is porosity_probability with chills at 0,
on both apps. Projects saved with the old "Shrinkage/Porosity" result type
open as Shrink. ``core/result_help.json`` holds the help text both apps show.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from core.result_help import RESULT_HELP_PATH, load_result_help, result_help_text
from core.solidification import SolidificationEngine
from ui.project_state import load_project_state
from ui.result_types import (
    DEFAULT_RESULT_TYPE,
    LEGACY_RESULT_TYPE_LABELS,
    RESULT_TYPE_MODES,
    canonical_result_type,
    result_type_help,
    result_type_help_html,
    result_type_mode,
)
from web.payloads import _full_grid_source

ROOT = Path(__file__).resolve().parent.parent
VIEW_STATE_TS = ROOT / "web_frontend" / "src" / "lib" / "viewState.ts"
MAIN_WINDOW_PY = ROOT / "ui" / "main_window.py"

PART_TYPES = np.array([0, 1, 3, 2], dtype=np.int32)  # cast, riser, riser post, chill
SHRINK = np.array([0.9, 0.2, 0.3, 0.7], dtype=np.float32)
POROSITY = np.array([0.4, 0.8, 0.1, 0.6], dtype=np.float32)


def _engine() -> SolidificationEngine:
    engine = SolidificationEngine.__new__(SolidificationEngine)
    engine.mesh = SimpleNamespace(voxel_part_types=PART_TYPES.copy())
    engine.shrinkage_probability = SHRINK.copy()
    engine.porosity_probability = POROSITY.copy()
    return engine


def _web_mode_ids() -> list[str]:
    source = VIEW_STATE_TS.read_text(encoding="utf-8")
    block = re.search(r"RESULT_MODE_IDS = \[(.*?)\] as const", source, re.S)
    assert block is not None
    return re.findall(r"'([a-z_]+)'", block.group(1))


def _web_mode_labels() -> dict[str, str]:
    source = VIEW_STATE_TS.read_text(encoding="utf-8")
    block = re.search(r"RESULT_MODE_LABELS: Record<ResultModeId, string> = \{(.*?)\};", source, re.S)
    assert block is not None
    return dict(re.findall(r"^\s*([a-z_]+): '([^']*)'", block.group(1), re.M))


class ShrinkMicroporosityArrayTests(unittest.TestCase):
    def test_shrink_is_shrinkage_only_with_chills_zeroed(self) -> None:
        display = _engine().get_shrink_display_intensity()
        np.testing.assert_allclose(display, [0.9, 0.2, 0.3, 0.0], atol=1e-6)

    def test_shrink_no_longer_includes_porosity(self) -> None:
        display = _engine().get_shrink_display_intensity()
        # The riser point has porosity 0.8 > shrink 0.2; Shrink keeps 0.2.
        self.assertAlmostEqual(float(display[1]), 0.2, places=6)

    def test_microporosity_is_porosity_with_chills_zeroed(self) -> None:
        display = _engine().get_microporosity_display_intensity()
        np.testing.assert_allclose(display, [0.4, 0.8, 0.1, 0.0], atol=1e-6)

    def test_combined_field_still_available(self) -> None:
        display = _engine().get_shrinkage_display_intensity()
        np.testing.assert_allclose(display, [0.9, 0.8, 0.3, 0.0], atol=1e-6)

    def test_missing_porosity_gives_empty_microporosity(self) -> None:
        engine = _engine()
        engine.porosity_probability = None
        self.assertEqual(engine.get_microporosity_display_intensity().size, 0)


class WebFieldArrayTests(unittest.TestCase):
    def test_grid_source_uses_desktop_display_arrays(self) -> None:
        engine = _engine()
        mesh = SimpleNamespace(
            voxel_centers=np.zeros((4, 3), dtype=np.float32),
            voxel_sizes=np.ones(4, dtype=np.float32),
            voxel_part_types=PART_TYPES.copy(),
        )
        source = _full_grid_source(engine, mesh, np.zeros((4, 3), dtype=np.float32))
        np.testing.assert_allclose(source["shrinkage"], engine.get_shrink_display_intensity())
        np.testing.assert_allclose(source["porosity"], engine.get_microporosity_display_intensity())

    def test_web_labels(self) -> None:
        labels = _web_mode_labels()
        self.assertEqual(labels["shrinkage"], "Shrink")
        self.assertEqual(labels["porosity"], "Microporosity")

    def test_no_shrinkage_porosity_label_left_in_frontend(self) -> None:
        for path in (ROOT / "web_frontend" / "src").rglob("*.ts*"):
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(
                re.search(r"Shrinkage\s*/\s*Porosity", text, re.I), f"old label in {path}"
            )


class DesktopNamingTests(unittest.TestCase):
    def test_desktop_labels_map_to_web_mode_ids(self) -> None:
        self.assertEqual(RESULT_TYPE_MODES["Shrink"], "shrinkage")
        self.assertEqual(RESULT_TYPE_MODES["Microporosity"], "porosity")
        self.assertEqual(DEFAULT_RESULT_TYPE, "Shrink")
        self.assertTrue(set(RESULT_TYPE_MODES.values()) <= set(_web_mode_ids()))

    def test_legacy_label_maps_to_shrink(self) -> None:
        self.assertEqual(canonical_result_type("Shrinkage/Porosity"), "Shrink")
        self.assertEqual(result_type_mode("Shrinkage/Porosity"), "shrinkage")
        self.assertEqual(canonical_result_type("Hot Spots"), "Hot Spots")

    def test_project_with_legacy_result_type_loads_as_shrink(self) -> None:
        loaded = load_project_state(
            {
                "schema_version": 2,
                "simulation": {"current_vis_mode": "shrinkage"},
                "ui_state": {"result_type": "Shrinkage/Porosity", "threshold": 70},
                "parts": [],
            }
        ).to_dict()
        self.assertEqual(loaded["ui_state"]["result_type"], "Shrink")
        self.assertEqual(loaded["simulation"]["current_vis_mode"], "shrinkage")

    def test_result_type_change_accepts_legacy_and_new_labels(self) -> None:
        from ui.main_window import MainWindow

        for label, mode in (("Shrinkage/Porosity", "shrinkage"), ("Shrink", "shrinkage"), ("Microporosity", "porosity")):
            host = mock.MagicMock()
            host.simulation_worker = None
            host.current_results = None
            MainWindow._on_result_type_changed(host, label)
            self.assertEqual(host._current_vis_mode, mode)

    def test_every_offered_label_has_a_mode(self) -> None:
        source = MAIN_WINDOW_PY.read_text(encoding="utf-8")
        offered = set(re.findall(r'available_types\.append\("([^"]+)"\)', source))
        self.assertIn("Shrink", offered)
        self.assertIn("Microporosity", offered)
        self.assertEqual(offered - set(RESULT_TYPE_MODES), set())
        self.assertNotIn("Shrinkage/Porosity Risk", source)
        self.assertNotIn('"Shrinkage/Porosity":', source)


class DesktopViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _setup_host(self, porosity):
        from PySide6.QtWidgets import QComboBox

        from ui.main_window import MainWindow

        host = mock.MagicMock()
        host._analysis_options = {"shrinkage": True}
        host.simulation_worker.simulation = SimpleNamespace(
            local_modulus=None,
            porosity_probability=porosity,
            get_fill_animation_frames=lambda: (),
        )
        host.result_type_combo = QComboBox()
        host._apply_result_type_item_tooltips = lambda: MainWindow._apply_result_type_item_tooltips(host)
        MainWindow._setup_result_type_options(host)
        return host

    def test_microporosity_offered_when_porosity_data_exists(self) -> None:
        from PySide6.QtCore import Qt

        host = self._setup_host(POROSITY.copy())
        combo = host.result_type_combo
        labels = [combo.itemText(i) for i in range(combo.count())]
        self.assertEqual(labels, ["Shrink", "Microporosity"])
        for index, label in enumerate(labels):
            tooltip = combo.itemData(index, Qt.ToolTipRole)
            self.assertEqual(tooltip, result_type_help_html(label))
            self.assertIn(result_help_text(result_type_mode(label))[:40], tooltip)

    def test_microporosity_hidden_without_porosity_data(self) -> None:
        host = self._setup_host(None)
        combo = host.result_type_combo
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["Shrink"])

    def _visualize(self, microporosity: bool):
        from ui.main_window import MainWindow

        host = mock.MagicMock()
        host._get_effective_shrinkage_threshold = MainWindow._get_effective_shrinkage_threshold
        engine = _engine()
        mesh = SimpleNamespace(
            voxel_centers=np.zeros((4, 3), dtype=np.float32),
            voxel_sizes=np.ones(4, dtype=np.float32),
        )
        MainWindow._visualize_shrinkage(host, engine, mesh, 0.05, microporosity=microporosity)
        return host, engine

    def test_shrink_view_paints_shrink_array(self) -> None:
        host, engine = self._visualize(False)
        painted = host.visualizer.load_shrinkage.call_args.args[2]
        np.testing.assert_allclose(painted, engine.get_shrink_display_intensity())
        host.visualizer.update_legend.assert_called_with("Shrink Risk", 0.0, 1.0)

    def test_microporosity_view_paints_porosity_array_and_keeps_inspector(self) -> None:
        host, engine = self._visualize(True)
        painted = host.visualizer.load_shrinkage.call_args.args[2]
        np.testing.assert_allclose(painted, engine.get_microporosity_display_intensity())
        host.visualizer.update_legend.assert_called_with("Microporosity Risk", 0.0, 1.0)
        host.visualizer.set_shrink_inspector_data.assert_called_once()

    def test_info_button_tooltip_follows_selection(self) -> None:
        from PySide6.QtWidgets import QComboBox, QToolButton

        from ui.main_window import MainWindow

        host = mock.MagicMock()
        host.result_type_combo = QComboBox()
        host.result_type_combo.addItems(["Shrink", "Microporosity"])
        host.result_type_help_btn = QToolButton()
        host._selected_result_type_help_html = lambda: MainWindow._selected_result_type_help_html(host)
        host.result_type_combo.setCurrentIndex(1)
        MainWindow._sync_result_type_help_button(host)
        self.assertEqual(
            host.result_type_help_btn.toolTip(),
            result_type_help_html("Microporosity", with_title=True),
        )


class ResultHelpCoverageTests(unittest.TestCase):
    def test_help_covers_every_web_mode_exactly(self) -> None:
        self.assertEqual(set(load_result_help()), set(_web_mode_ids()))

    def test_help_covers_every_desktop_result_type(self) -> None:
        for label, mode in RESULT_TYPE_MODES.items():
            self.assertTrue(result_type_help(label), f"no help for desktop result {label!r}")
        for legacy in LEGACY_RESULT_TYPE_LABELS:
            self.assertTrue(result_type_help(legacy))

    def test_web_reads_the_shared_json(self) -> None:
        source = VIEW_STATE_TS.read_text(encoding="utf-8")
        self.assertIn("from '../../../core/result_help.json'", source)
        self.assertIn("RESULT_MODE_HELP: Record<ResultModeId, string> = resultHelp", source)
        panel = (ROOT / "web_frontend" / "src" / "components" / "ResultsPanel.tsx").read_text(encoding="utf-8")
        self.assertIn("title={RESULT_MODE_HELP[mode]}", panel)
        self.assertIn("aria-describedby", panel)

    def test_help_texts_are_short_paragraphs(self) -> None:
        data = json.loads(RESULT_HELP_PATH.read_text(encoding="utf-8"))
        for mode, text in data.items():
            sentences = [s for s in re.split(r"(?<=[.])\s+(?=[A-Z])", text.strip()) if s]
            self.assertTrue(2 <= len(sentences) <= 4, f"{mode}: {len(sentences)} sentences")
            self.assertNotIn("Shrinkage/Porosity", text)

    def test_names_used_in_help(self) -> None:
        self.assertIn("Macro-shrinkage", result_help_text("shrinkage"))
        self.assertIn("Microporosity", result_help_text("porosity"))
        self.assertIn("70 %", result_help_text("shrinkage"))
        self.assertIn("70 %", result_help_text("porosity"))


if __name__ == "__main__":
    unittest.main()
