"""Tests for the MAGMASOFT-style pour (metal-entry) location picker.

Two testing strategies are combined here, matching what each layer needs:

* The real-Qt tests (``AnalysisOptionsDialog`` gating, ``MainWindow`` gate
  wiring) instantiate genuine PyQt5 widgets under ``QT_QPA_PLATFORM=offscreen``
  -- the same approach ``tests/test_performance_summary_label.py`` uses for
  ``PerformanceDialog``. These imports happen at module scope, before any
  vtk/PyQt5 stubbing below, so they keep using the real Qt classes even after
  the stub swap happens.
* The ``CastingVisualizer`` pick-mode/marker tests stub ``vtk`` and ``PyQt5``
  exactly the way ``tests/test_visualizer_inspector.py`` does, so they run
  without a display or real VTK render window.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Real PyQt5 + the modules under test, imported now so they are fully
# resolved (and cached in sys.modules) before the vtk/PyQt5 stub swap below.
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

from core.pour_location import PourLocation  # noqa: E402
from ui.dialogs.analysis_options_dialog import AnalysisOptionsDialog  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402

from tests.helpers.module_stubs import (  # noqa: E402
    install_stub_modules,
    restore_modules,
    snapshot_modules,
)

# Captured before any stub is installed, so tearDownModule can hand the real
# vtk/PyQt5 modules back to whatever runs next in this process.
_MODULE_SNAPSHOT = snapshot_modules()


def tearDownModule() -> None:
    restore_modules(_MODULE_SNAPSHOT)


def _qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ==========================================================================
# AnalysisOptionsDialog: the "Mold Filling" toggle gate
# ==========================================================================


class FillingToggleGateTests(unittest.TestCase):
    """The Mold Filling checkbox must stay gated behind set_filling_available."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _qapp()

    def setUp(self) -> None:
        self.dialog = AnalysisOptionsDialog(parent=None)

    def tearDown(self) -> None:
        self.dialog.close()
        self.dialog.deleteLater()

    def test_filling_check_starts_disabled_and_unchecked(self) -> None:
        self.assertFalse(self.dialog.filling_check.isEnabled())
        self.assertFalse(self.dialog.filling_check.isChecked())
        self.assertIn("pour location", self.dialog.filling_check.toolTip().lower())

    def test_set_filling_available_true_enables_with_explanatory_tooltip(self) -> None:
        self.dialog.set_filling_available(True)
        self.assertTrue(self.dialog.filling_check.isEnabled())
        self.assertIn("CFD", self.dialog.filling_check.toolTip())

    def test_set_filling_available_false_disables_and_forces_unchecked(self) -> None:
        self.dialog.set_filling_available(True)
        self.dialog.filling_check.setChecked(True)

        self.dialog.set_filling_available(False)

        self.assertFalse(self.dialog.filling_check.isEnabled())
        self.assertFalse(self.dialog.filling_check.isChecked())

    def test_get_options_reports_filling_state(self) -> None:
        self.dialog.set_filling_available(True)
        self.dialog.filling_check.setChecked(True)
        self.assertTrue(self.dialog.get_options()["filling"])

    def test_restore_original_values_does_not_reenable_filling(self) -> None:
        self.dialog.set_filling_available(True)
        self.dialog.filling_check.setChecked(True)
        self.dialog.store_original_values()

        # Pour location was cleared while the dialog held this state.
        self.dialog.set_filling_available(False)
        self.dialog.restore_original_values()

        self.assertFalse(self.dialog.filling_check.isChecked())
        self.assertFalse(self.dialog.filling_check.isEnabled())


# ==========================================================================
# MainWindow: the actual gate wiring (visualizer signal -> dialog + state)
# ==========================================================================


class MainWindowPourLocationGateTests(unittest.TestCase):
    """MainWindow._on_pour_location_changed must drive the gate end to end."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _qapp()

    def _make_host(self):
        host = mock.MagicMock()
        host.set_pour_location_btn = QPushButton()
        host.set_pour_location_btn.setCheckable(True)
        host.set_pour_location_btn.setChecked(True)
        host.pour_location_label = QLabel()
        host.clear_pour_location_btn = QPushButton()
        host.clear_pour_location_btn.setEnabled(False)
        host.analysis_dialog = AnalysisOptionsDialog(parent=None)
        host._analysis_options = {"filling": False}
        host._pour_location = None
        host.status_label = QLabel()
        return host

    def _close(self, host) -> None:
        host.analysis_dialog.close()
        host.analysis_dialog.deleteLater()

    def test_picked_location_enables_filling_and_updates_label(self) -> None:
        host = self._make_host()
        location = PourLocation(point_in=(1.0, 2.0, 3.0))

        MainWindow._on_pour_location_changed(host, location)

        try:
            self.assertFalse(host.set_pour_location_btn.isChecked())
            self.assertIn("1.00", host.pour_location_label.text())
            self.assertTrue(host.clear_pour_location_btn.isEnabled())
            self.assertTrue(host.analysis_dialog.filling_check.isEnabled())
            self.assertEqual(host._pour_location, location.to_dict())
        finally:
            self._close(host)

    def test_cleared_location_disables_and_forces_filling_off(self) -> None:
        host = self._make_host()
        host._analysis_options["filling"] = True
        host.analysis_dialog.set_filling_available(True)
        host.analysis_dialog.filling_check.setChecked(True)

        MainWindow._on_pour_location_changed(host, None)

        try:
            self.assertFalse(host.analysis_dialog.filling_check.isEnabled())
            self.assertFalse(host.analysis_dialog.filling_check.isChecked())
            self.assertFalse(host._analysis_options["filling"])
            self.assertIsNone(host._pour_location)
            self.assertEqual(host.pour_location_label.text(), "Pour Location: not set")
            self.assertFalse(host.clear_pour_location_btn.isEnabled())
        finally:
            self._close(host)


# ==========================================================================
# CastingVisualizer: click -> PourLocation, marker lifecycle (vtk-stubbed)
# ==========================================================================


def _install_vtk_stubs() -> None:
    vtk_module = types.ModuleType("vtk")
    vtk_module.VTK_FLOAT = 1
    vtk_module.VTK_UNSIGNED_CHAR = 2
    vtk_module.VTK_FOLLOWER = 3

    def _vtk_getattr(name):
        return _DummyVTKClass

    vtk_module.__getattr__ = _vtk_getattr

    vtk_util_module = types.ModuleType("vtk.util")
    numpy_support_module = types.ModuleType("vtk.util.numpy_support")
    numpy_support_module.numpy_to_vtk = lambda *args, **kwargs: _DummyVTKClass()
    vtk_util_module.numpy_support = numpy_support_module

    vtk_qt_module = types.ModuleType("vtk.qt")
    qvtk_module = types.ModuleType("vtk.qt.QVTKRenderWindowInteractor")
    qvtk_module.QVTKRenderWindowInteractor = _DummyVTKClass
    vtk_qt_module.QVTKRenderWindowInteractor = qvtk_module

    pyqt5_module = types.ModuleType("PyQt5")
    qtwidgets_module = types.ModuleType("PyQt5.QtWidgets")
    qtwidgets_module.QWidget = _DummyVTKClass
    qtwidgets_module.QVBoxLayout = _DummyVTKClass
    qtcore_module = types.ModuleType("PyQt5.QtCore")
    qtcore_module.QTimer = _DummyVTKClass
    qtcore_module.pyqtSignal = lambda *args, **kwargs: _DummySignal()

    install_stub_modules({
        "vtk": vtk_module,
        "vtk.util": vtk_util_module,
        "vtk.util.numpy_support": numpy_support_module,
        "vtk.qt": vtk_qt_module,
        "vtk.qt.QVTKRenderWindowInteractor": qvtk_module,
        "PyQt5": pyqt5_module,
        "PyQt5.QtWidgets": qtwidgets_module,
        "PyQt5.QtCore": qtcore_module,
    })


class _DummyVTKClass:
    """Generic VTK object stand-in that tolerates chained calls.

    Unlike a stub that returns ``None`` from every method (fine for code
    that never chains), ``_add_pour_location_marker`` does
    ``actor.GetProperty().SetColor(...)`` -- so accessor methods here return
    ``self`` rather than ``None``, letting arbitrary chains resolve to a
    harmless no-op instead of an ``AttributeError`` on ``None``.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return self
        return _method


class _DummySignal:
    """Stand-in for a bound pyqtSignal; records nothing unless replaced."""

    def emit(self, *args, **kwargs) -> None:
        return None


class _RecordingRenderer:
    def __init__(self) -> None:
        self.actors: list = []

    def AddActor(self, actor) -> None:
        self.actors.append(actor)

    def RemoveActor(self, actor) -> None:
        if actor in self.actors:
            self.actors.remove(actor)


class _RecordingInteractor:
    def __init__(self) -> None:
        self.event_pos = (0, 0)

    def GetEventPosition(self):
        return self.event_pos


class _RecordingCellPicker:
    """Stand-in for vtkCellPicker with a programmed hit (or miss)."""

    def __init__(self, cell_id: int, position=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 0.0), actor=None) -> None:
        self._cell_id = cell_id
        self._position = position
        self._normal = normal
        self._actor = actor
        self.tolerance = 0.0
        self.pick_calls = 0

    def SetTolerance(self, value) -> None:
        self.tolerance = value

    def Pick(self, x, y, z, renderer) -> None:
        self.pick_calls += 1

    def GetCellId(self) -> int:
        return self._cell_id

    def GetPickPosition(self):
        return self._position

    def GetPickNormal(self):
        return self._normal

    def GetActor(self):
        return self._actor


class _RecordingWorldPicker:
    """Stand-in for vtkWorldPointPicker."""

    def __init__(self, position=(0.0, 0.0, 0.0)) -> None:
        self._position = position
        self.pick_calls = 0

    def Pick(self, x, y, z, renderer) -> None:
        self.pick_calls += 1

    def GetPickPosition(self):
        return self._position


class PourPickerVisualizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_vtk_stubs()
        ui_module = sys.modules.get("ui")
        if ui_module is None:
            import ui  # noqa: F401
        viz_module = importlib.import_module("ui.visualizer")
        cls.viz = importlib.reload(viz_module)

    def _bare_visualizer(self):
        viz = self.viz.CastingVisualizer.__new__(self.viz.CastingVisualizer)
        viz._pour_pick_enabled = False
        viz._pour_location = None
        viz._pour_location_actors = []
        viz.part_actors = {}
        viz.model_bounds = (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)
        viz.measurement_tool = None
        viz._inspector_layer = None
        viz.renderer = _RecordingRenderer()
        viz.render_window = _DummyVTKClass()
        viz.render_window.Render = lambda: None
        viz.interactor = _RecordingInteractor()
        viz.pour_location_changed = mock.Mock()
        return viz

    # ---- entering / exiting pick mode ----

    def test_enable_and_disable_pick_mode(self) -> None:
        viz = self._bare_visualizer()
        self.assertFalse(viz.is_pour_pick_mode())

        viz.enable_pour_pick_mode()
        self.assertTrue(viz.is_pour_pick_mode())

        viz.disable_pour_pick_mode()
        self.assertFalse(viz.is_pour_pick_mode())

    # ---- a click producing a PourLocation ----

    def test_click_with_cell_hit_sets_location_with_normal_and_part(self) -> None:
        viz = self._bare_visualizer()
        part_actor = _DummyVTKClass()
        viz.part_actors = {"part-1": part_actor}
        viz.enable_pour_pick_mode()

        picker = _RecordingCellPicker(
            cell_id=3, position=(4.0, -2.0, 7.5), normal=(0.0, 1.0, 0.0), actor=part_actor
        )
        viz.interactor.event_pos = (50, 60)
        with mock.patch.object(self.viz.vtk, "vtkCellPicker", lambda: picker):
            viz._on_left_button_press(None, None)

        self.assertEqual(picker.pick_calls, 1)
        location = viz.pour_location
        self.assertIsInstance(location, self.viz.PourLocation)
        self.assertEqual(location.point_in, (4.0, -2.0, 7.5))
        self.assertEqual(location.normal, (0.0, 1.0, 0.0))
        self.assertEqual(location.part_id, "part-1")
        # A successful pick exits pick mode and notifies listeners once.
        self.assertFalse(viz.is_pour_pick_mode())
        viz.pour_location_changed.emit.assert_called_once_with(location)

    def test_click_with_cell_miss_falls_back_to_world_picker(self) -> None:
        viz = self._bare_visualizer()
        viz.enable_pour_pick_mode()

        cell_picker = _RecordingCellPicker(cell_id=-1)
        world_picker = _RecordingWorldPicker(position=(9.0, 9.0, 9.0))
        with mock.patch.object(self.viz.vtk, "vtkCellPicker", lambda: cell_picker), \
             mock.patch.object(self.viz.vtk, "vtkWorldPointPicker", lambda: world_picker):
            viz._on_left_button_press(None, None)

        location = viz.pour_location
        self.assertIsNotNone(location)
        self.assertEqual(location.point_in, (9.0, 9.0, 9.0))
        self.assertIsNone(location.normal)
        self.assertIsNone(location.part_id)

    def test_click_missing_everything_stays_in_pick_mode(self) -> None:
        viz = self._bare_visualizer()
        viz.enable_pour_pick_mode()

        cell_picker = _RecordingCellPicker(cell_id=-1)
        world_picker = _RecordingWorldPicker(position=(0.0, 0.0, 0.0))
        with mock.patch.object(self.viz.vtk, "vtkCellPicker", lambda: cell_picker), \
             mock.patch.object(self.viz.vtk, "vtkWorldPointPicker", lambda: world_picker):
            viz._on_left_button_press(None, None)

        self.assertIsNone(viz.pour_location)
        self.assertTrue(viz.is_pour_pick_mode())
        viz.pour_location_changed.emit.assert_not_called()

    # ---- marker actor lifecycle ----

    def test_marker_actors_created_on_pick_and_removed_on_clear(self) -> None:
        viz = self._bare_visualizer()
        viz.enable_pour_pick_mode()
        baseline = len(viz.renderer.actors)

        picker = _RecordingCellPicker(
            cell_id=1, position=(1.0, 1.0, 1.0), normal=(0.0, 0.0, 1.0)
        )
        with mock.patch.object(self.viz.vtk, "vtkCellPicker", lambda: picker):
            viz._on_left_button_press(None, None)

        # Sphere + normal cone.
        self.assertEqual(len(viz.renderer.actors), baseline + 2)
        self.assertEqual(len(viz._pour_location_actors), 2)

        viz.clear_pour_location()

        self.assertEqual(len(viz.renderer.actors), baseline)
        self.assertEqual(viz._pour_location_actors, [])
        self.assertIsNone(viz.pour_location)
        viz.pour_location_changed.emit.assert_called_with(None)

    def test_marker_without_normal_is_sphere_only(self) -> None:
        viz = self._bare_visualizer()
        viz.enable_pour_pick_mode()
        baseline = len(viz.renderer.actors)

        # GetPickNormal returns all-zero -> treated as "no normal".
        picker = _RecordingCellPicker(cell_id=0, position=(2.0, 2.0, 2.0), normal=(0.0, 0.0, 0.0))
        with mock.patch.object(self.viz.vtk, "vtkCellPicker", lambda: picker):
            viz._on_left_button_press(None, None)

        self.assertIsNone(viz.pour_location.normal)
        self.assertEqual(len(viz.renderer.actors), baseline + 1)

    # ---- set_pour_location() direct path (project restore) ----

    def test_set_pour_location_directly_draws_marker_and_emits(self) -> None:
        viz = self._bare_visualizer()
        location = self.viz.PourLocation(point_in=(5.0, 6.0, 7.0))

        viz.set_pour_location(location)

        self.assertIs(viz.pour_location, location)
        self.assertEqual(len(viz._pour_location_actors), 1)  # sphere only, no normal
        viz.pour_location_changed.emit.assert_called_once_with(location)

    def test_clear_pour_location_without_prior_pick_is_a_noop(self) -> None:
        viz = self._bare_visualizer()
        viz.clear_pour_location()
        self.assertIsNone(viz.pour_location)
        viz.pour_location_changed.emit.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
