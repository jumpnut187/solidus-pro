"""VTK-stubbed tests for the result-mode click inspector.

Covers the click routing, cell-id -> voxel-index mapping, leader
text formatting, and threshold-change clearing without requiring a
real VTK render window.
"""

from __future__ import annotations

import importlib
import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np


class _RecordingActor:
    def __init__(self) -> None:
        self.removed = False
        self.visibility = True

    def RemoveActor(self, actor) -> None:
        actor.removed = True

    def SetVisibility(self, value: bool) -> None:
        self.visibility = bool(value)


class _RecordingRenderer:
    def __init__(self) -> None:
        self.actors: list[_RecordingActor] = []
        self.camera = _TrackingCamera()

    def AddActor(self, actor) -> None:
        self.actors.append(actor)

    def RemoveActor(self, actor) -> None:
        if actor in self.actors:
            self.actors.remove(actor)

    def GetActiveCamera(self):
        return self.camera

    def GetActors(self):
        return self


class _TrackingCamera:
    def __init__(self) -> None:
        self.position = (0.0, 0.0, 0.0)
        self.view_up = (0.0, 0.0, 1.0)

    def SetPosition(self, *args) -> None:
        self.position = tuple(args)

    def GetViewUp(self):
        return self.view_up

    def SetViewUp(self, *args) -> None:
        self.view_up = tuple(args)

    def __getattr__(self, name):
        return lambda *a, **k: None


class _RecordingPicker:
    """Stand-in for vtkCellPicker that returns a programmed cell id."""

    def __init__(self, cell_id: int) -> None:
        self._cell_id = cell_id
        self.picked_pos = (0.0, 0.0, 0.0)
        self.pick_calls = 0
        self.pick_list: list = []
        self.from_list = False
        self.tolerance = 0.0

    def AddPickList(self, actor) -> None:
        self.pick_list.append(actor)

    def SetPickFromList(self, value: bool) -> None:
        self.from_list = bool(value)

    def SetTolerance(self, value: float) -> None:
        self.tolerance = float(value)

    def Pick(self, x, y, z, renderer) -> None:
        self.pick_calls += 1

    def GetCellId(self) -> int:
        return self._cell_id

    def GetPickPosition(self):
        return self.picked_pos


class _RecordingInteractor:
    def __init__(self) -> None:
        self.event_pos = (0, 0)
        self.abort_flag = 0
        self.style = _RecordingStyle()

    def GetEventPosition(self):
        return self.event_pos

    def GetInteractorStyle(self):
        return self.style


class _RecordingStyle:
    def __init__(self) -> None:
        self.abort = 0

    def SetAbortFlag(self, value: int) -> None:
        self.abort = int(value)


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        import threading as _t
        f = _t.Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except BaseException as e:
            f.set_exception(e)
        return f


def _install_stubs() -> None:
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

    sys.modules.setdefault("vtk", vtk_module)
    sys.modules.setdefault("vtk.util", vtk_util_module)
    sys.modules.setdefault("vtk.util.numpy_support", numpy_support_module)
    sys.modules.setdefault("vtk.qt", vtk_qt_module)
    sys.modules.setdefault("vtk.qt.QVTKRenderWindowInteractor", qvtk_module)
    sys.modules.setdefault("PyQt5", pyqt5_module)
    sys.modules.setdefault("PyQt5.QtWidgets", qtwidgets_module)
    sys.modules.setdefault("PyQt5.QtCore", qtcore_module)


class _DummyVTKClass:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return None
        return _method


class _DummySignal:
    def emit(self, *args, **kwargs):
        return None


class _StubFeedModulusLeader:
    """Records the last callout text + anchor for assertions."""

    def __init__(self, renderer) -> None:
        self.renderer = renderer
        self.last_callout = None
        self.cleared = 0
        self.is_empty_flag = True

    def clear(self) -> None:
        self.cleared += 1
        self.last_callout = None
        self.is_empty_flag = True

    def is_empty(self) -> bool:
        return self.is_empty_flag

    def show_callout(self, callout, model_bounds=None) -> None:
        self.last_callout = callout
        self.is_empty_flag = False

    def show_leader(self, world_pos, value, model_bounds=None) -> None:
        self.last_callout = types.SimpleNamespace(
            text=f"FM: {value:.2f} in",
            anchor_position=tuple(world_pos),
        )
        self.is_empty_flag = False


class _StubVisualizerModule:
    """Install a stub ``FeedModulusLeaderTool`` before the visualizer is imported."""

    pass


class InspectorRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_stubs()
        # Replace the FeedModulusLeaderTool with a stub so we can capture callouts.
        ui_module = sys.modules.get("ui")
        if ui_module is None:
            import ui  # noqa: F401
        viz_module = importlib.import_module("ui.visualizer")
        viz_module.FeedModulusLeaderTool = _StubFeedModulusLeader  # type: ignore[misc]
        cls.viz = importlib.reload(viz_module)

    def _bare_visualizer(self):
        viz = self.viz.CastingVisualizer.__new__(self.viz.CastingVisualizer)
        viz._cache_threshold_levels = [i / 100.0 for i in range(1, 101)]
        viz._shrinkage_cache = {}
        viz._shrinkage_cache_built = False
        viz._shrinkage_cache_lock = threading.Lock()
        viz._shrinkage_cache_generation = 0
        viz._pending_shrinkage_thresholds = set()
        viz._cache_executor = _ImmediateExecutor()
        viz._hotspot_cache = {}
        viz._niyama_cache = {}
        viz._hot_tear_cache = {}
        viz._sand_burn_in_cache = {}
        viz._gas_porosity_cache = {}
        viz.shrinkage_actor = _RecordingActor()
        viz.hotspot_actor = None
        viz.niyama_actor = None
        viz.hot_tear_actor = None
        viz.sand_burn_in_actor = None
        viz.gas_porosity_actor = None
        viz.fraction_liquid_actor = None
        viz.model_bounds = (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)
        viz._cross_section_enabled = False
        viz.cross_section_planes = {}
        viz.measurement_tool = None
        viz.feed_modulus_leader = _StubFeedModulusLeader(_RecordingRenderer())
        viz._feed_modulus_leader_enabled = False
        viz._inspector_layer = None
        viz._shrink_inspector_data = None
        viz._shrinkage_displayed_voxel_indices = None
        viz._shrinkage_cells_per_glyph = 6
        viz._animation_frames = []
        viz._is_recording = False
        viz._interactor_initialized = True
        viz._is_orthographic = True
        viz.renderer = _RecordingRenderer()
        viz.render_window = _DummyVTKClass()
        viz.render_window.Render = lambda: None
        viz.interactor = _RecordingInteractor()
        return viz

    def _install_layer(self, viz, *, voxel_index: int, label: str, anchor=(1.0, 2.0, 3.0)):
        """Install an InspectorLayer whose resolver returns a fixed callout."""
        from ui.visualizer import InspectorLayer, LeaderCallout

        seen_voxel_indices: list[int] = []

        def resolver(picked_voxel_index: int):
            seen_voxel_indices.append(picked_voxel_index)
            return LeaderCallout(text=label, anchor_position=anchor)

        viz._shrinkage_displayed_voxel_indices = np.array(
            [voxel_index, voxel_index + 1, voxel_index + 2], dtype=np.int32
        )
        viz._inspector_layer = InspectorLayer(
            mode="shrinkage",
            pick_actor=viz.shrinkage_actor,
            displayed_voxel_indices=viz._shrinkage_displayed_voxel_indices,
            cells_per_glyph=6,
            resolver=resolver,
        )
        viz._seen_voxel_indices = seen_voxel_indices

    def test_click_routes_to_resolver_and_renders_callout(self) -> None:
        viz = self._bare_visualizer()
        self._install_layer(
            viz,
            voxel_index=42,
            label="Shrink pocket: 0.125 in^3",
            anchor=(2.0, 0.0, 0.0),
        )
        # Picker returns cell_id = 12 -> glyph_index = 12 // 6 = 2 -> voxel_index = 44
        picker = _RecordingPicker(cell_id=12)
        with mock.patch.object(self.viz.vtk, "vtkCellPicker", lambda: picker):
            viz.interactor.event_pos = (123, 456)
            viz._on_left_button_press(None, None)
        self.assertEqual(picker.pick_calls, 1)
        self.assertEqual(picker.pick_list, [viz.shrinkage_actor])
        self.assertTrue(picker.from_list)
        # The third glyph in displayed_voxel_indices -> index 2 -> voxel_index 44
        self.assertEqual(viz._seen_voxel_indices, [44])
        self.assertEqual(viz.feed_modulus_leader.last_callout.text,
                         "Shrink pocket: 0.125 in^3")

    def test_click_with_no_layer_is_a_noop(self) -> None:
        viz = self._bare_visualizer()
        picker = _RecordingPicker(cell_id=0)
        with mock.patch.object(self.viz.vtk, "vtkCellPicker", lambda: picker):
            viz._on_left_button_press(None, None)
        # No layer, no leader update, but the handler should not crash.
        self.assertIsNone(viz.feed_modulus_leader.last_callout)

    def test_click_outside_glyphs_clears_leader(self) -> None:
        viz = self._bare_visualizer()
        self._install_layer(
            viz,
            voxel_index=10,
            label="Shrink pocket: 0.5 in^3",
        )
        # Pre-render a leader so we can see it gets cleared
        viz.feed_modulus_leader.last_callout = types.SimpleNamespace(
            text="previous", anchor_position=(0, 0, 0),
        )
        picker = _RecordingPicker(cell_id=-1)  # no hit
        with mock.patch.object(self.viz.vtk, "vtkCellPicker", lambda: picker):
            viz._on_left_button_press(None, None)
        self.assertEqual(viz.feed_modulus_leader.cleared, 1)
        self.assertIsNone(viz.feed_modulus_leader.last_callout)

    def test_resolver_returning_none_clears_leader(self) -> None:
        viz = self._bare_visualizer()

        def resolver(_voxel_index: int):
            return None

        viz._shrinkage_displayed_voxel_indices = np.array([7], dtype=np.int32)
        viz._inspector_layer = self.viz.InspectorLayer(
            mode="shrinkage",
            pick_actor=viz.shrinkage_actor,
            displayed_voxel_indices=viz._shrinkage_displayed_voxel_indices,
            cells_per_glyph=6,
            resolver=resolver,
        )
        viz.feed_modulus_leader.last_callout = types.SimpleNamespace(
            text="stale", anchor_position=(0, 0, 0),
        )
        picker = _RecordingPicker(cell_id=0)  # glyph 0 -> voxel 7
        with mock.patch.object(self.viz.vtk, "vtkCellPicker", lambda: picker):
            viz._on_left_button_press(None, None)
        self.assertIsNone(viz.feed_modulus_leader.last_callout)
        self.assertEqual(viz.feed_modulus_leader.cleared, 1)

    def test_clear_inspector_removes_layer_and_clears_leader(self) -> None:
        viz = self._bare_visualizer()
        self._install_layer(
            viz,
            voxel_index=1,
            label="Shrink pocket: 0.001 in^3",
        )
        viz.feed_modulus_leader.last_callout = types.SimpleNamespace(
            text="active", anchor_position=(0, 0, 0),
        )
        viz.clear_inspector()
        self.assertIsNone(viz._inspector_layer)
        self.assertIsNone(viz.feed_modulus_leader.last_callout)


class ShrinkInspectorLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_stubs()
        ui_module = sys.modules.get("ui")
        if ui_module is None:
            import ui  # noqa: F401
        viz_module = importlib.import_module("ui.visualizer")
        viz_module.FeedModulusLeaderTool = _StubFeedModulusLeader  # type: ignore[misc]
        cls.viz = importlib.reload(viz_module)

    def _bare_visualizer(self):
        viz = self.viz.CastingVisualizer.__new__(self.viz.CastingVisualizer)
        viz._cache_threshold_levels = [i / 100.0 for i in range(1, 101)]
        viz._shrinkage_cache = {}
        viz._shrinkage_cache_built = False
        viz._shrinkage_cache_lock = threading.Lock()
        viz._shrinkage_cache_generation = 0
        viz._pending_shrinkage_thresholds = set()
        viz._cache_executor = _ImmediateExecutor()
        viz._hotspot_cache = {}
        viz._niyama_cache = {}
        viz._hot_tear_cache = {}
        viz._sand_burn_in_cache = {}
        viz._gas_porosity_cache = {}
        viz.shrinkage_actor = _RecordingActor()
        viz.hotspot_actor = None
        viz.niyama_actor = None
        viz.hot_tear_actor = None
        viz.sand_burn_in_actor = None
        viz.sand_burn_in_actor = None
        viz.gas_porosity_actor = None
        viz.fraction_liquid_actor = None
        viz.model_bounds = (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)
        viz._cross_section_enabled = False
        viz.cross_section_planes = {}
        viz.measurement_tool = None
        viz.feed_modulus_leader = _StubFeedModulusLeader(_RecordingRenderer())
        viz._feed_modulus_leader_enabled = False
        viz._inspector_layer = None
        viz._shrink_inspector_data = None
        viz._shrinkage_displayed_voxel_indices = None
        viz._shrinkage_cells_per_glyph = 6
        viz._animation_frames = []
        viz._is_recording = False
        viz._interactor_initialized = True
        viz._is_orthographic = True
        viz.renderer = _RecordingRenderer()
        viz.render_window = _DummyVTKClass()
        viz.render_window.Render = lambda: None
        viz.interactor = _RecordingInteractor()
        return viz

    def _stub_mesh(self) -> types.SimpleNamespace:
        centers = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        sizes = np.ones(3, dtype=np.float32)
        volumes = (sizes ** 3).astype(np.float32)
        neighbors = np.array(
            [[1, -1, -1, -1, -1, -1], [0, 2, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1]],
            dtype=np.int32,
        )
        return types.SimpleNamespace(
            voxel_centers=centers,
            voxel_sizes=sizes,
            voxel_volumes=volumes,
            voxel_neighbors=neighbors,
        )

    def test_shrink_resolver_reports_pocket_volume(self) -> None:
        viz = self._bare_visualizer()
        mesh = self._stub_mesh()
        display = np.array([0.0, 0.0, 0.9], dtype=np.float32)
        viz._shrinkage_displayed_voxel_indices = np.array([0, 1, 2], dtype=np.int32)
        viz.set_shrink_inspector_data(
            display_intensity=display,
            mesh=mesh,
            shrinkage_risk=display,
            porosity_risk=None,
            threshold=0.5,
        )
        self.assertIsNotNone(viz._inspector_layer)
        self.assertEqual(viz._inspector_layer.mode, "shrinkage")
        # Resolver invoked on voxel 2 should report the pocket volume.
        callout = viz._inspector_layer.resolver(2)
        self.assertIsNotNone(callout)
        self.assertIn("1.000 in^3", callout.text)
        # Anchor should be the pocket's centroid, not the picked position.
        self.assertAlmostEqual(callout.anchor_position[0], 2.0, places=6)
        # Resolver invoked on a voxel below threshold returns None.
        self.assertIsNone(viz._inspector_layer.resolver(0))

    def test_shrink_resolver_uses_live_threshold(self) -> None:
        viz = self._bare_visualizer()
        mesh = self._stub_mesh()
        display = np.array([0.4, 0.6, 0.9], dtype=np.float32)
        viz._shrinkage_displayed_voxel_indices = np.array([0, 1, 2], dtype=np.int32)
        viz.set_shrink_inspector_data(
            display_intensity=display,
            mesh=mesh,
            shrinkage_risk=display,
            porosity_risk=None,
            threshold=0.5,
        )
        # Picked voxel 1 -> 1 above 0.5; voxel 0 below; one-voxel pocket.
        callout = viz._inspector_layer.resolver(1)
        self.assertIsNotNone(callout)
        # Now raise the threshold to 0.7; voxel 1 is below so resolver returns None.
        viz._shrink_inspector_data["threshold"] = 0.7
        self.assertIsNone(viz._inspector_layer.resolver(1))

    def test_shrink_data_disables_feedmod_intent(self) -> None:
        viz = self._bare_visualizer()
        viz._feed_modulus_leader_enabled = True
        viz._shrinkage_displayed_voxel_indices = np.array([0, 1, 2], dtype=np.int32)
        viz.set_shrink_inspector_data(
            display_intensity=np.array([0.0, 0.0, 0.9], dtype=np.float32),
            mesh=self._stub_mesh(),
            shrinkage_risk=None,
            porosity_risk=None,
            threshold=0.5,
        )
        self.assertFalse(viz._feed_modulus_leader_enabled)
        self.assertEqual(viz._inspector_layer.mode, "shrinkage")

    def test_enable_feedmod_disables_shrink_intent(self) -> None:
        viz = self._bare_visualizer()
        viz._shrinkage_displayed_voxel_indices = np.array([0, 1, 2], dtype=np.int32)
        viz.set_shrink_inspector_data(
            display_intensity=np.array([0.0, 0.0, 0.9], dtype=np.float32),
            mesh=self._stub_mesh(),
            shrinkage_risk=None,
            porosity_risk=None,
            threshold=0.5,
        )
        self.assertIsNotNone(viz._shrink_inspector_data)
        viz.enable_feed_modulus_leader(True)
        self.assertIsNone(viz._shrink_inspector_data)
        self.assertEqual(viz._inspector_layer.mode, "feed_modulus")

    def test_feedmod_resolver_formats_value(self) -> None:
        viz = self._bare_visualizer()
        centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        sizes = np.ones(2, dtype=np.float32)
        values = np.array([0.25, 0.50], dtype=np.float32)
        viz.current_shrinkage_data = (centers, sizes, values)
        viz.shrinkage_actor = _RecordingActor()
        viz._shrinkage_displayed_voxel_indices = np.array([0, 1], dtype=np.int32)
        viz._shrinkage_cells_per_glyph = 6
        viz.enable_feed_modulus_leader(True)
        self.assertIsNotNone(viz._inspector_layer)
        self.assertEqual(viz._inspector_layer.mode, "feed_modulus")
        callout = viz._inspector_layer.resolver(0)
        self.assertIsNotNone(callout)
        self.assertEqual(callout.text, "FM: 0.25 in")
        callout1 = viz._inspector_layer.resolver(1)
        self.assertIsNotNone(callout1)
        self.assertEqual(callout1.text, "FM: 0.50 in")


if __name__ == "__main__":
    unittest.main()
