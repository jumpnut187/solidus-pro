"""Review-time component hiding in the desktop visualizer.

Hiding a component must (1) switch off that part's skin actor while leaving
the others and the master "Show Model" toggle coherent, and (2) drop that
part's voxels from every result geometry builder through the shared display
mask. VTK and Qt are stubbed the same way as the other visualizer tests.
"""

import importlib
import threading
import types
import unittest
from collections import OrderedDict

import numpy as np

from tests.helpers.module_stubs import (
    install_stub_modules,
    make_qt_stub_modules,
    make_vtk_qt_stub_modules,
    restore_modules,
    snapshot_modules,
)

_MODULE_SNAPSHOT = snapshot_modules()


def tearDownModule() -> None:
    restore_modules(_MODULE_SNAPSHOT)


class _DummyVTKClass:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return None
        return _method


class _ActorStub:
    """Actor that remembers its visibility flag."""

    def __init__(self):
        self.visible = True

    def SetVisibility(self, flag):
        self.visible = bool(flag)

    def GetVisibility(self):
        return int(self.visible)

    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return None
        return _method


class _SurfaceStub:
    """Records what the visualizer forwards to the result renderer."""

    def __init__(self):
        self.mask = "unset"
        self.hidden = set()
        self.view = None
        self.style = "envelope"

    def set_display_mask(self, mask):
        self.mask = mask

    def set_hidden_parts(self, hidden):
        self.hidden = set(hidden)

    def painted_part_ids(self):
        return set()


class _DummySignal:
    def emit(self, *args, **kwargs):
        return None


def _install_stubs() -> None:
    vtk_module = types.ModuleType("vtk")
    vtk_module.VTK_FLOAT = 1
    vtk_module.VTK_UNSIGNED_CHAR = 2
    vtk_module.__getattr__ = lambda name: _DummyVTKClass  # type: ignore[attr-defined]

    vtk_util_module = types.ModuleType("vtk.util")
    numpy_support_module = types.ModuleType("vtk.util.numpy_support")
    numpy_support_module.numpy_to_vtk = lambda *args, **kwargs: _DummyVTKClass()
    vtk_util_module.numpy_support = numpy_support_module

    install_stub_modules({
        "vtk": vtk_module,
        "vtk.util": vtk_util_module,
        "vtk.util.numpy_support": numpy_support_module,
        **make_vtk_qt_stub_modules(_DummyVTKClass),
        **make_qt_stub_modules(_DummyVTKClass, lambda *args, **kwargs: _DummySignal()),
    })


class ComponentVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_stubs()
        cls.visualizer = importlib.import_module("ui.visualizer")

    def _bare_visualizer(self):
        viz = self.visualizer.CastingVisualizer.__new__(self.visualizer.CastingVisualizer)
        viz._hidden_part_ids = set()
        viz._voxel_display_mask = None
        viz._model_visible = True
        viz.part_actors = {}
        viz.mesh_actor = None
        viz.result_surface = _SurfaceStub()
        viz.renderer = _DummyVTKClass()
        viz.render_window = _DummyVTKClass()
        return viz

    # ---- skin actors ------------------------------------------------------

    def test_hidden_component_skin_is_off_and_others_stay_on(self) -> None:
        viz = self._bare_visualizer()
        viz.part_actors = {"cast": _ActorStub(), "riser": _ActorStub()}

        viz.set_component_hidden("riser", True)

        self.assertFalse(viz.part_actors["riser"].visible)
        self.assertTrue(viz.part_actors["cast"].visible)
        self.assertEqual(viz.hidden_part_ids(), {"riser"})

    def test_master_toggle_does_not_forget_hidden_components(self) -> None:
        viz = self._bare_visualizer()
        viz.part_actors = {"cast": _ActorStub(), "riser": _ActorStub()}
        viz.set_component_hidden("riser", True)

        viz.toggle_mesh_visibility(False)
        self.assertFalse(viz.part_actors["cast"].visible)
        self.assertFalse(viz.part_actors["riser"].visible)

        viz.toggle_mesh_visibility(True)
        self.assertTrue(viz.part_actors["cast"].visible)
        self.assertFalse(viz.part_actors["riser"].visible)

    def test_clear_hidden_components_restores_everything(self) -> None:
        viz = self._bare_visualizer()
        viz.part_actors = {"cast": _ActorStub(), "riser": _ActorStub()}
        viz.set_component_hidden("riser", True)
        viz.set_voxel_display_mask(np.array([True, False]))

        viz.clear_hidden_components()

        self.assertTrue(viz.part_actors["riser"].visible)
        self.assertEqual(viz.hidden_part_ids(), set())
        self.assertIsNone(viz.voxel_display_mask())

    # ---- voxel display mask ----------------------------------------------

    def test_display_mask_is_anded_into_builder_selection(self) -> None:
        viz = self._bare_visualizer()
        selection = np.array([True, True, False, True])
        viz.set_voxel_display_mask(np.array([True, False, True, True]))

        np.testing.assert_array_equal(
            viz._apply_display_mask(selection),
            np.array([True, False, False, True]),
        )

    def test_stale_display_mask_of_wrong_length_is_ignored(self) -> None:
        viz = self._bare_visualizer()
        selection = np.array([True, True, False])
        viz.set_voxel_display_mask(np.zeros(7, dtype=bool))

        np.testing.assert_array_equal(viz._apply_display_mask(selection), selection)

    def test_bare_instance_without_mask_attribute_passes_selection_through(self) -> None:
        viz = self.visualizer.CastingVisualizer.__new__(self.visualizer.CastingVisualizer)
        selection = np.array([False, True])
        np.testing.assert_array_equal(viz._apply_display_mask(selection), selection)

    def test_mask_and_hidden_parts_are_forwarded_to_the_result_renderer(self) -> None:
        viz = self._bare_visualizer()
        viz.part_actors = {"riser": _ActorStub()}

        viz.set_voxel_display_mask(np.array([True, False]))
        np.testing.assert_array_equal(viz.result_surface.mask, np.array([True, False]))

        viz.set_component_hidden("riser", True)
        self.assertEqual(viz.result_surface.hidden, {"riser"})

        viz.clear_hidden_components()
        self.assertIsNone(viz.result_surface.mask)
        self.assertEqual(viz.result_surface.hidden, set())


if __name__ == "__main__":
    unittest.main()
