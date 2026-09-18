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
        viz._shrinkage_cache = {}
        viz._shrinkage_cache_built = False
        viz._shrinkage_cache_lock = threading.Lock()
        viz._shrinkage_cache_generation = 0
        viz._pending_shrinkage_thresholds = set()
        viz._hotspot_cache = {}
        viz._niyama_cache = {}
        viz._hot_tear_cache = {}
        viz._sand_burn_in_cache = {}
        viz._gas_porosity_cache = {}
        viz._fraction_liquid_cache = OrderedDict()
        viz._fraction_liquid_cache_point_count = 0
        viz._fill_time_cache = OrderedDict()
        viz._fill_time_cache_point_count = 0
        viz._fill_animation_cache = OrderedDict()
        viz._fill_animation_cache_point_count = 0
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

    def test_setting_mask_invalidates_cached_geometry(self) -> None:
        viz = self._bare_visualizer()
        viz._shrinkage_cache[0.5] = _DummyVTKClass()
        viz._hotspot_cache[0.5] = _DummyVTKClass()
        viz._fill_time_cache[3] = _DummyVTKClass()
        viz._fill_time_cache_point_count = 12

        viz.set_voxel_display_mask(np.ones(4, dtype=bool))

        self.assertEqual(viz._shrinkage_cache, {})
        self.assertEqual(viz._hotspot_cache, {})
        self.assertEqual(viz._fill_time_cache, {})
        self.assertEqual(viz._fill_time_cache_point_count, 0)

    def test_display_voxel_count_respects_mask(self) -> None:
        viz = self._bare_visualizer()
        values = np.array([0.9, 0.9, 0.9, 0.1], dtype=np.float32)
        self.assertEqual(viz._display_voxel_count(values, 0.5), 3)

        viz.set_voxel_display_mask(np.array([True, False, True, True]))
        self.assertEqual(viz._display_voxel_count(values, 0.5), 2)


if __name__ == "__main__":
    unittest.main()
