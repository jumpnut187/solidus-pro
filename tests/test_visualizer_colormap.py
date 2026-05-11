import importlib
import sys
import threading
import types
import unittest

import numpy as np


class _DummyVTKClass:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return None
        return _method


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

    def add_done_callback(self, callback):
        callback(self)


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        return _ImmediateFuture(fn(*args, **kwargs))


def _install_visualizer_import_stubs() -> None:
    vtk_module = types.ModuleType("vtk")
    vtk_module.VTK_FLOAT = 1
    vtk_module.VTK_UNSIGNED_CHAR = 2

    def _vtk_getattr(name):
        return _DummyVTKClass

    vtk_module.__getattr__ = _vtk_getattr  # type: ignore[attr-defined]

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

    sys.modules.setdefault("vtk", vtk_module)
    sys.modules.setdefault("vtk.util", vtk_util_module)
    sys.modules.setdefault("vtk.util.numpy_support", numpy_support_module)
    sys.modules.setdefault("vtk.qt", vtk_qt_module)
    sys.modules.setdefault("vtk.qt.QVTKRenderWindowInteractor", qvtk_module)
    sys.modules.setdefault("PyQt5", pyqt5_module)
    sys.modules.setdefault("PyQt5.QtWidgets", qtwidgets_module)
    sys.modules.setdefault("PyQt5.QtCore", qtcore_module)


class VisualizerColormapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_visualizer_import_stubs()
        cls.visualizer = importlib.import_module("ui.visualizer")

    def test_apply_colormap_matches_scalar_mapping(self) -> None:
        intensities = np.array([-0.2, 0.0, 0.1, 0.25, 0.5, 0.73, 1.0, 1.5], dtype=np.float32)

        colors = self.visualizer.apply_colormap_to_array(
            intensities,
            colormap="viridis",
            min_val=0.0,
            max_val=1.0,
        )

        expected = np.array(
            [self.visualizer.get_color_for_value(float(np.clip(value, 0.0, 1.0)), "viridis") for value in intensities],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(colors, expected)

    def test_apply_colormap_uses_zero_band_when_scale_collapses(self) -> None:
        intensities = np.array([0.0, 0.4, 1.0], dtype=np.float32)

        colors = self.visualizer.apply_colormap_to_array(
            intensities,
            colormap="coolwarm",
            min_val=2.0,
            max_val=2.0,
        )

        expected_color = np.array(self.visualizer.get_color_for_value(0.0, "coolwarm"), dtype=np.uint8)
        expected = np.repeat(expected_color[np.newaxis, :], len(intensities), axis=0)

        np.testing.assert_array_equal(colors, expected)

    def test_compute_top_percent_mask_selects_expected_count(self) -> None:
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        mask = self.visualizer.compute_top_percent_mask(values, 0.4)
        np.testing.assert_array_equal(mask, np.array([False, False, False, True, True]))

    def test_compute_top_percent_mask_respects_base_mask(self) -> None:
        values = np.array([0.1, 0.8, 0.2, 0.7, 0.9], dtype=np.float32)
        base_mask = np.array([False, True, False, True, False])
        mask = self.visualizer.compute_top_percent_mask(values, 0.5, base_mask)
        np.testing.assert_array_equal(mask, np.array([False, True, False, False, False]))

    def test_compute_top_percent_mask_handles_extremes(self) -> None:
        values = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        none_mask = self.visualizer.compute_top_percent_mask(values, 0.0)
        all_mask = self.visualizer.compute_top_percent_mask(values, 1.0)
        np.testing.assert_array_equal(none_mask, np.array([False, False, False]))
        np.testing.assert_array_equal(all_mask, np.array([True, True, True]))

    def _bare_visualizer(self):
        viz = self.visualizer.CastingVisualizer.__new__(self.visualizer.CastingVisualizer)
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
        viz.shrinkage_actor = None
        viz.hotspot_actor = None
        viz.niyama_actor = None
        viz.hot_tear_actor = None
        viz.sand_burn_in_actor = None
        viz.fraction_liquid_actor = None
        viz.renderer = _DummyVTKClass()
        viz.render_window = _DummyVTKClass()
        return viz

    def test_large_result_arrays_do_not_prebuild_heavy_caches(self) -> None:
        viz = self._bare_visualizer()
        voxel_centers = np.zeros((20_001, 3), dtype=np.float32)
        voxel_sizes = np.ones(20_001, dtype=np.float32)
        intensities = np.ones(20_001, dtype=np.float32)
        built_thresholds = []

        def _build(*args):
            built_thresholds.append(args[-1])
            return _DummyVTKClass()

        viz._build_hotspot_geometry = _build
        viz._prebuild_hotspot_cache(voxel_centers, voxel_sizes, intensities, 0.50)

        self.assertEqual(viz._hotspot_cache, {})
        self.assertEqual(built_thresholds, [])

    def test_small_result_arrays_cache_bounded_nearby_thresholds(self) -> None:
        viz = self._bare_visualizer()
        voxel_centers = np.zeros((10, 3), dtype=np.float32)
        voxel_sizes = np.ones(10, dtype=np.float32)
        intensities = np.linspace(0.0, 1.0, 10, dtype=np.float32)

        viz._build_hotspot_geometry = lambda *args: self.visualizer.vtk.vtkPolyData()
        viz._prebuild_hotspot_cache(voxel_centers, voxel_sizes, intensities, 0.50)

        self.assertLessEqual(len(viz._hotspot_cache), viz.MAX_CACHED_THRESHOLD_LEVELS)
        self.assertEqual(set(viz._hotspot_cache), {0.48, 0.49, 0.51, 0.52})

    def test_switching_modes_clears_stale_cached_polydata(self) -> None:
        viz = self._bare_visualizer()
        viz._shrinkage_cache[0.5] = _DummyVTKClass()
        viz._hotspot_cache[0.5] = _DummyVTKClass()
        viz._niyama_cache[0.5] = _DummyVTKClass()
        viz._hot_tear_cache[0.5] = _DummyVTKClass()
        viz._sand_burn_in_cache[0.5] = _DummyVTKClass()

        viz.clear_result_actors()

        self.assertEqual(viz._shrinkage_cache, {})
        self.assertEqual(viz._hotspot_cache, {})
        self.assertEqual(viz._niyama_cache, {})
        self.assertEqual(viz._hot_tear_cache, {})
        self.assertEqual(viz._sand_burn_in_cache, {})


if __name__ == "__main__":
    unittest.main()
