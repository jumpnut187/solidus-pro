import importlib
import sys
import types
import unittest

import numpy as np

from tests.helpers.module_stubs import (
    install_stub_modules,
    make_qt_stub_modules,
    make_vtk_qt_stub_modules,
    restore_modules,
    snapshot_modules,
)

# Captured before any stub is installed, so tearDownModule can hand the real
# vtk/Qt modules back to whatever runs next in this process.
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


class _DummySignal:
    def emit(self, *args, **kwargs):
        return None


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

    # Forced, not setdefault: the stub must win even if an earlier test
    # already imported the real module, or this module's behavior would
    # depend on test execution order. tearDownModule puts them back.
    install_stub_modules({
        "vtk": vtk_module,
        "vtk.util": vtk_util_module,
        "vtk.util.numpy_support": numpy_support_module,
        **make_vtk_qt_stub_modules(_DummyVTKClass),
        **make_qt_stub_modules(_DummyVTKClass, lambda *args, **kwargs: _DummySignal()),
    })


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


if __name__ == "__main__":
    unittest.main()
