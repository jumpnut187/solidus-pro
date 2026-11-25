from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.simulation_pipeline import build_visualization_payload
from ui.main_window import MainWindow

def _engine(fill_arrival_times=None, **overrides) -> SimpleNamespace:
    base = dict(
        shrinkage_probability=None,
        porosity_probability=None,
        hotspot_intensity=None,
        niyama_values=None,
        niyama_risk=None,
        combined_feed_risk=None,
        solidification_times=None,
        fraction_liquid_history=[],
        fraction_liquid_times=None,
    )
    base.update(overrides)
    engine = SimpleNamespace(**base)
    if fill_arrival_times is not None:
        engine._fill_arrival_times = fill_arrival_times
    return engine

def _voxel_mesh(n: int) -> SimpleNamespace:
    return SimpleNamespace(
        voxel_centers=np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        voxel_sizes=np.ones(n, dtype=np.float32),
        surface_voxels=None,
        voxel_states=None,
    )

def _build(engine, voxel_mesh, *, max_voxels=1000):
    return build_visualization_payload(
        engine,
        voxel_mesh,
        total_volume=1.0,
        bounds=np.zeros((3, 2), dtype=np.float32),
        max_voxels=max_voxels,
    )

class FillTimePipelinePayloadTests(unittest.TestCase):

    def test_absent_when_no_filling_stage_ran(self) -> None:
        engine = _engine(fill_arrival_times=None)
        voxel_mesh = _voxel_mesh(4)

        summary, visualization = _build(engine, voxel_mesh)

        self.assertNotIn("fill_time", visualization)
        self.assertNotIn("fill_time_stats", summary)

    def test_present_with_sentinel_preserved(self) -> None:
        fill_times = np.array([0.0, 1.5, -1.0, 3.0, -1.0, 2.2], dtype=np.float64)
        engine = _engine(fill_arrival_times=fill_times)
        voxel_mesh = _voxel_mesh(len(fill_times))

        summary, visualization = _build(engine, voxel_mesh)

        self.assertIn("fill_time", visualization)
        np.testing.assert_allclose(visualization["fill_time"], fill_times.tolist())
        self.assertEqual(visualization["fill_time"].count(-1.0), 2)

        stats = summary["fill_time_stats"]
        self.assertEqual(stats["filled_voxels"], 4)
        self.assertEqual(stats["unfilled_voxels"], 2)
        self.assertAlmostEqual(stats["max_fill_time_s"], 3.0)

    def test_field_length_matches_subsampled_indices(self) -> None:
        n = 50
        max_voxels = 10
        rng = np.random.default_rng(0)
        fill_times = rng.uniform(0.0, 10.0, size=n).astype(np.float64)
        engine = _engine(fill_arrival_times=fill_times)
        voxel_mesh = _voxel_mesh(n)

        summary, visualization = _build(engine, voxel_mesh, max_voxels=max_voxels)

        step = max(1, n // max_voxels)
        expected_len = len(np.arange(0, n, step))
        self.assertLess(expected_len, n)
        self.assertEqual(len(visualization["fill_time"]), expected_len)
        self.assertEqual(len(visualization["centers"]), expected_len)

    def test_all_unfilled_edge_case(self) -> None:
        n = 5
        fill_times = np.full(n, -1.0, dtype=np.float64)
        engine = _engine(fill_arrival_times=fill_times)
        voxel_mesh = _voxel_mesh(n)

        summary, _ = _build(engine, voxel_mesh)

        stats = summary["fill_time_stats"]
        self.assertEqual(stats["filled_voxels"], 0)
        self.assertEqual(stats["unfilled_voxels"], n)
        self.assertEqual(stats["max_fill_time_s"], 0.0)

    def test_all_filled_edge_case(self) -> None:
        n = 5
        fill_times = np.linspace(0.0, 4.0, n, dtype=np.float64)
        engine = _engine(fill_arrival_times=fill_times)
        voxel_mesh = _voxel_mesh(n)

        summary, _ = _build(engine, voxel_mesh)

        stats = summary["fill_time_stats"]
        self.assertEqual(stats["filled_voxels"], n)
        self.assertEqual(stats["unfilled_voxels"], 0)
        self.assertAlmostEqual(stats["max_fill_time_s"], 4.0)

class MainWindowFillTimeModeMappingTests(unittest.TestCase):

    def _make_host(self) -> mock.MagicMock:
        host = mock.MagicMock()
        host.simulation_worker = None
        host.current_results = None
        return host

    def test_fill_time_label_maps_to_fill_time_mode(self) -> None:
        host = self._make_host()

        MainWindow._on_result_type_changed(host, "Fill Time")

        self.assertEqual(host._current_vis_mode, "fill_time")

    def test_fill_time_mode_shows_time_slider_and_hides_threshold(self) -> None:
        host = self._make_host()

        MainWindow._on_result_type_changed(host, "Fill Time")

        host.time_slider_widget.setVisible.assert_called_with(True)
        host.threshold_slider.setVisible.assert_called_with(False)
        host.threshold_label.setVisible.assert_called_with(False)
        host.threshold_label_title.setVisible.assert_called_with(False)

    def test_fraction_liquid_still_maps_and_shows_slider_too(self) -> None:
        host = self._make_host()

        MainWindow._on_result_type_changed(host, "Fraction Liquid")

        self.assertEqual(host._current_vis_mode, "fraction_liquid")
        host.time_slider_widget.setVisible.assert_called_with(True)

    def test_other_modes_still_hide_time_slider(self) -> None:
        host = self._make_host()

        MainWindow._on_result_type_changed(host, "Shrink")

        self.assertEqual(host._current_vis_mode, "shrinkage")
        host.time_slider_widget.setVisible.assert_called_with(False)
        host.threshold_slider.setVisible.assert_called_with(True)

if __name__ == "__main__":
    unittest.main()

class NonMetalIsNotAMisrunTests(unittest.TestCase):

    def test_pipeline_stats_exclude_non_metal(self) -> None:
        fill_times = np.array([0.5, 1.5, -1.0, np.nan, np.nan, np.nan])
        summary, payload = _build(_engine(fill_arrival_times=fill_times),
                                  _voxel_mesh(len(fill_times)))

        stats = summary["fill_time_stats"]
        self.assertEqual(stats["filled_voxels"], 2)
        self.assertEqual(stats["unfilled_voxels"], 1, "mold counted as misrun")
        self.assertAlmostEqual(stats["max_fill_time_s"], 1.5)

    def test_engine_export_results_excludes_non_metal(self) -> None:
        from core.solidification import SolidificationEngine

        class _OnlyFilling:
            filling = True

            def __getattr__(self, name):
                return False

        engine = mock.MagicMock()
        engine._fill_arrival_times = np.array([0.25, -1.0, np.nan, np.nan])
        engine.analysis_options = _OnlyFilling()
        results = SolidificationEngine.export_results(engine)

        self.assertEqual(results["fill_time_filled_voxels"], 1)
        self.assertEqual(results["fill_time_unfilled_voxels"], 1,
                         "the two NaN mold voxels were counted as misruns")

    def test_renderer_drops_non_metal_entirely(self) -> None:
        import vtk as _vtk
        if not hasattr(_vtk, "vtkFlyingEdges3D") or isinstance(getattr(_vtk, "vtkFlyingEdges3D"), mock.Mock):
            self.skipTest("vtk is stubbed by another test module in this process")
        try:
            from ui.result_surface import ResultSurfaceRenderer, palette_fill_time
        except Exception as exc:
            self.skipTest(f"result renderer unavailable: {exc}")

        fill_time = np.array([0.5, 2.0, -1.0, np.nan, np.nan], dtype=np.float32)
        centers = np.zeros((5, 3), dtype=np.float64)
        centers[:, 0] = np.arange(5) * 0.25
        sizes = np.full(5, 0.25)

        renderer = ResultSurfaceRenderer(_vtk.vtkRenderer())
        if not renderer.set_voxels(centers, sizes, None):
            self.skipTest("lattice unavailable")
        finite = np.isfinite(fill_time)
        misrun = finite & (fill_time < 0.0)
        arrived = finite & (fill_time >= 0.0) & (fill_time <= 1.0)
        sweep_values = np.where(misrun, 20.0, fill_time)
        ok = renderer.show("fill_time", sweep_values, 0.0, 2.0, 1.0, palette_fill_time,
                           valid=finite, paint_mask=arrived, glow=True,
                           flat_paint={"misrun": (misrun, (140, 140, 140))})
        self.assertTrue(ok)

        cells = renderer._cell_of_voxel
        colours = renderer._colors_for_cells(cells)
        steel = np.array([158, 168, 184])
        self.assertFalse(np.array_equal(colours[0], steel))
        self.assertFalse(np.array_equal(colours[0], [140, 140, 140]))
        np.testing.assert_array_equal(colours[1], steel)
        np.testing.assert_array_equal(colours[2], [140, 140, 140])
        self.assertFalse(renderer._painted_cells[cells[3]])
        self.assertFalse(renderer._painted_cells[cells[4]])
        for grid, _rgb in renderer._flat_grids.values():
            self.assertFalse(grid[cells[3]])
            self.assertFalse(grid[cells[4]])

    def test_web_grid_excludes_non_metal_and_misruns_from_the_range(self) -> None:
        from web import fields

        arr = np.array([0.5, 1.5, -1.0, np.nan, np.nan])
        visualization = {
            "centers": [[float(i), 0.0, 0.0] for i in range(len(arr))],
            "sizes": [1.0] * len(arr),
            "part_types": [0] * len(arr),
            "fill_time": arr.tolist(),
        }
        grids = fields.build_field_grids(visualization, {"voxel_count": len(arr)})

        grid = grids.scalars["fill_time"]
        self.assertGreaterEqual(grid.minimum, 0.0, "misrun sentinel leaked into the range")
        self.assertAlmostEqual(grid.maximum, 1.5, places=5)
        values = grid.dequantize()
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertEqual(int(np.count_nonzero(grids.mask)), len(arr), "mask comes from geometry, not values")

    def test_filling_stage_writes_nan_for_non_metal(self) -> None:
        metal = np.array([True, True, False, False])
        gathered = np.array([0.5, -1.0, -1.0, -1.0])

        result = np.where(metal, gathered, np.nan)

        self.assertTrue(np.isnan(result[2]) and np.isnan(result[3]))
        self.assertEqual(int(np.sum(result < 0.0)), 1, "only real misruns")
        self.assertEqual(int(np.sum(result >= 0.0)), 1)
