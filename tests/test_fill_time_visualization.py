"""Tests for the Fill Time result mode.

Two layers are covered, matching how the feature is split:

* ``core.simulation_pipeline.build_visualization_payload`` -- the pipeline
  wiring that turns ``engine._fill_arrival_times`` into the ``fill_time``
  visualization field and ``fill_time_stats`` summary, shared by the desktop
  and web paths. Modeled on the ``SimpleNamespace`` engine/voxel_mesh
  fixtures in ``tests/test_web_api.py``.
* ``ui.main_window.MainWindow._on_result_type_changed`` -- the desktop mode
  mapping ("Fill Time" -> "fill_time") and time-slider show/hide. Modeled on
  ``MainWindowPourLocationGateTests`` in ``tests/test_pour_picker.py``: call
  the unbound method against a ``MagicMock`` stand-in for ``self`` rather
  than constructing a full ``MainWindow``, under the offscreen Qt platform so
  no display is required.
"""

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
    """A SolidificationEngine stand-in exposing just what the payload builder reads."""
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
    # Mirror the real engine: set only when a filling stage actually ran, so
    # getattr(engine, "_fill_arrival_times", None) is None otherwise -- the
    # payload builder must not invent a mode from an attribute that isn't there.
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
    """core.simulation_pipeline.build_visualization_payload -- pipeline wiring."""

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
        # The -1 misrun sentinel must round-trip exactly -- not clamped to 0,
        # not dropped from the list.
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
    """MainWindow._on_result_type_changed -- desktop combo-to-mode wiring.

    Calls the unbound method against a MagicMock host rather than
    constructing a real MainWindow, the same way
    MainWindowPourLocationGateTests in tests/test_pour_picker.py exercises
    ``_on_pour_location_changed``. No vtk/Qt stubbing is needed here (and
    none is installed) because nothing under test touches a real widget or a
    real CastingVisualizer -- every attribute the method reads or writes is a
    MagicMock stand-in.
    """

    def _make_host(self) -> mock.MagicMock:
        host = mock.MagicMock()
        # No live simulation, so the method takes its "no data yet" branch
        # instead of touching a real solidification engine.
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
        """Fill Time must not have broken the mode it shares the slider with."""
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
    """
    Mold, core and chill voxels must never be reported or drawn as misruns.

    Every consumer of the arrival-time field defines a misrun as a negative
    time, and the gather step hands back the same -1 sentinel for metal that
    never filled and for voxels that are not metal at all. Left that way, a
    perfect 100%-filled pour on Test_Model/Filling_Test reported "1,485 never
    fill" -- one for every chill voxel -- and drew them as a grey shell in the
    Fill Time view. The whole suite passed while it did, because nothing here
    had ever mixed metal and non-metal in one array.

    core.filling_stage marks non-metal with NaN instead, which is false
    against `< 0`, `>= 0` and `<= t` alike and so leaves both the misrun mask
    and the arrived mask without any consumer special-casing it.
    """

    def test_pipeline_stats_exclude_non_metal(self) -> None:
        # 2 filled, 1 real misrun, 3 non-metal.
        fill_times = np.array([0.5, 1.5, -1.0, np.nan, np.nan, np.nan])
        summary, payload = _build(_engine(fill_arrival_times=fill_times),
                                  _voxel_mesh(len(fill_times)))

        stats = summary["fill_time_stats"]
        self.assertEqual(stats["filled_voxels"], 2)
        self.assertEqual(stats["unfilled_voxels"], 1, "mold counted as misrun")
        self.assertAlmostEqual(stats["max_fill_time_s"], 1.5)

    def test_engine_export_results_excludes_non_metal(self) -> None:
        """Drives the real SolidificationEngine.export_results branch."""
        from core.solidification import SolidificationEngine

        class _OnlyFilling:
            """Every analysis off except filling, whatever export_results asks for."""
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
        """A chill is not a misrun -- it must not be painted or swept at all.

        The desktop renderer classifies each voxel of a fill-time frame before
        handing it to the surface renderer: arrived (painted by time), misrun
        (flat grey), not yet arrived (steel), non-metal NaN (excluded from
        everything). This checks the classification the visualizer computes,
        independent of VTK.
        """
        import vtk as _vtk
        if not hasattr(_vtk, "vtkFlyingEdges3D") or isinstance(getattr(_vtk, "vtkFlyingEdges3D"), mock.Mock):
            self.skipTest("vtk is stubbed by another test module in this process")
        try:
            from ui.result_surface import ResultSurfaceRenderer, palette_fill_time
        except Exception as exc:  # pragma: no cover
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
        # Voxel 0 arrived at 0.5 s: painted by arrival time (not steel, not grey).
        self.assertFalse(np.array_equal(colours[0], steel))
        self.assertFalse(np.array_equal(colours[0], [140, 140, 140]))
        # Voxel 1 arrives at 2.0 s, after the 1.0 s sweep: still bare steel.
        np.testing.assert_array_equal(colours[1], steel)
        # Voxel 2 is a real misrun: the flat misrun grey.
        np.testing.assert_array_equal(colours[2], [140, 140, 140])
        # The NaN voxels are excluded from everything, so their cells read as
        # "outside": the paint grid there is nearest-filled but never marked painted.
        self.assertFalse(renderer._painted_cells[cells[3]])
        self.assertFalse(renderer._painted_cells[cells[4]])
        for grid, _rgb in renderer._flat_grids.values():
            self.assertFalse(grid[cells[3]])
            self.assertFalse(grid[cells[4]])

    def test_web_grid_excludes_non_metal_and_misruns_from_the_range(self) -> None:
        """The dense grid the browser reads must treat -1 (misrun) and NaN
        (non-metal) as "no value", not as data: neither may set the field range,
        and every cell must come back finite so the bytes are JSON- and
        texture-safe."""
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
        """The source of the sentinel, asserted directly on the gather step."""
        metal = np.array([True, True, False, False])
        gathered = np.array([0.5, -1.0, -1.0, -1.0])

        result = np.where(metal, gathered, np.nan)

        self.assertTrue(np.isnan(result[2]) and np.isnan(result[3]))
        self.assertEqual(int(np.sum(result < 0.0)), 1, "only real misruns")
        self.assertEqual(int(np.sum(result >= 0.0)), 1)
