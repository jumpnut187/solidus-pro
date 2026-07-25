import importlib
import sys
import threading
import types
import unittest


class _DummyVTKClass:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return None
        return _method


class _TrackingCamera:
    def __init__(self):
        self._parallel = True
        self._parallel_scale = 120.0
        self._position = (200, 200, 200)
        self._focal = (0, 0, 0)
        self._view_up = (0, 0, 1)

    def ParallelProjectionOn(self):
        self._parallel = True

    def ParallelProjectionOff(self):
        self._parallel = False

    def ParallelProjectionIsOn(self):
        return self._parallel

    def SetParallelScale(self, scale):
        self._parallel_scale = scale

    def GetParallelScale(self):
        return self._parallel_scale

    def SetPosition(self, x, y, z):
        self._position = (x, y, z)

    def SetFocalPoint(self, x, y, z):
        self._focal = (x, y, z)

    def SetViewUp(self, x, y, z):
        self._view_up = (x, y, z)

    def GetViewUp(self):
        return self._view_up

    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return None
        return _method


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        import threading as _t
        f = _t.Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except BaseException as e:
            f.set_exception(e)
        return f


class _DummySignal:
    def emit(self, *args, **kwargs):
        return None


def _install_stubs() -> None:
    vtk_module = types.ModuleType("vtk")
    vtk_module.VTK_FLOAT = 1
    vtk_module.VTK_UNSIGNED_CHAR = 2

    def _vtk_getattr(name):
        if name == "vtkCamera":
            return _TrackingCamera
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


class VisualizerProjectionModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_stubs()
        cls.visualizer = importlib.import_module("ui.visualizer")

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
        viz._gas_porosity_cache = {}
        viz.shrinkage_actor = None
        viz.hotspot_actor = None
        viz.niyama_actor = None
        viz.hot_tear_actor = None
        viz.sand_burn_in_actor = None
        viz.gas_porosity_actor = None
        viz.fraction_liquid_actor = None
        viz.model_bounds = None
        viz._cross_section_enabled = False
        viz.cross_section_planes = {}
        viz.measurement_tool = None
        viz.feed_modulus_leader = None
        viz._feed_modulus_leader_enabled = False
        viz._animation_frames = []
        viz._is_recording = False
        viz._interactor_initialized = False
        viz._is_orthographic = True
        viz._active_camera = _TrackingCamera()
        viz.renderer = _DummyVTKClass()
        viz.render_window = _DummyVTKClass()
        # Attach a persistent tracking camera
        viz.renderer.GetActiveCamera = lambda: viz._active_camera
        viz.render_window.Render = lambda: None
        return viz

    def test_default_is_orthographic(self) -> None:
        viz = self._bare_visualizer()
        viz._active_camera.ParallelProjectionOn()
        self.assertTrue(viz.is_orthographic)

    def test_set_perspective_disables_parallel(self) -> None:
        viz = self._bare_visualizer()
        viz._active_camera.ParallelProjectionOn()
        viz.set_projection_mode('perspective')
        self.assertFalse(viz._active_camera._parallel)
        self.assertFalse(viz.is_orthographic)

    def test_set_orthographic_enables_parallel(self) -> None:
        viz = self._bare_visualizer()
        viz._active_camera.ParallelProjectionOff()
        viz.set_projection_mode('orthographic')
        self.assertTrue(viz._active_camera._parallel)
        self.assertTrue(viz.is_orthographic)

    def test_set_projection_ignores_invalid_mode(self) -> None:
        viz = self._bare_visualizer()
        viz._active_camera.ParallelProjectionOn()
        viz.set_projection_mode('invalid')
        self.assertTrue(viz._active_camera._parallel)
        self.assertTrue(viz.is_orthographic)

    def test_set_camera_view_does_not_switch_projection(self) -> None:
        viz = self._bare_visualizer()
        viz.model_bounds = (-5, 5, -5, 5, -5, 5)
        viz._active_camera.ParallelProjectionOn()
        viz.set_camera_view('top')
        self.assertTrue(viz._active_camera._parallel)
        self.assertTrue(viz.is_orthographic)

    def test_reset_camera_preserves_orthographic(self) -> None:
        viz = self._bare_visualizer()
        viz.model_bounds = (-5, 5, -5, 5, -5, 5)
        viz._active_camera.ParallelProjectionOn()
        viz.reset_camera()
        self.assertTrue(viz._active_camera._parallel)
        self.assertTrue(viz.is_orthographic)

    def test_parallel_scale_updates_on_model_bounds(self) -> None:
        viz = self._bare_visualizer()
        viz.model_bounds = (0, 10, 0, 10, 0, 10)
        viz._active_camera.ParallelProjectionOn()
        viz._update_parallel_scale()
        expected = max(10, 10, 10, 1) * 1.15
        self.assertAlmostEqual(viz._active_camera._parallel_scale, expected)


if __name__ == "__main__":
    unittest.main()
