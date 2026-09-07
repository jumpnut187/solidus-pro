import importlib
import sys
import types
import unittest

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


class _SpyVTKPlane:
    """Minimal spy for vtkPlane that records normal and origin."""

    def __init__(self, *args, **kwargs):
        self._normal = (0.0, 0.0, 0.0)
        self._origin = (0.0, 0.0, 0.0)

    def SetNormal(self, x, y, z):
        self._normal = (x, y, z)

    def GetNormal(self):
        return self._normal

    def SetOrigin(self, x, y, z):
        self._origin = (x, y, z)

    def GetOrigin(self):
        return self._origin


class _DummyActorCollection:
    def __init__(self):
        self._actors = []

    def GetNumberOfItems(self):
        return len(self._actors)

    def GetItemAsObject(self, index):
        return self._actors[index]

    def AddItem(self, actor):
        self._actors.append(actor)


class _DummyRenderer:
    def __init__(self):
        self._actors = _DummyActorCollection()

    def GetActors(self):
        return self._actors

    def AddActor(self, actor):
        self._actors.AddItem(actor)


class _ChainingDummy:
    """Dummy VTK object where any method returns self, enabling chaining."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return self
        return _method


class _DummyVTKClass:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return None
        return _method


def _install_cross_section_stubs() -> None:
    vtk_module = types.ModuleType("vtk")
    vtk_module.VTK_FLOAT = 1
    vtk_module.VTK_UNSIGNED_CHAR = 2

    def _vtk_getattr(name):
        if name == "vtkPlane":
            return _SpyVTKPlane
        if name == "vtkInteractorObserver":
            return _DummyVTKClass
        if name == "vtkRenderer":
            return _DummyRenderer
        return _ChainingDummy

    vtk_module.__getattr__ = _vtk_getattr  # type: ignore[attr-defined]

    vtk_util_module = types.ModuleType("vtk.util")
    numpy_support_module = types.ModuleType("vtk.util.numpy_support")
    numpy_support_module.numpy_to_vtk = lambda *args, **kwargs: _DummyVTKClass()
    vtk_util_module.numpy_support = numpy_support_module

    # Forced install; tearDownModule restores the real modules. Also drops
    # ui.visualizer/ui.main_window so they re-import against these stubs.
    install_stub_modules({
        "vtk": vtk_module,
        "vtk.util": vtk_util_module,
        "vtk.util.numpy_support": numpy_support_module,
        **make_vtk_qt_stub_modules(_DummyVTKClass),
        **make_qt_stub_modules(_DummyVTKClass, lambda *a, **k: _DummyVTKClass()),
    })


class CrossSectionPlaneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_cross_section_stubs()
        cls.visualizer = importlib.import_module("ui.visualizer")

    def test_default_direction_is_positive(self) -> None:
        plane = self.visualizer.CrossSectionPlane(_DummyRenderer(), "X")
        plane.set_bounds((0, 10, 0, 10, 0, 10))
        plane.set_position(5.0)
        self.assertEqual(plane.direction, 1)
        self.assertEqual(plane.plane.GetNormal(), (1.0, 0.0, 0.0))
        self.assertEqual(plane.plane.GetOrigin(), (5.0, 0.0, 0.0))

    def test_flip_direction_changes_normal(self) -> None:
        plane = self.visualizer.CrossSectionPlane(_DummyRenderer(), "X")
        plane.set_bounds((0, 10, 0, 10, 0, 10))
        plane.set_direction(-1)
        plane.set_position(5.0)
        self.assertEqual(plane.direction, -1)
        self.assertEqual(plane.plane.GetNormal(), (-1.0, 0.0, 0.0))
        self.assertEqual(plane.plane.GetOrigin(), (5.0, 0.0, 0.0))

    def test_flip_all_axes(self) -> None:
        for axis, expected_pos in [("X", (5.0, 0.0, 0.0)), ("Y", (0.0, 5.0, 0.0)), ("Z", (0.0, 0.0, 5.0))]:
            with self.subTest(axis=axis):
                plane = self.visualizer.CrossSectionPlane(_DummyRenderer(), axis)
                plane.set_direction(-1)
                plane.set_position(5.0)
                normal = plane.plane.GetNormal()
                if axis == "X":
                    self.assertEqual(normal, (-1.0, 0.0, 0.0))
                elif axis == "Y":
                    self.assertEqual(normal, (0.0, -1.0, 0.0))
                else:
                    self.assertEqual(normal, (0.0, 0.0, -1.0))
                self.assertEqual(plane.plane.GetOrigin(), expected_pos)

    def test_invalid_direction_is_ignored(self) -> None:
        plane = self.visualizer.CrossSectionPlane(_DummyRenderer(), "X")
        plane.set_direction(0)
        self.assertEqual(plane.direction, 1)
        plane.set_direction(2)
        self.assertEqual(plane.direction, 1)

    def test_position_range_from_bounds(self) -> None:
        plane = self.visualizer.CrossSectionPlane(_DummyRenderer(), "Y")
        plane.set_bounds((0, 10, -5, 15, 0, 20))
        self.assertEqual(plane.get_position_range(), (-5, 15))

    def test_hide_visual_keeps_enabled(self) -> None:
        """Hiding the actor should not disable the plane."""
        plane = self.visualizer.CrossSectionPlane(_DummyRenderer(), "X")
        plane.enable(show_plane=True)
        plane.plane_actor.SetVisibility(False)
        self.assertTrue(plane.enabled)


if __name__ == "__main__":
    unittest.main()
