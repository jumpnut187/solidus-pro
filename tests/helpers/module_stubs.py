from __future__ import annotations

import sys
import types
from typing import Callable, Dict, Iterable

QT_BINDING = "PySide6"

QT_SUBMODULES = ("QtCore", "QtGui", "QtWidgets")

QT_STUB_MODULES = (QT_BINDING,) + tuple(
    f"{QT_BINDING}.{sub}" for sub in QT_SUBMODULES
)

VTK_QT_STUB_MODULES = (
    "vtkmodules.qt",
    "vtkmodules.qt.QVTKRenderWindowInteractor",
)

VISUALIZER_STUB_MODULES = (
    (
        "vtk",
        "vtk.util",
        "vtk.util.numpy_support",
    )
    + VTK_QT_STUB_MODULES
    + QT_STUB_MODULES
)

VISUALIZER_MODULES = ("ui.visualizer", "ui.qt")

VISUALIZER_MODULE = VISUALIZER_MODULES[0]

STUB_TAINTED_MODULES = ("ui.qt",)

_ABSENT = object()

_STUB_MARKER = "_castsim_stub_module"

_DISPLACED: Dict[str, object] = {}

def make_qt_stub_modules(
    dummy: type,
    signal_factory: Callable[..., object],
) -> Dict[str, object]:

    def _any_attr(attr: str, _dummy: type = dummy) -> type:
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _dummy

    package = types.ModuleType(QT_BINDING)
    modules: Dict[str, object] = {QT_BINDING: package}

    for sub in QT_SUBMODULES:
        name = f"{QT_BINDING}.{sub}"
        module = types.ModuleType(name)
        module.__getattr__ = _any_attr
        setattr(package, sub, module)
        modules[name] = module

    modules[f"{QT_BINDING}.QtCore"].Signal = signal_factory
    return modules

def make_vtk_qt_stub_modules(dummy: type) -> Dict[str, object]:
    interactor = types.ModuleType("vtkmodules.qt.QVTKRenderWindowInteractor")
    interactor.QVTKRenderWindowInteractor = dummy

    qt_package = types.ModuleType("vtkmodules.qt")
    qt_package.QVTKRenderWindowInteractor = interactor

    return {
        "vtkmodules.qt": qt_package,
        "vtkmodules.qt.QVTKRenderWindowInteractor": interactor,
    }

def install_stub_modules(modules: Dict[str, object]) -> None:
    for name, module in modules.items():
        current = sys.modules.get(name, _ABSENT)
        if current is not _ABSENT and not getattr(current, _STUB_MARKER, False):
            _DISPLACED[name] = current
        setattr(module, _STUB_MARKER, True)
        sys.modules[name] = module
    for name in VISUALIZER_MODULES:
        sys.modules.pop(name, None)

def snapshot_modules(
    names: Iterable[str] = VISUALIZER_STUB_MODULES + STUB_TAINTED_MODULES,
) -> Dict[str, object]:
    return {name: sys.modules.get(name, _ABSENT) for name in names}

def restore_modules(snapshot: Dict[str, object]) -> None:
    for name, original in snapshot.items():
        displaced = _DISPLACED.pop(name, _ABSENT)
        if displaced is not _ABSENT:
            sys.modules[name] = displaced
        elif original is _ABSENT:
            if getattr(sys.modules.get(name), _STUB_MARKER, False):
                sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
