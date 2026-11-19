"""Snapshot/restore for tests that fake ``vtk`` and the Qt binding in ``sys.modules``.

``ui/visualizer.py`` imports ``vtk``, ``vtk.util.numpy_support``,
``vtkmodules.qt.QVTKRenderWindowInteractor`` and ``ui.qt`` at module scope, so
tests that exercise only its pure logic replace those modules with lightweight
stubs before importing it.

Those stubs must be taken back out once the installing module's tests finish.
The stub ``vtk`` answers every method call with ``None``, so leaving it in
``sys.modules`` silently breaks unrelated tests that run later in the same
process. The case that motivated this helper: ``core/surface_extraction.py``
does ``import vtk`` inside ``_extract_with_vtk``, gets the stub, and fails with
``'NoneType' object has no attribute 'SetScalars'``.

Such failures are order-dependent, so they do not reproduce when a module is run
on its own -- only under ``python -m unittest discover tests``.

The Qt binding is named in exactly one place in this package -- :data:`QT_BINDING`
below -- and the stub modules for it are built by :func:`make_qt_stub_modules`
rather than spelled out by each test. Before that consolidation the string
``"PyQt5"`` appeared 33 times across six files, which made changing bindings a
scattered edit and made a missed site look like a genuine test failure.

Usage::

    from tests.helpers.module_stubs import restore_modules, snapshot_modules

    # At module import time, BEFORE any stub is installed.
    _MODULE_SNAPSHOT = snapshot_modules()

    def tearDownModule():
        restore_modules(_MODULE_SNAPSHOT)
"""

from __future__ import annotations

import sys
import types
from typing import Callable, Dict, Iterable

#: The Qt binding, named once. Everything else here derives from it, so moving
#: to another binding is this line plus ``ui/qt.py``.
QT_BINDING = "PySide6"

#: Qt submodules the visualizer stubs replace. ``ui/qt.py`` re-exports names
#: from all three, so all three have to be stubbed even though
#: ``ui/visualizer.py`` itself only needs a handful.
QT_SUBMODULES = ("QtCore", "QtGui", "QtWidgets")

#: Fully-qualified Qt module names, in ``sys.modules`` order.
QT_STUB_MODULES = (QT_BINDING,) + tuple(
    f"{QT_BINDING}.{sub}" for sub in QT_SUBMODULES
)

#: The VTK-Qt bridge. ``vtkmodules.qt`` is the supported path; the older
#: ``vtk.qt`` spelling is a shim over it and is not stubbed here.
VTK_QT_STUB_MODULES = (
    "vtkmodules.qt",
    "vtkmodules.qt.QVTKRenderWindowInteractor",
)

#: Modules replaced by the visualizer test stubs.
VISUALIZER_STUB_MODULES = (
    (
        "vtk",
        "vtk.util",
        "vtk.util.numpy_support",
    )
    + VTK_QT_STUB_MODULES
    + QT_STUB_MODULES
)

#: Imported against the stubs by the visualizer tests, so they have to be
#: dropped before the tests import them or they get whatever an earlier test
#: already cached. ``ui.qt`` is in here because it holds direct references to
#: the real Qt classes once imported -- leaving it cached would hand
#: ``ui.visualizer`` real widgets instead of the stubs just installed.
VISUALIZER_MODULES = ("ui.visualizer", "ui.qt")

#: Retained for callers that still refer to the single-module name.
VISUALIZER_MODULE = VISUALIZER_MODULES[0]

#: Modules that captured stub classes into their own globals and must NOT be
#: left that way once a stub-using module finishes. ``ui.qt`` re-exports the Qt
#: names every widget module builds on, so a stub-bound copy left in
#: ``sys.modules`` hands the next importer dummy base classes. When that next
#: importer is ``ui.main_window`` in a test that also builds a real
#: ``QApplication``, the mismatch is a segfault rather than an exception -- it
#: took out ``tests.test_pour_picker`` under full discovery while every module
#: still passed on its own. Snapshotting it means the entry goes back to the
#: real module, or is removed if there wasn't one.
#:
#: ``ui.visualizer`` is deliberately absent here; see :func:`restore_modules`.
STUB_TAINTED_MODULES = ("ui.qt",)

#: Sentinel for "this module was not in sys.modules when we looked".
_ABSENT = object()

#: Marks a module object as one of ours. :func:`install_stub_modules` stamps
#: everything it installs, so "is this entry a stub or the real module?" is a
#: direct question rather than a guess about ``__file__``.
_STUB_MARKER = "_castsim_stub_module"

#: Real modules displaced by a stub install, keyed by module name.
#:
#: This exists because :func:`snapshot_modules` runs too early to be trusted on
#: its own. ``unittest`` imports every named test module before running any of
#: them, so a snapshot taken at import time can record ``PySide6`` as absent
#: and then a *later* module -- ``tests.test_pour_picker`` -- imports it for
#: real before the first test runs. Restoring from that stale record pops a
#: live module out of ``sys.modules``.
#:
#: For a pure-Python module that is merely wasteful. For a C extension it is
#: fatal: the next import re-initialises it, shiboken reports "Type conversion
#: already registered from type PySide::PyObjectWrapper to type int", and the
#: process segfaults the next time a widget is constructed -- in a traceback
#: that points at innocent application code (``ui/dialogs/base.py``), not here.
#:
#: What is displaced at install time is always accurate, so record that and
#: prefer it over the snapshot.
_DISPLACED: Dict[str, object] = {}


def make_qt_stub_modules(
    dummy: type,
    signal_factory: Callable[..., object],
) -> Dict[str, object]:
    """Build stub Qt modules that answer any attribute with ``dummy``.

    ``ui/qt.py`` imports 53 names across the three Qt submodules, and the list
    grows whenever a dialog needs a new widget. Rather than enumerate them --
    which would make every such addition break these tests -- each stub
    submodule gets a module-level ``__getattr__`` (PEP 562) that hands back
    ``dummy`` for anything asked of it. This mirrors how the ``vtk`` stub in
    each test module already works.

    ``Signal`` is set as a real attribute so it shadows ``__getattr__``:
    callers need a factory that returns a signal-shaped object, not the dummy
    class itself.

    Dunder lookups deliberately raise ``AttributeError`` instead of returning
    ``dummy``. Claiming to have ``__path__`` would invite the import machinery
    to treat these plain modules as packages.
    """

    def _any_attr(attr: str, _dummy: type = dummy) -> type:
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _dummy

    package = types.ModuleType(QT_BINDING)
    modules: Dict[str, object] = {QT_BINDING: package}

    for sub in QT_SUBMODULES:
        name = f"{QT_BINDING}.{sub}"
        module = types.ModuleType(name)
        module.__getattr__ = _any_attr  # type: ignore[attr-defined]
        setattr(package, sub, module)
        modules[name] = module

    modules[f"{QT_BINDING}.QtCore"].Signal = signal_factory  # type: ignore[attr-defined]
    return modules


def make_vtk_qt_stub_modules(dummy: type) -> Dict[str, object]:
    """Build stub ``vtkmodules.qt`` modules exposing the render-window widget.

    Identical in every test that stubs the visualizer, so it lives here rather
    than being repeated five times.
    """
    interactor = types.ModuleType("vtkmodules.qt.QVTKRenderWindowInteractor")
    interactor.QVTKRenderWindowInteractor = dummy  # type: ignore[attr-defined]

    qt_package = types.ModuleType("vtkmodules.qt")
    qt_package.QVTKRenderWindowInteractor = interactor  # type: ignore[attr-defined]

    return {
        "vtkmodules.qt": qt_package,
        "vtkmodules.qt.QVTKRenderWindowInteractor": interactor,
    }


def install_stub_modules(modules: Dict[str, object]) -> None:
    """Force ``modules`` into ``sys.modules``, replacing anything already there.

    Use this rather than ``sys.modules.setdefault``. ``setdefault`` silently
    does nothing when the real module is already imported, which makes the test
    depend on whether some earlier test happened to import it -- the module
    then runs against real Qt/VTK in one ordering and against stubs in another.

    Always pair with :func:`restore_modules` in ``tearDownModule``.
    """
    for name, module in modules.items():
        current = sys.modules.get(name, _ABSENT)
        if current is not _ABSENT and not getattr(current, _STUB_MARKER, False):
            # A real module is being displaced right now. Remember it: this is
            # more accurate than any snapshot taken at import time.
            _DISPLACED[name] = current
        setattr(module, _STUB_MARKER, True)
        sys.modules[name] = module
    # Drop ui.visualizer and ui.qt so the caller's import binds to the stubs
    # just installed rather than to whatever an earlier test cached.
    #
    # ui.main_window is deliberately NOT dropped: several tests import it at
    # module scope and then patch its globals (e.g.
    # patch("ui.main_window.GPUComputeManager")). Re-importing it would leave
    # those references pointing at a stale module object, so the patches would
    # target a different module than the classes under test.
    for name in VISUALIZER_MODULES:
        sys.modules.pop(name, None)


def snapshot_modules(
    names: Iterable[str] = VISUALIZER_STUB_MODULES + STUB_TAINTED_MODULES,
) -> Dict[str, object]:
    """Record the current ``sys.modules`` entry for each name in ``names``.

    Call this at module import time, before any stub is installed, so the
    snapshot captures the real modules (or their absence).
    """
    return {name: sys.modules.get(name, _ABSENT) for name in names}


def restore_modules(snapshot: Dict[str, object]) -> None:
    """Put ``sys.modules`` back the way ``snapshot`` found it.

    Names that held a real module are rebound to it -- including names the
    snapshot thought were absent but which turned out to be occupied when the
    stub went in (see :data:`_DISPLACED`). Only entries that are still our own
    stubs are removed, so a real C-extension module is never left popped for a
    later import to re-initialise. Safe to call more than once.

    ``ui.visualizer`` is intentionally left as the stub-bound module it already
    is. Later tests only call pure helpers on it (``should_use_depth_peeling``,
    ``clamp_window_position``), and one looks it up through ``sys.modules``
    directly, so removing the entry would break it.

    ``ui.qt`` is the opposite case and is covered by the default snapshot (see
    :data:`STUB_TAINTED_MODULES`): other widget modules import their base
    classes from it, so it has to be real again before anything else builds on
    it. What also matters is that ``vtk`` and the Qt binding themselves are
    real again for code that imports them at call time, such as
    ``core.surface_extraction._extract_with_vtk``.
    """
    for name, original in snapshot.items():
        displaced = _DISPLACED.pop(name, _ABSENT)
        if displaced is not _ABSENT:
            # A real module was in place when the stub went in; put that back
            # regardless of what the snapshot recorded.
            sys.modules[name] = displaced
        elif original is _ABSENT:
            # Nothing real was ever imported under this name, so the entry can
            # only be our stub. Drop it -- but check, so a module imported for
            # real between install and restore is never popped.
            if getattr(sys.modules.get(name), _STUB_MARKER, False):
                sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
