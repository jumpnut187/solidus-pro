"""Snapshot/restore for tests that fake ``vtk`` and ``PyQt5`` in ``sys.modules``.

``ui/visualizer.py`` imports ``vtk``, ``vtk.util.numpy_support``,
``vtk.qt.QVTKRenderWindowInteractor`` and ``PyQt5`` at module scope, so tests
that exercise only its pure logic replace those modules with lightweight stubs
before importing it.

Those stubs must be taken back out once the installing module's tests finish.
The stub ``vtk`` answers every method call with ``None``, so leaving it in
``sys.modules`` silently breaks unrelated tests that run later in the same
process. The case that motivated this helper: ``core/surface_extraction.py``
does ``import vtk`` inside ``_extract_with_vtk``, gets the stub, and fails with
``'NoneType' object has no attribute 'SetScalars'``.

Such failures are order-dependent, so they do not reproduce when a module is run
on its own -- only under ``python -m unittest discover tests``.

Usage::

    from tests.helpers.module_stubs import restore_modules, snapshot_modules

    # At module import time, BEFORE any stub is installed.
    _MODULE_SNAPSHOT = snapshot_modules()

    def tearDownModule():
        restore_modules(_MODULE_SNAPSHOT)
"""

from __future__ import annotations

import sys
from typing import Dict, Iterable

#: Modules replaced by the visualizer test stubs.
VISUALIZER_STUB_MODULES = (
    "vtk",
    "vtk.util",
    "vtk.util.numpy_support",
    "vtk.qt",
    "vtk.qt.QVTKRenderWindowInteractor",
    "PyQt5",
    "PyQt5.QtWidgets",
    "PyQt5.QtCore",
)

#: Imported against the stubs by the visualizer tests, so it has to be dropped
#: before they import it or they get whatever an earlier test already cached.
VISUALIZER_MODULE = "ui.visualizer"

#: Sentinel for "this module was not in sys.modules when we looked".
_ABSENT = object()


def install_stub_modules(modules: Dict[str, object]) -> None:
    """Force ``modules`` into ``sys.modules``, replacing anything already there.

    Use this rather than ``sys.modules.setdefault``. ``setdefault`` silently
    does nothing when the real module is already imported, which makes the test
    depend on whether some earlier test happened to import it -- the module
    then runs against real Qt/VTK in one ordering and against stubs in another.

    Always pair with :func:`restore_modules` in ``tearDownModule``.
    """
    for name, module in modules.items():
        sys.modules[name] = module
    # Drop ui.visualizer so the caller's import binds to the stubs just
    # installed rather than to whatever an earlier test cached.
    #
    # ui.main_window is deliberately NOT dropped: several tests import it at
    # module scope and then patch its globals (e.g.
    # patch("ui.main_window.GPUComputeManager")). Re-importing it would leave
    # those references pointing at a stale module object, so the patches would
    # target a different module than the classes under test.
    sys.modules.pop(VISUALIZER_MODULE, None)


def snapshot_modules(
    names: Iterable[str] = VISUALIZER_STUB_MODULES,
) -> Dict[str, object]:
    """Record the current ``sys.modules`` entry for each name in ``names``.

    Call this at module import time, before any stub is installed, so the
    snapshot captures the real modules (or their absence).
    """
    return {name: sys.modules.get(name, _ABSENT) for name in names}


def restore_modules(snapshot: Dict[str, object]) -> None:
    """Put ``sys.modules`` back the way ``snapshot`` found it.

    Names that were absent are removed; names that held a real module are
    rebound to it. Safe to call more than once.

    ``ui.visualizer`` is intentionally left as the stub-bound module it already
    is. Later tests only call pure helpers on it (``should_use_depth_peeling``,
    ``clamp_window_position``), and one looks it up through ``sys.modules``
    directly, so removing the entry would break it. What matters is that
    ``vtk``/``PyQt5`` themselves are real again for code that imports them at
    call time, such as ``core.surface_extraction._extract_with_vtk``.
    """
    for name, original in snapshot.items():
        if original is _ABSENT:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
