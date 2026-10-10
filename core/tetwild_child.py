"""Run one fTetWild call in a child process: ``python tetwild_child.py <workdir>``.

``core.tet_mesh`` starts this script so a native fTetWild crash (SIGBUS and
SIGSEGV were seen on large assemblies) kills the child, not the app. It
imports only stdlib, numpy and pytetwild, never ``core``.

``<workdir>/request.json`` holds ``mode`` and the fTetWild parameters:

- ``"soup"``: ``pytetwild.tetrahedralize`` on ``<workdir>/surface.npz``
  (``vertices`` float64, ``faces`` int32) with an absolute edge length.
  ``filter_outside`` false keeps fTetWild's whole background mesh.
- ``"csg"``: ``PyfTetWildWrapper.tetrahedralize_csg`` on the tree file named
  by ``csg_path`` with a relative edge length.

The result is written atomically to ``<workdir>/result.npz`` (``vertices``,
``tets``, and ``markers`` in CSG mode). Exit status 0 means the file exists.

``CASTSIM_TETWILD_CHILD_FAULT`` (comma-separated modes) makes the child
raise SIGSEGV on itself before the native call in those modes. Tests use it
to exercise the parent's crash handling.
"""

from __future__ import annotations

import json
import os
import signal
import sys

if __name__ == "__main__":
    # Running as a script puts core/ first on sys.path. Drop it so module
    # names in core/ never shadow a dependency.
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _SCRIPT_DIR]

import numpy as np  # noqa: E402

FAULT_ENV = "CASTSIM_TETWILD_CHILD_FAULT"
REQUEST_FILE = "request.json"
SURFACE_FILE = "surface.npz"
RESULT_FILE = "result.npz"


def _crash_if_requested(mode: str) -> None:
    modes = {m.strip() for m in os.environ.get(FAULT_ENV, "").split(",")}
    if mode in modes:
        if hasattr(signal, "SIGSEGV") and os.name != "nt":
            os.kill(os.getpid(), signal.SIGSEGV)
        os.abort()


def run(workdir: str) -> None:
    with open(os.path.join(workdir, REQUEST_FILE), encoding="utf-8") as handle:
        request = json.load(handle)
    mode = request["mode"]
    _crash_if_requested(mode)

    if mode == "soup":
        import pytetwild

        with np.load(os.path.join(workdir, SURFACE_FILE)) as surface:
            vertices_in = np.ascontiguousarray(surface["vertices"], dtype=np.float64)
            faces_in = np.ascontiguousarray(surface["faces"], dtype=np.int32)
        vertices, tets = pytetwild.tetrahedralize(
            vertices_in,
            faces_in,
            edge_length_abs=float(request["edge_length_abs"]),
            epsilon=float(request["epsilon_rel"]),
            stop_energy=float(request["stop_energy"]),
            coarsen=bool(request["coarsen"]),
            simplify=True,
            num_threads=int(request["num_threads"]),
            quiet=True,
            disable_filtering=not bool(request["filter_outside"]),
        )
        arrays = {"vertices": vertices, "tets": tets}
    elif mode == "csg":
        from pytetwild import PyfTetWildWrapper

        vertices, tets, markers = PyfTetWildWrapper.tetrahedralize_csg(
            request["csg_path"],
            float(request["epsilon_rel"]),
            float(request["edge_length_r"]),
            float(request["stop_energy"]),
            bool(request["coarsen"]),
            int(request["num_threads"]),
            6,
            False,
        )
        arrays = {"vertices": vertices, "tets": tets, "markers": markers}
    else:
        raise ValueError(f"unknown mode {mode!r}")

    partial = os.path.join(workdir, "result.partial.npz")
    np.savez(partial, **{k: np.asarray(v) for k, v in arrays.items()})
    os.replace(partial, os.path.join(workdir, RESULT_FILE))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("usage: tetwild_child.py <workdir>\n")
        sys.exit(2)
    run(sys.argv[1])
