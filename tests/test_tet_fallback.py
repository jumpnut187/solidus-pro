"""Automatic tet -> voxel fallback (docs/tet_migration/voxel_default_fallback.md).

A tet run falls back to the voxel mesh, and records why, when the estimated
node count is above ``settings.mesh.tet_auto_max_nodes``, when every fTetWild
attempt fails, or when fTetWild runs past ``tet_mesh_timeout_s``. With
``tet_auto_fallback`` off the errors raise; cancel during tet meshing cancels.
Also covers the new ``MeshSettings`` fields and the desktop notice.
"""

from __future__ import annotations

import os
import subprocess
import time
import tomllib
import unittest
from typing import Any
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import yaml  # noqa: E402

from config.exceptions import MeshGenerationError  # noqa: E402
from config.settings import MeshSettings, Settings, TOML_TEMPLATE, YAML_TEMPLATE  # noqa: E402
from core import simulation_pipeline as sp  # noqa: E402
from core import tet_mesh as tet_mesh_module  # noqa: E402
from core import tetwild_child  # noqa: E402
from core.part_manager import PART_TYPE_CAST, PART_TYPE_CHILL  # noqa: E402
from core.tet_mesh import TetMeshTimeoutError, TetVolumeMesh, build_tet_mesh  # noqa: E402
from core.voxel_mesh import AdaptiveVoxelMesh  # noqa: E402


def _box(extents, center) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


class _PartManager:
    """The ``PartManager`` members ``prepare_mesh_and_geometry`` reads."""

    def __init__(self, parts) -> None:
        self._parts = list(parts)

    def get_part_count(self) -> int:
        return len(self._parts)

    def get_combined_mesh(self) -> trimesh.Trimesh:
        return trimesh.util.concatenate([m for m, _ in self._parts])

    def get_combined_bounds(self) -> np.ndarray:
        return np.asarray(self.get_combined_mesh().bounds).T

    def get_total_volume(self) -> float:
        return float(sum(m.volume for m, _ in self._parts))

    def get_parts_with_types(self):
        return list(self._parts)


#: Cast on a chill: an assembly, so the CSG attempt runs before the soup ones.
ASSEMBLY = [
    (_box((2.0, 1.0, 1.0), (0.0, 0.0, 0.5)), PART_TYPE_CAST),
    (_box((1.0, 1.0, 0.5), (0.0, 0.0, -0.25)), PART_TYPE_CHILL),
]
VOXEL_SIZE_IN = 0.25


def _settings(**mesh_fields: Any) -> Settings:
    settings = Settings()
    for name, value in mesh_fields.items():
        setattr(settings.mesh, name, value)
    settings.mesh.normalize_tet_fields()
    return settings


def _run(settings: Settings, *, parts=ASSEMBLY, voxel_size=VOXEL_SIZE_IN, cancel_callback=None, detailed=None):
    messages: list[str] = []
    inputs = sp.PipelineInputs("WCB", voxel_size, 0, mesh_type="tet", analysis_options={"shrinkage": True})
    with mock.patch.object(sp, "get_settings", return_value=settings), mock.patch.object(
        sp._error_logger, "warning"
    ) as warning:
        result = sp.run_pipeline(
            inputs,
            part_manager=_PartManager(parts),
            progress_message_callback=messages.append,
            detailed_progress_callback=detailed,
            cancel_callback=cancel_callback,
        )
    return result, messages, warning


class FallbackAssertions(unittest.TestCase):
    def assert_fell_back(self, result, messages, warning, *, reason_contains: str) -> str:
        self.assertEqual(result.mesh_type, "voxel")
        self.assertIsInstance(result.voxel_mesh, AdaptiveVoxelMesh)
        self.assertEqual(result.requested_mesh_type, "tet")
        reason = result.mesh_fallback_reason
        self.assertIsInstance(reason, str)
        self.assertIn(reason_contains, reason)
        # The run completed on the voxel mesh.
        n = result.voxel_mesh.get_voxel_count()
        self.assertGreater(n, 0)
        self.assertEqual(result.engine.solidification_times.shape, (n,))
        for payload in (result.result_data, result.visualization_data):
            self.assertEqual(payload["mesh_type"], "voxel")
            self.assertEqual(payload["requested_mesh_type"], "tet")
            self.assertEqual(payload["mesh_fallback_reason"], reason)
        self.assertIn(f"{sp.MESH_FALLBACK_MESSAGE_PREFIX}{reason}", messages)
        fallback_at = messages.index(f"{sp.MESH_FALLBACK_MESSAGE_PREFIX}{reason}")
        self.assertTrue(any(m.startswith("Generating adaptive voxel mesh") for m in messages[fallback_at:]))
        warning.assert_called_once()
        self.assertIn(reason, warning.call_args.args[0])
        return reason


class SizeGateFallbackTests(FallbackAssertions):
    def test_estimate_above_tet_auto_max_nodes_skips_tet(self) -> None:
        with mock.patch.object(sp, "build_tet_mesh") as built:
            result, messages, warning = _run(_settings(tet_auto_max_nodes=10))
        built.assert_not_called()
        reason = self.assert_fell_back(result, messages, warning, reason_contains="tet_auto_max_nodes")
        edge = VOXEL_SIZE_IN * 2.0
        self.assertIn(f"{sp.estimate_tet_node_count(ASSEMBLY, edge):,} nodes", reason)
        self.assertEqual(warning.call_args.kwargs["context"]["trigger"], "size")

    def test_voxel_fallback_uses_the_voxel_run_size(self) -> None:
        with mock.patch.object(sp, "AdaptiveVoxelMesh", wraps=AdaptiveVoxelMesh) as built:
            result, _messages, _warning = _run(_settings(tet_auto_max_nodes=10))
        self.assertEqual(built.call_args.kwargs["base_voxel_size"], VOXEL_SIZE_IN)
        self.assertEqual(result.voxel_mesh.base_voxel_size, VOXEL_SIZE_IN)

    def test_tet_max_nodes_guard_falls_back(self) -> None:
        settings = _settings(tet_max_nodes=10, tet_auto_max_nodes=10_000_000)
        with mock.patch.object(sp, "build_tet_mesh") as built:
            result, messages, warning = _run(settings)
        built.assert_not_called()
        self.assert_fell_back(result, messages, warning, reason_contains="tet_max_nodes")
        self.assertEqual(warning.call_args.kwargs["context"]["trigger"], "error")

    def test_surrogate_pipeline_falls_back_the_same_way(self) -> None:
        from core.surrogate import pipeline as surrogate_pipeline

        class _Model:
            manifest = mock.Mock(grid_resolution=8, accuracy={}, backend="numpy")

            def predict(self, channels, conditioning) -> np.ndarray:
                return np.ones((3, 8, 8, 8), dtype=np.float32)

        _Model.manifest.name = "stub"
        inputs = sp.PipelineInputs("WCB", VOXEL_SIZE_IN, 0, mesh_type="tet")
        messages: list[str] = []
        with mock.patch.object(sp, "get_settings", return_value=_settings(tet_auto_max_nodes=10)):
            result = surrogate_pipeline.run_surrogate_pipeline(
                inputs, model=_Model(), part_manager=_PartManager(ASSEMBLY), progress_message_callback=messages.append
            )
        self.assertEqual(result.mesh_type, "voxel")
        self.assertEqual(result.requested_mesh_type, "tet")
        self.assertIn("tet_auto_max_nodes", result.mesh_fallback_reason)
        self.assertEqual(result.result_data["mesh_fallback_reason"], result.mesh_fallback_reason)
        self.assertTrue(any(m.startswith(sp.MESH_FALLBACK_MESSAGE_PREFIX) for m in messages))


class MesherFailureFallbackTests(FallbackAssertions):
    def test_every_fTetWild_attempt_crashing_falls_back(self) -> None:
        with mock.patch.dict(os.environ, {tetwild_child.FAULT_ENV: "csg,soup"}), self.assertLogs(
            "core.tet_mesh", level="WARNING"
        ):
            result, messages, warning = _run(_settings())
        reason = self.assert_fell_back(result, messages, warning, reason_contains="SIGSEGV")
        self.assertIn("fTetWild", reason)
        self.assertEqual(warning.call_args.kwargs["context"]["trigger"], "error")

    def test_wall_time_limit_kills_the_mesher_and_falls_back(self) -> None:
        with self.assertLogs("core.tet_mesh", level="WARNING"):
            result, messages, warning = _run(_settings(tet_mesh_timeout_s=0.01))
        self.assert_fell_back(result, messages, warning, reason_contains="did not finish within 0.01 s")
        self.assertEqual(warning.call_args.kwargs["context"]["trigger"], "timeout")
        self.assertFalse(tet_mesh_module._NATIVE_MESH_LOCK.locked())


class AutoFallbackOffTests(unittest.TestCase):
    def test_crash_raises(self) -> None:
        with mock.patch.dict(os.environ, {tetwild_child.FAULT_ENV: "csg,soup"}), self.assertLogs(
            "core.tet_mesh", level="WARNING"
        ):
            with self.assertRaises(MeshGenerationError) as raised:
                _run(_settings(tet_auto_fallback=False))
        self.assertIn("SIGSEGV", str(raised.exception))

    def test_tet_max_nodes_guard_raises(self) -> None:
        with self.assertRaises(MeshGenerationError) as raised:
            _run(_settings(tet_auto_fallback=False, tet_max_nodes=10))
        self.assertIn("tet_max_nodes", str(raised.exception))

    def test_auto_node_limit_and_timeout_do_not_apply(self) -> None:
        settings = _settings(tet_auto_fallback=False, tet_auto_max_nodes=10, tet_mesh_timeout_s=0.01)
        with mock.patch.object(sp, "build_tet_mesh", wraps=build_tet_mesh) as built:
            result, messages, warning = _run(settings, voxel_size=0.3)
        self.assertIsNone(built.call_args.kwargs["timeout_s"])
        self.assertEqual(result.mesh_type, "tet")
        self.assertIsInstance(result.voxel_mesh, TetVolumeMesh)
        self.assertIsNone(result.mesh_fallback_reason)
        self.assertEqual(result.requested_mesh_type, "tet")
        self.assertNotIn("mesh_fallback_reason", result.result_data)
        self.assertFalse(any(m.startswith(sp.MESH_FALLBACK_MESSAGE_PREFIX) for m in messages))
        warning.assert_not_called()


class CancelDuringTetMeshingTests(unittest.TestCase):
    def test_cancel_raises_and_does_not_fall_back(self) -> None:
        sphere = trimesh.creation.icosphere(subdivisions=4, radius=2.0)
        meshing = {"started": False}
        processes: list = []
        real_popen = subprocess.Popen

        def popen(*args, **kwargs):
            processes.append(real_popen(*args, **kwargs))
            return processes[-1]

        def detailed(current: int, total: int, stage: str) -> None:
            meshing["started"] = True

        settings = _settings(tet_auto_max_nodes=50_000_000, tet_max_nodes=50_000_000)
        with mock.patch.object(tet_mesh_module.subprocess, "Popen", side_effect=popen), mock.patch.object(
            AdaptiveVoxelMesh, "generate"
        ) as voxel_generate:
            with self.assertRaises(sp.SimulationCancelledError):
                _run(
                    settings,
                    parts=[(sphere, PART_TYPE_CAST)],
                    voxel_size=0.02,
                    cancel_callback=lambda: meshing["started"],
                    detailed=detailed,
                )
        voxel_generate.assert_not_called()
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())
        self.assertFalse(tet_mesh_module._NATIVE_MESH_LOCK.locked())


class BuildTetMeshTimeoutTests(unittest.TestCase):
    def test_timeout_kills_the_child_without_retrying(self) -> None:
        sphere = trimesh.creation.icosphere(subdivisions=4, radius=2.0)
        processes: list = []
        real_popen = subprocess.Popen

        def popen(*args, **kwargs):
            processes.append(real_popen(*args, **kwargs))
            return processes[-1]

        started = time.monotonic()
        with mock.patch.object(tet_mesh_module.subprocess, "Popen", side_effect=popen), self.assertLogs(
            "core.tet_mesh", level="WARNING"
        ) as logs:
            with self.assertRaises(TetMeshTimeoutError) as raised:
                build_tet_mesh([(sphere, PART_TYPE_CAST)], 0.04, timeout_s=1.0)
        elapsed = time.monotonic() - started
        self.assertIsInstance(raised.exception, MeshGenerationError)
        self.assertEqual(raised.exception.timeout_s, 1.0)
        self.assertIn("did not finish within 1 s", str(raised.exception))
        self.assertIn("wall-time limit", "\n".join(logs.output))
        self.assertLess(elapsed, 1.0 + 3.0)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())
        self.assertFalse(tet_mesh_module._NATIVE_MESH_LOCK.locked())

    def test_invalid_timeout_raises(self) -> None:
        for bad in (0.0, -1.0, float("nan")):
            with self.subTest(timeout_s=bad), self.assertRaises(ValueError):
                build_tet_mesh(ASSEMBLY, 0.3, timeout_s=bad)


class FallbackSettingsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        mesh = MeshSettings()
        self.assertEqual(mesh.mesh_type, "voxel")
        self.assertEqual(mesh.tet_edge_length_factor, 2.0)
        self.assertIs(mesh.tet_auto_fallback, True)
        self.assertEqual(mesh.tet_mesh_timeout_s, 600.0)
        self.assertEqual(mesh.tet_auto_max_nodes, 300_000)
        self.assertIs(mesh.voxel_cut_cells, False)

    def test_environment_overrides(self) -> None:
        env = {
            "CASTSIM_TET_AUTO_FALLBACK": "false",
            "CASTSIM_TET_MESH_TIMEOUT_S": "90",
            "CASTSIM_TET_AUTO_MAX_NODES": "1234",
            "CASTSIM_VOXEL_CUT_CELLS": "1",
        }
        settings = Settings()
        with mock.patch.dict(os.environ, env):
            settings._apply_environment_variables()
        self.assertIs(settings.mesh.tet_auto_fallback, False)
        self.assertEqual(settings.mesh.tet_mesh_timeout_s, 90.0)
        self.assertEqual(settings.mesh.tet_auto_max_nodes, 1234)
        self.assertIs(settings.mesh.voxel_cut_cells, True)
        for name in env:
            self.assertIn(name, settings._env_vars_applied)

    def test_invalid_values_fall_back_to_defaults(self) -> None:
        for name, bad in (
            ("tet_mesh_timeout_s", 0.0),
            ("tet_mesh_timeout_s", "abc"),
            ("tet_auto_max_nodes", -5),
            ("tet_auto_fallback", "maybe"),
            ("voxel_cut_cells", 2),
        ):
            with self.subTest(name=name, value=bad):
                default = getattr(MeshSettings(), name)
                with self.assertWarns(RuntimeWarning):
                    mesh = MeshSettings(**{name: bad})
                self.assertEqual(getattr(mesh, name), default)

    def test_config_file_strings_are_read_as_booleans(self) -> None:
        settings = Settings()
        settings._apply_config_dict({"mesh": {"tet_auto_fallback": "no", "voxel_cut_cells": "yes"}})
        settings.mesh.normalize_tet_fields()
        self.assertIs(settings.mesh.tet_auto_fallback, False)
        self.assertIs(settings.mesh.voxel_cut_cells, True)

    def test_templates_carry_the_new_fields_at_their_defaults(self) -> None:
        defaults = MeshSettings()
        names = ("mesh_type", "tet_auto_fallback", "tet_mesh_timeout_s", "tet_auto_max_nodes", "voxel_cut_cells")
        for label, data in (("yaml", yaml.safe_load(YAML_TEMPLATE)), ("toml", tomllib.loads(TOML_TEMPLATE))):
            for name in names:
                with self.subTest(template=label, field=name):
                    self.assertEqual(data["mesh"][name], getattr(defaults, name))


class DesktopFallbackNoticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_worker_records_the_reason_and_reports_voxel_progress(self) -> None:
        from ui.workers import SimulationWorker

        worker = SimulationWorker(mock.MagicMock(), "WCB", 0.2, 3, performance_options={"mesh_type": "tet"})
        emitted: list[str] = []
        worker.progress.connect(emitted.append)
        self.assertEqual(worker.run_mesh_type(), "tet")
        worker._on_pipeline_message("Generating tetrahedral mesh")
        self.assertIsNone(worker.mesh_fallback_reason)
        worker._on_pipeline_message(f"{sp.MESH_FALLBACK_MESSAGE_PREFIX}Too big.")
        self.assertEqual(worker.mesh_fallback_reason, "Too big.")
        self.assertEqual(worker.run_mesh_type(), "voxel")
        self.assertEqual(emitted[-1], f"{sp.MESH_FALLBACK_MESSAGE_PREFIX}Too big.")

    def test_status_keeps_the_reason_and_the_notice_shows_after_the_run(self) -> None:
        from ui import main_window as mw

        window = mw.MainWindow.__new__(mw.MainWindow)
        window.status_label = mock.MagicMock()
        worker = mock.MagicMock(mesh_fallback_reason="Too big.")
        with mock.patch.object(window, "sender", return_value=worker):
            window._on_simulation_progress("Running solidification simulation...")
        text = window.status_label.setText.call_args.args[0]
        self.assertIn("Running solidification simulation...", text)
        self.assertIn("Too big.", text)

        with mock.patch.object(mw, "QMessageBox") as box:
            notice = window._show_mesh_fallback_notice(worker)
        box.information.assert_called_once()
        title, body = box.information.call_args.args[1:3]
        self.assertEqual(title, "Ran on the Voxel Mesh")
        self.assertIn("Too big.", body)
        self.assertTrue(notice["status_line"])

        with mock.patch.object(mw, "QMessageBox") as box:
            notice = window._show_mesh_fallback_notice(mock.MagicMock(mesh_fallback_reason=None))
        box.information.assert_not_called()
        self.assertEqual(notice["status_line"], "")


if __name__ == "__main__":
    unittest.main()
