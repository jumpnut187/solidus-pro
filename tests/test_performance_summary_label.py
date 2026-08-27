"""
Tests for the GPU-aware performance summary label and the GPU toggle in
the Performance dialog (regression: sidebar used to hard-code "GPU: Off"
even when GPU was detected).
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch, MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QLabel

from ui.main_window import MainWindow, PerformanceDialog
from core.gpu_compute import GPUBackend


class _StubHost:
    """Minimal host object exposing only what _update_performance_summary touches."""

    def __init__(self, options: dict, label: QLabel) -> None:
        self._performance_options = options
        self.performance_summary_label = label

    def _update_performance_summary(self) -> None:
        pass


def _make_gpu_manager(available: bool, backend_name: str = "METAL",
                     device_name: str = "Apple Silicon GPU (Metal via MLX + Taichi)"):
    mgr = MagicMock()
    mgr.is_gpu_available = available
    mgr.backend = GPUBackend(backend_name.lower()) if available else GPUBackend.NONE
    mgr.device_name = device_name
    return mgr


#: ``gpu_distance_chunk_size`` value that suppresses the ", DistChunk N"
#: suffix in _update_performance_summary. These tests assert the GPU portion
#: of the label exactly, so the suffix is pinned off rather than hard-coded --
#: the setting is user/env-configurable and would otherwise make the expected
#: text machine-dependent.
_DIST_CHUNK_SUPPRESSED = 10000


class PerformanceSummaryLabelTests(unittest.TestCase):
    """Sidebar label must reflect live GPUComputeManager + use_gpu toggle."""

    def setUp(self) -> None:
        if QApplication.instance() is None:
            self._app = QApplication([])
        else:
            self._app = None
        self.label = QLabel()

        from config.settings import get_settings

        self._perf_settings = get_settings().performance
        self._original_chunk = self._perf_settings.gpu_distance_chunk_size
        self._perf_settings.gpu_distance_chunk_size = _DIST_CHUNK_SUPPRESSED

    def tearDown(self) -> None:
        self._perf_settings.gpu_distance_chunk_size = self._original_chunk
        if self._app is not None:
            self._app.quit()

    def test_label_says_metal_when_gpu_available_and_use_gpu_true(self) -> None:
        host = _StubHost({"use_gpu": True, "chunked_voxelization": False,
                          "incremental_solidification": False}, self.label)
        with patch("ui.main_window.GPUComputeManager",
                   return_value=_make_gpu_manager(True, "METAL")):
            MainWindow._update_performance_summary(host)

        self.assertEqual(host.performance_summary_label.text(), "GPU: METAL (Apple Silicon GPU)")
        self.assertIn("#4a4", host.performance_summary_label.styleSheet())

    def test_label_says_off_when_use_gpu_false_even_if_gpu_available(self) -> None:
        host = _StubHost({"use_gpu": False, "chunked_voxelization": False,
                          "incremental_solidification": False}, self.label)
        with patch("ui.main_window.GPUComputeManager",
                   return_value=_make_gpu_manager(True, "METAL")):
            MainWindow._update_performance_summary(host)

        self.assertEqual(host.performance_summary_label.text(), "GPU: Off")
        self.assertIn("#888", host.performance_summary_label.styleSheet())

    def test_label_says_off_when_gpu_not_available(self) -> None:
        host = _StubHost({"use_gpu": True, "chunked_voxelization": False,
                          "incremental_solidification": False}, self.label)
        with patch("ui.main_window.GPUComputeManager",
                   return_value=_make_gpu_manager(False)):
            MainWindow._update_performance_summary(host)

        self.assertEqual(host.performance_summary_label.text(), "GPU: Off")

    def test_label_includes_chunk_when_chunked_voxelization_enabled(self) -> None:
        host = _StubHost({"use_gpu": True, "chunked_voxelization": True,
                          "incremental_solidification": False}, self.label)
        with patch("ui.main_window.GPUComputeManager",
                   return_value=_make_gpu_manager(True, "METAL")):
            MainWindow._update_performance_summary(host)

        self.assertIn(", Chunk", host.performance_summary_label.text())

    def test_label_includes_incr_when_incremental_solidification_enabled(self) -> None:
        host = _StubHost({"use_gpu": True, "chunked_voxelization": False,
                          "incremental_solidification": True}, self.label)
        with patch("ui.main_window.GPUComputeManager",
                   return_value=_make_gpu_manager(True, "METAL")):
            MainWindow._update_performance_summary(host)

        self.assertIn(", Incr", host.performance_summary_label.text())

    def test_label_uses_raw_device_name_for_non_apple_silicon(self) -> None:
        host = _StubHost({"use_gpu": True, "chunked_voxelization": False,
                          "incremental_solidification": False}, self.label)
        with patch("ui.main_window.GPUComputeManager",
                   return_value=_make_gpu_manager(True, "CUDA",
                                                  "NVIDIA GeForce RTX 4090")):
            MainWindow._update_performance_summary(host)

        self.assertEqual(
            host.performance_summary_label.text(),
            "GPU: CUDA (NVIDIA GeForce RTX 4090)",
        )


class PerformanceDialogUseGpuRoundTripTests(unittest.TestCase):
    """Dialog get/set/store/restore must carry use_gpu."""

    def setUp(self) -> None:
        if QApplication.instance() is None:
            self._app = QApplication([])
        else:
            self._app = None
        with patch("ui.main_window.GPUComputeManager",
                   return_value=_make_gpu_manager(True, "METAL")):
            self.dialog = PerformanceDialog(parent=None)

    def tearDown(self) -> None:
        if self._app is not None:
            self._app.quit()

    def test_get_options_includes_use_gpu(self) -> None:
        self.dialog.use_gpu_check.setChecked(True)
        opts = self.dialog.get_options()
        self.assertIn("use_gpu", opts)
        self.assertTrue(opts["use_gpu"])

    def test_set_options_applies_use_gpu(self) -> None:
        self.dialog.set_options({"use_gpu": False, "chunked_voxelization": False,
                                 "incremental_solidification": False, "use_octree": True})
        self.assertFalse(self.dialog.use_gpu_check.isChecked())

    def test_store_then_restore_preserves_use_gpu(self) -> None:
        self.dialog.use_gpu_check.setChecked(False)
        self.dialog.store_original_values()
        self.dialog.use_gpu_check.setChecked(True)
        self.dialog.restore_original_values()
        self.assertFalse(self.dialog.use_gpu_check.isChecked())

    def test_use_gpu_check_is_disabled_when_no_gpu(self) -> None:
        self.dialog.close()
        self.dialog.deleteLater()
        with patch("ui.main_window.GPUComputeManager",
                   return_value=_make_gpu_manager(False)):
            dlg = PerformanceDialog(parent=None)
        try:
            self.assertFalse(dlg.use_gpu_check.isEnabled())
        finally:
            dlg.close()
            dlg.deleteLater()


class OnPerformanceOptionsConfirmedTests(unittest.TestCase):
    """_on_performance_options_confirmed must update internal dict and persist to settings."""

    def setUp(self) -> None:
        if QApplication.instance() is None:
            self._app = QApplication([])
        else:
            self._app = None

    def tearDown(self) -> None:
        if self._app is not None:
            self._app.quit()

    def test_confirm_persists_use_gpu_to_settings(self) -> None:
        from config import settings as settings_module

        host = _StubHost({"use_gpu": True, "chunked_voxelization": False,
                          "incremental_solidification": False}, QLabel())
        host.performance_summary_label.setText("stale")

        original = settings_module.settings.use_gpu
        try:
            with patch("ui.main_window.GPUComputeManager",
                       return_value=_make_gpu_manager(True, "METAL")):
                MainWindow._on_performance_options_confirmed(
                    host, {"use_gpu": False, "chunked_voxelization": False,
                           "incremental_solidification": False}
                )
            self.assertFalse(host._performance_options["use_gpu"])
            self.assertFalse(settings_module.settings.use_gpu)
        finally:
            settings_module.settings.use_gpu = original

    def test_confirm_does_not_clobber_unrelated_options(self) -> None:
        host = _StubHost({"use_gpu": True, "chunked_voxelization": True,
                          "incremental_solidification": True}, QLabel())
        with patch("ui.main_window.GPUComputeManager",
                   return_value=_make_gpu_manager(True, "METAL")):
            MainWindow._on_performance_options_confirmed(
                host, {"use_gpu": True, "chunked_voxelization": True,
                       "incremental_solidification": True}
            )
        self.assertTrue(host._performance_options["chunked_voxelization"])
        self.assertTrue(host._performance_options["incremental_solidification"])


class HardwareInfoStatusBarTests(unittest.TestCase):
    """_show_hardware_info must respect the use_gpu toggle."""

    def setUp(self) -> None:
        if QApplication.instance() is None:
            self._app = QApplication([])
        else:
            self._app = None

    def tearDown(self) -> None:
        if self._app is not None:
            self._app.quit()

    def test_status_bar_says_off_when_use_gpu_false(self) -> None:
        host = MagicMock()
        host._performance_options = {"use_gpu": False}
        with patch("ui.main_window.GPUComputeManager",
                   return_value=_make_gpu_manager(True, "METAL")), \
             patch("ui.main_window.hardware") as hw:
            hw.cpu_count = 8
            hw.performance_cores = 4
            hw.efficiency_cores = 4
            MainWindow._show_hardware_info(host)

        msg = host.statusBar().showMessage.call_args[0][0]
        self.assertIn("GPU: Off", msg)

    def test_status_bar_includes_backend_when_use_gpu_true(self) -> None:
        host = MagicMock()
        host._performance_options = {"use_gpu": True}
        with patch("ui.main_window.GPUComputeManager",
                   return_value=_make_gpu_manager(True, "METAL")), \
             patch("ui.main_window.hardware") as hw:
            hw.cpu_count = 8
            hw.performance_cores = 4
            hw.efficiency_cores = 4
            MainWindow._show_hardware_info(host)

        msg = host.statusBar().showMessage.call_args[0][0]
        self.assertIn("GPU: METAL", msg)


if __name__ == "__main__":
    unittest.main()
