"""
GPU Distance Chunk Size Setting Tests

Verifies that the GPU distance-compute chunk size is configurable via the
performance settings, overridable by the CASTSIM_GPU_DISTANCE_CHUNK_SIZE
environment variable, persisted through the user config file, and safely
clamped by the GPU compute manager.
"""

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from config.settings import Settings
from core.gpu_compute import GPUComputeManager


class GpuDistanceChunkDefaultTests(unittest.TestCase):
    """Default value of the chunk-size setting."""

    def test_default_is_10000(self) -> None:
        self.assertEqual(Settings().performance.gpu_distance_chunk_size, 10000)


class GpuDistanceChunkEnvOverrideTests(unittest.TestCase):
    """Environment variable override of the chunk-size setting."""

    ENV_VAR = "CASTSIM_GPU_DISTANCE_CHUNK_SIZE"

    def setUp(self) -> None:
        self._old = os.environ.pop(self.ENV_VAR, None)

    def tearDown(self) -> None:
        if self._old is not None:
            os.environ[self.ENV_VAR] = self._old
        else:
            os.environ.pop(self.ENV_VAR, None)

    def test_env_var_overrides_default(self) -> None:
        os.environ[self.ENV_VAR] = "2000"
        settings = Settings()
        settings.reload()
        self.assertEqual(settings.performance.gpu_distance_chunk_size, 2000)


class GpuDistanceChunkPersistenceTests(unittest.TestCase):
    """User config persistence of the chunk-size setting."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._config_dir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _make_settings(self) -> Settings:
        settings = Settings()
        settings.paths.user_config_dir = str(self._config_dir)
        return settings

    def test_save_user_override_persists_and_preserves_other_keys(self) -> None:
        settings = self._make_settings()
        config_path = self._config_dir / "config.yaml"
        config_path.write_text(
            "performance:\n"
            "  max_workers: 4\n"
            "mesh:\n"
            "  memory_mode: disk\n"
        )

        settings.save_user_override("performance", "gpu_distance_chunk_size", 2000)

        # Reload a fresh Settings instance pointed at the same config dir.
        loaded = self._make_settings()
        loaded.reload()
        self.assertEqual(loaded.performance.gpu_distance_chunk_size, 2000)
        # Unrelated keys survive the merge write.
        self.assertEqual(loaded.performance.max_workers, 4)
        self.assertEqual(loaded.mesh.memory_mode, "disk")

    def test_save_user_override_creates_config_when_missing(self) -> None:
        settings = self._make_settings()
        settings.save_user_override("performance", "gpu_distance_chunk_size", 500)
        self.assertTrue((self._config_dir / "config.yaml").exists())

        loaded = self._make_settings()
        loaded.reload()
        self.assertEqual(loaded.performance.gpu_distance_chunk_size, 500)


class GpuDistanceChunkManagerTests(unittest.TestCase):
    """GPU compute manager chunk-size resolution and clamping."""

    def test_get_distance_chunk_size_reads_setting(self) -> None:
        fake = types.SimpleNamespace(
            performance=types.SimpleNamespace(gpu_distance_chunk_size=2000)
        )
        mgr = GPUComputeManager()
        with mock.patch("config.settings.get_settings", return_value=fake):
            self.assertEqual(mgr._get_distance_chunk_size(), 2000)

    def test_get_distance_chunk_size_clamps_to_minimum(self) -> None:
        for bad in (0, -5):
            fake = types.SimpleNamespace(
                performance=types.SimpleNamespace(gpu_distance_chunk_size=bad)
            )
            mgr = GPUComputeManager()
            with mock.patch("config.settings.get_settings", return_value=fake):
                self.assertEqual(mgr._get_distance_chunk_size(), 1)

    def test_get_distance_chunk_size_falls_back_on_error(self) -> None:
        mgr = GPUComputeManager()

        def boom() -> None:
            raise RuntimeError("settings unavailable")

        with mock.patch("config.settings.get_settings", side_effect=boom):
            self.assertEqual(mgr._get_distance_chunk_size(), 10000)


if __name__ == "__main__":
    unittest.main()
