"""
GPU Compute Package Export Tests

Verifies that the GPU compute acceleration layer is importable and correctly
re-exported at the package level (``core``), including the ``get_gpu_compute``
accessor that ``core/__init__.py`` depends on.

These tests assert that the module and its exports are *available*, not that a
physical GPU backend exists on the machine. They are safe to run on CPU-only
machines.
"""

import unittest

import core
from core import get_gpu_compute
from core.gpu_compute import (
    GPUBackend,
    GPUComputeManager,
    compute_distances_accelerated,
    get_gpu_compute as module_get_gpu_compute,
    gpu_compute,
    update_solidification_accelerated,
)


class GpuComputeExportTests(unittest.TestCase):
    """Package-level import/export of the GPU compute layer."""

    def test_package_import_sets_gpu_available(self) -> None:
        """core.GPU_AVAILABLE must be True once the module imports cleanly."""
        # The flag is flipped to True by core/__init__.py only when every
        # symbol in the gpu_compute import block resolves, including the
        # get_gpu_compute accessor.
        self.assertTrue(core.GPU_AVAILABLE)

    def test_get_gpu_compute_re_exports_singleton(self) -> None:
        """Package-level get_gpu_compute returns the module-global manager."""
        mgr = get_gpu_compute()
        self.assertIsInstance(mgr, GPUComputeManager)
        self.assertIs(mgr, gpu_compute)

    def test_module_get_gpu_compute_matches_package_export(self) -> None:
        """Module-level accessor is the same callable as the package export."""
        self.assertIs(module_get_gpu_compute, get_gpu_compute)
        self.assertIs(module_get_gpu_compute(), gpu_compute)

    def test_public_exports_importable(self) -> None:
        """All core.gpu_compute public symbols resolve at package import."""
        self.assertIs(core.gpu_compute, gpu_compute)
        self.assertIsInstance(core.GPUComputeManager, type)
        self.assertIs(core.GPUBackend, GPUBackend)
        self.assertIs(core.compute_distances_accelerated, compute_distances_accelerated)
        self.assertIs(
            core.update_solidification_accelerated,
            update_solidification_accelerated,
        )

    def test_manager_usable_without_physical_gpu(self) -> None:
        """The manager works regardless of whether a GPU backend exists."""
        mgr = get_gpu_compute()
        # Detection may legitimately find no backend (CPU-only machine). The
        # contract under test is that the accessor always returns a usable
        # manager and the availability probe is a plain boolean — never an
        # error and never a hard requirement of a physical GPU.
        self.assertIn(mgr.backend, tuple(GPUBackend))
        self.assertIsInstance(mgr.is_gpu_available, bool)


if __name__ == "__main__":
    unittest.main()
