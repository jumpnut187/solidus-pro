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

    def test_package_import_sets_gpu_available(self) -> None:
        self.assertTrue(core.GPU_AVAILABLE)

    def test_get_gpu_compute_re_exports_singleton(self) -> None:
        mgr = get_gpu_compute()
        self.assertIsInstance(mgr, GPUComputeManager)
        self.assertIs(mgr, gpu_compute)

    def test_module_get_gpu_compute_matches_package_export(self) -> None:
        self.assertIs(module_get_gpu_compute, get_gpu_compute)
        self.assertIs(module_get_gpu_compute(), gpu_compute)

    def test_public_exports_importable(self) -> None:
        self.assertIs(core.gpu_compute, gpu_compute)
        self.assertIsInstance(core.GPUComputeManager, type)
        self.assertIs(core.GPUBackend, GPUBackend)
        self.assertIs(core.compute_distances_accelerated, compute_distances_accelerated)
        self.assertIs(
            core.update_solidification_accelerated,
            update_solidification_accelerated,
        )

    def test_manager_usable_without_physical_gpu(self) -> None:
        mgr = get_gpu_compute()
        self.assertIn(mgr.backend, tuple(GPUBackend))
        self.assertIsInstance(mgr.is_gpu_available, bool)

if __name__ == "__main__":
    unittest.main()
