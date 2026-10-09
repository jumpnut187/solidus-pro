"""The Taichi Metal init must not leave flush-to-zero on the host thread.

``ti.init`` sets the FPCR flush-to-zero bit. With subnormals flushed, scipy's
cKDTree build recursed without end on a tet-mesh boundary point set and
crashed the process (SIGSEGV). ``core.gpu_compute`` restores the default FP
environment after init.
"""

from __future__ import annotations

import platform
import unittest

import numpy as np


class HostFloatingPointEnvironmentTests(unittest.TestCase):
    def test_subnormals_survive_gpu_backend_init(self) -> None:
        import core.gpu_compute  # noqa: F401  (initializes the GPU backend)

        smallest = np.float64(5e-324)
        self.assertGreater(smallest * np.float64(1.0), 0.0)

    @unittest.skipUnless(platform.system() == "Darwin", "macOS fesetenv path")
    def test_restore_helper_reports_success(self) -> None:
        from core.gpu_compute import _restore_default_fp_environment

        self.assertTrue(_restore_default_fp_environment())


if __name__ == "__main__":
    unittest.main()
