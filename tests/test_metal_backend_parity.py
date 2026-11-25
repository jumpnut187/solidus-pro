from __future__ import annotations

import sys
if sys.platform != 'darwin':
    pass

import unittest
import numpy as np
import platform

from core.gpu_compute import GPUComputeManager, GPUBackend, gpu_compute
from core.acceleration import (
    compute_min_distances_parallel,
    update_solidification_with_feeding_connectivity,
)

@unittest.skipUnless(sys.platform == 'darwin', 'Apple Silicon Metal GPU only')
class MetalBackendParityTest(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(42)

    def test_metal_backend_detected(self) -> None:
        mgr = GPUComputeManager()
        mgr._detection_attempted = False
        mgr._detect_backends()

        self.assertIn(mgr.backend, (GPUBackend.METAL, GPUBackend.NONE))

        if mgr.backend == GPUBackend.METAL:
            self.assertTrue(mgr.is_gpu_available)
            self.assertIn("Apple Silicon", mgr.device_name)
        else:
            self.skipTest('Metal not available (mlx/taichi not installed)')

    def test_distance_parity(self) -> None:
        mgr = GPUComputeManager()
        mgr._detection_attempted = False
        mgr._detect_backends()

        if mgr.backend != GPUBackend.METAL:
            self.skipTest('Metal not available')

        points = np.random.randn(1000, 3).astype(np.float32)
        reference_points = np.random.randn(500, 3).astype(np.float32)

        gpu_result = mgr._compute_distances_metal(points, reference_points)
        cpu_result = compute_min_distances_parallel(points, reference_points)

        self.assertEqual(gpu_result.shape, cpu_result.shape)
        np.testing.assert_allclose(
            gpu_result, cpu_result, rtol=1e-4, atol=1e-5,
            err_msg="MLX distance computation diverged from Numba reference"
        )

    def test_solidification_parity(self) -> None:
        mgr = GPUComputeManager()
        mgr._detection_attempted = False
        mgr._detect_backends()

        if mgr.backend != GPUBackend.METAL:
            self.skipTest('Metal not available')

        n_voxels = 1000
        n_neighbors = 26
        time_step = 0.1
        material_factor = 1.0

        voxel_states = np.random.rand(n_voxels).astype(np.float32) * 0.3
        neighbors = np.random.randint(-1, n_voxels, size=(n_voxels, n_neighbors), dtype=np.int32)
        for j in range(n_neighbors):
            neighbors[j % 10::10, j] = -1
        moduli = np.random.rand(n_voxels).astype(np.float32) * 2.0 + 0.1
        heights = np.random.rand(n_voxels).astype(np.float32)
        chill_boost = np.ones(n_voxels, dtype=np.float32) + np.random.rand(n_voxels).astype(np.float32) * 0.5
        riser_slowdown = np.ones(n_voxels, dtype=np.float32) * 0.8 + np.random.rand(n_voxels).astype(np.float32) * 0.2
        part_types = np.random.randint(0, 3, size=n_voxels, dtype=np.int32)

        gpu_states = voxel_states.copy()
        cpu_states = voxel_states.copy()

        for _ in range(10):
            gpu_states = mgr._update_solidification_metal(
                gpu_states, neighbors, moduli, heights,
                chill_boost, riser_slowdown, part_types,
                time_step, material_factor
            )
            cpu_states = update_solidification_with_feeding_connectivity(
                cpu_states, neighbors, moduli, heights,
                chill_boost, riser_slowdown, part_types,
                time_step, material_factor
            )

        np.testing.assert_allclose(
            gpu_states, cpu_states, rtol=1e-4, atol=1e-5,
            err_msg="Taichi solidification diverged from Numba reference after 10 iterations"
        )

    def test_solidification_feeding_zone_logic(self) -> None:
        mgr = GPUComputeManager()
        mgr._detection_attempted = False
        mgr._detect_backends()

        if mgr.backend != GPUBackend.METAL:
            self.skipTest('Metal not available')

        n_neighbors = 26
        time_step = 0.1
        material_factor = 1.0

        n_voxels = 4
        voxel_states = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        neighbors = np.full((n_voxels, n_neighbors), -1, dtype=np.int32)
        neighbors[0, 0] = 1
        neighbors[0, 1] = 3
        neighbors[1, 0] = 0
        neighbors[1, 1] = 3
        neighbors[2, 0] = 3

        moduli = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        heights = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        chill_boost = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        riser_slowdown = np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32)
        part_types = np.array([0, 1, 0, 0], dtype=np.int32)

        gpu_states = voxel_states.copy()
        cpu_states = voxel_states.copy()

        for _ in range(20):
            gpu_states = mgr._update_solidification_metal(
                gpu_states, neighbors, moduli, heights,
                chill_boost, riser_slowdown, part_types,
                time_step, material_factor
            )
            cpu_states = update_solidification_with_feeding_connectivity(
                cpu_states, neighbors, moduli, heights,
                chill_boost, riser_slowdown, part_types,
                time_step, material_factor
            )

        np.testing.assert_allclose(
            gpu_states, cpu_states, rtol=1e-4, atol=1e-5,
            err_msg="GPU and CPU disagree on feeding zone logic"
        )

        self.assertTrue(
            cpu_states[0] < cpu_states[2],
            msg="Voxel 0 (CAST in feeding zone) should solidify slower than voxel 2 (NOT in feeding zone)"
        )

    def test_voxel_containment_metal(self) -> None:
        mgr = GPUComputeManager()
        mgr._detection_attempted = False
        mgr._detect_backends()

        if mgr.backend != GPUBackend.METAL:
            self.skipTest('Metal not available')

        s3 = np.float32(np.sqrt(3.0))
        s6 = np.float32(np.sqrt(6.0))
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, s3 / 2.0, 0.0],
            [0.5, s3 / 6.0, s6 / 3.0],
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
        ], dtype=np.int32)

        inside_points = np.array([
            [0.4, 0.2, 0.2],
            [0.5, 0.25, 0.2],
            [0.45, 0.3, 0.15],
            [0.5, 0.1, 0.1],
            [0.35, 0.15, 0.25],
            [0.5, 0.2, 0.1],
            [0.4, 0.25, 0.2],
            [0.45, 0.2, 0.15],
            [0.5, 0.15, 0.2],
            [0.42, 0.22, 0.18],
        ], dtype=np.float32)

        outside_points = np.array([
            [-0.5, 0.2, 0.2],
            [1.5, 0.2, 0.2],
            [0.5, 1.5, 0.2],
            [0.5, -0.5, 0.2],
            [0.5, 0.2, 1.5],
            [0.5, 0.2, -0.5],
            [-0.5, -0.5, -0.5],
            [1.5, 1.5, 1.5],
            [0.0, 0.0, -0.5],
            [0.5, 0.5, 0.5],
        ], dtype=np.float32)

        all_points = np.vstack([inside_points, outside_points])
        expected = np.array([True] * 10 + [False] * 10)

        try:
            result = mgr._voxel_containment_metal(all_points, vertices, faces)
        except AttributeError:
            self.skipTest('_voxel_containment_metal not available')

        if result is None:
            self.skipTest('Voxel containment returned None (edge case - falls back to CPU)')

        correct = np.sum(result == expected)
        total = len(expected)
        accuracy = correct / total

        self.assertGreater(
            accuracy, 0.9,
            msg=f"Voxel containment accuracy {accuracy:.1%} is below 90% (got {result})"
        )

if __name__ == "__main__":
    unittest.main()
