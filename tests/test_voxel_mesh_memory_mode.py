import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import trimesh

from core.memory_manager import reset_memory_manager
from core.voxel_mesh import AdaptiveVoxelMesh, RTREE_AVAILABLE


class VoxelMeshMemoryModeTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_memory_manager()

    def tearDown(self) -> None:
        reset_memory_manager()

    def _build_minimal_mesh(self, memory_mode: str) -> AdaptiveVoxelMesh:
        mesh = AdaptiveVoxelMesh(base_voxel_size=1.0, memory_mode=memory_mode)
        centers = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        cube = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
        mesh._calculate_voxel_properties(centers, cube)
        return mesh

    def test_disk_mode_uses_memmap_for_core_mesh_arrays(self) -> None:
        mesh = self._build_minimal_mesh("disk")
        try:
            self.assertIsInstance(mesh.voxel_centers, np.memmap)
            self.assertIsInstance(mesh.voxel_sizes, np.memmap)
            self.assertIsInstance(mesh.voxel_volumes, np.memmap)
        finally:
            mesh.cleanup()

    def test_ram_mode_uses_ndarray_for_core_mesh_arrays(self) -> None:
        mesh = self._build_minimal_mesh("ram")
        try:
            self.assertNotIsInstance(mesh.voxel_centers, np.memmap)
            self.assertNotIsInstance(mesh.voxel_sizes, np.memmap)
            self.assertNotIsInstance(mesh.voxel_volumes, np.memmap)
        finally:
            mesh.cleanup()


class ChunkedVoxelizationTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_memory_manager()

    def tearDown(self) -> None:
        reset_memory_manager()

    def test_chunked_voxelization_fails_clearly_without_rtree(self) -> None:
        cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        mesh = AdaptiveVoxelMesh(base_voxel_size=0.5, memory_mode="ram")
        mesh.memory_manager = MagicMock()
        mesh.memory_manager.get_available_memory_gb.return_value = 8.0
        try:
            mesh.bounds = np.array([[-1.0, -1.0, -1.0], [3.0, 3.0, 3.0]])
            mesh.base_voxel_size = 0.5
            mesh.grid_shape = (8, 8, 8)
            mesh.use_chunked_voxelization = True

            with patch("core.voxel_mesh.RTREE_AVAILABLE", False):
                with self.assertRaises(RuntimeError) as ctx:
                    mesh._create_uniform_grid_chunked(
                        cube, -1.0, 3.0, -1.0, 3.0, -1.0, 3.0, 8, 8, 8
                    )
                self.assertIn("rtree", str(ctx.exception).lower())
        finally:
            mesh.cleanup()

    @unittest.skipUnless(RTREE_AVAILABLE, "rtree not installed")
    def test_chunked_voxelization_of_box_returns_nonzero_voxels(self) -> None:
        cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        mesh = AdaptiveVoxelMesh(base_voxel_size=0.25, memory_mode="ram")
        mesh.memory_manager = MagicMock()
        mesh.memory_manager.get_available_memory_gb.return_value = 8.0
        try:
            mesh.bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
            mesh.base_voxel_size = 0.25
            mesh.grid_shape = (8, 8, 8)
            mesh.use_chunked_voxelization = True

            voxels = mesh._create_uniform_grid_chunked(
                cube, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 8, 8, 8
            )
            self.assertGreater(len(voxels), 0)
        finally:
            mesh.cleanup()


if __name__ == "__main__":
    unittest.main()
