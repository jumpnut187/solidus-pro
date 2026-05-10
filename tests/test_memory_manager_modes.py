import unittest

import numpy as np

from core.memory_manager import create_managed_array, get_memory_manager, reset_memory_manager


class MemoryManagerModesTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_memory_manager()

    def tearDown(self) -> None:
        reset_memory_manager()

    def test_disk_mode_is_preserved_for_default_allocation_path(self) -> None:
        manager = get_memory_manager("disk")
        arr = create_managed_array((1024,), dtype=np.float32, name="disk_probe")

        self.assertEqual(manager.memory_mode, "disk")
        self.assertIsInstance(arr, np.memmap)
        self.assertGreater(manager.get_memory_stats()["disk_mb"], 0.0)

    def test_ram_mode_is_preserved_for_default_allocation_path(self) -> None:
        manager = get_memory_manager("ram")
        arr = create_managed_array((1024,), dtype=np.float32, name="ram_probe")

        self.assertEqual(manager.memory_mode, "ram")
        self.assertNotIsInstance(arr, np.memmap)
        self.assertGreater(manager.get_memory_stats()["ram_mb"], 0.0)

    def test_get_memory_manager_without_mode_does_not_reset_explicit_mode(self) -> None:
        manager = get_memory_manager("disk")
        same_manager = get_memory_manager()

        self.assertIs(manager, same_manager)
        self.assertEqual(same_manager.memory_mode, "disk")

    def test_disk_backed_arrays_are_zero_initialized(self) -> None:
        get_memory_manager("disk")
        arr = create_managed_array((64,), dtype=np.float32, name="disk_zero_init")

        self.assertIsInstance(arr, np.memmap)
        self.assertTrue(np.all(arr == 0.0))


if __name__ == "__main__":
    unittest.main()
