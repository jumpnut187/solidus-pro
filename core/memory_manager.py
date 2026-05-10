"""
Memory Manager with Disk-Backed Storage
Automatically uses memory-mapped files for large arrays to prevent RAM exhaustion
Compatible with all NumPy/Numba operations
"""

import numpy as np
import os
import tempfile
import shutil
import psutil
from typing import Optional, Tuple
from pathlib import Path

# Import error logger
try:
    from core.error_logger import logger
except ImportError:
    class DummyLogger:
        def debug(self, msg, **kwargs): pass
        def info(self, msg, **kwargs): pass
        def warning(self, msg, **kwargs): pass
        def error(self, msg, **kwargs): pass
        def critical(self, msg, **kwargs): pass
        def exception(self, msg, **kwargs): pass
        def log_memory_warning(self, msg, **kwargs): pass
    logger = DummyLogger()


class MemoryManager:
    """
    Manages memory-mapped arrays that transparently use disk storage
    when RAM is limited.
    
    Features:
    - Automatic RAM vs Disk mode selection
    - Memory-mapped file cleanup on exit
    - Transparent NumPy/Numba compatibility
    - Thread-safe temporary directory management
    """
    
    def __init__(self, memory_mode: str = "auto", temp_dir: Optional[str] = None):
        """
        Args:
            memory_mode: "ram" (force RAM), "disk" (force disk), or "auto" (choose based on available memory)
            temp_dir: Custom temporary directory (default: system temp)
        """
        self.memory_mode = memory_mode
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.session_dir = None
        self.managed_arrays = []  # Track all created arrays for cleanup
        self.total_allocated_mb = 0.0
        
        # Create session-specific temp directory
        self._create_session_dir()
    
    def _create_session_dir(self):
        """Create unique temporary directory for this session"""
        self.session_dir = tempfile.mkdtemp(prefix="cast_sim_", dir=self.temp_dir)
        print(f"Memory Manager: Session directory: {self.session_dir}")
    
    def get_available_memory_gb(self) -> float:
        """Get available system memory in GB"""
        try:
            mem = psutil.virtual_memory()
            return mem.available / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            print(f"Warning: Could not get memory info: {e}")
            return 8.0  # Conservative fallback
    
    def get_total_memory_gb(self) -> float:
        """Get total system memory in GB"""
        try:
            mem = psutil.virtual_memory()
            return mem.total / (1024 ** 3)
        except Exception as e:
            print(f"Warning: Could not get memory info: {e}")
            return 16.0  # Conservative fallback
    
    def estimate_array_memory_mb(self, shape: Tuple, dtype) -> float:
        """Estimate memory usage for array in MB"""
        dtype_obj = np.dtype(dtype)
        total_elements = np.prod(shape)
        total_bytes = total_elements * dtype_obj.itemsize
        return total_bytes / (1024 ** 2)
    
    def should_use_disk(self, estimated_mb: float) -> bool:
        """
        Decide whether to use disk vs RAM based on available memory
        
        Args:
            estimated_mb: Estimated memory needed for this array
        
        Returns:
            True if should use disk, False for RAM
        """
        if self.memory_mode == "ram":
            return False
        elif self.memory_mode == "disk":
            return True
        else:  # auto mode
            available_gb = self.get_available_memory_gb()
            available_mb = available_gb * 1024
            
            # Use disk if:
            # 1. Array needs more than 20% of available RAM
            # 2. Less than 2 GB RAM available
            # 3. Already allocated arrays + new array > 50% available RAM
            
            threshold_mb = available_mb * 0.2
            total_needed = self.total_allocated_mb + estimated_mb
            
            if available_gb < 2.0:
                return True
            elif estimated_mb > threshold_mb:
                return True
            elif total_needed > available_mb * 0.5:
                return True
            else:
                return False
    
    def create_array(self, shape: Tuple, dtype=np.float32, name: str = "array") -> np.ndarray:
        """
        Create array (either in RAM or on disk based on memory availability)
        
        Args:
            shape: Array shape
            dtype: NumPy dtype
            name: Descriptive name for debugging
        
        Returns:
            NumPy array (either normal or memory-mapped)
        """
        estimated_mb = self.estimate_array_memory_mb(shape, dtype)
        use_disk = self.should_use_disk(estimated_mb)
        
        if use_disk:
            # Create memory-mapped file
            filename = f"{name}_{len(self.managed_arrays)}.dat"
            filepath = os.path.join(self.session_dir, filename)
            
            arr = np.memmap(
                filepath,
                dtype=dtype,
                mode='w+',  # Create and open for reading/writing
                shape=shape
            )
            arr[:] = 0
            
            try:
                logger.log_memory_warning(
                    current_mb=self.total_allocated_mb + estimated_mb,
                    limit_mb=self.get_available_memory_gb() * 1024,
                    voxel_count=int(np.prod(shape)),
                )
            except TypeError:
                # Backward-compatible fallback for logger stubs that accept a free-form message.
                logger.log_memory_warning(f"Created disk-backed array '{name}': {estimated_mb:.1f} MB")
            print(f"  [{name}] Created DISK-backed array: {shape}, {estimated_mb:.1f} MB @ {filepath}")
            
            self.managed_arrays.append({
                'array': arr,
                'filepath': filepath,
                'name': name,
                'size_mb': estimated_mb,
                'mode': 'disk'
            })
        else:
            # Create normal RAM array
            arr = np.zeros(shape, dtype=dtype)
            
            print(f"  [{name}] Created RAM array: {shape}, {estimated_mb:.1f} MB")
            
            self.managed_arrays.append({
                'array': arr,
                'filepath': None,
                'name': name,
                'size_mb': estimated_mb,
                'mode': 'ram'
            })
        
        self.total_allocated_mb += estimated_mb
        return arr
    
    def get_memory_stats(self) -> dict:
        """Get current memory statistics"""
        ram_mb = sum(a['size_mb'] for a in self.managed_arrays if a['mode'] == 'ram')
        disk_mb = sum(a['size_mb'] for a in self.managed_arrays if a['mode'] == 'disk')
        
        return {
            'ram_mb': ram_mb,
            'disk_mb': disk_mb,
            'total_mb': self.total_allocated_mb,
            'available_gb': self.get_available_memory_gb(),
            'total_system_gb': self.get_total_memory_gb(),
            'array_count': len(self.managed_arrays),
            'mode': self.memory_mode
        }
    
    def print_stats(self):
        """Print memory usage statistics"""
        stats = self.get_memory_stats()
        
        print("\n" + "="*60)
        print("Memory Manager Statistics")
        print("="*60)
        print(f"Mode: {stats['mode'].upper()}")
        print(f"RAM Arrays: {stats['ram_mb']:.1f} MB")
        print(f"Disk Arrays: {stats['disk_mb']:.1f} MB")
        print(f"Total Allocated: {stats['total_mb']:.1f} MB ({stats['total_mb']/1024:.2f} GB)")
        print(f"System Available: {stats['available_gb']:.2f} GB / {stats['total_system_gb']:.2f} GB")
        print(f"Managed Arrays: {stats['array_count']}")
        print("="*60 + "\n")
    
    def cleanup(self):
        """
        Clean up memory-mapped files and temporary directory
        Call this when simulation is complete or on error
        """
        print("\nMemory Manager: Cleaning up...")
        
        # Flush and close memory-mapped arrays
        for info in self.managed_arrays:
            arr = info['array']
            if isinstance(arr, np.memmap):
                try:
                    arr.flush()
                    del arr  # Close the memmap
                except Exception as e:
                    logger.error(f"Error closing memmap {info['name']}: {e}")
                    print(f"Warning: Error closing memmap {info['name']}: {e}")
        
        # Remove temporary directory
        if self.session_dir and os.path.exists(self.session_dir):
            try:
                shutil.rmtree(self.session_dir)
                print(f"  Removed session directory: {self.session_dir}")
            except Exception as e:
                logger.error(f"Could not remove temp directory: {e}")
                print(f"Warning: Could not remove temp directory: {e}")
        
        self.managed_arrays.clear()
        self.total_allocated_mb = 0.0
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(memory_mode: Optional[str] = None) -> MemoryManager:
    """
    Get or create global memory manager instance
    
    Args:
        memory_mode: "ram", "disk", or "auto"
    
    Returns:
        MemoryManager instance
    """
    global _global_memory_manager
    
    requested_mode = memory_mode if memory_mode is not None else "auto"

    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager(memory_mode=requested_mode)
    elif memory_mode is not None and _global_memory_manager.memory_mode != requested_mode:
        # Only switch modes when explicitly requested by the caller.
        old_mode = _global_memory_manager.memory_mode
        _global_memory_manager.memory_mode = requested_mode
        print(f"Memory Manager: Mode changed from '{old_mode}' to '{requested_mode}'")
    
    return _global_memory_manager


def reset_memory_manager():
    """Reset global memory manager (cleanup and create new)"""
    global _global_memory_manager
    
    if _global_memory_manager is not None:
        _global_memory_manager.cleanup()
        _global_memory_manager = None


def create_managed_array(
    shape: Tuple,
    dtype=np.float32,
    name: str = "array",
    manager: Optional[MemoryManager] = None,
) -> np.ndarray:
    """
    Convenience function to create managed array
    
    Args:
        shape: Array shape
        dtype: NumPy dtype
        name: Descriptive name
    
    Returns:
        NumPy array (RAM or disk-backed)
    """
    manager_to_use = manager if manager is not None else get_memory_manager()
    return manager_to_use.create_array(shape, dtype, name)
