"""
Hardware Configuration and Detection for Cross-Platform Support
Supports Intel, AMD, and Apple Silicon processors
Optimizes compute allocation using physical cores only (no hyperthreading)
"""

import os
import psutil
import platform
import subprocess
from typing import Dict, Tuple, Optional

try:
    from core.error_logger import logger
except ImportError:
    from core.error_logger_dummy import logger

class HardwareConfig:
    """Detect and configure hardware resources for optimal performance"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if HardwareConfig._initialized:
            return
            
        self.cpu_count = os.cpu_count() or 4
        self.physical_cores = None
        self.logical_cores = None
        self.cpu_brand = "Unknown"
        self.cpu_vendor = "Unknown"
        self.performance_cores = None
        self.efficiency_cores = None
        self.gpu_available = False
        self.neural_engine_available = False
        self.metal_available = False
        self.thread_cap = None
        
        self._detect_hardware()
        HardwareConfig._initialized = True
    
    def _detect_hardware(self):
        """Detect CPU capabilities across Intel, AMD, and Apple Silicon"""
        system = platform.system()
        
        # Get CPU information
        self.physical_cores = psutil.cpu_count(logical=False) or (self.cpu_count // 2)
        self.logical_cores = psutil.cpu_count(logical=True) or self.cpu_count
        
        # Detect CPU vendor and brand
        if system == 'Darwin':  # macOS
            self._detect_macos_cpu()
        elif system == 'Windows':
            self._detect_windows_cpu()
        elif system == 'Linux':
            self._detect_linux_cpu()
        else:
            # Fallback for unknown systems
            self.cpu_vendor = platform.processor()
            self.cpu_brand = platform.processor()
        
        # Memory
        self.total_memory = psutil.virtual_memory().total
        self.available_memory = psutil.virtual_memory().available
        
        # Display configuration
        print(f"Hardware Configuration:")
        print(f"  CPU: {self.cpu_brand}")
        print(f"  Vendor: {self.cpu_vendor}")
        print(f"  Physical Cores: {self.physical_cores} (using physical cores only, no hyperthreading)")
        print(f"  Logical Cores: {self.logical_cores}")
        print(f"  Memory: {self.total_memory / (1024**3):.1f} GB")
        if self.metal_available:
            print(f"  GPU (Metal): Available")
        if self.neural_engine_available:
            print(f"  Neural Engine: Available")
    
    def _detect_macos_cpu(self):
        """Detect CPU on macOS (Apple Silicon or Intel)"""
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=2)
            self.cpu_brand = result.stdout.strip()
            
            if 'Apple' in self.cpu_brand:
                # Apple Silicon (M-series chips)
                self.cpu_vendor = "Apple"
                
                # Detect performance/efficiency core split
                try:
                    perf_result = subprocess.run(['sysctl', '-n', 'hw.perflevel0.logicalcpu'], 
                                                capture_output=True, text=True, timeout=1)
                    eff_result = subprocess.run(['sysctl', '-n', 'hw.perflevel1.logicalcpu'], 
                                              capture_output=True, text=True, timeout=1)
                    
                    if perf_result.returncode == 0:
                        self.performance_cores = int(perf_result.stdout.strip())
                    if eff_result.returncode == 0:
                        self.efficiency_cores = int(eff_result.stdout.strip())
                except:
                    pass
                
                # Apple Silicon has integrated GPU and Neural Engine
                self.gpu_available = True
                self.metal_available = True
                self.neural_engine_available = True
                
            elif 'Intel' in self.cpu_brand:
                # Intel CPU on Mac
                self.cpu_vendor = "Intel"
                self.metal_available = True  # Intel Macs still have Metal support
                
        except Exception as e:
            logger.warning(f"Could not detect macOS CPU details: {e}")
            print(f"  Warning: Could not detect macOS CPU details: {e}")
            self.cpu_brand = "Unknown macOS CPU"
    
    def _detect_windows_cpu(self):
        """Detect CPU on Windows (Intel or AMD)"""
        try:
            # Try to get CPU info from wmic
            result = subprocess.run(['wmic', 'cpu', 'get', 'name'], 
                                  capture_output=True, text=True, timeout=2)
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                self.cpu_brand = lines[1].strip()
        except:
            self.cpu_brand = platform.processor()
        
        # Determine vendor
        if 'Intel' in self.cpu_brand:
            self.cpu_vendor = "Intel"
        elif 'AMD' in self.cpu_brand:
            self.cpu_vendor = "AMD"
        else:
            self.cpu_vendor = "Unknown"
        
        # GPU detection would require additional libraries (not implemented)
        self.gpu_available = False
    
    def _detect_linux_cpu(self):
        """Detect CPU on Linux (Intel or AMD)"""
        try:
            # Try to read /proc/cpuinfo
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        self.cpu_brand = line.split(':', 1)[1].strip()
                        break
        except:
            self.cpu_brand = platform.processor()
        
        # Determine vendor
        if 'Intel' in self.cpu_brand:
            self.cpu_vendor = "Intel"
        elif 'AMD' in self.cpu_brand:
            self.cpu_vendor = "AMD"
        else:
            self.cpu_vendor = "Unknown"
        
        # GPU detection would require additional libraries (not implemented)
        self.gpu_available = False
    
    def set_thread_cap(self, cap):
        """
        Set a process-local cap on compute thread counts.

        Args:
            cap: Maximum thread count, or None to remove the cap.
                Values <= 0 are treated as "no cap".
        """
        self.thread_cap = cap if (cap is not None and cap > 0) else None

    def _apply_thread_cap(self, count: int) -> int:
        """Apply the configured thread cap to a computed thread count."""
        if self.thread_cap is not None:
            return max(1, min(count, self.thread_cap))
        try:
            env_cap = int(os.getenv("CASTSIM_THREAD_BUDGET", "") or 0)
        except ValueError:
            env_cap = 0
        if env_cap > 0:
            return max(1, min(count, env_cap))
        return count

    def get_optimal_thread_count(self, task_type: str = 'compute') -> int:
        """
        Get optimal thread count for different task types
        Uses PHYSICAL CORES ONLY (no hyperthreading) to minimize memory usage

        Args:
            task_type: 'compute' for CPU-intensive, 'io' for I/O-bound tasks

        Returns:
            Number of threads to use
        """
        if task_type == 'compute':
            # Use physical cores only, leave 1 core for system if we have more than 2
            if self.physical_cores and self.physical_cores > 2:
                count = self.physical_cores - 1
            elif self.physical_cores:
                count = self.physical_cores
            else:
                # Fallback: assume half of logical cores are physical
                count = max((self.logical_cores // 2) - 1, 1)
        elif task_type == 'io':
            # For I/O-bound tasks, can use more threads (but still prefer physical cores)
            count = self.physical_cores if self.physical_cores else max(self.logical_cores // 2, 1)
        else:
            # Default: use half of physical cores
            count = max((self.physical_cores or (self.logical_cores // 2)) // 2, 1)
        return self._apply_thread_cap(count)
    
    def get_memory_limit(self, percentage: float = 0.7) -> int:
        """Get memory limit in bytes (default 70% of available)"""
        return int(self.available_memory * percentage)
    
    def should_use_gpu(self) -> bool:
        """
        Check if GPU acceleration should be used
        
        Returns True if Metal (Apple Silicon) or CUDA (NVIDIA) is available.
        """
        return self.metal_available or self.gpu_available
    
    def should_use_neural_engine(self) -> bool:
        """
        Check if Neural Engine should be used
        
        Note: Neural Engine is for ML inference, not numerical simulation.
        Returns False as current physics engine doesn't use ML models.
        """
        return False
    
    def get_chunk_size(self, total_size: int) -> int:
        """Calculate optimal chunk size for parallel processing"""
        optimal_chunks = self.get_optimal_thread_count('compute')
        return max(total_size // optimal_chunks, 1)


# Global hardware configuration instance
hardware = HardwareConfig()
