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
        self.core_tiers = []
        self.gpu_available = False
        self.neural_engine_available = False
        self.metal_available = False
        self.thread_cap = None

        self._detect_hardware()
        HardwareConfig._initialized = True

    def _detect_hardware(self):
        system = platform.system()

        self.physical_cores = psutil.cpu_count(logical=False) or (self.cpu_count // 2)
        self.logical_cores = psutil.cpu_count(logical=True) or self.cpu_count

        if system == 'Darwin':
            self._detect_macos_cpu()
        elif system == 'Windows':
            self._detect_windows_cpu()
        elif system == 'Linux':
            self._detect_linux_cpu()
        else:
            self.cpu_vendor = platform.processor()
            self.cpu_brand = platform.processor()

        self.total_memory = psutil.virtual_memory().total
        self.available_memory = psutil.virtual_memory().available

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
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True, timeout=2)
            self.cpu_brand = result.stdout.strip()

            if 'Apple' in self.cpu_brand:
                self.cpu_vendor = "Apple"

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

                self.core_tiers = self._detect_core_tiers()

                self.gpu_available = True
                self.metal_available = True
                self.neural_engine_available = True

            elif 'Intel' in self.cpu_brand:
                self.cpu_vendor = "Intel"
                self.metal_available = True

        except Exception as e:
            logger.warning(f"Could not detect macOS CPU details: {e}")
            print(f"  Warning: Could not detect macOS CPU details: {e}")
            self.cpu_brand = "Unknown macOS CPU"

    def _detect_core_tiers(self):
        tiers = []
        try:
            count = subprocess.run(['sysctl', '-n', 'hw.nperflevels'],
                                   capture_output=True, text=True, timeout=1)
            levels = int(count.stdout.strip()) if count.returncode == 0 else 0
        except Exception:
            levels = 0

        for index in range(min(levels, 4)):
            try:
                name_result = subprocess.run(['sysctl', '-n', f'hw.perflevel{index}.name'],
                                             capture_output=True, text=True, timeout=1)
                cores_result = subprocess.run(['sysctl', '-n', f'hw.perflevel{index}.physicalcpu'],
                                              capture_output=True, text=True, timeout=1)
                if name_result.returncode != 0 or cores_result.returncode != 0:
                    continue
                name = name_result.stdout.strip()
                cores = int(cores_result.stdout.strip())
            except Exception:
                continue
            if name and cores > 0:
                tiers.append({"name": name, "cores": cores})
        return tiers

    def _detect_windows_cpu(self):
        try:
            result = subprocess.run(['wmic', 'cpu', 'get', 'name'],
                                  capture_output=True, text=True, timeout=2)
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                self.cpu_brand = lines[1].strip()
        except:
            self.cpu_brand = platform.processor()

        if 'Intel' in self.cpu_brand:
            self.cpu_vendor = "Intel"
        elif 'AMD' in self.cpu_brand:
            self.cpu_vendor = "AMD"
        else:
            self.cpu_vendor = "Unknown"

        self.gpu_available = False

    def _detect_linux_cpu(self):
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        self.cpu_brand = line.split(':', 1)[1].strip()
                        break
        except:
            self.cpu_brand = platform.processor()

        if 'Intel' in self.cpu_brand:
            self.cpu_vendor = "Intel"
        elif 'AMD' in self.cpu_brand:
            self.cpu_vendor = "AMD"
        else:
            self.cpu_vendor = "Unknown"

        self.gpu_available = False

    def set_thread_cap(self, cap):
        self.thread_cap = cap if (cap is not None and cap > 0) else None

    def _apply_thread_cap(self, count: int) -> int:
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
        if task_type == 'compute':
            if self.physical_cores and self.physical_cores > 2:
                count = self.physical_cores - 1
            elif self.physical_cores:
                count = self.physical_cores
            else:
                count = max((self.logical_cores // 2) - 1, 1)
        elif task_type == 'io':
            count = self.physical_cores if self.physical_cores else max(self.logical_cores // 2, 1)
        else:
            count = max((self.physical_cores or (self.logical_cores // 2)) // 2, 1)
        return self._apply_thread_cap(count)

    def get_memory_limit(self, percentage: float = 0.7) -> int:
        return int(self.available_memory * percentage)

    def should_use_gpu(self) -> bool:
        return self.metal_available or self.gpu_available

    def should_use_neural_engine(self) -> bool:
        if not self.neural_engine_available:
            return False
        try:
            from core.surrogate.backend import is_coreml_active
            return is_coreml_active()
        except Exception:
            return False

    def get_chunk_size(self, total_size: int) -> int:
        optimal_chunks = self.get_optimal_thread_count('compute')
        return max(total_size // optimal_chunks, 1)

hardware = HardwareConfig()
