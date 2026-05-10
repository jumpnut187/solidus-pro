"""
Dependency Injection Container for Casting Solidification Simulator

Replaces global singletons (hardware, _global_memory_manager) with
a proper dependency injection system. Enables:
- Easier testing and mocking
- Multiple configurations in same process
- Cleaner architecture with explicit dependencies

Usage:
    from core.container import Container, get_container
    
    # Get global container
    container = get_container()
    
    # Access services
    hardware = container.hardware
    memory_manager = container.memory_manager
    settings = container.settings
    
    # Create isolated container for testing
    test_container = Container()
    test_container.register('hardware', MockHardware())

Design Pattern: Service Locator with lazy initialization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Optional, Any, Dict, TypeVar, Generic, Callable, 
    Type, Union, Protocol, runtime_checkable
)
from functools import cached_property
import threading
from weakref import WeakValueDictionary


T = TypeVar('T')


# ============================================================================
# Service Protocols (Interfaces)
# ============================================================================

@runtime_checkable
class HardwareConfigProtocol(Protocol):
    """Protocol for hardware configuration service"""
    cpu_count: int
    physical_cores: Optional[int]
    logical_cores: Optional[int]
    cpu_brand: str
    gpu_available: bool
    metal_available: bool
    total_memory: int
    available_memory: int
    
    def get_optimal_thread_count(self, task_type: str) -> int: ...
    def get_memory_limit(self, percentage: float) -> int: ...
    def should_use_gpu(self) -> bool: ...


@runtime_checkable
class MemoryManagerProtocol(Protocol):
    """Protocol for memory manager service"""
    memory_mode: str
    total_allocated_mb: float
    
    def create_array(self, shape: tuple, dtype: Any, name: str) -> Any: ...
    def get_memory_stats(self) -> dict: ...
    def cleanup(self) -> None: ...


@runtime_checkable
class SettingsProtocol(Protocol):
    """Protocol for settings service"""
    mesh: Any
    simulation: Any
    visualization: Any
    performance: Any
    
    def reload(self) -> None: ...
    def to_dict(self) -> dict: ...


# ============================================================================
# Service Registry
# ============================================================================

class ServiceDescriptor(Generic[T]):
    """
    Descriptor for lazy service initialization.
    
    Services are created on first access and cached.
    Supports factory functions for deferred initialization.
    """
    
    def __init__(
        self, 
        factory: Callable[[], T],
        singleton: bool = True
    ):
        self.factory = factory
        self.singleton = singleton
        self._instance: Optional[T] = None
        self._lock = threading.Lock()
    
    def get(self) -> T:
        """Get or create service instance"""
        if self.singleton:
            if self._instance is None:
                with self._lock:
                    if self._instance is None:
                        self._instance = self.factory()
            return self._instance
        else:
            return self.factory()
    
    def reset(self) -> None:
        """Reset cached instance (for testing)"""
        with self._lock:
            self._instance = None


# ============================================================================
# Main Container
# ============================================================================

class Container:
    """
    Dependency injection container for application services.
    
    Provides centralized access to:
    - Hardware configuration
    - Memory manager
    - Application settings
    - GPU compute manager
    - Plugin manager (when implemented)
    
    Thread-safe with lazy initialization.
    
    Example:
        container = Container()
        
        # Access services via properties (lazy initialization)
        hw = container.hardware
        mem = container.memory_manager
        
        # Override for testing
        container.register('hardware', mock_hardware)
    """
    
    def __init__(self, parent: Optional[Container] = None):
        """
        Initialize container.
        
        Args:
            parent: Optional parent container for hierarchical lookup
        """
        self._services: Dict[str, ServiceDescriptor] = {}
        self._overrides: Dict[str, Any] = {}
        self._parent = parent
        self._lock = threading.RLock()
        
        # Register default service factories
        self._register_defaults()
    
    def _register_defaults(self) -> None:
        """Register default service factories"""
        # Hardware config
        self._services['hardware'] = ServiceDescriptor(
            self._create_hardware_config
        )
        
        # Memory manager
        self._services['memory_manager'] = ServiceDescriptor(
            self._create_memory_manager
        )
        
        # Settings
        self._services['settings'] = ServiceDescriptor(
            self._create_settings
        )
        
        # GPU compute manager
        self._services['gpu_compute'] = ServiceDescriptor(
            self._create_gpu_compute
        )
    
    def _create_hardware_config(self) -> Any:
        """Factory for hardware configuration"""
        from config.hardware_config import HardwareConfig
        return HardwareConfig()
    
    def _create_memory_manager(self) -> Any:
        """Factory for memory manager"""
        from core.memory_manager import MemoryManager
        # Get memory mode from settings if available
        try:
            settings = self.settings
            memory_mode = settings.mesh.memory_mode
        except Exception:
            memory_mode = "auto"
        return MemoryManager(memory_mode=memory_mode)
    
    def _create_settings(self) -> Any:
        """Factory for application settings"""
        from config.settings import create_settings
        return create_settings()
    
    def _create_gpu_compute(self) -> Any:
        """Factory for GPU compute manager"""
        try:
            from core.gpu_compute import GPUComputeManager
            return GPUComputeManager()
        except ImportError:
            return None
    
    # ========================================================================
    # Service Access Properties
    # ========================================================================
    
    @property
    def hardware(self) -> HardwareConfigProtocol:
        """Get hardware configuration service"""
        return self.get('hardware')
    
    @property
    def memory_manager(self) -> MemoryManagerProtocol:
        """Get memory manager service"""
        return self.get('memory_manager')
    
    @property
    def settings(self) -> SettingsProtocol:
        """Get application settings service"""
        return self.get('settings')
    
    @property
    def gpu_compute(self) -> Optional[Any]:
        """Get GPU compute manager (None if not available)"""
        return self.get('gpu_compute')
    
    # ========================================================================
    # Service Management
    # ========================================================================
    
    def get(self, name: str) -> Any:
        """
        Get a service by name.
        
        Lookup order:
        1. Explicit overrides
        2. Registered services
        3. Parent container (if set)
        
        Args:
            name: Service name
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service not found
        """
        with self._lock:
            # Check overrides first
            if name in self._overrides:
                return self._overrides[name]
            
            # Check registered services
            if name in self._services:
                return self._services[name].get()
            
            # Check parent
            if self._parent is not None:
                return self._parent.get(name)
            
            raise KeyError(f"Service not found: {name}")
    
    def register(
        self, 
        name: str, 
        service_or_factory: Union[Any, Callable[[], Any]],
        singleton: bool = True
    ) -> None:
        """
        Register a service or factory.
        
        Args:
            name: Service name
            service_or_factory: Service instance or factory function
            singleton: If True, factory result is cached
        """
        with self._lock:
            if callable(service_or_factory) and not isinstance(service_or_factory, type):
                # It's a factory function
                self._services[name] = ServiceDescriptor(
                    service_or_factory, 
                    singleton=singleton
                )
            else:
                # It's an instance - wrap in descriptor that returns it
                self._services[name] = ServiceDescriptor(
                    lambda s=service_or_factory: s,
                    singleton=True
                )
    
    def override(self, name: str, instance: Any) -> None:
        """
        Override a service with a specific instance.
        
        Useful for testing - overrides take precedence over
        registered services.
        
        Args:
            name: Service name
            instance: Instance to use
        """
        with self._lock:
            self._overrides[name] = instance
    
    def clear_override(self, name: str) -> None:
        """Clear an override"""
        with self._lock:
            self._overrides.pop(name, None)
    
    def clear_all_overrides(self) -> None:
        """Clear all overrides"""
        with self._lock:
            self._overrides.clear()
    
    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset service(s) to force re-creation.
        
        Args:
            name: Service name, or None to reset all
        """
        with self._lock:
            if name is not None:
                if name in self._services:
                    self._services[name].reset()
                self._overrides.pop(name, None)
            else:
                for service in self._services.values():
                    service.reset()
                self._overrides.clear()
    
    def child(self) -> Container:
        """Create a child container that inherits from this one"""
        return Container(parent=self)
    
    # ========================================================================
    # Context Manager Support
    # ========================================================================
    
    def __enter__(self) -> Container:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Cleanup memory manager
        try:
            mm = self.get('memory_manager')
            if mm and hasattr(mm, 'cleanup'):
                mm.cleanup()
        except KeyError:
            pass


# ============================================================================
# Global Container Instance
# ============================================================================

_global_container: Optional[Container] = None
_container_lock = threading.Lock()


def get_container() -> Container:
    """
    Get the global container instance.
    
    Thread-safe lazy initialization.
    
    Returns:
        Global Container instance
    """
    global _global_container
    
    if _global_container is None:
        with _container_lock:
            if _global_container is None:
                _global_container = Container()
    
    return _global_container


def reset_container() -> None:
    """Reset global container (useful for testing)"""
    global _global_container
    
    with _container_lock:
        if _global_container is not None:
            _global_container.reset()
        _global_container = None


def set_container(container: Container) -> None:
    """
    Set the global container instance.
    
    Useful for testing or custom configurations.
    
    Args:
        container: Container to use as global
    """
    global _global_container
    
    with _container_lock:
        _global_container = container


# ============================================================================
# Injection Decorator
# ============================================================================

def inject(*service_names: str):
    """
    Decorator for automatic dependency injection.
    
    Injects services as keyword arguments to the decorated function.
    
    Example:
        @inject('hardware', 'settings')
        def my_function(x, y, hardware=None, settings=None):
            # hardware and settings are injected automatically
            cores = hardware.physical_cores
            voxel_size = settings.mesh.base_voxel_size
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            container = get_container()
            for name in service_names:
                if name not in kwargs:
                    kwargs[name] = container.get(name)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# Testing Utilities
# ============================================================================

class MockHardwareConfig:
    """Mock hardware configuration for testing"""
    
    def __init__(
        self,
        cpu_count: int = 8,
        physical_cores: int = 4,
        memory_gb: float = 16.0,
        gpu_available: bool = False
    ):
        self.cpu_count = cpu_count
        self.physical_cores = physical_cores
        self.logical_cores = cpu_count
        self.cpu_brand = "Mock CPU"
        self.cpu_vendor = "Mock"
        self.gpu_available = gpu_available
        self.metal_available = gpu_available
        self.neural_engine_available = False
        self.total_memory = int(memory_gb * 1024 ** 3)
        self.available_memory = int(memory_gb * 0.8 * 1024 ** 3)
        self.performance_cores = physical_cores
        self.efficiency_cores = 0
    
    def get_optimal_thread_count(self, task_type: str = 'compute') -> int:
        if task_type == 'compute':
            return max(self.physical_cores - 1, 1)
        return self.physical_cores
    
    def get_memory_limit(self, percentage: float = 0.7) -> int:
        return int(self.available_memory * percentage)
    
    def should_use_gpu(self) -> bool:
        return self.gpu_available


class MockMemoryManager:
    """Mock memory manager for testing"""
    
    def __init__(self, memory_mode: str = "ram"):
        self.memory_mode = memory_mode
        self.total_allocated_mb = 0.0
        self._arrays: Dict[str, Any] = {}
    
    def create_array(self, shape: tuple, dtype: Any = None, name: str = "array") -> Any:
        import numpy as np
        arr = np.zeros(shape, dtype=dtype or np.float32)
        self._arrays[name] = arr
        self.total_allocated_mb += arr.nbytes / (1024 * 1024)
        return arr
    
    def get_memory_stats(self) -> dict:
        return {
            'ram_mb': self.total_allocated_mb,
            'disk_mb': 0.0,
            'total_mb': self.total_allocated_mb,
            'available_gb': 8.0,
            'total_system_gb': 16.0,
            'array_count': len(self._arrays),
            'mode': self.memory_mode
        }
    
    def cleanup(self) -> None:
        self._arrays.clear()
        self.total_allocated_mb = 0.0


def create_test_container(
    hardware: Optional[Any] = None,
    memory_manager: Optional[Any] = None,
    settings: Optional[Any] = None
) -> Container:
    """
    Create a container configured for testing.
    
    Args:
        hardware: Mock hardware config (default: MockHardwareConfig)
        memory_manager: Mock memory manager (default: MockMemoryManager)
        settings: Mock settings (default: None, uses real settings)
        
    Returns:
        Configured Container for testing
    """
    container = Container()
    
    if hardware is not None:
        container.override('hardware', hardware)
    else:
        container.override('hardware', MockHardwareConfig())
    
    if memory_manager is not None:
        container.override('memory_manager', memory_manager)
    else:
        container.override('memory_manager', MockMemoryManager())
    
    if settings is not None:
        container.override('settings', settings)
    
    return container
