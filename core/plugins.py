"""
Plugin System for Casting Solidification Simulator

Provides extensible interfaces for:
- Custom materials (alloys with custom properties)
- Custom analysis types (new simulation modes)
- Custom visualization modes (new rendering styles)

Plugin Discovery:
- Automatic discovery from plugin directories
- Entry point based loading (setuptools)
- Manual registration via API

Usage:
    from core.plugins import PluginManager, MaterialPlugin, AnalysisPlugin
    
    # Register a custom material plugin
    @material_plugin
    class MyCustomAlloy(MaterialPlugin):
        name = "CustomAlloy"
        ...
    
    # Get plugin manager
    pm = PluginManager()
    pm.discover_plugins()
    
    # Use custom material
    alloy = pm.get_material("CustomAlloy")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Optional, Any, Dict, List, Type, TypeVar, 
    Callable, Protocol, runtime_checkable, Union
)
from pathlib import Path
import importlib
import importlib.util
import sys
import logging

from config.exceptions import PluginError, PluginLoadError


# Type variables
T = TypeVar('T')
PluginT = TypeVar('PluginT', bound='BasePlugin')

# Logger
logger = logging.getLogger(__name__)


# ============================================================================
# Plugin Base Classes
# ============================================================================

@dataclass
class PluginMetadata:
    """Metadata about a plugin"""
    name: str
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "dependencies": self.dependencies
        }


class BasePlugin(ABC):
    """
    Base class for all plugins.
    
    Plugins must implement:
    - metadata property
    - initialize() method
    - cleanup() method (optional)
    """
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the plugin"""
        pass
    
    def cleanup(self) -> None:
        """Clean up plugin resources (optional)"""
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.metadata.name})>"


# ============================================================================
# Material Plugin Interface
# ============================================================================

@dataclass
class MaterialProperties:
    """
    Standard material properties for casting simulation.
    
    All properties use SI units internally but can be displayed
    in user-preferred units.
    """
    # Basic identification
    name: str
    
    # Temperature properties (°C)
    liquidus_temp: float
    solidus_temp: float
    pouring_temp: float
    
    # Density (kg/m³)
    density_liquid: float
    density_solid: float
    
    # Thermal properties
    thermal_conductivity: float      # W/(m·K)
    specific_heat: float             # J/(kg·K)
    latent_heat: float               # J/kg
    
    # Shrinkage
    linear_shrinkage: float          # %
    
    # Feeding
    feeding_distance_multiplier: float
    
    # Extended properties (optional)
    conductivity_liquid: Optional[float] = None
    specific_heat_liquid: Optional[float] = None
    dynamic_viscosity: float = 0.006
    thermal_expansion: float = 1.2e-4
    niyama_threshold: float = 1.0
    
    @property
    def thermal_diffusivity(self) -> float:
        """Calculate thermal diffusivity: k / (ρ * cp)"""
        return self.thermal_conductivity / (self.density_liquid * self.specific_heat)
    
    @property
    def freezing_range(self) -> float:
        """Temperature range between liquidus and solidus"""
        return self.liquidus_temp - self.solidus_temp
    
    @property
    def volumetric_shrinkage(self) -> float:
        """Density-derived volume loss from liquid volume, as a percentage."""
        return (1 - (self.density_liquid / self.density_solid)) * 100


class MaterialPlugin(BasePlugin):
    """
    Base class for material plugins.
    
    Implement this to add custom alloys to the material database.
    
    Example:
        class MyAlloy(MaterialPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="MyAlloy",
                    version="1.0.0",
                    description="Custom aluminum alloy"
                )
            
            def initialize(self) -> None:
                pass
            
            def get_properties(self) -> MaterialProperties:
                return MaterialProperties(
                    name="MyAlloy",
                    liquidus_temp=660,
                    solidus_temp=580,
                    ...
                )
    """
    
    @abstractmethod
    def get_properties(self) -> MaterialProperties:
        """Return material properties"""
        pass
    
    def get_temperature_dependent_conductivity(self, temperature: float) -> float:
        """
        Get thermal conductivity at given temperature.
        
        Override for temperature-dependent properties.
        Default returns constant conductivity.
        """
        return self.get_properties().thermal_conductivity
    
    def get_temperature_dependent_specific_heat(self, temperature: float) -> float:
        """
        Get specific heat at given temperature.
        
        Override for temperature-dependent properties.
        """
        return self.get_properties().specific_heat


# ============================================================================
# Analysis Plugin Interface
# ============================================================================

@dataclass
class AnalysisResultData:
    """Standard implementation of analysis result"""
    name: str
    data: Any  # numpy array of values
    units: str
    description: str = ""
    threshold_default: float = 0.3
    colormap: str = "red_gradient"
    
    def get_voxel_values(self) -> Any:
        """Get per-voxel values"""
        return self.data
    
    def get_threshold_mask(self, threshold: float) -> Any:
        """Get boolean mask of voxels above threshold"""
        import numpy as np
        return self.data > threshold


class AnalysisPlugin(BasePlugin):
    """
    Base class for analysis plugins.
    
    Implement this to add custom analysis types to the simulation.
    
    Example:
        class GrainSizeAnalysis(AnalysisPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="GrainSize",
                    description="Predict grain size from cooling rate"
                )
            
            def initialize(self) -> None:
                pass
            
            def analyze(self, mesh, simulation) -> AnalysisResultData:
                # Calculate grain size from cooling rate
                grain_size = self._calculate_grain_size(
                    simulation.cooling_rates
                )
                return AnalysisResultData(
                    name="Grain Size",
                    data=grain_size,
                    units="μm"
                )
    """
    
    @property
    @abstractmethod
    def analysis_name(self) -> str:
        """Name for this analysis type"""
        pass
    
    @abstractmethod
    def analyze(
        self, 
        mesh: Any,  # AdaptiveVoxelMesh
        simulation: Any  # SolidificationEngine
    ) -> AnalysisResultData:
        """
        Perform analysis on simulation results.
        
        Args:
            mesh: Voxel mesh with geometry
            simulation: Completed simulation engine
            
        Returns:
            AnalysisResultData with computed values
        """
        pass
    
    def get_ui_options(self) -> Dict[str, Any]:
        """
        Return UI options for this analysis.
        
        Override to customize visualization settings.
        """
        return {
            "colormap": "red_gradient",
            "threshold_min": 0.0,
            "threshold_max": 1.0,
            "threshold_default": 0.3
        }


# ============================================================================
# Visualization Plugin Interface
# ============================================================================

@dataclass
class ColorMap:
    """Definition of a colormap"""
    name: str
    key: str
    colors: List[tuple]  # List of (position, R, G, B) tuples
    description: str = ""
    
    def to_vtk_lut(self) -> Any:
        """Convert to VTK lookup table"""
        try:
            import vtk
            
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(256)
            lut.Build()
            
            # Interpolate colors
            for i in range(256):
                t = i / 255.0
                r, g, b = self._interpolate_color(t)
                lut.SetTableValue(i, r, g, b, 1.0)
            
            return lut
        except ImportError:
            return None
    
    def _interpolate_color(self, t: float) -> tuple:
        """Interpolate color at position t (0-1)"""
        # Find bracketing colors
        for i in range(len(self.colors) - 1):
            pos1, r1, g1, b1 = self.colors[i]
            pos2, r2, g2, b2 = self.colors[i + 1]
            
            if pos1 <= t <= pos2:
                # Linear interpolation
                alpha = (t - pos1) / (pos2 - pos1) if pos2 > pos1 else 0
                r = r1 + alpha * (r2 - r1)
                g = g1 + alpha * (g2 - g1)
                b = b1 + alpha * (b2 - b1)
                return (r, g, b)
        
        # Return last color if t > 1
        return (self.colors[-1][1], self.colors[-1][2], self.colors[-1][3])


class VisualizationPlugin(BasePlugin):
    """
    Base class for visualization plugins.
    
    Implement this to add custom visualization modes or colormaps.
    
    Example:
        class ThermalGradientVis(VisualizationPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="ThermalGradient",
                    description="Visualize thermal gradients"
                )
            
            def initialize(self) -> None:
                pass
            
            def get_colormap(self) -> ColorMap:
                return ColorMap(
                    name="Thermal Gradient",
                    key="thermal_gradient",
                    colors=[
                        (0.0, 0, 0, 1),    # Blue (cold)
                        (0.5, 0, 1, 0),    # Green (medium)
                        (1.0, 1, 0, 0),    # Red (hot)
                    ]
                )
    """
    
    @abstractmethod
    def get_colormap(self) -> ColorMap:
        """Return colormap definition"""
        pass
    
    def get_custom_actor(
        self,
        mesh: Any,
        data: Any,
        renderer: Any
    ) -> Optional[Any]:
        """
        Create custom VTK actor for visualization.
        
        Override for advanced custom rendering.
        Default returns None (use standard rendering).
        """
        return None


# ============================================================================
# Plugin Manager
# ============================================================================

class PluginManager:
    """
    Manages plugin discovery, loading, and access.
    
    Features:
    - Automatic discovery from configured directories
    - Entry point based loading (setuptools)
    - Thread-safe plugin access
    - Plugin lifecycle management
    """
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        """
        Initialize plugin manager.
        
        Args:
            plugin_dirs: List of directories to search for plugins
        """
        self._plugin_dirs = plugin_dirs or []
        
        # Plugin registries by type
        self._materials: Dict[str, MaterialPlugin] = {}
        self._analyses: Dict[str, AnalysisPlugin] = {}
        self._visualizations: Dict[str, VisualizationPlugin] = {}
        
        # All plugins for lifecycle management
        self._all_plugins: Dict[str, BasePlugin] = {}
        
        # Track loaded plugin modules
        self._loaded_modules: List[str] = []
    
    def add_plugin_directory(self, directory: Union[str, Path]) -> None:
        """Add a directory to search for plugins"""
        self._plugin_dirs.append(str(directory))
    
    def discover_plugins(self) -> int:
        """
        Discover and load all plugins.
        
        Searches:
        1. Configured plugin directories
        2. Entry points (setuptools)
        
        Returns:
            Number of plugins loaded
        """
        count = 0
        
        # Load from directories
        for plugin_dir in self._plugin_dirs:
            count += self._load_from_directory(Path(plugin_dir))
        
        # Load from entry points
        count += self._load_from_entry_points()
        
        logger.info(f"Discovered {count} plugins")
        return count
    
    def _load_from_directory(self, directory: Path) -> int:
        """Load plugins from a directory"""
        count = 0
        
        if not directory.exists():
            return 0
        
        # Find Python files
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            try:
                # Load module
                spec = importlib.util.spec_from_file_location(
                    py_file.stem, py_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find plugin classes in module
                    count += self._register_from_module(module)
                    self._loaded_modules.append(py_file.stem)
                    
            except Exception as e:
                logger.warning(f"Failed to load plugin from {py_file}: {e}")
        
        return count
    
    def _load_from_entry_points(self) -> int:
        """Load plugins from setuptools entry points"""
        count = 0
        
        try:
            if sys.version_info >= (3, 10):
                from importlib.metadata import entry_points
                eps = entry_points(group="castsim.plugins")
            else:
                from importlib.metadata import entry_points
                eps = entry_points().get("castsim.plugins", [])
            
            for ep in eps:
                try:
                    plugin_class = ep.load()
                    self.register_plugin(plugin_class())
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to load entry point {ep.name}: {e}")
                    
        except ImportError:
            pass
        
        return count
    
    def _register_from_module(self, module: Any) -> int:
        """Find and register plugin classes from a module"""
        count = 0
        
        for name in dir(module):
            obj = getattr(module, name)
            
            # Check if it's a plugin class (not the base classes)
            if (isinstance(obj, type) and 
                issubclass(obj, BasePlugin) and
                obj not in (BasePlugin, MaterialPlugin, AnalysisPlugin, VisualizationPlugin)):
                
                try:
                    instance = obj()
                    self.register_plugin(instance)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to instantiate plugin {name}: {e}")
        
        return count
    
    def register_plugin(self, plugin: BasePlugin) -> None:
        """
        Register a plugin instance.
        
        Args:
            plugin: Plugin instance to register
        """
        plugin_name = plugin.metadata.name
        
        # Check for duplicate
        if plugin_name in self._all_plugins:
            logger.warning(f"Plugin {plugin_name} already registered, skipping")
            return
        
        # Initialize plugin
        try:
            plugin.initialize()
        except Exception as e:
            raise PluginLoadError(plugin_name, f"Initialization failed: {e}")
        
        # Register by type
        self._all_plugins[plugin_name] = plugin
        
        if isinstance(plugin, MaterialPlugin):
            self._materials[plugin_name] = plugin
            logger.debug(f"Registered material plugin: {plugin_name}")
            
        elif isinstance(plugin, AnalysisPlugin):
            self._analyses[plugin_name] = plugin
            logger.debug(f"Registered analysis plugin: {plugin_name}")
            
        elif isinstance(plugin, VisualizationPlugin):
            self._visualizations[plugin_name] = plugin
            logger.debug(f"Registered visualization plugin: {plugin_name}")
    
    # ========================================================================
    # Material Plugin Access
    # ========================================================================
    
    def get_material(self, name: str) -> MaterialProperties:
        """Get material properties by name"""
        if name not in self._materials:
            raise PluginError(
                f"Material plugin not found: {name}",
                plugin_name=name,
                plugin_type="material"
            )
        return self._materials[name].get_properties()
    
    def get_all_materials(self) -> Dict[str, MaterialPlugin]:
        """Get all registered material plugins"""
        return dict(self._materials)
    
    def get_material_names(self) -> List[str]:
        """Get list of available material names"""
        return list(self._materials.keys())
    
    # ========================================================================
    # Analysis Plugin Access
    # ========================================================================
    
    def get_analysis(self, name: str) -> AnalysisPlugin:
        """Get analysis plugin by name"""
        if name not in self._analyses:
            raise PluginError(
                f"Analysis plugin not found: {name}",
                plugin_name=name,
                plugin_type="analysis"
            )
        return self._analyses[name]
    
    def get_all_analyses(self) -> Dict[str, AnalysisPlugin]:
        """Get all registered analysis plugins"""
        return dict(self._analyses)
    
    def run_analysis(
        self, 
        name: str, 
        mesh: Any, 
        simulation: Any
    ) -> AnalysisResultData:
        """Run analysis and return results"""
        plugin = self.get_analysis(name)
        return plugin.analyze(mesh, simulation)
    
    # ========================================================================
    # Visualization Plugin Access
    # ========================================================================
    
    def get_visualization(self, name: str) -> VisualizationPlugin:
        """Get visualization plugin by name"""
        if name not in self._visualizations:
            raise PluginError(
                f"Visualization plugin not found: {name}",
                plugin_name=name,
                plugin_type="visualization"
            )
        return self._visualizations[name]
    
    def get_colormap(self, name: str) -> ColorMap:
        """Get colormap from visualization plugin"""
        return self.get_visualization(name).get_colormap()
    
    def get_all_colormaps(self) -> Dict[str, ColorMap]:
        """Get all colormaps from visualization plugins"""
        return {
            name: plugin.get_colormap()
            for name, plugin in self._visualizations.items()
        }
    
    # ========================================================================
    # Lifecycle Management
    # ========================================================================
    
    def cleanup(self) -> None:
        """Clean up all plugins"""
        for plugin in self._all_plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up plugin {plugin.metadata.name}: {e}")
        
        self._all_plugins.clear()
        self._materials.clear()
        self._analyses.clear()
        self._visualizations.clear()
    
    def get_plugin_info(self) -> List[Dict[str, Any]]:
        """Get information about all loaded plugins"""
        return [
            {
                **plugin.metadata.to_dict(),
                "type": type(plugin).__bases__[0].__name__
            }
            for plugin in self._all_plugins.values()
        ]


# ============================================================================
# Plugin Decorators
# ============================================================================

# Global plugin manager for decorator registration
_default_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get or create global plugin manager"""
    global _default_plugin_manager
    if _default_plugin_manager is None:
        _default_plugin_manager = PluginManager()
    return _default_plugin_manager

