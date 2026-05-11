"""
Centralized Configuration System for Casting Solidification Simulator

Supports:
- YAML configuration files (user overrides)
- TOML configuration files (alternative format)
- Environment variable overrides (CI/CD)
- Default values for all settings
- Type validation and conversion

Configuration Hierarchy (highest priority first):
1. Environment variables (CASTSIM_*)
2. User config file (~/.castsim/config.yaml or config.toml)
3. Project config file (./castsim.yaml or ./castsim.toml)
4. Built-in defaults

Usage:
    from config.settings import settings, Settings
    
    # Access settings
    voxel_size = settings.mesh.base_voxel_size
    alloy = settings.simulation.default_alloy
    
    # Reload from files
    settings.reload()
    
    # Override programmatically
    settings.mesh.base_voxel_size = 0.05
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any, Dict, List, Union, TypeVar, Type
import os
import json

# Import exception classes
from config.exceptions import (
    ConfigurationError, 
    ConfigFileError, 
    MissingConfigError,
    ValidationError,
    BoundsError
)


# Type variable for generic config loading
T = TypeVar('T')


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class MeshSettings:
    """Voxel mesh generation settings"""
    base_voxel_size: float = 0.1  # inches
    max_refinement_level: int = 3
    min_voxel_size: float = 0.02  # inches
    max_voxel_size: float = 0.5   # inches
    memory_mode: str = "auto"     # "auto", "ram", "disk"
    use_octree: bool = True
    use_chunked_voxelization: bool = True
    neighbor_search_radius_factor: float = 2.0
    
    def __post_init__(self) -> None:
        """Validate mesh settings"""
        if self.base_voxel_size < self.min_voxel_size:
            self.base_voxel_size = self.min_voxel_size
        if self.base_voxel_size > self.max_voxel_size:
            self.base_voxel_size = self.max_voxel_size
        if self.memory_mode not in ("auto", "ram", "disk"):
            self.memory_mode = "auto"


@dataclass
class SimulationSettings:
    """Physics simulation settings"""
    default_alloy: str = "WCB"
    time_step: float = 0.1        # seconds
    max_simulation_time: float = 1000.0  # seconds
    convergence_threshold: float = 1e-6
    
    # Analysis options (defaults)
    enable_shrinkage: bool = True
    enable_hotspot: bool = True
    enable_niyama: bool = False
    enable_hot_tear: bool = False
    enable_sand_burn_in: bool = False
    enable_fraction_liquid: bool = True
    
    # Sand burn-in threshold (20% default)
    sand_burn_in_threshold: float = 0.2
    
    # Performance options
    use_gpu: bool = True
    use_incremental_solidification: bool = False
    
    # Niyama thresholds by alloy (can be overridden)
    niyama_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "WCB": 1.0,
        "CF8": 0.7,
        "C958": 1.2
    })


@dataclass  
class VisualizationSettings:
    """3D visualization settings"""
    default_colormap: str = "red_gradient"
    default_threshold: float = 0.2
    show_legend: bool = True
    show_mesh: bool = True
    show_results: bool = True
    
    # Animation export
    default_fps: int = 10
    animation_format: str = "gif"  # "gif" or "mp4"
    
    # Cross-section defaults
    cross_section_enabled: bool = False
    cross_section_axis: str = "Z"
    cross_section_position: float = 0.5


@dataclass
class UISettings:
    """User interface settings"""
    theme: str = "dark"           # "dark" or "light"
    language: str = "en"          # Locale code
    window_width: int = 1400
    window_height: int = 900
    sidebar_width: int = 250
    console_visible: bool = True
    recent_files_max: int = 10
    
    # Auto-save
    auto_save_enabled: bool = False
    auto_save_interval: int = 300  # seconds


@dataclass
class PerformanceSettings:
    """Performance and resource management settings"""
    # Memory limits
    max_ram_percent: float = 70.0       # Max % of RAM to use
    disk_threshold_mb: float = 500.0    # Switch to disk above this
    abort_threshold_percent: float = 90.0
    
    # Parallel processing
    max_workers: int = 0                # 0 = auto (physical cores - 1)
    chunk_size: int = 10000             # Voxels per chunk
    part_assignment_mode: str = "auto"  # auto, parallel, or legacy
    part_assignment_workers: int = 0    # 0 = auto
    part_assignment_batch_size: int = 2500
    
    # GPU settings
    gpu_backend: str = "auto"           # "auto", "metal", "cuda", "opencl", "cpu"
    gpu_memory_limit_mb: float = 0      # 0 = auto


@dataclass
class PathSettings:
    """File path settings"""
    default_stl_directory: str = ""
    default_output_directory: str = ""
    temp_directory: str = ""
    log_directory: str = ""
    
    # Config file paths (set during initialization)
    user_config_dir: str = ""
    
    def __post_init__(self) -> None:
        """Set default paths if not specified"""
        if not self.user_config_dir:
            self.user_config_dir = str(Path.home() / ".castsim")
        if not self.temp_directory:
            import tempfile
            self.temp_directory = tempfile.gettempdir()
        if not self.log_directory:
            self.log_directory = str(Path(self.user_config_dir) / "logs")


@dataclass
class ReportSettings:
    """Report generation settings"""
    default_format: str = "html"        # "html" or "pdf"
    include_screenshots: bool = True
    include_recommendations: bool = True
    company_name: str = ""
    company_logo_path: str = ""
    template_directory: str = ""


@dataclass
class PluginSettings:
    """Plugin system settings"""
    enabled: bool = True
    plugin_directories: List[str] = field(default_factory=list)
    disabled_plugins: List[str] = field(default_factory=list)


# ============================================================================
# Main Settings Container
# ============================================================================

@dataclass
class Settings:
    """
    Main settings container for the application.
    
    Aggregates all configuration sections and provides
    methods for loading/saving configuration.
    """
    mesh: MeshSettings = field(default_factory=MeshSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)
    ui: UISettings = field(default_factory=UISettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    report: ReportSettings = field(default_factory=ReportSettings)
    plugins: PluginSettings = field(default_factory=PluginSettings)
    solidification_params: Any = field(default=None)
    
    # Metadata
    _config_files_loaded: List[str] = field(default_factory=list)
    _env_vars_applied: List[str] = field(default_factory=list)
    
    def reload(self) -> None:
        """Reload configuration from all sources"""
        self._config_files_loaded.clear()
        self._env_vars_applied.clear()
        
        # Load in order of increasing priority
        self._load_project_config()
        self._load_user_config()
        self._apply_environment_variables()
    
    def _load_project_config(self) -> None:
        """Load project-local config file"""
        for filename in ["castsim.yaml", "castsim.yml", "castsim.toml"]:
            filepath = Path.cwd() / filename
            if filepath.exists():
                self._load_config_file(filepath)
                break
    
    def _load_user_config(self) -> None:
        """Load user config file from home directory"""
        user_config_dir = Path(self.paths.user_config_dir)
        for filename in ["config.yaml", "config.yml", "config.toml"]:
            filepath = user_config_dir / filename
            if filepath.exists():
                self._load_config_file(filepath)
                break
    
    def _load_config_file(self, filepath: Path) -> None:
        """Load configuration from a file"""
        try:
            suffix = filepath.suffix.lower()
            
            if suffix in (".yaml", ".yml"):
                data = self._load_yaml(filepath)
            elif suffix == ".toml":
                data = self._load_toml(filepath)
            else:
                raise ConfigFileError(
                    str(filepath),
                    f"Unsupported format: {suffix}",
                    suggestion="Use .yaml, .yml, or .toml format"
                )
            
            self._apply_config_dict(data)
            self._config_files_loaded.append(str(filepath))
            
        except ConfigFileError:
            raise
        except Exception as e:
            raise ConfigFileError(
                str(filepath),
                str(e)
            )
    
    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """Load YAML file, handling import gracefully"""
        try:
            import yaml
            with open(filepath, 'r') as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            raise ConfigFileError(
                str(filepath),
                "PyYAML not installed",
                suggestion="Install with: pip install pyyaml"
            )
    
    def _load_toml(self, filepath: Path) -> Dict[str, Any]:
        """Load TOML file, handling import gracefully"""
        try:
            # Python 3.11+ has built-in tomllib
            try:
                import tomllib
                with open(filepath, 'rb') as f:
                    return tomllib.load(f)
            except ImportError:
                import toml
                with open(filepath, 'r') as f:
                    return toml.load(f)
        except ImportError:
            raise ConfigFileError(
                str(filepath),
                "TOML parser not available",
                suggestion="Install with: pip install toml"
            )
    
    def _apply_config_dict(self, data: Dict[str, Any]) -> None:
        """Apply configuration dictionary to settings"""
        section_mapping = {
            "mesh": self.mesh,
            "simulation": self.simulation,
            "visualization": self.visualization,
            "ui": self.ui,
            "performance": self.performance,
            "paths": self.paths,
            "report": self.report,
            "plugins": self.plugins,
        }

        for section_name, section_data in data.items():
            if section_name in section_mapping and isinstance(section_data, dict):
                section = section_mapping[section_name]
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
            elif section_name == "solidification" and isinstance(section_data, dict):
                self.solidification_params = section_data
    
    def _apply_environment_variables(self) -> None:
        """Apply environment variable overrides"""
        env_prefix = "CASTSIM_"

        # Map environment variables to settings
        env_mapping = {
            # Mesh settings
            f"{env_prefix}VOXEL_SIZE": ("mesh", "base_voxel_size", float),
            f"{env_prefix}MEMORY_MODE": ("mesh", "memory_mode", str),

            # Simulation settings
            f"{env_prefix}DEFAULT_ALLOY": ("simulation", "default_alloy", str),
            f"{env_prefix}USE_GPU": ("simulation", "use_gpu", self._parse_bool),

            # Performance settings
            f"{env_prefix}MAX_WORKERS": ("performance", "max_workers", int),
            f"{env_prefix}GPU_BACKEND": ("performance", "gpu_backend", str),
            f"{env_prefix}PART_ASSIGNMENT_MODE": ("performance", "part_assignment_mode", str),
            f"{env_prefix}PART_ASSIGNMENT_WORKERS": ("performance", "part_assignment_workers", int),
            f"{env_prefix}PART_ASSIGNMENT_BATCH_SIZE": ("performance", "part_assignment_batch_size", int),

            # UI settings
            f"{env_prefix}THEME": ("ui", "theme", str),
        }

        section_mapping = {
            "mesh": self.mesh,
            "simulation": self.simulation,
            "performance": self.performance,
            "ui": self.ui,
        }

        for env_var, (section_name, attr_name, converter) in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    converted = converter(value)
                    section = section_mapping[section_name]
                    setattr(section, attr_name, converted)
                    self._env_vars_applied.append(env_var)
                except (ValueError, TypeError) as e:
                    # Log warning but don't fail
                    print(f"Warning: Invalid value for {env_var}: {value} ({e})")

        # Solidification params: nested CASTSIM_SOLIDIFICATION__SUBSECTION__KEY=VALUE
        # e.g. CASTSIM_SOLIDIFICATION__FEEDING__BASE_FEEDING_DISTANCE=8.0
        solid_params = self.solidification_params
        if solid_params is None:
            solid_params = {}
        if not isinstance(solid_params, dict):
            solid_params = {}

        import re

        def _set_nested(d, path, value):
            keys = path.split("__")
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value

        for env_var, env_val in os.environ.items():
            if env_var.startswith(f"{env_prefix}SOLIDIFICATION__"):
                key_path = env_var[len(f"{env_prefix}SOLIDIFICATION__"):]
                # Try float first, then int, then bool, else str
                for conv in (float, int, lambda v: v.lower() in ("true", "1", "yes", "on")):
                    try:
                        converted = conv(env_val)
                        break
                    except (ValueError, TypeError):
                        converted = env_val
                try:
                    _set_nested(solid_params, key_path, converted)
                    self._env_vars_applied.append(env_var)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {env_val} ({e})")

        self.solidification_params = solid_params if solid_params else None
    
    @staticmethod
    def _parse_bool(value: str) -> bool:
        """Parse boolean from string"""
        return value.lower() in ("true", "1", "yes", "on")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "mesh": asdict(self.mesh),
            "simulation": {
                k: v for k, v in asdict(self.simulation).items()
            },
            "visualization": asdict(self.visualization),
            "ui": asdict(self.ui),
            "performance": asdict(self.performance),
            "paths": asdict(self.paths),
            "report": asdict(self.report),
            "plugins": asdict(self.plugins),
        }
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save current settings to file"""
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        
        data = self.to_dict()
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(filepath, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                raise ConfigFileError(
                    str(filepath),
                    "PyYAML not installed",
                    suggestion="Install with: pip install pyyaml"
                )
        elif suffix == ".toml":
            try:
                import toml
                with open(filepath, 'w') as f:
                    toml.dump(data, f)
            except ImportError:
                raise ConfigFileError(
                    str(filepath),
                    "toml module not installed",
                    suggestion="Install with: pip install toml"
                )
        elif suffix == ".json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ConfigFileError(
                str(filepath),
                f"Unsupported format: {suffix}",
                suggestion="Use .yaml, .yml, .toml, or .json"
            )
    
    def print_config(self) -> None:
        """Print current configuration to console"""
        print("\n" + "=" * 60)
        print("Current Configuration")
        print("=" * 60)
        
        if self._config_files_loaded:
            print(f"Loaded files: {', '.join(self._config_files_loaded)}")
        if self._env_vars_applied:
            print(f"Environment overrides: {', '.join(self._env_vars_applied)}")
        
        print("\n[Mesh]")
        print(f"  Base voxel size: {self.mesh.base_voxel_size} inches")
        print(f"  Max refinement: {self.mesh.max_refinement_level}")
        print(f"  Memory mode: {self.mesh.memory_mode}")
        
        print("\n[Simulation]")
        print(f"  Default alloy: {self.simulation.default_alloy}")
        print(f"  GPU enabled: {self.simulation.use_gpu}")
        print(f"  Analyses: ", end="")
        analyses = []
        if self.simulation.enable_shrinkage:
            analyses.append("Shrinkage")
        if self.simulation.enable_hotspot:
            analyses.append("Hot Spots")
        if self.simulation.enable_niyama:
            analyses.append("Niyama")
        if self.simulation.enable_hot_tear:
            analyses.append("Hot Tear")
        if self.simulation.enable_sand_burn_in:
            analyses.append("Sand Burn-In")
        if self.simulation.enable_fraction_liquid:
            analyses.append("Fraction Liquid")
        print(", ".join(analyses) if analyses else "None")
        
        print("\n[Performance]")
        print(f"  Max RAM: {self.performance.max_ram_percent}%")
        print(f"  GPU backend: {self.performance.gpu_backend}")
        print(f"  Max workers: {self.performance.max_workers or 'auto'}")
        print(f"  Part assignment mode: {self.performance.part_assignment_mode}")
        print(f"  Part assignment workers: {self.performance.part_assignment_workers or 'auto'}")
        print(f"  Part assignment batch size: {self.performance.part_assignment_batch_size}")

        if self.solidification_params:
            print("\n[Solidification Parameters]")
            import json
            print(json.dumps(self.solidification_params, indent=2))

        print("=" * 60 + "\n")


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_voxel_size(value: float, settings: Settings) -> float:
    """Validate and clamp voxel size to allowed range"""
    min_val = settings.mesh.min_voxel_size
    max_val = settings.mesh.max_voxel_size
    
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


def validate_alloy_name(name: str) -> str:
    """Validate alloy name exists"""
    # Import here to avoid circular import
    from core.materials import get_all_alloy_names
    
    valid_names = get_all_alloy_names()
    if name not in valid_names:
        raise ValidationError(
            f"Unknown alloy: {name}",
            parameter="alloy",
            value=name,
            expected=f"One of: {', '.join(valid_names)}"
        )
    return name


# ============================================================================
# Global Settings Instance
# ============================================================================

def create_settings() -> Settings:
    """Create and initialize settings instance"""
    settings = Settings()
    settings.reload()
    return settings


# Global singleton instance
# Use get_settings() to access to allow for lazy initialization
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance (lazy initialization)"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = create_settings()
    return _settings_instance


def reset_settings() -> None:
    """Reset global settings to defaults"""
    global _settings_instance
    _settings_instance = create_settings()


# For convenient import: `from config.settings import settings`
# This creates the settings instance when the module is imported
settings = get_settings()


# ============================================================================
# Configuration File Templates
# ============================================================================

YAML_TEMPLATE = """# Casting Solidification Simulator Configuration
# Save as castsim.yaml in project directory or ~/.castsim/config.yaml

mesh:
  base_voxel_size: 0.1      # inches
  max_refinement_level: 3
  memory_mode: auto         # auto, ram, or disk
  use_octree: true

simulation:
  default_alloy: WCB
  enable_shrinkage: true
  enable_hotspot: true
  enable_niyama: false
  enable_hot_tear: false
  enable_sand_burn_in: false
  enable_fraction_liquid: false
  sand_burn_in_threshold: 0.2
  use_gpu: true

visualization:
  default_colormap: red_gradient
  default_threshold: 0.2
  show_legend: true

performance:
  max_ram_percent: 70
  gpu_backend: auto         # auto, metal, cuda, opencl, cpu
  part_assignment_mode: auto
  part_assignment_workers: 0
  part_assignment_batch_size: 2500

ui:
  theme: dark
  console_visible: true

report:
  default_format: html
  include_screenshots: true
  include_recommendations: true
"""

TOML_TEMPLATE = """# Casting Solidification Simulator Configuration
# Save as castsim.toml in project directory or ~/.castsim/config.toml

[mesh]
base_voxel_size = 0.1
max_refinement_level = 3
memory_mode = "auto"
use_octree = true

[simulation]
default_alloy = "WCB"
enable_shrinkage = true
enable_hotspot = true
enable_niyama = false
enable_hot_tear = false
enable_sand_burn_in = false
enable_fraction_liquid = false
sand_burn_in_threshold = 0.2
use_gpu = true

[visualization]
default_colormap = "red_gradient"
default_threshold = 0.2
show_legend = true

[performance]
max_ram_percent = 70
gpu_backend = "auto"
part_assignment_mode = "auto"
part_assignment_workers = 0
part_assignment_batch_size = 2500

[ui]
theme = "dark"
console_visible = true

[report]
default_format = "html"
include_screenshots = true
include_recommendations = true
"""


def create_default_config(directory: Union[str, Path], format: str = "yaml") -> Path:
    """
    Create default configuration file in specified directory.
    
    Args:
        directory: Directory to create config in
        format: "yaml" or "toml"
        
    Returns:
        Path to created config file
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    if format == "yaml":
        filepath = directory / "config.yaml"
        content = YAML_TEMPLATE
    elif format == "toml":
        filepath = directory / "config.toml"
        content = TOML_TEMPLATE
    else:
        raise ConfigurationError(
            f"Unknown config format: {format}",
            suggestion="Use 'yaml' or 'toml'"
        )
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return filepath
