"""
Custom Exception Hierarchy for Casting Solidification Simulator

Provides user-friendly error messages with suggested fixes.
All exceptions inherit from CastSimError for easy catching.

Usage:
    from config.exceptions import ValidationError, ConfigurationError
    
    if voxel_size <= 0:
        raise ValidationError(
            "Voxel size must be positive",
            parameter="voxel_size",
            value=voxel_size,
            suggestion="Use a value between 0.02 and 0.5 inches"
        )
"""

from typing import Optional, Any, Dict, List


class CastSimError(Exception):
    """
    Base exception for all Casting Simulator errors.
    
    Provides structured error information including:
    - User-friendly message
    - Technical details for debugging
    - Suggested fixes
    
    Attributes:
        message: Human-readable error description
        details: Technical details dict for debugging
        suggestion: Suggested fix or action
        error_code: Optional error code for categorization
    """
    
    def __init__(
        self, 
        message: str,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        error_code: Optional[str] = None
    ):
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion
        self.error_code = error_code
        
        # Build full error message
        full_message = message
        if suggestion:
            full_message += f"\n💡 Suggestion: {suggestion}"
        if error_code:
            full_message = f"[{error_code}] {full_message}"
            
        super().__init__(full_message)
    
    def __str__(self) -> str:
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "suggestion": self.suggestion,
            "details": self.details
        }


# ============================================================================
# Validation Errors (Invalid input data)
# ============================================================================

class ValidationError(CastSimError):
    """
    Raised when input validation fails.
    
    Use for:
    - Invalid parameter values
    - Missing required data
    - Data format issues
    """
    
    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Optional[Any] = None,
        expected: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        details = {}
        if parameter:
            details["parameter"] = parameter
        if value is not None:
            details["received_value"] = repr(value)
        if expected:
            details["expected"] = expected
            
        if not suggestion and expected:
            suggestion = f"Expected: {expected}"
            
        super().__init__(
            message,
            details=details,
            suggestion=suggestion,
            error_code="VAL001"
        )
        self.parameter = parameter
        self.value = value
        self.expected = expected


class BoundsError(ValidationError):
    """Raised when a value is outside acceptable bounds"""
    
    def __init__(
        self,
        parameter: str,
        value: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ):
        bounds_str = ""
        if min_val is not None and max_val is not None:
            bounds_str = f"{min_val} to {max_val}"
        elif min_val is not None:
            bounds_str = f">= {min_val}"
        elif max_val is not None:
            bounds_str = f"<= {max_val}"
            
        super().__init__(
            f"Parameter '{parameter}' value {value} is out of bounds",
            parameter=parameter,
            value=value,
            expected=bounds_str,
            suggestion=f"Use a value in range: {bounds_str}"
        )
        self.min_val = min_val
        self.max_val = max_val


class MissingDataError(ValidationError):
    """Raised when required data is missing"""
    
    def __init__(
        self,
        data_name: str,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            f"Required data missing: {data_name}",
            parameter=data_name,
            suggestion=suggestion or f"Provide {data_name} before proceeding"
        )


# ============================================================================
# Configuration Errors (Settings and setup issues)
# ============================================================================

class ConfigurationError(CastSimError):
    """
    Raised when configuration is invalid or incomplete.
    
    Use for:
    - Invalid config file format
    - Missing required settings
    - Incompatible configuration combinations
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_file:
            details["config_file"] = config_file
            
        super().__init__(
            message,
            details=details,
            suggestion=suggestion,
            error_code="CFG001"
        )
        self.config_key = config_key
        self.config_file = config_file


class ConfigFileError(ConfigurationError):
    """Raised when config file cannot be read or parsed"""
    
    def __init__(
        self,
        filepath: str,
        reason: str,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            f"Cannot load config file '{filepath}': {reason}",
            config_file=filepath,
            suggestion=suggestion or "Check file exists and has valid YAML/TOML format"
        )


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing"""
    
    def __init__(
        self,
        key: str,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            f"Required configuration missing: {key}",
            config_key=key,
            suggestion=suggestion or f"Add '{key}' to your configuration file"
        )


# ============================================================================
# File/IO Errors
# ============================================================================

class FileError(CastSimError):
    """Base class for file-related errors"""
    
    def __init__(
        self,
        message: str,
        filepath: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        details = {}
        if filepath:
            details["filepath"] = filepath
            
        super().__init__(
            message,
            details=details,
            suggestion=suggestion,
            error_code="FILE001"
        )
        self.filepath = filepath


class STLLoadError(FileError):
    """Raised when STL file cannot be loaded"""
    
    def __init__(
        self,
        filepath: str,
        reason: str,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            f"Cannot load STL file '{filepath}': {reason}",
            filepath=filepath,
            suggestion=suggestion or "Ensure file is a valid binary or ASCII STL"
        )


class MeshError(FileError):
    """Raised when mesh processing fails"""
    
    def __init__(
        self,
        message: str,
        mesh_info: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            message,
            suggestion=suggestion
        )
        if mesh_info:
            self.details.update(mesh_info)


class InvalidMeshError(MeshError):
    """Raised when mesh geometry is invalid"""
    
    def __init__(
        self,
        reason: str,
        vertex_count: Optional[int] = None,
        face_count: Optional[int] = None
    ):
        mesh_info = {}
        if vertex_count is not None:
            mesh_info["vertex_count"] = vertex_count
        if face_count is not None:
            mesh_info["face_count"] = face_count
            
        super().__init__(
            f"Invalid mesh geometry: {reason}",
            mesh_info=mesh_info,
            suggestion="Try repairing the mesh with a tool like MeshLab or Blender"
        )


# ============================================================================
# Simulation Errors
# ============================================================================

class SimulationError(CastSimError):
    """
    Raised when simulation encounters an error.
    
    Use for:
    - Physics calculation failures
    - Convergence issues
    - Invalid simulation state
    """
    
    def __init__(
        self,
        message: str,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
        suggestion: Optional[str] = None
    ):
        details = {}
        if phase:
            details["phase"] = phase
        if iteration is not None:
            details["iteration"] = iteration
            
        super().__init__(
            message,
            details=details,
            suggestion=suggestion,
            error_code="SIM001"
        )
        self.phase = phase
        self.iteration = iteration


class MeshGenerationError(SimulationError):
    """Raised when voxel mesh generation fails"""
    
    def __init__(
        self,
        reason: str,
        voxel_size: Optional[float] = None,
        estimated_voxels: Optional[int] = None
    ):
        details_dict = {}
        if voxel_size:
            details_dict["voxel_size"] = voxel_size
        if estimated_voxels:
            details_dict["estimated_voxels"] = estimated_voxels
            
        super().__init__(
            f"Mesh generation failed: {reason}",
            phase="mesh_generation",
            suggestion="Try increasing voxel size or reducing model complexity"
        )
        self.details.update(details_dict)


class SolidificationError(SimulationError):
    """Raised when solidification simulation fails"""
    
    def __init__(
        self,
        reason: str,
        time_step: Optional[float] = None,
        iteration: Optional[int] = None
    ):
        super().__init__(
            f"Solidification simulation failed: {reason}",
            phase="solidification",
            iteration=iteration,
            suggestion="Check material properties and mesh quality"
        )
        if time_step:
            self.details["time_step"] = time_step


class ConvergenceError(SimulationError):
    """Raised when simulation fails to converge"""
    
    def __init__(
        self,
        max_iterations: int,
        final_residual: Optional[float] = None
    ):
        super().__init__(
            f"Simulation did not converge after {max_iterations} iterations",
            phase="convergence",
            iteration=max_iterations,
            suggestion="Try smaller time steps or check boundary conditions"
        )
        if final_residual is not None:
            self.details["final_residual"] = final_residual


# ============================================================================
# Resource Errors (Memory, GPU, etc.)
# ============================================================================

class ResourceError(CastSimError):
    """Base class for resource-related errors"""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
            
        super().__init__(
            message,
            details=details,
            suggestion=suggestion,
            error_code="RES001"
        )
        self.resource_type = resource_type


class MemoryError(ResourceError):
    """Raised when memory limits are exceeded"""
    
    def __init__(
        self,
        required_mb: float,
        available_mb: float,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            f"Insufficient memory: need {required_mb:.0f} MB, only {available_mb:.0f} MB available",
            resource_type="memory",
            suggestion=suggestion or "Increase voxel size to reduce memory requirements"
        )
        self.details["required_mb"] = required_mb
        self.details["available_mb"] = available_mb


class GPUError(ResourceError):
    """Raised when GPU acceleration fails"""
    
    def __init__(
        self,
        message: str,
        backend: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            f"GPU error: {message}",
            resource_type="gpu",
            suggestion=suggestion or "Disable GPU acceleration in Performance Options"
        )
        if backend:
            self.details["backend"] = backend


class GPUNotAvailableError(GPUError):
    """Raised when GPU is requested but not available"""
    
    def __init__(
        self,
        reason: str = "No compatible GPU found"
    ):
        super().__init__(
            reason,
            suggestion="Install CUDA/cupy (NVIDIA) or PyOpenCL (Metal/OpenCL)"
        )


# ============================================================================
# Material Errors
# ============================================================================

class MaterialError(CastSimError):
    """Raised for material-related errors"""
    
    def __init__(
        self,
        message: str,
        material_name: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        details = {}
        if material_name:
            details["material_name"] = material_name
            
        super().__init__(
            message,
            details=details,
            suggestion=suggestion,
            error_code="MAT001"
        )
        self.material_name = material_name


class UnknownMaterialError(MaterialError):
    """Raised when material is not found in database"""
    
    def __init__(
        self,
        material_name: str,
        available_materials: Optional[List[str]] = None
    ):
        avail_str = ""
        if available_materials:
            avail_str = f". Available: {', '.join(available_materials)}"
            
        super().__init__(
            f"Unknown material: '{material_name}'{avail_str}",
            material_name=material_name,
            suggestion="Use one of the available materials or add a custom material definition"
        )
        self.available_materials = available_materials


class InvalidMaterialPropertyError(MaterialError):
    """Raised when material property is invalid"""
    
    def __init__(
        self,
        material_name: str,
        property_name: str,
        value: Any,
        expected: str
    ):
        super().__init__(
            f"Invalid property '{property_name}' for material '{material_name}': {value}",
            material_name=material_name,
            suggestion=f"Expected: {expected}"
        )
        self.details["property_name"] = property_name
        self.details["value"] = value


# ============================================================================
# Plugin Errors
# ============================================================================

class PluginError(CastSimError):
    """Base class for plugin-related errors"""
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        plugin_type: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        details = {}
        if plugin_name:
            details["plugin_name"] = plugin_name
        if plugin_type:
            details["plugin_type"] = plugin_type
            
        super().__init__(
            message,
            details=details,
            suggestion=suggestion,
            error_code="PLG001"
        )


class PluginLoadError(PluginError):
    """Raised when plugin fails to load"""
    
    def __init__(
        self,
        plugin_name: str,
        reason: str
    ):
        super().__init__(
            f"Failed to load plugin '{plugin_name}': {reason}",
            plugin_name=plugin_name,
            suggestion="Check plugin is installed correctly and compatible with current version"
        )


class PluginInterfaceError(PluginError):
    """Raised when plugin doesn't implement required interface"""
    
    def __init__(
        self,
        plugin_name: str,
        missing_method: str
    ):
        super().__init__(
            f"Plugin '{plugin_name}' missing required method: {missing_method}",
            plugin_name=plugin_name,
            suggestion=f"Plugin must implement the '{missing_method}' method"
        )


# ============================================================================
# Report Errors
# ============================================================================

class ReportError(CastSimError):
    """Raised when report generation fails"""
    
    def __init__(
        self,
        message: str,
        report_type: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        details = {}
        if report_type:
            details["report_type"] = report_type
            
        super().__init__(
            message,
            details=details,
            suggestion=suggestion,
            error_code="RPT001"
        )


class ReportTemplateError(ReportError):
    """Raised when report template is invalid"""
    
    def __init__(
        self,
        template_name: str,
        reason: str
    ):
        super().__init__(
            f"Invalid report template '{template_name}': {reason}",
            suggestion="Check template syntax and required variables"
        )
        self.details["template_name"] = template_name


# ============================================================================
# Utility Functions
# ============================================================================

def format_exception_chain(exc: Exception) -> str:
    """
    Format exception chain for logging.
    
    Args:
        exc: Exception to format
        
    Returns:
        Formatted string with full exception chain
    """
    lines = []
    current = exc
    depth = 0
    
    while current is not None:
        indent = "  " * depth
        if isinstance(current, CastSimError):
            lines.append(f"{indent}{current.__class__.__name__}: {current.message}")
            if current.suggestion:
                lines.append(f"{indent}  💡 {current.suggestion}")
        else:
            lines.append(f"{indent}{current.__class__.__name__}: {str(current)}")
        
        current = current.__cause__
        depth += 1
    
    return "\n".join(lines)


def validate_positive(value: float, name: str) -> None:
    """Validate that value is positive, raise BoundsError if not"""
    if value <= 0:
        raise BoundsError(name, value, min_val=0.0)


def validate_range(value: float, name: str, min_val: float, max_val: float) -> None:
    """Validate that value is within range, raise BoundsError if not"""
    if value < min_val or value > max_val:
        raise BoundsError(name, value, min_val=min_val, max_val=max_val)


def validate_not_none(value: Any, name: str) -> None:
    """Validate that value is not None, raise MissingDataError if it is"""
    if value is None:
        raise MissingDataError(name)
