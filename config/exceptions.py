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


class CADLoadError(InvalidMeshError):
    """Raised when a CAD file (STL, STEP, OBJ, ...) cannot be loaded.

    Covers both file-format errors and missing-optional-dependency
    errors for backend-specific formats (e.g. STEP requires the
    optional ``cascadio`` package). When raised from the part loader,
    the message directs the user to the right install command.
    """

    def __init__(
        self,
        reason: str,
        filepath: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        # Skip InvalidMeshError.__init__'s "Invalid mesh geometry: "
        # prefix — CADLoadError has its own domain (file loading), not
        # mesh-repair semantics. Bypass the parent constructor and
        # populate MeshError's state directly.
        self.message = f"CAD load failed: {reason}"
        self.details: dict = {"filepath": filepath} if filepath else {}
        self.suggestion = suggestion or (
            "Verify the file is a valid CAD mesh, or install missing "
            "optional dependencies (e.g. `pip install cascadio` for STEP files)."
        )
        self.original_error: Optional[BaseException] = None


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


# ============================================================================
# Utility Functions
# ============================================================================
