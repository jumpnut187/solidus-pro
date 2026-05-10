"""Core package initialization"""

from core.memory_limits import (
    get_max_voxels_for_system,
    get_hard_voxel_limit,
    validate_mesh_parameters,
    auto_adjust_voxel_size,
    MemoryMonitor,
    MemorySeverity,
    print_memory_limits_info,
)

from core.part_manager import (
    PartType,
    Part,
    PartManager,
    PART_TYPE_COLORS,
    PART_TYPE_KEYWORDS,
    infer_part_type_from_filename,
)

# Performance optimization modules (optional imports)
try:
    from core.gpu_compute import (
        GPUComputeManager,
        GPUBackend,
        gpu_compute,
        get_gpu_compute,
        compute_distances_accelerated,
        update_solidification_accelerated,
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from core.octree import (
        Octree,
        OctreeNode,
        SpatialIndex,
    )
    OCTREE_AVAILABLE = True
except ImportError:
    OCTREE_AVAILABLE = False

try:
    from core.incremental_solidification import (
        IncrementalSolidificationEngine,
        SolidificationFront,
        create_solidification_front,
    )
    INCREMENTAL_AVAILABLE = True
except ImportError:
    INCREMENTAL_AVAILABLE = False

# Physics simulation modules (new in v2.0)
try:
    from core.thermal_solver import (
        ThermalSolver,
        ThermalSolverConfig,
        ThermalProperties,
        create_thermal_properties_from_alloy,
        create_thermal_solver,
    )
    THERMAL_SOLVER_AVAILABLE = True
except ImportError:
    THERMAL_SOLVER_AVAILABLE = False

try:
    from core.mold_interface import (
        MoldInterfaceModel,
        MoldType,
        MoldProperties,
        get_mold_properties,
        get_all_mold_types,
        create_mold_interface,
    )
    MOLD_INTERFACE_AVAILABLE = True
except ImportError:
    MOLD_INTERFACE_AVAILABLE = False

try:
    from core.convection import (
        ConvectionModel,
        LiquidMetalProperties,
        create_liquid_properties_from_alloy,
        create_convection_model,
    )
    CONVECTION_AVAILABLE = True
except ImportError:
    CONVECTION_AVAILABLE = False

try:
    from core.thermal_simulation import (
        ThermalSimulationEngine,
        ThermalSimConfig,
        SimulationMode,
        SimulationResults,
        create_thermal_simulation,
    )
    THERMAL_SIMULATION_AVAILABLE = True
except ImportError:
    THERMAL_SIMULATION_AVAILABLE = False

__all__ = [
    'get_max_voxels_for_system',
    'get_hard_voxel_limit',
    'validate_mesh_parameters',
    'auto_adjust_voxel_size',
    'MemoryMonitor',
    'MemorySeverity',
    'print_memory_limits_info',
    'PartType',
    'Part',
    'PartManager',
    'PART_TYPE_COLORS',
    'PART_TYPE_KEYWORDS',
    'infer_part_type_from_filename',
    # Performance modules
    'GPU_AVAILABLE',
    'OCTREE_AVAILABLE',
    'INCREMENTAL_AVAILABLE',
    # Physics simulation modules
    'THERMAL_SOLVER_AVAILABLE',
    'MOLD_INTERFACE_AVAILABLE',
    'CONVECTION_AVAILABLE',
    'THERMAL_SIMULATION_AVAILABLE',
]
