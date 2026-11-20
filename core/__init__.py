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
    'GPU_AVAILABLE',
    'OCTREE_AVAILABLE',
    'INCREMENTAL_AVAILABLE',
]
