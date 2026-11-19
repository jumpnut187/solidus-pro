"""
Memory Limits and Validation System
Prevents RAM/swap exhaustion by enforcing hard voxel count limits
Based on the proven approach from the Mac-Cast Sim (Swift) implementation

CRITICAL: This module provides the safety guarantees to prevent system crashes.
"""

import psutil
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class MemorySeverity(Enum):
    """Severity level for memory warnings"""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MemoryValidationResult:
    """Result of memory validation check"""
    can_proceed: bool
    severity: MemorySeverity
    estimated_voxels: int
    max_allowed_voxels: int
    estimated_memory_mb: float
    available_memory_mb: float
    message: str
    suggestion: Optional[str] = None
    recommended_voxel_size: Optional[float] = None


# Memory footprint per voxel (bytes) - includes all arrays
# voxel_centers (12) + voxel_sizes (4) + voxel_volumes (4) + voxel_surface_areas (4) +
# voxel_moduli (4) + voxel_states (4) + voxel_neighbors (104) + surface_voxels (1) = 137 bytes
# Plus: simulation arrays (shrinkage, porosity, feeding, times) = 16 bytes
# Plus: Python/NumPy overhead = ~50 bytes
# KD-tree overhead = ~80 bytes per point
# TOTAL = ~280 bytes per voxel (conservative estimate)
BYTES_PER_VOXEL = 280

# Additional overhead for mesh processing (KD-trees, temp arrays, etc.)
PROCESSING_OVERHEAD_FACTOR = 1.5  # 50% extra for temporary allocations


def get_system_memory_info() -> Dict[str, float]:
    """Get current system memory status"""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'total_ram_gb': mem.total / (1024 ** 3),
        'available_ram_gb': mem.available / (1024 ** 3),
        'used_ram_gb': mem.used / (1024 ** 3),
        'ram_percent': mem.percent,
        'swap_total_gb': swap.total / (1024 ** 3),
        'swap_used_gb': swap.used / (1024 ** 3),
        'swap_percent': swap.percent,
    }


def get_max_voxels_for_system() -> int:
    """
    Calculate maximum safe voxel count based on system capabilities.
    
    Voxel limits scale with available RAM:
    - 8GB RAM: 500K voxels (baseline)
    - 16GB RAM: 1,000K voxels
    - 32GB RAM: 2,000K voxels
    - 64GB+ RAM: 3,000K voxels (capped for practical reasons)
    
    Apple Silicon gets a 20% bonus due to unified memory architecture.
    
    Returns:
        Maximum recommended voxel count
    """
    import platform
    
    mem_info = get_system_memory_info()
    available_gb = mem_info['available_ram_gb']
    total_gb = mem_info['total_ram_gb']
    
    # Detect chip type
    system = platform.system()
    is_apple_silicon = False
    chip_generation = 1
    
    if system == 'Darwin':
        import subprocess
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=2)
            cpu_brand = result.stdout.strip()
            if 'Apple' in cpu_brand:
                is_apple_silicon = True
                # Detect M1/M2/M3/M4 generation
                if 'M4' in cpu_brand:
                    chip_generation = 4
                elif 'M3' in cpu_brand:
                    chip_generation = 3
                elif 'M2' in cpu_brand:
                    chip_generation = 2
                else:
                    chip_generation = 1
        except:
            pass
    
    # Calculate max voxels based on available memory (use 40% of available RAM max)
    safe_bytes = available_gb * 0.40 * (1024 ** 3)
    memory_based_max = int(safe_bytes / (BYTES_PER_VOXEL * PROCESSING_OVERHEAD_FACTOR))
    
    # RAM-based scaling: 1M voxels per 16GB RAM as baseline
    # Using total_gb to determine tier, not available (which fluctuates)
    if total_gb >= 64:
        base_cap = 3_000_000  # 64GB+ systems
    elif total_gb >= 32:
        base_cap = 2_000_000  # 32GB systems
    elif total_gb >= 16:
        base_cap = 1_000_000  # 16GB systems
    elif total_gb > 8:
        base_cap = 750_000    # 12GB systems (some configs)
    else:
        base_cap = 500_000    # 8GB systems
    
    # Apple Silicon bonus (20% extra due to unified memory efficiency)
    if is_apple_silicon:
        silicon_bonus = 1.2
        # Extra bonus for newer generations with better memory bandwidth
        if chip_generation >= 3:  # M3, M4
            silicon_bonus = 1.3
        elif chip_generation >= 2:  # M2
            silicon_bonus = 1.25
        system_cap = int(base_cap * silicon_bonus)
    else:
        system_cap = base_cap
    
    # Use the more conservative of memory-based and system cap
    max_voxels = min(memory_based_max, system_cap)
    
    # Absolute minimum for meaningful simulation
    max_voxels = max(max_voxels, 10_000)
    
    return max_voxels


def get_hard_voxel_limit() -> int:
    """
    Get the absolute maximum voxel count that will trigger an error.
    
    Scales with system RAM:
    - 8GB: 1 million
    - 16GB: 2 million  
    - 32GB: 4 million
    - 64GB+: 5 million (practical cap)
    
    Returns:
        Hard limit (errors if exceeded)
    """
    mem_info = get_system_memory_info()
    total_gb = mem_info['total_ram_gb']
    
    if total_gb >= 64:
        return 5_000_000
    elif total_gb >= 32:
        return 4_000_000
    elif total_gb >= 16:
        return 2_000_000
    else:
        return 1_000_000


def estimate_voxel_count(
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    voxel_size: float,
    fill_ratio: float = 0.5
) -> int:
    """
    Estimate the number of voxels that will be created.
    
    Args:
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        voxel_size: Base voxel size in inches
        fill_ratio: Estimated fill ratio (typically 0.3-0.7 for castings)
    
    Returns:
        Estimated voxel count
    """
    x_range = bounds[0][1] - bounds[0][0]
    y_range = bounds[1][1] - bounds[1][0]
    z_range = bounds[2][1] - bounds[2][0]
    
    # Add padding (matching voxel_mesh.py logic)
    padding = voxel_size * 0.5
    x_range += 2 * padding
    y_range += 2 * padding
    z_range += 2 * padding
    
    nx = int((x_range / voxel_size) + 1)
    ny = int((y_range / voxel_size) + 1)
    nz = int((z_range / voxel_size) + 1)
    
    total_grid = nx * ny * nz
    
    # Estimate actual filled voxels
    estimated_filled = int(total_grid * fill_ratio)
    
    return estimated_filled


def estimate_memory_usage_mb(voxel_count: int) -> float:
    """
    Estimate total memory usage in MB for given voxel count.
    
    Args:
        voxel_count: Number of voxels
    
    Returns:
        Estimated memory usage in MB
    """
    base_bytes = voxel_count * BYTES_PER_VOXEL
    with_overhead = base_bytes * PROCESSING_OVERHEAD_FACTOR
    return with_overhead / (1024 ** 2)


def calculate_safe_voxel_size(
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    max_voxels: Optional[int] = None,
    fill_ratio: float = 0.5
) -> float:
    """
    Calculate the minimum voxel size needed to stay under memory limits.
    
    Args:
        bounds: Mesh bounds
        max_voxels: Target maximum voxels (uses system max if not specified)
        fill_ratio: Estimated fill ratio
    
    Returns:
        Recommended minimum voxel size in inches
    """
    if max_voxels is None:
        max_voxels = get_max_voxels_for_system()
    
    x_range = bounds[0][1] - bounds[0][0]
    y_range = bounds[1][1] - bounds[1][0]
    z_range = bounds[2][1] - bounds[2][0]
    
    # Total bounding box volume
    bbox_volume = x_range * y_range * z_range
    
    # Voxel volume needed: bbox_volume * fill_ratio / max_voxels
    voxel_volume = (bbox_volume * fill_ratio) / max_voxels
    
    # Voxel size is cube root of volume
    voxel_size = voxel_volume ** (1/3)
    
    # Apply reasonable limits
    voxel_size = max(0.05, min(voxel_size, 1.0))
    
    return voxel_size


def validate_mesh_parameters(
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    voxel_size: float,
    fill_ratio: float = 0.5
) -> MemoryValidationResult:
    """
    Validate that mesh parameters won't exhaust system memory.
    
    This is the main validation function that should be called BEFORE
    starting mesh generation.
    
    Args:
        bounds: Mesh bounds
        voxel_size: Proposed voxel size
        fill_ratio: Estimated fill ratio
    
    Returns:
        MemoryValidationResult with detailed information
    """
    mem_info = get_system_memory_info()
    available_mb = mem_info['available_ram_gb'] * 1024
    
    estimated_voxels = estimate_voxel_count(bounds, voxel_size, fill_ratio)
    estimated_memory_mb = estimate_memory_usage_mb(estimated_voxels)
    
    max_voxels = get_max_voxels_for_system()
    hard_limit = get_hard_voxel_limit()
    
    # Check severity levels
    if estimated_voxels > hard_limit:
        # CRITICAL: Will definitely crash
        safe_size = calculate_safe_voxel_size(bounds, max_voxels, fill_ratio)
        return MemoryValidationResult(
            can_proceed=False,
            severity=MemorySeverity.CRITICAL,
            estimated_voxels=estimated_voxels,
            max_allowed_voxels=hard_limit,
            estimated_memory_mb=estimated_memory_mb,
            available_memory_mb=available_mb,
            message=f"CRITICAL: Estimated {estimated_voxels:,} voxels exceeds hard limit of {hard_limit:,}. "
                   f"This WILL crash your system.",
            suggestion=f"Increase voxel size to at least {safe_size:.3f}\" to reduce voxel count.",
            recommended_voxel_size=safe_size
        )
    
    if estimated_voxels > max_voxels:
        # ERROR: Likely to cause problems
        safe_size = calculate_safe_voxel_size(bounds, max_voxels, fill_ratio)
        return MemoryValidationResult(
            can_proceed=False,
            severity=MemorySeverity.ERROR,
            estimated_voxels=estimated_voxels,
            max_allowed_voxels=max_voxels,
            estimated_memory_mb=estimated_memory_mb,
            available_memory_mb=available_mb,
            message=f"ERROR: Estimated {estimated_voxels:,} voxels exceeds safe limit of {max_voxels:,} "
                   f"for your system ({mem_info['total_ram_gb']:.1f}GB RAM).",
            suggestion=f"Increase voxel size to at least {safe_size:.3f}\" for safe operation.",
            recommended_voxel_size=safe_size
        )
    
    if estimated_memory_mb > available_mb * 0.7:
        # WARNING: Approaching limits
        safe_size = calculate_safe_voxel_size(bounds, int(max_voxels * 0.7), fill_ratio)
        return MemoryValidationResult(
            can_proceed=True,
            severity=MemorySeverity.WARNING,
            estimated_voxels=estimated_voxels,
            max_allowed_voxels=max_voxels,
            estimated_memory_mb=estimated_memory_mb,
            available_memory_mb=available_mb,
            message=f"WARNING: Estimated memory usage ({estimated_memory_mb:.0f}MB) is high. "
                   f"System may slow down.",
            suggestion=f"Consider increasing voxel size to {safe_size:.3f}\" for better performance.",
            recommended_voxel_size=safe_size
        )
    
    if estimated_voxels > max_voxels * 0.7:
        # WARNING: Getting close to limits
        return MemoryValidationResult(
            can_proceed=True,
            severity=MemorySeverity.WARNING,
            estimated_voxels=estimated_voxels,
            max_allowed_voxels=max_voxels,
            estimated_memory_mb=estimated_memory_mb,
            available_memory_mb=available_mb,
            message=f"Large mesh: {estimated_voxels:,} voxels. Simulation may take several minutes.",
            suggestion=None,
            recommended_voxel_size=None
        )
    
    # OK: Within safe limits
    return MemoryValidationResult(
        can_proceed=True,
        severity=MemorySeverity.OK,
        estimated_voxels=estimated_voxels,
        max_allowed_voxels=max_voxels,
        estimated_memory_mb=estimated_memory_mb,
        available_memory_mb=available_mb,
        message=f"Mesh size OK: ~{estimated_voxels:,} voxels, ~{estimated_memory_mb:.0f}MB memory",
        suggestion=None,
        recommended_voxel_size=None
    )


def auto_adjust_voxel_size(
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    requested_voxel_size: float,
    fill_ratio: float = 0.5
) -> Tuple[float, bool, str]:
    """
    Automatically adjust voxel size if it would exceed memory limits.
    
    This is the "auto-safe" function that guarantees the returned voxel size
    will not crash the system.
    
    Args:
        bounds: Mesh bounds
        requested_voxel_size: User-requested voxel size
        fill_ratio: Estimated fill ratio
    
    Returns:
        Tuple of (adjusted_voxel_size, was_adjusted, message)
    """
    validation = validate_mesh_parameters(bounds, requested_voxel_size, fill_ratio)
    
    if validation.can_proceed and validation.severity == MemorySeverity.OK:
        return (requested_voxel_size, False, "")
    
    if validation.recommended_voxel_size and validation.recommended_voxel_size > requested_voxel_size:
        adjusted_size = validation.recommended_voxel_size
        message = (f"Voxel size automatically increased from {requested_voxel_size:.3f}\" "
                  f"to {adjusted_size:.3f}\" to prevent memory exhaustion. "
                  f"Reduced from ~{validation.estimated_voxels:,} to "
                  f"~{estimate_voxel_count(bounds, adjusted_size, fill_ratio):,} voxels.")
        return (adjusted_size, True, message)
    
    return (requested_voxel_size, False, validation.message)


class MemoryMonitor:
    """
    Real-time memory monitoring during mesh generation.
    
    Can be used to abort operations if memory pressure becomes critical.
    """
    
    def __init__(self, abort_threshold_percent: float = 90.0):
        """
        Args:
            abort_threshold_percent: Abort if RAM usage exceeds this percentage
        """
        self.abort_threshold = abort_threshold_percent
        self.initial_memory = None
        self.peak_memory = 0
        self.should_abort = False
        
    def start(self):
        """Start monitoring"""
        mem = psutil.virtual_memory()
        self.initial_memory = mem.used
        self.peak_memory = mem.used
        self.should_abort = False
        
    def check(self) -> Tuple[bool, str]:
        """
        Check current memory status.
        
        Returns:
            Tuple of (should_continue, status_message)
        """
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        self.peak_memory = max(self.peak_memory, mem.used)
        
        current_mb = mem.used / (1024 ** 2)
        used_percent = mem.percent
        swap_percent = swap.percent
        
        status = f"RAM: {used_percent:.0f}% | Swap: {swap_percent:.0f}%"
        
        # Check for critical conditions
        if used_percent > self.abort_threshold:
            self.should_abort = True
            return (False, f"ABORT: RAM usage at {used_percent:.0f}% exceeds {self.abort_threshold:.0f}% threshold")
        
        if swap_percent > 50:
            # Swap usage is a warning sign
            return (True, f"WARNING: Heavy swap usage ({swap_percent:.0f}%). Performance degraded.")
        
        return (True, status)
    
    def get_used_mb(self) -> float:
        """
        System memory currently in use, in MB.

        Paired with :meth:`get_limit_mb` for ``logger.log_memory_warning``.
        Both existed only as call sites in ``core/voxel_mesh.py``'s abort path
        until that path first became reachable, at which point it raised
        ``AttributeError`` instead of reporting the memory problem it had
        correctly detected.
        """
        return psutil.virtual_memory().used / (1024 ** 2)

    def get_limit_mb(self) -> float:
        """Memory ceiling this monitor aborts at, in MB."""
        total_mb = psutil.virtual_memory().total / (1024 ** 2)
        return total_mb * (self.abort_threshold / 100.0)

    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        mem = psutil.virtual_memory()
        
        return {
            'current_used_mb': mem.used / (1024 ** 2),
            'peak_used_mb': self.peak_memory / (1024 ** 2),
            'initial_used_mb': (self.initial_memory or 0) / (1024 ** 2),
            'delta_mb': (mem.used - (self.initial_memory or 0)) / (1024 ** 2),
            'used_percent': mem.percent,
        }


def print_memory_limits_info():
    """Print current system memory limits for debugging"""
    mem_info = get_system_memory_info()
    max_voxels = get_max_voxels_for_system()
    hard_limit = get_hard_voxel_limit()
    
    print("\n" + "=" * 60)
    print("Memory Limits Configuration")
    print("=" * 60)
    print(f"System RAM: {mem_info['total_ram_gb']:.1f} GB")
    print(f"Available RAM: {mem_info['available_ram_gb']:.1f} GB ({100-mem_info['ram_percent']:.0f}% free)")
    print(f"Swap Usage: {mem_info['swap_percent']:.0f}%")
    print(f"\nVoxel Limits:")
    print(f"  Safe Maximum: {max_voxels:,} voxels")
    print(f"  Hard Limit: {hard_limit:,} voxels")
    print(f"  Memory per Voxel: ~{BYTES_PER_VOXEL} bytes")
    print(f"\nEstimated Capacity:")
    print(f"  At Safe Max: ~{estimate_memory_usage_mb(max_voxels):.0f} MB")
    print(f"  At Hard Limit: ~{estimate_memory_usage_mb(hard_limit):.0f} MB")
    print("=" * 60 + "\n")


@dataclass
class VoxelSizeRecommendation:
    """Recommendation for voxel size based on model dimensions"""
    model_dimensions: Tuple[float, float, float]  # x, y, z in inches
    recommended_voxel_size: float
    estimated_voxels: int
    estimated_memory_mb: float
    quality_level: str  # "high", "medium", "low"
    message: str
    alternatives: list  # List of (voxel_size, voxels, memory_mb, quality, est_time_s) tuples
    estimated_time_seconds: float = 0.0  # Estimated total simulation time


def _estimate_time_for_voxels(voxel_count: int, physical_cores: int = 8) -> float:
    """Estimate simulation time based on voxel count.
    
    Derived from analysis of mesh_runs.csv data:
    - Mesh generation: ~0.00005s per voxel (varies with complexity)
    - Simulation: ~0.00004s per voxel
    - Core scaling: roughly linear up to 8 cores
    """
    core_factor = max(1.0, 8.0 / max(physical_cores, 1))
    mesh_time = voxel_count * 0.00005 * core_factor
    sim_time = voxel_count * 0.00004 * core_factor
    return mesh_time + sim_time


def get_voxel_size_recommendations(
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    fill_ratio: float = 0.4
) -> VoxelSizeRecommendation:
    """
    Get smart voxel size recommendations based on model dimensions and system memory.
    
    Args:
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max)) in inches
        fill_ratio: Estimated fill ratio for the model (default 0.4 for typical castings)
    
    Returns:
        VoxelSizeRecommendation with recommended size and alternatives
    """
    x_range = bounds[0][1] - bounds[0][0]
    y_range = bounds[1][1] - bounds[1][0]
    z_range = bounds[2][1] - bounds[2][0]
    
    dimensions = (x_range, y_range, z_range)
    max_dim = max(dimensions)
    min_dim = min(d for d in dimensions if d > 0) if max(dimensions) > 0 else 1.0
    aspect_ratio = max_dim / min_dim if min_dim > 0 else 1.0
    
    # Get system limits
    max_voxels = get_max_voxels_for_system()
    mem_info = get_system_memory_info()
    
    # Estimate physical cores for time calculations
    import platform
    try:
        if platform.system() == 'Darwin':
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'hw.physicalcpu'], 
                                  capture_output=True, text=True, timeout=2)
            physical_cores = int(result.stdout.strip())
        else:
            import os
            physical_cores = os.cpu_count() or 4
    except:
        physical_cores = 4
    
    # Calculate alternatives for different quality levels
    # Extended list with finer granularity for better optimization
    alternatives = []
    
    # Expanded voxel size options with finer steps in common range
    voxel_sizes = [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4, 0.5]
    
    for voxel_size in voxel_sizes:
        est_voxels = estimate_voxel_count(bounds, voxel_size, fill_ratio)
        est_memory = estimate_memory_usage_mb(est_voxels)
        est_time = _estimate_time_for_voxels(est_voxels, physical_cores)
        
        # Determine quality level based on voxels per inch and min dimension coverage
        voxels_per_inch = 1.0 / voxel_size
        min_dim_voxels = min_dim / voxel_size  # How many voxels across thinnest dimension
        
        if voxels_per_inch >= 10 and min_dim_voxels >= 10:  # 0.1" or smaller, good thin coverage
            quality = "high"
        elif voxels_per_inch >= 6.5 and min_dim_voxels >= 6:  # 0.15" or smaller
            quality = "high"
        elif voxels_per_inch >= 5 and min_dim_voxels >= 4:  # 0.2" or smaller
            quality = "medium"
        elif voxels_per_inch >= 3.3:  # 0.3" or smaller
            quality = "medium"
        else:
            quality = "low"
        
        # Downgrade quality if thin dimension isn't well resolved
        if min_dim_voxels < 5 and quality == "high":
            quality = "medium"
        if min_dim_voxels < 3 and quality == "medium":
            quality = "low"
        
        # Check if this fits in memory
        if est_voxels <= max_voxels:
            alternatives.append((voxel_size, est_voxels, est_memory, quality, est_time))
    
    if not alternatives:
        # Nothing fits, calculate minimum safe size
        safe_size = calculate_safe_voxel_size(bounds, max_voxels, fill_ratio)
        safe_size = max(0.5, safe_size)  # At least 0.5"
        est_voxels = estimate_voxel_count(bounds, safe_size, fill_ratio)
        est_memory = estimate_memory_usage_mb(est_voxels)
        est_time = _estimate_time_for_voxels(est_voxels, physical_cores)
        alternatives.append((safe_size, est_voxels, est_memory, "minimal", est_time))
    
    # Choose recommendation based on model characteristics
    # Goal: balance quality, performance, and memory usage
    
    # Determine target based on part size and geometry
    if max_dim > 30:  # Very large model (> 30")
        target_voxels = min(150_000, int(max_voxels * 0.4))
        preferred_quality = "medium"
    elif max_dim > 20:  # Large model (20-30")
        target_voxels = min(200_000, int(max_voxels * 0.5))
        preferred_quality = "medium"
    elif max_dim > 10:  # Medium model (10-20")
        target_voxels = min(250_000, int(max_voxels * 0.6))
        preferred_quality = "medium"
    elif max_dim > 5:  # Small-medium model (5-10")
        target_voxels = min(200_000, int(max_voxels * 0.6))
        preferred_quality = "high"
    else:  # Small model (< 5")
        target_voxels = min(100_000, int(max_voxels * 0.5))
        preferred_quality = "high"
    
    # Adjust for high aspect ratio (long thin parts need more voxels)
    if aspect_ratio > 4.0:
        target_voxels = int(target_voxels * 1.3)
    elif aspect_ratio > 2.5:
        target_voxels = int(target_voxels * 1.15)
    
    # Cap at system maximum
    target_voxels = min(target_voxels, max_voxels)
    
    # Find best match: prefer quality first, then closest to target voxels
    best_match = None
    best_score = float('inf')
    
    for alt in alternatives:
        voxel_size, est_voxels, est_memory, quality, est_time = alt
        
        # Skip if exceeds target by too much
        if est_voxels > target_voxels * 1.5:
            continue
        
        # Score: lower is better
        # Prefer matching quality, then minimize voxel count difference
        quality_penalty = 0 if quality == preferred_quality else (1 if quality == "high" else 2)
        voxel_diff = abs(est_voxels - target_voxels) / target_voxels
        time_penalty = est_time / 60.0 if est_time > 30 else 0  # Penalize long simulations
        
        score = quality_penalty * 10 + voxel_diff + time_penalty * 0.5
        
        if score < best_score:
            best_score = score
            best_match = alt
    
    if best_match is None:
        # Fallback: use the option with voxels closest to target
        best_match = min(alternatives, key=lambda x: abs(x[1] - target_voxels))
    
    rec_size, rec_voxels, rec_memory, rec_quality, rec_time = best_match
    
    # Generate message with time estimate
    time_str = f"{rec_time:.0f}s" if rec_time < 60 else f"{rec_time/60:.1f}min"
    
    if rec_voxels > max_voxels * 0.8:
        message = f"⚠️ Large model. Using {rec_size}\" voxels (~{time_str})."
    elif rec_quality == "high":
        message = f"✓ {rec_size}\" voxels for high detail (~{rec_voxels//1000}k voxels, {time_str})"
    elif rec_quality == "medium":
        message = f"✓ {rec_size}\" voxels balanced (~{rec_voxels//1000}k voxels, {time_str})"
    else:
        message = f"Using {rec_size}\" voxels to fit memory (~{time_str})."
    
    return VoxelSizeRecommendation(
        model_dimensions=dimensions,
        recommended_voxel_size=rec_size,
        estimated_voxels=rec_voxels,
        estimated_memory_mb=rec_memory,
        quality_level=rec_quality,
        message=message,
        alternatives=[(a[0], a[1], a[2], a[3]) for a in alternatives],  # Strip time for backward compat
        estimated_time_seconds=rec_time
    )


def print_voxel_recommendations(bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]):
    """Print voxel size recommendations for given bounds"""
    rec = get_voxel_size_recommendations(bounds)
    
    print("\n" + "=" * 70)
    print("Voxel Size Recommendations")
    print("=" * 70)
    print(f"Model Dimensions: {rec.model_dimensions[0]:.1f}\" x {rec.model_dimensions[1]:.1f}\" x {rec.model_dimensions[2]:.1f}\"")
    print(f"\n{rec.message}")
    print(f"  Estimated voxels: {rec.estimated_voxels:,}")
    print(f"  Estimated memory: {rec.estimated_memory_mb:.0f} MB")
    
    print(f"\nAll Options:")
    print(f"  {'Size':>6} {'Voxels':>12} {'Memory':>10} {'Quality':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10}")
    for voxel_size, est_voxels, est_memory, quality in rec.alternatives:
        marker = " ◀" if voxel_size == rec.recommended_voxel_size else ""
        print(f"  {voxel_size:>5.2f}\" {est_voxels:>11,} {est_memory:>9.0f}MB {quality:>10}{marker}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test the memory limits system
    print_memory_limits_info()
    
    # Test validation with sample bounds
    sample_bounds = ((0, 10), (0, 8), (0, 6))  # 10" x 8" x 6" part
    
    for voxel_size in [0.05, 0.1, 0.15, 0.2, 0.25]:
        result = validate_mesh_parameters(sample_bounds, voxel_size)
        print(f"\nVoxel Size: {voxel_size}\"")
        print(f"  Severity: {result.severity.value}")
        print(f"  Estimated Voxels: {result.estimated_voxels:,}")
        print(f"  Can Proceed: {result.can_proceed}")
        print(f"  Message: {result.message}")
