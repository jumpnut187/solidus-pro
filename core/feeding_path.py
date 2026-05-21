"""
Feeding Path Analysis Module
Traces liquid metal feeding paths from risers to potential shrinkage zones.

Key Concepts:
    - Feeding path: Route through liquid/mushy zones from riser to voxel
    - Path quality: Based on permeability, distance, and fraction solid
    - Blocked path: When coherency reached (no liquid connectivity)

Physics:
    A voxel can be fed if:
    1. There exists a connected path of liquid voxels to a riser
    2. The path permeability is sufficient (Darcy's Law)
    3. Ferrostatic pressure > pressure drop along path

MAGMASOFT-style enhancement:
    - Niyama alone doesn't account for feed path availability
    - A voxel with acceptable Niyama but blocked path = DEFECT
    - A voxel with low Niyama but good feed path = OK

Usage:
    analyzer = FeedingPathAnalyzer(mesh, alloy)
    analyzer.analyze(riser_indices, fraction_solid)
    
    # Get feeding quality map
    quality = analyzer.feeding_quality  # 0-1 array
    blocked = analyzer.blocked_voxels   # Boolean array

Author: GitHub Copilot
Date: December 2025
"""

import numpy as np
from time import perf_counter

from numba import jit, prange
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import deque

# Import alloy properties for permeability calculations
try:
    from core.materials import AlloyProperties
except ImportError:
    AlloyProperties = None


@dataclass
class FeedingPathResult:
    """Results from feeding path analysis"""
    # Per-voxel results
    feeding_quality: np.ndarray       # 0-1 quality score (1=excellent, 0=blocked)
    path_length: np.ndarray           # Distance to nearest feeder (inches)
    path_permeability: np.ndarray     # Effective permeability along path
    feeding_source: np.ndarray        # Index of feeding riser (-1 if unfed)
    blocked_voxels: np.ndarray        # Boolean mask of voxels with blocked feed
    history_open_fraction: np.ndarray # Weighted fraction of pre-freeze history with an open path
    latest_open_time: np.ndarray      # Latest time an open path existed before freeze-off
    
    # Statistics
    fed_fraction: float               # Fraction of voxels with feeding path
    avg_path_length: float            # Average feeding distance
    max_path_length: float            # Maximum feeding distance
    n_blocked: int                    # Number of blocked voxels


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_path_permeability_batch(
    fraction_solid: np.ndarray,
    critical_fraction: float,
    dendrite_spacing_um: float
) -> np.ndarray:
    """
    Compute interdendritic permeability for all voxels using Kozeny-Carman.
    
    K = (λ² * (1 - fs)³) / (180 * fs²)
    
    Args:
        fraction_solid: Fraction solid at each voxel
        critical_fraction: Coherency point fraction solid
        dendrite_spacing_um: Secondary dendrite arm spacing in micrometers
        
    Returns:
        Permeability array (m²)
    """
    n_voxels = len(fraction_solid)
    permeability = np.zeros(n_voxels, dtype=np.float64)
    
    lambda_m = dendrite_spacing_um * 1e-6  # Convert μm to meters
    lambda_sq = lambda_m * lambda_m
    
    for i in prange(n_voxels):
        fs = fraction_solid[i]
        
        if fs <= 0.01:
            # Fully liquid - very high permeability
            permeability[i] = 1e-8  # Cap at reasonable value
        elif fs >= 0.99:
            # Fully solid - no permeability
            permeability[i] = 0.0
        else:
            fl = 1.0 - fs
            
            # Kozeny-Carman equation
            if fs > 0.01:
                K = (lambda_sq * (fl ** 3)) / (180.0 * (fs ** 2))
            else:
                K = 1e-8
            
            # Apply exponential decay above coherency point
            if fs >= critical_fraction:
                overshoot = (fs - critical_fraction) / (1.0 - critical_fraction)
                decay = np.exp(-10.0 * overshoot)
                K *= decay
            
            permeability[i] = min(K, 1e-8)  # Cap at reasonable max
    
    return permeability


@jit(nopython=True, cache=True)
def trace_single_path(
    start_idx: int,
    riser_mask: np.ndarray,
    neighbors: np.ndarray,
    permeability: np.ndarray,
    voxel_centers: np.ndarray,
    permeability_threshold: float
) -> Tuple[float, float, int]:
    """
    Trace feeding path from a single voxel to nearest riser using BFS.
    
    Returns:
        Tuple of (path_length, min_permeability_along_path, source_riser_index)
        Returns (-1, 0, -1) if no path found (blocked)
    """
    # Quick check if this is a riser voxel
    if riser_mask[start_idx]:
        return (0.0, 1.0, start_idx)
    
    n_voxels = len(riser_mask)
    max_neighbors = neighbors.shape[1]
    
    # BFS with permeability check
    visited = np.zeros(n_voxels, dtype=np.bool_)
    visited[start_idx] = True
    
    # Simple BFS using arrays (no deque in numba)
    queue = np.zeros(n_voxels, dtype=np.int32)
    distances = np.zeros(n_voxels, dtype=np.float64)
    min_perms = np.zeros(n_voxels, dtype=np.float64)
    
    queue[0] = start_idx
    distances[0] = 0.0
    min_perms[0] = permeability[start_idx]
    queue_start = 0
    queue_end = 1
    
    while queue_start < queue_end:
        current = queue[queue_start]
        current_dist = distances[queue_start]
        current_min_perm = min_perms[queue_start]
        queue_start += 1
        
        # Check if we reached a riser
        if riser_mask[current]:
            return (current_dist, current_min_perm, current)
        
        # Explore neighbors
        for j in range(max_neighbors):
            neighbor = neighbors[current, j]
            if neighbor < 0:
                continue
            if visited[neighbor]:
                continue
            
            # Check if path is blocked (permeability too low)
            neighbor_perm = permeability[neighbor]
            if neighbor_perm < permeability_threshold:
                continue
            
            visited[neighbor] = True
            
            # Calculate distance
            dx = voxel_centers[neighbor, 0] - voxel_centers[current, 0]
            dy = voxel_centers[neighbor, 1] - voxel_centers[current, 1]
            dz = voxel_centers[neighbor, 2] - voxel_centers[current, 2]
            step_dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            queue[queue_end] = neighbor
            distances[queue_end] = current_dist + step_dist
            min_perms[queue_end] = min(current_min_perm, neighbor_perm)
            queue_end += 1
    
    # No path found
    return (-1.0, 0.0, -1)


@jit(nopython=True, parallel=True, cache=True)
def trace_paths_batch(
    riser_mask: np.ndarray,
    fraction_solid: np.ndarray,
    neighbors: np.ndarray,
    permeability: np.ndarray,
    voxel_centers: np.ndarray,
    permeability_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_voxels = len(fraction_solid)
    path_length = np.full(n_voxels, -1.0, dtype=np.float32)
    path_permeability = np.zeros(n_voxels, dtype=np.float64)
    feeding_source = np.full(n_voxels, -1, dtype=np.int32)

    for i in prange(n_voxels):
        if riser_mask[i]:
            path_length[i] = 0.0
            path_permeability[i] = 1.0
            feeding_source[i] = i
            continue

        if fraction_solid[i] >= 0.99:
            continue

        if permeability[i] < permeability_threshold:
            continue

        dist, min_perm, source = trace_single_path(
            i,
            riser_mask,
            neighbors,
            permeability,
            voxel_centers,
            permeability_threshold,
        )

        if dist >= 0.0:
            path_length[i] = dist
            path_permeability[i] = min_perm
            feeding_source[i] = source

    return path_length, path_permeability, feeding_source


@jit(nopython=True, cache=True)
def trace_paths_from_risers_batch(
    riser_voxel_indices: np.ndarray,
    fraction_solid: np.ndarray,
    neighbors: np.ndarray,
    permeability: np.ndarray,
    voxel_centers: np.ndarray,
    permeability_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_voxels = len(fraction_solid)
    path_length = np.full(n_voxels, -1.0, dtype=np.float32)
    path_permeability = np.zeros(n_voxels, dtype=np.float64)
    feeding_source = np.full(n_voxels, -1, dtype=np.int32)

    sorted_risers = np.sort(riser_voxel_indices)
    for riser_idx in sorted_risers:
        path_length[riser_idx] = 0.0
        path_permeability[riser_idx] = 1.0
        feeding_source[riser_idx] = riser_idx

    visited = np.zeros(n_voxels, dtype=np.bool_)
    frontier = np.zeros(n_voxels, dtype=np.int64)
    frontier_size = 0

    for riser_idx in sorted_risers:
        frontier[frontier_size] = riser_idx
        frontier_size += 1
        visited[riser_idx] = True

    max_neighbors = neighbors.shape[1]
    c_epsilon = 1e-6

    while frontier_size > 0:
        min_idx = 0
        min_dist = path_length[frontier[0]]
        for i in range(1, frontier_size):
            if path_length[frontier[i]] < min_dist:
                min_dist = path_length[frontier[i]]
                min_idx = i

        current = frontier[min_idx]
        frontier[min_idx] = frontier[frontier_size - 1]
        frontier_size -= 1

        visited[current] = True

        current_dist = path_length[current]
        current_source = feeding_source[current]
        current_min_perm = path_permeability[current]

        if current_dist < 0.0:
            continue

        for j in range(max_neighbors):
            neighbor = neighbors[current, j]
            if neighbor < 0:
                continue
            if visited[neighbor]:
                continue

            if fraction_solid[neighbor] >= 0.99:
                continue

            neighbor_perm = permeability[neighbor]
            if neighbor_perm < permeability_threshold:
                continue

            dx = voxel_centers[neighbor, 0] - voxel_centers[current, 0]
            dy = voxel_centers[neighbor, 1] - voxel_centers[current, 1]
            dz = voxel_centers[neighbor, 2] - voxel_centers[current, 2]
            step_dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            new_dist = current_dist + step_dist
            new_min_perm = min(current_min_perm, neighbor_perm)

            old_dist = path_length[neighbor]

            if old_dist < 0.0:
                path_length[neighbor] = new_dist
                path_permeability[neighbor] = new_min_perm
                feeding_source[neighbor] = current_source
                frontier[frontier_size] = neighbor
                frontier_size += 1
            else:
                update = new_dist < old_dist - c_epsilon
                if not update:
                    tie_condition = abs(new_dist - old_dist) <= c_epsilon
                    source_condition = current_source < feeding_source[neighbor]
                    update = tie_condition and source_condition

                if update:
                    path_length[neighbor] = new_dist
                    path_permeability[neighbor] = new_min_perm
                    feeding_source[neighbor] = current_source
                    frontier[frontier_size] = neighbor
                    frontier_size += 1

    return path_length, path_permeability, feeding_source


class FeedingPathAnalyzer:
    """
    Analyzes feeding paths from risers to casting voxels.
    
    Key features:
    - BFS path tracing through liquid network
    - Permeability-based path quality
    - Identifies truly blocked regions vs low-Niyama but fed regions
    
    This is a MAGMASOFT-style enhancement that distinguishes:
    - Voxels with acceptable Niyama but blocked path → DEFECT
    - Voxels with low Niyama but good feed path → OK
    """
    
    def __init__(self, 
                 voxel_mesh,  # AdaptiveVoxelMesh
                 alloy: Optional['AlloyProperties'] = None):
        """
        Initialize feeding path analyzer.
        
        Args:
            voxel_mesh: AdaptiveVoxelMesh with neighbor connectivity
            alloy: AlloyProperties for material-specific parameters
        """
        self.mesh = voxel_mesh
        self.alloy = alloy
        
        # Material parameters (defaults for carbon steel)
        if alloy is not None:
            self.critical_fraction_solid = alloy.critical_fraction_solid
            self.dendrite_arm_spacing = alloy.dendrite_arm_spacing
        else:
            self.critical_fraction_solid = 0.52
            self.dendrite_arm_spacing = 80.0  # μm
        
        # Permeability threshold for path blockage
        # Below this, consider the path blocked
        self.permeability_threshold = 1e-16  # m²
        
        # Results
        self.result: Optional[FeedingPathResult] = None

    def _analyze_snapshot(
        self,
        riser_mask: np.ndarray,
        fraction_solid: np.ndarray,
        voxel_centers: np.ndarray,
        neighbors: np.ndarray,
    ) -> FeedingPathResult:
        """Analyze a single mush snapshot for riser connectivity."""
        n_voxels = self.mesh.get_voxel_count()

        permeability = compute_path_permeability_batch(
            np.ascontiguousarray(fraction_solid, dtype=np.float64),
            self.critical_fraction_solid,
            self.dendrite_arm_spacing,
        )

        riser_voxel_indices = np.where(riser_mask)[0]
        if len(riser_voxel_indices) > 0:
            path_length, path_permeability, feeding_source = trace_paths_from_risers_batch(
                np.ascontiguousarray(riser_voxel_indices, dtype=np.int64),
                np.ascontiguousarray(fraction_solid, dtype=np.float64),
                np.ascontiguousarray(neighbors, dtype=np.int32),
                permeability,
                np.ascontiguousarray(voxel_centers, dtype=np.float64),
                self.permeability_threshold,
            )
        else:
            path_length = np.full(n_voxels, -1.0, dtype=np.float32)
            path_permeability = np.zeros(n_voxels, dtype=np.float64)
            feeding_source = np.full(n_voxels, -1, dtype=np.int32)

        max_feeding_dist = 10.0  # inches - typical max feeding distance
        feeding_quality = np.zeros(n_voxels, dtype=np.float32)
        fed_mask = path_length >= 0.0
        if np.any(fed_mask):
            dist_factor = np.clip(1.0 - (path_length / max_feeding_dist), 0.0, 1.0)
            perm_factor = np.minimum(1.0, path_permeability / 1e-12)
            feeding_quality[fed_mask] = (
                dist_factor[fed_mask] * (0.5 + 0.5 * perm_factor[fed_mask])
            ).astype(np.float32)

        solid_mask = fraction_solid >= 0.99
        blocked_mask = (feeding_source < 0) & (~solid_mask)
        n_blocked = int(np.sum(solid_mask) + np.sum(blocked_mask))

        n_fed = int(np.sum(fed_mask))
        total_path_length = float(np.sum(path_length[fed_mask])) if n_fed > 0 else 0.0
        max_path_len = float(np.max(path_length[fed_mask])) if n_fed > 0 else 0.0

        blocked_voxels = (feeding_source < 0) & (fraction_solid < 0.99)
        open_mask = path_length >= 0.0
        latest_open_time = np.where(open_mask, 0.0, -1.0).astype(np.float32)

        return FeedingPathResult(
            feeding_quality=feeding_quality,
            path_length=path_length,
            path_permeability=path_permeability,
            feeding_source=feeding_source,
            blocked_voxels=blocked_voxels,
            history_open_fraction=open_mask.astype(np.float32),
            latest_open_time=latest_open_time,
            fed_fraction=n_fed / n_voxels if n_voxels > 0 else 0.0,
            avg_path_length=total_path_length / n_fed if n_fed > 0 else 0.0,
            max_path_length=max_path_len,
            n_blocked=n_blocked,
        )

    def _select_history_indices(
        self,
        history: List[np.ndarray],
        history_times: np.ndarray,
        max_reference_time: float,
        *,
        max_samples: int,
    ) -> np.ndarray:
        """Sample the pre-freeze timeline without flooding the analyzer."""
        candidate_indices = np.where(history_times <= max_reference_time + 1e-6)[0]
        if candidate_indices.size == 0:
            candidate_indices = np.array([0], dtype=np.int64)

        non_solid = np.array(
            [float(np.mean(np.asarray(history[idx], dtype=np.float32))) < 0.995 for idx in candidate_indices],
            dtype=bool,
        )
        if np.any(non_solid):
            candidate_indices = candidate_indices[non_solid]

        if candidate_indices.size > max_samples:
            sample_positions = np.linspace(0, candidate_indices.size - 1, max_samples)
            sampled = np.unique(np.round(sample_positions).astype(np.int64))
            candidate_indices = candidate_indices[sampled]

        return candidate_indices
        
    def analyze(self,
                riser_voxel_indices: np.ndarray,
                fraction_solid: np.ndarray,
                show_progress: bool = True,
                history: Optional[List[np.ndarray]] = None,
                history_times: Optional[List[float]] = None,
                reference_times: Optional[np.ndarray] = None,
                max_history_samples: int = 8) -> FeedingPathResult:
        """
        Analyze feeding paths from risers to all casting voxels.
        
        Args:
            riser_voxel_indices: Indices of voxels that are risers
            fraction_solid: Current fraction solid at each voxel
            show_progress: Print progress information
            
        Returns:
            FeedingPathResult with path quality and blocking info
        """
        n_voxels = self.mesh.get_voxel_count()
        
        if show_progress:
            print("\n" + "="*60)
            print("Analyzing Feeding Paths")
            print("="*60)
        
        # Create riser mask
        riser_mask = np.zeros(n_voxels, dtype=np.bool_)
        if len(riser_voxel_indices) > 0:
            riser_mask[riser_voxel_indices] = True
            
        if show_progress:
            print(f"  Riser voxels: {len(riser_voxel_indices):,}")
            print(f"  Total voxels: {n_voxels:,}")
        
        # Get mesh data
        voxel_centers = np.asarray(self.mesh.voxel_centers)
        neighbors = np.asarray(self.mesh.voxel_neighbors)

        snapshot_results: List[FeedingPathResult] = []
        snapshot_times: List[float] = []

        clipped_fraction_solid = np.ascontiguousarray(np.clip(fraction_solid, 0.0, 1.0), dtype=np.float64)
        clipped_reference_times = None
        if reference_times is not None:
            clipped_reference_times = np.asarray(reference_times, dtype=np.float64)

        if history is not None and history_times is not None and len(history) == len(history_times) and len(history) > 0:
            history_times_array = np.asarray(history_times, dtype=np.float64)
            if clipped_reference_times is not None and clipped_reference_times.size > 0:
                max_reference_time = float(np.max(clipped_reference_times))
            else:
                max_reference_time = float(history_times_array[-1])
            sample_indices = self._select_history_indices(
                history,
                history_times_array,
                max_reference_time,
                max_samples=max_history_samples,
            )
            if show_progress and sample_indices.size > 1:
                print(f"  Sampling {sample_indices.size} thermal-history snapshots for feeding paths...")
            for idx in sample_indices:
                snapshot_start = perf_counter()
                snapshot_results.append(
                    self._analyze_snapshot(
                        riser_mask,
                        np.ascontiguousarray(np.clip(history[idx], 0.0, 1.0), dtype=np.float64),
                        voxel_centers,
                        neighbors,
                    )
                )
                snapshot_times.append(float(history_times_array[idx]))
                if show_progress:
                    print(f"  Snapshot {int(idx)} feeding path time: {perf_counter() - snapshot_start:.2f}s")
        else:
            snapshot_start = perf_counter()
            snapshot_results.append(
                self._analyze_snapshot(
                    riser_mask,
                    clipped_fraction_solid,
                    voxel_centers,
                    neighbors,
                )
            )
            snapshot_times.append(0.0)
            if show_progress:
                print(f"  Snapshot feeding path time: {perf_counter() - snapshot_start:.2f}s")

        if len(snapshot_results) == 1 or clipped_reference_times is None:
            base_result = snapshot_results[-1]
            latest_open_time = np.where(
                base_result.path_length >= 0.0,
                np.float32(snapshot_times[-1]),
                np.float32(-1.0),
            )
            self.result = FeedingPathResult(
                feeding_quality=base_result.feeding_quality,
                path_length=base_result.path_length,
                path_permeability=base_result.path_permeability,
                feeding_source=base_result.feeding_source,
                blocked_voxels=base_result.blocked_voxels,
                history_open_fraction=(base_result.path_length >= 0.0).astype(np.float32),
                latest_open_time=latest_open_time.astype(np.float32),
                fed_fraction=base_result.fed_fraction,
                avg_path_length=base_result.avg_path_length,
                max_path_length=base_result.max_path_length,
                n_blocked=base_result.n_blocked,
            )
        else:
            merge_start = perf_counter()
            times_array = np.asarray(snapshot_times, dtype=np.float64)
            max_reference_time = max(float(np.max(clipped_reference_times)), 1e-6)
            path_length = np.full(n_voxels, -1.0, dtype=np.float32)
            path_permeability = np.zeros(n_voxels, dtype=np.float64)
            feeding_source = np.full(n_voxels, -1, dtype=np.int32)
            blocked_voxels = np.ones(n_voxels, dtype=np.bool_)
            feeding_quality = np.zeros(n_voxels, dtype=np.float32)
            history_open_fraction = np.zeros(n_voxels, dtype=np.float32)
            latest_open_time = np.full(n_voxels, -1.0, dtype=np.float32)
            total_weight = np.zeros(n_voxels, dtype=np.float32)

            for time_value, snapshot_result in zip(times_array, snapshot_results):
                eligible = time_value <= (clipped_reference_times + 1e-6)
                if not np.any(eligible):
                    continue

                time_weight = 0.2 + 0.8 * ((time_value / max_reference_time) ** 2)
                total_weight[eligible] += np.float32(time_weight)

                open_mask = eligible & (snapshot_result.path_length >= 0.0)
                history_open_fraction[open_mask] += np.float32(time_weight)
                latest_open_time[open_mask] = np.float32(time_value)

                feeding_quality[eligible] = snapshot_result.feeding_quality[eligible]
                path_length[eligible] = snapshot_result.path_length[eligible]
                path_permeability[eligible] = snapshot_result.path_permeability[eligible]
                feeding_source[eligible] = snapshot_result.feeding_source[eligible]
                blocked_voxels[eligible] = snapshot_result.blocked_voxels[eligible]

            valid_weight = total_weight > 0.0
            history_open_fraction[valid_weight] /= total_weight[valid_weight]
            feeding_quality = np.clip(
                (0.75 * feeding_quality) + (0.25 * history_open_fraction),
                0.0,
                1.0,
            ).astype(np.float32)
            fed_mask = path_length >= 0.0
            blocked_voxels = (~fed_mask) & ~riser_mask
            latest_open_time[riser_mask] = np.float32(max_reference_time)
            history_open_fraction[riser_mask] = 1.0
            feeding_quality[riser_mask] = 1.0
            path_length[riser_mask] = 0.0
            path_permeability[riser_mask] = 1.0
            feeding_source[riser_mask] = np.where(riser_mask)[0]

            fed_count = int(np.sum(fed_mask))
            valid_lengths = path_length[fed_mask]
            self.result = FeedingPathResult(
                feeding_quality=feeding_quality,
                path_length=path_length,
                path_permeability=path_permeability,
                feeding_source=feeding_source,
                blocked_voxels=blocked_voxels,
                history_open_fraction=history_open_fraction,
                latest_open_time=latest_open_time,
                fed_fraction=fed_count / n_voxels if n_voxels > 0 else 0.0,
                avg_path_length=float(np.mean(valid_lengths)) if valid_lengths.size > 0 else 0.0,
                max_path_length=float(np.max(valid_lengths)) if valid_lengths.size > 0 else 0.0,
                n_blocked=int(np.sum(blocked_voxels)),
            )
            if show_progress:
                print(f"  Feeding path history merge time: {perf_counter() - merge_start:.2f}s")
        
        if show_progress:
            print(f"\n  Results:")
            print(f"    Fed voxels: {int(np.sum(self.result.path_length >= 0.0)):,} ({self.result.fed_fraction*100:.1f}%)")
            print(f"    Blocked voxels: {self.result.n_blocked:,}")
            print(f"    Avg path length: {self.result.avg_path_length:.2f}\"")
            print(f"    Max path length: {self.result.max_path_length:.2f}\"")
            print("="*60 + "\n")
        
        return self.result
    
    def get_combined_defect_risk(self,
                                  niyama_risk: np.ndarray,
                                  weight_niyama: float = 0.5,
                                  weight_path: float = 0.5) -> np.ndarray:
        """
        Combine Niyama-based risk with feeding path quality for better prediction.
        
        Key insight: A voxel is truly at risk only if:
        1. Niyama criterion indicates shrinkage tendency, AND
        2. No good feeding path exists
        
        Args:
            niyama_risk: Boolean or float array of Niyama-based defect risk
            weight_niyama: Weight for Niyama factor (0-1)
            weight_path: Weight for path blockage factor (0-1)
            
        Returns:
            Combined defect risk array (0-1)
        """
        if self.result is None:
            raise RuntimeError("Call analyze() first")
        
        # Normalize weights
        total_weight = weight_niyama + weight_path
        w_n = weight_niyama / total_weight
        w_p = weight_path / total_weight
        
        # Convert Niyama risk to float if boolean
        if niyama_risk.dtype == np.bool_:
            niyama_float = niyama_risk.astype(np.float32)
        else:
            niyama_float = np.clip(niyama_risk, 0, 1).astype(np.float32)
        
        # Path blockage factor (1 = blocked, 0 = excellent path)
        path_blockage = 1.0 - self.result.feeding_quality
        
        # Combined risk: high if either factor is high
        # Use geometric mean to require both factors for highest risk
        combined = w_n * niyama_float + w_p * path_blockage
        
        # Boost risk for fully blocked voxels with any Niyama concern
        boost_mask = self.result.blocked_voxels & (niyama_float > 0.3)
        combined[boost_mask] = np.maximum(combined[boost_mask], 0.8)
        
        return np.clip(combined, 0, 1)
    
    def get_feeding_arrows(self, 
                           sample_fraction: float = 0.1,
                           min_path_length: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get arrow vectors for visualizing feeding directions.
        
        Returns list of (start_point, direction_vector) tuples for drawing
        feeding flow arrows in the 3D visualization.
        
        Args:
            sample_fraction: Fraction of voxels to include (for performance)
            min_path_length: Minimum path length to show arrow (inches)
            
        Returns:
            List of (start, direction) numpy array pairs
        """
        if self.result is None:
            raise RuntimeError("Call analyze() first")
        
        arrows = []
        voxel_centers = np.asarray(self.mesh.voxel_centers)
        
        # Get voxels with valid feeding paths
        valid_mask = (self.result.path_length > min_path_length) & \
                     (self.result.feeding_source >= 0)
        valid_indices = np.where(valid_mask)[0]
        
        # Sample for performance
        n_sample = max(10, int(len(valid_indices) * sample_fraction))
        if len(valid_indices) > n_sample:
            sample_indices = np.random.choice(valid_indices, n_sample, replace=False)
        else:
            sample_indices = valid_indices
        
        for idx in sample_indices:
            source_idx = self.result.feeding_source[idx]
            if source_idx < 0:
                continue
            
            start = voxel_centers[idx]
            end = voxel_centers[source_idx]
            direction = end - start
            
            # Normalize direction
            length = np.linalg.norm(direction)
            if length > 0.01:
                direction = direction / length
                arrows.append((start, direction))
        
        return arrows
    
    def get_statistics(self) -> Dict:
        """Get feeding path statistics"""
        if self.result is None:
            return {}
        
        return {
            'fed_fraction': self.result.fed_fraction,
            'blocked_count': self.result.n_blocked,
            'avg_path_length': self.result.avg_path_length,
            'max_path_length': self.result.max_path_length,
            'critical_fraction_solid': self.critical_fraction_solid,
            'dendrite_arm_spacing_um': self.dendrite_arm_spacing
        }


def create_feeding_analyzer(mesh, alloy=None) -> FeedingPathAnalyzer:
    """
    Convenience function to create FeedingPathAnalyzer.
    
    Args:
        mesh: AdaptiveVoxelMesh
        alloy: AlloyProperties (optional)
        
    Returns:
        Configured FeedingPathAnalyzer
    """
    return FeedingPathAnalyzer(mesh, alloy)


# Test function
if __name__ == '__main__':
    print("Feeding Path Analysis Module")
    print("="*60)
    print("This module provides feeding path tracing from risers to voxels.")
    print("\nKey features:")
    print("  - BFS path tracing through liquid/mushy zones")
    print("  - Kozeny-Carman permeability calculation")
    print("  - Combined Niyama + path quality defect prediction")
    print("  - Arrow visualization data for feeding flow")
