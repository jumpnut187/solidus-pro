import numpy as np
from time import perf_counter

from numba import jit, prange
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import deque

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

try:
    from core.materials import AlloyProperties
except ImportError:
    AlloyProperties = None

@dataclass
class FeedingPathResult:
    feeding_quality: np.ndarray
    path_length: np.ndarray
    path_permeability: np.ndarray
    feeding_source: np.ndarray
    blocked_voxels: np.ndarray
    history_open_fraction: np.ndarray
    latest_open_time: np.ndarray

    fed_fraction: float
    avg_path_length: float
    max_path_length: float
    n_blocked: int

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_path_permeability_batch(
    fraction_solid: np.ndarray,
    critical_fraction: float,
    dendrite_spacing_um: float
) -> np.ndarray:
    n_voxels = len(fraction_solid)
    permeability = np.zeros(n_voxels, dtype=np.float64)

    lambda_m = dendrite_spacing_um * 1e-6
    lambda_sq = lambda_m * lambda_m

    for i in prange(n_voxels):
        fs = fraction_solid[i]

        if fs <= 0.01:
            permeability[i] = 1e-8
        elif fs >= 0.99:
            permeability[i] = 0.0
        else:
            fl = 1.0 - fs

            if fs > 0.01:
                K = (lambda_sq * (fl ** 3)) / (180.0 * (fs ** 2))
            else:
                K = 1e-8

            if fs >= critical_fraction:
                overshoot = (fs - critical_fraction) / (1.0 - critical_fraction)
                decay = np.exp(-10.0 * overshoot)
                K *= decay

            permeability[i] = min(K, 1e-8)

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
    if riser_mask[start_idx]:
        return (0.0, 1.0, start_idx)

    n_voxels = len(riser_mask)
    max_neighbors = neighbors.shape[1]

    visited = np.zeros(n_voxels, dtype=np.bool_)
    visited[start_idx] = True

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

        if riser_mask[current]:
            return (current_dist, current_min_perm, current)

        for j in range(max_neighbors):
            neighbor = neighbors[current, j]
            if neighbor < 0:
                continue
            if visited[neighbor]:
                continue

            neighbor_perm = permeability[neighbor]
            if neighbor_perm < permeability_threshold:
                continue

            visited[neighbor] = True

            dx = voxel_centers[neighbor, 0] - voxel_centers[current, 0]
            dy = voxel_centers[neighbor, 1] - voxel_centers[current, 1]
            dz = voxel_centers[neighbor, 2] - voxel_centers[current, 2]
            step_dist = np.sqrt(dx*dx + dy*dy + dz*dz)

            queue[queue_end] = neighbor
            distances[queue_end] = current_dist + step_dist
            min_perms[queue_end] = min(current_min_perm, neighbor_perm)
            queue_end += 1

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
def _heap_sift_up(heap: np.ndarray, heap_pos: np.ndarray, keys: np.ndarray, i: int) -> None:
    node = heap[i]
    key = keys[node]
    while i > 0:
        parent = (i - 1) >> 1
        parent_node = heap[parent]
        if keys[parent_node] <= key:
            break
        heap[i] = parent_node
        heap_pos[parent_node] = i
        i = parent
    heap[i] = node
    heap_pos[node] = i

@jit(nopython=True, cache=True)
def _heap_sift_down(heap: np.ndarray, heap_pos: np.ndarray, keys: np.ndarray, i: int, size: int) -> None:
    node = heap[i]
    key = keys[node]
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        child_key = keys[heap[child]]
        right = child + 1
        if right < size:
            right_key = keys[heap[right]]
            if right_key < child_key:
                child = right
                child_key = right_key
        if child_key >= key:
            break
        child_node = heap[child]
        heap[i] = child_node
        heap_pos[child_node] = i
        i = child
    heap[i] = node
    heap_pos[node] = i

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
    heap = np.zeros(n_voxels, dtype=np.int64)
    heap_pos = np.full(n_voxels, -1, dtype=np.int64)
    heap_size = 0

    for riser_idx in sorted_risers:
        if visited[riser_idx]:
            continue
        heap[heap_size] = riser_idx
        heap_pos[riser_idx] = heap_size
        heap_size += 1
        visited[riser_idx] = True

    max_neighbors = neighbors.shape[1]
    c_epsilon = 1e-6

    while heap_size > 0:
        current = heap[0]
        heap_size -= 1
        if heap_size > 0:
            last = heap[heap_size]
            heap[0] = last
            heap_pos[last] = 0
            _heap_sift_down(heap, heap_pos, path_length, 0, heap_size)
        heap_pos[current] = -1

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
                heap[heap_size] = neighbor
                heap_pos[neighbor] = heap_size
                heap_size += 1
                _heap_sift_up(heap, heap_pos, path_length, heap_size - 1)
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
                    slot = heap_pos[neighbor]
                    if slot >= 0:
                        _heap_sift_up(heap, heap_pos, path_length, slot)

    return path_length, path_permeability, feeding_source

class FeedingPathAnalyzer:

    def _neighbor_edges(self, neighbors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cached = getattr(self, "_edge_cache", None)
        if cached is not None and cached[0] is neighbors:
            return cached[1], cached[2]
        neighbors = np.asarray(neighbors)
        n_voxels, width = neighbors.shape
        rows = np.repeat(np.arange(n_voxels, dtype=np.int64), width)
        cols = neighbors.reshape(-1).astype(np.int64)
        keep = (cols >= 0) & (cols > rows)
        edge_i, edge_j = rows[keep], cols[keep]
        self._edge_cache = (neighbors, edge_i, edge_j)
        return edge_i, edge_j

    def liquid_network_cutoff(
        self,
        riser_mask: np.ndarray,
        fraction_solid: np.ndarray,
        neighbors: np.ndarray,
    ) -> np.ndarray:
        n_voxels = len(fraction_solid)
        needs_feed = np.asarray(fraction_solid) < self.critical_fraction_solid
        if not np.any(needs_feed):
            return np.zeros(n_voxels, dtype=np.bool_)
        edge_i, edge_j = self._neighbor_edges(neighbors)
        live = needs_feed[edge_i] & needs_feed[edge_j]
        graph = coo_matrix(
            (np.ones(int(np.count_nonzero(live)), dtype=np.int8), (edge_i[live], edge_j[live])),
            shape=(n_voxels, n_voxels),
        )
        _, labels = connected_components(graph, directed=False)
        fed_labels = np.unique(labels[riser_mask & needs_feed])
        return needs_feed & ~np.isin(labels, fed_labels)

    def __init__(self,
                 voxel_mesh,
                 alloy: Optional['AlloyProperties'] = None):
        self.mesh = voxel_mesh
        self.alloy = alloy

        if alloy is not None:
            self.critical_fraction_solid = alloy.critical_fraction_solid
            self.dendrite_arm_spacing = alloy.dendrite_arm_spacing
        else:
            self.critical_fraction_solid = 0.52
            self.dendrite_arm_spacing = 80.0

        self.permeability_threshold = 1e-16

        self.result: Optional[FeedingPathResult] = None

    def _analyze_snapshot(
        self,
        riser_mask: np.ndarray,
        fraction_solid: np.ndarray,
        voxel_centers: np.ndarray,
        neighbors: np.ndarray,
    ) -> FeedingPathResult:
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

        max_feeding_dist = 10.0
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

        blocked_voxels = (feeding_source < 0) & (fraction_solid < self.critical_fraction_solid)
        blocked_voxels |= self.liquid_network_cutoff(riser_mask, fraction_solid, neighbors)
        open_mask = path_length >= 0.0
        latest_open_time = np.where(open_mask, 0.0, -1.0).astype(np.float32)
        fed_fraction = n_fed / n_voxels if n_voxels > 0 else 0.0
        avg_path_length = total_path_length / n_fed if n_fed > 0 else 0.0

        return FeedingPathResult(
            feeding_quality=feeding_quality,
            path_length=path_length,
            path_permeability=path_permeability,
            feeding_source=feeding_source,
            blocked_voxels=blocked_voxels,
            history_open_fraction=open_mask.astype(np.float32),
            latest_open_time=latest_open_time,
            fed_fraction=fed_fraction,
            avg_path_length=avg_path_length,
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
                max_history_samples: int = 48) -> FeedingPathResult:
        n_voxels = self.mesh.get_voxel_count()

        if show_progress:
            print("\n" + "="*60)
            print("Analyzing Feeding Paths")
            print("="*60)

        riser_mask = np.zeros(n_voxels, dtype=np.bool_)
        if len(riser_voxel_indices) > 0:
            riser_mask[riser_voxel_indices] = True

        if show_progress:
            print(f"  Riser voxels: {len(riser_voxel_indices):,}")
            print(f"  Total voxels: {n_voxels:,}")

        voxel_centers = np.asarray(self.mesh.voxel_centers)
        neighbors = np.asarray(self.mesh.voxel_neighbors)

        snapshot_results: List[FeedingPathResult] = []
        snapshot_times: List[float] = []
        snapshot_needs_feed: List[np.ndarray] = []

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
                snapshot_fs = np.ascontiguousarray(np.clip(history[idx], 0.0, 1.0), dtype=np.float64)
                snapshot_results.append(
                    self._analyze_snapshot(
                        riser_mask,
                        snapshot_fs,
                        voxel_centers,
                        neighbors,
                    )
                )
                snapshot_times.append(float(history_times_array[idx]))
                snapshot_needs_feed.append(snapshot_fs < self.critical_fraction_solid)
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
            any_blocked_accum = np.zeros(n_voxels, dtype=np.bool_)

            for time_value, snapshot_result, needs_feed in zip(
                times_array, snapshot_results, snapshot_needs_feed
            ):
                eligible = time_value <= (clipped_reference_times + 1e-6)
                active = eligible & needs_feed
                if not np.any(active):
                    continue

                time_weight = 0.2 + 0.8 * ((time_value / max_reference_time) ** 2)
                total_weight[active] += np.float32(time_weight)

                open_mask = active & (snapshot_result.path_length >= 0.0)
                history_open_fraction[open_mask] += np.float32(time_weight)
                latest_open_time[open_mask] = np.float32(time_value)

                feeding_quality[active] = snapshot_result.feeding_quality[active]
                path_length[active] = snapshot_result.path_length[active]
                path_permeability[active] = snapshot_result.path_permeability[active]
                feeding_source[active] = snapshot_result.feeding_source[active]
                any_blocked_accum[active] |= snapshot_result.blocked_voxels[active]

            if history is not None and history_times is not None and len(history) == len(history_times):
                for snapshot, history_time in zip(history, history_times):
                    history_time = float(history_time)
                    if history_time > max_reference_time + 1e-6:
                        continue
                    cutoff = self.liquid_network_cutoff(
                        riser_mask,
                        np.asarray(snapshot, dtype=np.float64),
                        neighbors,
                    )
                    if np.any(cutoff):
                        any_blocked_accum |= cutoff & (history_time <= clipped_reference_times + 1e-6)

            valid_weight = total_weight > 0.0
            history_open_fraction[valid_weight] /= total_weight[valid_weight]
            never_active = ~valid_weight
            if np.any(never_active):
                history_open_fraction[never_active] = 1.0
                feeding_quality[never_active] = 1.0

            mean_fs_per_snapshot = [float(np.mean(nf)) for nf in snapshot_needs_feed]
            most_liquid = int(np.argmax(mean_fs_per_snapshot))
            liquid_needs_feed = snapshot_needs_feed[most_liquid] & ~riser_mask
            n_needing = int(np.sum(liquid_needs_feed))
            if n_needing > 0:
                reached = snapshot_results[most_liquid].path_length >= 0.0
                coverage = float(np.sum(reached & liquid_needs_feed)) / n_needing
                if coverage < 0.05:
                    print(
                        "  WARNING: feeding-path trace reached "
                        f"{coverage * 100.0:.1f}% of the liquid casting - risers "
                        "appear disconnected from the casting in the voxel mesh "
                        "(check part alignment / riser necks). Ignoring "
                        "connectivity results for this run."
                    )
                    any_blocked_accum[:] = False
                    history_open_fraction[:] = 1.0
                    feeding_quality[:] = 1.0

            feeding_quality = np.clip(
                (0.75 * feeding_quality) + (0.25 * history_open_fraction),
                0.0,
                1.0,
            ).astype(np.float32)
            fed_mask = path_length >= 0.0
            blocked_voxels = any_blocked_accum & ~riser_mask
            latest_open_time[riser_mask] = np.float32(max_reference_time)
            history_open_fraction[riser_mask] = 1.0
            feeding_quality[riser_mask] = 1.0
            path_length[riser_mask] = 0.0
            path_permeability[riser_mask] = 1.0
            feeding_source[riser_mask] = np.where(riser_mask)[0]

            fed_count = int(np.sum(fed_mask))
            valid_lengths = path_length[fed_mask]
            fed_fraction = fed_count / n_voxels if n_voxels > 0 else 0.0
            avg_path_length = float(np.mean(valid_lengths)) if valid_lengths.size > 0 else 0.0
            self.result = FeedingPathResult(
                feeding_quality=feeding_quality,
                path_length=path_length,
                path_permeability=path_permeability,
                feeding_source=feeding_source,
                blocked_voxels=blocked_voxels,
                history_open_fraction=history_open_fraction,
                latest_open_time=latest_open_time,
                fed_fraction=fed_fraction,
                avg_path_length=avg_path_length,
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
        if self.result is None:
            raise RuntimeError("Call analyze() first")

        total_weight = weight_niyama + weight_path
        w_n = weight_niyama / total_weight
        w_p = weight_path / total_weight

        if niyama_risk.dtype == np.bool_:
            niyama_float = niyama_risk.astype(np.float32)
        else:
            niyama_float = np.clip(niyama_risk, 0, 1).astype(np.float32)

        path_blockage = 1.0 - self.result.feeding_quality

        combined = w_n * niyama_float + w_p * path_blockage

        boost_mask = self.result.blocked_voxels & (niyama_float > 0.3)
        combined[boost_mask] = np.maximum(combined[boost_mask], 0.8)

        return np.clip(combined, 0, 1)

    def get_feeding_arrows(self,
                           sample_fraction: float = 0.1,
                           min_path_length: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self.result is None:
            raise RuntimeError("Call analyze() first")

        arrows = []
        voxel_centers = np.asarray(self.mesh.voxel_centers)

        valid_mask = (self.result.path_length > min_path_length) & \
                     (self.result.feeding_source >= 0)
        valid_indices = np.where(valid_mask)[0]

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

            length = np.linalg.norm(direction)
            if length > 0.01:
                direction = direction / length
                arrows.append((start, direction))

        return arrows

    def get_statistics(self) -> Dict:
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

if __name__ == '__main__':
    print("Feeding Path Analysis Module")
    print("="*60)
    print("This module provides feeding path tracing from risers to voxels.")
    print("\nKey features:")
    print("  - BFS path tracing through liquid/mushy zones")
    print("  - Kozeny-Carman permeability calculation")
    print("  - Combined Niyama + path quality defect prediction")
    print("  - Arrow visualization data for feeding flow")
