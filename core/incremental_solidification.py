import numpy as np
from numba import jit, prange
from typing import Tuple, Set, List, Optional
from dataclasses import dataclass

@dataclass
class SolidificationFront:
    active_indices: np.ndarray
    n_active: int

    _new_solid_mask: np.ndarray
    _next_front: np.ndarray

    updates_performed: int = 0
    voxels_solidified: int = 0

@jit(nopython=True, cache=True)
def find_initial_front(voxel_states: np.ndarray,
                       neighbors: np.ndarray,
                       n_voxels: int) -> Tuple[np.ndarray, int]:
    front_count = 0
    for i in range(n_voxels):
        if voxel_states[i] < 1.0:
            for j in range(neighbors.shape[1]):
                neighbor_idx = neighbors[i, j]
                if neighbor_idx >= 0 and voxel_states[neighbor_idx] >= 1.0:
                    front_count += 1
                    break

    front_indices = np.empty(front_count, dtype=np.int32)
    idx = 0
    for i in range(n_voxels):
        if voxel_states[i] < 1.0:
            for j in range(neighbors.shape[1]):
                neighbor_idx = neighbors[i, j]
                if neighbor_idx >= 0 and voxel_states[neighbor_idx] >= 1.0:
                    front_indices[idx] = i
                    idx += 1
                    break

    return front_indices, front_count

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def update_front_incremental(
    voxel_states: np.ndarray,
    front_indices: np.ndarray,
    n_front: int,
    neighbors: np.ndarray,
    moduli: np.ndarray,
    heights: np.ndarray,
    chill_boost: np.ndarray,
    riser_slowdown: np.ndarray,
    time_step: float,
    material_factor: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    n_voxels = len(voxel_states)
    newly_solid_count = 0

    newly_solid = np.empty(n_front, dtype=np.int32)

    for idx in prange(n_front):
        i = front_indices[idx]

        if voxel_states[i] >= 1.0:
            continue

        base_rate = 0.02
        solidification_rate = base_rate / (1.0 + moduli[i] * moduli[i] * 2.0)

        vertical_factor = 1.0 + (1.0 - heights[i]) * 0.3

        part_type_factor = chill_boost[i] * riser_slowdown[i]

        delta = solidification_rate * vertical_factor * material_factor * part_type_factor * time_step

        new_state = min(voxel_states[i] + delta, 1.0)
        voxel_states[i] = new_state

    for idx in range(n_front):
        i = front_indices[idx]
        if voxel_states[i] >= 1.0:
            newly_solid[newly_solid_count] = i
            newly_solid_count += 1

    return voxel_states, newly_solid[:newly_solid_count], newly_solid_count

@jit(nopython=True, cache=True)
def expand_front(front_indices: np.ndarray,
                 n_front: int,
                 newly_solid_indices: np.ndarray,
                 n_newly_solid: int,
                 voxel_states: np.ndarray,
                 neighbors: np.ndarray) -> Tuple[np.ndarray, int]:
    n_voxels = len(voxel_states)

    in_front = np.zeros(n_voxels, dtype=np.bool_)
    for idx in range(n_front):
        in_front[front_indices[idx]] = True

    max_new_front = n_front + n_newly_solid * neighbors.shape[1]
    new_front = np.empty(max_new_front, dtype=np.int32)
    new_front_count = 0

    for idx in range(n_front):
        i = front_indices[idx]
        if voxel_states[i] < 1.0:
            new_front[new_front_count] = i
            new_front_count += 1

    for idx in range(n_newly_solid):
        solid_idx = newly_solid_indices[idx]

        for j in range(neighbors.shape[1]):
            neighbor_idx = neighbors[solid_idx, j]

            if neighbor_idx >= 0:
                if voxel_states[neighbor_idx] < 1.0 and not in_front[neighbor_idx]:
                    new_front[new_front_count] = neighbor_idx
                    new_front_count += 1
                    in_front[neighbor_idx] = True

    return new_front[:new_front_count].copy(), new_front_count

class IncrementalSolidificationEngine:

    def __init__(self, n_voxels: int):
        self.n_voxels = n_voxels
        self.front_indices = None
        self.n_front = 0

        self._newly_solid = np.empty(n_voxels, dtype=np.int32)

        self.total_updates = 0
        self.total_voxels_processed = 0
        self.full_array_updates_avoided = 0

    def initialize_front(self, voxel_states: np.ndarray,
                         neighbors: np.ndarray) -> None:
        self.front_indices, self.n_front = find_initial_front(
            voxel_states, neighbors, self.n_voxels
        )

        print(f"  Incremental solidification initialized:")
        print(f"    Total voxels: {self.n_voxels:,}")
        print(f"    Initial front size: {self.n_front:,}")
        print(f"    Front ratio: {self.n_front / self.n_voxels * 100:.1f}%")

    def update(self,
               voxel_states: np.ndarray,
               neighbors: np.ndarray,
               moduli: np.ndarray,
               heights: np.ndarray,
               chill_boost: np.ndarray,
               riser_slowdown: np.ndarray,
               time_step: float,
               material_factor: float) -> np.ndarray:
        if self.n_front == 0:
            return voxel_states

        voxel_states, newly_solid, n_newly_solid = update_front_incremental(
            voxel_states,
            self.front_indices,
            self.n_front,
            neighbors,
            moduli,
            heights,
            chill_boost,
            riser_slowdown,
            time_step,
            material_factor
        )

        if n_newly_solid > 0:
            self.front_indices, self.n_front = expand_front(
                self.front_indices,
                self.n_front,
                newly_solid,
                n_newly_solid,
                voxel_states,
                neighbors
            )

        self.total_updates += 1
        self.total_voxels_processed += self.n_front
        self.full_array_updates_avoided += self.n_voxels - self.n_front

        return voxel_states

    def is_complete(self, voxel_states: np.ndarray) -> bool:
        return self.n_front == 0 or np.all(voxel_states >= 1.0)

    def get_stats(self) -> dict:
        avg_front = self.total_voxels_processed / max(self.total_updates, 1)
        speedup_estimate = self.n_voxels / max(avg_front, 1)

        return {
            'total_updates': self.total_updates,
            'total_voxels_processed': self.total_voxels_processed,
            'full_array_updates_avoided': self.full_array_updates_avoided,
            'average_front_size': avg_front,
            'estimated_speedup': speedup_estimate,
            'current_front_size': self.n_front
        }

    def print_stats(self):
        stats = self.get_stats()

        print("\n" + "="*60)
        print("Incremental Solidification Statistics")
        print("="*60)
        print(f"Total updates: {stats['total_updates']:,}")
        print(f"Voxels processed: {stats['total_voxels_processed']:,}")
        print(f"Full-array updates avoided: {stats['full_array_updates_avoided']:,}")
        print(f"Average front size: {stats['average_front_size']:.0f}")
        print(f"Estimated speedup: {stats['estimated_speedup']:.1f}x")
        print(f"Current front size: {stats['current_front_size']:,}")
        print("="*60 + "\n")

@jit(nopython=True, cache=True)
def sparse_states_to_dense(sparse_indices: np.ndarray,
                           sparse_values: np.ndarray,
                           n_total: int,
                           default_value: float = 0.0) -> np.ndarray:
    dense = np.full(n_total, default_value, dtype=np.float32)
    for i in range(len(sparse_indices)):
        dense[sparse_indices[i]] = sparse_values[i]
    return dense
