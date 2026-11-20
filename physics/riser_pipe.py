from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

@njit(cache=True)
def _higher(z: np.ndarray, t: np.ndarray, a: int, b: int) -> bool:
    if z[a] != z[b]:
        return z[a] > z[b]
    return t[a] > t[b]

@njit(cache=True)
def _sift_up(heap: np.ndarray, pos: int, z: np.ndarray, t: np.ndarray) -> None:
    item = heap[pos]
    while pos > 0:
        parent = (pos - 1) >> 1
        if _higher(z, t, item, heap[parent]):
            heap[pos] = heap[parent]
            pos = parent
        else:
            break
    heap[pos] = item

@njit(cache=True)
def _sift_down(heap: np.ndarray, size: int, z: np.ndarray, t: np.ndarray) -> None:
    pos = 0
    item = heap[0]
    while True:
        child = 2 * pos + 1
        if child >= size:
            break
        right = child + 1
        if right < size and _higher(z, t, heap[right], heap[child]):
            child = right
        if _higher(z, t, heap[child], item):
            heap[pos] = heap[child]
            pos = child
        else:
            break
    heap[pos] = item

@njit(cache=True)
def _drain_pool(
    times: np.ndarray,
    heights: np.ndarray,
    volumes: np.ndarray,
    beta: float,
) -> tuple:
    n = times.shape[0]
    drained = np.zeros(n, dtype=np.float64)
    if n == 0:
        return drained, 0.0

    heap = np.empty(n, dtype=np.int64)
    size = 0
    for i in range(n):
        heap[size] = i
        _sift_up(heap, size, heights, times)
        size += 1

    order = np.argsort(times, kind="mergesort")
    demand = 0.0
    unfed = 0.0
    eps = 1e-12
    for k in range(n):
        i = order[k]
        now = times[i]
        demand += beta * (volumes[i] - drained[i])
        while demand > eps:
            while size > 0:
                top = heap[0]
                if times[top] > now and drained[top] < volumes[top] - eps:
                    break
                size -= 1
                if size > 0:
                    heap[0] = heap[size]
                    _sift_down(heap, size, heights, times)
            if size == 0:
                unfed += demand
                demand = 0.0
                break
            top = heap[0]
            take = min(volumes[top] - drained[top], demand)
            drained[top] += take
            demand -= take
    return drained, unfed

def label_unfed_pools(
    neighbors: np.ndarray,
    metal_mask: np.ndarray,
    feeder_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    metal = np.asarray(metal_mask, dtype=bool)
    feeder = np.asarray(feeder_mask, dtype=bool) & metal
    nbr = np.asarray(neighbors)
    n = metal.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0 or not np.any(metal):
        return labels, 0

    rows = np.repeat(np.arange(n), nbr.shape[1])
    cols = nbr.ravel()
    valid = cols >= 0
    valid[valid] = metal[rows[valid]] & metal[cols[valid]]
    graph = coo_matrix(
        (np.ones(int(valid.sum()), dtype=np.int8), (rows[valid], cols[valid])),
        shape=(n, n),
    )
    _, component = connected_components(graph, directed=False)

    keep = metal & ~np.isin(component, np.unique(component[feeder]))
    if not np.any(keep):
        return labels, 0
    _, labels[keep] = np.unique(component[keep], return_inverse=True)
    return labels, int(labels.max()) + 1

@dataclass(frozen=True)
class PoolPipe:

    demand_volume: float
    pipe_volume: float
    unfed_volume: float
    feeder_pipe_volume: float
    section_pipe_volume: float

def gravity_drain_pipe(
    solidification_times: np.ndarray,
    heights: np.ndarray,
    volumes: np.ndarray,
    pool_ids: np.ndarray,
    feeder_mask: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, list[PoolPipe]]:
    times = np.asarray(solidification_times, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    finite = np.isfinite(times)
    if not np.all(finite):
        latest = float(times[finite].max()) if np.any(finite) else 0.0
        late = np.flatnonzero(~finite)
        times = times.copy()
        times[late[np.argsort(heights[late], kind="stable")]] = latest + 1.0 + np.arange(len(late))
    volumes = np.asarray(volumes, dtype=np.float64)
    pool_ids = np.asarray(pool_ids, dtype=np.int64)
    feeder_mask = np.asarray(feeder_mask, dtype=bool)
    beta = float(max(beta, 0.0))

    fraction = np.zeros(times.shape[0], dtype=np.float32)
    pools: list[PoolPipe] = []
    n_pools = int(pool_ids.max()) + 1 if pool_ids.size and pool_ids.max() >= 0 else 0
    order = np.argsort(pool_ids, kind="stable")
    starts = np.searchsorted(pool_ids[order], np.arange(n_pools + 1))
    for pool in range(n_pools):
        idx = order[starts[pool]:starts[pool + 1]]
        if idx.size == 0:
            pools.append(PoolPipe(0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        vol = volumes[idx]
        drained, unfed = _drain_pool(times[idx], heights[idx], vol, beta)
        fraction[idx] = np.clip(drained / np.maximum(vol, 1e-30), 0.0, 1.0)
        in_feeder = feeder_mask[idx]
        pools.append(
            PoolPipe(
                demand_volume=float(beta * (vol - drained).sum()),
                pipe_volume=float(drained.sum()),
                unfed_volume=float(unfed),
                feeder_pipe_volume=float(drained[in_feeder].sum()),
                section_pipe_volume=float(drained[~in_feeder].sum()),
            )
        )
    return fraction, pools
