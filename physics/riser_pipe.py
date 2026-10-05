"""Primary shrinkage pipe by gravity-drained volume balance.

A feeder does not shrink in a fixed shape. Every voxel it feeds gives up
``beta * V`` of volume when it freezes, and that volume leaves the feed pool
at the pool's free surface: the highest voxel that is still liquid. Walls and
the top crust freeze first, so the free surface falls inside a shrinking
liquid core and the drained voxels trace the pipe. Its depth and shape follow
the solidification times. An insulating sleeve, a bigger riser or less feed
demand gives a shallower pipe. A feeder that freezes before the section it
feeds lets the free surface drop into the casting, and the pipe follows it.

This module is mesh-agnostic like the rest of ``physics/``. It takes plain
arrays and imports nothing from ``core/`` or ``config/``.

Assumptions:
- The liquid in one feed pool is hydraulically connected (the caller builds
  the pool from the feeding-path tracer's ``feeding_source``).
- Volume demand is booked when a voxel reaches its solidification time.
- The free surface is the highest liquid voxel. Ties go to the voxel that
  freezes last, the thermal centre.
- Liquid contraction above the liquidus is not counted. That underestimates
  the pipe a little for superheated pours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@njit(cache=True)
def _higher(z: np.ndarray, t: np.ndarray, a: int, b: int) -> bool:
    # True when voxel ``a`` sits above ``b`` in the free-surface order.
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
    """Drain one feed pool. Arrays are local to the pool.

    Returns ``(drained_volume_per_voxel, unfed_volume)``. ``unfed_volume`` is
    demand left once the pool had no liquid voxel to give, which only happens
    in the last instants of freezing.
    """
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
            # Lazily drop voxels that are already solid or fully drained.
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


@dataclass(frozen=True)
class PoolPipe:
    """Pipe summary for one feed pool."""

    demand_volume: float      # beta * volume that froze in the pool (drained volume never freezes)
    pipe_volume: float        # volume drained out of the pool
    unfed_volume: float       # demand nothing liquid was left to supply
    feeder_pipe_volume: float  # drained volume inside the feeder voxels
    section_pipe_volume: float  # drained volume outside the feeder (in the casting)


def gravity_drain_pipe(
    solidification_times: np.ndarray,
    heights: np.ndarray,
    volumes: np.ndarray,
    pool_ids: np.ndarray,
    feeder_mask: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, list[PoolPipe]]:
    """Per-voxel pipe fraction for every feed pool.

    Args:
        solidification_times: (N,) seconds. Non-finite means never froze in
            the simulated window and is treated as the latest time.
        heights: (N,) vertical coordinate, any length unit (gravity is -z).
        volumes: (N,) voxel volumes, any unit. Reported volumes use it.
        pool_ids: (N,) int. Voxels sharing an id >= 0 form one connected feed
            pool. -1 means the voxel takes no part (unfed, chill, core).
        feeder_mask: (N,) bool. Voxels that belong to a feeder (riser),
            used only to split the reported pipe volume.
        beta: volumetric solidification shrinkage as a fraction (0.03 = 3 %).

    Returns:
        ``(pipe_fraction, pools)``. ``pipe_fraction`` is the drained share of
        each voxel's volume, 0..1 (1 = open pipe). ``pools`` is indexed by
        pool id.
    """
    times = np.asarray(solidification_times, dtype=np.float64)
    finite = np.isfinite(times)
    latest = float(times[finite].max()) if np.any(finite) else 0.0
    times = np.where(finite, times, latest)
    heights = np.asarray(heights, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    pool_ids = np.asarray(pool_ids, dtype=np.int64)
    feeder_mask = np.asarray(feeder_mask, dtype=bool)
    beta = float(max(beta, 0.0))

    fraction = np.zeros(times.shape[0], dtype=np.float32)
    pools: list[PoolPipe] = []
    n_pools = int(pool_ids.max()) + 1 if pool_ids.size and pool_ids.max() >= 0 else 0
    for pool in range(n_pools):
        idx = np.flatnonzero(pool_ids == pool)
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
