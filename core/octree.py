import numpy as np
from typing import List, Tuple, Optional, Set, Generator
from dataclasses import dataclass, field
from numba import jit, prange
import math

@dataclass
class OctreeNode:
    center: np.ndarray
    half_size: float

    depth: int
    is_leaf: bool = True

    point_indices: List[int] = field(default_factory=list)

    children: List[Optional['OctreeNode']] = field(default_factory=lambda: [None] * 8)

    point_count: int = 0

    def get_octant(self, point: np.ndarray) -> int:
        octant = 0
        if point[0] >= self.center[0]:
            octant |= 1
        if point[1] >= self.center[1]:
            octant |= 2
        if point[2] >= self.center[2]:
            octant |= 4
        return octant

    def get_child_center(self, octant: int) -> np.ndarray:
        offset = self.half_size / 2
        child_center = self.center.copy()

        if octant & 1:
            child_center[0] += offset
        else:
            child_center[0] -= offset

        if octant & 2:
            child_center[1] += offset
        else:
            child_center[1] -= offset

        if octant & 4:
            child_center[2] += offset
        else:
            child_center[2] -= offset

        return child_center

    def contains_point(self, point: np.ndarray) -> bool:
        return np.all(np.abs(point - self.center) <= self.half_size)

    def intersects_sphere(self, center: np.ndarray, radius: float) -> bool:
        closest = np.clip(center,
                         self.center - self.half_size,
                         self.center + self.half_size)

        dist_sq = np.sum((center - closest) ** 2)
        return dist_sq <= radius * radius

class Octree:

    def __init__(self,
                 bounds_min: np.ndarray,
                 bounds_max: np.ndarray,
                 max_depth: int = 10,
                 max_points_per_leaf: int = 16):
        self.bounds_min = np.asarray(bounds_min, dtype=np.float32)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float32)
        self.max_depth = max_depth
        self.max_points_per_leaf = max_points_per_leaf

        center = (self.bounds_min + self.bounds_max) / 2
        half_size = np.max(self.bounds_max - self.bounds_min) / 2

        half_size *= 1.01

        self.root = OctreeNode(
            center=center,
            half_size=half_size,
            depth=0
        )

        self.points = None
        self.n_points = 0

        self.total_nodes = 1
        self.leaf_nodes = 1
        self.max_reached_depth = 0

    def build(self, points: np.ndarray, show_progress: bool = True) -> None:
        self.points = np.asarray(points, dtype=np.float32)
        self.n_points = len(points)

        if show_progress:
            print(f"Building octree for {self.n_points:,} points...")
            print(f"  Max depth: {self.max_depth}")
            print(f"  Max points per leaf: {self.max_points_per_leaf}")

        for i in range(self.n_points):
            self._insert(self.root, i, self.points[i])

            if show_progress and (i + 1) % 50000 == 0:
                print(f"  Inserted {i+1:,}/{self.n_points:,} points...")

        if show_progress:
            print(f"  ✓ Octree built:")
            print(f"    Total nodes: {self.total_nodes:,}")
            print(f"    Leaf nodes: {self.leaf_nodes:,}")
            print(f"    Max depth reached: {self.max_reached_depth}")

    def _insert(self, node: OctreeNode, point_idx: int, point: np.ndarray) -> None:
        self.max_reached_depth = max(self.max_reached_depth, node.depth)

        if node.is_leaf:
            node.point_indices.append(point_idx)
            node.point_count += 1

            if (node.point_count > self.max_points_per_leaf and
                node.depth < self.max_depth):
                self._subdivide(node)
        else:
            octant = node.get_octant(point)

            if node.children[octant] is None:
                child_center = node.get_child_center(octant)
                node.children[octant] = OctreeNode(
                    center=child_center,
                    half_size=node.half_size / 2,
                    depth=node.depth + 1
                )
                self.total_nodes += 1
                self.leaf_nodes += 1

            self._insert(node.children[octant], point_idx, point)

    def _subdivide(self, node: OctreeNode) -> None:
        node.is_leaf = False
        self.leaf_nodes -= 1

        for point_idx in node.point_indices:
            point = self.points[point_idx]
            octant = node.get_octant(point)

            if node.children[octant] is None:
                child_center = node.get_child_center(octant)
                node.children[octant] = OctreeNode(
                    center=child_center,
                    half_size=node.half_size / 2,
                    depth=node.depth + 1
                )
                self.total_nodes += 1
                self.leaf_nodes += 1

            self._insert(node.children[octant], point_idx, point)

        node.point_indices = []

    def query_radius(self, query_point: np.ndarray, radius: float) -> List[int]:
        query_point = np.asarray(query_point, dtype=np.float32)
        results = []
        radius_sq = radius * radius

        self._query_radius_recursive(self.root, query_point, radius, radius_sq, results)

        return results

    def _query_radius_recursive(self, node: OctreeNode, query_point: np.ndarray,
                                radius: float, radius_sq: float,
                                results: List[int]) -> None:
        if not node.intersects_sphere(query_point, radius):
            return

        if node.is_leaf:
            for point_idx in node.point_indices:
                point = self.points[point_idx]
                dist_sq = np.sum((point - query_point) ** 2)
                if dist_sq <= radius_sq:
                    results.append(point_idx)
        else:
            for child in node.children:
                if child is not None:
                    self._query_radius_recursive(child, query_point, radius,
                                                radius_sq, results)

    def query_nearest(self, query_point: np.ndarray, k: int = 1) -> List[Tuple[int, float]]:
        query_point = np.asarray(query_point, dtype=np.float32)

        est_density = self.n_points / (self.root.half_size ** 3 * 8)
        initial_radius = (k / (est_density * 4/3 * np.pi)) ** (1/3)
        initial_radius = max(initial_radius, self.root.half_size / 100)

        results = []
        radius = initial_radius

        while len(results) < k and radius < self.root.half_size * 3:
            results = self.query_radius(query_point, radius)
            radius *= 2

        if not results:
            return []

        distances = []
        for idx in results:
            dist = np.sqrt(np.sum((self.points[idx] - query_point) ** 2))
            distances.append((idx, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def query_box(self, box_min: np.ndarray, box_max: np.ndarray) -> List[int]:
        box_min = np.asarray(box_min, dtype=np.float32)
        box_max = np.asarray(box_max, dtype=np.float32)
        results = []

        self._query_box_recursive(self.root, box_min, box_max, results)

        return results

    def _query_box_recursive(self, node: OctreeNode,
                             box_min: np.ndarray, box_max: np.ndarray,
                             results: List[int]) -> None:
        node_min = node.center - node.half_size
        node_max = node.center + node.half_size

        if np.any(node_max < box_min) or np.any(node_min > box_max):
            return

        if node.is_leaf:
            for point_idx in node.point_indices:
                point = self.points[point_idx]
                if np.all(point >= box_min) and np.all(point <= box_max):
                    results.append(point_idx)
        else:
            for child in node.children:
                if child is not None:
                    self._query_box_recursive(child, box_min, box_max, results)

    def get_neighbors_batch(self, point_indices: np.ndarray,
                            radius: float,
                            max_neighbors: int = 26) -> np.ndarray:
        n_queries = len(point_indices)
        neighbors = np.full((n_queries, max_neighbors), -1, dtype=np.int32)

        for i, point_idx in enumerate(point_indices):
            query_point = self.points[point_idx]
            nearby = self.query_radius(query_point, radius)

            nearby = [n for n in nearby if n != point_idx]

            count = min(len(nearby), max_neighbors)
            neighbors[i, :count] = nearby[:count]

        return neighbors

    def get_points_at_lod(self, lod_depth: int) -> np.ndarray:
        results = []
        self._get_lod_points_recursive(self.root, lod_depth, results)
        return np.array(results, dtype=np.int32)

    def _get_lod_points_recursive(self, node: OctreeNode,
                                  max_depth: int,
                                  results: List[int]) -> None:
        if node.is_leaf or node.depth >= max_depth:
            if node.is_leaf and node.point_indices:
                results.append(node.point_indices[0])
            elif not node.is_leaf:
                for child in node.children:
                    if child is not None and child.point_count > 0:
                        self._get_lod_points_recursive(child, max_depth, results)
                        break
        else:
            for child in node.children:
                if child is not None:
                    self._get_lod_points_recursive(child, max_depth, results)

    def get_stats(self) -> dict:
        depth_counts = {}
        self._count_depths(self.root, depth_counts)

        return {
            'n_points': self.n_points,
            'total_nodes': self.total_nodes,
            'leaf_nodes': self.leaf_nodes,
            'internal_nodes': self.total_nodes - self.leaf_nodes,
            'max_depth': self.max_reached_depth,
            'bounds_min': self.bounds_min.tolist(),
            'bounds_max': self.bounds_max.tolist(),
            'depth_distribution': depth_counts,
            'avg_points_per_leaf': self.n_points / max(self.leaf_nodes, 1)
        }

    def _count_depths(self, node: OctreeNode, depth_counts: dict) -> None:
        depth_counts[node.depth] = depth_counts.get(node.depth, 0) + 1

        if not node.is_leaf:
            for child in node.children:
                if child is not None:
                    self._count_depths(child, depth_counts)

    def print_stats(self):
        stats = self.get_stats()

        print("\n" + "="*60)
        print("Octree Statistics")
        print("="*60)
        print(f"Points: {stats['n_points']:,}")
        print(f"Total nodes: {stats['total_nodes']:,}")
        print(f"  Leaf nodes: {stats['leaf_nodes']:,}")
        print(f"  Internal nodes: {stats['internal_nodes']:,}")
        print(f"Max depth: {stats['max_depth']}")
        print(f"Avg points/leaf: {stats['avg_points_per_leaf']:.1f}")
        print(f"Bounds: {stats['bounds_min']} to {stats['bounds_max']}")
        print("="*60 + "\n")

class SpatialIndex:

    def __init__(self, points: np.ndarray, use_octree: bool = True):
        self.points = np.asarray(points, dtype=np.float32)
        self.n_points = len(points)
        self.use_octree = use_octree

        self._octree = None
        self._kdtree = None

        if use_octree:
            bounds_min = np.min(points, axis=0)
            bounds_max = np.max(points, axis=0)

            self._octree = Octree(bounds_min, bounds_max)
            self._octree.build(points, show_progress=self.n_points > 10000)
        else:
            from scipy.spatial import cKDTree
            self._kdtree = cKDTree(points)

    def query_radius(self, query_point: np.ndarray, radius: float) -> List[int]:
        if self.use_octree:
            return self._octree.query_radius(query_point, radius)
        else:
            return self._kdtree.query_ball_point(query_point, radius)

    def query_nearest(self, query_point: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_octree:
            results = self._octree.query_nearest(query_point, k)
            if not results:
                return np.array([]), np.array([])
            indices = np.array([r[0] for r in results])
            distances = np.array([r[1] for r in results])
            return distances, indices
        else:
            return self._kdtree.query(query_point, k)

    def get_neighbors_batch(self, radius: float, max_neighbors: int = 26) -> np.ndarray:
        if self.use_octree:
            return self._octree.get_neighbors_batch(
                np.arange(self.n_points), radius, max_neighbors
            )
        else:
            neighbors = np.full((self.n_points, max_neighbors), -1, dtype=np.int32)

            for i in range(self.n_points):
                nearby = self._kdtree.query_ball_point(self.points[i], radius)
                nearby = [n for n in nearby if n != i]
                count = min(len(nearby), max_neighbors)
                neighbors[i, :count] = nearby[:count]

            return neighbors

    def get_stats(self) -> dict:
        if self.use_octree:
            return self._octree.get_stats()
        else:
            return {
                'type': 'kdtree',
                'n_points': self.n_points
            }

@jit(nopython=True, cache=True)
def _compute_octant(point: np.ndarray, center: np.ndarray) -> int:
    octant = 0
    if point[0] >= center[0]:
        octant |= 1
    if point[1] >= center[1]:
        octant |= 2
    if point[2] >= center[2]:
        octant |= 4
    return octant

@jit(nopython=True, parallel=True, cache=True)
def batch_distance_squared(points: np.ndarray, query: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    distances = np.empty(n, dtype=np.float32)

    for i in prange(n):
        dx = points[i, 0] - query[0]
        dy = points[i, 1] - query[1]
        dz = points[i, 2] - query[2]
        distances[i] = dx*dx + dy*dy + dz*dz

    return distances
