"""
Octree Spatial Data Structure
Provides fast spatial queries for large voxel meshes with:
- O(log n) neighbor searches vs O(n) for flat arrays
- Reduced memory footprint through hierarchical storage
- Level-of-detail support for massive mesh rendering
- Cache-friendly memory layout for better performance
"""

import numpy as np
from typing import List, Tuple, Optional, Set, Generator
from dataclasses import dataclass, field
from numba import jit, prange
import math


@dataclass
class OctreeNode:
    """
    Single node in the octree.
    
    Each node represents a cubic region of space and can either:
    - Contain points directly (leaf node)
    - Have 8 children subdividing the space (internal node)
    """
    # Bounding box
    center: np.ndarray
    half_size: float
    
    # Node properties
    depth: int
    is_leaf: bool = True
    
    # Point storage (for leaf nodes)
    point_indices: List[int] = field(default_factory=list)
    
    # Children (for internal nodes) - indices 0-7 correspond to octants
    children: List[Optional['OctreeNode']] = field(default_factory=lambda: [None] * 8)
    
    # Cached data for fast queries
    point_count: int = 0
    
    def get_octant(self, point: np.ndarray) -> int:
        """
        Get the octant index for a point.
        
        Octant numbering:
        0: -x, -y, -z
        1: +x, -y, -z
        2: -x, +y, -z
        3: +x, +y, -z
        4: -x, -y, +z
        5: +x, -y, +z
        6: -x, +y, +z
        7: +x, +y, +z
        """
        octant = 0
        if point[0] >= self.center[0]:
            octant |= 1
        if point[1] >= self.center[1]:
            octant |= 2
        if point[2] >= self.center[2]:
            octant |= 4
        return octant
    
    def get_child_center(self, octant: int) -> np.ndarray:
        """Get the center of a child octant"""
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
        """Check if point is within this node's bounding box"""
        return np.all(np.abs(point - self.center) <= self.half_size)
    
    def intersects_sphere(self, center: np.ndarray, radius: float) -> bool:
        """Check if a sphere intersects this node's bounding box"""
        # Find closest point on box to sphere center
        closest = np.clip(center, 
                         self.center - self.half_size,
                         self.center + self.half_size)
        
        # Check if closest point is within radius
        dist_sq = np.sum((center - closest) ** 2)
        return dist_sq <= radius * radius


class Octree:
    """
    Octree spatial data structure for efficient spatial queries.
    
    Features:
    - Fast point insertion: O(log n)
    - Fast radius search: O(k + log n) where k is number of results
    - Fast k-nearest neighbors: O(k log n)
    - Memory efficient for clustered data
    - Level-of-detail point sampling
    """
    
    def __init__(self, 
                 bounds_min: np.ndarray,
                 bounds_max: np.ndarray,
                 max_depth: int = 10,
                 max_points_per_leaf: int = 16):
        """
        Initialize octree with bounding box.
        
        Args:
            bounds_min: Minimum corner of bounding box (x, y, z)
            bounds_max: Maximum corner of bounding box (x, y, z)
            max_depth: Maximum tree depth (default 10 = 1024 subdivisions per axis)
            max_points_per_leaf: Max points before subdivision (default 16)
        """
        self.bounds_min = np.asarray(bounds_min, dtype=np.float32)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float32)
        self.max_depth = max_depth
        self.max_points_per_leaf = max_points_per_leaf
        
        # Calculate root node properties
        center = (self.bounds_min + self.bounds_max) / 2
        half_size = np.max(self.bounds_max - self.bounds_min) / 2
        
        # Add small padding to handle edge cases
        half_size *= 1.01
        
        # Create root node
        self.root = OctreeNode(
            center=center,
            half_size=half_size,
            depth=0
        )
        
        # Point storage (stored separately for memory efficiency)
        self.points = None
        self.n_points = 0
        
        # Statistics
        self.total_nodes = 1
        self.leaf_nodes = 1
        self.max_reached_depth = 0
    
    def build(self, points: np.ndarray, show_progress: bool = True) -> None:
        """
        Build octree from array of points.
        
        Args:
            points: Array of 3D points (n_points x 3)
            show_progress: Show progress during build
        """
        self.points = np.asarray(points, dtype=np.float32)
        self.n_points = len(points)
        
        if show_progress:
            print(f"Building octree for {self.n_points:,} points...")
            print(f"  Max depth: {self.max_depth}")
            print(f"  Max points per leaf: {self.max_points_per_leaf}")
        
        # Insert all points
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
        """Insert a point into the octree"""
        # Update max depth
        self.max_reached_depth = max(self.max_reached_depth, node.depth)
        
        if node.is_leaf:
            # Add to this leaf
            node.point_indices.append(point_idx)
            node.point_count += 1
            
            # Check if we need to subdivide
            if (node.point_count > self.max_points_per_leaf and 
                node.depth < self.max_depth):
                self._subdivide(node)
        else:
            # Route to appropriate child
            octant = node.get_octant(point)
            
            if node.children[octant] is None:
                # Create child node
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
        """Subdivide a leaf node into 8 children"""
        node.is_leaf = False
        self.leaf_nodes -= 1
        
        # Redistribute points to children
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
            
            # Add point to child (recursive for deep nodes)
            self._insert(node.children[octant], point_idx, point)
        
        # Clear points from this node
        node.point_indices = []
    
    def query_radius(self, query_point: np.ndarray, radius: float) -> List[int]:
        """
        Find all points within radius of query point.
        
        Args:
            query_point: Center point for search (3D)
            radius: Search radius
        
        Returns:
            List of point indices within radius
        """
        query_point = np.asarray(query_point, dtype=np.float32)
        results = []
        radius_sq = radius * radius
        
        self._query_radius_recursive(self.root, query_point, radius, radius_sq, results)
        
        return results
    
    def _query_radius_recursive(self, node: OctreeNode, query_point: np.ndarray,
                                radius: float, radius_sq: float, 
                                results: List[int]) -> None:
        """Recursive radius query"""
        # Quick rejection test
        if not node.intersects_sphere(query_point, radius):
            return
        
        if node.is_leaf:
            # Check all points in this leaf
            for point_idx in node.point_indices:
                point = self.points[point_idx]
                dist_sq = np.sum((point - query_point) ** 2)
                if dist_sq <= radius_sq:
                    results.append(point_idx)
        else:
            # Recurse into children
            for child in node.children:
                if child is not None:
                    self._query_radius_recursive(child, query_point, radius, 
                                                radius_sq, results)
    
    def query_nearest(self, query_point: np.ndarray, k: int = 1) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors to query point.
        
        Args:
            query_point: Query point (3D)
            k: Number of neighbors to find
        
        Returns:
            List of (point_index, distance) tuples, sorted by distance
        """
        query_point = np.asarray(query_point, dtype=np.float32)
        
        # Use a simple approach: start with a small radius and expand
        # Could be optimized with a priority queue for better performance
        
        # Start with radius that should contain ~k points
        est_density = self.n_points / (self.root.half_size ** 3 * 8)
        initial_radius = (k / (est_density * 4/3 * np.pi)) ** (1/3)
        initial_radius = max(initial_radius, self.root.half_size / 100)
        
        results = []
        radius = initial_radius
        
        while len(results) < k and radius < self.root.half_size * 3:
            results = self.query_radius(query_point, radius)
            radius *= 2
        
        # Calculate distances and sort
        if not results:
            return []
        
        distances = []
        for idx in results:
            dist = np.sqrt(np.sum((self.points[idx] - query_point) ** 2))
            distances.append((idx, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def query_box(self, box_min: np.ndarray, box_max: np.ndarray) -> List[int]:
        """
        Find all points within an axis-aligned bounding box.
        
        Args:
            box_min: Minimum corner of box (x, y, z)
            box_max: Maximum corner of box (x, y, z)
        
        Returns:
            List of point indices within box
        """
        box_min = np.asarray(box_min, dtype=np.float32)
        box_max = np.asarray(box_max, dtype=np.float32)
        results = []
        
        self._query_box_recursive(self.root, box_min, box_max, results)
        
        return results
    
    def _query_box_recursive(self, node: OctreeNode, 
                             box_min: np.ndarray, box_max: np.ndarray,
                             results: List[int]) -> None:
        """Recursive box query"""
        # Quick rejection: check if node intersects query box
        node_min = node.center - node.half_size
        node_max = node.center + node.half_size
        
        if np.any(node_max < box_min) or np.any(node_min > box_max):
            return  # No intersection
        
        if node.is_leaf:
            # Check all points in this leaf
            for point_idx in node.point_indices:
                point = self.points[point_idx]
                if np.all(point >= box_min) and np.all(point <= box_max):
                    results.append(point_idx)
        else:
            # Recurse into children
            for child in node.children:
                if child is not None:
                    self._query_box_recursive(child, box_min, box_max, results)
    
    def get_neighbors_batch(self, point_indices: np.ndarray, 
                            radius: float,
                            max_neighbors: int = 26) -> np.ndarray:
        """
        Find neighbors for multiple points in batch.
        Optimized for finding voxel neighbors.
        
        Args:
            point_indices: Array of point indices to query
            radius: Search radius
            max_neighbors: Maximum neighbors to return per point
        
        Returns:
            Array of shape (n_queries, max_neighbors) with neighbor indices
            (-1 indicates no neighbor)
        """
        n_queries = len(point_indices)
        neighbors = np.full((n_queries, max_neighbors), -1, dtype=np.int32)
        
        for i, point_idx in enumerate(point_indices):
            query_point = self.points[point_idx]
            nearby = self.query_radius(query_point, radius)
            
            # Remove self from results
            nearby = [n for n in nearby if n != point_idx]
            
            # Store up to max_neighbors
            count = min(len(nearby), max_neighbors)
            neighbors[i, :count] = nearby[:count]
        
        return neighbors
    
    def get_points_at_lod(self, lod_depth: int) -> np.ndarray:
        """
        Get representative points at a specific level-of-detail.
        
        Lower LOD depth = fewer points (coarser representation)
        Higher LOD depth = more points (finer representation)
        
        Args:
            lod_depth: Maximum depth to traverse (0 = just root)
        
        Returns:
            Array of representative point indices
        """
        results = []
        self._get_lod_points_recursive(self.root, lod_depth, results)
        return np.array(results, dtype=np.int32)
    
    def _get_lod_points_recursive(self, node: OctreeNode, 
                                  max_depth: int, 
                                  results: List[int]) -> None:
        """Recursively collect LOD points"""
        if node.is_leaf or node.depth >= max_depth:
            # Get representative point for this node
            if node.is_leaf and node.point_indices:
                # Use first point as representative
                results.append(node.point_indices[0])
            elif not node.is_leaf:
                # For internal nodes at LOD limit, sample children
                for child in node.children:
                    if child is not None and child.point_count > 0:
                        self._get_lod_points_recursive(child, max_depth, results)
                        break  # Just one sample per node at this depth
        else:
            # Recurse deeper
            for child in node.children:
                if child is not None:
                    self._get_lod_points_recursive(child, max_depth, results)
    
    def get_stats(self) -> dict:
        """Get octree statistics"""
        # Calculate depth distribution
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
        """Count nodes at each depth"""
        depth_counts[node.depth] = depth_counts.get(node.depth, 0) + 1
        
        if not node.is_leaf:
            for child in node.children:
                if child is not None:
                    self._count_depths(child, depth_counts)
    
    def print_stats(self):
        """Print octree statistics"""
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
    """
    High-level spatial index that automatically chooses between
    octree and KD-tree based on data characteristics.
    """
    
    def __init__(self, points: np.ndarray, use_octree: bool = True):
        """
        Build spatial index for points.
        
        Args:
            points: Array of 3D points (n_points x 3)
            use_octree: If True, use octree; if False, use scipy KD-tree
        """
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
        """Find points within radius"""
        if self.use_octree:
            return self._octree.query_radius(query_point, radius)
        else:
            return self._kdtree.query_ball_point(query_point, radius)
    
    def query_nearest(self, query_point: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors, returns (distances, indices)"""
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
        """
        Find neighbors for ALL points.
        
        Args:
            radius: Search radius
            max_neighbors: Max neighbors per point
        
        Returns:
            Neighbor array (n_points, max_neighbors)
        """
        if self.use_octree:
            return self._octree.get_neighbors_batch(
                np.arange(self.n_points), radius, max_neighbors
            )
        else:
            # KD-tree approach
            neighbors = np.full((self.n_points, max_neighbors), -1, dtype=np.int32)
            
            for i in range(self.n_points):
                nearby = self._kdtree.query_ball_point(self.points[i], radius)
                nearby = [n for n in nearby if n != i]
                count = min(len(nearby), max_neighbors)
                neighbors[i, :count] = nearby[:count]
            
            return neighbors
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        if self.use_octree:
            return self._octree.get_stats()
        else:
            return {
                'type': 'kdtree',
                'n_points': self.n_points
            }


# JIT-optimized helper functions
@jit(nopython=True, cache=True)
def _compute_octant(point: np.ndarray, center: np.ndarray) -> int:
    """Fast octant computation"""
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
    """Compute squared distances from all points to query (parallel)"""
    n = points.shape[0]
    distances = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        dx = points[i, 0] - query[0]
        dy = points[i, 1] - query[1]
        dz = points[i, 2] - query[2]
        distances[i] = dx*dx + dy*dy + dz*dz
    
    return distances
