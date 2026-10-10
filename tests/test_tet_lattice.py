"""Tests for core/tet_lattice.py: barycentric lattice sampling of tet meshes (DESIGN.md C3)."""

from __future__ import annotations

import itertools
import unittest
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import pytetwild
import trimesh
from scipy.spatial import cKDTree

from core.dense_grid import Lattice, derive_lattice, flat_indices
from core.part_manager import PART_TYPE_CAST, PART_TYPE_CHILL
from core.tet_lattice import CANONICAL_SLOT_OFFSETS, TetLatticeSampler
from core.voxel_mesh import AdaptiveVoxelMesh


def kuhn_hex_mesh(
    shape: Tuple[int, int, int], h: float, keep: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Conforming 6-tet Kuhn split of a ``shape`` hex grid of pitch ``h`` inches.

    ``keep`` is an optional (hx, hy, hz) bool mask of hexes to mesh. Tets are
    positively oriented. Returns ``(node_xyz, tets)``.
    """
    hx, hy, hz = shape
    gx, gy, gz = np.meshgrid(
        np.arange(hx + 1), np.arange(hy + 1), np.arange(hz + 1), indexing="ij"
    )
    nodes = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float64) * h

    def node_id(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        return (i * (hy + 1) + j) * (hz + 1) + k

    hexes = np.argwhere(np.ones(shape, dtype=bool) if keep is None else keep)
    tets = []
    for perm in itertools.permutations(range(3)):
        corner = hexes.copy()
        path = [node_id(*corner.T)]
        for axis in perm:
            corner = corner.copy()
            corner[:, axis] += 1
            path.append(node_id(*corner.T))
        tets.append(np.stack(path, axis=1))
    tet_arr = np.concatenate(tets, axis=0)
    p = nodes[tet_arr]
    det = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    flip = det < 0
    tet_arr[flip, 2], tet_arr[flip, 3] = tet_arr[flip, 3], tet_arr[flip, 2].copy()
    return nodes, tet_arr


def tetwild_mesh(surface: trimesh.Trimesh, edge_length: float) -> Tuple[np.ndarray, np.ndarray]:
    """Tetrahedralize a closed surface with pytetwild (inches in, inches out)."""
    nodes, tets = pytetwild.tetrahedralize(
        np.asarray(surface.vertices, dtype=np.float64),
        np.asarray(surface.faces, dtype=np.int32),
        edge_length_abs=edge_length,
        epsilon=1e-3,
    )
    return np.asarray(nodes, dtype=np.float64), np.asarray(tets, dtype=np.int64)


def tet_volumes(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p = nodes[tets]
    det = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    return np.abs(det) / 6.0


class TetLatticeSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cyl_nodes, cls.cyl_tets = tetwild_mesh(
            trimesh.creation.cylinder(radius=0.5, height=1.5, sections=64), 0.15
        )
        cls.box_nodes, cls.box_tets = tetwild_mesh(
            trimesh.creation.box(extents=(2.0, 1.3, 0.7)), 0.15
        )

    def test_flat_index_matches_dense_grid_helpers(self) -> None:
        nodes, tets = kuhn_hex_mesh((4, 3, 2), 0.25)
        sampler = TetLatticeSampler(nodes, tets, np.zeros(len(tets), dtype=np.int8), 0.1)
        self.assertEqual(sampler.dims, (10, 8, 5))
        lattice = Lattice(dims=sampler.dims, origin=tuple(sampler.origin), spacing=sampler.spacing)
        np.testing.assert_array_equal(flat_indices(sampler.cell_centers, lattice), sampler.cell_flat)
        # A box fills its lattice, so dense_grid recovers the same lattice from the centres.
        derived = derive_lattice(sampler.cell_centers, np.full(len(sampler.cell_flat), 0.1))
        self.assertEqual(derived.dims, sampler.dims)
        np.testing.assert_allclose(derived.origin, sampler.origin, atol=1e-12)
        self.assertEqual(len(sampler.cell_flat), lattice.cell_count)
        self.assertTrue(np.all(np.diff(sampler.cell_flat) > 0))

    def test_sample_reproduces_linear_field(self) -> None:
        sampler = TetLatticeSampler(
            self.cyl_nodes, self.cyl_tets, np.zeros(len(self.cyl_tets), dtype=np.int8), 0.05
        )
        coeffs = np.array([3.0, -1.5, 0.25])
        # Elementwise sums: numpy's Accelerate matmul raises spurious FP warnings on macOS.
        field = 7.0 + (self.cyl_nodes * coeffs).sum(axis=1)
        expected = 7.0 + (sampler.cell_centers * coeffs).sum(axis=1)
        np.testing.assert_allclose(sampler.sample(field), expected, rtol=0, atol=1e-10)

        stacked = np.stack([field, 2.0 * field])
        np.testing.assert_allclose(sampler.sample(stacked), np.stack([expected, 2.0 * expected]), atol=1e-10)

        dense = sampler.dense(field, fill=-1.0)
        self.assertEqual(dense.dtype, np.float32)
        self.assertEqual(dense.shape, (sampler.cell_count,))
        np.testing.assert_allclose(dense[sampler.cell_flat], expected, rtol=1e-6)
        outside = np.ones(sampler.cell_count, dtype=bool)
        outside[sampler.cell_flat] = False
        self.assertTrue(np.any(outside))
        self.assertTrue(np.all(dense[outside] == -1.0))
        self.assertTrue(np.all(np.isnan(sampler.dense(field)[outside])))

    def test_inside_volume_matches_box_mesh(self) -> None:
        spacing = 0.05
        sampler = TetLatticeSampler(
            self.box_nodes, self.box_tets, np.zeros(len(self.box_tets), dtype=np.int8), spacing
        )
        mesh_volume = float(tet_volumes(self.box_nodes, self.box_tets).sum())
        lattice_volume = len(sampler.cell_flat) * spacing**3
        self.assertLess(abs(lattice_volume - mesh_volume) / mesh_volume, 0.03)

    def test_inside_volume_matches_cylinder_mesh(self) -> None:
        spacing = 0.025
        sampler = TetLatticeSampler(
            self.cyl_nodes, self.cyl_tets, np.zeros(len(self.cyl_tets), dtype=np.int8), spacing
        )
        mesh_volume = float(tet_volumes(self.cyl_nodes, self.cyl_tets).sum())
        lattice_volume = len(sampler.cell_flat) * spacing**3
        self.assertLess(abs(lattice_volume - mesh_volume) / mesh_volume, 0.05)

    def test_two_region_mask_codes_and_part_types(self) -> None:
        # L-shaped 2x2x1 hex block, hex (1, 1, 0) missing; x < 1 is CAST, x > 1 is CHILL.
        keep = np.ones((2, 2, 1), dtype=bool)
        keep[1, 1, 0] = False
        nodes, tets = kuhn_hex_mesh((2, 2, 1), 1.0, keep)
        centroid_x = nodes[tets][:, :, 0].mean(axis=1)
        codes = np.where(centroid_x < 1.0, PART_TYPE_CAST, PART_TYPE_CHILL).astype(np.int8)
        sampler = TetLatticeSampler(nodes, tets, codes, 0.25)
        self.assertEqual(sampler.dims, (8, 8, 4))

        centers = sampler.cell_centers
        expected = np.where(centers[:, 0] < 1.0, PART_TYPE_CAST, PART_TYPE_CHILL)
        np.testing.assert_array_equal(sampler.cell_part_types, expected)
        self.assertEqual(sampler.cell_part_types.dtype, np.int8)
        self.assertEqual(len(sampler.cell_flat), 3 * 4 * 4 * 4)

        mask = sampler.mask_codes()
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, (sampler.cell_count,))
        volume = mask.reshape((4, 8, 8)).transpose(2, 1, 0)  # [ix, iy, iz]
        self.assertTrue(np.all(volume[:4, :, :] == PART_TYPE_CAST + 1))
        self.assertTrue(np.all(volume[4:, :4, :] == PART_TYPE_CHILL + 1))
        self.assertTrue(np.all(volume[4:, 4:, :] == 0))

        view = sampler.lattice_view()
        np.testing.assert_array_equal(view.get_voxels_by_type(PART_TYPE_CHILL), np.flatnonzero(expected == PART_TYPE_CHILL))

    def test_nearest_node_is_max_weight_node(self) -> None:
        sampler = TetLatticeSampler(
            self.cyl_nodes, self.cyl_tets, np.zeros(len(self.cyl_tets), dtype=np.int8), 0.05
        )
        rows = np.arange(len(sampler.cell_flat))
        cell_nodes = self.cyl_tets[sampler.cell_tet]
        weight_of_nearest = sampler.cell_bary[cell_nodes == sampler.nearest_node[:, None]]
        self.assertEqual(weight_of_nearest.shape, rows.shape)
        np.testing.assert_array_equal(weight_of_nearest, sampler.cell_bary.max(axis=1))
        self.assertEqual(sampler.nearest_node.dtype, np.int32)
        self.assertTrue(np.all(sampler.cell_bary >= -1e-9))
        np.testing.assert_allclose(sampler.cell_bary.sum(axis=1), 1.0, atol=1e-12)

    def test_point_on_shared_face_is_located_once(self) -> None:
        # Unit cube at spacing 0.5: every centre lies on the cube diagonal or on
        # a Kuhn face (x = y, y = z or x = z), so each is inside 2 to 6 tets.
        nodes, tets = kuhn_hex_mesh((1, 1, 1), 1.0)
        sampler = TetLatticeSampler(nodes, tets, np.zeros(len(tets), dtype=np.int8), 0.5)
        self.assertEqual(sampler.dims, (2, 2, 2))
        np.testing.assert_array_equal(sampler.cell_flat, np.arange(8))

        p = nodes[tets]
        inv = np.linalg.inv(np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2))
        for cell, centre in enumerate(sampler.cell_centers):
            lam = np.einsum("tij,tj->ti", inv, centre - p[:, 0])
            lam = np.column_stack([1.0 - lam.sum(axis=1), lam])
            containing = np.flatnonzero(np.all(lam >= -1e-9, axis=1))
            self.assertGreaterEqual(len(containing), 2)
            self.assertEqual(int(sampler.cell_tet[cell]), int(containing[0]))

    def test_lattice_view_face_slots_are_signed_axis_neighbours(self) -> None:
        keep = np.ones((2, 2, 1), dtype=bool)
        keep[1, 1, 0] = False
        nodes, tets = kuhn_hex_mesh((2, 2, 1), 1.0, keep)
        sampler = TetLatticeSampler(nodes, tets, np.zeros(len(tets), dtype=np.int8), 0.25)
        view = sampler.lattice_view()
        self.assertEqual(view.mesh_type, "lattice")
        self.assertIs(view.sampler, sampler)
        self.assertEqual(view.voxel_neighbors.shape, (len(sampler.cell_flat), 26))
        self.assertEqual(view.voxel_neighbors.dtype, np.int32)

        centers = sampler.cell_centers
        tree = cKDTree(centers)
        directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        for slot, direction in enumerate(directions):
            dist, idx = tree.query(centers + 0.25 * direction, k=1)
            expected = np.where(dist < 1e-9, idx, -1)
            np.testing.assert_array_equal(view.voxel_neighbors[:, slot], expected)

        # All 26 slots agree with the voxel mesh's canonical slot placement.
        mesh_like = SimpleNamespace(voxel_centers=view.voxel_centers, base_voxel_size=0.25)
        neighbour_lists = tree.query_ball_point(centers, r=0.25 * np.sqrt(3.0) + 1e-6)
        for i in range(0, len(centers), 7):
            canonical = AdaptiveVoxelMesh._canonicalize_neighbor_order(
                mesh_like, i, centers[i], neighbour_lists[i]
            )
            np.testing.assert_array_equal(view.voxel_neighbors[i], canonical)
        self.assertEqual(CANONICAL_SLOT_OFFSETS.shape, (26, 3))

    def test_box_surface_flags_areas_and_moduli(self) -> None:
        spacing = 0.25
        nodes, tets = kuhn_hex_mesh((4, 3, 2), 0.5)
        sampler = TetLatticeSampler(nodes, tets, np.zeros(len(tets), dtype=np.int8), spacing)
        view = sampler.lattice_view()
        nx, ny, nz = sampler.dims
        flat = sampler.cell_flat
        ijk = np.stack([flat % nx, (flat // nx) % ny, flat // (nx * ny)], axis=1)
        dims = np.array(sampler.dims)
        exposed = (ijk == 0).sum(axis=1) + (ijk == dims - 1).sum(axis=1)

        np.testing.assert_array_equal(view.surface_voxels, exposed > 0)
        np.testing.assert_allclose(view.voxel_surface_areas, exposed * spacing**2)
        np.testing.assert_array_equal(view.get_surface_voxel_indices(), np.flatnonzero(exposed > 0))
        self.assertEqual(view.get_voxel_count(), nx * ny * nz)

        surface_tree = cKDTree(sampler.cell_centers[view.surface_voxels])
        expected_moduli, _ = surface_tree.query(sampler.cell_centers)
        np.testing.assert_allclose(view.voxel_moduli, expected_moduli, atol=1e-6)
        self.assertTrue(np.all(view.voxel_moduli[view.surface_voxels] == 0.0))

        np.testing.assert_allclose(view.voxel_sizes, spacing)
        np.testing.assert_allclose(view.voxel_volumes, spacing**3)
        self.assertEqual(view.grid_shape, sampler.dims)
        self.assertEqual(view.base_voxel_size, spacing)
        np.testing.assert_allclose(view.bounds, [[0.0, 2.0], [0.0, 1.5], [0.0, 1.0]])

    def test_lattice_view_builds_topology_arrays_on_first_access(self) -> None:
        nodes, tets = kuhn_hex_mesh((4, 3, 2), 0.5)
        sampler = TetLatticeSampler(nodes, tets, np.zeros(len(tets), dtype=np.int8), 0.25)
        view = sampler.lattice_view()
        lazy = ("voxel_neighbors", "surface_voxels", "voxel_surface_areas", "voxel_moduli")
        self.assertFalse(set(lazy) & set(vars(view)))
        for name in lazy:
            self.assertEqual(len(getattr(view, name)), view.get_voxel_count())
        self.assertTrue(set(lazy) <= set(vars(view)))
        self.assertIs(view.voxel_neighbors, view.voxel_neighbors)

    def test_tet_larger_than_one_chunk_is_located(self) -> None:
        import core.tet_lattice as tet_lattice

        nodes, tets = kuhn_hex_mesh((1, 1, 1), 1.0)
        original = tet_lattice.CANDIDATE_CHUNK
        try:
            tet_lattice.CANDIDATE_CHUNK = 37
            chunked = TetLatticeSampler(nodes, tets, np.zeros(6, dtype=np.int8), 0.07)
        finally:
            tet_lattice.CANDIDATE_CHUNK = original
        whole = TetLatticeSampler(nodes, tets, np.zeros(6, dtype=np.int8), 0.07)
        np.testing.assert_array_equal(chunked.cell_flat, whole.cell_flat)
        np.testing.assert_array_equal(chunked.cell_tet, whole.cell_tet)
        np.testing.assert_array_equal(chunked.cell_bary, whole.cell_bary)
        self.assertEqual(len(whole.cell_flat), whole.cell_count)

    def test_rejects_bad_input(self) -> None:
        nodes, tets = kuhn_hex_mesh((1, 1, 1), 1.0)
        with self.assertRaises(ValueError):
            TetLatticeSampler(nodes, tets, np.zeros(5, dtype=np.int8), 0.1)
        with self.assertRaises(ValueError):
            TetLatticeSampler(nodes, tets, np.zeros(6, dtype=np.int8), 0.0)
        sampler = TetLatticeSampler(nodes, tets, np.zeros(6, dtype=np.int8), 0.5)
        with self.assertRaises(ValueError):
            sampler.sample(np.zeros(len(nodes) + 1))


if __name__ == "__main__":
    unittest.main()
