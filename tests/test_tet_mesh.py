"""Tests for core/tet_mesh.py (contract C1 in docs/tet_migration/DESIGN.md)."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh

from config.exceptions import MeshGenerationError
from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_RISER,
)
from core.tet_mesh import TetVolumeMesh, build_tet_mesh

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "sample_stl_files"


def _box(extents, center) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _cylinder(radius, height, center, sections=32) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    mesh.apply_translation(center)
    return mesh


def _code_volume(mesh: TetVolumeMesh, code: int) -> float:
    return float(mesh.tet_volumes[mesh.tet_part_types == code].sum())


def _nodes_of_code(mesh: TetVolumeMesh, code: int) -> np.ndarray:
    return np.unique(mesh.tets[mesh.tet_part_types == code])


def _enclosed_volume(mesh: TetVolumeMesh) -> float:
    """Divergence theorem over the boundary faces; positive only if faces point outward."""
    tri = mesh.node_xyz[mesh.boundary_faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(np.einsum("ij,ij->i", tri[:, 0], cross).sum() / 6.0)


# Casting box z in [0, 2], riser cylinder on top, chill box underneath.
CASTING = _box((4.0, 3.0, 2.0), (0.0, 0.0, 1.0))
RISER = _cylinder(0.8, 1.5, (0.0, 0.0, 2.75))
CHILL = _box((2.0, 2.0, 0.5), (0.0, 0.0, -0.25))
ASSEMBLY_H = 0.25


class TetMeshSampleStlTest(unittest.TestCase):
    """Single-part meshes of the sample STLs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.stl = {}
        cls.meshes = {}
        for name, h in (("cylinder", 0.15), ("l_bracket", 0.2)):
            stl = trimesh.load(SAMPLES / f"{name}.stl", force="mesh")
            cls.stl[name] = stl
            cls.meshes[name] = build_tet_mesh([(stl, PART_TYPE_CAST)], h)

    def test_l_bracket_is_not_watertight(self) -> None:
        self.assertFalse(self.stl["l_bracket"].is_watertight)

    def test_volume_matches_stl(self) -> None:
        for name, mesh in self.meshes.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    mesh.tet_volumes.sum() / abs(self.stl[name].volume), 1.0, delta=0.005
                )

    def test_lumped_node_volume_conserves_tet_volume(self) -> None:
        for name, mesh in self.meshes.items():
            with self.subTest(name=name):
                np.testing.assert_allclose(
                    mesh.voxel_volumes.astype(np.float64).sum(), mesh.tet_volumes.sum(), rtol=1e-5
                )

    def test_boundary_area_matches_stl(self) -> None:
        # l_bracket.stl is two concatenated boxes (base 4x2x0.5, upright 0.5x2x3).
        # Its 39 in2 STL area counts the 0.5x2 contact patch twice as internal
        # faces; the solid's skin is 39 - 2 = 37 in2.
        exterior_area = {"cylinder": self.stl["cylinder"].area, "l_bracket": self.stl["l_bracket"].area - 2.0}
        self.assertAlmostEqual(self.stl["l_bracket"].area, 39.0, places=6)
        for name, mesh in self.meshes.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    mesh.boundary_face_areas.sum() / exterior_area[name], 1.0, delta=0.02
                )
                np.testing.assert_allclose(
                    mesh.voxel_surface_areas.astype(np.float64).sum(),
                    mesh.boundary_face_areas.sum(),
                    rtol=1e-5,
                )

    def test_tets_positive_and_faces_outward(self) -> None:
        for name, mesh in self.meshes.items():
            with self.subTest(name=name):
                self.assertTrue(np.all(mesh.tet_volumes > 0.0))
                np.testing.assert_allclose(_enclosed_volume(mesh), mesh.tet_volumes.sum(), rtol=1e-9)

    def test_neighbor_table_symmetric_and_csr_consistent(self) -> None:
        for name, mesh in self.meshes.items():
            with self.subTest(name=name):
                n = mesh.get_voxel_count()
                indptr, indices = mesh.adj_indptr, mesh.adj_indices
                self.assertEqual(indptr.dtype, np.int64)
                self.assertEqual(indices.dtype, np.int32)
                self.assertEqual(indptr[0], 0)
                self.assertEqual(indptr[-1], len(indices))
                adjacency = sp.csr_matrix(
                    (np.ones(len(indices)), indices, indptr), shape=(n, n)
                )
                self.assertEqual((adjacency != adjacency.T).nnz, 0)
                self.assertEqual(adjacency.diagonal().sum(), 0)
                rows = np.repeat(np.arange(n), np.diff(indptr))
                same_row = rows[1:] == rows[:-1]
                self.assertTrue(np.all(indices[1:][same_row] > indices[:-1][same_row]))
                # Padded table equals the CSR rows.
                table = mesh.voxel_neighbors
                self.assertEqual(table.dtype, np.int32)
                self.assertEqual(table.shape, (n, int(np.diff(indptr).max())))
                np.testing.assert_array_equal((table >= 0).sum(axis=1), np.diff(indptr))
                np.testing.assert_array_equal(table[table >= 0], indices)
                # Every tet edge is an adjacency entry.
                a, b = mesh.tets[:, 0], mesh.tets[:, 3]
                self.assertTrue(np.all(np.asarray(adjacency[a, b]).ravel() == 1))

    def test_node_arrays_follow_contract(self) -> None:
        mesh = self.meshes["cylinder"]
        n = mesh.get_voxel_count()
        self.assertEqual(mesh.mesh_type, "tet")
        self.assertIsNone(mesh.grid_shape)
        self.assertEqual(mesh.voxel_centers.dtype, np.float32)
        np.testing.assert_allclose(mesh.voxel_centers, mesh.node_xyz, atol=1e-5)
        for name in ("voxel_volumes", "voxel_sizes", "voxel_surface_areas", "voxel_moduli", "voxel_states"):
            arr = getattr(mesh, name)
            self.assertEqual((arr.shape, arr.dtype), ((n,), np.float32), name)
        self.assertEqual(mesh.voxel_part_types.dtype, np.int8)
        self.assertEqual(mesh.tets.dtype, np.int32)
        self.assertEqual(mesh.tet_part_types.dtype, np.int8)
        np.testing.assert_array_equal(mesh.surface_voxels, mesh.voxel_surface_areas > 0)
        self.assertTrue(np.all(mesh.boundary_face_contact_codes == -1))
        self.assertTrue(np.all(mesh.boundary_face_part_types == PART_TYPE_CAST))
        self.assertEqual(mesh.bounds.shape, (3, 2))
        self.assertEqual(len(mesh.metal_part_bounds), 1)
        self.assertEqual(mesh.base_voxel_size, 0.15)
        self.assertAlmostEqual(mesh.mean_edge_length, 0.15, delta=0.05)
        self.assertTrue(np.all(mesh.voxel_sizes > 0))
        # No orphan nodes.
        self.assertEqual(len(np.unique(mesh.tets)), n)
        np.testing.assert_array_equal(mesh.get_surface_voxel_indices(), np.flatnonzero(mesh.surface_voxels))
        np.testing.assert_array_equal(mesh.get_voxels_by_type(PART_TYPE_CAST), np.arange(n))
        stats = mesh.get_memory_stats()
        for key in ("ram_mb", "disk_mb", "available_gb"):
            self.assertIn(key, stats)

    def test_moduli_match_exact_distance_on_curved_skin(self) -> None:
        for name, mesh in self.meshes.items():
            with self.subTest(name=name):
                skin = trimesh.Trimesh(mesh.node_xyz, mesh.boundary_faces, process=False)
                sample = np.flatnonzero(~mesh.surface_voxels)[::5][:400]
                _, distance, _ = trimesh.proximity.closest_point(skin, mesh.node_xyz[sample])
                np.testing.assert_allclose(mesh.voxel_moduli[sample], distance, atol=1e-5)

    def test_quality_summary(self) -> None:
        mesh = self.meshes["cylinder"]
        summary = mesh.quality_summary()
        self.assertEqual(summary["tet_count"], len(mesh.tets))
        self.assertEqual(summary["node_count"], mesh.get_voxel_count())
        self.assertEqual(summary["boundary_face_count"], len(mesh.boundary_faces))
        self.assertGreater(summary["min_dihedral_deg"], 0.0)
        self.assertGreater(summary["median_dihedral_deg"], summary["min_dihedral_deg"])
        self.assertLess(summary["median_dihedral_deg"], 180.0)
        self.assertGreaterEqual(summary["sliver_count"], 0)


class TetMeshAssemblyTest(unittest.TestCase):
    """Casting box with a riser on top and a chill underneath, both touching."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh = build_tet_mesh(
            [(CASTING, PART_TYPE_CAST), (RISER, PART_TYPE_RISER), (CHILL, PART_TYPE_CHILL)],
            ASSEMBLY_H,
        )
        # Permuted input order: proves marker k maps to the k-th leaf.
        cls.permuted = build_tet_mesh(
            [(CHILL, PART_TYPE_CHILL), (CASTING, PART_TYPE_CAST), (RISER, PART_TYPE_RISER)],
            ASSEMBLY_H,
        )

    def test_per_code_volumes(self) -> None:
        for label, mesh in (("input order", self.mesh), ("permuted", self.permuted)):
            for code, part in (
                (PART_TYPE_CAST, CASTING),
                (PART_TYPE_RISER, RISER),
                (PART_TYPE_CHILL, CHILL),
            ):
                with self.subTest(order=label, code=code):
                    self.assertAlmostEqual(_code_volume(mesh, code) / part.volume, 1.0, delta=0.01)

    def test_interfaces_share_nodes(self) -> None:
        cast = _nodes_of_code(self.mesh, PART_TYPE_CAST)
        self.assertGreater(len(np.intersect1d(cast, _nodes_of_code(self.mesh, PART_TYPE_RISER))), 10)
        self.assertGreater(len(np.intersect1d(cast, _nodes_of_code(self.mesh, PART_TYPE_CHILL))), 10)
        # Conforming: the interfaces are interior, so only the exterior skin is boundary.
        exterior = CASTING.area + RISER.area + CHILL.area - 2 * (np.pi * 0.8**2) - 2 * 4.0
        self.assertAlmostEqual(self.mesh.boundary_face_areas.sum() / exterior, 1.0, delta=0.02)
        self.assertTrue(np.all(self.mesh.boundary_face_contact_codes == -1))

    def test_node_part_type_rule_at_cast_chill_interface(self) -> None:
        mesh = self.mesh
        cast = _nodes_of_code(mesh, PART_TYPE_CAST)
        chill = _nodes_of_code(mesh, PART_TYPE_CHILL)
        riser = _nodes_of_code(mesh, PART_TYPE_RISER)
        cast_chill = np.intersect1d(cast, chill)
        self.assertTrue(np.all(mesh.voxel_part_types[cast_chill] == PART_TYPE_CAST))
        chill_only = np.setdiff1d(chill, cast)
        self.assertGreater(len(chill_only), 0)
        self.assertTrue(np.all(mesh.voxel_part_types[chill_only] == PART_TYPE_CHILL))
        # Metal-metal interface: highest metal code.
        self.assertTrue(np.all(mesh.voxel_part_types[np.intersect1d(cast, riser)] == PART_TYPE_RISER))
        # Interior cast/chill interface nodes count as material surface.
        material = mesh.get_material_surface_voxel_indices()
        interior_interface = cast_chill[~mesh.surface_voxels[cast_chill]]
        self.assertGreater(len(interior_interface), 0)
        self.assertTrue(np.all(np.isin(interior_interface, material)))
        self.assertTrue(np.all(np.isin(mesh.get_surface_voxel_indices(), material)))

    def test_boundary_face_owner_codes(self) -> None:
        mesh = self.mesh
        centroid_z = mesh.node_xyz[mesh.boundary_faces].mean(axis=1)[:, 2]
        owners = mesh.boundary_face_part_types
        self.assertTrue(np.all(centroid_z[owners == PART_TYPE_RISER] >= 2.0 - 1e-6))
        self.assertTrue(np.all(centroid_z[owners == PART_TYPE_CHILL] <= 1e-6))
        self.assertEqual(len(mesh.metal_part_bounds), 2)


class TetMeshCoreAndOverlapTest(unittest.TestCase):
    """CORE removal and highest-code-wins overlap resolution."""

    BOX = _box((2.0, 2.0, 2.0), (0.0, 0.0, 1.0))
    CORE = _cylinder(0.5, 1.0, (0.0, 0.0, 1.0), sections=24)
    # Riser cylinder sunk 1 in into the casting box (z 1..3).
    SUNK_RISER = _cylinder(0.8, 2.0, (0.0, 0.0, 2.0))

    @classmethod
    def setUpClass(cls) -> None:
        h = 0.2
        cls.core_meshes = {
            "box first": build_tet_mesh([(cls.BOX, PART_TYPE_CAST), (cls.CORE, PART_TYPE_CORE)], h),
            "core first": build_tet_mesh([(cls.CORE, PART_TYPE_CORE), (cls.BOX, PART_TYPE_CAST)], h),
        }
        cls.overlap_meshes = {
            "casting first": build_tet_mesh(
                [(CASTING, PART_TYPE_CAST), (cls.SUNK_RISER, PART_TYPE_RISER)], ASSEMBLY_H
            ),
            "riser first": build_tet_mesh(
                [(cls.SUNK_RISER, PART_TYPE_RISER), (CASTING, PART_TYPE_CAST)], ASSEMBLY_H
            ),
        }

    def test_core_tets_removed_and_contact_faces_marked(self) -> None:
        for label, mesh in self.core_meshes.items():
            with self.subTest(order=label):
                self.assertFalse(np.any(mesh.tet_part_types == PART_TYPE_CORE))
                self.assertAlmostEqual(
                    mesh.tet_volumes.sum() / (self.BOX.volume - self.CORE.volume), 1.0, delta=0.01
                )
                core_faces = mesh.boundary_face_contact_codes == PART_TYPE_CORE
                self.assertTrue(np.any(core_faces))
                self.assertTrue(np.all(mesh.boundary_face_contact_codes[~core_faces] == -1))
                self.assertAlmostEqual(
                    mesh.boundary_face_areas[core_faces].sum() / self.CORE.area, 1.0, delta=0.02
                )
                # Outward from the metal means into the core cavity: normals face the axis.
                tri = mesh.node_xyz[mesh.boundary_faces[core_faces]]
                normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                to_center = np.array([0.0, 0.0, 1.0]) - tri.mean(axis=1)
                self.assertTrue(np.all(np.einsum("ij,ij->i", normals, to_center) > 0.0))
                np.testing.assert_allclose(_enclosed_volume(mesh), mesh.tet_volumes.sum(), rtol=1e-9)
                # Core-contact nodes are surface nodes.
                contact_nodes = np.unique(mesh.boundary_faces[core_faces])
                self.assertTrue(np.all(mesh.surface_voxels[contact_nodes]))
                self.assertTrue(np.all(np.isin(contact_nodes, mesh.get_material_surface_voxel_indices())))

    def test_overlap_highest_code_wins(self) -> None:
        overlap = self.SUNK_RISER.volume / 2.0
        for label, mesh in self.overlap_meshes.items():
            with self.subTest(order=label):
                self.assertAlmostEqual(
                    _code_volume(mesh, PART_TYPE_RISER) / self.SUNK_RISER.volume, 1.0, delta=0.01
                )
                self.assertAlmostEqual(
                    _code_volume(mesh, PART_TYPE_CAST) / (CASTING.volume - overlap), 1.0, delta=0.01
                )

    def test_only_core_raises(self) -> None:
        with self.assertRaises(MeshGenerationError):
            build_tet_mesh([(self.CORE, PART_TYPE_CORE)], 0.3)


class TetMeshModulusTest(unittest.TestCase):
    """voxel_moduli is the exact distance to the boundary surface."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh = build_tet_mesh([(_box((2.0, 2.0, 2.0), (0.0, 0.0, 0.0)), PART_TYPE_CAST)], 0.2)

    def test_zero_on_boundary_and_max_near_half_width(self) -> None:
        mesh = self.mesh
        self.assertTrue(np.all(mesh.voxel_moduli[mesh.surface_voxels] == 0.0))
        self.assertTrue(np.all(mesh.voxel_moduli[~mesh.surface_voxels] > 0.0))
        self.assertAlmostEqual(float(mesh.voxel_moduli.max()), 1.0, delta=0.1)

    def test_matches_box_distance_and_trimesh_proximity(self) -> None:
        mesh = self.mesh
        expected = 1.0 - np.abs(mesh.node_xyz).max(axis=1)
        interior = ~mesh.surface_voxels
        np.testing.assert_allclose(mesh.voxel_moduli[interior], expected[interior], atol=0.01)
        skin = trimesh.Trimesh(mesh.node_xyz, mesh.boundary_faces, process=False)
        sample = np.flatnonzero(interior)[::7][:300]
        _, distance, _ = trimesh.proximity.closest_point(skin, mesh.node_xyz[sample])
        np.testing.assert_allclose(mesh.voxel_moduli[sample], distance, atol=1e-5)


class TetMeshPlumbingTest(unittest.TestCase):
    """Import laziness, stdout silencing, and input validation."""

    def test_import_does_not_load_pyvista_or_vtk(self) -> None:
        code = (
            "import sys; import core.tet_mesh; "
            "bad = [m for m in ('pyvista', 'vtk', 'vtkmodules', 'pytetwild') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_native_output_is_silenced(self) -> None:
        parts = [(_box((1.0, 1.0, 1.0), (0.0, 0.0, 0.5)), PART_TYPE_CAST),
                 (_box((1.0, 1.0, 0.5), (0.0, 0.0, -0.25)), PART_TYPE_CHILL)]
        python_out = io.StringIO()
        sys.stdout.flush()
        saved = os.dup(1)
        with tempfile.TemporaryFile() as fd_out:
            try:
                os.dup2(fd_out.fileno(), 1)
                with redirect_stdout(python_out):
                    build_tet_mesh(parts, 0.3)
            finally:
                os.dup2(saved, 1)
                os.close(saved)
            fd_out.seek(0)
            self.assertEqual(fd_out.read(), b"")
        self.assertEqual(python_out.getvalue(), "")

    def test_invalid_input_raises(self) -> None:
        with self.assertRaises(MeshGenerationError):
            build_tet_mesh([], 0.2)
        with self.assertRaises(MeshGenerationError):
            build_tet_mesh([(CASTING, PART_TYPE_CAST)], 0.0)


if __name__ == "__main__":
    unittest.main()
