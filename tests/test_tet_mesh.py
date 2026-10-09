"""Tests for core/tet_mesh.py (contract C1 in docs/tet_migration/DESIGN.md)."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import connected_components

import core.tet_mesh as tet_mesh_module
from core import tetwild_child
from config.exceptions import MeshGenerationError
from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_RISER,
)
from core.tet_mesh import TetVolumeMesh, build_tet_mesh, winding_number_contains

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


def _open_cylinder(radius, height, center, *, open_end: str, sections=32) -> trimesh.Trimesh:
    """Cylinder with its ``"top"`` or ``"bottom"`` cap removed: an open, non-watertight surface."""
    mesh = _cylinder(radius, height, center, sections)
    normal_z = mesh.face_normals[:, 2]
    keep = normal_z < 0.99 if open_end == "top" else normal_z > -0.99
    return trimesh.Trimesh(mesh.vertices, mesh.faces[keep], process=True)


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
                actual = np.asarray(mesh.voxel_moduli[sample], dtype=np.float64)
                # _distance_to_boundary tests the k nearest-centroid faces, so it may
                # only overestimate, by a bounded amount (docstring: < ~0.6 h).
                # fTetWild meshes differ run to run; a rare node lands a few 1e-5 in
                # high. Pin: never below exact, nearly all exact, worst case tiny.
                self.assertTrue(np.all(actual >= distance - 1e-6))
                close = np.abs(actual - distance) <= 1e-5
                self.assertGreaterEqual(close.mean(), 0.98)
                self.assertLess(float(np.max(actual - distance)), 0.05 * mesh.base_voxel_size)

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
    """Casting box with a riser on top and a chill underneath, both touching (CSG strategy)."""

    STRATEGY = "csg"
    VOLUME_TOLERANCE = 0.01

    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh = build_tet_mesh(
            [(CASTING, PART_TYPE_CAST), (RISER, PART_TYPE_RISER), (CHILL, PART_TYPE_CHILL)],
            ASSEMBLY_H,
            strategy=cls.STRATEGY,
        )
        # Permuted input order: proves marker k maps to the k-th leaf.
        cls.permuted = build_tet_mesh(
            [(CHILL, PART_TYPE_CHILL), (CASTING, PART_TYPE_CAST), (RISER, PART_TYPE_RISER)],
            ASSEMBLY_H,
            strategy=cls.STRATEGY,
        )

    def test_strategy_recorded(self) -> None:
        self.assertEqual(self.mesh.meshing_strategy, self.STRATEGY)
        self.assertEqual(self.permuted.meshing_strategy, self.STRATEGY)

    def test_per_code_volumes(self) -> None:
        for label, mesh in (("input order", self.mesh), ("permuted", self.permuted)):
            for code, part in (
                (PART_TYPE_CAST, CASTING),
                (PART_TYPE_RISER, RISER),
                (PART_TYPE_CHILL, CHILL),
            ):
                with self.subTest(order=label, code=code):
                    self.assertAlmostEqual(
                        _code_volume(mesh, code) / part.volume, 1.0, delta=self.VOLUME_TOLERANCE
                    )

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


class TetMeshSoupAssemblyTest(TetMeshAssemblyTest):
    """The same assembly meshed with the soup strategy (the CSG crash fallback).

    Soup meshes the concatenated surfaces with fTetWild's outside filter off
    and labels tets by CAD containment, so it must keep the conforming
    interfaces and the per-part volumes.
    """

    STRATEGY = "soup"
    VOLUME_TOLERANCE = 0.02

    def test_every_kept_tet_lies_inside_a_part(self) -> None:
        for label, mesh in (("input order", self.mesh), ("permuted", self.permuted)):
            with self.subTest(order=label):
                centroids = mesh.node_xyz[mesh.tets].mean(axis=1)
                inside = np.zeros(len(centroids), dtype=bool)
                for part in (CASTING, RISER, CHILL):
                    inside |= part.contains(centroids)
                self.assertTrue(np.all(inside))

    def test_boundary_faces_lie_on_a_part_surface(self) -> None:
        """Every boundary face is within the fTetWild envelope of some part surface."""
        parts = (CASTING, RISER, CHILL)
        assembly = trimesh.util.concatenate(parts)
        envelope = 1e-3 * float(np.linalg.norm(assembly.bounds[1] - assembly.bounds[0]))
        for label, mesh in (("input order", self.mesh), ("permuted", self.permuted)):
            with self.subTest(order=label):
                points = np.vstack(
                    [mesh.node_xyz[np.unique(mesh.boundary_faces)], mesh.node_xyz[mesh.boundary_faces].mean(axis=1)]
                )
                distance = np.min(
                    [trimesh.proximity.closest_point(part, points)[1] for part in parts], axis=0
                )
                self.assertLess(float(distance.max()), 2.0 * envelope)


class TetMeshSoupCoreTest(unittest.TestCase):
    """CORE removal works on the soup strategy too."""

    def test_core_removed_with_contact_faces(self) -> None:
        box = _box((2.0, 2.0, 2.0), (0.0, 0.0, 1.0))
        core = _cylinder(0.5, 1.0, (0.0, 0.0, 1.0), sections=24)
        mesh = build_tet_mesh([(box, PART_TYPE_CAST), (core, PART_TYPE_CORE)], 0.2, strategy="soup")
        self.assertEqual(mesh.meshing_strategy, "soup")
        self.assertFalse(np.any(mesh.tet_part_types == PART_TYPE_CORE))
        self.assertAlmostEqual(mesh.tet_volumes.sum() / (box.volume - core.volume), 1.0, delta=0.01)
        core_faces = mesh.boundary_face_contact_codes == PART_TYPE_CORE
        self.assertAlmostEqual(mesh.boundary_face_areas[core_faces].sum() / core.area, 1.0, delta=0.02)
        np.testing.assert_allclose(_enclosed_volume(mesh), mesh.tet_volumes.sum(), rtol=1e-9)


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
    """Import laziness, native stdout silencing, and input validation."""

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
        with self.assertRaises(ValueError):
            build_tet_mesh([(CASTING, PART_TYPE_CAST)], 0.2, strategy="tetgen")


SMALL_PARTS = [
    (_box((1.0, 1.0, 1.0), (0.0, 0.0, 0.5)), PART_TYPE_CAST),
    (_box((1.0, 1.0, 0.5), (0.0, 0.0, -0.25)), PART_TYPE_CHILL),
]


class _CancelRequested(Exception):
    pass


class TetwildChildProcessTest(unittest.TestCase):
    """fTetWild runs in ``core/tetwild_child.py``: crashes fall back or raise, cancel kills the child."""

    def _crash_in(self, modes: str):
        return mock.patch.dict(os.environ, {tetwild_child.FAULT_ENV: modes})

    def test_child_script_imports_only_stdlib_numpy_and_pytetwild(self) -> None:
        import ast

        tree = ast.parse((REPO / "core" / "tetwild_child.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        third_party = {m for m in imported if m not in sys.stdlib_module_names and m != "__future__"}
        self.assertEqual(third_party, {"numpy", "pytetwild"})

    def test_csg_crash_falls_back_to_soup(self) -> None:
        messages: list[str] = []
        with self._crash_in("csg"), self.assertLogs("core.tet_mesh", level="WARNING") as logs:
            mesh = build_tet_mesh(SMALL_PARTS, 0.3, progress_callback=lambda f, m: messages.append(m))
        # The test process survived a SIGSEGV in the mesher.
        self.assertEqual(mesh.meshing_strategy, "soup")
        self.assertIn("SIGSEGV", "\n".join(logs.output))
        self.assertTrue(any("retrying with merged part surfaces" in m for m in messages))
        self.assertIn("soup strategy", messages[-1])
        for (part, code) in SMALL_PARTS:
            self.assertAlmostEqual(_code_volume(mesh, code) / part.volume, 1.0, delta=0.02)

    def test_soup_crash_is_retried_once(self) -> None:
        real = tet_mesh_module._tetrahedralize_in_child
        calls: list[str] = []

        def crash_first_soup(mode, *args):
            calls.append(mode)
            if mode == "csg" or calls.count("soup") == 1:
                raise tet_mesh_module._ChildFailure("crashed with SIGBUS", "")
            return real(mode, *args)

        with mock.patch.object(
            tet_mesh_module, "_tetrahedralize_in_child", side_effect=crash_first_soup
        ), self.assertLogs("core.tet_mesh", level="INFO") as logs:
            mesh = build_tet_mesh(SMALL_PARTS, 0.3)
        self.assertEqual(calls, list(tet_mesh_module.MESHING_ATTEMPTS["auto"]))
        self.assertEqual(calls, ["csg", "soup", "soup"])
        self.assertEqual(mesh.meshing_strategy, "soup")
        output = "\n".join(logs.output)
        self.assertIn("attempt 1/3 (csg) failed", output)
        self.assertIn("attempt 2/3 (soup) failed", output)
        self.assertIn("attempt 3/3 (soup) succeeded", output)

    def test_every_strategy_crashing_raises_with_the_voxel_hint(self) -> None:
        cases = {"assembly": (SMALL_PARTS, "csg,soup"), "single part": (SMALL_PARTS[:1], "soup")}
        for label, (parts, modes) in cases.items():
            with self.subTest(label), self._crash_in(modes), self.assertLogs("core.tet_mesh", level="WARNING"):
                with self.assertRaises(MeshGenerationError) as raised:
                    build_tet_mesh(parts, 0.3)
                message = str(raised.exception)
                self.assertIn("SIGSEGV", message)
                self.assertIn("edge length 0.3 in", message)
                self.assertIn("Change the mesh size, or switch Mesh type to Voxel", message)

    def test_child_python_error_reports_exit_code_and_stderr(self) -> None:
        with mock.patch.object(tet_mesh_module, "_CHILD_SCRIPT", str(REPO / "core" / "no_such_child.py")):
            with self.assertLogs("core.tet_mesh", level="WARNING"), self.assertRaises(MeshGenerationError) as raised:
                build_tet_mesh(SMALL_PARTS, 0.3, strategy="csg")
        self.assertIn("exited with code", str(raised.exception))
        self.assertIn("no_such_child.py", str(raised.exception))

    def test_cancel_kills_the_running_child_promptly(self) -> None:
        sphere = trimesh.creation.icosphere(subdivisions=4, radius=2.0)
        started = time.monotonic()
        cancel_at: list[float] = []
        processes: list = []
        real_popen = subprocess.Popen

        def popen(*args, **kwargs):
            processes.append(real_popen(*args, **kwargs))
            return processes[-1]

        def progress(fraction: float, message: str) -> None:
            if time.monotonic() - started > 2.0:
                cancel_at.append(time.monotonic())
                raise _CancelRequested

        with mock.patch.object(tet_mesh_module.subprocess, "Popen", side_effect=popen):
            with self.assertRaises(_CancelRequested):
                build_tet_mesh([(sphere, PART_TYPE_CAST)], 0.04, progress_callback=progress)
        self.assertLess(time.monotonic() - cancel_at[0], 2.0)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())
        self.assertFalse(tet_mesh_module._NATIVE_MESH_LOCK.locked())

    def test_two_threads_meshing_concurrently_leave_stdout_intact(self) -> None:
        results: list = []
        errors: list = []

        def mesh() -> None:
            try:
                results.append(build_tet_mesh(SMALL_PARTS, 0.3).get_voxel_count())
            except Exception as exc:  # reported below
                errors.append(exc)

        python_out = io.StringIO()
        sys.stdout.flush()
        saved = os.dup(1)
        with tempfile.TemporaryFile() as fd_out:
            try:
                os.dup2(fd_out.fileno(), 1)
                with redirect_stdout(python_out):
                    threads = [threading.Thread(target=mesh) for _ in range(2)]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                    os.write(1, b"fd1 works")
            finally:
                os.dup2(saved, 1)
                os.close(saved)
            fd_out.seek(0)
            self.assertEqual(fd_out.read(), b"fd1 works")
        self.assertEqual(python_out.getvalue(), "")
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)


class OpenPartOverlapTest(unittest.TestCase):
    """Overlap pass on a non-watertight part uses the generalized winding number."""

    # Risers sunk 1 in into the casting box (z 1..3, casting z 0..2).
    OPEN_RISER = _open_cylinder(0.8, 2.0, (0.0, 0.0, 2.0), open_end="top")
    # Missing cap inside the casting: ray tests fail in the overlap itself.
    OPEN_BOTTOM_RISER = _open_cylinder(0.8, 2.0, (0.0, 0.0, 2.0), open_end="bottom")

    def test_winding_number_matches_exact_containment_where_ray_tests_fail(self) -> None:
        riser = self.OPEN_RISER
        self.assertFalse(riser.is_watertight)
        points = np.random.default_rng(0).uniform([-1.0, -1.0, 0.0], [1.0, 1.0, 3.0], (20000, 3))
        exact = (np.hypot(points[:, 0], points[:, 1]) < 0.8) & (points[:, 2] > 1.0) & (points[:, 2] < 3.0)
        winding_wrong = int(np.sum(winding_number_contains(riser, points) != exact))
        ray_wrong = int(np.sum(riser.contains(points) != exact))
        # Measured: 47 (faceting of the 32-gon) vs 1933 of 6737 inside points.
        self.assertLess(winding_wrong, 0.02 * len(points))
        self.assertGreater(ray_wrong, 0.05 * len(points))

    def test_open_riser_overlapping_a_casting_takes_the_riser_code(self) -> None:
        """Measured with ``Trimesh.contains``: riser volume 0.716 of exact in riser-first order."""
        riser_volume = np.pi * 0.8**2 * 2.0
        for order in ("casting first", "riser first"):
            parts = [(CASTING, PART_TYPE_CAST), (self.OPEN_BOTTOM_RISER, PART_TYPE_RISER)]
            if order == "riser first":
                parts.reverse()
            with self.subTest(order=order):
                mesh = build_tet_mesh(parts, ASSEMBLY_H)
                self.assertAlmostEqual(_code_volume(mesh, PART_TYPE_RISER) / riser_volume, 1.0, delta=0.01)
                self.assertAlmostEqual(
                    _code_volume(mesh, PART_TYPE_CAST) / (CASTING.volume - riser_volume / 2.0),
                    1.0,
                    delta=0.01,
                )

    def test_watertight_parts_keep_the_ray_test(self) -> None:
        sunk = _cylinder(0.8, 2.0, (0.0, 0.0, 2.0))
        with mock.patch.object(tet_mesh_module, "winding_number_contains") as winding:
            build_tet_mesh([(CASTING, PART_TYPE_CAST), (sunk, PART_TYPE_RISER)], 0.4)
        winding.assert_not_called()

    def test_work_above_the_pair_cap_falls_back_with_a_warning(self) -> None:
        points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 5.0]])
        with mock.patch.object(tet_mesh_module, "WINDING_NUMBER_MAX_PAIRS", 1), mock.patch(
            "core.voxel_mesh._robust_contains_check_impl", return_value=np.array([True, False])
        ) as robust, self.assertLogs("core.tet_mesh", level="WARNING"):
            inside = winding_number_contains(self.OPEN_RISER, points)
        robust.assert_called_once()
        np.testing.assert_array_equal(inside, [True, False])


class DisjointAssemblyTest(unittest.TestCase):
    """Parts that do not touch become separate components, each fully bounded."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cast = _box((2.0, 2.0, 1.0), (0.0, 0.0, 0.5))
        cls.riser = _box((1.0, 1.0, 1.0), (3.0, 0.0, 0.5))  # 1 in gap along x
        cls.mesh = build_tet_mesh([(cls.cast, PART_TYPE_CAST), (cls.riser, PART_TYPE_RISER)], 0.25)

    def test_two_components_each_closed_by_boundary_faces(self) -> None:
        mesh = self.mesh
        n = mesh.get_voxel_count()
        adjacency = sp.csr_matrix(
            (np.ones(len(mesh.adj_indices)), mesh.adj_indices, mesh.adj_indptr), shape=(n, n)
        )
        count, labels = connected_components(adjacency, directed=False)
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(mesh.boundary_face_contact_codes, -1)
        for component in range(2):
            with self.subTest(component=component):
                tets = np.all(labels[mesh.tets] == component, axis=1)
                faces = np.all(labels[mesh.boundary_faces] == component, axis=1)
                codes = set(np.unique(mesh.tet_part_types[tets]).tolist())
                self.assertEqual(len(codes), 1)
                part = self.cast if codes == {PART_TYPE_CAST} else self.riser
                self.assertAlmostEqual(mesh.boundary_face_areas[faces].sum() / part.area, 1.0, delta=0.01)
                self.assertTrue(np.all(mesh.boundary_face_part_types[faces] == codes.pop()))


if __name__ == "__main__":
    unittest.main()
