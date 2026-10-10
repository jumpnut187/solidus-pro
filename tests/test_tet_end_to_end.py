"""run_pipeline end to end with mesh_type="tet" (wave 2 integration, docs/tet_migration).

Real geometry, real tet mesher, real engine, default analysis options:

1. ``sample_stl_files/cylinder.stl`` loaded through ``STLLoader``.
2. A ``PartManager`` assembly of casting + riser + chill built from trimesh
   primitives.
3. The same assembly with mold filling and a ``PourLocation``.
4. The same assembly on the voxel path, as a cross-check of the hot spot.

Every engine per-point array must have one entry per tet node (DESIGN.md
decision 5).
"""

from __future__ import annotations

import sys
import time
import unittest
from unittest import mock

import numpy as np
import trimesh

from config.settings import Settings
from core import simulation_pipeline as sp
from core import tet_thermal
from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_RISER,
    Part,
    PartManager,
    PartType,
    is_metal,
)
from core.pour_location import PourLocation
from core.stl_loader import STLLoader
from core.tet_mesh import TetVolumeMesh

EDGE_LENGTH_IN = 0.2
CYLINDER_STL = "sample_stl_files/cylinder.stl"
# Top face of the riser cylinder in ``_assembly`` is z = 2.2 in.
RISER_POUR = PourLocation(point_in=(0.0, 0.0, 2.15))


def _add_part(manager: PartManager, name: str, mesh: trimesh.Trimesh, part_type: PartType) -> None:
    """Register a mesh as a part without touching disk (as tests/test_desktop_orientation.py does)."""
    part = Part(id=name, name=name, filepath=f"{name}.stl", trimesh_obj=mesh)
    part.set_part_type(part_type)
    manager.parts[name] = part
    manager._part_order.append(name)


def _assembly() -> PartManager:
    """2 x 2 x 1 in casting, 1.2 in dia riser on top, 1 x 1 x 0.4 in chill under one half."""
    manager = PartManager()
    cast = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
    cast.apply_translation((0.0, 0.0, 0.5))
    riser = trimesh.creation.cylinder(radius=0.6, height=1.2, sections=48)
    riser.apply_translation((0.0, 0.0, 1.6))
    chill = trimesh.creation.box(extents=(1.0, 1.0, 0.4))
    chill.apply_translation((0.5, 0.0, -0.2))
    _add_part(manager, "cast", cast, PartType.CAST)
    _add_part(manager, "riser", riser, PartType.RISER)
    _add_part(manager, "chill", chill, PartType.CHILL)
    return manager


def run_settings() -> Settings:
    """Default settings with edge factor 1.0, so ``EDGE_LENGTH_IN`` is the tet edge length."""
    settings = Settings()
    settings.mesh.tet_edge_length_factor = 1.0
    return settings


def _run(mesh_type: str, **kwargs):
    """run_pipeline with default settings (no user config files) and default analysis options."""
    pipeline_kwargs = {k: kwargs.pop(k) for k in ("part_manager", "stl_loader") if k in kwargs}
    inputs = sp.PipelineInputs("WCB", EDGE_LENGTH_IN, 0, mesh_type=mesh_type, **kwargs)
    settings = run_settings()
    started = time.perf_counter()
    with mock.patch.object(sp, "get_settings", return_value=settings), mock.patch.object(
        tet_thermal, "get_settings", return_value=settings
    ):
        result = sp.run_pipeline(inputs, **pipeline_kwargs)
    return result, time.perf_counter() - started


def _cast_hot_spot(result) -> np.ndarray:
    """Coordinates (in) of the CAST point with the largest solidification time."""
    part_types = np.asarray(result.voxel_mesh.voxel_part_types)
    times = np.asarray(result.engine.solidification_times, dtype=np.float64)
    cast = np.flatnonzero(part_types == PART_TYPE_CAST)
    return np.asarray(result.voxel_mesh.voxel_centers, dtype=np.float64)[cast[np.nanargmax(times[cast])]]


def _summary(label: str, result, wall_s: float) -> str:
    mesh = result.voxel_mesh
    times = np.asarray(result.engine.solidification_times, dtype=np.float64)
    metal = is_metal(np.asarray(mesh.voxel_part_types, dtype=np.int32))
    x, y, z = _cast_hot_spot(result)
    return (
        f"{label}: {mesh.get_voxel_count()} points, wall {wall_s:.1f} s, "
        f"total solidification {np.nanmax(times[metal]):.1f} s, "
        f"cast hot spot ({x:.2f}, {y:.2f}, {z:.2f}) in"
    )


class _TetRunAssertions(unittest.TestCase):
    def assert_tet_run_complete(self, result) -> None:
        mesh = result.voxel_mesh
        engine = result.engine
        self.assertIsInstance(mesh, TetVolumeMesh)
        self.assertEqual(result.mesh_type, "tet")
        n = mesh.get_voxel_count()
        self.assertEqual(len(mesh.node_xyz), n)

        # Every per-point engine array is per node.
        self.assertEqual(np.asarray(mesh.voxel_states).shape, (n,))
        checked = []
        for name, value in vars(engine).items():
            if isinstance(value, np.ndarray) and value.ndim >= 1:
                self.assertEqual(value.shape[0], n, f"engine.{name} has shape {value.shape}")
                checked.append(name)
        self.assertIn("solidification_times", checked)
        self.assertIn("thermal_gradients", checked)
        for snapshot in engine.fraction_liquid_history or []:
            self.assertEqual(np.asarray(snapshot).shape, (n,))
        for name, value in engine.export_results().items():
            if isinstance(value, np.ndarray) and value.ndim >= 1:
                self.assertEqual(value.shape[0], n, f"export_results()[{name!r}] has shape {value.shape}")

        # Every metal node finished solidifying.
        metal = is_metal(np.asarray(mesh.voxel_part_types, dtype=np.int32))
        self.assertTrue(np.any(metal))
        times = np.asarray(engine.solidification_times, dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(times[metal])), "unfinished metal nodes")
        self.assertTrue(np.all(times[metal] > 0.0))

        visualization = result.visualization_data
        self.assertEqual(visualization["mesh_type"], "tet")
        self.assertEqual(len(visualization["centers"]), n)
        self.assertEqual(len(visualization["solidification_times"]), n)
        self.assertEqual(result.result_data["voxel_count"], n)
        self.assertEqual(result.result_data["tet_count"], len(mesh.tets))


class CylinderTetEndToEndTests(_TetRunAssertions):
    @classmethod
    def setUpClass(cls) -> None:
        loader = STLLoader()
        if not loader.load_stl(CYLINDER_STL):
            raise RuntimeError(f"could not load {CYLINDER_STL}")
        cls.result, cls.wall_s = _run("tet", stl_loader=loader)
        print(f"\n  {_summary('tet cylinder', cls.result, cls.wall_s)}", file=sys.stderr)

    def test_run_completes_with_node_arrays(self) -> None:
        self.assert_tet_run_complete(self.result)
        self.assertEqual(set(np.unique(self.result.voxel_mesh.tet_part_types)), {PART_TYPE_CAST})

    def test_hot_spot_is_on_the_axis_inside_the_cylinder(self) -> None:
        # Cylinder r = 1 in, z = 0..3 in, cooled on every face.
        x, y, z = _cast_hot_spot(self.result)
        self.assertLess(float(np.hypot(x, y)), 0.5)
        self.assertGreater(z, 0.75)
        self.assertLess(z, 2.25)


class AssemblyTetEndToEndTests(_TetRunAssertions):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result, cls.wall_s = _run("tet", part_manager=_assembly())
        print(f"\n  {_summary('tet assembly', cls.result, cls.wall_s)}", file=sys.stderr)

    def test_run_completes_with_node_arrays(self) -> None:
        self.assert_tet_run_complete(self.result)
        codes = set(np.unique(self.result.voxel_mesh.tet_part_types))
        self.assertEqual(codes, {PART_TYPE_CAST, PART_TYPE_RISER, PART_TYPE_CHILL})

    def test_chill_nodes_start_solid(self) -> None:
        mesh = self.result.voxel_mesh
        chill = np.asarray(mesh.voxel_part_types) == PART_TYPE_CHILL
        self.assertTrue(np.any(chill))
        np.testing.assert_array_equal(self.result.engine.solidification_start_times[chill], 0.0)

    def test_riser_freezes_after_the_casting(self) -> None:
        part_types = np.asarray(self.result.voxel_mesh.voxel_part_types)
        times = np.asarray(self.result.engine.solidification_times, dtype=np.float64)
        self.assertGreater(np.max(times[part_types == PART_TYPE_RISER]), np.max(times[part_types == PART_TYPE_CAST]))

    def test_cast_hot_spot_is_under_the_riser(self) -> None:
        x, y, z = _cast_hot_spot(self.result)
        self.assertLess(float(np.hypot(x, y)), 0.6)  # riser radius
        self.assertGreater(z, 0.5)

    def test_voxel_path_puts_the_cast_hot_spot_in_the_same_place(self) -> None:
        voxel, wall_s = _run("voxel", part_manager=_assembly())
        print(f"\n  {_summary('voxel assembly', voxel, wall_s)}", file=sys.stderr)
        self.assertEqual(voxel.mesh_type, "voxel")
        distance = float(np.linalg.norm(_cast_hot_spot(voxel) - _cast_hot_spot(self.result)))
        self.assertLess(distance, 2.0 * EDGE_LENGTH_IN)


class AssemblyTetFillingEndToEndTests(_TetRunAssertions):
    @classmethod
    def setUpClass(cls) -> None:
        cls.messages: list = []
        loaded = _assembly()
        # Default filling parameters: the fill finishes in about 1 s of pour
        # on an auto-coarsened (x2) lattice. A fill cut short by
        # max_fill_time_s seeds misrun nodes at mold temperature, which moves
        # the hot spot, so the cap is left at its default.
        inputs = {
            "analysis_options": {"filling": True},
            "pour_location": RISER_POUR.to_dict(),
        }
        settings = run_settings()
        started = time.perf_counter()
        with mock.patch.object(sp, "get_settings", return_value=settings), mock.patch.object(
            tet_thermal, "get_settings", return_value=settings
        ):
            cls.result = sp.run_pipeline(
                sp.PipelineInputs("WCB", EDGE_LENGTH_IN, 0, mesh_type="tet", **inputs),
                part_manager=loaded,
                progress_message_callback=cls.messages.append,
            )
        cls.wall_s = time.perf_counter() - started
        print(f"\n  {_summary('tet assembly + filling', cls.result, cls.wall_s)}", file=sys.stderr)

    def test_run_completes_with_node_arrays(self) -> None:
        self.assert_tet_run_complete(self.result)

    def test_fill_seeds_the_solve_with_node_fields(self) -> None:
        engine = self.result.engine
        n = self.result.voxel_mesh.get_voxel_count()
        self.assertTrue(any("seeded initial temperature" in m for m in self.messages), self.messages)
        self.assertEqual(engine._initial_temperature_field.shape, (n,))
        self.assertTrue(np.all(np.isfinite(engine._initial_temperature_field)))
        self.assertEqual(engine._fill_arrival_times.shape, (n,))
        self.assertTrue(engine._fill_frames)
        self.assertTrue(all(len(frame.fill_fraction) == n for frame in engine._fill_frames))
        self.assertEqual(len(self.result.visualization_data["fill_time"]), n)
        self.assertEqual(len(self.result.visualization_data["fill_animation_history"][0]), n)

    def test_the_fill_reaches_the_metal_and_seeds_it_hot(self) -> None:
        mesh = self.result.voxel_mesh
        engine = self.result.engine
        metal = is_metal(np.asarray(mesh.voxel_part_types, dtype=np.int32))
        arrival = np.asarray(engine._fill_arrival_times, dtype=np.float64)
        self.assertGreaterEqual(float(np.mean(arrival[metal] >= 0.0)), 0.95)
        seeded = engine._initial_temperature_field[metal & (arrival >= 0.0)]
        self.assertGreater(float(np.min(seeded)), float(engine.alloy.liquidus_temp))
        self.assertLessEqual(float(np.max(seeded)), float(engine.alloy.pouring_temp) + 1e-6)

    def test_cast_hot_spot_is_under_the_riser(self) -> None:
        x, y, z = _cast_hot_spot(self.result)
        self.assertLess(float(np.hypot(x, y)), 0.6)
        self.assertGreater(z, 0.5)


if __name__ == "__main__":
    unittest.main()
