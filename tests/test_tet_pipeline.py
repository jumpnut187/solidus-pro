"""Tests for the tet pipeline seams (contract C5 in docs/tet_migration/DESIGN.md).

Covers ``MeshSettings`` tet fields, mesh selection in
``core.simulation_pipeline``, the node-count guard, ``mesh_type`` in the
payload, the lattice filling seam in ``core.tet_filling``, and
``run_pipeline`` / ``run_surrogate_pipeline`` on a tet mesh with the real
``SolidificationEngine`` (contract C4).
"""

from __future__ import annotations

import os
import sys
import time
import tomllib
import unittest
import warnings
from dataclasses import fields
from typing import Any, List
from unittest import mock

import numpy as np
import trimesh
import yaml

from config.exceptions import MeshGenerationError, ValidationError
from config.settings import (
    MeshSettings,
    Settings,
    TOML_TEMPLATE,
    YAML_TEMPLATE,
)
from core import simulation_pipeline as sp
from core.filling_stage import FillFrame, FillingResult, build_grid_mapping
from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_FILTER,
    PART_TYPE_SPRUE,
    is_metal,
)
from core.pour_location import PourLocation
from core.tet_filling import (
    FillingLatticeMesh,
    build_filling_lattice,
    map_filling_result_to_nodes,
    maybe_run_tet_filling,
    node_lattice_cells,
)
from core.tet_mesh import TetVolumeMesh, build_tet_mesh
from core.voxel_mesh import AdaptiveVoxelMesh


def _box(extents, center) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


class _StlLoader:
    """The two members ``prepare_mesh_and_geometry`` reads from an STL loader."""

    def __init__(self, mesh: trimesh.Trimesh) -> None:
        self.trimesh_obj = mesh
        self.volume = float(mesh.volume)

    def get_bounds_array(self) -> np.ndarray:
        return np.asarray(self.trimesh_obj.bounds).T


class _PartManager:
    """The ``PartManager`` members ``prepare_mesh_and_geometry`` reads."""

    def __init__(self, parts) -> None:
        self._parts = list(parts)

    def get_part_count(self) -> int:
        return len(self._parts)

    def get_combined_mesh(self) -> trimesh.Trimesh:
        return trimesh.util.concatenate([m for m, _ in self._parts])

    def get_combined_bounds(self) -> np.ndarray:
        return np.asarray(self.get_combined_mesh().bounds).T

    def get_total_volume(self) -> float:
        return float(sum(m.volume for m, _ in self._parts))

    def get_parts_with_types(self):
        return list(self._parts)


def _settings(**mesh_fields: Any) -> Settings:
    settings = Settings()
    for name, value in mesh_fields.items():
        setattr(settings.mesh, name, value)
    settings.mesh.normalize_tet_fields()
    return settings


# Filling option stand-ins (same shape as tests/test_filling_stage.py, own
# class attributes so the fill time can be set without touching that module).


class _FilterParams:
    porosity = 0.85
    pore_diameter_mm = 2.54
    ergun_prefactor = 2.2
    forchheimer_c_f = 0.55
    contact_angle_deg = 120.0
    window_fraction = 0.45


class _FillingParams:
    fill_grid_coarsening = 1
    lattice_velocity_max = 0.05
    max_fill_time_s = 0.2
    pour_temperature_c = 0.0
    initial_head_in = 4.0
    filter = _FilterParams()


class _ThermalBoundary:
    mold_temperature = 21.0


class _Params:
    filling = _FillingParams()
    thermal_boundary = _ThermalBoundary()


class _Options:
    def __init__(self, filling: bool = True) -> None:
        self.filling = filling
        self.solver_mode = "thermal_coupled"
        self.solidification_params = _Params()


class _Inputs:
    def __init__(self, pour_location=None) -> None:
        self.pour_location = pour_location


class _Alloy:
    dynamic_viscosity = 0.006
    density_liquid = 7200.0
    pouring_temp = 1600.0
    surface_tension = 1.85
    thermal_conductivity = 30.0
    specific_heat = 600.0
    liquidus_temp = 1450.0


class _PayloadEngine:
    """The engine arrays ``build_visualization_payload`` reads, for payload unit tests."""

    def __init__(self, n: int) -> None:
        self.time = 0.0
        self.shrinkage_probability = np.zeros(n, dtype=np.float32)
        self.hotspot_intensity = np.zeros(n, dtype=np.float32)
        self.niyama_values = None
        self.solidification_times = np.zeros(n, dtype=np.float32)
        self.fraction_liquid_history: List[np.ndarray] = []
        self.fraction_liquid_times: List[float] = []


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class MeshSettingsTetFieldTests(unittest.TestCase):
    def test_defaults(self) -> None:
        mesh = MeshSettings()
        self.assertEqual(mesh.mesh_type, "tet")
        self.assertEqual(mesh.tet_edge_length_factor, 2.0)
        self.assertEqual(mesh.tet_epsilon_rel, 1e-3)
        self.assertEqual(mesh.tet_max_time_step_s, 2.0)
        self.assertEqual(mesh.tet_max_delta_t_c, 15.0)
        self.assertEqual(mesh.tet_lattice_spacing_factor, 1.0)
        self.assertEqual(mesh.tet_max_nodes, 3_000_000)

    def test_mesh_type_is_case_insensitive(self) -> None:
        self.assertEqual(MeshSettings(mesh_type=" TET ").mesh_type, "tet")

    def test_unknown_mesh_type_falls_back_to_tet_with_a_warning(self) -> None:
        with self.assertWarns(RuntimeWarning):
            mesh = MeshSettings(mesh_type="hex")
        self.assertEqual(mesh.mesh_type, "tet")

    def test_non_positive_tunables_fall_back_to_defaults(self) -> None:
        for name, bad in (
            ("tet_edge_length_factor", 0.0),
            ("tet_epsilon_rel", -1e-3),
            ("tet_max_time_step_s", float("nan")),
            ("tet_max_delta_t_c", "abc"),
            ("tet_lattice_spacing_factor", -2.0),
            ("tet_max_nodes", 0),
        ):
            with self.subTest(name=name):
                default = getattr(MeshSettings(), name)
                with self.assertWarns(RuntimeWarning):
                    mesh = MeshSettings(**{name: bad})
                self.assertEqual(getattr(mesh, name), default)

    def test_valid_values_are_kept_without_a_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mesh = MeshSettings(mesh_type="tet", tet_edge_length_factor=0.5, tet_max_nodes=10)
        self.assertEqual((mesh.mesh_type, mesh.tet_edge_length_factor, mesh.tet_max_nodes), ("tet", 0.5, 10))

    def test_environment_overrides(self) -> None:
        env = {
            "CASTSIM_MESH_TYPE": "voxel",
            "CASTSIM_TET_EDGE_LENGTH_FACTOR": "0.75",
            "CASTSIM_TET_MAX_TIME_STEP_S": "1.5",
            "CASTSIM_TET_MAX_DELTA_T_C": "5",
            "CASTSIM_TET_LATTICE_SPACING_FACTOR": "2",
        }
        settings = Settings()
        with mock.patch.dict(os.environ, env):
            settings._apply_environment_variables()
        self.assertEqual(settings.mesh.mesh_type, "voxel")
        self.assertEqual(settings.mesh.tet_edge_length_factor, 0.75)
        self.assertEqual(settings.mesh.tet_max_time_step_s, 1.5)
        self.assertEqual(settings.mesh.tet_max_delta_t_c, 5.0)
        self.assertEqual(settings.mesh.tet_lattice_spacing_factor, 2.0)
        for name in env:
            self.assertIn(name, settings._env_vars_applied)

    def test_invalid_environment_values_are_normalized_after_load(self) -> None:
        settings = Settings()
        with mock.patch.dict(
            os.environ, {"CASTSIM_MESH_TYPE": "hex", "CASTSIM_TET_EDGE_LENGTH_FACTOR": "-1"}
        ), self.assertWarns(RuntimeWarning):
            settings._apply_environment_variables()
        self.assertEqual(settings.mesh.mesh_type, "tet")
        self.assertEqual(settings.mesh.tet_edge_length_factor, 2.0)

    def test_config_file_values_apply(self) -> None:
        settings = Settings()
        settings._apply_config_dict({"mesh": {"mesh_type": "voxel", "tet_max_nodes": 1234}})
        self.assertEqual(settings.mesh.mesh_type, "voxel")
        self.assertEqual(settings.mesh.tet_max_nodes, 1234)

    def test_templates_carry_every_tet_field_at_its_default(self) -> None:
        defaults = MeshSettings()
        names = [f.name for f in fields(MeshSettings) if f.name == "mesh_type" or f.name.startswith("tet_")]
        for label, data in (
            ("yaml", yaml.safe_load(YAML_TEMPLATE)),
            ("toml", tomllib.loads(TOML_TEMPLATE)),
        ):
            for name in names:
                with self.subTest(template=label, field=name):
                    self.assertEqual(data["mesh"][name], getattr(defaults, name))

    def test_to_dict_serializes_the_tet_fields(self) -> None:
        mesh = Settings().to_dict()["mesh"]
        self.assertEqual(mesh["mesh_type"], "tet")
        self.assertIn("tet_lattice_spacing_factor", mesh)


# ---------------------------------------------------------------------------
# Mesh selection and the node-count guard
# ---------------------------------------------------------------------------


class MeshTypeResolutionTests(unittest.TestCase):
    def _inputs(self, mesh_type=None) -> sp.PipelineInputs:
        return sp.PipelineInputs("WCB", 0.25, 0, mesh_type=mesh_type)

    def test_inputs_default_to_none(self) -> None:
        self.assertIsNone(sp.PipelineInputs("WCB", 0.25, 0).mesh_type)

    def test_none_uses_settings(self) -> None:
        self.assertEqual(sp.resolve_mesh_type(self._inputs(), _settings(mesh_type="voxel").mesh), "voxel")
        self.assertEqual(sp.resolve_mesh_type(self._inputs(), _settings().mesh), "tet")

    def test_explicit_inputs_win_over_settings(self) -> None:
        self.assertEqual(sp.resolve_mesh_type(self._inputs("voxel"), _settings(mesh_type="tet").mesh), "voxel")
        self.assertEqual(sp.resolve_mesh_type(self._inputs("Tet"), _settings(mesh_type="voxel").mesh), "tet")

    def test_explicit_unknown_value_raises(self) -> None:
        with self.assertRaises(ValidationError):
            sp.resolve_mesh_type(self._inputs("hex"), _settings().mesh)


class PrepareTetMeshTests(unittest.TestCase):
    BOX = _box((2.0, 1.0, 1.0), (0.0, 0.0, 0.5))

    def _prepare(self, inputs, settings, **kwargs):
        with mock.patch.object(sp, "get_settings", return_value=settings):
            return sp.prepare_mesh_and_geometry(inputs, part_manager=None, stl_loader=_StlLoader(self.BOX), **kwargs)

    def test_tet_inputs_build_a_tet_mesh_with_progress(self) -> None:
        messages: list = []
        progress: list = []
        result = self._prepare(
            sp.PipelineInputs("WCB", 0.25, 0, mesh_type="tet"),
            _settings(),
            progress_message_callback=messages.append,
            detailed_progress_callback=lambda c, t, s: progress.append((c, t, s)),
        )
        tet_mesh = result[4]
        self.assertIsInstance(tet_mesh, TetVolumeMesh)
        self.assertEqual(result[3], [])
        self.assertIn("ram_mb", result[5])
        self.assertTrue(progress and all(t == 100 for _, t, _ in progress))
        self.assertEqual(progress[-1][0], 100)
        self.assertTrue(any("Tet mesh:" in m for m in messages))

    def test_settings_select_tet_when_inputs_do_not(self) -> None:
        result = self._prepare(sp.PipelineInputs("WCB", 0.25, 0), _settings(mesh_type="tet"))
        self.assertIsInstance(result[4], TetVolumeMesh)

    def test_edge_length_and_epsilon_come_from_settings(self) -> None:
        settings = _settings(tet_edge_length_factor=1.2, tet_epsilon_rel=2e-3)
        with mock.patch.object(sp, "build_tet_mesh", wraps=build_tet_mesh) as built:
            self._prepare(sp.PipelineInputs("WCB", 0.2, 0, mesh_type="tet"), settings)
        args, kwargs = built.call_args
        self.assertEqual(args[0][0][1], PART_TYPE_CAST)
        self.assertAlmostEqual(args[1], 0.24)
        self.assertEqual(kwargs["epsilon_rel"], 2e-3)

    def test_part_manager_parts_are_meshed_with_their_codes(self) -> None:
        parts = [
            (_box((2.0, 2.0, 1.0), (0.0, 0.0, 0.5)), PART_TYPE_CAST),
            (_box((1.0, 1.0, 0.5), (0.0, 0.0, -0.25)), PART_TYPE_CHILL),
        ]
        with mock.patch.object(sp, "get_settings", return_value=_settings()):
            result = sp.prepare_mesh_and_geometry(
                sp.PipelineInputs("WCB", 0.25, 0, mesh_type="tet"),
                part_manager=_PartManager(parts),
                stl_loader=None,
            )
        tet_mesh = result[4]
        self.assertEqual(len(result[3]), 2)
        self.assertEqual(set(np.unique(tet_mesh.tet_part_types)), {PART_TYPE_CAST, PART_TYPE_CHILL})

    def test_voxel_path_is_unchanged(self) -> None:
        result = self._prepare(sp.PipelineInputs("WCB", 0.25, 0, mesh_type="voxel"), _settings(mesh_type="tet"))
        self.assertIsInstance(result[4], AdaptiveVoxelMesh)


def _hollow_box(outer, wall) -> trimesh.Trimesh:
    """Closed box with a closed inner cavity: a thin-walled part."""
    shell = _box(outer, (0.0, 0.0, 0.0))
    cavity = _box([e - 2.0 * wall for e in outer], (0.0, 0.0, 0.0))
    cavity.invert()
    return trimesh.Trimesh(
        np.vstack([shell.vertices, cavity.vertices]),
        np.vstack([shell.faces, cavity.faces + len(shell.vertices)]),
        process=False,
    )


class NodeCountGuardTests(unittest.TestCase):
    def test_estimate_has_a_volume_and_an_area_term(self) -> None:
        box = _box((2.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        # 1.55 * 2 / 0.2^3 + 3.3 * 10 / 0.2^2 = 387.5 + 825.
        self.assertEqual(sp.estimate_tet_node_count([(box, PART_TYPE_CAST)], 0.2), 1213)

    def test_open_surface_counts_its_convex_hull(self) -> None:
        box = _box((2.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        open_box = trimesh.Trimesh(box.vertices, box.faces[:-2], process=False)
        self.assertFalse(open_box.is_watertight)
        expected = int(np.ceil(1.55 * 2.0 / 0.2**3 + 3.3 * open_box.area / 0.2**2))
        self.assertEqual(sp.estimate_tet_node_count([(open_box, PART_TYPE_CAST)], 0.2), expected)

    def test_estimate_bounds_measured_thin_wall_and_block_meshes(self) -> None:
        """The old 2.0 * V / h^3 under-counted the thin-walled box (measured 2.19 x V / h^3)."""
        for label, part, h in (("thin-walled box", _hollow_box((4.0, 3.0, 2.0), 0.3), 0.1),
                               ("2x1x1 box", _box((2.0, 1.0, 1.0), (0.0, 0.0, 0.0)), 0.1)):
            with self.subTest(part=label):
                parts = [(part, PART_TYPE_CAST)]
                measured = build_tet_mesh(parts, h).get_voxel_count()
                estimate = sp.estimate_tet_node_count(parts, h)
                self.assertGreaterEqual(estimate, 1.05 * measured)
                self.assertLessEqual(estimate, 2.0 * measured)

    def test_edge_length_for_node_limit_lands_at_the_limit(self) -> None:
        parts = [(_hollow_box((4.0, 3.0, 2.0), 0.3), PART_TYPE_CAST)]
        for max_nodes in (1_000, 250_000, 3_000_000):
            with self.subTest(max_nodes=max_nodes):
                h = sp.tet_edge_length_for_node_limit(parts, max_nodes)
                self.assertLessEqual(sp.estimate_tet_node_count(parts, h), max_nodes)
                self.assertGreater(sp.estimate_tet_node_count(parts, 0.999 * h), max_nodes)
        with self.assertRaises(ValueError):
            sp.tet_edge_length_for_node_limit(parts, 0)

    def test_guard_refuses_before_meshing(self) -> None:
        box = _box((2.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        # Factor 1.0: the 0.2 in voxel size is the edge length the 1,213 estimate is for.
        settings = _settings(tet_max_nodes=100, tet_edge_length_factor=1.0)
        with mock.patch.object(sp, "get_settings", return_value=settings), mock.patch.object(
            sp, "build_tet_mesh"
        ) as built:
            with self.assertRaises(MeshGenerationError) as caught:
                sp.prepare_mesh_and_geometry(
                    sp.PipelineInputs("WCB", 0.2, 0, mesh_type="tet"),
                    part_manager=None,
                    stl_loader=_StlLoader(box),
                )
        built.assert_not_called()
        message = str(caught.exception)
        self.assertIn("tet_max_nodes", message)
        self.assertIn("1,213", message)
        self.assertIn("voxel size of at least", message)
        suggested = float(message.split("voxel size of at least ")[1].split(" in")[0])
        self.assertLessEqual(sp.estimate_tet_node_count([(box, PART_TYPE_CAST)], suggested), 100)


# ---------------------------------------------------------------------------
# Payload and result metadata
# ---------------------------------------------------------------------------


class _VoxelLike:
    def __init__(self, n: int) -> None:
        rng = np.random.default_rng(0)
        self.voxel_centers = rng.random((n, 3)).astype(np.float32)
        self.voxel_sizes = np.full(n, 0.1, dtype=np.float32)
        self.surface_voxels = np.zeros(n, dtype=bool)
        self.voxel_part_types = np.zeros(n, dtype=np.int8)
        self.voxel_moduli = np.zeros(n, dtype=np.float32)
        self.voxel_states = np.zeros(n, dtype=np.float32)

    def get_voxel_count(self) -> int:
        return len(self.voxel_centers)


def _two_tet_mesh() -> TetVolumeMesh:
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64)
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    vol = np.einsum("ij,ij->i", np.cross(nodes[tets[:, 1]] - nodes[tets[:, 0]], nodes[tets[:, 2]] - nodes[tets[:, 0]]), nodes[tets[:, 3]] - nodes[tets[:, 0]])
    tets[vol < 0, 2:4] = tets[vol < 0][:, [3, 2]]
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 4, 3], [2, 3, 4], [1, 2, 4]])
    return TetVolumeMesh(
        nodes, tets, np.zeros(2, dtype=np.int8), faces,
        np.zeros(len(faces), dtype=np.int8), np.full(len(faces), -1, dtype=np.int8),
        base_voxel_size=1.0,
    )


class PayloadMeshTypeTests(unittest.TestCase):
    def _payload(self, mesh):
        engine = _PayloadEngine(mesh.get_voxel_count())
        return sp.build_visualization_payload(
            engine, mesh, total_volume=1.0, bounds=np.array([[0, 1], [0, 1], [0, 1]]), max_voxels=1000
        )

    def test_tet_mesh_is_labelled(self) -> None:
        mesh = _two_tet_mesh()
        result_data, visualization = self._payload(mesh)
        self.assertEqual(visualization["mesh_type"], "tet")
        self.assertEqual(result_data["mesh_type"], "tet")
        self.assertEqual(result_data["voxel_count"], 5)
        self.assertEqual(result_data["tet_count"], 2)
        self.assertEqual(len(visualization["centers"]), 5)

    def test_voxel_mesh_is_labelled(self) -> None:
        result_data, visualization = self._payload(_VoxelLike(8))
        self.assertEqual(visualization["mesh_type"], "voxel")
        self.assertEqual(result_data["mesh_type"], "voxel")
        self.assertNotIn("tet_count", result_data)

    def test_pipeline_result_records_the_mesh_type(self) -> None:
        tet = sp.PipelineResult(None, _two_tet_mesh(), None, {}, {})
        voxel = sp.PipelineResult(None, _VoxelLike(3), None, {}, {})
        self.assertEqual((tet.mesh_type, voxel.mesh_type), ("tet", "voxel"))


# ---------------------------------------------------------------------------
# Filling seam
# ---------------------------------------------------------------------------


CAST_CHILL_PARTS = [
    (_box((2.0, 1.5, 1.0), (0.0, 0.0, 0.5)), PART_TYPE_CAST),
    (_box((1.0, 1.0, 0.5), (0.0, 0.0, -0.25)), PART_TYPE_CHILL),
]


def _synthetic_result(values: np.ndarray, frames: int = 2) -> FillingResult:
    n = len(values)
    return FillingResult(
        fill_time_s=values.copy(),
        temperature_c=1500.0 + values,
        inlet_voxel_count=1,
        filled_voxel_count=n,
        misrun_voxel_count=0,
        choke_area_cm2=0.0,
        filter_zones=[],
        steps=1,
        fill_seconds=1.0,
        coarsening=1,
        min_fill_temperature_c=1500.0,
        mean_fill_temperature_c=1500.0,
        entrapped_gas=values > np.median(values),
        frames=tuple(
            FillFrame(time_s=0.1 * k, fill_fraction=(values + k).astype(np.float32), speed_m_s=(2 * values).astype(np.float32))
            for k in range(frames)
        ),
    )


class NodeMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh = build_tet_mesh(CAST_CHILL_PARTS, 0.25)
        cls.lattice = build_filling_lattice(cls.mesh, 0.1)

    def test_every_node_maps_to_a_cell_of_its_class(self) -> None:
        cells = self.lattice.node_cells
        self.assertEqual(cells.shape, (self.mesh.get_voxel_count(),))
        node_metal = is_metal(self.mesh.voxel_part_types.astype(np.int32))
        cell_metal = is_metal(self.lattice.view.voxel_part_types.astype(np.int32))
        np.testing.assert_array_equal(cell_metal[cells], node_metal)

    def test_nodes_map_to_a_nearby_cell(self) -> None:
        centers = self.lattice.view.voxel_centers.astype(np.float64)[self.lattice.node_cells]
        distance = np.linalg.norm(centers - self.mesh.node_xyz, axis=1)
        self.assertLess(float(distance.max()), np.sqrt(3.0) * 0.1)

    def test_mapping_gathers_every_per_cell_field(self) -> None:
        view_cells = self.lattice.view.get_voxel_count()
        values = np.arange(view_cells, dtype=np.float64)
        mapped = map_filling_result_to_nodes(_synthetic_result(values), self.lattice.node_cells, view_cells)
        n = self.mesh.get_voxel_count()
        idx = self.lattice.node_cells
        np.testing.assert_array_equal(mapped.fill_time_s, values[idx])
        np.testing.assert_array_equal(mapped.temperature_c, 1500.0 + values[idx])
        np.testing.assert_array_equal(mapped.entrapped_gas, (values > np.median(values))[idx])
        self.assertEqual(mapped.cold_shut.shape, (0,))  # empty default left alone
        self.assertEqual(len(mapped.frames), 2)
        for k, frame in enumerate(mapped.frames):
            self.assertEqual(frame.fill_fraction.shape, (n,))
            self.assertEqual(frame.speed_m_s.shape, (n,))
            np.testing.assert_allclose(frame.fill_fraction, values[idx] + k)
            self.assertAlmostEqual(frame.time_s, 0.1 * k)
        self.assertEqual(mapped.filled_voxel_count, view_cells)

    def test_linear_cell_field_lands_within_a_cell_of_the_node_value(self) -> None:
        z = self.lattice.view.voxel_centers[:, 2].astype(np.float64)
        mapped = map_filling_result_to_nodes(
            _synthetic_result(100.0 * z), self.lattice.node_cells, len(z)
        )
        error = np.abs(mapped.fill_time_s - 100.0 * self.mesh.node_xyz[:, 2])
        self.assertLessEqual(float(error.max()), 100.0 * 0.1 + 1e-6)

    def test_no_node_is_far_from_its_cell(self) -> None:
        self.assertEqual(self.lattice.far_node_count, 0)

    def test_non_metal_view_falls_back_to_any_cell(self) -> None:
        view = self.lattice.view
        all_cast = mock.Mock(voxel_centers=view.voxel_centers, voxel_part_types=np.zeros(view.get_voxel_count(), dtype=np.int8))
        cells = node_lattice_cells(self.mesh.node_xyz, self.mesh.voxel_part_types, all_cast)
        self.assertEqual(cells.shape, (self.mesh.get_voxel_count(),))


class TetFillingGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh = build_tet_mesh(CAST_CHILL_PARTS[:1], 0.25)

    def test_disabled_filling_builds_no_lattice(self) -> None:
        with mock.patch("core.tet_filling.build_filling_lattice") as built:
            result = maybe_run_tet_filling(
                self.mesh, _Alloy(), _Options(filling=False), _Inputs(PourLocation(point_in=(0, 0, 1))), spacing_in=0.1
            )
        self.assertIsNone(result)
        built.assert_not_called()

    def test_missing_pour_location_is_reported(self) -> None:
        messages: list = []
        with mock.patch("core.tet_filling.build_filling_lattice") as built:
            result = maybe_run_tet_filling(
                self.mesh, _Alloy(), _Options(), _Inputs(None), spacing_in=0.1,
                progress_message_callback=messages.append,
            )
        self.assertIsNone(result)
        built.assert_not_called()
        self.assertTrue(any("pour location" in m.lower() for m in messages))

    def test_filling_does_not_build_lattice_topology(self) -> None:
        """The fill reads centres, sizes, volumes and codes only; neighbours and the EDT stay unbuilt."""
        import core.tet_filling as tet_filling

        built: list = []

        def capture(*args, **kwargs):
            built.append(real(*args, **kwargs))
            return built[-1]

        real = tet_filling.build_filling_lattice
        with mock.patch("core.tet_filling.build_filling_lattice", side_effect=capture):
            result = maybe_run_tet_filling(
                self.mesh, _Alloy(), _Options(), _Inputs(PourLocation(point_in=(0.0, 0.0, 0.95))),
                spacing_in=0.1,
            )
        self.assertIsNotNone(result)
        lazy = {"voxel_neighbors", "surface_voxels", "voxel_surface_areas", "voxel_moduli"}
        self.assertFalse(lazy & set(vars(built[0].view)))

    def test_too_large_lattice_is_skipped_and_reported(self) -> None:
        messages: list = []
        with mock.patch("core.tet_lattice.MAX_LATTICE_CELLS", 10):
            result = maybe_run_tet_filling(
                self.mesh, _Alloy(), _Options(), _Inputs(PourLocation(point_in=(0, 0, 1))), spacing_in=0.1,
                progress_message_callback=messages.append,
            )
        self.assertIsNone(result)
        self.assertTrue(any("skipped" in m for m in messages))


class FarNodeMappingTests(unittest.TestCase):
    """A metal part thinner than the pitch has no cell; its nodes are counted and reported."""

    @classmethod
    def setUpClass(cls) -> None:
        cast = _box((2.0, 2.0, 1.0), (0.0, 0.0, 0.5))
        # Separate 0.04 in fin, 1 in away, between the 0.2 in lattice planes z = 0.5 and 0.7.
        fin = _box((1.0, 1.0, 0.04), (3.0, 0.0, 0.6))
        cls.mesh = build_tet_mesh([(cast, PART_TYPE_CAST), (fin, PART_TYPE_SPRUE)], 0.25)
        cls.fin_nodes = int(np.sum(cls.mesh.node_xyz[:, 0] > 2.4))

    def test_far_nodes_are_counted(self) -> None:
        lattice = build_filling_lattice(self.mesh, 0.2)
        self.assertGreater(self.fin_nodes, 0)
        self.assertGreaterEqual(lattice.far_node_count, self.fin_nodes // 2)
        self.assertLessEqual(lattice.far_node_count, self.fin_nodes)

    def test_far_nodes_are_reported(self) -> None:
        messages: list = []
        with mock.patch("core.tet_filling.maybe_run_filling", return_value=None), self.assertLogs(
            "core.tet_filling", level="WARNING"
        ):
            maybe_run_tet_filling(
                self.mesh, _Alloy(), _Options(), _Inputs(PourLocation(point_in=(0.0, 0.0, 0.95))),
                spacing_in=0.2, progress_message_callback=messages.append,
            )
        self.assertTrue(any("tet nodes are more than" in m for m in messages))


# Casting at the bottom, a FILTER slab on top of it, a sprue above the filter.
FILTER_PARTS = [
    (_box((1.5, 1.5, 1.0), (0.0, 0.0, 0.5)), PART_TYPE_CAST),
    (_box((0.6, 0.6, 0.4), (0.0, 0.0, 1.2)), PART_TYPE_FILTER),
    (_box((0.6, 0.6, 0.8), (0.0, 0.0, 1.8)), PART_TYPE_SPRUE),
]
FILTER_POUR = PourLocation(point_in=(0.0, 0.0, 2.15))


class FilterAssemblyFillingTests(unittest.TestCase):
    """FILTER tets are removed from the tet mesh; the fill must still flow through the filter."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh = build_tet_mesh(FILTER_PARTS, 0.2)
        cls.lattice = build_filling_lattice(cls.mesh, 0.1, filter_meshes=[FILTER_PARTS[1][0]])
        # Casting nodes below the filter, clear of the filter face.
        cls.below = (cls.mesh.voxel_part_types == PART_TYPE_CAST) & (cls.mesh.node_xyz[:, 2] < 0.9)

    def test_tet_mesh_has_no_filter(self) -> None:
        self.assertNotIn(PART_TYPE_FILTER, set(np.unique(self.mesh.tet_part_types)))

    def test_filling_view_adds_filter_cells_on_the_sampler_lattice(self) -> None:
        filling_mesh = self.lattice.filling_mesh
        self.assertIsInstance(filling_mesh, FillingLatticeMesh)
        self.assertGreater(self.lattice.filter_cell_count, 0)
        codes = filling_mesh.voxel_part_types[filling_mesh.view_cell_count:]
        self.assertTrue(np.all(codes == PART_TYPE_FILTER))
        # 0.6 x 0.6 x 0.4 in slab at 0.1 in pitch: about 6 x 6 x 4 cells.
        self.assertGreaterEqual(self.lattice.filter_cell_count, 100)
        self.assertLessEqual(self.lattice.filter_cell_count, 7 * 7 * 6)
        mapping = build_grid_mapping(filling_mesh.voxel_centers, 0.1)
        self.assertEqual(mapping.n_voxels, filling_mesh.get_voxel_count())

    def test_nodes_never_map_to_filter_cells(self) -> None:
        self.assertLess(int(self.lattice.node_cells.max()), self.lattice.filling_mesh.view_cell_count)

    def _fill(self, parts):
        messages: list = []
        result = maybe_run_tet_filling(
            self.mesh, _Alloy(), _Options(), _Inputs(FILTER_POUR),
            spacing_in=0.1, parts_with_types=parts, progress_message_callback=messages.append,
        )
        return result, messages

    def test_metal_reaches_the_casting_through_the_filter(self) -> None:
        started = time.perf_counter()
        result, messages = self._fill(FILTER_PARTS)
        elapsed = time.perf_counter() - started
        self.assertIsNotNone(result)
        n = self.mesh.get_voxel_count()
        self.assertEqual(result.fill_time_s.shape, (n,))
        self.assertEqual(result.temperature_c.shape, (n,))
        self.assertTrue(np.all(np.isfinite(result.temperature_c)))
        self.assertTrue(result.frames)
        self.assertTrue(all(f.fill_fraction.shape == (n,) for f in result.frames))
        reached = int(np.sum(result.fill_time_s[self.below] >= 0.0))
        self.assertGreater(reached, 0, "no casting node below the filter was filled")
        self.assertTrue(any("FILTER cells" in m for m in messages))
        print(f"\n  filter fill: {reached}/{int(self.below.sum())} casting nodes reached in {elapsed:.1f} s", file=sys.stderr)

    def test_without_filter_cells_the_fill_stops_at_the_filter(self) -> None:
        """Contrast case: the gap where the filter tets were acts as a wall."""
        no_filter = [p for p in FILTER_PARTS if p[1] != PART_TYPE_FILTER]
        result, _ = self._fill(no_filter)
        if result is None:
            return  # the stage refused a severed path: also no metal below the filter
        self.assertEqual(int(np.sum(result.fill_time_s[self.below] >= 0.0)), 0)


# ---------------------------------------------------------------------------
# run_pipeline and run_surrogate_pipeline on a tet mesh
# ---------------------------------------------------------------------------


SMOKE_BOX = _box((2.0, 1.0, 1.0), (0.0, 0.0, 0.5))
SMOKE_FILL_PARAMS = {"filling": {"max_fill_time_s": 0.05, "fill_grid_coarsening": 1}}


class RunPipelineTetTests(unittest.TestCase):
    """run_pipeline end to end on a tet mesh with the real engine."""

    def _run(self, **input_kwargs):
        inputs = sp.PipelineInputs("WCB", 0.25, 0, mesh_type="tet", **input_kwargs)
        with mock.patch.object(sp, "get_settings", return_value=_settings()):
            return sp.run_pipeline(inputs, stl_loader=_StlLoader(SMOKE_BOX))

    def test_run_pipeline_without_filling(self) -> None:
        result = self._run(analysis_options={"shrinkage": True, "hotspot": True})
        n = result.voxel_mesh.get_voxel_count()
        self.assertEqual(result.mesh_type, "tet")
        self.assertIsNone(getattr(result.engine, "_initial_temperature_field", None))
        self.assertEqual(result.engine.solidification_times.shape, (n,))
        self.assertEqual(result.visualization_data["mesh_type"], "tet")
        self.assertEqual(len(result.visualization_data["centers"]), n)
        self.assertEqual(len(result.visualization_data["solidification_times"]), n)
        self.assertEqual(result.result_data["voxel_count"], n)

    def test_non_touching_two_part_assembly_runs(self) -> None:
        """Separate components: every face is in a boundary group and every metal node solidifies."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        from core.part_manager import PART_TYPE_RISER
        from core.tet_thermal import tet_boundary_groups

        parts = [
            (_box((2.0, 1.0, 1.0), (0.0, 0.0, 0.5)), PART_TYPE_CAST),
            (_box((1.0, 1.0, 1.0), (2.5, 0.0, 0.5)), PART_TYPE_RISER),  # 1 in gap
        ]
        inputs = sp.PipelineInputs("WCB", 0.25, 0, mesh_type="tet")
        with mock.patch.object(sp, "get_settings", return_value=_settings()):
            result = sp.run_pipeline(inputs, part_manager=_PartManager(parts))
        mesh = result.voxel_mesh
        n = mesh.get_voxel_count()
        adjacency = csr_matrix((np.ones(len(mesh.adj_indices)), mesh.adj_indices, mesh.adj_indptr), shape=(n, n))
        self.assertEqual(connected_components(adjacency, directed=False)[0], 2)
        grouped = np.concatenate([g.face_indices for g in tet_boundary_groups(result.engine)])
        np.testing.assert_array_equal(np.sort(grouped), np.arange(len(mesh.boundary_faces)))
        metal = is_metal(mesh.voxel_part_types.astype(np.int32))
        self.assertTrue(np.all(np.isfinite(result.engine.solidification_times[metal])))

    def test_run_pipeline_hands_node_fields_from_the_fill(self) -> None:
        result = self._run(
            analysis_options={"filling": True},
            pour_location=PourLocation(point_in=(0.0, 0.0, 0.95)).to_dict(),
            solidification_params=SMOKE_FILL_PARAMS,
        )
        engine = result.engine
        n = result.voxel_mesh.get_voxel_count()
        self.assertEqual(engine._initial_temperature_field.shape, (n,))
        self.assertEqual(engine._fill_arrival_times.shape, (n,))
        self.assertTrue(engine._fill_frames)
        self.assertTrue(all(len(f.fill_fraction) == n for f in engine._fill_frames))
        self.assertEqual(len(result.visualization_data["fill_time"]), n)
        self.assertEqual(len(result.visualization_data["fill_animation_history"][0]), n)


class _StubSurrogateModel:
    """Predicts start = 10 * x cell index, end = 0, local = 5 everywhere on its grid."""

    def __init__(self, resolution: int = 16) -> None:
        self.manifest = mock.Mock(grid_resolution=resolution, accuracy={}, backend="numpy")
        self.manifest.name = "stub"
        self.resolution = resolution

    def predict(self, channels, conditioning) -> np.ndarray:
        r = self.resolution
        out = np.zeros((3, r, r, r), dtype=np.float32)
        # Grid flat index is x-fastest (core.surrogate.fields), so x is the last axis.
        out[0] = 10.0 * np.arange(r, dtype=np.float32)[None, None, :]
        out[2] = 5.0
        return out


class SurrogateOnTetTests(unittest.TestCase):
    def test_prediction_unprojects_onto_nodes(self) -> None:
        from core.surrogate import pipeline as surrogate_pipeline
        from core.surrogate.fields import unproject_grid

        inputs = sp.PipelineInputs("WCB", 0.25, 0, mesh_type="tet")
        with mock.patch.object(sp, "get_settings", return_value=_settings()), mock.patch.object(
            surrogate_pipeline, "unproject_grid", wraps=unproject_grid
        ) as unproject:
            result = surrogate_pipeline.run_surrogate_pipeline(
                inputs, model=_StubSurrogateModel(), stl_loader=_StlLoader(SMOKE_BOX)
            )
        mesh = result.voxel_mesh
        self.assertIsInstance(mesh, TetVolumeMesh)
        self.assertEqual(result.mesh_type, "tet")
        np.testing.assert_array_equal(unproject.call_args_list[0].args[1], mesh.node_xyz.astype(np.float32))
        engine = result.engine
        n = mesh.get_voxel_count()
        self.assertEqual(engine.solidification_start_times.shape, (n,))
        x = mesh.node_xyz[:, 0]
        # start grows with x across the casting.
        self.assertGreater(float(engine.solidification_start_times[np.argmax(x)]), float(engine.solidification_start_times[np.argmin(x)]))
        np.testing.assert_allclose(engine.local_solidification_time, 5.0)
        self.assertEqual(result.visualization_data["mesh_type"], "tet")
        self.assertEqual(len(result.visualization_data["solidification_times"]), n)


if __name__ == "__main__":
    unittest.main()
