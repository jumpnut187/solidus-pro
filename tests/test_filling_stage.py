"""Tests for the mesh-to-lattice seam.

The mapping tests are the load-bearing ones. Everything downstream -- the
solver, the filter drag, the handover back to solidification -- assumes the
voxel list and the dense lattice are in exact correspondence, and a mapping
that quietly loses voxels would show up only as a fill that ends in the wrong
place.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.filling_stage import (
    COARSENING_PRIORITY,
    EMPTY_CELL,
    _expand_inlet_to_mouth,
    build_filter_zone_specs,
    build_grid_mapping,
    build_porous_zone,
    cells_along_flow,
    coarsen_part_types,
    dense_part_types,
    filter_geometry_from_mesh,
    gather_to_voxels,
    measure_choke_area_cm2,
    measure_choke_thickness_by_part,
    scatter_to_grid,
    thickness_correction,
)
from core.filling_stage import (
    MIN_CHOKE_THICKNESS_CELLS,
    choose_fill_coarsening,
)
from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_FILTER,
    PART_TYPE_GATE,
    PART_TYPE_RISER,
    PART_TYPE_RUNNER,
    PART_TYPE_SPRUE,
    PART_TYPE_TO_CODE,
)

DX_IN = 0.25


def lattice_centers(shape, dx=DX_IN, origin=(1.0, 2.0, 3.0)):
    """Voxel centres for a fully occupied box, in the mesh's own convention."""
    nx, ny, nz = shape
    idx = np.stack(np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    ), axis=-1).reshape(-1, 3)
    return np.asarray(origin, dtype=np.float64) + idx * dx


class GridMappingTests(unittest.TestCase):
    def test_shape_comes_from_the_centres(self) -> None:
        centers = lattice_centers((5, 4, 3))
        mapping = build_grid_mapping(centers, DX_IN)
        self.assertEqual(mapping.shape, (5, 4, 3))

    def test_mapping_is_bijective(self) -> None:
        centers = lattice_centers((6, 5, 4))
        mapping = build_grid_mapping(centers, DX_IN)
        self.assertEqual(mapping.n_voxels, centers.shape[0])
        self.assertEqual(
            int(np.count_nonzero(mapping.grid_to_voxel >= 0)), centers.shape[0]
        )

    def test_sparse_geometry_leaves_holes_marked(self) -> None:
        centers = lattice_centers((4, 4, 4))
        keep = np.ones(centers.shape[0], dtype=bool)
        keep[::3] = False
        mapping = build_grid_mapping(centers[keep], DX_IN)
        self.assertEqual(mapping.n_voxels, int(keep.sum()))
        self.assertEqual(
            int(np.count_nonzero(mapping.grid_to_voxel < 0)),
            int(np.prod(mapping.shape)) - int(keep.sum()),
        )

    def test_full_box_has_unit_occupancy(self) -> None:
        mapping = build_grid_mapping(lattice_centers((4, 4, 4)), DX_IN)
        self.assertAlmostEqual(mapping.occupancy, 1.0, places=12)

    def test_hollow_geometry_reports_partial_occupancy(self) -> None:
        """
        Real gating assemblies are mostly empty bounding box -- the
        Filling_Test rigging measures 0.147 -- so the dense lattice is several
        times the occupied voxel count and the memory budget must be per dense
        cell, not per voxel.
        """
        centers = lattice_centers((5, 5, 5))
        # Hollow out the interior, keeping the shell so the bounding box (and
        # therefore the derived dense shape) is unchanged.
        idx = np.round((centers - centers.min(axis=0)) / DX_IN).astype(int)
        shell = (idx.min(axis=1) == 0) | (idx.max(axis=1) == 4)
        mapping = build_grid_mapping(centers[shell], DX_IN)
        self.assertEqual(mapping.shape, (5, 5, 5))
        self.assertAlmostEqual(
            mapping.occupancy, int(shell.sum()) / 125.0, places=12
        )
        self.assertLess(mapping.occupancy, 1.0)

    def test_dx_converts_to_metres(self) -> None:
        mapping = build_grid_mapping(lattice_centers((2, 2, 2), dx=1.0), 1.0)
        self.assertAlmostEqual(mapping.dx_m, 0.0254, places=9)

    def test_origin_offset_does_not_matter(self) -> None:
        a = build_grid_mapping(lattice_centers((4, 3, 2), origin=(0, 0, 0)), DX_IN)
        b = build_grid_mapping(lattice_centers((4, 3, 2), origin=(-9, 17, 4.5)), DX_IN)
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(a.voxel_to_grid, b.voxel_to_grid)

    def test_non_uniform_centres_are_rejected(self) -> None:
        """
        The dense representation is only valid on a uniform lattice, so a
        non-uniform mesh must fail loudly rather than silently drop voxels.
        """
        centers = lattice_centers((3, 3, 3))
        centers[5] += 0.11  # nudge one voxel well off the lattice
        with self.assertRaises(ValueError):
            build_grid_mapping(centers, DX_IN)

    def test_small_float_noise_is_tolerated(self) -> None:
        """Real meshes carry float32 rounding; 3.8e-6 cells was measured."""
        centers = lattice_centers((3, 3, 3))
        centers += 1e-6
        mapping = build_grid_mapping(centers, DX_IN)
        self.assertEqual(mapping.shape, (3, 3, 3))

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            build_grid_mapping(np.zeros((0, 3)), DX_IN)
        with self.assertRaises(ValueError):
            build_grid_mapping(np.zeros((4, 2)), DX_IN)
        with self.assertRaises(ValueError):
            build_grid_mapping(lattice_centers((2, 2, 2)), 0.0)


class ScatterGatherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.centers = lattice_centers((4, 3, 2))
        self.mapping = build_grid_mapping(self.centers, DX_IN)

    def test_scatter_gather_round_trips(self) -> None:
        values = np.arange(self.mapping.n_voxels, dtype=np.float64)
        dense = scatter_to_grid(self.mapping, values)
        np.testing.assert_array_equal(gather_to_voxels(self.mapping, dense), values)

    def test_gathered_field_has_voxel_length(self) -> None:
        """
        set_initial_temperature_field wants (n_voxels,), i.e. the occupied
        count -- not the dense cell count. Conflating the two is the shape
        mismatch the handover would fail on.
        """
        dense = np.zeros(self.mapping.shape)
        self.assertEqual(
            gather_to_voxels(self.mapping, dense).shape, (self.mapping.n_voxels,)
        )

    def test_empty_cells_are_not_confused_with_cast(self) -> None:
        centers = lattice_centers((3, 3, 3))
        keep = np.ones(centers.shape[0], dtype=bool)
        keep[0] = False
        mapping = build_grid_mapping(centers[keep], DX_IN)
        dense = dense_part_types(
            mapping, np.full(mapping.n_voxels, PART_TYPE_CAST, dtype=np.int8)
        )
        self.assertEqual(int(np.count_nonzero(dense == EMPTY_CELL)), 1)
        self.assertNotEqual(EMPTY_CELL, PART_TYPE_CAST)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            scatter_to_grid(self.mapping, np.zeros(3))
        with self.assertRaises(ValueError):
            dense_part_types(self.mapping, np.zeros(3, dtype=np.int8))
        with self.assertRaises(ValueError):
            gather_to_voxels(self.mapping, np.zeros((2, 2, 2)))


class CoarseningTests(unittest.TestCase):
    def test_factor_one_is_identity(self) -> None:
        dense = np.full((4, 4, 4), PART_TYPE_CAST, dtype=np.int8)
        np.testing.assert_array_equal(coarsen_part_types(dense, 1), dense)

    def test_shape_halves(self) -> None:
        dense = np.full((8, 6, 4), PART_TYPE_CAST, dtype=np.int8)
        self.assertEqual(coarsen_part_types(dense, 2).shape, (4, 3, 2))

    def test_odd_dimensions_round_up(self) -> None:
        dense = np.full((5, 5, 5), PART_TYPE_CAST, dtype=np.int8)
        self.assertEqual(coarsen_part_types(dense, 2).shape, (3, 3, 3))

    def test_a_one_cell_filter_survives_coarsening(self) -> None:
        """
        The reason coarsening resolves by priority and not majority. A filter
        is one or two cells thick; a majority vote erases it, and the run then
        silently omits the restriction the user placed.
        """
        dense = np.full((4, 4, 4), PART_TYPE_CAST, dtype=np.int8)
        dense[1, 1, 1] = PART_TYPE_FILTER
        coarse = coarsen_part_types(dense, 2)
        self.assertEqual(int(np.count_nonzero(coarse == PART_TYPE_FILTER)), 1)

    def test_filter_outranks_everything(self) -> None:
        for competitor in (PART_TYPE_CAST, PART_TYPE_CHILL, PART_TYPE_GATE):
            with self.subTest(competitor=competitor):
                dense = np.full((2, 2, 2), competitor, dtype=np.int8)
                dense[0, 0, 0] = PART_TYPE_FILTER
                self.assertEqual(int(coarsen_part_types(dense, 2)[0, 0, 0]), PART_TYPE_FILTER)

    def test_empty_blocks_stay_empty(self) -> None:
        dense = np.full((4, 4, 4), EMPTY_CELL, dtype=np.int8)
        dense[0, 0, 0] = PART_TYPE_CAST
        coarse = coarsen_part_types(dense, 2)
        self.assertEqual(int(coarse[0, 0, 0]), PART_TYPE_CAST)
        self.assertEqual(int(coarse[1, 1, 1]), EMPTY_CELL)

    def test_priority_covers_every_part_type(self) -> None:
        """A code missing from the table would be dropped by coarsening."""
        self.assertEqual(set(COARSENING_PRIORITY), set(PART_TYPE_TO_CODE.values()))

    def test_invalid_factor_raises(self) -> None:
        dense = np.zeros((2, 2, 2), dtype=np.int8)
        for bad in (0, -1, 1.5):
            with self.subTest(factor=bad):
                with self.assertRaises(ValueError):
                    coarsen_part_types(dense, bad)


class FilterGeometryTests(unittest.TestCase):
    def test_square_filter_from_a_mesh(self) -> None:
        import trimesh

        # 3 x 3 x 1 inch slab.
        body = trimesh.creation.box(extents=(3.0, 3.0, 1.0))
        area_m2, thickness_m = filter_geometry_from_mesh(body)
        self.assertAlmostEqual(thickness_m, 0.0254, places=6)
        self.assertAlmostEqual(area_m2, 9.0 * 0.0254 ** 2, places=9)

    def test_round_filter_area_is_not_the_bbox_face(self) -> None:
        """
        volume / thickness is exact for a prism of any cross-section. The
        product of the two larger bbox extents over-estimates a cylinder by
        4/pi = 27 %, and seven of the thirteen catalog sizes are round.
        """
        import trimesh

        body = trimesh.creation.cylinder(radius=1.5, height=1.0, sections=128)
        area_m2, thickness_m = filter_geometry_from_mesh(body)
        expected = np.pi * 1.5 ** 2 * 0.0254 ** 2
        self.assertAlmostEqual(area_m2 / expected, 1.0, delta=0.02)
        bbox_face = 3.0 * 3.0 * 0.0254 ** 2
        self.assertAlmostEqual(bbox_face / expected, 4.0 / np.pi, delta=0.02)

    def test_degenerate_body_raises(self) -> None:
        import trimesh

        flat = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]]
        )
        with self.assertRaises(ValueError):
            filter_geometry_from_mesh(flat)


class ThicknessCorrectionTests(unittest.TestCase):
    def test_exactly_resolved_thickness_needs_no_correction(self) -> None:
        self.assertAlmostEqual(thickness_correction(0.025, 1.0, 0.025), 1.0, places=9)

    def test_under_resolved_thickness_scales_up(self) -> None:
        # A 25 mm filter resolved by one 12.5 mm cell is half as thick as it
        # should be, so the correction doubles the drop.
        self.assertAlmostEqual(thickness_correction(0.025, 1.0, 0.0125), 2.0, places=9)

    def test_correction_is_clamped(self) -> None:
        self.assertEqual(thickness_correction(1.0, 1.0, 1e-6), 4.0)
        self.assertEqual(thickness_correction(1e-9, 1.0, 1.0), 0.25)

    def test_cells_along_flow_from_volume(self) -> None:
        # 4 cells over a 2-cell face, dx = 1 -> 2 cells deep.
        dx = 0.01
        area = 2 * dx ** 2
        self.assertAlmostEqual(cells_along_flow(4, area, dx), 2.0, places=9)

    def test_cells_along_flow_rejects_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            cells_along_flow(4, 0.0, 0.01)


class FilterZoneTests(unittest.TestCase):
    @staticmethod
    def _grid_with_filters(n_filters: int = 1) -> np.ndarray:
        dense = np.full((10, 6, 6), PART_TYPE_CAST, dtype=np.int8)
        dense[2:4, 1:5, 1:5] = PART_TYPE_FILTER
        if n_filters > 1:
            dense[7:9, 1:5, 1:5] = PART_TYPE_FILTER
        return dense

    def test_no_filter_yields_no_zones(self) -> None:
        dense = np.full((4, 4, 4), PART_TYPE_CAST, dtype=np.int8)
        self.assertEqual(build_filter_zone_specs(dense, dx_m=0.00635), [])

    def test_one_filter_yields_one_zone(self) -> None:
        specs = build_filter_zone_specs(self._grid_with_filters(), dx_m=0.00635)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].cell_count, 2 * 4 * 4)

    def test_two_separate_filters_are_distinct_zones(self) -> None:
        specs = build_filter_zone_specs(self._grid_with_filters(2), dx_m=0.00635)
        self.assertEqual(len(specs), 2)
        self.assertEqual({s.zone_id for s in specs}, {1, 2})

    def test_zone_carries_a_usable_closure(self) -> None:
        spec = build_filter_zone_specs(self._grid_with_filters(), dx_m=0.00635)[0]
        self.assertGreater(spec.k1_m2, 0.0)
        self.assertGreater(spec.inv_k2_per_m, 0.0)
        self.assertGreater(spec.entry_pressure_pa, 0.0)
        # 10 ppi steel primes on 40-80 mm of head.
        self.assertGreater(spec.priming_head_m, 0.040)
        self.assertLess(spec.priming_head_m, 0.080)

    def test_thickness_correction_is_applied_to_both_coefficients(self) -> None:
        """
        k1 divides and 1/k2 multiplies, so the integrated drop is invariant.
        A correction applied in one direction only, or to one coefficient only,
        is otherwise invisible.
        """
        spec = build_filter_zone_specs(self._grid_with_filters(), dx_m=0.00635)[0]
        from physics.fluid_flow import (
            ceramic_foam_permeability,
            forchheimer_inverse_length,
        )

        base_k1 = ceramic_foam_permeability(0.85, 2.54e-3)
        base_inv_k2 = forchheimer_inverse_length(base_k1)
        self.assertAlmostEqual(
            spec.k1_m2, base_k1 / spec.thickness_correction, places=12
        )
        self.assertAlmostEqual(
            spec.inv_k2_per_m, base_inv_k2 * spec.thickness_correction, places=6
        )


class PorousZoneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dense = np.full((10, 6, 6), PART_TYPE_CAST, dtype=np.int8)
        self.dense[2:4, 1:5, 1:5] = PART_TYPE_FILTER
        self.specs = build_filter_zone_specs(self.dense, dx_m=0.00635)

    def test_no_filter_gives_an_inert_zone(self) -> None:
        dense = np.full((4, 4, 4), PART_TYPE_CAST, dtype=np.int8)
        zone = build_porous_zone(dense, [], dx_m=0.00635, dt_s=1e-4, nu_lattice=1e-5)
        self.assertTrue(zone.is_inert)

    def test_filter_cells_get_drag_and_others_do_not(self) -> None:
        zone = build_porous_zone(
            self.dense, self.specs, dx_m=0.00635, dt_s=1e-4, nu_lattice=1e-5
        )
        cells = self.dense == PART_TYPE_FILTER
        self.assertTrue(np.all(zone.drag_c1[cells] > 0.0))
        self.assertTrue(np.all(zone.drag_c1[~cells] == 0.0))
        self.assertTrue(np.all(zone.drag_c0[~cells] == 0.5))
        self.assertTrue(np.all(zone.porosity[~cells] == 1.0))

    def test_filter_cells_carry_a_capillary_offset(self) -> None:
        zone = build_porous_zone(
            self.dense, self.specs, dx_m=0.00635, dt_s=1e-4, nu_lattice=1e-5
        )
        cells = self.dense == PART_TYPE_FILTER
        self.assertTrue(np.all(zone.entry_offset[cells] > 0.0))
        self.assertTrue(np.all(zone.entry_offset[~cells] == 0.0))

    def test_nothing_starts_primed(self) -> None:
        zone = build_porous_zone(
            self.dense, self.specs, dx_m=0.00635, dt_s=1e-4, nu_lattice=1e-5
        )
        self.assertFalse(bool(zone.primed.any()))

    def test_zone_shape_matches_the_grid(self) -> None:
        zone = build_porous_zone(
            self.dense, self.specs, dx_m=0.00635, dt_s=1e-4, nu_lattice=1e-5
        )
        self.assertEqual(zone.drag_c0.shape, self.dense.shape)


class ChokeAreaTests(unittest.TestCase):
    def test_no_gating_metal_gives_zero(self) -> None:
        dense = np.full((4, 4, 4), PART_TYPE_CAST, dtype=np.int8)
        self.assertEqual(measure_choke_area_cm2(dense, 0.01), 0.0)

    def test_choke_is_the_narrowest_section(self) -> None:
        dense = np.full((10, 10, 10), EMPTY_CELL, dtype=np.int8)
        # A runner 3x3 in cross-section, necking to a 1x1 gate.
        dense[0:6, 4:7, 4:7] = PART_TYPE_RUNNER
        dense[6:10, 5:6, 5:6] = PART_TYPE_GATE
        dx = 0.01  # 1 cm cells -> 1 cm^2 each
        self.assertAlmostEqual(measure_choke_area_cm2(dense, dx), 1.0, places=6)

    def test_pour_cup_is_excluded(self) -> None:
        """
        The cup is a reservoir, not a metering section. Including it would let
        a wide basin or a narrow one change the reported choke.
        """
        from core.part_manager import PART_TYPE_POUR_CUP

        dense = np.full((10, 10, 10), EMPTY_CELL, dtype=np.int8)
        dense[0:5, 4:7, 4:7] = PART_TYPE_SPRUE
        with_cup = dense.copy()
        with_cup[5:10, 0:10, 0:10] = PART_TYPE_POUR_CUP
        self.assertEqual(
            measure_choke_area_cm2(dense, 0.01),
            measure_choke_area_cm2(with_cup, 0.01),
        )

    def test_area_scales_with_cell_size(self) -> None:
        dense = np.full((6, 6, 6), EMPTY_CELL, dtype=np.int8)
        dense[0:6, 2:4, 2:4] = PART_TYPE_SPRUE
        coarse = measure_choke_area_cm2(dense, 0.02)
        fine = measure_choke_area_cm2(dense, 0.01)
        self.assertAlmostEqual(coarse / fine, 4.0, places=6)


if __name__ == "__main__":
    unittest.main()


class _FakeMesh:
    """Minimal stand-in for AdaptiveVoxelMesh: the stage only reads three fields."""

    def __init__(self, centers, part_types, dx):
        self.voxel_centers = centers
        self.voxel_part_types = np.asarray(part_types, dtype=np.int8)
        self.voxel_sizes = np.full(len(centers), dx, dtype=np.float64)

    def get_voxel_count(self):
        return len(self.voxel_centers)


class _Filter:
    porosity = 0.85
    pore_diameter_mm = 2.54
    ergun_prefactor = 2.2
    forchheimer_c_f = 0.55
    contact_angle_deg = 120.0
    window_fraction = 0.45


class _Filling:
    fill_grid_coarsening = 1
    lattice_velocity_max = 0.05
    max_fill_time_s = 0.02
    pour_temperature_c = 0.0
    initial_head_in = 4.0
    filter = _Filter()


class _ThermalBoundary:
    mold_temperature = 21.0


class _Params:
    filling = _Filling()
    thermal_boundary = _ThermalBoundary()


class _Options:
    def __init__(self, filling=True, solver_mode="thermal_coupled"):
        self.filling = filling
        self.solver_mode = solver_mode
        self.solidification_params = _Params()


class _Inputs:
    def __init__(self, pour_location=None):
        self.pour_location = pour_location


class _Alloy:
    dynamic_viscosity = 0.006
    density_liquid = 7200.0
    pouring_temp = 1600.0
    surface_tension = 1.85
    # F2 thermal coupling: physics.filling advects and cools the fill, which
    # needs the same material properties the solidification solve already
    # uses (see AlloyProperties in core/materials.py).
    thermal_conductivity = 30.0
    specific_heat = 600.0
    liquidus_temp = 1450.0


def _column_mesh(nz=6, dx=0.25):
    """A 2x2xNz metal column; the pour goes in at the top."""
    centers = lattice_centers((2, 2, nz), dx=dx, origin=(0.0, 0.0, 0.0))
    return _FakeMesh(centers, np.full(len(centers), PART_TYPE_CAST), dx)


def _top_pour(mesh, dx=0.25, nz=6):
    from core.pour_location import PourLocation

    return PourLocation(point_in=(0.0, 0.0, (nz - 1) * dx))


class FillingGateTests(unittest.TestCase):
    """
    Requirement: the fill runs only when a pour location has been picked.

    A skip must be *reported*, not silent -- an unnoticed skip is
    indistinguishable from a fill that ran and found nothing wrong.
    """

    def setUp(self) -> None:
        from core.filling_stage import maybe_run_filling

        self.run = maybe_run_filling
        self.mesh = _column_mesh()
        self.alloy = _Alloy()
        self.messages: list = []

    def _call(self, options, inputs):
        return self.run(
            self.mesh, self.alloy, options, inputs,
            progress_message_callback=self.messages.append,
        )

    def test_disabled_analysis_does_not_run(self) -> None:
        result = self._call(_Options(filling=False), _Inputs(_top_pour(self.mesh)))
        self.assertIsNone(result)

    def test_disabled_analysis_is_silent(self) -> None:
        """Not selecting an analysis is not a condition worth reporting."""
        self._call(_Options(filling=False), _Inputs(_top_pour(self.mesh)))
        self.assertEqual(self.messages, [])

    def test_enabled_without_a_pour_location_is_skipped_and_reported(self) -> None:
        result = self._call(_Options(filling=True), _Inputs(None))
        self.assertIsNone(result)
        self.assertTrue(
            any("pour location" in m.lower() for m in self.messages),
            f"skip was not explained: {self.messages}",
        )

    def test_invalid_pour_location_is_skipped_and_reported(self) -> None:
        result = self._call(_Options(filling=True), _Inputs({"point_in": "nonsense"}))
        self.assertIsNone(result)
        self.assertTrue(any("invalid" in m.lower() for m in self.messages))

    def test_serialized_pour_location_is_accepted(self) -> None:
        payload = _top_pour(self.mesh).to_dict()
        result = self._call(_Options(filling=True), _Inputs(payload))
        self.assertIsNotNone(result)

    def test_pour_location_off_the_metal_is_skipped(self) -> None:
        from core.pour_location import PourLocation

        far = PourLocation(point_in=(500.0, 500.0, 500.0))
        result = self._call(_Options(filling=True), _Inputs(far))
        self.assertIsNone(result)
        self.assertTrue(any("metal cell" in m for m in self.messages))


class FillingResultTests(unittest.TestCase):
    def setUp(self) -> None:
        from core.filling_stage import maybe_run_filling

        self.mesh = _column_mesh()
        self.messages: list = []
        self.result = maybe_run_filling(
            self.mesh, _Alloy(), _Options(), _Inputs(_top_pour(self.mesh)),
            progress_message_callback=self.messages.append,
        )
        self.assertIsNotNone(self.result)

    def test_fields_are_voxel_shaped(self) -> None:
        """
        set_initial_temperature_field validates shape == (n_voxels,). Returning
        a dense-grid-shaped array is the mismatch that would fail the handover.
        """
        n = self.mesh.get_voxel_count()
        self.assertEqual(self.result.temperature_c.shape, (n,))
        self.assertEqual(self.result.fill_time_s.shape, (n,))

    def test_temperature_field_is_finite(self) -> None:
        self.assertTrue(bool(np.all(np.isfinite(self.result.temperature_c))))

    def test_temperatures_are_bounded_by_pour_and_mold(self) -> None:
        """
        F2 advects and cools the fill, so filled voxels are no longer pinned
        to the pour temperature exactly -- but the scheme must not overshoot
        either bound (see tests/test_filling_thermal.py for the general case).
        """
        self.assertTrue(bool(np.all(self.result.temperature_c <= 1600.0 + 1e-6)))
        self.assertTrue(bool(np.all(self.result.temperature_c >= 21.0 - 1e-6)))

    def test_the_inlet_filled(self) -> None:
        self.assertGreater(self.result.inlet_voxel_count, 0)
        self.assertGreater(self.result.filled_voxel_count, 0)

    def test_counts_partition_the_metal(self) -> None:
        n = self.mesh.get_voxel_count()
        self.assertEqual(
            self.result.filled_voxel_count + self.result.misrun_voxel_count, n
        )

    def test_unfilled_metal_is_seeded_at_mold_temperature(self) -> None:
        """
        A misrun must not be hidden behind a plausible pouring temperature.
        The seam requires every voxel finite, so blank is not an option --
        mold temperature is the honest encoding of "no metal arrived".
        """
        unfilled = self.result.fill_time_s < 0
        if unfilled.any():
            np.testing.assert_allclose(self.result.temperature_c[unfilled], 21.0)

    def test_filled_voxels_have_a_nonnegative_arrival_time(self) -> None:
        """Anything warmer than mold temperature must have actually arrived."""
        warmed = self.result.temperature_c > 21.0 + 1e-6
        self.assertTrue(bool(np.all(self.result.fill_time_s[warmed] >= 0.0)))

    def test_progress_was_reported(self) -> None:
        self.assertTrue(any("Filling:" in m for m in self.messages))


class SolverModeWarningTests(unittest.TestCase):
    def test_legacy_solver_mode_warns_that_the_seed_is_ignored(self) -> None:
        """
        core/surrogate/pipeline.py pins schedule_legacy, and that mode ignores
        the seeded field. Running a fill whose thermal result is discarded is
        worth saying out loud rather than leaving as a silent no-op.
        """
        from core.filling_stage import maybe_run_filling

        mesh = _column_mesh()
        messages: list = []
        result = maybe_run_filling(
            mesh, _Alloy(), _Options(solver_mode="schedule_legacy"),
            _Inputs(_top_pour(mesh)), progress_message_callback=messages.append,
        )
        self.assertIsNotNone(result)
        self.assertTrue(
            any("schedule_legacy" in w for w in result.warnings),
            f"no warning about the ignored seed: {result.warnings}",
        )


def _sprue_grid(nx=7, ny=7, nz=6, lo=2, hi=4):
    """A 3x3 sprue bore standing in otherwise empty space."""
    types = np.full((nx, ny, nz), EMPTY_CELL, dtype=np.int8)
    types[lo:hi + 1, lo:hi + 1, :] = PART_TYPE_SPRUE
    return types


class InletMouthWideningTests(unittest.TestCase):
    """
    The pour location picks the feature; its whole mouth is the inlet.

    Measured on ``Test_Model/Full_Gating_Test``: the 1.5-voxel resolution
    radius picked 8 cells of the sprue top's cross-section (48 on the current
    mesh). The 8-cell jet atomised on the way down the sprue and the run
    diverged at 24 % by mass. Widening it to the whole mouth clears that: the
    same model now runs the full 45 s cap without diverging, at 91 % of cells
    and 97 % by mass. See :func:`core.filling_stage._expand_inlet_to_mouth`.
    """

    def test_a_single_seed_widens_to_the_whole_mouth(self) -> None:
        """One cell on a 3x3 sprue top becomes all nine."""
        types = _sprue_grid()
        seed = np.array([[3, 3, 5]], dtype=np.int64)
        widened = _expand_inlet_to_mouth(seed, types)
        self.assertEqual(widened.shape[0], 9)
        self.assertEqual(
            {tuple(c) for c in widened},
            {(x, y, 5) for x in (2, 3, 4) for y in (2, 3, 4)},
        )

    def test_it_does_not_leak_into_an_adjacent_part(self) -> None:
        """
        A runner sharing the layer, touching the sprue, stays out.

        This is the property that keeps the widening a *geometry* step rather
        than a flood over all the metal in the assembly.
        """
        types = _sprue_grid()
        types[5, 2:5, 5] = PART_TYPE_RUNNER  # face-adjacent to the sprue
        types[1, 1, 5] = PART_TYPE_CAST      # corner-adjacent to the sprue
        widened = _expand_inlet_to_mouth(np.array([[3, 3, 5]]), types)
        codes = types[widened[:, 0], widened[:, 1], widened[:, 2]]
        self.assertEqual(set(np.unique(codes).tolist()), {int(PART_TYPE_SPRUE)})
        self.assertEqual(widened.shape[0], 9)

    def test_it_does_not_leak_into_a_non_metal_neighbour(self) -> None:
        """A core touching the mouth is mold, not inlet."""
        types = _sprue_grid()
        types[5, 2:5, 5] = PART_TYPE_CORE
        widened = _expand_inlet_to_mouth(np.array([[3, 3, 5]]), types)
        self.assertEqual(widened.shape[0], 9)
        self.assertTrue(bool(np.all(widened[:, 0] <= 4)))

    def test_it_does_not_leave_the_layer(self) -> None:
        """The bore below the mouth is not inlet; only the top face is."""
        types = _sprue_grid()
        widened = _expand_inlet_to_mouth(np.array([[3, 3, 5]]), types)
        self.assertEqual(set(np.unique(widened[:, 2]).tolist()), {5})

    def test_a_seed_already_covering_the_mouth_is_unchanged(self) -> None:
        """Widening is idempotent, so it can be applied unconditionally."""
        types = _sprue_grid()
        mouth = np.array(
            [[x, y, 5] for x in (2, 3, 4) for y in (2, 3, 4)], dtype=np.int64
        )
        widened = _expand_inlet_to_mouth(mouth, types)
        np.testing.assert_array_equal(
            widened, np.unique(mouth, axis=0)
        )
        np.testing.assert_array_equal(
            _expand_inlet_to_mouth(widened, types), widened
        )

    def test_seeds_below_the_top_layer_are_kept(self) -> None:
        """The seeds are a floor on the result, not a starting guess."""
        types = _sprue_grid()
        seed = np.array([[3, 3, 5], [3, 3, 4]], dtype=np.int64)
        widened = _expand_inlet_to_mouth(seed, types)
        self.assertEqual(widened.shape[0], 10)
        self.assertIn((3, 3, 4), {tuple(c) for c in widened})

    def test_a_bare_casting_widens_over_its_own_top_face(self) -> None:
        """No gating in the model is not a special case."""
        types = np.full((5, 5, 4), EMPTY_CELL, dtype=np.int8)
        types[1:4, 1:4, :] = PART_TYPE_CAST
        widened = _expand_inlet_to_mouth(np.array([[2, 2, 3]]), types)
        self.assertEqual(widened.shape[0], 9)

    def test_an_empty_inlet_stays_empty(self) -> None:
        """An unresolvable pour point is reported upstream, not widened."""
        types = _sprue_grid()
        widened = _expand_inlet_to_mouth(
            np.zeros((0, 3), dtype=np.int64), types
        )
        self.assertEqual(widened.shape, (0, 3))

    def test_two_separate_mouths_do_not_merge(self) -> None:
        """Flood-fill connectivity, not "every cell of this type"."""
        types = np.full((9, 5, 4), EMPTY_CELL, dtype=np.int8)
        types[1:4, 1:4, :] = PART_TYPE_SPRUE
        types[6:9, 1:4, :] = PART_TYPE_SPRUE  # a second, disconnected sprue
        widened = _expand_inlet_to_mouth(np.array([[2, 2, 3]]), types)
        self.assertEqual(widened.shape[0], 9)
        self.assertTrue(bool(np.all(widened[:, 0] <= 3)))

    def _wide_bore(self, width=7, nz=6, dx=0.25):
        """
        A metal bore wider than the resolution radius, inside a mold.

        Bounded rather than a bare block: a fully-occupied mesh has no
        obstacle cells, so the lattice wraps periodically -- see
        ``tests/test_filling_stability._walled_tower``.
        """
        from core.pour_location import PourLocation

        n = width + 2
        centers = lattice_centers((n, n, nz + 2), dx=dx, origin=(0.0, 0.0, 0.0))
        index = np.rint(np.asarray(centers) / dx).astype(int)
        types = np.full(len(centers), PART_TYPE_CORE, dtype=np.int8)
        inside = (
            (index[:, 0] >= 1) & (index[:, 0] <= width)
            & (index[:, 1] >= 1) & (index[:, 1] <= width)
            & (index[:, 2] >= 1) & (index[:, 2] <= nz)
        )
        types[inside] = PART_TYPE_CAST
        mesh = _FakeMesh(centers, types, dx)
        metal = np.asarray(centers)[inside]
        pour = PourLocation(point_in=(
            float(metal[:, 0].mean()),
            float(metal[:, 1].mean()),
            float(metal[:, 2].max()),
        ))
        return mesh, pour

    def _brief_options(self):
        """Options with their own FillingParams, capped to a few steps.

        ``_Options`` shares one class-level ``_Filling``, so overriding on it
        would leak into every other test in the file.
        """
        params = _Params()
        params.filling = _Filling()
        params.filling.max_fill_time_s = 1e-4
        options = _Options()
        options.solidification_params = params
        return options

    def test_the_stage_widens_a_real_pour_to_the_whole_mouth(self) -> None:
        """End to end: the pinned inlet is the mouth, not the click."""
        from core.filling_stage import (
            _resolve_inlet_cells, build_grid_mapping, coarsen_part_types,
            dense_part_types, maybe_run_filling,
        )

        mesh, pour = self._wide_bore()
        mapping = build_grid_mapping(np.asarray(mesh.voxel_centers), 0.25)
        types = coarsen_part_types(
            dense_part_types(mapping, mesh.voxel_part_types), 1
        )
        seeds = _resolve_inlet_cells(pour, mapping, types, 1)
        self.assertLess(seeds.shape[0], 49, "fixture no longer under-resolves")

        messages: list = []
        result = maybe_run_filling(
            mesh, _Alloy(), self._brief_options(), _Inputs(pour),
            progress_message_callback=messages.append,
        )
        self.assertIsNotNone(result)
        # 49 for the 7x7 mouth, plus the seeds one layer down that the
        # resolution radius also picked up and that the union keeps.
        self.assertGreaterEqual(result.inlet_voxel_count, 49)
        widened = _expand_inlet_to_mouth(seeds, types)
        self.assertEqual(result.inlet_voxel_count, widened.shape[0])
        self.assertEqual(int((widened[:, 2] == widened[:, 2].max()).sum()), 49)

    def test_the_stage_reports_the_widening(self) -> None:
        """The emit has to say the inlet is not where the user clicked."""
        from core.filling_stage import maybe_run_filling

        mesh, pour = self._wide_bore()
        messages: list = []
        maybe_run_filling(
            mesh, _Alloy(), self._brief_options(), _Inputs(pour),
            progress_message_callback=messages.append,
        )
        inlet_lines = [m for m in messages if m.startswith("Filling: inlet")]
        self.assertTrue(inlet_lines, f"no inlet line in {messages}")
        self.assertIn(
            "inlet 54 cells (14 at the pour point, widened to the cast mouth)",
            inlet_lines[0],
        )

    def test_an_already_full_mouth_is_reported_without_a_widening_note(
        self,
    ) -> None:
        """A 2x2 column resolves to its whole mouth already; say nothing."""
        from core.filling_stage import maybe_run_filling

        mesh = _column_mesh()
        messages: list = []
        maybe_run_filling(
            mesh, _Alloy(), self._brief_options(), _Inputs(_top_pour(mesh)),
            progress_message_callback=messages.append,
        )
        inlet_lines = [m for m in messages if m.startswith("Filling: inlet")]
        self.assertTrue(inlet_lines, f"no inlet line in {messages}")
        self.assertNotIn("widened", inlet_lines[0])


def _choke_rig(gate_thickness, sprue_thickness=8, n=24):
    """
    A synthetic sprue-and-gate on the mesh grid, sized in fine cells.

    A square sprue column of ``sprue_thickness`` cells, crossed by a gate
    plate ``gate_thickness`` cells thick. Coarsening halves both, so the
    gate's thickness at any factor is predictable and the auto-resolution
    decision can be exercised without running a fill.

    Verified against measure_choke_thickness_by_part rather than assumed:
    a 4-cell gate reads 3.0 cells at coarsening 1 and 1.0 at coarsening 2,
    which is exactly the "resolved / sealed" pair the decision turns on.
    """
    grid = np.full((n, n, n), EMPTY_CELL, dtype=np.int8)
    a = (n - sprue_thickness) // 2
    grid[a:a + sprue_thickness, a:a + sprue_thickness, 8:] = PART_TYPE_SPRUE
    b = (n - gate_thickness) // 2
    grid[:, a:a + sprue_thickness, b:b + gate_thickness] = PART_TYPE_GATE
    return grid


class ChokeAutoResolutionTests(unittest.TestCase):
    """
    Requirement: a gating channel too thin for the grid refines the grid.

    A channel at or below MIN_CHOKE_THICKNESS_CELLS has no interior once
    halfway bounce-back has placed the walls, so it seals and the fill
    through it is fiction. Warning about a grid the stage picked itself is
    only half an answer when it could pick a better one.

    The decision is exercised on synthetic type grids: it is pure geometry,
    and running fills to test it would test the solver instead.
    """

    def test_a_sealed_gate_forces_the_grid_finer(self) -> None:
        grid = _choke_rig(gate_thickness=4)
        # Precondition: the case is the one being claimed.
        self.assertLessEqual(
            measure_choke_thickness_by_part(coarsen_part_types(grid, 2))[PART_TYPE_GATE],
            MIN_CHOKE_THICKNESS_CELLS,
        )
        decision = choose_fill_coarsening(grid, 2)
        self.assertEqual(decision.coarsening, 1)
        self.assertEqual(decision.configured, 2)
        self.assertTrue(decision.refined)
        self.assertTrue(decision.resolved)
        self.assertEqual(decision.forced_by, PART_TYPE_GATE)
        self.assertGreater(
            decision.by_part[PART_TYPE_GATE], MIN_CHOKE_THICKNESS_CELLS
        )

    def test_the_chosen_grid_comes_back_with_the_decision(self) -> None:
        """The caller must not have to re-derive -- or re-downsample -- it."""
        grid = _choke_rig(gate_thickness=4)
        decision = choose_fill_coarsening(grid, 2)
        np.testing.assert_array_equal(
            decision.solve_types, coarsen_part_types(grid, decision.coarsening)
        )

    def test_a_resolved_gate_is_left_alone(self) -> None:
        """The common case: nothing is sealed, so nothing is paid for."""
        grid = _choke_rig(gate_thickness=6)
        decision = choose_fill_coarsening(grid, 2)
        self.assertEqual(decision.coarsening, 2)
        self.assertFalse(decision.refined)
        self.assertTrue(decision.resolved)
        self.assertIsNone(decision.forced_by)

    def test_a_gate_that_cannot_be_fixed_lands_at_one_and_says_so(self) -> None:
        """
        Two fine cells across is the mesh's fault, not the coarsening's.

        Refining to 1 cannot help, so the decision stops there, reports
        resolved=False, and the caller keeps the hard warning. An honest
        misrun beats a silent one.
        """
        grid = _choke_rig(gate_thickness=2)
        decision = choose_fill_coarsening(grid, 2)
        self.assertEqual(decision.coarsening, 1)
        self.assertFalse(decision.resolved)
        self.assertEqual(decision.forced_by, PART_TYPE_GATE)
        self.assertLessEqual(
            decision.by_part[PART_TYPE_GATE], MIN_CHOKE_THICKNESS_CELLS
        )

    def test_the_switch_off_pins_the_configured_coarsening(self) -> None:
        """
        A user who pinned the grid gets the grid they pinned.

        The measurement still happens -- forced_by still names the sealed
        part, so the caller's warning is unaffected -- it simply is not acted
        on.
        """
        grid = _choke_rig(gate_thickness=4)
        decision = choose_fill_coarsening(grid, 2, enabled=False)
        self.assertEqual(decision.coarsening, 2)
        self.assertFalse(decision.refined)
        self.assertFalse(decision.resolved)
        self.assertEqual(decision.forced_by, PART_TYPE_GATE)
        self.assertFalse(decision.enabled)

    def test_it_steps_down_rather_than_jumping_to_one(self) -> None:
        """
        The cheapest grid that carries the fill, not the finest available.

        This rig is sealed at coarsening 4 and clear at 3, so 3 is the
        answer -- refining to 1 would cost 64x the cells for nothing.
        """
        grid = _choke_rig(gate_thickness=6)
        self.assertLessEqual(
            measure_choke_thickness_by_part(coarsen_part_types(grid, 4))[PART_TYPE_GATE],
            MIN_CHOKE_THICKNESS_CELLS,
        )
        decision = choose_fill_coarsening(grid, 4)
        self.assertEqual(decision.coarsening, 3)
        self.assertTrue(decision.resolved)

    def test_coarsening_one_is_returned_untouched(self) -> None:
        """Nothing to refine to; the measurement still has to be honest."""
        grid = _choke_rig(gate_thickness=2)
        decision = choose_fill_coarsening(grid, 1)
        self.assertEqual(decision.coarsening, 1)
        self.assertFalse(decision.resolved)
        self.assertEqual(decision.cells_configured, decision.cells_chosen)

    def test_a_grid_with_no_gating_metal_is_a_no_op(self) -> None:
        grid = np.full((8, 8, 8), PART_TYPE_CAST, dtype=np.int8)
        decision = choose_fill_coarsening(grid, 2)
        self.assertEqual(decision.coarsening, 2)
        self.assertEqual(decision.by_part, {})
        self.assertTrue(decision.resolved)


def _isolated_options():
    """
    A fresh options tree, unshared with any other test module.

    ``_Options`` hands out the module-level ``_Params``, whose ``filling`` is
    a single shared instance -- and ``tests/test_filling_stability`` imports
    ``_Options`` and writes FillingParams overrides straight onto it, which
    survive into whatever runs next. A test that has to run at the shipped
    defaults cannot use the shared one: under full-suite ordering it inherits
    a lattice velocity cap and a smagorinsky constant chosen for somebody
    else's fixture, and the tiny column then diverges.
    """
    class _IsolatedParams:
        filling = _Filling()
        thermal_boundary = _ThermalBoundary()

    options = _Options(filling=True)
    options.solidification_params = _IsolatedParams()
    return options


class DeliveryLedgerTests(unittest.TestCase):
    """
    Requirement: the poured counter is the integral of what was injected.

    The inlet now adds mass in two places -- the pressure pin's refill and
    the metered top-up into the throat -- and a delivery figure that did not
    account for both would be the same lie the old self-metering inlet told,
    just from the other direction.

    Measured across the step boundary: everything the stage adds happens
    between one call to physics.filling.step and the next, so differencing
    the domain's total mass there captures exactly the injected term and
    nothing the free-surface pass does inside the step.
    """

    def test_poured_equals_the_mass_actually_injected(self) -> None:
        import physics.filling as physics_filling
        from core.filling_stage import maybe_run_filling
        from physics.filling import REFERENCE_DENSITY

        mesh = _column_mesh(nz=8)
        options = _isolated_options()
        # run_filling_stage imports both of these by name inside the call, so
        # the module attributes are what it will bind.
        original_step = physics_filling.step
        original_skin = physics_filling.ensure_interface_skin
        original_create = physics_filling.create_state
        seen = {
            "between_steps": 0.0, "skin": 0.0, "seed": 0.0, "previous_exit": None,
        }

        def recording_create_state(*args, **kwargs):
            # create_state seeds the pour point FLUID *and* an interface skin
            # around it at half fill. That mass was placed before the pour
            # started and is not delivery, so it is measured rather than
            # assumed -- the skin's share is 4.5 of 11.5 on this column, and
            # hard-coding either number would make the test a copy of the
            # implementation.
            state = original_create(*args, **kwargs)
            seen["seed"] = float(np.sum(state.mass))
            return state

        def recording_skin(state, halo, **kwargs):
            # The inlet's interface skin seeds newly promoted cells, and that
            # seeding is a mass source the pour ledger deliberately does not
            # claim. Measured so the ledger can be checked against the
            # injection alone.
            before = float(np.sum(state.mass))
            promoted = original_skin(state, halo, **kwargs)
            seen["skin"] += float(np.sum(state.mass)) - before
            return promoted

        def recording_step(state, units, **kwargs):
            entry = float(np.sum(state.mass))
            previous = seen["previous_exit"]
            seen["between_steps"] += entry - (previous if previous is not None else 0.0)
            out = original_step(state, units, **kwargs)
            seen["previous_exit"] = float(np.sum(state.mass))
            return out

        physics_filling.step = recording_step
        physics_filling.ensure_interface_skin = recording_skin
        physics_filling.create_state = recording_create_state
        try:
            result = maybe_run_filling(
                mesh, _Alloy(), options, _Inputs(_top_pour(mesh, nz=8)),
            )
        finally:
            physics_filling.step = original_step
            physics_filling.ensure_interface_skin = original_skin
            physics_filling.create_state = original_create

        self.assertIsNotNone(result)
        assert result is not None
        # Everything that entered state.mass between one step and the next,
        # less the two sources that are not the pour: the cells create_state
        # seeded before the loop, and the interface skin's seeding.
        dx_m = float(0.25 * 0.0254) * result.coarsening
        lb_per_mass = (
            _Alloy.density_liquid * dx_m ** 3
            / float(REFERENCE_DENSITY) / 0.45359237
        )
        injected = seen["between_steps"] - seen["seed"] - seen["skin"]
        self.assertGreater(seen["seed"], 0.0)
        self.assertGreater(result.delivered_lb, 0.0)
        self.assertAlmostEqual(
            result.delivered_lb / lb_per_mass,
            injected,
            delta=1e-9 * max(1.0, abs(injected)),
        )

    def test_the_target_rate_is_reported(self) -> None:
        """Starvation has to be legible without instrumenting the solver."""
        from core.filling_stage import maybe_run_filling

        mesh = _column_mesh(nz=8)
        messages: list = []
        result = maybe_run_filling(
            mesh, _Alloy(), _isolated_options(), _Inputs(_top_pour(mesh, nz=8)),
            progress_message_callback=messages.append,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreater(result.target_delivery_lb_s, 0.0)
        self.assertTrue(
            any("metering" in m for m in messages),
            f"the setup line does not name the target rate: {messages}",
        )


def _flush_riser_mesh(nx=10, ny=8, cast_top=5, nz=13, dx=DX_IN):
    """
    Test 2's topology in miniature: metal flush against two opposite z faces.

    A CAST box whose floor is the grid's own ``z = 0`` plane -- there is no mold
    below it, because ``build_grid_mapping`` sizes the grid to the voxel
    centres' bounding box and nothing was meshed beyond the casting -- with a
    RISER column and a SPRUE column both reaching the ``z = nz - 1`` plane.
    Under periodic neighbour indexing the cast floor and those tops are
    neighbours of each other, 12 empty planes apart, which is what seeded a
    phantom pool in the real assembly's riser top at step 12,156. Cores fill the
    x/y margin, so z is the only unwalled axis and the assertions can name it.

    The sprue is there so the pour has somewhere to go that is not the riser:
    the headspace deliberately skips the inlet mouth (see
    :data:`~core.filling_stage.HEADSPACE_CELLS`), so a fixture poured into its
    only top-face feature could not show the headspace at all.
    """
    centers = lattice_centers((nx, ny, nz), dx=dx, origin=(0.0, 0.0, 0.0))
    index = np.rint(np.asarray(centers) / dx).astype(int)
    types = np.full(len(centers), PART_TYPE_CORE, dtype=np.int8)
    in_x = (index[:, 0] >= 1) & (index[:, 0] <= nx - 3)
    in_y = (index[:, 1] >= 1) & (index[:, 1] <= ny - 2)
    cast = in_x & in_y & (index[:, 2] <= cast_top)
    column = (index[:, 1] >= 3) & (index[:, 1] <= 4) & (index[:, 2] >= cast_top)
    riser = column & (index[:, 0] >= 2) & (index[:, 0] <= 3)
    sprue = column & (index[:, 0] >= 6) & (index[:, 0] <= 7)
    types[cast] = PART_TYPE_CAST
    types[riser] = PART_TYPE_RISER
    types[sprue] = PART_TYPE_SPRUE
    mesh = _FakeMesh(centers, types, dx)
    mesh._riser_top_xy = (2.5 * dx, 3.5 * dx)
    mesh._sprue_top_xy = (6.5 * dx, 3.5 * dx)
    mesh._top_z = (nz - 1) * dx
    return mesh


def _touching_tops_mesh(nx=10, ny=8, cast_top=5, nz=13, dx=DX_IN):
    """
    :func:`_flush_riser_mesh` with its riser and sprue columns side by side.

    The separated version cannot see the headspace's lateral leak. The inlet's
    ``ensure_interface_skin`` halo is an 18-direction dilation, so its
    ``c_z = +1`` diagonals reach the first headspace plane over the columns
    *beside* the mouth -- and the exclusion has to cover the mouth's whole
    8-neighbourhood in ``(x, y)``, not just the mouth. With the riser two cells
    from the sprue there is no column in that ring to leak onto; with them
    touching, a mouth-only exclusion leaves the riser's near column open inside
    the halo's reach.
    """
    centers = lattice_centers((nx, ny, nz), dx=dx, origin=(0.0, 0.0, 0.0))
    index = np.rint(np.asarray(centers) / dx).astype(int)
    types = np.full(len(centers), PART_TYPE_CORE, dtype=np.int8)
    in_x = (index[:, 0] >= 1) & (index[:, 0] <= nx - 2)
    in_y = (index[:, 1] >= 1) & (index[:, 1] <= ny - 2)
    types[in_x & in_y & (index[:, 2] <= cast_top)] = PART_TYPE_CAST
    column = (index[:, 1] >= 3) & (index[:, 1] <= 4) & (index[:, 2] >= cast_top)
    types[column & (index[:, 0] >= 3) & (index[:, 0] <= 4)] = PART_TYPE_RISER
    types[column & (index[:, 0] >= 5) & (index[:, 0] <= 6)] = PART_TYPE_SPRUE
    mesh = _FakeMesh(centers, types, dx)
    mesh._riser_top_xy = (3.5 * dx, 3.5 * dx)
    mesh._sprue_top_xy = (5.5 * dx, 3.5 * dx)
    mesh._top_z = (nz - 1) * dx
    return mesh


class MoldPadTests(unittest.TestCase):
    """
    The solve grid is padded with mold before it is solved, and nothing about
    the mapping back to voxels moves.

    ``core.filling_stage.MOLD_PAD_CELLS`` carries the measurement. The short
    version: every neighbour access in both step paths is periodic, both
    document that they rely on an obstacle shell, and the grid the stage used
    to build had none -- 302 metal-to-metal wrap pairs across z on the real
    assembly, 298 of them putting a cast-floor cell directly opposite a
    riser-top cell 93 planes away.

    Padding is the whole fix, so it must cost nothing that matters: no metal
    cell, no riser-top vent, and no voxel's identity. Those are the four
    assertions here.
    """

    def _unpadded_types(self, mesh):
        """The bounding-box part-type grid, before the stage pads it."""
        from core.filling_stage import coarsen_part_types, dense_part_types

        centers = np.asarray(mesh.voxel_centers, dtype=np.float64)
        dx_in = float(np.asarray(mesh.voxel_sizes).ravel()[0])
        mapping = build_grid_mapping(centers, dx_in)
        return coarsen_part_types(
            dense_part_types(mapping, mesh.voxel_part_types), 1
        )

    def _padded_types(self, mesh):
        """The same grid padded exactly as ``run_filling_stage`` pads it."""
        from core.filling_stage import HEADSPACE_CELLS, MOLD_PAD_CELLS

        pad = int(MOLD_PAD_CELLS)
        return np.pad(
            self._unpadded_types(mesh),
            ((pad, pad), (pad, pad), (pad, pad + int(HEADSPACE_CELLS))),
            mode="constant",
            constant_values=EMPTY_CELL,
        )

    def _solved(self, mesh, pour):
        """
        Run the real stage and hand back the ``obstacle`` mask it built.

        Captured from ``create_state``'s own argument rather than rebuilt here,
        so the headspace and the invariant are the stage's and not a
        re-derivation that could agree with a broken original.
        """
        import physics.filling as pf
        from core.filling_stage import maybe_run_filling

        held = {}
        original = pf.create_state

        def capture(obstacle, **kwargs):
            state = original(obstacle, **kwargs)
            held.setdefault("obstacle", np.asarray(obstacle).copy())
            return state

        pf.create_state = capture
        try:
            result = maybe_run_filling(
                mesh, _Alloy(), _isolated_options(), _Inputs(pour)
            )
        finally:
            pf.create_state = original
        self.assertIn("obstacle", held, "the stage never built a solver state")
        return result, held["obstacle"]

    def _sprue_pour(self, mesh):
        from core.pour_location import PourLocation

        return PourLocation(
            point_in=(mesh._sprue_top_xy[0], mesh._sprue_top_xy[1], mesh._top_z)
        )

    def _riser_pour(self, mesh):
        from core.pour_location import PourLocation

        return PourLocation(
            point_in=(mesh._riser_top_xy[0], mesh._riser_top_xy[1], mesh._top_z)
        )

    def _inlet_of(self, mesh, pour):
        """The padded inlet cells the stage will resolve this pour point to."""
        from core.filling_stage import (
            MOLD_PAD_CELLS,
            _expand_inlet_to_mouth,
            _resolve_inlet_cells,
        )

        centers = np.asarray(mesh.voxel_centers, dtype=np.float64)
        dx_in = float(np.asarray(mesh.voxel_sizes).ravel()[0])
        mapping = build_grid_mapping(centers, dx_in)
        core_types = self._unpadded_types(mesh)
        seeds = _resolve_inlet_cells(pour, mapping, core_types, 1)
        return _expand_inlet_to_mouth(seeds, core_types) + int(MOLD_PAD_CELLS)

    def test_the_unpadded_grid_really_is_open(self) -> None:
        """
        Guard the fixture. If the miniature already had walls on every face,
        every assertion below would pass on a grid that never had the defect.
        """
        from core.filling_stage import _obstacle_mask

        open_faces = _obstacle_mask(self._unpadded_types(_flush_riser_mesh()))
        self.assertFalse(
            bool(open_faces[:, :, 0].all()),
            "the fixture's z = 0 face is already walled",
        )
        self.assertFalse(
            bool(open_faces[:, :, -1].all()),
            "the fixture's top face is already walled",
        )
        self.assertTrue(bool(open_faces[0].all()), "x should be walled by cores")
        self.assertTrue(bool(open_faces[:, 0].all()), "y should be walled")

    def test_every_face_of_the_solved_grid_is_obstacle(self) -> None:
        """
        The invariant the stage now checks and refuses to run on. Both step
        paths wrap silently, so a single non-obstacle cell on an array face
        makes opposite ends of the domain neighbours and there is no error to
        read anywhere downstream.
        """
        mesh = _flush_riser_mesh()
        _, obstacle = self._solved(mesh, self._sprue_pour(mesh))
        for axis in (0, 1, 2):
            for face in (0, -1):
                plane = np.take(obstacle, face, axis=axis)
                self.assertTrue(
                    bool(plane.all()),
                    f"axis {axis} face {face} leaves "
                    f"{int(np.count_nonzero(~plane))} cells open",
                )

    def test_the_pad_costs_no_metal(self) -> None:
        """
        The pad is ``EMPTY_CELL``, which ``_metal_mask`` excludes, so the fill
        denominator is untouched and a padded run can still report 100 %.

        This is what separates padding from the one-line stopgap
        (``obstacle |= <the whole array envelope>``), which sealed the wrap by
        making 3,858 of the real assembly's 339,824 metal cells solid --
        including the riser's entire 1,801-cell top plane -- so the fill could
        never reach 100 % and the riser read 1,801 cells short.
        """
        from core.filling_stage import _metal_mask

        mesh = _flush_riser_mesh()
        padded_metal = _metal_mask(self._padded_types(mesh))
        core_metal = _metal_mask(self._unpadded_types(mesh))
        self.assertEqual(int(padded_metal.sum()), int(core_metal.sum()))
        self.assertGreater(int(padded_metal.sum()), 0)

    def test_the_pad_costs_no_riser_top_vent(self) -> None:
        """
        The riser keeps its atmosphere boundary, which is the other half of
        what padding buys over the stopgap.

        ``_static_vent_mask`` tests the *type* grid -- "the cell above a RISER
        cell is EMPTY_CELL" -- and above a padded riser top the pad is still
        EMPTY_CELL, so the count is unchanged with no code change at all.
        Measured on the real assembly: 2,247 riser-top vents before and after.
        The stopgap takes the same count to zero, and the riser's whole gas
        column then reads as entrapped.
        """
        from core.filling_stage import _static_vent_mask
        from core.part_manager import PART_TYPE_RISER

        mesh = _flush_riser_mesh()
        core_types = self._unpadded_types(mesh)
        padded = self._padded_types(mesh)
        no_inlet = np.zeros((0, 3), dtype=np.int64)
        before = _static_vent_mask(core_types, no_inlet, True) & (
            core_types == PART_TYPE_RISER
        )
        after = _static_vent_mask(padded, no_inlet, True) & (
            padded == PART_TYPE_RISER
        )
        self.assertGreater(int(before.sum()), 0, "the fixture has no open riser")
        self.assertEqual(int(after.sum()), int(before.sum()))

    def test_the_riser_top_keeps_an_open_headspace(self) -> None:
        """
        Metal reaching the original ``+z`` face gets real atmosphere above it,
        not a bounce-back lid one cell up.

        :data:`~core.filling_stage.HEADSPACE_CELLS` planes of the pad are left
        non-obstacle directly over it, and the plane above *those* stays solid
        so the array face is still sealed. The cells are ``EMPTY_CELL``-typed,
        so they are GAS to the solver and invisible to ``_metal_mask``.

        The inlet mouth is deliberately excluded: its stream halo is kept wet
        every step, and headspace there would grow an interface skin above the
        pour boundary and pool metal outside the fill ledger.
        """
        from core.filling_stage import (
            HEADSPACE_CELLS,
            MOLD_PAD_CELLS,
            _metal_mask,
        )

        mesh = _flush_riser_mesh()
        _, obstacle = self._solved(mesh, self._sprue_pour(mesh))
        pad = int(MOLD_PAD_CELLS)
        core_nz = self._unpadded_types(mesh).shape[2]
        riser_top_cells = 4  # the 2x2 riser column

        for offset in range(int(HEADSPACE_CELLS)):
            plane = obstacle[:, :, pad + core_nz + offset]
            self.assertEqual(
                int(np.count_nonzero(~plane)), riser_top_cells,
                f"headspace plane {offset} is open over "
                f"{int(np.count_nonzero(~plane))} cells, expected the riser's "
                f"{riser_top_cells} and not the sprue mouth's",
            )
        self.assertTrue(
            bool(obstacle[:, :, -1].all()),
            "the outermost pad plane must stay solid or the face wraps again",
        )
        # Headspace is not metal, so it changes neither side of the fill
        # fraction.
        padded_metal = _metal_mask(self._padded_types(mesh))
        self.assertEqual(
            int(padded_metal[:, :, pad + core_nz:].sum()), 0
        )

    def test_no_headspace_cell_is_reachable_by_the_inlet_halo(self) -> None:
        """
        The headspace must not hand the inlet's stream halo a cell to wet.

        ``inlet_halo`` is an 18-direction ``np.roll`` dilation of the mouth, and
        five of those directions carry ``c_z = +1``: they land in the first
        headspace plane over the columns diagonally beside the mouth. Clearing
        the headspace only over the mouth's own columns leaves such a column
        open, ``inlet_halo &= ~obstacle`` keeps it, and
        ``ensure_interface_skin`` promotes it on every step -- INTERFACE cells
        above the geometry, outside ``metal`` and cropped out of every result
        field, so their mass lands in ``poured`` and is reported nowhere else.
        Measured on Test 2's grid before the exclusion was widened: 0 such cells
        for a sprue-top pour, but 35 for a riser-top pour and 6 for a cast-top
        pour, both legal placements.

        Asserted on the fixture whose two top-face features touch, from both
        pour points, structurally (no headspace inside the mouth's
        8-neighbourhood) and behaviourally (nothing above the original top face
        is ever wet, including at ``t = 0``, where ``create_state``'s own
        ``_dilate`` would seed it before step 1).
        """
        import physics.filling as pf
        from scipy import ndimage

        from core.filling_stage import (
            HEADSPACE_CELLS,
            MOLD_PAD_CELLS,
            _metal_mask,
        )

        mesh = _touching_tops_mesh()
        pad = int(MOLD_PAD_CELLS)
        core_nz = self._unpadded_types(mesh).shape[2]
        padded_metal = _metal_mask(self._padded_types(mesh))
        top_metal = padded_metal[:, :, pad + core_nz - 1]

        for name, pour in (
            ("sprue", self._sprue_pour(mesh)),
            ("riser", self._riser_pour(mesh)),
        ):
            with self.subTest(pour=name):
                wet_above = []
                original = pf.step

                def traced(state, units, _wet=wet_above, **kwargs):
                    original(state, units, **kwargs)
                    above = state.flags[:, :, pad + core_nz:]
                    if np.any(
                        (above == pf.FLAG_FLUID) | (above == pf.FLAG_INTERFACE)
                    ):
                        _wet.append(int(state.step_count))
                    return state

                pf.step = traced
                try:
                    _, obstacle = self._solved(mesh, pour)
                finally:
                    pf.step = original

                # Structural: the mouth's whole 8-neighbourhood in (x, y) is
                # closed across every headspace plane.
                mouth = np.zeros(obstacle.shape[:2], dtype=bool)
                mouth_z = pad + core_nz - 1
                inlet_here = self._inlet_of(mesh, pour)
                on_top = inlet_here[:, 2] == mouth_z
                mouth[inlet_here[on_top, 0], inlet_here[on_top, 1]] = True
                reach = ndimage.binary_dilation(
                    mouth, structure=np.ones((3, 3), dtype=bool)
                )
                for offset in range(int(HEADSPACE_CELLS)):
                    plane = obstacle[:, :, pad + core_nz + offset]
                    self.assertEqual(
                        int(np.count_nonzero(~plane & reach)), 0,
                        f"headspace plane {offset} is open inside the mouth's "
                        f"8-neighbourhood, which is exactly where the halo's "
                        f"c_z = +1 diagonals land",
                    )
                    # And it is still open where it should be, or the test
                    # would pass by sealing everything.
                    self.assertGreater(
                        int(np.count_nonzero(~plane)), 0,
                        "no headspace at all -- the fixture stopped testing "
                        "the exclusion and started testing a lid",
                    )
                    self.assertTrue(
                        bool(np.all(top_metal[~plane])),
                        "headspace opened over a column that is not top-face "
                        "metal",
                    )

                # Behavioural: nothing above the geometry ever holds metal.
                self.assertEqual(
                    wet_above[:3], [],
                    f"cells above the original top face were wet at steps "
                    f"{wet_above[:3]}: the halo reached into the headspace",
                )

    def _vent_mask(self, mesh, pour, flags):
        """
        The vent mask ``run_filling_stage`` assembles, for arbitrary flags.

        ``obstacle`` is captured from the stage rather than re-derived, and the
        headspace is then simply "the non-obstacle cells above the original top
        face" -- read off that captured mask, not rebuilt from the rule -- so
        this cannot agree with a broken construction.
        """
        from core.filling_stage import (
            MOLD_PAD_CELLS,
            _open_air_mask,
            _static_vent_mask,
        )

        pad = int(MOLD_PAD_CELLS)
        core_shape = self._unpadded_types(mesh).shape
        padded = self._padded_types(mesh)
        _, obstacle = self._solved(mesh, pour)
        self.assertEqual(obstacle.shape, padded.shape)

        open_air = np.zeros(padded.shape, dtype=bool)
        open_air[
            pad:pad + core_shape[0], pad:pad + core_shape[1],
            pad:pad + core_shape[2],
        ] = _open_air_mask(core_shape)
        # The headspace, as the stage actually left it.
        open_air[:, :, pad + core_shape[2]:] |= ~obstacle[
            :, :, pad + core_shape[2]:
        ]
        open_air &= ~obstacle

        inlet = self._inlet_of(mesh, pour)
        static_vent = _static_vent_mask(padded, inlet, True)
        from physics.filling import FLAG_GAS

        return static_vent | (open_air & (flags == FLAG_GAS))

    def test_a_pocket_against_the_casting_floor_reads_as_entrapped(self) -> None:
        """
        The floor of the meshed bounding box is sand, so gas pinned against it
        is trapped -- and that is the classic gas-porosity site.

        The vent term used to take every one of the six original faces as "open
        to atmosphere by omission", which was the only defensible reading while
        nothing was modelled beyond the box. The mold pad settled it: five of
        those faces are now explicitly walled *and* physically are sand, so only
        ``+z`` is air. See ``core.filling_stage._open_air_mask``.

        Measured on this fixture: the six-face mask reports **0** entrapped
        cells for a pocket sitting on the casting floor; the shipped mask
        reports 1. The old answer was the exact inverse of the one the detector
        exists to give.

        The floor and the riser top only, because this fixture's ``x``/``y``
        margins are cores by design -- that is what makes ``z`` its only
        unwalled axis -- so no metal reaches a side face here. The four side
        faces are covered in
        ``tests/test_filling_defects.py::TestVentMaskConstruction`` on a grid
        whose metal does reach them.
        """
        from physics.filling import FLAG_FLUID, FLAG_GAS
        from physics.filling import find_entrapped_gas
        from core.filling_stage import MOLD_PAD_CELLS, _metal_mask, _static_vent_mask

        mesh = _flush_riser_mesh()
        pour = self._sprue_pour(mesh)
        pad = int(MOLD_PAD_CELLS)
        core_shape = self._unpadded_types(mesh).shape
        padded = self._padded_types(mesh)
        metal = _metal_mask(padded)

        # Everything metal is FLUID; three isolated GAS cells, chosen by where
        # they sit on the ORIGINAL grid's faces.
        flags = np.where(metal, FLAG_FLUID, np.uint8(0)).astype(np.uint8)
        floor = (2 + pad, 2 + pad, pad)                       # original z = 0
        riser_top = (2 + pad, 3 + pad, pad + core_shape[2] - 1)  # original top
        for cell in (floor, riser_top):
            self.assertTrue(
                bool(metal[cell]), f"fixture drifted: {cell} is not metal"
            )
            flags[cell] = FLAG_GAS

        vent_mask = self._vent_mask(mesh, pour, flags)
        entrapped = find_entrapped_gas(flags, vent_mask)

        self.assertTrue(
            bool(entrapped[floor]),
            "a gas pocket on the casting floor read as vented",
        )
        self.assertFalse(
            bool(entrapped[riser_top]),
            "the riser top must still vent to the headspace above it",
        )

        # What the six-face term gave, so this test names what it guards.
        six_face = np.zeros(padded.shape, dtype=bool)
        core = (
            slice(pad, pad + core_shape[0]),
            slice(pad, pad + core_shape[1]),
            slice(pad, pad + core_shape[2]),
        )
        envelope = np.zeros(core_shape, dtype=bool)
        envelope[[0, -1], :, :] = True
        envelope[:, [0, -1], :] = True
        envelope[:, :, [0, -1]] = True
        six_face[core] = envelope
        stale = _static_vent_mask(padded, self._inlet_of(mesh, pour), True) | (
            six_face & (flags == FLAG_GAS)
        )
        stale_entrapped = find_entrapped_gas(flags, stale)
        self.assertFalse(
            bool(stale_entrapped[floor]),
            "the fixture no longer discriminates: the six-face mask already "
            "called this pocket entrapped, so a pass above proves nothing",
        )

    def test_a_riser_enclosed_except_its_top_still_vents(self) -> None:
        """
        The regression the change must not cause.

        A riser walled by mold on every side with only its top open is the
        normal case, and it has to keep reading as vented -- from two
        independent routes, so trimming one cannot silence it: the type-based
        ``_static_vent_mask`` (a RISER cell with EMPTY_CELL above it) and the
        headspace the pad opens over the same column. Asserted with the gas
        column filling the riser's whole height, which is what a riser looks
        like before the metal arrives.
        """
        from physics.filling import FLAG_FLUID, FLAG_GAS, find_entrapped_gas
        from core.filling_stage import MOLD_PAD_CELLS, _metal_mask
        from core.part_manager import PART_TYPE_RISER

        mesh = _flush_riser_mesh()
        pour = self._sprue_pour(mesh)
        pad = int(MOLD_PAD_CELLS)
        padded = self._padded_types(mesh)
        metal = _metal_mask(padded)
        riser = (padded == PART_TYPE_RISER) & metal
        self.assertGreater(int(riser.sum()), 0, "the fixture lost its riser")

        flags = np.where(metal, FLAG_FLUID, np.uint8(0)).astype(np.uint8)
        flags[riser] = FLAG_GAS

        vent_mask = self._vent_mask(mesh, pour, flags)
        entrapped = find_entrapped_gas(flags, vent_mask)
        self.assertEqual(
            int(np.count_nonzero(entrapped)), 0,
            f"{int(np.count_nonzero(entrapped))} riser cells read as entrapped "
            f"though the riser is open at the top",
        )

    def test_the_result_still_maps_back_to_the_right_voxels(self) -> None:
        """
        The off-by-one this fix could have introduced, asserted end to end.

        Every solve field is cropped back to the original grid before it is
        upsampled and gathered (``run_filling_stage``'s ``to_voxels``). Miss the
        crop and ``gather_to_voxels`` either raises on the shape or -- at a
        coarsening above 1 -- silently reads every voxel one cell off along all
        three axes. ``fill_time_s`` is NaN exactly where a voxel is not metal,
        so comparing that NaN set against the mesh's own part types catches a
        shift of one cell in any direction.
        """
        from core.filling_stage import STATUS_DIVERGED
        from core.part_manager import is_metal

        mesh = _flush_riser_mesh()
        result, _ = self._solved(mesh, self._sprue_pour(mesh))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertNotEqual(result.status, STATUS_DIVERGED, result.warnings)

        voxel_metal = is_metal(np.asarray(mesh.voxel_part_types, dtype=np.int32))
        np.testing.assert_array_equal(
            np.isnan(result.fill_time_s), ~voxel_metal,
            "fill_time_s is NaN on the wrong voxels: the pad was not cropped "
            "off before the gather, or was cropped in the wrong place",
        )
        self.assertEqual(
            result.filled_voxel_count + result.misrun_voxel_count,
            int(voxel_metal.sum()),
        )
        self.assertEqual(len(result.temperature_c), mesh.get_voxel_count())
        self.assertTrue(bool(np.all(np.isfinite(result.temperature_c))))
        # At coarsening 1 the correspondence is exact. Above 1 it is not, and
        # never was: COARSENING_PRIORITY can vote a coarse cell CORE over a
        # metal voxel, so 100 of a 2,068-voxel variant of this geometry read as
        # non-metal at coarsening 2 -- measured identical before and after the
        # pad, which is how the crop was checked at that coarsening.
        self.assertEqual(result.coarsening, 1)
