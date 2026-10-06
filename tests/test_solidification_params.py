from __future__ import annotations

import unittest

from core.solidification_params import (
    SHRINK_SCALAR_DEFAULT,
    SHRINK_SCALAR_MAX,
    SHRINK_SCALAR_MIN,
    SolidificationParameters,
)


class SolidificationParametersTests(unittest.TestCase):
    def test_shrink_scalar_default_and_serialization(self) -> None:
        params = SolidificationParameters()

        self.assertAlmostEqual(params.feeding.shrink_scalar, SHRINK_SCALAR_DEFAULT)
        self.assertAlmostEqual(
            params.to_dict()["feeding"]["shrink_scalar"],
            SHRINK_SCALAR_DEFAULT,
        )

    def test_shrink_scalar_parses_from_nested_dict(self) -> None:
        params = SolidificationParameters.from_dict(
            {"feeding": {"shrink_scalar": 2.25}}
        )

        self.assertAlmostEqual(params.feeding.shrink_scalar, 2.25)

    def test_shrink_scalar_parsing_is_case_insensitive_for_environment_shape(self) -> None:
        params = SolidificationParameters.from_dict(
            {"FEEDING": {"SHRINK_SCALAR": 1.75}}
        )

        self.assertAlmostEqual(params.feeding.shrink_scalar, 1.75)

    def test_shrink_scalar_clamps_to_supported_range(self) -> None:
        high = SolidificationParameters.from_dict(
            {"feeding": {"shrink_scalar": 99.0}}
        )
        low = SolidificationParameters.from_dict(
            {"feeding": {"shrink_scalar": -3.0}}
        )

        self.assertAlmostEqual(high.feeding.shrink_scalar, SHRINK_SCALAR_MAX)
        self.assertAlmostEqual(low.feeding.shrink_scalar, SHRINK_SCALAR_MIN)

    def test_shrink_scalar_invalid_value_uses_default(self) -> None:
        params = SolidificationParameters.from_dict(
            {"feeding": {"shrink_scalar": "not-a-number"}}
        )

        self.assertAlmostEqual(params.feeding.shrink_scalar, SHRINK_SCALAR_DEFAULT)

    def test_feeding_zone_vertical_bias_default_is_half(self) -> None:
        params = SolidificationParameters()

        self.assertAlmostEqual(
            params.feeding.feeding_zone_vertical_bias, 0.0, places=6
        )
        self.assertAlmostEqual(
            params.to_dict()["feeding"]["feeding_zone_vertical_bias"], 0.0, places=6
        )

    def test_feeding_zone_vertical_bias_parses_from_nested_dict(self) -> None:
        params = SolidificationParameters.from_dict(
            {"feeding": {"feeding_zone_vertical_bias": 1.25}}
        )

        self.assertAlmostEqual(params.feeding.feeding_zone_vertical_bias, 1.25)

    def test_feeding_zone_vertical_bias_clamps_to_supported_range(self) -> None:
        high = SolidificationParameters.from_dict(
            {"feeding": {"feeding_zone_vertical_bias": 99.0}}
        )
        low = SolidificationParameters.from_dict(
            {"feeding": {"feeding_zone_vertical_bias": -2.0}}
        )

        self.assertAlmostEqual(high.feeding.feeding_zone_vertical_bias, 2.0)
        self.assertAlmostEqual(low.feeding.feeding_zone_vertical_bias, 0.0)

    def test_feeding_zone_vertical_bias_invalid_value_uses_default(self) -> None:
        params = SolidificationParameters.from_dict(
            {"feeding": {"feeding_zone_vertical_bias": "not-a-number"}}
        )

        self.assertAlmostEqual(
            params.feeding.feeding_zone_vertical_bias, 0.0, places=6
        )

    def test_feeding_zone_vertical_bias_round_trips(self) -> None:
        original = SolidificationParameters.from_dict(
            {"feeding": {"feeding_zone_vertical_bias": 0.85}}
        )
        round_tripped = SolidificationParameters.from_dict(original.to_dict())

        self.assertAlmostEqual(
            round_tripped.feeding.feeding_zone_vertical_bias, 0.85, places=6
        )

    def test_feeding_zone_vertical_bias_strengthens_directional_solidification(self) -> None:
        """Higher bias should make bottom-feeding-zone voxels solidify faster
        while leaving feeding-zone voxels at the top of the casting unchanged."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable")

        from core.acceleration import update_solidification_with_feeding_connectivity

        n_neighbors = 26
        n_voxels = 4
        initial = np.zeros(n_voxels, dtype=np.float32)
        initial[3] = 1.0
        neighbors = np.full((n_voxels, n_neighbors), -1, dtype=np.int32)
        # Voxel 0 (CAST, in feeding zone via riser neighbor, height varies)
        # needs both a riser neighbor (1) AND a solid neighbor (3) to trigger
        # the solidification-update branch.
        neighbors[0, 0] = 1
        neighbors[0, 1] = 3
        # Voxel 1 (RISER) neighbors the solid voxel (3).
        neighbors[1, 0] = 3
        # Voxel 2 (CAST, outside feeding zone) neighbors the solid voxel.
        neighbors[2, 0] = 3
        moduli = np.full(n_voxels, 1.0, dtype=np.float32)
        heights_bottom = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        heights_top = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        chill_boost = np.ones(n_voxels, dtype=np.float32)
        riser_slowdown = np.array([0.5, 0.5, 0.9, 1.0], dtype=np.float32)
        part_types = np.array([0, 1, 0, 0], dtype=np.int32)

        def run(bias: float, heights: np.ndarray) -> np.ndarray:
            states = initial.copy()
            for _ in range(50):
                states = update_solidification_with_feeding_connectivity(
                    states, neighbors, moduli, heights,
                    chill_boost, riser_slowdown, part_types,
                    0.1, np.float32(1.0), np.float32(bias),
                )
            return states

        weak_bottom = run(0.3, heights_bottom)
        strong_bottom = run(1.2, heights_bottom)
        self.assertGreater(
            float(strong_bottom[0]),
            float(weak_bottom[0]),
            "Stronger feeding-zone bias must accelerate bottom-voxel solidification",
        )

        weak_top = run(0.3, heights_top)
        strong_top = run(1.2, heights_top)
        self.assertAlmostEqual(
            float(strong_top[0]), float(weak_top[0]), places=6,
            msg=(
                "At z=1 the vertical_factor equals 1.0 in the feeding zone, "
                "so the bias must not change the top-voxel solidification rate"
            ),
        )

    def test_air_gap_htc_floor_default_and_clamp(self) -> None:
        params = SolidificationParameters()

        self.assertAlmostEqual(params.air_gap.htc_floor_fraction, 0.35)

        high = SolidificationParameters.from_dict(
            {"air_gap": {"htc_floor_fraction": 2.0}}
        )
        low = SolidificationParameters.from_dict(
            {"air_gap": {"htc_floor_fraction": 0.0}}
        )

        self.assertAlmostEqual(high.air_gap.htc_floor_fraction, 1.0)
        self.assertAlmostEqual(low.air_gap.htc_floor_fraction, 0.05)

    def test_chill_ambient_htc_default_and_clamp(self) -> None:
        params = SolidificationParameters()

        self.assertAlmostEqual(params.thermal_boundary.chill_ambient_htc, 500.0)

        high = SolidificationParameters.from_dict(
            {"thermal_boundary": {"chill_ambient_htc": 50000.0}}
        )
        low = SolidificationParameters.from_dict(
            {"thermal_boundary": {"chill_ambient_htc": 0.0}}
        )

        self.assertAlmostEqual(high.thermal_boundary.chill_ambient_htc, 10000.0)
        self.assertAlmostEqual(low.thermal_boundary.chill_ambient_htc, 1.0)

    def test_air_gap_and_chill_ambient_round_trip(self) -> None:
        original = SolidificationParameters.from_dict(
            {
                "air_gap": {"htc_floor_fraction": 0.5},
                "thermal_boundary": {
                    "chill_ambient_htc": 250.0,
                    "mold_thermal_conductivity": 1.2,
                    "mold_thermal_diffusivity": 5.0e-7,
                },
            }
        )
        serialized = original.to_dict()
        self.assertIn("htc_floor_fraction", serialized["air_gap"])
        self.assertIn("chill_ambient_htc", serialized["thermal_boundary"])
        self.assertIn("mold_thermal_conductivity", serialized["thermal_boundary"])
        self.assertIn("mold_thermal_diffusivity", serialized["thermal_boundary"])

        round_tripped = SolidificationParameters.from_dict(serialized)

        self.assertAlmostEqual(round_tripped.air_gap.htc_floor_fraction, 0.5)
        self.assertAlmostEqual(round_tripped.thermal_boundary.chill_ambient_htc, 250.0)
        self.assertAlmostEqual(round_tripped.thermal_boundary.mold_thermal_conductivity, 1.2)
        self.assertAlmostEqual(round_tripped.thermal_boundary.mold_thermal_diffusivity, 5.0e-7)

    def test_mold_penetration_params_default_and_clamp(self) -> None:
        params = SolidificationParameters()
        self.assertAlmostEqual(params.thermal_boundary.mold_thermal_conductivity, 0.6)
        self.assertAlmostEqual(params.thermal_boundary.mold_thermal_diffusivity, 3.3e-7)

        clamped = SolidificationParameters.from_dict(
            {
                "thermal_boundary": {
                    "mold_thermal_conductivity": 0.0,
                    "mold_thermal_diffusivity": 0.0,
                }
            }
        )
        self.assertAlmostEqual(clamped.thermal_boundary.mold_thermal_conductivity, 0.05)
        self.assertAlmostEqual(clamped.thermal_boundary.mold_thermal_diffusivity, 1e-9)

    def test_filling_defaults_present(self) -> None:
        params = SolidificationParameters()

        # 0 = derive from the square-root rule against the casting's own
        # weight, not a fixed rate that suits one size of casting.
        self.assertAlmostEqual(params.filling.pour_rate_lb_s, 0.0)
        self.assertAlmostEqual(params.filling.pour_time_s, 0.0)
        self.assertAlmostEqual(params.filling.pour_temperature_c, 0.0)
        self.assertAlmostEqual(params.filling.initial_head_in, 4.0)
        self.assertEqual(params.filling.fill_grid_coarsening, 2)
        self.assertAlmostEqual(params.filling.max_fill_time_s, 60.0)
        self.assertAlmostEqual(params.filling.smagorinsky_cs, 0.16)
        self.assertAlmostEqual(params.filling.lattice_velocity_max, 0.05)
        self.assertAlmostEqual(params.filling.wall_htc_multiplier, 1.0)
        self.assertEqual(params.filling.inlet_mode_code, 0)
        self.assertEqual(params.filling.pour_mode_code, 0)
        self.assertEqual(params.filling.vent_risers, 1)

        filt = params.filling.filter
        self.assertAlmostEqual(filt.porosity, 0.85)
        self.assertAlmostEqual(filt.pore_diameter_mm, 2.54)
        self.assertAlmostEqual(filt.ergun_prefactor, 2.2)
        self.assertAlmostEqual(filt.forchheimer_c_f, 0.55)
        self.assertAlmostEqual(filt.contact_angle_deg, 120.0)
        self.assertAlmostEqual(filt.window_fraction, 0.45)
        self.assertAlmostEqual(filt.rated_velocity_warn_factor, 1.5)
        self.assertAlmostEqual(filt.head_fraction_warn, 0.25)
        self.assertEqual(filt.priming_enabled, 1)
        self.assertEqual(filt.drag_enabled, 1)

        serialized = params.to_dict()
        self.assertIn("filling", serialized)
        self.assertIn("filter", serialized["filling"])
        self.assertAlmostEqual(serialized["filling"]["pour_rate_lb_s"], 0.0)
        self.assertAlmostEqual(serialized["filling"]["filter"]["porosity"], 0.85)

    def test_filling_filter_deep_merge_does_not_clobber_siblings(self) -> None:
        params = SolidificationParameters.from_dict(
            {
                "filling": {
                    "pour_rate_lb_s": 8.0,
                    "filter": {"porosity": 0.7},
                }
            }
        )

        # The overridden values took effect.
        self.assertAlmostEqual(params.filling.pour_rate_lb_s, 8.0)
        self.assertAlmostEqual(params.filling.filter.porosity, 0.7)

        # Sibling fields on FillingParams, and sibling fields on the nested
        # FilterParams, are untouched by the partial update.
        self.assertAlmostEqual(params.filling.pour_time_s, 0.0)
        self.assertAlmostEqual(params.filling.initial_head_in, 4.0)
        self.assertEqual(params.filling.vent_risers, 1)
        self.assertAlmostEqual(params.filling.filter.pore_diameter_mm, 2.54)
        self.assertAlmostEqual(params.filling.filter.ergun_prefactor, 2.2)
        self.assertEqual(params.filling.filter.drag_enabled, 1)

        # Untouched sibling top-level groups are also unaffected.
        self.assertAlmostEqual(params.feeding.base_feeding_distance, 3.0)
        self.assertAlmostEqual(params.thermal_boundary.htc_gating, 400.0)

    def test_filling_round_trips_through_from_dict(self) -> None:
        original = SolidificationParameters.from_dict(
            {
                "filling": {
                    "pour_rate_lb_s": 12.5,
                    "fill_grid_coarsening": 4,
                    "inlet_mode_code": 1,
                    "filter": {
                        "porosity": 0.6,
                        "priming_enabled": 0,
                        "rated_velocity_warn_factor": 2.0,
                    },
                }
            }
        )
        round_tripped = SolidificationParameters.from_dict(original.to_dict())

        self.assertAlmostEqual(round_tripped.filling.pour_rate_lb_s, 12.5)
        self.assertEqual(round_tripped.filling.fill_grid_coarsening, 4)
        self.assertEqual(round_tripped.filling.inlet_mode_code, 1)
        self.assertAlmostEqual(round_tripped.filling.filter.porosity, 0.6)
        self.assertEqual(round_tripped.filling.filter.priming_enabled, 0)
        self.assertAlmostEqual(round_tripped.filling.filter.rated_velocity_warn_factor, 2.0)

    def test_filling_filter_clamps_bound_both_ends(self) -> None:
        bounds = {
            "porosity": (0.50, 0.95),
            "pore_diameter_mm": (0.5, 8.0),
            "ergun_prefactor": (0.5, 10.0),
            "forchheimer_c_f": (0.1, 2.0),
            "contact_angle_deg": (90.0, 160.0),
            "window_fraction": (0.2, 0.8),
        }
        for field_name, (low, high) in bounds.items():
            too_low = SolidificationParameters.from_dict(
                {"filling": {"filter": {field_name: low - 1000.0}}}
            )
            too_high = SolidificationParameters.from_dict(
                {"filling": {"filter": {field_name: high + 1000.0}}}
            )
            self.assertAlmostEqual(
                getattr(too_low.filling.filter, field_name), low,
                msg=f"{field_name} low clamp failed",
            )
            self.assertAlmostEqual(
                getattr(too_high.filling.filter, field_name), high,
                msg=f"{field_name} high clamp failed",
            )

    def test_filling_clamps_bound_both_ends(self) -> None:
        low_case = SolidificationParameters.from_dict(
            {
                "filling": {
                    "fill_grid_coarsening": -5,
                    "lattice_velocity_max": -1.0,
                    "max_fill_time_s": -100.0,
                }
            }
        )
        high_case = SolidificationParameters.from_dict(
            {
                "filling": {
                    "fill_grid_coarsening": 500,
                    "lattice_velocity_max": 5.0,
                    "max_fill_time_s": 100000.0,
                }
            }
        )

        self.assertEqual(low_case.filling.fill_grid_coarsening, 1)
        self.assertAlmostEqual(low_case.filling.lattice_velocity_max, 0.005)
        self.assertAlmostEqual(low_case.filling.max_fill_time_s, 1.0)

        self.assertEqual(high_case.filling.fill_grid_coarsening, 8)
        self.assertAlmostEqual(high_case.filling.lattice_velocity_max, 0.2)
        self.assertAlmostEqual(high_case.filling.max_fill_time_s, 600.0)

    def test_all_leaves_are_float_or_int(self) -> None:
        """Every leaf value in to_dict() must be a plain float or int (not
        bool, not str) because both the desktop and web settings UIs only
        render numeric slider/spinbox fields. This guards that constraint
        for all present and future parameter groups.
        """

        def check(node, path: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    check(value, f"{path}.{key}")
                return
            # bool is a subclass of int in Python, so exclude it explicitly:
            # a bool leaf would silently fail to render as a numeric field.
            self.assertIsInstance(
                node, (int, float),
                f"Leaf at {path} is {type(node).__name__}, expected float or int",
            )
            self.assertNotIsInstance(
                node, bool,
                f"Leaf at {path} is a bool; only float/int leaves are supported",
            )

        params = SolidificationParameters()
        check(params.to_dict(), "root")


if __name__ == "__main__":
    unittest.main()
