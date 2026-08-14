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
            params.feeding.feeding_zone_vertical_bias, 0.5, places=6
        )
        self.assertAlmostEqual(
            params.to_dict()["feeding"]["feeding_zone_vertical_bias"], 0.5, places=6
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
            params.feeding.feeding_zone_vertical_bias, 0.5, places=6
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


if __name__ == "__main__":
    unittest.main()
