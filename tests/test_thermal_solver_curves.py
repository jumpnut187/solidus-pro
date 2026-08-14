from __future__ import annotations

import unittest

import numpy as np

from core.thermal_solver import ThermalProperties, create_thermal_properties_from_alloy


class _FakeAlloy:
    """Minimal stand-in for AlloyProperties used by create_thermal_properties_from_alloy."""

    def __init__(self) -> None:
        self.solidus_temp = 1399.0
        self.liquidus_temp = 1460.0
        self.thermal_conductivity = 45.0
        self.specific_heat = 600.0
        self.density_solid = 7850.0
        self.density_liquid = 7200.0
        self.latent_heat = 272000.0
        self.name = "WCB"


class ThermalPropertiesCurvesTests(unittest.TestCase):
    def test_vectorized_matches_scalar(self) -> None:
        props = ThermalProperties()
        rng = np.random.default_rng(42)
        temps = rng.uniform(20.0, 1500.0, size=500)

        vectorized = props.get_diffusivity_array(temps)
        scalar = np.array([props.get_diffusivity(float(t)) for t in temps])

        self.assertEqual(vectorized.dtype, np.float64)
        self.assertEqual(vectorized.shape, temps.shape)
        self.assertTrue(
            np.allclose(vectorized, scalar, rtol=1e-9),
            msg="Vectorized diffusivity must match scalar interpolation element-wise",
        )

    def test_to_curves_dict_keys_and_values(self) -> None:
        props = ThermalProperties()
        curves = props.to_curves_dict()

        self.assertEqual(
            set(curves.keys()),
            {"temp_points", "conductivity", "specific_heat", "density"},
        )
        self.assertTrue(np.array_equal(curves["temp_points"], props.temp_points))
        self.assertTrue(np.array_equal(curves["conductivity"], props.conductivity_values))
        self.assertTrue(np.array_equal(curves["specific_heat"], props.specific_heat_values))
        self.assertTrue(np.array_equal(curves["density"], props.density_values))

    def test_create_thermal_properties_from_alloy(self) -> None:
        alloy = _FakeAlloy()
        props = create_thermal_properties_from_alloy(alloy)

        self.assertEqual(len(props.temp_points), 6)
        self.assertEqual(len(props.conductivity_values), 6)
        self.assertEqual(len(props.specific_heat_values), 6)
        self.assertEqual(len(props.density_values), 6)

        # Conductivity must be monotone (decreasing) across the table
        self.assertTrue(
            np.all(np.diff(props.conductivity_values) <= 0),
            msg="Conductivity should not increase with temperature",
        )

        # Density must interpolate to the liquid density exactly at liquidus
        rho_at_liquidus = np.interp(
            alloy.liquidus_temp, props.temp_points, props.density_values
        )
        self.assertEqual(rho_at_liquidus, alloy.density_liquid)

        # Sanity: solidus/liquidus carried through
        self.assertEqual(props.solidus_temp, alloy.solidus_temp)
        self.assertEqual(props.liquidus_temp, alloy.liquidus_temp)
        self.assertEqual(props.latent_heat, alloy.latent_heat)


if __name__ == "__main__":
    unittest.main()
