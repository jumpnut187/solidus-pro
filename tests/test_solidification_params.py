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


if __name__ == "__main__":
    unittest.main()
