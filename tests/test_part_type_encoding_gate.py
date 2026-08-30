"""The part-type channel encoding is versioned in the model manifest.

``core/surrogate/fields.py`` encodes voxel part-type codes into channel 1 as
``code / PART_TYPE_DIVISOR``, and ``core/deep_model/dataset.py`` recovers chill
voxels by testing that value. Nothing recorded which divisor an artifact was
trained under, so changing the part-type set -- which a mold-filling stage will
do, adding SPRUE/RUNNER/GATE/POUR_CUP -- would have shifted every channel-1 value
and silently degraded every existing model rather than failing.

These tests pin the contract: the divisor is recorded, old manifests without the
field still load, and a mismatch is refused with an actionable message.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.part_manager import PART_TYPE_CHILL
from core.surrogate.fields import PART_TYPE_DIVISOR
from core.surrogate.model import ModelManifest, load_model


def _write_manifest(directory: Path, **overrides) -> None:
    manifest = {
        "name": "gate-test",
        "version": "1",
        "kind": "numpy_demo",
        "backend": "numpy",
    }
    manifest.update(overrides)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class ManifestFieldTests(unittest.TestCase):
    def test_divisor_defaults_to_the_current_encoding(self) -> None:
        manifest = ModelManifest(name="m", version="1", kind="numpy_demo", backend="numpy")
        self.assertEqual(manifest.part_type_divisor, PART_TYPE_DIVISOR)

    def test_divisor_round_trips_through_to_dict(self) -> None:
        manifest = ModelManifest(
            name="m", version="1", kind="numpy_demo", backend="numpy",
            part_type_divisor=7.0,
        )
        self.assertEqual(manifest.to_dict()["part_type_divisor"], 7.0)
        self.assertEqual(ModelManifest.from_dict(manifest.to_dict()).part_type_divisor, 7.0)

    def test_manifest_without_the_field_is_accepted(self) -> None:
        """Artifacts trained before this field existed must keep working."""
        legacy = {"name": "old", "version": "1", "kind": "numpy_demo", "backend": "numpy"}
        self.assertEqual(
            ModelManifest.from_dict(legacy).part_type_divisor, PART_TYPE_DIVISOR
        )


class LoadGateTests(unittest.TestCase):
    def test_legacy_manifest_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_manifest(directory)
            result = load_model(directory)
            self.assertFalse(result.error, msg=f"unexpected load error: {result.error}")
            self.assertIsNotNone(result.model)

    def test_matching_divisor_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_manifest(directory, part_type_divisor=PART_TYPE_DIVISOR)
            result = load_model(directory)
            self.assertFalse(result.error, msg=f"unexpected load error: {result.error}")

    def test_mismatched_divisor_is_refused(self) -> None:
        """A model trained under a different encoding must not load silently."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_manifest(directory, part_type_divisor=PART_TYPE_DIVISOR + 2.0)
            result = load_model(directory)

            self.assertIsNone(result.model)
            self.assertTrue(result.error)
            self.assertIn("part_type_divisor", result.error)
            self.assertIn("Retrain", result.error)


class GatingAliasTests(unittest.TestCase):
    """
    Existing models were trained on codes 0-4 only. The gating codes must not
    reach the encoder raw, or channel 1 carries values the network has never
    seen -- with no divisor mismatch to make the problem loud.
    """

    def test_every_code_encodes_in_distribution(self) -> None:
        import numpy as np

        from core.part_manager import PART_TYPE_TO_CODE
        from core.surrogate.fields import apply_part_type_aliases

        codes = np.array(sorted(PART_TYPE_TO_CODE.values()), dtype=np.int32)
        encoded = apply_part_type_aliases(codes) / PART_TYPE_DIVISOR
        highest_trained = 4 / PART_TYPE_DIVISOR
        for code, value in zip(codes.tolist(), encoded.tolist()):
            with self.subTest(code=code):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(
                    value,
                    highest_trained + 1e-9,
                    f"code {code} encodes to {value:.3f}, above the trained "
                    f"maximum of {highest_trained:.3f}",
                )

    def test_original_codes_are_left_untouched(self) -> None:
        import numpy as np

        from core.surrogate.fields import apply_part_type_aliases

        original = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        np.testing.assert_array_equal(apply_part_type_aliases(original), original)

    def test_aliasing_does_not_mutate_its_input(self) -> None:
        import numpy as np

        from core.part_manager import PART_TYPE_SPRUE
        from core.surrogate.fields import apply_part_type_aliases

        codes = np.array([PART_TYPE_SPRUE], dtype=np.int32)
        apply_part_type_aliases(codes)
        self.assertEqual(codes[0], PART_TYPE_SPRUE)

    def test_filter_is_not_aliased_into_the_chill_window(self) -> None:
        """dataset.py pulls chill cells with a +/-0.05 window around 2/3."""
        import numpy as np

        from core.deep_model.dataset import CHILL_CHANNEL_TOL
        from core.part_manager import PART_TYPE_CHILL, PART_TYPE_FILTER
        from core.surrogate.fields import apply_part_type_aliases

        encoded = float(
            apply_part_type_aliases(np.array([PART_TYPE_FILTER], dtype=np.int32))[0]
            / PART_TYPE_DIVISOR
        )
        chill_value = PART_TYPE_CHILL / PART_TYPE_DIVISOR
        self.assertGreater(abs(encoded - chill_value), CHILL_CHANNEL_TOL)


class ChillDecodingTests(unittest.TestCase):
    def test_chill_encoding_is_recoverable(self) -> None:
        """deep_model recovers chill from channel 1; the two must agree."""
        from core.deep_model.dataset import CHILL_CHANNEL_TOL, CHILL_PART_TYPE

        self.assertEqual(CHILL_PART_TYPE, PART_TYPE_CHILL)
        encoded = PART_TYPE_CHILL / PART_TYPE_DIVISOR
        self.assertLess(abs(encoded - CHILL_PART_TYPE / PART_TYPE_DIVISOR), CHILL_CHANNEL_TOL)

    def test_tolerance_does_not_span_adjacent_codes(self) -> None:
        """The chill window must not also capture RISER_POST or RISER."""
        from core.deep_model.dataset import CHILL_CHANNEL_TOL

        spacing = 1.0 / PART_TYPE_DIVISOR
        self.assertLess(
            CHILL_CHANNEL_TOL,
            spacing / 2.0,
            msg="chill detection tolerance overlaps a neighbouring part-type code",
        )


if __name__ == "__main__":
    unittest.main()
