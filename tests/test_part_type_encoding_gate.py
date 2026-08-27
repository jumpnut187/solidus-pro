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
