"""The record of what a run simulated, which the client compares with the parts on screen."""

from __future__ import annotations

import unittest

from web.payloads import SIMULATED_PART_FIELDS, simulated_parts_snapshot


class SimulatedPartsSnapshotTests(unittest.TestCase):
    def test_keeps_exactly_the_fields_a_run_depends_on(self) -> None:
        part = {
            "id": "a1",
            "name": "Riser 1",
            "filepath": "/uploads/x.stl",
            "part_type": "riser",
            "visible": True,
            "is_open_top": 1,
            "position_x": 1,
            "position_y": 2.5,
            "position_z": 0,
            "rotation_x": 0,
            "rotation_y": 90,
            "rotation_z": 0,
            "scale": 1,
            "mesh_data": {"vertices": [0.0] * 9},
            "feature": {"kind": "extrude"},
        }
        [entry] = simulated_parts_snapshot([part])
        self.assertEqual(tuple(entry), SIMULATED_PART_FIELDS)
        self.assertIs(entry["is_open_top"], True)
        self.assertEqual(entry["position_y"], 2.5)
        self.assertEqual(entry["rotation_y"], 90.0)

    def test_hidden_parts_are_listed_and_defaults_filled(self) -> None:
        snapshot = simulated_parts_snapshot([{"id": "h", "filepath": "/f.stl", "visible": False}])
        self.assertEqual(len(snapshot), 1)
        self.assertIs(snapshot[0]["visible"], False)
        self.assertEqual(snapshot[0]["scale"], 1.0)
        self.assertEqual(snapshot[0]["part_type"], "")
        self.assertIs(simulated_parts_snapshot([{"id": "v"}])[0]["visible"], True)


if __name__ == "__main__":
    unittest.main()
