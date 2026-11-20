from __future__ import annotations

import unittest

from ui.main_window import SolidificationParamsDialog as Dialog

_SOLVER_SOURCES = (
    "core/filling_stage.py",
    "core/filter_database.py",
    "physics/filling.py",
    "physics/fluid_flow.py",
)

def _solver_text() -> str:
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks = []
    for rel in _SOLVER_SOURCES:
        path = os.path.join(root, rel)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                chunks.append(handle.read())
    return "\n".join(chunks)

class InertFieldTests(unittest.TestCase):
    def test_hidden_fields_are_not_offered_as_controls(self) -> None:
        shown = {
            fd[1] for fd in Dialog.FIELD_DEFS if fd[1] not in Dialog.INERT_FIELD_KEYS
        }
        self.assertTrue(Dialog.INERT_FIELD_KEYS.isdisjoint(shown))

    def test_every_hidden_key_is_a_real_field(self) -> None:
        declared = {fd[1] for fd in Dialog.FIELD_DEFS}
        for key in Dialog.INERT_FIELD_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, declared)

    def test_hidden_fields_are_genuinely_unread(self) -> None:
        text = _solver_text()
        self.assertTrue(text, "could not read solver sources")
        for key in Dialog.INERT_FIELD_KEYS:
            with self.subTest(key=key):
                consumed = f'"{key}"' in text or f".{key}" in text
                self.assertFalse(
                    consumed,
                    f"{key} is hidden but appears to be read by a solver -- "
                    f"remove it from INERT_FIELD_KEYS",
                )

    def test_live_filling_fields_are_still_shown(self) -> None:
        shown = {
            fd[1] for fd in Dialog.FIELD_DEFS if fd[1] not in Dialog.INERT_FIELD_KEYS
        }
        for key in (
            "pour_rate_lb_s",
            "smagorinsky_cs",
            "initial_head_in",
            "fill_grid_coarsening",
            "max_fill_time_s",
            "lattice_velocity_max",
            "inlet_mode_code",
            "pour_mode_code",
            "vent_risers",
        ):
            with self.subTest(key=key):
                self.assertIn(key, shown)

class StoredValueSurvivalTests(unittest.TestCase):

    def test_the_dataclass_still_carries_every_hidden_field(self) -> None:
        from core.solidification_params import SolidificationParameters

        params = SolidificationParameters()
        for key in Dialog.INERT_FIELD_KEYS:
            with self.subTest(key=key):
                holder = (
                    params.filling.filter
                    if hasattr(params.filling.filter, key)
                    else params.filling
                )
                self.assertTrue(
                    hasattr(holder, key),
                    f"{key} was removed from the dataclass; saved projects "
                    f"carrying it would no longer round-trip",
                )

    def test_a_saved_value_round_trips_through_serialization(self) -> None:
        from core.solidification_params import SolidificationParameters

        params = SolidificationParameters()
        params.filling.pour_time_s = 12.5
        params.filling.filter.priming_enabled = 0

        restored = SolidificationParameters.from_dict(params.to_dict())

        self.assertAlmostEqual(restored.filling.pour_time_s, 12.5)
        self.assertEqual(restored.filling.filter.priming_enabled, 0)

if __name__ == "__main__":
    unittest.main()
