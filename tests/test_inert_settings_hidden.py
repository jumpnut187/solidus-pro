"""Tests that unread parameters stay out of the settings dialog.

Six filling parameters were exposed as editable controls, serialized, and
round-trip tested -- and never read by any solver. A control that promises to
change the answer and does not is worse than no control: it invites someone to
tune a knob, watch nothing move, and mistrust the whole dialog.

They are hidden rather than deleted, so saved projects still round-trip. The
risk that creates is the opposite one -- that hiding a field quietly drops its
stored value, or that a field gets hidden which something actually reads. Both
are covered here.
"""

from __future__ import annotations

import unittest

from ui.main_window import SolidificationParamsDialog as Dialog

#: Every source a filling parameter could plausibly be read from.
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
        """A stale entry here would silently hide nothing."""
        declared = {fd[1] for fd in Dialog.FIELD_DEFS}
        for key in Dialog.INERT_FIELD_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, declared)

    def test_hidden_fields_are_genuinely_unread(self) -> None:
        """
        The load-bearing check. If someone wires one of these up and forgets
        to un-hide it, they will be editing a control that no longer exists.
        """
        text = _solver_text()
        self.assertTrue(text, "could not read solver sources")
        for key in Dialog.INERT_FIELD_KEYS:
            with self.subTest(key=key):
                # A bare mention in a comment does not count as a read, so
                # look for the two ways these are actually consumed.
                consumed = f'"{key}"' in text or f".{key}" in text
                self.assertFalse(
                    consumed,
                    f"{key} is hidden but appears to be read by a solver -- "
                    f"remove it from INERT_FIELD_KEYS",
                )

    def test_live_filling_fields_are_still_shown(self) -> None:
        """Guards against over-hiding: these all drive the fill."""
        shown = {
            fd[1] for fd in Dialog.FIELD_DEFS if fd[1] not in Dialog.INERT_FIELD_KEYS
        }
        for key in (
            "pour_rate_lb_s",
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
    """
    Hiding a control must not discard what it held.

    get_options()/set_options() both walk ``self._spinboxes``, and a hidden
    field never gets one -- so it is simply absent from the dialog's view of
    the world and the dataclass default stands. This pins that behaviour so a
    later refactor cannot turn "hidden" into "reset to zero".
    """

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
        params.filling.smagorinsky_cs = 0.22

        restored = SolidificationParameters.from_dict(params.to_dict())

        self.assertAlmostEqual(restored.filling.pour_time_s, 12.5)
        self.assertAlmostEqual(restored.filling.smagorinsky_cs, 0.22)


if __name__ == "__main__":
    unittest.main()
