"""The tet presentation lattice pitch is fixed at run time.

``tet_presentation_spacing`` used to derive the pitch from the *current*
``tet_edge_length_factor``. A result meshed at factor 1.0 and re-rendered
after the default moved to 2.0 got half its run's pitch (8x the cells) and
could trip ``lattice_too_large``. The pipeline now records the pitch on the
mesh, and that recorded value wins.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import trimesh

from config.settings import Settings
from core.dense_grid import tet_presentation_spacing
from core.part_manager import PART_TYPE_CAST


def _settings(edge_factor: float, lattice_factor: float = 1.0) -> Settings:
    settings = Settings()
    settings.mesh.tet_edge_length_factor = edge_factor
    settings.mesh.tet_lattice_spacing_factor = lattice_factor
    return settings


class RecordedSpacingTests(unittest.TestCase):
    def test_recorded_pitch_wins_over_current_settings(self) -> None:
        mesh = types.SimpleNamespace(base_voxel_size=0.2, lattice_spacing_in=0.2)
        with mock.patch("config.settings.get_settings", return_value=_settings(2.0)):
            self.assertAlmostEqual(tet_presentation_spacing(mesh), 0.2)

    def test_meshes_without_a_record_fall_back_to_settings(self) -> None:
        mesh = types.SimpleNamespace(base_voxel_size=0.2)
        with mock.patch("config.settings.get_settings", return_value=_settings(2.0)):
            self.assertAlmostEqual(tet_presentation_spacing(mesh), 0.1)
        with mock.patch("config.settings.get_settings", return_value=_settings(1.0, 1.5)):
            self.assertAlmostEqual(tet_presentation_spacing(mesh), 0.3)

    def test_pipeline_records_the_run_pitch(self) -> None:
        from core import simulation_pipeline as sp

        settings = _settings(2.0, 1.0)
        inputs = types.SimpleNamespace(voxel_size=0.15, thread_budget_provider=None)
        box = trimesh.creation.box((1.2, 1.0, 0.8))
        mesh = sp._build_tet_simulation_mesh(
            inputs,
            settings.mesh,
            [(box, PART_TYPE_CAST)],
            progress_message_callback=None,
            detailed_progress_callback=None,
            cancel_callback=None,
        )
        self.assertAlmostEqual(mesh.lattice_spacing_in, 0.15)
        # A later settings change does not move the pitch of this run.
        with mock.patch("config.settings.get_settings", return_value=_settings(1.0, 2.0)):
            self.assertAlmostEqual(tet_presentation_spacing(mesh), 0.15)


if __name__ == "__main__":
    unittest.main()
