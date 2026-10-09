"""Wave 3 cross-lane seams on tet runs.

Covers the fixes the integration step made between lanes:
- ``result_data['shrinkage_percentage']`` is the engine's volume share on tets.
- The surrogate pipeline conditions on the tet edge length, as capture does.
- The feeder-neck factor ignores -1 padding in tet adjacency rows.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from core import simulation_pipeline as sp
from core.solidification import SolidificationEngine
from core.tet_mesh import TetVolumeMesh
from tests.test_tet_pipeline import SMOKE_BOX, _settings, _StlLoader, _StubSurrogateModel


class ShrinkagePercentageSummaryTest(unittest.TestCase):
    def _engine(self, mesh_type: str) -> SimpleNamespace:
        return SimpleNamespace(
            mesh=SimpleNamespace(mesh_type=mesh_type),
            affected_voxel_count=3,
            get_shrinkage_percentage=lambda: 12.5,
        )

    def test_tet_uses_engine_volume_share(self) -> None:
        result_data = {"voxel_count": 10}
        sp._add_summary_statistics(result_data, self._engine("tet"))
        self.assertEqual(result_data["shrinkage_percentage"], 12.5)
        self.assertEqual(result_data["shrinkage_voxel_count"], 3)

    def test_voxel_keeps_count_share(self) -> None:
        result_data = {"voxel_count": 10}
        sp._add_summary_statistics(result_data, self._engine("voxel"))
        self.assertAlmostEqual(result_data["shrinkage_percentage"], 30.0)


class FeederNeckPaddingTest(unittest.TestCase):
    """Node 0 is RISER_POST; padded -1 slots must not mark cast nodes as necks."""

    neighbors = np.array(
        [
            [1, -1, -1],  # 0: riser post
            [0, 2, -1],   # 1: cast, real post neighbour
            [1, -1, -1],  # 2: cast, padding only
        ],
        dtype=np.int32,
    )
    part_types = np.array([3, 0, 0], dtype=np.int8)

    def _factors(self, mesh_type: str) -> np.ndarray:
        mesh = SimpleNamespace(
            mesh_type=mesh_type,
            voxel_part_types=self.part_types,
            voxel_neighbors=self.neighbors,
            voxel_centers=np.zeros((3, 3), dtype=np.float32),
        )
        engine = SimpleNamespace(mesh=mesh)
        return SolidificationEngine._estimate_feeder_neck_resistance_factors(engine)

    def test_tet_padding_is_not_a_post_neighbour(self) -> None:
        np.testing.assert_allclose(self._factors("tet"), [1.20, 1.35, 1.0])

    def test_voxel_behaviour_unchanged(self) -> None:
        # Voxel rows keep the pre-existing mapping of -1 to node 0.
        np.testing.assert_allclose(self._factors("voxel"), [1.20, 1.35, 1.35])


class SurrogateTetConditioningTest(unittest.TestCase):
    def test_conditioning_uses_tet_edge_length(self) -> None:
        from core.surrogate import pipeline as surrogate_pipeline
        from core.surrogate.conditioning import build_conditioning_vector

        # voxel_size 0.25 with edge factor 0.8 -> tet edge 0.2 in.
        inputs = sp.PipelineInputs("WCB", 0.25, 0, mesh_type="tet")
        with mock.patch.object(
            sp, "get_settings", return_value=_settings(tet_edge_length_factor=0.8)
        ), mock.patch.object(
            surrogate_pipeline, "build_conditioning_vector", wraps=build_conditioning_vector
        ) as conditioning:
            result = surrogate_pipeline.run_surrogate_pipeline(
                inputs, model=_StubSurrogateModel(), stl_loader=_StlLoader(SMOKE_BOX)
            )
        mesh = result.voxel_mesh
        self.assertIsInstance(mesh, TetVolumeMesh)
        self.assertAlmostEqual(mesh.base_voxel_size, 0.2)
        config = conditioning.call_args.args[1]
        self.assertAlmostEqual(config["voxel_size"], 0.2)
        # The web Overview's "Element size" reads this key.
        self.assertAlmostEqual(result.result_data["tet_edge_length_in"], 0.2)
        # The surrogate runs the shrinkage statistics, so the summary is the
        # engine's volume share.
        self.assertAlmostEqual(
            result.result_data["shrinkage_percentage"],
            result.engine.get_shrinkage_percentage(),
        )


class RiserDialogTetEndToEndTest(unittest.TestCase):
    """Desktop riser dialog on a tet mesh with the real RiserDesigner (offscreen)."""

    def test_dialog_designs_risers_on_a_tet_mesh(self) -> None:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import trimesh
        from PySide6.QtWidgets import QApplication, QWidget

        from core.part_manager import Part, PartManager, PartType
        from ui import main_window as mw

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)
        manager = PartManager()
        cast = trimesh.creation.box(extents=(3.0, 2.0, 1.0))
        cast.apply_translation((0.0, 0.0, 0.5))
        part = Part(id="cast", name="cast", filepath="cast.stl", trimesh_obj=cast)
        part.set_part_type(PartType.CAST)
        manager.parts["cast"] = part
        manager._part_order.append("cast")

        parent = QWidget()
        parent.part_manager = manager
        parent._current_voxel_size = 0.2
        parent._performance_options = {"mesh_type": "tet"}
        parent._current_alloy = "WCB"
        dialog = mw.RiserDesignDialog(parent)
        with mock.patch.object(mw, "QMessageBox") as box:
            dialog._run_analysis()
        box.critical.assert_not_called()
        box.warning.assert_not_called()

        mesh = dialog._voxel_mesh
        self.assertIsInstance(mesh, TetVolumeMesh)
        self.assertAlmostEqual(float(np.sum(mesh.voxel_volumes)), 6.0, delta=0.05)
        self.assertGreater(len(dialog._recommendations), 0)
        self.assertEqual(dialog.results_list.count(), len(dialog._recommendations))
        self.assertIn("Coverage:", dialog.coverage_label.text())
        for riser in dialog._recommendations:
            self.assertTrue(np.all(np.isfinite(riser.position)))
            self.assertGreater(riser.diameter, 0.0)
        dialog.deleteLater()
        parent.deleteLater()


if __name__ == "__main__":
    unittest.main()
