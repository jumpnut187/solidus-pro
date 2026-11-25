from __future__ import annotations

import os
import subprocess
import sys
import unittest

from tests.helpers.ci_models import ci_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_STL = ci_model("l_bracket.stl")

def run_smoke() -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    out = sys.__stdout__

    from PySide6.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication([])

    import numpy as np
    import vtk

    import ui.main_window as main_window_module
    import ui.visualizer as visualizer_module
    from core import orientation
    from ui.orientation_tool import part_world_bounds

    class OffscreenVTKWidget(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._render_window = vtk.vtkRenderWindow()
            self._render_window.SetOffScreenRendering(1)
            self._render_window.SetSize(600, 600)
            self._interactor = vtk.vtkRenderWindowInteractor()
            self._interactor.SetRenderWindow(self._render_window)

        def GetRenderWindow(self):
            return self._render_window

    visualizer_module.QVTKRenderWindowInteractor = OffscreenVTKWidget
    window = main_window_module.MainWindow()

    main_window_module.QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([SAMPLE_STL], ""))
    window._add_part()
    parts = window.part_manager.get_all_parts()
    assert len(parts) == 1 and parts[0].part_type.value == "cast", parts
    part = parts[0]
    dialog = window.orientation_dialog
    assert dialog.isVisible(), "tool should auto-open after importing a lone casting"
    assert dialog.windowTitle() == "Orient & Parting Line"

    dialog.rotate_buttons["Rotate X +"].click()
    assert np.allclose(part.rotation, (90.0, 0.0, 0.0)), part.rotation
    assert abs(part_world_bounds(part)[0, 2]) < 1e-9
    actor_bounds = window.visualizer.part_actors[part.id].GetBounds()
    world = part_world_bounds(part)
    assert np.allclose(actor_bounds[0::2], world[0], atol=1e-6), (actor_bounds, world)
    assert np.allclose(actor_bounds[1::2], world[1], atol=1e-6), (actor_bounds, world)

    dialog.preset_buttons["widest"].click()
    expected = orientation.suggest_parting_z(part.trimesh_obj, part.rotation, part.position, part.scale)
    parting = window.orientation_controller.parting
    assert parting is not None and abs(parting.z - expected["max_section"]) < 1e-9, parting
    assert window.visualizer.is_parting_plane_shown()

    z_before = parting.z
    dialog.lower_part_btn.click()
    assert abs(parting.z - (z_before - 0.25)) < 1e-9, parting.z
    dialog.drop_to_floor_btn.click()
    assert abs(parting.z - z_before) < 1e-9, parting.z

    dialog.parting_z_spin.setValue(0.5)
    window.visualizer.set_camera_view("top")
    window.visualizer.render_window.Render()
    dialog.pick_height_btn.click()
    assert window.visualizer.is_surface_pick_mode()
    mesh = part.get_transformed_mesh()
    top = float(mesh.bounds[1, 2])
    up_faces = np.where((mesh.face_normals[:, 2] > 0.99) & (mesh.triangles_center[:, 2] > top - 1e-6))[0]
    target = mesh.triangles_center[up_faces[0]]
    renderer = window.visualizer.renderer
    renderer.SetWorldPoint(target[0], target[1], target[2], 1.0)
    renderer.WorldToDisplay()
    x, y, _ = renderer.GetDisplayPoint()
    window.visualizer._handle_surface_pick(int(round(x)), int(round(y)))
    assert not window.visualizer.is_surface_pick_mode()
    assert dialog.pick_mode is None
    assert abs(parting.z - top) < 1e-3, (parting.z, top)

    model_normal = orientation.euler_to_matrix(part.rotation).T @ np.array([0.0, 0.0, 1.0])
    dialog.face_down_btn.click()
    assert window.visualizer.is_surface_pick_mode()
    window.visualizer._handle_surface_pick(int(round(x)), int(round(y)))
    assert dialog.pick_mode is None
    turned = orientation.euler_to_matrix(part.rotation) @ model_normal
    assert np.allclose(turned, (0.0, 0.0, -1.0), atol=1e-6), (part.rotation, turned)
    assert abs(part_world_bounds(part)[0, 2]) < 1e-9

    state = window._collect_project_state()
    assert state["ui_state"]["parting_line"]["z"] == parting.z, state["ui_state"]

    dialog.show_parting_check.setChecked(False)
    assert not window.visualizer.is_parting_plane_shown()

    dialog.close()
    window.close()
    app.processEvents()
    print("SMOKE OK", file=out, flush=True)
    out.flush()
    os._exit(0)

class DesktopOrientationSmokeTest(unittest.TestCase):
    def test_main_window_orient_and_parting_line(self) -> None:
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        result = subprocess.run(
            [sys.executable, "-c", "import tests.test_desktop_orientation_smoke as m; m.run_smoke()"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        self.assertIn("SMOKE OK", result.stdout)

if __name__ == "__main__":
    unittest.main()
