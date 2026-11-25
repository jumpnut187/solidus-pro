from __future__ import annotations

import unittest

import numpy as np
import vtk

from core.dense_grid import Lattice, derive_lattice, flat_indices, nearest_known_map, scatter_and_fill
from core.part_manager import PartType
from ui.result_surface import (
    STEEL_RGB,
    ResultSurfaceRenderer,
    palette_hotspot,
    palette_niyama,
    subdivide_skin,
)

def _block(nx=8, ny=6, nz=5, spacing=0.5):
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    centers = np.column_stack([ix.ravel(), iy.ravel(), iz.ravel()]).astype(np.float64) * spacing
    sizes = np.full(len(centers), spacing)
    c = np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2]) * spacing
    d = np.linalg.norm(centers - c, axis=1)
    field = np.clip(1.0 - d / (d.max() or 1.0), 0.0, 1.0).astype(np.float32)
    return centers, sizes, field

def _box_mesh(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float64)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, f

class DenseGridTests(unittest.TestCase):
    def test_lattice_roundtrip_and_fill(self) -> None:
        centers, sizes, field = _block()
        lattice = derive_lattice(centers, sizes)
        self.assertEqual(lattice.dims, (8, 6, 5))
        cells = flat_indices(centers, lattice)
        self.assertEqual(len(np.unique(cells)), len(centers))
        back = lattice.world_to_cell(centers)
        np.testing.assert_array_equal(back, cells)
        known = np.zeros(lattice.cell_count, dtype=bool)
        known[cells[::2]] = True
        nearest = nearest_known_map(known, lattice)
        self.assertTrue(np.all(known[nearest]))
        dense = scatter_and_fill(field, cells, np.ones(len(cells), dtype=bool), nearest_known_map(np.ones(lattice.cell_count, bool), lattice), lattice)
        self.assertAlmostEqual(float(dense[cells[0]]), float(field[0]), places=6)

class SubdivisionTests(unittest.TestCase):
    def test_subdivision_refines_without_cracks(self) -> None:
        v, f = _box_mesh((0, 0, 0), (2, 1, 1))
        pd = subdivide_skin(v, f, max_edge=0.3)
        self.assertGreater(pd.GetNumberOfPolys(), len(f) * 10)
        feat = vtk.vtkFeatureEdges()
        feat.SetInputData(pd)
        feat.BoundaryEdgesOn(); feat.FeatureEdgesOff(); feat.ManifoldEdgesOff(); feat.NonManifoldEdgesOn()
        feat.Update()
        raw = vtk.vtkAdaptiveSubdivisionFilter()
        raw_in = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        for p in v:
            pts.InsertNextPoint(*p)
        cells = vtk.vtkCellArray()
        for tri in f:
            cells.InsertNextCell(3, [int(t) for t in tri])
        raw_in.SetPoints(pts); raw_in.SetPolys(cells)
        raw.SetInputData(raw_in); raw.SetMaximumEdgeLength(0.3); raw.Update()
        feat2 = vtk.vtkFeatureEdges()
        feat2.SetInputConnection(raw.GetOutputPort())
        feat2.BoundaryEdgesOn(); feat2.FeatureEdgesOff(); feat2.ManifoldEdgesOff(); feat2.NonManifoldEdgesOn()
        feat2.Update()
        self.assertEqual(feat2.GetOutput().GetNumberOfCells(), 0)

    def test_triangle_cap_binds(self) -> None:
        v, f = _box_mesh((0, 0, 0), (4, 4, 4))
        pd = subdivide_skin(v, f, max_edge=0.01, max_triangles=2000)
        self.assertLessEqual(pd.GetNumberOfPolys(), 2000 * 4)

class RendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.centers, self.sizes, self.field = _block()
        self.renderer = vtk.vtkRenderer()
        self.rs = ResultSurfaceRenderer(self.renderer)
        self.assertTrue(self.rs.set_voxels(self.centers, self.sizes, None))
        lo = self.centers.min(axis=0) - 0.25
        hi = self.centers.max(axis=0) + 0.25
        v, f = _box_mesh(lo, hi)
        self.rs.set_skins([("cast", v, f, PartType.CAST), ("chill", v, f, PartType.CHILL)])

    def test_envelope_shrinks_as_threshold_rises(self) -> None:
        self.assertTrue(self.rs.show("hotspot", self.field, 0.0, 1.0, 0.3, palette_hotspot))
        low = self.rs.envelope_triangle_count
        self.assertGreater(low, 0)
        self.rs.set_iso(0.8)
        high = self.rs.envelope_triangle_count
        self.assertGreater(high, 0)
        self.assertLess(high, low)
        self.rs.set_iso(5.0)
        self.assertEqual(self.rs.envelope_triangle_count, 0)
        self.assertFalse(bool(self.rs.envelope_actor.GetVisibility()))

    def test_only_metal_parts_get_a_painted_twin(self) -> None:
        self.rs.show("hotspot", self.field, 0.0, 1.0, 0.3, palette_hotspot)
        self.assertEqual(set(self.rs._skins), {"cast"})

    def test_style_controls_which_actors_show(self) -> None:
        self.rs.show("hotspot", self.field, 0.0, 1.0, 0.3, palette_hotspot)
        skin = self.rs._skins["cast"].actor
        self.assertTrue(self.rs.envelope_actor.GetVisibility()); self.assertFalse(skin.GetVisibility())
        self.assertEqual(self.rs.painted_part_ids(), set())
        self.rs.set_style("surface")
        self.assertFalse(self.rs.envelope_actor.GetVisibility()); self.assertTrue(skin.GetVisibility())
        self.assertEqual(self.rs.painted_part_ids(), {"cast"})
        self.rs.set_style("both")
        self.assertTrue(self.rs.envelope_actor.GetVisibility()); self.assertTrue(skin.GetVisibility())
        self.rs.set_visible(False)
        self.assertFalse(self.rs.envelope_actor.GetVisibility()); self.assertFalse(skin.GetVisibility())

    def test_skin_paint_follows_the_field(self) -> None:
        self.rs.show("niyama", self.field, 0.0, 1.0, 0.3, palette_niyama)
        colours = self.rs._colors_for_cells(self.rs._cell_of_voxel)
        centre = int(np.argmax(self.field)); corner = int(np.argmin(self.field))
        self.assertGreater(colours[centre][0], 200); self.assertLess(colours[centre][2], 50)
        self.assertGreater(colours[corner][2], 200); self.assertLess(colours[corner][0], 50)

    def test_valid_mask_removes_voxels_from_paint_and_envelope(self) -> None:
        valid = self.centers[:, 0] > 1.0
        self.rs.show("hot_tear", self.field, 0.0, 1.0, 0.3, palette_hotspot, valid=valid)
        cells = self.rs._cell_of_voxel
        self.assertFalse(np.any(self.rs._painted_cells[cells[~valid]]))
        self.assertTrue(np.any(self.rs._painted_cells[cells[valid]]))

    def test_display_mask_hides_voxels_and_reshows(self) -> None:
        self.rs.show("hotspot", self.field, 0.0, 1.0, 0.3, palette_hotspot)
        full = self.rs.envelope_triangle_count
        self.rs.set_display_mask(self.centers[:, 2] < 1.0)
        self.assertLess(self.rs.envelope_triangle_count, full)
        self.rs.set_display_mask(None)
        self.assertEqual(self.rs.envelope_triangle_count, full)

    def test_section_cap_is_cut_and_painted(self) -> None:
        self.rs.show("hotspot", self.field, 0.0, 1.0, 0.3, palette_hotspot)
        plane = vtk.vtkPlane(); plane.SetOrigin(1.75, 0, 0); plane.SetNormal(1, 0, 0)
        self.rs.set_section_planes([plane])
        self.assertEqual(len(self.rs.cap_actors), 1)
        cap = self.rs.cap_actors[0].GetMapper().GetInput()
        self.assertGreater(cap.GetNumberOfCells(), 0)
        self.assertIsNotNone(cap.GetPointData().GetScalars())
        self.rs.set_section_planes([])
        self.assertEqual(self.rs.cap_actors, [])

    def test_voxel_at_maps_world_positions_back(self) -> None:
        self.rs.show("hotspot", self.field, 0.0, 1.0, 0.3, palette_hotspot)
        for idx in (0, 17, len(self.centers) - 1):
            self.assertEqual(self.rs.voxel_at(self.centers[idx] + 0.1), idx)
        self.assertGreaterEqual(self.rs.voxel_at((-50.0, -50.0, -50.0)), 0)

    def test_clear_leaves_ghost_only(self) -> None:
        self.rs.show("hotspot", self.field, 0.0, 1.0, 0.3, palette_hotspot)
        self.rs.set_style("surface")
        self.rs.clear()
        self.assertIsNone(self.rs.view)
        self.assertFalse(self.rs.envelope_actor.GetVisibility())
        self.assertFalse(self.rs._skins["cast"].actor.GetVisibility())
        self.assertEqual(self.rs.pick_actors(), [])

    def test_wrong_length_values_are_refused(self) -> None:
        self.assertFalse(self.rs.show("hotspot", self.field[:-1], 0.0, 1.0, 0.3, palette_hotspot))

if __name__ == "__main__":
    unittest.main()
