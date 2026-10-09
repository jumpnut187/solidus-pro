"""
Surface-based result rendering for the desktop viewport (Qt-free).

Replaces the one-cube-per-voxel glyph overlays with what the web viewport
draws: the result field painted onto the casting's own surface, a smooth
isosurface ("envelope") of the threshold inside a ghosted shell, and a
painted cap wherever a cross-section plane cuts the metal. Time modes
(fraction liquid, fill time, fill animation) reuse the same three pieces with
a moving front instead of a fixed threshold.

Data flow, once per simulation:

1. :meth:`ResultSurfaceRenderer.set_voxels` turns the engine's sparse voxels
   into a dense uniform lattice (``core.dense_grid``): an occupancy mask,
   the voxel <-> cell maps, and a nearest-metal fill map. A tet run calls
   :meth:`ResultSurfaceRenderer.set_tet_mesh` instead, which samples the
   per-node values onto a ``core.tet_lattice.TetLatticeSampler`` lattice.
2. :meth:`ResultSurfaceRenderer.set_skins` subdivides each metal part's skin
   to roughly the voxel pitch and records, per skin vertex, the lattice cell
   a half voxel inside the surface. Vertex colours then come straight from the
   dense field; no probe filter, no per-frame resampling.

Then per result / threshold / frame, :meth:`ResultSurfaceRenderer.show`
scatters the per-voxel values onto the lattice, repaints the skins and caps,
and re-contours the envelope. Colours are produced by a *palette* callable the
caller supplies (normalised 0..1 values -> uint8 RGB), so this module knows
nothing about colormap names.

Everything here works on a plain ``vtkRenderer``; there is no Qt in this
file, which is what keeps it unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import vtk
from vtk.util import numpy_support as ns

from core.dense_grid import (
    Lattice,
    LatticeTooLargeError,
    derive_lattice,
    flat_indices,
    lattice_from_sampler,
    nearest_known_map,
    sample_node_values,
    scatter_and_fill,
)
from core.part_manager import METAL_PART_CODES, PART_TYPE_TO_CODE

try:
    from core.error_logger import logger
except Exception:  # pragma: no cover
    from core.error_logger_dummy import logger  # type: ignore


Palette = Callable[[np.ndarray], np.ndarray]
"""Normalised values in [0, 1] (any shape) -> uint8 RGB array (..., 3)."""

#: Steel grey the skin shows where it is not painted.
STEEL_RGB = np.array([158, 168, 184], dtype=np.uint8)
#: Emissive tint of a moving liquid front.
MOLTEN_FRONT = (1.0, 0.62, 0.18)

#: Upper bound on skin triangles after subdivision, summed over all parts.
#: A 500k-face STL subdivided to voxel pitch could reach tens of millions of
#: triangles; past this the paint just gets coarser than the voxels.
MAX_SKIN_TRIANGLES = 1_500_000
#: Envelope smoothing is skipped above this many triangles (it is O(n * iters)).
MAX_SMOOTHED_ENVELOPE_TRIANGLES = 600_000


# ---------------------------------------------------------------------------
# Mode palettes (normalised 0..1 -> uint8 RGB). These keep the colour
# conventions the cube overlays used, vectorised.
# ---------------------------------------------------------------------------


def _ramp(t: np.ndarray, stops: Sequence[float], channels: Sequence[Sequence[float]]) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float64).ravel(), 0.0, 1.0)
    out = np.stack([np.interp(t, stops, ch) for ch in channels], axis=1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def palette_hotspot(t: np.ndarray) -> np.ndarray:
    """Yellow (mild) -> red (last to freeze)."""
    return _ramp(t, (0.0, 1.0), ((255, 255), (255, 0), (0, 0)))


def palette_niyama(t: np.ndarray) -> np.ndarray:
    """Blue (safe) -> cyan -> yellow -> red (defect)."""
    return _ramp(t, (0.0, 0.33, 0.66, 1.0), ((0, 0, 255, 255), (0, 255, 255, 0), (255, 255, 0, 0)))


def palette_hot_tear(t: np.ndarray) -> np.ndarray:
    """Cyan -> purple -> magenta."""
    return _ramp(t, (0.0, 0.5, 1.0), ((0, 128, 255), (255, 0, 0), (255, 255, 255)))


def palette_sand_burn_in(t: np.ndarray) -> np.ndarray:
    """Tan -> orange -> deep brown (sand penetration)."""
    return _ramp(t, (0.0, 0.5, 1.0), ((210, 255, 139), (180, 140, 69), (140, 0, 19)))


def palette_liquid(t: np.ndarray) -> np.ndarray:
    """Molten metal: dark orange (mushy) -> bright yellow (fully liquid)."""
    return _ramp(t, (0.0, 1.0), ((200, 255), (100, 255), (0, 30)))


def palette_fill_time(t: np.ndarray) -> np.ndarray:
    """Arrival time: blue (early) -> red (late), small green bump mid-way."""
    tt = np.clip(np.asarray(t, dtype=np.float64).ravel(), 0.0, 1.0)
    r = 20 + 235 * tt
    g = 60 + 40 * (1.0 - np.abs(tt - 0.5) * 2)
    b = 235 - 215 * tt
    return np.clip(np.rint(np.stack([r, g, b], axis=1)), 0, 255).astype(np.uint8)


def palette_fill_speed(t: np.ndarray) -> np.ndarray:
    """Flow speed: blue -> cyan -> yellow -> red."""
    return _ramp(t, (0.0, 0.33, 0.66, 1.0), ((0, 0, 255, 255), (0, 255, 255, 0), (255, 255, 0, 0)))


def make_environment_texture(size: Tuple[int, int] = (512, 256)) -> vtk.vtkTexture:
    """Equirectangular studio environment for image-based lighting.

    Soft sky above, dark floor below, two warm key lights near the horizon so
    brushed metal picks up highlights. Generated, so no HDR asset ships with
    the app. Pass the result to ``renderer.SetEnvironmentTexture(tex, True)``
    (``True``: the texture is sRGB).
    """
    w, h = size
    v = np.linspace(1.0, 0.0, h)[:, None]
    u = np.linspace(0.0, 1.0, w)[None, :]
    sky = np.array([0.55, 0.62, 0.72])
    horizon = np.array([0.32, 0.34, 0.38])
    floor = np.array([0.10, 0.10, 0.11])
    t = np.clip((v - 0.5) * 2, 0, 1)
    img = horizon * (1 - t)[..., None] + sky * t[..., None]
    b = np.clip((0.5 - v) * 2, 0, 1)
    img = img * (1 - b)[..., None] + floor * b[..., None]
    for cu, cv, amp in ((0.25, 0.72, 1.6), (0.7, 0.66, 1.1)):
        img = img + amp * np.exp(-(((u - cu) / 0.06) ** 2 + ((v - cv) / 0.08) ** 2))[..., None]
    rgb = (np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    image = vtk.vtkImageData()
    image.SetDimensions(w, h, 1)
    arr = ns.numpy_to_vtk(np.ascontiguousarray(rgb.reshape(-1, 3)), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    arr.SetName("rgb")
    image.GetPointData().SetScalars(arr)
    tex = vtk.vtkTexture()
    tex.SetInputData(image)
    tex.MipmapOn()
    tex.InterpolateOn()
    tex.SetColorModeToDirectScalars()
    return tex


def _polydata_from_arrays(vertices: np.ndarray, faces: np.ndarray) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(vertices, dtype=np.float64), deep=True))
    faces = np.asarray(faces, dtype=np.int64)
    cells = vtk.vtkCellArray()
    if len(faces):
        packed = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
        cells.SetCells(len(faces), ns.numpy_to_vtkIdTypeArray(packed, deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    return pd


def _polydata_arrays(pd: vtk.vtkPolyData) -> Tuple[np.ndarray, np.ndarray]:
    if pd is None or pd.GetNumberOfPoints() == 0 or pd.GetNumberOfPolys() == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    pts = ns.vtk_to_numpy(pd.GetPoints().GetData())
    conn = ns.vtk_to_numpy(pd.GetPolys().GetData())
    if conn.size and conn.size % 4 == 0 and np.all(conn.reshape(-1, 4)[:, 0] == 3):
        return pts, conn.reshape(-1, 4)[:, 1:].astype(np.int64)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(pd)
    tri.Update()
    out = tri.GetOutput()
    conn = ns.vtk_to_numpy(out.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    return ns.vtk_to_numpy(out.GetPoints().GetData()), conn.astype(np.int64)


def subdivide_skin(vertices: np.ndarray, faces: np.ndarray, max_edge: float,
                   max_triangles: int = MAX_SKIN_TRIANGLES) -> vtk.vtkPolyData:
    """Subdivide a triangle skin until no edge is longer than ``max_edge``.

    ``vtkAdaptiveSubdivisionFilter`` splits a shared edge identically on both
    sides, so the result has no T-junctions (which show as a shading sawtooth
    along box edges). ``max_triangles`` caps the work on very fine STLs; when
    the cap binds, the edge length is raised so the paint is merely coarser.
    Normals are computed with sharp edges split so a box stays crisp while
    curved faces shade smoothly.
    """
    pd = _polydata_from_arrays(vertices, faces)
    n_faces = max(int(pd.GetNumberOfPolys()), 1)
    if n_faces < max_triangles and max_edge > 0.0:
        # Estimate the triangle count at this edge length from surface area.
        mass = vtk.vtkMassProperties()
        mass.SetInputData(pd)
        mass.Update()
        area = float(mass.GetSurfaceArea())
        est = area / max(0.433 * max_edge * max_edge, 1e-12)
        if est > max_triangles:
            max_edge = float(np.sqrt(area / (0.433 * max_triangles)))
        sub = vtk.vtkAdaptiveSubdivisionFilter()
        sub.SetInputData(pd)
        sub.SetMaximumEdgeLength(max_edge)
        sub.SetMaximumNumberOfTriangles(max_triangles)
        sub.SetMaximumNumberOfPasses(12)
        sub.Update()
        pd = sub.GetOutput()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(pd)
    normals.SplittingOn()
    normals.SetFeatureAngle(40.0)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    return normals.GetOutput()


@dataclass
class _Skin:
    part_id: str
    polydata: vtk.vtkPolyData
    actor: vtk.vtkActor
    #: Flat lattice cell sampled for each skin vertex (a half voxel inside).
    cells: np.ndarray
    n_points: int


@dataclass
class ResultView:
    """What is currently shown; kept so re-paints (planes moved, mask changed) can redo it."""

    mode: str
    values: np.ndarray                      # per voxel, field units
    vmin: float
    vmax: float
    iso: float                              # envelope / front value in field units
    palette: Palette
    valid: Optional[np.ndarray] = None      # per voxel; None = all
    paint_mask: Optional[np.ndarray] = None  # per voxel; where the skin/cap shows field colour
    paint_values: Optional[np.ndarray] = None  # per voxel colour source; None = values
    paint_min: Optional[float] = None
    paint_max: Optional[float] = None
    glow: bool = False                      # emissive front (time modes)
    envelope_color: Optional[Tuple[float, float, float]] = None
    flat_paint: Dict[str, Tuple[np.ndarray, Sequence[int]]] = field(default_factory=dict)  # name -> (per-voxel flag, RGB)


class ResultSurfaceRenderer:
    """Painted skin + envelope + section caps for one result at a time."""

    def __init__(self, renderer: vtk.vtkRenderer) -> None:
        self.renderer = renderer
        self.style: str = "envelope"        # surface | envelope | both
        self.visible: bool = True
        self.ghost_opacity: float = 0.35

        # Lattice state (set_voxels / set_tet_mesh). On a tet run the
        # per-"voxel" arrays below are per inside lattice cell, and show()
        # samples its per-node inputs onto those cells first.
        self.lattice: Optional[Lattice] = None
        self._sampler = None                             # TetLatticeSampler on tet runs
        self._input_count: int = 0                       # length show() expects
        self._cell_of_voxel: Optional[np.ndarray] = None
        self._voxel_of_cell: Optional[np.ndarray] = None
        self._occupancy: Optional[np.ndarray] = None    # bool, per cell, before display mask
        self._metal: Optional[np.ndarray] = None        # bool, per cell, metal part types only
        self._nearest: Optional[np.ndarray] = None
        self._display_mask: Optional[np.ndarray] = None  # per voxel bool (hidden components)
        self._image: Optional[vtk.vtkImageData] = None

        # Skins (set_skins)
        self._skins: Dict[str, _Skin] = {}
        self._skin_specs: List[Tuple[str, np.ndarray, np.ndarray, object]] = []
        self._skins_dirty = False
        self._skin_distance: Optional[vtk.vtkImplicitPolyDataDistance] = None
        self._part_types: Dict[str, int] = {}
        self._hidden_parts: set = set()

        # Envelope pipeline
        self._contour = vtk.vtkFlyingEdges3D()
        self._contour.ComputeNormalsOff()
        self._contour.ComputeScalarsOff()
        self._smooth = vtk.vtkWindowedSincPolyDataFilter()
        self._smooth.SetInputConnection(self._contour.GetOutputPort())
        self._smooth.SetNumberOfIterations(12)
        self._smooth.SetPassBand(0.1)
        self._smooth.NonManifoldSmoothingOn()
        self._smooth.NormalizeCoordinatesOn()
        self._env_normals = vtk.vtkPolyDataNormals()
        self._env_normals.SplittingOff()
        self._env_mapper = vtk.vtkPolyDataMapper()
        self._env_mapper.ScalarVisibilityOff()
        self.envelope_actor = vtk.vtkActor()
        self.envelope_actor.SetMapper(self._env_mapper)
        prop = self.envelope_actor.GetProperty()
        prop.SetInterpolationToPBR()
        prop.SetMetallic(0.1)
        prop.SetRoughness(0.35)
        self.envelope_actor.SetVisibility(False)
        self.renderer.AddActor(self.envelope_actor)
        self._envelope_triangles = 0

        # Section caps
        self._planes: List[vtk.vtkPlane] = []
        self.cap_actors: List[vtk.vtkActor] = []

        self.view: Optional[ResultView] = None

    # ------------------------------------------------------------------
    # Setup per simulation
    # ------------------------------------------------------------------

    def set_voxels(self, centers: np.ndarray, sizes: Optional[np.ndarray],
                   part_types: Optional[np.ndarray] = None) -> bool:
        """Build the dense lattice for a run. Returns False when it is too large."""
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        sizes_arr = None if sizes is None else np.asarray(sizes, dtype=np.float64).reshape(-1)
        self._sampler = None
        try:
            lattice = derive_lattice(centers, sizes_arr if sizes_arr is not None and sizes_arr.size == len(centers) else None)
        except LatticeTooLargeError as exc:
            logger.warning("Result lattice too large for surface rendering", context={"error": str(exc)})
            self.lattice = None
            return False
        cells = flat_indices(centers, lattice)
        codes = None
        if part_types is not None and len(part_types) == len(centers):
            codes = np.asarray(part_types, dtype=np.int32)
        self._install_lattice(lattice, cells, codes)
        self._input_count = len(centers)
        return True

    def set_tet_mesh(self, node_xyz: np.ndarray, tets: np.ndarray, tet_part_types: np.ndarray,
                     spacing: float) -> bool:
        """Build the dense lattice for a tet run. Returns False when it is too large.

        A :class:`core.tet_lattice.TetLatticeSampler` at ``spacing`` (inches)
        locates every cell centre in a tet. :meth:`show` then takes per-node
        values and samples them onto the inside cells barycentrically; per-node
        flags (``valid``, ``paint_mask``, ``flat_paint``, the display mask)
        follow each cell's ``nearest_node``. :meth:`voxel_at` returns a node.
        """
        from core.tet_lattice import TetLatticeSampler

        node_xyz = np.asarray(node_xyz, dtype=np.float64).reshape(-1, 3)
        try:
            sampler = TetLatticeSampler(node_xyz, tets, tet_part_types, float(spacing))
        except LatticeTooLargeError as exc:
            logger.warning("Result lattice too large for surface rendering", context={"error": str(exc)})
            self.lattice = None
            self._sampler = None
            return False
        self._sampler = sampler
        self._install_lattice(
            lattice_from_sampler(sampler),
            np.asarray(sampler.cell_flat, dtype=np.int64),
            np.asarray(sampler.cell_part_types, dtype=np.int32),
        )
        self._input_count = len(node_xyz)
        return True

    def _install_lattice(self, lattice: Lattice, cells: np.ndarray, codes: Optional[np.ndarray]) -> None:
        """Lattice, cell maps, occupancy and the VTK image for one run.

        ``cells`` is the flat cell of each lattice "voxel": a voxel of a voxel
        run, or an inside cell of a tet run. ``codes`` are their part codes.
        """
        self.lattice = lattice
        self._cell_of_voxel = cells
        voxel_of_cell = np.full(lattice.cell_count, -1, dtype=np.int64)
        voxel_of_cell[cells] = np.arange(len(cells), dtype=np.int64)
        self._voxel_of_cell = voxel_of_cell
        occupancy = np.zeros(lattice.cell_count, dtype=bool)
        occupancy[cells] = True
        self._occupancy = occupancy
        if codes is not None:
            metal_voxels = np.isin(codes, np.asarray(METAL_PART_CODES, dtype=np.int32))
            metal = np.zeros(lattice.cell_count, dtype=bool)
            metal[cells[metal_voxels]] = True
            self._metal = metal
        else:
            self._metal = occupancy.copy()
        self._nearest = nearest_known_map(occupancy, lattice)
        self._display_mask = None

        nx, ny, nz = lattice.dims
        image = vtk.vtkImageData()
        image.SetDimensions(nx, ny, nz)
        image.SetOrigin(*lattice.origin)
        image.SetSpacing(lattice.spacing, lattice.spacing, lattice.spacing)
        for name in ("field", "paint", "occupancy"):
            arr = ns.numpy_to_vtk(np.zeros(lattice.cell_count, dtype=np.float32), deep=True)
            arr.SetName(name)
            image.GetPointData().AddArray(arr)
        image.GetPointData().SetActiveScalars("field")
        self._image = image
        self._contour.SetInputData(image)
        self._contour.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "field")
        self._write_occupancy()
        self._resample_skins()

    def _node_flags_to_cells(self, flags: np.ndarray) -> np.ndarray:
        """Per-node bools -> per inside cell, through each cell's nearest node."""
        return np.asarray(flags, dtype=bool).reshape(-1)[self._sampler.nearest_node]

    def _node_values_to_cells(self, values: np.ndarray) -> np.ndarray:
        """Per-node values -> per inside cell; NaN where no usable node reaches."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        cell_values, known = sample_node_values(self._sampler, values, np.isfinite(values))
        return np.where(known, cell_values, np.nan).astype(np.float32)

    def set_display_mask(self, mask: Optional[np.ndarray]) -> None:
        """Per-voxel (per-node on tet runs) drawable mask (hidden components); None shows everything."""
        if mask is not None and self._cell_of_voxel is not None and len(mask) != self._input_count:
            mask = None
        if mask is not None and self._sampler is not None:
            mask = self._node_flags_to_cells(mask)
        self._display_mask = None if mask is None else np.asarray(mask, dtype=bool)
        if self._image is not None:
            self._write_occupancy()
        if self.view is not None:
            self.show(**self._view_kwargs())

    def set_skins(self, parts: Sequence[Tuple[str, np.ndarray, np.ndarray, object]]) -> None:
        """Register the model skins: (part_id, vertices, faces, part_type) per part.

        Only metal parts get a painted twin; chills, cores and filters keep
        their own actor. Subdivision is deferred to the first :meth:`show`,
        because this is called on every part edit and a fine STL takes a
        second to subdivide.
        """
        self.clear_skins()
        self._skin_specs = [(pid, np.asarray(v), np.asarray(f), t) for pid, v, f, t in parts]
        self._skins_dirty = bool(self._skin_specs)

    def _build_skins(self) -> None:
        self._skins_dirty = False
        for skin in self._skins.values():
            self.renderer.RemoveActor(skin.actor)
        self._skins.clear()
        self._skin_distance = None
        metal_pds: List[vtk.vtkPolyData] = []
        spacing = self.lattice.spacing if self.lattice else 0.0
        for part_id, vertices, faces, part_type in self._skin_specs:
            code = PART_TYPE_TO_CODE.get(part_type, None)
            if code is None:
                code = int(part_type) if isinstance(part_type, (int, np.integer)) else 0
            self._part_types[part_id] = int(code)
            if int(code) not in METAL_PART_CODES:
                continue
            raw = _polydata_from_arrays(vertices, faces)
            metal_pds.append(raw)
            pd = subdivide_skin(vertices, faces, max_edge=spacing * 0.9 if spacing > 0 else 0.0)
            n_points = int(pd.GetNumberOfPoints())
            colors = ns.numpy_to_vtk(np.tile(STEEL_RGB, (max(n_points, 1), 1)), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            colors.SetName("Colors")
            colors.SetNumberOfComponents(3)
            pd.GetPointData().SetScalars(colors)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pd)
            mapper.SetColorModeToDirectScalars()
            mapper.ScalarVisibilityOn()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetInterpolationToPBR()
            prop.SetMetallic(0.45)
            prop.SetRoughness(0.45)
            actor.SetVisibility(False)
            self.renderer.AddActor(actor)
            self._skins[part_id] = _Skin(part_id=part_id, polydata=pd, actor=actor,
                                        cells=np.zeros(n_points, dtype=np.int64), n_points=n_points)
        if metal_pds:
            append = vtk.vtkAppendPolyData()
            for pd in metal_pds:
                append.AddInputData(pd)
            append.Update()
            self._skin_distance = vtk.vtkImplicitPolyDataDistance()
            self._skin_distance.SetInput(append.GetOutput())
        self._resample_skins()

    def clear_skins(self) -> None:
        for skin in self._skins.values():
            self.renderer.RemoveActor(skin.actor)
        self._skins.clear()
        self._part_types.clear()
        self._skin_distance = None
        self._skin_specs = []
        self._skins_dirty = False

    def set_hidden_parts(self, hidden: set) -> None:
        self._hidden_parts = set(hidden)
        self._apply_visibility()

    def _resample_skins(self) -> None:
        """Per skin vertex, the lattice cell a half voxel inside the surface."""
        if self.lattice is None:
            return
        depth = 0.5 * self.lattice.spacing
        for skin in self._skins.values():
            pd = skin.polydata
            if pd.GetNumberOfPoints() == 0:
                continue
            pts = ns.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
            normals = pd.GetPointData().GetNormals()
            if normals is not None:
                pts = pts - ns.vtk_to_numpy(normals).astype(np.float64) * depth
            skin.cells = self.lattice.world_to_cell(pts)

    # ------------------------------------------------------------------
    # Showing a result
    # ------------------------------------------------------------------

    def _view_kwargs(self) -> dict:
        v = self.view
        return dict(mode=v.mode, values=v.values, vmin=v.vmin, vmax=v.vmax, iso=v.iso, palette=v.palette,
                    valid=v.valid, paint_mask=v.paint_mask, paint_values=v.paint_values,
                    paint_min=v.paint_min, paint_max=v.paint_max, glow=v.glow,
                    envelope_color=v.envelope_color, flat_paint=v.flat_paint)

    def show(self, mode: str, values: np.ndarray, vmin: float, vmax: float, iso: float, palette: Palette,
             valid: Optional[np.ndarray] = None, paint_mask: Optional[np.ndarray] = None,
             paint_values: Optional[np.ndarray] = None, paint_min: Optional[float] = None,
             paint_max: Optional[float] = None, glow: bool = False,
             envelope_color: Optional[Tuple[float, float, float]] = None,
             flat_paint: Optional[Dict[str, Tuple[np.ndarray, Sequence[int]]]] = None) -> bool:
        """Render one result. Returns False when nothing could be drawn.

        ``values`` are per voxel in field units; the envelope is the isosurface
        at ``iso``. Skins and caps are painted with ``palette`` applied to
        ``paint_values`` (default ``values``) normalised over
        ``[paint_min, paint_max]`` (default ``[vmin, vmax]``), where
        ``paint_mask`` is true; elsewhere they show steel. ``valid`` excludes
        voxels from everything (e.g. hot-tear cells outside the eligible zone;
        NaN fill times). ``flat_paint`` maps a per-voxel bool array to a fixed
        RGB (misrun grey, molten liquid) that overrides the palette.
        """
        if self.lattice is None or self._cell_of_voxel is None or self._image is None:
            return False
        if self._skins_dirty:
            self._build_skins()
        n = self._input_count
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(values) != n:
            logger.warning("Result values do not match the voxel count", context={"values": len(values), "voxels": n})
            return False
        self.view = ResultView(mode=mode, values=values, vmin=float(vmin), vmax=float(vmax), iso=float(iso),
                               palette=palette, valid=valid, paint_mask=paint_mask, paint_values=paint_values,
                               paint_min=paint_min, paint_max=paint_max, glow=glow,
                               envelope_color=envelope_color, flat_paint=dict(flat_paint or {}))
        flat_items = dict(self.view.flat_paint)
        if self._sampler is not None:
            # The view keeps the per-node inputs (a re-show resamples them);
            # everything below works on the inside lattice cells.
            values = self._node_values_to_cells(values)
            if valid is not None:
                valid = self._node_flags_to_cells(valid)
            if paint_mask is not None:
                paint_mask = self._node_flags_to_cells(paint_mask)
            if paint_values is not None:
                paint_values = self._node_values_to_cells(paint_values)
            flat_items = {key: (self._node_flags_to_cells(flag[0]), flag[1]) for key, flag in flat_items.items()}

        drawable = np.isfinite(values)
        if valid is not None:
            drawable &= np.asarray(valid, dtype=bool)
        if self._display_mask is not None:
            drawable &= self._display_mask

        lattice = self.lattice
        cells = self._cell_of_voxel
        # Envelope field: real values where drawable, a floor below iso elsewhere so
        # the contour closes at the metal and never leaks into the mold.
        span = max(abs(self.view.vmax - self.view.vmin), abs(iso), 1.0)
        floor = min(float(iso), float(np.min(values[drawable])) if np.any(drawable) else float(iso)) - 1e-3 * span
        field_grid = np.full(lattice.cell_count, floor, dtype=np.float32)
        field_grid[cells[drawable]] = values[drawable]
        self._write_array("field", field_grid)

        # Paint grid: nearest-filled colour source.
        src = values if paint_values is None else np.asarray(paint_values, dtype=np.float32).reshape(-1)
        pmin = self.view.vmin if paint_min is None else float(paint_min)
        pmax = self.view.vmax if paint_max is None else float(paint_max)
        paint_ok = drawable & np.isfinite(src)
        if paint_mask is not None:
            paint_ok &= np.asarray(paint_mask, dtype=bool)
        norm_src = np.clip((src - pmin) / max(pmax - pmin, 1e-12), 0.0, 1.0).astype(np.float32)
        paint_grid = scatter_and_fill(norm_src, cells, paint_ok, self._nearest, lattice) if np.any(paint_ok) else np.zeros(lattice.cell_count, dtype=np.float32)
        # Which cells are painted at all (nearest-filled so skin vertices just outside still read metal).
        painted_cells = np.zeros(lattice.cell_count, dtype=bool)
        painted_cells[cells[paint_ok]] = True
        painted_cells = painted_cells[self._nearest] & self._nearest_is_drawable(drawable)
        self._paint_grid = paint_grid
        self._painted_cells = painted_cells
        self._flat_grids = {}
        for key, flag in flat_items.items():
            flag_arr = np.asarray(flag[0], dtype=bool) & drawable
            grid = np.zeros(lattice.cell_count, dtype=bool)
            grid[cells[flag_arr]] = True
            self._flat_grids[key] = (grid[self._nearest] & self._nearest_is_drawable(drawable), np.asarray(flag[1], dtype=np.uint8))
        self._write_array("paint", paint_grid)

        self._repaint_skins()
        self._rebuild_envelope()
        self._rebuild_caps()
        self._apply_visibility()
        return True

    def _nearest_is_drawable(self, drawable_voxels: np.ndarray) -> np.ndarray:
        """Per cell: is the nearest occupied cell one whose voxel is drawable?"""
        cell_ok = np.zeros(self.lattice.cell_count, dtype=bool)
        cell_ok[self._cell_of_voxel[drawable_voxels]] = True
        return cell_ok[self._nearest]

    def clear(self) -> None:
        """Remove the result: envelope, caps and paint. Skins return to ghost."""
        self.view = None
        self.envelope_actor.SetVisibility(False)
        self._envelope_triangles = 0
        for actor in self.cap_actors:
            self.renderer.RemoveActor(actor)
        self.cap_actors = []
        for skin in self._skins.values():
            skin.actor.SetVisibility(False)

    def set_style(self, style: str) -> None:
        if style not in ("surface", "envelope", "both"):
            raise ValueError(f"unknown result style {style!r}")
        self.style = style
        self._apply_visibility()

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        self._apply_visibility()

    def set_iso(self, iso: float) -> bool:
        """Move the threshold / front without re-scattering the paint."""
        if self.view is None:
            return False
        return self.show(**{**self._view_kwargs(), "iso": float(iso)})

    # ------------------------------------------------------------------
    # Pieces
    # ------------------------------------------------------------------

    def _write_array(self, name: str, flat: np.ndarray) -> None:
        arr = self._image.GetPointData().GetArray(name)
        if arr is None or arr.GetNumberOfTuples() != len(flat):
            arr = ns.numpy_to_vtk(np.ascontiguousarray(flat, dtype=np.float32), deep=True)
            arr.SetName(name)
            self._image.GetPointData().AddArray(arr)
        else:
            ns.vtk_to_numpy(arr)[:] = flat
            arr.Modified()
        self._image.Modified()

    def _write_occupancy(self) -> None:
        occ = self._occupancy.copy()
        if self._display_mask is not None:
            hidden_cells = self._cell_of_voxel[~self._display_mask]
            occ[hidden_cells] = False
        self._write_array("occupancy", occ.astype(np.float32))

    def _colors_for_cells(self, cell_ids: np.ndarray) -> np.ndarray:
        """RGB for arbitrary lattice cells under the current view."""
        rgb = np.tile(STEEL_RGB, (len(cell_ids), 1))
        if self.view is None:
            return rgb
        painted = self._painted_cells[cell_ids]
        if np.any(painted):
            rgb[painted] = self.view.palette(self._paint_grid[cell_ids[painted]])
        for grid, color in self._flat_grids.values():
            hit = grid[cell_ids]
            rgb[hit] = color
        return rgb

    def _repaint_skins(self) -> None:
        for skin in self._skins.values():
            if skin.n_points == 0:
                continue
            rgb = self._colors_for_cells(skin.cells)
            arr = ns.numpy_to_vtk(np.ascontiguousarray(rgb, dtype=np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            arr.SetName("Colors")
            arr.SetNumberOfComponents(3)
            skin.polydata.GetPointData().SetScalars(arr)
            skin.polydata.Modified()

    def _rebuild_envelope(self) -> None:
        v = self.view
        self._contour.SetValue(0, v.iso)
        self._contour.Update()
        raw = self._contour.GetOutput()
        n_tri = int(raw.GetNumberOfPolys())
        self._envelope_triangles = n_tri
        if n_tri == 0:
            self._env_mapper.SetInputData(vtk.vtkPolyData())
            return
        if n_tri <= MAX_SMOOTHED_ENVELOPE_TRIANGLES:
            self._env_normals.SetInputConnection(self._smooth.GetOutputPort())
        else:
            self._env_normals.SetInputConnection(self._contour.GetOutputPort())
        self._env_mapper.SetInputConnection(self._env_normals.GetOutputPort())
        prop = self.envelope_actor.GetProperty()
        if v.glow:
            prop.SetColor(*MOLTEN_FRONT)
            prop.SetEmissiveFactor(0.9, 0.5, 0.2)
            prop.SetOpacity(0.95)
        else:
            if v.envelope_color is not None:
                color = v.envelope_color
            else:
                t = np.clip((v.iso - v.vmin) / max(v.vmax - v.vmin, 1e-12), 0.0, 1.0)
                c = v.palette(np.array([t], dtype=np.float32))[0]
                color = (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
            prop.SetColor(*color)
            prop.SetEmissiveFactor(0.0, 0.0, 0.0)
            prop.SetOpacity(1.0)

    @property
    def envelope_triangle_count(self) -> int:
        return self._envelope_triangles

    def set_section_planes(self, planes: Sequence[vtk.vtkPlane]) -> None:
        """Cutting planes currently enabled; caps are rebuilt for each."""
        self._planes = list(planes)
        if self.view is not None:
            self._rebuild_caps()
            self._apply_visibility()

    def update_caps(self) -> None:
        """Re-cut after a plane moved (the vtkPlane objects are shared)."""
        if self.view is not None:
            self._rebuild_caps()
            self._apply_visibility()

    def _rebuild_caps(self) -> None:
        for actor in self.cap_actors:
            self.renderer.RemoveActor(actor)
        self.cap_actors = []
        if self._image is None or self.view is None:
            return
        for plane in self._planes:
            cutter = vtk.vtkCutter()
            cutter.SetInputData(self._image)
            cutter.SetCutFunction(plane)
            # Keep only the metal: clip by interpolated occupancy...
            occ = vtk.vtkAssignAttribute()
            occ.SetInputConnection(cutter.GetOutputPort())
            occ.Assign("occupancy", vtk.vtkDataSetAttributes.SCALARS, vtk.vtkAssignAttribute.POINT_DATA)
            clip = vtk.vtkClipPolyData()
            clip.SetInputConnection(occ.GetOutputPort())
            clip.SetValue(0.5)
            clip.InsideOutOff()
            clip.GenerateClipScalarsOff()
            last = clip
            # ...then trim to the CAD skin so the outline is the part edge, not a staircase.
            if self._skin_distance is not None:
                trim = vtk.vtkClipPolyData()
                trim.SetInputConnection(clip.GetOutputPort())
                trim.SetClipFunction(self._skin_distance)
                trim.SetValue(0.0)
                trim.InsideOutOn()
                trim.GenerateClipScalarsOff()
                last = trim
            last.Update()
            cap = last.GetOutput()
            if cap is None or cap.GetNumberOfPoints() == 0 or cap.GetNumberOfCells() == 0:
                continue
            pts = ns.vtk_to_numpy(cap.GetPoints().GetData()).astype(np.float64)
            cell_ids = self.lattice.world_to_cell(pts)
            rgb = self._colors_for_cells(cell_ids)
            colors = ns.numpy_to_vtk(np.ascontiguousarray(rgb, dtype=np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            colors.SetName("Colors")
            colors.SetNumberOfComponents(3)
            cap.GetPointData().SetScalars(colors)
            # Nudge a hair toward the kept side so the cap never z-fights the clipped skin.
            normal = np.asarray(plane.GetNormal(), dtype=np.float64)
            offset = ns.vtk_to_numpy(cap.GetPoints().GetData())
            offset += (normal / max(np.linalg.norm(normal), 1e-12)) * (0.02 * self.lattice.spacing)
            cap.GetPoints().Modified()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(cap)
            mapper.SetColorModeToDirectScalars()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetInterpolationToPBR()
            prop.SetMetallic(0.2)
            prop.SetRoughness(0.7)
            self.renderer.AddActor(actor)
            self.cap_actors.append(actor)

    def _apply_visibility(self) -> None:
        showing = self.view is not None and self.visible
        painted = showing and self.style in ("surface", "both")
        envelope = showing and self.style in ("envelope", "both")
        for part_id, skin in self._skins.items():
            skin.actor.SetVisibility(painted and part_id not in self._hidden_parts)
            skin.actor.GetProperty().SetOpacity(1.0 if self.style == "surface" else 0.55)
        self.envelope_actor.SetVisibility(envelope and self._envelope_triangles > 0)
        for actor in self.cap_actors:
            actor.SetVisibility(showing)

    def painted_part_ids(self) -> set:
        """Parts whose own actor should hide while the painted twin is shown."""
        if self.view is None or not self.visible or self.style not in ("surface", "both"):
            return set()
        return set(self._skins.keys())

    # ------------------------------------------------------------------
    # Picking support
    # ------------------------------------------------------------------

    def pick_actors(self) -> List[vtk.vtkActor]:
        actors: List[vtk.vtkActor] = []
        if self.envelope_actor.GetVisibility():
            actors.append(self.envelope_actor)
        actors.extend(skin.actor for skin in self._skins.values() if skin.actor.GetVisibility())
        actors.extend(a for a in self.cap_actors if a.GetVisibility())
        return actors

    def voxel_at(self, position: Sequence[float]) -> int:
        """Index of the voxel nearest a picked world position, or -1.

        On a tet run this is a node: the ``nearest_node`` of the picked cell.
        """
        if self.lattice is None or self._voxel_of_cell is None:
            return -1
        cell = int(self.lattice.world_to_cell(np.asarray(position, dtype=np.float64))[0])
        voxel = int(self._voxel_of_cell[cell])
        if voxel < 0 and self._nearest is not None:
            voxel = int(self._voxel_of_cell[int(self._nearest[cell])])
        if voxel >= 0 and self._sampler is not None:
            voxel = int(self._sampler.nearest_node[voxel])
        return voxel

    def actors_for_clipping(self) -> List[vtk.vtkActor]:
        """Actors the cross-section planes should clip (caps are never clipped)."""
        return [self.envelope_actor] + [skin.actor for skin in self._skins.values()]
