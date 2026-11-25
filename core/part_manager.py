from __future__ import annotations

import numpy as np
import trimesh
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import re
import uuid

try:
    from core.error_logger import logger
except ImportError:
    from core.error_logger_dummy import logger

try:
    import cascadio
    STEP_SUPPORTED = True
except ImportError:
    STEP_SUPPORTED = False

_STEP_EXTENSIONS = (".step", ".stp")

CAD_FILE_EXTENSIONS = (".stl", ".obj", ".ply", ".glb", ".gltf", ".3mf", ".off") + _STEP_EXTENSIONS

def strip_cad_extension(name: str) -> str:
    root, ext = os.path.splitext(name)
    return root if ext.lower() in CAD_FILE_EXTENSIONS else name

_METERS_TO_INCHES = 39.3701

_MM_PER_INCH = 25.4

TESSELLATION_VOXELS_PER_DEFLECTION = 50.0

MIN_TESSELLATION_TOLERANCE_MM = 0.01

MAX_TESSELLATION_TOLERANCE_MM = 0.5

def tessellation_tolerance_mm(voxel_size_in: Optional[float]) -> Optional[float]:
    if voxel_size_in is None:
        return None
    try:
        voxel_size_in = float(voxel_size_in)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(voxel_size_in) or voxel_size_in <= 0.0:
        return None

    tolerance = (voxel_size_in * _MM_PER_INCH) / TESSELLATION_VOXELS_PER_DEFLECTION
    return float(
        min(
            MAX_TESSELLATION_TOLERANCE_MM,
            max(MIN_TESSELLATION_TOLERANCE_MM, tolerance),
        )
    )

_INCHES_PER_METRE = 1000.0 / _MM_PER_INCH

_STEP_SI_PREFIX_METRES = {
    "$": 1.0,
    "KILO": 1e3,
    "HECTO": 1e2,
    "DECA": 1e1,
    "DECI": 1e-1,
    "CENTI": 1e-2,
    "MILLI": 1e-3,
    "MICRO": 1e-6,
    "NANO": 1e-9,
}

_STEP_NAMED_LENGTH_UNIT_INCHES = {
    "INCH": 1.0,
    "FOOT": 12.0,
    "YARD": 36.0,
    "MIL": 1e-3,
    "THOU": 1e-3,
    "MILLIMETRE": 1.0 / _MM_PER_INCH,
    "MILLIMETER": 1.0 / _MM_PER_INCH,
    "CENTIMETRE": 10.0 / _MM_PER_INCH,
    "CENTIMETER": 10.0 / _MM_PER_INCH,
    "METRE": _INCHES_PER_METRE,
    "METER": _INCHES_PER_METRE,
}

_STEP_ENTITY_RE = re.compile(r"#(\d+)\s*=\s*((?:[^;']|'(?:[^']|'')*')*);")
_STEP_UNIT_CONTEXT_RE = re.compile(r"GLOBAL_UNIT_ASSIGNED_CONTEXT\s*\(\s*\(([^)]*)\)")
_STEP_SI_LENGTH_RE = re.compile(r"SI_UNIT\s*\(\s*(\$|\.\w+\.)\s*,\s*\.METRE\.\s*\)")
_STEP_CONVERSION_UNIT_RE = re.compile(
    r"CONVERSION_BASED_UNIT\s*\(\s*'([^']*)'\s*,\s*#(\d+)\s*\)"
)
_STEP_LENGTH_MEASURE_RE = re.compile(
    r"LENGTH_MEASURE_WITH_UNIT\s*\(\s*LENGTH_MEASURE\s*\(\s*([-+0-9.EeDd]+)\s*\)\s*,\s*#(\d+)\s*\)"
)
_STEP_VERTEX_POINT_RE = re.compile(r"VERTEX_POINT\s*\(\s*'(?:[^']|'')*'\s*,\s*#(\d+)\s*\)")
_STEP_CARTESIAN_POINT_RE = re.compile(
    r"CARTESIAN_POINT\s*\(\s*'(?:[^']|'')*'\s*,\s*\(([^)]*)\)\s*\)"
)

STEP_SCALE_SHRINK_TOLERANCE = 0.05

STEP_SCALE_MAX_GROWTH = 10.0

def _parse_step_entities(text: str) -> Dict[int, str]:
    return {int(m.group(1)): m.group(2) for m in _STEP_ENTITY_RE.finditer(text)}

def _step_length_unit_to_inches(
    entities: Dict[int, str], unit_id: int, _depth: int = 0
) -> Optional[float]:
    body = entities.get(unit_id)
    if body is None or _depth > 4:
        return None

    conversion = _STEP_CONVERSION_UNIT_RE.search(body)
    if conversion is not None:
        name = conversion.group(1).strip().upper()
        if name in _STEP_NAMED_LENGTH_UNIT_INCHES:
            return _STEP_NAMED_LENGTH_UNIT_INCHES[name]
        measure = entities.get(int(conversion.group(2)), "")
        measure_match = _STEP_LENGTH_MEASURE_RE.search(measure)
        if measure_match is None:
            return None
        base = _step_length_unit_to_inches(
            entities, int(measure_match.group(2)), _depth + 1
        )
        if base is None:
            return None
        return float(measure_match.group(1).replace("D", "E").replace("d", "e")) * base

    si_unit = _STEP_SI_LENGTH_RE.search(body)
    if si_unit is not None:
        prefix = si_unit.group(1).strip(".").upper()
        metres = _STEP_SI_PREFIX_METRES.get(prefix)
        return None if metres is None else metres * _INCHES_PER_METRE
    return None

def step_declared_length_unit_inches(text: str) -> Optional[float]:
    entities = _parse_step_entities(text)
    return _declared_length_unit_inches(entities)

def _declared_length_unit_inches(entities: Dict[int, str]) -> Optional[float]:
    factors = set()
    for body in entities.values():
        context = _STEP_UNIT_CONTEXT_RE.search(body)
        if context is None:
            continue
        for ref in re.findall(r"#(\d+)", context.group(1)):
            unit_body = entities.get(int(ref), "")
            if "LENGTH_UNIT" not in unit_body:
                continue
            factor = _step_length_unit_to_inches(entities, int(ref))
            factors.add(None if factor is None else round(factor, 9))
    if len(factors) != 1:
        return None
    return factors.pop()

def _step_vertex_points(entities: Dict[int, str]) -> np.ndarray:
    points = []
    for body in entities.values():
        vertex = _STEP_VERTEX_POINT_RE.match(body)
        if vertex is None:
            continue
        point = _STEP_CARTESIAN_POINT_RE.match(entities.get(int(vertex.group(1)), ""))
        if point is None:
            continue
        try:
            coords = [float(v) for v in point.group(1).split(",")]
        except ValueError:
            continue
        if len(coords) == 3:
            points.append(coords)
    return np.asarray(points, dtype=float).reshape(-1, 3)

STEP_SCALE_CHECK_MAX_BYTES = 32 * 1024 * 1024

def _check_step_scale(geometry, filepath: str) -> None:
    try:
        size = os.path.getsize(filepath)
    except OSError as exc:
        logger.warning(f"Cannot read '{filepath}' to check its STEP length unit: {exc}")
        return
    if size > STEP_SCALE_CHECK_MAX_BYTES:
        logger.info(
            f"STEP unit check skipped for '{filepath}': {size / 1e6:.0f} MB exceeds "
            f"the {STEP_SCALE_CHECK_MAX_BYTES / 1e6:.0f} MB check limit."
        )
        return
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        logger.warning(f"Cannot read '{filepath}' to check its STEP length unit: {exc}")
        return

    entities = _parse_step_entities(text)
    unit_in = _declared_length_unit_inches(entities)
    if unit_in is None:
        logger.warning(
            f"Could not read the declared length unit of '{filepath}'; "
            f"its size is unchecked and relies on cascadio's unit handling."
        )
        return

    vertices = _step_vertex_points(entities)
    if len(vertices) < 2:
        return
    declared_diag = float(np.linalg.norm(np.ptp(vertices, axis=0))) * unit_in
    if declared_diag <= 0.0:
        return

    if isinstance(geometry, trimesh.Scene):
        meshes = list(geometry.geometry.values())
    else:
        meshes = [geometry]
    loaded = [m.vertices for m in meshes if m is not None and len(m.faces) > 0]
    if not loaded:
        return
    loaded_diag = float(np.linalg.norm(np.ptp(np.vstack(loaded), axis=0)))

    ratio = loaded_diag / declared_diag
    if ratio < 1.0 - STEP_SCALE_SHRINK_TOLERANCE or ratio > STEP_SCALE_MAX_GROWTH:
        logger.warning(
            f"STEP unit check failed for '{filepath}': loaded size "
            f"{loaded_diag:.4g} in vs {declared_diag:.4g} in spanned by its B-rep "
            f"vertices in the declared unit ({unit_in:.6g} in/unit), ratio "
            f"{ratio:.3g}. The part is probably mis-scaled."
        )

def _scale_step_to_inches(geometry, filepath: Optional[str] = None) -> None:
    if isinstance(geometry, trimesh.Scene):
        bodies = list(geometry.geometry.values())
        for body in bodies:
            if body is None or len(body.faces) == 0:
                continue
            body.apply_scale(_METERS_TO_INCHES)
    elif geometry is not None and len(geometry.faces) > 0:
        geometry.apply_scale(_METERS_TO_INCHES)
    else:
        return

    if filepath is not None:
        try:
            _check_step_scale(geometry, filepath)
        except Exception as exc:
            logger.warning(f"STEP unit check skipped for '{filepath}': {exc}")

def _load_cad_scene(filepath: str, tol_linear: Optional[float]) -> trimesh.Scene:
    if tol_linear is None:
        scene = trimesh.load(filepath, force="scene")
    else:
        scene = trimesh.load(filepath, force="scene", tol_linear=tol_linear)
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(geometry={"Body_1": scene})
    if os.path.splitext(filepath)[1].lower() in _STEP_EXTENSIONS:
        _scale_step_to_inches(scene, filepath)
    return scene

def _count_nonempty_bodies(scene: trimesh.Scene) -> int:
    return sum(
        1 for mesh in scene.geometry.values() if mesh is not None and len(mesh.faces) > 0
    )

def _default_body_identity(
    filepath: str, body_name: str, single_body: bool
) -> Tuple[str, PartType]:
    if single_body:
        return os.path.splitext(os.path.basename(filepath))[0], infer_part_type_from_name(filepath)
    return body_name, infer_part_type_from_name(body_name)

class PartType(Enum):
    CAST = "cast"
    RISER = "riser"
    RISER_POST = "riser_post"
    CHILL = "chill"
    CORE = "core"
    POUR_CUP = "pour_cup"
    SPRUE = "sprue"
    RUNNER = "runner"
    FILTER = "filter"
    GATE = "gate"
    MOLD = "mold"

    def __str__(self):
        return self.value.replace('_', ' ').title()

PART_TYPE_CAST = 0
PART_TYPE_RISER = 1
PART_TYPE_CHILL = 2
PART_TYPE_RISER_POST = 3
PART_TYPE_CORE = 4
PART_TYPE_POUR_CUP = 5
PART_TYPE_SPRUE = 6
PART_TYPE_RUNNER = 7
PART_TYPE_FILTER = 8
PART_TYPE_GATE = 9
PART_TYPE_MOLD = 10

PART_TYPE_TO_CODE = {
    PartType.CAST: PART_TYPE_CAST,
    PartType.RISER: PART_TYPE_RISER,
    PartType.CHILL: PART_TYPE_CHILL,
    PartType.RISER_POST: PART_TYPE_RISER_POST,
    PartType.CORE: PART_TYPE_CORE,
    PartType.POUR_CUP: PART_TYPE_POUR_CUP,
    PartType.SPRUE: PART_TYPE_SPRUE,
    PartType.RUNNER: PART_TYPE_RUNNER,
    PartType.FILTER: PART_TYPE_FILTER,
    PartType.GATE: PART_TYPE_GATE,
    PartType.MOLD: PART_TYPE_MOLD,
}

CODE_TO_PART_TYPE = {code: part_type for part_type, code in PART_TYPE_TO_CODE.items()}

METAL_PART_CODES = (
    PART_TYPE_CAST,
    PART_TYPE_RISER,
    PART_TYPE_RISER_POST,
    PART_TYPE_POUR_CUP,
    PART_TYPE_SPRUE,
    PART_TYPE_RUNNER,
    PART_TYPE_GATE,
)

FEEDER_PART_CODES = (PART_TYPE_RISER, PART_TYPE_RISER_POST)

MOLD_PART_CODES = (PART_TYPE_CHILL, PART_TYPE_CORE, PART_TYPE_FILTER, PART_TYPE_MOLD)

SAND_PART_CODES = (PART_TYPE_CORE, PART_TYPE_MOLD)

SCORED_METAL_PART_CODES = (PART_TYPE_CAST, PART_TYPE_RISER, PART_TYPE_RISER_POST)

FLOW_PATH_PART_CODES = (
    PART_TYPE_POUR_CUP,
    PART_TYPE_SPRUE,
    PART_TYPE_RUNNER,
    PART_TYPE_FILTER,
    PART_TYPE_GATE,
)

GATING_METAL_PART_CODES = (
    PART_TYPE_POUR_CUP,
    PART_TYPE_SPRUE,
    PART_TYPE_RUNNER,
    PART_TYPE_GATE,
)

FLOW_RESTRICTOR_PART_CODES = (PART_TYPE_FILTER,)

def _code_mask(part_types: "np.ndarray", codes) -> "np.ndarray":
    arr = np.asarray(part_types)
    mask = np.zeros(arr.shape, dtype=bool)
    for code in codes:
        mask |= arr == code
    return mask

def is_metal(part_types: "np.ndarray") -> "np.ndarray":
    return _code_mask(part_types, METAL_PART_CODES)

def is_feeder(part_types: "np.ndarray") -> "np.ndarray":
    return _code_mask(part_types, FEEDER_PART_CODES)

def is_mold(part_types: "np.ndarray") -> "np.ndarray":
    return _code_mask(part_types, MOLD_PART_CODES)

def is_cast(part_types: "np.ndarray") -> "np.ndarray":
    return _code_mask(part_types, (PART_TYPE_CAST,))

def is_chill(part_types: "np.ndarray") -> "np.ndarray":
    return _code_mask(part_types, (PART_TYPE_CHILL,))

def is_scored_metal(part_types: "np.ndarray") -> "np.ndarray":
    return _code_mask(part_types, SCORED_METAL_PART_CODES)

def is_flow_path(part_types: "np.ndarray") -> "np.ndarray":
    return _code_mask(part_types, FLOW_PATH_PART_CODES)

def is_gating_metal(part_types: "np.ndarray") -> "np.ndarray":
    return _code_mask(part_types, GATING_METAL_PART_CODES)

def is_flow_restrictor(part_types: "np.ndarray") -> "np.ndarray":
    return _code_mask(part_types, FLOW_RESTRICTOR_PART_CODES)

def is_filter(part_types: "np.ndarray") -> "np.ndarray":
    return _code_mask(part_types, (PART_TYPE_FILTER,))

PART_TYPE_COLORS = {
    PartType.CAST: (0.7, 0.75, 0.8),
    PartType.RISER: (0.9, 0.6, 0.2),
    PartType.RISER_POST: (0.8, 0.5, 0.3),
    PartType.CHILL: (0.3, 0.3, 0.5),
    PartType.CORE: (0.30, 0.78, 0.40),
    PartType.MOLD: (0.62, 0.40, 0.86),
    PartType.POUR_CUP: (0.45, 0.80, 0.75),
    PartType.SPRUE: (0.30, 0.68, 0.70),
    PartType.RUNNER: (0.20, 0.55, 0.65),
    PartType.FILTER: (0.85, 0.85, 0.35),
    PartType.GATE: (0.15, 0.45, 0.60),
}

PART_TYPE_DISPLAY_NAMES = {part_type: str(part_type) for part_type in PartType}

_PART_TYPE_BY_DISPLAY_NAME = {
    display: part_type for part_type, display in PART_TYPE_DISPLAY_NAMES.items()
}

def part_type_from_display_name(
    text: str, default: PartType = PartType.CAST
) -> PartType:
    return _PART_TYPE_BY_DISPLAY_NAME.get(str(text).strip(), default)

RISER_SLEEVE_INSULATING = "insulating"
RISER_SLEEVE_ADIABATIC = "adiabatic"
RISER_SLEEVE_NONE = "none"
RISER_SLEEVE_CHOICES = (RISER_SLEEVE_INSULATING, RISER_SLEEVE_ADIABATIC, RISER_SLEEVE_NONE)

def riser_sleeve_from_value(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in RISER_SLEEVE_CHOICES else RISER_SLEEVE_INSULATING

def part_type_from_value(value: Any) -> PartType:
    try:
        return PartType(str(value))
    except ValueError:
        logger.warning(f"Unknown part type {value!r}; treating it as cast.")
        return PartType.CAST

PART_TYPE_ABBREVIATIONS = {
    PartType.CAST: "C",
    PartType.RISER: "R",
    PartType.RISER_POST: "RP",
    PartType.CHILL: "Ch",
    PartType.CORE: "Co",
    PartType.POUR_CUP: "PC",
    PartType.SPRUE: "S",
    PartType.RUNNER: "Ru",
    PartType.FILTER: "F",
    PartType.GATE: "G",
    PartType.MOLD: "M",
}

PART_TYPE_BADGES = {
    PartType.CAST: "🔩",
    PartType.RISER: "🔥",
    PartType.RISER_POST: "📍",
    PartType.CHILL: "❄️",
    PartType.CORE: "🕳️",
    PartType.POUR_CUP: "🥣",
    PartType.SPRUE: "⬇️",
    PartType.RUNNER: "➡️",
    PartType.FILTER: "🧱",
    PartType.GATE: "🚪",
    PartType.MOLD: "📦",
}

PART_TYPE_KEYWORDS = {
    PartType.CAST: ['casting', 'part'],
    PartType.RISER_POST: ['riser_post', 'riser post', 'post', 'neck'],
    PartType.RISER: ['feeder', 'riser', 'bridge', 'pad'],
    PartType.CHILL: ['chill'],
    PartType.CORE: ['core', 'sand_core', 'sand core'],
    PartType.FILTER: ['filter', 'stelex', 'sedex'],
    PartType.POUR_CUP: ['pour_cup', 'pour cup', 'pouring cup', 'pouring basin', 'basin'],
    PartType.SPRUE: ['downsprue', 'down_sprue', 'down sprue', 'sprue'],
    PartType.RUNNER: ['runner'],
    PartType.GATE: ['ingate', 'in_gate', 'in gate', 'gate'],
    PartType.MOLD: ['mold', 'mould', 'flask', 'cope', 'drag'],
}

def infer_part_type_from_name(name: str) -> PartType:
    base_name = strip_cad_extension(os.path.basename(name)).lower()

    for keyword in PART_TYPE_KEYWORDS[PartType.CORE]:
        if keyword in base_name:
            return PartType.CORE

    for keyword in PART_TYPE_KEYWORDS[PartType.CHILL]:
        if keyword in base_name:
            return PartType.CHILL

    for gating_type in (
        PartType.FILTER,
        PartType.POUR_CUP,
        PartType.SPRUE,
        PartType.RUNNER,
        PartType.GATE,
    ):
        for keyword in PART_TYPE_KEYWORDS[gating_type]:
            if keyword in base_name:
                return gating_type

    for keyword in PART_TYPE_KEYWORDS[PartType.RISER_POST]:
        if keyword in base_name:
            return PartType.RISER_POST

    for keyword in PART_TYPE_KEYWORDS[PartType.RISER]:
        if keyword in base_name:
            return PartType.RISER

    for keyword in PART_TYPE_KEYWORDS[PartType.CAST]:
        if keyword in base_name:
            return PartType.CAST

    for keyword in PART_TYPE_KEYWORDS[PartType.MOLD]:
        if keyword in base_name:
            return PartType.MOLD

    return PartType.CAST

def infer_part_type_from_filename(filename: str) -> PartType:
    return infer_part_type_from_name(filename)

@dataclass
class Part:
    id: str
    name: str
    filepath: str
    trimesh_obj: trimesh.Trimesh

    part_type: PartType = PartType.CAST

    body_name: Optional[str] = None
    body_index: Optional[int] = None
    tessellation_tolerance_mm: Optional[float] = None

    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    scale: float = 1.0

    _transformed_mesh: Optional[trimesh.Trimesh] = field(default=None, repr=False)
    _transform_dirty: bool = field(default=True, repr=False)

    color: Tuple[float, float, float] = (0.7, 0.75, 0.8)
    visible: bool = True

    is_open_top: bool = False

    riser_sleeve: str = RISER_SLEEVE_INSULATING

    def set_part_type(self, part_type: PartType):
        self.part_type = part_type
        self.color = PART_TYPE_COLORS.get(part_type, (0.7, 0.75, 0.8))

    def get_transformed_mesh(self) -> trimesh.Trimesh:
        if self._transform_dirty or self._transformed_mesh is None:
            self._apply_transform()
        return self._transformed_mesh

    def _apply_transform(self):
        self._transformed_mesh = self.trimesh_obj.copy()

        if self.scale != 1.0:
            self._transformed_mesh.apply_scale(self.scale)

        if np.any(self.rotation != 0):
            rot_matrix = trimesh.transformations.euler_matrix(
                np.radians(self.rotation[0]),
                np.radians(self.rotation[1]),
                np.radians(self.rotation[2]),
                'sxyz'
            )
            self._transformed_mesh.apply_transform(rot_matrix)

        if np.any(self.position != 0):
            translation = trimesh.transformations.translation_matrix(self.position)
            self._transformed_mesh.apply_transform(translation)

        self._transform_dirty = False

    def set_position(self, x: float, y: float, z: float):
        self.position = np.array([x, y, z])
        self._transform_dirty = True

    def set_rotation(self, rx: float, ry: float, rz: float):
        self.rotation = np.array([rx, ry, rz])
        self._transform_dirty = True

    def set_scale(self, scale: float):
        self.scale = scale
        self._transform_dirty = True

    def translate(self, dx: float, dy: float, dz: float):
        self.position += np.array([dx, dy, dz])
        self._transform_dirty = True

    def get_bounds(self) -> np.ndarray:
        mesh = self.get_transformed_mesh()
        return np.array([
            [mesh.bounds[0][0], mesh.bounds[1][0]],
            [mesh.bounds[0][1], mesh.bounds[1][1]],
            [mesh.bounds[0][2], mesh.bounds[1][2]]
        ])

    def get_volume(self) -> float:
        return abs(self.get_transformed_mesh().volume)

    def get_surface_area(self) -> float:
        return self.get_transformed_mesh().area

    def get_center(self) -> np.ndarray:
        return self.get_transformed_mesh().center_mass

    def get_vertices(self) -> np.ndarray:
        return self.get_transformed_mesh().vertices

    def get_faces(self) -> np.ndarray:
        return self.get_transformed_mesh().faces

    def get_original_bounds(self) -> np.ndarray:
        mesh = self.trimesh_obj
        if self.scale != 1.0:
            scaled = mesh.copy()
            scaled.apply_scale(self.scale)
            mesh = scaled
        return np.array([
            [mesh.bounds[0][0], mesh.bounds[1][0]],
            [mesh.bounds[0][1], mesh.bounds[1][1]],
            [mesh.bounds[0][2], mesh.bounds[1][2]]
        ])

    def get_original_size(self) -> Tuple[float, float, float]:
        bounds = self.get_original_bounds()
        return (
            bounds[0][1] - bounds[0][0],
            bounds[1][1] - bounds[1][0],
            bounds[2][1] - bounds[2][0]
        )

class PartManager:

    def __init__(self):
        self.parts: Dict[str, Part] = {}
        self._part_order: List[str] = []
        self.tessellation_voxel_size_in: Optional[float] = None

    def load_part(self, filepath: str, name: Optional[str] = None,
                  position: Optional[Tuple[float, float, float]] = None,
                  preserve_origin: bool = True) -> Optional[str]:
        logger.info(f"Loading part from: {filepath}")

        if not os.path.exists(filepath):
            logger.error(f"Part file not found: {filepath}")
            print(f"Error: File not found: {filepath}")
            return None

        ext = os.path.splitext(filepath)[1].lower()
        if ext in _STEP_EXTENSIONS and not STEP_SUPPORTED:
            message = (
                "STEP support requires the 'cascadio' package. "
                "Install with: pip install cascadio"
            )
            logger.error(message)
            print(f"Error: {message}")
            return None

        try:
            part_id = str(uuid.uuid4())[:8]

            trimesh_obj = trimesh.load(filepath)

            check_path: Optional[str] = filepath
            if isinstance(trimesh_obj, trimesh.Scene):
                if len(trimesh_obj.geometry) > 1:
                    check_path = None
                if not trimesh_obj.geometry:
                    raise ValueError("STEP file contains no bodies")
                first_body_name, first_body_mesh = next(
                    iter(trimesh_obj.geometry.items())
                )
                logger.warning(
                    f"STEP file '{filepath}' contains multiple bodies; "
                    f"registering only '{first_body_name}'. "
                    f"Use load_assembly_parts() to load all bodies."
                )
                trimesh_obj = first_body_mesh

            if ext in _STEP_EXTENSIONS:
                _scale_step_to_inches(trimesh_obj, check_path)

            if name is None:
                name = os.path.splitext(os.path.basename(filepath))[0]

            base_name = name
            counter = 1
            while any(p.name == name for p in self.parts.values()):
                name = f"{base_name}_{counter}"
                counter += 1

            inferred_type = infer_part_type_from_name(filepath)

            part = Part(
                id=part_id,
                name=name,
                filepath=filepath,
                trimesh_obj=trimesh_obj,
                part_type=inferred_type,
                color=PART_TYPE_COLORS.get(inferred_type, (0.7, 0.75, 0.8)),
            )

            if position is not None:
                part.set_position(*position)
            elif not preserve_origin:
                bounds = part.get_original_bounds()
                center_x = -(bounds[0][0] + bounds[0][1]) / 2
                center_y = -(bounds[1][0] + bounds[1][1]) / 2
                center_z = -bounds[2][0]
                part.set_position(center_x, center_y, center_z)

            self.parts[part_id] = part
            self._part_order.append(part_id)

            bounds = trimesh_obj.bounds
            logger.info(f"Part loaded: '{name}' (ID: {part_id}), type={part.part_type.value}, vertices={len(trimesh_obj.vertices):,}, faces={len(trimesh_obj.faces):,}, volume={part.get_volume():.4f} in³")
            print(f"✓ Loaded part '{name}' (ID: {part_id})")
            print(f"  Type: {part.part_type.value.upper()} (auto-detected from filename)")
            print(f"  Vertices: {len(trimesh_obj.vertices):,}")
            print(f"  Faces: {len(trimesh_obj.faces):,}")
            print(f"  Bounds: X[{bounds[0][0]:.2f}, {bounds[1][0]:.2f}] Y[{bounds[0][1]:.2f}, {bounds[1][1]:.2f}] Z[{bounds[0][2]:.2f}, {bounds[1][2]:.2f}]")
            print(f"  Volume: {part.get_volume():.4f} in³")

            return part_id

        except Exception as e:
            logger.exception(f"Failed to load CAD part: {filepath}")
            print(f"Error loading CAD file: {e}")
            return None

    def load_assembly_parts(
        self,
        filepath: str,
        name_prefix: Optional[str] = None,
        voxel_size_in: Optional[float] = None,
    ) -> List[str]:
        logger.info(f"Loading assembly from: {filepath}")

        if not os.path.exists(filepath):
            logger.error(f"Assembly file not found: {filepath}")
            print(f"Error: File not found: {filepath}")
            return []

        ext = os.path.splitext(filepath)[1].lower()
        if ext in _STEP_EXTENSIONS and not STEP_SUPPORTED:
            message = (
                "STEP support requires the 'cascadio' package. "
                "Install with: pip install cascadio"
            )
            logger.error(message)
            print(f"Error: {message}")
            return []

        try:
            if voxel_size_in is None:
                voxel_size_in = self.tessellation_voxel_size_in
            tol_linear = (
                tessellation_tolerance_mm(voxel_size_in)
                if ext in _STEP_EXTENSIONS
                else None
            )
            scene = _load_cad_scene(filepath, tol_linear)

            if not scene.geometry:
                raise ValueError("Assembly contains no bodies")

            part_ids: List[str] = []
            base_prefix = name_prefix or os.path.splitext(os.path.basename(filepath))[0]
            single_body = _count_nonempty_bodies(scene) == 1

            for body_index, (body_name, body_mesh) in enumerate(scene.geometry.items()):
                if body_mesh is None or len(body_mesh.faces) == 0:
                    logger.warning(
                        f"Skipping empty body '{body_name}' in '{filepath}'"
                    )
                    continue

                part_id = str(uuid.uuid4())[:8]
                display_name, inferred_type = _default_body_identity(
                    filepath, body_name, single_body
                )
                part_name = f"{base_prefix}_{display_name}" if name_prefix else display_name
                unique_name = part_name
                counter = 1
                while any(p.name == unique_name for p in self.parts.values()):
                    unique_name = f"{part_name}_{counter}"
                    counter += 1

                part = Part(
                    id=part_id,
                    name=unique_name,
                    filepath=filepath,
                    trimesh_obj=body_mesh,
                    part_type=inferred_type,
                    color=PART_TYPE_COLORS.get(inferred_type, (0.7, 0.75, 0.8)),
                    body_name=body_name,
                    body_index=body_index,
                    tessellation_tolerance_mm=tol_linear,
                )

                self.parts[part_id] = part
                self._part_order.append(part_id)
                part_ids.append(part_id)

                logger.info(
                    f"Assembly body loaded: '{unique_name}' (ID: {part_id}), "
                    f"type={part.part_type.value}, "
                    f"vertices={len(body_mesh.vertices):,}, "
                    f"faces={len(body_mesh.faces):,}"
                )

            print(
                f"✓ Loaded assembly '{base_prefix}' with {len(part_ids)} body(ies) "
                f"from {os.path.basename(filepath)}"
            )
            return part_ids

        except Exception as e:
            logger.exception(f"Failed to load assembly: {filepath}")
            print(f"Error loading assembly: {e}")
            return []

    def retessellate_for_voxel_size(self, voxel_size_in: Optional[float]) -> int:
        target = tessellation_tolerance_mm(voxel_size_in)
        self.tessellation_voxel_size_in = voxel_size_in
        if target is None:
            return 0

        by_file: Dict[str, List[Part]] = {}
        for part in self.parts.values():
            if os.path.splitext(part.filepath)[1].lower() not in _STEP_EXTENSIONS:
                continue
            current = getattr(part, "tessellation_tolerance_mm", None)
            if current is not None and current <= target + 1e-12:
                continue
            by_file.setdefault(part.filepath, []).append(part)

        retessellated = 0
        for filepath, parts in by_file.items():
            if not os.path.exists(filepath):
                logger.warning(
                    f"Cannot re-tessellate '{filepath}': file no longer exists. "
                    f"Keeping the geometry loaded earlier."
                )
                continue
            try:
                scene = _load_cad_scene(filepath, target)
                bodies = list(scene.geometry.items())
            except Exception:
                logger.exception(f"Re-tessellation failed for '{filepath}'")
                continue

            for part in parts:
                replacement = self._match_body(part, bodies)
                if replacement is None:
                    logger.warning(
                        f"Could not match body '{part.body_name or part.name}' "
                        f"in '{filepath}' while re-tessellating; keeping the "
                        f"geometry loaded earlier."
                    )
                    continue
                part.trimesh_obj = replacement
                part.tessellation_tolerance_mm = target
                part._transform_dirty = True
                part._transformed_mesh = None
                retessellated += 1

        if retessellated:
            logger.info(
                f"Re-tessellated {retessellated} STEP body(ies) to "
                f"{target:.4g} mm for a {voxel_size_in:.3f} in voxel"
            )
        return retessellated

    @staticmethod
    def _match_body(part: Part, bodies: List[Tuple[str, trimesh.Trimesh]]):
        if part.body_name:
            for name, mesh in bodies:
                if name == part.body_name:
                    return mesh
        index = part.body_index
        if index is not None and 0 <= index < len(bodies):
            if part.body_name:
                logger.warning(
                    f"CAD body '{part.body_name}' of part '{part.name}' is no "
                    f"longer in '{part.filepath}'; using body #{index} "
                    f"('{bodies[index][0]}') by position instead."
                )
            return bodies[index][1]
        return None

    def load_step_bodies(
        self,
        filepath: str,
        body_refs: Sequence[Mapping[str, Any]],
        voxel_size_in: Optional[float] = None,
    ) -> List[Optional[str]]:
        refs = list(body_refs)
        result: List[Optional[str]] = [None] * len(refs)
        if not refs:
            return result
        if not os.path.exists(filepath):
            logger.error(f"STEP file not found: {filepath}")
            return result
        if not STEP_SUPPORTED:
            logger.error(
                "STEP support requires the 'cascadio' package. "
                "Install with: pip install cascadio"
            )
            return result

        if voxel_size_in is None:
            voxel_size_in = self.tessellation_voxel_size_in
        tol_linear = tessellation_tolerance_mm(voxel_size_in)
        try:
            scene = _load_cad_scene(filepath, tol_linear)
        except Exception:
            logger.exception(f"Failed to load STEP file: {filepath}")
            return result

        bodies = list(scene.geometry.items())
        nonempty = [
            index for index, (_, mesh) in enumerate(bodies)
            if mesh is not None and len(mesh.faces) > 0
        ]
        single_body = len(nonempty) == 1
        name_to_index = {name: index for index, (name, _) in enumerate(bodies)}
        used = set()
        all_legacy = all(
            not ref.get("body_name") and ref.get("body_index") is None
            for ref in refs
        )

        legacy_refs = 0
        for position, ref in enumerate(refs):
            body_name = ref.get("body_name") or None
            body_index = ref.get("body_index")
            index: Optional[int] = None
            if body_name is not None and body_name in name_to_index:
                index = name_to_index[body_name]
            elif body_index is not None and 0 <= int(body_index) < len(bodies):
                index = int(body_index)
                if body_name is not None:
                    logger.warning(
                        f"CAD body '{body_name}' of saved part '{ref.get('name')}' "
                        f"is no longer in '{filepath}'; using body #{index} "
                        f"('{bodies[index][0]}') by position instead."
                    )
            elif body_name is None and body_index is None:
                legacy_refs += 1
                if all_legacy:
                    if position < len(nonempty):
                        index = nonempty[position]
                elif nonempty:
                    index = nonempty[0]

            if index is None or index not in nonempty:
                logger.warning(
                    f"No body in '{filepath}' matches saved part "
                    f"'{ref.get('name')}' (body_name={body_name!r}, "
                    f"body_index={body_index!r}); it was not loaded."
                )
                continue

            scene_body_name, mesh = bodies[index]
            if index in used:
                mesh = mesh.copy()
            used.add(index)

            default_name, inferred_type = _default_body_identity(
                filepath, scene_body_name, single_body
            )
            base_name = ref.get("name") or default_name
            unique_name = base_name
            counter = 1
            while any(p.name == unique_name for p in self.parts.values()):
                unique_name = f"{base_name}_{counter}"
                counter += 1

            part_id = str(uuid.uuid4())[:8]
            part = Part(
                id=part_id,
                name=unique_name,
                filepath=filepath,
                trimesh_obj=mesh,
                part_type=inferred_type,
                color=PART_TYPE_COLORS.get(inferred_type, (0.7, 0.75, 0.8)),
                body_name=scene_body_name,
                body_index=index,
                tessellation_tolerance_mm=tol_linear,
            )
            self.parts[part_id] = part
            self._part_order.append(part_id)
            result[position] = part_id

        if legacy_refs:
            how = (
                "matched to bodies by saved order"
                if all_legacy
                else "given the file's first body"
            )
            logger.warning(
                f"{legacy_refs} saved part(s) from '{filepath}' have no recorded "
                f"CAD body; {how}. Re-save the project to record body names."
            )
        return result

    def restore_parts(
        self,
        entries: Sequence[Mapping[str, Any]],
        voxel_size_in: Optional[float] = None,
    ) -> List[Optional[str]]:
        entries = list(entries)
        part_ids: List[Optional[str]] = [None] * len(entries)
        step_groups: Dict[str, List[int]] = {}
        for position, entry in enumerate(entries):
            filepath = str(entry["filepath"])
            if os.path.splitext(filepath)[1].lower() in _STEP_EXTENSIONS:
                step_groups.setdefault(filepath, []).append(position)
            else:
                part_ids[position] = self.load_part(
                    filepath, name=entry.get("name"), preserve_origin=True
                )

        for filepath, positions in step_groups.items():
            loaded = self.load_step_bodies(
                filepath, [entries[p] for p in positions], voxel_size_in
            )
            for position, part_id in zip(positions, loaded):
                part_ids[position] = part_id

        restored = [part_id for part_id in part_ids if part_id]
        restored_set = set(restored)
        self._part_order = [
            part_id for part_id in self._part_order if part_id not in restored_set
        ] + restored

        for entry, part_id in zip(entries, part_ids):
            part = self.parts.get(part_id) if part_id else None
            if part is None:
                continue
            part.set_part_type(part_type_from_value(entry.get("part_type", "cast")))
            part.set_position(*entry.get("position", (0.0, 0.0, 0.0)))
            part.set_rotation(*entry.get("rotation", (0.0, 0.0, 0.0)))
            part.set_scale(float(entry.get("scale", 1.0)))
            if entry.get("color") is not None:
                part.color = tuple(entry["color"])
            part.visible = bool(entry.get("visible", True))
            part.is_open_top = bool(entry.get("is_open_top", False))
            part.riser_sleeve = riser_sleeve_from_value(entry.get("riser_sleeve"))
        return part_ids

    def remove_part(self, part_id: str) -> bool:
        if part_id in self.parts:
            name = self.parts[part_id].name
            del self.parts[part_id]
            self._part_order.remove(part_id)
            logger.info(f"Part removed: '{name}' (ID: {part_id})")
            print(f"Removed part '{name}'")
            return True
        logger.warning(f"Attempted to remove non-existent part: {part_id}")
        return False

    def get_part(self, part_id: str) -> Optional[Part]:
        return self.parts.get(part_id)

    def get_part_by_name(self, name: str) -> Optional[Part]:
        for part in self.parts.values():
            if part.name == name:
                return part
        return None

    def get_all_parts(self) -> List[Part]:
        return [self.parts[pid] for pid in self._part_order if pid in self.parts]

    def get_part_count(self) -> int:
        return len(self.parts)

    def get_parts_with_types(self) -> List[Tuple[trimesh.Trimesh, int]]:
        result = []

        for part in self._voxel_parts():
            mesh = part.get_transformed_mesh()
            type_code = PART_TYPE_TO_CODE.get(part.part_type, PART_TYPE_CAST)
            result.append((mesh, type_code))

        return result

    def clear_all(self):
        self.parts.clear()
        self._part_order.clear()
        print("Cleared all parts")

    def _voxel_parts(self) -> List[Part]:
        return [
            p for p in self.get_all_parts()
            if p.visible and p.part_type is not PartType.MOLD
        ]

    def get_mold_meshes(self) -> List[trimesh.Trimesh]:
        return [
            p.get_transformed_mesh() for p in self.get_all_parts()
            if p.visible and p.part_type is PartType.MOLD
        ]

    def get_mold_setup_parts(self) -> List[Tuple[trimesh.Trimesh, int, bool]]:
        return [
            (
                p.get_transformed_mesh(),
                PART_TYPE_TO_CODE.get(p.part_type, PART_TYPE_CAST),
                bool(p.is_open_top),
            )
            for p in self._voxel_parts()
        ]

    def get_riser_sleeve_parts(self) -> List[Tuple[trimesh.Trimesh, str]]:
        return [
            (p.get_transformed_mesh(), riser_sleeve_from_value(p.riser_sleeve))
            for p in self._voxel_parts()
            if p.part_type is PartType.RISER
        ]

    def get_combined_mesh(self) -> Optional[trimesh.Trimesh]:
        visible_parts = self._voxel_parts()

        if not visible_parts:
            return None

        if len(visible_parts) == 1:
            return visible_parts[0].get_transformed_mesh()

        meshes = [p.get_transformed_mesh() for p in visible_parts]
        combined = trimesh.util.concatenate(meshes)

        return combined

    def get_combined_bounds(self) -> Optional[np.ndarray]:
        visible_parts = self._voxel_parts()

        if not visible_parts:
            return None

        all_bounds = [p.get_bounds() for p in visible_parts]

        combined = np.array([
            [min(b[0][0] for b in all_bounds), max(b[0][1] for b in all_bounds)],
            [min(b[1][0] for b in all_bounds), max(b[1][1] for b in all_bounds)],
            [min(b[2][0] for b in all_bounds), max(b[2][1] for b in all_bounds)],
        ])

        return combined

    def get_total_volume(self) -> float:
        return sum(p.get_volume() for p in self._voxel_parts())

    def get_total_surface_area(self) -> float:
        return sum(p.get_surface_area() for p in self.parts.values() if p.visible)

    def auto_arrange_grid(self, spacing: float = 1.0):
        parts = self.get_all_parts()
        if not parts:
            return

        n = len(parts)
        cols = int(np.ceil(np.sqrt(n)))

        max_x = max(p.get_original_size()[0] for p in parts)
        max_y = max(p.get_original_size()[1] for p in parts)

        cell_x = max_x + spacing
        cell_y = max_y + spacing

        for i, part in enumerate(parts):
            row = i // cols
            col = i % cols

            orig_bounds = part.get_original_bounds()
            part_width = orig_bounds[0][1] - orig_bounds[0][0]
            part_depth = orig_bounds[1][1] - orig_bounds[1][0]

            x = col * cell_x + (cell_x - part_width) / 2 - orig_bounds[0][0]
            y = row * cell_y + (cell_y - part_depth) / 2 - orig_bounds[1][0]
            z = -orig_bounds[2][0]

            part.set_position(x, y, z)

        print(f"Arranged {n} parts in {cols}x{(n + cols - 1) // cols} grid")

    def auto_arrange_line(self, axis: str = 'x', spacing: float = 1.0):
        parts = self.get_all_parts()
        if not parts:
            return

        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]

        current_pos = 0.0

        for part in parts:
            orig_bounds = part.get_original_bounds()
            part_size = orig_bounds[axis_idx][1] - orig_bounds[axis_idx][0]

            pos = [0.0, 0.0, 0.0]
            pos[axis_idx] = current_pos - orig_bounds[axis_idx][0]

            for i in range(3):
                if i != axis_idx:
                    pos[i] = -orig_bounds[i][0]

            part.set_position(*pos)

            current_pos += part_size + spacing

        print(f"Arranged {len(parts)} parts in line along {axis.upper()}-axis")

    def duplicate_part(self, part_id: str, offset: Tuple[float, float, float] = (2.0, 0.0, 0.0)) -> Optional[str]:
        original = self.get_part(part_id)
        if not original:
            return None

        base_name = f"{original.name}_copy"
        name = base_name
        counter = 1
        while any(p.name == name for p in self.parts.values()):
            name = f"{base_name}_{counter}"
            counter += 1

        new_id = str(uuid.uuid4())[:8]
        new_part = Part(
            id=new_id,
            name=name,
            filepath=original.filepath,
            trimesh_obj=original.trimesh_obj.copy(),
            part_type=original.part_type,
            body_name=original.body_name,
            body_index=original.body_index,
            tessellation_tolerance_mm=original.tessellation_tolerance_mm,
            is_open_top=original.is_open_top,
            riser_sleeve=original.riser_sleeve,
        )
        self.parts[new_id] = new_part
        self._part_order.append(new_id)
        logger.info(f"Part duplicated: '{original.name}' -> '{name}' (ID: {new_id})")

        new_part.set_position(
            original.position[0] + offset[0],
            original.position[1] + offset[1],
            original.position[2] + offset[2]
        )
        new_part.set_rotation(*original.rotation)
        new_part.set_scale(original.scale)
        new_part.color = original.color

        return new_id

    def is_valid(self) -> bool:
        return len(self.parts) > 0

    def get_summary(self) -> str:
        if not self.parts:
            return "No parts loaded"

        lines = [f"Parts: {len(self.parts)}"]
        for part in self.get_all_parts():
            bounds = part.get_bounds()
            dims = f"{bounds[0][1]-bounds[0][0]:.2f}\" x {bounds[1][1]-bounds[1][0]:.2f}\" x {bounds[2][1]-bounds[2][0]:.2f}\""
            pos = f"({part.position[0]:.2f}, {part.position[1]:.2f}, {part.position[2]:.2f})"
            lines.append(f"  • {part.name}: {dims} @ {pos}")

        lines.append(f"Total volume: {self.get_total_volume():.2f} in³")
        return "\n".join(lines)

class MultiPartSTLLoader:

    def __init__(self):
        self.part_manager = PartManager()
        self._current_part_id: Optional[str] = None

        self.mesh = None
        self.trimesh_obj = None
        self.filename = None
        self.bounds = None
        self.volume = None
        self.surface_area = None
        self.center = None

    def load_stl(self, filepath: str, units: str = 'inches') -> bool:
        self.part_manager.clear_all()
        part_id = self.part_manager.load_part(filepath)

        if part_id:
            self._current_part_id = part_id
            self._update_legacy_properties()
            return True
        return False

    def add_part(self, filepath: str, name: Optional[str] = None,
                 position: Optional[Tuple[float, float, float]] = None) -> Optional[str]:
        part_id = self.part_manager.load_part(filepath, name, position)
        if part_id:
            self._update_legacy_properties()
        return part_id

    def _update_legacy_properties(self):
        combined = self.part_manager.get_combined_mesh()
        if combined is None:
            return

        self.trimesh_obj = combined
        self.bounds = self.part_manager.get_combined_bounds()
        self.volume = self.part_manager.get_total_volume()
        self.surface_area = self.part_manager.get_total_surface_area()
        self.center = combined.center_mass

        names = [p.name for p in self.part_manager.get_all_parts()]
        self.filename = ", ".join(names) if names else None

    def get_dimensions(self) -> str:
        if self.bounds is None:
            return "N/A"

        x_size = self.bounds[0][1] - self.bounds[0][0]
        y_size = self.bounds[1][1] - self.bounds[1][0]
        z_size = self.bounds[2][1] - self.bounds[2][0]

        return f"{x_size:.2f}\" x {y_size:.2f}\" x {z_size:.2f}\""

    def get_vertices(self) -> np.ndarray:
        if self.trimesh_obj is None:
            return np.array([])
        return self.trimesh_obj.vertices

    def get_faces(self) -> np.ndarray:
        if self.trimesh_obj is None:
            return np.array([])
        return self.trimesh_obj.faces

    def get_bounds_array(self) -> np.ndarray:
        return self.bounds

    def is_valid(self) -> bool:
        return self.part_manager.is_valid()

    def get_part_manager(self) -> PartManager:
        return self.part_manager

_PART_TYPE_KEY_MAP: Dict[str, PartType] = {
    part_type.value: part_type for part_type in PartType
}

def build_part_manager(
    parts_data: Sequence[Dict[str, Any]],
    *,
    preserve_origin: bool = True,
    default_visible: bool = True,
    default_is_open_top: bool = False,
) -> PartManager:
    manager = PartManager()
    for part_data in parts_data:
        filepath = str(part_data.get("filepath", ""))
        if not filepath or not os.path.exists(filepath):
            raise FileNotFoundError(f"Part file not found: {filepath}")

        part_id = manager.load_part(
            filepath,
            name=part_data.get("name"),
            preserve_origin=preserve_origin,
        )
        if not part_id:
            raise RuntimeError(f"Failed to load part: {filepath}")

        part = manager.get_part(part_id)
        if part is None:
            raise RuntimeError(f"Loaded part missing from manager: {part_id}")

        part.set_position(
            float(part_data.get("position_x", 0.0)),
            float(part_data.get("position_y", 0.0)),
            float(part_data.get("position_z", 0.0)),
        )
        part.set_rotation(
            float(part_data.get("rotation_x", 0.0)),
            float(part_data.get("rotation_y", 0.0)),
            float(part_data.get("rotation_z", 0.0)),
        )
        part.set_scale(float(part_data.get("scale", 1.0)))

        inferred_type = infer_part_type_from_filename(filepath)
        part_name = str(part_data.get("name", "")).lower()
        if inferred_type != PartType.RISER_POST:
            if any(
                keyword in part_name
                for keyword in ("riser post", "riser_post", "post", "neck")
            ):
                inferred_type = PartType.RISER_POST
        part_type = _PART_TYPE_KEY_MAP.get(
            str(part_data.get("part_type", "cast")), PartType.CAST
        )
        if part_type == PartType.RISER and inferred_type == PartType.RISER_POST:
            part_type = PartType.RISER_POST
        part.set_part_type(part_type)
        part.visible = bool(part_data.get("visible", default_visible))
        part.is_open_top = bool(part_data.get("is_open_top", default_is_open_top))
        part.riser_sleeve = riser_sleeve_from_value(part_data.get("riser_sleeve"))

    return manager
