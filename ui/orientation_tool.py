"""Qt-free logic behind the desktop "Orient & Parting Line" tool.

The dialog in ``ui/dialogs/orientation_dialog.py`` only forwards button
presses here and redraws. Every transform goes through ``core.orientation``
so the desktop and web apps share one set of conventions: Z-up, inches,
``x = R(s*v) + p`` with ``R`` the static-XYZ ('sxyz') Euler matrix.

The parting line is a horizontal plane ``z = parting.z``. It is persisted in
the project ``ui_state["parting_line"]``; a missing key means "not set yet".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import trimesh

from core import orientation
from core.part_manager import Part, PartManager, PartType
from ui.project_state import normalize_parting_line

#: Rotation step choices (degrees). The first entry is the default.
ROTATION_STEPS_DEG: Tuple[float, ...] = (90.0, 45.0, 15.0, 5.0)

#: Height / parting nudge step choices (inches). Index 1 (0.25 in) is the default.
HEIGHT_STEPS_IN: Tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)

#: Preset name -> key of ``core.orientation.suggest_parting_z``.
PARTING_PRESET_KEYS: Dict[str, str] = {
    "widest": "max_section",
    "top": "top",
    "bottom": "bottom",
    "center": "center",
}

#: Part types that normally hang below the parting plane (drag side). Every
#: other non-casting type defaults to sitting on top of it (cope side).
_DRAG_SIDE_TYPES = frozenset({PartType.RUNNER, PartType.GATE, PartType.FILTER})

#: Fraction of the assembly's largest horizontal size added around the
#: parting plane so it reads as a plane, not as the casting's footprint.
PARTING_PLANE_MARGIN = 0.25


@dataclass
class PartingLine:
    """Flat horizontal parting plane. Above = cope, below = drag."""

    z: float
    visible: bool = True
    locked_to_casting: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "z": float(self.z),
            "visible": bool(self.visible),
            "locked_to_casting": bool(self.locked_to_casting),
        }

    @classmethod
    def from_dict(cls, value: object) -> Optional["PartingLine"]:
        """Parse a persisted ``ui_state["parting_line"]``; ``None`` if absent or malformed."""
        normalized = normalize_parting_line(value)
        if normalized is None:
            return None
        return cls(
            z=float(normalized["z"]),
            visible=bool(normalized["visible"]),
            locked_to_casting=bool(normalized["locked_to_casting"]),
        )


def default_seat_side(part_type: PartType) -> str:
    """``'below'`` for runner/gate/filter (drag side), ``'above'`` otherwise."""
    return "below" if part_type in _DRAG_SIDE_TYPES else "above"


def outward_face_normal(triangle: Sequence[Sequence[float]], toward_viewer: Sequence[float]) -> Optional[np.ndarray]:
    """Unit normal of a picked triangle, flipped to face the viewer.

    The face a user clicks is the one facing the camera, so its outward
    normal points back toward the viewer. Using the triangle geometry
    instead of the stored winding keeps "Face down" correct on STL files
    with inconsistent winding. Returns ``None`` for a degenerate triangle.
    """
    tri = np.asarray(triangle, dtype=float)
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        return None
    normal = normal / length
    if float(np.dot(normal, np.asarray(toward_viewer, dtype=float))) < 0.0:
        normal = -normal
    return normal


def _shift_z(part: Part, dz: float) -> None:
    """Move a part vertically (float math: ``Part.translate`` fails on an int position array)."""
    if dz:
        x, y, z = (float(v) for v in part.position)
        part.set_position(x, y, z + float(dz))


def part_world_bounds(part: Part) -> np.ndarray:
    """``[[xmin, ymin, zmin], [xmax, ymax, zmax]]`` of a placed part."""
    return orientation.world_bounds(part.trimesh_obj, part.rotation, part.position, part.scale)


@dataclass
class OrientationSnapshot:
    """Part transforms plus parting line, for "Revert changes"."""

    transforms: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]
    parting: Optional[Dict[str, object]]


class OrientationController:
    """Orientation, height and parting-line operations on a ``PartManager``.

    The target is the selected part, or the casting when nothing is selected.
    Every mutating method returns ``True`` when it changed something so the
    caller knows whether to redraw.
    """

    def __init__(
        self,
        part_manager: PartManager,
        selected_part_id: Callable[[], Optional[str]] = lambda: None,
    ) -> None:
        self.part_manager = part_manager
        self._selected_part_id = selected_part_id
        self.parting: Optional[PartingLine] = None
        self.keep_on_floor: bool = True
        #: "Lock to casting" value used when the parting line is first created.
        self.lock_default: bool = True

    # ------------------------------------------------------------------ parts

    def casting(self) -> Optional[Part]:
        for part in self.part_manager.get_all_parts():
            if part.part_type == PartType.CAST:
                return part
        return None

    def target(self) -> Optional[Part]:
        part_id = self._selected_part_id()
        if part_id:
            part = self.part_manager.get_part(part_id)
            if part is not None:
                return part
        return self.casting()

    def target_is_casting(self) -> bool:
        part = self.target()
        return part is not None and part.part_type == PartType.CAST

    def reference_part(self) -> Optional[Part]:
        """Part the parting presets and cope/drag split are measured on."""
        return self.casting() or self.target()

    def is_single_casting_project(self) -> bool:
        parts = self.part_manager.get_all_parts()
        return len(parts) == 1 and parts[0].part_type == PartType.CAST

    # ------------------------------------------------------------ orientation

    def _place(self, part: Part, rotation: Sequence[float], position: Sequence[float], bottom_before: float) -> None:
        part.set_rotation(*(float(a) for a in rotation))
        part.set_position(*(float(p) for p in position))
        if self.keep_on_floor:
            _shift_z(part, bottom_before - float(part_world_bounds(part)[0, 2]))

    def rotate(self, axis: str, angle_deg: float) -> bool:
        """Turn the target ``angle_deg`` about world ``axis`` through its bbox center."""
        part = self.target()
        if part is None or angle_deg == 0:
            return False
        bounds = part_world_bounds(part)
        pivot = bounds.mean(axis=0)
        rotation, position = orientation.rotate_about_world_axis(
            part.rotation, part.position, axis, angle_deg, pivot
        )
        self._place(part, rotation, position, float(bounds[0, 2]))
        return True

    def face_to(self, world_normal: Sequence[float], direction: str) -> bool:
        """Turn the target so a face with ``world_normal`` points down (-Z) or up (+Z)."""
        part = self.target()
        if part is None:
            return False
        target = (0.0, 0.0, -1.0) if direction == "down" else (0.0, 0.0, 1.0)
        bounds = part_world_bounds(part)
        rotation, position = orientation.face_to_direction(
            part.rotation, part.position, world_normal, bounds.mean(axis=0), target
        )
        self._place(part, rotation, position, float(bounds[0, 2]))
        return True

    def set_rotation(self, rx: float, ry: float, rz: float) -> bool:
        """Exact Euler entry (advanced). Keeps the bbox center, then the floor rule."""
        part = self.target()
        if part is None:
            return False
        return self._set_rotation_keep_center(part, (rx, ry, rz))

    def reset_orientation(self) -> bool:
        """Rotation back to (0, 0, 0) as imported."""
        part = self.target()
        if part is None:
            return False
        return self._set_rotation_keep_center(part, (0.0, 0.0, 0.0))

    def _set_rotation_keep_center(self, part: Part, rotation: Sequence[float]) -> bool:
        old = part_world_bounds(part)
        if np.allclose(part.rotation, rotation):
            return False
        center_before = old.mean(axis=0)
        new_bounds = orientation.world_bounds(part.trimesh_obj, rotation, part.position, part.scale)
        position = np.asarray(part.position, dtype=float) + (center_before - new_bounds.mean(axis=0))
        self._place(part, rotation, position, float(old[0, 2]))
        return True

    def size(self) -> Optional[Tuple[float, float, float]]:
        """World-axis W x D x H of the target (in)."""
        part = self.target()
        if part is None:
            return None
        bounds = part_world_bounds(part)
        extent = bounds[1] - bounds[0]
        return (float(extent[0]), float(extent[1]), float(extent[2]))

    # ----------------------------------------------------------------- height

    def bottom_z(self) -> Optional[float]:
        part = self.target()
        if part is None:
            return None
        return float(part_world_bounds(part)[0, 2])

    def move_z(self, dz: float) -> bool:
        """Raise (+) or lower (-) the target. A locked parting line follows the casting."""
        part = self.target()
        if part is None or dz == 0:
            return False
        _shift_z(part, float(dz))
        if (
            part.part_type == PartType.CAST
            and part is self.casting()
            and self.parting is not None
            and self.parting.locked_to_casting
        ):
            self.parting.z = round(self.parting.z + float(dz), 6)
        return True

    def set_bottom_z(self, z: float) -> bool:
        bottom = self.bottom_z()
        if bottom is None:
            return False
        return self.move_z(float(z) - bottom)

    def drop_to_floor(self) -> bool:
        part = self.target()
        if part is None:
            return False
        return self.move_z(orientation.drop_to_floor_dz(part_world_bounds(part), 0.0))

    # ---------------------------------------------------------- parting line

    def parting_presets(self) -> Optional[Dict[str, float]]:
        part = self.reference_part()
        if part is None:
            return None
        return orientation.suggest_parting_z(part.trimesh_obj, part.rotation, part.position, part.scale)

    def ensure_parting(self) -> Optional[PartingLine]:
        """Create the parting line at the widest section if it is not set yet."""
        if self.parting is None:
            presets = self.parting_presets()
            if presets is None:
                return None
            self.parting = PartingLine(
                z=float(presets["max_section"]), visible=True, locked_to_casting=self.lock_default
            )
        return self.parting

    def set_parting_z(self, z: float) -> bool:
        if self.parting is None:
            self.parting = PartingLine(z=float(z), visible=True, locked_to_casting=self.lock_default)
            return True
        if self.parting.z == float(z):
            return False
        self.parting.z = float(z)
        return True

    def nudge_parting(self, dz: float) -> bool:
        parting = self.ensure_parting()
        if parting is None:
            return False
        return self.set_parting_z(round(parting.z + float(dz), 6))

    def apply_parting_preset(self, name: str) -> bool:
        presets = self.parting_presets()
        if presets is None:
            return False
        changed = self.set_parting_z(float(presets[PARTING_PRESET_KEYS[name]]))
        if self.parting is not None and not self.parting.visible:
            self.parting.visible = True
            changed = True
        return changed

    def set_parting_visible(self, visible: bool) -> bool:
        if visible:
            parting = self.ensure_parting()
            if parting is None:
                return False
            parting.visible = True
            return True
        if self.parting is None or not self.parting.visible:
            return False
        self.parting.visible = False
        return True

    def set_parting_locked(self, locked: bool) -> None:
        self.lock_default = bool(locked)
        if self.parting is not None:
            self.parting.locked_to_casting = bool(locked)

    def parting_split(self) -> Optional[Dict[str, float]]:
        part = self.reference_part()
        if part is None or self.parting is None:
            return None
        return orientation.cope_drag_split(
            part.trimesh_obj, part.rotation, part.position, self.parting.z, part.scale
        )

    def suggested_seat_side(self) -> Optional[str]:
        part = self.target()
        if part is None or part.part_type == PartType.CAST:
            return None
        return default_seat_side(part.part_type)

    def seat_target(self, side: str) -> bool:
        """``'above'``: bottom on the plane (Seat). ``'below'``: top on the plane (Hang)."""
        part = self.target()
        if part is None or part.part_type == PartType.CAST or self.parting is None:
            return False
        dz = orientation.snap_to_parting_dz(part_world_bounds(part), self.parting.z, side)
        if dz == 0:
            return False
        _shift_z(part, dz)
        return True

    def section_segments(self) -> np.ndarray:
        """``(n, 2, 3)`` segments where the parting plane cuts the casting."""
        part = self.reference_part()
        if part is None or self.parting is None:
            return np.zeros((0, 2, 3))
        try:
            lines = trimesh.intersections.mesh_plane(
                part.get_transformed_mesh(),
                plane_normal=[0.0, 0.0, 1.0],
                plane_origin=[0.0, 0.0, float(self.parting.z)],
            )
        except Exception:
            return np.zeros((0, 2, 3))
        return np.asarray(lines, dtype=float).reshape(-1, 2, 3)

    def plane_extent(self, margin: float = PARTING_PLANE_MARGIN) -> Optional[Tuple[float, float, float, float]]:
        """``(xmin, xmax, ymin, ymax)`` of the drawn parting plane: assembly plus margin."""
        parts = [p for p in self.part_manager.get_all_parts() if p.visible] or self.part_manager.get_all_parts()
        if not parts:
            return None
        bounds = np.vstack([part_world_bounds(p) for p in parts])
        lo = bounds.min(axis=0)
        hi = bounds.max(axis=0)
        pad = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 1.0) * float(margin)
        return (float(lo[0] - pad), float(hi[0] + pad), float(lo[1] - pad), float(hi[1] + pad))

    # ---------------------------------------------------------------- revert

    def snapshot(self) -> OrientationSnapshot:
        transforms = {
            part.id: (orientation.as_vec3(part.position), orientation.as_vec3(part.rotation))
            for part in self.part_manager.get_all_parts()
        }
        return OrientationSnapshot(transforms, self.parting.to_dict() if self.parting else None)

    def restore(self, snapshot: OrientationSnapshot) -> bool:
        changed = False
        for part_id, (position, rotation) in snapshot.transforms.items():
            part = self.part_manager.get_part(part_id)
            if part is None:
                continue
            if not (np.allclose(part.position, position) and np.allclose(part.rotation, rotation)):
                part.set_position(*position)
                part.set_rotation(*rotation)
                changed = True
        parting = PartingLine.from_dict(snapshot.parting)
        current = self.parting.to_dict() if self.parting else None
        if current != snapshot.parting:
            self.parting = parting
            changed = True
        return changed
