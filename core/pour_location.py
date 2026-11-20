from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from physics.units import inches_to_meters

SUPPORTED_SCHEMA_VERSIONS = (1,)

Point3 = Tuple[float, float, float]

def _validated_point(values: Sequence[float], label: str) -> Point3:
    tupled = tuple(values)
    if len(tupled) != 3:
        raise ValueError(f"{label} must have exactly 3 components, got {tupled!r}")
    as_floats = tuple(float(v) for v in tupled)
    if not all(math.isfinite(v) for v in as_floats):
        raise ValueError(f"{label} must be finite, got {as_floats!r}")
    return as_floats

def _normalized(vector: Point3, label: str) -> Point3:
    arr = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(arr))
    if length == 0.0:
        raise ValueError(f"{label} must not be the zero vector, got {vector!r}")
    normalized = arr / length
    return (float(normalized[0]), float(normalized[1]), float(normalized[2]))

@dataclass(frozen=True)
class PourLocation:

    point_in: Point3
    normal: Optional[Point3] = None
    part_id: Optional[str] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "point_in", _validated_point(self.point_in, "point_in")
        )
        if self.normal is not None:
            validated_normal = _validated_point(self.normal, "normal")
            object.__setattr__(
                self, "normal", _normalized(validated_normal, "normal")
            )
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported PourLocation schema_version={self.schema_version!r}; "
                f"this build understands {SUPPORTED_SCHEMA_VERSIONS!r}"
            )

    @property
    def point_m(self) -> Point3:
        x, y, z = self.point_in
        return (
            float(inches_to_meters(x)),
            float(inches_to_meters(y)),
            float(inches_to_meters(z)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "point_in": list(self.point_in),
            "normal": list(self.normal) if self.normal is not None else None,
            "part_id": self.part_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PourLocation":
        normal = data.get("normal")
        return cls(
            point_in=tuple(data["point_in"]),
            normal=tuple(normal) if normal is not None else None,
            part_id=data.get("part_id"),
            schema_version=int(data.get("schema_version", 1)),
        )

def resolve_inlet_voxels(
    location: PourLocation,
    voxel_centers: np.ndarray,
    voxel_size_in: float,
    *,
    part_types: Optional[np.ndarray] = None,
    allowed_codes: Optional[Sequence[int]] = None,
    radius_voxels: float = 1.5,
) -> np.ndarray:
    centers = np.asarray(voxel_centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(
            f"voxel_centers must have shape (n, 3), got {centers.shape!r}"
        )

    if not math.isfinite(voxel_size_in) or voxel_size_in <= 0:
        raise ValueError(
            f"voxel_size_in must be positive and finite, got {voxel_size_in!r}"
        )

    if allowed_codes is not None and part_types is None:
        raise ValueError("allowed_codes requires part_types to be provided")

    part_types_arr: Optional[np.ndarray] = None
    if part_types is not None:
        part_types_arr = np.asarray(part_types)
        if part_types_arr.shape[0] != centers.shape[0]:
            raise ValueError(
                "part_types must have one entry per voxel_centers row: "
                f"{part_types_arr.shape[0]} != {centers.shape[0]}"
            )

    point = np.asarray(location.point_in, dtype=float)
    radius_in = radius_voxels * voxel_size_in

    distances = np.linalg.norm(centers - point, axis=1)
    mask = distances <= radius_in

    if allowed_codes is not None:
        mask &= np.isin(part_types_arr, np.asarray(list(allowed_codes)))

    return np.flatnonzero(mask).astype(np.intp)
