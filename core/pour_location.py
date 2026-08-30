"""Pure data model for a user-selected pour (metal-entry) location.

A pour location is the point where the user tells the simulation metal
enters the mold -- clicked on the 3D assembly, the way MAGMASOFT lets an
operator place an ingate. The mold-filling stage only runs once such a
location exists; everything upstream of it (part loading, voxelization) is
unaffected by whether one has been chosen yet.

Two front ends produce a :class:`PourLocation`: the VTK desktop viewer and
the Babylon web viewer. Both resolve a mouse click to a point on the
assembly surface and hand this module a world-space point (and, where
available, the hit surface's normal and the part that was clicked). This
module has no idea which viewer produced the point.

Units
-----
``point_in`` is in **inches**, matching every other geometry layer in this
project (the mesh, ``core.part_manager``, and both viewers). See
:mod:`physics.units` for the project's unit zones. Only the solver, which
works in SI, needs metres -- use the :attr:`PourLocation.point_m` property
to get there rather than converting inline at call sites.

Why the point, not the voxel indices, gets persisted
------------------------------------------------------
It is tempting to store which voxels the pour location maps to, since
that is ultimately what the filling solver needs as an inlet boundary.
That would be wrong to persist: voxel indices are keyed to one particular
voxelization, and the user can re-mesh the same assembly at a different
voxel size at any time, which renumbers (and resizes) every voxel. The
world-space point survives a re-mesh unchanged. So a :class:`PourLocation`
stores only the stable world point, and :func:`resolve_inlet_voxels` maps
it to voxel indices fresh, at run time, against whatever voxelization is
currently loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from physics.units import inches_to_meters

#: Schema versions this module knows how to interpret. Bump this and add a
#: migration branch in :meth:`PourLocation.from_dict` when the persisted
#: shape changes -- an unrecognized version must fail loudly rather than be
#: silently misinterpreted as the current shape.
SUPPORTED_SCHEMA_VERSIONS = (1,)

#: A plain (x, y, z) tuple. Not a numpy array -- this type is meant to be
#: cheap to construct, compare and hash, and to round-trip through JSON
#: without a custom encoder.
Point3 = Tuple[float, float, float]


def _validated_point(values: Sequence[float], label: str) -> Point3:
    """Coerce ``values`` to a finite 3-tuple of floats.

    Args:
        values: Sequence of exactly 3 numbers.
        label: Field name, used only to make the error message specific.

    Returns:
        The values as a ``(float, float, float)`` tuple.

    Raises:
        ValueError: If ``values`` does not have exactly 3 elements, or any
            element is not finite (NaN or +/-inf).
    """
    tupled = tuple(values)
    if len(tupled) != 3:
        raise ValueError(f"{label} must have exactly 3 components, got {tupled!r}")
    as_floats = tuple(float(v) for v in tupled)
    if not all(math.isfinite(v) for v in as_floats):
        raise ValueError(f"{label} must be finite, got {as_floats!r}")
    return as_floats  # type: ignore[return-value]


def _normalized(vector: Point3, label: str) -> Point3:
    """Return ``vector`` scaled to unit length.

    Args:
        vector: A finite 3-tuple (already validated by ``_validated_point``).
        label: Field name, used only to make the error message specific.

    Returns:
        A unit-length ``(x, y, z)`` tuple.

    Raises:
        ValueError: If ``vector`` has zero length (it has no direction to
            normalize to).
    """
    arr = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(arr))
    if length == 0.0:
        raise ValueError(f"{label} must not be the zero vector, got {vector!r}")
    normalized = arr / length
    return (float(normalized[0]), float(normalized[1]), float(normalized[2]))


@dataclass(frozen=True)
class PourLocation:
    """A user-chosen point where metal is poured into the mold.

    Immutable so a resolved pour location can be shared and cached safely
    (e.g. attached to a simulation run's inputs) without a caller mutating
    it out from under another reader.

    Attributes:
        point_in: World-space (x, y, z) of the pour point, in inches --
            the same coordinate space as the loaded parts and the mesh.
        normal: Optional unit surface normal at the clicked point, in the
            same coordinate space. Normalized on construction; ``None``
            when the front end could not determine a normal (e.g. a click
            resolved by nearest-point search rather than a ray hit).
        part_id: Optional :class:`core.part_manager.Part` id identifying
            which part was hit. Informational only -- nothing in this
            module depends on it, but a UI can use it to highlight the
            clicked part or a solver could use it to restrict search.
        schema_version: Persistence schema version. Present so a future
            format change can be detected and migrated instead of
            silently misread; see :data:`SUPPORTED_SCHEMA_VERSIONS`.
    """

    point_in: Point3
    normal: Optional[Point3] = None
    part_id: Optional[str] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        # dataclass(frozen=True) forbids plain attribute assignment, even in
        # __post_init__, hence object.__setattr__ to normalize in place.
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
        """``point_in`` converted to metres, for solver consumption.

        Returns:
            The pour point as an (x, y, z) tuple in metres.
        """
        x, y, z = self.point_in
        return (
            float(inches_to_meters(x)),
            float(inches_to_meters(y)),
            float(inches_to_meters(z)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Returns:
            A dict with ``schema_version``, ``point_in``, ``normal`` and
            ``part_id`` keys, suitable for ``json.dumps``.
        """
        return {
            "schema_version": self.schema_version,
            "point_in": list(self.point_in),
            "normal": list(self.normal) if self.normal is not None else None,
            "part_id": self.part_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PourLocation":
        """Deserialize from a dict produced by :meth:`to_dict`.

        Args:
            data: Dict with at least a ``point_in`` key. ``normal``,
                ``part_id`` and ``schema_version`` are optional and default
                the same way the dataclass fields do.

        Returns:
            The reconstructed :class:`PourLocation`.

        Raises:
            KeyError: If ``point_in`` is missing.
            ValueError: If the data is invalid (see ``__post_init__``),
                including an unrecognized ``schema_version``.
        """
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
    """Map a pour location's world point to nearby voxel indices.

    This is how a clicked point becomes an inlet boundary for the filling
    solver: it is a run-time lookup against whatever voxelization is
    currently loaded, deliberately *not* something stored on the
    :class:`PourLocation` itself -- see the module docstring for why voxel
    indices are not stable across a re-mesh but the world point is.

    Takes plain arrays rather than an ``AdaptiveVoxelMesh`` (or any mesh
    object) so this function stays mesh-agnostic and trivially testable;
    the caller is responsible for pulling ``voxel_centers`` /
    ``voxel_part_types`` off whatever mesh object it actually has.

    Args:
        location: The pour location to resolve.
        voxel_centers: ``(n, 3)`` float array of voxel center coordinates,
            in the same inch coordinate space as ``location.point_in``.
        voxel_size_in: Edge length of one voxel, in inches. Must be
            positive and finite.
        part_types: Optional ``(n,)`` int array of per-voxel part-type
            codes (see ``core.part_manager.PART_TYPE_*``), parallel to
            ``voxel_centers``. Required if ``allowed_codes`` is given.
        allowed_codes: Optional set of part-type codes to restrict the
            result to (e.g. only voxels currently classified as the
            gating system). When ``None``, part type is not considered.
        radius_voxels: Search radius, in units of ``voxel_size_in``. The
            physical search radius is ``radius_voxels * voxel_size_in``,
            so it scales automatically with whatever voxel size the
            current mesh happens to use.

    Returns:
        ``int`` array (dtype ``numpy.intp``) of indices into
        ``voxel_centers`` that fall within the search radius (and, if
        ``allowed_codes`` is given, match one of those codes). Empty if
        nothing is in range.

    Raises:
        ValueError: If ``voxel_centers`` is not shaped ``(n, 3)``, if
            ``voxel_size_in`` is not positive and finite, if
            ``allowed_codes`` is given without ``part_types``, or if
            ``part_types`` does not have one entry per row of
            ``voxel_centers``.
    """
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
