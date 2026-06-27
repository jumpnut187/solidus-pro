"""
Part Manager for Multiple STL Parts
Allows loading and positioning multiple castings in the simulation space
Supports part types: CAST, RISER, CHILL for simulation behavior

CAD format support
------------------
Parts can be loaded from any format that ``trimesh.load`` understands.
STL files are handled natively; STEP (``.step`` / ``.stp``) files are
handled via the optional ``cascadio`` backend, which statically links
OpenCASCADE and tessellates the B-rep to a triangle mesh on the fly.

Multi-body STEP assemblies are returned by ``trimesh.load`` as a
``trimesh.Scene``; :meth:`PartManager.load_assembly_parts` iterates
``scene.geometry`` and registers each body as its own ``Part`` with
its CAD body name preserved and part type auto-inferred.

If ``cascadio`` is not installed, ``STEP_SUPPORTED`` is ``False`` and
the loader raises a clear error directing users to install it.
"""

from __future__ import annotations

import numpy as np
import trimesh
from typing import Any, Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import uuid

try:
    from core.error_logger import logger
except ImportError:
    from core.error_logger_dummy import logger

# Optional STEP backend. Mirrors the GMSH_AVAILABLE pattern from
# riser_geometry_loader.py — silently absent is acceptable for STL-only
# users, but STEP loading must raise a clear error in that case.
try:
    import cascadio  # noqa: F401  (presence test only)
    STEP_SUPPORTED = True
except ImportError:
    STEP_SUPPORTED = False

# File extensions we treat as STEP assemblies. Anything else is assumed
# to be a single-body mesh (STL, OBJ, PLY, GLB, ...).
_STEP_EXTENSIONS = (".step", ".stp")

# Conversion factor from meters to inches. Used to undo cascadio's
# meters-unit GLB output for STEP files (see ``_scale_step_to_inches``).
_METERS_TO_INCHES = 39.3701


def _scale_step_to_inches(geometry):
    """Rescale cascadio/trimesh STEP output from meters back to inches.

    ``trimesh.load`` on a STEP file dispatches to ``cascadio``, which
    converts the STEP assembly to GLB as an intermediate. The glTF 2.0
    spec mandates that POSITION coordinates are in **meters**, so
    trimesh hands us STEP meshes in meters regardless of the STEP file's
    authored units (mm, inches, m, ...). Every other code path in this
    project (STLLoader, voxel_mesh, payloads, UI) assumes inches, so
    without rescaling a STEP file authored in mm is loaded ~39× too
    small — every downstream dimension collapses and defect analyses
    silently return 0.

    This helper applies the meters→inches scale to either a single
    ``Trimesh`` or to every body of a ``Scene``. Empty / placeholder
    meshes (zero faces) are skipped because ``trimesh`` cannot compute
    meaningful extents for them.

    Limitation: cascadio's own unit-detection heuristic is fragile for
    inch-authored STEP files (it conflates any extent > 1.0 with
    millimeters), so for that rare case the returned mesh would still
    be mis-scaled. STEP files from mainstream CAD tools
    (Solidworks, Onshape, etc.) almost always export in mm, where this
    helper is correct.
    """
    import numpy as np

    if isinstance(geometry, trimesh.Scene):
        bodies = list(geometry.geometry.values())
        for body in bodies:
            if body is None or len(body.faces) == 0:
                continue
            body.apply_scale(_METERS_TO_INCHES)
    elif geometry is not None and len(geometry.faces) > 0:
        geometry.apply_scale(_METERS_TO_INCHES)


class PartType(Enum):
    """
    Part type determines simulation behavior:
    - CAST: The actual casting - subject to shrinkage/porosity analysis
    - RISER: Feeder/riser with exothermic effect - slows cooling, not counted as defect
    - RISER_POST: Riser post/neck connecting riser to casting - behaves like riser but can be distinguished
    - CHILL: External chill - accelerates cooling, promotes directional solidification
    - CORE: Sand core - internal cavity form, excluded from defect analysis
    """
    CAST = "cast"
    RISER = "riser"
    RISER_POST = "riser_post"
    CHILL = "chill"
    CORE = "core"
    
    def __str__(self):
        return self.value.replace('_', ' ').title()


# Part type colors for visualization
PART_TYPE_COLORS = {
    PartType.CAST: (0.7, 0.75, 0.8),    # Default grey-blue (steel appearance)
    PartType.RISER: (0.9, 0.6, 0.2),    # Orange (exothermic sleeve color)
    PartType.RISER_POST: (0.8, 0.5, 0.3),  # Darker orange (riser neck/post)
    PartType.CHILL: (0.3, 0.3, 0.5),    # Dark blue-grey (cold metal appearance)
    PartType.CORE: (0.6, 0.5, 0.4),     # Brownish (sand core appearance)
}


# Keywords for automatic part type detection from filenames
# All keywords are matched case-insensitively
# Order matters: more specific patterns should be checked first
PART_TYPE_KEYWORDS = {
    PartType.CAST: ['casting', 'part'],
    PartType.RISER_POST: ['riser_post', 'riser post', 'post', 'neck'],  # Check before RISER
    PartType.RISER: ['feeder', 'riser', 'bridge', 'pad'],
    PartType.CHILL: ['chill'],
    PartType.CORE: ['core', 'sand_core', 'sand core'],
}


def infer_part_type_from_name(name: str) -> PartType:
    """
    Infer part type from a name string based on keywords.

    Keywords (case-insensitive, first match wins):
    - CAST: 'casting', 'part'
    - RISER_POST: 'riser_post', 'riser post', 'post', 'neck'
    - RISER: 'feeder', 'riser', 'bridge', 'pad'
    - CHILL: 'chill'
    - CORE: 'core', 'sand_core', 'sand core'

    The input is lowercased and matched as substrings, so underscores,
    spaces, and concatenated words are all handled. When checking a
    multi-body STEP assembly, this lets us infer the part type from
    each body's CAD name (e.g. ``"Riser_Sleeve"`` → ``RISER``).

    Args:
        name: A bare name or full filename. Path and extension are
              stripped if present; matching is otherwise identical to
              :func:`infer_part_type_from_filename`.

    Returns:
        Inferred :class:`PartType`, defaults to ``CAST`` if no
        keywords match.
    """
    base_name = os.path.splitext(os.path.basename(name))[0].lower()

    # CORE first (most specific for internal cavities)
    for keyword in PART_TYPE_KEYWORDS[PartType.CORE]:
        if keyword in base_name:
            return PartType.CORE

    # CHILL next (specific material/semantic)
    for keyword in PART_TYPE_KEYWORDS[PartType.CHILL]:
        if keyword in base_name:
            return PartType.CHILL

    # RISER_POST before RISER (since 'post' is more specific than 'riser')
    for keyword in PART_TYPE_KEYWORDS[PartType.RISER_POST]:
        if keyword in base_name:
            return PartType.RISER_POST

    for keyword in PART_TYPE_KEYWORDS[PartType.RISER]:
        if keyword in base_name:
            return PartType.RISER

    for keyword in PART_TYPE_KEYWORDS[PartType.CAST]:
        if keyword in base_name:
            return PartType.CAST

    return PartType.CAST


def infer_part_type_from_filename(filename: str) -> PartType:
    """
    Infer part type from a filename based on keywords.

    Convenience wrapper around :func:`infer_part_type_from_name` that
    accepts a full path or filename. Equivalent semantics; provided for
    backwards compatibility with callers that already pass file paths.

    Args:
        filename: The filename (with or without path/extension).

    Returns:
        Inferred :class:`PartType`, defaults to ``CAST`` if no
        keywords match.
    """
    return infer_part_type_from_name(filename)


@dataclass
class Part:
    """Represents a single part in the simulation"""
    id: str
    name: str
    filepath: str
    trimesh_obj: trimesh.Trimesh
    
    # Part type for simulation behavior
    part_type: PartType = PartType.CAST
    
    # Transform properties
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # Euler angles in degrees
    scale: float = 1.0
    
    # Cached transformed mesh
    _transformed_mesh: Optional[trimesh.Trimesh] = field(default=None, repr=False)
    _transform_dirty: bool = field(default=True, repr=False)
    
    # Visual properties
    color: Tuple[float, float, float] = (0.7, 0.75, 0.8)
    visible: bool = True
    
    # Riser-specific property: Is the riser open-top (vs blind)?
    # Open-top risers have atmospheric pressure pushing metal down for better feeding.
    # Blind risers can form vacuum as metal shrinks, reducing feeding effectiveness.
    # This property only has effect when part_type == RISER.
    is_open_top: bool = False
    
    def set_part_type(self, part_type: PartType):
        """Set part type and update color accordingly"""
        self.part_type = part_type
        self.color = PART_TYPE_COLORS.get(part_type, (0.7, 0.75, 0.8))
    
    def get_transformed_mesh(self) -> trimesh.Trimesh:
        """Get the transformed mesh (cached)"""
        if self._transform_dirty or self._transformed_mesh is None:
            self._apply_transform()
        return self._transformed_mesh
    
    def _apply_transform(self):
        """Apply position, rotation, and scale transforms"""
        # Start with a copy of the original mesh
        self._transformed_mesh = self.trimesh_obj.copy()
        
        # Apply scale
        if self.scale != 1.0:
            self._transformed_mesh.apply_scale(self.scale)
        
        # Apply rotation (Euler angles in degrees, XYZ order)
        if np.any(self.rotation != 0):
            rot_matrix = trimesh.transformations.euler_matrix(
                np.radians(self.rotation[0]),
                np.radians(self.rotation[1]),
                np.radians(self.rotation[2]),
                'sxyz'
            )
            self._transformed_mesh.apply_transform(rot_matrix)
        
        # Apply translation
        if np.any(self.position != 0):
            translation = trimesh.transformations.translation_matrix(self.position)
            self._transformed_mesh.apply_transform(translation)
        
        self._transform_dirty = False
    
    def set_position(self, x: float, y: float, z: float):
        """Set part position"""
        self.position = np.array([x, y, z])
        self._transform_dirty = True
    
    def set_rotation(self, rx: float, ry: float, rz: float):
        """Set part rotation (degrees)"""
        self.rotation = np.array([rx, ry, rz])
        self._transform_dirty = True
    
    def set_scale(self, scale: float):
        """Set part scale"""
        self.scale = scale
        self._transform_dirty = True
    
    def translate(self, dx: float, dy: float, dz: float):
        """Move part by delta"""
        self.position += np.array([dx, dy, dz])
        self._transform_dirty = True
    
    def get_bounds(self) -> np.ndarray:
        """Get bounds of transformed mesh"""
        mesh = self.get_transformed_mesh()
        return np.array([
            [mesh.bounds[0][0], mesh.bounds[1][0]],  # X
            [mesh.bounds[0][1], mesh.bounds[1][1]],  # Y
            [mesh.bounds[0][2], mesh.bounds[1][2]]   # Z
        ])
    
    def get_volume(self) -> float:
        """Get volume of part (accounts for scale)"""
        return abs(self.get_transformed_mesh().volume)
    
    def get_surface_area(self) -> float:
        """Get surface area of part"""
        return self.get_transformed_mesh().area
    
    def get_center(self) -> np.ndarray:
        """Get center of mass of transformed part"""
        return self.get_transformed_mesh().center_mass
    
    def get_vertices(self) -> np.ndarray:
        """Get transformed vertices"""
        return self.get_transformed_mesh().vertices
    
    def get_faces(self) -> np.ndarray:
        """Get faces"""
        return self.get_transformed_mesh().faces
    
    def get_original_bounds(self) -> np.ndarray:
        """Get bounds of the ORIGINAL mesh (ignoring current position/rotation)"""
        mesh = self.trimesh_obj
        # Apply only scale (not position/rotation) for size calculation
        if self.scale != 1.0:
            scaled = mesh.copy()
            scaled.apply_scale(self.scale)
            mesh = scaled
        return np.array([
            [mesh.bounds[0][0], mesh.bounds[1][0]],  # X
            [mesh.bounds[0][1], mesh.bounds[1][1]],  # Y
            [mesh.bounds[0][2], mesh.bounds[1][2]]   # Z
        ])
    
    def get_original_size(self) -> Tuple[float, float, float]:
        """Get the size (width, depth, height) of the original mesh"""
        bounds = self.get_original_bounds()
        return (
            bounds[0][1] - bounds[0][0],  # width (X)
            bounds[1][1] - bounds[1][0],  # depth (Y)
            bounds[2][1] - bounds[2][0]   # height (Z)
        )


class PartManager:
    """
    Manages multiple parts in the simulation.
    Allows loading, positioning, and combining parts for simulation.
    """
    
    def __init__(self):
        self.parts: Dict[str, Part] = {}
        self._part_order: List[str] = []  # Maintains insertion order
    
    def load_part(self, filepath: str, name: Optional[str] = None,
                  position: Optional[Tuple[float, float, float]] = None,
                  preserve_origin: bool = True) -> Optional[str]:
        """
        Load a CAD file (STL, STEP, OBJ, ...) as a new part.

        Args:
            filepath: Path to a CAD mesh file. STL/OBJ/PLY are handled
                      natively by ``trimesh``. ``.step``/``.stp`` require
                      the optional ``cascadio`` package.
            name: Optional name for the part (defaults to filename).
            position: Optional initial position (x, y, z) - overrides
                      the mesh's stored origin.
            preserve_origin: If True, keep the original mesh coordinates
                              (useful for assemblies). If False and no
                              position is given, center at origin with
                              the bottom face on Z=0.

        Returns:
            Part ID if successful, ``None`` otherwise.

        Notes:
            For multi-body STEP assemblies, use
            :meth:`load_assembly_parts` instead — this method returns
            only the first body.
        """
        logger.info(f"Loading part from: {filepath}")

        if not os.path.exists(filepath):
            logger.error(f"Part file not found: {filepath}")
            print(f"Error: File not found: {filepath}")
            return None

        # STEP-specific guard: fail fast with a clear install hint
        # instead of letting trimesh raise an opaque ImportError.
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
            # Generate unique ID
            part_id = str(uuid.uuid4())[:8]

            # Load mesh via trimesh (handles STL, STEP via cascadio, and other formats)
            trimesh_obj = trimesh.load(filepath)

            # If a multi-body STEP slipped through, only register the first body.
            # Callers should use load_assembly_parts() for full Scene support.
            if isinstance(trimesh_obj, trimesh.Scene):
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

            # STEP files load in meters (cascadio's GLB intermediate).
            # Rescale to inches so downstream STL-equivalent units match.
            if ext in _STEP_EXTENSIONS:
                _scale_step_to_inches(trimesh_obj)

            # Default name from filename
            if name is None:
                name = os.path.splitext(os.path.basename(filepath))[0]

            # Make name unique if needed
            base_name = name
            counter = 1
            while any(p.name == name for p in self.parts.values()):
                name = f"{base_name}_{counter}"
                counter += 1

            # Infer part type from filename (works for STL; for single-body
            # STEP, falls back to filename rather than the CAD body name).
            inferred_type = infer_part_type_from_name(filepath)

            # Create part with inferred type. Mirror load_part's existing
            # pattern of explicitly setting color from PART_TYPE_COLORS for
            # consistency with load_assembly_parts().
            part = Part(
                id=part_id,
                name=name,
                filepath=filepath,
                trimesh_obj=trimesh_obj,
                part_type=inferred_type,
                color=PART_TYPE_COLORS.get(inferred_type, (0.7, 0.75, 0.8)),
            )

            # Set position based on arguments
            if position is not None:
                # Explicit position provided - use it
                part.set_position(*position)
            elif not preserve_origin:
                # Center at origin if not preserving STL coordinates
                bounds = part.get_original_bounds()
                center_x = -(bounds[0][0] + bounds[0][1]) / 2
                center_y = -(bounds[1][0] + bounds[1][1]) / 2
                center_z = -bounds[2][0]  # Place bottom on Z=0
                part.set_position(center_x, center_y, center_z)
            # else: preserve_origin=True, position=None -> keep STL coordinates (position stays 0,0,0)

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
    ) -> List[str]:
        """
        Load a multi-body CAD assembly (typically a STEP file) and
        register each body as its own ``Part``.

        Bodies are loaded via ``trimesh.load(filepath, force="scene")``;
        each entry in ``scene.geometry`` becomes a Part whose name is
        the CAD body name and whose ``part_type`` is auto-inferred from
        that name via :func:`infer_part_type_from_name` (e.g. a body
        named ``Riser_Sleeve`` is classified as ``RISER``).

        Single-body files are handled by registering the lone mesh and
        returning a one-element list — semantically equivalent to
        :meth:`load_part` but uniformly returning a list.

        Args:
            filepath: Path to a CAD assembly file. ``.step``/``.stp``
                      require the optional ``cascadio`` package.
            name_prefix: Optional prefix prepended to each body name
                         (useful when loading multiple assemblies).

        Returns:
            List of newly created Part IDs in the same order as
            ``scene.geometry.items()``. Empty list on failure.
        """
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
            # force="scene" ensures trimesh returns a Scene even for
            # single-body files, so we can iterate uniformly.
            scene = trimesh.load(filepath, force="scene")

            if not isinstance(scene, trimesh.Scene):
                # Defensive: if trimesh ever returns a bare Trimesh
                # despite force="scene", wrap it.
                scene = trimesh.Scene(geometry={"Body_1": scene})

            if not scene.geometry:
                raise ValueError("Assembly contains no bodies")

            # STEP assemblies come back from cascadio in meters. Rescale
            # the whole scene once so every body's vertices are in
            # inches — matches STLLoader and the rest of the pipeline.
            ext = os.path.splitext(filepath)[1].lower()
            if ext in _STEP_EXTENSIONS:
                _scale_step_to_inches(scene)

            part_ids: List[str] = []
            base_prefix = name_prefix or os.path.splitext(os.path.basename(filepath))[0]

            for body_index, (body_name, body_mesh) in enumerate(scene.geometry.items()):
                if body_mesh is None or len(body_mesh.faces) == 0:
                    logger.warning(
                        f"Skipping empty body '{body_name}' in '{filepath}'"
                    )
                    continue

                part_id = str(uuid.uuid4())[:8]
                # Use CAD body name as the Part name; prefix ensures
                # uniqueness when multiple assemblies are loaded.
                part_name = f"{base_prefix}_{body_name}" if name_prefix else body_name
                # Make name unique if needed
                unique_name = part_name
                counter = 1
                while any(p.name == unique_name for p in self.parts.values()):
                    unique_name = f"{part_name}_{counter}"
                    counter += 1

                inferred_type = infer_part_type_from_name(body_name)

                part = Part(
                    id=part_id,
                    name=unique_name,
                    filepath=filepath,
                    trimesh_obj=body_mesh,
                    part_type=inferred_type,
                    color=PART_TYPE_COLORS.get(inferred_type, (0.7, 0.75, 0.8)),
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
    
    def remove_part(self, part_id: str) -> bool:
        """Remove a part by ID"""
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
        """Get a part by ID"""
        return self.parts.get(part_id)
    
    def get_part_by_name(self, name: str) -> Optional[Part]:
        """Get a part by name"""
        for part in self.parts.values():
            if part.name == name:
                return part
        return None
    
    def get_all_parts(self) -> List[Part]:
        """Get all parts in order"""
        return [self.parts[pid] for pid in self._part_order if pid in self.parts]
    
    def get_part_count(self) -> int:
        """Get number of parts"""
        return len(self.parts)
    
    def get_parts_with_types(self) -> List[Tuple['Part', int]]:
        """
        Get all visible parts with their type codes for voxel assignment.
        
        Returns:
            List of tuples (transformed_mesh, part_type_int) where:
            - transformed_mesh: The part's transformed trimesh
            - part_type_int: 0=CAST, 1=RISER, 2=CHILL, 3=RISER_POST, 4=CORE
        """
        type_map = {
            PartType.CAST: 0, 
            PartType.RISER: 1, 
            PartType.CHILL: 2,
            PartType.RISER_POST: 3,  # Distinct type for hot sand saturation effect
            PartType.CORE: 4  # Sand core - excluded from defect analysis
        }
        result = []
        
        for part in self.get_all_parts():
            if part.visible:
                mesh = part.get_transformed_mesh()
                type_code = type_map.get(part.part_type, 0)
                result.append((mesh, type_code))
        
        return result
    
    def clear_all(self):
        """Remove all parts"""
        self.parts.clear()
        self._part_order.clear()
        print("Cleared all parts")
    
    def get_combined_mesh(self) -> Optional[trimesh.Trimesh]:
        """
        Combine all visible parts into a single mesh for simulation.
        
        Returns:
            Combined trimesh object, or None if no parts
        """
        visible_parts = [p for p in self.parts.values() if p.visible]
        
        if not visible_parts:
            return None
        
        if len(visible_parts) == 1:
            return visible_parts[0].get_transformed_mesh()
        
        # Combine multiple meshes
        meshes = [p.get_transformed_mesh() for p in visible_parts]
        combined = trimesh.util.concatenate(meshes)
        
        return combined
    
    def get_combined_bounds(self) -> Optional[np.ndarray]:
        """Get bounds encompassing all visible parts"""
        visible_parts = [p for p in self.parts.values() if p.visible]
        
        if not visible_parts:
            return None
        
        # Start with first part's bounds
        all_bounds = [p.get_bounds() for p in visible_parts]
        
        combined = np.array([
            [min(b[0][0] for b in all_bounds), max(b[0][1] for b in all_bounds)],  # X
            [min(b[1][0] for b in all_bounds), max(b[1][1] for b in all_bounds)],  # Y
            [min(b[2][0] for b in all_bounds), max(b[2][1] for b in all_bounds)],  # Z
        ])
        
        return combined
    
    def get_total_volume(self) -> float:
        """Get total volume of all visible parts"""
        return sum(p.get_volume() for p in self.parts.values() if p.visible)
    
    def get_total_surface_area(self) -> float:
        """Get total surface area of all visible parts"""
        return sum(p.get_surface_area() for p in self.parts.values() if p.visible)
    
    def auto_arrange_grid(self, spacing: float = 1.0):
        """
        Automatically arrange parts in a grid pattern.
        
        Args:
            spacing: Space between parts in inches
        """
        parts = self.get_all_parts()
        if not parts:
            return
        
        # Calculate grid size
        n = len(parts)
        cols = int(np.ceil(np.sqrt(n)))
        
        # Get max dimensions from ORIGINAL mesh bounds (not transformed)
        max_x = max(p.get_original_size()[0] for p in parts)
        max_y = max(p.get_original_size()[1] for p in parts)
        
        cell_x = max_x + spacing
        cell_y = max_y + spacing
        
        for i, part in enumerate(parts):
            row = i // cols
            col = i % cols
            
            # Get original mesh bounds for centering
            orig_bounds = part.get_original_bounds()
            part_width = orig_bounds[0][1] - orig_bounds[0][0]
            part_depth = orig_bounds[1][1] - orig_bounds[1][0]
            
            # Position: cell position + centering offset - original mesh offset
            x = col * cell_x + (cell_x - part_width) / 2 - orig_bounds[0][0]
            y = row * cell_y + (cell_y - part_depth) / 2 - orig_bounds[1][0]
            z = -orig_bounds[2][0]  # Place bottom on Z=0 plane
            
            part.set_position(x, y, z)
        
        print(f"Arranged {n} parts in {cols}x{(n + cols - 1) // cols} grid")
    
    def auto_arrange_line(self, axis: str = 'x', spacing: float = 1.0):
        """
        Arrange parts in a line along an axis.
        
        Args:
            axis: 'x', 'y', or 'z'
            spacing: Space between parts
        """
        parts = self.get_all_parts()
        if not parts:
            return
        
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        
        current_pos = 0.0
        
        for part in parts:
            # Use original bounds (not transformed)
            orig_bounds = part.get_original_bounds()
            part_size = orig_bounds[axis_idx][1] - orig_bounds[axis_idx][0]
            
            # Position part
            pos = [0.0, 0.0, 0.0]
            pos[axis_idx] = current_pos - orig_bounds[axis_idx][0]
            
            # Center on other axes (align to origin)
            for i in range(3):
                if i != axis_idx:
                    pos[i] = -orig_bounds[i][0]
            
            part.set_position(*pos)
            
            current_pos += part_size + spacing
        
        print(f"Arranged {len(parts)} parts in line along {axis.upper()}-axis")
    
    def duplicate_part(self, part_id: str, offset: Tuple[float, float, float] = (2.0, 0.0, 0.0)) -> Optional[str]:
        """
        Duplicate an existing part with an offset.
        
        Args:
            part_id: ID of part to duplicate
            offset: Position offset from original
        
        Returns:
            New part ID if successful
        """
        original = self.get_part(part_id)
        if not original:
            return None
        
        # Load new instance
        new_id = self.load_part(original.filepath, name=f"{original.name}_copy")
        if not new_id:
            return None
        
        # Apply same transforms plus offset
        new_part = self.get_part(new_id)
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
        """Check if there are any valid parts loaded"""
        return len(self.parts) > 0
    
    def get_summary(self) -> str:
        """Get summary of all parts"""
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


# For backwards compatibility with existing STLLoader interface
class MultiPartSTLLoader:
    """
    Drop-in replacement for STLLoader that supports multiple parts.
    Maintains backwards compatibility while adding multi-part functionality.
    """
    
    def __init__(self):
        self.part_manager = PartManager()
        self._current_part_id: Optional[str] = None
        
        # Legacy properties (from combined mesh)
        self.mesh = None
        self.trimesh_obj = None
        self.filename = None
        self.bounds = None
        self.volume = None
        self.surface_area = None
        self.center = None
    
    def load_stl(self, filepath: str, units: str = 'inches') -> bool:
        """
        Load STL file (legacy interface).
        Clears existing parts and loads as single part.
        """
        self.part_manager.clear_all()
        part_id = self.part_manager.load_part(filepath)
        
        if part_id:
            self._current_part_id = part_id
            self._update_legacy_properties()
            return True
        return False
    
    def add_part(self, filepath: str, name: Optional[str] = None,
                 position: Optional[Tuple[float, float, float]] = None) -> Optional[str]:
        """
        Add a new part without clearing existing ones.
        
        Returns:
            Part ID if successful
        """
        part_id = self.part_manager.load_part(filepath, name, position)
        if part_id:
            self._update_legacy_properties()
        return part_id
    
    def _update_legacy_properties(self):
        """Update legacy properties from combined mesh"""
        combined = self.part_manager.get_combined_mesh()
        if combined is None:
            return
        
        self.trimesh_obj = combined
        self.bounds = self.part_manager.get_combined_bounds()
        self.volume = self.part_manager.get_total_volume()
        self.surface_area = self.part_manager.get_total_surface_area()
        self.center = combined.center_mass
        
        # Filename shows all parts
        names = [p.name for p in self.part_manager.get_all_parts()]
        self.filename = ", ".join(names) if names else None
    
    def get_dimensions(self) -> str:
        """Get dimensions as formatted string"""
        if self.bounds is None:
            return "N/A"
        
        x_size = self.bounds[0][1] - self.bounds[0][0]
        y_size = self.bounds[1][1] - self.bounds[1][0]
        z_size = self.bounds[2][1] - self.bounds[2][0]
        
        return f"{x_size:.2f}\" x {y_size:.2f}\" x {z_size:.2f}\""
    
    def get_vertices(self) -> np.ndarray:
        """Get combined vertices"""
        if self.trimesh_obj is None:
            return np.array([])
        return self.trimesh_obj.vertices
    
    def get_faces(self) -> np.ndarray:
        """Get combined faces"""
        if self.trimesh_obj is None:
            return np.array([])
        return self.trimesh_obj.faces
    
    def get_bounds_array(self) -> np.ndarray:
        """Get bounds as numpy array"""
        return self.bounds
    
    def is_valid(self) -> bool:
        """Check if any parts are loaded"""
        return self.part_manager.is_valid()
    
    def get_part_manager(self) -> PartManager:
        """Get the underlying part manager"""
        return self.part_manager


_PART_TYPE_KEY_MAP: Dict[str, PartType] = {
    "cast": PartType.CAST,
    "riser": PartType.RISER,
    "riser_post": PartType.RISER_POST,
    "chill": PartType.CHILL,
    "core": PartType.CORE,
}


def build_part_manager(
    parts_data: Sequence[Dict[str, Any]],
    *,
    preserve_origin: bool = True,
    default_visible: bool = True,
    default_is_open_top: bool = False,
) -> PartManager:
    """Build a ``PartManager`` from serialized web API part payloads.

    The ``parts_data`` entries should be dictionaries with at least a
    ``filepath`` field and may carry optional ``name``, ``part_type``,
    ``position_x/y/z``, ``rotation_x/y/z``, ``scale``, ``visible`` and
    ``is_open_top`` overrides. Unknown ``part_type`` values fall back to
    ``PartType.CAST`` unless filename heuristics suggest a riser post.

    Args:
        parts_data: Sequence of part dictionaries (web payload shape).
        preserve_origin: When True, keep each STL's original coordinates so
            assembled parts line up. When False, center each loaded part at
            the origin unless ``position_x/y/z`` overrides are provided.
        default_visible: Default value applied to ``Part.visible`` when the
            payload omits the ``visible`` flag.
        default_is_open_top: Default value applied to ``Part.is_open_top``
            when the payload omits the ``is_open_top`` flag.
    """
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

    return manager
