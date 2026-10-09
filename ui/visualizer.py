"""
3D Visualization using VTK
Translucent ghost model with red shrinkage overlay
Supports multiple parts with individual colors and transforms
Optimized with pre-rendered shrinkage caching for fast threshold updates

Enhanced Features:
- Cross-section/slice views with X/Y/Z cutting planes
- Animation export (MP4/GIF)
- Multiple color map options (viridis, plasma, coolwarm, etc.)
- Legend/color bar with scale
- Measurement tools (distance, volume)
"""

import importlib
import numpy as np
import vtk
from vtk.util import numpy_support
# Qt 6 makes every NSView layer-backed. VTK's default widget base, QWidget,
# hands vtkCocoaRenderWindow a raw native window handle instead -- the Qt 5
# model -- and under Qt 6 that leaves the process spinning inside
# CA::Layer::display_if_needed: the first Core Animation transaction never
# commits, so the window never paints and no Qt timer is ever dispatched. The
# app starts, prints its banner, and shows nothing.
#
# QOpenGLWidget is the base VTK supports for PySide6/PyQt6 (its own
# compatibility check in QVTKRenderWindowInteractor pairs them explicitly); it
# renders through Qt's GL context rather than taking over the view. This has to
# be set before QVTKRenderWindowInteractor is imported, and this module is the
# only importer of it. import_module rather than `import vtkmodules.qt` so it
# also works when the visualizer tests pre-load a stub under that name.
importlib.import_module("vtkmodules.qt").QVTKRWIBase = "QOpenGLWidget"

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from ui.qt import (
    Qt,
    QTimer,
    QVBoxLayout,
    QWidget,
    Signal,
)
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Callable, Set
import time
import os
import tempfile
import sys

# Error logging
try:
    from core.error_logger import logger
except ImportError:
    from core.error_logger_dummy import logger

from core.pour_location import PourLocation
from ui.result_surface import (
    ResultSurfaceRenderer,
    make_environment_texture,
    palette_fill_speed,
    palette_fill_time,
    palette_hot_tear,
    palette_hotspot,
    palette_liquid,
    palette_niyama,
    palette_sand_burn_in,
)

#: Marker color for the picked pour (metal-entry) location -- magenta, chosen
#: to be distinct from every entry in ``core.part_manager.PART_TYPE_COLORS``
#: (grey-blue casting, orange risers, dark blue-grey chills, brown cores, and
#: the teal/yellow gating system) so it reads unambiguously as a UI marker
#: rather than a part.
POUR_LOCATION_MARKER_COLOR = (1.0, 0.0, 1.0)

#: Parting-plane overlay colors: plane fill/outline and label, the section
#: line where it cuts the casting, and the cope (above) / drag (below) tags.
PARTING_PLANE_COLOR = (0.30, 0.75, 1.0)
PARTING_SECTION_COLOR = (1.0, 0.85, 0.1)
PARTING_COPE_COLOR = (1.0, 0.55, 0.35)
PARTING_DRAG_COLOR = (0.45, 0.85, 0.55)

#: Direction metal travels from the pour point. Gravity pour, so straight
#: down -- matches the fixed -Z gravity in physics.filling.gravity_force.
POUR_DIRECTION = (0.0, 0.0, -1.0)

#: Color for a voxel the mold-filling stage never reached (a misrun), in the
#: Fill Time visualization. Every other color in the scene carries a hue --
#: the teal/yellow gating system, orange risers, dark blue-grey chills, the
#: magenta pour marker, and the blue-to-red fill-time ramp itself -- so a
#: neutral, hue-free grey is unmistakable as "this is a void, not a colored
#: material". A near-black grey was considered and rejected: the renderer
#: background is a dark blue-grey (0.1, 0.1, 0.15), and anything much darker
#: than this would nearly vanish against it. Mid-grey reads clearly against
#: that background while staying visually distinct from every part/marker
#: color in the scene.
FILL_TIME_MISRUN_COLOR = (0.55, 0.55, 0.55)

#: Minimum fill_fraction to draw a voxel in the Fill Animation view. The
#: partial-fill free surface (see the FillFrame docstring in
#: core/filling_stage.py) trails a long tail of near-zero numerical noise
#: behind the true front; a bare `> 0` threshold would fog the advancing
#: front with a haze of barely-visible cubes instead of the clean wavefront
#: an industry-standard fill plot shows.
FILL_ANIMATION_WETTED_THRESHOLD = 0.05


# ==================== Color Map Definitions ====================

COLOR_MAPS = {
    'red_gradient': {
        'name': 'Red Gradient (Default)',
        'colors': [(0.0, (50, 0, 0)), (0.5, (200, 50, 0)), (1.0, (255, 100, 50))],
        'description': 'Classic red gradient for shrinkage'
    },
    'viridis': {
        'name': 'Viridis',
        'colors': [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)), 
                   (0.75, (94, 201, 98)), (1.0, (253, 231, 37))],
        'description': 'Perceptually uniform, colorblind-friendly'
    },
    'plasma': {
        'name': 'Plasma',
        'colors': [(0.0, (13, 8, 135)), (0.25, (126, 3, 168)), (0.5, (204, 71, 120)),
                   (0.75, (248, 149, 64)), (1.0, (240, 249, 33))],
        'description': 'Perceptually uniform, high contrast'
    },
    'coolwarm': {
        'name': 'Cool-Warm',
        'colors': [(0.0, (59, 76, 192)), (0.5, (221, 221, 221)), (1.0, (180, 4, 38))],
        'description': 'Blue-white-red diverging'
    },
    'hot_cold': {
        'name': 'Hot-Cold',
        'colors': [(0.0, (0, 0, 255)), (0.25, (0, 255, 255)), (0.5, (0, 255, 0)),
                   (0.75, (255, 255, 0)), (1.0, (255, 0, 0))],
        'description': 'Temperature-like gradient'
    },
    'defect_industry': {
        'name': 'Industry Standard',
        'colors': [(0.0, (0, 100, 0)), (0.3, (255, 255, 0)), (0.6, (255, 165, 0)), (1.0, (255, 0, 0))],
        'description': 'Green-Yellow-Orange-Red defect scale'
    },
    'niyama': {
        'name': 'Niyama Criterion',
        'colors': [(0.0, (0, 0, 255)), (0.33, (0, 255, 255)), (0.66, (255, 255, 0)), (1.0, (255, 0, 0))],
        'description': 'Blue-Cyan-Yellow-Red gradient for Niyama defect risk'
    },
    'feedmod': {
        'name': 'Feed Modulus',
        'colors': [(0.0, (0, 210, 230)), (0.25, (0, 70, 230)), (0.50, (210, 0, 120)),
                   (0.68, (255, 45, 0)), (0.84, (255, 210, 0)), (1.0, (246, 242, 208))],
        'description': 'industry-standard feed modulus scale in inches'
    }
}



def get_color_for_value(value: float, colormap: str = 'red_gradient') -> Tuple[int, int, int]:
    """Get RGB color for a normalized value (0-1) using specified colormap"""
    cmap = COLOR_MAPS.get(colormap, COLOR_MAPS['red_gradient'])
    colors = cmap['colors']
    
    # Find the two colors to interpolate between
    for i in range(len(colors) - 1):
        pos1, color1 = colors[i]
        pos2, color2 = colors[i + 1]
        if pos1 <= value <= pos2:
            # Linear interpolation
            t = (value - pos1) / (pos2 - pos1) if pos2 != pos1 else 0
            r = int(color1[0] + t * (color2[0] - color1[0]))
            g = int(color1[1] + t * (color2[1] - color1[1]))
            b = int(color1[2] + t * (color2[2] - color1[2]))
            return (r, g, b)
    
    # Return last color if value is 1.0
    return colors[-1][1]


def apply_colormap_to_array(intensities: np.ndarray, colormap: str = 'red_gradient',
                            min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """Apply colormap to intensity array, returning Nx3 uint8 color array"""
    intensities_array = np.asarray(intensities, dtype=np.float32)

    # Normalize intensities to 0-1 range based on min/max
    if max_val > min_val:
        normalized = np.clip((intensities_array - min_val) / (max_val - min_val), 0.0, 1.0)
    else:
        normalized = np.zeros_like(intensities_array, dtype=np.float32)

    if normalized.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    cmap = COLOR_MAPS.get(colormap, COLOR_MAPS['red_gradient'])
    positions = np.asarray([stop[0] for stop in cmap['colors']], dtype=np.float32)
    colors = np.asarray([stop[1] for stop in cmap['colors']], dtype=np.float32)

    if len(positions) == 1:
        return np.repeat(colors.astype(np.uint8), normalized.size, axis=0)

    # Vectorized interpolation preserves the existing piecewise linear color mapping.
    indices = np.searchsorted(positions, normalized, side='right') - 1
    indices = np.clip(indices, 0, len(positions) - 2)

    pos1 = positions[indices]
    pos2 = positions[indices + 1]
    color1 = colors[indices]
    color2 = colors[indices + 1]

    denom = np.where(pos2 != pos1, pos2 - pos1, 1.0)
    t = ((normalized - pos1) / denom).reshape(-1, 1)
    interpolated = color1 + t * (color2 - color1)

    return interpolated.astype(np.uint8)


def apply_display_mask(viz: Any, mask: np.ndarray) -> np.ndarray:
    """AND the visualizer's review-time voxel display mask into ``mask``.

    Module-level on purpose: the geometry builders are also exercised as
    unbound functions with mocked or bare ``self`` objects, so the mask is
    read with ``getattr`` and only honoured when it is a real bool array of
    the right length. A stale mask from a previous mesh must never blank the
    view, so a length mismatch also falls through to ``mask`` unchanged.
    """
    display = getattr(viz, "_voxel_display_mask", None)
    if not isinstance(display, np.ndarray) or len(display) != len(mask):
        return mask
    return mask & display


def compute_top_percent_mask(values: np.ndarray,
                             top_fraction: float,
                             base_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return a boolean mask selecting the top-X% values.

    Args:
        values: 1D numeric array.
        top_fraction: Fraction from 0.0-1.0 where 0.2 means "top 20%".
        base_mask: Optional eligibility mask. Percentile is computed only on
            eligible entries and output mask is constrained to this mask.

    Returns:
        Boolean array with same shape as `values`.
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if arr.size == 0:
        return np.zeros_like(arr, dtype=bool)

    if base_mask is None:
        eligible = np.ones(arr.shape, dtype=bool)
    else:
        eligible = np.asarray(base_mask, dtype=bool).reshape(arr.shape)

    finite = np.isfinite(arr)
    eligible = eligible & finite

    if not np.any(eligible):
        return np.zeros_like(arr, dtype=bool)

    if top_fraction <= 0.0:
        return np.zeros_like(arr, dtype=bool)
    if top_fraction >= 1.0:
        return eligible.copy()

    eligible_values = arr[eligible]
    n = eligible_values.size
    keep_count = max(1, int(np.ceil(top_fraction * n)))
    kth_index = n - keep_count
    cutoff = float(np.partition(eligible_values, kth_index)[kth_index])
    return eligible & (arr >= cutoff)


def should_use_depth_peeling(platform: str = sys.platform) -> bool:
    """Depth peeling is unreliable on some Windows OpenGL/VTK driver combinations."""
    return platform != "win32"


class SolidWorksInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """
    Custom VTK interactor style matching SolidWorks mouse controls:
    - Middle Mouse Button (MMB): Rotate
    - Shift + MMB: Pan
    - Scroll Wheel: Zoom
    - Right Mouse Button: Pan (alternative)
    """
    
    def __init__(self):
        super().__init__()
        self.AddObserver("MiddleButtonPressEvent", self.middle_button_press_event)
        self.AddObserver("MiddleButtonReleaseEvent", self.middle_button_release_event)
        self.AddObserver("RightButtonPressEvent", self.right_button_press_event)
        self.AddObserver("RightButtonReleaseEvent", self.right_button_release_event)
        self.AddObserver("MouseWheelForwardEvent", self.mouse_wheel_forward_event)
        self.AddObserver("MouseWheelBackwardEvent", self.mouse_wheel_backward_event)
        
        # Reduce rotation sensitivity for smoother control
        self.SetMotionFactor(5.0)  # Default is 10.0
    
    def middle_button_press_event(self, obj, event):
        """Handle middle mouse button press - Rotate or Pan based on Shift key"""
        if self.GetInteractor().GetShiftKey():
            # Shift + MMB = Pan (like SolidWorks)
            self.OnMiddleButtonDown()
            self.StartPan()
        else:
            # MMB alone = Rotate (like SolidWorks)
            self.OnMiddleButtonDown()
            self.StartRotate()
    
    def middle_button_release_event(self, obj, event):
        """Handle middle mouse button release"""
        self.OnMiddleButtonUp()
        self.EndPan()
        self.EndRotate()
    
    def right_button_press_event(self, obj, event):
        """Handle right mouse button press - Pan (alternative to Shift+MMB)"""
        self.OnRightButtonDown()
        self.StartPan()
    
    def right_button_release_event(self, obj, event):
        """Handle right mouse button release"""
        self.OnRightButtonUp()
        self.EndPan()
    
    def mouse_wheel_forward_event(self, obj, event):
        """Handle mouse wheel forward - Zoom in"""
        self.OnMouseWheelForward()
    
    def mouse_wheel_backward_event(self, obj, event):
        """Handle mouse wheel backward - Zoom out"""
        self.OnMouseWheelBackward()


# ==================== Cross-Section Cutting Plane ====================

class CrossSectionPlane:
    """Manages a cutting plane for cross-section visualization"""
    
    def __init__(self, renderer: vtk.vtkRenderer, axis: str = 'X'):
        """
        Initialize a cutting plane
        
        Args:
            renderer: VTK renderer to add plane to
            axis: 'X', 'Y', or 'Z' axis for the cutting plane
        """
        self.renderer = renderer
        self.axis = axis.upper()
        self.enabled = False
        self.position = 0.0  # Position along the axis
        self.bounds = None  # Model bounds
        self.direction = 1  # +1 or -1: which side of the plane to keep

        # Create implicit plane for clipping
        self.plane = vtk.vtkPlane()
        self._update_plane_orientation()
        
        # Visual plane representation (optional, for seeing the cut plane)
        self.plane_source = vtk.vtkPlaneSource()
        self.plane_mapper = vtk.vtkPolyDataMapper()
        self.plane_actor = vtk.vtkActor()
        
        self._setup_visual_plane()
        
        # Clippers for each actor that should be cut
        self.clippers: Dict[vtk.vtkActor, vtk.vtkClipPolyData] = {}
    
    def _update_plane_orientation(self):
        """Set plane normal based on axis and direction"""
        d = self.direction
        if self.axis == 'X':
            self.plane.SetNormal(d, 0, 0)
        elif self.axis == 'Y':
            self.plane.SetNormal(0, d, 0)
        else:  # Z
            self.plane.SetNormal(0, 0, d)

        self.plane.SetOrigin(0, 0, 0)

    def set_direction(self, direction: int):
        """Set clipping direction (+1 keeps negative side, -1 keeps positive side)"""
        if direction not in (-1, 1):
            return
        self.direction = direction
        self._update_plane_orientation()
        # Re-apply current position so origin updates with new normal
        self.set_position(self.position)
    
    def _setup_visual_plane(self):
        """Create visual representation of the cutting plane"""
        self.plane_mapper.SetInputConnection(self.plane_source.GetOutputPort())
        self.plane_actor.SetMapper(self.plane_mapper)
        
        # Semi-transparent blue plane
        prop = self.plane_actor.GetProperty()
        prop.SetColor(0.2, 0.4, 0.8)
        prop.SetOpacity(0.15)
        prop.SetRepresentationToSurface()
        prop.LightingOff()
        
        self.plane_actor.SetVisibility(False)
    
    def set_bounds(self, bounds: Tuple[float, float, float, float, float, float]):
        """
        Set model bounds for the cutting plane
        
        Args:
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        self.bounds = bounds
        self._update_visual_plane_size()
    
    def _update_visual_plane_size(self):
        """Update the visual plane size based on bounds"""
        if self.bounds is None:
            return
        
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        margin = 0.1  # 10% margin
        
        if self.axis == 'X':
            width = (y_max - y_min) * (1 + margin)
            height = (z_max - z_min) * (1 + margin)
            self.plane_source.SetOrigin(self.position, (y_min + y_max) / 2 - width / 2, 
                                         (z_min + z_max) / 2 - height / 2)
            self.plane_source.SetPoint1(self.position, (y_min + y_max) / 2 + width / 2,
                                         (z_min + z_max) / 2 - height / 2)
            self.plane_source.SetPoint2(self.position, (y_min + y_max) / 2 - width / 2,
                                         (z_min + z_max) / 2 + height / 2)
        elif self.axis == 'Y':
            width = (x_max - x_min) * (1 + margin)
            height = (z_max - z_min) * (1 + margin)
            self.plane_source.SetOrigin((x_min + x_max) / 2 - width / 2, self.position,
                                         (z_min + z_max) / 2 - height / 2)
            self.plane_source.SetPoint1((x_min + x_max) / 2 + width / 2, self.position,
                                         (z_min + z_max) / 2 - height / 2)
            self.plane_source.SetPoint2((x_min + x_max) / 2 - width / 2, self.position,
                                         (z_min + z_max) / 2 + height / 2)
        else:  # Z
            width = (x_max - x_min) * (1 + margin)
            height = (y_max - y_min) * (1 + margin)
            self.plane_source.SetOrigin((x_min + x_max) / 2 - width / 2,
                                         (y_min + y_max) / 2 - height / 2, self.position)
            self.plane_source.SetPoint1((x_min + x_max) / 2 + width / 2,
                                         (y_min + y_max) / 2 - height / 2, self.position)
            self.plane_source.SetPoint2((x_min + x_max) / 2 - width / 2,
                                         (y_min + y_max) / 2 + height / 2, self.position)
        
        self.plane_source.Update()
    
    def set_position(self, position: float):
        """
        Set cutting plane position along its axis
        
        Args:
            position: Position value along the axis
        """
        self.position = position
        
        # Update implicit plane origin
        if self.axis == 'X':
            self.plane.SetOrigin(position, 0, 0)
        elif self.axis == 'Y':
            self.plane.SetOrigin(0, position, 0)
        else:
            self.plane.SetOrigin(0, 0, position)
        
        # Update visual plane
        self._update_visual_plane_size()
    
    def get_position_range(self) -> Tuple[float, float]:
        """Get min/max position values based on bounds"""
        if self.bounds is None:
            return (0, 1)
        
        if self.axis == 'X':
            return (self.bounds[0], self.bounds[1])
        elif self.axis == 'Y':
            return (self.bounds[2], self.bounds[3])
        else:
            return (self.bounds[4], self.bounds[5])
    
    def enable(self, show_plane: bool = True):
        """Enable the cutting plane"""
        self.enabled = True
        if show_plane:
            self.plane_actor.SetVisibility(True)
            if self.plane_actor not in [self.renderer.GetActors().GetItemAsObject(i) 
                                         for i in range(self.renderer.GetActors().GetNumberOfItems())]:
                self.renderer.AddActor(self.plane_actor)
    
    def disable(self):
        """Disable the cutting plane"""
        self.enabled = False
        self.plane_actor.SetVisibility(False)
    
    def get_plane(self) -> vtk.vtkPlane:
        """Get the implicit plane for clipping operations"""
        return self.plane


# ==================== Measurement Tool ====================

class MeasurementTool:
    """Tool for measuring distances and volumes in the visualization"""
    
    def __init__(self, renderer: vtk.vtkRenderer, interactor: vtk.vtkRenderWindowInteractor):
        self.renderer = renderer
        self.interactor = interactor
        self.enabled = False
        self.mode = 'distance'  # 'distance', 'thickness', 'volume'
        
        # Distance measurement
        self.point1 = None
        self.point2 = None
        self.distance_line_actor = None
        self.distance_text_actor = None
        
        # All measurement actors
        self.measurement_actors: List[vtk.vtkActor] = []
        
        # Picker for selecting points - use world picker for better accuracy
        self.picker = vtk.vtkWorldPointPicker()
        
        # Also use a cell picker as backup for more precise picking on surfaces
        self.cell_picker = vtk.vtkCellPicker()
        self.cell_picker.SetTolerance(0.01)  # Increased tolerance for easier clicking
        
        # Callback for point picking
        self._pick_callback = None
        
        # Marker size (will be scaled based on model bounds)
        self._marker_scale = 1.0
    
    def set_marker_scale(self, model_bounds: Tuple[float, float, float, float, float, float]):
        """Set marker scale based on model bounds"""
        if model_bounds:
            # Calculate a reasonable marker size (about 1% of model diagonal)
            dx = model_bounds[1] - model_bounds[0]
            dy = model_bounds[3] - model_bounds[2]
            dz = model_bounds[5] - model_bounds[4]
            diagonal = np.sqrt(dx*dx + dy*dy + dz*dz)
            self._marker_scale = max(0.1, diagonal * 0.01)
    
    def enable_distance_mode(self):
        """Enable distance measurement mode"""
        self.enabled = True
        self.mode = 'distance'
        self.point1 = None
        self.point2 = None
        print("Distance measurement mode: Click two points on the model to measure")
    
    def disable(self):
        """Disable measurement mode"""
        self.enabled = False
        self.point1 = None
        self.point2 = None
    
    def pick_point(self, x: int, y: int) -> Optional[Tuple[float, float, float]]:
        """Pick a point at screen coordinates"""
        # First try cell picker for surface picking (more precise)
        self.cell_picker.Pick(x, y, 0, self.renderer)
        
        if self.cell_picker.GetCellId() >= 0:
            return self.cell_picker.GetPickPosition()
        
        # Fallback to world point picker
        self.picker.Pick(x, y, 0, self.renderer)
        pick_pos = self.picker.GetPickPosition()
        
        # Check if we actually hit something (not just the far plane)
        if pick_pos[0] != 0 or pick_pos[1] != 0 or pick_pos[2] != 0:
            return pick_pos
        
        return None
    
    def add_point(self, point: Tuple[float, float, float]):
        """Add a measurement point"""
        if self.mode == 'distance':
            if self.point1 is None:
                self.point1 = point
                self._add_point_marker(point, (0, 1, 0))  # Green
                print(f"Point 1: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) - Click second point")
            elif self.point2 is None:
                self.point2 = point
                self._add_point_marker(point, (1, 0, 0))  # Red
                self._show_distance_measurement()
                print(f"Point 2: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
                # Reset for next measurement (keep actors visible)
                self.point1 = None
                self.point2 = None
    
    def _add_point_marker(self, point: Tuple[float, float, float], color: Tuple[float, float, float]):
        """Add a sphere marker at a point"""
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*point)
        sphere.SetRadius(self._marker_scale)  # Scaled sphere
        sphere.SetPhiResolution(16)
        sphere.SetThetaResolution(16)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(0.9)  # Slightly transparent
        
        self.renderer.AddActor(actor)
        self.measurement_actors.append(actor)
    
    def _show_distance_measurement(self):
        """Display distance measurement between two points"""
        if self.point1 is None or self.point2 is None:
            return
        
        # Calculate distance
        dx = self.point2[0] - self.point1[0]
        dy = self.point2[1] - self.point1[1]
        dz = self.point2[2] - self.point1[2]
        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Create line between points
        line = vtk.vtkLineSource()
        line.SetPoint1(*self.point1)
        line.SetPoint2(*self.point2)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(line.GetOutputPort())
        
        self.distance_line_actor = vtk.vtkActor()
        self.distance_line_actor.SetMapper(mapper)
        self.distance_line_actor.GetProperty().SetColor(1, 1, 0)  # Yellow
        self.distance_line_actor.GetProperty().SetLineWidth(3)  # Thicker line
        
        self.renderer.AddActor(self.distance_line_actor)
        self.measurement_actors.append(self.distance_line_actor)
        
        # Create text label at midpoint, slightly offset up
        midpoint = ((self.point1[0] + self.point2[0]) / 2,
                    (self.point1[1] + self.point2[1]) / 2,
                    (self.point1[2] + self.point2[2]) / 2 + self._marker_scale * 2)  # Offset up
        
        text = vtk.vtkVectorText()
        text.SetText(f"{distance:.3f}\"")
        
        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(text.GetOutputPort())
        
        # Scale text based on marker scale for visibility
        text_scale = self._marker_scale * 2
        
        self.distance_text_actor = vtk.vtkFollower()
        self.distance_text_actor.SetMapper(text_mapper)
        self.distance_text_actor.SetPosition(*midpoint)
        self.distance_text_actor.SetScale(text_scale, text_scale, text_scale)
        self.distance_text_actor.GetProperty().SetColor(1, 1, 1)
        self.distance_text_actor.SetCamera(self.renderer.GetActiveCamera())
        
        self.renderer.AddActor(self.distance_text_actor)
        self.measurement_actors.append(self.distance_text_actor)
        
        print(f"Distance: {distance:.3f} inches")
        
        # Reset for next measurement
        self.point1 = None
        self.point2 = None
    
    def clear_measurements(self):
        """Clear all measurement markers"""
        for actor in self.measurement_actors:
            self.renderer.RemoveActor(actor)
        self.measurement_actors.clear()
        self.point1 = None
        self.point2 = None
    
    def calculate_defect_volume(self, voxel_centers: np.ndarray, voxel_sizes: np.ndarray,
                                 intensities: np.ndarray, threshold: float,
                                 voxel_volumes: Optional[np.ndarray] = None) -> float:
        """
        Calculate total volume of defect voxels above minimum risk.
        
        Args:
            voxel_centers: Nx3 array of voxel centers
            voxel_sizes: N array of voxel sizes
            intensities: N array of intensity values
            threshold: Minimum risk as 0-1 fraction (0.7 = 70%-100% risk)
            voxel_volumes: N array of per-point volumes (in3). Given on tet
                runs, where voxel_sizes is an edge length, not a cube side.
            
        Returns:
            Total volume in cubic inches
        """
        mask = intensities >= threshold
        if voxel_volumes is not None:
            return float(np.sum(np.asarray(voxel_volumes, dtype=np.float64)[mask]))
        defect_sizes = voxel_sizes[mask]
        total_volume = np.sum(defect_sizes ** 3)  # Volume of cubes
        return total_volume


# ==================== Generic Leader Annotation Tool ====================


@dataclass(frozen=True)
class LeaderCallout:
    """Description of a single leader annotation to display.

    Attributes:
        text: The label rendered alongside the leader.
        anchor_position: World position the leader's marker points to.
            The elbow and label are offset radially from this point using
            the same model-diagonal scaling the feedmod tool always used.
    """

    text: str
    anchor_position: Tuple[float, float, float]


@dataclass
class InspectorLayer:
    """Describes what the click handler should pick on and how to render it.

    The visualizer keeps at most one active layer. Setting ``shrink_inspector``
    to a new layer swaps the active inspector; passing ``None`` clears it.

    Attributes:
        mode: Free-form identifier such as ``"feed_modulus"`` or ``"shrinkage"``.
        pick_actors: VTK actors the cell picker is restricted to (the result
            envelope, the painted skins and the section caps).
        voxel_at: Maps the picked world position to a global voxel index
            (``-1`` when the position lies outside the result lattice).
        resolver: Callable invoked with the picked voxel index. Returns a
            ``LeaderCallout`` describing the label text and anchor position
            to render. Returning ``None`` clears the existing leader.
    """

    mode: str
    pick_actors: List[vtk.vtkActor]
    voxel_at: Callable[[Tuple[float, float, float]], int]
    resolver: Callable[[int], Optional[LeaderCallout]]


class FeedModulusLeaderTool:
    """L-shaped leader with a sphere marker and camera-facing text flag.

    Now generalized so the same geometry can host arbitrary ``text`` strings
    while preserving the original feedmod rendering. New inspectors should
    call :meth:`show_leader` directly with their own label.
    """

    def __init__(self, renderer: vtk.vtkRenderer):
        self.renderer = renderer
        self._leader_actors: List[vtk.vtkActor] = []
        self._text_actor: Optional[vtk.vtkFollower] = None
        self._cone_actor: Optional[vtk.vtkActor] = None

    def clear(self):
        """Remove all leader actors from the renderer"""
        for actor in self._leader_actors:
            self.renderer.RemoveActor(actor)
        self._leader_actors.clear()
        if self._text_actor:
            self.renderer.RemoveActor(self._text_actor)
            self._text_actor = None
        if self._cone_actor:
            self.renderer.RemoveActor(self._cone_actor)
            self._cone_actor = None

    def is_empty(self) -> bool:
        """Return True if no leader is currently displayed"""
        return len(self._leader_actors) == 0 and self._text_actor is None

    def show_leader(self, world_pos: Tuple[float, float, float], value: float,
                    model_bounds: Optional[Tuple[float, float, float, float, float, float]] = None):
        """Backwards-compatible feedmod entry point.

        Formats ``"FM: {value:.2f} in"`` and delegates to
        :meth:`show_callout` so the call site stays unchanged.
        """
        self.show_callout(
            LeaderCallout(
                text=f"FM: {value:.2f} in",
                anchor_position=tuple(world_pos),
            ),
            model_bounds,
        )

    def show_callout(self, callout: LeaderCallout,
                     model_bounds: Optional[Tuple[float, float, float, float, float, float]] = None):
        """Display an L-shaped leader with the supplied callout text.

        Args:
            callout: ``LeaderCallout`` describing the text and anchor position.
            model_bounds: Optional ``(xmin, xmax, ymin, ymax, zmin, zmax)``
                model bounds used to size the leader.
        """
        self.clear()

        world_pos = tuple(callout.anchor_position)
        text_value = callout.text

        diagonal = 1.0
        if model_bounds:
            dx = model_bounds[1] - model_bounds[0]
            dy = model_bounds[3] - model_bounds[2]
            dz = model_bounds[5] - model_bounds[4]
            diagonal = np.sqrt(dx*dx + dy*dy + dz*dz)

        offset = diagonal * 0.12
        leader_offset = diagonal * 0.22

        cx = (model_bounds[0] + model_bounds[1]) / 2 if model_bounds else 0.0
        cy = (model_bounds[2] + model_bounds[3]) / 2 if model_bounds else 0.0
        cz = (model_bounds[4] + model_bounds[5]) / 2 if model_bounds else 0.0

        direction = np.array(world_pos) - np.array([cx, cy, cz])
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        else:
            direction = np.array([1.0, 0.0, 0.0])

        camera = self.renderer.GetActiveCamera()
        view_up = camera.GetViewUp()
        view_up_arr = np.array(view_up)

        perp = np.cross(direction, view_up_arr)
        norm_perp = np.linalg.norm(perp)
        if norm_perp > 1e-6:
            perp = perp / norm_perp
        else:
            perp = np.array([0.0, 0.0, 1.0])

        elbow = np.array(world_pos) + direction * leader_offset
        text_base = elbow + perp * leader_offset * 1.2

        elbow_tuple = tuple(elbow)
        text_tuple = tuple(text_base)
        world_tuple = world_pos

        line_pts = vtk.vtkPoints()
        line_pts.InsertNextPoint(*world_tuple)
        line_pts.InsertNextPoint(*elbow_tuple)
        line_pts.InsertNextPoint(*text_tuple)

        line_cell = vtk.vtkPolyLine()
        line_cell.GetPointIds().SetNumberOfIds(3)
        line_cell.GetPointIds().SetId(0, 0)
        line_cell.GetPointIds().SetId(1, 1)
        line_cell.GetPointIds().SetId(2, 2)

        line_cells = vtk.vtkCellArray()
        line_cells.InsertNextCell(line_cell)

        line_pd = vtk.vtkPolyData()
        line_pd.SetPoints(line_pts)
        line_pd.SetLines(line_cells)

        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputData(line_pd)

        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        line_actor.GetProperty().SetLineWidth(2.0)

        self.renderer.AddActor(line_actor)
        self._leader_actors.append(line_actor)

        sphere_src = vtk.vtkSphereSource()
        sphere_src.SetRadius(offset * 0.15)
        sphere_src.SetThetaResolution(24)
        sphere_src.SetPhiResolution(16)

        sphere_transform = vtk.vtkTransform()
        sphere_transform.Translate(*world_tuple)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(sphere_src.GetOutputPort())
        transform_filter.SetTransform(sphere_transform)
        transform_filter.Update()

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(transform_filter.GetOutputPort())

        self._cone_actor = vtk.vtkActor()
        self._cone_actor.SetMapper(sphere_mapper)
        self._cone_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        self.renderer.AddActor(self._cone_actor)

        text = vtk.vtkVectorText()
        text.SetText(text_value)

        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(text.GetOutputPort())

        text_scale = offset * 0.2
        text_padding = offset * 0.3
        text_pos = text_base + perp * text_padding
        self._text_actor = vtk.vtkFollower()
        self._text_actor.SetMapper(text_mapper)
        self._text_actor.SetPosition(*text_pos)
        self._text_actor.SetScale(text_scale, text_scale, text_scale)
        self._text_actor.GetProperty().SetColor(1.0, 1.0, 0.8)
        self._text_actor.GetProperty().SetOpacity(0.95)
        self._text_actor.SetCamera(camera)

        self.renderer.AddActor(self._text_actor)


class CastingVisualizer(QWidget):
    """
    VTK-based 3D visualization for casting simulation
    - Translucent ghost model of the part
    - Red shrinkage/porosity regions inside the model
    - Supports multiple parts with individual actors
    
    Enhanced Features:
    - Cross-section views (X/Y/Z cutting planes)
    - Color bar / legend
    - Multiple colormaps
    - Animation export
    - Measurement tools
    """
    
    # Predefined colors for multiple parts (RGB tuples)
    PART_COLORS = [
        (0.7, 0.75, 0.8),   # Default grey-blue
        (0.8, 0.6, 0.6),    # Salmon
        (0.6, 0.8, 0.6),    # Light green
        (0.6, 0.6, 0.8),    # Light blue
        (0.8, 0.8, 0.6),    # Light yellow
        (0.8, 0.6, 0.8),    # Light purple
        (0.6, 0.8, 0.8),    # Cyan
        (0.9, 0.7, 0.5),    # Tan
    ]

    # Emitted with the new PourLocation when a pick lands, or with None when
    # the pour location is cleared. main_window listens to keep its label
    # and the "Mold Filling" analysis gate in sync.
    pour_location_changed = Signal(object)

    # Orient & Parting Line surface picks. ``surface_picked`` carries
    # {"point", "triangle", "toward_viewer", "part_id"} in world inches;
    # ``surface_pick_cancelled`` fires when Esc ends the pick in the viewport.
    surface_picked = Signal(object)
    surface_pick_cancelled = Signal()

    # Class-level defaults so an instance built without __init__ (the
    # visualizer tests use __new__) never enters surface pick mode.
    _surface_pick_enabled = False
    _surface_pick_part_ids: Optional[List[str]] = None

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # VTK components
        self.vtk_widget = None
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.orientation_widget = None
        
        # Single mesh actor (legacy)
        self.mesh_actor = None
        
        # Multiple part actors
        self.part_actors: Dict[str, vtk.vtkActor] = {}

        # Review-time component hiding. ``_hidden_part_ids`` are parts whose
        # skin actor is switched off; ``_voxel_display_mask`` (None or an
        # N-bool array, True = drawable) is ANDed into every result geometry
        # builder so a hidden component's voxels drop out of the analysis
        # overlays too. ``_model_visible`` is the master "Show Model" flag.
        self._hidden_part_ids: set = set()
        self._voxel_display_mask: Optional[np.ndarray] = None
        self._model_visible: bool = True
        
        # Result overlays: painted skin, threshold envelope and section caps.
        # Built in _init_ui once the renderer exists.
        self.result_surface = None
        self._result_lattice_key = None
        # Lumped node volumes (in3) of a registered tet mesh; defect volume
        # sums these, because a node's voxel_sizes is an edge length.
        self._result_point_volumes: Optional[np.ndarray] = None

        # Data the main window reads back (mode -> (centers, sizes, values))
        self.current_mesh = None
        self.current_shrinkage_data = None
        self.current_hotspot_data = None
        self.current_niyama_data = None
        self.current_hot_tear_data = None
        self.current_sand_burn_in_data = None
        self.current_gas_porosity_data = None
        self.current_fraction_liquid_data = None
        self.current_fill_time_data = None
        self.current_fill_animation_data = None

        # ==================== NEW FEATURES ====================
        
        # Color mapping
        self._current_colormap = 'red_gradient'
        self._color_scale_min = 0.0
        self._color_scale_max = 1.0
        
        # Color bar / Legend
        self.scalar_bar_actor = None
        self._legend_title = "Intensity"
        self._material_name = ""
        self._simulation_time = 0.0
        
        # Cross-section planes
        self.cross_section_planes: Dict[str, CrossSectionPlane] = {}
        self._cross_section_enabled = False

        # Measurement tool
        self.measurement_tool = None

        # Feed modulus leader tool
        self.feed_modulus_leader = None
        self._feed_modulus_leader_enabled = False

        # Pour (metal-entry) location picking -- industry-standard ingate
        # placement. See the "Pour Location Picking" section below.
        self._pour_pick_enabled = False
        self._pour_location: Optional[PourLocation] = None
        self._pour_location_actors: List[vtk.vtkActor] = []

        # Orient & Parting Line: one-shot surface pick and the parting plane.
        # ``_surface_pick_part_ids`` None = any part actor may be picked.
        self._surface_pick_enabled = False
        self._surface_pick_part_ids: Optional[List[str]] = None
        self._parting_actors: List[Any] = []

        # Active result-mode inspector (feedmod, shrink/porosity, ...)
        self._inspector_layer: Optional[InspectorLayer] = None

        # Animation export
        self._animation_frames: List[np.ndarray] = []
        self._is_recording = False
        self._interactor_initialized = False

        # Projection mode (orthographic by default)
        self._is_orthographic = True
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize VTK widget"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)
        
        self.setLayout(layout)
        
        # Setup VTK rendering
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.04, 0.05, 0.07)  # web app --bg-0
        
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self._setup_lighting_and_passes()

        # Result overlays (painted skin, envelope, section caps).
        self.result_surface = ResultSurfaceRenderer(self.renderer)
        
        self.interactor = self.render_window.GetInteractor()
        
        # Setup camera
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(200, 200, 200)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)

        # Default to orthographic projection
        camera.ParallelProjectionOn()
        camera.SetParallelScale(120.0)

        # Store bounds for camera positioning
        self.model_bounds = None
        
        # Add axes
        self._add_axes()
        
        # Set custom SolidWorks-style interactor
        style = SolidWorksInteractorStyle()
        self.interactor.SetInteractorStyle(style)
        
        # Initialize cross-section planes
        self._init_cross_section_planes()
        
        # Initialize measurement tool
        self.measurement_tool = MeasurementTool(self.renderer, self.interactor)

        # Initialize feed modulus leader tool
        self.feed_modulus_leader = FeedModulusLeaderTool(self.renderer)

        # Add click handler for measurement tool
        self.interactor.AddObserver("LeftButtonPressEvent", self._on_left_button_press, 1.0)
        self.interactor.AddObserver("KeyPressEvent", self._on_key_press, 1.0)

        # Initialize color bar (hidden by default)
        self._init_color_bar()
        
        QTimer.singleShot(0, self._initialize_interactor)

    def _initialize_interactor(self):
        """Initialize the VTK interactor under Qt's event loop."""
        if self._interactor_initialized or self.interactor is None:
            return

        try:
            self.interactor.Initialize()
            self._interactor_initialized = True
        except Exception as exc:
            logger.exception(f"Failed to initialize VTK interactor: {exc}")

    def showEvent(self, event):
        super().showEvent(event)
        self._initialize_interactor()
    
    def _on_left_button_press(self, obj, event):
        """Handle left mouse button press for measurement/pour-location picking"""
        # Orient & Parting Line picks (face down/up, parting height) are
        # one-shot modes the user just started, so they come first.
        if self._surface_pick_enabled:
            click_pos = self.interactor.GetEventPosition()
            self._handle_surface_pick(click_pos[0], click_pos[1])
            return

        # Pour-location picking takes priority: it is a deliberate,
        # one-shot placement the user just started via "Set Pour Location".
        if self._pour_pick_enabled:
            click_pos = self.interactor.GetEventPosition()
            x, y = click_pos[0], click_pos[1]
            self._handle_pour_pick(x, y)
            return

        # Only handle if measurement mode is enabled
        if self.measurement_tool and self.measurement_tool.enabled:
            # Get click position
            click_pos = self.interactor.GetEventPosition()
            x, y = click_pos[0], click_pos[1]

            # Pick point on the model
            point = self.measurement_tool.pick_point(x, y)
            if point:
                self.measurement_tool.add_point(point)
                self.render_window.Render()
            return

        # Active result-mode inspector (feedmod, shrink/porosity, ...).
        # The layer's resolver decides how to format the label, so feedmod
        # keeps its old "FM: X.XX in" text while shrink/porosity can
        # extract a connected pocket and report its volume in in^3.
        layer = self._inspector_layer
        if layer is not None and layer.pick_actors:
            click_pos = self.interactor.GetEventPosition()
            x, y = click_pos[0], click_pos[1]

            picker = vtk.vtkCellPicker()
            for actor in layer.pick_actors:
                picker.AddPickList(actor)
            picker.SetPickFromList(True)
            picker.SetTolerance(0.005)
            picker.Pick(x, y, 0, self.renderer)

            try:
                voxel_index = -1
                if picker.GetCellId() >= 0:
                    voxel_index = int(layer.voxel_at(tuple(picker.GetPickPosition())))
                if voxel_index < 0:
                    if self.feed_modulus_leader is not None:
                        self.feed_modulus_leader.clear()
                    self.render_window.Render()
                    try:
                        self.interactor.GetInteractorStyle().SetAbortFlag(1)
                    except AttributeError:
                        pass
                    return

                callout = layer.resolver(voxel_index)
                if callout is None:
                    if self.feed_modulus_leader is not None:
                        self.feed_modulus_leader.clear()
                else:
                    if self.feed_modulus_leader is not None:
                        self.feed_modulus_leader.show_callout(callout, self.model_bounds)
                self.render_window.Render()
                try:
                    self.interactor.GetInteractorStyle().SetAbortFlag(1)
                except AttributeError:
                    pass
            except Exception as exc:
                logger.exception(f"Inspector click failed: {exc}")
                if self.feed_modulus_leader is not None:
                    self.feed_modulus_leader.clear()
                self.render_window.Render()
    
    def _setup_lighting_and_passes(self):
        """PBR image-based lighting plus SSAO / FXAA, the web viewport's look.

        The environment is generated (``make_environment_texture``), so no
        HDR asset is needed. Depth peeling stays behind the platform guard
        that already existed for the translucent parts. Any failure here
        leaves the plain-lit renderer in place rather than taking the view
        down; set ``CASTSIM_UI_PBR=0`` to skip it entirely.
        """
        if os.environ.get("CASTSIM_UI_PBR", "1") == "0":
            return
        try:
            self.render_window.SetMultiSamples(0)  # FXAA does the anti-aliasing
            env = make_environment_texture()
            self.renderer.UseImageBasedLightingOn()
            self.renderer.SetEnvironmentTexture(env, True)
            self.renderer.UseSphericalHarmonicsOff()
            if should_use_depth_peeling():
                self.renderer.SetUseDepthPeeling(1)
                self.renderer.SetMaximumNumberOfPeels(6)
                self.renderer.SetOcclusionRatio(0.05)
            self.renderer.SetUseSSAO(True)
            self.renderer.SSAOBlurOn()
            self.renderer.SetSSAOKernelSize(48)
            self.renderer.UseFXAAOn()
            self._pbr_enabled = True
        except Exception as exc:  # pragma: no cover - driver dependent
            logger.warning("PBR render setup failed; using plain lighting", context={"error": str(exc)})
            self._pbr_enabled = False

    def _configure_ssao(self):
        """SSAO radius and bias scale with the model, so re-derive them on load."""
        if not getattr(self, "_pbr_enabled", False) or not self.model_bounds:
            return
        b = self.model_bounds
        diag = float(np.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)) or 1.0
        self.renderer.SetSSAORadius(0.05 * diag)
        self.renderer.SetSSAOBias(0.001 * diag)

    def _pbr_surface(self, prop, metallic: float = 0.45, roughness: float = 0.5):
        """Apply the shared surface material to an actor property."""
        if getattr(self, "_pbr_enabled", False):
            prop.SetInterpolationToPBR()
            prop.SetMetallic(metallic)
            prop.SetRoughness(roughness)
        else:
            prop.SetInterpolationToPhong()
            prop.SetSpecular(0.3)
            prop.SetSpecularPower(20)

    def _init_cross_section_planes(self):
        """Initialize cutting planes for X, Y, Z axes"""
        for axis in ['X', 'Y', 'Z']:
            self.cross_section_planes[axis] = CrossSectionPlane(self.renderer, axis)
    
    def _init_color_bar(self):
        """Initialize the scalar bar (color bar / legend)"""
        # Create lookup table for color mapping
        self.lut = vtk.vtkLookupTable()
        self._update_lookup_table()
        
        # Create scalar bar actor
        self.scalar_bar_actor = vtk.vtkScalarBarActor()
        self.scalar_bar_actor.SetLookupTable(self.lut)
        self.scalar_bar_actor.SetTitle("Intensity")
        self.scalar_bar_actor.SetNumberOfLabels(5)
        self.scalar_bar_actor.SetOrientationToVertical()
        
        # Position and size - wider bar with better proportions
        # Positioned in TOP RIGHT corner to avoid XYZ axes widget
        self.scalar_bar_actor.SetPosition(0.85, 0.40)  # Top right
        self.scalar_bar_actor.SetWidth(0.10)  # Wider for better readability
        self.scalar_bar_actor.SetHeight(0.50)  # Slightly shorter to fit in top area
        
        # Critical: Set proper ratio between bar width and total width
        # This controls how much space is allocated for labels vs the color bar
        self.scalar_bar_actor.SetBarRatio(0.35)  # 35% of width is the color bar
        
        # Text properties for title - positioned ABOVE the bar with padding
        title_prop = self.scalar_bar_actor.GetTitleTextProperty()
        title_prop.SetFontSize(14)  # Larger font for title
        title_prop.SetColor(1, 1, 1)
        title_prop.BoldOn()
        title_prop.ItalicOff()
        title_prop.ShadowOn()  # Add shadow for better contrast
        title_prop.SetShadowOffset(1, 1)
        title_prop.SetJustificationToCentered()
        title_prop.SetVerticalJustificationToTop()
        title_prop.SetFontFamilyToArial()
        
        # Label properties - clean and legible
        label_prop = self.scalar_bar_actor.GetLabelTextProperty()
        label_prop.SetFontSize(12)  # Slightly larger for readability
        label_prop.SetColor(0.9, 0.9, 0.9)  # Brighter for contrast
        label_prop.BoldOff()
        label_prop.ItalicOff()
        label_prop.ShadowOn()  # Add subtle shadow
        label_prop.SetShadowOffset(1, 1)
        label_prop.SetFontFamilyToArial()
        
        # Use 5 labels for better scale reference: Min, 25%, Mid, 75%, Max
        self.scalar_bar_actor.SetNumberOfLabels(5)
        self.scalar_bar_actor.SetLabelFormat("%.2f")
        
        # Add spacing between title and bar to prevent overlap
        self.scalar_bar_actor.SetTitleRatio(0.15)  # 15% of height for title
        
        # Use text positioning that prevents overlap
        self.scalar_bar_actor.SetTextPositionToPrecedeScalarBar()
        
        # Enable subtle background for better visibility against any scene
        self.scalar_bar_actor.DrawBackgroundOn()
        self.scalar_bar_actor.GetBackgroundProperty().SetColor(0.1, 0.1, 0.15)
        self.scalar_bar_actor.GetBackgroundProperty().SetOpacity(0.7)
        
        # Add subtle frame
        self.scalar_bar_actor.DrawFrameOn()
        self.scalar_bar_actor.GetFrameProperty().SetColor(0.3, 0.3, 0.35)
        
        # Ensure annotations don't clip
        self.scalar_bar_actor.UnconstrainedFontSizeOn()
        
        # Initially hidden
        self.scalar_bar_actor.SetVisibility(False)
        self.renderer.AddActor2D(self.scalar_bar_actor)
        
        # Create text actor for material/time info
        self.info_text_actor = vtk.vtkTextActor()
        self.info_text_actor.SetInput("")
        self.info_text_actor.GetTextProperty().SetFontSize(14)
        self.info_text_actor.GetTextProperty().SetColor(1, 1, 1)
        self.info_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.info_text_actor.SetPosition(0.02, 0.95)
        self.renderer.AddActor2D(self.info_text_actor)
        
    def update_legend(self, title: str, min_val: float, max_val: float, units: str = "", render: bool = True):
        """
        Update the legend title and scale
        
        Args:
            title: Legend title (e.g., "Shrinkage/Porosity Risk")
            min_val: Minimum value for scale
            max_val: Maximum value for scale
            units: Units string (e.g., "MPa", "μm")
            render: Whether to trigger a render after updating. Callers that
                already render (e.g. per-tick playback) pass False to avoid a
                second full scene render.
        """
        self._legend_title = title
        
        # Validate and fix range order for VTK (requires min <= max)
        if min_val > max_val:
            # Swap values to ensure valid range for VTK
            min_val, max_val = max_val, min_val
        
        # Handle edge case where min == max (avoid zero-range lookups)
        if min_val == max_val:
            max_val = min_val + 1.0
        
        self._color_scale_min = min_val
        self._color_scale_max = max_val
        
        # Update scalar bar title
        full_title = f"{title}\n({units})" if units else title
        self.scalar_bar_actor.SetTitle(full_title)
        
        # Update lookup table range
        self.lut.SetRange(min_val, max_val)
        self.scalar_bar_actor.SetVisibility(True)
        
        if render:
            self.render_window.Render()
        
    def update_info_text(self, material: str, time_s: float):
        """Update simulation info text overlay"""
        self._material_name = material
        self._simulation_time = time_s
        
        text = f"Material: {material}\nTime: {time_s:.1f}s"
        self.info_text_actor.SetInput(text)
        self.render_window.Render()
    
    def _update_lookup_table(self):
        """Update lookup table based on current colormap"""
        cmap = COLOR_MAPS.get(self._current_colormap, COLOR_MAPS['red_gradient'])
        colors = cmap['colors']
        
        self.lut.SetNumberOfTableValues(256)
        self.lut.SetRange(self._color_scale_min, self._color_scale_max)
        
        for i in range(256):
            t = i / 255.0
            r, g, b = get_color_for_value(t, self._current_colormap)
            self.lut.SetTableValue(i, r/255.0, g/255.0, b/255.0, 1.0)
        
        self.lut.Build()
    
    # ==================== Color Map Methods ====================
    
    def set_colormap(self, colormap: str):
        """
        Set the current colormap
        
        Args:
            colormap: Name of colormap (see COLOR_MAPS keys)
        """
        if colormap in COLOR_MAPS:
            self._current_colormap = colormap
            self._update_lookup_table()
            # Refresh visualization if we have data
            self._refresh_current_visualization()
            self.render_window.Render()
    
    def get_available_colormaps(self) -> List[str]:
        """Get list of available colormap names"""
        return list(COLOR_MAPS.keys())
    
    def get_colormap_info(self, colormap: str) -> Dict:
        """Get info about a colormap"""
        return COLOR_MAPS.get(colormap, {})
    
    def set_color_scale_bounds(self, min_val: float, max_val: float):
        """
        Set custom color scale bounds
        
        Args:
            min_val: Minimum value for color scale
            max_val: Maximum value for color scale
        """
        self._color_scale_min = min_val
        self._color_scale_max = max_val
        self._update_lookup_table()
        self.scalar_bar_actor.SetLookupTable(self.lut)
        self._refresh_current_visualization()
        self.render_window.Render()
    
    # ==================== Color Bar / Legend Methods ====================
    
    def show_color_bar(self, title: str = None, show: bool = True):
        """
        Show or hide the color bar legend
        
        Args:
            title: Title for the color bar
            show: Whether to show (True) or hide (False)
        """
        if title:
            self.scalar_bar_actor.SetTitle(title)
        self.scalar_bar_actor.SetVisibility(show)
        self.render_window.Render()
    
    def set_legend_info(self, material_name: str = "", simulation_time: float = 0.0):
        """
        Set additional info to display with the legend
        
        Args:
            material_name: Name of the material/alloy
            simulation_time: Simulation time in seconds
        """
        self._material_name = material_name
        self._simulation_time = simulation_time
        
        # Update title with additional info
        title_parts = [self._legend_title]
        if material_name:
            title_parts.append(f"\n{material_name}")
        if simulation_time > 0:
            title_parts.append(f"\n{simulation_time:.1f}s")
        
        self.scalar_bar_actor.SetTitle("".join(title_parts))
        self.render_window.Render()
    
    # ==================== Cross-Section Methods ====================
    
    def enable_cross_section(self, axis: str, position: float = None, show_plane: bool = False, direction: int = 1):
        """
        Enable a cross-section cutting plane

        Args:
            axis: 'X', 'Y', or 'Z'
            position: Position along the axis (None = middle)
            show_plane: Whether to show the visual plane indicator
                (default False; the plane visual is shown only while the
                user is actively adjusting the slider)
            direction: +1 (default) or -1 to flip which side is kept
        """
        axis = axis.upper()
        if axis not in self.cross_section_planes:
            return

        plane = self.cross_section_planes[axis]

        # Update bounds if we have model bounds
        if self.model_bounds:
            plane.set_bounds(self.model_bounds)

        # Set direction before position so orientation is correct
        plane.set_direction(direction)

        # Set position (default to middle)
        if position is None:
            min_pos, max_pos = plane.get_position_range()
            position = (min_pos + max_pos) / 2

        plane.set_position(position)
        plane.enable(show_plane)

        self._cross_section_enabled = True
        self._apply_cross_section_clipping()
        self.render_window.Render()

    def hide_cross_section_planes_visuals(self):
        """Hide the blue visual plane actors while keeping clipping active"""
        for plane in self.cross_section_planes.values():
            if plane.enabled:
                plane.plane_actor.SetVisibility(False)
        self.render_window.Render()

    def show_cross_section_plane_visual(self, axis: str):
        """Show the blue plane visual for a single enabled axis"""
        axis = axis.upper()
        if axis not in self.cross_section_planes:
            return
        plane = self.cross_section_planes[axis]
        if plane.enabled:
            plane.plane_actor.SetVisibility(True)
            self.render_window.Render()

    def hide_cross_section_plane_visual(self, axis: str):
        """Hide the blue plane visual for a single enabled axis"""
        axis = axis.upper()
        if axis not in self.cross_section_planes:
            return
        plane = self.cross_section_planes[axis]
        if plane.enabled:
            plane.plane_actor.SetVisibility(False)
            self.render_window.Render()
    
    def disable_cross_section(self, axis: str = None):
        """
        Disable cross-section cutting plane(s)
        
        Args:
            axis: 'X', 'Y', 'Z', or None to disable all
        """
        if axis:
            if axis.upper() in self.cross_section_planes:
                self.cross_section_planes[axis.upper()].disable()
        else:
            for plane in self.cross_section_planes.values():
                plane.disable()
        
        self._cross_section_enabled = any(p.enabled for p in self.cross_section_planes.values())
        self._remove_cross_section_clipping()
        self.render_window.Render()
    
    def set_cross_section_position(self, axis: str, position: float):
        """
        Update cross-section plane position
        
        Args:
            axis: 'X', 'Y', or 'Z'
            position: Position along the axis
        """
        axis = axis.upper()
        if axis in self.cross_section_planes:
            self.cross_section_planes[axis].set_position(position)
            self._apply_cross_section_clipping()
            self.render_window.Render()
    
    def get_cross_section_range(self, axis: str) -> Tuple[float, float]:
        """Get the valid position range for a cross-section axis"""
        axis = axis.upper()
        if axis in self.cross_section_planes:
            return self.cross_section_planes[axis].get_position_range()
        return (0, 1)
    
    def _apply_cross_section_clipping(self):
        """Apply clipping to all actors using enabled cutting planes"""
        enabled_planes = [p for p in self.cross_section_planes.values() if p.enabled]

        if self.result_surface is not None:
            # Caps are cut from the volume on each plane; the envelope and the
            # painted skins are clipped like everything else.
            self.result_surface.set_section_planes([p.get_plane() for p in enabled_planes])
            if enabled_planes:
                for actor in self.result_surface.actors_for_clipping():
                    self._clip_actor(actor, enabled_planes)

        if not enabled_planes:
            return

        # Apply to mesh actor
        if self.mesh_actor:
            self._clip_actor(self.mesh_actor, enabled_planes)

        # Apply to part actors
        for actor in self.part_actors.values():
            self._clip_actor(actor, enabled_planes)

    def _clip_actor(self, actor: vtk.vtkActor, planes: List[CrossSectionPlane]):
        """Apply clipping planes to an actor"""
        if not planes or not actor:
            return
        
        mapper = actor.GetMapper()
        if mapper is None:
            return
        
        # Create clip planes collection
        clip_planes = vtk.vtkPlaneCollection()
        for plane in planes:
            clip_planes.AddItem(plane.get_plane())
        
        mapper.SetClippingPlanes(clip_planes)
    
    def _remove_cross_section_clipping(self):
        """Remove all clipping from actors"""
        actors = [self.mesh_actor]
        actors.extend(self.part_actors.values())
        if self.result_surface is not None:
            actors.extend(self.result_surface.actors_for_clipping())
            self.result_surface.set_section_planes([])
        
        for actor in actors:
            if actor and actor.GetMapper():
                actor.GetMapper().RemoveAllClippingPlanes()
    
    # ==================== Measurement Methods ====================
    
    def enable_measurement_mode(self, mode: str = 'distance'):
        """
        Enable measurement mode
        
        Args:
            mode: 'distance' for point-to-point measurement
        """
        if self.measurement_tool:
            # Set marker scale based on model bounds
            if self.model_bounds:
                self.measurement_tool.set_marker_scale(self.model_bounds)
            if mode == 'distance':
                self.measurement_tool.enable_distance_mode()
    
    def disable_measurement_mode(self):
        """Disable measurement mode"""
        if self.measurement_tool:
            self.measurement_tool.disable()
    
    def clear_measurements(self):
        """Clear all measurement markers"""
        if self.measurement_tool:
            self.measurement_tool.clear_measurements()
            self.render_window.Render()

    # ==================== Pour Location Picking ====================
    #
    # industry-standard ingate placement: the user clicks a point on the 3D
    # assembly and that point becomes the metal-entry location the
    # mold-filling stage uses as its inlet. Reuses the same cell-picker /
    # world-picker fallback pattern as MeasurementTool.pick_point above,
    # rather than adding a second picking stack.

    def enable_pour_pick_mode(self) -> None:
        """Enter pour-location pick mode; the next left click sets it."""
        self._pour_pick_enabled = True

    def disable_pour_pick_mode(self) -> None:
        """Leave pour-location pick mode without changing any stored location."""
        self._pour_pick_enabled = False

    def is_pour_pick_mode(self) -> bool:
        """Return True while a click is expected to set the pour location."""
        return self._pour_pick_enabled

    @property
    def pour_location(self) -> Optional[PourLocation]:
        """The currently stored pour location, or ``None`` if unset."""
        return self._pour_location

    def set_pour_location(self, location: PourLocation) -> None:
        """Set the pour location directly (e.g. restoring a saved project).

        Draws the marker and emits :attr:`pour_location_changed` the same
        way a click-driven pick does, so callers only need one code path
        to react to "the pour location changed".
        """
        self._pour_pick_enabled = False
        self._pour_location = location
        self._add_pour_location_marker(location)
        if self.render_window is not None:
            self.render_window.Render()
        self.pour_location_changed.emit(location)

    def clear_pour_location(self) -> None:
        """Clear the stored pour location and remove its marker."""
        self._pour_pick_enabled = False
        self._pour_location = None
        self._clear_pour_location_marker()
        if self.render_window is not None:
            self.render_window.Render()
        self.pour_location_changed.emit(None)

    def _part_id_for_actor(self, actor) -> Optional[str]:
        """Reverse-lookup the part id owning a picked actor, if any."""
        if actor is None:
            return None
        for part_id, part_actor in self.part_actors.items():
            if part_actor is actor:
                return part_id
        return None

    def _handle_pour_pick(self, x: int, y: int) -> None:
        """Resolve a click to a world point and store it as the pour location."""
        cell_picker = vtk.vtkCellPicker()
        cell_picker.SetTolerance(0.01)
        cell_picker.Pick(x, y, 0, self.renderer)

        point = None
        normal = None
        part_id = None

        if cell_picker.GetCellId() >= 0:
            point = tuple(cell_picker.GetPickPosition())
            part_id = self._part_id_for_actor(cell_picker.GetActor())
            try:
                raw_normal = tuple(cell_picker.GetPickNormal())
            except AttributeError:
                raw_normal = (0.0, 0.0, 0.0)
            if any(abs(component) > 1e-9 for component in raw_normal):
                normal = raw_normal
        else:
            world_picker = vtk.vtkWorldPointPicker()
            world_picker.Pick(x, y, 0, self.renderer)
            pick_pos = world_picker.GetPickPosition()
            if pick_pos[0] != 0 or pick_pos[1] != 0 or pick_pos[2] != 0:
                point = tuple(pick_pos)

        if point is None:
            # Missed the model entirely -- stay in pick mode and let the
            # user try again rather than silently failing the pick.
            return

        location = PourLocation(point_in=point, normal=normal, part_id=part_id)
        self._pour_pick_enabled = False
        self._pour_location = location
        self._add_pour_location_marker(location)
        self.render_window.Render()
        self.pour_location_changed.emit(location)

    def _add_pour_location_marker(self, location: PourLocation) -> None:
        """Render a sphere (plus a normal arrow, if known) at ``location``."""
        self._clear_pour_location_marker()

        scale = 1.0
        if self.model_bounds:
            dx = self.model_bounds[1] - self.model_bounds[0]
            dy = self.model_bounds[3] - self.model_bounds[2]
            dz = self.model_bounds[5] - self.model_bounds[4]
            diagonal = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            scale = max(0.1, diagonal * 0.015)

        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*location.point_in)
        sphere.SetRadius(scale)
        sphere.SetPhiResolution(20)
        sphere.SetThetaResolution(20)

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())

        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(*POUR_LOCATION_MARKER_COLOR)

        self.renderer.AddActor(sphere_actor)
        self._pour_location_actors.append(sphere_actor)

        # The arrow shows the POUR DIRECTION, which is always straight down.
        #
        # It deliberately does NOT follow the picked surface normal. On the top
        # face of a down sprue -- the most common place to pour -- the outward
        # normal points +Z, so a normal-aligned arrow reads as metal being
        # fired upward out of the sprue. The solver never consults the normal
        # anyway: gravity is fixed at -Z in physics.filling.gravity_force, so
        # the metal falls -Z regardless of which face was clicked. Drawing the
        # normal here would advertise a behaviour the solver does not have.
        #
        # The normal is still stored on the PourLocation; it is useful for
        # offsetting the marker clear of the surface and will matter if a
        # future inlet models a directed stream rather than a gravity pour.
        arrow_length = scale * 3.0
        nx, ny, nz = POUR_DIRECTION
        px, py, pz = location.point_in
        tip = (px + nx * arrow_length, py + ny * arrow_length, pz + nz * arrow_length)

        cone = vtk.vtkConeSource()
        cone.SetHeight(arrow_length)
        cone.SetRadius(scale * 0.6)
        cone.SetResolution(16)
        cone.SetDirection(nx, ny, nz)
        cone.SetCenter(
            (px + tip[0]) / 2.0,
            (py + tip[1]) / 2.0,
            (pz + tip[2]) / 2.0,
        )

        cone_mapper = vtk.vtkPolyDataMapper()
        cone_mapper.SetInputConnection(cone.GetOutputPort())

        cone_actor = vtk.vtkActor()
        cone_actor.SetMapper(cone_mapper)
        cone_actor.GetProperty().SetColor(*POUR_LOCATION_MARKER_COLOR)

        self.renderer.AddActor(cone_actor)
        self._pour_location_actors.append(cone_actor)

    def _clear_pour_location_marker(self) -> None:
        """Remove any pour-location marker actors from the scene."""
        for actor in self._pour_location_actors:
            self.renderer.RemoveActor(actor)
        self._pour_location_actors = []

    # ==================== Orient & Parting Line ====================
    #
    # A one-shot surface pick (face down / face up / parting height) and the
    # parting-plane overlay. Neither touches the result overlays, and the
    # plane actors are unpickable so pour, measurement and inspector clicks
    # keep hitting the parts underneath.

    def start_surface_pick(self, part_ids: Optional[List[str]] = None) -> None:
        """Enter surface pick mode; the next left click on a part emits ``surface_picked``.

        ``part_ids`` limits the pick to those part actors (None = any part).
        Pour-location pick mode is switched off so one click cannot do both.
        """
        self._pour_pick_enabled = False
        self._surface_pick_enabled = True
        self._surface_pick_part_ids = list(part_ids) if part_ids else None
        if self.vtk_widget is not None:
            self.vtk_widget.setCursor(Qt.CrossCursor)

    def cancel_surface_pick(self) -> None:
        """Leave surface pick mode without emitting anything."""
        self._surface_pick_enabled = False
        self._surface_pick_part_ids = None
        if self.vtk_widget is not None:
            self.vtk_widget.unsetCursor()

    def is_surface_pick_mode(self) -> bool:
        return self._surface_pick_enabled

    def _on_key_press(self, obj, event) -> None:
        """Esc in the viewport cancels an active surface pick."""
        if not self._surface_pick_enabled:
            return
        if self.interactor.GetKeySym() == "Escape":
            self.cancel_surface_pick()
            self.surface_pick_cancelled.emit()

    def _handle_surface_pick(self, x: int, y: int) -> None:
        """Resolve a click to a picked triangle and emit it; a miss keeps the mode on."""
        if self._surface_pick_part_ids is None:
            actors = list(self.part_actors.values())
        else:
            actors = [
                self.part_actors[pid] for pid in self._surface_pick_part_ids if pid in self.part_actors
            ]
        if not actors:
            return
        picker = vtk.vtkCellPicker()
        for actor in actors:
            picker.AddPickList(actor)
        picker.SetPickFromList(True)
        picker.SetTolerance(0.005)
        picker.Pick(x, y, 0, self.renderer)
        cell_id = picker.GetCellId()
        dataset = picker.GetDataSet()
        if cell_id < 0 or dataset is None:
            return
        point = tuple(float(v) for v in picker.GetPickPosition())
        cell_points = dataset.GetCell(cell_id).GetPoints()
        triangle = [tuple(cell_points.GetPoint(i)) for i in range(min(3, cell_points.GetNumberOfPoints()))]
        if len(triangle) < 3:
            return
        camera = self.renderer.GetActiveCamera()
        if camera.GetParallelProjection():
            toward_viewer = tuple(-float(v) for v in camera.GetDirectionOfProjection())
        else:
            cam = camera.GetPosition()
            toward_viewer = tuple(float(cam[i]) - point[i] for i in range(3))
        self.cancel_surface_pick()
        self.surface_picked.emit(
            {
                "point": point,
                "triangle": triangle,
                "toward_viewer": toward_viewer,
                "part_id": self._part_id_for_actor(picker.GetActor()),
            }
        )

    def show_parting_plane(
        self,
        z: float,
        extent: Tuple[float, float, float, float],
        segments: Optional[np.ndarray] = None,
        label: str = "",
    ) -> None:
        """Draw the parting plane at ``z`` over ``extent`` = (xmin, xmax, ymin, ymax).

        Translucent fill, solid outline, a bright section line where the plane
        cuts the casting (``segments``, shape (n, 2, 3)), a label, and COPE /
        DRAG tags just above and below the plane.
        """
        self.hide_parting_plane(render=False)
        xmin, xmax, ymin, ymax = (float(v) for v in extent)
        z = float(z)

        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(xmin, ymin, z)
        plane.SetPoint1(xmax, ymin, z)
        plane.SetPoint2(xmin, ymax, z)
        plane_mapper = vtk.vtkPolyDataMapper()
        plane_mapper.SetInputConnection(plane.GetOutputPort())

        fill = vtk.vtkActor()
        fill.SetMapper(plane_mapper)
        fill_prop = fill.GetProperty()
        fill_prop.SetColor(*PARTING_PLANE_COLOR)
        fill_prop.SetOpacity(0.22)
        fill_prop.LightingOff()
        fill.PickableOff()

        outline = vtk.vtkActor()
        outline.SetMapper(plane_mapper)
        outline_prop = outline.GetProperty()
        outline_prop.SetRepresentationToWireframe()
        outline_prop.SetColor(*PARTING_PLANE_COLOR)
        outline_prop.SetLineWidth(2.0)
        outline_prop.LightingOff()
        outline.PickableOff()

        actors: List[Any] = [fill, outline]

        if segments is not None and len(segments):
            seg = np.asarray(segments, dtype=float).reshape(-1, 3)
            points = vtk.vtkPoints()
            points.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(seg), deep=True))
            lines = vtk.vtkCellArray()
            for i in range(0, seg.shape[0], 2):
                lines.InsertNextCell(2)
                lines.InsertCellPoint(i)
                lines.InsertCellPoint(i + 1)
            section = vtk.vtkPolyData()
            section.SetPoints(points)
            section.SetLines(lines)
            section_mapper = vtk.vtkPolyDataMapper()
            section_mapper.SetInputData(section)
            section_actor = vtk.vtkActor()
            section_actor.SetMapper(section_mapper)
            section_prop = section_actor.GetProperty()
            section_prop.SetColor(*PARTING_SECTION_COLOR)
            section_prop.SetLineWidth(4.0)
            section_prop.LightingOff()
            section_actor.PickableOff()
            actors.append(section_actor)

        tag_offset = max(0.05, 0.04 * max(xmax - xmin, ymax - ymin))
        texts = [
            (label, (xmin, ymin, z), PARTING_PLANE_COLOR, 16),
            ("COPE (above)", (xmax, ymax, z + tag_offset), PARTING_COPE_COLOR, 14),
            ("DRAG (below)", (xmax, ymax, z - tag_offset), PARTING_DRAG_COLOR, 14),
        ]
        for text, position, color, size in texts:
            if not text:
                continue
            text_actor = vtk.vtkBillboardTextActor3D()
            text_actor.SetInput(text)
            text_actor.SetPosition(*position)
            text_prop = text_actor.GetTextProperty()
            text_prop.SetColor(*color)
            text_prop.SetFontSize(size)
            text_prop.BoldOn()
            text_actor.PickableOff()
            actors.append(text_actor)

        for actor in actors:
            self.renderer.AddActor(actor)
        self._parting_actors = actors
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()

    def hide_parting_plane(self, render: bool = True) -> None:
        """Remove the parting plane overlay."""
        for actor in self._parting_actors:
            self.renderer.RemoveActor(actor)
        self._parting_actors = []
        if render and self.render_window is not None:
            self.render_window.Render()

    def is_parting_plane_shown(self) -> bool:
        return bool(self._parting_actors)

    def enable_feed_modulus_leader(self, enabled: bool):
        """Enable or disable feed modulus leader mode.

        Backwards-compatible wrapper around :meth:`set_inspector`. The feedmod
        layer reads its values from ``current_shrinkage_data``; the layer is
        installed (or refreshed) when shrinkage geometry is (re)built.
        Enabling feedmod also clears any registered shrink/porosity data
        so the two inspector modes do not race.
        """
        self._feed_modulus_leader_enabled = enabled
        if not enabled:
            if (
                self._inspector_layer is not None
                and self._inspector_layer.mode == "feed_modulus"
            ):
                self._inspector_layer = None
            self.clear_feed_modulus_leader()
            return
        # Disabling the shrink inspector intent so feedmod takes over.
        self._shrink_inspector_data = None
        if (
            self._inspector_layer is not None
            and self._inspector_layer.mode == "shrinkage"
        ):
            self._inspector_layer = None
            if self.feed_modulus_leader is not None:
                self.feed_modulus_leader.clear()
        self._install_feedmod_inspector()

    def clear_feed_modulus_leader(self):
        """Clear the feed modulus leader annotation"""
        if self.feed_modulus_leader:
            self.feed_modulus_leader.clear()
            self.render_window.Render()

    def set_inspector(self, layer: Optional[InspectorLayer]) -> None:
        """Install or remove the active result-mode inspector.

        Passing ``None`` clears the inspector and any visible leader. The
        layer's ``resolver`` is invoked on each click to format the label.
        """
        self._inspector_layer = layer
        if layer is None and self.feed_modulus_leader is not None:
            self.feed_modulus_leader.clear()
            self.render_window.Render()

    def clear_inspector(self) -> None:
        """Clear the active inspector and its leader (if any)."""
        self.set_inspector(None)

    def set_shrink_inspector_data(
        self,
        *,
        display_intensity: np.ndarray,
        mesh,
        shrinkage_risk: Optional[np.ndarray],
        porosity_risk: Optional[np.ndarray],
        threshold: float,
        connectivity: int = 6,
    ) -> None:
        """Register the data the shrink/porosity inspector needs.

        Called by ``main_window`` after :meth:`load_shrinkage` so the click
        handler can build a connected defect pocket and report its volume.
        The layer is reinstalled after every shrinkage geometry rebuild so
        the displayed voxel indices always match the active actor.

        Registering shrink data also disables the feedmod inspector intent
        so the two modes do not race for the click handler.
        """
        self._shrink_inspector_data = {
            "display_intensity": np.asarray(display_intensity, dtype=np.float32),
            "mesh": mesh,
            "shrinkage_risk": (
                None if shrinkage_risk is None
                else np.asarray(shrinkage_risk, dtype=np.float32)
            ),
            "porosity_risk": (
                None if porosity_risk is None
                else np.asarray(porosity_risk, dtype=np.float32)
            ),
            "threshold": float(threshold),
            "connectivity": int(connectivity),
        }
        # Disabling feedmod intent so shrink mode owns the click handler.
        self._feed_modulus_leader_enabled = False
        if (
            self._inspector_layer is not None
            and self._inspector_layer.mode == "feed_modulus"
        ):
            self._inspector_layer = None
            if self.feed_modulus_leader is not None:
                self.feed_modulus_leader.clear()
        self._install_shrink_inspector()

    def clear_shrink_inspector_data(self) -> None:
        """Forget shrink/porosity inspector data and clear the active leader."""
        had_layer = (
            self._inspector_layer is not None
            and self._inspector_layer.mode == "shrinkage"
        )
        self._shrink_inspector_data = None
        if had_layer:
            self._inspector_layer = None
            if self.feed_modulus_leader is not None:
                self.feed_modulus_leader.clear()
                self.render_window.Render()

    def _install_feedmod_inspector(self) -> None:
        """Install the feedmod layer using ``current_shrinkage_data``."""
        if not self._feed_modulus_leader_enabled:
            return
        if self.current_shrinkage_data is None or self.result_surface is None or self.result_surface.view is None:
            return

        def _resolver(voxel_index: int) -> Optional[LeaderCallout]:
            centers, _, values = self.current_shrinkage_data
            if voxel_index < 0 or voxel_index >= int(values.shape[0]):
                return None
            v = float(values[voxel_index])
            if not np.isfinite(v) or v <= 0.0:
                return None
            anchor = tuple(np.asarray(centers[voxel_index], dtype=np.float64).tolist())
            return LeaderCallout(text=f"FM: {v:.2f} in", anchor_position=anchor)

        self._inspector_layer = InspectorLayer(
            mode="feed_modulus",
            pick_actors=self.result_surface.pick_actors(),
            voxel_at=self.result_surface.voxel_at,
            resolver=_resolver,
        )

    def _install_shrink_inspector(self) -> None:
        """Install the shrink/porosity layer using registered data."""
        data = getattr(self, "_shrink_inspector_data", None)
        if data is None or self.result_surface is None or self.result_surface.view is None:
            return

        from core.defect_regions import extract_pocket_at, format_pocket_volume_label

        def _resolver(voxel_index: int) -> Optional[LeaderCallout]:
            mesh = data["mesh"]
            pocket = extract_pocket_at(
                voxel_index=voxel_index,
                display_intensity=data["display_intensity"],
                voxel_neighbors=mesh.voxel_neighbors,
                voxel_centers=mesh.voxel_centers,
                voxel_volumes=mesh.voxel_volumes,
                shrinkage_risk=data["shrinkage_risk"],
                porosity_risk=data["porosity_risk"],
                threshold=data["threshold"],
                connectivity=data["connectivity"],
                # On a tet mesh the neighbour rows are node adjacency, not
                # 26 lattice slots; defect_regions then uses the full row.
                mesh_type=getattr(mesh, "mesh_type", "voxel"),
            )
            if pocket is None:
                return None
            return LeaderCallout(
                text=format_pocket_volume_label(pocket),
                anchor_position=pocket.centroid,
            )

        self._inspector_layer = InspectorLayer(
            mode="shrinkage",
            pick_actors=self.result_surface.pick_actors(),
            voxel_at=self.result_surface.voxel_at,
            resolver=_resolver,
        )

    def _refresh_active_inspector(self) -> None:
        """Reinstall the active inspector after a geometry rebuild.

        Called after every result load / threshold change so the pick list
        matches the actors currently on screen. If a mode
        is already installed the data is refreshed in place; otherwise the
        call installs the mode whose intent flag is currently set.
        """
        if self._inspector_layer is not None:
            mode = self._inspector_layer.mode
            if mode == "feed_modulus":
                self._install_feedmod_inspector()
            elif mode == "shrinkage":
                self._install_shrink_inspector()
            return

        # No active layer: pick the most-recently-enabled mode.
        if self._feed_modulus_leader_enabled:
            self._install_feedmod_inspector()
            return
        if getattr(self, "_shrink_inspector_data", None) is not None:
            self._install_shrink_inspector()

    def handle_click_for_measurement(self, x: int, y: int):
        """Handle a click event for measurement"""
        if self.measurement_tool and self.measurement_tool.enabled:
            point = self.measurement_tool.pick_point(x, y)
            if point:
                self.measurement_tool.add_point(point)
                self.render_window.Render()
    
    def calculate_defect_volume(self, threshold: float = 0.3) -> Optional[float]:
        """
        Calculate total volume of defects above threshold
        
        Args:
            threshold: Intensity threshold
            
        Returns:
            Volume in cubic inches, or None if no data
        """
        if self.current_shrinkage_data is None:
            return None
        
        voxel_centers, voxel_sizes, intensities = self.current_shrinkage_data
        volumes = self._result_point_volumes
        if volumes is not None and len(volumes) != len(intensities):
            volumes = None
        return self.measurement_tool.calculate_defect_volume(
            voxel_centers, voxel_sizes, intensities, threshold, voxel_volumes=volumes
        )
    
    # ==================== Animation Export Methods ====================
    
    def start_animation_recording(self):
        """Start recording frames for animation"""
        self._animation_frames = []
        self._is_recording = True
        print("Animation recording started")
    
    def capture_frame(self):
        """Capture current view as an animation frame"""
        if not self._is_recording:
            return
        
        # Render to ensure we capture the current state
        self.render_window.Render()
        
        # Capture frame
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.render_window)
        w2if.SetInputBufferTypeToRGB()
        w2if.ReadFrontBufferOff()
        w2if.Update()
        
        # Convert to numpy array
        vtk_image = w2if.GetOutput()
        dims = vtk_image.GetDimensions()
        array = numpy_support.vtk_to_numpy(vtk_image.GetPointData().GetScalars())
        array = array.reshape(dims[1], dims[0], 3)
        array = np.flipud(array)  # VTK images are flipped
        
        self._animation_frames.append(array.copy())
    
    def stop_animation_recording(self) -> int:
        """
        Stop recording and return number of frames captured
        
        Returns:
            Number of frames captured
        """
        self._is_recording = False
        count = len(self._animation_frames)
        print(f"Animation recording stopped: {count} frames captured")
        return count
    
    def export_animation_gif(self, filepath: str, fps: int = 10) -> bool:
        """
        Export recorded frames as GIF
        
        Args:
            filepath: Output GIF file path
            fps: Frames per second
            
        Returns:
            True if successful
        """
        if not self._animation_frames:
            print("No frames to export")
            return False
        
        try:
            from PIL import Image
            
            # Convert frames to PIL images
            images = []
            for frame in self._animation_frames:
                img = Image.fromarray(frame)
                images.append(img)
            
            # Save as GIF
            duration = int(1000 / fps)  # ms per frame
            images[0].save(
                filepath,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0
            )
            
            print(f"Animation exported to: {filepath}")
            return True
            
        except ImportError:
            print("Error: PIL/Pillow is required for GIF export")
            print("Install with: pip install Pillow")
            return False
        except Exception as e:
            print(f"Error exporting GIF: {e}")
            return False
    
    def export_animation_mp4(self, filepath: str, fps: int = 30) -> bool:
        """
        Export recorded frames as MP4 video
        
        Args:
            filepath: Output MP4 file path
            fps: Frames per second
            
        Returns:
            True if successful
        """
        if not self._animation_frames:
            print("No frames to export")
            return False
        
        try:
            import cv2
            
            # Get dimensions from first frame
            height, width = self._animation_frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            
            for frame in self._animation_frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"Animation exported to: {filepath}")
            return True
            
        except ImportError:
            print("Error: OpenCV is required for MP4 export")
            print("Install with: pip install opencv-python")
            return False
        except Exception as e:
            print(f"Error exporting MP4: {e}")
            return False
    
    def clear_animation_frames(self):
        """Clear recorded animation frames"""
        self._animation_frames = []
    
    def _refresh_current_visualization(self):
        """Re-paint the current result after a colormap or scale change."""
        surface = getattr(self, "result_surface", None)
        if surface is None or surface.view is None:
            return
        view = surface.view
        if view.mode in ("shrinkage",) or view.mode.startswith("gas_"):
            name = "defect_industry" if view.mode.startswith("gas_") else None
            view.palette = self._colormap_palette(name)
        surface.show(**surface._view_kwargs())

    def _apply_display_mask(self, mask: np.ndarray) -> np.ndarray:
        """AND the review-time voxel display mask into a builder's selection mask."""
        return apply_display_mask(self, mask)

    def _part_actor_should_show(self, part_id: str) -> bool:
        model_visible = getattr(self, "_model_visible", True)
        hidden = getattr(self, "_hidden_part_ids", None)
        if not isinstance(hidden, set):
            hidden = set()
        return bool(model_visible is not False) and part_id not in hidden

    def set_voxel_display_mask(self, mask: Optional[np.ndarray]) -> None:
        """Install the per-voxel display mask (True = drawable).

        The result renderer re-scatters the current result under the new
        mask; the caller may also re-run the visualization. Does not render.
        """
        if mask is None:
            self._voxel_display_mask = None
        else:
            self._voxel_display_mask = np.ascontiguousarray(mask, dtype=bool)
        if self.result_surface is not None:
            self.result_surface.set_display_mask(self._voxel_display_mask)

    def voxel_display_mask(self) -> Optional[np.ndarray]:
        """The installed per-voxel display mask, or None when nothing is hidden."""
        return self._voxel_display_mask

    def hidden_part_ids(self) -> set:
        """Parts hidden for review (display only, not simulation membership)."""
        return set(self._hidden_part_ids)

    def set_component_hidden(self, part_id: str, hidden: bool) -> None:
        """Hide or show one component's skin actor without redrawing results."""
        if hidden:
            self._hidden_part_ids.add(part_id)
        else:
            self._hidden_part_ids.discard(part_id)
        self._apply_part_actor_visibility()
        self.render_window.Render()

    def clear_hidden_components(self) -> None:
        """Show every component skin and drop the voxel display mask."""
        self._hidden_part_ids.clear()
        self._voxel_display_mask = None
        if self.result_surface is not None:
            self.result_surface.set_display_mask(None)
        self._apply_part_actor_visibility()
        self.render_window.Render()

    def _apply_part_actor_visibility(self) -> None:
        """Part skin visible = master Show Model flag AND not hidden for review."""
        surface = getattr(self, "result_surface", None)
        if surface is not None:
            surface.set_hidden_parts(set(getattr(self, "_hidden_part_ids", set())))
        # A part whose painted twin is on screen hides its own ghost actor.
        painted = surface.painted_part_ids() if surface is not None else set()
        for part_id, actor in self.part_actors.items():
            actor.SetVisibility(self._part_actor_should_show(part_id) and part_id not in painted)

    def _add_axes(self):
        """Add coordinate axes to scene in bottom right corner"""
        # Create orientation widget for bottom right
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.5, 1.5, 1.5)  # Small size
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.03)
        axes.SetTipTypeToCone()
        
        # Add labels
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y")
        axes.SetZAxisLabelText("Z")
        
        # Make axis colors more visible
        axes.GetXAxisShaftProperty().SetColor(1, 0, 0)  # Red
        axes.GetYAxisShaftProperty().SetColor(0, 1, 0)  # Green
        axes.GetZAxisShaftProperty().SetColor(0, 0, 1)  # Blue
        
        # Create orientation marker widget for bottom right
        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(axes)
        self.orientation_widget.SetInteractor(self.interactor)
        self.orientation_widget.SetViewport(0.8, 0.0, 1.0, 0.2)  # Bottom right corner
        self.orientation_widget.SetEnabled(1)
        self.orientation_widget.InteractiveOff()
    
    def load_mesh(self, vertices: np.ndarray, faces: np.ndarray):
        """
        Load and display translucent mesh (ghost model)
        
        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices
        """
        # Remove old mesh actor
        if self.mesh_actor is not None:
            self.renderer.RemoveActor(self.mesh_actor)
        
        # Create VTK points
        points = vtk.vtkPoints()
        for vertex in vertices:
            points.InsertNextPoint(vertex)
        
        # Create VTK cells (triangles)
        triangles = vtk.vtkCellArray()
        for face in faces:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, face[0])
            triangle.GetPointIds().SetId(1, face[1])
            triangle.GetPointIds().SetId(2, face[2])
            triangles.InsertNextCell(triangle)
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(triangles)
        
        # Compute normals for smooth shading
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.Update()
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        
        # Create actor
        self.mesh_actor = vtk.vtkActor()
        self.mesh_actor.SetMapper(mapper)
        
        # Set translucent appearance (ghost-like)
        property = self.mesh_actor.GetProperty()
        property.SetColor(0.7, 0.75, 0.8)  # Slightly darker blue-grey
        property.SetOpacity(0.35)  # 35% opacity (more visible)
        self._pbr_surface(property)
        
        if should_use_depth_peeling():
            self.renderer.SetUseDepthPeeling(True)
            self.renderer.SetMaximumNumberOfPeels(4)
            self.renderer.SetOcclusionRatio(0.1)
        else:
            self.renderer.SetUseDepthPeeling(False)
        
        self.renderer.AddActor(self.mesh_actor)
        
        self.current_mesh = polydata
        
        # Store bounds for camera presets
        self.model_bounds = self.mesh_actor.GetBounds()
        self._configure_ssao()
        
        # Reset camera
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        if self._is_orthographic:
            self._update_parallel_scale()
        self.render_window.Render()
    
    # ==================== Result overlays (surface renderer) ====================
    #
    # Every load_* below hands the per-voxel values to ResultSurfaceRenderer,
    # which paints them on the casting skin, contours the threshold envelope
    # and cuts painted section caps. The signatures are the ones the main
    # window has always called; only the drawing changed.

    def set_result_mesh(self, voxel_mesh) -> bool:
        """Register the voxel mesh a result set belongs to (once per run).

        Builds the dense lattice the surface renderer paints from. Part types
        decide which cells count as metal for painting. Idempotent for the
        same mesh object.
        """
        if self.result_surface is None or voxel_mesh is None or getattr(voxel_mesh, "voxel_centers", None) is None:
            return False
        centers = np.asarray(voxel_mesh.voxel_centers)
        key = (id(voxel_mesh), int(len(centers)))
        if key == self._result_lattice_key and self.result_surface.lattice is not None:
            return True
        self._result_point_volumes = None
        if getattr(voxel_mesh, "mesh_type", "voxel") == "tet":
            # Tet nodes are not on a lattice: sample through the tets at the
            # pitch the web sidecar and the fill use.
            from core.dense_grid import tet_presentation_spacing

            ok = self.result_surface.set_tet_mesh(
                np.asarray(voxel_mesh.node_xyz),
                np.asarray(voxel_mesh.tets),
                np.asarray(voxel_mesh.tet_part_types),
                tet_presentation_spacing(voxel_mesh),
            )
            if ok:
                self._result_point_volumes = np.asarray(voxel_mesh.voxel_volumes, dtype=np.float64)
        else:
            sizes = getattr(voxel_mesh, "voxel_sizes", None)
            part_types = getattr(voxel_mesh, "voxel_part_types", None)
            ok = self.result_surface.set_voxels(
                centers,
                None if sizes is None else np.asarray(sizes),
                None if part_types is None else np.asarray(part_types),
            )
        self._result_lattice_key = key if ok else None
        if ok and self._voxel_display_mask is not None:
            self.result_surface.set_display_mask(self._voxel_display_mask)
        self._configure_ssao()
        return ok

    def _ensure_result_lattice(self, voxel_centers: np.ndarray, voxel_sizes: np.ndarray) -> bool:
        """Lazy fallback when the caller never registered the mesh."""
        if self.result_surface is None:
            return False
        key = (id(voxel_centers), int(len(voxel_centers)))
        if self.result_surface.lattice is not None and self._result_lattice_key is not None:
            if self._result_lattice_key[1] == int(len(voxel_centers)):
                return True
        ok = self.result_surface.set_voxels(np.asarray(voxel_centers), np.asarray(voxel_sizes), None)
        self._result_lattice_key = key if ok else None
        if ok and self._voxel_display_mask is not None:
            self.result_surface.set_display_mask(self._voxel_display_mask)
        self._configure_ssao()
        return ok

    def _colormap_palette(self, colormap: Optional[str] = None):
        """Palette callable for the surface renderer: normalised value -> RGB."""
        name = colormap or self._current_colormap
        lo, hi = self._color_scale_min, self._color_scale_max

        def _palette(t: np.ndarray) -> np.ndarray:
            return apply_colormap_to_array(np.asarray(t, dtype=np.float32), name, lo, hi)

        return _palette

    def _show_result(self, mode: str, voxel_centers, voxel_sizes, values, iso, palette, **kwargs) -> bool:
        if not self._ensure_result_lattice(voxel_centers, voxel_sizes):
            print("Result lattice unavailable; nothing drawn")
            return False
        ok = self.result_surface.show(mode, np.asarray(values, dtype=np.float32), iso=float(iso), palette=palette, **kwargs)
        self._apply_cross_section_clipping()
        self._apply_part_actor_visibility()
        self._refresh_active_inspector()
        self.render_window.Render()
        return ok

    def _set_result_iso(self, iso: float) -> None:
        if self.result_surface is None or self.result_surface.view is None:
            return
        self.result_surface.set_iso(float(iso))
        self._apply_cross_section_clipping()
        self._refresh_active_inspector()
        self.render_window.Render()

    # -- threshold modes ------------------------------------------------------

    def load_shrinkage(self, voxel_centers: np.ndarray, voxel_sizes: np.ndarray,
                       shrinkage_intensity: np.ndarray, threshold: float = 0.3,
                       vmin: float = 0.0, vmax: float = 1.0):
        """Shrinkage / porosity risk (also feed modulus, in inches, via vmin/vmax)."""
        self.current_shrinkage_data = (voxel_centers, voxel_sizes, shrinkage_intensity)
        if self.feed_modulus_leader:
            self.feed_modulus_leader.clear()
        self._show_result("shrinkage", voxel_centers, voxel_sizes, shrinkage_intensity, threshold,
                          self._colormap_palette(), vmin=vmin, vmax=vmax)

    def update_shrinkage_threshold(self, threshold: float):
        if self.current_shrinkage_data is None:
            return
        self._set_result_iso(threshold)

    def load_hotspots(self, voxel_centers: np.ndarray, voxel_sizes: np.ndarray,
                      hotspot_intensity: np.ndarray, threshold: float = 0.5):
        """Hot spots: yellow (mild) to red (last to freeze)."""
        self.current_hotspot_data = (voxel_centers, voxel_sizes, hotspot_intensity)
        self._show_result("hotspot", voxel_centers, voxel_sizes, hotspot_intensity, threshold,
                          palette_hotspot, vmin=0.0, vmax=1.0)

    def update_hotspot_threshold(self, threshold: float):
        if self.current_hotspot_data is None:
            return
        self._set_result_iso(threshold)

    def load_niyama(self, voxel_centers: np.ndarray, voxel_sizes: np.ndarray,
                    niyama_values: np.ndarray, defect_risk: np.ndarray, threshold: float = 0.5):
        """Niyama defect risk: blue (safe) through cyan and yellow to red."""
        self.current_niyama_data = (voxel_centers, voxel_sizes, niyama_values, defect_risk)
        self._show_result("niyama", voxel_centers, voxel_sizes, defect_risk, threshold,
                          palette_niyama, vmin=0.0, vmax=1.0)

    def update_niyama_threshold(self, threshold: float):
        if self.current_niyama_data is None:
            return
        self._set_result_iso(threshold)

    def load_hot_tear(self, voxel_centers: np.ndarray, voxel_sizes: np.ndarray,
                      hot_tear_risk: np.ndarray, hot_tear_zones: np.ndarray, threshold: float = 0.1):
        """Hot tear risk inside the physics-eligible zones: cyan to magenta."""
        self.current_hot_tear_data = (voxel_centers, voxel_sizes, hot_tear_risk, hot_tear_zones)
        self._show_result("hot_tear", voxel_centers, voxel_sizes, hot_tear_risk, threshold,
                          palette_hot_tear, vmin=0.0, vmax=1.0,
                          valid=np.asarray(hot_tear_zones, dtype=bool))

    def update_hot_tear_threshold(self, threshold: float):
        if self.current_hot_tear_data is None:
            return
        self._set_result_iso(threshold)

    def load_sand_burn_in(self, voxel_centers: np.ndarray, voxel_sizes: np.ndarray,
                          sand_burn_in_risk: np.ndarray, sand_burn_in_zones: np.ndarray,
                          threshold: float = 0.2):
        """Sand burn-in risk on the mold contact zones: tan, orange, brown."""
        self.current_sand_burn_in_data = (voxel_centers, voxel_sizes, sand_burn_in_risk, sand_burn_in_zones)
        self._show_result("sand_burn_in", voxel_centers, voxel_sizes, sand_burn_in_risk, threshold,
                          palette_sand_burn_in, vmin=0.0, vmax=1.0,
                          valid=np.asarray(sand_burn_in_zones, dtype=bool))

    def update_sand_burn_in_threshold(self, threshold: float):
        if self.current_sand_burn_in_data is None:
            return
        self._set_result_iso(threshold)

    def load_gas_porosity(self, voxel_centers: np.ndarray, voxel_sizes: np.ndarray,
                          intensities: np.ndarray, threshold: float = 0.1, mode: str = "gas_porosity"):
        """Gas generation / entrapment / porosity in the industry-standard palette."""
        self.current_gas_porosity_data = (voxel_centers, voxel_sizes, intensities, mode)
        self._show_result(mode, voxel_centers, voxel_sizes, intensities, threshold,
                          self._colormap_palette("defect_industry"), vmin=0.0, vmax=1.0)

    def update_gas_porosity_threshold(self, threshold: float):
        if self.current_gas_porosity_data is None:
            return
        self._set_result_iso(threshold)

    # -- time modes -------------------------------------------------------------

    def load_fraction_liquid(self, voxel_centers: np.ndarray, voxel_sizes: np.ndarray,
                             fraction_liquid: np.ndarray, threshold: float = 0.01,
                             cache_key: Optional[int] = None):
        """One solidification frame: molten paint on liquid metal, a glowing
        front at 50 % liquid, steel where the metal has frozen."""
        self.current_fraction_liquid_data = (voxel_centers, voxel_sizes, fraction_liquid)
        fl = np.asarray(fraction_liquid, dtype=np.float32)
        self._show_result("fraction_liquid", voxel_centers, voxel_sizes, fl, 0.5, palette_liquid,
                          vmin=0.0, vmax=1.0, paint_mask=fl > float(threshold), glow=True)

    def load_fill_time(self, voxel_centers: np.ndarray, voxel_sizes: np.ndarray,
                       fill_time: np.ndarray, time_threshold: float, max_fill_time_s: float,
                       cache_key: Optional[int] = None):
        """Mold filling swept to ``time_threshold``: arrived metal painted by
        arrival time (blue early, red late), misruns grey, the front glowing.

        ``fill_time`` marks a misrun with -1 and a non-metal voxel with NaN.
        """
        self.current_fill_time_data = (voxel_centers, voxel_sizes, fill_time)
        ft = np.asarray(fill_time, dtype=np.float32)
        finite = np.isfinite(ft)
        misrun = finite & (ft < 0.0)
        arrived = finite & (ft >= 0.0) & (ft <= float(time_threshold))
        safe_max = float(max_fill_time_s) if max_fill_time_s > 0.0 else 1.0
        # Misruns never arrive: push them past the sweep so the front ignores them.
        sweep_values = np.where(misrun, safe_max * 10.0, ft)
        misrun_rgb = tuple(int(round(c * 255)) for c in FILL_TIME_MISRUN_COLOR)
        self._show_result("fill_time", voxel_centers, voxel_sizes, sweep_values, float(time_threshold),
                          palette_fill_time, vmin=0.0, vmax=safe_max, valid=finite,
                          paint_mask=arrived, glow=True,
                          flat_paint={"misrun": (misrun, misrun_rgb)})

    def load_fill_animation(self, voxel_centers: np.ndarray, voxel_sizes: np.ndarray,
                            fill_fraction: np.ndarray, speed_m_s: np.ndarray, max_speed: float,
                            cache_key: Optional[int] = None):
        """One filling frame: wetted metal painted by flow speed, the free
        surface glowing at the wetted threshold."""
        self.current_fill_animation_data = (voxel_centers, voxel_sizes, fill_fraction, speed_m_s)
        ff = np.asarray(fill_fraction, dtype=np.float32)
        safe_max = float(max_speed) if max_speed > 0.0 else 1.0
        self._show_result("fill_animation", voxel_centers, voxel_sizes, ff, FILL_ANIMATION_WETTED_THRESHOLD,
                          palette_fill_speed, vmin=0.0, vmax=1.0,
                          paint_mask=ff > FILL_ANIMATION_WETTED_THRESHOLD,
                          paint_values=np.asarray(speed_m_s, dtype=np.float32), paint_min=0.0, paint_max=safe_max,
                          glow=True)

    # -- style / visibility ---------------------------------------------------------

    def set_result_style(self, style: str) -> None:
        """'surface' paints the skin, 'envelope' shows the isosurface in a ghost shell, 'both' does both."""
        if self.result_surface is None:
            return
        self.result_surface.set_style(style)
        self._apply_part_actor_visibility()
        self._refresh_active_inspector()
        self.render_window.Render()

    def result_style(self) -> str:
        return self.result_surface.style if self.result_surface is not None else "envelope"

    def hide_all_analysis(self):
        """Hide the result overlay (model stays)."""
        if self.result_surface is not None:
            self.result_surface.set_visible(False)
            self._apply_part_actor_visibility()
        self.render_window.Render()

    def show_analysis(self, analysis_type: str):
        """Show the result overlay again (one result is loaded at a time)."""
        if self.result_surface is not None:
            self.result_surface.set_visible(True)
            self._apply_part_actor_visibility()
        self.render_window.Render()

    def clear_visualization(self):
        """Clear all visualization: model, parts and the result overlay."""
        if self.mesh_actor is not None:
            self.renderer.RemoveActor(self.mesh_actor)
            self.mesh_actor = None

        # Clear all part actors
        for actor in self.part_actors.values():
            self.renderer.RemoveActor(actor)
        self.part_actors.clear()

        if self.result_surface is not None:
            self.result_surface.clear()
            self.result_surface.clear_skins()
        self._result_lattice_key = None
        self._result_point_volumes = None
        # The voxel array is about to change; the hidden component ids are
        # kept so the main window can rebuild the mask for the next mesh.
        self._voxel_display_mask = None

        # Clear stored data
        self.current_shrinkage_data = None
        self.current_hotspot_data = None
        self.current_niyama_data = None
        self.current_hot_tear_data = None
        self.current_sand_burn_in_data = None
        self.current_gas_porosity_data = None
        self.current_fraction_liquid_data = None
        self.current_fill_time_data = None
        self.current_fill_animation_data = None

        self._inspector_layer = None
        if self.feed_modulus_leader is not None:
            self.feed_modulus_leader.clear()

        self.render_window.Render()

    def clear_result_actors(self):
        """Remove the result overlay but keep the model and part actors.

        Call this when switching between visualization modes. The inspector
        intent flags (``_feed_modulus_leader_enabled``, ``_shrink_inspector_data``)
        are kept so the next load reinstalls the appropriate layer.
        """
        if self.result_surface is not None:
            self.result_surface.clear()
        self._apply_part_actor_visibility()
        self._inspector_layer = None
        if self.feed_modulus_leader is not None:
            self.feed_modulus_leader.clear()
        self.render_window.Render()

    def load_part(self, part_id: str, vertices: np.ndarray, faces: np.ndarray, 
                  color: Optional[Tuple[float, float, float]] = None,
                  opacity: float = 0.35) -> vtk.vtkActor:
        """
        Load a single part with its own actor.
        
        Args:
            part_id: Unique identifier for the part
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices
            color: RGB tuple (0-1 range), or auto-assigned if None
            opacity: Transparency (0-1)
        
        Returns:
            The created VTK actor
        """
        # Remove existing actor for this part if present
        if part_id in self.part_actors:
            self.renderer.RemoveActor(self.part_actors[part_id])
        
        # Auto-assign color if not provided
        if color is None:
            color_idx = len(self.part_actors) % len(self.PART_COLORS)
            color = self.PART_COLORS[color_idx]
        
        # Create VTK points
        points = vtk.vtkPoints()
        for vertex in vertices:
            points.InsertNextPoint(vertex)
        
        # Create VTK cells (triangles)
        triangles = vtk.vtkCellArray()
        for face in faces:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, face[0])
            triangle.GetPointIds().SetId(1, face[1])
            triangle.GetPointIds().SetId(2, face[2])
            triangles.InsertNextCell(triangle)
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(triangles)
        
        # Compute normals for smooth shading
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.Update()
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set appearance
        prop = actor.GetProperty()
        prop.SetColor(*color)
        prop.SetOpacity(opacity)
        self._pbr_surface(prop)
        
        if should_use_depth_peeling():
            self.renderer.SetUseDepthPeeling(True)
            self.renderer.SetMaximumNumberOfPeels(4)
            self.renderer.SetOcclusionRatio(0.1)
        else:
            self.renderer.SetUseDepthPeeling(False)
        
        self.renderer.AddActor(actor)
        self.part_actors[part_id] = actor
        actor.SetVisibility(self._part_actor_should_show(part_id))
        
        # Update bounds
        self._update_model_bounds()
        self.renderer.ResetCameraClippingRange()
        
        self.render_window.Render()
        return actor
    
    def remove_part(self, part_id: str):
        """Remove a specific part by ID"""
        if part_id in self.part_actors:
            self.renderer.RemoveActor(self.part_actors[part_id])
            del self.part_actors[part_id]
            self._update_model_bounds()
            self.render_window.Render()
    
    def set_part_color(self, part_id: str, color: Tuple[float, float, float]):
        """Change color of a specific part"""
        if part_id in self.part_actors:
            self.part_actors[part_id].GetProperty().SetColor(*color)
            self.render_window.Render()
    
    def set_part_opacity(self, part_id: str, opacity: float):
        """Change opacity of a specific part"""
        if part_id in self.part_actors:
            self.part_actors[part_id].GetProperty().SetOpacity(opacity)
            self.render_window.Render()
    
    def set_part_visibility(self, part_id: str, visible: bool):
        """Toggle visibility of a specific part (alias of set_component_hidden)."""
        self.set_component_hidden(part_id, not visible)
    
    def highlight_part(self, part_id: str, highlight: bool = True):
        """Highlight or unhighlight a part (makes it more opaque and adds edge lines)"""
        if part_id in self.part_actors:
            prop = self.part_actors[part_id].GetProperty()
            if highlight:
                prop.SetOpacity(0.7)
                prop.EdgeVisibilityOn()
                prop.SetEdgeColor(1.0, 1.0, 0.0)  # Yellow edges
                prop.SetLineWidth(2.0)
            else:
                prop.SetOpacity(0.35)
                prop.EdgeVisibilityOff()
            self.render_window.Render()
    
    def _update_model_bounds(self):
        """Update model bounds from all part actors"""
        if not self.part_actors:
            self.model_bounds = None
            return
        
        # Combine bounds from all actors
        all_bounds = []
        for actor in self.part_actors.values():
            bounds = actor.GetBounds()
            all_bounds.append(bounds)
        
        if all_bounds:
            self.model_bounds = (
                min(b[0] for b in all_bounds),  # x_min
                max(b[1] for b in all_bounds),  # x_max
                min(b[2] for b in all_bounds),  # y_min
                max(b[3] for b in all_bounds),  # y_max
                min(b[4] for b in all_bounds),  # z_min
                max(b[5] for b in all_bounds),  # z_max
            )
    
    def load_parts_from_manager(self, part_manager, reset_camera: bool = True):
        """
        Load all parts from a PartManager instance.
        
        Args:
            part_manager: PartManager instance with parts to display
            reset_camera: Refit the camera to the parts. The Orient & Parting
                Line tool passes False so the view holds still while the
                user turns the part.
        """
        # Clear existing parts
        for actor in self.part_actors.values():
            self.renderer.RemoveActor(actor)
        self.part_actors.clear()
        
        # Load each part
        skins = []
        for i, part in enumerate(part_manager.get_all_parts()):
            if part.visible:
                color = part.color if part.color else self.PART_COLORS[i % len(self.PART_COLORS)]
                vertices = part.get_vertices()
                faces = part.get_faces()
                self.load_part(part.id, vertices, faces, color=color)
                skins.append((part.id, vertices, faces, part.part_type))
        # The result renderer paints on its own subdivided copy of the metal skins.
        if self.result_surface is not None:
            self.result_surface.set_skins(skins)
        self._configure_ssao()
        
        # Reset camera to show all parts
        if reset_camera:
            self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        if reset_camera and self._is_orthographic:
            self._update_parallel_scale()
        if not self.part_actors and part_manager.get_all_parts():
            logger.warning("No visible VTK part actors were created from loaded parts.")
        else:
            logger.info(
                f"Loaded {len(self.part_actors)} VTK part actor(s); bounds={self.model_bounds}"
            )
        self.render_window.Render()
    
    def toggle_mesh_visibility(self, visible: bool):
        """Master model toggle: the legacy combined mesh and every part skin."""
        self._model_visible = bool(visible)
        if self.mesh_actor is not None:
            self.mesh_actor.SetVisibility(self._model_visible)
        self._apply_part_actor_visibility()
        self.render_window.Render()
    
    def toggle_shrinkage_visibility(self, visible: bool):
        """Show or hide the result overlay ("Show Results")."""
        if self.result_surface is not None:
            self.result_surface.set_visible(bool(visible))
            self._apply_part_actor_visibility()
        self.render_window.Render()
    
    def export_screenshot(self, filepath: str):
        """Export current view as image"""
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.render_window)
        w2if.Update()
        
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filepath)
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()
        
        print(f"Screenshot saved to: {filepath}")
    
    @property
    def is_orthographic(self) -> bool:
        return self._is_orthographic

    def set_projection_mode(self, mode: str):
        """
        Set camera projection mode.

        Args:
            mode: 'orthographic' or 'perspective'
        """
        if mode not in ('orthographic', 'perspective'):
            return
        camera = self.renderer.GetActiveCamera()
        if mode == 'orthographic':
            camera.ParallelProjectionOn()
            self._is_orthographic = True
            self._update_parallel_scale()
        else:
            camera.ParallelProjectionOff()
            self._is_orthographic = False
        self.render_window.Render()

    def _update_parallel_scale(self):
        """Recalculate the parallel scale based on model bounds."""
        camera = self.renderer.GetActiveCamera()
        if self.model_bounds:
            x_min, x_max, y_min, y_max, z_min, z_max = self.model_bounds
            size = max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)
            camera.SetParallelScale(size * 1.15)
        else:
            camera.SetParallelScale(120.0)

    def set_camera_view(self, view_type: str):
        """
        Set camera to preset view

        Args:
            view_type: 'front', 'top', 'bottom', 'left', 'right', 'iso'
        """
        if self.model_bounds is None:
            return

        # Calculate model center and size
        x_min, x_max, y_min, y_max, z_min, z_max = self.model_bounds
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = (z_min + z_max) / 2

        # Calculate distance based on model size
        size = max(x_max - x_min, y_max - y_min, z_max - z_min)
        distance = size * 2.5

        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(center_x, center_y, center_z)

        if view_type == 'front':
            # Looking from +Y towards -Y
            camera.SetPosition(center_x, center_y + distance, center_z)
            camera.SetViewUp(0, 0, 1)
        elif view_type == 'top':
            # Looking from +Z down to -Z
            camera.SetPosition(center_x, center_y, center_z + distance)
            camera.SetViewUp(0, 1, 0)
        elif view_type == 'bottom':
            # Looking from -Z up to +Z
            camera.SetPosition(center_x, center_y, center_z - distance)
            camera.SetViewUp(0, -1, 0)
        elif view_type == 'left':
            # Looking from -X towards +X
            camera.SetPosition(center_x - distance, center_y, center_z)
            camera.SetViewUp(0, 0, 1)
        elif view_type == 'right':
            # Looking from +X towards -X
            camera.SetPosition(center_x + distance, center_y, center_z)
            camera.SetViewUp(0, 0, 1)
        elif view_type == 'iso':
            # Isometric view
            camera.SetPosition(center_x + distance, center_y + distance, center_z + distance)
            camera.SetViewUp(0, 0, 1)

        self.renderer.ResetCamera()
        if self._is_orthographic:
            self._update_parallel_scale()
        self.render_window.Render()

    def reset_camera(self):
        """Reset camera to default isometric view centered on the model"""
        if self.model_bounds is None:
            # No model loaded, just reset to origin
            self.renderer.ResetCamera()
            self.render_window.Render()
            return

        # Use isometric view as default
        self.set_camera_view('iso')

    def closeEvent(self, event):
        super().closeEvent(event)
