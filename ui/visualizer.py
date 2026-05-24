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

import numpy as np
import vtk
from vtk.util import numpy_support
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from typing import Optional, Dict, List, Tuple, Callable, Set
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import os
import tempfile
import sys

# Error logging
try:
    from core.error_logger import logger
except ImportError:
    # Fallback if error_logger not available
    class DummyLogger:
        def info(self, msg, **kwargs): print(f"INFO: {msg}")
        def warning(self, msg, **kwargs): print(f"WARNING: {msg}")
        def error(self, msg, **kwargs): print(f"ERROR: {msg}")
        def exception(self, msg, **kwargs): print(f"EXCEPTION: {msg}")
    logger = DummyLogger()


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
        'description': 'MAGMA-style feed modulus scale in inches'
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
        """Set plane normal based on axis"""
        if self.axis == 'X':
            self.plane.SetNormal(1, 0, 0)
        elif self.axis == 'Y':
            self.plane.SetNormal(0, 1, 0)
        else:  # Z
            self.plane.SetNormal(0, 0, 1)
        
        self.plane.SetOrigin(0, 0, 0)
    
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
                                 intensities: np.ndarray, threshold: float) -> float:
        """
        Calculate total volume of defect voxels above minimum risk.
        
        Args:
            voxel_centers: Nx3 array of voxel centers
            voxel_sizes: N array of voxel sizes
            intensities: N array of intensity values
            threshold: Minimum risk as 0-1 fraction (0.7 = 70%-100% risk)
            
        Returns:
            Total volume in cubic inches
        """
        mask = intensities >= threshold
        defect_sizes = voxel_sizes[mask]
        total_volume = np.sum(defect_sizes ** 3)  # Volume of cubes
        return total_volume


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

    MAX_CACHED_THRESHOLD_LEVELS = 4
    MAX_CACHED_DISPLAY_VOXELS = 20_000
    
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
        
        # Shrinkage actor
        self.shrinkage_actor = None
        
        # Hot spots actor
        self.hotspot_actor = None
        
        # Niyama actor
        self.niyama_actor = None
        
        # Hot tear actor
        self.hot_tear_actor = None
        
        # Sand burn-in actor
        self.sand_burn_in_actor = None
        
        # Fraction liquid actor
        self.fraction_liquid_actor = None
        
        # Data
        self.current_mesh = None
        self.current_shrinkage_data = None
        self.current_hotspot_data = None
        self.current_niyama_data = None
        self.current_hot_tear_data = None
        self.current_sand_burn_in_data = None
        self.current_fraction_liquid_data = None
        
        # Pre-rendered shrinkage cache for fast threshold updates
        # Maps threshold levels (0.05, 0.10, ..., 1.0) to pre-built VTK polydata
        self._shrinkage_cache: Dict[float, vtk.vtkPolyData] = {}
        self._shrinkage_cache_built = False
        self._cache_threshold_levels = [i / 100.0 for i in range(1, 101, 1)]  # 1% to 100%
        self._shrinkage_cache_lock = threading.Lock()
        self._shrinkage_cache_generation = 0
        self._pending_shrinkage_thresholds: Set[float] = set()
        self._cache_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="viz-cache")
        
        # Caches for other analysis types
        self._hotspot_cache: Dict[float, vtk.vtkPolyData] = {}
        self._niyama_cache: Dict[float, vtk.vtkPolyData] = {}
        self._hot_tear_cache: Dict[float, vtk.vtkPolyData] = {}
        self._sand_burn_in_cache: Dict[float, vtk.vtkPolyData] = {}
        
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
        
        # Animation export
        self._animation_frames: List[np.ndarray] = []
        self._is_recording = False
        self._interactor_initialized = False
        
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
        self.renderer.SetBackground(0.1, 0.1, 0.15)  # Dark blue-grey
        
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        
        self.interactor = self.render_window.GetInteractor()
        
        # Setup camera
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(200, 200, 200)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
        
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
        
        # Add click handler for measurement tool
        self.interactor.AddObserver("LeftButtonPressEvent", self._on_left_button_press)
        
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
        """Handle left mouse button press for measurement picking"""
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
        
    def update_legend(self, title: str, min_val: float, max_val: float, units: str = ""):
        """
        Update the legend title and scale
        
        Args:
            title: Legend title (e.g., "Shrinkage/Porosity Risk")
            min_val: Minimum value for scale
            max_val: Maximum value for scale
            units: Units string (e.g., "MPa", "μm")
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
    
    def enable_cross_section(self, axis: str, position: float = None, show_plane: bool = True):
        """
        Enable a cross-section cutting plane
        
        Args:
            axis: 'X', 'Y', or 'Z'
            position: Position along the axis (None = middle)
            show_plane: Whether to show the visual plane indicator
        """
        axis = axis.upper()
        if axis not in self.cross_section_planes:
            return
        
        plane = self.cross_section_planes[axis]
        
        # Update bounds if we have model bounds
        if self.model_bounds:
            plane.set_bounds(self.model_bounds)
        
        # Set position (default to middle)
        if position is None:
            min_pos, max_pos = plane.get_position_range()
            position = (min_pos + max_pos) / 2
        
        plane.set_position(position)
        plane.enable(show_plane)
        
        self._cross_section_enabled = True
        self._apply_cross_section_clipping()
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
        # Collect enabled planes
        enabled_planes = [p for p in self.cross_section_planes.values() if p.enabled]
        
        if not enabled_planes:
            return
        
        # Apply clipping to shrinkage actor
        if self.shrinkage_actor:
            self._clip_actor(self.shrinkage_actor, enabled_planes)
        
        # Apply to other analysis actors
        if self.hotspot_actor:
            self._clip_actor(self.hotspot_actor, enabled_planes)
        if self.niyama_actor:
            self._clip_actor(self.niyama_actor, enabled_planes)
        if self.fraction_liquid_actor:
            self._clip_actor(self.fraction_liquid_actor, enabled_planes)
        
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
        actors = [self.shrinkage_actor, self.hotspot_actor, self.niyama_actor, 
                  self.fraction_liquid_actor, self.mesh_actor]
        actors.extend(self.part_actors.values())
        
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
        return self.measurement_tool.calculate_defect_volume(
            voxel_centers, voxel_sizes, intensities, threshold
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
        """Refresh the current visualization with new colormap settings"""
        # Re-apply colors to current shrinkage data
        if self.current_shrinkage_data is not None:
            # Clear cache to force rebuild with new colormap
            self._reset_shrinkage_cache()
        if self.current_hotspot_data is not None:
            self._hotspot_cache.clear()
        if self.current_niyama_data is not None:
            self._niyama_cache.clear()

    def _reset_shrinkage_cache(self) -> None:
        """Invalidate cached shrinkage geometry after data or palette changes."""
        with self._shrinkage_cache_lock:
            self._shrinkage_cache_generation += 1
            self._shrinkage_cache.clear()
            self._pending_shrinkage_thresholds.clear()
        self._shrinkage_cache_built = False

    def _clear_result_caches(self) -> None:
        """Release cached polydata for all result visualization modes."""
        self._reset_shrinkage_cache()
        self._hotspot_cache.clear()
        self._niyama_cache.clear()
        self._hot_tear_cache.clear()
        self._sand_burn_in_cache.clear()

    def _get_shrinkage_threshold_levels(self, current_threshold: float) -> List[float]:
        """Return nearby threshold levels that should be pre-built."""
        return self._get_bounded_threshold_levels(current_threshold)

    def _get_bounded_threshold_levels(self, current_threshold: float) -> List[float]:
        """Return at most a few thresholds nearest to the current slider value."""
        current_key = round(current_threshold, 2)
        levels = sorted(
            (
                round(t, 2)
                for t in self._cache_threshold_levels
                if round(t, 2) != current_key
            ),
            key=lambda t: (abs(t - current_threshold), t),
        )
        return levels[:self.MAX_CACHED_THRESHOLD_LEVELS]

    def _display_voxel_count(
        self,
        values: np.ndarray,
        threshold: float,
        base_mask: Optional[np.ndarray] = None,
    ) -> int:
        """Count displayed voxels for direct cutoff visualizations."""
        mask = values >= threshold
        if base_mask is not None:
            mask = np.asarray(base_mask, dtype=bool) & mask
        return int(np.count_nonzero(mask))

    def _should_cache_threshold(
        self,
        values: np.ndarray,
        threshold: float,
        base_mask: Optional[np.ndarray] = None,
    ) -> bool:
        """Skip cached polydata for thresholds that would retain large geometry."""
        return self._display_voxel_count(values, threshold, base_mask) <= self.MAX_CACHED_DISPLAY_VOXELS

    def _cache_polydata(
        self,
        cache: Dict[float, vtk.vtkPolyData],
        key: float,
        polydata: Optional[vtk.vtkPolyData],
    ) -> None:
        if polydata is None:
            return
        cached = vtk.vtkPolyData()
        cached.DeepCopy(polydata)
        cache[key] = cached

    def _build_shrinkage_cache_entries(
        self,
        voxel_centers: np.ndarray,
        voxel_sizes: np.ndarray,
        shrinkage_intensity: np.ndarray,
        levels: List[float]
    ) -> Dict[float, vtk.vtkPolyData]:
        """Build shrinkage cache entries in a worker thread."""
        built_entries: Dict[float, vtk.vtkPolyData] = {}
        for thresh in levels:
            if not self._should_cache_threshold(shrinkage_intensity, thresh):
                continue
            polydata = self._build_shrinkage_geometry_fast(
                voxel_centers,
                voxel_sizes,
                shrinkage_intensity,
                thresh
            )
            if polydata is not None:
                cached = vtk.vtkPolyData()
                cached.DeepCopy(polydata)
                built_entries[thresh] = cached
        return built_entries
    
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
        property.SetInterpolationToPhong()
        property.SetSpecular(0.3)
        property.SetSpecularPower(20)
        
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
        
        # Reset camera
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()
    
    def load_shrinkage(self, voxel_centers: np.ndarray, 
                      voxel_sizes: np.ndarray,
                      shrinkage_intensity: np.ndarray,
                      threshold: float = 0.3):
        """
        Load and display shrinkage regions using optimized batch rendering.
        Pre-builds geometry cache for fast threshold slider updates.
        
        Args:
            voxel_centers: Nx3 array of voxel centers
            voxel_sizes: N array of voxel sizes
            shrinkage_intensity: N array of shrinkage intensity (0-1)
            threshold: Minimum risk as 0-1 fraction (0.7 = 70%-100% risk)
        """
        # Remove old shrinkage actor
        if self.shrinkage_actor is not None:
            self.renderer.RemoveActor(self.shrinkage_actor)
        
        # Store data for cache building
        self.current_shrinkage_data = (voxel_centers, voxel_sizes, shrinkage_intensity)
        
        # Clear old result caches when loading new data
        self._clear_result_caches()
        
        # Build geometry for current threshold using optimized method
        polydata = self._build_shrinkage_geometry_fast(
            voxel_centers, voxel_sizes, shrinkage_intensity, threshold
        )
        
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            print("No shrinkage voxels to display")
            return
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarModeToUsePointData()
        
        self.shrinkage_actor = vtk.vtkActor()
        self.shrinkage_actor.SetMapper(mapper)
        
        # Set appearance
        prop = self.shrinkage_actor.GetProperty()
        prop.SetOpacity(0.5)
        prop.SetInterpolationToFlat()
        
        self.renderer.AddActor(self.shrinkage_actor)
        self.render_window.Render()
        
        # Count voxels displayed
        num_displayed = polydata.GetNumberOfCells() // 6  # Each cube has 6 faces
        print(f"Displayed {num_displayed:,} shrinkage voxels")
        
        # Pre-build cache for nearby threshold levels in background
        self._prebuild_cache_async(voxel_centers, voxel_sizes, shrinkage_intensity, threshold)
    
    def _build_shrinkage_geometry_fast(self, voxel_centers: np.ndarray,
                                        voxel_sizes: np.ndarray,
                                        shrinkage_intensity: np.ndarray,
                                        threshold: float) -> Optional[vtk.vtkPolyData]:
        """
        Build shrinkage visualization geometry using optimized vtkGlyph3D.
        Uses NumPy vectorization and VTK numpy_support for zero-copy data transfer.
        Supports custom colormaps.
        
        Args:
            voxel_centers: Nx3 array of voxel centers
            voxel_sizes: N array of voxel sizes
            shrinkage_intensity: N array of shrinkage intensity (0-1)
            threshold: Minimum risk as 0-1 fraction (0.7 = 70%-100% risk)
            
        Returns:
            VTK polydata with colored cubes, or None if no voxels to display
        """
        # Shrinkage uses direct risk cutoff: 70% means show 0.70-1.00 risk voxels.
        mask = shrinkage_intensity >= threshold
        filtered_centers = voxel_centers[mask]
        filtered_sizes = voxel_sizes[mask]
        filtered_intensities = shrinkage_intensity[mask]
        
        n_voxels = len(filtered_centers)
        if n_voxels == 0:
            return None
        
        # Use numpy_support for zero-copy data transfer to VTK
        # Ensure contiguous float32 array for VTK compatibility
        centers_f32 = np.ascontiguousarray(filtered_centers, dtype=np.float32)
        vtk_points_array = numpy_support.numpy_to_vtk(centers_f32, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_points_array.SetNumberOfComponents(3)
        
        points = vtk.vtkPoints()
        points.SetData(vtk_points_array)
        
        # Create scale array using numpy_support (cube size * 0.8)
        scaled_sizes = np.ascontiguousarray(filtered_sizes * 0.8, dtype=np.float32)
        vtk_scales = numpy_support.numpy_to_vtk(scaled_sizes, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_scales.SetName("Scales")
        
        # Create color array using current colormap
        color_array = apply_colormap_to_array(
            filtered_intensities, 
            self._current_colormap,
            self._color_scale_min,
            self._color_scale_max
        )
        
        vtk_colors = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(color_array), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
        )
        vtk_colors.SetName("Colors")
        vtk_colors.SetNumberOfComponents(3)
        
        # Create polydata with points
        pointsPolydata = vtk.vtkPolyData()
        pointsPolydata.SetPoints(points)
        pointsPolydata.GetPointData().SetScalars(vtk_colors)
        pointsPolydata.GetPointData().AddArray(vtk_scales)
        
        # Create cube source for glyph (single cube used as template)
        cubeSource = vtk.vtkCubeSource()
        cubeSource.SetXLength(1.0)
        cubeSource.SetYLength(1.0)
        cubeSource.SetZLength(1.0)
        cubeSource.Update()
        
        # Use vtkGlyph3D for batch rendering - stamps cube at each point
        glyph3D = vtk.vtkGlyph3D()
        glyph3D.SetSourceConnection(cubeSource.GetOutputPort())
        glyph3D.SetInputData(pointsPolydata)
        glyph3D.SetScaleModeToScaleByScalar()
        glyph3D.SetInputArrayToProcess(0, 0, 0, 0, "Scales")
        glyph3D.SetColorModeToColorByScalar()
        glyph3D.Update()
        
        return glyph3D.GetOutput()
    
    def _prebuild_cache_async(self, voxel_centers: np.ndarray,
                               voxel_sizes: np.ndarray, 
                               shrinkage_intensity: np.ndarray,
                               current_threshold: float):
        """
        Pre-build geometry cache for threshold levels near current value.
        This enables instant threshold slider updates.
        
        Args:
            voxel_centers: Voxel center positions
            voxel_sizes: Voxel sizes
            shrinkage_intensity: Intensity values
            current_threshold: Current threshold to build around
        """
        with self._shrinkage_cache_lock:
            generation = self._shrinkage_cache_generation
            levels_to_cache = [
                t for t in self._get_shrinkage_threshold_levels(current_threshold)
                if t not in self._shrinkage_cache and t not in self._pending_shrinkage_thresholds
                and self._should_cache_threshold(shrinkage_intensity, t)
            ]
            if not levels_to_cache:
                return
            self._pending_shrinkage_thresholds.update(levels_to_cache)

        future = self._cache_executor.submit(
            self._build_shrinkage_cache_entries,
            voxel_centers,
            voxel_sizes,
            shrinkage_intensity,
            levels_to_cache
        )

        def _cache_done(completed_future):
            try:
                built_entries = completed_future.result()
            except Exception as exc:
                logger.warning(
                    "Background shrinkage cache build failed",
                    context={"error": str(exc)}
                )
                built_entries = {}

            with self._shrinkage_cache_lock:
                self._pending_shrinkage_thresholds.difference_update(levels_to_cache)
                if generation != self._shrinkage_cache_generation:
                    return
                self._shrinkage_cache.update(built_entries)
                if self._shrinkage_cache:
                    self._shrinkage_cache_built = True

        future.add_done_callback(_cache_done)
    
    def _prebuild_hotspot_cache(self, voxel_centers: np.ndarray,
                                 voxel_sizes: np.ndarray,
                                 hotspot_intensity: np.ndarray,
                                 current_threshold: float):
        """Pre-build hotspot geometry cache for nearby threshold levels"""
        levels_to_cache = [
            t for t in self._get_bounded_threshold_levels(current_threshold)
            if t not in self._hotspot_cache
            and self._should_cache_threshold(hotspot_intensity, t)
        ]
        
        for thresh in levels_to_cache:
            polydata = self._build_hotspot_geometry(
                voxel_centers, voxel_sizes, hotspot_intensity, thresh
            )
            self._cache_polydata(self._hotspot_cache, thresh, polydata)
    
    def _prebuild_niyama_cache(self, voxel_centers: np.ndarray,
                                voxel_sizes: np.ndarray,
                                niyama_values: np.ndarray,
                                defect_risk: np.ndarray,
                                current_threshold: float):
        """Pre-build Niyama geometry cache for nearby threshold levels"""
        levels_to_cache = [
            t for t in self._get_bounded_threshold_levels(current_threshold)
            if t not in self._niyama_cache
            and self._should_cache_threshold(defect_risk, t)
        ]
        
        for thresh in levels_to_cache:
            polydata = self._build_niyama_geometry(
                voxel_centers, voxel_sizes, niyama_values, defect_risk, thresh
            )
            self._cache_polydata(self._niyama_cache, thresh, polydata)

    def load_hotspots(self, voxel_centers: np.ndarray,
                      voxel_sizes: np.ndarray,
                      hotspot_intensity: np.ndarray,
                      threshold: float = 0.5):
        """
        Load and display hot spot regions using orange-yellow color gradient.
        Hot spots are areas that solidify last and may have feeding issues.
        
        Args:
            voxel_centers: Nx3 array of voxel centers
            voxel_sizes: N array of voxel sizes
            hotspot_intensity: N array of hot spot intensity (0-1)
            threshold: Minimum intensity as 0-1 fraction (0.7 = 70%-100%)
        """
        # Remove old hotspot actor
        if self.hotspot_actor is not None:
            self.renderer.RemoveActor(self.hotspot_actor)
        
        # Store data for cache building and threshold updates
        self.current_hotspot_data = (voxel_centers, voxel_sizes, hotspot_intensity)
        
        # Clear stale result caches when switching modes or loading new data
        self._clear_result_caches()
        
        # Build geometry for current threshold
        polydata = self._build_hotspot_geometry(
            voxel_centers, voxel_sizes, hotspot_intensity, threshold
        )
        
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            print("No hot spot voxels to display")
            return
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarModeToUsePointData()
        
        self.hotspot_actor = vtk.vtkActor()
        self.hotspot_actor.SetMapper(mapper)
        
        # Set appearance
        prop = self.hotspot_actor.GetProperty()
        prop.SetOpacity(0.6)
        prop.SetInterpolationToFlat()
        
        self.renderer.AddActor(self.hotspot_actor)
        self.render_window.Render()
        
        # Count voxels displayed
        num_displayed = polydata.GetNumberOfCells() // 6
        print(f"Displayed {num_displayed:,} hot spot voxels")
        
        # Pre-build cache for nearby threshold levels
        self._prebuild_hotspot_cache(voxel_centers, voxel_sizes, hotspot_intensity, threshold)
    
    def _build_hotspot_geometry(self, voxel_centers: np.ndarray,
                                 voxel_sizes: np.ndarray,
                                 hotspot_intensity: np.ndarray,
                                 threshold: float) -> Optional[vtk.vtkPolyData]:
        """
        Build hot spot visualization geometry.
        Color gradient: Yellow (low intensity) -> Orange -> Red (high intensity)
        
        Args:
            voxel_centers: Nx3 array of voxel centers
            voxel_sizes: N array of voxel sizes
            hotspot_intensity: N array of hot spot intensity (0-1)
            threshold: Top percentage as 0-1 fraction (0.2 = top 20%)
            
        Returns:
            VTK polydata with colored cubes
        """
        # Direct intensity cutoff: 70% means show 0.70-1.00.
        mask = hotspot_intensity >= threshold
        filtered_centers = voxel_centers[mask]
        filtered_sizes = voxel_sizes[mask]
        filtered_intensities = hotspot_intensity[mask]
        
        n_voxels = len(filtered_centers)
        if n_voxels == 0:
            return None
        
        # Create VTK points
        centers_f32 = np.ascontiguousarray(filtered_centers, dtype=np.float32)
        vtk_points_array = numpy_support.numpy_to_vtk(centers_f32, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_points_array.SetNumberOfComponents(3)
        
        points = vtk.vtkPoints()
        points.SetData(vtk_points_array)
        
        # Create scale array
        scaled_sizes = np.ascontiguousarray(filtered_sizes * 0.8, dtype=np.float32)
        vtk_scales = numpy_support.numpy_to_vtk(scaled_sizes, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_scales.SetName("Scales")
        
        # Create color array: Yellow (255,255,0) -> Orange (255,165,0) -> Red (255,0,0)
        clamped_intensity = np.clip(filtered_intensities, 0, 1)
        color_array = np.zeros((n_voxels, 3), dtype=np.uint8)
        color_array[:, 0] = 255  # Red always 255
        # Green decreases with intensity: 255 at low, 0 at high
        color_array[:, 1] = (255 * (1 - clamped_intensity)).astype(np.uint8)
        # Blue stays 0
        
        vtk_colors = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(color_array), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
        )
        vtk_colors.SetName("Colors")
        vtk_colors.SetNumberOfComponents(3)
        
        # Create polydata with points
        pointsPolydata = vtk.vtkPolyData()
        pointsPolydata.SetPoints(points)
        pointsPolydata.GetPointData().SetScalars(vtk_colors)
        pointsPolydata.GetPointData().AddArray(vtk_scales)
        
        # Create cube source for glyph
        cubeSource = vtk.vtkCubeSource()
        cubeSource.SetXLength(1.0)
        cubeSource.SetYLength(1.0)
        cubeSource.SetZLength(1.0)
        cubeSource.Update()
        
        # Use vtkGlyph3D for batch rendering
        glyph3D = vtk.vtkGlyph3D()
        glyph3D.SetSourceConnection(cubeSource.GetOutputPort())
        glyph3D.SetInputData(pointsPolydata)
        glyph3D.SetScaleModeToScaleByScalar()
        glyph3D.SetInputArrayToProcess(0, 0, 0, 0, "Scales")
        glyph3D.SetColorModeToColorByScalar()
        glyph3D.Update()
        
        return glyph3D.GetOutput()
    
    def load_niyama(self, voxel_centers: np.ndarray,
                    voxel_sizes: np.ndarray,
                    niyama_values: np.ndarray,
                    defect_risk: np.ndarray,
                    threshold: float = 0.5):
        """
        Load and display Niyama criterion results.
        Low Niyama (high defect risk) shown in red, high Niyama (low risk) in blue.
        
        Args:
            voxel_centers: Nx3 array of voxel centers
            voxel_sizes: N array of voxel sizes
            niyama_values: N array of Niyama criterion values (G/sqrt(R))
            defect_risk: N array of defect risk (0-1), where 1 = high risk
            threshold: Minimum defect risk as 0-1 fraction (0.7 = 70%-100%)
        """
        # Remove old niyama actor
        if self.niyama_actor is not None:
            self.renderer.RemoveActor(self.niyama_actor)
        
        # Store data for cache building and threshold updates
        self.current_niyama_data = (voxel_centers, voxel_sizes, niyama_values, defect_risk)
        
        # Clear stale result caches when switching modes or loading new data
        self._clear_result_caches()
        
        # Build geometry for current threshold
        polydata = self._build_niyama_geometry(
            voxel_centers, voxel_sizes, niyama_values, defect_risk, threshold
        )
        
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            print("No Niyama defect risk voxels to display")
            return
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarModeToUsePointData()
        
        self.niyama_actor = vtk.vtkActor()
        self.niyama_actor.SetMapper(mapper)
        
        # Set appearance
        prop = self.niyama_actor.GetProperty()
        prop.SetOpacity(0.6)
        prop.SetInterpolationToFlat()
        
        self.renderer.AddActor(self.niyama_actor)
        self.render_window.Render()
        
        # Count voxels displayed
        num_displayed = polydata.GetNumberOfCells() // 6
        print(f"Displayed {num_displayed:,} Niyama defect risk voxels")
        
        # Pre-build cache for nearby threshold levels
        self._prebuild_niyama_cache(voxel_centers, voxel_sizes, niyama_values, defect_risk, threshold)
    
    def _build_niyama_geometry(self, voxel_centers: np.ndarray,
                                voxel_sizes: np.ndarray,
                                niyama_values: np.ndarray,
                                defect_risk: np.ndarray,
                                threshold: float) -> Optional[vtk.vtkPolyData]:
        """
        Build Niyama visualization geometry.
        Color gradient: Blue (low risk) -> Cyan -> Yellow -> Orange -> Red (high risk)
        
        Args:
            voxel_centers: Nx3 array of voxel centers
            voxel_sizes: N array of voxel sizes
            niyama_values: N array of Niyama values
            defect_risk: N array of defect risk (0-1)
            threshold: Top percentage as 0-1 fraction (0.2 = top 20%)
            
        Returns:
            VTK polydata with colored cubes
        """
        # Direct risk cutoff: 70% means show 0.70-1.00.
        mask = defect_risk >= threshold
        filtered_centers = voxel_centers[mask]
        filtered_sizes = voxel_sizes[mask]
        filtered_risk = defect_risk[mask]
        
        n_voxels = len(filtered_centers)
        if n_voxels == 0:
            return None
        
        # Create VTK points
        centers_f32 = np.ascontiguousarray(filtered_centers, dtype=np.float32)
        vtk_points_array = numpy_support.numpy_to_vtk(centers_f32, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_points_array.SetNumberOfComponents(3)
        
        points = vtk.vtkPoints()
        points.SetData(vtk_points_array)
        
        # Create scale array
        scaled_sizes = np.ascontiguousarray(filtered_sizes * 0.8, dtype=np.float32)
        vtk_scales = numpy_support.numpy_to_vtk(scaled_sizes, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_scales.SetName("Scales")
        
        # Create color array using blue-cyan-yellow-red gradient based on risk
        # High risk = red (255,0,0), Low risk = blue (0,0,255)
        clamped_risk = np.clip(filtered_risk, 0, 1)
        color_array = np.zeros((n_voxels, 3), dtype=np.uint8)
        
        # Blue-to-red gradient through cyan and yellow
        # 0.0-0.33: Blue to Cyan (increase G)
        # 0.33-0.66: Cyan to Yellow (R up, B down)
        # 0.66-1.0: Yellow to Red (G down)
        for i, risk in enumerate(clamped_risk):
            if risk < 0.33:
                # Blue to Cyan
                t = risk / 0.33
                color_array[i] = [0, int(255 * t), 255]
            elif risk < 0.66:
                # Cyan to Yellow
                t = (risk - 0.33) / 0.33
                color_array[i] = [int(255 * t), 255, int(255 * (1 - t))]
            else:
                # Yellow to Red
                t = (risk - 0.66) / 0.34
                color_array[i] = [255, int(255 * (1 - t)), 0]
        
        vtk_colors = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(color_array), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
        )
        vtk_colors.SetName("Colors")
        vtk_colors.SetNumberOfComponents(3)
        
        # Create polydata with points
        pointsPolydata = vtk.vtkPolyData()
        pointsPolydata.SetPoints(points)
        pointsPolydata.GetPointData().SetScalars(vtk_colors)
        pointsPolydata.GetPointData().AddArray(vtk_scales)
        
        # Create cube source for glyph
        cubeSource = vtk.vtkCubeSource()
        cubeSource.SetXLength(1.0)
        cubeSource.SetYLength(1.0)
        cubeSource.SetZLength(1.0)
        cubeSource.Update()
        
        # Use vtkGlyph3D for batch rendering
        glyph3D = vtk.vtkGlyph3D()
        glyph3D.SetSourceConnection(cubeSource.GetOutputPort())
        glyph3D.SetInputData(pointsPolydata)
        glyph3D.SetScaleModeToScaleByScalar()
        glyph3D.SetInputArrayToProcess(0, 0, 0, 0, "Scales")
        glyph3D.SetColorModeToColorByScalar()
        glyph3D.Update()
        
        return glyph3D.GetOutput()
    
    def load_fraction_liquid(self, voxel_centers: np.ndarray,
                             voxel_sizes: np.ndarray,
                             fraction_liquid: np.ndarray,
                             threshold: float = 0.01):
        """
        Load and display fraction liquid visualization.
        Shows currently liquid regions during solidification.
        
        Args:
            voxel_centers: Nx3 array of voxel centers
            voxel_sizes: N array of voxel sizes
            fraction_liquid: N array of liquid fraction (0-1)
            threshold: Minimum liquid fraction to display
        """
        # Remove old fraction liquid actor
        if self.fraction_liquid_actor is not None:
            self.renderer.RemoveActor(self.fraction_liquid_actor)
        
        # Store data
        self.current_fraction_liquid_data = (voxel_centers, voxel_sizes, fraction_liquid)
        self._clear_result_caches()
        
        # Build geometry
        polydata = self._build_fraction_liquid_geometry(
            voxel_centers, voxel_sizes, fraction_liquid, threshold
        )
        
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            print("No liquid voxels to display")
            return
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarModeToUsePointData()
        
        self.fraction_liquid_actor = vtk.vtkActor()
        self.fraction_liquid_actor.SetMapper(mapper)
        
        # Set appearance - semi-transparent
        prop = self.fraction_liquid_actor.GetProperty()
        prop.SetOpacity(0.5)
        prop.SetInterpolationToFlat()
        
        self.renderer.AddActor(self.fraction_liquid_actor)
        self.render_window.Render()
        
        # Count voxels displayed
        num_displayed = polydata.GetNumberOfCells() // 6
        print(f"Displayed {num_displayed:,} liquid voxels")
    
    def _build_fraction_liquid_geometry(self, voxel_centers: np.ndarray,
                                         voxel_sizes: np.ndarray,
                                         fraction_liquid: np.ndarray,
                                         threshold: float) -> Optional[vtk.vtkPolyData]:
        """
        Build fraction liquid visualization geometry.
        
        Foundry physics color convention:
        - YELLOW/ORANGE = liquid metal (molten, hot - like glowing steel)
        - TRANSPARENT = solid metal (cooled, no longer visible)
        
        The slider shows solidification progression:
        - Left (t=0): All liquid (yellow) - just poured
        - Right (t=max): All solid (transparent) - fully solidified
        
        Solidification follows foundry physics:
        - Edge-inward: Surfaces solidify first (contact with mold)
        - Bottom-up: Lower sections solidify faster (density/gravity)
        - Hot spots: Interior thick sections stay liquid longest (shrinkage risk)
        
        Args:
            voxel_centers: Nx3 array of voxel centers
            voxel_sizes: N array of voxel sizes
            fraction_liquid: N array of liquid fraction (0=solid, 1=liquid)
            threshold: Minimum liquid fraction to display (solid voxels hidden)
            
        Returns:
            VTK polydata with colored cubes (only liquid voxels shown)
        """
        # Filter voxels by threshold - only show liquid voxels (solid becomes transparent)
        mask = fraction_liquid > threshold
        filtered_centers = voxel_centers[mask]
        filtered_sizes = voxel_sizes[mask]
        filtered_liquid = fraction_liquid[mask]
        
        n_voxels = len(filtered_centers)
        if n_voxels == 0:
            return None
        
        # Create VTK points
        centers_f32 = np.ascontiguousarray(filtered_centers, dtype=np.float32)
        vtk_points_array = numpy_support.numpy_to_vtk(centers_f32, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_points_array.SetNumberOfComponents(3)
        
        points = vtk.vtkPoints()
        points.SetData(vtk_points_array)
        
        # Create scale array
        scaled_sizes = np.ascontiguousarray(filtered_sizes * 0.8, dtype=np.float32)
        vtk_scales = numpy_support.numpy_to_vtk(scaled_sizes, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_scales.SetName("Scales")
        
        # Create color array: Yellow/Orange gradient for liquid metal
        # Foundry convention: molten metal glows yellow-orange like hot steel
        # High liquid fraction = bright yellow (fully molten)
        # Low liquid fraction = darker orange (mushy zone, about to solidify)
        clamped_liquid = np.clip(filtered_liquid, 0, 1)
        color_array = np.zeros((n_voxels, 3), dtype=np.uint8)
        # R: High for all liquid (yellow-orange base)
        color_array[:, 0] = (200 + 55 * clamped_liquid).astype(np.uint8)  # 200-255
        # G: Varies with liquid fraction (yellow when high, orange when low)
        color_array[:, 1] = (100 + 155 * clamped_liquid).astype(np.uint8)  # 100-255
        # B: Low (no blue in molten metal)
        color_array[:, 2] = (30 * clamped_liquid).astype(np.uint8)  # 0-30
        
        vtk_colors = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(color_array), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
        )
        vtk_colors.SetName("Colors")
        vtk_colors.SetNumberOfComponents(3)
        
        # Create polydata with points
        pointsPolydata = vtk.vtkPolyData()
        pointsPolydata.SetPoints(points)
        pointsPolydata.GetPointData().SetScalars(vtk_colors)
        pointsPolydata.GetPointData().AddArray(vtk_scales)
        
        # Create cube source for glyph
        cubeSource = vtk.vtkCubeSource()
        cubeSource.SetXLength(1.0)
        cubeSource.SetYLength(1.0)
        cubeSource.SetZLength(1.0)
        cubeSource.Update()
        
        # Use vtkGlyph3D for batch rendering
        glyph3D = vtk.vtkGlyph3D()
        glyph3D.SetSourceConnection(cubeSource.GetOutputPort())
        glyph3D.SetInputData(pointsPolydata)
        glyph3D.SetScaleModeToScaleByScalar()
        glyph3D.SetInputArrayToProcess(0, 0, 0, 0, "Scales")
        glyph3D.SetColorModeToColorByScalar()
        glyph3D.Update()
        
        return glyph3D.GetOutput()
    
    def update_hotspot_threshold(self, threshold: float):
        """Update hot spot visualization threshold"""
        if self.current_hotspot_data is None:
            return
        
        voxel_centers, voxel_sizes, intensities = self.current_hotspot_data
        
        # Round threshold to cache key
        cache_key = round(threshold, 2)
        
        # Check cache
        if cache_key in self._hotspot_cache:
            polydata = self._hotspot_cache[cache_key]
        else:
            polydata = self._build_hotspot_geometry(
                voxel_centers, voxel_sizes, intensities, threshold
            )
        
        # Remove old actor
        if self.hotspot_actor is not None:
            self.renderer.RemoveActor(self.hotspot_actor)
        
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            print("No hot spots at this threshold")
            self.render_window.Render()
            return
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarModeToUsePointData()
        
        self.hotspot_actor = vtk.vtkActor()
        self.hotspot_actor.SetMapper(mapper)
        
        prop = self.hotspot_actor.GetProperty()
        prop.SetOpacity(0.6)
        prop.SetInterpolationToFlat()
        
        self.renderer.AddActor(self.hotspot_actor)
        self.render_window.Render()
        
        num_displayed = polydata.GetNumberOfCells() // 6
        print(f"Displayed {num_displayed:,} hot spot voxels")
        
        # Pre-build nearby thresholds for smoother slider movement
        self._prebuild_hotspot_cache(voxel_centers, voxel_sizes, intensities, threshold)
    
    def update_niyama_threshold(self, threshold: float):
        """Update Niyama visualization threshold"""
        if self.current_niyama_data is None:
            return
        
        voxel_centers, voxel_sizes, niyama_values, defect_risk = self.current_niyama_data
        
        # Round threshold to cache key
        cache_key = round(threshold, 2)
        
        # Check cache
        if cache_key in self._niyama_cache:
            polydata = self._niyama_cache[cache_key]
        else:
            polydata = self._build_niyama_geometry(
                voxel_centers, voxel_sizes, niyama_values, defect_risk, threshold
            )
        
        # Remove old actor
        if self.niyama_actor is not None:
            self.renderer.RemoveActor(self.niyama_actor)
        
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            print("No Niyama defect risk at this threshold")
            self.render_window.Render()
            return
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarModeToUsePointData()
        
        self.niyama_actor = vtk.vtkActor()
        self.niyama_actor.SetMapper(mapper)
        
        prop = self.niyama_actor.GetProperty()
        prop.SetOpacity(0.6)
        prop.SetInterpolationToFlat()
        
        self.renderer.AddActor(self.niyama_actor)
        self.render_window.Render()
        
        num_displayed = polydata.GetNumberOfCells() // 6
        print(f"Displayed {num_displayed:,} Niyama defect risk voxels")
        
        # Pre-build nearby thresholds for smoother slider movement
        self._prebuild_niyama_cache(voxel_centers, voxel_sizes, niyama_values, defect_risk, threshold)
    
    def load_hot_tear(self, voxel_centers: np.ndarray,
                      voxel_sizes: np.ndarray,
                      hot_tear_risk: np.ndarray,
                      hot_tear_zones: np.ndarray,
                      threshold: float = 0.1):
        """
        Load and display hot tear risk zones.
        Uses cyan-purple color scheme to distinguish from shrinkage (red) and hotspots (orange).
        
        Args:
            voxel_centers: Nx3 array of voxel centers
            voxel_sizes: N array of voxel sizes
            hot_tear_risk: N array of hot tear risk (0-1)
            hot_tear_zones: N boolean array of hot tear zone voxels
            threshold: Top percentage as 0-1 fraction (within hot tear zones)
        """
        # Remove old hot tear actor
        if self.hot_tear_actor is not None:
            self.renderer.RemoveActor(self.hot_tear_actor)
        
        # Store data for cache building and threshold updates
        self.current_hot_tear_data = (voxel_centers, voxel_sizes, hot_tear_risk, hot_tear_zones)
        
        # Clear stale result caches when switching modes or loading new data
        self._clear_result_caches()
        
        # Build geometry for current threshold
        polydata = self._build_hot_tear_geometry(
            voxel_centers, voxel_sizes, hot_tear_risk, hot_tear_zones, threshold
        )
        
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            print("No hot tear risk voxels to display")
            return
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarModeToUsePointData()
        
        self.hot_tear_actor = vtk.vtkActor()
        self.hot_tear_actor.SetMapper(mapper)
        
        # Set appearance
        prop = self.hot_tear_actor.GetProperty()
        prop.SetOpacity(0.7)
        prop.SetInterpolationToFlat()
        
        self.renderer.AddActor(self.hot_tear_actor)
        self.render_window.Render()
        
        # Count voxels displayed
        num_displayed = polydata.GetNumberOfCells() // 6
        print(f"Displayed {num_displayed:,} hot tear risk voxels")
        
        # Pre-build cache for nearby threshold levels
        self._prebuild_hot_tear_cache(voxel_centers, voxel_sizes, hot_tear_risk, hot_tear_zones, threshold)
    
    def _build_hot_tear_geometry(self, voxel_centers: np.ndarray,
                                  voxel_sizes: np.ndarray,
                                  hot_tear_risk: np.ndarray,
                                  hot_tear_zones: np.ndarray,
                                  threshold: float) -> Optional[vtk.vtkPolyData]:
        """
        Build hot tear visualization geometry.
        Color gradient: Cyan (low risk) -> Purple -> Magenta (high risk)
        This distinguishes from shrinkage (red) and hotspots (orange).
        
        Args:
            voxel_centers: Nx3 array of voxel centers
            voxel_sizes: N array of voxel sizes
            hot_tear_risk: N array of hot tear risk (0-1)
            hot_tear_zones: N boolean array of hot tear zone voxels
            threshold: Minimum hot tear risk as 0-1 fraction (within zones)
            
        Returns:
            VTK polydata with colored cubes
        """
        # Apply direct cutoff within physics-derived eligible zones.
        mask = hot_tear_zones & (hot_tear_risk >= threshold)
        filtered_centers = voxel_centers[mask]
        filtered_sizes = voxel_sizes[mask]
        filtered_risk = hot_tear_risk[mask]
        
        n_voxels = len(filtered_centers)
        if n_voxels == 0:
            return None
        
        # Create VTK points
        centers_f32 = np.ascontiguousarray(filtered_centers, dtype=np.float32)
        vtk_points_array = numpy_support.numpy_to_vtk(centers_f32, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_points_array.SetNumberOfComponents(3)
        
        points = vtk.vtkPoints()
        points.SetData(vtk_points_array)
        
        # Create scale array
        scaled_sizes = np.ascontiguousarray(filtered_sizes * 0.85, dtype=np.float32)
        vtk_scales = numpy_support.numpy_to_vtk(scaled_sizes, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_scales.SetName("Scale")
        
        # Create polydata for glyph source centers
        point_polydata = vtk.vtkPolyData()
        point_polydata.SetPoints(points)
        point_polydata.GetPointData().AddArray(vtk_scales)
        point_polydata.GetPointData().SetActiveScalars("Scale")
        
        # Create colors (Cyan -> Purple -> Magenta gradient)
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        # Normalize risk values relative to the filtered set for better color spread
        # This ensures we use the full color gradient even if absolute values are low
        min_risk = np.min(filtered_risk)
        max_risk = np.max(filtered_risk)
        risk_range = max_risk - min_risk if max_risk > min_risk else 1.0
        
        for risk in filtered_risk:
            # Normalize risk relative to the displayed voxels' range
            t = (risk - min_risk) / risk_range
            t = min(max(t, 0.0), 1.0)  # Clamp to 0-1
            
            # Cyan (0, 255, 255) -> Purple (128, 0, 255) -> Magenta (255, 0, 255)
            if t < 0.5:
                # Cyan to Purple
                t2 = t * 2  # 0-1 for first half
                r = int(128 * t2)
                g = int(255 * (1 - t2))
                b = 255
            else:
                # Purple to Magenta
                t2 = (t - 0.5) * 2  # 0-1 for second half
                r = int(128 + 127 * t2)
                g = 0
                b = 255
            
            colors.InsertNextTuple3(r, g, b)
        
        point_polydata.GetPointData().AddArray(colors)
        
        # Create cube source
        cube = vtk.vtkCubeSource()
        cube.SetXLength(1.0)
        cube.SetYLength(1.0)
        cube.SetZLength(1.0)
        
        # Use glyph3D for instanced rendering
        glyph3D = vtk.vtkGlyph3D()
        glyph3D.SetSourceConnection(cube.GetOutputPort())
        glyph3D.SetInputData(point_polydata)
        glyph3D.SetScaleModeToScaleByScalar()
        glyph3D.SetColorModeToColorByScalar()
        glyph3D.Update()
        
        return glyph3D.GetOutput()
    
    def _prebuild_hot_tear_cache(self, voxel_centers, voxel_sizes, hot_tear_risk, hot_tear_zones, current_threshold):
        """Pre-build geometry for nearby threshold levels"""
        nearby_thresholds = self._get_bounded_threshold_levels(current_threshold)
        
        for thresh in nearby_thresholds:
            cache_key = round(thresh, 2)
            if (
                cache_key not in self._hot_tear_cache
                and self._should_cache_threshold(hot_tear_risk, cache_key, hot_tear_zones)
            ):
                polydata = self._build_hot_tear_geometry(
                    voxel_centers, voxel_sizes, hot_tear_risk, hot_tear_zones, thresh
                )
                self._cache_polydata(self._hot_tear_cache, cache_key, polydata)
    
    def update_hot_tear_threshold(self, threshold: float):
        """Update hot tear visualization threshold"""
        if self.current_hot_tear_data is None:
            return
        
        voxel_centers, voxel_sizes, hot_tear_risk, hot_tear_zones = self.current_hot_tear_data
        
        # Round threshold to cache key
        cache_key = round(threshold, 2)
        
        # Check cache
        if cache_key in self._hot_tear_cache:
            polydata = self._hot_tear_cache[cache_key]
        else:
            polydata = self._build_hot_tear_geometry(
                voxel_centers, voxel_sizes, hot_tear_risk, hot_tear_zones, threshold
            )
        
        # Remove old actor
        if self.hot_tear_actor is not None:
            self.renderer.RemoveActor(self.hot_tear_actor)
        
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            print("No hot tear risk at this threshold")
            self.render_window.Render()
            return
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarModeToUsePointData()
        
        self.hot_tear_actor = vtk.vtkActor()
        self.hot_tear_actor.SetMapper(mapper)
        
        prop = self.hot_tear_actor.GetProperty()
        prop.SetOpacity(0.7)
        prop.SetInterpolationToFlat()
        
        self.renderer.AddActor(self.hot_tear_actor)
        self.render_window.Render()
        
        num_displayed = polydata.GetNumberOfCells() // 6
        print(f"Displayed {num_displayed:,} hot tear risk voxels")
        
        # Pre-build nearby thresholds for smoother slider movement
        self._prebuild_hot_tear_cache(voxel_centers, voxel_sizes, hot_tear_risk, hot_tear_zones, threshold)
    
    # ==================== SAND BURN-IN VISUALIZATION ====================
    
    def load_sand_burn_in(self, voxel_centers: np.ndarray,
                          voxel_sizes: np.ndarray,
                          sand_burn_in_risk: np.ndarray,
                          sand_burn_in_zones: np.ndarray,
                          threshold: float = 0.2):
        """
        Load and display sand burn-in visualization (surface defect zones).
        
        Sand burn-in is a surface defect where molten metal penetrates the sand mold.
        Only surface voxels are analyzed, so we use a warm color gradient to indicate
        surface damage severity.
        
        Color gradient: Tan (low) -> Orange -> Deep Brown (high risk)
        This color scheme evokes burned/scorched sand appearance.
        
        Args:
            voxel_centers: Nx3 array of voxel centers
            voxel_sizes: N array of voxel sizes
            sand_burn_in_risk: N array of sand burn-in risk (0-1)
            sand_burn_in_zones: N boolean array of sand burn-in zone voxels
            threshold: Top percentage as 0-1 fraction (within sand burn-in zones)
        """
        try:
            # Validate input arrays
            if voxel_centers is None or len(voxel_centers) == 0:
                logger.warning("load_sand_burn_in called with empty voxel_centers")
                print("Warning: No voxel data for sand burn-in visualization")
                return
            
            if sand_burn_in_risk is None or sand_burn_in_zones is None:
                logger.warning("load_sand_burn_in called with missing risk/zones data")
                print("Warning: Sand burn-in risk data not available")
                return
            
            # Remove old sand burn-in actor
            if self.sand_burn_in_actor is not None:
                self.renderer.RemoveActor(self.sand_burn_in_actor)
            
            # Store data for cache building and threshold updates
            self.current_sand_burn_in_data = (voxel_centers, voxel_sizes, sand_burn_in_risk, sand_burn_in_zones)
            
            # Clear stale result caches when switching modes or loading new data
            self._clear_result_caches()
            
            # Build geometry for current threshold
            polydata = self._build_sand_burn_in_geometry(
                voxel_centers, voxel_sizes, sand_burn_in_risk, sand_burn_in_zones, threshold
            )
            
            if polydata is None or polydata.GetNumberOfPoints() == 0:
                print("No sand burn-in risk voxels to display")
                return
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            mapper.SetScalarModeToUsePointData()
            
            self.sand_burn_in_actor = vtk.vtkActor()
            self.sand_burn_in_actor.SetMapper(mapper)
            
            # Set appearance
            prop = self.sand_burn_in_actor.GetProperty()
            prop.SetOpacity(0.8)  # Slightly higher opacity for surface defects
            prop.SetInterpolationToFlat()
            
            self.renderer.AddActor(self.sand_burn_in_actor)
            self.render_window.Render()
            
            # Count voxels displayed
            num_displayed = polydata.GetNumberOfCells() // 6
            print(f"Displayed {num_displayed:,} sand burn-in risk voxels")
            
            # Pre-build cache for nearby threshold levels
            self._prebuild_sand_burn_in_cache(voxel_centers, voxel_sizes, sand_burn_in_risk, sand_burn_in_zones, threshold)
            
        except Exception as e:
            logger.error(
                f"Failed to load sand burn-in visualization: {e}",
                exc_info=True,
                context={
                    'n_voxels': len(voxel_centers) if voxel_centers is not None else 0,
                    'threshold': threshold
                }
            )
            print(f"Error loading sand burn-in visualization: {e}")
    
    def _build_sand_burn_in_geometry(self, voxel_centers: np.ndarray,
                                      voxel_sizes: np.ndarray,
                                      sand_burn_in_risk: np.ndarray,
                                      sand_burn_in_zones: np.ndarray,
                                      threshold: float) -> Optional[vtk.vtkPolyData]:
        """
        Build sand burn-in visualization geometry.
        Color gradient: Tan (low risk) -> Orange -> Deep Brown (high risk)
        This evokes the appearance of burned/scorched sand on casting surfaces.
        
        Args:
            voxel_centers: Nx3 array of voxel centers
            voxel_sizes: N array of voxel sizes
            sand_burn_in_risk: N array of sand burn-in risk (0-1)
            sand_burn_in_zones: N boolean array of sand burn-in zone voxels
            threshold: Minimum sand burn-in risk as 0-1 fraction (within zones)
            
        Returns:
            VTK polydata with colored cubes, or None if no voxels to display
        """
        try:
            # Apply direct cutoff within burn-in eligibility zones.
            mask = sand_burn_in_zones & (sand_burn_in_risk >= threshold)
            filtered_centers = voxel_centers[mask]
            filtered_sizes = voxel_sizes[mask]
            filtered_risk = sand_burn_in_risk[mask]
            
            n_voxels = len(filtered_centers)
            if n_voxels == 0:
                return None
            
            # Create VTK points
            centers_f32 = np.ascontiguousarray(filtered_centers, dtype=np.float32)
            vtk_points_array = numpy_support.numpy_to_vtk(centers_f32, deep=True, array_type=vtk.VTK_FLOAT)
            vtk_points_array.SetNumberOfComponents(3)
            
            points = vtk.vtkPoints()
            points.SetData(vtk_points_array)
            
            # Create scale array
            scaled_sizes = np.ascontiguousarray(filtered_sizes * 0.85, dtype=np.float32)
            vtk_scales = numpy_support.numpy_to_vtk(scaled_sizes, deep=True, array_type=vtk.VTK_FLOAT)
            vtk_scales.SetName("Scale")
            
            # Create polydata for glyph source centers
            point_polydata = vtk.vtkPolyData()
            point_polydata.SetPoints(points)
            point_polydata.GetPointData().AddArray(vtk_scales)
            point_polydata.GetPointData().SetActiveScalars("Scale")
            
            # Create colors (Tan -> Orange -> Deep Brown gradient)
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetName("Colors")
            
            # Normalize risk values relative to the filtered set for better color spread
            min_risk = np.min(filtered_risk)
            max_risk = np.max(filtered_risk)
            risk_range = max_risk - min_risk if max_risk > min_risk else 1.0
            
            for risk in filtered_risk:
                # Normalize risk relative to the displayed voxels' range
                t = (risk - min_risk) / risk_range
                t = min(max(t, 0.0), 1.0)  # Clamp to 0-1
                
                # Tan (210, 180, 140) -> Orange (255, 140, 0) -> Deep Brown (139, 69, 19)
                if t < 0.5:
                    # Tan to Orange
                    t2 = t * 2  # 0-1 for first half
                    r = int(210 + (255 - 210) * t2)
                    g = int(180 + (140 - 180) * t2)
                    b = int(140 + (0 - 140) * t2)
                else:
                    # Orange to Deep Brown
                    t2 = (t - 0.5) * 2  # 0-1 for second half
                    r = int(255 + (139 - 255) * t2)
                    g = int(140 + (69 - 140) * t2)
                    b = int(0 + (19 - 0) * t2)
                
                colors.InsertNextTuple3(r, g, b)
            
            point_polydata.GetPointData().AddArray(colors)
            
            # Create cube source
            cube = vtk.vtkCubeSource()
            cube.SetXLength(1.0)
            cube.SetYLength(1.0)
            cube.SetZLength(1.0)
            
            # Use glyph3D for instanced rendering
            glyph3D = vtk.vtkGlyph3D()
            glyph3D.SetSourceConnection(cube.GetOutputPort())
            glyph3D.SetInputData(point_polydata)
            glyph3D.SetScaleModeToScaleByScalar()
            glyph3D.SetColorModeToColorByScalar()
            glyph3D.Update()
            
            return glyph3D.GetOutput()
            
        except Exception as e:
            logger.error(
                f"Failed to build sand burn-in geometry: {e}",
                exc_info=True,
                context={
                    'n_voxels': len(voxel_centers) if voxel_centers is not None else 0,
                    'threshold': threshold
                }
            )
            return None
    
    def _prebuild_sand_burn_in_cache(self, voxel_centers, voxel_sizes, sand_burn_in_risk, sand_burn_in_zones, current_threshold):
        """Pre-build geometry for nearby threshold levels"""
        nearby_thresholds = self._get_bounded_threshold_levels(current_threshold)
        
        for thresh in nearby_thresholds:
            cache_key = round(thresh, 2)
            if (
                cache_key not in self._sand_burn_in_cache
                and self._should_cache_threshold(sand_burn_in_risk, cache_key, sand_burn_in_zones)
            ):
                polydata = self._build_sand_burn_in_geometry(
                    voxel_centers, voxel_sizes, sand_burn_in_risk, sand_burn_in_zones, thresh
                )
                self._cache_polydata(self._sand_burn_in_cache, cache_key, polydata)
    
    def update_sand_burn_in_threshold(self, threshold: float):
        """Update sand burn-in visualization threshold"""
        try:
            if self.current_sand_burn_in_data is None:
                return
            
            voxel_centers, voxel_sizes, sand_burn_in_risk, sand_burn_in_zones = self.current_sand_burn_in_data
            
            # Round threshold to cache key
            cache_key = round(threshold, 2)
            
            # Check cache
            if cache_key in self._sand_burn_in_cache:
                polydata = self._sand_burn_in_cache[cache_key]
            else:
                polydata = self._build_sand_burn_in_geometry(
                    voxel_centers, voxel_sizes, sand_burn_in_risk, sand_burn_in_zones, threshold
                )
            
            # Remove old actor
            if self.sand_burn_in_actor is not None:
                self.renderer.RemoveActor(self.sand_burn_in_actor)
            
            if polydata is None or polydata.GetNumberOfPoints() == 0:
                print("No sand burn-in risk at this threshold")
                self.render_window.Render()
                return
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            mapper.SetScalarModeToUsePointData()
            
            self.sand_burn_in_actor = vtk.vtkActor()
            self.sand_burn_in_actor.SetMapper(mapper)
            
            prop = self.sand_burn_in_actor.GetProperty()
            prop.SetOpacity(0.8)
            prop.SetInterpolationToFlat()
            
            self.renderer.AddActor(self.sand_burn_in_actor)
            self.render_window.Render()
            
            num_displayed = polydata.GetNumberOfCells() // 6
            print(f"Displayed {num_displayed:,} sand burn-in risk voxels")
            
            # Pre-build nearby thresholds for smoother slider movement
            self._prebuild_sand_burn_in_cache(voxel_centers, voxel_sizes, sand_burn_in_risk, sand_burn_in_zones, threshold)
            
        except Exception as e:
            logger.error(
                f"Failed to update sand burn-in threshold: {e}",
                exc_info=True,
                context={'threshold': threshold}
            )
            print(f"Error updating sand burn-in threshold: {e}")

    def hide_all_analysis(self):
        """Hide all analysis visualization actors"""
        if self.shrinkage_actor is not None:
            self.shrinkage_actor.SetVisibility(False)
        if self.hotspot_actor is not None:
            self.hotspot_actor.SetVisibility(False)
        if self.niyama_actor is not None:
            self.niyama_actor.SetVisibility(False)
        if self.hot_tear_actor is not None:
            self.hot_tear_actor.SetVisibility(False)
        if self.sand_burn_in_actor is not None:
            self.sand_burn_in_actor.SetVisibility(False)
        if self.fraction_liquid_actor is not None:
            self.fraction_liquid_actor.SetVisibility(False)
        self.render_window.Render()
    
    def show_analysis(self, analysis_type: str):
        """Show a specific analysis visualization and hide others"""
        # Hide all first
        self.hide_all_analysis()
        
        # Show the requested one
        if analysis_type == "shrinkage" and self.shrinkage_actor is not None:
            self.shrinkage_actor.SetVisibility(True)
        elif analysis_type == "hotspots" and self.hotspot_actor is not None:
            self.hotspot_actor.SetVisibility(True)
        elif analysis_type == "niyama" and self.niyama_actor is not None:
            self.niyama_actor.SetVisibility(True)
        elif analysis_type == "hot_tear" and self.hot_tear_actor is not None:
            self.hot_tear_actor.SetVisibility(True)
        elif analysis_type == "sand_burn_in" and self.sand_burn_in_actor is not None:
            self.sand_burn_in_actor.SetVisibility(True)
        elif analysis_type == "fraction_liquid" and self.fraction_liquid_actor is not None:
            self.fraction_liquid_actor.SetVisibility(True)
        
        self.render_window.Render()

    def clear_visualization(self):
        """Clear all visualization"""
        if self.mesh_actor is not None:
            self.renderer.RemoveActor(self.mesh_actor)
            self.mesh_actor = None
        
        # Clear all part actors
        for actor in self.part_actors.values():
            self.renderer.RemoveActor(actor)
        self.part_actors.clear()
        
        if self.shrinkage_actor is not None:
            self.renderer.RemoveActor(self.shrinkage_actor)
            self.shrinkage_actor = None
        
        if self.hotspot_actor is not None:
            self.renderer.RemoveActor(self.hotspot_actor)
            self.hotspot_actor = None
        
        if self.niyama_actor is not None:
            self.renderer.RemoveActor(self.niyama_actor)
            self.niyama_actor = None
        
        if self.hot_tear_actor is not None:
            self.renderer.RemoveActor(self.hot_tear_actor)
            self.hot_tear_actor = None
        
        if self.sand_burn_in_actor is not None:
            self.renderer.RemoveActor(self.sand_burn_in_actor)
            self.sand_burn_in_actor = None
        
        if self.fraction_liquid_actor is not None:
            self.renderer.RemoveActor(self.fraction_liquid_actor)
            self.fraction_liquid_actor = None
        
        # Clear all cached result geometry
        self._clear_result_caches()
        
        # Clear stored data
        self.current_shrinkage_data = None
        self.current_hotspot_data = None
        self.current_niyama_data = None
        self.current_hot_tear_data = None
        self.current_sand_burn_in_data = None
        self.current_fraction_liquid_data = None
        
        self.render_window.Render()
    
    def clear_result_actors(self):
        """Clear only the result visualization actors (shrinkage, hotspot, niyama, hot tear, sand burn-in, fraction liquid).
        
        This keeps the mesh/part actors intact but removes all analysis overlays.
        Call this when switching between visualization modes.
        """
        if self.shrinkage_actor is not None:
            self.renderer.RemoveActor(self.shrinkage_actor)
            self.shrinkage_actor = None
        
        if self.hotspot_actor is not None:
            self.renderer.RemoveActor(self.hotspot_actor)
            self.hotspot_actor = None
        
        if self.niyama_actor is not None:
            self.renderer.RemoveActor(self.niyama_actor)
            self.niyama_actor = None
        
        if self.hot_tear_actor is not None:
            self.renderer.RemoveActor(self.hot_tear_actor)
            self.hot_tear_actor = None
        
        if self.sand_burn_in_actor is not None:
            self.renderer.RemoveActor(self.sand_burn_in_actor)
            self.sand_burn_in_actor = None
        
        if self.fraction_liquid_actor is not None:
            self.renderer.RemoveActor(self.fraction_liquid_actor)
            self.fraction_liquid_actor = None

        self._clear_result_caches()
        
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
        prop.SetInterpolationToPhong()
        prop.SetSpecular(0.3)
        prop.SetSpecularPower(20)
        
        if should_use_depth_peeling():
            self.renderer.SetUseDepthPeeling(True)
            self.renderer.SetMaximumNumberOfPeels(4)
            self.renderer.SetOcclusionRatio(0.1)
        else:
            self.renderer.SetUseDepthPeeling(False)
        
        self.renderer.AddActor(actor)
        self.part_actors[part_id] = actor
        
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
        """Toggle visibility of a specific part"""
        if part_id in self.part_actors:
            self.part_actors[part_id].SetVisibility(visible)
            self.render_window.Render()
    
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
    
    def load_parts_from_manager(self, part_manager):
        """
        Load all parts from a PartManager instance.
        
        Args:
            part_manager: PartManager instance with parts to display
        """
        # Clear existing parts
        for actor in self.part_actors.values():
            self.renderer.RemoveActor(actor)
        self.part_actors.clear()
        
        # Load each part
        for i, part in enumerate(part_manager.get_all_parts()):
            if part.visible:
                color = part.color if part.color else self.PART_COLORS[i % len(self.PART_COLORS)]
                self.load_part(
                    part.id,
                    part.get_vertices(),
                    part.get_faces(),
                    color=color
                )
        
        # Reset camera to show all parts
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        if not self.part_actors and part_manager.get_all_parts():
            logger.warning("No visible VTK part actors were created from loaded parts.")
        else:
            logger.info(
                f"Loaded {len(self.part_actors)} VTK part actor(s); bounds={self.model_bounds}"
            )
        self.render_window.Render()
    
    def update_shrinkage_threshold(self, threshold: float):
        """
        Update shrinkage visualization threshold - optimized with caching.
        Uses pre-built geometry when available for instant updates.
        
        Args:
            threshold: New threshold (0-1)
        """
        if self.current_shrinkage_data is None:
            return
            
        voxel_centers, voxel_sizes, intensities = self.current_shrinkage_data
        
        # Round threshold to cache key (1% increments)
        cache_key = round(threshold, 2)
        
        # Check if we have cached geometry
        with self._shrinkage_cache_lock:
            polydata = self._shrinkage_cache.get(cache_key)

        if polydata is None:
            # Build geometry (will be fast with glyph3D)
            polydata = self._build_shrinkage_geometry_fast(
                voxel_centers, voxel_sizes, intensities, threshold
            )
        
        # Remove old actor
        if self.shrinkage_actor is not None:
            self.renderer.RemoveActor(self.shrinkage_actor)
        
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            print("No shrinkage voxels at this threshold")
            self.render_window.Render()
            return
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarModeToUsePointData()
        
        self.shrinkage_actor = vtk.vtkActor()
        self.shrinkage_actor.SetMapper(mapper)
        
        prop = self.shrinkage_actor.GetProperty()
        prop.SetOpacity(0.5)
        prop.SetInterpolationToFlat()
        
        self.renderer.AddActor(self.shrinkage_actor)
        self.render_window.Render()
        
        # Count voxels displayed
        num_displayed = polydata.GetNumberOfCells() // 6
        print(f"Displayed {num_displayed:,} shrinkage voxels")
        
        # Pre-build nearby thresholds in cache for smoother slider movement
        self._prebuild_cache_async(voxel_centers, voxel_sizes, intensities, threshold)
    
    def toggle_mesh_visibility(self, visible: bool):
        """Toggle mesh visibility"""
        if self.mesh_actor is not None:
            self.mesh_actor.SetVisibility(visible)
            self.render_window.Render()
    
    def toggle_shrinkage_visibility(self, visible: bool):
        """Toggle shrinkage visibility"""
        if self.shrinkage_actor is not None:
            self.shrinkage_actor.SetVisibility(visible)
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
        """Shut down background workers when the widget is closed."""
        try:
            self._cache_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._cache_executor.shutdown(wait=False)
        super().closeEvent(event)
