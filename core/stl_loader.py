"""
STL File Loader
Loads STL files in inches and keeps everything in standard units
"""

import numpy as np
from stl import mesh
import trimesh
from typing import Tuple, Optional
import os

# Import error logger
try:
    from core.error_logger import logger
except ImportError:
    # Fallback if logger not available
    class DummyLogger:
        def debug(self, msg, **kwargs): pass
        def info(self, msg, **kwargs): pass
        def warning(self, msg, **kwargs): pass
        def error(self, msg, **kwargs): pass
        def exception(self, msg, **kwargs): pass
    logger = DummyLogger()


class STLLoader:
    """Load and process STL files for casting simulation"""
    
    def __init__(self):
        self.mesh = None
        self.trimesh_obj = None
        self.filename = None
        self.bounds = None
        self.volume = None
        self.surface_area = None
        self.center = None
        
    def load_stl(self, filepath: str, units: str = 'inches') -> bool:
        """
        Load STL file in standard units (inches)
        
        Args:
            filepath: Path to STL file
            units: 'inches' (default) - all units kept in inches
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Loading STL file: {filepath}")
        
        if not os.path.exists(filepath):
            logger.error(f"STL file not found: {filepath}")
            print(f"Error: File not found: {filepath}")
            return False
        
        try:
            # Load with numpy-stl
            self.mesh = mesh.Mesh.from_file(filepath)
            logger.debug(f"numpy-stl loaded successfully")
            
            # Load with trimesh for better geometry operations
            self.trimesh_obj = trimesh.load(filepath)
            logger.debug(f"trimesh loaded successfully")
            
            # Everything stays in inches - no conversion
            print(f"Loaded in standard units (inches)")
            
            # Calculate properties
            self._calculate_properties()
            
            self.filename = os.path.basename(filepath)
            logger.info(f"STL loaded: {self.filename}, Vertices: {len(self.trimesh_obj.vertices):,}, Faces: {len(self.trimesh_obj.faces):,}, Volume: {self.volume:.4f} in³")
            print(f"✓ Loaded: {self.filename}")
            print(f"  Vertices: {len(self.trimesh_obj.vertices):,}")
            print(f"  Faces: {len(self.trimesh_obj.faces):,}")
            print(f"  Volume: {self.volume:.4f} in³")
            print(f"  Surface Area: {self.surface_area:.2f} in²")
            print(f"  Bounds: {self.get_dimensions()}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Failed to load STL file: {filepath}")
            print(f"Error loading STL: {e}")
            return False
    
    def _calculate_properties(self):
        """Calculate mesh properties"""
        try:
            # Bounding box
            self.bounds = np.array([
                [self.trimesh_obj.bounds[0][0], self.trimesh_obj.bounds[1][0]],  # X
                [self.trimesh_obj.bounds[0][1], self.trimesh_obj.bounds[1][1]],  # Y
                [self.trimesh_obj.bounds[0][2], self.trimesh_obj.bounds[1][2]]   # Z
            ])
            
            # Volume and surface area
            if self.trimesh_obj.is_watertight:
                self.volume = abs(self.trimesh_obj.volume)
                self.surface_area = self.trimesh_obj.area
            else:
                logger.warning(f"Mesh is not watertight - attempting to fix")
                print("  Warning: Mesh is not watertight. Volume may be inaccurate.")
                # Try to fix
                self.trimesh_obj.fill_holes()
                # Use volume even if not watertight (better than 0 for complex models)
                self.volume = abs(self.trimesh_obj.volume) if abs(self.trimesh_obj.volume) > 0 else 0
                self.surface_area = self.trimesh_obj.area
            
            # Center of mass
            self.center = self.trimesh_obj.center_mass
            logger.debug(f"Mesh properties calculated: bounds={self.get_dimensions()}, volume={self.volume:.4f}")
        except Exception as e:
            logger.exception(f"Error calculating mesh properties")
            raise
    
    def get_dimensions(self) -> str:
        """Get dimensions as formatted string"""
        if self.bounds is None:
            return "N/A"
        
        x_size = self.bounds[0][1] - self.bounds[0][0]
        y_size = self.bounds[1][1] - self.bounds[1][0]
        z_size = self.bounds[2][1] - self.bounds[2][0]
        
        return f"{x_size:.2f}\" x {y_size:.2f}\" x {z_size:.2f}\""
    
    def get_max_dimension(self) -> float:
        """Get maximum dimension"""
        if self.bounds is None:
            return 0
        
        x_size = self.bounds[0][1] - self.bounds[0][0]
        y_size = self.bounds[1][1] - self.bounds[1][0]
        z_size = self.bounds[2][1] - self.bounds[2][0]
        
        return max(x_size, y_size, z_size)
    
    def get_vertices(self) -> np.ndarray:
        """Get vertices as numpy array"""
        if self.trimesh_obj is None:
            return np.array([])
        return self.trimesh_obj.vertices
    
    def get_faces(self) -> np.ndarray:
        """Get faces as numpy array"""
        if self.trimesh_obj is None:
            return np.array([])
        return self.trimesh_obj.faces
    
    def get_face_normals(self) -> np.ndarray:
        """Get face normal vectors"""
        if self.trimesh_obj is None:
            return np.array([])
        return self.trimesh_obj.face_normals
    
    def get_bounds_array(self) -> np.ndarray:
        """Get bounds as numpy array [min, max] for each axis"""
        return self.bounds
    
    def is_valid(self) -> bool:
        """Check if mesh is loaded and valid"""
        return self.mesh is not None and self.trimesh_obj is not None
    
    def save_stl(self, filepath: str):
        """Save processed STL file"""
        if self.trimesh_obj is not None:
            self.trimesh_obj.export(filepath)
            print(f"Saved to: {filepath}")
    
    def get_section_thickness(self, point: np.ndarray) -> float:
        """
        Estimate local section thickness at a point
        Uses ray casting to find opposite surface
        
        Args:
            point: 3D point coordinates
        
        Returns:
            Estimated thickness in inches
        """
        if self.trimesh_obj is None:
            return 0.0
        
        # Cast rays in multiple directions and take minimum distance
        directions = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ]
        
        min_thickness = float('inf')
        
        for direction in directions:
            # Cast ray from point in direction
            locations, index_ray, index_tri = self.trimesh_obj.ray.intersects_location(
                ray_origins=[point],
                ray_directions=[direction]
            )
            
            if len(locations) > 0:
                # Find closest intersection
                distances = np.linalg.norm(locations - point, axis=1)
                min_dist = np.min(distances[distances > 0.001])  # Ignore very close hits
                min_thickness = min(min_thickness, min_dist)
        
        return min_thickness if min_thickness != float('inf') else 0.5  # Default 0.5 inches
    
    def detect_features(self) -> dict:
        """
        Detect important casting features
        - Thin sections
        - Hot spots (thick sections)
        - Isolated regions
        
        Returns:
            Dictionary with feature information
        """
        features = {
            'thin_sections': [],
            'hot_spots': [],
            'surface_area_to_volume': 0
        }
        
        if not self.is_valid():
            return features
        
        # Surface area to volume ratio (indicator of complexity)
        if self.volume > 0:
            features['surface_area_to_volume'] = self.surface_area / self.volume
        
        return features


if __name__ == '__main__':
    # Test STL loader
    loader = STLLoader()
    print("STL Loader ready. Use load_stl(filepath, units='inches') to load a file.")
