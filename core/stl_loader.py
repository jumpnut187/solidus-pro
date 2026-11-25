import numpy as np
from stl import mesh
import trimesh
from typing import Tuple, Optional
import os

try:
    from core.error_logger import logger
except ImportError:
    from core.error_logger_dummy import logger

class STLLoader:

    def __init__(self):
        self.mesh = None
        self.trimesh_obj = None
        self.filename = None
        self.bounds = None
        self.volume = None
        self.surface_area = None
        self.center = None

    def load_stl(self, filepath: str, units: str = 'inches') -> bool:
        logger.info(f"Loading STL file: {filepath}")

        if not os.path.exists(filepath):
            logger.error(f"STL file not found: {filepath}")
            print(f"Error: File not found: {filepath}")
            return False

        try:
            self.mesh = mesh.Mesh.from_file(filepath)
            logger.debug(f"numpy-stl loaded successfully")

            self.trimesh_obj = trimesh.load(filepath)
            logger.debug(f"trimesh loaded successfully")

            print(f"Loaded in standard units (inches)")

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
        try:
            self.bounds = np.array([
                [self.trimesh_obj.bounds[0][0], self.trimesh_obj.bounds[1][0]],
                [self.trimesh_obj.bounds[0][1], self.trimesh_obj.bounds[1][1]],
                [self.trimesh_obj.bounds[0][2], self.trimesh_obj.bounds[1][2]]
            ])

            if self.trimesh_obj.is_watertight:
                self.volume = abs(self.trimesh_obj.volume)
                self.surface_area = self.trimesh_obj.area
            else:
                logger.warning(f"Mesh is not watertight - attempting to fix")
                print("  Warning: Mesh is not watertight. Volume may be inaccurate.")
                self.trimesh_obj.fill_holes()
                self.volume = abs(self.trimesh_obj.volume) if abs(self.trimesh_obj.volume) > 0 else 0
                self.surface_area = self.trimesh_obj.area

            self.center = self.trimesh_obj.center_mass
            logger.debug(f"Mesh properties calculated: bounds={self.get_dimensions()}, volume={self.volume:.4f}")
        except Exception as e:
            logger.exception(f"Error calculating mesh properties")
            raise

    def get_dimensions(self) -> str:
        if self.bounds is None:
            return "N/A"

        x_size = self.bounds[0][1] - self.bounds[0][0]
        y_size = self.bounds[1][1] - self.bounds[1][0]
        z_size = self.bounds[2][1] - self.bounds[2][0]

        return f"{x_size:.2f}\" x {y_size:.2f}\" x {z_size:.2f}\""

    def get_max_dimension(self) -> float:
        if self.bounds is None:
            return 0

        x_size = self.bounds[0][1] - self.bounds[0][0]
        y_size = self.bounds[1][1] - self.bounds[1][0]
        z_size = self.bounds[2][1] - self.bounds[2][0]

        return max(x_size, y_size, z_size)

    def get_vertices(self) -> np.ndarray:
        if self.trimesh_obj is None:
            return np.array([])
        return self.trimesh_obj.vertices

    def get_faces(self) -> np.ndarray:
        if self.trimesh_obj is None:
            return np.array([])
        return self.trimesh_obj.faces

    def get_face_normals(self) -> np.ndarray:
        if self.trimesh_obj is None:
            return np.array([])
        return self.trimesh_obj.face_normals

    def get_bounds_array(self) -> np.ndarray:
        return self.bounds

    def is_valid(self) -> bool:
        return self.mesh is not None and self.trimesh_obj is not None

    def save_stl(self, filepath: str):
        if self.trimesh_obj is not None:
            self.trimesh_obj.export(filepath)
            print(f"Saved to: {filepath}")

    def get_section_thickness(self, point: np.ndarray) -> float:
        if self.trimesh_obj is None:
            return 0.0

        directions = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ]

        min_thickness = float('inf')

        for direction in directions:
            locations, index_ray, index_tri = self.trimesh_obj.ray.intersects_location(
                ray_origins=[point],
                ray_directions=[direction]
            )

            if len(locations) > 0:
                distances = np.linalg.norm(locations - point, axis=1)
                min_dist = np.min(distances[distances > 0.001])
                min_thickness = min(min_thickness, min_dist)

        return min_thickness if min_thickness != float('inf') else 0.5

    def detect_features(self) -> dict:
        features = {
            'thin_sections': [],
            'hot_spots': [],
            'surface_area_to_volume': 0
        }

        if not self.is_valid():
            return features

        if self.volume > 0:
            features['surface_area_to_volume'] = self.surface_area / self.volume

        return features

if __name__ == '__main__':
    loader = STLLoader()
    print("STL Loader ready. Use load_stl(filepath, units='inches') to load a file.")
