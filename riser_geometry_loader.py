"""
Riser Geometry Loader
=====================
Parses industry-standard riser and riser post STEP files from the
machine_learning_risers_and_posts folder and builds a comprehensive
riser database for ML training and auto-riser selection.

Riser File Naming Convention:
    riser_##_D{diameter}_H{height}.step
    riser_post_##_D{diameter}_H{height}.step

Where diameter and height are in inches.

STEP loading note
-----------------
This module's ``_get_step_geometry`` function uses ``gmsh`` directly
to obtain BREP-exact analytic volume and mesh-derived surface area.
It is intentionally separate from the user-facing CAD upload path
(``web/payloads.py`` / ``core/part_manager.py``), which uses ``trimesh``
+ ``cascadio`` for STEP-to-mesh conversion.

If ``gmsh`` is unavailable (it is not declared in ``requirements.txt``),
``GMSH_AVAILABLE`` is ``False`` and ``_get_step_geometry`` returns
``(0, 0)`` silently. Migrating this path to ``cascadio`` is tracked
as a future change — see ``docs/STEP-integration-plan.md``.
"""

import os
import re
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

# Try to import gmsh, but make it optional since we can extract
# dimensions from filenames
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


@dataclass
class RiserGeometry:
    """Complete riser geometry data from STEP file."""
    id: int
    filename: str
    diameter: float  # inches
    height: float  # inches
    volume: float  # cubic inches
    surface_area: float  # square inches
    modulus: float  # V/A ratio (thermal modulus)
    h_d_ratio: float  # height to diameter ratio
    step_file_path: str  # Full path to STEP file
    
    # Derived foundry parameters
    feeding_capacity: float = 0.0  # Volume available for feeding
    solidification_time_factor: float = 0.0  # Proportional to modulus²


@dataclass
class RiserPostGeometry:
    """Complete riser post geometry data from STEP file."""
    id: int
    filename: str
    width: float  # inches (X dimension for rectangular posts)
    length: float  # inches (Y dimension for rectangular posts)
    height: float  # inches (Z dimension - post height)
    volume: float  # cubic inches
    step_file_path: str  # Full path to STEP file
    
    # Matching riser ID (posts are matched to risers by ID)
    matching_riser_id: int = 0
    
    @property
    def diameter(self) -> float:
        """For backward compatibility - returns width as diameter equivalent."""
        return self.width


@dataclass
class RiserPair:
    """A matched riser and post pair - the complete assembly."""
    riser: RiserGeometry
    post: RiserPostGeometry
    
    # Combined properties
    total_volume: float = 0.0
    total_height: float = 0.0  # Riser height + post height
    
    def __post_init__(self):
        self.total_volume = self.riser.volume + self.post.volume
        self.total_height = self.riser.height + self.post.height


class RiserGeometryLoader:
    """
    Loads and analyzes riser geometries from STEP files.
    Builds a comprehensive database for ML-based riser selection.
    """
    
    # Regex patterns to extract dimensions from filenames
    RISER_PATTERN = re.compile(r'riser_(\d+)_D([\d.]+)_H([\d.]+)\.step', re.IGNORECASE)
    # New rectangular post pattern: W=width, L=length, H=height
    POST_PATTERN_RECT = re.compile(r'riser_post_(\d+)_W([\d.]+)_L([\d.]+)_H([\d.]+)\.step', re.IGNORECASE)
    # Legacy cylindrical post pattern (for backward compatibility)
    POST_PATTERN_CYL = re.compile(r'riser_post_(\d+)_D([\d.]+)_H([\d.]+)\.step', re.IGNORECASE)
    
    def __init__(self, riser_folder: str = 'machine_learning_risers_and_posts'):
        """
        Initialize the loader with the folder containing riser STEP files.
        
        Args:
            riser_folder: Path to folder containing riser and post STEP files
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.riser_folder = os.path.join(script_dir, riser_folder)
        
        self.risers: List[RiserGeometry] = []
        self.posts: List[RiserPostGeometry] = []
        self.riser_pairs: List[RiserPair] = []
        
        # Quick lookup by ID
        self.risers_by_id: Dict[int, RiserGeometry] = {}
        self.posts_by_id: Dict[int, RiserPostGeometry] = {}
        
        # Quick lookup by diameter range for selection
        self.risers_by_diameter: List[Tuple[float, RiserGeometry]] = []  # Sorted by diameter
    
    def load_all_geometries(self, verify_step: bool = False) -> bool:
        """
        Load all riser and post geometries from the folder.
        
        Args:
            verify_step: If True, open each STEP file to verify/compute exact volumes
                        (slower but more accurate). If False, compute from filename dims.
        
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.riser_folder):
            print(f"Error: Riser folder not found: {self.riser_folder}")
            return False
        
        files = os.listdir(self.riser_folder)
        riser_files = [f for f in files if self.RISER_PATTERN.match(f)]
        # Support both rectangular (new) and cylindrical (legacy) post patterns
        post_files = [f for f in files if self.POST_PATTERN_RECT.match(f) or self.POST_PATTERN_CYL.match(f)]
        
        print(f"\nLoading industry riser geometries from: {self.riser_folder}")
        print(f"  Found {len(riser_files)} riser files")
        print(f"  Found {len(post_files)} riser post files")
        
        # Load risers
        for filename in sorted(riser_files):
            riser = self._parse_riser_file(filename, verify_step)
            if riser:
                self.risers.append(riser)
                self.risers_by_id[riser.id] = riser
        
        # Load posts
        for filename in sorted(post_files):
            post = self._parse_post_file(filename, verify_step)
            if post:
                self.posts.append(post)
                self.posts_by_id[post.id] = post
        
        # Match risers to posts and create pairs
        self._match_riser_post_pairs()
        
        # Build sorted lookup for diameter-based selection
        self.risers_by_diameter = sorted(
            [(r.diameter, r) for r in self.risers],
            key=lambda x: x[0]
        )
        
        print(f"\n  Successfully loaded:")
        print(f"    {len(self.risers)} risers")
        print(f"    {len(self.posts)} riser posts")
        print(f"    {len(self.riser_pairs)} matched riser/post pairs")
        
        # Print diameter range
        if self.risers:
            min_d = min(r.diameter for r in self.risers)
            max_d = max(r.diameter for r in self.risers)
            print(f"\n  Riser diameter range: {min_d:.2f}\" to {max_d:.2f}\"")
        
        return len(self.risers) > 0
    
    def _parse_riser_file(self, filename: str, verify_step: bool = False) -> Optional[RiserGeometry]:
        """Parse a riser STEP file and extract geometry data."""
        match = self.RISER_PATTERN.match(filename)
        if not match:
            return None
        
        riser_id = int(match.group(1))
        diameter = float(match.group(2))
        height = float(match.group(3))
        
        # Calculate volume and surface area for a cylinder
        # Volume = π × r² × h
        radius = diameter / 2
        volume = np.pi * (radius ** 2) * height
        
        # Surface area = 2πr² + 2πrh (top + bottom + side)
        # For a riser, typically open top, so: πr² + 2πrh
        surface_area = np.pi * (radius ** 2) + 2 * np.pi * radius * height
        
        # Thermal modulus (V/A ratio)
        modulus = volume / surface_area if surface_area > 0 else 0
        
        # H/D ratio
        h_d_ratio = height / diameter if diameter > 0 else 0
        
        step_path = os.path.join(self.riser_folder, filename)
        
        # Optionally verify with actual STEP file
        if verify_step and os.path.exists(step_path):
            actual_vol, actual_area = self._get_step_geometry(step_path)
            if actual_vol > 0:
                volume = actual_vol
                surface_area = actual_area
                modulus = volume / surface_area if surface_area > 0 else 0
        
        # Solidification time factor (proportional to modulus²)
        solidification_time_factor = modulus ** 2
        
        # Feeding capacity - usable feeding volume (typically 15-20% of riser volume)
        feeding_capacity = volume * 0.18  # 18% feeding efficiency
        
        riser = RiserGeometry(
            id=riser_id,
            filename=filename,
            diameter=diameter,
            height=height,
            volume=volume,
            surface_area=surface_area,
            modulus=modulus,
            h_d_ratio=h_d_ratio,
            step_file_path=step_path,
            feeding_capacity=feeding_capacity,
            solidification_time_factor=solidification_time_factor
        )
        
        return riser
    
    def _parse_post_file(self, filename: str, verify_step: bool = False) -> Optional[RiserPostGeometry]:
        """Parse a riser post STEP file and extract geometry data."""
        # Try rectangular pattern first (new format)
        match = self.POST_PATTERN_RECT.match(filename)
        is_rectangular = True
        
        if not match:
            # Fall back to cylindrical pattern (legacy format)
            match = self.POST_PATTERN_CYL.match(filename)
            is_rectangular = False
        
        if not match:
            return None
        
        post_id = int(match.group(1))
        
        if is_rectangular:
            # New rectangular format: W=width, L=length, H=height
            width = float(match.group(2))
            length = float(match.group(3))
            height = float(match.group(4))
            # Volume for rectangular box
            volume = width * length * height
        else:
            # Legacy cylindrical format: D=diameter, H=height
            diameter = float(match.group(2))
            height = float(match.group(3))
            # For backward compatibility, width = diameter, length = diameter
            width = diameter
            length = diameter
            # Volume for cylinder
            radius = diameter / 2
            volume = np.pi * (radius ** 2) * height
        
        step_path = os.path.join(self.riser_folder, filename)
        
        # Optionally verify with actual STEP file
        if verify_step and os.path.exists(step_path):
            actual_vol, _ = self._get_step_geometry(step_path)
            if actual_vol > 0:
                volume = actual_vol
        
        post = RiserPostGeometry(
            id=post_id,
            filename=filename,
            width=width,
            length=length,
            height=height,
            volume=volume,
            step_file_path=step_path,
            matching_riser_id=post_id  # Posts match risers by ID
        )
        
        return post
    
    def _get_step_geometry(self, step_path: str) -> Tuple[float, float]:
        """
        Load a STEP file and compute actual volume and surface area.
        
        Returns:
            (volume, surface_area) in cubic/square inches
        """
        if not GMSH_AVAILABLE:
            return 0, 0
            
        if not os.path.exists(step_path):
            return 0, 0
        
        try:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(step_path)
            
            volumes = gmsh.model.getEntities(dim=3)
            if not volumes:
                gmsh.model.occ.synchronize()
                volumes = gmsh.model.getEntities(dim=3)
            
            if not volumes:
                gmsh.finalize()
                return 0, 0
            
            # Get bounding box for unit detection
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
            max_extent = max(xmax - xmin, ymax - ymin, zmax - zmin)
            
            # Auto-detect units
            if max_extent > 100:
                unit_scale = 1.0 / 25.4  # mm to inches
            elif max_extent > 40:
                unit_scale = 1.0 / 2.54  # cm to inches
            elif max_extent < 1:
                unit_scale = 39.3701  # m to inches
            else:
                unit_scale = 1.0
            
            # Compute volume
            total_volume = 0
            for dim, tag in volumes:
                vol = gmsh.model.occ.getMass(dim, tag)
                total_volume += vol
            
            volume = total_volume * (unit_scale ** 3)
            
            # Generate mesh for surface area
            mesh_size = max_extent * 0.05
            gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.5)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size * 2)
            gmsh.model.mesh.generate(2)
            
            # Compute surface area from mesh
            surface_area = self._compute_surface_area_from_mesh() * (unit_scale ** 2)
            
            gmsh.finalize()
            return volume, surface_area
            
        except Exception as e:
            try:
                gmsh.finalize()
            except:
                pass
            return 0, 0
    
    def _compute_surface_area_from_mesh(self) -> float:
        """Compute surface area from gmsh mesh triangles."""
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1, 3)
        node_lookup = {int(tag): idx for idx, tag in enumerate(node_tags)}
        
        surfaces = gmsh.model.getEntities(dim=2)
        total_area = 0
        
        for dim, tag in surfaces:
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim, tag)
            for etype, tags, nodes in zip(elem_types, elem_tags, elem_node_tags):
                if etype == 2:  # Triangles
                    triangles = nodes.reshape(-1, 3)
                    for tri in triangles:
                        try:
                            v0 = node_coords[node_lookup[int(tri[0])]]
                            v1 = node_coords[node_lookup[int(tri[1])]]
                            v2 = node_coords[node_lookup[int(tri[2])]]
                            edge1 = v1 - v0
                            edge2 = v2 - v0
                            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                            total_area += area
                        except KeyError:
                            continue
        
        return total_area
    
    def _match_riser_post_pairs(self):
        """Match risers with their corresponding posts by ID."""
        self.riser_pairs = []
        
        for riser in self.risers:
            if riser.id in self.posts_by_id:
                post = self.posts_by_id[riser.id]
                pair = RiserPair(riser=riser, post=post)
                self.riser_pairs.append(pair)
    
    def select_riser_for_thickness(self, section_thickness: float, 
                                    d_t_ratio: float = 1.2,
                                    prefer_larger: bool = True) -> Optional[RiserPair]:
        """
        Select the best riser for a given section thickness using D/T ratio.
        
        Args:
            section_thickness: Local casting section thickness in inches
            d_t_ratio: Desired diameter-to-thickness ratio (default 1.2)
            prefer_larger: If True, round up to next larger size
        
        Returns:
            Best matching RiserPair, or None if no suitable riser found
        """
        required_diameter = d_t_ratio * section_thickness
        
        # Find risers that meet the requirement
        suitable_risers = [r for r in self.risers if r.diameter >= required_diameter * 0.9]
        
        if not suitable_risers:
            # Return largest available
            if self.risers:
                largest = max(self.risers, key=lambda r: r.diameter)
                if largest.id in self.posts_by_id:
                    return RiserPair(riser=largest, post=self.posts_by_id[largest.id])
            return None
        
        # Sort by diameter
        suitable_risers.sort(key=lambda r: r.diameter)
        
        if prefer_larger:
            # Select smallest that meets requirement (with margin)
            selected = suitable_risers[0]
        else:
            # Find closest match
            selected = min(suitable_risers, 
                          key=lambda r: abs(r.diameter - required_diameter))
        
        # Return as pair with matching post
        if selected.id in self.posts_by_id:
            return RiserPair(riser=selected, post=self.posts_by_id[selected.id])
        
        # Return with default post if no matching post found
        return RiserPair(
            riser=selected,
            post=RiserPostGeometry(
                id=selected.id,
                filename="",
                width=selected.diameter * 0.7,
                length=selected.diameter * 0.7 + 3.0,  # length = width + 3"
                height=selected.diameter * 0.6,
                volume=0,
                step_file_path="",
                matching_riser_id=selected.id
            )
        )
    
    def select_riser_for_modulus(self, casting_section_modulus: float,
                                  modulus_ratio: float = 1.2) -> Optional[RiserPair]:
        """
        Select riser based on thermal modulus requirement.
        Riser modulus must exceed casting section modulus for proper feeding.
        
        Args:
            casting_section_modulus: V/A ratio of casting section being fed
            modulus_ratio: Required riser/casting modulus ratio (default 1.2)
        
        Returns:
            Best matching RiserPair
        """
        required_modulus = casting_section_modulus * modulus_ratio
        
        # Find risers with adequate modulus
        suitable = [r for r in self.risers if r.modulus >= required_modulus]
        
        if not suitable:
            # Return largest modulus available
            if self.risers:
                largest = max(self.risers, key=lambda r: r.modulus)
                if largest.id in self.posts_by_id:
                    return RiserPair(riser=largest, post=self.posts_by_id[largest.id])
            return None
        
        # Select smallest adequate riser
        selected = min(suitable, key=lambda r: r.modulus)
        
        if selected.id in self.posts_by_id:
            return RiserPair(riser=selected, post=self.posts_by_id[selected.id])
        
        return None
    
    def get_all_risers_sorted(self) -> List[RiserGeometry]:
        """Get all risers sorted by diameter (smallest to largest)."""
        return sorted(self.risers, key=lambda r: r.diameter)
    
    def get_all_pairs_sorted(self) -> List[RiserPair]:
        """Get all riser/post pairs sorted by riser diameter."""
        return sorted(self.riser_pairs, key=lambda p: p.riser.diameter)
    
    def export_to_csv(self, output_path: str = 'riser_database_enhanced.csv'):
        """Export the loaded riser database to enhanced CSV format."""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header - updated for rectangular posts
            writer.writerow([
                'ID', 'Riser Diameter', 'Riser Height', 'Riser Volume', 
                'Riser Modulus', 'H/D Ratio', 'Riser STEP File',
                'Post Width', 'Post Length', 'Post Height', 'Post Volume', 'Post STEP File',
                'Feeding Capacity', 'Solidification Factor'
            ])
            
            # Data rows
            for pair in self.get_all_pairs_sorted():
                r = pair.riser
                p = pair.post
                writer.writerow([
                    r.id,
                    f"{r.diameter:.2f}",
                    f"{r.height:.2f}",
                    f"{r.volume:.3f}",
                    f"{r.modulus:.4f}",
                    f"{r.h_d_ratio:.2f}",
                    r.filename,
                    f"{p.width:.2f}",
                    f"{p.length:.2f}",
                    f"{p.height:.2f}",
                    f"{p.volume:.3f}",
                    p.filename,
                    f"{r.feeding_capacity:.3f}",
                    f"{r.solidification_time_factor:.4f}"
                ])
        
        print(f"\nExported enhanced riser database to: {output_path}")
    
    def export_to_json(self, output_path: str = 'riser_database_enhanced.json'):
        """Export the loaded riser database to JSON format for web/ML use."""
        data = {
            'metadata': {
                'source_folder': self.riser_folder,
                'riser_count': len(self.risers),
                'post_count': len(self.posts),
                'pair_count': len(self.riser_pairs),
            },
            'risers': [asdict(r) for r in self.get_all_risers_sorted()],
            'posts': [asdict(p) for p in sorted(self.posts, key=lambda x: x.id)],
            'pairs': []
        }
        
        for pair in self.get_all_pairs_sorted():
            data['pairs'].append({
                'id': pair.riser.id,
                'riser': asdict(pair.riser),
                'post': asdict(pair.post),
                'total_volume': pair.total_volume,
                'total_height': pair.total_height
            })
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nExported enhanced riser database to: {output_path}")
    
    def print_summary(self):
        """Print a summary of all loaded riser geometries."""
        print("\n" + "=" * 70)
        print("  INDUSTRY RISER DATABASE SUMMARY")
        print("=" * 70)
        
        print("\n  RISERS:")
        print("  " + "-" * 66)
        print(f"  {'ID':>3} | {'Diameter':>8} | {'Height':>8} | {'Volume':>10} | {'Modulus':>8} | {'H/D':>6}")
        print("  " + "-" * 66)
        
        for r in self.get_all_risers_sorted():
            print(f"  {r.id:>3} | {r.diameter:>7.2f}\" | {r.height:>7.2f}\" | "
                  f"{r.volume:>9.2f}in³ | {r.modulus:>7.4f}\" | {r.h_d_ratio:>6.2f}")
        
        print("\n  RISER POSTS:")
        print("  " + "-" * 66)
        print(f"  {'ID':>3} | {'Width':>8} | {'Length':>8} | {'Height':>8} | {'Volume':>10}")
        print("  " + "-" * 66)
        
        for p in sorted(self.posts, key=lambda x: x.id):
            print(f"  {p.id:>3} | {p.width:>7.2f}\" | {p.length:>7.2f}\" | {p.height:>7.2f}\" | {p.volume:>9.2f}in³")
        
        print("\n" + "=" * 70)


# Singleton instance for global access
_riser_loader_instance: Optional[RiserGeometryLoader] = None


def get_riser_loader(reload: bool = False) -> RiserGeometryLoader:
    """
    Get the global riser geometry loader instance.
    
    Args:
        reload: If True, force reload of all geometries
    
    Returns:
        RiserGeometryLoader instance with all geometries loaded
    """
    global _riser_loader_instance
    
    if _riser_loader_instance is None or reload:
        _riser_loader_instance = RiserGeometryLoader()
        _riser_loader_instance.load_all_geometries()
    
    return _riser_loader_instance


def main():
    """Test the riser geometry loader."""
    print("\n" + "=" * 70)
    print("  RISER GEOMETRY LOADER - Industry Standard Riser Database")
    print("=" * 70)
    
    loader = RiserGeometryLoader()
    success = loader.load_all_geometries(verify_step=False)
    
    if success:
        loader.print_summary()
        
        # Export databases
        script_dir = os.path.dirname(os.path.abspath(__file__))
        loader.export_to_csv(os.path.join(script_dir, 'riser_database_enhanced.csv'))
        loader.export_to_json(os.path.join(script_dir, 'riser_database_enhanced.json'))
        
        # Test riser selection
        print("\n" + "-" * 70)
        print("  RISER SELECTION TEST")
        print("-" * 70)
        
        test_thicknesses = [1.0, 2.0, 3.0, 4.0, 5.0]
        for thickness in test_thicknesses:
            pair = loader.select_riser_for_thickness(thickness, d_t_ratio=1.2)
            if pair:
                print(f"\n  Section thickness: {thickness:.1f}\"")
                print(f"    → Selected: Riser #{pair.riser.id} "
                      f"(Ø{pair.riser.diameter:.2f}\" × {pair.riser.height:.2f}\")")
                print(f"    → Post: {pair.post.width:.2f}\" × {pair.post.length:.2f}\" × {pair.post.height:.2f}\" (W×L×H)")
                print(f"    → Actual D/T: {pair.riser.diameter / thickness:.2f}")
    else:
        print("\nFailed to load riser geometries!")


if __name__ == "__main__":
    main()
