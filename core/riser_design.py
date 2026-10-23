"""
Automatic Riser Design Module for Casting Simulation

Implements automatic riser sizing and positioning using:
- Heuvers' Circle Method (inscribed circles for modulus estimation)
- Caine's Method (modulus-based riser sizing)
- Chvorinov's Rule (solidification time calculation)
- Feeding distance analysis

Key Equations:
    Modulus: M = V / A (Volume / Surface Area)
    
    Riser Size (Caine): Mr > 1.2 * Mc (riser modulus > casting modulus)
    
    Chvorinov Time: t = B * M² (solidification time)
    
    Feeding Distance (Pellini): Lf = T * √M

Usage:
    from core.riser_design import RiserDesigner, HeuversCircleMethod
    
    designer = RiserDesigner(mesh, alloy)
    recommendations = designer.analyze()
    
    for riser in recommendations:
        print(f"Riser at {riser.position}: diameter={riser.diameter}m")
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple
import numpy as np
from numpy.typing import NDArray


# Type aliases
FloatArray = NDArray[Any]
IntArray = NDArray[Any]
BoolArray = NDArray[Any]
Vector3 = Tuple[float, float, float]


class RiserType(Enum):
    """Types of risers"""
    BLIND = auto()       # Closed top (atmospheric)
    OPEN = auto()        # Open top (atmospheric + metallostatic)
    INSULATED = auto()   # Insulated riser (extended feeding)
    EXOTHERMIC = auto()  # Exothermic riser (heated)


class RiserShape(Enum):
    """Geometric shapes for risers"""
    CYLINDRICAL = auto()
    RECTANGULAR = auto()
    SPHERICAL = auto()


@dataclass
class RiserRecommendation:
    """
    Recommendation for a single riser.
    
    Attributes:
        position: (x, y, z) position in model space
        voxel_index: Index of attachment voxel
        riser_type: Type of riser
        shape: Geometric shape
        diameter: Diameter or equivalent dimension (m)
        height: Height of riser (m)
        modulus: Calculated modulus of riser
        feeding_zone: Indices of voxels this riser can feed
        confidence: Confidence score (0-1)
        rationale: Explanation of recommendation
    """
    position: Vector3
    voxel_index: int
    riser_type: RiserType = RiserType.BLIND
    shape: RiserShape = RiserShape.CYLINDRICAL
    diameter: float = 0.0
    height: float = 0.0
    modulus: float = 0.0
    feeding_zone: List[int] = field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    
    @property
    def volume(self) -> float:
        """Calculate riser volume"""
        if self.shape == RiserShape.CYLINDRICAL:
            return np.pi * (self.diameter / 2) ** 2 * self.height
        elif self.shape == RiserShape.RECTANGULAR:
            return self.diameter ** 2 * self.height
        elif self.shape == RiserShape.SPHERICAL:
            return (4/3) * np.pi * (self.diameter / 2) ** 3
        return 0.0
    
    @property
    def surface_area(self) -> float:
        """Calculate riser surface area (excluding contact)"""
        if self.shape == RiserShape.CYLINDRICAL:
            # Side + top (contact face not counted)
            r = self.diameter / 2
            return 2 * np.pi * r * self.height + np.pi * r ** 2
        elif self.shape == RiserShape.RECTANGULAR:
            # 4 sides + top
            return 4 * self.diameter * self.height + self.diameter ** 2
        elif self.shape == RiserShape.SPHERICAL:
            return 4 * np.pi * (self.diameter / 2) ** 2
        return 0.0


@dataclass
class RiserDesignParameters:
    """
    Parameters for riser design calculations.
    
    Attributes:
        modulus_factor: Riser modulus / casting modulus ratio (typically 1.2-1.5)
        height_ratio: Riser height / diameter ratio (typically 1.0-1.5)
        feeding_constant: Pellini feeding distance constant
        safety_factor: Safety factor for sizing (typically 1.1-1.3)
        min_diameter: Minimum riser diameter (m)
        max_diameter: Maximum riser diameter (m)
        preferred_type: Preferred riser type
        preferred_shape: Preferred riser shape
    """
    modulus_factor: float = 1.2
    height_ratio: float = 1.0
    feeding_constant: float = 4.5  # For steel
    safety_factor: float = 1.15
    min_diameter: float = 0.02  # 20mm minimum
    max_diameter: float = 0.5   # 500mm maximum
    preferred_type: RiserType = RiserType.BLIND
    preferred_shape: RiserShape = RiserShape.CYLINDRICAL


@dataclass
class StandardRiserSize:
    """
    Standard industry riser size from database.
    
    Attributes:
        diameter: Riser diameter in inches
        height: Riser height in inches
        post_diameter: Recommended riser post diameter in inches
    """
    diameter: float
    height: float
    post_diameter: float
    
    @property
    def modulus(self) -> float:
        """Calculate modulus of this riser (V/A) in inches"""
        r = self.diameter / 2
        volume = np.pi * r**2 * self.height
        # Side + top (bottom contacts casting)
        area = 2 * np.pi * r * self.height + np.pi * r**2
        return volume / area


def load_riser_database() -> List[StandardRiserSize]:
    """
    Load standard riser sizes from database CSV.
    
    Returns:
        List of StandardRiserSize objects sorted by diameter
    """
    # Find the database file relative to this module
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(module_dir, 'riser_database.csv')
    
    sizes = []
    
    if not os.path.exists(db_path):
        print(f"Warning: Riser database not found at {db_path}")
        return sizes
    
    try:
        with open(db_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    diameter = float(row['Riser Diameter'])
                    height = float(row['Riser Height'])
                    post_diameter = float(row['Riser Post Diameter'])
                    sizes.append(StandardRiserSize(diameter, height, post_diameter))
                except (ValueError, KeyError) as e:
                    continue
    except Exception as e:
        print(f"Error loading riser database: {e}")
    
    # Sort by diameter
    sizes.sort(key=lambda s: s.diameter)
    return sizes


def select_standard_riser(required_modulus: float, riser_db: List[StandardRiserSize]) -> Optional[StandardRiserSize]:
    """
    Select the smallest standard riser that meets the required modulus.
    
    Args:
        required_modulus: Required riser modulus in inches
        riser_db: List of available standard riser sizes
        
    Returns:
        StandardRiserSize or None if no suitable size found
    """
    if not riser_db:
        return None
    
    # Find smallest riser with modulus >= required
    for riser in riser_db:
        if riser.modulus >= required_modulus:
            return riser
    
    # If no riser is large enough, return the largest available
    return riser_db[-1]


@dataclass
class HotSpot:
    """
    Identified thermal hotspot requiring feeding.
    
    Attributes:
        center_index: Voxel index at center of hotspot
        center_position: (x, y, z) position
        indices: All voxel indices in hotspot
        modulus: Local geometric modulus
        volume: Approximate volume of hotspot
        priority: Priority ranking (higher = more critical)
    """
    center_index: int
    center_position: Vector3
    indices: List[int]
    modulus: float
    volume: float
    priority: float


class MaterialRiserProperties(Protocol):
    """Protocol for material properties needed for riser design"""
    density_liquid: float
    density_solid: float
    liquidus_temp: float
    solidus_temp: float


class HeuversCircleMethod:
    """
    Heuvers' Circle Method for modulus estimation.
    
    The method inscribes circles within a 2D cross-section of the casting.
    The diameter of the largest inscribed circle is related to the local
    modulus and helps identify thermal hotspots.
    
    Key Concept:
        In a uniform cross-section, the center of the largest inscribed
        circle is the thermal center (last point to solidify).
    
    Usage:
        method = HeuversCircleMethod(mesh_data)
        hotspots = method.find_hotspots()
    """
    
    def __init__(
        self,
        voxel_positions: FloatArray,
        voxel_size: float,
        surface_mask: BoolArray,
    ):
        """
        Initialize Heuvers' circle method.
        
        Args:
            voxel_positions: Position of each voxel (n x 3)
            voxel_size: Size of each voxel
            surface_mask: Boolean mask of surface voxels
        """
        self.positions = voxel_positions
        self.voxel_size = voxel_size
        self.surface_mask = surface_mask
        self.n_voxels = len(voxel_positions)
        
        # Cached distance-to-surface
        self._distance_to_surface: Optional[FloatArray] = None
    
    def calculate_distance_to_surface(self) -> FloatArray:
        """
        Calculate distance from each voxel to nearest surface voxel.
        
        This approximates the inscribed circle radius at each point.
        """
        if self._distance_to_surface is not None:
            return self._distance_to_surface
        
        # Get surface voxel positions
        surface_indices = np.where(self.surface_mask)[0]
        surface_positions = self.positions[surface_indices]
        
        distances = np.zeros(self.n_voxels, dtype=np.float64)
        
        # For each voxel, find minimum distance to surface
        for i in range(self.n_voxels):
            if self.surface_mask[i]:
                distances[i] = 0.0
            else:
                # Distance to all surface voxels
                diffs = surface_positions - self.positions[i]
                dists = np.sqrt(np.sum(diffs ** 2, axis=1))
                distances[i] = np.min(dists)
        
        self._distance_to_surface = distances
        return distances
    
    def find_hotspots(
        self,
        n_hotspots: int = 5,
        min_separation: float = 0.05
    ) -> List[HotSpot]:
        """
        Find thermal hotspots using distance-to-surface maxima.
        
        Args:
            n_hotspots: Maximum number of hotspots to return
            min_separation: Minimum separation between hotspots (m)
            
        Returns:
            List of HotSpot objects sorted by priority
        """
        distances = self.calculate_distance_to_surface()
        
        # Find local maxima (voxels where distance is larger than neighbors)
        # For simplicity, use global ranking with separation constraint
        
        # Sort by distance (descending)
        sorted_indices = np.argsort(-distances)
        
        hotspots = []
        selected_positions = []
        
        for idx in sorted_indices:
            if len(hotspots) >= n_hotspots:
                break
            
            if self.surface_mask[idx]:
                continue
            
            pos = self.positions[idx]
            
            # Check separation from existing hotspots
            too_close = False
            for selected_pos in selected_positions:
                dist = np.sqrt(np.sum((np.array(pos) - np.array(selected_pos)) ** 2))
                if dist < min_separation:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Calculate local modulus (approximate using distance)
            local_modulus = distances[idx]  # Distance to surface ≈ modulus
            
            # Find cluster of high-distance voxels around this point
            threshold = 0.8 * distances[idx]
            cluster_mask = distances >= threshold
            cluster_dists = np.sqrt(np.sum(
                (self.positions - pos) ** 2, axis=1
            ))
            cluster_mask &= cluster_dists < 3 * distances[idx]
            
            cluster_indices = np.where(cluster_mask)[0].tolist()
            cluster_volume = len(cluster_indices) * (self.voxel_size ** 3)
            cluster_size = len(cluster_indices)
            
            hotspot = HotSpot(
                center_index=idx,
                center_position=tuple(pos),
                indices=cluster_indices,
                modulus=local_modulus,
                volume=cluster_volume,
                priority=distances[idx] * cluster_size
            )
            hotspots.append(hotspot)
            selected_positions.append(pos)
        
        # Sort by priority
        hotspots.sort(key=lambda h: -h.priority)
        
        return hotspots
    
    def estimate_modulus_2d(
        self,
        slice_axis: int = 2,
        slice_position: float = 0.0
    ) -> Tuple[FloatArray, FloatArray]:
        """
        Estimate modulus in a 2D slice using inscribed circles.
        
        Args:
            slice_axis: Axis perpendicular to slice (0=x, 1=y, 2=z)
            slice_position: Position along axis
            
        Returns:
            Tuple of (2D positions, modulus values)
        """
        # Find voxels in slice
        tolerance = self.voxel_size * 0.5
        in_slice = np.abs(self.positions[:, slice_axis] - slice_position) < tolerance
        
        slice_indices = np.where(in_slice)[0]
        if len(slice_indices) == 0:
            return np.array([]), np.array([])
        
        # Get 2D positions (drop slice axis)
        axes = [i for i in range(3) if i != slice_axis]
        positions_2d = self.positions[slice_indices][:, axes]
        
        # Get distances as modulus estimate
        distances = self.calculate_distance_to_surface()
        moduli = distances[slice_indices]
        
        return positions_2d, moduli


class RiserDesigner:
    """
    Automatic riser design system.
    
    Analyzes casting geometry and thermal characteristics to recommend:
    - Number of risers needed
    - Riser positions (attachment points)
    - Riser sizes (diameter, height)
    - Riser types (blind, open, insulated, exothermic)
    
    Methods Used:
    1. Heuvers' Circle Method - Identify hotspots
    2. Caine's Method - Size risers based on modulus
    3. Feeding Distance Analysis - Position risers within range
    
    Example:
        designer = RiserDesigner(
            voxel_positions=mesh.centers,
            voxel_size=mesh.voxel_size,
            neighbors=mesh.neighbors,
            surface_mask=mesh.surface,
            alloy=material
        )
        
        recommendations = designer.design_risers()
        
        for riser in recommendations:
            print(f"Add {riser.riser_type.name} riser at {riser.position}")
            print(f"  Diameter: {riser.diameter*1000:.1f} mm")
            print(f"  Height: {riser.height*1000:.1f} mm")
    """
    
    def __init__(
        self,
        voxel_positions: FloatArray,
        voxel_size: float,
        neighbors: IntArray,
        surface_mask: BoolArray,
        alloy: MaterialRiserProperties,
        params: Optional[RiserDesignParameters] = None,
    ):
        """
        Initialize riser designer.
        
        Args:
            voxel_positions: Position of each voxel (n x 3)
            voxel_size: Size of each voxel
            neighbors: Neighbor connectivity array
            surface_mask: Boolean mask of surface voxels
            alloy: Material properties
            params: Design parameters
        """
        self.positions = voxel_positions
        self.voxel_size = voxel_size
        self.neighbors = neighbors
        self.surface_mask = surface_mask
        self.alloy = alloy
        self.params = params or RiserDesignParameters()
        
        self.n_voxels = len(voxel_positions)
        
        # Initialize Heuvers method
        self._heuvers = HeuversCircleMethod(voxel_positions, voxel_size, surface_mask)
        
        # Load standard riser database
        self._riser_db = load_riser_database()
        
        # Calculated values
        self._modulus: Optional[FloatArray] = None
        self._hotspots: Optional[List[HotSpot]] = None
        self._recommendations: List[RiserRecommendation] = []
    
    @property
    def hotspots(self) -> List[HotSpot]:
        """List of identified hotspots"""
        if self._hotspots is None:
            self._hotspots = self._heuvers.find_hotspots()
        return self._hotspots
    
    @property
    def recommendations(self) -> List[RiserRecommendation]:
        """List of riser recommendations"""
        return self._recommendations
    
    def calculate_local_modulus(self) -> FloatArray:
        """
        Calculate local geometric modulus for each voxel.
        
        Uses distance-to-surface as modulus approximation.
        True modulus (V/A) requires mesh analysis.
        """
        if self._modulus is not None:
            return self._modulus
        
        self._modulus = self._heuvers.calculate_distance_to_surface()
        return self._modulus
    
    def design_risers(
        self,
        max_risers: int = 5,
        modulus_override: Optional[FloatArray] = None
    ) -> List[RiserRecommendation]:
        """
        Design risers for the casting.
        
        Process:
        1. Calculate modulus field (or use provided)
        2. Find hotspots using Heuvers' method
        3. For each hotspot:
           a. Calculate required riser modulus
           b. Size riser using Caine's method
           c. Find optimal attachment point
           d. Calculate feeding zone
        
        Args:
            max_risers: Maximum number of risers to recommend
            modulus_override: Optional pre-calculated modulus field
            
        Returns:
            List of RiserRecommendation objects
        """
        self._recommendations = []
        
        # Calculate modulus
        if modulus_override is not None:
            self._modulus = modulus_override
        else:
            self.calculate_local_modulus()
        
        # Find hotspots
        self._hotspots = self._heuvers.find_hotspots(
            n_hotspots=max_risers,
            min_separation=self.params.min_diameter * 2
        )
        
        # Design riser for each hotspot
        for hotspot in self._hotspots:
            riser = self._design_single_riser(hotspot)
            if riser is not None:
                self._recommendations.append(riser)
        
        return self._recommendations
    
    def _design_single_riser(self, hotspot: HotSpot) -> Optional[RiserRecommendation]:
        """Design a single riser for a hotspot using standard industry sizes."""
        
        # Required riser modulus (Caine's method)
        # Mr = modulus_factor * Mc
        # hotspot.modulus is already in inches (simulation units)
        required_modulus = self.params.modulus_factor * hotspot.modulus * self.params.safety_factor
        
        # Try to select from standard industry riser sizes
        std_riser = None
        if self._riser_db:
            std_riser = select_standard_riser(required_modulus, self._riser_db)
            if std_riser:
                # Use standard size (already in inches)
                diameter = std_riser.diameter
                height = std_riser.height
                print(f"Selected standard riser: Ø{diameter:.2f}\" x {height:.2f}\" (post: Ø{std_riser.post_diameter:.2f}\")")
            else:
                # Fall back to calculated size
                diameter = 2.5 * required_modulus
                height = diameter * self.params.height_ratio
        else:
            # No database - use calculated size
            diameter = 2.5 * required_modulus
            height = diameter * self.params.height_ratio
        
        # Clamp to limits (min/max are in inches)
        diameter = np.clip(
            diameter,
            self.params.min_diameter,
            self.params.max_diameter
        )
        
        # If we used a standard riser, keep its height; otherwise calculate
        if std_riser is None:
            height = diameter * self.params.height_ratio
        
        # Find attachment point (surface voxel nearest to hotspot)
        attachment_idx, attachment_pos = self._find_attachment_point(
            hotspot.center_position
        )
        
        if attachment_idx is None:
            return None
        
        # Calculate feeding zone (voxels within feeding distance)
        feeding_zone = self._calculate_feeding_zone(
            attachment_idx,
            diameter
        )
        
        # Calculate actual riser modulus
        riser_volume = np.pi * (diameter / 2) ** 2 * height
        riser_area = 2 * np.pi * (diameter / 2) * height + np.pi * (diameter / 2) ** 2
        riser_modulus = riser_volume / riser_area
        
        # Confidence based on modulus ratio
        confidence = min(1.0, riser_modulus / required_modulus / 1.2)
        
        # Build rationale - show inches (simulation units)
        if self._riser_db and std_riser:
            size_source = f"Standard size: Ø{diameter:.2f}\" x {height:.2f}\" (post: Ø{std_riser.post_diameter:.2f}\")"
        else:
            size_source = f"Calculated size: Ø{diameter:.2f}\" x {height:.2f}\""
            
        rationale = (
            f"Hotspot modulus: {hotspot.modulus:.2f}\"\n"
            f"Required riser modulus: {required_modulus:.2f}\"\n"
            f"Actual riser modulus: {riser_modulus:.2f}\"\n"
            f"{size_source}\n"
            f"Feeding distance: {self._feeding_distance(diameter):.2f}\"\n"
            f"Voxels fed: {len(feeding_zone)}"
        )
        
        return RiserRecommendation(
            position=attachment_pos,
            voxel_index=attachment_idx,
            riser_type=self.params.preferred_type,
            shape=self.params.preferred_shape,
            diameter=diameter,
            height=height,
            modulus=riser_modulus,
            feeding_zone=feeding_zone,
            confidence=confidence,
            rationale=rationale
        )
    
    def _find_attachment_point(
        self,
        target_position: Vector3
    ) -> Tuple[Optional[int], Optional[Vector3]]:
        """Find best surface voxel for riser attachment."""
        target = np.array(target_position)
        
        # Find surface voxels
        surface_indices = np.where(self.surface_mask)[0]
        
        if len(surface_indices) == 0:
            return None, None
        
        # Find topmost surface voxels (risers typically on top)
        surface_positions = self.positions[surface_indices]
        max_z = np.max(surface_positions[:, 2])
        top_surface_mask = surface_positions[:, 2] > max_z - self.voxel_size * 3
        
        if np.any(top_surface_mask):
            # Prefer top surface
            top_indices = surface_indices[top_surface_mask]
            top_positions = surface_positions[top_surface_mask]
        else:
            # Use all surface
            top_indices = surface_indices
            top_positions = surface_positions
        
        # Find closest to target (in x-y)
        xy_dists = np.sqrt(
            (top_positions[:, 0] - target[0]) ** 2 +
            (top_positions[:, 1] - target[1]) ** 2
        )
        
        best_idx = np.argmin(xy_dists)
        
        return int(top_indices[best_idx]), tuple(top_positions[best_idx])
    
    def _calculate_feeding_zone(
        self,
        riser_index: int,
        diameter: float
    ) -> List[int]:
        """Calculate which voxels can be fed by this riser."""
        feeding_distance = self._feeding_distance(diameter)
        
        riser_pos = self.positions[riser_index]
        
        # Find all voxels within feeding distance
        distances = np.sqrt(np.sum((self.positions - riser_pos) ** 2, axis=1))
        
        # Include voxels within feeding distance
        within_range = distances <= feeding_distance
        
        return np.where(within_range)[0].tolist()
    
    def _feeding_distance(self, diameter: float) -> float:
        """Calculate feeding distance for given riser diameter."""
        # Approximate modulus from diameter
        ratio = self.params.height_ratio
        height = diameter * ratio
        volume = np.pi * (diameter / 2) ** 2 * height
        area = 2 * np.pi * (diameter / 2) * height + np.pi * (diameter / 2) ** 2
        modulus = volume / area
        
        # Pellini's rule: Lf = T * sqrt(M)
        return self.params.feeding_constant * np.sqrt(modulus)
    
    def _covered_fraction(self, covered: set) -> float:
        """Fed share of the casting (voxel count share)."""
        return len(covered) / self.n_voxels

    def analyze_coverage(self) -> Dict[str, float]:
        """
        Analyze what fraction of the casting is covered by riser feeding.
        
        Returns:
            Dict with coverage statistics
        """
        if not self._recommendations:
            return {'covered_fraction': 0.0, 'uncovered_voxels': self.n_voxels}
        
        # Combine all feeding zones
        covered = set()
        for riser in self._recommendations:
            covered.update(riser.feeding_zone)
        
        covered_fraction = self._covered_fraction(covered)
        
        return {
            'covered_fraction': covered_fraction,
            'covered_voxels': len(covered),
            'uncovered_voxels': self.n_voxels - len(covered),
            'n_risers': len(self._recommendations),
            'avg_riser_diameter': np.mean([r.diameter for r in self._recommendations]),
            'total_riser_volume': sum(r.volume for r in self._recommendations)
        }
    
    def optimize_positions(
        self,
        iterations: int = 100
    ) -> List[RiserRecommendation]:
        """
        Optimize riser positions using gradient descent.
        
        Minimizes total riser volume while maintaining coverage.
        
        Args:
            iterations: Number of optimization iterations
            
        Returns:
            Optimized riser recommendations
        """
        if not self._recommendations:
            return []
        
        # Simple optimization: try moving each riser slightly
        # and keep move if coverage improves or volume decreases
        
        best_recommendations = list(self._recommendations)
        best_coverage = self.analyze_coverage()['covered_fraction']
        
        for _ in range(iterations):
            for i, riser in enumerate(best_recommendations):
                for new_idx, new_pos in self._candidate_attachment_moves(riser):
                    # Update feeding zone
                    new_zone = self._calculate_feeding_zone(
                        new_idx, riser.diameter
                    )
                    
                    # Check if improvement
                    test_recs = list(best_recommendations)
                    test_recs[i] = RiserRecommendation(
                        position=new_pos,
                        voxel_index=new_idx,
                        riser_type=riser.riser_type,
                        shape=riser.shape,
                        diameter=riser.diameter,
                        height=riser.height,
                        modulus=riser.modulus,
                        feeding_zone=new_zone,
                        confidence=riser.confidence,
                        rationale=riser.rationale
                    )
                    
                    # Calculate new coverage
                    covered = set()
                    for r in test_recs:
                        covered.update(r.feeding_zone)
                    new_coverage = self._covered_fraction(covered)
                    
                    if new_coverage > best_coverage:
                        best_recommendations = test_recs
                        best_coverage = new_coverage
        
        self._recommendations = best_recommendations
        return best_recommendations
    
    def _candidate_attachment_moves(
        self,
        riser: RiserRecommendation,
    ) -> List[Tuple[int, Vector3]]:
        """Attachment points one step away from ``riser``'s current one.

        The 8 lattice offsets of +-voxel_size in x and y, each snapped to
        the nearest top-surface voxel.
        """
        moves: List[Tuple[int, Vector3]] = []
        for dx in [-self.voxel_size, 0, self.voxel_size]:
            for dy in [-self.voxel_size, 0, self.voxel_size]:
                if dx == 0 and dy == 0:
                    continue
                new_target = (
                    riser.position[0] + dx,
                    riser.position[1] + dy,
                    riser.position[2]
                )
                new_idx, new_pos = self._find_attachment_point(new_target)
                if new_idx is None:
                    continue
                moves.append((new_idx, new_pos))
        return moves

    def generate_report(self) -> str:
        """Generate text report of riser design."""
        lines = [
            "=" * 60,
            "AUTOMATIC RISER DESIGN REPORT",
            "=" * 60,
            "",
            f"Number of risers recommended: {len(self._recommendations)}",
            ""
        ]
        
        for i, riser in enumerate(self._recommendations, 1):
            lines.extend([
                f"Riser {i}:",
                f"  Type: {riser.riser_type.name}",
                f"  Shape: {riser.shape.name}",
                f"  Position: ({riser.position[0]:.4f}, {riser.position[1]:.4f}, {riser.position[2]:.4f}) m",
                f"  Diameter: {riser.diameter * 1000:.1f} mm",
                f"  Height: {riser.height * 1000:.1f} mm",
                f"  Volume: {riser.volume * 1e6:.1f} cm³",
                f"  Modulus: {riser.modulus * 1000:.2f} mm",
                f"  Voxels fed: {len(riser.feeding_zone)}",
                f"  Confidence: {riser.confidence:.1%}",
                ""
            ])
        
        # Coverage analysis
        coverage = self.analyze_coverage()
        lines.extend([
            "Coverage Analysis:",
            f"  Covered fraction: {coverage['covered_fraction']:.1%}",
            f"  Uncovered voxels: {coverage['uncovered_voxels']}",
            f"  Total riser volume: {coverage['total_riser_volume'] * 1e6:.1f} cm³",
            "",
            "=" * 60
        ])
        
        return "\n".join(lines)


# ============================================================================
# Utility Functions
# ============================================================================

def chvorinov_time(
    modulus: float,
    mold_constant: float = 0.05
) -> float:
    """
    Calculate solidification time using Chvorinov's rule.

    t = B * M²

    Args:
        modulus: Section modulus
        mold_constant: Mold constant B (m²/min)

    Returns:
        Solidification time in minutes
    """
    return mold_constant * modulus ** 2
