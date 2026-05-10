"""
Shrinkage Prediction Module for Casting Simulation

Implements shrinkage porosity prediction:
- Volumetric shrinkage calculation
- Feeding distance analysis
- Isolated liquid region detection
- Hotspot identification

Key Concepts:
    Volumetric Shrinkage:
        ΔV/V = (ρ_solid - ρ_liquid) / ρ_solid ≈ 3-7% for most alloys
    
    Feeding Distance (Pellini):
        Lf = T * √(M)
    
where:
    T = constant for alloy (~4.5 for steel)
    M = modulus (V/A)

Usage:
    from physics.shrinkage import ShrinkageModel
    
    model = ShrinkageModel(mesh, alloy)
    model.analyze(solidification_state, temperatures)
    predictions = model.get_predictions()

Status:
    This module is retained as a reference/experimental path.
    The application's authoritative shrinkage and porosity outputs are produced
    by ``core/solidification.py`` in the geometry-driven solver path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional, Protocol, Set, Tuple
import numpy as np
from numpy.typing import NDArray


# Type aliases
FloatArray = NDArray[Any]
IntArray = NDArray[Any]
BoolArray = NDArray[Any]


class ShrinkageType(Enum):
    """Types of shrinkage defects"""
    MACRO_SHRINKAGE = auto()     # Large cavities (pipe)
    MICRO_SHRINKAGE = auto()     # Fine dispersed porosity
    CENTERLINE_POROSITY = auto() # Along thermal centerline
    SURFACE_SINK = auto()        # External surface depression


@dataclass
class ShrinkagePrediction:
    """
    Shrinkage prediction for a specific location.
    
    Attributes:
        indices: Voxel indices with predicted shrinkage
        type: Type of shrinkage defect
        severity: Severity score (0-1)
        volume_fraction: Predicted porosity volume fraction
        feeding_distance: Distance to nearest feeder
        confidence: Confidence in prediction
    """
    indices: IntArray
    type: ShrinkageType
    severity: float
    volume_fraction: float
    feeding_distance: float
    confidence: float
    
    @property
    def is_critical(self) -> bool:
        """Whether this is a critical defect"""
        return self.severity > 0.7 or self.volume_fraction > 0.02


@dataclass
class FeedingPath:
    """
    Describes a feeding path from riser to casting.
    
    Attributes:
        riser_index: Index of riser voxel
        path_indices: Sequence of voxel indices along path
        effective_length: Effective feeding length
        bottleneck_modulus: Smallest modulus along path
        is_viable: Whether path can provide adequate feeding
    """
    riser_index: int
    path_indices: List[int]
    effective_length: float
    bottleneck_modulus: float
    is_viable: bool


class MaterialShrinkageProperties(Protocol):
    """Protocol for material shrinkage properties"""
    density_liquid: float
    density_solid: float
    liquidus_temp: float
    solidus_temp: float


class ShrinkageModel:
    """
    Model for predicting shrinkage defects.
    
    Uses multiple criteria:
    1. Isolated liquid regions (last to solidify)
    2. Feeding distance analysis
    3. Thermal modulus comparison
    4. Niyama criterion (delegated to niyama.py)
    
    Example:
        model = ShrinkageModel(n_voxels, neighbors, alloy, voxel_size)
        model.set_modulus(modulus_array)
        model.set_feeder_locations(riser_indices)
        
        # After solidification simulation:
        predictions = model.analyze(fraction_solid, temperatures)
        
        for pred in predictions:
            if pred.is_critical:
                print(f"Critical shrinkage at indices {pred.indices}")
    """
    
    def __init__(
        self,
        n_voxels: int,
        neighbors: IntArray,
        alloy: MaterialShrinkageProperties,
        voxel_size: float
    ):
        """
        Initialize shrinkage model.
        
        Args:
            n_voxels: Number of voxels
            neighbors: Neighbor connectivity array
            alloy: Material properties
            voxel_size: Voxel size in meters
        """
        self.n_voxels = n_voxels
        self.neighbors = neighbors
        self.alloy = alloy
        self.voxel_size = voxel_size
        
        # Calculate volumetric shrinkage
        self.volumetric_shrinkage = (
            (alloy.density_solid - alloy.density_liquid) / alloy.density_solid
        )
        
        # Feeding parameters
        self._modulus: Optional[FloatArray] = None
        self._feeder_indices: Set[int] = set()
        
        # Results
        self._predictions: List[ShrinkagePrediction] = []
        self._hotspots: Optional[IntArray] = None
        self._feeding_distances: Optional[FloatArray] = None
    
    @property
    def predictions(self) -> List[ShrinkagePrediction]:
        """List of shrinkage predictions"""
        return self._predictions
    
    @property
    def hotspots(self) -> Optional[IntArray]:
        """Indices of detected hotspots (last to solidify)"""
        return self._hotspots
    
    @property
    def feeding_distances(self) -> Optional[FloatArray]:
        """Distance to nearest feeder for each voxel"""
        return self._feeding_distances
    
    def set_modulus(self, modulus: FloatArray) -> None:
        """
        Set geometric modulus (V/A) for each voxel.
        
        Modulus = Volume / Surface Area
        Higher modulus = slower cooling = solidifies last
        """
        self._modulus = modulus
    
    def set_feeder_locations(self, indices: List[int]) -> None:
        """Set indices of feeder (riser) voxels"""
        self._feeder_indices = set(indices)
    
    def analyze(
        self,
        fraction_solid: FloatArray,
        temperatures: FloatArray,
        solidification_time: Optional[FloatArray] = None
    ) -> List[ShrinkagePrediction]:
        """
        Analyze for shrinkage defects.
        
        Args:
            fraction_solid: Current fraction solid field
            temperatures: Current temperature field
            solidification_time: Time when each voxel solidified
            
        Returns:
            List of shrinkage predictions
        """
        self._predictions = []
        
        # 1. Find isolated liquid regions
        isolated = self._find_isolated_liquid(fraction_solid)
        
        # 2. Calculate feeding distances
        self._calculate_feeding_distances()
        
        # 3. Find thermal hotspots
        if solidification_time is not None:
            self._hotspots = self._find_hotspots(solidification_time)
        
        # 4. Identify macro-shrinkage risk
        macro_risks = self._analyze_macro_shrinkage(
            fraction_solid, 
            isolated,
            solidification_time
        )
        self._predictions.extend(macro_risks)
        
        # 5. Identify micro-shrinkage risk (based on mushy zone)
        micro_risks = self._analyze_micro_shrinkage(
            fraction_solid,
            temperatures
        )
        self._predictions.extend(micro_risks)
        
        return self._predictions
    
    def _find_isolated_liquid(self, fraction_solid: FloatArray) -> List[IntArray]:
        """
        Find isolated liquid regions (not connected to feeders).
        
        Uses flood fill from feeder locations.
        
        Returns:
            List of arrays, each containing indices of an isolated region
        """
        # Identify liquid voxels
        is_liquid = fraction_solid < 0.5
        
        # Flood fill from feeders through liquid
        connected_to_feeder = np.zeros(self.n_voxels, dtype=bool)
        
        if self._feeder_indices:
            # BFS from feeder locations
            queue = list(self._feeder_indices)
            visited = set(self._feeder_indices)
            
            for idx in self._feeder_indices:
                if is_liquid[idx]:
                    connected_to_feeder[idx] = True
            
            while queue:
                current = queue.pop(0)
                
                for j in range(self.neighbors.shape[1]):
                    neighbor = self.neighbors[current, j]
                    if neighbor >= 0 and neighbor not in visited:
                        visited.add(neighbor)
                        if is_liquid[neighbor]:
                            connected_to_feeder[neighbor] = True
                            queue.append(neighbor)
        
        # Find isolated liquid
        isolated_liquid = is_liquid & ~connected_to_feeder
        
        # Group into connected components
        isolated_regions = self._find_connected_components(isolated_liquid)
        
        return isolated_regions
    
    def _find_connected_components(self, mask: BoolArray) -> List[IntArray]:
        """Find connected components in masked region"""
        components = []
        visited = np.zeros(self.n_voxels, dtype=bool)
        
        for start in np.where(mask)[0]:
            if visited[start]:
                continue
            
            # BFS to find component
            component = []
            queue = [start]
            
            while queue:
                current = queue.pop(0)
                if visited[current]:
                    continue
                    
                visited[current] = True
                if mask[current]:
                    component.append(current)
                    
                    for j in range(self.neighbors.shape[1]):
                        neighbor = self.neighbors[current, j]
                        if neighbor >= 0 and not visited[neighbor] and mask[neighbor]:
                            queue.append(neighbor)
            
            if component:
                components.append(np.array(component, dtype=np.int32))
        
        return components
    
    def _calculate_feeding_distances(self) -> None:
        """Calculate distance to nearest feeder for each voxel"""
        if not self._feeder_indices:
            self._feeding_distances = np.full(self.n_voxels, np.inf)
            return
        
        # Dijkstra-like propagation from feeders
        distances = np.full(self.n_voxels, np.inf)
        
        for idx in self._feeder_indices:
            distances[idx] = 0
        
        # Simple iterative distance propagation
        changed = True
        while changed:
            changed = False
            for i in range(self.n_voxels):
                for j in range(self.neighbors.shape[1]):
                    neighbor = self.neighbors[i, j]
                    if neighbor >= 0:
                        new_dist = distances[neighbor] + self.voxel_size
                        if new_dist < distances[i]:
                            distances[i] = new_dist
                            changed = True
        
        self._feeding_distances = distances
    
    def _find_hotspots(self, solidification_time: FloatArray) -> IntArray:
        """
        Find thermal hotspots (last regions to solidify).
        
        Hotspots are local maxima of solidification time.
        """
        valid_times = solidification_time[np.isfinite(solidification_time)]
        if len(valid_times) == 0:
            return np.array([], dtype=np.int32)
        
        # Threshold for "late" solidification
        time_threshold = np.percentile(valid_times, 90)
        
        # Find local maxima
        is_hotspot = np.zeros(self.n_voxels, dtype=bool)
        
        for i in range(self.n_voxels):
            if not np.isfinite(solidification_time[i]):
                continue
            if solidification_time[i] < time_threshold:
                continue
            
            # Check if local maximum
            is_maximum = True
            for j in range(self.neighbors.shape[1]):
                neighbor = self.neighbors[i, j]
                if neighbor >= 0:
                    if solidification_time[neighbor] > solidification_time[i]:
                        is_maximum = False
                        break
            
            if is_maximum:
                is_hotspot[i] = True
        
        return np.where(is_hotspot)[0].astype(np.int32)
    
    def _analyze_macro_shrinkage(
        self,
        fraction_solid: FloatArray,
        isolated_regions: List[IntArray],
        solidification_time: Optional[FloatArray]
    ) -> List[ShrinkagePrediction]:
        """Analyze for macro-shrinkage (pipe) defects"""
        predictions = []
        
        for region in isolated_regions:
            if len(region) < 5:  # Too small to be significant
                continue
            
            # Calculate severity based on region size and location
            volume = len(region) * (self.voxel_size ** 3)
            
            # Check feeding distance
            if self._feeding_distances is not None:
                avg_feeding_dist = np.mean(self._feeding_distances[region])
            else:
                avg_feeding_dist = float('inf')
            
            # Severity increases with size and feeding distance
            severity = min(1.0, len(region) / 100 + avg_feeding_dist / 0.1)
            
            # Volume fraction of shrinkage
            vol_fraction = self.volumetric_shrinkage * 0.8  # Not all shrinkage forms porosity
            
            predictions.append(ShrinkagePrediction(
                indices=region,
                type=ShrinkageType.MACRO_SHRINKAGE,
                severity=severity,
                volume_fraction=vol_fraction,
                feeding_distance=avg_feeding_dist,
                confidence=0.8
            ))
        
        return predictions
    
    def _analyze_micro_shrinkage(
        self,
        fraction_solid: FloatArray,
        temperatures: FloatArray
    ) -> List[ShrinkagePrediction]:
        """Analyze for micro-shrinkage (dispersed porosity)"""
        predictions = []
        
        # Micro-shrinkage typically forms in mushy zone
        mushy_zone = (fraction_solid > 0.3) & (fraction_solid < 0.95)
        
        if not np.any(mushy_zone):
            return predictions
        
        # Group mushy zone by connectivity
        mushy_regions = self._find_connected_components(mushy_zone)
        
        for region in mushy_regions:
            if len(region) < 10:
                continue
            
            # Average fraction solid in region
            avg_fs = np.mean(fraction_solid[region])
            
            # Micro-shrinkage risk increases with fraction solid
            risk = (avg_fs - 0.3) / 0.65  # Normalized to 0-1
            
            # Check feeding distance
            if self._feeding_distances is not None:
                avg_dist = np.mean(self._feeding_distances[region])
                
                # Risk increases if poorly fed
                feeding_factor = min(1.0, avg_dist / 0.05)
                risk = risk * (0.5 + 0.5 * feeding_factor)
            else:
                avg_dist = float('inf')
            
            if risk > 0.3:  # Threshold for reporting
                predictions.append(ShrinkagePrediction(
                    indices=region,
                    type=ShrinkageType.MICRO_SHRINKAGE,
                    severity=risk,
                    volume_fraction=self.volumetric_shrinkage * risk * 0.3,
                    feeding_distance=avg_dist,
                    confidence=0.6
                ))
        
        return predictions


class FeedingDistanceCalculator:
    """
    Calculator for feeding distances and feeding paths.
    
    Implements various feeding distance rules:
    - Pellini's rule for steel
    - Campbell's rule for aluminum
    - Chvorinov's rule with extensions
    """
    
    # Feeding distance constants for different alloys
    FEEDING_CONSTANTS = {
        'steel': 4.5,
        'aluminum': 3.0,
        'cast_iron': 3.5,
        'copper': 4.0,
        'magnesium': 2.5
    }
    
    @staticmethod
    def pellini_rule(modulus: float, alloy_type: str = 'steel') -> float:
        """
        Calculate maximum feeding distance using Pellini's rule.
        
        Lf = T * √M
        
        Args:
            modulus: Geometric modulus (V/A) in meters
            alloy_type: Type of alloy for constant selection
            
        Returns:
            Maximum feeding distance in meters
        """
        T = FeedingDistanceCalculator.FEEDING_CONSTANTS.get(alloy_type, 4.0)
        return T * np.sqrt(modulus)
    
    @staticmethod
    def calculate_modulus(volume: float, surface_area: float) -> float:
        """
        Calculate geometric modulus.
        
        M = V / A
        
        Args:
            volume: Volume in m³
            surface_area: Surface area in m²
            
        Returns:
            Modulus in meters
        """
        if surface_area <= 0:
            return 0.0
        return volume / surface_area
    
    @staticmethod
    def chvorinov_time(
        modulus: float,
        mold_constant: float = 0.05  # m²/min for sand mold
    ) -> float:
        """
        Calculate solidification time using Chvorinov's rule.
        
        t = B * M²
        
        Args:
            modulus: Geometric modulus in meters
            mold_constant: Mold constant B in m²/min
            
        Returns:
            Solidification time in minutes
        """
        return mold_constant * modulus ** 2
    
    @staticmethod
    def analyze_feeding_path(
        path_moduli: List[float],
        riser_modulus: float,
        alloy_type: str = 'steel'
    ) -> Tuple[bool, str]:
        """
        Analyze whether a feeding path is viable.
        
        Criteria:
        1. Riser modulus must exceed all path moduli (directional solidification)
        2. No bottlenecks (local minima in modulus)
        3. Path length within feeding distance
        
        Args:
            path_moduli: Modulus values along feeding path
            riser_modulus: Modulus of the riser
            alloy_type: Type of alloy
            
        Returns:
            Tuple of (is_viable, reason)
        """
        if not path_moduli:
            return False, "Empty feeding path"
        
        # Check riser modulus
        max_path_modulus = max(path_moduli)
        if riser_modulus <= max_path_modulus:
            return False, f"Riser modulus ({riser_modulus:.4f}) must exceed path max ({max_path_modulus:.4f})"
        
        # Check for bottlenecks (local minima that would freeze off)
        for i in range(1, len(path_moduli) - 1):
            if path_moduli[i] < path_moduli[i-1] and path_moduli[i] < path_moduli[i+1]:
                if path_moduli[i] < 0.5 * riser_modulus:
                    return False, f"Bottleneck at position {i} with modulus {path_moduli[i]:.4f}"
        
        # Check feeding distance
        avg_modulus = np.mean(path_moduli)
        max_distance = FeedingDistanceCalculator.pellini_rule(avg_modulus, alloy_type)
        path_length = len(path_moduli) * avg_modulus  # Approximate
        
        if path_length > max_distance:
            return False, f"Path length ({path_length:.3f}m) exceeds feeding distance ({max_distance:.3f}m)"
        
        return True, "Viable feeding path"


# ============================================================================
# Numba-accelerated functions
# ============================================================================

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def find_isolated_liquid_fast(
        fraction_solid: FloatArray,
        neighbors: IntArray,
        feeder_indices: IntArray,
        liquid_threshold: float = 0.5
    ) -> BoolArray:
        """
        Fast isolated liquid detection using Numba.
        
        Returns boolean array where True = isolated liquid.
        """
        n = len(fraction_solid)
        is_liquid = fraction_solid < liquid_threshold
        connected = np.zeros(n, dtype=np.bool_)
        
        # Mark feeders
        for idx in feeder_indices:
            if is_liquid[idx]:
                connected[idx] = True
        
        # Propagate connectivity (multiple passes)
        for _ in range(n):  # Worst case iterations
            changed = False
            for i in range(n):
                if connected[i]:
                    continue
                if not is_liquid[i]:
                    continue
                    
                # Check neighbors
                for j in range(neighbors.shape[1]):
                    neighbor = neighbors[i, j]
                    if neighbor >= 0 and connected[neighbor] and is_liquid[i]:
                        connected[i] = True
                        changed = True
                        break
            
            if not changed:
                break
        
        return is_liquid & ~connected
    
    @njit(parallel=True, cache=True)
    def calculate_feeding_distances_parallel(
        n_voxels: int,
        neighbors: IntArray,
        feeder_indices: IntArray,
        voxel_size: float
    ) -> FloatArray:
        """
        Calculate minimum distance to feeder for each voxel (parallel).
        """
        distances = np.full(n_voxels, np.inf, dtype=np.float64)
        
        for idx in feeder_indices:
            distances[idx] = 0.0
        
        # Iterative propagation (not truly parallel but cached)
        for _ in range(n_voxels):
            changed = False
            for i in range(n_voxels):
                for j in range(neighbors.shape[1]):
                    neighbor = neighbors[i, j]
                    if neighbor >= 0:
                        new_dist = distances[neighbor] + voxel_size
                        if new_dist < distances[i]:
                            distances[i] = new_dist
                            changed = True
            if not changed:
                break
        
        return distances
