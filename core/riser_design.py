from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[Any]
IntArray = NDArray[Any]
BoolArray = NDArray[Any]
Vector3 = Tuple[float, float, float]

class RiserType(Enum):
    BLIND = auto()
    OPEN = auto()
    INSULATED = auto()
    EXOTHERMIC = auto()

class RiserShape(Enum):
    CYLINDRICAL = auto()
    RECTANGULAR = auto()
    SPHERICAL = auto()

@dataclass
class RiserRecommendation:
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
        if self.shape == RiserShape.CYLINDRICAL:
            return np.pi * (self.diameter / 2) ** 2 * self.height
        elif self.shape == RiserShape.RECTANGULAR:
            return self.diameter ** 2 * self.height
        elif self.shape == RiserShape.SPHERICAL:
            return (4/3) * np.pi * (self.diameter / 2) ** 3
        return 0.0

    @property
    def surface_area(self) -> float:
        if self.shape == RiserShape.CYLINDRICAL:
            r = self.diameter / 2
            return 2 * np.pi * r * self.height + np.pi * r ** 2
        elif self.shape == RiserShape.RECTANGULAR:
            return 4 * self.diameter * self.height + self.diameter ** 2
        elif self.shape == RiserShape.SPHERICAL:
            return 4 * np.pi * (self.diameter / 2) ** 2
        return 0.0

@dataclass
class RiserDesignParameters:
    modulus_factor: float = 1.2
    height_ratio: float = 1.0
    feeding_constant: float = 4.5
    safety_factor: float = 1.15
    min_diameter: float = 0.02
    max_diameter: float = 0.5
    preferred_type: RiserType = RiserType.BLIND
    preferred_shape: RiserShape = RiserShape.CYLINDRICAL

@dataclass
class StandardRiserSize:
    diameter: float
    height: float
    post_diameter: float

    @property
    def modulus(self) -> float:
        r = self.diameter / 2
        volume = np.pi * r**2 * self.height
        area = 2 * np.pi * r * self.height + np.pi * r**2
        return volume / area

def load_riser_database() -> List[StandardRiserSize]:
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

    sizes.sort(key=lambda s: s.diameter)
    return sizes

def select_standard_riser(required_modulus: float, riser_db: List[StandardRiserSize]) -> Optional[StandardRiserSize]:
    if not riser_db:
        return None

    for riser in riser_db:
        if riser.modulus >= required_modulus:
            return riser

    return riser_db[-1]

@dataclass
class HotSpot:
    center_index: int
    center_position: Vector3
    indices: List[int]
    modulus: float
    volume: float
    priority: float

class MaterialRiserProperties(Protocol):
    density_liquid: float
    density_solid: float
    liquidus_temp: float
    solidus_temp: float

class HeuversCircleMethod:

    def __init__(
        self,
        voxel_positions: FloatArray,
        voxel_size: float,
        surface_mask: BoolArray,
    ):
        self.positions = voxel_positions
        self.voxel_size = voxel_size
        self.surface_mask = surface_mask
        self.n_voxels = len(voxel_positions)

        self._distance_to_surface: Optional[FloatArray] = None

    def calculate_distance_to_surface(self) -> FloatArray:
        if self._distance_to_surface is not None:
            return self._distance_to_surface

        surface_indices = np.where(self.surface_mask)[0]
        surface_positions = self.positions[surface_indices]

        distances = np.zeros(self.n_voxels, dtype=np.float64)

        for i in range(self.n_voxels):
            if self.surface_mask[i]:
                distances[i] = 0.0
            else:
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
        distances = self.calculate_distance_to_surface()

        sorted_indices = np.argsort(-distances)

        hotspots = []
        selected_positions = []

        for idx in sorted_indices:
            if len(hotspots) >= n_hotspots:
                break

            if self.surface_mask[idx]:
                continue

            pos = self.positions[idx]

            too_close = False
            for selected_pos in selected_positions:
                dist = np.sqrt(np.sum((np.array(pos) - np.array(selected_pos)) ** 2))
                if dist < min_separation:
                    too_close = True
                    break

            if too_close:
                continue

            local_modulus = distances[idx]

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

        hotspots.sort(key=lambda h: -h.priority)

        return hotspots

    def estimate_modulus_2d(
        self,
        slice_axis: int = 2,
        slice_position: float = 0.0
    ) -> Tuple[FloatArray, FloatArray]:
        tolerance = self.voxel_size * 0.5
        in_slice = np.abs(self.positions[:, slice_axis] - slice_position) < tolerance

        slice_indices = np.where(in_slice)[0]
        if len(slice_indices) == 0:
            return np.array([]), np.array([])

        axes = [i for i in range(3) if i != slice_axis]
        positions_2d = self.positions[slice_indices][:, axes]

        distances = self.calculate_distance_to_surface()
        moduli = distances[slice_indices]

        return positions_2d, moduli

class RiserDesigner:

    def __init__(
        self,
        voxel_positions: FloatArray,
        voxel_size: float,
        neighbors: IntArray,
        surface_mask: BoolArray,
        alloy: MaterialRiserProperties,
        params: Optional[RiserDesignParameters] = None,
    ):
        self.positions = voxel_positions
        self.voxel_size = voxel_size
        self.neighbors = neighbors
        self.surface_mask = surface_mask
        self.alloy = alloy
        self.params = params or RiserDesignParameters()

        self.n_voxels = len(voxel_positions)

        self._heuvers = HeuversCircleMethod(voxel_positions, voxel_size, surface_mask)

        self._riser_db = load_riser_database()

        self._modulus: Optional[FloatArray] = None
        self._hotspots: Optional[List[HotSpot]] = None
        self._recommendations: List[RiserRecommendation] = []

    @property
    def hotspots(self) -> List[HotSpot]:
        if self._hotspots is None:
            self._hotspots = self._heuvers.find_hotspots()
        return self._hotspots

    @property
    def recommendations(self) -> List[RiserRecommendation]:
        return self._recommendations

    def calculate_local_modulus(self) -> FloatArray:
        if self._modulus is not None:
            return self._modulus

        self._modulus = self._heuvers.calculate_distance_to_surface()
        return self._modulus

    def design_risers(
        self,
        max_risers: int = 5,
        modulus_override: Optional[FloatArray] = None
    ) -> List[RiserRecommendation]:
        self._recommendations = []

        if modulus_override is not None:
            self._modulus = modulus_override
        else:
            self.calculate_local_modulus()

        self._hotspots = self._heuvers.find_hotspots(
            n_hotspots=max_risers,
            min_separation=self.params.min_diameter * 2
        )

        for hotspot in self._hotspots:
            riser = self._design_single_riser(hotspot)
            if riser is not None:
                self._recommendations.append(riser)

        return self._recommendations

    def _design_single_riser(self, hotspot: HotSpot) -> Optional[RiserRecommendation]:

        required_modulus = self.params.modulus_factor * hotspot.modulus * self.params.safety_factor

        std_riser = None
        if self._riser_db:
            std_riser = select_standard_riser(required_modulus, self._riser_db)
            if std_riser:
                diameter = std_riser.diameter
                height = std_riser.height
                print(f"Selected standard riser: Ø{diameter:.2f}\" x {height:.2f}\" (post: Ø{std_riser.post_diameter:.2f}\")")
            else:
                diameter = 2.5 * required_modulus
                height = diameter * self.params.height_ratio
        else:
            diameter = 2.5 * required_modulus
            height = diameter * self.params.height_ratio

        diameter = np.clip(
            diameter,
            self.params.min_diameter,
            self.params.max_diameter
        )

        if std_riser is None:
            height = diameter * self.params.height_ratio

        attachment_idx, attachment_pos = self._find_attachment_point(
            hotspot.center_position
        )

        if attachment_idx is None:
            return None

        feeding_zone = self._calculate_feeding_zone(
            attachment_idx,
            diameter
        )

        riser_volume = np.pi * (diameter / 2) ** 2 * height
        riser_area = 2 * np.pi * (diameter / 2) * height + np.pi * (diameter / 2) ** 2
        riser_modulus = riser_volume / riser_area

        confidence = min(1.0, riser_modulus / required_modulus / 1.2)

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
        target = np.array(target_position)

        surface_indices = np.where(self.surface_mask)[0]

        if len(surface_indices) == 0:
            return None, None

        surface_positions = self.positions[surface_indices]
        max_z = np.max(surface_positions[:, 2])
        top_surface_mask = surface_positions[:, 2] > max_z - self.voxel_size * 3

        if np.any(top_surface_mask):
            top_indices = surface_indices[top_surface_mask]
            top_positions = surface_positions[top_surface_mask]
        else:
            top_indices = surface_indices
            top_positions = surface_positions

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
        feeding_distance = self._feeding_distance(diameter)

        riser_pos = self.positions[riser_index]

        distances = np.sqrt(np.sum((self.positions - riser_pos) ** 2, axis=1))

        within_range = distances <= feeding_distance

        return np.where(within_range)[0].tolist()

    def _feeding_distance(self, diameter: float) -> float:
        ratio = self.params.height_ratio
        height = diameter * ratio
        volume = np.pi * (diameter / 2) ** 2 * height
        area = 2 * np.pi * (diameter / 2) * height + np.pi * (diameter / 2) ** 2
        modulus = volume / area

        return self.params.feeding_constant * np.sqrt(modulus)

    def _covered_fraction(self, covered: set) -> float:
        return len(covered) / self.n_voxels

    def analyze_coverage(self) -> Dict[str, float]:
        if not self._recommendations:
            return {'covered_fraction': 0.0, 'uncovered_voxels': self.n_voxels}

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
        if not self._recommendations:
            return []

        best_recommendations = list(self._recommendations)
        best_coverage = self.analyze_coverage()['covered_fraction']

        for _ in range(iterations):
            for i, riser in enumerate(best_recommendations):
                for new_idx, new_pos in self._candidate_attachment_moves(riser):
                    new_zone = self._calculate_feeding_zone(
                        new_idx, riser.diameter
                    )

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

def chvorinov_time(
    modulus: float,
    mold_constant: float = 0.05
) -> float:
    return mold_constant * modulus ** 2
