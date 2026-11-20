from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from core.error_logger import logger
except ImportError:
    from core.error_logger_dummy import logger

STEEL_CARBON_LOW_ALLOY = "carbon_low_alloy"
STEEL_HIGH_ALLOY_STAINLESS = "high_alloy_stainless"
STEEL_CLASSES = (STEEL_CARBON_LOW_ALLOY, STEEL_HIGH_ALLOY_STAINLESS)

POUR_MODE_LIP = "lip"
POUR_MODE_BOTTOM = "bottom"
POUR_MODES = (POUR_MODE_LIP, POUR_MODE_BOTTOM)

DEOX_ZR_TI = "zr_ti"
DEOX_CASI_AL = "casi_al"
DEOXIDATION_PRACTICES = (DEOX_ZR_TI, DEOX_CASI_AL)

SHAPE_SQUARE = "square"
SHAPE_ROUND = "round"
SHAPES = (SHAPE_SQUARE, SHAPE_ROUND)

FLOW_RATE_KG_S_PER_CM2: Dict[Tuple[str, str], float] = {
    (STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP): 0.100,
    (STEEL_CARBON_LOW_ALLOY, POUR_MODE_BOTTOM): 0.125,
    (STEEL_HIGH_ALLOY_STAINLESS, POUR_MODE_LIP): 0.140,
    (STEEL_HIGH_ALLOY_STAINLESS, POUR_MODE_BOTTOM): 0.175,
}

CAPACITY_KG_PER_CM2: Dict[Tuple[str, str, str], float] = {
    (STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP, DEOX_ZR_TI): 1.50,
    (STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP, DEOX_CASI_AL): 2.26,
    (STEEL_CARBON_LOW_ALLOY, POUR_MODE_BOTTOM, DEOX_ZR_TI): 1.96,
    (STEEL_CARBON_LOW_ALLOY, POUR_MODE_BOTTOM, DEOX_CASI_AL): 2.93,
    (STEEL_HIGH_ALLOY_STAINLESS, POUR_MODE_LIP, DEOX_ZR_TI): 1.96,
    (STEEL_HIGH_ALLOY_STAINLESS, POUR_MODE_LIP, DEOX_CASI_AL): 2.93,
    (STEEL_HIGH_ALLOY_STAINLESS, POUR_MODE_BOTTOM, DEOX_ZR_TI): 2.64,
    (STEEL_HIGH_ALLOY_STAINLESS, POUR_MODE_BOTTOM, DEOX_CASI_AL): 4.40,
}

CAPACITY_VALIDATED_THICKNESS_MM = 25.0

THIN_FILTER_CAPACITY_FACTOR = 0.934

MIN_FILTER_TO_CHOKE_AREA_RATIO = 3.0
RECOMMENDED_FILTER_TO_CHOKE_AREA_RATIO = 4.5

ALLOY_STEEL_CLASS: Dict[str, str] = {
    "WCB": STEEL_CARBON_LOW_ALLOY,
    "CF8": STEEL_HIGH_ALLOY_STAINLESS,
    "CA6NM": STEEL_HIGH_ALLOY_STAINLESS,
}

LIQUID_STEEL_DENSITY_KG_M3 = 7000.0

@dataclass(frozen=True)
class FilterSpec:

    brand: str
    product_line: str
    material: str
    nominal_size: str
    shape: str
    width_mm: Optional[float]
    height_mm: Optional[float]
    diameter_mm: Optional[float]
    thickness_mm: float
    front_face_area_cm2: float
    steel_class: str
    pour_mode: str
    deoxidation: str
    flow_rate_kg_s: float
    capacity_kg: float
    max_pour_temp_c: float
    source: str

    @property
    def thickness_m(self) -> float:
        return self.thickness_mm / 1000.0

    @property
    def front_face_area_m2(self) -> float:
        return self.front_face_area_cm2 / 10_000.0

    def rated_superficial_velocity_m_s(
        self, density_liquid_kg_m3: float = LIQUID_STEEL_DENSITY_KG_M3
    ) -> float:
        return self.flow_rate_kg_s / (
            float(density_liquid_kg_m3) * self.front_face_area_m2
        )

def _database_path() -> str:
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path: str = os.path.join(module_dir, "filter_database.csv")
    return path

def _optional_float(value: str) -> Optional[float]:
    text = (value or "").strip()
    return float(text) if text else None

def load_filter_database(path: Optional[str] = None) -> List[FilterSpec]:
    db_path = path or _database_path()
    specs: List[FilterSpec] = []

    if not os.path.exists(db_path):
        logger.warning(f"Filter database not found at {db_path}")
        return specs

    try:
        with open(db_path, "r", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    steel_class = row["steel_class"].strip()
                    pour_mode = row["pour_mode"].strip()
                    deoxidation = row["deoxidation"].strip()
                    shape = row["shape"].strip()
                    if (
                        steel_class not in STEEL_CLASSES
                        or pour_mode not in POUR_MODES
                        or deoxidation not in DEOXIDATION_PRACTICES
                        or shape not in SHAPES
                    ):
                        logger.warning(
                            f"Filter database: skipping row with unknown "
                            f"steel_class={steel_class!r} pour_mode={pour_mode!r} "
                            f"deoxidation={deoxidation!r} shape={shape!r}"
                        )
                        continue
                    specs.append(
                        FilterSpec(
                            brand=row["brand"].strip(),
                            product_line=row["product_line"].strip(),
                            material=row["material"].strip(),
                            nominal_size=row["nominal_size"].strip(),
                            shape=shape,
                            width_mm=_optional_float(row["width_mm"]),
                            height_mm=_optional_float(row["height_mm"]),
                            diameter_mm=_optional_float(row["diameter_mm"]),
                            thickness_mm=float(row["thickness_mm"]),
                            front_face_area_cm2=float(row["front_face_area_cm2"]),
                            steel_class=steel_class,
                            pour_mode=pour_mode,
                            deoxidation=deoxidation,
                            flow_rate_kg_s=float(row["flow_rate_kg_s"]),
                            capacity_kg=float(row["capacity_kg"]),
                            max_pour_temp_c=float(row["max_pour_temp_c"]),
                            source=row["source"].strip(),
                        )
                    )
                except (ValueError, KeyError, AttributeError):
                    continue
    except OSError as exc:
        logger.error(f"Error loading filter database: {exc}")
        return []

    specs.sort(
        key=lambda s: (
            s.shape,
            s.front_face_area_cm2,
            s.steel_class,
            s.pour_mode,
            s.deoxidation,
        )
    )
    return specs

def steel_class_for_alloy(alloy_name: str) -> Optional[str]:
    return ALLOY_STEEL_CLASS.get(str(alloy_name).strip().upper())

def flow_rate_kg_s_for_area(
    front_face_area_cm2: float,
    steel_class: str,
    pour_mode: str = POUR_MODE_LIP,
) -> float:
    if front_face_area_cm2 <= 0:
        raise ValueError(f"front_face_area_cm2 must be positive, got {front_face_area_cm2}")
    key = (steel_class, pour_mode)
    if key not in FLOW_RATE_KG_S_PER_CM2:
        raise ValueError(
            f"No Foseco flow rating for steel_class={steel_class!r} "
            f"pour_mode={pour_mode!r}; expected one of "
            f"{sorted(FLOW_RATE_KG_S_PER_CM2)}"
        )
    return FLOW_RATE_KG_S_PER_CM2[key] * float(front_face_area_cm2)

def capacity_kg_for_area(
    front_face_area_cm2: float,
    steel_class: str,
    pour_mode: str = POUR_MODE_LIP,
    deoxidation: str = DEOX_ZR_TI,
    thickness_mm: float = CAPACITY_VALIDATED_THICKNESS_MM,
) -> float:
    if front_face_area_cm2 <= 0:
        raise ValueError(f"front_face_area_cm2 must be positive, got {front_face_area_cm2}")
    key = (steel_class, pour_mode, deoxidation)
    if key not in CAPACITY_KG_PER_CM2:
        raise ValueError(
            f"No Foseco capacity rating for steel_class={steel_class!r} "
            f"pour_mode={pour_mode!r} deoxidation={deoxidation!r}; expected one "
            f"of {sorted(CAPACITY_KG_PER_CM2)}"
        )
    capacity = CAPACITY_KG_PER_CM2[key] * float(front_face_area_cm2)
    if float(thickness_mm) < CAPACITY_VALIDATED_THICKNESS_MM:
        capacity *= THIN_FILTER_CAPACITY_FACTOR
    return capacity

def _matching(
    filter_db: Sequence[FilterSpec],
    steel_class: str,
    pour_mode: str,
    deoxidation: str,
    shape: Optional[str],
) -> List[FilterSpec]:
    return [
        spec
        for spec in filter_db
        if spec.steel_class == steel_class
        and spec.pour_mode == pour_mode
        and spec.deoxidation == deoxidation
        and (shape is None or spec.shape == shape)
    ]

def select_filter(
    required_flow_kg_s: float,
    filter_db: Sequence[FilterSpec],
    steel_class: str,
    pour_mode: str = POUR_MODE_LIP,
    deoxidation: str = DEOX_ZR_TI,
    shape: Optional[str] = SHAPE_SQUARE,
) -> Optional[FilterSpec]:
    candidates = [
        spec
        for spec in _matching(filter_db, steel_class, pour_mode, deoxidation, shape)
        if spec.flow_rate_kg_s >= required_flow_kg_s
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda s: s.front_face_area_cm2)

def match_filter_by_area(
    front_face_area_cm2: float,
    filter_db: Sequence[FilterSpec],
    steel_class: str,
    pour_mode: str = POUR_MODE_LIP,
    deoxidation: str = DEOX_ZR_TI,
    shape: Optional[str] = None,
    tolerance: float = 0.15,
) -> Optional[FilterSpec]:
    if front_face_area_cm2 <= 0:
        return None
    candidates = _matching(filter_db, steel_class, pour_mode, deoxidation, shape)
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda s: abs(s.front_face_area_cm2 - front_face_area_cm2),
    )
    relative_error = abs(best.front_face_area_cm2 - front_face_area_cm2) / front_face_area_cm2
    return best if relative_error <= tolerance else None
