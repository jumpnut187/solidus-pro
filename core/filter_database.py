"""
Ceramic foam filter database for the mold-filling stage.

Mirrors :mod:`core.riser_design`'s ``load_riser_database`` pattern: a CSV of
standard vendor sizes at the repository root, a dataclass per row, and a
selection helper that picks the smallest catalog entry meeting a requirement.

Scope
-----
Zirconia ceramic foam filters rated for **steel** -- the Foseco STELEX family.
Foseco's other filter lines are deliberately excluded: ``STELEX PrO`` is
carbon-bonded alumina, ``SEDEX`` is silicon carbide and rated for iron, and
``SIVEX FC`` is for aluminium.

Provenance
----------
``filter_database.csv`` reproduces all 104 published cells of *Foseco Ferrous
Foundryman's Handbook* (Brown, ed., Butterworth-Heinemann, 2000) Table 18.2
(carbon/low-alloy steel, p.287) and Table 18.3 (high-alloy/stainless, p.288),
extracted from the publisher's born-digital PDF text layer and cross-checked
against 200 dpi renders of both pages. Those tables are the only published
Foseco source for filter flow rate and capacity: the public STELEX ZR and ZR
ULTRA datasheets say only that "further information regarding filter sizes,
flow rates and filter capacities can be obtained from your local Foseco team".

Each table is 13 sizes (6 square, 7 round) x {lip, bottom pour} x {Zr/Ti,
CaSi/Al deoxidation}, giving a capacity in kg and a flow rate in kg/sec.

What is and is not an area law
------------------------------
**Flow rate is an exact area law for square filters.**
``k_flow * front_face_area_cm2`` reproduces every published square flow cell to
within 0.96 %, all of it explained by the table being printed to one decimal.
The exact hits at 100x100x25 (10.0 / 12.5 / 14.0 / 17.5 = k x 100 with zero
residual) indicate Foseco generated the column from face-area constants.

For **round** filters the same constants drift up to 3.45 % (worst: 60 dia. x 25
high-alloy lip pour, 3.96 predicted vs 4.1 published), which is more than
rounding can explain -- so treat the law as a good approximation there, not as
the vendor's own arithmetic, and prefer the published cell.

Flow rate is independent of deoxidation practice: the Zr/Ti and CaSi/Al flow
columns are identical in all 26 rows.

**Capacity is not.** It fits within +/-1.6 % for the five square sizes 25-30 mm
thick, but breaks in two places:

- The 50x50x20 -- the only 20 mm-thick size -- runs 6-8 % *below* the area law
  in all eight columns (measured ratio 0.926-0.941, mean 0.934). An area law
  would over-predict how much metal that filter can pass, i.e. it fails in the
  unsafe direction. :data:`THIN_FILTER_CAPACITY_FACTOR` corrects it.
- The round sizes drift up to 6.6 % the other way, and their implied per-size
  constants range 1.49-1.59 rather than sitting on one value. Round capacity is
  therefore taken from the catalog only, never computed.

So prefer the catalog lookup for capacity; :func:`capacity_kg_for_area` exists
for filters modelled at non-catalog sizes and is explicitly an approximation.

Foseco's own hedge, which the tool should carry: both captions read "*Typical*
capacity", and p.284 states that capacity "depends on many factors, including
pouring temperature, alloy composition, dirtiness of the metal and metallostatic
pressure on the filter. An accurate prediction of filter capacity for a specific
alloy in a specific foundry is difficult to make."

No permeability, porosity or pressure-drop coefficient is published for any
STELEX product, so none is stored here. See :mod:`physics.fluid_flow` for how a
throughput rating becomes a restriction the solver can apply.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from core.error_logger import logger
except ImportError:  # pragma: no cover - mirrors core/part_manager.py
    from core.error_logger_dummy import logger  # type: ignore[assignment]


#: Steel classes the Foseco tables distinguish. Flow rate differs by ~40 %
#: between them, so picking the wrong one is not a rounding error.
STEEL_CARBON_LOW_ALLOY = "carbon_low_alloy"
STEEL_HIGH_ALLOY_STAINLESS = "high_alloy_stainless"
STEEL_CLASSES = (STEEL_CARBON_LOW_ALLOY, STEEL_HIGH_ALLOY_STAINLESS)

#: How the metal reaches the filter. Bottom pouring gives a higher rate than
#: lip pouring because the metallostatic head is larger and better sustained.
POUR_MODE_LIP = "lip"
POUR_MODE_BOTTOM = "bottom"
POUR_MODES = (POUR_MODE_LIP, POUR_MODE_BOTTOM)

#: Ladle deoxidation practice. Affects capacity only, never flow rate:
#: "titanium and zirconium will dirty the metal and reduce the capacity of the
#: filter" (Handbook p.284). CaSi/Al is the cleaner practice.
DEOX_ZR_TI = "zr_ti"
DEOX_CASI_AL = "casi_al"
DEOXIDATION_PRACTICES = (DEOX_ZR_TI, DEOX_CASI_AL)

#: Filter face shapes present in the catalog.
SHAPE_SQUARE = "square"
SHAPE_ROUND = "round"
SHAPES = (SHAPE_SQUARE, SHAPE_ROUND)

#: Mass flow per unit filter front-face area, kg/s per cm^2.
#:
#: Keyed on (steel_class, pour_mode) only -- deliberately NOT on deoxidation,
#: because the published flow columns are identical across the two practices.
#: Keying flow on deoxidation would invent a dependence the source does not have.
FLOW_RATE_KG_S_PER_CM2: Dict[Tuple[str, str], float] = {
    (STEEL_CARBON_LOW_ALLOY, POUR_MODE_LIP): 0.100,
    (STEEL_CARBON_LOW_ALLOY, POUR_MODE_BOTTOM): 0.125,
    (STEEL_HIGH_ALLOY_STAINLESS, POUR_MODE_LIP): 0.140,
    (STEEL_HIGH_ALLOY_STAINLESS, POUR_MODE_BOTTOM): 0.175,
}

#: Total throughput before blockage, kg per cm^2, fitted to the square sizes of
#: 25-30 mm thickness only. Keyed on all three axes, because capacity -- unlike
#: flow -- does depend on deoxidation practice.
#:
#: Note the CaSi/Al-to-Zr/Ti ratio is ~1.50 for three columns but ~1.67 for
#: high-alloy bottom pour, so these cannot be regenerated by scaling the Zr/Ti
#: half by one factor.
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

#: Thickness at or above which the capacity area law is validated.
CAPACITY_VALIDATED_THICKNESS_MM = 25.0

#: Capacity de-rating for filters thinner than
#: :data:`CAPACITY_VALIDATED_THICKNESS_MM`.
#:
#: The 50x50x20 is the range's only 20 mm filter and its published capacity sits
#: at 0.926-0.941 of the area law across all eight columns (mean 0.934). Dirt
#: holding scales with ceramic body, not only face area. Without this the
#: smallest STELEX is over-rated by 6-8 % -- unsafe, since it would pass more
#: steel through the filter than the vendor says it can take.
THIN_FILTER_CAPACITY_FACTOR = 0.934

#: Minimum ratio of filter front-face area to choke (sprue base) area.
#:
#: Foseco's two sources disagree: the STELEX ZR / ZR ULTRA / PrO datasheets say
#: "at least 3.0 times larger than the calculated choke area", while the
#: Handbook ch.18 specifies sprue 1 : filter 4.5 for steel. Both are reported
#: and neither is resolved -- that is the user's engineering judgement.
MIN_FILTER_TO_CHOKE_AREA_RATIO = 3.0
RECOMMENDED_FILTER_TO_CHOKE_AREA_RATIO = 4.5

#: Which Foseco table applies to each alloy in :mod:`core.materials`.
#:
#: The catalog carries three steels and four non-ferrous alloys. The Foseco
#: steel tables do not apply to the latter, and this module refuses to guess a
#: flow rate for them rather than silently returning a steel number.
ALLOY_STEEL_CLASS: Dict[str, str] = {
    "WCB": STEEL_CARBON_LOW_ALLOY,
    "CF8": STEEL_HIGH_ALLOY_STAINLESS,
    "CA6NM": STEEL_HIGH_ALLOY_STAINLESS,
}

#: Fallback density for converting the vendor's kg/s rating into a face
#: velocity. Matches the melt density used in the ceramic-filter literature
#: (Asad et al. 2021). Callers with an alloy in hand should pass its
#: ``density_liquid`` instead -- WCB is 7200 kg/m^3.
LIQUID_STEEL_DENSITY_KG_M3 = 7000.0


@dataclass(frozen=True)
class FilterSpec:
    """
    One published cell: a vendor size at one steel class, pour mode and
    deoxidation practice.

    Attributes:
        brand: Vendor, e.g. ``"Foseco"``.
        product_line: Product family, e.g. ``"STELEX"``.
        material: Filter ceramic, e.g. ``"zirconia"``.
        nominal_size: Vendor size label, e.g. ``"75x75x25"`` or ``"90dia x 25"``.
        shape: One of :data:`SHAPES`.
        width_mm: Face width; ``None`` for round filters.
        height_mm: Face height; ``None`` for round filters.
        diameter_mm: Face diameter; ``None`` for square filters.
        thickness_mm: Flow-direction thickness.
        front_face_area_cm2: Face area.
        steel_class: One of :data:`STEEL_CLASSES`.
        pour_mode: One of :data:`POUR_MODES`.
        deoxidation: One of :data:`DEOXIDATION_PRACTICES`.
        flow_rate_kg_s: Published mass flow rate through the filter.
        capacity_kg: Published total throughput before blockage.
        max_pour_temp_c: Vendor maximum pouring temperature.
        source: Which table the row came from.
    """

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
        """Flow-direction thickness in metres, for the solver."""
        return self.thickness_mm / 1000.0

    @property
    def front_face_area_m2(self) -> float:
        """Front-face area in m^2, for the solver."""
        return self.front_face_area_cm2 / 10_000.0

    def rated_superficial_velocity_m_s(
        self, density_liquid_kg_m3: float = LIQUID_STEEL_DENSITY_KG_M3
    ) -> float:
        """
        Rated superficial (empty-tube) velocity through the filter face.

        This is the quantity the hydrodynamic model actually needs: the vendor
        rating converted from a mass flow to the face velocity it implies.

        Pass the alloy's own ``density_liquid``. The default is the generic melt
        density used in the ceramic-filter literature, but WCB is 7200 kg/m^3,
        so taking the default for a specific alloy is a 3 % error in a quantity
        that is then compared against a rated threshold.

        Because flow rate is an exact area law, this velocity is the *same* for
        every catalog size within a steel class and pour mode -- which is what
        makes it usable as an acceptance criterion at all.

        Note this is a *rated* velocity, not a measured one; see the module
        docstring on why no measured permeability exists for these products.
        """
        return self.flow_rate_kg_s / (
            float(density_liquid_kg_m3) * self.front_face_area_m2
        )


def _database_path() -> str:
    """Absolute path to the filter CSV at the repository root."""
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path: str = os.path.join(module_dir, "filter_database.csv")
    return path


def _optional_float(value: str) -> Optional[float]:
    text = (value or "").strip()
    return float(text) if text else None


def load_filter_database(path: Optional[str] = None) -> List[FilterSpec]:
    """
    Load the ceramic filter catalog from CSV.

    Malformed rows are skipped rather than raising, matching
    :func:`core.riser_design.load_riser_database`: a corrupt row should cost the
    user one filter option, not the whole run.

    Args:
        path: Override the default repository-root CSV. Used by tests.

    Returns:
        Filter specs sorted by shape, then face area, then steel class, pour
        mode and deoxidation. Empty if the database is missing or unreadable.
    """
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
    """
    Which Foseco steel table applies to an alloy, or ``None`` if none does.

    Returns ``None`` for the catalog's non-ferrous alloys (the bronzes, Monel
    and Inconel). Callers must treat that as "the vendor rating does not apply"
    rather than substituting a steel figure.
    """
    return ALLOY_STEEL_CLASS.get(str(alloy_name).strip().upper())


def flow_rate_kg_s_for_area(
    front_face_area_cm2: float,
    steel_class: str,
    pour_mode: str = POUR_MODE_LIP,
) -> float:
    """
    Rated mass flow through a filter of arbitrary front-face area.

    Applies the exact area law the published flow columns follow, so a filter
    modelled at a non-catalog size still gets a defensible rating. Valid for
    round faces as well as square.

    Deliberately takes no deoxidation argument: the published flow columns are
    identical across the two practices.

    Args:
        front_face_area_cm2: Filter front-face area.
        steel_class: One of :data:`STEEL_CLASSES`.
        pour_mode: One of :data:`POUR_MODES`.

    Returns:
        Mass flow rate in kg/s.

    Raises:
        ValueError: If the steel class or pour mode is unknown, or the area is
            not positive. Guessing here would produce a plausible-looking flow
            rate off by 40 %, which is exactly the failure this refuses to make.
    """
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
    """
    Approximate total throughput before blockage, for a non-catalog filter.

    Unlike :func:`flow_rate_kg_s_for_area` this is an *approximation*, not the
    vendor's own rule -- capacity is not a pure area law. Prefer a catalog
    lookup where one applies, and see the module docstring for where and how
    badly this drifts. Round filters in particular scatter up to 6.6 %, so their
    published values should be used directly.

    ``thickness_mm`` is required to avoid over-rating thin filters: below
    :data:`CAPACITY_VALIDATED_THICKNESS_MM` the result is de-rated by
    :data:`THIN_FILTER_CAPACITY_FACTOR`.

    Args:
        front_face_area_cm2: Filter front-face area.
        steel_class: One of :data:`STEEL_CLASSES`.
        pour_mode: One of :data:`POUR_MODES`.
        deoxidation: One of :data:`DEOXIDATION_PRACTICES`.
        thickness_mm: Flow-direction thickness.

    Returns:
        Approximate capacity in kg.

    Raises:
        ValueError: On an unknown key combination or a non-positive area.
    """
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
    """
    Smallest catalog filter that carries the required flow rate.

    Mirrors :func:`core.riser_design.select_standard_riser`.

    Args:
        required_flow_kg_s: Flow the gating system must pass.
        filter_db: Catalog, typically from :func:`load_filter_database`.
        steel_class: One of :data:`STEEL_CLASSES`.
        pour_mode: One of :data:`POUR_MODES`.
        deoxidation: One of :data:`DEOXIDATION_PRACTICES`.
        shape: Restrict to one of :data:`SHAPES`, or ``None`` for either.

    Returns:
        The smallest matching spec, or ``None`` if the required flow exceeds
        every catalog size -- a real condition meaning the design needs more
        than one filter, not an error.
    """
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
    """
    Identify which catalog filter a modelled body most likely represents.

    A user places a filter as a solid in their CAD assembly, so its face area is
    whatever they drew. This maps that back to a catalog entry for labelling and
    for the capacity warning.

    Args:
        front_face_area_cm2: Measured front-face area of the placed body.
        filter_db: Catalog, typically from :func:`load_filter_database`.
        steel_class: One of :data:`STEEL_CLASSES`.
        pour_mode: One of :data:`POUR_MODES`.
        deoxidation: One of :data:`DEOXIDATION_PRACTICES`.
        shape: Restrict to one of :data:`SHAPES`, or ``None`` for either.
        tolerance: Maximum relative area difference to still call it a match.

    Returns:
        The closest catalog spec within ``tolerance``, else ``None``.
    """
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
