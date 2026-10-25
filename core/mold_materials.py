"""Molding and core sands: temperature-dependent thermal properties.

Each sand is a bonded, packed aggregate as it sits in a mold or core: the
density is the bulk (packed) density, not the grain density, and k(T) is
the effective conductivity of the packed bed, which rises steeply at high
temperature as radiation across the pores takes over.

The thermal solve carries a sand as a :class:`physics.heat_transfer.TabulatedBody`:
the apparent volumetric heat capacity rho * cp(T) with the sand's latent
effects added as peaks, integrated to an enthalpy table, so a cell crossing
a peak in one step still pays its heat:

- free water in green sand evaporates at the boiling point
  (2257 kJ/kg of water);
- quartz-bearing sands (silica, green sand) take the alpha-beta quartz
  inversion at 573 °C.

Codes in :data:`SAND_MATERIALS` are stored in saved projects and settings
(``MoldParams.mold_material_code`` / ``core_material_code``). Append new
sands; never renumber. Code 0 is the constant custom sand of
ThermalBoundaryParams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from core.solidification_params import SAND_MATERIAL_CUSTOM, ThermalBoundaryParams
from physics.heat_transfer import TabulatedBody

#: Latent heat of evaporation of water, J/kg.
WATER_LATENT_HEAT = 2.257e6
#: Specific heat of liquid water, J/(kg K), carried by green sand below 100 °C.
WATER_SPECIFIC_HEAT = 4180.0
#: Temperature band (°C) the evaporation peak is spread over. Popielarski and
#: Ignaszak fitted 98-102 °C for 3.5 % water; the enthalpy update holds a
#: cell on the band however large its step, so the band can stay narrow.
WATER_EVAPORATION_BAND = (100.0, 104.0)
#: Alpha-beta quartz inversion: 0.729 kJ/mol = 12.1 kJ/kg of quartz
#: (NIST-JANAF), at 573 °C. About 0.8 % of the sand's enthalpy to 1300 °C:
#: it matters to expansion defects far more than to heat flow.
QUARTZ_INVERSION_HEAT = 12.1e3
QUARTZ_INVERSION_BAND = (568.0, 578.0)

#: Source keys used in SandMaterial.sources.
SOURCES = {
    "midea_shah": "Midea & Shah, Mold material thermophysical data, AFS Trans. 110 (2002) 121-136 "
    "(tables in Modern Casting, Aug 2004)",
    "janaf": "NIST-JANAF thermochemical tables (Chase 1998), Shomate fits via the NIST WebBook",
    "ziemniak": "Ziemniak et al. 2007, FeCr2O4 heat capacity by DSC (OSTI 903084)",
    "robie": "Robie & Hemingway 1995, USGS Bulletin 2131 (forsterite heat capacity)",
    "imono_1977": "Heat transfer properties of some kinds of sand molds, IMONO 49(12) 747 (1977)",
    "carbo": "CARBO ACCUCAST technical data sheet (values at 1100 °C, incl. zircon, chromite, silica)",
    "itochu": "Itochu Ceratech CERABEADS data (bulk densities, refractoriness, expansion)",
    "svidro": "Svidro et al., Arch. Metall. Mater. 62 (2017) 795 (bulk densities, expansion)",
    "popielarski": "Popielarski & Ignaszak, green sand over-moisture zone (moisture, k of green sand)",
    "brooks": "Brooks & Beckermann 2006, burn-on and mold penetration in steel castings (SFSA)",
    "patent_5355931": "US Patent 5,355,931, Table I (unbonded silica, chromite, silicon carbide)",
}


@dataclass(frozen=True)
class SandMaterial:
    """A molding or core sand as packed in the mold."""

    code: int
    key: str
    label: str
    #: Table temperatures, °C, ascending.
    temp_points: Tuple[float, ...]
    #: Effective conductivity of the packed sand, W/(m K), at temp_points.
    conductivity: Tuple[float, ...]
    #: Sensible specific heat, J/(kg K), at temp_points (latent peaks excluded).
    specific_heat: Tuple[float, ...]
    #: Packed bulk density, kg/m^3: one value, or one per temp_point (the
    #: measured density falls as binder burns and grains rearrange).
    density: Tuple[float, ...]
    #: Mass fraction of free water (green sand).
    moisture: float = 0.0
    #: Mass fraction of quartz whose inversion the cp table leaves out.
    quartz_fraction: float = 0.0
    #: Sintering / fusion onset of the grain, °C.
    sintering_temperature: float = 0.0
    uses: str = ""
    #: Keys of SOURCES, and a note on which values are estimates.
    sources: Tuple[str, ...] = ()
    notes: str = ""

    @property
    def bulk_density(self) -> float:
        """Packed bulk density at room temperature, kg/m^3."""
        return float(self.density[0])

    def _density_at(self, temps: np.ndarray) -> np.ndarray:
        if len(self.density) == 1:
            return np.full(len(temps), float(self.density[0]))
        return np.interp(temps, self.temp_points, self.density)

    def volumetric_heat_capacity(self) -> Tuple[np.ndarray, np.ndarray]:
        """(temps, rho(T) * cp_apparent(T)) with the latent peaks folded in, J/(m^3 K).

        The cells keep their volume, so the measured density change goes into
        the heat capacity (rho(T) cp(T) per unit volume). Peaks are top hats
        over their bands: the extra capacity times the band width is the latent
        heat per volume. Water also adds its own sensible heat below boiling.
        """
        temps = np.asarray(self.temp_points, dtype=np.float64)
        peaks = []
        if self.moisture > 0.0:
            peaks.append((WATER_EVAPORATION_BAND, self.moisture * WATER_LATENT_HEAT))
        if self.quartz_fraction > 0.0:
            peaks.append((QUARTZ_INVERSION_BAND, self.quartz_fraction * QUARTZ_INVERSION_HEAT))
        edges = [t for (lo, hi), _ in peaks for t in (lo - 0.01, lo, hi, hi + 0.01)]
        grid = np.unique(np.concatenate([temps, np.asarray(edges, dtype=np.float64)]))
        cp = np.interp(grid, temps, np.asarray(self.specific_heat, dtype=np.float64))
        if self.moisture > 0.0:
            cp = cp + np.where(grid < WATER_EVAPORATION_BAND[0], self.moisture * WATER_SPECIFIC_HEAT, 0.0)
        for (lo, hi), heat in peaks:
            inside = (grid >= lo) & (grid <= hi)
            cp[inside] += heat / (hi - lo)
        return grid, self._density_at(grid) * cp

    def tables(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(temps, k, rho*cp_apparent) on one grid for a TabulatedBody."""
        grid, rho_cp = self.volumetric_heat_capacity()
        k = np.interp(grid, np.asarray(self.temp_points), np.asarray(self.conductivity))
        return grid, k, rho_cp

    def thermal_effusivity(self, temperature: float = 20.0) -> float:
        """sqrt(k rho cp) at ``temperature`` (sensible cp), W s^0.5 / (m^2 K): the sand's chilling power."""
        k = float(np.interp(temperature, self.temp_points, self.conductivity))
        cp = float(np.interp(temperature, self.temp_points, self.specific_heat))
        rho = float(self._density_at(np.array([temperature]))[0])
        return float(np.sqrt(k * rho * cp))


_T100 = (20.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
         1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0)
_T200 = (20.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0)

#: Pure-quartz heat capacity (NIST-JANAF), J/(kg K), on _T100.
_QUARTZ_CP = (742.0, 855.0, 968.0, 1052.0, 1125.0, 1203.0, 1126.0, 1143.0, 1160.0, 1177.0, 1193.0,
              1210.0, 1227.0, 1243.0, 1260.0, 1277.0, 1294.0)
#: Mullite (NIST-JANAF), J/(kg K), on _T100.
_MULLITE_CP = (763.0, 889.0, 992.0, 1062.0, 1112.0, 1150.0, 1179.0, 1201.0, 1219.0, 1233.0, 1244.0,
               1254.0, 1263.0, 1274.0, 1285.0, 1292.0, 1300.0)
#: Synthetic ceramic bead conductivity (estimate anchored to CARBO and Naigai single values).
_CERAMIC_K = (0.60, 0.598, 0.595, 0.59, 0.58, 0.585, 0.59, 0.60, 0.62, 0.65, 0.685,
              0.72, 0.78, 0.84, 0.90, 0.975, 1.05)

#: Every sand by code. Append only (codes are saved in projects).
SAND_MATERIALS: Dict[int, SandMaterial] = {
    m.code: m
    for m in (
        SandMaterial(
            code=1,
            key="green_sand",
            label="Green sand (silica, bentonite, 3.5 % water)",
            temp_points=(20.0, 95.0, 110.0, 200.0, 300.0, 500.0, 700.0, 900.0, 1000.0,
                         1200.0, 1400.0, 1600.0),
            conductivity=(0.95, 0.90, 0.65, 0.63, 0.62, 0.62, 0.62, 0.65, 0.70, 0.75, 0.85, 1.10),
            specific_heat=(900.0, 950.0, 957.0, 994.0, 1035.0, 1117.0, 1198.0, 1280.0, 1321.0,
                           1403.0, 1485.0, 1567.0),
            density=(1500.0,),
            moisture=0.035,
            sintering_temperature=1450.0,
            uses="Clay-bonded molds for iron and smaller steel castings.",
            sources=("midea_shah", "popielarski"),
            notes="cp: Midea & Shah machine-molded green sand (20 °C extrapolated). k: blend of the "
            "Midea & Shah machine- and high-pressure-molded sets with Popielarski's moist values, "
            "dropping as the sand dries at the boiling point (estimate, about +-30 %). Moisture: "
            "3.5 % water, latent heat over 100-104 °C and liquid-water cp below.",
        ),
        SandMaterial(
            code=2,
            key="silica_nobake",
            label="Silica sand, chemically bonded (phenolic urethane / furan no-bake)",
            temp_points=_T100,
            conductivity=(0.82, 0.785, 0.732, 0.693, 0.668, 0.660, 0.668, 0.691, 0.733, 0.790, 0.863,
                          0.951, 1.054, 1.175, 1.303, 1.439, 1.582),
            specific_heat=(1000.0, 1118.0, 1114.0, 1160.0, 1178.0, 1196.0, 1214.0, 1232.0, 1250.0,
                           1268.0, 1285.0, 1303.0, 1321.0, 1339.0, 1356.0, 1374.0, 1392.0),
            density=(1488.0, 1482.0, 1475.0, 1468.0, 1462.0, 1456.0, 1448.0, 1442.0, 1437.0, 1433.0,
                     1428.0, 1422.0, 1417.0, 1413.0, 1401.0, 1382.0, 1360.0),
            sintering_temperature=1450.0,
            uses="No-bake molds; the usual steel mold sand.",
            sources=("midea_shah",),
            notes="Midea & Shah PUNB mold sand (CBUMS-M), the set validated against a steel casting "
            "trial; the 20 °C point is extrapolated. Measured cp already holds the binder and the "
            "smoothed quartz inversion.",
        ),
        SandMaterial(
            code=3,
            key="silica_core",
            label="Silica core sand, fine (furan / phenolic urethane cold box)",
            temp_points=_T100,
            conductivity=(0.47, 0.478, 0.505, 0.517, 0.516, 0.511, 0.507, 0.507, 0.517, 0.547, 0.600,
                          0.682, 0.805, 0.973, 1.194, 1.479, 1.836),
            specific_heat=(900.0, 1045.0, 1071.0, 1096.0, 1120.0, 1143.0, 1167.0, 1191.0, 1215.0,
                           1238.0, 1262.0, 1285.0, 1309.0, 1333.0, 1356.0, 1380.0, 1404.0),
            density=(1565.0, 1561.0, 1557.0, 1553.0, 1548.0, 1542.0, 1535.0, 1530.0, 1526.0, 1522.0,
                     1518.0, 1512.0, 1508.0, 1503.0, 1498.0, 1493.0, 1488.0),
            sintering_temperature=1450.0,
            uses="Blown and no-bake cores.",
            sources=("midea_shah",),
            notes="Midea & Shah furan core sand (CBFMS-C); 20 °C point estimated.",
        ),
        SandMaterial(
            code=4,
            key="ceramic",
            label="Ceramic sand (sintered mullite beads, Cerabeads type)",
            temp_points=_T100,
            conductivity=_CERAMIC_K,
            specific_heat=_MULLITE_CP,
            density=(1690.0,),
            sintering_temperature=1825.0,
            uses="Cores and facing where silica expansion defects (veining) matter; low expansion.",
            sources=("janaf", "itochu", "carbo"),
            notes="cp: mullite (JANAF). Density: Cerabeads vibrated bulk. k: estimate anchored to "
            "0.75 at 1100 °C (CARBO LD) and 0.56 (Naigai), about +-30 %.",
        ),
        SandMaterial(
            code=5,
            key="chromite",
            label="Chromite sand",
            temp_points=_T100,
            conductivity=(0.95, 0.92, 0.88, 0.85, 0.82, 0.80, 0.78, 0.78, 0.78, 0.80, 0.83,
                          0.875, 0.92, 0.985, 1.05, 1.125, 1.20),
            specific_heat=(688.0, 739.0, 781.0, 809.0, 831.0, 848.0, 862.0, 874.0, 884.0, 894.0,
                           902.0, 908.0, 914.0, 919.0, 923.0, 926.0, 928.0),
            density=(2750.0,),
            sintering_temperature=1800.0,
            uses="Steel facing sand and cores at heavy sections; least burn-on and penetration.",
            sources=("ziemniak", "imono_1977", "carbo", "svidro", "brooks"),
            notes="cp: 1.15 x FeCr2O4 (DSC), matching the measured mold mean. k: estimate anchored to "
            "0.754 mean (IMONO) and 0.94 at 1100 °C (CARBO), about +-30 %.",
        ),
        SandMaterial(
            code=6,
            key="zircon",
            label="Zircon sand",
            temp_points=_T100,
            conductivity=(1.00, 0.96, 0.92, 0.885, 0.85, 0.825, 0.80, 0.78, 0.76, 0.755, 0.75,
                          0.775, 0.80, 0.85, 0.90, 0.96, 1.02),
            specific_heat=(538.0, 610.0, 673.0, 717.0, 749.0, 773.0, 790.0, 802.0, 811.0, 816.0,
                           818.0, 820.0, 820.0, 821.0, 822.0, 822.0, 822.0),
            density=(2950.0,),
            sintering_temperature=1540.0,
            uses="Steel and stainless facing, thin cores; lowest expansion. Dissociates near 1540-1676 °C.",
            sources=("janaf", "imono_1977", "carbo", "svidro"),
            notes="cp: zircon (JANAF), matching the measured mold mean. k: estimate anchored to 0.879 "
            "mean (IMONO) and 0.63 at 1100 °C (CARBO), about +-30 %. Chilling power is about "
            "1.1-1.3 x silica at mid temperature, not the 4 x often quoted.",
        ),
        SandMaterial(
            code=7,
            key="olivine",
            label="Olivine sand",
            temp_points=_T100,
            conductivity=(0.80, 0.78, 0.76, 0.745, 0.73, 0.725, 0.72, 0.73, 0.74, 0.77, 0.80,
                          0.85, 0.90, 0.96, 1.02, 1.085, 1.15),
            specific_heat=(784.0, 885.0, 964.0, 1018.0, 1059.0, 1094.0, 1125.0, 1151.0, 1175.0,
                           1195.0, 1213.0, 1227.0, 1239.0, 1249.0, 1255.0, 1259.0, 1260.0),
            density=(1850.0,),
            sintering_temperature=1480.0,
            uses="Manganese (Hadfield) steel molds and cores; basic, no silica reaction.",
            sources=("robie", "imono_1977", "svidro"),
            notes="cp: 0.93 x forsterite (Robie & Hemingway), matching the measured mold mean. k: "
            "estimate anchored to 0.754 mean (IMONO), about +-30 %.",
        ),
        SandMaterial(
            code=8,
            key="silica_shell",
            label="Shell sand (resin-coated silica)",
            temp_points=(22.0,) + _T100[1:],
            conductivity=(0.918, 0.961, 0.950, 0.894, 0.827, 0.764, 0.711, 0.672, 0.649, 0.643, 0.654,
                          0.680, 0.729, 0.796, 0.882, 0.988, 1.113),
            specific_heat=(790.0, 910.0, 1015.0, 1076.0, 1120.0, 1154.0, 1181.0, 1205.0, 1225.0,
                           1243.0, 1259.0, 1273.0, 1286.0, 1298.0, 1310.0, 1320.0, 1330.0),
            density=(1357.0, 1355.0, 1354.0, 1351.0, 1343.0, 1335.0, 1326.0, 1318.0, 1312.0, 1305.0,
                     1298.0, 1285.0, 1281.0, 1276.0, 1271.0, 1266.0, 1261.0),
            sintering_temperature=1450.0,
            uses="Shell molds and cores.",
            sources=("midea_shah",),
            notes="Midea & Shah shell sand (CBSMS-M).",
        ),
        SandMaterial(
            code=9,
            key="silica_dry",
            label="Silica sand, dry / unbonded",
            temp_points=_T100,
            conductivity=(0.62, 0.61, 0.60, 0.60, 0.60, 0.61, 0.64, 0.69, 0.74, 0.82, 0.92,
                          1.05, 1.18, 1.30, 1.42, 1.56, 1.70),
            specific_heat=tuple(round(0.95 * c, 1) for c in _QUARTZ_CP),
            density=(1500.0,),
            quartz_fraction=0.98,
            sintering_temperature=1450.0,
            uses="Lost foam and dry-sand molds.",
            sources=("janaf", "imono_1977", "carbo", "patent_5355931", "svidro"),
            notes="cp: 0.95 x quartz (JANAF), fitted to the measured dry-mold mean; the inversion is "
            "added. k: estimate through 0.63 mean (IMONO) and 1.14 at 1100 °C (CARBO).",
        ),
        SandMaterial(
            code=10,
            key="bauxite",
            label="Bauxite ceramic sand (high-alumina beads)",
            temp_points=(20.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0),
            conductivity=(0.66, 0.66, 0.655, 0.64, 0.65, 0.68, 0.75, 0.86, 0.99, 1.155),
            specific_heat=(773.0, 914.0, 1020.0, 1132.0, 1196.0, 1239.0, 1270.0, 1295.0, 1314.0, 1330.0),
            density=(1900.0,),
            sintering_temperature=1800.0,
            uses="Steel cores and facing (CARBO ACCUCAST ID, fused bauxite ceramsite).",
            sources=("janaf", "carbo"),
            notes="cp: alpha-alumina (JANAF). Density: between CARBO ID loose and packed. k: 1.1 x the "
            "ceramic bead estimate (CARBO ID 0.70 at 1100 °C).",
        ),
        SandMaterial(
            code=11,
            key="kerphalite",
            label="Alumina-silicate sand (andalusite, Kerphalite type)",
            temp_points=_T100,
            conductivity=_CERAMIC_K,
            specific_heat=_MULLITE_CP,
            density=(1650.0,),
            sintering_temperature=1800.0,
            uses="Low-expansion cores for iron and steel; 3D-printed cores.",
            sources=("janaf",),
            notes="No measured thermal data found: mullite cp and the ceramic bead k estimate.",
        ),
        SandMaterial(
            code=12,
            key="silicon_carbide",
            label="Silicon carbide sand",
            temp_points=(20.0, 1600.0),
            conductivity=(3.25, 3.25),
            specific_heat=(840.0, 840.0),
            density=(2000.0,),
            sintering_temperature=2200.0,
            uses="Chill sand for aluminum and lost foam; not for steel (silicon pickup).",
            sources=("patent_5355931",),
            notes="Single room-temperature values; chilling power about 2 x silica.",
        ),
    )
}


def get_sand(code: int) -> SandMaterial:
    """The database sand for ``code`` (KeyError for 0 or an unknown code)."""
    return SAND_MATERIALS[int(code)]


def is_database_sand(code: int) -> bool:
    """True for a code in :data:`SAND_MATERIALS` (not custom, not unknown)."""
    return int(code) in SAND_MATERIALS


def sand_label(code: int) -> str:
    if int(code) == SAND_MATERIAL_CUSTOM:
        return "Custom (constant properties)"
    sand = SAND_MATERIALS.get(int(code))
    return sand.label if sand is not None else f"Unknown sand {code}"


def sand_choices() -> Sequence[Tuple[int, str]]:
    """(code, label) for the settings menus, custom first."""
    return [(SAND_MATERIAL_CUSTOM, sand_label(SAND_MATERIAL_CUSTOM))] + [
        (code, m.label) for code, m in sorted(SAND_MATERIALS.items())
    ]


def tabulated_body(indices: np.ndarray, code: int) -> TabulatedBody:
    """A TabulatedBody of database sand ``code`` on cells ``indices``."""
    temps, k, rho_cp = get_sand(code).tables()
    return TabulatedBody(
        indices=np.asarray(indices, dtype=np.int64),
        temp_points=temps,
        conductivity=k,
        volumetric_heat_capacity=rho_cp,
    )


def mold_tabulated_body(
    indices: np.ndarray, code: int, boundary: Optional[ThermalBoundaryParams] = None
) -> TabulatedBody:
    """Mold sand cells: database sand ``code``, or for code 0 the constant mold sand of ``boundary``."""
    if int(code) != SAND_MATERIAL_CUSTOM and int(code) in SAND_MATERIALS:
        return tabulated_body(indices, code)
    boundary = boundary or ThermalBoundaryParams()
    k = float(boundary.mold_thermal_conductivity)
    rho_cp = k / float(boundary.mold_thermal_diffusivity)
    return TabulatedBody(
        indices=np.asarray(indices, dtype=np.int64),
        temp_points=np.array([0.0, 2000.0]),
        conductivity=np.array([k, k]),
        volumetric_heat_capacity=np.array([rho_cp, rho_cp]),
    )
