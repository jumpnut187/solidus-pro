"""Length-unit conversions between Solidus Pro's three unit zones.

This project does **not** use one length unit throughout, despite what some of
the documentation says. There are three zones, and every boundary crossing has
to convert explicitly:

===========================  ==========  ====================================
Zone                         Length      Where
===========================  ==========  ====================================
Geometry / mesh / UI / web   **inches**  STL input, ``AdaptiveVoxelMesh``
                                         (``voxel_centers``, ``voxel_sizes``,
                                         ``voxel_moduli``), feeding distances,
                                         every user-facing dimension
``physics/heat_transfer``    **metres**  diffusivity in m^2/s, HTC in
                                         W/(m^2 K); ``HeatTransferModel``
                                         takes ``voxel_size`` in metres
Niyama criterion output      **mm**      reported in degC*sqrt(s)/mm; see
                                         ``_least_squares_gradients_c_per_mm``
===========================  ==========  ====================================

Temperatures are degrees Celsius and times are seconds everywhere; only length
is ambiguous.

Material properties in :mod:`core.materials` are SI regardless of zone --
viscosity in Pa*s, density in kg/m^3, gravity 9.81 m/s^2 -- so any expression
combining a mesh length with a material property needs a conversion here.

**Rule for new solvers:** work internally in SI and convert only at the module
boundary. Every standard correlation (CFL condition, Reynolds, Weber, Rayleigh,
gravity head) assumes SI, and mixing an inch-valued length into one of them
produces a plausible wrong answer rather than an error.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np

#: Exact by definition (1 inch == 25.4 mm).
IN_TO_M = 0.0254
M_TO_IN = 1.0 / IN_TO_M
IN_TO_MM = 25.4
MM_TO_IN = 1.0 / IN_TO_MM

#: Standard gravity, m/s^2. Paired with metre-valued lengths, never inches.
GRAVITY_M_S2 = 9.81

_T = TypeVar("_T", float, np.ndarray)


def inches_to_meters(values: _T) -> _T:
    """Convert an inch length (scalar or array) to metres."""
    return values * IN_TO_M


def meters_to_inches(values: _T) -> _T:
    """Convert a metre length (scalar or array) to inches."""
    return values * M_TO_IN


def inches_to_mm(values: _T) -> _T:
    """Convert an inch length (scalar or array) to millimetres."""
    return values * IN_TO_MM


def mm_to_inches(values: _T) -> _T:
    """Convert a millimetre length (scalar or array) to inches."""
    return values * MM_TO_IN
