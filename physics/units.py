from __future__ import annotations

from typing import TypeVar

import numpy as np

IN_TO_M = 0.0254
M_TO_IN = 1.0 / IN_TO_M
IN_TO_MM = 25.4
MM_TO_IN = 1.0 / IN_TO_MM

GRAVITY_M_S2 = 9.81

_T = TypeVar("_T", float, np.ndarray)

def inches_to_meters(values: _T) -> _T:
    return values * IN_TO_M

def meters_to_inches(values: _T) -> _T:
    return values * M_TO_IN

def inches_to_mm(values: _T) -> _T:
    return values * IN_TO_MM

def mm_to_inches(values: _T) -> _T:
    return values * MM_TO_IN
