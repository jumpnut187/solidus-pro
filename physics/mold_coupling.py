"""A casting model and a sand-mold model on two time scales, coupled through contact pairs.

The mold is mostly sand far from the metal, and sand conducts about fifteen
times more slowly than steel, so it is stable at a far larger time step than
the metal sets. Solving both in one field makes every sand cell pay the
metal's step. Here the casting side (metal, chills, cores) and the mold are
two :class:`~physics.heat_transfer.HeatTransferModel` instances:

- the casting model steps every call, with each coupled cell losing heat to
  the mold cells it touches, held at their temperature for the step
  (``HeatTransferModel.set_external_sink``);
- the heat each pair carried, G_p * (integral of T_i dt - tau * T_j), is
  collected per mold cell, so the exchange is exactly conservative;
- the mold model takes that heat and steps once ``mold_interval`` seconds have
  gathered (or every call, when the casting's step is already that long).

Pairs are of two kinds. Contact pairs (metal and chills against sand) carry
area A and a contact coefficient h, conductance
A / (dx_i / (2 k_i) + 1 / h + dx_j / (2 k_j)) with h = htc[g] times a
temperature factor of the casting-side cell. Stencil pairs (a sand core
against mold sand) carry the 27-point stencil's geometric factor F = w * s * dx
of the link they replace and conduct F * k_ij, k_ij the harmonic pair
conductivity with the core's contact resistance, as the stencil would.

Mesh-agnostic like the rest of ``physics/``: it sees cell indices, areas and
two models, nothing of the voxel mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from physics.heat_transfer import FloatArray, HeatTransferModel, IntArray

#: Mold step (s of simulated time) gathered before the mold model advances.
#: The mold lags the casting by at most this; the first sand cell's response
#: time is about 15 s at 1000 W/(m^2 K) and 3 mm cells.
DEFAULT_MOLD_INTERVAL_S = 1.0


@dataclass
class MoldPairs:
    """Contact and stencil pairs between casting cells (``first``) and mold cells (``second``)."""

    first: IntArray
    second: IntArray
    #: Contact pairs: area in m^2 and htc group; stencil pairs: area 0, group -1.
    area: FloatArray
    group: IntArray
    #: Stencil pairs: geometric factor w * s * dx in m; 0 on contact pairs.
    stencil_factor: FloatArray
    htc: Sequence[float]
    temperature_factors: Sequence[Optional[Tuple[FloatArray, FloatArray]]]
    #: Contact coefficient (W/(m^2 K)) on the casting side of stencil pairs (sand cores).
    stencil_contact_htc: float = 0.0


class CoupledMoldSolver:
    """Steps ``casting`` every call and ``mold`` every ``mold_interval`` seconds.

    Exposes the casting model's interface (``temperatures``, ``time``,
    ``step``, ``volumetric_heat_capacity``, ...) so the solidification loop
    drives it like a single model; anything not defined here is the casting
    model's.
    """

    def __init__(
        self,
        casting: HeatTransferModel,
        mold: HeatTransferModel,
        pairs: MoldPairs,
        mold_interval: float = DEFAULT_MOLD_INTERVAL_S,
    ) -> None:
        self.casting = casting
        self.mold = mold
        self.mold_interval = float(mold_interval)
        self._first = np.asarray(pairs.first, dtype=np.int64)
        self._second = np.asarray(pairs.second, dtype=np.int64)
        self._area = np.asarray(pairs.area, dtype=np.float64)
        self._group = np.asarray(pairs.group, dtype=np.int64)
        self._stencil = np.asarray(pairs.stencil_factor, dtype=np.float64)
        self._is_stencil = self._stencil > 0.0
        self._htc = np.asarray(list(pairs.htc), dtype=np.float64)
        self._factors = list(pairs.temperature_factors)
        self._stencil_contact = float(pairs.stencil_contact_htc)
        self._cells, self._local = np.unique(self._first, return_inverse=True)
        self._mold_cells, self._mold_local = np.unique(self._second, return_inverse=True)
        sizes_c = casting._voxel_sizes
        sizes_m = mold._voxel_sizes
        self._half_dx_first = 0.5 * (sizes_c[self._first] if sizes_c is not None else np.full(len(self._first), casting.voxel_size))
        self._half_dx_second = 0.5 * (sizes_m[self._second] if sizes_m is not None else np.full(len(self._second), mold.voxel_size))
        self._pending = np.zeros(len(self._mold_cells), dtype=np.float64)
        self._clock = 0.0
        self._G = np.zeros(len(self._first), dtype=np.float64)
        self._T_second = np.zeros(len(self._first), dtype=np.float64)
        self._couple()

    # -- the casting model's interface ---------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self.casting, name)

    @property
    def temperatures(self) -> FloatArray:
        return self.casting.temperatures

    @property
    def time(self) -> float:
        return self.casting.time

    @property
    def mold_temperatures(self) -> FloatArray:
        """Mold cell temperatures (°C), as of the mold's last step."""
        return self.mold.temperatures

    def step(self, dt: float) -> None:
        self._couple()
        self.casting.step(dt)
        tau, integral = self.casting.take_external_exchange()
        if tau > 0.0 and len(self._first):
            heat = self._G * (integral[self._local] - tau * self._T_second)
            self._pending += np.bincount(self._mold_local, weights=heat, minlength=len(self._mold_cells))
        self._clock += dt
        if self._clock >= self.mold_interval - 1e-12:
            self.flush()

    def flush(self) -> None:
        """Hand the mold the heat gathered so far and advance it to the casting's time."""
        if self._clock <= 0.0:
            return
        self.mold.add_heat(self._mold_cells, self._pending)
        self._pending[:] = 0.0
        self.mold.step(self._clock)
        self._clock = 0.0

    # -- coupling -------------------------------------------------------------

    def _couple(self) -> None:
        """Pair conductances at the current fields, and each coupled cell's sink."""
        if len(self._first) == 0:
            return
        T_first = self.casting.temperatures[self._first]
        T_second = self.mold.temperatures[self._second]
        k_first = self.casting.conductivity_at(self._first)
        k_second = self.mold.conductivity_at(self._second)
        G = np.zeros(len(self._first), dtype=np.float64)
        contact = ~self._is_stencil
        if np.any(contact):
            h = self._htc[self._group[contact]].copy()
            for g, curve in enumerate(self._factors):
                if curve is None:
                    continue
                sel = self._group[contact] == g
                if np.any(sel):
                    h[sel] *= np.interp(T_first[contact][sel], curve[0], curve[1])
            resistance = (
                self._half_dx_first[contact] / np.maximum(k_first[contact], 1e-12)
                + 1.0 / np.maximum(h, 1e-12)
                + self._half_dx_second[contact] / np.maximum(k_second[contact], 1e-12)
            )
            G[contact] = self._area[contact] / resistance
        if np.any(self._is_stencil):
            ki, kj = k_first[self._is_stencil], k_second[self._is_stencil]
            k_pair = 2.0 * ki * kj / (ki + kj + 1e-30)
            if self._stencil_contact > 0.0:
                g = self._stencil_contact * 2.0 * self._half_dx_first[self._is_stencil]
                k_pair = k_pair / (1.0 + k_pair / g)
            G[self._is_stencil] = self._stencil[self._is_stencil] * k_pair
        self._G = G
        self._T_second = T_second
        G_cell = np.bincount(self._local, weights=G, minlength=len(self._cells))
        GT_cell = np.bincount(self._local, weights=G * T_second, minlength=len(self._cells))
        sink = np.divide(GT_cell, G_cell, out=np.zeros(len(self._cells)), where=G_cell > 0.0)
        self.casting.set_external_sink(self._cells, G_cell, sink)
