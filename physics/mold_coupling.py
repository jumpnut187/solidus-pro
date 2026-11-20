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

from physics import kernel_dispatch
from physics.heat_transfer import FloatArray, HeatTransferModel, IntArray

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where numba is absent
    NUMBA_AVAILABLE = False


def _couple_pairs_kernel_py(
    T_casting: FloatArray,
    T_mold: FloatArray,
    first: IntArray,
    second: IntArray,
    local: IntArray,
    mold_local: IntArray,
    k_first_cell: FloatArray,
    k_second_cell: FloatArray,
    is_stencil: Any,
    group: IntArray,
    htc: FloatArray,
    factor_length: IntArray,
    factor_temps: FloatArray,
    factor_values: FloatArray,
    half_dx_first: FloatArray,
    half_dx_second: FloatArray,
    area: FloatArray,
    stencil: FloatArray,
    stencil_contact: float,
    G: FloatArray,
    T_second: FloatArray,
    G_cell: FloatArray,
    sink: FloatArray,
) -> None:  # pragma: no cover - numba
    """``CoupledMoldSolver._couple`` after the two conductivity lookups, in one ordered pass.

    Per pair: the contact conductance A / (dx_i/(2 k_i) + 1/h + dx_j/(2 k_j))
    with h = htc[group] times the group's temperature factor at the casting
    cell, or the stencil-pair conductance. Per coupled casting cell: the sum
    of its pair conductances and the conductance-weighted mold temperature it
    sinks to. The pairs are visited in order, which is the order
    ``np.bincount`` adds them in. ``k_first_cell`` / ``k_second_cell`` are per
    coupled cell (``local`` / ``mold_local`` map a pair to them).
    """
    n_cells = len(G_cell)
    weighted = np.zeros(n_cells, dtype=np.float64)
    for c in range(n_cells):
        G_cell[c] = 0.0
    for p in range(len(first)):
        k_f = k_first_cell[local[p]]
        k_s = k_second_cell[mold_local[p]]
        t_second = T_mold[second[p]]
        if is_stencil[p]:
            k_pair = 2.0 * k_f * k_s / (k_f + k_s + 1e-30)
            if stencil_contact > 0.0:
                g = stencil_contact * 2.0 * half_dx_first[p]
                k_pair = k_pair / (1.0 + k_pair / g)
            conductance = stencil[p] * k_pair
        else:
            gr = group[p]
            h = htc[gr]
            m = factor_length[gr]
            if m > 0:
                # np.interp of the group's factor curve at the casting cell's temperature.
                t = T_casting[first[p]]
                if t != t:
                    factor = t
                elif t <= factor_temps[gr, 0]:
                    factor = factor_values[gr, 0]
                elif t >= factor_temps[gr, m - 1]:
                    factor = factor_values[gr, m - 1]
                else:
                    j = 0
                    for seg in range(m - 1):
                        if t < factor_temps[gr, seg + 1]:
                            j = seg
                            break
                    x0 = factor_temps[gr, j]
                    slope = (factor_values[gr, j + 1] - factor_values[gr, j]) / (factor_temps[gr, j + 1] - x0)
                    factor = slope * (t - x0) + factor_values[gr, j]
                h = h * factor
            # np.maximum(x, 1e-12): the floor for a small x, x itself otherwise (NaN included).
            if k_f < 1e-12:
                k_f = 1e-12
            if k_s < 1e-12:
                k_s = 1e-12
            if h < 1e-12:
                h = 1e-12
            resistance = half_dx_first[p] / k_f + 1.0 / h + half_dx_second[p] / k_s
            conductance = area[p] / resistance
        G[p] = conductance
        T_second[p] = t_second
        G_cell[local[p]] += conductance
        weighted[local[p]] += conductance * t_second
    for c in range(n_cells):
        if G_cell[c] > 0.0:
            sink[c] = weighted[c] / G_cell[c]
        else:
            sink[c] = 0.0


if NUMBA_AVAILABLE:
    # error_model="numpy": a zero divisor gives inf/NaN, as in the NumPy form.
    couple_pairs_kernel = njit(cache=True, error_model="numpy")(_couple_pairs_kernel_py)

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
        # The pair sets never change, so every selection ``_couple`` needs is
        # worked out here once: the contact pairs, the stencil pairs, and the
        # contact pairs of each group that carries a temperature factor.
        self._contact_pairs = np.flatnonzero(~self._is_stencil)
        self._stencil_pairs = np.flatnonzero(self._is_stencil)
        self._all_contact = len(self._stencil_pairs) == 0
        contact_group = self._group[self._contact_pairs]
        self._contact_htc = self._htc[contact_group]
        self._contact_half_dx_first = self._half_dx_first[self._contact_pairs]
        self._contact_half_dx_second = self._half_dx_second[self._contact_pairs]
        self._contact_area = self._area[self._contact_pairs]
        # (curve, positions among the contact pairs, casting cell of each).
        self._contact_factors = []
        for g, curve in enumerate(self._factors):
            if curve is None:
                continue
            positions = np.flatnonzero(contact_group == g)
            if positions.size:
                self._contact_factors.append((curve, positions, self._first[self._contact_pairs][positions]))
        # The factor curves as one padded table per group, for the fused kernel.
        longest = max([len(curve[0]) for curve in self._factors if curve is not None] + [1])
        self._factor_length = np.zeros(max(len(self._factors), 1), dtype=np.int64)
        self._factor_temps = np.zeros((max(len(self._factors), 1), longest), dtype=np.float64)
        self._factor_values = np.zeros((max(len(self._factors), 1), longest), dtype=np.float64)
        for g, curve in enumerate(self._factors):
            if curve is None:
                continue
            temps = np.asarray(curve[0], dtype=np.float64).reshape(-1)
            self._factor_length[g] = len(temps)
            self._factor_temps[g, : len(temps)] = temps
            self._factor_values[g, : len(temps)] = np.asarray(curve[1], dtype=np.float64).reshape(-1)
        # A curve the kernel's forward scan cannot read (not strictly increasing,
        # or more groups than curves) keeps the NumPy form.
        self._factors_fit_kernel = len(self._factors) == len(self._htc) and all(
            curve is None or (len(curve[0]) >= 2 and bool(np.all(np.diff(np.asarray(curve[0], dtype=np.float64)) > 0.0)))
            for curve in self._factors
        )
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
        T_casting = self.casting.temperatures
        if NUMBA_AVAILABLE and self._factors_fit_kernel and kernel_dispatch.fused_kernels_enabled():
            G = np.empty(len(self._first), dtype=np.float64)
            T_second = np.empty(len(self._first), dtype=np.float64)
            G_cell = np.empty(len(self._cells), dtype=np.float64)
            sink = np.empty(len(self._cells), dtype=np.float64)
            couple_pairs_kernel(
                T_casting,
                self.mold.temperatures,
                self._first,
                self._second,
                self._local,
                self._mold_local,
                self.casting.conductivity_at(self._cells),
                self.mold.conductivity_at(self._mold_cells),
                self._is_stencil,
                self._group,
                self._htc,
                self._factor_length,
                self._factor_temps,
                self._factor_values,
                self._half_dx_first,
                self._half_dx_second,
                self._area,
                self._stencil,
                self._stencil_contact,
                G,
                T_second,
                G_cell,
                sink,
            )
            self._G = G
            self._T_second = T_second
            self.casting.set_external_sink(self._cells, G_cell, sink)
            return
        T_second = self.mold.temperatures[self._second]
        # One evaluation per coupled cell, spread to its pairs (a cell usually
        # has several): conductivity is a per-cell value.
        k_first = self.casting.conductivity_at(self._cells)[self._local]
        k_second = self.mold.conductivity_at(self._mold_cells)[self._mold_local]
        G = np.zeros(len(self._first), dtype=np.float64)
        contact = self._contact_pairs
        if contact.size:
            h = self._contact_htc.copy()
            for curve, positions, casting_cells in self._contact_factors:
                h[positions] *= np.interp(T_casting[casting_cells], curve[0], curve[1])
            k_first_contact = k_first if self._all_contact else k_first[contact]
            k_second_contact = k_second if self._all_contact else k_second[contact]
            resistance = (
                self._contact_half_dx_first / np.maximum(k_first_contact, 1e-12)
                + 1.0 / np.maximum(h, 1e-12)
                + self._contact_half_dx_second / np.maximum(k_second_contact, 1e-12)
            )
            G[contact] = self._contact_area / resistance
        stencil = self._stencil_pairs
        if stencil.size:
            ki, kj = k_first[stencil], k_second[stencil]
            k_pair = 2.0 * ki * kj / (ki + kj + 1e-30)
            if self._stencil_contact > 0.0:
                g = self._stencil_contact * 2.0 * self._half_dx_first[stencil]
                k_pair = k_pair / (1.0 + k_pair / g)
            G[stencil] = self._stencil[stencil] * k_pair
        self._G = G
        self._T_second = T_second
        G_cell = np.bincount(self._local, weights=G, minlength=len(self._cells))
        GT_cell = np.bincount(self._local, weights=G * T_second, minlength=len(self._cells))
        sink = np.divide(GT_cell, G_cell, out=np.zeros(len(self._cells)), where=G_cell > 0.0)
        self.casting.set_external_sink(self._cells, G_cell, sink)
