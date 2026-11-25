from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[Any]
IntArray = NDArray[Any]
BoolArray = NDArray[Any]

class BoundaryType(Enum):
    FIXED_TEMPERATURE = auto()
    HEAT_FLUX = auto()
    CONVECTION = auto()
    ADIABATIC = auto()
    INTERFACE = auto()
    RADIATION = auto()

STEFAN_BOLTZMANN = 5.670374419e-8
_KELVIN = 273.15

def radiation_htc(T: FloatArray, T_inf: float, emissivity: float) -> FloatArray:
    tk = np.asarray(T, dtype=np.float64) + _KELVIN
    ti = float(T_inf) + _KELVIN
    return float(emissivity) * STEFAN_BOLTZMANN * (tk * tk + ti * ti) * (tk + ti)

@dataclass
class BoundaryCondition:
    type: BoundaryType
    indices: IntArray
    value: float = 0.0
    htc: float = 0.0
    T_infinity: float = 25.0
    time_function: Optional[Callable[[float], float]] = None
    weights: Optional[FloatArray] = None

    def surface_htc(self, temperatures: FloatArray, time: float) -> Union[float, FloatArray]:
        h: Union[float, FloatArray] = self.htc
        if self.type == BoundaryType.INTERFACE and self.time_function is not None:
            h = self.htc * self.time_function(time)
        elif self.type == BoundaryType.RADIATION:
            h = self.htc + radiation_htc(temperatures[self.indices], self.T_infinity, self.value)
        if self.weights is not None:
            h = h * self.weights
        return h

    def get_effective_value(self, time: float) -> float:
        if self.time_function is not None:
            return self.value * self.time_function(time)
        return self.value

@dataclass
class TemperatureField:
    values: FloatArray
    time: float = 0.0

    @property
    def min(self) -> float:
        return float(np.min(self.values))

    @property
    def max(self) -> float:
        return float(np.max(self.values))

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def std(self) -> float:
        return float(np.std(self.values))

    def get_gradient_magnitude(self, neighbors: IntArray, voxel_size: float) -> FloatArray:
        if NUMBA_AVAILABLE:
            return compute_gradient_magnitude_parallel(
                np.asarray(self.values, dtype=np.float64),
                np.asarray(neighbors, dtype=np.int32),
                float(voxel_size),
            )

        n_voxels = len(self.values)
        gradients = np.zeros(n_voxels, dtype=np.float32)
        for i in range(n_voxels):
            grad_sq = 0.0
            count = 0
            for j in range(neighbors.shape[1]):
                neighbor_idx = neighbors[i, j]
                if neighbor_idx >= 0:
                    dT = self.values[neighbor_idx] - self.values[i]
                    grad_sq += (dT / voxel_size) ** 2
                    count += 1
            if count > 0:
                gradients[i] = np.sqrt(grad_sq / count)
        return gradients

@dataclass
class TabulatedBody:
    indices: IntArray
    temp_points: FloatArray
    conductivity: FloatArray
    volumetric_heat_capacity: FloatArray

@dataclass
class InterfaceLinks:
    first: IntArray
    second: IntArray
    area: FloatArray
    group: IntArray
    htc: Sequence[float]
    time_functions: Sequence[Optional[Callable[[float], float]]]
    temperature_factors: Optional[Sequence[Optional[tuple[FloatArray, FloatArray]]]] = None

    def group_htc(self, time: float) -> FloatArray:
        return np.array(
            [
                float(h) * (float(fn(time)) if fn is not None else 1.0)
                for h, fn in zip(self.htc, self.time_functions)
            ],
            dtype=np.float64,
        )

TABULATED_ENTHALPY_RANGE = (-100.0, 2500.0)
TABULATED_ENTHALPY_STEP = 0.5
TABULATED_PROPERTY_TOLERANCE = 0.01
TABULATED_ENTHALPY_TOLERANCE = 1e-3

class MaterialThermalProperties(Protocol):
    thermal_conductivity: float
    specific_heat: float
    density_liquid: float
    density_solid: float
    liquidus_temp: float
    solidus_temp: float
    latent_heat: float

    @property
    def thermal_diffusivity(self) -> float: ...

class HeatTransferModel:

    def __init__(
        self,
        n_voxels: int,
        voxel_size: float,
        neighbors: IntArray,
        alloy: MaterialThermalProperties,
        use_temperature_dependent: bool = False,
        voxel_sizes: Optional[np.ndarray] = None,
        diffusivity_overrides: Optional[tuple[np.ndarray, np.ndarray]] = None,
        property_curves: Optional[Dict[str, np.ndarray]] = None,
        rho_cp_overrides: Optional[tuple[np.ndarray, np.ndarray]] = None,
        capacity_weights: Optional[np.ndarray] = None,
        boundary_areas: Optional[np.ndarray] = None,
        slave_cells: Optional[tuple[np.ndarray, np.ndarray]] = None,
        cut_cell_apertures: bool = True,
        contact_conductance: Optional[tuple[np.ndarray, float]] = None,
        tabulated_bodies: Optional[Sequence[TabulatedBody]] = None,
        interface_links: Optional[InterfaceLinks] = None,
    ):
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.neighbors = neighbors
        self.alloy = alloy
        self.use_temperature_dependent = use_temperature_dependent
        self._voxel_sizes = (
            np.asarray(voxel_sizes, dtype=np.float64) if voxel_sizes is not None else None
        )
        self._diffusivity_overrides = diffusivity_overrides
        self._property_curves = property_curves
        if property_curves is not None:
            self._curve_temp = np.ascontiguousarray(property_curves["temp_points"], dtype=np.float64)
            self._curve_k = np.ascontiguousarray(property_curves["conductivity"], dtype=np.float64)
            self._curve_rho = np.ascontiguousarray(property_curves["density"], dtype=np.float64)
            self._curve_cp = np.ascontiguousarray(property_curves["specific_heat"], dtype=np.float64)
        else:
            self._curve_temp = None
            self._curve_k = None
            self._curve_rho = None
            self._curve_cp = None
        self._rho_cp_scalar = alloy.density_liquid * alloy.specific_heat
        self._rho_cp_overrides = rho_cp_overrides

        self._temperatures = np.zeros(n_voxels, dtype=np.float64)
        self._temperatures_new = np.zeros(n_voxels, dtype=np.float64)

        self.time = 0.0

        self._boundaries: List[BoundaryCondition] = []

        self._heat_source = np.zeros(n_voxels, dtype=np.float64)

        self._alpha = alloy.thermal_diffusivity
        self._dx2 = voxel_size ** 2
        if self._voxel_sizes is not None:
            self._inv_dx2 = 1.0 / (self._voxel_sizes ** 2)
        else:
            self._inv_dx2 = 1.0 / (voxel_size ** 2)
        self._inv_dx2_arr = (
            self._inv_dx2
            if isinstance(self._inv_dx2, np.ndarray)
            else np.full(n_voxels, self._inv_dx2, dtype=np.float64)
        )
        self._numba_parallel_eligible = bool(NUMBA_AVAILABLE and not use_temperature_dependent)
        self._last_step_used_numba = False

        self._capacity_weights: Optional[FloatArray] = None
        self._conductance_factors: Optional[FloatArray] = None
        self._link_apertures: Optional[FloatArray] = None
        self._boundary_area_per_volume: Optional[FloatArray] = None
        self._slave_cells: Optional[tuple[IntArray, IntArray]] = None
        self._capacity_curve: Optional[FloatArray] = None
        self._step_rho_cp: Optional[FloatArray] = None
        if capacity_weights is None and boundary_areas is not None:
            capacity_weights = np.ones(n_voxels, dtype=np.float64)
        if capacity_weights is not None:
            weights = np.ascontiguousarray(capacity_weights, dtype=np.float64).reshape(-1)
            if weights.shape != (n_voxels,) or np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
                raise ValueError("capacity_weights must be (n_voxels,) finite and >= 0")
            self._capacity_weights = weights
            self._link_apertures = (
                cut_cell_link_apertures(weights) if cut_cell_apertures else np.ones(n_voxels, dtype=np.float64)
            )
            self._conductance_factors = cut_cell_conductance_factors(weights, neighbors, self._link_apertures)
            if property_curves is not None and use_temperature_dependent:
                self._capacity_curve = np.ascontiguousarray(self._curve_rho * self._curve_cp)
        if boundary_areas is not None:
            areas = np.asarray(boundary_areas, dtype=np.float64).reshape(-1)
            if areas.shape != (n_voxels,):
                raise ValueError("boundary_areas must be (n_voxels,)")
            sizes = (
                self._voxel_sizes
                if self._voxel_sizes is not None
                else np.full(n_voxels, voxel_size, dtype=np.float64)
            )
            weights = self._capacity_weights if self._capacity_weights is not None else np.ones(n_voxels)
            cell_volume = weights * sizes ** 3
            self._boundary_area_per_volume = np.divide(
                areas, cell_volume, out=np.zeros(n_voxels, dtype=np.float64), where=cell_volume > 0.0
            )
        if slave_cells is not None:
            slaves, masters = slave_cells
            self._slave_cells = (
                np.asarray(slaves, dtype=np.int64).reshape(-1),
                np.asarray(masters, dtype=np.int64).reshape(-1),
            )

        self._contact_hdx = np.zeros(n_voxels, dtype=np.float64)
        self._has_contact = False
        if contact_conductance is not None:
            contact_idx, contact_h = contact_conductance
            contact_idx = np.asarray(contact_idx, dtype=np.int64).reshape(-1)
            if contact_idx.size > 0 and float(contact_h) > 0.0:
                sizes = (
                    self._voxel_sizes[contact_idx]
                    if self._voxel_sizes is not None
                    else np.full(contact_idx.size, voxel_size, dtype=np.float64)
                )
                self._contact_hdx[contact_idx] = float(contact_h) * sizes
                self._has_contact = True

        self._tabulated: List[Dict[str, Any]] = []
        if tabulated_bodies:
            self._init_tabulated_bodies(tabulated_bodies)
        self._interface_links = interface_links
        if interface_links is not None:
            self._init_interface_links(interface_links)
        self._sink_cells: Optional[IntArray] = None
        self._sink_G: Optional[FloatArray] = None
        self._sink_T: Optional[FloatArray] = None
        self._sink_volume: Optional[FloatArray] = None
        self._sink_time = 0.0
        self._sink_integral: Optional[FloatArray] = None

        self._stable_rho_cp: Optional[FloatArray] = None
        self._stable_link_g: Optional[FloatArray] = None

        self._neighbor_counts = np.sum(neighbors >= 0, axis=1).astype(np.float64)

    def _init_tabulated_bodies(self, bodies: Sequence[TabulatedBody]) -> None:
        lo, hi = TABULATED_ENTHALPY_RANGE
        fine_t = np.arange(lo, hi + TABULATED_ENTHALPY_STEP, TABULATED_ENTHALPY_STEP)
        diff_idx = [] if self._diffusivity_overrides is None else [np.asarray(self._diffusivity_overrides[0], dtype=np.int64)]
        diff_val = [] if self._diffusivity_overrides is None else [np.asarray(self._diffusivity_overrides[1], dtype=np.float64)]
        rcp_idx = [] if self._rho_cp_overrides is None else [np.asarray(self._rho_cp_overrides[0], dtype=np.int64)]
        rcp_val = [] if self._rho_cp_overrides is None else [np.asarray(self._rho_cp_overrides[1], dtype=np.float64)]
        diff_offset = int(sum(len(a) for a in diff_idx))
        rcp_offset = int(sum(len(a) for a in rcp_idx))
        for body in bodies:
            idx = np.asarray(body.indices, dtype=np.int64).reshape(-1)
            if idx.size == 0:
                continue
            temps = np.ascontiguousarray(body.temp_points, dtype=np.float64)
            k_pts = np.ascontiguousarray(body.conductivity, dtype=np.float64)
            c_pts = np.ascontiguousarray(body.volumetric_heat_capacity, dtype=np.float64)
            if temps.ndim != 1 or len(temps) < 2 or np.any(np.diff(temps) <= 0.0):
                raise ValueError("TabulatedBody.temp_points must be strictly increasing, at least 2 points")
            if k_pts.shape != temps.shape or c_pts.shape != temps.shape:
                raise ValueError("TabulatedBody tables must match temp_points")
            if np.any(k_pts <= 0.0) or np.any(c_pts <= 0.0):
                raise ValueError("TabulatedBody conductivity and heat capacity must be positive")
            fine_c = np.interp(fine_t, temps, c_pts)
            fine_h = np.concatenate(([0.0], np.cumsum(0.5 * (fine_c[1:] + fine_c[:-1]) * np.diff(fine_t))))
            n = idx.size
            diff_idx.append(idx)
            diff_val.append(np.zeros(n, dtype=np.float64))
            rcp_idx.append(idx)
            rcp_val.append(np.zeros(n, dtype=np.float64))
            self._tabulated.append({
                "indices": idx,
                "temps": temps,
                "k": k_pts,
                "c": c_pts,
                "fine_t": fine_t,
                "fine_h": fine_h,
                "diff_slice": slice(diff_offset, diff_offset + n),
                "rcp_slice": slice(rcp_offset, rcp_offset + n),
                "last_T": np.full(n, np.nan),
                "sorted_indices": np.sort(idx),
            })
            diff_offset += n
            rcp_offset += n
        if not self._tabulated:
            return
        self._diffusivity_overrides = (np.concatenate(diff_idx), np.concatenate(diff_val))
        self._rho_cp_overrides = (np.concatenate(rcp_idx), np.concatenate(rcp_val))
        self._update_tabulated_bodies(self._temperatures)

    def _update_tabulated_bodies(self, T: FloatArray) -> None:
        if not self._tabulated or self._diffusivity_overrides is None or self._rho_cp_overrides is None:
            return
        diff_vals = self._diffusivity_overrides[1]
        rcp_vals = self._rho_cp_overrides[1]
        for body in self._tabulated:
            if NUMBA_AVAILABLE:
                update_kernel = (
                    update_tabulated_serial
                    if runs_serial(len(body["indices"]), _COST_TABULATED_UPDATE_NS)
                    else update_tabulated_parallel
                )
                update_kernel(
                    T, body["indices"], body["last_T"], body["temps"], body["k"], body["c"],
                    diff_vals, body["diff_slice"].start, rcp_vals, body["rcp_slice"].start,
                    TABULATED_PROPERTY_TOLERANCE,
                )
                continue
            t = T[body["indices"]]
            moved = np.flatnonzero(~(np.abs(t - body["last_T"]) < TABULATED_PROPERTY_TOLERANCE))
            if moved.size == 0:
                continue
            t_moved = t[moved]
            k = _interp(t_moved, body["temps"], body["k"])
            c = _interp(t_moved, body["temps"], body["c"])
            rcp_vals[body["rcp_slice"].start + moved] = c
            diff_vals[body["diff_slice"].start + moved] = k / c
            body["last_T"][moved] = t_moved

    def _correct_tabulated_enthalpy(self, T_new: FloatArray, T_old: FloatArray) -> None:
        if not self._tabulated or self._rho_cp_overrides is None:
            return
        rcp_vals = self._rho_cp_overrides[1]
        for body in self._tabulated:
            idx = body["indices"]
            if NUMBA_AVAILABLE:
                correct_kernel = (
                    correct_tabulated_serial
                    if runs_serial(len(idx), _COST_TABULATED_CORRECT_NS)
                    else correct_tabulated_parallel
                )
                correct_kernel(
                    T_new, T_old, idx, rcp_vals, body["rcp_slice"].start,
                    body["fine_t"], body["fine_h"], TABULATED_ENTHALPY_TOLERANCE,
                )
                continue
            t_old = T_old[idx]
            step = T_new[idx] - t_old
            moved = np.flatnonzero(np.abs(step) >= TABULATED_ENTHALPY_TOLERANCE)
            if moved.size == 0:
                continue
            cells = idx[moved]
            t_moved = t_old[moved]
            h_new = (
                _interp(t_moved, body["fine_t"], body["fine_h"])
                + step[moved] * rcp_vals[body["rcp_slice"].start + moved]
            )
            T_new[cells] = _interp(h_new, body["fine_h"], body["fine_t"])

    def tabulated_enthalpy(self, body_index: int, temperatures: FloatArray) -> FloatArray:
        body = self._tabulated[body_index]
        return np.interp(np.asarray(temperatures, dtype=np.float64), body["fine_t"], body["fine_h"])

    def _init_interface_links(self, links: InterfaceLinks) -> None:
        first = np.asarray(links.first, dtype=np.int64).reshape(-1)
        second = np.asarray(links.second, dtype=np.int64).reshape(-1)
        area = np.asarray(links.area, dtype=np.float64).reshape(-1)
        group = np.asarray(links.group, dtype=np.int64).reshape(-1)
        if not (first.shape == second.shape == area.shape == group.shape):
            raise ValueError("InterfaceLinks arrays must have one entry per pair")
        if len(links.htc) != len(links.time_functions):
            raise ValueError("InterfaceLinks needs one time function (or None) per htc group")
        if links.temperature_factors is not None and len(links.temperature_factors) != len(links.htc):
            raise ValueError("InterfaceLinks needs one temperature factor curve (or None) per htc group")
        if group.size and (group.min() < 0 or group.max() >= len(links.htc)):
            raise ValueError("InterfaceLinks.group out of range")
        sizes = (
            self._voxel_sizes
            if self._voxel_sizes is not None
            else np.full(self.n_voxels, self.voxel_size, dtype=np.float64)
        )
        self._link_first = first
        self._link_second = second
        self._link_area = area
        self._link_group = group
        self._link_group_pairs = [np.flatnonzero(group == g) for g in range(len(links.htc))]
        self._link_half_dx_first = 0.5 * sizes[first]
        self._link_half_dx_second = 0.5 * sizes[second]
        cells = np.unique(np.concatenate([first, second]))
        self._link_cells = cells
        self._link_first_local = np.searchsorted(cells, first)
        self._link_second_local = np.searchsorted(cells, second)
        volume = sizes[cells] ** 3
        if self._capacity_weights is not None:
            volume = volume * self._capacity_weights[cells]
        self._link_cell_volume = volume

    def _link_cell_capacity(self, rho_cp: FloatArray) -> FloatArray:
        return rho_cp[self._link_cells] * self._link_cell_volume

    def _interface_link_conductance(
        self, alpha: FloatArray, rho_cp: FloatArray, T: Optional[FloatArray] = None
    ) -> FloatArray:
        links = self._interface_links
        assert links is not None
        h = links.group_htc(self.time)[self._link_group]
        if links.temperature_factors is not None:
            T_first = (self._temperatures if T is None else T)[self._link_first]
            for g, curve in enumerate(links.temperature_factors):
                if curve is None:
                    continue
                sel = self._link_group_pairs[g]
                h[sel] = h[sel] * np.interp(T_first[sel], curve[0], curve[1])
        if NUMBA_AVAILABLE:
            return interface_link_conductance_kernel(
                h,
                np.ascontiguousarray(alpha, dtype=np.float64),
                np.ascontiguousarray(rho_cp, dtype=np.float64),
                self._link_first,
                self._link_second,
                self._link_half_dx_first,
                self._link_half_dx_second,
                self._link_area,
            )
        k_first = alpha[self._link_first] * rho_cp[self._link_first]
        k_second = alpha[self._link_second] * rho_cp[self._link_second]
        resistance = (
            self._link_half_dx_first / np.maximum(k_first, 1e-12)
            + 1.0 / np.maximum(h, 1e-12)
            + self._link_half_dx_second / np.maximum(k_second, 1e-12)
        )
        return np.where(h > 0.0, self._link_area / resistance, 0.0)

    def _interface_link_rates(self, alpha: FloatArray, rho_cp: FloatArray) -> FloatArray:
        if self._interface_links is None or self._link_first.size == 0:
            return np.zeros(self.n_voxels, dtype=np.float64)
        m = len(self._link_cells)
        g = self._interface_link_conductance(alpha, rho_cp)
        self._stable_link_g = g
        if NUMBA_AVAILABLE:
            return interface_link_rates_kernel(
                g,
                self._link_first_local,
                self._link_second_local,
                self._link_cells,
                self._link_cell_volume,
                np.ascontiguousarray(rho_cp, dtype=np.float64),
                self.n_voxels,
            )
        rates = np.zeros(self.n_voxels, dtype=np.float64)
        conductance = np.bincount(self._link_first_local, weights=g, minlength=m)
        conductance += np.bincount(self._link_second_local, weights=g, minlength=m)
        capacity = self._link_cell_capacity(rho_cp)
        local = np.zeros(m, dtype=np.float64)
        np.divide(conductance, capacity, out=local, where=capacity > 0.0)
        rates[self._link_cells] = local
        return rates

    def _apply_interface_links(
        self,
        T_new: FloatArray,
        T_old: FloatArray,
        dt: float,
        alpha: FloatArray,
        rho_cp: FloatArray,
        conductance: Optional[FloatArray] = None,
    ) -> None:
        if self._interface_links is None or self._link_first.size == 0:
            return
        m = len(self._link_cells)
        g = conductance if conductance is not None else self._interface_link_conductance(alpha, rho_cp, T_old)
        if NUMBA_AVAILABLE:
            apply_interface_links_kernel(
                T_new,
                T_old,
                float(dt),
                g,
                self._link_first,
                self._link_second,
                self._link_first_local,
                self._link_second_local,
                self._link_cells,
                self._link_cell_volume,
                np.ascontiguousarray(rho_cp, dtype=np.float64),
            )
            return
        q = g * (T_old[self._link_first] - T_old[self._link_second])
        heat = np.bincount(self._link_second_local, weights=q, minlength=m)
        heat -= np.bincount(self._link_first_local, weights=q, minlength=m)
        capacity = self._link_cell_capacity(rho_cp)
        delta = np.zeros(m, dtype=np.float64)
        np.divide(dt * heat, capacity, out=delta, where=capacity > 0.0)
        T_new[self._link_cells] += delta

    def _cell_volumes(self, cells: IntArray) -> FloatArray:
        sizes = self._voxel_sizes[cells] if self._voxel_sizes is not None else np.full(len(cells), self.voxel_size)
        volume = sizes ** 3
        if self._capacity_weights is not None:
            volume = volume * self._capacity_weights[cells]
        return volume

    def set_external_sink(self, cells: IntArray, conductance: FloatArray, sink_temperature: FloatArray) -> None:
        cells = np.asarray(cells, dtype=np.int64).reshape(-1)
        if self._sink_cells is None or len(self._sink_cells) != len(cells) or np.any(self._sink_cells != cells):
            self._sink_cells = cells
            self._sink_volume = self._cell_volumes(cells)
            self._sink_integral = np.zeros(len(cells), dtype=np.float64)
            self._sink_time = 0.0
        self._sink_G = np.asarray(conductance, dtype=np.float64).reshape(-1)
        self._sink_T = np.asarray(sink_temperature, dtype=np.float64).reshape(-1)

    def take_external_exchange(self) -> tuple[float, FloatArray]:
        if self._sink_integral is None:
            return 0.0, np.zeros(0)
        out = (self._sink_time, self._sink_integral.copy())
        self._sink_integral[:] = 0.0
        self._sink_time = 0.0
        return out

    def _apply_external_sink(self, T_new: FloatArray, T_old: FloatArray, dt: float, rho_cp: FloatArray) -> None:
        if self._sink_cells is None or self._sink_G is None or self._sink_T is None:
            return
        cells = self._sink_cells
        t_old = T_old[cells]
        capacity = rho_cp[cells] * self._sink_volume
        T_new[cells] -= dt * self._sink_G * (t_old - self._sink_T) / np.maximum(capacity, 1e-30)
        self._sink_integral += dt * t_old
        self._sink_time += dt

    def _external_sink_rates(self, rho_cp: FloatArray) -> FloatArray:
        rates = np.zeros(self.n_voxels, dtype=np.float64)
        if self._sink_cells is None or self._sink_G is None:
            return rates
        capacity = rho_cp[self._sink_cells] * self._sink_volume
        np.add.at(rates, self._sink_cells, self._sink_G / np.maximum(capacity, 1e-30))
        return rates

    def add_heat(self, cells: IntArray, joules: FloatArray) -> None:
        cells = np.asarray(cells, dtype=np.int64).reshape(-1)
        joules = np.asarray(joules, dtype=np.float64).reshape(-1)
        if cells.size == 0:
            return
        dh = joules / np.maximum(self._cell_set_volumes(cells), 1e-30)
        T = self._temperatures
        done = np.zeros(len(cells), dtype=bool)
        for body, inside in zip(self._tabulated, self._tabulated_membership(cells)):
            if inside.size == 0:
                continue
            sel = cells[inside]
            if NUMBA_AVAILABLE and len(sel) > 4096:
                shift_kernel = (
                    shift_enthalpy_serial
                    if runs_serial(len(sel), _COST_SHIFT_ENTHALPY_NS)
                    else shift_enthalpy_parallel
                )
                T[sel] = shift_kernel(
                    np.ascontiguousarray(T[sel], dtype=np.float64),
                    np.ascontiguousarray(dh[inside], dtype=np.float64),
                    body["fine_t"],
                    body["fine_h"],
                )
            else:
                h = _interp(T[sel], body["fine_t"], body["fine_h"]) + dh[inside]
                T[sel] = _interp(h, body["fine_h"], body["fine_t"])
            done[inside] = True
        rest = ~done
        if np.any(rest):
            T[cells[rest]] += dh[rest] / self.volumetric_heat_capacity()[cells[rest]]

    def conductivity_at(self, cells: IntArray) -> FloatArray:
        cells = np.asarray(cells, dtype=np.int64)
        T = self._temperatures
        k = np.full(len(cells), np.nan)
        for body, inside in zip(self._tabulated, self._tabulated_membership(cells)):
            if inside.size:
                k[inside] = np.interp(T[cells[inside]], body["temps"], body["k"])
        rest = np.isnan(k)
        if np.any(rest):
            rest_cells = cells[rest]
            k[rest] = self._alpha_at(rest_cells, T) * self._rho_cp_at(rest_cells, T)
        return k

    def _tabulated_membership(self, cells: IntArray) -> List[IntArray]:
        if not self._tabulated:
            return []
        cache = self.__dict__.setdefault("_membership_cache", [])
        for known, membership in cache:
            if known is cells or (known.shape == cells.shape and np.array_equal(known, cells)):
                return membership
        membership = []
        for body in self._tabulated:
            sorted_indices = body["sorted_indices"]
            pos = np.minimum(np.searchsorted(sorted_indices, cells), len(sorted_indices) - 1)
            membership.append(np.flatnonzero(sorted_indices[pos] == cells))
        if len(cache) >= 8:
            cache.pop(0)
        cache.append((cells.copy(), membership))
        return membership

    def _cell_set_volumes(self, cells: IntArray) -> FloatArray:
        cache = self.__dict__.setdefault("_volume_cache", [])
        for known, volumes in cache:
            if known is cells or (known.shape == cells.shape and np.array_equal(known, cells)):
                return volumes
        volumes = self._cell_volumes(cells)
        if len(cache) >= 8:
            cache.pop(0)
        cache.append((cells.copy(), volumes))
        return volumes

    def _override_positions(self, overrides: Optional[tuple[np.ndarray, np.ndarray]], name: str) -> Optional[IntArray]:
        if overrides is None:
            return None
        cached = self.__dict__.get(name)
        if cached is not None and cached[0] is overrides[0]:
            return cached[1]
        positions = np.full(self.n_voxels, -1, dtype=np.int64)
        indices = overrides[0]
        positions[indices] = np.arange(len(indices), dtype=np.int64)
        self.__dict__[name] = (indices, positions)
        return positions

    def _alpha_at(self, cells: IntArray, T: FloatArray) -> FloatArray:
        if self.use_temperature_dependent:
            alpha = np.array(self._get_temperature_dependent_diffusivity(T[cells]), dtype=np.float64, copy=True)
        else:
            alpha = np.full(len(cells), float(self._alpha), dtype=np.float64)
        positions = self._override_positions(self._diffusivity_overrides, "_diffusivity_override_positions")
        if positions is not None:
            at = positions[cells]
            overridden = at >= 0
            if np.any(overridden):
                alpha[overridden] = self._diffusivity_overrides[1][at[overridden]]
        return alpha

    def _rho_cp_at(self, cells: IntArray, T: FloatArray) -> FloatArray:
        if self._capacity_curve is not None:
            t_cells = np.ascontiguousarray(T[cells], dtype=np.float64)
            if NUMBA_AVAILABLE:
                curve_kernel = (
                    interpolate_curve_serial if runs_serial(len(t_cells), _COST_CURVE_NS) else interpolate_curve_parallel
                )
                rho_cp = curve_kernel(t_cells, self._curve_temp, self._capacity_curve)
            else:
                rho_cp = np.interp(t_cells, self._curve_temp, self._capacity_curve)
        else:
            rho_cp = np.full(len(cells), self._rho_cp_scalar, dtype=np.float64)
        positions = self._override_positions(self._rho_cp_overrides, "_rho_cp_override_positions")
        if positions is not None:
            at = positions[cells]
            overridden = at >= 0
            if np.any(overridden):
                rho_cp[overridden] = self._rho_cp_overrides[1][at[overridden]]
        return rho_cp

    def _step_alpha_array(self, T: FloatArray) -> FloatArray:
        if self.use_temperature_dependent:
            alpha = np.array(self._get_temperature_dependent_diffusivity(T), dtype=np.float64, copy=True)
        else:
            alpha = np.full(self.n_voxels, float(self._alpha), dtype=np.float64)
        if self._diffusivity_overrides is not None:
            override_idx, override_vals = self._diffusivity_overrides
            alpha[override_idx] = override_vals
        return alpha

    @property
    def numba_parallel_eligible(self) -> bool:
        return self._numba_parallel_eligible

    @property
    def backend_description(self) -> str:
        return "Numba parallel kernels" if NUMBA_AVAILABLE else "NumPy/vectorized kernels"

    @property
    def last_step_used_numba(self) -> bool:
        return bool(self._last_step_used_numba)

    @property
    def temperatures(self) -> FloatArray:
        return self._temperatures

    @property
    def temperature_field(self) -> TemperatureField:
        return TemperatureField(self._temperatures.copy(), self.time)

    def set_initial_temperature(self, temperature: Union[float, FloatArray]) -> None:
        if isinstance(temperature, (int, float)):
            self._temperatures.fill(temperature)
        else:
            self._temperatures[:] = temperature
        self._update_tabulated_bodies(self._temperatures)

    def add_boundary(
        self,
        indices: IntArray,
        boundary_type: BoundaryType,
        value: float = 0.0,
        htc: float = 0.0,
        T_infinity: float = 25.0,
        time_function: Optional[Callable[[float], float]] = None,
        weights: Optional[FloatArray] = None,
    ) -> None:
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if len(weights) != len(np.asarray(indices).reshape(-1)):
                raise ValueError("weights must have one entry per boundary index")
        bc = BoundaryCondition(
            type=boundary_type,
            indices=np.asarray(indices, dtype=np.int32),
            value=value,
            htc=htc,
            T_infinity=T_infinity,
            time_function=time_function,
            weights=weights,
        )
        self._boundaries.append(bc)

    @property
    def boundaries(self) -> List["BoundaryCondition"]:
        return list(self._boundaries)

    def clear_boundaries(self, boundary_type: Optional[BoundaryType] = None) -> int:
        before = len(self._boundaries)
        if boundary_type is None:
            self._boundaries.clear()
        else:
            self._boundaries = [
                bc for bc in self._boundaries if bc.type is not boundary_type
            ]
        return before - len(self._boundaries)

    def set_heat_source(self, source: FloatArray) -> None:
        self._heat_source[:] = source

    def get_stable_timestep(
        self, alpha: Optional[FloatArray] = None, rho_cp: Optional[FloatArray] = None
    ) -> float:
        if isinstance(self._inv_dx2, np.ndarray):
            min_dx2 = 1.0 / float(np.max(self._inv_dx2))
        else:
            min_dx2 = 1.0 / float(self._inv_dx2)
        if self._capacity_weights is not None:
            return self._cut_cell_stable_timestep(alpha, rho_cp)
        dt = 0.9 * min_dx2 / (4.0 * self._get_alpha_max(alpha))
        if self._sink_cells is not None:
            sink_rate = float(np.max(self._external_sink_rates(self.volumetric_heat_capacity())))
            if np.isfinite(sink_rate) and sink_rate > 0.0:
                dt = min(dt, 0.9 / sink_rate)
        if self._interface_links is not None:
            T = self._temperatures
            link_rates = self._interface_link_rates(self._step_alpha_array(T), self.volumetric_heat_capacity(T))
            max_link_rate = float(np.max(link_rates)) if len(link_rates) else 0.0
            if np.isfinite(max_link_rate) and max_link_rate > 0.0:
                dt = min(dt, 0.9 / max_link_rate)
        return dt

    def volumetric_heat_capacity(self, temperatures: Optional[FloatArray] = None) -> FloatArray:
        T = self._temperatures if temperatures is None else np.asarray(temperatures, dtype=np.float64)
        if self._capacity_curve is not None:
            if NUMBA_AVAILABLE:
                curve_kernel = (
                    interpolate_curve_serial if runs_serial(len(T), _COST_CURVE_NS) else interpolate_curve_parallel
                )
                rho_cp = curve_kernel(
                    np.ascontiguousarray(T, dtype=np.float64), self._curve_temp, self._capacity_curve
                )
            else:
                rho_cp = np.interp(T, self._curve_temp, self._capacity_curve)
        else:
            rho_cp = np.full(len(T), self._rho_cp_scalar, dtype=np.float64)
        if self._rho_cp_overrides is not None:
            rcp_idx, rcp_vals = self._rho_cp_overrides
            rho_cp[rcp_idx] = rcp_vals
        return rho_cp

    def _fuses_flux(self) -> bool:
        return bool(
            NUMBA_AVAILABLE
            and self._capacity_weights is not None
            and self.neighbors.ndim == 2
            and self.neighbors.shape[1] == 26
            and fused_kernels_enabled()
        )

    def _cut_cell_flux(
        self,
        T: FloatArray,
        alpha: FloatArray,
        rho_cp: FloatArray,
        *,
        rates: Optional[FloatArray],
        apply_dt: Optional[float],
        source: Optional[FloatArray],
        out: FloatArray,
    ) -> None:
        kernel = (
            cut_cell_flux_serial if runs_serial(self.n_voxels, _COST_CUT_CELL_FLUX_NS) else cut_cell_flux_parallel
        )
        unused = self._inv_dx2_arr
        kernel(
            T,
            self.neighbors,
            np.ascontiguousarray(alpha, dtype=np.float64),
            rho_cp,
            self._capacity_weights,
            self._conductance_factors,
            self._link_apertures,
            self._inv_dx2_arr,
            self._contact_hdx,
            rates is not None,
            rates if rates is not None else unused,
            apply_dt is not None,
            float(apply_dt) if apply_dt is not None else 0.0,
            source if source is not None else unused,
            out,
        )

    def heat_capacity_lookup(self) -> tuple:
        empty = np.zeros(1, dtype=np.float64)
        use_curve = self._capacity_curve is not None
        positions = self._override_positions(self._rho_cp_overrides, "_rho_cp_override_positions")
        return (
            use_curve,
            self._curve_temp if use_curve else empty,
            self._capacity_curve if use_curve else empty,
            float(self._rho_cp_scalar),
            positions is not None,
            positions if positions is not None else np.zeros(1, dtype=np.int64),
            np.ascontiguousarray(self._rho_cp_overrides[1], dtype=np.float64) if positions is not None else empty,
        )

    def _fuses_properties(self) -> bool:
        return bool(
            NUMBA_AVAILABLE
            and self.use_temperature_dependent
            and self._curve_temp is not None
            and self._capacity_curve is not None
        )

    def _diffusivity_and_capacity(self, T: FloatArray) -> tuple[FloatArray, FloatArray]:
        field = np.ascontiguousarray(T, dtype=np.float64)
        kernel = (
            compute_diffusivity_and_capacity_serial
            if runs_serial(len(field), _COST_DIFFUSIVITY_AND_CAPACITY_NS)
            else compute_diffusivity_and_capacity_parallel
        )
        alpha, rho_cp = kernel(
            field, self._curve_temp, self._curve_k, self._curve_rho, self._curve_cp, self._capacity_curve
        )
        if self._rho_cp_overrides is not None:
            rcp_idx, rcp_vals = self._rho_cp_overrides
            rho_cp[rcp_idx] = rcp_vals
        return alpha, rho_cp

    def _cut_cell_stable_timestep(
        self,
        alpha: Optional[FloatArray] = None,
        rho_cp: Optional[FloatArray] = None,
        conduction_rates: Optional[FloatArray] = None,
    ) -> float:
        T = self._temperatures
        if self.use_temperature_dependent:
            alpha_arr = (
                np.array(alpha, dtype=np.float64, copy=True)
                if alpha is not None
                else self._get_temperature_dependent_diffusivity(T)
            )
        else:
            alpha_arr = np.full(self.n_voxels, float(self._alpha), dtype=np.float64)
        if self._diffusivity_overrides is not None:
            override_idx, override_vals = self._diffusivity_overrides
            alpha_arr[override_idx] = override_vals
        if rho_cp is None:
            rho_cp = self.volumetric_heat_capacity(T)
        self._stable_rho_cp = rho_cp
        if not NUMBA_AVAILABLE:
            rates_kernel = cut_cell_conduction_rates_numpy
        elif runs_serial(self.n_voxels, _COST_CONDUCTION_RATES_NS):
            rates_kernel = cut_cell_conduction_rates_serial
        else:
            rates_kernel = cut_cell_conduction_rates_parallel
        if conduction_rates is not None:
            rates = conduction_rates
        else:
            rates = rates_kernel(
                self.neighbors,
                np.ascontiguousarray(alpha_arr),
                rho_cp,
                self._capacity_weights,
                self._conductance_factors,
                self._link_apertures,
                self._inv_dx2_arr,
            )
        if self._boundary_area_per_volume is not None:
            apv = self._boundary_area_per_volume
            for bc in self._boundaries:
                if (
                    bc.type not in (BoundaryType.CONVECTION, BoundaryType.INTERFACE, BoundaryType.RADIATION)
                    or len(bc.indices) == 0
                ):
                    continue
                idx = bc.indices
                np.add.at(rates, idx, bc.surface_htc(T, self.time) * apv[idx] / rho_cp[idx])
        if self._interface_links is not None:
            rates = rates + self._interface_link_rates(alpha_arr, rho_cp)
        if self._sink_cells is not None:
            rates = rates + self._external_sink_rates(rho_cp)
        max_rate = float(np.max(rates)) if len(rates) else 0.0
        if not np.isfinite(max_rate) or max_rate <= 0.0:
            min_dx2 = 1.0 / float(np.max(self._inv_dx2_arr))
            return 0.9 * min_dx2 / (4.0 * float(self._alpha))
        return 0.9 / max_rate

    def _get_alpha_max(self, alpha: Optional[FloatArray] = None) -> float:
        override_all = getattr(self, "_overrides_cover_all", None)
        if override_all is None:
            override_all = self._diffusivity_overrides is not None and (
                len(np.unique(self._diffusivity_overrides[0])) == self.n_voxels
            )
            self._overrides_cover_all = override_all
        if override_all:
            alpha_max = 0.0
        elif self.use_temperature_dependent:
            if alpha is None:
                alpha = self._get_temperature_dependent_diffusivity(self._temperatures)
            alpha_max = float(np.max(alpha))
        else:
            alpha_max = float(self._alpha)
        if self._diffusivity_overrides is not None:
            override_idx, override_vals = self._diffusivity_overrides
            if len(override_idx) > 0:
                alpha_max = max(alpha_max, float(np.max(override_vals)))
        if self._rho_cp_overrides is not None:
            rcp_idx, rcp_vals = self._rho_cp_overrides
            if len(rcp_idx) > 0:
                alpha_max = 2.0 * alpha_max
        if not np.isfinite(alpha_max) or alpha_max <= 0.0:
            return float(self._alpha)
        return alpha_max

    def get_gradient_magnitude(self, neighbors: IntArray, voxel_size: float) -> FloatArray:
        neighbor_arr = np.asarray(neighbors, dtype=np.int32)
        values = np.ascontiguousarray(self._temperatures[: len(neighbor_arr)], dtype=np.float64)
        if NUMBA_AVAILABLE:
            gradient_kernel = (
                compute_gradient_magnitude_serial
                if runs_serial(len(values), _COST_GRADIENT_NS)
                else compute_gradient_magnitude_parallel
            )
            return gradient_kernel(
                values,
                neighbor_arr,
                float(voxel_size),
            )

        n_voxels = len(values)
        gradients = np.zeros(n_voxels, dtype=np.float32)
        for i in range(n_voxels):
            grad_sq = 0.0
            count = 0
            for j in range(neighbor_arr.shape[1]):
                neighbor_idx = neighbor_arr[i, j]
                if neighbor_idx >= 0:
                    dT = values[neighbor_idx] - values[i]
                    grad_sq += (dT / voxel_size) ** 2
                    count += 1
            if count > 0:
                gradients[i] = np.sqrt(grad_sq / count)
        return gradients

    def step(self, dt: float) -> None:
        self._update_tabulated_bodies(self._temperatures)
        rho_cp_fused = None
        if not self.use_temperature_dependent:
            alpha0 = None
        elif self._fuses_properties():
            alpha0, rho_cp_fused = self._diffusivity_and_capacity(self._temperatures)
        else:
            alpha0 = self._get_temperature_dependent_diffusivity(self._temperatures)

        self._stable_rho_cp = None
        self._stable_link_g = None
        flux0 = None
        if self._fuses_flux():
            T0 = self._temperatures
            if alpha0 is None:
                alpha0 = np.full(self.n_voxels, float(self._alpha), dtype=np.float64)
            if self._diffusivity_overrides is not None:
                override_idx, override_vals = self._diffusivity_overrides
                alpha0[override_idx] = override_vals
            if rho_cp_fused is None:
                rho_cp_fused = self.volumetric_heat_capacity(T0)
            conduction_rates = np.empty(self.n_voxels, dtype=np.float64)
            flux0 = np.empty(self.n_voxels, dtype=np.float64)
            self._cut_cell_flux(
                T0, alpha0, rho_cp_fused, rates=conduction_rates, apply_dt=None, source=None, out=flux0
            )
            dt_stable = self._cut_cell_stable_timestep(alpha0, rho_cp_fused, conduction_rates)
        else:
            dt_stable = (
                self.get_stable_timestep(alpha=alpha0, rho_cp=rho_cp_fused)
                if alpha0 is not None
                else self.get_stable_timestep()
            )
        rho_cp0 = self._stable_rho_cp
        link_g0 = self._stable_link_g
        self._stable_rho_cp = None
        self._stable_link_g = None
        if dt > dt_stable:
            n_substeps = int(np.ceil(dt / dt_stable))
            dt_sub = dt / n_substeps
            for substep in range(n_substeps):
                if substep > 0:
                    self._update_tabulated_bodies(self._temperatures)
                self._step_internal(
                    dt_sub,
                    alpha_precomputed=alpha0 if substep == 0 else None,
                    rho_cp_precomputed=rho_cp0 if substep == 0 else None,
                    link_conductance_precomputed=link_g0 if substep == 0 else None,
                    flux_precomputed=flux0 if substep == 0 else None,
                )
        else:
            self._step_internal(
                dt,
                alpha_precomputed=alpha0,
                rho_cp_precomputed=rho_cp0,
                link_conductance_precomputed=link_g0,
                flux_precomputed=flux0,
            )

        if self._slave_cells is not None and len(self._slave_cells[0]) > 0:
            slaves, masters = self._slave_cells
            self._temperatures[slaves] = self._temperatures[masters]

        self.time += dt

    def _step_internal(
        self,
        dt: float,
        alpha_precomputed: Optional[FloatArray] = None,
        rho_cp_precomputed: Optional[FloatArray] = None,
        link_conductance_precomputed: Optional[FloatArray] = None,
        flux_precomputed: Optional[FloatArray] = None,
    ) -> None:
        T = self._temperatures
        T_new = self._temperatures_new

        if not self.use_temperature_dependent:
            alpha = self._alpha
        elif alpha_precomputed is not None:
            alpha = alpha_precomputed
        elif rho_cp_precomputed is None and self._fuses_properties():
            alpha, rho_cp_precomputed = self._diffusivity_and_capacity(T)
        else:
            alpha = self._get_temperature_dependent_diffusivity(T)

        if self._diffusivity_overrides is not None:
            if not isinstance(alpha, np.ndarray):
                alpha = np.full(self.n_voxels, alpha, dtype=np.float64)
            override_idx, override_vals = self._diffusivity_overrides
            alpha[override_idx] = override_vals

        cut_cells = self._capacity_weights is not None
        links = self._interface_links is not None or self._sink_cells is not None
        use_numba_kernels = bool(
            self._numba_parallel_eligible
            and not isinstance(alpha, np.ndarray)
            and self._rho_cp_overrides is None
            and not cut_cells
            and not self._has_contact
            and not links
        )
        heterogeneous = (
            isinstance(alpha, np.ndarray)
            or self._rho_cp_overrides is not None
            or cut_cells
            or self._has_contact
            or links
        )
        use_heterogeneous_numba = bool(NUMBA_AVAILABLE and heterogeneous)
        self._last_step_used_numba = use_numba_kernels or use_heterogeneous_numba

        if heterogeneous:
            laplacian = None
        elif use_numba_kernels:
            laplacian = compute_laplacian_parallel(
                T,
                self.neighbors,
                self._inv_dx2_arr,
            )
        else:
            laplacian = np.zeros(self.n_voxels, dtype=np.float64)
            max_slot = self.neighbors.shape[1]

            slot_pairs = [(0, 1), (2, 3), (4, 5)]
            for j_a, j_b in slot_pairs:
                if j_a >= max_slot or j_b >= max_slot:
                    continue
                n_a = self.neighbors[:, j_a]
                n_b = self.neighbors[:, j_b]
                v_a = n_a >= 0
                v_b = n_b >= 0
                dT_a = np.where(v_a, T[np.maximum(n_a, 0)] - T, 0.0)
                dT_b = np.where(v_b, T[np.maximum(n_b, 0)] - T, 0.0)
                laplacian += 0.5 * (dT_a + dT_b)

            corner_start = min(18, max_slot)
            corner_end = min(26, max_slot)
            j = corner_start
            while j < corner_end - 1:
                n_a = self.neighbors[:, j]
                n_b = self.neighbors[:, j + 1]
                v_a = n_a >= 0
                v_b = n_b >= 0
                dT_a = np.where(v_a, T[np.maximum(n_a, 0)] - T, 0.0)
                dT_b = np.where(v_b, T[np.maximum(n_b, 0)] - T, 0.0)
                laplacian += 0.125 * (dT_a + dT_b)
                j += 2
            if j < corner_end:
                n_j = self.neighbors[:, j]
                valid = n_j >= 0
                laplacian += 0.125 * np.where(valid, T[np.maximum(n_j, 0)] - T, 0.0)

            laplacian *= self._inv_dx2

        rho_cp = self._rho_cp_scalar
        source_term = self._heat_source / rho_cp

        if heterogeneous:
            if not cut_cells:
                rho_cp_arr = np.full(self.n_voxels, rho_cp, dtype=np.float64)
                if self._rho_cp_overrides is not None:
                    rcp_idx, rcp_vals = self._rho_cp_overrides
                    rho_cp_arr[rcp_idx] = rcp_vals
            if cut_cells:
                rho_cp_arr = (
                    rho_cp_precomputed if rho_cp_precomputed is not None else self.volumetric_heat_capacity(T)
                )
                self._step_rho_cp = rho_cp_arr
                alpha_arr = (
                    alpha
                    if isinstance(alpha, np.ndarray)
                    else np.full(self.n_voxels, alpha, dtype=np.float64)
                )
                if flux_precomputed is not None:
                    apply_kernel = (
                        apply_cut_cell_flux_serial
                        if runs_serial(self.n_voxels, _COST_APPLY_FLUX_NS)
                        else apply_cut_cell_flux_parallel
                    )
                    apply_kernel(
                        T, flux_precomputed, rho_cp_arr, self._capacity_weights, source_term, float(dt), T_new
                    )
                elif self._fuses_flux():
                    self._cut_cell_flux(
                        T, alpha_arr, rho_cp_arr, rates=None, apply_dt=dt, source=source_term, out=T_new
                    )
                else:
                    cut_cell_update = (
                        update_heterogeneous_cut_cell_parallel
                        if use_heterogeneous_numba
                        else update_heterogeneous_cut_cell_numpy
                    )
                    T_new[:] = cut_cell_update(
                        T,
                        self.neighbors,
                        np.ascontiguousarray(alpha_arr, dtype=np.float64),
                        rho_cp_arr,
                        self._capacity_weights,
                        self._conductance_factors,
                        self._link_apertures,
                        self._inv_dx2_arr,
                        source_term,
                        dt,
                        self._contact_hdx,
                    )
            elif use_heterogeneous_numba:
                alpha_arr = (
                    alpha
                    if isinstance(alpha, np.ndarray)
                    else np.full(self.n_voxels, alpha, dtype=np.float64)
                )
                T_new[:] = update_heterogeneous_parallel(
                    T,
                    self.neighbors,
                    alpha_arr,
                    rho_cp_arr,
                    self._inv_dx2_arr,
                    source_term,
                    dt,
                    self._contact_hdx,
                )
            else:
                max_slot = self.neighbors.shape[1]
                slot_pairs = [(0, 1), (2, 3), (4, 5)]
                corner_start = min(18, max_slot)
                corner_end = min(26, max_slot)
                k_arr = alpha * rho_cp_arr
                g_arr = self._contact_hdx

                flux = np.zeros(self.n_voxels, dtype=np.float64)
                k0 = k_arr
                for j_a, j_b in slot_pairs:
                    if j_a >= max_slot or j_b >= max_slot:
                        continue
                    n_a = self.neighbors[:, j_a]
                    n_b = self.neighbors[:, j_b]
                    v_a = n_a >= 0
                    v_b = n_b >= 0
                    k_a = np.where(v_a, k_arr[np.maximum(n_a, 0)], 0.0)
                    k_b = np.where(v_b, k_arr[np.maximum(n_b, 0)], 0.0)
                    k_ia = contact_limited_conductivity(
                        2.0 * k0 * k_a / (k0 + k_a + 1e-30), g_arr, g_arr[np.maximum(n_a, 0)]
                    )
                    k_ib = contact_limited_conductivity(
                        2.0 * k0 * k_b / (k0 + k_b + 1e-30), g_arr, g_arr[np.maximum(n_b, 0)]
                    )
                    dT_a = np.where(v_a, T[np.maximum(n_a, 0)] - T, 0.0)
                    dT_b = np.where(v_b, T[np.maximum(n_b, 0)] - T, 0.0)
                    flux += 0.5 * (k_ia * dT_a + k_ib * dT_b)
                j = corner_start
                while j < corner_end - 1:
                    n_a = self.neighbors[:, j]
                    n_b = self.neighbors[:, j + 1]
                    v_a = n_a >= 0
                    v_b = n_b >= 0
                    k_a = np.where(v_a, k_arr[np.maximum(n_a, 0)], 0.0)
                    k_b = np.where(v_b, k_arr[np.maximum(n_b, 0)], 0.0)
                    k_ia = contact_limited_conductivity(
                        2.0 * k0 * k_a / (k0 + k_a + 1e-30), g_arr, g_arr[np.maximum(n_a, 0)]
                    )
                    k_ib = contact_limited_conductivity(
                        2.0 * k0 * k_b / (k0 + k_b + 1e-30), g_arr, g_arr[np.maximum(n_b, 0)]
                    )
                    dT_a = np.where(v_a, T[np.maximum(n_a, 0)] - T, 0.0)
                    dT_b = np.where(v_b, T[np.maximum(n_b, 0)] - T, 0.0)
                    flux += 0.125 * (k_ia * dT_a + k_ib * dT_b)
                    j += 2
                if j < corner_end:
                    n_j = self.neighbors[:, j]
                    valid = n_j >= 0
                    k_j = np.where(valid, k_arr[np.maximum(n_j, 0)], 0.0)
                    k_ij = contact_limited_conductivity(
                        2.0 * k0 * k_j / (k0 + k_j + 1e-30), g_arr, g_arr[np.maximum(n_j, 0)]
                    )
                    dT_j = np.where(valid, T[np.maximum(n_j, 0)] - T, 0.0)
                    flux += 0.125 * (k_ij * dT_j)

                T_new[:] = T + dt * (flux * self._inv_dx2) / rho_cp_arr + dt * source_term
        else:
            if use_numba_kernels:
                T_new[:] = update_temperatures_parallel(
                    T,
                    laplacian,
                    source_term,
                    float(alpha),
                    dt,
                )
            else:
                T_new[:] = T + dt * (alpha * laplacian + source_term)

        self._apply_boundaries(T_new, T, dt)

        if links:
            step_alpha = (
                alpha if isinstance(alpha, np.ndarray) else np.full(self.n_voxels, float(alpha), dtype=np.float64)
            )
            if self._interface_links is not None:
                self._apply_interface_links(
                    T_new, T, dt, step_alpha, rho_cp_arr, conductance=link_conductance_precomputed
                )
            self._apply_external_sink(T_new, T, dt, rho_cp_arr)
        self._correct_tabulated_enthalpy(T_new, T)

        self._temperatures, self._temperatures_new = self._temperatures_new, self._temperatures

    def _apply_boundaries(self, T_new: FloatArray, T_old: FloatArray, dt: float) -> None:
        rho_cp = self._rho_cp_scalar
        rho_cp_arr = None
        if self._rho_cp_overrides is not None:
            rcp_idx, rcp_vals = self._rho_cp_overrides
            rho_cp_arr = np.full(self.n_voxels, rho_cp, dtype=np.float64)
            rho_cp_arr[rcp_idx] = rcp_vals
        if self._boundary_area_per_volume is not None:
            step_rho_cp = self._step_rho_cp
            self._apply_cut_cell_boundaries(
                T_new, T_old, dt, step_rho_cp if step_rho_cp is not None else rho_cp_arr
            )
            return
        for bc in self._boundaries:
            if bc.type == BoundaryType.FIXED_TEMPERATURE:
                T_new[bc.indices] = bc.get_effective_value(self.time)

            elif bc.type in (BoundaryType.CONVECTION, BoundaryType.RADIATION):
                h = bc.surface_htc(T_old, self.time)
                T_inf = bc.T_infinity
                idx = bc.indices
                dx_i = (
                    self._voxel_sizes[idx]
                    if self._voxel_sizes is not None
                    else self.voxel_size
                )
                rc_i = rho_cp_arr[idx] if rho_cp_arr is not None else rho_cp
                T_new[idx] -= dt * h * (T_old[idx] - T_inf) / (rc_i * dx_i)

            elif bc.type == BoundaryType.INTERFACE:
                h = bc.surface_htc(T_old, self.time)
                T_mold = bc.T_infinity
                idx = bc.indices
                dx_i = (
                    self._voxel_sizes[idx]
                    if self._voxel_sizes is not None
                    else self.voxel_size
                )
                rc_i = rho_cp_arr[idx] if rho_cp_arr is not None else rho_cp
                T_new[idx] -= dt * h * (T_old[idx] - T_mold) / (rc_i * dx_i)

            elif bc.type == BoundaryType.HEAT_FLUX:
                q = bc.get_effective_value(self.time)
                idx = bc.indices
                dx_i = (
                    self._voxel_sizes[idx]
                    if self._voxel_sizes is not None
                    else self.voxel_size
                )
                rc_i = rho_cp_arr[idx] if rho_cp_arr is not None else rho_cp
                T_new[idx] += dt * q / (rc_i * dx_i)

            elif bc.type == BoundaryType.ADIABATIC:
                pass

    def _apply_cut_cell_boundaries(
        self,
        T_new: FloatArray,
        T_old: FloatArray,
        dt: float,
        rho_cp_arr: Optional[FloatArray],
    ) -> None:
        apv = self._boundary_area_per_volume
        rho_cp = self._rho_cp_scalar
        for bc in self._boundaries:
            idx = bc.indices
            rc_i = rho_cp_arr[idx] if rho_cp_arr is not None else rho_cp
            if bc.type == BoundaryType.FIXED_TEMPERATURE:
                T_new[idx] = bc.get_effective_value(self.time)
            elif bc.type in (BoundaryType.CONVECTION, BoundaryType.INTERFACE, BoundaryType.RADIATION):
                h = bc.surface_htc(T_old, self.time)
                T_new[idx] -= dt * h * (T_old[idx] - bc.T_infinity) * apv[idx] / rc_i
            elif bc.type == BoundaryType.HEAT_FLUX:
                T_new[idx] += dt * bc.get_effective_value(self.time) * apv[idx] / rc_i

    def _get_temperature_dependent_diffusivity(self, T: FloatArray) -> FloatArray:
        if self._curve_temp is not None:
            if NUMBA_AVAILABLE:
                diffusivity_kernel = (
                    compute_diffusivity_from_curves_serial
                    if runs_serial(len(T), _COST_DIFFUSIVITY_NS)
                    else compute_diffusivity_from_curves_parallel
                )
                return diffusivity_kernel(
                    np.ascontiguousarray(T, dtype=np.float64),
                    self._curve_temp,
                    self._curve_k,
                    self._curve_rho,
                    self._curve_cp,
                )
            k = np.interp(T, self._curve_temp, self._curve_k)
            rho = np.interp(T, self._curve_temp, self._curve_rho)
            cp = np.interp(T, self._curve_temp, self._curve_cp)
            return k / (rho * cp)

        T_solidus = self.alloy.solidus_temp
        T_liquidus = self.alloy.liquidus_temp

        f_liquid = np.clip(
            (T - T_solidus) / (T_liquidus - T_solidus),
            0.0, 1.0
        )

        alpha_solid = self._alpha
        alpha_liquid = self._alpha * 0.7

        return alpha_solid * (1 - f_liquid) + alpha_liquid * f_liquid

    def is_solidified(self, threshold_temp: Optional[float] = None) -> bool:
        threshold = threshold_temp or self.alloy.solidus_temp
        return np.all(self._temperatures <= threshold)

    def get_cooling_rate(self, prev_temperatures: FloatArray, dt: float) -> FloatArray:
        return (prev_temperatures - self._temperatures) / dt

def _interp(x: FloatArray, xp: FloatArray, fp: FloatArray) -> FloatArray:
    x = np.ascontiguousarray(x, dtype=np.float64)
    if NUMBA_AVAILABLE and len(x) > 4096:
        return interpolate_sorted_parallel(x, xp, fp)
    return np.interp(x, xp, fp)

def _stencil_slot_weights(max_slot: int) -> np.ndarray:
    weights = np.zeros(max_slot, dtype=np.float64)
    weights[: min(6, max_slot)] = 0.5
    weights[min(18, max_slot) : min(26, max_slot)] = 0.125
    return weights

CUT_CELL_RATE_ALLOWANCE = 1.0

def cut_cell_link_apertures(capacity_weights: FloatArray) -> FloatArray:
    return np.clip(np.asarray(capacity_weights, dtype=np.float64), 0.0, 1.0)

def cut_cell_conductance_factors(
    capacity_weights: FloatArray,
    neighbors: IntArray,
    apertures: Optional[FloatArray] = None,
    rate_allowance: float = CUT_CELL_RATE_ALLOWANCE,
) -> FloatArray:
    nbrs = np.asarray(neighbors)
    weights = np.asarray(capacity_weights, dtype=np.float64)
    a = np.ones(len(weights)) if apertures is None else np.asarray(apertures, dtype=np.float64)
    slot_weights = _stencil_slot_weights(nbrs.shape[1])
    valid = nbrs >= 0
    pair = 0.5 * (a[:, None] + a[np.maximum(nbrs, 0)])
    present = np.sum(np.where(valid, slot_weights[None, :] * pair, 0.0), axis=1)
    full = float(np.sum(slot_weights))
    factors = np.ones(len(weights), dtype=np.float64)
    linked = present > 0.0
    factors[linked] = np.minimum(1.0, rate_allowance * full * weights[linked] / present[linked])
    factors[weights <= 0.0] = 0.0
    return factors

def contact_limited_conductivity(k_pair: FloatArray, g_i: FloatArray, g_j: FloatArray) -> FloatArray:
    crossing = (g_i > 0.0) != (g_j > 0.0)
    if not np.any(crossing):
        return k_pair
    g = np.where(crossing, g_i + g_j, 1.0)
    return np.where(crossing, k_pair / (1.0 + k_pair / g), k_pair)

def update_heterogeneous_cut_cell_numpy(
    T: FloatArray,
    neighbors: IntArray,
    alpha: FloatArray,
    rho_cp: FloatArray,
    capacity_weights: FloatArray,
    conductance_factors: FloatArray,
    apertures: FloatArray,
    inv_dx2: FloatArray,
    source: FloatArray,
    dt: float,
    contact_hdx: FloatArray,
) -> FloatArray:
    n = len(T)
    max_slot = neighbors.shape[1]
    k_arr = alpha * rho_cp
    f = conductance_factors
    flux = np.zeros(n, dtype=np.float64)

    def pair_term(slot: int) -> FloatArray:
        nb = neighbors[:, slot]
        valid = nb >= 0
        safe = np.maximum(nb, 0)
        k_n = np.where(valid, k_arr[safe], 0.0)
        s = np.minimum(f, np.where(valid, f[safe], 0.0)) * 0.5 * (apertures + apertures[safe])
        k_in = s * contact_limited_conductivity(
            2.0 * k_arr * k_n / (k_arr + k_n + 1e-30), contact_hdx, contact_hdx[safe]
        )
        return np.where(valid, k_in * (T[safe] - T), 0.0)

    for j_a in range(0, min(6, max_slot) - 1, 2):
        flux += 0.5 * (pair_term(j_a) + pair_term(j_a + 1))
    j = min(18, max_slot)
    corner_end = min(26, max_slot)
    while j < corner_end - 1:
        flux += 0.125 * (pair_term(j) + pair_term(j + 1))
        j += 2
    if j < corner_end:
        flux += 0.125 * pair_term(j)

    capacity = rho_cp * capacity_weights
    T_new = np.array(T, dtype=np.float64, copy=True)
    active = capacity_weights > 0.0
    T_new[active] = (
        T[active]
        + dt * (flux[active] * inv_dx2[active]) / capacity[active]
        + dt * source[active]
    )
    return T_new

def cut_cell_conduction_rates_numpy(
    neighbors: IntArray,
    alpha: FloatArray,
    rho_cp: FloatArray,
    capacity_weights: FloatArray,
    conductance_factors: FloatArray,
    apertures: FloatArray,
    inv_dx2: FloatArray,
) -> FloatArray:
    max_slot = neighbors.shape[1]
    slot_weights = _stencil_slot_weights(max_slot)
    k_arr = alpha * rho_cp
    total = np.zeros(len(alpha), dtype=np.float64)
    for slot in range(max_slot):
        if slot_weights[slot] == 0.0:
            continue
        nb = neighbors[:, slot]
        valid = nb >= 0
        safe = np.maximum(nb, 0)
        k_n = k_arr[safe]
        s = np.minimum(conductance_factors, conductance_factors[safe]) * 0.5 * (apertures + apertures[safe])
        total += np.where(valid, slot_weights[slot] * s * (2.0 * k_arr * k_n / (k_arr + k_n + 1e-30)), 0.0)
    capacity = rho_cp * capacity_weights
    return np.divide(total * inv_dx2, capacity, out=np.zeros(len(alpha)), where=capacity_weights > 0.0)

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def compute_diffusivity_from_curves_parallel(
        T: FloatArray,
        temp_points: FloatArray,
        k_vals: FloatArray,
        rho_vals: FloatArray,
        cp_vals: FloatArray,
    ) -> FloatArray:
        n = len(T)
        m = len(temp_points)
        alpha = np.empty(n, dtype=np.float64)
        t_lo = temp_points[0]
        t_hi = temp_points[m - 1]

        for i in prange(n):
            t = T[i]
            if t <= t_lo:
                k = k_vals[0]
                rho = rho_vals[0]
                cp = cp_vals[0]
            elif t >= t_hi:
                k = k_vals[m - 1]
                rho = rho_vals[m - 1]
                cp = cp_vals[m - 1]
            else:
                j = 0
                for s in range(m - 1):
                    if t < temp_points[s + 1]:
                        j = s
                        break
                x0 = temp_points[j]
                inv_dx = 1.0 / (temp_points[j + 1] - x0)
                offset = t - x0
                k = (k_vals[j + 1] - k_vals[j]) * inv_dx * offset + k_vals[j]
                rho = (rho_vals[j + 1] - rho_vals[j]) * inv_dx * offset + rho_vals[j]
                cp = (cp_vals[j + 1] - cp_vals[j]) * inv_dx * offset + cp_vals[j]
            alpha[i] = k / (rho * cp)

        return alpha

    @njit(parallel=True, cache=True)
    def compute_laplacian_parallel(
        T: FloatArray,
        neighbors: IntArray,
        inv_dx2: FloatArray,
    ) -> FloatArray:
        n = len(T)
        laplacian = np.zeros(n, dtype=np.float64)
        max_slot = neighbors.shape[1]
        slot_face_end = min(6, max_slot)
        slot_corner_start = min(18, max_slot)
        slot_corner_end = min(26, max_slot)

        for i in prange(n):
            t_i = T[i]
            sum_dT = 0.0

            for j in range(slot_face_end):
                neighbor_idx = neighbors[i, j]
                if neighbor_idx >= 0:
                    sum_dT += 0.5 * (T[neighbor_idx] - t_i)

            for j in range(slot_corner_start, slot_corner_end):
                neighbor_idx = neighbors[i, j]
                if neighbor_idx >= 0:
                    sum_dT += 0.125 * (T[neighbor_idx] - t_i)

            laplacian[i] = sum_dT * inv_dx2[i]

        return laplacian

    @njit(parallel=True, cache=True)
    def update_temperatures_parallel(
        T: FloatArray,
        laplacian: FloatArray,
        source: FloatArray,
        alpha: float,
        dt: float
    ) -> FloatArray:
        n = len(T)
        T_new = np.empty(n, dtype=np.float64)

        for i in prange(n):
            T_new[i] = T[i] + dt * (alpha * laplacian[i] + source[i])

        return T_new

    @njit(parallel=True, cache=True)
    def compute_diffusivity_and_capacity_parallel(
        T: FloatArray,
        temp_points: FloatArray,
        k_vals: FloatArray,
        rho_vals: FloatArray,
        cp_vals: FloatArray,
        capacity_vals: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        n = len(T)
        m = len(temp_points)
        alpha = np.empty(n, dtype=np.float64)
        rho_cp = np.empty(n, dtype=np.float64)
        t_lo = temp_points[0]
        t_hi = temp_points[m - 1]

        for i in prange(n):
            t = T[i]
            if t <= t_lo:
                k = k_vals[0]
                rho = rho_vals[0]
                cp = cp_vals[0]
                capacity = capacity_vals[0]
            elif t >= t_hi:
                k = k_vals[m - 1]
                rho = rho_vals[m - 1]
                cp = cp_vals[m - 1]
                capacity = capacity_vals[m - 1]
            else:
                j = 0
                for s in range(m - 1):
                    if t < temp_points[s + 1]:
                        j = s
                        break
                x0 = temp_points[j]
                inv_dx = 1.0 / (temp_points[j + 1] - x0)
                offset = t - x0
                k = (k_vals[j + 1] - k_vals[j]) * inv_dx * offset + k_vals[j]
                rho = (rho_vals[j + 1] - rho_vals[j]) * inv_dx * offset + rho_vals[j]
                cp = (cp_vals[j + 1] - cp_vals[j]) * inv_dx * offset + cp_vals[j]
                capacity = (capacity_vals[j + 1] - capacity_vals[j]) * inv_dx * (t - x0) + capacity_vals[j]
            alpha[i] = k / (rho * cp)
            rho_cp[i] = capacity

        return alpha, rho_cp

    @njit(inline="always")
    def _link_conductivity(k_i: float, k_j: float, g_i: float, g_j: float) -> float:
        k = (2.0 * k_i * k_j) / (k_i + k_j + 1e-30)
        if (g_i > 0.0) != (g_j > 0.0):
            k = k / (1.0 + k / (g_i + g_j))
        return k

    @njit(parallel=True, fastmath=True, cache=True)
    def update_heterogeneous_parallel(
        T: FloatArray,
        neighbors: IntArray,
        alpha: FloatArray,
        rho_cp: FloatArray,
        inv_dx2: FloatArray,
        source: FloatArray,
        dt: float,
        contact_hdx: FloatArray,
    ) -> FloatArray:
        n = len(T)
        max_slot = neighbors.shape[1]
        slot_face_end = min(6, max_slot)
        slot_corner_start = min(18, max_slot)
        slot_corner_end = min(26, max_slot)
        T_new = np.empty(n, dtype=np.float64)

        for i in prange(n):
            t_i = T[i]
            k_i = alpha[i] * rho_cp[i]
            g_i = contact_hdx[i]
            flux_sum = 0.0

            for j in range(0, slot_face_end - 1, 2):
                x_a = 0.0
                n_a = neighbors[i, j]
                if n_a >= 0:
                    k_a = alpha[n_a] * rho_cp[n_a]
                    x_a = _link_conductivity(k_i, k_a, g_i, contact_hdx[n_a]) * (T[n_a] - t_i)
                x_b = 0.0
                n_b = neighbors[i, j + 1]
                if n_b >= 0:
                    k_b = alpha[n_b] * rho_cp[n_b]
                    x_b = _link_conductivity(k_i, k_b, g_i, contact_hdx[n_b]) * (T[n_b] - t_i)
                flux_sum += 0.5 * (x_a + x_b)

            j = slot_corner_start
            while j < slot_corner_end - 1:
                x_a = 0.0
                n_a = neighbors[i, j]
                if n_a >= 0:
                    k_a = alpha[n_a] * rho_cp[n_a]
                    x_a = _link_conductivity(k_i, k_a, g_i, contact_hdx[n_a]) * (T[n_a] - t_i)
                x_b = 0.0
                n_b = neighbors[i, j + 1]
                if n_b >= 0:
                    k_b = alpha[n_b] * rho_cp[n_b]
                    x_b = _link_conductivity(k_i, k_b, g_i, contact_hdx[n_b]) * (T[n_b] - t_i)
                flux_sum += 0.125 * (x_a + x_b)
                j += 2
            if j < slot_corner_end:
                n_a = neighbors[i, j]
                if n_a >= 0:
                    k_a = alpha[n_a] * rho_cp[n_a]
                    flux_sum += 0.125 * (
                        _link_conductivity(k_i, k_a, g_i, contact_hdx[n_a]) * (T[n_a] - t_i)
                    )

            T_new[i] = (
                t_i
                + (dt * (flux_sum * inv_dx2[i]) / rho_cp[i])
                + dt * source[i]
            )

        return T_new

    @njit(parallel=True, fastmath=True, cache=True)
    def update_heterogeneous_cut_cell_parallel(
        T: FloatArray,
        neighbors: IntArray,
        alpha: FloatArray,
        rho_cp: FloatArray,
        capacity_weights: FloatArray,
        conductance_factors: FloatArray,
        apertures: FloatArray,
        inv_dx2: FloatArray,
        source: FloatArray,
        dt: float,
        contact_hdx: FloatArray,
    ) -> FloatArray:
        n = len(T)
        max_slot = neighbors.shape[1]
        slot_face_end = min(6, max_slot)
        slot_corner_start = min(18, max_slot)
        slot_corner_end = min(26, max_slot)
        T_new = np.empty(n, dtype=np.float64)

        for i in prange(n):
            t_i = T[i]
            w_i = capacity_weights[i]
            if w_i <= 0.0:
                T_new[i] = t_i
                continue
            k_i = alpha[i] * rho_cp[i]
            g_i = contact_hdx[i]
            f_i = conductance_factors[i]
            a_i = apertures[i]
            flux_sum = 0.0

            for j in range(0, slot_face_end - 1, 2):
                x_a = 0.0
                n_a = neighbors[i, j]
                if n_a >= 0:
                    k_a = alpha[n_a] * rho_cp[n_a]
                    s_a = min(f_i, conductance_factors[n_a]) * 0.5 * (a_i + apertures[n_a])
                    x_a = s_a * _link_conductivity(k_i, k_a, g_i, contact_hdx[n_a]) * (T[n_a] - t_i)
                x_b = 0.0
                n_b = neighbors[i, j + 1]
                if n_b >= 0:
                    k_b = alpha[n_b] * rho_cp[n_b]
                    s_b = min(f_i, conductance_factors[n_b]) * 0.5 * (a_i + apertures[n_b])
                    x_b = s_b * _link_conductivity(k_i, k_b, g_i, contact_hdx[n_b]) * (T[n_b] - t_i)
                flux_sum += 0.5 * (x_a + x_b)

            j = slot_corner_start
            while j < slot_corner_end - 1:
                x_a = 0.0
                n_a = neighbors[i, j]
                if n_a >= 0:
                    k_a = alpha[n_a] * rho_cp[n_a]
                    s_a = min(f_i, conductance_factors[n_a]) * 0.5 * (a_i + apertures[n_a])
                    x_a = s_a * _link_conductivity(k_i, k_a, g_i, contact_hdx[n_a]) * (T[n_a] - t_i)
                x_b = 0.0
                n_b = neighbors[i, j + 1]
                if n_b >= 0:
                    k_b = alpha[n_b] * rho_cp[n_b]
                    s_b = min(f_i, conductance_factors[n_b]) * 0.5 * (a_i + apertures[n_b])
                    x_b = s_b * _link_conductivity(k_i, k_b, g_i, contact_hdx[n_b]) * (T[n_b] - t_i)
                flux_sum += 0.125 * (x_a + x_b)
                j += 2
            if j < slot_corner_end:
                n_a = neighbors[i, j]
                if n_a >= 0:
                    k_a = alpha[n_a] * rho_cp[n_a]
                    s_a = min(f_i, conductance_factors[n_a]) * 0.5 * (a_i + apertures[n_a])
                    flux_sum += 0.125 * (
                        s_a * _link_conductivity(k_i, k_a, g_i, contact_hdx[n_a]) * (T[n_a] - t_i)
                    )

            T_new[i] = (
                t_i
                + (dt * (flux_sum * inv_dx2[i]) / (rho_cp[i] * w_i))
                + dt * source[i]
            )

        return T_new

    @njit(parallel=True, cache=True)
    def cut_cell_flux_parallel(
        T: FloatArray,
        neighbors: IntArray,
        alpha: FloatArray,
        rho_cp: FloatArray,
        capacity_weights: FloatArray,
        conductance_factors: FloatArray,
        apertures: FloatArray,
        inv_dx2: FloatArray,
        contact_hdx: FloatArray,
        want_rates: bool,
        rates: FloatArray,
        apply_now: bool,
        dt: float,
        source: FloatArray,
        out: FloatArray,
    ) -> None:
        n = len(T)
        for i in prange(n):
            t_i = T[i]
            w_i = capacity_weights[i]
            if w_i <= 0.0:
                if want_rates:
                    rates[i] = 0.0
                out[i] = t_i if apply_now else 0.0
                continue
            k_i = alpha[i] * rho_cp[i]
            g_i = contact_hdx[i]
            f_i = conductance_factors[i]
            a_i = apertures[i]
            flux_sum = 0.0
            total = 0.0

            for j in range(0, 6, 2):
                x_a = 0.0
                n_a = neighbors[i, j]
                if n_a >= 0:
                    k_a = alpha[n_a] * rho_cp[n_a]
                    min_f = min(f_i, conductance_factors[n_a])
                    a_sum = a_i + apertures[n_a]
                    link = (2.0 * k_i * k_a) / (k_i + k_a + 1e-30)
                    total += 0.5 * min_f * 0.5 * a_sum * link
                    g_a = contact_hdx[n_a]
                    if (g_i > 0.0) != (g_a > 0.0):
                        link = link / (1.0 + link / (g_i + g_a))
                    x_a = min_f * 0.5 * a_sum * link * (T[n_a] - t_i)
                x_b = 0.0
                n_b = neighbors[i, j + 1]
                if n_b >= 0:
                    k_b = alpha[n_b] * rho_cp[n_b]
                    min_f = min(f_i, conductance_factors[n_b])
                    a_sum = a_i + apertures[n_b]
                    link = (2.0 * k_i * k_b) / (k_i + k_b + 1e-30)
                    total += 0.5 * min_f * 0.5 * a_sum * link
                    g_b = contact_hdx[n_b]
                    if (g_i > 0.0) != (g_b > 0.0):
                        link = link / (1.0 + link / (g_i + g_b))
                    x_b = min_f * 0.5 * a_sum * link * (T[n_b] - t_i)
                flux_sum += 0.5 * (x_a + x_b)

            for j in range(18, 26, 2):
                x_a = 0.0
                n_a = neighbors[i, j]
                if n_a >= 0:
                    k_a = alpha[n_a] * rho_cp[n_a]
                    min_f = min(f_i, conductance_factors[n_a])
                    a_sum = a_i + apertures[n_a]
                    link = (2.0 * k_i * k_a) / (k_i + k_a + 1e-30)
                    total += 0.125 * min_f * 0.5 * a_sum * link
                    g_a = contact_hdx[n_a]
                    if (g_i > 0.0) != (g_a > 0.0):
                        link = link / (1.0 + link / (g_i + g_a))
                    x_a = min_f * 0.5 * a_sum * link * (T[n_a] - t_i)
                x_b = 0.0
                n_b = neighbors[i, j + 1]
                if n_b >= 0:
                    k_b = alpha[n_b] * rho_cp[n_b]
                    min_f = min(f_i, conductance_factors[n_b])
                    a_sum = a_i + apertures[n_b]
                    link = (2.0 * k_i * k_b) / (k_i + k_b + 1e-30)
                    total += 0.125 * min_f * 0.5 * a_sum * link
                    g_b = contact_hdx[n_b]
                    if (g_i > 0.0) != (g_b > 0.0):
                        link = link / (1.0 + link / (g_i + g_b))
                    x_b = min_f * 0.5 * a_sum * link * (T[n_b] - t_i)
                flux_sum += 0.125 * (x_a + x_b)

            capacity = rho_cp[i] * w_i
            if want_rates:
                rates[i] = total * inv_dx2[i] / capacity
            q = flux_sum * inv_dx2[i]
            if apply_now:
                out[i] = t_i + (dt * q) / capacity + dt * source[i]
            else:
                out[i] = q

    @njit(parallel=True, cache=True)
    def apply_cut_cell_flux_parallel(
        T: FloatArray,
        q: FloatArray,
        rho_cp: FloatArray,
        capacity_weights: FloatArray,
        source: FloatArray,
        dt: float,
        T_new: FloatArray,
    ) -> None:
        for i in prange(len(T)):
            w_i = capacity_weights[i]
            if w_i <= 0.0:
                T_new[i] = T[i]
            else:
                T_new[i] = T[i] + (dt * q[i]) / (rho_cp[i] * w_i) + dt * source[i]

    @njit(parallel=True, cache=True)
    def interpolate_curve_parallel(
        T: FloatArray,
        temp_points: FloatArray,
        values: FloatArray,
    ) -> FloatArray:
        n = len(T)
        m = len(temp_points)
        out = np.empty(n, dtype=np.float64)
        t_lo = temp_points[0]
        t_hi = temp_points[m - 1]
        for i in prange(n):
            t = T[i]
            if t <= t_lo:
                out[i] = values[0]
            elif t >= t_hi:
                out[i] = values[m - 1]
            else:
                j = 0
                for s in range(m - 1):
                    if t < temp_points[s + 1]:
                        j = s
                        break
                x0 = temp_points[j]
                inv_dx = 1.0 / (temp_points[j + 1] - x0)
                out[i] = (values[j + 1] - values[j]) * inv_dx * (t - x0) + values[j]
        return out

    @njit(inline="always")
    def _interp_scalar(t: float, xp: FloatArray, fp: FloatArray) -> float:
        m = len(xp)
        if t <= xp[0]:
            return fp[0]
        if t >= xp[m - 1]:
            return fp[m - 1]
        j = np.searchsorted(xp, t, side="right") - 1
        x0 = xp[j]
        return (fp[j + 1] - fp[j]) / (xp[j + 1] - x0) * (t - x0) + fp[j]

    @njit(inline="always")
    def _interp_scalar_from(t: float, xp: FloatArray, fp: FloatArray, hint: int) -> float:
        m = len(xp)
        if t <= xp[0]:
            return fp[0]
        if t >= xp[m - 1]:
            return fp[m - 1]
        j = hint
        if j < 0:
            j = 0
        elif j > m - 2:
            j = m - 2
        walked = 0
        while xp[j] > t and walked < 16:
            j -= 1
            walked += 1
        while xp[j + 1] <= t and walked < 16:
            j += 1
            walked += 1
        if xp[j] > t or xp[j + 1] <= t:
            j = np.searchsorted(xp, t, side="right") - 1
        x0 = xp[j]
        return (fp[j + 1] - fp[j]) / (xp[j + 1] - x0) * (t - x0) + fp[j]

    @njit(parallel=True, cache=True)
    def update_tabulated_parallel(
        T: FloatArray,
        indices: IntArray,
        last_T: FloatArray,
        temps: FloatArray,
        k_table: FloatArray,
        c_table: FloatArray,
        diff_vals: FloatArray,
        diff_start: int,
        rcp_vals: FloatArray,
        rcp_start: int,
        tolerance: float,
    ) -> None:
        for p in prange(len(indices)):
            t = T[indices[p]]
            if abs(t - last_T[p]) < tolerance:
                continue
            k = _interp_scalar(t, temps, k_table)
            c = _interp_scalar(t, temps, c_table)
            rcp_vals[rcp_start + p] = c
            diff_vals[diff_start + p] = k / c
            last_T[p] = t

    @njit(parallel=True, cache=True)
    def correct_tabulated_parallel(
        T_new: FloatArray,
        T_old: FloatArray,
        indices: IntArray,
        rcp_vals: FloatArray,
        rcp_start: int,
        fine_t: FloatArray,
        fine_h: FloatArray,
        tolerance: float,
    ) -> None:
        m = len(fine_t)
        t_first = fine_t[0]
        entries_per_kelvin = (m - 1) / (fine_t[m - 1] - t_first)
        for p in prange(len(indices)):
            i = indices[p]
            t_old = T_old[i]
            t_new = T_new[i]
            step = t_new - t_old
            if abs(step) < tolerance:
                continue
            at_old = (t_old - t_first) * entries_per_kelvin
            at_new = (t_new - t_first) * entries_per_kelvin
            hint_old = int(at_old) if 0.0 <= at_old < m else 0
            hint_new = int(at_new) if 0.0 <= at_new < m else 0
            h = _interp_scalar_from(t_old, fine_t, fine_h, hint_old) + step * rcp_vals[rcp_start + p]
            T_new[i] = _interp_scalar_from(h, fine_h, fine_t, hint_new)

    @njit(parallel=True, cache=True)
    def shift_enthalpy_parallel(
        temperatures: FloatArray,
        dh: FloatArray,
        fine_t: FloatArray,
        fine_h: FloatArray,
    ) -> FloatArray:
        n = len(temperatures)
        m = len(fine_t)
        out = np.empty(n, dtype=np.float64)
        t_first = fine_t[0]
        entries_per_kelvin = (m - 1) / (fine_t[m - 1] - t_first)
        for p in prange(n):
            t = temperatures[p]
            at = (t - t_first) * entries_per_kelvin
            hint = int(at) if 0.0 <= at < m else 0
            if hint > m - 2:
                hint = m - 2
            h = _interp_scalar_from(t, fine_t, fine_h, hint) + dh[p]
            entries = dh[p] / (fine_h[hint + 1] - fine_h[hint])
            if entries != entries:
                hint_back = hint
            elif entries > m:
                hint_back = m - 2
            elif entries < -m:
                hint_back = 0
            else:
                hint_back = hint + int(entries)
            out[p] = _interp_scalar_from(h, fine_h, fine_t, hint_back)
        return out

    @njit(cache=True, error_model="numpy")
    def interface_link_conductance_kernel(
        h: FloatArray,
        alpha: FloatArray,
        rho_cp: FloatArray,
        first: IntArray,
        second: IntArray,
        half_dx_first: FloatArray,
        half_dx_second: FloatArray,
        area: FloatArray,
    ) -> FloatArray:
        n = len(first)
        g = np.empty(n, dtype=np.float64)
        for p in range(n):
            h_p = h[p]
            if h_p > 0.0:
                i = first[p]
                j = second[p]
                k_first = alpha[i] * rho_cp[i]
                k_second = alpha[j] * rho_cp[j]
                if k_first < 1e-12:
                    k_first = 1e-12
                if k_second < 1e-12:
                    k_second = 1e-12
                if h_p < 1e-12:
                    h_p = 1e-12
                resistance = half_dx_first[p] / k_first + 1.0 / h_p + half_dx_second[p] / k_second
                g[p] = area[p] / resistance
            else:
                g[p] = 0.0
        return g

    @njit(cache=True, error_model="numpy")
    def interface_link_rates_kernel(
        g: FloatArray,
        first_local: IntArray,
        second_local: IntArray,
        link_cells: IntArray,
        cell_volume: FloatArray,
        rho_cp: FloatArray,
        n_voxels: int,
    ) -> FloatArray:
        m = len(link_cells)
        from_first = np.zeros(m, dtype=np.float64)
        from_second = np.zeros(m, dtype=np.float64)
        for p in range(len(g)):
            from_first[first_local[p]] += g[p]
            from_second[second_local[p]] += g[p]
        rates = np.zeros(n_voxels, dtype=np.float64)
        for c in range(m):
            cell = link_cells[c]
            capacity = rho_cp[cell] * cell_volume[c]
            if capacity > 0.0:
                rates[cell] = (from_first[c] + from_second[c]) / capacity
            else:
                rates[cell] = 0.0
        return rates

    @njit(cache=True, error_model="numpy")
    def apply_interface_links_kernel(
        T_new: FloatArray,
        T_old: FloatArray,
        dt: float,
        g: FloatArray,
        first: IntArray,
        second: IntArray,
        first_local: IntArray,
        second_local: IntArray,
        link_cells: IntArray,
        cell_volume: FloatArray,
        rho_cp: FloatArray,
    ) -> None:
        m = len(link_cells)
        into_second = np.zeros(m, dtype=np.float64)
        out_of_first = np.zeros(m, dtype=np.float64)
        for p in range(len(g)):
            q = g[p] * (T_old[first[p]] - T_old[second[p]])
            into_second[second_local[p]] += q
            out_of_first[first_local[p]] += q
        for c in range(m):
            cell = link_cells[c]
            capacity = rho_cp[cell] * cell_volume[c]
            if capacity > 0.0:
                T_new[cell] += dt * (into_second[c] - out_of_first[c]) / capacity
            else:
                T_new[cell] += 0.0

    @njit(parallel=True, cache=True)
    def interpolate_sorted_parallel(
        x: FloatArray,
        xp: FloatArray,
        fp: FloatArray,
    ) -> FloatArray:
        n = len(x)
        m = len(xp)
        out = np.empty(n, dtype=np.float64)
        lo_x = xp[0]
        hi_x = xp[m - 1]
        for i in prange(n):
            t = x[i]
            if t <= lo_x:
                out[i] = fp[0]
            elif t >= hi_x:
                out[i] = fp[m - 1]
            else:
                j = np.searchsorted(xp, t, side="right") - 1
                x0 = xp[j]
                out[i] = (fp[j + 1] - fp[j]) / (xp[j + 1] - x0) * (t - x0) + fp[j]
        return out

    @njit(parallel=True, cache=True)
    def cut_cell_conduction_rates_parallel(
        neighbors: IntArray,
        alpha: FloatArray,
        rho_cp: FloatArray,
        capacity_weights: FloatArray,
        conductance_factors: FloatArray,
        apertures: FloatArray,
        inv_dx2: FloatArray,
    ) -> FloatArray:
        n = len(alpha)
        max_slot = neighbors.shape[1]
        slot_face_end = min(6, max_slot)
        slot_corner_start = min(18, max_slot)
        slot_corner_end = min(26, max_slot)
        rates = np.zeros(n, dtype=np.float64)
        for i in prange(n):
            w_i = capacity_weights[i]
            if w_i <= 0.0:
                continue
            k_i = alpha[i] * rho_cp[i]
            f_i = conductance_factors[i]
            a_i = apertures[i]
            total = 0.0
            for j in range(slot_face_end):
                n_a = neighbors[i, j]
                if n_a >= 0:
                    k_a = alpha[n_a] * rho_cp[n_a]
                    total += 0.5 * min(f_i, conductance_factors[n_a]) * 0.5 * (a_i + apertures[n_a]) * (2.0 * k_i * k_a / (k_i + k_a + 1e-30))
            for j in range(slot_corner_start, slot_corner_end):
                n_a = neighbors[i, j]
                if n_a >= 0:
                    k_a = alpha[n_a] * rho_cp[n_a]
                    total += 0.125 * min(f_i, conductance_factors[n_a]) * 0.5 * (a_i + apertures[n_a]) * (2.0 * k_i * k_a / (k_i + k_a + 1e-30))
            rates[i] = total * inv_dx2[i] / (rho_cp[i] * w_i)
        return rates

    @njit(parallel=True, cache=True)
    def compute_gradient_magnitude_parallel(
        values: FloatArray,
        neighbors: IntArray,
        voxel_size: float,
    ) -> FloatArray:
        n = len(values)
        gradients = np.zeros(n, dtype=np.float32)
        inv_h = 1.0 / max(voxel_size, 1e-12)

        for i in prange(n):
            grad_sq = 0.0
            count = 0
            base = values[i]
            for j in range(neighbors.shape[1]):
                neighbor_idx = neighbors[i, j]
                if neighbor_idx >= 0:
                    dT = (values[neighbor_idx] - base) * inv_h
                    grad_sq += dT * dT
                    count += 1
            if count > 0:
                gradients[i] = np.sqrt(grad_sq / count)

        return gradients

_COST_DIFFUSIVITY_NS = 2.3
_COST_CURVE_NS = 1.7
_COST_DIFFUSIVITY_AND_CAPACITY_NS = 2.6
_COST_TABULATED_UPDATE_NS = 11.0
_COST_TABULATED_CORRECT_NS = 6.5
_COST_CONDUCTION_RATES_NS = 13.0
_COST_GRADIENT_NS = 8.0
_COST_SHIFT_ENTHALPY_NS = 14.0
_COST_CUT_CELL_FLUX_NS = 16.0
_COST_APPLY_FLUX_NS = 2.0

if NUMBA_AVAILABLE:
    from physics.kernel_dispatch import fused_kernels_enabled, runs_serial, serial_twin

    compute_diffusivity_from_curves_serial = serial_twin(compute_diffusivity_from_curves_parallel)
    compute_diffusivity_and_capacity_serial = serial_twin(compute_diffusivity_and_capacity_parallel)
    interpolate_curve_serial = serial_twin(interpolate_curve_parallel)
    update_tabulated_serial = serial_twin(update_tabulated_parallel)
    correct_tabulated_serial = serial_twin(correct_tabulated_parallel)
    cut_cell_conduction_rates_serial = serial_twin(cut_cell_conduction_rates_parallel)
    compute_gradient_magnitude_serial = serial_twin(compute_gradient_magnitude_parallel)
    shift_enthalpy_serial = serial_twin(shift_enthalpy_parallel)
    cut_cell_flux_serial = serial_twin(cut_cell_flux_parallel)
    apply_cut_cell_flux_serial = serial_twin(apply_cut_cell_flux_parallel)
