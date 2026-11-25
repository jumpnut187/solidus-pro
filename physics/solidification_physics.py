from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Tuple, List, Protocol
try:
    from typing import TypeAlias
except ImportError:
    try:
        from typing_extensions import TypeAlias
    except ImportError:
        from typing import Any
        TypeAlias = Any
import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]
BoolArray: TypeAlias = NDArray[np.bool_]

class PhaseState(Enum):
    LIQUID = auto()
    MUSHY = auto()
    SOLID = auto()

class SolidificationMethod(Enum):
    LEVER_RULE = auto()
    SCHEIL = auto()
    BACK_DIFFUSION = auto()
    EQUILIBRIUM = auto()

@dataclass
class SolidificationParameters:
    partition_coefficient: float = 0.14
    melting_point_solvent: Optional[float] = None
    eutectic_temperature: Optional[float] = None
    eutectic_fraction: float = 0.0
    back_diffusion_alpha: float = 0.0
    grain_refinement_factor: float = 1.0
    cooling_rate_exponent: float = -0.33

@dataclass
class SolidificationState:
    fraction_solid: FloatArray
    phase_state: List[PhaseState]
    latent_heat_rate: FloatArray
    solidification_time: FloatArray
    local_solidification_time: FloatArray

    def __init__(
        self,
        fraction_solid: FloatArray,
        phase_state: List[PhaseState],
        latent_heat_rate: FloatArray,
        solidification_time: FloatArray,
        local_solidification_time: FloatArray,
    ) -> None:
        self.fraction_solid = fraction_solid
        self.phase_state = phase_state
        self.latent_heat_rate = latent_heat_rate
        self.solidification_time = solidification_time
        self.local_solidification_time = local_solidification_time

    @property
    def is_fully_solid(self) -> BoolArray:
        return self.fraction_solid >= 0.99

    @property
    def is_liquid(self) -> BoolArray:
        return self.fraction_solid <= 0.01

    @property
    def solidification_complete(self) -> bool:
        return bool(np.all(self.fraction_solid >= 0.99))

class MaterialProperties(Protocol):
    liquidus_temp: float
    solidus_temp: float
    latent_heat: float
    density_liquid: float
    density_solid: float

class SolidificationModel:

    def __init__(
        self,
        n_voxels: int,
        alloy: MaterialProperties,
        method: SolidificationMethod = SolidificationMethod.LEVER_RULE,
        params: Optional[SolidificationParameters] = None
    ):
        self.n_voxels = n_voxels
        self.alloy = alloy
        self.method = method
        self.params = params or SolidificationParameters()

        if getattr(alloy, "melting_point_solvent", None) is not None:
            self.params.melting_point_solvent = alloy.melting_point_solvent
        if getattr(alloy, "partition_coefficient", None) is not None:
            self.params.partition_coefficient = alloy.partition_coefficient
        if getattr(alloy, "eutectic_temperature", None) is not None:
            self.params.eutectic_temperature = alloy.eutectic_temperature
        if getattr(alloy, "eutectic_fraction", None) is not None:
            self.params.eutectic_fraction = alloy.eutectic_fraction

        self._fraction_solid = np.zeros(n_voxels, dtype=np.float64)
        self._fraction_solid_prev = np.zeros(n_voxels, dtype=np.float64)

        self.time = 0.0
        self._solidification_start = np.full(n_voxels, np.inf)
        self._solidification_end = np.full(n_voxels, np.inf)
        self._last_dt = 0.0
        self._last_latent_heat_rate = np.zeros(n_voxels, dtype=np.float64)

        self.T_liq = alloy.liquidus_temp
        self.T_sol = alloy.solidus_temp
        self.dT = self.T_liq - self.T_sol

        self.L_vol = alloy.latent_heat * (alloy.density_liquid + alloy.density_solid) / 2

    @property
    def fraction_solid(self) -> FloatArray:
        return self._fraction_solid

    @property
    def solidification_time(self) -> FloatArray:
        return self._solidification_start

    @property
    def local_solidification_time(self) -> FloatArray:
        local_time = np.full(self.n_voxels, np.inf, dtype=np.float64)
        finite_mask = np.isfinite(self._solidification_end) & np.isfinite(self._solidification_start)
        local_time[finite_mask] = self._solidification_end[finite_mask] - self._solidification_start[finite_mask]
        return local_time

    def get_state(self) -> SolidificationState:
        phase_states = []
        for fs in self._fraction_solid:
            if fs < 0.01:
                phase_states.append(PhaseState.LIQUID)
            elif fs > 0.99:
                phase_states.append(PhaseState.SOLID)
            else:
                phase_states.append(PhaseState.MUSHY)

        return SolidificationState(
            self._fraction_solid.copy(),
            phase_states,
            self._last_latent_heat_rate.copy(),
            self._solidification_start.copy(),
            self.local_solidification_time.copy(),
        )

    def update(self, temperatures: FloatArray, dt: float) -> FloatArray:
        if NUMBA_AVAILABLE:
            return self._update_parallel(temperatures, dt)

        self._fraction_solid_prev[:] = self._fraction_solid

        if self.method == SolidificationMethod.LEVER_RULE:
            fs_new = self._lever_rule(temperatures)
        elif self.method == SolidificationMethod.SCHEIL:
            fs_new = self._scheil(temperatures)
        elif self.method == SolidificationMethod.BACK_DIFFUSION:
            fs_new = self._back_diffusion(temperatures)
        else:
            fs_new = self._lever_rule(temperatures)

        self._fraction_solid = np.maximum(self._fraction_solid, fs_new)

        self.time += dt

        starting = (self._fraction_solid_prev < 0.01) & (self._fraction_solid >= 0.01)
        ending = (self._fraction_solid_prev < 0.99) & (self._fraction_solid >= 0.99)
        single_step_crossing = starting & ending
        self._solidification_start[starting & ~single_step_crossing] = self.time
        self._solidification_start[single_step_crossing] = self.time - dt

        self._solidification_end[ending] = self.time

        dfs = self._fraction_solid - self._fraction_solid_prev

        dfs = np.maximum(dfs, 0)

        latent_heat_rate = self.L_vol * dfs / dt
        self._last_dt = dt
        self._last_latent_heat_rate[:] = latent_heat_rate

        return latent_heat_rate

    def can_finish_step_from_enthalpy(self) -> bool:
        return bool(NUMBA_AVAILABLE and self.method == SolidificationMethod.LEVER_RULE)

    def finish_step_from_enthalpy(
        self,
        temperatures: FloatArray,
        heat_capacity: tuple,
        alloy_T_liq: float,
        alloy_T_sol: float,
        latent_per_volume: float,
        dt: float,
    ) -> None:
        use_curve, curve_temp, capacity_curve, rho_cp_scalar, use_overrides, positions, values = heat_capacity
        self.time += dt
        kernel = (
            invert_enthalpy_and_commit_serial
            if runs_serial(len(self._fraction_solid), _COST_INVERT_AND_COMMIT_NS)
            else invert_enthalpy_and_commit_parallel
        )
        kernel(
            temperatures,
            self._fraction_solid,
            self._fraction_solid_prev,
            self._solidification_start,
            self._solidification_end,
            self._last_latent_heat_rate,
            float(alloy_T_liq),
            float(alloy_T_sol),
            float(latent_per_volume),
            use_curve,
            curve_temp,
            capacity_curve,
            float(rho_cp_scalar),
            use_overrides,
            positions,
            values,
            float(self.T_liq),
            float(self.dT),
            float(self.time),
            float(dt),
            float(self.L_vol),
        )
        self._last_dt = dt

    def _update_parallel(self, temperatures: FloatArray, dt: float) -> FloatArray:
        T = np.ascontiguousarray(temperatures, dtype=np.float64)
        use_lever = self.method == SolidificationMethod.LEVER_RULE

        if use_lever:
            fs_new = T
        elif self.method == SolidificationMethod.SCHEIL:
            fs_new = np.ascontiguousarray(self._scheil(temperatures), dtype=np.float64)
        elif self.method == SolidificationMethod.BACK_DIFFUSION:
            fs_new = np.ascontiguousarray(self._back_diffusion(temperatures), dtype=np.float64)
        else:
            fs_new = np.ascontiguousarray(self._lever_rule(temperatures), dtype=np.float64)

        self.time += dt
        commit_kernel = (
            commit_solidification_update_serial
            if runs_serial(len(T), _COST_COMMIT_NS)
            else commit_solidification_update_parallel
        )
        commit_kernel(
            fs_new,
            T,
            self._fraction_solid,
            self._fraction_solid_prev,
            self._solidification_start,
            self._solidification_end,
            self._last_latent_heat_rate,
            use_lever,
            float(self.T_liq),
            float(self.dT),
            float(self.time),
            float(dt),
            float(self.L_vol),
        )
        self._last_dt = dt
        return self._last_latent_heat_rate.copy()

    def _lever_rule(self, T: FloatArray) -> FloatArray:
        fs = (self.T_liq - T) / self.dT
        return np.clip(fs, 0.0, 1.0)

    def _scheil(self, T: FloatArray) -> FloatArray:
        k = self.params.partition_coefficient
        T_liq = self.T_liq
        T_sol = self.T_sol
        T_f = self.params.melting_point_solvent
        if abs(k - 1.0) < 1e-6 or k > 1.0:
            fs = self._lever_rule(T)
        elif T_f is None:
            theta = (T - T_sol) / max(T_liq - T_sol, 1e-12)
            theta = np.clip(theta, 0.0, 1.0)
            exponent = 1.0 / max(1.0 - k, 1e-6)
            fs = 1.0 - np.power(theta, exponent)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = (T_f - T) / max(T_f - T_liq, 1e-12)
                fs = 1.0 - np.power(ratio, 1.0 / (k - 1.0))
            fs = np.where(T >= T_liq, 0.0, fs)
            fs = np.where(T <= T_sol, 1.0, fs)
        e_frac = self.params.eutectic_fraction
        e_temp = self.params.eutectic_temperature
        if e_frac > 0.0 and e_temp is not None:
            fs = np.where(T <= e_temp, np.minimum(fs, 1.0 - e_frac), fs)
        return np.clip(fs, 0.0, 1.0)

    def _back_diffusion(self, T: FloatArray) -> FloatArray:
        alpha = self.params.back_diffusion_alpha

        fs_lever = self._lever_rule(T)
        fs_scheil = self._scheil(T)

        return alpha * fs_lever + (1 - alpha) * fs_scheil

    def get_phase_state(self) -> List[PhaseState]:
        states = []
        for fs in self._fraction_solid:
            if fs < 0.01:
                states.append(PhaseState.LIQUID)
            elif fs > 0.99:
                states.append(PhaseState.SOLID)
            else:
                states.append(PhaseState.MUSHY)
        return states

    def is_complete(self) -> bool:
        return bool(np.all(self._fraction_solid >= 0.99))

class FractionSolidCalculator:

    @staticmethod
    def lever_rule(T: float, T_liq: float, T_sol: float) -> float:
        if T >= T_liq:
            return 0.0
        elif T <= T_sol:
            return 1.0
        else:
            return (T_liq - T) / (T_liq - T_sol)

    @staticmethod
    def scheil(
        T: float,
        T_liq: float,
        T_sol: float,
        k: float,
        melting_point: Optional[float] = None,
        eutectic_temperature: Optional[float] = None,
        eutectic_fraction: float = 0.0,
    ) -> float:
        if T >= T_liq:
            return 0.0
        if abs(k - 1.0) < 1e-6 or k > 1.0:
            fs = FractionSolidCalculator.lever_rule(T, T_liq, T_sol)
        elif T <= T_sol:
            fs = 1.0
        elif melting_point is None:
            theta = (T - T_sol) / max(T_liq - T_sol, 1e-12)
            theta = min(max(theta, 0.0), 1.0)
            exponent = 1.0 / max(1.0 - k, 1e-6)
            fs = 1.0 - theta ** exponent
        else:
            denom = max(melting_point - T_liq, 1e-12)
            fs = 1.0 - ((melting_point - T) / denom) ** (1.0 / (k - 1.0))
        if (
            eutectic_fraction > 0.0
            and eutectic_temperature is not None
            and T <= eutectic_temperature
        ):
            fs = min(fs, 1.0 - eutectic_fraction)
        return float(min(max(fs, 0.0), 1.0))

    @staticmethod
    def equilibrium_path(
        T: FloatArray,
        T_liq: float,
        T_sol: float,
        phase_diagram_func: Optional[Callable[[FloatArray], FloatArray]] = None
    ) -> FloatArray:
        if phase_diagram_func is not None:
            return phase_diagram_func(T)
        else:
            fs = (T_liq - T) / (T_liq - T_sol)
            return np.clip(fs, 0.0, 1.0)

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def compute_fraction_solid_scheil_parallel(
        T: FloatArray,
        T_liq: float,
        T_sol: float,
        k: float,
        melting_point: Optional[float] = None,
        eutectic_temperature: Optional[float] = None,
        eutectic_fraction: float = 0.0,
    ) -> FloatArray:
        n = len(T)
        fs = np.empty(n, dtype=np.float64)
        dT = max(T_liq - T_sol, 1e-12)
        if abs(k - 1.0) < 1e-6 or k > 1.0:
            use_lever = True
        else:
            use_lever = False
            exp = 1.0 / max(1.0 - k, 1e-6)

        for i in prange(n):
            if T[i] >= T_liq:
                fs[i] = 0.0
            elif T[i] <= T_sol:
                fs[i] = 1.0
            elif use_lever:
                fs[i] = (T_liq - T[i]) / dT
            elif melting_point is None:
                theta = (T[i] - T_sol) / dT
                fs[i] = 1.0 - theta ** exp
            else:
                ratio = (melting_point - T[i]) / max(melting_point - T_liq, 1e-12)
                fs[i] = 1.0 - ratio ** (1.0 / (k - 1.0))
            if (
                eutectic_fraction > 0.0
                and eutectic_temperature is not None
                and T[i] <= eutectic_temperature
            ):
                cap = 1.0 - eutectic_fraction
                if fs[i] > cap:
                    fs[i] = cap

        return fs

    @njit(parallel=True, cache=True)
    def commit_solidification_update_parallel(
        fs_new: FloatArray,
        T: FloatArray,
        fs: FloatArray,
        fs_prev: FloatArray,
        sol_start: FloatArray,
        sol_end: FloatArray,
        latent_rate: FloatArray,
        use_lever: bool,
        T_liq: float,
        dT: float,
        time_now: float,
        dt: float,
        L_vol: float,
    ) -> None:
        n = len(fs)
        for i in prange(n):
            prev = fs[i]
            fs_prev[i] = prev

            if use_lever:
                candidate = (T_liq - T[i]) / dT
                if candidate < 0.0:
                    candidate = 0.0
                elif candidate > 1.0:
                    candidate = 1.0
            else:
                candidate = fs_new[i]

            f = prev if prev > candidate else candidate
            fs[i] = f

            starting = prev < 0.01 and f >= 0.01
            ending = prev < 0.99 and f >= 0.99
            if starting:
                sol_start[i] = time_now - dt if ending else time_now
            if ending:
                sol_end[i] = time_now

            dfs = f - prev
            if dfs < 0.0:
                dfs = 0.0
            latent_rate[i] = L_vol * dfs / dt

    @njit(parallel=True, cache=True)
    def invert_enthalpy_and_commit_parallel(
        T: FloatArray,
        fs: FloatArray,
        fs_prev: FloatArray,
        sol_start: FloatArray,
        sol_end: FloatArray,
        latent_rate: FloatArray,
        alloy_T_liq: float,
        alloy_T_sol: float,
        latent_per_volume: float,
        use_curve: bool,
        curve_temp: FloatArray,
        capacity_curve: FloatArray,
        rho_cp_scalar: float,
        use_overrides: bool,
        override_positions: IntArray,
        override_values: FloatArray,
        T_liq: float,
        dT: float,
        time_now: float,
        dt: float,
        L_vol: float,
    ) -> None:
        n = len(fs)
        m = len(curve_temp)
        inversion_dT = max(alloy_T_liq - alloy_T_sol, 1e-6)
        for i in prange(n):
            t = T[i]
            if use_overrides and override_positions[i] >= 0:
                rho_cp = override_values[override_positions[i]]
            elif use_curve:
                if t <= curve_temp[0]:
                    rho_cp = capacity_curve[0]
                elif t >= curve_temp[m - 1]:
                    rho_cp = capacity_curve[m - 1]
                else:
                    j = 0
                    for s in range(m - 1):
                        if t < curve_temp[s + 1]:
                            j = s
                            break
                    x0 = curve_temp[j]
                    inv_dx = 1.0 / (curve_temp[j + 1] - x0)
                    rho_cp = (capacity_curve[j + 1] - capacity_curve[j]) * inv_dx * (t - x0) + capacity_curve[j]
            else:
                rho_cp = rho_cp_scalar
            beta = latent_per_volume / rho_cp

            prev = fs[i]
            clipped = min(max(prev, 0.0), 1.0)
            t_p = alloy_T_liq - clipped * inversion_dT
            if t >= t_p:
                t_final = t
            else:
                h = t + beta * (1.0 - clipped)
                if h < alloy_T_sol:
                    t_final = h
                else:
                    t_final = (t * inversion_dT + beta * t_p) / (inversion_dT + beta)
            T[i] = t_final

            fs_prev[i] = prev
            candidate = (T_liq - t_final) / dT
            if candidate < 0.0:
                candidate = 0.0
            elif candidate > 1.0:
                candidate = 1.0
            f = prev if prev > candidate else candidate
            fs[i] = f
            starting = prev < 0.01 and f >= 0.01
            ending = prev < 0.99 and f >= 0.99
            if starting:
                sol_start[i] = time_now - dt if ending else time_now
            if ending:
                sol_end[i] = time_now
            dfs = f - prev
            if dfs < 0.0:
                dfs = 0.0
            latent_rate[i] = L_vol * dfs / dt

    @njit(parallel=True, cache=True)
    def apply_enthalpy_correction_parallel(
        T_post: FloatArray,
        fs_prev: FloatArray,
        T_out: FloatArray,
        T_liq: float,
        T_sol: float,
        dT: float,
        beta: float,
    ) -> None:
        n = len(T_post)
        liq_over_dT = T_liq / dT
        denom = 1.0 + beta / dT

        for i in prange(n):
            t_post = T_post[i]
            if not (t_post < T_liq):
                T_out[i] = t_post
                continue

            prev = fs_prev[i]
            if t_post <= T_sol:
                remaining = 1.0 - prev
                if remaining < 0.0:
                    remaining = 0.0
                elif remaining > 1.0:
                    remaining = 1.0
                recalesced = t_post + beta * remaining
                T_out[i] = recalesced if recalesced < T_sol else T_sol
                continue

            fs_max = (T_liq - t_post) / dT
            if fs_max < 0.0:
                fs_max = 0.0
            elif fs_max > 1.0:
                fs_max = 1.0

            if fs_max > prev:
                t_mush = (t_post + beta * (liq_over_dT - prev)) / denom
                if t_mush >= T_liq:
                    T_out[i] = t_post
                elif t_mush <= T_sol:
                    T_out[i] = t_post + beta * (1.0 - prev)
                else:
                    T_out[i] = t_mush
            else:
                T_out[i] = t_post

_COST_COMMIT_NS = 1.5
_COST_ENTHALPY_CORRECTION_NS = 1.0
_COST_INVERT_AND_COMMIT_NS = 4.0

if NUMBA_AVAILABLE:
    from physics.kernel_dispatch import runs_serial, serial_twin

    commit_solidification_update_serial = serial_twin(commit_solidification_update_parallel)
    apply_enthalpy_correction_serial = serial_twin(apply_enthalpy_correction_parallel)
    invert_enthalpy_and_commit_serial = serial_twin(invert_enthalpy_and_commit_parallel)

    def apply_enthalpy_correction(
        T_post: FloatArray,
        fs_prev: FloatArray,
        T_out: FloatArray,
        T_liq: float,
        T_sol: float,
        dT: float,
        beta: float,
    ) -> None:
        kernel = (
            apply_enthalpy_correction_serial
            if runs_serial(len(T_post), _COST_ENTHALPY_CORRECTION_NS)
            else apply_enthalpy_correction_parallel
        )
        kernel(T_post, fs_prev, T_out, T_liq, T_sol, dT, beta)
