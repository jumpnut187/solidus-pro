"""
Solidification Physics Module for Casting Simulation

Implements phase transformation physics:
- Fraction solid evolution
- Latent heat release
- Scheil equation for microsegregation
- Mushy zone modeling

Key Equations:
    Fraction Solid (Linear / Lever Rule):
        fₛ = (T_liquidus - T) / (T_liquidus - T_solidus)

    Scheil Equation (microsegregation, k < 1):
        fₛ = 1 - ((T_f - T) / (T_f - T_L))^(1/(k-1))    for T < T_L, else 0

    where T_f is the melting point of the pure solvent metal and T_L the
    alloy liquidus. A linear lever-rule floor completes the model at the
    solidus (fₛ = 1 at T_S, since the raw Scheil curve leaves residual
    liquid there for k < 1); an optional eutectic reaction caps fₛ at
    (1 - f_eutectic) at/below the eutectic temperature.

    Latent Heat Release:
        Q = ρ * L * (∂fₛ/∂t)

Usage:
    from physics.solidification_physics import SolidificationModel, PhaseState
    
    model = SolidificationModel(mesh, alloy, method='lever_rule')
    model.update(temperatures, dt)
    fraction_solid = model.get_fraction_solid()
"""

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


# Type aliases
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]
BoolArray: TypeAlias = NDArray[np.bool_]


class PhaseState(Enum):
    """Phase state of a voxel"""
    LIQUID = auto()
    MUSHY = auto()  # In solidification range
    SOLID = auto()


class SolidificationMethod(Enum):
    """Method for calculating fraction solid"""
    LEVER_RULE = auto()       # Linear interpolation
    SCHEIL = auto()           # Scheil equation (microsegregation)
    BACK_DIFFUSION = auto()   # Partial back-diffusion
    EQUILIBRIUM = auto()      # Phase diagram lookup


@dataclass
class SolidificationParameters:
    """
    Parameters for solidification model.
    
    Attributes:
        partition_coefficient: Solute partition coefficient k = Cs/Cl
        melting_point_solvent: Melting point of the pure solvent metal
            (T_f, °C). Required by the true Scheil equation; when None the
            bounded liquidus/solidus surrogate is used instead.
        eutectic_temperature: Eutectic reaction temperature (°C); None if the
            alloy has no eutectic.
        eutectic_fraction: Fraction solid at which the eutectic reaction
            begins; 0 = no eutectic.
        back_diffusion_alpha: Clyne-Kurz back-diffusion parameter
        grain_refinement_factor: Modification to dendrite arm spacing
        cooling_rate_exponent: Exponent for cooling rate effect on DAS
    """
    partition_coefficient: float = 0.14  # For Al-Si
    melting_point_solvent: Optional[float] = None  # °C; None = use surrogate
    eutectic_temperature: Optional[float] = None  # °C
    eutectic_fraction: float = 0.0  # 0 = no eutectic
    back_diffusion_alpha: float = 0.0  # 0 = Scheil, 1 = lever rule
    grain_refinement_factor: float = 1.0
    cooling_rate_exponent: float = -0.33  # DAS ~ cooling_rate^n


@dataclass
class SolidificationState:
    """
    Container for solidification state at a point in time.
    """
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
        """Boolean array of fully solidified voxels"""
        return self.fraction_solid >= 0.99
    
    @property
    def is_liquid(self) -> BoolArray:
        """Boolean array of liquid voxels"""
        return self.fraction_solid <= 0.01
    
    @property
    def solidification_complete(self) -> bool:
        """Whether all voxels are solidified"""
        return bool(np.all(self.fraction_solid >= 0.99))


class MaterialProperties(Protocol):
    """Protocol for material solidification properties"""
    liquidus_temp: float
    solidus_temp: float
    latent_heat: float
    density_liquid: float
    density_solid: float


class SolidificationModel:
    """
    Solidification model tracking phase transformation.
    
    Computes fraction solid evolution and latent heat release
    for coupling with thermal solver.
    
    Features:
    - Multiple solidification calculation methods
    - Latent heat release calculation
    - Local solidification time tracking
    - Microsegregation modeling (Scheil)
    
    Example:
        model = SolidificationModel(
            n_voxels=mesh.n_active,
            alloy=alloy,
            method=SolidificationMethod.SCHEIL
        )
        
        # In time stepping loop:
        latent_heat = model.update(temperatures, dt)
        heat_model.set_heat_source(latent_heat)
    """
    
    def __init__(
        self,
        n_voxels: int,
        alloy: MaterialProperties,
        method: SolidificationMethod = SolidificationMethod.LEVER_RULE,
        params: Optional[SolidificationParameters] = None
    ):
        """
        Initialize solidification model.
        
        Args:
            n_voxels: Number of voxels
            alloy: Material properties
            method: Method for fraction solid calculation
            params: Additional solidification parameters
        """
        self.n_voxels = n_voxels
        self.alloy = alloy
        self.method = method
        self.params = params or SolidificationParameters()

        # Override Scheil parameters from the alloy when it provides them.
        # getattr with None defaults keeps the MaterialProperties protocol
        # (and any plain dummy alloy) unbroken.
        if getattr(alloy, "melting_point_solvent", None) is not None:
            self.params.melting_point_solvent = alloy.melting_point_solvent
        if getattr(alloy, "partition_coefficient", None) is not None:
            self.params.partition_coefficient = alloy.partition_coefficient
        if getattr(alloy, "eutectic_temperature", None) is not None:
            self.params.eutectic_temperature = alloy.eutectic_temperature
        if getattr(alloy, "eutectic_fraction", None) is not None:
            self.params.eutectic_fraction = alloy.eutectic_fraction
        
        # State arrays
        self._fraction_solid = np.zeros(n_voxels, dtype=np.float64)
        self._fraction_solid_prev = np.zeros(n_voxels, dtype=np.float64)
        
        # Time tracking
        self.time = 0.0
        self._solidification_start = np.full(n_voxels, np.inf)
        self._solidification_end = np.full(n_voxels, np.inf)
        self._last_dt = 0.0
        self._last_latent_heat_rate = np.zeros(n_voxels, dtype=np.float64)
        
        # Cache key temperatures
        self.T_liq = alloy.liquidus_temp
        self.T_sol = alloy.solidus_temp
        self.dT = self.T_liq - self.T_sol
        
        # Latent heat per unit volume
        self.L_vol = alloy.latent_heat * (alloy.density_liquid + alloy.density_solid) / 2
    
    @property
    def fraction_solid(self) -> FloatArray:
        """Current fraction solid field"""
        return self._fraction_solid
    
    @property
    def solidification_time(self) -> FloatArray:
        """Time when each voxel started solidifying"""
        return self._solidification_start
    
    @property
    def local_solidification_time(self) -> FloatArray:
        """
        Local solidification time (time from start to end of solidification).
        Returns inf for voxels still solidifying.
        """
        local_time = np.full(self.n_voxels, np.inf, dtype=np.float64)
        finite_mask = np.isfinite(self._solidification_end) & np.isfinite(self._solidification_start)
        local_time[finite_mask] = self._solidification_end[finite_mask] - self._solidification_start[finite_mask]
        return local_time
    
    def get_state(self) -> SolidificationState:
        """Get current solidification state"""
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
        """
        Update solidification state based on current temperatures.
        
        Args:
            temperatures: Current temperature field
            dt: Time step
            
        Returns:
            Volumetric latent heat release rate (W/m³)
        """
        # Save previous state
        self._fraction_solid_prev[:] = self._fraction_solid
        
        # Calculate new fraction solid
        if self.method == SolidificationMethod.LEVER_RULE:
            fs_new = self._lever_rule(temperatures)
        elif self.method == SolidificationMethod.SCHEIL:
            fs_new = self._scheil(temperatures)
        elif self.method == SolidificationMethod.BACK_DIFFUSION:
            fs_new = self._back_diffusion(temperatures)
        else:
            fs_new = self._lever_rule(temperatures)
        
        # Solidification is irreversible (no remelting for simplicity)
        # This could be relaxed for more advanced models
        self._fraction_solid = np.maximum(self._fraction_solid, fs_new)
        
        # Update time tracking
        self.time += dt
        
        # Track when solidification starts (fs first exceeds 0)
        starting = (self._fraction_solid_prev < 0.01) & (self._fraction_solid >= 0.01)
        ending = (self._fraction_solid_prev < 0.99) & (self._fraction_solid >= 0.99)
        # A voxel that jumps from liquid to (essentially) solid within a single
        # step would otherwise get a degenerate start == end interval
        # (local_solidification_time == 0.0), which downstream schedule
        # evaluation reads as "never solidified". Anchor such crossings at the
        # step beginning so the interval spans at least one step.
        single_step_crossing = starting & ending
        self._solidification_start[starting & ~single_step_crossing] = self.time
        self._solidification_start[single_step_crossing] = self.time - dt
        
        # Track when solidification ends (fs reaches 1)
        self._solidification_end[ending] = self.time
        
        # Calculate latent heat release
        dfs = self._fraction_solid - self._fraction_solid_prev
        
        # Only release heat for increasing fraction solid
        dfs = np.maximum(dfs, 0)
        
        # Volumetric heat release rate: Q = ρL * (dfs/dt)
        latent_heat_rate = self.L_vol * dfs / dt
        self._last_dt = dt
        self._last_latent_heat_rate[:] = latent_heat_rate
        
        return latent_heat_rate
    
    def _lever_rule(self, T: FloatArray) -> FloatArray:
        """
        Calculate fraction solid using lever rule (linear).
        
        fₛ = (T_liquidus - T) / (T_liquidus - T_solidus)
        """
        fs = (self.T_liq - T) / self.dT
        return np.clip(fs, 0.0, 1.0)
    
    def _scheil(self, T: FloatArray) -> FloatArray:
        """
        Calculate fraction solid with the Scheil equation.

        For k < 1 with the solvent melting point (T_f) known, the true Scheil
        microsegregation curve is used:

            f_s = 1 - ((T_f - T) / (T_f - T_liq)) ** (1 / (k - 1))   for T < T_liq

        The raw curve leaves residual liquid at the solidus (e.g.
        f_s(T_sol) ~ 0.50 for WCB, k = 0.17), so a linear lever-rule floor
        completes the model there: f_s = max(scheil_raw, lever_rule) pins the
        fraction to 1.0 at/below T_sol while the mushy-zone values keep the
        analytic Scheil shape. An optional eutectic reaction caps f_s at
        (1 - f_eutectic) at/below the eutectic temperature.

        Falls back to the bounded liquidus/solidus surrogate when T_f is
        unavailable; uses the lever rule for k >= 1. The eutectic cap is
        applied last so it also bounds the completion value at/below T_sol.
        """
        k = self.params.partition_coefficient
        T_liq = self.T_liq
        T_sol = self.T_sol
        T_f = self.params.melting_point_solvent
        if abs(k - 1.0) < 1e-6 or k > 1.0:
            fs = self._lever_rule(T)
        elif T_f is None:
            # Bounded surrogate anchored to liquidus/solidus (no T_f known).
            theta = (T - T_sol) / max(T_liq - T_sol, 1e-12)
            theta = np.clip(theta, 0.0, 1.0)
            exponent = 1.0 / max(1.0 - k, 1e-6)
            fs = 1.0 - np.power(theta, exponent)
        else:
            # True Scheil equation (k < 1, T_f known).
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = (T_f - T) / max(T_f - T_liq, 1e-12)
                fs = 1.0 - np.power(ratio, 1.0 / (k - 1.0))
            # Liquidus/solidus bounds. The raw curve reaches 0 at T_liq;
            # below T_sol it would leave residual liquid, so the lever-rule
            # floor (1.0 at/below T_sol) completes the model.
            fs = np.where(T >= T_liq, 0.0, fs)
            fs = np.where(T <= T_sol, 1.0, fs)
        # Eutectic cap: remaining liquid freezes isothermally at the eutectic.
        e_frac = self.params.eutectic_fraction
        e_temp = self.params.eutectic_temperature
        if e_frac > 0.0 and e_temp is not None:
            fs = np.where(T <= e_temp, np.minimum(fs, 1.0 - e_frac), fs)
        # Defensive clamp only; the formula is naturally bounded below T_liq.
        return np.clip(fs, 0.0, 1.0)
    
    def _back_diffusion(self, T: FloatArray) -> FloatArray:
        """
        Calculate fraction solid with back-diffusion (Clyne-Kurz).
        
        Interpolates between Scheil (α=0) and lever rule (α=1).
        """
        alpha = self.params.back_diffusion_alpha
        
        fs_lever = self._lever_rule(T)
        fs_scheil = self._scheil(T)
        
        return alpha * fs_lever + (1 - alpha) * fs_scheil
    
    def get_phase_state(self) -> List[PhaseState]:
        """Get phase state for each voxel"""
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
        """Check if all voxels are fully solid"""
        return bool(np.all(self._fraction_solid >= 0.99))


class FractionSolidCalculator:
    """
    Utility class for fraction solid calculations.
    
    Provides static methods for various fraction solid models
    without requiring full model setup.
    """
    
    @staticmethod
    def lever_rule(T: float, T_liq: float, T_sol: float) -> float:
        """Simple lever rule calculation"""
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
        """
        Scheil equation calculation.

        For k < 1 with the solvent melting point (T_f) known, the true Scheil
        microsegregation curve is used:

            f_s = 1 - ((T_f - T) / (T_f - T_liq)) ** (1 / (k - 1))

        The raw curve leaves residual liquid at the solidus, so a linear
        lever-rule floor completes the model there (f_s = 1 at/below T_sol).
        An optional eutectic reaction caps f_s at (1 - eutectic_fraction)
        at/below the eutectic temperature.

        Falls back to the bounded liquidus/solidus surrogate when
        ``melting_point`` is None; uses the lever rule for k >= 1. The
        eutectic cap is applied last so it also bounds the completion value
        at/below T_sol.
        """
        if T >= T_liq:
            return 0.0
        if abs(k - 1.0) < 1e-6 or k > 1.0:
            fs = FractionSolidCalculator.lever_rule(T, T_liq, T_sol)
        elif T <= T_sol:
            # Solidus completion: the raw Scheil curve leaves residual liquid
            # here (e.g. f_s(T_sol) ~ 0.50 for WCB, k = 0.17), so the
            # lever-rule floor (1.0 at/below T_sol) completes the model.
            fs = 1.0
        elif melting_point is None:
            # Bounded surrogate anchored to liquidus/solidus (no T_f known).
            theta = (T - T_sol) / max(T_liq - T_sol, 1e-12)
            theta = min(max(theta, 0.0), 1.0)
            exponent = 1.0 / max(1.0 - k, 1e-6)
            fs = 1.0 - theta ** exponent
        else:
            # True Scheil equation (k < 1, T_f known).
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
        """
        Calculate fraction solid from phase diagram.
        
        Args:
            T: Temperature array
            T_liq: Liquidus temperature
            T_sol: Solidus temperature  
            phase_diagram_func: Optional function f(T) -> fs from phase diagram
        """
        if phase_diagram_func is not None:
            return phase_diagram_func(T)
        else:
            # Default to lever rule
            fs = (T_liq - T) / (T_liq - T_sol)
            return np.clip(fs, 0.0, 1.0)


# ============================================================================
# Numba-accelerated functions
# ============================================================================

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
        """Parallel Scheil equation calculation.

        Mirrors FractionSolidCalculator.scheil element-wise: true Scheil for
        k < 1 with the solvent melting point known, bounded liquidus/solidus
        surrogate fallback when it is None, lever rule for k >= 1, lever-rule
        floor completing the model at the solidus, and an optional eutectic cap.
        """
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
                # Bounded surrogate anchored to liquidus/solidus (no T_f).
                theta = (T[i] - T_sol) / dT
                fs[i] = 1.0 - theta ** exp
            else:
                # True Scheil equation (k < 1, T_f known).
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
    

