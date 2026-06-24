"""
Solidification Physics Module for Casting Simulation

Implements phase transformation physics:
- Fraction solid evolution
- Latent heat release
- Scheil equation for microsegregation
- Mushy zone modeling

Key Equations:
    Fraction Solid (Linear):
        fₛ = (T_liquidus - T) / (T_liquidus - T_solidus)
    
    Scheil Equation:
        fₛ = 1 - [(T_liquidus - T) / (T_liquidus - T_melt)]^(1/(k-1))
    
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
        eutectic_fraction: Fraction solid at eutectic
        back_diffusion_alpha: Clyne-Kurz back-diffusion parameter
        grain_refinement_factor: Modification to dendrite arm spacing
        cooling_rate_exponent: Exponent for cooling rate effect on DAS
    """
    partition_coefficient: float = 0.14  # For Al-Si
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
        self._solidification_start[starting] = self.time
        
        # Track when solidification ends (fs reaches 1)
        ending = (self._fraction_solid_prev < 0.99) & (self._fraction_solid >= 0.99)
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
        Calculate fraction solid using a bounded linearized Scheil surrogate.

        This project does not carry a full phase-diagram/CALPHAD path, so the
        Scheil option uses a normalized binary-alloy surrogate anchored to the
        alloy liquidus and solidus. The formulation preserves the expected
        qualitative behavior without relying on clipping to hide invalid values:

            θ = (T - T_solidus) / (T_liquidus - T_solidus)
            f_s = 1 - θ^(1 / (1 - k))    for k < 1

        which gives:
        - f_s = 0 at liquidus
        - f_s = 1 at solidus
        - monotonic increase as temperature decreases
        """
        k = self.params.partition_coefficient
        T_liq = self.T_liq
        T_sol = self.T_sol
        if abs(k - 1.0) < 1e-6:
            fs = (T_liq - T) / max(T_liq - T_sol, 1e-12)
            return np.clip(fs, 0.0, 1.0)
        theta = (T - T_sol) / max(T_liq - T_sol, 1e-12)
        theta = np.clip(theta, 0.0, 1.0)
        if abs(k - 1.0) < 1e-6:
            exponent = 1.0
        elif k < 1.0:
            exponent = 1.0 / max(1.0 - k, 1e-6)
        else:
            exponent = max(k, 1.0)
        return 1.0 - np.power(theta, exponent)
    
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
    def scheil(T: float, T_liq: float, T_sol: float, k: float) -> float:
        """Scheil equation calculation"""
        if T >= T_liq:
            return 0.0
        if T <= T_sol:
            return 1.0
        if abs(k - 1.0) < 1e-6:
            return FractionSolidCalculator.lever_rule(T, T_liq, T_sol)
        theta = (T - T_sol) / max(T_liq - T_sol, 1e-12)
        theta = min(max(theta, 0.0), 1.0)
        if abs(k - 1.0) < 1e-6:
            exponent = 1.0
        elif k < 1.0:
            exponent = 1.0 / max(1.0 - k, 1e-6)
        else:
            exponent = max(k, 1.0)
        return float(1.0 - theta ** exponent)

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
        k: float
    ) -> FloatArray:
        """Parallel normalized Scheil surrogate calculation"""
        n = len(T)
        fs = np.empty(n, dtype=np.float64)
        dT = T_liq - T_sol
        if abs(k - 1.0) < 1e-6:
            exp = 1.0
        elif k < 1.0:
            exp = 1.0 / max(1.0 - k, 1e-6)
        else:
            exp = max(k, 1.0)
        
        for i in prange(n):
            if T[i] >= T_liq:
                fs[i] = 0.0
            elif T[i] <= T_sol:
                fs[i] = 1.0
            else:
                theta = (T[i] - T_sol) / dT
                fs[i] = 1.0 - theta ** exp
        
        return fs
    

