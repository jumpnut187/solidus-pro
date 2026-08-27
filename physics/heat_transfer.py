"""
Heat Transfer Module for Casting Solidification Simulation

Implements heat conduction and convection physics:
- Fourier's law of heat conduction
- Finite difference discretization
- Boundary conditions (mold interface, ambient)
- Temperature-dependent material properties

Mathematical Model:
    ∂T/∂t = α∇²T + Q/ρcₚ
    
where:
    T = temperature
    α = thermal diffusivity (k/ρcₚ)
    Q = heat source/sink (e.g., latent heat)
    ρ = density
    cₚ = specific heat

Usage:
    from physics.heat_transfer import HeatTransferModel, BoundaryType
    
    model = HeatTransferModel(mesh, alloy)
    model.add_boundary(surface_indices, BoundaryType.CONVECTION, h=10, T_inf=25)
    model.step(dt)
    temperatures = model.get_temperatures()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Union
import numpy as np
from numpy.typing import NDArray


# Type aliases
FloatArray = NDArray[Any]
IntArray = NDArray[Any]
BoolArray = NDArray[Any]


class BoundaryType(Enum):
    """Types of boundary conditions"""
    FIXED_TEMPERATURE = auto()  # Dirichlet: T = T_boundary
    HEAT_FLUX = auto()          # Neumann: -k∂T/∂n = q
    CONVECTION = auto()         # Robin: -k∂T/∂n = h(T - T_inf)
    ADIABATIC = auto()          # Neumann: ∂T/∂n = 0
    INTERFACE = auto()          # Mold-metal interface with IHTC


@dataclass
class BoundaryCondition:
    """
    Specification of a boundary condition.
    
    Attributes:
        type: Type of boundary condition
        indices: Voxel indices where this condition applies
        value: Condition-specific value (temperature, heat flux, etc.)
        htc: Heat transfer coefficient for convection/interface
        T_infinity: Ambient temperature for convection
        time_function: Optional time-dependent modifier
    """
    type: BoundaryType
    indices: IntArray
    value: float = 0.0
    htc: float = 0.0  # W/(m²·K)
    T_infinity: float = 25.0  # °C
    time_function: Optional[Callable[[float], float]] = None
    
    def get_effective_value(self, time: float) -> float:
        """Get value at given time (applies time function if present)"""
        if self.time_function is not None:
            return self.value * self.time_function(time)
        return self.value


@dataclass  
class TemperatureField:
    """
    Container for temperature field data.
    
    Provides convenient access to temperature statistics and gradients.
    """
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
        """
        Calculate temperature gradient magnitude at each voxel.
        
        Uses central difference approximation.
        """
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


class MaterialThermalProperties(Protocol):
    """Protocol for material thermal properties"""
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
    """
    Heat transfer model using finite differences.
    
    Solves the heat equation:
        ∂T/∂t = α∇²T + Q/(ρcₚ)
    
    Using explicit Euler method with adaptive time stepping.
    
    Features:
    - Multiple boundary condition types
    - Temperature-dependent properties (optional)
    - Latent heat release during solidification
    - Vectorized NumPy operations for performance
    
    Example:
        model = HeatTransferModel(mesh, alloy)
        model.set_initial_temperature(alloy.pouring_temp)
        model.add_boundary(
            surface_indices, 
            BoundaryType.CONVECTION,
            htc=500,  # W/(m²·K)
            T_infinity=200  # Mold temperature
        )
        
        while not model.is_solidified():
            model.step(dt)
            print(f"T_max = {model.temperature_field.max:.1f}°C")
    """
    
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
    ):
        """
        Initialize heat transfer model.
        
        Args:
            n_voxels: Number of voxels in mesh
            voxel_size: Voxel size in meters
            neighbors: Neighbor connectivity array (n_voxels x max_neighbors)
            alloy: Material thermal properties
            use_temperature_dependent: Use temperature-dependent k, cp
            voxel_sizes: Optional per-voxel size in meters (default: scalar
                ``voxel_size`` applied uniformly)
            diffusivity_overrides: Optional ``(indices, alpha_values)`` pair
                for per-voxel thermal diffusivity (e.g., chill regions)
            property_curves: Optional dict with keys "temp_points" (°C),
                "conductivity" (W/m·K), "specific_heat" (J/kg·K), "density"
                (kg/m³) for temperature-dependent properties
            rho_cp_overrides: Optional ``(indices, rho_cp_values)`` pair for
                per-voxel volumetric heat capacity (e.g., graphite chills)
        """
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
        # Contiguous float64 copies of the curves, cast once so the per-step
        # diffusivity evaluation does no per-call conversion.
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
        
        # Temperature field
        self._temperatures = np.zeros(n_voxels, dtype=np.float64)
        self._temperatures_new = np.zeros(n_voxels, dtype=np.float64)
        
        # Time tracking
        self.time = 0.0
        
        # Boundary conditions
        self._boundaries: List[BoundaryCondition] = []
        
        # Heat source (latent heat during solidification)
        self._heat_source = np.zeros(n_voxels, dtype=np.float64)
        
        # Precompute constants
        self._alpha = alloy.thermal_diffusivity  # m²/s
        self._dx2 = voxel_size ** 2
        # Per-voxel inverse squared voxel size (1/dx_i²). Stored as a scalar
        # for uniform grids and as an array when per-voxel sizes are supplied.
        if self._voxel_sizes is not None:
            self._inv_dx2 = 1.0 / (self._voxel_sizes ** 2)
        else:
            self._inv_dx2 = 1.0 / (voxel_size ** 2)
        # Always-array form for the Numba parallel Laplacian kernel.
        self._inv_dx2_arr = (
            self._inv_dx2
            if isinstance(self._inv_dx2, np.ndarray)
            else np.full(n_voxels, self._inv_dx2, dtype=np.float64)
        )
        self._numba_parallel_eligible = bool(NUMBA_AVAILABLE and not use_temperature_dependent)
        self._last_step_used_numba = False
        
        # Count neighbors for each voxel (kept for API compatibility; the
        # weighted 27-point Laplacian no longer normalizes by neighbor count).
        self._neighbor_counts = np.sum(neighbors >= 0, axis=1).astype(np.float64)

    @property
    def numba_parallel_eligible(self) -> bool:
        """Return True when the model can use Numba parallel kernels."""
        return self._numba_parallel_eligible

    @property
    def backend_description(self) -> str:
        """
        Human-readable name of the kernel backend used per step.

        With Numba present every ``_step_internal`` branch dispatches a
        parallel kernel: the uniform Laplacian/update pair when alpha is
        scalar, the conservative flux kernel when it is per-voxel.
        ``numba_parallel_eligible`` covers only the uniform branch, so it
        is the wrong flag to label the backend with.
        """
        return "Numba parallel kernels" if NUMBA_AVAILABLE else "NumPy/vectorized kernels"

    @property
    def last_step_used_numba(self) -> bool:
        """Return whether the most recent step used Numba parallel kernels."""
        return bool(self._last_step_used_numba)
    
    @property
    def temperatures(self) -> FloatArray:
        """Current temperature field"""
        return self._temperatures
    
    @property
    def temperature_field(self) -> TemperatureField:
        """Get temperature field object"""
        return TemperatureField(self._temperatures.copy(), self.time)
    
    def set_initial_temperature(self, temperature: Union[float, FloatArray]) -> None:
        """
        Set initial temperature field.
        
        Args:
            temperature: Uniform temperature or per-voxel array
        """
        if isinstance(temperature, (int, float)):
            self._temperatures.fill(temperature)
        else:
            self._temperatures[:] = temperature
    
    def add_boundary(
        self,
        indices: IntArray,
        boundary_type: BoundaryType,
        value: float = 0.0,
        htc: float = 0.0,
        T_infinity: float = 25.0,
        time_function: Optional[Callable[[float], float]] = None
    ) -> None:
        """
        Add a boundary condition.
        
        Args:
            indices: Voxel indices where condition applies
            boundary_type: Type of boundary condition
            value: Temperature (FIXED) or heat flux (HEAT_FLUX)
            htc: Heat transfer coefficient for CONVECTION/INTERFACE
            T_infinity: Ambient temperature for CONVECTION
            time_function: Optional time-dependent multiplier
        """
        bc = BoundaryCondition(
            type=boundary_type,
            indices=np.asarray(indices, dtype=np.int32),
            value=value,
            htc=htc,
            T_infinity=T_infinity,
            time_function=time_function
        )
        self._boundaries.append(bc)

    @property
    def boundaries(self) -> List["BoundaryCondition"]:
        """Read-only view of the registered boundary conditions."""
        return list(self._boundaries)

    def clear_boundaries(self, boundary_type: Optional[BoundaryType] = None) -> int:
        """Remove boundary conditions, optionally only those of one type.

        ``add_boundary`` could previously only append, so a boundary whose voxel
        set changes over time -- a mold-filling front, where cells become wetted
        as metal arrives -- had no way to be revised. Clear and re-add instead.

        Args:
            boundary_type: Only remove conditions of this type. ``None``
                removes all of them.

        Returns:
            How many conditions were removed.
        """
        before = len(self._boundaries)
        if boundary_type is None:
            self._boundaries.clear()
        else:
            self._boundaries = [
                bc for bc in self._boundaries if bc.type is not boundary_type
            ]
        return before - len(self._boundaries)

    def set_heat_source(self, source: FloatArray) -> None:
        """
        Set volumetric heat source term (W/m³).
        
        Typically used for latent heat release during solidification.
        """
        self._heat_source[:] = source
    
    def get_stable_timestep(self, alpha: Optional[FloatArray] = None) -> float:
        """
        Calculate maximum stable time step for explicit scheme.

        Stability criterion (3D, isotropic 27-point weighted stencil):
            dt < dx² / (4α)
        The stencil's spectral radius is bounded by
        (0.5 * 6 + 0.125 * 8) / dx² = 4 / dx², so 4α/dx² is the correct
        diffusion bound. Uses the smallest per-voxel size and the largest
        per-voxel diffusivity when those vary.

        Args:
            alpha: Optional diffusivity field already evaluated at the
                current temperatures, reused instead of recomputing it.
        """
        if isinstance(self._inv_dx2, np.ndarray):
            min_dx2 = 1.0 / float(np.max(self._inv_dx2))
        else:
            min_dx2 = 1.0 / float(self._inv_dx2)
        return 0.9 * min_dx2 / (4.0 * self._get_alpha_max(alpha))

    def _get_alpha_max(self, alpha: Optional[FloatArray] = None) -> float:
        """
        Return the maximum thermal diffusivity over all voxels.

        Uses the current temperature field when temperature-dependent
        properties are active and ALWAYS includes per-voxel diffusivity
        override values (e.g., graphite chills). Missing the overrides here
        would under-estimate the CFL diffusivity and let the explicit
        scheme run unstable at the chill interface (observed as +/-inf
        blow-up within a few hundred steps).

        Args:
            alpha: Optional diffusivity field already evaluated at the
                current temperatures, reused instead of recomputing it.
        """
        if self.use_temperature_dependent:
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
            # The conservative flux scheme bounds the pair conductivity by
            # k_ij <= 2*min(k_i, k_j), so the effective diffusivity of any
            # voxel is at most 2x its own alpha; reflect that in the CFL.
            rcp_idx, rcp_vals = self._rho_cp_overrides
            if len(rcp_idx) > 0:
                alpha_max = 2.0 * alpha_max
        if not np.isfinite(alpha_max) or alpha_max <= 0.0:
            return float(self._alpha)
        return alpha_max

    def get_gradient_magnitude(self, neighbors: IntArray, voxel_size: float) -> FloatArray:
        """Return gradient magnitude from the current thermal field without copying."""
        neighbor_arr = np.asarray(neighbors, dtype=np.int32)
        if NUMBA_AVAILABLE:
            return compute_gradient_magnitude_parallel(
                np.asarray(self._temperatures, dtype=np.float64),
                neighbor_arr,
                float(voxel_size),
            )

        values = np.asarray(self._temperatures, dtype=np.float64)
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
        """
        Advance temperature field by one time step.
        
        Uses explicit finite difference method.
        
        Args:
            dt: Time step in seconds
        """
        # The CFL bound and the first substep both need alpha at the current
        # field, so evaluate the property curves once and share the result.
        # The bound is deliberately NOT cached across steps: alpha rises as
        # the casting cools (k climbs toward room temperature while cp falls),
        # so a stale bound is stale in the unstable direction, and the
        # adaptive-dt path (CASTSIM_THERMAL_ADAPTIVE_DT) sets the outer step
        # near this bound, making the subdivision below load-bearing.
        alpha0 = (
            self._get_temperature_dependent_diffusivity(self._temperatures)
            if self.use_temperature_dependent
            else None
        )

        # Check stability
        dt_stable = (
            self.get_stable_timestep(alpha=alpha0)
            if alpha0 is not None
            else self.get_stable_timestep()
        )
        if dt > dt_stable:
            # Subdivide step
            n_substeps = int(np.ceil(dt / dt_stable))
            dt_sub = dt / n_substeps
            for substep in range(n_substeps):
                # Only the first substep sees the field alpha0 was built from;
                # later ones recompute against the updated temperatures.
                self._step_internal(dt_sub, alpha_precomputed=alpha0 if substep == 0 else None)
        else:
            self._step_internal(dt, alpha_precomputed=alpha0)

        self.time += dt

    def _step_internal(self, dt: float, alpha_precomputed: Optional[FloatArray] = None) -> None:
        """
        Internal time stepping (assumes dt is stable).

        Args:
            dt: Time step in seconds
            alpha_precomputed: Optional diffusivity field already evaluated at
                the current temperatures (see ``step``).
        """
        T = self._temperatures
        T_new = self._temperatures_new

        # Get thermal diffusivity (potentially temperature-dependent)
        if self.use_temperature_dependent:
            alpha = (
                alpha_precomputed
                if alpha_precomputed is not None
                else self._get_temperature_dependent_diffusivity(T)
            )
        else:
            alpha = self._alpha

        # Apply per-voxel diffusivity overrides (e.g., chill regions). This
        # mutates alpha in place, including a field handed down from step();
        # that array is a per-call local whose CFL maximum was already taken,
        # and _get_alpha_max folds the override values in separately, so the
        # mutation cannot leak into a later step or under-state the bound.
        if self._diffusivity_overrides is not None:
            if not isinstance(alpha, np.ndarray):
                alpha = np.full(self.n_voxels, alpha, dtype=np.float64)
            override_idx, override_vals = self._diffusivity_overrides
            alpha[override_idx] = override_vals

        use_numba_kernels = bool(
            self._numba_parallel_eligible
            and not isinstance(alpha, np.ndarray)
            and self._rho_cp_overrides is None
        )
        heterogeneous = isinstance(alpha, np.ndarray) or self._rho_cp_overrides is not None
        use_heterogeneous_numba = bool(NUMBA_AVAILABLE and heterogeneous)
        self._last_step_used_numba = use_numba_kernels or use_heterogeneous_numba

        if heterogeneous:
            # The conservative two-point flux branch below rebuilds the stencil
            # as a pair-conductivity sum; the uniform Laplacian is unused there.
            laplacian = None
        elif use_numba_kernels:
            laplacian = compute_laplacian_parallel(
                T,
                self.neighbors,
                self._inv_dx2_arr,
            )
        else:
            # Compute Laplacian using an isotropic 27-point weighted stencil.
            # Slot layout (set by core/voxel_mesh.py._canonicalize_neighbor_order):
            #   0-5   = face neighbors  [+X, -X, +Y, -Y, +Z, -Z]: weight 0.5
            #   6-17  = edge neighbors: weight 0.0 (excluded; they cancel
            #          exactly in the weighted sum)
            #   18-25 = corner neighbors: weight 0.125
            # laplacian[i] = Σ_present w_j * (T_nb_j - T_i) * (1/dx_i²)
            #
            # This weighted difference form is exact for all quadratic fields
            # (faces 0.5*6 + corners 0.125*24 = 6) with isotropic 4th-order
            # truncation error; it does NOT divide by neighbor count. Summing
            # in slot pairs (face pairs, then corner pairs) keeps the
            # accumulation order symmetric between geometrically symmetric
            # voxels and removes the per-step floating-point bias that would
            # otherwise compound over thousands of time steps.
            #
            # Some unit tests pass a smaller neighbour array (e.g. only the
            # 6 face slots); we clamp each loop range to whatever shape[1]
            # actually provides.
            laplacian = np.zeros(self.n_voxels, dtype=np.float64)
            max_slot = self.neighbors.shape[1]

            # Faces (weight 0.5), summed in symmetric pairs.
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

            # Edge neighbors (slots 6-17) carry weight 0.0 and are skipped.

            # Corners (weight 0.125), summed in symmetric pairs.
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
            # Handle a lone trailing corner slot for non-canonical shapes.
            if j < corner_end:
                n_j = self.neighbors[:, j]
                valid = n_j >= 0
                laplacian += 0.125 * np.where(valid, T[np.maximum(n_j, 0)] - T, 0.0)

            # Normalize by the per-voxel 1/dx_i² (no neighbor-count division).
            laplacian *= self._inv_dx2
        
        # Heat source term: Q / (ρ * cp)
        rho_cp = self._rho_cp_scalar
        source_term = self._heat_source / rho_cp

        if heterogeneous:
            # Conservative two-point flux update. The plain "T += dt*alpha*lap"
            # form is only conservative when the diffusivity is uniform: each
            # voxel applies its OWN alpha to the shared interface gradient, so
            # at a material interface (e.g. cast vs graphite chill) the high-
            # diffusivity side gains energy at k_hi/k_lo times the rate the
            # low side loses it (a numerical heat pump). The fix uses the
            # harmonic-mean conductivity of each voxel pair and each voxel's
            # own heat capacity:
            #   T_new[i] = T[i] + dt/(rho_cp_i*dx²) * Σ_j w_j*k_ij*(T_j - T_i)
            #   k_ij = 2*k_i*k_j/(k_i + k_j)
            # which conserves energy exactly for any material layout.
            rho_cp_arr = np.full(self.n_voxels, rho_cp, dtype=np.float64)
            if self._rho_cp_overrides is not None:
                rcp_idx, rcp_vals = self._rho_cp_overrides
                rho_cp_arr[rcp_idx] = rcp_vals
            if use_heterogeneous_numba:
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
                )
            else:
                # NumPy reference for the conservative two-point flux (used
                # when Numba is unavailable; must stay numerically equivalent
                # to ``update_heterogeneous_parallel``).
                max_slot = self.neighbors.shape[1]
                slot_pairs = [(0, 1), (2, 3), (4, 5)]
                corner_start = min(18, max_slot)
                corner_end = min(26, max_slot)
                k_arr = alpha * rho_cp_arr

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
                    k_ia = 2.0 * k0 * k_a / (k0 + k_a + 1e-30)
                    k_ib = 2.0 * k0 * k_b / (k0 + k_b + 1e-30)
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
                    k_ia = 2.0 * k0 * k_a / (k0 + k_a + 1e-30)
                    k_ib = 2.0 * k0 * k_b / (k0 + k_b + 1e-30)
                    dT_a = np.where(v_a, T[np.maximum(n_a, 0)] - T, 0.0)
                    dT_b = np.where(v_b, T[np.maximum(n_b, 0)] - T, 0.0)
                    flux += 0.125 * (k_ia * dT_a + k_ib * dT_b)
                    j += 2
                if j < corner_end:
                    n_j = self.neighbors[:, j]
                    valid = n_j >= 0
                    k_j = np.where(valid, k_arr[np.maximum(n_j, 0)], 0.0)
                    k_ij = 2.0 * k0 * k_j / (k0 + k_j + 1e-30)
                    dT_j = np.where(valid, T[np.maximum(n_j, 0)] - T, 0.0)
                    flux += 0.125 * (k_ij * dT_j)

                T_new[:] = T + dt * (flux * self._inv_dx2) / rho_cp_arr + dt * source_term
        else:
            # Update: T_new = T + dt * (α∇²T + Q/ρcp)
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
        
        # Apply boundary conditions
        self._apply_boundaries(T_new, T, dt)
        
        # Swap arrays
        self._temperatures, self._temperatures_new = self._temperatures_new, self._temperatures
    
    def _apply_boundaries(self, T_new: FloatArray, T_old: FloatArray, dt: float) -> None:
        """
        Apply all boundary conditions.

        Robin (convection/interface) fluxes are evaluated explicitly against
        the PRE-update field and applied to the post-update field:
            T_new[i] -= dt * h * (T_old[i] - T_inf) / (rho_cp * dx_i)
        The old treatment lumped the boundary cell as dx/2 (a factor-2 error)
        and evaluated the flux against the new temperature.
        """
        rho_cp = self._rho_cp_scalar
        rho_cp_arr = None
        if self._rho_cp_overrides is not None:
            rcp_idx, rcp_vals = self._rho_cp_overrides
            rho_cp_arr = np.full(self.n_voxels, rho_cp, dtype=np.float64)
            rho_cp_arr[rcp_idx] = rcp_vals
        for bc in self._boundaries:
            if bc.type == BoundaryType.FIXED_TEMPERATURE:
                # Dirichlet: T = T_boundary
                T_new[bc.indices] = bc.get_effective_value(self.time)
                
            elif bc.type == BoundaryType.CONVECTION:
                # Convective boundary: -k∂T/∂n = h(T - T_inf)
                h = bc.htc
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
                # Mold-metal interface with IHTC
                # Similar to convection but may have time-dependent htc
                h = bc.htc * (bc.time_function(self.time) if bc.time_function else 1.0)
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
                # Neumann: -k∂T/∂n = q, with n the outward normal, so a
                # positive q is a flux INTO the domain and heats the cell.
                #
                # Energy balance on a boundary cell of thickness dx_i:
                #     rho*cp*dx_i * dT/dt = q   [W/m^2]
                # giving the same explicit form as the Robin branches above --
                # evaluated against the pre-update field, and lumping the cell
                # as its full dx rather than dx/2.
                #
                # q is in W/m^2 and dx_i is in metres, matching the SI
                # convention of this module (see physics/units.py).
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
                # No-flux boundary: ∂T/∂n = 0
                # Nothing to do (handled by finite difference stencil)
                pass
    
    def _get_temperature_dependent_diffusivity(self, T: FloatArray) -> FloatArray:
        """
        Get temperature-dependent thermal diffusivity.

        When ``property_curves`` were supplied to the constructor, the
        diffusivity is computed vectorized from the conductivity, specific
        heat, and density curves:
            alpha(T) = k(T) / (rho(T) * cp(T))
        Otherwise fall back to a simple linear blend between solid and
        liquid values.
        """
        if self._curve_temp is not None:
            if NUMBA_AVAILABLE:
                # One parallel pass computing k, rho, cp and their quotient,
                # replacing three serial np.interp sweeps over the full field.
                return compute_diffusivity_from_curves_parallel(
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
        
        # Fraction liquid
        f_liquid = np.clip(
            (T - T_solidus) / (T_liquidus - T_solidus),
            0.0, 1.0
        )
        
        # Assume liquid diffusivity is lower
        alpha_solid = self._alpha
        alpha_liquid = self._alpha * 0.7
        
        return alpha_solid * (1 - f_liquid) + alpha_liquid * f_liquid
    
    def is_solidified(self, threshold_temp: Optional[float] = None) -> bool:
        """
        Check if all voxels are below solidus temperature.
        
        Args:
            threshold_temp: Temperature threshold (default: solidus)
        """
        threshold = threshold_temp or self.alloy.solidus_temp
        return np.all(self._temperatures <= threshold)
    
    def get_cooling_rate(self, prev_temperatures: FloatArray, dt: float) -> FloatArray:
        """
        Calculate cooling rate at each voxel.
        
        Args:
            prev_temperatures: Temperature field from previous step
            dt: Time step
            
        Returns:
            Cooling rate in °C/s
        """
        return (prev_temperatures - self._temperatures) / dt


# ============================================================================
# Numba-accelerated kernels (optional)
# ============================================================================

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
        """
        Evaluate alpha(T) = k(T) / (rho(T) * cp(T)) in one parallel pass.

        Replaces three ``np.interp`` sweeps (each strictly single-threaded,
        and together the dominant serial cost of a thermal step) with a
        single traversal that interpolates all three curves per voxel and
        materializes only the quotient.

        ``temp_points`` is a short ascending grid (six points in the
        production curves), so a forward scan beats a binary search. Edge
        behaviour matches ``np.interp``: values outside the grid clamp to the
        end points, and NaN propagates.
        """
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
                # Segment j with temp_points[j] <= t < temp_points[j+1]. A NaN
                # t matches no segment and falls through with j = 0, where the
                # arithmetic below carries the NaN into alpha as np.interp does.
                j = 0
                for s in range(m - 1):
                    if t < temp_points[s + 1]:
                        j = s
                        break
                x0 = temp_points[j]
                inv_dx = 1.0 / (temp_points[j + 1] - x0)
                offset = t - x0
                # Slope-then-offset form, matching np.interp's arithmetic.
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
        """
        Compute discrete Laplacian in parallel.

        Uses Numba for acceleration. Applies an isotropic 27-point weighted
        stencil:
            faces  (slots 0-5):  weight 0.5
            edges  (slots 6-17): weight 0.0 (excluded)
            corners(slots 18-25): weight 0.125
            laplacian[i] = Σ_present w_j * (T_nb_j - T_i) * inv_dx2[i]

        The weighted difference form is exact for all quadratic fields
        (faces 0.5*6 + corners 0.125*24 = 6) with isotropic 4th-order
        truncation error; it does NOT divide by neighbor count.

        Slots follow the canonical layout set by
        ``core/voxel_mesh.py._canonicalize_neighbor_order``: slots 0-5
        are face neighbors in [+X, -X, +Y, -Y, +Z, -Z] order, slots
        6-17 are edges, slots 18-25 are corners. Iterating the slots in
        fixed order keeps the float-summation order consistent between
        geometrically symmetric voxels.
        """
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

            # Edge neighbors (slots 6-17) carry weight 0.0 and are skipped.

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
        """
        Update temperature field in parallel.
        """
        n = len(T)
        T_new = np.empty(n, dtype=np.float64)
        
        for i in prange(n):
            T_new[i] = T[i] + dt * (alpha * laplacian[i] + source[i])
        
        return T_new

    @njit(parallel=True, fastmath=True, cache=True)
    def update_heterogeneous_parallel(
        T: FloatArray,
        neighbors: IntArray,
        alpha: FloatArray,
        rho_cp: FloatArray,
        inv_dx2: FloatArray,
        source: FloatArray,
        dt: float,
    ) -> FloatArray:
        """
        Conservative two-point flux update for heterogeneous material layouts.

        Replaces the NumPy branch of ``HeatTransferModel._step_internal`` used
        when per-voxel diffusivities (chill overrides or temperature-dependent
        properties) or per-voxel heat capacities are active:
            T_new[i] = T[i] + dt/(rho_cp_i * dx_i^2)
                       * sum_j w_j * k_ij * (T_j - T_i)
                       + dt * source[i]
            k_ij = 2 * k_i * k_j / (k_i + k_j)   (harmonic mean pair
                                                  conductivity)

        Slot weights match the canonical 27-point stencil: faces (slots 0-5)
        weight 0.5, edges (slots 6-17) weight 0.0 (excluded), corners (slots
        18-25) weight 0.125. Slots are iterated in symmetric pairs in the same
        fixed order as the NumPy reference so the per-voxel float accumulation
        order is identical.
        """
        n = len(T)
        max_slot = neighbors.shape[1]
        slot_face_end = min(6, max_slot)
        slot_corner_start = min(18, max_slot)
        slot_corner_end = min(26, max_slot)
        T_new = np.empty(n, dtype=np.float64)

        for i in prange(n):
            t_i = T[i]
            k_i = alpha[i] * rho_cp[i]
            flux_sum = 0.0

            # Faces (weight 0.5), summed in symmetric pairs (slots 0-5).
            for j in range(0, slot_face_end - 1, 2):
                x_a = 0.0
                n_a = neighbors[i, j]
                if n_a >= 0:
                    k_a = alpha[n_a] * rho_cp[n_a]
                    x_a = (2.0 * k_i * k_a) / (k_i + k_a + 1e-30) * (T[n_a] - t_i)
                x_b = 0.0
                n_b = neighbors[i, j + 1]
                if n_b >= 0:
                    k_b = alpha[n_b] * rho_cp[n_b]
                    x_b = (2.0 * k_i * k_b) / (k_i + k_b + 1e-30) * (T[n_b] - t_i)
                flux_sum += 0.5 * (x_a + x_b)

            # Edge neighbors (slots 6-17) carry weight 0.0 and are skipped.

            # Corners (weight 0.125), summed in symmetric pairs (slots 18-25).
            j = slot_corner_start
            while j < slot_corner_end - 1:
                x_a = 0.0
                n_a = neighbors[i, j]
                if n_a >= 0:
                    k_a = alpha[n_a] * rho_cp[n_a]
                    x_a = (2.0 * k_i * k_a) / (k_i + k_a + 1e-30) * (T[n_a] - t_i)
                x_b = 0.0
                n_b = neighbors[i, j + 1]
                if n_b >= 0:
                    k_b = alpha[n_b] * rho_cp[n_b]
                    x_b = (2.0 * k_i * k_b) / (k_i + k_b + 1e-30) * (T[n_b] - t_i)
                flux_sum += 0.125 * (x_a + x_b)
                j += 2
            # Lone trailing corner slot for non-canonical shapes.
            if j < slot_corner_end:
                n_a = neighbors[i, j]
                if n_a >= 0:
                    k_a = alpha[n_a] * rho_cp[n_a]
                    flux_sum += 0.125 * (
                        (2.0 * k_i * k_a) / (k_i + k_a + 1e-30) * (T[n_a] - t_i)
                    )

            T_new[i] = (
                t_i
                + (dt * (flux_sum * inv_dx2[i]) / rho_cp[i])
                + dt * source[i]
            )

        return T_new

    @njit(parallel=True, cache=True)
    def compute_gradient_magnitude_parallel(
        values: FloatArray,
        neighbors: IntArray,
        voxel_size: float,
    ) -> FloatArray:
        """Compute temperature gradient magnitude in parallel."""
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
