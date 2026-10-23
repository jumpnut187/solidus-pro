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
        capacity_weights: Optional[np.ndarray] = None,
        boundary_areas: Optional[np.ndarray] = None,
        slave_cells: Optional[tuple[np.ndarray, np.ndarray]] = None,
        cut_cell_apertures: bool = True,
        contact_conductance: Optional[tuple[np.ndarray, float]] = None,
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
            capacity_weights: Optional (N,) share of each cell's dx_i^3 that
                holds material (cut cells). Heat capacity becomes
                rho_cp * w_i * dx_i^3. A weight of 0 marks a slaved cell
                (see ``slave_cells``). A link i-j carries
                min(f_i, f_j) * (a_i + a_j) / 2 (``cut_cell_conductance_factors``,
                ``cut_cell_link_apertures``): symmetric, so conservative, and
                no cell's explicit rate exceeds the full-cell rate. With
                property curves, capacity is rho(T)*cp(T) from the curves.
            boundary_areas: Optional (N,) boundary area per cell in m^2.
                Robin and heat-flux terms then use A_i / (w_i * dx_i^3)
                instead of 1/dx_i.
            cut_cell_apertures: Scale cut-cell links by the open-face
                estimate (a_i + a_j) / 2 with a = clip(w, 0, 1)
                (``cut_cell_link_apertures``). False keeps full links.
            slave_cells: Optional ``(slaves, masters)`` index pair. After
                each step a slave copies its master's temperature; the
                caller has moved the slave's capacity and area to the master.
            contact_conductance: Optional ``(indices, h)`` pair: the cells of
                a contacting body (e.g. a sand core) and the interface heat
                transfer coefficient h in W/(m^2 K). Every link with exactly
                one end in ``indices`` carries h in series with the pair
                conductivity: k_ij -> k_ij / (1 + k_ij / (h * dx)), dx the
                contacting cell's size, so the interface conductance per area
                is 1 / (dx / k_ij + 1 / h). Links only lose conductance, so the
                stable step is unchanged.
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

        # Cut cells (all None when not supplied: the full-cell code paths
        # below stay exactly as they were).
        self._capacity_weights: Optional[FloatArray] = None
        self._conductance_factors: Optional[FloatArray] = None
        self._link_apertures: Optional[FloatArray] = None
        self._boundary_area_per_volume: Optional[FloatArray] = None
        self._slave_cells: Optional[tuple[IntArray, IntArray]] = None
        # rho*cp at the curve points (J/m3/K). Cut cells with property curves
        # use rho(T)*cp(T) for capacity, Robin and latent beta.
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
        
        # h * dx_i on contacting cells, 0 elsewhere (``contact_conductance``).
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
        if self._capacity_weights is not None:
            return self._cut_cell_stable_timestep(alpha)
        return 0.9 * min_dx2 / (4.0 * self._get_alpha_max(alpha))

    def volumetric_heat_capacity(self, temperatures: Optional[FloatArray] = None) -> FloatArray:
        """(N,) rho*cp in J/m3/K at ``temperatures`` (default: the current field).

        Cut cells with property curves: rho(T)*cp(T) from the curves, chill
        overrides applied. Otherwise the scalar rho_l*cp with the overrides.
        """
        T = self._temperatures if temperatures is None else np.asarray(temperatures, dtype=np.float64)
        if self._capacity_curve is not None:
            if NUMBA_AVAILABLE:
                rho_cp = interpolate_curve_parallel(
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

    def _cut_cell_stable_timestep(self, alpha: Optional[FloatArray] = None) -> float:
        """0.9 / max_i(conduction rate_i + Robin rate_i), the exact per-cell explicit bound.

        rate_i = sum_j w_slot * min(f_i, f_j) * k_ij / (rho_cp_i * w_i * dx_i^2)
               + h(t) * A_i / (rho_cp_i * w_i * dx_i^3)
        with rho_cp_i at the current field (the minimum capacity governs).
        """
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
        rho_cp = self.volumetric_heat_capacity(T)
        rates = (cut_cell_conduction_rates_parallel if NUMBA_AVAILABLE else cut_cell_conduction_rates_numpy)(
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
                if bc.type not in (BoundaryType.CONVECTION, BoundaryType.INTERFACE) or len(bc.indices) == 0:
                    continue
                h = bc.htc
                if bc.type == BoundaryType.INTERFACE and bc.time_function is not None:
                    h = bc.htc * bc.time_function(self.time)
                idx = bc.indices
                rates[idx] += float(h) * apv[idx] / rho_cp[idx]
        max_rate = float(np.max(rates)) if len(rates) else 0.0
        if not np.isfinite(max_rate) or max_rate <= 0.0:
            min_dx2 = 1.0 / float(np.max(self._inv_dx2_arr))
            return 0.9 * min_dx2 / (4.0 * float(self._alpha))
        return 0.9 / max_rate

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

        if self._slave_cells is not None and len(self._slave_cells[0]) > 0:
            slaves, masters = self._slave_cells
            self._temperatures[slaves] = self._temperatures[masters]

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

        cut_cells = self._capacity_weights is not None
        use_numba_kernels = bool(
            self._numba_parallel_eligible
            and not isinstance(alpha, np.ndarray)
            and self._rho_cp_overrides is None
            and not cut_cells
            and not self._has_contact
        )
        heterogeneous = (
            isinstance(alpha, np.ndarray)
            or self._rho_cp_overrides is not None
            or cut_cells
            or self._has_contact
        )
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
            if cut_cells:
                rho_cp_arr = self.volumetric_heat_capacity(T)
                self._step_rho_cp = rho_cp_arr
                alpha_arr = (
                    alpha
                    if isinstance(alpha, np.ndarray)
                    else np.full(self.n_voxels, alpha, dtype=np.float64)
                )
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
                # NumPy reference for the conservative two-point flux (used
                # when Numba is unavailable; must stay numerically equivalent
                # to ``update_heterogeneous_parallel``).
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
        if self._boundary_area_per_volume is not None:
            step_rho_cp = self._step_rho_cp
            self._apply_cut_cell_boundaries(
                T_new, T_old, dt, step_rho_cp if step_rho_cp is not None else rho_cp_arr
            )
            return
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

    def _apply_cut_cell_boundaries(
        self,
        T_new: FloatArray,
        T_old: FloatArray,
        dt: float,
        rho_cp_arr: Optional[FloatArray],
    ) -> None:
        """``_apply_boundaries`` with area per volume A_i/(w_i dx_i^3) in place of 1/dx_i.

        Same explicit form, evaluated against the pre-update field:
            T_new[i] -= dt * h * (T_old[i] - T_inf) * A_i / (rho_cp_i * w_i * dx_i^3)
        A cell with no boundary area (or a slaved cell) gets no flux even
        when it is listed in a boundary set.
        """
        apv = self._boundary_area_per_volume
        rho_cp = self._rho_cp_scalar
        for bc in self._boundaries:
            idx = bc.indices
            rc_i = rho_cp_arr[idx] if rho_cp_arr is not None else rho_cp
            if bc.type == BoundaryType.FIXED_TEMPERATURE:
                T_new[idx] = bc.get_effective_value(self.time)
            elif bc.type == BoundaryType.CONVECTION:
                T_new[idx] -= dt * bc.htc * (T_old[idx] - bc.T_infinity) * apv[idx] / rc_i
            elif bc.type == BoundaryType.INTERFACE:
                h = bc.htc * (bc.time_function(self.time) if bc.time_function else 1.0)
                T_new[idx] -= dt * h * (T_old[idx] - bc.T_infinity) * apv[idx] / rc_i
            elif bc.type == BoundaryType.HEAT_FLUX:
                T_new[idx] += dt * bc.get_effective_value(self.time) * apv[idx] / rc_i
    
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


def _stencil_slot_weights(max_slot: int) -> np.ndarray:
    """Per-slot weights of the 27-point stencil: faces 0.5, edges 0, corners 0.125."""
    weights = np.zeros(max_slot, dtype=np.float64)
    weights[: min(6, max_slot)] = 0.5
    weights[min(18, max_slot) : min(26, max_slot)] = 0.125
    return weights


#: Cut-cell link scaling lets a cell's explicit rate reach this multiple of
#: the full-cell rate 4*alpha/dx^2 before its links are scaled down. 1.0 keeps
#: the flag-off time step; larger values keep more of the conduction between
#: a cut cell and its neighbours at the price of a smaller time step.
CUT_CELL_RATE_ALLOWANCE = 1.0


def cut_cell_link_apertures(capacity_weights: FloatArray) -> FloatArray:
    """a_i = clip(w_i, 0, 1). A link carries the open-face estimate (a_i + a_j) / 2.

    A cheap stand-in for FAVOR face apertures: the face between a cut cell
    and a full cell is estimated half open, the face between two equal cut
    cells as open as the cells are full.
    """
    return np.clip(np.asarray(capacity_weights, dtype=np.float64), 0.0, 1.0)


def cut_cell_conductance_factors(
    capacity_weights: FloatArray,
    neighbors: IntArray,
    apertures: Optional[FloatArray] = None,
    rate_allowance: float = CUT_CELL_RATE_ALLOWANCE,
) -> FloatArray:
    """f_i = min(1, allowance * S_full * w_i / S_i): link scale that caps cell i's explicit rate.

    S_i = sum_j w_slot_j * (a_i + a_j) / 2 over the neighbours cell i has
    (a = 1 without apertures), S_full = 4 for the 26-slot stencil. With
    pair scale s_ij = min(f_i, f_j) * (a_i + a_j) / 2 the explicit rate of
    cell i, alpha * sum_j w_slot_j * s_ij / (w_i dx^2), stays at or below
    allowance * alpha * S_full / dx^2, so allowance 1 keeps every cell
    stable at the full-cell step. A cut cell at a wall has lost neighbours
    and open face, so it needs little or no scaling. Cells with w_i = 0
    (slaves) get f_i = 0 and lose every link.
    """
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
    """Pair conductivity with a contact resistance on links that cross a contact.

    ``g`` is h * dx (W/(m K)) on the cells of a contacting body and 0
    elsewhere. A link with exactly one end in the body becomes
    k / (1 + k / g); every other link keeps k.
    """
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
    """NumPy reference for ``update_heterogeneous_cut_cell_parallel``.

    The conservative two-point flux update with cut cells:
        T_new[i] = T[i] + dt / (rho_cp_i * w_i * dx_i^2)
                   * sum_j w_slot_j * min(f_i, f_j) * k_ij * (T_j - T_i)
                   + dt * source[i]
    Cells with w_i = 0 keep T[i]. Slots are summed in the same symmetric
    pairs as ``update_heterogeneous_parallel``. ``contact_hdx`` adds the
    contact resistance of :func:`contact_limited_conductivity`.
    """
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
    """(N,) explicit conduction rate per cell, 1/s (NumPy reference of the Numba kernel).

    rate_i = sum_j w_slot_j * min(f_i, f_j) * k_ij * inv_dx2_i / (rho_cp_i * w_i),
    the diagonal of the cut-cell update; 0 for cells with w_i = 0. A step
    dt <= 1 / max(rate) keeps every cell a convex combination of its
    neighbours.
    """
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

    @njit(inline="always")
    def _link_conductivity(k_i: float, k_j: float, g_i: float, g_j: float) -> float:
        """Harmonic-mean pair conductivity, with ``contact_limited_conductivity`` applied."""
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
        order is identical. ``contact_hdx`` adds the contact resistance of
        ``contact_limited_conductivity``.
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
            g_i = contact_hdx[i]
            flux_sum = 0.0

            # Faces (weight 0.5), summed in symmetric pairs (slots 0-5).
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

            # Edge neighbors (slots 6-17) carry weight 0.0 and are skipped.

            # Corners (weight 0.125), summed in symmetric pairs (slots 18-25).
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
            # Lone trailing corner slot for non-canonical shapes.
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
        """Cut-cell form of ``update_heterogeneous_parallel``.

            T_new[i] = T[i] + dt / (rho_cp_i * w_i * dx_i^2)
                       * sum_j w_slot_j * min(f_i, f_j) * k_ij * (T_j - T_i)
                       + dt * source[i]

        ``capacity_weights`` w_i scale the heat capacity, and
        ``conductance_factors`` f_i scale each link (pair value min(f_i, f_j),
        so the flux stays antisymmetric). Cells with w_i = 0 keep T[i].
        ``contact_hdx`` adds the contact resistance of
        ``contact_limited_conductivity``.
        NumPy reference: ``update_heterogeneous_cut_cell_numpy``.
        """
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
    def interpolate_curve_parallel(
        T: FloatArray,
        temp_points: FloatArray,
        values: FloatArray,
    ) -> FloatArray:
        """``np.interp(T, temp_points, values)`` in one parallel pass (same forward scan as the diffusivity kernel)."""
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
        """Numba form of ``cut_cell_conduction_rates_numpy``."""
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
