import numpy as np
import os
import time
import warnings
from typing import Any, Callable, Tuple, Dict, Optional, List
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from core.solidification_params import (
    RISER_SLEEVE_EXPONENT,
    SAND_MATERIAL_CUSTOM,
    SHRINK_SCALAR_DEFAULT,
    SHRINK_SCALAR_MAX,
    SHRINK_SCALAR_MIN,
    SolidificationParameters,
)
from dataclasses import dataclass, field
from physics.heat_transfer import BoundaryType, HeatTransferModel, InterfaceLinks
from physics.units import GRAVITY_M_S2, inches_to_meters, inches_to_mm
from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_RISER,
    PART_TYPE_RISER_POST,
    SAND_PART_CODES,
    is_feeder,
    is_flow_path,
    is_gating_metal,
    is_metal,
    is_mold,
    is_scored_metal,
)
from physics.solidification_physics import SolidificationModel as ThermalSolidificationModel
from physics.riser_pipe import gravity_drain_pipe, label_unfed_pools
from core.acceleration import (
    update_solidification_with_part_types_parallel,
    update_solidification_with_feeding_connectivity,
    compute_min_distances_parallel,
    compute_modulus_variance_parallel,
    compute_constraint_strain_parallel,
    compute_chill_internal_thickness_parallel,
    compute_dynamic_chill_boost_parallel,
    compute_mold_air_gap_effect_parallel,
    compute_riser_influence_parallel,
    compute_solid_path_strain_parallel,
    compute_isolation_score_parallel,
)
from core.materials import AlloyProperties
from core.voxel_mesh import AdaptiveVoxelMesh, _build_kdtree
from core.memory_manager import create_managed_array, get_memory_manager
from config.hardware_config import hardware
from core.solidification_loop import (
    DEFAULT_SCHEDULE_CURVE_EXPONENT,
    NIYAMA_EVAL_FRACTION_SOLID,
    compute_schedule_fraction_solid,
    run_time_integration,
)
from core.analysis_strategies import run_post_analyses

SHRINK_SCALAR_FORMULA = "adjusted_risk = clamp(raw_risk / shrink_scalar, 0, 1)"

try:
    from core.error_logger import logger
except ImportError:
    from core.error_logger_dummy import logger

try:
    from core.gpu_compute import gpu_compute, compute_distances_accelerated, update_solidification_accelerated
    GPU_AVAILABLE = gpu_compute.is_gpu_available
except ImportError:
    GPU_AVAILABLE = False

try:
    from core.incremental_solidification import IncrementalSolidificationEngine
    INCREMENTAL_AVAILABLE = True
except ImportError:
    INCREMENTAL_AVAILABLE = False

try:
    from core.feeding_path import FeedingPathAnalyzer, FeedingPathResult
    FEEDING_PATH_AVAILABLE = True
except ImportError:
    FEEDING_PATH_AVAILABLE = False

try:
    from core.stress_analysis import StressAnalyzer, StressAnalysisResult
    STRESS_ANALYSIS_AVAILABLE = True
except ImportError:
    STRESS_ANALYSIS_AVAILABLE = False

try:
    from core.microstructure import MicrostructurePredictor, MicrostructureResult
    MICROSTRUCTURE_AVAILABLE = True
except ImportError:
    MICROSTRUCTURE_AVAILABLE = False

@dataclass
class AnalysisOptions:
    shrinkage: bool = True
    hotspot: bool = True
    niyama: bool = False
    hot_tear: bool = False
    sand_burn_in: bool = False
    fraction_liquid: bool = False
    feeding_path: bool = False
    stress_analysis: bool = False
    microstructure: bool = False
    gas_porosity: bool = False
    filling: bool = False
    sand_restraint_index: float = 0.5
    use_gpu: bool = False
    use_incremental: bool = False
    solver_mode: str = "thermal_coupled"
    solidification_params: SolidificationParameters = field(
        default_factory=SolidificationParameters
    )

SOLVER_MODE_SCHEDULE_LEGACY = "schedule_legacy"
SOLVER_MODE_THERMAL_COUPLED = "thermal_coupled"
VALID_SOLVER_MODES = {
    SOLVER_MODE_SCHEDULE_LEGACY,
    SOLVER_MODE_THERMAL_COUPLED,
}

SHRINKAGE_STATISTICS_THRESHOLD = 0.20

def _has_cut_cells(mesh) -> bool:
    return bool(getattr(mesh, "cut_cells", False)) and getattr(mesh, "voxel_metal_fractions", None) is not None

CHILL_THERMAL_DIFFUSIVITY_M2_S = 1.17e-4
CHILL_RHO_CP_J_M3_K = 1800.0 * 710.0

EXPLICIT_MOLD_TIME_FACTOR = 4.0

MOLD_GRADING_LINK_HTC = float("inf")

def is_database_core_sand(code: int) -> bool:
    if int(code) == SAND_MATERIAL_CUSTOM:
        return False
    from core.mold_materials import is_database_sand

    return is_database_sand(code)

STEEL_SAND_AIR_GAP_CURVE = (
    np.array([-1500.0, -730.0, -530.0, -430.0, -30.0, 0.0]),
    np.array([0.005, 0.05, 0.25, 0.38, 0.8, 1.0]),
)

NIYAMA_THRESHOLDS = {
    'WCB': 1.0,
    'CF8': 0.7,
    'C958': 1.2,
    'C964': 1.1,
    'CA6NM': 0.95,
    'MONEL400': 0.85,
    'IN625': 0.9,
}

ANALYSIS_LABELS: Tuple[Tuple[str, str], ...] = (
    ("shrinkage", "Shrinkage"),
    ("hotspot", "Hot Spots"),
    ("niyama", "Niyama"),
    ("sand_burn_in", "Sand Burn-In"),
    ("gas_porosity", "Gas Porosity"),
    ("fraction_liquid", "Fraction Liquid"),
    ("hot_tear", "Hot Tear"),
    ("feeding_path", "Feeding Path"),
    ("stress_analysis", "Stress"),
    ("microstructure", "Microstructure"),
    ("filling", "Mold Filling (CFD)"),
)

def describe_analyses(analysis_options, alloy=None) -> List[str]:
    labels: List[str] = []
    for attribute, label in ANALYSIS_LABELS:
        if not getattr(analysis_options, attribute, False):
            continue
        if attribute == "niyama" and alloy is not None:
            threshold = NIYAMA_THRESHOLDS.get(str(alloy.name).split()[0], 1.0)
            labels.append(f"Niyama (threshold={threshold})")
        else:
            labels.append(label)
    return labels

class SolidificationEngine:

    def __init__(self, voxel_mesh: AdaptiveVoxelMesh, alloy: AlloyProperties,
                 analysis_options: Optional[AnalysisOptions] = None):
        self.mesh = voxel_mesh
        self.alloy = alloy
        self.analysis_options = analysis_options or AnalysisOptions()

        self.time = 0.0

        self._chill_centers: Optional[np.ndarray] = None
        self._chill_thicknesses: Optional[np.ndarray] = None
        self._chill_influence_thicknesses: Optional[np.ndarray] = None
        self._chill_surface_centers: Optional[np.ndarray] = None
        self._chill_boost: Optional[np.ndarray] = None

        self._air_gap_slowdown: Optional[np.ndarray] = None
        self._thermal_heat_model: Optional[HeatTransferModel] = None
        self._thermal_solidification_model: Optional[ThermalSolidificationModel] = None
        self._mold_extension = None
        self._initial_temperature_field: Optional[np.ndarray] = None
        self._fill_arrival_times: Optional[np.ndarray] = None
        self._fill_frames: Tuple[Any, ...] = ()
        self._filling_summary: Optional[Dict[str, Any]] = None
        self._thread_budget_provider: Optional[Callable[[], Optional[int]]] = None
        self._active_thread_budget: Optional[int] = None
        self._thermal_peak_cooling_rates: Optional[np.ndarray] = None
        self._thermal_latest_temperatures: Optional[np.ndarray] = None
        self._thermal_latest_gradients: Optional[np.ndarray] = None
        self._thermal_voxel_size_m: Optional[float] = None
        self._thermal_niyama_sampled: Optional[np.ndarray] = None
        self._thermal_niyama_gradient_at_eval: Optional[np.ndarray] = None
        self._thermal_niyama_cooling_rate_at_eval: Optional[np.ndarray] = None
        self._thermal_niyama_temperature_at_eval: Optional[np.ndarray] = None
        self._thermal_niyama_time_at_eval: Optional[np.ndarray] = None
        self._thermal_niyama_values_at_eval: Optional[np.ndarray] = None
        self._thermal_niyama_quality: Optional[np.ndarray] = None
        self.niyama_eval_fraction_solid = NIYAMA_EVAL_FRACTION_SOLID

        self.time_step = 0.1
        self.max_time = None

        self.solidification_times = None
        self.solidification_start_times = None
        self.solidification_end_times = None
        self.local_solidification_time = None
        self.surface_distances = None
        self.cast_surface_distances = None
        self._risk_moduli = None
        self.feeding_fraction_solid = None
        self.feeding_path_lengths = None

        self.shrinkage_risk = None
        self.shrinkage_probability = None
        self.porosity_risk = None
        self.porosity_probability = None
        self.raw_shrinkage_probability = None
        self.raw_porosity_probability = None
        self.shrink_scalar = SHRINK_SCALAR_DEFAULT
        self.shrink_scalar_formula = SHRINK_SCALAR_FORMULA
        self.shrink_scalar_applied = False

        self.hotspot_intensity = None

        self.niyama_values = None
        self.niyama_defect_risk = None
        self.niyama_risk = None
        self.combined_feed_risk = None
        self.niyama_quality = None
        self.niyama_metadata = None
        self.thermal_gradients = None
        self.cooling_rates = None

        self.hot_tear_risk = None
        self.hot_tear_zones = None
        self.strain_rate = None
        self.btr_mask = None

        self.sand_burn_in_risk = None
        self.sand_burn_in_zones = None

        self.gas_generation = None
        self.gas_entrapment = None
        self.gas_porosity_risk = None
        self.gas_porosity_hotspots = None

        self.feeding_path_result = None
        self.feeding_quality = None
        self.feeding_blocked = None
        self.feeding_open_fraction = None
        self.feeding_latest_open_time = None
        self.feeding_effectiveness = None
        self.feeding_darcy_margin = None

        self.stress_result = None
        self.effective_stress = None
        self.distortion_magnitude = None
        self.crack_susceptibility = None

        self.microstructure_result = None
        self.sdas = None
        self.predicted_uts = None

        self.fraction_liquid_history = None
        self.fraction_liquid_times = None
        self._schedule_curve_exponent = DEFAULT_SCHEDULE_CURVE_EXPONENT

        self.total_shrinkage_volume = 0.0
        self.affected_voxel_count = 0
        self.hotspot_count = 0
        self.niyama_defect_count = 0
        self.hot_tear_count = 0
        self.sand_burn_in_count = 0
        self.gas_porosity_count = 0
        self.feeding_blocked_count = 0
        self.max_stress_mpa = 0.0
        self.avg_sdas_um = 0.0

        self.chvorinov_time = None
        self.local_modulus = None

        self._incremental_engine = None
        self._use_gpu = self.analysis_options.use_gpu and GPU_AVAILABLE
        self._use_incremental = self.analysis_options.use_incremental and INCREMENTAL_AVAILABLE
        self._logger = logger
        self._gpu_update_fn = update_solidification_accelerated if GPU_AVAILABLE else None
        self._incremental_engine_cls = IncrementalSolidificationEngine if INCREMENTAL_AVAILABLE else None
        self._gpu_device_name = gpu_compute.device_name if GPU_AVAILABLE else "GPU"
        self._feeding_path_available = FEEDING_PATH_AVAILABLE
        self._stress_analysis_available = STRESS_ANALYSIS_AVAILABLE
        self._microstructure_available = MICROSTRUCTURE_AVAILABLE
        self._niyama_thresholds = NIYAMA_THRESHOLDS

    def _get_solver_mode(self) -> str:
        solver_mode = getattr(self.analysis_options, "solver_mode", SOLVER_MODE_THERMAL_COUPLED)
        if solver_mode not in VALID_SOLVER_MODES:
            return SOLVER_MODE_THERMAL_COUPLED
        return solver_mode

    def _get_mean_voxel_size_inches(self) -> float:
        voxel_sizes = getattr(self.mesh, "voxel_sizes", None)
        if voxel_sizes is not None and len(voxel_sizes) > 0:
            return float(np.mean(np.asarray(voxel_sizes, dtype=np.float64)))
        base_voxel_size = getattr(self.mesh, "base_voxel_size", None)
        if base_voxel_size is not None:
            return float(base_voxel_size)
        return 0.1

    def _alloy_property_curves(self) -> Dict[str, np.ndarray]:
        solidus = float(self.alloy.solidus_temp)
        liquidus = float(self.alloy.liquidus_temp)
        k_s = float(self.alloy.thermal_conductivity)
        cp = float(self.alloy.specific_heat)
        rho_s = float(self.alloy.density_solid)
        rho_l = float(self.alloy.density_liquid)
        return {
            "temp_points": np.array(
                [20.0, 0.5 * solidus, solidus, (solidus + liquidus) / 2, liquidus, liquidus + 100]
            ),
            "conductivity": k_s * np.array([1.0, 0.95, 0.85, 0.77, 0.70, 0.665]),
            "specific_heat": cp * np.array([0.85, 0.92, 1.0, 1.05, 1.10, 1.15]),
            "density": np.array(
                [rho_s, 0.98 * rho_s, 0.96 * rho_s, (rho_s + rho_l) / 2, rho_l, 0.98 * rho_l]
            ),
        }

    def _allocate_thermal_tracking_arrays(self, n_voxels: int) -> None:
        self._thermal_peak_cooling_rates = np.zeros(n_voxels, dtype=np.float32)
        self._thermal_niyama_sampled = np.zeros(n_voxels, dtype=bool)
        self._thermal_niyama_gradient_at_eval = np.zeros(n_voxels, dtype=np.float32)
        self._thermal_niyama_cooling_rate_at_eval = np.zeros(n_voxels, dtype=np.float32)
        self._thermal_niyama_temperature_at_eval = np.zeros(n_voxels, dtype=np.float32)
        self._thermal_niyama_time_at_eval = np.zeros(n_voxels, dtype=np.float32)
        self._thermal_niyama_values_at_eval = np.full(n_voxels, np.inf, dtype=np.float32)
        self._thermal_niyama_quality = np.zeros(n_voxels, dtype=np.int8)

    def _initial_thermal_temperatures(
        self,
        n_voxels: int,
        chill_indices: np.ndarray,
        chill_temperature: float,
        core_indices: Optional[np.ndarray] = None,
        core_temperature: Optional[float] = None,
    ) -> np.ndarray:
        if self._initial_temperature_field is not None:
            initial_temperatures = np.array(
                self._initial_temperature_field, dtype=np.float64, copy=True
            )
        else:
            initial_temperatures = np.full(n_voxels, self.alloy.pouring_temp, dtype=np.float64)

        if len(chill_indices) > 0:
            initial_temperatures[chill_indices] = chill_temperature
        if core_indices is not None and len(core_indices) > 0 and core_temperature is not None:
            initial_temperatures[core_indices] = core_temperature
        cut_cell_merge = getattr(self, "_cut_cell_merge", None)
        if cut_cell_merge is not None and len(cut_cell_merge.slaves) > 0:
            initial_temperatures[cut_cell_merge.slaves] = initial_temperatures[cut_cell_merge.masters]
        return initial_temperatures

    def _preset_mold_solid_state(self, mold_indices: np.ndarray) -> None:
        if len(mold_indices) > 0:
            thermal_model = self._thermal_solidification_model
            thermal_model._fraction_solid[mold_indices] = 1.0
            thermal_model._fraction_solid_prev[mold_indices] = 1.0
            thermal_model._solidification_start[mold_indices] = 0.0
            thermal_model._solidification_end[mold_indices] = 0.0

        self.mesh.voxel_states[:] = 0.0
        if len(mold_indices) > 0:
            self.mesh.voxel_states[mold_indices] = 1.0
            self.solidification_start_times[mold_indices] = 0.0
            self.solidification_end_times[mold_indices] = 0.0
            self.local_solidification_time[mold_indices] = 0.0
            self.solidification_times[mold_indices] = 0.0
            if self.feeding_fraction_solid is not None:
                self.feeding_fraction_solid[mold_indices] = 1.0

    def _mold_interface_decay(self, h0: float, floor: Optional[float] = None) -> Callable[[float], float]:
        tb = self.analysis_options.solidification_params.thermal_boundary
        if floor is None:
            ap = getattr(self.analysis_options.solidification_params, "air_gap", None)
            floor = float(getattr(ap, "htc_floor_fraction", 0.35))
        mold_k = float(getattr(tb, "mold_thermal_conductivity", 0.6))
        mold_alpha = float(getattr(tb, "mold_thermal_diffusivity", 3.3e-7))
        pen_prefactor = mold_k / np.sqrt(np.pi * max(mold_alpha, 1e-12))

        def multiplier(t: float) -> float:
            if t <= 0.0:
                return 1.0
            h_pen = pen_prefactor / np.sqrt(max(t, 1e-9))
            h_eff = 1.0 / (1.0 / max(h0, 1e-9) + 1.0 / max(h_pen, 1e-9))
            return max(h_eff / max(h0, 1e-9), floor)

        return multiplier

    def _initialize_thermal_models(self, show_progress: bool) -> None:
        n_voxels = len(self.mesh.voxel_centers)
        voxel_size_in = self._get_mean_voxel_size_inches()
        voxel_size_m = max(inches_to_meters(voxel_size_in), 1e-6)
        self._thermal_voxel_size_m = voxel_size_m

        mesh_voxel_sizes = getattr(self.mesh, "voxel_sizes", None)
        if mesh_voxel_sizes is not None and len(mesh_voxel_sizes) == n_voxels:
            voxel_sizes_m = inches_to_meters(
                np.asarray(mesh_voxel_sizes, dtype=np.float64)
            )
        else:
            voxel_sizes_m = np.full(n_voxels, voxel_size_m, dtype=np.float64)

        chill_indices = self._get_chill_voxel_indices()
        chill_alpha = CHILL_THERMAL_DIFFUSIVITY_M2_S
        chill_rho_cp = CHILL_RHO_CP_J_M3_K
        boundary_params = self.analysis_options.solidification_params.thermal_boundary
        mold_params = self.analysis_options.solidification_params.mold
        core_indices = self._get_core_voxel_indices()
        core_rho_cp = float(boundary_params.core_density) * float(boundary_params.core_specific_heat)
        core_alpha = float(boundary_params.core_thermal_conductivity) / core_rho_cp
        tabulated_bodies = []
        constant_core_indices = core_indices
        if is_database_core_sand(mold_params.core_material_code) and len(core_indices) > 0:
            from core.mold_materials import tabulated_body

            tabulated_bodies.append(tabulated_body(core_indices, mold_params.core_material_code))
            constant_core_indices = core_indices[:0]

        layout = getattr(self.mesh, "mold_layout", None)
        extension = None
        if layout is not None and layout.mold_cell_count > 0:
            from core.mold_setup import build_thermal_extension

            extension = build_thermal_extension(
                self.mesh,
                layout,
                grading_levels=mold_params.mold_grading_levels,
                fine_band_cells=mold_params.mold_fine_band_cells,
            )
        self._mold_extension = extension

        diffusivity_overrides = None
        rho_cp_overrides = None
        if len(chill_indices) > 0 or len(constant_core_indices) > 0:
            override_idx = np.concatenate([chill_indices, constant_core_indices]).astype(np.int64)
            diffusivity_overrides = (
                override_idx,
                np.concatenate([
                    np.full(len(chill_indices), chill_alpha, dtype=np.float64),
                    np.full(len(constant_core_indices), core_alpha, dtype=np.float64),
                ]),
            )
            rho_cp_overrides = (
                override_idx,
                np.concatenate([
                    np.full(len(chill_indices), chill_rho_cp, dtype=np.float64),
                    np.full(len(constant_core_indices), core_rho_cp, dtype=np.float64),
                ]),
            )
        contact_conductance = (
            (core_indices.astype(np.int64), float(boundary_params.htc_core_contact))
            if len(core_indices) > 0
            else None
        )

        property_curves = self._alloy_property_curves()
        mesh_neighbors = np.asarray(self.mesh.voxel_neighbors, dtype=np.int32)
        cut_cell_inputs = self._cut_cell_solver_inputs(mesh_neighbors)
        if extension is not None:
            neighbors = extension.mesh_neighbors
            air_boundaries = self._mesh_air_boundaries(extension, cut_cell_inputs)
        else:
            neighbors = mesh_neighbors
            air_boundaries = []
        neighbors = self._without_sleeved_riser_sand_links(neighbors)

        casting_model = HeatTransferModel(
            n_voxels=n_voxels,
            voxel_size=voxel_size_m,
            neighbors=neighbors,
            alloy=self.alloy,
            use_temperature_dependent=True,
            voxel_sizes=voxel_sizes_m,
            diffusivity_overrides=diffusivity_overrides,
            property_curves=property_curves,
            rho_cp_overrides=rho_cp_overrides,
            contact_conductance=contact_conductance,
            tabulated_bodies=tabulated_bodies or None,
            **cut_cell_inputs,
        )
        self._thermal_heat_model = casting_model
        self._thermal_solidification_model = ThermalSolidificationModel(
            n_voxels=n_voxels,
            alloy=self.alloy,
        )
        self._allocate_thermal_tracking_arrays(n_voxels)

        mold_temperature = boundary_params.mold_temperature
        chill_temperature = boundary_params.chill_temperature
        initial_temperatures = self._initial_thermal_temperatures(
            n_voxels, chill_indices, chill_temperature, core_indices, mold_temperature
        )
        casting_model.set_initial_temperature(initial_temperatures)
        casting_model.set_heat_source(np.zeros(n_voxels, dtype=np.float64))
        if extension is None:
            self._configure_thermal_boundaries(
                mold_temperature=mold_temperature, chill_temperature=chill_temperature
            )
        else:
            for kwargs in air_boundaries:
                casting_model.add_boundary(**kwargs)
            self._thermal_heat_model = self._build_coupled_mold(extension, casting_model, mold_temperature)

        self._preset_mold_solid_state(np.concatenate([chill_indices, core_indices]).astype(np.int64))

        self._thermal_latest_temperatures = initial_temperatures.astype(np.float32, copy=True)
        self._thermal_latest_gradients = np.zeros(n_voxels, dtype=np.float32)

        if show_progress:
            print("  Thermal-coupled mode active")
            print(f"    Representative voxel size: {voxel_size_in:.4f}\" ({voxel_size_m:.6f} m)")
            print(f"    Surface mold temperature: {mold_temperature:.1f} C")
            print(f"    Chill temperature: {chill_temperature:.1f} C")
            backend_label = getattr(
                self._thermal_heat_model, "backend_description", "NumPy/vectorized kernels"
            )
            print(f"    Thermal solver backend: {backend_label}")
            print("    Temperature-dependent properties: enabled")
            print("    Chill model: finite heat sink (graphite, ambient HTC)")
            if extension is None:
                print("    Mold: semi-infinite sand boundary, air-gap HTC decay (floor factor)")
            else:
                from core.mold_materials import sand_label

                graded = (
                    f", graded from {extension.lattice_mold_cells:,} lattice cells, up to {extension.max_span}x"
                    if extension.graded
                    else ""
                )
                print(
                    f"    Mold: {extension.n_mold:,} sand cells{graded} ({sand_label(mold_params.mold_material_code)}), "
                    f"{len(extension.pair_first):,} metal-sand contacts, "
                    f"{len(extension.air_mesh_cells) + len(extension.air_mold_cells):,} cells open to the air"
                )
            if is_database_core_sand(mold_params.core_material_code) and len(core_indices) > 0:
                from core.mold_materials import sand_label

                print(f"    Core sand: {sand_label(mold_params.core_material_code)}")
            sleeved_cells = int(np.count_nonzero(self._adiabatic_riser_cells(n_voxels)))
            if sleeved_cells > 0:
                print(f"    Riser sleeves: {sleeved_cells:,} riser cells exchange no heat with mold or core sand")

        try:
            adaptive_dt_enabled = os.getenv("CASTSIM_THERMAL_ADAPTIVE_DT", "1").strip().lower() not in ("0", "false", "no")
        except Exception:
            adaptive_dt_enabled = True
        if adaptive_dt_enabled and self._thermal_heat_model is not None:
            try:
                dt_cfl = float(self._thermal_heat_model.get_stable_timestep())
            except Exception:
                dt_cfl = 0.0
            if dt_cfl > 0.0 and np.isfinite(dt_cfl):
                dt_cfl = float(np.clip(dt_cfl, 0.05, 2.0))
                if dt_cfl >= 2.0 * self.time_step:
                    self.time_step = float(f"{dt_cfl:.4g}")
                    if show_progress:
                        print(f"    Adaptive thermal timestep: {self.time_step:.4f}s (CFL)")

    def _mesh_master_of(self) -> np.ndarray:
        n_mesh = len(self.mesh.voxel_centers)
        master_of = np.arange(n_mesh, dtype=np.int64)
        merge = getattr(self, "_cut_cell_merge", None)
        if merge is not None and len(merge.slaves) > 0:
            master_of[np.asarray(merge.slaves, dtype=np.int64)] = np.asarray(merge.masters, dtype=np.int64)
        return master_of

    def _mesh_part_types(self) -> np.ndarray:
        raw_types = getattr(self.mesh, "voxel_part_types", None)
        n_mesh = len(self.mesh.voxel_centers)
        if raw_types is None:
            return np.full(n_mesh, PART_TYPE_CAST, dtype=np.int32)
        return np.asarray(raw_types, dtype=np.int32)

    def _mesh_air_boundaries(self, extension, cut_cell_inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        mold = self.analysis_options.solidification_params.mold
        n_mesh = extension.n_mesh
        part_types = self._mesh_part_types()
        master_of = self._mesh_master_of()
        area_in2 = np.bincount(master_of[extension.air_mesh_cells], weights=extension.air_mesh_area, minlength=n_mesh)
        air_cells = np.flatnonzero(area_in2 > 0.0)
        air_in2 = area_in2[air_cells]
        sand = np.isin(part_types, SAND_PART_CODES)
        if cut_cell_inputs:
            areas = np.asarray(cut_cell_inputs["boundary_areas"], dtype=np.float64).copy()
            air_m2 = inches_to_meters(inches_to_meters(air_in2))
            sand_air = sand[air_cells]
            areas[air_cells[sand_air]] = air_m2[sand_air]
            weights = np.ones(len(air_cells), dtype=np.float64)
            metal_air = ~sand_air
            weights[metal_air] = np.divide(
                air_m2[metal_air], areas[air_cells[metal_air]],
                out=np.zeros(int(metal_air.sum())), where=areas[air_cells[metal_air]] > 0.0,
            )
            cut_cell_inputs["boundary_areas"] = areas
            weights = np.minimum(weights, 1.0)
        else:
            pitch_in = float(self.mesh.base_voxel_size)
            weights = air_in2 / (pitch_in * pitch_in)
        boundaries: List[Dict[str, Any]] = []
        for is_sand, emissivity in ((False, mold.metal_emissivity), (True, mold.sand_emissivity)):
            sel = sand[air_cells] == is_sand
            if np.any(sel):
                boundaries.append({
                    "indices": air_cells[sel].astype(np.int32),
                    "boundary_type": BoundaryType.RADIATION,
                    "value": float(emissivity),
                    "htc": float(mold.air_htc),
                    "T_infinity": float(mold.ambient_temperature),
                    "weights": weights[sel],
                })
        if extension.unassigned_area > 0.0:
            logger.warning(
                f"Mold coupling: {extension.unassigned_area:.3g} in^2 of metal surface faces neither "
                f"mold sand nor air and carries no boundary"
            )
        return boundaries

    def _build_coupled_mold(self, extension, casting_model: HeatTransferModel, mold_temperature: float):
        from core.mold_materials import mold_tabulated_body
        from physics.heat_transfer import _stencil_slot_weights
        from physics.mold_coupling import CoupledMoldSolver, MoldPairs

        tb = self.analysis_options.solidification_params.thermal_boundary
        mold = self.analysis_options.solidification_params.mold
        layout = self.mesh.mold_layout
        part_types = self._mesh_part_types()
        master_of = self._mesh_master_of()
        pitch_m = inches_to_meters(float(layout.lattice.pitch))

        graded_inputs: Dict[str, Any] = {}
        if extension.graded:
            graded_inputs["voxel_sizes"] = pitch_m * np.asarray(extension.mold_span, dtype=np.float64)
            if len(extension.link_first):
                graded_inputs["interface_links"] = InterfaceLinks(
                    first=np.asarray(extension.link_first, dtype=np.int64),
                    second=np.asarray(extension.link_second, dtype=np.int64),
                    area=inches_to_meters(inches_to_meters(np.asarray(extension.link_area, dtype=np.float64))),
                    group=np.zeros(len(extension.link_first), dtype=np.int64),
                    htc=[MOLD_GRADING_LINK_HTC],
                    time_functions=[None],
                )
        mold_model = HeatTransferModel(
            n_voxels=extension.n_mold,
            voxel_size=pitch_m,
            neighbors=np.asarray(extension.mold_neighbors, dtype=np.int32),
            alloy=self.alloy,
            tabulated_bodies=[mold_tabulated_body(np.arange(extension.n_mold), mold.mold_material_code, tb)],
            **graded_inputs,
        )
        mold_model.set_initial_temperature(float(mold_temperature))
        if len(extension.air_mold_cells):
            pitch_in = float(layout.lattice.pitch)
            if extension.graded:
                cell_in = pitch_in * np.asarray(extension.mold_span, dtype=np.float64)[extension.air_mold_cells]
                air_weights = extension.air_mold_area / (cell_in * cell_in)
            else:
                air_weights = extension.air_mold_area / (pitch_in * pitch_in)
            mold_model.add_boundary(
                extension.air_mold_cells.astype(np.int32),
                BoundaryType.RADIATION,
                value=float(mold.sand_emissivity),
                htc=float(mold.air_htc),
                T_infinity=float(mold.ambient_temperature),
                weights=air_weights,
            )

        first = master_of[extension.pair_first]
        second = extension.pair_second
        area = extension.pair_area
        adiabatic = self._adiabatic_riser_cells(extension.n_mesh)
        cut = adiabatic[first]
        if np.any(cut):
            first, second, area = first[~cut], second[~cut], area[~cut]
        codes = part_types[first]
        group = np.zeros(len(first), dtype=np.int64)
        group[codes == PART_TYPE_RISER_POST] = 1
        group[codes == PART_TYPE_CHILL] = 2
        curve = None
        if mold.interface_htc_curve:
            solidus = float(self.alloy.solidus_temp)
            curve = (STEEL_SAND_AIR_GAP_CURVE[0] + solidus, STEEL_SAND_AIR_GAP_CURVE[1])
        htc = [mold.htc_cast_mold, tb.htc_riser_post, mold.htc_chill_mold]
        factors: List[Any] = [curve, None, None]
        if np.any(codes == PART_TYPE_RISER):
            riser_label, riser_moduli = self._riser_moduli_for_sleeves(extension, master_of, part_types)
            skip = self._bare_riser_cells(extension.n_mesh) | adiabatic
            other_risers = set(np.unique(riser_label[skip & (riser_label >= 0)]).tolist())
            for n, modulus in enumerate(riser_moduli):
                if n in other_risers:
                    continue
                sel = (codes == PART_TYPE_RISER) & (riser_label[first] == n)
                group[sel] = len(htc)
                htc.append(float(mold.riser_sleeve_htc) * max(modulus, 0.05) ** -RISER_SLEEVE_EXPONENT)
                factors.append(None)

        core_first = master_of[extension.stencil_first]
        slot_weight = _stencil_slot_weights(26)[extension.stencil_slot]
        sizes = casting_model._voxel_sizes
        dx = sizes[core_first] if sizes is not None else np.full(len(core_first), casting_model.voxel_size)
        scale = np.ones(len(core_first))
        if casting_model._conductance_factors is not None:
            f = casting_model._conductance_factors[core_first]
            a = casting_model._link_apertures[core_first]
            scale = np.minimum(f, 1.0) * 0.5 * (a + 1.0)
        keep = slot_weight > 0.0
        n_contact = len(first)
        pairs = MoldPairs(
            first=np.concatenate([first, core_first[keep]]),
            second=np.concatenate([second, extension.stencil_second[keep]]),
            area=np.concatenate([inches_to_meters(inches_to_meters(area)), np.zeros(int(keep.sum()))]),
            group=np.concatenate([group, np.full(int(keep.sum()), -1, dtype=np.int64)]),
            stencil_factor=np.concatenate([np.zeros(n_contact), (slot_weight * scale * dx)[keep]]),
            htc=htc,
            temperature_factors=factors,
            stencil_contact_htc=float(tb.htc_core_contact),
        )
        return CoupledMoldSolver(casting_model, mold_model, pairs)

    def _bare_riser_cells(self, n_mesh: int) -> np.ndarray:
        bare = getattr(self.mesh, "voxel_riser_bare", None)
        if bare is None or len(bare) != n_mesh:
            return np.zeros(n_mesh, dtype=bool)
        return np.asarray(bare, dtype=bool)

    def _adiabatic_riser_cells(self, n_mesh: int) -> np.ndarray:
        adiabatic = getattr(self.mesh, "voxel_riser_adiabatic", None)
        if adiabatic is None or len(adiabatic) != n_mesh:
            return np.zeros(n_mesh, dtype=bool)
        return (self._mesh_part_types() == PART_TYPE_RISER) & np.asarray(adiabatic, dtype=bool)

    def _without_sleeved_riser_sand_links(self, neighbors: np.ndarray) -> np.ndarray:
        part_types = self._mesh_part_types()
        sleeved = self._adiabatic_riser_cells(len(part_types))
        sand = np.isin(part_types, SAND_PART_CODES)
        if not sleeved.any() or not sand.any():
            return neighbors
        rows, slots = np.nonzero(neighbors >= 0)
        cols = neighbors[rows, slots]
        cut = (sleeved[rows] & sand[cols]) | (sand[rows] & sleeved[cols])
        if not cut.any():
            return neighbors
        out = np.array(neighbors, copy=True)
        out[rows[cut], slots[cut]] = -1
        return out

    def _riser_moduli_for_sleeves(
        self, extension, master_of: np.ndarray, part_types: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        n_mesh = extension.n_mesh
        label = np.full(n_mesh, -1, dtype=np.int64)
        risers = np.flatnonzero(part_types == PART_TYPE_RISER)
        if len(risers) == 0:
            return label, []
        neighbors = np.asarray(self.mesh.voxel_neighbors)[risers]
        local = np.full(n_mesh, -1, dtype=np.int64)
        local[risers] = np.arange(len(risers))
        rows, slots = np.nonzero(neighbors >= 0)
        cols = local[neighbors[rows, slots]]
        keep = cols >= 0
        graph = coo_matrix(
            (np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(len(risers), len(risers))
        )
        n_components, component = connected_components(graph, directed=False)
        label[risers] = component
        volumes = np.bincount(
            component, weights=np.asarray(self.mesh.voxel_volumes, dtype=np.float64)[risers], minlength=n_components
        )
        area = np.zeros(n_mesh, dtype=np.float64)
        np.add.at(area, master_of[extension.pair_first], extension.pair_area)
        np.add.at(area, master_of[extension.air_mesh_cells], extension.air_mesh_area)
        areas = np.bincount(component, weights=area[risers], minlength=n_components)
        moduli = [float(v / a) if a > 0.0 else 0.0 for v, a in zip(volumes, areas)]
        return label, moduli

    def _cut_cell_solver_inputs(self, neighbors: np.ndarray) -> Dict[str, Any]:
        self._cut_cell_merge = None
        if not _has_cut_cells(self.mesh):
            return {}
        from core.cut_cells import DEFAULT_PHI_MIN, merge_small_cut_cells

        part_types = getattr(self.mesh, "voxel_part_types", None)
        merge = merge_small_cut_cells(
            np.asarray(self.mesh.voxel_metal_fractions, dtype=np.float64),
            np.asarray(self.mesh.voxel_surface_areas, dtype=np.float64),
            neighbors,
            None if part_types is None else np.asarray(part_types),
            DEFAULT_PHI_MIN,
        )
        self._cut_cell_merge = merge
        return {
            "capacity_weights": merge.capacity_weights,
            "boundary_areas": inches_to_meters(inches_to_meters(merge.boundary_areas)),
            "slave_cells": (merge.slaves, merge.masters),
        }

    def _configure_thermal_boundaries(self, *, mold_temperature: float, chill_temperature: float) -> None:
        if self._thermal_heat_model is None:
            return

        tb = self.analysis_options.solidification_params.thermal_boundary
        _mold_interface_decay = self._mold_interface_decay

        cut_cell_merge = getattr(self, "_cut_cell_merge", None)
        if cut_cell_merge is not None:
            surface_indices = np.flatnonzero(cut_cell_merge.boundary_areas > 0.0).astype(np.int32)
        else:
            surface_indices = np.asarray(self.mesh.get_surface_voxel_indices(), dtype=np.int32)
        if surface_indices.size == 0:
            return

        if self.mesh.voxel_part_types is None:
            part_types = np.zeros(len(self.mesh.voxel_centers), dtype=np.int32)
        else:
            part_types = np.asarray(self.mesh.voxel_part_types, dtype=np.int32)

        surface_part_types = part_types[surface_indices]
        cast_surface = surface_indices[surface_part_types == PART_TYPE_CAST]
        riser_surface = surface_indices[surface_part_types == PART_TYPE_RISER]
        riser_post_surface = surface_indices[surface_part_types == PART_TYPE_RISER_POST]
        flow_path_surface = surface_indices[is_flow_path(surface_part_types)]
        core_surface = surface_indices[surface_part_types == PART_TYPE_CORE]
        chill_indices = self._get_chill_voxel_indices().astype(np.int32, copy=False)

        if cast_surface.size > 0:
            self._thermal_heat_model.add_boundary(
                cast_surface,
                BoundaryType.INTERFACE,
                htc=tb.htc_cast_interface,
                T_infinity=mold_temperature,
                time_function=_mold_interface_decay(tb.htc_cast_interface),
            )
        if riser_post_surface.size > 0:
            self._thermal_heat_model.add_boundary(
                riser_post_surface,
                BoundaryType.INTERFACE,
                htc=tb.htc_riser_post,
                T_infinity=mold_temperature,
                time_function=_mold_interface_decay(tb.htc_riser_post),
            )
        bare_riser = self._bare_riser_cells(len(part_types))
        if riser_surface.size > 0 and np.any(bare_riser[riser_surface]):
            bare_surface = riser_surface[bare_riser[riser_surface]]
            riser_surface = riser_surface[~bare_riser[riser_surface]]
            self._thermal_heat_model.add_boundary(
                bare_surface,
                BoundaryType.INTERFACE,
                htc=tb.htc_cast_interface,
                T_infinity=mold_temperature,
                time_function=_mold_interface_decay(tb.htc_cast_interface),
            )
        riser_surface = riser_surface[~self._adiabatic_riser_cells(len(part_types))[riser_surface]]
        if riser_surface.size > 0:
            self._thermal_heat_model.add_boundary(
                riser_surface,
                BoundaryType.INTERFACE,
                htc=tb.htc_riser,
                T_infinity=mold_temperature,
                time_function=_mold_interface_decay(tb.htc_riser),
            )
        if flow_path_surface.size > 0:
            self._thermal_heat_model.add_boundary(
                flow_path_surface,
                BoundaryType.INTERFACE,
                htc=tb.htc_gating,
                T_infinity=mold_temperature,
                time_function=_mold_interface_decay(tb.htc_gating),
            )
        if core_surface.size > 0:
            self._thermal_heat_model.add_boundary(
                core_surface,
                BoundaryType.INTERFACE,
                htc=tb.htc_core_print,
                T_infinity=mold_temperature,
                time_function=_mold_interface_decay(tb.htc_core_print, tb.core_print_htc_floor_fraction),
            )
        chill_surface = np.intersect1d(chill_indices, surface_indices)
        if chill_surface.size > 0:
            chill_back_htc = float(getattr(tb, "chill_ambient_htc", 500.0))
            chill_floor = float(getattr(tb, "chill_htc_floor_fraction", 0.7))
            self._thermal_heat_model.add_boundary(
                chill_surface,
                BoundaryType.INTERFACE,
                htc=chill_back_htc,
                T_infinity=mold_temperature,
                time_function=_mold_interface_decay(chill_back_htc, chill_floor),
            )

    def _calculate_max_simulation_time(self):
        max_modulus = np.max(self.mesh.voxel_moduli)

        tp = self.analysis_options.solidification_params.timing
        base_mold_constant = tp.chvorinov_mold_constant

        wcb_diffusivity = 45 / (7200 * 600)
        diffusivity_ratio = wcb_diffusivity / self.alloy.thermal_diffusivity

        wcb_latent = 272000
        latent_ratio = self.alloy.latent_heat / wcb_latent

        material_factor = np.sqrt(diffusivity_ratio * latent_ratio)

        mold_constant = base_mold_constant * material_factor

        base_solidification_time = mold_constant * (max_modulus ** 2)

        riser_indices = self._get_riser_voxel_indices()
        if len(riser_indices) > 0:
            safety_factor = tp.safety_factor_with_risers
        else:
            safety_factor = tp.safety_factor_no_risers

        calculated_time = base_solidification_time * safety_factor

        explicit_mold = getattr(getattr(self.mesh, "mold_layout", None), "mold_cell_count", 0) > 0
        if explicit_mold:
            calculated_time = max(calculated_time * EXPLICIT_MOLD_TIME_FACTOR, 2.0 * tp.min_simulation_time)

        min_time = tp.min_simulation_time
        solver_mode = self._get_solver_mode()
        if solver_mode == SOLVER_MODE_THERMAL_COUPLED:
            max_time = tp.max_simulation_time
            self.max_time = np.clip(calculated_time, min_time, max_time)
        else:
            self.max_time = max(calculated_time, min_time)

        self._max_modulus = max_modulus
        self._mold_constant = mold_constant
        self._material_factor = material_factor

    def set_initial_temperature_field(
        self,
        temperatures: np.ndarray,
        *,
        fill_arrival_times: Optional[np.ndarray] = None,
    ) -> None:
        n_voxels = self.mesh.get_voxel_count()

        field = np.asarray(temperatures, dtype=np.float64)
        if field.shape != (n_voxels,):
            raise ValueError(
                f"initial temperature field has shape {field.shape}, "
                f"expected ({n_voxels},) to match the mesh"
            )
        if not np.all(np.isfinite(field)):
            raise ValueError("initial temperature field contains non-finite values")
        self._initial_temperature_field = field.copy()

        if fill_arrival_times is None:
            self._fill_arrival_times = None
            return

        times = np.asarray(fill_arrival_times, dtype=np.float64)
        if times.shape != (n_voxels,):
            raise ValueError(
                f"fill arrival times have shape {times.shape}, "
                f"expected ({n_voxels},) to match the mesh"
            )
        self._fill_arrival_times = times.copy()

    def initialize(self):
        n_voxels = self.mesh.get_voxel_count()

        print("\n  Allocating simulation arrays...")

        self.solidification_times = create_managed_array(n_voxels, dtype=np.float32, name="solidification_times")
        self.solidification_start_times = create_managed_array(
            n_voxels, dtype=np.float32, name="solidification_start_times"
        )
        self.solidification_end_times = create_managed_array(
            n_voxels, dtype=np.float32, name="solidification_end_times"
        )
        self.local_solidification_time = create_managed_array(
            n_voxels, dtype=np.float32, name="local_solidification_time"
        )
        self.surface_distances = create_managed_array(n_voxels, dtype=np.float32, name="surface_distances")
        self.feeding_fraction_solid = create_managed_array(
            n_voxels,
            dtype=np.float32,
            name="feeding_fraction_solid",
        )
        self.feeding_fraction_solid[:] = np.nan

        if self.analysis_options.shrinkage:
            self.shrinkage_probability = create_managed_array(n_voxels, dtype=np.float32, name="shrinkage_probability")
            self.shrinkage_risk = self.shrinkage_probability
            self.porosity_probability = create_managed_array(n_voxels, dtype=np.float32, name="porosity_probability")
            self.porosity_risk = self.porosity_probability

        if self.analysis_options.hotspot:
            self.hotspot_intensity = create_managed_array(n_voxels, dtype=np.float32, name="hotspot_intensity")

        if self.analysis_options.niyama:
            self.niyama_values = create_managed_array(n_voxels, dtype=np.float32, name="niyama_values")
            self.niyama_defect_risk = create_managed_array(n_voxels, dtype=np.bool_, name="niyama_defect_risk")
            self.thermal_gradients = create_managed_array(n_voxels, dtype=np.float32, name="thermal_gradients")
            self.cooling_rates = create_managed_array(n_voxels, dtype=np.float32, name="cooling_rates")

        if self.analysis_options.sand_burn_in:
            self.sand_burn_in_risk = create_managed_array(n_voxels, dtype=np.float32, name="sand_burn_in_risk")
            self.sand_burn_in_zones = create_managed_array(n_voxels, dtype=np.bool_, name="sand_burn_in_zones")

        if self.analysis_options.gas_porosity:
            self.gas_generation = create_managed_array(n_voxels, dtype=np.float32, name="gas_generation")
            self.gas_entrapment = create_managed_array(n_voxels, dtype=np.float32, name="gas_entrapment")
            self.gas_porosity_risk = create_managed_array(n_voxels, dtype=np.float32, name="gas_porosity_risk")
            self.gas_porosity_hotspots = create_managed_array(n_voxels, dtype=np.bool_, name="gas_porosity_hotspots")

        if self._should_capture_fraction_history():
            self.fraction_liquid_history = []
            self.fraction_liquid_times = []

        logger.info(f"Initializing solidification simulation", context={
            'alloy': self.alloy.name,
            'voxel_count': n_voxels,
            'time_step': self.time_step,
            'analyses': {
                'shrinkage': self.analysis_options.shrinkage,
                'hotspot': self.analysis_options.hotspot,
                'niyama': self.analysis_options.niyama,
                'sand_burn_in': self.analysis_options.sand_burn_in,
                'gas_porosity': self.analysis_options.gas_porosity,
                'fraction_liquid': self.analysis_options.fraction_liquid
            }
        })

        print("\n" + "="*60)
        print("Initializing Solidification Simulation")
        print("="*60)
        print(f"Alloy: {self.alloy.name}")
        print(f"Voxels: {n_voxels:,}")
        print(f"Time step: {self.time_step}s")

        analyses = describe_analyses(self.analysis_options, self.alloy)
        print(f"Analyses: {', '.join(analyses)}")
        print()
        print(f"  SOLVER MODE: {self._get_solver_mode().upper()}")
        print(f"  {'='*56}")

        if self.mesh.voxel_part_types is not None:
            cast_count = np.sum(self.mesh.voxel_part_types == PART_TYPE_CAST)
            riser_count = np.sum(self.mesh.voxel_part_types == PART_TYPE_RISER)
            chill_count = np.sum(self.mesh.voxel_part_types == PART_TYPE_CHILL)
            print(f"Part types: CAST={cast_count:,}, RISER={riser_count:,}, CHILL={chill_count:,}")

        self._calculate_max_simulation_time()
        print(f"\nSolidification Time Estimate (Chvorinov's Rule):")
        print(f"  Max modulus (thickest section): {self._max_modulus:.2f}\"")
        print(f"  Mold constant: {self._mold_constant:.0f} s/in² (material factor: {self._material_factor:.2f}x)")
        print(f"  Estimated max time: {self.max_time:.0f}s ({self.max_time/60:.1f} min)")
        print(f"  Formula: t = B × M² × safety_factor")

        print("="*60 + "\n")

    def run_simulation(self, show_progress: bool = True, progress_callback=None) -> bool:
        if show_progress:
            print("Running solidification simulation...")

        run_start = time.perf_counter()
        stage_timings: Dict[str, float] = {}
        self.stage_timings = stage_timings

        def _record_stage(stage_name: str, started_at: float) -> None:
            stage_timings[stage_name] = stage_timings.get(stage_name, 0.0) + (time.perf_counter() - started_at)

        self._feeding_path_cache = None
        self._connected_feeder_mask_cache = None

        solver_mode = self._get_solver_mode()
        surface_indices = self.mesh.get_surface_voxel_indices()

        self.mesh.voxel_states[:] = 0.0
        chill_indices = self._get_chill_voxel_indices()
        if len(chill_indices) > 0:
            self.mesh.voxel_states[chill_indices] = 1.0
            if show_progress:
                print(f"  Chill voxels initialized: {len(chill_indices):,}")
        core_indices = self._get_core_voxel_indices()
        if len(core_indices) > 0:
            self.mesh.voxel_states[core_indices] = 1.0
            if show_progress:
                print(f"  Core voxels initialized: {len(core_indices):,}")

        if show_progress:
            print(f"  Initial solidification fronts: {len(surface_indices):,}")

        if show_progress:
            print("  Calculating feeding distances...")

        stage_start = time.perf_counter()
        self._calculate_feeding_distances()
        _record_stage("feeding_distances", stage_start)

        if solver_mode == SOLVER_MODE_THERMAL_COUPLED:
            chill_boost = np.ones(len(self.mesh.voxel_centers), dtype=np.float32)
            riser_slowdown = np.ones(len(self.mesh.voxel_centers), dtype=np.float32)
            self._air_gap_slowdown = np.ones(len(self.mesh.voxel_centers), dtype=np.float32)

            stage_start = time.perf_counter()
            self._initialize_thermal_models(show_progress=show_progress)
            _record_stage("thermal_model_init", stage_start)
        else:
            stage_start = time.perf_counter()
            chill_boost = self._calculate_chill_influence()
            _record_stage("chill_influence", stage_start)

            stage_start = time.perf_counter()
            riser_slowdown = self._calculate_riser_influence()
            _record_stage("riser_influence", stage_start)

            stage_start = time.perf_counter()
            self._air_gap_slowdown = np.ones(len(self.mesh.voxel_centers), dtype=np.float32)
            _record_stage("air_gap_init", stage_start)
            if show_progress:
                print("  Mold air gap effect initialized (will develop during solidification)...")

            stage_start = time.perf_counter()
            self._initialize_solidification_schedule(chill_boost, riser_slowdown)
            _record_stage("schedule_init", stage_start)

        stage_start = time.perf_counter()
        run_time_integration(
            self,
            chill_boost,
            riser_slowdown,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )
        _record_stage("time_integration", stage_start)

        stage_start = time.perf_counter()
        run_post_analyses(self, show_progress=show_progress)
        _record_stage("post_analyses", stage_start)

        stage_timings["total_simulation_wall"] = time.perf_counter() - run_start
        self.last_stage_timings = dict(stage_timings)

        if show_progress:
            timing_rows = [(name, sec) for name, sec in stage_timings.items() if name != "total_simulation_wall"]
            if timing_rows:
                total_tracked = max(sum(sec for _, sec in timing_rows), 1e-12)
                print("  Stage timing breakdown:")
                for name, seconds in sorted(timing_rows, key=lambda item: item[1], reverse=True):
                    pct = (seconds / total_tracked) * 100.0
                    print(f"    {name}: {seconds:.3f}s ({pct:.1f}%)")
            print(f"  Total simulation wall time: {stage_timings['total_simulation_wall']:.3f}s")
        return True

    def _compute_graph_distance_to_riser(self, active_mask: np.ndarray) -> np.ndarray:
        n_voxels = len(self.mesh.voxel_centers)
        distances = np.full(n_voxels, np.inf, dtype=np.float32)
        riser_indices = self._get_riser_voxel_indices()
        if len(riser_indices) == 0:
            return distances

        q: deque[int] = deque()
        for idx in riser_indices:
            if active_mask[idx]:
                distances[idx] = 0.0
                q.append(int(idx))

        neighbors = np.asarray(self.mesh.voxel_neighbors, dtype=np.int32)
        while q:
            current = q.popleft()
            next_distance = distances[current] + 1.0
            for nb in neighbors[current]:
                if nb < 0 or not active_mask[nb]:
                    continue
                if next_distance < distances[nb]:
                    distances[nb] = next_distance
                    q.append(int(nb))
        return distances

    def _compute_graph_distance_to_chill(self, active_mask: np.ndarray) -> np.ndarray:
        n_voxels = len(self.mesh.voxel_centers)
        distances = np.full(n_voxels, np.inf, dtype=np.float32)
        chill_indices = self._get_chill_voxel_indices()
        if len(chill_indices) == 0:
            return distances

        traversable = active_mask.copy()
        traversable[chill_indices] = True
        q: deque[int] = deque()
        for idx in chill_indices:
            distances[idx] = 0.0
            q.append(int(idx))

        neighbors = np.asarray(self.mesh.voxel_neighbors, dtype=np.int32)
        while q:
            current = q.popleft()
            next_distance = distances[current] + 1.0
            for nb in neighbors[current]:
                if nb < 0 or not traversable[nb]:
                    continue
                if next_distance < distances[nb]:
                    distances[nb] = next_distance
                    q.append(int(nb))
        return distances

    def _average_exothermic_factor_over_duration(self, duration_s: float) -> float:
        ep = self.analysis_options.solidification_params.exothermic
        if duration_s <= 0.0:
            return 1.0

        ignition_delay_s = ep.ignition_delay
        burn_duration_s = ep.burn_duration
        insulation_only_factor = ep.insulation_only_factor
        full_exothermic_factor = ep.full_exothermic_factor
        burnout_start_s = ignition_delay_s + burn_duration_s
        decay_time_constant = ep.decay_time_constant

        ramp_duration = min(duration_s, ignition_delay_s)
        ramp_area = ramp_duration
        if ramp_duration > 0.0:
            ramp_area -= (1.0 - insulation_only_factor) * (ramp_duration ** 2) / (2.0 * ignition_delay_s)

        burn_duration = max(min(duration_s, burnout_start_s) - ignition_delay_s, 0.0)
        burn_area = burn_duration * full_exothermic_factor

        burnout_duration = max(duration_s - burnout_start_s, 0.0)
        burnout_area = 0.0
        if burnout_duration > 0.0:
            burnout_area = (
                insulation_only_factor * burnout_duration
                + (full_exothermic_factor - insulation_only_factor)
                * decay_time_constant
                * (1.0 - np.exp(-burnout_duration / decay_time_constant))
            )

        average_factor = (ramp_area + burn_area + burnout_area) / duration_s
        return float(np.clip(average_factor, full_exothermic_factor, 1.0))

    def _average_riser_post_factor_over_duration(self, duration_s: float) -> float:
        tp = self.analysis_options.solidification_params.timing
        if duration_s <= 0.0:
            return 1.0

        sand_saturation_time_constant = tp.sand_saturation_time
        max_hot_sand_slowdown = tp.max_hot_sand_slowdown
        average_factor = (
            max_hot_sand_slowdown
            + (1.0 - max_hot_sand_slowdown)
            * sand_saturation_time_constant
            * (1.0 - np.exp(-duration_s / sand_saturation_time_constant))
            / duration_s
        )
        return float(np.clip(average_factor, max_hot_sand_slowdown, 1.0))

    def _calculate_schedule_riser_slowdown(self, base_completion_times: np.ndarray) -> np.ndarray:
        n_voxels = len(self.mesh.voxel_centers)
        schedule_slowdown = np.ones(n_voxels, dtype=np.float32)

        if self.mesh.voxel_part_types is None:
            return schedule_slowdown

        fp = self.analysis_options.solidification_params.feeding
        part_types = np.asarray(self.mesh.voxel_part_types, dtype=np.int32)
        riser_indices = np.where(part_types == PART_TYPE_RISER)[0]
        if len(riser_indices) == 0:
            return schedule_slowdown

        avg_riser_factor = self._average_exothermic_factor_over_duration(
            float(np.max(base_completion_times[riser_indices]))
        )
        schedule_slowdown[riser_indices] = avg_riser_factor

        riser_post_indices = np.where(part_types == PART_TYPE_RISER_POST)[0]
        if len(riser_post_indices) > 0:
            avg_post_factor = self._average_riser_post_factor_over_duration(
                float(np.max(base_completion_times[riser_post_indices]))
            )
            schedule_slowdown[riser_post_indices] = avg_post_factor

        cast_indices = np.where(part_types == PART_TYPE_CAST)[0]
        if len(cast_indices) == 0:
            return schedule_slowdown

        riser_centers = self.mesh.voxel_centers[riser_indices]
        cast_centers = self.mesh.voxel_centers[cast_indices]
        if len(riser_centers) == 0 or len(cast_centers) == 0:
            return schedule_slowdown

        base_feeding_distance = fp.base_feeding_distance
        material_factor = self.alloy.feeding_distance_multiplier / 3.0
        riser_influence_radius = base_feeding_distance * material_factor
        min_influence_factor = fp.min_influence_factor

        riser_tree = _build_kdtree(riser_centers)
        distances, _ = riser_tree.query(cast_centers, k=1)
        compute_riser_influence_parallel(
            schedule_slowdown,
            np.ascontiguousarray(cast_indices, dtype=np.int64),
            np.ascontiguousarray(distances, dtype=np.float32),
            np.float32(riser_influence_radius),
            np.float32(avg_riser_factor),
            np.float32(min_influence_factor),
        )
        return schedule_slowdown

    def _initialize_solidification_schedule(
        self,
        chill_boost: np.ndarray,
        riser_slowdown: np.ndarray,
    ) -> None:
        n_voxels = len(self.mesh.voxel_centers)
        if n_voxels == 0:
            return

        if self.mesh.voxel_part_types is None:
            part_types = np.zeros(n_voxels, dtype=np.int32)
        else:
            part_types = np.asarray(self.mesh.voxel_part_types, dtype=np.int32)

        cast_mask = part_types == PART_TYPE_CAST
        riser_mask = part_types == PART_TYPE_RISER
        chill_mask = part_types == PART_TYPE_CHILL
        riser_post_mask = part_types == PART_TYPE_RISER_POST
        active_mask = cast_mask | riser_mask | riser_post_mask

        start_times = np.zeros(n_voxels, dtype=np.float32)
        end_times = np.zeros(n_voxels, dtype=np.float32)
        local_times = np.zeros(n_voxels, dtype=np.float32)
        self._calculate_chvorinov_times()
        base_completion_times = np.asarray(self.chvorinov_time, dtype=np.float32)
        chill_term = np.maximum(np.asarray(chill_boost, dtype=np.float32), 1.0)
        schedule_riser_slowdown = self._calculate_schedule_riser_slowdown(base_completion_times)

        completion_times = base_completion_times.copy()
        completion_times = completion_times / chill_term
        completion_times = completion_times / np.maximum(schedule_riser_slowdown, 0.15)
        completion_times[~active_mask] = 0.0

        solver_mode = self._get_solver_mode()
        if solver_mode == SOLVER_MODE_THERMAL_COUPLED:
            completion_times[active_mask] = np.clip(
                completion_times[active_mask],
                self.time_step,
                self.max_time - self.time_step,
            )

        riser_distance = self._compute_graph_distance_to_riser(active_mask)
        connected = active_mask & np.isfinite(riser_distance)
        if np.any(connected):
            max_dist = max(float(np.max(riser_distance[connected])), 1.0)
            toward_riser = 1.0 - (riser_distance / max_dist)
            directional_extension = 0.20 * float(np.max(completion_times[active_mask]))
            completion_times[connected] += directional_extension * np.clip(
                toward_riser[connected],
                0.0,
                1.0,
            )

        chill_distance = self._compute_graph_distance_to_chill(active_mask)
        chilled_cast = cast_mask & np.isfinite(chill_distance)
        if np.any(chilled_cast):
            cast_peak = max(float(np.max(completion_times[cast_mask])), self.time_step)
            min_chill_dist = float(np.min(chill_distance[chilled_cast]))
            max_chill_dist = float(np.max(chill_distance[chilled_cast]))
            chill_span = max(max_chill_dist - min_chill_dist, 1.0)
            away_from_chill = np.clip(
                (chill_distance[chilled_cast] - min_chill_dist) / chill_span,
                0.0,
                1.0,
            )
            chill_wave_completion = cast_peak * (
                0.04 + 0.68 * np.power(away_from_chill, 1.35)
            )
            completion_times[chilled_cast] = np.minimum(
                completion_times[chilled_cast],
                np.maximum(chill_wave_completion.astype(np.float32), self.time_step),
            )

        if np.any(active_mask):
            active_peak = float(np.max(completion_times[active_mask]))
            post_min = active_peak * 0.84
            post_max = active_peak * 0.92
            riser_min = max(post_max + self.time_step, active_peak * 0.92)
            riser_max = active_peak

            moduli = np.asarray(self.mesh.voxel_moduli, dtype=np.float32)
            if np.any(riser_post_mask):
                post_distance = riser_distance[riser_post_mask]
                max_dist = max(float(np.max(post_distance[post_distance < np.inf])), 1.0)
                near_riser = np.clip(1.0 - post_distance / max_dist, 0.0, 1.0)
                modulus_post = moduli[riser_post_mask]
                modulus_range = (
                    np.max(modulus_post) - np.min(modulus_post)
                    if np.max(modulus_post) > np.min(modulus_post)
                    else 1.0
                )
                modulus_factor = np.clip(
                    (modulus_post - np.min(modulus_post)) / max(modulus_range, 1e-6),
                    0.0,
                    1.0,
                )
                post_lateness = 0.80 * near_riser + 0.20 * modulus_factor
                post_completion = post_min + post_lateness * (post_max - post_min)
                completion_times[riser_post_mask] = np.maximum(
                    completion_times[riser_post_mask], post_completion
                )

            if np.any(riser_mask):
                components = self._group_risers_by_component()
                comp_ids = np.full(n_voxels, -1, dtype=np.int32)
                for cid, riser_ids in enumerate(components):
                    comp_ids[riser_ids] = cid

                riser_comp_ids = comp_ids[riser_mask]
                unique_comps = np.unique(riser_comp_ids)
                z_min_arr = np.zeros(len(unique_comps), dtype=np.float32)
                z_max_arr = np.zeros(len(unique_comps), dtype=np.float32)
                for k, cid in enumerate(unique_comps):
                    riser_centers_comp = self.mesh.voxel_centers[riser_mask][riser_comp_ids == cid]
                    z_min_arr[k] = np.min(riser_centers_comp[:, 2])
                    z_max_arr[k] = np.max(riser_centers_comp[:, 2])

                riser_heights = self.mesh.voxel_centers[riser_mask][:, 2]
                cidx = np.searchsorted(unique_comps, riser_comp_ids)
                z_min_local = z_min_arr[cidx]
                z_max_local = z_max_arr[cidx]
                z_range_local = np.maximum(z_max_local - z_min_local, 1e-6)
                height_factor = np.clip(
                    (riser_heights - z_min_local) / z_range_local,
                    0.0,
                    1.0,
                )
                modulus_riser = moduli[riser_mask]
                modulus_range_riser = (
                    np.max(modulus_riser) - np.min(modulus_riser)
                    if np.max(modulus_riser) > np.min(modulus_riser)
                    else 1.0
                )
                modulus_factor_riser = np.clip(
                    (modulus_riser - np.min(modulus_riser)) / max(modulus_range_riser, 1e-6),
                    0.0,
                    1.0,
                )
                riser_lateness = 0.65 * modulus_factor_riser + 0.35 * height_factor
                riser_completion = riser_min + riser_lateness * (riser_max - riser_min)
                completion_times[riser_mask] = np.maximum(
                    completion_times[riser_mask], riser_completion
                )

        superheat = max(float(self.alloy.pouring_temp - self.alloy.liquidus_temp), 1.0)
        freezing_range = max(float(self.alloy.liquidus_temp - self.alloy.solidus_temp), 1.0)
        mushy_fraction = np.clip(
            freezing_range / (superheat + freezing_range),
            0.15,
            0.45,
        )
        local_times[active_mask] = np.maximum(
            completion_times[active_mask] * mushy_fraction,
            self.time_step,
        )
        end_times[active_mask] = completion_times[active_mask]
        start_times[active_mask] = np.maximum(
            end_times[active_mask] - local_times[active_mask],
            0.0,
        )

        start_times[chill_mask] = 0.0
        end_times[chill_mask] = 0.0

        if solver_mode == SOLVER_MODE_THERMAL_COUPLED:
            end_times[active_mask] = np.clip(
                end_times[active_mask], self.time_step, self.max_time - self.time_step
            )
            start_times[active_mask] = np.minimum(
                start_times[active_mask],
                np.maximum(end_times[active_mask] - self.time_step, 0.0),
            )
            local_times[active_mask] = np.maximum(
                end_times[active_mask] - start_times[active_mask], self.time_step
            )

        self.solidification_start_times[:] = start_times
        self.solidification_end_times[:] = end_times
        self.local_solidification_time[:] = local_times

    @property
    def feeding_distances(self) -> Optional[np.ndarray]:
        warnings.warn(
            "`feeding_distances` is deprecated; use `surface_distances` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.surface_distances

    @feeding_distances.setter
    def feeding_distances(self, value: Optional[np.ndarray]) -> None:
        warnings.warn(
            "Setting `feeding_distances` is deprecated; assign to "
            "`surface_distances` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.surface_distances = value

    def _calculate_feeding_distances(self):
        surface_indices = self.mesh.get_surface_voxel_indices()
        surface_centers = self.mesh.voxel_centers[surface_indices]

        n_voxels = len(self.mesh.voxel_centers)

        print(f"  Computing feeding distances from {len(surface_indices):,} surface voxels...")

        if len(surface_centers) == 0:
            self.surface_distances = np.zeros(n_voxels, dtype=np.float32)
            self.cast_surface_distances = self.surface_distances
            self._risk_moduli = None
            return

        voxel_centers = np.ascontiguousarray(self.mesh.voxel_centers, dtype=np.float32)
        surface_centers = np.ascontiguousarray(surface_centers, dtype=np.float32)

        if self._use_gpu:
            print(f"  Using GPU-accelerated distance computation...")
            self.surface_distances = compute_distances_accelerated(voxel_centers, surface_centers)
        else:
            try:
                surface_tree = _build_kdtree(surface_centers)
                distances, _ = surface_tree.query(voxel_centers, k=1, workers=-1)
                self.surface_distances = np.asarray(distances, dtype=np.float32)
                print("  Feeding distances computed (KD-tree)")
            except Exception:
                self.surface_distances = compute_min_distances_parallel(voxel_centers, surface_centers)
                print(f"  Feeding distances computed (JIT-parallel)")

        self.cast_surface_distances = self._compute_cast_surface_distances(voxel_centers)
        self._risk_moduli = None

    def _get_material_surface_indices(self, include_cores: bool = True) -> np.ndarray:
        getter = getattr(self.mesh, "get_material_surface_voxel_indices", None)
        if callable(getter):
            return getter(include_cores=include_cores)
        return self.mesh.get_surface_voxel_indices()

    def _compute_cast_surface_distances(self, voxel_centers: np.ndarray) -> np.ndarray:
        material_indices = self._get_material_surface_indices()
        extra = np.setdiff1d(material_indices, self.mesh.get_surface_voxel_indices())
        if len(extra) == 0:
            return self.surface_distances

        extra_centers = np.ascontiguousarray(self.mesh.voxel_centers[extra], dtype=np.float32)
        try:
            distances, _ = _build_kdtree(extra_centers).query(voxel_centers, k=1, workers=-1)
            extra_distances = np.asarray(distances, dtype=np.float32)
        except Exception:
            extra_distances = compute_min_distances_parallel(voxel_centers, extra_centers)
        return np.minimum(self.surface_distances, extra_distances).astype(np.float32)

    def _get_cast_surface_distances(self) -> np.ndarray:
        cast_distances = getattr(self, "cast_surface_distances", None)
        if cast_distances is not None:
            return cast_distances
        return self.surface_distances

    def _get_risk_moduli(self) -> np.ndarray:
        cached = getattr(self, "_risk_moduli", None)
        if cached is not None:
            return cached

        moduli = np.asarray(self.mesh.voxel_moduli, dtype=np.float32)
        cast_distances = getattr(self, "cast_surface_distances", None)
        if cast_distances is None or len(cast_distances) != len(moduli):
            return moduli

        self._risk_moduli = np.minimum(moduli, np.asarray(cast_distances, dtype=np.float32))
        return self._risk_moduli

    def _get_chill_voxel_indices(self) -> np.ndarray:
        if self.mesh.voxel_part_types is None:
            return np.array([], dtype=int)
        return np.where(self.mesh.voxel_part_types == PART_TYPE_CHILL)[0]

    def _get_core_voxel_indices(self) -> np.ndarray:
        if self.mesh.voxel_part_types is None:
            return np.array([], dtype=int)
        return np.where(self.mesh.voxel_part_types == PART_TYPE_CORE)[0]

    def _get_core_contact_cast_indices(self) -> np.ndarray:
        part_types = self.mesh.voxel_part_types
        neighbors = self.mesh.voxel_neighbors
        if part_types is None or neighbors is None:
            return np.array([], dtype=int)
        part_types = np.asarray(part_types, dtype=np.int32)
        if not np.any(part_types == PART_TYPE_CORE):
            return np.array([], dtype=int)
        neighbors = np.asarray(neighbors)
        touches_core = np.zeros(len(part_types), dtype=bool)
        chunk = 100_000
        for start in range(0, len(part_types), chunk):
            nbr = neighbors[start:start + chunk]
            nbr_types = np.where(nbr >= 0, part_types[np.maximum(nbr, 0)], -1)
            touches_core[start:start + chunk] = np.any(nbr_types == PART_TYPE_CORE, axis=1)
        return np.flatnonzero(touches_core & (part_types == PART_TYPE_CAST))

    def _get_orphan_metal_mask(self) -> np.ndarray:
        cached = getattr(self, "_orphan_metal_mask", None)
        if cached is not None:
            return cached

        part_types = getattr(self.mesh, "voxel_part_types", None)
        neighbors = getattr(self.mesh, "voxel_neighbors", None)
        if part_types is not None:
            n_voxels = len(part_types)
        else:
            centers = getattr(self.mesh, "voxel_centers", None)
            n_voxels = len(centers) if centers is not None else 0
        orphan = np.zeros(n_voxels, dtype=bool)
        if part_types is None or neighbors is None:
            self._orphan_metal_mask = orphan
            return orphan

        part_types = np.asarray(part_types)
        metal = is_metal(part_types)
        n_metal = int(np.sum(metal))
        if n_metal == 0:
            self._orphan_metal_mask = orphan
            return orphan

        outside_boxes = np.zeros(n_voxels, dtype=bool)
        metal_bounds = getattr(self.mesh, "metal_part_bounds", None)
        centers = getattr(self.mesh, "voxel_centers", None)
        sizes = getattr(self.mesh, "voxel_sizes", None)
        if metal_bounds and centers is not None:
            centers_arr = np.asarray(centers)
            pad = 0.5 * float(np.max(np.asarray(sizes))) if sizes is not None else 0.0
            inside_any = np.zeros(n_voxels, dtype=bool)
            for box in metal_bounds:
                lo, hi = np.asarray(box[0]), np.asarray(box[1])
                inside_any |= np.all(
                    (centers_arr >= lo - pad) & (centers_arr <= hi + pad), axis=1
                )
            outside_boxes = metal & ~inside_any

        nbr = np.asarray(neighbors)
        valid_nbr = nbr >= 0
        nbr_types = np.where(valid_nbr, part_types[np.maximum(nbr, 0)], -1)
        n_chillcore = ((nbr_types == 2) | (nbr_types == 4)).sum(axis=1)
        n_metal_nb = (
            (nbr_types == 0) | (nbr_types == 1) | (nbr_types == 3)
        ).sum(axis=1)
        smothered = metal & (n_chillcore >= 8) & (n_chillcore >= n_metal_nb)

        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        core_metal = metal & ~outside_boxes & ~smothered
        disconnected = np.zeros(n_voxels, dtype=bool)
        n_pockets = 0
        if np.any(core_metal):
            rows = np.repeat(np.arange(n_voxels), nbr.shape[1])
            cols = nbr.ravel()
            valid = (cols >= 0) & core_metal[rows] & core_metal[np.maximum(cols, 0)]
            rows, cols = rows[valid], cols[valid]
            graph = coo_matrix(
                (np.ones(len(rows), dtype=np.int8), (rows, cols)),
                shape=(n_voxels, n_voxels),
            )
            _, labels = connected_components(graph, directed=False)
            unique, counts = np.unique(labels[core_metal], return_counts=True)
            largest = int(counts.max())
            threshold = max(2.0, 0.05 * largest)
            keep = set(unique[(counts >= threshold) | (counts == largest)].tolist())
            disconnected = core_metal & ~np.isin(labels, list(keep))
            n_pockets = len(unique) - len(keep)

        orphan = outside_boxes | smothered | disconnected

        n_orphan = int(np.sum(orphan))
        if n_orphan > 0:
            detail = (
                f"{int(outside_boxes.sum()):,} outside metal parts, "
                f"{int((smothered & ~outside_boxes).sum()):,} chill-smothered, "
                f"{int(disconnected.sum()):,} disconnected ({n_pockets} pocket(s))"
            )
            print(
                f"  NOTE: excluding {n_orphan:,} phantom metal voxels "
                f"(voxelization gap-fill around chills/gaps: {detail}) "
                "from defect scoring and liquid displays."
            )
        self._orphan_metal_mask = orphan
        return orphan

    def _get_orphan_metal_indices(self) -> np.ndarray:
        return np.flatnonzero(self._get_orphan_metal_mask())

    def _get_riser_post_voxel_indices(self) -> np.ndarray:
        if self.mesh.voxel_part_types is None:
            return np.array([], dtype=int)
        return np.where(self.mesh.voxel_part_types == PART_TYPE_RISER_POST)[0]

    def _get_riser_voxel_indices(self) -> np.ndarray:
        if self.mesh.voxel_part_types is None:
            return np.array([], dtype=int)
        return np.where(self.mesh.voxel_part_types == PART_TYPE_RISER)[0]

    def _get_feeder_voxel_indices(self) -> np.ndarray:
        if self.mesh.voxel_part_types is None:
            return np.array([], dtype=int)
        return np.where(
            is_feeder(self.mesh.voxel_part_types)
        )[0]

    def _get_connected_feeder_mask(self) -> np.ndarray:
        cached = getattr(self, "_connected_feeder_mask_cache", None)
        if cached is not None:
            return cached

        n_voxels = len(self.mesh.voxel_centers)
        connected = np.zeros(n_voxels, dtype=bool)
        feeder_indices = self._get_feeder_voxel_indices()
        riser_indices = self._get_riser_voxel_indices()
        if len(feeder_indices) == 0 or len(riser_indices) == 0:
            return connected

        feeder_mask = np.zeros(n_voxels, dtype=bool)
        feeder_mask[feeder_indices] = True
        connected_dist = self._compute_graph_distance_to_riser(feeder_mask)
        connected[feeder_mask] = np.isfinite(connected_dist[feeder_mask])
        self._connected_feeder_mask_cache = connected
        return connected

    def _group_risers_by_component(self) -> List[np.ndarray]:
        riser_indices = self._get_riser_voxel_indices()
        if len(riser_indices) == 0:
            return []

        riser_set = set(riser_indices)
        visited = set()
        components = []

        for start_idx in riser_indices:
            if start_idx in visited:
                continue

            component = []
            queue: deque[int] = deque([start_idx])

            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)

                for ni in range(self.mesh.voxel_neighbors.shape[1]):
                    neighbor = self.mesh.voxel_neighbors[current, ni]
                    if neighbor >= 0 and neighbor in riser_set and neighbor not in visited:
                        queue.append(neighbor)

            if component:
                components.append(np.array(component, dtype=np.int64))

        return components

    def _calculate_riser_centerlines(self) -> Dict[int, Tuple[float, float, float, float, float]]:
        components = self._group_risers_by_component()
        if not components:
            return {}

        voxel_to_centerline = {}

        for riser_indices in components:
            if len(riser_indices) == 0:
                continue

            riser_centers = self.mesh.voxel_centers[riser_indices]

            center_x = np.mean(riser_centers[:, 0])
            center_y = np.mean(riser_centers[:, 1])

            xy_distances = np.sqrt(
                (riser_centers[:, 0] - center_x) ** 2 +
                (riser_centers[:, 1] - center_y) ** 2
            )
            radius = np.max(xy_distances) if len(xy_distances) > 0 else 1.0

            z_min = np.min(riser_centers[:, 2])
            z_max = np.max(riser_centers[:, 2])

            for idx in riser_indices:
                voxel_to_centerline[idx] = (center_x, center_y, radius, z_min, z_max)

        return voxel_to_centerline

    def _calculate_chill_influence_thicknesses(
        self,
        chill_indices: np.ndarray,
        raw_thicknesses: np.ndarray,
    ) -> np.ndarray:
        raw = np.asarray(raw_thicknesses, dtype=np.float32)
        if raw.size == 0:
            return raw

        base_voxel_size = max(float(getattr(self.mesh, "base_voxel_size", 0.0)), 1e-6)
        component_floor = max(float(np.max(raw)) if np.any(raw > 0.0) else 0.0, base_voxel_size)
        effective = np.maximum(raw, component_floor).astype(np.float32, copy=True)

        neighbors = getattr(self.mesh, "voxel_neighbors", None)
        if neighbors is not None and len(chill_indices) == len(raw):
            chill_indices_arr = np.asarray(chill_indices, dtype=np.int64)
            local_by_global = {int(global_idx): local_idx for local_idx, global_idx in enumerate(chill_indices_arr)}
            visited = np.zeros(len(chill_indices_arr), dtype=bool)

            for start_local in range(len(chill_indices_arr)):
                if visited[start_local]:
                    continue

                stack = [start_local]
                component: List[int] = []
                visited[start_local] = True

                while stack:
                    local_idx = stack.pop()
                    component.append(local_idx)
                    global_idx = int(chill_indices_arr[local_idx])
                    for neighbor_idx in np.asarray(neighbors[global_idx], dtype=np.int64):
                        if neighbor_idx < 0:
                            continue
                        neighbor_local = local_by_global.get(int(neighbor_idx))
                        if neighbor_local is not None and not visited[neighbor_local]:
                            visited[neighbor_local] = True
                            stack.append(neighbor_local)

                component_arr = np.asarray(component, dtype=np.int64)
                component_max = float(np.max(raw[component_arr])) if component_arr.size else 0.0
                component_radius = max(component_max, base_voxel_size)
                effective[component_arr] = np.maximum(raw[component_arr], component_radius)

        return (effective + base_voxel_size).astype(np.float32, copy=False)

    def _calculate_chill_influence(self) -> np.ndarray:
        n_voxels = len(self.mesh.voxel_centers)

        chill_indices = self._get_chill_voxel_indices()
        if len(chill_indices) == 0:
            return np.ones(n_voxels, dtype=np.float32)

        if self._chill_centers is None:
            self._chill_centers = np.ascontiguousarray(self.mesh.voxel_centers[chill_indices], dtype=np.float32)

            chill_mask = (self.mesh.voxel_part_types == PART_TYPE_CHILL)
            chill_surface_mask = chill_mask & self.mesh.surface_voxels
            chill_surface_indices = np.where(chill_surface_mask)[0]

            if len(chill_surface_indices) == 0:
                chill_surface_indices = chill_indices

            self._chill_surface_centers = np.ascontiguousarray(self.mesh.voxel_centers[chill_surface_indices], dtype=np.float32)

            print(f"  Calculating spatially-varying chill thickness (EDT)...")
            self._chill_thicknesses = compute_chill_internal_thickness_parallel(
                self._chill_centers, self._chill_surface_centers
            )
            self._chill_influence_thicknesses = self._calculate_chill_influence_thicknesses(
                chill_indices,
                self._chill_thicknesses,
            )

            max_t = np.max(self._chill_thicknesses)
            avg_t = np.mean(self._chill_thicknesses)
            max_influence_t = np.max(self._chill_influence_thicknesses)
            print(f"    Total chill voxels: {len(chill_indices):,}")
            print(f"    Varying thickness: max={max_t:.2f}\", avg={avg_t:.2f}\"")
            print(f"    Effective influence radius: max={max_influence_t:.2f}\"")

        cp = self.analysis_options.solidification_params.chill
        max_chill_boost = cp.max_chill_boost
        voxel_centers = np.ascontiguousarray(self.mesh.voxel_centers, dtype=np.float32)
        voxel_states = np.ascontiguousarray(self.mesh.voxel_states, dtype=np.float32)

        if self.mesh.voxel_part_types is not None:
            part_types = np.ascontiguousarray(self.mesh.voxel_part_types, dtype=np.int32)
        else:
            part_types = np.zeros(n_voxels, dtype=np.int32)

        if self._chill_influence_thicknesses is None:
            if self._chill_thicknesses is None:
                return np.ones(n_voxels, dtype=np.float32)
            self._chill_influence_thicknesses = self._calculate_chill_influence_thicknesses(
                chill_indices,
                self._chill_thicknesses,
            )
        if self._chill_centers is None or self._chill_influence_thicknesses is None:
            return np.ones(n_voxels, dtype=np.float32)

        self._chill_boost = compute_dynamic_chill_boost_parallel(
            voxel_centers, self._chill_centers, self._chill_influence_thicknesses,
            voxel_states, part_types, self.time, max_chill_boost
        )

        return self._chill_boost

    def compute_initial_chill_influence(self) -> np.ndarray:
        n_voxels = len(self.mesh.voxel_centers)
        chill_indices = self._get_chill_voxel_indices()
        if len(chill_indices) == 0:
            return np.ones(n_voxels, dtype=np.float32)

        if self._chill_centers is None or self._chill_influence_thicknesses is None:
            saved_boost = self._chill_boost
            saved_time = self.time
            saved_states = np.array(self.mesh.voxel_states, copy=True)
            try:
                self.time = 0.0
                self.mesh.voxel_states[:] = 0.0
                result = np.asarray(self._calculate_chill_influence(), dtype=np.float32).copy()
            finally:
                self.time = saved_time
                self.mesh.voxel_states[:] = saved_states
                self._chill_boost = saved_boost
            return result

        cp = self.analysis_options.solidification_params.chill
        max_chill_boost = cp.max_chill_boost
        voxel_centers = np.ascontiguousarray(self.mesh.voxel_centers, dtype=np.float32)
        liquid_states = np.zeros(n_voxels, dtype=np.float32)
        if self.mesh.voxel_part_types is not None:
            part_types = np.ascontiguousarray(self.mesh.voxel_part_types, dtype=np.int32)
        else:
            part_types = np.zeros(n_voxels, dtype=np.int32)

        return np.asarray(
            compute_dynamic_chill_boost_parallel(
                voxel_centers, self._chill_centers, self._chill_influence_thicknesses,
                liquid_states, part_types, 0.0, max_chill_boost
            ),
            dtype=np.float32,
        )

    def _update_dynamic_chill_boost(self):
        if self._chill_centers is None:
            return
        if self._chill_influence_thicknesses is None:
            chill_indices = self._get_chill_voxel_indices()
            if self._chill_thicknesses is None or len(chill_indices) == 0:
                return
            self._chill_influence_thicknesses = self._calculate_chill_influence_thicknesses(
                chill_indices,
                self._chill_thicknesses,
            )

        n_voxels = len(self.mesh.voxel_centers)
        cp = self.analysis_options.solidification_params.chill
        max_chill_boost = cp.max_chill_boost
        voxel_centers = np.ascontiguousarray(self.mesh.voxel_centers, dtype=np.float32)
        voxel_states = np.ascontiguousarray(self.mesh.voxel_states, dtype=np.float32)

        if self.mesh.voxel_part_types is not None:
            part_types = np.ascontiguousarray(self.mesh.voxel_part_types, dtype=np.int32)
        else:
            part_types = np.zeros(n_voxels, dtype=np.int32)

        self._chill_boost[:] = compute_dynamic_chill_boost_parallel(
            voxel_centers, self._chill_centers, self._chill_influence_thicknesses,
            voxel_states, part_types, self.time, max_chill_boost
        )

    def _update_mold_air_gap_effect(self):
        if self._air_gap_slowdown is None:
            return

        if self.surface_distances is None:
            return

        voxel_states = np.ascontiguousarray(self.mesh.voxel_states, dtype=np.float32)
        surface_distances = np.ascontiguousarray(self.surface_distances, dtype=np.float32)
        surface_mask = np.ascontiguousarray(self.mesh.surface_voxels, dtype=np.bool_)

        agp = self.analysis_options.solidification_params.air_gap
        air_gap_time_constant = agp.time_constant
        surface_influence_depth = agp.surface_influence_depth
        max_slowdown = agp.max_slowdown

        self._air_gap_slowdown[:] = compute_mold_air_gap_effect_parallel(
            voxel_states,
            surface_distances,
            surface_mask,
            self.time,
            air_gap_time_constant,
            surface_influence_depth,
            max_slowdown
        )

    def _calculate_riser_influence(self) -> np.ndarray:
        n_voxels = len(self.mesh.voxel_centers)
        riser_slowdown = np.ones(n_voxels, dtype=np.float32)

        riser_indices = self._get_riser_voxel_indices()
        if len(riser_indices) == 0:
            return riser_slowdown

        riser_centers = self.mesh.voxel_centers[riser_indices]

        ep = self.analysis_options.solidification_params.exothermic
        ignition_delay_s = ep.ignition_delay
        burn_duration_s = ep.burn_duration
        insulation_only_factor = ep.insulation_only_factor
        full_exothermic_factor = ep.full_exothermic_factor
        decay_time_constant = ep.decay_time_constant

        current_time = self.time

        if current_time < ignition_delay_s:
            ignition_progress = current_time / ignition_delay_s
            exothermic_factor = 1.0 - (1.0 - insulation_only_factor) * ignition_progress
        elif current_time < (ignition_delay_s + burn_duration_s):
            exothermic_factor = full_exothermic_factor
        else:
            time_since_burnout = current_time - (ignition_delay_s + burn_duration_s)
            decay_factor = np.exp(-time_since_burnout / decay_time_constant)

            exothermic_factor = insulation_only_factor + \
                (full_exothermic_factor - insulation_only_factor) * decay_factor

        fp = self.analysis_options.solidification_params.feeding
        tp = self.analysis_options.solidification_params.timing
        base_feeding_distance = fp.base_feeding_distance
        material_factor = self.alloy.feeding_distance_multiplier / 3.0
        riser_influence_radius = base_feeding_distance * material_factor
        min_influence_factor = fp.min_influence_factor

        if self.time < self.time_step:
            print(f"  Time-dependent exothermic model active:")
            print(f"    Ignition delay: {ignition_delay_s:.0f}s, Burn duration: {burn_duration_s:.0f}s")
            print(f"  Applying riser thermal influence to {len(riser_indices):,} riser voxels...")
            print(f"  Riser influence radius: {riser_influence_radius:.1f}\" (feeding distance)")

        riser_slowdown[riser_indices] = exothermic_factor

        riser_post_indices = np.where(self.mesh.voxel_part_types == PART_TYPE_RISER_POST)[0]
        if len(riser_post_indices) > 0:
            sand_saturation_time_constant = tp.sand_saturation_time
            max_hot_sand_slowdown = tp.max_hot_sand_slowdown

            sand_saturation_factor = 1.0 - np.exp(-self.time / sand_saturation_time_constant)

            post_slowdown = 1.0 - (1.0 - max_hot_sand_slowdown) * sand_saturation_factor

            riser_slowdown[riser_post_indices] = post_slowdown

            if self.time < self.time_step:
                print(f"  Hot sand saturation active for {len(riser_post_indices):,} RISER_POST voxels")

        cast_indices = np.where(self.mesh.voxel_part_types == PART_TYPE_CAST)[0]

        if len(cast_indices) > 0:
            riser_tree = _build_kdtree(riser_centers)

            cast_centers = self.mesh.voxel_centers[cast_indices]

            distances, _ = riser_tree.query(cast_centers, k=1)

            cast_indices_arr = np.ascontiguousarray(cast_indices, dtype=np.int64)
            distances_arr = np.ascontiguousarray(distances, dtype=np.float32)

            influenced_count = compute_riser_influence_parallel(
                riser_slowdown,
                cast_indices_arr,
                distances_arr,
                np.float32(riser_influence_radius),
                np.float32(exothermic_factor),
                np.float32(min_influence_factor)
            )

            if self.time < self.time_step and influenced_count > 0:
                print(f"    Riser feeding influence extends to {influenced_count:,} casting voxels")
                print(f"    Slowdown range: {exothermic_factor:.0%} to {min_influence_factor:.0%} of normal rate")

        return riser_slowdown

    def _update_riser_exothermic_effect(self, riser_slowdown: np.ndarray):
        riser_indices = self._get_riser_voxel_indices()
        if len(riser_indices) == 0:
            return

        new_slowdown = self._calculate_riser_influence()
        riser_slowdown[:] = new_slowdown

    def _calculate_chill_shrinkage_reduction(self) -> np.ndarray:
        n_voxels = len(self.mesh.voxel_centers)
        chill_reduction = np.ones(n_voxels, dtype=np.float32)

        chill_indices = self._get_chill_voxel_indices()
        if len(chill_indices) == 0:
            return chill_reduction

        chill_centers = self.mesh.voxel_centers[chill_indices]

        cp = self.analysis_options.solidification_params.chill
        if self._chill_thicknesses is not None and len(self._chill_thicknesses) > 0:
            chill_thickness = np.max(self._chill_thicknesses)
        else:
            chill_thickness = self.mesh.base_voxel_size * cp.thickness_fallback_voxels

        chill_influence_radius = chill_thickness * cp.shrinkage_radius_multiplier

        max_reduction = cp.shrinkage_max_reduction
        min_reduction = cp.shrinkage_min_reduction

        print(f"  Chill shrinkage reduction zone: {chill_influence_radius:.2f}\" (1.5 × {chill_thickness:.2f}\" thickness)")

        chill_tree = _build_kdtree(chill_centers)

        if self.mesh.voxel_part_types is not None:
            cast_mask = self.mesh.voxel_part_types == PART_TYPE_CAST
        else:
            cast_mask = np.ones(n_voxels, dtype=bool)

        cast_indices = np.where(cast_mask)[0]

        distances = np.empty(0, dtype=np.float64)
        if len(cast_indices) > 0:
            cast_centers = self.mesh.voxel_centers[cast_indices]

            query_start = time.perf_counter()
            distances, _ = chill_tree.query(
                cast_centers,
                k=1,
                distance_upper_bound=chill_influence_radius,
                workers=-1,
            )
            self.stage_timings["chill_shrinkage_query"] = (
                self.stage_timings.get("chill_shrinkage_query", 0.0) + time.perf_counter() - query_start
            )

            within_radius = distances < chill_influence_radius
            if np.any(within_radius):
                normalized_dist = distances[within_radius] / chill_influence_radius
                reduction = max_reduction + (min_reduction - max_reduction) * (normalized_dist * np.sqrt(normalized_dist))
                chill_reduction[cast_indices[within_radius]] = reduction

        return chill_reduction

    def _calculate_riser_feeding_reduction(self) -> np.ndarray:
        stage_start = time.perf_counter()
        stage_timings = self.stage_timings
        n_voxels = len(self.mesh.voxel_centers)
        riser_feeding_reduction = np.ones(n_voxels, dtype=np.float32)

        riser_indices = self._get_riser_voxel_indices()
        print(f"  DEBUG: Found {len(riser_indices)} riser voxels for feeding calculation")

        if len(riser_indices) == 0:
            print(f"  DEBUG: No riser voxels found - voxel_part_types exists: {self.mesh.voxel_part_types is not None}")
            if self.mesh.voxel_part_types is not None:
                unique_types = np.unique(self.mesh.voxel_part_types)
                print(f"  DEBUG: Unique part types: {unique_types}")
            return riser_feeding_reduction

        fp = self.analysis_options.solidification_params.feeding
        base_feeding_distance = fp.base_feeding_distance
        material_factor = self.alloy.feeding_distance_multiplier / 3.0
        feeding_distance = base_feeding_distance * material_factor

        print(f"  Riser feeding distance: {feeding_distance:.1f}\" (material factor: {material_factor:.2f})")

        max_reduction = fp.riser_max_reduction
        min_reduction = fp.riser_min_reduction

        if self.mesh.voxel_part_types is not None:
            part_types = np.asarray(self.mesh.voxel_part_types, dtype=np.int32)
            cast_mask = part_types == PART_TYPE_CAST
            riser_post_mask = part_types == PART_TYPE_RISER_POST
        else:
            cast_mask = np.ones(n_voxels, dtype=bool)
            riser_post_mask = np.zeros(n_voxels, dtype=bool)

        target_indices = np.where(cast_mask | riser_post_mask)[0]
        connected_feeder_mask = self._get_connected_feeder_mask()
        path_lengths = getattr(self, "feeding_path_lengths", None)
        if path_lengths is not None:
            path_lengths = np.asarray(path_lengths, dtype=np.float32)

        if len(target_indices) > 0:
            post_contact_reduction = fp.post_contact_reduction

            fed_cast_count = 0
            fed_post_count = 0
            if path_lengths is None:
                riser_centers = self.mesh.voxel_centers[riser_indices]
                riser_tree = _build_kdtree(riser_centers)
                target_centers = self.mesh.voxel_centers[target_indices]
                distances, _ = riser_tree.query(target_centers, k=1)
            else:
                distances = path_lengths[target_indices]

            target_indices_arr = np.asarray(target_indices, dtype=np.int64)
            distances_arr = np.asarray(distances, dtype=np.float32)
            riser_post_targets = riser_post_mask[target_indices_arr]
            connected_targets = connected_feeder_mask[target_indices_arr]
            in_range = (
                (distances_arr >= 0.0)
                & (distances_arr < feeding_distance)
            )

            if np.any(in_range):
                valid_targets = target_indices_arr[in_range]
                valid_distances = distances_arr[in_range]
                valid_post = riser_post_targets[in_range]
                valid_connected = connected_targets[in_range]
                normalized_dist = valid_distances / feeding_distance
                base_contact = np.where(
                    valid_post,
                    post_contact_reduction,
                    max_reduction,
                )
                reduction = base_contact + (min_reduction - base_contact) * (normalized_dist ** 2)

                feeder_neck_mask = valid_post & (~valid_connected)
                if np.any(feeder_neck_mask):
                    reduction[feeder_neck_mask] = np.maximum(
                        reduction[feeder_neck_mask],
                        fp.disconnected_neck_floor,
                    )

                riser_feeding_reduction[valid_targets] = reduction
                fed_post_count = int(np.count_nonzero(valid_post))
                fed_cast_count = int(np.count_nonzero(~valid_post))

            if fed_cast_count > 0 or fed_post_count > 0:
                print(
                    "  Riser feeding reduces shrinkage in "
                    f"{fed_cast_count:,} casting voxels and {fed_post_count:,} feeder-neck voxels"
                )
                print(
                    "  Feeding reduction range: "
                    f"{max_reduction:.0%} at cast contact, {post_contact_reduction:.0%} at feeder-neck contact"
                )

        stage_timings["riser_feeding_reduction"] = (
            stage_timings.get("riser_feeding_reduction", 0.0) + time.perf_counter() - stage_start
        )
        return riser_feeding_reduction

    def _calculate_ferrostatic_pressure(self) -> np.ndarray:
        voxel_heights = self.mesh.voxel_centers[:, 2]
        n_voxels = len(voxel_heights)

        riser_indices = self._get_riser_voxel_indices()

        if len(riser_indices) > 0:
            max_riser_height = float(np.max(voxel_heights[riser_indices]))
        else:
            max_riser_height = float(np.max(voxel_heights))

        head_heights = np.full(n_voxels, max_riser_height, dtype=np.float32)
        feeding_result = getattr(self, "feeding_path_result", None)
        feeding_source = getattr(feeding_result, "feeding_source", None)
        if feeding_source is not None and len(feeding_source) == n_voxels and len(riser_indices) > 0:
            source_top_heights = np.full(n_voxels, np.nan, dtype=np.float32)
            riser_groups = self._group_risers_by_component()
            for riser_group in riser_groups:
                group_indices = np.asarray(riser_group, dtype=np.int64)
                group_indices = group_indices[
                    (group_indices >= 0) & (group_indices < n_voxels)
                ]
                if len(group_indices) == 0:
                    continue
                group_top = float(np.max(voxel_heights[group_indices]))
                source_top_heights[group_indices] = group_top

            missing_riser_tops = np.asarray(riser_indices, dtype=np.int64)
            missing_riser_tops = missing_riser_tops[
                (missing_riser_tops >= 0) & (missing_riser_tops < n_voxels)
            ]
            missing_riser_tops = missing_riser_tops[
                np.isnan(source_top_heights[missing_riser_tops])
            ]
            source_top_heights[missing_riser_tops] = voxel_heights[missing_riser_tops]

            source_indices = np.asarray(feeding_source, dtype=np.int64)
            valid_source = (
                (source_indices >= 0)
                & (source_indices < n_voxels)
                & np.isfinite(source_top_heights[np.clip(source_indices, 0, n_voxels - 1)])
            )
            if np.any(valid_source):
                safe_sources = np.clip(source_indices[valid_source], 0, n_voxels - 1)
                head_heights[valid_source] = source_top_heights[safe_sources]

        g = GRAVITY_M_S2

        height_diff_meters = np.maximum(
            0.0, inches_to_meters(head_heights - voxel_heights)
        )
        ferrostatic_pressure = (
            self.alloy.density_liquid * g * height_diff_meters
        ).astype(np.float32)

        return ferrostatic_pressure

    def _should_capture_fraction_history(self) -> bool:
        return bool(
            self.analysis_options.fraction_liquid
            or self.analysis_options.hot_tear
            or self.analysis_options.shrinkage
            or self.analysis_options.feeding_path
        )

    def _estimate_feeding_reference_times(self) -> Optional[np.ndarray]:
        history = getattr(self, "fraction_liquid_history", None)
        history_times = getattr(self, "fraction_liquid_times", None)
        if not history or not history_times:
            return None

        critical_fs = float(getattr(self.alloy, "critical_fraction_solid", 1.0))
        n_voxels = len(np.asarray(history[0], dtype=np.float32))
        reference_times = np.full(n_voxels, float(history_times[-1]), dtype=np.float32)
        assigned = np.zeros(n_voxels, dtype=bool)

        for snapshot, time_value in zip(history, history_times):
            snapshot_fs = np.asarray(snapshot, dtype=np.float32)
            crossed = (~assigned) & (snapshot_fs >= critical_fs)
            if np.any(crossed):
                reference_times[crossed] = np.float32(time_value)
                assigned[crossed] = True

        return reference_times

    def _ensure_feeding_path_result(
        self,
        fraction_solid: np.ndarray,
    ) -> Optional["FeedingPathResult"]:
        if not FEEDING_PATH_AVAILABLE:
            return None

        riser_indices = self._get_riser_voxel_indices()
        if len(riser_indices) == 0:
            return None

        cache_key = id(fraction_solid)
        cached = getattr(self, "_feeding_path_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        analyzer = FeedingPathAnalyzer(self.mesh, self.alloy)
        history = getattr(self, "fraction_liquid_history", None)
        history_times = getattr(self, "fraction_liquid_times", None)
        reference_times = self._estimate_feeding_reference_times()
        result = analyzer.analyze(
            np.asarray(riser_indices, dtype=np.int64),
            np.ascontiguousarray(np.clip(fraction_solid, 0.0, 1.0), dtype=np.float64),
            show_progress=False,
            history=history,
            history_times=history_times,
            reference_times=reference_times,
        )
        self.feeding_path_result = result
        self.feeding_quality = result.feeding_quality
        self.feeding_blocked = result.blocked_voxels
        self.feeding_blocked_count = int(np.count_nonzero(result.blocked_voxels))
        self.feeding_open_fraction = result.history_open_fraction
        self.feeding_latest_open_time = result.latest_open_time
        self.feeding_path_lengths = result.path_length.astype(np.float32)
        self._feeding_path_cache = (cache_key, result)
        return result

    def _get_effective_feeding_distance_field(
        self,
        fraction_solid: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        path_lengths = getattr(self, "feeding_path_lengths", None)
        if path_lengths is None and fraction_solid is not None:
            ensure_feeding_path_result = getattr(self, "_ensure_feeding_path_result", None)
            if callable(ensure_feeding_path_result):
                result = ensure_feeding_path_result(fraction_solid)
                if result is not None:
                    path_lengths = result.path_length

        fallback = getattr(self, "surface_distances", None)
        if fallback is None:
            fallback = getattr(self, "feeding_distances", None)

        if path_lengths is not None:
            path_lengths = np.asarray(path_lengths, dtype=np.float32)
            valid = path_lengths >= 0.0
            if np.any(valid):
                effective = np.asarray(fallback, dtype=np.float32).copy()
                effective[valid] = path_lengths[valid]
                return effective

        return np.asarray(fallback, dtype=np.float32)

    def _estimate_feeder_neck_resistance_factors(self) -> np.ndarray:
        part_types = getattr(self.mesh, "voxel_part_types", None)
        neighbors = getattr(self.mesh, "voxel_neighbors", None)
        if part_types is None or neighbors is None:
            return np.ones(len(self.mesh.voxel_centers), dtype=np.float32)

        part_types_arr = np.asarray(part_types, dtype=np.int32)
        neighbors_arr = np.asarray(neighbors, dtype=np.int32)
        riser_post_mask = part_types_arr == 3

        n_voxels = len(part_types_arr)
        factors = np.ones(n_voxels, dtype=np.float32)
        factors[riser_post_mask] = 1.20

        cast_mask = part_types_arr == 0
        cast_indices = np.flatnonzero(cast_mask)
        if len(cast_indices) == 0:
            return factors

        valid_neighbors = np.where(neighbors_arr >= 0, neighbors_arr, 0)
        neighbor_is_post = riser_post_mask[valid_neighbors] & (neighbors_arr >= 0)
        any_post_neighbor = np.any(neighbor_is_post, axis=1)
        factors[cast_indices[any_post_neighbor[cast_indices]]] = 1.35

        return factors

    def _calculate_feeding_effectiveness(self) -> np.ndarray:
        stage_start = time.perf_counter()
        stage_timings = getattr(self, "stage_timings", {})
        n_voxels = len(self.mesh.voxel_centers)
        feeding_effectiveness = np.ones(n_voxels, dtype=np.float32)
        darcy_margin = np.ones(n_voxels, dtype=np.float32)

        critical_fs = self.alloy.critical_fraction_solid
        transient_fraction_solid = getattr(self, "feeding_fraction_solid", None)
        connected_feeder_mask = self._get_connected_feeder_mask()
        feeding_path_result = self._ensure_feeding_path_result(
            transient_fraction_solid
        ) if transient_fraction_solid is not None else None
        ferrostatic_pressure = self._calculate_ferrostatic_pressure()
        effective_feeding_distances = SolidificationEngine._get_effective_feeding_distance_field(
            self,
            transient_fraction_solid,
        )
        feeder_neck_factors = np.asarray(
            self._estimate_feeder_neck_resistance_factors(), dtype=np.float32
        )
        local_temperatures = getattr(self, "_thermal_latest_temperatures", None)
        local_cooling_rates = getattr(self, "cooling_rates", None)
        local_sdas = getattr(self, "sdas", None)

        beta = float(self.alloy.density_change_ratio - 1.0)
        local_gradients = getattr(self, "thermal_gradients", None)
        if (
            local_cooling_rates is not None
            and local_gradients is not None
            and len(local_cooling_rates) == n_voxels
            and len(local_gradients) == n_voxels
        ):
            grad_c_per_m = np.asarray(local_gradients, dtype=np.float64) * 1000.0
            front_speed = np.asarray(local_cooling_rates, dtype=np.float64) / np.maximum(
                grad_c_per_m, 1e-6
            )
            velocity_arr = np.clip(beta * front_speed, 1e-6, 1e-2)
        else:
            velocity_arr = np.full(n_voxels, 5.0e-4, dtype=np.float64)

        print(f"  Calculating feeding effectiveness (P_ferro vs ΔP_darcy)...")
        print(f"    Material: {self.alloy.name}")
        print(f"    Critical fraction solid: {critical_fs:.2f}")

        path_length_arr = None
        history_open_arr = None
        if feeding_path_result is not None:
            path_length_arr = np.asarray(
                feeding_path_result.path_length, dtype=np.float32
            )
            hof = getattr(feeding_path_result, "history_open_fraction", None)
            if hof is not None:
                history_open_arr = np.asarray(hof, dtype=np.float32)

        fraction_solid_arr = np.where(
            transient_fraction_solid is not None,
            np.where(np.isnan(transient_fraction_solid), self.mesh.voxel_states, transient_fraction_solid),
            self.mesh.voxel_states,
        )
        fraction_solid_arr = np.clip(fraction_solid_arr, 0.0, 1.0)

        feeding_dist_arr = effective_feeding_distances.copy()
        if path_length_arr is not None:
            use_path = path_length_arr >= 0.0
            np.copyto(feeding_dist_arr, path_length_arr, where=use_path)
            blocked_mask = (~use_path) & (feeding_dist_arr >= 0.001) & (~connected_feeder_mask)
            feeding_effectiveness[blocked_mask] = 0.0
            darcy_margin[blocked_mask] = -1.0

        surface_mask = feeding_dist_arr < 0.001
        connected_mask = connected_feeder_mask

        fully_fed_mask = surface_mask | connected_mask
        fully_fed_count = int(np.count_nonzero(fully_fed_mask))
        feeding_effectiveness[fully_fed_mask] = 1.0
        darcy_margin[fully_fed_mask] = 1.0

        process_mask = ~fully_fed_mask

        if np.any(process_mask):
            idx_arr = np.where(process_mask)[0]
            fs_proc = fraction_solid_arr[idx_arr]
            dist_proc = feeding_dist_arr[idx_arr]

            if local_temperatures is not None:
                local_temp_arr = local_temperatures[idx_arr]
            else:
                local_temp_arr = self.alloy.liquidus_temp - (
                    fs_proc * max(self.alloy.freezing_range, 1.0)
                )

            vis_arr = self.alloy.vectorized_dynamic_viscosity(
                local_temp_arr, fs_proc
            ).astype(np.float64)

            if local_sdas is not None:
                sdas_arr = np.asarray(local_sdas[idx_arr], dtype=np.float64)
            elif local_cooling_rates is not None:
                sdas_arr = self.alloy.vectorized_sdas(
                    local_cooling_rates[idx_arr]
                ).astype(np.float64)
            else:
                sdas_arr = np.full(
                    len(idx_arr),
                    float(self.alloy.dendrite_arm_spacing),
                    dtype=np.float64,
                )

            perm_arr = self.alloy.vectorized_permeability(fs_proc, sdas_arr)
            dist_m_arr = inches_to_meters(dist_proc)
            pd_arr = (
                vis_arr * dist_m_arr * velocity_arr[idx_arr] / np.maximum(perm_arr, 1e-20)
            )
            pd_arr = np.where(perm_arr < 1e-20, 1e10, pd_arr) * feeder_neck_factors[idx_arr]

            p_ferro_proc = ferrostatic_pressure[idx_arr]

            pressure_ratio = np.where(pd_arr > 0, p_ferro_proc / pd_arr, np.inf)

            darcy_margin[idx_arr] = np.clip(
                (p_ferro_proc - pd_arr) / np.maximum(p_ferro_proc + pd_arr, 1e-9),
                -1.0,
                1.0,
            ).astype(np.float32)

            feeding_eff_proc = np.ones(len(idx_arr), dtype=np.float32)
            gt2 = pressure_ratio >= 2.0
            gte1_lt2 = (pressure_ratio >= 1.0) & (pressure_ratio < 2.0)
            gte05_lt1 = (pressure_ratio >= 0.5) & (pressure_ratio < 1.0)
            lt05 = pressure_ratio < 0.5

            feeding_eff_proc[gt2] = 1.0
            feeding_eff_proc[gte1_lt2] = (0.5 + 0.5 * (pressure_ratio[gte1_lt2] - 1.0)).astype(np.float32)
            feeding_eff_proc[gte05_lt1] = pressure_ratio[gte05_lt1].astype(np.float32)
            feeding_eff_proc[lt05] = np.clip(pressure_ratio[lt05] * 2.0, 0.1, 1.0).astype(np.float32)

            if history_open_arr is not None:
                open_frac = history_open_arr[idx_arr]
                can_scale = (dist_m_arr >= 0.001) & (~connected_feeder_mask[idx_arr])
                feeding_eff_proc[can_scale] *= (0.4 + 0.6 * open_frac[can_scale]).astype(np.float32)
                low_open = can_scale & (open_frac < 0.15) & (feeding_eff_proc < 0.5)
                feeding_eff_proc[low_open] = np.minimum(feeding_eff_proc[low_open], 0.15).astype(np.float32)
                darcy_margin[idx_arr[low_open]] = np.minimum(
                    darcy_margin[idx_arr[low_open]], -0.5
                ).astype(np.float32)

            feeding_effectiveness[idx_arr] = feeding_eff_proc.astype(np.float32)

            partially_fed_count = int(np.count_nonzero(~gt2 & ~surface_mask[idx_arr]))
            blocked_count = int(np.count_nonzero(lt05))

            feed_hist, _ = np.histogram(feeding_effectiveness, bins=[0.0, 0.25, 0.5, 0.75, 1.01])
            print(f"    Fully fed voxels: {fully_fed_count:,}")
            print(f"    Partially fed voxels: {partially_fed_count:,}")
            print(f"    Blocked voxels: {blocked_count:,}")

            print(
                f"    Ferro-static pressure range: {np.min(ferrostatic_pressure):.0f} - "
                f"{np.max(ferrostatic_pressure):.0f} Pa"
            )
            positive_dist = feeding_dist_arr[feeding_dist_arr > 0.0]
            if len(positive_dist) > 0:
                print(
                    f"    Path length range used: {np.min(positive_dist):.3f} - "
                    f"{np.max(positive_dist):.3f} in"
                )
            positive_perm = perm_arr[perm_arr > 0.0]
            if len(positive_perm) > 0:
                print(
                    f"    Permeability range: {np.min(positive_perm):.3e} - "
                    f"{np.max(positive_perm):.3e} m^2"
                )
            positive_drop = pd_arr[pd_arr > 0.0]
            if len(positive_drop) > 0:
                print(
                    f"    Darcy pressure-drop range: {np.min(positive_drop):.0f} - "
                    f"{np.max(positive_drop):.0f} Pa"
                )
            print(
                f"    Feed velocity range: {np.min(velocity_arr[idx_arr]) * 1000.0:.4f} - "
                f"{np.max(velocity_arr[idx_arr]) * 1000.0:.4f} mm/s"
            )
            positive_visc = vis_arr[vis_arr > 0.0]
            if len(positive_visc) > 0:
                print(
                    f"    Dynamic viscosity range: {np.min(positive_visc):.4f} - "
                    f"{np.max(positive_visc):.4f} Pa·s"
                )
            positive_sdas = sdas_arr[sdas_arr > 0.0]
            if len(positive_sdas) > 0:
                print(
                    f"    SDAS range used: {np.min(positive_sdas):.1f} - "
                    f"{np.max(positive_sdas):.1f} μm"
                )
            print(
                "    Feeder-neck resistance factor range: "
                f"{np.min(feeder_neck_factors):.2f} - {np.max(feeder_neck_factors):.2f}"
            )
            print(
                "    Fraction-solid at feeding check: "
                f"min={np.min(fs_proc):.2f}, mean={np.mean(fs_proc):.2f}, max={np.max(fs_proc):.2f}"
            )
            print(
                "    Feeding effectiveness bins [0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0]: "
                f"{feed_hist.tolist()}"
            )
            if history_open_arr is not None:
                print(
                    "    Open-path history fraction: "
                    f"min={np.min(history_open_arr):.2f}, "
                    f"mean={np.mean(history_open_arr):.2f}, "
                    f"max={np.max(history_open_arr):.2f}"
                )
            print(
                "    Darcy margin range: "
                f"{np.min(darcy_margin):.2f} - {np.max(darcy_margin):.2f}"
            )

        stage_timings["feeding_effectiveness"] = (
            stage_timings.get("feeding_effectiveness", 0.0) + time.perf_counter() - stage_start
        )

        self.feeding_effectiveness = feeding_effectiveness.astype(np.float32)
        self.feeding_darcy_margin = darcy_margin.astype(np.float32)

        return feeding_effectiveness

    def _update_solidification_with_part_types(self,
                                               chill_boost: np.ndarray,
                                               riser_slowdown: np.ndarray,
                                               material_factor: float = 1.0) -> np.ndarray:
        heights = self._get_normalized_heights()

        voxel_states = np.ascontiguousarray(self.mesh.voxel_states, dtype=np.float32)
        neighbors = np.ascontiguousarray(self.mesh.voxel_neighbors, dtype=np.int32)
        moduli = np.ascontiguousarray(self.mesh.voxel_moduli, dtype=np.float32)
        heights = np.ascontiguousarray(heights, dtype=np.float32)
        chill_boost = np.ascontiguousarray(chill_boost, dtype=np.float32)
        riser_slowdown = np.ascontiguousarray(riser_slowdown, dtype=np.float32)

        if self.mesh.voxel_part_types is not None:
            part_types = np.ascontiguousarray(self.mesh.voxel_part_types, dtype=np.int32)

            feeding_bias = float(
                self.analysis_options.solidification_params.feeding.feeding_zone_vertical_bias
            )

            new_states = update_solidification_with_feeding_connectivity(
                voxel_states, neighbors, moduli, heights,
                chill_boost, riser_slowdown, part_types, self.time_step,
                np.float32(material_factor),
                np.float32(feeding_bias),
            )
        else:
            new_states = update_solidification_with_part_types_parallel(
                voxel_states, neighbors, moduli, heights,
                chill_boost, riser_slowdown, self.time_step,
                np.float32(material_factor)
            )

        return new_states

    def _estimate_local_section_size(self) -> np.ndarray:
        return self._get_risk_moduli() * 2.0

    def _compute_shrinkage_base_risk(
        self,
        valid_indices: np.ndarray,
        normalized_heights: np.ndarray,
        local_section_size: np.ndarray,
        riser_mask: np.ndarray,
        riser_post_mask: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        heights = normalized_heights[valid_indices]
        moduli = self._get_risk_moduli()[valid_indices]
        thicknesses = local_section_size[valid_indices]
        times = np.asarray(self.solidification_times, dtype=np.float32)
        max_time = getattr(self, "max_time", None)
        if max_time is None or not np.isfinite(max_time) or max_time <= 0:
            max_time = np.nanmax(times) if np.any(np.isfinite(times)) else 0.0
        times = np.where(np.isnan(times), float(max_time), times)
        times = times[valid_indices]
        is_riser = riser_mask[valid_indices]
        is_riser_post = riser_post_mask[valid_indices]

        height_factor = heights ** 2.0
        height_factor = np.where(is_riser_post, heights ** 1.2, height_factor)
        interior_factor = np.minimum(moduli / 1.5, 1.0)
        thickness_factor = np.clip((thicknesses - 0.5) / 1.5, 0.0, 1.0)
        thickness_factor = np.where(
            is_riser,
            np.clip((thicknesses - 0.5) / 1.0, 0.0, 1.0),
            thickness_factor,
        )
        thickness_factor = np.where(
            is_riser_post,
            np.clip((thicknesses - 0.5) / 1.2, 0.0, 1.0),
            thickness_factor,
        )
        time_factor = np.where(times > 0, np.minimum(times / 800.0, 1.0), 0.0)
        time_factor = np.where(is_riser, time_factor * 0.85, time_factor)
        time_factor = np.where(is_riser_post, time_factor * 0.65, time_factor)

        isolation_factor = np.zeros(len(valid_indices), dtype=np.float32)
        if self.hotspot_intensity is not None:
            isolation_factor = np.maximum(
                isolation_factor,
                self.hotspot_intensity[valid_indices],
            )
        feeding_blocked_arr = getattr(self, "feeding_blocked", None)
        feeding_open_arr = getattr(self, "feeding_open_fraction", None)
        if feeding_blocked_arr is not None and len(feeding_blocked_arr) == len(isolation_factor):
            blocked_local = np.asarray(feeding_blocked_arr, dtype=bool)[valid_indices]
            isolation_factor = np.maximum(
                isolation_factor,
                np.where(blocked_local, 0.85, 0.0).astype(np.float32),
            )
        if feeding_open_arr is not None and len(feeding_open_arr) == len(isolation_factor):
            open_local = np.asarray(feeding_open_arr, dtype=np.float32)[valid_indices]
            isolation_factor = np.maximum(
                isolation_factor,
                np.clip(1.0 - open_local, 0.0, 1.0).astype(np.float32),
            )

        base_risk = (
            0.40 * isolation_factor +
            0.20 * height_factor +
            0.20 * interior_factor +
            0.15 * thickness_factor +
            0.05 * time_factor
        )
        base_risk = base_risk ** 1.5
        return (
            base_risk,
            heights,
            moduli,
            thicknesses,
            times,
            is_riser,
            is_riser_post,
            height_factor,
            interior_factor,
            thickness_factor,
            time_factor,
        )

    def _compute_riser_pipe_fraction(self) -> np.ndarray:
        n_voxels = len(self.mesh.voxel_centers)
        fraction = np.zeros(n_voxels, dtype=np.float32)
        self.riser_pipe_report = []
        self.casting_pipe_report = []
        if self.solidification_times is None:
            return fraction
        part_types = self._mesh_part_types()
        groups = self._group_risers_by_component()

        riser_comp = np.full(n_voxels, -1, dtype=np.int64)
        for comp_id, group in enumerate(groups):
            riser_comp[np.asarray(group, dtype=np.int64)] = comp_id

        pool_ids = riser_comp.copy()
        feeding_result = getattr(self, "feeding_path_result", None)
        feeding_source = getattr(feeding_result, "feeding_source", None)
        if groups and feeding_source is not None and len(feeding_source) == n_voxels:
            source = np.asarray(feeding_source, dtype=np.int64)
            fed = (source >= 0) & (source < n_voxels)
            fed &= part_types != PART_TYPE_CHILL
            blocked = getattr(self, "feeding_blocked", None)
            if blocked is not None and len(blocked) == n_voxels:
                fed &= ~np.asarray(blocked, dtype=bool)
            pool_ids[fed] = riser_comp[source[fed]]
            pool_ids[riser_comp >= 0] = riser_comp[riser_comp >= 0]

        scored = is_scored_metal(part_types)
        get_orphans = getattr(self, "_get_orphan_metal_mask", None)
        orphan = get_orphans() if callable(get_orphans) else None
        if orphan is not None and len(orphan) == n_voxels:
            scored &= ~np.asarray(orphan, dtype=bool)
        neighbors = getattr(self.mesh, "voxel_neighbors", None)
        n_riser_pools = len(groups)
        unrisered = np.full(n_voxels, -1, dtype=np.int64)
        n_unrisered = 0
        if neighbors is not None:
            unrisered, n_unrisered = label_unfed_pools(
                neighbors, scored, part_types == PART_TYPE_RISER
            )
            take = (unrisered >= 0) & (pool_ids < 0)
            pool_ids[take] = n_riser_pools + unrisered[take]
        if n_riser_pools + n_unrisered == 0:
            return fraction

        volumes = getattr(self.mesh, "voxel_volumes", None)
        if volumes is None or len(volumes) != n_voxels:
            sizes = np.asarray(self.mesh.voxel_sizes, dtype=np.float64).reshape(-1)
            volumes = np.broadcast_to(sizes, (n_voxels,)) ** 3
        beta = float(self.alloy.feeding_shrinkage) / 100.0
        heights = np.asarray(self.mesh.voxel_centers, dtype=np.float64)[:, 2]
        fraction, pools = gravity_drain_pipe(
            np.asarray(self.solidification_times, dtype=np.float64),
            heights,
            np.asarray(volumes, dtype=np.float64),
            pool_ids,
            riser_comp >= 0,
            beta,
        )
        for comp_id in range(n_riser_pools):
            pool = pools[comp_id]
            group = np.asarray(groups[comp_id], dtype=np.int64)
            piped = group[fraction[group] > 0.5]
            riser_bottom = float(np.min(self.mesh.voxel_centers[group, 2]))
            pipe_bottom = (
                float(np.min(self.mesh.voxel_centers[piped, 2])) if len(piped) else float("nan")
            )
            self.riser_pipe_report.append(
                {
                    "riser": comp_id,
                    "demand_in3": pool.demand_volume,
                    "pipe_in3": pool.pipe_volume,
                    "pipe_in_riser_in3": pool.feeder_pipe_volume,
                    "pipe_in_casting_in3": pool.section_pipe_volume,
                    "pipe_bottom_above_riser_base_in": pipe_bottom - riser_bottom,
                }
            )
            print(
                f"  Riser {comp_id} pipe: demand {pool.demand_volume:.2f} in^3, "
                f"in riser {pool.feeder_pipe_volume:.2f}, in casting {pool.section_pipe_volume:.2f}, "
                f"bottom {pipe_bottom - riser_bottom:.2f} in above riser base"
            )
        by_pool = np.argsort(pool_ids, kind="stable")
        starts = np.searchsorted(pool_ids[by_pool], np.arange(n_riser_pools + n_unrisered + 1))
        for k in range(n_unrisered):
            pool = pools[n_riser_pools + k]
            members = by_pool[starts[n_riser_pools + k]:starts[n_riser_pools + k + 1]]
            if len(members) == 0:
                continue
            piped = members[fraction[members] > 0.5]
            top = float(np.max(heights[members]))
            depth = top - float(np.min(heights[piped])) if len(piped) else 0.0
            self.casting_pipe_report.append(
                {
                    "pool": k,
                    "voxels": int(len(members)),
                    "demand_in3": pool.demand_volume,
                    "pipe_in3": pool.pipe_volume,
                    "pipe_depth_in": depth,
                }
            )
            print(
                f"  Unrisered pool {k} pipe: {len(members):,} voxels, "
                f"demand {pool.demand_volume:.2f} in^3, pipe {pool.pipe_volume:.2f} in^3, "
                f"depth {depth:.2f} in below the top"
            )
        return fraction

    def _compute_riser_pipe_modifier(
        self,
        valid_indices: np.ndarray,
        is_riser: np.ndarray,
        riser_centerlines: Dict[int, Tuple[float, float, float, float, float]],
    ) -> np.ndarray:
        n = len(valid_indices)
        riser_modifier = np.ones(n, dtype=np.float32)

        riser_mask = is_riser
        if not np.any(riser_mask):
            return riser_modifier

        riser_idx_arr = valid_indices[riser_mask]
        centers = self.mesh.voxel_centers[riser_idx_arr]

        comp_ids = np.full(len(self.mesh.voxel_centers), -1, dtype=np.int32)
        for comp_id, riser_ids in enumerate(self._group_risers_by_component()):
            comp_ids[riser_ids] = comp_id

        local_comp_ids = comp_ids[riser_idx_arr]

        unique_comps = np.unique(local_comp_ids)
        comp_center_x = np.zeros(len(unique_comps), dtype=np.float32)
        comp_center_y = np.zeros(len(unique_comps), dtype=np.float32)
        comp_radius = np.zeros(len(unique_comps), dtype=np.float32)
        comp_z_min = np.zeros(len(unique_comps), dtype=np.float32)
        comp_z_max = np.zeros(len(unique_comps), dtype=np.float32)

        for k, cid in enumerate(unique_comps):
            center_x, center_y, radius, z_min, z_max = riser_centerlines[riser_idx_arr[local_comp_ids == cid][0]]
            comp_center_x[k] = center_x
            comp_center_y[k] = center_y
            comp_radius[k] = radius
            comp_z_min[k] = z_min
            comp_z_max[k] = z_max

        cidx = np.searchsorted(unique_comps, local_comp_ids)
        cx = comp_center_x[cidx]
        cy = comp_center_y[cidx]
        cr = comp_radius[cidx]
        cz_min = comp_z_min[cidx]
        cz_max = comp_z_max[cidx]

        radial_dist = np.sqrt(
            (centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2
        )
        normalized_radial = np.minimum(1.0, radial_dist / np.maximum(cr, 0.1))

        z_range = np.maximum(cz_max - cz_min, 0.1)
        riser_heights = np.clip((centers[:, 2] - cz_min) / z_range, 0.0, 1.0)

        pipe_radius_fraction = 0.40
        pipe_start_height = 0.25

        radial_pipe_factor = np.where(
            normalized_radial < pipe_radius_fraction,
            1.0,
            np.where(
                normalized_radial < pipe_radius_fraction * 2,
                1.0 - np.sqrt(
                    np.clip(
                        (normalized_radial - pipe_radius_fraction) / pipe_radius_fraction,
                        0.0,
                        1.0,
                    )
                ),
                0.15,
            ),
        )

        height_pipe_factor = np.where(
            riser_heights > pipe_start_height,
            (riser_heights - pipe_start_height) / (1.0 - pipe_start_height),
            0.15,
        )

        pipe_factor = radial_pipe_factor * height_pipe_factor
        riser_modifier[riser_mask] = (0.18 + pipe_factor * 1.32).astype(np.float32)

        return riser_modifier

    def _calculate_shrinkage_evidence(
        self,
        feeding_effectiveness: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mesh = getattr(self, "mesh", None)
        voxel_centers = getattr(mesh, "voxel_centers", None) if mesh is not None else None
        if voxel_centers is not None:
            n_voxels = len(voxel_centers)
        elif feeding_effectiveness is not None:
            n_voxels = len(feeding_effectiveness)
        elif getattr(self, "shrinkage_risk", None) is not None:
            n_voxels = len(self.shrinkage_risk)
        elif getattr(self, "shrinkage_probability", None) is not None:
            n_voxels = len(self.shrinkage_probability)
        else:
            n_voxels = 0

        directional = np.zeros(n_voxels, dtype=np.float32)
        niyama_severity = getattr(self, "niyama_severity", None)
        if niyama_severity is not None:
            directional = np.clip(np.asarray(niyama_severity, dtype=np.float32), 0.0, 1.0)
        else:
            niyama_values = getattr(self, "niyama_values", None)
            alloy = getattr(self, "alloy", None)
            if niyama_values is not None and alloy is not None:
                alloy_key = alloy.name.split()[0]
                threshold = float(
                    getattr(alloy, "niyama_threshold", NIYAMA_THRESHOLDS.get(alloy_key, 1.0))
                )
                directional = np.clip(
                    1.0 - (np.asarray(niyama_values, dtype=np.float32) / max(threshold, 1e-6)),
                    0.0,
                    1.0,
                )

        feed_loss = np.zeros(n_voxels, dtype=np.float32)
        if feeding_effectiveness is None:
            feeding_effectiveness = getattr(self, "feeding_effectiveness", None)
        if feeding_effectiveness is not None:
            feed_loss = np.maximum(
                feed_loss,
                1.0 - np.clip(np.asarray(feeding_effectiveness, dtype=np.float32), 0.0, 1.0),
            )

        feeding_quality = getattr(self, "feeding_quality", None)
        if feeding_quality is not None:
            feed_loss = np.maximum(
                feed_loss,
                1.0 - np.clip(np.asarray(feeding_quality, dtype=np.float32), 0.0, 1.0),
            )

        open_fraction = getattr(self, "feeding_open_fraction", None)
        if open_fraction is not None:
            feed_loss = np.maximum(
                feed_loss,
                1.0 - np.clip(np.asarray(open_fraction, dtype=np.float32), 0.0, 1.0),
            )

        feeding_blocked = getattr(self, "feeding_blocked", None)
        blocked_mask = np.zeros(n_voxels, dtype=bool)
        if feeding_blocked is not None:
            blocked_mask = np.asarray(feeding_blocked, dtype=bool)
            feed_loss = np.where(blocked_mask, np.maximum(feed_loss, 0.95), feed_loss)

        darcy_margin = getattr(self, "feeding_darcy_margin", None)
        if darcy_margin is not None:
            darcy_loss = np.clip(
                (1.0 - np.asarray(darcy_margin, dtype=np.float32)) * 0.5,
                0.0,
                1.0,
            )
            feed_loss = np.maximum(feed_loss, darcy_loss)

        path_health = np.zeros(n_voxels, dtype=np.float32)
        if feeding_quality is not None:
            path_health = np.maximum(
                path_health,
                np.clip(np.asarray(feeding_quality, dtype=np.float32), 0.0, 1.0),
            )
        if open_fraction is not None:
            path_health = np.maximum(
                path_health,
                np.clip(np.asarray(open_fraction, dtype=np.float32), 0.0, 1.0),
            )
        healthy_path = (path_health >= 0.85) & (~blocked_mask)
        feed_loss = np.where(
            healthy_path,
            np.minimum(feed_loss, 0.35),
            feed_loss,
        )

        try:
            isolation = self._calculate_isolation_score()
        except Exception:
            isolation = np.zeros(n_voxels, dtype=np.float32)
        isolation = np.clip(np.asarray(isolation, dtype=np.float32), 0.0, 1.0)

        solid_times = np.asarray(getattr(self, "solidification_times", np.zeros(n_voxels)), dtype=np.float32)
        finite_times = solid_times[np.isfinite(solid_times) & (solid_times > 0.0)]
        late_factor = np.zeros(n_voxels, dtype=np.float32)
        if len(finite_times) > 1:
            p50 = float(np.percentile(finite_times, 50.0))
            p95 = float(np.percentile(finite_times, 95.0))
            late_factor = np.clip((solid_times - p50) / max(p95 - p50, 1e-6), 0.0, 1.0)

        weighted = (
            (0.40 * directional)
            + (0.35 * feed_loss)
            + (0.25 * isolation)
        ).astype(np.float32)
        if np.any(blocked_mask):
            boosted_weighted = (
                (0.35 * directional)
                + (0.40 * feed_loss)
                + (0.25 * isolation)
            ).astype(np.float32)
            weighted = np.where(blocked_mask, boosted_weighted, weighted)
        trigger = np.maximum.reduce([directional, feed_loss, isolation]).astype(np.float32)
        evidence = np.maximum(weighted, late_factor * trigger)

        connected_feeder_fn = getattr(self, "_get_connected_feeder_mask", None)
        if callable(connected_feeder_fn):
            connected_feeder_mask = np.asarray(connected_feeder_fn(), dtype=bool)
        else:
            connected_feeder_mask = np.zeros(n_voxels, dtype=bool)

        if open_fraction is not None:
            open_ok = np.asarray(open_fraction, dtype=np.float32) >= 0.65
        else:
            open_ok = np.ones(n_voxels, dtype=bool)
        if feeding_quality is not None:
            quality_ok = np.asarray(feeding_quality, dtype=np.float32) >= 0.50
        else:
            quality_ok = np.ones(n_voxels, dtype=bool)

        healthy_connected_feeder = (
            connected_feeder_mask
            & (~blocked_mask)
            & open_ok
            & quality_ok
            & (feed_loss < 0.65)
        )

        return (
            np.clip(evidence, 0.0, 1.0).astype(np.float32),
            np.clip(feed_loss, 0.0, 1.0).astype(np.float32),
            healthy_connected_feeder,
        )

    def _calculate_shrinkage(self):
        n_voxels = len(self.mesh.voxel_centers)
        self.shrinkage_risk = np.zeros(n_voxels, dtype=np.float32)
        self.shrinkage_probability = self.shrinkage_risk
        stage_timings = self.stage_timings

        local_thickness = self._estimate_local_section_size()

        normalized_heights = self._get_normalized_heights()

        riser_indices = self._get_riser_voxel_indices()
        riser_post_indices = self._get_riser_post_voxel_indices()
        chill_indices = self._get_chill_voxel_indices()
        connected_feeder_mask = self._get_connected_feeder_mask()

        stage_start = time.perf_counter()
        chill_reduction = self._calculate_chill_shrinkage_reduction()
        stage_timings["shrinkage_chill_reduction"] = (
            stage_timings.get("shrinkage_chill_reduction", 0.0) + time.perf_counter() - stage_start
        )

        if len(riser_indices) > 0:
            print("  Tracing feeding paths from risers...")
        stage_start = time.perf_counter()
        self._ensure_feeding_path_result(self.feeding_fraction_solid)
        stage_timings["shrinkage_feeding_path_analysis"] = (
            stage_timings.get("shrinkage_feeding_path_analysis", 0.0) + time.perf_counter() - stage_start
        )
        if self.feeding_path_result is not None:
            blocked = int(np.sum(self.feeding_path_result.blocked_voxels))
            print(f"  Feeding path tracing complete: {blocked:,} blocked voxels")
        riser_feeding_reduction = self._calculate_riser_feeding_reduction()

        stage_start = time.perf_counter()
        feeding_effectiveness = self._calculate_feeding_effectiveness()
        stage_timings["shrinkage_feeding_effectiveness"] = (
            stage_timings.get("shrinkage_feeding_effectiveness", 0.0) + time.perf_counter() - stage_start
        )

        if self.feeding_blocked is not None:
            blocked_count = np.sum(self.feeding_blocked)
            if blocked_count > 0:
                print(f"  Using feeding path connectivity: {blocked_count:,} blocked voxels")
                path_lengths = getattr(self, "feeding_path_lengths", None)
                has_valid_path = np.zeros(len(feeding_effectiveness), dtype=bool)
                if path_lengths is not None:
                    has_valid_path = np.asarray(path_lengths, dtype=np.float32) >= 0.0
                blocked_unfed = self.feeding_blocked & ~(
                    connected_feeder_mask & has_valid_path
                )
                feeding_effectiveness = np.where(
                    blocked_unfed,
                    0.1,
                    feeding_effectiveness
                )
            else:
                print(f"  Feeding paths: all voxels have feed path connectivity")

        shrinkage_evidence, feed_loss_evidence, healthy_connected_feeder = (
            self._calculate_shrinkage_evidence(feeding_effectiveness)
        )

        core_indices = self._get_core_voxel_indices()
        mold_indices = np.concatenate([chill_indices, core_indices]).astype(np.int64)
        chill_set = set(chill_indices)
        print(f"  Including {len(riser_indices):,} riser voxels in shrinkage analysis (risers contain shrinkage)")
        print(f"  Excluding {len(chill_set):,} chill voxels (external cooling devices)")
        if len(core_indices) > 0:
            print(f"  Excluding {len(core_indices):,} core voxels (sand)")
        if len(chill_indices) > 0:
            min_reduction = np.min(chill_reduction)
            affected_by_chill = np.sum(chill_reduction < 1.0)
            print(f"  Chill shrinkage reduction: up to {(1.0-min_reduction)*100:.0f}% near chill contact")
            print(f"  Voxels with reduced shrinkage from chills: {affected_by_chill:,}")

        print(f"  Solidification time range: {np.min(self.solidification_times):.1f} - {np.max(self.solidification_times):.1f}s")
        print(f"  Mean solidification time: {np.mean(self.solidification_times):.1f}s")
        print(f"  Voxels with time > 0: {np.sum(self.solidification_times > 0):,}")

        n_workers = hardware.get_optimal_thread_count('compute')
        print(f"  Computing shrinkage probability (vectorized + multiprocessing/JIT)...")

        exclude_mask = np.zeros(n_voxels, dtype=bool)
        exclude_mask[mold_indices] = True
        include_mask = ~exclude_mask

        riser_mask = np.zeros(n_voxels, dtype=bool)
        for idx in riser_indices:
            riser_mask[idx] = True
        riser_post_mask = np.zeros(n_voxels, dtype=bool)
        for idx in riser_post_indices:
            riser_post_mask[idx] = True

        riser_centerlines = self._calculate_riser_centerlines()
        print(f"  Riser centerlines calculated for {len(set(riser_centerlines.values())):,} risers")

        thick_enough_cast = local_thickness >= 0.5
        thick_enough_riser = local_thickness >= 0.5
        risk_moduli = self._get_risk_moduli()
        interior_enough_cast = risk_moduli >= 0.3
        interior_enough_riser = risk_moduli >= 0.2

        valid_cast = include_mask & ~riser_mask & ~riser_post_mask & thick_enough_cast & interior_enough_cast
        valid_riser = include_mask & riser_mask & thick_enough_riser & interior_enough_riser
        valid_post = include_mask & riser_post_mask & thick_enough_riser & interior_enough_riser
        valid_mask = valid_cast | valid_riser | valid_post
        feeding_blocked_arr = getattr(self, "feeding_blocked", None)
        if feeding_blocked_arr is not None:
            valid_mask |= (np.asarray(feeding_blocked_arr, dtype=bool) & include_mask)

        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0:
            (
                base_risk,
                heights,
                moduli,
                thicknesses,
                times,
                is_riser,
                is_riser_post,
                height_factor,
                interior_factor,
                thickness_factor,
                time_factor,
            ) = self._compute_shrinkage_base_risk(
                valid_indices,
                normalized_heights,
                local_thickness,
                riser_mask,
                riser_post_mask,
            )
            riser_modifier = np.where(is_riser, 0.18, 1.0).astype(np.float32)
            base_risk = base_risk * riser_modifier
            connected_valid_feeders = healthy_connected_feeder[valid_indices]
            feeder_neck_modifier = np.where(
                is_riser_post & connected_valid_feeders,
                0.35,
                np.where(is_riser_post, 0.75, 1.0),
            )
            base_risk = base_risk * feeder_neck_modifier

            shrinkage_scale = self.alloy.volumetric_shrinkage / 6.0

            feeding_efficiency = 3.0 / self.alloy.feeding_distance_multiplier

            scaled_risk = base_risk * shrinkage_scale * feeding_efficiency

            chill_reduction_factors = chill_reduction[valid_indices]

            riser_feeding_factors = riser_feeding_reduction[valid_indices]

            feeding_effectiveness_factors = feeding_effectiveness[valid_indices]

            feeding_penalty = 2.0 - feeding_effectiveness_factors

            stage_start = time.perf_counter()
            final_risk = np.clip(
                scaled_risk * chill_reduction_factors * riser_feeding_factors * feeding_penalty,
                0.0,
                1.0,
            ).astype(np.float32)

            evidence_factors = shrinkage_evidence[valid_indices]
            feed_loss_factors = feed_loss_evidence[valid_indices]
            cast_or_post = ~is_riser

            feeding_blocked_full = getattr(self, "feeding_blocked", None)
            feeding_blocked_local = None
            if feeding_blocked_full is not None and len(feeding_blocked_full) == n_voxels:
                feeding_blocked_local = np.asarray(feeding_blocked_full, dtype=bool)[valid_indices]
            blocked_mask = (
                np.asarray(feeding_blocked_local, dtype=bool)
                if feeding_blocked_local is not None
                else np.zeros(len(valid_indices), dtype=bool)
            )
            blocked_floor = np.where(
                blocked_mask,
                np.minimum(0.35 + 0.55 * feed_loss_factors, 0.95).astype(np.float32),
                np.zeros(len(valid_indices), dtype=np.float32),
            )

            low_evidence = cast_or_post & (evidence_factors < 0.28)
            low_evidence_cap = 0.04 + (0.10 * evidence_factors)
            final_risk = np.where(
                low_evidence,
                np.minimum(final_risk * (0.35 + evidence_factors), low_evidence_cap),
                final_risk,
            )
            final_risk = np.where(
                blocked_mask,
                np.maximum(final_risk, blocked_floor),
                final_risk,
            )

            high_evidence = cast_or_post & (evidence_factors >= 0.45)
            evidence_floor = (
                0.08
                + (0.30 * np.clip((evidence_factors - 0.45) / 0.55, 0.0, 1.0))
            )
            feed_loss_floor = 0.10 + (0.25 * feed_loss_factors)
            risk_floor = np.maximum(evidence_floor, feed_loss_floor) * np.sqrt(chill_reduction_factors)
            final_risk = np.where(
                high_evidence,
                np.maximum(final_risk, risk_floor),
                final_risk,
            )
            final_risk = np.where(
                is_riser,
                np.maximum(final_risk, np.clip(base_risk * 1.80, 0.0, 1.0)),
                final_risk,
            )
            final_risk = np.clip(final_risk, 0.0, 1.0).astype(np.float32)
            stage_timings["riser_pipe_modifier"] = (
                stage_timings.get("riser_pipe_modifier", 0.0) + time.perf_counter() - stage_start
            )

            self.shrinkage_risk[valid_indices] = final_risk.astype(np.float32)

        stage_start = time.perf_counter()
        pipe_fraction = self._compute_riser_pipe_fraction()
        self.riser_pipe_fraction = pipe_fraction
        self.shrinkage_risk = np.maximum(self.shrinkage_risk, pipe_fraction).astype(np.float32)
        if len(mold_indices) > 0:
            self.shrinkage_risk[mold_indices] = 0.0
        self.shrinkage_probability = self.shrinkage_risk
        stage_timings["riser_pipe_balance"] = (
            stage_timings.get("riser_pipe_balance", 0.0) + time.perf_counter() - stage_start
        )

        if len(valid_indices) > 0:
            if len(mold_indices) > 0:
                self.shrinkage_risk[mold_indices] = 0.0
            get_orphans = getattr(self, "_get_orphan_metal_indices", None)
            orphan_indices = get_orphans() if callable(get_orphans) else np.array([], dtype=int)
            if len(orphan_indices) > 0:
                self.shrinkage_risk[orphan_indices] = 0.0
            self.shrinkage_probability = self.shrinkage_risk

            debug_count = min(10, len(valid_indices))
            for j in range(debug_count):
                i = valid_indices[j]
                riser_label = " [RISER]" if is_riser[j] else ""
                if is_riser_post[j]:
                    riser_label = " [RISER_POST]"
                chill_label = f", chill_red={chill_reduction_factors[j]:.2f}" if chill_reduction_factors[j] < 1.0 else ""
                riser_feed_label = f", riser_feed={riser_feeding_factors[j]:.2f}" if riser_feeding_factors[j] < 1.0 else ""
                feed_eff_label = f", feed_eff={feeding_effectiveness_factors[j]:.2f}" if feeding_effectiveness_factors[j] < 0.9 else ""
                print(f"    Debug voxel {i}{riser_label}: thick={thicknesses[j]:.2f}\", mod={moduli[j]:.2f}\", "
                      f"height={heights[j]:.2f}, height_f={height_factor[j]:.2f}, "
                      f"interior_f={interior_factor[j]:.2f}, thick_f={thickness_factor[j]:.2f}, "
                      f"time_f={time_factor[j]:.2f}, time={times[j]:.1f}s{chill_label}{riser_feed_label}{feed_eff_label}, "
                      f"base_r={base_risk[j]:.3f}, final_r={final_risk[j]:.3f}")

        cast_shrinkage = self.shrinkage_risk[~riser_mask & ~riser_post_mask & include_mask]
        riser_shrinkage = self.shrinkage_risk[riser_mask]
        riser_post_shrinkage = self.shrinkage_risk[riser_post_mask]

        print(f"  Max shrinkage risk: {np.max(self.shrinkage_risk):.3f}")
        print(f"  Mean shrinkage risk: {np.mean(self.shrinkage_risk):.3f}")
        if len(cast_shrinkage) > 0:
            print(f"  CAST shrinkage: max={np.max(cast_shrinkage):.3f}, mean={np.mean(cast_shrinkage):.3f}")
        else:
            print(f"  CAST shrinkage: (no cast voxels)")
        if len(riser_shrinkage) > 0:
            print(f"  RISER shrinkage: max={np.max(riser_shrinkage):.3f}, mean={np.mean(riser_shrinkage):.3f}")
            print(
                f"  RISER shrinkage voxels above 15%: {np.sum(riser_shrinkage > 0.15):,}, "
                f"above 30%: {np.sum(riser_shrinkage > 0.30):,}"
            )
        else:
            print(f"  RISER shrinkage: (no riser voxels)")
        if len(riser_post_shrinkage) > 0:
            print(
                f"  RISER_POST shrinkage: max={np.max(riser_post_shrinkage):.3f}, "
                f"mean={np.mean(riser_post_shrinkage):.3f}"
            )
        else:
            print(f"  RISER_POST shrinkage: (no riser-post voxels)")
        print(f"  Thickness range: {np.min(local_thickness):.4f} - {np.max(local_thickness):.4f} inches")
        print(f"  Modulus range: {np.min(self.mesh.voxel_moduli):.4f} - {np.max(self.mesh.voxel_moduli):.4f} inches")
        print(
            f"  Surface-distance range: {np.min(self.surface_distances):.4f} - "
            f"{np.max(self.surface_distances):.4f} inches"
        )

        thick_enough_count = np.sum(local_thickness >= 1.5)
        interior_enough_count = np.sum(risk_moduli >= 0.3)
        both = np.sum((local_thickness >= 1.5) & (risk_moduli >= 0.3))

        print(f"  Thick sections (≥1.5\"): {thick_enough_count:,}")
        print(f"  Interior voxels (≥0.3\" from surface): {interior_enough_count:,}")
        print(f"  Both thick AND interior: {both:,}")

        above_threshold = np.sum(self.shrinkage_risk > 0.05)
        print(f"  Voxels above 5% threshold: {above_threshold:,}")

    def _calculate_porosity(self):
        shrinkage_risk = getattr(self, "shrinkage_risk", None)
        shrinkage_probability = getattr(self, "shrinkage_probability", None)
        if shrinkage_risk is None and shrinkage_probability is None:
            return

        print("  Computing porosity risk (Niyama + path blockage + Darcy margin)...")
        if shrinkage_risk is None:
            shrinkage_risk = shrinkage_probability
        shrinkage_risk = np.asarray(shrinkage_risk, dtype=np.float32)
        n_voxels = len(shrinkage_risk)

        mesh = getattr(self, "mesh", None)
        part_types = getattr(mesh, "voxel_part_types", None) if mesh is not None else None
        if part_types is not None:
            part_types = np.asarray(part_types, dtype=np.int32)
            cast_mask = part_types == PART_TYPE_CAST
            riser_mask = part_types == PART_TYPE_RISER
            chill_mask = (part_types == PART_TYPE_CHILL) | (part_types == PART_TYPE_CORE)
            riser_post_mask = part_types == PART_TYPE_RISER_POST
        else:
            cast_mask = np.ones(n_voxels, dtype=bool)
            riser_mask = np.zeros(n_voxels, dtype=bool)
            chill_mask = np.zeros(n_voxels, dtype=bool)
            riser_post_mask = np.zeros(n_voxels, dtype=bool)
        connected_feeder_fn = getattr(self, "_get_connected_feeder_mask", None)
        if callable(connected_feeder_fn):
            connected_feeder_mask = np.asarray(connected_feeder_fn(), dtype=bool)
        else:
            connected_feeder_mask = np.zeros(n_voxels, dtype=bool)
        _, _, healthy_connected_feeder = SolidificationEngine._calculate_shrinkage_evidence(self)

        niyama_severity = getattr(self, "niyama_severity", None)
        if niyama_severity is not None:
            directional_risk = np.clip(np.asarray(niyama_severity, dtype=np.float32), 0.0, 1.0)
        else:
            niyama_values = getattr(self, "niyama_values", None)
            if niyama_values is not None:
                alloy = getattr(self, "alloy", None)
                alloy_key = alloy.name.split()[0] if alloy is not None else ""
                niyama_threshold = float(
                    getattr(
                        alloy,
                        "niyama_threshold",
                        NIYAMA_THRESHOLDS.get(alloy_key, 1.0),
                    )
                )
                directional_risk = np.clip(
                    1.0 - (np.asarray(niyama_values, dtype=np.float32) / max(niyama_threshold, 1e-6)),
                    0.0,
                    1.0,
                )
            else:
                directional_risk = np.zeros(n_voxels, dtype=np.float32)

        blockage_severity = np.zeros(n_voxels, dtype=np.float32)
        feeding_quality = getattr(self, "feeding_quality", None)
        if feeding_quality is not None:
            blockage_severity = np.maximum(
                blockage_severity,
                1.0 - np.clip(np.asarray(feeding_quality, dtype=np.float32), 0.0, 1.0),
            )
        feeding_open_fraction = getattr(self, "feeding_open_fraction", None)
        if feeding_open_fraction is not None:
            blockage_severity = np.maximum(
                blockage_severity,
                1.0 - np.clip(np.asarray(feeding_open_fraction, dtype=np.float32), 0.0, 1.0),
            )
        feeding_blocked = getattr(self, "feeding_blocked", None)
        if feeding_blocked is not None:
            blockage_severity = np.where(
                np.asarray(feeding_blocked, dtype=bool),
                np.maximum(blockage_severity, 0.95),
                blockage_severity,
            )

        darcy_margin = getattr(self, "feeding_darcy_margin", None)
        if darcy_margin is not None:
            darcy_severity = np.clip(
                (1.0 - np.asarray(darcy_margin, dtype=np.float32)) * 0.5,
                0.0,
                1.0,
            )
        else:
            feeding_effectiveness = getattr(self, "feeding_effectiveness", None)
            if feeding_effectiveness is not None:
                darcy_severity = 1.0 - np.clip(np.asarray(feeding_effectiveness, dtype=np.float32), 0.0, 1.0)
            else:
                darcy_severity = blockage_severity.copy()

        criterion = (
            (0.45 * directional_risk)
            + (0.35 * blockage_severity)
            + (0.20 * darcy_severity)
        ).astype(np.float32)
        shrinkage_floor = np.where(cast_mask, 0.18 * shrinkage_risk, 0.35 * shrinkage_risk)
        criterion = np.maximum(criterion, shrinkage_floor)
        susceptibility = (0.35 + (0.65 * shrinkage_risk)).astype(np.float32)
        porosity_risk = criterion * susceptibility
        severe_feed_loss = (blockage_severity > 0.8) & (darcy_severity > 0.65)
        porosity_risk = np.where(
            severe_feed_loss,
            np.maximum(porosity_risk, 0.70 + (0.20 * directional_risk)),
            porosity_risk,
        )
        porosity_risk = np.where(
            riser_post_mask & healthy_connected_feeder,
            porosity_risk * 0.35,
            porosity_risk,
        )
        porosity_risk = np.where(
            cast_mask & healthy_connected_feeder,
            porosity_risk * 0.28,
            porosity_risk,
        )
        porosity_risk = np.where(
            cast_mask,
            porosity_risk * 0.80,
            porosity_risk,
        )
        porosity_risk = np.where(
            riser_mask,
            np.maximum(porosity_risk, 0.60 * shrinkage_risk),
            porosity_risk,
        )

        self.porosity_risk = np.clip(porosity_risk, 0.0, 1.0).astype(np.float32)
        if np.any(chill_mask):
            self.porosity_risk[chill_mask] = 0.0
        get_orphans = getattr(self, "_get_orphan_metal_mask", None)
        if callable(get_orphans):
            orphan_mask = get_orphans()
            if np.any(orphan_mask):
                self.porosity_risk[orphan_mask] = 0.0
        self.porosity_probability = self.porosity_risk
        print(
            "    Porosity criterion components: "
            f"Niyama mean={np.mean(directional_risk):.2f}, "
            f"blockage mean={np.mean(blockage_severity):.2f}, "
            f"Darcy mean={np.mean(darcy_severity):.2f}"
        )

    def _get_shrink_scalar(self) -> float:
        params = getattr(self.analysis_options, "solidification_params", None)
        feeding = getattr(params, "feeding", None)
        raw_value = getattr(feeding, "shrink_scalar", SHRINK_SCALAR_DEFAULT)
        try:
            scalar = float(raw_value)
        except (TypeError, ValueError):
            scalar = SHRINK_SCALAR_DEFAULT
        if scalar != scalar:
            scalar = SHRINK_SCALAR_DEFAULT
        return float(np.clip(scalar, SHRINK_SCALAR_MIN, SHRINK_SCALAR_MAX))

    def _apply_shrink_scalar(self) -> None:
        scalar = self._get_shrink_scalar()
        self.shrink_scalar = scalar
        self.shrink_scalar_formula = SHRINK_SCALAR_FORMULA

        shrinkage_raw = getattr(self, "shrinkage_risk", None)
        if shrinkage_raw is None:
            shrinkage_raw = getattr(self, "shrinkage_probability", None)
        if shrinkage_raw is not None:
            shrinkage_arr = np.asarray(shrinkage_raw, dtype=np.float32)
            self.shrinkage_risk = shrinkage_raw
            self.raw_shrinkage_probability = shrinkage_raw
            if scalar == 1.0:
                self.shrinkage_probability = shrinkage_raw
            else:
                self.shrinkage_probability = np.clip(
                    shrinkage_arr / scalar,
                    0.0,
                    1.0,
                ).astype(np.float32)

        porosity_raw = getattr(self, "porosity_risk", None)
        if porosity_raw is None:
            porosity_raw = getattr(self, "porosity_probability", None)
        if porosity_raw is not None:
            porosity_arr = np.asarray(porosity_raw, dtype=np.float32)
            self.porosity_risk = porosity_raw
            self.raw_porosity_probability = porosity_raw
            if scalar == 1.0:
                self.porosity_probability = porosity_raw
            else:
                self.porosity_probability = np.clip(
                    porosity_arr / scalar,
                    0.0,
                    1.0,
                ).astype(np.float32)

        self.shrink_scalar_applied = True
        print(f"  Shrink scalar: {scalar:.2f} ({SHRINK_SCALAR_FORMULA})")

    def _calculate_hotspots(self):
        n_voxels = len(self.mesh.voxel_centers)
        self.hotspot_intensity = np.zeros(n_voxels, dtype=np.float32)

        times = np.asarray(self.solidification_times, dtype=np.float32)
        max_time = getattr(self, "max_time", None)
        if max_time is None or not np.isfinite(max_time) or max_time <= 0:
            max_time = np.nanmax(times) if np.any(np.isfinite(times)) else 0.0
        times = np.where(np.isnan(times), float(max_time), times)
        max_solid_time = np.max(times)
        if max_solid_time <= 0:
            print("  Warning: No solidification time data for hot spot analysis")
            return

        chill_indices = self._get_chill_voxel_indices()
        chill_set = set(chill_indices)

        normalized_time = times / max_solid_time

        surface_distances = self._get_cast_surface_distances()
        max_feeding_dist = np.max(surface_distances)
        normalized_feeding = surface_distances / max(max_feeding_dist, 0.1)

        risk_moduli = self._get_risk_moduli()
        max_modulus = np.max(risk_moduli)
        normalized_modulus = risk_moduli / max(max_modulus, 0.1)

        isolation_score = self._calculate_isolation_score()

        hotspot_score = (
            0.35 * normalized_time +
            0.20 * normalized_feeding +
            0.15 * normalized_modulus +
            0.30 * isolation_score
        )

        late_threshold = 0.60
        hotspot_mask = normalized_time > late_threshold

        hotspot_mask |= (isolation_score > 0.5)

        self.hotspot_intensity = np.where(
            hotspot_mask,
            np.clip((hotspot_score - 0.3) * 1.5, 0.0, 1.0),
            0.0
        ).astype(np.float32)

        for idx in chill_set:
            self.hotspot_intensity[idx] = 0.0
        self.hotspot_intensity[self._get_core_voxel_indices()] = 0.0

        self.hotspot_count = np.sum(self.hotspot_intensity > 0.2)

        print(f"  Hot spots identified: {self.hotspot_count:,}")
        print(f"  Max hot spot intensity: {np.max(self.hotspot_intensity):.3f}")
        print(f"  Isolated liquid pool factor max: {np.max(isolation_score):.3f}")
        print(f"  Mean hot spot intensity (non-zero): {np.mean(self.hotspot_intensity[self.hotspot_intensity > 0]):.3f}"
              if self.hotspot_count > 0 else "  Mean hot spot intensity: 0")

    def _calculate_isolation_score(self) -> np.ndarray:
        n_voxels = len(self.mesh.voxel_centers)
        neighbors = np.ascontiguousarray(self.mesh.voxel_neighbors, dtype=np.int32)
        solid_times = np.ascontiguousarray(self.solidification_times, dtype=np.float32)
        max_time = getattr(self, "max_time", None)
        if max_time is None or not np.isfinite(max_time) or max_time <= 0:
            max_time = np.nanmax(solid_times) if np.any(np.isfinite(solid_times)) else 0.0
        solid_times = np.ascontiguousarray(
            np.where(np.isnan(solid_times), float(max_time), solid_times)
        )

        if self.mesh.voxel_part_types is not None:
            part_types = np.ascontiguousarray(self.mesh.voxel_part_types, dtype=np.int32)
        else:
            part_types = np.zeros(n_voxels, dtype=np.int32)

        isolation_score = compute_isolation_score_parallel(solid_times, neighbors, part_types)

        surface_distances = self._get_cast_surface_distances()
        surface_factor = surface_distances / np.max(surface_distances + 0.1)
        isolation_score = isolation_score * (0.5 + 0.5 * surface_factor)

        return np.clip(isolation_score, 0.0, 1.0).astype(np.float32)

    def _calculate_niyama(self):
        n_voxels = len(self.mesh.voxel_centers)
        self.niyama_values = np.zeros(n_voxels, dtype=np.float32)
        self.thermal_gradients = np.zeros(n_voxels, dtype=np.float32)
        self.cooling_rates = np.zeros(n_voxels, dtype=np.float32)
        self.niyama_risk = np.zeros(n_voxels, dtype=np.float32)
        self.combined_feed_risk = np.zeros(n_voxels, dtype=np.float32)
        self.niyama_quality = np.zeros(n_voxels, dtype=np.int8)

        alloy_key = self.alloy.name.split()[0]
        niyama_threshold = float(getattr(self.alloy, "niyama_threshold", NIYAMA_THRESHOLDS.get(alloy_key, 1.0)))

        temp_range = self.alloy.liquidus_temp - self.alloy.solidus_temp
        solver_mode = self._get_solver_mode()

        if (
            solver_mode == SOLVER_MODE_THERMAL_COUPLED
            and self._thermal_niyama_values_at_eval is not None
            and self._thermal_niyama_sampled is not None
        ):
            sampled = np.asarray(self._thermal_niyama_sampled, dtype=bool)
            values = np.asarray(self._thermal_niyama_values_at_eval, dtype=np.float32)
            gradients = np.asarray(self._thermal_niyama_gradient_at_eval, dtype=np.float32)
            rates = np.asarray(self._thermal_niyama_cooling_rate_at_eval, dtype=np.float32)
            quality = np.asarray(self._thermal_niyama_quality, dtype=np.int8)

            self.niyama_values = np.where(sampled, values, np.inf).astype(np.float32)
            self.thermal_gradients = np.where(sampled, gradients, 0.0).astype(np.float32)
            self.cooling_rates = np.where(sampled, np.maximum(rates, 1e-6), 0.0).astype(np.float32)
            self.niyama_quality = quality.copy()
            directional_extraction_factor = np.ones(n_voxels, dtype=np.float32)
            niyama_source = "thermal_history"
            threshold_status = "literature_default" if alloy_key == "WCB" else "estimated_default"
        else:
            surface_distances = self._get_cast_surface_distances()
            surface_feeding_dists = surface_distances

            min_distance_to_boundary = 0.1
            effective_distance_in = np.maximum(surface_feeding_dists, min_distance_to_boundary)
            effective_distance_mm = inches_to_mm(effective_distance_in)

            base_gradient = temp_range / effective_distance_mm
            directional_extraction_factor = np.ones(n_voxels, dtype=np.float32)
            if self._chill_boost is not None:
                directional_extraction_factor = np.sqrt(
                    np.maximum(np.asarray(self._chill_boost, dtype=np.float32), 1.0)
                )
            self.thermal_gradients = (base_gradient * directional_extraction_factor).astype(np.float32)

            min_cooling_rate_value = 0.01
            solidification_times = np.asarray(self.solidification_times, dtype=np.float32)
            max_time = getattr(self, "max_time", None)
            if max_time is None or not np.isfinite(max_time) or max_time <= 0:
                max_time = (
                    np.nanmax(solidification_times)
                    if np.any(np.isfinite(solidification_times))
                    else 0.0
                )
            solidification_times = np.where(
                np.isnan(solidification_times), float(max_time), solidification_times
            )
            time_factor = np.where(
                solidification_times > 0,
                1.0 / np.maximum(solidification_times, 0.1),
                1.0
            )
            self.cooling_rates = (temp_range * time_factor + min_cooling_rate_value).astype(np.float32)
            safe_cooling_rates = np.maximum(self.cooling_rates, 1e-10)
            self.niyama_values = (self.thermal_gradients / np.sqrt(safe_cooling_rates)).astype(np.float32)
            self.niyama_quality = np.where(self.niyama_values > 0.0, 1, 0).astype(np.int8)
            niyama_source = "schedule_legacy_approximate"
            threshold_status = "approximate_engine_units"

        finite_niyama = np.isfinite(self.niyama_values)
        self.niyama_risk = np.zeros(n_voxels, dtype=np.float32)
        self.niyama_risk[finite_niyama] = np.clip(
            1.0 - (self.niyama_values[finite_niyama] / max(niyama_threshold, 1e-6)),
            0.0,
            1.0,
        ).astype(np.float32)

        directional_factor = np.ones(n_voxels, dtype=np.float32)
        chill_reduction = np.ones(n_voxels, dtype=np.float32)
        if self._chill_centers is not None:
            chill_reduction = self._calculate_chill_shrinkage_reduction()
            directional_factor = 1.0 + np.clip(1.0 - chill_reduction, 0.0, 0.75)

        isolation_score = self._calculate_isolation_score()
        if self._chill_centers is not None:
            isolation_relief = 1.0 - 0.7 * np.clip(1.0 - chill_reduction, 0.0, 1.0)
            isolation_score = isolation_score * np.clip(isolation_relief, 0.25, 1.0)

        basic_niyama_risk = finite_niyama & (self.niyama_values < niyama_threshold)

        moderate_niyama_risk = finite_niyama & (self.niyama_values < (niyama_threshold * 1.5))
        high_isolation = isolation_score > 0.4
        isolation_enhanced_risk = moderate_niyama_risk & high_isolation

        self.niyama_defect_risk = basic_niyama_risk.copy()
        effective_niyama = self.niyama_values * directional_factor
        directional_niyama_risk = np.zeros(n_voxels, dtype=np.float32)
        directional_niyama_risk[finite_niyama] = np.clip(
            1.0 - (effective_niyama[finite_niyama] / max(niyama_threshold, 1e-6)),
            0.0,
            1.0,
        ).astype(np.float32)
        self.combined_feed_risk = np.maximum(
            directional_niyama_risk,
            isolation_score * 0.7,
        ).astype(np.float32)

        surface_indices = self._get_material_surface_indices(include_cores=False)
        self.niyama_defect_risk[surface_indices] = False
        self.niyama_risk[surface_indices] = 0.0
        self.combined_feed_risk[surface_indices] = 0.0

        chill_indices = self._get_chill_voxel_indices()
        self.niyama_defect_risk[chill_indices] = False
        self.niyama_risk[chill_indices] = 0.0
        self.combined_feed_risk[chill_indices] = 0.0

        get_orphans = getattr(self, "_get_orphan_metal_indices", None)
        orphan_indices = get_orphans() if callable(get_orphans) else np.array([], dtype=int)
        if len(orphan_indices) > 0:
            self.niyama_defect_risk[orphan_indices] = False
            self.niyama_risk[orphan_indices] = 0.0
            self.combined_feed_risk[orphan_indices] = 0.0

        self.niyama_severity = self.combined_feed_risk.astype(np.float32)

        self.niyama_defect_count = np.sum(self.niyama_defect_risk)
        isolation_enhanced_count = np.sum(isolation_enhanced_risk & ~basic_niyama_risk)
        self.niyama_metadata = {
            "source": niyama_source,
            "units": "(C*s)^0.5/mm",
            "gradient_units": "C/mm",
            "cooling_rate_units": "C/s",
            "eval_fraction_solid": float(self.niyama_eval_fraction_solid),
            "eval_temperature_c": float(
                self.alloy.solidus_temp
                + (1.0 - self.niyama_eval_fraction_solid) * temp_range
            ),
            "threshold": niyama_threshold,
            "threshold_status": threshold_status,
        }

        print(f"  Niyama threshold for {alloy_key}: {niyama_threshold:.2f}")
        valid_niyama = self.niyama_values[finite_niyama]
        if len(valid_niyama) > 0:
            print(f"  Niyama value range: {np.min(valid_niyama):.3f} - {np.max(valid_niyama):.3f}")
            print(f"  Mean Niyama: {np.mean(valid_niyama):.3f}")
        else:
            print("  Niyama value range: no sampled cast voxels")
            print("  Mean Niyama: no sampled cast voxels")
        print(
            f"  Directional-solidification factor range: "
            f"{np.min(directional_factor):.2f} - {np.max(directional_factor):.2f}"
        )
        print(f"  Basic defect risk (Ny < {niyama_threshold}): {np.sum(basic_niyama_risk):,}")
        print(f"  Combined feed-risk voxels added by isolation: {isolation_enhanced_count:,}")
        print(f"  Raw Niyama defect risk voxels: {self.niyama_defect_count:,}")
        print(f"  Thermal gradient range: {np.min(self.thermal_gradients):.2f} - {np.max(self.thermal_gradients):.2f} C/mm")
        print(f"  Cooling rate range: {np.min(self.cooling_rates):.2f} - {np.max(self.cooling_rates):.2f} C/s")
        print(f"  Niyama source: {niyama_source}")
        print("  Material scaling: alloy threshold only (no extra k/cp multiplier)")

    def _analyze_btr_history(
        self,
        btr_start: float,
        btr_end: float,
        critical_fraction_solid: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_voxels = len(self.mesh.voxel_centers)
        btr_dwell_time = np.zeros(n_voxels, dtype=np.float32)
        supporting_fraction = np.zeros(n_voxels, dtype=np.float32)
        in_btr = np.zeros(n_voxels, dtype=bool)

        if not self.fraction_liquid_history:
            return btr_dwell_time, supporting_fraction, in_btr

        snapshots = self.fraction_liquid_history
        times = self.fraction_liquid_times or []
        neighbors = np.asarray(self.mesh.voxel_neighbors, dtype=np.int32)
        has_matching_times = len(times) == len(snapshots)

        for k, snapshot in enumerate(snapshots):
            fraction_solid = np.asarray(snapshot, dtype=np.float32)
            snapshot_btr = (fraction_solid >= btr_start) & (fraction_solid <= btr_end)
            if not np.any(snapshot_btr):
                continue
            in_btr |= snapshot_btr

            btr_voxels = np.flatnonzero(snapshot_btr)
            for voxel_index in btr_voxels:
                neighbor_indices = neighbors[voxel_index]
                valid_neighbors = neighbor_indices[neighbor_indices >= 0]
                if len(valid_neighbors) == 0:
                    continue
                supp = float(
                    np.mean(fraction_solid[valid_neighbors] >= critical_fraction_solid)
                )
                if supp > supporting_fraction[voxel_index]:
                    supporting_fraction[voxel_index] = supp

            if has_matching_times and k + 1 < len(snapshots):
                dt = float(times[k + 1] - times[k])
                if dt > 0:
                    next_snapshot = np.asarray(snapshots[k + 1], dtype=np.float32)
                    next_btr = (next_snapshot >= btr_start) & (next_snapshot <= btr_end)
                    frac_in_btr = (snapshot_btr.astype(np.float32) + next_btr.astype(np.float32)) * 0.5
                    btr_dwell_time += frac_in_btr * dt
            elif not has_matching_times:
                pass

        return btr_dwell_time, supporting_fraction, in_btr

    def _get_sand_restraint_index(self) -> float:
        ht = self.analysis_options.solidification_params.hot_tear
        mold_restraint = getattr(ht, 'mold_restraint_index', 0.5)
        default_mold_restraint = 0.5

        if mold_restraint != default_mold_restraint:
            try:
                return float(np.clip(float(mold_restraint), 0.0, 1.0))
            except (TypeError, ValueError):
                pass

        raw_value = getattr(self.analysis_options, "sand_restraint_index", 0.5)
        try:
            return float(np.clip(float(raw_value), 0.0, 1.0))
        except (TypeError, ValueError):
            return 0.5

    def _compute_surface_restraint_weight(self) -> np.ndarray:
        n_voxels = len(self.mesh.voxel_centers)
        surface_distances = self.surface_distances
        if surface_distances is None:
            return np.zeros(n_voxels, dtype=np.float32)

        influence_distance = max(0.5, float(self.mesh.base_voxel_size) * 5.0)
        distances = np.asarray(surface_distances, dtype=np.float32)
        weights = np.clip(1.0 - distances / influence_distance, 0.0, 1.0)
        return weights.astype(np.float32)

    def _calculate_hot_tear_risk(self):
        n_voxels = len(self.mesh.voxel_centers)
        hot_tear_params = self.analysis_options.solidification_params.hot_tear

        self.hot_tear_risk = np.zeros(n_voxels, dtype=np.float32)
        self.strain_rate = np.zeros(n_voxels, dtype=np.float32)
        self.btr_mask = np.zeros(n_voxels, dtype=bool)

        hts = self.alloy.hot_tear_susceptibility
        solid_expansion = getattr(self.alloy, 'solid_thermal_expansion', 1.2e-5)
        btr_critical_strain = getattr(self.alloy, 'btr_critical_strain', 0.008)
        btr_critical_strain_rate_sensitivity = getattr(
            self.alloy, 'btr_critical_strain_rate_sensitivity', 0.0
        )
        btr_critical_strain_ref_rate = getattr(
            self.alloy, 'btr_critical_strain_ref_rate', 1.0e-3
        )
        healing_factor = getattr(self.alloy, 'hot_tear_healing_factor', 0.55)
        btr_start = self.alloy.btr_fraction_solid_start
        btr_end = self.alloy.btr_fraction_solid_end
        btr_cooling_rate_exponent = getattr(
            self.alloy, 'btr_cooling_rate_exponent', 0.0
        )
        btr_cooling_rate_ref = getattr(
            self.alloy, 'btr_btr_cooling_rate_ref', 1.0
        )
        full_temp_range = float(self.alloy.liquidus_temp - self.alloy.solidus_temp)

        btr_width = btr_end - btr_start
        btr_temp_range = btr_width * full_temp_range

        print(f"  Hot tear susceptibility: {hts:.2f}")
        print(f"  BTR: {btr_start:.0%} to {btr_end:.0%} solid")
        print(f"  Solid thermal expansion: {solid_expansion:.2e} 1/K")
        print(f"  BTR critical strain: {btr_critical_strain:.4f}")
        if btr_critical_strain_rate_sensitivity > 0:
            print(f"  BTR critical strain rate-sensitivity: "
                  f"{btr_critical_strain_rate_sensitivity:.2f} "
                  f"(ref rate {btr_critical_strain_ref_rate:.1e} 1/s)")
        print(f"  Healing factor: {healing_factor:.2f}")
        print(f"  Freezing range: {full_temp_range:.1f} °C (full), {btr_temp_range:.2f} °C (BTR)")
        if btr_cooling_rate_exponent != 0:
            print(f"  BTR cooling-rate exponent: {btr_cooling_rate_exponent:.3f} "
                  f"(ref rate {btr_cooling_rate_ref:.1f} °C/s)")

        part_types = np.asarray(self.mesh.voxel_part_types, dtype=np.int32) if self.mesh.voxel_part_types is not None else np.zeros(n_voxels, dtype=np.int32)

        cast_mask = (part_types == PART_TYPE_CAST)
        if not np.any(cast_mask):
            cast_mask = np.ones(n_voxels, dtype=bool)

        surface_indices = self.mesh.get_surface_voxel_indices()
        surface_mask = np.zeros(n_voxels, dtype=bool)
        surface_mask[surface_indices] = True

        exclude_mask = np.zeros(n_voxels, dtype=bool)
        exclude_mask |= is_feeder(part_types)
        exclude_mask |= is_mold(part_types)
        exclude_mask |= is_gating_metal(part_types)
        exclude_mask |= surface_mask

        analysis_mask = cast_mask & ~exclude_mask

        neighbors = np.ascontiguousarray(self.mesh.voxel_neighbors, dtype=np.int64)
        voxel_moduli = np.ascontiguousarray(self.mesh.voxel_moduli, dtype=np.float32)
        solidification_times = np.ascontiguousarray(self.solidification_times, dtype=np.float32)
        voxel_centers = np.ascontiguousarray(self.mesh.voxel_centers, dtype=np.float32)

        print(f"  Analyzing {np.sum(analysis_mask):,} voxels for hot tear risk")

        btr_dwell_time, supporting_fraction, history_btr_mask = self._analyze_btr_history(
            btr_start, btr_end, self.alloy.critical_fraction_solid
        )
        self.btr_mask = history_btr_mask.copy()
        self.btr_mask[~analysis_mask] = False

        times = self.fraction_liquid_times or []
        if len(times) >= 2:
            total_solidification_time = max(float(times[-1] - times[0]), 1e-6)
        else:
            total_solidification_time = max(float(np.max(solidification_times)), 1e-6)
        btr_exposure = np.clip(btr_dwell_time / total_solidification_time, 0.0, 1.0).astype(np.float32)

        if btr_cooling_rate_exponent != 0:
            cooling_rates = getattr(self, "cooling_rates", None)
            if cooling_rates is not None and len(cooling_rates) == n_voxels:
                cr = np.maximum(np.asarray(cooling_rates, dtype=np.float32), 1e-4)
                ref_cr = max(float(btr_cooling_rate_ref), 1e-4)
                btr_width_ratio = np.power(cr / ref_cr, btr_cooling_rate_exponent)
                btr_width_ratio = np.clip(btr_width_ratio, 0.5, 2.0)
                btr_dwell_time = btr_dwell_time * btr_width_ratio
                btr_exposure = np.clip(
                    btr_dwell_time / total_solidification_time, 0.0, 1.0
                ).astype(np.float32)

        if np.any(supporting_fraction > 0.0):
            coherency_weight = np.clip(supporting_fraction * 2.0, 0.0, 1.0)
            btr_exposure = (btr_exposure * coherency_weight).astype(np.float32)

        max_solid_time = np.max(solidification_times) if np.max(solidification_times) > 0 else 1.0
        normalized_time = solidification_times / max_solid_time

        has_history = np.any(history_btr_mask)
        if not has_history:
            print(
                "  WARNING: hot_tear analysis running without fraction_liquid history; "
                "using coarse time-based BTR window [0.90, 0.99]. "
                "Set AnalysisOptions.hot_tear=True before run_simulation() for accurate "
                "BTR exposure."
            )
            btr_time_start = 0.90
            btr_time_end = 0.99
            time_in_btr = (normalized_time >= btr_time_start) & (normalized_time <= btr_time_end)
            btr_exposure[analysis_mask & time_in_btr] = 1.0

        mechanical_strain = compute_constraint_strain_parallel(
            solidification_times,
            voxel_moduli,
            neighbors,
            np.float32(solid_expansion),
            np.float32(btr_temp_range),
        )

        modulus_variance = compute_modulus_variance_parallel(voxel_moduli, neighbors)

        voxel_size = float(getattr(self.mesh, "base_voxel_size", 0.5))
        search_depth = int(getattr(hot_tear_params, "solid_path_search_depth", 8))
        max_path_length = max(voxel_size * search_depth, voxel_size)
        path_strain = compute_solid_path_strain_parallel(
            solidification_times,
            voxel_centers,
            neighbors,
            np.float32(solid_expansion),
            np.float32(btr_temp_range),
            np.float32(max_path_length),
        )

        section_change = np.abs(modulus_variance) / (np.max(modulus_variance) + 1e-6)

        stress_concentration = np.ones(n_voxels, dtype=np.float32)
        max_var = np.max(modulus_variance) if np.max(modulus_variance) > 0 else 1.0
        norm_variance = modulus_variance / max_var
        max_path = np.max(path_strain) if np.max(path_strain) > 0 else 1.0
        stress_concentration = np.clip(
            1.0 + 0.3 * norm_variance + 0.15 * section_change + 0.10 * (path_strain / max_path),
            0.75, 1.75
        )

        safe_dwell = np.maximum(btr_dwell_time, 1e-3)
        strain_rate = np.where(
            btr_dwell_time > 0.0,
            mechanical_strain / safe_dwell,
            0.0,
        ).astype(np.float32)
        self.strain_rate = strain_rate

        if btr_critical_strain_rate_sensitivity > 0:
            rate_log = np.log10(
                (btr_critical_strain_ref_rate + strain_rate) / btr_critical_strain_ref_rate
            )
            rate_factor = 1.0 - btr_critical_strain_rate_sensitivity * rate_log
            rate_factor = np.clip(rate_factor, 0.5, 1.0)
            btr_critical_strain_effective = (
                np.full(n_voxels, float(btr_critical_strain), dtype=np.float32) * rate_factor
            )
        else:
            btr_critical_strain_effective = np.full(
                n_voxels, float(btr_critical_strain), dtype=np.float32
            )

        strain_index = np.clip(
            mechanical_strain / np.maximum(btr_critical_strain_effective, 1e-6),
            0.0,
            1.0,
        )

        surface_distances = np.ascontiguousarray(
            self.surface_distances,
            dtype=np.float32,
        )

        restraint_index = np.zeros(n_voxels, dtype=np.float32)

        mold_restraint_value = getattr(hot_tear_params, 'mold_restraint_index', 0.5)
        mold_is_default = (mold_restraint_value == 0.5)
        fallback_value = getattr(self.analysis_options, 'sand_restraint_index', 0.5)
        mold_restraint = float(mold_restraint_value) if not mold_is_default else float(fallback_value)
        friction_value = getattr(hot_tear_params, 'metal_mold_friction_index', 0.5)
        friction_is_default = (friction_value == 0.5)
        friction = float(friction_value) if not friction_is_default else float(fallback_value)
        core_restraint = float(getattr(hot_tear_params, 'core_restraint_index', 0.0))
        restraint_depth = max(0.1, float(getattr(hot_tear_params, 'surface_restraint_depth_inches', 0.5)))

        if surface_distances is not None:
            surface_weight = np.clip(1.0 - surface_distances / restraint_depth, 0.0, 1.0)
        else:
            surface_weight = np.zeros(n_voxels, dtype=np.float32)

        restraint_index = surface_weight * (0.6 * mold_restraint + 0.4 * friction)

        global_restraint = 0.25 * (0.6 * mold_restraint + 0.4 * friction)
        restraint_index = np.maximum(restraint_index, global_restraint)

        core_voxels = (part_types == PART_TYPE_CORE)
        if np.any(core_voxels):
            core_centers = voxel_centers[core_voxels]
            if len(core_centers) > 0:
                kdtree = _build_kdtree(core_centers)
                distances, _ = kdtree.query(voxel_centers)
                core_weight = np.clip(1.0 - distances / restraint_depth, 0.0, 1.0)
                restraint_index = np.maximum(restraint_index, core_restraint * core_weight)
        elif core_restraint > 0:
            restraint_index = np.maximum(restraint_index, core_restraint * 0.2)

        restraint_multiplier = np.clip(0.6 + 0.8 * restraint_index, 0.4, 1.4)

        feeding_result = self._ensure_feeding_path_result(self.feeding_fraction_solid)

        healing_capacity = np.zeros(n_voxels, dtype=np.float32)
        if feeding_result is not None:
            healing_capacity = np.maximum(
                healing_capacity,
                np.asarray(feeding_result.feeding_quality, dtype=np.float32),
            )
        history_open_fraction = getattr(self, 'feeding_open_fraction', None)
        if history_open_fraction is not None:
            healing_capacity = np.maximum(
                healing_capacity,
                np.asarray(history_open_fraction, dtype=np.float32),
            )

        latest_open_time = None
        if feeding_result is not None:
            latest_open_time = getattr(feeding_result, "latest_open_time", None)
        if latest_open_time is None:
            latest_open_time = getattr(self, "feeding_latest_open_time", None)
        if latest_open_time is not None and total_solidification_time > 0:
            latest_open_arr = np.asarray(latest_open_time, dtype=np.float32)
            latest_open_norm = np.clip(
                latest_open_arr / total_solidification_time, 0.0, 1.0
            )
            btr_window_open = np.clip(
                (latest_open_norm - btr_start) / max(btr_width, 1e-6), 0.0, 1.0
            )
            healing_capacity = np.maximum(healing_capacity, btr_window_open)

        healing_capacity *= healing_factor

        healing_relief = float(getattr(hot_tear_params, 'healing_relief_strength', 0.65))
        healing_penalty = 1.0 - healing_relief * np.clip(healing_capacity, 0.0, 1.0)

        material_factor = np.full(n_voxels, float(hts), dtype=np.float32)
        material_factor *= (1.0 + 0.3 * (btr_width / 0.15))

        impurity_index = float(getattr(hot_tear_params, 'low_melting_impurity_index', 0.0))
        material_factor *= (1.0 + 0.4 * impurity_index)

        superheat = float(self.alloy.pouring_temp - self.alloy.liquidus_temp)
        material_factor *= (1.0 + 0.1 * (superheat / 200.0))

        cooling_rates = getattr(self, "cooling_rates", None)
        if cooling_rates is not None and len(cooling_rates) == n_voxels:
            cr = np.maximum(np.asarray(cooling_rates, dtype=np.float32), 1e-4)
            sdas_ref = max(float(self.alloy.dendrite_arm_spacing), 1.0)
            sdas_exp = float(self.alloy.sdas_cooling_rate_exponent)
            ref_cr = max(float(self.alloy.sdas_reference_cooling_rate), 1e-4)
            local_sdas = sdas_ref * np.power(cr / ref_cr, sdas_exp)
            local_sdas = np.clip(local_sdas, 10.0, 500.0)
            sdas_ratio = local_sdas / sdas_ref
            sdas_ratio = np.clip(sdas_ratio, 0.5, 2.0)
            grain_refinement_factor = np.sqrt(sdas_ratio)
            material_factor = material_factor * grain_refinement_factor

        material_factor = np.clip(material_factor, 0.1, 1.5)

        risk = (
            material_factor
            * btr_exposure
            * strain_index
            * restraint_multiplier
            * stress_concentration
            * healing_penalty
        )

        self.hot_tear_risk = np.clip(risk, 0.0, 1.0).astype(np.float32)
        self.hot_tear_risk[~analysis_mask] = 0.0

        configured_threshold = float(getattr(hot_tear_params, 'hot_tear_zone_threshold', 0.0))
        threshold = configured_threshold if configured_threshold > 0 else 0.06 * (1.0 - hts * 0.5)
        self.hot_tear_zones = self.hot_tear_risk >= threshold
        self.hot_tear_zones[exclude_mask] = False
        self.hot_tear_zones[~analysis_mask] = False

        self.hot_tear_components = {
            "btr_exposure": btr_exposure,
            "btr_dwell_time": btr_dwell_time,
            "btr_temp_range_used": np.full(n_voxels, btr_temp_range, dtype=np.float32),
            "coherency_support_fraction": supporting_fraction,
            "btr_critical_strain_effective": btr_critical_strain_effective,
            "mechanical_strain": mechanical_strain,
            "strain_index": strain_index,
            "strain_rate": strain_rate,
            "restraint_index": restraint_index,
            "stress_concentration": stress_concentration,
            "healing_capacity": healing_capacity,
            "material_factor": material_factor,
            "healing_penalty": healing_penalty,
        }

        self.hot_tear_count = np.sum(self.hot_tear_zones)

        print(f"  === Hot Tear Analysis Summary ===")
        print(f"  BTR dwell time: total {float(np.sum(btr_dwell_time)):.1f}s, "
              f"max per voxel {float(np.max(btr_dwell_time)):.1f}s")
        print(f"  BTR exposure range: {float(np.min(btr_exposure)):.4f} - "
              f"{float(np.max(btr_exposure)):.4f}")
        print(f"  Strain index range: {float(np.min(strain_index)):.4f} - "
              f"{float(np.max(strain_index)):.4f}")
        print(f"  Strain rate (BTR) range: {float(np.min(strain_rate)):.2e} - "
              f"{float(np.max(strain_rate)):.2e} 1/s")
        print(f"  Restraint multiplier: {float(np.min(restraint_multiplier)):.3f} - "
              f"{float(np.max(restraint_multiplier)):.3f}")
        print(f"  Stress concentration: {float(np.min(stress_concentration)):.3f} - "
              f"{float(np.max(stress_concentration)):.3f}")
        print(f"  Healing capacity: {float(np.mean(healing_capacity)):.3f}, "
              f"penalty: {float(np.mean(healing_penalty)):.3f}")
        print(f"  Material factor: {float(material_factor[0]):.3f}")
        print(f"  Hot tear threshold: {threshold:.3f}")
        print(f"  Risk range: {float(np.min(self.hot_tear_risk)):.4f} - "
              f"{float(np.max(self.hot_tear_risk)):.4f}")
        print(f"  Hot tear zone voxels: {self.hot_tear_count:,}")

    def _calculate_sand_burn_in(self):
        n_voxels = len(self.mesh.voxel_centers)

        if n_voxels == 0:
            raise ValueError("Mesh has no voxels for sand burn-in analysis")

        if not hasattr(self.alloy, 'sand_burn_in_susceptibility'):
            logger.warning(
                f"Alloy '{self.alloy.name}' missing sand_burn_in_susceptibility, using default 0.5",
                context={'alloy': self.alloy.name}
            )
            material_susceptibility = 0.5
        else:
            material_susceptibility = self.alloy.sand_burn_in_susceptibility

        self.sand_burn_in_risk = np.zeros(n_voxels, dtype=np.float32)
        self.sand_burn_in_zones = np.zeros(n_voxels, dtype=np.bool_)

        try:
            surface_indices = self.mesh.get_surface_voxel_indices()
        except Exception as e:
            logger.error(
                f"Failed to get surface voxel indices for sand burn-in analysis: {e}",
                exc_info=True,
                context={'n_voxels': n_voxels}
            )
            raise RuntimeError(f"Surface voxel detection failed: {e}") from e

        n_surface = len(surface_indices)

        if n_surface == 0:
            logger.warning(
                "No surface voxels found for sand burn-in analysis",
                context={'n_voxels': n_voxels}
            )
            print("  Warning: No surface voxels found for sand burn-in analysis")
            self.sand_burn_in_count = 0
            return

        print(f"  Analyzing {n_surface:,} surface voxels for sand burn-in risk...")

        print(f"    Material susceptibility: {material_susceptibility:.2f} ({self.alloy.name})")

        superheat = self.alloy.pouring_temp - self.alloy.liquidus_temp
        temp_factor = np.clip(superheat / 200.0, 0.0, 1.0)
        print(f"    Superheat: {superheat:.0f}°C (factor: {temp_factor:.2f})")

        voxel_heights = self.mesh.voxel_centers[:, 2]
        max_height = np.max(voxel_heights)
        min_height = np.min(voxel_heights)
        height_range = max_height - min_height

        if height_range > 0:
            surface_heights = voxel_heights[surface_indices]
            pressure_factor = 1.0 - (surface_heights - min_height) / height_range
        else:
            pressure_factor = np.ones(n_surface) * 0.5

        metal_head_factor = np.clip(height_range / 12.0, 0.1, 1.0)
        pressure_factor = pressure_factor * metal_head_factor

        print(f"    Metal head height: {height_range:.2f}\" (factor: {metal_head_factor:.2f})")

        surface_solid_times = self.solidification_times[surface_indices]
        max_surface_time = np.max(surface_solid_times) if np.max(surface_solid_times) > 0 else 1.0

        time_factor = np.clip(surface_solid_times / 60.0, 0.0, 1.0)

        print(f"    Surface solidification time range: {np.min(surface_solid_times):.1f} - {np.max(surface_solid_times):.1f}s")

        default_permeability = 80.0
        default_grain_fineness = 55.0

        perm_factor = np.clip((default_permeability - 30) / 120.0, 0.0, 1.0)

        gfn_factor = np.clip((100 - default_grain_fineness) / 60.0, 0.0, 1.0)

        sand_factor = (perm_factor + gfn_factor) / 2.0

        print(f"    Sand permeability: {default_permeability:.0f} AFS (factor: {perm_factor:.2f})")
        print(f"    Sand grain fineness: {default_grain_fineness:.0f} AFS (factor: {gfn_factor:.2f})")
        print(f"    Combined sand factor: {sand_factor:.2f}")

        chill_reduction = np.ones(n_surface, dtype=np.float32)

        try:
            chill_indices = self._get_chill_voxel_indices()

            if len(chill_indices) > 0:
                try:
                    chill_centers = self.mesh.voxel_centers[chill_indices]
                    surface_centers = self.mesh.voxel_centers[surface_indices]

                    if self._chill_thicknesses is not None and len(self._chill_thicknesses) > 0:
                        chill_thickness = np.mean(self._chill_thicknesses)
                    else:
                        chill_thickness = self.mesh.base_voxel_size * 5.0
                    chill_burn_in_radius = chill_thickness * 1.5
                    max_reduction = 0.3

                    chill_tree = _build_kdtree(chill_centers)
                    distances, _ = chill_tree.query(
                        surface_centers,
                        k=1,
                        distance_upper_bound=chill_burn_in_radius,
                    )

                    within_radius = np.isfinite(distances) & (distances < chill_burn_in_radius)
                    if np.any(within_radius):
                        normalized_dist = distances[within_radius] / chill_burn_in_radius
                        chill_reduction[within_radius] = (
                            max_reduction + (1.0 - max_reduction) * normalized_dist
                        )

                    chill_affected = np.sum(chill_reduction < 1.0)
                    if chill_affected > 0:
                        print(f"    Chill reduces burn-in risk for {chill_affected:,} surface voxels (radius: {chill_burn_in_radius:.2f}\")")
                except ImportError:
                    logger.warning(
                        "scipy not available for chill proximity calculation in sand burn-in analysis",
                        context={'n_chill_voxels': len(chill_indices)}
                    )
                    print("    Warning: scipy not available, skipping chill proximity reduction")
                except NameError:
                    raise
                except Exception as e:
                    logger.warning(
                        f"Chill proximity calculation failed in sand burn-in analysis: {e}",
                        context={'n_chill_voxels': len(chill_indices), 'error': str(e)}
                    )
                    print(f"    Warning: Chill proximity calculation failed: {e}")
        except Exception as e:
            logger.warning(
                f"Failed to get chill voxel indices for sand burn-in analysis: {e}",
                context={'error': str(e)}
            )
            print(f"    Warning: Could not detect chill voxels: {e}")

        w_temp = 0.35
        w_pressure = 0.30
        w_time = 0.20
        w_sand = 0.15

        base_risk = (
            w_temp * temp_factor +
            w_pressure * pressure_factor +
            w_time * time_factor +
            w_sand * sand_factor
        )

        surface_risk = base_risk * material_susceptibility * chill_reduction

        surface_risk = np.clip(surface_risk, 0.0, 1.0).astype(np.float32)

        self.sand_burn_in_risk[surface_indices] = surface_risk

        risk_threshold = 0.20

        zone_mask = np.zeros(n_voxels, dtype=bool)
        zone_mask[surface_indices] = surface_risk > risk_threshold
        self.sand_burn_in_zones = zone_mask

        try:
            chill_indices = self._get_chill_voxel_indices()
            if len(chill_indices) > 0:
                self.sand_burn_in_zones[chill_indices] = False
        except Exception:
            pass
        core_indices = self._get_core_voxel_indices()
        if len(core_indices) > 0:
            self.sand_burn_in_zones[core_indices] = False
            self.sand_burn_in_risk[core_indices] = 0.0

        self.sand_burn_in_count = np.sum(self.sand_burn_in_zones)

        print(f"  Sand burn-in risk range: {np.min(surface_risk):.3f} - {np.max(surface_risk):.3f}")
        print(f"  Mean surface burn-in risk: {np.mean(surface_risk):.3f}")
        print(f"  Sand burn-in threshold: {risk_threshold:.2f}")
        print(f"  Surface voxels at risk: {self.sand_burn_in_count:,} ({self.sand_burn_in_count/n_surface*100:.1f}% of surface)")

    def _calculate_gas_porosity(self):
        from physics.gas_porosity import (
            compute_surface_gas_generation,
            propagate_gas_to_interior,
            compute_combined_risk,
            compute_hotspots,
        )

        n_voxels = len(self.mesh.voxel_centers)

        if n_voxels == 0:
            raise ValueError("Mesh has no voxels for gas porosity analysis")

        gas_sensitivity = getattr(self.alloy, "gas_sensitivity", 0.5)

        surface_indices = np.asarray(
            self.mesh.get_surface_voxel_indices(), dtype=np.int32
        )
        n_surface = len(surface_indices)

        if n_surface == 0:
            print("  Warning: No surface voxels found for gas porosity analysis")
            self.gas_generation = np.zeros(n_voxels, dtype=np.float32)
            self.gas_entrapment = np.zeros(n_voxels, dtype=np.float32)
            self.gas_porosity_risk = np.zeros(n_voxels, dtype=np.float32)
            self.gas_porosity_hotspots = np.zeros(n_voxels, dtype=bool)
            self.gas_porosity_count = 0
            return

        print(f"  Analyzing gas porosity for {n_surface:,} surface voxels...")

        humidity = 0.5
        binder_type = "phenolic_urethane"
        binder_content = 1.5
        core_venting = 0.5
        penetration_depth = 3.0
        entrapment_exponent = 2.0
        niyama_coupling_alpha = 0.5

        gas_part_types = self.mesh.voxel_part_types
        core_contact = self._get_core_contact_cast_indices()
        if len(core_contact) > 0:
            gas_part_types = np.asarray(gas_part_types, dtype=np.int32).copy()
            gas_part_types[core_contact] = PART_TYPE_CORE
            surface_indices = np.union1d(surface_indices, core_contact).astype(np.int32)
        sgi0 = compute_surface_gas_generation(
            surface_indices=surface_indices,
            voxel_part_types=gas_part_types,
            local_solidification_times=self.local_solidification_time,
            humidity=humidity,
            binder_type=binder_type,
            binder_content=binder_content,
            core_venting_effectiveness=core_venting,
        )

        gas_gen, _nearest_map = propagate_gas_to_interior(
            voxel_centers=self.mesh.voxel_centers,
            surface_indices=surface_indices,
            surface_sgi0=sgi0,
            gas_penetration_depth=penetration_depth,
        )

        self.gas_generation = gas_gen.astype(np.float32)

        niyama_risk = getattr(self, "niyama_risk", None)

        entrapment, _niyama_coupling, risk = compute_combined_risk(
            self.gas_generation,
            self.local_solidification_time,
            niyama_risk,
            gas_sensitivity,
            entrapment_exponent,
            niyama_coupling_alpha,
        )

        self.gas_entrapment = entrapment.astype(np.float32)
        self.gas_porosity_risk = risk.astype(np.float32)

        self.gas_porosity_hotspots = compute_hotspots(
            self.gas_porosity_risk, threshold=0.65
        )

        self.gas_porosity_count = int(np.sum(self.gas_porosity_hotspots))

        print(f"    Alloy gas sensitivity (χ): {gas_sensitivity}")
        print(f"    Gas penetration depth (λ): {penetration_depth:.1f} in")
        print(f"    Entrapment exponent (n): {entrapment_exponent:.1f}")
        print(f"    Niyama coupling (α): {niyama_coupling_alpha:.2f}")
        print(f"    Gas generation range: {np.min(self.gas_generation):.4f} - {np.max(self.gas_generation):.4f}")
        print(f"    Gas entrapment range: {np.min(self.gas_entrapment):.4f} - {np.max(self.gas_entrapment):.4f}")
        print(f"    Gas porosity risk range: {np.min(self.gas_porosity_risk):.4f} - {np.max(self.gas_porosity_risk):.4f}")
        print(f"    Gas porosity hotspots: {self.gas_porosity_count:,} "
              f"({self.gas_porosity_count / n_voxels * 100:.1f}% of total)")

    def _generate_statistics(self):
        solidification_times = np.asarray(self.solidification_times, dtype=np.float32)
        finished = np.isfinite(solidification_times) & (solidification_times > 0)
        statistics: Dict[str, object] = {
            "mean_solidification_time": (
                float(np.mean(solidification_times[finished])) if np.any(finished) else 0.0
            ),
            "max_solidification_time": (
                float(np.max(solidification_times[finished])) if np.any(finished) else 0.0
            ),
        }
        n_voxels = len(solidification_times)
        part_types = getattr(self.mesh, "voxel_part_types", None)
        if part_types is not None:
            cast_mask = np.asarray(part_types, dtype=np.int32) == 0
        else:
            cast_mask = np.ones(n_voxels, dtype=bool)
        statistics["unfinished_voxel_count"] = int(np.sum(~finished & cast_mask))

        shrinkage_threshold = SHRINKAGE_STATISTICS_THRESHOLD

        if self.analysis_options.shrinkage and self.shrinkage_probability is not None:
            self.affected_voxel_count = np.sum(
                (self.shrinkage_probability > shrinkage_threshold) |
                (self.porosity_probability > shrinkage_threshold)
            )

            affected_mask = (self.shrinkage_probability > shrinkage_threshold) | \
                           (self.porosity_probability > shrinkage_threshold)

            if np.any(affected_mask):
                affected_volumes = self.mesh.voxel_volumes[affected_mask]
                affected_shrinkage = self.shrinkage_probability[affected_mask]
                affected_porosity = self.porosity_probability[affected_mask]

                combined_probability = np.maximum(affected_shrinkage, affected_porosity)

                self.total_shrinkage_volume = np.sum(affected_volumes * combined_probability)
            else:
                self.total_shrinkage_volume = 0.0

            print(f"  Shrinkage threshold: {shrinkage_threshold*100:.1f}%")
            print(f"  Shrinkage-affected voxels: {self.affected_voxel_count:,}")

        if self.analysis_options.hotspot and self.hotspot_intensity is not None:
            hotspot_threshold = 0.3
            self.hotspot_count = np.sum(self.hotspot_intensity > hotspot_threshold)

        if self.analysis_options.niyama and self.niyama_values is not None:
            alloy_key = self.alloy.name.split()[0]
            niyama_threshold = NIYAMA_THRESHOLDS.get(alloy_key, 1.0)
            self.niyama_defect_count = np.sum(self.niyama_defect_risk)

        if self.analysis_options.hot_tear and self.hot_tear_risk is not None:
            self.hot_tear_count = np.sum(self.hot_tear_zones)
            if self.hot_tear_count > 0:
                max_risk = np.max(self.hot_tear_risk)
                mean_risk = np.mean(self.hot_tear_risk[self.hot_tear_zones])
                print(f"  Hot tear zones: {self.hot_tear_count:,} voxels")
                print(f"  Max hot tear risk: {max_risk:.3f}, Mean in zones: {mean_risk:.3f}")

        if self.mesh.voxel_part_types is not None:
            cast_voxels = np.sum(self.mesh.voxel_part_types == PART_TYPE_CAST)
            riser_voxels = np.sum(self.mesh.voxel_part_types == PART_TYPE_RISER)
            chill_voxels = np.sum(self.mesh.voxel_part_types == PART_TYPE_CHILL)
            print(f"  Part breakdown: CAST={cast_voxels:,}, RISER={riser_voxels:,}, CHILL={chill_voxels:,}")

        return statistics

    def get_shrinkage_percentage(self) -> float:
        n_points = len(self.mesh.voxel_centers)
        if n_points == 0:
            return 0.0
        if not _has_cut_cells(self.mesh):
            return float(self.affected_voxel_count / n_points * 100)
        shrink = getattr(self, "shrinkage_probability", None)
        poro = getattr(self, "porosity_probability", None)
        if shrink is None or poro is None:
            return 0.0
        threshold = SHRINKAGE_STATISTICS_THRESHOLD
        affected = (np.asarray(shrink) > threshold) | (np.asarray(poro) > threshold)
        volumes = np.asarray(self.mesh.voxel_volumes, dtype=np.float64)
        total = float(np.sum(volumes))
        return float(np.sum(volumes[affected]) / total * 100.0) if total > 0.0 else 0.0

    def _get_normalized_heights(self) -> np.ndarray:
        z_coords = self.mesh.voxel_centers[:, 2]
        z_min = np.min(z_coords)
        z_max = np.max(z_coords)

        if z_max > z_min:
            return (z_coords - z_min) / (z_max - z_min)
        else:
            return np.zeros_like(z_coords)

    def get_shrinkage_voxels(self, threshold: float = 0.05) -> np.ndarray:
        shrinkage_mask = (
            (self.shrinkage_probability > threshold) |
            (self.porosity_probability > threshold)
        )
        part_types = getattr(self.mesh, "voxel_part_types", None)
        if part_types is not None and len(part_types) == len(shrinkage_mask):
            chill_mask = np.isin(np.asarray(part_types, dtype=np.int32), (PART_TYPE_CHILL, PART_TYPE_CORE))
            shrinkage_mask = shrinkage_mask & ~chill_mask
        return np.where(shrinkage_mask)[0]

    def get_shrinkage_intensity(self, voxel_index: int) -> float:
        intensities = self.get_shrinkage_display_intensity()
        if voxel_index < 0 or voxel_index >= len(intensities):
            return 0.0
        return float(intensities[voxel_index])

    def _mask_display_risk(self, values: Optional[np.ndarray]) -> np.ndarray:
        if values is None:
            return np.zeros(0, dtype=np.float32)
        display = np.asarray(values, dtype=np.float32).copy()
        part_types = getattr(self.mesh, "voxel_part_types", None)
        if part_types is not None and len(part_types) == len(display):
            chill_mask = np.isin(np.asarray(part_types, dtype=np.int32), (PART_TYPE_CHILL, PART_TYPE_CORE))
            if np.any(chill_mask):
                display[chill_mask] = 0.0
            orphan_mask = self._get_orphan_metal_mask()
            if len(orphan_mask) == len(display) and np.any(orphan_mask):
                display[orphan_mask] = 0.0
        return np.clip(display, 0.0, 1.0).astype(np.float32, copy=False)

    def get_shrink_display_intensity(self) -> np.ndarray:
        return self._mask_display_risk(getattr(self, "shrinkage_probability", None))

    def get_microporosity_display_intensity(self) -> np.ndarray:
        return self._mask_display_risk(getattr(self, "porosity_probability", None))

    def get_shrinkage_display_intensity(self) -> np.ndarray:
        shrink = getattr(self, "shrinkage_probability", None)
        poro = getattr(self, "porosity_probability", None)

        if shrink is None and poro is None:
            return np.zeros(0, dtype=np.float32)
        if shrink is None:
            shrink_arr = np.zeros_like(np.asarray(poro, dtype=np.float32))
        else:
            shrink_arr = np.asarray(shrink, dtype=np.float32)
        if poro is None:
            poro_arr = np.zeros_like(shrink_arr, dtype=np.float32)
        else:
            poro_arr = np.asarray(poro, dtype=np.float32)

        if len(poro_arr) != len(shrink_arr):
            min_len = min(len(shrink_arr), len(poro_arr))
            shrink_arr = shrink_arr[:min_len]
            poro_arr = poro_arr[:min_len]

        display = np.maximum(shrink_arr, poro_arr).astype(np.float32, copy=True)
        part_types = getattr(self.mesh, "voxel_part_types", None)
        if part_types is not None and len(part_types) == len(display):
            part_types_arr = np.asarray(part_types, dtype=np.int32)
            chill_mask = np.isin(part_types_arr, (PART_TYPE_CHILL, PART_TYPE_CORE))
            if np.any(chill_mask):
                display[chill_mask] = 0.0
            orphan_mask = self._get_orphan_metal_mask()
            if len(orphan_mask) == len(display) and np.any(orphan_mask):
                display[orphan_mask] = 0.0

        return np.clip(display, 0.0, 1.0).astype(np.float32, copy=False)

    def get_hotspot_voxels(self, threshold: float = 0.3) -> np.ndarray:
        if self.hotspot_intensity is None:
            return np.array([], dtype=int)
        return np.where(self.hotspot_intensity > threshold)[0]

    def get_niyama_defect_voxels(self) -> np.ndarray:
        if self.niyama_defect_risk is None:
            return np.array([], dtype=int)
        return np.where(self.niyama_defect_risk)[0]

    def get_fraction_liquid_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        if (
            self.solidification_start_times is not None
            and self.local_solidification_time is not None
            and self.solidification_end_times is not None
            and np.any(np.asarray(self.local_solidification_time, dtype=np.float32) > 0.0)
        ):
            start_times = np.asarray(self.solidification_start_times, dtype=np.float32)
            local_times = np.asarray(self.local_solidification_time, dtype=np.float32)
            end_times = np.asarray(self.solidification_end_times, dtype=np.float32)
            fraction_solid = compute_schedule_fraction_solid(
                start_times,
                local_times,
                time_seconds,
                (local_times <= 0.0) & (end_times <= 0.0),
                curve_exponent=self._schedule_curve_exponent,
            )
            fraction_liquid = 1.0 - fraction_solid
            return self._mask_non_cast_fraction_liquid(fraction_liquid)

        if not self.fraction_liquid_times or not self.fraction_liquid_history:
            return None

        times = np.array(self.fraction_liquid_times)

        if time_seconds <= times[0]:
            fraction_liquid = 1.0 - np.asarray(self.fraction_liquid_history[0], dtype=np.float32)
            return self._mask_non_cast_fraction_liquid(fraction_liquid)
        if time_seconds >= times[-1]:
            fraction_liquid = 1.0 - np.asarray(self.fraction_liquid_history[-1], dtype=np.float32)
            return self._mask_non_cast_fraction_liquid(fraction_liquid)

        idx_after = np.searchsorted(times, time_seconds)
        idx_before = idx_after - 1

        t_before = times[idx_before]
        t_after = times[idx_after]
        alpha = (time_seconds - t_before) / (t_after - t_before) if t_after > t_before else 0.0

        state_before = np.asarray(self.fraction_liquid_history[idx_before], dtype=np.float32)
        state_after = np.asarray(self.fraction_liquid_history[idx_after], dtype=np.float32)
        interpolated_state = state_before * (1 - alpha) + state_after * alpha

        fraction_liquid = 1.0 - interpolated_state
        return self._mask_non_cast_fraction_liquid(fraction_liquid)

    def _mask_non_cast_fraction_liquid(self, fraction_liquid: np.ndarray) -> np.ndarray:
        part_types = getattr(self.mesh, "voxel_part_types", None)
        if part_types is not None and len(part_types) == len(fraction_liquid):
            non_cast = np.isin(np.asarray(part_types, dtype=np.int32), (PART_TYPE_CHILL, PART_TYPE_CORE))
            fraction_liquid = np.asarray(fraction_liquid, dtype=np.float32).copy()
            fraction_liquid[non_cast] = 0.0
            get_orphans = getattr(self, "_get_orphan_metal_mask", None)
            if callable(get_orphans):
                orphan_mask = get_orphans()
                if len(orphan_mask) == len(fraction_liquid) and np.any(orphan_mask):
                    fraction_liquid[orphan_mask] = 0.0
        return fraction_liquid

    def get_fraction_liquid_time_range(self) -> Tuple[float, float]:
        if not self.fraction_liquid_times:
            return (0.0, 0.0)
        return (self.fraction_liquid_times[0], self.fraction_liquid_times[-1])

    def get_fill_time_range(self) -> Tuple[float, float]:
        times = getattr(self, "_fill_arrival_times", None)
        if times is None:
            return (0.0, 0.0)
        times = np.asarray(times, dtype=np.float64)
        finite = times[times >= 0.0]
        if finite.size == 0:
            return (0.0, 0.0)
        return (0.0, float(np.max(finite)))

    def get_filling_summary(self) -> Optional[Dict[str, Any]]:
        return getattr(self, "_filling_summary", None)

    def get_fill_animation_frames(self) -> Tuple[Any, ...]:
        return getattr(self, "_fill_frames", ()) or ()

    def get_fill_animation_range(self) -> Tuple[float, float]:
        frames = self.get_fill_animation_frames()
        if not frames:
            return (0.0, 0.0)
        return (float(frames[0].time_s), float(frames[-1].time_s))

    def _calculate_feeding_paths(self):
        n_voxels = len(self.mesh.voxel_centers)

        riser_indices = self._get_riser_voxel_indices()

        if len(riser_indices) == 0:
            surface_indices = self.mesh.get_surface_voxel_indices()
            riser_indices = surface_indices

        analyzer = FeedingPathAnalyzer(self.mesh, self.alloy)

        if self.feeding_fraction_solid is not None:
            fraction_solid = np.clip(np.asarray(self.feeding_fraction_solid, dtype=np.float32), 0.0, 1.0)
        else:
            fraction_solid = 0.5 * np.ones(n_voxels, dtype=np.float32)

        self.feeding_path_result = analyzer.analyze(
            riser_indices,
            fraction_solid,
            show_progress=False,
            history=getattr(self, "fraction_liquid_history", None),
            history_times=getattr(self, "fraction_liquid_times", None),
            reference_times=self._estimate_feeding_reference_times(),
        )

        self.feeding_quality = self.feeding_path_result.feeding_quality
        self.feeding_blocked = self.feeding_path_result.blocked_voxels
        self.feeding_open_fraction = self.feeding_path_result.history_open_fraction
        self.feeding_latest_open_time = self.feeding_path_result.latest_open_time
        self.feeding_blocked_count = self.feeding_path_result.n_blocked

        if self.niyama_defect_risk is not None:
            self.combined_defect_risk = analyzer.get_combined_defect_risk(
                self.niyama_defect_risk
            )

        print(f"    Feeding path results: {self.feeding_path_result.fed_fraction*100:.1f}% fed")
        print(f"    Blocked voxels: {self.feeding_blocked_count:,}")

    def _calculate_stress_analysis(self):
        n_voxels = len(self.mesh.voxel_centers)

        analyzer = StressAnalyzer(self.mesh, self.alloy)

        fraction_solid = np.clip(self.mesh.voxel_states, 0, 1).astype(np.float32)

        self.stress_result = analyzer.analyze(
            np.asarray(self.solidification_times),
            fraction_solid,
            show_progress=False
        )

        self.effective_stress = self.stress_result.effective_stress
        self.distortion_magnitude = self.stress_result.distortion_magnitude
        self.crack_susceptibility = self.stress_result.crack_susceptibility

        sand_restraint_index = self._get_sand_restraint_index()
        restraint_offset = sand_restraint_index - 0.5
        crack_multiplier = np.clip(
            1.0 + 0.5 * restraint_offset * self._compute_surface_restraint_weight(),
            0.8,
            1.2,
        ).astype(np.float32)
        self.crack_susceptibility = np.clip(
            np.asarray(self.crack_susceptibility, dtype=np.float32) * crack_multiplier,
            0.0,
            1.0,
        )
        self.stress_result.crack_susceptibility = self.crack_susceptibility
        self.max_stress_mpa = self.stress_result.max_stress

        print(f"    Max stress: {self.max_stress_mpa:.1f} MPa")
        print(
            "    Sand restraint index: "
            f"{sand_restraint_index:.2f} (crack multiplier range: "
            f"{np.min(crack_multiplier):.3f} - {np.max(crack_multiplier):.3f})"
        )
        print(f"    Max distortion: {self.stress_result.max_distortion*1000:.3f} mils")

    def _calculate_microstructure(self):
        n_voxels = len(self.mesh.voxel_centers)

        predictor = MicrostructurePredictor(self.mesh, self.alloy)

        self.microstructure_result = predictor.predict_from_solidification(
            np.asarray(self.solidification_times),
            show_progress=False
        )

        self.sdas = self.microstructure_result.sdas
        self.predicted_uts = self.microstructure_result.predicted_uts
        self.avg_sdas_um = self.microstructure_result.avg_sdas

        print(f"    Average SDAS: {self.avg_sdas_um:.0f} μm")
        print(f"    SDAS range: {self.microstructure_result.min_sdas:.0f} - {self.microstructure_result.max_sdas:.0f} μm")

    def _calculate_chvorinov_times(self):
        n_voxels = len(self.mesh.voxel_centers)

        self.local_modulus = np.asarray(self.mesh.voxel_moduli).copy()

        if hasattr(self, '_mold_constant'):
            B = self._mold_constant
        else:
            B = 280.0

        self.chvorinov_time = B * (self.local_modulus ** 2)

        self.chvorinov_time = np.clip(self.chvorinov_time, 1.0, self.max_time)

    def get_chvorinov_time_at_voxel(self, voxel_index: int) -> float:
        if self.chvorinov_time is None:
            return 0.0
        return float(self.chvorinov_time[voxel_index])

    def get_defect_volume_estimate(self, threshold: float = 0.3) -> Dict:
        n_voxels = len(self.mesh.voxel_centers)

        if self.shrinkage_probability is not None and self.porosity_probability is not None:
            shrinkage_prob = np.nan_to_num(
                np.asarray(self.shrinkage_probability, dtype=np.float32), nan=0.0
            )
            porosity_prob = np.nan_to_num(
                np.asarray(self.porosity_probability, dtype=np.float32), nan=0.0
            )
            defect_mask = (shrinkage_prob > threshold) | (porosity_prob > threshold)
            combined_prob = np.maximum(shrinkage_prob, porosity_prob)
        elif self.shrinkage_probability is not None:
            shrinkage_prob = np.nan_to_num(
                np.asarray(self.shrinkage_probability, dtype=np.float32), nan=0.0
            )
            defect_mask = shrinkage_prob > threshold
            combined_prob = shrinkage_prob
        else:
            return {'total_volume_in3': 0.0, 'defect_count': 0, 'casting_volume_in3': 0.0}

        casting_volume = float(np.sum(self.mesh.voxel_volumes))

        if np.any(defect_mask):
            defect_volumes = self.mesh.voxel_volumes[defect_mask]
            defect_probs = combined_prob[defect_mask]

            weighted_volume = float(np.sum(defect_volumes * defect_probs))

            max_defect_volume = float(np.sum(defect_volumes))

            defect_count = int(np.sum(defect_mask))
        else:
            weighted_volume = 0.0
            max_defect_volume = 0.0
            defect_count = 0

        return {
            'total_volume_in3': weighted_volume,
            'max_volume_in3': max_defect_volume,
            'defect_count': defect_count,
            'casting_volume_in3': casting_volume,
            'defect_percentage': weighted_volume / casting_volume * 100 if casting_volume > 0 else 0,
            'threshold_used': threshold
        }

    def get_feeding_arrows(self, sample_fraction: float = 0.05) -> List:
        if self.feeding_path_result is None:
            return []

        try:
            analyzer = FeedingPathAnalyzer(self.mesh, self.alloy)
            analyzer.result = self.feeding_path_result
            return analyzer.get_feeding_arrows(sample_fraction)
        except Exception:
            return []

    def export_results(self) -> Dict:
        results = {
            'alloy': self.alloy.name,
            'simulation_time': self.time,
            'voxel_count': len(self.mesh.voxel_centers),
            'solidification_times': self.solidification_times,
            'solidification_start_times': self.solidification_start_times,
            'solidification_end_times': self.solidification_end_times,
            'local_solidification_time': self.local_solidification_time,
            'analysis_options': {
                'shrinkage': self.analysis_options.shrinkage,
                'hotspot': self.analysis_options.hotspot,
                'niyama': self.analysis_options.niyama,
                'hot_tear': self.analysis_options.hot_tear,
                'sand_burn_in': self.analysis_options.sand_burn_in,
                'gas_porosity': self.analysis_options.gas_porosity,
                'fraction_liquid': self.analysis_options.fraction_liquid
            }
        }

        if self.analysis_options.shrinkage:
            shrink_scalar = getattr(self, 'shrink_scalar', None)
            if shrink_scalar is None:
                shrink_scalar = self._get_shrink_scalar()
            results['affected_voxels'] = self.affected_voxel_count
            results['shrinkage_volume'] = self.total_shrinkage_volume
            if _has_cut_cells(self.mesh):
                results['shrinkage_percentage'] = SolidificationEngine.get_shrinkage_percentage(self)
            else:
                results['shrinkage_percentage'] = (self.affected_voxel_count /
                                                   len(self.mesh.voxel_centers) * 100)
            results['shrink_scalar'] = float(shrink_scalar)
            results['shrink_scalar_formula'] = getattr(self, 'shrink_scalar_formula', SHRINK_SCALAR_FORMULA)
            results['shrinkage_risk'] = self.shrinkage_risk
            results['porosity_risk'] = self.porosity_risk
            results['shrinkage_data'] = self.shrinkage_probability
            results['porosity_data'] = self.porosity_probability
            results['shrinkage_raw_data'] = (
                self.raw_shrinkage_probability
                if self.raw_shrinkage_probability is not None
                else self.shrinkage_risk
            )
            results['porosity_raw_data'] = (
                self.raw_porosity_probability
                if self.raw_porosity_probability is not None
                else self.porosity_risk
            )

        if self.analysis_options.hotspot:
            results['hotspot_count'] = self.hotspot_count
            results['hotspot_intensity'] = self.hotspot_intensity

        if self.analysis_options.niyama:
            alloy_key = self.alloy.name.split()[0]
            results['niyama_threshold'] = float(
                getattr(self.alloy, "niyama_threshold", NIYAMA_THRESHOLDS.get(alloy_key, 1.0))
            )
            results['niyama_defect_count'] = self.niyama_defect_count
            results['niyama_values'] = self.niyama_values
            results['niyama_risk'] = self.niyama_risk
            results['niyama_defect_risk'] = self.niyama_defect_risk
            results['combined_feed_risk'] = self.combined_feed_risk
            results['niyama_quality'] = self.niyama_quality
            results['niyama_metadata'] = self.niyama_metadata
            results['thermal_gradients'] = self.thermal_gradients
            results['cooling_rates'] = self.cooling_rates

        if self.analysis_options.hot_tear:
            results['hot_tear_count'] = self.hot_tear_count
            results['hot_tear_risk'] = self.hot_tear_risk
            results['hot_tear_zones'] = self.hot_tear_zones

        if self.analysis_options.sand_burn_in:
            results['sand_burn_in_count'] = self.sand_burn_in_count
            results['sand_burn_in_risk'] = self.sand_burn_in_risk
            results['sand_burn_in_zones'] = self.sand_burn_in_zones

        if self.analysis_options.gas_porosity:
            results['gas_porosity_count'] = self.gas_porosity_count
            results['gas_generation'] = self.gas_generation
            results['gas_entrapment'] = self.gas_entrapment
            results['gas_porosity_risk'] = self.gas_porosity_risk
            results['gas_porosity_hotspots'] = self.gas_porosity_hotspots

        if self.analysis_options.fraction_liquid:
            results['fraction_liquid_times'] = self.fraction_liquid_times
            results['fraction_liquid_snapshots'] = len(self.fraction_liquid_history)

        if self.analysis_options.filling and self._fill_arrival_times is not None:
            fill_times = np.asarray(self._fill_arrival_times, dtype=np.float64)
            filled_mask = fill_times >= 0.0
            results['fill_time_max_s'] = (
                float(np.max(fill_times[filled_mask])) if np.any(filled_mask) else 0.0
            )
            results['fill_time_filled_voxels'] = int(np.sum(filled_mask))
            results['fill_time_unfilled_voxels'] = int(
                np.sum(np.isfinite(fill_times) & ~filled_mask)
            )

        if self.analysis_options.feeding_path and self.feeding_path_result is not None:
            results['feeding_quality'] = self.feeding_quality
            results['feeding_blocked_count'] = self.feeding_blocked_count
            results['fed_fraction'] = self.feeding_path_result.fed_fraction

        if self.analysis_options.stress_analysis and self.stress_result is not None:
            results['max_stress_mpa'] = self.max_stress_mpa
            results['effective_stress'] = self.effective_stress
            results['distortion_magnitude'] = self.distortion_magnitude
            results['crack_susceptibility'] = self.crack_susceptibility

        if self.analysis_options.microstructure and self.microstructure_result is not None:
            results['avg_sdas_um'] = self.avg_sdas_um
            results['sdas'] = self.sdas
            results['predicted_uts'] = self.predicted_uts

        if self.chvorinov_time is not None:
            results['chvorinov_time'] = self.chvorinov_time
            results['local_modulus'] = self.local_modulus

        results['defect_volume_estimate'] = self.get_defect_volume_estimate()

        return results
