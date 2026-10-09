"""Thermal-coupled solve of ``SolidificationEngine`` on a ``TetVolumeMesh`` (contract C4).

``core/solidification.py`` and ``core/solidification_loop.py`` dispatch here
when ``mesh.mesh_type == "tet"``. The DOF is the mesh node: every engine array
written here has one entry per node, in node order, with the voxel-path dtypes.

The integration loop mirrors ``_run_thermal_coupled_integration`` (sentinels,
termination, fraction-liquid snapshots, feeding fraction solid, peak cooling
rate, Niyama capture) with three differences:

1. The heat step is the implicit P1 FEM step of
   :class:`physics.tet_heat_transfer.TetHeatTransferModel`.
2. Latent heat is released by an exact per-node enthalpy inversion
   (:func:`invert_tet_enthalpy`), not the voxel loop's correction, so
   ``sum(C_i T_i + L_i (1 - fs_i))`` is conserved on every step size.
3. The time step adapts: it grows x1.25 per accepted step up to
   ``settings.mesh.tet_max_time_step_s`` and a step whose largest metal-node
   |dT| exceeds ``settings.mesh.tet_max_delta_t_c`` is rejected and retried
   at half the size. The loop runs on simulated time up to ``engine.max_time``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from config.settings import get_settings
from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_FILTER,
    PART_TYPE_RISER,
    PART_TYPE_RISER_POST,
    is_flow_path,
    is_metal,
)
from core.solidification import (
    CHILL_RHO_CP_J_M3_K,
    CHILL_THERMAL_DIFFUSIVITY_M2_S,
    SOLVER_MODE_THERMAL_COUPLED,
)
from core.solidification_loop import (
    NIYAMA_EVAL_FRACTION_SOLID,
    THREAD_BUDGET_CHECK_INTERVAL,
    ProgressCallback,
    _apply_live_thread_budget,
)
from core.stepping import should_capture_fraction_history
from physics.solidification_physics import SolidificationModel
from physics.tet_heat_transfer import TetHeatTransferModel, TetMaterial
from physics.units import inches_to_meters, inches_to_mm

#: Fallbacks for ``settings.mesh.tet_max_time_step_s`` / ``tet_max_delta_t_c``.
#: Total solidification time is within 1.5 % of the dt-converged value on the
#: 2x2x1 box and the box + riser (docs/tet_migration/wave3_enthalpy.md).
DEFAULT_TET_MAX_TIME_STEP_S = 2.0
DEFAULT_TET_MAX_DELTA_T_C = 15.0
#: Accepted-step growth factor and the smallest step a rejection may produce.
TET_TIME_STEP_GROWTH = 1.25
TET_MIN_TIME_STEP_S = 1e-3

METAL_MATERIAL_INDEX = 0
CHILL_MATERIAL_INDEX = 1

#: K/m -> K/mm, from the inch conversions (1 mm = IN_TO_M / IN_TO_MM metres).
_PER_M_TO_PER_MM = inches_to_meters(1.0) / inches_to_mm(1.0)


@dataclass(frozen=True)
class TetBoundaryGroupSpec:
    """One Robin group: boundary-face rows with the voxel-path h0 and decay floor."""

    label: str
    face_indices: np.ndarray
    htc: float
    floor: Optional[float]  # None = air-gap floor (``_mold_interface_decay`` default)


def resolve_tet_solver_mode(engine: Any, requested: str) -> str:
    """A tet mesh always runs thermal_coupled. Warn once when another mode was requested."""
    if requested != SOLVER_MODE_THERMAL_COUPLED and not getattr(
        engine, "_tet_solver_mode_warned", False
    ):
        engine._tet_solver_mode_warned = True
        message = (
            f"solver_mode {requested!r} is not available on a tet mesh; "
            f"running {SOLVER_MODE_THERMAL_COUPLED!r}"
        )
        engine._logger.warning(message)
        print(f"  WARNING: {message}")
    return SOLVER_MODE_THERMAL_COUPLED


def tet_materials(engine: Any) -> List[TetMaterial]:
    """[metal, chill] materials. Metal uses the voxel path's 6-point alloy curves."""
    curves = engine._alloy_property_curves()
    alloy = engine.alloy
    metal = TetMaterial(
        temp_points=curves["temp_points"],
        conductivity=curves["conductivity"],
        rho_cp=curves["density"] * curves["specific_heat"],
        latent_heat_vol=float(
            alloy.latent_heat * (alloy.density_liquid + alloy.density_solid) / 2.0
        ),
    )
    chill = TetMaterial(
        temp_points=np.zeros(1),
        conductivity=np.array([CHILL_THERMAL_DIFFUSIVITY_M2_S * CHILL_RHO_CP_J_M3_K]),
        rho_cp=np.array([CHILL_RHO_CP_J_M3_K]),
    )
    return [metal, chill]


def tet_material_indices(tet_part_types: np.ndarray) -> np.ndarray:
    """(T,) material index per tet. Raises ValueError for codes with no thermal material."""
    codes = np.asarray(tet_part_types)
    metal = is_metal(codes)
    chill = codes == PART_TYPE_CHILL
    unknown = ~(metal | chill)
    if np.any(unknown):
        raise ValueError(
            f"tet part codes {sorted(set(codes[unknown].tolist()))} have no thermal material"
        )
    return np.where(metal, METAL_MATERIAL_INDEX, CHILL_MATERIAL_INDEX).astype(np.int64)


def tet_boundary_groups(engine: Any) -> List[TetBoundaryGroupSpec]:
    """Boundary groups by face owner code and contact code, with the voxel-path h0 per part type.

    Faces against a removed CORE use the CORE h0 (``htc_cast_interface``) and
    faces against a removed FILTER use ``htc_gating``, whatever owns them.
    Raises ValueError if a face fits no group, so no face is silently adiabatic.
    """
    tb = engine.analysis_options.solidification_params.thermal_boundary
    owner = np.asarray(engine.mesh.boundary_face_part_types).astype(np.int64)
    contact = np.asarray(engine.mesh.boundary_face_contact_codes).astype(np.int64)
    core_contact = contact == PART_TYPE_CORE
    filter_contact = contact == PART_TYPE_FILTER
    mold_contact = ~(core_contact | filter_contact)
    rules: Tuple[Tuple[str, np.ndarray, float, Optional[float]], ...] = (
        ("cast", mold_contact & (owner == PART_TYPE_CAST), float(tb.htc_cast_interface), None),
        ("riser_post", mold_contact & (owner == PART_TYPE_RISER_POST), float(tb.htc_riser_post), None),
        ("riser", mold_contact & (owner == PART_TYPE_RISER), float(tb.htc_riser), None),
        ("flow_path", mold_contact & is_flow_path(owner), float(tb.htc_gating), None),
        ("core_contact", core_contact, float(tb.htc_cast_interface), None),
        ("filter_contact", filter_contact, float(tb.htc_gating), None),
        (
            "chill",
            mold_contact & (owner == PART_TYPE_CHILL),
            float(getattr(tb, "chill_ambient_htc", 500.0)),
            float(getattr(tb, "chill_htc_floor_fraction", 0.7)),
        ),
    )
    assigned = np.zeros(len(owner), dtype=bool)
    groups: List[TetBoundaryGroupSpec] = []
    for label, mask, htc, floor in rules:
        assigned |= mask
        if np.any(mask):
            groups.append(TetBoundaryGroupSpec(label, np.flatnonzero(mask), htc, floor))
    if not np.all(assigned):
        raise ValueError(
            "boundary faces with owner codes "
            f"{sorted(set(owner[~assigned].tolist()))} have no boundary condition"
        )
    return groups


def build_tet_heat_model(engine: Any) -> TetHeatTransferModel:
    """Heat model in SI from the engine's tet mesh, with every boundary group registered."""
    mesh = engine.mesh
    tb = engine.analysis_options.solidification_params.thermal_boundary
    model = TetHeatTransferModel(
        inches_to_meters(np.asarray(mesh.node_xyz, dtype=np.float64)),
        mesh.tets,
        tet_material_indices(mesh.tet_part_types),
        tet_materials(engine),
        mesh.boundary_faces,
    )
    for group in tet_boundary_groups(engine):
        model.add_boundary_group(
            group.face_indices,
            htc=group.htc,
            t_inf=float(tb.mold_temperature),
            time_function=engine._mold_interface_decay(group.htc, group.floor),
        )
    return model


def initialize_tet_thermal_models(engine: Any, *, show_progress: bool) -> None:
    """Tet counterpart of ``SolidificationEngine._initialize_thermal_models``."""
    mesh = engine.mesh
    n_nodes = int(mesh.get_voxel_count())
    voxel_size_in = engine._get_mean_voxel_size_inches()
    engine._thermal_voxel_size_m = max(inches_to_meters(voxel_size_in), 1e-6)

    heat_model = build_tet_heat_model(engine)
    engine._thermal_heat_model = heat_model
    engine._thermal_solidification_model = SolidificationModel(n_voxels=n_nodes, alloy=engine.alloy)
    engine._allocate_thermal_tracking_arrays(n_nodes)

    tb = engine.analysis_options.solidification_params.thermal_boundary
    chill_indices = engine._get_chill_voxel_indices()
    initial_temperatures = engine._initial_thermal_temperatures(
        n_nodes, chill_indices, tb.chill_temperature
    )
    heat_model.set_initial_temperature(initial_temperatures)
    engine._preset_chill_solid_state(chill_indices)

    engine._thermal_latest_temperatures = initial_temperatures.astype(np.float32, copy=True)
    engine._thermal_latest_gradients = np.zeros(n_nodes, dtype=np.float32)

    if show_progress:
        max_dt, max_delta = tet_time_step_limits()
        print("  Thermal-coupled mode active (tet mesh, implicit P1 FEM)")
        print(f"    Nodes: {n_nodes:,}, tets: {len(mesh.tets):,}, boundary faces: {len(mesh.boundary_faces):,}")
        print(f"    Mean local edge length: {voxel_size_in:.4f}\"")
        print(f"    Surface mold temperature: {tb.mold_temperature:.1f} C")
        print(f"    Chill temperature: {tb.chill_temperature:.1f} C")
        print(f"    Thermal solver backend: {heat_model.backend_description}")
        print(
            f"    Time step: start {engine.time_step:.3g}s, max {max_dt:.3g}s, "
            f"max metal dT per step {max_delta:.3g} C"
        )


def invert_tet_enthalpy(
    T_post: np.ndarray,
    fs_prev: np.ndarray,
    beta: np.ndarray,
    T_liq: float,
    T_sol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-node (T_new, fs_new) that hold the step's enthalpy exactly. Arrays in degC and K.

    With ``beta_i = L_i / C_i`` the node enthalpy per unit capacity is
    ``h = T + beta * (1 - fs)``. The heat step left ``h_new = T_post + beta *
    (1 - fs_prev)``. Solve ``h_new = T_new + beta * (1 - fs_new)`` with
    ``fs_new = max(fs_prev, lever(T_new))``, the irreversible lever rule that
    ``SolidificationModel.update`` applies. The right side is continuous and
    strictly increasing in ``T_new`` with three linear segments:

    - ``T_new >= T_p = T_liq - fs_prev * dT``: fs stays fs_prev, so T_new = T_post.
    - ``T_sol <= T_new < T_p``: mushy, T_new = (T_post * dT + beta * T_p) / (dT + beta).
    - ``T_new < T_sol``: fully solid, T_new = h_new.

    ``fs_new`` is computed with the model's own formula from ``T_new``, so
    ``SolidificationModel.update(T_new)`` reproduces it bit for bit.
    """
    T_post = np.asarray(T_post, dtype=np.float64)
    fs_prev = np.clip(np.asarray(fs_prev, dtype=np.float64), 0.0, 1.0)
    beta = np.asarray(beta, dtype=np.float64)
    dT = max(float(T_liq) - float(T_sol), 1e-6)
    h_new = T_post + beta * (1.0 - fs_prev)
    T_p = T_liq - fs_prev * dT
    T_mush = (T_post * dT + beta * T_p) / (dT + beta)
    T_new = np.where(T_post >= T_p, T_post, np.where(h_new < T_sol, h_new, T_mush))
    fs_new = np.maximum(fs_prev, np.clip((T_liq - T_new) / dT, 0.0, 1.0))
    return T_new, fs_new


def tet_time_step_limits() -> Tuple[float, float]:
    """(max time step s, max metal |dT| per step C) from ``settings.mesh``."""
    mesh_settings = getattr(get_settings(), "mesh", None)
    max_dt = float(getattr(mesh_settings, "tet_max_time_step_s", DEFAULT_TET_MAX_TIME_STEP_S))
    max_delta = float(getattr(mesh_settings, "tet_max_delta_t_c", DEFAULT_TET_MAX_DELTA_T_C))
    return max_dt, max_delta


def _capture_tet_niyama_step(
    engine: Any,
    *,
    previous_states: np.ndarray,
    current_states: np.ndarray,
    previous_temperatures: np.ndarray,
    temperatures: np.ndarray,
    cooling_rate_step: np.ndarray,
    gradients_c_per_mm: np.ndarray,
    cast_mask: np.ndarray,
    dt: float,
) -> None:
    """Sample Niyama inputs once when each CAST node crosses the evaluation fraction solid.

    Same selection and interpolation as ``_capture_niyama_history_step``. The
    gradient is the P1 nodal gradient, so quality is 2 for every sampled node.
    """
    sampled = getattr(engine, "_thermal_niyama_sampled", None)
    if sampled is None:
        return
    eval_fs = float(getattr(engine, "niyama_eval_fraction_solid", NIYAMA_EVAL_FRACTION_SOLID))
    crossed = (~sampled) & (previous_states < eval_fs) & (current_states >= eval_fs) & cast_mask
    target = np.flatnonzero(crossed)
    if len(target) == 0:
        return

    denom = np.maximum(current_states[target] - previous_states[target], 1e-9)
    step_fraction = np.clip((eval_fs - previous_states[target]) / denom, 0.0, 1.0)
    eval_temperatures = previous_temperatures[target] + (
        temperatures[target] - previous_temperatures[target]
    ) * step_fraction
    eval_times = (engine.time - dt) + dt * step_fraction
    gradients = gradients_c_per_mm[target].astype(np.float32)
    rates = np.maximum(np.asarray(cooling_rate_step[target], dtype=np.float32), 1e-6)

    engine._thermal_niyama_gradient_at_eval[target] = gradients
    engine._thermal_niyama_cooling_rate_at_eval[target] = rates
    engine._thermal_niyama_temperature_at_eval[target] = eval_temperatures.astype(np.float32)
    engine._thermal_niyama_time_at_eval[target] = eval_times.astype(np.float32)
    engine._thermal_niyama_values_at_eval[target] = (gradients / np.sqrt(rates)).astype(np.float32)
    engine._thermal_niyama_quality[target] = 2
    sampled[target] = True


def run_tet_thermal_integration(
    engine: Any,
    *,
    show_progress: bool,
    progress_callback: ProgressCallback,
) -> int:
    """Tet counterpart of ``_run_thermal_coupled_integration``. Returns accepted steps."""
    heat_model = getattr(engine, "_thermal_heat_model", None)
    solidification_model = getattr(engine, "_thermal_solidification_model", None)
    if not isinstance(heat_model, TetHeatTransferModel) or solidification_model is None:
        raise RuntimeError("Tet thermal integration requires initialized tet thermal models.")

    max_dt, max_delta_t = tet_time_step_limits()
    max_time = float(engine.max_time)
    dt = min(float(engine.time_step), max_dt)

    iteration = 0
    rejected_steps = 0
    last_progress_report = 0
    last_budget_check = time.monotonic()
    last_snapshot_progress = -1.0
    progress_interval = 1.0
    next_print_time = max_time / 10.0
    capture_fraction_history = should_capture_fraction_history(engine)

    if capture_fraction_history:
        engine.fraction_liquid_history = [np.asarray(engine.mesh.voxel_states, dtype=np.float32).copy()]
        engine.fraction_liquid_times = [0.0]

    # Same gate as the voxel loop: gradients and Niyama samples have consumers
    # only in these analyses.
    needs_thermal_history = bool(
        getattr(engine.analysis_options, "niyama", False)
        or getattr(engine.analysis_options, "hot_tear", False)
        or getattr(engine.analysis_options, "shrinkage", True)
        or getattr(engine.analysis_options, "feeding_path", False)
    )

    n_nodes = heat_model.n_nodes
    part_types = np.asarray(engine.mesh.voxel_part_types)
    metal_mask = is_metal(part_types)
    cast_mask = part_types == PART_TYPE_CAST
    latent_capacity = heat_model.node_latent_capacity()
    latent_nodes = np.flatnonzero(latent_capacity > 0.0)
    latent_of_latent_nodes = latent_capacity[latent_nodes]
    T_liq = float(engine.alloy.liquidus_temp)
    T_sol = float(engine.alloy.solidus_temp)
    T_final = np.empty(n_nodes, dtype=np.float64)
    fs_prev = np.empty(n_nodes, dtype=np.float64)

    while heat_model.time < max_time - 1e-9:
        previous_states = np.asarray(engine.mesh.voxel_states, dtype=np.float32).copy()
        if np.all(previous_states >= 0.99):
            break

        step_dt = min(dt, max_time - heat_model.time)
        previous_time = heat_model.time
        previous_temperatures = heat_model.temperatures.copy()
        heat_model.step(step_dt)  # diffusion + Robin only

        # Operator-split latent heat: exact per-node enthalpy inversion with
        # beta_i = L_i / C_i(T_post) (K per unit fraction solid). Nodes without
        # latent heat keep T_post. solidification_model.update(T_final) below
        # recomputes the same fs from T_final.
        np.copyto(fs_prev, np.asarray(solidification_model.fraction_solid, dtype=np.float64))
        T_post = heat_model.temperatures
        np.copyto(T_final, T_post)
        if latent_nodes.size:
            beta = latent_of_latent_nodes / heat_model.node_heat_capacity()[latent_nodes]
            T_final[latent_nodes], _ = invert_tet_enthalpy(
                T_post[latent_nodes], fs_prev[latent_nodes], beta, T_liq, T_sol
            )

        if np.any(metal_mask) and step_dt > TET_MIN_TIME_STEP_S:
            max_change = float(np.max(np.abs(T_final[metal_mask] - previous_temperatures[metal_mask])))
            if max_change > max_delta_t:
                heat_model.temperatures = previous_temperatures
                heat_model.time = previous_time
                dt = max(0.5 * step_dt, TET_MIN_TIME_STEP_S)
                rejected_steps += 1
                continue

        heat_model.temperatures[:] = T_final
        engine.time_step = step_dt
        solidification_model.update(T_final, step_dt)

        temperatures = np.asarray(heat_model.temperatures, dtype=np.float64)
        cooling_rate_step = np.maximum(heat_model.get_cooling_rate(previous_temperatures, step_dt), 0.0)

        current_states = np.asarray(solidification_model.fraction_solid, dtype=np.float32)
        engine.mesh.voxel_states[:] = current_states
        engine.time = heat_model.time
        iteration += 1
        dt = min(dt * TET_TIME_STEP_GROWTH, max_dt)

        start_times = np.asarray(solidification_model.solidification_time, dtype=np.float64)
        end_times = np.asarray(solidification_model._solidification_end, dtype=np.float64)
        finite_start = np.isfinite(start_times)
        finite_end = np.isfinite(end_times)
        finite_local = finite_start & finite_end
        local_times = np.full_like(start_times, np.inf)
        np.subtract(end_times, start_times, out=local_times, where=finite_local)

        engine.solidification_start_times[:] = np.where(finite_start, start_times, 0.0).astype(np.float32)
        engine.solidification_end_times[:] = np.where(finite_end, end_times, np.nan).astype(np.float32)
        engine.local_solidification_time[:] = np.where(
            finite_local,
            local_times,
            np.where(finite_start, np.maximum(engine.time - start_times, step_dt), np.nan),
        ).astype(np.float32)
        engine.solidification_times[:] = engine.solidification_end_times

        if getattr(engine, "feeding_fraction_solid", None) is not None:
            critical_fs = float(getattr(engine.alloy, "critical_fraction_solid", 1.0))
            crossed_feeding_limit = (
                (previous_states < critical_fs)
                & (current_states >= critical_fs)
                & np.isnan(engine.feeding_fraction_solid)
            )
            full_solid_crossing = crossed_feeding_limit & (current_states >= 0.99)
            engine.feeding_fraction_solid[full_solid_crossing] = critical_fs
            engine.feeding_fraction_solid[crossed_feeding_limit & ~full_solid_crossing] = (
                current_states[crossed_feeding_limit & ~full_solid_crossing]
            )

        if getattr(engine, "_thermal_peak_cooling_rates", None) is not None:
            engine._thermal_peak_cooling_rates[:] = np.maximum(
                np.asarray(engine._thermal_peak_cooling_rates, dtype=np.float32),
                cooling_rate_step.astype(np.float32),
            )
            engine.cooling_rates = np.asarray(engine._thermal_peak_cooling_rates, dtype=np.float32).copy()

        engine._thermal_latest_temperatures = temperatures.astype(np.float32, copy=True)
        if needs_thermal_history:
            gradients_c_per_mm = heat_model.gradient_magnitude() * _PER_M_TO_PER_MM
            _capture_tet_niyama_step(
                engine,
                previous_states=previous_states,
                current_states=current_states,
                previous_temperatures=previous_temperatures,
                temperatures=temperatures,
                cooling_rate_step=cooling_rate_step,
                gradients_c_per_mm=gradients_c_per_mm,
                cast_mask=cast_mask,
                dt=step_dt,
            )
            engine._thermal_latest_gradients = gradients_c_per_mm.astype(np.float32)
            engine.thermal_gradients = engine._thermal_latest_gradients.copy()

        current_progress_pct = float(np.mean(current_states) * 100.0)
        if capture_fraction_history and current_progress_pct - last_snapshot_progress >= progress_interval:
            engine.fraction_liquid_history.append(current_states.copy())
            engine.fraction_liquid_times.append(engine.time)
            last_snapshot_progress = current_progress_pct

        if progress_callback and int(current_progress_pct) > last_progress_report:
            progress_callback(int(current_progress_pct), 100, "Solidification")
            last_progress_report = int(current_progress_pct)

        if show_progress and engine.time >= next_print_time:
            next_print_time += max_time / 10.0
            print(
                f"    Progress: {current_progress_pct:.1f}% solidified "
                f"(t={engine.time:.1f}s, dt={step_dt:.3g}s, "
                f"Tmin={np.min(temperatures):.1f}C, Tmax={np.max(temperatures):.1f}C)"
            )

        if time.monotonic() - last_budget_check >= THREAD_BUDGET_CHECK_INTERVAL:
            last_budget_check = time.monotonic()
            _apply_live_thread_budget(engine)

    if capture_fraction_history:
        engine.fraction_liquid_history.append(np.asarray(engine.mesh.voxel_states, dtype=np.float32).copy())
        engine.fraction_liquid_times.append(engine.time)

    unfinished = np.asarray(engine.mesh.voxel_states, dtype=np.float32) < 0.99
    engine.solidification_end_times[unfinished] = np.nan
    engine.solidification_times[unfinished] = np.nan

    if getattr(engine, "feeding_fraction_solid", None) is not None:
        unresolved = np.isnan(engine.feeding_fraction_solid)
        engine.feeding_fraction_solid[unresolved] = np.asarray(engine.mesh.voxel_states, dtype=np.float32)[unresolved]

    if progress_callback:
        progress_callback(100, 100, "Solidification")

    engine._logger.info(
        "Solidification simulation completed",
        context={
            "total_time_seconds": engine.time,
            "iterations": iteration,
            "rejected_steps": rejected_steps,
            "alloy": engine.alloy.name,
            "voxel_count": len(engine.mesh.voxel_states),
            "mode": SOLVER_MODE_THERMAL_COUPLED,
            "mesh_type": "tet",
        },
    )

    if show_progress:
        print(
            f"  ✓ Solidification complete at t={engine.time:.1f}s "
            f"({iteration} steps, {rejected_steps} rejected)"
        )
        if engine.analysis_options.fraction_liquid:
            print(f"    Fraction liquid snapshots: {len(engine.fraction_liquid_times)}")

    return iteration
