# Wave 2: engine integration (contract C4)

`SolidificationEngine.run_simulation()` runs end to end on a `TetVolumeMesh` with default `AnalysisOptions` and with every analysis on. `export_results()` returns 57 keys. Nothing is committed or staged.

## Files changed

- `core/tet_thermal.py` (new). Holds all tet logic: `resolve_tet_solver_mode`, `tet_materials`, `tet_material_indices`, `tet_boundary_groups` (returns `TetBoundaryGroupSpec`), `build_tet_heat_model`, `initialize_tet_thermal_models`, `tet_time_step_limits`, `run_tet_thermal_integration`.
- `core/solidification.py`. Two dispatch points plus behaviour-preserving extractions:
  1. `_get_solver_mode` dispatches to `resolve_tet_solver_mode` on tet meshes.
  2. `_initialize_thermal_models` dispatches to `initialize_tet_thermal_models`.
  3. Module constants `CHILL_THERMAL_DIFFUSIVITY_M2_S` and `CHILL_RHO_CP_J_M3_K` replace the local literals.
  4. Extracted methods: `_alloy_property_curves()` (the 6-point curves), `_allocate_thermal_tracking_arrays(n)`, `_initial_thermal_temperatures(n, chill_idx, chill_T)`, `_preset_chill_solid_state(chill_idx)`, and `_mold_interface_decay(h0, floor=None)`, which was the closure inside `_configure_thermal_boundaries`. The voxel path calls all five, so the voxel code has one copy of each.
- `core/solidification_loop.py`. `run_time_integration` dispatches on `mesh_type == "tet"` before reading the solver mode. `_apply_enthalpy_correction_numpy` now accepts a scalar or a per-point `beta`, and the scalar results are bit-identical. The Numba kernel is unchanged.
- `tests/test_tet_thermal.py` (new, 22 tests).

The dispatch imports are lazy (inside functions) because `core/tet_thermal.py` imports constants from `core/solidification.py`.

## Behaviour as implemented

- **Solver mode:** a tet mesh always runs `thermal_coupled`. If any other mode was requested, `engine._logger.warning` and a console line fire once per engine.
- **Model:** `node_xyz` is converted from inches to metres with `inches_to_meters`.
  - Metal tets use the `_alloy_property_curves()` curves, with `rho_cp = density * specific_heat` and `latent_heat_vol = L*(rho_l+rho_s)/2`.
  - CHILL tets use constant `rho_cp = CHILL_RHO_CP_J_M3_K` and `k = CHILL_THERMAL_DIFFUSIVITY_M2_S * rho_cp` (149.5 W/m/K), with no latent heat.
  - Any other tet code raises `ValueError`.
- **Boundary groups:** `h0` and the decay come from `engine._mold_interface_decay`, and `T_inf` is the mold temperature.
  - By owner part code: cast 500, riser_post 350, riser 220, flow_path (`is_flow_path` codes) 400, chill `chill_ambient_htc` with floor `chill_htc_floor_fraction`.
  - By contact code, overriding the owner: CORE 500, FILTER 400.
  - A face that matches no rule raises `ValueError`, so no face is ever silently adiabatic.
- **Initial state:** reuses the voxel helpers. The starting temperature is the filling field if one was set, else the pouring temperature. Chill nodes start at `chill_temperature` and are pre-solid with 0 times and feeding fs 1.
- **Loop:** same arrays, dtypes, sentinels, termination (`all states >= 0.99`), 1 % snapshots plus a final snapshot, feeding fs, peak cooling rate, progress callbacks and live thread budget as `_run_thermal_coupled_integration`. It loops on `heat_model.time < engine.max_time` and clips the last step to `max_time`.
- **Latent heat:** NumPy correction on nodes with `L_i > 0` only, with `beta_i = L_i / C_i(T_post)`.
- **Time step:** starts at `engine.time_step` and grows x1.25 per accepted step up to `tet_max_time_step_s`. A step whose max metal-node |dT| exceeds `tet_max_delta_t_c` is rejected, the temperatures and model time are restored, and the step is retried at half size (floor 1e-3 s). Both limits are read with `getattr` defaults of 2.0 and 10.0. `engine.time_step` holds the last accepted dt.
- **Gradients:** `gradient_magnitude()` converted from K/m to degC/mm by `inches_to_meters(1)/inches_to_mm(1)`. They feed `thermal_gradients` and the Niyama capture, and are computed under the same `needs_thermal_history` gate as the voxel path.
- **Niyama:** sampled on CAST nodes crossing `niyama_eval_fraction_solid`, with `quality = 2`.
- **Cancellation:** an exception raised from `progress_callback` propagates out of `run_simulation`, as on the voxel path (tested).

## Deviations from DESIGN.md

1. **"Halve when |dT| exceeds" is implemented as reject-and-retry.** The alternative, accepting the step and halving the next one, would let the step that broke the limit stand.
2. **Niyama is sampled on CAST nodes only,** identical to the voxel path. I read "quality = 2 for every metal node" as the quality value. Sampling riser and gating nodes would change `_calculate_niyama` semantics.
3. **Groups are per rule, not strictly per part code.** All gating codes share one `flow_path` group, and CORE and FILTER contact get their own groups. The physics is the same.
4. **The voxel path's adaptive CFL time step does not apply on tet meshes,** because the solver is implicit. `engine.time_step` stays 0.1 s at init.
5. **The Robin `h(t)` is evaluated at t+dt** (backward Euler, wave 1 solver). The voxel path evaluates it at t.

## Measured results (WCB, h = 0.2 in)

| case | nodes | total time (s) | test wall time |
|---|---|---|---|
| 2x2x1 box + riser cylinder (r 0.6, h 1.2) | 1330 | 134.8 (cast last 109.2, riser last 134.8) | 0.7 s |
| same, tet h 0.25 / 0.15 / 0.1 | 1055 / 4481 / 8147 | 134.4 / 135.7 / 135.9 | 0.6 / 2.4 / 4.3 s |
| voxel 0.2 / 0.1 / 0.05 | 996 / 6495 / 47229 | 184.2 / 166.8 / 157.8 | 0.4 / 1.4 / 7.1 s |
| 2x2x1 box | 935 | 93.0 | 0.8 s |
| same + 2x2x0.5 chill below | 1391 | 56.0 (casting nodes with z < 0.25: mean 50.7 s without the chill, 5.3 s with it) | 1.2 s |

- **Tet/voxel ratio:** 0.85 against voxel 0.05 (the case asserted in the test), 0.81 against voxel 0.1 and 0.73 against voxel 0.2.
  - The gap comes from the voxel geometry, not the solver. The voxel mesh holds 7.97, 6.50 and 5.90 in³ at 0.2, 0.1 and 0.05 in, against a CAD volume of 5.36 in³, which inflates V/A to 0.40, 0.33 and 0.30 against a true 0.26.
  - The tet mesh volume is 5.35 in³ and its time is mesh-converged at h = 0.25.
  - At equal nominal size (0.2 in) the 20 % bound fails. The test therefore uses voxel 0.05.
- **Step-control error:** on the box, the default limits (2 s, 10 C) give 93.0 s. Tighter limits give 96.5 s at (2, 5), 97.9 s at (0.5, 3), and a converged 99.7 s at (0.05, 0.25). **The default control finishes about 7 % early.** The likely cause is the reused enthalpy correction capping recalescence at T_sol below solidus, which discards latent heat on coarse steps. The voxel path shares that math.
- **Step rejections:** 131 of 543 attempted steps on the riser case.

## Tests run

- `venv/bin/python -m unittest tests.test_tet_thermal`: 22 tests OK in 12.3 s (13.1 s process).
- `tests.test_thermal_coupled_integration`, `tests.test_thermal_loop_regressions`, `tests.test_thermal_boundary_config`, `tests.test_initial_temperature_seam`, `tests.test_math_audit_regressions` and `tests.test_heat_transfer_physics`: 78 tests OK.
- `tests.test_architecture_boundaries`: 5 OK. `tests.test_parallel_kernel_parity`: OK (it pins the NumPy correction against the kernel).
- Every test module importing the engine or loop (25 modules): 344 tests, 1 failure (`test_shrink_scalar_below_one...`, a known baseline failure).
- Full suite: 1923 tests, 3 failures, exactly the 3 baseline failures listed in DESIGN.md, 15 skipped, 300 s.

## Open risks

- **The default step limits under-predict freeze time by about 7 %.** Tightening `tet_max_delta_t_c` to 5 gives about 3 %. An owner decision is needed: tighten the default, or add an fs-change criterion.
- **fTetWild is non-deterministic.** Tet times varied by less than 1 % across runs. Test bounds have at least 5 % margin, except the voxel ratio (0.85 against a 0.80 bound).
- **Node codes shift interface nodes.** CAST/RISER interface nodes carry the RISER code under the C1 node rule, so they are neither Niyama-sampled nor CAST-scored.
- **Unweighted node means.** Progress, the fraction-liquid snapshot trigger and the termination mean are unweighted over nodes, including chill nodes, as on the voxel path. Node density is not uniform on tets.

## For wave 3

- **Post-analyses run without exceptions but are unvalidated on tets.** With every analysis on (riser + chill assembly), nothing raised or logged a failure. None of the graph-based outputs are validated: `_compute_graph_distance_to_riser/chill` count edge hops, and feeding path, defect regions, stress, microstructure and isolation score depend on neighbour count.
- **`_get_orphan_metal_mask` may misfire at a chill.** The "smothered" rule (≥ 8 chill/core neighbours and ≥ metal neighbours) assumes 26-slot rows. A tet metal node at a chill face can meet it. It was not triggered in these geometries.
- **Length scales are not grid pitches on tets.** `voxel_sizes` is the mean incident edge length and `_thermal_voxel_size_m` is its mean, which the tet loop does not use. Scalar `base_voxel_size` length scales still need the C6 port.
- **Two engine fields change meaning.** `engine.time_step` varies through the run (last accepted dt). `thermal_gradients` is the true |grad T| in degC/mm, not the voxel RMS neighbour slope, so hot-tear and shrinkage numbers calibrated on the voxel value will shift.
- **ML and surrogate dtype.** Arrays have node length and voxel dtypes. `voxel_part_types` is int8 on tets and int32 on voxels, so ML encodings should cast.
