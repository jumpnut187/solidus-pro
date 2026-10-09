# Scout: Thermal Engine Seam (voxel → tet/P1 FEM)

Read-only reconnaissance. All line numbers refer to branch feature/tet-mesh at the time of the scout.

## 1. Entry points and call graph

```
run_pipeline (core/simulation_pipeline.py:660)
├─ get_alloy(name)                          core/materials.py:828
├─ prepare_mesh_and_geometry                core/simulation_pipeline.py:555-657
│   └─ AdaptiveVoxelMesh(base_voxel_size, max_refinement_level, ...) :605-613
│       .generate(...) :623 ; .assign_part_types(parts_with_types) :643
├─ create_analysis_options                  core/simulation_pipeline.py:160-200
├─ CASTSIM_DEFAULT_SOLVER_MODE override     :45-55, applied :687-689
├─ maybe_run_filling(voxel_mesh, ...)       :716 (core/filling_stage.py)
├─ SolidificationEngine(voxel_mesh, alloy, opts) :726 ; engine.initialize() :727
├─ engine.set_initial_temperature_field(filling_result.temperature_c, fill_arrival_times=fill_time_s) :781-784
│   (only when the fill is trustworthy. A diverged fill is discarded at :769-778)
├─ engine.run_simulation(...)               core/solidification.py:936-1042
│   ├─ _calculate_feeding_distances()       :980
│   ├─ [thermal_coupled] _initialize_thermal_models :988 → :411-572
│   │     HeatTransferModel(...) :470 ; ThermalSolidificationModel(...) :480
│   │     _configure_thermal_boundaries :574-692
│   ├─ [schedule_legacy] chill/riser influence + _initialize_solidification_schedule :991-1013
│   ├─ run_time_integration                 core/solidification_loop.py:1122-1160
│   │     → _run_thermal_coupled_integration :876-1119
│   │     → _run_schedule_based_integration :714 | _run_front_based_integration :458
│   └─ run_post_analyses(self)              :1026
├─ engine.export_results()                  core/solidification.py:5881-6021
└─ build_visualization_payload              core/simulation_pipeline.py:203-438
```

The surrogate path is `run_surrogate_pipeline` (core/surrogate/pipeline.py:146). It uses the same `prepare_mesh_and_geometry`. It calls `engine.initialize()` (:211) and `set_initial_temperature_field` (:215). Then it runs `_calculate_feeding_distances`, `compute_initial_chill_influence` and `_calculate_riser_influence` (:221-226). It projects the model prediction back onto the voxels with `unproject_grid(prediction, centers, meta)` and writes start and local times through `_apply_schedule_to_engine` (:74-133). There is no thermal solve.

### Solver modes
- The constants are `SOLVER_MODE_SCHEDULE_LEGACY="schedule_legacy"` and `SOLVER_MODE_THERMAL_COUPLED="thermal_coupled"` (core/solidification.py:139-144). The loop holds a duplicate literal: `THERMAL_COUPLED_SOLVER_MODE` (core/solidification_loop.py:41).
- The default is `AnalysisOptions.solver_mode="thermal_coupled"` (solidification.py:133).
- The mode is selected in three places:
  1. The UI dict key `solver_mode` (simulation_pipeline.py:192-195).
  2. The env var `CASTSIM_DEFAULT_SOLVER_MODE` (:45-55, :687-689).
  3. `_get_solver_mode` normalizes invalid values to thermal_coupled (solidification.py:394-399).
- Dispatch:
  - `run_time_integration` (solidification_loop.py:1135-1160) chooses thermal_coupled first.
  - Otherwise it uses the schedule path if the schedule arrays exist, and the front-based path if they do not.
  - The surrogate pipeline is a third, separate route.
- Kernel backends: Numba parallel or the NumPy fallback (heat_transfer.py:276-286, 864-869). The enthalpy kernel falls back via `ENTHALPY_KERNEL_AVAILABLE` (solidification_loop.py:25, 951-957).

## 2. Exact data consumed and produced

### Mesh attributes the engine reads
Counts below are from grep. N is the voxel count.

| attr | shape / dtype | units | uses |
|---|---|---|---|
| `voxel_centers` | (N,3) float | inches | 68× in solidification.py; Niyama LSQ in loop.py:173 |
| `voxel_neighbors` | (N,26) int32, -1 = none | slot-ordered | heat model :472, gradient :1032, Niyama LSQ :175 |
| `voxel_sizes` | (N,) float | inches | per-cell dx, solidification.py:419-425 |
| `base_voxel_size` | scalar | inches | fallback, :405-407 |
| `voxel_part_types` | (N,) int32 | codes (CAST=0) | BC partition :618-631; Niyama cast mask loop.py:420 |
| `voxel_states` | (N,) float32 | fraction solid 0..1 | written every step, loop.py:964 |
| `voxel_moduli` | (N,) | inches | Chvorinov max_time :716 |
| `voxel_volumes`, `surface_voxels` | (N,) | in³, bool | post-analyses |
| `get_surface_voxel_indices()` | int idx | – | BC sets :612, run_simulation :960 |
| `get_voxel_count()` | int | – | :829, :801 |

### Heat model construction (solidification.py:411-572)
- `voxel_size_m` is the scalar mean of `voxel_sizes`, converted from inches to metres (:412-415). It is stored as `engine._thermal_voxel_size_m`.
- `voxel_sizes_m` is an (N,) float64 array in metres (:419-425).
- Chill overrides:
  - `diffusivity_overrides=(idx int64, alpha=1.17e-4 m²/s)`.
  - `rho_cp_overrides=(idx, 1800*710 J/m³K)`.
  - Both are set at :432-446.
- Property curves (:450-466) use 6 points: `[20, 0.5Ts, Ts, (Ts+Tl)/2, Tl, Tl+100]` °C.
  - k = k_s·[1,.95,.85,.77,.70,.665]
  - cp = cp·[.85,.92,1,1.05,1.10,1.15]
  - ρ: ρs, .98ρs, .96ρs, avg, ρl, .98ρl
  - `use_temperature_dependent=True` is always set (:475).
- The initial T is either `_initial_temperature_field` (fill) or `alloy.pouring_temp`. Chills are then overwritten with `chill_temperature` (:503-516).
- Chills start pre-solid (:523-538):
  - fs=1
  - start, end and local times = 0.
  - `feeding_fraction_solid` = 1.
- Adaptive dt (:556-572):
  - It is controlled by env `CASTSIM_THERMAL_ADAPTIVE_DT` (default on).
  - `dt_cfl` is clipped to [0.05, 2.0].
  - `engine.time_step` (default 0.1 s, :283) is replaced only when dt_cfl ≥ 2×time_step.

### HeatTransferModel API (physics/heat_transfer.py)
- The constructor is `__init__(n_voxels, voxel_size, neighbors, alloy, use_temperature_dependent, voxel_sizes, diffusivity_overrides, property_curves, rho_cp_overrides)` (:179-269).
- Public methods and properties:
  - `temperatures` (N,) float64 °C, mutable in place (:294-296)
  - `set_initial_temperature` (:303)
  - `add_boundary` (:315)
  - `clear_boundaries` (:348; used by filling)
  - `set_heat_source` W/m³ (:372)
  - `get_stable_timestep` (:380)
  - `step(dt)` (:525-563)
  - `get_cooling_rate(prev, dt)` °C/s (:857-859)
  - `get_gradient_magnitude(neighbors, voxel_size)` K/m (:500-523)
  - `time` (s)
  - `boundaries`
  - `backend_description`
- Diffusion is explicit Euler with CFL subdivision. The bound is `dt < 0.9·dx_min²/(4α_max)` (:380-398). α_max is doubled when rho_cp overrides exist (:426-432).
- Heterogeneous update (`k_ij` harmonic mean, two-point flux): :685-700 (NumPy) and `update_heterogeneous_parallel` (Numba).
- BC types (`BoundaryType` :44-52): FIXED_TEMPERATURE, CONVECTION, INTERFACE, HEAT_FLUX, ADIABATIC.
- Robin formula (:720-767). It is explicit and evaluated against the pre-step T:
  `T_new[i] -= dt·h(t)·(T_old[i]−T_mold)/(ρc_i·dx_i)` (INTERFACE, :754-766).
  - `h(t) = bc.htc·time_function(t)`.
  - The ρc used here is the scalar `alloy.density_liquid*specific_heat` (:229, :730), not the curve value. Chill cells use the override.
- Fallback α(T) without curves is a linear blend with α_liq = 0.7α (:795-835).

### Mold BC (solidification.py:574-692)
- `h_eff(t) = 1/(1/h0 + 1/h_pen(t))`, where `h_pen = k_mold/sqrt(π·α_mold·t)`.
- The time function returns `max(h_eff/h0, floor)` (:596-614).
- Default parameters (core/solidification_params.py:138-164):
  - k_mold = 0.6 W/mK
  - α_mold = 3.3e-7 m²/s
  - floor = `air_gap.htc_floor_fraction` = 0.35 (:126)
- Surface sets by part type, each with its h0 (W/m²K):

| part type | h0 |
|---|---|
| CAST | htc_cast_interface = 500 |
| RISER_POST | 350 |
| RISER | 220 |
| flow path (`is_flow_path`) | htc_gating = 400 |
| CORE | 500 |

- Every set uses `T_inf = mold_temperature` (21 °C).
- Chill surface = chill ∩ surface (:680-692). It uses h0 = `chill_ambient_htc` 500 and floor 0.7.
- The mold is a BC, not a meshed domain. Sand is not discretized.

### Coupled loop (core/solidification_loop.py:876-1119)
- `max_iterations = int(max_time/time_step)` (:889).
- `max_time` comes from Chvorinov: `B·M²·safety`, clipped to [600, 14400] s for thermal_coupled (solidification.py:694-766; TimingParams :194-202).
- Each step:
  1. Termination check: `all(voxel_states ≥ 0.99)` (:921-923) or the iteration cap. All voxels count, including risers, gating and cores.
  2. `heat_model.step(dt)` performs pure diffusion plus BC (:926).
  3. Enthalpy correction (:943-958), using lever fs(T) = (T_liq−T)/dT:
     - β = L_vol/ρc with ρc = ρ_l·cp (scalar) and L_vol = L·(ρl+ρs)/2.
     - Mushy zone: `T=(T_post+β(T_liq/dT−fs_prev))/(1+β/dT)`.
     - Remelt branch: keep T_post.
     - All-solid branch: `T_post+β(1−fs_prev)`.
     - Below solidus: `min(T_post+β·remaining, T_sol)` (:814-873).
  4. `solidification_model.update(T_final, dt)` keeps fs monotone (physics/solidification_physics.py:266-326). Thresholds:
     - start fs ≥ 0.01
     - end fs ≥ 0.99
     - a single-step crossing anchors start at t−dt
  5. Engine arrays written (float32 (N,)). Times are s and rates °C/s.
     - `voxel_states`
     - `solidification_start_times` (0 where unset)
     - `solidification_end_times` (NaN unfinished)
     - `local_solidification_time`
     - `solidification_times = end` (:963-990)
  6. `feeding_fraction_solid` is recorded when fs crosses `alloy.critical_fraction_solid` (:992-1009).
  7. Running peak `cooling_rates` (:1011-1016).
  8. Niyama capture (:1018-1026 → :405-455). Runs only when niyama, hot_tear, shrinkage or feeding_path is on (:906-911). It applies when a CAST voxel crosses `niyama_eval_fraction_solid`.
     - Uses linear interpolation in the step.
     - Uses the LSQ gradient in °C/mm from neighbour centres (:133-186).
     - Ny = G/√Ṫ.
     - Stored in `_thermal_niyama_{gradient,cooling_rate,temperature,time,values}_at_eval`, `_quality` and `_sampled` (allocated at solidification.py:484-491).
     - Consumed at solidification.py:4141-4151.
  9. `thermal_gradients = get_gradient_magnitude(neighbors, scalar voxel_size_m)/1000` gives °C/mm (:1028-1043).
  10. Fraction-liquid history: an fs snapshot is appended per 1% of mean progress (:1045-1049). The final snapshot is appended at :1064-1066.
- After the loop:
  - Unfinished voxels get NaN end times (:1071-1073).
  - Unresolved feeding_fs is filled with the current fs (:1075-1077).

Post-analyses read these engine fields:
- `_thermal_latest_temperatures` (solidification.py:2641)
- `thermal_gradients` (:2650, 4151)
- the Niyama arrays
- the time arrays

### Results the pipeline reads
- simulation_pipeline.py reads these engine attributes:
  - `shrinkage_probability`, `get_shrinkage_display_intensity`
  - `hotspot_intensity`
  - `niyama_values`, `niyama_risk`, `combined_feed_risk`
  - `gas_*`
  - `solidification_times`
  - `fraction_liquid_history`, `fraction_liquid_times`
  - `_fill_frames`, `_filling_summary`
- export_results (solidification.py:5881-6021) emits about 50 per-voxel arrays keyed by voxel index.

### Alloy API (core/materials.py:18-230)
- Fields read by the thermal path: `liquidus_temp`, `solidus_temp`, `pouring_temp`, `density_liquid`, `density_solid`, `thermal_conductivity`, `specific_heat`, `latent_heat`, `critical_fraction_solid`.
- Scheil fields: `melting_point_solvent`, `partition_coefficient`, `eutectic_*`.
- Property: `thermal_diffusivity = k/(ρ_l·cp)` (:168-171).

### Filling handover
- `FillingResult.temperature_c` and `.fill_time_s` are (N_voxels,) on occupied voxels (filling_stage.py:2083-2110, 2183-2184).
- `fill_time_s` is −1 where metal never arrived and NaN for non-metal.
- The setter validates shape (N,) and finite values (solidification.py:768-825).
- Arrival times are stored but not consumed by the solver (:792-793). They are used only in export (:5975-5988).

### core/stepping.py
- `should_capture_fraction_history` (:19-37). It is the only thing the thermal loop uses from this module.
- `SnapshotSchedule(frames)` (:40-77). The thermal loop does not use it and keeps its own 1% interval (loop.py:893).

## 3. Voxel-specific assumptions
1. **26-slot neighbour semantics.** Slots 0-5 are faces (+X,−X,+Y,−Y,+Z,−Z) with weight 0.5. Slots 6-17 are edges with weight 0. Slots 18-25 are corners with weight 0.125. Sources: heat_transfer.py:621-680 (NumPy), :931-982 (`compute_laplacian_parallel`), and the heterogeneous kernels. The ordering is set by `core/voxel_mesh.py._canonicalize_neighbor_order` (cited in heat_transfer.py:623, 950).
2. **Lumped cell = cube.**
   - The Laplacian is scaled by `1/dx_i²` (heat_transfer.py:242-247, 682).
   - The CFL bound uses `dx²/(4α)` (:380-398).
3. **Boundary area/volume = 1/dx_i.** Robin, convection and flux terms divide by `ρc·dx_i` (heat_transfer.py:745-788). This assumes one exposed face per surface voxel, so a corner voxel with three exposed faces is under-cooled. FEM needs `A_i/(ρc·V_i)` or a proper boundary mass and stiffness term.
4. **Scalar representative voxel size.**
   - `_thermal_voxel_size_m` is the mean size (solidification.py:401-415). It is used for gradients (loop.py:1028-1040).
   - `get_gradient_magnitude` treats every neighbour as distance dx, including diagonals at √2·dx and √3·dx (heat_transfer.py:500-523). The result is an RMS over neighbours, not a vector gradient.
5. **Neighbour graph for LSQ Niyama gradient** (loop.py:133-186). Rank and quality: 2 when there are ≥4 neighbours and full rank. Rank is satisfied by the 26-neighbourhood. On tets a one-ring of nodes works, but the quality semantics change.
6. **Neighbour graph distances to riser and chill** (solidification.py:1044-1095, `_compute_graph_distance_to_*`). These are graph BFS over `voxel_neighbors`. Also used in feeding_path.py, defect_regions.py, stress_analysis.py and microstructure.py (grep hits).
7. **Dense ijk lattice in filling.** `build_grid_mapping` requires a uniform lattice. It raises an error otherwise (filling_stage.py:588-646). `gather_to_voxels` and `scatter_to_grid` map back to N voxels (:672-705). The surrogate's `unproject_grid` uses the same pattern.
8. **Chvorinov `voxel_moduli`** gives a per-voxel modulus in inches and sets `max_time` (solidification.py:716).
9. **Units.** Geometry is in inches (centres and sizes), the solver works in metres, and gradients are reported in °C/mm (`/1000` at loop.py:1042).
10. **Per-cell DOF = per-voxel output.** Every result array is (N_voxels,) and is visualized as cubes (`centers` and `sizes` in the payload, simulation_pipeline.py:237-245).

## 4. What a tet/P1 replacement must provide
1. **Node arrays.** The DOF is the node (P1).
   - `node_xyz` (Nn,3) float64. Pick one unit (inches) for consistency with centres, or convert explicitly.
   - Lumped mass `M_i = Σ ρc·V_tet/4`.
   - Sparse stiffness `K` (CSR, Nn×Nn) built from per-tet k.
   - Per-node material id (cast, riser, chill, core, gating) replacing `voxel_part_types` for BCs. Interface nodes are ambiguous.
2. **Boundary data.**
   - Boundary triangles (Nf,3) int with a part-type tag.
   - The area-weighted boundary mass `A_i` per node, or the full Robin matrix `∫h·N_iN_j`.
   - Reuse `_mold_interface_decay` (solidification.py:596-614) for the time-dependent h.
3. **Material heterogeneity.**
   - Replace the chill α and ρc override tuples with per-tet k and ρc. Harmonic averaging becomes unnecessary because element-wise assembly is conservative.
   - The k(T), ρ(T) and cp(T) curves need either Picard reassembly or a lagged k per step.
4. **Implicit step.**
   - Solve `(M + dt(K + R(t))) T^{n+1} = M T^n + dt·R(t)·T_mold`.
   - This removes the CFL subdivision and the adaptive-dt heuristic (solidification.py:549-572). Keep `engine.time_step` semantics or a new dt policy. `max_iterations` derives from it.
5. **Enthalpy correction.** It is pointwise per DOF, so it ports unchanged to nodes if the mass is lumped (loop.py:814-873). With a consistent mass it is not exact.
6. **Same interface for the loop.** The loop needs:
   - `temperatures` (mutable, assignable via `[:]`)
   - `step(dt)`
   - `time`
   - `get_cooling_rate`
   - a gradient provider
   On P1 elements, compute the per-tet constant ∇T and average it to nodes. This replaces both `get_gradient_magnitude` and the LSQ Niyama gradient.
7. **Output mapping.** Every engine array is (N,) per voxel today. Decide whether N becomes Nn (nodes) or the tet count. Then provide node→cell or node→display mappings for the pipeline payload (`centers`, `sizes`) and for every post-analysis that uses `voxel_neighbors`, `voxel_volumes` or `voxel_moduli`.
8. **Filling handover.** `set_initial_temperature_field` must accept an array shaped like the new DOF count. The filling LBM stays on a dense lattice, so it needs lattice→node interpolation in place of `gather_to_voxels`.
9. **Chvorinov `max_time`** needs a modulus per node or global. One option is V/A from the surface mesh.

## 5. Risks and pinning tests

### Risks
- **SCHEIL is not active in thermal_coupled.**
  - The engine constructs `ThermalSolidificationModel(n_voxels, alloy)` with the default `LEVER_RULE` (solidification.py:480-483; default at solidification_physics.py:178).
  - The enthalpy correction hard-codes linear fs (loop.py:944-948).
  - Scheil exists only as an available method (solidification_physics.py:381-430).
  - `test_scheil_regression.py` pins the model, not the loop.
  - Switching to Scheil requires a nonlinear fs(T) inversion in the correction.
- **ρc inconsistency.**
  - The curves feed only α(T). The BC and source terms use the scalar ρ_l·cp (heat_transfer.py:229).
  - The enthalpy β uses the scalar ρ_l·cp (loop.py:938).
  - An FEM energy-consistent solver will shift freeze times. Re-baseline the benchmarks.
- **Termination counts non-cast parts.** A non-solidifying or adiabatic region stalls the run to the cap.
  - Previously flow-path surfaces got no BC (comment at solidification.py:623-627).
  - On a tet mesh, make sure every part has boundary faces tagged.
- **The gradient magnitude is not a true gradient** (heat_transfer.py:500-523). Hot-tear and shrinkage numbers calibrated on it will move.
- **Explicit Robin** against T_old plus a floor of 0.35 per surface set. An implicit treatment changes early-time cooling.
- Niyama `quality` int8 semantics (0/1/2) are tied to neighbour count.
- **Time conventions.** The loop rewrites `engine.time_step` (adaptive). Chill pre-solid uses 0.0 times and unfinished uses NaN. Downstream relies on these sentinels (loop.py:976-979, 1068-1073).
- The surrogate path is grid-projected (surrogate/pipeline.py:94-99) and independent of the thermal solver. It still requires a voxel mesh unless reworked.

### Tests that pin behaviour (tests/)
- **Heat model:**
  - test_heat_transfer_physics.py
  - test_heat_transfer_numba_backend.py
  - test_parallel_kernel_parity.py
  - test_heat_flux_boundary.py
  - test_chill_interface_safety.py
  - test_chill_exclusion.py
- **Boundaries:**
  - test_thermal_boundary_config.py (part-type sets, decay)
  - verify_chill_decay.py
- **Loop:**
  - test_thermal_coupled_integration.py
  - test_thermal_loop_regressions.py
  - test_math_audit_regressions.py
  - verify_thermal_benchmarks.py (benchmark script, not a pytest)
- **fs model:** test_scheil_regression.py
- **History:**
  - test_fraction_liquid_final_snapshot.py
  - test_fraction_liquid_timing_consistency.py
  - test_fraction_liquid_state_label.py
  - test_stepping.py
- **Handover:**
  - test_initial_temperature_seam.py
  - test_filling_thermal.py
  - test_filling_stage.py
  - test_filling_notice.py
- **Pipeline and options:**
  - test_simulation_pipeline_options.py (ANALYSIS_LABELS coverage, solidification.py:162-185)
  - test_solidification_params.py
- **Surrogate:**
  - test_surrogate.py
  - test_surrogate_split.py
  - test_surrogate_geometries.py
  - test_surrogate_chill_channel.py

### Ambiguities
- Whether downstream post-analyses (feeding_path, defect_regions, stress, microstructure) are in scope. Their use of `voxel_neighbors` was found by grep and not read.
- The exact contents of `update_heterogeneous_parallel` and `compute_gradient_magnitude_parallel` were not read line by line. They are assumed to mirror the NumPy references, per the parity tests.
