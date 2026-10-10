# Fix lane: tet physics and speed (review_physics.md findings 1-4, tet run time)

Files: `core/tet_thermal.py`, `physics/tet_heat_transfer.py`, `tests/test_tet_thermal.py`,
`tests/test_tet_heat_transfer.py`, new `tools/benchmark_tet_vs_voxel.py`. `requirements.txt` unchanged (no new dependency). Nothing staged.

## Findings

| review id | change | test |
|---|---|---|
| 1 chill-interface preheat | `blend_chill_interface_temperatures`: a node incident to both metal and chill tets starts at `(C_m T_init + C_c T_chill)/(C_m + C_c)`, with `C_m` at rho_cp(T_init). Called in `initialize_tet_thermal_models`. | `BlendChillInterfaceTemperaturesTest` (exact value, identity without chill), `TetChillInterfaceConvergenceTest`, updated `test_chill_nodes_start_presolid_at_chill_temperature` |
| 2 Niyama G averaged with chill | `nodal_gradient(tet_mask=None, nodes=None)` and `gradient_magnitude(...)`. A node with no masked tet falls back to all tets. Niyama capture and the final `thermal_gradients` pass the metal-tet mask. | `test_masked_gradient_at_material_interface_is_one_sided` (flat interface, k 25/150: interface G = metal G to 1e-9; unmasked < 0.8 G_m), `test_cast_face_gradients_use_metal_tets_only` (every engine call uses the metal mask) |
| 3 grow after reject | Retry at `dt * clip(0.9 * limit/|dT|, 0.2, 0.5)`. No growth on the first accepted step after a rejection. Other accepted steps grow by `min(1.25, max(1, 0.9 * limit/|dT|))`. | `TetStepControllerTest` (ratio in [0.2, 0.5], no growth after retry, growth <= 1.25, rejection rate < 20 %) |
| 4 docstring | Module docstring says exact conservation holds only for constant properties and cites the 0.55 % drift. | n/a |

### Measurements (same pickled meshes before/after; HEAD 9514111 = before)

- Finding 1, chill case (2x2x1 cast on 2x2x0.5 chill), mean CAST end time at fixed probes 0.05-0.2 in above the chill:
  - before: 8.07 s at h 0.2, 4.80 s at h 0.1. Gap 3.27 s.
  - after: 5.00 s at h 0.2, 3.92 s at h 0.1. Gap 1.09 s. At h 0.07 the value is about 3.6 s.
  - Probe at z = 0.1 in, error against h 0.07: before 4.7 s at h 0.2 and 0.53 s at h 0.1. After: 2.07 s and 0.28 s.
  - Chill-case total time: before 61.6 / 61.4 / 60.3 / 59.9 s at h 0.4 / 0.2 / 0.1 / 0.07. After: 58.2 / 59.2 / 59.2 / 59.2 s.
- Finding 3, rejected / attempted solves:
  - riser: 79 / 336 → 48 / 324.
  - box: 48 / 208 → 26-29 / 192-200.
  - thick_thin at 0.1 in: 150 / 619 → 74 / 556.
  - Most of the remaining rejections come from nodes that have just reached fs = 1 at the solidus and drop more than 15 C in the next step. The controller cannot predict that drop.
- Results against the 0.5 % budget: box 101.374 → 101.277 s (-0.10 %). Riser 145.073 → 144.699 s (-0.26 %). thick_thin 191.46 → 191.43 s.

## Speed (thick_thin 0.1 in, desktop analyses, run_pipeline, cProfile, warm process)

| stage | before s | after s |
|---|---|---|
| run_pipeline wall | 21.0 | 10.6 (9.3-11.0 unprofiled) |
| fTetWild meshing (build_tet_mesh) | 3.7 | 3.7 (not in this lane) |
| solidification loop | 17.0 | 6.4 |
| heat step: CG solve | 5.6 (scipy cg) | 5.0 (numba PCG) |
| heat step: assembly (properties + gathers) | 7.0 | 0.7 |
| node_heat_capacity for beta | 1.9 | 0.25 |
| nodal_gradient | 3.5 (469 full calls) | 0.07 (crossing nodes only) |
| enthalpy inversion / SolidificationModel.update | 0.08 / 0.08 | 0.08 / 0.07 |
| post-analyses + snapshots (run_simulation minus loop) | 0.3 | 0.6 |

Changes:
- Stiffness assembly and lumped capacity are precomputed gather matrices (`K.data = S @ k_e`, `C = Q @ rho_cp_e`). They run as parallel numba matvecs. The element properties are one parallel numba kernel.
- CG is a numba Jacobi PCG. It is serial below 100k nodes and parallel from there (`PARALLEL_CG_MIN_NODES`). Measured serial vs parallel: 25 vs 33 ms at 35k nodes, 315 vs 123 ms at 215k nodes. The initial guess is `T_old + (dt/dt_prev) * last change`, which cuts iterations 15-25 %. Without numba the scipy path still runs. `test_numba_and_scipy_paths_agree` checks the two match to 1e-9.
- Gradients are computed only for the CAST nodes that cross the Niyama fs on that step, plus once at the end. Fewer rejected solves (finding 3) also saves time.

Tried and not adopted:
- pyamg smoothed aggregation (installed in /tmp only). At 35k nodes, Jacobi PCG took 7.6 / 20.5 ms against 20.9 / 37.1 ms for SA-PCG, plus 41 ms setup. At 215k nodes it took 19 / 58 ms against 156 / 255 ms, plus 250 ms setup.
- Factorization reuse. splu at 35k nodes takes 2.6 s to factor and 12.5 ms per solve, which is slower than the PCG.
- Robin diagonal cache. Its cost does not show in the profile.

## Benchmark (`tools/benchmark_tet_vs_voxel.py --factors 0.7 1.0 1.5 2.0 2.5 3.0`, base 0.1 in)

Reference: tet with 0.05 in edges. thick_thin has 281k nodes, total 191.34 s. l_bracket has 114k nodes, total 72.93 s. "field err" is the mean |dt| against the reference at reference nodes, as a % of the reference total.

| part | mesh | size in | points | wall s | total s | vs ref | field err |
|---|---|---|---|---|---|---|---|
| thick_thin | voxel | 0.1 | 23811 | 2.5 | 194.39 | +1.6 % | 4.0 % |
| thick_thin | voxel | 0.05 | 179021 | 19.1 | 190.20 | -0.6 % | 0.8 % |
| thick_thin | tet | 0.1 (x1) | 34480 | 10.7 | 191.41 | +0.0 % | 0.2 % |
| thick_thin | tet | 0.15 (x1.5) | 8440 | 2.6 | 192.24 | +0.5 % | 0.4 % |
| thick_thin | tet | 0.2 (x2) | 4579 | 1.4 | 192.01 | +0.4 % | 0.6 % |
| thick_thin | tet | 0.25 (x2.5) | 3335 | 1.2 | 191.68 | +0.2 % | 0.7 % |
| l_bracket | voxel | 0.1 | 8946 | 1.0 | 82.67 | +13.4 % | 14.1 % |
| l_bracket | voxel | 0.05 | 63591 | 5.8 | 76.00 | +4.2 % | 4.4 % |
| l_bracket | tet | 0.1 (x1) | 13473 | 4.1 | 73.18 | +0.3 % | 0.3 % |
| l_bracket | tet | 0.15 (x1.5) | 4078 | 1.2 | 73.13 | +0.3 % | 0.5 % |
| l_bracket | tet | 0.2 (x2) | 2098 | 0.8 | 73.40 | +0.6 % | 0.7 % |
| l_bracket | tet | 0.25 (x2.5) | 1207 | 0.6 | 73.46 | +0.7 % | 1.1 % |

Tet at 0.07 in is within 0.1 % of the reference on both parts, so the reference is h-converged. Factor 3.0 also passes on total time (+0.8 %), but its field error is 1.4 %.

**Recommendation: `tet_edge_length_factor = 2.0`.** It is the fastest factor in the requested set (1.0 / 1.5 / 2.0), with the total within 0.6 % and the field error at or below 0.7 %. At that factor tet runs faster than voxel at 0.1 in and are more accurate. The script's rule (least summed wall time) picks 2.5 when 2.5 is in the set, but that gains only 0.2 s on these parts while the l_bracket field error rises from 0.7 % to 1.1 % and thin walls are left about 2 elements thick. I did not edit config/settings.py.

## Deviations
- DESIGN C2 specifies `scipy.sparse.linalg.cg` with `x0 = T_old`. The solver is now a numba PCG with an extrapolated x0, and scipy cg remains as the fallback. CG_RTOL stays at 1e-10. Both are implementation changes and the API is unchanged.
- `nodal_gradient` / `gradient_magnitude` gained `nodes=` as well as `tet_mask=`. The predictive growth cap goes beyond the review's suggested fix for finding 3.
- During my runs other workers were editing simulation_pipeline, tet_mesh and solidification, so the benchmark wall times include their in-progress code.

## Tests
- `venv/bin/python -m unittest tests.test_tet_thermal tests.test_tet_heat_transfer tests.test_tet_end_to_end tests.test_tet_post_analyses tests.test_thermal_coupled_integration tests.test_architecture_boundaries`: 103 run, OK.
  - tet_thermal has 41 tests (7 new). tet_heat_transfer has 15 (2 new).
- The new chill-convergence and step-controller tests passed 3 extra times, each on a fresh fTetWild mesh.

## Open risks
- CG is now about 47 % of the 0.1 in run. Relaxing CG_RTOL to 1e-8 cuts iterations by about 30 % and moves T by at most 2e-3 K, but it breaks 3 exactness asserts in test_tet_heat_transfer (1e-9 energy, residual and bound). That needs a decision.
- The parallel PCG build is not cached (numba's cache key ignores `parallel`). The first run with 100k or more nodes in each process pays 0.6 s to compile it.
- An interface node starts below the solidus with fs = 0. On the first step the enthalpy inversion moves it back into the mushy zone, so chill runs still reject at startup.
- The timings and the factor recommendation come from 2 parts on 1 machine (M5 Pro). fTetWild totals vary about 0.3 % between runs.
