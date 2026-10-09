# Wave 3: exact tet enthalpy update and step-control defaults

The tet latent-heat update now conserves enthalpy exactly. The default `tet_max_delta_t_c` moves from 10 to 15 C, with `tet_max_time_step_s` staying at 2 s. Total solidification time is within 1.4 % of the dt-converged value on both study cases. Nothing is committed or staged.

## Files changed

- `core/tet_thermal.py`
  - New `invert_tet_enthalpy(T_post, fs_prev, beta, T_liq, T_sol) -> (T_new, fs_new)`.
  - `run_tet_thermal_integration` calls it in place of `_apply_enthalpy_correction_numpy`.
  - `DEFAULT_TET_MAX_DELTA_T_C` changes from 10.0 to 15.0.
- `config/settings.py`: `tet_max_delta_t_c` 10.0 -> 15.0 in four places: `_TET_POSITIVE_DEFAULTS`, the `MeshSettings` default, the YAML template and the TOML template.
- `tests/test_tet_thermal.py`: 12 new tests and one replaced test (details under Tests).
- `tests/test_tet_pipeline.py`: one literal changed (`tet_max_delta_t_c` 10.0 -> 15.0). The supervisor approved this edit.

`core/solidification_loop.py`, the voxel kernel and the voxel path are untouched.

## Behaviour as implemented

1. **Inversion.** With `beta_i = L_i / C_i(T_post)` and `h = T + beta (1 - fs)`, the code solves `T_post + beta (1 - fs_prev) = T_new + beta (1 - fs_new)` with `fs_new = max(fs_prev, lever(T_new))`. It is closed form per segment and vectorized. With `T_p = T_liq - fs_prev dT`:
   - `T_post >= T_p`: T_new = T_post, and fs is unchanged.
   - `h < T_sol`: T_new = h, fs = 1.
   - Otherwise (mushy): T_new = (T_post dT + beta T_p) / (dT + beta).
2. **Nodes with L_i = 0** (chill) skip the inversion, as before.
3. **fs consistency.** `SolidificationModel.update(T_final)` still recomputes fs from T with the lever rule, which is the engine's method. `invert_tet_enthalpy` computes `fs_new` with that same formula from `T_new`, so the model and `voxel_states` get bit-identical fs. A test asserts array equality against `SolidificationModel.update`, which runs the Numba commit kernel. The returned `fs_new` is therefore not written to the model.
4. **Behaviour change against wave 2.** A node whose post-diffusion T falls below solidus while liquid remains now recalesces into the mushy zone and keeps the enthalpy. Wave 2 capped it at T_sol and discarded the rest. The old comment warned that this could stall runs. No stall was seen in any run below.
5. **Conservation, measured.** In the adiabatic box (constant properties, 20 s steps, top half 1600 C and bottom half 100 C, full solidification), the relative enthalpy drift is 2.9e-14. With the wave 2 correction swapped in, the same test gives -4.6 %.

## dt convergence study

- **Setup:** WCB, tet edge 0.2 in, default AnalysisOptions, `tet_time_step_limits` mocked.
- **Meshes:** box 952 nodes, box + riser 1331 nodes, from one meshing per case.
- **Columns:** "solves" counts `heat_model.step` calls, rejected steps included. Wall time is `run_simulation` including post-analyses. Error is measured against (0.05, 0.25).

| (max_dt s, max_dT C) | box total s | box err | box solves | box wall s | riser total s | riser err | riser solves | riser wall s |
|---|---|---|---|---|---|---|---|---|
| (2, 10) old default | 101.01 | +1.15 % | 325 | 0.35 | 144.66 | +0.84 % | 492 | 0.58 |
| (2, 5) | 100.40 | +0.54 % | 649 | 0.57 | 144.04 | +0.40 % | 995 | 1.04 |
| (1, 5) | 100.40 | +0.54 % | 649 | 0.58 | 144.27 | +0.56 % | 967 | 1.02 |
| (0.5, 3) | 100.18 | +0.32 % | 1088 | 0.90 | 143.88 | +0.29 % | 1609 | 1.57 |
| (0.05, 0.25) converged | 99.86 | 0 | 12752 | 8.46 | 143.46 | 0 | 19286 | 17.43 |
| (2, 12) | 101.11 | +1.25 % | 269 | 0.25 | 144.65 | +0.83 % | 414 | 0.55 |
| **(2, 15) new default** | 101.26 | +1.40 % | 218 | 0.20 | 144.92 | +1.02 % | 328 | 0.46 |
| (4, 15) | 101.40 | +1.54 % | 213 | 0.20 | 144.81 | +0.94 % | 335 | 0.46 |
| (2, 20) | 101.73 | +1.87 % | 164 | 0.16 | 145.44 | +1.38 % | 239 | 0.36 |
| (4, 40) | 103.43 | +3.58 % | 76 | 0.09 | 147.08 | +2.52 % | 117 | 0.22 |

- **Convergence check:** (0.02, 0.1) gave box 99.85 s and riser 143.36 s, so (0.05, 0.25) is converged to 0.07 %.
- **Edge 0.15 in** (2800 and 4660 nodes):
  - (2, 15): +1.24 % on the box and +1.01 % on the riser case.
  - (2, 20): +1.59 % and +1.29 %.
  - (2, 10): +0.76 % and +0.67 %.
- **Repeat meshings at edge 0.2:**
  - (2, 10) box error ranged 1.0-1.15 %.
  - (2, 20) box error ranged 1.76-2.0 %, which is too close to the 2 % bound.
- **Choice: (2, 15).** It is the cheapest setting with a safe margin under 2 %: about 33 % fewer solves than (2, 10).
- **Why max_dt stays 2.** max_dt barely binds on these small cases: (2, 15) and (4, 15) agree within 0.15 %. With no data on a case where max_dt binds, I did not raise it.
- **Coarse steps now finish late, not early.** The remaining error is backward-Euler lag. In wave 2, (2, 10) finished 7 % early on the box (93.0 s). It now finishes 1.2 % late (101.0 s). Riser-case total time at the defaults rose from 134.8 s to 144.9 s.

## Deviations

1. **No `tet_max_delta_fs` criterion.** The study capped the max per-step dfs at 0.05, 0.1 and 0.2:
   - At (4, 20) and (4, 40) with a cap of 0.1 or 0.2, accuracy did not change.
   - (4, 1000) with a 0.05 cap gave +1.7 % / +2.0 % at 159 / 230 solves. This is no better than (2, 20) and worse than (2, 15).
   - The |dT| limit already bounds dfs in the mushy zone. No setting was added.
2. **One edit outside my file list:** a single literal in `tests/test_tet_pipeline.py`. The supervisor approved it.
3. **Wave 2 test replaced.** `test_time_step_refinement_changes_total_time_little` asserted that a finer step finishes later, which is no longer true. It is replaced by `test_default_step_control_within_2_percent_of_converged`, which compares the default against (0.1, 0.5) on both the box and the riser assembly.

## Tests

- **Added:**
  - `InvertTetEnthalpyTest`, 9 tests: 8 hand-computed single-node cases plus fs parity against `SolidificationModel.update`. The single-node cases are all-liquid, entering the mushy zone, a mushy step, below solidus recalescing into the mushy zone, crossing solidus in one step, liquid to solid in one step, fully solid, and a reheated mushy node.
  - `TetAdiabaticEnthalpyConservationTest`: full solidification at dt 20 s with no boundary groups and constant properties. The check is relative 1e-6. The measured drift is 2.9e-14.
  - `TetStepControlDefaultsTest`: the fallbacks in `core/tet_thermal.py` equal the `MeshSettings` defaults.
  - Two `test_default_step_control_within_2_percent_of_converged` tests, one on the box and one on the riser assembly.
- **Results (`venv/bin/python -m unittest <module>`):**
  - `tests.test_tet_thermal`: 34 OK, 29.8 s.
  - `tests.test_tet_end_to_end`: 11 OK.
  - `tests.test_tet_pipeline`: 41 OK.
  - `tests.test_thermal_coupled_integration`: 10 OK.
  - `tests.test_thermal_loop_regressions`: 7 OK.
  - `tests.test_inert_settings_hidden`: 6 OK.
  - `tests.test_gpu_distance_chunk_setting`: 7 OK.
  - `tests.test_architecture_boundaries`: 5 OK.
  - `tests.test_parallel_kernel_parity`: 10 OK.
- **Not run:** the full suite (left to the integration step).

## Open risks

- **Temperature-dependent C.** Real runs use the 6-point alloy curves. `beta` uses C(T_post) while the heat step used C(T_old), and `C*T` is not the true enthalpy integral when C varies. Conservation is therefore exact only with constant properties, which is the tested case. Drift with the real curves was not measured.
- **Dead path in the voxel loop.** The per-point `beta` support that wave 2 added to `_apply_enthalpy_correction_numpy` now has no tet caller. `PerNodeEnthalpyCorrectionTest` still covers it. I left it alone because `core/solidification_loop.py` is not mine.
- **Defaults rest on two small cases.** Both total 100-145 s. On large castings the 2 s max_dt binds more often. The relative error should shrink there, but this is unmeasured.
- **Calibrated thresholds may shift.** Tet total times rose about 7 % at the defaults. The tet/voxel ratio against voxel 0.05 is now about 0.91, up from 0.85. Thresholds calibrated on wave 2 tet times will move.
- **The voxel path keeps the energy-losing T_sol cap.** It is unchanged by design.
