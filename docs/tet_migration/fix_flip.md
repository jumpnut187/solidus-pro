# Flip lane: tet becomes the default mesh

Branch feature/tet-mesh at 9514111, uncommitted. Nothing staged. The voxel code path is unchanged; only defaults, labels, docs and tests moved.

## Settings and defaults changed

- `config/settings.py`:
  - `MeshSettings.mesh_type` now defaults to `"tet"`. An unknown value warns and falls back to `"tet"`.
  - `tet_edge_length_factor` defaults to 2.0 in the dataclass, `_TET_POSITIVE_DEFAULTS`, and the YAML and TOML templates.
- Why factor 2.0 (fix_physics data, re-measured below): at factor 2 the tet total is within 0.4 % of the 0.05 in tet reference on both parts and the field error is at most 0.7 %. Voxel at 0.1 in is +1.6 % / 4.0 % on thick_thin and +13.2 % / 14.1 % on l_bracket. Tet at factor 2 is also faster on thick_thin (1.4 s vs 2.6 s). On l_bracket it was 2.4 s against voxel's 1.0 s in this run, and fix_physics measured 0.8 s.
- Web (`web_frontend/src/lib/defaults.ts`): `performanceDefaults.mesh_type` is now `'tet'`. `meshTypeOf` returns `'tet'` unless the value is `'voxel'`. A legacy project with no mesh type therefore opens as tet (App.tsx merges `performanceDefaults` on load).
- Desktop (`ui/main_window.py`):
  - `MeshDialog` still starts on `settings.mesh.mesh_type`, which is now tet. `set_mesh_type` maps an unknown value to Tetrahedral.
  - Legacy projects already fall back to the settings value (main_window.py:5338), so both UIs open them as tet.
- Effective edge display (parent decision): the size control is the base size, edge = size x factor.
  - Desktop `MeshDialog`: label "Mesh size (in):" and a live "Tet edge ≈ 0.200 in" line under the spin box (hidden on voxel), via new `ui.main_window.tet_edge_length_in`. The sidebar summary reads `Tet: 0.100", edge ≈ 0.200"`, and it is refreshed at startup (the old hardcoded "Voxel: 0.100\"" no longer shows).
  - Web `MeshPanel`: "Mesh size" plus "Tet edge ≈ X in". The factor is `edge_length_in / voxel_size` from the mesh-preview response (`tetEdgeFactorOf` in defaults.ts). It is fetched once when the panel opens on tet with parts, and refreshed by Estimate mesh and ML mesh. Before it is known, the panel says the edge is the mesh size x the server factor. The preview warning reads "Safe mesh size".
- `CG_RTOL` stays at 1e-10 (physics/tet_heat_transfer.py:59, untouched).
- Docs: README.md, QUICKSTART.md, AGENT.md and docs/ARCHITECTURE.md now say tet is the default and voxel is the fallback (`CASTSIM_MESH_TYPE=voxel`), with a factor of 2.0.

## Cross-lane mismatches resolved

- fix_integration risk 2: `tools/generate_surrogate_data.py` now resolves `settings.mesh.mesh_type` once. It passes that value to `generation_plan_key` and to `generate_dataset`, so the dry-run status table reads the same keys the run writes. Verified: `--dry-run` prints `mesh: tet`.
- fix_integration risk 3 (web client default `'voxel'`): fixed as described above.
- fix_geometry's estimate change in `test_tet_ui` was already adopted by fix_integration. No conflict remains.

## Tests pinned to voxel, and why

- `tests/test_simulation_scenarios._run_pipeline_with_patched_memory` now takes `mesh_type="voxel"` by default, with a docstring explaining why:
  - It patches `core.voxel_mesh` memory checks, which do nothing on tet.
  - Its 0.5 in size gives 1.0 in tet edges, only 2-4 elements across these parts.
  - Its expected values were calibrated on the voxel solver.
  - On tet, `test_higher_sand_restraint...` and `test_riser_post_behaves...` flipped between pass and fail from run to run.

  The helper's callers therefore run voxel: test_simulation_scenarios, test_porosity_benchmark_case_pack, test_unfinished_voxel_masking, test_chill_exclusion, test_sand_burn_in, test_fraction_liquid_timing_consistency, test_surrogate_chill_channel, and test_chill_interface_safety (apart from the new tet class).
- `test_initial_temperature_seam._build_engine` is pinned to voxel:
  it patches `core.voxel_mesh`, compares two meshings element by element, and expects chill-interface metal to keep its seed, but tet blends shared interface nodes (fix_physics finding 1).
- `test_centered_riser_symmetry` is pinned to voxel. It is a regression test for the voxel neighbour-order bias, and an unweighted mean of tet node positions is not the centroid.
- `test_surrogate_geometries.test_plate_with_boss_voxelizes_solid_and_captures` is pinned to voxel because it checks that voxelization is solid.
- `test_surrogate.test_surrogate_vs_traditional_parity` is pinned to voxel. It compares two meshings point by point (753 vs 749 nodes on tet).
- `tests/verify_deep_benchmarks.py` `_inputs` is pinned to voxel because it benchmarks surrogates trained on voxel records. `verify_thermal_benchmarks.py` and `verify_chill_decay.py` call the solver directly, so they need no pin.

## Tests changed for the tet default (not pinned)

- `test_tet_pipeline`:
  - Settings tests now expect tet, factor 2.0, and an unknown value falling back to tet.
  - The env and config-file override tests now override to voxel.
  - `test_none_uses_settings` and `test_explicit_inputs_win_over_settings` use the new default.
  - `test_guard_refuses_before_meshing` sets factor 1.0, because its 1,213 estimate is for a 0.2 in edge.
- `test_tet_post_analyses._run_all_analyses` and `test_tet_ml._default_settings` set factor 1.0, because the size they pass is meant as the edge length (`TET_EDGE_IN` / `EDGE_LENGTH_IN`). At factor 2, distance monotonicity had only 3 z-levels against the 4 required, and the isolation-range difference was 0.158 against a limit of 0.15.
- `test_tet_end_to_end` (new `run_settings()`) and `test_tet_presentation` use factor 1.0: `EDGE_LENGTH_IN` is the edge, and presentation spacing divides by the current factor (open risk 1). The hot-spot distance test failed 1 of 3 full runs at factor 2.
- `test_tet_ui`: unknown values now select tet. New `test_tet_shows_the_effective_edge_length` and `test_sidebar_summary_shows_the_effective_edge_length`.
- `test_chill_interface_safety`:
  - On tet, `_index_chill_bands` defines the chill-adjacent set as the CAST nodes incident to a CHILL tet.
  - The voxel rule ("within one pitch of a chill node") picks the second node layer on tet, which is why the first full run failed there.
  - New `ChillInterfaceRiskTetTests` runs all 8 guarantees on tet: 7 pass and 1 is skipped (same skip as voxel). It passed 3 of 3 reruns.
  - Probe at 0.1 and 0.5 in: interface nodes have niyama risk, feed risk and risk modulus all 0, porosity ≤ 0.056, and none is missing from the material surface. This was a test definition problem, not a tet bug.
- No production code was changed for these tests. No real tet bug was found.

## Full suite

- `venv/bin/python -m unittest discover tests > /tmp/flip_full.log 2>&1` (tet default): **2070 run, 3 failures, 16 skipped**. These are the 3 baseline failures: `test_a_deeper_tower_leaves_none_either`, `test_case_pack_reference_targets` (riser_post_block), and `test_shrink_scalar_below_one_...`.
- `CASTSIM_MESH_TYPE=voxel venv/bin/python -m unittest discover tests`: **2070 run, the same 3 baseline failures, 16 skipped**.
- Targeted runs:
  - 176 tests (chill_interface_safety, tet_pipeline, tet_ml, tet_ui, unfinished_voxel_masking, initial_temperature_seam, centered_riser_symmetry, surrogate_geometries, surrogate, architecture_boundaries): OK, 2 skipped.
  - scenarios + case_pack + unfinished_masking: 24 run, only the baseline case_pack failure.
- `cd web_frontend && npm run build` passes, and `npx eslint` is clean on defaults.ts, MeshPanel.tsx and App.tsx. tests.test_tet_ui, test_web_api, test_architecture_boundaries and test_simulation_worker: 49 run OK.
- `mypy --config-file mypy.ini`: no errors in the files I changed. The 62 errors are in simulation_queue, filling_stage, simulation_pipeline and skin.
- `compileall` passes on the changed Python files.

## Benchmark (`tools/benchmark_tet_vs_voxel.py`, defaults, base 0.1 in)

Reference: tet with 0.05 in edges. thick_thin has 278,740 nodes and a total of 191.31 s. l_bracket has 112,181 nodes and a total of 73.00 s.
| part | mesh | size in | factor | points | wall s | total s | vs tet ref | field err |
|---|---|---|---|---|---|---|---|---|
| thick_thin | voxel | 0.1 | - | 23811 | 2.6 | 194.39 | +1.6 % | 4.0 % |
| thick_thin | voxel | 0.05 | - | 179021 | 19.4 | 190.20 | -0.6 % | 0.8 % |
| thick_thin | tet | 0.1 | 1 | 34575 | 11.2 | 191.40 | +0.0 % | 0.2 % |
| thick_thin | tet | 0.15 | 1.5 | 8256 | 2.7 | 192.12 | +0.4 % | 0.4 % |
| thick_thin | tet | 0.2 | 2 | 4455 | 1.4 | 192.08 | +0.4 % | 0.6 % |
| l_bracket | voxel | 0.1 | - | 8946 | 1.0 | 82.67 | +13.2 % | 14.1 % |
| l_bracket | voxel | 0.05 | - | 63591 | 5.9 | 76.00 | +4.1 % | 4.4 % |
| l_bracket | tet | 0.1 | 1 | 13431 | 4.4 | 73.11 | +0.1 % | 0.3 % |
| l_bracket | tet | 0.15 | 1.5 | 4322 | 1.5 | 73.24 | +0.3 % | 0.5 % |
| l_bracket | tet | 0.2 | 2 | 2709 | 2.4 | 73.04 | +0.1 % | 0.7 % |
Script output: "Recommended tet_edge_length_factor: 2".

## Deviations

- The tet size control is renamed "Mesh size" in both UIs, because at factor 2 the value is half the edge length. The run statistics still show "Element size" from `tet_edge_length_in`, which is the true edge length. A comment in `web/routes.py` (ml_mesh) is reworded to match.
- `tools/generate_surrogate_data.py` was edited to close fix_integration's risk 2.

## Open risks

1. `tet_presentation_spacing` divides by the current factor. A tet result saved at factor 1.0 and re-rendered now gets a lattice pitch half its run's, so 8x the cells, and may return 413 `lattice_too_large`. Web sidecars written at run time are not affected.
2. Surrogate conditioning `voxel_size` on tet is the edge length, so it is now 2x the mesh size. Tet records captured at factor 1 and factor 2 differ in that feature. The shipped surrogate and deep models are trained on voxel records.
3. Scenario and case-pack fixtures no longer exercise tet, because the helper defaults to voxel. Tet coverage comes from the test_tet_* modules and the new chill-interface tet class.
4. At the default 0.1 in mesh size the tet edge is 0.2 in, so walls thinner than about 0.4 in get fewer than 2 elements. The factor choice rests on 2 parts measured on 1 machine.
5. Existing projects keep their saved size, so after the flip they re-run as tet with 2x that size as the edge length.
