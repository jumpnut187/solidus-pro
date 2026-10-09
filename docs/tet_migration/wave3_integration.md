# Wave 3 integration

The five wave-3 lanes (enthalpy, post, present, ml, ui) are integrated. The full suite fails only the 3 baseline tests. `npm run build` passes. The tet e2e runs finish with no non-finite metal-node values. Nothing is staged or committed.

## Cross-lane fixes

1. **Riser design contract** (ui ↔ post). This was already consistent and needed no change: `RiserDesigner(..., voxel_volumes=None, mesh_type="voxel")`, called from `ui/main_window.py:1957`. UI risk 1 is closed.
2. **Fill-count labels** (present ↔ ui). `web/payloads.py` renames the tet fill counts to `fill_nodes_filled_count` / `fill_nodes_unfilled_count`, and they count nodes. The frontend relabelled the old keys as "lattice cells", so it was wrong twice: those keys never arrive on tet runs, and the counts are nodes, not cells.
   - Fix in `web_frontend/src/lib/statLabels.ts`: `TET_FILL_COUNT_LABELS` is removed, and OVERRIDES gain "Filled nodes" / "Unfilled nodes".
   - On tet runs the Overview rows now read "Nodes", "Cast nodes" and "Lattice spacing" (present/ui risk).
3. **`shrinkage_percentage`** (post ↔ pipeline). On tet runs it is now `engine.get_shrinkage_percentage()`, the volume share.
   - Changed in `core/simulation_pipeline.py::_add_summary_statistics`, plus the tet branch of the progress print in `core/analysis_strategies.py`.
   - The voxel branch is unchanged.
4. **Surrogate conditioning** (ml ↔ pipeline). On tet runs, `core/surrogate/pipeline.py` now sets `config["voxel_size"] = mesh.base_voxel_size` (the tet edge length). This matches `capture_run`.
5. **Feeder-neck padding** (post risk). On tets, `_estimate_feeder_neck_resistance_factors` now masks the -1 padding slots, which were being read as node 0. The voxel behaviour is kept as it was and is covered by a test.

## Files changed (this step)

- `core/simulation_pipeline.py`, `core/analysis_strategies.py`, `core/surrogate/pipeline.py`, `core/solidification.py` (tet branch only)
- `web_frontend/src/lib/statLabels.ts`
- `tests/test_tet_integration.py` (new, 5 tests, 0.5 s)
- Docs:
  - `AGENT.md`: tet architecture bullets, tet meanings, settings and env vars, pytetwild licence, Taichi FP gotcha, fTetWild stdout and orientation.
  - `docs/ARCHITECTURE.md`: live-path diagram with the tet branch, and the mesh object contract table.
  - `README.md`: mesh type section and tech-stack row.
  - `QUICKSTART.md`: mesh type option and the tet node-limit error. The STEP/STL guidance is unchanged.

## Tests

| command | result |
|---|---|
| `venv/bin/python -m unittest tests.test_tet_integration` | 5/5 OK |
| `QT_QPA_PLATFORM=offscreen venv/bin/python -m unittest tests.test_tet_integration tests.test_tet_thermal tests.test_tet_post_analyses tests.test_tet_presentation tests.test_tet_ml tests.test_tet_ui tests.test_tet_pipeline tests.test_tet_end_to_end tests.test_architecture_boundaries` | 162 run, OK, 69 s |
| `venv/bin/python -m unittest discover tests > /tmp/wave3_full.log 2>&1` | 2014 run, 3 failures (baseline only: `test_a_deeper_tower_leaves_none_either`, `test_case_pack_reference_targets` riser_post_block, `test_shrink_scalar_below_one...`), 15 skipped, 310 s. The same result with `QT_QPA_PLATFORM=offscreen`. |
| `cd web_frontend && npm run build` | pass (tsc + vite; only the existing chunk-size warning) |
| `venv/bin/python -m compileall` on the changed .py files | OK |

## End-to-end check (`/tmp/wave3_e2e.py`, artifacts in `/tmp/wave3_e2e/`)

- **Setup:** WCB, size 0.1 in, all 11 analyses on (filling included, pour at the centroid of the top face). Each STL is loaded as one CAST part through `PartManager`.
- **Artifact:** `build_fork_artifact` + `write_artifact` (JSON + `.grid.npz` sidecar).
- **Wall time:** `run_pipeline` only. The artifact build took 0.5-1.6 s on top.

| case | points | tets | metal in3 (CAD) | wall s | total solid. s | fill | shrink % | sidecar dims |
|---|---|---|---|---|---|---|---|---|
| thick_thin tet | 34,621 nodes | 186,630 | 21.00 (21.0) | 64.9 | 184.2 | complete 100 % | 57.0 | 31x31x50 |
| thick_thin voxel | 23,811 | - | 23.81 | 80.9 | 183.0 | complete 100 % | 55.8 | 31x31x51 |
| l_bracket tet | 13,482 nodes | 66,939 | 7.00 (7.0) | 37.2 | 66.3 | complete 100 % | 29.2 | 40x20x35 |
| l_bracket voxel | 8,946 | - | 8.95 | 33.2 | 76.1 | complete 100 % | 35.4 | 41x21x36 |

- **Non-finite values:** none. Every float `export_results()` array of length N was checked on the metal nodes: no NaN or Inf, and no unfinished metal nodes. No `NaN` or `Infinity` appears in any artifact JSON. Each tet sidecar records `mesh_type="tet"`.
- **Repeatability:** a second thick_thin tet run (with an STLLoader input) gave 34,491 nodes and 183.9 s, within fTetWild's run-to-run variation.

## Deviations

1. The e2e driver uses a `PartManager`, not a bare `STLLoader`. A voxel run with filling and a bare `stl_loader` crashes in `core/filling_stage.py:661` with `IndexError`, because `voxel_part_types` is None. This is pre-existing: the file is untouched by waves 1-3. The tet path works with both inputs, and both UIs pass a `PartManager`.
2. I added the overview relabelling (Nodes / Lattice spacing) in the frontend. The present and ui lanes had both listed it as an open risk, and it is a label-only change.
3. The full suite was run both with and without `QT_QPA_PLATFORM=offscreen`. The results are identical.

## Open risks

- The voxel-path filling crash with a bare `stl_loader` (deviation 1) is unfixed. It is outside wave 3 scope and voxel behaviour must not change.
- The web client default `mesh_type` is `'voxel'` (`web_frontend/src/lib/defaults.ts`). Flipping the default in wave 4 needs both `settings.mesh.mesh_type` and `performanceDefaults` changed, or the server default exposed.
- Tet enthalpy conservation is exact only for constant properties. Drift with the 6-point alloy curves is unmeasured (enthalpy lane).
- Stress distortion magnitude is not calibrated on tets, because node degree differs from 26. The orphan pocket rule counts nodes, not volume (post lane).
- Desktop and web read `tet_lattice_spacing_factor` and `tet_edge_length_factor` from the current settings, not from settings saved with the run.
- Refinement level and memory mode stay editable on tet in both UIs and are ignored there.
- On these sizes, tet wall time is similar to voxel: 0.8x on thick_thin and 1.1x on l_bracket, with 1.45-1.5x more points. Large castings are unmeasured.
