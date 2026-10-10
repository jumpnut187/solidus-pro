# Wave 2: integration (engine C4 + pipeline C5)

`run_pipeline(mesh_type="tet")` runs end to end with the real engine on a bare STL, on a casting + riser + chill `PartManager` assembly, and on that assembly with mold filling. The full suite shows only the 3 baseline failures. No source file needed a change. Nothing is staged or committed.

## Files changed

- `tests/test_tet_pipeline.py`: the engine stubs are gone.
  - Removed `_FakeEngine`, the `TET_THERMAL_READY` skip gate and `RunPipelineTetSmokeTests`.
  - Merged them with `RealEngineTetPipelineTests` into `RunPipelineTetTests`, which uses the real engine. It keeps the strongest assertions from both: the node-length initial field, arrival times, fill frames and payload `fill_time` / `fill_animation_history`.
  - `SurrogateOnTetTests` now runs the real engine and real `run_post_analyses`. It still wraps `unproject_grid` to check that the node coordinates are passed.
  - `PayloadMeshTypeTests` keeps a 9-line `_PayloadEngine` that holds only the arrays `build_visualization_payload` reads. It is a unit test of the payload builder on a hand-built 2-tet mesh and on a `_VoxelLike`, not a stand-in for the solve. The test count goes from 44 to 41 because 6 stubbed or duplicated tests became 3.
- `tests/test_tet_end_to_end.py` (new, 11 tests).
- No changes to `core/tet_thermal.py`, `core/tet_filling.py`, `core/simulation_pipeline.py`, `core/solidification*.py`, `core/surrogate/pipeline.py` or `config/settings.py`.

## Behaviour verified (tests/test_tet_end_to_end.py)

- **Settings isolation:** every run patches `get_settings` in `core.simulation_pipeline` and `core.tet_thermal` to a fresh `Settings()`, so user config files cannot change the result. The edge length is 0.2 in, and analysis options are the defaults (shrinkage and hotspot) unless noted.
- **Common checks (`assert_tet_run_complete`):**
  - The run returns a `TetVolumeMesh` and `PipelineResult.mesh_type == "tet"`.
  - Every `np.ndarray` attribute of the engine (about 45) has first dimension N nodes, and so do `mesh.voxel_states`, every `fraction_liquid_history` snapshot and every `export_results()` array.
  - Every metal node has a finite solidification time > 0.
  - The payload has `mesh_type` "tet", and `centers` and `solidification_times` have length N. `result_data` has `voxel_count == N` and `tet_count == len(tets)`.
- **(a) `cylinder.stl` through `STLLoader`:** the common checks pass, all tets are CAST, and the hot spot is on the axis at mid-height, (0.00, 0.00, 1.50) in.
- **(b) Assembly:** a 2x2x1 in box casting, a 1.2 in dia x 1.2 in riser on top, and a 1x1x0.4 in chill under half the casting. Parts are registered as `Part` objects, as `tests/test_desktop_orientation.py` does.
  - The common checks pass.
  - Tet codes are {CAST, RISER, CHILL}, and chill nodes have start time 0.
  - The riser freezes after the casting, and the cast hot spot lies under the riser.
  - A voxel run on the same assembly puts the cast hot spot within 2h (0.4 in) of the tet one.
- **(c) Assembly + filling** with `PourLocation(0, 0, 2.15)` on the riser top and default filling parameters:
  - The common checks pass, and "seeded initial temperature" is emitted.
  - The initial field, arrival times, every fill frame, and the payload `fill_time` and `fill_animation_history` all have length N.
  - At least 95 % of metal nodes are reached (100 % measured). Seeded temperatures lie between liquidus (1460 C) and pour (1621 C).
  - The cast hot spot is under the riser.
  - The fill finishes in about 1.05 s of pour on the default auto-coarsened (x2) lattice of 102 cells.

## Voxel vs tet, same assembly (WCB, default options)

CAD metal volume is 5.36 in³. Total = max solidification time over metal = `engine.time`. The cast hot spot is the argmax of `solidification_times` on CAST points.

| mesh | h (in) | points | metal vol (in³) | wall (s) | total (s) | cast max (s) | cast hot spot (in) |
|---|---|---|---|---|---|---|---|
| tet | 0.2 | 1654-1722 | 5.43 | 2.1-2.4 | 124.2-124.8 | 90.7 | (-0.10..-0.13, 0.0, 0.89) |
| voxel | 0.2 | 1056 | 7.73 | 3.6-3.7 | 194.7 | 150.1 | (0.00, 0.00, 0.80) |
| tet | 0.1 | 9014 | 5.39 | 12.3 | 125.8 | 92.3 | (-0.14, 0.00, 0.92) |
| voxel | 0.1 | 6979 | 6.37 | 9.6 | 156.5 | 113.0 | (-0.10, 0.00, 0.90) |
| voxel | 0.05 | 50757 | 5.85 | 44.6 | 143.3 | 101.8 | (-0.15, 0.00, 0.95) |
| tet + fill | 0.2 | 1649-1665 | - | 11.5-11.8 | 118.8-119.0 | - | (-0.12..-0.15, 0.0, 0.88-0.90) |

- In every run the hot spot lies under the riser neck, and the riser freezes last.
- **The totals differ because of voxel geometry, not the solver.** The voxel metal volume is 44 % over CAD at 0.2 in and 9 % over at 0.05 in, and its time falls toward the tet value as h shrinks. The tet result is h-converged: 124.5 s at 0.2 in and 125.8 s at 0.1 in.
- **Tet/voxel ratio at 0.05 in is 0.87.** That agrees with the 0.85 in `wave2_engine.md`, part of which is the known ~7 % early finish of the default step control.
- **Tet runs are faster at equal h** (2.3 s against 3.7 s). At equal geometric fidelity the gap is 2.3 s against 44.6 s.
- Wall times are from `/tmp` scripts and the test's stderr. These are single-process runs on an M5 Pro.

## Deviations from DESIGN.md / brief, with reasons

1. **Case (c) uses default filling parameters, not a short `max_fill_time_s` cap.**
   - `SolidificationParameters` clamps the cap to at least 1.0 s, so 0.05 s is ignored.
   - A fill cut at 1 s at coarsening x1 left 39 % of the metal misrun. Misrun nodes are seeded at mold temperature, and in one run that moved the cast hot spot to a bottom corner at (-0.81, 1.00, 0.00).
   - With default parameters the fill completes in about 1 s of pour and the run takes 11.5 s.
2. **`_PayloadEngine` remains in `test_tet_pipeline.py`** for the payload unit tests (see above). Every stub standing in for the tet solve is removed.
3. No production-code deviations. Wave 2 code was integrated as delivered.

## Tests run

| command | result | wall |
|---|---|---|
| `venv/bin/python -m unittest tests.test_tet_end_to_end` (x2) | 11/11 OK | 19.5-19.8 s |
| `venv/bin/python -m unittest tests.test_tet_pipeline` | 41/41 OK | 14 s |
| `venv/bin/python -m unittest tests.test_tet_thermal` | 22/22 OK | 13 s |
| `venv/bin/python -m unittest tests.test_architecture_boundaries` | 5/5 OK | 0.6 s |
| `venv/bin/python -m unittest tests.test_tet_mesh tests.test_tet_lattice tests.test_tet_heat_transfer` | 45/45 OK | 17 s |
| `venv/bin/python -m unittest discover tests > /tmp/wave2_full.log 2>&1` | 1931 run, 3 failures, 15 skipped | 290 s |

The 3 full-suite failures are exactly the DESIGN.md baseline failures:
- `test_filling_stability...test_a_deeper_tower_leaves_none_either`
- `test_porosity_benchmark_case_pack...test_case_pack_reference_targets` (riser_post_block)
- `test_shrinkage_display_intensity...test_shrink_scalar_below_one...`

There are no other failures or errors. The `ERROR:` lines in the log are expected log output from negative-path tests.

## Open risks

- **fTetWild is non-deterministic.** Node counts vary by about 4 % across runs (1654-1722) and the total by less than 0.5 %. The end-to-end tests bound geometry with 0.4-0.6 in tolerances and not exact positions.
- **Misrun seeding is cold.** Misrun nodes start at mold temperature and still pass through the enthalpy correction, so a truncated fill distorts solidification times. This is filling-stage semantics shared with the voxel path. It is visible on tets because node maps are coarse.
- **The default step control under-predicts freeze time by about 7 %** (`wave2_engine.md`). This needs an owner decision on `tet_max_delta_t_c`.
- **The voxel and tet totals are not comparable at equal h.** Any benchmark or calibration that compares them must use matched geometric volume.

## Wave 3 must handle

- **Post-analyses:** they run without exceptions on tets but are unvalidated.
  - Graph hop distances to riser and chill, feeding path, defect regions, isolation score and the orphan "smothered" rule all assume 26-slot rows.
  - Scalar `base_voxel_size` length scales must become local `voxel_sizes`.
  - Volume sums must use `voxel_volumes`.
- **Presentation:**
  - Payload `centers` are nodes and `sizes` are mean incident edge lengths, not cube sizes.
  - Ship `tets` and `tet_part_types`, and build the lattice with `TetLatticeSampler`. `core.tet_filling.build_filling_lattice` already pairs the sampler with the node mapping.
  - `shrinkage_percentage` is a node ratio, not a volume ratio.
- **Fill summaries:** counts are in fill-lattice cells, after coarsening (102 cells here against about 1650 nodes). The UI must label them as cells.
- **Two engine fields change meaning.** `engine.time_step` is the last accepted dt, and `thermal_gradients` is the true |grad T| in degC/mm.
- **ML:** `encode_channels` still scatters the raw node cloud and should encode from the lattice view. Records need `mesh_type`. `voxel_part_types` is int8 on tets, so encodings must cast.
- **UI and web:** plumb `PipelineInputs.mesh_type` through, preview with `estimate_tet_node_count` and `tet_max_nodes`, and show `MeshGenerationError` text as is.
