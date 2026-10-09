# Wave 2: pipeline, settings, filling (contract C5)

## Files changed

- `config/settings.py`: `MESH_TYPES`, tet fields on `MeshSettings`, `normalize_tet_fields()`, 5 env overrides, YAML/TOML templates.
- `core/simulation_pipeline.py`: mesh selection, tet meshing with node-count guard, filling dispatch, `mesh_type` in payload and `PipelineResult`.
- `core/tet_filling.py` (new): lattice filling seam, FILTER rasterization, node mapping.
- `core/surrogate/pipeline.py`: filling through the dispatch, comment on node unprojection. ML encoding untouched.
- `tests/test_tet_pipeline.py` (new): 44 tests.
- `core/filling_stage.py`: not changed and no change needed. It reads only `voxel_centers`, `voxel_sizes`, `voxel_part_types`, `voxel_volumes` and `get_voxel_count()`.

## Behaviour

1. `MeshSettings` adds `mesh_type="voxel"`, `tet_edge_length_factor=1.0`, `tet_epsilon_rel=1e-3`, `tet_max_time_step_s=2.0`, `tet_max_delta_t_c=10.0`, `tet_lattice_spacing_factor=1.0` and `tet_max_nodes=3_000_000`.
   - `normalize_tet_fields()` runs in `__post_init__` and again after file and env load, next to `deep.normalize()`.
   - An unknown `mesh_type` falls back to "voxel" with a `RuntimeWarning`. A non-positive, NaN or non-numeric value falls back to its default with a warning.
   - The existing `base_voxel_size` clamp still runs only at construction, so voxel-path behaviour is unchanged.
2. The env overrides are `CASTSIM_MESH_TYPE`, `CASTSIM_TET_EDGE_LENGTH_FACTOR`, `CASTSIM_TET_MAX_TIME_STEP_S`, `CASTSIM_TET_MAX_DELTA_T_C` and `CASTSIM_TET_LATTICE_SPACING_FACTOR`.
3. `PipelineInputs.mesh_type=None` uses the settings value. `resolve_mesh_type()` lets an explicit value win and is case-insensitive.
4. On a tet run, `prepare_mesh_and_geometry` calls `build_tet_mesh(parts_with_types or [(mesh, CAST)], voxel_size * tet_edge_length_factor, epsilon_rel=tet_epsilon_rel)`.
   - Progress is mapped to `detailed_progress_callback(round(100*f), 100, stage)`, and cancel is checked on every callback.
   - `num_threads` is the fair-share thread budget when a provider is set, otherwise 0 (all threads).
   - It emits messages for the node, tet and face counts and for memory.
   - The return tuple is unchanged: `parts_with_types` stays `[]` for a bare STL.
5. Node-count guard: `estimate_tet_node_count = ceil(2.0 * sum(part volume) / h^3)`.
   - Watertight parts use `|volume|`. Open parts use the convex-hull volume, falling back to the bounding box.
   - Above `tet_max_nodes` it raises `MeshGenerationError` before meshing. The message names the estimate, the limit, the `tet_max_nodes` setting and the smallest voxel size that fits.
6. Payload: `visualization_data["mesh_type"]`, `result_data["mesh_type"]`, and `result_data["tet_count"]` on tet runs.
   - `PipelineResult.mesh_type` is "tet" or "voxel". There was no metadata dict, so it is an attribute.
   - `voxel_count` is the node count on tet runs.
7. `run_filling_for_mesh()` in `simulation_pipeline` dispatches to `maybe_run_filling` for voxel meshes and to `maybe_run_tet_filling` for tet meshes. Both pipelines call it, so the surrogate path cannot drift from `run_pipeline`.
8. `maybe_run_tet_filling()` runs these steps:
   1. Applies the same gate as the voxel path. When filling is off or no pour location is set, no lattice is built and the voxel path's notice is emitted.
   2. Builds `TetLatticeSampler(node_xyz, tets, tet_part_types, voxel_size * tet_lattice_spacing_factor).lattice_view()`.
   3. Rasterizes FILTER parts (below).
   4. Runs `maybe_run_filling` on the result.
   5. Maps `fill_time_s`, `temperature_c`, `entrapped_gas`, `cold_shut` and each frame's `fill_fraction` and `speed_m_s` to nodes.
   - `LatticeTooLargeError` or an empty lattice gives "Filling stage skipped: ..." and returns None, matching how the voxel path handles a mapping `ValueError`.
9. Node mapping uses one cKDTree on cell centres per class. Metal nodes take the nearest metal cell and non-metal nodes take the nearest non-metal cell. When the view has no cell of the node's class, it uses the nearest cell of any class.
10. The surrogate pipeline works on a `TetVolumeMesh`. `unproject_grid` receives `voxel_centers == node_xyz`, so start and local times come back per node. The filling seam is the same as in item 8.

## FILTER rasterization (approved by supervisor during this run)

- C1 removes FILTER tets, so the sampled lattice had no filter cells. The fill then treated the filter gap as a wall. `test_without_filter_cells_the_fill_stops_at_the_filter` shows 0 casting nodes reached when the filter is missing.
- `filter_lattice_centers()` takes the sampler's own cell centres (`origin + spacing*ijk`) inside each FILTER part's bounding box, skipping cells already inside the tet mesh.
  - It tests containment with the voxel path's `_robust_contains_check_impl`, using the same 0.5-pitch tolerance as `assign_part_types`. That helper handles non-watertight meshes through signed-distance, proximity and bounding-box fallbacks.
  - It emits the message "rasterized N FILTER cells ... with the voxel-path containment check".
- `FillingLatticeMesh` appends the filter cells after the view cells. When there is no filter, the fill runs on the `LatticeView` itself.
- Filter cells are filling-only. `node_cells` indexes view cells only, which is asserted in a test. `build_grid_mapping` accepts the combined set, also asserted.
- In the casting + 0.4 in filter + sprue assembly, metal reached 45 of 533 casting nodes below the filter in 0.2 s of fill, and 0 without the filter cells.

## Deviations from DESIGN.md / brief, with reasons

1. FILTER rasterization was added, as described above. Without it, filling stalls on every filtered assembly.
2. The mapping is the nearest cell of the same class (metal or non-metal), not just the nearest cell. Otherwise metal nodes on a chill face can inherit the 21 C seed of an adjacent chill cell.
3. An explicit unknown `PipelineInputs.mesh_type` raises `ValidationError`, while the settings value falls back with a warning. A typo from a caller should fail loudly.
4. The 2.0 nodes/(V/h^3) factor is not an upper bound for small or thin parts. Measured ratios:

   | Part | h (in) | ratio |
   |---|---|---|
   | 2x1x1 box | 0.2 | 2.8 |
   | l_bracket | 0.1 | 2.12 |
   | cylinder | 0.1 | 1.24 |
   | cylinder | 0.05 | 1.18 |

   I kept 2.0 as briefed. Near the 3 M limit h is small relative to the features and the ratio falls, so the guard can under-count by about 6 % on thin parts.
5. `tet_epsilon_rel` and `tet_max_nodes` have no env override because the brief listed five. They are settable from files.
6. A local in `build_visualization_payload` was renamed (`fill_times` to `fill_frame_times`) to fix a pre-existing mypy redefinition. mypy goes from 63 to 62 errors, with no new errors in my files.

## Tests

- `venv/bin/python -m unittest tests.test_tet_pipeline`: 44 run, 44 pass, 17-24 s wall.
  - `FilterAssemblyFillingTests`: 13.9 s, including a real fill of 0.2 s simulated time.
  - `RunPipelineTetSmokeTests`: 5.9 s. `_FakeEngine` stands in for the engine and the real fill hands over node-length fields.
  - `RealEngineTetPipelineTests`: 5.0 s. These are gated on `core.tet_thermal` importing. They now run against the engine worker's uncommitted `core/tet_thermal.py` and pass for `run_pipeline`, `run_pipeline` with filling, and `run_surrogate_pipeline`.
- `tests.test_architecture_boundaries`: 5/5 OK. The other existing modules I ran all pass, about 450 tests in total:

  | Module | Tests |
  |---|---|
  | `test_simulation_pipeline_options` | 14 |
  | `test_filling_stage` | 111 |
  | `test_initial_temperature_seam` | 11 |
  | `test_surrogate` | 25 |
  | `test_inert_settings_hidden` | 6 |
  | `test_gpu_distance_chunk_setting` | 7 |
  | `test_fill_time_visualization` | 14 |
  | `test_simulation_scenarios` | 16 |
  | `test_surrogate_geometries` | 9 |
  | `test_web_api` | 16 |
  | `test_web_fields` | 80, 5 skipped |
  | `test_web_simulation_process` | 17 |
  | `test_simulation_worker` | 3 |
  | `test_tet_mesh` | 21 |
  | `test_tet_lattice` | 11 |
  | `test_deep_model` | 44, 2 skipped |
  | `test_centered_riser_symmetry` | 3 |
  | `test_fraction_liquid_timing_consistency` | 12, 5 skipped |
  | `test_hot_tear_physics` | 13 |
  | `test_performance_summary_label` | 14 |

- `venv/bin/python -m mypy --config-file mypy.ini`: 62 errors, against 63 at HEAD, all pre-existing. I did not run the full suite.

## Open risks

- `FillingResult` counts (`filled_voxel_count`, `misrun_voxel_count`, `inlet_voxel_count`, `filled_fraction`) and `engine._filling_summary` are in lattice cells. The handover message says "lattice cells". `fill_time_stats` in the payload counts nodes.
- A filter whose box lies outside the metal-node bounding box is clipped to the lattice, so cells beyond the outermost metal are lost.
- The 0.5-pitch proximity tolerance can make a filter up to half a cell fatter into the mold. It never overrides metal or chill cells.
- On the filter assembly, `filling_stage.py:3670` emits matmul `RuntimeWarning`s (divide by zero, overflow). I trapped them with `np.seterrcall`: the inputs were finite and rho was about 1, so the flags look spurious (Accelerate BLAS). They do not appear in `test_filling_stage`.
- The surrogate pins `schedule_legacy`. The tet engine prints "schedule_legacy is not available on a tet mesh; running thermal_coupled", which is harmless because the surrogate never calls `run_simulation`.

## Wave 3 must handle

- Presentation:
  - Payload `sizes` on tet runs are local edge lengths, not cube sizes.
  - `centers` are nodes. Ship `tets` and `tet_part_types`, and build the lattice with `TetLatticeSampler`. `core/tet_filling.build_filling_lattice` can be reused for the sampler and the node mapping.
  - `shrinkage_percentage` = affected/voxel_count is a node ratio, not a volume ratio.
- ML: `encode_channels` still scatters the raw node cloud (C6 says encode from the lattice view). `unproject_grid` onto nodes already works.
- UI and web: pass `PipelineInputs.mesh_type` from the toggles, preview with `estimate_tet_node_count` and `tet_max_nodes`, and surface the `MeshGenerationError` text as is.
- Filling summary UI: label counts as lattice cells on tet runs.
