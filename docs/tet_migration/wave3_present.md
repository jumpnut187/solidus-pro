# Wave 3: presentation (web sidecar, desktop result surface, inspectors)

## Files changed

- `core/dense_grid.py`: adds `tet_presentation_spacing(mesh)`, `lattice_from_sampler(sampler)`, `sample_node_values(sampler, node_values, node_ok)` and `NODE_SAMPLE_FRAME_BATCH`. The voxel functions are unchanged.
- `web/payloads.py`: tet grid source, `mesh_type` in the JSON preview, and node labels for the tet fill counts.
- `web/fields.py`: `_build_tet_field_grids`, `_build_tet_frame_stack`, `TetGridUnavailableError`, `TET_MESH_TYPE`, `FieldGrids.mesh_type`, and `mesh_type` in the sidecar `meta_json` (tet runs only).
- `web/routes_fields.py` (approved by the supervisor for this one change): returns 404 `tet_grid_missing` when a tet run cannot be rebuilt.
- `ui/result_surface.py`: `set_tet_mesh(node_xyz, tets, tet_part_types, spacing)`, and a shared `_install_lattice` used by both paths.
- `ui/visualizer.py`: tet branch in `set_result_mesh`, `mesh_type=` passed to `extract_pocket_at`, and defect volume summed from node volumes.
- `tests/test_tet_presentation.py` (new): 20 tests.

## Behaviour

1. Grid source on tet runs (`_full_grid_source`):
   - Holds `mesh_type="tet"`, `tets` (int32, (T,4)), `tet_part_types` (int8) and `lattice_spacing`, plus the existing per-node field arrays.
   - `centers` is `node_xyz` as float64. With float32 nodes the sampler box was one cell taller (10,10,14 against 10,10,13), so the lattice no longer matched the one desktop and filling build.
   - Everything stays numpy and is never written to JSON. `write_artifact` still pops the grid source.
2. `build_field_grids` on a tet source:
   - Builds `TetLatticeSampler(nodes, tets, tet_part_types, lattice_spacing)`. The mask is `mask_codes()`.
   - Scalars and frames are `sample()` values at the inside cells. Cells outside the mesh are filled from the nearest inside cell through the existing `scatter_and_fill`.
   - Frames are sampled 8 at a time to cap memory.
   - The sidecar layout (dims, origin, spacing, uint8, x-fastest, mask = code+1) is unchanged.
   - `voxelCount`, `castVoxelCount` and `partTypes` are node counts. A code that only tets carry gets a row with count 0, so every mask code has a colour.
3. A tet node cell can have one or more unusable corner nodes (NaN, or a misrun when fill_time is -1). Such a cell takes the value of its `nearest_node`, and is unknown (then hole-filled) when that node is also unusable. Bad values therefore never leak through the interpolation weights.
4. Lattice spacing is `voxel_size * tet_lattice_spacing_factor`, the same pitch filling uses. The sidecar `spacing` field stores it.
5. A tet artifact with no usable sidecar:
   - `build_field_grids` raises `TetGridUnavailableError` and never falls back to `derive_lattice`.
   - Every grid endpoint returns 404 `{"code": "tet_grid_missing", "error": "This tet-mesh run has no saved field grid. Re-run it to regenerate the result fields."}`.
   - The frontend shows this text as is: `gridApi.failure()` puts `body.error` into `GridRequestError.message`, and `ResultsViewport` passes it to `onError`.
6. The JSON `visualization` keeps the strided node preview (`centers`, `part_types`, fields) and `mesh_type`. `sample_field_at`, `build_envelope`, the skin grid and the byte endpoints need no change.
7. Desktop:
   - `set_tet_mesh` installs the sampler's inside cells as the renderer's "voxels".
   - `show()` requires node-length `values`. Wrong lengths, including the cell count, return False.
   - Per-node values are sampled barycentrically onto cells. Per-node flags (`valid`, `paint_mask`, `flat_paint`, display mask) follow `nearest_node`.
   - `view.values` stays per node, so a re-show resamples.
   - `voxel_at` returns `nearest_node` of the picked cell.
   - `show()`, the envelope and the caps are otherwise unchanged.
8. Visualizer:
   - `set_result_mesh` calls `set_tet_mesh` when `mesh_type == "tet"`.
   - The shrink inspector passes `mesh_type=getattr(mesh, "mesh_type", "voxel")` to `extract_pocket_at`. The post worker's `core/defect_regions.py` already accepts it.
   - `calculate_defect_volume` sums node `voxel_volumes` on tets. `MeasurementTool.calculate_defect_volume` gains an optional `voxel_volumes`.
9. Fill counts: on tet runs the summary keys `fill_filled_voxel_count` and `fill_unfilled_voxel_count` become `fill_nodes_filled_count` and `fill_nodes_unfilled_count`, and `fill_time_stats.*_voxels` become `*_nodes`. These are counted from node arrival times, so they are nodes, not lattice cells. None of my files shows the FillingResult counts (lattice cells). The pipeline message already says "lattice cells", and `ui/main_window.format_filling_notice` is not my file.

## Deviations, with reasons

1. Spacing: the brief says `base_voxel_size * tet_lattice_spacing_factor`. On a `TetVolumeMesh`, `base_voxel_size` is the edge length (`voxel_size * tet_edge_length_factor`), so `tet_presentation_spacing` divides the edge factor back out to match the filling lattice. With default factors both give the same value.
2. Nothing beyond the sidecar is persisted (supervisor decision). A tet run whose sidecar is lost has to be re-run.
3. The tet fill counts are labelled "nodes", not "lattice cells", because they count nodes (item 9 above).
4. `web/routes_fields.py` was edited with supervisor approval. The voxel responses are unchanged.

## Tests

- `venv/bin/python -m unittest tests.test_tet_presentation`: 20 run, 20 pass, about 3 s (one real tet pipeline run at edge 0.2 in). Two consecutive runs both passed.
- The same command with `tests.test_web_fields tests.test_result_surface tests.test_web_api tests.test_web_simulation_process tests.test_fill_time_visualization tests.test_architecture_boundaries tests.test_visualizer_inspector tests.test_web_skin` added: 192 run, all pass, 5 skipped (pre-existing skips).
- `test_fill_time_visualization` logs a `vtkFlyingEdges3D returned failure`. It comes from its 5x1x1 voxel lattice on the unchanged voxel path, and the test passes.
- mypy (`mypy.ini` allowlist): 62 errors, all in files I did not touch. None in `web/fields.py`, `web/payloads.py`, `web/routes_fields.py` or `core/dense_grid.py`.
- `compileall` on the changed files: OK.

## Open risks

- The frontend labels `voxelCount` "Voxels". On tet runs it is the node count, and changing the label is web_frontend work for the UI worker. The renamed fill keys get auto-labels: "Fill nodes filled" and "Fill nodes unfilled".
- `voxel_at` picks a node within one edge (test: at least 90 % of interior nodes return the node itself or a neighbour, all within 2 local edges). Near boundaries it can be up to about 2.2 edges away.
- Desktop and web read `tet_lattice_spacing_factor` and `tet_edge_length_factor` from the current settings, not from settings saved with the run. Changing either between a run and its rendering changes the lattice pitch, but not correctness.
- `_ensure_result_lattice` (the desktop lazy fallback when `set_result_mesh` was never called) would still derive a lattice from a node cloud. `main_window` always calls `set_result_mesh` first.
