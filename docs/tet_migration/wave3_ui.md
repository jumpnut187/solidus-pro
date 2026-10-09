# Wave 3: UI mesh-type toggle (desktop + web)

Owner: UI worker. Nothing committed or staged.

## Files changed

- `ui/main_window.py`, `ui/workers.py`, `ui/project_state.py`
- `web/services/simulation_queue.py`, `web/routes.py` (`web/repository.py` untouched, no schema change)
- `web_frontend/src/App.tsx`, `components/panels/MeshPanel.tsx`, `lib/api.ts`, `lib/defaults.ts`, `lib/statLabels.ts`
- `tests/test_tet_ui.py` (new, 19 tests, ~2.5 s)

## Behaviour as implemented

Storage: `mesh_type` ("tet" | "voxel") lives in `performance_options` on both front ends. No new column, signal argument or project schema version.

Desktop
1. `MeshDialog`: "Mesh type" combo (Tetrahedral / Voxel, item data `tet`/`voxel`) above the size spin box. The size label reads "Element size (in):" on tet and "Base Voxel Size (inches):" on voxel. The default is `settings.mesh.mesh_type`. The combo is covered by `store/restore_original_values`. New `get_mesh_type()` and `set_mesh_type()` (an unknown value selects Voxel). `set_values(..., mesh_type=None)` leaves the type unchanged when None.
2. `MainWindow`:
   - Seeds `_performance_options['mesh_type']` from the dialog.
   - `_on_mesh_settings_confirmed` stores the dialog value. Project load restores it, falling back to the settings default when absent.
   - The sidebar summary shows `Tet: 0.100"` on tet.
   - The surrogate worker gets `mesh_type`.
3. `_run_simulation`: on tet, the voxel memory pre-validation (`validate_mesh_parameters`) is skipped. The pipeline's `tet_max_nodes` guard applies instead. The voxel branch is unchanged.
4. Failure dialog: `SimulationWorker` stores `mesh_error_message = str(exc)` for any `MeshGenerationError`. `_on_simulation_finished` shows that text verbatim in a "Mesh Generation Error" box. Other failures keep the generic box.
5. Riser design (`RiserDesignDialog`):
   - On tet, `_prepare_mesh_from_parent` calls the new `build_riser_design_tet_mesh(cast_mesh, voxel_size)`. That function runs `estimate_tet_node_count` against `tet_max_nodes` (raising `MeshGenerationError`), then `build_tet_mesh` at `voxel_size * tet_edge_length_factor`.
   - `_run_analysis` passes `voxel_volumes=mesh.voxel_volumes`, `mesh_type=mesh_type_of(mesh)`, `neighbors=mesh.voxel_neighbors` and `surface_mask=mesh.surface_voxels`.
   - A `MeshGenerationError` is shown verbatim.
6. `ui/workers.py`: `SimulationWorker` passes `performance_options.get("mesh_type")` to `PipelineInputs.mesh_type` (None = settings default). `SurrogateWorker` gets a new `mesh_type=None` kwarg and passes it the same way. On tet, the progress text says "Generating tetrahedral mesh".
7. `ui/project_state.py`: `load_project_state` normalizes `performance_options.mesh_type` (strip, lower-case) and drops unknown values. Absent stays absent. A non-dict `performance_options` now loads as `{}` instead of raising.

Web backend
1. `build_pipeline_inputs`: `mesh_type=performance.get("mesh_type")`, None when absent.
2. `web/routes.py`:
   - New `requested_mesh_type(body)`: request `mesh_type`, else `settings.mesh.mesh_type`. Unknown values return 400.
   - New `tet_mesh_estimate(manager, voxel_size)`.
   - `mesh-preview` on tet returns `{mesh_type:"tet", estimated_nodes, max_nodes, edge_length_in, can_proceed, message, recommended_voxel_size}`. `recommended_voxel_size` is the size that lands at the limit, rounded up to 0.001 in, or null when under the limit.
   - `ml-mesh` on tet adds `mesh_type, estimated_nodes, max_nodes, edge_length_in, can_proceed` for the recommended size.
   - Voxel responses are byte-for-byte the same key sets as before.

Web frontend
1. `performanceDefaults.mesh_type = 'voxel'`, because the server does not expose its default. New `meshTypeOf()` and `MeshType`.
2. `MeshPanel`: a segmented Tetrahedral / Voxel control above the slider. The slider is labelled "Element size" on tet, and the tet hint and ML note say "nodes".
3. Mesh reset restores `mesh_type`. Performance reset keeps the current `mesh_type`.
4. `api.meshPreview(id, size, meshType)` and `api.mlMesh(id, meshType, targetAccuracy?)` send `mesh_type`. `MeshPreview` is now a voxel | tet union.
5. The estimate text on tet reads "About N nodes at h in (limit M)" plus a safe element size when over the limit.
6. `buildStatGroups`: when `stats.mesh_type === 'tet'` (from `result_data.mesh_type`), the fill counts are labelled "Filled lattice cells" and "Unfilled lattice cells".

## Deviations from the brief, with reasons

1. There is no `mesh_data` dict. `RiserDesigner` takes kwargs, so the supervisor fixed the contract: `voxel_volumes: Optional[np.ndarray] = None` and `mesh_type: str = "voxel"`, passed unconditionally. The ~1968 "voxel preview" is the riser-design mesh, so on tet it builds the tet mesh rather than showing an estimate.
2. `neighbors` now comes from `mesh.voxel_neighbors`. The old code read `.neighbors`, which `AdaptiveVoxelMesh` lacks, so it always passed a 6-column -1 array. Supervisor approved. `RiserDesigner` only stores `neighbors`, so voxel results do not change.
3. The desktop tet run skips the voxel memory pre-check. Without this, voxel limits would block tet runs.
4. A voxel-path `MeshGenerationError` (`core/voxel_mesh.py:1074`) now also shows its text instead of "Check console". This is a UI message change only.

## Test commands (all with QT_QPA_PLATFORM=offscreen, venv/bin/python -m unittest)

- tests.test_tet_ui: 19 OK
- tests.test_project_state: 14 OK; tests.test_web_api: 16 OK; tests.test_web_simulation_process: 17 OK
- tests.test_desktop_analysis_options: 3 OK; tests.test_simulation_worker: 3 OK; tests.test_inert_settings_hidden: 6 OK
- tests.test_architecture_boundaries: 5 OK; tests.test_desktop_orientation: 28 OK (uses project_state)
- `cd web_frontend && npm run build`: pass (tsc + vite). ESLint on the changed files: clean. `npm run check:orientation` and `check:kernel`: pass. There is no `test` script.
- `mypy --config-file mypy.ini`: no errors in `web/routes.py` or `ui/workers.py`. The 55 errors in `simulation_queue.py` are pre-existing (queue_lock typing) and none are on the changed line.

## Open risks

1. Until the post worker adds `voxel_volumes`/`mesh_type` to `RiserDesigner.__init__`, desktop riser analysis raises TypeError, which the "Analysis Error" box shows. The test uses a stub designer. The integration step must run both changes together.
2. The web client default is 'voxel' and the client always sends it. When wave 4 flips `settings.mesh.mesh_type` to tet, `performanceDefaults` must change too, or the server default must be exposed (for example via `web/routes_tools.py` /hardware, not owned here). Stored web projects with no `mesh_type` follow the server default.
3. The riser-design tet mesh covers the first CAST part only, matching the voxel path. Like the voxel mesh, it is cached until the dialog is recreated, so switching mesh type afterwards reuses the old mesh.
4. Refinement level and memory mode stay visible and editable on tet in both UIs, but the tet path ignores them.
5. Overview stats still say "Voxels" and "Voxel size" for `GridMeta` counts on tet runs. Only the fill counts were relabelled, per the brief.
