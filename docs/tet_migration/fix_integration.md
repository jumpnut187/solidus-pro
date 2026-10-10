# Integration fixes before the tet default flip

Lane: integration worker. Nothing committed or staged.

## Findings addressed (review id -> change -> test)

1. **P1-1, tet runs in the voxel recommender's CSV.** `SimulationWorker._log_mesh_training_example` skips the `log_mesh_training_example` call on the success path and on the failure path when the result mesh is tet, or when `run_mesh_type()` is tet. `run_mesh_type()` returns `performance_options["mesh_type"]`, else `settings.mesh.mesh_type`. No CSV column was added. Test: `test_tet_ui.TetRunLoggingAndProgressTests.test_tet_runs_never_feed_the_voxel_size_recommender`.
2. **P1-2, riser node counts.**
   - `RiserDesigner._covered_fraction` returns the node-volume share on tets. Both `analyze_coverage` and the `optimize_positions` acceptance test use it.
   - Hot-spot priority is distance × cluster volume on tets.
   - The voxel branch still uses counts.
   - Tests in `test_tet_post_analyses.TetBranchUnitTests`:
     - `test_riser_coverage_is_a_node_volume_share_on_tets`
     - `test_optimize_positions_moves_toward_more_volume_not_more_nodes`: the volume share moves the riser to the big node, while the count share moves it to the cluster of small nodes.
     - `test_hot_spot_priority_weights_cluster_volume_on_tets`
3. **P2-1, gas porosity cores.**
   - Call-site change (`core/solidification.py`, gas-porosity block only): on tets, a copy of the part types marks the nodes of faces with contact code CORE as code 4.
   - The call-site fix alone was not enough. `physics/gas_porosity.py` applied venting only when the last surface entry was a core. When it was, it raised a shape `ValueError`.
   - The supervisor approved option A, a mask-based venting fix. Before the fix, venting had no effect on cores placed mid-array, and the core-last case raised `ValueError: ... cannot assign 2 input values to the 4 output values`. I reproduced both on HEAD.
   - Tests:
     - `test_tet_post_analyses...test_core_venting_changes_the_tet_gas_result`
     - `test_gas_porosity...test_core_venting_applies_wherever_core_entries_sit`
     - `test_gas_porosity...test_core_last_with_other_entries_does_not_raise`
4. **P2-2, tet lattice too large.**
   - `web/payloads.write_artifact` builds the tet grids before writing the JSON.
   - A `TetLatticeTooLargeError` (a `LatticeTooLargeError` subclass) stores `summary["result_lattice_too_large"] = {dims, cell_count, spacing}`.
   - The sidecar is saved after the JSON, so its mtime is not older than the JSON's.
   - `build_field_grids` raises the error again from that record, and `_grids_or_error` returns the existing 413 `lattice_too_large` body. On tet runs the body says "coarser element size".
   - Voxel artifacts still go through `write_sidecar_for_artifact` after the JSON is written.
   - Tests in `test_tet_presentation.TetLatticeTooLargeTests` (2), covering the grid, mask and field endpoints.
5. **P2-3, web wording.**
   - `statLabels.ts`: on tet runs `shrinkage_percentage` reads "Share of volume flagged", and every other stat label and hint swaps voxel for node.
   - `CanvasOverlays.tsx` StatsChip shows "Nodes" and "Cast nodes", and labels the spacing "Lattice".
   - There is no frontend test runner. `npm run build` and eslint pass.
6. **Default flip, surrogate dataset.**
   - `generate_dataset(..., mesh_type=None)` resolves the mesh type once (None means settings) and passes it to `PipelineInputs`.
   - It records the type in each plan entry and keys on it. Tet keys end in `|mesh=tet`. Voxel keys are unchanged, so old plans still resume.
   - Tests: `test_tet_ml.GenerateDatasetMeshTypeTests` (3), including that a voxel `done` entry is not reused for a tet batch.
7. **Default flip, worker progress text.** When `mesh_type` is None and the settings default is tet, the first message is "Generating tetrahedral mesh". Test: `test_settings_default_tet_is_a_tet_run`.
8. **Owner (a), meshing progress text.** On a tet run, `_on_detailed_progress` shows "`<stage>` NN%", e.g. "Tetrahedralizing parts 40%". Voxel runs keep the Z-slice text. The run's mesh type comes from the emitting worker's `run_mesh_type()`. Training workers do not have that method, so they keep the old text. Test: `test_tet_progress_shows_stage_percent_without_z_slices`.
9. **Owner (b), completion rate.** `log_performance(..., count_unit="voxels")` takes a unit, and `_on_simulation_finished` passes "nodes" for tet meshes. Test: `test_completion_rate_counts_nodes_on_tet_runs`.
10. **Owner (d), legend palette.**
    - `CastingVisualizer._update_lookup_table` samples the palette of the result on screen, and `_show_result` rebuilds the legend after each show.
    - `set_colormap` and `set_color_scale_bounds` refresh the result before rebuilding the legend.
    - Hot Spots is now yellow to red on voxel and on tet. Shrinkage still shows the colormap.
    - Tests: `test_tet_presentation.LegendPaletteTests` (tet and voxel).
11. **Owner (c), flat yellow skin. Not reproduced.**
    - The full `MainWindow` segfaults offscreen (`QOpenGLWidget` is not supported).
    - Instead I ran a real tet run of `thick_thin_test.stl` at 0.1 in (34,654 nodes, shrinkage display max 0.578, 4,909 nodes ≥ 0.3).
    - I drove `CastingVisualizer` methods on a fresh `ResultSurfaceRenderer`, in the same order as `_on_result_type_changed` → `_update_results_visualization`: style Envelope first, then `set_colormap('red_gradient')`, `clear_result_actors`, `set_result_mesh`, `load_shrinkage(0.3)`, `update_legend`.
    - Results:
      - The envelope has 7,996 triangles, coloured (140, 30, 0).
      - The skins carry 53 distinct colours.
      - An offscreen PBR + IBL render showed no yellow pixels in Envelope, Surface or Both, on the first show or the second.
    - Possible cause, not proven: red_gradient contains no yellow, while `defect_industry` at the default iso of 0.3 is exactly (255, 255, 0). A shrinkage view painted while the colormap was still `defect_industry` (gas modes, or the Visualization dialog) would give a flat yellow envelope.

## Voxel path changes

- Gas porosity: core surface voxels are now vented. At the default 0.5 they get SGI0 = 0.5 × tf, where before they got 1.0 × tf unvented. Core-last runs no longer raise. The old `test_core_has_higher_sgi_than_mold_wall` depended on venting being skipped. It now passes venting 0.0, and the docstring says why.
- Legend bar colours for modes painted with a fixed palette now match the skin. These are the hot-spot palette modes (hot spot, stress, distortion, crack, SDAS, UTS), plus Niyama, hot tear, sand burn-in, liquid, fill time and fill speed.
- `write_artifact` imports `web.fields` before the JSON write. The voxel sidecar order is unchanged.

## Deviations

- I edited `physics/gas_porosity.py` and `tests/test_gas_porosity.py`. Neither is in my file list, and the supervisor approved both.
- `tests/test_tet_ui.py`: three preview assertions hard-coded `2·V/h³`. The geometry lane changed `estimate_tet_node_count` to `1.55·V/h³ + 3.3·A/h²`, so, per the parent's contract update:
  - the tests now call `estimate_tet_node_count` on the same box
  - the over-limit test also checks `recommended_voxel_size == ceil(1000 · tet_edge_length_for_node_limit(box, 100) / factor) / 1000`

## Commands (venv/bin/python -m unittest, QT_QPA_PLATFORM=offscreen)

| module | result |
|---|---|
| tests.test_tet_ui | 23 OK |
| tests.test_tet_post_analyses | 21 OK |
| tests.test_tet_presentation | 24 OK |
| tests.test_tet_ml | 13 OK |
| tests.test_web_fields | 80 OK (5 skipped) |
| tests.test_web_api | 16 OK |
| tests.test_result_surface | 13 OK |
| tests.test_visualizer_inspector | 10 OK |
| tests.test_simulation_worker | 3 OK |
| tests.test_gas_porosity | 15 OK |
| tests.test_surrogate | 25 OK |
| tests.test_architecture_boundaries | 5 OK |

Other checks:
- `cd web_frontend && npm run build` passes, and `npx eslint` on the two changed files is clean.
- `mypy --config-file mypy.ini` reports no errors in `ui/workers.py`.
- `compileall` passes on every changed Python file.

## Open risks

1. Pre-existing and voxel too, not fixed: `_colormap_palette` normalizes a value the renderer has already normalized, using the colour-scale bounds captured when the palette was built. Feed Modulus (legend min to max in inches) paints a distorted ramp. A shrinkage view built after a wide legend range paints at the low end until the next colormap refresh. This changes voxel colours, so it needs an owner decision.
2. `tools/generate_surrogate_data.py` (not mine) calls `generation_plan_key` without `mesh_type`. After the flip, its dry-run status table reads voxel keys for a tet batch. The generation itself is correct.
3. The web client default `performanceDefaults.mesh_type = 'voxel'` (`defaults.ts`) is still in place. The flip step owns that change.
4. Owner finding (c) is not reproduced. A GUI check with the gas-to-shrinkage colormap sequence is needed.
