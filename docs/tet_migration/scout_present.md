# Scout: presentation / persistence / ML / UI-controls seam (tet migration)

Read-only recon. All line numbers on branch `feature/tet-mesh` at time of scan.

## Headline

Every downstream consumer in this seam takes a **point cloud** (`voxel_centers (N,3)`, optional `voxel_sizes (N,)`, `voxel_part_types (N,)`) plus per-point field arrays of length N, and **re-derives a uniform lattice from the points**. Nothing here reads tet connectivity. Two consumers hard-fail on non-lattice points:
- `core/filling_stage.py:588-645` `build_grid_mapping` raises `ValueError` if centers are not on a uniform pitch (filling is then skipped, `:2454-2458`).
- `core/dense_grid.py:92-144` `derive_lattice` silently produces a wrong/huge grid for scattered tet nodes (spacing = min inter-plane gap, `:116-125`) or raises `LatticeTooLargeError` (`:137-138`).
So a tet solver must either (a) resample its nodal fields onto a uniform lattice before this seam (cheapest: keep every contract), or (b) replace `derive_lattice`/`flat_indices`/`scatter_and_fill` with point-in-tet barycentric sampling onto a chosen lattice.

## 1) Entry points and call graph

Web results path:
1. `web/services/simulation_queue.py:90-126` `build_pipeline_inputs(sim_data)` -> `PipelineInputs` (`core/simulation_pipeline.py:62`).
2. Pipeline -> `PipelineResult` with `.voxel_mesh`, `.engine`, `.visualization_data`, `.result_data`.
3. `web/payloads.py:369-437` `build_fork_artifact(result)` -> artifact dict `{summary, visualization, _grid_source}`.
4. `web/payloads.py:444-523` `_full_grid_source(engine, mesh, source_centers)` -> unstrided arrays (key `GRID_SOURCE_KEY="_grid_source"`, `:441`).
5. `web/payloads.py:539-575` `write_artifact` -> JSON + `web.fields.write_sidecar_for_artifact` (`web/fields.py:615`).
6. `web/fields.py:198-313` `build_field_grids(visualization, summary)` -> `derive_lattice` (`:225`) -> `_flat_indices` (`:227`) -> mask (`:237-238`) -> `_nearest_known_map` (`:240`) -> per mode `_scatter_and_fill` + `_quantize` (`:260-284`) -> frame stacks (`:286-311`).
7. `web/fields.py:490` `save_sidecar` (npz) / `:549` `load_sidecar` / `:729` `load_grids` (mtime cache `:665`).
8. Byte endpoints: `mask_bytes :791`, `scalar_bytes :795`, `frame_bytes :810`, `sample_field_at :896`, `build_envelope :939`, `envelope_bytes :980`.
9. Frontend: `web_frontend/src/lib/gridApi.ts:91 fetchGridMeta`, `:98 fetchMaskGrid`, `:123 fetchSkinGrid`, `:139 fetchFieldGrid`, `:157 fetchFieldGridFloat`, `:182 fetchFrames`, `:242 fetchEnvelope`; dims from header `X-Grid-Dims` (`:104,129,149,169,198`).
10. `web_frontend/src/viewport/fieldResources.ts:105-123` builds 3D textures via `fieldTextures.ts:26-63` (`createMaskVolume` NEAREST, `createFieldVolume`/`createSkinVolume` BILINEAR); `fieldPaintPlugin.ts:469-483` maps world->uvw; `CanvasOverlays.tsx:36-39` shows `meta.spacing` "in"; also consumers `ResultsViewport.tsx`, `ResultsPanel.tsx`, `modals/StatisticsModal.tsx`, `lib/viewState.ts`.

Desktop results path:
1. `ui/workers.py:105-120` and `:324` build `PipelineInputs` from dialog values.
2. `ui/visualizer.py:2529-2556` -> `ResultSurfaceRenderer.set_voxels(centers, sizes, part_types)` (`ui/result_surface.py:316-362`) -> `derive_lattice :322`, `flat_indices :328`, `voxel_of_cell :331`.
3. `ResultSurfaceRenderer.show(mode, values,...)` `ui/result_surface.py:471`; envelope `:626`; caps `:674`; `voxel_at :760-768`.
4. Picking: `ui/visualizer.py:1172` `layer.voxel_at(pick_pos)`; inspectors `:2116-2167` pass `voxel_at` and `mesh.voxel_neighbors/voxel_centers/voxel_volumes` to `core.defect_regions.extract_pocket_at` (`:2146-2155`).
5. `ui/visualizer.py:682-697` `calculate_defect_volume(voxel_centers, voxel_sizes, ...)` sums `voxel_sizes[mask]` (cubed elsewhere: check).

Filling seam: `core/filling_stage.py:4241 maybe_run_filling(voxel_mesh,...)` -> `:2390 run_filling_stage` -> `:2452-2455` centers + `dx_in = voxel_sizes.ravel()[0]` -> `build_grid_mapping` -> `dense_part_types :650`, `scatter_to_grid :672`, `gather_to_voxels :688`. Results go back to engine `_fill_arrival_times`, `_fill_frames` (`core/simulation_pipeline.py:394,418,786`; `core/solidification.py:252-258,816-825`).

ML: `core/surrogate/dataset.py:77 capture_run` -> `fields.encode_channels :149` (`:113`) and `_config_for_engine :46`; `core/surrogate/fields.py:228 unproject_grid`; `core/deep_model/dataset.py:195 DeepDataset` consumes records.

## 2) Data consumed / produced

Mesh attributes read (length N = n_voxels):
| name | shape/dtype | units | read at |
|---|---|---|---|
| `voxel_centers` | (N,3) float | in | payloads.py:375, result_surface.py:319, filling_stage.py:2452, surrogate/fields.py:170, surrogate/dataset.py:49,117, visualizer.py:2543 |
| `voxel_sizes` | (N,) float, edge length | in | payloads.py:456, visualizer.py:2547, filling_stage.py:2453 (`[0]` only) |
| `voxel_part_types` | (N,) int32 codes (0 cast, 2 chill, ...) | - | payloads.py:378,400,457, fields.py:228-232, filling_stage.py:2460, surrogate/fields.py:179 |
| `voxel_moduli` | (N,) float32 | in | surrogate/fields.py:189 |
| `voxel_neighbors`, `voxel_volumes` | voxel adjacency / volume | - / in^3 | visualizer.py:2149-2151 |

Engine attributes read (all (N,), cast to float32, must be per-mesh-point; `_full_array` drops any with wrong length, payloads.py:526-531): `local_modulus, hot_tear_risk, sand_burn_in_risk, feeding_quality, niyama_values, niyama_risk, combined_feed_risk, effective_stress, distortion_magnitude, crack_susceptibility, sdas, predicted_uts, porosity_probability, feeding_blocked, feeding_path_lengths, feeding_open_fraction` (payloads.py:381-398) plus `shrinkage_probability, hotspot_intensity, solidification_times, gas_generation, gas_entrapment, gas_porosity_risk, _fill_arrival_times` (payloads.py:461-480). Time series: `fraction_liquid_history` list of (N,), `fraction_liquid_times` list[float s], `_mask_non_cast_fraction_liquid` callable (payloads.py:497-504); `_fill_frames` of `FillFrame(time_s, fill_fraction (N,), speed_m_s (N,))` (payloads.py:506-517, filling_stage.py:2064-2080). Surrogate also uses `engine.surface_distances` (N,) and `engine._get_normalized_heights()` (surrogate/fields.py:183-194).

Strided JSON `visualization` (`centers`, `part_types`, fields as lists) is subsampled by `max_voxels_for_visualization` via `_matching_indices` (payloads.py:376, 379, 415); sidecar uses unstrided `_grid_source` (payloads.py:425-436).

Sidecar npz (`web/fields.py:490-547`): `version int32`, `dims int32[3]` (nx,ny,nz), `origin float64[3]` (center of cell 0, in), `spacing float64[1]` (in), counts int64, `total_volume float64`, `mask uint8[nx*ny*nz]` (part code+1, 0 = outside, fields.py:234-238), per-mode `uint8` quantized values + `min/max/mean` float64, `stack__*` uint8 (F, cells) + `stack_times__*` float32 (s). Flat order x-fastest: `index = ix + nx*(iy + ny*iz)` (dense_grid.py:15, 79). Metadata JSON `build_grid_metadata` fields.py:405-455 (`dims, origin, spacing, quantization{uint8, levels}`). Envelope bytes: positions/normals `<f4`, indices `<u4`, optional scalars `<f4` (fields.py:980-1001).

Surrogate grid: `GridMeta(resolution, bbox, cell_size, max_dim)` (surrogate/fields.py:102-123); channels `(C,R,R,R)` float32 (`:201`), floor-binned average `_scatter :131-146`; `unproject_grid :228-255` returns (C,N)/(N,). Conditioning config includes `voxel_size` normalized to range (0.02, 0.5) (`core/surrogate/conditioning.py:46`; `dataset.py:67`).

## 3) Voxel-specific assumptions

- Uniform pitch: `dense_grid.py:111-114` spacing = median of `sizes`; `fields.py:18-19` doc "Every voxel is base_voxel_size"; `filling_stage.py:2453` takes `voxel_sizes[0]` as pitch; `filling_stage.py:612-625` residual check raises if not on lattice.
- Bijection centers<->cells: `filling_stage.py:26-29, 626-640` raise on two voxels per cell; `result_surface.py:331` `voxel_of_cell[cells] = arange(N)` (collision silently overwrites).
- Dense lattice from extent: `dense_grid.py:127-135`; cap `MAX_LATTICE_CELLS` `:137`.
- ijk via `rint((p-origin)/spacing)`: `dense_grid.py:85, 151-156`; `filling_stage.py:617`; floor variant in `surrogate/fields.py:138, 250`.
- Cell-center convention, uvw = `((p-origin)/spacing+0.5)/dims`: `fields.py:29-31`, `fieldPaintPlugin.ts:474-483`.
- Scalar `spacing` everywhere (cubic cells): `fieldPaintPlugin.ts:75,470`, `CanvasOverlays.tsx:36-39`, `fields.py:925-932` (`sample_field_at` divides by one spacing).
- Value-at-point = cell value (piecewise constant), holes filled by nearest known cell EDT: `dense_grid.py:158-215`.
- Neighbor graph: `visualizer.py:2149` `mesh.voxel_neighbors` -> `core/defect_regions.extract_pocket_at` (face-neighbor flood fill; slot semantics live in `core/defect_regions.py`, outside this seam; not inspected).
- Defect volume = sum of voxel volumes/sizes: `visualizer.py:682-697`, `voxel_volumes` `:2151`.
- Filling solver itself is a dense lattice LBM (`filling_stage.py:2-13`, physics.filling); it never needs tets but needs a lattice + `dx`.
- Upload tessellation tolerance tied to voxel size: `web/payloads.py:132-147, 208-213`.

## 4) What a tet/node-based replacement must provide

Minimum to keep this whole seam unchanged (recommended phase 1):
1. A "presentation point set" object exposing `voxel_centers (N,3) in`, `voxel_sizes (N,) in` uniform, `voxel_part_types (N,) int32`, `voxel_volumes`, `voxel_neighbors`, `voxel_moduli` - produced by sampling the tet solution onto a uniform lattice (barycentric interpolation of P1 nodal values at cell centers inside the tet mesh; part type from containing tet's region tag).
2. Every engine field listed in section 2 as length-N arrays on that lattice, same units; time histories resampled per frame.
3. Filling: either keep the voxel mesh for the fill stage (it already lives on a lattice) and transfer `fill_time_s`/`temperature_c` onto tet nodes for the heat solve, or build `GridMapping` from a rasterized lattice of the tet mesh directly (`build_grid_mapping(centers, dx_in)` only needs lattice-aligned centers).
4. If phase 2 drops the lattice: replace `derive_lattice`+`flat_indices`+`scatter_and_fill` (`dense_grid.py:92-215`) with `tet_to_lattice(nodes (M,3), tets (K,4) int, node_values (M,), lattice) -> dense float32[cells]`; the sidecar/frontend contract (dims/origin/spacing/uint8) can stay. `voxel_at` (`result_surface.py:760`) must then return a node or tet index, and `extract_pocket_at` needs a tet/node adjacency.
5. Surrogate/deep model: `_scatter`/`unproject_grid` work on any point cloud, but averaging nonuniform tet nodes biases cells; datasets captured at voxel resolution are not comparable to tet captures. Add mesh type to conditioning/record metadata or freeze ML on lattice-resampled fields.

## 5) Toggle insertion points

Config:
- `config/settings.py:55-64` `MeshSettings`: add `mesh_type: str = "voxel"` (`"voxel"|"tet"`) and tet params (e.g. `tet_edge_length_rel`, `tet_epsilon`). Validate in `__post_init__ :66-71`.
- Env override map `config/settings.py:402-405`: add `f"{env_prefix}MESH_TYPE": ("mesh", "mesh_type", str)`.
- Serialized defaults: YAML `config/settings.py:671-675`, TOML `:734-738`; `asdict` at `:511`.

Pipeline input:
- `core/simulation_pipeline.py:62-70` `PipelineInputs.__init__`: add `mesh_type: str = "voxel"` kwarg.

Web backend:
- `web/services/simulation_queue.py:110-125`: add `mesh_type=str(performance.get("mesh_type", "voxel"))`. Store in `performance_options` JSON (no SQL migration; `web/repository.py:104-119` has fixed columns `voxel_size, refinement_level, memory_mode`; adding a column would need the `ALTER TABLE` helper at `:156-162` and edits at `:194-207, 272-284, 363-376`).
- `web/routes.py:227-257` `mesh_preview` and `:259-289` `ml_mesh` assume voxel counts (`estimate_voxel_count`); need a tet branch or a mode guard. `run_project :291-303` needs no change (passes stored project).

Web frontend:
- `web_frontend/src/lib/defaults.ts:30-38` `performanceDefaults`: add `mesh_type: 'voxel'`.
- `web_frontend/src/components/panels/PerformancePanel.tsx:~59` (next to `use_chunked_voxelization` checkbox) or `components/panels/MeshPanel.tsx:40-58` (beside voxel size slider; better UX home): add select. Type is open `Record` (`lib/api.ts:98`), so no type change required.

Desktop:
- `ui/main_window.py:381-480` `MeshDialog`: add combo after `voxel_size_spin :397-402`; include in `store/restore :443-452`, getters `:460-478`.
- Or `PerformanceDialog :486-672` near `octree_check :596`, `get_options :655-662`, `set_options :665-669`, default dict `:2531`.
- `ui/workers.py:105-120` and `:324`: pass `mesh_type` into `PipelineInputs`.
- Persistence: `ui/project_state.py:40-98` `build_project_state` writes `performance_options` (`:48`) and `simulation.voxel_size` (`:83`); `load_project_state :101-180` reads back (`:164`). Putting `mesh_type` in `performance_options` round-trips for free. Desktop project files store inputs only; no mesh/results persisted (no results keys found in project_state.py).

## 6) Risks and pinning tests

Risks:
- `derive_lattice` on tet nodes: spacing collapses to tiny min-gap -> `LatticeTooLargeError` or OOM (`dense_grid.py:116-138`).
- `_full_array` silently drops fields whose length != len(centers) (`payloads.py:526-531`, `:404-405`): a mismatch between node count and center count yields empty results, no error.
- `voxel_of_cell` overwrite on collisions (`result_surface.py:331`) gives wrong picks silently.
- Filling skipped silently-ish (`filling_stage.py:2456-2458` emits message only) if centers are off-lattice.
- `_fill_arrival_times` and `fraction_liquid_history` must be on the same point set as `voxel_centers`.
- Surrogate conditioning `voxel_size` range (conditioning.py:46) meaningless for tet edge length; mixed datasets corrupt training.
- Frontend assumes cubic cells and cell-center origin; any anisotropic spacing breaks `fieldPaintPlugin.ts:469-483`.
- Ambiguity: `defect_regions.extract_pocket_at` neighbor slot semantics not inspected here (other seam).

Tests that pin behavior:
- `tests/test_web_fields.py`: `test_derives_dims_origin_and_spacing_from_centers :145`, `test_dims_survive_an_entirely_empty_lattice_plane :158`, `test_mask_holds_part_type_code_plus_one_and_zero_outside :179`, `test_scalar_grid_reproduces_occupied_values :198`, `test_holes_take_the_nearest_inside_value_and_never_bleed_to_zero :214`, `test_frames_are_quantized_bytes_with_their_times :239`, `test_metadata_matches_the_grids :254`, `test_round_trip_preserves_every_grid :302`, `test_write_artifact_also_writes_the_sidecar :325`, endpoint tests `:408-533`, fill stack/envelope tests `:559-642`.
- `tests/test_result_surface.py`: `test_lattice_roundtrip_and_fill :47`, `test_voxel_at_maps_world_positions_back :170`, `test_valid_mask_removes_voxels_from_paint_and_envelope :144`, `test_section_cap_is_cut_and_painted :159`, `test_wrong_length_values_are_refused :185`.
- `tests/test_filling_stage.py`, `tests/test_filling_stability.py` (use `build_grid_mapping`).
- `tests/test_web_api.py`, `tests/test_web_simulation_process.py` (`build_pipeline_inputs` / `build_fork_artifact`).
- `tests/test_project_state.py` (desktop persistence round-trip).
- `tests/test_surrogate.py`, `test_surrogate_chill_channel.py`, `test_surrogate_geometries.py`, `test_surrogate_split.py`, `tests/test_deep_model.py`.
- `tests/test_inert_settings_hidden.py` (likely asserts which settings are exposed; a new toggle may need registering).
- `tests/test_centered_riser_symmetry.py` references `derive_lattice`/mapping.

## Start here

`web/payloads.py:444-523` `_full_grid_source`: it is the single list of every per-point array the presentation layer needs; a tet engine that fills these at lattice resolution keeps sidecar, frontend, and desktop renderer unchanged.
