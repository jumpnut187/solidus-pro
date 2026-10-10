# Integration review: tet mesh migration, wave 3 (3f67c76 on cb655bc and 20e098d)

There are no P0 findings. I found two P1 issues and three P2 issues. Nothing was edited and no tests were run.

## Scope and limits
- I read the code at HEAD. `watchdog_diff` shows no tracked working-tree changes. I could not diff the three commits, so "voxel path unchanged" means the voxel branch next to each tet branch reads as untouched, not that I compared it line by line against 90a4a16.
- I covered every file named in the brief, plus the callers needed to trace the default flip.

## Verified correct
- **Dispatch:** every tet branch checks `getattr(mesh, "mesh_type", "voxel") == "tet"`. No code detects tets from the neighbour-table width or from `hasattr(tets)`. `LatticeView` ("lattice") correctly falls through to the voxel branches.
- **Voxel branches left as they were:**
  - Graph distances still use breadth-first hop counting (solidification.py:1153-1169).
  - The orphan "8 of 26" rule (:1712), isolation kernel (:4227), path strain (:4791), surface restraint (:4581) and feeder-neck padding (:2725) are unchanged.
  - Stress and microstructure averages (stress_analysis.py:499, microstructure.py:476) are unchanged.
  - The riser move stencil (riser_design.py:868-882), the incremental buffer sized by `shape[1]` (same value, 26, on voxels) and `extract_pocket_at` with `connectivity` are unchanged.
- **Units:** tet graph distances are inches divided by `mean_edge_length`, giving hops. Per-node length scales come from `voxel_sizes` in inches. Nothing in these files introduces a bare 0.0254.
- **Volume weighting:**
  - Tet `fed_fraction`, average stress, high-stress fraction, average SDAS and UTS, shrinkage % and isolation all weight by node volume.
  - The denominators include every node, which matches the voxel denominators (every voxel).
  - `export_results` and `_add_summary_statistics` both call `get_shrinkage_percentage()`.
- **int8 part types:** every consumer casts to int32 before comparing or passing to a kernel. I found no `bincount` on part types. JSON output goes through `int32.tolist()` (payloads.py:379), and `part_type_counts` keys are wrapped in `int()`.
- **Old voxel web artifacts:**
  - Without `visualization.mesh_type`, `build_field_grids` takes the `derive_lattice` path (fields.py:235).
  - The sidecar `mesh_type` key is written only for tet runs (fields.py:650), and `load_sidecar` defaults it to "voxel" (:737).
- **Desktop picking:** `voxel_at` maps cell → inside cell → `nearest_node` (result_surface.py:845). The display mask and flags follow `nearest_node`, and the shrink inspector passes `mesh_type`.
- **ML index:**
  - Rows appended to a legacy 6-column index carry a trailing field, which `dataset_summary` reads back (dataset.py:216-219).
  - Records without `mesh_type` load as "voxel".
- **Persistence:**
  - Desktop project files keep `mesh_type` inside `performance_options`, and the loader drops unknown values (project_state.py:183-197).
  - The web client merges `performanceDefaults` on project load and on local-storage load (App.tsx:323, defaults.ts:154).

## Findings

**P1-1. Tet runs are logged into the voxel-size recommender's training data.**
- **Where:** `ui/workers.py:154-170` (success) and `:193-207` (failure) call `log_mesh_training_example` for every run, with no mesh type.
- **Failure:**
  - A desktop tet run appends a row to `data/mesh_runs.csv` with `voxel_count` = node count (about 1.5× the voxel count at the same h), `refinement_level` set to a value tet ignores, and tet wall times. Nothing marks the row as tet.
  - `tools/train_mesh_model.py:107-128` fits `base_voxel_size` and `refinement_level` on these rows and weights them by `voxel_count`, so the voxel recommender learns from tet runs.
  - Once the default flips, almost every new row is tet.
- **Fix:** skip both calls when `getattr(self.voxel_mesh, "mesh_type", None) == "tet"` or `performance_options.get("mesh_type") == "tet"`. Do not add a column: `DictWriter` writes values by position under the existing header, so a new column would misalign.

**P1-2. Riser coverage and hot-spot priority count nodes, which breaks contract C6 ("riser design uses node volumes").**
- **Where:**
  - `core/riser_design.py:778`: `covered_fraction = len(covered)/n_voxels`.
  - `:840`: the step-acceptance test in `optimize_positions` uses the same count.
  - `:405`: `priority = distance × len(cluster_indices)`.
- **Failure:**
  - Tet node volumes vary widely. Wave-3 measured shrinkage share at 7.4 % by node count against 4.4 % by volume on the same run.
  - The dialog's "Coverage: x % of casting fed" (main_window.py:1996-2000) is therefore a node share. Riser moves are accepted when they cover more nodes, which favours finely meshed regions over larger volume.
  - Only `HotSpot.volume` was ported.
- **Fix:** on tet runs with `voxel_volumes` set, replace each count with `voxel_volumes[idx].sum()`, divided by the total volume where it is a fraction. Leave the voxel branch as it is.

**P2-1. Gas porosity loses the core source type on tets.**
- **Where:** `physics/gas_porosity.py:139-148` classifies each surface point by its own part code. The call site is `core/solidification.py:5382-5390`.
- **Failure:**
  - CORE tets are removed, so no node carries code 4.
  - Metal nodes on core-contact faces are classified `mold_wall` (weight 0.6) instead of `core` (1.0 × (1 − venting) = 0.5).
  - `core_venting_effectiveness` therefore has no effect on any tet run. Wave-3 said this file "needed no tet branch".
- **Fix:** on tets, before calling, mark the surface nodes of faces whose `boundary_face_contact_codes == PART_TYPE_CORE` as code 4 in a copy of the part-type array, as `get_material_surface_voxel_indices` already does.

**P2-2. A tet sidecar that fails to build turns into a permanent "re-run it" 404.**
- **Where:** `web/fields.py:791-802` swallows every exception. The fallback then raises `TetGridUnavailableError` (:235-238), which `routes_fields.py:427-439` reports as 404 `tet_grid_missing` with the text "Re-run it".
- **Failure:** a tet run whose sampler lattice exceeds `MAX_LATTICE_CELLS` (a thin part with a large bounding box) gets that 404 on every request. Re-running reproduces it. The voxel path returns 413 `lattice_too_large` with a coarser-size hint for the same geometry.
- **Fix:** in `write_artifact`, catch `LatticeTooLargeError` from the tet sidecar build and store `dims`, `cell_count` and `spacing` in `summary` before writing the JSON. Have `_grids_or_error` return the existing 413 body when that key is present.

**P2-3. The web stats panel labels the tet shrinkage share as a voxel count.**
- **Where:** `web_frontend/src/lib/statLabels.ts:76` labels `shrinkage_percentage` "Share of voxels flagged". On tet runs the value is a volume share (simulation_pipeline.py:584-586), and `:72` says "Voxels flagged".
- **Failure:** a tet run shows 4.4 % labelled as a voxel share, while the node share is 7.4 %.
- **Fix:** when `stats.mesh_type === 'tet'`, relabel it "Share of volume flagged", with matching "nodes" wording, in `buildStatGroups`.

## What changes when `settings.mesh.mesh_type` flips to "tet"
These callers pass `mesh_type=None` or omit it, so they switch to tet without any warning:
1. **Production code:**
   - `core/surrogate/training.py:266`: surrogate dataset generation. Its resume keys (:276) have no mesh type, so a resumed plan mixes the voxel records already marked done with new tet records.
   - `web/services/simulation_queue.py:127`: any API client that omits `mesh_type`.
   - `web/routes.py:234-236`: `mesh-preview` and `ml-mesh` called without `mesh_type`.
2. **Desktop:**
   - `ui/main_window.py:459`: the `MeshDialog` default.
   - `:5338-5340`: legacy projects with no `mesh_type` re-run as tet on desktop, while the web client opens the same project as voxel (App.tsx:323). The two UIs diverge.
   - `ui/workers.py:108`: the progress text still says "voxel" when `mesh_type` is None.
3. **Node limit:** existing projects near the voxel memory limit now hit `tet_max_nodes` (about 2 nodes per h³) and fail with `MeshGenerationError`.
4. **Tests with no `mesh_type`:**
   - `tests/test_initial_temperature_seam.py:34`: its `core.voxel_mesh` patches become no-ops.
   - `test_simulation_scenarios.py:54`, `test_centered_riser_symmetry.py:35` (exact symmetry against fTetWild's run-to-run jitter), `test_fraction_liquid_timing_consistency.py:62`.
   - `test_surrogate.py:486/514/543`, `test_surrogate_geometries.py:84`, `test_deep_model.py:529`, `tests/verify_deep_benchmarks.py:60`. These compare a tet reference against surrogates trained on voxel data.
   - The untracked `Test/session_diag_*` drivers.

Remaining risk: the riser dialog, gas porosity and the lattice-too-large web path have no tet test. Suggested commands for the supervisor: `venv/bin/python -m unittest tests.test_tet_post_analyses tests.test_tet_presentation tests.test_tet_ui tests.test_web_api`, and the flip check `CASTSIM_MESH_TYPE=tet venv/bin/python -m unittest discover tests`.

Merge verdict: OK with notes