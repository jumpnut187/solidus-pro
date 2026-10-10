# Voxel default, tet opt-in with automatic voxel fallback. Branch `feature/tet-mesh`, base c96b061, nothing staged.

## Changes

1. `config/settings.py`
   - `mesh_type` defaults to `"voxel"`. An unknown value falls back to `"voxel"` with a RuntimeWarning. `tet_edge_length_factor` stays 2.0.
   - New `MeshSettings` fields: `tet_auto_fallback` (True), `tet_mesh_timeout_s` (600.0), `tet_auto_max_nodes` (300,000), `voxel_cut_cells`
     (False). `normalize_tet_fields` validates them: a non-positive or non-numeric value restores the default, and a bool accepts
     bool, 0/1 or yes/no/true/false/on/off strings. Each invalid value warns.
   - Env overrides: `CASTSIM_TET_AUTO_FALLBACK`, `CASTSIM_TET_MESH_TIMEOUT_S`, `CASTSIM_TET_AUTO_MAX_NODES`, `CASTSIM_VOXEL_CUT_CELLS`.
     The YAML and TOML templates carry all four fields and `mesh_type: voxel`.
2. `core/tet_mesh.py` (wall-time limit only)
   - `build_tet_mesh(..., timeout_s=None)`. The deadline covers all fTetWild attempts together, including the wait for
     `_NATIVE_MESH_LOCK`. `_run_child` checks it on every 0.15 s poll and stops the child (terminate, then kill after 1 s).
   - New `TetMeshTimeoutError(MeshGenerationError)` has a `.timeout_s` attribute. A timeout does not retry the remaining attempts.
   - Labelling and node arrays after fTetWild are not covered (13 + 11 s on C05813).
3. `core/simulation_pipeline.py`. The fallback lives in `prepare_mesh_and_geometry`, so `run_pipeline` and `core/surrogate/pipeline.py` both
   get it, and the surrogate file is unchanged.
   - The fallback runs only when the resolved type is tet and `tet_auto_fallback` is on. It has three triggers:
     1. Size gate: `estimate_tet_node_count` > `tet_auto_max_nodes`. fTetWild is not started, and the reason gives the mesh size that
        stays on tet.
     2. `MeshGenerationError` from `_build_tet_simulation_mesh`: every fTetWild attempt failed, or the `tet_max_nodes` guard fired.
     3. `TetMeshTimeoutError` (`timeout_s=tet_mesh_timeout_s`).
   - The voxel mesh is then built by the unchanged voxel code: same `inputs.voxel_size`, and `AdaptiveVoxelMesh.generate` does its own
     memory validation and auto-adjust.
   - The reason is stored on the mesh (`mesh_fallback_reason`) and read by `mesh_fallback_reason_of`. `PipelineResult` gains
     `mesh_fallback_reason` and `requested_mesh_type`. On a fallback only, `result_data` and `visualization_data` get both keys.
   - The pipeline logs a warning to `core.error_logger` with context (`trigger`: size, error or timeout) and emits a progress message that
     starts with `MESH_FALLBACK_MESSAGE_PREFIX`.
   - With `tet_auto_fallback` off, errors raise as before and no timeout applies. Cancel raises `SimulationCancelledError`, which is not a
     `MeshGenerationError`, so it propagates.
4. `ui/workers.py`
   - `SimulationWorker` and `SurrogateWorker` route pipeline messages through `_on_pipeline_message`, which records `mesh_fallback_reason`
     from the prefixed message.
   - After a fallback, `run_mesh_type()` returns "voxel", so detailed progress shows Z-slices again.
5. `ui/main_window.py`
   - `MeshDialog` lists Voxel first and follows settings, and an unknown value selects Voxel. Its tooltip describes the fallback.
   - During the run, `_on_simulation_progress` adds "⚠ Running on the voxel mesh: <reason>" under every later status message.
   - After the run, `_show_mesh_fallback_notice` shows `QMessageBox.information`, the same mechanism the stalled-fill notice uses, and puts
     a status line above the filling status. It runs for simulation runs and surrogate runs. The notice text comes from the Qt-free
     `format_mesh_fallback_notice`.
6. Web
   - `web/routes.py::tet_mesh_estimate`: with auto fallback on, the limit is `min(tet_auto_max_nodes, tet_max_nodes)`. A new
     `falls_back_to_voxel` field is added, and the message says the run will use the voxel mesh.
   - `web/services/simulation_queue.py`: the fallback message sets progress to 12 %. The final status reads "Simulation complete (ran on
     the voxel mesh: tet mesh fallback)".
   - `web/payloads.py` and `web/fields.py` were not edited: the summary is `result_data`, and `_flatten_stats` carries the string into
     `GridMeta.stats`.
   - Frontend: `performanceDefaults.mesh_type = 'voxel'`, and `meshTypeOf` defaults to 'voxel'. MeshPanel lists Voxel first and adds a
     fallback hint on tet.
   - ResultsPanel shows a `status-note-warning` banner when `meta.stats.mesh_fallback_reason` is set. The statistics list skips that
     key. The estimate text names the fallback, and `TetMeshPreview.falls_back_to_voxel` is typed.
7. `.gitignore` adds `__tracked_surface.stl`. README, QUICKSTART, AGENT.md and docs/ARCHITECTURE.md now say voxel is the default, tet is
   opt-in and fallback is automatic, and they list the new settings.

## Decisions and evidence

- Size gate on the real model (`/tmp/fbcheck/fallback_gate.py`, C05813, 17 parts, factor 2.0). The estimate takes 0.29 s cold.

| voxel size (in) | edge (in) | estimate | outcome at 300k |
|---|---|---|---|
| 0.10 | 0.20 | 537,111 | fallback. The reason says "at least 0.126 in stays on tet" |
| 0.15 | 0.30 | 188,673 | tet |
| 0.20 | 0.40 | 92,054 | tet |
| 0.30 | 0.60 | 34,658 | tet |

  At the default 0.1 in, the owner's model skips the 15-37 min mesh. At 0.15 in it stays on tet. The 600 s timeout covers runs that pass
  the gate but mesh slowly, because the soup bbox fill is not in the estimate.
- The timeout applies only when auto fallback is on. "Off" keeps today's behaviour exactly.
- The reason is carried on the mesh object, not added to the 6-tuple returned by `prepare_mesh_and_geometry`. This keeps the surrogate
  pipeline untouched and gives it the same fallback.
- `result_data` and the payload carry `requested_mesh_type` and `mesh_fallback_reason` only on a fallback. Otherwise every run would add a
  "Requested mesh type: voxel" row to the web statistics. `PipelineResult` always has both attributes.
- `_tetrahedralize_in_child` takes `deadline` as an optional positional argument. `tests/test_tet_mesh.py` patches it with
  `side_effect(mode, *args)`, and a keyword argument would break that test.

## Deviations

1. The desktop pre-run voxel memory dialog (`validate_mesh_parameters` with the Yes/No prompt) still does not run for a tet run. A fallback
   relies on the checks inside `AdaptiveVoxelMesh.generate`: it auto-adjusts the voxel size, or blocks with `SimulationError`. A prompt
   cannot be shown from the worker thread mid-run.

## Tests

- New `tests/test_tet_fallback.py`, 19 tests:
  - size gate (fTetWild never called; reason, payload keys, error-log context `size`)
  - voxel fallback uses the run's voxel size
  - `tet_max_nodes` guard falls back
  - the surrogate pipeline falls back
  - all attempts crash (`CASTSIM_TETWILD_CHILD_FAULT=csg,soup`) → fallback with SIGSEGV in the reason
  - `tet_mesh_timeout_s=0.01` → fallback, and the lock is released
  - with `tet_auto_fallback=False`: a crash raises, the guard raises, and the auto limit and timeout are not applied (the run stays tet)
  - cancel during tet meshing raises `SimulationCancelledError`, kills the child and never calls voxel `generate`
  - `build_tet_mesh(timeout_s=1)` kills the child in under 4 s with no retry, and an invalid `timeout_s` raises
  - settings defaults, env overrides, invalid values, string bools, templates
  - desktop worker message parsing, status-line persistence and the after-run notice
- Updated: `tests/test_tet_pipeline.py` (voxel defaults, voxel for unknown values, guard test pins `tet_auto_fallback=False`) and
  `tests/test_tet_ui.py` (preview limit = min(auto, max), two new preview tests, dialog selects voxel for an unknown value).
- Commands:
  - `venv/bin/python -m unittest tests.test_tet_fallback`: 19 OK (8.6 s). `tests.test_tet_end_to_end`: 11 OK.
  - `venv/bin/python -m unittest tests.test_tet_pipeline tests.test_tet_ui tests.test_web_api tests.test_project_state
    tests.test_simulation_worker tests.test_inert_settings_hidden tests.test_architecture_boundaries`: 119 OK.
  - `venv/bin/python -m unittest tests.test_tet_mesh tests.test_web_simulation_process tests.test_surrogate tests.test_tet_presentation
    tests.test_tet_integration tests.test_tet_ml`: 127 OK.
  - `venv/bin/python -m unittest discover tests`: 2105 run, 16 skipped, 3 failures. All 3 are the DESIGN.md baselines (filling_stability
    deeper tower, porosity riser_post_block, shrinkage_display_intensity). The run includes the cut-cell worker's in-progress edits.
  - `cd web_frontend && npm run build`: passes.
  - `python -m mypy --config-file mypy.ini`: no new errors in my files. The remaining ones in `simulation_pipeline.py` and
    `simulation_queue.py` pre-date this change.

## Open risks

1. No end-to-end fallback run was done on C05813. The gate was measured on it, but the voxel solve at 0.1 in on 1,750 in^3 was not run.
2. Post-fTetWild work (labelling and node arrays, about 24 s on C05813) runs after the timeout check, and a tet mesh just under the gate
   can still take a long time in the solve.
3. `tet_auto_max_nodes` is checked against the upper-bound estimate (measured meshes are 53-89 % of it). Some meshes that would have
   finished are skipped. That is the intended bias.
4. Surrogate dataset generation (`core/surrogate/training.py`) keys plans by the requested mesh type. A record from a fallback run stores
   the mesh's real type (voxel), but the plan key still says tet.
