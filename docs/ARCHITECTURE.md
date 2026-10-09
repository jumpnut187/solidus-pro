# Architecture

What actually runs, what the layers guarantee, and where a new physics stage
plugs in. Written for someone about to add a module — currently, mold filling.

`AGENT.md` is the companion to this file and covers entrypoints, verify
commands and gotchas.

## The live path

There is exactly one simulation path. Both UIs funnel through it.

```
main.py (desktop)                    run_web.py -> web/app_factory.py
  -> ui/workers.py                     -> web/routes.py
                                       -> web/services/simulation_queue.py
                    \                 /
                     v               v
        core/simulation_pipeline.py :: run_pipeline(PipelineInputs)
                            |
    prepare_mesh_and_geometry()  ->  core/voxel_mesh.py  AdaptiveVoxelMesh
    create_analysis_options()    ->  AnalysisOptions
    SolidificationEngine(...)    ->  core/solidification.py
        .initialize()
        .run_simulation()
             -> core/solidification_loop.py  run_time_integration
                  -> physics/heat_transfer.py         HeatTransferModel
                  -> physics/solidification_physics.py SolidificationModel
             -> core/analysis_strategies.py  run_post_analyses
        .export_results()
    build_visualization_payload()
                            |
                     PipelineResult
```

`run_pipeline` has three production callers: `ui/workers.py`,
`web/services/simulation_queue.py` and `core/surrogate/training.py`.
`web/routes.py` no longer calls it -- that duplicate worker path was removed in
4ee4509, and the web mapping now lives in
`web/services/simulation_queue.py::build_pipeline_inputs`.

`core/surrogate/pipeline.py` is a **sibling** pipeline: it reuses
`prepare_mesh_and_geometry` and `run_post_analyses` but substitutes its own
solve stage. It is the pattern to copy for a new stage — do not add physics
inside `SolidificationEngine`, which is already 5,400 lines and one class.

## Layer contracts

| Layer | May import | Must not import | Holds |
|---|---|---|---|
| `physics/` | stdlib, numpy, scipy, numba | `core/`, `config/`, `ui/`, `web/` | Mesh-agnostic solvers |
| `core/` | `physics/`, `config/` | `ui/`, `web/` | Mesh, part types, orchestration |
| `config/` | stdlib | everything else | Settings, hardware, exceptions |
| `ui/`, `web/` | `core/`, `config/` | each other | Presentation |

`tests/test_architecture_boundaries.py` enforces these by parsing the import
graph. It also asserts `physics/` stays on the live path and that the deleted
duplicate thermal modules do not come back.

**`physics/` being mesh-agnostic is the important one.** `HeatTransferModel`
takes `(n_voxels, voxel_size, neighbors, alloy)` — plain arrays and scalars.
That is why a filling solver can reuse it directly.

## Units: three zones

Fully documented in `physics/units.py`. Summary:

| Zone | Unit | Where |
|---|---|---|
| Geometry, mesh, UI, web | **inches** | STL input, `voxel_centers`, `voxel_sizes`, moduli, feeding distances |
| `physics/heat_transfer.py` | **metres** | diffusivity m²/s, HTC W/(m²·K) |
| Niyama output | **mm** | °C·√s/mm |

Material properties in `core/materials.py` are SI regardless (Pa·s, kg/m³,
gravity 9.81 m/s²). Use `physics.units`; never add another bare `0.0254`.

**Write new solvers fully in SI internally**, converting only at the boundary.

## Voxel data

`AdaptiveVoxelMesh` holds geometry *and* live physics state:

- `voxel_centers` (N,3, inches), `voxel_sizes`, `voxel_volumes`, `voxel_moduli`
- `voxel_neighbors` (N,26) int32, `-1` for absent. **Slot layout is documented at
  `core/voxel_mesh.py:1228`: slots 0–5 are the ±X/±Y/±Z faces**, 6–17 edges,
  18–25 corners. A face-flux solver can use slots 0–5 directly and gets signed
  directions for free; slots 4/5 (±Z) give gravity alignment.
- `voxel_states` — fraction solid. Physics state living on the mesh object.
- `voxel_part_types` — int32 codes, see below.
- `surface_voxels` — bool mask.

### Part types

Codes are `PART_TYPE_CAST=0, RISER=1, CHILL=2, RISER_POST=3, CORE=4`, then the
gating system in flow order: `POUR_CUP=5, SPRUE=6, RUNNER=7, FILTER=8, GATE=9`.
Defined in `core/part_manager.py`. They are persisted in saved projects and
encoded into ML training records, so they are **not free to renumber**.

Use the predicates rather than integer comparisons:

```python
from core.part_manager import is_metal, is_feeder, is_mold, is_cast, is_chill
```

Adding a part type means adding it to `METAL_PART_CODES` / `MOLD_PART_CODES`
once, instead of auditing every comparison site. `tests/test_part_type_predicates.py`
fails if a new type belongs to neither group.

Two groupings exist beyond that binary, because two distinctions cut across it:

- `SCORED_METAL_PART_CODES` vs `GATING_METAL_PART_CODES` -- a sprue holds metal
  and solidifies, so it is metal, but it is not the casting and must never be
  scored for shrinkage.
- `FLOW_RESTRICTOR_PART_CODES` -- a ceramic FILTER groups with the mold
  (refractory, never scored) but metal flows *through* it. Building an obstacle
  mask from `is_mold()` alone would make it an impermeable wall and stop the
  fill dead at the gate.

**A new code produces no voxels until `AdaptiveVoxelMesh.assign_part_types`
knows about it.** That function used to bucket parts with three literal
comparisons, so CORE silently produced zero voxels for its entire existence. It
is generic now, but `tests/test_part_assignment_generality.py` asserts every
code claims voxels, because the failure mode is a part that vanishes with no
error.

**Adding a part type also changes the ML encoding.** `core/surrogate/fields.py`
encodes channel 1 as `code / PART_TYPE_DIVISOR`, and that divisor is recorded in
`ModelManifest.part_type_divisor`. `load_model` refuses artifacts whose recorded
divisor differs from the current one, so bumping it forces a retrain instead of
silently degrading predictions.

## The mold-filling stage

Built, and the pattern to copy for any future pre-solidification stage.

**Shape.** `physics/fluid_flow.py` holds the D3Q19 lattice, TRT collision, Guo
forcing, Smagorinsky closure and the ceramic-filter drag closure.
`physics/filling.py` holds the free-surface solver: Koerner mass tracking, cell
transitions, temperature advection and the defect indicators.
`core/filling_stage.py` owns the mesh-to-lattice seam and orchestration. That
split is what the layer contract forces -- `physics/` may not import `core/`,
so anything that knows what a voxel mesh is has to live in `core/`.

**Insertion.** `core/filling_stage.py::maybe_run_filling` is called from
`run_pipeline` before `SolidificationEngine` is constructed, and again from
`core/surrogate/pipeline.py`, which re-implements `run_pipeline`'s sequence
rather than calling it. A stage wired into only one of those silently no-ops in
the other.

**Handover.** `SolidificationEngine.set_initial_temperature_field(temperatures,
fill_arrival_times=...)`. It requires `(n_voxels,)` -- the OCCUPIED voxel count,
not the dense lattice -- and every entry finite, so misrun cells are seeded at
mold temperature rather than left blank. Only `thermal_coupled` consumes the
field; `schedule_legacy` derives its own timing and ignores it, and the stage
warns when it runs under that mode.

**It runs as a standalone pre-pass**, with its own state. `HeatTransferModel`
integrates all `n_voxels` unconditionally and has no active-cell mask, so a
growing fill domain cannot reuse it. Do not retrofit one.

**Two facts the mesh forces on any dense-grid stage.** `mesh.grid_shape` is a
pre-voxelization bounding-box estimate and disagrees with the real extent --
derive the shape from voxel centres instead. And `ijk = round((centre - min)/dx)`
is bijective, verified across geometries and both voxelization paths, which is
what makes a dense array representation viable at all.

**Gating.** The stage runs only when `AnalysisOptions.filling` is set AND a
`PourLocation` exists. Both UIs disable the toggle until a location is picked,
clearing a location forces the flag off, and `maybe_run_filling` refuses
regardless -- three independent layers, because a silently skipped stage is
indistinguishable from one that ran and found nothing.

Material properties for the momentum solver: `dynamic_viscosity` (Pa.s),
`get_dynamic_viscosity(T, fs)` (scalar; the array form is the separately-named
`vectorized_dynamic_viscosity`), `kinematic_viscosity`, `density_liquid`,
`surface_tension` (added for the free-surface and capillary filter-priming
terms), permeability and a Darcy pressure drop.

## Deleted, so you do not go looking

These were unreachable and were removed; recover from git if ever needed.

| Module | Why |
|---|---|
| `core/thermal_solver.py` | Duplicate heat solver shadowing `physics/heat_transfer.py` |
| `core/thermal_simulation.py` | Parallel orchestrator, no callers |
| `core/convection.py` | Natural-convection correlations, reachable only from the above |
| `core/mold_interface.py` | IHTC model, same |
| `core/optimizer.py`, `core/plugins.py`, `core/container.py`, `core/results_manager.py`, `core/shrinkage_parallel.py` | No importers. `plugins.py` and `container.py` were complete but never wired |
| `riser_geometry_loader.py` | Root-level, referenced only by a comment |

`core/plugins.py` in particular is not the extension point for a filling stage:
its `AnalysisPlugin.analyze(mesh, simulation)` is a *post-hoc* hook over a
finished simulation, and filling runs *before* solidification.

## Known rough edges

- `core/solidification.py` is ~5,800 lines in one class with ~90 attributes
  declared in `__init__` and more created outside it; access is largely
  defensive `getattr`.
- `ui/main_window.py` is ~6,400 lines and mixes ten dialog classes with the
  window.
- Adding one result field touches the engine, `build_visualization_payload`,
  `web/payloads.py`'s `field_specs`, `web/fields.py`'s `SCALAR_MODE_SOURCES`
  and `MODE_UNITS`, and the frontend mode tables. Failures are silent.
- `web/services/simulation_queue.py` keeps job state in a `Manager()` dict, so
  gunicorn `--workers 1` is required.
