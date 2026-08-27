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

`run_pipeline` has four callers: `ui/workers.py`, `web/routes.py`,
`web/services/simulation_queue.py` and `core/surrogate/training.py`.

`core/surrogate/pipeline.py` is a **sibling** pipeline: it reuses
`prepare_mesh_and_geometry` and `run_post_analyses` but substitutes its own
solve stage. It is the pattern to copy for a new stage — do not add physics
inside `SolidificationEngine`, which is already 5,400 lines and one class.

## Layer contracts

| Layer | May import | Must not import | Holds |
|---|---|---|---|
| `physics/` | stdlib, numpy, numba | `core/`, `config/`, `ui/`, `web/` | Mesh-agnostic solvers |
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

Codes are `PART_TYPE_CAST=0, RISER=1, CHILL=2, RISER_POST=3, CORE=4`, defined in
`core/part_manager.py`. They are persisted in saved projects and encoded into ML
training records, so they are **not free to renumber**.

Use the predicates rather than integer comparisons:

```python
from core.part_manager import is_metal, is_feeder, is_mold, is_cast, is_chill
```

Adding a part type means adding it to `METAL_PART_CODES` / `MOLD_PART_CODES`
once, instead of auditing every comparison site. `tests/test_part_type_predicates.py`
fails if a new type belongs to neither group.

**Adding a part type also changes the ML encoding.** `core/surrogate/fields.py`
encodes channel 1 as `code / PART_TYPE_DIVISOR`, and that divisor is recorded in
`ModelManifest.part_type_divisor`. `load_model` refuses artifacts whose recorded
divisor differs from the current one, so bumping it forces a retrain instead of
silently degrading predictions.

## Where a new physics stage plugs in

For mold filling specifically:

1. **Insertion point** — `core/simulation_pipeline.py`, between
   `prepare_mesh_and_geometry()` and constructing `SolidificationEngine`.
2. **The seam to cut** — `core/solidification.py:407` currently does
   `initial_temperatures = np.full(n_voxels, alloy.pouring_temp)`. That single
   uniform assignment is the whole "pouring model" today. The line below it
   already demonstrates the override pattern for chills, and
   `HeatTransferModel.set_initial_temperature` accepts a full array.
3. **Where the solver lives** — `physics/filling.py` (integrator) plus
   `physics/fluid_flow.py` (kernels), mirroring `heat_transfer.py`'s shape.
4. **Run it as a standalone pre-pass.** `HeatTransferModel` integrates all
   `n_voxels` unconditionally and has no active-cell mask, so a growing fill
   domain cannot reuse it directly. Give the filling solver its own state and
   hand back only `fill_time[]` and `T_after_fill[]`. Do not retrofit an active
   mask into the thermal model.
5. **Inlet boundaries** — `BoundaryType.HEAT_FLUX` is implemented and
   `clear_boundaries()` lets a moving front revise its wetted cell set each step.
   `add_boundary` accepts arbitrary index sets and a `time_function`.

Material properties already present for a momentum solver: `dynamic_viscosity`
(Pa·s), `get_dynamic_viscosity(T, fs)` (temperature- and mush-dependent),
`vectorized_dynamic_viscosity`, `kinematic_viscosity`, Prandtl, `density_liquid`,
permeability and a Darcy pressure drop. **Missing: surface tension**, needed for
the Weber number and gate capillary behaviour.

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

- `core/solidification.py` is ~5,400 lines in one class with ~90 attributes
  declared in `__init__` and more created outside it; access is largely
  defensive `getattr`.
- `ui/main_window.py` is ~6,400 lines and mixes ten dialog classes with the
  window.
- Adding one result field touches the engine, `build_visualization_payload`,
  `web/payloads.py`'s `field_specs` and `SURFACE_MODES`,
  `core/surface_extraction.py`'s `VALID_SURFACE_MODES`, and four frontend
  tables. Failures are silent.
- `web/services/simulation_queue.py` keeps job state in a `Manager()` dict, so
  gunicorn `--workers 1` is required.
