# Scout: post-analyses seam (voxel -> tet nodes)

Scope: `run_post_analyses` and everything it dispatches. Read-only scout. Line numbers are from branch `feature/tet-mesh` at the time of the scout.

## 1) Entry points and call graph

`core/analysis_strategies.py:11 run_post_analyses(engine, show_progress)` runs these stages in order. Each one is gated by `engine.analysis_options`.

1. Feeding path (`:21-39`) → `engine._calculate_feeding_paths` (`core/solidification.py:5637`) → `FeedingPathAnalyzer(mesh, alloy).analyze(...)` (`core/solidification.py:5655-5663`, `core/feeding_path.py:605`).
2. Hotspot (`:41-46`) → `_calculate_hotspots` (`core/solidification.py:3963`), which uses `_calculate_isolation_score` (`:4060`) → `compute_isolation_score_parallel` (`core/acceleration.py:910`).
3. Niyama (`:48-53`) → `_calculate_niyama` (`core/solidification.py:4095`). It consumes the per-step thermal Niyama arrays `_thermal_niyama_*` that `_least_squares_gradients_c_per_mm` fills during the solve loop (`core/solidification_loop.py:133`, called at `:439-451`).
4. Shrinkage (`:55-70`) → `_calculate_shrinkage` (`core/solidification.py:3349`), then `_calculate_porosity` (`:3743`), then `_apply_shrink_scalar` (`:3916`). The riser pipe path at `core/solidification.py:3036-3090` calls `physics/riser_pipe.py:138 gravity_drain_pipe`.
5. Hot tear (`:72-77`) → `_calculate_hot_tear_risk` (`core/solidification.py:4445`). BTR support scan is at `:4346-4365`.
6. Sand burn-in (`analysis_strategies.py:79-~100`) → `_calculate_sand_burn_in` (`core/solidification.py:4891`).
7. Gas porosity (`analysis_strategies.py:~103`) → `_calculate_gas_porosity` (`core/solidification.py:5169`) → `physics/gas_porosity.py:57,151,216`.
8. Stress (`analysis_strategies.py:~130`) → `_calculate_stress_analysis` (`core/solidification.py:5688`) → `StressAnalyzer.analyze` (`core/stress_analysis.py:393`).
9. Microstructure (`analysis_strategies.py:~150`) → `_calculate_microstructure` (`core/solidification.py:5735`) → `core/microstructure.py:466-478`.
10. Always: `_calculate_chvorinov_times` (`core/solidification.py:5758`), then `_generate_statistics`.

These are **not** dispatched by `run_post_analyses`:

- `core/riser_design.py` runs only from the UI (`ui/main_window.py:1863-1881`).
- `core/report_generator.py` runs only from the UI (`ui/main_window.py:7413`). It reads no mesh attributes (grep for `mesh|voxel|neighbor` finds only docstrings).
- `core/incremental_solidification.py` is a solver-time engine (`core/solidification.py:91`), not a post-analysis.
- `core/defect_regions.py` (`extract_pocket_at`, `:126`) is called on demand from the UI/display layer.

## 2) Data consumed and produced

These mesh attributes appear across the seam:

| attr | shape/dtype | units | readers (file:line) |
|---|---|---|---|
| `voxel_neighbors` | (N, 26) int32, -1 padded | idx | feeding_path.py:646; solidification.py:2877, 4072, 4357, 4538 (cast to **int64**); solidification_loop.py:444; stress_analysis.py:436; microstructure.py:471; defect_regions.py:127, 184 |
| `voxel_centers` | (N, 3) float | in | nearly every stage. It is also the N source (`len(mesh.voxel_centers)`), e.g. solidification.py:3376, 4123, 4932, 5644 |
| `voxel_volumes` | (N,) float32 | in³ | defect_regions.py:129, 197; solidification.py:3062-3066 (falls back to `voxel_sizes**3`) |
| `voxel_sizes` | (N,) or scalar | in | solidification.py:3065 (fallback only) |
| `base_voxel_size` | scalar | in | solidification.py:4440, 4633, 5065; stress_analysis.py:465; microstructure.py:478 |
| `voxel_moduli` | (N,) float32 | in | solidification.py:2878, 3724, 4539, and `_get_risk_moduli` ~:1511 |
| `voxel_part_types` | (N,) int32 (RISER=1, CHILL) | enum | solidification.py:2885, 3041, 4082, 4513; gas_porosity.py:59-101; defect_regions.py:102 |
| `get_surface_voxel_indices()` | (S,) int | idx | solidification.py:4524, 4953, 5185, 5651; gas_porosity.py:183 |
| `voxel_states` | (N,) float32, fs 0..1 | — | solidification.py:2683, 2876, 5698 |
| `get_voxel_count()` | int | — | feeding_path.py:513, 628; stress_analysis.py:408; microstructure.py:386, 466 |

Engine fields consumed downstream:

- `solidification_times` (N,) s.
- `surface_distances` and `cast_surface_distances` (N,) in. These are point-to-surface-center distances (`core/solidification.py:1454-1468`), not an EDT.
- `fraction_liquid_history`, a list of (N,) fs snapshots (solidification.py:4346-4365).
- `feeding_path_result.feeding_source` (N,) int (solidification.py:3050-3058).

Niyama kernel contract (`solidification_loop.py:133-183`):

- Inputs: `centers_in` (N, 3) f64 in inches; `temperatures` (N,) f64 °C; `neighbors` (N, K) int32; `target_indices` int64; `cast_mask` bool.
- Outputs: `gradients` (M,) f32 in °C/mm and `quality` (M,) int8. Quality is 2 when the lstsq rank is 3 and there are at least 4 neighbors, otherwise 1.
- Niyama is `G / sqrt(rate)` (`:452`).

`gravity_drain_pipe` (`physics/riser_pipe.py:138`):

- Inputs: times (N,), heights (N,) as z, volumes (N,), pool_ids (N,), feeder_mask, beta.
- Output: a per-node pipe fraction. It has no neighbor input.

## 3) Voxel-specific assumptions

Neighbor slot semantics:

- **defect_regions.py:9-17, 111-122.** Slot ordering 0-5 face, 6-17 edge, 18-25 corner. `connectivity` 6/18/26 truncates the slot range (`row[slot] for slot in range(slot_count)`, `:183-186`). This is the only lattice-slot-dependent code in the seam. It requires `shape[1] >= 6` (`:162`).
- No other stage indexes specific slots. The grep for `[:, :6]`, `range(6)` and `slot` in feeding_path, stress, microstructure, physics and solidification.py found only heap slots (feeding_path.py:248-414) and comments.

Graph-generic code. These loop `for j in range(neighbors.shape[1])` and skip `<0`:

- feeding_path.py:145-195 and 338-414, with Dijkstra edge length taken from `voxel_centers`.
- `FeedingPathAnalyzer._neighbor_edges` (`:433-445`), which converts padded rows to an i<j edge list.
- acceleration.py:910-960 (isolation).
- stress_analysis.py:142-189.
- microstructure.py:282-316.
- incremental_solidification.py:64-77 and 209-214.
- solidification_loop.py feeding-zone kernel (~:190-225).
- Niyama lstsq (`solidification_loop.py:154-158`). It uses real center offsets, so any neighbor geometry works.

Implicit uniform-lattice weighting:

- Isolation score (acceleration.py:939-951) is an unweighted neighbor count fraction. With 26 neighbors this is ≈ isotropic. Tet node valence varies (≈ 10-20+), so scores shift. It is graph-generic but not calibrated.
- Stress constraint strain averages over neighbors without weighting (stress_analysis.py:186-189). Microstructure does the same (`:298-316`).

Uniform `voxel_size` used as a length scale:

- solidification.py:4440 (restraint influence = 5 × size).
- `:4633-4635` (hot-tear path length = size × search_depth).
- `:5065` (chill thickness = 5 × size).
- stress_analysis.py:251-260, 465 (distortion length scale).
- microstructure.py:260, 276, 478 (SDAS gradient refinement uses size as dx).
- riser_design.py:378 (`cluster_volume = len × size³`), `:412`, `:678` (top band = 3 × size), `:782-783` (lattice ±size stencil).

Dense lattice / ijk:

- None of the post stages use ijk or `grid_shape`. `physics/filling.py:2146-2184` uses a scipy 6-connectivity structuring element, but it is outside this seam (filling).

API drift (open question):

- riser_design.py:447-450 reads `mesh.centers`, `mesh.voxel_size`, `mesh.neighbors` and `mesh.surface`. These are not the `voxel_*` names used everywhere else, so they are probably a different mesh_data adapter built in `ui/main_window.py:1863-1881`. Not verified.

Volume semantics:

- `voxel_volumes` is in³ per cell. For tet nodes, lumped volume (Σ tet_vol / 4 per node) is a direct replacement for pocket volume (defect_regions.py:197-199) and the pipe (solidification.py:3062).

Surface:

- `get_surface_voxel_indices()` returns boundary cells. Gas porosity, sand burn-in and hot tear (`:4524-4532`) zero or exclude these. Tet equivalent: boundary nodes of the surface triangles. Material-interface surfaces come from `_get_material_surface_indices` (`solidification.py:4264`).

## 4) What a tet-node replacement must provide

Minimum change: keep the padded `(N, K)` int32 table with -1 fill. Generate it from the edge CSR as `neighbors[i, :deg_i] = indices[indptr[i]:indptr[i+1]]` with K = max degree. Every generic kernel above then runs unchanged.

The CSR alone would need rewrites in about 8 numba kernels. Recommend exposing both: padded (N, K) for the existing kernels, and CSR (`indptr`, `indices`) for new code. Watch out for one high-valence node (K ≫ mean), because it inflates memory to N × K. Cap or verify max degree.

Required node arrays, under the same attribute names (or a thin adapter `TetMeshView` that aliases them):

1. `voxel_centers` → node coordinates (N, 3), inches.
2. `voxel_volumes` → lumped node volume (N,) f32, in³. This removes the `voxel_sizes**3` fallback dependency (solidification.py:3065).
3. `voxel_neighbors` → padded edge adjacency, int32.
4. `voxel_part_types` → per-node part type. Interface nodes are ambiguous: a node shared by casting and chill or mold tets needs a defined rule. The Niyama `cast_mask` and gas surface typing (gas_porosity.py:139-148) depend on it.
5. `voxel_moduli` → node modulus (V/A via distance field or local). Producer outside this seam.
6. `get_surface_voxel_indices()` → boundary node indices. Also define the material-interface surface (`_get_material_surface_indices`).
7. `get_voxel_count()`.
8. `base_voxel_size` → a scalar characteristic length. Use the mean edge length or the h used for meshing. Ideally replace it per node with local h (mean incident edge length) at solidification.py:4440, 4633, 5065, stress_analysis.py:465 and microstructure.py:478.

Per-stage minimum change:

| stage | change |
|---|---|
| feeding path | none (generic; edge weights from centers) |
| hotspot / isolation | none to run. Recalibrate thresholds for valence |
| Niyama lstsq | none to run. Optional: replace it with the P1 element gradient averaged to nodes (more accurate, uses tets). Quality rule of 4+ neighbors and rank 3 holds for tets |
| shrinkage / porosity / pipe | provide `voxel_volumes`. Pipe uses z = `voxel_centers[:, 2]` (solidification.py:3072) |
| hot tear | `base_voxel_size` → local h. Neighbors cast to int64 at :4538 is fine |
| sand burn-in | surface node indices plus scalar size at :5065 |
| gas porosity | surface node indices plus part types. Distance to surface centers is generic |
| stress | scalar size → local h |
| microstructure | dx = size → actual edge length `|c_i - c_j|` per neighbor (microstructure.py:298-316) |
| defect_regions pocket | `connectivity` 6/18/26 has no meaning on tets. Treat the full row as connectivity (ignore or deprecate the argument). Volume = lumped volumes |
| riser_design (UI) | replace the ±size lattice stencil (:782-783) and `len × size³` (:378) with node volumes |
| report_generator | none |
| incremental_solidification | generic. "Up to 26" buffer sizing at :193 must use K |

## 5) Risks and pinning tests

Risks:

1. Valence-dependent heuristics change risk maps: isolation fraction, and averaged stress and microstructure. Golden-value tests will drift without any bug.
2. Boundary nodes sit on the surface, whereas voxel centers sit half a cell inside. Surface distances become 0 at the boundary, which shifts `surface_distances` normalisation (solidification.py:4007-4009, 4089-4091) and Niyama surface suppression (`:4264-4267`).
3. Non-uniform tet h breaks `base_voxel_size` length scales.
4. Part-type ambiguity at shared interface nodes. A tet mesh of a multi-body assembly needs conforming interfaces or duplicated nodes. Unknown which approach fTetWild output gives.
5. Padded K memory if a node has extreme valence.
6. The `riser_design` mesh attribute names differ (`centers`, `neighbors`, `surface`). The adapter location is unverified.
7. The `defect_regions` `connectivity` API contract has no tet analogue.

Tests that pin current behavior (`tests/`):

- test_feeding_path_analysis.py
- test_defect_regions.py
- test_isolated_pocket_shrinkage.py
- test_isolation_score_kernel.py
- test_shrinkage_display_intensity.py
- test_riser_pipe.py
- test_centered_riser_symmetry.py
- test_gas_porosity.py
- test_hot_tear_physics.py
- test_hot_tear_phase2.py
- test_sand_burn_in.py
- test_desktop_analysis_options.py
- test_simulation_pipeline_options.py

No dedicated test files exist for Niyama, stress, microstructure, report or incremental (grep of tests/ by name). Their coverage, if any, is indirect.

## Start here

Open `core/solidification_loop.py:133` first (Niyama lstsq), then `core/defect_regions.py:111-200`. The first is the most accuracy-sensitive consumer. The second is the only slot-semantic consumer.
