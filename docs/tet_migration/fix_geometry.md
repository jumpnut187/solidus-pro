# Fix lane: geometry (review_geometry.md findings)

Branch feature/tet-mesh at 9514111, uncommitted. Nothing staged. The voxel path is unchanged.

## Findings addressed (review id -> change -> test)

- **P1-1.** `core/tet_mesh.py`: a module-level `_NATIVE_MESH_LOCK` (`threading.Lock`) is held for the whole `_silence_c_stdout` context, so the native fTetWild call is serialized too.
  Tests: `StdoutGuardTest.test_two_threads_meshing_concurrently_leave_stdout_intact`, which runs two real meshes in two threads and then checks that `sys.stdout` is the same object, that fd 1 has the same inode and that `print()` and `os.write(1)` both land. `test_overlapping_entries_are_serialized` forces A-enters, B-enters, A-exits and asserts that the peak occupancy is 1.
- **P1-2.** `_silence_c_stdout` is now exception-safe:
  - All fds and files are opened inside `try`.
  - `sys.stdout` is restored first in `finally`.
  - `_flush_c_stdio()` never raises. It catches any exception and skips when `fflush` is missing.
  - fd 1 is restored after the flush, and the stdout pre-flush tolerates None or a closed stream.
  - `_libc()` catches `Exception`. On Windows it tries `ucrtbase` and then `msvcrt`.

  Tests: `test_libc_failures_do_not_break_the_guard` covers `_libc` raising TypeError and `_libc` returning an object without `fflush`. `test_body_and_open_failures_restore_stdout` covers an exception in the body, `os.open` raising, and the lock being released afterwards. `test_libc_returns_none_when_every_runtime_fails_to_load` covers the Windows name order.
- **P1-3.** `core/simulation_pipeline.py`:
  - The estimate is now `1.55*V/h^3 + 3.3*A/h^2`. `TET_NODES_PER_UNIT_VOLUME` = 1.55 and `TET_NODES_PER_UNIT_AREA` = 3.3. The docstrings and constant comments are fixed.
  - New public `tet_edge_length_for_node_limit(parts, max_nodes)` returns the positive root of the cubic, stepped past the ceiling. The guard message uses it, rounded up to 0.001 in.
  - `web/routes.py` uses the same helper for `recommended_voxel_size` (supervisor-approved). Voxel responses are unchanged.

  Tests: `NodeCountGuardTests` covers the formula, the open surface, meshing the thin-walled box and the 2x1x1 box (estimate between 1.05x and 2.0x of measured), the helper landing at the limit for 1e3, 2.5e5 and 3e6, and the suggested size passing the guard.
- **P1-4.** Deviation, approved by the supervisor. The briefed fix (`_robust_contains_check_impl`) returns exactly `mesh.contains` on open parts, because its first strategy is `mesh.contains`.
  - New `winding_number_contains(mesh, points, *, near_surface_tol=0.0)` uses the generalized winding number, computed by a numba-parallel solid-angle sum. It is the rule fTetWild meshes open inputs by.
  - Watertight parts keep `mesh.contains`.
  - Above `WINDING_NUMBER_MAX_PAIRS` = 1e10 point x face pairs, it logs a warning and falls back to `_robust_contains_check_impl`.
  - `core/tet_filling.py` FILTER rasterization uses it for open filters, with the same 0.5-pitch tolerance.

  Tests in `OpenPartOverlapTest`:
  - Winding number against exact containment (< 2 % wrong).
  - An open-bottom riser sunk into the casting, in both part orders.
  - Watertight parts never call the winding test.
  - Fallback above the pair cap.
- **P2-1.** `core/gpu_compute.py`: new `_restore_fp_environment_after_taichi_init()` raises a `RuntimeWarning` on Darwin when the restore returns False. It is called after `ti.init`. Test: `test_failed_restore_warns_on_macos` checks the Darwin warning, no warning on Linux, and no warning on success.
- **P2-2.** `core/tet_lattice.py`: `voxel_neighbors`, `surface_voxels`, `voxel_surface_areas` and `voxel_moduli` are now `functools.cached_property`. Tests: `test_lattice_view_builds_topology_arrays_on_first_access` and `TetFillingGateTests.test_filling_does_not_build_lattice_topology`, which runs a real tet fill and asserts that none of the four is in `vars(view)`.
- **P2-3.** `core/tet_filling.py`: `node_lattice_cells(..., return_distance=True)` returns the mapping distances. `FillingLattice.far_node_count` counts nodes farther than `sqrt(3)*spacing` from their cell. `maybe_run_tet_filling` logs a warning and emits a progress message with the count. Tests: `FarNodeMappingTests` (a 0.04 in fin 1 in from the casting is counted and reported) and `NodeMappingTests.test_no_node_is_far_from_its_cell`. On the sample STLs (thick_thin, l_bracket and cylinder at 0.1 and 0.2) the count is 0, so normal parts do not trigger the warning.
- **Non-touching assembly.** `DisjointAssemblyTest` checks two adjacency components, all contact codes -1, and per-component boundary area equal to the part area within 1 %. `RunPipelineTetTests.test_non_touching_two_part_assembly_runs` runs `run_pipeline` end to end and checks that `tet_boundary_groups` covers every face index exactly once and that every metal node has a finite solidification time.

## Measurements

The node-count fit covers 14 meshes (7 shapes x 2 edge lengths). Each mesh is a single CAST part, meshed with `build_tet_mesh` at epsilon 1e-3. V and A are in in^3 and in^2.

The coefficients come from an LP that minimizes the summed estimate/measured ratio subject to est >= 1.10 x measured. The LP gave a = 1.518, b = 3.276, which I rounded up to 1.55 and 3.3.

| shape | V | A | h | nodes | old 2.0V/h^3 / meas | new est | new / meas |
|---|---|---|---|---|---|---|---|
| thin plate 6x6x0.5 | 18.00 | 84.00 | 0.100 | 33,510 | 1.07 | 55,620 | 1.66 |
| thin plate 6x6x0.5 | 18.00 | 84.00 | 0.050 | 204,881 | 1.41 | 334,080 | 1.63 |
| l_bracket (open) | 7.00 | 39.00 | 0.200 | 2,872 | 0.61 | 4,574 | 1.59 |
| l_bracket (open) | 7.00 | 39.00 | 0.100 | 13,602 | 1.03 | 23,720 | 1.74 |
| cylinder | 9.36 | 25.06 | 0.150 | 6,856 | 0.81 | 7,977 | 1.16 |
| cylinder | 9.36 | 25.06 | 0.075 | 43,901 | 1.01 | 49,109 | 1.12 |
| thick_thin | 21.00 | 56.00 | 0.200 | 4,647 | 1.13 | 8,689 | 1.87 |
| thick_thin | 21.00 | 56.00 | 0.100 | 34,849 | 1.21 | 51,030 | 1.46 |
| box 2x1x1 | 2.00 | 10.00 | 0.200 | 691 | 0.72 | 1,213 | 1.76 |
| box 2x1x1 | 2.00 | 10.00 | 0.100 | 5,738 | 0.70 | 6,400 | 1.12 |
| hollow box 4x3x2, wall 0.3 | 12.58 | 84.56 | 0.100 | 27,583 | 0.91 | 47,398 | 1.72 |
| hollow box 4x3x2, wall 0.3 | 12.58 | 84.56 | 0.050 | 187,094 | 1.08 | 267,562 | 1.43 |
| block 2 in | 8.00 | 24.00 | 0.100 | 12,206 | 1.31 | 20,320 | 1.66 |
| block 2 in | 8.00 | 24.00 | 0.050 | 96,953 | 1.32 | 130,880 | 1.35 |

- The old estimate under-counted 6 of 14 meshes. The new one over-counts every mesh by a factor of 1.12-1.87.
- The table uses the true volume for l_bracket. The estimator itself charges open parts their convex hull (17.5 in^3), so its l_bracket estimate is higher still.

Containment on an open-top riser (r 0.8, z 1..3, 32 sides), 20k random points:

| method | wrong points (of 6,737 inside) |
|---|---|
| `mesh.contains` | 1,933 |
| `_robust_contains_check_impl` | 1,933 |
| winding number | 47 (faceting) |

Assembly with an open-bottom riser, riser listed first: the riser code volume is 0.716 of exact with `mesh.contains` and 0.998 with the winding number.

Winding-number cost is 1.8-1.9e9 pairs/s on 18 cores. Measured: 20,480 faces x 1e5 points took 1.08 s, and 327,680 faces x 1e5 points took 18.7 s. That gives about 0.5 s at 1e4 faces and about 5.5 s at 1e5 faces per 1e5 points. Compiling on first call takes about 0.3 s (cached).

## Deviations

- **P1-4.** The fix uses the winding number, not `_robust_contains_check_impl`. The supervisor approved this, including its use in the filter path. I raised the pair cap from the suggested 2e9 to 1e10, which is about 6 s here, so 1e5-face risers stay on the correct path.
- **`web/routes.py`.** I edited it to use the new helper (approved). I did not touch `tests/test_tet_ui.py`.

## Test commands

- `venv/bin/python -m unittest tests.test_tet_mesh tests.test_tet_lattice tests.test_tet_pipeline tests.test_gpu_fp_environment tests.test_tet_end_to_end tests.test_filling_stage tests.test_architecture_boundaries`
  - 221 run, 220 pass, 1 failure: `test_moduli_match_exact_distance_on_curved_skin` (cylinder), off by 9.3e-5 against an atol of 1e-5. This existing test depends on fTetWild's non-deterministic mesh and the k=16 candidate bound in `_distance_to_boundary`, which I did not change. It passed on 3 of 3 reruns.
- `venv/bin/python -m unittest tests.test_tet_ui`: 23 run, 2 fail.
  - `test_tet_preview_reports_node_estimate_and_limit` and `test_ml_mesh_adds_the_tet_estimate_only_on_tet` fail because they hardcode `2.0*4/edge^3`. This is expected after P1-3. The integration lane owns that file and is updating it.
- `venv/bin/python -m mypy --config-file mypy.ini`: the 3 errors in `simulation_pipeline.py` (lines 683, 691, 798) are not in the code I changed. Lines 683 and 691 hold `# type: ignore` comments that are also at HEAD.

## Open risks

- The estimate over-counts by up to 1.87x on small or thick parts. Near `tet_max_nodes`, the guard can refuse meshes that would have fit by up to about 40 %. The fit has no meshes above 335k nodes.
- `_NATIVE_MESH_LOCK` serializes meshing. A riser Analyze started during a run now waits for the run's fTetWild call (up to about 1 min at 2 M tets) instead of corrupting stdout. While meshing runs, other threads' `print()` still goes to devnull (review note, unchanged).
- The Windows `ucrtbase`/`msvcrt` path is tested only by mocks. Nothing was run on Windows.
- The winding number treats a point near an open cap as outside once w < 0.5. That matches fTetWild's own classification, so tets near a missing cap follow the mesher.
- The `filling_stage.py:3670` matmul RuntimeWarnings appear in both the existing filter-fill test and the new fill test. They come from `filling_stage`, not from these changes.
