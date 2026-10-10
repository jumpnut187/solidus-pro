## Review: geometry, meshing, lattice, filling seam, pipeline (feature/tet-mesh, 20e098d..3f67c76)

I read the code statically and ran nothing. `watchdog_diff` covers only uncommitted changes, so I read the three wave commits from the current files and the wave reports.

### Correct (verified)
- **Boundary faces.** The `_TET_FACES` ordering gives outward normals for positive tets (tet_mesh.py:52). I checked all four faces against the unit tet. Boundary faces are kept-tet faces with no kept partner. The contact code is the removed partner's code, else -1 (tet_mesh.py:455-486). Nodes are re-indexed through `np.unique(kept_tets)`, so no orphan nodes are left.
- **Disjoint parts.** Parts that do not touch become separate components in the CSG union. Every exterior face of each component has no partner, so it gets a Robin group, and K stays nonsingular. No test covers a non-touching assembly.
- **Highest code wins.** The result does not depend on the order parts are processed: each pass only raises a tet's code toward the code of the part containing its centroid (tet_mesh.py:402-431). The node part-type rule matches C1 (tet_mesh.py:516-526). Codes match the voxel rule, so GATE (9) beats FILTER (8).
- **Moduli bound.** The "overestimate at most r_max of a face" bound holds. If the true nearest face f* is not among the k candidates, the best candidate is at most d* + r(f*) away.
- **Lattice location.** Point location is chunked, assigns each cell to the lowest-index tet, and decodes indices x-fastest consistently. The EDT reshape `(nz,ny,nx)` matches (tet_lattice.py:150-275, 360-372). FILTER rasterization uses the sampler's own centres with the voxel path's robust containment check (tet_filling.py:99-138).
- **FP fix.** `fesetenv` is per-thread. It runs on the thread that called `ti.init`, which is the only thread Taichi changed (gpu_compute.py:241-242). Web jobs are `spawn` processes that re-import `gpu_compute` (simulation_queue.py:27, 628), and forked gunicorn workers inherit the restored register. `ti.init` has no other call site.
- **Settings.** Tet fields are re-validated after file and env load (settings.py:574). The STEP tessellation tolerance (voxel/50, at most 0.5 mm) is well below the tet edge length and the same order as the fTetWild envelope.

### Findings

**P1-1. `_silence_c_stdout` is not reentrant or thread-safe, and the desktop can enter it twice at once.** (core/tet_mesh.py:349-374)
- **How it is reached:** the desktop meshes in a `SimulationWorker` QThread (ui/workers.py:132) and, for riser Analyze, on the GUI thread (ui/main_window.py:2046). Nothing stops the riser dialog during a run (main_window.py:7388).
- **Failure:** thread A (simulation) enters, then thread B (riser) enters, then A exits first. B's saved fd is A's devnull and B's `saved_stdout` is A's `python_devnull`, which A's `with` has closed. When B exits, fd 1 is left on /dev/null for good and `sys.stdout` is a closed file. Every later `print()` raises `ValueError: I/O operation on closed file`, which crashes the rest of the run.
- **Single entry also misbehaves:** other threads' prints are dropped while meshing runs (up to about 1 min at 2 M tets). The console's `start_capture`/`stop_capture` (main_window.py:1735-1747) is overwritten when meshing restores `sys.stdout`.
- **Fix:** hold a module-level `threading.Lock` for the whole context. This also serializes concurrent native fTetWild calls, whose thread safety is unknown.

**P1-2. The stdout guard can break fTetWild on Windows and leave `sys.stdout` closed.** (core/tet_mesh.py:356-381)
- `_libc()` catches only `OSError`. On Windows, `ctypes.CDLL(None)` is expected to raise `TypeError`. Even a returned handle has no `fflush` export, which gives `AttributeError`. I did not test this on Windows. Windows is a supported desktop target (tests/test_windows_desktop_compat.py).
- With `TypeError`, the exception is raised at :360, before the `try`. The fds `saved` and `devnull` leak, and every tet mesh fails as "fTetWild failed: ...".
- With `AttributeError` at :369, `sys.stdout = saved_stdout` (:370) never runs. `sys.stdout` is left pointing at the closed devnull file.
- **Fix:**
  - Catch `Exception` in `_libc` and use `msvcrt`/`ucrtbase` on Windows.
  - Restore `sys.stdout` before calling `fflush`, or wrap the `fflush` call in its own try.
  - Open the fds inside the `try`.

**P1-3. The node-count guard is called an "upper envelope", but the measurements show it under-counts.** (core/simulation_pipeline.py:51-52, 171-191)
- Measured nodes / (V/h³) was 2.8 on a 2x1x1 box at h = 0.2 and 2.12 on the l_bracket at 0.1 (wave2_pipeline.md, deviation 4). The guard uses 2.0.
- **Failure:** the ratio depends on wall thickness / h, not on overall size, because surface nodes scale as A/h². A thin-walled casting with walls 5-10 h thick and an estimate of 2.9 M passes the 3 M guard, then meshes to about 3.0-4.1 M nodes (+6 % to +40 %).
- The worker's argument that "the ratio falls near the limit" holds only when features are thick relative to h.
- **Fix:** estimate `a·V/h³ + b·A/h²` (trimesh gives `mesh.area`), fitted on the four measured cases, or raise the factor to 3.0. Correct the docstring and the constant's comment either way.

**P1-4. The overlap pass uses `Trimesh.contains`, which is unreliable on non-watertight parts.** (core/tet_mesh.py:429)
- fTetWild meshes open parts by winding number, but the code-correction pass ray-tests with `mesh.contains`. Trimesh documents `contains` as valid only for watertight meshes.
- **Failure:** in an overlapping assembly where the higher-code part is open (for example a riser STL with a missing cap overlapping the casting), cast tets inside the riser's bbox can be wrongly raised to RISER. Or riser tets keep a lower marker. Either way part codes and BC groups are wrong.
- The voxel path and `tet_filling` use `_robust_contains_check_impl` instead (tet_filling.py:132). The tests cover only watertight boxes (tests/test_tet_mesh.py:303).
- **Fix:** call `_robust_contains_check_impl(mesh, pts, 0.0)` when `not mesh.is_watertight`.

**P2-1. A failed FP restore is silent.** (core/gpu_compute.py:242) The `False` return of `_restore_default_fp_environment()` is discarded. If `fesetenv` or the `_FE_DFL_ENV` lookup fails, the next symptom is a SIGSEGV inside `cKDTree` (tet_mesh.py:548, tet_filling.py:161) with no diagnostic. **Fix:** on Darwin, `warnings.warn` when it returns False.

**P2-2. Filling builds a full `LatticeView` it never reads.** (core/tet_filling.py:180)
- `filling_stage` reads only centres, sizes, volumes, part types and the count (wave2_pipeline.md:10).
- `LatticeView.__init__` still builds an (M,26) int32 neighbour table (about 520 MB at 5 M cells) and a full-box EDT (tet_lattice.py:331-372).
- **Fix:** compute `voxel_neighbors`, `surface_voxels` and `voxel_moduli` lazily (cached properties).

**P2-3. Node-to-cell mapping has no distance cap.** (core/tet_filling.py:141-163) A disconnected metal part thinner than the lattice pitch has no inside cell. Its nodes copy fill time and temperature from the nearest metal cell of a different part, with no warning. **Fix:** count the nodes whose nearest distance is above `spacing·√3` and warn about them.

### Residual risks
- I ran no tests and nothing on Windows. The P1-2 Windows behaviour is inferred from the CPython `ctypes` source and is unverified here.
- Whether Taichi sets flush-to-zero again on kernel launch from other threads is not covered by tests/test_gpu_fp_environment.py, which checks only the importing thread.
- Cancel is checked only between meshing stages. An fTetWild call (about 1 min at 2 M tets) cannot be interrupted.
- Suggested commands for the supervisor: `venv/bin/python -m unittest tests.test_tet_mesh tests.test_tet_lattice tests.test_tet_pipeline tests.test_gpu_fp_environment`.

Merge verdict: OK with notes. There is no P0. The voxel path is still the default, so these issues are reached only on tet runs. Fix P1-1 and P1-2 before the tet default flips in wave 4.