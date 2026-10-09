# Fix: fTetWild crash kills the desktop app (C05813 Rigging, 17 parts). Branch `feature/tet-mesh`, base 8ba4caf, nothing staged.
## Root cause (as far as determined)

- Every crash report (app, CSG child, in-process soup) shows the same thing: `EXC_BAD_ACCESS / SIGBUS (KERN_PROTECTION_FAILURE)` in
  `floatTetWild::remove_duplicates <- floatTetWild::simplify`, on the calling thread.
- The fault address is in unallocated VM 7-8 GB past the last `Malloc Small` region, and `sp` is far from it. That is an out-of-bounds heap
  read, not a stack overflow, which matches the parent's 256 MB stack result.
- The crash is not deterministic. CSG crashed in every run at 0.2 and 0.3 in with default threads and passed with `num_threads=1`. Soup crashed
  in 2 of 5 runs, within seconds of the call. One of those was the filtered call at 0.3 in, the same call that passed earlier in 222 s; it
  died with SIGSEGV in a second place, `find_boundary_edges <- AABBWrapper::init_b_mesh_and_tree`.
- Hypothesis, not verified (no fTetWild debug build): thread-order-dependent output of the parallel simplification feeds `remove_duplicates`
  a bad index or input that breaks the strict weak ordering of a sort, and libc++ `std::sort` then reads out of bounds.

## Changes

1. `core/tetwild_child.py` (new) imports only stdlib, numpy and pytetwild. It is run as `sys.executable core/tetwild_child.py <workdir>`.
   Its input is `request.json` plus `surface.npz` (soup) or OBJ files and `csg.json` (CSG). It writes `result.npz` atomically.
   `CASTSIM_TETWILD_CHILD_FAULT=csg,soup` makes it SIGSEGV itself before the native call. Tests use this.
2. `core/tet_mesh.py`
   - Every fTetWild call runs in that child (`_run_child`). The child's cwd is a TemporaryDirectory, so `__tracked_surface.stl` lands there.
     stdout goes to DEVNULL and stderr to a file. The parent polls every 0.15 s and calls `progress_callback`.
   - If the callback raises (cancel), the parent terminates the child, kills it after 1 s, and re-raises. `_NATIVE_MESH_LOCK` is kept, and the
     lock wait also polls the callback.
   - `_silence_c_stdout`, `_flush_c_stdio` and `_libc` were removed.
   - `MESHING_ATTEMPTS` sets the attempt order: `auto` = csg, soup, soup; `csg` = csg; `soup` = soup, soup. One part always uses soup.
   - Each attempt is logged with its outcome and wall time. A failed attempt logs a warning with the signal name and the last 20 stderr lines,
     and reports "retrying with ..." through progress.
   - When every attempt fails, `MeshGenerationError` is raised. Its text: "the tet mesher (fTetWild) failed at edge length 0.2 in (CSG union:
     crashed with SIGBUS, ...). Change the mesh size, or switch Mesh type to Voxel."
   - `build_tet_mesh(..., strategy="auto"|"csg"|"soup")` is a new keyword. `TetVolumeMesh.meshing_strategy` is `"csg"` or `"soup"`. The final
     progress message names the strategy.
   - **Deviation from the brief (supervisor-approved).** The soup attempt calls `tetrahedralize(..., simplify=True, disable_filtering=True)`.
     `_label_soup_tets` then labels every tet by CAD centroid containment: highest code wins, `Trimesh.contains` (embree) for watertight parts,
     `winding_number_contains` for open parts. Only centroids inside a part's bounding box are tested.
     - Tets inside no part are exterior and are dropped. Faces against them become sand boundary with contact code -1.
     - This replaces the brief's "slop takes the neighbour or nearest code" rule. The reason: with fTetWild's own filter on, soup lost a
       0.25 in slab of metal at the casting/riser interface (riser 0.975 of CAD, the same with simplify off or epsilon 1e-4). Unfiltered soup
       gave 0.998.
     - One part keeps fTetWild's filter on, as before.
   - CORE/FILTER removal, boundary faces, contact codes and node arrays use the existing code path.
3. `tests/test_tet_mesh.py`
   - `TetMeshAssemblyTest` is parametrized (`STRATEGY`, `VOLUME_TOLERANCE`).
   - New `TetMeshSoupAssemblyTest` runs the same three-part assembly in soup mode in both input orders. Per-code volumes are within 2 % and
     interfaces share more than 10 nodes. It also asserts that every kept tet centroid is inside a part and that every boundary vertex or face
     centroid is within 2 x envelope of a part surface.
   - New `TetMeshSoupCoreTest`.
   - New `TetwildChildProcessTest`:
     - the child imports only numpy and pytetwild
     - a CSG SIGSEGV falls back to soup and the test process survives
     - one soup retry (csg, soup, soup)
     - all attempts crashing, for an assembly and for one part, raises with SIGSEGV, the edge length and the Voxel hint
     - a Python error in the child reports the exit code and stderr
     - cancel kills a running child in under 2 s and releases the lock
     - two threads meshing at once leave stdout and fd 1 untouched
   - The 4 `_silence_c_stdout` / `_libc` tests were removed.

## Tests

- `venv/bin/python -m unittest tests.test_tet_mesh tests.test_tet_pipeline tests.test_tet_end_to_end tests.test_tet_lattice tests.test_architecture_boundaries`
  ran 118 tests, all OK (78 s), after the final edit.
- `venv/bin/python -m unittest discover tests > /tmp/crashfix_full.log`: 2083 run, 16 skipped, 3 failures. All 3 are the DESIGN.md baselines
  (filling_stability deeper tower, porosity case pack riser_post_block, shrinkage_display_intensity). This run came before the soup-retry edit
  and its test. The tet modules were re-run afterwards (above).

## Real model: `run_pipeline(mesh_type="tet")`, voxel 0.1 in (edge 0.2 in), default analyses

Script `/tmp/crashfix/real.py`, log `/tmp/crashfix/real.log`. It ran before the soup retry existed. A filtered soup run was going at the same time.

| item | value |
|---|---|
| strategy | CSG child crashed with SIGBUS after 8.2 s, then soup (the app survived) |
| build_tet_mesh wall | 930 s: fTetWild soup about 897 s, orient + labelling 13 s (containment 10.9 s), node arrays 11 s |
| tets generated / kept | 17,359,994 / 2,126,895 (the unfiltered bbox fill costs 8x) |
| nodes / boundary faces | 397,177 / 130,426 |
| quality_summary | min dihedral 11.13 deg, median 67.9 deg, 0 slivers (< 5 deg) |
| peak RSS | child 13.7 GB, parent 9.5 GB |
| run_pipeline wall | 1085 s (mesh 930 s, then filling skipped, solidification + post about 155 s) |
| solidification total time | 748.6 s; NaN on metal nodes: 0 of 385,822 |

| code | parts | CAD in3 | mesh in3 | error |
|---|---|---|---|---|
| 0 CAST | 1 | 792.81 | 792.66 | -0.02 % |
| 1 RISER (6 risers + feeder pad) | 7 | 289.17 | 288.34 | -0.29 % |
| 2 CHILL | 4 | 48.00 | 47.90 | -0.20 % |
| 3 RISER_POST | 5 | 613.48 | 612.73 | -0.12 % |

All 17 parts are watertight, so containment always used embree.

## Speed investigation, soup at 0.2 in (report only; defaults unchanged)

| run | fTetWild wall | quality | note |
|---|---|---|---|
| filtered, stop_energy 10 (old soup call) | over 54 min, killed unfinished | - | this was `speed.py`, partly concurrent with the real run |
| 0.3 in unfiltered (`build_tet_mesh(strategy="soup")`) | 253 s | min 12.0 deg, 0 slivers, 639k kept of 3.8M | parent's filtered call: 222 s (+14 %) |
| 0.3 in filtered, rerun | SIGSEGV after 8 s | - | `find_boundary_edges`, in-process |
| unfiltered se10, run 1 (pipeline above) | 897 s | min 11.1 deg, 0 slivers | concurrent with the filtered run |
| unfiltered se10, run 2 (`/tmp/crashfix/speed2.py`) | 2207 s | min 0.00095 deg, 21 slivers, 2,118,267 kept of 17,058,947 | uncontended; volumes within 0.3 % |
| unfiltered se20 | SIGBUS in `remove_duplicates` within seconds | - | in-process, so the sweep process died |
| se40, coarsen | not run | - | skipped on supervisor instruction |

Filtering is not the cost. Unfiltered soup is 14 % slower at 0.3 in and finished at 0.2 in, while the filtered call did not finish in
54 min. So the filtered-soup hybrid the parent suggested would not buy time and would bring the interface void back; I did not add it.
The time is fTetWild at 0.2 in on this model. Recommendation: do not change stop_energy or coarsen. No faster setting was measured, and run-to-run variance (2.5x in wall time, and 0 vs 21
slivers at identical settings) is larger than any effect a setting could show in one run. At 0.2 in, this model takes 15-37 min to mesh in the
app.

## Open risks

1. Soup also crashes (2 of 5 runs, in two different frames). One retry lowers the odds but does not remove them. When every attempt fails, the user gets the
   MeshGenerationError above. The parent plans an automatic voxel fallback (time limit + size gate).
2. Wall time and quality are not reproducible: 897-2207 s, and slivers can appear (21 below 5 deg, min 0.001 deg).
3. Memory: 13.7 GB child peak plus 9.5 GB parent on this model. Machines with 16 GB will swap or be OOM-killed. A child killed this way is
   reported as a SIGKILL failure.
4. Unfiltered soup generates 8x the kept tets on this assembly. Assemblies with a sparser bounding box cost more.
5. Shared interface nodes in soup mode are verified on the synthetic assembly only, not counted on the real model.
6. The child needs `sys.executable` to be a Python interpreter. A frozen build would need a different launcher. On Windows, crashes show as
   exit codes, not signal names.
7. `tests/test_tet_lattice.py` still calls pytetwild in-process with cwd = repo. That is where the untracked `__tracked_surface.stl` in the
   repo root comes from. It existed before this change, and my first sweep also rewrote it.
