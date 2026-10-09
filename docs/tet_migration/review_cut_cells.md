# Cut-cell review: commit b912460, `feature/tet-mesh`

## Review

**Scope and limitation.** `watchdog_diff` shows no tracked working-tree changes, so I could not read the committed diff for b912460. I reviewed the files as they are at HEAD. I compared each flag-on branch with the flag-off code next to it. I ran no tests.

### Correct
- **Flag-off isolation (by reading the code; not proven bit-for-bit).** Every cut-cell path sits behind a guard:
  - `core/voxel_mesh.py:878` runs only when `_cut_cells_enabled()` is true. `_apply_cut_cell_part_types` returns early unless `self.cut_cells` is set (`:994`).
  - `HeatTransferModel` keeps all cut-cell state `None` unless arrays are passed (`physics/heat_transfer.py:289-325`). The old heterogeneous, uniform, CFL and boundary branches are reached unchanged (`:677-684`, `:902-907`).
  - `_cut_cell_solver_inputs` returns `{}` when there are no cut cells (`core/solidification.py:716-718`). The thermal loop takes the old enthalpy correction unless `_cut_cell_merge` is set (`core/solidification_loop.py:976,1015`).
  - `FlagOffTest` checks that the cut-cell code is never reached with the flag off. It does not check that results are bit-identical.
- **Robin units.** Area is converted in² to m² twice (`solidification.py:732`). The Robin term uses A/(w·dx³), with dx in metres (`heat_transfer.py:316-325`, `:967-993`). Convection, interface and heat-flux branches all use it. Slaved cells get 0.
- **rho(T)·cp(T) is used consistently.** One function, `volumetric_heat_capacity`, feeds the conduction step (`:779`), the Robin term (`_step_rho_cp`, `:903-906`), the CFL rate (`:511`) and beta (`solidification_loop.py:1019`). Chill overrides are applied after the curve (`:487-489`).
- **Enthalpy inversion matches fs.** The engine's `SolidificationModel` uses the lever rule by default (`physics/solidification_physics.py:178`), which is the same fs(T) the inversion assumes. The Numba kernel is tested against both the NumPy version and `invert_tet_enthalpy`. Chills start with fs = 1 (`solidification.py:538`), so the inversion leaves them unchanged.
- **Stable time step.** The bound is the exact per-cell rate: conduction plus h·A/(rho·cp·w·dx³), with time-dependent interface HTC included (`heat_transfer.py:492-532`). Conductance factors cap every cell at the full-cell rate (`:1087-1116`). Cells with w = 0 are excluded.
- **Small-cell merge conserves totals.** The slave's capacity weight and area move to the master and the slave's are set to 0 (`core/cut_cells.py:536-541`), so the totals do not change. The master must have phi ≥ phi_min and the same part code. Latent capacity scales with w, so it is conserved too. `MergeTest` asserts the volume and area sums.
- **Interfaces and chills.** Faces between parts, including cast–chill, are left out of A (`cut_cells.py:229-244`). The sand-facing surface of a chill gets the chill HTC (`solidification.py:806-815`). Bodies inside one STL are split, so a shared face counts as an interface.
- **Downstream volumes.** These consumers read per-cell `voxel_volumes`, so they are correct with phi-weighted volumes:
  - shrinkage and defect volumes (`solidification.py:5522`, `:6062`)
  - `get_shrinkage_percentage`, which is volume-weighted when cut cells are on (`:5576`)
  - pour weight (`filling_stage.py:1194`), feeding (`feeding_path.py:451`) and gravity drain (`solidification.py:3236`)

### Findings

**P1. The chunked voxelizer creates duplicate cells when the flag is on.**
- **Location:** `core/cut_cells.py:399-401` together with `core/voxel_mesh.py:1149-1154`.
- **Cause:** The chunked path places centres with `linspace(x_min+s/2, x_max-s/2, nx)` and `nx = ceil(E/s)`. The real spacing is `(E-s)/(nx-1)`, which is smaller than `s` whenever E/s is not an integer. `compute_cut_cell_geometry` maps centres to cells with `rint((c-origin)/pitch)` using `pitch = base_voxel_size`. Once the accumulated offset passes half a cell (k·(1−spacing/s) > 0.5), two input centres map to the same cell index.
- **Effect:** Both copies are kept with phi = 1. That doubles metal volume and capacity on one lattice plane for each affected axis, and puts two coincident cells in the neighbour graph (the zero-offset neighbour is skipped at `:1727`).
- **Reachability:** The "chunked voxelization" checkbox in the desktop app (`ui/main_window.py:715`), `--chunked` in the benchmark, and the fallback when trimesh voxelization fails (`voxel_mesh.py:1097`).
- **Repro:** A 5.03 in box at 0.1 in with `chunked=True` and cut cells on. The grid is 52³ candidates, so the chunked path is used. The total offset is 0.7 of a cell, so duplicates start at k ≈ 37. Expect `len(np.unique(centers, axis=0)) < len(centers)` and a total volume about 2 % over CAD per affected axis.
- **Why the test misses it:** `test_chunked_path_adds_dropped_surface_cells` only catches this if the collision plane happens to fall inside the sphere.
- **Smallest fix (flag-on only, keeps flag-off identical):** in `compute_cut_cell_geometry`, run `in_keys, first = np.unique(in_keys, return_index=True); ijk = ijk[first]` before building `all_keys`. Adjust `n_input`/`n_dropped` to match. Then add a chunked test whose padded extent is not a multiple of the pitch.

**P2. Open (non-watertight) parts are not handled reliably.**
- `split_into_bodies` (`cut_cells.py:203`) splits a part when every face-connected piece has `volume > 0`. For an open piece that volume depends on the coordinate origin, so an open STL with cracks can be split into patches that are each tested alone.
- Open meshes go to `winding_number_contains` (`:112`). That function falls back to the ray test (which its own docstring calls unreliable on open surfaces) above 1e10 point×face pairs (`core/tet_mesh.py:703`). With 2 M-point sampling chunks (`:353`), any open unit with more than about 5 k faces takes the ray test. `_flood_missing_interior` (`:316`) is not chunked at all.
- **Fix:** split only when every piece `is_watertight`. Chunk the open-mesh path so it stays below the pair cap, and log when the fallback happens.

**P2. The interface probe misses clearance gaps between parts.**
- The probe sits 0.02 of a pitch outward (`cut_cells.py:227`), which is 0.002 in at 0.1 in.
- If assembly CAD leaves a clearance gap larger than that between two parts, both faces are counted as sand contact. The lattice still links the cells across the gap with conduction, so the gap both conducts and loses heat to the mold.
- **Fix:** probe at about 0.5 pitch, or treat a sample as an interface when the cell on the far side holds another part.

**P2. Slaved cells are not given their master's initial state.**
- `_initial_thermal_temperatures` (`solidification.py:514-532`) does not copy the master's temperature and fs to slaved cells.
- With a filling-stage initial field where some fs_prev values are above 0, a slaved cell's inversion uses its own fs_prev, so the solidification time reported for it drifts from its master's.
- **Fix:** after the merge, set `T0[slaves] = T0[masters]` and copy fs the same way.

**P2. Surrogate training records do not mark cut-cell runs.**
- `core/surrogate/dataset.py:61-62` records `mesh_type="voxel"` for flag-on runs. The dataset then mixes two kinds of voxel physics, about 15 % apart in total solidification time, with nothing to tell them apart.
- **Fix:** write a `cut_cells` field to the record and the index.

### Notes (not defects)
- **Count-based consumers.** The isolation score (`solidification.py:4260`) and the riser hot-spot cluster volume (`riser_design.py:394-399`) still treat every cell as size³. Hot-spot cells are interior (phi = 1), so the effect is small.
- **Surface sets.** `voxel_moduli` uses voxel-face exposure, while `surface_voxels` uses A > 0. These are different surface sets.
- **beta timing.** beta uses rho·cp at the post-step temperature while the heat step used the pre-step value. The resulting drift is O(dt), measured at 3.8e-4 and bounded at 2e-3 by the test. The comment at `solidification_loop.py:1016` says it is "the capacity the heat step used", which is not exact.
- **Memory at about 2 M cells.** The part counts array is (N, P) uint16, about 68 MB at P = 17, plus one more copy while it is built. It stays on the mesh (`_cut_cell_part_counts`) after part assignment; free it there. The first flood pass builds about 12 M keys. Sampling costs boundary cells × 64 point-in-mesh tests, and its time at 0.1 in on C05813 has not been measured.
- **Stale doc reference.** `cut_cells.py:44` cites `docs/tet_migration/cut_cells.md`, which has not been written yet.

### Commands for the supervisor
1. `git diff b912460~1 b912460 -- physics/heat_transfer.py core/voxel_mesh.py core/solidification.py core/solidification_loop.py` to confirm the flag-off code is untouched.
2. `venv/bin/python -m pytest -q` with the flag off. Then compare flag-off `solidification_times` at b912460~1 and b912460 with `np.array_equal` on thick_thin, l_bracket and the assembly.
3. `CASTSIM_VOXEL_CUT_CELLS=1 venv/bin/python -m pytest -q` for the full suite with the flag on.
4. `venv/bin/python tools/benchmark_cut_cells.py --c05813 "/Users/brendonjarett/Documents/Models/C05813 Rigging" --sizes 0.1` (off and on), plus a run with `--chunked` after the P1 fix.

## Merge verdict: OK with notes
The commit can stay with `voxel_cut_cells` off by default. Do not make it the default until three things are done:
1. The P1 duplicate-cell fix is in, with a chunked test.
2. The C05813 run at 0.1 in meets the 2 % and 1.5x targets.
3. The full suite passes with the flag on.