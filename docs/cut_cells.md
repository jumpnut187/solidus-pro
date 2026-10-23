# Voxel cut cells: report (branch `feature/tet-mesh`, base b912460, nothing staged or committed)

## What cut cells do (flag `settings.mesh.voxel_cut_cells`, default False)

1. The only flag read is `core/voxel_mesh.py::_cut_cells_enabled()`, called in `AdaptiveVoxelMesh.generate`. Every later branch keys off
   `mesh.cut_cells`, then `engine._cut_cell_merge`, then the `HeatTransferModel` arrays, which stay None when the flag is off.
2. `core/cut_cells.py::compute_cut_cell_geometry` runs after voxelization:
   - STLs are split into closed bodies. Exterior area comes from R2 surface samples (16 per pitch^2) binned per cell. A sample whose
     outward probe lands inside another part is a part-to-part interface, not mold area.
   - Missed surface cells are added and interior holes flood-filled. phi is sampled in boundary cells only (64 R3 points), interior
     cells get 1, and cells with phi = 0 are dropped.
   - Each boundary cell takes the part code holding most of its sub-points. Ties go to the higher code (`_apply_cut_cell_part_types`).
3. Mesh: `voxel_volumes = phi * h^3`, `voxel_surface_areas` = exterior CAD area, plus `voxel_metal_fractions` and `cut_cell_stats`.
4. Small-cell rule (`merge_small_cut_cells`, phi_min 0.25): a cell with phi < 0.25 is slaved to its fullest face neighbour that has the same
   part code and phi >= 0.25. Edge and corner neighbours are tried next. The master takes the slave's phi and area, so totals do not change.
   The slave gets capacity weight 0 and copies the master's temperature after each step. A small cell with no master stays unmerged and is
   kept stable by the conductance cap.
5. Solver (`physics/heat_transfer.py`, `core/solidification*.py`):
   - Capacity is `rho_cp * w * h^3`, with rho(T)*cp(T) interpolated as a product (as in the tet solver).
   - Link i-j carries `min(f_i, f_j) * (a_i + a_j) / 2`, a = clip(w, 0, 1), f capping each cell at the full-cell rate. Symmetric, so
     conservative. Robin and heat flux use `A_i / (w_i h^3)` in place of 1/h, on cells with merged area > 0.
   - Stable dt = 0.9 / max_i(conduction + Robin rate). Latent heat uses the exact enthalpy inversion (no recalescence cap) with
     beta_i = L / rho_cp_i(T). Shrinkage % is volume weighted.
6. Numba kernels added by b912460, each with a NumPy reference: `update_heterogeneous_cut_cell_parallel`,
   `cut_cell_conduction_rates_parallel`, `interpolate_curve_parallel` and `_invert_enthalpy_kernel`.

## Benchmark (tet 0.05 in reference; `cut_cells_benchmark_partial.txt` plus this run)

| case | mesh | size in | cut cells | points | wall s | metal in3 | vs CAD | total s | vs ref | field err | hot spot in | hot spot dist in |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| thick_thin | tet | 0.05 | - | 278918 | 184.4 | 21.002 | +0.0 % | 190.1 | +0.0 % | 0.00 % | (0.00, 0.00, 0.94) | 0.00 |
| thick_thin | voxel | 0.2 | off | 3554 | 0.5 | 28.432 | +35.4 % | 199.3 | +4.8 % | 12.53 % | (0.00, 0.00, 1.00) | 0.06 |
| thick_thin | voxel | 0.2 | on | 2850 | 0.6 | 21.000 | +0.0 % | 192.6 | +1.3 % | 2.02 % | (-0.00, -0.00, 1.00) | 0.06 |
| thick_thin | voxel | 0.1 | off | 23811 | 2.5 | 23.811 | +13.4 % | 194.4 | +2.2 % | 5.17 % | (0.00, 0.00, 0.90) | 0.04 |
| thick_thin | voxel | 0.1 | on | 23811 | 3.3 | 21.000 | +0.0 % | 191.7 | +0.8 % | 0.78 % | (0.00, 0.00, 0.90) | 0.04 |
| thick_thin | voxel | 0.05 | off | 179021 | 18.9 | 22.378 | +6.6 % | 190.2 | +0.0 % | 1.36 % | (-0.05, 0.00, 0.95) | 0.05 |
| thick_thin | voxel | 0.05 | on | 179021 | 23.0 | 21.000 | +0.0 % | 190.7 | +0.3 % | 0.28 % | (0.00, 0.00, 0.90) | 0.04 |
| l_bracket | tet | 0.05 | - | 113522 | 74.8 | 7.000 | -0.0 % | 72.4 | +0.0 % | 0.00 % | (0.72, 1.00, 0.31) | 0.00 |
| l_bracket | voxel | 0.2 | off | 1221 | 0.3 | 9.768 | +39.5 % | 81.2 | +12.3 % | 22.33 % | (0.40, 1.00, 0.40) | 0.33 |
| l_bracket | voxel | 0.2 | on | 1188 | 0.3 | 7.000 | +0.0 % | 74.8 | +3.3 % | 3.34 % | (0.40, 1.00, 0.40) | 0.33 |
| l_bracket | voxel | 0.1 | off | 8946 | 1.0 | 8.946 | +27.8 % | 82.7 | +14.2 % | 16.02 % | (0.40, 1.00, 0.50) | 0.37 |
| l_bracket | voxel | 0.1 | on | 8946 | 1.3 | 7.000 | +0.0 % | 73.1 | +1.0 % | 1.55 % | (0.30, 1.00, 0.70) | 0.58 |
| l_bracket | voxel | 0.05 | off | 63591 | 5.7 | 7.949 | +13.6 % | 76.0 | +5.0 % | 6.18 % | (0.45, 1.00, 0.45) | 0.31 |
| l_bracket | voxel | 0.05 | on | 63591 | 7.0 | 7.000 | +0.0 % | 72.7 | +0.4 % | 0.67 % | (0.30, 1.00, 0.65) | 0.54 |
| assembly | tet | 0.05 | - | 64218 | 120.5 | 5.352 | -0.0 % | 133.7 | +0.0 % | 0.00 % | (-0.17, 0.06, 0.98) | 0.00 |
| assembly | voxel | 0.2 | off | 1056 | 3.7 | 7.728 | +44.4 % | 194.7 | +45.7 % | 34.93 % | (0.00, 0.00, 0.80) | 0.25 |
| assembly | voxel | 0.2 | on | 1020 | 2.5 | 5.277 | -1.4 % | 131.4 | -1.7 % | 5.67 % | (0.00, 0.00, 0.80) | 0.25 |
| assembly | voxel | 0.1 | off | 6979 | 9.0 | 6.374 | +19.1 % | 156.5 | +17.1 % | 10.87 % | (-0.10, 0.00, 0.90) | 0.12 |
| assembly | voxel | 0.1 | on | 6979 | 6.9 | 5.319 | -0.6 % | 134.0 | +0.3 % | 1.83 % | (-0.10, 0.00, 0.90) | 0.12 |
| assembly | voxel | 0.05 | off | 50757 | 43.2 | 5.849 | +9.3 % | 143.3 | +7.2 % | 4.26 % | (-0.15, 0.00, 0.95) | 0.07 |
| assembly | voxel | 0.05 | on | 50753 | 31.6 | 5.332 | -0.4 % | 134.3 | +0.5 % | 0.88 % | (-0.15, 0.00, 0.95) | 0.07 |

C05813 (17 parts, voxel only, default analyses). The reference is the tet run at 0.2 in edge, 748.6 s (`fix_mesher_crash.md`). There is no
0.05 in tet run for this model, and the tet hot-spot position was not recorded.

| size in | cut cells | cells | wall s | peak RSS GB | total s | vs tet | metal in3 (CAD 1695.46) | cast / riser / chill / post vs CAD | cast hot spot in |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | off | 248,529 | 110.7 | 8.53 | 883.4 | +18.0 % | 1926.45 (+13.62 %) | +12.3 / +16.4 / +28.7 / +14.1 % | (11.6, 2.6, 0.2) |
| 0.2 | on | 244,608 | 106.0 | 8.53 | 765.5 | +2.3 % | 1695.29 (-0.01 %) | -0.26 / -0.47 / +0.04 / +0.53 % | (11.8, 2.6, 0.2) |
| 0.1 | off | 1,872,205 | 1241.9 | 27.1 | 822.0 | +9.8 % | 1817.52 (+7.20 %) | +6.9 / +8.9 / +13.9 / +6.8 % | (11.8, 2.6, 0.2) |
| 0.1 | on | 1,856,306 | 982.0 | 27.1 | 755.9 | +1.0 % | 1694.82 (-0.04 %) | -0.35 / -1.08 / +1.48 / +0.86 % | (11.8, 2.5, 0.1) |

- 0.1 in on: 104 cells added, 16,003 dropped (phi = 0), 248,971 sampled, cut-cell pass 20.3 s. Wall on/off: 0.79x at 0.1, 0.96x at 0.2.
  The 0.1 off run shared the machine with about 3 min of my investigation jobs, so its wall time is somewhat inflated.
- Peak RSS 27.1 GB with the flag off and on, reached in trimesh `voxelized()` before cut cells run. One process per run; logs in
  `/tmp/cutcells2/c05813_01_{off,on}.log`.

## Voxelization path on C05813

- The benchmark, the desktop app (the `chunked_voxelization` check box defaults to False) and the web app (`use_chunked_voxelization`
  False) all use `trimesh.voxelized().fill()` plus `_unfill_sealed_pockets`. The grid is 1.51M candidates at 0.2 in and 11.9M at 0.1 in.
  `settings.mesh.use_chunked_voxelization` (True) is not read by these callers.
- Chunked path, C05813 at 0.2 in: out of memory with the flag off (killed at 25 GB RSS in the 17-worker voxelizer, before cut cells;
  the flag-on attempt died at 30.7 GB).
- Reviewer P1, duplicate cells on the chunked path: fixed in `compute_cut_cell_geometry`. Input centres that round to the same lattice key
  are reduced to the first one, in input order (`n_duplicates`, `cut_cell_stats["duplicate_voxels"]`, printed). Flag off is untouched.
  - Repro (5.03 in box, 0.1 in, chunked): 303 duplicates and +0.31 % volume at HEAD; 0 duplicates and +0.075 % with the fix. New test
    `test_chunked_path_off_pitch_extent_has_no_duplicate_cells`.
  - My C05813 runs did not hit it: trimesh path, 0 duplicates at 0.2 and 0.1 in (`/tmp/cutcells2/inv/dup_*.log`). Cell counts and
    volumes are unchanged, and the fix is a no-op when there are no duplicates, so the C05813 rows stand without a re-run. The small
    cases also ran on the trimesh path. The -2.8 % gap is unrelated (next section).
  - Found while testing, not fixed (it would change flag off): the chunked voxelizer keeps only 7,803 of about 127k cells for that box
    (bottom 3 z-slices). `_process_z_slice_batch_parallel` silently counts failed slices, or the GPU containment is wrong. Cut cells
    flood-fill the rest (125,151 added).

## Volume gap (-2.8 % at 0.2 in): bookkeeping, not a defect

- 1,743.46 in3 is the CAD volume of all 17 parts including the 48.0 in3 of chills. CAD metal (cast + riser + post) is 1,695.46 in3. The
  cut-cell metal volume of 1,695.29 is -0.01 %, and the union of all cells, 1,743.31 in3, is -0.009 %.
- All 17 parts are watertight, single-body and positive-volume. Pairwise overlap is 0 (Monte Carlo over every bbox intersection,
  `/tmp/cutcells2/inv/cad.py`). The trimesh path adds 24 cells and drops 3,945 with phi = 0. Merging keeps the totals.
- Per-code errors (up to ±1.5 % at 0.1 in) come from interface cells: a cell carries one code. At 0.1 in, 0.71 in3 of metal is labelled
  chill (chill +1.48 %, metal -0.04 %). The cause is the dominant-code rule plus ties going to the higher code (chill 2 > cast 0).
- Fix made: the `--c05813` record now prints `cad_metal_in3` and `metal_vs_cad` (metal codes only).

## Flag-off bit identity

- New `FlagOffBitIdentityTest` runs the `tests/test_simulation_scenarios.py` cast + chill + riser scenario with `voxel_cut_cells=False`
  and with a `settings.mesh` proxy that raises AttributeError for the flag (getattr default). All 73 arrays (engine, mesh and heat
  model ndarrays, plus dict-held ones) match by dtype, shape and bytes.
- One-off check against the pre-cut-cell code (`git archive 8a74a31`, `/tmp/cutcells2/inv/dump.py`): 73 arrays, 0 differing bytes.

## Tests (`venv/bin/python -m unittest <module>`, all OK, 0 skipped)

Counts (re-run after the P1 fix): test_cut_cells 26 (6 new), parallel_kernel_parity 10, heat_transfer_physics 12,
heat_transfer_numba_backend 3, thermal_coupled_integration 10, thermal_boundary_config 8, simulation_scenarios 16,
architecture_boundaries 5.

All four kernels already had a parity test in b912460. New in `tests/test_cut_cells.py`: the P1 chunked test,
`FlagOffBitIdentityTest` and `CutCellKernelParityTest`, which covers:
- update and rate kernels on 6-, 21- and 26-slot stencils (odd corner remainder included);
- curve interpolation at breakpoints, nextafter neighbours, NaN and ±inf;
- `HeatTransferModel` dispatch with `NUMBA_AVAILABLE` True vs False (40 steps: curves, time-dependent Robin, slaves);
- end-to-end `_run_thermal_coupled_integration` with a cut-cell engine, all four availability flags patched.

Other changes: the P1 dedupe (`core/cut_cells.py` +17/-4, `core/voxel_mesh.py` +2/-1, flag-on only) and `tools/benchmark_cut_cells.py`
+4 (`cad_metal_in3`, `metal_vs_cad`). No solver code changed.

## Risks

- The C05813 reference (tet, 0.2 in edge) is itself coarse. At 0.2 in, cut cells miss 2 % on small cases (l_bracket +3.3 %, assembly
  field error 5.67 %); the targets are at 0.1 in.
- Interface labelling: up to 1.5 % of a part's volume can carry a neighbour's code; metal labelled chill over-chills slightly.
- Full suite not run with the flag on or with `CASTSIM_MESH_TYPE=tet` (RESUME step 3).
- Flag-off problems: 27 GB peak at 0.1 in (trimesh voxelization), and the chunked path both runs out of memory on C05813 and drops slices.

## Recommendation: make `voxel_cut_cells` default True

Every owner target is met at 0.1 in:

| check | target | measured |
|---|---|---|
| total vs tet 0.05 | within 2 % | thick_thin +0.8 %, l_bracket +1.0 %, assembly +0.3 % (off: +2.2 / +14.2 / +17.1 %) |
| field error | < 2 % | 0.78 / 1.55 / 1.83 % |
| wall time vs flag off | <= 1.5x | 1.32x / 1.30x / 0.77x on the small cases, 0.79x on C05813 |
| flag off | bit-identical | 73 of 73 arrays identical to 8a74a31 |

- C05813 vs tet 0.2: +1.0 % (0.1 in) and +2.3 % (0.2 in), against +9.8 % and +18.0 % off. Metal volume within 0.04 % of CAD.
- Before flipping: full suite with flag on and with tet, desktop check on C05813, decide whether chill/metal ties go to metal.
