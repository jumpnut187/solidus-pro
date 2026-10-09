# Resume point (paused by owner, travel)

Branch `feature/tet-mesh`. Two commits saved at pause:
1. Voxel default restored + tet auto-fallback (lane complete; report `voxel_default_fallback.md`).
2. Cut cells WIP behind `settings.mesh.voxel_cut_cells` (default False, flag off = old voxel path).

## Cut cells: status at pause
- Implemented: `core/cut_cells.py` (phi = metal fraction per voxel, A = mold-contact area),
  solver weighting in `physics/heat_transfer.py`, exact enthalpy inversion and rho(T)*cp(T)
  when the flag is on (approved by owner), `tests/test_cut_cells.py`, `tools/benchmark_cut_cells.py`.
- Benchmarks done (see `cut_cells_benchmark_partial.txt`), voxel vs tet 0.05 in reference:
  | case | 0.1 in off | 0.1 in on | field err on |
  |---|---|---|---|
  | thick_thin | +2.2 % | +0.8 % | 0.78 % |
  | l_bracket | +14.2 % | +1.0 % | 1.55 % |
  | box+riser+chill | +17.1 % | +0.3 % | 1.83 % |
  C05813 at 0.2 in: off 883.4 s (+18.0 % vs tet 748.6 s, 111 s wall), on 765.5 s (+2.3 %, 106 s wall).
- Not done: C05813 at 0.1 in (off/on), worker report `cut_cells.md`, kernel parity checks for
  any new Numba kernels, full suite with the flag on.

## Next steps
1. Re-run `tools/benchmark_cut_cells.py` for C05813 at 0.1 in (off and on).
2. Review `core/cut_cells.py` + solver diff; confirm flag-off bit-identical (full suite).
3. Integration: if targets hold (<= 2 % total, < 2 % field error at 0.1 in, <= 1.5x run time),
   set `voxel_cut_cells` default True; full suite with voxel and with CASTSIM_MESH_TYPE=tet;
   `npm run build`; docs (AGENT.md, ARCHITECTURE.md).
4. Desktop check: C05813 on voxel with cut cells on.
