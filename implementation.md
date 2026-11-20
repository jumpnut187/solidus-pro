# Solver Performance: Measurements, Changes, Next Steps

This file started as a plan to move the thermal time loop to a Rust extension.
The measurements below changed the decision: the kernels stay in Numba, and the
time went into removing the overhead around them. This file records what was
measured, what was changed, how it was checked, and what is left.

Baseline: branch `riser-sleeve-rebased`, commit `1b7180f`, measured 2026-11-20
on the M5 Pro (12P + 6E, 48 GB), project venv, Numba 0.68 on the workqueue
threading layer.

The work has two tiers:

- **Exact changes** (section 4) keep every result bit for bit.
- **Fused kernels** (section 5) change the last digits. They are on by default
  and `CASTSIM_FUSED_KERNELS=0` turns them off, which gives the exact tier.

---

## 1. Decision: no Rust extension

The main stencil kernel (`update_heterogeneous_cut_cell_parallel`) and the
property kernel (`compute_diffusivity_from_curves_parallel`) were translated to
Rust with rayon and timed on arrays dumped from a real run (99,508 cells).

| Kernel | Threads | Numba | Rust |
|---|---:|---:|---:|
| Cut-cell stencil update | 17 | 0.277 ms | 0.299 ms |
| Cut-cell stencil update | 1 | 1.25 ms | 2.93 ms |
| Diffusivity from curves | 17 | 0.153 ms | 0.029 to 0.057 ms |
| Diffusivity from curves | 1 | 0.279 ms | 0.196 ms |

- On the stencil, which is where the compute is, Rust was not faster and was
  2.3x slower on one thread. The Rust was safe code and not tuned.
- Rust was 3 to 5 times faster on the cheap kernel. That gap is thread-pool
  launch cost, not code quality (section 3.1), and it can be removed inside
  Numba.
- Rust differed from the Numba `fastmath` stencil in 14 of 99,508 cells
  (largest gap 2.3e-13 K). Numba compiled without `fastmath` gives the same 14
  cells, so that difference is `fastmath`, not the language.

A Rust backend would have added a build step and a second implementation of
every kernel for no gain on the expensive kernel.

---

## 2. Where the time went at the start

Case: `tests/ci_models/cast.stl` plus `riser.stl`, alloy WCB, 0.15 in voxels,
cut cells on, auto mold on. 99,508 metal voxels, 149,489 graded sand cells,
4,326 outer steps, 2,219 s simulated.

| Item | Time | Share of wall |
|---|---:|---:|
| Whole run | 39.8 s | 100 % |
| Time loop | 31.6 s | 79 % |
| Mesh and geometry | 5.4 s | 14 % |
| Post analyses | 2.6 s | 6 % |
| Compiled Numba kernels | 16.3 s | 41 % |
| Python and single-thread NumPy | about 23 s | 59 % |

Average busy cores: 3.1 of 18. Kernel launches: 71,380, about 16.5 per outer
step.

---

## 3. Findings

### 3.1 A parallel launch has a fixed cost

Every Numba parallel region costs a fixed amount to launch, whatever the array
length: 0.065 ms at 2 threads, 0.070 ms at 4, 0.081 ms at 8, 0.107 ms at 17.
An element-wise pass over 100,000 cells takes 0.012 ms on one thread and
0.110 ms through the pool. Masking the pool to one thread does not help: the
launch still costs 0.068 ms.

Consequences:

- On a 13,000-voxel mesh most kernels cost more to launch than to run.
- A cheap kernel should run on the calling thread unless the mesh is large.
- Two kernels that read the same field should be one kernel.

### 3.2 Allocation is not a cost

A kernel that returns a new array and one that writes into a preallocated
array take the same time (0.137 ms both). macOS reuses large blocks.

### 3.3 Work was repeated

- `volumetric_heat_capacity` ran 5.5 times per outer step. Three of those were
  at the same field.
- The casting-to-mold coupling evaluated conductivity over the whole field
  (99,508 cells) to read 15,493 of them, and searched a sorted index of
  149,489 cells on every call for cell sets that never change.
- The mesh build slotted neighbors with one Python call per voxel.
- The loop rewrote six result arrays and a gradient field every step. They
  are read only after the loop.
- The stable-timestep rates and the temperature update each swept the same
  14 links of every cell.

### 3.4 A fresh Numba compile can change the last bit

The unmodified code gives final temperatures up to 1.6e-11 degC apart on the
100k case when Numba compiles the kernels in the running process instead of
loading them from its cache. Every other output is equal. Runs on a warm cache
are identical to each other. A bit-for-bit comparison must use a warm cache on
both sides (section 7).

### 3.5 `fastmath` buys almost nothing on the stencil

The cut-cell stencil without `fastmath`: 0.283 ms against 0.270 ms on 17
threads, 1.29 ms against 1.17 ms on one. Without it the kernel gives the same
bits at every thread count and from a serial compile.

### 3.6 Rules for a serial copy of a kernel

`physics/kernel_dispatch.py` compiles a kernel a second time without the
thread pool (a serial twin). Three rules, all tested:

1. Only a kernel compiled without `fastmath` may be twinned. The twin then
   gives the same bits. `serial_twin` refuses a `fastmath` kernel.
2. The twin is compiled with `error_model="numpy"`. Inside `prange` a division
   by zero gives inf or NaN. A plain `njit` function raises. The full suite
   caught this: `SolidificationModel.update(T, 0.0)` raised in the twin.
3. The twin gets its own name, because the Numba cache is keyed by function
   name and not by compile options.

The twin is chosen when cells times cost per cell is below the launch cost at
the current thread count (`runs_serial`).

---

## 4. Exact changes (bit for bit)

| # | Change | Where |
|---|---|---|
| 1 | Neighbor slotting in one vectorized pass per batch | `core/voxel_mesh.py` `_build_neighbor_graph`, `_canonicalize_pairs` |
| 2 | Serial twins for the cheap kernels | `physics/kernel_dispatch.py`, `physics/heat_transfer.py`, `physics/solidification_physics.py` |
| 3 | Result arrays, latest temperature and gradient fields written once after the loop | `core/solidification_loop.py` `_write_final_thermal_products` |
| 4 | Conductivity evaluated only at the coupled cells, once per cell. Tabulated-body membership and cell volumes kept per cell set | `physics/heat_transfer.py` `conductivity_at`, `_alpha_at`, `_rho_cp_at`, `_tabulated_membership`, `_cell_set_volumes`. `physics/mold_coupling.py` |
| 5 | The stability check hands its heat-capacity field and link conductances to the first substep | `physics/heat_transfer.py` `step`, `_step_internal` |
| 6 | Diffusivity and heat capacity from one kernel launch | `compute_diffusivity_and_capacity_parallel` |
| 7 | Enthalpy-table lookups start from a hint instead of a binary search | `_interp_scalar_from`, `correct_tabulated_parallel`, `shift_enthalpy_parallel` |
| 8 | Interface-link exchange in three ordered kernels | `interface_link_conductance_kernel`, `interface_link_rates_kernel`, `apply_interface_links_kernel` |
| 9 | Mold coupling selections worked out once | `physics/mold_coupling.py` `CoupledMoldSolver` |
| 10 | Cut cells no longer build a scalar heat-capacity field that is then discarded | `physics/heat_transfer.py` `_step_internal` |

The per-voxel functions `_canonicalize_neighbor_order` and
`_canonicalize_batch` are unchanged. The first is the reference the vectorized
routine is tested against. The octree branch still uses the second.

The NumPy reference kernels used when Numba is absent are unchanged.

---

## 5. Fused kernels (match to rounding)

On by default. `CASTSIM_FUSED_KERNELS=0` (read once at import, in
`physics/kernel_dispatch.py`) runs the exact path of section 4, which stays in
the code as the reference.

| Fused kernel | Replaces | Why the digits move |
|---|---|---|
| `cut_cell_flux_parallel` with `apply_cut_cell_flux_parallel` (`physics/heat_transfer.py`) | The stable-timestep rates kernel and the stencil update: one sweep of the 14 links gives both | Compiled without `fastmath`. The rates keep the old bits, so the stable step and the substep count do not change |
| `invert_enthalpy_and_commit_parallel` (`physics/solidification_physics.py`) | Heat capacity at the post-step field, the enthalpy inversion and the solidification update: three launches | The inversion was a `fastmath` kernel |
| `couple_pairs_kernel` (`physics/mold_coupling.py`) | The NumPy pair arithmetic of `CoupledMoldSolver._couple` | Own interpolation in place of `np.interp`. Sums in the same order |
| `_least_squares_gradients_kernel` (`core/solidification_loop.py`) | One `np.linalg.lstsq` call per voxel for the Niyama gradient | Normal equations by Cholesky. A voxel whose stencil does not span three directions goes to the old function |

Scope:

- The stencil and end-of-step kernels apply to cut-cell models only. The
  flag-off voxel path (no cut cells) is not fused and stays bit for bit.
- The end-of-step kernel applies with the lever rule. Other fraction-solid
  methods use the old path.
- The stencil kernel applies to the 26-slot neighbor layout.

Measured difference from the original code, ten cases, 956 output arrays:

- 938 arrays are bit-identical, including every solidification time,
  shrinkage, hotspot, feeding and Niyama array and every integer and boolean
  array.
- 18 arrays differ:
  - Final temperature fields (float64): largest gap 2.5e-11 degC, 1.8e-14
    relative.
  - Fraction-liquid history (float32): 1 to 11 entries per case out of up to
    10 million, one unit in the last place.
  - Peak cooling rate (float32): 1 entry in one case, one unit in the last
    place.

---

## 6. Results

Wall time of `run_pipeline`, one run each, warm Numba cache, same machine.

| Case | Voxels | Original | Exact tier | Fused (default) | Ratio |
|---|---:|---:|---:|---:|---:|
| Plate and riser, 0.15 in, graded mold, cut cells | 99,508 | 39.4 s | 25.9 s | 21.4 s | 1.8x |
| Plate and riser, 0.30 in, graded mold, cut cells | 12,955 | 6.0 s | 3.4 s | 2.8 s | 2.1x |
| Plate and riser, 0.30 in, no mold | 12,955 | 2.25 s | 1.49 s | 1.25 s | 1.8x |
| Plate and riser, 0.30 in, no cut cells | 12,987 | 4.8 s | 2.8 s | 2.6 s | 1.9x |
| L bracket, 0.20 in, uniform mold | 1,188 | 0.53 s | 0.31 s | 0.26 s | 2.0x |
| Thick-thin, 0.20 in, no mold, no cut cells | 3,554 | 0.47 s | 0.23 s | 0.19 s | 2.5x |
| Block, chill, core, riser, 0.10 in | 22,178 | 31.1 s | 17.4 s | 14.9 s | 2.1x |
| Block, chill, core, 0.10 in, no mold | 11,681 | 6.8 s | 3.6 s | 2.9 s | 2.3x |
| Block, chill, 0.10 in, no mold, no cut cells | 10,471 | 7.4 s | 4.5 s | 4.4 s | 1.7x |
| Block, chill, riser, 0.15 in, uniform mold, no cut cells | 7,091 | 15.4 s | 8.2 s | 8.0 s | 1.9x |

The fused tier was estimated to bring the 100k case near 15 s. It reached
21 s. The estimate counted the Python glue as removable and it mostly was, but
60 % of the loop is now the kernels' own compute (section 8).

The filling solver was not touched and not measured.

---

## 7. How it was checked

1. `tests/verify_solver_snapshot.py` runs the ten cases above and saves every
   numeric array of the engine, the mesh and the heat model (89 to 100 arrays
   per case). The reference was saved from the unmodified code at `1b7180f`.
2. Exact tier: with `CASTSIM_FUSED_KERNELS=0` the final code compares
   bit for bit with the reference on all ten cases, NaN equal to NaN.
3. Fused tier: the final code compares within `--rtol 1e-8 --atol 1e-9` on all
   ten cases. Section 5 gives the actual gaps.
4. Both sides used a warm Numba cache (section 3.4).
5. Unit tests:
   - `tests/test_kernel_dispatch.py`: each serial twin against its parallel
     kernel, the property kernel against the two it replaces, the hinted
     lookup against `np.searchsorted`, division by zero, the size rule.
   - `tests/test_fused_kernels.py`: each fused kernel against the path it
     replaces, and a whole run with the switch on against the same run with
     it off.
   - Two cases in `tests/test_neighbor_canonicalization.py`.
6. The full regression suite, `python -m unittest discover tests`: 2,024 tests
   pass, 20 skipped.

Not run: mypy. It fails locally on a stub and Python-version mismatch, the
same way on the original code.

To repeat the check after a solver change:

```bash
CASTSIM_FUSED_KERNELS=0 python tests/verify_solver_snapshot.py save /tmp/before
# change the code
CASTSIM_FUSED_KERNELS=0 python tests/verify_solver_snapshot.py save /tmp/warmup
CASTSIM_FUSED_KERNELS=0 python tests/verify_solver_snapshot.py save /tmp/after
python tests/verify_solver_snapshot.py compare /tmp/before /tmp/after
# and the fused tier against the same reference
python tests/verify_solver_snapshot.py save /tmp/fused
python tests/verify_solver_snapshot.py compare /tmp/before /tmp/fused --rtol 1e-8 --atol 1e-9
```

Not covered by the ten cases: the filling stage, the octree neighbor branch,
interface links with temperature factors, and the schedule solver modes.

---

## 8. What is left

Measured on the 100k case with the fused kernels: whole run 20.8 s, time loop
15.5 s, of which compiled kernels 9.3 s.

| Item | Cost now | Note |
|---|---:|---|
| Fused stencil sweep and apply | 3.3 s | The actual work |
| Mold kernels (property update, stencil, enthalpy correction, heat deposit, links) | about 2.9 s | One mold step per simulated second |
| Property kernel | 1.3 s | Two launches per outer step |
| Coupling: kernel and two conductivity lookups | about 1.5 s | |
| External sink, boundaries | about 1.2 s | NumPy scatter per substep |
| Loop body bookkeeping | about 1.0 s | About 25 short NumPy passes per step |
| End-of-step kernel | 0.7 s | |
| Post analyses | 2.6 s | Not looked at. `core/feeding_path.py` is most of it |
| Mesh and geometry | 2.5 s | |

Small meshes with many substeps (chill cases) are bounded by one stencil
launch per substep: 0.11 ms for a 7,000-cell mesh.

Options that would go further, each with its cost:

1. **OpenMP threading layer for Numba.** Its threads spin between regions, so
   the launch cost should drop. `libomp` is not installed and the layer does
   not load. Not tested.
2. **Fewer threads for small meshes.** Launch cost falls from 0.107 ms at 17
   threads to 0.065 ms at 2. The fused kernels give the same bits at any
   thread count, so this is now safe for cut-cell runs.
3. **Stable timestep less often.** The rates sweep runs every outer step. A
   bound reused over several steps changes the substep schedule, which is a
   numerical change and needs validation against the thermal benchmarks.
4. **Post analyses and mesh build.** 5 s of the 21 s, not profiled in detail.
5. **Filling solver.** The filling field notes (`docs/filling_field_notes.md`
   at commit `1b7180f`) record runs of 37 and 58 minutes. Profile it with the
   method of section 2 before any work.
6. **GPU.** The remaining cost is stencil compute on 100k to 250k cells. The
   compute scaling plan (`docs/compute_scaling_plan.md` at `1b7180f`) measured
   the Metal path against the CPU.

---

## 9. Rules for further work

- Measure before changing. Two expected costs here were not costs (allocation,
  Rust code generation) and one unexpected one was (launch).
- A performance change keeps results bit for bit on the
  `CASTSIM_FUSED_KERNELS=0` path, or it is reviewed as a physics change.
- Compare on a warm Numba cache.
- Do not twin a `fastmath` kernel. Do not add `fastmath` to a twinned or fused
  kernel.
- A sum that NumPy takes in input order (`np.bincount`, `np.add.at`) stays in
  that order when it moves into a kernel. A parallel loop would reorder it.
- Run the full suite. It found a defect the ten cases did not.

---

## 10. When to look at Rust again

- If the solver is to ship without Numba and llvmlite.
- If a GPU backend is wanted and wgpu is preferred over Taichi or MLX.

For speed alone on this CPU path, the measurements in section 1 do not support
it.
