# Wave 1 solver report (contract C2)

## Files changed

- `physics/tet_heat_transfer.py` (new, 488 lines)
- `tests/test_tet_heat_transfer.py` (new, 13 tests)

No other file touched. Nothing staged or committed.

## Public API as implemented

```python
CG_RTOL = 1e-10; CG_MAX_ITER = 5000; DEGENERATE_VOLUME_REL = 1e-12

@dataclass(frozen=True)
class TetMaterial:                      # validated and coerced to float64 in __post_init__
    temp_points; conductivity; rho_cp; latent_heat_vol: float = 0.0

class TetHeatTransferModel:
    def __init__(self, nodes_m, tets, tet_material, materials, boundary_faces)
    def add_boundary_group(self, face_indices, htc, t_inf, time_function=None) -> int
    temperatures            # property: live (N,) float64 buffer; setter copies into it
    time: float
    def set_initial_temperature(self, value) -> None
    def step(self, dt) -> None
    def node_heat_capacity(self) -> (N,) J/K at current T
    def node_latent_capacity(self) -> (N,) J
    def nodal_gradient(self) -> (N,3) K/m
    def gradient_magnitude(self) -> (N,) K/m
    def get_cooling_rate(self, prev, dt) -> (N,) degC/s, positive when cooling
    def total_energy(self) -> float, sum C_i T_i in J
    def stiffness_matrix(self) -> csr_array, K in W/K at current T   # addition
    last_iterations: int; last_residual: float; backend_description: str
    last_used_direct_solver: bool                                    # addition
    n_nodes, n_tets, tet_volumes (T,) m3, boundary_face_areas (F,) m2  # additions
```

Implementation:
- Init: batched `det`/`inv` of the (T,3,3) edge matrices give |V_e| and P1 gradients (T,4,3). Local `V_e g_a.g_b` stored as (T,16). CSR pattern from `np.unique(row*N+col, return_inverse=True)`; the inverse is the (tet,a,b) -> data-index scatter map. Diagonal positions cached.
- Step: `T_e` = mean of 4 nodal temperatures at step start. `k`, `rho_cp` by `np.interp` per material (loop over materials, not tets; constant curves use a fill). K data refilled with one `np.bincount` over the scatter map. `C/dt + R(t+dt)` added to the cached diagonal. CG (`rtol=1e-10`, `x0=T_old`, Jacobi `LinearOperator`). Falls back to `spsolve` when `info != 0` or the CG result is non-finite. `last_residual = ||b - A x|| / ||b||`.
- Robin groups store unique nodes and lumped area `sum A_face/3`. `h = htc * time_function(t+dt)`, evaluated once per step.
- Installed scipy is 1.18.0: `cg(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)`. `rtol` is used.

## Deviations from DESIGN.md and additions

1. `temperatures` is a property, not a plain attribute. Reads return the live buffer, so in-place edits (latent correction) are used by the next step. Assignment copies into the same buffer, so references held by callers stay valid. `step` writes the solution in place.
2. Nodes not referenced by any tet raise `ValueError`. They would make a zero row. C1 re-indexes nodes after CORE/FILTER removal, so a valid mesh never trips this. The mesh worker should confirm pytetwild output is re-indexed too.
3. Tets with `|V| <= 1e-12 * mean|V|` raise `ValueError`. Either orientation is accepted.
4. Duplicate `face_indices` in one group raise `ValueError`. The same face in two groups is allowed and adds both terms.
5. `h(t)` that evaluates to negative or non-finite raises `ValueError` at step time.
6. CG is capped at 5000 iterations before the spsolve fallback. Scipy's default cap is `10*N`, which on a 200k-node mesh would run millions of iterations before falling back.
7. Properties outside `temp_points` clamp to the end values (`np.interp` behaviour).
8. `total_energy()` uses C at current temperatures. With temperature-dependent `rho_cp` it is not enthalpy. It is meant for conservation tests, as C2 states.
9. Additions beyond C2: `stiffness_matrix()` (tests need K), `last_used_direct_solver`, `n_nodes`, `n_tets`, `tet_volumes`, `boundary_face_areas`.
10. `docs/ARCHITECTURE.md` lists physics imports as stdlib/numpy/numba. DESIGN C2 also allows scipy. `test_architecture_boundaries` checks only project-local imports and passes. The ARCHITECTURE.md table is now out of date. I did not edit it because it is outside my file list.

## Tests

`venv/bin/python -m unittest tests.test_tet_heat_transfer -v`: 13 run, 13 pass, 0.20 s (0.49 s wall).

| Test | Result |
|---|---|
| K symmetric, row sums 0, min eigenvalue >= -1e-10 scale (3-point k curve, random T) | pass |
| Linear field on a randomly perturbed Kuhn mesh: exact gradient, atol 1e-8 K/m | pass |
| Latent capacity sum equals L*V; zero on the non-latent material | pass |
| Adiabatic, 100 steps: max relative energy drift 1.06e-10 (bound 1e-9); field converges to the C-weighted mean within 1e-6 degC | pass |
| Lumped cube, Bi = 1.7e-5, dt 0.5 s, t up to 1.8 tau: error < 1% of (T - T_inf) | pass |
| Slab, Bi = 1, 20 cells, dt 0.01 s, Fo 0.1-1.0: centre error <= 0.04% (bound 2%) | pass |
| dt = 1e4 s, 10 steps: finite, within [T_inf, T0], converges to T_inf | pass |
| `time_function` called only with t+dt (0.5, then 1.0). A switch-on at 1 s has no effect in step 1 and acts in step 2 | pass |
| Two-material bar, k 10/50, h 1e5: flux in and out within 1% of series resistance, interface T within rtol 1e-6 | pass |
| Cooling-rate sign. In-place edits to `temperatures` used by `step` | pass |
| ValueError on bad shapes, non-finite values, index ranges, integer dtype, material count, zero-volume tets, unreferenced nodes, bad material curves, bad group, bad dt | pass |

Other runs:
- `venv/bin/python -m unittest tests.test_architecture_boundaries`: 5 run, 5 pass.
- `venv/bin/python -m unittest tests.test_heat_transfer_physics`: 12 run, 12 pass (existing solver unaffected).
- `venv/bin/python -m mypy --ignore-missing-imports --follow-imports=silent physics/tet_heat_transfer.py`: no issues. The file is not in the `mypy.ini` allowlist.

## Performance (Apple M-series, 18 cores; scipy CG is single-threaded)

The benchmark scripts are in `/tmp/tetbench/` and are not in the repo. The material is a 6-point Al-like curve, with h = 500 on all faces.

Kuhn grid, 32^3 hexes, pitch 2.54 mm: 35,937 nodes, 196,608 tets, nnz 513k. Init 0.22 s.

| dt (s) | median step | CG iters |
|---|---|---|
| 0.1 | 15.4 ms | 34 |
| 1.0 | 29.1 ms | 97 |
| 2.0 | 36.7 ms | 130 |
| 10.0 | 52.8 ms | 205 |

Per-step assembly cost: property interpolation 0.8 ms, K refill 3.6 ms, C 0.9 ms. `nodal_gradient` takes 6.7 ms. All steps converged by CG with residual < 1e-10.

pytetwild mesh of a cylinder (r 40 mm, h 80 mm), edge 2.4 mm: 58,671 nodes, 329,678 tets. Init 0.45 s.

| dt (s) | median step | CG iters |
|---|---|---|
| 0.1 | 28.6 ms | 26 |
| 1.0 | 58.9 ms | 87 |
| 2.0 | 87.0 ms | 149 |

At edge 1.6 mm (633k tets): 72 / 170 / 252 ms per step for the same dt values.

## Open risks

- With Jacobi preconditioning, CG iterations grow roughly with sqrt(alpha*dt)/h. Very fine meshes at dt = 2 s will cost several hundred ms per step. If that matters, the next step is AMG or IC preconditioning; pyamg is not installed.
- On large 3D meshes the `spsolve` fallback is slow and memory-heavy (SuperLU fill-in). It never triggered in any run.
- fTetWild meshes contain obtuse tets and slivers. P1 stiffness on them has positive off-diagonals, so the discrete maximum principle does not hold. Small over- or undershoots near sharp fronts are possible. The bounded-temperature test uses non-obtuse Kuhn tets only.
- Lagged properties at large dt across the liquidus can lag k and rho_cp. C4's dT-based step halving is the guard.
