# Tet thermal physics and numerics review (waves 1-3)

I found no P0 or P1 issues. The FEM solver, the enthalpy inversion, the step control and the boundary-group mapping all check out. There are 3 P2 findings and 1 docstring fix. I did not run any tests.

## Checked and correct

- **P1 assembly, orientation and volumes** (`physics/tet_heat_transfer.py`):
  - Volume is `|det J|/6` (`:225`) and the shape-function gradients come from `inv(J)^T` (`:233`). Both work for either tet orientation.
  - The element stiffness is `k·V·∇Na·∇Nb`, in W/K with node coordinates in metres.
  - The lumped capacity uses `rho_cp·V/4` (`:375`). The Robin term uses `A/3` per face node (`:345`).
  - h is evaluated at `t+dt` (`:446`). The CG solve falls back to `spsolve` on any `info != 0` or non-finite result (`:476`).
- **Units:** nodes are converted from inches with `inches_to_meters`. The gradient factor K/m → °C/mm is `0.0254/25.4 = 1e-3` (`core/tet_thermal.py:70`), the same as the voxel path's `/1000`.
- **Enthalpy inversion** (`core/tet_thermal.py:259`): I re-derived the mushy formula `T = (T_post·dT + β·T_p)/(dT + β)` from `h = T + β(1−fs)`. The three branches meet continuously at `T_p` and `T_sol`. These edge cases hold:
  - β = 0 gives T_post.
  - fs_prev = 1 gives T_post.
  - T above liquidus gives T_post.
  - A reheated mushy node keeps its fs.
  - `fs_new ≥ fs_prev`, and the lever formula matches the Numba commit kernel (`physics/solidification_physics.py:660-672`).
- **Fraction solid and time tracking:** whenever fs increases, `T_new ≤ T_prev`, so the Niyama cooling rate stays positive. A liquid-to-solid crossing within one step sets start = t − dt, as on the voxel path. `SolidificationModel.time` and `heat_model.time` add the same sequence of accepted steps.
- **Step rejection:**
  - A rejected step restores the temperature buffer and the model time (`:393`). The solidification model is not touched before the accept decision.
  - The loop always ends: a step at or below 1e-3 s is always accepted, and every step advances time by more than 1e-9 s.
  - The last step is clipped to `max_time`.
- **Termination and outputs** match `_run_thermal_coupled_integration`: sentinels, the end-of-run NaN pass, feeding fraction solid, peak cooling rate and the 1 % snapshots.
- **Boundary groups** use the same h0 and floors per part type as `_configure_thermal_boundaries` (`core/solidification.py:697-777`): cast 500, riser_post 350, riser 220, flow path 400, core contact 500, chill `chill_ambient_htc` with floor 0.7.
  - Chill tets use `k = 1.17e-4·rho_cp` with no latent heat. Chill nodes start pre-solid at the chill temperature.
- **Initial temperature from filling:** unfilled metal nodes are seeded at mold temperature, so no NaN reaches `set_initial_temperature`.

## Findings

### 1. P2: Nodes on the cast/chill interface start at pouring temperature, which preheats part of the chill
- **Where:** `core/tet_thermal.py:207-210` together with `core/solidification.py:504-521` and `:1628-1632`. `tests/test_tet_thermal.py:273` asserts this behaviour.
- **Cause:** a node on the interface counts as metal, because the metal code wins at shared nodes. Only nodes whose tets are all chill are set to `chill_temperature`. The interface node's lumped capacity also includes `rho_cp_chill·V/4` from every chill tet around it, and that share starts at the pouring temperature.
- **Size (my estimate, not measured):** that share is about `A·h/2` of chill volume. On the wave 2 case (2×2×0.5 in chill, h = 0.2 in) that is about 0.4 of 2 in³, so roughly 20 % of the chill's heat capacity starts about 1,500 K too hot. The error halves with h.
- **Why it matters:**
  - The voxel path starts every chill voxel cold, so chill strength on tets depends on mesh size.
  - The wave 2 h-convergence (124.5-125.8 s) is a total time set by the riser, so it does not test cooling near the chill.
- **Fix:** before `set_initial_temperature`, set each metal node that touches a chill tet to the capacity-weighted mean `(C_metal·T_init + C_chill·T_chill)/C_i`. Compute the per-node chill capacity with a `bincount` over chill tets. Alternatively, add an h-convergence test on cast solidification times near the chill.

### 2. P2: Niyama G at chill-face nodes is averaged with the chill's much smaller gradient
- **Where:** `core/tet_thermal.py:451` → `physics/tet_heat_transfer.py:409-417`.
- **Cause:** the nodal gradient is the volume-weighted mean over all incident tets, chill tets included. Heat-flux continuity gives `G_chill = G_metal·k_m/k_c`, which is about 0.17·G_metal for WCB (about 25 W/m/K) against graphite (149.5 W/m/K).
- **Effect:** a CAST node on a flat chill face, with about half its incident volume in chill, reports G ≈ 0.58·G_metal. Niyama scales with G, so it is about 40 % low on exactly the nodes a chill is meant to sound.
- **Contrast:** at the same face the voxel path takes the steep slope from its own temperature to the chill voxel's (`core/solidification_loop.py:143-147`).
- **Fix:** add an optional tet mask to `nodal_gradient` that weights metal tets only, and use it for the Niyama capture and for `thermal_gradients`.

### 3. P2: The step controller grows the step right after a rejection, so it keeps rejecting
- **Where:** `core/tet_thermal.py:392-395` and `:410`.
- **Cause:** after a rejection at dt, the next accepted step grows by 1.25 from 0.5·dt. The step crosses the limit again after about 3-4 accepted steps.
- **Evidence:** wave 2 measured 131 rejected out of 543 attempted solves (`docs/tet_migration/wave2_engine.md:60`), so about 24 % of solves are thrown away.
- **Startup:** the first step on any chill or misrun case is halved down to the 1e-3 s floor (about 7 wasted solves), then takes about 34 steps to grow back to 2 s.
- **Fix:** skip the growth on the first accepted step after a rejection. Or set the retry step to `step_dt·clip(0.9·max_delta_t/max_change, 0.2, 0.5)`. Results are unchanged; this only saves time.

### 4. P2: The module docstring claims exact conservation
- **Where:** `core/tet_thermal.py:15` says `sum(C_i T_i + L_i(1−fs_i))` "is conserved on every step size".
- **Problem:** β uses C(T_post) (`:385`) while the heat step used C(T_old). DESIGN records 0.55 % drift with the real curves.
- **Fix:** say the conservation is exact only for constant properties.

## Residual risks (not shown as defects)
- If CG fails on a large mesh, the `spsolve` fallback could run out of memory. I did not show that CG can fail on fTetWild meshes.
- On misrun nodes (seeded at mold temperature) the first step jumps them by β ≈ 390 K and sets fs = 1. The voxel path does the same, so the tet path inherits it rather than causing it.

## Tests for the supervisor to run
`venv/bin/python -m unittest tests.test_tet_heat_transfer tests.test_tet_thermal tests.test_thermal_coupled_integration`

Merge verdict: OK with notes