# Adversarial review — uncommitted fix for MECH-1 (porous pair weight) and MECH-2 (torus lattice)

Branch `refactor/pouring-seams`, HEAD `f1daf97`, reviewing the **uncommitted working tree**
(`core/filling_stage.py`, `physics/filling.py`, `physics/fluid_flow.py`, three test modules,
two docs). Strictly read-only: no file was edited, no test or simulation was run, and no
`core/` or `physics/` module was imported. Evidence below comes from `git diff`, reading the
working tree, the two diagnosis reports, the in-flight validation logs under
`Test/session_diag_2026-09-02/validation/`, and small pure-NumPy/scipy scratch scripts run
against the saved `*.types.npy` grids (`/tmp/pad_audit.py`, `/tmp/halo_audit.py`,
`/tmp/vent.py` — masks re-implemented locally from the part-code tables, nothing imported).

Line numbers are working-tree lines.

---

## FINDINGS

### 1. MAJOR — `core/filling_stage.py:1993-2002` + `:2674-2680`, `:2754`
**The headspace exclusion only covers the inlet mouth's own columns, but the inlet's
`ensure_interface_skin` halo is an 18-direction dilation — so a metal top-face column
*beside* the mouth hands the halo an open headspace cell, and the stage grows exactly the
wet-cells-above-the-domain-outside-the-ledger that `HEADSPACE_CELLS`' docstring says it
prevents.**

`open_above = metal[:, :, z_top]` is cleared only at `inlet[on_mouth, 0], inlet[on_mouth, 1]`
(line 1998) — the mouth's own `(x, y)`. `inlet_halo` (line 2674-2679) is
`|= np.roll(inlet_mask, c)` over all 18 lattice directions, minus `inlet_mask` and
`~obstacle`. The five `c_z = +1` diagonals land at `(x±1, y, z_top+1)` and
`(x, y±1, z_top+1)`, i.e. in the **first headspace plane over the neighbouring column**. If
that neighbouring column is metal on the original top face, `obstacle &= ~headspace`
(line 2002) has just opened it, so it survives `inlet_halo &= ~obstacle` and lands in the
halo. Line 2754 then calls `ensure_interface_skin(state, inlet_halo)` on every step the halo
has a GAS cell, which flips it to `FLAG_INTERFACE` and hands it a full equilibrium population
set averaged from its donors — the FLUID inlet, at the inlet density and the inlet's
*downward* velocity (`physics/filling.py:1112-1116` → `_seed_promoted_cells`).
`create_state`'s own `_dilate(fluid)` interface layer (`physics/filling.py:413-415`) seeds the
same cells at `t = 0`, before step 1.

Those cells sit above `z_top`, so they are outside `metal` (not in `metal_full_mass`, not in
`sum(state.mass[metal])`, not in `reached`) **and** outside `core`, so `to_voxels` crops them
out of every result field. Their mass is counted in `poured` when the pin replaces it, and
never appears anywhere else: the reported `% by mass` silently under-reads, and a spurious
source injects downward momentum into the casting's top plane from cells that do not exist.
The pool is bounded (`HEADSPACE_CELLS = 2` planes over the affected columns, laterally walled)
so this is not a MECH-2-scale divergence — but it is the same mechanism, re-opened, in the one
place the fix explicitly claimed to close.

Secondary: when `delivery_layers` comes back empty (mouth sitting on a floor), the back-pressure
probe is `thx, thy, thz = (hx, hy, hz)` (line 2735-2737) — the halo — so those phantom cells'
density feeds the head boundary's `rho_throat` and mis-throttles the rate controller.

**Concrete failure scenario, measured on the real Test-2 solve grid**
(`Test/session_diag_2026-09-01/r3_t2_ledger.csv.types.npy`, `(135,135,94)`, padded to
`(137,137,98)`; mouth = largest 8-connected component of the pour part on the top plane, halo =
the 18-direction roll, exactly as the stage builds them):

```
pour placement        mouth   headspace cols   faces sealed   halo & headspace
sprue top (shipped)     132            1836           True                  0
riser top               1801            167           True   35  (all over CAST columns)
cast top                   5           1963           True    6  (all over RISER columns)
```

So the pour point the validation suite is running (the sprue mouth) is clean, and the defect is
invisible to it. Move the pour location onto the riser top — a legal placement that
`_expand_inlet_to_mouth` explicitly supports ("a bare casting over its own top face") — and 35
INTERFACE cells are created at padded `z = 95`, one plane above the geometry, on step 1 and
re-created every step thereafter. Same geometry, same code, different click.

Fix shape: clear `open_above` over the mouth's 8-neighbourhood in `(x, y)` (or, equivalently,
`headspace &= ~inlet_halo_footprint`), not just over the mouth itself.

---

### 2. MINOR — `core/filling_stage.py:2010-2023`
**The new closed-lattice invariant `raise`s a `RuntimeError` out of `run_filling_stage`, where
every other unrunnable-geometry condition in this function `emit()`s and returns `None`.**

`core/simulation_pipeline.py:574` calls `maybe_run_filling` with no `try/except`, and
`core/surrogate/pipeline.py:201` likewise, so the exception tears down the whole run after the
meshing stage has already been paid for, with no `FillingResult` and no message the UI can
present. Compare lines 1934-1939 ("pour location does not resolve"), 1987-1989 ("no metal
cells"), 1848-1852 — all emit-and-skip.

I could not construct an input that reaches it: `open_above` is `False` on the pad ring by
construction (the ring is `EMPTY_CELL`, so `_metal_mask` is `False` there), and the two planes
the check reads on the `z` axis (`[:, :, 0]` and `[:, :, -1]`) are pure pad. It is reachable
only by setting `MOLD_PAD_CELLS = 0`, which turns the guard from an assert into a crash on the
very configuration it is meant to describe. Severity is minor because of the unreachability, not
because the crash is benign.

---

### 3. MINOR — `physics/filling.py:398`, `:410`, `:415`
**`create_state` now reads `zone.porosity_f64`, which caches on first access and is invalidated
only on a *shape* change — so a caller that fills in `porous.porosity` after building the state
now silently solves with porosity 1 everywhere.**

`PorousZone.porosity_f64` (`physics/filling.py:208-215`) refreshes only when
`cached.shape != self.porosity.shape`. Before this diff `create_state` stored `porous` without
touching the float64 view, so the first access — and therefore the cache fill — happened inside
the first `step()`, after any caller-side mutation. Now the cache is populated at construction.

Every in-tree caller happens to set `porosity` before `create_state`
(`core/filling_stage.py:884` inside `build_porous_zone`; `tests/test_filling_parity.py:137`;
`tests/test_filling_stability.py:836`, `:982`, `:1090`, `:3153`), so nothing is broken today.
The failure it newly admits: `state = create_state(obstacle, porous=zone)` followed by
`zone.porosity[band] = 0.85` gives a state whose `fluid_fraction`, `apply_transitions` threshold,
`heal_submerged_interface` capacity and both exchange paths all see porosity 1 while
`state.porous.porosity` (float32, used by the stage's inlet pin at line 2474 and the delivery
layers at line 2712) sees 0.85 — an inconsistent capacity, in the direction that reads a full
filter cell as 18 % over full, i.e. straight into finding-adjacent territory with the new
`FLUID_FILL_FRACTION_MAX` guard.

---

### 4. MINOR — `tests/test_filling_stability.py:2206-2232`, `:2536-2563`
**The two raised `_tower` time caps are attributed entirely to the removed floor
recirculation, but the dominant change is that `_tower` is a 2x2 column and the pad has just
given it no-slip bounce-back walls on all four sides — a 2-cell channel, which this module's own
`MIN_CHOKE_THICKNESS_CELLS = 2` calls unresolvable.**

`_tower(nx=2, ny=2, nz=...)` (line 68-74) is fully dense metal: before the pad `_obstacle_mask`
found nothing solid, so the lattice was periodic in **x and y as well as z**. The pad does not
merely add a floor; it wraps the 2x2 metal channel in halfway bounce-back, which is exactly the
geometry the stage warns about ("Halfway bounce-back seals it: this grid cannot resolve the
{label}", line 2643-2652). A 7.2x slowdown on `nz=40` (0.25 s -> 1.80 s) is what a 2-cell
no-slip channel costs; removing a recirculation path does not slow a fill by 7x.

Consequence: `MassProgressReportingTests` now completes at 1.80 s against a 2.0 s cap — 11 %
headroom on a viscosity-dominated fixture at the sealing threshold, so the next small change to
the collision or inlet path flips it from `complete` to `time_capped`. And both docstrings now
tell a future reader that these fixtures "fill honestly from a real floor upward" when what they
actually measure is creep through a channel the production code refuses to trust.

(For the record, the caps are *not* hiding a production regression: `a1_fgt20` and `a1b_fgt17`
both reach `status=complete` in the in-flight validation logs, and the 1230-test suite in
`validation/suite.log` has exactly one failure —
`SubmergedInterfaceHealingTests.test_a_deeper_tower_leaves_none_either`, `'diverged' != 'complete'`
— which is byte-identical to the pre-existing failure at HEAD in
`Test/session_diag_2026-09-01/r3_suite.log:9325`.)

---

### 5. MINOR — `core/filling_stage.py:3121-3128` (with `:331-332`)
**The new `[-0.1, 1.1]` FLUID band is a per-step `Δρ/ρ` meter, not a fill-fraction meter, and a
legitimate hydraulic transient can trip it into `STATUS_DIVERGED` with results discarded.**

`state.rho` is the *start-of-step* density (`physics/filling.py:114`) while `state.mass` is the
end-of-step value, and under the now-universal weight-1 rule a FLUID cell's mass increment is
exactly its own `Δρ`. So for a FLUID cell `eps = 1 + Δρ/ρ` almost identically — the band is
"no FLUID cell may change density by more than 10 % in one step". The calibration cited in the
constant's docstring (1-2 %, `eps_max` 1.006-1.022) was measured on Test 2 *before the front
reached the filter*, i.e. on a quiescent field; the same file's negative-population comment
records a healthy `Full_Gating_Test` run whose density spread reaches **0.347** (line 3063).

Concrete scenario: a front free-falling down a tall sprue slams into a closed gate or a
just-sealed pocket. The stagnation cell's density jumps >10 % in one 200-step window, the guard
fires, `status = STATUS_DIVERGED`, `frames.clear()`, and the pipeline hands solidification a
discarded fill and tells the user "the mass bookkeeping having come apart" about a physically
correct water hammer. Nothing in the 1230-test suite or the two completed validation models
trips it, so this is a residual-risk flag rather than an observed failure — but the band is
narrower in the quantity it actually measures than its docstring claims.

---

## VERIFIED OK (the riskiest things checked that were fine)

**Fix 1 — parity and correctness**
- Kernel `physics/fluid_flow.py:1591-1626` and NumPy `physics/filling.py:753-769` now implement
  the identical rule: weight 1 if either cell is FLUID, else `0.5*(clip(eps_A)+clip(eps_B))`,
  both sides clipped to `[0,1]`, `eps = mass/(rho*porosity)` with the same `denom > 0 -> else 0`
  fallback that `fluid_fraction` uses. Same direction order (`i = 1..Q`), same float64
  accumulation of float32 populations, same `wet & neighbour_wet` gate, same
  `continue`-on-non-wet-neighbour (obstacle and gas directions skipped) in both.
- The kernel receives `state.porous.porosity_f64` and `state.rho` (float32 on purpose,
  `physics/filling.py:1858-1865`), which is exactly what `fluid_fraction` divides by — no new
  dtype asymmetry.
- No dangling references: `eps_here`, `eps_raw`, `por_nb`, `porous`, `any_porous` are gone from
  both files with no remaining reader (grep-verified across `physics/`).
- Every capacity expression still carries the porosity factor and now agrees with `create_state`:
  `fluid_fraction` (`:485`), `apply_transitions`, `heal_submerged_interface` (`:1027`), the stage's
  inlet pin (`:2473-2475`), the top-up `room` (`:2786-2793`), `metal_full_mass` (`:2643-2649`).
- `porosity < 1` can only occur on `PART_TYPE_FILTER` cells (`build_porous_zone` writes
  `zone.porosity` only on filter components, `:884`), and `_resolve_inlet_cells` filters through
  `_metal_mask`, which excludes FILTER — so the velocity-mode pin's porosity-free
  `inlet_full_mass = REFERENCE_DENSITY` (`:2456-2459`) can never sit on a porous cell and cannot
  false-positive the new guard. (That also makes the head-mode comment's stated hazard moot.)
- Choked-outlet physics: with weight 1 a porous FLUID cell's `mass` telescopes to
  `rho(t0)*p + rho(t) - rho(t0)`, so `eps = 1 + (1-p)*Δrho/(rho*p)` — 0.35 % at `p = 0.85` and a
  2 % back-pressure rise, bounded and non-amplifying, well inside the new band. At `p = 1` it
  reduces exactly, and the deleted branch was already guarded by `if any_porous`, so a
  filter-free model is byte-identical (asserted by the new bit-identity test against an
  independent Table-1 transcription).
- `advance_mass` monkeypatching in the new tests works because `physics.filling.step` resolves
  `advance_mass` through the module globals.

**Fix 2 — pad index bookkeeping**
- `metal` preserved exactly and all six faces sealed, measured on the real grid:
  `339,824 == 339,824`, `faces_sealed=True` for every pour placement tried.
- Riser-top vent detection unchanged: `_static_vent_mask`'s `above == EMPTY_CELL` finds
  **2,247** riser vents on both the unpadded and the padded Test-2 grid (measured), because the
  headspace cells and the pad are both `EMPTY_CELL`.
- The 9 result gathers all go through `to_voxels`; `grep` for `gather_to_voxels(` /
  `_upsample(` in `core/filling_stage.py` finds no site outside the helper (only its own
  definition at `:538` / `:1773`). `to_voxels` crops with `core` *before* `_upsample`, so the
  coarsening-1 identity path and the `mapping.shape` assertion both see the original shape.
- Coordinate spaces are consistent: `_resolve_inlet_cells` and `_expand_inlet_to_mouth` run on
  `solve_core` (mapping coordinates) and the result is shifted `+ pad_lo` once, on all three axes,
  before anything indexes a state array. `ix,iy,iz` (`:2198`), `inlet_mask`, `inlet_halo`,
  `delivery_layers`, `static_vent`, `seed`, `out_of_band`/`solve_types[out_of_band]` are all in
  padded space; `measure_choke_area_cm2(solve_core, ...)` and `choke_area_fine_cm2(fine_types, ...)`
  are deliberately unpadded and mutually consistent.
- `filter_zones`, `build_porous_zone`, `htc_dense` and `metal`/`obstacle` are all built *after*
  the pad from the padded `solve_types`, so the porosity/drag/htc arrays line up with the
  solver's grid; both `ndimage.label` calls see the same padded mask, so zone ids still match
  between the two functions.
- `_local_thickness_cells` (`:1176-1204`) already pads the region with one cell of background
  ("the lattice ends in obstacle, not in more metal"), so `measure_choke_thickness_by_part` and
  `choose_fill_coarsening` — which run on the *unpadded* grid, before the pad — were measuring
  the walled geometry all along. The pad makes the measurement and the solve agree for the first
  time; nothing needs shifting there.
- Neighbour-style ops in the stage are `grep`-clean: only one `np.roll` (`inlet_halo`, `:2676`),
  and `_static_vent_mask`'s `above` (`:920`) and `delivery_layers`' `_below` (`:2705`) are
  explicit slices, not rolls, so neither wraps onto the pad.
- `z_centres_in` (`:2503-2510`) subtracts `pad_lo` so index `pad_lo` is original `z = 0`; only
  `metal` and `iz` index it and both live in the core, so the `+z` headspace asymmetry never
  reaches a physical height. `height_m` uses `mapping.shape[2]` (fine grid), untouched.
- Headspace geometry is exactly as documented: with `pad_lo = 1`, `HEADSPACE_CELLS = 2` the
  padded `nz = core_nz + 4`, the headspace occupies indices `core_nz+1, core_nz+2`, and index
  `core_nz+3` stays pad — the outermost plane is solid. `open_above` is `False` on the whole
  pad ring, so the `x`/`y` faces stay obstacle across the headspace planes too.
- Vent/entrapment semantics are unchanged for core cells: `boundary_mask[core]` reproduces the
  old mask exactly, `&= ~obstacle` removes only cells that could never have been GAS, and
  `|= headspace` only adds cells that `to_voxels` crops out of the reported counts.
  `find_entrapped_gas` is 6-connected and non-periodic, so a filled top-face column leaves its
  2 headspace cells as an isolated GAS component — which `boundary_mask` covers, so no spurious
  entrapment.
- A riser column can never be wrongly excluded from the headspace: the exclusion is keyed on the
  `(x, y)` of inlet cells *at* `z_top`, and those cells' type at `z_top` is by construction the
  poured feature's own, so no riser column is cleared by a mouth of a different part.
- The abort sets `status = STATUS_DIVERGED` and `diverged_at_step = taken` in the same shape as
  the existing mass-ceiling and negative-population aborts, so `frames.clear()`,
  `FillingResult.trustworthy`, the pipeline's `_filling_summary` and the solidification handover
  all behave identically.
- `MOLD_PAD_CELLS` / `HEADSPACE_CELLS` / `_domain_boundary_mask` / `_static_vent_mask` /
  `_obstacle_mask` / `_metal_mask` have no production consumer outside `core/filling_stage.py`
  (grep across `core/`, `web/`, `physics/`), so nothing else needed the pad offset.

**Tests**
- `tests/test_filling_parity.py` changes are comment-only; no assertion, tolerance or regime was
  weakened (`DIVERGENCE_REGIMES` still carries both wet-filter variants).
- The new tests discriminate: `PorousPairWeightTests` fails on HEAD (the override grows a
  5.000e-2 seed to 5.21e-2 against a `< 1.01 * seeded` bound) and carries its own
  anti-rot check (`test_the_deleted_weighting_is_what_grew_it` reimplements the HEAD rule and
  asserts the fixture still fails under it);
  `test_a_prefilled_filter_starts_exactly_full` fails on HEAD at `eps = 1/0.85`;
  `test_an_over_full_fluid_cell_aborts_the_run` needs a constant HEAD does not have;
  `PeriodicWrapTests.test_the_stage_never_builds_an_unwalled_grid` and all of
  `MoldPadTests` fail on HEAD by construction. `MoldPadTests` captures `create_state`'s own
  `obstacle` argument rather than re-deriving it, and
  `test_the_riser_top_keeps_an_open_headspace` does genuinely discriminate the inlet-mouth
  exclusion (SPRUE *is* in `METAL_PART_CODES`, so without the exclusion the fixture would show
  8 open cells, not 4).
- `_inlet_cells`' `+ MOLD_PAD_CELLS` shift is correct for its only consumer
  (`test_the_pinned_inlet_keeps_its_interface_skin`, `:2403`), which indexes live `state` arrays.

**Not a regression (checked and dismissed)**
- The `divide by zero / overflow / invalid value encountered in matmul` warnings at the top of
  `validation/b_t2.log` are pre-existing: the same three appear at HEAD in
  `Test/session_diag_2026-09-01/r2d_t2.log:368-372` and `r3_eps.log:299-303`.
- The casting floor being marked a vent by `boundary_mask` while the pad now walls it is
  physically wrong but bit-identical to HEAD's behaviour, so it is not this diff's defect.
