# Adversarial review — round 5, the `TAU_MINUS_MAX` ghost-mode clamp

Branch `refactor/pouring-seams`, HEAD **`fe866cf`** (round 4, with my five round-4 findings
closed), reviewing the **uncommitted round-5 working tree**: `physics/fluid_flow.py`,
`physics/filling.py`, `tests/test_filling_lattice.py`, `tests/test_filling_benchmarks.py`,
`tests/test_filling_stability.py`, `docs/filling_implementation.md`,
`docs/filling_field_notes.md` (722 insertions). Spec:
`Test/session_diag_2026-09-02/mech3_filter_inlet/REPORT.md`.

Read-only throughout: no file edited, no test run, no simulation run, no `core/` or
`physics/` module imported. Evidence is `git diff`, reading the working tree, and two
pure-NumPy/SciPy scratch programs that re-implement the D3Q19 TRT from scratch
(`/tmp/trt_wall2.py`, `/tmp/trt_nu.py`, `/tmp/checks.py`) — nothing from the repo is
imported by them.

Line numbers are working-tree lines.

---

## FINDINGS

### 1. MAJOR — `physics/fluid_flow.py:187-208`, `:299-300`; `physics/filling.py:1943`, `:1947`, `:2093`
**`tau_minus_max` is a new public parameter of `lattice_units`, but it is not carried on
`LatticeUnits`, and two of the three relaxation paths read the module global
`physics.filling.TAU_MINUS_MAX` instead — so the parameter is honoured in one path out of
three, the documented `float("inf")` escape hatch is a no-op in the production
configuration, and the seam is a live NumPy/Numba parity break.**

Where the clamp comes from, per path:

| path | source of the clamp |
|---|---|
| `step(..., smagorinsky=True, use_numba=False)` — the production default | `physics/filling.py:1943`, module global `TAU_MINUS_MAX`. `units.omega_minus` is **never read** |
| `step(..., smagorinsky=False, use_numba=False)` | `physics/filling.py:1947`, `units.omega_minus` — the only path that honours `lattice_units`' argument |
| `step(..., use_numba=True)` (either `smagorinsky`) | `physics/filling.py:2093`, module global `TAU_MINUS_MAX`; the kernel recomputes `omega_minus` from `tau` and ignores `units.omega_minus` |

Two concrete failures follow.

*(a) The documented escape hatch does nothing.* `TAU_MINUS_MAX`'s own docstring
(`physics/fluid_flow.py:170-172`) and `lattice_units`' (`:259-261`) both say "Pass
`float("inf")` wherever an exact `Lambda = 3/16` is what is being measured", and
`docs/filling_field_notes.md` repeats it. Build
`units = lattice_units(0.0043, 0.006/7200, u_max_physical_m_s=2.8, tau_minus_max=float("inf"))`
— `units.omega_minus` is 5.54e-5, exact Λ = 3/16, as `test_lattice_units_carries_the_clamp`
(`tests/test_filling_lattice.py:309-322`) asserts — then run `step(state, units)` with the
defaults. Both the NumPy path (Smagorinsky on) and the Numba path clamp at 5.0 anyway. The
caller measures the clamped scheme and reports it as Λ = 3/16. That is exactly the mistake
`docs/filling_implementation.md:559-563` warns the Poiseuille benchmark against, and the
warning it gives ("run it at τ = 0.8") only works because at τ = 0.8 the clamp is inert —
the `tau_minus_max` argument itself cannot deliver it.

*(b) A parity break the parity suite cannot see.* With those same `inf` units and
`smagorinsky=False`, the NumPy path relaxes the odd moments at `omega_minus = 5.54e-5`
while the Numba kernel relaxes them at `0.2`. The two step paths are then solving
different schemes — a 3,600x difference in the antisymmetric relaxation rate, i.e. exactly
the difference this whole round is about. `tests/test_filling_parity.py` cannot catch it:
its fixture builds units through `lattice_units(DX_M, WCB_NU, u_max_physical_m_s=2.8)` with
the default, giving τ = 0.5000029370 (verified by arithmetic), so both paths clamp and
agree.

The new tests work around this rather than closing it.
`GhostModeDampingTests._trajectory` (`tests/test_filling_stability.py:1551-1577`) has to
monkeypatch `pf.TAU_MINUS_MAX` *and* pass `tau_minus_max` into `lattice_units`, with a
comment that states the hazard outright: *"The kernel reads the clamp through this module
global, the NumPy path through `units.omega_minus`; both have to move together or the two
paths would be measuring different schemes."* That comment is a description of the defect.

Fix shape: put `tau_minus_max` on `LatticeUnits` and have `step`/`_collide_stream_numba`
read `units.tau_minus_max`, so there is one authority per state, the way `tau` and `magic`
already are.

---

### 2. MAJOR — `physics/fluid_flow.py:103-173` (the constant's own docstring), with `core/filling_stage.py:136`, `:157`
**The clamp's accuracy cost is not "the wall moves off the midpoint"; it is a fixed
`-0.25 cell²` loss of flow-carrying width, which is `-11.1 %` of the flow rate through a
3-cell channel and `-25 %` through a 2-cell one — measured — and the choke-thickness
thresholds those numbers govern were calibrated at Λ = 3/16.**

I solved the steady Stokes-TRT Poiseuille problem exactly as a linear system in pure NumPy
(D3Q19, halfway bounce-back, Guo body force with the `1 - omega_minus/2` prefactor, no repo
imports) and fitted the effective half-width. The solver validates against the textbook
result: with BGK (`Λ = (τ-½)²`) the effective wall lands exactly on the node midpoint at
`τ = ½ + √3/4` (Λ = 3/16) and nowhere else — measured offset `-0.000000` there,
`-0.013` at τ = 0.8, `+0.040` at τ = 1.2.

Measured law, then confirmed analytically to 8 digits at four (τ, τ⁻, H) combinations:

```
(H_eff/2)^2  =  (H/2)^2  +  (4/3) (Lambda - 3/16)
```

so the clamped Λ = 4.6719e-5 at Test 2's τ = 0.500010382 costs a flat **-0.24993771 cell²**,
independent of channel width, against **exactly 0** before this diff (a casting τ had
Λ = 3/16 *by construction*, and I measured offset 0.00000 at τ = 0.500010, 0.5001 and 0.51
under the unclamped relation). Expressed as a relative peak-velocity / flow-rate deficit
`-0.25/(H/2)²`:

| channel (Test 2's own measured thicknesses) | H | flow-rate deficit |
|---|---|---|
| gate | 3 cells | **-11.11 %** |
| runner | 7 cells | -2.04 % |
| sprue | 11 cells | -0.83 % |
| filter | 15 cells | -0.44 % |
| a channel at `MIN_CHOKE_THICKNESS_CELLS` | 2 cells | **-25.00 %** |

Effective viscosity is untouched (fitted `nu_eff` agrees with `(τ-½)/3` to 1.3e-10 under
both Λ, verified) — so the docstring's "What the clamp does *not* change is the fluid
viscosity" is correct, and the entire cost lands on the boundary.

**Concrete scenario.** `Test_Model/Filling_Test`, whose gate `measure_choke_thickness_by_part`
reads at 3 cells, and Test 2, whose gate reads 3.0 cells at 4.3 mm: after the clamp those
gates pass ~11 % less metal for the same head. `core/filling_stage.py:157`'s user-facing
warning says a 3-cell channel is *"the floor at which this solver has ever produced a
usable fill — Test_Model/Filling_Test, whose gate is also three cells, time-caps at 57% of
cells"*. That 57 % datum, and the choice of 3.0 as the MARGINAL threshold and 2.0 as the
hard MIN, were all measured at Λ = 3/16. They are now optimistic by 11 % and 25 %
respectively on exactly the geometries they exist to flag, and nothing in the diff
re-derives them. The direction is consistent with the diagnosis's own measurement
(wet cells `-2.06 %` on Test 2 — less flow, not more).

The diff and `docs/filling_field_notes.md:609+` are honest that this is unquantified
("moves off the node midpoint by an amount nobody has quantified against the Poiseuille
benchmark"), and the spec's §4(c) asked for it to be quantified. It was not; the number
above is that quantification, and it is large enough at the marginal resolutions to need
recording next to those thresholds rather than left as an open item.

---

### 3. MINOR — `physics/fluid_flow.py:284-287`, `:317-340`
**`tau_minus_max` is unvalidated, unlike its sibling `magic_parameter` two lines away, and
values that produce an unconditionally unstable scheme are accepted silently.**

`lattice_units` refuses `dx_m`, `kinematic_viscosity_m2_s`, `u_max_physical_m_s`,
`u_max_lattice`, `magic_parameter <= 0` and `tau <= 0.5`, and `omega_minus_from_magic`
refuses `omega_plus` outside `(0, 2)` — but neither checks `tau_minus_max`. Verified by
arithmetic:

* `tau_minus_max = 0.4` → `omega_minus = 2.5`, outside the TRT stability range `(0, 2)`,
  returned with no error. `collide_trt` over-relaxes the odd moments past the limit and the
  run diverges with no diagnostic.
* `tau_minus_max = 0.5` → `omega_minus = 2.0` exactly, the limit, and
  `guo_prefactor = 1 - 0.5*2 = 0`, which silently zeroes the entire Guo source term.
* `tau_minus_max = 0.0` → `ZeroDivisionError` from inside `omega_minus_from_magic`, a
  traceback rather than the `ValueError` every neighbouring bad input gets.
* negative → negative `omega_minus`.

`lattice_units`' `Raises:` section and `omega_minus_from_magic`'s docstring were not
extended for the new parameter, and `test_omega_minus_stays_in_the_stable_range`
(`tests/test_filling_lattice.py:295-307`) only exercises the default, so the `< 2.0` half of
its assertion is now the only guard and it never sees a non-default clamp.

---

### 4. MINOR — `tests/test_filling_stability.py:907`, `:929`, `:961`, `:967`, `:1048`
**The MECH-1 regression net is now one assertion wide, and the discriminator's primary
bound can pass vacuously.**

The recalibration itself checks out. Measured values from the diff's own docstrings: shipped
rule `+0.055 %`, override `+0.51 %`, bound `1.002` (`+0.2 %`) — the shipped side has 3.6x
headroom and the override overshoots by 2.5x, so the bound genuinely discriminates and is
not tight. Determinism is fine too: `collide_kernel` and `free_surface_kernel` are
`prange(nx)` loops with per-cell writes and no cross-thread accumulator (the only
order-dependent reduction in the module is `promoted_donor_average_kernel`,
`physics/fluid_flow.py:2412-2418`, which this fixture never reaches — it asserts zero GAS
and zero INTERFACE cells), `fastmath` is off, and `advance_mass` sums 18 directions in a
fixed order. A 0.2 % bound on this quantity is safe.

What did degrade is the net's shape. `test_a_free_surface_through_a_filter_stays_bounded` is
explicitly relabelled a non-discriminator ("the two rules now agree to six figures — both
reach 5.0035e-02"), so it would now pass with the porous override reinstated. That leaves
`test_the_deleted_weighting_is_what_grew_it` plus the two `1.002` witnesses as the whole
guard against a reintroduction, on a fixture whose failure mode (runaway) no longer
reproduces at all.

And that test's *primary* assertion is
`assertGreater(broken - seeded, 4.0 * (fixed - seeded))` — a ratio of two differences of
nearly equal numbers. `fixed - seeded` is measured at 2.7e-5 out of 0.05 (5.4e-4 relative);
if a future change makes the shipped rule *decay* the seed instead of growing it,
`fixed - seeded` goes `<= 0` and the assertion becomes trivially true regardless of what
`broken` does. The real anti-rot is the second assertion,
`assertGreater(broken, 1.003 * seeded)`, which is absolute and correct. Worth reversing the
emphasis, or asserting `fixed - seeded > 0` first, so the 4x separation claim cannot become
unfalsifiable.

---

### 5. MINOR — `physics/fluid_flow.py:447-448`, `:826`; `physics/filling.py:1917`; `physics/fluid_flow.py:128-131`
**The diff updated the *mirrors* of two docstrings and left the originals, so
`physics/fluid_flow.py` now contradicts the tests that quote it; plus two number checks.**

* `physics/fluid_flow.py:447-448` (`guo_force_terms`): *"`omega_minus` falls to ~2.6e-4, so
  using `tau` injects roughly 2e-4 of the intended force."* The diff rewrote the two
  docstrings that mirror this sentence — `tests/test_filling_lattice.py:445-452` and
  `tests/test_filling_benchmarks.py:12-16`, both now correctly saying "a tenth" — and left
  the source of truth in the module it edits. Verified: with `omega_minus = 0.2` the buggy
  BGK prefactor injects `1.000208e-01` of `F`, so "a tenth" is right and "2e-4" is now
  wrong.
* `physics/fluid_flow.py:826` (`porous_drag_force`) and its exact mirror at
  `physics/filling.py:1917`: *"With `omega_minus ~ 5.6e-5` the TRT barely relaxes the
  antisymmetric part, so ... the gap `f_asym - e_asym` is frozen in and grows."* The diff
  rewrote the test docstring that mirrors this (`tests/test_filling_stability.py:3433-3441`,
  now: *"`TAU_MINUS_MAX` has since bounded `omega_minus` at 0.2 ... so the freezing is
  gone"*) and left both originals. These two are the *recorded justification* for the
  `f1daf97` explicit drag term, and their stated premise no longer exists — a reader
  auditing whether that term is still needed will now find the source saying yes for a
  reason the clamp removed and the test saying the reason is gone. (The drag should stay;
  §2 and §3b of the diagnosis measure it as the stabiliser. It is the recorded *mechanism*
  that has gone stale.)
* Retained figure: both updated docstrings keep "2e-4" as the pre-clamp mismatch while
  replacing the `omega_minus ~ 2.6e-4` that made 2e-4 self-consistent. On the lattice the
  same docstrings now cite (τ = 0.500010, ω⁻ = 5.54e-5) the pre-clamp figure is
  **4.84e-5**, not 2e-4 — verified.
* `physics/fluid_flow.py:128-131`: *"It takes `tau_eff` above 0.53 before the eddy viscosity
  reaches `tau_minus` ~ 5 on its own."* At `tau_eff = 0.53`, `tau_minus = 6.75`;
  `tau_minus = 5` needs `tau_eff = 0.5416667`, which is the same threshold the docstring
  states correctly seven lines later. True as written, misleading as read.

---

### 6. MINOR — spec §5 not implemented (no file)
**The velocity precursor the spec calls for is absent, and §5(a)'s measurement stands: the
guards that exist give this failure class zero lead.**

The spec was "verdict + §4 fix + §5 precursor". §4 shipped; §5 did not — `git diff --stat`
shows no `core/filling_stage.py`. `docs/filling_field_notes.md` records the deferral and a
defensible reason (`N = 3` was calibrated on 913 cells in one pocket, not 339,824, and a
rising front legitimately accelerates), so this reads as a deliberate scope call rather than
an oversight, and I am reporting it only so it is not lost.

What remains uncovered: §5(a) measured that a precursor on the global `max|u|` is
*structurally* blind here (any setting tight enough to ignore the 123 damped excursions
misses the event entirely, because the sprue's free-fall jet dominates `max|u|` until step
19,960), and the round-4 `eps` band gave 32 steps of warning against the 352-532 steps the
per-cell counter would have given. So any *other* undamped localised mode — the diagnosis
names one, the Smagorinsky-suppressed filter footprint, which `jet_fixture`'s `full_supOnly`
shows is unconditionally fatal at the 0.05 design velocity with no filter present at all —
still has no detector. The clamp removes that specific hazard (see verified-OK), but the
detection gap it exposed is unchanged.

---

## VERIFIED OK (the riskiest things checked that were fine)

**Lens 1 — consistency of the clamp across the three sites**
- Identical formula at all three: `tau_minus = magic/half + 0.5`, then capped, then
  inverted. `omega_minus_from_magic` (`physics/fluid_flow.py:337-340`),
  `smagorinsky_relaxation` (`:1087-1088`, `np.minimum`), `collide_kernel`
  (`:1445-1448`, scalar branch). The `half` floor is `1e-12` in both the kernel and the
  NumPy Smagorinsky path, matching.
- **Convention confirmed as `1/5`, not `1/5.5`.** `tau_minus` is defined as `1/omega_minus`
  (the original `1.0/(magic/half + 0.5)` makes `tau_minus - 1/2 = magic/half`), so
  `TAU_MINUS_MAX = 5` means `omega_minus >= 0.2` and
  `Lambda = (tau_plus - 1/2)(5 - 1/2) = 4.5 * half`. Verified against the diagnosis's own
  validation emulation: `magic = 4.5 * half = 4.6719e-5` reproduces `tau_minus = 5.0`
  exactly, and gives `Lambda = 4.6719e-5` — identical to the shipped clamp. A `1/5.5`
  reading would have given `Lambda = 5.191e-5`; the implementer's correction is right.
- **Clamp applied after the eddy-viscosity adjustment, to the same quantity, in both.**
  In the kernel the `half`/`tau_minus`/cap block sits *outside* the
  `if use_smagorinsky and c1 == 0.0:` branch (`:1431-1448`), so it runs on the finalised
  `omega_plus` whether or not Smagorinsky fired; `smagorinsky_relaxation` computes
  `tau_eff`, applies `suppress`, clips to `[tau, 10]`, then takes `half` from
  `1/omega_plus` and caps. Both derive `half` via the same double reciprocal
  (`1.0/(1.0/tau) - 0.5`), so they are bit-identical rather than merely algebraically equal.
  Filter cells (where Smagorinsky is suppressed) therefore *do* get the clamp on both paths
  — which is what the diagnosis requires, and it closes `jet_fixture`'s `full_supOnly`
  hazard as a side effect.
- Float64 throughout: `float(TAU_MINUS_MAX)` at the kernel call site, float64 numpy arrays
  in `smagorinsky_relaxation`, Python floats in `omega_minus_from_magic`.
- **No unclamped `omega_minus` survives.** Grepped `TRT_MAGIC_PARAMETER`, `magic`,
  `omega_minus`, `tau_minus`, `TAU_MINUS_MAX` across the whole repo excluding `Test/` and
  `venv/`: the only producers are the three sites above. `physics/heat_transfer.py` has no
  TRT at all; there is no GPU/`tools`/`benchmarks` lattice module; the only non-test
  importers of `physics.fluid_flow` are `core/filling_stage.py` (which calls
  `lattice_units` with defaults), `ui/main_window.py` and `core/filter_database.py` (comment
  references only).
- **`LatticeUnits` is only ever built by `lattice_units`** in production
  (`physics/fluid_flow.py:304` is the sole constructor call). The one hand-built instance is
  `tests/test_filling_benchmarks.py:241` via `dataclasses.replace`, at `TAU = 0.8` where the
  clamp is inert (`tau_minus = 1.125`) — so the Poiseuille wall benchmark is measuring exact
  Λ = 3/16, as the diff claims. Nothing pickles or persists `omega_minus`.

**Lens 2 — kernel argument threading / parity**
- `collide_kernel` has exactly one call site (`physics/filling.py:2078-2094`), and
  `tau_minus_max` is passed positionally in the right slot, immediately after `magic` and
  before `rho_out` — matching the signature at `physics/fluid_flow.py:1138-1143`. Grep finds
  no other caller and no test that invokes it directly, so there is no stale-arity call.
- `@njit(parallel=True, cache=True)` with no explicit eager signature, so adding a float
  argument needs no type list; `cache=True` keys on the bytecode, so the on-disk cache
  invalidates itself.
- `float("inf")` is safe in both: `tau_minus > inf` is `False` in the kernel and
  `np.minimum(tau_minus, inf)` is the identity in NumPy — no NaN, no warning.
- With default units (the shipped configuration) all three paths compute the same
  `omega_minus`, so there is no *shipped* parity break; finding 1 is reachable only through
  a non-default `tau_minus_max`.
- The `guo_prefactor = 1 - 0.5*omega_minus` in the kernel and `guo_force_terms`' prefactor
  in NumPy are the same expression on the same clamped `omega_minus`, and the momentum
  identity `omega_minus*F/2 + (1 - omega_minus/2)*F == F` is exact for any `omega_minus` —
  verified numerically at both the clamped and unclamped values, so the clamp does not
  change how much force the body-force term injects, only how it is split between source and
  relaxation. The porous drag enters through that same force and is likewise unchanged in
  net effect.

**Lens 3 — test recalibration**
- The `1.002` bounds discriminate with real margin (3.6x on the shipped side, 2.5x
  overshoot on the override) and are robust: no order-dependent parallel reduction is
  reachable from that fixture, `fastmath` is off, and the seeded value is exact by
  construction. Both step paths are covered, so a one-sided reintroduction of the override
  still fails.
- The anti-rot assertion `broken > 1.003 * seeded` is present and absolute, so the fixture
  cannot quietly lose its ability to fail (finding 4 is about the *other* assertion in the
  same test).
- **`MagicParameterTests` did not lose exact-Λ coverage above the threshold** — it gained it.
  The τ list moved from `(0.5001, 0.51, 0.8, 1.0, 1.9)` to `(0.5417, 0.55, 0.8, 1.0, 1.9)`,
  i.e. it dropped only the two values that are now inside the clamp, and added
  `assertGreater(tau, threshold)` so the list cannot drift back in, plus three new tests:
  the clamp binding at the casting τ, the clamp being inert at τ = 0.6/0.8 with
  `assertEqual(clamped, unclamped)` and the Λ round trip, and the threshold constant itself
  pinned to `0.5416666666666666`.
- **No off-by-epsilon at the threshold.** Verified by exact arithmetic:
  `threshold = 0.5416666666666666`; at exactly that τ the raw `tau_minus` is
  `5.000000000000004`, so the clamp binds and sets `Lambda = 0.18749999999999983` — a 1e-16
  relative perturbation, invisible to the `places=12` round-trip assertions. At
  `threshold + 1e-12` and at `0.5417` the clamp is inert and Λ is 0.1875 to 16 digits. The
  `threshold ± 1e-6` probes in `test_the_clamp_threshold_is_where_it_is_documented` land
  either side with ~5e-5 of margin in `tau_minus`, which is deterministic arithmetic, not
  accumulated error.
- `GhostModeDampingTests`' amplitude projection is correct: with `u = (A·s, -A·s, 0)` the
  estimator `mean(s·(u₀ - u₁))/2` returns `A` exactly, and it is a projection rather than a
  global maximum — which is the right choice, since §3a of the diagnosis is precisely that a
  global maximum cannot separate this mode from the mean flow. The `np.where(wet, seeded,
  f)` broadcast over the leading `Q` axis is correct, the box is asserted to contain no
  INTERFACE and no GAS cell, and the discriminator accepts either growth *or* non-finiteness
  so it cannot be defeated by the unclamped run detonating earlier than expected.
- The `parity` docstring claim that `tests/test_filling_parity.py` runs "at tau = 0.500003,
  deep inside the clamp" checks out: `DX_M = 0.0152`, `WCB_NU = 0.006/7200`,
  `u_max_physical = 2.8` → τ = 0.5000029370300751.

**Lens 4 — physics**
- Λ at Test 2's τ is **4.6719e-5** against 0.1875, exactly as the docstring states.
- The **effective viscosity is unchanged** by the clamp: fitted `nu_eff` from the steady
  Poiseuille profile agrees with `(τ-½)/3` to 1.3e-10 at τ = 0.500010, 0.51 and 0.8 under
  both Λ. So `omega_plus`, the shear viscosity and the Smagorinsky closure are genuinely
  untouched and the whole cost is at the boundary (quantified in finding 2).
- **Nothing outside the choke heuristics numerically assumes Λ = 3/16 wall placement.**
  `tests/test_filling_benchmarks.py`'s `wall_low, wall_high = 0.5, NZ - 1.5` runs at τ = 0.8
  where the clamp is inert (verified inert, and the diff pins the benchmark there in both
  docs and the new lattice test). `mouth_area_m2 = mouth_cells · dx²`,
  `measure_choke_area_cm2` and `_local_thickness_cells`' `2r - 1` are cell counts and
  distance transforms with no Λ term. The one place the ±0.5-cell wall is used as a
  *justification* is `MIN_CHOKE_THICKNESS_CELLS` — finding 2.
- The clamped scheme stays inside the TRT stability box: `omega_plus = 1.999958`,
  `omega_minus = 0.2`, both in `(0, 2)`, with the odd moments now *under*-relaxed
  (per-step factor 0.8) rather than over-relaxed. The diagnosis's 23,284-step run on the
  real assembly with no non-finite cell is consistent.
- The clamp reaches suppressed filter cells on both paths, so the separate hazard the
  diagnosis records (`full_supOnly`: a dissipation-free non-porous patch is unconditionally
  unstable at τ = 0.500010, dying at step 1,402 even at the design velocity, with no filter
  present) is closed by this fix without un-suppressing Smagorinsky — which is what §4(d)
  asks for.

**Lens 5 — docs/claims**
- "a tenth of its true strength" (`tests/test_filling_benchmarks.py:15`,
  `tests/test_filling_lattice.py:448-452`) is **correct**: with `omega_minus = 0.2` the BGK
  prefactor injects `1.000208e-01` of `F`, verified.
- The threshold `0.541667` is correct and is now pinned by a test.
- `"tau_eff` 0.5001 only brings `tau_minus` down to 1,400-4,700"` is consistent
  (τ_eff = 0.5001 → τ⁻ = 1875.5, and 1,400-4,700 corresponds to τ_eff in
  0.50004-0.500135).
- Everything in the `TAU_MINUS_MAX` docstring and the field notes that I could cross-check
  against `mech3_filter_inlet/REPORT.md` matches it: τ⁻ = 18,060, the 31-step alternation,
  `(-a, +a, 0)`, |u| 0.016 → 0.94, step 20,192, 23,284 steps, max|u| 0.052 vs 0.938,
  130 → 0 rows above 0.1, 0.0055 flat, wet mass 0.015 %, wet cells -2.06 %,
  sup|m'/m| = 0.99998, R = 0, max|f| = 6.07, ρ ∈ [1.0000, 1.0121].

**Out of scope, noted once:** `data/surrogate_dataset/` (untracked, 74 samples) has no
solver-version or commit column in `index.csv` — only a timestamp — so post-clamp samples
appended by `core/surrogate/pipeline.py` will be indistinguishable from the pre-round-4 ones
already there. Not caused by this diff, but this diff is the second physics change in a row
that invalidates them.
