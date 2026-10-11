# Adversarial review — the three surviving rounds-1–2 findings (F1 choke bodies, F2 erased path, F3 cascade persistence)

Branch `refactor/pouring-seams`, HEAD **`37a9ad8`** (round 5, with my six round-5 findings
closed), reviewing the **uncommitted working tree**: `core/filling_stage.py`,
`core/simulation_pipeline.py`, `tests/test_filling_stage.py`,
`tests/test_filling_stability.py`, `docs/filling_field_notes.md` (+870/−58; `physics/`
untouched).

Read-only throughout: no file edited, no test run, no simulation run, no `core/` or
`physics/` module imported (a Test 2 pour is running detached). Evidence is `git diff`,
reading the working tree, and pure-NumPy/SciPy scratch programs that re-implement
`coarsen_part_types`, `_local_thickness_cells`, `measure_choke_thickness_components` and
`erased_metal_path_bodies` from the source text (`/tmp/f1f2.py`, `/tmp/e2.py`, `/tmp/e3.py`,
`/tmp/e5.py`, `/tmp/cost.py`) — nothing from the repo is imported by them.

Line numbers are working-tree lines.

---

## FINDINGS

### 1. MAJOR — `core/filling_stage.py:1553` (`erased_metal_path_bodies`)
**The severing test is "did this body lose the metal path in *every* block", so a body that
keeps a single metal-path block anywhere is never flagged — and a body severed in its
*middle* is exactly that. Measured: a grid whose metal path is one connected component on
the fine grid and two on the solve grid, with `erased_metal_path_bodies` returning `{}` and
no thickness guard firing either.**

`if not np.any(np.isin(solve_types[cx, cy, cz], _METAL_PATH_CODES)): gone += 1` is evaluated
over the whole body's block set at once. That detects *total* erasure and nothing else. A
wide, thin gate plate with a thick boss or fillet somewhere along it — the geometry
`measure_choke_thickness_by_part`'s own docstring says Test 2's gates have ("a single fat
cell where each gate meets the runner set the maximum for the entire plate") — keeps the
boss's blocks as GATE while the thin span goes to CORE. One surviving block, `gone` stays 0.

**Measured, `/tmp/e5.py`** (factor 2; a thick RUNNER, one GATE body consisting of a
1-fine-cell-thick plate plus a chunky boss, CORE elsewhere, CAST downstream):

```
fine GATE 3180 -> solve GATE 385
erased():                       {}                       <-- not flagged
solve components:               gate (3.0, 5.0)  runner (3.0,)
sealed check (0 < t <= 2.0)?    gate False, runner False <-- no refinement forced
marginal warning (t <= 3.0)?    gate True,  runner True
FINE  grid: metal-path components = 1, runner<->gate connected = True
SOLVE grid: metal-path components = 2, runner<->gate connected = True (but the
            path from the runner through the plate to the casting is gone)
```

**Concrete failure scenario.** That assembly is meshed, `fill_grid_coarsening = 2`,
auto-resolution on (the default). `blocked()` returns `None` (nothing erased, nothing at or
below 2.0), so the stage solves the severed grid and the only thing the user hears is the
generic *"the gate resolves to 3.0 cells across … treat the flow through the gate as
marginal"* — the identical warning a perfectly healthy 3-cell channel gets. The fill then
reports a misrun on a path the solve grid never had, which is verbatim the outcome the
function's own docstring calls "the worst shape a defect can have here".

The exact test is cheap and is already implied by the measurement above: label the metal
path on the fine grid and on the solve grid and compare — the component count goes 1 → 2 at
precisely the severing, and it also catches every total-erasure case F2 currently catches
(verified: `/tmp/e3.py` shows 1 → 2 for the partial case, `/tmp/f1f2.py` §E shows the total
case). Two extra `ndimage.label` calls, ~0.14 s on a 13.7 M-cell grid (§ verified-OK).

---

### 2. MAJOR — `core/filling_stage.py:1456` (26-connectivity in `measure_choke_thickness_components`)
**26-connectivity fuses two same-code bodies that touch at a single corner, which reinstates
the exact "healthy body outvotes a sealed one" defect F1 was written to remove — and the
docstring's justification for choosing it ("over-splitting would invent thin components out
of corner contacts and cry wolf") is measurably false, because the thickness map is global
over the whole metal path and does not change when a body is split.**

**Measured, `/tmp/f1f2.py` §C** — two GATE plates, one 8 cells thick and one 2 cells thick,
sharing exactly one corner:

```
6-connectivity  bodies -> (1.0, 7.0)   thinnest 1.0  => hard warning + refine to 1
26-connectivity bodies -> (7.0,)       thinnest 7.0  => silent
```

So the shipped connectivity reports 7.0 cells for a pair that contains a sealed 2-cell
plate. That is the same number the pooled median used to report (5.0 in the diff's own
fixture) and the same silence.

**And the stated cost of 6-connectivity does not exist.** Because
`_local_thickness_cells` is computed once over `np.isin(dense_types, _METAL_PATH_CODES)` and
each body only *indexes* into that map, splitting a body cannot make its pieces read thin.
Measured, `/tmp/f1f2.py` §D:

```
4-cell-thick diagonal staircase channel: 26-conn -> 1 body, thinnest 3.0
                                          6-conn -> 1 body, thinnest 3.0   (identical)
1-cell diagonal string (corner-linked):   26-conn -> 1 body,  thinnest 1.0
                                          6-conn -> 20 bodies, thinnest 1.0 (identical)
```

Neither connectivity cries wolf; the only difference is that 26 hides the thin plate in §C.
A corner-only contact is also not a channel the solver can carry flow through — one diagonal
D3Q19 link — so fusing across it both hides the thin body *and* asserts a channel exists
where there is none.

**Concrete failure scenario.** Two gates whose voxelisation leaves them diagonally adjacent
(routine after `coarsen_part_types`, which the diff's own F2 docstring notes makes "two gates
a couple of cells apart legitimately *merge* into one body at coarsening 2"): one is 6 cells
thick, one is 2. The stage reports 5–7 cells, emits no warning, does not refine, and the
metal cannot cross the thin gate. `tests/test_filling_stage.py:934-945`
(`test_a_corner_touch_is_one_body`) asserts the fusion but uses two *equally healthy* 4×4×4
cubes, so it never exercises whether the fusion hides anything.

---

### 3. MAJOR — `core/filling_stage.py:2197-2210` and `:2246-2262` (both new message paths are dead)
**`CoarseningDecision.erased_parts` can only be non-empty when `choke_auto_resolution` is
*off*, so the new "the {part} disappeared from the grid entirely" cause and the refusal's
auto-resolution-enabled arm are both unreachable — and on the default path, a refinement
forced by an erased body reports the wrong reason.**

Traced through `choose_fill_coarsening` (`:1717-1763`):

* the early return keeps `coarsening == configured`, so `refined` (`:1625-1628`,
  `coarsening != configured`) is `False`;
* every return inside the loop has `erased == {}` — either because `blocked(m) is None`
  (which requires `m.erased` empty, `:1702`) or because `factor == 1` and
  `erased_metal_path_bodies` returns `{}` for `factor == 1` by construction (`:1531-1532`).

Therefore **`refined is True ⟹ erased_parts == {}`**, and **`erased_parts` non-empty ⟹
`not enabled`** (`configured == 1` also forces `erased == {}`). The diff's own test encodes
this: `tests/test_filling_stage.py:1091`
(`test_the_decision_refines_away_from_the_severed_grid`) asserts
`self.assertEqual(decision.erased_parts, {})` after a refinement.

Three consequences:

* **`:2197-2199`** — `erased_at_configured = coarsening_decision.forced_by in
  (coarsening_decision.erased_parts or {})` is inside the `if coarsening_decision.refined:`
  block, where `erased_parts` is always `{}`. So it is always `False` and the entire
  `cause` branch at `:2205-2210` is dead code.
* **What the user gets instead**, on the default path: `forced_by_thickness =
  base.by_part.get(forced_by, 0.0)`, and an erased code has zero cells so
  `measure_choke_thickness_components` skips it and it is absent from `by_part` — the value
  is **0.0**. The emitted line reads *"refining the fill grid -- the gate measured 0.0 cells
  across at coarsening x2, at or below the 2 cells a channel needs before it has an
  interior"*, for a gate that has no cells at all. The correct sentence exists eight lines
  away and cannot fire.
* **`:2258-2260`** — the refusal's `else` arm, *"Reduce fill_grid_coarsening or the voxel
  size."*, is selected when `coarsening_decision.enabled` is `True`, which cannot co-occur
  with a non-empty `erased_parts`. Dead. `tests/test_filling_stage.py:1176` only asserts the
  reachable `not enabled` arm (`assertIn("choke_auto_resolution", …)`).

`CoarseningDecision.erased_parts`' docstring (`:1590-1594`) says "Non-empty means the metal
path is severed on `solve_types` and the caller must refuse rather than solve it" without
saying that this can only happen with auto-resolution disabled — which is the fact that
makes both new messages unreachable.

---

### 4. MINOR — `core/filling_stage.py:1465-1471`, `:1706-1712`
**There is no minimum body size, so one stray metal-path body now dictates the grid for the
whole assembly: it forces coarsening down to 1 and then cannot be resolved, at ~8× the cells
and a shorter timestep, where the pooled median used to dilute it.**

`bodies.append(max(0.0, float(np.median(thickness[body]))))` runs for every labelled
component regardless of size, and `blocked()` refines whenever *any* body of *any*
metal-path code is at or below 2.0. A body of a handful of voxels always measures 1.0
(measured: a single cell, a 1-cell plate and a 2-cell plate all read **1.0**,
`/tmp/f1f2.py` §A), so it always trips the hard minimum, and refining cannot fix it because
it is still 1.0 at coarsening 1 — the loop then falls through to `resolved=False` and the
run proceeds at the finest grid under the "cannot be trusted" warning.

**Concrete scenario.** A gate that voxelises into a main plate plus a small fragment
separated by one core cell (so 26-connectivity does not fuse them), on an assembly whose
pooled median was 5.0. Before: coarsening 2, 217 k cells on Test 2's mesh, no warning.
After: coarsening 1, 1.71 M cells (8×) *and* a proportionally shorter timestep, plus a hard
warning that no setting can clear. The intended two-gate case is indistinguishable from the
fragment case because both are "a body at 1.0".

Mitigation is a size floor (ignore bodies below a few cells, or require a body to carry some
minimum cell count before it can force the grid), or reporting the fragment as a modelling
warning rather than a resolution one. Nothing in the diff bounds it.

---

## VERIFIED OK (the riskiest things checked that were fine)

**F1 — correctness**
- **The claimed discriminator reproduces exactly.** Two GATE plates 6 and 2 cells thick
  sharing the code: pooled median **5.0** (above `MARGINAL_CHOKE_THICKNESS_CELLS`, so the
  pre-diff code said nothing at all) against thinnest body **1.0** (at or below
  `MIN_CHOKE_THICKNESS_CELLS`, so it hard-warns and refines). Measured, `/tmp/f1f2.py` §B —
  identical to `tests/test_filling_stage.py:900-925`.
- **`_local_thickness_cells` never returns 0.0 for a non-empty body** — a single cell, a
  1-cell plate, a 2-cell plate and a 1×1×N pencil all read 1.0 (measured, §A). So the
  thinnest-body rule cannot produce a value that the `0.0 < value` filter in `blocked()`
  and the `thickness <= 0.0: continue` guard in the warning loop would silently drop. Both
  guards are now dead but harmless, and a body touching the pad is not special — the
  function pads its own bounding box with one background cell regardless.
- **`by_part` and `components_by_part` are measured on the same grid**, the unpadded
  `coarsen_part_types(fine_types, factor)` inside `choose_fill_coarsening`, which runs
  *before* the mold pad is applied. The stage reads both from `coarsening_decision`, so the
  thickness numbers, the body counts, the warning labels and the emitted per-body list all
  come from one measurement. `by_part[code] == components_by_part[code][0]` by construction
  (`:1682`, `:1417-1420`).
- **Thresholds still agree at exactly 2.0 and 3.0.** `blocked()` uses
  `0.0 < value <= MIN_CHOKE_THICKNESS_CELLS`; the warning loop uses `<= MIN` for the hard
  warning and `elif <= MARGINAL` for the soft one. At exactly 2.0 the grid refines *and*
  hard-warns; at exactly 3.0 it does not refine and warns softly; and because a chosen grid
  satisfies `blocked() is None`, only the soft warning is reachable after a successful
  refinement. Body medians can be half-integers (a median of {1, 2}), which the `<=`
  comparisons handle.
- **No consumer disagrees about body counts in a way that breaks.**
  `build_filter_zone_specs` and `build_porous_zone` keep the default 6-connectivity, so on a
  corner-touching pair of inserts the filter specs would report 2 zones while the choke
  message says "the filter" (singular). That is a reporting mismatch only — nothing keys a
  zone id off the choke body count, and the zone ids are still derived consistently between
  those two functions. (The *thickness* consequence of the same fusion is finding 2.)
- **Downstream is safe.** `choke_thickness_by_part` keeps its `Dict[str, float]` shape;
  `choke_thickness_bodies_by_part` values are `list(...)` of `float(np.median(...))` —
  plain Python floats, JSON-serializable, no numpy scalars. `_METAL_PATH_CODES` and
  `_METAL_PATH_LABELS` have the same four keys, so the direct `_METAL_PATH_LABELS[code]`
  indexing in the `FillingResult` construction (`:3913-3920`) cannot `KeyError`. The only
  readers are `core/simulation_pipeline.py:608-616` (a dict copy) and
  `ui/main_window.format_filling_notice`, which reads the summary entirely through `.get()`
  and enumerates nothing — an added payload key is inert. `tests/test_filling_notice.py`'s
  `_summary` builds its dict with `.update(overrides)`, so no strict schema.
- Existing choke assertions still hold: `_plate_welded_to_a_block` and `_choke_rig` each
  have one GATE body (the diff asserts this at `:961-968`), so
  `choke_thickness_by_part["gate"] == 1.0` and the `>= 5.0` generous-channel assertions are
  unchanged. The warning label `"thinnest of the N gate bodies"` still contains the part
  name, so `assertIn("gate", …)` survives either branch.
- **Cost is negligible.** `measure_choke_thickness_components` adds one `ndimage.label` per
  code plus one full-grid `labels == index` and one `np.median` per body, against
  `_local_thickness_cells`' existing one-EDT-per-half-cell-of-radius sweep. Measured
  (`/tmp/cost.py`): 26-connected `label` is **9 ms** on 1.7 M cells and **68 ms** on 13.7 M;
  a per-body `np.nonzero(labels == i)` is 3 ms and 33 ms respectively.

**F2 — correctness**
- **The fine→coarse index arithmetic is exact.** `coarsen_part_types` pads on the *high*
  side with `EMPTY_CELL` to `ceil(shape/factor)` and reshapes, so fine index `i` belongs to
  block `i // factor` — exactly what `erased_metal_path_bodies` computes. Verified
  exhaustively for factors 2, 3, 4, 5 across divisible and non-divisible shapes
  ((7,5,11), (8,8,8), (13,4,9), (1,1,1)): the coarse shape always equals
  `ceil(shape/factor)`, **the `np.minimum(..., shape-1)` clamp never binds** (0 of every
  fine cell in every case), and every fine cell's code is present in the block the mapping
  sends it to (`/tmp/f1f2.py` §F). The clamp is redundant, not wrong.
- **"Any metal-path code" is the right call for the false positive it fixes.** Keying on the
  body's own code fires on the stage's own `_choke_rig`: a one-cell SPRUE stub where a gate
  plate cuts a sprue column becomes GATE at coarsening 2 (GATE outranks SPRUE) while the
  channel stays open and stays measured, because the thickness map is built over the whole
  metal path. Verified: `measure_choke_thickness_components(_choke_rig())[SPRUE]` has two
  bodies and `erased_metal_path_bodies` returns `{}` — the diff's own
  `test_absorption_into_another_metal_part_is_not_severing`. (The *permissiveness* this buys
  is finding 1.)
- `factor == 1` returns `{}` before any work, so the finest grid can never be reported as
  severed and the loop's fallback return is safe. FILTER heads `COARSENING_PRIORITY`, so a
  filter body can never be erased — the detector is effectively about SPRUE/RUNNER/GATE,
  which is the correct set.
- **The refusal follows the established pattern**: `emit(...)` then `return None`, the same
  as the unresolvable-pour-point and no-metal-cells skips, so `maybe_run_filling` hands the
  pipeline `None` and `core/simulation_pipeline.py` proceeds to solidification without a
  fill exactly as it already does. The message names the part, the count, the mechanism and
  the remedy.
- **`choke_auto_resolution = 0` does now refuse where it previously ran.** That is the
  intended behaviour change and it is messaged ("Set choke_auto_resolution to 1 and the
  stage will refine the grid itself"), and it is the *only* reachable refusal path
  (finding 3). Note the resulting asymmetry, which looks deliberate: with auto-resolution
  off, a *sealed* choke still runs under a warning while an *erased* body refuses outright.
- **The fine grid is labelled once per factor tried, not once per call**, so a
  `configured = 4` run labels at factors 4, 3 and 2 — 3 × 4 codes = 12 labelings. At 13.7 M
  cells that is ~0.07 s each plus ~0.03 s per body, i.e. under ~2 s total, against a
  `_local_thickness_cells` sweep that runs one EDT per half-cell of maximum inscribed radius
  on the same grid at every factor. The redundancy is real (the fine labels do not depend on
  the factor and could be hoisted) but it is not a cost worth acting on.

**F3 — the two-check cascade**
- **State handling matches the eps guard exactly.** `cascade_streak` is initialised to 0
  before the loop (`:3168-3170`), incremented only on a breach and reset to 0 on any clean
  check (`:3538`), and compared with `>= NEGATIVE_POPULATION_CASCADE_CHECKS` (`:3539`) — the
  same shape as `fill_fraction_breaches` / `FLUID_FILL_FRACTION_CHECKS`. A check with zero
  wet cells yields `negative_cells = 0`, so the streak resets rather than persisting.
- **The precedence and the message are unchanged in shape**: `if cascading or prevalence >=
  …` then `if cascading:` — the cascade message still wins when both conditions hold, which
  is right, since it is the more specific statement.
- **The recorded round-2 event still aborts, by the trigger arithmetic.** The check that saw
  5 negatives fails the 8-cell floor (streak 0); the next, at 3,681 of ~3,958 wet cells
  (93 %), passes both halves (streak 1); the following check, with 93 % of the field
  negative, passes again (streak 2 → abort). The abort moves 200 steps later, out of the
  27,958 steps of lead the event had.
- **Alternation cannot evade indefinitely.** A 2-cycle (breach, clean, breach, …) never
  reaches streak 2, but every breach check has `negative_cells > 0` so it contributes a 1 to
  `negative_history`; with `NEGATIVE_POPULATION_WINDOW = 40` and
  `NEGATIVE_POPULATION_PREVALENCE = 0.5`, a 50 % duty cycle reaches prevalence exactly 0.5
  and fires, and any negatives at all on the "clean" checks push it above. The observed
  round-2 pattern (negatives present in 59 % of samples, in bursts) is above the threshold.
  A 3-cycle at a 33 % duty cycle with zero negatives on the other two checks would evade
  both, but a reconstruction cascade absent on two of every three checks is not the failure
  mode either guard describes; the prevalence backstop is doing its stated job.
- One consequence worth recording rather than fixing: the cascade guard can now fire no
  earlier than step 400 (checks at 200 and 400), so a run whose `max_steps` is under 400 —
  short test fixtures — has no cascade guard at all. The prevalence guard already needed
  8,000 steps, so this is not a new class of gap.

**Tests**
- **Zero removed or altered lines in either test file** (`git diff tests/ | grep -c '^-[^-]'`
  = 0), so nothing elsewhere was weakened.
- **The 15 new tests fail on HEAD**, by reading their assertions:
  `measure_choke_thickness_components` and `erased_metal_path_bodies` are imported at the
  top of `tests/test_filling_stage.py`, so the whole module fails to import on HEAD (7 of
  the 13 stage tests depend on them directly);
  `test_the_reported_thickness_is_the_thinnest_body` asserts
  `by_part[GATE] <= 2.0` where HEAD's pooled value is 5.0;
  `test_the_decision_refines_for_the_thin_body` asserts `coarsening == 1` where HEAD does
  not refine at all; `test_the_stage_refuses_rather_than_solving_a_severed_path` asserts
  `result is None` where HEAD returns a result;
  `NegativeCascadePersistenceTests` imports `NEGATIVE_POPULATION_CASCADE_CHECKS` (absent on
  HEAD) and, arithmetically, `test_one_check_of_negatives_does_not_abort` asserts a
  non-diverged status where HEAD aborts on the single sample, while
  `test_two_consecutive_checks_do_abort` pins `diverged_at_step == 400` where HEAD would
  abort at 200.
- `test_a_pooled_median_hides_a_sealed_body` and
  `test_the_gate_vanishes_and_no_thickness_guard_can_see_it` are deliberate
  *discriminators* that assert HEAD's behaviour and pass on both, which is the right shape —
  they are what stops the other tests passing on a fixture that lost the ability to fail.
- The injection harness in `NegativeCascadePersistenceTests` is correctly ordered: it wraps
  `pf.step`, and the stage's divergence block runs after `step()` returns within the same
  `taken` iteration, so an injection at `state.step_count == 200` is visible to the check at
  `taken == 200`; `advect_temperature` and `apply_wall_heat_loss` touch only temperature, so
  the injected populations survive to the check. `want = max(8, ceil(0.02 * wet))` is the
  trigger's own threshold, and `assertLess(samples[0]["wet"], 500)` pins the fixture in the
  regime the finding is about. One robustness note: the single-check test depends on the
  step-201 clear removing the injected negatives *and* on the solver not generating ≥8 of its
  own by step 400 — a solver-trajectory dependency the harness cannot assert away, though
  the `elif step_index - 1 in at_steps` clear only resets direction 0.
- The new cascade message reuses the committed `CHECKS * DIVERGENCE_CHECK_INTERVAL`
  convention from the eps guard (`:3614-3615`), so "2 consecutive checks (400 steps)" is
  consistent with its sibling even though two consecutive checks span 200 steps of
  elapsed time; not a new inconsistency.

**Out of scope, noted once:** `choke_thickness_by_part` changed meaning (pooled median →
thinnest body) under an unchanged key, in an unchanged payload, with no version marker — the
same class of problem as the deliberately-skipped surrogate `solver_commit` column, and it
now affects `data/mesh_runs.csv`, `web/simulations.db` and any stored payload alongside the
74 pre-round-4 surrogate samples.
