# VALIDATION ROUND 5 — the `TAU_MINUS_MAX` odd-parity ghost-mode clamp

Baseline **`fe866cf`** (round 4, committed) + the uncommitted round-5 diff
(10 files, 1,045 insertions). No tracked file was modified by this validation;
nothing was committed. All scripts, logs and ledgers are in
`Test/session_diag_2026-09-02/validation_r5/`.
Python `./venv/bin/python`, `QT_QPA_PLATFORM=offscreen`.
Tree fingerprint at the start of the round: `tree_fingerprint.txt`
(`physics/fluid_flow.py` = `34c02e14...`).

---

## VERDICT

| part | result | one line |
|---|---|---|
| **A1** | **PASS** | Both `Full_Gating_Test` grids complete, cleaner than the round-4 baseline on **every** metric: max\|u\| **−45 %** and **−75 %**, negative populations to **zero**, and the round-4 `eps_min` INTERFACE excursion of **−0.71 collapses to −0.0011**. Fills got *fuller and faster*, not slower. |
| **A2** | **PASS** | `Filling_Test` still completes, and **faster**: 98,160 → **65,440 steps** (18.0 s → 12.0 s), 97 % → **98 %** by mass, max\|u\| **−92 %**, negatives 6/5 → **0**. Consequence (b) did not dominate — see the judgement below. One new isolated single-row `eps_min = −2.014` characterized. |
| **A3** | **PASS** | 1,244 tests (+8), **1 failure, 0 errors**, 11 skipped. The failure is the known pre-existing ordering flake and passes in isolation. **Zero new failures**, and the suite ran 120 s faster. |
| **B** | **MIXED — conserved quantities PASS, endpoint FAIL (running)** | Test 2 is **5.5x past** the round-4 death step with **0** non-finite, **0** negative populations on every row, filter eps <= **1.0023**, wet mass exactly monotonic — the round-4 ghost mode is gone. But (i) **delivery collapsed to a settled 4.33 lb/s = 21 %** of target, so it will **time-cap, not complete**, and (ii) after the gate unprimes at step ~39,500 `max|u|` runs intermittently to **0.36** (48/239 rows > 0.1) — bounded and non-growing (e-fold ~350k steps vs round 4's 64), but not quiet. Judgement: the 3-cell gate overstates the physical choke by ~**5x** vs the metered rate and **12-22x** vs Torricelli — a resolution artefact, not a round-5 regression. |

**The fix does what §4 of the mech3 report predicts, and the predicted cost did
not materialise as a slowdown.** Expected consequence (c) is confirmed
emphatically; expected consequence (b) is real in the wall position but is
**more than offset** by the delivery the ghost mode was stealing.

---

## 0. What changed, as I read it

Non-comment changes only:

* `physics/fluid_flow.py` — `TAU_MINUS_MAX = 5.0`; `_validate_tau_minus_max`
  (refuses `<= 0.5`, allows `inf`); `tau_minus_max` added to `LatticeUnits` and
  to `lattice_units(...)`, `omega_minus_from_magic(...)`,
  `smagorinsky_relaxation(...)` and threaded into `collide_kernel`. The clamp is
  applied **identically at all three relaxation sites** — scalar
  (`tau_minus = magic/half + 0.5; if tau_minus > tau_minus_max: ...`), array
  (`np.minimum(tau_minus, tau_minus_max)`), and Numba (same scalar form inside
  the kernel). One authority, carried on the state.
* `physics/filling.py` — **2 lines**, both just forwarding
  `units.tau_minus_max` to the two step paths.
* `core/filling_stage.py` — **0 executable lines**. All 33 changed lines are
  comments/docstrings.
* tests — `test_filling_lattice.py` (+109), `test_filling_stability.py` (+406),
  `test_filling_benchmarks.py` (+13).

The claim I checked and accept: the clamp binds only for
`tau_plus < magic/(TAU_MINUS_MAX − 1/2) + 1/2 = 0.541667`. Every casting lattice
here sits at `tau_plus ≈ 0.50001`, so it binds on all four assemblies, and
`tests/test_filling_benchmarks.py`'s Poiseuille wall benchmark at `tau = 0.8`
keeps exact `Lambda = 3/16`.

### A methodological caveat that governs every "fill time" below

Fill time is **quantised to the progress-report interval**, `report_every =
max_steps // 20`, because the loop exits on a report boundary. On all three
sentinels that quantum is **3.00 s**:

| assembly | `max_steps` | `report_every` | quantum |
|---|---|---|---|
| FGT @0.20 | 288,831 | 14,441 | 3.00 s |
| FGT @0.17 | 339,660 | 16,983 | 3.00 s |
| Filling_Test @0.20 | 327,200 | 16,360 | 3.00 s |
| Test 2 | 436,593 | 21,829 | 1.50 s |

So FGT @0.20 completing at "28,882 steps / 6.00 s" in **both** rounds means only
*within the same 3 s window*, not identically; and Filling_Test's 6 intervals →
4 intervals **is** a genuine two-interval improvement. I do not read equal step
counts as byte-identity, and the coordinator is right that nothing is
byte-identical this round.

---

## A1 — `Full_Gating_Test` — PASS

Same driver and params as round 4. `--voxel 0.20` (1,354 cells @ 10.2 mm) and
`--voxel 0.17` (2,239 cells @ 8.6 mm). Baselines are my own round-4 ledgers
`validation/a1_fgt20_ledger.csv` and `validation/a1b_fgt17_ledger.csv`.

### A1a, FGT @0.20 — column by column vs round 4

| column | R4 | R5 | judgement |
|---|---|---|---|
| status / steps / fill time | complete / 28,882 / 6.00 s | complete / 28,882 / 6.00 s | same 3 s window (see quantisation caveat) |
| cells filled | 1,354/1,354 (100 %) | 1,354/1,354 (100 %) | unchanged |
| **by mass** | 95 % | **96 %** | **(c)** — fuller |
| **lb delivered** | 20.8 | **21.2** | **(c)** — +1.9 % |
| `wet_mass` final | 1282.92 | **1302.99** | **(c)** — +1.6 % |
| delivery @3.00 s | 100 % of 4.39 lb/s | 100 % | unchanged, both unthrottled |
| delivery @6.00 s | 58 % | **61 %** | **(c)** |
| wet cells @3.00 s | 688 | **798** | **(c)** — front is *ahead*, not behind |
| **max\|u\| max** | 0.05567 | **0.03068** | **(c) confirmed — −44.9 %** |
| max\|u\| p99 / median | 0.05042 / 0.01616 | **0.02840 / 0.00849** | **(c)** — median halved |
| rows with max\|u\| > 0.1 | 0/57 | 0/57 | unchanged |
| `eps_max` = `eps_fluid_max` max | 1.01111 | **1.00409** | **(c)** — −0.7 %, 27x margin on the 1.1 guard |
| **`eps_min` min (INTERFACE)** | **−0.712381** | **−0.00113494** | **(c) — the round-4 excursion is gone**, 628x smaller |
| `eps_fluid_min` min | +0.00179 | **−0.00105** | sign change, flagged below |
| `neg_f` / `neg_cells` / `neg_porous` max | 0 / 0 / 0 | 0 / 0 / 0 | unchanged |
| non-finite rows | 0/57 | 0/57 | unchanged |
| `min_temp` min | 1502.3 °C | 1490.3 °C | −12 °C, still above the 1460 °C liquidus; no cold-shut warning |
| wet mass monotonic | yes (1 rounding step) | yes (2 rounding steps, worst −0.1 on 1303 = 8e-5) | unchanged in substance |
| wet-cell drops | 4 | **1** | **(c)** — front flicker nearly gone |
| riser first wet | step 15,000, **z=8** (foot), 2 cells, cast z1–8 n628 | step 17,500, **z=8** (foot), 10 cells, cast z1–8 n598 | **still fills from the foot**; later and with more cells at once, consistent with a smoother front |
| rows with a z-gap in the wet riser span | 0/57 | **0/57** | unchanged — contiguous, bottom-up |
| warnings | 3 × 3-cell channel | 3 × 3-cell channel (**text rewritten**, see below) | — |

### A1b, FGT @0.17 — column by column vs round 4

| column | R4 | R5 | judgement |
|---|---|---|---|
| status / steps / fill time | complete / 50,949 / 9.00 s | complete / 50,949 / 9.00 s | same 3 s window |
| cells / by mass / lb | 2,239 (100 %) / 96 % / 21.6 | 2,239 (100 %) / **97 %** / **21.9** | **(c)** |
| `wet_mass` final | 2142.53 | **2168.74** | **(c)** — +1.2 % |
| delivery @3.00 / 6.00 s | 100 % / 66 % | 100 % / **68 %** | **(c)** |
| wet cells @3.00 s | 969 | **1,029** | **(c)** — ahead |
| **max\|u\| max** | 0.24862 | **0.06145** | **(c) confirmed — −75.3 %** |
| max\|u\| p99 / median | 0.15422 / 0.01395 | **0.03536 / 0.00943** | **(c)** — p99 down 4.4x |
| rows > 0.1 / > 0.2 | **4** / **1** of 101 | **0** / **0** | **(c) — the excursions are gone** |
| `eps_max` max | 1.02791 | 1.01402 | **(c)** |
| `eps_fluid_max` max | 1.02791 | **1.00485** | **(c)** — −2.2 %, 20x margin |
| **`eps_min` min** | **−0.420479** | **−0.0108714** | **(c)** — 39x smaller |
| `eps_fluid_min` min | +0.00581 | **−0.00104** | sign change, flagged below |
| **`neg_f` / `neg_cells` max** | **1 / 1** | **0 / 0** | **(c) — zero negative populations** |
| non-finite rows | 0/101 | 0/101 | unchanged |
| `min_temp` min | 1425.8 °C | 1401.6 °C | −24 °C |
| cold-shut warnings | min fill 1423 °C; **65** voxels | min fill 1399 °C; **61** voxels | slightly *fewer* cold-shut voxels despite the lower minimum |
| wet-mass drops | 6 (worst −0.41) | **1** (worst −0.16) | **(c)** |
| riser first wet | step 17,000, **z=9** (foot), cast z1–9 n1,033 | step 18,000, **z=9** (foot), cast z1–9 n1,019 | **still fills from the foot** |
| z-gaps in the wet riser span | 0/101 | **0/101** | unchanged |

### The coordinator's specific question: the round-4 `eps_min` −0.71

**It is gone.** −0.712381 → **−0.00113494** on FGT @0.20, and −0.420479 →
**−0.0108714** on FGT @0.17. That is a 628x and 39x reduction, and it is the
single most direct corroboration that my round-4 reading was right: I flagged
that excursion as the faster, un-throttled front driving `_redistribute` harder,
and said "the same excursion landing on a FLUID cell would abort the run". The
real cause was the undamped odd mode, and clamping `tau_minus` removed it.
**Round 4's deepest interface mass debt was ghost-mode noise.**

### Two things that got *worse*, stated plainly

1. **`eps_fluid_min` changed sign on all three sentinels**: +0.00179 → −0.00105
   (FGT @0.20), +0.00581 → −0.00104 (FGT @0.17), +0.00178 → −0.00070
   (Filling_Test). FLUID cells now dip **barely** below empty. The magnitude is
   ~1e-3, i.e. **95x inside** the guard's `FLUID_FILL_FRACTION_MIN = −0.1`
   floor, and it is flat across the run rather than trending. Not a blocker, but
   it is a sign change in the quantity the new guard watches and belongs on the
   record.
2. **`min_temp` fell 12 °C (@0.20) and 24 °C (@0.17)** on FGT. Both remain
   plausible; @0.17's 1399 °C is below liquidus and warned about, as it was in
   round 4 at 1423 °C. Cold-shut voxel count actually improved (65 → 61).

### Resolved from round 4

The stale warning string I flagged — "*Test_Model/Filling_Test … time-caps at
57 % of cells*" — **has been rewritten** to "*the thinnest channel this solver
has ever carried a complete fill through — and it carries it at the edge of the
velocity range the timestep was sized for … if it matters to the answer, resolve
it and rerun*". Accurate now, and it no longer asserts a time-cap that does not
happen.

---

## A2 — `Filling_Test` @0.20 — PASS, and it got faster

**It still completes, and it completes sooner.**

| column | R4 | R5 | judgement |
|---|---|---|---|
| status | complete | **complete** | — |
| **steps / fill time** | 98,160 / 18.00 s | **65,440 / 12.00 s** | **−33 %**, a genuine two-report-interval improvement |
| cells | 4,357/4,357 (100 %) | 4,357/4,357 (100 %) | unchanged |
| **by mass / lb** | 97 % / 70.0 | **98 % / 70.3** | fuller |
| `wet_mass` final | 4233.95 | **4255.29** | +0.5 % |
| **delivery** | 96 % → **54 → 52 → 50 → 40 → 1 %** | **100 % → 100 % → 91 % → 2 %** | **the collapse is gone** |
| **max\|u\| max** | 0.44566 | **0.03526** | **−92.1 %** |
| max\|u\| p99 / median | 0.09376 / 0.01230 | **0.03345 / 0.01210** | p99 −64 % |
| rows > 0.1 | 1/196 | **0/130** | **(c)** |
| `eps_max` = `eps_fluid_max` max | 1.02657 | **1.00790** | −1.8 % |
| **`neg_f` / `neg_cells` max** | **6 / 5** | **0 / 0** | **zero negative populations** |
| non-finite rows | 0/196 | 0/130 | unchanged |
| `min_temp` min | 1283.1 °C | **1345.3 °C** | **+62 °C** — fills faster, so less time to cool |
| cold-shut voxels | 4,255 | **684** | **−84 %**, the direct consequence of the warmer, faster fill |
| wet-cell drops | 23 | **3** | **(c)** |
| riser first wet | step 18,500, **z=8** (foot) | step 19,000, **z=8** (foot) | still from the foot |
| **z-gaps in the wet riser span** | **10/196** | **0/130** | **the round-4 "splash" is gone** |
| warnings | 5 | 5 (same set, rewritten channel text) | — |

**The round-4 splash finding is corroborated.** I characterized those 10
gap-rows as "splash off metal entering the riser foot … 1–4 cells, bounded,
non-growing" and explicitly distinguished them from a wrap seed. They are now
**0 of 130 rows**. They were the ghost mode, and they vanished with it — the
same way the mech3 report says the 123 Test 2 excursions do.

### Judgement on expected consequence (b) — asked for, so here it is

**(b) is real and I can see it, but it does not dominate, and A2 is the proof.**
The −0.25 cell² of flow-carrying width is a *steady Stokes–Poiseuille* result: it
is what the clamp costs a **laminar, fully-developed** channel flow. Round 4's
gate was not that. It was carrying an undamped odd-parity mode with peak
`|u|` at **0.446 lattice (Ma ≈ 0.77)**, 9x the 0.05 design velocity, plus
negative populations. That noise fed the inlet rate controller spurious
back-pressure, and its delivery decayed to **54 → 52 → 50 → 40 → 1 %** of
target. Round 5 holds **100 % for the first 6 s and 91 % at 9 s**. Sustained
delivery roughly doubles; the wall-position deficit through a 3-cell channel is
−11 %. **+100 % beats −11 %,** so the fill finishes 33 % sooner despite each
cell of channel carrying slightly less. The same sign appears on both FGT grids
(wet cells at 3.00 s: 688 → 798 and 969 → 1,029) where the ghost mode was
milder, so the effect is consistent, not a one-off.

I therefore record **no fill-time regression, and no delivered-mass drop at
time-cap** — nothing time-capped. If the clamp's −11 % is ever to be seen
cleanly it will need a fixture that was already laminar and ghost-free, not
these three.

### One new observation: an isolated `eps_min = −2.014`

`eps_min` on Filling_Test went −0.09316 → **−2.01410**, i.e. worse, and it is
the only metric on any sentinel that moved the wrong way by a large factor. What
it actually is:

* **One ledger row.** Step 31,000, in a sequence reading −0.00097, **−2.01410**,
  −0.00093, −0.00073. `rows with eps_min < −0.5: 1 of 130`.
* **An INTERFACE cell, not FLUID.** `eps_fluid_min` at that row is −0.000132,
  so the new `[−0.1, 1.1]` FLUID guard is not applicable and correctly silent.
* **It left no trace on anything else.** At step 31,000: `eps_max` 1.00168
  (identical to its neighbours), `wet_mass` 2,756 on a perfectly linear +44 per
  500 steps, `max|u|` 0.01219 (flat), `neg_f` 0, non-finite 0. It self-corrected
  within one ledger interval.
* **The distribution improved.** Median `eps_min` −0.000721 → −0.000645 and the
  1st percentile −0.05204 → **−0.01153**. So typical and near-worst interface
  mass debt both got ~4.5x better; there is one deeper isolated spike.
* **Plausible mechanism, not verified:** `iface_cells` is collapsing fast right
  there (527 → 501 → 480 → **403** → 374) as the casting consolidates, so many
  interface cells turn FLUID at once and `_redistribute` has more surplus to
  place with fewer interface recipients — the documented donor-stranding case.

**Honest bottom line on it:** it did not affect this run, and the guard was right
not to fire. But if `_redistribute` can strand −2 cells of mass debt on a single
interface cell, the same event on a FLUID cell trips the −0.1 floor and aborts a
five-hour pour. Worth the author's eye; **not** a reason to hold the diff, and I
did not attempt a fix.

---

## A3 — full test suite — PASS

```
CASTSIM_THREAD_BUDGET=8 QT_QPA_PLATFORM=offscreen ./venv/bin/python -m unittest discover tests
Ran 1244 tests in 244.919s
FAILED (failures=1, skipped=11)
```

**failures = 1, errors = 0, skipped = 11.**

The one failure:
```
FAIL: test_a_deeper_tower_leaves_none_either
      (test_filling_stability.SubmergedInterfaceHealingTests.test_a_deeper_tower_leaves_none_either)
  File tests/test_filling_stability.py, line 1874
    self.assertEqual(result.status, STATUS_COMPLETE)
AssertionError: 'diverged' != 'complete'
```

Reconciled against the known list:

| known item | fired? | reconciliation |
|---|---|---|
| `SubmergedInterfaceHealingTests::test_a_deeper_tower_leaves_none_either`, full-suite ordering only | **yes** | **On the list.** Passes in isolation: `python -m unittest tests.test_filling_stability.SubmergedInterfaceHealingTests` → **Ran 9 tests, OK**. It is the *only* failure, identically, in every suite run on record — `r3_suite.log` (1212), round-4 `suite.log` (1230), round-4 `r2_suite.log` (1236), this round (1244) — all `failures=1, skipped=11`. |
| `test_deep_model InferenceTests::test_pipeline_end_to_end` order-flake (`pipeline.py:102 completion` unbound) | no | Did not fire. It is a flake and `test_deep_model` is not under `tests/`. |
| `tests/test_filling_stability` importing `_Options` from `test_filling_stage`, mutating a shared `_Params.filling` | n/a | The ordering-leak mechanism behind item 1; no separate failure. |
| global `vtk` stub | no | No `vtk`-attributable failure. |

**Nothing outside the known list failed or errored. Zero new failures.**
Test count **1,236 → 1,244 (+8)**, matching the round-5 test additions in
`test_filling_lattice.py`, `test_filling_stability.py` and
`test_filling_benchmarks.py`. The suite also ran **120 s faster** (364.3 s →
244.9 s), consistent with fixtures no longer grinding through ghost-mode
excursions.

---

## B — the acceptance run, full Test 2 pour — DELIVERY COLLAPSE, run in progress

```
CASTSIM_THREAD_BUDGET=14 QT_QPA_PLATFORM=offscreen ./venv/bin/python fill_driver.py \
  --dir "/Users/brendonjarett/Downloads/Test 2" --voxel 0.17 --refine 4 \
  --max-fill-time 30 --ledger b_t2_ledger.csv --ledger-every 500
```
PID 7079, started 23:58:48. 339,824 metal cells at 4.3 mm, solve grid
(137,137,98), inlet 136 cells, metering 20.8 lb/s, gate 3.0 cells.

Two commits landed after this run imported and therefore do **not** affect it:
round 5 as **`37a9ad8`** and the choke-guard/coarsening/cascade-persistence
change as **`4f057d3`** (`core/filling_stage.py` + `core/simulation_pipeline.py`,
no physics).

### Numerical health — every acceptance criterion except the endpoint is met

| criterion | status at step ~114,500 (t = 7.87 s) |
|---|---|
| no non-finite | **PASS** — 0 rows of 229 |
| no precursor abort | **PASS** — none |
| filter eps bounded | **PASS** — `eps_filter_max` max **1.00232**, filter+dilation max **1.00408** |
| no wet riser cells above the front | **PASS** — the riser has **0 wet cells**, period |
| wet mass monotonic | **PASS** — 0 decreases |
| delivery ≈ 20.8 lb/s unthrottled | **FAIL** — see below |
| honest endpoint | pending; it will **time-cap**, not complete |

* negative populations: `neg_f` **0**, `neg_cells` **0**, `neg_porous_cells` **0**
  on every row
* `max|u|`: **see the correction below — this run does carry velocity
  excursions after step 39,500**, peak **0.360**, 48 of 239 rows above 0.1.
  They are bounded and non-growing, and they are not the round-4 ghost mode.
* `eps_fluid_max` max 1.00903; `eps_min` (INTERFACE) min −0.881
* warnings: only the 3 pre-existing `matmul` RuntimeWarnings

#### CORRECTION to my own earlier statement, and what the excursions are

In my previous message I wrote that this run had "**0 rows above 0.1** in 229
rows" and that `max|u|` peaked at 0.0923. **That was wrong.** I computed it from
a snapshot taken at step ~30,000 — before the delivery collapse — and then
carried the claim forward when writing this section at step 114,500 without
recomputing. The correct figures over the run so far (239 rows):

| step window | max\|u\| max | max\|u\| median | rows > 0.1 | `eps_min` min |
|---|---|---|---|---|
| 0–20,000 | 0.0470 | 0.0356 | **0/39** | −0.748 |
| 20,000–40,000 | 0.1067 | 0.0317 | 1/40 | −0.881 |
| 40,000–60,000 | **0.3120** | 0.0847 | 14/40 | −0.497 |
| 60,000–80,000 | 0.2544 | 0.0867 | 13/40 | −0.418 |
| 80,000–100,000 | 0.2019 | 0.0761 | 6/40 | −0.920 |
| 100,000–120,000 | **0.3602** | 0.0865 | 14/40 | −0.644 |

**First row above 0.1: step 39,500. Total 48/239. Peak 0.3602.**

Three things follow, and they matter to the verdict:

1. **The excursions and the delivery collapse are the same event.** Both begin in
   the 30,000–40,000 window. Before step ~30,000 — while the gating was filling
   and delivery held 99 % — `max|u|` was 0/39 rows above 0.1 with a median of
   0.036. The excursions start precisely when the gate starts discharging into
   air as a free-surface stream.
2. **They are NOT the round-4 ghost mode.** An exponential fit over steps
   ≥ 100,000 gives **+2.9e-6 per step, an e-folding time of ~350,000 steps** —
   i.e. flat. Round 4's mode grew at **+1.55e-2 per step, e-folding 64 steps**:
   this is **5,400x slower**, which is to say stationary. The windowed maxima
   (0.107, 0.312, 0.254, 0.202, 0.360) and `eps_min` minima (−0.75, −0.88,
   −0.50, −0.42, −0.92, −0.64) both wander without trend over 80,000 steps. And
   `neg_f`, `neg_cells` and `neg_porous_cells` are **0 on every one of the 239
   rows** — round 4's mode was producing negative populations long before it
   died. So round 5's clamp is holding; this is a different, bounded phenomenon.
3. **They are still not physical.** 0.360 lattice is ~22.6 m/s at this run's
   scaling (0.05 lattice = 3.14 m/s), and 7x the design velocity. Free-fall from
   the gate to the pool 95 mm below gives only 1.37 m/s = 0.022 lattice, so
   impact splash does not account for it. The most likely source, and it is
   consistent with the simultaneous deep `eps_min` swings to −0.92, is the
   free-surface reconstruction on a **thin falling stream**: interface cells
   carrying very little mass but having their velocity reconstructed from it, so
   a small mass error becomes a large apparent velocity. That is a hypothesis I
   have not tested, not a measured mechanism, and I am not going to dress it up
   as one.

So the honest statement is: **this run is stable and clean in every conserved
quantity — no non-finite, no negative populations, monotonic mass, bounded
filter eps — but it is not quiet.** After the gate unprimes, `max|u|` runs
intermittently at 2–7x the design velocity in a bounded, non-growing way. That
is a further reason to distrust this run's *rates* while trusting its
*stability*, and it strengthens rather than weakens the resolution argument
below.

**Round 5 fixed what it set out to fix.** The step-20,200 death is gone, the run
is 5.5x past it, and it is numerically the cleanest Test 2 pour on record. What
it did not fix — and was not meant to — is that this mold will not fill.

### The delivery collapse — it has SETTLED, and it is choke-limited

Instantaneous delivery from the ledger in 10,000-step windows:

| steps | t (s) | lb/s | % of 20.84 target |
|---|---|---|---|
| 0–30,000 | 0.00–2.06 | **20.58** | **99 %** |
| 30,000–40,000 | 2.06–2.75 | 8.00 | 38 % |
| 40,000–50,000 | 2.75–3.44 | 4.17 | 20 % |
| 50,000–60,000 | 3.44–4.12 | 4.30 | 21 % |
| 60,000–70,000 | 4.12–4.81 | 4.29 | 21 % |
| 70,000–80,000 | 4.81–5.50 | 4.33 | 21 % |
| 80,000–90,000 | 5.50–6.18 | 4.34 | 21 % |
| 90,000–100,000 | 6.18–6.87 | 4.34 | 21 % |
| 100,000–110,000 | 6.87–7.56 | 4.33 | 21 % |

**It is settling, not still falling.** 99 % for the first 2.06 s while the
gating filled, a two-window transition, then **dead flat at 4.29–4.34 lb/s for
60,000 steps / 4 s**. This is a stable equilibrium, not a decay.

### Where the free surface is — the diagnostic answer

| part | wet | state |
|---|---|---|
| runner | 17,640/17,642 = **100 %** | full, pressurised |
| filter | 4,705/4,708 = **100 %** | full |
| **gate** | 2,420/4,072 = **59 %** | **never fills** — peaked at 76 % (step 36,500), settled 57–60 % |
| sprue | 8,984/11,182 = **80 %** | peaked at **99 %** (step 30,500), then **drained** to 80 % and held |
| cast | 33,417/193,814 = **17 %** | filling steadily, bottom-up |
| riser | 0/113,114 | untouched |

**Is the sprue filling toward the mouth? No — the opposite.** Its per-plane wet
count shows the well (z15–33) full at 250–273 cells/plane, the **column
(z34–89) only 40–55 % occupied** (39–103 cells against 132 for a full plane),
and only the top five planes (z90–94) full — and those are the inlet halo that
`ensure_interface_skin` holds wet every step by construction. So there is a
free-falling stream down a **substantially empty downsprue**: the system is
**unprimed**, and the metallostatic column that the gating design assumes never
develops.

**The gate never pressurises.** Wet cells per plane, from its floor upward
(1,018 cells to a full plane): **z33 = 746 (73 %), z34 = 703 (69 %),
z35 = 578 (57 %), z36 = 397 (39 %)**. A free surface runs through the gate along
its whole length; the metal is sliding along the bottom of the slot.

**And it is not back-pressure.** The casting's own level is at **z ≈ 11**
(z1–z11 carry 1,365→4,091 cells/plane, then z12 drops to 291), i.e. **22 planes
= 95 mm below the gate floor**. The gate is discharging into open air. Nothing
downstream is holding it back. The planes z12–z33 carry only 130–280 cells each
— the free-falling stream wetting the casting's walls on the way down.

### JUDGEMENT — is the 3-cell gate's numerical resistance overstating the physical choke?

**Yes, and by roughly an order of magnitude. The gates are not a physical choke
at all.**

Gate geometry, measured off the solve grid (two components, 2,036 cells each):
each gate is **21 × 4 cells normal to the flow = 90.3 × 17.2 mm = 3.56 × 0.677 in
= 15.53 cm² (2.41 in²)**; **both gates = 31.06 cm² (4.81 in²)**. The 0.677 in
matches the coordinator's 0.68 in. Note in passing that the stage's own
`measure_choke_area_cm2` reports **2.9584 cm²** — a thickness-derived proxy that
**understates the true gate cross-section by 10.5x**; it is used for reporting
and the sqrt-rule cross-check, not to set the flow, but it should not be read as
a flow area.

**Torricelli through both gates** (ρ = 7,200 kg/m³):

| available head | v = √(2gh) | Cd=1 | Cd=0.75 | Cd=0.5 | Cd=0.25 |
|---|---|---|---|---|---|
| 4.0 in basin only (102 mm) | 1.412 m/s | 69.6 lb/s | 52.2 | 34.8 | 17.4 |
| basin + full sprue above the gate (351 mm) | 2.624 m/s | 129.4 lb/s | 97.0 | 64.7 | 32.3 |

The 1.412 m/s figure is exactly the "1.41 m/s efflux" the stage prints for the
mouth, so the head bookkeeping agrees.

**Observed 4.33 lb/s against those:**

| comparison | ratio |
|---|---|
| vs the 20.84 lb/s metered target | 21 % → **4.8x shortfall** |
| vs Torricelli, basin only, Cd = 0.75 | 8.3 % → **12.1x understatement** |
| vs Torricelli, basin + full sprue, Cd = 0.75 | 4.5 % → **22.4x understatement** |
| vs Torricelli, basin only, **Cd = 0.25** (absurdly pessimistic) | 24.9 % → **4.0x** |

Put the other way round, which is the cleanest statement of it: the two gates,
discharging freely into air, need only **643 Pa = 0.36 in of WCB** of dynamic
head to pass the **full** metered 20.84 lb/s (v = 0.42 m/s through 31 cm²). The
4.0 in basin alone supplies **7,176 Pa — 11x more than required**, before any
sprue column is counted. At the observed 4.33 lb/s the gates are passing metal
at **0.088 m/s**, needing **27.8 Pa**. **Even at Cd = 0.25 the gates out-pass the
metered pour rate.** They are not the restriction. The solver's gate is.

**It is not the round-5 clamp.** The −0.25 cell² wall offset gives
H_eff = 2 √(1.5² − 0.25) = **2.828 cells vs 3.000**, a **−11.1 %** flow deficit.
That is 11 percentage points, not a factor of 4.8. And a viscous/Poiseuille
resistance argument cannot carry it either: at `tau_plus = 0.500010` the lattice
viscosity is **ν = (τ−½)/3 = 3.3e-6**, essentially inviscid, so viscous channel
resistance is negligible. The clamp is a small, correctly-priced contributor
sitting on top of a much larger pre-existing problem.

**What it actually is — under-resolution, and the numbers identify it.** Model
the gate as the partially-filled open channel the wet-fraction data says it is:
60 % of a 17.2 mm slot is a **10.3 mm flow depth**, whose critical velocity is
√(g·d) = **0.318 m/s** (not the 1.412 m/s Torricelli efflux — a 4.4x velocity
deficit), over 60 % of the area:

> **Q = 0.60 · 31.06 cm² · 0.318 m/s · 7,200 = 9.4 lb/s, against the observed
> 4.33 lb/s — within 2.2x.**

The open-channel model is within a factor of 2; the pressurised-orifice model is
out by 12–22x. **The solver is running the gate as a shallow open channel
instead of a full pressurised passage**, and that is a resolution failure with a
clear mechanism: the gate is 4 cells tall, so once both side walls carry
bounce-back and the top boundary carries a Koerner free surface, there is
effectively **no interior fluid node left across the gate**. The scheme has
nothing left to represent "full, pressurised gate" with, so the free surface
sits inside the passage and the discharge collapses to gravity-driven shallow
flow — and, self-consistently, the downsprue then cannot stay primed either,
which removes the metallostatic head as well.

**Bottom line.** The delivery collapse is **a resolution artefact, not a round-5
regression and not a physical choke**: roughly **5x** against the metered pour
rate and **12–22x** against Torricelli at the available head. It is exactly what
the stage's own warning tells the user to distrust — *"the gate resolves to 3.0
cells across … Treat the flow through the gate as marginal, not as a result; if
it matters to the answer, resolve it and rerun."* That warning fired on this run
and it is correct. **Test 2 at 4.3 mm cannot answer "how fast does this mold
fill."** A trustworthy answer needs the gate at ≥ 6–8 cells across — voxels of
~2.1–2.9 mm, i.e. 3–4x the linear resolution and roughly 10–20 M cells, which is
out of reach at this solver's current cost. The fill's *stability* result stands;
its *fill-rate* result does not, and nothing in round 5 claims otherwise.

## Files

All under `Test/session_diag_2026-09-02/validation_r5/`: `REPORT.md`;
`fill_driver.py`, `analyze.py`, `face_audit_any.py` (carried over from round 4
unchanged — the driver already has the per-step mass-only non-finite check, the
`eps_fluid_*`/`eps_filter_*`/`u_p99`/`u_med` columns and `wetz_by_type`);
`tree_fingerprint.txt`; `a1_fgt20*`, `a1b_fgt17*`, `a2_ft20*`, `suite.log`,
`b_t2*` (log, ledger, `.types.npy`, `.pid`).
