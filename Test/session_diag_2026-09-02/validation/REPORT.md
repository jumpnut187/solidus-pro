# VALIDATION — uncommitted two-part filling fix (filter exchange + mold pad)

Branch `refactor/pouring-seams`, HEAD `f1daf97` + the uncommitted 8-file diff.
No tracked file was modified by this validation. All scripts, logs and ledgers
live in `Test/session_diag_2026-09-02/validation/`.
Python: `./venv/bin/python`. `QT_QPA_PLATFORM=offscreen` throughout.

---

## VERDICT

| part | result | one line |
|---|---|---|
| **A1** | **PASS** | `Full_Gating_Test` completes clean on both grids; the riser now wets from its own foot at z=8/z=9 *after* the casting below it, peak lattice velocity drops 4.9x, and the fill time falls from a non-physical 18.0 s to 6.0 s (metered pour = 4.3 s). |
| **A2** | **PASS** | `Filling_Test` no longer time-caps: it completes 4,357/4,357 cells at 18.0 s, 97 % by mass. Clean, no non-finite, riser wets from the foot. (Consequence: the "time-caps at 57 %" claim now baked into the solver's own warning text and into `docs/` is **stale**.) |
| **A3** | **PASS** | 1,230 tests, **1 failure, 0 errors**, 11 skipped, 365.7 s. The single failure is the known pre-existing full-suite-ordering flake and passes in isolation. **Zero new failures.** |
| **B** | **FAIL** | Test 2 **diverged at step 20,200 (1.39 s)**, first non-finite mass at step 20,192, **bit-reproducibly**. A **NEW failure, different mechanism**: both target mechanisms are demonstrably gone, but the run is **marginally stable from step 4,180 onward** (`max|u|` spiking to 0.28-0.49 lattice on 123 sampled rows and damping each time) and at step ~20,020 a spike fails to damp, growing at 0.01554/step to a momentum/density blow-up at the **filter #1 inlet / filter-runner interface**. The new fill-fraction guard is breached only **32 steps** before the NaN -- inside its own 200-step cadence -- so it never fires. Characterized below; **not fixed**. |

Both diagnosed mechanisms are confirmed eliminated. Fix 1 holds: filter-cell
`eps` stayed within **[0.930, 1.0079]** for the whole Test 2 run against **7.76**
pre-fix. Fix 2 holds: the Test 2 riser had **zero** wet cells at the moment of
death, against 719 over-filled riser cells at z 86-90 pre-fix. **The fix does
what it claims. It is not sufficient to make Test 2 pour.**

---

## IMPORTANT — the working tree moved under this validation, and it was re-run

The tracked files under test were **edited by another actor at 21:44-21:48**,
after every one of my first-round runs had finished. `git diff --stat` grew from
**10 files / 1,400 insertions** (the state I was briefed on and first tested) to
**11 files / 2,036 insertions**. The semantic changes are real, not cosmetic:

* `_domain_boundary_mask` -> **`_open_air_mask`**, which now marks **only the
  `+z` face** rather than all six. Sides and floor are mold, not air, and are
  deliberately no longer vents. This changes the vent term I had validated.
* the faces-are-obstacle invariant changed from **`raise RuntimeError`** to a
  graceful `emit(...)` + `return None` listing the offending planes.
* the fill-fraction guard gained a **debounce**: `FLUID_FILL_FRACTION_CHECKS = 2`
  consecutive breaching checks (200 steps apart) are now required before it
  aborts.
* `create_state` now seeds via `zone.porosity` + `.astype(np.float64)` rather
  than `zone.porosity_f64` (equivalent).
* `tests/test_filling_defects.py` joined the diff; `tests/test_filling_stage.py`
  grew 283 -> 585 insertions.

**Everything was therefore re-run on the current tree, and the results hold.**
Tree fingerprint at the time of the second round:
`tree_fingerprint_r2.txt` (sha256 of all 7 changed code/test files;
`core/filling_stage.py` = `e35a6fce...`).

| re-run | result on the current tree |
|---|---|
| A1a, FGT @ 0.20 | **full ledger bit-identical** to the first round (`diff` clean over all 57 rows) — complete, 28,882 steps, 6.00 s, 1,354/1,354, 95 % by mass, 20.8 lb, same 3 warnings |
| A1b, FGT @ 0.17 | **full ledger bit-identical** — complete, 50,949 steps, 9.00 s |
| A2, Filling_Test | **full ledger bit-identical** — complete, 98,160 steps, 18.00 s, same 5 warnings |
| A3, suite | **1,236 tests** (was 1,230; +6 more new tests), **1 failure, 0 errors**, 11 skipped, 364.3 s — the same single known ordering flake |
| B, Test 2 | **bit-identical** at every shared row — **same** non-finite step **20,192**, **same** abort step **20,200**, same 5 warnings, 848.3 s wall. The new `FLUID_FILL_FRACTION_CHECKS = 2` debounce did **not** help, exactly as the 32-step lead time predicts: it now needs 400 steps of sustained breach. |

So the 21:44-21:48 edits are **behaviour-preserving on all three regression
sentinels**, and every A1/A2/A3 number in this report is current. Logs:
`r2_a1_fgt20.log`, `r2_a1b_fgt17.log`, `r2_a2_ft20.log`, `r2_suite.log`,
`r2_b_t2.log`, with matching `r2_*_ledger.csv`.

Verified after the last re-run finished (22:05): the sha256 of
`core/filling_stage.py` and `physics/filling.py` still match the round-2
fingerprint, so all five re-runs describe **one** consistent tree state.

**Caveat I cannot remove:** the tree is being actively edited by someone else.
An edit after 22:05 would invalidate this round too. The fingerprint file is
there so the next reader can tell.

---

## 0. Method / instrumentation

`fill_driver.py` here is a copy of `Test/session_diag_2026-09-01/fill_driver.py`
with three changes, as instructed:

1. The per-step non-finite check now tests **`state.mass` only**; `state.f` is
   widened to float64 and checked **only on the ledger cadence**. (Widening
   `f` every step is a 276 MB copy on the Test 2 grid and dominated the run.)
2. Added columns `eps_fluid_min`, `eps_fluid_max` (FLUID cells only, i.e. the
   population the new guard watches), `eps_filter_max`, `eps_filternb_max`
   (FILTER cells and their 3x3x3 dilation), `u_p99`, `u_med`.
3. Added `wetz_by_type`: **per-part wet-cell counts by z**, `part[z:n;z:n]`.
   This is the column the riser-order and front-contiguity checks read.

Ledger columns 1-17 are byte-compatible with the r2/r3 ledgers, so the
column-by-column comparisons below are direct. The last two columns are ragged
(they contain commas); `analyze.py` handles that.

`analyze.py` — summary + two-ledger diff + riser-order/contiguity check.
`face_audit_any.py` — copied unchanged from the mech2 fixture.

**On the pad and the audits.** `build_porous_zone` is called *after* the pad, so
the captured `*.types.npy` is the **padded** grid; cropping `[1:-1, 1:-1, 1:-3]`
recovers the original. For FGT@0.20 that crop reproduces the pre-fix grid
`r3_fgt20_ledger.csv.types.npy` **exactly** — identical face composition and
identical wrap-pair counts — which proves the pad is purely additive and
changes no geometry.

---

## A1 — `Test_Model/Full_Gating_Test` — PASS

The brief named two configs; both were run, because they have different
pre-fix baselines.

* **A1a, `--voxel 0.20 --refine 4`** -> 9,251 voxels, 1,354 solve cells at
  10.2 mm, coarsening x2. This is the "1,354 cells at 10.2 mm" config the brief
  quotes; baseline `r3_fgt20_ledger.csv` / `r3_fgt20.log` (**complete**, 18.0 s).
* **A1b, `--voxel 0.17 --refine 4`** -> 14,652 voxels, 2,239 cells at 8.6 mm.
  This is the config of `r3_fgt_ledger.csv`, the ledger the brief names.
  **Caveat, stated plainly:** that baseline is *not* a complete run. Its ledger
  stops at step 42,500 and `r3_fgt.log`'s last line is a 46 %-of-cells progress
  report at 6.00 s with no completion, i.e. the process was killed. There is no
  complete pre-fix baseline at 8.6 mm; comparison is over shared steps only.

### A1a results (post-fix)

| | value |
|---|---|
| status | **complete**, 28,882 steps, **6.00 s** sim, 72.0 s wall |
| fill | **1,354 / 1,354 cells (100 %)**, **95 % by mass** |
| delivered | **20.8 lb**; delivery 100 % of the 4.39 lb/s target at 3.00 s, 58 % at 6.00 s |
| non-finite | **0** rows of 57 |
| precursor aborts | **none** |
| negative populations | `neg_f` **0**, `neg_cells` **0**, `neg_porous_cells` **0** at every row |
| `eps_max` (all wet) | max **1.01111** |
| `eps_fluid_max` | max **1.01111** — inside the new `[-0.1, 1.1]` band by 8x |
| `eps_min` (all wet) | min **-0.71238** (INTERFACE cells; `eps_fluid_min` min **+0.00179**) |
| max\|u\| (lattice) | max **0.05567**, p99 **0.05042**, median **0.01616** |
| min fill temperature | 1502.3 C (above the 1460 C liquidus; no cold-shut warning) |
| wet mass monotonic | **yes** — 1 apparent -0.01 step on 1283, i.e. 8e-6, the ledger's `%.6g` rounding |
| warnings | 3, all the pre-existing 3-cell-channel marginality warning (sprue / runner / gate) |

**20.8 lb is not an over-pour.** 1,354 cells at 10.16 mm hold 1.420e-3 m^3 =
21.9 lb of WCB liquid; 95 % of that is 20.8 lb. The "19 lb" in the metering
line is the pre-pour sqrt-rule estimate used only to set the rate, not the
cavity capacity. Delivered mass and filled mass agree exactly.

### A1a — the riser now fills FROM the casting

The riser occupies z 8-15 in this grid; its own foot is z=8.

| | pre-fix (`r3_fgt20`) | post-fix (`a1_fgt20`) |
|---|---|---|
| first step with any wet riser cell | **1,500** | **15,000** |
| casting state at that moment | **10 wet cells**, metal still in the sprue | **628 wet cells**, casting full through z1-z8 |
| where in the riser | (pre-fix ledger has no per-part z profile; the mass z-profile shows mass at every plane z1-z20 at step 1,500, i.e. spread over the whole domain height while only 10 cast cells are wet) | **z=8 only, 2 cells** — the riser's own lowest plane |
| riser wet-cell count monotonic | **NO**: 14 -> 6 -> 32, and 214 -> 206 -> 214 (the flicker signature of a phantom pool) | **yes** |
| z-gaps inside the wet riser span | n/a | **0 of 57 rows** |

Post-fix riser progression: `z8(2)` -> `z8-9(48)` -> `z8-11(120)` ->
`z8-12(150)` -> `z8-14(210)` -> `z8-15(236/236)`. Monotonic, contiguous,
bottom-up. **This is exactly the requested check and it passes.**

No wet cells above the front anywhere: the per-part wet-z spans are contiguous
at every ledger row, for every part, for the whole run.

### A1a — column-by-column vs `r3_fgt20_ledger.csv`

Byte-identity is not expected (the brief says so, and the ledgers diverge from
step 500 — 76 wet cells post vs 96 pre — because the pre-fix sprue at x=134
wraps into the casting at x=0 from the first steps).

| column | pre-fix | post-fix | explanation |
|---|---|---|---|
| `wet_cells` final | 1354 | 1354 | same; both complete |
| **steps / fill time** | 86,646 / **18.0 s** | 28,882 / **6.00 s** | **removed wrap, intended.** Pre-fix delivery decayed 88 % -> 36 % -> 21 % -> **2 %** -> 3 % -> 3 % of target; the wrap fed the sprue mouth back-pressure from spurious wet cells and the rate controller throttled to nothing, so the endgame crawled for 12 s. Post-fix delivery holds **100 %** to 3.00 s. The metered pour is 19 lb / 4.39 lb/s = **4.3 s**; 6.0 s is credible, 18.0 s is 4x non-physical. |
| `wet_mass` final | 1252.79 | **1282.91** | **intended.** The post-fix casting ends **2.4 % fuller** (95 % vs 93 % by mass). |
| `max_abs_u` max | 0.27295 | **0.05567** | **intended.** 4.9x lower peak lattice velocity — the wrap was injecting momentum across 93 planes. p99 0.23329 -> 0.05042; median 0.02292 -> 0.01616. |
| `rho_max` max | 1.14212 | **1.03823** | **intended.** 14 % pressure excursions -> 3.8 %. |
| `wet_cells` monotonic | **50 drops** (worst -26) | **4 drops** (worst -24) | **intended.** Pre-fix flicker was the phantom pool forming and collapsing. The 4 remaining drops are real front rearrangement at a 3-cell choke. |
| `eps_max` max | 1.01087 | 1.01111 | **unchanged (+0.02 %).** Important control: FGT has **no filter**, so the deleted porous override was inert here. Fix 1 provably does not perturb a filter-free assembly. |
| `iface_cells` max | 558 | 442 | fewer spurious interface cells; consistent with the wrap's promote-dilation being gone. |
| `neg_*` | 0 / 0 / 0 | 0 / 0 / 0 | unchanged. |
| `min_temp` final | 1550.6 | **1502.3** | **intended, and physically correct.** Pre-fix the x-wrap let the continuously-fed sprue supply the casting and riser directly, so the coldest wet cell was always near fresh 1620 C metal. Post-fix the metal must travel sprue -> runner -> gate -> casting -> riser, a far longer thermal path, so the front arrives cooler. Still above the 1460 C liquidus. |
| `eps_min` min | -0.05765 | **-0.42048 / -0.71238** | **observation, flagged, not a regression.** These are INTERFACE cells (`eps_fluid_min` stays >= +0.0018 all run, so the new FLUID guard correctly does not fire). Post-fix the front advances at full metered rate instead of throttled, and `_redistribute` drains donor interface cells harder, so transient interface mass debt is deeper. It resolves — the run completes clean. Worth a look by the author: -0.71 of a cell of mass debt is large, and the same excursion landing on a FLUID cell would abort the run. |
| `nonfinite` | 0 | 0 | unchanged. |

### A1b results (`--voxel 0.17`, 8.6 mm, 2,239 cells)

| | value |
|---|---|
| status | **complete**, 50,949 steps, **9.00 s**, 190.7 s wall |
| fill | 2,239 / 2,239 cells (100 %), 96 % by mass, 21.6 lb |
| delivery | 100 % of the 4.33 lb/s target at 3.00 s, 66 % at 6.00 s |
| non-finite | 0 rows of 101; no precursor abort |
| `neg_f` / `neg_cells` max | **1** / **1** (single transient cell) — pre-fix same grid: **9** / **6** |
| `eps_fluid_max` max | 1.02791 (pre-fix 1.02279) |
| `eps_min` min | -0.42048 (interface; pre-fix -0.15101) |
| max\|u\| | max **0.24862**, p99 **0.15422**, median **0.01395** — pre-fix **0.48819 / 0.41790 / 0.04598** |
| `wet_cells` drops | **1** (worst -2) — pre-fix **10** (worst -31) |
| riser | first wet at step **17,000**, at **z=9 = the riser's own foot**, with the casting already 122-130 cells wet on each of z1-z8. **0 z-gaps** at any row. Final riser z9-17, 360 cells. |
| min fill temperature | 1425.8 C |
| warnings | 3-cell gate marginality; **min fill temp 1423 C below the 1460 C liquidus — incipient cold-shut**; **65 voxels with a cold-shut signature** |

Post-fix is better on every stability metric at this grid: peak velocity halved,
negative populations 9 -> 1, front flicker 10 -> 1. The two new warnings (cold
shut) are the honest consequence of the corrected, longer thermal path — pre-fix
the wrap hid it.

### A1a — face audit (the requested pad confirmation)

```
POST-FIX PADDED GRID (what the solver actually ran), shape (23, 10, 25):
  x=0 : 0 part cells      x=22 : 0 part cells
  y=0 : 0 part cells      y=9  : 0 part cells
  z=0 : 0 part cells      z=24 : 0 part cells
  WRAP-x / WRAP-y / WRAP-z : 0 metal<->metal pairs
```
**All six faces obstacle. Confirmed.** Independently, `_obstacle_mask` on the
saved grid returns `True` for all six faces, and the stage's own
faces-are-obstacle `RuntimeError` invariant did not fire on any of the four
assemblies run here.

```
SAME GRID, PAD CROPPED OFF (= the pre-fix grid), shape (21, 8, 21):
  x=0 : 84 {RUNNER 16, SPRUE 68}     x=20 : 92 {CAST 60, RISER 32}
  y=0 : 88 {CAST 88}                 y=7  : 88 {CAST 88}
  z=0 : 128 {RUNNER 28, GATE 20, CAST 80}   z=20 : 16 {SPRUE 16}
  WRAP-x: 60 pairs {RUNNER-CAST 16, SPRUE-CAST 12, SPRUE-RISER 32}
  WRAP-y: 88 pairs {CAST-CAST 88}
  WRAP-z: 16 pairs {RUNNER-SPRUE 16}
```
Byte-for-byte identical to the actual pre-fix `r3_fgt20_ledger.csv.types.npy`.
So FGT wrapped on **all three axes** pre-fix, including the **32 SPRUE->RISER
pairs across x** the brief names as the riser's direct feed. At 8.6 mm the same
audit gives 59 / 130 / 21 pairs (27 of them SPRUE-RISER). **Confirmed: the FGT
pre-fix numbers are not a valid baseline, and byte-identical comparison is
correctly not expected.**

### `HEADSPACE_CELLS` on FGT — a documented no-op, verified

On FGT the only metal on the original `+z` face is **16 SPRUE cells, and all 16
are the inlet mouth**, which the code deliberately excludes. `open_above` is
therefore empty and the headspace opens **0 cells**. That is the design working
as its docstring says ("A pour mouth is already at atmosphere by definition; it
needs no headspace"), not a defect. The riser top at z=15 is five planes below
the domain top and is vented by `_static_vent_mask`, not by the headspace.
The headspace *is* exercised on Test 2 — see B.

---

## A2 — `Test_Model/Filling_Test` — PASS

`--voxel 0.20 --refine 4` (the config `docs/filling_field_notes.md:107` records:
"0.20 in, coarsening 2"). 4,357 solve cells at 10.2 mm; padded grid (45, 27, 33)
from (43, 25, 29).

### Face audit of the UNPADDED types first, as the report asks

```
UNPADDED (= the pre-fix grid), shape (43, 25, 29):
  x=0 : 38 {CAST 8, RUNNER 30}       x=42 : 107 {CAST 44, RISER 63}
  y=0 : 39 {CAST 2, RUNNER 37}       y=24 : 93 {CAST 93}
  z=0 : 102 {CAST 2, RUNNER 30, CHILL 70}   z=28 : 20 {SPRUE 20}
  WRAP-x: 0 pairs (safe)
  WRAP-y: 0 pairs (safe)
  WRAP-z: 20 metal<->metal pairs {RUNNER-SPRUE 20}
```

**Yes, `Filling_Test` wrapped before the fix — on one axis only.** 20 columns
put a RUNNER cell on `z=0` directly opposite the **SPRUE mouth** on `z=28`.
That is the worst possible partner for a wrap: the sprue mouth is the Dirichlet
inlet, held wet every step by `ensure_interface_skin`, so the runner floor and
the pour mouth were 18-direction neighbours. It is a plausible cause of the
documented inlet throttling and the 57 % time-cap.

Post-fix padded grid: **all six faces 0 part cells, 0 wrap pairs on all three
axes.**

### A2 results

| | value |
|---|---|
| status | **complete**, 98,160 steps, **18.00 s**, 333.3 s wall |
| fill | **4,357 / 4,357 cells (100 %)**, **97 % by mass**, 70.0 lb |
| **vs documented pre-fix** | **time-caps at ~57 % of cells** |
| delivery | 96 % of the 7.98 lb/s target at 3.00 s, then 54 / 52 / 50 / 40 / 1 % |
| non-finite | **0** rows of 196; no precursor abort |
| `neg_f` / `neg_cells` / `neg_porous` max | 6 / 5 / **0** — transient, and far under the cascade guard's 2 %-of-wet-cells threshold (~87 cells) |
| `eps_fluid_max` max | **1.02657** — inside the band by 4x |
| `eps_min` min | -0.09316 (interface) |
| max\|u\| | max **0.44566**, p99 **0.09376**, median **0.01230** |
| wet mass monotonic | **yes, 0 decreases** |
| min fill temperature | **1283.1 C** |
| warnings | 3x 3-cell channel (sprue / runner / gate); **min fill temp 1283 C below the 1460 C liquidus**; **4,255 voxels with a cold-shut signature** |
| riser | first wet at step **18,500**, at **z=8 = the riser foot**, casting already wet z1-z10 (262 cells at z=6). Final riser z8-21, 1,511 cells. |

### Two honest caveats on A2

1. **10 of 196 rows show a one-plane gap in the riser wet-z profile.** Not a
   phantom pool. The profile at those rows is e.g. `{8:110, 10:1, 11:1}` and
   `{8:116, 9:104, 11:1, 12:1}`: the riser foot holds 95-116 cells while
   **1-4 cells** sit two to four planes higher. It appears only *after* the
   riser foot is substantially wet, never before; it is bounded at 1-4 cells and
   does not grow over 60,000 steps; and it resolves to a fully contiguous riser.
   Read as **splash off the metal entering the riser foot**, which is physical.
   Contrast the pre-fix Test 2 signature it must be distinguished from: 719
   over-filled riser cells at z 86-90, growing, seeded *before* the front
   arrived. Note `Filling_Test` has two risers, so `riser[...]` is their union.
2. **max\|u\| peaks at 0.446 lattice** (~Mach 0.77) — the highest of the four
   runs, on a 3-cell sprue/runner/gate. Pre-existing marginality of a 3-cell
   channel, not new (pre-fix FGT@8.6 mm peaked at 0.488), but it is the reason
   the 3-cell warning exists and it should stay on the record.

### Documentation defect this uncovers

`Filling_Test` **completes** post-fix. The string
"`Test_Model/Filling_Test, whose gate is also three cells, time-caps at 57 % of
cells`" is now false, and it is emitted by the solver itself — it appeared 3x in
A1a, 3x in A1b, 3x in A2 and 1x in B, i.e. in **every run**. It is baked into
`core/filling_stage.py`'s warning text and referenced in `docs/`. **The diff
under validation does not update it.** Flagged for the author; I did not edit it.

---

## A3 — full test suite — PASS

```
CASTSIM_THREAD_BUDGET=6 QT_QPA_PLATFORM=offscreen ./venv/bin/python -m unittest discover tests
Ran 1230 tests in 365.707s
FAILED (failures=1, skipped=11)
```
(Budget 6 rather than 8, because Part B was running concurrently on 10 of 18
cores. This affects wall time only.)

**failures = 1, errors = 0, skipped = 11.**

The one failure:
```
FAIL: test_a_deeper_tower_leaves_none_either
      (test_filling_stability.SubmergedInterfaceHealingTests.test_a_deeper_tower_leaves_none_either)
  File tests/test_filling_stability.py, line 1453
    self.assertEqual(result.status, STATUS_COMPLETE)
AssertionError: 'diverged' != 'complete'
```

Reconciled against the known pre-existing list:

| known item | fired? | reconciliation |
|---|---|---|
| `SubmergedInterfaceHealingTests::test_a_deeper_tower_leaves_none_either`, full-suite ordering only | **yes** | **On the list.** Confirmed by isolation: `python -m unittest tests.test_filling_stability.SubmergedInterfaceHealingTests` -> **Ran 9 tests, OK**. Also present, identically and as the *only* failure, in all three pre-fix baselines: `r3_suite.log` (1212 tests), `r2f_suite.log` (1202), `suite_after.log` (1192). |
| `test_deep_model InferenceTests::test_pipeline_end_to_end` order-flake (`pipeline.py:102 completion` unbound) | no | Did not fire this run. It is a flake and `test_deep_model` is not under `tests/`. Not counted against the diff. |
| `tests/test_filling_stability` importing `_Options` from `test_filling_stage` and mutating a shared `_Params.filling` | n/a | The ordering-leak mechanism behind item 1; no separate failure. |
| a test module stubbing `vtk` globally | no | No `vtk`-attributable failure. |

**Nothing outside the known list failed or errored. Zero new failures.**

The test count rose **1212 -> 1230 (+18)**, which is exactly the count of
`def test_` added by the diff: `PorousPairWeightTests` (6), the two
fill-fraction-guard tests, `PeriodicWrapTests` (4), `MoldPadTests` (6) —
including `test_the_riser_top_keeps_an_open_headspace` and
`test_the_result_still_maps_back_to_the_right_voxels`, which cover the
headspace and the `to_voxels` cropping respectively. The new behaviour is
covered by tests, and they pass.

---

## B — the acceptance run, full Test 2 pour — **FAIL**

```
CASTSIM_THREAD_BUDGET=10 QT_QPA_PLATFORM=offscreen ./venv/bin/python fill_driver.py \
  --dir "/Users/brendonjarett/Downloads/Test 2" --voxel 0.17 --refine 4 \
  --max-fill-time 30 --ledger b_t2_ledger.csv --ledger-every 500
```
Log `b_t2.log`, ledger `b_t2_ledger.csv`, types `b_t2_ledger.csv.types.npy`.

Setup reproduced the prior rounds exactly: 344,532 voxels; auto-refine
coarsening x2 -> x1 (the gate measured 1.0 cell at x2); **339,824 metal cells at
4.3 mm**; solve grid **(137, 137, 98)** = (135, 135, 94) + 1 pad on every face
+ 2 headspace planes; inlet 136 cells (11 at the pour point, widened to the
sprue mouth); metering **20.8 lb/s**; gate 3.0 cells (narrowest).

### Outcome

**`status=diverged`, 20,200 steps, 1.39 s of a 22 s pour. 949.5 s wall.**

First non-finite `state.mass` at **step 20,192**; the stage's wet-mass ceiling
guard caught it on the next 200-step boundary, **step 20,200**, and aborted with
correct, unsoftened messaging ("SOLVER DIVERGED at step 20,200 -- this fill
result is not physical and must not be used"), discarding the results and
labelling the misrun/entrapment figures as artefacts. **The abort machinery
behaved impeccably; the physics did not.**

| acceptance criterion | result |
|---|---|
| no non-finite anywhere | **FAIL** — non-finite at step 20,192 |
| no precursor abort | **PASS in letter, FAIL in spirit** — neither the negative-population nor the new fill-fraction guard fired. The fill-fraction guard checked at step 20,000 and found nothing (`eps_fluid_max` = 1.00827). `eps_fluid_max` first breached 1.1 at step **20,160** -- **32 steps** before the NaN, and inside the guard's own 200-step interval. |
| filter-outlet fill fraction bounded | **PASS, decisively** — see below |
| no wet riser cells above the front | **PASS, absolutely** — see below |
| wet mass monotonic | **PASS up to the divergence** — 0 decreases across 40 clean rows |
| delivery near 20.8 lb/s | **PASS** — measured **20.58 lb/s**, 99 % of target |
| honest endpoint | reported here as a failure |

### Fix 1 — VERIFIED WORKING

`eps` among FILTER cells, and among FILTER cells plus their full 3x3x3
dilation, over all 40 clean ledger rows:

| | pre-fix (mech1 REPORT) | post-fix (this run) |
|---|---|---|
| `eps` in the filter zone | 1.046 -> 3.711 -> **7.76**, rising 2.7e-4/step | **max 1.00793**, min 0.93038, **flat** |
| `eps_min` in the zone | fell to -1.17 then **-9.5** | never below 0.930 among filter cells |
| over-full FLUID samples at the filter's bottom plane | 786 of 2,009 | `eps_fluid_max` never above 1.033 grid-wide |

The positive-feedback amplifier is **gone**. Filter-cell `eps` sits on 1.005
and stays there. Fix 1 is validated on the real assembly, at the exact location
the mechanism was diagnosed.

### Fix 2 — VERIFIED WORKING

* Face audit of the padded Test 2 grid: **all six faces obstacle** (`_obstacle_mask`
  returns True on all six); 0 metal<->metal wrap pairs on all three axes. The
  pre-fix grid had **302 across z (298 of them CAST@z=0 <-> RISER@z=93), 117
  across x, 55 across y**.
* All **339,824** metal cells preserved; part extents shifted by exactly +1 on
  each axis and otherwise unchanged.
* **`HEADSPACE_CELLS` is exercised here** (unlike FGT): 1,968 metal cells sit on
  the original `+z` face (RISER 1,801, SPRUE 132, CAST 35); the 132 sprue-mouth
  columns are excluded as designed, leaving **1,836 columns x 2 planes = 3,672
  cells of atmosphere** opened above the riser tops.
* **The riser had ZERO wet cells at the moment of death**, and zero at every one
  of the 40 clean ledger rows. Pre-fix, by step 16,493, it held 719 over-filled
  interface cells at z 86-90 and **1,263 negative-population riser cells**.
  The wrap seed is **completely** gone.

### The clean run, up to the failure

| step | t (s) | wet cells | % cells | wet mass | lb | max\|u\| | eps_max | eps_filter_max | neg_f | neg_cells | nonfinite |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 0.034 | 884 | 0.26 % | 766.4 | 0.97 | 0.0159 | 1.00165 | (dry) | 0 | 0 | 0 |
| 18,000 | 1.237 | 24,108 | 7.09 % | 20,378.9 | 25.7 | 0.0497 | 1.00821 | 1.00538 | 0 | 0 | 0 |
| 18,500 | 1.271 | 24,703 | 7.27 % | 20,939.1 | 26.4 | 0.0475 | 1.00747 | 1.00585 | 0 | 0 | 0 |
| 19,000 | 1.306 | 25,144 | 7.40 % | 21,499.3 | 27.1 | 0.0653 | 1.00991 | 1.00573 | 0 | 0 | 0 |
| 19,500 | 1.340 | 25,657 | 7.55 % | 22,059.6 | 27.8 | 0.0512 | 1.00747 | 1.00460 | 0 | 0 | 0 |
| **20,000** | 1.374 | 26,278 | **7.73 %** | 22,619.8 | **28.5** | **0.0680** | **1.00827** | **1.00498** | **1** | **1** | **0** |
| **20,192** | 1.388 | 26,437 | 7.78 % | **nan** | - | **6.53e+16** | nan | nan | **3,812** | **739** | **1** |

`dt` = 68.714 us; cell capacity 0.5725 g = 1.262e-3 lb; 339,824 metal cells =
194.5 kg = **428.9 lb** of cavity (against the 459 lb nameplate and the stage's
own 434 lb sqrt-rule figure).

**Reported figures at the abort** (all invalidated by the divergence, and the
stage says so):
* fill: **7.78 % by cells**, **6.66 % by mass**
* poured: **~28.5 lb of 459 lb**
* fill time: n/a — did not complete
* min fill temperature over the clean run: **1517.3 C** (still above liquidus)
* delivery: **20.58 lb/s measured vs 20.8 lb/s target = 99 %**, unthrottled
* negative populations: **0** at every clean row until step 20,000, which had
  exactly **1** cell; then 3,812 populations across 739 cells at the failure
* max\|u\| over the clean run: **max 0.48485, p99 0.32343, median 0.05745** (1,000 rows at 20-step cadence from the bisection rerun; the 500-step ledger's 0.24035 / 0.19430 / 0.06203 is aliased -- see below)
* wall: 949.5 s (15.8 min), 20,200 steps, ~47 ms/step
* warnings (4): WCB not in the Foseco steel flow tables so the filter's rated
  flow could not be checked; gate 3.0 cells marginal; **SOLVER DIVERGED**;
  323,356 metal voxels never filled (95.2 % misrun); 886 voxels of trapped air
  (peak 1,507) — the last two being divergence artefacts by the stage's own
  admission.

### Characterization of the NEW failure

**Step.** Non-finite mass first observed at **step 20,192** (t = 1.3880 s).
Fully healthy at step 20,000: `eps_fluid_max` 1.00827, `max|u|` 0.0680, one
negative-population cell. **The blow-up takes 192 steps to go from `max|u|` = 0.068 to 6.53e+16**, at a
measured exponential growth rate of **0.01554/step** (doubling every 45 steps).
That is a linear LBM instability, not the pre-fix mechanism, which grew at
2.7e-4/step and announced itself in `eps` 8,800 steps ahead.

**Location.** The 287 non-finite cells (25 dumped) occupy a single small pocket:
padded `x[116,117] y[49,52] z[21,27]`, i.e. **original `x[115,116] y[48,51]
z[20,26]`** — a 2 x 4 x 7 pocket. Composition of the dumped cells:
**14 FILTER, 11 RUNNER**. All **25 are `flag=3` = `FLAG_FLUID`**.

By part: Test 2 has **two** filters, `Filter.step` and `Filter (1).step`, which
label as two components, padded `x[105,123] y[41,56] z[15,33]` (#1) and
`x[105,123] y[80,95] z[15,33]` (#2), 2,354 cells each. **All 14 filter cells are
in filter #1; filter #2 has none.** The pocket sits at the filter's **high-x
end**, adjacent to the sprue (`x[118,135]`), i.e. **the filter INLET, not the
outlet** — and it straddles the filter/runner boundary within the plane (at
x=116, y=49-50 are FILTER while y=51-52 are RUNNER). So: **the filter #1 inlet
face, at the filter-runner interface, mid-height (z 20-26 of the filter's
15-33 span).** Pre-fix, mechanism 1 blew up at the filter **outlet**.

**Signature — a momentum/density runaway, not a mass one.** From the ledger's
non-finite dump:

```
#  cell (117,49,23) type=8 flag=3 rho=2.04542e+14 mass=-inf eps=-inf por=0.85 u=[-0.5039 -0.5039 -3.0e-04]
#  cell (117,50,23) type=8 flag=3 rho=-2.04518e+14 mass=nan  eps=0    por=0.85 u=[ 1.2166e+07 -7.4e-03 6.5e+01]
#  cell (117,51,23) type=7 flag=3 rho=-1.02264e+15 mass=nan  eps=0    por=1    u=[ 1.0226e+15 -1.0226e+15 5.48e+09]
#  cell (116,51,24) type=7 flag=3 rho=10.9773      mass=-inf eps=-inf por=1    u=[-0.9985 -0.0821 -0.0059]
#    nb (115,49,24) type=8 flag=3 mass=4.4182e+19 por=0.85
```
`rho` reaches **+/-2e14 to 1e15** and `u` reaches **1.2e7 and 1.0e15 lattice**,
while `eps` is 0 or -inf only *because* mass has already gone infinite. Several
cells carry a near-perfectly antisymmetric `u = [+a, -a, ~0]` — one lattice
direction dominating, the classic single-mode LBM instability. Adjacent healthy
cells one step away still hold `rho` ~0.88-11 and mass ~0.29, so the runaway is
**spatially sharp**, not a diffuse drift.

**Why the new guard did not catch it.** `FLUID_FILL_FRACTION_MIN/MAX` watches
`eps = m / (rho * porosity)`. Here `rho` blows up **in the denominator at the
same time** as `m` blows up in the numerator, so `eps` stays near 1 until both
are non-finite. The guard is designed for a mass-bookkeeping failure at healthy
`rho` — which is exactly the pre-fix mechanism it was written against, and it
would have caught that one 8,800 steps early. It is structurally blind to a
density/momentum runaway, and this failure is one. **No criticism of the guard
is intended; the observation is that it does not cover this mode.**

**Operation.** The only *new* RuntimeWarning versus the pre-fix run is
`physics/filling.py:1319: RuntimeWarning: overflow encountered in cast` at
`state.f[:, px, py, pz] = seeded.astype(np.float32)` — the promoted-cell
seeding in `_seed_promoted_cells` writing float64 populations already past the
float32 range. That is downstream of the blow-up, not its cause, but it means
`state.f` is float32, so a population only needs to exceed **3.4e38** to become
`inf` — a much lower ceiling than float64 and worth the author's attention given
the observed 4.4e19 masses. The other three warnings (matmul divide-by-zero /
overflow / invalid at `filling_stage.py:2916`, the inlet velocity measurement on
still-dry cells) are **pre-existing** — they appear identically in the pre-fix
`r3_t2.log`.

**Is it a regression from this diff?** Cannot be answered from these two runs,
and I will not guess. The pre-fix run died at step 16,493 by a different
mechanism at a different location, so it never reached step 20,192; there is no
observation of HEAD's behaviour there. What *can* be said:
* Post-fix survives **3,700 steps further** than pre-fix (20,192 vs 16,493).
* Both diagnosed mechanisms are gone, verified at their own diagnosed locations.
* The new failure is at a **different location** (filter #1 inlet vs the filter
  outlet), with a **different observable** (`rho`/`u` vs `eps`), on a
  **~50x faster timescale** (<192 steps vs 8,800).
* Fix 1 removed the eps-weighted exchange **in porous cells**, and this failure
  is **in porous cells**. That adjacency is suggestive and should be
  investigated, but the diff's own reasoning (a weight of 1 makes a porous FLUID
  cell's mass track `rho` and drift by `(1-p)*drho`, bounded) predicts a *mass*
  drift, and no mass drift is visible: `eps_filter_max` is pinned at 1.005 for
  20,000 steps. The observed failure is in momentum. **A momentum-side
  interaction with the committed `f1daf97` change ("apply the porous drag to the
  momentum, not only the equilibrium; filter cells gain a terminal velocity") is
  at least as plausible a suspect and is not part of this diff.** I did not test
  either hypothesis.

**Reproduction — the failure is reproduced three times, bit for bit:** the
acceptance run (21:07, old tree), the 20-step bisection rerun (21:26, old tree)
and the re-run on the current tree (21:50). All three give non-finite mass at
step **20,192** and abort at **20,200** with identical ledger values at every
shared row. This is deterministic, not a race.

**Onset bisection — DONE.**
A second Test 2 run (`b2_t2_fine.log`, `b2_t2_fine_ledger.csv`) with
`--max-fill-time 1.45` and `--ledger-every 20` (identical trajectory:
`max_fill_time_s` only sets `max_steps = max_time / dt`; `dt`, the metering and
the fixed 200-step `DIVERGENCE_CHECK_INTERVAL` are untouched). It is
**bit-identical** to the first run at every shared 500-step boundary —
`wet_cells`, `wet_mass` and `max|u|` agree to the last printed digit at steps
500 through 20,000 — and it produced the **same** non-finite step (20,192), the
**same** abort step (20,200) and the same warning set. 1,095.1 s wall.

At 20-step resolution the runaway is a clean exponential:

| step | max\|u\| | eps_fluid_max | eps_filter_max | neg_f | neg_cells |
|---|---|---|---|---|---|
| 20,000 | 0.0680 | 1.00827 | 1.00498 | 1 | 1 |
| 20,020 | 0.0778 | 1.00721 | 1.00472 | 1 | 1 |
| 20,040 | 0.0911 | 1.00805 | 1.00562 | 1 | 1 |
| 20,060 | 0.1096 | 1.00804 | 1.00710 | 1 | 1 |
| 20,080 | 0.1357 | 1.01053 | 1.00627 | 1 | 1 |
| 20,100 | 0.1741 | 1.01635 | 1.00494 | 3 | 1 |
| 20,120 | 0.2418 | 1.02961 | 1.00591 | 5 | 2 |
| 20,140 | 0.4080 | 1.08635 | 1.00700 | 14 | 6 |
| **20,160** | 0.5660 | **1.73963** | **1.73963** | 66 | 25 |
| 20,180 | 0.9376 | 1.31199 | 1.18676 | 50 | 27 |
| **20,192** | **6.53e+16** | nan | nan | 3,812 | 739 |

An exponential fit over steps 20,020-20,180 gives a growth rate of
**0.01554 per step — e-folding every 64 steps, doubling every 45**. That is a
**linear LBM instability**, three orders of magnitude faster than the pre-fix
mass amplifier's 2.7e-4/step.

**The decisive finding: this run was marginally stable for 16,000 steps before
it tipped.** The 500-step ledger of the acceptance run badly undersampled it.
At 20-step resolution:

| step window | max\|u\| max | max\|u\| median | rows with max\|u\| > 0.1 | eps_fluid_max max | eps_filter_max max | neg_cells max |
|---|---|---|---|---|---|---|
| 0-2,000 | 0.0640 | 0.0448 | 0/100 | 1.0038 | (filter dry) | 0 |
| 2,000-4,000 | 0.0785 | 0.0474 | 0/100 | 1.0330 | (dry) | 0 |
| 4,000-6,000 | **0.2790** | 0.0670 | 19/100 | 1.0302 | (dry) | 4 |
| 6,000-8,000 | **0.4664** | 0.0760 | 36/100 | 1.0342 | 1.00341 | 12 |
| 8,000-10,000 | **0.4849** | 0.0557 | 5/100 | 1.0394 | 1.00905 | 12 |
| 10,000-12,000 | 0.1758 | 0.0573 | 3/100 | 1.0130 | 1.00937 | 0 |
| 12,000-14,000 | 0.2689 | 0.0754 | 26/100 | 1.0261 | 1.00871 | 3 |
| 14,000-16,000 | 0.1385 | 0.0542 | 5/100 | 1.0115 | 1.00844 | 1 |
| 16,000-18,000 | **0.4833** | 0.0658 | 17/100 | **1.0637** | 1.00942 | 4 |
| 18,000-20,000 | 0.1793 | 0.0515 | 12/100 | 1.0215 | 1.00863 | 1 |
| 20,000+ | **6.5e+16** | 0.2079 | 8/11 | **1.7396** | **1.7396** | **739** |

So `max|u|` spiked to **0.28-0.49 lattice** on **123 separate ledger rows**
starting at step 4,180, and **damped every single time** — until the spike that
began at step ~20,020 did not. Corrected clean-run velocity statistics
(steps <= 20,000, 1,000 rows at 20-step cadence): **max 0.48485, p99 0.32343,
median 0.05745**. The figures I gave from the 500-step ledger
(0.24035 / 0.19430 / 0.06203) were aliased and are superseded.

This reframes the failure. It is **not** a monotone runaway. It is a
**marginally stable configuration** — a 3-cell gate, a jet from the sprue into a
porous zone, `max|u|` routinely at 6-10x the 0.05 lattice design velocity — that
finally failed to damp. Note the geometry: the gate is 3.0 cells, which fires
the solver's own "marginal, not a result" warning, and the excursions begin at
step 4,180, well before the filter is even wet (first `eps_filter_max` sample is
in the 6,000-8,000 window).

**Fix 1 is exonerated in the same table.** Over ~14,000 steps of a wet filter,
`eps_filter_max` never exceeds **1.00942** and shows no trend. It only reaches
1.74 at step 20,160, *after* `max|u|` has already grown 8x — the filter is
**dragged along by the instability, not driving it**. `eps_fluid_max` likewise
oscillates in 1.01-1.06 with no drift across 20,000 steps.

**Why the new fill-fraction guard missed it — precisely.**
`eps_fluid_max` first exceeds the band's `1.1` ceiling at step **20,160**, which
is **32 steps before the NaN** and **less than the guard's own 200-step check
cadence**. The guard checked at 20,000 (1.00827, clean) and next at 20,200 (by
then NaN, and the wet-mass ceiling guard fires first in the same block). The
guard's own docstring calibrates it on a mechanism that gave **8,800 steps** of
warning; on this failure mode the band gives **32**. For reference, `1.05` would
have been crossed at step **17,420 — 2,772 steps of lead**. Two observations for
the author, offered as measurements and not as a fix:
* the band `[-0.1, 1.1]` cannot see a failure whose e-folding time (64 steps) is
  shorter than the 200-step check interval;
* `1.05` would have flagged this run 2,772 steps early, but the same table shows
  `eps_fluid_max` reaching 1.0637 at step ~17,420 in a run that then continued
  normally for 2,700 steps, so 1.05 would also be a **false positive** on a
  healthy stretch of this very assembly. Tightening the band alone does not
  resolve this; the missing precursor is a **velocity** one, and none exists.

### Not attempted

**The full pipeline (fill -> solidification -> defects) was NOT run.** Its
precondition was a completed fill, and the fill diverged; seeding
solidification from a divergent fill would produce a fabricated casting result.

---

## Files

| path (all under `Test/session_diag_2026-09-02/validation/`) | what |
|---|---|
| `REPORT.md` | this report |
| `fill_driver.py` | adapted headless driver (3 changes above) |
| `analyze.py` | ledger summary / diff / riser-order check |
| `face_audit_any.py` | copied from the mech2 fixture, unchanged |
| `a1_fgt20.log`, `a1_fgt20_ledger.csv`, `.types.npy` | A1a, FGT @ 0.20 |
| `a1b_fgt17.log`, `a1b_fgt17_ledger.csv`, `.types.npy` | A1b, FGT @ 0.17 |
| `a2_ft20.log`, `a2_ft20_ledger.csv`, `.types.npy` | A2, Filling_Test @ 0.20 |
| `*_types_unpadded.npy` | pad-cropped grids used for the face audits |
| `suite.log` | A3, full unittest run |
| `b_t2.log`, `b_t2_ledger.csv`, `.types.npy` | **B, the acceptance run that diverged** |
| `b2_t2_fine.log`, `b2_t2_fine_ledger.csv` | B onset-bisection rerun, 20-step cadence, 1,011 rows -- bit-reproduces the divergence |
| `b_t2.pid`, `b2.pid` | PIDs of the two Test 2 runs (both finished) |
