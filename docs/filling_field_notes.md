# Mold filling: field-test notes

What the filling stage does, how to drive it, and — more importantly — what its
answers are and are not good for. Written for a field-test campaign, not as a
design document; `docs/filling_implementation.md` is the plan and
`docs/ARCHITECTURE.md` describes where the code sits.

## Current state — read this first

**Round 5 removed the divergence round 4 uncovered.** The mold-pad and
filter-exchange fixes below cleared both of Test 2's blow-ups and exposed a
third, at step 20,200: an **undamped odd-parity momentum mode**. It is fixed by
one clamp, and the fix is not free — read "What moved, and what it cost".
Separately, "Rounds 1-2 review" below covers three guards that were reporting on
the wrong object and so failed silently; those are independent of the clamp and
of the pad.

### The defect: ω⁻ → 0 as τ → ½

In TRT, `ω⁺` sets the shear viscosity and `ω⁻` does not. `ω⁻` enters only
through Λ = (1/ω⁺ − ½)(1/ω⁻ − ½), which fixes where halfway bounce-back puts the
wall **and** how fast the odd, momentum-carrying "ghost" modes are damped.
Λ = 3/16 is the wall-optimal choice — and holding it as τ⁺ → ½ sends ω⁻ → 0.
Every casting lattice sits within 1e-5 of ½, and `lattice_units`' own docstring
says shrinking `dt` pushes τ *closer* to the limit, so this is not tunable away
by resolution.

Measured on Test 2 at τ⁺ = 0.500010: **τ⁻ = 18,060 steps.** Over the ~200 steps
a mode needs to double, the odd part of `f` is a pure accumulator of the forcing
with no restoring term. What grew in that hole was a **period-2 checkerboard on
one lattice diagonal**, in the **non-porous runner cells one cell outside filter
#1's inlet face** — the argmax of |u| alternating between two diagonal
neighbours for **31 consecutive steps without a break**, `u` converging on
(−a, +a, 0), growing super-exponentially from |u| = 0.016 at step 19,000 to 0.94
at 20,180 and taking the field non-finite at **20,192**. Extrapolating the
earliest growth rate backwards puts the onset near step **15,000**: it had been
growing for 5,000 steps.

**Neither dissipation mechanism the solver has could touch it.** Smagorinsky
senses the *symmetric* non-equilibrium stress, and an odd-parity staggered mode
puts its energy in the antisymmetric moments, so `tau_eff` barely moves — and it
is switched off inside filter cells by design anyway. Measured against
`smagorinsky_relaxation` directly at this τ: on a quiescent field it leaves
τ⁻ at 18,060, and at a realistic strain (`tau_eff` 0.5001) it only brings τ⁻
down to 1,400–4,700 — still hundreds of times too slow. It takes a *violent*
strain, `tau_eff` above 0.53, before the eddy viscosity reaches τ⁻ ≈ 5 on its
own, which is to say only after the damage is done. The Guo-Zhao porous drag
is a strict contraction at every velocity (sup |m′/m| = 0.99998, maximum damping
at |u| = 0.72 where momentum is annihilated in one step) and is **identically
absent** in the cells the mode grew in: they are porosity 1 with c₁ = 0 and a
measured damping ratio of exactly 0 at all 61 samples. Removing the drag from
the jet fixture kills it at *every* velocity including the 0.05 design velocity;
doubling it cuts peak |u| from 11.27 to 3.51. **The drag is the stabiliser** —
`f1daf97` was right and should stay.

Also ruled out by measurement: Mach/compressibility (the mode grows from
Ma 0.027, *below* the design velocity, with ρ inside 1.0000–1.0121 throughout),
any interaction with round 4 (the runaway cells are FLUID at ε ≈ 1 with no
mass-side observable at all), and float32 population overflow (max |f| over the
whole growth window is 6.07, 37 orders below the ceiling — the overflow warning
is downstream).

### The fix: cap τ⁻ at 5

`physics.fluid_flow.TAU_MINUS_MAX = 5`, applied in `omega_minus_from_magic`, in
`smagorinsky_relaxation` and inlined in `collide_kernel` — all three, or the two
step paths would solve different schemes and `tests/test_filling_parity.py`
would fail, which is the safety property. It binds only where
τ⁺ < Λ/(τ⁻ₘₐₓ − ½) + ½ = **0.541667**, so anything at a larger τ keeps the exact
Λ = 3/16 wall location, including the Poiseuille wall-position benchmark at
τ = 0.8. Pass `tau_minus_max=float("inf")` to `lattice_units` where exact Λ is
the thing being measured; the ceiling rides on `LatticeUnits.tau_minus_max` and
all three relaxation sites read it from there, so that escape hatch reaches the
kernel and the eddy-viscosity path as well as `units.omega_minus`. It has to be
one authority per state: while two of the three read a module global instead,
asking for `inf` changed one site in three and put a 3,600× split between the
two step paths.

### What moved, and what it cost

On Test 2, τ⁻ = 18,060 → 5:

| | before | after |
|---|---|---|
| outcome | `diverged`, **step 20,200** | `time_capped`, **step 23,284** |
| first non-finite | step 20,192 | **none** |
| max\|u\| over the run | **0.938** | **0.052** |
| max\|u\| p99 / median | 0.352 / 0.058 | 0.046 / 0.034 |
| ledger rows with max\|u\| > 0.1 | **130** | **0** |
| negative populations, last row | 152 | **0** |
| \|u\| at the runaway cell | 0.016 → 0.94 | **0.0055, flat** |

Three things in that table matter beyond the divergence:

1. **The mode never forms**, rather than being damped after the fact: |u| at the
   runaway cell sits at 0.0050–0.0064 for 4,200 consecutive samples with a
   fitted growth rate of +4.9e-5/step, which is zero to the precision of the
   fit.
2. **All 123 of the earlier "damped velocity excursions" are gone too.** The
   "marginally stable for 16,000 steps" character recorded in round 4 was not
   physical turbulence — it was this same ghost mode, excited repeatedly and
   only marginally decaying each time.
3. **It is an accuracy fix as much as a stability fix.** Compared at equal
   steps, max|u| roughly halves throughout (0.069 → 0.039 at step 6,000;
   0.068 → 0.033 at 20,000). The unfixed velocity field's maximum was ghost-mode
   contamination by about a factor of two for the whole run, and the median now
   sits at 0.034, under the 0.05 the timestep was sized for.

**The cost, stated plainly: this is not byte-identical for any real pour.** Every
casting lattice runs below τ = 0.541667, and the mode is in non-porous cells so
the change cannot be confined to filters. Wet mass is unaffected — it agrees to
**0.015 %** at step 20,000 — but wet cells differ by **−2.06 %**, and that is
the effective wall moving off the node midpoint. Solved exactly for steady
Stokes-TRT Poiseuille, `(H_eff/2)² = (H/2)² + (4/3)(Λ − 3/16)`: the clamp costs
a flat **0.25 cell²** of flow-carrying width at any width, which is **−11.1 % of
the flow through a 3-cell channel** and −25 % through a 2-cell one, against
exactly 0 before. Viscosity is untouched. It is intrinsic to damping the mode
rather than a property of the value 5 — see "Λ is not 3/16 on any real pour"
under Known limits for the full table and what it means for the choke
thresholds.
`Full_Gating_Test` and `Filling_Test` are therefore **no longer bitwise
regression sentinels** for this change and must be re-baselined on physical
outcomes (fill fraction, fill time, delivered mass, misrun count). On the
evidence above the new front is the better answer, since the old one was being
advanced by a velocity field with a spurious factor-of-two maximum.

**Test 2 still does not finish.** It stops on the fill-time cap, not a
divergence. Mechanism 3 being fixed makes it a physically meaningful run that
has to be given enough time; it does not by itself complete the pour.

**One side effect worth knowing about: the clamp partially masks round 4's
filter defect on quiescent fixtures.** The deleted eps-weighted porous exchange
got its gain from the exchange *amplitude*, and relaxing the odd moments at
τ⁻ = 5 instead of 18,060 cuts that amplitude. On the closed-box witness the old
rule used to grow a 5 % seeded error by +4.2 % over 2,000 steps and +14 % over
4,000; it now settles at a fixed +0.51 % and stays there to step 8,000. On the
free-surface variant the two rules agree to six figures. The round-4 fix is not
thereby unnecessary — the real assembly's runaway had a live front and reached
ε = 7.76 at the filter outlet — but the two-second fixtures can no longer show a
runaway, so `PorousPairWeightTests` was recalibrated to assert the 9x *bias*
separation that remains, and its free-surface case is now labelled an invariant
test rather than a discriminator. A future reader tempted to conclude the porous
override was harmless should read the diagnosis report, not the fixture.

### The next precursor to build: a per-cell velocity rise counter

Designed and calibrated, **deliberately not shipped this round** — see the
caveat at the end.

A precursor on the **global** max|u| cannot work here, and that is measured. A
sliding log-linear fit of max|u| over the run, swept over window length and
slope threshold: the one setting that catches the real event (10 samples, slope
0.0015) also fires **22 false positives** before step 19,700, and every tighter
setting either fires only inside the pour transient or never fires at all. The
reason is that the growing mode is not the global maximum until step 19,960 —
the sprue's free-fall jet dominates max|u| for the first 19,900 steps. **A global scalar cannot see a localised mode growing
three times slower than the noise floor.** For reference, the FLUID fill-fraction
band gave 32 steps of warning on this failure and global max|u| gives none.

What distinguishes the real event is not magnitude but **persistence at a fixed
location**, and that needs two extra grids and O(1) work per cell per check: a
`float32` grid of |u| at the previous check (7 MB at Test 2's size) and an `int8`
grid of consecutive-rise counts. Per check: `counter += 1` where |u| rose, else
0; fire when at least N wet cells have `counter >= K` and `|u| > FLOOR`.
Calibrated on 913 traced pocket cells:

| K | FLOOR | N | fires at | lead to NaN |
|---|---|---|---|---|
| 6 | 0.01 | 3 | 19,800 | 392 steps |
| **8** | **0.01** | **3** | **19,840** | **352 steps (27 ms)** |
| 8 | 0.02 | 3 | 19,940 | 252 steps |
| 10 | 0.01 | 3 | 19,880 | 312 steps |

An equivalent per-cell log-slope detector (ln|u| slope > 0.0015/step over 10
samples, r² > 0.95) fires at step 19,660 — 532 steps of lead — in 33 of the 913
cells, all inside the one growing mode.

**Why it is not shipped: N = 3 is calibrated on 913 cells in one pocket, not on
339,824.** A rising front legitimately accelerates, so N will need to be well
above 3 on a full domain, or the counter restricted to cells that have been
FLUID for a few hundred steps. Re-tune it against a *healthy* fill
(`Full_Gating_Test`, `Filling_Test`) and raise N until it is silent before
shipping it. A cheaper qualitative alternative worth measuring first: the mode's
signature is a period-2 alternation, and a physical flow at dt = 77 µs cannot
change by 3× in one step, so a detector on
max |u(n) − u(n−1)| / max(|u(n)|, floor) would key on the mode's *character*
rather than its amplitude, for one extra grid and two consecutive steps.

Full diagnosis, fixture and validation: `Test/session_diag_2026-09-02/`.

### Rounds 1-2 review: three guards that could not see what they were built for

Independent of the τ⁻ clamp and the mold pad — these are the three findings from
the adversarial review of `460ca0f..57b07d7` that survived three-refuter
verification. All three are guards that report on the *wrong object*, so all
three failed silently rather than loudly.

**The choke-thickness guard pooled every body of a part into one median.** Part
codes are per *type*, not per instance: every gate in an assembly is
`PART_TYPE_GATE`. `measure_choke_thickness_by_part` took one median over all of
them, so a healthy body outvoted a sealed one — and `Test 2` itself ships **two
GATE bodies and two FILTER bodies**. Measured on a synthetic pair sharing the
code, one 6 cells thick and one 2: the pooled median reads **5.0 cells**, above
even the marginal threshold, so the stage said nothing at all, while the 2-cell
plate reads **1.0** and is sealed shut. `choose_fill_coarsening` consumed the
same dict, so it did not refine either. Each code is now split into connected
bodies and the **thinnest body** is what is reported and acted on.

The connectivity is **6** — face contact only. It was 26 for one round, on the
argument that a corner touch is one fused piece and that splitting would invent
thin components; both halves were wrong. 26-connectivity fuses two same-code
bodies meeting at a single corner, which reinstates the very defect being fixed:
measured on an 8-cell and a 2-cell GATE plate sharing one corner, 26 reports
**(7.0,)** and says nothing where 6 reports **(1.0, 7.0)** and hard-warns — the
same silence as the pooled median. And splitting costs nothing, because
`_local_thickness_cells` is one map over the whole metal path that bodies only
*index* into: a 4-cell-thick diagonal staircase reads 3.0 under both, and a
1-cell diagonal string reads 1.0 under both. A corner-only contact is one
diagonal D3Q19 link, so fusing across it asserts a channel that is not there as
well as hiding the thin body.

**Bodies below 8 cells do not vote.** Separating bodies exposed a second
problem: a body of a handful of cells reads 1.0 at *every* coarsening, so one
stray fragment forced the refinement loop to coarsening 1 — 8x the cells and a
shorter timestep — and was still 1.0 there, leaving a hard warning nothing could
clear. `MIN_CHOKE_BODY_CELLS = 8` is a 2x2x2 block: below that a body has no
cross-section in any orientation. The floor is deliberately not the 27 of a
3x3x3, because real sealed gates land between the two (a 3x3x1 plate is 9 cells,
a 5x5x1 is 25) and those must still force a refinement. Fragments are reported —
in `choke_thickness_bodies_by_part`, in `choke_body_fragments_by_part` and in a
warning of their own — and simply do not drive the grid. A code made *only* of
fragments is still measured on them, so the guard cannot go quiet on the
geometry it should be loudest about. `choke_thickness_by_part`
keeps its scalar-per-part shape, because the pipeline payload and its readers
depend on it; the per-body detail is a new field alongside,
`choke_thickness_bodies_by_part`. Warnings now name the body — "the thinnest of
the 2 gate bodies" — because sending a reader to "the gate" when one gate is
fine and the other is sealed sends them to the wrong geometry.

**A coarse block could delete a metal-path part outright, and nothing
downstream could tell.** `COARSENING_PRIORITY` puts CORE and CHILL *above* GATE
and RUNNER — deliberately, and the docstring says why: a chill that vanishes
changes the thermal answer — and `coarsen_part_types` gives each coarse block
the single highest-priority code in it. So a gate one *fine* cell thick against
a core becomes CORE at coarsening 2. Measured on the fixture: **256 fine GATE
cells → 0 solve cells**, and because every thickness guard is keyed on parts
that are *present*, `measure_choke_thickness_by_part` simply has no GATE key and
nothing fires. The metal path is severed and the run reports a misrun as though
it were a result. The priority order is left alone — it is right for the reason
it gives, and reordering it globally would need `Full_Gating_Test` re-measured.
Instead `metal_path_severing` compares the metal path's **connectivity** across
the coarsening: label the SPRUE|RUNNER|GATE|FILTER mask on the fine grid and on
the solve grid with the same 6-connectivity, map each fine component onto the
coarse blocks its cells fall in, and flag a severing when a component lands on
two or more coarse components.

A presence test — "did this body keep *any* metal-path cell" — was the first
version and it only ever catches **total** erasure. A body severed in its
*middle* keeps cells at both ends, and that is the common geometry, not a corner
case: a wide thin gate plate with a boss or fillet along it, which is what
Test 2's gates have. Measured on exactly that shape at coarsening 2 — the gate
keeps **175** cells so nothing is erased, and every part measures **3.0 or
more** so nothing is at or below the hard minimum — the fine metal path is
**1 connected run and the solve grid's is 2**, and the only thing the user heard
was the generic 3-cell marginal note, the identical sentence a healthy channel
gets.

Total erasure is kept as a **second** check rather than folded in, because a
whole part can vanish while the survivors stay connected: a one-cell gate stub
hanging off a sprue column shares its fine component with the sprue, so the
component still lands on one coarse component and the connectivity test says
intact — while all 256 of the gate's cells became CORE. Both tests run.

What neither fires on is losing the *code* while keeping the *path*, which is
routine and harmless: a gate plate cutting a sprue column leaves a one-cell
sprue stub that becomes GATE, since GATE outranks SPRUE, and the channel is
still open, still connected and still measured. Keying on the body's own code
instead reports the stage's own choke rig as severed.

`choose_fill_coarsening` treats a severing as a sealed choke and refines toward
1, carrying **why** it refined (`forced_by_cause`) rather than only which part:
a severed part may have no cells at all, and quoting a thickness for it printed
"the gate measured 0.0 cells across" for a gate that was simply gone. If the
grid the stage is about to solve is still cut — which can only happen with
`choke_auto_resolution` off, since every grid the loop *chooses* has an intact
path and coarsening 1 is the mesh grid itself — the stage refuses with the
emit-and-return-None pattern it already uses for an unresolvable pour point,
with one remedy rather than two (the second arm was unreachable).

**The negative-population cascade guard acted on a single sample.** The trigger
is `>= max(8 cells, 2 % of wet cells)`, and the 8-cell floor was calibrated
against models with tens of thousands of wet cells where 8 is negligible. On a
small model it is not: at ~400 wet cells the 2 % fraction *is* 8 cells, so both
halves are satisfied by 8 — and the first divergence check lands at **step
200**, while a splat rim still routinely carries that many transient negatives.
Measured on the fixture: 8 negative cells out of 68 wet, one sample, and the run
was discarded. It now requires the condition at **two consecutive checks**, the
same shape as the FLUID-eps band, rather than a warm-up window — a warm-up
blinds the guard for its first N steps, and a genuine cascade at step 200 is
exactly as fatal as one at step 20,000, so it should still abort 200 steps
later rather than not at all. This keeps the recorded event: the round-2
`Test 2` cascade went 5 → **3,681** negative cells (93 % of wet) inside one
200-step interval and fired **27,958 steps** before the field went non-finite; a
field that far gone does not recover in 200 steps, so the abort moves from step
35,750 to 35,950 out of 27,958 steps of warning. And if a cascade ever did clear
after one sample, the prevalence window is still watching — that is its job.

Not done, and why: `data/surrogate_dataset/index.csv` still has no
solver-version column, so its 74 captured samples cannot be told apart by solver
generation. It is not fixable the way it was framed. `_append_index`
(`core/surrogate/dataset.py`) writes a header only when the file does not exist,
so adding a key to new rows would append seven values under the existing
six-column header and `csv.DictReader` would swallow the extra silently;
making it consistent means rewriting the existing index, which is modifying the
dataset. The 74 existing rows also cannot be labelled retroactively whatever we
do, and their `timestamp` column already separates them from anything captured
after round 4 landed. The backward-compatible home for a `solver_commit` is the
`.npz` record itself — `np.savez_compressed` in `capture_run`, read back with
`npz.get("solver_commit", "")`, which `load_dataset` already tolerates for
absent keys — and that is where it should go when someone wants it.

### Round 4: the mold pad and the filter exchange

**Round 4 fixed two measured defects, and the numbers in this section are the
first this stage produced on a lattice that is not a torus.** They were measured
*before* the τ⁻ clamp above, so the fill times and front shapes in it will move
again by roughly the deltas that section quotes; the mechanisms and the
before/after directions stand. The block between "Everything below this line
predates the mold pad" and "Running a fill" is older still and is kept for
history, not for reference; the sections from "Running a fill" onward are
current.

### The two defects

**Mechanism 1 — the filter exchange amplified its own bookkeeping error.** For
any exchange pair with a porous cell on either side, the Koerner mass exchange
overrode Table 1 and weighted the pair by the *unclamped* fill-fraction average
`0.5·(ε_A + ε_B)`. The weight scales the exchange amplitude and is itself a
function of the cell's own ε, so an ε error amplified its own exchanges — and
nothing could arrest it, because every transition and heal scan selects on
`FLAG_INTERFACE` only, so a FLUID cell's mass is written by the exchange and by
nothing else. Measured at Test 2's filter outlet: the effective weight, which
the scheme requires to be 1, reached **+10.7 and −13.0**, with the sign inverted
on 7.5 % of outflow samples; **786 of 2,009** FLUID samples of the filter's
bottom plane sat over full against **1 of 754** INTERFACE samples of the same
cells; ε climbed 1.046 → 3.711 → **7.76** while the donors it stole from fell to
**−9.5**, and the run surfaced as a NaN 8,800 steps later. Mass was conserved to
1e-16 throughout — this was never a leak, which is why every mass-ceiling check
passed while it happened. The branch did not even achieve its stated purpose: at
ε ≈ 1 the average *is* 1, so it only began to differ once ε had already drifted.
Fixed by deleting it: plain Table 1 everywhere. `create_state` also now seeds
mass at `ρ·porosity`, so a cell that *starts* FLUID inside a filter starts at
ε = 1 rather than 1/porosity.

**Mechanism 2 — the solve lattice was a torus.** The dense grid is sized exactly
to the geometry's bounding box, with no mold margin, while both step paths index
neighbours modulo the array (Numba `% n`, NumPy `np.roll`) and both document
that they rely on an obstacle shell to make that wrap unreachable. There was no
shell. On Test 2, **298** CAST cells on `z = 0` were the periodic neighbours of
RISER cells on `z = 93`, 93 empty planes away; at step 12,156 — the step the
front first occupied `z = 0` — `apply_transitions`' promote dilation created five
INTERFACE cells at `z = 93` with every wet donor a wrapped cast-floor cell, and
the phantom pool grew *downward* from the riser top for 3,594 steps until the
field went non-finite with 1,263 negative-population riser cells. Both
self-repair paths were blinded by the same wrap. Fixed in the stage, with no
solver change: one cell of `EMPTY_CELL` mold on every face, plus two pad planes
above metal that reaches the original `+z` face left open as atmosphere so an
open riser can still overflow. The stage now refuses to run and names the face
if any array face is left non-obstacle.

Full diagnosis, probes, validation runs and ledgers:
`Test/session_diag_2026-09-02/`.

### `Test_Model/Full_Gating_Test` @ 0.20 in, coarsening 2 — complete

**complete, 28,882 steps, 6.00 s of simulated pour** (was 18.0 s),
1,354/1,354 cells, **95 % by mass**, **20.8 lb** delivered into a 21.9 lb
cavity. `max|u|` **0.0557** against 0.273 before; `rho_max` **1.038** against
1.142. Zero negative populations at every sample, no precursor abort.

**The 18.0 s was not a slow fill, it was a throttled one.** The wrap fed the
sprue mouth back-pressure from spurious wet cells, and the rate controller did
what it is supposed to do with back-pressure: delivery decayed 88 → 36 → 21 →
**2 %** of target and the endgame crawled for 12 s. Post-fix delivery holds
**100 %** of the 4.39 lb/s target to 3.00 s. The metered pour is 19 lb ÷
4.39 lb/s = **4.3 s**, so 6.0 s is credible and 18.0 s was 4x non-physical.

**The riser now fills from the casting, not through a wormhole.** First wet
riser cell at step **15,000**, at `z = 8` — the riser's own lowest plane — with
**628** casting cells already wet below it. Before: step **1,500**, with **10**
casting cells wet, fed directly by a 32-cell SPRUE↔RISER wrap across `x`. The
riser's wet-cell count is now monotonic and its wet-z span contiguous at every
one of 57 ledger rows; before, it flickered 14 → 6 → 32.

**One thing to look at, not a regression.** `eps_min` over wet cells reaches
**−0.71** against −0.058 before. These are INTERFACE cells — `eps_fluid_min`
stays at **+0.0018** or above for the whole run, so the FLUID guard correctly
does not fire — and the run completes clean. The front now advances at the full
metered rate instead of a throttled one, so `_redistribute` drains donor
interface cells harder and the transient mass debt is deeper. It resolves. But
0.71 of a cell of debt is large, and the same excursion landing on a FLUID cell
would abort the run.

At **0.17 in** the same model also completes, 9.00 s, 50,949 steps, 96 % by
mass. `max|u|` **0.488 → 0.249**, negative populations 9 → 1, front flicker
10 → 1. It gains **two new warnings** — minimum fill temperature 1423 °C against
a 1460 °C liquidus, and 65 voxels with a cold-shut signature — and they are
honest: pre-fix the `x` wrap let the continuously fed sprue supply the casting
and riser directly, so the coldest wet cell was always near fresh 1620 °C metal.
The metal now has to travel sprue → runner → gate → casting → riser, and it
arrives cooler. The wrap was hiding a real cold-shut risk.

### `Test_Model/Filling_Test` @ 0.20 in, coarsening 2 — complete

**complete, 4,357/4,357 cells at 18.0 s, 97 % by mass, 70.0 lb.** This model
used to `time_cap` at 57 % of cells, and that figure was quoted in this file, in
`MARGINAL_CHOKE_THICKNESS_CELLS`' docstring and in the solver's own marginal-gate
warning text. **It was not the gate.** A **20-column** wrap across `z` put the
runner floor directly opposite the **sprue mouth** — the Dirichlet inlet, held
wet every step — so the pour boundary was an 18-direction neighbour of the
runner floor and was being fed back into itself. The fill completed the first
time it ran with a mold pad in place. Every citation of the 57 % has been
removed.

`max|u|` still peaks at **0.446** here, about nine times the 0.05 the timestep is
sized for and the highest of the four validated runs. That is the pre-existing
marginality of a 3-cell sprue/runner/gate, not new, and it is why the marginal
warning exists. **Complete is not the same as trustworthy at three cells.**

### What is no longer a baseline

Pre-fix `Full_Gating_Test` wrapped on **all three axes**: 60 metal-to-metal
pairs across `x` (including **32 SPRUE↔RISER**), 88 CAST↔CAST across `y` — the
casting had no `y` walls at all — and 16 RUNNER↔SPRUE across `z`. **No number
recorded against that model before this round is a baseline**, including the ones
kept below. Any threshold calibrated against it needs re-measuring.

**Entrapped-gas counts will rise** on models whose casting is flush with the
bottom or the sides of its own bounding box. The four sides and the floor of the
box are sand — the mold continues past the meshed geometry, which is why the pad
is mold and not air — so they are no longer counted as vents. Only the top is.
A pocket pinned against the casting floor now reads as entrapped, which is the
right answer: it is the textbook gas-porosity site, and it used to read as
vented.

### `Test 2` (459 lb) — both mechanisms gone, and it still diverged

**Resolved in round 5.** The step-20,200 divergence this section characterises
as an open item was the undamped ω⁻ ghost mode; see the τ⁻ clamp above. The
characterisation below is kept because it is what led to it.

4.3 mm auto-refined grid, solve grid **(137, 137, 98)** = (135, 135, 94) plus
the pad and two headspace planes, **3,672** cells of atmosphere opened above the
riser tops, all 339,824 metal cells preserved, all six faces obstacle, zero wrap
pairs on any axis.

Both target mechanisms are **verified eliminated at their own diagnosed
locations**. Filter-cell ε stays at **≤ 1.0094 with no trend** across ~14,000
steps of a wet filter, against 1.05 → **7.76** before. The riser holds **zero**
wet cells above the front at every sample, against 719 over-filled cells at
z 86-90 before. Delivery runs at 20.58 of 20.8 lb/s — 99 % of target,
unthrottled.

**It still does not pour.** The run survives to step **20,200** (first
non-finite mass at 20,192) against 16,493 before, then diverges by a **new
mechanism at a new location**: an exponential momentum/density runaway,
**e-folding every 64 steps**, in a 2×4×7 pocket at filter #1's **inlet** —
original `x[115,116] y[48,51] z[20,26]`, 14 FILTER and 11 RUNNER cells, all
FLUID — where `rho` reaches ±1e15 and `u` 1e15 lattice. Mechanism 1 was at the
filter *outlet*, grew at 2.7e-4/step and announced itself in ε 8,800 steps
ahead; this grows at 0.0155/step — about 60 times faster — and shows up in
momentum, not mass. `eps_filter_max` is pinned at 1.005 for 20,000 steps and only reaches 1.74
*after* `max|u|` has already grown 8x, so the filter is dragged along by the
instability rather than driving it.

**The run was marginally stable for 16,000 steps before it tipped.** At 20-step
sampling, `max|u|` spiked to **0.28–0.49 lattice** — six to ten times the design
velocity — on **123** separate rows starting at step **4,180**, well before the
filter was even wet, and damped every single time until the spike at ~20,020 did
not. A 3-cell gate and a jet from the sprue into a porous zone is a marginally
stable configuration, and this is what marginal looks like when it finally
fails.

**The new FLUID fill-fraction guard cannot see this one, and the reason is
structural.** It watches `ε = m/(ρ·porosity)`; here `ρ` blows up in the
denominator at the same time as `m` blows up in the numerator, so ε sat at 1.008
until step **20,160** — **32 steps** before the NaN, inside the guard's own
200-step cadence. Tightening the band does not fix it: 1.05 would have flagged
this run 2,772 steps early, but the same assembly reached 1.064 on a healthy
stretch and carried on for 2,700 more steps. **The missing precursor is a
velocity one, and none exists.** Open item for round 5, along with the
`overflow encountered in cast` at `_seed_promoted_cells` — `state.f` is float32,
so a population only needs to pass 3.4e38 to become `inf`.

Full test suite at the end of the round: **1,236 tests, 1 failure** — the known
pre-existing full-suite-ordering flake in
`SubmergedInterfaceHealingTests::test_a_deeper_tower_leaves_none_either`, which
passes in isolation and fails identically at `HEAD`.

### Everything below this line predates the mold pad

**The paragraphs from here to "Running a fill" were measured on the torus, and
several of their headline numbers are now known to be artefacts of it** — the
`Filling_Test` 57 %/41 % figures and `Full_Gating_Test`'s 15 s and 45 s fill
times in particular. They are kept because the mechanisms they document (the
heal dwell, the ping-pong, the Mach excursion) are real and were fixed on their
own merits, and because the history of what was believed when matters. Do not
quote a number from below without checking it against the round-4 section above.

**A configuration now produces a genuine `complete` fill, and it survives the
whole pipeline.** `Test_Model/Full_Gating_Test` at 0.20 in voxels with
`fill_grid_coarsening` 2 — the shipped default, giving a 0.40 in fill grid —
fills every one of its 9,251 metal voxels in 9.0 s of simulated pour,
43,324 steps, **0 misruns and no warnings at all**. `max|u|` is 0.025
against the 0.05 Mach guard and **no wet cell exceeds it at any point**.
(15.0 s before the heal dwell below; the ping-pong was taxing even the run
that finished.)
That is the first completed run this stage has produced; every "complete"
cited before the build report was a blow-up.

Driven end to end through `core.simulation_pipeline.run_pipeline` (WCB,
`thermal_coupled`, shrinkage + hotspot + Niyama + feeding path), the fill is
`complete` and `trustworthy`, seeds solidification from its own thermal field
rather than the two-constant placeholder, and solidification then runs its full
1,372 s horizon and returns solidification times for all 9,251 voxels. **No
NaN reaches any handed-over field.** The defect picture is the one the geometry
predicts: the casting is clean (shrinkage max 0.21, nothing above 0.5) and the
riser carries a pipe (max 0.79, 50 voxels above 0.5), which is feeding working.
The sprue and runner score highest of all (0.87) because nothing feeds a
gating system — expected, and not a casting defect.

**What made this possible was the flag-reinitialisation rule the previous
round identified as missing** — `physics.filling.heal_submerged_interface`. An
interface cell the front has closed over exchanges at weight 1 on every link
to a FLUID neighbour, so its mass change is its own density change and its
fill fraction freezes wherever the front left it; nothing could ever convert
it. Körner's scheme reinitialises such a cell to FLUID, and now so does this
one, with the mass deficit pushed through the same `_redistribute` the
transition path uses. Total mass is conserved exactly (a closed box holds to
one ulp over 2,000 steps); where a whole submerged pocket heals at once and no
interface cell is left beside it to draw on, the deficit is stranded back onto
the donor rather than invented.

The narrower rule — convert only when *every* neighbour is FLUID or obstacle,
which is literally the weight-1 condition — was tried and does not work:
measured across three fixtures, **not one** stranded cell met it. Every
stranded cell had another interface cell beside it. The topological rule (no
gas neighbour) is the one that matches the failure.

### What the rule costs, and where it still hurts

A healed cell is flagged FLUID at whatever fill fraction it had. That is
usually almost nothing — the cell-weighted mean fill fraction at the moment of
healing is **0.978** — but it accumulates: on the runs below, 1.9–4.2 % of the
metal volume ends up inside cells flagged full that hold less than a full
cell's metal. The mass accounting stays honest (18.0 lb delivered against
93.5 % of the metal volume wet: the two agree to within 1 %), but "100 % of
cells FLUID" means "the cavity is closed", not "the cavity is full".

**On a model with a thick, sloshing front the rule ping-ponged; a dwell now
holds it back.** A healed cell whose neighbour later drains is demoted back to
interface by the retreat path, refills, and is healed again — and each cycle
pulls its deficit out of the *advancing* front. Measured on `Filling_Test`
before the dwell: 828,368 heal events over 327,200 steps on a 4,346-cell
domain, roughly 190 conversions per cell, and 34 % less metal delivered
through the inlet than the same run without the rule.

The dwell (`HEAL_DWELL_STEPS` = 200) converts a cell only after 200
*consecutive* gasless steps. Persistence is the property that actually
separates a stranded cell from a busy one: a permanently closed-over cell is
gasless forever and merely waits 200 steps once; a sloshing cell loses its
streak the moment the surface re-opens beside it. Measured effect, same
`Filling_Test` run: heal events 828,368 → 45,250, delivery 36.6 → 41.1 lb,
cells filled at the 60 s cap 41 % → 57 % — level with the pre-heal run's 58 %,
except those cells are now real, convertible metal rather than frozen flags.
And `Full_Gating_Test` at the default coarsening completes *faster* with the
dwell than without it — 9.0 s of pour against 15.0 s — with its last heal at
step 42,282 of 43,324 and zero in the final stretch.

*Measured on `Test_Model/Full_Gating_Test` at 0.20 in, coarsening 1* (the fine
grid, 45 s cap). Still `time_capped` — 433,247 steps, 58 min wall — but at
8,753 of 9,251 cells (94.6 %, up from 91 %), 94 % by mass, 18.0 lb into a
19.25 lb model, and **zero interface cells left without a gas neighbour**,
against 796 last round. The residual is the riser, which is 66 % FLUID and
still rising when the clock runs out; everything upstream is 99.5–100 % full.
Arrival is physically ordered — mean `fill_time_s` sprue 0.35 s, runner 0.97 s,
gate 1.19 s, casting 5.23 s, riser 17.57 s.

*Measured on the same model at coarsening 1 through the full pipeline*
(2026-09-01, dwell active, `Test/`-side log `gate4_dwell.log`): the fill
reached **96 % of cells / 94 % by mass at 13.5 s and then receded** — 96 → 91
→ 90 → 87 → 86 % over five reports with the poured total falling 17.9 →
16.9 lb, i.e. metal genuinely flowing back out through the basin — and the
stall guard ended it at 86 % with the honest warning. **Un-diagnosed; do not
theorize without instrumenting.** The plausible-sounding story (a late slosh
wave reflecting off the full cavity, absorbed by the pressure inlet) is
exactly the kind of mechanism this project has refuted by measurement five
times. The defaults (coarsening 2) complete cleanly; only the fine grid shows
this.

**The transient Mach excursion documented last round is gone.** `max|u|` is
0.042 at the cap with the 99th percentile at 0.021, and **0 of 8,996 wet cells
are over the guard** — against a 0.457 peak and 87 of 866 samples over it
before. Give this grid a cap well past 45 s, or run it at the default
coarsening 2, which finishes.

*Measured on `Test_Model/Filling_Test` at 0.20 in, coarsening 2* — the harder
model, whose 0.5 in gate is 3 cells on this grid. **SUPERSEDED: this model now
completes, 4,357/4,357 cells at 18.0 s, 97 % by mass, 70.0 lb. The 41 %/52 %
below and the 57 % it is compared against were both measured with a 20-column
runner-floor↔sprue-mouth wrap feeding the pour boundary back into itself; see
the round-4 section. The gate was never the blocker.** `time_capped` at the 60 s
cap (327,200 steps, 37 min wall), 41 % of cells and 52 % by mass, 36.6 lb into
a 63.6 lb model. **All 1,191 frozen submerged interface cells are gone** — 0
remain — and `max|u|` is 0.029 with 0 of 2,506 wet cells over the guard.

**This model delivers less metal than it did last round (36.6 lb against
55.4 lb, 41 % of cells against 58 %), and the ping-pong above is the leading
suspect.** The choke is now plainly visible where it was not before: 2 of 154
gate cells are FLUID and only 186 of 714 gate voxels ever see metal. Whether
the front is being held back by the heal draining it, or the previous run's
higher numbers were frozen interface cells holding mass they could never
convert, is not yet separated. Do not read this model's fill percentage as a
result either way.

## Running a fill

1. Load the assembly and set part types. Gating bodies auto-detect from CAD
   names: `sprue`/`downsprue`, `runner`, `gate`/`ingate`, `pour cup`/`basin`,
   `filter`/`stelex`. Verify them in the parts manager — the fill only follows
   metal, so a mistyped body changes the answer.
2. **Set a pour location.** Click a point on the assembly, normally the top face
   of the down sprue. Until one exists, the Mold Filling analysis stays disabled;
   that is deliberate, not a bug.
3. Enable **Mold Filling (CFD)** and run. The header line lists it when it is
   active, so a log tells you whether the fill actually ran. The run ends with
   one of four statuses: `complete`, `time_capped`, `stalled`, or `diverged`.
   These used to be indistinguishable — all surfaced as a misrun count. Read
   the status before anything else the run reports.
4. Results → **Fill Time** animates metal arrival. Never-filled *metal* stays
   grey at every slider position — that is the misrun view. Only metal appears
   in this view at all; mold, cores and chills are excluded, because "when did
   it fill?" has no answer for them. The slider's "never fill" count is
   therefore the misrun count. Its Play button is now enabled — previously the
   only way to watch it was dragging the slider.
5. Results → **Fill Animation** is new: it replays ~60–120 captured frames of
   per-voxel fill fraction and velocity magnitude, so you can watch metal
   actually flowing rather than infer it from arrival time. It has no frames
   on a `diverged` run — frames are discarded on divergence, because playing
   back a blow-up would be the most convincing possible presentation of a
   failed solve.

## Settings that matter

**Voxel size is the dominant control, and not for the obvious reason.** Work
scales as roughly `1/dx^4`: a coarser cell means fewer cells *and* a larger
timestep. Halving resolution is about 16x less work.

Resolution also decides whether a fill can even attempt to work: the metal
has to get through the thinnest section of the gating, and a channel one cell
across is sealed by halfway bounce-back — the walls sit at ±0.5 cells and
leave no interior. It cuts both ways, though, and not in the direction you
would guess: see "Current state" above, where `Full_Gating_Test` completes at
coarsening 2 and is still filling at coarsening 1. Finer is not automatically
better here until the grid-convergence question is settled.
`fill_grid_coarsening` multiplies
the mesh size, so mesh 0.50 with coarsening 2 gives a 1.00 in fill grid —
check the product, not the mesh, when reporting any result.

Give `max_fill_time_s` real headroom regardless. A pour cut off at the cap
(`time_capped`) can look like a stall in raw voxel counts; the status field is
what tells them apart now.

**Pour rate** derives from the square-root rule by default: a casting of W lb
pours in √W seconds, so the rate is √W lb/s. On the reference assembly, 67 lb
→ 8.2 lb/s. Set `pour_rate_lb_s` to 0 to use the derived value; a nonzero
value overrides it. This replaced a fixed 5 lb/s that took no account of the
casting.

**Filter flow cap.** When a filter sits in the flow path, pour rate is capped
at its Foseco rated flow (0.100/0.125/0.140/0.175 kg/s per cm² by steel class
and pour mode), in **both** inlet modes, because a filter restricts the flow
path whatever drives it. Multiple filters sum (one per ingate, in parallel).
For a non-ferrous alloy it declines to rate rather than substituting a steel
figure.

A handful of exposed settings never reached a solver at all — `pour_time_s`,
`smagorinsky_cs`, `rated_velocity_warn_factor`, `head_fraction_warn`,
`priming_enabled`, `drag_enabled` — and are now hidden from the settings
dialog rather than left to imply control that doesn't exist (the fields are
kept so saved projects still round-trip). Note Smagorinsky viscosity is
active but at a hardcoded constant; the removed knob fed nothing.

## What the results are good for

Every claim below previously assumed a completed fill existed to validate
against. It didn't — the "complete" result this document used to cite was a
numerical blow-up (see the build report). One configuration now does complete
(`Full_Gating_Test`, coarsening 2), so these are no longer suspended outright;
they rest on a single model at a single resolution, which is one data point,
not a validation:

**Verified on one model at one resolution, unverified in general:**
- *Fill sequence* — the order regions fill, and where the front goes. On
  `Full_Gating_Test` this is now right at both resolutions: runner and gate
  first, then the sprue backing up, then the casting, then the riser last
  (mean arrival 1.16 / 1.20 / 1.37 / 3.48 / 8.78 s at coarsening 2), with every
  wet cell inside the Mach guard. The jet down an empty sprue is gone — the
  inlet is the sprue's whole mouth now. On `Filling_Test` the sequence is
  plausible as far as it gets, but it does not get far; see "Current state".
- *Misrun.* Meaningful only against a fill that completed, or stalled for an
  understood reason. `Full_Gating_Test` at coarsening 2 reports 0, which is
  correct for that geometry. A `time_capped` run's misrun count is still just
  a record of where the clock stopped it — the stage says so in a warning.
- *Relative comparisons at fixed resolution* — gating layout A against B, or
  pour point A against B. Not established: the two models tested so far
  fail by different mechanisms, so there is no known-good baseline yet to
  compare against.

**Independent of the fill problem, unchanged:**
- *Filter pressure drop.* The closure is Ergun-fitted on alumina-in-water data
  with Ward's inertial coefficient. No permeability is published for any
  STELEX product, so treat the drop as an estimate with a ±2x band. The one
  cheap measurement that would retire this is a cold-water head-flow rig on a
  real filter — see the note in `core/filter_database.py`.
- *Cold shut.* Requires fronts meeting **and** metal below liquidus. It cannot
  see a meeting that happened transiently and was then overwritten by
  continued flow. Moot until a run completes far enough to populate it
  meaningfully.

## Where the filter numbers come from

`filter_database.csv` reproduces all 104 published cells of the *Foseco Ferrous
Foundryman's Handbook* Tables 18.2/18.3, read from the publisher PDF and
cross-checked against page renders. Flow rate is an exact area law
(0.100/0.125/0.140/0.175 kg/s per cm² by steel class and pour mode). Capacity is
**not** — the 50x50x20, the range's only 20 mm filter, sits 6–8 % below the law
in the unsafe direction and carries a measured correction.

Foseco's own hedge applies: both tables read "*typical* capacity", and the text
notes capacity depends on pouring temperature, alloy, metal cleanliness and
metallostatic pressure.

## Known limits

- **Throughput** is ~19 M cell-updates/s against a 100–200 M/s target for a
  kernel of this shape. Remaining levers are the float32/float64 conversion glue
  (~27 % of a step) and dense-box iteration, which wastes ~6.8x at the 0.147
  occupancy typical of gating assemblies.
- **Grid convergence is unvalidated, and has to be re-measured from scratch.**
  The design doc asks for results to converge as `fill_grid_coarsening` → 1. The
  disagreement this bullet used to report — `Full_Gating_Test` completing at
  coarsening 2 in 15 s and still filling its riser at coarsening 1 after 45 s —
  was measured through a lattice that wrapped on all three axes, so it says
  nothing about convergence. Post-fix, both validated grids of that model
  complete (10.2 mm in 6.00 s, 8.6 mm in 9.00 s), but those are two voxel sizes
  at the same coarsening, not the coarsening sweep the question needs. Until
  someone runs it, quote the fill grid with every result.
- **Λ is not 3/16 on any real pour, and it costs a thin channel ~11 % of its
  flow.** `TAU_MINUS_MAX = 5` caps the antisymmetric relaxation time, which is
  what keeps the odd ghost modes damped at a casting τ, and it binds below
  τ = 0.541667 — every casting lattice. Solving steady Stokes-TRT Poiseuille
  with halfway bounce-back exactly gives the effective half-width as

      (H_eff/2)² = (H/2)² + (4/3)(Λ − 3/16)

  so the clamp removes a **flat 0.25 cell² of flow-carrying width** at any
  channel width, against exactly 0 before (a casting τ had Λ = 3/16 by
  construction). As a flow-rate deficit that is **−11.1 % through a 3-cell
  channel** — Test 2's and `Filling_Test`'s gate — **−25 % through a 2-cell
  one**, −2.0 % at 7 cells, −0.8 % at 11. Effective viscosity is untouched
  (fitted `nu_eff` matches (τ−½)/3 to 1.3e-10 either way): the entire cost is at
  the wall. Consistent with the whole-assembly measurement above — wet mass
  unchanged to 0.015 %, wet cells −2.06 %, i.e. less flow, not more.

  **The cost is intrinsic to damping the mode, not to the value 5.** Any
  practical ceiling gives Λ ≈ 0 at τ ≈ ½, so the (4/3)(Λ − 3/16) term is
  −0.25 cell² whatever the ceiling; 3/16 dominates it. There is no setting that
  both damps the ghost mode and keeps the wall. `MIN_CHOKE_THICKNESS_CELLS` and
  `MARGINAL_CHOKE_THICKNESS_CELLS` were calibrated at Λ = 3/16 and are left
  alone deliberately — they are resolution thresholds and the resolution has not
  changed — but their docstrings now carry these numbers, and the marginal
  warning's "treat the flow through it as marginal, not as a result" is the
  right reading of an 11 % boundary bias as well as of a coarse grid. Keep the
  Poiseuille wall benchmark at τ = 0.8, where the clamp is inert, so it goes on
  measuring 3/16.
- **No detector exists for a localised undamped mode, and one cannot be built
  from a global scalar.** The mass-side guard
  (`FLUID_FILL_FRACTION_MIN`/`MAX`) gives thousands of steps of warning on a
  mass-side amplifier and 32 steps on a momentum-side one. A precursor on the
  global max|u| is **structurally blind** here, measured, not assumed: the only
  fit setting that catches the real event also fires 22 false positives before
  it, because the sprue's free-fall jet dominates max|u| until 240 steps before
  the NaN. So mechanism 3 is fixed, but **any other localised undamped mode
  still has no warning at all** — and the diagnosis names a live candidate, the
  Smagorinsky-suppressed filter footprint, which is unconditionally unstable at
  the 0.05 design velocity in the jet fixture with no filter present. The τ⁻
  clamp closes that particular hazard; the detection gap it exposed is
  unchanged. The design that works is in "The next precursor to build" above and
  is uncalibrated on a full domain.
- **The inlet** imposes a *pressure* in head mode: the metallostatic head
  rho = rho_0 (1 + g h / c_s^2), with the velocity measured back out of the
  inlet cells' own populations each step and clamped at the Mach guard. It is
  no longer Torricelli √(2gh) held constant — a fixed-velocity pin cannot feel
  back-pressure, and on `Filling_Test` its delivery fell to 6 % of the
  prescribed rate and then went negative while the sprue sat in suction.
  Velocity mode still prescribes pour rate ÷ inlet area, because there the
  user is asserting a delivery rate.
- **The inlet is the poured feature's whole mouth**, not the 1.5-voxel radius
  the pour point resolves to. The click chooses *which* feature is poured into;
  `_expand_inlet_to_mouth` then floods that feature's own top layer (2-D,
  8-connected, confined to the part types the seeds stand on) and pins the
  union. On `Full_Gating_Test` that is 48 cells instead of 8. The 8-cell inlet
  was the measured blocker: metal entered as a jet, free-fell down a mostly
  empty sprue and atomised into low-fill interface cells, 90 % of which
  exceeded the Mach guard, and the run diverged at 24 % by mass. In velocity
  mode the prescribed rate is now divided by the larger area, so the imposed
  velocity drops for the same kg/s — which is what a basin sitting on the whole
  sprue does.
- **No fluidity-spiral validation.** F2's thermal field is checked against
  physical invariants — bounded by pour and mold temperature, metal further
  from the inlet is cooler — not against measured spiral lengths, and no run
  has completed far enough to have a spiral to compare in the first place.

## If something goes wrong

- *Run status is `diverged`* — the run hit the mass ceiling and its results
  were discarded. The GUI shows a modal explaining this and that
  solidification fell back to the uniform pour-temperature placeholder. Do
  not read fill, misrun, or defect numbers off this run — none are valid.
- *Run status is `stalled`* — the front stopped advancing before the time
  cap, in **both** the filled-cell count and the mass fraction. Check the
  choke resolution first (below); given the open problems in "Current state,"
  treat the reported fill percentage as where the run stopped, not a result to
  compare against. A run that is taking on mass without flipping cells to
  FLUID is *not* stalled and will no longer be called one — the progress line
  reports both numbers so the two can be told apart.
- *Run status is `time_capped`* — the pour was still advancing when
  `max_fill_time_s` ran out. Give it more headroom and rerun.
- *Fill stalls at a low percentage* — check the choke resolution first.
  Report it with the fill grid size, not the mesh size. The choke warning
  measures thickness with a distance transform rather than cross-sectional
  area, so it can flag a wide, thin gate — the geometry that actually seals —
  which the first version of this check missed. It is now **per part and per
  cell**: the median local thickness over each of the sprue, runner, filter
  and gate, on the grid that was solved rather than the mesh it came from.
  Taking one maximum per part, as the second version did, let a single fat
  cell at the gate's root speak for the whole plate: on the `Test 2` assembly
  at 8.6 mm the gates are 11 × 13 × 2 — sealed — and the check reported 3.00
  cells and said nothing. Below 2 cells is a hard warning naming the part;
  exactly 3 is a softer note, because 3 cells is the thinnest channel this
  solver has carried a complete fill through — and it carries it with `max|u|`
  at nine times the design velocity, so read "marginal" as written.
- *"no pour location has been selected"* — the gate working; set one.
- *Filling absent from the run header* — the analysis was not enabled.
- *Every gas pocket reads as entrapped* — the vent mask is empty. The stage
  warns about this; a cavity with no vent cannot fill physically either. Note
  that only the **top** of the meshed bounding box counts as open air: the sides
  and the floor of the box are sand, so a pocket pinned against the casting
  floor now reports as entrapped rather than vented. That is a change, and it is
  the correct direction — a pocket on the floor is the textbook gas-porosity
  site — so expect entrapped counts to go up on models whose casting is flush
  with the bottom or the sides of its own bounding box.
