# Mold filling: field-test notes

What the filling stage does, how to drive it, and — more importantly — what its
answers are and are not good for. Written for a field-test campaign, not as a
design document; `docs/filling_implementation.md` is the plan and
`docs/ARCHITECTURE.md` describes where the code sits.

## Current state — read this first

**Round 4 fixed two measured defects, and the numbers in this section are the
first this stage has produced on a lattice that is not a torus.** The block
between "Everything below this line predates the mold pad" and "Running a fill"
was measured before that and is kept for history, not for reference; the
sections from "Running a fill" onward are current.

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

### `Test 2` (459 lb) — both mechanisms gone, and it still diverges

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
