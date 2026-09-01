# Mold filling: field-test notes

What the filling stage does, how to drive it, and — more importantly — what its
answers are and are not good for. Written for a field-test campaign, not as a
design document; `docs/filling_implementation.md` is the plan and
`docs/ARCHITECTURE.md` describes where the code sits.

## Current state — read this first

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
model, whose 0.5 in gate is 3 cells on this grid. `time_capped` at the 60 s
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
- **Grid convergence is unvalidated, and the two grids now disagree.** The
  design doc asks for results to converge as `fill_grid_coarsening` → 1.
  `Full_Gating_Test` completes at coarsening 2 in 15 s of pour and is still
  filling its riser at coarsening 1 after 45 s, so the finer grid is not merely
  slower to compute — its front advances differently in *simulated* time. That
  is the next thing to measure, and until it is, quote the fill grid with every
  result.
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
  Report it with the fill grid size, not the mesh size. The choke warning now
  measures thickness with a distance transform rather than cross-sectional
  area, so it correctly flags a wide, thin gate — the geometry that actually
  seals — which the first version of this check missed.
- *"no pour location has been selected"* — the gate working; set one.
- *Filling absent from the run header* — the analysis was not enabled.
- *Every gas pocket reads as entrapped* — the vent mask is empty. The stage
  warns about this; a cavity with no vent cannot fill physically either.
