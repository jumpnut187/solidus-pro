# Mold filling: field-test notes

What the filling stage does, how to drive it, and — more importantly — what its
answers are and are not good for. Written for a field-test campaign, not as a
design document; `docs/filling_implementation.md` is the plan and
`docs/ARCHITECTURE.md` describes where the code sits.

## Current state — read this first

**The filling stage is not validated: no configuration yet produces a
*complete* fill.** What has changed is that neither test model diverges any
more. The two mechanisms that used to end these runs in a blow-up are gone,
and what stops a fill now is a third, identified thing.

*Measured on `Test_Model/Full_Gating_Test` at 0.20 in voxels, coarsening 1* —
a bottom-gated rig built so a head-driven fill can finish. **No divergence.**
The run uses the whole 45 s cap (433,247 steps, 54 min wall) and ends
`time_capped`, still advancing, at 8,459 of 9,251 metal cells FLUID (91 %),
97 % by mass, 18.5 lb delivered into a 19.3 lb model. `max|u|` holds at 0.033
against the 0.05 Mach guard, with the 99th percentile over wet cells at 0.015,
and per-cell FLUID mass stays inside [0.978, 1.007] for the entire run. The
previous inlet diverged this model at step 30,600 with 24 % by mass.

Arrival is physically ordered — mean `fill_time_s` by part: sprue 0.52 s, gate
0.87 s, runner 1.55 s, casting 2.60 s, riser 4.52 s; by *first* arrival, sprue
0.00 s, runner 0.23 s, casting 0.45 s, gate 0.50 s, riser 0.62 s. (The runner's
*mean* trails the gate's because a handful of runner cells at the far end
arrive as late as 31 s; its front gets there first.) The pour itself is quick
and about the right speed: 11.5 lb of the 18.5 is delivered by 4.5 s, against
the ~4.4 s the √W rule gives for this weight. The remaining 40 s is the solver
topping up the last interface cells, not metal still flowing.

*Measured on `Test_Model/Filling_Test` at 0.20 in voxels, coarsening 2* — the
harder model, whose 0.5 in gate is 3 cells on this grid. **No divergence**
(the previous run diverged at step 251,000). It runs the full 60 s cap
(327,200 steps, 42 min wall) and ends `time_capped`, still advancing, at 58 %
of cells and 78 % by mass, 55.4 lb into a 63.6 lb model. Nothing anywhere in
the domain exceeds the Mach guard — 0 of 3,845 wet cells over 0.05, with
`max|u|` 0.0155 and the 99th percentile 0.0098 — and per-cell FLUID mass stays
inside [0.9970, 1.0033] against the ±70,000 the old weighting reached. The
sample-to-sample wet-mass alternation is gone: the last twenty samples rise
monotonically, 3,355.4 → 3,374.0.

The gating system is *not* the bottleneck at this resolution, which is worth
saying because the 3-cell gate was the obvious suspect. By mass the sprue is
98.3 % full, the runner 94.9 % and the gate 96.3 %; the shortfall is the
casting (83.2 % by mass, 57.0 % of its cells FLUID) and the riser (54.9 % /
29.0 %). Of the 1,322 interface cells left at the cap, **1,191 have no gas
neighbour** — see below. Chills correctly take no metal at all.

**What stops both runs short is a gap in the flag bookkeeping, not the flow.**
Under Körner's mass-exchange weighting an interface cell whose neighbours are
all wet or solid exchanges at weight 1 in every direction, so its mass change
is exactly its own density change — and a *submerged* interface cell therefore
freezes wherever the front left it, at whatever ε it had. Measured on a
synthetic bounded cavity (`_walled_tower(nz=4, width=4)`): the run ends with 44
FLUID cells and 20 INTERFACE cells, and **all twenty have no gas neighbour
left**, at ε between 0.05 and 0.84. The same signature on the real model —
1,191 of `Filling_Test`'s 1,322 remaining interface cells, ε 0.000 to 1.002.

Körner's scheme reinitialises such a cell to FLUID; this implementation has no
such rule. It was invisible while the mass exchange let FLUID mass drift upward
past the transition threshold and convert those cells anyway — that drift *was*
the ±70,000 runaway, so removing it exposed this, it did not cause it. Adding
the reinitialisation rule is the next piece of work: it needs a mass account
for the deficit it fills, which is why it was not bolted on to the weighting
fix.

Do not run this stage in a field test expecting a completed fill. If you run
it anyway, read the status field before reading anything else — see "If
something goes wrong" — and do not treat a `stalled` run's numbers as more
than a record of where and how it stopped.

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
leave no interior. But resolution is not sufficient on its own — see "Current
state" above: the two models tested fail by different mechanisms (mass drift in
a full channel on `Filling_Test`, an under-covered inlet on
`Full_Gating_Test`), and neither has been made to complete.
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
numerical blow-up (see the build report). Until a configuration produces a
genuine `complete` run, treat the following as suspended, not merely
cautious:

**Currently unverified, pending a completed run:**
- *Fill sequence* — the order regions fill, and where the front goes. On
  `Filling_Test` at coarsening 2 the sequence is now plausible for most of the
  run — sprue full, then runner, then casting, with velocity inside the Mach
  guard — but the mass drift described in "Current state" is accumulating
  underneath it the whole time, so it is not yet evidence of anything. On
  `Full_Gating_Test` the front is a jet spraying down an empty sprue and is
  not physical at all.
- *Misrun.* Meaningful only against a fill that completed, or stalled for an
  understood reason. Right now every stall or divergence *is* the open
  problem, not a measurement you can build on.
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
- **Grid convergence is unvalidated.** The design doc asks for results to
  converge as `fill_grid_coarsening` → 1; that has not been demonstrated and
  cannot be until a run completes. Both models now end the same way — capped
  while still advancing, with the residual dominated by frozen submerged
  interface cells — so the two configurations at least fail by the *same*
  mechanism now, which they did not before.
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
