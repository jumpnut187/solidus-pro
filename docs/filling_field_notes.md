# Mold filling: field-test notes

What the filling stage does, how to drive it, and — more importantly — what its
answers are and are not good for. Written for a field-test campaign, not as a
design document; `docs/filling_implementation.md` is the plan and
`docs/ARCHITECTURE.md` describes where the code sits.

## Current state — read this first

**The filling stage is not validated. No configuration produces a verified
complete fill.** On `Test_Model/Filling_Test`, the best case tested (0.20 in
voxel, coarsening 2 — a 0.40 in fill grid) reaches ~25 % filled and plateaus
before aborting on a mass-divergence ceiling; a 0.60 in fill grid stalls
honestly at 17 %, with the choke warning firing. The open problem is that
lattice velocity grows to roughly 9× the Mach guard the timestep is sized
for — measured `max|u|` reaches 0.44 by step 97,000 against a
`lattice_velocity_max` of 0.05 — which drags density to 1.34 and inflates wet
mass past the divergence ceiling. On the 0.40 in case it is already climbing
well before the endpoint: the mass ratio was at 0.81 by step 59,000, long
before the 25 % plateau. Whether the same runaway affects the 0.60 in stall
has not been measured — that run stays finite and its mass stays bounded, but
its velocities were not instrumented.

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
state" above: the two resolutions tested fail by different mechanisms (a 17 %
stall at the coarser grid, a 25 % plateau followed by divergence at the finer
one), and neither has been made to complete. `fill_grid_coarsening` multiplies
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
- *Fill sequence* — the order regions fill, and where the front goes. On the
  0.40 in case the velocity runaway is underway long before the run ends, so
  the front there is not physical. The 0.60 in stall was not instrumented for
  velocity, so treat its sequence as unverified rather than as known-bad:
  nobody has checked it either way.
- *Misrun.* Meaningful only against a fill that completed, or stalled for an
  understood reason. Right now every stall or divergence *is* the open
  problem, not a measurement you can build on.
- *Relative comparisons at fixed resolution* — gating layout A against B, or
  pour point A against B. Not established: the two resolutions tested so far
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
- **Grid convergence is unvalidated**, and now more than aspirational: see
  "Current state" — the two resolutions tested fail by different mechanisms,
  not merely by different fill percentages. The design doc asks for results
  to converge as `fill_grid_coarsening` → 1; that has not been demonstrated
  and cannot be, until a run completes.
- **The inlet** prescribes a constant entry velocity — Torricelli √(2gh) from
  the head in head mode, or pour rate ÷ inlet area in velocity mode, both
  capped at the Mach guard. It does not ramp and does not model a falling
  stream; a Zou–He inlet spanning the sprue cross-section remains the intended
  replacement.
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
  cap. Check the choke resolution first (below); given the open
  velocity-runaway problem in "Current state," treat the reported fill
  percentage as where the run stopped, not a result to compare against.
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
