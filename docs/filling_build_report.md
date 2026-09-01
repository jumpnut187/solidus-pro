# Mold filling: build report

What was built for the mold-filling stage, why each choice was made, and what
was actually verified rather than merely written. Written to be read once, by
someone deciding whether to trust this code in a field test.

Three companion documents divide the same subject: `filling_implementation.md`
is the original plan, `filling_field_notes.md` is the operator's guide, and
`ARCHITECTURE.md` says where the stage sits in the pipeline. This file is the
record of the work itself.

Delivered across 12 commits, `f17147d..fb274ac`, on `refactor/pouring-seams`.

**Status (2026-09-01): the stage now produces a genuine, verified complete
fill.** `Test_Model/Full_Gating_Test` (a purpose-built validation rig: bottom
gated, every channel at least 5 cells across on its fill grid) fills all
9,251 metal voxels in 9.0 s of simulated pour at the shipped defaults, with
zero misruns, zero warnings, mass conserved, and every wet cell under the Mach
guard — and survives the whole pipeline into solidification and defect
analysis. What made that possible, and what remains open (under-resolved
models such as `Filling_Test`'s 3-cell gate still time-cap honestly rather
than finish), is in `filling_field_notes.md` under "Current state".

A caution this file keeps on purpose: an earlier revision certified a 100 %
fill that was a numerical blow-up dressed as a perfect pour — the completion
check read flags alone. Every number below predates the 2026-08-31/09-01
rework (pressure inlet, mouth-wide inlet resolution, Körner Table 1 mass
exchange, submerged-interface healing with a dwell); read "What was actually
verified" before trusting any fill-percentage or timing figure elsewhere in
this file.

## What exists now

Five new modules, ~4,700 lines:

| Module | Lines | Role |
|---|---|---|
| `physics/fluid_flow.py` | 1,220 | D3Q19 lattice, TRT collision, Guo forcing, Smagorinsky, porous drag, Numba kernels |
| `physics/filling.py` | 1,293 | Körner free-surface solver, mass tracking, thermal advection, defect detection |
| `core/filling_stage.py` | 1,340 | The mesh↔lattice seam: grid mapping, filter geometry, orchestration, handover |
| `core/filter_database.py` | 540 | Foseco filter lookup and flow-rate/capacity laws |
| `core/pour_location.py` | 287 | The picked pour point, and its resolution to inlet voxels |

Plus `filter_database.csv` (104 rows), twelve new test files (~4,900 lines), and
edits to `core/part_manager.py`, `core/voxel_mesh.py`, `core/solidification.py`,
`core/simulation_pipeline.py`, `ui/main_window.py`, `ui/visualizer.py`,
`web/payloads.py` and the React front end.

The layering rule holds: `physics/` imports neither `core/` nor `config/`, and
that is enforced by the AST walk in `tests/test_architecture_boundaries.py`,
which also catches function-local imports.

## The physics, and why

**D3Q19 lattice Boltzmann** for the flow. Nineteen velocities, rest at index 0
and opposite pairs adjacent so `OPPOSITE[i]` is a branch-free index flip.

**TRT collision, not BGK.** This is not a refinement — it is the difference
between working and not. Molten steel at practical resolutions gives a
relaxation time of τ = 0.50000x, and BGK's error term blows up as τ → ½. TRT
splits the populations into symmetric and antisymmetric parts and ties them
through the magic parameter Λ = 3/16 (`TRT_MAGIC_PARAMETER`), which cancels the
offending term and puts the bounce-back wall exactly halfway between nodes.

**Guo forcing with the TRT prefactor.** The body force uses `1 - ω⁻/2`, not
BGK's `1 - 1/(2τ)`. This was got wrong first, and the symptom was quiet rather
than loud: gravity applied at roughly 2×10⁻⁴ of its intended strength, so metal
crept instead of pouring and nothing raised an error.

**Körner free-surface tracking.** Cells are OBSTACLE, GAS, INTERFACE or FLUID;
the interface layer carries a fluid fraction ε = m/(ρ·porosity) and the mass
exchange between neighbours is what actually moves the front. Transitions use
hysteresis so a cell hovering at ε ≈ 1 does not oscillate between states.

**Smagorinsky subgrid viscosity.** Required on the physics: pouring Reynolds
numbers here run 6,000–720,000, far past anything the resolution resolves. It
also turned out to be the stabiliser that made driven fills survive at all —
wiring it in took the real assembly from 0.5 % filled and diverging to 27.7 %
and climbing. Suppressed inside filter cells, where the porous closure already
supplies the momentum sink. The exposed `smagorinsky_cs` knob, however, feeds
nothing — see "Parameters wired up, and unwired" below.

**Guo–Zhao Brinkman–Forchheimer drag** for the ceramic filters, solved in
closed form rather than iterated.

## Filters, and how far to trust them

`filter_database.csv` reproduces all 104 published cells of the *Foseco Ferrous
Foundryman's Handbook* tables 18.2/18.3 — 13 sizes across both deoxidation
practices — read from the publisher PDF and cross-checked against page renders.

Two honest caveats, both load-bearing:

- **Flow rate is an exact area law** (0.100/0.125/0.140/0.175 kg/s per cm² by
  steel class and pour mode). **Capacity is not.** The 50×50×20 — the range's
  only 20 mm filter — sits 6–8 % below the law, and in the *unsafe* direction.
  It carries a measured correction, `THIN_FILTER_CAPACITY_FACTOR = 0.934`.
- **Permeability is inferred, not published.** No permeability figure exists for
  any STELEX product. The closure is Ergun fitted at `ERGUN_PREFACTOR = 2.2`
  against a *nominal* `NOMINAL_FOAM_POROSITY = 0.85` rather than a measured
  per-ppi value, with Ward's inertial coefficient `FORCHHEIMER_WARD_CF = 0.55`.
  Treat the pressure drop as an estimate with a ±2× band.

The first build of this CSV was wrong — 24 rows, 6 sizes, 2 capacity columns —
and was rebuilt programmatically from `pdftotext` after two independent
verifications reached the checksum-verified source. That is worth knowing
because it is the kind of error that reads as plausible indefinitely.

## Parts, pour location, and the gate

`PartType` gained five members: `POUR_CUP=5`, `SPRUE=6`, `RUNNER=7`,
`FILTER=8`, `GATE=9`, with filters classified as mold (a flow restrictor) and
the rest as metal. Gating bodies auto-detect from CAD names.

The pour location is a frozen `PourLocation` — point, normal, part id, schema
version — carried in inches. It is picked by clicking the assembly in **both**
front ends: PyQt/VTK on the desktop and React/Babylon on the web.

**The stage only runs if a pour location exists.** The Mold Filling analysis
checkbox stays disabled until one is set, and clearing the location forces it
back off. This was verified in both directions rather than assumed.

## Results

Fill Time animates metal arrival, on the conventional flow-front palette: dark
blue arrives first, red last. Never-filled *metal* stays grey at every slider
position — that is the misrun view, and it is intended to be the strongest
output the stage produces, because it is topological rather than
quantitative — but see "Current state": no run has completed, so this has not
yet been exercised on a result worth trusting.

Fill Animation is new: it replays ~60–120 captured frames of per-voxel fill
fraction and velocity magnitude, so metal can be watched actually flowing
rather than inferred from arrival time alone. See "Animation" below.

F2 hands the fill's real temperature field to solidification instead of the
old two-constant placeholder, so metal that travelled furthest starts cooler.
F3 reports entrapped gas and cold shuts. On a `diverged` run neither handover
happens: solidification falls back to the uniform pour-temperature placeholder
instead (see "New guards").

## What was actually verified

Not just unit tests. The whole thing was driven end-to-end through the real
`MainWindow` — real `PartManager`, real analysis-options dialog, real
`SimulationWorker` thread, real VTK pipeline — on `Test_Model/Filling_Test`:

- All 8 STEP files loaded; types auto-detected.
- Pour location on the Down Sprue top face, `(-9.42, -5.25, 9.44) in`.
- The animation swept through the real signal chain at 11 slider positions,
  with voxel counts read off actual VTK polydata.

**The headline result this section used to report was invalid, and the way it
was invalid is the most instructive finding in this whole episode.** The run
at 0.25 in voxel reported "16,494 / 16,494 metal voxels filled — 100 %, 0
misrun, 0 entrapped gas, 0 cold shuts, in 4.50 s of a 30 s cap." That exact
result was reproduced today — and the per-step trace shows it is the
signature of a blow-up, not a fill:

- Mass conservation broke at step 27,000, at roughly 30 % filled.
- Total wet mass then grew exponentially, reaching ~1e43 by step 35,500.
- At step 35,500 the explosion flagged **every** metal cell FLUID at once and
  the interface-cell count dropped to **exactly 0** — which cannot happen in
  a physical fill, where an interface layer persists until the very end.
- The stage's completion check tested cell **flags** only, never whether
  density or mass were physical, so it read that as a perfect pour and
  exited.

Fixing the two bugs that caused this (below) did not produce a completed fill
at any tested resolution — see "Current state."

Suite-wide: **953 passed, 11 skipped**. The filling-specific suites account for
300 tests and 761 subtests. That count and the three bullets above are what
remain verified; the fill-percentage and timing numbers this section used to
carry are not, and have been removed.

## Current state

**The stage does not produce a completed fill at any tested resolution. There
is no validated configuration.** Specifically, on `Test_Model/Filling_Test`:

| configuration | outcome |
|---|---|
| 0.20 in voxel, coarsening 2 (a 0.40 in fill grid) | reaches ~25 % filled and plateaus. Survives to step 152,200 (27.9 s of pour) with **no NaN at all**, then aborts on the mass ceiling |
| 0.60 in fill grid | runs stably and honestly stalls at 17 %, with the choke warning firing |

The open problem: **lattice velocity grows to about 9× the Mach guard.**
Measured `max|u|` climbs 0.026 → 0.12 by step 40,000 → **0.44 by step
97,000**, against a `lattice_velocity_max` of 0.05. That is roughly 24 m/s
physical in a pour whose free-fall bound is 2.75 m/s. The resulting
compressibility error drags density to 1.34, and since mass ≈ ρ·porosity·ε
that is what inflates wet mass into the divergence ceiling.

Two things have been measured and refuted as the cause:

- **The zero-floor on the Körner reconstruction (bug 5 below) is not the
  cause.** It fired 0 times in the first 59,000 steps while the mass ratio
  had already climbed from 0.55 to 0.81 — the runaway was already underway
  before the floor could have mattered.
- **Neither the gate nor the inlet explains the 25 % plateau.** Coarsening
  *inflates* thin features (the 0.5 in gate becomes 3 cells, 1.2 in, on a
  0.40 in grid), and the 4-cell inlet still delivers ~9.3 lb/s against the
  8.2 lb/s √W target.

The velocity runaway is unresolved, and the next question is deliberately a
measurement rather than a theory: **is the high velocity a bulk phenomenon or
a handful of pathological cells?** Those need opposite fixes -- a velocity
limiter in the equilibrium, versus a repair to the free-surface closure -- and
the reported figure is a maximum over the whole domain, which a single bad
interface cell would produce just as readily as a genuinely fast bulk flow.
Split the |u| distribution by cell flag before choosing.

Three hypotheses about this stage have already been refuted by measurement
after looking convincing on paper. Measure first.

## Five bugs that only running it for real could find

Worth recording, because each passed every test that existed at the time.

1. **A NaN leak into the handover** (`349c298`). `fill_time_s` is written once
   and never cleared, but temperature is NaN-reset when a cell drains. A drained
   cell therefore had a valid fill time and a NaN temperature, and solidification
   rejected the field.

2. **Mold reported as misrun** (`fb274ac`). The Fill Time view counted and drew
   every non-metal voxel as metal the pour never reached, because
   `gather_to_voxels` returns the same `-1` for "metal that never filled" and
   "not metal at all". The run that exposed it reported **1,485 never fill** —
   one per chill voxel — against a true count of 0. (That run was the invalid
   100 % result described above; the miscount was real and independent of it,
   since every chill voxel was drawn as misrun regardless of what the metal
   did.) Non-metal is now `NaN`, which is
   false against `< 0`, `>= 0` and `<= t` alike and so leaves every mask without
   special-casing. Two things had hidden it: no test had ever mixed metal and
   non-metal in one array, and the end-to-end harness computed its ground truth
   by mirroring the renderer's own masking rule, so it could only detect a
   renderer/rule mismatch and never an error in the rule itself.

3. **A mesh segfault** (`4e3f455`). `AdaptiveVoxelMesh.generate()` died with
   SIGSEGV at five of eight voxel sizes — no exception, no traceback, and not
   monotonic in size, which made it look like data corruption. It was a C-stack
   overflow in SciPy's recursive kd-tree build against the 8 MB macOS main-thread
   stack. `balanced_tree=False` bounds recursion by the bounding box instead.
   The diagnostic that settled it, worth reusing: the identical case run on a
   256 MB thread stack passed unchanged.

4. **The inlet was a momentum pump.** It rebuilt its populations each step from
   `state.u`, which is the Guo force-corrected velocity and already carries
   half of that step's gravitational impulse. Re-imposing it added
   `0.5 * gravity_lattice` of velocity every step, forever. Velocity grew
   linearly, crossed the Mach guard the timestep was sized for, and detonated
   the solve. Critically: **the fill was being driven by that artifact** —
   removing it without replacement dropped the fill from ~25 % to 1 %. It now
   prescribes a constant entry velocity: Torricelli √(2gh) from the head in
   head mode, or pour rate ÷ inlet area in velocity mode, both capped at the
   Mach guard.

5. **The Körner gas-side reconstruction produced negative populations.**
   `f_ibar = f_i^eq(ρ_G,u) + f_ibar^eq(ρ_G,u) − f_i*` is unbounded below;
   where a cell has most of its neighbours in gas it goes negative, which is
   a negative density contribution. Measured cascade: isolated, self-healing
   negatives from step 4,638; a cascade from 1 cell at step 12,606 to 417 by
   12,651; density through zero at 12,644; the field non-finite at 12,650.
   Now floored at zero in both the Numba kernel and the NumPy fallback. Two
   smaller repairs rode along: promoted cells with no donor were left at
   ρ=0 (latent — measured 0 occurrences on this geometry), and retreating
   cells kept a stale density.

Note: neither of the two new bugs above is the source of the velocity
runaway described in "Current state" — that problem surfaced after both were
fixed and remains open.

## New guards

The stage used to be unable to tell a real fill from a blow-up. It now can,
which is how the invalid 100 % result was caught in the first place.

- **Divergence detection.** Wet mass is checked against a physical ceiling
  every 200 steps; on breach the run **aborts and its results are
  discarded**, and solidification falls back to the uniform pour-temperature
  placeholder rather than being seeded from debris.
- **Honest status.** A run now reports `complete` / `time_capped` /
  `stalled` / `diverged`. These used to be indistinguishable — all surfaced
  as a misrun count.
- **Stall detection.** A front that stops advancing ends the run instead of
  grinding to the time cap.
- **Choke warning.** Warns when the thinnest gating section is under 2 cells
  thick. Note the correction — the first version measured cross-sectional
  *area*, which passes a wide, thin gate (the geometry that actually seals).
  It now measures thickness via a distance transform.
- The GUI now shows a modal warning on a diverged run explaining that fill
  results are discarded **and** that solidification used a placeholder.

## Parameters wired up, and unwired

Six were exposed in the UI, serialized, and never read by any solver:
`pour_time_s`, `smagorinsky_cs`, `rated_velocity_warn_factor`,
`head_fraction_warn`, `priming_enabled`, `drag_enabled`. They are now hidden
from the settings dialog (fields kept so saved projects round-trip). Note
specifically that **Smagorinsky is active but at a hardcoded constant** — the
knob fed nothing.

Three were dead and are now wired up: `pour_rate_lb_s`, `inlet_mode_code`,
`pour_mode_code`.

**Pour rate now derives from the square-root rule**: a casting of W lb pours
in √W seconds, so the rate is √W lb/s (pour time in seconds and rate in lb/s
are the same number). 0 means derive. On the reference assembly: 67 lb →
8.2 lb/s. It replaces a fixed 5 lb/s that took no account of the casting.

**Filter flow cap**: when a filter is in the flow path the pour rate is
capped at its Foseco rated flow (0.100/0.125/0.140/0.175 kg/s per cm² by
steel class and pour mode). It applies in **both** inlet modes, because a
filter restricts the flow path whatever drives it. Multiple filters sum (one
per ingate, in parallel). For a non-ferrous alloy it declines to rate rather
than substituting a steel figure.

## CAD tessellation is now tied to voxel size

STEP bodies were tessellated at cascadio's 0.01 mm default, turning a
cylindrical riser into 18,004 triangles. Containment testing ray-casts
against every triangle, and it was 84 % of mesh-build time. Deflection is
now voxel_size/50, clamped to [0.01, 0.5] mm. On the reference assembly at a
0.20 in voxel: mesh build 16.2 s → 4.4 s, 54,558 triangles → 5,968, with
0.18 % of voxels changing part assignment.

## Animation

The solver overwrote its whole state every step; only a write-once arrival
time survived. It now captures ~60–120 evenly spaced frames of per-voxel
fill fraction and velocity magnitude, feeding a new "Fill Animation" results
view that shows metal actually flowing. Frames are discarded on a diverged
run — playing back a blow-up would be the most convincing possible
presentation of a failed solve. Also: the existing "Fill Time" view's Play
button was never enabled, so the only way to watch it was dragging the
slider; that is fixed.

## Performance

The solver was ported to Numba (`@njit(parallel=True, cache=True)`), taking the
real assembly from 67.4 s to 18.5 s — about 8×, to ~19 M cell-updates/s. The
NumPy path is kept and a parity test holds the two together; that test found
three real bugs in the port, including a bounce-back that tested `x + cᵢ` where
the reference tests `x - cᵢ`.

## What is not done

Stated plainly, because a field test should not discover these:

- **No configuration produces a verified complete fill.** See "Current
  state" above — this supersedes the resolution table this document used to
  carry, which rested on the invalid 100 % run.
- **Throughput** is ~19 M cell-updates/s against a 100–200 M/s target for a
  kernel of this shape. The remaining levers are float32/float64 conversion
  glue and dense-box iteration, which wastes ~6.8× at typical gating
  occupancy.
- **The inlet** now prescribes a constant entry velocity — Torricelli
  √(2gh) in head mode, or pour rate ÷ inlet area in velocity mode, both
  capped at the Mach guard — rather than rebuilding from the flow solution
  every step (the bug described above). It still does not ramp and does not
  model a falling stream; a Zou–He inlet spanning the sprue cross-section
  remains the intended longer-term replacement.
- **No fluidity-spiral validation.** F2 is checked against physical
  invariants, not measured spiral lengths. Absolute fill times are
  unvalidated — and, at present, unmeasurable in the first place, since no
  run has completed to time.
- **`QVTKRenderWindowInteractor` segfaults under `QT_QPA_PLATFORM=offscreen`**,
  because VTK's `SetWindowInfo(winId())` dereferences an NSView the offscreen
  plugin never creates. Worked around inside the test harness only; no repo code
  changed. This blocks headless GUI testing generally and is still open.

Until a configuration produces a `complete` run, treat every fill-percentage,
timing, and defect number this stage reports as unverified. A `diverged`
run's results are discarded by the stage itself and must not be read at all,
by design. Filter-capacity and pressure-drop caveats above stand independent
of this problem and remain estimates regardless of how the velocity runaway
is resolved.
