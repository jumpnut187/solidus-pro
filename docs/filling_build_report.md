# Mold filling: build report

What was built for the mold-filling stage, why each choice was made, and what
was actually verified rather than merely written. Written to be read once, by
someone deciding whether to trust this code in a field test.

Three companion documents divide the same subject: `filling_implementation.md`
is the original plan, `filling_field_notes.md` is the operator's guide, and
`ARCHITECTURE.md` says where the stage sits in the pipeline. This file is the
record of the work itself.

Delivered across 12 commits, `f17147d..fb274ac`, on `refactor/pouring-seams`.

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
supplies the momentum sink.

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
position — that is the misrun view, and it is the strongest output the stage
produces, because it is topological rather than quantitative.

F2 hands the fill's real temperature field to solidification instead of the old
two-constant placeholder, so metal that travelled furthest starts cooler. F3
reports entrapped gas and cold shuts.

## What was actually verified

Not just unit tests. The whole thing was driven end-to-end through the real
`MainWindow` — real `PartManager`, real analysis-options dialog, real
`SimulationWorker` thread, real VTK pipeline — on `Test_Model/Filling_Test`:

- All 8 STEP files loaded; types auto-detected.
- Pour location on the Down Sprue top face, `(-9.42, -5.25, 9.44) in`.
- **16,494 / 16,494 metal voxels filled — 100 %**, 0 misrun, 0 entrapped gas,
  0 cold shuts, in **4.50 s of a 30 s cap** (finished, not truncated).
  832 s wall at voxel 0.25.
- The animation swept through the real signal chain at 11 slider positions,
  with voxel counts read off actual VTK polydata.

Suite-wide: **953 passed, 11 skipped**. The filling-specific suites account for
300 tests and 761 subtests.

## Three bugs that only running it for real could find

Worth recording, because each passed every test that existed at the time.

1. **A NaN leak into the handover** (`349c298`). `fill_time_s` is written once
   and never cleared, but temperature is NaN-reset when a cell drains. A drained
   cell therefore had a valid fill time and a NaN temperature, and solidification
   rejected the field.

2. **Mold reported as misrun** (`fb274ac`). The Fill Time view counted and drew
   every non-metal voxel as metal the pour never reached, because
   `gather_to_voxels` returns the same `-1` for "metal that never filled" and
   "not metal at all". A perfect 100 % fill reported **1,485 never fill** — one
   per chill voxel — against a true count of 0. Non-metal is now `NaN`, which is
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

## Performance

The solver was ported to Numba (`@njit(parallel=True, cache=True)`), taking the
real assembly from 67.4 s to 18.5 s — about 8×, to ~19 M cell-updates/s. The
NumPy path is kept and a parity test holds the two together; that test found
three real bugs in the port, including a bounce-back that tested `x + cᵢ` where
the reference tests `x - cᵢ`.

## What is not done

Stated plainly, because a field test should not discover these:

- **Grid convergence is unvalidated.** The same geometry gave 5.8 % to 100 %
  filled depending on resolution. Hold voxel size fixed within any comparison.
- **Resolution decides whether the fill works at all.** A channel one cell
  across is sealed by halfway bounce-back. This gate is 0.5 in: at a 0.60 in
  fill grid it stalls at ~12 %, at 0.25 in it fills completely. Size the grid
  against the thinnest flow section, not the casting.
- **Throughput** is ~19 M cell-updates/s against a 100–200 M/s target for a
  kernel of this shape. The remaining levers are float32/float64 conversion glue
  and dense-box iteration, which wastes ~6.8× at typical gating occupancy.
- **The inlet** is a Dirichlet pressure source over the picked cells. It does
  not ramp and does not model a falling stream; a Zou–He inlet spanning the
  sprue cross-section is the intended replacement.
- **No fluidity-spiral validation.** F2 is checked against physical invariants,
  not measured spiral lengths. Absolute fill times are unvalidated.
- **`QVTKRenderWindowInteractor` segfaults under `QT_QPA_PLATFORM=offscreen`**,
  because VTK's `SetWindowInfo(winId())` dereferences an NSView the offscreen
  plugin never creates. Worked around inside the test harness only; no repo code
  changed. This blocks headless GUI testing generally and is still open.

Treat fill sequence, misrun and same-resolution comparisons as trustworthy.
Treat absolute times, cross-resolution percentages and filter pressure drop as
estimates.
