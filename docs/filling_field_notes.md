# Mold filling: field-test notes

What the filling stage does, how to drive it, and — more importantly — what its
answers are and are not good for. Written for a field-test campaign, not as a
design document; `docs/filling_implementation.md` is the plan and
`docs/ARCHITECTURE.md` describes where the code sits.

## Running a fill

1. Load the assembly and set part types. Gating bodies auto-detect from CAD
   names: `sprue`/`downsprue`, `runner`, `gate`/`ingate`, `pour cup`/`basin`,
   `filter`/`stelex`. Verify them in the parts manager — the fill only follows
   metal, so a mistyped body changes the answer.
2. **Set a pour location.** Click a point on the assembly, normally the top face
   of the down sprue. Until one exists, the Mold Filling analysis stays disabled;
   that is deliberate, not a bug.
3. Enable **Mold Filling (CFD)** and run. The header line lists it when it is
   active, so a log tells you whether the fill actually ran.
4. Results → **Fill Time** animates metal arrival. Never-filled voxels stay grey
   at every slider position — that is the misrun view.

## Settings that matter

**Voxel size is the dominant control, and not for the obvious reason.** Work
scales as roughly `1/dx^4`: a coarser cell means fewer cells *and* a larger
timestep. Halving resolution is about 16x less work.

But resolution also decides whether the fill works at all. The metal has to get
through the thinnest section of the gating, and a channel one cell across is
sealed by halfway bounce-back — the walls sit at ±0.5 cells and leave no
interior. Measured on `Test_Model/Filling_Test`, whose gate is 0.5 in thick:

| fill grid | gate thickness | outcome |
|---|---|---|
| 0.60 in | <1 cell | stalls at ~12 % |
| 0.60 in (0.30 mesh, coarsening 2) | ~1 cell | stalls at ~28 % |
| **0.25 in** | **2 cells** | **fills 100 % in 4.5 s** |

The lesson generalises: **size the fill grid against the thinnest flow section,
not against the casting.** Aim for at least 2 cells across the choke, preferably
3–4. `fill_grid_coarsening` multiplies the mesh size, so mesh 0.50 with
coarsening 2 gives a 1.00 in fill grid — check the product, not the mesh.

Give `max_fill_time_s` real headroom. A pour cut off at the cap looks identical
to a stall in the output; the run above finished at 4.5 s of a 30 s cap.

## What the results are good for

**Trustworthy:**
- *Fill sequence* — the order regions fill, and where the front goes.
- *Misrun* — regions metal never reaches. This is the strongest output: it is a
  topological result, not a quantitative one.
- *Relative comparisons* at fixed resolution — gating layout A against B, or
  pour point A against B.

**Treat with caution:**
- *Absolute fill times.* Not validated against measured pours.
- *Fill percentage across different resolutions.* The result is **not
  grid-converged**: the same geometry gave 5.8 % to 100 % filled depending on
  resolution, mostly because of the gate-resolution effect above. Hold voxel size
  fixed within a comparison.
- *Filter pressure drop.* The closure is Ergun-fitted on alumina-in-water data
  with Ward's inertial coefficient. No permeability is published for any STELEX
  product, so treat the drop as an estimate with a ±2x band. The one cheap
  measurement that would retire this is a cold-water head-flow rig on a real
  filter — see the note in `core/filter_database.py`.
- *Cold shut.* Requires fronts meeting **and** metal below liquidus. It cannot
  see a meeting that happened transiently and was then overwritten by continued
  flow.

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
  converge as `fill_grid_coarsening` → 1; that has not been demonstrated.
- **The inlet** is a Dirichlet pressure source over the picked cells. It does not
  ramp, and it does not model a falling stream. A Zou–He inlet spanning the sprue
  cross-section is the intended replacement.
- **No fluidity-spiral validation.** F2's thermal field is checked against
  physical invariants — bounded by pour and mold temperature, metal further from
  the inlet is cooler — not against measured spiral lengths.

## If something goes wrong

- *Fill stalls at a low percentage* — check the choke resolution first. Report it
  with the fill grid size, not the mesh size.
- *"no pour location has been selected"* — the gate working; set one.
- *Filling absent from the run header* — the analysis was not enabled.
- *Every gas pocket reads as entrapped* — the vent mask is empty. The stage warns
  about this; a cavity with no vent cannot fill physically either.
