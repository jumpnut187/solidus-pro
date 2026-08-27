# Plan: Mold-Filling Simulation for Solidus Pro

Design document for the pouring/filling module. Follows the pattern of
`gas_implementation/gas_implementation.md`: equations, fields produced, files
touched, validation, and what it explicitly does not require.

Nothing here is implemented yet. The **seam is**: `SolidificationEngine`
already accepts a computed initial temperature field via
`set_initial_temperature_field()`, so a filling stage has somewhere to hand its
result.

---

## 1. Scope and staging

Ship in three stages. Each is independently useful and independently
verifiable; do not start the next until the previous one validates.

| Stage | Produces | Validates against |
|---|---|---|
| **F1 — Isothermal hydrodynamics** | `fill_time[]`, `fill_fraction[]`, final velocity field | Analytical: hydrostatic rest, Poiseuille, dam break |
| **F2 — Thermal coupling** | `T_after_fill[]`, feeds `set_initial_temperature_field` | Fluidity spiral length, measured pour temperatures |
| **F3 — Defect indicators** | air entrapment, cold-shut risk, misrun risk | Known-bad castings, foundry trials |

**F1 is the whole risk.** If the hydrodynamics are stable and conservative,
F2 and F3 are comparatively routine. Do not attempt F2 until F1 passes its
analytical benchmarks.

The first user-visible payoff arrives at F2: solidification stops assuming a
uniform pour temperature, and every downstream analysis (shrinkage, Niyama, hot
tear) inherits a physically-earned initial condition. No new result views are
required for that — the effect shows up in fields that are already rendered.

---

## 2. Constraints that decide the design

These were verified against the code, not assumed. Each one rules something in
or out.

### 2.1 The mesh is a uniform Cartesian grid

`AdaptiveVoxelMesh` is a misnomer. `_adaptive_refinement` is defined at
`core/voxel_mesh.py:923` and **never called** — `generate()` explicitly skips it:

```python
# core/voxel_mesh.py:496
# Step 2: Skip refinement for memory efficiency
refined_voxels = initial_voxels  # No refinement to save memory
```

Measured with `refinement_level=3` requested: a single unique voxel size, and
uniform spacing on all three axes. This is the single most important fact for
the design — it makes lattice methods viable, which they would not be on a
genuinely adaptive mesh.

### 2.2 The voxel list maps bijectively to a dense array

`mesh.grid_shape` is populated (`core/voxel_mesh.py:571`) and agrees exactly
with the index derived from centers:

```python
ijk = np.round((centers - centers.min(axis=0)) / dx).astype(int)   # bijective
```

So the solver can work on dense `(nx, ny, nz)` arrays with array stencils rather
than traversing the 26-neighbour graph. For a solver taking 10⁴+ steps, that is
the difference between viable and not.

Measured on a test assembly: 859 voxels into an 11×9×11 box, occupancy 0.79.
Expect 0.3–0.8 for real castings — the gap is bounding-box waste and is the
price of the dense representation.

### 2.3 Face neighbours are already canonicalised

`core/voxel_mesh.py:1229` documents slots **0–5 as +X, −X, +Y, −Y, +Z, −Z**,
6–17 edges, 18–25 corners. Interior cells measured 5.73/6 face neighbours
present (surface cells legitimately have fewer). Slots 4/5 give gravity
alignment for free. Useful as a cross-check, though the dense mapping in §2.2 is
the primary access path.

### 2.4 There is no linear-solver infrastructure

No multigrid, no Krylov solver, no sparse-matrix assembly anywhere in the
codebase. Any method needing a pressure Poisson solve requires building and
parallelising that first. This is the deciding constraint in §3.

### 2.5 The parallel infrastructure that does exist

Numba `prange` kernels in ten modules, a thread-budget system
(`core/thread_budget.py`), and GPU backends already wired (`GPUBackend.METAL`
via MLX/Taichi, `CUDA` via CuPy, `OPENCL`). A method whose inner loop is local
and uniform maps onto all of it directly.

### 2.6 Memory budget

`get_max_voxels_for_system()` returns 2.4M on the development machine, hard
limit 4M. A D3Q19 double-buffered population array at 1.7M cells is
1.7e6 × 19 × 2 × 4 B ≈ **258 MB**, plus mass/flags/velocity ≈ 400 MB total.
Within budget, but it must be accounted for in `core/memory_limits.py` rather
than allocated blindly.

---

## 3. Solver choice: free-surface LBM with TRT collision

**Recommendation: lattice Boltzmann, D3Q19, two-relaxation-time collision, with
Körner-style free-surface mass tracking.**

### Why not VOF + projection

The conventional choice would be finite-volume VOF with a projection method.
It takes ~10× fewer timesteps (§7). It is rejected because every step needs a
pressure Poisson solve, and §2.4 says that infrastructure does not exist. A
robust, well-parallelised multigrid Poisson solver is plausibly more work than
the entire LBM implementation, and PLIC interface reconstruction is genuinely
hard to get right. The timestep advantage is real but does not survive the
implementation cost.

### Why LBM fits here

- Collision is purely local, streaming is a shift. No global solve.
- Maps directly onto Numba `prange` and the existing GPU backends (§2.5).
- Complex geometry is handled by bounce-back on the occupancy mask — no
  body-fitted meshing, no cut cells.
- Requires a uniform lattice, which §2.1 confirms is what the mesh already is.
- Free-surface LBM was developed specifically for foundry filling
  (Körner et al., *J. Stat. Phys.* 121 (2005) 179–196).

### The problem that will sink a naive implementation

Molten metal has a very low kinematic viscosity: ν = μ/ρ = 0.006/7200 =
**8.33 × 10⁻⁷ m²/s** for WCB. Converting to lattice units at every realistic
configuration gives a relaxation time essentially at the stability limit:

| dx | u_lattice | dt | ν_lattice | τ |
|---|---|---|---|---|
| 0.1 in (2.54 mm) | 0.05 | 127 µs | 1.64e-5 | **0.500049** |
| 0.1 in | 0.10 | 254 µs | 3.28e-5 | **0.500098** |
| 0.2 in (5.08 mm) | 0.05 | 254 µs | 8.20e-6 | **0.500025** |

BGK LBM is unconditionally unstable as τ → 0.5. **A plain BGK implementation
will diverge**, and it will do so in a way that looks like a geometry or
boundary-condition bug.

Two mitigations, both needed:

1. **TRT collision (mandatory).** Split each population into symmetric and
   antisymmetric parts with separate rates. With the magic parameter
   Λ = (1/ω⁺ − ½)(1/ω⁻ − ½) = 3/16 it is stable at very low τ and places the
   bounce-back wall exactly halfway between nodes. Cost is roughly 1.2× BGK.
   MRT is also viable and marginally more stable, at ~2× BGK cost and much more
   code.

2. **Smagorinsky subgrid viscosity (physically required anyway).** Reynolds
   numbers here are 6,000–720,000 (10 mm to 200 mm sections at 0.5–3 m/s), so
   the flow is strongly turbulent and nowhere near resolved. A subgrid model is
   correct physics, not a numerical crutch:

   ν_eff = ν + ν_t,  ν_t = (C_s Δ)² |S|,  C_s ≈ 0.16, Δ = dx

   Measured effect: ν_t is 5× molecular at bulk shear rates and ~80× near the
   gate, lifting τ to 0.500–0.504. **Helpful but not sufficient on its own** —
   which is why TRT is not optional.

Do not attempt to fix low τ by shrinking dt. It scales the wrong way: halving
dt halves ν_lattice and pushes τ *closer* to 0.5 while doubling the step count.

---

## 4. Core equations

### 4.1 Lattice and equilibrium (D3Q19)

Weights w₀ = 1/3 (rest), w₁₋₆ = 1/18 (faces), w₇₋₁₈ = 1/36 (edges); c_s² = 1/3.

```
f_i^eq = w_i ρ [ 1 + 3(c_i·u) + 4.5(c_i·u)² − 1.5 u² ]

ρ   = Σ f_i
ρu  = Σ c_i f_i  + F dt/2        (Guo forcing, see 4.3)
```

### 4.2 TRT collision and streaming

```
f_i^±     = ( f_i     ± f_ī     ) / 2
f_i^eq±   = ( f_i^eq  ± f_ī^eq  ) / 2

f_i*  = f_i − ω⁺ ( f_i^+ − f_i^eq+ ) − ω⁻ ( f_i^- − f_i^eq- ) + F_i

ω⁺ = 1/τ,   τ = 3 ν_lattice + 1/2
ω⁻ chosen from  Λ = (1/ω⁺ − ½)(1/ω⁻ − ½) = 3/16

f_i(x + c_i, t + 1) = f_i*(x, t)          (streaming)
```

### 4.3 Gravity (Guo forcing)

```
F_i = w_i (1 − 1/(2τ)) [ 3(c_i − u) + 9(c_i·u) c_i ] · F
F   = ρ g_lattice ẑ,   g_lattice = g · dt²/dx
```

Use `physics.units.GRAVITY_M_S2`; do not re-introduce a literal.

### 4.4 Free surface (Körner mass tracking)

Cell flags: `OBSTACLE`, `GAS`, `INTERFACE`, `FLUID`. Fluid fraction ε = m/ρ.

```
Δm_i(x, t+1) = ( f_ī(x + c_i, t) − f_i(x, t) ) · ( ε(x) + ε(x + c_i) ) / 2
m(x, t+1)    = m(x, t) + Σ_i Δm_i
```

Interface cells have no neighbour populations on the gas side; reconstruct:

```
f_ī(x, t+1) = f_i^eq(ρ_G, u) + f_ī^eq(ρ_G, u) − f_i(x, t)
```

with ρ_G the gas density. Surface tension enters through it:

```
ρ_G = ρ_atm − 2 σ κ / c_s²          (κ = local interface curvature)
```

Cell transitions with hysteresis κ_h ≈ 10⁻³ to prevent flip-flopping:
INTERFACE→FLUID when ε > 1 + κ_h; INTERFACE→GAS when ε < −κ_h. Newly filled
cells convert their gas neighbours to INTERFACE and initialise them from the
local equilibrium.

**Mass conservation is the primary correctness invariant.** Track total mass
every step and assert it against metal delivered through the inlet. Free-surface
LBM leaks mass when transitions are handled sloppily, and it is the failure mode
most likely to go unnoticed.

### 4.5 Thermal coupling (F2)

Advect temperature as a passive scalar over filled cells only. Either a second
D3Q7 population or explicit finite-difference advection–diffusion:

```
∂T/∂t + u·∇T = α ∇²T − q_wall / (ρ c_p Δ)
```

Wall heat loss `q_wall` uses the interface HTCs already configured in
`ThermalBoundaryParams` (`htc_cast_interface` etc.). A cell that has not filled
yet has no temperature — mask it, do not initialise it to ambient.

At handover, cells that never filled are a **misrun**: report them rather than
silently seeding them at pouring temperature.

---

## 5. Data structures

`physics/filling.py` owns its own state. It does **not** reuse
`HeatTransferModel`'s arrays, which have no active-cell mask and integrate all
`n_voxels` unconditionally.

```python
class FillingState:
    shape:      tuple[int, int, int]   # from mesh.grid_shape
    dx_m:       float                  # SI throughout
    dt_s:       float
    f:          np.ndarray  # (19, nx, ny, nz) float32, double-buffered
    rho:        np.ndarray  # (nx, ny, nz)
    u:          np.ndarray  # (3, nx, ny, nz)
    mass:       np.ndarray  # (nx, ny, nz)
    flags:      np.ndarray  # (nx, ny, nz) uint8: OBSTACLE|GAS|INTERFACE|FLUID
    fill_time:  np.ndarray  # (nx, ny, nz) float32, -1 = never filled
    temperature: np.ndarray # (nx, ny, nz) float32, F2 only
```

Use `float32` for populations. It halves bandwidth, and LBM is bandwidth-bound;
accumulate mass in `float64` because that is the conserved quantity.

Two index maps between the dense grid and the voxel list:

```python
voxel_to_grid: np.ndarray  # (n_voxels, 3) int32
grid_to_voxel: np.ndarray  # (nx, ny, nz) int32, -1 outside the mesh
```

Build once in `core/filling_stage.py`; keep `physics/filling.py` unaware of
`AdaptiveVoxelMesh` so the §2.4 layer boundary holds (enforced by
`tests/test_architecture_boundaries.py`).

---

## 6. Geometry and boundary conditions

### 6.1 Inlet part types

Extend `PartType` in `core/part_manager.py`:

```python
SPRUE     = "sprue"
RUNNER    = "runner"
GATE      = "gate"
POUR_CUP  = "pour_cup"
```

Then, in the same edit:

- add codes 5–8 to `PART_TYPE_TO_CODE`
- add all four to `METAL_PART_CODES` (they hold metal and solidify)
- add `FLOW_PATH_PART_CODES = (SPRUE, RUNNER, GATE, POUR_CUP)`
- add keywords to `PART_TYPE_KEYWORDS` (`sprue`, `runner`, `gate`, `ingate`,
  `pour_cup`, `basin`, `downsprue`)
- add display colours to `PART_TYPE_COLORS`

`tests/test_part_type_predicates.py` fails if a new type belongs to neither
`METAL_PART_CODES` nor `MOLD_PART_CODES`, so the grouping cannot be forgotten.

**These are metal but not casting.** They must be excluded from shrinkage and
porosity scoring exactly as risers are — reuse the existing exclusion mask in
`_calculate_sand_burn_in`/`_calculate_shrinkage` rather than inventing a new one.

**ML encoding:** adding part types does *not* require bumping
`PART_TYPE_DIVISOR`. New codes encode above `CORE`'s 4/3 and existing artifacts
keep loading. Bump it only to renormalise, and expect
`ModelManifest.part_type_divisor` to force a retrain if you do.

### 6.2 Inlet condition

The pour enters at the top face of the `POUR_CUP` (or the `SPRUE` top when no
cup exists). Two usable forms:

- **Velocity inlet** — impose u from pour rate ÷ inlet area. Simple, but ignores
  the falling stream and over-predicts gate velocity at low head.
- **Pressure/head inlet (preferred)** — impose ρ from the metallostatic head
  `ρ g h(t)`, with h following the pour schedule. Physically closer, and it lets
  the gate meter the flow rather than the boundary condition dictating it.

Implement with Zou–He on the inlet face. Ramp from zero over ~10 ms to avoid a
startup shock that will look like an instability.

### 6.3 Walls and vents

- Mold walls: **halfway bounce-back** on `OBSTACLE` cells. With TRT and
  Λ = 3/16 the wall sits exactly halfway between nodes.
- Chills and cores are obstacles hydrodynamically (`is_mold()` already gives the
  mask) but thermally active in F2.
- Vents/risers open to atmosphere: gas cells at ρ_atm that accept displaced air.
  Without at least one, the cavity cannot fill and air compresses unphysically.
  Default every top-facing riser surface to vented, with a parameter to close it.

### 6.4 Air entrapment (F3)

A gas region that becomes fully enclosed by FLUID with no path to a vent is
entrapped. Detect with a connected-component labelling of GAS cells each *output
frame* (not each step) against the vent set — `scipy.sparse.csgraph` is already
used for this pattern in `_get_orphan_metal_mask`.

---

## 7. Computational budget

Advective CFL for VOF vs the LBM Mach constraint, computed for a 4 s fill at
1 m/s characteristic velocity:

| Casting | dx | cells | VOF steps | LBM steps | LBM cell-updates |
|---|---|---|---|---|---|
| 6 in cube | 0.1 in | 216 k | 3,149 | 31,496 | 6.8e9 |
| 12 in cube | 0.1 in | 1.73 M | 3,149 | 31,496 | 5.4e10 |
| 12 in cube | 0.2 in | 216 k | 1,574 | 15,748 | 3.4e9 |
| 24 in cube | 0.25 in | 885 k | 1,259 | 12,598 | 1.1e10 |

At an optimistic 100–200 M cell-updates/s for a Numba `prange` D3Q19 kernel on
18 cores, the 12 in / 0.1 in case is **5–9 minutes**. Acceptable for a foundry
tool; not acceptable as an interactive preview.

### The single biggest lever: run the fill on a coarser grid

Filling does not need solidification's resolution. Fill sequence and heat loss
are governed by bulk flow, not by the millimetre detail that matters for a hot
spot. Halving fill resolution gives **16×** (8× cells × 2× steps):

```
12 in @ 0.1 in fill grid:  1,728,000 cells, 31,496 steps -> 5.4e10 updates
12 in @ 0.2 in fill grid:    216,000 cells, 15,748 steps -> 3.4e9  updates
```

Then trilinearly interpolate `T_after_fill` onto the fine solidification grid.
Make the ratio a parameter (`fill_grid_coarsening`, default 2) and validate that
results converge as it → 1.

---

## 8. Parameters

Add `FillingParams` to `core/solidification_params.py` and register it on
`SolidificationParameters`. That inherits `from_dict` deep-merge, `to_dict`,
`_clamp_values`, the `CASTSIM_SOLIDIFICATION__FILLING__*` env override, SQLite
round-trip, and UI plumbing at no extra cost.

```python
@dataclass
class FillingParams:
    enabled: bool = False
    pour_rate_lb_s: float = 5.0        # or pour_time_s, whichever the user sets
    pour_time_s: float = 0.0           # 0 = derive from rate and volume
    pour_temperature_c: float = 0.0    # 0 = use alloy.pouring_temp
    inlet_mode: str = "head"           # "head" | "velocity"
    initial_head_in: float = 4.0
    fill_grid_coarsening: int = 2      # see 7
    max_fill_time_s: float = 60.0      # abort guard
    smagorinsky_cs: float = 0.16
    trt_magic: float = 0.1875          # 3/16
    lattice_velocity_max: float = 0.05 # Mach guard; sets dt
    vent_risers: bool = True
    wall_htc_multiplier: float = 1.0   # F2
```

Do not thread these as flat `PipelineInputs` arguments. The nested-dict route is
why adding a solidification parameter costs ~9 touch points instead of ~54.

---

## 9. Material additions

`core/materials.py` already carries almost everything a momentum solver needs:
`dynamic_viscosity` (Pa·s), `get_dynamic_viscosity(T, fs)` — temperature- and
mush-dependent, already vectorised — `kinematic_viscosity`, `density_liquid`,
`thermal_expansion`, permeability and a Darcy pressure drop.

**Missing: `surface_tension`** (N/m). Required for §4.4's gas-density
correction and for the Weber number. Add to `AlloyProperties` with sourced
per-alloy values; molten steels are ≈1.7–1.9 N/m, copper alloys ≈1.1–1.3 N/m.

Optional and lower priority: `contact_angle` (mold wetting) and a fluidity
index. Note `sand_burn_in_susceptibility` is *not* a fluidity proxy despite its
comment mentioning fluidity — do not reuse it as one.

---

## 10. Units discipline

`physics/filling.py` works **entirely in SI**. Convert at the boundary only:

```python
from physics.units import GRAVITY_M_S2, inches_to_meters
dx_m = inches_to_meters(float(mesh.voxel_sizes[0]))
```

Lattice conversion, in one place, asserted in tests:

```
dt      = u_lattice_max · dx / u_physical_max
ν_lat   = ν_SI · dt / dx²
τ        = 3 ν_lat + 1/2
g_lat   = g · dt² / dx
```

Return `fill_time` in seconds and `T_after_fill` in °C so the result drops
straight into `set_initial_temperature_field` without further conversion.

`tests/test_units.py` already demonstrates the failure this prevents: a head
pressure computed with h in inches is wrong by 39× and looks entirely plausible.

---

## 11. Integration points

```
core/simulation_pipeline.py :: run_pipeline
    prepare_mesh_and_geometry()            <- unchanged
    if analysis_options.filling:           <- NEW
        result = run_filling_stage(mesh, alloy, params)
    engine = SolidificationEngine(...)
    engine.initialize()
    if result is not None:                 <- NEW
        engine.set_initial_temperature_field(
            result.temperature_c,
            fill_arrival_times=result.fill_time_s,
        )
    engine.run_simulation()
```

Model the stage on `core/surrogate/pipeline.py` — a sibling that reuses
`prepare_mesh_and_geometry` and owns its own solve. **Do not put filling logic
inside `SolidificationEngine`**, which is already ~5,400 lines in one class.

Filling and solidification stay **separate sequential stages with separate
timestep policies**. They are not co-solved: fill runs to completion in seconds
of physical time, then hands over. Interleaving CFD-scale steps across a
minutes-long solidification run is not feasible and is not necessary — the
timescales are three orders of magnitude apart.

---

## 12. Validation plan

Each stage gates the next. Analytical cases first; they catch the errors that
foundry cases only obscure.

**F1 — hydrodynamics**

1. **Hydrostatic rest.** Closed box, gravity on, zero initial velocity. Velocity
   must stay at zero to round-off. Catches spurious currents from the forcing
   scheme and bad wall treatment. *The single most valuable test — run it first.*
2. **Poiseuille flow.** Pressure-driven channel, compare to the parabolic
   profile. Validates viscosity calibration and the bounce-back wall position;
   with TRT Λ = 3/16 the fitted wall should land within a few percent of dx/2.
3. **Dam break.** Classic free-surface benchmark with published experimental
   surge-front data (Martin & Moyce). Validates mass tracking and interface
   advection.
4. **Mass conservation.** Every case: total mass equals inlet-delivered mass to
   within 0.1 %. Assert every step in debug builds.

**F2 — thermal**

5. **Fluidity spiral.** Standard foundry test; spiral length vs pour temperature
   is tabulated for common alloys. Validates the coupling of heat loss to flow
   arrest.
6. **Known misrun.** A thin section that demonstrably fails to fill should be
   reported as a misrun, not silently seeded at pouring temperature.

**F3 — defects**

7. **Air entrapment in a known-bad gating design** — a bottom-gated part with a
   blind pocket.
8. **Cold shut** where two fronts meet late; compare against the sectioned part.

Record measured-vs-predicted in `tests/data/` following the existing
`porosity_reference_targets.json` pattern — including its lesson: state which
commit each baseline was measured at, and use `subTest` so one drifted metric
does not mask the others.

---

## 13. Test strategy

- `tests/test_filling_lattice.py` — unit tests for equilibrium, TRT collision,
  streaming, and the lattice-unit conversion. No mesh needed.
- `tests/test_filling_free_surface.py` — mass conservation, cell-flag
  transitions, hysteresis.
- `tests/test_filling_benchmarks.py` — the analytical cases in §12. Mark slow
  ones and keep the fast ones in the default suite.
- `tests/test_filling_stage.py` — grid↔voxel mapping round-trip, handover to
  `set_initial_temperature_field`, and that `filling=False` leaves results
  byte-identical.

Add every new file to `mypy.ini` `files=`, or it is not type-checked at all.

The suite runs in ~55 s today and that is worth protecting. Benchmark cases
should run on small grids by default with a `CASTSIM_FILLING_FULL_BENCH=1`
escape hatch for the full-resolution versions.

---

## 14. Files to create and modify

**Create**

| File | Purpose |
|---|---|
| `physics/fluid_flow.py` | D3Q19 lattice constants, equilibrium, TRT collision, streaming, Numba kernels |
| `physics/filling.py` | `FillingState`, free-surface mass tracking, the step loop, inlet/wall BCs |
| `core/filling_stage.py` | Mesh↔grid mapping, parameter translation, coarsening/interpolation, orchestration |
| `tests/test_filling_*.py` | Per §13 |

**Modify**

| File | Change |
|---|---|
| `core/part_manager.py` | Four inlet `PartType` members, codes, groupings, keywords, colours |
| `core/materials.py` | `surface_tension` on `AlloyProperties`, values for all 7 alloys |
| `core/solidification_params.py` | `FillingParams`, registered on `SolidificationParameters` |
| `core/solidification.py` | `filling: bool` on `AnalysisOptions` |
| `core/simulation_pipeline.py` | Stage call in `run_pipeline`; `filling` in `create_analysis_options` |
| `core/memory_limits.py` | Account for the population arrays (§2.6) |
| `mypy.ini` | Add the new files to `files=` |
| `docs/ARCHITECTURE.md` | Document the new stage once it exists |

---

## 15. What this does NOT require

- **No result-field registry.** F1/F2 produce no new viewable fields; the effect
  appears in solidification outputs that are already rendered. The registry work
  (deferred Phase 4) becomes worthwhile at F3, when fill time and entrapment
  need to be seen.
- **No changes to `HeatTransferModel`.** Filling owns its own state. Do not
  retrofit an active-cell mask into the thermal model.
- **No second timeline in the UI.** Follows from the first point.
- **No `PART_TYPE_DIVISOR` bump.** See §6.1.
- **No new GPU backend.** The existing MLX/Taichi/CuPy managers cover it; start
  CPU-only with Numba and port the collide-stream kernel once F1 validates.
- **No mesh refactor.** The grid is already uniform (§2.1) and already maps to a
  dense array (§2.2).
- **No checkpointing yet.** Wanted eventually for long runs, but a 5–9 minute
  fill does not need it. Add when F1 timings say otherwise.

---

## 16. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| BGK instability at τ ≈ 0.5 | **High — will happen** | TRT from the start (§3). Do not prototype with BGK "for simplicity"; it will diverge and cost days of misdirected debugging. |
| Mass leakage in free-surface transitions | **High** | Assert conservation every step in debug; test 4 in §12 |
| Wall-clock too slow at full resolution | Medium | `fill_grid_coarsening` (§7), then GPU port |
| Spurious currents from the forcing scheme | Medium | Hydrostatic rest test (§12.1) before anything else |
| Turbulence unresolved at these grids | Medium | Smagorinsky; be explicit that this is LES, not DNS |
| Mushy-zone viscosity blowup near the gate | Medium | `get_dynamic_viscosity` already clamps to 250× base; feed ν through the same clamp |
| Air compression with no vent | Low | Default risers vented (§6.3) |
| Memory at 4M voxels | Low | §2.6; budget in `core/memory_limits.py` |

---

## 17. Suggested milestones

1. `physics/fluid_flow.py` — lattice constants, equilibrium, TRT collision,
   streaming, unit tests. No free surface, no geometry.
2. Single-phase flow in a closed box: hydrostatic rest + Poiseuille pass.
3. Free-surface mass tracking; dam break passes; mass conserved to 0.1 %.
4. `core/filling_stage.py` — grid mapping, inlet from `PartType.SPRUE`, fill a
   real casting geometry and produce `fill_time[]`. **End of F1.**
5. Thermal advection + wall loss; wire to `set_initial_temperature_field`;
   fluidity-spiral validation. **End of F2.**
6. Air entrapment and cold-shut indicators; result-field registry work becomes
   worthwhile here. **End of F3.**

---

## References

- Körner, Thies, Hofmann, Thürey, Rüde. *Lattice Boltzmann Model for Free
  Surface Flow for Modeling Foundry Processes.* J. Stat. Phys. 121 (2005)
  179–196. The method this plan follows.
- Ginzburg, d'Humières, Kuzmin. *Optimal stability of advection-diffusion
  lattice Boltzmann models with two relaxation times.* Comm. Comput. Phys. 10
  (2011). Source of Λ = 3/16.
- Thürey, Rüde. *Free surface lattice-Boltzmann fluid simulations with and
  without level sets.* VMV 2004. Practical treatment of cell transitions.
- Martin, Moyce. *An experimental study of the collapse of liquid columns on a
  rigid horizontal plane.* Phil. Trans. R. Soc. A 244 (1952). Dam-break data.
- Campbell. *Complete Casting Handbook.* Gating design, fluidity, cold shuts.
