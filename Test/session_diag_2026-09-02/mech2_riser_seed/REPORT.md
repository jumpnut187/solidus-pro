# MECHANISM 2 — VERIFIED TRUE (with a corrected causal chain)

**Verdict: the riser-top wetting is real, and it is a periodic streaming/neighbourhood
wrap-around through un-walled domain faces. But the wrap that seeds the riser is NOT the
riser's own open top — it is the CASTING'S MISSING FLOOR at `z = 0`.**

The solve domain is sized exactly to the geometry's bounding box, so every one of the six
array faces has metal cells lying flush on it with no obstacle beyond. Both step paths
index neighbours with `% n`. Metal reaching `CAST` cells on the `z = 0` plane therefore
becomes a direct 18-direction neighbour of `RISER` cells on the `z = 93` plane, 93 empty
planes away. 298 `(x, y)` columns pair the two. The riser top is promoted to `INTERFACE`,
seeded with populations and mass by `_seed_promoted_cells`, and then grows a phantom pool
*downward* from `z = 93` — which is precisely the observed signature (isolated wet cell at
`z = 91`, then 719 over-filled riser interface cells at `z 86-90` and 1,263
negative-population riser cells at step 16,493, while the real front is still below
`z ≈ 40`).

An open top **on its own is harmless** and is proven so by control fixture below. This
matters a great deal for the fix: sealing the riser top is not required and not desirable.

Repo state: branch `refactor/pouring-seams`, HEAD `f1daf97`, no tracked file modified.

---

## 1. STATIC FACE AUDIT

Probe: `Test/session_diag_2026-09-02/mech2_riser_seed/face_audit.py`,
`wrap_pairs.py`, `face_audit_any.py`
Input: `Test/session_diag_2026-09-01/r3_t2_ledger.csv.types.npy`, shape `(135, 135, 94)`,
344,532 part cells (the stage reports 339,824 metal cells; the difference is the 4,708
`FILTER` cells, which `_obstacle_mask` deliberately keeps permeable but `_metal_mask`
excludes).

### 1a. Every one of the six faces carries non-obstacle cells

| face | part cells on it | composition |
|---|---|---|
| `x = 0`   | 237  | CAST 237 |
| `x = 134` | 171  | SPRUE 169, CAST 2 |
| `y = 0`   | 55   | RUNNER 35, GATE 20 |
| `y = 134` | 55   | RUNNER 35, GATE 20 |
| `z = 0`   | 1372 | CAST 1372 |
| `z = 93`  | 1968 | RISER 1801, SPRUE 132, CAST 35 |

Per-part extents confirm the grid is the geometry's bounding box with **zero mold margin**:

```
CAST     z[0,93]  x[0,134] y[1,133]  n=193814
RISER    z[32,93] x[24,76] y[41,93]  n=113114
SPRUE    z[14,93] x[117,134] y[58,76] n=11182
RUNNER   z[16,32] x[27,124] y[0,134] n=17642
FILTER   z[14,32] x[104,122] y[40,94] n=4708
GATE     z[32,35] x[40,60] y[0,134]  n=4072
```

**Does the riser top reach `z = 93`? Yes** — 1,801 RISER cells sit on the top plane
(`x[26,74] y[43,91]`). **Does the sprue mouth reach `z = 93`? Yes** — 132 SPRUE cells
(`x[120,131] y[62,72]`); that is the inlet (the stage log reports "inlet 136 cells
(11 at the pour point, widened to the sprue mouth)"). The riser is the *only* part that
touches a face other than through the gating system's own extremities: its `y` extent tops
out at 93 < 134 and its `x` extent at 24-76, so **`z = 93` is the riser's only exposed
face**, and the `z = 0` plane is its only wrap partner.

### 1b. Opposite-face wrap pairs — metal streaming straight into metal

| axis | metal↔metal cells | pairing |
|---|---|---|
| `z` | **302** | **CAST@z=0 ↔ RISER@z=93: 298**; CAST@z=0 ↔ CAST@z=93: 4 |
| `x` | 117 | CAST@x=0 ↔ SPRUE@x=134 (at `y[63,71] z[20,32]`) |
| `y` | 55 | RUNNER@y=0 ↔ RUNNER@y=134 (35); GATE@y=0 ↔ GATE@y=134 (20) |

The 298 CAST↔RISER columns occupy `x[37,63] y[43,91]`. Counting the full 18-direction
dilation (diagonals included), **360 RISER cells at `z = 93` are directly promotable by a
CAST cell turning FLUID at `z = 0`** (probe: `init_seed_audit.py`).

Two further wrap leaks exist and are separate spurious sources, not part of mechanism 2:
the sprue at `x = 134` (wet from the first seconds) wraps into CAST at `x = 0`, and the
runner/gate wrap into their own opposite ends across `y`.

### 1c. How each path indexes a neighbour at those faces — all six faces WRAP

Nothing clamps, nothing pads, nothing is guarded. Both paths are periodic everywhere, and
both are explicitly documented as *relying on an obstacle shell that this assembly does
not have*:

**Numba path (`physics/fluid_flow.py`)**

| operation | lines | indexing |
|---|---|---|
| `stream_kernel` source gather + bounce-back test | `1435-1441` — `sx = (x - cx) % nx` etc. | **wrap** |
| `free_surface_kernel` gas-side reconstruction | `1541-1543` — `bx = (x + Ci[i,0]) % nx` | **wrap** |
| `free_surface_kernel` Koerner mass exchange | `1596-1598` | **wrap** |
| `wall_heat_loss_kernel` obstacle-adjacency (`exposed`) | `1767-1777` — `(x + 1) % nx` … | **wrap** |
| `interface_neighbourhood_kernel` (gasless / detached) | `1837-1839` | **wrap** |
| `transition_front_kernel` promote/demote scatter | `2022-2024` | **wrap** |
| `redistribute_share_kernel` | `2069-2071` | **wrap** |
| `redistribute_gather_kernel` | `2123-2125` | **wrap** |
| promoted-cell / temperature seeding kernels | `2239`, `2319` | **wrap** |

`stream_kernel`'s own docstring (`physics/fluid_flow.py:1390-1392`):

> "Wraps periodically to match the NumPy reference; **real domains are closed by obstacle
> cells, so the wrap never fires.**"

**NumPy path (`physics/filling.py`)** — every neighbour access is `np.roll`, i.e. periodic:

| operation | lines |
|---|---|
| `stream` helper | `fluid_flow.py:464` |
| `_stream_with_bounce_back` blocked-test + transport | `2246`, `2248` |
| `_dilate` (used by promote/demote, heal, initial interface) | `457` |
| `create_state`'s initial interface layer | `404-407` |
| `advance_mass` Koerner pairing | `714-725` |
| `reconstruct_gas_populations` gas-neighbour test | `609-611` |
| `apply_transitions` promote/demote | `864`, `866` |
| `heal_submerged_interface` submerged test | `1013` |
| `reset_detached_interface_cells` support count | `1137` |
| `_initialise_promoted_cells` donor average | `1301` |
| `_initialise_promoted_temperature` | `1397` |
| `_redistribute` (NumPy fallback) | `1456-1466` |

`physics/fluid_flow.py:449-452` (`stream`'s docstring) makes the same assumption:

> "Periodicity is a property of this bare helper only — **real domains are closed by the
> obstacle mask and bounce-back** in `physics.filling`, which overwrite anything that
> wraps."

`tests/test_filling_stability.py:77-95` (`_walled_tower`) already knows the assumption can
fail — it was written *because* `_tower` was a torus — but nothing checks it for real
assemblies.

### 1d. The stage's own docstring states the contradiction outright

`core/filling_stage.py:844-857`, `_domain_boundary_mask`:

> "`build_grid_mapping` sizes the dense grid **exactly to the voxel centres' bounding box**
> … so a cell at index 0 or `shape - 1` along any axis is at the true edge of the meshed
> geometry — **nothing is modelled beyond it, and no wall was ever asserted there either**.
> A GAS cell found there is therefore open to atmosphere by omission."

`_obstacle_mask` (`core/filling_stage.py:1537-1552`) makes solid only `EMPTY_CELL` and
mold; the array boundary is not `EMPTY_CELL`, it simply does not exist. So the solver's
"real domains are closed by obstacle cells" and the stage's "no wall was ever asserted
there" are in direct contradiction, and the boundary plane is where they meet.

---

## 2. CODE AUDIT OF THE FAR-FIELD WRITERS

Question for each: **can it write flags, mass, or populations at riser `z 79-91` while the
front is below `z 40`?**

| # | candidate | can it? | why |
|---|---|---|---|
| **A** | **`apply_transitions` promote dilation** — `filling.py:864` / `transition_front_kernel` `fluid_flow.py:2022-2024` | **YES — this is the seed** | `promote = _dilate(to_fluid) & (flags == GAS)`. `_dilate` is `np.roll` (`filling.py:457`); `np.roll(mask, (0,0,-1))` moves the `z=0` plane onto `z=93`. A CAST cell at `z=0` crossing the fill threshold promotes the RISER cell at `z=93` in its own or a diagonally adjacent column. The Numba twin scatters through `tx/ty/tz = (s + Ci[i]) % n` and is bit-identical. |
| **B** | **`_seed_promoted_cells` / `_initialise_promoted_cells`** `filling.py:1185-1305` | **YES — writes the populations and rho** | Called unconditionally from both promotion sites. Its donor average is 18 `np.roll` sums (`1301`), so the *donor* for the `z=93` riser cell is the wet CAST cell at `z=0`: it inherits that cell's density **and its velocity**. The "all donors are gas" fallback is not even reached — the wrap supplies a real donor. |
| **C** | **`free_surface_kernel` / `advance_mass` Koerner exchange** `fluid_flow.py:1596-1598` / `filling.py:714-725` | **YES — sustains it** | Once the riser cell is `INTERFACE`, `wet & neighbour_wet` is true across the wrap, so it exchanges mass every step with the cast bottom. This is the pump that fills the phantom pool. |
| **D** | **streaming** `fluid_flow.py:1435-1441` / `filling.py:2246-2250` | **YES — sustains it** | `flags[sx,sy,sz]` at the wrapped source is CAST metal, not obstacle, so bounce-back does not fire and populations are copied straight across 93 planes. |
| **E** | **`streamed[:, flags == GAS] = 0.0`** `filling.py:1911`; Numba `clear_gas` `fluid_flow.py:1415-1418` | **no — and it is why the seed must come from A** | Both paths zero populations arriving in a `GAS` cell (Numba does it *before* the gather, as a `continue`, so nothing reads them; NumPy does it after streaming and before `reconstruct_gas_populations`, so nothing reads them either — **the two are in parity and neither leaks**). This is exactly why the entry point is the *flag* promotion (A), not raw streaming: the wrap can only pollute a cell that is already non-GAS. |
| **F** | **`heal_submerged_interface`** `filling.py:916-1041` | no (cannot create), **but it is defeated** | Acts only on cells already `INTERFACE`. Its `submerged` test is `interface & ~_dilate(flags == GAS)` (`1013`) — the riser-top cell has plenty of real gas neighbours, so it never heals it. `HEAL_DWELL_STEPS = 200` never starts counting. |
| **G** | **`reset_detached_interface_cells`** `filling.py:1084-1153` | no (cannot create), **but it is defeated** | It would zero the spurious cell's momentum — but only if the cell has **no wet neighbour in all 18 directions** (`support == 0`, `1137`). The wrap gives it wet neighbours: measured in §3d, cell `(38,87,93)` has **five** wrapped wet CAST neighbours at `z = 0` plus four riser siblings, so `support == 9` and the guard never fires. **Both of the solver's two self-repair mechanisms are blinded by the same wrap that creates the defect.** |
| **H** | **`ensure_interface_skin`** `filling.py:1044-1082` | **no** | Restricted to `halo & (flags == GAS)`. The only caller-supplied halo is the inlet's, and measurement (`init_seed_audit.py`) shows that halo is **`z = 92` only**: the inlet's `+z` wrap lands at `(x 120-131, y 62-72, z 0)`, which is mold/obstacle, so `inlet_halo &= ~obstacle` (`filling_stage.py:2503`) masks it. Ruled out. |
| **I** | **`_expand_inlet_to_mouth`** `filling_stage.py:1603-1673` | **no** | Confined to the seeds' own `z_top` plane and to the part-type codes the seeds stand on (`codes = np.unique(layer[top])`, `member = np.isin(layer, codes) & _metal_mask(layer)`), then a 2-D connected-component label. Codes = `{SPRUE}`; the riser at `z=93` is `RISER` and a different component. Measured: 132 cells, all SPRUE, all at `z=93`. **Not** a z-slab selection that could reach the riser. Ruled out. |
| **J** | **`delivery_layers` top-up** `filling_stage.py:2528-2535` | **no** | Grown by `_below[:, :, :-1] = _frontier[:, :, 1:]` — an explicit **slice, not a roll**, so it does not wrap; and it is `&= metal & ~obstacle & ~_claimed`, i.e. strictly downward connectivity from the mouth through metal. It is the one neighbourhood operation in the whole pipeline that is correctly clamped. Ruled out. |
| **K** | **inlet pin** `filling_stage.py:2497-2503` + around 2700 | **no** | Writes only `inlet` (sprue mouth, `z=93`, `x[120,131]`) and its halo (`z=92`). Ruled out. |
| **L** | **vent mask / `find_entrapped_gas` / `_entrapment_warning_if_no_vent`** `filling.py:2086-2139`, `filling_stage.py:807-885, 2941-2949, 3045-3048` | **no** | `find_entrapped_gas` is pure: it labels components and **returns a mask**. Its callers only count voxels into `peak_entrapped_voxels` / `entrapped_coarse` and emit a warning string. Nothing writes `state.flags`, `state.mass` or `state.f`. `_static_vent_mask` is read-only and never touches state. Ruled out. |
| **M** | **frame / progress bookkeeping** | **no** | Frame capture reads `state`; no write path found into flags/mass/populations. Ruled out by the measurement in §3 as well — the "STAGE-between-steps" attribution hook was armed for the whole run. |
| **N** | `_redistribute` `filling.py:1407` / kernels `fluid_flow.py:2069, 2123` | YES (secondary) | Periodic, so excess mass can be pushed across a face. Not the seed, but another wrap-conserved-mass error once the pool exists. |
| **O** | `wall_heat_loss_kernel` `fluid_flow.py:1767-1777` | YES (thermal only) | Counts a wrapped obstacle as an exposed face, so cells on a boundary plane get the wrong wall heat loss. Cosmetic relative to the blow-up; fix it with the same change. |

**Conclusion of the audit: the writer is A + B (promote dilation + promoted-cell seeding),
the sustainers are C + D + N, and the two mechanisms that should have caught it (F, G) are
disabled by the same wrap.** Candidates E, H, I, J, K, L, M are ruled out.

### 2b. The GAS-cell zeroing is in parity, and it happens BEFORE any read (candidate E in detail)

The task asked specifically whether the Numba path matches `physics/filling.py:1911` and
whether the zeroing precedes any read. Both checked:

* **NumPy:** `streamed[:, state.flags == FLAG_GAS] = 0.0` (`filling.py:1911`) runs
  immediately after `_stream_with_bounce_back` (`1901`) and **before**
  `reconstruct_gas_populations` (`1915`) and `advance_mass` (`1917`). Nothing reads the
  wrapped population.
* **Numba:** `stream_kernel` (`fluid_flow.py:1415-1418`) tests
  `if clear_gas and flags[x,y,z] == _FLAG_GAS` as the *first* action for that destination
  cell, zeroes all 19 directions and `continue`s — so a GAS cell is never even gathered
  into. `free_surface_kernel` then skips every cell that is not INTERFACE or FLUID
  (`fluid_flow.py:1526`), so a GAS cell's populations cannot be read in the same step
  either.
* Note `collide_kernel` zeroes `post_out` only on OBSTACLE cells
  (`fluid_flow.py:1172-1183`), not on GAS cells — it collides them with `u` forced to zero
  (`1232-1235`). That is harmless only because `state.f` is held at zero on GAS cells by
  `create_state` (`filling.py:421`, `429`), `apply_transitions` (`filling.py:844`) and
  `stream_kernel`'s own `clear_gas`. It is worth knowing when auditing the padding fix:
  the pad cells are OBSTACLE, which is the branch that *is* explicitly zeroed.

**This is exactly why the wrap's entry point must be the flag promotion (A) and not raw
streaming: the wrap can only pollute a cell that is already non-GAS.**

---

## 3. INSTRUMENTED SIMULATION

### 3a. Minimal fixture — the wrap proven on BOTH step paths, with controls

Probe: `Test/session_diag_2026-09-02/mech2_riser_seed/wrap_fixture.py` (runs in ~30 s).

A 3x3 channel walled in `x` and `y` by obstacle cells, `nz = 14`, metal seeded at the
**bottom** (`z ≤ 1`). Three configurations, each on `use_numba=False` and `use_numba=True`:

```
=== A. OPEN z faces (Test 2's real condition: no floor, no lid) ===
  !! NumPy   open: TOP PLANE WET at step 1: 9 cells, first (1,1,13) flag=IFACE mass=0.5 rho=1 |f|=1
  !! Numba   open: TOP PLANE WET at step 1: 9 cells, first (1,1,13) flag=IFACE mass=0.5 rho=1 |f|=1
=== B. CONTROL: obstacle floor + obstacle lid (properly closed) ===
  NumPy closed: after 40 steps  wet-z=1-3  top-plane wet=0  first_top_wet_step=None
  Numba closed: after 40 steps  wet-z=1-3  top-plane wet=0  first_top_wet_step=None
=== C. floor only, top still open (riser-top-only case) ===
  NumPy  floor: after 40 steps  wet-z=1-3  top-plane wet=0  first_top_wet_step=None
  Numba  floor: after 40 steps  wet-z=1-3  top-plane wet=0  first_top_wet_step=None
```

Three things are established here, all by measurement:

1. **The wrap fires, identically on both paths.** 12 planes of empty gas separate the metal
   from `z = 13`, and `z = 13` is `INTERFACE` with `mass = 0.5`, `rho = 1`, `|f| = 1`.
2. **`mass = 0.5` identifies the writer exactly**: that is
   `mass[interface] = 0.5 * REFERENCE_DENSITY` at `physics/filling.py:407` — i.e. in this
   fixture the contamination is written by **`create_state`'s initial interface layer**
   (`neighbours = _dilate(fluid)`, `filling.py:404`), *before step 1 even runs*. On the
   real assembly the same `_dilate` is reached later, through
   `apply_transitions`' promote (`filling.py:864`).
3. **Configuration C is the decisive control: an open top ALONE is completely clean.** The
   wrap needs metal on the *opposite* face. This is what redirects the diagnosis from "the
   riser's open top" to "the casting's missing floor", and it is what makes a fix possible
   that keeps the riser vented.

### 3b. The seed does not self-heal — it grows a phantom pool downward

Probe: `Test/session_diag_2026-09-02/mech2_riser_seed/runaway_fixture.py` (~2 min).
Same channel at `nz = 48`, bottom two planes pinned `FLUID` every step with
`ensure_interface_skin` on their halo — the stage's own inlet-pin pattern — standing in for
a continuously fed cast bottom. Everything above `z ≈ 4` is spurious by construction.

```
### OPEN both z faces (Test 2's condition)   channel z 0..47, pinned FLUID at z 0-1
  step     1 wet=  36 | TOP cluster:    9 cells, reaching down to z=47 | neg=0 finite=True
  step   500 wet=  45 | TOP cluster:   18 cells, reaching down to z=46 | neg=0 finite=True
  step  1500 wet=  50 | TOP cluster:   23 cells, reaching down to z=45 | neg=0 finite=True
  step  2500 wet=  51 | TOP cluster:   24 cells, reaching down to z=44 | neg=0 finite=True
  step  3000 wet=  59 | TOP cluster:   32 cells, reaching down to z=44 | neg=0 finite=True
  step  3500 wet=  64 | TOP cluster:   37 cells, reaching down to z=43 | neg=0 finite=True
  step  4000 wet=  69 | TOP cluster:   42 cells, reaching down to z=42 | neg=0 finite=True

### CONTROL: obstacle floor + lid            step 1..4000: wet=27, z=1-3, TOP cluster: 0 cells
### CONTROL: floor only, top still open      step 1..4000: wet=27, z=1-3, TOP cluster: 0 cells
```

The spurious cluster **grows monotonically and never heals**: 9 cells at `z = 47` at step
1, 42 cells reaching down to `z = 42` by step 4,000, while the real metal never leaves
`z 0-1`. Both controls are flat at 27 wet cells for 4,000 steps.

This is quantitatively the Test 2 signature. On the real grid the phantom pool starts at
`z = 93` and eats downward, which is why the eps locator saw **an isolated cell at `z = 91`
at step 15,750, then a band of 719 over-filled riser interface cells at `z 86-90` at step
16,493** — a front descending from the top plane, ~7 planes deep, not an advancing metal
front. (The fixture's pool is hydrostatically quiet so it produces no negatives; on Test 2
the same pool is fed by wrapped populations copied from a violently accelerating cast
bottom — a real velocity written into a cell with no physical support — which is where the
1,263 negative-population riser cells come from.)

### 3c. The initial state and the inlet halo are clean (ruling out H, I, K)

Probe: `Test/session_diag_2026-09-02/mech2_riser_seed/init_seed_audit.py`.

```
inlet (sprue mouth at z=93): 132 cells
initial interface layer from that seed: 132   -> z extent 92-92 (SPRUE, x[120,131] y[62,72])
inlet halo z planes: [92]
298 (x,y) columns pair CAST@z=0 with RISER@z=93
riser cells (z>=60) directly promotable by the z=0 wrap: 360, all at z=93
```

The inlet's own `+z` wrap lands on mold at `z = 0` and is masked by `~obstacle`, so unlike
the fixture the *real* initial state is uncontaminated. The riser wrap can only fire once
metal reaches the `z = 0` CAST plane.

And it will: `reach_audit.py` labels the non-obstacle region and finds **exactly one
connected component of 344,532 cells containing the inlet** — so all 1,372 CAST cells on
`z = 0`, including all 298 wrap columns, are reachable metal, not a sealed pocket.

### 3d. Full-run instrumented probe on Test 2

Probe: `Test/session_diag_2026-09-02/mech2_riser_seed/riser_seed_probe.py`
Log: `Test/session_diag_2026-09-02/mech2_riser_seed/riser_seed.log`

Watch set: all 61,234 `RISER` cells with `z ≥ 60` (0 of them obstacle). Any non-GAS flag or
nonzero mass there is spurious while the front is below `z 40`. Attribution wrappers are
installed on `_collide_stream_numba`, `free_surface_kernel`, `apply_transitions`,
`heal_submerged_interface`, `reset_detached_interface_cells`, and on the gap between
`step()` returning and the next `step()` entering — which isolates every write
`core/filling_stage.py` makes between steps (inlet pin, metered top-up,
`ensure_interface_skin`, frame capture).

**Result — the wrap hypothesis is confirmed by direct measurement.**

The front descends the sprue, pools at the filter, enters the casting and falls:

```
hb step  4000 wet= 6245 zrange=14-93 watched_wet=0   <- metal at the sprue/filter bottom
hb step  8000 wet=13836 zrange=13-93 watched_wet=0   <- entering the casting
hb step 10000 wet=17012 zrange= 7-93 watched_wet=0
hb step 11500 wet=19764 zrange= 3-93 watched_wet=0
hb step 12000 wet=20734 zrange= 1-93 watched_wet=0   <- one plane above the cast floor
```

For 12,155 steps the 61,234 watched upper-riser cells are untouched. Then, in the **very
step in which metal reaches the `z = 0` plane**:

```
*** FIRST SPURIOUS WRITE  step=12156  op=apply_transitions  cells=5
    global wet-cell z extent: 0-93;  wet total 21061;  z<60: 17682, z>=60: 3379
    watched cells z histogram: {93: 5}
  --- cell (38,87,93) part=riser flag=IFACE mass=0.00056736 rho=1.00057 |f|sum=1.00057 fmin=0.0265786
        nb i= 1 c=( 1, 0, 0) -> (39,87,93)         part=riser flag=IFACE mass=0.00056736 |f|=1.00048
        nb i= 5 c=( 0, 0, 1) -> (38,87,0) WRAPPED  part=cast  flag=IFACE mass=0.691041  |f|=0.999465
        nb i= 7 c=( 1, 1, 0) -> (39,88,93)         part=riser flag=IFACE mass=0.00056736 |f|=1.00112
        nb i= 9 c=( 1,-1, 0) -> (39,86,93)         part=riser flag=IFACE mass=0.00056736 |f|=1.00011
        nb i=11 c=( 1, 0, 1) -> (39,87,0) WRAPPED  part=cast  flag=FLUID mass=1.00098   |f|=0.999892
        nb i=14 c=(-1, 0, 1) -> (37,87,0) WRAPPED  part=cast  flag=IFACE mass=0.0926423 |f|=1.00038
        nb i=15 c=( 0, 1, 1) -> (38,88,0) WRAPPED  part=cast  flag=IFACE mass=0.342547  |f|=1.00082
        nb i=18 c=( 0,-1, 1) -> (38,86,0) WRAPPED  part=cast  flag=IFACE mass=0.323635  |f|=0.999176
```

Read off the dump, every element of the diagnosis is now measured rather than inferred:

| question | measured answer |
|---|---|
| **step** | **12,156** — the same step the front first occupies `z = 0` |
| **cell** | `(38,87,93)` and 4 siblings, **all RISER, all at `z = 93`**, the top plane; z histogram is exactly `{93: 5}` |
| **operation** | **`apply_transitions`** — the promote dilation. The two sub-operations that run *before* it in the same step (`_collide_stream_numba`, `free_surface_kernel`) were wrapped with the same check and did **not** trip. |
| **values written** | `flag = INTERFACE`, `rho = 1.00057`, a **complete equilibrium population set** (`|f|sum = 1.00057`, `fmin = +0.0265786`), `mass = 0.00056736`. So the cell is created holding a full cell-volume of populations at reference density while its Koerner mass bookkeeping says it is empty — exactly what `_seed_promoted_cells` → `_initialise_promoted_cells` (`filling.py:1185-1305`) writes, with the density and velocity averaged from its donors. |
| **is the source across an array edge?** | **YES — every wet donor is.** Five of the eight listed neighbours are marked `WRAPPED`, all in the `c_z = +1` family (`i = 5, 11, 14, 15, 18`), and **all five are `part=cast` cells on `z = 0`**. The trigger is `i = 11 -> (39,87,0)`, flag **`FLUID`**, `mass = 1.00098` — a just-filled cast-floor cell, i.e. exactly the `to_fluid` input to `_dilate` at `filling.py:864`. The only non-wrapped neighbours are the 4 riser siblings promoted in the same step, which carry the identical `mass = 0.00056736` and so are not donors either. |
| **distance from the real front** | the front is at `z ≤ 1`; the write is at `z = 93`. **92 empty planes.** There is no physical path between them. |

**And the two self-repair mechanisms are measured to be blinded.** The probe checked again
after `heal_submerged_interface` and after `reset_detached_interface_cells` ran in the same
step 12,156, and after `_collide_stream_numba` on step 12,157: **all 5 cells are still
present with byte-identical values every time.** Neither guard fires, for the reason the
code audit predicted — `reset_detached_interface_cells` needs `support == 0`
(`filling.py:1137`) and these cells have wet neighbours (four riser siblings plus five
wrapped cast cells); `heal_submerged_interface` needs no gas neighbour
(`filling.py:1013`) and these cells are surrounded by real riser gas.

**The `STAGE-between-steps` hook never fired once in 12,157 steps**, which independently
rules out every stage-level writer (inlet pin, metered top-up, `ensure_interface_skin`,
frame/progress bookkeeping — candidates H, J, K, M).

**Timeline closure:** the seed lands at step **12,156**; the previously logged first
negative-population riser cell is at step **15,750** (`z = 91`) and the non-finite state at
**16,493** (1,263 negative riser cells, 719 over-filled riser interface cells at `z 86-90`).
That is 3,594 steps of the phantom pool growing downward from `z = 93` before it produces
its first negative — the same latency and the same downward direction the
`runaway_fixture.py` measurement shows (§3b), and the reason the earlier `eps_locator`
sampling at 250-step intervals saw the *symptom* at `z = 91` rather than the *seed* at
`z = 93`.

---

## 4. VERDICT AND FIX DIRECTION

### 4a. Verdict

**MECHANISM 2 IS VERIFIED TRUE**, with one correction to its stated cause.

* **True:** isolated wet cells appear in the upper riser far above the front; the blow-up
  starts at the riser *top*, seeded by a detached wet cell, not by advancing metal; the
  cause is **streaming and neighbourhood wrap-around through un-walled domain faces**
  (candidate 1 of the six listed).
* **Corrected:** the guilty face is **not** the riser's own open top. It is the
  **casting's missing floor at `z = 0`**. `CAST` cells lie on the `z = 0` plane with no
  obstacle beneath them; 298 `(x, y)` columns place `CAST@z=0` directly opposite
  `RISER@z=93`; the periodic `-z` neighbour of a filling cast-bottom cell **is** a riser-top
  cell. The `floor only, top still open` control (§3a-C) is clean for 4,000 steps on both
  paths, which proves an open top is harmless by itself.
* **Ruled out** (with the lines that make them impossible): vent/trapped-air bookkeeping
  (L — `find_entrapped_gas` is pure and its callers only count),
  `ensure_interface_skin` / the inlet-pin halo (H, K — halo measured at `z = 92` only, the
  wrap component masked by `~obstacle` at `filling_stage.py:2503`),
  `heal_submerged_interface` (F — requires no gas neighbour, `filling.py:1013`),
  `_expand_inlet_to_mouth` (I — confined to `z_top` and to the seeds' own part codes and
  connected component, measured 132 SPRUE cells at `z = 93`),
  the `delivery_layers` top-up (J — grown by an explicit slice `_below[:, :, :-1] =
  _frontier[:, :, 1:]` at `filling_stage.py:2529`, the one correctly clamped neighbourhood
  op in the pipeline), and frame/progress bookkeeping (M).
* **Not the same event as Mechanism 1.** Mechanism 1 lives at the filter outlet, `z 14-15`;
  this one is seeded at `z = 93` and eats downward. They share only the final NaN.

### 4b. The exact lines responsible

**Root cause (one line of intent, two files of consequence):** the solve grid is sized to
the geometry's bounding box with no mold margin
(`core/filling_stage.py:844-857`, `_domain_boundary_mask`'s own docstring: *"nothing is
modelled beyond it, and no wall was ever asserted there either"*), while both solver paths
index neighbours modulo the array and document that they rely on an obstacle shell that is
therefore absent (`physics/fluid_flow.py:1390-1392` and `449-452`).

**The seeding write, both paths:**

| | line |
|---|---|
| NumPy | `physics/filling.py:864` — `promote = _dilate(to_fluid) & (state.flags == FLAG_GAS)`, with `_dilate`'s `np.roll` at `physics/filling.py:457` |
| Numba | `physics/fluid_flow.py:2022-2024` — `tx = (sx + Ci[i,0]) % nx`, `ty`, `tz` in `transition_front_kernel` |
| the values written | `physics/filling.py:1185-1305`, `_seed_promoted_cells` → `_initialise_promoted_cells`; donor average rolls at `1301`, so the riser-top cell inherits the cast bottom's density **and velocity** |
| same bug at construction | `physics/filling.py:404-407` — `neighbours = _dilate(fluid)` |

**The sustaining writes:** `physics/fluid_flow.py:1435-1441` (`stream_kernel`),
`1541-1543` and `1596-1598` (`free_surface_kernel` reconstruction and Koerner exchange),
`2069-2071` / `2123-2125` (redistribute); NumPy twins at `physics/filling.py:2246-2250`,
`609-611`, `714-725`, `1456-1466`.

**The two blinded guards:** `physics/filling.py:1013` (`heal_submerged_interface`'s
`~_dilate(gas)`) and `physics/filling.py:1137` (`reset_detached_interface_cells`'
`support == 0`, plus the Numba twin at `physics/fluid_flow.py:1837-1839`). Both would have
caught a genuinely isolated droplet; the wrap hands the spurious cell one real wet
neighbour, so neither fires.

### 4c. Concrete fix — recommended: pad the solve grid, change no solver code

Measured on Test 2 (`Test/session_diag_2026-09-02/mech2_riser_seed/fix_cost.py`):

```
PAD  np.pad(solve_types, 1, constant_values=EMPTY_CELL):
   shape (135,135,94) -> (137,137,96)   cells 1,713,150 -> 1,801,824  (+5.2%)
   metal cells preserved: 339824 == 339824  True
   every one of the six faces all-obstacle: True (x0 x-1 y0 y-1 z0 z-1)
   _static_vent_mask riser-top vents: unpadded 2247, padded 2247 -> preserved
```

**Step 1 — seal every face (fixes all ~40 wrap sites at once).** In
`core/filling_stage.py`, immediately after `solve_types` is chosen (around line 1771) and
before `_obstacle_mask` at line 1851:

```python
solve_types = np.pad(solve_types, 1, mode="constant", constant_values=EMPTY_CELL)
```

`_obstacle_mask` makes `EMPTY_CELL` solid (`filling_stage.py:1551`), so the pad *is* the
obstacle shell both paths already assume. **No change to `physics/fluid_flow.py` or
`physics/filling.py` is needed, so `tests/test_filling_parity.py` stays valid** — this is
the decisive argument for padding over clamping. Every wrapped access now lands on an
obstacle cell, and each of the ~40 sites already handles obstacle correctly: `stream_kernel`
bounce-backs, the reconstruction sees `!= FLAG_GAS`, the mass exchange sees `not wet`,
`transition_front_kernel` sees `!= FLAG_GAS` so it does not promote, `_dilate & (flags ==
FLAG_GAS)` is False on an obstacle cell.

Index bookkeeping this forces, all inside `core/filling_stage.py` (the honest cost):

* `inlet_seeds` / `inlet` — `_resolve_inlet_cells` (`1798`) and `_expand_inlet_to_mouth`
  (`1813`) must run on the padded grid, or their results shifted by `+1` per axis.
* result gather — crop the pad off before `_upsample` (`1681`) and `gather_to_voxels`
  (`453`), which assert against `mapping.shape`.
* `solve_z_in` at `2336-2339` — subtract the one-cell pad offset.
* `_static_vent_mask` (`807`) — **needs no change**: it tests the *type* grid
  (`above == EMPTY_CELL`, `836-838`), and above a padded riser top the pad is still
  `EMPTY_CELL`. Verified: 2,247 riser-top vents before and after.
* `_domain_boundary_mask` (`844`) — **does need re-pointing**. Measured: on the padded grid
  it selects only obstacle cells, so the `boundary_mask & (flags == FLAG_GAS)` vent term at
  `2941` and `3045` becomes identically empty (0 non-obstacle cells). Re-point it at the
  original boundary (index `1` and `shape-2`), or better at "non-obstacle cells face-
  adjacent to the pad".
* `filter_zones` / `build_porous_zone` / `htc_dense` / `measure_choke_thickness_cells` all
  take `solve_types` and are shape-agnostic — no change beyond receiving the padded grid.

**Step 2 — keep the riser's open top a real atmosphere boundary.** Step 1 alone caps the
riser with bounce-back one cell above its top. That is far less harmful than it sounds —
gas here is a void, not a compressible phase, so the riser still fills; the only physical
loss is that it can no longer *overflow*, and a metered over-pour would pressurize the top
cell instead of spilling. To keep it exact, insert a headspace **before** the obstacle cap
wherever metal cells lie on the original `+z` face (the 1,801 riser and 132 sprue cells):
2-3 planes of a **non-obstacle, non-metal "atmosphere" type**, then the `EMPTY_CELL` pad
above that. The cleanest implementation is a new part-type code for which
`core/part_manager.is_mold()` is False (so `_obstacle_mask` leaves it open) and
`is_metal()` is False (so `_metal_mask` excludes it and the fill-fraction denominator is
untouched, and headspace cells never read as unfilled metal). Then point the
`_domain_boundary_mask` vent term at those atmosphere cells, which is the physically right
place for it anyway.

**Step 3 — make this unrepresentable, and regression-test it.** Add a cheap invariant right
after `obstacle = _obstacle_mask(solve_types)` (`1851`):

```python
if np.any(~obstacle[[0, -1], :, :]) or np.any(~obstacle[:, [0, -1], :]) \
        or np.any(~obstacle[:, :, [0, -1]]):
    raise/emit(...)   # a non-obstacle cell on an array face means the lattice is a torus
```

and a unit test built from the fixtures in §3a — the `open both faces` / `floor only` /
`closed` triple is exactly the right regression test, runs in ~30 s, and covers both paths.

**Optional belt-and-braces (not the primary fix): clamp the edge indexing.** Replace the
`% n` in the ~10 Numba sites in §1c with an explicit range test that behaves as "neighbour
is obstacle", and the ~30 `np.roll` calls in `physics/filling.py` with a shift-with-fill
helper (fill `False` for masks, `0.0` for fields, `FLAG_OBSTACLE` for flags). Correct, but
it is 40 individually-correct edits plus 40 parity re-verifications, versus one padding
change — do it as hardening after Step 1 lands, not instead of it.

**Emergency stopgap, if something must ship today** — one line after
`filling_stage.py:1851`:

```python
obstacle |= _domain_boundary_mask(solve_types.shape)
```

Measured cost on Test 2: **3,858 of 339,824 metal cells become solid (1.14%)** — CAST 1,646,
**RISER 1,801 (the riser's entire top plane)**, SPRUE 301, RUNNER 70, GATE 40. Because
`_metal_mask` is computed separately (`1852`), those cells stay in the fill denominator and
can never fill, so the fill can never report 100 % and the riser reads 1,801 cells short.
It does remove the divergence mechanism. Present it as a stopgap, not a fix.


### 4c-bis. THE FIX IS VALIDATED BY MEASUREMENT, on both paths

Probe: `Test/session_diag_2026-09-02/mech2_riser_seed/fix_validation.py` (~1 min).
A miniature with Test 2's exact topology — a `(12,12,24)` grid, a CAST box **flush with
`z = 0`** (no floor) and a RISER column **flush with `z = 23`** (open top), mold elsewhere.
Metal pinned at the cast floor only; 1,200 steps; three variants x both step paths:

```
grid (12,12,24): metal cells 1140
non-obstacle cells on the six faces (today): z=0 100, z=23 16, x/y faces 0

  BROKEN   numpy: upper-riser wet first=1    peak=140 final=140 | wet z=0-23 | metal=1140 | riser-top vents=16
  BROKEN   numba: upper-riser wet first=1    peak=140 final=140 | wet z=0-23 | metal=1140 | riser-top vents=16
  PADDED   numpy: upper-riser wet first=None peak=  0 final=  0 | wet z=1-3  | metal=1140 | riser-top vents=16
  PADDED   numba: upper-riser wet first=None peak=  0 final=  0 | wet z=1-3  | metal=1140 | riser-top vents=16
  STOPGAP  numpy: upper-riser wet first=None peak=  0 final=  0 | wet z=1-2  | metal=1140 | riser-top vents= 0
  STOPGAP  numba: upper-riser wet first=None peak=  0 final=  0 | wet z=1-2  | metal=1140 | riser-top vents= 0
```

* **BROKEN**: the entire upper riser floods from step 1 — 140 wet cells, `wet z = 0-23` —
  while the metal is pinned to the cast floor. Bit-for-bit the same on both paths.
* **PADDED (the recommended fix)**: **zero** spurious cells over 1,200 steps on both paths,
  the real front correctly confined to `z = 1-3`, **all 1,140 metal cells preserved**, and
  **the riser-top vent detection still returns 16 cells** — i.e. the padding fix sterilises
  the wrap *without* costing metal and *without* costing the riser's atmosphere boundary.
  This is the direct answer to "how does the fix preserve the riser's open top": under
  padding, `_static_vent_mask` keeps working unchanged because the pad above the riser top
  is still `EMPTY_CELL`.
* **STOPGAP**: also sterilises the wrap, but **riser-top vents drop from 16 to 0** — the
  riser's top plane becomes obstacle, so it is never GAS, so `find_entrapped_gas` loses the
  riser's vent and the riser's gas column will read as entrapped. Confirms the stopgap is a
  stopgap.

### 4d. What this changes for Full_Gating_Test — it will NOT stay unchanged

Audited with `Test/session_diag_2026-09-02/mech2_riser_seed/face_audit_any.py` against
`Test/session_diag_2026-09-01/r3_fgt_ledger.csv.types.npy` `(25,10,24)` and
`r3_fgt20_ledger.csv.types.npy` `(21,8,21)`. **Full_Gating_Test has metal-to-metal wraps on
all three axes**, worse per-cell than Test 2:

| grid | wrap-x | wrap-y | wrap-z |
|---|---|---|---|
| `(25,10,24)` | 59 (RUNNER↔CAST 20, **SPRUE↔RISER 27**, SPRUE↔CAST 12) | 130 (CAST↔CAST) | 21 (**RUNNER@z=0 ↔ SPRUE@z=23**) |
| `(21,8,21)`  | 60 (RUNNER↔CAST 16, **SPRUE↔RISER 32**, SPRUE↔CAST 12) | 88 (CAST↔CAST) | 16 (**RUNNER@z=0 ↔ SPRUE@z=23**) |

Concretely, on today's code:

* **SPRUE@x=0 ↔ RISER@x=24 (27-32 cells)** — the sprue, wet within the first seconds,
  streams and exchanges mass **directly into the riser**. Full_Gating_Test's riser is being
  fed through a wormhole, not through the casting.
* **RUNNER@z=0 ↔ SPRUE@z=23 (16-21 cells)** — the runner floor wraps onto the sprue's top
  plane, which is the *inlet*. That is a recirculation loop back into the pour boundary.
* **CAST@y=0 ↔ CAST@y=7/9 (88-130 cells)** — the casting spans the full `y` extent, so it
  is an infinite periodic slab in `y`: **the casting has no `y` walls at all.**

So: the answer to "does the fix keep Full_Gating_Test unchanged" is **no, and it should
not**. The fix adds two real walls to that casting, removes a sprue→riser shortcut and
removes a runner→inlet recirculation. Expect its fill time, fill fraction, velocity field
and riser fill order all to move — and note that the constant `SPRUE↔RISER` feed means the
*current* Full_Gating_Test numbers are not a trustworthy baseline. Any threshold or
benchmark calibrated against it (the round-2 inlet-drive work cites
`Test_Model/Full_Gating_Test` measurements repeatedly, e.g.
`core/filling_stage.py:1614-1625`) must be re-baselined after the fix, and
`Test_Model/Filling_Test` should be face-audited the same way with `face_audit_any.py`
before it is trusted either.

The one thing that genuinely does *not* change for Full_Gating_Test: its sprue top is its
highest feature and reaches `z = 23`, so under Step 1 + Step 2 that mouth keeps a real
atmosphere headspace and the inlet keeps behaving as a free surface — the pour boundary
itself is not altered.

---

## 5. FILES

Probes and logs, all under
`/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim/Test/session_diag_2026-09-02/mech2_riser_seed/`:

| file | what it does |
|---|---|
| `face_audit.py` | static six-face audit + per-part extents of the Test 2 solve grid |
| `wrap_pairs.py` | opposite-face metal↔metal wrap pairing, incl. the 298 CAST@z=0 ↔ RISER@z=93 columns |
| `face_audit_any.py` | the same audit for any saved `*.types.npy` (used for Full_Gating_Test) |
| `init_seed_audit.py` | initial-state / inlet-halo contamination check; 360 promotable riser cells |
| `wrap_fixture.py` | minimal both-paths proof + the two controls (~30 s) |
| `runaway_fixture.py` | 4,000-step growth of the phantom pool vs. both controls (~2 min) |
| `fix_cost.py` | measured cost of the padding fix and of the stopgap |
| `fix_validation.py` | Test 2 miniature: BROKEN vs PADDED vs STOPGAP, both paths |
| `reach_audit.py` | proves the 298 wrap columns are reachable metal, one connected component |
| `riser_seed_probe.py` | full instrumented Test 2 run, per-sub-operation attribution |
| `riser_seed.log` | its log |
| `REPORT.md` | this report |

No tracked file was modified. `data/mesh_runs.csv` and `web/simulations.db` remain the only
dirty tracked files, untouched by this work.
