# MECHANISM 1 — filter-outlet mass overload: verified diagnosis

Branch `refactor/pouring-seams`, HEAD `f1daf97`. No tracked file was modified.
Model: `~/Downloads/Test 2`, WCB, 4.3 mm grid (coarsening x1), solve grid
(135,135,94), filter z 14-32.

---

## 0. VERDICT (summary)

**Mechanism 1 is VERIFIED, but its mechanism is NOT the one the lead proposed.**

* The Koerner mass exchange **conserves mass exactly**. Measured on the real
  assembly: whole-grid `sum(mass_out - mass_in)` from `free_surface_kernel` is
  **1e-16 to 4e-15** against an `sum|dm|` of **16-29** per step, i.e. ~1e-16
  relative, every sample. **No mass is created, pairwise or globally.** The
  straddling-pair antisymmetry hypothesis is **FALSE** — the weight is
  provably and measurably symmetric under the pair swap in *both* step paths.
* What actually happens is a **positive-feedback amplifier**. For any pair with
  a porous cell on either side, the code overrides Koerner Table 1 and weights
  the exchange by `0.5*(eps_A + eps_B)` with **eps unclamped**
  (`physics/fluid_flow.py:1603-1608`, `physics/filling.py:722-727`). Because
  the weight *scales the exchange amplitude* and is itself a function of the
  cell's own `eps`, any `eps` error amplifies its own exchanges. Filter cells
  are FLUID, and **nothing in the solver can ever remove mass from a FLUID
  cell** (both transition scans select on `FLAG_INTERFACE` only), so the error
  has no restoring force at all. The over-full population's envelope grows
  exponentially until the donors go negative and the run detonates. Measured:
  **786 of 2,009 FLUID samples of the filter's bottom plane are over full,
  against 1 of 754 INTERFACE samples of the same cells.**
* This is precisely the failure mode
  `tests/test_filling_stability.py::FluidMassTracksDensityTests` (line 453) was
  written to prevent — "a random walk whose step size was proportional to the
  error", which reached +-70,000 per cell before Table 1 was adopted. The
  porous branch **re-enables it inside and around every filter.**
* Worse: the porous branch **does not even achieve its stated purpose.** Its
  justification (`physics/filling.py:699-705`) is that weight-1 makes a porous
  cell's mass track `rho` instead of `rho*porosity`. But when `eps ~ 1` the
  eps-average *is* 1, so the branch behaves identically to weight 1 and
  prevents nothing. It only starts to differ once `eps` has already drifted —
  and then it amplifies the drift instead of correcting it. A negative feedback
  was intended; a positive one was written.
* **The fix is a deletion.** Removing the porous override — six lines in each
  path, restoring plain Koerner Table 1 — holds the error dead flat where HEAD
  grows it 28-30 %, conserves mass to 2e-16, and is bit-identical off a filter
  over 400 free-surface steps. Verified in `verify_fix.py` and
  `weight_choice.py`. Two plausible-looking alternatives (weighting by
  porosity; re-pinning through `_redistribute`) were implemented and **measured
  to be worse or inert** — details in section 3.

---

## 1. CODE AUDIT

### 1(a) A porous FILTER cell A paired with a non-porous FLUID/INTERFACE cell B

**Kernel** — `physics/fluid_flow.py:1586-1628`, inside `free_surface_kernel`:

```
1586  por_here = porosity[x, y, z]
1587  denom    = float64(rho[x, y, z]) * por_here
1588  eps_here = mass_in[x, y, z] / denom if denom > 0 else 0.0      # UNCLAMPED
1589  eps_here_c = clip(eps_here, 0, 1)
...
1602  por_nb = porosity[sx, sy, sz]
1603  if por_here < 1.0 or por_nb < 1.0:              # <-- the porous override
1604      nden   = float64(rho[nb]) * por_nb
1605      eps_nb = mass_in[nb] / nden if nden > 0 else 0.0           # UNCLAMPED
1608      weight = 0.5 * (eps_here + eps_nb)
1609  elif flag == FLUID or nflag == FLUID:
1610      weight = 1.0
1611  else:
1612-1620  weight = 0.5 * (clip(eps_here,0,1) + clip(eps_nb,0,1))    # CLAMPED
1625  delta += (float64(f_out[ib,x]) - float64(post[i,x])) * weight
```

From A's side: `por_here = 0.85`, `por_nb = 1.0`, so line 1603 is true and
`w_A = 0.5*(eps_A + eps_B)` with **both terms unclamped**, `eps_X =
mass_X / (rho_X * por_X)`.

From B's side the same code runs with the roles swapped: `por_here = 1.0`,
`por_nb = 0.85`, line 1603 is **still true** (it is an `or` over the pair), and
`w_B = 0.5*(eps_B + eps_A)`.

**So `w_A == w_B` exactly.** The `or` in the branch condition is symmetric, and
`0.5*(a+b)` is the same float as `0.5*(b+a)`. Combined with the population
identities the kernel docstring spells out (lines 1487-1503) — `f_out[ibar,x]
== post[ibar,nb]` and `f_out[i,nb] == post[i,x]`, bit for bit, because
streaming copies and the gas reconstruction only ever rewrites directions the
exchange skips — the two contributions are **exact negations**:

```
A: (post[ibar, nb] - post[i, x]) * w
B: (post[i, x] - post[ibar, nb]) * w
```

**dm_AB = -dm_BA is guaranteed. There is no pairwise mass creation at the
filter boundary.** (Measured confirmation in section 2.)

### 1(b) porous-porous and plain-plain pairs

* **porous-porous**: line 1603 true from both sides, `w = 0.5*(eps_A+eps_B)`,
  unclamped, symmetric. When both cells are full, `eps = 1` and `w = 1` — i.e.
  identical to the plain FLUID branch, which is the point made in section 0.
* **plain FLUID-anything**: line 1609, `w = 1.0`. Symmetric trivially. This is
  Koerner Table 1 and is what makes `dm == drho`, so mass tracks density and
  cannot drift.
* **plain INTERFACE-INTERFACE**: lines 1611-1620, `w = 0.5*(clip(eps_A) +
  clip(eps_B))`, **clamped to [0,1] on both sides**. Symmetric.

The clamping asymmetry hypothesis (d) is therefore **FALSE**: within any single
branch both sides are treated identically. The real defect is that the *porous*
branch clamps **neither** side, so the weight is unbounded above.

### 1(c) `eps` for a porous cell, and whether it is used consistently

`eps = mass / (rho * porosity)` — `fluid_fraction`, `physics/filling.py:466-479`.
Consistency, checked call site by call site:

| consumer | file:line | capacity used | verdict |
|---|---|---|---|
| exchange weight, kernel | `fluid_flow.py:1587`, `:1604` | `rho * porosity` | consistent, but **unclamped** |
| exchange weight, NumPy | `filling.py:693-694`, `:725` | `rho * porosity` | consistent, but **unclamped** |
| fill/empty threshold, kernel | `fluid_flow.py:1931` `full = rho*porosity` | `rho * porosity` | **correct** |
| fill/empty threshold, NumPy | `filling.py:815` `full_mass = rho*porosity` | `rho * porosity` | **correct** |
| `heal_submerged_interface` | `filling.py:1027` | `rho * porosity` | **correct** |
| `_redistribute` recipients | `filling.py:1451` | *no capacity check at all* | see below |
| stage inlet top-up `room` | `core/filling_stage.py:2793-2795` | `rho * _porosity` | **correct** |

So hypothesis (b) — a wrong capacity (`rho` vs `rho*porosity`) somewhere — is
**FALSE**. Every capacity expression in the codebase carries the porosity
factor. That was not the bug.

The load-bearing inconsistency is elsewhere: **`eps` is a free, unbounded
variable for a FLUID cell.**

* `transition_scan_kernel`, `fluid_flow.py:1927-1929`:
  `if flags[x,y,z] != _FLAG_INTERFACE: continue`
* `apply_transitions` NumPy body, `filling.py:803-806`:
  `interface = state.flags == FLAG_INTERFACE; to_fluid = interface & (eps > ...)`
* `heal_submerged_interface`, `filling.py:997`: `interface = flags == FLAG_INTERFACE`

**Both step paths only ever inspect, cap or redistribute the mass of INTERFACE
cells.** A FLUID cell's `mass` is written by exactly one operation — the
exchange — and by nothing else, ever. Once a FLUID cell is over-full there is
no code path that can bring it back. `_redistribute` cannot: it is only ever
called with an `excess` array that is nonzero on converting INTERFACE cells
(`filling.py:816-817`, `fluid_flow.py:1933-1943`), and it only ever *adds* to
its recipients (`filling.py:1462-1467`), with **no capacity check on the
recipient** — so it can push a receiving interface cell over full too, though
that cell is at least re-examined next step.

This also resolves hypothesis (a): `_redistribute`'s "strand it back on the
donor when no interface neighbour exists" path (`filling.py:1461`,
`redistribute_share_kernel`) is *not* what strands mass at the filter outlet,
because `_redistribute` is never handed an over-full FLUID cell in the first
place. **Hypothesis (a) is FALSE as stated**; the correct statement is stronger
— the mass is not "stranded by `_redistribute`", it is *structurally
unreachable* by every corrective operation the solver has.

### 1(d) NumPy path vs kernel

`advance_mass`, `physics/filling.py:706-730`:

```
706  porous = state.porous.porosity_f64 < 1.0
717  weight = np.where(fluid | roll(fluid), 1.0, 0.5*(eps + roll(eps)))   # eps CLAMPED
722  if any_porous:
723      weight = np.where(porous | roll(porous),
725                        0.5*(eps_raw + roll(eps_raw)),                 # UNCLAMPED
726                        weight)
```

Identical rule, identical precedence (the porous `np.where` overrides the
Table-1 result, exactly as the kernel tests `por < 1` first), identical
clamping choice (`eps` clipped at line 694, `eps_raw` not), and identical
symmetry (`porous | roll(porous)` and `0.5*(a + roll(a))` are both symmetric
under the pair swap). **The two paths agree, so the defect is in both and any
fix must be applied to both.**

Measured parity, `parity_porous_weight.py` — same seeded state, one step down
each path, on a lattice with a porous band, a straddling INTERFACE cell, and a
deliberately over-full porous FLUID region:

```
seed_eps=1.0   max|dmass|=6.542e-09  max|drho|=0  max|df|=0  total numba=121.550000644 numpy=121.550000644
seed_eps=1.05  max|dmass|=6.908e-09  max|drho|=0  max|df|=0  total numba=122.697500676 numpy=122.697500676
seed_eps=2.0   max|dmass|=1.386e-08  max|drho|=0  max|df|=0  total numba=144.500001287 numpy=144.500001287
seed_eps=7.8   max|dmass|=5.633e-08  max|drho|=0  max|df|=0  total numba=277.610005021 numpy=277.610005021
```

Populations and density are **bit-identical**; only `mass` differs, by the
float32 accumulation order of the weighted direction sum, and the total agrees
to 12 digits. The residual grows with the seed (6.5e-9 at eps=1 -> 5.6e-8 at
eps=7.8) — a direct, if incidental, demonstration that the weight scales the
exchange amplitude.

### 1(e) Where the seed comes from (and why it is not itself a bug)

`_collide_stream_numba` (`physics/filling.py:114`) assigns `state.rho` the
density computed from `state.f` at the **start** of the step, then the exchange
produces `mass(t+1)`. So every consumer of `eps` sees

```
eps = mass(t+1) / (rho(t) * porosity)
```

At weight 1 the direction sum telescopes to `mass(t+1) = mass(t) + rho(t+1) -
rho(t)`, so `mass(t+1) == rho(t+1)*porosity` exactly (given `porosity == 1`) —
but it is *divided by* `rho(t)`. The reported `eps` is therefore high by
`Δrho/rho`, a one-step reporting lag of order 1-2 % wherever density is
changing. At weight 1 that lag is inert: it never feeds back, and the next step
re-evaluates it from scratch.

Measured on Test 2, before the metal reaches the filter (steps 5,000-7,000,
`fml2_global.csv`): **592-806 FLUID cells sit above eps 1.001 with eps_max
1.008-1.024 and eps_min ~ -0.001, and those numbers do not grow.** That is the
benign baseline — and it is exactly the 1-2 % seed the porous branch then
amplifies.

`create_state` (`physics/filling.py:402`, `:407`) seeds
`mass[fluid] = REFERENCE_DENSITY` and `mass[interface] = 0.5*REFERENCE_DENSITY`
with **no porosity factor**, so a cell that *starts* FLUID inside a filter
begins at `eps = 1/porosity = 1.176`. This is a genuine secondary
inconsistency (worth a one-line fix) but it does **not** drive Test 2, which
starts empty: filter cells there arrive via `apply_transitions`, which snaps
them to `rho*porosity` correctly.

---

## 2. INSTRUMENTED SIMULATION

Probe: `fml2.py` (derived from the never-run
`Test/session_diag_2026-09-01/filter_mass_ledger.py`; its replicated weight
rule was re-verified line by line against the f1daf97 kernel and was **already
correct** — no staleness to fix. The changes made were instrumentation
coverage, listed in the module docstring). Run: `START=5000 STOP=13000
LOG_EVERY=50 AUDIT_EVERY=250 CASTSIM_THREAD_BUDGET=8`, 8,000 instrumented steps
on the real Test 2 assembly. Log `fml2.log`, ledgers `fml2_global.csv`
(161 samples) and `fml2_cells.csv`.

### 2(a) Is the global sum of exchange dm nonzero? — NO. Emphatically.

Whole-grid `sum(mass_out - mass_in)` straight out of `free_surface_kernel`,
against the same step's `sum|dm|`:

| step | `sum dm` | `sum |dm|` | relative |
|---|---|---|---|
| 5,000 | +3.30e-15 | 15.9 | 2e-16 |
| 7,000 | -2.64e-16 | 20.3 | 1e-17 |
| 9,000 | -1.28e-15 | 23.7 | 5e-17 |
| 11,000 | +2.28e-15 | 27.6 | 8e-17 |
| 13,000 | +1.44e-15 | 30.5 | 5e-17 |

Every one of the 161 samples is between -4.9e-15 and +6.5e-15. **The exchange
is conservative to float64 accumulation round-off. No mass is created.**

The other operations, summed over all non-obstacle cells:

* `apply_transitions` (incl. `_redistribute`): **±1.8e-12** on a total of
  5,810 -> 15,000+ lattice mass units. Round-off.
* `heal_submerged_interface`: **0.0** at every sample but one (+1.8e-12).
* `reset_detached_interface_cells`: **0.0** at every sample.
* everything outside `step()` — the stage's inlet pin and metered top-up:
  **+1.108 to +1.126 per step**, flat across the whole run. That is the honest
  delivery (20.6 lb/s) and it is the *only* source term in the domain.

### 2(b) Is it localised to straddling pairs? — The pairs are EXACTLY antisymmetric.

The whole-grid pair audit walks every wet cell in or adjacent to a filter,
every wet neighbour, recomputes the weight independently from each side using
the kernel's own formula, and forms `dm_AB + dm_BA`:

```
step   8,000   pairs =  5,256   max relative residual = 0.000e+00   straddling = 0.000e+00
step   9,000   pairs = 15,642   max relative residual = 0.000e+00   straddling = 0.000e+00
step  11,000   pairs = 28,784   max relative residual = 0.000e+00   straddling = 0.000e+00
step  13,000   pairs = 36,550   max relative residual = 0.000e+00   straddling = 0.000e+00
```

**Zero. Not small — bit-for-bit zero, over 36,550 pairs, including every pair
straddling the porous boundary.** The mixed porous/non-porous exchange is
perfectly antisymmetric, exactly as the audit in 1(a) predicted.

**The UNTESTED LEAD from the last session is FALSE and is now closed.**

### 2(c) What IS localised is the fill fraction itself

`eps_max` over all wet cells vs `eps_max` restricted to filter cells and their
neighbours, plus the count of over-full cells:

| step | `eps_max` (whole domain) | `eps_max` (filter + nbrs) | `eps_min` | over-full FLUID | over-full near filter |
|---|---|---|---|---|---|
| 5,000 | 1.008 | — (dry) | -0.0010 | 432 | 0 |
| 6,500 | 1.022 | — (dry) | -0.0011 | 806 | 0 |
| 7,500 | 1.020 | 0.540 (wetting) | -0.0010 | 663 | 0 |
| **7,650** | **1.046** | **1.046** | -0.0010 | 658 | **6** |
| 8,000 | 1.346 | 1.346 | -0.036 | 707 | 69 |
| 8,500 | 1.468 | 1.468 | -0.058 | 1,045 | 151 |
| 9,000 | 1.916 | 1.916 | -0.110 | 902 | 195 |
| 10,000 | 1.638 | 1.638 | -0.125 | 1,359 | 294 |
| 11,000 | **3.711** | **3.711** | **-1.173** | 1,198 | 344 |
| 12,000 | 1.972 | 1.972 | -0.108 | 1,324 | 358 |
| 13,000 | 2.691 | 2.691 | -0.023 | 1,388 | **399** |

Three things to read off this:

1. **Before the filter wets, `eps_max` is 1.006-1.022 and does not grow.**
   Several hundred plain FLUID cells sit ~1 % over full — the one-step `rho`
   lag from 1(e). At weight 1 it never feeds back, so it plateaus.
2. **From step 7,650 — the first step any filter-adjacent cell crosses 1 —
   `eps_max == eps_max_por` at every single subsequent sample.** In a
   1.7-million-cell domain the global maximum fill fraction is *always* a
   filter cell or a filter neighbour, for the rest of the run. The pathology is
   perfectly localised to the porous zone.
3. **The donors go negative in lockstep.** Because the exchange conserves mass
   exactly, every unit of phantom fill at the outlet is a unit of *missing*
   mass elsewhere. `eps_min` tracks `eps_max`: -0.001 baseline -> -0.036 ->
   -0.110 -> -0.125 -> **-1.173 at step 11,000, the same sample `eps_max`
   spikes to 3.711.** Negative-mass wet cells are the precursor the guard
   eventually catches at step 16,600, ~5,000 steps after the cause.

And the throughput collapses while this happens: net exchange mass entering the
porous neighbourhood per step falls **0.47 (step 8,500) -> 0.11-0.13 (steps
12,500-13,000)** while the stage keeps delivering a flat 1.12/step. The fill
stalls at 8.9 % because metal that should be passing the filter is being
converted into over-full bookkeeping instead of transported.

### 2(d) Which operation delivers the mass, and does anything take it out?

Two per-cell traces. `fml2` armed on the first cell to cross the threshold —
(114,82,18), a RUNNER cell with `porosity = 1.0`, caught by the branch because
its *neighbour* is porous — and followed it from the onset. `fml3` (`START=9000`,
`EPS_WATCH=1.3`, watch restricted to cells that are themselves porous) armed on
**39 cells, all FILTER-typed, z 14-20** — exactly the reported population — and
followed them for 3,500 steps (2,769 samples).

**Nothing outside the exchange ever writes these cells' mass, except when they
are INTERFACE.** Over `fml3`'s 2,769 samples:

```
              dm_transitions nonzero   dm_heal nonzero   dm_external nonzero
  FLUID       7 / 2009  (0.3 %)        2 / 2009          0 / 2009
  INTERFACE  93 /  754  (12.3 %)       4 /  754          0 /  754
```

`dm_external` is **0.000e+00 in all 2,769 samples** — **hypothesis (c), the
inlet pin or top-up writing mass at z=14, is measured FALSE.** `dm_reset` is
identically zero too.

And here is the mechanism in a single pair of numbers. Of the 39 watched filter
cells:

* **786 of 2,009 FLUID samples had `eps > 1.05`.**
* **1 of 754 INTERFACE samples had `eps > 1.05` — and it was capped by
  `apply_transitions` on that very step.**

The over-full state exists **only** in FLUID cells, and it exists there because
FLUID is the one flag with no cap. The same cell, over-full at `eps = 1.4` as
FLUID, is repaired within one step of being demoted to INTERFACE. That is the
structural hole from audit 1(c), measured.

**The exchange does not deliver the physically justified mass.** `dm_w1` is the
direction sum recomputed at weight 1 (mass tracks `rho`) and `dm_wpor` the same
sum times `porosity` (mass tracks `rho*porosity` — the correct target for a
porous cell). Summed over the 2,009 FLUID samples:

```
  sum dm_exchange     = -0.3449      what the code did
  sum dm_wpor         = -0.0543      what it should have done
  sum dm_w1           = -0.0639
  error               =  0.2907      6.4x the correct magnitude, wrong by 0.9 %
                                     of the 39 cells' total capacity (33.15)
```

**The effective weight — `dm_exchange / dm_w1`, which the scheme requires to be
1 — is neither 1 nor bounded.** Restricted to well-conditioned samples
(`|dm_w1|` above its median, so the ratio is meaningful):

```
  inflow  (dm_w1 > 0): n=702  median 0.823  p90 1.114  max  +10.65
  outflow (dm_w1 < 0): n=682  median 0.898  p10 0.458  min  -12.98
  aggregate  sum|dm_exchange| / sum|dm_w1| = 0.901
```

Excursions to **10x and -13x** the correct amplitude, and the sign inverts —
the cell gains mass on a step it should have lost mass — in **102 of 1,363
outflow samples (7.5 %)**.

At the *onset*, before the population settled, the bias was cleanly one-signed.
The `fml2` trace of the first cell to go over-full, `eps` ratcheting
1.004 -> 1.233 over 450 steps with `rho` pinned flat at 0.998-1.002:

```
step   eps_pre    rho          dm_exchange    dm_w1          effective weight
7700   1.07937    1.0007733    +1.779503e-03  +9.943731e-04  1.790
7750   1.00396    0.99937731   +3.224826e-03  +2.542643e-03  1.268
7800   1.04011    0.99957931   +1.297184e-03  +5.547348e-04  2.338
7850   1.07528    0.99841434   +1.112103e-03  +4.579760e-04  2.428
7900   1.10690    1.0012023    +2.446673e-03  +1.829546e-03  1.337
7950   1.13697    0.99890953   -7.192024e-04  -1.268709e-03  0.567
8000   1.16298    1.0010707    +1.792776e-04  -3.303550e-04  -0.543   <-- sign inverted
8050   1.18574    1.0002342    -3.586719e-04  -7.932615e-04  0.452
8100   1.20482    1.0006615    -7.068756e-04  -1.064816e-03  0.664
8150   1.22047    1.0014564    -1.837554e-04  -4.712120e-04  0.390
8200   1.23291    0.99990809   +2.124844e-03  +1.943763e-03  1.093
```

Inflow amplified (1.09-2.43), outflow attenuated (0.39-0.66), one sign
inversion: a rectifier. The direction of that bias is geometric — the weight is
`0.5*(eps_here + eps_nb)`, so directions bringing mass in from the *fuller*
upstream side (the filter interior, `eps >= 1`) are weighted higher than
directions carrying mass out to the *emptier* downstream side (the partly filled
runner/gate, `eps <= 1`). At a full filter's outlet face facing a partly filled
runner the bias therefore points **into** the cell, which is exactly and only
where the reported population lives.

**Important correction to the reported picture.** The 30-60 over-full cells are
**not a fixed set climbing monotonically.** Individual cells churn: of the 39
watched, `eps` *mean* falls 1.41 -> 0.82 over the 3,500 steps and the count
above 1 drops 39 -> ~10, because cells keep flipping between FLUID and
INTERFACE (2,009 FLUID vs 754 INTERFACE samples on the same 39 cells) and get
repaired whenever they land on INTERFACE. What grows is the **population
envelope**, not any one cell: `n_gt1_por` 6 -> 399 and `eps_max_por` 1.046 ->
2.69 -> 7.76 (section 2(c)). The steady count with a rising maximum in
`r3_eps2.log` is exactly this signature. The churn is why the run survives to
step 16,500 rather than diverging in hundreds of steps — partial repair by
accidental demotion is the only brake in the system, and it is not enough.

### 2(e) What SHOULD happen at a choked filter outlet, vs what the code does

**Physically.** Metal reaches the filter, the Darcy-Forchheimer drag limits
throughput, and metal backs up. The back-pressure appears as a rise in
**density** (pressure is `rho*c_s^2` in LBM): a 0.26 m head of steel is ~18 kPa,
which at this unit mapping is a **few percent** on `rho`. The flow slows to
whatever the filter passes; the free surface upstream rises; delivery falls
below the commanded 20.6 lb/s. Throughout, a FLUID cell's fill fraction is
identically **1** — a full cell holds exactly its pore volume of metal, no more.
`eps` is a *geometric* quantity bounded in [0,1] by construction; all of the
"backing up" belongs in `rho`, none of it in `eps`.

**What the code does.** Measured over `fml3`'s 2,769 samples of the choked
filter outlet: `rho` **median 1.000338, mean 0.998243**, and only **2.64 % of
samples exceed 1.01** (max 1.0767). A 0.26 m head of steel choked by a ceramic
filter should hold a *persistent* few-percent density rise at the outlet; what
is there is a median 0.03 % above reference. **The back-pressure essentially
never forms.** Instead the solver stores it as **phantom mass**:
`eps` climbs to 1.9, then 3.7, then 7.76, meaning a cell is asked to hold up to
7.76x the metal that physically fits in it. Because the exchange conserves mass
exactly, that phantom fill is *stolen* from real cells, which go to `eps = -9.5`.
The choke never expresses itself as pressure, so the flow never physically
slows for the right reason; it stalls because the bookkeeping has diverged. The
run then dies ~5,000 steps later when a negative-mass cell finally reaches the
guard.

### 2(f) Minimal counterexample, NumPy path

`witness_porous_weight.py`. Fixture: the closed all-FLUID box
`tests/test_filling_stability.py::FluidMassTracksDensityTests` uses to assert
`|mass - rho*porosity|` stays at ~1e-4 forever, with **one** change — a
3-cell-deep porosity-0.85 band across the middle. There is no GAS and no
INTERFACE cell anywhere, so `apply_transitions`, `heal_submerged_interface` and
`_redistribute` are all no-ops by construction and **the only thing that
touches `state.mass` is `advance_mass`.** 126 FLUID cells, NumPy path
(`use_numba=False`).

`max|eps - 1|` over FLUID cells, and total mass:

| variant | step 1 | step 4,000 | total mass drift |
|---|---|---|---|
| `plain` — porosity 1.0 everywhere (control) | 4.75e-05 | **1.02e-04** (flat) | 2e-16 |
| `por` — porosity band, drag off | 1.7648e-01 | **1.9892e-01** (rising, monotone) | 0 |
| `por_fix` — porosity band, Koerner Table 1 restored | 1.7647e-01 | **1.7647e-01** (dead flat) | 8e-16 |
| `por_drag` — porosity band + real Guo-Zhao drag | 1.7648e-01 | **1.7737e-01** (rising) | 1e-16 |
| `por_fixd` — same + Table 1 restored | 1.7647e-01 | **1.7647e-01** (dead flat) | 9e-16 |

Then, seeding the band's `eps` explicitly to separate *self-starting* from
*amplifying* (6,000 steps):

| variant | seed `eps` | step 1 | step 6,000 |
|---|---|---|---|
| `por_e1` | 1.00 | 1.17e-10 | **5.4e-05** — no growth |
| `por_e1_d` (+ drag) | 1.00 | 3.08e-07 | **6.3e-05** — no growth |
| `por_e105` | 1.05 | 5.0000e-02 | **6.2908e-02** — +25.8 %, monotone |
| `por_e105f` (Table 1) | 1.05 | 5.0000e-02 | **4.9994e-02** — dead flat |

**Conclusions, all measured, total mass conserved to 1e-15 in every variant:**

1. The porous weight branch **does not create mass** — confirming 2(a) on an
   independent fixture.
2. It **does not self-start**: seeded at exactly `eps = 1` it stays there for
   6,000 steps. It is a pure **error amplifier**.
3. It amplifies **exponentially with a rate independent of the error size** —
   the signature of linear positive feedback. 0.17647 -> 0.19892 over 4,000
   steps gives 2.99e-5/step; 0.05000 -> 0.06291 over 6,000 steps gives
   3.83e-5/step. Same rate, 3.5x different seed.
4. **Restoring Koerner Table 1 removes the amplification exactly.** In every
   patched variant the error neither grows nor decays — it is held at whatever
   it was seeded with, to 4 significant figures, for 6,000 steps.

Rate cross-check against Test 2: `eps` 1.9 -> 7.76 over steps 11,000-16,250 is
`ln(4.084)/5250` = **2.68e-4/step**, ~8x the fixture's rate. Expected: the
growth rate scales with the local exchange amplitude, and the fixture is a
near-static closed box while Test 2 has real pour-velocity metal crossing the
filter. Extrapolating the Test-2 rate backwards from `eps = 1.9` at step 11,000
reaches `eps ~ 1.01` at step ~8,600 — and the probe measured the first
filter-adjacent crossing of `eps = 1` at **step 7,650**. Consistent.

Also incidentally confirmed: `create_state` starts a pre-filled porous cell at
`eps = 1/0.85 = 1.1765` (the 0.176 offset present at step 1 in every porous
variant), the secondary bug noted in 1(e).

---

## 3. VERDICT

### Mechanism 1: VERIFIED — with a corrected mechanism

The reported observable is real and reproduced, in exactly the reported place:
an over-full FLUID population at the filter's bottom plane whose maximum `eps`
climbs past 1 to 7.76 while its count holds in the tens
(`n_gt1_por` 6 at step 7,650 -> 399 at step 13,000; `eps_max == eps_max_por` at
every sample thereafter; the 39 watched cells are 100 % FILTER-typed at z 14-20).
One correction to the reported picture: the climb is in the population
*envelope*, not within a fixed set of cells — individual cells churn in and out
as they flip between FLUID and INTERFACE (section 2(d)). That detail is what
explains the otherwise puzzling steady 30-60 count with a rising maximum.

The **cause is not mass creation**. It is an **unbounded positive-feedback loop
in the porous pair weight, on a quantity that has no restoring force.**

Causal chain, each link measured or cited:

1. **Seed.** `state.rho` is the *start-of-step* density (`physics/filling.py:114`)
   while `mass` is the end-of-step value, so every FLUID cell's reported `eps`
   is high by `Δrho/rho`, ~1-2 %. Measured: 432-806 plain FLUID cells above
   1.001 with `eps_max` 1.006-1.022, **not growing** (steps 5,000-7,000). At
   weight 1 this lag is inert.
2. **Amplifier.** The moment a filter cell wets, every pair touching it is
   diverted to `weight = 0.5*(eps_A + eps_B)` with **eps unclamped**
   (`physics/fluid_flow.py:1603-1608`, `physics/filling.py:722-727`). The weight
   scales the exchange amplitude and is a function of the cell's own `eps`, so
   the 1-2 % seed amplifies its own exchanges.
3. **Rectification.** The exchange amplitude is corrupted by an eps-dependent
   factor that should be 1 and is neither 1 nor bounded: measured excursions to
   **+10.7 and -13.0**, with the sign inverted (mass gained on a step the cell
   should have lost mass) in **102 of 1,363 outflow samples, 7.5 %**. At the
   onset the bias is cleanly one-signed by geometry — incoming directions come
   from the fuller filter interior (higher `eps`, higher weight), outgoing
   directions go to the emptier runner (lower `eps`, lower weight) — measured
   **1.09-2.43 on inflow vs 0.39-0.66 on outflow** while `eps` ratcheted
   1.004 -> 1.233 in 450 steps. Once the population is established the
   per-sample sign varies with the local `eps` gradient, but the aggregate FLUID
   mass budget is still wrong by **6.4x the correct magnitude** over 2,009
   samples.
4. **No restoring force on the flag that matters.** `transition_scan_kernel`
   (`physics/fluid_flow.py:1929`), `apply_transitions`
   (`physics/filling.py:803-806`) and `heal_submerged_interface`
   (`physics/filling.py:997`) all select on `FLAG_INTERFACE` only. **A FLUID
   cell's mass is written by the exchange and by nothing else, ever.** Measured
   on 39 filter cells over 3,500 steps: **786 of 2,009 FLUID samples were over
   full (`eps > 1.05`) against 1 of 754 INTERFACE samples**, and that one was
   capped the same step. `dm_external` and `dm_reset` are identically 0.0 in all
   2,769 samples.
5. **Divergence.** The *population envelope* diverges — `n_gt1_por` 6 -> 399 and
   `eps_max_por` 1.046 -> 2.69 -> 7.76 between steps 7,650 and 16,250 — at an
   exponential rate of 2.68e-4/step. Individual cells churn in and out of the
   over-full set as they flip between FLUID and INTERFACE, and accidental
   demotion to INTERFACE is the only repair mechanism in the system; it is why
   the run lasts 9,000 steps rather than a few hundred, and it is not enough.
6. **Detonation.** Mass is conserved exactly, so the phantom fill is stolen:
   `eps_min` falls to -1.17 by step 11,000 and -9.5 later. A negative-mass wet
   cell is invisible to every corrective path (it is FLUID), and is what
   finally trips the guard at step 16,600 — ~5,000 steps and ~9 s of wall clock
   after the actual defect.

### Lines responsible

| what | where |
|---|---|
| **the defect (kernel)** | `physics/fluid_flow.py:1603-1608` — the `por_here < 1.0 or por_nb < 1.0` branch and its unclamped `0.5*(eps_here + eps_nb)` |
| **the defect (NumPy)** | `physics/filling.py:722-727` — the `if any_porous:` override, unclamped `0.5*(eps_raw + roll(eps_raw))` |
| **the reasoning that is wrong** | `physics/filling.py:699-705` and `physics/fluid_flow.py:1584-1585` — "Filter cells are few and their eps is bounded by the priming logic". Nothing bounds it; the priming logic (`porous.primed`) is used at `physics/filling.py:557-558` and nowhere else, to gate the capillary entry *density offset* — it has no bearing on `eps` whatsoever. |
| **why nothing corrects it** | `physics/fluid_flow.py:1929`, `physics/filling.py:803-806`, `physics/filling.py:997` — all three select on `FLAG_INTERFACE` |
| **secondary** | `physics/filling.py:402`, `:407` — `create_state` seeds mass without the porosity factor |

### Concrete fix direction (verified, `verify_fix.py` + `weight_choice.py`)

**THE FIX: delete the porous override. Use plain Koerner Table 1 everywhere.**

Kernel — `physics/fluid_flow.py:1603-1620` collapses to what lines 1609-1620
already say, with the `por_here < 1.0 or por_nb < 1.0` branch **removed**:

```python
if flag == _FLAG_FLUID or nflag == _FLAG_FLUID:
    weight = 1.0
else:
    nden = np.float64(rho[sx, sy, sz]) * por_nb
    eps_nb = mass_in[sx, sy, sz] / nden if nden > 0.0 else 0.0
    if eps_nb < 0.0:
        eps_nb = 0.0
    elif eps_nb > 1.0:
        eps_nb = 1.0
    weight = 0.5 * (eps_here_c + eps_nb)
```

NumPy — `physics/filling.py:722-727`, the whole `if any_porous:` block is
**deleted**, leaving lines 717-721 as they stand. `eps_raw` then has no consumer
except line 694's clip and can be inlined.

That is the entire change: **six lines out of the kernel, six out of the NumPy
path, nothing added.** Porous INTERFACE-INTERFACE pairs fall through to the
existing clamped branch, which is the second half of the fix — the weight is
then bounded in [0,1] instead of unbounded.

**Verified against all three constraints:**

* **Conserves mass to the ledger.** `weight = 1.0` and
  `0.5*(clip(eps_A) + clip(eps_B))` are both symmetric under the pair swap, so
  `dm_AB = -dm_BA` exactly — the property section 2(b) measures at
  0.000e+00 residual. Measured `dM/M0` = 2.3e-16 over 6,000 steps.
* **Reduces to current behaviour at porosity == 1.0 — by construction, not by
  luck.** HEAD already guards the override with `if any_porous:`
  (`physics/filling.py:722`), so on a lattice with no filter the deleted code
  never executed. The remaining path is byte-for-byte HEAD's. Measured anyway,
  on a lattice with a real free surface (45 FLUID, 9 INTERFACE, 90 GAS,
  transitions firing): **400/400 steps bit-identical on `state.mass`,
  `state.flags` and `state.f`** (`np.array_equal`). `Full_Gating_Test` is
  unaffected.
* **Removes the amplification.** Fully submerged filter seeded 5 % over full,
  6,000 steps, `rho` flat at 0.99997-1.00005 so nothing else can move `eps`:

  | weight rule | `max|eps-1|` after 6,000 steps (seeded 5.0000e-02) |
  |---|---|
  | **HEAD** (unclamped eps average) | **6.3832e-02**  (+27.7 %) |
  | **weight 1 — the fix** | **5.0007e-02**  (+0.014 %, flat) |
  | weight = `0.5*(p_A+p_B)` — see below | 6.5199e-02  (+30.4 %) |

  and on the free-surface fixture, 4,000 steps: HEAD 5.196e-02 -> 8.138e-02,
  the fix holds it. Section 2(f) shows the same on the isolated 126-cell box
  across ten variants.

**Do NOT "generalise" the weight to the porosity.** The obvious-looking repair —
weight the FLUID branch by `0.5*(por_A + por_B)` so that `dm = p*drho` and mass
tracks `rho*porosity` instead of `rho` — was implemented and measured, and it is
**worse than HEAD (6.52e-02 vs 6.38e-02).** The reason is worth recording
because it is the whole reason Table 1 works: the direction sum only telescopes
into `drho` when the weight is **uniform across a cell's directions.** A
porosity-dependent weight takes three different values at the filter's face
(0.85 inside, 0.925 straddling, 1.0 outside), and the resulting
`sum_i D_i (w_i - w_bar)` residual is set by the size of the *individual
populations* `D_i ~ 1e-2`, not by `Δrho ~ 1e-5`. It injects three orders of
magnitude more error than the `(1-p)*Δrho` drift it was meant to remove. **Any
spatially non-uniform weight has this defect. The weight must be 1.**

**The residual, and why to accept it.** With weight 1 a porous FLUID cell's mass
tracks `rho`, not `rho*porosity`, so it drifts from capacity by `(1-p)*Δrho` per
step — the concern `physics/filling.py:699-705` raises. That concern is real but
tiny and, critically, **bounded**: the increment telescopes, so the total drift
is `(1-p)` times the *total* density excursion over the whole fill, i.e. ~0.15 x
a few percent = a few tenths of one percent of a cell's capacity. Measured:
+0.014 % over 6,000 steps. It does not amplify, because with an eps-free weight
there is no path from `eps` back into the exchange amplitude. Trading an
exponential divergence for a bounded 0.1 % offset is the correct trade, and it
is the trade Koerner's scheme makes.

**Explicitly NOT recommended: re-pinning porous FLUID cells through
`_redistribute`.** This was the natural way to remove even the bounded residual
— set `mass = rho*porosity` on porous FLUID cells and hand the difference to
`_redistribute`, guarded on `porosity < 1.0` so it is a no-op off a filter. It
was implemented and measured. On a fixture where a free surface passes through
the filter it works perfectly (`max|eps-1|` **exactly 0.0** at every sample,
mass conserved to 1e-15). But on a **fully submerged** filter — which is the
state Test 2's filter is in for most of the run — it does **nothing at all**
(6.38e-02, identical to HEAD to every digit), because `_redistribute` can only
donate to INTERFACE neighbours and, finding none, strands the correction back
onto the donor cell (`physics/filling.py:1461`,
`redistribute_share_kernel`). A repair that silently disables itself in the
case that matters is worse than no repair. Do not ship it as the fix; the
weight-1 change stands on its own.

**Also fix (secondary, one line).** `create_state` (`physics/filling.py:402`,
`:407`) seeds `mass[fluid] = REFERENCE_DENSITY` and
`mass[interface] = 0.5*REFERENCE_DENSITY` with no porosity factor, so a cell
that *starts* FLUID inside a filter starts at `eps = 1/porosity = 1.176`
(visible as the 0.176 offset at step 1 of every porous variant in 2(f)).
Multiply by `porous.porosity_f64`. Does not affect Test 2, which starts empty,
but does affect fixtures and any restart that pre-fills a filter.

**Also add (recommended independently).** The structural hole behind all of this
is that a FLUID cell's `eps` is unconstrained **and unobserved**: measured
**786 of 2,009 FLUID samples over full against 1 of 754 INTERFACE samples.**
Add `eps` outside roughly `[-0.1, 1.1]` on a FLUID cell to the precursor abort
57b07d7 introduced. It is nearly free — `transition_scan_kernel` already
computes `eps` for every cell it visits and already skips FLUID cells at line
1929, so it is one comparison in an existing loop — and it would have fired at
step ~7,900 instead of surfacing as a NaN at 16,493.

**Regression test.** `witness_porous_weight.py` in this directory is the test:
`FluidMassTracksDensityTests` with a porosity band. Assertion: `por_e105`'s
`max|eps-1|` must not grow over 6,000 steps. Fails on HEAD (0.0500 -> 0.0629),
passes with the fix (0.0500 -> 0.0500). `weight_choice.py` is the fully
submerged variant, which is the stricter one (HEAD 0.0638 vs fix 0.0500) and
the one that discriminates between the candidate weights.

### Alternative hypotheses: what was ruled out, and how

| hypothesis | verdict | how |
|---|---|---|
| **LEAD** — mixed porous/non-porous pairs not antisymmetric; `dm_AB != -dm_BA`; pairwise creation at the outlet face | **FALSE** | Audit 1(a): the branch condition is an `or` over the pair and `0.5*(a+b)` is swap-symmetric, so both sides compute the same float. Measured 2(b): whole-grid pair audit, residual **exactly 0.000e+00** over 36,550 pairs including every straddling pair, at 4 separate steps. Measured 2(a): global `sum dm` = 1e-16 relative for 8,000 consecutive steps. Measured 2(f): total mass conserved to 1e-15 in an isolated 126-cell fixture. |
| **(a)** — mass honestly arrives faster than the cell can shed it and `_redistribute` strands it | **FALSE as stated** | `_redistribute` is only ever called with an `excess` array that is nonzero exclusively on *converting INTERFACE* cells (`physics/filling.py:816-817`, `fluid_flow.py:1933-1943`), so it is never handed an over-full FLUID cell and its stranding path (`filling.py:1461`) never sees one. Measured: `dm_transitions` is 0.0 on the traced cell for every sample while it was FLUID. The true situation is stronger — the mass is not *stranded*, it is structurally unreachable by every corrective operation. |
| **(b)** — transition thresholds or capacity checks use the wrong capacity (`rho` vs `rho*porosity`) | **FALSE** | Every capacity expression in the codebase carries the porosity factor: `fluid_fraction` (`filling.py:476`), `transition_scan_kernel` (`fluid_flow.py:1931`), `apply_transitions` NumPy (`filling.py:815`), `heal_submerged_interface` (`filling.py:1027`), the stage's top-up `room` (`filling_stage.py:2793-2795`). Tabulated in 1(c). |
| **(c)** — the inlet top-up or heal writes mass at z=14 | **FALSE** | Audit: the top-up only targets `FLAG_INTERFACE` cells in `delivery_layers` and only into `room = max(rho*porosity - mass, 0)`, so it cannot overfill (`filling_stage.py:2786-2800`). Measured: `max|dm_external|` = **0.000e+00** and `max|dm_heal|` = **0.000e+00** over 107 samples; globally `g_heal` = 0.0 and `g_ext` = a flat +1.12/step of honest delivery. |
| **(d)** — an exchange weight computed with `eps` clamped on one side and not the other | **FALSE** | Audit 1(a)/1(d): within each branch both sides are treated identically — the porous branch clamps **neither** side, the interface-interface branch clamps **both**. The real defect is that the porous branch clamps neither, making the weight unbounded above, not that the two sides differ. |
| kernel/NumPy divergence on the porous weight | **NO** | `parity_porous_weight.py`, section 1(d): populations and density bit-identical, mass agrees to 6.5e-9 (eps=1) through 5.6e-8 (eps=7.8) — float32 accumulation order only — and totals agree to 12 digits. Both paths carry the same defect. |
| the gas reconstruction floor (`val < 0 -> 0`, `fluid_flow.py:1571-1573`) as the source | **NOT the source** | It only rewrites directions pointing at GAS neighbours, and the exchange explicitly skips those directions (`fluid_flow.py:1600-1601`), so it cannot enter the mass exchange. It also cannot act on a FLUID cell, which has no gas neighbours by invariant. It adds density, not mass, and a density rise *lowers* `eps`. |

### Scope note

This diagnosis covers Mechanism 1 only. The terminal event in the prior run —
809 over-full cells at step 16,493 concentrated in the **RISER** at z 86-90,
with the `overflow encountered in cast` at `physics/filling.py:1282`
(`_initialise_promoted_cells`) — is a different locus and is presumably
Mechanism 2. The two are coupled: Mechanism 1 manufactures the negative-mass
cells (`eps_min` -> -9.5) that the riser then promotes from, so fixing
Mechanism 1 should be expected to change, and may mask rather than fix,
Mechanism 2's symptom.

---

## Artefacts

All under `Test/session_diag_2026-09-02/mech1_filter_outlet/`:

| file | what |
|---|---|
| `fml2.py` | instrumented Test-2 probe (per-operation mass ledger, whole-grid pair antisymmetry audit, weight-1 counterfactual) |
| `fml2.log` | run log, steps 1-13,000 |
| `fml2_global.csv` | 161 whole-grid samples: `exch_sum`, per-operation global deltas, `eps` extremes, over-full counts, pair residuals |
| `fml2_cells.csv` | 107 per-cell samples with per-operation attribution and the `dm_w1` counterfactual |
| `fml3.py` | second pass, watch restricted to genuinely porous (FILTER) cells at the filter's bottom plane |
| `fml3.log`, `fml3_global.csv`, `fml3_cells.csv` | its outputs |
| `witness_porous_weight.py` | minimal 126-cell NumPy-path counterexample and fix validation (10 variants) |
| `witness_main.log`, `witness_seeded.log` | its outputs: the five base variants, and the `eps`-seeded ones (self-start vs amplify) |
| `parity_porous_weight.py` | kernel-vs-NumPy parity on the porous weight at `eps` up to 7.8 |
| `verify_fix.py` | fix validation: bit-identity off a filter over 400 free-surface steps, amplification removal, and the `_redistribute` re-pin's failure on a submerged filter |
| `weight_choice.py` | the discriminating experiment: HEAD vs weight-1 vs weight-porosity, +/- re-pin, on a fully submerged filter |
| `REPORT.md` | this file |

Side effect to be aware of: running the pipeline appends telemetry rows to
`data/mesh_runs.csv` and `web/simulations.db`, which were already dirty before
this session. **No tracked source file was modified** (`git diff --stat` shows
only those two data files). The test suite was not run and nothing was
committed.
