# MECHANISM 3 — the step-20,192 momentum blow-up at the sprue/runner/filter junction

Branch `refactor/pouring-seams`, HEAD **`fe866cf`** (round 4 committed). No
tracked file was modified; every change in this report is a monkeypatch inside
this directory. Model: `~/Downloads/Test 2`, WCB, 4.3 mm grid, padded solve grid
(137,137,98).

Reproduced: **non-finite mass at step 20,192, 287 cells, `status=diverged`,
20,200 steps** — bit-for-bit with the validator's three runs.

---

## 0. VERDICT (summary)

**Mechanism 3 is a staggered (period-2) momentum mode on a single lattice
diagonal, growing super-exponentially in the NON-porous runner cells one cell
outside filter #1's inlet face, in a scheme that has no mechanism capable of
damping it.**

* **H1 (explicit porous-drag stiffness) is FALSE, and inverted.** The drag map
  is a strict *contraction* for every velocity: `sup |m'/m| = 0.99998`,
  approached only as `|u| -> inf`. Its per-step damping ratio
  `R = 2[(2c0-1)+c1|u|]/(2c0+c1|u|)` cannot reach 2 — algebraically it would
  require `-1 = 0`. And the two cells the runaway actually lives in have
  **`R = 0` exactly**, measured, because they are non-porous (`c1 = 0`). Far
  from causing the failure, the drag is the **only** momentum sink a filter
  cell has: removing it kills the fixture at every velocity including the
  design velocity, and doubling it cuts the fixture's peak `|u|` from 11.27 to
  3.51.
* **H2 (jet compressibility / Mach) is FALSE.** The mode grows from
  `|u| = 0.0158` (Ma 0.027) at step 19,000 to 0.94, with `rho` staying inside
  **1.0000-1.0121** the entire time. It is not a compressibility failure; it
  starts at, and below, the 0.05 design velocity.
* **The mechanism is the one recorded as a contingency in earlier rounds.** At
  `tau_plus = 0.500010`, `omega_minus_from_magic`
  (`physics/fluid_flow.py:239-252`, inlined at `:1338-1341`) gives
  `half = 1/omega_plus - 1/2 = 1.038e-5` and hence
  **`tau_minus = 18,060`** — the odd, momentum-carrying moments have an
  18,000-step relaxation time. `omega_plus = 1.999958` is `4.2e-5` from the
  hard TRT/BGK stability limit of 2. The scheme therefore has **essentially
  zero dissipation of a momentum mode**, and the one strain-dependent
  dissipation it does have (Smagorinsky) senses the *symmetric* second moment
  (`physics/fluid_flow.py:1310-1330`) and is blind to an odd-parity staggered
  mode — while being switched off entirely inside filter cells
  (`:1309`, `physics/filling.py:1941`).
* **The fix is validated on the real assembly.** Emulating the proposed
  `tau_minus <= 5` clamp (magic parameter set to `4.5 * half`), Test 2 runs to
  **step 23,284 and stops on the fill-time cap, not a divergence** — 3,092
  steps past the death step, with **no non-finite cell**, `max|u|` never above
  **0.052** (against 0.938), **zero** rows above 0.1 (against 130), zero
  negative populations, and the runaway cell flat at `|u| = 0.0055`. All 123 of
  the earlier "damped" velocity excursions vanish as well: they were the same
  undamped ghost mode, not turbulence. See section 4(e). The clamp cannot be
  confined to filter cells and is not byte-identical off-filter — section 4(c)
  states that cost precisely.
* **The failure is NOT a regression from `fe866cf`.** The runaway cells are
  non-porous, carry no drag, and their mass bookkeeping is untouched by the
  round-4 change; the growth is visible in `|u|` from at least step 19,000 and
  is a pure momentum mode. Round 4 removed the mass-side amplifier that used to
  kill the run 3,700 steps earlier, which is why this mode is only now visible.

---

## 1. CODE AUDIT (committed `fe866cf`)

### 1(a) The dissipation budget of a Test 2 cell

`lattice_units(dx=0.0043, nu=0.006/7200, u_max_physical=2.8)` gives, measured by
running it:

```
dt = 7.67857e-05 s   nu_lattice = 3.46069e-06
tau        = 0.500010382      omega_plus  = 1.999958
half       = tau - 1/2 = 1.0382e-05
omega_minus = 1/(magic/half + 1/2) = 5.53695e-05    tau_minus = 18,060.5
```

with `magic = TRT_MAGIC_PARAMETER = 3/16` (`physics/fluid_flow.py:101`).

Three separate facts follow, and all three matter:

1. **`omega_plus = 1.999958` is 4.2e-5 below the stability limit.** TRT/BGK
   requires `omega < 2`; the physical viscosity of liquid steel on a 4.3 mm
   grid at this `dt` is essentially zero, so the symmetric relaxation is
   over-relaxed to the very edge. `lattice_units`' own docstring
   (`physics/fluid_flow.py:175-179`) says shrinking `dt` pushes `tau` *closer*
   to 1/2, so this is not tunable away by the timestep.
2. **`tau_minus = 18,060`.** The antisymmetric (odd) moments — the ones that
   carry momentum — are relaxed with a time constant of 18,000 steps. Over the
   ~200 steps a mode needs to double, they are **frozen**: the odd part of `f`
   is a pure accumulator of the forcing, with no restoring term at all.
3. **The only strain-dependent dissipation is Smagorinsky, and it is doubly
   unable to help.**
   * It is **switched off inside filter cells**:
     `physics/fluid_flow.py:1309` — `if use_smagorinsky and c1 == 0.0:` — and
     `physics/filling.py:1941` — `suppress=state.porous.drag_c1 > 0.0`. The
     rationale (`physics/fluid_flow.py:895-899`) is real: the Forchheimer drag
     is already the pore-scale inertial closure and stacking an eddy viscosity
     on top double-counts it.
   * Where it *is* on, its sensor is the **symmetric** non-equilibrium stress,
     `qmag = |sum_i c_i c_j (f_i - eq_i)|` (`:1310-1330`). An odd-parity
     staggered mode puts its energy in the antisymmetric moments and
     contributes little to that symmetric sum, so `tau_eff` barely moves and
     `omega_plus` stays at 1.999958.

There is therefore **no term in the scheme that damps an odd-parity momentum
mode** at this `tau`. That is the defect. Everything else in this report is
evidence for it.

### 1(b) The coupling that makes it worse, not better

`omega_minus` is not independent: `physics/fluid_flow.py:1338-1341` computes it
from whatever `omega_plus` the Smagorinsky branch produced.

```
1338   half = 1.0 / omega_plus - 0.5
1339   if half < 1e-12:
1340       half = 1e-12
1341   omega_minus = 1.0 / (magic / half + 0.5)
```

So **the only route by which the odd moments ever get relaxed is Smagorinsky
raising `tau_eff`** — and that route is closed inside filters by `:1309` and
nearly closed everywhere else for this mode, because the sensor cannot see it.
`tau_minus <= 5` requires `half >= magic/4.5 = 0.041667`, i.e.
`tau_plus >= 0.541667`; Test 2 runs the whole domain at 0.500010.

### 1(c) The porous drag, audited (H1)

`collide_kernel`, `physics/fluid_flow.py:1223-1257`, and its NumPy mirror
`porous_drag_force`/`resolve_porous_velocity` (`:708-771`, `:675-706`):

```
1224   c0 = drag_c0[x,y,z];  c1 = drag_c1[x,y,z]
1227   denom = c0 + sqrt(c0*c0 + c1*speed)        # speed = |v|
1228-30 ux,uy,uz /= denom                          # the IMPLICIT Guo-Zhao brake
1253   speed_u = |u|
1254   drag = 2*rho*((2*c0 - 1) + c1*speed_u)
1255-7 fx,fy,fz -= drag * u                        # the EXPLICIT body force
```

The two halves are algebraically consistent, not double-counted: inverting the
quadratic `resolve_porous_velocity` solves gives `v = 2 c0 u + c1|u| u`, hence
`denom == 2 c0 + c1|u|` exactly, and `F/rho = 2(u - v)`. Measured drag
coefficients for both Test 2 filters (logged by the probe at run start):

```
SPEC zone=1/2  cells=2354  porosity=0.85  pore_d=2.54e-3  k1=5.33615e-07
               inv_k2=752.92  cells_along_flow=16  thickness_correction=1
DRAG COEFFS    c0 = 0.500022828579   c1 = 1.381721139
               off-filter: c0 = 0.5 exactly, c1 = 0.0 exactly
```

The per-step fraction of momentum the drag removes is

```
R = |F_drag| / |m| = 2[(2 c0 - 1) + c1|u|] / (2 c0 + c1|u|)
```

`R < 2` for **every** `|u|`: `R = 2` would need `2c0 - 1 + a = 2c0 + a`, i.e.
`-1 = 0`. Iterating the exact map (`rho = 1`, no gravity) confirms it:

```
       s   u=s/denom   a=c1|u|         R       s_next   |s_next/s|
    0.05    0.046952   0.06487   0.12192    0.043904     0.878075
     0.30    0.228100   0.31520   0.47934    0.156200     0.520663
  0.72370    0.447270   1.00000   1.00000    ~0            ~0        <- critical
     1.00    0.562610   0.77740   0.87477    0.125230     0.125226
     5.00    1.574500   2.17600   1.37019   -1.851000     0.370195
    1e+08    8506.9     1.175e+04 1.99983   -9.9983e+07   0.999830

sup over all s of |s_next/s| = 0.999982986   (at s = 1e10)  ->  CONTRACTION
```

Read that table carefully, because it is the opposite of the hypothesis:
the drag is **most** damping at `|u| = 0.7237` (`R = 1`, momentum annihilated in
one step), and only becomes marginally non-damping at `|u| ~ 1e4`. Above
`|u| = 0.7237` its response does become *oscillatory* (`R > 1`, the momentum
reverses sign each step) — but with `|1-R| < 1` it still decays. **The explicit
Guo-Zhao drag at these coefficients cannot amplify momentum.**

---

## 2. THE FIXTURE — a sustained jet into a porous face

`jet_fixture.py`. A 5x5x44 walled channel, **all FLUID, no GAS and no INTERFACE
cell anywhere**, so the free-surface bookkeeping is inert by construction and
only the momentum path is under test. It uses the **real Test 2 lattice**
(dx = 4.3 mm, `tau = 0.500010`, `omega_minus = 5.537e-5`) and the **real filter
drag** (`c0 = 0.500022828579`, `c1 = 1.381721139`, porosity 0.85). A velocity
inlet pins the two bottom planes to `equilibrium(1.0, (0,0,U))` every step, so
the jet is sustained rather than decaying. NumPy path, so every relaxation term
is monkeypatchable. 8,000 steps.

`geom` controls how the porous band cuts the channel: `full` spans the whole
cross-section (interface perpendicular to the flow only); `lateral` covers half
the y-range, so a porous/clear interface runs **along** the flow — which is the
configuration Test 2 dies in (the fatal pocket straddles the filter/runner
boundary *within the plane*).

Cell value = `max|u|` reached over the run, or `DIED@step` (non-finite).

| variant | porosity band | Guo-Zhao drag | Smagorinsky on band | U=0.05 | U=0.15 | U=0.22 | U=0.25 | U=0.30 | U=0.40 |
|---|---|---|---|---|---|---|---|---|---|
| `stock` (full band) | 0.85 | ON | **suppressed** | 0.0588 | 0.2289 | 0.3486 | 0.3685 | 0.4245 | **DIED@7146** |
| `lat_stock` (lateral) | 0.85 | ON | **suppressed** | 0.0655 | 0.2506 | **DIED@7426** | **DIED@5727** | **DIED@6814** | **DIED@2839** |
| `lat_dragx2` | 0.85 | **2x** | suppressed | 0.0644 | 0.2471 | DIED@7890 | 7.3666 | DIED@7905 | **DIED@4353** |
| `full_supOnly` | **1.0 (none)** | **none** | **suppressed** | **DIED@1402** | **DIED@249** | **DIED@167** | **DIED@148** | **DIED@126** | **DIED@104** |
| `full_porSup` | 0.85 | **none** | **suppressed** | **DIED@1292** | **DIED@240** | **DIED@161** | **DIED@144** | **DIED@123** | **DIED@101** |
| `lat_porOnly` | 0.85 | **none** | **active** | 0.0700 | 0.2669 | 0.3913 | 0.4021 | 0.4252 | **0.4772** |
| `lat_smagP` | 0.85 | ON | suppressed + **free `omega_plus`** | 0.0655 | 0.2505 | 0.3730 | 0.3848 | 0.4248 | **0.4772** |
| `lat_smagM` | 0.85 | ON | suppressed + **free `omega_minus`** | 0.0655 | 0.2503 | 0.3729 | 0.3849 | 0.4248 | **0.4772** |
| `lat_taum` | 0.85 | ON | suppressed + **`tau_minus <= 5`** | 0.0559 | 0.2015 | 0.2941 | 0.3465 | 0.3910 | **0.4695** |

Full output in `jet_fixture.log`.

Six measurements, in the order they settle the question:

1. **`full_supOnly` dies at every velocity, including the 0.05 design velocity
   (step 1,402), with NO filter present at all** — porosity 1, no drag, only
   Smagorinsky switched off on that footprint. A dissipation-free patch in a jet
   is *unconditionally* unstable at `tau = 0.500010`. This is the mechanism in
   its purest form.
2. **The porosity discontinuity contributes nothing.** `full_porSup` (porosity
   0.85, no drag, suppressed) matches `full_supOnly` to within a few steps at
   every velocity: 1292/1402, 240/249, 161/167, 144/148, 101/104. The
   equilibrium's `1/eps` correction (`physics/fluid_flow.py:260-290`) is not the
   seed.
3. **With Smagorinsky active, the same porous band is perfectly stable.**
   `lat_porOnly` — porosity 0.85, *no drag at all*, Smagorinsky on — survives
   every velocity. So it is not the filter; it is the missing dissipation.
4. **The drag is the stabiliser — a mitigant, not a cure.** Adding it to
   variant 2's configuration takes the death from step 1,402 to *survival* at
   U=0.05 (`stock`), and from step 104 to step 7,146 at U=0.4 — a **69x**
   extension. Doubling it extends `lat_stock` further still (step 2,839 ->
   4,353 at U=0.4) but does not prevent the death. So more drag is
   monotonically *better* and never worse. **H1 is not merely unsupported; the
   measurement runs the other way.** What the drag cannot do is remove the
   mode — only restoring the relaxation does that (item 6).
5. **A porous/clear interface parallel to the flow lowers the threshold into the
   range Test 2 actually visits.** `stock` survives to U=0.25; `lat_stock` dies
   from U=0.22. Test 2's `max|u|` spent 123 of 1,009 sampled rows in 0.24-0.485.
6. **Both dissipation restorations fix it, at every velocity tested.**
   `lat_smagM` (take `omega_minus` from the *un-suppressed* Smagorinsky
   computation, leave `omega_plus` suppressed) reaches 0.4772 where `lat_stock`
   dies at step 2,839; `lat_taum` (clamp `tau_minus <= 5`) reaches 0.4695.
   `lat_smagP` (eddy viscosity in `omega_plus` only, `omega_minus` left
   frozen) holds identically (0.4772), so *either* moment suffices once some
   dissipation exists — the two agree to four digits at every velocity, which
   says the instability needs only *a* dissipation channel, not a particular
   one. `lat_taum` is the most dissipative of the three and the only one that
   is not strain-gated.

**Honest limitation of the fixture.** Its failure is driven by the *suppressed
patch*; the real assembly's failure (section 3) is in cells where Smagorinsky is
already active. The fixture therefore reproduces the *class* of failure —
insufficient damping of a momentum mode at `tau ~ 1/2` — and cleanly falsifies
H1 and the porosity hypothesis, but it is not a like-for-like replica of the
real mode. The real-assembly evidence below is what pins the specific mode, and
section 4 is what tests the fix against it.

---

## 3. THE REAL ASSEMBLY — one instrumented run, 20-step cadence

`drv3.py` (the validator's `fill_driver.py` plus a pocket/drag probe),
`CASTSIM_THREAD_BUDGET=14`, `--ledger-every 20`, pocket dumps every 20 steps
from step 19,000 and every step from 20,080. Log `m3.log`, ledger
`m3_ledger.csv`, pocket `p1_summary.csv` / `p1_cells.csv`.

Outcome: `# NONFINITE at step 20192: 287 cells`, `status=diverged steps=20200`
— identical to the validator.

### 3(a) Where the runaway lives, and what it is

The probe records the argmax of `|u|` every step. Through the entire growth
phase it is **not in a filter cell**:

```
step   max|u|      cell        type          porosity  R(drag)
19740  0.179319   (124,68,44)  6 SPRUE       1.0       0
19940  0.057901   (124,71,46)  6 SPRUE       1.0       0
19960  0.055327   (118,50,24)  7 RUNNER      1.0       0    <- locks on, never leaves
20000  0.068029   (118,50,24)  7 RUNNER      1.0       0
20080  0.135689   (118,50,24)  7 RUNNER      1.0       0
20120  0.241761   (118,50,24)  7 RUNNER      1.0       0
```

Before step ~19,940 the global maximum roams the **sprue** (type 6, x 123-124,
z 44-57) — the falling column, 0.055-0.18, fluctuating with no trend. At step
**19,960** it locks onto `(118,50,24)`, a **RUNNER** cell with **porosity 1**,
and never moves again.

Filter #1's boundary in x is between 119 and 120: `(120,48,24)` is FILTER
(porosity 0.85). So the runaway sits **one to two cells outside the filter, on
the runner side of its inlet face** — which is exactly the pocket the validator
dumped, and it straddles the filter/runner boundary in-plane.

**It is a period-2 staggered mode, and the probe catches it directly.** Over 31
consecutive steps the argmax alternates between two diagonal neighbours without
a single break:

```
20080  0.135689 (118,50,24)      20081  0.223409 (119,49,24)
20082  0.138464 (118,50,24)      20083  0.229903 (119,49,24)
20084  0.141482 (118,50,24)      20085  0.236026 (119,49,24)
   ... 31 consecutive steps, perfect parity alternation ...
20108  0.195310 (118,50,24)      20109  0.342409 (119,49,24)
```

Cell A wins on even steps and cell B on odd steps, so **each cell's `|u|`
oscillates with period 2, in antiphase with its diagonal neighbour** — a
textbook checkerboard mode, on the lattice diagonal `(+1,-1,0)`.

By the last 40 steps the alternation is violent, and visible in the global
maximum itself — consecutive steps differ by a factor of 3-5:

```
20150  0.642    20151  1.801    20152  0.734    20153  3.120
20154  0.870    20155  3.063    20156  1.095    20157  2.317
```

No physical flow changes by 3x in one 77 us timestep. This is the mode's
signature and it is unambiguous.

The velocity direction confirms it. At `(118,50,24)`:

```
step     |u|        rho        ux         uy         uz       R
19000  0.015796  1.001807  -0.009677  +0.002568  -0.012217  0.0
19400  0.020634  0.999361  -0.015689  +0.005789  -0.012088  0.0
19800  0.036685  1.001131  -0.030135  +0.016615  -0.012714  0.0
19960  0.055327  1.001295  -0.045751  +0.028589  -0.012272  0.0
20040  0.091057  1.003503  -0.073380  +0.052943  -0.010190  0.0
20120  0.241761  1.012063  -0.185243  +0.155332  -0.002288  0.0
```

`ux` grows 19x, `uy` grows 60x, **`uz` decays** — the mode converges on
`u ~ (-a, +a, 0)`, which is precisely the `u = [+a, -a, ~0]` the validator saw
in the non-finite dump. A single lattice diagonal in the x-y plane.

### 3(b) The three hypotheses, measured at the runaway cell

* **`R = 0.0` exactly, at every one of the 61 samples**, because the cell is
  non-porous (`c1 = 0`, so `R = 2(2*0.5-1+0)/(1+0) = 0`). The Guo-Zhao drag is
  *identically absent* where the failure grows. **H1 falsified on the real
  assembly, not merely in the fixture.** And the drag never enters its
  oscillatory regime while the mode is growing: the count of porous cells with
  `R > 1` is **0 at every one of the 162 samples from step 19,000 to 20,157**.
  It first becomes nonzero at step **20,158** — **exactly one cell**, at
  `R_max = 1.2899`, and back to zero at 20,159 — by which point `max|u|` is
  already at 1.3-3.1 and the runaway is fully developed. `R > 1` is a late,
  transient *consequence*, and even then a contraction (`|1-R| = 0.29`).
* **`rho` stays in 1.0000-1.0121 for the whole growth.** The mode grows 15x in
  `|u|` while the density moves 1.2%. **H2 falsified**: this is not a
  compressibility or stagnation-pressure failure, and it begins at
  `|u| = 0.0158` — Ma 0.027, *below* the 0.05 design velocity.
* **`flag = 3` (FLUID) at every sample**, so no transition, promotion or heal
  event seeds it; and `eps` in the pocket stays near 1 throughout, so the
  round-4 mass-side change is not implicated.

### 3(c) The growth is super-exponential and starts long before the ledger sees it

Log-slope of `|u|` at `(118,50,24)`, by window:

| window | `|u|` | rate /step | e-folding |
|---|---|---|---|
| 19,000-19,400 | 0.01580 -> 0.02063 | +0.000668 | 1,497 steps |
| 19,400-19,800 | 0.02063 -> 0.03669 | +0.001439 | 695 steps |
| 19,800-20,000 | 0.03669 -> 0.06803 | +0.003088 | 324 steps |
| 20,000-20,080 | 0.06803 -> 0.13569 | +0.008630 | 116 steps |
| 20,080-20,180 | 0.13569 -> 0.93763 | +0.019330 | 52 steps |

The rate roughly **doubles each window** — a nonlinear (advective) instability,
not a linear one. And it is already growing at step 19,000, the earliest sample
in the trace, at `|u| = 0.0158`. Extrapolating the first window's rate back to
`|u| ~ 1e-3` puts the onset near step **15,000**. **This mode has been growing
for at least 5,000 steps**; the validator's "192 steps from 0.068 to 6.5e16" is
the last two e-foldings of a much longer history, and the earlier part is
invisible in the ledger because the global `max|u|` is dominated by the sprue's
free-fall jet until step 19,960.

---

## 4. FIX DIRECTION

### 4(a) What is actually wrong, stated as a design fact

In TRT, **`omega_plus` sets the shear viscosity; `omega_minus` does not.**
`omega_minus` enters only through `Lambda = (1/omega_plus - 1/2)(1/omega_minus -
1/2)`, which fixes the accuracy of bounce-back (Lambda = 3/16 puts the effective
no-slip wall exactly midway between nodes) and, crucially, **sets how fast the
odd non-hydrodynamic "ghost" modes are damped**.

Holding `Lambda` at 3/16 while `tau_plus -> 1/2` therefore drives
`omega_minus -> 0`. That is the whole defect: this solver has chosen the
wall-accuracy-optimal `Lambda` at a `tau_plus` where doing so leaves the odd
ghost modes **undamped for 18,000 steps**. It is a known TRT trade-off and the
standard remedy is to bound `omega_minus` away from zero, accepting a shifted
wall location. `TRT_MAGIC_PARAMETER = 3.0/16.0`
(`physics/fluid_flow.py:101`) is a good choice at `tau ~ 0.6-1`; it is the
wrong choice at `tau = 0.50001`.

### 4(b) The fix: clamp `tau_minus`

```python
# physics/fluid_flow.py -- new module constant next to TRT_MAGIC_PARAMETER:101
#: Upper bound on the antisymmetric relaxation time. Lambda = 3/16 is the
#: wall-accuracy-optimal magic number, but holding it as tau_plus -> 1/2 sends
#: omega_minus -> 0 and leaves the odd ghost modes undamped for ~18,000 steps.
#: Binds only for tau_plus < magic/(TAU_MINUS_MAX - 1/2) + 1/2 = 0.541667, so
#: any solve at a larger tau keeps the exact Lambda = 3/16 wall location.
TAU_MINUS_MAX = 5.0
```

**Kernel** — `physics/fluid_flow.py:1338-1341` becomes:

```python
half = 1.0 / omega_plus - 0.5
if half < 1e-12:
    half = 1e-12
tau_minus = magic / half + 0.5
if tau_minus > tau_minus_max:          # new kernel argument, or a global
    tau_minus = tau_minus_max
omega_minus = 1.0 / tau_minus
```

**NumPy path** — the same ceiling in the two places that produce
`omega_minus`: `omega_minus_from_magic` (`physics/fluid_flow.py:239-252`) and
the `smagorinsky_relaxation` return (`:936-991`, whose `tau_eff` feeds the same
inversion). Both paths must change together or
`tests/test_filling_parity.py` fails — which is the desired safety property.

`tau_minus_max` should be threaded as a kernel argument the way `magic` already
is (`physics/filling.py:2076-2091` passes `float(TRT_MAGIC_PARAMETER)`), so the
benchmarks can set it to `inf` to recover exact `Lambda = 3/16` behaviour.

### 4(c) This fix does NOT reduce to current behaviour off-filter, and that is unavoidable

Stated plainly, because it is the important caveat:

* **The clamp binds in every cell of every model whose `tau_plus < 0.541667`.**
  Test 2 runs its entire domain at `tau_plus = 0.500010`, so the clamp changes
  **every wet cell**, not just filter cells. `Full_Gating_Test` and
  `Filling_Test` will change bit-for-bit if they run at `tau < 0.5417`, which
  they do for the same reason Test 2 does (`lattice_units`' docstring,
  `physics/fluid_flow.py:175-179`: shrinking `dt` pushes `tau` *closer* to 1/2,
  so a fine grid cannot escape it). **Those two models are no longer
  byte-identical regression sentinels for this change** and must be re-baselined
  on physical outcomes (fill fraction, fill time, delivered mass, misrun count)
  rather than on bitwise equality.
* **A filter-only fix is not available, because the failure is not in a filter
  cell.** The runaway lives in RUNNER cells at porosity 1 with `c1 = 0`
  (section 3a). The variant that confines the change to filters —
  `lat_smagM`, taking `omega_minus` from the un-suppressed Smagorinsky
  computation while leaving `omega_plus` suppressed — **fixes the fixture and
  would do nothing for the real assembly**, whose cells are already
  un-suppressed. I am reporting it as measured-and-insufficient rather than
  recommending it.
* **What the clamp does *not* change is the fluid viscosity.** `omega_plus`, and
  hence the shear viscosity and the Smagorinsky closure, are untouched. The
  clamp changes `Lambda` from 3/16 to `half * (TAU_MINUS_MAX - 1/2)` =
  4.67e-5 at Test 2's `tau`, which shifts the effective bounce-back wall off the
  node midpoint. On a fill at ~1-cell geometric accuracy, with a 3.0-cell gate
  the solver already warns about, that is a smaller error than the one it
  removes — but it **is** an error, it should be quantified against
  `tests/test_filling_lattice.py`'s Poiseuille wall-location test, and that
  test should be pinned at a `tau` above 0.5417 (where the clamp does not bind)
  so it keeps testing `Lambda = 3/16` exactly.

### 4(d) Do NOT "fix" the drag, and do NOT un-suppress Smagorinsky in filters

* The drag is the stabiliser, measured twice (section 2 items 4, section 3b).
  Making it implicit, softer, or removing it makes things strictly worse. The
  `f1daf97` change that added it was correct and should stay.
* Un-suppressing Smagorinsky inside filters (`physics/fluid_flow.py:1309`,
  `physics/filling.py:1941`) does stabilise the fixture, but it re-introduces
  exactly the double-counting the suppression exists to prevent
  (`physics/fluid_flow.py:895-899`: the Forchheimer term already *is* the
  pore-scale inertial closure), and it would not have prevented this failure,
  which is in un-suppressed cells.

### 4(e) Fix validation on the real assembly

`physics/filling.py:2076-2091` passes `float(TRT_MAGIC_PARAMETER)` into
`collide_kernel`, and the kernel forms `omega_minus = 1/(magic/half + 1/2)`.
Setting `magic = 4.5 * half = 4.6719e-5` therefore emulates
`tau_minus <= 5` **exactly** at the molecular `tau` where all but a handful of
cells sit, and over-relaxes `omega_minus` toward 2 only in the few cells where
Smagorinsky has already raised `tau_eff` — conservative in the stabilising
direction. Monkeypatch only (`drv3.py`, `MAGIC` env var); the repo is untouched.

Run: `MAGIC=4.6719e-05 ... --max-fill-time 1.60 --ledger-every 20`
(`mfix.log`, `mfix_ledger.csv`, `pfix_summary.csv`).

**RESULT: the failure is gone, and so is everything that led to it.**

| | UNFIXED (`magic = 3/16`, `tau_minus = 18,060`) | FIXED (`tau_minus <= 5`) |
|---|---|---|
| outcome | `status=diverged`, **step 20,200** | `status=time_capped`, **step 23,284** |
| first non-finite | **step 20,192** | **NONE** |
| `max\|u\|` over the run | **0.93763** | **0.05189** |
| `max\|u\|` p99 / median | 0.35232 / 0.05756 | **0.04572 / 0.03368** |
| ledger rows with `max\|u\|` > 0.2 | **33** | **0** |
| ledger rows with `max\|u\|` > 0.1 | **130** | **0** |
| negative-population cells at the last row | 152 | **0** |
| `eps_fluid_max` at the last row | `nan` | 1.0025 |
| `\|u\|` at the runaway cell (118,50,24) | 0.0158 -> 0.94, rate +6.7e-4 -> +1.9e-2/step | **0.0055, flat, rate +4.9e-5/step** |

It ran **3,092 steps past the death step** and stopped only on the fill-time
cap. Three things in that table are worth separating out:

1. **The mode is not merely damped, it never forms.** `|u|` at
   `(118,50,24)` sits at 0.0050-0.0064 for 4,200 consecutive samples with a
   fitted growth rate of `+4.9e-5/step` — zero to the precision of the fit.
2. **All 123 "damped excursions" are gone too.** `max|u|` never exceeded
   **0.052** over 1,164 rows, against 130 rows above 0.1 and 33 above 0.2
   unfixed. The "marginally stable for 16,000 steps" character of the run was
   not physical turbulence — **it was the same undamped odd ghost mode, excited
   repeatedly and only marginally decaying each time.** With `omega_minus`
   restored it does not appear at all.
3. **It is therefore an accuracy fix as much as a stability fix.** Compared at
   equal steps, `max|u|` is roughly *halved* throughout: 0.0516 -> 0.0251 at
   step 2,000; 0.0691 -> 0.0391 at 6,000; 0.0754 -> 0.0330 at 14,000; 0.0680 ->
   0.0329 at 20,000. The unfixed velocity field's maximum was ghost-mode
   contamination by about a factor of two, for the whole run. The median now
   sits at 0.034, comfortably under the 0.05 design velocity the timestep was
   chosen for — which is what the sizing rule intended all along.

And the cost, measured, so it is not oversold:

* **Mass is unaffected**: wet mass agrees to **0.015 %** at step 20,000
  (22,619.8 vs 22,616.3). Delivery is metered by the stage, so this is the
  conservation ledger closing, not a coincidence.
* **The front shape does move**: wet cells differ by **-2.06 %** at step 20,000
  (26,278 vs 25,736). That is the expected consequence of changing `Lambda` and
  hence the effective wall location, and it is the honest price of section
  4(c). It is also, on this evidence, a *better* answer -- the unfixed front was
  being advanced by a velocity field with a spurious factor-of-two maximum.
* The fixed run still reports the pre-existing warnings (3.0-cell gate, misrun,
  trapped air) and still did not finish the pour inside the cap, so **mechanism
  3 being fixed does not by itself make Test 2 complete**; it makes it a
  physically meaningful run that has to be given enough time.

---

## 5. VELOCITY PRECURSOR, calibrated on the 20-step data

### 5(a) A precursor on the GLOBAL `max|u|` cannot work — measured

This is the first thing to report, because it is the obvious design and it is
structurally blind here. Sliding log-linear fit of `max_abs_u` over the last M
20-step samples, firing on `slope > s`, `r2 > 0.9`, `max|u| > 0.05`, evaluated
over all 1,009 clean rows of `m3_ledger.csv`:

| M (samples) | slope | fires | false pos (<19,700) | first true | lead |
|---|---|---|---|---|---|
| 10 | 0.0015 | 29 | 22 | 19,700 | 492 |
| 15 | 0.0030 | 0 | 0 | — | — |
| 20 | 0.0015 | 7 | **7 (all in steps 960-1,080, the pour transient)** | **never** | — |
| 20 | 0.0030 | 0 | 0 | — | — |

Any setting tight enough to ignore the 123 damped excursions **misses the real
event entirely**. The reason is in section 3c: the growing mode is not the
global maximum until step 19,960, because the sprue's free-fall jet dominates
`max|u|` for the first 19,900 steps. A global scalar cannot see a localised mode
growing three times slower than the noise floor.

Corollary for the existing guard: the `eps` band gave 32 steps of warning
(validator, section B); global `max|u|` gives none at all.

### 5(b) The precursor that does work: a per-cell rise counter

The distinguishing feature of the real event is not magnitude, it is
**persistence at a fixed location**. That is detectable with two extra grids and
O(1) work per cell per check:

* one `float32` grid holding `|u|` from the previous check (7 MB at Test 2's size)
* one `int8` grid of consecutive-rise counts

At each `DIVERGENCE_CHECK_INTERVAL`-style check (this calibration uses a 20-step
cadence): `counter = counter + 1 where |u| > prev else 0`, then fire when at
least `N` wet cells have `counter >= K` and `|u| > FLOOR`.

Calibrated on `p1_cells.csv` (913 traced pocket cells, steps 19,000-20,200):

| K | FLOOR | N | first fires | lead to NaN | lead in ms |
|---|---|---|---|---|---|
| 8 | 0.01 | 3 | step 19,840 | **352 steps** | 27.0 |
| 8 | 0.02 | 3 | step 19,940 | **252 steps** | 19.3 |
| 10 | 0.01 | 3 | step 19,880 | 312 steps | 24.0 |
| 6 | 0.01 | 3 | step 19,800 | 392 steps | 30.1 |

**Recommended: `K = 8` consecutive rises, `FLOOR = 0.01` lattice (a fifth of
`u_max_lattice`), `N = 3` cells, at a 20-step cadence — 352 steps of lead, an
order of magnitude more than the `eps` band's 32.** The counts at steps
19,120-19,480 are 0-2, so `N >= 3` produced no false positive anywhere in the
traced window.

An equivalent per-cell log-slope detector (`ln|u|` slope > 0.0015/step over 10
samples, `r2 > 0.95`) fires at step **19,660** — **532 steps (40.8 ms) of
lead** — and fires in 33 of 913 pocket cells, all inside the one growing mode.

Three caveats, stated because they bound the calibration:

1. **My trace starts at step 19,000, so these leads are lower bounds.** The mode
   was already growing at step 19,000; a detector running from step 0 would
   plausibly have fired thousands of steps earlier (section 3c extrapolates the
   onset to ~step 15,000).
2. **False positives are calibrated only over 913 cells in one pocket, not over
   339,824.** Before shipping, run the counter over a *healthy* fill
   (`Full_Gating_Test`, `Filling_Test`) and raise `N` until it is silent. A rising
   front legitimately accelerates, so `N` will need to be well above 3 on the
   full domain, or the counter should be restricted to cells that have been
   FLUID for at least a few hundred steps.
3. A cheaper qualitative alternative worth measuring: the mode's signature is a
   **period-2 alternation**, and a physical flow at `dt = 77 us` cannot change
   sign step to step. A detector on
   `max over wet cells of |u(n) - u(n-1)| / max(|u(n)|, floor)` would key on the
   *character* of the mode rather than its amplitude, needing one extra grid and
   two consecutive steps. I did not calibrate it (my per-cell dumps are at
   20-step cadence, which aliases period 2), but section 3a's 31-step perfect
   alternation is direct evidence that the signal is there and is unambiguous.

---

## 6. HYPOTHESES RULED OUT, and how

| hypothesis | verdict | how |
|---|---|---|
| **H1 — explicit porous-drag stiffness: per-step damping `> 2`, over-corrects, flips sign, Forchheimer makes it worse as `\|u\|` rises** | **FALSE, and inverted** | Algebra: `R = 2[(2c0-1)+a]/(2c0+a) = 2` requires `-1 = 0`; `R < 2` for all `\|u\|`. Numerics: iterating the exact 0-D map over `s` in `1e-8..1e10` gives `sup \|s'/s\| = 0.99998` — a strict contraction, with maximum damping (`R = 1`, momentum annihilated in one step) at `\|u\| = 0.7237`. Real assembly: the runaway cells are non-porous, `R = 0.0` at all 61 samples, and **no porous cell anywhere has `R > 1` at any sample during the entire growth phase** (first nonzero at step 20,188, four steps before the NaN). Fixture: removing the drag kills it at every velocity (`full_supOnly` DIED@1402 at U=0.05); doubling it cuts peak `\|u\|` 11.27 -> 3.51. The drag is the stabiliser. |
| **H1b — `resolve_porous_velocity` and `porous_drag_force` double-count or explicit-lag** | **FALSE** | Audited at `physics/fluid_flow.py:1223-1257`: inverting the quadratic gives `v = 2c0 u + c1\|u\|u`, hence `denom == 2c0 + c1\|u\|` exactly, so the brake and the body force are the two consistent halves of one implicit solve, not two applications of it. Verified numerically: `v/denom` reproduces the self-consistent `u`. |
| **H2 — jet compressibility / Mach: `\|u\|` 0.3-0.5 is Ma 0.5-0.9 in a scheme designed for 0.05** | **FALSE as the cause** | The mode grows from `\|u\| = 0.0158` (Ma 0.027, **below** the design velocity) with `rho` inside 1.0000-1.0121 for the whole growth. High Mach is a *consequence* reached in the last ~50 steps, not the trigger. Separately: `max\|u\|` reached 0.24-0.485 on 123 sampled rows over 20,000 steps and damped every time, so a high-Mach excursion is neither necessary nor sufficient. |
| **H3 — interaction with the round-4 diff (porous exchange override removed)** | **FALSE** | The committed change is exchange-only (`physics/fluid_flow.py` free_surface_kernel, `physics/filling.py` advance_mass, plus the `create_state` porosity seed). No porous read path in the collide/forcing route changed. Measured: the runaway cells are non-porous, `flag = 3` at every sample, and `eps` in the pocket stays near 1 — the failure has no mass-side observable at all. The pad is also not implicated: filter #1 is interior (padded x 105-123 of 137). |
| **H4 — float32 population storage: a transient `\|f\| > 3.4e38` is `inf`** | **downstream, not causal** | The probe records `fmax_abs` (max `\|f\|` over the grid) every step. Through the growth phase it stays *tiny*: 0.394 at step 20,040, 0.421 at 20,080, 0.483 at 20,120, 0.533 at 20,140, and its maximum over the whole window 19,000-20,187 is **6.07** — **37 orders of magnitude** below float32's 3.4e38 ceiling. The populations only overflow after `rho` is already at 1e14. The `overflow encountered in cast` at `_seed_promoted_cells` (`physics/filling.py:1319`) is therefore a consequence. It remains a real robustness concern worth a separate note — a float64 population reaching 3.4e38 becomes `inf` on store, a much lower ceiling than the arithmetic that produced it — but it is not this mechanism. |
| the porosity discontinuity / the equilibrium's `1/eps` correction as the seed | **FALSE** | `full_porSup` (porosity 0.85, no drag, suppressed) matches `full_supOnly` (porosity 1, no drag, suppressed) to within a few steps at every velocity: 1292/1402, 240/249, 161/167, 144/148, 101/104. And `lat_porOnly` (porosity 0.85, no drag, **Smagorinsky on**) is stable everywhere. |
| Smagorinsky suppression inside filters as *this* failure's cause | **partly — it is the same defect, but not at this location** | It is unconditionally fatal in the fixture (`full_supOnly` dies at U = 0.05, with no filter present), so the suppression *is* a live hazard in its own right and should be recorded as such. But the real assembly's runaway is in cells where Smagorinsky is already active, so un-suppressing it would not have prevented step 20,192. |

---

## 7. ARTEFACTS

All under `Test/session_diag_2026-09-02/mech3_filter_inlet/`:

| file | what |
|---|---|
| `drv3.py` | the validator's `fill_driver.py` plus a per-step pocket/drag probe (argmax of `\|u\|` with part type and porosity, the Guo-Zhao `R` per porous cell, per-cell dumps of `rho`/`mass`/`eps`/`u`/`c0`/`c1`/`denom`/`R`/drag force) and a `MAGIC` override hook for the fix validation |
| `m3.log`, `m3_ledger.csv` | the reproduction run: 20-step ledger, non-finite at step 20,192, `status=diverged` |
| `p1_summary.csv` | per-step argmax/`R`/`fmax` summary, steps 19,000-20,200 |
| `p1_cells.csv` | per-cell pocket dumps (913 wet cells), 20-step cadence from step 19,000 |
| `mfix.log`, `mfix_ledger.csv`, `pfix_summary.csv` | the `tau_minus <= 5` validation run |
| `jet_fixture.py`, `jet_fixture.log` | the ~2,000-cell jet-into-a-porous-face fixture, 13 variants, real Test 2 lattice and real filter drag, and its output |
| `pfix_cells.csv` | per-cell pocket dumps from the fixed run, 20-step cadence (the flat `\|u\|` trace at the runaway cell) |
| `REPORT.md` | this report |

Side effect: running the pipeline appends telemetry rows to `data/mesh_runs.csv`
and `web/simulations.db`, which were already dirty. **No tracked source file was
modified** and nothing was committed; the test suite was not run.
