# Compute Scaling Plan

How to make simulations finish faster, given the solver we actually have.

Written in response to: *"can two computers on the same LAN combine CPU power
to run one simulation faster?"* Short answer — no, not without an MPI rewrite
that wouldn't pay off at two nodes. The real speedup is already sitting in the
tree, mis-wired. This document ranks the options and plans the work.

`docs/ARCHITECTURE.md` describes the live path; `AGENT.md` covers entrypoints
and verify commands.

---

## 1. Measured starting point

All numbers from this machine (M5 Pro, 12P+6E, 20-core GPU, 48 GB unified),
project venv, `n` = voxel count, per-timestep wall time for the solidification
front update. Bench scripts were throwaway; reproduce with §6.

| n voxels | Numba CPU (17 threads) | Metal (as shipped) | Ratio |
|---:|---:|---:|---:|
| 50,000 | 0.52 ms | 5.75 ms | **0.09x** |
| 200,000 | 1.92 ms | 9.62 ms | **0.20x** |
| 800,000 | 8.18 ms | 15.85 ms | **0.52x** |

**The Metal path is currently 2–11x slower than the CPU path.** It is not a
latent 10x win being withheld; it is a regression that happens to be disabled
by default.

Cause is a fixed per-call cost, not the kernel:

| n voxels | Metal, as shipped |
|---:|---:|
| 64 | 4.70 ms |
| 1,000 | 4.48 ms |
| 10,000 | 5.17 ms |

~4.5–5 ms independent of `n`. `_update_solidification_metal` declares its
`@ti.kernel` *inside the method body* (`core/gpu_compute.py:494`), so Taichi
re-traces and re-compiles a fresh closure on **every timestep**. A run of
5,000 steps pays that 5 ms roughly 5,000 times — ~25 s of pure compile churn.

Hoisting the kernel to module scope and timing only launch + host↔device copy:

| n voxels | launch+copy floor | CPU | Implied ceiling |
|---:|---:|---:|---:|
| 50,000 | 0.80 ms | 0.52 ms | 0.65x (CPU still wins) |
| 200,000 | 1.19 ms | 1.92 ms | 1.6x |
| 800,000 | 2.59 ms | 8.18 ms | **3.2x** |

Caveat, stated plainly: that floor was measured with a *simplified* kernel
(no 26-neighbour feeding-connectivity scan). The real hoisted kernel will land
between the floor and the current number. Treat 3.2x at 800k as an optimistic
bound and **~2x as the plausible target** — to be confirmed in Phase 1, not
assumed.

Also measured: `tests/test_metal_backend_parity.py` — **5 passed in 0.93 s**.
Numerical parity with the CPU reference already holds. The problem is purely
performance and wiring.

---

## 2. Why not two machines

The solver is shared-memory by construction:

- Every hot kernel is `@jit(nopython=True, parallel=True)` with `prange` over
  one contiguous set of numpy arrays (`core/acceleration.py:121` onward).
  Numba threads share an address space; they cannot reach another host's RAM.
- Those kernels are driven by a strictly sequential explicit time loop
  (`core/solidification_loop.py:517`, `:921`) at an adaptive-CFL step around
  `dt = 0.1 s` (`core/solidification.py:283`, `:550`). Step *n+1* needs every
  neighbour value from step *n*.

Splitting one run across hosts therefore requires domain decomposition plus a
halo exchange **every timestep**. There is no MPI anywhere in the tree.

The economics don't work at two nodes. Per-step CPU kernel time is 0.5–8 ms
(§1); a gigabit round trip is 0.1–0.5 ms before payload, and the exchange is a
hard synchronisation point — both hosts idle at the slowest one, every step.
Expect 1.2–1.4x at best, frequently <1x. Distributed CFD earns its keep at 8+
nodes on 10 GbE or InfiniBand, not at two over a LAN.

**Decision: not pursued.** Recorded here so it isn't re-proposed.

---

## 3. What the second machine *is* good for (today, zero code)

The web queue already fair-shares cores across concurrent simulations
(`web/services/fair_share.py`, `web/services/simulation_queue.py`, tuned by
`CASTSIM_MAX_CONCURRENT_SIMULATIONS`). Run the app on both machines and split
*different jobs* across them:

- parameter sweeps, riser/chill variants
- batches of separate parts
- surrogate dataset generation (`generate_samples.py` → `data/surrogate_dataset/`)

That is near-linear **throughput** scaling — 2x runs per hour — which for sweep
work is indistinguishable from 2x speed. It will not make any single run finish
sooner. No work required beyond starting the second instance; both modes already
bind `0.0.0.0` (`web/netinfo.py`).

Optional later nicety, explicitly out of scope here: a shared job queue so one
UI dispatches to both hosts. Worth it only if manual splitting becomes a chore.

---

## 4. Wiring defects found while measuring

Independent of performance, four things are inconsistent. Fix these regardless
of whether Phase 2 proceeds.

1. **Incremental silently wins over GPU.** The loop is
   `if _use_incremental ... elif _use_gpu ... else CPU`
   (`core/solidification_loop.py:529`, `:540`). Enabling both means GPU never
   runs, with no warning. Should warn, or define precedence deliberately.
2. **Desktop hardcodes GPU off.** `ui/workers.py:330` passes `use_gpu=False`
   unconditionally, while `:89` honours the Performance dialog. One of those is
   a bug.
3. **Defaults disagree across entrypoints.** `config/settings.py:95` says
   `use_gpu: bool = True`; `PipelineInputs` (`core/simulation_pipeline.py:72`)
   and `AnalysisOptions` (`core/solidification.py:131`) default `False`; web
   seeds `True` (`web/repository.py:613`). Given §1, `False` is currently the
   *correct* default — but it should be one deliberate value, not four accidents.
4. **Stale comment.** `core/solidification.py:131` says GPU "requires cupy or
   pyopencl". Metal via MLX + Taichi is the working backend; neither cupy nor
   pyopencl is installed.

---

## 5. Plan

Sequenced so each phase is independently shippable and independently revertible.
Stop after any phase if the numbers don't justify continuing.

### Phase 0 — Wiring and honesty (small, no perf risk)

- Fix the four items in §4.
- Make the GPU/incremental precedence explicit and logged.
- Keep `use_gpu` defaulting **off** everywhere until Phase 1 proves otherwise.

Verify: `python -m pytest tests/ -q` shows no new failures against the known
baseline (see `AGENT.md`; the pre-existing failures are not ours).

### Phase 1 — Hoist the Taichi kernel (the actual fix)

The whole of §1's regression is one structural mistake.

- Lift the `@ti.kernel` out of `_update_solidification_metal`
  (`core/gpu_compute.py:494`) to module scope, compiled once per process.
- Keep the full feeding-connectivity logic — parity is non-negotiable.
- Pass `n_voxels` as a kernel argument rather than closing over it (the current
  closure capture is *why* it can't be hoisted as-is).
- Reuse device-side buffers across steps instead of `states.copy()` per call.

Verify:
- `python -m pytest tests/test_metal_backend_parity.py -q` — must stay 5 passed.
  Parity gates this phase; a faster kernel that changes results is a failure.
- Re-run the §6 benchmark. **Gate: Metal must beat CPU by >1.5x at n=800k.**
  If it doesn't, stop — revert and close out at Phase 0 plus §3.

### Phase 2 — Size-aware dispatch

CPU wins below roughly 100–200k voxels even with a perfect kernel (§1); fixed
launch and copy cost cannot be amortised over a small mesh.

- Add a crossover threshold in `update_solidification_gpu`
  (`core/gpu_compute.py:704`): below it, delegate to the Numba path — the `else`
  branch at `:754` already does exactly this, so the fallback exists.
- Derive the threshold from measurement, not a guess. Make it overridable
  (`CASTSIM_GPU_MIN_VOXELS`), consistent with the existing env-var conventions
  in `web/services/simulation_queue.py:66`.
- Only then consider flipping the default to on, for meshes above the threshold.

Verify: benchmark sweep across n ∈ {25k, 50k, 100k, 200k, 400k, 800k} showing
the dispatcher picks the faster backend at every size.

### Phase 3 — Optional, only if Phase 1 clears its gate

- `_voxel_containment_metal` has the same in-function-kernel defect
  (`core/gpu_compute.py:645`). It runs once per mesh build rather than per
  timestep, so it costs ~5 ms total instead of ~25 s — worth fixing for
  consistency, not for speed.
- Move the geometry/setup GPU calls onto reused buffers too
  (`core/voxel_mesh.py:975`, `core/solidification.py:1454`).
- Consider keeping `voxel_states` resident on-device across timesteps so the
  loop stops round-tripping through host memory every step. This is the largest
  remaining win and the largest change; it touches the loop's ownership of
  `engine.mesh.voxel_states` (`core/solidification_loop.py:544`) and deserves
  its own plan.

---

## 6. Reproducing the benchmark

No committed harness for this yet; Phase 1 should add one under `tests/` as a
standalone `verify_*.py` script (see `AGENT.md` — `verify_*.py` are scripts, not
suites). Until then, the shape that produced §1:

```
source venv/bin/activate
# random n-voxel state, 26 neighbours, warm both paths once, then time N reps of:
#   core.acceleration.update_solidification_with_feeding_connectivity   (CPU)
#   core.gpu_compute.gpu_compute.update_solidification_gpu              (Metal)
# report ms/step at n = 50k / 200k / 800k
```

Two things that will mislead you if skipped:

- **Warm up both paths before timing.** Numba JITs on first call and Taichi
  compiles on first launch; an unwarmed first call measures the compiler.
- **`CASTSIM_THREAD_BUDGET` silently caps the CPU side.** If it's set in the
  environment, the CPU baseline is throttled and every GPU ratio looks better
  than it is. Check it before trusting a comparison.

---

## 7. Summary

| Option | Single-run speedup | Effort | Verdict |
|---|---|---|---|
| Second machine, one run (MPI) | 1.0–1.4x | Very high | **No** — §2 |
| Second machine, separate jobs | 1.0x (2x throughput) | None | **Yes, today** — §3 |
| Fix + enable Metal | ~2x at 800k voxels (target) | Low–moderate | **Yes** — Phases 0–2 |
| Device-resident state | Unknown, larger | High | Later, own plan |

The second computer is worth using — for more runs at once, not for one faster
run. The single-run win is a ~5 ms per-timestep compile bug in the Metal
backend, and the parity tests to prove a fix are already written and passing.
