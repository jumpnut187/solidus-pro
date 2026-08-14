# Mathematical Calculation Audit Checklist

This checklist captures the concrete work needed to bring the simulator's mathematical calculations into a defensible state.

It is based on the current code audit of:

- `core/solidification.py`
- `core/solidification_loop.py`
- `core/acceleration.py`
- `core/materials.py`
- `core/stress_analysis.py`
- `core/microstructure.py`
- `core/riser_design.py`
- `physics/solidification_physics.py`
- `physics/shrinkage.py`
- `physics/niyama.py`
- `physics/heat_transfer.py`

The goal is not only to remove obvious bugs, but to separate:

- true mathematical defects
- unit inconsistencies
- intentional heuristics
- undocumented assumptions that currently look like physics but behave like tuning rules

## 1. Critical Math Defects To Fix First

### Scheil fraction solid implementation

- [x] Replace the current Scheil fraction-solid implementation in `physics/solidification_physics.py`.
- [x] Remove or rewrite the closed-form equation currently used in:
  - `SolidificationModel._scheil()`
  - `FractionSolidCalculator.scheil()`
  - `compute_fraction_solid_scheil_parallel()`
- [x] Confirm the replacement formulation is physically valid for the alloy model being used.
- [x] Ensure the result is:
  - monotonic as temperature decreases
  - bounded in `[0, 1]` without relying on clipping to hide bad math
  - consistent with `f_s = 0` at liquidus and `f_s = 1` at or below solidus
- [x] Decide whether to:
  - implement a proper iterative Scheil path, or
  - remove the `SCHEIL` option until a defensible formulation is in place
- [x] Add regression tests that fail if intermediate temperatures produce negative fraction solid before clipping.
- [x] Add regression tests that fail if the Scheil path collapses into a near-step function at solidus.

Why this is first:

- The current implementation produces invalid negative intermediate values.
- That makes `SCHEIL` mathematically unreliable.
- `BACK_DIFFUSION` also inherits the issue because it blends with the current Scheil result.

### Feeding effectiveness uses final solid fraction instead of feedable mush state

- [x] Redesign `SolidificationEngine._calculate_feeding_effectiveness()` in `core/solidification.py`.
- [x] Stop using the final `self.mesh.voxel_states` field as the fraction solid input for Darcy resistance.
- [x] Decide which state is actually intended for feeding calculations:
  - fraction solid at the moment the voxel becomes feeding-limited
  - peak mushy resistance during freezing
  - a time-window average over the vulnerable feeding interval
- [x] Store the required transient state during the solidification loop instead of reconstructing feeding from the fully solid end state.
- [x] Prevent the current failure mode where permeability is evaluated near `f_s = 1`, driving Darcy pressure drop to unrealistic values.
- [x] Re-check the pressure comparison:
  - `P_ferro = rho * g * h`
  - `delta_P = mu * L * v / K`
- [x] Confirm that the path length `L` reflects an actual feeding path, not just distance to surface.
- [x] Confirm the chosen velocity estimate is justifiable and documented.
- [x] Add regression tests with representative WCB cases showing:
  - moderate mushy fractions remain feedable under reasonable head
  - nearly solid fractions become strongly feeding-limited
  - surface voxels remain effectively fed

Why this is first:

- The current logic can mark almost everything as feeding-blocked because it evaluates permeability too late.
- That contaminates shrinkage and porosity results downstream.

### Distortion magnitude has unit inconsistency

- [x] Fix `compute_distortion_magnitude()` in `core/stress_analysis.py`.
- [x] Decide whether `distortion_vectors` represent:
  - directions only, or
  - already strain-weighted displacement tendencies
- [x] Remove the current double application of strain if vectors are already weighted by `diff_strain`.
- [x] Re-derive the formula so the output has displacement units, not strain-squared times length.
- [x] Add a simple two-voxel analytical regression case with known expected displacement scaling.
- [x] Verify that doubling the strain doubles distortion, rather than quadrupling it.

Why this is first:

- The current function over-suppresses or distorts displacement magnitude numerically.
- The units are not internally consistent.

### Porosity probability can exceed 1.0

- [x] Clamp `self.porosity_probability` to `[0.0, 1.0]` in `core/solidification.py`.
- [x] Decide whether the field is intended to be:
  - probability
  - severity
  - unbounded heuristic score
- [x] Rename the variable if it is not actually a probability.
- [x] Add regression tests that fail if any porosity output exceeds `1.0`.

Why this is first:

- A value named probability must not exceed 1.
- This is a correctness and interpretation problem, not just presentation.

### Latent heat rate in `get_state()` uses cumulative time instead of last `dt`

- [x] Fix `SolidificationModel.get_state()` in `physics/solidification_physics.py`.
- [x] Store the last update time step or the last computed latent heat field.
- [x] Make `get_state().latent_heat_rate` consistent with what `update()` returned for the most recent step.
- [x] Add regression tests over multiple time steps to ensure repeated equal solid-fraction increments produce equal latent-heat rates.

Why this is first:

- The current state object reports the wrong instantaneous rate after the first step.
- That can mislead any later thermal coupling or diagnostics.

## 2. Consistency Problems To Resolve Next

### Feeding distance multiplier values disagree between code and documentation

- [x] Reconcile the values in `core/materials.py` with the values documented in:
  - `README.md`
  - `QUICKSTART.md`
- [x] Decide whether the intended baseline is:
  - WCB `6.0`, CF8 `5.5`, C958 `8.0`, or
  - WCB `3.0`, CF8 `2.5`, C958 `4.0`
- [x] Confirm whether the code value is:
  - literal feeding-distance multiplier
  - doubled empirical radius
  - UI-facing simplified label
- [x] Update code comments and docs so one meaning exists everywhere.

Why this matters:

- Shrinkage, riser influence, and user guidance all rely on this parameter.
- Right now the same concept has conflicting values depending on where you look.

### Clarify whether “feeding distance” means distance to surface or distance to feeder

- [x] Audit every use of `self.feeding_distances` in `core/solidification.py`.
- [x] Rename it if necessary.
- [x] Separate these concepts if both are needed:
  - distance to nearest surface
  - distance to riser or feeder
  - true connected feeding path length through the mush
- [x] Prevent pressure-drop math from using a surface-distance proxy when a feeder-path length is actually required.

Why this matters:

- Several formulas currently use one field for multiple physical meanings.
- That makes the outputs hard to trust even when the code runs.

### Clarify whether `voxel_states` means fraction solid, binary solid flag, or progress proxy

- [x] Document exactly what `self.mesh.voxel_states` represents throughout the solver.
- [x] Check every place that assumes:
  - `0 = liquid`
  - `1 = solid`
  - intermediate values are meaningful as fraction solid
- [x] Separate “front progression state” from “thermodynamic fraction solid” if they are not truly the same quantity.

Why this matters:

- Many downstream calculations treat `voxel_states` as if it were thermodynamic fraction solid.
- If it is really a heuristic progress field, permeability, hot tearing, and air-gap calculations are all on shaky footing.

## 3. Shrinkage Model Cleanup

### Distinguish heuristic scoring from physics-based prediction

- [x] Review `_calculate_shrinkage()` in `core/solidification.py`.
- [x] Split the logic into clearly named components:
  - geometry-driven heuristic terms
  - feeding correction terms
  - chill correction terms
  - riser pipe logic
- [x] Label the final output honestly as either:
  - probability
  - relative risk
  - severity score
- [x] Remove any implication that the current weighted sum is first-principles physics if it is actually heuristic.

### Re-check the use of modulus-to-thickness approximation

- [x] Review the assumption `local_thickness = 2 * modulus`.
- [x] Decide whether this is:
  - an acceptable rough proxy for local section size, or
  - too shape-dependent to be used as a direct thickness substitute
- [x] Document where the approximation is safe and where it is not.

### Re-check riser shrinkage pipe logic

- [x] Review the riser cone and pipe suppression logic for:
  - radial falloff
  - height thresholds
  - modifier range
- [x] Confirm the result does not artificially suppress legitimate riser shrinkage outside a narrow pipe zone.
- [x] Add one or more visual regression cases for:
  - simple cylindrical riser
  - riser-to-casting neck
  - multi-riser geometry

## 4. Niyama Model Review

### Keep the top-level formula, review the field construction

- [x] Keep the core definition `Ny = G / sqrt(R)` unless a stronger model change is justified.
- [x] Review whether `G` and `R` in `core/solidification.py` are adequate approximations or should be replaced by thermal-field-derived values.
- [x] Decide whether to use the newer `physics/niyama.py` model instead of the current heuristic engine-level implementation.
- [x] Ensure unit reporting is consistent:
  - if `G` is in `C/in`, thresholds must be interpreted in compatible units
  - if thresholds come from literature in SI-style units, convert explicitly
- [x] Document the thresholds per alloy and their source basis.

### Review conductivity and heat-capacity scaling choices

- [x] Re-check the conductivity factor applied to gradient construction.
- [x] Re-check the specific-heat factor applied to cooling rate construction.
- [x] Confirm the sign and physical intuition are both correct.
- [x] Add comments explaining whether the scaling is empirical or physically derived.

## 5. Stress And Hot Tear Model Review

### Re-check thermal strain assumptions

- [x] Review `compute_thermal_strain_field()` in `core/stress_analysis.py`.
- [x] Confirm the current “earlier solidification means more cooling” factor is acceptable as a heuristic.
- [x] If not, replace it with a temperature-history-based strain estimate.

### Re-check constraint strain scaling

- [x] Review the neighbor time-difference scaling in:
  - `core/stress_analysis.py`
  - `core/acceleration.py`
- [x] Confirm the current normalized time-difference approach does not overstate strain in sparse or noisy neighborhoods.

### Align hot tear analysis with alloy-specific BTR data

- [x] Verify that the BTR window and coherency assumptions from `core/materials.py` are applied consistently across hot-tear calculations.
- [x] Check whether hot-tear logic should use:
  - actual fraction solid history, or
  - the existing solidification-time heuristics
- [x] Add alloy-specific regression scenarios to ensure WCB, CF8, and C958 do not all behave as scaled copies of each other.

## 6. Microstructure Model Review

### Review SDAS and property correlations

- [x] Re-check `core/microstructure.py` for:
  - SDAS exponent usage
  - clamping range
  - property-correlation signs
- [x] Confirm the Hall-Petch-style UTS correlation is appropriate for each alloy family represented.
- [x] Confirm elongation degradation does not produce unrealistic values for common cooling rates.
- [x] Add benchmark cases with representative cooling rates:
  - low cooling rate
  - medium cooling rate
  - high cooling rate

### Review cooling-rate estimation from solidification time

- [x] Audit `estimate_cooling_rate_from_solidification()`.
- [x] Decide whether the current neighbor-time refinement is acceptable or too ad hoc.
- [x] Clarify that this is an estimate when no thermal simulation is driving the field.

## 7. Heat Transfer And Phase Model Integration

### Decide whether the repo has one thermal model or two competing ones

- [x] Compare:
  - `physics/heat_transfer.py`
  - `physics/solidification_physics.py`
  - `core/solidification.py`
  - `core/thermal_simulation.py`
- [x] Decide whether the project should:
  - keep the simplified geometry-driven engine as the main path, or
  - migrate analyses onto the thermal and phase-field modules
- [x] Remove or clearly demote dead-end math modules if they are not used by the application.

Why this matters:

- Right now the codebase mixes a heuristic solidification engine with more physics-like modules.
- That makes it difficult to know which formulas actually govern the user-visible outputs.

## 8. Test Coverage To Add

### Core regression tests

- [x] Add a Scheil monotonicity test.
- [x] Add a Scheil bounds test.
- [x] Add a back-diffusion interpolation test.
- [x] Add a latent heat state-reporting test across multiple time steps.
- [x] Add a porosity clamping test.
- [x] Add a feeding-effectiveness test for:
  - low `f_s`
  - mushy `f_s`
  - near-solid `f_s`
- [x] Add a distortion unit-consistency test.
- [x] Add a feeding-distance semantics test if surface distance and feeder distance are split.

### Scenario tests

- [x] Create one minimal casting-with-riser case.
- [x] Create one minimal casting-with-chill case.
- [x] Create one isolated-hot-spot case.
- [x] Create one thin-to-thick transition case for hot tearing.
- [x] Use these as deterministic regression fixtures.

### Output sanity tests

- [x] Assert that probabilities remain in `[0, 1]`.
- [x] Assert that fraction solid remains monotonic in time.
- [x] Assert that solidification time fields do not contain impossible values after a nominal run.
- [x] Assert that computed feeding effectiveness is not universally near-zero for nominal feedable cases.

## 9. Documentation Work

### Make the model limitations explicit

- [x] Update `README.md` to separate:
  - heuristic parts of the solver
  - physics-based parts
  - experimental modules
- [x] Document the exact meaning of:
  - `voxel_states`
  - `feeding_distances`
  - `shrinkage_probability`
  - `porosity_probability`
  - `niyama_values`
- [x] Add a “Model Assumptions” section.
- [x] Add a “Known Simplifications” section.

### Record formula provenance

- [x] Add source notes for:
  - Niyama criterion
  - Darcy feeding resistance logic
  - Chvorinov time scaling
  - SDAS power-law correlations
  - any riser or chill heuristics inspired by foundry rules of thumb

## 10. Validation Checklist Before Calling The Math Stable

- [x] `python3 tests/verify_chill_decay.py`
- [x] `python3 -m unittest tests/test_fraction_liquid_final_snapshot.py`
- [x] Add and run new regression tests for all fixed math defects
- [x] `python3 -m compileall config core physics ui web`
- [x] `python3 -m mypy config core physics ui`
- [x] Run one end-to-end simulation per alloy:
  - WCB
  - CF8
  - C958
- [x] Compare output trends against expectation:
  - thicker sections solidify slower
  - chills reduce nearby shrinkage
  - risers reduce nearby shrinkage but show pipe behavior
  - Niyama risk decreases where directional solidification improves

## 11. Optional But Recommended Structural Cleanup

### Reduce duplication

- [x] Consolidate duplicate formula logic between:
  - `core/solidification.py`
  - `physics/shrinkage.py`
  - `physics/niyama.py`
  - `physics/solidification_physics.py`
- [x] Make one implementation authoritative for each major calculation.

### Improve naming

- [x] Rename outputs that are heuristic severity scores but currently called probabilities.
- [x] Rename distance fields whose meaning has drifted over time.

### Improve observability

- [x] Add targeted debug logs for:
  - permeability range
  - pressure-drop range
  - ferrostatic-pressure range
  - feeding-effectiveness histogram
  - fraction-solid histogram at the time feeding is assessed

## 12. Confirmed Findings Already Observed

- [x] Scheil implementation produces invalid negative intermediate values before clipping.
- [x] Feeding effectiveness currently uses end-state voxel solid fraction.
- [x] Distortion magnitude currently multiplies strain by a strain-weighted vector magnitude.
- [x] Porosity probability can exceed 1.0.
- [x] `SolidificationModel.get_state()` reports latent heat rate using cumulative time instead of the last time step.
- [x] Documentation and code disagree on feeding-distance multipliers.
- [x] Current baseline checks were run:
  - `tests/verify_chill_decay.py` passed
  - `tests/test_fraction_liquid_final_snapshot.py` passed
  - `compileall` passed
  - `mypy` currently fails with many existing typing issues outside the narrow math defects

## 13. Reference Notes

These references informed the audit and should be used when implementing the fixes:

- [x] NIST Scheil solidification notes: https://www.ctcms.nist.gov/~kattner/solidifc/scheil.html
- [x] MDPI Materials article discussing Niyama criterion usage and threshold ranges: https://www.mdpi.com/1996-1944/19/1/198

## 14. Suggested Execution Order

- [x] Step 1: Fix Scheil and back-diffusion math.
- [x] Step 2: Fix latent heat state reporting.
- [x] Step 3: Fix porosity probability bounds and naming.
- [x] Step 4: Fix distortion magnitude units.
- [x] Step 5: Redesign feeding-effectiveness timing and path semantics.
- [x] Step 6: Reconcile feeding-distance multipliers and docs.
- [x] Step 7: Add focused regression tests for all fixed defects.
- [x] Step 8: Re-run compile, tests, and mypy.
- [x] Step 9: Re-check alloy-by-alloy simulation behavior.
- [x] Step 10: Update docs to reflect what is truly physics-based versus heuristic.

## 15. Round 2 Physics Audit (Aug 2026) — Completed

Second audit pass over the thermal-coupled solver path
(`core/solidification_loop.py`, `physics/heat_transfer.py`,
`physics/solidification_physics.py`, `core/solidification.py`) and the
supporting `core/solidification_params.py` tunables. Items A cover thermal
solver correctness; items B cover physical model improvements. Each fix is
covered by new regression tests.

### A. Thermal solver correctness

- [x] **A1 — Laplacian stencil fixed.** The old kernel was a 26-neighbor
  equal-weight average normalized by neighbor count, which reduced the
  effective diffusivity to roughly 0.35x the stated value and made the
  error position-dependent. Replaced with an isotropic weighted stencil:
  face neighbors 0.5, corner neighbors 0.125, edge neighbors excluded
  (they cancel exactly in the weighted sum). The weighted difference form
  is exact for all quadratic fields and no longer divides by neighbor
  count. Implemented in `_step_internal()` and
  `compute_laplacian_parallel()` in `physics/heat_transfer.py`.
- [x] **A2 — Per-voxel cell sizes and diffusivity.** `HeatTransferModel`
  now accepts `voxel_sizes` (per-voxel cell size) and
  `diffusivity_overrides` (per-voxel thermal diffusivity), making the
  solver octree-aware on graded meshes. The stencil remains approximate on
  graded meshes (documented limitation); it is exact on uniform grids.
- [x] **A3 — Same-step latent heat.** Latent heat is released in-step:
  `_run_thermal_coupled_integration()` applies the latent-heat rate as a
  direct temperature correction on the same step (predictor scheme) rather
  than through a heat source that only takes effect on the next step. No
  one-step lag; per-step energy is conserved.
- [x] **A4 — Thermal-coupled mode is purely temperature-driven.** The
  heuristic vertical solidification bias was removed from the
  thermal-coupled path (`core/solidification_loop.py`); the legacy
  schedule path is unchanged.
- [x] **A5 — True Scheil implementation.** `_scheil()` and
  `FractionSolidCalculator.scheil()` now implement the true Scheil
  equation using per-alloy `melting_point_solvent` and
  `partition_coefficient`, with a lever-rule solidus completion
  (`f_s = 1` at/below solidus) and an optional eutectic cap. Falls back to
  a bounded surrogate when the solvent melting point is unavailable and to
  the lever rule for `k >= 1`. Module docstring corrected.
- [x] **A6 — Robin boundary flux corrected.** Convection/interface
  boundaries in `_apply_boundaries()` now evaluate the flux against the
  PRE-update temperature and use the full cell size `dx` (previously the
  boundary cell was lumped as `dx/2` and the flux used the new
  temperature).

### B. Physical model improvements

- [x] **B4 — Chills are finite heat sinks.** Chill voxels receive a graphite
  diffusivity override (k = 100 W/(m·K), rho = 1800 kg/m³, cp = 710
  J/(kg·K) → alpha = 7.83e-5 m²/s) and an ambient-HTC interface boundary
  instead of being pinned at a fixed temperature. Tunable via
  `thermal_boundary.chill_ambient_htc`. See `_initialize_thermal_models()`
  and `_configure_thermal_boundaries()` in `core/solidification.py`.
  Follow-up fix: the CFL stability bound (`get_stable_timestep`) now
  includes per-voxel diffusivity override values in `_get_alpha_max()`;
  without it the explicit scheme ignored the ~7x graphite alpha, went
  unstable at the chill interface (+/-inf within ~300 steps, surfacing as
  NaN fraction solid and a ValueError in the progress callback).
- [x] **B5 — Darcy feed velocity is shrinkage-driven.** Feeding
  effectiveness uses `v = beta * R` with `beta = rho_s/rho_l - 1`
  (volumetric shrinkage) and `R = cooling_rate / |grad T|` (local
  solidification-front speed); falls back to a constant 0.5 mm/s when
  thermal fields are unavailable. See `_calculate_feeding_effectiveness()`
  in `core/solidification.py`.
- [x] **B6 — Temperature-dependent properties.** The default thermal path
  enables temperature-dependent k(T), cp(T), and rho(T) via
  `property_curves` (reference points spanning room temperature through
  above-liquidus). See `_initialize_thermal_models()` and
  `_get_temperature_dependent_diffusivity()`.
- [x] **B7 — Air-gap HTC decay.** Mold-interface HTC decays exponentially
  on cast, riser, and riser-post surfaces:
  `h(t) = h0 * (floor + (1 - floor) * exp(-t/tau))` with
  `air_gap.htc_floor_fraction` defaulting to 0.35.
- [x] **B8 — NaN handling for unfinished voxels.** Voxels still liquid at
  the time cap carry NaN solidification times. Downstream analyses treat
  NaN as max_time (last-to-freeze) in `_compute_shrinkage_base_risk()` and
  `_calculate_isolation_score()`, and `_generate_statistics()` excludes
  unfinished voxels from mean/max solidification-time statistics.

### Supporting changes

- [x] New tunable parameters: `air_gap.htc_floor_fraction` (default 0.35,
  clamp 0.05–1.0) and `thermal_boundary.chill_ambient_htc` (default
  100 W/(m²·K), clamp 1–10000) in `core/solidification_params.py`.
- [x] New regression tests: `tests/test_scheil_regression.py`,
  `tests/test_heat_transfer_physics.py`,
  `tests/test_heat_transfer_numba_backend.py`,
  `tests/test_thermal_loop_regressions.py`,
  `tests/test_thermal_solver_curves.py`,
  `tests/test_thermal_boundary_config.py`,
  `tests/test_darcy_velocity.py`,
  `tests/test_unfinished_voxel_masking.py`; updated
  `tests/test_math_audit_regressions.py`,
  `tests/test_thermal_coupled_integration.py`,
  `tests/test_solidification_params.py`.
- [x] Benchmark script: `tests/verify_thermal_benchmarks.py`.
