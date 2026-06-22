# Plan: Gas Porosity Analysis for Solidus Pro Physics Model

A post-processing analysis that computes a gas porosity risk score (0-1) per voxel based on mold gas generation and entrapment.

## Core Equations

Gas generation and entrapment are kept as separable internals so visualization can show each layer independently:

```
GasGeneration(i)  = SGI(i)
GasTrap(i)        = E(i)

R_gas(i) = SGI(i) × E(i) × χ × F_niyama(i)
```

| Term | Meaning |
|------|---------|
| SGI(i) | Surface gas generation index — binder decomposition + moisture gas, propagated inward with exponential distance decay |
| E(i) | Entrapment factor — (local_st(i) / max_st)^n — amplifies risk in thick sections |
| χ | Alloy gas sensitivity (CA6NM ~0.75) |
| F_niyama(i) | Niyama coupling factor — amplifies risk where feeding is poor |

### Visualization Layers

Three independent visualization modes are offered:

1. **Gas Generation** — raw SGI field (source + propagation)
2. **Gas Entrapment** — E field (solidification-time-based)
3. **Combined Gas Risk** — full R_gas field

This helps users understand *why* a defect is predicted (source vs. geometry vs. feeding).

### Surface Gas Generation with Source Weighting

Not all surfaces generate equal gas. Core surfaces are the worst offenders.

```
SGI(d) = SGI₀ × w_surface × e^(-d / λ)
```

| Surface Type | Weight (w_surface) |
|---|---|
| Core | 1.0 |
| Mold wall (cope/drag) | 0.6 |
| Open riser | 0.2 |
| Chill | 0.0 |

SGI₀ is computed per surface voxel as the integrated binder decomposition + moisture gas over solidification time. λ is `gas_penetration_depth`.

### Exponential Distance Decay

```
SGI(d) = SGI₀ · e^(-d / λ)
```

Exponential decay captures the physical reality that gas effects diminish rapidly away from the source. The `gas_penetration_depth` parameter λ maps directly to penetration physics and is tunable by material.

### Entrapment Factor

```
E(i) = (local_st(i) / max_st)^n
```

Where `n` is `gas_entrapment_exponent`. Highest E occurs where solidification is slowest — thick sections allow gas to invade and stretch into worm-holes before the shell traps it. Thin sections freeze fast → gas vents away.

### Niyama Coupling

Gas defects are far more likely where feeding is poor. Niyama is already computed by the solver:

```
F_niyama(i) = 1 + α × (1 − N_norm(i))
```

Where:
- N_norm(i) is the locally normalized Niyama criterion (0 = worst feeding, 1 = best)
- α is a coupling strength coefficient (default ~0.5)

Low-Niyama regions become more susceptible to gas invasion. This often matches reality in steel castings where porosity results from gas + shrinkage interaction.

## Alloy Sensitivities

| Alloy | Gas Sensitivity (χ) |
|---|---|
| CA6NM | 0.75 |
| CF8M | 0.60 |
| WCB | 0.50 |
| LCC | 0.45 |

Foundries can tune these per their own casting data.

## Tunable Parameters

| Parameter | Type | Range | Default |
|---|---|---|---|
| ambient_humidity | float | 0-1 | 0.5 |
| binder_type | enum | — | phenolic_urethane |
| core_venting_effectiveness | float | 0-1 | 0.5 |
| gas_penetration_depth (λ) | float | > 0 | 3.0 mm |
| gas_entrapment_exponent (n) | float | 1.0-3.0 | 2.0 |
| niyama_coupling_alpha (α) | float | 0-1 | 0.5 |
| alloy.gas_sensitivity (χ) | float | 0-1 | 0.5 |

## Hotspot Output

In addition to the continuous risk field, generate:

```
gas_porosity_hotspots
```

Using a threshold of `risk > 0.65`. This gives users:
- A heat map of the continuous field
- Discrete hotspot markers

This matches the existing shrinkage hotspot workflow and is generally more actionable than a continuous field alone.

## Files to Create/Modify

1. **NEW** — `physics/gas_porosity.py`
   - `compute_surface_gas_generation()` — per surface voxel, integrates binder decomposition + moisture gas over solidification time, weighted by surface type
   - `compute_interior_gas_risk()` — propagates surface gas inward with exponential decay (e^(-d/λ)) + thick-section entrapment + Niyama coupling
   - `compute_hotspots()` — thresholds risk > 0.65 to produce discrete hotspot markers
   - All Numba-JIT compiled with `parallel=True`

2. `core/materials.py` — Add `gas_sensitivity: float = 0.5` to `AlloyProperties` dataclass. Set CA6NM=0.75, CF8M=0.60, WCB=0.50, LCC=0.45.

3. `core/mold_interface.py` — Add `BinderType` enum, add `binder_type`, `binder_content`, `binder_gas_yield`, `core_venting_effectiveness` to `MoldProperties`. Define decomposition gas yields per binder type. Add `surface_type` classification for voxels.

4. `core/solidification.py` — Add `gas_porosity: bool = False` to `AnalysisOptions`. Add result arrays: `gas_generation`, `gas_entrapment`, `gas_porosity_risk`, `gas_porosity_hotspots`. Add `_calculate_gas_porosity()` method following the established pattern.

5. `core/analysis_strategies.py` — Register gas porosity in `run_post_analyses()` after Niyama (required for coupling), before sand burn-in (order slot 5.5).

6. `core/simulation_pipeline.py` — Map `gas_porosity` option in `create_analysis_options()` and add all four result fields to `build_visualization_payload()`.

7. `ui/main_window.py` — Add "Gas Porosity" checkbox to `AnalysisOptionsDialog`. Register three visualization modes: `"gas_generation"`, `"gas_entrapment"`, `"gas_porosity"`. Add `_visualize_gas_porosity()` method.

8. `ui/visualizer.py` — Add `load_gas_porosity()` method supporting all three sub-modes. Add `load_gas_porosity_hotspots()` for discrete marker rendering.

9. `ui/visualizer.py` — Add gas porosity entries to the colormap/mode dropdown using defect_industry colormap.

10. `tests/` — Add `test_gas_porosity.py` with smoke tests verifying:
    - Gas risk is higher in thick interior voxels than thin surface voxels
    - Increased humidity → increased risk
    - CA6NM sensitivity produces higher risk than WCB for same geometry
    - Core surfaces produce higher SGI than mold wall surfaces
    - Exponential decay drops SGI below 5% at d > 3λ
    - Niyama coupling amplifies risk in poor-feeding regions

## Computational Complexity

All steps are O(N) with Numba parallelization:

| Step | Complexity |
|---|---|
| Surface Identification | O(N) |
| Distance Propagation | O(N) |
| Risk Calculation | O(N) |

For a 200³ voxel mesh (8M voxels), expected runtime is measured in seconds, not minutes.

## What This Does NOT Require

- No changes to the solidification loop itself (it's a post-analysis)
- No new dependencies (Numba JIT already available)
- No changes to the voxel mesh generation
- Works with all three solver modes (uses `local_solidification_time` and `niyama` which are always available)

## Risk Level

**LOW** — This is an additive post-analysis following the exact pattern of 9 existing analyses. No solver changes. The gas model is intentionally simple (engineering/heuristic) and all parameters have defaults.
