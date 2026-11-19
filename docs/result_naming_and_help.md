# Result naming and help popovers

## Names and semantics (identical on both apps)

| Mode id | Desktop / web label | Array | Default threshold |
|---|---|---|---|
| `shrinkage` | Shrink | `engine.get_shrink_display_intensity()` = `shrinkage_probability`, chill and orphan-metal points 0, clipped 0-1. No porosity mixed in. | 70 % |
| `porosity` | Microporosity | `engine.get_microporosity_display_intensity()` = `porosity_probability`, chill and orphan-metal points 0, clipped 0-1. | 70 % |

- `get_shrinkage_display_intensity()` (max of the two) is kept and documented as the old combined field. Remaining callers: `get_shrinkage_intensity`, `core/defect_regions.compute_merged_shrink_display_intensity` (tests only) and regression tests.
- Desktop offers Microporosity when the Shrink analysis ran and `porosity_probability` exists. Its view reuses the Shrink path (envelope, threshold auto-lower, pocket inspector), legend "Microporosity Risk". Shrink legend is "Shrink Risk".
- Web JSON preview (`core/simulation_pipeline.py`), artifact (`web/payloads.build_fork_artifact`) and grid sidecar (`web/payloads._full_grid_source`) now take the same two engine methods; engines without them fall back to the raw attribute. `shrinkage_stats` (max/mean/affected) are now on Shrink only.
- Defaults: desktop `SHRINKAGE_DEFAULT_THRESHOLD_PERCENT = 70`, new `MICROPOROSITY_DEFAULT_THRESHOLD_PERCENT = 70`; web `DEFAULT_THRESHOLD` 70/70 unchanged.

## Back-compat

- `ui/result_types.LEGACY_RESULT_TYPE_LABELS` maps "Shrinkage/Porosity" to "Shrink". Applied in `ui/project_state._normalize_ui_state` (saved projects) and in `MainWindow` project load and `_on_result_type_changed`.
- Web mode ids did not change (`shrinkage`, `porosity`), so stored web state needs no migration.

## Help text

- One source: `core/result_help.json` (mode id -> paragraph). Python reads it through `core/result_help.py`. The web imports it at build time in `viewState.ts` as `RESULT_MODE_HELP: Record<ResultModeId, string>`, so a missing mode fails `tsc`. `vite.config.ts` allows the dev server to serve that one file only (checked: JSON 200, other repo files 403).
- Web: info button beside the Result type select opens a popover (`role="tooltip"`, `aria-controls`/`aria-describedby`, closes on Escape with focus back to the button, outside click, or mode change). The select has `aria-describedby` pointing at a hidden copy of the selected help. Each `<option>` has the help as `title`. Styles in `styles/results.css` follow `.menu-pop`.
- Desktop: `ui/result_types.py` (Qt-free) holds the label -> mode table that replaced the two inline dicts. Every combo item carries rich-text help in `Qt.ToolTipRole` (rich text so Qt wraps it). A `QToolButton` (ⓘ) next to the combo shows the selected help via `QToolTip`, and its own tooltip follows the selection.

| Mode | Help text |
|---|---|
| `shrinkage` | Macro-shrinkage risk from 0 to 1 (shown as %): where metal is likely to leave a shrink cavity. High values mark metal that freezes late, is cut off from feed metal, or sits in a thick section out of riser reach. Computed from isolation, section size and height, reduced near chills and within riser feeding reach, raised where Darcy feeding through the mush fails, then scaled by the Shrink Scalar. Default envelope threshold 70 %; chills always read 0. |
| `porosity` | Microporosity risk from 0 to 1 (shown as %): fine, dispersed interdendritic porosity rather than an open cavity. High values mark metal with a low Niyama value, a blocked feed path, or a poor Darcy pressure margin. Computed as 0.45 × Niyama severity + 0.35 × feed-path blockage + 0.20 × Darcy deficit, weighted by the local Shrink risk, then scaled by the Shrink Scalar. Default envelope threshold 70 %; chills always read 0. |
| `feed_modulus` | Local section size in inches: the distance from each point to the nearest mould-contact surface. Large values are thick sections that freeze late and need feeding, so feed metal should run from high to low modulus toward the risers. Measured from the mesh geometry; chills read 0. The threshold is a percent of the modulus range (default 1 %). |
| `hotspot` | Hot spot intensity from 0 to 1: regions that freeze last or become isolated liquid pools. High values are thick, late, cut-off areas that need a riser, a chill or a design change. Computed from solidification time (35 %), distance to surface (20 %), section modulus (15 %) and liquid isolation (30 %); only points in the last 40 % of freezing or strongly isolated score above 0. Default threshold 30 %. |
| `solidification_time` | Time in seconds at which each point finished freezing (reached the solidus). The highest values freeze last, and the last metal to freeze should sit in a riser, not in the casting. Taken from the thermal solve; points still above the solidus at the time cap count as last. Use the time bar to see what is solid at each moment; there is no threshold. |
| `niyama_raw` | Niyama criterion G / sqrt(R) in °C·√s/mm, with the thermal gradient G and cooling rate R taken at 90 % solid. Low values mean a shallow gradient for the cooling rate and predict centreline shrinkage and microporosity; high values are safe. The critical value depends on the alloy, for example 1.0 for WCB, 0.7 for CF8 and 1.2 for C958. The envelope encloses values above the threshold (default 10 % of the range), so it wraps the safer metal. |
| `niyama` | Niyama risk from 0 to 1: how far the Niyama value G / sqrt(R) falls below the alloy's critical value. High values mark interior metal prone to centreline shrinkage and microporosity; surface points and chills read 0. Computed as 1 - Niyama / critical value, with G and R taken at 90 % solid. Default threshold 30 % (the desktop uses 20 % for CF8 and 40 % for C958). |
| `combined_feed_risk` | Combined feed risk from 0 to 1: Niyama risk merged with liquid isolation. High values mark metal that either freezes with a poor thermal gradient or is cut off as an isolated liquid pool. Computed as the larger of the chill-adjusted Niyama risk and 0.7 × the isolation score; surface points and chills read 0. Default threshold 10 %. |
| `fraction_liquid` | Liquid fraction at the time on the time bar, from 1 (fully liquid) to 0 (solid). Play or scrub it to watch the freezing front; liquid that becomes enclosed by solid is a likely shrink site. Taken from snapshots of the thermal solve. There is no threshold. |
| `fill_time` | Time in seconds at which metal first reaches each point while the mould fills. The latest points to fill are prone to cold shuts and misruns; points the metal never reached count as unfilled. Computed by the filling simulation from the pour location and pour rate. Use the time bar to follow the fill front; there is no threshold. |
| `fill_animation` | Frame-by-frame playback of mould filling. The web colours each cell by how full it is (0 to 1); the desktop colours the metal by flow speed in m/s, where high speeds point to turbulence and mould erosion. Frames come from the filling simulation. There is no threshold. |
| `hot_tear` | Hot tear risk from 0 to 1: the chance of tearing while the metal is weak near the end of freezing (the brittle temperature range). High values sit where restrained metal contracts across hot, late-freezing spots such as junctions and sharp corners. Computed from time spent in the brittle range, strain from solid neighbours, geometric stress concentration and mould and core restraint, relieved where feed metal can heal a tear. Default threshold 10 %. |
| `sand_burn_in` | Sand burn-in (metal penetration) risk from 0 to 1 on surfaces in contact with sand. High values mark faces where metal is likely to push into the sand and leave a rough, fused skin. Computed from superheat, ferrostatic head, metal-to-sand contact time, sand permeability and alloy fluidity. Default threshold 20 %. |
| `gas_generation` | Gas generated by the mould and cores, from 0 to 1, carried inward from the sand surfaces. High values sit next to cores and binder-rich mould faces. Computed from binder decomposition and moisture at each sand surface, weighted by surface type (core 1.0, mould wall 0.6, open riser 0.2, chill 0) and decayed exponentially with depth into the metal. Default threshold 10 %. |
| `gas_entrapment` | Gas entrapment tendency from 0 to 1: how likely gas that reaches the metal stays trapped. High values are thick, slow-freezing sections where bubbles have time to collect before the metal closes around them. Computed as (local solidification time / longest local solidification time) squared. Default threshold 10 %. |
| `gas_porosity` | Gas porosity risk from 0 to 1: the chance of blowholes or pinholes from mould and core gas. High values sit where gas generation is high, the section freezes slowly and the Niyama risk is high. Computed as gas generation × entrapment × alloy gas sensitivity × a Niyama factor between 1 and 1.5. Default threshold 10 %. |
| `feeding_quality` | Feeding blockage from 0 to 1, shown as 1 minus feeding quality: 0 is fully fed, 1 is cut off. High values mark metal that no riser can reach through still-liquid paths, so its shrinkage cannot be fed. Computed by tracing liquid paths from the risers (or from the surface when there are none) through the fraction-solid history. Default threshold 20 %. |
| `thermal_stress` | Residual thermal stress from uneven cooling, in MPa; the colour scale runs from 0 to the run's maximum. High values sit where neighbouring regions cool at different rates, such as thin walls joined to thick sections. Simplified model: stress = E × (thermal strain + constraint strain from neighbours), not a full FEA. Default threshold 20 %. |
| `distortion` | Expected distortion from uneven contraction; the web legend is in inches and the desktop legend in mils, and the colour scale runs from 0 to the run's maximum. High values show where the casting is most likely to move out of shape. Computed from the constraint strain and its direction times the local cell size; use it to compare designs, not as an FEA result. Default threshold 20 %. |
| `crack_susceptibility` | Crack susceptibility from 0 to 1. High values sit where stress approaches the alloy's yield strength while strain is high late in freezing. Computed from effective stress, constraint strain and fraction solid, scaled by the sand restraint index. Default threshold 20 %. |
| `sdas` | Secondary dendrite arm spacing (SDAS) in µm; the colour scale spans the run's minimum to maximum. Large values mean a coarse structure from slow cooling, usually in thick, late-freezing sections, and lower strength. Computed as SDAS = A × (cooling rate)^-n with alloy constants, clamped to 10-500 µm. Default threshold 20 %. |
| `predicted_uts` | Predicted ultimate tensile strength in MPa, coloured inverted so that the strongest colour marks the weakest metal. Weak regions are the coarse-structured, slow-cooling sections. Computed from SDAS with a Hall-Petch relation, UTS = σ0 + k / sqrt(SDAS). Default threshold 20 %. |

## Other user-facing strings changed

Desktop help dialog, Shrink Scalar tooltip ("Global calibration of the Shrink and Microporosity results"), legend title list, analysis option "Shrink and Microporosity", parts dialog tooltip. Web AnalysisPanel toggle, Shrink Scalar help, statLabels (group "Shrink and microporosity", "Max/Mean Shrink risk"). README, QUICKSTART, PROJECT_OVERVIEW, solidus_pro_paper tables.

## Files

New: `core/result_help.json`, `core/result_help.py`, `ui/result_types.py`, `tests/test_result_naming_and_help.py`. Changed: `core/solidification.py`, `core/simulation_pipeline.py`, `core/defect_regions.py` (docstrings), `web/payloads.py`, `ui/main_window.py`, `ui/project_state.py`, `ui/qt.py` (QToolButton, QToolTip), `ui/visualizer.py` (docstrings), `ui/dialogs/analysis_options_dialog.py`, `ui/dialogs/parts_dialog.py`, `web_frontend/src/{lib/viewState.ts, lib/statLabels.ts, lib/solidificationFields.ts, components/ResultsPanel.tsx, components/panels/AnalysisPanel.tsx, styles/results.css}`, `web_frontend/vite.config.ts`, tests `test_web_api.py`, `test_shrinkage_default_threshold.py`, `test_fill_time_visualization.py`, docs above.

## Tests

- New `test_result_naming_and_help.py`: 23 tests (arrays, grid source, labels, legacy load, combo items and tooltips, Microporosity view and inspector, info button, help coverage of all 22 web modes and all 19 desktop labels, 2-4 sentences each).
- Updated: Shrink payload is shrink only (`test_web_api`), desktop Microporosity default and `_get_recommended_threshold` 70/70 (`test_shrinkage_default_threshold`).
- Targeted run (13 modules): 266 tests, 1 failure = baseline `test_shrink_scalar_below_one...`.
- Full suite: 2169 tests, 3 failures, all the DESIGN.md baseline (`test_a_deeper_tower_leaves_none_either`, `test_case_pack_reference_targets` riser_post_block, `test_shrink_scalar_below_one...`). Log: `/tmp/naming_full.log`.
- `npm run build` passes; eslint clean on changed TS; mypy allowlist 64 errors before and after.

## Not done / risks

- The run statistics `shrinkage_volume_in3`, `shrinkage_voxel_count`, `shrinkage_percentage` and `defect_*` still count points where shrink OR porosity exceeds the solver threshold. Only their group heading changed.
- Chrome on macOS does not show `title` tooltips inside a native `<select>` list. The info button covers that case.
- Not checked in a live desktop session or browser: popover placement and tooltip wrapping were verified by build and unit tests only.
