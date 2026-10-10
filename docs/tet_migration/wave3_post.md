# Wave 3: post-analyses on tet nodes

Every post-analysis now runs on tet nodes with tet-specific length, volume and adjacency semantics. Each tet branch sits behind `mesh_type == "tet"`. Voxel code paths are unchanged, and no existing kernel signature changed. Nothing is staged or committed.

## Files changed

- `core/solidification.py`: post-analysis and graph-distance code only. Module helpers `_is_tet_mesh`, `_tet_graph_length_in`, `_tet_hop_length_in`, `_local_length_scale_in`, plus `SHRINKAGE_STATISTICS_THRESHOLD` (0.20, the value it replaces) and `SolidificationEngine.get_shrinkage_percentage()`. The helpers are module functions because tests call engine methods unbound on `SimpleNamespace` objects.
- `core/acceleration.py`: new kernels `compute_isolation_score_volume_weighted` and `compute_solid_path_strain_local`.
- `core/feeding_path.py`, `core/stress_analysis.py`, `core/microstructure.py` (new kernel `estimate_cooling_rate_from_solidification_edges`), `core/defect_regions.py`, `core/riser_design.py`, `core/incremental_solidification.py`.
- `tests/test_tet_post_analyses.py` (new, 17 tests).
- `physics/gas_porosity.py` and `physics/riser_pipe.py` are unchanged. Both already take plain arrays (surface indices, centres, volumes) and need no tet branch.

## Behaviour on tets, by brief item

1. **Graph distances.** `_compute_graph_distance_to_riser` and `_compute_graph_distance_to_chill` run a multi-source `scipy.sparse.csgraph.dijkstra` over edges whose two ends are traversable, weighted by `|c_i - c_j|`. The result is divided by `mesh.mean_edge_length`, so callers still get hops (inches = hops × mean edge). This keeps the schedule's 1-hop floors and max-normalisation. On tets the only live consumer is `_get_connected_feeder_mask`, which reads `isfinite`. The schedule path is voxel-only because tets force `thermal_coupled`. `FeedingPathAnalyzer.path_length` was already in inches from node centres and is unchanged.
2. **Orphan "smothered" rule.** "≥ 8 of 26 chill/core neighbours" becomes "≥ 8/26 of the node degree". The other conditions stay: chill/core neighbours ≥ metal neighbours, and > 0. No other rule in these files uses an absolute neighbour count:
   - BTR support, constraint strain and microstructure already average or take a max over valid neighbours.
   - The feeder-neck factor uses "any neighbour".
   - The disconnected-pocket rule keeps its node-count threshold, max(2, 5 % of the largest component).
3. **Length scales.** All use per-node `voxel_sizes` (mean incident edge length):
   - Surface restraint (stress crack multiplier): `max(0.5, 5·h_i)` per node.
   - Hot-tear path strain: the cap is `search_depth·h_i` per node, through the new kernel. Edge lengths inside the kernel were already `|c_i - c_j|`.
   - Chill-thickness fallback, in both chill shrinkage reduction and sand burn-in: N × the mean `h` over chill nodes.
   - Stress distortion: `h_i·|v_i|`.
   - Microstructure: each neighbour Δt is scaled by `h_i/|c_i - c_j|`. The voxel kernel never used its `voxel_size` argument.
   - Riser design: the ±size stencil is replaced (item 7).
4. **Volume semantics.** Changed on tets:
   - `export_results['shrinkage_percentage']`: affected node volume / total node volume.
   - `FeedingPathResult.fed_fraction` and `avg_path_length`: volume-weighted, in both the snapshot and merge branches.
   - `StressAnalysisResult.avg_stress` and `high_stress_fraction`: volume-weighted.
   - `MicrostructureResult.avg_sdas` and `avg_uts`: volume-weighted.
   - Already volume-based and unchanged: `total_shrinkage_volume`, `get_defect_volume_estimate`, the riser pipe (`voxel_volumes` into `gravity_drain_pipe`) and pocket volume.
5. **Isolation.** The tet path uses the volume-weighted kernel: the fraction of neighbour lumped volume that froze first. With uniform volumes it equals the count fraction, which a test checks.
6. **Pockets.** `extract_pocket_at(..., mesh_type="tet")` walks the full adjacency row and ignores `connectivity`. The docstring and module doc say so. The UI already passes `mesh_type`.
7. **Riser design.** `RiserDesigner` and `HeuversCircleMethod` take `voxel_volumes=None, mesh_type="voxel"`.
   - On tets, cluster volume is the sum of `voxel_volumes`. With `voxel_volumes=None` it keeps `count × size³` (parent contract update). Volumes of the wrong length, or a neighbours table that is not `(n, K)`, raise ValueError. Voxel meshes ignore `voxel_volumes`.
   - In `optimize_positions`, a step moves to an edge-adjacent surface node in the top band, through the new `_candidate_attachment_moves`. The voxel moves are the same 8 snapped offsets in the same order.
   - `ui/main_window.py` already calls the new signature.
8. **Incremental.** `expand_front` sizes its buffer with `neighbors.shape[1]` instead of 26.
9. **Surface semantics.** Checked, with no code change. Tet boundary nodes are at depth 0, while voxel surface centres sit h/2 inside, so tet `surface_distances` and `voxel_moduli` read the true depth and voxel ones read about depth − h_vox/2. That is a 0.05 in shift at 0.1 in.
   - Every normalisation (hot spot, isolation surface factor) is a ratio, and the shift does not move the peaks (see tests).
   - The absolute thresholds (interior ≥ 0.3 in, section ≥ 0.5 in, restraint depth) now apply to the true depth. That is the h→0 limit of the voxel value.
   - Niyama surface suppression zeroes boundary nodes on tets and the outer cell shell on voxels. The first unsuppressed tet node is about one edge deep, where the P1 gradient is already high.
   - I did not add an artificial h/2 offset to tets. It would make tet results depend on the tet h.

## Measured (assembly from `test_tet_end_to_end`, all analyses on, tet 0.2 in vs voxel 0.1 in)

| quantity | tet | voxel |
|---|---|---|
| wall time | 2.0 s | 9-10 s |
| hot-spot peak centroid (in) | (-0.16, 0.0, 0.67) | (-0.14, 0, 0.75) |
| cast shrinkage peak centroid (in) | (0.0, 0.0, 0.51) | (-0.02, 0, 0.48) |
| isolation p50 / p95 / max (volume-weighted) | 0.29 / 0.51 / 0.70 | 0.31 / 0.43 / 0.58 |
| shrinkage % (volume) | 4.4-4.6 | 4.31 |
| fed fraction | 0.94 | 0.91 |

Before this change, tet `shrinkage_percentage` was a node ratio of 7.4 %.

## Deviations, with reasons

1. Graph distances return hop-equivalents, not inches. The brief asks to keep the consumer contract, and the schedule floors at 1.0 hop. The test checks `hops × mean_edge_length` in inches against the expected depth below the riser along the axis.
2. The faithful 8/26 mapping still flags 1-3 real metal nodes on a conforming tet assembly. They sit on the line where the chill side is flush with the casting edge (half chill, half metal neighbours). The voxel rule flags 13 voxels on the same edge. The test bounds them at < 1 % of metal volume, all adjacent to chill.
3. `physics/gas_porosity.py` and `physics/riser_pipe.py` needed no edit.

## Tests

| command | result |
|---|---|
| `venv/bin/python -m unittest tests.test_tet_post_analyses` (×4) | 17/17 OK, about 11 s |
| `tests.test_feeding_path_analysis`, `test_defect_regions`, `test_isolated_pocket_shrinkage`, `test_isolation_score_kernel`, `test_shrinkage_display_intensity`, `test_riser_pipe`, `test_centered_riser_symmetry`, `test_gas_porosity`, `test_hot_tear_physics`, `test_hot_tear_phase2`, `test_sand_burn_in`, `test_architecture_boundaries` | 109 run, 1 failure: the baseline `test_shrink_scalar_below_one...` (shrink-scalar division, untouched) |
| `tests.test_tet_end_to_end` | 11/11 OK |
| extra callers: `test_chill_interface_safety`, `test_math_audit_regressions`, `test_metal_backend_parity`, `test_performance_summary_label`, `test_solidification_params`, `test_unfinished_voxel_masking`, `test_darcy_velocity`, `test_fill_time_visualization` | 129 OK, 2 skipped |

## Open risks

- `core/simulation_pipeline.py:583` (`result_data['shrinkage_percentage']`) and `core/analysis_strategies.py:184` (a print) still compute affected count / point count. On tets they should call `engine.get_shrinkage_percentage()`. Neither file is mine. This is for the integration step.
- Stress distortion vectors are an unnormalised sum over neighbours. Tet degree (about 14-29) differs from 26, so distortion magnitude is not calibrated against voxel. Only the length scale was ported.
- `_estimate_feeder_neck_resistance_factors` maps padded `-1` slots to node 0 (pre-existing on voxels). It is wrong only if node 0 is a RISER_POST node. It is more likely to matter on tets, where rows are shorter than K.
- The orphan disconnected-pocket rule still counts nodes, not volume.
