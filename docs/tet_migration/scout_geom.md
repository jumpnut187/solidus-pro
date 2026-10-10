# Scout: geometry input and voxelization seam

Branch `feature/tet-mesh`. Read-only. Line numbers as of this scan.

## 1. Entry points and call graph

Production path (`core/simulation_pipeline.py:580-660`):
1. `PartManager.get_combined_mesh()` (`core/part_manager.py:1611-1630`) â `trimesh.util.concatenate` of every visible part's transformed mesh (no boolean union).
2. `PartManager.get_combined_bounds()` (`core/part_manager.py:1632`).
3. `PartManager.get_parts_with_types()` (`core/part_manager.py:1586-1603`) â `[(transformed_trimesh, int code)]`, visible parts only.
4. `AdaptiveVoxelMesh(base_voxel_size, max_refinement_level, memory_mode, use_chunked_voxelization, part_assignment_*)` (`core/simulation_pipeline.py:610-618`, ctor `core/voxel_mesh.py:662-733`).
5. `voxel_mesh.generate(mesh, bounds, show_progress=False, progress_callback, part_meshes=[...])` (`core/simulation_pipeline.py:626-633`, `core/voxel_mesh.py:735-906`).
6. `voxel_mesh.assign_part_types(parts_with_types)` (`core/simulation_pipeline.py:646`, `core/voxel_mesh.py:1740-1875`).
- Fallback single-STL path: `stl_loader.trimesh_obj` with no part types (`core/simulation_pipeline.py:594-597`) â all voxels CAST.
- Second caller: `ui/main_window.py:1968-1972` (generate only, no part_meshes).
- `core/filling_stage.py:86` mentions generate() segfaults at some voxel sizes.

`generate()` steps (`core/voxel_mesh.py:772-906`):
1. Memory pre-check: `auto_adjust_voxel_size` + `validate_mesh_parameters` from `core/memory_limits.py:372,267`; may silently change `base_voxel_size` (`voxel_mesh.py:780-790`). Uses `trimesh_obj.volume` for fill ratio (`:776`).
2. `_create_uniform_grid` (`:908-1007`): padded bbox, `grid_shape=(nx,ny,nz)` (`:925-929`). Small models: `trimesh_obj.voxelized(pitch).fill()` (`:957-961`) then `_unfill_sealed_pockets` (`:275`) using per-part meshes. >100k candidates: `_create_uniform_grid_chunked` (`:1009`) z-slice containment workers (`:1106-1232`); fallback `:1234`.
3. Refinement skipped; uniform grid only (`:846-850`). `_adaptive_refinement` (`:1296`) unused.
4. `_calculate_voxel_properties` (`:1343-1514`).
5. `_build_neighbor_graph` (`:1516-1600`) via `SpatialIndex` (`core/octree.py:481-556`) or KD-tree, then canonical slot placement (`:1602-1723`).
6. `_identify_surface_voxels` (`:1725-1738`).

`assign_part_types` (`:1740-1875`): bucket parts by code (`:1784-1796`), CAST default, fast path for single CAST (`:1800-1804`), `_assign_part_types_loop` (`:1959`) containment with `proximity_tolerance = 0.5*voxel` (`:1825`), higher code overrides lower (`:1790-1796`, override at `:2047-2050`), `_claim_unowned_boundary_voxels` (`:1877`), `_bridge_part_interfaces` (`:2052-2153`).

## 2. Data consumed / produced

Inputs:
| name | type | units |
|---|---|---|
| `trimesh_obj` | concatenated `trimesh.Trimesh` | inches |
| `bounds` | `(3,2)` `[[xmin,xmax],...]` (`voxel_mesh.py:745`) | inches |
| `part_meshes` | list of per-part `Trimesh` world coords | inches |
| `parts_data` | list `(Trimesh, int code)` | inches |

AdaptiveVoxelMesh public attributes (all via `create_managed_array`, RAM or memmap; `voxel_mesh.py:698-724,1358-1514,1532,1728,1759`):
| attr | shape | dtype | meaning |
|---|---|---|---|
| `voxel_centers` | (N,3) | float32 | inches (`:1358`) |
| `voxel_sizes` | (N,) | float32 | all = `base_voxel_size` (`:1361,1373`) |
| `voxel_volumes` | (N,) | float32 | size^3 in^3 (`:1390-1394`) |
| `voxel_surface_areas` | (N,) | float32 | exposed face count * size^2, in^2 (`:1471`) |
| `voxel_moduli` | (N,) | float32 | distance (in) from center to nearest surface voxel center, 0 on surface (`:1479-1497`); NOT V/A except fallback `:1499-1504` |
| `voxel_states` | (N,) | float32 | zeros init (`:1508`) |
| `voxel_neighbors` | (N,26) | int32 | -1 = none (`:1532-1535`) |
| `surface_voxels` | (N,) | bool | `surface_areas>0` (`:1728-1738`) |
| `voxel_part_types` | (N,) | int8 | part code, default 0 (`:1759-1762`) |
| `metal_part_bounds` | list of (2,3) float64 | per metal part bbox (`:1773-1777`) |
| `bounds`, `grid_shape` | input bounds, (nx,ny,nz) | (`:763,929`) |
| `base_voxel_size`, `original_voxel_size`, `voxel_size_was_adjusted`, `adjustment_message` | scalars (`:679-696,787-790`) |
| `memory_manager`, `memory_monitor` | (`:725-733`) |

Public methods: `preview_voxel_count` (`:516`), `generate`, `assign_part_types`, `get_voxels_by_type(code)` (`:2574`), `get_voxel_count` (`:2618`), `get_surface_voxel_indices` (`:2622`), `get_material_surface_voxel_indices` (`:2628-2669`, cached `_material_surface_indices`), `get_memory_stats`/`print_memory_stats` (`:2610-2616`), `cleanup` (`:2671`).

Part codes (`core/part_manager.py:454-463`): CAST 0, RISER 1, CHILL 2, RISER_POST 3, CORE 4, POUR_CUP 5, SPRUE 6, RUNNER 7, FILTER 8, GATE 9. Sets: METAL (0,1,3,5,6,7,9) `:491`, FEEDER (1,3) `:502`, MOLD (2,4,8) `:511`, SCORED_METAL (0,1,3) `:516`, FLOW_PATH (5,6,7,8,9) `:520`, GATING_METAL (5,6,7,9) `:531`, FLOW_RESTRICTOR (8) `:541`. Predicates `is_metal/is_feeder/is_mold/is_cast/is_chill/is_scored_metal/is_flow_path/is_gating_metal/is_flow_restrictor/is_filter` (`:552-599`). Codes are divided by `fields.PART_TYPE_DIVISOR` (3.0) as an ML channel (`:450-452`); renumbering is unsafe.

Part storage: `Part` dataclass (`part_manager.py:803-947`) holds raw `trimesh_obj`, `position`, `rotation` (Euler deg), `scale`, cached `_transformed_mesh` rebuilt on dirty (`:849-879`: scale â rotate â translate). STEP provenance `body_name/body_index/tessellation_tolerance_mm` (`:813-822`).

STEP: optional `cascadio` (`part_manager.py:42`). Tolerance `tessellation_tolerance_mm(voxel_in) = voxel_in*25.4/50`, clamped [0.01, 0.5] mm (`:57-117`); passed as `tol_linear` (`:370-380`). `retessellate_for_voxel_size` re-tessellates when voxel size changes (`:1243-1310`). Meters â inches rescale `_scale_step_to_inches` (`:328-367`); unit sanity check vs declared STEP unit (`:178-325`). Multi-body fan-out: `load_assembly_parts` (`:1106`), `load_step_bodies` (`:1343`); one Part per non-empty scene body. Type inferred from names `infer_part_type_from_name/filename` (`:708,785`).

`core/stl_loader.py`: single-STL legacy loader, inches assumed, no scaling (`:30-58`); warns if not watertight (`:90-98`). `MultiPartSTLLoader` (`part_manager.py:1809-1905`) wraps PartManager for legacy API.

## 3. Voxel-specific assumptions

- Uniform size: `voxel_sizes[:] = base_voxel_size` (`voxel_mesh.py:1366,1373`); refinement disabled (`:846-850`).
- Dense lattice: `voxelized(pitch).fill()` + dense `matrix` arrays (`:957-972`); `grid_shape` (`:929`); `_unfill_sealed_pockets` uses connected components on dense occupancy (`:275-358`).
- 26-slot neighbor layout (`:1610-1616`): 0-5 faces +X,-X,+Y,-Y,+Z,-Z; 6-17 edges; 18-25 corners. Consumers named: `core/thermal_solver.py:compute_thermal_gradient`, `physics/heat_transfer.py` laplacian kernels (`:1609-1611`). Sign tolerance `0.25*voxel` (`:1636`); nearest per slot (`:1659-1671`).
- Search radius `2.0*voxel` (`:1541`), `settings.mesh.neighbor_search_radius_factor` (`config/settings.py:64`) not wired into this line.
- Bridging breaks slot semantics: `_bridge_two_types` writes extra neighbors into the FIRST empty slot (`voxel_mesh.py:2136-2148`), so slot index != direction for bridged links.
- Surface detection by 6-axis KD probe at `center Âą size` (`:1418-1474`).
- Modulus = distance to surface voxel centers (`:1479-1497`).
- Containment tolerance 0.5 voxel (`:1825`); boundary claim uses half cell diagonal (`:1877-1895`).
- Mold-adjacency hardcodes codes 2 and 4 (`:2665`), excluding FILTER though FILTER is in MOLD_PART_CODES.
- Heat BC uses `dx_i = voxel_sizes[idx]` for convection/interface flux `h*(T-T_inf)/(rho_cp*dx)` (`physics/heat_transfer.py:741-766`), i.e. area/volume = 1/dx.
- Memory model is per-voxel (`core/memory_limits.py:176-267,372-405`, ~32 B/candidate in `voxel_mesh.py:935`).
- `core/mesh_ml.py`: recommends voxel size/refinement from STL features and hardware (`:72,172,384`); logs training examples (`:462`). Voxel-count-based time estimate (`:141`).
- `MeshSettings` (`config/settings.py:55-75`): `base_voxel_size=0.1`, min 0.02, max 0.5 in, `max_refinement_level`, `memory_mode`, `use_octree`, `use_chunked_voxelization`, `neighbor_search_radius_factor`; env `VOXEL_SIZE`, `MEMORY_MODE` (`:404-405`).

## 4. Answers

- Mold (sand): NOT meshed. Represented only as a boundary condition on the outer surface of the casting assembly (CONVECTION/INTERFACE with `T_infinity`, `physics/heat_transfer.py:741-766`).
- Chills, cores, filters: meshed as solid voxels in the same grid with codes 2/4/8; "mold material" per `MOLD_PART_CODES` (`part_manager.py:511`). Chill contact is via neighbor conduction plus bridging (`voxel_mesh.py:46-58`).
- Risers, riser posts, gating (pour cup, sprue, runner, gate): metal voxels with their own codes; thermal-solved, excluded from scoring except risers (`part_manager.py:491-535`).
- Parts are overlapping, not unioned: combined mesh is `concatenate` (`part_manager.py:1628`). Overlaps resolved by "higher code wins" during assignment (`voxel_mesh.py:1790-1796`). Unclaimed fill defaults CAST; phantom fill limited by `_unfill_sealed_pockets` and `metal_part_bounds`.
- Watertight requirements: `trimesh.voxelized().fill()` and `contains` on each non-CAST part (`_robust_contains_check` `:2155`, `_robust_contains_check_impl` `:196`); `_unfill_sealed_pockets` keeps regions inside non-watertight parts (`:291-294`); `_claim_unowned_boundary_voxels` skips non-watertight parts (`:1891-1893`); 0-voxel warning for chills (`:1855-1862`); `Part.get_volume` uses `abs(volume)` (`part_manager.py:912`).

## 5. What a tet replacement must provide

1. A conforming multi-region tet mesh: one volume with per-tet region code (int8 part code). fTetWild on a concatenated soup does not label regions; options are per-part tetrahedralization then interface merge, or tetrahedralize the union and label tets by centroid containment with "higher code wins" to preserve current overlap semantics.
2. Per-node arrays replacing per-voxel: `node_xyz (M,3) float64 in`, `node_volume` lumped (in^3), node part code (rule needed for nodes on interfaces), `boundary_faces (F,3)` with area (in^2) for the mold BC, node boundary area for lumped Robin BC.
3. Distance-to-surface field replacing `voxel_moduli` (nodes 0 on outer boundary).
4. `metal_part_bounds`, `get_voxels_by_type`, `get_surface_voxel_indices`, `get_material_surface_voxel_indices` equivalents (chill/core-adjacent cast nodes).
5. Interface contact between separately modeled parts (cast/chill gaps) currently fixed by `_bridge_part_interfaces`; a tet mesh needs shared nodes (conformal) or explicit contact/IHTC elements.
6. Size control: map `base_voxel_size` to `edge_length_abs`; keep `tessellation_tolerance_mm` tied to that length; replace `memory_limits` estimates with tet/node counts.

pytetwild: installed in `venv/lib/python3.13/site-packages/pytetwild` v0.4.2. First import attempt in this session raised `ModuleNotFoundError`; a later attempt succeeded (likely installed concurrently). Exports `tetrahedralize`, `tetrahedralize_pv`, `tetrahedralize_csg` (`pytetwild/__init__.py:4`).
`tetrahedralize(vertices f32|f64 (n,3), faces int32 (m,3), edge_length_fac=0.05 (of bbox diagonal, range (0,1]), edge_length_abs=None (absolute, overrides fac), optimize=True, simplify=True, epsilon=1e-3 (envelope, relative to bbox), stop_energy=10.0, coarsen=False, num_threads=0, num_opt_iter=80, loglevel=3, quiet=True, vtk_ordering=False, disable_filtering=False, bg_vertices/bg_tets/bg_values (sizing field, requires optimize=True)) -> (verts float64 (k,3), tets int32 (t,4))` (`pytetwild/pytetwild.py:368-470`). `tetrahedralize_pv` returns a pyvista UnstructuredGrid (`:222-365`). `tetrahedralize_csg(csg_file json, epsilon, edge_length_r, stop_energy, coarsen=True, ...)` (`:488-518`) supports boolean CSG of multiple inputs, the only built-in path to union/difference; no per-region labels returned by `tetrahedralize`. Epsilon is relative to bbox, so absolute surface error scales with assembly size.

## 6. Risks and pinning tests

Risks:
- Slot layout is consumed by the thermal gradient and laplacian; any node-graph replacement breaks those consumers (`voxel_mesh.py:1609-1611`).
- Bridged neighbors in arbitrary slots already violate slot semantics (`:2136-2148`).
- Overlap resolution, sealed-pocket removal and boundary claim are voxel-specific heuristics with no tet analog yet.
- Part-code ML channel ties codes to surrogate (`part_manager.py:450-452`).
- fTetWild `simplify=True` and envelope epsilon can move small features (gate fillets, thin chills) when epsilon is relative to a large bbox.
- `ui/main_window.py:1968` builds a voxel mesh outside the pipeline; also must migrate.
- `voxel_moduli` is a distance, not V/A; downstream consumers expect inches.

Tests touching this seam (`tests/`): `test_neighbor_canonicalization.py`, `test_centered_riser_symmetry.py`, `test_voxel_sealed_pockets.py`, `test_boundary_claim.py`, `test_part_assignment_generality.py`, `test_voxel_part_assignment_parallel.py`, `test_part_type_predicates.py`, `test_part_type_encoding_gate.py`, `test_part_manager.py`, `test_tessellation_tolerance.py`, `test_step_import_reopen.py`, `test_web_step_upload.py`, `test_voxel_mesh_kdtree.py`, `test_voxel_mesh_memory_mode.py`, `test_memory_manager_modes.py`, `test_unfinished_voxel_masking.py`, plus indirect: `test_simulation_scenarios.py`, `test_defect_regions.py`, `test_surrogate.py`, `test_surrogate_chill_channel.py`, `test_main_window_component_mask.py`, `test_filling_defects.py`, `test_fraction_liquid_timing_consistency.py`, `test_orientation.py`, `test_project_state.py`, `test_web_api.py`, `test_web_simulation_process.py`.

Not read in detail: `core/octree.py` internals beyond API (`:104-556`), `core/memory_manager.py` (301 lines, provides `create_managed_array`/`get_memory_manager`), chunked voxelization internals (`voxel_mesh.py:1009-1294`).
