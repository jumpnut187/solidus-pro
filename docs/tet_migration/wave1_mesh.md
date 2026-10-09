# Wave 1 mesh worker report (contract C1)

## Files changed
- `core/tet_mesh.py` (new): `build_tet_mesh`, `TetVolumeMesh`.
- `tests/test_tet_mesh.py` (new): 21 tests.
- `requirements.txt`: added `pytetwild>=0.4.2` and `pyvista>=0.48` with a licence note (MPL-2.0 / MIT).

## Public API as implemented
- `build_tet_mesh(parts, edge_length_in, *, epsilon_rel=1e-3, stop_energy=10.0, num_threads=0, progress_callback=None) -> TetVolumeMesh`. Signature matches C1.
- `TetVolumeMesh(node_xyz, tets, tet_part_types, boundary_faces, boundary_face_part_types, boundary_face_contact_codes, *, base_voxel_size, metal_part_bounds=())`. The constructor takes an extracted domain and derives every node array. Use `build_tet_mesh` in normal code.
- Module constants: `REMOVED_PART_CODES = (CORE, FILTER)`, `SLIVER_DIHEDRAL_DEG = 5.0`.
- Every attribute in the C1 table, with the stated shape and dtype. `bounds` comes from the node coordinates.
- `quality_summary()` returns `min_dihedral_deg`, `median_dihedral_deg` (median over all 6T angles), `sliver_count`, `sliver_threshold_deg`, `node_count`, `tet_count`, `boundary_face_count`.

## Pipeline
1. One part goes through `pytetwild.tetrahedralize(edge_length_abs=h, coarsen=False)`. Two or more parts go through a left-deep OBJ CSG union in a `TemporaryDirectory`, then `PyfTetWildWrapper.tetrahedralize_csg(csg, epsilon, h/diag, stop_energy, False, num_threads, 6, False)`. The argument order was checked against `PyfTetWildWrapper.pyi`.
2. Output is silenced with `os.dup2` of fd 1 to devnull, plus `libc fflush`, plus swapping `sys.stdout`. Both are restored in `finally`. Finding: the CSG wrapper sends `std::cout` to Python's `sys.stdout`, so the fd redirect alone still leaked "Pre-checking CSG input meshes...". A test now captures fd 1 and `sys.stdout` and asserts both are empty.
3. Tets are oriented positively by swapping vertices 2 and 3 where the signed volume is negative.
4. Markers: marker k maps to the k-th leaf, checked empirically. The assembly built in two input orders gives matching per-code volumes. Where parts overlap, fTetWild marks the tet with the last containing leaf, not the highest code. With [riser, cast], the raw riser volume was 1.995 against a target of 3.990. With [core, box], the raw core volume was 0. The highest-code-wins pass fixes both. It tests only tets whose marker part's bbox overlaps the testing part's bbox, whose centroid is inside that bbox, and whose current code is lower, then calls `Trimesh.contains`.
5. CORE and FILTER tets are removed. Face matching uses a lexsort over 4T sorted face triples. A boundary face is a kept tet face with no kept partner. Its contact code is the removed partner's code, or -1. Faces use the outward local ordering. Nodes are re-indexed and orphans dropped.
6. Node arrays are built with bincount and `np.maximum.at`. Edge CSR uses lexsort. The padded table uses one scatter. There are no Python loops over tets or nodes.

## AdaptiveVoxelMesh surface
I grepped `core/`, `web/` and `ui/` for `mesh.`, `voxel_mesh.`, `self.mesh.` and `getattr(mesh, "...")`.
- Mirrored: `voxel_centers`, `voxel_sizes`, `voxel_volumes`, `voxel_surface_areas`, `voxel_moduli`, `voxel_states`, `voxel_neighbors`, `surface_voxels`, `voxel_part_types`, `metal_part_bounds` (`solidification.py:1571`), `bounds`, `grid_shape` (None), `base_voxel_size`, `original_voxel_size`, `voxel_size_was_adjusted`, `adjustment_message`, `get_voxel_count`, `get_surface_voxel_indices`, `get_voxels_by_type`, `get_material_surface_voxel_indices` (`solidification.py:1473`), `get_memory_stats` (same keys as `MemoryManager`, read at `simulation_pipeline.py:648` and `simulation_queue.py:285`), `print_memory_stats`, `cleanup` (no-op, no memmaps).
- Not mirrored, deliberately: `generate`, `assign_part_types` and `preview_voxel_count`, because `build_tet_mesh` replaces them and the pipeline dispatch is C5. Also not mirrored: `voxels`, `memory_manager`, `memory_monitor`, `memory_mode`, `use_octree`, `use_chunked_voxelization`, `max_refinement_level` and `part_assignment_*`. These are constructor config or internal state, and nothing reads them off a mesh object.

## Deviations and interpretations
1. `voxel_moduli` is the exact point-triangle distance (`trimesh.triangles.closest_point`) to the 16 boundary faces with the nearest centroids. It is not a certified global minimum.
   - The worst-case overestimate is the largest centroid-to-vertex distance of a boundary face, about 0.6 h.
   - Measured, it matches the exact `trimesh.proximity.closest_point` to 1e-5 on sampled nodes of the cylinder, the l_bracket and the 2 in box. On a 183k-node box it matched a k=512 run exactly.
   - A certified version, which grows k until `kth_centroid_dist >= best + r_max`, took 25 s on 381k nodes against 0.95 s now.
2. `get_material_surface_voxel_indices` returns geometric boundary nodes plus metal nodes on a CHILL tet. CORE-contact nodes are boundary nodes already. This matches the voxel version, which includes the geometric surface. One difference: the voxel version limits the chill/core extension to CAST, and C1 says metal, so I followed C1.
3. `TetVolumeMesh.__init__` takes `boundary_face_part_types` as an argument. C1 does not specify the constructor.
4. `edge_length_r = min(h / diag, 1.0)` for the CSG path.
5. Brief vs data: `l_bracket.stl` is two concatenated boxes. Its 39 in² STL area counts the 1 in² contact patch twice as internal faces, so the real skin is 37 in². The area test checks the l_bracket against 37 (the mesh gives 36.98) and the cylinder against its STL area. The 0.5 % volume test uses the STL volume unchanged.
6. `voxel_neighbors` width K is the max degree (23–29 measured), as C1 says. It is not 26-slot canonical.

## Tests
- `venv/bin/python -m unittest tests.test_tet_mesh -v`: 21 run, 21 passed, 12.4 s wall.
- `venv/bin/python -m unittest tests.test_architecture_boundaries`: 5 run, 5 passed.
- `mypy core/tet_mesh.py` (follow-imports silent): no issues. `compileall` passes.
- Coverage: cylinder and l_bracket volume (0.5 %), lumped volume equals tet volume, boundary area (2 %), positive tets, outward faces (divergence volume equals tet volume), CSR symmetric, sorted and equal to the padded table, three-part per-code volumes (1 %) in two input orders, shared interface nodes with no interface boundary area, node-code rule at cast/chill and cast/riser, embedded core in both orders (no core tets, CORE contact area within 2 % of the core skin, normals into the cavity), overlap highest-code-wins in both orders, moduli 0 on the boundary with max 1.0 ± 0.1 in a 2 in box, lazy import, silenced output, error paths.

## Performance (M5 Pro, 18 cores; 4-part assembly with a CORE)
| h (in) | nodes | tets | fTetWild | codes | extract | node arrays | quality |
|---|---|---|---|---|---|---|---|
| 0.10 | 48.6 k | 263 k | 7.1 s | 0.04 s | 0.25 s | 0.22 s | 0.08 s |
| 0.05 | 376 k | 2.14 M | 53–82 s | 0.33 s | 3.6 s | 2.2 s (moduli 0.95 s) | 0.65 s |

- Quality at h = 0.05: min dihedral about 16–22°, median about 63°, 0 slivers. Mean edge is 0.0506 in against 0.05 requested, so the CSG relative length maps correctly.
- Sample STLs: cylinder at h = 0.15 takes 1.4 s, volume error 0.04 %. l_bracket at h = 0.2 takes 2.2 s, volume error 0.01 %.

## Open risks
- fTetWild is multithreaded and not bit-deterministic. Node counts vary about 0.5 % between runs. The tests use tolerances. Pass `num_threads=1` if exact repeatability is needed.
- fTetWild time dominates and grows steeply: 2 M tets takes about 1 min. C5 should size `tet_edge_length_factor` with that in mind.
- The overlap pass uses `Trimesh.contains`, which is unreliable on non-watertight parts. It runs only where part bboxes overlap.
- Faces against a removed FILTER get contact code 8. C4 lists an HTC only for CORE contact, so C4 must decide the FILTER-contact HTC.
- `voxel_neighbors` is no longer 26-slot canonical. Any consumer that reads slots 0–5 as ±X/±Y/±Z must branch on `mesh_type`. That work belongs to C4 and C6.

## Status
All C1 attributes and methods are implemented. 21 of 21 tests pass, and the boundary tests pass. Nothing is staged or committed.
