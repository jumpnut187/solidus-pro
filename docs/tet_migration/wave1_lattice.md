# Wave 1 lattice worker report (contract C3)

## Files changed

- `core/tet_lattice.py` (new)
- `tests/test_tet_lattice.py` (new)

No other files touched. Nothing staged or committed.

## Public API as implemented

```python
CANDIDATE_CHUNK: int = 1 << 20           # candidate (tet, cell) pairs per location chunk
CANONICAL_SLOT_OFFSETS: np.ndarray       # (26, 3) int64 ijk offsets in voxel_mesh slot order

class TetLatticeSampler:
    def __init__(self, node_xyz_in, tets, tet_part_types, spacing_in, *, inside_tol=1e-9)
    dims: tuple[int, int, int]; origin: (3,) float64 (centre of cell 0, in); spacing: float
    bounds: (3, 2) float64 mesh bbox; inside_tol: float; cell_count (property) = nx*ny*nz
    cell_flat (M,) int64 ascending, x-fastest; cell_centers (M,3) float64
    cell_tet (M,) int32; cell_bary (M,4) float64; cell_part_types (M,) int8; nearest_node (M,) int32
    sample(node_values)  # (N,) -> (M,), (F,N) -> (F,M), float64
    dense(node_values, fill=np.nan)  # (nx*ny*nz,) or (F, nx*ny*nz) float32
    mask_codes()         # (nx*ny*nz,) uint8, code+1 inside, 0 outside
    lattice_view()       # LatticeView, built once and cached

class LatticeView:
    mesh_type = "lattice"; sampler; base_voxel_size; bounds; grid_shape (= dims)
    voxel_centers (M,3) float32; voxel_sizes, voxel_volumes (M,) float32
    voxel_part_types (M,) int8; voxel_neighbors (M,26) int32, -1 = not an inside cell
    surface_voxels (M,) bool; voxel_surface_areas (M,) float32 in2; voxel_moduli (M,) float32 in
    get_voxel_count(); get_surface_voxel_indices(); get_voxels_by_type(code)
```

Behaviour:

- Lattice: `dims = max(1, ceil(extent / spacing))` per axis, centred on the mesh bbox. Cells tile the bbox and no centre sits on a bbox face, so a box mesh whose extent is a multiple of the pitch gets exactly `extent / spacing` cells per axis.
- Raises `core.dense_grid.LatticeTooLargeError` above `MAX_LATTICE_CELLS` (64 M), before any allocation. Raises `ValueError` on bad shapes, out-of-range node indices, non-finite coordinates or non-positive spacing.
- Point location: each tet's bbox gives a cell ijk range. The global candidate list (all tets, ascending tet index) is walked in chunks of `CANDIDATE_CHUNK` pairs, so a single tet spanning the whole lattice is split across chunks. Already-claimed cells are dropped before the barycentric test. The first containing tet wins, which is the lowest tet index. Degenerate tets (`|det| <= 1e-12 * max_edge^3`) are skipped. Tet orientation does not matter.
- `inside_tol` is absolute on barycentric weights.
- Neighbour slots come from `CANONICAL_SLOT_OFFSETS`, built with the same sign rules as `AdaptiveVoxelMesh._canonicalize_neighbor_order`. A test compares all 26 slots with that method.
- `surface_voxels`: at least one face slot (0-5) is -1. Cells on the lattice border count as exposed. `voxel_surface_areas = exposed_face_count * spacing^2`.

## Deviations from DESIGN.md and the brief

1. `voxel_moduli` is the distance from each cell centre to the nearest surface cell centre, 0 on surface cells. The brief suggested "distance to the nearest outside cell centre minus half a cell", or a documented equivalent. I chose the voxel path's own definition (`core/voxel_mesh.py:1478-1499` queries a KD-tree of surface voxel centres) because C1 asks for "same semantics as voxel path". It is computed with `scipy.ndimage.distance_transform_edt(sampling=spacing)` on the dense lattice, which is exact Euclidean. The brief's formula would give 0.5 cell on surface cells instead of 0.
2. `bounds` (sampler and view) is the mesh node bbox, not the lattice cell extent. This matches `AdaptiveVoxelMesh.bounds`, which holds the geometry bounds.
3. Additions beyond C3: `cell_count`, `bounds`, `inside_tol` attributes, `dense()` also accepts `(F, N)`, and the module-level `CANDIDATE_CHUNK` and `CANONICAL_SLOT_OFFSETS`. Each is used by the class or its tests.
4. The view has no `voxel_states`. C3 does not list it, and none of the lattice consumers (`core/filling_stage.py`, ML field encoding) read it from the filling mesh.

## Tests

`tests/test_tet_lattice.py`, 11 tests, 1.4-1.8 s wall time:

- `test_flat_index_matches_dense_grid_helpers`: `dense_grid.flat_indices` and `derive_lattice` reproduce `cell_flat`, dims and origin.
- `test_sample_reproduces_linear_field`: pytetwild cylinder, exact to 1e-10 for `(N,)` and `(F,N)`. Also checks dense float32 values and the fill value.
- `test_inside_volume_matches_box_mesh`: pytetwild box at 0.05 in, error < 3 %.
- `test_inside_volume_matches_cylinder_mesh`: pytetwild cylinder at 0.025 in, error < 5 %.
- `test_two_region_mask_codes_and_part_types`: L-shaped Kuhn mesh with CAST and CHILL regions. Checks `mask_codes` 1/3/0, `cell_part_types` and `get_voxels_by_type`.
- `test_nearest_node_is_max_weight_node`: also checks weights are >= -tol and sum to 1.
- `test_point_on_shared_face_is_located_once`: every centre of a 2x2x2 lattice on a Kuhn cube lies in 2 to 6 tets. Each is located once, by the lowest containing tet index.
- `test_lattice_view_face_slots_are_signed_axis_neighbours`: slots 0-5 match KD-tree +X,-X,+Y,-Y,+Z,-Z neighbours. All 26 slots match `AdaptiveVoxelMesh._canonicalize_neighbor_order`.
- `test_box_surface_flags_areas_and_moduli`: border flags, areas, moduli against a KD-tree of surface centres, sizes, volumes, grid_shape and bounds.
- `test_tet_larger_than_one_chunk_is_located`: `CANDIDATE_CHUNK=37` gives the same result as one chunk.
- `test_rejects_bad_input`.

Commands:

- `venv/bin/python -m unittest tests.test_tet_lattice -v`: 11 run, 11 pass.
- `venv/bin/python -m unittest tests.test_architecture_boundaries tests.test_web_fields tests.test_neighbor_canonicalization`: 88 run, 88 pass (5 skipped).
- `venv/bin/python -m compileall -q core/tet_lattice.py tests/test_tet_lattice.py`: passed.

## Performance (Apple M5 Pro, single process)

| case | tets | lattice cells (inside) | locate | sample | dense | lattice_view |
|---|---|---|---|---|---|---|
| Kuhn 55^3 grid | 998,250 | 1,000,000 (1,000,000) | 0.57 s | 0.007 s | 0.008 s | 0.63 s |
| pytetwild cylinder r=1 h=3 | 337,404 | 1,290,575 (1,005,396) | 0.52 s | 0.007 s | 0.008 s | 0.65 s |
| Kuhn 55^3 grid | 998,250 | 5,000,211 (5,000,211) | 1.83 s | 0.040 s | 0.039 s | 3.20 s |

Memory at 5 M cells and 1 M tets, measured with tracemalloc:

- Sampler: 938 MB peak, 424 MB retained.
- View: 1.61 GB peak, 1.06 GB retained. `voxel_neighbors` alone is 520 MB, the same cost as the voxel path.

## Open risks

- Location memory is O(T + M) for per-tet setup and results (about 500 MB of transient arrays at 1 M tets and 5 M cells). Only the candidate expansion is chunked. To lower the peak further, chunk the per-tet setup as well.
- Cells whose centre lies in no tet are outside. Near a curved boundary the inside count depends on how the tet surface approximates the CAD (fTetWild `epsilon`).
- If two parts overlap in the mesh, the lowest tet index wins at a shared face. C1 resolves overlaps per tet before this stage, so this applies only at conforming interfaces.
- The tests import `core.voxel_mesh` for the slot cross-check. That pulls in the Taichi/MLX startup banner but adds less than 1 s.
