# Wave 3: ML seam (surrogate + deep model on tet runs)

## Files changed

- `core/surrogate/fields.py`: `is_tet_mesh`, `EncodingPoints`, `encoding_points(mesh, resolution)`, `_grid_cells` (the floor binning shared by `_scatter` and `unproject_grid`, same math as before). `encode_channels(..., *, points=None)` is a new optional keyword argument.
- `core/surrogate/dataset.py`: `mesh_type` stored in records and in index rows. `load_record` returns `mesh_type`. `dataset_summary` normalizes it. On tets, `_config_for_engine` uses the edge length.
- `core/surrogate/conditioning.py`: comment only. It documents what the `voxel_size` feature means on a tet run.
- `tests/test_tet_ml.py` (new, 10 tests, about 3.5 s).
- Not changed: `core/deep_model/dataset.py` and `core/surrogate/training.py`. Mixed voxel and tet records already load and train. `test_capture_writes_tet_record_that_loads_with_voxel_records` checks this through `DeepDataset`.

## Behaviour as implemented

1. Voxel mesh: `encoding_points` returns the voxel centres with an identity `values()`. The output is the same as before, and every existing voxel test passes unchanged.
2. Tet mesh: `encode_channels` does not scatter the raw node cloud.
   - A `TetLatticeSampler` is built from `node_xyz`/`tets`/`tet_part_types` at pitch `min(base_voxel_size, grid cell)`. If that lattice exceeds `MAX_LATTICE_CELLS`, the pitch becomes the grid cell size.
   - These per-node fields are sampled barycentrically at the cell centres: `surface_distances`, `voxel_moduli`, `_get_normalized_heights()`, chill boost and riser slowdown.
   - Part type comes from `cell_part_types`, cast to int32 as on the voxel path.
3. The grid frame (`GridMeta.bbox`) on a tet run is the node box, which is the part box. This is also the voxel path's frame, because its voxel centres start on the part bounds. The test checks that the two boxes agree to 1e-3 in.
4. Boundary cells:
   - Lattice centres sit at least h/2 inside the surface. A grid cell cut by the surface can therefore hold nodes but no lattice centre.
   - The nodes in those cells only (`EncodingPoints.extra_nodes`) are added as points with their own values.
   - Every node then unprojects from an occupied cell. Without this, 17 % of nodes landed in empty cells at R = 16 and got a start/local time of 0. Interior cells average lattice samples only.
5. `unproject_grid` still uses node coordinates. `tests.test_tet_pipeline.SurrogateOnTetTests` passes unchanged.
6. `capture_run` on a tet run:
   - Channels and start/end/local targets are encoded from the same `EncodingPoints`.
   - The record gets `mesh_type="tet"`.
   - The index `voxel_count` column holds the node count.
   - The conditioning `voxel_size` is `mesh.base_voxel_size`, the tet edge length. The caller's `voxel_size` is ignored on tets.
   - `mesh_type` is not in the conditioning vector, so the vector length and model input shape are unchanged.
7. Index:
   - `mesh_type` is appended as the last column. A fresh index gets the header `...,path,mesh_type`.
   - A legacy 6-column index is never rewritten. New rows carry the value as a trailing unnamed field, and `dataset_summary` reads it as `mesh_type`.
   - Old rows and old records without the field read as "voxel".
   - `data/surrogate_dataset/` was not touched: 367 index lines before and after.

## Measured: voxel vs tet, same part, h = 0.2 in, R = 8

| case | occupancy IoU | part type on overlap | moduli corr | height corr |
|---|---|---|---|---|
| box 2.0 x 1.6 x 1.2 | 1.00 | equal (all cast) | 0.94 | 0.99 |
| box + 0.8 x 0.8 x 0.4 chill | 1.00 | 95 % of cells equal | 0.91 | 0.99 |

- Mixed part-type cells sit on the cast/chill interface. On the voxel path, interface-plane voxel centres take the higher code (chill), so the voxel chill is one layer thicker. Every tet chill cell is also a voxel chill cell, and the test asserts this.
- The chill-boost channel correlates only 0.65-0.73. That field comes from the engine's per-node chill computation, not from the encoding.

## Deviations

1. The pitch is `min(edge length, grid cell)`, not the edge length alone. When h is coarser than the grid cell, a voxel run leaves interior holes in a phase-dependent pattern (at most one voxel per cell). The tet path cannot reproduce that pattern, so it samples one point per grid cell with no holes.
   - Consequence: at R >= 16 with h = 0.2 the voxel and tet occupancy differ by design (IoU 0.33 at R = 16). Tests compare at R = 8, where the cell (0.25 in) is larger than h.
2. Boundary cells use raw node values (behaviour item 4). The brief said "lattice-sampled values". This applies only to cells with no lattice centre, and it is needed so that no node reads an empty cell.
3. I did not use `core.tet_filling.build_filling_lattice`. It adds FILTER rasterization and a node KD-tree that the encoding does not need. The sampler is built directly.

## Tests

| command | result |
|---|---|
| `venv/bin/python -m unittest tests.test_tet_ml` | 10 run, OK (run 4 times, all OK, about 3.5 s each) |
| `venv/bin/python -m unittest tests.test_surrogate tests.test_surrogate_split tests.test_surrogate_geometries tests.test_surrogate_chill_channel tests.test_deep_model tests.test_architecture_boundaries tests.test_tet_pipeline.SurrogateOnTetTests` | 102 run, OK, 2 skipped (tflite toolchain) |

## Open risks

1. `core/surrogate/pipeline.py::_build_config` (not my file) still conditions on `inputs.voxel_size`, while capture uses the tet edge length, which is `voxel_size * tet_edge_length_factor`. They agree at the default factor of 1.0. For any other factor, the pipeline owner needs a one-line change: on a tet run, use `voxel_mesh.base_voxel_size`.
2. Models trained on voxel data saw the voxel path's h/2 surface dilation and its interior holes when h is coarser than the grid cell. Tet inputs are close but not identical, so expect a small bias until tet records are in the training set.
3. During the `NumpyWeightsModel.predict` matmul, numpy/Accelerate emits spurious `RuntimeWarning: divide by zero encountered in matmul`. It reproduces with random arrays in a two-line script, so it is environmental, and the outputs are finite.
4. The new index rows under a legacy header have 7 fields. `csv.DictReader` handles this. A strict reader such as `pandas.read_csv` with default options would reject the file. In-repo, only `dataset_summary` reads the index.
