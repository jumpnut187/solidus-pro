# Tet mesh migration: design and contracts

Branch `feature/tet-mesh`. This file is the authority for every subagent working
on the migration. Scout reports in this folder (`scout_engine.md`,
`scout_post.md`, `scout_geom.md`, `scout_present.md`) hold the file:line maps.

## Decisions (owner-approved)

1. The simulation mesh becomes a triangle-surface + tetrahedral-volume mesh.
   Voxels stay as a selectable fallback (`mesh_type = "voxel"`) until the tet
   path is verified. Then the default flips to `"tet"`.
2. Mesher: `pytetwild` (fTetWild, MPL-2.0). It needs `pyvista` (MIT) at import
   time. Both are installed in `venv/`. gmsh and TetGen are banned (GPL/AGPL; the
   product is proprietary).
3. Thermal solver: implicit P1 FEM, backward Euler, lumped mass, scipy sparse.
4. Lattice-native stages stay on a lattice by construction: mold filling (LBM),
   the surrogate and the deep 3D U-Net, and the web/desktop field rendering
   (3D textures). They consume tet results by **barycentric sampling of nodal
   values onto a uniform lattice** (`core/tet_lattice.py`). Do not rewrite them.
5. **The DOF is the mesh node.** On a tet run, every per-point engine array
   (`voxel_states`, `solidification_*_times`, `niyama_*`, `fraction_liquid_history`
   entries, `_fill_frames` fields, all post-analysis outputs) has length
   `N = number of tet nodes`, in node order. Nothing stores per-tet or per-cell
   engine state.
6. The tet mesh object duck-types `AdaptiveVoxelMesh`: it exposes the same
   `voxel_*` attribute names with node semantics. That keeps the 6,000-line
   engine and the graph-generic post-analyses running unchanged. Code that
   needs tet connectivity checks `getattr(mesh, "mesh_type", "voxel") == "tet"`.

## Verified pytetwild facts (parent, this session)

- `pytetwild.tetrahedralize(V, F, edge_length_abs=h, epsilon=1e-3)` meshes the
  sample STLs in 1.6-3.5 s at h = 0.1 in, volume error < 0.03 %, including the
  non-watertight `l_bracket.stl`.
- `pytetwild.tetrahedralize_csg(json_path, epsilon, edge_length_r, stop_energy,
  coarsen, num_threads, loglevel)` returns a `pv.UnstructuredGrid` with a
  per-cell `"marker"` (1-based, input order of the CSG leaves). A union of
  box + touching cylinder + touching chill gave a conforming mesh with shared
  interface nodes; per-marker volumes matched the CAD within 0.1 %.
  The raw wrapper `pytetwild.PyfTetWildWrapper.tetrahedralize_csg(...)` returns
  `(V, T, marker)` arrays without pyvista. CSG takes only a RELATIVE edge length
  (`edge_length_r = h / bbox_diagonal`).
- Output tets have **negative** orientation. Reorder to positive.
- The C++ code prints to stdout regardless of loglevel. Redirect fd 1 (os.dup2)
  around the call.
- CSG leaf format: `{"operation": "union", "left": <leaf|node>, "right": <leaf|node>}`
  where a leaf is a mesh file path (OBJ works).

## Units

Geometry, mesh and UI stay in **inches**. `physics/tet_heat_transfer.py` is
**SI only** (metres, W/m/K, J/m3/K). Convert at the boundary with
`physics.units`. Never add a bare `0.0254`.

## Contracts

### C1. `core/tet_mesh.py` (wave 1, owner: mesh worker)

```python
build_tet_mesh(
    parts: Sequence[tuple[trimesh.Trimesh, int]],   # world coords, inches, part code
    edge_length_in: float,
    *, epsilon_rel: float = 1e-3, stop_energy: float = 10.0,
    num_threads: int = 0, progress_callback: Callable[[float, str], None] | None = None,
) -> TetVolumeMesh
```

- Input is exactly `PartManager.get_parts_with_types()`. A single bare mesh is
  `[(mesh, PART_TYPE_CAST)]`.
- Mesh all parts in one CSG union so interfaces are conforming (shared nodes).
  Overlapping parts: a tet whose centroid lies inside more than one part takes
  the **highest code** (same rule as `AdaptiveVoxelMesh.assign_part_types`).
- CORE (4) and FILTER (8) tets are **removed** from the thermal domain after
  meshing. Faces they expose become boundary faces (sand/ceramic contact).
- Re-index nodes after removal. Orient every tet positively.
- Raise `config.exceptions.MeshGenerationError` on failure, empty output or
  non-finite coordinates.

`TetVolumeMesh` attributes (N nodes, T tets, F boundary faces):

| attribute | shape / dtype | meaning |
|---|---|---|
| `mesh_type` | `"tet"` | discriminator |
| `node_xyz` | (N,3) float64, in | node coordinates |
| `tets` | (T,4) int32 | positive orientation |
| `tet_volumes` | (T,) float64, in3 | |
| `tet_part_types` | (T,) int8 | part code per tet |
| `boundary_faces` | (F,3) int32 | outward-oriented boundary triangles (faces of exactly one tet) |
| `boundary_face_areas` | (F,) float64, in2 | |
| `boundary_face_part_types` | (F,) int8 | code of the owning tet |
| `boundary_face_contact_codes` | (F,) int8 | code of the removed part across the face (CORE/FILTER), else -1 (sand mold) |
| `voxel_centers` | (N,3) float32, in | `node_xyz` as float32 |
| `voxel_volumes` | (N,) float32, in3 | lumped: sum of incident tet volume / 4 |
| `voxel_sizes` | (N,) float32, in | local h = mean incident edge length (NOT a lattice pitch) |
| `voxel_surface_areas` | (N,) float32, in2 | lumped boundary area: sum of incident boundary face area / 3 |
| `surface_voxels` | (N,) bool | `voxel_surface_areas > 0` |
| `voxel_neighbors` | (N,K) int32, -1 padded | edge adjacency, K = max degree, ascending order |
| `adj_indptr`, `adj_indices` | CSR int64 / int32 | same adjacency |
| `voxel_part_types` | (N,) int8 | node code: max metal code among incident metal tets if any incident tet is metal, else max code among incident tets |
| `voxel_moduli` | (N,) float32, in | distance from node to nearest boundary face, 0 on boundary (same semantics as voxel path) |
| `voxel_states` | (N,) float32 | zeros |
| `base_voxel_size` | float, in | requested `edge_length_in` |
| `mean_edge_length` | float, in | measured |
| `bounds` | (3,2) float64 | `[[xmin,xmax],...]` |
| `metal_part_bounds` | list of (2,3) float64 | per metal part |
| `grid_shape` | `None` | no lattice |
| `original_voxel_size`, `voxel_size_was_adjusted`, `adjustment_message` | mirrors | `edge_length_in`, False, "" |

Methods mirroring `AdaptiveVoxelMesh`: `get_voxel_count()`,
`get_surface_voxel_indices()`, `get_voxels_by_type(code)`,
`get_material_surface_voxel_indices()` (metal nodes shared with a CHILL tet, or
on a face with contact code CORE), `get_memory_stats()`, `print_memory_stats()`,
`cleanup()`, plus any other public member the engine/pipeline reads (the mesh
worker greps for them). Also `quality_summary()` -> dict (min/median dihedral
angle in degrees, sliver count with min dihedral < 5 deg, node/tet/face counts).

### C2. `physics/tet_heat_transfer.py` (wave 1, owner: solver worker)

Mesh-agnostic, SI only, imports stdlib/numpy/scipy/numba only (no `core`,
`config`). Backward Euler, lumped mass, properties lagged one step.

```python
@dataclass(frozen=True)
class TetMaterial:
    temp_points: np.ndarray        # (P,) degC, ascending; len 1 = constant
    conductivity: np.ndarray       # (P,) W/m/K
    rho_cp: np.ndarray             # (P,) J/m3/K  (density * specific heat)
    latent_heat_vol: float = 0.0   # J/m3, 0 for non-solidifying materials

class TetHeatTransferModel:
    def __init__(self, nodes_m, tets, tet_material, materials, boundary_faces): ...
    def add_boundary_group(self, face_indices, htc, t_inf,
                           time_function: Callable[[float], float] | None = None) -> int: ...
    temperatures: np.ndarray (N,) float64 degC, mutable in place
    time: float  # s
    def set_initial_temperature(self, value: float | np.ndarray) -> None
    def step(self, dt: float) -> None
    def node_heat_capacity(self) -> np.ndarray     # (N,) J/K at current T
    def node_latent_capacity(self) -> np.ndarray   # (N,) J, sum latent_heat_vol * V/4
    def nodal_gradient(self) -> np.ndarray         # (N,3) K/m, volume-weighted P1 tet gradients
    def gradient_magnitude(self) -> np.ndarray     # (N,) K/m
    def get_cooling_rate(self, prev, dt) -> np.ndarray  # degC/s, positive when cooling
    def total_energy(self) -> float                # sum C_i T_i, for conservation tests
    last_iterations: int; last_residual: float; backend_description: str
```

- Step: `(C/dt + K + R(t+dt)) T_new = (C/dt) T_old + R(t+dt) T_inf`, with
  `R` the lumped Robin term (`h(t) * A_face / 3` per node per group, `h(t) = htc
  * time_function(t)`). `C_i = sum_e rho_cp(T_e) V_e / 4`, `T_e` = mean of the
  element's 4 nodal temperatures. `K` assembled from per-element `k(T_e)`.
- Precompute per-tet shape-function gradients and the CSR pattern once; per
  step only refill `data` with `np.bincount` (or a numba kernel). No Python loop
  over tets.
- Solve with `scipy.sparse.linalg.cg`, Jacobi preconditioner, `x0 = T_old`,
  rtol 1e-10. Fall back to `spsolve` if CG does not converge.
- Unconditionally stable: no CFL sub-stepping.

### C3. `core/tet_lattice.py` (wave 1, owner: lattice worker)

```python
class TetLatticeSampler:
    def __init__(self, node_xyz_in, tets, tet_part_types, spacing_in, *, inside_tol=1e-9): ...
    dims: tuple[int, int, int]; origin: np.ndarray (3,)  # centre of cell 0, inches
    spacing: float
    cell_flat: np.ndarray (M,) int64   # x-fastest: ix + nx*(iy + ny*iz), matches core/dense_grid.py
    cell_centers: (M,3) float64; cell_tet: (M,) int32; cell_bary: (M,4) float64
    cell_part_types: (M,) int8         # from the containing tet
    nearest_node: (M,) int32           # node with the largest barycentric weight
    def sample(self, node_values) -> np.ndarray      # (N,) -> (M,), (F,N) -> (F,M)
    def dense(self, node_values, fill=np.nan) -> np.ndarray  # (nx*ny*nz,) float32
    def mask_codes(self) -> np.ndarray               # uint8 dense, code+1 inside, 0 outside
    def lattice_view(self) -> LatticeView
```

- Lattice covers the mesh bounding box, cell centres at `origin + spacing*ijk`.
- A cell is inside when its centre lies in a tet (barycentric >= -inside_tol).
  Point location must be vectorized and memory-bounded (chunked), able to
  handle 5 M cells.
- `LatticeView` duck-types the voxel mesh for lattice consumers (filling, ML):
  `voxel_centers` (M,3) float32, `voxel_sizes` (M,) = spacing, `voxel_volumes`,
  `voxel_part_types` (M,) int8, `voxel_neighbors` (M,26) int32 in the canonical
  slot layout of `core/voxel_mesh.py` (slots 0-5 = +X,-X,+Y,-Y,+Z,-Z faces,
  6-17 edges, 18-25 corners; read `_canonicalize_neighbor_order` for the exact
  order), `surface_voxels`, `voxel_surface_areas`, `voxel_moduli`,
  `base_voxel_size`, `bounds`, `grid_shape`, `get_voxel_count()`,
  `get_surface_voxel_indices()`, `get_voxels_by_type()`, and `sampler` (back
  reference). `mesh_type = "lattice"`.

### C4. Engine integration (wave 2, owner: engine worker)

- New `core/tet_thermal.py` holds all tet-specific engine logic. Edits to
  `core/solidification.py` and `core/solidification_loop.py` are dispatch lines
  only.
- Tet mesh forces `thermal_coupled` (log a warning if `schedule_legacy` was
  requested).
- Materials: metal tets use the same 6-point alloy curves the voxel path builds
  (`core/solidification.py` ~:450), with `rho_cp = density * specific_heat` and
  `latent_heat_vol = L * (rho_l + rho_s)/2`. CHILL tets: constant
  `rho_cp = 1800*710`, `k = 1.17e-4 * rho_cp`, no latent heat.
- Boundary groups: one group per boundary-face part code, `h0` and
  `_mold_interface_decay` exactly as `_configure_thermal_boundaries` assigns
  them per part type (CAST 500, RISER_POST 350, RISER 220, flow path 400,
  CORE contact 500, CHILL `chill_ambient_htc` with floor 0.7). Faces with
  contact code CORE use the CORE h0.
- Latent heat: same operator-split enthalpy correction as the voxel loop, with a
  **per-node** `beta_i = L_i / C_i` (K). Nodes with `L_i = 0` skip it.
- Gradients: Niyama and `thermal_gradients` use `nodal_gradient()` (K/m ->
  degC/mm via `physics.units`). Niyama quality = 2 for every metal node.
- Time step: implicit. Start at `engine.time_step`, grow x1.25 per step up to
  `settings.mesh.tet_max_time_step_s` (2.0), halve when the max metal nodal
  |dT| in a step exceeds `settings.mesh.tet_max_delta_t_c` (10.0).
- Termination, sentinels (0 start for chills, NaN unfinished end), 1 %
  fraction-liquid snapshots, feeding fraction, peak cooling rate: identical to
  `_run_thermal_coupled_integration`.

### C5. Pipeline, settings, filling (wave 2, owner: pipeline worker)

- `MeshSettings`: `mesh_type` ("voxel"|"tet", default "voxel" until wave 4),
  `tet_edge_length_factor` (1.0, x base_voxel_size), `tet_epsilon_rel` (1e-3),
  `tet_max_time_step_s` (2.0), `tet_max_delta_t_c` (10.0),
  `tet_lattice_spacing_factor` (1.0, x base_voxel_size). Env overrides
  `CASTSIM_MESH_TYPE`, etc. YAML/TOML defaults updated.
- `PipelineInputs.mesh_type: str | None` (None -> settings).
  `prepare_mesh_and_geometry` builds `TetVolumeMesh` when tet.
- Filling on a tet run: build `TetLatticeSampler(...).lattice_view()` at the
  lattice pitch, run the existing filling stage on it, then map
  `temperature_c`, `fill_time_s` and every `_fill_frames` field to nodes
  (nearest inside lattice cell) before `set_initial_temperature_field`.
- `core/surrogate/pipeline.py` must work with a tet mesh (prediction unprojects
  onto node coordinates).

### C6. Wave 3 ports (post-analyses, presentation, ML, UI)

- Post: `defect_regions` treats the full adjacency row as connectivity on tets;
  scalar `base_voxel_size` length scales use local `voxel_sizes`; volume sums
  use `voxel_volumes`; isolation score normalized by node degree; riser design
  uses node volumes.
- Presentation: tet runs ship `tets`, `tet_part_types`, `mesh_type` in the grid
  source; `web/fields.py` and `ui/result_surface.py` build the lattice with
  `TetLatticeSampler` instead of `derive_lattice`; the sidecar/frontend
  contract (dims, origin, spacing, uint8, x-fastest) is unchanged; picking maps
  a cell to `nearest_node`.
- ML: capture and inference encode channels from the lattice view, not the raw
  node cloud. Records store `mesh_type` (missing = "voxel").
- UI: mesh-type toggle in desktop `MeshDialog` and web `MeshPanel`, plumbed
  through `ui/workers.py`, `ui/project_state.py`,
  `web/services/simulation_queue.py`; `web/routes.py` previews estimate tet
  node counts.

## Rules for every worker

- Edit only the files your brief assigns. Other workers run in the same
  checkout at the same time.
- Do not commit, stash, reset or switch branches. The parent commits.
- Use `venv/bin/python`. Tests use `unittest` (`tests/test_*.py`).
- Keep the layer contract (`tests/test_architecture_boundaries.py`).
- Report: files changed, public API, test commands run with pass/fail counts,
  open risks.
