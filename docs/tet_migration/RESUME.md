# Resume point

Branch `feature/tet-mesh`. Commits:
1. Voxel default restored + tet auto-fallback (lane complete; report `voxel_default_fallback.md`).
2. b912460 cut cells WIP behind `settings.mesh.voxel_cut_cells`.

## Cut cells: complete (uncommitted on top of b912460)

- Report: `cut_cells_integration.md` (fixes, default decision, suite counts, benchmark).
  Earlier reports: `cut_cells.md` (implementation, C05813 runs), `review_cut_cells.md` (review).
- Review P1 (chunked duplicates) and all four P2s fixed, each with a test.
- `voxel_cut_cells` now defaults to True: every owner target is met at 0.1 in
  (thick_thin +0.8 %, l_bracket +1.0 %, assembly +0.3 %, field error < 2 %, wall <= 1.32x;
  C05813 +1.0 % at 0.1 in and +2.3 % at 0.2 in vs tet 0.2 in).
- `PipelineInputs.voxel_cut_cells`, desktop Mesh dialog and web Mesh panel checkbox
  ("Cut cells (CAD-accurate boundary)"), persisted in `performance_options.voxel_cut_cells`.
- Full suite (defaults, CASTSIM_VOXEL_CUT_CELLS=0, CASTSIM_MESH_TYPE=tet): 2141 run, only the 3
  DESIGN.md baseline failures. `npm run build` passes.

## Open items (owner decisions, not blockers)

1. Interface cells take the dominant part code with ties to the higher code (chill 2 > cast 0):
   at 0.1 in on C05813, 0.7 in3 of metal is labelled chill. Ties to metal would be a one-line change.
2. Desktop check on C05813 in the app (voxel, cut cells on) not yet done by the owner.
3. The chunked voxelizer drops z-slices (`cut_cells.md`); flag-off behaviour, untouched.
