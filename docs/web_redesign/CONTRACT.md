# Web Redesign: Working Contract

Two workstreams build the new Solidus web application in parallel. This document is the
interface between them. Read it fully before touching code. `desktop_inventory.md` beside it
is the feature-parity source of truth.

## Product direction

- Results render the way commercial casting simulators render them: smooth, continuous color
  painted on the real casting geometry, capped section cuts that show the interior field,
  smooth isosurface envelopes for defect regions, and smooth time animation. Never voxel cubes.
- Default colormap is `rainbow` (blue through red). The eight desktop maps remain selectable.
- Every feature the desktop app exposes is reachable in the browser, except the Surrogate and
  Deep Model menus, which are deliberately out of scope.
- Everything fits a 1366 by 768 window without scrolling the 3D canvas.

## Hard rules

1. Vocabulary: do not write the brand names of commercial simulation products anywhere in
   code, comments, docs, tests, commit messages, or notes. Say "commercial casting simulators"
   or "industry-standard". The repository was scrubbed; keep it that way.
2. Stay inside your owned paths (table below). Shared files are additive-only for the side
   that does not own them, and only where this document says so.
3. TypeScript strict, no `any`, `npm run lint` and `npm run build` clean. Python: type hints,
   tests under `tests/`, `venv/bin/python -m pytest tests/test_web_api.py tests/test_surface_extraction.py tests/test_web_simulation_process.py -q` green plus your new test files.
4. New dependencies: `@babylonjs/materials` is pre-approved for the viewport. Anything else,
   stop and ask the integration lead before adding it.
5. Never edit `web/simulations.db` or anything under `web/static/` in the main checkout.
6. Commit on your own worktree branch at the end of each milestone with a clear message.
   Do not push. Do not merge. The integration lead merges into `feature/web-redesign`.
7. Do not delete the legacy `web/routes.py` endpoints; the old client contract stays until
   the lead removes it after integration.

## Ownership

| Area | Workstream A: rendering and data pipeline | Workstream B: shell, UX, parity |
|---|---|---|
| Python | `web/fields.py` (new), `web/routes_fields.py`, `core/surface_extraction.py`, `core/simulation_pipeline.py` (only `build_visualization_payload` and artifact writing), `web/payloads.py` (only `build_fork_artifact`, `write_artifact`, `read_artifact` and the field/glyph builders), `tests/test_web_fields.py` | `web/routes_tools.py`, `web/routes.py`, `web/repository.py`, `web/services/**`, `core/report_generator.py`, `core/riser_design.py` (wrapper functions only), `tests/test_web_tools.py`, `tests/test_web_api.py` |
| Frontend | `web_frontend/src/viewport/**`, `web_frontend/src/lib/colormaps.ts`, `web_frontend/src/lib/gridApi.ts` (new), `web_frontend/src/workers/**` (new), `web_frontend/public/**` | `web_frontend/src/App.tsx`, `main.tsx`, `src/components/**`, `src/lib/api.ts`, `src/lib/defaults.ts`, `src/lib/uiPrefs.ts`, `src/lib/projectFile.ts` (new), `src/hooks/**` (new), `src/styles/**`, `src/styles.css`, `index.html`, `package.json` |
| Shared, lead-owned | `web_frontend/src/lib/viewState.ts` (A may add fields additively; B consumes), `web/app_factory.py`, `web_frontend/vite.config.ts`, `docs/web_redesign/**` |
| Legacy to retire | A deletes `src/components/BabylonViewport.tsx` when the new viewport no longer needs it | B deletes `src/components/ResultLegend.tsx`, `ConfigDrawer.tsx`, `Sidebar.tsx`, `drawers/**` as they are replaced |

Workstream A must not touch `src/components/**` except to delete `BabylonViewport.tsx`.
Workstream B must not touch `src/viewport/**`; it renders `<ResultsViewport>` and `<ColorBar>` from
there and treats them as black boxes.

## The component contract

`web_frontend/src/lib/viewState.ts` defines everything the two sides exchange:

- `ResultViewState` is owned by the shell (React state in `App.tsx`, persisted to localStorage)
  and passed to the viewport as `view`. The viewport requests changes through `onViewChange`
  (advancing playback, snapping auto scale bounds); it never mutates state itself.
- `ViewportProps` is the full prop surface of `<ResultsViewport>`; `ViewportHandle` is the
  imperative surface reached through a ref (screenshot, camera presets, clear measurements,
  defect volume).
- `GridMeta` is what the viewport reports back after loading results: available modes, ranges,
  frame timings, counts, and the summary statistics dictionary. The shell populates the result
  type menu, the legend range, and the statistics panel from it.
- `ColorBarProps` is the legend component; the shell decides where it sits.
- `RESULT_MODE_IDS` is the canonical mode list. Modes without data for a run are simply absent
  from `GridMeta.modes`; the shell disables them.

The placeholder `src/viewport/ResultsViewport.tsx` implements this contract on top of the legacy
renderer so the shell can be built and exercised on day one. Workstream A replaces it in place.

## Data flow (Workstream A designs the details, these names are fixed)

The voxel lattice is uniform at `base_voxel_size`; `AdaptiveVoxelMesh.grid_shape` and the
padded bounds give a dense grid. Store the full lattice as a compact binary sidecar next to the
JSON artifact and serve dense grids from it:

| Endpoint | Returns |
|---|---|
| `GET /api/projects/<id>/grid` | JSON: dims, origin, spacing, available modes with ranges and units, frame counts and times, part-type code table, statistics |
| `GET /api/projects/<id>/grid/mask` | binary Uint8 grid, 0 = outside, otherwise part-type code |
| `GET /api/projects/<id>/grid/<mode>` | binary Float32 (or normalized Uint8/Uint16 with a scale header) dense grid, outside cells filled with the nearest inside value so trilinear sampling does not bleed |
| `GET /api/projects/<id>/grid/<mode>/frames?start=&count=` | consecutive time frames for `fraction_liquid` and `fill_animation`, quantized to Uint8 |
| `GET /api/projects/<id>/surface?...` | existing isosurface endpoint, extended to every scalar mode, binary payload allowed |

Grid order is x fastest: `index = ix + nx * (iy + ny * iz)`, matching a WebGL 3D texture whose
width is `nx`. Responses are gzip-encoded when the client accepts it. Artifacts are cached in
process with an LRU keyed by project id and file mtime; never re-read a 100 MB JSON per request.

Workstream B's new endpoints live under these prefixes so nothing collides:
`/api/projects/<id>/rename`, `/api/projects/<id>/duplicate`, `/api/projects/<id>/export`,
`/api/projects/import`, `/api/projects/<id>/riser-design`, `/api/projects/<id>/report`,
`/api/projects/<id>/log`, `/api/hardware`, `/api/help/alloys`.

## Environment

- Python: always `/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim/venv/bin/python`
  (the main checkout's venv, usable from any worktree). Never `pip3`.
- Node 22 is on PATH through nvm. Run `npm install` once inside your worktree's `web_frontend/`.
- Backend in a worktree: `venv/bin/python run_web.py --dev --port <your port>`; A uses 5061,
  B uses 5062. The user's own server on 5050 runs old code; do not rely on it and do not stop it.
- Frontend dev server: `CASTSIM_API_PORT=<your port> npm run dev` (Vite on 5175, proxying `/api`).
- Test data: completed run `ef17b6fc-ac9` with artifact
  `/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim/web/static/temp/artifacts/ef17b6fc-ac9.json`
  (50 MB, uniform 0.2 inch lattice, 77 by 31 by 43, 41,593 voxels, 93 fraction-liquid frames) and
  uploaded STLs `web/static/uploads/602e27b4_b0_cast.stl` and `web/static/uploads/00f638b5_b0_riser.stl`.
  Copy `web/simulations.db`, the artifact, and those two STLs into the same relative paths of your
  worktree to seed it. Both STLs are also in `Test/` and `Test_Model/`.
- A quick real run: those two parts at 0.2 inch voxels finish in about four minutes.
- Visual checks: the lead's puppeteer harness is at
  `/private/tmp/claude-501/-Users-brendonjarett-Desktop-Active-Codebase-Python-Cast-Sim/f5c14e95-bdeb-42ae-b568-1d513ab86a75/scratchpad/uitest/`
  (`audit.mjs`, `node_modules` included). Copy it, point `BASE` at your port, and screenshot your own
  work before declaring a milestone done. Headless Chrome renders WebGL through SwiftShader there.

## Milestones

| | Workstream A | Workstream B |
|---|---|---|
| M1 | Binary sidecar and grid endpoints with tests. New `ResultsViewport`: environment lighting, 0 to 100 percent opacity, smooth field painting on part surfaces through a 3D texture and colormap LUT, `ColorBar` with ticks and units, camera presets with animated transitions, no camera re-fit on unrelated prop changes, solidification-time mode. | New shell: top bar, left icon rail with one setup panel at a time, right results panel, bottom transport bar, modals. Project list, open, save, rename, delete, restore on refresh. Parts panel with 10 types, clean names, rename, visibility, arrange. Solidification parameters as a tabbed modal. Analysis, mesh, material, performance panels at desktop parity (GPU toggle, gas porosity). |
| M2 | Capped sections on all three axes, isosurface envelopes for threshold modes, client-side contouring for fraction-liquid and fill fronts with smooth playback, probe and measure tools, fill animation frames, statistics in `GridMeta`, ambient occlusion and anti-aliasing. | Riser design modal and endpoint, measurement and defect volume UI, PNG screenshot and animation export, working report endpoint and modal, help guide, hardware info, job log panel, statistics panel, keyboard shortcuts, project file import and export compatible with the desktop format. |
| M3 | Integration fixes from the lead's visual review, performance on 1M-voxel grids, retire legacy renderer. | Integration fixes, responsive polish at 1366 by 768, empty and error states, retire legacy components, update `web/README.md`. |

## Definition of done for a milestone

- Lint, build, and the Python test commands above pass in your worktree.
- You exercised the feature in a browser (or the headless harness) against real results and
  looked at the screenshot.
- You committed on your worktree branch and reported: what changed, what you verified, what is
  left, and any contract additions you made to `viewState.ts`.
