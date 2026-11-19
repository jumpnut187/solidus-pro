# Solidus Pro — Web Application

The browser interface to the casting solidification and mold-filling engine.
Everything the desktop application exposes is reachable here, except the
Surrogate and Deep Model menus, which are deliberately out of scope.

## Running it

```bash
pip install -r requirements.txt          # from the repository root
cd web_frontend && npm install && npm run build && cd ..
python run_web.py                        # serves the built frontend on 5050
```

Then open <http://localhost:5050>.

For frontend work, run the two servers separately so Vite can hot-reload:

```bash
python run_web.py --dev --port 5062      # API only
cd web_frontend && CASTSIM_API_PORT=5062 npm run dev   # Vite on 5175, proxying /api
```

## Architecture

```
run_web.py
  └── web/app_factory.py        Flask app; registers three blueprints
        ├── web/routes.py       projects, parts, uploads, runs, legacy payloads
        ├── web/routes_fields.py dense-grid endpoints the 3D viewport streams
        └── web/routes_tools.py riser design, reports, project files, log, hardware
      web/repository.py         SQLite persistence (projects, parts)
      web/payloads.py           CAD upload, mesh payloads, artifact read/write
      web/fields.py             dense field grids and the binary sidecar
      web/services/simulation_queue.py  the job queue and its worker process

web_frontend/src
      App.tsx                   the shell: state, project lifecycle, modals
      components/               top bar, icon rail, panels, results panel, modals
      viewport/                 the 3D renderer (owns its own contract)
      lib/viewState.ts          the contract between the shell and the viewport
      hooks/                    simulation job polling, job log, shortcuts, toasts
      styles/                   design tokens and the layout that uses them
```

**Data flow for a run.** The client creates a project row, uploads CAD files
(one part per body; a multi-body STEP fans out), then POSTs to `/run`. The
queue manager starts a separate OS process per job — separate so the solver
gets a clean Numba thread pool and cannot contend with Flask — and the worker
reports progress and log lines back through `multiprocessing.Manager` proxies.
On completion it writes a JSON artifact under `static/temp/artifacts/<id>.json`
plus a binary sidecar, records the path on the project row, and drops its
in-memory copy. Everything downstream re-reads from disk through small caches.

**Layout.** The interface is a CSS grid: top bar, optional connection banner,
workspace, transport bar. The workspace is a flex row of icon rail, at most one
docked setup panel, the canvas, and the results panel. Opening a panel narrows
the canvas rather than covering it, and nothing scrolls the 3D view — the whole
thing fits 1366x768.

## Storage

- `web/simulations.db` — SQLite. `projects` holds settings as JSON columns
  (`analysis_options`, `performance_options`, `solidification_params`,
  `ui_state`); `parts` holds one row per body.
- `web/static/uploads/` — stored CAD files, named `<file_id>_b<body>_<name>`.
  **A part's `filepath` must resolve inside this folder.** It is checked on
  every write (`repository.resolve_upload_path`) because the folder is served
  over HTTP and the duplicate endpoint copies part files into it; without the
  check, a part pointing anywhere on disk was a file-read primitive.
- `web/static/temp/artifacts/` — run artifacts, 50–180 MB of JSON each.

The pour location lives inside the `ui_state` blob and is mirrored to a
top-level `pour_location` key on read; write it through `repository`, never
straight into the column.

## API

All routes are under `/api`.

### Projects and parts
| Method | Path | Purpose |
|---|---|---|
| GET | `/projects` | list projects, newest first |
| POST | `/projects` | create from a settings body |
| GET | `/projects/<id>` | project and its parts; `?include_mesh=1` attaches geometry |
| PUT | `/projects/<id>` | replace settings |
| DELETE | `/projects/<id>` | delete the project, its parts and its files |
| POST | `/projects/<id>/rename` | rename |
| POST | `/projects/<id>/duplicate` | copy parts and settings, not results |
| POST | `/projects/<id>/parts` | add a part |
| PUT/DELETE | `/projects/<id>/parts/<part_id>` | update or remove a part |
| GET | `/projects/<id>/parts/<part_id>/mesh` | triangles for one stored part |
| POST | `/uploads` | upload CAD files; returns one entry per body |
| GET | `/uploads/<filename>` | serve a stored CAD file |

### Running
| Method | Path | Purpose |
|---|---|---|
| POST | `/projects/<id>/mesh-preview` | voxel count and memory estimate |
| POST | `/projects/<id>/ml-mesh` | recommended voxel size and refinement |
| POST | `/projects/<id>/run` | queue a simulation |
| POST | `/projects/<id>/cancel` | stop a queued or running job |
| GET | `/jobs/<id>/status` | status, progress and message |
| GET | `/projects/<id>/log?since=<n>` | job log; `since` is a monotonic cursor, and the reply carries `next` and `dropped` |

### Results
| Method | Path | Purpose |
|---|---|---|
| GET | `/projects/<id>/grid` | dims, origin, spacing, modes, ranges, frames, statistics |
| GET | `/projects/<id>/grid/mask` | binary part-type mask |
| GET | `/projects/<id>/grid/skin` | binary signed distance to the parts' real surface, in voxels; 404 when the part files are gone |
| GET | `/projects/<id>/grid/<mode>` | binary dense field |
| GET | `/projects/<id>/grid/<mode>/frames` | consecutive time frames |
| GET | `/projects/<id>/grid/<mode>/isosurface` | contoured geometry as a binary block |
| GET | `/projects/<id>/results` | summary statistics |

Everything the viewport draws comes from `/grid`. The sampled-field, surface
and volume JSON routes that preceded it were removed once nothing in the
client read them; a request for one now 404s.

### Tools
| Method | Path | Purpose |
|---|---|---|
| GET | `/report/formats` | which report formats this server can render, and why not |
| POST | `/projects/<id>/report` | build a report and download it |
| GET | `/projects/<id>/export` | download a `.solidus-project.json` |
| POST | `/projects/import` | create a project from one, plus its CAD files |
| GET | `/materials` | the seven alloys with their full properties |
| GET | `/help/alloys` | alloy reference table for the help guide |
| GET | `/hardware` | CPU, memory, GPU backend and core tiers |

## Notes for anyone changing this

- **Report output is escaped.** `core/report_generator.py` builds HTML with
  f-strings, so every user string goes through `_esc` (and `_esc_md` for
  Markdown). A project name is user input and a report gets mailed around.
- **PDF is optional.** `_html_to_pdf` degrades *silently* when neither
  `weasyprint` nor `pdfkit` is installed: it writes an `.html` next to the path
  it promised and raises nothing. `/report/formats` probes for the backend so
  the UI can grey the option out, and the endpoint verifies the file exists
  before serving it.
- **Artifacts are big.** Do not hold one in the queue Manager or read one per
  request; `routes_tools._read_artifact_cached` keeps a two-entry LRU keyed by
  path, mtime and size. Prefer the summary on the project row where it will do.
- **Riser design should move to the sidecar.** It reads the strided JSON
  artifact and refuses a downsampled run, because a thinned surface set inflates
  every modulus. The intended follow-up is to read the full-resolution binary
  sidecar instead, which removes the refusal and the voxel cap that causes it.
- **The job worker is a separate process** started with `spawn`, so it shares
  nothing implicitly. Anything it needs to report has to travel through a
  Manager proxy passed in its argument list — including the recursive call at
  the end of the worker, which is how a queued job is chained onto a finishing
  one.
- **Project files** follow the desktop schema (`ui/project_state.py`, v2, with
  v1 migrated on load). The web row is flat where the file nests under
  `simulation`, and the file calls the alloy `alloy_name`; translate at the
  endpoint, not by hand elsewhere.
- **Vocabulary.** Do not name commercial simulation products anywhere in this
  repository. Say "commercial casting simulators" or "industry-standard".
