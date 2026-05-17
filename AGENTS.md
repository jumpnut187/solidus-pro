# Repository Guidelines

## Agent Tooling
Use context-mode MCP tools by default to keep context usage low and execution efficient.

- Prefer `mcp__context_mode__ctx_batch_execute` for grouped repo inspection and query workflows.
- Prefer `mcp__context_mode__ctx_execute` for commands with non-trivial output.
- Prefer `mcp__context_mode__ctx_execute_file` for large file analysis instead of raw `cat`/`sed`.

## Project Structure & Module Organization
Desktop entrypoints are `main.py` and `run.py`; the web entrypoint is `run_web.py`. Core simulation orchestration lives in `core/`, with the current user-facing pipeline centered on `core/simulation_pipeline.py`, `core/solidification_loop.py`, and `core/solidification.py`. Physics helper formulas and reference models live in `physics/`. Runtime settings and exceptions live in `config/`. The PyQt desktop UI is under `ui/`, while the Flask/React/Babylon web app is split across `web/app.py`, `web/app_factory.py`, `web/routes.py`, `web/payloads.py`, `web/repository.py`, `web/services/`, and `web/static/`. The React frontend sources live in `web_frontend/` with its own build system (npm). Treat `data/`, `logs/`, `models/`, `web/static/uploads/`, `web/simulations.db`, and `web_frontend/dist/` as generated or local-state paths unless the task explicitly targets them.

## Build, Test, and Development Commands
Create an environment with `python3 -m venv venv`, activate it with `source venv/bin/activate`, and install dependencies with `pip install -r requirements.txt`.

- `python3 main.py` launches the desktop simulator directly.
- `python3 run.py` checks core dependencies, then launches the desktop app.
- `python3 run_web.py` checks web dependencies, then serves Flask on `http://localhost:5050`.
- `python3 -m unittest discover tests` runs the unittest-based regression suite.
- `python3 tests/verify_chill_decay.py` runs the chill-specific verification script.
- `python3 -m mypy --config-file mypy.ini` runs the repo's current typed-module checks.
- `python3 -m compileall config core physics ui web` performs the required syntax pass.

There is no separate build system; verification is command-driven.

## Coding Style & Naming Conventions
Use 4-space indentation, `snake_case` for modules, functions, and variables, and `PascalCase` for classes. Keep new Python code type-annotated, but match the repo's current incremental-typing baseline in `mypy.ini` instead of assuming repo-wide strict mode. Preserve existing module boundaries: simulation pipeline logic stays in `core/`, lower-level formulas in `physics/`, desktop UI in `ui/`, and Flask request/data layers in `web/`. Follow existing structured logging and explicit error-handling patterns in backend and pipeline code.

## Testing Guidelines
Add focused regression coverage near the behavior you change. Follow the current `unittest`-based pattern in `tests/`, using `test_*.py` for discoverable suites and `verify_*.py` for standalone validation scripts. Prefer reproducing bugs with a test before fixing them. Before finishing a change, run the most relevant regression command plus `python3 -m mypy --config-file mypy.ini` and `python3 -m compileall config core physics ui web`.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects such as `Fix riser path normalization` or `Add fraction liquid regression`. Keep commits focused. PRs should describe the behavior change, list the validation commands you ran, link related issues, and include screenshots when desktop UI or web output changes.

## Security & Configuration Tips
Do not hardcode secrets or machine-specific paths. Keep credentials in environment variables and out of client-side code, templates, and logs. Avoid committing generated outputs or local databases, and never delete or overwrite `.db` files without explicit approval.

## Performance Controls
Part-type assignment now supports explicit tuning controls for large CAST/RISER/CHILL assemblies:

- `performance.part_assignment_mode`: `auto`, `parallel`, or `legacy`
- `performance.part_assignment_workers`: worker processes for containment checks (`0` means auto)
- `performance.part_assignment_batch_size`: candidate points per containment batch
- Environment overrides: `CASTSIM_PART_ASSIGNMENT_MODE`, `CASTSIM_PART_ASSIGNMENT_WORKERS`, `CASTSIM_PART_ASSIGNMENT_BATCH_SIZE`

By design, GPU/backend detection and Numba thread banners are emitted by the main process only. Worker processes suppress those startup prints to keep logs readable during parallel runs.
