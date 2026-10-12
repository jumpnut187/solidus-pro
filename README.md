# Solidus Pro — Python Based Foundry Simulation Software

> **Predict defects before you pour.** The casting simulation engine that runs on your laptop — macOS, Windows, or Linux — and catches shrinkage, porosity, hot spots, and hot tears before they hit the foundry floor.

Built on established solidification physics and published casting references. Solidus Pro brings industrial-grade solidification analysis to your desktop and browser. No HPC cluster required.

---

## Why Solidus Pro?

Most casting simulators are six-figure enterprise packages locked to Windows workstations. Solidus Pro runs everywhere, handles multi-part assemblies (castings, risers, chills, cores), and gives you the same defect-prediction insights — shrinkage risk, hot spots, Niyama criterion, fraction liquid, and more — with a fraction of the overhead.

**Desktop app for power users. Web app for the team.** Same engine, same results.

---

## Features That Matter

### Dual Interface — Desktop + Web
- **Desktop**: PySide6 + VTK — full 3D interactivity, cross-sections, animations, measurement tools
- **Web**: Flask + React + Babylon.js — collaborate in-browser at `localhost:5050`
- **One engine**: `core/simulation_pipeline.py` drives both, so you get identical results everywhere

### Multi-Part Assemblies
Load your full casting assembly in one go:

| Part | Color | Purpose |
|---|---|---|
| CAST | Grey | The part you're pouring |
| RISER | Orange | Feeds liquid metal during shrinkage |
| RISER_POST | Dark Orange | Connects riser to casting |
| CHILL | Dark Blue | Accelerates local cooling |
| CORE | Brown | Internal cavity former |

Automatic part-type detection from filenames. Parallel point-containment assignment that scales to massive assemblies.

### Mesh Type — Voxel (default) or Tetrahedral (opt-in)
The Mesh dialog (desktop) and Mesh panel (web) choose the simulation mesh:
- **Voxel** (default) — adaptive voxel grid, explicit thermal solver.
- **Tetrahedral** (opt-in) — conforming tet mesh from fTetWild (`pytetwild`), implicit P1 finite-element thermal solve. It follows the CAD surface, so metal volume matches the CAD instead of overshooting at coarse sizes. The tet edge length is the mesh size times `mesh.tet_edge_length_factor` (2.0). Requires `thermal_coupled` (forced). Tet results are the accuracy reference for the voxel path.

Projects saved without a mesh type open as Voxel in both UIs. Select tet with `mesh.mesh_type: tet` in `castsim.yaml`, `CASTSIM_MESH_TYPE=tet`, or the Mesh dialog/panel.

A tet run falls back to the voxel mesh automatically (`mesh.tet_auto_fallback`, on by default) and finishes as a voxel run when the estimated node count is above `mesh.tet_auto_max_nodes` (300,000), when fTetWild fails on every attempt, or when fTetWild runs longer than `mesh.tet_mesh_timeout_s` (600 s). The reason shows in the status line and an after-run notice on the desktop, and in a banner on the web Results panel. With the fallback off, those cases stop the run with an error, and tet meshes above `mesh.tet_max_nodes` (3,000,000) are refused with a message giving a safe mesh size.

### Cut Cells — CAD-Accurate Voxel Boundary (default on)
"Cut cells (CAD-accurate boundary)" in the Mesh dialog (desktop) and Mesh panel (web), `mesh.voxel_cut_cells` in `castsim.yaml`, or `CASTSIM_VOXEL_CUT_CELLS`. Each boundary voxel carries the metal fraction and the mold-contact area of the CAD inside it (`core/cut_cells.py`), so the voxel mesh holds the CAD metal volume and cooling surface instead of a staircase. It applies to voxel runs, including a tet run that falls back to voxel. Turn it off to reproduce results from before the option existed, bit for bit.

Measured against a tet reference at 0.05 in (`tools/benchmark_cut_cells.py`), voxel size 0.1 in:

| case | total time, plain voxels | total time, cut cells | field error, cut cells | wall time on/off |
|---|---|---|---|---|
| thick_thin | +2.2 % | +0.8 % | 0.78 % | 1.3x |
| l_bracket | +14.2 % | +1.0 % | 1.55 % | 1.2x |
| box + riser + chill | +17.1 % | +0.3 % | 1.83 % | 0.8x |
| C05813 (17 parts, vs tet 0.2 in) | +9.8 % | +1.0 % | - | 0.8x |

At 0.2 in: C05813 +18.0 % plain, +2.3 % with cut cells. At 0.1 in, metal volume is within 0.6 % of CAD with cut cells (C05813: 0.04 %) and 7 to 28 % over without.

### GPU Acceleration — Pick Your Weapon
| Hardware | Backend | Status |
|---|---|---|
| Apple Silicon (M1–M4) | Metal (MLX + Taichi) | Supported |
| NVIDIA | CUDA (CuPy) | Supported |
| Intel / AMD | OpenCL (PyOpenCL) | Supported |

Lazy detection means zero overhead until you hit "Run" — no startup banners, no driver conflicts.

### Multi-Core CPU Smarts
- Numba JIT compiled hot loops — `parallel=True`, **physical cores only** (no hyperthread thrashing)
- joblib for embarrassingly parallel task distribution
- Automatic chunk sizing tuned to your CPU topology

### Memory That Won't Crash
Three modes for any hardware budget:
- **Auto** (recommended) — picks the right strategy for your RAM
- **Force RAM** — pure speed, all in-memory
- **Force Disk** — memory-mapped arrays via `np.memmap`, 50–90% RAM savings on 10M+ voxel models

---

## Seven Analysis Types, One Run

| Analysis | What It Tells You |
|---|---|
| **Shrink and Microporosity** | Shrink: where the metal will pull away into a cavity — your primary defect map. Microporosity: dispersed porosity from poor Niyama, blocked feed paths and Darcy margin |
| **Hot Spots** | Last-to-freeze zones begging for a riser |
| **Niyama Criterion** | `G / √R` — the gold-standard directional solidification check |
| **Fraction Liquid** | Time-resolved melt progression, full timeline playback |
| **Hot Tear** | Brittle Temperature Range strain — will it crack? |
| **Sand Burn-In** | Mold penetration risk in sand castings |
| **Feeding Paths** | Traced connectivity — can the riser actually feed every voxel? |

---

## Supported Alloys

Each alloy's feeding distance multiplier is calibrated **from the riser's center radius** — the distance liquid metal can travel to feed a solidifying region before the path freezes shut.

| Alloy | Type | Feeding Multiplier |
|---|---|---|
| WCB | Carbon Steel Casting | 3.0× |
| CF8 | Stainless Steel 304 | 2.5× |
| C958 | Aluminum Bronze | 4.0× |
| CA6NM | Cast Steel | 4.0× |
| C964 | Copper-Nickel (CuNi) | — |
| MONEL400 | Monel 400 | — |
| IN625 | Inconel 625 | — |

Every alloy packs 40+ properties: liquidus/solidus/pouring temps, density curves, conductivity, latent heat, volumetric shrinkage, viscosity, SDAS parameters, Niyama thresholds, hot-tear susceptibility coefficients, sand burn-in sensitivity, and gas porosity tendencies. Property values come from published casting references.

---

## System Requirements

The voxel caps below come from `core/memory_limits.py`, which sets them from total RAM. The web simulation limits come from `web/services/simulation_queue.py`, which allows one simulation per 4 physical cores, up to 4 at once.

### Local (Desktop and Web)

| | Minimum | Recommended | Max (useful) |
|---|---|---|---|
| OS | macOS 15 (Apple Silicon), Windows 10 64-bit, Linux x86_64 with glibc 2.28+ (Ubuntu 20.04+) | macOS 15+ on M2 or later, Windows 11, Ubuntu 22.04 / 24.04 | Same as recommended |
| Python | 3.10 | 3.13 (CI version) | 3.14 |
| CPU | 4 physical cores | 8 physical cores | 16+ physical cores |
| RAM | 8 GB | 32 GB | 64 GB |
| Voxel cap, soft / hard | 500K / 1M | 2M / 4M | 3M / 5M |
| GPU | None (CPU path) | Apple M2+ (Metal) or NVIDIA with CUDA 12 | Apple M3+ (soft cap +30 %) or NVIDIA with CUDA 12 |
| Disk | 10 GB free (the venv is about 4 GB) | 50 GB SSD | 100+ GB NVMe |
| Display (desktop) | OpenGL 3.2, 1920×1080 | 2560×1440 | Any |

- RAM above 64 GB does not raise the voxel cap. Cores above 16 still speed up the Numba loops, because the code sets no core cap.
- The tiers use total RAM as the OS reports it. A 64 GB machine that reports 63.x GB lands in the 32 GB tier.
- Apple Silicon raises the soft cap by 20 % (M1), 25 % (M2) or 30 % (M3, M4).
- Intel Macs are not supported. `torch>=2.6` and `pytetwild` publish no macOS x86_64 wheels.
- Tet meshes fall back to voxel above 300,000 nodes and are refused above 3,000,000 nodes.

### GCP (Web App on Compute Engine)

A GCP vCPU is one hyperthread, so physical cores = vCPU ÷ 2. Numba sizes its threads from physical cores.

| | Minimum | Recommended | Max (useful) |
|---|---|---|---|
| Machine type | e2-standard-8 (8 vCPU, 32 GB) | n2-standard-16 (16 vCPU, 64 GB) | n2-standard-32 (32 vCPU, 128 GB) |
| Physical cores | 4 | 8 | 16 |
| Concurrent simulations (auto) | 1 | 2 | 4 (code cap) |
| Voxel cap, soft / hard | 1M / 2M | 2M / 4M | 3M / 5M |
| GPU (optional, CUDA via CuPy) | None | g2-standard-16 (1× NVIDIA L4, 16 vCPU, 64 GB) | g2-standard-32 (1× NVIDIA L4, 32 vCPU, 128 GB) |
| Boot disk | 50 GB pd-balanced | 100 GB pd-ssd | 200 GB pd-ssd |
| OS image | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS | Ubuntu 24.04 LTS |
| Python | 3.10 | 3.13 | 3.14 |

- GCP machines report slightly less RAM than their nominal size, so 32 GB lands in the 16 GB tier and 64 GB lands in the 32 GB tier. The voxel caps above reflect that.
- Each extra simulation needs 2 GB free (`CASTSIM_WEB_MIN_SIM_MEMORY_MB`) beyond a reserve of the larger of 2 GB or 10 % of RAM (`CASTSIM_WEB_RESERVE_MEMORY_MB`).
- Above 32 vCPU the queue still stops at 4 simulations. Set `CASTSIM_MAX_CONCURRENT_SIMULATIONS` to use more.
- gunicorn runs 1 worker with 8 threads (`run_web.py`). Simulations run in separate processes.
- Set `CASTSIM_SECRET_KEY`, and put the app behind Identity-Aware Proxy.

---

## Installation — 3 Commands

```bash
python3 -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

**Prerequisites**: Python 3.10–3.14 (3.13 recommended). See [System Requirements](#system-requirements).

### Optional: GPU Dependencies

| Platform | Command |
|---|---|
| Apple Silicon (M1–M4) | `pip install mlx taichi` |
| NVIDIA CUDA 12.x | `pip install cupy-cuda12x` |
| NVIDIA CUDA 11.x | `pip install cupy-cuda11x` |
| Intel / AMD / legacy macOS | `pip install pyopencl` |

---

## Usage

### Desktop App
```bash
python run.py       # checks deps, then launches
# or
python main.py      # launches directly
```
Load CAD (STL or STEP) → assign parts → pick alloy → run → inspect with full 3D controls.

### Web App
```bash
python run_web.py   # serves at http://localhost:5050
```
Multi-project management, FIFO job queue (1 concurrent, 6 queued), in-browser Babylon.js 3D viewer. Perfect for sharing results with the team.

---

## Visualization That Sells

- Translucent ghost model with defect heatmap overlay
- Cross-section slicing on X, Y, Z — see *inside* the casting
- Six colormaps (Red Gradient, Viridis, Plasma, Cool-Warm, Inferno, Turbo)
- Animated fraction-liquid timeline — watch the melt freeze
- Measurement tools, screenshot capture, MP4/GIF export
- Per-part visibility toggles and opacity controls

---

## Solver Modes

| Mode | Philosophy |
|---|---|
| `schedule_legacy` | Blazing fast. Precomputed surface/chill/riser schedule. Geometry-driven. |
| `thermal_coupled` (default) | Physics-rich. Solves the thermal diffusion field coupled to fraction-solid evolution. |

The default `thermal_coupled` solver uses temperature-dependent k/cp/rho property curves, an isotropic weighted Laplacian stencil (exact on quadratic fields), and a conservative two-point-flux update with implicit (enthalpy) latent-heat release on the same time step. Mold interfaces use a Robin condition combining contact conductance with semi-infinite sand-mold penetration decay; chills act as finite graphite heat sinks (diffusivity + heat-capacity override, higher late-time HTC floor, since a rigid chill does not air-gap like the casting) rather than fixed-temperature pins. The `SCHEIL` fraction-solid option implements the true Scheil microsegregation equation using per-alloy solvent melting point and partition coefficient, with lever-rule solidus completion and an optional eutectic cap.

---

## Model Assumptions

- `voxel_states` — authoritative fraction-solid proxy from precomputed start/end schedules
- `shrinkage_risk` / `porosity_risk` — primary 0–1 defect fields (legacy aliases preserved)
- `porosity_risk` — hybrid signal: shrinkage base + solidification time amplification + Niyama directional modifiers
- `niyama_values` — standard `G / √R` form, estimated from geometric/time proxies
- `local_thickness = 2 × modulus` — section-size proxy for ranking, not literal wall thickness

## Known Simplifications

- No mold-filling simulation — castings start at a uniform pouring temperature
- No explicit mold domain — the mold is a constant-temperature sink with an interface HTC
- The diffusion stencil is approximate on graded octree meshes (exact on uniform grids)
- Shrinkage / porosity / hot-tear outputs are calibrated engineering risk scores (0–1), not first-principles porosity fractions
- Feeding paths use traced riser connectivity where available; fall back to surface-distance otherwise
- Niyama interpretation includes directional-solidification modifiers near chills
- Stress, hot tearing, and microstructure are comparative guidance — not full FEA or CALPHAD

---

## Formula Provenance

| Model | Form | Notes |
|---|---|---|
| **Niyama** | `Ny = G / √R` | Geometry-driven field estimates |
| **Darcy** | `ΔP = μ × L × v / K` | Kozeny-Carman permeability |
| **Chvorinov** | `t = B × M²` | Material-tuned mold constants |
| **SDAS** | Power-law cooling rate | Comparative foundry guidance |

References: [NIST Scheil notes](https://www.ctcms.nist.gov/~kattner/solidifc/scheil.html) · [MDPI Niyama discussion](https://www.mdpi.com/1996-1944/19/1/198)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Desktop GUI | PySide6 (Qt 6) |
| Desktop 3D | VTK |
| Web Backend | Flask + Flask-CORS |
| Web Frontend | React 19 + TypeScript + Vite + Babylon.js |
| Numerics | NumPy, SciPy |
| CPU Accel | Numba (JIT + parallel), joblib |
| GPU Accel | MLX, Taichi (Metal) · CuPy (CUDA) · PyOpenCL (OpenCL) |
| Mesh I/O | Trimesh, numpy-stl, cascadio (optional, for STEP) |
| Tet meshing | pytetwild (fTetWild, MPL-2.0) + pyvista |
| Persistence | SQLite (web projects) |
| Reports | Jinja2 (HTML / PDF / Markdown) |
| Plugin System | Custom materials, analysis types, viz modes |

---

## Testing

```bash
python -m unittest discover tests         # full regression suite
python -m mypy --config-file mypy.ini     # type checking
python -m compileall config core physics ui web   # syntax pass
```

---

## License

Property of DW Clark & Brendon Jarrett


---
