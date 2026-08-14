# Solidus Pro — Python Based Foundry Simulation Software

> **Predict defects before you pour.** The casting simulation engine that runs on your laptop — macOS, Windows, or Linux — and catches shrinkage, porosity, hot spots, and hot tears before they hit the foundry floor.

Built on **real-world casting results** and validated against production foundry data. Solidus Pro brings industrial-grade solidification analysis to your desktop and browser. No HPC cluster required.

---

## Why Solidus Pro?

Most casting simulators are six-figure enterprise packages locked to Windows workstations. Solidus Pro runs everywhere, handles multi-part assemblies (castings, risers, chills, cores), and gives you the same defect-prediction insights — shrinkage risk, hot spots, Niyama criterion, fraction liquid, and more — with a fraction of the overhead.

**Desktop app for power users. Web app for the team.** Same engine, same results.

---

## Features That Matter

### Dual Interface — Desktop + Web
- **Desktop**: PyQt5 + VTK — full 3D interactivity, cross-sections, animations, measurement tools
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
| **Shrinkage / Porosity** | Where the metal will pull away — your primary defect map |
| **Hot Spots** | Last-to-freeze zones begging for a riser |
| **Niyama Criterion** | `G / √R` — the gold-standard directional solidification check |
| **Fraction Liquid** | Time-resolved melt progression, full timeline playback |
| **Hot Tear** | Brittle Temperature Range strain — will it crack? |
| **Sand Burn-In** | Mold penetration risk in sand castings |
| **Feeding Paths** | Traced connectivity — can the riser actually feed every voxel? |

---

## Supported Alloys — Tuned from Real Foundry Data

Each alloy's feeding distance multiplier is calibrated **from the riser's center radius** — the distance liquid metal can travel to feed a solidifying region before the path freezes shut.

| Alloy | Type | Feeding Multiplier |
|---|---|---|
| WCB | Carbon Steel Casting | 6.0× |
| CF8 | Stainless Steel 304 | 5.5× |
| C958 | Aluminum Bronze | 8.0× |
| CA6NM | Cast Steel | 5.9× |
| C964 | Copper-Nickel (CuNi) | — |
| MONEL400 | Monel 400 | — |
| IN625 | Inconel 625 | — |

Every alloy packs 40+ properties: liquidus/solidus/pouring temps, density curves, conductivity, latent heat, volumetric shrinkage, viscosity, SDAS parameters, Niyama thresholds, hot-tear susceptibility coefficients, sand burn-in sensitivity, and gas porosity tendencies. All validated against published casting references and production data.

---

## Developed on Real Casting Results

Solidus Pro wasn't built in a vacuum. The defect prediction engine has been tuned and validated against **actual production casting outcomes** — including radiographed sections, cut-up analyses, and foundry floor feedback. The feeding-distance models, chill effectiveness curves, and Niyama threshold calibrations all trace back to real pour results, not textbook approximations alone.

**What this means for you**: when Solidus Pro flags a hot spot, it's flagging the same kind of region that showed up in a physical casting.

---

## Installation — 3 Commands

```bash
python3 -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

**Prerequisites**: Python 3.9+, any OS.

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
| Desktop GUI | PyQt5 |
| Desktop 3D | VTK |
| Web Backend | Flask + Flask-CORS |
| Web Frontend | React 19 + TypeScript + Vite + Babylon.js |
| Numerics | NumPy, SciPy |
| CPU Accel | Numba (JIT + parallel), joblib |
| GPU Accel | MLX, Taichi (Metal) · CuPy (CUDA) · PyOpenCL (OpenCL) |
| Mesh I/O | Trimesh, numpy-stl, cascadio (optional, for STEP) |
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

Educational and research use.

---

*Solidus Pro provides engineering predictions based on geometric analysis and simplified solidification models. For production-critical castings, always validate with physical trials.*
