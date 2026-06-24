# Solidus Pro — Casting Solidification Simulator

Advanced casting solidification simulation with multi-part assemblies, defect prediction, and dual-interface access (desktop + web). Inspired by SolidCast and Magmasoft.

**Supports macOS, Windows, and Linux.**

## Features

**Multi-Interface Access**
- **Desktop app** — PyQt5 + VTK 3D visualization with full interactive controls
- **Web app** — Flask backend + React/Babylon.js frontend served at `localhost:5050`
- Both share a unified `core/simulation_pipeline.py` engine

**Multi-Part Assemblies**
- CAST (grey), RISER (orange), RISER_POST, CHILL (dark blue), CORE (brown)
- Automatic part-type inference from filenames
- Parallel point-containment checking for fast voxel-to-part assignment
- Configurable transforms (position, rotation, scale) per part

**GPU Acceleration**
- Apple Silicon: Metal via MLX + Taichi (stencil kernels compiled to Metal shaders)
- NVIDIA: CUDA via CuPy
- Intel/AMD: OpenCL via PyOpenCL
- Lazy detection — backends only initialise when a simulation runs

**Multi-Core CPU Acceleration**
- Numba JIT with `parallel=True`, restricted to physical cores only
- joblib parallel task distribution
- Worker processes suppress startup banners for clean log output

**Intelligent Memory Management**
- Three modes: Auto (recommended), Force RAM, Force Disk
- Memory-mapped arrays via `np.memmap` for 50–90 % RAM savings on large models
- Session-scoped temp directories with automatic cleanup

**Seven Analysis Types**
| Analysis | Description |
|---|---|
| Shrinkage / Porosity | Feed-failure defect risk (0–1) |
| Hot Spots | Last-to-freeze regions |
| Niyama Criterion | Thermal gradient `G / √R` criterion |
| Fraction Liquid | Time-resolved liquid fraction with playback |
| Hot Tear | Brittle Temperature Range strain analysis |
| Sand Burn-In | Mold penetration risk |
| Feeding Paths | Traced riser-to-voxel connectivity |

**Visualization**
- Translucent ghost model with defect overlay
- Cross-section planes (X / Y / Z)
- Six colormaps: Red Gradient, Viridis, Plasma, Cool-Warm, Inferno, Turbo
- Measurement tools, screenshot export, MP4/GIF animation export
- Playable fraction-liquid timeline (desktop + web)

**Professional Reporting**
- Export HTML, PDF, or Markdown simulation reports with Jinja2 templates

**Plugin System**
- Register custom materials, analysis types, and visualization modes

## Supported Alloys

| Alloy | Type | Feeding Multiplier |
|---|---|---|
| WCB | Carbon Steel Casting | 6.0× |
| CF8 | Stainless Steel 304 | 5.5× |
| C958 | Aluminum Bronze | 8.0× |
| CA6NM | Cast Steel | 5.9× |
| C964 | Copper-Nickel (CuNi) | — |
| MONEL400 | Monel 400 | — |
| IN625 | Inconel 625 | — |

Each alloy includes 40+ properties: liquidus/solidus/pouring temperatures, density, conductivity, latent heat, shrinkage characteristics, SDAS parameters, Niyama threshold, hot-tear susceptibility, and more.

## Installation

### Prerequisites
- Python 3.9 or higher
- macOS, Windows, or Linux

### Setup

```bash
python3 -m venv venv
source venv/bin/activate    # Linux / macOS
# venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Optional GPU Dependencies

| Platform | Command |
|---|---|
| Apple Silicon (M1–M4) | `pip install mlx taichi` |
| NVIDIA CUDA 12.x | `pip install cupy-cuda12x` |
| NVIDIA CUDA 11.x | `pip install cupy-cuda11x` |
| Intel / AMD / legacy macOS | `pip install pyopencl` |

## Usage

### Desktop App

```bash
python run.py         # dependency check then launch
# or
python main.py        # launch directly
```

1. Load a STL file and assign part types
2. Select an alloy and configure mesh parameters
3. Run the simulation — progress shown in status area
4. Analyse results with the 3D VTK viewer — toggle layers, adjust thresholds, slice

### Web App

```bash
python run_web.py     # dependency check then serve at http://localhost:5050
```

The Flask backend serves the React/Babylon.js frontend. Upload STL files, manage multiple projects, run simulations in a FIFO job queue (max 1 concurrent, 6 queued, 1 per user), and visualise results in-browser.

## Solver Modes

| Mode | Description |
|---|---|
| `schedule_legacy` | Fast geometry-driven solidification from precomputed surface/chill/riser schedules |
| `thermal_coupled` (default) | Physics-richer mode coupling thermal diffusion with solidification progression |

The SCHEIL fraction-solid option uses a bounded linearised Scheil surrogate (full phase-diagram path not yet included).

## Configuration

Tunable via environment variables or YAML/TOML config files (see `config/settings.py`). Key controls:

- `CASTSIM_PART_ASSIGNMENT_MODE` — `auto` / `parallel` / `legacy`
- `CASTSIM_PART_ASSIGNMENT_WORKERS` — worker processes (0 = auto)
- `CASTSIM_PART_ASSIGNMENT_BATCH_SIZE` — points per containment batch

## Model Assumptions

- `voxel_states` is the authoritative fraction-solid proxy, generated from precomputed start/end solidification schedules (surface/chill early, riser regions late).
- `shrinkage_risk` and `porosity_risk` are the primary bounded 0–1 defect-risk fields. `shrinkage_probability` and `porosity_probability` remain as legacy aliases.
- `porosity_risk` is a hybrid: shrinkage base + solidification time amplification + Niyama directional-solidification modifiers. It is not a calibrated void-volume fraction.
- The merged Shrinkage/Porosity visualization uses `max(shrinkage_risk, porosity_risk)`.
- `niyama_values` uses the standard `G / √R` form with fields estimated from geometric/time proxies rather than a full thermal solve.
- `local_thickness = 2 × modulus` is a section-size proxy for relative ranking, not a literal wall-thickness measurement.

## Known Simplifications

- The default solver is geometry-driven, using a schedule-driven directional-solidification proxy rather than a solved thermal PDE.
- Feeding effectiveness prefers traced riser path lengths in riser-fed cases; open-casting cases fall back to a surface-distance proxy.
- Niyama risk interpretation includes directional-solidification modifiers near chills.
- Stress/distortion, hot tearing, and microstructure results are comparative engineering guidance, not full FEA or CALPHAD predictions.

## Formula Provenance

- **Niyama**: `Ny = G / √R` — approximate, geometry-driven field construction.
- **Darcy feeding resistance**: `ΔP = μ × L × v / K` with Kozeny-Carman permeability.
- **Chvorinov time scaling**: `t = B × M²` with material-tuned mold constants.
- **SDAS correlations**: power-law cooling-rate correlations for comparative foundry guidance.
- References: [NIST Scheil notes](https://www.ctcms.nist.gov/~kattner/solidifc/scheil.html), [MDPI Niyama discussion](https://www.mdpi.com/1996-1944/19/1/198).

## Tech Stack

- **Desktop GUI**: PyQt5
- **3D (desktop)**: VTK
- **Backend**: Flask, Flask-CORS
- **Frontend**: React 19 + TypeScript + Vite + Babylon.js
- **Numerical**: NumPy, SciPy
- **Acceleration**: Numba, joblib, MLX, Taichi, CuPy, PyOpenCL
- **Mesh**: Trimesh, numpy-stl
- **Persistence**: SQLite (web)
- **Reports**: Jinja2

## Testing

```bash
python -m unittest discover tests     # full regression suite
python -m mypy --config-file mypy.ini # type checking
python -m compileall config core physics ui web  # syntax verification
```

## License

Educational and research use.

---

*This simulator provides predictions based on geometric analysis and simplified solidification models. For production casting, validation with actual trials is recommended.*
