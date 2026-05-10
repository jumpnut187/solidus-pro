# Casting Solidification Simulator - Web Version

A browser-based interface for the casting solidification simulation engine with SQLite database for saving and loading simulations.

## Quick Start

```bash
# Install dependencies
pip install flask flask-cors

# Run web server
python run_web.py
```

Then open **http://localhost:5050** in your browser.

## Features

### 3D Visualization
- **Three.js-based** rendering with orbit controls
- **STL file upload** via drag & drop or file browser
- **Multi-part support** with CAST/RISER/CHILL types
- **Color-coded** parts by type (grey=cast, orange=riser, blue=chill)
- **Wireframe toggle**, axes, and grid display
- **Cross-section** slicing on X/Y/Z planes
- **Measurement tool** for distance between points
- **Screenshot** export

### Simulation Controls
- **Material selection**: WCB (Carbon Steel), CF8 (Stainless), C958 (Bronze), CA6NM (Cast Steel)
- **Mesh settings**: Voxel size and refinement level
- **Live preview** of estimated voxel count and memory usage
- **Analysis options**: Shrinkage, Hot Spots, Niyama, Hot Tear, Sand Burn-In

### Results Visualization
- **Shrinkage/Porosity** display with threshold control
- **Hot Spot** detection
- **Niyama Criterion** visualization
- **Color maps**: Red gradient, Viridis, Plasma, Cool-Warm
- **Adjustable opacity** for model transparency

### Save/Load System
- **SQLite database** for persistent storage
- **Save** complete simulation setups with all parts
- **Load** previous simulations
- **Results stored** for later viewing

## Browser Requirements
- Chrome, Firefox, Safari, or Edge (modern versions)
- WebGL support required for 3D visualization

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/materials` | GET | List available alloys |
| `/api/upload` | POST | Upload STL file |
| `/api/simulations` | GET/POST | List/Create simulations |
| `/api/simulations/<id>` | GET/PUT/DELETE | CRUD operations |
| `/api/simulations/<id>/parts` | POST | Add part to simulation |
| `/api/simulations/<id>/run` | POST | Start simulation |
| `/api/simulations/<id>/status` | GET | Get simulation progress |
| `/api/simulations/<id>/results` | GET | Get visualization data |
| `/api/measure/distance` | POST | Calculate distance between points |

## Architecture

```
web/
├── app.py              # Flask application & API routes
├── simulations.db      # SQLite database (auto-created)
├── static/
│   ├── css/
│   │   └── styles.css  # Modern dark theme UI
│   ├── js/
│   │   └── app.js      # Three.js visualization & state management
│   ├── uploads/        # Uploaded STL files
│   └── temp/           # Temporary files
└── templates/
    └── index.html      # Main application page
```

## Keyboard Shortcuts

- **F** - Front view
- **T** - Top view
- **S** - Side view
- **I** - Isometric view
- **Mouse drag** - Rotate view
- **Mouse wheel** - Zoom
- **Right click + drag** - Pan

## Database Schema

### simulations
- id, name, description, created_at, updated_at
- alloy, voxel_size, refinement_level, analysis_options
- status, total_volume, defect_volume, voxel_count

### simulation_parts
- id, simulation_id, name, part_type, filepath
- position_x/y/z, rotation_x/y/z, scale

### simulation_results
- id, simulation_id, result_type, data_json, voxel_data_path

## Differences from Desktop Version

| Feature | Desktop (PyQt5) | Web (Flask/Three.js) |
|---------|----------------|---------------------|
| 3D Engine | VTK | Three.js |
| Storage | Files | SQLite |
| Install | Python packages | pip + browser |
| Multi-user | Single | Multi-user capable |
| GPU Acceleration | OpenCL/Metal | WebGL |
