# Casting Solidification Simulator - Web Version

A browser-based interface for the casting solidification simulation engine with SQLite database for saving and loading simulations, built with React and Babylon.js.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build frontend
cd web_frontend && npm install && npm run build && cd ..

# Run web server
python run_web.py
```

Then open **http://localhost:5050** in your browser.

## Features

### 3D Visualization
- **Babylon.js-based** rendering with orbit controls
- **STL file upload** via drag & drop or file browser
- **Multi-part support** with CAST/RISER/CHILL/RISER_POST types
- **Color-coded** parts by type
- **Wireframe toggle**, axes, and grid display
- **Cross-section** slicing on X/Y/Z planes
- **Measurement tool** for distance between points
- **Screenshot** export

### Simulation Controls
- **Material selection**: WCB (Carbon Steel), CF8 (Stainless), C958 (Bronze), CA6NM (Cast Steel)
- **Mesh settings**: Voxel size and refinement level
- **Live preview** of estimated voxel count and memory usage
- **Analysis options**: Shrinkage, Hot Spots, Niyama, Hot Tear, Sand Burn-In, Fraction Liquid

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
| `/api/health` | GET | Health check |
| `/api/materials` | GET | List available alloys |
| `/api/projects` | GET/POST | List/Create projects |
| `/api/projects/<id>` | GET/PUT/DELETE | CRUD operations |
| `/api/projects/<id>/parts` | POST | Add part to project |
| `/api/projects/<id>/parts/<part_id>` | PUT/DELETE | Update/delete part |
| `/api/projects/<id>/mesh-preview` | POST | Preview mesh estimate |
| `/api/projects/<id>/run` | POST | Start simulation |
| `/api/projects/<id>/cancel` | POST | Cancel simulation |
| `/api/jobs/<id>/status` | GET | Get job progress |
| `/api/projects/<id>/results` | GET | Get visualization data |
| `/api/projects/<id>/result-fields` | GET | Get detailed field data |
| `/api/projects/<id>/surface` | GET | Get surface mesh data |
| `/api/projects/<id>/volume` | GET | Get volume data |
| `/api/uploads` | POST | Upload STL file |
| `/api/uploads/<filename>` | GET | Serve uploaded file |

## Architecture

```
web/
├── app.py              # Flask application entrypoint
├── app_factory.py     # Flask application factory
├── routes.py           # API route handlers
├── repository.py      # SQLite persistence
├── payloads.py        # Babylon-oriented payload helpers
├── services/
│   └── simulation_queue.py  # Queue management
└── static/
    ├── uploads/       # Uploaded STL files
    └── temp/          # Temporary files

web_frontend/
├── src/
│   ├── App.tsx        # Main React component
│   ├── lib/
│   │   └── api.ts     # API client
│   └── components/    # React components
├── dist/              # Built frontend (served by Flask)
└── package.json
```

## Running the Development Server

For development with hot reload:

```bash
cd web_frontend
npm install
npm run dev    # Starts Vite dev server on port 5173 with proxy to Flask
```

In another terminal:

```bash
python run_web.py   # Flask API on port 5050
```

Then open **http://localhost:5173** for the frontend.

## Database Schema

### projects
- id, name, description, created_at, updated_at
- alloy, voxel_size, refinement_level, memory_mode
- analysis_options, performance_options, solidification_params
- ui_state, status, artifact_path, summary_json

### parts
- id, project_id, name, part_type, filepath
- position_x/y/z, rotation_x/y/z, scale
- visible, is_open_top, created_at