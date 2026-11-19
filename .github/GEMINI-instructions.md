# Casting Solidification Simulator - AI Agent Instructions

## Project Overview
This is a professional-grade casting solidification simulator that predicts shrinkage and porosity defects in metal castings. It's **cross-platform** and supports Intel, AMD, and Apple Silicon processors on Windows, Linux, and macOS. The system includes ML-powered adaptive mesh recommendations that learn from simulation history. The simulator uses multi-part management (CAST/RISER/CHILL parts), memory-safe disk-backed storage for large models, and physics-based edge-inward solidification with vertical progression effects.

## Core Architecture

### Module Hierarchy
```
core/               # Physics simulation engine
├── materials.py    # Alloy properties (WCB, CF8, C958/Bronze)
├── stl_loader.py   # STL import with dimension handling (all in inches, no conversion)
├── voxel_mesh.py   # Adaptive voxelization with part type tracking (CAST/RISER/CHILL)
├── solidification.py  # Physics simulation with vertical & chill/riser effects
├── acceleration.py # Hardware acceleration layer (Numba JIT, parallel compute)
├── memory_manager.py # Disk-backed memory-mapped arrays for large models
├── memory_limits.py # Memory validation, auto voxel size adjustment
├── part_manager.py # Multi-part STL loading with positioning/transforms
└── mesh_ml.py      # ML-powered mesh recommendations from simulation history

ui/                 # PySide6 GUI
├── main_window.py  # Main application with sidebar controls + multi-part support
└── visualizer.py   # VTK-based 3D visualization with SolidWorks-style controls

config/
└── hardware_config.py  # Cross-platform CPU detection (Intel/AMD/Apple)

tools/
└── train_mesh_model.py  # Train mesh recommender from simulation history

data/
└── mesh_runs.csv   # Logged simulation performance data (auto-populated)

models/             # Trained ML models (generated after training)
└── mesh_recommender.joblib, mesh_scaler.joblib
```

### Data Flow
1. **STL Import**: `stl_loader.py` loads STL files (in inches), validates mesh (no conversion - stays in inches internally)
2. **Multi-Part Setup**: `part_manager.py` allows multiple parts with types (CAST/RISER/CHILL) and transforms (position, rotation, scale)
3. **ML Mesh Recommendation**: `mesh_ml.py` suggests optimal voxel size + refinement level based on part geometry and hardware
4. **Memory Validation**: `memory_limits.py` checks if voxel count will fit in system memory; auto-adjusts voxel size if needed
5. **Voxelization**: `voxel_mesh.py` creates adaptive grid with multi-level refinement and assigns part types to voxels
6. **Physics Simulation**: `solidification.py` simulates edge-inward solidification with chill/riser effects and vertical progression
7. **Data Logging**: Performance metrics saved to `data/mesh_runs.csv` for ML training
8. **Visualization**: `visualizer.py` renders translucent "ghost" model + red shrinkage regions

## Hardware Acceleration Patterns

### Cross-Platform CPU Optimization
- **CPU Detection**: Auto-detects Intel, AMD, and Apple Silicon processors via `config/hardware_config.py`
- **Physical Cores Only**: Uses physical cores only (no hyperthreading) to minimize memory usage
- **Thread Allocation**: Reserves 1 core for system tasks (if 3+ cores available)
- **Numba JIT**: All hot loops use `@jit(nopython=True, parallel=True, fastmath=True, cache=True)`
- **GPU**: PyOpenCL with Metal backend (currently disabled due to Qt stability issues)
- **Memory Manager**: Automatic disk-backed storage for large arrays to prevent RAM exhaustion

### Parallel Processing
```python
# Pattern: Use Numba for tight loops
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_modulus_parallel(volumes, areas):
    result = np.empty_like(volumes)
    for i in prange(len(volumes)):
        result[i] = volumes[i] / areas[i] if areas[i] > 0 else 0
    return result
```

**Key**: Always set Numba threads via `set_num_threads(hardware.get_optimal_thread_count('compute'))` in `acceleration.py`

### Memory Management Strategy
- **BYTES_PER_VOXEL**: 280 bytes (includes all voxel arrays + overhead)
- **Max Voxels** (scales with RAM):
  - 8GB RAM: 500K voxels
  - 16GB RAM: 1,000K voxels
  - 32GB RAM: 2,000K voxels
  - 64GB+ RAM: 3,000K voxels
  - Apple Silicon gets 20-30% bonus due to unified memory
- **Auto-Adjustment**: If voxel count would exceed limits, `memory_limits.py` automatically increases voxel size
- **Disk-Backed Storage**: `memory_manager.py` uses memory-mapped files for large arrays, transparent to code
- **Three Modes**: "auto" (recommended), "ram" (force RAM), "disk" (force disk)
- **Cleanup**: Session temp directories auto-deleted on exit

### Thread Allocation Examples
- Intel Core i7 (8 cores, 16 threads) → Uses **7 physical cores**
- AMD Ryzen 7 (8 cores, 16 threads) → Uses **7 physical cores**
- Apple M3 (4P + 4E cores) → Uses **7 cores** (3P + 4E)
- Intel Xeon (12+ cores) → Uses **11+ physical cores**

## Critical Conventions

### Unit Handling
- **INPUT**: STL files are assumed to be in **inches**
- **INTERNAL**: All calculations use **inches** (NO conversion to mm)
- **Note**: Previous documentation mentioned mm conversion - this is NO LONGER DONE
- **Why**: Keeps all physics calculations in consistent units throughout the pipeline

### Part Type System
The simulator supports three part types via `core/part_manager.py`:
- **CAST (type 0)**: The actual casting - subject to shrinkage/porosity analysis, shown in default grey
- **RISER (type 1)**: Feeder/riser with exothermic effect - slows cooling (0.4x factor), shown in orange
- **CHILL (type 2)**: External chill - accelerates cooling (up to 3x faster), shown in dark blue
- **Colors**: `PART_TYPE_COLORS` dict defines RGB tuples for visualization

### Material Properties
Materials defined as dataclasses in `materials.py` with physics-based calculations:
- `volumetric_shrinkage`: Computed from density ratio (solid/liquid), not user-specified
- `feeding_distance`: Calculated as `thickness * feeding_distance_multiplier`
- `thermal_diffusivity`: Computed as `k / (ρ * cp)` - affects solidification rate
- `density_change_ratio`: `density_solid / density_liquid`
- `freezing_range`: `liquidus_temp - solidus_temp` - affects mushy zone behavior
- Access via `get_alloy(name)` - never hardcode material values

**Material-Specific Solidification**:
The simulation uses material thermal properties to calculate solidification rate:
- WCB (baseline): thermal_diffusivity factor = 1.0x
- CF8 (stainless): ~0.24x slower (low thermal conductivity)
- C958 (bronze): ~0.85x (good conductivity but wider freezing range)

### Simulation State
Voxel mesh (`AdaptiveVoxelMesh`) maintains:
- `voxel_states`: 0.0 (liquid) → 1.0 (solid) progression
- `surface_voxels`: Boolean array marking boundary voxels
- `voxel_neighbors`: Connectivity graph for heat flow
- `voxel_part_types`: 0=CAST, 1=RISER, 2=CHILL for each voxel
- `solidification_times`: When each voxel solidified

**Important**: Surface voxels initialize solidification front; simulation propagates inward with part-type modifiers

## Physics Engine Equations

### Geometric Modulus
```python
# Standard formula: Modulus = Volume / Surface Area
modulus[i] = volume[i] / surface_area[i]  # inches

# Alternate (distance-based): Distance to nearest surface voxel
modulus[i] = distance_to_surface[i]  # Used in current implementation
```

### Volumetric Shrinkage
```python
# Computed from density change during solidification
volumetric_shrinkage = (1 - density_liquid / density_solid) * 100  # percentage

# Example: WCB steel
# ρ_liquid = 7200 kg/m³, ρ_solid = 7850 kg/m³
# shrinkage = (1 - 7200/7850) * 100 = 8.3%
```

### Feeding Distance
```python
# Maximum distance liquid metal can feed shrinkage
feeding_distance = thickness * feeding_distance_multiplier
# feeding_distance_multiplier: WCB=3.0, CF8=2.5, C958=4.0
```

### Solidification Rate with Part Effects
```python
# Edge-inward progression with vertical bias
solidification_rate = 1.0 / (1.0 + modulus * 0.1)
vertical_factor = 1.0 + (1.0 - normalized_height) * 0.3  # Bottom solidifies 30% faster

# Part type modifiers
if part_type == CHILL:
    rate *= chill_boost  # 1.0 to 3.0x faster (linear falloff with distance)
elif part_type == RISER:
    rate *= 0.4  # Exothermic effect: 2.5x slower solidification

# State update per time step
new_state[i] = min(state[i] + solidification_rate * vertical_factor * part_modifier * dt, 1.0)
```

### Shrinkage Probability (Multi-Factor Model for Thick Sections ≥1.5")
```python
interior_factor = min(modulus / 1.0, 1.0)  # Distance from surface (0-1)
thickness_factor = min((thickness - 1.5) / 1.5, 1.0)  # Excess thickness (0-1)
time_factor = min(solidification_time / 800.0, 1.0)  # Cooling duration (0-1)

base_probability = (
    0.4 * interior_factor +    # 40% weight: distance from surface
    0.4 * thickness_factor +   # 40% weight: section thickness
    0.2 * time_factor          # 20% weight: solidification time
)

# Scale by material shrinkage characteristic
shrinkage_probability[i] = base_probability * (volumetric_shrinkage / 6.0)
```

### Porosity Calculation
```python
# Porosity amplifies shrinkage based on solidification duration
time_factor = min(solidification_time[i] / critical_time, 1.0)
porosity[i] = shrinkage[i] * (1.0 + time_factor)
```

### Chill Influence Model
```python
# Chill accelerates cooling of nearby voxels by heat extraction
chill_influence_radius = 2.0  # inches - max distance of chill effect on solidification rate
max_chill_boost = 3.0  # 3x faster solidification at chill contact

min_dist_to_chill = distance_to_nearest_chill_voxel
if min_dist_to_chill < chill_influence_radius:
    influence = 1.0 - (min_dist_to_chill / chill_influence_radius)
    chill_boost[i] = 1.0 + (max_chill_boost - 1.0) * influence
```

### Chill Shrinkage Reduction Model
```python
# Chills REDUCE shrinkage near them because of directional solidification benefit
# Metal near chills solidifies FIRST while liquid feed from interior is still available
# This is the key foundry principle: chills promote feeding, reducing porosity

chill_shrinkage_radius = 3.0  # inches - larger than solidification effect zone
max_reduction = 0.15  # At contact: only 15% of calculated shrinkage (85% reduction)

if distance_to_chill < chill_shrinkage_radius:
    normalized_dist = distance_to_chill / chill_shrinkage_radius
    # Quadratic falloff for smooth gradient
    shrinkage_reduction[i] = 0.15 + 0.85 * (normalized_dist ** 1.5)
    # Final shrinkage = calculated_shrinkage * shrinkage_reduction
```

### Riser Exothermic Effect
```python
# Risers with exothermic sleeves stay liquid longer
exothermic_factor = 0.4  # Voxels in risers solidify at 40% normal rate (2.5x longer)
# Applied uniformly to all riser voxels, no distance-based falloff
```

### Local Thickness Estimation
```python
# Approximate thickness from modulus (for simple geometries)
local_thickness = modulus * 2.0  # inches
```

### Interdendritic Permeability (Kozeny-Carman Equation)
```python
# Permeability determines how easily liquid can flow through the mushy zone
# Based on dendrite arm spacing (SDAS) and fraction solid
# K = (λ² * (1 - fs)³) / (180 * fs²)

lambda_m = dendrite_arm_spacing * 1e-6  # SDAS in meters
liquid_fraction = 1.0 - fraction_solid

# Above critical_fraction_solid: permeability drops exponentially (coherency point)
if fraction_solid >= critical_fraction_solid:
    overshoot = (fs - critical_fraction_solid) / (1.0 - critical_fraction_solid)
    decay_factor = exp(-10 * overshoot)
    K = K_base * decay_factor
```

### Ferro-static Pressure (Driving Force for Feeding)
```python
# Pressure from metal head above voxel: P = ρ * g * h
# This is the PUSH force that drives feeding from risers into casting

height_diff_meters = (max_riser_height - voxel_height) * 0.0254
ferrostatic_pressure = density_liquid * 9.81 * height_diff_meters  # Pa
```

### Interdendritic Pressure Drop (Darcy's Law - The Resistance)
```python
# Pressure drop through mushy zone: ΔP = (μ * L * v) / K
# This is the FRICTION that resists feeding flow

pressure_drop = (viscosity * path_length * velocity) / permeability  # Pa

# Porosity forms when P_ferro < ΔP (driving force < resistance)
pressure_ratio = ferrostatic_pressure / pressure_drop
feeding_effective = pressure_ratio >= 1.0  # Can the feed path stay open?
```

### Critical Fraction Solid (Coherency Point)
```python
# Point where dendrites touch and block interdendritic flow
# Material-specific: different alloys have different feeding windows
# WCB (carbon steel): 0.52 (narrow window)
# CF8 (stainless):    0.48 (narrower due to wide freezing range)
# C958 (bronze):      0.62 (wider window, better fluidity)

if fraction_solid > critical_fraction_solid:
    # Dendrites have touched - liquid cannot flow through
    # Only shrinkage can occur, not feeding
    permeability = near_zero
```

**Critical Thresholds**:
- Minimum thickness for shrinkage: 1.5 inches
- Minimum modulus (interior depth): 0.3 inches
- Default shrinkage threshold: 5% probability
- Critical solidification time: 100-800 seconds (material-dependent)
- Chill solidification influence radius: 2.0 inches
- Chill shrinkage reduction radius: 3.0 inches (larger zone for feeding benefit)
- Maximum chill shrinkage reduction: 85% (at contact)
- Critical fraction solid: WCB=0.52, CF8=0.48, C958=0.62
- Dendrite arm spacing: WCB=80μm, CF8=60μm, C958=120μm

## Development Workflows

### Running Locally
```bash
python3 main.py  # Launch GUI directly
python3 run.py   # Check dependencies first
```

### Testing Hardware Optimization
```bash
python3 test_hardware.py    # Verify CPU detection (Intel/AMD/Apple)
python3 test_materials.py   # Show material database
python3 test_cross_platform.py  # Cross-platform compatibility test
```

### Generating Test Cases
```bash
python3 generate_samples.py  # Creates sample STL files in sample_stl_files/
```

### Installing Dependencies
```bash
pip3 install -r requirements.txt  # Use pip3 on macOS/Linux
./install.sh                      # Automated script
```

### Training ML Mesh Recommender
```bash
# After accumulating 20+ simulation runs
pip3 install scikit-learn  # Required for training
python3 tools/train_mesh_model.py
```

### Multi-Part Workflow
1. Load first STL as CAST (main part)
2. Add RISER parts via "Add Part" button
3. Position/rotate/scale risers using transform controls
4. Add CHILL parts (external cooling blocks)
5. Run simulation - part types automatically included in physics
6. View results with part-type colors in 3D visualization

## Common Patterns

### Adding New Materials
Edit `core/materials.py`:
```python
ALLOY_DATABASE['NEW_ALLOY'] = AlloyProperties(
    name='Display Name',
    liquidus_temp=1500,  # °C
    solidus_temp=1450,
    pouring_temp=1600,
    density_liquid=7000,
    density_solid=7500,
    thermal_conductivity=40,
    specific_heat=600,
    latent_heat=250000,
    linear_shrinkage=2.0,
    feeding_distance_multiplier=3.0
)
```

### Modifying Simulation Physics
1. Update `solidification.py` methods
2. Ensure compatibility with `acceleration.py` parallel functions
3. Test with `test_simulation.py` before GUI integration

### UI Changes
- Main window layout: `ui/main_window.py` (sidebar pattern with QSplitter)
- 3D rendering: `ui/visualizer.py` (VTK pipeline: mesh → mapper → actor)
- Background tasks: Use `QThread` workers (see `SimulationWorker` class)

### Memory Management Patterns
- Use `create_managed_array()` for large simulation arrays
- Arrays automatically choose RAM vs disk based on system memory
- Transparent to code - works like normal NumPy arrays
- Set memory_mode in voxel mesh or simulation engine

### Part Manager Usage
```python
from core.part_manager import PartManager, PartType

# Create and add parts
part_manager = PartManager()
part_manager.add_part_from_file("casting.stl", PartType.CAST)
part_manager.add_part_from_file("riser.stl", PartType.RISER, 
                               position=(0, 0, 5), scale=1.5)
part_manager.add_part_from_file("chill.stl", PartType.CHILL,
                               position=(2, 0, 0))

# Get part data for voxel mesh
parts_data = part_manager.get_parts_with_types()
voxel_mesh.assign_part_types(parts_data)
```

### ML Mesh Recommendation System
`mesh_ml.py` provides two functions:
1. `extract_mesh_features(stl_loader)` - Get geometry features (volume, complexity, etc.)
2. `recommend_mesh_settings(stl_loader)` - Returns MeshRecommendation with voxel_size, refinement_level, confidence

The system:
- Falls back to heuristics if no trained model available
- Learns from `data/mesh_runs.csv` after 20+ simulations
- Uses scikit-learn gradient boosting for predictions
- Logs performance metrics automatically after each run via `log_mesh_training_example()`

## Integration Points

### VTK Visualization Pipeline
```python
# Pattern in visualizer.py
mesh = self._create_vtk_mesh(voxel_centers, values)
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(mesh)
actor = vtk.vtkActor()
actor.SetMapper(mapper)
```

### Qt Threading
**Critical**: Long-running simulations must use `QThread` to avoid blocking UI:
```python
from ui.qt import QThread, Signal


class SimulationWorker(QThread):
    progress = Signal(str)
    detailed_progress = Signal(int, int, str)
    finished = Signal(bool)
    
    def run(self):
        # Run simulation in background
        # Emit progress signals to update UI
```

### macOS-Specific Considerations
- Multiprocessing requires `spawn` start method (set in `main.py`)
- High-DPI scaling is automatic in Qt 6; the Qt 5 `AA_*` attributes are gone from `main.py`
- Metal backend via OpenCL (GPU acceleration currently disabled)
- CPU detection uses `sysctl` commands

### Windows-Specific Considerations
- If `wmic` command fails during CPU detection, program falls back gracefully
- Qt requires appropriate Visual C++ redistributables
- Use backslashes in file paths or raw strings

### Linux-Specific Considerations
- CPU detection reads `/proc/cpuinfo` for vendor identification
- Works on all major distributions (Ubuntu, Fedora, Arch, etc.)
- Memory limits adapt to available system RAM

## File Naming & Organization
- Test files: `test_*.py` in root
- Config files: `config/` directory with singleton pattern
- UI components: `ui/` directory, one main file
- Documentation: Markdown files in root (README.md, QUICKSTART.md, PROJECT_OVERVIEW.md)
- Sample data: `sample_stl_files/`, `data/mesh_runs.csv`
- Models: `models/` for trained ML artifacts

## Entry Points

### Main Application
- **`main.py`**: PySide6 application entry point
  - Sets multiprocessing method for macOS
  - Enables high DPI scaling
  - Launches `MainWindow` from `ui/main_window.py`

### Setup Script
- **`run.py`**: Dependency checker and launcher
  - Verifies Python 3.9+
  - Checks required packages
  - Offers to install missing dependencies
  - Launches `main.py`

### Utilities
- **`generate_samples.py`**: Creates test STL files
  - `create_simple_block()` - 2" cube
  - `create_thick_thin_test()` - Tests feeding distance
  - `create_l_bracket()` - Tests corner effects
  - `create_pin_boss()` - Tests thin features

- **`tools/train_mesh_model.py`**: ML model training
  - Loads simulation history from `data/mesh_runs.csv`
  - Trains gradient boosting regressor
  - Saves model to `models/mesh_recommender.joblib`
  - Requires scikit-learn package

## Dependencies to Respect
- **Numba**: Required for performance; do not remove parallel flags
- **VTK**: Specific version (>=9.2.6) for macOS compatibility; use memory-mapped arrays for compatibility
- **Qt**: Use `Fusion` style for consistent cross-platform appearance
- **Trimesh**: Primary mesh library; use `trimesh_obj` attribute consistently
- **NumPy/SciPy**: Use `np.` functions directly, not deprecated aliases
- **scikit-learn**: Optional, only needed for training ML mesh recommender
- **joblib**: Required for ML model persistence and parallel processing
- **psutil**: Required for memory monitoring and CPU detection
- **numpy-stl**: Required for STL file parsing

## Anti-Patterns to Avoid
- ❌ Don't use OpenCV or PIL for visualization (use VTK)
- ❌ Don't add threading without QThread for GUI operations
- ❌ Don't modify unit conversions - keep everything in inches
- ❌ Don't bypass `hardware_config.py` for thread/core allocation
- ❌ Don't enable GPU acceleration without testing Qt stability
- ❌ Don't create large arrays without using `create_managed_array()`
- ❌ Don't hardcode material properties - use `get_alloy(name)`
- ❌ Don't ignore memory warnings from `memory_limits.py`
- ❌ Don't modify voxel state arrays directly - use solidification engine methods

