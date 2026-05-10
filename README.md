# Casting Solidification Simulator
### Optimized for Apple M3 MacBook Air

Advanced casting solidification simulator with shrinkage and porosity prediction, inspired by SolidCast and Magmasoft.

## Features

✨ **Multi-Core CPU Acceleration**
- Utilizes all 8 cores (4 Performance + 4 Efficiency) of M3 chip
- Numba JIT compilation with parallel processing
- Joblib for efficient task distribution

💾 **Intelligent Memory Management** ⭐ NEW!
- Automatic RAM vs disk storage for large models
- Prevents crashes on complex parts (10M+ voxels)
- Memory-mapped arrays for 50-90% RAM savings
- Three modes: Auto (recommended), Force RAM, Force Disk
- See [MEMORY_QUICKSTART.md](MEMORY_QUICKSTART.md) for details

🎮 **GPU Acceleration (Metal)**
- OpenCL with Metal backend for GPU compute
- Parallel voxelization and distance calculations

🧠 **Neural Engine Ready**
- CoreML integration for future ML-based predictions
- ONNX runtime support

## Capabilities

### STL Import
- Load STL files in inches (auto-converts to mm)
- Watertight mesh validation and repair
- Volume and surface area calculations

### Adaptive Voxel Mesh
- Automatic refinement in intricate sections
- Coarse voxels in solid regions
- Multi-level octree refinement (up to 4 levels)
- Parallel voxel generation

### Physics-Based Solidification
- **Geometric modulus** calculation for each voxel
- **Edge-inward** solidification from surfaces
- **Vertical progression** due to density changes (liquidus→solidus)
- **Material-specific feeding-distance multipliers**
- High refinement for accurate results

### Material Properties
- **WCB** (Carbon Steel Casting)
- **CF8** (Stainless Steel 304)
- **C958** (Aluminum Bronze)
- **CA6NM** (Cast Steel)

Each material includes:
- Liquidus/solidus temperatures
- Density changes
- Thermal properties
- Shrinkage characteristics
- Feeding distance multipliers

### Visualization
- **3D translucent "ghost" model** of the casting
- **Red shrinkage/porosity regions** displayed internally
- High accuracy placement inside model geometry
- Interactive rotation, zoom, pan
- Adjustable shrinkage threshold
- Screenshot export

## Installation

### Prerequisites
- Python 3.9 or higher
- macOS (for M3 optimization)

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Note: For GPU acceleration, ensure Xcode Command Line Tools are installed
xcode-select --install
```

## Usage

### Run the Simulator

```bash
python main.py
```

### Workflow

1. **Load STL File**
   - Click "Load STL File (Inches)"
   - Select your STL model
   - Model will be converted from inches to millimeters

2. **Select Material**
   - Choose alloy: WCB, CF8, C958, or CA6NM
   - View material properties

3. **Configure Mesh**
   - Base Voxel Size: 0.5-10mm (default: 2mm)
   - Refinement Level: 1-4 (default: 3)
   - Higher refinement = more accuracy but slower

4. **Run Simulation**
   - Click "Run Simulation"
   - Watch progress in status area
   - Simulation uses all CPU cores

5. **Analyze Results**
   - Translucent ghost model shows part geometry
   - Red regions indicate shrinkage/porosity
   - Adjust threshold slider to filter severity
   - Toggle visibility of components
   - Export screenshot

## Hardware Optimization

### CPU (M3 Chip)
The simulator automatically detects and optimizes for your M3:

- **Performance Cores (4)**: Used for compute-intensive tasks
  - Voxel generation
  - Solidification simulation
  - Modulus calculations

- **Efficiency Cores (4)**: Used for I/O and background tasks
  - File loading
  - UI updates
  - Data management

### Parallel Processing
- Numba JIT with `parallel=True` for hot loops
- Joblib for embarrassingly parallel operations
- Automatic chunk sizing based on core count

### Memory Management
- Uses ~70% of available RAM
- Automatic memory estimation
- Efficient numpy arrays (float32)

## Technical Details

### Solidification Algorithm

1. **Initialization**
   - Surface voxels marked as solidification fronts
   - All internal voxels start as liquid

2. **Time-Stepping**
   - Each time step updates solidification state
   - Solidification rate = f(geometric modulus)
   - Smaller modulus → faster solidification
   - Vertical factor: bottom solidifies faster

3. **Shrinkage Calculation**
   - Distance to nearest feeding source
   - Beyond feeding distance → high shrinkage
   - Combined with modulus factor

4. **Porosity Prediction**
   - Based on shrinkage probability
   - Amplified by solidification time
   - Longer solidification → more porosity

### Geometric Modulus

```
Modulus = Volume / Surface Area
```

- Fundamental indicator of solidification behavior
- Higher modulus = slower cooling = more shrinkage
- Used to predict hot spots and feeding issues

### Feeding Distance

```
Feeding Distance = Thickness × Material Multiplier
```

- **WCB**: 6.0x thickness
- **CF8**: 5.5x thickness
- **C958**: 8.0x thickness
- **CA6NM**: 5.9x thickness

Regions beyond feeding distance cannot be fed by risers and will experience shrinkage.

## Model Assumptions

- `voxel_states` is the authoritative fraction-solid proxy for the fast engine. It is now generated from precomputed start/end solidification schedules (surface/chill early, feeder-connected riser regions late), not from a pure neighbor-fill front.
- `feeding_distances` is the legacy public name for the engine's nearest-surface distance field.
- `feeding_path_lengths` is only populated for riser-fed cases where the solver can trace a connected path to a riser through the mush.
- `shrinkage_risk` and `porosity_risk` are the primary bounded 0-1 defect-risk fields. `shrinkage_probability` and `porosity_probability` remain as legacy aliases for compatibility with existing UI/export code.
- `porosity_risk` is a hybrid risk field built from shrinkage as the base feed-failure signal, then amplified by relative solidification time and Niyama directional-solidification severity. It is not a calibrated void-volume fraction.
- The merged "Shrinkage/Porosity" visualization uses the maximum of the shrinkage and porosity risk fields for thresholding and color intensity.
- `niyama_values` uses the standard `G / sqrt(R)` form, but the gradient and cooling-rate fields are estimated from the engine's geometric/time fields rather than a full thermal solve. Thresholds are interpreted in the engine's approximate `C * sqrt(s) / mm` units.
- `local_thickness = 2 * modulus` is treated as a section-size proxy for relative ranking, not a literal wall-thickness measurement.

## Known Simplifications

- The default solver is still a geometry-driven solidification engine. It is faster than a full thermal/phase simulation, and uses a schedule-driven directional-solidification proxy rather than a solved thermal PDE.
- The `SCHEIL` option uses a bounded linearized Scheil surrogate anchored to liquidus and solidus because the repo does not yet include a full phase-diagram path.
- Feeding effectiveness uses a transient mush-state snapshot for Darcy resistance. In riser-fed cases it now prefers a traced riser path length, but open-casting cases still fall back to the surface-distance proxy.
- Niyama risk interpretation includes directional-solidification modifiers near chills so the reported risk field tracks improved feeding conditions without changing the raw `Ny` output itself.
- Stress/distortion, hot tearing, and microstructure results are comparative engineering guidance rather than full FEA or CALPHAD predictions.
- `core/solidification.py` is the authoritative application path for user-visible results today. `core/thermal_simulation.py`, `physics/niyama.py`, and `physics/shrinkage.py` are secondary/reference modules and are not currently the authoritative output path for the UI.

## Formula Provenance

- Niyama: uses the standard `Ny = G / sqrt(R)` criterion form. This repo's field construction is approximate and geometry-driven unless a thermal solve is added.
- Darcy feeding resistance: uses the common mushy-zone pressure-drop form `delta_P = mu * L * v / K` with permeability from the alloy's Kozeny-Carman-style relation.
- Chvorinov time scaling: uses the standard modulus-based `t = B * M^2` family with material-tuned mold constants.
- SDAS correlations: use power-law cooling-rate correlations and simplified property mappings intended for comparative foundry guidance, not alloy certification.
- Chill and riser modifiers: use engineering heuristics inspired by directional-solidification rules of thumb rather than a calibrated foundry handbook dataset.
- Reference notes: NIST Scheil notes (`https://www.ctcms.nist.gov/~kattner/solidifc/scheil.html`) and the MDPI Materials Niyama discussion (`https://www.mdpi.com/1996-1944/19/1/198`) informed the audit and naming/unit decisions.

## Results Interpretation

### Shrinkage (Red Regions)
- **Dark Red**: High probability (>70%)
- **Red**: Medium probability (50-70%)
- **Orange**: Low probability (30-50%)

### What to Look For
- ✅ **Good**: Minimal red regions, localized near risers
- ⚠️ **Caution**: Moderate red regions, consider adding risers
- ❌ **Poor**: Extensive red throughout part, redesign needed

### Recommended Actions
- Add risers near hot spots (thick sections)
- Modify part geometry to improve feeding
- Consider different alloy with better fluidity
- Adjust pouring temperature

## Performance Benchmarks

### M3 MacBook Air (8-core)

| Model Complexity | Voxels | Generation | Simulation | Total |
|-----------------|--------|------------|------------|-------|
| Simple (1K vertices) | ~10K | 5s | 15s | 20s |
| Medium (10K vertices) | ~50K | 20s | 60s | 80s |
| Complex (100K vertices) | ~200K | 90s | 240s | 330s |

*Times are approximate and vary with refinement level*

## Troubleshooting

### Import Errors
If you get import errors for VTK or PyQt5:
```bash
pip install --upgrade vtk PyQt5
```

### GPU Not Detected
Metal should be automatically available on M3. If not:
```bash
# Check OpenCL platforms
python -c "import pyopencl as cl; print(cl.get_platforms())"
```

### Slow Performance
- Reduce voxel refinement level
- Increase base voxel size
- Close other applications
- Check Activity Monitor for background processes

### Mesh Not Loading
- Ensure STL file is valid (watertight)
- Try opening in mesh repair tool (Meshmixer, Netfabb)
- Check file units (should be inches)

## Future Enhancements

- [ ] Riser design automation
- [ ] Thermal gradient visualization
- [ ] Gating system analysis
- [ ] Neural network for instant predictions
- [ ] Multi-part assemblies
- [ ] Animation of solidification progression
- [ ] Export results to CAD formats

## Technical Stack

- **GUI**: PyQt5
- **3D Visualization**: VTK
- **Numerical Computing**: NumPy, SciPy
- **Acceleration**: Numba, Joblib, PyOpenCL
- **Mesh Processing**: Trimesh, numpy-stl
- **ML Ready**: CoreML, ONNX Runtime

## License

Educational and research use.

## Credits

Based on casting simulation principles from SolidCast and Magmasoft.

Optimized for Apple Silicon (M3) by leveraging:
- Multi-core CPU parallelism
- Metal GPU acceleration
- Neural Engine capabilities

---

**Note**: This simulator provides predictions based on geometric analysis and simplified solidification models. For production casting, validation with actual trials is recommended.
