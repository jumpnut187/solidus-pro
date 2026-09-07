# Casting Solidification Simulator
## Complete Project Overview

### 🎯 What This Simulator Does

This is a professional-grade casting solidification simulator that predicts **shrinkage** and **porosity** defects in metal castings. It's based on the same principles as commercial software like SolidCast and Magmasoft, but optimized specifically for your **M3 MacBook Air**.

### 🚀 Key Features

#### 1. **Full Hardware Optimization**
- ✅ Uses all **8 CPU cores** (4 Performance + 4 Efficiency)
- ✅ **GPU acceleration** via Metal (OpenCL backend)
- ✅ **Neural Engine** ready for future ML features
- ✅ Numba JIT compilation for hot loops
- ✅ Parallel processing with joblib

#### 2. **Advanced Physics**
- ✅ **Geometric modulus** based solidification
- ✅ **Edge-inward** progression from surfaces
- ✅ **Vertical solidification** (bottom-up due to density)
- ✅ **Feeding distance** = 3x material thickness
- ✅ High refinement adaptive voxel mesh

#### 3. **Material Library**
- **WCB** (Carbon Steel) - General purpose, 6% shrinkage
- **CF8** (Stainless Steel 304) - Corrosion resistant, 7.5% shrinkage
- **Bronze** (Copper-Tin) - Best fluidity, 4.5% shrinkage

#### 4. **Professional Visualization**
- 🔍 **Translucent "ghost" 3D model** of your part
- 🔴 **Red shrinkage/porosity** regions shown internally
- 🎚️ **Adjustable threshold** for defect sensitivity
- 📸 **Export screenshots** for documentation

### 📁 Project Structure

> The structural sections of this file drifted badly for a long time; the
> tree below was corrected 2026-08-27. For the live call path, layer
> contracts and unit zones see `docs/ARCHITECTURE.md`, and for day-to-day
> facts see `AGENT.md`. The "What's Modeled / Simplifications" section
> further down has been maintained and is reliable.


```
Python_Cast_Sim/
│
├── main.py                    # Desktop entry point (PySide6 + VTK)
├── run_web.py                 # Web entry point (Flask + React/Babylon)
├── run.py                     # Dependency check, then launches main.py
├── install.sh                 # Installation script
├── requirements.txt
│
├── AGENT.md                   # Authoritative notes; read this first
├── README.md                  # Full documentation
├── QUICKSTART.md
├── MATH_AUDIT_CHECKLIST.md    # Known-weak formulas
├── PROJECT_OVERVIEW.md        # This file
│
├── config/                    # Settings, hardware detection, exceptions
│   ├── settings.py            # Settings singleton; castsim.yaml / CASTSIM_* env
│   ├── hardware_config.py
│   └── exceptions.py
│
├── core/                      # Orchestration; knows the mesh and part types
│   ├── simulation_pipeline.py # run_pipeline -- the one entry both UIs use
│   ├── solidification.py      # SolidificationEngine (large; see ARCHITECTURE)
│   ├── solidification_loop.py # Time integration
│   ├── solidification_params.py
│   ├── analysis_strategies.py # Post-analysis dispatch
│   ├── voxel_mesh.py          # AdaptiveVoxelMesh
│   ├── part_manager.py        # PartType, part-type codes and predicates
│   ├── materials.py           # 7 alloys, 40+ properties each
│   ├── feeding_path.py, riser_design.py, microstructure.py,
│   ├── stress_analysis.py, defect_regions.py, report_generator.py
│   ├── acceleration.py, gpu_compute.py, octree.py, memory_*.py
│   ├── surrogate/             # NumPy/Core ML surrogate model
│   └── deep_model/            # PyTorch 3D U-Net (optional, lazy torch import)
│
├── physics/                   # Mesh-agnostic solvers -- LIVE, not reference
│   ├── heat_transfer.py       # HeatTransferModel (SI/metres internally)
│   ├── solidification_physics.py
│   ├── gas_porosity.py
│   └── units.py               # The three length zones live here
│
├── ui/                        # PySide6 desktop
│   ├── main_window.py, visualizer.py, workers.py
│   └── dialogs/
│
├── web/                       # Flask API
│   ├── app_factory.py, routes.py, payloads.py, repository.py
│   └── services/              # simulation_queue.py, fair_share.py
│
├── web_frontend/              # React + TypeScript + Babylon.js client
├── tests/                     # ~580 unittest tests
├── tools/                     # CLI utilities, model training, diagnostics
└── docs/
    └── ARCHITECTURE.md        # Live path, layer contracts, extension points
```

### 🔧 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **GUI** | PySide6 | Modern, native macOS interface |
| **3D Visualization** | VTK | Professional CAD-quality rendering |
| **Numerical** | NumPy, SciPy | Fast array operations |
| **Acceleration** | Numba | JIT compilation, parallel loops |
| **GPU** | PyOpenCL | Metal backend for M3 GPU |
| **Parallel** | Joblib | Multi-core task distribution |
| **Mesh** | Trimesh, numpy-stl, cascadio (optional) | STL/STEP processing and analysis |
| **ML Ready** | CoreML, ONNX | Neural Engine support |

### 🎓 How It Works

#### Step 1: Load Geometry
```
STL or STEP (inches) → Validate mesh → Extract properties   [no unit conversion]
```

#### Step 2: Generate Voxel Mesh
```
Create uniform grid → Detect features → Adaptive refinement → Build neighbors
```
- Fine voxels in intricate areas
- Coarse voxels in solid sections
- Up to 4 refinement levels

#### Step 3: Simulate Solidification
```
Initialize surface → Time-step iteration → Update states → Record times
```

**Legacy path (`schedule_legacy`):**
- Starts at edges (surface voxels)
- Progresses inward based on **geometric modulus**
- Bottom solidifies faster (density effects)
- Rate: `1 / (1 + modulus × 0.1)`

**Default path (`thermal_coupled`):** the primary solver now integrates an
explicit thermal diffusion field coupled to fraction-solid evolution. It
applies temperature-dependent conductivity, specific heat, and density
curves per voxel, uses an isotropic weighted Laplacian stencil (exact on
quadratic fields), and releases latent heat on the same time step it is
computed (in-step temperature correction, no one-step lag). Mold interfaces
use a Robin condition with exponential air-gap HTC decay; chills act as
finite graphite heat sinks (diffusivity override + ambient-HTC boundary)
rather than fixed-temperature pins. Fraction solid can use the lever rule
or the true Scheil microsegregation equation with per-alloy solvent melting
point and partition coefficient, lever-rule solidus completion, and an
optional eutectic cap.

#### Step 4: Calculate Defects
```
Feeding distances → Shrinkage probability → Porosity probability → Results
```

**Shrinkage Formula:**
```
shrinkage = material_shrinkage × distance_factor × modulus_factor
```

**Distance Factor:**
```
distance_factor = min(distance_from_surface / feeding_distance, 1.0)
```

**Modulus Factor:**
```
modulus_factor = min(geometric_modulus / 50mm, 1.0)
```

#### Step 5: Visualize
```
Ghost model (25% opacity) + Red defects (80% opacity) → Interactive 3D
```

### 📊 Performance

#### M3 MacBook Air (8-core) Benchmarks

| Model | Vertices | Voxels | Time | Cores Used |
|-------|----------|--------|------|------------|
| Simple block | 1K | 10K | 20s | 8 (100%) |
| L-bracket | 5K | 30K | 45s | 8 (100%) |
| Complex part | 20K | 100K | 180s | 8 (100%) |
| Very complex | 100K | 500K | 15min | 8 (100%) |

**Speedup vs Single-Core:** 5-7x faster

### 🎯 Validation Metrics

The simulator calculates:

1. **Geometric Modulus** (mm)
   - Formula: `Volume / Surface Area`
   - Indicates cooling rate
   - Higher = slower cooling = more shrinkage

2. **Feeding Distance** (mm)
   - Formula: `Thickness × Material Factor`
   - WCB: 3.0x, CF8: 2.5x, Bronze: 4.0x
   - Beyond this = unfed regions

3. **Shrinkage Risk** (0-1)
   - Combined distance and modulus factors
   - 0 = No shrinkage, 1 = Definite shrinkage

4. **Porosity Risk** (0-1)
   - Shrinkage-based feed-failure risk, amplified by relative solidification time
   - Also shaped by Niyama directional-solidification severity

5. **Hot Tear Risk** (0-1) - Mechanical Constraint Model
   - Hot tears are MECHANICAL failures at grain boundaries during BTR (Brittle Temperature Range)
   - **Key Physics**: Thin sections solidify first → become rigid → contract → pull on adjacent mushy zones
   - Four strain components:
     - **Constraint Strain (40%)**: Accumulated strain from already-solid neighbors pulling on mushy voxels
     - **Path Strain (25%)**: Strain accumulated along connected solid paths (longer paths = more pull)
     - **Geometry Strain (20%)**: Section change stress concentrators (T-junctions, L-junctions)
     - **Differential Strain (15%)**: Cooling rate differences between neighbors
   - BTR vulnerability window: ~85-99% solid (material-dependent)
   - Material susceptibility: WCB=0.30, CF8=0.65 (high - wide freezing range), C958=0.20 (low)

### 🔬 Physical Accuracy

#### What's Modeled:
✅ Geometric modulus (volume/area)
✅ Feeding distance limits
✅ Density-driven vertical flow (legacy schedule path only)
✅ Edge-inward solidification (legacy schedule path only)
✅ Material-specific shrinkage rates
✅ Explicit thermal diffusion (default `thermal_coupled` path)
✅ Temperature-dependent k/cp/rho property curves
✅ Isotropic weighted Laplacian stencil (exact on quadratic fields)
✅ Mold-interface Robin boundaries with exponential air-gap HTC decay
✅ Finite-heat-sink chills (graphite diffusivity + ambient HTC)
✅ Same-step latent heat release
✅ True Scheil microsegregation option
✅ Shrinkage-driven Darcy feeding velocity (v = β·R)

#### Simplifications:
⚠️ No mold-filling simulation — castings start at a uniform pouring temperature
⚠️ No explicit mold domain — constant-temperature sink + interface HTC
⚠️ Diffusion stencil approximate on graded octree meshes (exact on uniform grids)
⚠️ Shrinkage/porosity/hot-tear outputs are calibrated engineering risk scores (0–1), not first-principles porosity fractions
⚠️ Simplified convection model
⚠️ No turbulence effects
⚠️ Steady-state assumptions

**Accuracy Level:** Good for preliminary design validation and hot spot identification. For production castings, validate with trials.

### 💡 Use Cases

#### ✅ Perfect For:
- 🎯 Design validation
- 🎯 Hot spot identification
- 🎯 Riser placement planning
- 🎯 Material selection
- 🎯 Section thickness optimization
- 🎯 Student learning
- 🎯 Research projects

#### ⚠️ Not Ideal For:
- Final production approval (needs testing)
- Complex alloy systems
- Investment casting fine details
- Sand mold thermal effects
- Detailed gating analysis

### 📈 Typical Workflow

```
Design Part (CAD)
    ↓
Export STL (inches)
    ↓
Load in Simulator
    ↓
Select Material
    ↓
Run Simulation (20s - 5min)
    ↓
Analyze Results
    ↓
┌─────────────┐
│ Good? → Yes → Proceed to casting
│           ↓
│          No → Modify design
│               ↓
└───────────────┘
      Repeat
```

### 🎨 Result Interpretation

#### Color Guide:
- **Translucent Grey/Blue**: Your part (ghost model)
- **Dark Red (>70%)**: Critical defects - will definitely occur
- **Red (50-70%)**: High risk - likely to occur
- **Orange (30-50%)**: Moderate risk - may occur
- **No color**: Safe regions

#### Common Patterns:

**Hot Spots (Thick Sections)**
```
Problem: Large red blob in thick area
Cause: Slow cooling, long feeding distance
Solution: Add riser near this region
```

**Feeding Streaks**
```
Problem: Red paths from thin to thick
Cause: Restricted metal flow
Solution: Smooth thickness transitions
```

**Isolated Regions**
```
Problem: Red pockets far from surface
Cause: Beyond feeding distance
Solution: Add feeding channels or redesign
```

### 🔧 Customization

#### Adjust Voxel Settings:
```python
# In UI or code:
base_voxel_size = 0.1  # inches (smaller = more accurate, slower)
refinement_level = 3   # 1-4 (higher = more detail)
```

#### Adjust Solidification Parameters:
```python
# In core/solidification_params.py (nested dataclasses):
params.air_gap.htc_floor_fraction = 0.35            # 0.05-1.0; fraction of interface HTC remaining after the air gap forms
params.thermal_boundary.chill_ambient_htc = 100.0   # 1-10000 W/(m^2*K); HTC from chill surfaces to mold/ambient (finite-heat-sink chills)
```

#### Modify Materials:
```python
# In core/materials.py:
ALLOY_DATABASE['MyAlloy'] = AlloyProperties(
    name='My Custom Alloy',
    liquidus_temp=1500,
    # ... etc
)
```

#### Add New Physics:
```python
# In core/solidification.py:
# Extend SolidificationEngine class
# Override calculate methods
```

### 📚 Learning Resources

#### Understanding Casting:
1. **Geometric Modulus**: Key predictor of solidification
2. **Chvorinov's Rule**: Solidification time ∝ (V/A)²
3. **Directional Solidification**: Bottom-up, edge-inward
4. **Feeding**: Metal must feed from liquid to solid

#### References:
- Campbell, J. "Castings" (2003)
- Beeley, P. "Foundry Technology" (2001)
- ASM Handbook Vol. 15 "Casting"

### 🐛 Troubleshooting

#### Performance Issues:
```bash
# Check if using all cores:
# Open Activity Monitor → Search "Python" → Should see 400-800% CPU
```

#### Memory Issues:
```python
# Reduce memory usage:
base_voxel_size = 5.0  # Larger voxels
refinement_level = 2   # Less refinement
```

#### Visualization Issues:
```bash
# Update VTK:
pip3 install --upgrade vtk
```

### 🔮 Future Enhancements

Planned features:
- [ ] Real-time simulation preview
- [ ] Automatic riser design
- [ ] Neural network predictions (using Neural Engine)
- [ ] Thermal gradient visualization
- [ ] Animation of solidification
- [ ] Multi-part assemblies
- [ ] Gating system analysis
- [ ] Export to CAD formats

### 📞 Support

For issues:
1. Check `QUICKSTART.md` for common solutions
2. Run `python3 test_hardware.py` to verify setup
3. Check console output for error messages
4. Ensure STL file is valid (watertight)

### 🏆 Credits

**Physics Models:** Based on SolidCast and Magmasoft principles
**Optimization:** Leverages Apple M3 architecture
**Development:** Modern Python scientific stack

---

## Quick Commands

```bash
# Install everything
./install.sh

# Or manually:
pip3 install -r requirements.txt

# Test hardware
python3 test_hardware.py

# Generate samples
python3 generate_samples.py

# Run simulator
python3 main.py

# Quick launcher
python3 run.py
```

---

## MAGMASOFT-Style Advanced Features (New!)

### 🔬 Advanced Analysis Modules

The simulator now includes MAGMASOFT-style advanced analysis capabilities:

#### 1. **Feeding Path Analysis** (`core/feeding_path.py`)
- Traces liquid metal flow paths from risers to casting voxels
- Uses Kozeny-Carman permeability calculation
- Identifies blocked regions that cannot be fed
- Combines with Niyama for better defect prediction:
  - Low Niyama + blocked path = HIGH RISK
  - Blocked path alone = DEFECT LIKELY

#### 2. **Thermal Stress & Distortion** (`core/stress_analysis.py`)
- Thermal strain from cooling
- Constraint strain from differential solidification
- Effective stress magnitude calculation
- Crack susceptibility mapping
- Distortion vectors for visualization

#### 3. **Microstructure Prediction** (`core/microstructure.py`)
- SDAS prediction: `λ₂ = A × Ṫ^(-n)`
- Mechanical property estimation (UTS, elongation)
- Fine/coarse structure zone identification
- Material-specific correlations (WCB, CF8, C958, A356, etc.)

#### 4. **Design Optimization** (`core/optimizer.py`)
- Riser size optimization
- Riser position optimization
- Multi-objective optimization (shrinkage, yield, coverage)
- MAGMASOFT "Autonomous Engineering" style iteration

#### 5. **Results Comparison** (`core/results_manager.py`)
- Save/load simulation results
- Compare before/after design changes
- Generate comparison reports
- Track design iterations

### 📊 Quick Wins Implemented

1. **Chvorinov Time Display**: Predicted solidification time per region
2. **Feeding Path Visualization**: Arrow data for feeding flow direction
3. **Defect Volume Estimation**: Total defect volume in cm³/in³
4. **Comparison Mode**: Save/load results for design iteration
5. **Material Property Tooltips**: Detailed calculated properties in UI

### 🎛️ New Analysis Options (UI)

Enable in Analysis Options dialog:
- ☑️ Feeding Path Analysis
- ☑️ Thermal Stress & Distortion  
- ☑️ Microstructure (SDAS) Prediction

### 📐 New Equations

**SDAS Prediction:**
```
λ₂ = A × cooling_rate^(-n)
UTS = σ₀ + k/√λ₂  (Hall-Petch)
```

**Stress Calculation:**
```
ε_thermal = α × ΔT
ε_constraint = α × (T_neighbor - T_self)
σ = E_eff × (ε_thermal + ε_constraint)
```

**Feeding Path Quality:**
```
K = (λ² × (1-fs)³) / (180 × fs²)  (Kozeny-Carman)
quality = distance_factor × permeability_factor
```

### 📁 New Files

```
core/
├── feeding_path.py      # Feeding path analysis (BFS + permeability)
├── stress_analysis.py   # Thermal stress & distortion
├── microstructure.py    # SDAS & mechanical properties
├── optimizer.py         # Design optimization
└── results_manager.py   # Save/load/compare results

data/
└── results/            # Saved simulation results
```

### 🔄 Comparison with MAGMASOFT

| Feature | MAGMASOFT | This Simulator |
|---------|-----------|----------------|
| Shrinkage/Porosity | ✅ | ✅ |
| Hot Spot Detection | ✅ | ✅ |
| Niyama Criterion | ✅ | ✅ |
| Feeding Path Analysis | ✅ | ✅ (NEW) |
| Thermal Stress | ✅ | ✅ (NEW) |
| SDAS Prediction | ✅ | ✅ (NEW) |
| Design Optimization | ✅ | ✅ (NEW) |
| Hot Tear Risk | ✅ | ✅ |
| Sand Burn-In | ✅ | ✅ |
| Real Temperature Field | ✅ | 🔶 (exists, not integrated) |
| Filling Simulation | ✅ | ❌ |

---

**Ready to predict casting defects on your M3 MacBook Air! 🚀**
