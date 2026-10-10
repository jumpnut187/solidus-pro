# Quick Start Guide

## Installation (5 minutes)

### 1. Open Terminal
Navigate to the project folder:
```bash
cd "/Users/brendonjarett/Desktop/untitled folder"
```

### 2. Install Dependencies
```bash
pip3 install -r requirements.txt
```

This will install:
- PySide6 (GUI framework)
- VTK (3D visualization)
- NumPy, SciPy (numerical computing)
- Numba (JIT compilation for M3 optimization)
- Trimesh, numpy-stl (mesh processing)
- PyOpenCL (GPU acceleration via Metal)
- And more...

**Note**: Installation may take 5-10 minutes on first run.

### 3. Run the Simulator

**Option A - Simple:**
```bash
python3 main.py
```

**Option B - With setup check:**
```bash
python3 run.py
```

## Checking Hardware Detection

`test_hardware.py` and `test_materials.py` were referenced here for a long time
but have never existed in the repository. Use these instead.

### Hardware / GPU detection
```bash
python3 -c "import config.hardware_config"      # prints the detection summary
python3 tools/diagnose_gpu.py                   # detailed GPU backend report
```

Shows CPU core configuration, Metal/CUDA availability, Neural Engine detection
and the selected compute backend.

### Material database
```bash
python3 -c "from core.materials import get_all_alloy_names, print_alloy_info; \
    [print_alloy_info(n) for n in get_all_alloy_names()]"
```

Covers all seven alloys: WCB, CF8, C958, C964, CA6NM, MONEL400 and IN625.

## Using the Simulator

### Step-by-Step Workflow

1. **Launch Application**
   ```bash
   python3 main.py
   ```

2. **Load CAD File (STL or STEP)**
   - Click "Load CAD File (Inches)"
   - Select an STL or STEP (.step / .stp) file
   - STEP files are auto-fanned-out: each body in the assembly becomes its own part
   - Geometry is read and kept in **inches** — no unit conversion happens on import
   - **Note:** STEP support requires the optional `cascadio` package — install with `pip install cascadio` if a STEP upload fails

3. **Select Material**
   - Choose from dropdown: WCB, CF8, C958, C964, CA6NM, MONEL400 or IN625
   - View material properties in sidebar

4. **Configure Mesh** (Optional)
   - Mesh type: Voxel (default) or Tetrahedral (opt-in)
     - Tetrahedral follows the CAD surface and solves heat flow with finite elements; the size field becomes "Mesh size" (tet edge length = mesh size x `mesh.tet_edge_length_factor`, 2.0)
     - Default from `mesh.mesh_type` in `castsim.yaml` or `CASTSIM_MESH_TYPE`; set `tet` to make tet the default
     - A tet run that is too large (`mesh.tet_auto_max_nodes`, 300,000 nodes), fails to mesh, or meshes longer than `mesh.tet_mesh_timeout_s` (600 s) finishes on the voxel mesh and says why
   - Cut cells (CAD-accurate boundary): on (default)
     - Voxel runs use the CAD metal volume and mold-contact area in each boundary voxel
     - At 0.1 in, total solidification time within about 1 % of a fine tet reference (plain voxels: +2 to +17 %)
     - Turn off, or set `mesh.voxel_cut_cells: false` / `CASTSIM_VOXEL_CUT_CELLS=0`, to reproduce older voxel results
   - Base Voxel Size: 0.1 in (default)
     - Smaller = more accurate, slower
     - Larger = faster, less accurate
   - Refinement Level: 3 (default)
     - Higher = better detail in intricate areas
     - Voxel only; the tet mesh ignores refinement and memory mode

5. **Run Simulation**
   - Click "Run Simulation"
   - Progress shown in status area
   - Uses all 8 cores of M3 automatically

6. **Analyze Results**
   - **Ghost Model**: Translucent view of your part
   - **Red Regions**: Shrinkage and porosity areas
   - **Threshold Slider**: Filter by severity (30% default)

### Controls

**Visualization:**
- ☑️ Show Ghost Model - Toggle translucent part
- ☑️ Show Shrinkage - Toggle red defect regions
- 🎚️ Shrinkage Threshold - Adjust sensitivity

**3D View:**
- **Rotate**: Left mouse drag
- **Pan**: Middle mouse drag (or Shift + left drag)
- **Zoom**: Scroll wheel

**Export:**
- Click "Export Screenshot" to save view as PNG

## Performance Tips

### For Faster Results
- Increase base voxel size (0.15-0.25 in)
- Reduce refinement level (1-2)
- Use simpler geometry

### For More Accuracy
- Decrease voxel size (0.03-0.05 in)
- Increase refinement (3-4 levels)
- May take longer on complex parts

### Expected Times (M3 MacBook Air)

| Part Complexity | Voxels | Time |
|----------------|--------|------|
| Simple bracket | 10K | ~20s |
| Medium housing | 50K | ~80s |
| Complex manifold | 200K | ~5min |

## Interpreting Results

### Shrinkage Indicators

**Color Guide:**
- 🟥 **Dark Red** (>70%): Critical - Will definitely shrink
- 🟧 **Red** (50-70%): High risk - Likely to shrink
- 🟠 **Orange** (30-50%): Moderate - May shrink

### What to Look For

✅ **Good Casting:**
- Minimal red regions
- Defects only in non-critical areas
- Uniform distribution

⚠️ **Needs Improvement:**
- Moderate red in functional areas
- Consider adding risers
- May need gating changes

❌ **Poor Design:**
- Extensive red throughout
- Thick sections with no feeding
- Redesign recommended

### Common Issues & Solutions

**Problem: Large red region in thick section**
- **Solution**: Add riser near hot spot
- **Why**: Thick sections solidify slowly and need feeding

**Problem: Red streaks from thin to thick**
- **Solution**: Improve transition geometry
- **Why**: Abrupt thickness changes restrict feeding

**Problem: Red throughout entire part**
- **Solution**: 
   1. Try different alloy (`C958` has better fluidity, CA6NM is a balanced medium-alloy cast steel option)
  2. Increase pouring temperature
  3. Redesign with more uniform thickness

## Materials Guide

### WCB (Carbon Steel)
- **Best for**: General purpose castings
- **Shrinkage**: 6% volumetric
- **Feeding**: 6x thickness
- **Pros**: Good strength, economical
- **Cons**: Higher shrinkage than bronze

### CF8 (Stainless Steel 304)
- **Best for**: Corrosion resistance
- **Shrinkage**: 7.5% volumetric (highest)
- **Feeding**: 5.5x thickness
- **Pros**: Corrosion resistant
- **Cons**: Hardest to cast, highest shrinkage

### C958 (Aluminum Bronze)
- **Best for**: Complex geometry, thin sections
- **Shrinkage**: 4.5% volumetric (lowest)
- **Feeding**: 8x thickness
- **Pros**: Best fluidity, lowest shrinkage
- **Cons**: More expensive

### CA6NM (Cast Steel)
- **Best for**: General cast steels requiring balanced performance and moderate castability
- **Shrinkage**: ~6.3% volumetric (modeled)
- **Feeding**: 5.9x thickness
- **Pros**: Balanced thermal response, lower shrinkage than CF8
- **Cons**: Moderate castability sensitivity

## Model Notes

- `voxel_states` is the solver's solidification-progress field and is used as a fraction-solid proxy in post-processing.
- `feeding_distances` is the legacy name for the nearest-surface distance field.
- `feeding_path_lengths` is populated when riser-fed path tracing is available.
- `shrinkage_risk` and `porosity_risk` are the primary bounded defect-risk outputs. The `*_probability` names remain as legacy aliases.
- `porosity_risk` is a hybrid defect-risk field built from shrinkage plus relative solidification time and Niyama severity; it is not a calibrated porosity percentage.
- The "Shrink" view shows `shrinkage_probability` and the "Microporosity" view shows `porosity_probability`, each with chill points at 0 (`get_shrink_display_intensity` / `get_microporosity_display_intensity`). Projects saved with the old merged "Shrinkage/Porosity" view open on Shrink. Each result type has help text in `core/result_help.json`, shown by the info button next to the result selector in both apps.
- `niyama_values` are reported in the engine's approximate `C * sqrt(s) / mm` unit convention, and the derived Niyama risk field includes directional-solidification modifiers near chills.
- `local_thickness = 2 * modulus` is a section-size proxy, not a literal thickness measurement.
- The default solver is geometry-driven. It includes physics-inspired terms, but it is not a full thermal/phase-field simulation.
- `core/solidification.py` is the current authoritative output path; the thermal/physics modules outside that path are reference implementations unless explicitly wired into the app.

## Troubleshooting

### Application Won't Start

**Error: "No module named 'PySide6'"**
```bash
pip3 install PySide6
```

**Error: "No module named 'vtk'"**
```bash
pip3 install vtk
```

**Install all at once:**
```bash
pip3 install -r requirements.txt
```

### CAD File Won't Load

**Check file:**
- Is it a valid STL or STEP file?
- Is it in inches? (If in mm, select from file dialog)
- For STEP files specifically: install `cascadio` (`pip install cascadio`)
- Try opening in mesh viewer first

**Fix mesh issues:**
- Use Meshmixer or Netfabb to repair
- Ensure watertight (no holes)
- Check for flipped normals

### Simulation Fails

**"Not enough memory"**
- Reduce voxel refinement
- Increase base voxel size
- Close other applications

**"Mesh Generation Error" on a tetrahedral run**
- The estimated node count exceeds `mesh.tet_max_nodes` (3,000,000)
- Increase the mesh size; the message gives a size that fits

**"Mesh not watertight"**
- Repair STL file in mesh editing software
- Fill holes
- Fix inverted faces

### Slow Performance

**Check Activity Monitor:**
- Ensure Python using multiple cores
- Look for "Python" process at 400-800% CPU (4-8 cores)

**Optimize settings:**
- Increase voxel size
- Reduce refinement level
- Simplify geometry

### Visualization Issues

**Black screen:**
- Update graphics drivers
- Check VTK installation: `pip3 install --upgrade vtk`

**Can't see shrinkage:**
- Lower threshold slider
- Check "Show Shrinkage" is enabled
- Simulation may predict no defects (good!)

## Advanced Usage

### Command Line Testing

**Check hardware detection:**
```bash
python3 -c "import config.hardware_config"
python3 tools/diagnose_gpu.py
```

**Run the regression suite:**
```bash
python3 -m unittest discover tests
```

**Check Python environment:**
```bash
python3 -c "import sys; print(f'Python {sys.version}')"
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"
python3 -c "import vtk; print(f'VTK {vtk.vtkVersion.GetVTKVersion()}')"
```

### Performance Monitoring

While simulation runs, open Activity Monitor:
1. Search for "Python"
2. Look at CPU usage
3. Should see 400-800% (8 cores active)
4. Memory usage depends on model size

## Getting Help

### Check Logs
Console output shows detailed progress:
- Hardware detection
- Voxel generation stats
- Simulation progress
- Error messages

### Common Error Messages

**"Mesh is not watertight"**
- Fix: Repair STL file

**"GPU not available"**
- Info only, will use CPU (still fast on M3)

**"Failed to generate voxel mesh"**
- Increase voxel size or reduce refinement

## Next Steps

Once familiar with basic usage:

1. **Experiment with materials** - Compare WCB, CF8, C958, and CA6NM
2. **Test voxel settings** - Find balance of speed/accuracy
3. **Try complex parts** - Import your actual casting designs
4. **Compare designs** - Test multiple versions of same part
5. **Document results** - Export screenshots for reports

## Support

For technical issues:
1. Check console output for errors
2. Verify all dependencies installed
3. Test with simple STL first
4. Ensure STL file is valid

---

**Ready to start?**
```bash
python3 main.py
```

**Enjoy optimized casting simulation on your M3 MacBook Air! 🚀**
