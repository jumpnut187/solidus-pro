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
- PyQt5 (GUI framework)
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

## Testing Hardware Optimization

### Test M3 Detection
```bash
python3 test_hardware.py
```

This will show:
- CPU core configuration (4P + 4E)
- GPU/Metal availability
- Neural Engine detection
- Parallel performance benchmark

### Test Material Database
```bash
python3 test_materials.py
```

Shows all alloy properties for WCB, CF8, C958, and CA6NM.

## Using the Simulator

### Step-by-Step Workflow

1. **Launch Application**
   ```bash
   python3 main.py
   ```

2. **Load STL File**
   - Click "Load STL File (Inches)"
   - Select your STL model
   - File will auto-convert from inches to millimeters

3. **Select Material**
   - Choose from dropdown: WCB, CF8, C958, or CA6NM
   - View material properties in sidebar

4. **Configure Mesh** (Optional)
   - Base Voxel Size: 2mm (default)
     - Smaller = more accurate, slower
     - Larger = faster, less accurate
   - Refinement Level: 3 (default)
     - Higher = better detail in intricate areas

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
- Increase base voxel size (3-5mm)
- Reduce refinement level (1-2)
- Use simpler geometry

### For More Accuracy
- Decrease voxel size (0.5-1mm)
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
- The merged "Shrinkage/Porosity" view uses the maximum of the shrinkage and porosity risk fields.
- `niyama_values` are reported in the engine's approximate `C * sqrt(s) / mm` unit convention, and the derived Niyama risk field includes directional-solidification modifiers near chills.
- `local_thickness = 2 * modulus` is a section-size proxy, not a literal thickness measurement.
- The default solver is geometry-driven. It includes physics-inspired terms, but it is not a full thermal/phase-field simulation.
- `core/solidification.py` is the current authoritative output path; the thermal/physics modules outside that path are reference implementations unless explicitly wired into the app.

## Troubleshooting

### Application Won't Start

**Error: "No module named 'PyQt5'"**
```bash
pip3 install PyQt5
```

**Error: "No module named 'vtk'"**
```bash
pip3 install vtk
```

**Install all at once:**
```bash
pip3 install -r requirements.txt
```

### STL Won't Load

**Check file:**
- Is it a valid STL file?
- Is it in inches? (If in mm, select from file dialog)
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

**Test hardware:**
```bash
python3 test_hardware.py
```

**Test materials:**
```bash
python3 test_materials.py
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
