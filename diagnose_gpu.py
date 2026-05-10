#!/usr/bin/env python3
"""
GPU Diagnostic Script for Python Cast Sim
Run this to diagnose why GPU acceleration is not available.
"""
import sys
import platform

print("=" * 70)
print("GPU DIAGNOSTIC REPORT")
print("=" * 70)
print()

# System info
print("SYSTEM INFO:")
print(f"  Platform: {platform.system()} {platform.machine()}")
print(f"  Python: {sys.version}")
print(f"  Python executable: {sys.executable}")
print()

# Check for Apple Silicon
is_apple_silicon = platform.system() == 'Darwin' and platform.machine() == 'arm64'
print(f"  Apple Silicon detected: {is_apple_silicon}")
print()

# Check PyOpenCL
print("PYOPENCL STATUS:")
try:
    import pyopencl as cl
    print(f"  PyOpenCL installed: YES")
    print(f"  Version: {cl.VERSION_TEXT}")
    
    # Check platforms
    try:
        platforms = cl.get_platforms()
        print(f"  OpenCL platforms found: {len(platforms)}")
        
        if len(platforms) == 0:
            print()
            print("  ERROR: No OpenCL platforms found!")
            print("  This is likely the root cause of the issue.")
            print()
            print("  POSSIBLE CAUSES ON MACOS:")
            print("    1. OpenCL is being phased out by Apple in favor of Metal")
            print("    2. macOS may need OpenCL ICD (Installable Client Driver)")
            print("    3. Sandboxing or permissions may be blocking access")
            print()
        else:
            for i, plat in enumerate(platforms):
                print()
                print(f"  Platform {i}: {plat.name}")
                print(f"    Vendor: {plat.vendor}")
                print(f"    Version: {plat.version}")
                
                try:
                    # Try GPU devices first
                    gpu_devices = plat.get_devices(device_type=cl.device_type.GPU)
                    print(f"    GPU Devices: {len(gpu_devices)}")
                    for j, dev in enumerate(gpu_devices):
                        print(f"      GPU {j}: {dev.name}")
                        print(f"        Compute units: {dev.max_compute_units}")
                        print(f"        Global memory: {dev.global_mem_size / (1024**3):.2f} GB")
                except cl.cffi_cl.LogicError as e:
                    print(f"    GPU Devices: None found ({e})")
                except Exception as e:
                    print(f"    GPU Devices: Error - {type(e).__name__}: {e}")
                
                try:
                    # Also check all device types
                    all_devices = plat.get_devices()
                    print(f"    All Devices: {len(all_devices)}")
                    for j, dev in enumerate(all_devices):
                        dev_type = cl.device_type.to_string(dev.type)
                        print(f"      Device {j}: {dev.name} ({dev_type})")
                except Exception as e:
                    print(f"    All Devices: Error - {e}")
                    
    except Exception as e:
        print(f"  Error getting platforms: {type(e).__name__}: {e}")
        
except ImportError as e:
    print(f"  PyOpenCL installed: NO")
    print(f"  Import error: {e}")
except Exception as e:
    print(f"  PyOpenCL error: {type(e).__name__}: {e}")

print()

# Check CuPy (for NVIDIA)
print("CUPY STATUS (NVIDIA CUDA):")
try:
    import cupy as cp
    print(f"  CuPy installed: YES")
    device_count = cp.cuda.runtime.getDeviceCount()
    print(f"  CUDA devices found: {device_count}")
except ImportError:
    print(f"  CuPy installed: NO (not needed for Apple Silicon)")
except Exception as e:
    print(f"  CuPy error: {type(e).__name__}: {e}")

print()

# Test the application's GPU detection
print("APPLICATION GPU DETECTION:")
try:
    sys.path.insert(0, '.')
    from core.gpu_compute import GPUComputeManager, GPUBackend
    
    # Force re-detection
    manager = GPUComputeManager()
    manager._detection_attempted = False
    manager._detect_backends()
    
    print()
    print(f"  Selected backend: {manager.backend.value}")
    print(f"  Device name: {manager.device_name}")
    print(f"  GPU available: {manager.is_gpu_available}")
    
except Exception as e:
    print(f"  Error testing app GPU: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("RECOMMENDATIONS:")
print("=" * 70)

if platform.system() == 'Darwin':
    print("""
On macOS, OpenCL support is limited and being deprecated by Apple.
The M3 GPU may not be fully accessible via OpenCL/PyOpenCL.

ALTERNATIVE SOLUTIONS:

1. USE METAL DIRECTLY (recommended for Apple Silicon):
   - Install: pip install pyobjc-framework-Metal pyobjc-framework-MetalKit
   - This requires code changes to use Metal Performance Shaders
   
2. USE MLX (Apple's ML framework):
   - Install: pip install mlx
   - MLX is designed for Apple Silicon and uses Metal directly
   
3. CHECK OPENCL AVAILABILITY:
   - Run: /System/Library/Frameworks/OpenCL.framework/Versions/A/Headers
   - Modern macOS may have limited OpenCL GPU support
   
4. NUMBA JIT FALLBACK (current behavior):
   - Your app falls back to Numba JIT on CPU
   - This is still reasonably fast with multi-threading

To properly leverage M3 GPU, the code would need to be refactored to use:
- Metal Performance Shaders (via pyobjc)
- MLX framework
- Or a Metal-native compute solution
""")
