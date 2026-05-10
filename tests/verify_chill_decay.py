import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from core.acceleration import (
    compute_chill_internal_thickness_parallel,
    compute_dynamic_chill_boost_parallel
)

def test_chill_thickness():
    print("Testing Chill Thickness (EDT)...")
    # 2x2x2 block of chill voxels at origin (0.1" size)
    # Centers would be +/- 0.05 in each axis
    chill_centers = np.array([
        [-0.05, -0.05, -0.05], [0.05, -0.05, -0.05],
        [-0.05,  0.05, -0.05], [0.05,  0.05, -0.05],
        [-0.05, -0.05,  0.05], [0.05, -0.05,  0.05],
        [-0.05,  0.05,  0.05], [0.05,  0.05,  0.05]
    ], dtype=np.float32)
    
    # All are surface voxels since it's only 2x2x2
    chill_surface_centers = chill_centers.copy()
    
    thicknesses = compute_chill_internal_thickness_parallel(chill_centers, chill_surface_centers)
    print(f"  Thicknesses: {thicknesses}")
    # Distance to self surface is 0, so thickness should be 0 for surface voxels
    assert np.all(thicknesses == 0.0)
    
    # Add an internal voxel
    chill_centers = np.vstack([chill_centers, [0, 0, 0]])
    thicknesses = compute_chill_internal_thickness_parallel(chill_centers, chill_surface_centers)
    # Distance from (0,0,0) to nearest (+/-0.05, +/-0.05, +/-0.05) is sqrt(0.05^2 * 3)
    # Actually nearest surface center might be just one axis if we have a larger grid, 
    # but here it's 0.05 * sqrt(3) ~= 0.0866. Thickness = 2 * 0.0866 = 0.1732.
    print(f"  Internal thickness at (0,0,0): {thicknesses[-1]:.4f}")
    assert thicknesses[-1] > 0
    print("✓ Chill Thickness Test Passed")

def test_dynamic_boost():
    print("Testing Dynamic Boost...")
    voxel_centers = np.array([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    chill_centers = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    chill_thicknesses = np.array([1.0], dtype=np.float32)
    voxel_states = np.zeros(2, dtype=np.float32)
    part_types = np.zeros(2, dtype=np.int32)
    
    # t=0, no solidification
    boost_t0 = compute_dynamic_chill_boost_parallel(
        voxel_centers, chill_centers, chill_thicknesses,
        voxel_states, part_types, 0.0, 3.0
    )
    print(f"  Boost at t=0: {boost_t0}")
    # Voxel 1 (dist 0.5) is within 1.0" radius. Boost should be 1.0 + (3.0-1.0)*(1 - 0.5/1.0) = 2.0
    # Voxel 2 (dist 2.0) is outside. Boost should be 1.0
    assert np.isclose(boost_t0[0], 2.0)
    assert np.isclose(boost_t0[1], 1.0)
    
    # t=120s (Air Gap Decay, tau=120)
    boost_t120 = compute_dynamic_chill_boost_parallel(
        voxel_centers, chill_centers, chill_thicknesses,
        voxel_states, part_types, 120.0, 3.0
    )
    print(f"  Boost at t=120: {boost_t120}")
    # exp(-120/120) = 0.3678. effective_max = 1.0 + (3.0-1.0)*0.3678 = 1.7356
    # Boost at 0.5" = 1.0 + (1.7356-1.0)*0.5 = 1.3678
    assert boost_t120[0] < boost_t0[0]
    
    # Solidification Saturation
    voxel_states[0] = 0.4 # 50% through threshold (0.8)
    boost_sat = compute_dynamic_chill_boost_parallel(
        voxel_centers, chill_centers, chill_thicknesses,
        voxel_states, part_types, 0.0, 3.0
    )
    print(f"  Boost with 50% saturation: {boost_sat}")
    # sat_factor = (1 - 0.4/0.8) = 0.5
    # effective_max = 1.0 + (3.0-1.0)*0.5 = 2.0
    # Boost at 0.5" = 1.0 + (2.0-1.0)*0.5 = 1.5
    assert np.isclose(boost_sat[0], 1.5)
    
    print("✓ Dynamic Boost Test Passed")

if __name__ == "__main__":
    try:
        test_chill_thickness()
        test_dynamic_boost()
    except Exception as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
