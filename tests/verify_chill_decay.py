import sys
import os
import numpy as np

sys.path.append(os.getcwd())

from core.acceleration import (
    compute_chill_internal_thickness_parallel,
    compute_dynamic_chill_boost_parallel
)

def test_chill_thickness():
    print("Testing Chill Thickness (EDT)...")
    chill_centers = np.array([
        [-0.05, -0.05, -0.05], [0.05, -0.05, -0.05],
        [-0.05,  0.05, -0.05], [0.05,  0.05, -0.05],
        [-0.05, -0.05,  0.05], [0.05, -0.05,  0.05],
        [-0.05,  0.05,  0.05], [0.05,  0.05,  0.05]
    ], dtype=np.float32)

    chill_surface_centers = chill_centers.copy()

    thicknesses = compute_chill_internal_thickness_parallel(chill_centers, chill_surface_centers)
    print(f"  Thicknesses: {thicknesses}")
    assert np.all(thicknesses == 0.0)

    chill_centers = np.vstack([chill_centers, [0, 0, 0]])
    thicknesses = compute_chill_internal_thickness_parallel(chill_centers, chill_surface_centers)
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

    boost_t0 = compute_dynamic_chill_boost_parallel(
        voxel_centers, chill_centers, chill_thicknesses,
        voxel_states, part_types, 0.0, 3.0
    )
    print(f"  Boost at t=0: {boost_t0}")
    assert np.isclose(boost_t0[0], 2.0)
    assert np.isclose(boost_t0[1], 1.0)

    boost_t120 = compute_dynamic_chill_boost_parallel(
        voxel_centers, chill_centers, chill_thicknesses,
        voxel_states, part_types, 120.0, 3.0
    )
    print(f"  Boost at t=120: {boost_t120}")
    assert boost_t120[0] < boost_t0[0]

    voxel_states[0] = 0.4
    boost_sat = compute_dynamic_chill_boost_parallel(
        voxel_centers, chill_centers, chill_thicknesses,
        voxel_states, part_types, 0.0, 3.0
    )
    print(f"  Boost with 50% saturation: {boost_sat}")
    assert np.isclose(boost_sat[0], 1.5)

    print("✓ Dynamic Boost Test Passed")

if __name__ == "__main__":
    try:
        test_chill_thickness()
        test_dynamic_boost()
    except Exception as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
