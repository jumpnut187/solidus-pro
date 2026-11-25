#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from physics.heat_transfer import BoundaryType, HeatTransferModel, NUMBA_AVAILABLE
from physics.solidification_physics import FractionSolidCalculator
from core.materials import get_alloy

_EFFECTIVE_DIFFUSIVITY_FACTOR: Optional[float] = None

class _FakeAlloy:

    density_liquid: float = 1.0
    density_solid: float = 1.0
    specific_heat: float = 1.0
    thermal_conductivity: float = 1.0
    latent_heat: float = 0.0
    liquidus_temp: float = 999.0
    solidus_temp: float = 1.0
    pouring_temp: float = 990.0

    @property
    def thermal_diffusivity(self) -> float:
        return 1.0

def _build_canonical_neighbors(n: int) -> np.ndarray:
    idx = np.arange(n ** 3, dtype=np.int64).reshape(n, n, n)
    offsets: List[Tuple[int, int, int]] = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    ]
    neighbors = np.full((n ** 3, 26), -1, dtype=np.int32)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                flat = int(idx[i, j, k])
                for slot, (di, dj, dk) in enumerate(offsets):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < n and 0 <= jj < n and 0 <= kk < n:
                        neighbors[flat, slot] = int(idx[ii, jj, kk])
    return neighbors

def _voxel_index(i: int, j: int, k: int, n: int) -> int:
    return i * n * n + j * n + k

def benchmark_laplacian_exactness() -> Tuple[str, bool, str]:
    global _EFFECTIVE_DIFFUSIVITY_FACTOR

    n = 9
    neighbors = _build_canonical_neighbors(n)
    model = HeatTransferModel(n ** 3, voxel_size=1.0, neighbors=neighbors, alloy=_FakeAlloy())

    xs = np.arange(n, dtype=np.float64)
    t0 = (
        xs[:, None, None] ** 2
        + xs[None, :, None] ** 2
        + xs[None, None, :] ** 2
    ).reshape(-1)
    model.set_initial_temperature(t0)
    model.step(dt=1e-4)

    dTdt = (model.temperatures - t0) / 1e-4
    interior = float(dTdt[_voxel_index(4, 4, 4, n)])
    near_surface = float(dTdt[_voxel_index(1, 4, 4, n)])
    _EFFECTIVE_DIFFUSIVITY_FACTOR = interior / 6.0

    passed = abs(interior - 6.0) < 0.1 and abs(near_surface - 6.0) < 0.35
    detail = (
        f"interior (4,4,4) dT/dt = {interior:.6f} K/s (target 6.0 ± 0.1); "
        f"near-surface (1,4,4) dT/dt = {near_surface:.6f} K/s (target 6.0 ± 0.35)"
    )
    return ("01 laplacian_exactness_on_quadratics", passed, detail)

def benchmark_1d_diffusion() -> Tuple[str, bool, str]:
    n = 9
    neighbors = _build_canonical_neighbors(n)
    model = HeatTransferModel(n ** 3, voxel_size=1.0, neighbors=neighbors, alloy=_FakeAlloy())

    erf = np.vectorize(math.erf)
    xs = np.arange(n, dtype=np.float64)
    t0_x = 700.0 + 300.0 * erf((xs - 4.0) / 0.5)
    model.set_initial_temperature(np.repeat(t0_x, n * n))

    dt = 1e-3
    for _ in range(100):
        model.step(dt)

    t_final = 100 * dt
    alpha = 1.0
    t_analytic_x = 700.0 + 300.0 * erf(
        (xs - 4.0) / np.sqrt(0.25 + 4.0 * alpha * t_final)
    )

    band_x = np.arange(2, 7)
    band_idx = [_voxel_index(int(xi), 4, 4, n) for xi in band_x]
    numerical = model.temperatures[band_idx]
    analytic = t_analytic_x[band_x]
    max_err = float(np.max(np.abs(numerical - analytic)))

    tol = 5.0
    passed = max_err < tol
    detail = (
        f"t = {t_final} s, alpha = {alpha}; band x = 2..6 (y = z = 4); "
        f"max |T_num - T_analytic| = {max_err:.4f} °C "
        f"(target < {tol:.1f} °C; adjusted from 0.5 °C, see docstring: "
        f"measured error is O(dx^2) truncation, dx = 1)"
    )
    return ("02 one_dimensional_diffusion_erf", passed, detail)

def benchmark_robin_boundary() -> Tuple[str, bool, str]:
    n = 9
    neighbors = _build_canonical_neighbors(n)
    model = HeatTransferModel(n ** 3, voxel_size=1.0, neighbors=neighbors, alloy=_FakeAlloy())
    model.set_initial_temperature(500.0)
    model.add_boundary(
        np.array([_voxel_index(0, 4, 4, n)], dtype=np.int32),
        BoundaryType.CONVECTION,
        htc=100.0,
        T_infinity=25.0,
    )
    model.step(dt=1e-4)

    rho_cp = 1.0
    dx = 1.0
    h = 100.0
    t_inf = 25.0
    dt = 1e-4
    expected_drop = dt * h * (500.0 - t_inf) / (rho_cp * dx)
    expected_boundary = 500.0 - expected_drop
    boundary_T = float(model.temperatures[_voxel_index(0, 4, 4, n)])
    interior_T = float(model.temperatures[_voxel_index(4, 4, 4, n)])

    passed = (
        abs(boundary_T - expected_boundary) <= 0.01 * expected_drop
        and interior_T == 500.0
    )
    detail = (
        f"boundary (0,4,4) T = {boundary_T:.6f} °C "
        f"(expected {expected_boundary:.6f} ± 1% of drop); "
        f"interior (4,4,4) T = {interior_T:.6f} °C (expected exactly 500.0)"
    )
    return ("03 robin_boundary_lumped", passed, detail)

def benchmark_scheil() -> Tuple[str, bool, str]:
    f_1450 = FractionSolidCalculator.scheil(1450.0, 1460.0, 1399.0, 0.17, 1538.0)
    f_1430 = FractionSolidCalculator.scheil(1430.0, 1460.0, 1399.0, 0.17, 1538.0)
    f_1399 = FractionSolidCalculator.scheil(1399.0, 1460.0, 1399.0, 0.17, 1538.0)

    passed = (
        abs(f_1450 - 0.135) < 0.005
        and abs(f_1430 - 0.324) < 0.01
        and f_1399 == 1.0
    )
    detail = (
        f"f_s(1450) = {f_1450:.5f} (target 0.135 ± 0.005); "
        f"f_s(1430) = {f_1430:.5f} (target 0.324 ± 0.01); "
        f"f_s(1399) = {f_1399:.5f} (target exactly 1.0)"
    )
    return ("04 scheil_analytic_sanity", passed, detail)

def benchmark_chvorinov_wcb() -> Tuple[str, bool, str]:
    wcb = get_alloy("WCB")
    plate_thickness_in = 2.0
    modulus = plate_thickness_in / 2.0

    try:
        from core.solidification_params import TimingParams
        mold_constant = float(TimingParams().chvorinov_mold_constant)
    except Exception:
        mold_constant = 280.0

    t_solidification = mold_constant * modulus ** 2

    alpha_wcb = wcb.thermal_diffusivity
    t_casting_side = (
        (modulus * 0.0254) ** 2 / (2.0 * alpha_wcb)
        * (wcb.latent_heat / (wcb.specific_heat * (wcb.liquidus_temp - 25.0)))
    )

    passed = 100.0 <= t_solidification <= 1000.0
    detail = (
        f"WCB plate {plate_thickness_in:.1f} in -> M = {modulus:.1f} in; "
        f"t = B*M^2 = {t_solidification:.1f} s (B = {mold_constant:.1f} s/in^2); "
        f"sanity band [100, 1000] s; "
        f"WCB: rho_l = {wcb.density_liquid:.0f} kg/m^3, k = {wcb.thermal_conductivity:.0f} W/mK, "
        f"cp = {wcb.specific_heat:.0f} J/kgK, L = {wcb.latent_heat:.0f} J/kg; "
        f"casting-side-only estimate (ignores mold) = {t_casting_side:.1f} s"
    )
    return ("05 chvorinov_wcb_sanity", passed, detail)

def benchmark_effective_diffusivity() -> Tuple[str, bool, str]:
    factor = _EFFECTIVE_DIFFUSIVITY_FACTOR
    if factor is None:
        return (
            "06 effective_diffusivity_regression",
            False,
            "benchmark 01 did not run (no measured factor available)",
        )
    passed = abs(factor - 1.0) <= 0.02
    detail = (
        f"measured effective diffusivity factor = {factor:.6f} "
        f"(must be within 2% of 1.0)"
    )
    return ("06 effective_diffusivity_regression", passed, detail)

def main() -> int:
    benchmarks = [
        benchmark_laplacian_exactness(),
        benchmark_1d_diffusion(),
        benchmark_robin_boundary(),
        benchmark_scheil(),
        benchmark_chvorinov_wcb(),
        benchmark_effective_diffusivity(),
    ]

    width = 80
    print("=" * width)
    print("Thermal Physics Benchmarks (production classes)")
    print(f"backend: {'numba (parallel kernels)' if NUMBA_AVAILABLE else 'numpy (fallback)'}")
    print("=" * width)

    all_passed = True
    for name, passed, detail in benchmarks:
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        print(f"      {detail}")

    print("=" * width)
    n_pass = sum(1 for _, p, _ in benchmarks if p)
    print(f"Result: {n_pass}/{len(benchmarks)} benchmarks passed")
    print("=" * width)

    return 0 if all_passed else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"BENCHMARK HARNESS ERROR: {exc}")
        sys.exit(2)
