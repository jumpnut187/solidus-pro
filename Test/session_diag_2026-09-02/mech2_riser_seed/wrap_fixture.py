import sys, os
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from physics.filling import (create_state, step, FLAG_GAS, FLAG_OBSTACLE,
                             FLAG_INTERFACE, FLAG_FLUID)
from physics.fluid_flow import lattice_units

FN = {0: "OBST", 1: "GAS", 2: "IFACE", 3: "FLUID"}
DX_M = 0.004
NU = 6.5e-7

def build(nz=14, width=3, lid=False, floor=False):
    n = width + 2
    obstacle = np.ones((n, n, nz), dtype=bool)
    obstacle[1:1 + width, 1:1 + width, :] = False
    if floor:
        obstacle[:, :, 0] = True
    if lid:
        obstacle[:, :, nz - 1] = True
    fluid = np.zeros_like(obstacle)
    zlo = 1 if floor else 0
    fluid[1:1 + width, 1:1 + width, zlo:zlo + 2] = True
    return obstacle, fluid

def run(label, use_numba, lid, floor, nsteps=40, nz=14):
    obstacle, fluid = build(nz=nz, lid=lid, floor=floor)
    units = lattice_units(DX_M, NU, u_max_physical_m_s=3.0)
    st = create_state(obstacle, dx_m=DX_M, dt_s=units.dt_s,
                      initial_fluid=fluid)
    open_cells = ~obstacle
    top = np.zeros_like(obstacle); top[:, :, nz - 1] = True
    top &= open_cells
    seeded_z = np.nonzero(fluid.any(axis=(0, 1)))[0]
    first = None
    for s in range(1, nsteps + 1):
        step(st, units, use_numba=use_numba)
        wet = (st.flags == FLAG_FLUID) | (st.flags == FLAG_INTERFACE)
        if first is None and (wet & top).any():
            xs, ys, zs = np.nonzero(wet & top)
            f = np.asarray(st.f, dtype=np.float64)
            first = (s, int(xs[0]), int(ys[0]), int(zs[0]))
            print(f"  !! {label}: TOP PLANE WET at step {s}: "
                  f"{int((wet & top).sum())} cells, first ({xs[0]},{ys[0]},{zs[0]}) "
                  f"flag={FN[int(st.flags[xs[0],ys[0],zs[0]])]} "
                  f"mass={float(st.mass[xs[0],ys[0],zs[0]]):.4g} "
                  f"rho={float(st.rho[xs[0],ys[0],zs[0]]):.4g} "
                  f"|f|={float(np.abs(f[:,xs[0],ys[0],zs[0]]).sum()):.4g}")
    wet = (st.flags == FLAG_FLUID) | (st.flags == FLAG_INTERFACE)
    wz = np.nonzero(wet.any(axis=(0, 1)))[0]
    tot = float(st.mass[open_cells].sum())
    print(f"  {label}: after {nsteps} steps  wet={int(wet.sum())} "
          f"wet-z={wz.min()}-{wz.max()} (seeded z {seeded_z.min()}-{seeded_z.max()}) "
          f"top-plane wet={int((wet & top).sum())} mass={tot:.5g} "
          f"finite={bool(np.isfinite(st.mass).all())} "
          f"first_top_wet_step={first[0] if first else None}")
    return first

print("=== A. OPEN z faces (Test 2's real condition: no floor, no lid) ===")
a_np = run("NumPy   open", False, lid=False, floor=False)
a_nb = run("Numba   open", True,  lid=False, floor=False)
print("=== B. CONTROL: obstacle floor + obstacle lid (properly closed) ===")
b_np = run("NumPy closed", False, lid=True, floor=True)
b_nb = run("Numba closed", True,  lid=True, floor=True)
print("=== C. floor only, top still open (riser-top-only case) ===")
c_np = run("NumPy  floor", False, lid=False, floor=True)
c_nb = run("Numba  floor", True,  lid=False, floor=True)
print()
print("VERDICT open-both:  numpy", a_np, " numba", a_nb)
print("VERDICT closed:     numpy", b_np, " numba", b_nb)
print("VERDICT floor-only: numpy", c_np, " numba", c_nb)
