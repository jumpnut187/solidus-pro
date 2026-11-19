"""Prototype: inlet velocity driven by the head DIFFERENCE, not measured.

u_z = -Cd sqrt(2 cs^2 (rho_head - rho_throat) / rho0), rho_throat measured on
the layer below the mouth. Reduces to Torricelli at an open throat and to zero
at a throat backed up to the head. Reports delivery against the target.
"""
import sys, numpy as np
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
from physics.filling import (create_state, step, ensure_interface_skin,
                             FLAG_FLUID, FLAG_GAS, FLAG_INTERFACE, REFERENCE_DENSITY)
from physics.fluid_flow import lattice_units, equilibrium

MODE = sys.argv[1] if len(sys.argv) > 1 else "driven"
nx = ny = 4; nz = 40
dx = 0.0043
obstacle = np.ones((nx+2, ny+2, nz), dtype=bool)
obstacle[1:nx+1, 1:ny+1, :] = False
units = lattice_units(dx, 0.006/7200.0, u_max_physical_m_s=3.5, u_max_lattice=0.05)
inlet = np.zeros_like(obstacle); inlet[1:nx+1, 1:ny+1, nz-1] = True
state = create_state(obstacle, dx_m=dx, dt_s=units.dt_s, initial_fluid=inlet,
                     initial_temperature_c=1600.0)
ix, iy, iz = np.nonzero(inlet)
rho_in = REFERENCE_DENSITY * (1.0 + units.gravity_lattice * (0.1016/dx) * 3.0)
rho_arr = np.full(ix.size, rho_in)
halo = np.zeros_like(obstacle); halo[1:nx+1, 1:ny+1, nz-2] = True
tx, ty, tz = np.nonzero(halo)
CD = 0.8
u_target = CD * np.sqrt(2.0 * (1.0/3.0) * (rho_in - REFERENCE_DENSITY) / REFERENCE_DENSITY)
u_cap = 0.05
target_per_step = rho_in * min(u_target, u_cap) * ix.size
print(f"u_target={u_target:.5f} target/step={target_per_step:.5f}")
u_drive = 0.0
dprev = [rho_in - REFERENCE_DENSITY]
poured = 0.0; last = 0.0
for n in range(1, 12001):
    state.flags[ix, iy, iz] = FLAG_FLUID
    if np.any(state.flags[halo] == FLAG_GAS):
        ensure_interface_skin(state, halo)
    if MODE == "driven":
        wet_t = state.flags[tx, ty, tz] >= FLAG_INTERFACE
        rho_t = (float(np.asarray(state.rho[tx, ty, tz], np.float64)[wet_t].mean())
                 if wet_t.any() else REFERENCE_DENSITY)
        import os
        BR = float(os.environ.get("BR", "0.02"))
        dprev[0] += BR * ((rho_in - rho_t) - dprev[0])
        raw = CD * np.sqrt(2.0/3.0 * max(0.0, dprev[0]) / REFERENCE_DENSITY)
        u_drive = min(raw, u_cap)
        u = np.zeros((3, ix.size)); u[2, :] = -u_drive
    else:
        from physics.fluid_flow import LATTICE_VELOCITIES
        cT = LATTICE_VELOCITIES.T.astype(np.float64)
        f = state.f[:, ix, iy, iz].astype(np.float64)
        u = cT @ f / np.clip(f.sum(axis=0), 1e-9, None)
    state.f[:, ix, iy, iz] = equilibrium(rho_arr, u).astype(np.float32)
    poured += float(rho_in*ix.size) - float(np.sum(state.mass[ix, iy, iz]))
    state.mass[ix, iy, iz] = rho_in
    step(state, units, track_free_surface=True)
    if n % 1000 == 0:
        wet = state.flags >= 2
        f_ = np.asarray(state.f, np.float64)
        print(f"{n:6d} u_drive={u_drive:+.5f} delivery/step={(poured-last)/1000:.5f} "
              f"({(poured-last)/1000/target_per_step:5.1%} of target) "
              f"wet={int(wet.sum())}/{nx*ny*nz} fluid={int((state.flags==3).sum())} "
              f"mass={float(state.mass[wet].sum()):.1f} neg={int((f_[:,wet]<0).sum())}")
        last = poured
