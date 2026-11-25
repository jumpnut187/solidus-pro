import sys, numpy as np
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
from physics.filling import (create_state, step, ensure_interface_skin,
                             FLAG_FLUID, FLAG_GAS, REFERENCE_DENSITY)
from physics.fluid_flow import lattice_units, equilibrium, LATTICE_VELOCITIES

nx = ny = 4; nz = 40
dx = 0.0043
obstacle = np.ones((nx+2, ny+2, nz), dtype=bool)
obstacle[1:nx+1, 1:ny+1, :] = False
units = lattice_units(dx, 0.006/7200.0, u_max_physical_m_s=3.5, u_max_lattice=0.05)
inlet = np.zeros_like(obstacle); inlet[1:nx+1, 1:ny+1, nz-1] = True
state = create_state(obstacle, dx_m=dx, dt_s=units.dt_s, initial_fluid=inlet,
                     initial_temperature_c=1600.0)
ix, iy, iz = np.nonzero(inlet)
head_l = 0.1016 / dx
rho_in = REFERENCE_DENSITY * (1.0 + units.gravity_lattice * head_l * 3.0)
rho_arr = np.full(ix.size, rho_in)
cT = LATTICE_VELOCITIES.T.astype(np.float64)
halo = np.zeros_like(obstacle); halo[1:nx+1, 1:ny+1, nz-2] = True
um = np.zeros((3, ix.size))
torr = np.sqrt(2.0 * (1.0/3.0) * (rho_in - 1.0))
print(f"g_lattice={units.gravity_lattice:.3e} rho_in={rho_in:.6f} Torricelli={torr:.5f} lattice")
for n in range(1, 6001):
    state.flags[ix, iy, iz] = FLAG_FLUID
    if np.any(state.flags[halo] == FLAG_GAS):
        ensure_interface_skin(state, halo)
    f = state.f[:, ix, iy, iz].astype(np.float64)
    rho_now = f.sum(axis=0)
    u_raw = cT @ f / np.clip(rho_now, 1e-9, None)
    um *= 0.8; um += 0.2 * u_raw
    state.f[:, ix, iy, iz] = equilibrium(rho_arr, um).astype(np.float32)
    state.mass[ix, iy, iz] = rho_in
    step(state, units, track_free_surface=True)
    if n % 1000 == 0:
        u = np.asarray(state.u, dtype=np.float64)
        wet = state.flags >= 2
        def uz(z):
            m = wet[1:nx+1, 1:ny+1, z]
            return float(u[2, 1:nx+1, 1:ny+1, z][m].mean()) if m.any() else float("nan")
        front = int(np.nonzero(wet.any(axis=(0, 1)))[0].min())
        print(f"{n:6d} u_z inlet={uz(nz-1):+.5f} -1={uz(nz-2):+.5f} "
              f"-5={uz(nz-6):+.5f} front_z={front} wet={int(wet.sum())}")
