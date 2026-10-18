"""Does the wrap seed at the far face self-heal, or does it run away?

Emulates Test 2: a tall channel walled in x/y, with NO obstacle plane at z=0
(the casting's floor is flush with the grid's bounding box) and NO lid at
z=nz-1 (the riser's open top). The bottom two planes are pinned FLUID every
step, standing in for the continuously-fed cast bottom. Everything above z~4
is therefore spurious.
"""
import sys, os
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from physics.filling import (create_state, step, ensure_interface_skin,
                             REFERENCE_DENSITY, FLAG_GAS, FLAG_OBSTACLE,
                             FLAG_INTERFACE, FLAG_FLUID)
from physics.fluid_flow import lattice_units, LATTICE_VELOCITIES

DX_M, NU, NZ, W = 0.0043, 6.5e-7, 48, 3
NSTEPS = int(os.environ.get("NSTEPS", "4000"))


def go(label, lid, floor):
    n = W + 2
    obstacle = np.ones((n, n, NZ), dtype=bool)
    obstacle[1:1+W, 1:1+W, :] = False
    if floor: obstacle[:, :, 0] = True
    if lid:   obstacle[:, :, NZ-1] = True
    zlo = 1 if floor else 0
    pin = np.zeros_like(obstacle)
    pin[1:1+W, 1:1+W, zlo:zlo+2] = True
    units = lattice_units(DX_M, NU, u_max_physical_m_s=3.0)
    st = create_state(obstacle, dx_m=DX_M, dt_s=units.dt_s, initial_fluid=pin)
    halo = np.zeros_like(obstacle)
    for c in LATTICE_VELOCITIES[1:]:
        halo |= np.roll(pin, shift=(int(c[0]), int(c[1]), int(c[2])), axis=(0,1,2))
    halo &= ~pin & ~obstacle
    top = np.zeros_like(obstacle); top[:, :, NZ-1] = True; top &= ~obstacle
    openc = ~obstacle
    print(f"\n### {label}  (lid={lid} floor={floor})  channel z {zlo}..{NZ-1}, "
          f"pinned FLUID at z {zlo}-{zlo+1}")
    for s in range(1, NSTEPS+1):
        step(st, units, use_numba=True)
        # inlet-style pin, as core/filling_stage does
        st.flags[pin] = FLAG_FLUID
        st.mass[pin] = REFERENCE_DENSITY
        ensure_interface_skin(st, halo)
        if s % 500 == 0 or s == 1 or s == 5:
            f = np.asarray(st.f, dtype=np.float64)
            wet = (st.flags == FLAG_FLUID) | (st.flags == FLAG_INTERFACE)
            wz = np.nonzero(wet.any(axis=(0,1)))[0]
            eps = st.mass / np.maximum(np.asarray(st.rho, np.float64), 1e-30)
            neg = (np.any(f < 0.0, axis=0) & wet)
            fin = bool(np.isfinite(st.mass).all() and np.isfinite(f).all())
            # the spurious cluster = wet cells hanging off the TOP face
            spur = wet.copy(); spur[:, :, :NZ//2] = False
            szs = np.nonzero(spur.any(axis=(0,1)))[0]
            print(f"  step {s:5d} wet={int(wet.sum()):4d} z={wz.min()}-{wz.max()} "
                  f"| TOP cluster: {int(spur.sum()):4d} cells, reaching down to "
                  f"z={szs.min() if szs.size else '-'} "
                  f"| neg-pop cells={int(neg.sum())} epsmax={eps[wet].max():.3f} "
                  f"finite={fin}")
            if not fin:
                print("  --> NON-FINITE, stopping"); return
    return


go("OPEN both z faces (Test 2's condition)", lid=False, floor=False)
go("CONTROL: obstacle floor + lid",          lid=True,  floor=True)
go("CONTROL: floor only, top still open",    lid=False, floor=True)
