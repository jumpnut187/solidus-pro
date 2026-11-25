import sys, os
sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from core.filling_stage import (_obstacle_mask, _metal_mask,
                                _domain_boundary_mask, _static_vent_mask,
                                EMPTY_CELL)
from core.part_manager import PART_TYPE_CAST, PART_TYPE_RISER, PART_TYPE_CORE
from physics.filling import (create_state, step, ensure_interface_skin,
                             REFERENCE_DENSITY, FLAG_GAS, FLAG_OBSTACLE,
                             FLAG_INTERFACE, FLAG_FLUID)
from physics.fluid_flow import lattice_units, LATTICE_VELOCITIES

NX = NY = 12
NZ = 24
CASTTOP = 8
NSTEPS = int(os.environ.get("NSTEPS", "1200"))

def types_grid():
    t = np.full((NX, NY, NZ), PART_TYPE_CORE, dtype=np.int8)
    t[1:NX-1, 1:NY-1, 0:CASTTOP+1] = PART_TYPE_CAST
    t[4:8, 4:8, CASTTOP:NZ] = PART_TYPE_RISER
    return t

def run(variant, use_numba):
    t = types_grid()
    pad = 0
    if variant == "PADDED":
        t = np.pad(t, 1, mode="constant", constant_values=EMPTY_CELL); pad = 1
    obstacle = _obstacle_mask(t)
    if variant == "STOPGAP":
        obstacle = obstacle | _domain_boundary_mask(t.shape)
    riser = (t == PART_TYPE_RISER) & ~obstacle
    nz = t.shape[2]
    zi = np.arange(nz)[None, None, :]
    upper = riser & (zi >= CASTTOP + pad + 6)
    pin = np.zeros(t.shape, bool)
    pin[(t == PART_TYPE_CAST) & ~obstacle & (zi <= pad + 1)] = True
    units = lattice_units(0.0043, 6.5e-7, u_max_physical_m_s=3.0)
    st = create_state(obstacle, dx_m=0.0043, dt_s=units.dt_s, initial_fluid=pin)
    halo = np.zeros(t.shape, bool)
    for c in LATTICE_VELOCITIES[1:]:
        halo |= np.roll(pin, shift=(int(c[0]), int(c[1]), int(c[2])), axis=(0,1,2))
    halo &= ~pin & ~obstacle
    first = None; peak = 0
    for s in range(1, NSTEPS + 1):
        step(st, units, use_numba=use_numba)
        st.flags[pin] = FLAG_FLUID; st.mass[pin] = REFERENCE_DENSITY
        ensure_interface_skin(st, halo)
        wet = (st.flags == FLAG_FLUID) | (st.flags == FLAG_INTERFACE)
        n = int((wet & upper).sum())
        peak = max(peak, n)
        if first is None and n:
            first = s
    wet = (st.flags == FLAG_FLUID) | (st.flags == FLAG_INTERFACE)
    wz = np.nonzero(wet.any(axis=(0,1)))[0]
    metal = _metal_mask(t)
    vent = int((_static_vent_mask(t, np.zeros((0,3), np.int64), True) & riser).sum())
    tag = "numba" if use_numba else "numpy"
    print(f"  {variant:8s} {tag}: upper-riser wet cells first={first} peak={peak} final={int((wet&upper).sum())} "
          f"| wet z={wz.min()}-{wz.max()} | metal cells={int(metal.sum())} "
          f"| riser-top vents detected={vent} | finite={bool(np.isfinite(st.mass).all())}")
    return first, peak

t = types_grid()
o = _obstacle_mask(t)
print(f"grid {t.shape}: CAST flush with z=0, RISER flush with z={NZ-1}; "
      f"metal cells {int(_metal_mask(t).sum())}")
print(f"non-obstacle cells on the six faces (today): "
      f"z=0 {int((~o[:,:,0]).sum())}, z={NZ-1} {int((~o[:,:,NZ-1]).sum())}, "
      f"x/y faces {int((~o[0]).sum()+(~o[-1]).sum()+(~o[:,0]).sum()+(~o[:,-1]).sum())}")
print(f"\n{NSTEPS} steps each, metal pinned at the cast floor only:")
for v in ("BROKEN", "PADDED", "STOPGAP"):
    for nb in (False, True):
        run(v, nb)
