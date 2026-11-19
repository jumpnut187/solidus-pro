"""Minimal NumPy-path witness for MECHANISM 1.

Fixture: the closed, all-FLUID box that
tests/test_filling_stability.py::FluidMassTracksDensityTests uses to assert
|mass - rho*porosity| stays at ~1e-4 forever, with ONE change -- a porosity
band laid across the middle.  There is no GAS and no INTERFACE cell anywhere,
so apply_transitions / heal_submerged_interface / _redistribute are all no-ops
by construction and the ONLY thing that touches state.mass is
physics.filling.advance_mass.

Variants
  plain     porosity 1.0 everywhere            -> the existing test's control
  por       porosity 0.85 in z-band, drag off  -> isolates the WEIGHT branch
  por_drag  porosity 0.85 + real Guo-Zhao drag -> what Test 2 actually runs
  por_fix   porosity 0.85, drag off, advance_mass patched to Koerner Table 1
            unconditionally (weight 1 whenever either cell is FLUID)
  por_fixd  porosity 0.85 + real drag + the same patch

Reports, per sample: total mass (conservation), and max over FLUID cells of
|mass - rho*porosity| / (rho*porosity) == |eps - 1|.

  ./venv/bin/python witness_porous_weight.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

import physics.filling as pf  # noqa: E402
from physics.filling import PorousZone, FLAG_FLUID, FLAG_INTERFACE, FLAG_GAS  # noqa: E402
from physics.fluid_flow import (  # noqa: E402
    lattice_units, lattice_tables, porous_drag_coefficients, LATTICE_VELOCITIES, Q,
    OPPOSITE,
)

NU = 0.006 / 7200.0
DX_M = 0.0152
NZ = int(os.environ.get("NZ", "14"))
W = int(os.environ.get("W", "3"))
BAND = (5, 7)
STEPS = int(os.environ.get("STEPS", "4000"))
SAMPLE = int(os.environ.get("SAMPLE", "500"))
POROSITY = 0.85


def units():
    return lattice_units(DX_M, NU, u_max_physical_m_s=2.8)


def build(porous_band: bool, drag: bool, seed_eps=None):
    n = W + 2
    shape = (n, n, NZ + 2)
    obstacle = np.zeros(shape, dtype=bool)
    obstacle[0] = obstacle[-1] = True
    obstacle[:, 0] = obstacle[:, -1] = True
    obstacle[:, :, 0] = obstacle[:, :, -1] = True
    u = units()
    zone = PorousZone.inert(shape)
    if porous_band:
        band = np.zeros(shape, dtype=bool)
        band[1:W + 1, 1:W + 1, BAND[0]:BAND[1] + 1] = True
        zone.porosity[band] = np.float32(POROSITY)
        if drag:
            # The same coefficients build_porous_zone stamps for a real
            # ceramic foam filter (10 ppi-ish): k1 and inv_k2 from the
            # Ergun/Forchheimer closure the stage uses.
            c0, c1 = porous_drag_coefficients(
                POROSITY, 1.0e-8, 1.0e4, nu_lattice=u.nu_lattice, dx_m=DX_M
            )
            zone.drag_c0[band] = np.float32(c0)
            zone.drag_c1[band] = np.float32(c1)
    state = pf.create_state(
        obstacle, dx_m=DX_M, dt_s=u.dt_s, initial_fluid=~obstacle, porous=zone
    )
    assert int((state.flags == FLAG_INTERFACE).sum()) == 0
    assert int((state.flags == FLAG_GAS).sum()) == 0
    if seed_eps is not None:
        # create_state seeds mass[fluid] = REFERENCE_DENSITY with NO porosity
        # factor, so a porous cell starts at eps = 1/porosity.  Re-seed it to
        # the requested eps so "does the drift self-start from eps == 1" can be
        # asked separately from "does it amplify a pre-existing error".
        full = state.rho.astype(np.float64) * state.porous.porosity_f64
        band = state.porous.porosity_f64 < 1.0
        state.mass[band] = full[band] * seed_eps
    return state, u


# ---------------------------------------------------------------------------
# the patched exchange: Koerner Table 1 unconditionally (no porous override)
# ---------------------------------------------------------------------------
def advance_mass_table1(state, f_post_collision, f_streamed):
    outgoing = np.asarray(f_post_collision, dtype=np.float64)
    incoming = np.asarray(f_streamed, dtype=np.float64)
    eps = np.clip(pf.fluid_fraction(state), 0.0, 1.0)
    fluid = state.flags == FLAG_FLUID
    wet = (state.flags == FLAG_INTERFACE) | fluid
    opp = OPPOSITE.astype(np.intp)
    axes = (0, 1, 2)
    delta = np.zeros(state.shape, dtype=np.float64)
    for i in range(1, Q):
        c = LATTICE_VELOCITIES[i]
        shift = (-int(c[0]), -int(c[1]), -int(c[2]))
        neighbour_wet = np.roll(wet, shift=shift, axis=axes)
        ibar = int(opp[i])
        weight = np.where(
            fluid | np.roll(fluid, shift=shift, axis=axes),
            1.0,
            0.5 * (eps + np.roll(eps, shift=shift, axis=axes)),
        )
        exchange = (incoming[ibar] - outgoing[i]) * weight
        delta += np.where(wet & neighbour_wet, exchange, 0.0)
    return state.mass + delta


def run(name, porous_band, drag, patch, seed_eps=None):
    orig = pf.advance_mass
    if patch:
        pf.advance_mass = advance_mass_table1
    try:
        state, u = build(porous_band, drag, seed_eps)
        full0 = state.rho.astype(np.float64) * state.porous.porosity_f64
        m0 = float(state.mass.sum())
        print(f"\n=== {name}: porosity_band={porous_band} drag={drag} "
              f"table1_patch={patch} seed_eps={seed_eps} ===", flush=True)
        print(f"    cells={int((state.flags==FLAG_FLUID).sum())} FLUID, "
              f"band z{BAND[0]}-{BAND[1]}, m0={m0:.10g}", flush=True)
        print(f"    {'step':>6} {'total_mass':>18} {'dM/M0':>11} "
              f"{'max|eps-1|':>12} {'max|eps-1| band':>16} "
              f"{'min eps':>9} {'max eps':>9}", flush=True)
        rows = []
        for s in range(1, STEPS + 1):
            pf.step(state, u, use_numba=False)
            if s % SAMPLE == 0 or s == 1:
                if not np.isfinite(state.mass).all():
                    print(f"    {s:>6}   NON-FINITE mass", flush=True)
                    break
                full = state.rho.astype(np.float64) * state.porous.porosity_f64
                fl = state.flags == FLAG_FLUID
                with np.errstate(divide="ignore", invalid="ignore"):
                    eps = np.where(full > 0, state.mass / full, 1.0)
                band = state.porous.porosity_f64 < 1.0
                e = np.abs(eps - 1.0)
                mt = float(state.mass.sum())
                rows.append((s, mt, float(e[fl].max()),
                             float(e[fl & band].max()) if (fl & band).any() else 0.0))
                print(f"    {s:>6} {mt:>18.10g} {(mt-m0)/abs(m0):>11.3e} "
                      f"{float(e[fl].max()):>12.4e} "
                      f"{(float(e[fl & band].max()) if (fl&band).any() else 0.0):>16.4e} "
                      f"{float(eps[fl].min()):>9.4f} {float(eps[fl].max()):>9.4f}",
                      flush=True)
        return rows
    finally:
        pf.advance_mass = orig


def main():
    res = {}
    only = os.environ.get("ONLY", "")
    plan = [
        ("plain",     dict(porous_band=False, drag=False, patch=False)),
        ("por",       dict(porous_band=True,  drag=False, patch=False)),
        ("por_fix",   dict(porous_band=True,  drag=False, patch=True)),
        ("por_drag",  dict(porous_band=True,  drag=True,  patch=False)),
        ("por_fixd",  dict(porous_band=True,  drag=True,  patch=True)),
        # eps seeded to exactly 1: does the drift START on its own?
        ("por_e1",    dict(porous_band=True,  drag=False, patch=False, seed_eps=1.0)),
        ("por_e1_d",  dict(porous_band=True,  drag=True,  patch=False, seed_eps=1.0)),
        ("por_e1_fix",dict(porous_band=True,  drag=False, patch=True,  seed_eps=1.0)),
        # eps seeded 5% high: the amplifier's gain on a small pre-existing error
        ("por_e105",  dict(porous_band=True,  drag=False, patch=False, seed_eps=1.05)),
        ("por_e105f", dict(porous_band=True,  drag=False, patch=True,  seed_eps=1.05)),
    ]
    for name, kw in plan:
        if only and name not in only.split(","):
            continue
        res[name] = run(name, **kw)

    print("\n=== SUMMARY: final max|eps-1| over FLUID cells ===", flush=True)
    for k, v in res.items():
        if v:
            s, mt, allm, bandm = v[-1]
            print(f"  {k:>9}  step {s:>5}  all {allm:.4e}  band {bandm:.4e}  "
                  f"total_mass {mt:.10g}", flush=True)
        else:
            print(f"  {k:>9}  (no samples)", flush=True)


if __name__ == "__main__":
    main()
