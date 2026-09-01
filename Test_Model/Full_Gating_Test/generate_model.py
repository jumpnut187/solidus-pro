"""Generate the Full_Gating_Test model: a complete, physically fillable
bottom-gated rigging built for validating the mold-filling stage end to end.

Design constraints, chosen deliberately:

- Every flow channel is at least 1.0 in across, so at the 0.20 in fill grid
  this model targets (voxel_size=0.20, fill_grid_coarsening=1) the narrowest
  section is ~5 cells -- three interior nodes after halfway bounce-back takes
  the two wall layers. The Filling_Test model's 0.5 in gate has NO interior
  at a 0.40 in grid, which makes wall friction the entire flow; this model is
  the control case where resolution is known to be sufficient.

- Bottom-gated: sprue drops to a horizontal runner, the gate enters the
  casting at its base, and metal must CLIMB the cavity and the riser under
  metallostatic head alone. This exercises exactly the physics a
  prescribed-velocity inlet cannot represent.

- The sprue top (z = 8.0 in) is above the riser top (z = 5.5 in), so with any
  positive basin head the hydrostatic equilibrium surface clears every metal
  cell and a complete fill is physically possible. A model that violates this
  cannot finish no matter what the solver does.

- Adjacent parts overlap by 0.25 in so voxelization can never disconnect the
  flow path by rounding.

Coordinates in inches, Z up. Run with the project venv:
    ./venv/bin/python Test_Model/Full_Gating_Test/generate_model.py
"""

import numpy as np
import trimesh


def box(x0, x1, y0, y1, z0, z1):
    b = trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0))
    b.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    return b


def cylinder(cx, cy, z0, z1, radius):
    c = trimesh.creation.cylinder(radius=radius, height=z1 - z0, sections=48)
    c.apply_translation((cx, cy, (z0 + z1) / 2))
    return c


PARTS = {
    # 4 x 3 x 3 in cavity sitting on the parting line at z=0.
    "Casting": box(0.0, 4.0, 0.0, 3.0, 0.0, 3.0),
    # d=2 in riser on top of the casting, far end from the gate.
    "Riser": cylinder(3.0, 1.5, 3.0 - 0.25, 5.5, 1.0),
    # 1 x 1 in ingate entering the casting base; overlaps casting 0.25 in.
    "Gate": box(-1.25, 0.25, 1.0, 2.0, 0.0, 1.0),
    # 1.25 x 1.25 in runner from below the sprue to the gate.
    "Runner": box(-4.25, -1.0, 0.875, 2.125, 0.0, 1.25),
    # d=1.25 in vertical down sprue; bottom penetrates the runner.
    "Down Sprue": cylinder(-3.5, 1.5, 0.5, 8.0, 0.625),
}


def main() -> None:
    import os

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for name, mesh in PARTS.items():
        assert mesh.is_watertight, f"{name} is not watertight"
        path = os.path.join(out_dir, f"Full Gating Test - {name}.stl")
        mesh.export(path)
        print(f"{path}: volume {mesh.volume:.2f} in^3, {len(mesh.faces)} faces")
    total = sum(m.volume for m in PARTS.values())
    print(f"total (with overlaps double-counted): {total:.2f} in^3")


if __name__ == "__main__":
    main()
