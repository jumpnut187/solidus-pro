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
    "Casting": box(0.0, 4.0, 0.0, 3.0, 0.0, 3.0),
    "Riser": cylinder(3.0, 1.5, 3.0 - 0.25, 5.5, 1.0),
    "Gate": box(-1.25, 0.25, 1.0, 2.0, 0.0, 1.0),
    "Runner": box(-4.25, -1.0, 0.875, 2.125, 0.0, 1.25),
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
