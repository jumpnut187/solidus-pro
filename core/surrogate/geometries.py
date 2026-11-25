from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

MIN_MAX_DIM_IN = 2.0
MAX_MAX_DIM_IN = 24.0

JUNCTION_OVERLAP_FRACTION = 0.25

FAMILIES: List[str] = [
    "box",
    "cylinder",
    "plate_with_boss",
    "t_section",
    "l_bracket",
    "stepped_plate",
    "hollow_cylinder",
    "flanged_cylinder",
    "framed_box",
    "tapered_bar",
    "wedge",
]

@dataclass
class GeometrySpec:

    name: str
    family: str
    mesh: trimesh.Trimesh
    params: Dict[str, float] = field(default_factory=dict)
    min_thickness: float = 1.0
    parts: Optional[List[Tuple[trimesh.Trimesh, int]]] = None

    @property
    def max_dim(self) -> float:
        return float(np.max(self.mesh.extents))

    def as_pair(self):
        return self.name, self.mesh

def _round_params(params: Dict[str, float]) -> Dict[str, float]:
    return {k: round(float(v), 3) for k, v in params.items()}

def _spec_name(family: str, index: int, params: Dict[str, float]) -> str:
    h = hashlib.sha1()
    h.update(family.encode())
    for key in sorted(params):
        h.update(f"{key}={params[key]:.3f};".encode())
    return f"{family}_{index:03d}_{h.hexdigest()[:6]}"

def _box(extents, center=(0.0, 0.0, 0.0)) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=tuple(float(e) for e in extents))
    mesh.apply_translation(np.asarray(center, dtype=np.float64))
    return mesh

def _cylinder(radius: float, height: float, center=(0.0, 0.0, 0.0), sections: int = 48) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=float(radius), height=float(height), sections=sections)
    mesh.apply_translation(np.asarray(center, dtype=np.float64))
    return mesh

def _concat(parts: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    mesh = trimesh.util.concatenate(list(parts))
    mesh.process(validate=False)
    return mesh

def build_box(p: Dict[str, float]):
    return _box((p["length"], p["width"], p["height"])), min(p["length"], p["width"], p["height"])

def build_cylinder(p: Dict[str, float]):
    return _cylinder(p["radius"], p["height"]), min(2.0 * p["radius"], p["height"])

def build_plate_with_boss(p: Dict[str, float]):
    L, W, t = p["length"], p["width"], p["thickness"]
    r, h = p["boss_radius"], p["boss_height"]
    n = int(round(p["boss_count"]))
    overlap = JUNCTION_OVERLAP_FRACTION * min(t, h)
    plate = _box((L, W, t))
    parts = [plate]
    xs = np.linspace(-(L / 2 - r - 0.25 * L / n), (L / 2 - r - 0.25 * L / n), n) if n > 1 else np.array([0.0])
    for x in xs:
        parts.append(_cylinder(r, h, center=(float(x), 0.0, t / 2 - overlap + h / 2)))
    return _concat(parts), min(t, 2.0 * r)

def build_t_section(p: Dict[str, float]):
    L, Wf, tf = p["length"], p["flange_width"], p["flange_thickness"]
    tw, Hw = p["web_thickness"], p["web_height"]
    overlap = JUNCTION_OVERLAP_FRACTION * min(tf, tw)
    flange = _box((L, Wf, tf))
    web = _box((L, tw, Hw), center=(0.0, 0.0, tf / 2 - overlap + Hw / 2))
    return _concat([flange, web]), min(tf, tw)

def build_l_bracket(p: Dict[str, float]):
    L, a, b, ta, tb = p["length"], p["leg_a"], p["leg_b"], p["thickness_a"], p["thickness_b"]
    overlap = JUNCTION_OVERLAP_FRACTION * min(ta, tb)
    base = _box((L, a, ta))
    upright = _box((L, tb, b), center=(0.0, -(a / 2 - tb / 2), ta / 2 - overlap + b / 2))
    return _concat([base, upright]), min(ta, tb)

def build_stepped_plate(p: Dict[str, float]):
    W = p["width"]
    steps = int(round(p["step_count"]))
    lengths = [p[f"step_length_{i}"] for i in range(steps)]
    thicks = [p[f"step_thickness_{i}"] for i in range(steps)]
    overlap = JUNCTION_OVERLAP_FRACTION * min(thicks)
    parts = []
    x = -sum(lengths) / 2
    for i, (ln, th) in enumerate(zip(lengths, thicks)):
        lead = overlap if i > 0 else 0.0
        trail = overlap if i < steps - 1 else 0.0
        parts.append(_box((ln + lead + trail, W, th), center=(x + ln / 2 + (trail - lead) / 2, 0.0, th / 2)))
        x += ln
    return _concat(parts), min(thicks)

def build_hollow_cylinder(p: Dict[str, float]):
    mesh = trimesh.creation.annulus(r_min=p["inner_radius"], r_max=p["outer_radius"], height=p["height"], sections=48)
    return mesh, min(p["outer_radius"] - p["inner_radius"], p["height"])

def build_flanged_cylinder(p: Dict[str, float]):
    r, h = p["radius"], p["height"]
    rf, tf = p["flange_radius"], p["flange_thickness"]
    overlap = JUNCTION_OVERLAP_FRACTION * min(tf, h)
    flange = _cylinder(rf, tf, center=(0.0, 0.0, tf / 2))
    body = _cylinder(r, h, center=(0.0, 0.0, tf - overlap + h / 2))
    return _concat([flange, body]), min(tf, 2.0 * r)

def build_framed_box(p: Dict[str, float]):
    L, W, H = p["length"], p["width"], p["height"]
    hl, hw = p["hole_length"], p["hole_width"]
    wall_y = (W - hw) / 2
    wall_x = (L - hl) / 2
    overlap = JUNCTION_OVERLAP_FRACTION * min(wall_x, wall_y)
    parts = [
        _box((L, wall_y, H), center=(0.0, W / 2 - wall_y / 2, 0.0)),
        _box((L, wall_y, H), center=(0.0, -(W / 2 - wall_y / 2), 0.0)),
        _box((wall_x, hw + 2 * overlap, H), center=(L / 2 - wall_x / 2, 0.0, 0.0)),
        _box((wall_x, hw + 2 * overlap, H), center=(-(L / 2 - wall_x / 2), 0.0, 0.0)),
    ]
    return _concat(parts), min(wall_x, wall_y, H)

def build_tapered_bar(p: Dict[str, float]):
    L = p["length"]
    a0, b0, a1, b1 = p["base_width"], p["base_height"], p["tip_width"], p["tip_height"]
    pts = np.array(
        [
            [-L / 2, -a0 / 2, -b0 / 2], [-L / 2, a0 / 2, -b0 / 2],
            [-L / 2, a0 / 2, b0 / 2], [-L / 2, -a0 / 2, b0 / 2],
            [L / 2, -a1 / 2, -b1 / 2], [L / 2, a1 / 2, -b1 / 2],
            [L / 2, a1 / 2, b1 / 2], [L / 2, -a1 / 2, b1 / 2],
        ]
    )
    return trimesh.convex.convex_hull(pts), min(a1, b1)

def build_wedge(p: Dict[str, float]):
    L, W, H = p["length"], p["width"], p["height"]
    tip = p["tip_thickness"]
    pts = np.array(
        [
            [-L / 2, -W / 2, 0.0], [-L / 2, W / 2, 0.0], [-L / 2, W / 2, H], [-L / 2, -W / 2, H],
            [L / 2, -W / 2, 0.0], [L / 2, W / 2, 0.0], [L / 2, W / 2, tip], [L / 2, -W / 2, tip],
        ]
    )
    return trimesh.convex.convex_hull(pts), tip

BUILDERS = {
    "box": build_box,
    "cylinder": build_cylinder,
    "plate_with_boss": build_plate_with_boss,
    "t_section": build_t_section,
    "l_bracket": build_l_bracket,
    "stepped_plate": build_stepped_plate,
    "hollow_cylinder": build_hollow_cylinder,
    "flanged_cylinder": build_flanged_cylinder,
    "framed_box": build_framed_box,
    "tapered_bar": build_tapered_bar,
    "wedge": build_wedge,
}

def _sample_max_dim(rng: np.random.Generator) -> float:
    return float(np.exp(rng.uniform(np.log(MIN_MAX_DIM_IN), np.log(MAX_MAX_DIM_IN))))

def sample_params(family: str, rng: np.random.Generator) -> Dict[str, float]:
    D = _sample_max_dim(rng)
    u = rng.uniform
    if family == "box":
        return {"length": D, "width": D * u(0.3, 1.0), "height": max(0.25, D * u(0.1, 1.0))}
    if family == "cylinder":
        if u() < 0.5:
            return {"height": D, "radius": max(0.25, D * u(0.08, 0.4))}
        return {"radius": D / 2, "height": max(0.25, D * u(0.1, 0.8))}
    if family == "plate_with_boss":
        t = max(0.3, D * u(0.04, 0.12))
        W = D * u(0.4, 1.0)
        n = int(rng.integers(1, 4))
        r_max = min(W / 2, D / (2 * n)) * 0.7
        r = max(0.3, r_max * u(0.4, 1.0))
        return {
            "length": D, "width": W, "thickness": t,
            "boss_radius": r, "boss_height": t * u(1.0, 3.0), "boss_count": float(n),
        }
    if family == "t_section":
        return {
            "length": D,
            "flange_width": D * u(0.3, 0.8),
            "flange_thickness": max(0.3, D * u(0.04, 0.12)),
            "web_thickness": max(0.3, D * u(0.04, 0.12)),
            "web_height": D * u(0.2, 0.6),
        }
    if family == "l_bracket":
        return {
            "length": D,
            "leg_a": D * u(0.3, 0.8),
            "leg_b": D * u(0.3, 0.8),
            "thickness_a": max(0.3, D * u(0.04, 0.12)),
            "thickness_b": max(0.3, D * u(0.04, 0.12)),
        }
    if family == "stepped_plate":
        steps = int(rng.integers(2, 4))
        fractions = rng.dirichlet(np.ones(steps) * 2.0)
        params: Dict[str, float] = {"width": D * u(0.3, 0.8), "step_count": float(steps)}
        thick_max = D * u(0.15, 0.4)
        for i in range(steps):
            params[f"step_length_{i}"] = max(0.5, D * fractions[i])
            params[f"step_thickness_{i}"] = max(0.3, thick_max * u(0.2, 1.0))
        return params
    if family == "hollow_cylinder":
        tall = u() < 0.5
        r_out = max(0.75, D * u(0.1, 0.4)) if tall else D / 2
        h = D if tall else max(0.5, D * u(0.15, 0.8))
        wall = max(0.25, r_out * u(0.15, 0.6))
        return {"outer_radius": r_out, "inner_radius": r_out - wall, "height": h}
    if family == "flanged_cylinder":
        tall = u() < 0.5
        if tall:
            rf = max(0.75, D * u(0.1, 0.4))
            h_total = D
        else:
            rf = D / 2
            h_total = max(0.75, D * u(0.2, 0.9))
        tf = max(0.3, h_total * u(0.08, 0.3))
        return {
            "flange_radius": rf, "flange_thickness": tf,
            "radius": max(0.3, rf * u(0.3, 0.7)), "height": h_total - tf,
        }
    if family == "framed_box":
        W = D * u(0.4, 1.0)
        H = max(0.3, D * u(0.08, 0.5))
        wall = max(0.3, min(D, W) * u(0.1, 0.3))
        return {
            "length": D, "width": W, "height": H,
            "hole_length": D - 2 * wall, "hole_width": W - 2 * wall,
        }
    if family == "tapered_bar":
        a0 = max(0.5, D * u(0.1, 0.4))
        b0 = max(0.5, D * u(0.1, 0.4))
        return {
            "length": D, "base_width": a0, "base_height": b0,
            "tip_width": max(0.25, a0 * u(0.2, 0.7)), "tip_height": max(0.25, b0 * u(0.2, 0.7)),
        }
    if family == "wedge":
        H = max(0.5, D * u(0.1, 0.5))
        return {"length": D, "width": D * u(0.2, 0.8), "height": H, "tip_thickness": max(0.25, H * u(0.1, 0.4))}
    raise ValueError(f"Unknown geometry family: {family}")

def make_spec(family: str, params: Dict[str, float], index: int = 0) -> GeometrySpec:
    params = _round_params(params)
    mesh, min_thickness = BUILDERS[family](params)
    return GeometrySpec(
        name=_spec_name(family, index, params),
        family=family,
        mesh=mesh,
        params=params,
        min_thickness=float(min_thickness),
    )

def iter_geometries(
    count: int,
    seed: int = 0,
    families: Sequence[str] = tuple(FAMILIES),
) -> Iterator[GeometrySpec]:
    families = list(families)
    unknown = [f for f in families if f not in BUILDERS]
    if unknown:
        raise ValueError(f"Unknown geometry families: {unknown}")
    rng = np.random.default_rng(seed)
    for index in range(int(count)):
        family = families[index % len(families)]
        params = sample_params(family, rng)
        yield make_spec(family, params, index=index)

def generate_geometries(count: int, seed: int = 0, families: Sequence[str] = tuple(FAMILIES)) -> List[GeometrySpec]:
    return list(iter_geometries(count, seed=seed, families=families))

def named_catalog() -> Dict[str, GeometrySpec]:
    fixed: Dict[str, Any] = {
        "box": {"length": 4.0, "width": 4.0, "height": 4.0},
        "cylinder": {"radius": 1.5, "height": 4.0},
        "plate_with_boss": {
            "length": 6.0, "width": 4.0, "thickness": 1.0,
            "boss_radius": 1.0, "boss_height": 2.0, "boss_count": 1.0,
        },
        "t_section": {
            "length": 6.0, "flange_width": 4.0, "flange_thickness": 0.75,
            "web_thickness": 0.75, "web_height": 3.0,
        },
        "l_bracket": {"length": 6.0, "leg_a": 4.0, "leg_b": 3.0, "thickness_a": 0.75, "thickness_b": 0.75},
        "stepped_plate": {
            "width": 3.0, "step_count": 3.0,
            "step_length_0": 2.0, "step_thickness_0": 0.5,
            "step_length_1": 2.0, "step_thickness_1": 1.0,
            "step_length_2": 2.0, "step_thickness_2": 2.0,
        },
        "hollow_cylinder": {"outer_radius": 2.0, "inner_radius": 1.25, "height": 4.0},
        "flanged_cylinder": {"flange_radius": 2.5, "flange_thickness": 0.75, "radius": 1.25, "height": 4.0},
        "framed_box": {"length": 6.0, "width": 5.0, "height": 1.5, "hole_length": 3.0, "hole_width": 2.0},
        "tapered_bar": {"length": 8.0, "base_width": 2.0, "base_height": 2.0, "tip_width": 0.75, "tip_height": 0.75},
        "wedge": {"length": 6.0, "width": 3.0, "height": 2.0, "tip_thickness": 0.375},
    }
    catalog: Dict[str, GeometrySpec] = {}
    for index, (family, params) in enumerate(fixed.items()):
        spec = make_spec(family, params, index=index)
        catalog[family] = spec
    return catalog

def voxel_size_for(
    spec: GeometrySpec,
    grid: int = 64,
    target_voxels: int = 20_000,
    max_voxels: int = 50_000,
    dense: bool = False,
) -> float:
    volume = max(float(spec.mesh.volume), 1e-6)
    cell = spec.max_dim / float(grid)
    if dense:
        s = min(cell, spec.min_thickness / 3.0)
    else:
        s = max(cell, (volume / float(target_voxels)) ** (1.0 / 3.0))
        s = min(s, spec.min_thickness / 3.0)
        s = max(s, (volume / float(max_voxels)) ** (1.0 / 3.0))
    return float(max(0.05, round(s / 0.005) * 0.005))
