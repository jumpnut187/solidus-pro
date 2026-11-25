from __future__ import annotations

from functools import partial
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

from core.part_manager import PART_TYPE_CHILL
from core.surrogate import design_rules
from core.surrogate.geometries import (
    JUNCTION_OVERLAP_FRACTION,
    GeometrySpec,
    _box,
    _concat,
    _cylinder,
    _round_params,
    _spec_name,
    make_spec,
    sample_params,
)

MIN_MAX_DIM_IN = 3.0
MAX_MAX_DIM_IN = 14.0
COMPLEX_MIN_DIM_IN = 12.0
COMPLEX_MAX_DIM_IN = 36.0
HEAVY_MIN_DIM_IN = 18.0
HEAVY_MAX_DIM_IN = 48.0
HEAVY_SECTION_IN = (6.0, 14.0)
MIN_WALL_IN = 0.375
MIN_CORE_DIAMETER_IN = design_rules.MIN_CORE_DIAMETER_IN
MAX_DESIGN_ATTEMPTS = 25

CORED_FAMILIES: List[str] = ["cored_box", "cored_cylinder", "cored_flanged_cylinder"]
RIGGED_FAMILIES: List[str] = ["plate_with_boss", "t_section", "l_bracket", "stepped_plate", "box", "wedge"]
ASSEMBLY_FAMILIES: List[str] = CORED_FAMILIES + RIGGED_FAMILIES
COMPLEX_FAMILIES: List[str] = [
    "complex_ribbed_plate",
    "complex_cored_housing",
    "complex_flanged_hub",
    "complex_stepped_block",
]
HEAVY_FAMILIES: List[str] = [
    "heavy_cored_block",
    "heavy_stepped_block",
    "heavy_bossed_plate",
    "heavy_flanged_hub",
]

Built = Tuple[trimesh.Trimesh, List[trimesh.Trimesh], Dict[str, float], float]

class DesignError(RuntimeError):
    pass

def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

def _scaled_params(family: str, rng: np.random.Generator) -> Dict[str, float]:
    params = sample_params(family, rng)
    spec = make_spec(family, params)
    scale = _log_uniform(rng, MIN_MAX_DIM_IN, MAX_MAX_DIM_IN) / max(spec.max_dim, 1e-6)
    return {k: v if k.endswith("_count") else max(0.25, v * scale) for k, v in params.items()}

def _y_cylinder(radius: float, length: float, centre) -> trimesh.Trimesh:
    mesh = _cylinder(radius, length)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
    mesh.apply_translation(np.asarray(centre, dtype=np.float64))
    return mesh

def _through_core_y(width: float, height: float, length_in_metal: float, centre, rng, cylindrical: bool):
    length = length_in_metal + 2 * rng.uniform(0.75, 1.5)
    if cylindrical:
        return _y_cylinder(0.5 * min(width, height), length, centre)
    return _box((width, length, height), centre)

def _cored_box(rng: np.random.Generator) -> Built:
    D = _log_uniform(rng, MIN_MAX_DIM_IN, MAX_MAX_DIM_IN)
    L, W, H = D, max(2.0, D * rng.uniform(0.4, 1.0)), max(2.0, D * rng.uniform(0.3, 0.8))
    cast = _box((L, W, H))
    wall = float(max(min(L, W, H) * rng.uniform(0.12, 0.3), MIN_WALL_IN))
    core_h = H - 2 * wall
    core_w = (L - 2 * wall) * rng.uniform(0.3, 1.0)
    cylindrical = bool(rng.uniform() < 0.5)
    core = _through_core_y(core_w, core_h, W, (0.0, 0.0, 0.0), rng, cylindrical)
    params = {"length": L, "width": W, "height": H, "wall": wall, "core_width": core_w,
              "core_height": core_h, "core_cylindrical": float(cylindrical)}
    return cast, [core], params, wall

def _cored_cylinder(rng: np.random.Generator) -> Built:
    D = _log_uniform(rng, MIN_MAX_DIM_IN, MAX_MAX_DIM_IN)
    tall = bool(rng.uniform() < 0.5)
    r_out = max(1.25, (D * rng.uniform(0.15, 0.4)) if tall else D / 2)
    height = D if tall else max(2.0, D * rng.uniform(0.3, 0.8))
    cast = _cylinder(r_out, height)
    r_core = max(0.5 * MIN_CORE_DIAMETER_IN, (r_out - MIN_WALL_IN) * rng.uniform(0.25, 0.7))
    print_len = rng.uniform(0.75, 1.5)
    if rng.uniform() < 0.5:
        cap = max(MIN_WALL_IN, 0.2 * height)
        length = height - cap + print_len
        core = _cylinder(r_core, length, (0.0, 0.0, -height / 2 - print_len + length / 2))
    else:
        core = _cylinder(r_core, height + 2 * print_len)
    params = {"outer_radius": r_out, "height": height, "core_radius": r_core}
    return cast, [core], params, r_out - r_core

def _cored_flanged_cylinder(rng: np.random.Generator) -> Built:
    params = _scaled_params("flanged_cylinder", rng)
    params["radius"] = max(params["radius"], 1.25)
    params["flange_radius"] = max(params["flange_radius"], 1.4 * params["radius"])
    params["flange_thickness"] = max(params["flange_thickness"], MIN_WALL_IN)
    cast = make_spec("flanged_cylinder", params).mesh
    r_core = max(0.5 * MIN_CORE_DIAMETER_IN, (params["radius"] - MIN_WALL_IN) * rng.uniform(0.45, 0.75))
    lo, hi = cast.bounds
    print_len = rng.uniform(0.75, 1.5)
    length = float(hi[2] - lo[2]) + 2 * print_len
    core = _cylinder(r_core, length, (0.0, 0.0, 0.5 * float(lo[2] + hi[2])))
    params["core_radius"] = r_core
    return cast, [core], params, min(params["flange_thickness"], params["radius"] - r_core)

def _rigged(family: str):
    def build(rng: np.random.Generator) -> Built:
        params = _scaled_params(family, rng)
        spec = make_spec(family, params)
        return spec.mesh, [], params, spec.min_thickness
    return build

def _complex_ribbed_plate(rng: np.random.Generator) -> Built:
    L = _log_uniform(rng, COMPLEX_MIN_DIM_IN, COMPLEX_MAX_DIM_IN)
    W = L * rng.uniform(0.45, 0.8)
    t = float(np.clip(L * rng.uniform(0.03, 0.06), 0.6, 2.0))
    n_ribs = int(rng.integers(2, 5))
    rib_t = t * rng.uniform(0.7, 1.1)
    rib_h = t * rng.uniform(1.5, 3.0)
    overlap = JUNCTION_OVERLAP_FRACTION * min(t, rib_t)
    shells = [_box((L, W, t), (0.0, 0.0, t / 2))]
    rib_y = [-W / 2 + W * (i + 1) / (n_ribs + 1) for i in range(n_ribs)]
    for y in rib_y:
        shells.append(_box((0.92 * L, rib_t, rib_h), (0.0, y, t - overlap + rib_h / 2)))
    bay_y = [(-W / 2 + rib_y[0]) / 2] + [(a + b) / 2 for a, b in zip(rib_y, rib_y[1:])] + [(rib_y[-1] + W / 2) / 2]
    bay_w = W / (n_ribs + 1) - rib_t
    n_bosses = int(rng.integers(2, 5))
    boss_r = float(max(0.42 * bay_w * rng.uniform(0.6, 1.0), 0.75))
    boss_h = rib_h * rng.uniform(1.1, 1.8)
    cores = []
    for x in np.linspace(-0.38 * L, 0.38 * L, n_bosses):
        y = float(rng.choice(bay_y))
        shells.append(_cylinder(boss_r, boss_h, (float(x), y, t - overlap + boss_h / 2)))
        top = t - overlap + boss_h
        r_core = boss_r * rng.uniform(0.35, 0.55)
        if rng.uniform() < 0.5 and 2 * r_core >= MIN_CORE_DIAMETER_IN and boss_r - r_core >= MIN_WALL_IN:
            print_len = rng.uniform(0.75, 1.5)
            cores.append(_cylinder(r_core, top + 2 * print_len, (float(x), y, top / 2)))
    params = {"length": L, "width": W, "thickness": t, "rib_count": float(n_ribs), "rib_thickness": rib_t,
              "rib_height": rib_h, "boss_count": float(n_bosses), "boss_radius": boss_r,
              "boss_height": boss_h, "core_count": float(len(cores))}
    return _concat(shells), cores, params, min(t, rib_t)

def _complex_cored_housing(rng: np.random.Generator) -> Built:
    L = _log_uniform(rng, COMPLEX_MIN_DIM_IN, COMPLEX_MAX_DIM_IN)
    W = L * rng.uniform(0.5, 0.85)
    H = L * rng.uniform(0.3, 0.55)
    wall = float(max(min(L, W, H) * rng.uniform(0.08, 0.16), 0.6))
    pad_l = L * rng.uniform(0.25, 0.4)
    pad_w = W * rng.uniform(0.45, 0.7)
    pad_h = float(max(H * rng.uniform(0.2, 0.4), 2.0))
    pad_x = float(rng.choice([-1.0, 1.0])) * (L / 2 - pad_l / 2 - wall)
    overlap = JUNCTION_OVERLAP_FRACTION * min(wall, pad_h)
    cast = _concat([_box((L, W, H), (0.0, 0.0, H / 2)), _box((pad_l, pad_w, pad_h), (pad_x, 0.0, H - overlap + pad_h / 2))])
    cavity = _through_core_y(L - 2 * wall, H - 2 * wall, W, (0.0, 0.0, H / 2), rng, cylindrical=False)
    pad_core_d = float(max(min(pad_l, pad_h) * rng.uniform(0.25, 0.45), MIN_CORE_DIAMETER_IN))
    pad_core = _through_core_y(pad_core_d, pad_core_d, pad_w, (pad_x, 0.0, H - overlap + pad_h / 2), rng, cylindrical=True)
    params = {"length": L, "width": W, "height": H, "wall": wall, "pad_length": pad_l, "pad_width": pad_w,
              "pad_height": pad_h, "pad_core_diameter": pad_core_d}
    return cast, [cavity, pad_core], params, wall

def _complex_flanged_hub(
    rng: np.random.Generator,
    dims: Tuple[float, float] = (COMPLEX_MIN_DIM_IN, COMPLEX_MAX_DIM_IN),
    flange_frac: Tuple[float, float] = (0.08, 0.15),
    min_flange: float = 0.75,
    bore_frac: Tuple[float, float] = (0.55, 0.75),
    hub_wall: Optional[Tuple[float, float]] = None,
) -> Built:
    rf = _log_uniform(rng, *dims) / 2
    flange = rng.uniform(*flange_frac)
    r = rf * rng.uniform(0.4, 0.6)
    wall = None
    if hub_wall is not None:
        wall = min(rng.uniform(*hub_wall), 0.85 * dims[1] / 2 / 1.5)
        rf = max(rf, 1.5 * wall / 0.85)
        r = min(max(r, 1.5 * wall), 0.85 * rf)
    tf = float(max(rf * flange, min_flange))
    h = rf * rng.uniform(0.5, 1.1)
    overlap = JUNCTION_OVERLAP_FRACTION * min(tf, h)
    shells = [_cylinder(rf, tf, (0.0, 0.0, tf / 2)), _cylinder(r, h, (0.0, 0.0, tf - overlap + h / 2))]
    top = tf - overlap + h
    r_bore = r * rng.uniform(*bore_frac) if wall is None else r - wall
    print_len = rng.uniform(1.0, 2.0)
    cores = [_cylinder(r_bore, top + 2 * print_len, (0.0, 0.0, top / 2))]
    n_bolts = int(rng.integers(3, 7))
    bolt_at = 0.5 * (r + rf)
    boss_r = float(max(0.35 * (rf - r) * rng.uniform(0.6, 1.0), 0.75))
    boss_h = tf * rng.uniform(0.8, 1.6)
    for a in np.linspace(0.0, 2 * np.pi, n_bolts, endpoint=False) + rng.uniform(0, np.pi):
        x, y = bolt_at * np.cos(a), bolt_at * np.sin(a)
        shells.append(_cylinder(boss_r, boss_h, (x, y, tf - overlap + boss_h / 2)))
        cores.append(_cylinder(boss_r * 0.45, tf + boss_h + 2.0, (x, y, (tf + boss_h) / 2)))
    params = {"flange_radius": rf, "flange_thickness": tf, "radius": r, "height": h,
              "bore_radius": r_bore, "bolt_count": float(n_bolts), "boss_radius": boss_r}
    return _concat(shells), cores, params, min(tf, r - r_bore, 0.55 * boss_r)

def _complex_stepped_block(
    rng: np.random.Generator,
    dims: Tuple[float, float] = (COMPLEX_MIN_DIM_IN, COMPLEX_MAX_DIM_IN),
    heavy_step: Optional[Tuple[float, float]] = None,
) -> Built:
    L = _log_uniform(rng, *dims)
    W = L * rng.uniform(0.3, 0.6)
    steps = int(rng.integers(3, 5))
    lengths = rng.dirichlet(np.ones(steps) * 3.0) * L
    thick = np.sort(np.clip(L * rng.uniform(0.04, 0.25, steps), 0.75, None))
    if heavy_step is not None:
        thick[-1] = max(min(rng.uniform(*heavy_step), W), thick[-2])
    if rng.uniform() < 0.5:
        thick = thick[::-1]
    shells, centres, x0 = [], [], -L / 2
    for length, t in zip(lengths, thick):
        shells.append(_box((length + 0.25, W, t), (x0 + length / 2, 0.0, t / 2)))
        centres.append(x0 + length / 2)
        x0 += length
    heavy = int(np.argmax(thick))
    t_h, l_h = float(thick[heavy]), float(lengths[heavy])
    cores = []
    core_d = min(t_h, l_h) * rng.uniform(0.3, 0.5)
    if core_d >= MIN_CORE_DIAMETER_IN and (t_h - core_d) / 2 >= MIN_WALL_IN:
        cores.append(_through_core_y(core_d, core_d, W, (centres[heavy], 0.0, t_h / 2), rng, cylindrical=True))
    params = {"length": L, "width": W, "step_count": float(steps), "max_thickness": t_h,
              "min_thickness": float(thick.min()), "core_diameter": float(core_d if cores else 0.0)}
    for i in range(steps):
        params[f"step_length_{i}"] = float(lengths[i])
        params[f"step_thickness_{i}"] = float(thick[i])
    return _concat(shells), cores, params, float(thick.min())

def _heavy_cored_block(rng: np.random.Generator) -> Built:
    L = _log_uniform(rng, HEAVY_MIN_DIM_IN, HEAVY_MAX_DIM_IN)
    W = L * rng.uniform(0.4, 0.8)
    H = float(min(rng.uniform(*HEAVY_SECTION_IN), W))
    block_l = L * rng.uniform(0.45, 0.75)
    foot_t = float(np.clip(H * rng.uniform(0.15, 0.35), 1.25, 3.5))
    block_x = float(rng.uniform(-1.0, 1.0)) * (L - block_l) / 2
    overlap = JUNCTION_OVERLAP_FRACTION * foot_t
    cast = _concat([_box((L, W, foot_t), (0.0, 0.0, foot_t / 2)),
                    _box((block_l, W, H), (block_x, 0.0, foot_t - overlap + H / 2))])
    z_mid = foot_t - overlap + H / 2
    core_d = H * rng.uniform(0.2, 0.35)
    n_cores = int(rng.integers(0, 3)) if block_l >= 2 * core_d + 3 * MIN_WALL_IN else 0
    cores = [_through_core_y(core_d, core_d, W, (block_x + dx, 0.0, z_mid), rng, cylindrical=True)
             for dx in (np.linspace(-0.25, 0.25, n_cores) * block_l if n_cores > 1 else [0.0] * n_cores)]
    params = {"length": L, "width": W, "height": H, "block_length": block_l, "foot_thickness": foot_t,
              "core_count": float(n_cores), "core_diameter": float(core_d if cores else 0.0)}
    return cast, cores, params, foot_t

def _heavy_bossed_plate(rng: np.random.Generator) -> Built:
    L = _log_uniform(rng, HEAVY_MIN_DIM_IN, HEAVY_MAX_DIM_IN)
    W = L * rng.uniform(0.5, 0.9)
    t = float(np.clip(L * rng.uniform(0.06, 0.12), 2.0, 5.0))
    n = int(rng.integers(1, 4))
    boss_d = float(min(rng.uniform(*HEAVY_SECTION_IN), 0.8 * W, 0.8 * L / n))
    boss_h = boss_d * rng.uniform(0.6, 1.2)
    overlap = JUNCTION_OVERLAP_FRACTION * t
    shells = [_box((L, W, t), (0.0, 0.0, t / 2))]
    cores = []
    for x in ((np.arange(n) + 0.5) / n - 0.5) * L:
        y = float(rng.uniform(-0.25, 0.25)) * (W - boss_d)
        shells.append(_cylinder(boss_d / 2, boss_h, (float(x), y, t - overlap + boss_h / 2)))
        if rng.uniform() < 0.5:
            r_core = boss_d / 2 * rng.uniform(0.2, 0.35)
            print_len = rng.uniform(0.75, 1.5)
            top = t - overlap + boss_h * rng.uniform(0.4, 0.7)
            cores.append(_cylinder(r_core, top + print_len, (float(x), y, (top - print_len) / 2)))
    params = {"length": L, "width": W, "thickness": t, "boss_count": float(n), "boss_diameter": boss_d,
              "boss_height": boss_h, "core_count": float(len(cores))}
    return _concat(shells), cores, params, t

_BUILDERS = {
    "cored_box": _cored_box,
    "cored_cylinder": _cored_cylinder,
    "cored_flanged_cylinder": _cored_flanged_cylinder,
    **{family: _rigged(family) for family in RIGGED_FAMILIES},
    "complex_ribbed_plate": _complex_ribbed_plate,
    "complex_cored_housing": _complex_cored_housing,
    "complex_flanged_hub": _complex_flanged_hub,
    "complex_stepped_block": _complex_stepped_block,
    "heavy_cored_block": _heavy_cored_block,
    "heavy_stepped_block": partial(
        _complex_stepped_block, dims=(HEAVY_MIN_DIM_IN, HEAVY_MAX_DIM_IN), heavy_step=HEAVY_SECTION_IN,
    ),
    "heavy_bossed_plate": _heavy_bossed_plate,
    "heavy_flanged_hub": partial(
        _complex_flanged_hub, dims=(HEAVY_MIN_DIM_IN, HEAVY_MAX_DIM_IN),
        flange_frac=(0.15, 0.25), min_flange=2.0, hub_wall=HEAVY_SECTION_IN,
    ),
}

def build_assembly(family: str, rng: np.random.Generator, index: int = 0) -> GeometrySpec:
    if family not in _BUILDERS:
        raise ValueError(f"Unknown assembly family: {family}")
    for _attempt in range(MAX_DESIGN_ATTEMPTS):
        cast, cores, params, min_thickness = _BUILDERS[family](rng)
        parts = design_rules.rig(cast, cores, rng)
        if parts is None:
            continue
        report = design_rules.check(parts)
        params = dict(params)
        params["riser_count"] = report.metrics.get("riser_count", 0.0)
        params["chill_count"] = float(sum(1 for _, c in parts if c == PART_TYPE_CHILL))
        params["fed_fraction"] = report.metrics.get("fed_fraction", 0.0)
        params = _round_params(params)
        for mesh, _code in parts:
            mesh._cache.clear()
        return GeometrySpec(
            name=_spec_name(f"rig_{family}", index, params),
            family=f"rig_{family}",
            mesh=_concat([mesh for mesh, _ in parts]),
            params=params,
            min_thickness=float(max(min_thickness, 0.25)),
            parts=parts,
        )
    raise DesignError(f"no sound {family} design in {MAX_DESIGN_ATTEMPTS} draws")

def iter_assemblies(
    count: int,
    seed: int = 0,
    families: Sequence[str] = tuple(ASSEMBLY_FAMILIES),
) -> Iterator[GeometrySpec]:
    families = list(families)
    unknown = [f for f in families if f not in _BUILDERS]
    if unknown:
        raise ValueError(f"Unknown assembly families: {unknown}")
    rng = np.random.default_rng(seed)
    for index in range(int(count)):
        try:
            yield build_assembly(families[index % len(families)], rng, index=index)
        except DesignError:
            continue
