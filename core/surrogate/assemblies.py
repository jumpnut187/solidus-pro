"""Seeded casting assemblies (casting + riser + chill + sand core) for surrogate training data.

``core.surrogate.geometries`` yields bare castings, so every record trained on
them is a single CAST body cooling into sand. Real jobs are rigged: a riser
feeds the heavy section, chills pull heat from a face, and sand cores form
internal cavities that saturate and turn into hot spots. This module builds
those assemblies as ``GeometrySpec`` objects carrying a ``parts`` list of
``(mesh, part_code)`` pairs, which ``generate_dataset`` loads into a
``PartManager`` like a user's multi-STL job.

Rigging rules (inches; every assembly gets a riser):

- Riser: a cylinder of height 1.5 D standing on the top face, sunk
  ``RISER_SINK_IN`` into it. D follows the modulus rule
  ``M_riser = 0.214 D >= 1.2 M_cast`` with ``M_cast = V / A`` of the
  casting, times a factor in [0.7, 1.5], so some risers are undersized and
  the casting shows shrinkage. D is capped by the top face.
- Chill (probability ``CHILL_PROBABILITY``): a graphite block flush against
  the bottom or the -x face, covering part of it.
- Core (cored families only): a box or cylinder inside the casting, either
  fully enclosed (no prints: the saturating case) or running through it with
  prints that stand out of the casting. Walls around a core are at least
  ``MIN_WALL_IN``. Core overlap with the casting is claimed by the core
  (the higher part code), which is how a core forms a cavity.

Part codes resolve overlap toward the higher code (CAST 0 < RISER 1 < CHILL 2
< CORE 4), so the parts only need to touch or overlap; no booleans are needed.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import trimesh

from core.part_manager import PART_TYPE_CAST, PART_TYPE_CHILL, PART_TYPE_CORE, PART_TYPE_RISER
from core.surrogate.geometries import (
    GeometrySpec,
    _box,
    _concat,
    _cylinder,
    _round_params,
    _spec_name,
    make_spec,
    sample_params,
)

#: Casting size range for assemblies (smaller than bare geometries: a rig
#: adds a riser and core detail that need voxels).
MIN_MAX_DIM_IN = 3.0
MAX_MAX_DIM_IN = 14.0
MIN_WALL_IN = 0.3
RISER_SINK_IN = 0.1
CHILL_PROBABILITY = 0.5

#: Families with a sand core.
CORED_FAMILIES: List[str] = ["cored_box", "cored_cylinder", "cored_flanged_cylinder"]
#: Families rigged with a riser and optional chill, no core.
RIGGED_FAMILIES: List[str] = ["plate_with_boss", "t_section", "l_bracket", "stepped_plate", "box", "wedge"]
ASSEMBLY_FAMILIES: List[str] = CORED_FAMILIES + RIGGED_FAMILIES

Part = Tuple[trimesh.Trimesh, int]


def _sample_dim(rng: np.random.Generator) -> float:
    return float(np.exp(rng.uniform(np.log(MIN_MAX_DIM_IN), np.log(MAX_MAX_DIM_IN))))


def _scaled_params(family: str, rng: np.random.Generator) -> Dict[str, float]:
    """``geometries.sample_params`` for ``family``, rescaled into the assembly size range."""
    params = sample_params(family, rng)
    spec = make_spec(family, params)
    scale = _sample_dim(rng) / max(spec.max_dim, 1e-6)
    out = {}
    for key, value in params.items():
        out[key] = value if key.endswith("_count") else max(0.25, value * scale)
    return out


def _modulus(mesh: trimesh.Trimesh) -> float:
    """Casting modulus V / A (inches); concatenated shells overcount both a little."""
    return float(abs(mesh.volume)) / max(float(mesh.area), 1e-9)


def _riser_site(cast: trimesh.Trimesh) -> Tuple[Tuple[float, float], float]:
    """Point on the topmost face to stand a riser on, and the largest diameter that fits there.

    Samples a 21 x 21 grid just below the top of the bounding box, keeps the
    points inside the casting (so a riser over an off-centre boss lands on
    the boss, not in air), and picks the one nearest their centroid. The
    diameter limit is twice its distance to the nearest point that missed.
    """
    lo, hi = cast.bounds
    xs = np.linspace(lo[0], hi[0], 23)[1:-1]
    ys = np.linspace(lo[1], hi[1], 23)[1:-1]
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, hi[2] - 1.5 * RISER_SINK_IN)])
    inside = np.asarray(cast.contains(points), dtype=bool)
    if not np.any(inside):
        return (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])), 0.8 * float(min(hi[0] - lo[0], hi[1] - lo[1]))
    hits = points[inside, :2]
    centre = hits[np.argmin(np.linalg.norm(hits - hits.mean(axis=0), axis=1))]
    misses = points[~inside, :2]
    edge = np.min(np.linalg.norm(misses - centre, axis=1)) if len(misses) else 0.5 * float(min(hi[0] - lo[0], hi[1] - lo[1]))
    return (float(centre[0]), float(centre[1])), 1.6 * float(edge)


def _riser(cast: trimesh.Trimesh, rng: np.random.Generator, xy=None, max_diameter=None) -> Tuple[trimesh.Trimesh, float]:
    """Cylinder riser on the top face at ``xy`` (default: ``_riser_site``)."""
    lo, hi = cast.bounds
    face_limit = 0.8 * float(min(hi[0] - lo[0], hi[1] - lo[1]))
    if xy is None:
        xy, site_limit = _riser_site(cast)
        face_limit = min(face_limit, site_limit)
    if max_diameter is not None:
        face_limit = min(face_limit, float(max_diameter))
    diameter = 1.2 * _modulus(cast) / 0.214 * rng.uniform(0.7, 1.5)
    diameter = float(np.clip(diameter, 0.5, max(0.5, face_limit)))
    height = 1.5 * diameter
    # Stand on the local top surface (sloped and stepped tops sit below the
    # bounding box), found by a ray straight down through the site.
    hits, _, _ = cast.ray.intersects_location(
        ray_origins=[[xy[0], xy[1], float(hi[2]) + 1.0]], ray_directions=[[0.0, 0.0, -1.0]]
    )
    top = float(np.max(hits[:, 2])) if len(hits) else float(hi[2])
    z0 = top - RISER_SINK_IN
    return _cylinder(diameter / 2.0, height, (xy[0], xy[1], z0 + height / 2.0)), diameter


def _chill(cast: trimesh.Trimesh, rng: np.random.Generator, faces: Sequence[str]) -> Tuple[trimesh.Trimesh, str]:
    """Graphite block flush against one of ``faces`` ("bottom" or "-x")."""
    lo, hi = cast.bounds
    ext = hi - lo
    face = str(rng.choice(list(faces)))
    thickness = float(np.clip(0.3 * float(ext.min()) * rng.uniform(0.6, 1.4), 0.3, 1.5))
    cover = rng.uniform(0.35, 0.8)
    if face == "bottom":
        size = (ext[0] * cover, ext[1] * cover, thickness)
        centre = (lo[0] + ext[0] * rng.uniform(cover / 2, 1 - cover / 2), 0.5 * (lo[1] + hi[1]), lo[2] - thickness / 2)
    else:
        size = (thickness, ext[1] * cover, ext[2] * cover)
        centre = (lo[0] - thickness / 2, 0.5 * (lo[1] + hi[1]), lo[2] + ext[2] * rng.uniform(cover / 2, 1 - cover / 2))
    return _box(size, centre), face


def _cored_box(rng: np.random.Generator) -> Tuple[List[Part], Dict[str, float], float]:
    D = _sample_dim(rng)
    L, W, H = D, max(1.5, D * rng.uniform(0.4, 1.0)), max(1.5, D * rng.uniform(0.3, 0.8))
    cast = _box((L, W, H))
    wall = float(np.clip(min(L, W, H) * rng.uniform(0.12, 0.35), MIN_WALL_IN, None))
    core_h = max(0.4, H - 2 * wall)
    core_w = max(0.4, (L - 2 * wall) * rng.uniform(0.3, 1.0))
    enclosed = bool(rng.uniform() < 0.5)
    # Through cores run along y with prints standing 0.5-1.5 in out of both
    # faces, so a bottom or -x chill and a top riser never meet them.
    core_l = W - 2 * wall if enclosed else W + 2 * rng.uniform(0.5, 1.5)
    cylindrical = bool(rng.uniform() < 0.5)
    if cylindrical:
        radius = 0.5 * min(core_h, core_w)
        core = _cylinder(radius, max(0.4, core_l), (0.0, 0.0, 0.0))
        core.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
        core_w = core_h = 2 * radius
    else:
        core = _box((core_w, max(0.4, core_l), core_h))
    parts: List[Part] = [(cast, PART_TYPE_CAST), (core, PART_TYPE_CORE)]
    params = {
        "length": L, "width": W, "height": H, "wall": wall,
        "core_width": core_w, "core_height": core_h, "core_length": core_l,
        "core_enclosed": float(enclosed), "core_cylindrical": float(cylindrical),
    }
    return parts, params, wall


def _cored_cylinder(rng: np.random.Generator) -> Tuple[List[Part], Dict[str, float], float]:
    D = _sample_dim(rng)
    tall = bool(rng.uniform() < 0.5)
    r_out = max(1.0, (D * rng.uniform(0.15, 0.4)) if tall else D / 2)
    height = D if tall else max(1.5, D * rng.uniform(0.3, 0.8))
    cast = _cylinder(r_out, height)
    r_core = max(0.25, (r_out - MIN_WALL_IN) * rng.uniform(0.2, 0.7))
    enclosed = bool(rng.uniform() < 0.4)
    if enclosed:
        core_len = max(0.5, height - 2 * max(MIN_WALL_IN, r_out - r_core))
        core = _cylinder(r_core, core_len)
    else:
        # Through bore with a print out of the bottom only: the riser stands
        # on the top face ring, and prints at both ends would sit under it.
        print_len = rng.uniform(0.5, 1.5)
        core_len = height - max(MIN_WALL_IN, 0.25 * height) + print_len
        core = _cylinder(r_core, core_len, (0.0, 0.0, -height / 2 - print_len + core_len / 2))
    parts: List[Part] = [(cast, PART_TYPE_CAST), (core, PART_TYPE_CORE)]
    params = {
        "outer_radius": r_out, "height": height, "core_radius": r_core,
        "core_length": core_len, "core_enclosed": float(enclosed),
    }
    return parts, params, r_out - r_core


def _cored_flanged_cylinder(rng: np.random.Generator) -> Tuple[List[Part], Dict[str, float], float]:
    params = _scaled_params("flanged_cylinder", rng)
    # Room for a bore: body radius >= 1 in, flange wider than the body.
    params["radius"] = max(params["radius"], 1.0)
    params["flange_radius"] = max(params["flange_radius"], 1.3 * params["radius"])
    cast = make_spec("flanged_cylinder", params).mesh
    r_core = max(0.25, (params["radius"] - MIN_WALL_IN) * rng.uniform(0.3, 0.75))
    lo, hi = cast.bounds
    print_len = rng.uniform(0.5, 1.5)
    length = float(hi[2] - lo[2]) + print_len
    core = _cylinder(r_core, length, (0.0, 0.0, float(lo[2]) - print_len + length / 2 - MIN_WALL_IN))
    params = dict(params, core_radius=r_core, core_length=length)
    parts: List[Part] = [(cast, PART_TYPE_CAST), (core, PART_TYPE_CORE)]
    return parts, params, min(params["flange_thickness"], params["radius"] - r_core)


def build_assembly(family: str, rng: np.random.Generator, index: int = 0) -> GeometrySpec:
    """One seeded assembly of ``family`` (see module docstring)."""
    if family in CORED_FAMILIES:
        parts, params, min_thickness = {
            "cored_box": _cored_box,
            "cored_cylinder": _cored_cylinder,
            "cored_flanged_cylinder": _cored_flanged_cylinder,
        }[family](rng)
    elif family in RIGGED_FAMILIES:
        params = _scaled_params(family, rng)
        spec = make_spec(family, params)
        parts, min_thickness = [(spec.mesh, PART_TYPE_CAST)], spec.min_thickness
    else:
        raise ValueError(f"Unknown assembly family: {family}")

    cast = parts[0][0]
    if family == "cored_cylinder":
        # Ring top face: stand the riser on the metal between bore and rim.
        r_out, r_core = params["outer_radius"], params["core_radius"]
        ring = r_out - r_core
        riser, diameter = _riser(cast, rng, xy=(r_core + ring / 2, 0.0), max_diameter=0.9 * ring)
    elif family == "cored_flanged_cylinder":
        ring = params["radius"] - params["core_radius"]
        riser, diameter = _riser(
            cast, rng, xy=(params["core_radius"] + ring / 2, 0.0), max_diameter=0.9 * ring
        )
    else:
        riser, diameter = _riser(cast, rng)
    parts.append((riser, PART_TYPE_RISER))
    params["riser_diameter"] = diameter

    if rng.uniform() < CHILL_PROBABILITY:
        faces = ("bottom",) if family in ("cored_cylinder", "cored_flanged_cylinder") else ("bottom", "-x")
        if family in ("cored_cylinder", "cored_flanged_cylinder") and not params.get("core_enclosed", 0.0):
            faces = ()  # the bore print stands out of the bottom face
        if faces:
            chill, face = _chill(cast, rng, faces)
            parts.append((chill, PART_TYPE_CHILL))
            params["chill_face_bottom"] = float(face == "bottom")

    params = _round_params(params)
    return GeometrySpec(
        name=_spec_name(f"rig_{family}", index, params),
        family=f"rig_{family}",
        mesh=_concat([mesh for mesh, _ in parts]),
        params=params,
        min_thickness=float(max(min_thickness, 0.25)),
        parts=parts,
    )


def iter_assemblies(
    count: int,
    seed: int = 0,
    families: Sequence[str] = tuple(ASSEMBLY_FAMILIES),
) -> Iterator[GeometrySpec]:
    """Yield ``count`` seeded assemblies, cycling through ``families``.

    Deterministic in ``(count, seed, families)``; a shorter run is a prefix
    of a longer one.
    """
    families = list(families)
    unknown = [f for f in families if f not in ASSEMBLY_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown assembly families: {unknown}")
    rng = np.random.default_rng(seed)
    for index in range(int(count)):
        yield build_assembly(families[index % len(families)], rng, index=index)
