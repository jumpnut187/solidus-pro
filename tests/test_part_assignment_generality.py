from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
import trimesh

from core.part_manager import (
    CODE_TO_PART_TYPE,
    PART_TYPE_CAST,
    PART_TYPE_TO_CODE,
    PartType,
)
from core.voxel_mesh import (
    _INTERFACE_BRIDGE_PAIRS,
    AdaptiveVoxelMesh,
)

VOXEL_SIZE = 0.25

NON_CAST_CODES = tuple(sorted(set(PART_TYPE_TO_CODE.values()) - {PART_TYPE_CAST}))

def _two_body_assembly():
    base = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    base.apply_translation((0.0, 0.0, 1.0))
    upper = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    upper.apply_translation((0.0, 0.0, 2.5))
    return base, upper

def _build_mesh(bodies):
    combined = trimesh.util.concatenate(bodies)
    mesh = AdaptiveVoxelMesh(base_voxel_size=VOXEL_SIZE)
    bounds = np.asarray(combined.bounds).T
    with contextlib.redirect_stdout(io.StringIO()):
        ok = mesh.generate(combined, bounds, show_progress=False)
    if not ok:
        raise unittest.SkipTest("mesh generation failed for the fixture geometry")
    return mesh

def _assign(mesh, parts_data):
    with contextlib.redirect_stdout(io.StringIO()):
        mesh.assign_part_types(parts_data)
    return np.asarray(mesh.voxel_part_types)

class EveryPartTypeClaimsVoxelsTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.base, cls.upper = _two_body_assembly()

    def test_each_non_cast_type_claims_its_voxels(self) -> None:
        for code in NON_CAST_CODES:
            with self.subTest(part_type=CODE_TO_PART_TYPE[code].name, code=code):
                mesh = _build_mesh([self.base, self.upper])
                types = _assign(
                    mesh, [(self.base, PART_TYPE_CAST), (self.upper, code)]
                )
                claimed = int(np.sum(types == code))
                self.assertGreater(
                    claimed,
                    0,
                    f"{CODE_TO_PART_TYPE[code].name} (code {code}) claimed no voxels - "
                    f"assign_part_types is dropping it",
                )

    def test_core_specifically_is_no_longer_dropped(self) -> None:
        mesh = _build_mesh([self.base, self.upper])
        types = _assign(
            mesh,
            [(self.base, PART_TYPE_CAST), (self.upper, PART_TYPE_TO_CODE[PartType.CORE])],
        )
        self.assertGreater(int(np.sum(types == PART_TYPE_TO_CODE[PartType.CORE])), 0)

    def test_cast_still_receives_the_remaining_voxels(self) -> None:
        for code in NON_CAST_CODES:
            with self.subTest(part_type=CODE_TO_PART_TYPE[code].name):
                mesh = _build_mesh([self.base, self.upper])
                types = _assign(
                    mesh, [(self.base, PART_TYPE_CAST), (self.upper, code)]
                )
                self.assertGreater(int(np.sum(types == PART_TYPE_CAST)), 0)

    def test_assignment_partitions_every_voxel(self) -> None:
        mesh = _build_mesh([self.base, self.upper])
        types = _assign(
            mesh,
            [(self.base, PART_TYPE_CAST), (self.upper, PART_TYPE_TO_CODE[PartType.SPRUE])],
        )
        known = set(PART_TYPE_TO_CODE.values())
        self.assertTrue(set(np.unique(types)).issubset(known))

class FastPathTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.base, cls.upper = _two_body_assembly()

    def test_single_cast_part_still_takes_the_fast_path(self) -> None:
        mesh = _build_mesh([self.base])
        types = _assign(mesh, [(self.base, PART_TYPE_CAST)])
        self.assertTrue(np.all(types == PART_TYPE_CAST))

    def test_a_lone_gating_part_does_not_trigger_the_fast_path(self) -> None:
        sprue_code = PART_TYPE_TO_CODE[PartType.SPRUE]
        mesh = _build_mesh([self.base, self.upper])
        types = _assign(mesh, [(self.base, PART_TYPE_CAST), (self.upper, sprue_code)])
        self.assertGreater(
            int(np.sum(types == sprue_code)),
            0,
            "cast + sprue took the single-CAST fast path and lost the sprue",
        )

    def test_no_parts_data_leaves_everything_cast(self) -> None:
        mesh = _build_mesh([self.base])
        types = _assign(mesh, [])
        self.assertTrue(np.all(types == PART_TYPE_CAST))

class MetalPartBoundsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base, cls.upper = _two_body_assembly()

    def test_gating_metal_is_recorded_in_metal_part_bounds(self) -> None:
        mesh = _build_mesh([self.base, self.upper])
        _assign(
            mesh,
            [
                (self.base, PART_TYPE_CAST),
                (self.upper, PART_TYPE_TO_CODE[PartType.SPRUE]),
            ],
        )
        self.assertEqual(len(mesh.metal_part_bounds), 2)

    def test_a_filter_is_not_counted_as_metal(self) -> None:
        mesh = _build_mesh([self.base, self.upper])
        _assign(
            mesh,
            [
                (self.base, PART_TYPE_CAST),
                (self.upper, PART_TYPE_TO_CODE[PartType.FILTER]),
            ],
        )
        self.assertEqual(len(mesh.metal_part_bounds), 1)

class BridgePairTests(unittest.TestCase):

    def test_pairs_reference_real_codes(self) -> None:
        known = set(PART_TYPE_TO_CODE.values())
        for code_a, code_b in _INTERFACE_BRIDGE_PAIRS:
            with self.subTest(pair=(code_a, code_b)):
                self.assertIn(code_a, known)
                self.assertIn(code_b, known)

    def test_pairs_are_distinct_and_unique(self) -> None:
        seen = set()
        for pair in _INTERFACE_BRIDGE_PAIRS:
            self.assertNotEqual(pair[0], pair[1], f"{pair} bridges a type to itself")
            key = frozenset(pair)
            self.assertNotIn(key, seen, f"{pair} is listed twice")
            seen.add(key)

    def test_the_gating_chain_can_reach_the_casting(self) -> None:
        adjacency: dict[int, set[int]] = {}
        for a, b in _INTERFACE_BRIDGE_PAIRS:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

        for part_type in (
            PartType.POUR_CUP,
            PartType.SPRUE,
            PartType.RUNNER,
            PartType.FILTER,
            PartType.GATE,
        ):
            with self.subTest(part_type=part_type.name):
                start = PART_TYPE_TO_CODE[part_type]
                seen, stack = {start}, [start]
                while stack:
                    node = stack.pop()
                    for neighbour in adjacency.get(node, ()):
                        if neighbour not in seen:
                            seen.add(neighbour)
                            stack.append(neighbour)
                self.assertIn(
                    PART_TYPE_CAST,
                    seen,
                    f"{part_type.name} has no bridge path to the casting",
                )

if __name__ == "__main__":
    unittest.main()
