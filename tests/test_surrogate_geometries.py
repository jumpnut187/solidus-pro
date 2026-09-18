"""Tests for core.surrogate.geometries and the synthetic-data generation path."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest

import numpy as np

from core.surrogate.geometries import (
    FAMILIES,
    MAX_MAX_DIM_IN,
    MIN_MAX_DIM_IN,
    GeometrySpec,
    generate_geometries,
    named_catalog,
    voxel_size_for,
)


class GeneratorDeterminismTests(unittest.TestCase):
    def test_same_seed_same_geometries(self):
        a = generate_geometries(12, seed=3)
        b = generate_geometries(12, seed=3)
        self.assertEqual([s.name for s in a], [s.name for s in b])
        for sa, sb in zip(a, b):
            self.assertEqual(sa.params, sb.params)
            np.testing.assert_array_equal(sa.mesh.vertices, sb.mesh.vertices)

    def test_prefix_stable_and_seed_dependent(self):
        short = [s.name for s in generate_geometries(5, seed=1)]
        long = [s.name for s in generate_geometries(9, seed=1)]
        self.assertEqual(short, long[:5])
        self.assertNotEqual(short, [s.name for s in generate_geometries(5, seed=2)])

    def test_families_cycle(self):
        specs = generate_geometries(len(FAMILIES) * 2, seed=0)
        self.assertEqual([s.family for s in specs], FAMILIES + FAMILIES)


class MeshQualityTests(unittest.TestCase):
    def _check_spec(self, spec: GeometrySpec):
        self.assertTrue(spec.mesh.is_watertight, spec.name)
        self.assertTrue(spec.mesh.is_volume, spec.name)
        self.assertGreater(spec.mesh.volume, 0.0, spec.name)
        self.assertGreaterEqual(spec.max_dim, MIN_MAX_DIM_IN - 1e-6, spec.name)
        self.assertLessEqual(spec.max_dim, MAX_MAX_DIM_IN + 1e-6, spec.name)
        self.assertGreaterEqual(spec.min_thickness, 0.25, spec.name)

    def test_catalog_meshes(self):
        catalog = named_catalog()
        self.assertEqual(sorted(catalog), sorted(FAMILIES))
        for spec in catalog.values():
            self._check_spec(spec)

    def test_seeded_meshes(self):
        for spec in generate_geometries(44, seed=7):
            self._check_spec(spec)

    def test_voxel_size_rule_bounds(self):
        for spec in generate_geometries(33, seed=11):
            vs = voxel_size_for(spec, grid=64, target_voxels=20_000, max_voxels=50_000)
            estimate = spec.mesh.volume / vs ** 3
            self.assertGreaterEqual(vs, 0.05, spec.name)
            # Count cap can be nudged by the 0.005 in rounding; allow 10%.
            self.assertLessEqual(estimate, 55_000, spec.name)
            self.assertGreaterEqual(estimate, 2_000, spec.name)


class CompoundVoxelizeCaptureTests(unittest.TestCase):
    def test_plate_with_boss_voxelizes_solid_and_captures(self):
        """Overlapping shells (plate + cylinder boss) must voxelize with no hollow
        interior cells, then solve and capture into a valid record."""
        from core.simulation_pipeline import PipelineInputs, run_pipeline
        from core.surrogate.dataset import capture_run, load_dataset, validate_record
        from core.surrogate.training import _loader_for_trimesh, _run_on_large_stack

        spec = named_catalog()["plate_with_boss"]
        p = spec.params
        voxel = 0.25
        inputs = PipelineInputs(
            alloy_name="WCB",
            voxel_size=voxel,
            refinement_level=1,
            memory_mode="ram",
            analysis_options={"shrinkage": False, "hotspot": False, "fraction_liquid": False},
        )

        def _solve():
            with contextlib.redirect_stdout(io.StringIO()):
                return run_pipeline(inputs, stl_loader=_loader_for_trimesh(spec.mesh))

        result = _run_on_large_stack(_solve)
        centers = np.asarray(result.engine.mesh.voxel_centers, dtype=np.float64)

        # Rebuild the occupancy lattice and compare to the analytic solid.
        origin = centers.min(axis=0)
        idx = np.round((centers - origin) / voxel).astype(int)
        shape = tuple(idx.max(axis=0) + 1)
        occupied = np.zeros(shape, dtype=bool)
        occupied[tuple(idx.T)] = True
        lattice = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3) * voxel + origin
        L, W, t = p["length"], p["width"], p["thickness"]
        r, h = p["boss_radius"], p["boss_height"]
        boss_bottom = t / 2 - 0.25 * min(t, h)
        in_plate = (np.abs(lattice[:, 0]) <= L / 2) & (np.abs(lattice[:, 1]) <= W / 2) & (np.abs(lattice[:, 2]) <= t / 2)
        in_boss = (np.hypot(lattice[:, 0], lattice[:, 1]) <= r) & (lattice[:, 2] >= boss_bottom) & (lattice[:, 2] <= boss_bottom + h)
        analytic = (in_plate | in_boss).reshape(shape)
        interior = analytic.copy()
        for axis in range(3):
            for shift in (1, -1):
                interior &= np.roll(analytic, shift, axis=axis)
        hollow = int(np.sum(interior & ~occupied))
        self.assertEqual(hollow, 0, f"{hollow} hollow interior cells in plate+boss overlap")
        self.assertGreater(int(np.sum(occupied & in_boss.reshape(shape))), 0)

        with tempfile.TemporaryDirectory() as tmp:
            path = capture_run(result.engine, voxel_size=voxel, resolution=32, out_dir=tmp)
            self.assertIsNotNone(path)
            records = load_dataset(tmp)
            self.assertEqual(len(records), 1)
            self.assertIsNone(validate_record(records[0]))
            self.assertEqual(records[0]["end"].size, 32 ** 3)


class RecordValidationTests(unittest.TestCase):
    def _record(self, end_value: float = 100.0, start_value: float = 50.0):
        channels = np.zeros((7, 4, 4, 4), dtype=np.float32)
        channels[0, 1:3, 1:3, 1:3] = 1.0
        end = np.zeros(64, dtype=np.float32)
        start = np.zeros(64, dtype=np.float32)
        active = channels[0].reshape(-1) > 0.5
        end[active] = end_value
        start[active] = start_value
        return {"channels": channels, "end": end, "start": start, "local": end - start}

    def test_reasons(self):
        from core.surrogate.dataset import validate_record

        self.assertIsNone(validate_record(self._record()))
        self.assertEqual(validate_record(self._record(end_value=0.0, start_value=0.0)), "no_positive_end")
        self.assertEqual(validate_record(self._record(end_value=0.5, start_value=0.0)), "median_end_below_1s")
        self.assertEqual(validate_record(self._record(end_value=np.nan)), "nonfinite_end")
        self.assertEqual(validate_record(self._record(end_value=10.0, start_value=50.0)), "end_before_start_gt_1pct")
        empty = self._record()
        empty["channels"][0] = 0.0
        self.assertEqual(validate_record(empty), "no_active_cells")

    def test_chill_cells_excluded(self):
        from core.surrogate.dataset import validate_record
        from core.surrogate.fields import PART_TYPE_DIVISOR

        rec = self._record()
        # Turn most active cells into chills with end 0; the remaining cast cells stay valid.
        active_idx = np.flatnonzero(rec["channels"][0].reshape(-1) > 0.5)
        chill_idx = active_idx[:-2]
        rec["channels"][1].reshape(-1)[chill_idx] = 2.0 / PART_TYPE_DIVISOR
        rec["end"][chill_idx] = 0.0
        rec["start"][chill_idx] = 0.0
        self.assertIsNone(validate_record(rec))


if __name__ == "__main__":
    unittest.main()
