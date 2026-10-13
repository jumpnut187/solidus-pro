"""Tests for fair-share CPU/thread budgeting and memory gating."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.hardware_config import hardware
from core.solidification_loop import _resolve_solidification_thread_target
from core.thread_budget import (
    get_process_thread_budget,
    resolve_thread_budget,
    set_process_thread_budget,
)
from web.services.fair_share import (
    can_start_simulation,
    compute_thread_share,
    make_thread_budget_provider,
    resolve_usable_cores,
)
from web.services.simulation_queue import default_queue_config


class ThreadShareTests(unittest.TestCase):
    """Fair-share core division math."""

    def test_resolve_usable_cores(self) -> None:
        self.assertEqual(resolve_usable_cores(24, 1), 23)
        self.assertEqual(resolve_usable_cores(24, 0), 24)
        self.assertEqual(resolve_usable_cores(1, 1), 1)
        self.assertEqual(resolve_usable_cores(8, 4), 4)

    def test_compute_thread_share(self) -> None:
        self.assertEqual(compute_thread_share(23, 1), 23)
        self.assertEqual(compute_thread_share(23, 2), 11)
        self.assertEqual(compute_thread_share(23, 3), 7)
        self.assertEqual(compute_thread_share(24, 4), 6)
        self.assertEqual(compute_thread_share(1, 1), 1)
        self.assertEqual(compute_thread_share(8, 16), 1)

    def test_share_never_zero(self) -> None:
        self.assertGreaterEqual(compute_thread_share(4, 1000), 1)


class MemoryGateTests(unittest.TestCase):
    """Memory-aware concurrency gating."""

    def test_cap_reached_blocks(self) -> None:
        self.assertFalse(
            can_start_simulation(
                running_count=2,
                max_concurrent=2,
                available_mb=99999.0,
                active_memory_mb=0.0,
                reserve_mb=0.0,
                min_sim_memory_mb=0.0,
            )
        )

    def test_first_sim_always_allowed(self) -> None:
        self.assertTrue(
            can_start_simulation(
                running_count=0,
                max_concurrent=4,
                available_mb=1.0,
                active_memory_mb=0.0,
                reserve_mb=0.0,
                min_sim_memory_mb=5000.0,
            )
        )

    def test_headroom_sufficient(self) -> None:
        self.assertTrue(
            can_start_simulation(
                running_count=1,
                max_concurrent=4,
                available_mb=12000.0,
                active_memory_mb=2000.0,
                reserve_mb=2000.0,
                min_sim_memory_mb=2000.0,
            )
        )

    def test_headroom_insufficient(self) -> None:
        self.assertFalse(
            can_start_simulation(
                running_count=1,
                max_concurrent=4,
                available_mb=3000.0,
                active_memory_mb=2000.0,
                reserve_mb=2000.0,
                min_sim_memory_mb=2000.0,
            )
        )

    def test_large_active_footprint_blocks(self) -> None:
        self.assertFalse(
            can_start_simulation(
                running_count=1,
                max_concurrent=4,
                available_mb=9000.0,
                active_memory_mb=8000.0,
                reserve_mb=2000.0,
                min_sim_memory_mb=2000.0,
            )
        )


class BudgetProviderTests(unittest.TestCase):
    """Budget provider applies shared values to the process."""

    def tearDown(self) -> None:
        set_process_thread_budget(None)
        hardware.set_thread_cap(None)
        os.environ.pop("CASTSIM_THREAD_BUDGET", None)

    def test_provider_applies_budget(self) -> None:
        class FakeValue:
            value = 8

        provider = make_thread_budget_provider(FakeValue())
        self.assertEqual(provider(), 8)
        self.assertEqual(get_process_thread_budget(), 8)
        self.assertEqual(hardware.thread_cap, 8)
        self.assertEqual(os.environ["CASTSIM_THREAD_BUDGET"], "8")

    def test_provider_clears_budget(self) -> None:
        class FakeValue:
            value = 0

        provider = make_thread_budget_provider(FakeValue())
        self.assertIsNone(provider())
        self.assertIsNone(get_process_thread_budget())
        self.assertIsNone(hardware.thread_cap)
        self.assertNotIn("CASTSIM_THREAD_BUDGET", os.environ)

    def test_provider_idempotent(self) -> None:
        class FakeValue:
            value = 4

        provider = make_thread_budget_provider(FakeValue())
        self.assertEqual(provider(), 4)
        self.assertEqual(provider(), 4)
        self.assertEqual(get_process_thread_budget(), 4)

    def test_provider_none_when_no_shared_value(self) -> None:
        self.assertIsNone(make_thread_budget_provider(None))


class ThreadBudgetResolutionTests(unittest.TestCase):
    """Budget resolution precedence and solidification target capping."""

    def tearDown(self) -> None:
        set_process_thread_budget(None)
        hardware.set_thread_cap(None)
        os.environ.pop("CASTSIM_THREAD_BUDGET", None)
        os.environ.pop("CASTSIM_SOLIDIFICATION_THREADS", None)
        os.environ.pop("CASTSIM_NUMBA_THREADS", None)

    def test_process_override_wins_over_env(self) -> None:
        os.environ["CASTSIM_THREAD_BUDGET"] = "4"
        set_process_thread_budget(8)
        self.assertEqual(resolve_thread_budget(), 8)

    def test_env_fallback(self) -> None:
        set_process_thread_budget(None)
        os.environ["CASTSIM_THREAD_BUDGET"] = "6"
        self.assertEqual(resolve_thread_budget(), 6)

    def test_invalid_env_ignored(self) -> None:
        set_process_thread_budget(None)
        os.environ["CASTSIM_THREAD_BUDGET"] = "banana"
        self.assertIsNone(resolve_thread_budget())

    def test_solidification_target_respects_budget(self) -> None:
        logical = int(getattr(hardware, "logical_cores", 0) or (os.cpu_count() or 1))
        os.environ["CASTSIM_NUMBA_THREADS"] = "all"
        set_process_thread_budget(4)
        self.assertEqual(_resolve_solidification_thread_target(), min(4, logical))
        set_process_thread_budget(2)
        self.assertEqual(_resolve_solidification_thread_target(), min(2, logical))
        set_process_thread_budget(None)
        self.assertEqual(_resolve_solidification_thread_target(), logical)

    def test_hardware_cap_lowers_compute_count(self) -> None:
        base = hardware.get_optimal_thread_count('compute')
        hardware.set_thread_cap(2)
        # A cap can only lower the count: on a runner with fewer than two
        # usable cores the machine's own optimum wins. Mirrors the env-cap
        # test below, which already expects min(base, cap).
        self.assertEqual(hardware.get_optimal_thread_count('compute'), min(base, 2))
        hardware.set_thread_cap(None)
        self.assertEqual(hardware.get_optimal_thread_count('compute'), base)

    def test_hardware_env_cap_lowers_compute_count(self) -> None:
        base = hardware.get_optimal_thread_count('compute')
        os.environ["CASTSIM_THREAD_BUDGET"] = "3"
        self.assertEqual(hardware.get_optimal_thread_count('compute'), min(base, 3))


class DefaultQueueConfigTests(unittest.TestCase):
    """Env-driven queue configuration."""

    def tearDown(self) -> None:
        for name in (
            "CASTSIM_MAX_CONCURRENT_SIMULATIONS",
            "CASTSIM_WEB_RESERVED_CORES",
            "CASTSIM_WEB_RESERVE_MEMORY_MB",
            "CASTSIM_WEB_MIN_SIM_MEMORY_MB",
        ):
            os.environ.pop(name, None)

    def test_env_overrides(self) -> None:
        os.environ["CASTSIM_MAX_CONCURRENT_SIMULATIONS"] = "3"
        os.environ["CASTSIM_WEB_RESERVED_CORES"] = "2"
        os.environ["CASTSIM_WEB_RESERVE_MEMORY_MB"] = "4096"
        os.environ["CASTSIM_WEB_MIN_SIM_MEMORY_MB"] = "1024"
        config = default_queue_config()
        self.assertEqual(config.max_concurrent_simulations, 3)
        self.assertEqual(config.reserved_cores, 2)
        self.assertEqual(config.reserve_memory_mb, 4096)
        self.assertEqual(config.min_sim_memory_mb, 1024)

    def test_auto_default_by_cores(self) -> None:
        physical = int(getattr(hardware, "physical_cores", 0) or (os.cpu_count() or 4))
        expected = max(1, min(4, physical // 4))
        config = default_queue_config()
        self.assertEqual(config.max_concurrent_simulations, expected)
        self.assertEqual(config.reserved_cores, 1)

    def test_invalid_env_falls_back(self) -> None:
        os.environ["CASTSIM_MAX_CONCURRENT_SIMULATIONS"] = "not-a-number"
        config = default_queue_config()
        self.assertGreaterEqual(config.max_concurrent_simulations, 1)


if __name__ == "__main__":
    unittest.main()
