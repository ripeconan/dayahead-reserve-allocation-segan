from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy_uq.reliability_shield import (  # noqa: E402
    StrictLag2ReliabilityShield,
    action_cap,
    rolling_worst_service,
)
from energy_uq.inference import empirical_two_sided_pvalue, holm_adjust, linear_quantile  # noqa: E402


class ReliabilityShieldTests(unittest.TestCase):
    def test_action_cap(self) -> None:
        self.assertEqual(action_cap(100), 15)
        self.assertEqual(action_cap(7), 1)

    def test_strict_lag_two(self) -> None:
        controller = StrictLag2ReliabilityShield(10, budget_fraction=1.0)
        first = controller.step(0, 0.0, "a")
        second = controller.step(1, 1.0, "a")
        third = controller.step(2, 1.0, "a")
        self.assertFalse(first.action)
        self.assertFalse(second.action)
        self.assertTrue(third.action)
        self.assertEqual(third.revealed_indices, (0,))

    def test_history_does_not_cross_segments(self) -> None:
        controller = StrictLag2ReliabilityShield(10, budget_fraction=1.0)
        controller.step(0, 0.0, "a")
        controller.step(1, 0.0, "a")
        result = controller.step(2, 1.0, "b")
        self.assertFalse(result.action)
        self.assertEqual(result.revealed_indices, ())

    def test_budget_is_enforced(self) -> None:
        controller = StrictLag2ReliabilityShield(7, budget_fraction=0.15)
        results = controller.replay([0.0] * 7, ["a"] * 7)
        self.assertEqual(sum(result.action for result in results), 1)

    def test_partial_action_effectiveness(self) -> None:
        controller = StrictLag2ReliabilityShield(
            3, budget_fraction=1.0, action_effectiveness=0.25
        )
        results = controller.replay([0.0, 1.0, 0.0], ["a"] * 3)
        self.assertTrue(results[2].action)
        self.assertEqual(results[2].service, 0.25)

    def test_action_effectiveness_is_bounded(self) -> None:
        with self.assertRaises(ValueError):
            StrictLag2ReliabilityShield(3, action_effectiveness=1.01)

    def test_rolling_windows_stay_within_segment(self) -> None:
        service = [0.0, 1.0, 1.0, 0.0]
        segments = ["a", "a", "b", "b"]
        self.assertEqual(rolling_worst_service(service, segments, 2), 0.5)
        with self.assertRaises(ValueError):
            rolling_worst_service(service, segments, 3)

    def test_inference_utilities(self) -> None:
        self.assertEqual(linear_quantile([0.0, 1.0], 0.25), 0.25)
        self.assertEqual(empirical_two_sided_pvalue([0.0, 1.0, -1.0], 0.5), 0.75)
        self.assertEqual(holm_adjust([0.01, 0.04, 1.0]), [0.03, 0.08, 1.0])


if __name__ == "__main__":
    unittest.main()
