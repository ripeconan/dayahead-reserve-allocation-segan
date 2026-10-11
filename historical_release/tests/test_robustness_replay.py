from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reproduce_robustness", ROOT / "scripts/reproduce_robustness.py"
)
assert SPEC and SPEC.loader
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class RobustnessReplayTests(unittest.TestCase):
    def test_frozen_objects(self) -> None:
        result = REPLAY.verify_frozen_objects()
        self.assertEqual(result["production_transfer/daily_trace.csv.gz"], 8376)

    def test_transfer_trace(self) -> None:
        result = REPLAY.verify_transfer_trace()
        self.assertEqual(result["origins"], 23)
        self.assertEqual(result["days"], 2094)

    def test_hyperparameter_reference(self) -> None:
        result = REPLAY.verify_hyperparameter_reference()
        self.assertEqual(result["configurations"], 7)


if __name__ == "__main__":
    unittest.main()
