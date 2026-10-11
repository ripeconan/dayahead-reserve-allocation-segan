from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reproduce_artifacts", ROOT / "scripts/reproduce_artifacts.py"
)
assert SPEC and SPEC.loader
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)

CONFIRM_SPEC = importlib.util.spec_from_file_location(
    "reproduce_confirmatory", ROOT / "scripts/reproduce_confirmatory.py"
)
assert CONFIRM_SPEC and CONFIRM_SPEC.loader
CONFIRM = importlib.util.module_from_spec(CONFIRM_SPEC)
CONFIRM_SPEC.loader.exec_module(CONFIRM)


class ArtifactReplayTests(unittest.TestCase):
    def test_expected_shapes(self) -> None:
        observed = REPLAY.verify_shapes()
        self.assertEqual(observed["baseline_matrix/common_shell_raw.csv.gz"], 32)

    def test_controller_matches_retained_trace(self) -> None:
        result = REPLAY.verify_daily_controller_trace()
        self.assertEqual(result["origins"], 7)
        self.assertGreater(result["days"], 2000)

    def test_headline_results(self) -> None:
        results = REPLAY.headline_rows()
        self.assertGreaterEqual(len(results), 9)

    def test_confirmatory_bootstrap_reconstruction(self) -> None:
        result = CONFIRM.reproduce()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["n_bootstrap_draws"], 20000)


if __name__ == "__main__":
    unittest.main()
