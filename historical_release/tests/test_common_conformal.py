from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy_uq.common_conformal import (  # noqa: E402
    assign_segment_ids,
    day_partition,
    parse_local_timestamp,
)
from energy_uq.geometry import conformal_quantile, fit_ellipsoid  # noqa: E402
from energy_uq.saocp_stream import run_verified_stream  # noqa: E402


class CommonConformalTests(unittest.TestCase):
    def test_conformal_quantile_uses_finite_sample_rank(self) -> None:
        values = np.arange(10, dtype=float)
        self.assertEqual(conformal_quantile(values, 0.8), 8.0)

    def test_oas_ellipsoid_is_finite(self) -> None:
        values = np.arange(44, dtype=float).reshape(11, 4)
        model = fit_ellipsoid(values, 0.9)
        self.assertEqual(model.covariance.shape, (4, 4))
        self.assertTrue(np.isfinite(model.precision).all())
        self.assertGreater(model.threshold, 0.0)

    def test_local_day_partition_handles_dst(self) -> None:
        timestamps = pd.Series(
            [
                "2020-11-01T00:00:00-04:00",
                "2020-11-01T01:00:00-04:00",
                "2020-11-01T01:00:00-05:00",
                "2020-11-01T02:00:00-05:00",
                "2020-11-02T00:00:00-05:00",
            ]
        )
        local = parse_local_timestamp(timestamps)
        positions, labels = day_partition(local)
        self.assertEqual(labels, ["2020-11-01", "2020-11-02"])
        self.assertEqual([len(part) for part in positions], [4, 1])

    def test_segments_split_on_calendar_gaps(self) -> None:
        segments = assign_segment_ids(["2020-01-01", "2020-01-02", "2020-01-05"])
        self.assertEqual(segments.tolist(), [0, 0, 1])

    @unittest.skipUnless(os.environ.get("ENERGY_UQ_SAOCP_REPO"), "pinned checkout not configured")
    def test_verified_saocp_reference_hash(self) -> None:
        result = run_verified_stream(
            repo=Path(os.environ["ENERGY_UQ_SAOCP_REPO"]),
            calibration_prefix=[1.0, 2.0, 3.0, 4.0, 5.0],
            scores=[1.5, 2.5, 3.5, 4.5],
            segment_ids=[0, 0, 0, 0],
            feedback_lag=2,
        )
        self.assertEqual(
            result["pre_hash"],
            "8252a371b0e6b5707dc1b1bddb11f2a7ada6a5390967f38924a7a5cf4ba5149b",
        )


if __name__ == "__main__":
    unittest.main()

