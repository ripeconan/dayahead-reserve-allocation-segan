#!/usr/bin/env python3
"""Paired origin differences and block-bootstrap inference for R78."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from card10_energy_baselines.r78.inference.bootstrap import (
    paired_inference_table,
    paired_origin_differences,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/experiment_outputs/card10_r78_energy_baselines_2026-07-18",
    )
    args = parser.parse_args()
    out = Path(args.output_dir)

    point = pd.read_csv(out / "point_metrics_per_origin.csv")
    point = point[point["status"] == "PASS"]
    diffs = []
    for metric in ("mae", "rmse", "wape"):
        if metric not in point.columns:
            continue
        d = paired_origin_differences(
            point,
            metric=metric,
            baseline="nyiso_p7_official",
            candidates=[
                "ridge_lag2",
                "lightgbm_energy",
                "catboost_energy_quantile",
                "seasonal_naive_d7",
            ],
        )
        diffs.append(d)
    # Also contrast ridge vs lightgbm for best energy model selection evidence.
    d2 = paired_origin_differences(
        point,
        metric="mae",
        baseline="ridge_lag2",
        candidates=["lightgbm_energy", "catboost_energy_quantile"],
    )
    diffs.append(d2)
    diff_df = pd.concat(diffs, ignore_index=True) if diffs else pd.DataFrame()
    diff_df.to_csv(out / "paired_daily_differences.csv.gz", index=False, compression="gzip")
    # Note: differences here are origin-level (primary units), not hour-level.

    inf = paired_inference_table(diff_df, n_boot=10000, blocks=(7, 14))
    # For origin-level series, block lengths 7/14 are longer than n_origins=7;
    # circular bootstrap still runs but is disclosed as origin-level.
    inf["unit"] = "origin"
    inf["note"] = "primary units are origins; block lengths apply to origin sequence"
    inf.to_csv(out / "paired_inference.csv", index=False)

    # Common-UQ paired if available.
    saocp_path = out / "common_saocp_metrics_per_origin.csv"
    if saocp_path.is_file():
        sa = pd.read_csv(saocp_path)
        sa = sa[sa["status"].astype(str).str.contains("PASS|STATIC", regex=True)]
        if "service_worst7" in sa.columns and len(sa):
            sd = paired_origin_differences(
                sa,
                metric="service_worst7",
                baseline="nyiso_p7_official",
                candidates=["ridge_lag2", "lightgbm_energy", "catboost_energy_quantile"],
            )
            sd.to_csv(out / "paired_common_uq_differences.csv", index=False)
            si = paired_inference_table(sd, n_boot=10000, blocks=(7, 14))
            si.to_csv(out / "paired_common_uq_inference.csv", index=False)
    print(f"wrote paired tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
