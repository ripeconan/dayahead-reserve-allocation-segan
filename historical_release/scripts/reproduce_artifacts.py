#!/usr/bin/env python3
"""Verify frozen objects and reconstruct the paper's headline results."""

from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy_uq.reliability_shield import StrictLag2ReliabilityShield  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require_close(actual: float, expected: float, *, atol: float = 5e-7) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
        raise AssertionError(f"expected {expected}, found {actual}")


def verify_shapes() -> dict[str, int]:
    expected = {
        "baseline_matrix/common_shell_raw.csv.gz": 32,
        "baseline_matrix/coverage_matched.csv.gz": 32,
        "baseline_matrix/static_volume_matched.csv.gz": 32,
        "baseline_matrix/action_layer.csv.gz": 60,
        "causal_comparator/nyiso_strong_comparator.csv.gz": 4,
        "verified_saocp/selected_per_origin_results.csv": 14,
        "fair_deep/deepar/policy_metrics_per_origin.csv": 18,
        "fair_deep/multimodel/policy_metrics_per_origin.csv": 54,
    }
    observed = {}
    for relative, count in expected.items():
        rows = read_csv(ROOT / "artifacts" / relative)
        observed[relative] = len(rows)
        if len(rows) != count:
            raise AssertionError(f"{relative}: expected {count} rows, found {len(rows)}")
    return observed


def verify_daily_controller_trace() -> dict[str, int]:
    rows = read_csv(ROOT / "artifacts/verified_saocp/daily_policy_trace.csv.gz")
    origins = sorted({row["origin"] for row in rows})
    checked_days = 0
    checked_actions = 0
    for origin in origins:
        base = sorted(
            (
                row
                for row in rows
                if row["origin"] == origin and row["layer"] == "base_no_fallback"
            ),
            key=lambda row: row["delivery_date"],
        )
        served = sorted(
            (
                row
                for row in rows
                if row["origin"] == origin
                and row["layer"] == "base_plus_frozen_drb_lag2"
            ),
            key=lambda row: row["delivery_date"],
        )
        if [row["delivery_date"] for row in base] != [row["delivery_date"] for row in served]:
            raise AssertionError(f"date mismatch for {origin}")
        controller = StrictLag2ReliabilityShield(len(base))
        replay = controller.replay(
            [float(row["base_daily_coverage"]) for row in base],
            [row["segment_id"] for row in base],
        )
        for result, target in zip(replay, served):
            target_action = target["action"].lower() == "true"
            if result.action != target_action:
                raise AssertionError(
                    f"action mismatch for {origin} on {target['delivery_date']}"
                )
            require_close(result.base_coverage, float(target["base_daily_coverage"]), atol=1e-12)
            target_service = 1.0 if target_action else float(target["base_daily_service"])
            require_close(result.service, target_service, atol=1e-12)
            checked_days += 1
            checked_actions += int(result.action)
    return {"origins": len(origins), "days": checked_days, "actions": checked_actions}


def headline_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    comparator = read_csv(
        ROOT / "artifacts/causal_comparator/nyiso_strong_comparator.csv.gz"
    )
    by_policy = {row["policy"]: row for row in comparator}
    drb = by_policy["drb_reactive_lag2"]
    cocd = by_policy["cocd_one_credit_causal_path"]
    drb_total_actions = int(float(drb["action_total"]) * int(drb["n_origins"]))
    cocd_total_actions = int(float(cocd["action_total"]) * int(cocd["n_origins"]))
    if not (drb_total_actions == cocd_total_actions == 280):
        raise AssertionError("unexpected exact-action-count comparison")
    require_close(float(drb["service_worst7"]), 0.730442, atol=5e-7)
    require_close(float(cocd["service_worst7"]), 0.702381, atol=5e-7)
    rows.extend(
        [
            {
                "result": "same-base DRB worst-7",
                "value": float(drb["service_worst7"]),
                "source": "causal_comparator/nyiso_strong_comparator.csv.gz",
            },
            {
                "result": "same-base COCD worst-7",
                "value": float(cocd["service_worst7"]),
                "source": "causal_comparator/nyiso_strong_comparator.csv.gz",
            },
            {
                "result": "matched actions per policy",
                "value": drb_total_actions,
                "source": "causal_comparator/nyiso_strong_comparator.csv.gz",
            },
        ]
    )

    confirmatory = json.loads(
        (ROOT / "artifacts/confirmatory/inference_summary.json").read_text(encoding="utf-8")
    )
    records = {row["contrast_id"]: row for row in confirmatory["records"]}
    c1 = records["C1_D1_minus_BASE"]
    require_close(float(c1["effect"]), 0.0143849206, atol=5e-11)
    require_close(float(c1["holm_adjusted_pvalue"]), 0.00704965, atol=5e-9)
    rows.extend(
        [
            {
                "result": "recovery D1 minus base effect",
                "value": float(c1["effect"]),
                "source": "confirmatory/inference_summary.json",
            },
            {
                "result": "recovery D1 minus base Holm p",
                "value": float(c1["holm_adjusted_pvalue"]),
                "source": "confirmatory/inference_summary.json",
            },
        ]
    )

    for group in ("deepar", "multimodel"):
        inference = read_csv(
            ROOT / f"artifacts/fair_deep/{group}/common_paired_inference.csv"
        )
        for row in inference:
            if row["endpoint"] == "mean_service" and "drb_reactive_lag2_minus" in row["contrast"]:
                rows.append(
                    {
                        "result": f"{row['contrast']} mean-service effect",
                        "value": float(row["effect"]),
                        "source": f"fair_deep/{group}/common_paired_inference.csv",
                    }
                )
    return rows


def write_outputs(summary: dict[str, object], results: Iterable[dict[str, object]]) -> None:
    output = ROOT / "derived"
    output.mkdir(exist_ok=True)
    (output / "reproduction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "headline_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["result", "value", "source"])
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    shapes = verify_shapes()
    trace = verify_daily_controller_trace()
    results = headline_rows()
    summary = {
        "status": "PASS",
        "artifact_shapes": shapes,
        "controller_trace": trace,
        "headline_result_count": len(results),
    }
    write_outputs(summary, results)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
