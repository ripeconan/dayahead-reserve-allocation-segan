#!/usr/bin/env python3
"""Recompute the recovery-confirmatory summary from retained bootstrap draws."""

from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy_uq.inference import (  # noqa: E402
    empirical_two_sided_pvalue,
    holm_adjust,
    linear_quantile,
)


def read_draws(filename: str) -> dict[str, list[float]]:
    path = ROOT / "artifacts/confirmatory" / filename
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        name: [float(row[name]) for row in rows]
        for name in (rows[0] if rows else {})
    }


def close(actual: float, expected: float, tolerance: float = 1e-14) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"expected {expected}, found {actual}")


def reproduce() -> dict[str, object]:
    reference = json.loads(
        (ROOT / "artifacts/confirmatory/inference_summary.json").read_text(
            encoding="utf-8"
        )
    )
    paired = json.loads(
        (ROOT / "artifacts/confirmatory/paired_differences.json").read_text(
            encoding="utf-8"
        )
    )
    block2 = read_draws("bootstrap_means_block2.csv.gz")
    block3 = read_draws("bootstrap_means_block3.csv.gz")
    iid = read_draws("bootstrap_means_iid.csv.gz")
    null = read_draws("bootstrap_means_pvalue.csv.gz")

    reconstructed = []
    raw_pvalues = []
    for record in reference["records"]:
        contrast = record["contrast_id"]
        differences = paired.get(contrast)
        if differences is None:
            reconstructed.append({"contrast_id": contrast, "status": "NOT_EVALUABLE"})
            raw_pvalues.append(1.0)
            continue
        effect = sum(differences) / len(differences)
        primary_ci = [
            linear_quantile(block2[contrast], 0.025),
            linear_quantile(block2[contrast], 0.975),
        ]
        sensitivity_ci = [
            linear_quantile(block3[contrast], 0.025),
            linear_quantile(block3[contrast], 0.975),
        ]
        descriptive_ci = [
            linear_quantile(iid[contrast], 0.025),
            linear_quantile(iid[contrast], 0.975),
        ]
        pvalue = empirical_two_sided_pvalue(null[contrast], effect)

        close(effect, float(record["effect"]))
        close(primary_ci[0], float(record["primary_ci_block2"]["ci_low"]))
        close(primary_ci[1], float(record["primary_ci_block2"]["ci_high"]))
        close(sensitivity_ci[0], float(record["sensitivity_ci_block3"]["ci_low"]))
        close(sensitivity_ci[1], float(record["sensitivity_ci_block3"]["ci_high"]))
        close(descriptive_ci[0], float(record["iid_descriptive_ci"]["ci_low"]))
        close(descriptive_ci[1], float(record["iid_descriptive_ci"]["ci_high"]))
        close(pvalue, float(record["p_for_holm"]))

        raw_pvalues.append(pvalue)
        reconstructed.append(
            {
                "contrast_id": contrast,
                "status": "EVALUABLE",
                "effect": effect,
                "primary_block2_ci": primary_ci,
                "sensitivity_block3_ci": sensitivity_ci,
                "iid_descriptive_ci": descriptive_ci,
                "p_for_holm": pvalue,
            }
        )

    adjusted = holm_adjust(raw_pvalues)
    for value, expected in zip(adjusted, reference["family"]["adjusted_pvalues"]):
        close(value, float(expected))
    for record, adjusted_value in zip(reconstructed, adjusted):
        record["holm_adjusted_pvalue"] = adjusted_value

    result = {
        "status": "PASS",
        "n_bootstrap_draws": len(next(iter(block2.values()))),
        "family_size": len(reconstructed),
        "records": reconstructed,
    }
    output = ROOT / "derived/confirmatory_reproduction.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(reproduce(), indent=2))

