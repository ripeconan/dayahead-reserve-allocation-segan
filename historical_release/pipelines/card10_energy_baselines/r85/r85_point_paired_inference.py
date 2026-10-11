#!/usr/bin/env python3
"""Pair fair DeepAR point errors with the admitted R78 energy baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEEPAR = "deepar_fair_p7_residual"
COMPARATORS = (
    "ridge_lag2",
    "lightgbm_energy",
    "catboost_energy_quantile",
    "nyiso_p7_official",
)
SEED = 20260718
KEY = ["origin", "timestamp", "zone"]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def circular_indices(length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    block = max(1, min(block, length))
    starts = rng.integers(0, length, size=math.ceil(length / block))
    return np.concatenate(
        [(start + np.arange(block)) % length for start in starts]
    )[:length]


def segment_ids(dates: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(dates, format="%Y-%m-%d", errors="raise")
    return parsed.diff().dt.days.ne(1).cumsum().to_numpy(int) - 1


def holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (len(values) - rank)))
        result[index] = running
    return result


def paired_bootstrap(
    daily: pd.DataFrame, comparator: str, n_boot: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    current = daily[daily["comparator"] == comparator].copy()
    origins = sorted(current["origin"].unique())
    stored: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    origin_rows = []
    for origin in origins:
        part = current[current["origin"] == origin].sort_values("delivery_date")
        difference = part["daily_mae_difference"].to_numpy(float)
        segments = part["segment_id"].to_numpy(int)
        stored[str(origin)] = (difference, segments)
        origin_rows.append(
            {
                "comparator": comparator,
                "origin": origin,
                "n_days": int(len(part)),
                "n_zone_hours": int(part["n_zone_hours"].sum()),
                "deepar_mae": float(
                    np.average(part["deepar_mae"], weights=part["n_zone_hours"])
                ),
                "comparator_mae": float(
                    np.average(part["comparator_mae"], weights=part["n_zone_hours"])
                ),
                "mean_daily_mae_difference": float(difference.mean()),
            }
        )
    per_origin = pd.DataFrame(origin_rows)
    observed = per_origin["mean_daily_mae_difference"].to_numpy(float)
    rng_iid = np.random.default_rng(SEED + 101)
    iid = rng_iid.choice(
        observed, size=(n_boot, len(observed)), replace=True
    ).mean(axis=1)
    blocks: dict[int, np.ndarray] = {}
    for block in (7, 14):
        rng = np.random.default_rng(SEED + block * 1009)
        boot = np.empty(n_boot, dtype=float)
        for replicate in range(n_boot):
            selected = rng.choice(origins, size=len(origins), replace=True)
            effects = []
            for origin in selected:
                values, segments = stored[str(origin)]
                sampled = []
                for segment in pd.unique(segments):
                    part = values[segments == segment]
                    sampled.append(part[circular_indices(len(part), block, rng)])
                effects.append(float(np.concatenate(sampled).mean()))
            boot[replicate] = float(np.mean(effects))
        blocks[block] = boot
    primary = blocks[7]
    p_value = min(
        1.0,
        2.0
        * min(
            (float(np.sum(primary <= 0.0)) + 1.0) / (len(primary) + 1.0),
            (float(np.sum(primary >= 0.0)) + 1.0) / (len(primary) + 1.0),
        ),
    )
    record: dict[str, object] = {
        "endpoint": "mean_daily_mae",
        "contrast": f"{DEEPAR}_minus_{comparator}",
        "effect": float(observed.mean()),
        "direction": "negative_favors_deepar",
        "n_origins": len(origins),
        "n_boot": n_boot,
        "iid_origin_ci_lo": float(np.quantile(iid, 0.025)),
        "iid_origin_ci_hi": float(np.quantile(iid, 0.975)),
        "cluster_day_block7_ci_lo": float(np.quantile(blocks[7], 0.025)),
        "cluster_day_block7_ci_hi": float(np.quantile(blocks[7], 0.975)),
        "cluster_day_block14_ci_lo": float(np.quantile(blocks[14], 0.025)),
        "cluster_day_block14_ci_hi": float(np.quantile(blocks[14], 0.975)),
        "p_raw_empirical_block7": p_value,
        "inference_semantics": (
            "paired daily absolute-error differences; circular day blocks within "
            "contiguous segment and origin; origins resampled as clusters"
        ),
    }
    return per_origin, record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r78-predictions", type=Path, required=True)
    parser.add_argument("--r85-combined", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("paired-inference output directory must not exist")

    deep_all = pd.read_csv(args.r85_combined)
    deep = deep_all[deep_all["model"].astype(str) == DEEPAR].copy()
    base = pd.read_csv(args.r78_predictions)
    if deep.empty or not set(COMPARATORS).issubset(set(base["model"].astype(str))):
        raise RuntimeError("required DeepAR/comparator predictions are absent")
    deep = deep[deep["eligibility"].astype(str) == "eligible"]
    deep = deep[np.isfinite(deep[["actual", "point"]].to_numpy(float)).all(axis=1)]
    if deep.duplicated(KEY).any():
        raise RuntimeError("duplicate DeepAR keys")

    daily_parts = []
    support_rows = []
    for comparator in COMPARATORS:
        right = base[base["model"].astype(str) == comparator].copy()
        right = right[right["eligibility"].astype(str) == "eligible"]
        right = right[np.isfinite(right[["actual", "point"]].to_numpy(float)).all(axis=1)]
        if right.duplicated(KEY).any():
            raise RuntimeError(f"duplicate comparator keys: {comparator}")
        merged = deep[KEY + ["delivery_date", "actual", "point"]].merge(
            right[KEY + ["actual", "point"]],
            on=KEY,
            how="inner",
            suffixes=("_deepar", "_comparator"),
            validate="one_to_one",
        )
        if not np.allclose(
            merged["actual_deepar"], merged["actual_comparator"], rtol=0.0, atol=1e-9
        ):
            raise RuntimeError(f"actual mismatch after pairing: {comparator}")
        merged["deepar_abs_error"] = np.abs(
            merged["actual_deepar"] - merged["point_deepar"]
        )
        merged["comparator_abs_error"] = np.abs(
            merged["actual_comparator"] - merged["point_comparator"]
        )
        daily = (
            merged.groupby(["origin", "delivery_date"], sort=True)
            .agg(
                deepar_mae=("deepar_abs_error", "mean"),
                comparator_mae=("comparator_abs_error", "mean"),
                n_zone_hours=("zone", "size"),
            )
            .reset_index()
        )
        daily["daily_mae_difference"] = daily["deepar_mae"] - daily["comparator_mae"]
        daily["comparator"] = comparator
        daily["segment_id"] = daily.groupby("origin", sort=False)[
            "delivery_date"
        ].transform(segment_ids)
        daily_parts.append(daily)
        support_rows.append(
            {
                "comparator": comparator,
                "deepar_eligible_rows": int(len(deep)),
                "comparator_eligible_rows": int(len(right)),
                "paired_rows": int(len(merged)),
                "paired_days": int(len(daily)),
                "origins": int(daily["origin"].nunique()),
            }
        )

    daily_all = pd.concat(daily_parts, ignore_index=True)
    per_origin_parts = []
    inference_rows = []
    for comparator in COMPARATORS:
        per_origin, record = paired_bootstrap(daily_all, comparator, args.n_boot)
        per_origin_parts.append(per_origin)
        inference_rows.append(record)
    inference = pd.DataFrame(inference_rows)
    inference["p_holm"] = holm(inference["p_raw_empirical_block7"].to_numpy(float))
    per_origin_all = pd.concat(per_origin_parts, ignore_index=True)
    support = pd.DataFrame(support_rows)

    output.mkdir(parents=True, exist_ok=False)
    daily_path = output / "paired_daily_mae.csv.gz"
    per_origin_path = output / "paired_mae_per_origin.csv"
    inference_path = output / "paired_mae_inference.csv"
    support_path = output / "pairwise_support.csv"
    daily_all.to_csv(
        daily_path, index=False, compression={"method": "gzip", "mtime": 0}
    )
    per_origin_all.to_csv(per_origin_path, index=False)
    inference.to_csv(inference_path, index=False)
    support.to_csv(support_path, index=False)
    files = (daily_path, per_origin_path, inference_path, support_path)
    manifest = {
        "schema": "card10_r85_deepar_fair_point_paired_inference_v1",
        "scientific_status": "exposed_evaluation_posthoc_inference_not_selection",
        "inputs": {
            "r78_predictions": {
                "path": str(args.r78_predictions.resolve()),
                "sha256": sha256_path(args.r78_predictions),
            },
            "r85_combined": {
                "path": str(args.r85_combined.resolve()),
                "sha256": sha256_path(args.r85_combined),
            },
        },
        "n_boot": args.n_boot,
        "seed": SEED,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_path(path)}
            for path in files
        },
        "access_flags": {
            "confirmatory_2026_outcome_accessed": False,
            "confirmatory_2026_forecast_payload_accessed": False,
            "reserved_confirmatory_path_accessed": False,
            "r77_d2_values_accessed": False,
            "sota_claimed": False,
        },
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(inference.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
