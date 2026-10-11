#!/usr/bin/env python3
"""Paired daily point-error inference for the four frozen R86 winners."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE = "nyiso_p7_official"
COMPARATORS = ("ridge_lag2", BASELINE)
SEED = 20260720
KEY = ["origin", "timestamp", "zone"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def circular_indices(length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    block = max(1, min(block, length))
    starts = rng.integers(0, length, size=math.ceil(length / block))
    return np.concatenate([(start + np.arange(block)) % length for start in starts])[:length]


def segment_ids(dates: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(dates, format="%Y-%m-%d", errors="raise")
    return parsed.diff().dt.days.ne(1).cumsum().to_numpy(int) - 1


def holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (len(values) - rank)))
        adjusted[index] = running
    return adjusted


def contrast_seed(candidate: str, comparator: str, offset: int) -> int:
    key = hashlib.sha256(f"{candidate}|{comparator}".encode()).digest()
    return SEED + int.from_bytes(key[:4], "big") % 1_000_000 + offset


def paired_bootstrap(
    daily: pd.DataFrame, candidate: str, comparator: str, n_boot: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    current = daily[
        (daily["candidate"] == candidate) & (daily["comparator"] == comparator)
    ].copy()
    origins = sorted(current["origin"].astype(str).unique())
    if len(origins) != 7:
        raise RuntimeError(f"point contrast lacks seven origins: {candidate} vs {comparator}")
    stored: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    origin_rows = []
    for origin in origins:
        part = current[current["origin"].astype(str) == origin].sort_values("delivery_date")
        difference = part["daily_mae_difference"].to_numpy(float)
        segments = part["segment_id"].to_numpy(int)
        stored[origin] = (difference, segments)
        origin_rows.append(
            {
                "candidate": candidate, "comparator": comparator, "origin": origin,
                "n_days": int(len(part)), "n_zone_hours": int(part["n_zone_hours"].sum()),
                "candidate_mae": float(np.average(part["candidate_mae"], weights=part["n_zone_hours"])),
                "comparator_mae": float(np.average(part["comparator_mae"], weights=part["n_zone_hours"])),
                "mean_daily_mae_difference": float(difference.mean()),
            }
        )
    per_origin = pd.DataFrame(origin_rows)
    observed = per_origin["mean_daily_mae_difference"].to_numpy(float)
    iid_rng = np.random.default_rng(contrast_seed(candidate, comparator, 101))
    iid = iid_rng.choice(observed, size=(n_boot, len(observed)), replace=True).mean(axis=1)
    bootstraps = {}
    for block in (7, 14):
        rng = np.random.default_rng(contrast_seed(candidate, comparator, block * 1009))
        boot = np.empty(n_boot, dtype=float)
        for replicate in range(n_boot):
            selected = rng.choice(origins, size=len(origins), replace=True)
            effects = []
            for origin in selected:
                values, segments = stored[str(origin)]
                samples = []
                for segment in pd.unique(segments):
                    part = values[segments == segment]
                    samples.append(part[circular_indices(len(part), block, rng)])
                effects.append(float(np.concatenate(samples).mean()))
            boot[replicate] = float(np.mean(effects))
        bootstraps[block] = boot
    primary = bootstraps[7]
    p_value = min(
        1.0,
        2.0 * min(
            (float(np.sum(primary <= 0.0)) + 1.0) / (n_boot + 1.0),
            (float(np.sum(primary >= 0.0)) + 1.0) / (n_boot + 1.0),
        ),
    )
    return per_origin, {
        "endpoint": "mean_daily_mae", "contrast": f"{candidate}_minus_{comparator}",
        "candidate": candidate, "comparator": comparator,
        "effect": float(observed.mean()), "direction": "negative_favors_candidate",
        "n_origins": len(origins), "n_boot": n_boot,
        "iid_origin_ci_lo": float(np.quantile(iid, 0.025)),
        "iid_origin_ci_hi": float(np.quantile(iid, 0.975)),
        "cluster_day_block7_ci_lo": float(np.quantile(primary, 0.025)),
        "cluster_day_block7_ci_hi": float(np.quantile(primary, 0.975)),
        "cluster_day_block14_ci_lo": float(np.quantile(bootstraps[14], 0.025)),
        "cluster_day_block14_ci_hi": float(np.quantile(bootstraps[14], 0.975)),
        "p_raw_empirical_block7": p_value,
        "inference_semantics": (
            "paired daily absolute-error differences; circular day blocks within contiguous "
            "segment and origin; origins resampled as clusters"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r78-predictions", type=Path, required=True)
    parser.add_argument("--r86-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("R86 point-inference output directory must not exist")

    r86 = pd.read_csv(args.r86_predictions)
    candidates = sorted(set(r86["model"].astype(str)))
    if len(candidates) != 4 or BASELINE in candidates:
        raise RuntimeError("R86 point input must contain exactly four frozen winners")
    base = pd.read_csv(args.r78_predictions)
    if not set(COMPARATORS).issubset(set(base["model"].astype(str))):
        raise RuntimeError("R78 point input lacks Ridge or P7")
    daily_parts = []
    support_rows = []
    for candidate in candidates:
        left = r86[r86["model"].astype(str) == candidate].copy()
        left = left[left["eligibility"].astype(str) == "eligible"]
        left = left[np.isfinite(left[["actual", "point"]].to_numpy(float)).all(axis=1)]
        if left.duplicated(KEY).any():
            raise RuntimeError(f"duplicate R86 keys: {candidate}")
        for comparator in COMPARATORS:
            right = base[base["model"].astype(str) == comparator].copy()
            right = right[right["eligibility"].astype(str) == "eligible"]
            right = right[np.isfinite(right[["actual", "point"]].to_numpy(float)).all(axis=1)]
            if right.duplicated(KEY).any():
                raise RuntimeError(f"duplicate R78 keys: {comparator}")
            merged = left[KEY + ["delivery_date", "actual", "point"]].merge(
                right[KEY + ["actual", "point"]], on=KEY, how="inner",
                suffixes=("_candidate", "_comparator"), validate="one_to_one",
            )
            if not np.allclose(merged["actual_candidate"], merged["actual_comparator"], atol=1e-9, rtol=0):
                raise RuntimeError(f"actual mismatch: {candidate} vs {comparator}")
            merged["candidate_abs_error"] = np.abs(merged["actual_candidate"] - merged["point_candidate"])
            merged["comparator_abs_error"] = np.abs(merged["actual_comparator"] - merged["point_comparator"])
            daily = merged.groupby(["origin", "delivery_date"], sort=True).agg(
                candidate_mae=("candidate_abs_error", "mean"),
                comparator_mae=("comparator_abs_error", "mean"),
                n_zone_hours=("zone", "size"),
            ).reset_index()
            daily["daily_mae_difference"] = daily["candidate_mae"] - daily["comparator_mae"]
            daily["candidate"] = candidate
            daily["comparator"] = comparator
            daily["segment_id"] = daily.groupby("origin", sort=False)["delivery_date"].transform(segment_ids)
            daily_parts.append(daily)
            support_rows.append(
                {"candidate": candidate, "comparator": comparator,
                 "candidate_eligible_rows": int(len(left)), "comparator_eligible_rows": int(len(right)),
                 "paired_rows": int(len(merged)), "paired_days": int(len(daily)),
                 "origins": int(daily["origin"].nunique())}
            )

    daily_all = pd.concat(daily_parts, ignore_index=True)
    per_origin_parts, inference_rows = [], []
    for candidate in candidates:
        for comparator in COMPARATORS:
            per_origin, record = paired_bootstrap(daily_all, candidate, comparator, args.n_boot)
            per_origin_parts.append(per_origin)
            inference_rows.append(record)
    inference = pd.DataFrame(inference_rows)
    inference["p_holm"] = holm(inference["p_raw_empirical_block7"].to_numpy(float))
    per_origin_all = pd.concat(per_origin_parts, ignore_index=True)
    support = pd.DataFrame(support_rows)

    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "paired_daily_mae.csv.gz": daily_all,
        "paired_mae_per_origin.csv": per_origin_all,
        "paired_mae_inference.csv": inference,
        "pairwise_support.csv": support,
    }
    for name, frame in outputs.items():
        compression = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
        frame.to_csv(output / name, index=False, compression=compression)
    manifest = {
        "schema": "card10_r86_fair_deep_point_paired_inference_v1",
        "scientific_status": "exposed_evaluation_posthoc_inference_not_selection",
        "inputs": {
            "r78_predictions": {"path": str(args.r78_predictions.resolve()), "sha256": sha256_file(args.r78_predictions)},
            "r86_predictions": {"path": str(args.r86_predictions.resolve()), "sha256": sha256_file(args.r86_predictions)},
        },
        "candidates": candidates, "comparators": list(COMPARATORS),
        "n_boot": args.n_boot, "seed": SEED,
        "outputs": {name: {"bytes": (output / name).stat().st_size, "sha256": sha256_file(output / name)} for name in outputs},
        "access_flags": {
            "confirmatory_2026_outcome_accessed": False,
            "confirmatory_2026_forecast_payload_accessed": False,
            "reserved_confirmatory_path_accessed": False,
            "r77_d2_values_accessed": False, "sota_claimed": False,
        },
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(inference.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
