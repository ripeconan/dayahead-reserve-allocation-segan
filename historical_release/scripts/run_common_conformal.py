#!/usr/bin/env python3
"""Run the common OAS-region, verified SAOCP, and lag-2 controller pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from energy_uq.common_conformal import (  # noqa: E402
    BASELINE_MODEL,
    ORIGIN_LABELS,
    evaluate_model_origin,
    load_predictions,
    policy_outputs,
)

DEFAULT_MODELS = (
    "seasonal_naive_d7",
    "nyiso_p7_official",
    "ridge_lag2",
    "lightgbm_energy",
    "catboost_energy_quantile",
)
DEFAULT_CONTROLLER_MODELS = ("ridge_lag2", "lightgbm_energy")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".gz":
        frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    else:
        frame.to_csv(path, index=False)


def aggregate_metrics(per_origin: pd.DataFrame) -> pd.DataFrame:
    passed = per_origin[per_origin["status"] == "PASS"].copy()
    rows: list[dict[str, object]] = []
    excluded = {"model", "origin", "status", "reference_origin", "time_semantics"}
    for model, current in passed.groupby("model", sort=True):
        row: dict[str, object] = {
            "model": model,
            "n_origins": int(current["origin"].nunique()),
            "status": "PASS_COMMON_ORIGINS",
        }
        for column in current.columns:
            if column in excluded or not pd.api.types.is_numeric_dtype(current[column]):
                continue
            row[column] = float(current[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--saocp-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--origins", nargs="+", default=list(ORIGIN_LABELS))
    parser.add_argument("--controller-models", nargs="+", default=list(DEFAULT_CONTROLLER_MODELS))
    args = parser.parse_args()

    unknown_origins = sorted(set(args.origins) - set(ORIGIN_LABELS))
    if unknown_origins:
        raise ValueError(f"unknown origins: {unknown_origins}")
    unknown_controller = sorted(set(args.controller_models) - set(args.models))
    if unknown_controller:
        raise ValueError(f"controller models absent from --models: {unknown_controller}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("output directory must be new and empty")
    predictions = args.predictions.resolve()
    saocp_repo = args.saocp_repo.resolve()
    started = time.perf_counter()
    frame = load_predictions(predictions, args.models)

    hourly_parts: list[pd.DataFrame] = []
    identity_parts: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []
    policy_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    controller_models = set(args.controller_models)
    for model in args.models:
        for origin in args.origins:
            print(f"evaluating {model} {origin}", flush=True)
            detail, identity, status = evaluate_model_origin(frame, model, origin, saocp_repo)
            status_rows.append(status)
            if status["status"] != "PASS":
                continue
            hourly_parts.append(detail)
            identity_parts.append(identity)
            policy, daily = policy_outputs(detail, model, origin, model in controller_models)
            policy_parts.append(policy)
            daily_parts.append(daily)

    hourly = pd.concat(hourly_parts, ignore_index=True) if hourly_parts else pd.DataFrame()
    identities = pd.concat(identity_parts, ignore_index=True) if identity_parts else pd.DataFrame()
    status = pd.DataFrame(status_rows)
    policy = pd.concat(policy_parts, ignore_index=True) if policy_parts else pd.DataFrame()
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    aggregate = aggregate_metrics(status)

    write_csv(hourly, output / "common_hourly_trace.csv.gz")
    write_csv(identities, output / "saocp_prefix_identity.csv")
    write_csv(status, output / "common_metrics_per_origin.csv")
    write_csv(aggregate, output / "common_metrics_aggregate.csv")
    write_csv(policy, output / "policy_metrics_per_origin.csv")
    write_csv(daily, output / "daily_policy_trace.csv.gz")

    files = {}
    for path in sorted(output.iterdir()):
        if path.is_file():
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema": "energy_uq_common_conformal_v1",
        "input_point_predictions": {
            "filename": predictions.name,
            "bytes": predictions.stat().st_size,
            "sha256": sha256_file(predictions),
        },
        "models": list(args.models),
        "origins": list(args.origins),
        "controller_models": list(args.controller_models),
        "files": files,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "pass_cells": int((status["status"] == "PASS").sum()),
                "blocked_cells": int((status["status"] != "PASS").sum()),
                "hourly_rows": len(hourly),
                "daily_rows": len(daily),
                "output_dir": str(output),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

