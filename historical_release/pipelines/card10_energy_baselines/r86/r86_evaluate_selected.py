#!/usr/bin/env python3
"""Evaluate one frozen R86 model/origin cell without re-selection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from card10_energy_baselines.r78.constants import ACCESS_FLAGS, ZONES  # noqa: E402
from card10_energy_baselines.r78.data_loader import load_nyiso_raw  # noqa: E402
from card10_energy_baselines.r78.models import ADAPTERS  # noqa: E402


MODEL_TO_ADAPTER = {
    "TFT": "tft_quantile",
    "NHiTS": "nhits_quantile",
    "PatchTST": "patchtst_quantile",
    "ProLoaF": "proloaf_native",
}
FORBIDDEN = (
    "confirmatory_locked/card10_nyiso_2026h1",
    "card10_confirmatory_unseen/card10_nyiso_2026h1",
    "card10_nyiso_2026h1",
    "gefs_revision_r77_d2_tmprh",
    "r77_d2",
)


def safe_path(path: str | Path) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if any(token in normalized for token in FORBIDDEN):
        raise RuntimeError(f"FORBIDDEN_PATH_BEFORE_ENUMERATION: {path}")
    if "nyiso" in normalized and "2026" in normalized:
        raise RuntimeError(f"NYISO_2026_PATH_BEFORE_ENUMERATION: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def local_boundary(year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-01-01").tz_localize("America/New_York")


def metrics(actual: pd.DataFrame, prediction: Any) -> dict[str, Any]:
    y = actual.loc[prediction.point.index, list(ZONES)].to_numpy(float)
    point = prediction.point.to_numpy(float)
    q05 = prediction.q05.to_numpy(float)
    q50 = prediction.q50.to_numpy(float)
    q95 = prediction.q95.to_numpy(float)
    if not all(np.isfinite(array).all() for array in (y, point, q05, q50, q95)):
        raise RuntimeError("non-finite R86 evaluation values")
    denom = float(np.abs(y).sum())
    normalized = []
    pinballs: dict[str, float] = {}
    for q, name, qhat in ((0.05, "q05", q05), (0.50, "q50", q50), (0.95, "q95", q95)):
        error = y - qhat
        loss = np.maximum(q * error, (q - 1.0) * error)
        pinballs[f"pinball_{name}"] = float(loss.mean())
        normalized.append(float(2.0 * loss.sum() / denom))
    zone_mae = np.abs(y - point).mean(axis=0)
    covered = (y >= q05) & (y <= q95)
    return {
        "n": int(y.size),
        "n_timestamps": int(y.shape[0]),
        "mae": float(np.abs(y - point).mean()),
        "rmse": float(np.sqrt(np.square(y - point).mean())),
        "wape": float(np.abs(y - point).sum() / denom),
        "worst_zone": str(ZONES[int(np.argmax(zone_mae))]),
        "worst_zone_mae": float(zone_mae.max()),
        "mean_pinball": float(np.mean(list(pinballs.values()))),
        "normalized_wql": float(np.mean(normalized)),
        "marginal_coverage_q05_q95": float(covered.mean()),
        "simultaneous_11zone_coverage": float(covered.all(axis=1).mean()),
        "mean_width_q05_q95": float((q95 - q05).mean()),
        **pinballs,
    }


def save_prediction(
    path: Path, model: str, candidate_id: str, origin: str, actual: pd.DataFrame, pred: Any
) -> None:
    parts = []
    for zone in ZONES:
        parts.append(
            pd.DataFrame(
                {
                    "timestamp_utc": pred.point.index.tz_convert("UTC").astype(str),
                    "timestamp_local": pred.point.index.astype(str),
                    "origin": origin,
                    "model": model,
                    "candidate_id": candidate_id,
                    "zone": zone,
                    "actual": actual.loc[pred.point.index, zone].to_numpy(float),
                    "point": pred.point[zone].to_numpy(float),
                    "q05": pred.q05[zone].to_numpy(float),
                    "q50": pred.q50[zone].to_numpy(float),
                    "q95": pred.q95[zone].to_numpy(float),
                }
            )
        )
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="R86 frozen-winner evaluator")
    parser.add_argument("--data", default=None)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--model", choices=list(MODEL_TO_ADAPTER), required=True)
    parser.add_argument("--test-year", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--proloaf-repo",
        default=os.environ.get("PROLOAF_REPO"),
        help="path to the pinned ProLoaF checkout (required for ProLoaF runs)",
    )
    args = parser.parse_args()
    if args.test_year not in range(2019, 2026):
        raise RuntimeError("R86 evaluation test year must be 2019--2025")
    for path in (args.grid, args.selected, args.output_dir):
        safe_path(path)
    if any(ACCESS_FLAGS.values()):
        raise RuntimeError("access flags are not clean")

    grid_path = Path(args.grid).resolve()
    selected_path = Path(args.selected).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected["candidate_grid_file_sha256"] != sha256_file(grid_path):
        raise RuntimeError("selected contract does not bind the supplied candidate grid")
    if any(grid["access_flags"].values()) or any(selected["access_flags"].values()):
        raise RuntimeError("grid/selected access flags are not clean")
    selected_model = selected["models"][args.model]
    if selected_model["status"] != "DEVELOPMENT_SELECTED_NOT_YET_EVALUATED":
        raise RuntimeError(f"{args.model} does not have a frozen development winner")
    candidate = dict(selected_model["selected"]["effective_candidate"])
    if candidate["candidate_id"] != selected_model["selected_candidate_id"]:
        raise RuntimeError("selected candidate identity mismatch")
    frozen_rows = [
        row
        for row in grid["candidates"]
        if row["model"] == args.model and row["candidate_id"] == candidate["candidate_id"]
    ]
    if len(frozen_rows) != 1 or any(
        candidate.get(key) != value
        for key, value in frozen_rows[0].items()
        if key != "model"
    ):
        raise RuntimeError("selected effective candidate differs from frozen grid")
    if args.model == "ProLoaF":
        candidate["repo"] = args.proloaf_repo
        candidate["work_dir"] = str(out_dir / "proloaf_runtime")

    existing_result = out_dir / "result.json"
    if existing_result.exists():
        existing = json.loads(existing_result.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "PASS"
            and existing.get("candidate_grid_sha256") == sha256_file(grid_path)
            and existing.get("selected_contract_sha256") == sha256_file(selected_path)
        ):
            print(f"[R86 evaluation] skip sealed PASS cell {existing['model']} {existing['origin']}")
            return 0
        raise RuntimeError("output directory contains a nonmatching prior result")

    panel = load_nyiso_raw(args.data)
    actual, forecast = panel.actual, panel.forecast
    boundary = local_boundary(args.test_year)
    valid_start = boundary - pd.DateOffset(days=90)
    fit_idx = actual.index[actual.index < valid_start]
    valid_idx = actual.index[(actual.index >= valid_start) & (actual.index < boundary)]
    test_end = local_boundary(args.test_year + 1)
    test_idx = actual.index[(actual.index >= boundary) & (actual.index < test_end)]
    if not set(test_idx.year) == {args.test_year}:
        raise RuntimeError("evaluation timestamp identity mismatch")
    for frame, name in ((actual, "actual"), (forecast, "forecast")):
        if frame.loc[fit_idx.append(valid_idx).append(test_idx)].isna().any().any():
            raise RuntimeError(f"missing {name} in R86 evaluation slices")

    started = time.time()
    if args.model == "TFT":
        # Operational fault recovery is cell-local and deliberately excluded
        # from the frozen candidate/config hash. The adapter only consults this
        # opt-in directory for equivalent Lightning epoch checkpoints.
        os.environ["CARD10_TFT_RECOVERY_DIR"] = str(
            out_dir / "tft_epoch_recovery"
        )
    adapter = ADAPTERS[MODEL_TO_ADAPTER[args.model]](candidate)
    adapter.fit(actual, forecast, fit_idx, valid_index=valid_idx)
    prediction = adapter.predict(actual, forecast, test_idx)
    runtime = float(time.time() - started)
    origin = f"{args.test_year - 1}->{args.test_year}"
    prediction_path = out_dir / "predictions.csv.gz"
    save_prediction(
        prediction_path, args.model, candidate["candidate_id"], origin, actual, prediction
    )
    checkpoint_path = out_dir / "checkpoint.pt"
    if not hasattr(adapter, "save_checkpoint"):
        raise RuntimeError("selected adapter lacks checkpoint support")
    adapter.save_checkpoint(str(checkpoint_path))
    package_versions = {}
    for package in ("torch", "pytorch-forecasting", "neuralforecast", "lightning"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "unavailable"
    result = {
        "schema": "card10_r86_fair_deep_evaluation_cell_v1",
        "status": "PASS",
        "model": args.model,
        "candidate_id": candidate["candidate_id"],
        "target_mode": candidate["target_mode"],
        "origin": origin,
        "test_year": args.test_year,
        "runtime_train_predict_sec": runtime,
        "config_hash": adapter.config_hash,
        "model_hash": adapter.model_hash(),
        "model_size_bytes": int(checkpoint_path.stat().st_size),
        "prediction_sha256": sha256_file(prediction_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "candidate_grid_sha256": sha256_file(grid_path),
        "selected_contract_sha256": sha256_file(selected_path),
        "input_sha256": panel.source_sha256,
        "fit_start": str(fit_idx.min()),
        "fit_end": str(fit_idx.max()),
        "fit_rows": int(len(fit_idx)),
        "validation_start": str(valid_idx.min()),
        "validation_end": str(valid_idx.max()),
        "validation_rows": int(len(valid_idx)),
        "test_start": str(test_idx.min()),
        "test_end": str(test_idx.max()),
        "test_rows": int(len(test_idx)),
        "fitted_state": adapter.fitted_state,
        "prediction_meta": prediction.meta,
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "python": sys.version,
            "packages": package_versions,
            "argv": sys.argv,
        },
        "metrics": metrics(actual, prediction),
        "access_flags": grid["access_flags"],
    }
    write_json(out_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
