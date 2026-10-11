#!/usr/bin/env python3
"""R86 development-only selection for four fairly budgeted deep baselines.

The command is deliberately limited to 2018 development data.  It writes one
terminal record per candidate and refuses to freeze a winner until every
pre-registered candidate for the requested model has a PASS/FAIL record.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
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


SCHEMA = "card10_r86_fair_deep_candidate_grid_v1"
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


def canonical_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_grid(path: Path) -> dict[str, Any]:
    safe_path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema") != SCHEMA:
        raise RuntimeError("R86 candidate-grid schema mismatch")
    if not obj.get("frozen_before_2019_2025_evaluation"):
        raise RuntimeError("R86 grid was not frozen before evaluation")
    candidates = obj.get("candidates", [])
    if len(candidates) != 16:
        raise RuntimeError("R86 grid must contain exactly 16 candidates")
    counts = pd.Series([row.get("model") for row in candidates]).value_counts().to_dict()
    if counts != {"TFT": 4, "NHiTS": 4, "PatchTST": 4, "ProLoaF": 4}:
        raise RuntimeError(f"unexpected per-model candidate counts: {counts}")
    if any(obj.get("access_flags", {}).values()) or any(ACCESS_FLAGS.values()):
        raise RuntimeError("access flags are not clean")
    return obj


def local_boundary(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize("America/New_York")


def development_indices(
    index: pd.DatetimeIndex, grid: dict[str, Any]
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    dev = grid["development"]
    train_end = local_boundary(dev["train_end_exclusive_local"])
    valid_end = local_boundary(dev["validation_end_exclusive_local"])
    select_start = local_boundary(dev["selection_start_local"])
    select_end_exclusive = local_boundary(dev["selection_end_local"]) + pd.DateOffset(days=1)
    train_idx = index[index < train_end]
    valid_idx = index[(index >= train_end) & (index < valid_end)]
    days = pd.date_range(
        select_start,
        select_end_exclusive - pd.DateOffset(days=1),
        freq=f"{int(dev['selection_day_stride'])}D",
    )
    selection_idx = index[
        (index >= select_start)
        & (index < select_end_exclusive)
        & index.normalize().isin(days.normalize())
    ]
    if set(train_idx.year) & set(grid["evaluation"]["origins"]):
        raise RuntimeError("evaluation year leaked into R86 training")
    if set(valid_idx.year) & set(grid["evaluation"]["origins"]):
        raise RuntimeError("evaluation year leaked into R86 validation")
    if set(selection_idx.year) != {2018}:
        raise RuntimeError("R86 selection must be 2018-only")
    return train_idx, valid_idx, selection_idx


def effective_candidate(
    row: dict[str, Any], grid: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    candidate = dict(row)
    candidate.pop("model", None)
    candidate["seed"] = int(grid["shared"]["seed"])
    candidate["forecast_horizon"] = 49
    if row["model"] == "TFT":
        candidate["max_encoder_length"] = int(candidate["encoder_length"])
        candidate["max_prediction_length"] = 49
    elif row["model"] == "ProLoaF":
        candidate["history_horizon"] = int(candidate["history_hours"])
        candidate["repo"] = args.proloaf_repo
        candidate["work_dir"] = str(Path(args.output_dir) / "proloaf_runtime")
    return candidate


def quantile_metrics(
    actual: pd.DataFrame, prediction: Any
) -> dict[str, Any]:
    frames = {0.05: prediction.q05, 0.50: prediction.q50, 0.95: prediction.q95}
    if any(frame is None for frame in frames.values()):
        raise RuntimeError("R86 selection requires q05/q50/q95 from every candidate")
    y = actual.loc[prediction.point.index, list(ZONES)].to_numpy(float)
    point = prediction.point.to_numpy(float)
    finite = np.isfinite(y) & np.isfinite(point)
    if not finite.all():
        raise RuntimeError("non-finite R86 selection point prediction")
    denom = float(np.abs(y).sum())
    if denom <= 0:
        raise RuntimeError("invalid normalized-WQL denominator")
    pinball_means: dict[str, float] = {}
    normalized: list[float] = []
    q_arrays: dict[float, np.ndarray] = {}
    for q, frame in frames.items():
        qhat = frame.to_numpy(float)
        if not np.isfinite(qhat).all():
            raise RuntimeError(f"non-finite q={q} prediction")
        q_arrays[q] = qhat
        error = y - qhat
        loss = np.maximum(q * error, (q - 1.0) * error)
        name = f"q{int(q * 100):02d}"
        pinball_means[f"pinball_{name}"] = float(loss.mean())
        normalized.append(float(2.0 * loss.sum() / denom))
    daily_joint = (
        (y >= q_arrays[0.05]) & (y <= q_arrays[0.95])
    ).all(axis=1)
    return {
        "n": int(y.size),
        "n_timestamps": int(y.shape[0]),
        "mae": float(np.abs(y - point).mean()),
        "rmse": float(np.sqrt(np.square(y - point).mean())),
        "wape": float(np.abs(y - point).sum() / denom),
        "mean_pinball": float(np.mean(list(pinball_means.values()))),
        "normalized_wql": float(np.mean(normalized)),
        "marginal_coverage_q05_q95": float(
            ((y >= q_arrays[0.05]) & (y <= q_arrays[0.95])).mean()
        ),
        "simultaneous_11zone_coverage": float(daily_joint.mean()),
        "mean_width_q05_q95": float((q_arrays[0.95] - q_arrays[0.05]).mean()),
        **pinball_means,
    }


def save_prediction(path: Path, model: str, candidate_id: str, actual: pd.DataFrame, pred: Any) -> None:
    parts = []
    for zone in ZONES:
        part = pd.DataFrame(
            {
                "timestamp_utc": pred.point.index.tz_convert("UTC").astype(str),
                "timestamp_local": pred.point.index.astype(str),
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
        parts.append(part)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
    }
    for package in ("torch", "pytorch-forecasting", "neuralforecast", "lightning"):
        try:
            record[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            record[package] = "unavailable"
    return record


def run_candidate(
    row: dict[str, Any],
    grid: dict[str, Any],
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    train_idx: pd.DatetimeIndex,
    valid_idx: pd.DatetimeIndex,
    selection_idx: pd.DatetimeIndex,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model = row["model"]
    candidate = effective_candidate(row, grid, args)
    candidate_id = candidate["candidate_id"]
    started = time.time()
    adapter = ADAPTERS[MODEL_TO_ADAPTER[model]](candidate)
    adapter.fit(actual, forecast, train_idx, valid_index=valid_idx)
    prediction = adapter.predict(actual, forecast, selection_idx)
    metrics = quantile_metrics(actual, prediction)
    prediction_path = out_dir / "development_predictions" / f"{candidate_id}.csv.gz"
    save_prediction(prediction_path, model, candidate_id, actual, prediction)
    checkpoint_path = out_dir / "development_checkpoints" / f"{candidate_id}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(adapter, "save_checkpoint"):
        raise RuntimeError(f"{model} adapter lacks deterministic checkpoint support")
    adapter.save_checkpoint(str(checkpoint_path))
    return {
        "status": "PASS",
        "model": model,
        "candidate_id": candidate_id,
        "target_mode": candidate["target_mode"],
        "effective_candidate": candidate,
        "config_hash": adapter.config_hash,
        "model_hash": adapter.model_hash(),
        "runtime_train_predict_sec": float(time.time() - started),
        "model_size_proxy": int(checkpoint_path.stat().st_size),
        "checkpoint": str(checkpoint_path.relative_to(out_dir)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "development_prediction": str(prediction_path.relative_to(out_dir)),
        "development_prediction_sha256": sha256_file(prediction_path),
        "fitted_state": adapter.fitted_state,
        "prediction_meta": prediction.meta,
        **metrics,
    }


def freeze_selected(
    grid: dict[str, Any], results: dict[str, Any], models: list[str], out_dir: Path, grid_path: Path
) -> Path:
    selected_models: dict[str, Any] = {}
    for model in models:
        expected = [row for row in grid["candidates"] if row["model"] == model]
        terminal = [results.get("candidates", {}).get(row["candidate_id"]) for row in expected]
        missing = [row["candidate_id"] for row, result in zip(expected, terminal) if result is None]
        if missing:
            raise RuntimeError(f"cannot freeze {model}; missing terminal candidates: {missing}")
        passed = [result for result in terminal if result.get("status") == "PASS"]
        if not passed:
            selected_models[model] = {
                "status": "FAILED_CLOSED",
                "selected_candidate_id": None,
                "all_candidates": terminal,
            }
            continue
        passed.sort(
            key=lambda r: (
                r["normalized_wql"],
                r["mae"],
                r["runtime_train_predict_sec"],
                r["model_size_proxy"],
                r["candidate_id"],
            )
        )
        selected_models[model] = {
            "status": "DEVELOPMENT_SELECTED_NOT_YET_EVALUATED",
            "selected_candidate_id": passed[0]["candidate_id"],
            "selected": passed[0],
            "all_candidates": terminal,
        }
    selected = {
        "schema": "card10_r86_fair_deep_selected_v1",
        "status": "DEVELOPMENT_SELECTED_NOT_YET_EVALUATED",
        "selection_rule": grid["development"]["primary_loss"],
        "tie_break": grid["development"]["tie_break"],
        "candidate_grid_file_sha256": sha256_file(grid_path),
        "candidate_grid_canonical_sha256": canonical_sha256(grid),
        "development_results_sha256": sha256_file(out_dir / "development_results.json"),
        "models": selected_models,
        "access_flags": grid["access_flags"],
    }
    path = out_dir / "fair_deep_selected_v1.json"
    write_json(path, selected)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="R86 fair deep development selector")
    parser.add_argument("--data", default=None)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_TO_ADAPTER), default=list(MODEL_TO_ADAPTER)
    )
    parser.add_argument("--candidate-id", nargs="+", default=None)
    parser.add_argument(
        "--proloaf-repo",
        default=os.environ.get("PROLOAF_REPO"),
        help="path to the pinned ProLoaF checkout (required for ProLoaF runs)",
    )
    parser.add_argument("--overwrite-candidate", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    grid_path = Path(args.grid).resolve()
    out_dir = Path(args.output_dir).resolve()
    safe_path(grid_path)
    safe_path(out_dir)
    grid = load_grid(grid_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "environment.json", environment_record())
    panel = load_nyiso_raw(args.data)
    actual, forecast = panel.actual, panel.forecast
    train_idx, valid_idx, selection_idx = development_indices(actual.index, grid)
    if actual.loc[train_idx.append(valid_idx).append(selection_idx)].isna().any().any():
        raise RuntimeError("R86 development slices contain missing outcomes")
    if forecast.loc[train_idx.append(valid_idx).append(selection_idx)].isna().any().any():
        raise RuntimeError("R86 development slices contain missing P7 forecasts")

    results_path = out_dir / "development_results.json"
    results = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists()
        else {
            "schema": "card10_r86_fair_deep_development_results_v1",
            "candidate_grid_file_sha256": sha256_file(grid_path),
            "candidate_grid_canonical_sha256": canonical_sha256(grid),
            "input_sha256": panel.source_sha256,
            "split": {
                "train_start": str(train_idx.min()),
                "train_end": str(train_idx.max()),
                "train_rows": int(len(train_idx)),
                "validation_start": str(valid_idx.min()),
                "validation_end": str(valid_idx.max()),
                "validation_rows": int(len(valid_idx)),
                "selection_start": str(selection_idx.min()),
                "selection_end": str(selection_idx.max()),
                "selection_rows": int(len(selection_idx)),
            },
            "candidates": {},
            "access_flags": grid["access_flags"],
        }
    )
    if results["candidate_grid_file_sha256"] != sha256_file(grid_path):
        raise RuntimeError("candidate grid changed after R86 results were started")

    wanted_ids = set(args.candidate_id or [])
    rows = [
        row
        for row in grid["candidates"]
        if row["model"] in args.models and (not wanted_ids or row["candidate_id"] in wanted_ids)
    ]
    if wanted_ids - {row["candidate_id"] for row in rows}:
        raise RuntimeError(f"unknown or model-excluded candidate IDs: {sorted(wanted_ids)}")
    for row in rows:
        cid = row["candidate_id"]
        if cid in results["candidates"] and not args.overwrite_candidate:
            print(f"[R86] skip terminal candidate {cid}", flush=True)
            continue
        print(f"[R86] run {row['model']} {cid}", flush=True)
        try:
            result = run_candidate(
                row, grid, actual, forecast, train_idx, valid_idx, selection_idx, out_dir, args
            )
        except Exception as exc:  # noqa: BLE001
            failure_dir = out_dir / "failure_logs"
            failure_dir.mkdir(parents=True, exist_ok=True)
            failure_path = failure_dir / f"{cid}.txt"
            failure_path.write_text(traceback.format_exc(), encoding="utf-8")
            result = {
                "status": "FAIL",
                "model": row["model"],
                "candidate_id": cid,
                "target_mode": row["target_mode"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failure_log": str(failure_path.relative_to(out_dir)),
                "failure_log_sha256": sha256_file(failure_path),
            }
        results["candidates"][cid] = result
        write_json(results_path, results)
        print(
            f"[R86] terminal {cid}: {result['status']} "
            f"nwql={result.get('normalized_wql', 'NA')}",
            flush=True,
        )

    if args.finalize:
        selected_path = freeze_selected(grid, results, list(args.models), out_dir, grid_path)
        print(f"[R86] froze selected contract: {selected_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
