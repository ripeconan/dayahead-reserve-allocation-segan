"""Development-selected DeepAR pipeline for the NYISO benchmark.

The model predicts residuals relative to the frozen P7 forecast and preserves
the strict delivery-day lag-2 information contract. Development selection and
evaluation are separate command-line modes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ZONES = (
    "CAPITL",
    "CENTRL",
    "DUNWOD",
    "GENESE",
    "HUD VL",
    "LONGIL",
    "MHK VL",
    "MILLWD",
    "N.Y.C.",
    "NORTH",
    "WEST",
)
LOCAL_TZ = "America/New_York"
PREDICTION_LENGTH = 49
EXPECTED_RAW_SHA256 = "d9e492532e22bc3f41c6967acf99cc5b49a49ae2e5ba3ae3b7dbc8092d9e5d0d"
EXPECTED_GRID_SCHEMA = "card10_r85_deepar_fair_candidate_grid_v1"
FORBIDDEN = (
    "confirmatory_locked/card10_nyiso_2026h1",
    "card10_confirmatory_unseen/card10_nyiso_2026h1",
    "card10_nyiso_2026h1",
    "gefs_revision_r77_d2_tmprh",
    "r77_d2",
)


@dataclass(frozen=True)
class Panel:
    actual: pd.DataFrame
    forecast: pd.DataFrame
    raw_sha256: str


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_path(path: str | Path) -> None:
    text = str(path).replace("\\", "/").lower()
    for forbidden in FORBIDDEN:
        if forbidden in text:
            raise RuntimeError(f"FORBIDDEN_PATH_BEFORE_ENUMERATION: {path}")
    if "nyiso" in text and "2026" in text:
        raise RuntimeError(f"NYISO_2026_PATH_BEFORE_ENUMERATION: {path}")


def load_grid(path: Path) -> dict[str, Any]:
    safe_path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema") != EXPECTED_GRID_SCHEMA:
        raise RuntimeError("candidate grid schema mismatch")
    if not obj.get("frozen_before_2019_2025_evaluation"):
        raise RuntimeError("candidate grid is not frozen")
    if len(obj.get("candidates", [])) != 8:
        raise RuntimeError("candidate grid must contain exactly eight candidates")
    if any(obj["access_flags"].values()):
        raise RuntimeError("candidate grid access flags are not clean")
    return obj


def load_panel(path: Path) -> Panel:
    safe_path(path)
    observed = sha256_path(path)
    if observed != EXPECTED_RAW_SHA256:
        raise RuntimeError(f"RAW_SHA256_MISMATCH expected={EXPECTED_RAW_SHA256} got={observed}")
    raw = pd.read_csv(path, low_memory=False)
    if "timestamp_utc" not in raw:
        raise ValueError("missing timestamp_utc")
    utc = pd.to_datetime(raw["timestamp_utc"], utc=True)
    local = pd.DatetimeIndex(utc.dt.tz_convert(LOCAL_TZ))
    actual = pd.DataFrame(
        raw[[f"{zone}_actual" for zone in ZONES]].to_numpy(float),
        index=local,
        columns=ZONES,
    )
    forecast = pd.DataFrame(
        raw[[f"{zone}_forecast" for zone in ZONES]].to_numpy(float),
        index=local,
        columns=ZONES,
    )
    if actual.index.duplicated().any() or not actual.index.is_monotonic_increasing:
        raise ValueError("NYISO timestamp identity is duplicated or unsorted")
    if actual.index.max().year >= 2026:
        raise RuntimeError("unexpected 2026 row in exposed input")
    return Panel(actual=actual, forecast=forecast, raw_sha256=observed)


def local_cutoff(day: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(day)
    if stamp.tzinfo is None:
        return stamp.tz_localize(LOCAL_TZ)
    return stamp.tz_convert(LOCAL_TZ)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def dynamic_features(index: pd.DatetimeIndex, p7: np.ndarray) -> np.ndarray:
    local = pd.DatetimeIndex(index).tz_convert(LOCAL_TZ)
    hour = local.hour.to_numpy(float)
    dow = local.dayofweek.to_numpy(float)
    month = local.month.to_numpy(float)
    values = np.asarray(p7, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite P7 feature")
    return np.vstack(
        [
            values / 1000.0,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            np.sin(2 * np.pi * (month - 1.0) / 12.0),
            np.cos(2 * np.pi * (month - 1.0) / 12.0),
            (dow >= 5).astype(float),
        ]
    ).astype("float32")


def target_frame(panel: Panel, target_mode: str) -> pd.DataFrame:
    if target_mode == "load_level":
        return panel.actual
    if target_mode == "p7_residual":
        return panel.actual - panel.forecast
    raise ValueError(f"unknown target mode: {target_mode}")


def contiguous_segments(index: pd.DatetimeIndex, minimum: int = 800) -> list[pd.DatetimeIndex]:
    if len(index) == 0:
        return []
    utc = pd.DatetimeIndex(index).tz_convert("UTC")
    breaks = np.flatnonzero((utc[1:] - utc[:-1]) != pd.Timedelta(hours=1)) + 1
    pieces = np.split(np.arange(len(index)), breaks)
    return [pd.DatetimeIndex(index.take(piece)) for piece in pieces if len(piece) >= minimum]


def make_series_dataset(panel: Panel, target_mode: str, end_exclusive: pd.Timestamp):
    from gluonts.dataset.common import ListDataset

    end_exclusive = local_cutoff(end_exclusive)
    complete = panel.actual.notna().all(axis=1) & panel.forecast.notna().all(axis=1)
    valid = panel.actual.index[complete & (panel.actual.index < end_exclusive)]
    target = target_frame(panel, target_mode)
    entries: list[dict[str, Any]] = []
    for segment in contiguous_segments(valid):
        for zone_id, zone in enumerate(ZONES):
            entries.append(
                {
                    "start": segment[0].tz_convert("UTC").tz_localize(None),
                    "target": target.loc[segment, zone].to_numpy("float32"),
                    "feat_dynamic_real": dynamic_features(
                        segment, panel.forecast.loc[segment, zone].to_numpy(float)
                    ),
                    "feat_static_cat": [zone_id],
                    "item_id": f"{target_mode}:{zone}:{segment[0].isoformat()}",
                }
            )
    if len(entries) < len(ZONES):
        raise RuntimeError("insufficient contiguous training entries")
    return ListDataset(entries, freq="h"), len(entries)


def contiguous_history_suffix(
    panel: Panel,
    zone: str,
    cutoff: pd.Timestamp,
    history_days_max: int,
) -> pd.DatetimeIndex:
    valid = panel.actual[zone].notna() & panel.forecast[zone].notna()
    eligible = panel.actual.index[valid & (panel.actual.index <= cutoff)]
    if len(eligible) == 0 or eligible[-1] != cutoff:
        return pd.DatetimeIndex([])
    utc = eligible.tz_convert("UTC")
    breaks = np.flatnonzero((utc[1:] - utc[:-1]) != pd.Timedelta(hours=1))
    start_pos = int(breaks[-1] + 1) if len(breaks) else 0
    suffix = eligible[start_pos:]
    lower = cutoff.tz_convert("UTC") - pd.Timedelta(days=history_days_max)
    return suffix[suffix.tz_convert("UTC") >= lower]


def make_prediction_entries(
    panel: Panel,
    target_mode: str,
    target_days: Iterable[pd.Timestamp],
    history_days_max: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    target = target_frame(panel, target_mode)
    for raw_day in target_days:
        day = local_cutoff(raw_day).normalize()
        bridge_start = day - pd.DateOffset(days=1)
        cutoff = bridge_start.tz_convert("UTC") - pd.Timedelta(hours=1)
        cutoff = cutoff.tz_convert(LOCAL_TZ)
        future = pd.date_range(bridge_start, periods=PREDICTION_LENGTH, freq="h")
        target_positions = np.flatnonzero(np.asarray(future.date) == day.date())
        if len(target_positions) not in (23, 24, 25):
            raise RuntimeError(f"target day not represented exactly: {day} positions={len(target_positions)}")
        target_index = future[target_positions]
        if panel.actual.reindex(target_index).isna().any().any():
            continue
        # DeepAR always emits 49 steps.  On a normal 24-hour target day the
        # final step lies after the scored day; a missing P7 value there may be
        # forward-filled because a suffix covariate cannot affect earlier RNN
        # outputs.  Every bridge/target P7 cell must still be present.
        if panel.forecast.reindex(future[: int(target_positions[-1]) + 1]).isna().any().any():
            continue
        for zone_id, zone in enumerate(ZONES):
            history = contiguous_history_suffix(panel, zone, cutoff, history_days_max)
            if len(history) < 500:
                continue
            if history[-1].tz_convert("UTC") + pd.Timedelta(hours=1) != future[0].tz_convert("UTC"):
                raise RuntimeError("history and bridge are not adjacent")
            full_index = history.append(future)
            p7 = panel.forecast.reindex(full_index)[zone]
            if p7.loc[: target_index[-1]].isna().any():
                continue
            p7 = p7.ffill()
            if p7.isna().any():
                continue
            entries.append(
                {
                    "start": full_index[0].tz_convert("UTC").tz_localize(None),
                    "target": target.loc[history, zone].to_numpy("float32"),
                    "feat_dynamic_real": dynamic_features(
                        full_index, p7.to_numpy(float)
                    ),
                    "feat_static_cat": [zone_id],
                    "item_id": f"{target_mode}:{zone}:{day.date().isoformat()}",
                }
            )
            meta.append(
                {
                    "day": day,
                    "zone": zone,
                    "future": future,
                    "target_positions": target_positions,
                    "target_index": target_index,
                    "history_end": history[-1],
                }
            )
    return entries, meta


def estimator_for(
    candidate: dict[str, Any],
    shared: dict[str, Any],
    root: Path,
    use_gpu: bool,
    *,
    with_validation: bool,
):
    from gluonts.torch.model.deepar import DeepAREstimator
    from lightning.pytorch.callbacks import EarlyStopping

    callbacks = []
    if with_validation:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=int(shared["early_stopping_patience"]),
                check_finite=True,
            )
        )
    return DeepAREstimator(
        freq="h",
        prediction_length=PREDICTION_LENGTH,
        context_length=int(candidate["context_length"]),
        num_layers=int(shared["num_layers"]),
        hidden_size=int(candidate["hidden_size"]),
        dropout_rate=float(shared["dropout_rate"]),
        lr=float(shared["learning_rate"]),
        patience=int(shared["lr_scheduler_patience"]),
        batch_size=int(shared["batch_size"]),
        num_batches_per_epoch=int(shared["num_batches_per_epoch"]),
        num_feat_dynamic_real=8,
        num_feat_static_cat=1,
        cardinality=[len(ZONES)],
        scaling=bool(shared["scaling"]),
        time_features=[],
        num_parallel_samples=int(shared["num_samples"]),
        nonnegative_pred_samples=candidate["target_mode"] == "load_level",
        trainer_kwargs={
            "max_epochs": int(shared["max_epochs"]),
            "accelerator": "gpu" if use_gpu else "cpu",
            "devices": 1,
            "default_root_dir": str(root),
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "callbacks": callbacks,
        },
    )


def extract_best_epoch(train_output: Any) -> tuple[int, float | None, int]:
    callback = train_output.trainer.checkpoint_callbacks[0]
    best_path = str(getattr(callback, "best_model_path", ""))
    match = re.search(r"epoch=(\d+)", best_path)
    best_epoch = int(match.group(1)) + 1 if match else int(train_output.trainer.current_epoch)
    score_obj = getattr(callback, "best_model_score", None)
    best_score = float(score_obj.detach().cpu()) if score_obj is not None else None
    return best_epoch, best_score, int(train_output.trainer.current_epoch)


def forecast_entries(
    predictor: Any,
    entries: list[dict[str, Any]],
    meta: list[dict[str, Any]],
    panel: Panel,
    target_mode: str,
    num_samples: int,
) -> pd.DataFrame:
    from gluonts.dataset.common import ListDataset

    if len(entries) != len(meta):
        raise RuntimeError("prediction entry metadata mismatch")
    dataset = ListDataset(entries, freq="h")
    rows: list[dict[str, Any]] = []
    forecasts = predictor.predict(dataset, num_samples=num_samples)
    count = 0
    for forecast_obj, item in zip(forecasts, meta):
        samples = np.asarray(forecast_obj.samples, dtype=float)
        positions = np.asarray(item["target_positions"], dtype=int)
        selected = samples[:, positions]
        index = pd.DatetimeIndex(item["target_index"])
        if target_mode == "p7_residual":
            selected = selected + panel.forecast.loc[index, item["zone"]].to_numpy(float)[None, :]
        q05, q50, q95 = np.quantile(selected, [0.05, 0.50, 0.95], axis=0)
        for pos, timestamp in enumerate(index):
            rows.append(
                {
                    "timestamp_utc": timestamp.tz_convert("UTC").isoformat(),
                    "delivery_date_local": timestamp.date().isoformat(),
                    "delivery_hour_local": int(timestamp.hour),
                    "utc_offset_hours": float(timestamp.utcoffset().total_seconds() / 3600.0),
                    "zone": item["zone"],
                    "q05": float(q05[pos]),
                    "q50": float(q50[pos]),
                    "q95": float(q95[pos]),
                    "history_end_utc": item["history_end"].tz_convert("UTC").isoformat(),
                }
            )
        count += 1
    if count != len(entries):
        raise RuntimeError(f"predictor returned {count} forecasts for {len(entries)} entries")
    result = pd.DataFrame(rows)
    if result.empty or not np.isfinite(result[["q05", "q50", "q95"]].to_numpy()).all():
        raise RuntimeError("empty or non-finite forecasts")
    return result


def attach_actual(predictions: pd.DataFrame, panel: Panel) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    timestamps = pd.to_datetime(predictions["timestamp_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
    frame = predictions.copy()
    frame["_local_index"] = pd.DatetimeIndex(timestamps)
    for zone in ZONES:
        part = frame[frame["zone"] == zone].copy()
        part["actual"] = panel.actual.reindex(pd.DatetimeIndex(part["_local_index"]))[zone].to_numpy(float)
        rows.append(part)
    out = pd.concat(rows, ignore_index=True).drop(columns="_local_index")
    if out["actual"].isna().any():
        raise RuntimeError("sealed prediction does not align to actual")
    return out


def metric_record(frame: pd.DataFrame) -> dict[str, Any]:
    actual = frame["actual"].to_numpy(float)
    q05 = frame["q05"].to_numpy(float)
    q50 = frame["q50"].to_numpy(float)
    q95 = frame["q95"].to_numpy(float)
    error = actual - q50
    absolute = np.abs(error)
    pinballs: dict[str, float] = {}
    normalized_wql: list[float] = []
    denom = float(np.abs(actual).sum())
    for quantile, values, name in ((0.05, q05, "q05"), (0.50, q50, "q50"), (0.95, q95, "q95")):
        residual = actual - values
        loss = np.maximum(quantile * residual, (quantile - 1.0) * residual)
        pinballs[f"pinball_{name}"] = float(loss.mean())
        normalized_wql.append(float(2.0 * loss.sum() / denom))
    zone_mae = frame.assign(abs_error=absolute).groupby("zone")["abs_error"].mean()
    covered = (actual >= q05) & (actual <= q95)
    simultaneous = (
        frame.assign(covered=covered)
        .groupby("timestamp_utc", sort=False)["covered"]
        .all()
    )
    return {
        "n": int(len(frame)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "wape": float(absolute.sum() / denom),
        "worst_zone_mae": float(zone_mae.max()),
        "worst_zone": str(zone_mae.idxmax()),
        "tail_mae_p99": float(absolute[absolute >= np.quantile(absolute, 0.99)].mean()),
        "marginal_coverage_q05_q95": float(covered.mean()),
        "simultaneous_11zone_coverage": float(simultaneous.mean()),
        "mean_width_q05_q95": float(np.mean(q95 - q05)),
        "mean_pinball": float(np.mean(list(pinballs.values()))),
        "normalized_wql": float(np.mean(normalized_wql)),
        **pinballs,
    }


def atomic_prediction_write(frame: pd.DataFrame, path: Path, extra: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    frame.to_csv(temporary, index=False, compression={"method": "gzip"})
    os.replace(temporary, path)
    seal = {
        "schema": "card10_r85_sealed_prediction_v1",
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "rows": int(len(frame)),
        **extra,
    }
    write_json(path.with_suffix(path.suffix + ".seal.json"), seal)
    return seal


def run_development(args: argparse.Namespace) -> None:
    grid = load_grid(Path(args.grid))
    panel = load_panel(Path(args.nyiso))
    root = Path(args.out_dir)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("development output directory must be new and empty")
    root.mkdir(parents=True, exist_ok=True)
    shared = grid["shared"]
    development = grid["development"]
    selection_days = pd.date_range(
        development["selection_start_local"],
        development["selection_end_local"],
        freq=f"{int(development['selection_day_stride'])}D",
        tz=LOCAL_TZ,
    )
    results: list[dict[str, Any]] = []
    for candidate in grid["candidates"]:
        candidate_id = candidate["candidate_id"]
        candidate_dir = root / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=False)
        seed_everything(int(shared["seed"]))
        train_ds, train_entries = make_series_dataset(
            panel, candidate["target_mode"], local_cutoff(development["train_end_exclusive_local"])
        )
        validation_ds, validation_entries = make_series_dataset(
            panel,
            candidate["target_mode"],
            local_cutoff(development["validation_end_exclusive_local"]),
        )
        estimator = estimator_for(
            candidate, shared, candidate_dir, args.use_gpu, with_validation=True
        )
        started = time.time()
        train_output = estimator.train_model(
            train_ds,
            validation_data=validation_ds,
            cache_data=True,
        )
        runtime_train = time.time() - started
        best_epoch, best_val_loss, stopped_epoch = extract_best_epoch(train_output)
        entries, meta = make_prediction_entries(
            panel,
            candidate["target_mode"],
            selection_days,
            int(shared["history_days_max"]),
        )
        forecast = forecast_entries(
            train_output.predictor,
            entries,
            meta,
            panel,
            candidate["target_mode"],
            int(shared["num_samples"]),
        )
        forecast["candidate_id"] = candidate_id
        forecast_path = candidate_dir / "development_predictions.csv.gz"
        seal = atomic_prediction_write(
            forecast,
            forecast_path,
            {"cohort": "2018Q4_every_3_days", "candidate_id": candidate_id},
        )
        scored = attach_actual(forecast, panel)
        metrics = metric_record(scored)
        result = {
            **candidate,
            **metrics,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "stopped_epoch": stopped_epoch,
            "runtime_train_sec": runtime_train,
            "train_entries": train_entries,
            "validation_entries": validation_entries,
            "development_prediction_sha256": seal["sha256"],
        }
        results.append(result)
        write_json(candidate_dir / "development_result.json", result)
    ranked = sorted(
        results,
        key=lambda row: (
            row["normalized_wql"],
            row["mae"],
            int(row["hidden_size"]) * int(row["context_length"]),
            int(row["context_length"]),
            row["candidate_id"],
        ),
    )
    table = pd.DataFrame(results).sort_values(
        ["normalized_wql", "mae", "candidate_id"], kind="mergesort"
    )
    table.to_csv(root / "development_candidate_results.csv", index=False)
    selected = {
        "schema": "card10_r85_deepar_fair_selected_v1",
        "status": "DEVELOPMENT_SELECTED_NOT_YET_EVALUATED",
        "candidate_grid_canonical_sha256": canonical_json_sha256(grid),
        "candidate_grid_file_sha256": sha256_path(Path(args.grid)),
        "development_results_sha256": sha256_path(root / "development_candidate_results.csv"),
        "selection_rule": development["primary_loss"],
        "selected_candidate": ranked[0],
        "refit_epochs": int(ranked[0]["best_epoch"]),
        "access_flags": grid["access_flags"],
    }
    write_json(root / "selected_candidate.json", selected)
    manifest = {
        "schema": "card10_r85_deepar_fair_development_manifest_v1",
        "raw_sha256": panel.raw_sha256,
        "candidate_grid_file_sha256": sha256_path(Path(args.grid)),
        "selected_candidate_sha256": sha256_path(root / "selected_candidate.json"),
        "candidate_count": len(results),
        "selected_candidate_id": ranked[0]["candidate_id"],
        "files": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ],
        "access_flags": grid["access_flags"],
    }
    write_json(root / "MANIFEST.json", manifest)
    print(json.dumps(selected, sort_keys=True))


def load_selected(selected_path: Path, grid: dict[str, Any]) -> dict[str, Any]:
    safe_path(selected_path)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected.get("schema") != "card10_r85_deepar_fair_selected_v1":
        raise RuntimeError("selected contract schema mismatch")
    candidate = selected.get("selected_candidate", {})
    frozen = {row["candidate_id"]: row for row in grid["candidates"]}
    if candidate.get("candidate_id") not in frozen:
        raise RuntimeError("selected candidate is outside frozen grid")
    for field in ("candidate_id", "target_mode", "context_length", "hidden_size"):
        if candidate.get(field) != frozen[candidate["candidate_id"]].get(field):
            raise RuntimeError(f"selected candidate drift: {field}")
    if not (1 <= int(selected.get("refit_epochs", 0)) <= int(grid["shared"]["max_epochs"])):
        raise RuntimeError("invalid selected refit epoch")
    if any(selected.get("access_flags", {}).values()):
        raise RuntimeError("selected contract access flags are not clean")
    return selected


def run_evaluation_predictions(args: argparse.Namespace) -> None:
    grid = load_grid(Path(args.grid))
    selected = load_selected(Path(args.selected), grid)
    panel = load_panel(Path(args.nyiso))
    root = Path(args.out_dir)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("evaluation output directory must be new and empty")
    root.mkdir(parents=True, exist_ok=True)
    candidate = {
        field: selected["selected_candidate"][field]
        for field in ("candidate_id", "target_mode", "context_length", "hidden_size")
    }
    shared = dict(grid["shared"])
    shared["max_epochs"] = int(selected["refit_epochs"])
    # Refit uses the frozen development epoch count and cannot inspect evaluation val_loss.
    shared["early_stopping_patience"] = int(selected["refit_epochs"]) + 1
    origin_rows: list[dict[str, Any]] = []
    for year in range(2019, 2026):
        origin = f"{year - 1}->{year}"
        origin_dir = root / origin.replace("->", "_to_")
        origin_dir.mkdir(parents=True, exist_ok=False)
        seed_everything(int(shared["seed"]))
        training, train_entries = make_series_dataset(
            panel, candidate["target_mode"], local_cutoff(f"{year}-01-01")
        )
        estimator = estimator_for(
            candidate, shared, origin_dir, args.use_gpu, with_validation=False
        )
        # No evaluation-derived validation dataset is supplied.
        started_train = time.time()
        predictor = estimator.train(training, cache_data=True)
        runtime_train = time.time() - started_train
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D", tz=LOCAL_TZ)
        parts: list[pd.DataFrame] = []
        started_predict = time.time()
        chunk_days = int(args.chunk_days)
        for start in range(0, len(days), chunk_days):
            entries, meta = make_prediction_entries(
                panel,
                candidate["target_mode"],
                days[start : start + chunk_days],
                int(shared["history_days_max"]),
            )
            if entries:
                parts.append(
                    forecast_entries(
                        predictor,
                        entries,
                        meta,
                        panel,
                        candidate["target_mode"],
                        int(shared["num_samples"]),
                    )
                )
        if not parts:
            raise RuntimeError(f"no predictions for {origin}")
        predictions = pd.concat(parts, ignore_index=True)
        predictions.insert(0, "origin", origin)
        predictions.insert(1, "model", "deepar_fair_" + candidate["target_mode"])
        prediction_path = origin_dir / "predictions.csv.gz"
        seal = atomic_prediction_write(
            predictions,
            prediction_path,
            {
                "origin": origin,
                "selected_candidate_id": candidate["candidate_id"],
                "selected_contract_sha256": sha256_path(Path(args.selected)),
            },
        )
        predictor_dir = origin_dir / "predictor"
        predictor_dir.mkdir(parents=True, exist_ok=False)
        predictor.serialize(predictor_dir)
        meta = {
            "origin": origin,
            "train_end_exclusive_local": f"{year}-01-01",
            "train_entries": train_entries,
            "runtime_train_sec": runtime_train,
            "runtime_predict_sec": time.time() - started_predict,
            "prediction_rows": int(len(predictions)),
            "prediction_sha256": seal["sha256"],
            "selected_candidate": candidate,
            "refit_epochs": int(selected["refit_epochs"]),
        }
        write_json(origin_dir / "run_meta.json", meta)
        origin_rows.append(meta)
    write_json(
        root / "prediction_run_summary.json",
        {
            "schema": "card10_r85_deepar_fair_prediction_summary_v1",
            "raw_sha256": panel.raw_sha256,
            "grid_sha256": sha256_path(Path(args.grid)),
            "selected_sha256": sha256_path(Path(args.selected)),
            "origins": origin_rows,
            "access_flags": grid["access_flags"],
        },
    )


def run_evaluate(args: argparse.Namespace) -> None:
    grid = load_grid(Path(args.grid))
    panel = load_panel(Path(args.nyiso))
    prediction_root = Path(args.prediction_root)
    output = Path(args.out_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("metric output directory must be new and empty")
    output.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for origin_dir in sorted(path for path in prediction_root.iterdir() if path.is_dir()):
        prediction_path = origin_dir / "predictions.csv.gz"
        seal_path = prediction_path.with_suffix(prediction_path.suffix + ".seal.json")
        if not prediction_path.exists() or not seal_path.exists():
            continue
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if sha256_path(prediction_path) != seal.get("sha256"):
            raise RuntimeError(f"prediction seal mismatch: {prediction_path}")
        prediction = pd.read_csv(prediction_path)
        if len(prediction) != int(seal["rows"]):
            raise RuntimeError("prediction row count mismatch")
        scored = attach_actual(prediction, panel)
        record = metric_record(scored)
        record["origin"] = str(prediction["origin"].iloc[0])
        record["prediction_sha256"] = seal["sha256"]
        records.append(record)
        frames.append(scored)
    if len(records) != 7:
        raise RuntimeError(f"expected seven sealed origins, got {len(records)}")
    per_origin = pd.DataFrame(records).sort_values("origin")
    per_origin.to_csv(output / "metrics_per_origin.csv", index=False)
    all_scored = pd.concat(frames, ignore_index=True)
    aggregate = metric_record(all_scored)
    # Macro values are the paper comparison; pooled values remain diagnostic.
    macro_fields = [
        "mae",
        "rmse",
        "wape",
        "worst_zone_mae",
        "tail_mae_p99",
        "marginal_coverage_q05_q95",
        "simultaneous_11zone_coverage",
        "mean_width_q05_q95",
        "mean_pinball",
        "normalized_wql",
    ]
    for field in macro_fields:
        aggregate[f"macro_{field}"] = float(per_origin[field].mean())
    write_json(output / "metrics_aggregate.json", aggregate)
    manifest = {
        "schema": "card10_r85_deepar_fair_metrics_manifest_v1",
        "raw_sha256": panel.raw_sha256,
        "origin_count": len(records),
        "metric_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_path(path)}
            for path in sorted(output.iterdir())
            if path.is_file()
        },
        "access_flags": grid["access_flags"],
    }
    write_json(output / "MANIFEST.json", manifest)
    print(json.dumps(aggregate, sort_keys=True))


def run_self_test() -> None:
    try:
        safe_path("x/confirmatory_locked/card10_nyiso_2026h1/y")
    except RuntimeError:
        pass
    else:
        raise AssertionError("forbidden path did not fail closed")
    spring = pd.date_range("2024-03-10", "2024-03-11", inclusive="left", freq="h", tz=LOCAL_TZ)
    fall = pd.date_range("2024-11-03", "2024-11-04", inclusive="left", freq="h", tz=LOCAL_TZ)
    if len(spring) != 23 or len(fall) != 25:
        raise AssertionError("DST local-day identity failed")
    if len(set(fall.tz_convert("UTC"))) != 25:
        raise AssertionError("fall-back UTC identity is not unique")
    fake_index = pd.date_range("2024-01-01", periods=72, freq="h", tz=LOCAL_TZ)
    features = dynamic_features(fake_index, np.full(72, 1000.0))
    if features.shape != (8, 72) or not np.isfinite(features).all():
        raise AssertionError("dynamic feature surface failed")
    print("R85_DEEPAR_FAIR_SELF_TEST_PASS")


def run_data_smoke(nyiso: str) -> None:
    panel = load_panel(Path(nyiso))
    days = [
        pd.Timestamp("2024-03-10", tz=LOCAL_TZ),
        pd.Timestamp("2024-07-15", tz=LOCAL_TZ),
        pd.Timestamp("2024-11-03", tz=LOCAL_TZ),
    ]
    entries, meta = make_prediction_entries(panel, "p7_residual", days, 400)
    if len(entries) != 3 * len(ZONES):
        raise AssertionError(f"actual-data prediction entry count failed: {len(entries)}")
    if sorted({len(item["target_positions"]) for item in meta}) != [23, 24, 25]:
        raise AssertionError("actual-data 23/24/25 target-day support failed")
    probe_day = days[1]
    day_t_minus_1 = panel.actual.index[panel.actual.index.date == (probe_day - pd.DateOffset(days=1)).date()]
    perturbed_actual = panel.actual.copy()
    perturbed_actual.loc[day_t_minus_1] += 9999.0
    perturbed = Panel(perturbed_actual, panel.forecast, panel.raw_sha256)
    base_entries, _ = make_prediction_entries(panel, "p7_residual", [probe_day], 400)
    perturbed_entries, _ = make_prediction_entries(perturbed, "p7_residual", [probe_day], 400)
    for base, changed in zip(base_entries, perturbed_entries):
        if not np.array_equal(base["target"], changed["target"]):
            raise AssertionError("day-t-1 outcome changed day-t DeepAR target history")
        if not np.array_equal(base["feat_dynamic_real"], changed["feat_dynamic_real"]):
            raise AssertionError("day-t-1 outcome changed known-future features")
    print("R85_DEEPAR_FAIR_ACTUAL_DATA_SMOKE_PASS", panel.raw_sha256, len(entries))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("develop", "predict", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--nyiso", required=True)
        command.add_argument("--grid", required=True)
        command.add_argument("--out-dir", required=True)
        if name in ("develop", "predict"):
            command.add_argument("--use-gpu", action="store_true")
        if name == "predict":
            command.add_argument("--selected", required=True)
            command.add_argument("--chunk-days", type=int, default=14)
        if name == "evaluate":
            command.add_argument("--prediction-root", required=True)
    sub.add_parser("self-test")
    data_smoke = sub.add_parser("data-smoke")
    data_smoke.add_argument("--nyiso", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "develop":
        run_development(args)
    elif args.command == "predict":
        run_evaluation_predictions(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "self-test":
        run_self_test()
    else:
        run_data_smoke(args.nyiso)


if __name__ == "__main__":
    main()
