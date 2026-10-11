#!/usr/bin/env python3
"""Reconstruct the sequential GEFCom2014 load track as 24-D daily profiles."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HOURS = list(range(1, 25))
TARGET_START = pd.Timestamp("2010-10-01")


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def gzip_content_identity(path: Path) -> tuple[int, str]:
    value = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            value.update(block)
    return size, value.hexdigest()


def parse_timestamp(values: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray, pd.DatetimeIndex]:
    parsed = pd.to_datetime(values.astype(str), format="%m%d%Y %H:%M", errors="raise")
    hour = parsed.dt.hour.to_numpy()
    horizon = np.where(hour == 0, 24, hour)
    delivery = parsed.dt.normalize() - pd.to_timedelta((hour == 0).astype(int), unit="D")
    return pd.DatetimeIndex(parsed), horizon, pd.DatetimeIndex(delivery)


def tidy_hourly(
    frame: pd.DataFrame,
    value_column: str,
    start_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    # The published TIMESTAMP field concatenates unpadded month and day, so
    # strings such as 1112001 are intrinsically ambiguous.  Every task file is
    # documented and stored as a continuous hourly sequence; reconstructing
    # from its known first delivery hour avoids a lossy parser heuristic.
    timestamp = pd.date_range(start_timestamp, periods=len(frame), freq="h")
    hour = timestamp.hour.to_numpy()
    horizon = np.where(hour == 0, 24, hour)
    delivery = timestamp.normalize() - pd.to_timedelta((hour == 0).astype(int), unit="D")
    output = pd.DataFrame(
        {
            "timestamp": timestamp,
            "delivery_date": delivery,
            "horizon": horizon,
            "value": pd.to_numeric(frame[value_column], errors="coerce"),
        }
    )
    output = output.dropna(subset=["value"])
    if output.duplicated(["delivery_date", "horizon"]).any():
        raise ValueError("duplicate delivery_date/horizon")
    return output


def task_paths(root: Path, task: int) -> tuple[Path, Path]:
    directory = root / f"Task {task}"
    return directory / f"L{task}-train.csv", directory / f"L{task}-benchmark.csv"


def reconstruct_actual(root: Path) -> pd.DataFrame:
    task1_train, _ = task_paths(root, 1)
    history = tidy_hourly(
        pd.read_csv(task1_train, low_memory=False),
        "LOAD",
        pd.Timestamp("2001-01-01 01:00"),
    )
    history["actual_source"] = "task1_history"
    parts = [history]
    for task in range(1, 15):
        next_train, _ = task_paths(root, task + 1)
        current = tidy_hourly(
            pd.read_csv(next_train, low_memory=False),
            "LOAD",
            TARGET_START + pd.DateOffset(months=task - 1) + pd.Timedelta(hours=1),
        )
        current["actual_source"] = f"task{task + 1}_increment"
        parts.append(current)
    solution = tidy_hourly(
        pd.read_csv(root / "Solution to Task 15" / "solution15_L.csv"),
        "LOAD",
        TARGET_START + pd.DateOffset(months=14) + pd.Timedelta(hours=1),
    )
    solution["actual_source"] = "task15_solution"
    parts.append(solution)
    output = pd.concat(parts, ignore_index=True).sort_values(["delivery_date", "horizon"])
    duplicate = output.duplicated(["delivery_date", "horizon"], keep=False)
    if duplicate.any():
        duplicated = output.loc[duplicate]
        spread = duplicated.groupby(["delivery_date", "horizon"])["value"].agg(np.ptp)
        if (spread > 1e-9).any():
            raise ValueError("conflicting actual values across incremental files")
        output = output.drop_duplicates(["delivery_date", "horizon"], keep="last")
    return output


def reconstruct_task_forecast(root: Path) -> pd.DataFrame:
    parts = []
    for task in range(1, 16):
        _, benchmark = task_paths(root, task)
        frame = pd.read_csv(benchmark, low_memory=False)
        if "0.5" not in frame.columns:
            raise ValueError(f"{benchmark}: missing median benchmark")
        current = tidy_hourly(
            frame,
            "0.5",
            TARGET_START + pd.DateOffset(months=task - 1) + pd.Timedelta(hours=1),
        )
        current["forecast_source"] = f"task{task}_benchmark_median"
        current["task"] = task
        parts.append(current)
    output = pd.concat(parts, ignore_index=True).sort_values(["delivery_date", "horizon"])
    if output.duplicated(["delivery_date", "horizon"]).any():
        raise ValueError("overlapping benchmark target timestamps")
    return output


def historical_lag_forecast(actual: pd.DataFrame) -> pd.DataFrame:
    lookup = actual.set_index(["delivery_date", "horizon"])["value"]
    rows = []
    for row in actual.itertuples(index=False):
        previous = pd.Timestamp(row.delivery_date) - pd.DateOffset(years=1)
        key = (previous, int(row.horizon))
        if key in lookup.index:
            rows.append(
                {
                    "delivery_date": row.delivery_date,
                    "horizon": row.horizon,
                    "value": float(lookup.loc[key]),
                    "forecast_source": "same_calendar_day_previous_year",
                    "task": 0,
                }
            )
    return pd.DataFrame(rows)


def daily_profiles(actual: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    paired = actual.merge(
        forecast, on=["delivery_date", "horizon"], how="inner", suffixes=("_actual", "_forecast")
    )
    counts = paired.groupby("delivery_date")["horizon"].agg(["count", "nunique"])
    valid_dates = counts.index[(counts["count"] == 24) & (counts["nunique"] == 24)]
    paired = paired.loc[paired["delivery_date"].isin(valid_dates)]
    actual_wide = paired.pivot(index="delivery_date", columns="horizon", values="value_actual")
    forecast_wide = paired.pivot(index="delivery_date", columns="horizon", values="value_forecast")
    actual_wide = actual_wide.reindex(columns=HOURS)
    forecast_wide = forecast_wide.reindex(columns=HOURS)
    output = pd.DataFrame(index=actual_wide.index)
    output.index.name = "delivery_date"
    source = paired.groupby("delivery_date")["forecast_source"].first()
    task = paired.groupby("delivery_date")["task"].first()
    output["forecast_source"] = source
    output["task"] = task.astype(int)
    for hour in HOURS:
        output[f"H{hour:02d}_forecast"] = forecast_wide[hour]
    for hour in HOURS:
        output[f"H{hour:02d}_actual"] = actual_wide[hour]
    return output.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/gefcom2014")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actual = reconstruct_actual(args.root)
    task_forecast = reconstruct_task_forecast(args.root)
    history_forecast = historical_lag_forecast(actual)
    target_start = TARGET_START
    history_forecast = history_forecast.loc[history_forecast["delivery_date"] < target_start]
    forecast = pd.concat([history_forecast, task_forecast], ignore_index=True)
    profiles = daily_profiles(actual, forecast)

    target = profiles.loc[profiles["task"].between(1, 15)]
    expected_days = pd.date_range("2010-10-01", "2011-12-31", freq="D")
    if len(target) != len(expected_days):
        raise RuntimeError(f"target daily profiles {len(target)} != expected {len(expected_days)}")
    if not pd.DatetimeIndex(target["delivery_date"]).equals(expected_days):
        raise RuntimeError("target delivery dates are not the complete expected sequence")
    value_columns = [f"H{hour:02d}_{kind}" for kind in ("forecast", "actual") for hour in HOURS]
    if profiles[value_columns].isna().any().any():
        raise RuntimeError("NaN remains in complete daily profiles")

    data_path = args.output_dir / "gefcom2014_load_daily24.csv.gz"
    profiles.to_csv(
        data_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    content_bytes, content_sha256 = gzip_content_identity(data_path)
    manifest = {
        "schema": "gefcom2014-load-daily24-preparation-v1",
        "source": "GEFCom2014 Load track sequential tasks",
        "daily_profiles": int(len(profiles)),
        "first_delivery_date": str(profiles["delivery_date"].min().date()),
        "last_delivery_date": str(profiles["delivery_date"].max().date()),
        "historical_reference_days": int((profiles["task"] == 0).sum()),
        "competition_target_days": int(len(target)),
        "target_start": "2010-10-01",
        "target_end": "2011-12-31",
        "tasks": {str(int(k)): int(v) for k, v in target.groupby("task").size().items()},
        "dimension": 24,
        "missing_profile_cells": int(profiles[value_columns].isna().sum().sum()),
        "derived_output": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "uncompressed_bytes": content_bytes,
            "uncompressed_sha256": content_sha256,
        },
    }
    (args.output_dir / "gefcom2014_prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"wrote {data_path}")


if __name__ == "__main__":
    main()
