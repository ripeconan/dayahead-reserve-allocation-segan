#!/usr/bin/env python3
"""Download and prepare NYISO zonal day-ahead forecast and actual-load data.

The official `isolf` daily files contain several forecast days.  The frozen
strict-D-1 representation keeps only the first delivery date encoded by each
member filename.  Five-minute `pal` values are averaged within physical local
hours after explicit EST/EDT disambiguation, then all timestamps are stored in
UTC.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


ZONE_MAP = {
    "Capitl": "CAPITL",
    "Centrl": "CENTRL",
    "Dunwod": "DUNWOD",
    "Genese": "GENESE",
    "Hud Vl": "HUD VL",
    "Longil": "LONGIL",
    "Mhk Vl": "MHK VL",
    "Millwd": "MILLWD",
    "N.Y.C.": "N.Y.C.",
    "North": "NORTH",
    "West": "WEST",
}
ZONES = list(ZONE_MAP.values())
BASE_URL = "https://mis.nyiso.com/public/csv/{dataset}/{ym}01{dataset}_csv.zip"


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


@dataclass(frozen=True)
class Archive:
    dataset: str
    ym: str
    path: Path

    @property
    def url(self) -> str:
        return BASE_URL.format(dataset=self.dataset, ym=self.ym)


def months(start_year: int, end_year: int) -> list[str]:
    return [f"{year}{month:02d}" for year in range(start_year, end_year + 1) for month in range(1, 13)]


def download_one(archive: Archive, retries: int = 4) -> tuple[str, int, bool]:
    archive.path.parent.mkdir(parents=True, exist_ok=True)
    if archive.path.exists() and archive.path.stat().st_size > 1000:
        try:
            with zipfile.ZipFile(archive.path) as handle:
                if handle.testzip() is None:
                    return archive.path.name, archive.path.stat().st_size, True
        except zipfile.BadZipFile:
            pass
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                archive.url, headers={"User-Agent": "energy-uq-reproduction/1.0"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                raw = response.read()
            with zipfile.ZipFile(BytesIO(raw)) as handle:
                bad = handle.testzip()
                if bad is not None:
                    raise zipfile.BadZipFile(f"corrupt member {bad}")
            temporary = archive.path.with_suffix(".zip.part")
            temporary.write_bytes(raw)
            temporary.replace(archive.path)
            return archive.path.name, len(raw), False
        except Exception as exc:  # network retries are deliberately bounded
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"failed {archive.url}: {last_error}")


def delivery_date_from_member(name: str) -> pd.Timestamp:
    stem = Path(name).name[:8]
    return pd.to_datetime(stem, format="%Y%m%d")


def parse_forecast(archive: Archive) -> tuple[pd.DataFrame, dict[str, int]]:
    parts: list[pd.DataFrame] = []
    stats = {"members": 0, "rows_all_horizons": 0, "rows_d1": 0, "bad_timestamp": 0, "duplicates": 0}
    with zipfile.ZipFile(archive.path) as handle:
        for member in sorted(name for name in handle.namelist() if name.lower().endswith(".csv")):
            delivery = delivery_date_from_member(member)
            frame = pd.read_csv(handle.open(member), low_memory=False)
            missing = set(ZONE_MAP) - set(frame.columns)
            if missing:
                raise ValueError(f"{member}: missing forecast columns {sorted(missing)}")
            naive = pd.to_datetime(frame["Time Stamp"], format="%m/%d/%Y %H:%M", errors="coerce")
            keep = naive.dt.normalize().eq(delivery)
            current = frame.loc[keep, list(ZONE_MAP)].copy()
            current_naive = pd.DatetimeIndex(naive.loc[keep])
            try:
                local = current_naive.tz_localize("America/New_York", ambiguous="infer", nonexistent="raise")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{member}: DST localization failed: {exc}") from exc
            current.index = local.tz_convert("UTC")
            current.columns = ZONES
            current = current.apply(pd.to_numeric, errors="coerce")
            stats["members"] += 1
            stats["rows_all_horizons"] += len(frame)
            stats["rows_d1"] += len(current)
            stats["bad_timestamp"] += int(current.index.isna().sum())
            stats["duplicates"] += int(current.index.duplicated(keep="last").sum())
            current = current.loc[~current.index.duplicated(keep="last")]
            parts.append(current)
    output = pd.concat(parts).sort_index()
    stats["duplicates"] += int(output.index.duplicated(keep="last").sum())
    return output.loc[~output.index.duplicated(keep="last")], stats


def parse_actual(archive: Archive) -> tuple[pd.DataFrame, dict[str, int]]:
    parts: list[pd.DataFrame] = []
    stats = {"members": 0, "rows_raw": 0, "rows_zones": 0, "bad_timestamp": 0, "duplicates": 0, "hourly_cells_lt12": 0}
    with zipfile.ZipFile(archive.path) as handle:
        for member in sorted(name for name in handle.namelist() if name.lower().endswith(".csv")):
            delivery = delivery_date_from_member(member)
            frame = pd.read_csv(handle.open(member), low_memory=False)
            required = {"Time Stamp", "Time Zone", "Name", "Load"}
            if not required.issubset(frame.columns):
                raise ValueError(f"{member}: missing actual columns {sorted(required - set(frame.columns))}")
            naive = pd.to_datetime(frame["Time Stamp"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
            keep = naive.dt.normalize().eq(delivery) & frame["Name"].isin(ZONES)
            current = frame.loc[keep, ["Time Zone", "Name", "Load"]].copy()
            current_naive = pd.DatetimeIndex(naive.loc[keep])
            labels = current["Time Zone"].astype(str).str.upper()
            unknown = ~labels.isin(["EST", "EDT"])
            if unknown.any():
                raise ValueError(f"{member}: unknown time-zone labels {sorted(labels.loc[unknown].unique())}")
            local = current_naive.tz_localize(
                "America/New_York",
                ambiguous=labels.eq("EDT").to_numpy(),
                nonexistent="NaT",
            )
            current["timestamp_utc"] = local.tz_convert("UTC")
            current["Load"] = pd.to_numeric(current["Load"], errors="coerce")
            stats["members"] += 1
            stats["rows_raw"] += len(frame)
            stats["rows_zones"] += len(current)
            stats["bad_timestamp"] += int(current["timestamp_utc"].isna().sum())
            subset = ["timestamp_utc", "Name"]
            stats["duplicates"] += int(current.duplicated(subset=subset, keep="last").sum())
            current = current.dropna(subset=["timestamp_utc"]).drop_duplicates(subset=subset, keep="last")
            current["hour_utc"] = current["timestamp_utc"].dt.floor("h")
            grouped = current.groupby(["hour_utc", "Name"], observed=True)["Load"].agg(["mean", "count"])
            stats["hourly_cells_lt12"] += int((grouped["count"] < 12).sum())
            hourly = grouped["mean"].unstack("Name").reindex(columns=ZONES)
            parts.append(hourly)
    output = pd.concat(parts).sort_index()
    stats["duplicates"] += int(output.index.duplicated(keep="last").sum())
    return output.loc[~output.index.duplicated(keep="last")], stats


def ideal_index(year: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="America/New_York")
    return pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")


def quality_tables(forecast: pd.DataFrame, actual: pd.DataFrame, start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    zone_rows: list[dict[str, object]] = []
    complete_rows: list[dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        expected = ideal_index(year)
        f = forecast.reindex(expected)
        a = actual.reindex(expected)
        expected_hours = len(expected)
        for zone in ZONES:
            forecast_present = int(f[zone].notna().sum())
            actual_present = int(a[zone].notna().sum())
            paired = int((f[zone].notna() & a[zone].notna()).sum())
            zone_rows.append({
                "year": year,
                "zone": zone,
                "expected_hours": expected_hours,
                "forecast_present": forecast_present,
                "forecast_missing_rate": 1 - forecast_present / expected_hours,
                "actual_present": actual_present,
                "actual_missing_rate": 1 - actual_present / expected_hours,
                "paired_present": paired,
                "paired_missing_rate": 1 - paired / expected_hours,
            })
        complete = f.notna().all(axis=1) & a.notna().all(axis=1)
        complete_rows.append({
            "year": year,
            "expected_hours": expected_hours,
            "complete_hours": int(complete.sum()),
            "complete_rate": float(complete.mean()),
            "missing_rate": float(1 - complete.mean()),
        })
    return pd.DataFrame(zone_rows), pd.DataFrame(complete_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/downloads/nyiso"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/nyiso"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archives = [
        Archive(dataset, ym, args.raw_dir / dataset / f"{ym}01{dataset}_csv.zip")
        for ym in months(args.start_year, args.end_year)
        for dataset in ("isolf", "pal")
    ]
    cached = 0
    downloaded_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, archive): archive for archive in archives}
        for completed, future in enumerate(as_completed(futures), start=1):
            name, size, was_cached = future.result()
            cached += int(was_cached)
            downloaded_bytes += 0 if was_cached else size
            if completed % 12 == 0 or completed == len(archives):
                print(f"download {completed}/{len(archives)} cached={cached} new_MB={downloaded_bytes / 1e6:.1f}", flush=True)

    forecast_parts: list[pd.DataFrame] = []
    actual_parts: list[pd.DataFrame] = []
    parse_stats: dict[str, dict[str, int]] = {}
    for index, ym in enumerate(months(args.start_year, args.end_year), start=1):
        f_archive = Archive("isolf", ym, args.raw_dir / "isolf" / f"{ym}01isolf_csv.zip")
        a_archive = Archive("pal", ym, args.raw_dir / "pal" / f"{ym}01pal_csv.zip")
        forecast, f_stats = parse_forecast(f_archive)
        actual, a_stats = parse_actual(a_archive)
        forecast_parts.append(forecast)
        actual_parts.append(actual)
        parse_stats[ym] = {**{f"forecast_{k}": v for k, v in f_stats.items()}, **{f"actual_{k}": v for k, v in a_stats.items()}}
        if index % 12 == 0:
            print(f"parse {index}/{12 * (args.end_year - args.start_year + 1)}", flush=True)

    forecast = pd.concat(forecast_parts).sort_index()
    actual = pd.concat(actual_parts).sort_index()
    if forecast.index.duplicated().any() or actual.index.duplicated().any():
        raise RuntimeError("cross-month duplicate UTC hour remains after parsing")

    zone_quality, complete_quality = quality_tables(forecast, actual, args.start_year, args.end_year)
    union = forecast.join(actual, how="outer", lsuffix="_forecast", rsuffix="_actual")
    complete = union.dropna(how="any").copy()
    local = complete.index.tz_convert("America/New_York")
    complete.insert(0, "delivery_hour", local.hour)
    complete.insert(0, "delivery_date", local.date)
    complete.insert(0, "timestamp_utc", complete.index.strftime("%Y-%m-%dT%H:%M:%SZ"))
    complete = complete.reset_index(drop=True)

    stem = f"nyiso_zonal_{args.start_year}_{args.end_year}"
    data_path = args.output_dir / f"{stem}.csv.gz"
    zone_path = args.output_dir / f"{stem}_zone_quality.csv"
    complete_path = args.output_dir / f"{stem}_complete_quality.csv"
    manifest_path = args.output_dir / f"{stem}_manifest.json"
    complete.to_csv(
        data_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    zone_quality.to_csv(zone_path, index=False)
    complete_quality.to_csv(complete_path, index=False)
    archive_ledger = [
        {
            "dataset": archive.dataset,
            "month": archive.ym,
            "url": archive.url,
            "bytes": archive.path.stat().st_size,
            "sha256": sha256_file(archive.path),
        }
        for archive in archives
    ]
    content_bytes, content_sha256 = gzip_content_identity(data_path)
    manifest = {
        "schema": "nyiso-zonal-acquisition-v1",
        "source": "NYISO MIS public isolf/pal monthly archives",
        "start_year": args.start_year,
        "end_year": args.end_year,
        "zones": ZONES,
        "archives": len(archives),
        "cached_archives": cached,
        "new_download_bytes": downloaded_bytes,
        "forecast_utc_hours": len(forecast),
        "actual_utc_hours": len(actual),
        "complete_utc_hours": len(complete),
        "archive_ledger": archive_ledger,
        "derived_output": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "uncompressed_bytes": content_bytes,
            "uncompressed_sha256": content_sha256,
        },
        "parse_stats": parse_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {data_path} rows={len(complete)}")
    print(complete_quality.to_string(index=False))
    print(f"max zone paired missing rate={zone_quality.paired_missing_rate.max():.6f}")


if __name__ == "__main__":
    main()
