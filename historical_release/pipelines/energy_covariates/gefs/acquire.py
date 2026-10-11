#!/usr/bin/env python3
"""Resumable D-2 GEFS temperature/humidity byte-range acquisition."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import os
import platform
import socket
import threading
import time
from pathlib import Path
from typing import Iterable

import requests

from . import acquisition_contract as C
from . import source_layout as R71

CONNECT_TIMEOUT = 20
READ_TIMEOUT = 120
MAX_RETRIES = 5
LEADS_D2 = (54, 60, 66, 72)
VARIABLES = (
    ("v_tmp2m", "TMP", "2 m above ground"),
    ("v_rh2m", "RH", "2 m above ground"),
)
DEFAULT_CAP_BYTES = 12 * 2**30
_THREAD_LOCAL = threading.local()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=2, max_retries=0)
        session.mount("https://", adapter)
        _THREAD_LOCAL.session = session
    return session


def get_retry(session: requests.Session, url: str, headers: dict | None = None) -> requests.Response:
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, headers=headers or {}, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            if response.status_code in (200, 206, 403, 404):
                return response
            last = RuntimeError(f"HTTP {response.status_code}: {url}")
        except requests.RequestException as exc:
            last = exc
        time.sleep(min(20.0, 1.6**attempt))
    raise last or RuntimeError(f"request retries exhausted: {url}")


class StorageBudget:
    def __init__(self, used: int, cap: int):
        if used < 0 or cap <= 0 or used > cap:
            raise ValueError("R77_INVALID_STORAGE_BUDGET")
        self.used = int(used)
        self.cap = int(cap)
        self._lock = threading.Lock()

    def reserve(self, amount: int) -> bool:
        if amount < 0:
            raise ValueError("R77_NEGATIVE_RESERVATION")
        with self._lock:
            if self.used + amount > self.cap:
                return False
            self.used += amount
            return True

    def release(self, amount: int) -> None:
        with self._lock:
            self.used -= amount
            if self.used < 0:
                raise RuntimeError("R77_STORAGE_BUDGET_UNDERFLOW")


def date_range(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    day = start
    while day <= end:
        yield day
        day += dt.timedelta(days=1)


def cycle_date_d2(day: dt.date) -> dt.date:
    return day - dt.timedelta(days=2)


def paired_d1_lead(lead_d2: int) -> int:
    if lead_d2 not in LEADS_D2:
        raise ValueError("R77_UNFROZEN_D2_LEAD")
    return lead_d2 - 24


def valid_time(day: dt.date, lead_d2: int) -> dt.datetime:
    return dt.datetime.combine(cycle_date_d2(day), dt.time(), tzinfo=dt.timezone.utc) + dt.timedelta(hours=lead_d2)


def paired_valid_time(day: dt.date, lead_d2: int) -> dt.datetime:
    d1_init = dt.datetime.combine(day - dt.timedelta(days=1), dt.time(), tzinfo=dt.timezone.utc)
    return d1_init + dt.timedelta(hours=paired_d1_lead(lead_d2))


def object_key(day: dt.date, source: str, lead_d2: int) -> str:
    route = C.route_for_delivery_day(day)
    C.source_role(day, source)
    cycle_ds = cycle_date_d2(day).strftime("%Y%m%d")
    return R71.object_key_for_lead(route.stratum, cycle_ds, source, lead_d2)


def object_url(day: dt.date, source: str, lead_d2: int) -> str:
    return f"{R71.BUCKET_URL}/{object_key(day, source, lead_d2)}"


def cell_key(day: dt.date, source: str, lead_d2: int) -> str:
    return f"{day.isoformat()}|{C.route_for_delivery_day(day).stratum}|d2_00z|{source}|f{lead_d2:03d}"


def raw_relpath(day: dt.date, source: str, lead_d2: int) -> Path:
    return (
        Path("raw") / C.route_for_delivery_day(day).stratum / day.isoformat()
        / f"{source}_f{lead_d2:03d}.tmprh.block"
    )


def build_jobs(start: dt.date, end: dt.date) -> list[tuple[dt.date, str, int]]:
    jobs: list[tuple[dt.date, str, int]] = []
    for day in date_range(start, end):
        route = C.route_for_delivery_day(day)
        for lead in LEADS_D2:
            for source in route.sources:
                jobs.append((day, source, lead))
    return jobs


def _required_entries(entries: list[R71.IdxEntry]) -> list[tuple[str, str, str, R71.IdxEntry | None]]:
    return [(vid, var, level, R71.find_message(entries, var, level)) for vid, var, level in VARIABLES]


def acquire_cell(staging: Path, day: dt.date, source: str, lead_d2: int, budget: StorageBudget) -> dict:
    route = C.route_for_delivery_day(day)
    key = cell_key(day, source, lead_d2)
    cycle = cycle_date_d2(day)
    cycle_ds = cycle.strftime("%Y%m%d")
    valid = valid_time(day, lead_d2)
    paired = paired_valid_time(day, lead_d2)
    if valid != paired:
        raise AssertionError("R77_REVISION_VALID_TIME_MISMATCH")
    url = object_url(day, source, lead_d2)
    idx_url = url + ".idx"
    session = thread_session()
    base = {
        "schema": "card10_r77_gefs_revision_cell_v1",
        "cell_key": key,
        "delivery_day": day.isoformat(),
        "cycle_init_utc": f"{cycle.isoformat()}T00:00:00Z",
        "paired_d1_cycle_init_utc": f"{(day - dt.timedelta(days=1)).isoformat()}T00:00:00Z",
        "stratum": route.stratum,
        "source": source,
        "source_role": C.source_role(day, source),
        "aggregation_identity": route.aggregation,
        "product_identity": route.product,
        "lead_d2": lead_d2,
        "paired_d1_lead": paired_d1_lead(lead_d2),
        "valid_utc": valid.isoformat(),
        "paired_valid_utc": paired.isoformat(),
        "object_key": object_key(day, source, lead_d2),
        "object_url": url,
        "idx_url": idx_url,
        "frozen_cycle_rule": "00Z_on_calendar_day_t_minus_2_only",
        "forecast_revision_pair": "D2_00Z_to_D1_00Z_same_valid_time",
        "later_cycle_backfill_used": False,
        "observed_weather_used": False,
        "outcome_values_accessed": False,
    }
    try:
        idx_response = get_retry(session, idx_url)
    except Exception as exc:
        return {**base, "status": "IDX_FETCH_FAILURE", "error": repr(exc), "completed_utc": utcnow()}
    idx_meta = {
        "http_status": idx_response.status_code,
        "content_length": len(idx_response.content),
        "sha256": sha256_bytes(idx_response.content),
        "etag": idx_response.headers.get("ETag"),
        "last_modified": idx_response.headers.get("Last-Modified"),
    }
    if idx_response.status_code != 200:
        return {**base, "status": "STRUCTURALLY_MISSING_FROZEN_CYCLE", "idx": idx_meta, "completed_utc": utcnow()}
    entries = R71.parse_idx(idx_response.text)
    required = _required_entries(entries)
    missing = [vid for vid, _, _, entry in required if entry is None]
    if missing:
        return {
            **base, "status": "REQUIRED_VARIABLE_ABSENT", "missing_vars": missing,
            "idx": idx_meta, "completed_utc": utcnow(),
        }
    first_tag = entries[0].dtag if entries else ""
    admissible, reason = R71.cutoff_check(first_tag, cycle_ds)
    if not admissible:
        return {
            **base, "status": "CUTOFF_VIOLATION", "cutoff_reason": reason,
            "idx_init_tag": first_tag, "idx": idx_meta, "completed_utc": utcnow(),
        }
    present = [(vid, var, level, entry) for vid, var, level, entry in required if entry is not None]
    starts = [entry.offset for _, _, _, entry in present]
    ends = [R71.message_byte_range(entries, entry)[1] for _, _, _, entry in present]
    if any(value is None for value in ends):
        return {**base, "status": "OPEN_ENDED_REQUIRED_MESSAGE", "idx": idx_meta, "completed_utc": utcnow()}
    start = min(starts)
    end = max(int(value) for value in ends)
    range_header = f"bytes={start}-{end}"
    try:
        response = get_retry(session, url, {"Range": range_header})
    except Exception as exc:
        return {
            **base, "status": "BLOCK_FETCH_FAILURE", "range": range_header,
            "idx": idx_meta, "error": repr(exc), "completed_utc": utcnow(),
        }
    if response.status_code != 206:
        return {
            **base, "status": "RANGE_PROTOCOL_FAILURE", "range": range_header,
            "http_status": response.status_code, "idx": idx_meta, "completed_utc": utcnow(),
        }
    block = response.content
    if not budget.reserve(len(block)):
        return {
            **base, "status": "STORAGE_CAP_REJECTED", "range": range_header,
            "retrieved_but_not_staged_bytes": len(block), "storage_cap_bytes": budget.cap,
            "idx": idx_meta, "completed_utc": utcnow(),
        }
    relpath = raw_relpath(day, source, lead_d2)
    destination = staging / relpath
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(block)
        os.replace(temporary, destination)
    except Exception:
        budget.release(len(block))
        raise
    vars_meta: dict[str, dict] = {}
    for vid, var, level, entry in present:
        relative_start = entry.offset - start
        _, absolute_end = R71.message_byte_range(entries, entry)
        relative_end = int(absolute_end) + 1 - start
        message = block[relative_start:relative_end]
        vars_meta[vid] = {
            "grib_var": var,
            "level": level,
            "forecast_tag": entry.ftag,
            "message_offset_absolute": entry.offset,
            "message_offset_in_block": relative_start,
            "message_length": len(message),
            "message_sha256": sha256_bytes(message),
            "starts_with_GRIB": message.startswith(b"GRIB"),
        }
    status = "FETCHED" if all(value["starts_with_GRIB"] for value in vars_meta.values()) else "GRIB_BOUNDARY_FAILURE"
    return {
        **base,
        "status": status,
        "idx": idx_meta,
        "range": range_header,
        "http_status": response.status_code,
        "content_range": response.headers.get("Content-Range"),
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "retrieved_bytes": len(block),
        "block_sha256": sha256_bytes(block),
        "raw_relpath": relpath.as_posix(),
        "vars": vars_meta,
        "completed_utc": utcnow(),
    }


def read_last_records(ledger: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if ledger.is_file():
        with ledger.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    records[record["cell_key"]] = record
    return records


def is_complete(staging: Path, record: dict) -> bool:
    if record.get("status") != "FETCHED" or not record.get("raw_relpath"):
        return False
    path = staging / record["raw_relpath"]
    return path.is_file() and path.stat().st_size == int(record["retrieved_bytes"]) and sha256_file(path) == record["block_sha256"]


def _tree_hash(records: list[dict]) -> str:
    h = hashlib.sha256()
    for record in sorted(records, key=lambda value: value["raw_relpath"]):
        h.update(
            f"{record['raw_relpath']}\0{record['block_sha256']}\0{record['retrieved_bytes']}\n".encode("utf-8")
        )
    return h.hexdigest()


def acquire(staging: Path, start: dt.date, end: dt.date, concurrency: int, cap_bytes: int) -> dict:
    began = time.time()
    staging.mkdir(parents=True, exist_ok=True)
    ledger = staging / "cells.jsonl"
    last = read_last_records(ledger)
    jobs = build_jobs(start, end)
    verified: dict[str, dict] = {}
    for job in jobs:
        key = cell_key(*job)
        record = last.get(key, {})
        if is_complete(staging, record):
            verified[key] = record
    used = sum(int(record["retrieved_bytes"]) for record in verified.values())
    budget = StorageBudget(used, cap_bytes)
    pending = [job for job in jobs if cell_key(*job) not in verified]
    print(
        f"[R77 acquire] total={len(jobs)} resumed={len(verified)} pending={len(pending)} "
        f"concurrency={concurrency} used={used} cap={cap_bytes}", flush=True,
    )
    status_counts: dict[str, int] = {}
    bytes_new = 0
    with ledger.open("a", encoding="utf-8", newline="\n") as handle:
        with futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            submitted = {executor.submit(acquire_cell, staging, *job, budget): job for job in pending}
            for completed, future in enumerate(futures.as_completed(submitted), 1):
                try:
                    record = future.result()
                except Exception as exc:
                    day, source, lead = submitted[future]
                    record = {
                        "schema": "card10_r77_gefs_revision_cell_v1",
                        "cell_key": cell_key(day, source, lead),
                        "delivery_day": day.isoformat(), "source": source, "lead_d2": lead,
                        "status": "UNHANDLED_EXCEPTION", "error": repr(exc),
                        "completed_utc": utcnow(), "outcome_values_accessed": False,
                    }
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                status = record.get("status", "UNKNOWN")
                status_counts[status] = status_counts.get(status, 0) + 1
                bytes_new += int(record.get("retrieved_bytes", 0))
                if completed % 250 == 0 or completed == len(pending):
                    print(
                        f"[R77 acquire] {completed}/{len(pending)} bytes_new={bytes_new} "
                        f"budget_used={budget.used} status={status_counts}", flush=True,
                    )
    final = read_last_records(ledger)
    selected = [final.get(cell_key(*job), {}) for job in jobs]
    final_counts: dict[str, int] = {}
    for record in selected:
        status = record.get("status", "MISSING_LEDGER_RECORD")
        final_counts[status] = final_counts.get(status, 0) + 1
    fetched = [record for record in selected if record.get("status") == "FETCHED" and is_complete(staging, record)]
    summary = {
        "schema": "card10_r77_gefs_revision_acquisition_summary_v1",
        "generated_utc": utcnow(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "expected_cells": len(jobs),
        "verified_fetched_cells": len(fetched),
        "status_counts": final_counts,
        "ledger_sha256": sha256_file(ledger),
        "ledger_bytes": ledger.stat().st_size,
        "retrieved_bytes_bound_records": sum(int(record["retrieved_bytes"]) for record in fetched),
        "storage_cap_bytes": cap_bytes,
        "raw_payload_tree_convention": "sha256(sorted(raw_relpath\\0block_sha256\\0retrieved_bytes\\n))",
        "raw_payload_tree_sha256": _tree_hash(fetched),
        "d2_leads": list(LEADS_D2),
        "paired_d1_leads": [paired_d1_lead(lead) for lead in LEADS_D2],
        "variables": [vid for vid, _, _ in VARIABLES],
        "cutoff_violations": final_counts.get("CUTOFF_VIOLATION", 0),
        "valid_time_mismatches": sum(record.get("valid_utc") != record.get("paired_valid_utc") for record in fetched),
        "elapsed_seconds": time.time() - began,
        "concurrency": concurrency,
        "code_commit": os.environ.get("CARD10_CODE_COMMIT", "UNSET_NONPAPER"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": os.sys.version,
        "requests": requests.__version__,
        "later_cycle_backfill_used": False,
        "observed_weather_used": False,
        "outcome_values_accessed": False,
        "confirmatory_2026_outcome_accessed": False,
        "complete": len(fetched) == len(jobs) and final_counts == {"FETCHED": len(jobs)},
    }
    summary_path = staging / "acquisition_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "schema": "card10_r77_gefs_revision_external_manifest_v1",
        "payloads": [
            {"path": "cells.jsonl", "bytes": ledger.stat().st_size, "sha256": sha256_file(ledger)},
            {"path": "acquisition_summary.json", "bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
        ],
        "raw_payload_tree_sha256": summary["raw_payload_tree_sha256"],
        "raw_payload_bytes": summary["retrieved_bytes_bound_records"],
        "self_referential_entries_excluded": True,
        "paper_table_eligible": False,
    }
    manifest_path = staging / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (staging / "MANIFEST.sha256").write_text(
        f"{sha256_file(manifest_path)}  MANIFEST.json\n", encoding="ascii", newline="\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--start", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--storage-cap-bytes", type=int, default=DEFAULT_CAP_BYTES)
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start must not exceed --end")
    if not 1 <= args.concurrency <= 96:
        parser.error("--concurrency must be in [1,96]")
    acquire(args.staging, args.start, args.end, args.concurrency, args.storage_cap_bytes)


if __name__ == "__main__":
    main()
