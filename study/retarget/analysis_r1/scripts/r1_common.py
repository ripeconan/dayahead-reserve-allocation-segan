"""R-A (SEGAN R1) shared utilities: deterministic IO, loaders, metrics.

Pure module: no side effects on import (no computation, no file writes).

All inputs are ADMITTED EXPOSED replay artifacts under the read-only evidence
worktree.  This module never writes outside the caller-provided output dir.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Frozen constants (mirror the admitted production/R51 implementations).
ELL = 2                    # strict lag-2, in calendar days
MEMORY = 6                 # recent eligible records
THETA = 1.0                # production trigger threshold
BUDGET_FRAC = 0.15         # horizon cap fraction
K2_SERVICE_THRESHOLD = 0.9  # R51 thresholded-count miss definition
K2_COUNT = 2               # R51 trigger count
PERSIST_THRESHOLD = 0.5    # R1-F4 single-bad-day rule
LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)
LAMBDA_FINE = tuple(round(0.02 * i, 2) for i in range(11))  # 0.00 .. 0.20
N_BOOT = 10_000
BOOT_SEED = 20261011
DELTA_MATERIAL = 0.01
SERVICE_DAY_THRESHOLD = 0.8  # violation-day reporting threshold (as in p0_2)

# Evidence root (read-only) relative to this package:
#   .../mypaper/DayAhead-.../11_SEGAN_retarget_2026-10-10/analysis_r1/scripts/
# parents[4] = D:\mypaper
EVIDENCE_ROOT = Path(__file__).resolve().parents[4] / "literature-infra" / "card10-workers" / "codex-r78fix-r80"

R39_DAILY = "reports/experiment_outputs/card10_p0_4_nyiso_verified_saocp_r39_2026-07-16/daily_policy_trace.csv.gz"
R39_HOURLY = "reports/experiment_outputs/card10_p0_4_nyiso_verified_saocp_r39_2026-07-16/selected_hourly_results.csv.gz"
R02_DAILY = "reports/experiment_outputs/card10_nyiso_saocp_drb_current_r02_2026-07-15/policy_daily_trace.csv.gz"
R02_HOURLY = "reports/experiment_outputs/card10_nyiso_saocp_drb_current_r02_2026-07-15/saocp_hourly_base.csv.gz"
R02_RANDOM = "reports/experiment_outputs/card10_nyiso_saocp_drb_current_r02_2026-07-15/random_draw_endpoints.csv.gz"
R51_DAILY = "reports/experiment_outputs/card10_p0_3_p0_4_cross_dataset_transfer_r51_2026-07-17/daily_trace.csv.gz"
P02_DAILY = "reports/experiment_outputs/card10_mr_p0_2_production_transfer_2026-07-22/daily_trace.csv.gz"
P02_AGG = "reports/experiment_outputs/card10_mr_p0_2_production_transfer_2026-07-22/aggregate_results.csv.gz"
R45_LEDGER = "reports/experiment_outputs/card10_p0_3_cocd_r45_2026-07-16/cocd_daily_ledger.csv.gz"
R45_AGG = "reports/experiment_outputs/card10_p0_3_cocd_r45_2026-07-16/aggregate_results.csv.gz"
R45_RANDOM = "reports/experiment_outputs/card10_p0_3_cocd_r45_2026-07-16/random_draw_metrics.csv.gz"
R43_AGG = "reports/experiment_outputs/card10_p0_3_exact_oracle_exchange_r43_2026-07-16/aggregate_results.csv.gz"
R43_TRACE = "reports/experiment_outputs/card10_p0_3_exact_oracle_exchange_r43_2026-07-16/oracle_exchange_daily_trace.csv.gz"
R43_SOLVER = "reports/experiment_outputs/card10_p0_3_exact_oracle_exchange_r43_2026-07-16/solver_diagnostics.csv.gz"

# Frozen SHA-256 of every input the protocol binds (fail closed on mismatch).
INPUT_HASHES = {
    R39_DAILY: "c12f16f655a11d91ba26b1b942018c592d275a3cb941a4a2f2e35aedc4ed3ca5",
    R39_HOURLY: "f66f795a0f9aa779262d46f9c54c90bcdf7e661a853eeb19383a4fc0f699e9b4",
    R02_DAILY: "4ee214335ae62351b9e9875a91afcc30d5f930df2013bbc10b949db44276f958",
    R02_HOURLY: "ce2ce87e57fb7d6e05b319022b35c67c736706172ecb6658eecc2e483744135c",
    R02_RANDOM: "530dbe426aafc95416f965c7e45647accf68e1acdeca74ae21b26acd3599b723",
    R51_DAILY: "3e84b66dd2fff4a57151d5f67b21516043642c3acb3221bdda175bcfe4f88e42",
    P02_DAILY: "d50a106b2511da0747b5b3e7e96fd52761df7a91c6bc0124e7092c6251dc2b0a",
    P02_AGG: "8498143f30f81a8fc814eebaf5570f3153abdfb2ccf078b02f1f4393eeaece9e",
    R45_LEDGER: "edc3b6714c2961f71f15affd6efaa6e327c4037aa9c861b5c0c0aafcc8f86566",
    R45_AGG: "40a8a6c30f8abb7b9303232703c93d110da6e43014fd21f0e66b07a1becd957d",
    R45_RANDOM: "333e035d185423a90074aa7b408b89e4f48f50d1267b2e22b392609b195050bb",
    R43_AGG: "36e04946bb4f2714351a32cf87dd4e893b8630e351939ea8b8eed09f47d92f89",
    R43_TRACE: "becc34f45ca32ecbfc5ebfd3f9a299162271376a2dadb0b3180464b82bc1ca61",
    R43_SOLVER: "9d92861ef8477941e6588fac50f1843542e375cb82811a2127e5b555f096b772",
}

# Declared horizon end per transfer origin (calendar period named by the
# admitted R51 protocol cohort/origin label).  NYISO ends are derived from the
# origin year label (e.g. 2018->2019 -> 2019-12-31).
QUARTER_END = {
    "Q1": "-03-31", "Q2": "-06-30", "Q3": "-09-30", "Q4": "-12-31",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_fingerprint(code_files: Sequence[str]) -> str:
    """Bind policy/runner code hashes AND every bound input hash.

    Used by resumable runners so a code or input change invalidates stale
    outputs instead of being silently reused.
    """
    payload = json.dumps(
        {"inputs": INPUT_HASHES,
         "code": {name: sha256_file(Path(__file__).resolve().parent / name) for name in code_files}},
        sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def key_set(frame: pd.DataFrame, keys: Sequence[str]) -> set:
    return set(map(tuple, frame[list(keys)].to_numpy().tolist()))


def assert_same_support(left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str],
                        left_name: str, right_name: str) -> dict:
    """Exact support equality: no missing, extra or duplicate key rows."""
    for name, frame in ((left_name, left), (right_name, right)):
        dupes = int(frame.duplicated(list(keys)).sum())
        if dupes:
            raise RuntimeError(f"{name}: {dupes} duplicate key rows on {list(keys)}")
    lset = key_set(left, keys)
    rset = key_set(right, keys)
    missing = sorted(rset - lset)
    extra = sorted(lset - rset)
    if missing or extra:
        raise RuntimeError(f"support mismatch {left_name} vs {right_name}: "
                           f"missing={len(missing)} extra={len(extra)} "
                           f"first_missing={missing[:3]} first_extra={extra[:3]}")
    return {"rows_left": int(len(left)), "rows_right": int(len(right)), "matched": len(lset)}


def crossing_lambda(delta_mean_shortfall: float, delta_action_rate: float,
                    tol: float = 1e-12) -> dict:
    """Pairwise L_lambda crossing: DL(l) = d_shortfall + l * d_rate.

    Returns the crossing value when rates differ; an equal-action-rate pair has
    no crossing and is reported explicitly (None + constant sign).
    """
    if abs(delta_action_rate) <= tol:
        return {"crossing_lambda": None, "equal_action_rate": True,
                "delta_L_constant": float(delta_mean_shortfall),
                "sign": ("positive" if delta_mean_shortfall > 0 else
                         "negative" if delta_mean_shortfall < 0 else "zero")}
    lam = -delta_mean_shortfall / delta_action_rate
    return {"crossing_lambda": float(lam), "equal_action_rate": False,
            "delta_L_constant": None,
            "crossing_in_frozen_fine_grid": bool(0.0 <= lam <= 0.20)}


def verify_inputs(root: Path) -> dict[str, str]:
    """Fail-closed hash verification of every bound input."""
    seen: dict[str, str] = {}
    for rel, expected in INPUT_HASHES.items():
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"missing bound input: {rel}")
        got = sha256_file(path)
        if got != expected:
            raise RuntimeError(f"hash mismatch for {rel}: {got} != {expected}")
        seen[rel] = got
    return seen


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    """Deterministic gzip CSV (mtime=0, no embedded path)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    payload = buffer.getvalue().encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(payload)


def read_gzip_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def load_nyiso_daily(root: Path) -> pd.DataFrame:
    """R39 seven-origin NYISO daily trace: q_t, segments, production actions."""
    frame = read_gzip_csv(root / R39_DAILY)
    need = {"origin", "delivery_date", "layer", "segment_id", "base_daily_coverage", "action"}
    missing = need - set(frame.columns)
    if missing:
        raise RuntimeError(f"R39 daily missing columns: {sorted(missing)}")
    base = frame[frame["layer"] == "base_no_fallback"].copy()
    prod = frame[frame["layer"] == "base_plus_frozen_drb_lag2"].copy()
    key = ["origin", "delivery_date"]
    assert_same_support(base, prod, key, "R39 base layer", "R39 production layer")
    # NOTE: the base_no_fallback layer stores action_cap = 0 (no policy); the
    # production layer carries the real per-origin cap.  Take the cap from the
    # production layer and let Stage 0 verify it against round(0.15 * n_days).
    merged = base[key + ["segment_id", "base_daily_coverage"]].merge(
        prod[key + ["action", "action_cap"]], on=key, how="inner", validate="one_to_one"
    )
    merged = merged.rename(columns={"base_daily_coverage": "q", "action": "production_action"})
    merged["production_action"] = merged["production_action"].astype(bool)
    merged["delivery_date"] = pd.to_datetime(merged["delivery_date"], errors="raise")
    merged = merged.sort_values(["origin", "delivery_date"], kind="mergesort").reset_index(drop=True)
    if merged.duplicated(key).any():
        raise RuntimeError("R39 merged daily has duplicate origin/date keys")
    return merged


def load_nyiso_hourly(root: Path) -> pd.DataFrame:
    """R02 hourly SAOCP base: per-hour thresholds (pre-action) and coverage."""
    frame = read_gzip_csv(root / R02_HOURLY)
    need = {"origin", "delivery_date", "hour", "method", "segment_id", "threshold", "covered"}
    missing = need - set(frame.columns)
    if missing:
        raise RuntimeError(f"R02 hourly missing columns: {sorted(missing)}")
    frame = frame[frame["method"] == "saocp"].copy()
    frame["delivery_date"] = pd.to_datetime(frame["delivery_date"], errors="raise")
    return frame.sort_values(["origin", "delivery_date", "hour"], kind="mergesort").reset_index(drop=True)


def load_transfer_daily(root: Path) -> pd.DataFrame:
    """R51 cross-dataset daily trace restricted to the verified SAOCP base.

    Identity note (verified 2026-10-11, recorded in the R1 protocol): the R51
    column named ``drb_action`` carries the thresholded-count K=2 rule (the
    later p0_2 "r51_thresholded_count_k2"); the CURRENT continuous production
    controller is the p0_2 ``production_shortfall_sum_lag2`` path.  Both are
    merged here so every consumer states which identity it uses.
    """
    frame = read_gzip_csv(root / R51_DAILY)
    need = {"dataset", "cohort", "origin", "method", "delivery_date", "segment_id",
            "base_covered", "threshold", "drb_action", "cocd_action"}
    missing = need - set(frame.columns)
    if missing:
        raise RuntimeError(f"R51 daily missing columns: {sorted(missing)}")
    frame = frame[frame["method"] == "saocp"].copy()
    frame["delivery_date"] = pd.to_datetime(frame["delivery_date"], errors="raise")
    frame = frame.rename(columns={"base_covered": "q", "drb_action": "r51_original_drb_action"})
    frame["r51_original_drb_action"] = frame["r51_original_drb_action"].astype(bool)
    frame["cocd_action"] = frame["cocd_action"].astype(bool)
    frame = frame.sort_values(["cohort", "origin", "delivery_date"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["origin", "delivery_date"]).any():
        raise RuntimeError("R51 SAOCP daily has duplicate origin/date keys")
    p02 = read_gzip_csv(root / P02_DAILY)
    p02 = p02[p02["policy"].isin(["production_shortfall_sum_lag2", "r51_thresholded_count_k2"])]
    prod = p02[p02["policy"] == "production_shortfall_sum_lag2"][
        ["origin", "decision_index", "action", "base_covered", "action_cap"]].rename(
        columns={"action": "production_action", "action_cap": "cap_p0_2"})
    k2 = p02[p02["policy"] == "r51_thresholded_count_k2"][
        ["origin", "decision_index", "action"]].rename(columns={"action": "k2_action"})
    key = ["origin", "decision_index"]
    assert_same_support(frame, prod, key, "R51 SAOCP daily", "p0_2 production rows")
    assert_same_support(frame, k2, key, "R51 SAOCP daily", "p0_2 K2 rows")
    merged = frame.merge(prod, on=key, how="inner", validate="one_to_one")
    merged = merged.merge(k2, on=key, how="inner", validate="one_to_one")
    merged["production_action"] = merged["production_action"].astype(bool)
    merged["k2_action"] = merged["k2_action"].astype(bool)
    q_diff = np.abs(merged["q"].to_numpy(float) - merged["base_covered"].to_numpy(float))
    if q_diff.max() > 1e-9:
        raise RuntimeError("R51 and p0_2 base coverage disagree on transfer cohorts")
    if not np.array_equal(merged["r51_original_drb_action"].to_numpy(bool), merged["k2_action"].to_numpy(bool)):
        raise RuntimeError("R51 drb_action identity changed: expected equality with p0_2 K2 rows")
    merged["action_cap"] = merged["cap_p0_2"]
    return merged.drop(columns=["base_covered", "cap_p0_2"])


def declared_horizon_end(origin: str) -> pd.Timestamp:
    """Declared calendar end for an origin label (data-independent)."""
    if "->" in origin:  # NYISO style 2018->2019 (delivery year 2019)
        delivery_year = int(origin.split("->")[1])
        return pd.Timestamp(f"{delivery_year}-12-31")
    if "Q" in origin:  # transfer style 2010Q4 / 2019Q1
        year = int(origin[:4])
        quarter = origin[4:]
        if quarter not in QUARTER_END:
            raise ValueError(f"unknown quarter label: {origin}")
        return pd.Timestamp(f"{year}{QUARTER_END[quarter]}")
    raise ValueError(f"cannot derive declared horizon end from origin: {origin}")


# --------------------------------------------------------------------------
# Metrics (mirror p0_2 / R43 definitions)
# --------------------------------------------------------------------------

def segment_windows(segments: np.ndarray, width: int) -> list[np.ndarray]:
    """All contiguous within-segment index windows of the given width."""
    windows: list[np.ndarray] = []
    start = 0
    n = len(segments)
    while start < n:
        end = start
        while end + 1 < n and segments[end + 1] == segments[start]:
            end += 1
        for w0 in range(start, end - width + 2):
            windows.append(np.arange(w0, w0 + width))
        start = end + 1
    return windows


def worst_window(service: np.ndarray, segments: np.ndarray, width: int) -> float:
    windows = segment_windows(segments, width)
    if not windows:
        return math.nan
    return float(min(float(np.mean(service[w])) for w in windows))


def policy_metrics(q: np.ndarray, actions: np.ndarray, segments: np.ndarray) -> dict[str, float]:
    service = np.where(actions, 1.0, q)
    shortfall = 1.0 - service
    action_total = int(actions.sum())
    n = len(q)
    rate = action_total / n if n else math.nan
    rescue = q < 1.0
    direct = int(np.sum(actions & rescue))
    row: dict[str, float] = {
        "n_days": n,
        "action_total": action_total,
        "action_rate": rate,
        "mean_service": float(np.mean(service)),
        "mean_shortfall": float(np.mean(shortfall)),
        "service_worst3": worst_window(service, segments, 3),
        "service_worst7": worst_window(service, segments, 7),
        "service_worst14": worst_window(service, segments, 14),
        "direct_rescue_count": direct,
        "rescue_precision": (direct / action_total) if action_total else math.nan,
        "rescued_shortfall_sum": float(np.sum(1.0 - q[actions])),
        "rescued_shortfall_per_action": ((float(np.sum(1.0 - q[actions])) / action_total) if action_total else math.nan),
        "violation_days": int(np.sum(service < SERVICE_DAY_THRESHOLD)),
    }
    for lam in LAMBDAS:
        row[f"L_lambda_{lam:g}"] = float(row["mean_shortfall"] + lam * rate)
    return row


def macrorow(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """Equal-weight origin macro over the numeric keys present everywhere."""
    keys = [k for k in rows[0] if all(k in r and isinstance(r[k], (int, float)) for r in rows)]
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([r[key] for r in rows], dtype=float)
        if np.isfinite(values).all():
            out[key] = float(np.mean(values))
    return out


def paired_origin_bootstrap(diffs: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict[str, float]:
    """Descriptive IID-over-origin bootstrap of the paired difference mean.

    Descriptive only for exposed replay: no Holm family, no equivalence claim.
    """
    rng = np.random.default_rng(seed)
    n = len(diffs)
    draws = np.asarray([np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(n_boot)])
    return {
        "effect": float(np.mean(diffs)),
        "ci_lo": float(np.quantile(draws, 0.025)),
        "ci_hi": float(np.quantile(draws, 0.975)),
        "share_positive": float(np.mean(draws > 0.0)),
        "n_origins": int(n),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def leave_one_out(diffs: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, label in enumerate(labels):
        keep = np.ones(len(diffs), dtype=bool)
        keep[i] = False
        out[str(label)] = float(np.mean(diffs[keep]))
    return out
