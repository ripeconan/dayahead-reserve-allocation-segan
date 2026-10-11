"""Common OAS-region and verified SAOCP evaluation pipeline."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .geometry import fit_ellipsoid, log_volume_proxy, squared_distance
from .reliability_shield import StrictLag2ReliabilityShield, rolling_worst_service
from .saocp_stream import BACKEND_KIND, prefix_sha256, run_verified_stream

LOCAL_TIMEZONE = "America/New_York"
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
ORIGIN_LABELS = tuple(f"{year}->{year + 1}" for year in range(2018, 2025))
BASELINE_MODEL = "nyiso_p7_official"
TARGET_COVERAGE = 0.90
FEEDBACK_LAG_DAYS = 2
SAOCP_COMMIT = "95790a9b9e4061166e6d7a8a13282e019b1a8663"
SAOCP_LIFETIME = 8


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def parse_local_timestamp(values: pd.Series) -> pd.DatetimeIndex:
    """Parse offset-bearing timestamps in the NYISO delivery timezone."""
    try:
        parsed = pd.to_datetime(values, utc=True, errors="raise", format="mixed")
    except TypeError:  # pandas before 2.0
        parsed = pd.to_datetime(values, utc=True, errors="raise")
    local = pd.DatetimeIndex(parsed).tz_convert(LOCAL_TIMEZONE)
    if local.hasnans:
        raise ValueError("timestamp parser produced NaT")
    return local


def load_predictions(path: Path, models: Sequence[str]) -> pd.DataFrame:
    """Load the long-form point-prediction artifact and verify its key contract."""
    required = {
        "model",
        "candidate_id",
        "origin",
        "timestamp",
        "delivery_date",
        "zone",
        "actual",
        "point",
        "eligibility",
        "model_hash",
        "config_hash",
    }
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"point predictions missing columns: {missing}")
    requested = set(models) | {BASELINE_MODEL}
    frame = frame[frame["model"].isin(requested)].copy()
    if frame.empty or not set(models).issubset(set(frame["model"].unique())):
        raise ValueError("one or more requested models have no predictions")

    frame["ts_local"] = parse_local_timestamp(frame["timestamp"])
    stored = pd.to_datetime(frame["delivery_date"], format="%Y-%m-%d", errors="raise")
    local = frame["ts_local"].dt
    mismatch = (
        (stored.dt.year.to_numpy() != local.year.to_numpy())
        | (stored.dt.month.to_numpy() != local.month.to_numpy())
        | (stored.dt.day.to_numpy() != local.day.to_numpy())
    )
    if bool(mismatch.any()):
        raise ValueError("delivery_date differs from the New York local date")
    if max(timestamp.year for timestamp in frame["ts_local"]) >= 2026:
        raise ValueError("reserved 2026 outcomes are outside this reproduction package")
    if not set(frame["origin"].astype(str).unique()).issubset(set(ORIGIN_LABELS)):
        raise ValueError("unexpected origin label")
    if not set(frame["zone"].unique()).issubset(set(ZONES)):
        raise ValueError("unknown NYISO zone")
    if frame.duplicated(["model", "origin", "ts_local", "zone"]).any():
        raise ValueError("duplicate model/origin/timestamp/zone rows")
    return frame


def residual_wide(frame: pd.DataFrame, model: str, origin: str) -> pd.DataFrame:
    part = frame[(frame["model"] == model) & (frame["origin"] == origin)].copy()
    part = part[part["eligibility"].astype(str) == "eligible"]
    finite = np.isfinite(part["actual"].to_numpy(float)) & np.isfinite(
        part["point"].to_numpy(float)
    )
    part = part[finite]
    if part.empty:
        return pd.DataFrame(columns=ZONES)
    actual = part.pivot(index="ts_local", columns="zone", values="actual")
    point = part.pivot(index="ts_local", columns="zone", values="point")
    common = actual.index.intersection(point.index)
    residual = actual.reindex(common).reindex(columns=ZONES) - point.reindex(common).reindex(columns=ZONES)
    residual = residual.replace([np.inf, -np.inf], np.nan).dropna(how="any").sort_index()
    residual.index = pd.DatetimeIndex(residual.index)
    if residual.index.tz is None or str(residual.index.tz) != LOCAL_TIMEZONE:
        raise ValueError("residual index lost the delivery timezone")
    return residual


def day_partition(index: pd.DatetimeIndex) -> tuple[list[np.ndarray], list[str]]:
    """Partition chronological rows by local delivery date, including DST days."""
    keys = index.normalize().asi8
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    order = np.argsort(first)
    inverse_order = np.empty(len(order), dtype=np.intp)
    inverse_order[order] = np.arange(len(order))
    inverse = inverse_order[inverse]
    positions = [np.flatnonzero(inverse == day) for day in range(len(order))]
    labels = [index[first[order[day]]].strftime("%Y-%m-%d") for day in range(len(order))]
    if len(index) and not np.array_equal(np.sort(np.concatenate(positions)), np.arange(len(index))):
        raise RuntimeError("day partition omitted or duplicated rows")
    return positions, labels


def assign_segment_ids(dates: Sequence[str]) -> np.ndarray:
    parsed = pd.to_datetime(pd.Series(dates)).dt.normalize()
    if not parsed.is_monotonic_increasing:
        raise ValueError("delivery dates must be sorted")
    segments = np.zeros(len(parsed), dtype=int)
    current = 0
    for index in range(1, len(parsed)):
        gap = (parsed.iloc[index] - parsed.iloc[index - 1]).days
        if gap > 1:
            current += 1
        elif gap < 1:
            raise ValueError("delivery dates must be unique and increasing")
        segments[index] = current
    return segments


def row_segment_ids(
    day_positions: Sequence[np.ndarray], segments: Sequence[int], n_rows: int
) -> np.ndarray:
    if len(day_positions) != len(segments):
        raise ValueError("day_positions and segments have different lengths")
    output = np.full(n_rows, -1, dtype=np.int64)
    for positions, segment in zip(day_positions, segments):
        output[np.asarray(positions, dtype=np.int64)] = int(segment)
    if n_rows and np.any(output < 0):
        raise ValueError("unassigned row segment")
    return output


def _unit_ball_log_volume(dimension: int) -> float:
    return 0.5 * dimension * math.log(math.pi) - math.lgamma(0.5 * dimension + 1.0)


def segment_worst(values: Sequence[float], segments: Sequence[int], window: int) -> float:
    try:
        return rolling_worst_service(list(values), list(segments), window)
    except ValueError:
        return float(np.mean(values))


def _verified_hour(
    repo: Path,
    prefix: tuple[float, ...],
    scores: tuple[float, ...],
    segments: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    result = run_verified_stream(
        repo=repo,
        calibration_prefix=prefix,
        scores=scores,
        segment_ids=segments,
        feedback_lag=FEEDBACK_LAG_DAYS,
    )
    thresholds = np.asarray(result["pre_action"], dtype=float)
    if thresholds.shape != (len(scores),) or not np.isfinite(thresholds).all():
        raise RuntimeError("SAOCP returned incomplete thresholds")
    identity = dict(result["identity"])
    expected = {
        "backend_kind": BACKEND_KIND,
        "production_git_identity_verified": True,
        "public_oracle_parity": False,
        "official_commit": SAOCP_COMMIT,
        "lifetime": SAOCP_LIFETIME,
        "feedback_lag": FEEDBACK_LAG_DAYS,
        "calibration_prefix_sha256": prefix_sha256(prefix),
        "calibration_prefix_count": len(prefix),
    }
    mismatch = {key: (identity.get(key), value) for key, value in expected.items() if identity.get(key) != value}
    if mismatch:
        raise RuntimeError(f"SAOCP identity mismatch: {mismatch}")
    prefix_record = {
        "prefix_count": len(prefix),
        "prefix_byte_encoding": "numpy-native-float64-tobytes-no-byteswap",
        "prefix_sha256": prefix_sha256(prefix),
        "prefix_min": float(min(prefix)),
        "prefix_max": float(max(prefix)),
        "prefix_all_finite": True,
        "threshold_pre_hash": str(result["pre_hash"]),
    }
    return thresholds, identity, prefix_record


def evaluate_model_origin(
    frame: pd.DataFrame,
    model: str,
    origin: str,
    saocp_repo: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate one model/origin cell and return hourly trace, identity, metrics."""
    reference_year, test_year = (int(value) for value in origin.split("->"))
    reference_origin = f"{reference_year - 1}->{reference_year}"
    reference = residual_wide(frame, model, reference_origin)
    test = residual_wide(frame, model, origin)
    p7_reference = residual_wide(frame, BASELINE_MODEL, reference_origin)
    if min(len(reference), len(test), len(p7_reference)) < 100:
        status = {
            "model": model,
            "origin": origin,
            "status": "BLOCKED_NO_LEGAL_PRIOR_ORIGIN_REFERENCE_PREDICTIONS",
            "reference_origin": reference_origin,
            "n_reference": int(len(reference)),
            "n_test": int(len(test)),
            "n_p7_reference": int(len(p7_reference)),
        }
        return pd.DataFrame(), pd.DataFrame(), status

    ellipsoid = fit_ellipsoid(reference.to_numpy(float), TARGET_COVERAGE)
    p7_ellipsoid = fit_ellipsoid(p7_reference.to_numpy(float), TARGET_COVERAGE)
    sign, log_determinant = np.linalg.slogdet(ellipsoid.covariance)
    if sign <= 0:
        raise ValueError("OAS covariance has a nonpositive determinant")
    log_covariance = 0.5 * float(log_determinant)
    reference_log_volume = log_volume_proxy(p7_ellipsoid)
    reference_scores = squared_distance(reference.to_numpy(float), ellipsoid.center, ellipsoid.precision)
    test_scores = squared_distance(test.to_numpy(float), ellipsoid.center, ellipsoid.precision)

    day_positions, day_labels = day_partition(test.index)
    day_segments = assign_segment_ids(day_labels)
    row_segments = row_segment_ids(day_positions, day_segments, len(test))
    reference_hours = reference.index.hour.to_numpy()
    test_hours = test.index.hour.to_numpy()
    thresholds = np.full(len(test), np.nan, dtype=float)
    prefix_hash_by_hour: dict[int, str] = {}
    identities: list[dict[str, Any]] = []
    for hour in range(24):
        positions = np.flatnonzero(test_hours == hour)
        prefix = tuple(float(value) for value in reference_scores[reference_hours == hour])
        if not prefix or len(positions) == 0:
            raise ValueError(f"missing local-hour data for hour {hour}")
        hour_thresholds, identity, record = _verified_hour(
            saocp_repo,
            prefix,
            tuple(float(test_scores[position]) for position in positions),
            tuple(int(row_segments[position]) for position in positions),
        )
        thresholds[positions] = hour_thresholds
        prefix_hash_by_hour[hour] = record["prefix_sha256"]
        identities.append(
            {
                "model": model,
                "origin": origin,
                "reference_origin": reference_origin,
                "reference_year": reference_year,
                "evaluation_year": test_year,
                "local_delivery_hour": hour,
                "n_reference_scores": len(prefix),
                "n_evaluation_scores": len(positions),
                **record,
                "backend_kind": identity["backend_kind"],
                "production_git_identity_verified": identity["production_git_identity_verified"],
                "public_oracle_parity": identity["public_oracle_parity"],
                "official_commit": identity["official_commit"],
                "official_tree": identity["official_tree"],
                "lifetime": identity["lifetime"],
                "feedback_lag": identity["feedback_lag"],
            }
        )
    if not np.isfinite(thresholds).all():
        raise RuntimeError("SAOCP left non-finite thresholds")

    dimension = len(ZONES)
    directions = np.vstack([np.eye(dimension), np.ones(dimension) / dimension])
    direction_names = list(ZONES) + ["AGGREGATE_EQ"]
    raw_log_volume = _unit_ball_log_volume(dimension) + log_covariance + 0.5 * dimension * np.log(
        np.maximum(thresholds, 1e-12)
    )
    normalized_volume = np.exp(
        log_covariance
        + 0.5 * dimension * np.log(np.maximum(thresholds, 1e-12))
        - reference_log_volume
    )
    covariance_hash = sha256_array(ellipsoid.covariance)
    center_hash = sha256_array(ellipsoid.center)
    candidate_id = str(frame.loc[frame["model"] == model, "candidate_id"].iloc[0])
    rows: list[dict[str, Any]] = []
    for position, timestamp in enumerate(test.index):
        threshold = float(thresholds[position])
        residual = test.iloc[position].to_numpy(float)
        radii = np.sqrt(np.maximum(threshold * np.diag(ellipsoid.covariance), 0.0))
        directional_radii = np.sqrt(
            np.maximum(threshold * np.einsum("ij,jk,ik->i", directions, ellipsoid.covariance, directions), 0.0)
        )
        row: dict[str, Any] = {
            "model": model,
            "candidate_id": candidate_id,
            "origin": origin,
            "reference_origin": reference_origin,
            "timestamp": timestamp.isoformat(),
            "delivery_date": timestamp.date().isoformat(),
            "local_delivery_hour": int(timestamp.hour),
            "segment_id": int(row_segments[position]),
            "score": float(test_scores[position]),
            "threshold": threshold,
            "covered": float(test_scores[position] <= threshold),
            "volume": float(normalized_volume[position]),
            "log_volume": float(raw_log_volume[position]),
            "volume_reference": "same-origin P7 reference OAS conformal ellipsoid",
            "oas_covariance_sha256": covariance_hash,
            "oas_center_sha256": center_hash,
            "saocp_prefix_sha256": prefix_hash_by_hour[int(timestamp.hour)],
            "saocp_commit": SAOCP_COMMIT,
            "feedback_lag_days": FEEDBACK_LAG_DAYS,
            "layer": "base_no_fallback",
        }
        for zone, covered in zip(ZONES, np.abs(residual - ellipsoid.center) <= radii):
            row[f"covered_{zone}"] = float(covered)
        centers = directions @ ellipsoid.center
        actuals = directions @ residual
        for name, reserve, actual in zip(direction_names, centers + directional_radii, actuals):
            row[f"reserve_{name}"] = float(reserve)
            row[f"actual_{name}"] = float(actual)
        rows.append(row)

    detail = pd.DataFrame(rows)
    daily = detail.groupby(["delivery_date", "segment_id"], sort=True).agg(
        daily_coverage=("covered", "mean"),
        daily_volume=("volume", "mean"),
        daily_log_volume=("log_volume", "mean"),
    ).reset_index()
    coverage = daily["daily_coverage"].to_numpy(float)
    segments = daily["segment_id"].to_numpy(int)
    status = {
        "model": model,
        "origin": origin,
        "status": "PASS",
        "reference_origin": reference_origin,
        "n_reference": int(len(reference)),
        "n_test": int(len(test)),
        "n_days": int(len(daily)),
        "joint_coverage": float(detail["covered"].mean()),
        "service_coverage": float(coverage.mean()),
        "service_worst3": segment_worst(coverage, segments, 3),
        "service_worst7": segment_worst(coverage, segments, 7),
        "service_worst14": segment_worst(coverage, segments, 14),
        "mean_volume": float(detail["volume"].mean()),
        "median_volume": float(detail["volume"].median()),
        "p95_volume": float(detail["volume"].quantile(0.95)),
        "mean_log_volume": float(detail["log_volume"].mean()),
        "target_coverage": TARGET_COVERAGE,
        "lifetime": SAOCP_LIFETIME,
        "feedback_lag_days": FEEDBACK_LAG_DAYS,
        "time_semantics": "America/New_York local delivery hour/day",
    }
    return detail, pd.DataFrame(identities), status


def policy_outputs(
    detail: pd.DataFrame,
    model: str,
    origin: str,
    include_controller: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate daily base service and optionally replay the lag-2 controller."""
    daily = detail.groupby(["delivery_date", "segment_id"], sort=True)["covered"].mean().rename(
        "base_daily_coverage"
    ).reset_index()
    traces: list[dict[str, Any]] = []
    for row in daily.itertuples(index=False):
        traces.append(
            {
                "model": model,
                "origin": origin,
                "delivery_date": row.delivery_date,
                "segment_id": int(row.segment_id),
                "policy": "no_fallback",
                "base_daily_coverage": float(row.base_daily_coverage),
                "served": float(row.base_daily_coverage),
                "action": False,
                "action_used_before": 0,
                "action_used_after": 0,
                "eligible_shortfall_sum": 0.0,
            }
        )
    if include_controller:
        controller = StrictLag2ReliabilityShield(len(daily))
        outcomes = controller.replay(
            daily["base_daily_coverage"].to_list(), daily["segment_id"].to_list()
        )
        for source, outcome in zip(daily.itertuples(index=False), outcomes):
            traces.append(
                {
                    "model": model,
                    "origin": origin,
                    "delivery_date": source.delivery_date,
                    "segment_id": int(source.segment_id),
                    "policy": "drb_reactive_lag2",
                    "base_daily_coverage": float(source.base_daily_coverage),
                    "served": outcome.service,
                    "action": outcome.action,
                    "action_used_before": outcome.actions_used - int(outcome.action),
                    "action_used_after": outcome.actions_used,
                    "eligible_shortfall_sum": outcome.eligible_shortfall_sum,
                }
            )
    trace = pd.DataFrame(traces)
    metrics: list[dict[str, Any]] = []
    for policy, rows in trace.groupby("policy", sort=True):
        service = rows["served"].to_numpy(float)
        segments = rows["segment_id"].to_numpy(int)
        metrics.append(
            {
                "model": model,
                "origin": origin,
                "policy": policy,
                "n_days": len(rows),
                "mean_service": float(service.mean()),
                "service_worst3": segment_worst(service, segments, 3),
                "service_worst7": segment_worst(service, segments, 7),
                "service_worst14": segment_worst(service, segments, 14),
                "n_actions": int(rows["action"].sum()),
                "action_rate": float(rows["action"].mean()),
            }
        )
    return pd.DataFrame(metrics), trace

