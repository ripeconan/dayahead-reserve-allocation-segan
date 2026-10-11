"""R2 cutoff-safe point correction — v3 (post R03B review).

Declared contract (one variant, no menu):
  * realized-error lags 72 and 168 ELAPSED hours; ridge penalty 10; rolling
    365-day training window; newest training label strictly before local
    midnight of (t-1) i.e. latest completed label local date t-2;
  * per-row provenance for EVERY lagged feature used: source instant exactly
    target-72h / target-168h on the UTC timeline AND source local calendar age
    >= 2 / >= 6 days;
  * issuance never depends on the current/future target actual: a day-d
    forecast is issued whenever its pre-action inputs (official forecast,
    lagged errors, calendar features) exist.  Missing target actuals only
    delay the day's score.
  * daily grouping is by LOCAL delivery date: target rows are converted to
    America/New_York FIRST and then normalized (R03B item 1).  Each daily
    audit/fit slice contains exactly that local day's rows.

Returns (correction, audit) - two values; the audit frame is indexed by the
local delivery date and records per-day train/issued counts and the newest
training-label timestamp actually used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import r2_env  # noqa: F401  (thread limits before numeric imports)

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import r2_common as C  # noqa: E402


def build_features(actual: pd.DataFrame, forecast: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not actual.index.equals(forecast.index):
        raise ValueError("actual/forecast index mismatch")
    if not isinstance(actual.index, pd.DatetimeIndex) or actual.index.tz is None:
        raise ValueError("index must be a tz-aware DatetimeIndex")
    errors = actual - forecast
    index = errors.index
    local = index.tz_convert(C.TZ)
    features = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * local.hour.to_numpy() / 24),
        "hour_cos": np.cos(2 * np.pi * local.hour.to_numpy() / 24),
        "week_sin": np.sin(2 * np.pi * local.dayofweek.to_numpy() / 7),
        "week_cos": np.cos(2 * np.pi * local.dayofweek.to_numpy() / 7),
    }, index=index)
    for lag in C.LAG_HOURS:
        lagged = errors.shift(freq=pd.Timedelta(hours=lag)).reindex(index)
        lagged.columns = [f"error_{zone}_lag{lag}" for zone in C.ZONES]
        features = features.join(lagged)
    current = forecast.copy()
    current.columns = [f"forecast_{zone}" for zone in C.ZONES]
    features = features.join(current)
    return features, errors


def provenance_report(index: pd.DatetimeIndex) -> dict:
    report = {"rows": int(len(index)), "violations": [], "lags": {}}
    for lag in C.LAG_HOURS:
        sources = index - pd.Timedelta(hours=lag)
        ok = np.asarray([C.lag_provenance_ok(t, s, lag) for t, s in zip(index, sources)])
        report["lags"][lag] = {"checked": int(len(ok)), "n_violations": int((~ok).sum())}
        if not ok.all():
            first = int(np.flatnonzero(~ok)[0])
            report["violations"].append({"lag": lag, "target": str(index[first]),
                                         "source": str(sources[first])})
    return report


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                  penalty: float = C.RIDGE_PENALTY) -> np.ndarray:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    train = np.column_stack([np.ones(len(train)), train])
    test = np.column_stack([np.ones(len(test)), test])
    regularizer = np.eye(train.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(train.T @ train + regularizer, train.T @ y_train)
    return test @ coefficients


def forward_correction_calendar(features: pd.DataFrame, errors: pd.DataFrame,
                                start: str, end: str, *,
                                train_days: int = C.TRAIN_DAYS,
                                penalty: float = C.RIDGE_PENALTY,
                                min_train_rows: int = C.MIN_TRAIN_ROWS,
                                require_issue: bool = False
                                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Issue corrections for every local day in [start, end] with eligible inputs."""
    start_ts = pd.Timestamp(start, tz=C.TZ)
    end_ts = pd.Timestamp(f"{end} 23:59:59", tz=C.TZ)
    target_index = errors.index[(errors.index >= start_ts) & (errors.index <= end_ts)]
    target_local_dates = target_index.tz_convert(C.TZ).normalize()  # LOCAL first
    feature_cols = list(features.columns)
    correction_parts: list[pd.DataFrame] = []
    audit: list[dict] = []
    observed = np.isfinite(errors.to_numpy(float)).all(axis=1)
    complete = np.isfinite(features.to_numpy(float)).all(axis=1)
    for day, day_index in pd.Series(target_index, index=target_index).groupby(target_local_dates):
        rows = pd.DatetimeIndex(day_index.to_numpy())
        boundary = C.newest_label_boundary(day.tz_localize(None)).tz_localize(C.TZ)
        train_start = boundary - pd.DateOffset(days=train_days)
        in_window = (features.index >= train_start) & (features.index < boundary)
        train_index = features.index[in_window & observed & complete]
        issue_index = rows[np.isfinite(features.loc[rows, feature_cols].to_numpy(float)).all(axis=1)]
        newest_label = train_index.max() if len(train_index) else pd.NaT
        audit.append({
            "local_date": str(day.date()),
            "target_rows": int(len(rows)),
            "train_rows": int(len(train_index)),
            "issued_rows": int(len(issue_index)),
            "newest_train_label_utc": str(newest_label) if newest_label is not pd.NaT else None,
            "boundary_utc": str(boundary),
        })
        if len(train_index) < min_train_rows:
            if require_issue:
                raise RuntimeError(f"insufficient training rows on {day.date()}: {len(train_index)}")
            continue
        if not len(issue_index):
            if require_issue:
                raise RuntimeError(f"no issuable rows on {day.date()}")
            continue
        prediction = ridge_predict(features.loc[train_index, feature_cols].to_numpy(float),
                                   errors.loc[train_index].to_numpy(float),
                                   features.loc[issue_index, feature_cols].to_numpy(float),
                                   penalty)
        block = pd.DataFrame(prediction, index=issue_index, columns=list(C.ZONES))
        # each daily slice contains exactly that local day's target rows
        block_local = block.index.tz_convert(C.TZ).normalize()
        if not bool((block_local == day).all()):
            raise RuntimeError(f"daily slice spans multiple local dates on {day.date()}")
        correction_parts.append(block)
    correction = (pd.concat(correction_parts).sort_index()
                  if correction_parts else pd.DataFrame(columns=list(C.ZONES), dtype=float,
                                                        index=pd.DatetimeIndex([], tz=C.TZ)))
    audit_df = pd.DataFrame(audit).set_index("local_date") if audit else pd.DataFrame()
    if not audit_df.empty:
        violations = []
        for date_str, row in audit_df.iterrows():
            if row["newest_train_label_utc"] is None:
                continue
            if pd.Timestamp(row["newest_train_label_utc"]) >= pd.Timestamp(row["boundary_utc"]):
                violations.append(date_str)
        if violations:
            raise RuntimeError(f"training-label boundary violated on {violations[:3]}")
    return correction, audit_df


def issued_point_and_residual(forecast: pd.DataFrame, correction: pd.DataFrame,
                              actual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """issued_point_MW = official forecast + issued correction (only where issued);
    residual_MW = actual_load_MW - issued_point_MW (NaN where either is missing)."""
    aligned = correction.reindex(forecast.index)
    issued = forecast.add(aligned)
    issued = issued.where(aligned.notna())
    residual = actual - issued
    return issued, residual
