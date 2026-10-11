"""Lag-2 feature builder with explicit reveal timestamps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import ERROR_LAGS_HOURS, FEEDBACK_LAG_DAYS, MIN_OUTCOME_LAG_HOURS, ZONES
from .firewall import FirewallViolation


@dataclass(frozen=True)
class FeatureBundle:
    features: pd.DataFrame
    errors: pd.DataFrame
    feature_source_max_reveal: pd.Series
    information_cutoff: pd.Series
    feature_names: tuple[str, ...]


def _calendar_block(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour.to_numpy()
    weekday = index.dayofweek.to_numpy()
    month = index.month.to_numpy()
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * weekday / 7.0),
            "dow_cos": np.cos(2 * np.pi * weekday / 7.0),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
            "is_weekend": (weekday >= 5).astype(float),
        },
        index=index,
    )


def _holiday_flag(index: pd.DatetimeIndex) -> pd.Series:
    """US federal holidays via pinned holidays package when available."""
    try:
        import holidays as holidays_lib
    except Exception:
        return pd.Series(0.0, index=index, name="is_holiday_us_ny")
    # Explicit per-timestamp date() calls: DatetimeIndex.date can surface
    # method-descriptor edge cases under some pandas/holidays combinations.
    idx = pd.DatetimeIndex(index)
    if getattr(idx, "tz", None) is not None:
        local_ts = idx.tz_convert("America/New_York")
    else:
        local_ts = idx
    year_vals = sorted({int(ts.year) for ts in local_ts})
    cal = holidays_lib.country_holidays("US", years=year_vals)
    local_dates = [ts.date() for ts in local_ts]
    flags = np.asarray([1.0 if d in cal else 0.0 for d in local_dates], dtype=float)
    return pd.Series(flags, index=index, name="is_holiday_us_ny")


def build_primary_features(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    include_load_lag48: bool = True,
) -> FeatureBundle:
    """Build primary no-weather features with legal lag-2 reveal only."""
    if not actual.index.equals(forecast.index):
        raise ValueError("actual/forecast index mismatch")
    if list(actual.columns) != list(ZONES) or list(forecast.columns) != list(ZONES):
        raise ValueError("zone columns must match frozen ZONES order")

    errors = actual - forecast
    index = errors.index
    features = _calendar_block(index)
    features = features.join(_holiday_flag(index))

    # Current P-7 forecast is known at freeze time for day t.
    current = forecast.copy()
    current.columns = [f"forecast_{zone}" for zone in ZONES]
    features = features.join(current)

    reveal_parts: list[pd.Series] = []
    # Calendar and forecast are known by the day-t freeze (modeled as start of day t local).
    # Realized error/load lags use explicit source timestamps.
    for lag in ERROR_LAGS_HOURS:
        if lag < MIN_OUTCOME_LAG_HOURS:
            raise FirewallViolation(f"illegal outcome lag hours={lag}")
        lagged_err = errors.shift(freq=pd.Timedelta(hours=lag)).reindex(index)
        lagged_err.columns = [f"error_{zone}_lag{lag}" for zone in ZONES]
        features = features.join(lagged_err)
        # Source reveal time for lag L at timestamp t is t - L hours.
        source_time = index - pd.Timedelta(hours=lag)
        reveal_parts.append(pd.Series(source_time, index=index))

    if include_load_lag48:
        lagged_load = actual.shift(freq=pd.Timedelta(hours=48)).reindex(index)
        lagged_load.columns = [f"load_{zone}_lag48" for zone in ZONES]
        features = features.join(lagged_load)
        reveal_parts.append(pd.Series(index - pd.Timedelta(hours=48), index=index))

    forbidden = [c for c in features.columns if "lag24" in str(c) or "lag1" in str(c)]
    # Allow lag168 which contains "lag1" as substring? "lag168" contains "lag1" yes!
    # Be precise: ban lag24 and lag1 as whole lag tokens, not lag168.
    forbidden = [
        c
        for c in features.columns
        if str(c).endswith("_lag24")
        or str(c).endswith("_lag1")
        or "_lag24_" in str(c)
    ]
    if forbidden:
        raise FirewallViolation(f"illegal lag-1/24 feature present: {forbidden}")

    # Information cutoff for day-t prediction: start of local delivery day t
    # The P-7 freeze is modeled at the beginning of the delivery day;
    # realized outcomes remain lag-2 constrained via feature_source_max_reveal).
    if getattr(index, "tz", None) is not None:
        local = index.tz_convert("America/New_York")
        info_cutoff = local.normalize()
    else:
        info_cutoff = pd.DatetimeIndex(index).normalize()
    information_cutoff = pd.Series(info_cutoff, index=index, name="information_cutoff")

    if reveal_parts:
        # Max reveal among realized sources; calendar/forecast not in this max.
        stacked = pd.concat(reveal_parts, axis=1)
        feature_source_max_reveal = stacked.max(axis=1)
    else:
        feature_source_max_reveal = pd.Series(pd.NaT, index=index)

    # Validity: finite features/errors, plus hard lag-2 reveal constraints.
    # Near fall-back, absolute 48h can land on calendar day t-1; those rows are
    # ineligible (dropped), not silently accepted.
    valid = features.notna().all(axis=1) & errors.notna().all(axis=1)
    if len(feature_source_max_reveal):
        idx = features.index
        src = pd.DatetimeIndex(feature_source_max_reveal)
        # Timedelta arithmetic is unit-safe across pandas ns/us resolutions.
        age_hours = ((idx - src) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
        hour_ok = age_hours >= float(MIN_OUTCOME_LAG_HOURS) - 1e-9
        if getattr(idx, "tz", None) is not None:
            deliv_dates = idx.tz_convert("America/New_York").date
            src_dates = src.tz_convert("America/New_York").date
        else:
            deliv_dates = pd.DatetimeIndex(idx).date
            src_dates = pd.DatetimeIndex(src).date
        age_days = np.asarray(
            [(d - s).days for d, s in zip(deliv_dates, src_dates)], dtype=int
        )
        day_ok = age_days >= FEEDBACK_LAG_DAYS
        valid = valid & pd.Series(hour_ok & day_ok, index=idx)

    features = features.loc[valid]
    errors = errors.loc[valid]
    information_cutoff = information_cutoff.loc[valid]
    feature_source_max_reveal = feature_source_max_reveal.loc[valid]

    return FeatureBundle(
        features=features,
        errors=errors,
        feature_source_max_reveal=feature_source_max_reveal,
        information_cutoff=information_cutoff,
        feature_names=tuple(features.columns),
    )


def assert_t_minus_1_invariance(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    target_timestamps: pd.DatetimeIndex,
) -> None:
    """Perturb day t-1 outcomes; day-t features must be byte-identical."""
    if len(target_timestamps) == 0:
        return
    base = build_primary_features(actual, forecast)
    # Choose a target day with a previous calendar day in sample.
    t = pd.Timestamp(target_timestamps[len(target_timestamps) // 2])
    if getattr(t, "tzinfo", None) is not None:
        t_day = t.tz_convert("America/New_York").normalize()
        t_prev = t_day - pd.DateOffset(days=1)
        mask_prev = actual.index.tz_convert("America/New_York").normalize() == t_prev
        mask_t = actual.index.tz_convert("America/New_York").normalize() == t_day
    else:
        t_day = t.normalize()
        t_prev = t_day - pd.DateOffset(days=1)
        mask_prev = actual.index.normalize() == t_prev
        mask_t = actual.index.normalize() == t_day
    if not mask_prev.any() or not mask_t.any():
        return
    actual_pert = actual.copy()
    actual_pert.loc[mask_prev] = actual_pert.loc[mask_prev] + 1000.0
    pert = build_primary_features(actual_pert, forecast)
    t_index = actual.index[mask_t].intersection(base.features.index).intersection(
        pert.features.index
    )
    if len(t_index) == 0:
        return
    left = base.features.loc[t_index]
    right = pert.features.loc[t_index]
    if not np.allclose(left.to_numpy(), right.to_numpy(), equal_nan=True):
        raise AssertionError("day t-1 outcome perturbation changed day-t features")
