"""Segment-aware calendar utilities (gap resets, DST day identity)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def delivery_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalized local delivery dates (tz-naive midnights)."""
    if getattr(index, "tz", None) is not None:
        local = index.tz_convert("America/New_York")
        return pd.DatetimeIndex(local.normalize().tz_localize(None))
    return pd.DatetimeIndex(pd.DatetimeIndex(index).normalize())


def assign_segment_ids(dates: pd.Series | np.ndarray | pd.DatetimeIndex) -> np.ndarray:
    """Assign segment ids so calendar gaps > 1 day start a new segment."""
    d = pd.to_datetime(pd.Series(list(dates))).dt.normalize()
    if len(d) == 0:
        return np.zeros(0, dtype=int)
    if not d.is_monotonic_increasing:
        # Allow equal consecutive dates (multiple hours); sort-check non-decreasing.
        if (d.diff().dropna() < pd.Timedelta(0)).any():
            raise ValueError("dates must be non-decreasing for segment assignment")
    seg = np.zeros(len(d), dtype=int)
    cur = 0
    for i in range(1, len(d)):
        gap = (d.iloc[i] - d.iloc[i - 1]).days
        if gap > 1:
            cur += 1
        elif gap < 0:
            raise ValueError(
                f"non-increasing dates at i={i}: {d.iloc[i - 1]} -> {d.iloc[i]}"
            )
        seg[i] = cur
    return seg


def unique_day_segment_table(index: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per delivery day with segment id."""
    days = delivery_dates(index)
    # Preserve first-seen order of unique days.
    unique_days = pd.Index(pd.unique(days))
    segments = assign_segment_ids(unique_days)
    return pd.DataFrame({"delivery_date": unique_days, "segment_id": segments})


def row_segment_ids(index: pd.DatetimeIndex) -> np.ndarray:
    day_table = unique_day_segment_table(index)
    day_to_seg = {
        pd.Timestamp(row.delivery_date): int(row.segment_id)
        for row in day_table.itertuples(index=False)
    }
    days = delivery_dates(index)
    return np.asarray([day_to_seg[pd.Timestamp(d)] for d in days], dtype=int)


def utc_identity_key(ts: pd.Timestamp) -> str:
    """Unique UTC identity for a local delivery hour (DST-safe)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        # Interpret as America/New_York wall time then convert.
        t = t.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
    return t.tz_convert("UTC").isoformat()


def count_native_hours_per_day(index: pd.DatetimeIndex) -> pd.Series:
    days = delivery_dates(index)
    return pd.Series(1, index=days).groupby(level=0).sum()
