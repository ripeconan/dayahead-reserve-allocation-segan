"""Point-forecast metrics (stratum 1.1)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..constants import ZONES


def point_metrics(
    actual: pd.DataFrame,
    point: pd.DataFrame,
    *,
    upper_tail_threshold: float | None = None,
    development_abs_errors: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute MAE/RMSE/WAPE, peak, ramp, and upper-tail absolute error.

    upper_tail uses top 1% threshold frozen from development absolute errors when provided.
    """
    common = actual.index.intersection(point.index)
    a = actual.reindex(common)[list(ZONES)].to_numpy(float)
    p = point.reindex(common)[list(ZONES)].to_numpy(float)
    mask = np.isfinite(a) & np.isfinite(p)
    eligible = mask.all(axis=1)
    n_total = int(len(common))
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        return {
            "n_total": n_total,
            "n_eligible": 0,
            "n_missing": n_total,
            "mae": float("nan"),
            "rmse": float("nan"),
            "wape": float("nan"),
        }

    a_e = a[eligible]
    p_e = p[eligible]
    abs_err = np.abs(a_e - p_e)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((a_e - p_e) ** 2)))
    denom = float(np.sum(np.abs(a_e)))
    wape = float(np.sum(abs_err) / denom) if denom > 0 else float("nan")

    zone_mae = {f"mae_{zone}": float(np.mean(abs_err[:, i])) for i, zone in enumerate(ZONES)}
    worst_zone_mae = float(max(zone_mae.values())) if zone_mae else float("nan")

    # Daily peak magnitude MAE and peak-hour error (local day).
    idx = common[eligible]
    day_keys = (
        idx.tz_convert("America/New_York").normalize()
        if getattr(idx, "tz", None) is not None
        else pd.DatetimeIndex(idx).normalize()
    )
    peak_mag_err = []
    peak_hour_err = []
    for day in pd.unique(day_keys):
        sel = day_keys == day
        if sel.sum() < 1:
            continue
        a_day = a_e[sel]
        p_day = p_e[sel]
        # System peak = sum across zones per hour.
        a_sys = a_day.sum(axis=1)
        p_sys = p_day.sum(axis=1)
        a_peak_i = int(np.argmax(a_sys))
        p_peak_i = int(np.argmax(p_sys))
        peak_mag_err.append(abs(float(a_sys[a_peak_i] - p_sys[a_peak_i])))
        # Peak-hour error in hours (absolute).
        peak_hour_err.append(abs(a_peak_i - p_peak_i))
    peak_magnitude_mae = float(np.mean(peak_mag_err)) if peak_mag_err else float("nan")
    peak_hour_mae = float(np.mean(peak_hour_err)) if peak_hour_err else float("nan")

    # Hourly ramp MAE on system load.
    a_sys_all = a_e.sum(axis=1)
    p_sys_all = p_e.sum(axis=1)
    if len(a_sys_all) >= 2:
        ramp_mae = float(np.mean(np.abs(np.diff(a_sys_all) - np.diff(p_sys_all))))
    else:
        ramp_mae = float("nan")

    # Upper-tail: mean abs error among hours exceeding top-1% development threshold.
    flat_abs = abs_err.mean(axis=1)  # mean zone abs error per hour
    if development_abs_errors is not None and len(development_abs_errors):
        thr = float(np.quantile(development_abs_errors, 0.99))
    elif upper_tail_threshold is not None:
        thr = float(upper_tail_threshold)
    else:
        thr = float(np.quantile(flat_abs, 0.99))
    tail = flat_abs[flat_abs >= thr]
    upper_tail_mae = float(np.mean(tail)) if len(tail) else float("nan")

    out = {
        "n_total": n_total,
        "n_eligible": n_eligible,
        "n_missing": n_total - n_eligible,
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "worst_zone_mae": worst_zone_mae,
        "peak_magnitude_mae": peak_magnitude_mae,
        "peak_hour_mae": peak_hour_mae,
        "ramp_mae": ramp_mae,
        "upper_tail_threshold": thr,
        "upper_tail_mae": upper_tail_mae,
        "upper_tail_n": int(len(tail)),
    }
    out.update(zone_mae)
    return out
