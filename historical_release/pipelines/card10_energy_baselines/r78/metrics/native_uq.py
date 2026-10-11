"""Native rectangular/quantile probabilistic metrics (stratum 1.2).

These metrics must not be ranked as if they were the common 11-d OAS ellipsoid.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..constants import ZONES


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    e = y - q
    return float(np.mean(np.maximum(alpha * e, (alpha - 1.0) * e)))


def winkler_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.1) -> float:
    """Mean Winkler score for central (1-alpha) intervals."""
    width = hi - lo
    below = y < lo
    above = y > hi
    score = width.copy()
    score[below] = width[below] + (2.0 / alpha) * (lo[below] - y[below])
    score[above] = width[above] + (2.0 / alpha) * (y[above] - hi[above])
    return float(np.mean(score))


def native_uq_metrics(
    actual: pd.DataFrame,
    q05: pd.DataFrame,
    q50: pd.DataFrame,
    q95: pd.DataFrame,
) -> dict[str, Any]:
    common = actual.index.intersection(q05.index).intersection(q50.index).intersection(q95.index)
    a = actual.reindex(common)[list(ZONES)].to_numpy(float)
    lo = q05.reindex(common)[list(ZONES)].to_numpy(float)
    mid = q50.reindex(common)[list(ZONES)].to_numpy(float)
    hi = q95.reindex(common)[list(ZONES)].to_numpy(float)
    mask = np.isfinite(a) & np.isfinite(lo) & np.isfinite(mid) & np.isfinite(hi)
    eligible = mask.all(axis=1)
    n_total = int(len(common))
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        return {
            "n_total": n_total,
            "n_eligible": 0,
            "estimand": "native_rectangular_quantile",
            "not_oas_ellipsoid": True,
        }

    a_e, lo_e, mid_e, hi_e = a[eligible], lo[eligible], mid[eligible], hi[eligible]
    # Ensure non-crossing for metric evaluation.
    stacked = np.stack([lo_e, mid_e, hi_e], axis=2)
    stacked.sort(axis=2)
    lo_e, mid_e, hi_e = stacked[:, :, 0], stacked[:, :, 1], stacked[:, :, 2]

    marginal_cover = float(np.mean((a_e >= lo_e) & (a_e <= hi_e)))
    simultaneous = float(np.mean(np.all((a_e >= lo_e) & (a_e <= hi_e), axis=1)))
    width = hi_e - lo_e
    mean_width = float(np.mean(width))
    # Rectangular log-volume = sum_z log(width_z); guard tiny widths.
    safe_w = np.maximum(width, 1e-12)
    rect_log_volume = float(np.mean(np.sum(np.log(safe_w), axis=1)))

    pinball_scores = {
        "pinball_q05": pinball(a_e, lo_e, 0.05),
        "pinball_q50": pinball(a_e, mid_e, 0.50),
        "pinball_q95": pinball(a_e, hi_e, 0.95),
    }
    mean_pinball = float(np.mean(list(pinball_scores.values())))
    winkler = winkler_score(a_e.ravel(), lo_e.ravel(), hi_e.ravel(), alpha=0.1)

    # CRPS from three quantiles is not uniquely defined; do not invent.
    return {
        "n_total": n_total,
        "n_eligible": n_eligible,
        "n_missing": n_total - n_eligible,
        "estimand": "native_rectangular_quantile",
        "not_oas_ellipsoid": True,
        "ranking_namespace": "native_probabilistic_stratum",
        "marginal_coverage": marginal_cover,
        "simultaneous_11zone_coverage": simultaneous,
        "mean_interval_width": mean_width,
        "rectangular_log_volume": rect_log_volume,
        "mean_pinball": mean_pinball,
        "winkler_90": winkler,
        "crps": None,
        "crps_status": "NOT_RECOVERABLE_FROM_THREE_QUANTILES_ONLY",
        **pinball_scores,
    }
