"""Prediction long-table builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import ZONES
from ..models.base import PointForecast
from ..segments import delivery_dates


def predictions_to_long(
    *,
    model: str,
    candidate_id: str,
    origin: str,
    stratum: str,
    actual: pd.DataFrame,
    result: PointForecast,
    model_hash: str | None = None,
    config_hash: str | None = None,
) -> pd.DataFrame:
    idx = result.point.index.intersection(actual.index)
    if len(idx) == 0:
        return pd.DataFrame()

    days = delivery_dates(idx)
    n = len(idx)
    z = len(ZONES)
    # Tile metadata across zones.
    timestamps = np.repeat(idx.astype(str).to_numpy(), z)
    delivery = np.repeat(pd.DatetimeIndex(days).strftime("%Y-%m-%d").to_numpy(), z)
    zones = np.tile(np.asarray(list(ZONES), dtype=object), n)
    actual_v = actual.reindex(idx)[list(ZONES)].to_numpy(float).reshape(-1)
    point_v = result.point.reindex(idx)[list(ZONES)].to_numpy(float).reshape(-1)
    if result.eligibility is not None:
        elig = np.repeat(result.eligibility.reindex(idx).astype(str).to_numpy(), z)
    else:
        elig = np.repeat(np.asarray(["eligible"] * n, dtype=object), z)

    data = {
        "model": model,
        "candidate_id": candidate_id,
        "origin": origin,
        "stratum": stratum,
        "timestamp": timestamps,
        "delivery_date": delivery,
        "zone": zones,
        "actual": actual_v,
        "point": point_v,
        "eligibility": elig,
        "model_hash": model_hash,
        "config_hash": config_hash,
        "checkpoint_hash": result.meta.get("checkpoint_hash"),
    }
    if result.q05 is not None:
        data["q05"] = result.q05.reindex(idx)[list(ZONES)].to_numpy(float).reshape(-1)
    if result.q50 is not None:
        data["q50"] = result.q50.reindex(idx)[list(ZONES)].to_numpy(float).reshape(-1)
    if result.q95 is not None:
        data["q95"] = result.q95.reindex(idx)[list(ZONES)].to_numpy(float).reshape(-1)
    return pd.DataFrame(data)
