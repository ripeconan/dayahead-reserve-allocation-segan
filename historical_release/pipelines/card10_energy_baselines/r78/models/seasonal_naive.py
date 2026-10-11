"""M0: seasonal naive d7 (same zone and local delivery hour, 7 days earlier)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..constants import ZONES
from .base import ModelAdapter, PointForecast, empty_like


class SeasonalNaiveAdapter(ModelAdapter):
    model_name = "seasonal_naive_d7"
    supports_quantiles = False

    def __init__(self, candidate: dict[str, Any] | None = None):
        candidate = candidate or {
            "candidate_id": "seasonal_naive_d7_v1",
            "lag_days": 7,
            "seed": 0,
            "trainable": False,
        }
        super().__init__(candidate)
        self.lag_days = int(candidate.get("lag_days", 7))

    def fit(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        train_index: pd.DatetimeIndex,
        *,
        valid_index: pd.DatetimeIndex | None = None,
    ) -> None:
        self.fitted_state = {
            "trainable": False,
            "train_rows": int(len(train_index)),
            "lag_days": self.lag_days,
        }

    def predict(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        predict_index: pd.DatetimeIndex,
    ) -> PointForecast:
        # Same zone and local delivery hour from seven calendar days earlier.
        # DateOffset preserves local hour across DST; missing legal lag -> NA.
        point = empty_like(predict_index)
        eligibility = pd.Series("eligible", index=predict_index, dtype=object)
        actual_zones = actual.reindex(columns=list(ZONES))

        if len(predict_index) == 0:
            return PointForecast(point=point, eligibility=eligibility, meta={})

        # Shift by local calendar days, not fixed 168h. Nonexistent/ambiguous DST
        # hours become NaT and are recorded as missing legal lag (no silent drop).
        local = predict_index.tz_convert("America/New_York")
        naive = local.tz_localize(None)
        shifted_naive = naive - pd.DateOffset(days=self.lag_days)
        src_index = shifted_naive.tz_localize(
            "America/New_York", nonexistent="NaT", ambiguous="NaT"
        )
        valid_src = pd.notna(src_index)
        src_frame = pd.DataFrame(
            np.nan, index=predict_index, columns=list(ZONES), dtype=float
        )
        if valid_src.any():
            looked = actual_zones.reindex(src_index[valid_src])
            src_frame.loc[predict_index[valid_src]] = looked.to_numpy(dtype=float)
        point.loc[predict_index] = src_frame.to_numpy(dtype=float)

        missing_idx = src_frame.isna().any(axis=1)
        present = np.zeros(len(predict_index), dtype=bool)
        if valid_src.any():
            present[valid_src] = pd.Index(src_index[valid_src]).isin(actual_zones.index)
        eligibility.loc[predict_index[~present]] = "missing_legal_lag"
        present_but_bad = present & missing_idx.to_numpy()
        eligibility.loc[predict_index[present_but_bad]] = "missing_lag_nonfinite"

        n_missing = int((eligibility != "eligible").sum())
        return PointForecast(
            point=point,
            eligibility=eligibility,
            meta={
                "n_missing_legal_lag": n_missing,
                "lag_days": self.lag_days,
                "model_hash": self.model_hash(),
                "config_hash": self.config_hash,
            },
        )
