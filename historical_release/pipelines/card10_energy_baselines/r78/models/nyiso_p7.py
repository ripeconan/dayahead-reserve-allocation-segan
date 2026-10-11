"""M1: published NYISO P-7 eleven-zone day-ahead forecast (byte-for-byte)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..constants import ZONES
from .base import ModelAdapter, PointForecast, align_zones


class NyisoP7Adapter(ModelAdapter):
    model_name = "nyiso_p7_official"
    supports_quantiles = False

    def __init__(self, candidate: dict[str, Any] | None = None):
        candidate = candidate or {
            "candidate_id": "nyiso_p7_isolf_byte_identical",
            "source_product": "NYISO_MIS_P-7_ISO_Load_Forecast",
            "seed": 0,
            "trainable": False,
        }
        super().__init__(candidate)

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
            "train_rows": 0,
            "source_product": self.candidate.get(
                "source_product", "NYISO_MIS_P-7_ISO_Load_Forecast"
            ),
            "vintage_semantics": (
                "operator-published day-ahead P-7 isolf embedded in authorized "
                "nyiso_zonal_2017_2025 panel; no learned parameters"
            ),
        }

    def predict(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        predict_index: pd.DatetimeIndex,
    ) -> PointForecast:
        point = align_zones(forecast.reindex(predict_index))
        eligibility = pd.Series("eligible", index=predict_index, dtype=object)
        missing = point.isna().any(axis=1)
        eligibility.loc[missing] = "missing_p7_value"
        return PointForecast(
            point=point,
            eligibility=eligibility,
            meta={
                "source_product": self.fitted_state.get("source_product"),
                "vintage_semantics": self.fitted_state.get("vintage_semantics"),
                "model_hash": self.model_hash(),
                "config_hash": self.config_hash,
                "open_source_model": False,
                "operational_black_box": True,
            },
        )
