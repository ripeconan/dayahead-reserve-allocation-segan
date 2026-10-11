"""M2: admitted lag-2 rolling ridge residual correction on top of P-7.

Implements the frozen lag-2 information set and ridge parameters from
``scripts/card10_confirmatory/lag2_pipeline.py`` without importing the heavy
killtest dependency graph (opsd_* modules).  Parameters and feature construction
are byte-compatible with the admitted pipeline for the residual-correction path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..constants import ZONES
from .base import ModelAdapter, PointForecast, align_zones, empty_like

# Admitted constants from lag2_pipeline.py
FEEDBACK_LAG_DAYS = 2
ERROR_LAGS_HOURS = (48, 168)
RIDGE_PENALTY = 10.0
TRAIN_DAYS = 365
SOURCE_MODULE = "scripts/card10_confirmatory/lag2_pipeline.py"


def ridge_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, penalty: float
) -> np.ndarray:
    """Fit the fixed ridge predictor used by the lag-2 baseline."""
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


def build_lag2_features(
    actual: pd.DataFrame, forecast: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Admitted lag-2 feature surface (no lag-24)."""
    if not actual.index.equals(forecast.index):
        raise ValueError("actual/forecast index mismatch")
    errors = actual - forecast
    index = errors.index
    hour = index.hour.to_numpy()
    weekday = index.dayofweek.to_numpy()
    features = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "week_sin": np.sin(2 * np.pi * weekday / 7),
            "week_cos": np.cos(2 * np.pi * weekday / 7),
        },
        index=index,
    )
    for lag in ERROR_LAGS_HOURS:
        lagged = errors.shift(freq=pd.Timedelta(hours=lag)).reindex(index)
        lagged.columns = [f"error_{zone}_lag{lag}" for zone in ZONES]
        features = features.join(lagged)
    current = forecast.copy()
    current.columns = [f"forecast_{zone}" for zone in ZONES]
    features = features.join(current)
    valid = features.notna().all(axis=1) & errors.notna().all(axis=1)
    result = features.loc[valid]
    forbidden = [column for column in result if str(column).endswith("_lag24")]
    if forbidden:
        raise RuntimeError(f"lag-1 feature reached lag-2 surface: {forbidden}")
    return result, errors


def forward_correction_lag2(
    features: pd.DataFrame,
    errors: pd.DataFrame,
    start: str,
    end: str,
    *,
    train_days: int = TRAIN_DAYS,
    penalty: float = RIDGE_PENALTY,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Admitted day-wise lag-2 ridge residual correction."""
    target = errors.loc[start : f"{end} 23:59:59"]
    predicted_parts: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for day, current in target.groupby(target.index.normalize(), sort=True):
        training_end_exclusive = day - pd.DateOffset(days=1)
        training_start = training_end_exclusive - pd.DateOffset(days=train_days)
        train_index = features.index[
            (features.index >= training_start) & (features.index < training_end_exclusive)
        ]
        current_index = current.index.intersection(features.index)
        if len(train_index) < 1000 or not len(current_index):
            continue
        newest = train_index.max()
        calendar_age = int((day.date() - newest.date()).days)
        if calendar_age < FEEDBACK_LAG_DAYS:
            raise RuntimeError("training surface includes incomplete t-1 outcome")
        prediction = ridge_predict(
            features.loc[train_index].to_numpy(),
            errors.loc[train_index].to_numpy(),
            features.loc[current_index].to_numpy(),
            penalty,
        )
        predicted_parts.append(
            pd.DataFrame(prediction, index=current_index, columns=list(ZONES))
        )
        audit.append(
            {
                "delivery_date": day.date().isoformat(),
                "training_start": train_index.min().isoformat(),
                "training_end_inclusive": newest.isoformat(),
                "current_rows": len(current_index),
                "training_rows": len(train_index),
                "newest_realized_error_age_days": calendar_age,
            }
        )
    if not predicted_parts:
        raise RuntimeError("lag-2 correction produced no rows")
    predicted = pd.concat(predicted_parts).sort_index()
    corrected = errors.loc[predicted.index] - predicted
    if corrected.index.duplicated().any() or not corrected.index.is_monotonic_increasing:
        raise RuntimeError("corrected index invalid")
    if not np.isfinite(corrected.to_numpy()).all():
        raise RuntimeError("corrected residual is non-finite")
    return predicted, corrected, audit


class RidgeLag2Adapter(ModelAdapter):
    model_name = "ridge_lag2"
    supports_quantiles = False

    def __init__(self, candidate: dict[str, Any] | None = None):
        candidate = candidate or {
            "candidate_id": "ridge_lag2_admitted_pipeline",
            "penalty": RIDGE_PENALTY,
            "train_days": TRAIN_DAYS,
            "error_lags_hours": list(ERROR_LAGS_HOURS),
            "seed": 0,
            "source_module": SOURCE_MODULE,
        }
        super().__init__(candidate)
        self.penalty = float(candidate.get("penalty", RIDGE_PENALTY))
        self.train_days = int(candidate.get("train_days", TRAIN_DAYS))

    def fit(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        train_index: pd.DatetimeIndex,
        *,
        valid_index: pd.DatetimeIndex | None = None,
    ) -> None:
        self.fitted_state = {
            "penalty": self.penalty,
            "train_days": self.train_days,
            "feedback_lag_days": FEEDBACK_LAG_DAYS,
            "error_lags_hours": list(ERROR_LAGS_HOURS),
            "train_rows": int(len(train_index)),
            "source_module": SOURCE_MODULE,
            "implementation_note": (
                "admitted lag2_pipeline parameters and feature/train surface; "
                "isolated copy to avoid killtest opsd_* import graph"
            ),
        }

    def predict(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        predict_index: pd.DatetimeIndex,
    ) -> PointForecast:
        features, errors = build_lag2_features(actual, forecast)
        if len(predict_index) == 0:
            return PointForecast(point=empty_like(predict_index), meta={})

        start = str(pd.Timestamp(predict_index.min()).date())
        end = str(pd.Timestamp(predict_index.max()).date())
        predicted_err, _corrected, audit = forward_correction_lag2(
            features,
            errors,
            start,
            end,
            train_days=self.train_days,
            penalty=self.penalty,
        )
        point = empty_like(predict_index)
        common = predicted_err.index.intersection(predict_index).intersection(forecast.index)
        if len(common):
            point.loc[common] = (
                forecast.reindex(common)[list(ZONES)].to_numpy(float)
                + predicted_err.reindex(common)[list(ZONES)].to_numpy(float)
            )
        eligibility = pd.Series("eligible", index=predict_index, dtype=object)
        missing = point.isna().any(axis=1)
        eligibility.loc[missing] = "ridge_lag2_no_row"
        return PointForecast(
            point=align_zones(point),
            eligibility=eligibility,
            meta={
                "n_audit_days": len(audit),
                "penalty": self.penalty,
                "train_days": self.train_days,
                "model_hash": self.model_hash(),
                "config_hash": self.config_hash,
                "feedback_lag_days": FEEDBACK_LAG_DAYS,
                "source_module": SOURCE_MODULE,
            },
        )
