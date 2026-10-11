"""Common adapter surface for R78 point and quantile models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..constants import ZONES
from ..hash_util import sha256_json


@dataclass
class PointForecast:
    """Wide point forecasts aligned to a DatetimeIndex, columns = ZONES."""

    point: pd.DataFrame
    q05: pd.DataFrame | None = None
    q50: pd.DataFrame | None = None
    q95: pd.DataFrame | None = None
    eligibility: pd.Series | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if list(self.point.columns) != list(ZONES):
            raise ValueError("point columns must match frozen ZONES order")
        for frame in (self.q05, self.q50, self.q95):
            if frame is not None and list(frame.columns) != list(ZONES):
                raise ValueError("quantile columns must match frozen ZONES order")


class ModelAdapter(ABC):
    model_name: str
    supports_quantiles: bool = False

    def __init__(self, candidate: dict[str, Any]):
        self.candidate = dict(candidate)
        self.candidate_id = str(candidate.get("candidate_id", self.model_name))
        self.config_hash = sha256_json(self.candidate)
        self.fitted_state: dict[str, Any] = {}

    @abstractmethod
    def fit(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        train_index: pd.DatetimeIndex,
        *,
        valid_index: pd.DatetimeIndex | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        predict_index: pd.DatetimeIndex,
    ) -> PointForecast:
        raise NotImplementedError

    def model_hash(self) -> str:
        payload = {
            "model": self.model_name,
            "candidate_id": self.candidate_id,
            "config_hash": self.config_hash,
            "state_keys": sorted(self.fitted_state.keys()),
        }
        # Include lightweight state fingerprints when present.
        for key in (
            "selected_iteration",
            "n_estimators",
            "penalty",
            "train_rows",
            "state_dict_sha256",
        ):
            if key in self.fitted_state:
                payload[key] = self.fitted_state[key]
        return sha256_json(payload)

    def _model_inputs(
        self, actual: pd.DataFrame, forecast: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the frozen target/known-input representation for this candidate.

        ``p7_residual`` is an explicit engineering adaptation: the learner sees
        actual-minus-P7 as its target while adapters that support known-future
        P7 may retain it as an exogenous input. Prediction is restored to load
        level by :meth:`_restore_prediction`. Defaults remain byte-for-byte
        equivalent to the original load-level adapters.
        """
        mode = str(self.candidate.get("target_mode", "load_level"))
        if mode == "load_level":
            return actual, forecast
        if mode == "p7_residual":
            aligned = forecast.reindex(index=actual.index, columns=actual.columns)
            residual = actual - aligned
            return residual, forecast
        raise ValueError(f"unknown target_mode: {mode}")

    def _fill_history(
        self, observed: pd.DataFrame, known_forecast: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the disclosed connector-only history-gap rule in target space."""
        mode = str(self.candidate.get("target_mode", "load_level"))
        if mode == "load_level":
            return observed.fillna(known_forecast)
        if mode == "p7_residual":
            # The neutral residual corresponding to the known P7 forecast is 0,
            # not the P7 load level itself.
            return observed.fillna(0.0)
        raise ValueError(f"unknown target_mode: {mode}")

    def _restore_prediction(
        self, result: PointForecast, original_forecast: pd.DataFrame
    ) -> PointForecast:
        mode = str(self.candidate.get("target_mode", "load_level"))
        result.meta["target_mode"] = mode
        if mode == "load_level":
            return result
        if mode != "p7_residual":
            raise ValueError(f"unknown target_mode: {mode}")
        base = original_forecast.reindex(
            index=result.point.index, columns=result.point.columns
        )
        if not np.isfinite(base.to_numpy(float)).all():
            raise ValueError("P7 residual restoration requires complete known forecasts")
        result.point = result.point + base
        for field_name in ("q05", "q50", "q95"):
            frame = getattr(result, field_name)
            if frame is not None:
                setattr(result, field_name, frame + base)
        result.meta["adaptation"] = "predict_actual_minus_p7_then_add_known_p7"
        result.meta["vanilla_model_claimed"] = False
        return result


def empty_like(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index, columns=list(ZONES), dtype=float)


def align_zones(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reindex(columns=list(ZONES))
