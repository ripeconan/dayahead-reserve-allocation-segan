"""M3: LightGBM zonal point and quantile energy models (lag-2 features)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..constants import QUANTILES, ZONES
from ..features import build_primary_features
from .base import ModelAdapter, PointForecast, empty_like


class LightGBMEnergyAdapter(ModelAdapter):
    model_name = "lightgbm_energy"
    supports_quantiles = True

    def __init__(self, candidate: dict[str, Any] | None = None):
        candidate = candidate or {
            "candidate_id": "lgbm_zonal_d3_lr005_leaves31",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "n_estimators_max": 800,
            "early_stopping_rounds": 20,
            "seed": 20260718,
        }
        super().__init__(candidate)
        self.point_models: dict[str, Any] = {}
        self.quantile_models: dict[tuple[str, float], Any] = {}
        self.feature_names: list[str] = []

    def _params(self, objective: str, alpha: float | None = None) -> dict[str, Any]:
        params = {
            "objective": objective,
            "learning_rate": float(self.candidate.get("learning_rate", 0.05)),
            "num_leaves": int(self.candidate.get("num_leaves", 31)),
            "min_data_in_leaf": int(self.candidate.get("min_data_in_leaf", 50)),
            "feature_fraction": float(self.candidate.get("feature_fraction", 0.9)),
            "bagging_fraction": float(self.candidate.get("bagging_fraction", 0.9)),
            "bagging_freq": int(self.candidate.get("bagging_freq", 1)),
            "verbosity": -1,
            "seed": int(self.candidate.get("seed", 20260718)),
            "deterministic": True,
            "force_col_wise": True,
            "n_jobs": 4,
        }
        if alpha is not None:
            params["alpha"] = float(alpha)
        return params

    def fit(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        train_index: pd.DatetimeIndex,
        *,
        valid_index: pd.DatetimeIndex | None = None,
    ) -> None:
        import lightgbm as lgb

        bundle = build_primary_features(actual, forecast)
        x_all = bundle.features
        train_idx = train_index.intersection(x_all.index)
        if len(train_idx) < 100:
            raise RuntimeError("insufficient LightGBM training rows after lag-2 features")
        valid_idx = (
            valid_index.intersection(x_all.index)
            if valid_index is not None
            else train_idx[-min(len(train_idx) // 10, 24 * 90) :]
        )
        self.feature_names = list(x_all.columns)
        x_train = x_all.loc[train_idx]
        x_valid = x_all.loc[valid_idx]
        n_estimators = int(self.candidate.get("n_estimators_max", 800))
        early = int(self.candidate.get("early_stopping_rounds", 20))
        selected_iters: dict[str, int] = {}

        self.point_models = {}
        for zone in ZONES:
            y_train = actual.loc[train_idx, zone].to_numpy(float)
            y_valid = actual.loc[valid_idx, zone].to_numpy(float)
            dtrain = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
            dvalid = lgb.Dataset(x_valid, label=y_valid, reference=dtrain, free_raw_data=False)
            booster = lgb.train(
                self._params("regression"),
                dtrain,
                num_boost_round=n_estimators,
                valid_sets=[dvalid],
                valid_names=["valid"],
                callbacks=[
                    lgb.early_stopping(early, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            self.point_models[zone] = booster
            selected_iters[zone] = int(booster.best_iteration or n_estimators)

        self.quantile_models = {}
        for zone in ZONES:
            y_train = actual.loc[train_idx, zone].to_numpy(float)
            y_valid = actual.loc[valid_idx, zone].to_numpy(float)
            dtrain = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
            dvalid = lgb.Dataset(x_valid, label=y_valid, reference=dtrain, free_raw_data=False)
            for q in QUANTILES:
                booster = lgb.train(
                    self._params("quantile", alpha=q),
                    dtrain,
                    num_boost_round=n_estimators,
                    valid_sets=[dvalid],
                    valid_names=["valid"],
                    callbacks=[
                        lgb.early_stopping(early, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
                self.quantile_models[(zone, float(q))] = booster

        self.fitted_state = {
            "train_rows": int(len(train_idx)),
            "valid_rows": int(len(valid_idx)),
            "n_estimators": n_estimators,
            "selected_iteration": selected_iters,
            "feature_names": self.feature_names,
        }

    def predict(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        predict_index: pd.DatetimeIndex,
    ) -> PointForecast:
        if not self.point_models:
            raise RuntimeError("LightGBM adapter not fitted")
        bundle = build_primary_features(actual, forecast)
        idx = predict_index.intersection(bundle.features.index)
        point = empty_like(predict_index)
        q05 = empty_like(predict_index)
        q50 = empty_like(predict_index)
        q95 = empty_like(predict_index)
        if len(idx) == 0:
            elig = pd.Series("missing_features", index=predict_index, dtype=object)
            return PointForecast(point=point, q05=q05, q50=q50, q95=q95, eligibility=elig)

        x = bundle.features.loc[idx, self.feature_names]
        for zone in ZONES:
            point.loc[idx, zone] = self.point_models[zone].predict(x)
            q05.loc[idx, zone] = self.quantile_models[(zone, 0.05)].predict(x)
            q50.loc[idx, zone] = self.quantile_models[(zone, 0.50)].predict(x)
            q95.loc[idx, zone] = self.quantile_models[(zone, 0.95)].predict(x)

        # Predeclared non-crossing correction: monotone sort quantiles per cell.
        stacked = np.stack(
            [q05.loc[idx].to_numpy(), q50.loc[idx].to_numpy(), q95.loc[idx].to_numpy()],
            axis=2,
        )
        stacked.sort(axis=2)
        q05.loc[idx] = stacked[:, :, 0]
        q50.loc[idx] = stacked[:, :, 1]
        q95.loc[idx] = stacked[:, :, 2]

        eligibility = pd.Series("eligible", index=predict_index, dtype=object)
        eligibility.loc[~predict_index.isin(idx)] = "missing_features"
        return PointForecast(
            point=point,
            q05=q05,
            q50=q50,
            q95=q95,
            eligibility=eligibility,
            meta={
                "model_hash": self.model_hash(),
                "config_hash": self.config_hash,
                "quantile_noncrossing_correction": "sort_q05_q50_q95_per_cell",
            },
        )
