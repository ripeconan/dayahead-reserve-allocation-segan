"""M4: CatBoost quantile models (separate model per quantile per zone)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..constants import QUANTILES, ZONES
from ..features import build_primary_features
from .base import ModelAdapter, PointForecast, empty_like


class CatBoostEnergyAdapter(ModelAdapter):
    model_name = "catboost_energy_quantile"
    supports_quantiles = True

    def __init__(self, candidate: dict[str, Any] | None = None):
        candidate = candidate or {
            "candidate_id": "cat_depth6_lr003_l2_3",
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "iterations_max": 1000,
            "early_stopping_rounds": 20,
            "seed": 20260718,
        }
        super().__init__(candidate)
        self.models: dict[tuple[str, float], Any] = {}
        self.feature_names: list[str] = []

    def fit(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        train_index: pd.DatetimeIndex,
        *,
        valid_index: pd.DatetimeIndex | None = None,
    ) -> None:
        from catboost import CatBoostRegressor, Pool

        bundle = build_primary_features(actual, forecast)
        x_all = bundle.features
        train_idx = train_index.intersection(x_all.index)
        if len(train_idx) < 100:
            raise RuntimeError("insufficient CatBoost training rows")
        valid_idx = (
            valid_index.intersection(x_all.index)
            if valid_index is not None
            else train_idx[-min(len(train_idx) // 10, 24 * 90) :]
        )
        self.feature_names = list(x_all.columns)
        x_train = x_all.loc[train_idx]
        x_valid = x_all.loc[valid_idx]
        iterations = int(self.candidate.get("iterations_max", 1000))
        early = int(self.candidate.get("early_stopping_rounds", 20))
        selected: dict[str, int] = {}

        self.models = {}
        for zone in ZONES:
            y_train = actual.loc[train_idx, zone].to_numpy(float)
            y_valid = actual.loc[valid_idx, zone].to_numpy(float)
            train_pool = Pool(x_train, y_train)
            valid_pool = Pool(x_valid, y_valid)
            for q in QUANTILES:
                model = CatBoostRegressor(
                    loss_function=f"Quantile:alpha={q}",
                    depth=int(self.candidate.get("depth", 6)),
                    learning_rate=float(self.candidate.get("learning_rate", 0.03)),
                    l2_leaf_reg=float(self.candidate.get("l2_leaf_reg", 3.0)),
                    iterations=iterations,
                    random_seed=int(self.candidate.get("seed", 20260718)),
                    thread_count=8,
                    task_type="CPU",
                    bootstrap_type="Bernoulli",
                    subsample=0.9,
                    verbose=False,
                    allow_writing_files=False,
                )
                model.fit(
                    train_pool,
                    eval_set=valid_pool,
                    early_stopping_rounds=early,
                    verbose=False,
                )
                self.models[(zone, float(q))] = model
                selected[f"{zone}_q{q}"] = int(model.tree_count_)

        self.fitted_state = {
            "train_rows": int(len(train_idx)),
            "valid_rows": int(len(valid_idx)),
            "selected_iteration": selected,
            "quantile_mode": "separate_model_per_quantile_per_zone",
            "task_type": "CPU",
            "feature_names": self.feature_names,
        }

    def predict(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        predict_index: pd.DatetimeIndex,
    ) -> PointForecast:
        if not self.models:
            raise RuntimeError("CatBoost adapter not fitted")
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
            q05.loc[idx, zone] = self.models[(zone, 0.05)].predict(x)
            q50.loc[idx, zone] = self.models[(zone, 0.50)].predict(x)
            q95.loc[idx, zone] = self.models[(zone, 0.95)].predict(x)
            # Median is the point forecast for CatBoost quantile model.
            point.loc[idx, zone] = q50.loc[idx, zone]

        stacked = np.stack(
            [q05.loc[idx].to_numpy(), q50.loc[idx].to_numpy(), q95.loc[idx].to_numpy()],
            axis=2,
        )
        stacked.sort(axis=2)
        q05.loc[idx] = stacked[:, :, 0]
        q50.loc[idx] = stacked[:, :, 1]
        q95.loc[idx] = stacked[:, :, 2]
        point.loc[idx] = q50.loc[idx]

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
                "point_source": "median_q50",
                "quantile_mode": "separate_model_per_quantile_per_zone",
                "quantile_noncrossing_correction": "sort_q05_q50_q95_per_cell",
            },
        )
