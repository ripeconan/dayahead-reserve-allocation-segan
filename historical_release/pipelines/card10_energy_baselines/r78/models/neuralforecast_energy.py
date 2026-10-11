"""Designated NeuralForecast N-HiTS and PatchTST quantile baselines."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from ..constants import QUANTILES, ZONES
from .base import ModelAdapter, PointForecast, empty_like


class _NeuralForecastEnergyAdapter(ModelAdapter):
    supports_quantiles = True
    implementation_label = "nixtla_neuralforecast_1_7_7_designated_implementation"
    architecture: str

    def __init__(self, candidate: dict[str, Any]):
        super().__init__(candidate)
        self._mode = "unfitted"

    def _future_exog_names(self) -> list[str]:
        if self.architecture == "nhits":
            return ["p7", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        if self.architecture == "patchtst":
            return []
        raise RuntimeError(f"unknown NeuralForecast architecture {self.architecture}")

    @staticmethod
    def _known_forecast_on_grid(
        forecast: pd.DataFrame, index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        union = forecast.index.union(index).sort_values()
        filled = forecast.reindex(union).interpolate(
            method="time", limit_area="inside"
        ).ffill(limit=1).bfill(limit=1).reindex(index)
        if not np.isfinite(filled.to_numpy(float)).all():
            raise ValueError("known P7 forecast cannot fill NeuralForecast grid")
        return filled

    @staticmethod
    def _calendar(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        hour = index.hour.to_numpy(float)
        dow = index.dayofweek.to_numpy(float)
        return {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        }

    def _long_frame(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        ds = index.tz_convert("UTC").tz_localize(None)
        calendar = self._calendar(index)
        parts = []
        for zone in ZONES:
            mean, scale = self._zone_scale[zone]
            parts.append(
                pd.DataFrame(
                    {
                        "unique_id": zone,
                        "ds": ds,
                        "y": (actual.loc[index, zone].to_numpy(float) - mean) / scale,
                        "p7": (forecast.loc[index, zone].to_numpy(float) - mean) / scale,
                        **calendar,
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)

    def fit(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        train_index: pd.DatetimeIndex,
        *,
        valid_index: pd.DatetimeIndex | None = None,
    ) -> None:
        actual, forecast = self._model_inputs(actual, forecast)
        if valid_index is None or len(valid_index) < 49:
            raise RuntimeError("NeuralForecast requires a development-only validation split")
        import torch
        import neuralforecast
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast.models import NHITS, PatchTST

        if neuralforecast.__version__ != "1.7.7":
            raise RuntimeError(
                f"NeuralForecast version mismatch: {neuralforecast.__version__} != 1.7.7"
            )

        seed = int(self.candidate.get("seed", 20260718))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
        self._zone_scale = {}
        for zone in ZONES:
            values = actual.loc[train_index, zone].to_numpy(float)
            mean = float(np.mean(values))
            scale = float(np.std(values))
            if not np.isfinite(scale) or scale <= 0:
                raise RuntimeError(f"invalid training scale for {zone}")
            self._zone_scale[zone] = (mean, scale)

        combined_index = pd.DatetimeIndex(train_index).append(
            pd.DatetimeIndex(valid_index)
        )
        data = self._long_frame(actual, forecast, combined_index)
        h = int(self.candidate.get("forecast_horizon", 49))
        input_size = int(self.candidate.get("input_size", 168))
        if h != 49:
            raise ValueError("NeuralForecast DST-complete lag-2 horizon must be 49")
        loss = MQLoss(quantiles=list(QUANTILES))
        common = dict(
            h=h,
            input_size=input_size,
            loss=loss,
            valid_loss=loss,
            max_steps=int(self.candidate.get("max_steps", 300)),
            learning_rate=float(self.candidate.get("learning_rate", 1e-3)),
            val_check_steps=int(self.candidate.get("val_check_steps", 50)),
            early_stop_patience_steps=int(
                self.candidate.get("early_stop_patience_steps", 3)
            ),
            batch_size=int(self.candidate.get("batch_size", 64)),
            windows_batch_size=int(self.candidate.get("windows_batch_size", 512)),
            inference_windows_batch_size=int(
                self.candidate.get("inference_windows_batch_size", 1024)
            ),
            scaler_type="identity",
            random_seed=seed,
            num_workers_loader=0,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        future_exog_names = self._future_exog_names()
        if future_exog_names:
            common["futr_exog_list"] = future_exog_names
        if self.architecture == "nhits":
            model = NHITS(
                **common,
                stack_types=["identity", "identity", "identity"],
                n_blocks=[1, 1, 1],
                mlp_units=[[256, 256], [256, 256], [256, 256]],
                n_pool_kernel_size=[8, 4, 1],
                n_freq_downsample=[8, 4, 1],
                alias=self.model_name,
            )
        elif self.architecture == "patchtst":
            model = PatchTST(
                **common,
                encoder_layers=2,
                n_heads=4,
                hidden_size=64,
                linear_hidden_size=128,
                patch_len=24,
                stride=12,
                alias=self.model_name,
            )
        else:
            raise RuntimeError(f"unknown NeuralForecast architecture {self.architecture}")
        nf = NeuralForecast(models=[model], freq="h")
        nf.fit(df=data, val_size=len(valid_index), verbose=False)
        self._nf = nf
        self._horizon = h
        self._input_size = input_size
        self._uses_future_exog = bool(future_exog_names)
        self._mode = f"neuralforecast_{self.architecture}_mqloss"

        state_hash = hashlib.sha256()
        for key, value in sorted(nf.models[0].state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            state_hash.update(key.encode("utf-8"))
            state_hash.update(str(tensor.dtype).encode("ascii"))
            state_hash.update(str(tuple(tensor.shape)).encode("ascii"))
            state_hash.update(tensor.numpy().tobytes())
        self.fitted_state = {
            "mode": self._mode,
            "architecture": self.architecture,
            "implementation_label": self.implementation_label,
            "package": "neuralforecast",
            "package_version": "1.7.7",
            "train_rows": int(len(train_index)),
            "valid_rows": int(len(valid_index)),
            "input_size": input_size,
            "forecast_horizon": h,
            "feedback_lag_days": 2,
            "information_set": (
                "known_future_p7_and_calendar"
                if future_exog_names
                else "univariate_target_history_only_no_p7_or_calendar_exog"
            ),
            "state_dict_sha256": state_hash.hexdigest(),
            "cuda_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
        }

    def predict(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        predict_index: pd.DatetimeIndex,
    ) -> PointForecast:
        if not self._mode.startswith("neuralforecast_"):
            raise RuntimeError("NeuralForecast adapter not fitted")
        original_forecast = forecast
        actual, forecast = self._model_inputs(actual, forecast)
        predict_index = pd.DatetimeIndex(predict_index).sort_values().unique()
        point = empty_like(predict_index)
        q05 = empty_like(predict_index)
        q50 = empty_like(predict_index)
        q95 = empty_like(predict_index)
        eligibility = pd.Series("not_scored", index=predict_index, dtype=object)

        histories = []
        futures = []
        mapping: dict[str, tuple[str, pd.DatetimeIndex]] = {}
        history_gap_rows = 0
        known_forecast_gap_rows = 0
        for day_start in pd.DatetimeIndex(predict_index.normalize().unique()).sort_values():
            prior_day_start = day_start - pd.DateOffset(days=1)
            history_index = pd.date_range(
                end=prior_day_start - pd.Timedelta(hours=1),
                periods=self._input_size,
                freq="h",
            )
            future_index = pd.date_range(
                start=history_index[-1] + pd.Timedelta(hours=1),
                periods=self._horizon,
                freq="h",
            )
            requested = predict_index[predict_index.normalize() == day_start]
            requested_utc = requested.tz_convert("UTC").tz_localize(None)
            if (future_index.tz_convert("UTC").tz_localize(None).get_indexer(requested_utc) < 0).any():
                raise RuntimeError(f"49-hour forecast misses delivery day {day_start}")
            known_history = self._known_forecast_on_grid(forecast, history_index)
            known_future = self._known_forecast_on_grid(forecast, future_index)
            observed_history = self._fill_history(
                actual.reindex(history_index), known_history
            )
            if not np.isfinite(observed_history.to_numpy(float)).all():
                raise ValueError("NeuralForecast historical gap cannot be filled from known P7")
            history_gap_rows += int((~history_index.isin(actual.index)).sum())
            known_forecast_gap_rows += int(
                (~history_index.isin(forecast.index)).sum()
                + (~future_index.isin(forecast.index)).sum()
            )
            calendar_hist = self._calendar(history_index)
            calendar_futr = self._calendar(future_index)
            for zone in ZONES:
                mean, scale = self._zone_scale[zone]
                uid = f"{zone}__{day_start.date().isoformat()}"
                histories.append(
                    pd.DataFrame(
                        {
                            "unique_id": uid,
                            "ds": history_index.tz_convert("UTC").tz_localize(None),
                            "y": (observed_history[zone].to_numpy(float) - mean) / scale,
                            "p7": (known_history[zone].to_numpy(float) - mean) / scale,
                            **calendar_hist,
                        }
                    )
                )
                futures.append(
                    pd.DataFrame(
                        {
                            "unique_id": uid,
                            "ds": future_index.tz_convert("UTC").tz_localize(None),
                            "p7": (known_future[zone].to_numpy(float) - mean) / scale,
                            **calendar_futr,
                        }
                    )
                )
                mapping[uid] = (zone, requested)
        history_frame = pd.concat(histories, ignore_index=True)
        if self._uses_future_exog:
            prediction = self._nf.predict(
                df=history_frame,
                futr_df=pd.concat(futures, ignore_index=True),
                verbose=False,
            )
        else:
            prediction = self._nf.predict(df=history_frame, verbose=False)
        if "unique_id" not in prediction.columns:
            prediction = prediction.reset_index()
        value_columns = [c for c in prediction.columns if c not in {"unique_id", "ds"}]
        lo_col = next(c for c in value_columns if "lo-90" in c)
        mid_col = next(c for c in value_columns if "median" in c)
        hi_col = next(c for c in value_columns if "hi-90" in c)
        for uid, (zone, requested) in mapping.items():
            part = prediction[prediction["unique_id"].astype(str) == uid].copy()
            part = part.set_index(pd.to_datetime(part["ds"]))
            requested_utc = requested.tz_convert("UTC").tz_localize(None)
            selected = part.reindex(requested_utc)[[lo_col, mid_col, hi_col]].to_numpy(float)
            if not np.isfinite(selected).all():
                raise RuntimeError(f"{self.model_name} produced non-finite quantiles")
            selected = np.sort(selected, axis=1)
            mean, scale = self._zone_scale[zone]
            selected = selected * scale + mean
            q05.loc[requested, zone] = selected[:, 0]
            q50.loc[requested, zone] = selected[:, 1]
            q95.loc[requested, zone] = selected[:, 2]
            point.loc[requested, zone] = selected[:, 1]
            eligibility.loc[requested] = "eligible"
        if point.isna().any().any():
            raise RuntimeError(f"{self.model_name} produced missing delivery predictions")
        return self._restore_prediction(PointForecast(
            point=point,
            q05=q05,
            q50=q50,
            q95=q95,
            eligibility=eligibility,
            meta={
                "mode": self._mode,
                "implementation_label": self.implementation_label,
                "status": "PASS_LAG2_DST_COMPLETE_49H",
                "information_set": (
                    "known_future_p7_and_calendar"
                    if self._uses_future_exog
                    else "univariate_target_history_only_no_p7_or_calendar_exog"
                ),
                "quantile_crossing_rule": "rowwise_monotone_rearrangement",
                "history_gap_rows_filled_from_known_p7": history_gap_rows,
                "known_p7_gap_rows_filled": known_forecast_gap_rows,
                "gap_fill_scope": "connector_only_no_outcome_interpolation_one_step_endpoint_p7_padding",
                "model_hash": self.model_hash(),
                "config_hash": self.config_hash,
            },
        ), original_forecast)

    def save_checkpoint(self, path: str) -> None:
        if not self._mode.startswith("neuralforecast_"):
            raise RuntimeError("cannot save unfitted NeuralForecast model")
        import torch

        torch.save(
            {
                "state_dict": self._nf.models[0].state_dict(),
                "candidate": self.candidate,
                "fitted_state": self.fitted_state,
                "zone_scale": self._zone_scale,
            },
            path,
        )


class NHITSQuantileAdapter(_NeuralForecastEnergyAdapter):
    model_name = "nhits_quantile"
    architecture = "nhits"

    def __init__(self, candidate: dict[str, Any] | None = None):
        super().__init__(
            candidate
            or {
                "candidate_id": "nhits_mq_h49_i168_s300",
                "seed": 20260718,
                "forecast_horizon": 49,
                "input_size": 168,
                "max_steps": 300,
                "learning_rate": 0.001,
                "val_check_steps": 50,
                "early_stop_patience_steps": 3,
                "batch_size": 64,
                "windows_batch_size": 512,
                "inference_windows_batch_size": 1024,
            }
        )


class PatchTSTQuantileAdapter(_NeuralForecastEnergyAdapter):
    model_name = "patchtst_quantile"
    architecture = "patchtst"

    def __init__(self, candidate: dict[str, Any] | None = None):
        super().__init__(
            candidate
            or {
                "candidate_id": "patchtst_mq_h49_i168_s300",
                "seed": 20260718,
                "forecast_horizon": 49,
                "input_size": 168,
                "max_steps": 300,
                "learning_rate": 0.0005,
                "val_check_steps": 50,
                "early_stop_patience_steps": 3,
                "batch_size": 64,
                "windows_batch_size": 512,
                "inference_windows_batch_size": 1024,
            }
        )
