"""M5: Temporal Fusion Transformer quantile adapter.

Uses pytorch-forecasting as a designated maintained implementation, explicitly
not claimed as official Google paper-release parity.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import QUANTILES, ZONES
from ..features import build_primary_features
from .base import ModelAdapter, PointForecast, empty_like


class TFTQuantileAdapter(ModelAdapter):
    model_name = "tft_quantile"
    supports_quantiles = True
    implementation_label = (
        "designated_maintained_implementation_not_official_google_paper_release"
    )

    def __init__(self, candidate: dict[str, Any] | None = None):
        candidate = candidate or {
            "candidate_id": "tft_h32_lstm1_attn2_lr1e3_e8_b512",
            "hidden_size": 32,
            "lstm_layers": 1,
            "attention_head_size": 2,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "max_epochs": 8,
            "batch_size": 512,
            "early_stopping_patience": 2,
            "limit_train_batches": 350,
            "seed": 20260718,
        }
        super().__init__(candidate)
        self._fallback_models: dict[str, Any] = {}
        self._mode = "unfitted"

    @staticmethod
    def _lag2_indices(
        day_start: pd.Timestamp, encoder_length: int, decoder_length: int
    ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        prior_day_start = day_start - pd.DateOffset(days=1)
        history_index = pd.date_range(
            end=prior_day_start - pd.Timedelta(hours=1),
            periods=encoder_length,
            freq="h",
        )
        decoder_index = pd.date_range(
            start=history_index[-1] + pd.Timedelta(hours=1),
            periods=decoder_length,
            freq="h",
        )
        complete_utc = history_index.append(decoder_index).tz_convert("UTC")
        if not (
            np.diff(complete_utc.asi8) == pd.Timedelta(hours=1).value
        ).all():
            raise RuntimeError("TFT lag-2 history/decoder is not UTC-hour contiguous")
        return history_index, decoder_index

    @staticmethod
    def _known_forecast_on_grid(
        forecast: pd.DataFrame, index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        union = forecast.index.union(index).sort_values()
        filled = forecast.reindex(union).interpolate(
            method="time", limit_area="inside"
        ).ffill(limit=1).bfill(limit=1).reindex(index)
        if not np.isfinite(filled.to_numpy(float)).all():
            raise ValueError("known P7 forecast cannot fill the TFT continuous grid")
        return filled

    def fit(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        train_index: pd.DatetimeIndex,
        *,
        valid_index: pd.DatetimeIndex | None = None,
    ) -> None:
        actual, forecast = self._model_inputs(actual, forecast)
        # Prefer pytorch-forecasting TFT when available; otherwise a disclosed
        # multi-horizon LightGBM quantile fallback is NOT labelled as TFT.
        try:
            import pytorch_forecasting  # noqa: F401
            import torch  # noqa: F401

            self._fit_pytorch_forecasting(actual, forecast, train_index, valid_index)
            return
        except Exception as exc:  # noqa: BLE001
            self.fitted_state = {
                "status": "BLOCKED_OR_FALLBACK",
                "pytorch_forecasting_error": str(exc),
                "implementation_label": self.implementation_label,
            }
            # Disclosed non-TFT fallback for residual common-SAOCP continuity only
            # is intentionally NOT used here: fail closed for native TFT row.
            self._mode = "blocked"
            raise RuntimeError(
                f"TFT pytorch-forecasting path unavailable: {exc}"
            ) from exc

    def _fit_pytorch_forecasting(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        train_index: pd.DatetimeIndex,
        valid_index: pd.DatetimeIndex | None,
    ) -> None:
        """Train a compact multi-target quantile model via designated library.

        Full 11-zone 24h multi-horizon TFT with TimeSeriesDataSet is heavy; this
        adapter uses a zonal multi-quantile feed through pytorch-forecasting's
        TemporalFusionTransformer where environment supports it. Any deviation
        from paper/demo configs is recorded in fitted_state.
        """
        import lightning.pytorch as pl
        import torch
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss

        seed = int(self.candidate.get("seed", 20260718))
        pl.seed_everything(seed, workers=True)

        # Long panel: one series per zone.
        bundle = build_primary_features(actual, forecast)
        train_idx = train_index.intersection(bundle.features.index)
        valid_idx = (
            pd.DatetimeIndex(valid_index).intersection(bundle.features.index)
            if valid_index is not None
            else pd.DatetimeIndex([])
        )
        idx = train_idx.append(valid_idx).sort_values().unique()
        if len(train_idx) < 24 * 60 or len(valid_idx) < 49:
            raise RuntimeError("insufficient rows for TFT training")

        # Vectorized panel construction is both faster and avoids hundreds of
        # thousands of Python scalar conversions on the remote runtime.
        time_idx = idx.asi8 // 10**9
        hour = idx.hour.to_numpy(dtype=np.int64)
        dow = idx.dayofweek.to_numpy(dtype=np.int64)
        panels = []
        for zone in ZONES:
            panels.append(
                pd.DataFrame(
                    {
                        "time_idx": time_idx,
                        "zone": zone,
                        "target": actual.loc[idx, zone].to_numpy(dtype=float),
                        "forecast": forecast.loc[idx, zone].to_numpy(dtype=float),
                        "hour": hour,
                        "dow": dow,
                    }
                )
            )
        data = pd.concat(panels, ignore_index=True)
        # Compact integer time index per zone.
        data = data.sort_values(["zone", "time_idx"]).reset_index(drop=True)
        data["t"] = data.groupby("zone").cumcount()

        max_encoder = int(
            self.candidate.get(
                "max_encoder_length", self.candidate.get("encoder_length", 168)
            )
        )
        # A continuous 49-hour decoder covers the preceding local day plus
        # every 23/24/25-hour delivery day without touching day-(t-1) outcomes.
        max_prediction = int(self.candidate.get("max_prediction_length", 49))
        if max_prediction != 49:
            raise ValueError("TFT DST-complete lag-2 decoder must have length 49")
        train_rows_per_zone = len(train_idx)
        training = TimeSeriesDataSet(
            data[data["t"] < train_rows_per_zone],
            time_idx="t",
            target="target",
            group_ids=["zone"],
            min_encoder_length=max_encoder // 2,
            max_encoder_length=max_encoder,
            min_prediction_length=1,
            max_prediction_length=max_prediction,
            time_varying_known_reals=["forecast", "hour", "dow"],
            time_varying_unknown_reals=["target"],
            target_normalizer=GroupNormalizer(groups=["zone"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        validation = TimeSeriesDataSet.from_dataset(
            training,
            data,
            min_prediction_idx=train_rows_per_zone,
            stop_randomization=True,
        )
        train_loader = training.to_dataloader(
            train=True, batch_size=int(self.candidate.get("batch_size", 64)), num_workers=0
        )
        val_loader = validation.to_dataloader(
            train=False, batch_size=int(self.candidate.get("batch_size", 64)), num_workers=0
        )

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=float(self.candidate.get("learning_rate", 1e-3)),
            hidden_size=int(self.candidate.get("hidden_size", 32)),
            attention_head_size=int(self.candidate.get("attention_head_size", 2)),
            dropout=float(self.candidate.get("dropout", 0.1)),
            hidden_continuous_size=int(self.candidate.get("hidden_size", 32)) // 2,
            lstm_layers=int(self.candidate.get("lstm_layers", 1)),
            loss=QuantileLoss(quantiles=list(QUANTILES)),
            reduce_on_plateau_patience=3,
        )
        early = EarlyStopping(
            monitor="val_loss",
            patience=int(self.candidate.get("early_stopping_patience", 5)),
            mode="min",
        )
        callbacks = [early]
        recovery_dir_value = os.environ.get("CARD10_TFT_RECOVERY_DIR", "").strip()
        recovery_dir = Path(recovery_dir_value) if recovery_dir_value else None
        resume_checkpoint = None
        rejected_recovery_checkpoints: list[str] = []
        if recovery_dir is not None:
            recovery_dir.mkdir(parents=True, exist_ok=True)
            # Retain each completed epoch separately. A native crash during one
            # write therefore cannot destroy the previous completed epoch.
            checkpoint_callback = ModelCheckpoint(
                dirpath=str(recovery_dir),
                filename="epoch-{epoch:03d}",
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_top_k=-1,
                save_last=False,
            )
            callbacks.append(checkpoint_callback)
            for path in sorted(
                recovery_dir.glob("*.ckpt"),
                key=lambda item: item.stat().st_mtime_ns,
                reverse=True,
            ):
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    if not isinstance(payload, dict) or "epoch" not in payload:
                        raise ValueError("checkpoint has no Lightning epoch")
                    resume_checkpoint = str(path)
                    break
                except Exception:  # noqa: BLE001
                    # Preserve an incomplete checkpoint for diagnosis; try the
                    # preceding independently written epoch instead.
                    rejected_recovery_checkpoints.append(path.name)
        trainer = pl.Trainer(
            max_epochs=int(self.candidate.get("max_epochs", 30)),
            accelerator="auto",
            devices=1,
            gradient_clip_val=0.1,
            callbacks=callbacks,
            enable_checkpointing=recovery_dir is not None,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=bool(self.candidate.get("enable_progress_bar", False)),
            # PyTorch has no deterministic CUDA backward for TFT's linear
            # upsampling kernel. ``warn`` keeps all available deterministic
            # kernels enabled while honestly recording this one limitation.
            deterministic="warn",
            limit_train_batches=int(self.candidate.get("limit_train_batches", 350)),
        )
        trainer.fit(
            tft,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_checkpoint,
        )
        state_hash = hashlib.sha256()
        for key, value in sorted(tft.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            state_hash.update(key.encode("utf-8"))
            state_hash.update(str(tensor.dtype).encode("ascii"))
            state_hash.update(str(tuple(tensor.shape)).encode("ascii"))
            state_hash.update(tensor.numpy().tobytes())
        self._tft = tft
        self._training_dataset = training
        self._data_template = data
        self._max_encoder_length = max_encoder
        self._max_prediction_length = max_prediction
        self._mode = "pytorch_forecasting_tft"
        self.fitted_state = {
            "mode": self._mode,
            "implementation_label": self.implementation_label,
            "package": "pytorch-forecasting",
            "selected_epoch": int(trainer.current_epoch),
            "train_rows": int(len(train_idx)),
            "valid_rows": int(len(valid_idx)),
            "max_encoder_length": max_encoder,
            "max_prediction_length": max_prediction,
            "scored_delivery_day_via_timestamp_lookup": True,
            "feedback_lag_days": 2,
            "state_dict_sha256": state_hash.hexdigest(),
            "cuda_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
            "determinism_mode": "warn",
            "known_nondeterministic_cuda_kernel": "upsample_linear1d_backward_out_cuda",
            "limit_train_batches": int(self.candidate.get("limit_train_batches", 350)),
            "hidden_size": int(self.candidate.get("hidden_size", 32)),
            "quantiles": list(QUANTILES),
            "fault_recovery_enabled": recovery_dir is not None,
            "fault_recovery_resumed": resume_checkpoint is not None,
            "fault_recovery_resume_checkpoint": (
                Path(resume_checkpoint).name if resume_checkpoint else None
            ),
            "fault_recovery_rejected_checkpoints": rejected_recovery_checkpoints,
            "deviation_notes": (
                "NYISO multi-zone panel wired through TimeSeriesDataSet; "
                "not official Google TFT paper-release parity"
            ),
        }

    def save_checkpoint(self, path: str) -> None:
        """Persist a self-identifying TFT checkpoint for one origin."""
        if self._mode != "pytorch_forecasting_tft":
            raise RuntimeError("cannot save an unfitted TFT checkpoint")
        import torch

        torch.save(
            {
                "state_dict": self._tft.state_dict(),
                "candidate": self.candidate,
                "fitted_state": self.fitted_state,
                "implementation_label": self.implementation_label,
            },
            path,
        )

    def predict(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        predict_index: pd.DatetimeIndex,
    ) -> PointForecast:
        if self._mode != "pytorch_forecasting_tft":
            raise RuntimeError("TFT adapter not fitted in pytorch-forecasting mode")
        original_forecast = forecast
        actual, forecast = self._model_inputs(actual, forecast)
        # Lag-2 daily prediction is implemented below.
        predict_index = pd.DatetimeIndex(predict_index).sort_values().unique()
        point = empty_like(predict_index)
        q05 = empty_like(predict_index)
        q50 = empty_like(predict_index)
        q95 = empty_like(predict_index)

        import torch
        from pytorch_forecasting import TimeSeriesDataSet

        recovery_dir_value = os.environ.get("CARD10_TFT_RECOVERY_DIR", "").strip()
        prediction_cache_dir = (
            Path(recovery_dir_value) / "prediction_datasets"
            if recovery_dir_value
            else None
        )
        if prediction_cache_dir is not None:
            prediction_cache_dir.mkdir(parents=True, exist_ok=True)
        prediction_cache_hits = 0
        prediction_cache_builds = 0
        rejected_prediction_caches: list[str] = []
        eligibility = pd.Series("not_scored", index=predict_index, dtype=object)
        if predict_index.empty:
            return self._restore_prediction(PointForecast(
                point=point,
                q05=q05,
                q50=q50,
                q95=q95,
                eligibility=eligibility,
                meta={"mode": self._mode, "status": "PASS_EMPTY"},
            ), original_forecast)
        if not predict_index.isin(forecast.index).all():
            raise ValueError("TFT predict_index is not fully covered by known forecasts")

        encoder_length = int(self._max_encoder_length)
        decoder_length = int(self._max_prediction_length)
        if decoder_length != 49:
            raise RuntimeError("current lag-2 TFT inference requires a 49-hour decoder")

        scored_days = pd.DatetimeIndex(predict_index.normalize().unique()).sort_values()
        datasets = []
        day_records: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
        history_gap_rows = 0
        known_forecast_gap_rows = 0
        for day_start in scored_days:
            history_index, decoder_index = self._lag2_indices(
                day_start, encoder_length, decoder_length
            )
            requested_day = predict_index[predict_index.normalize() == day_start]
            complete_index = history_index.append(decoder_index)
            known_forecast = self._known_forecast_on_grid(forecast, complete_index)
            observed_history = self._fill_history(
                actual.reindex(history_index), known_forecast.reindex(history_index)
            )
            if not np.isfinite(observed_history.to_numpy(float)).all():
                raise ValueError("TFT historical gaps cannot be filled from known P7")
            history_gap_rows += int((~history_index.isin(actual.index)).sum())
            known_forecast_gap_rows += int((~complete_index.isin(forecast.index)).sum())
            time_idx = complete_index.asi8 // 10**9
            hour = complete_index.hour.to_numpy(dtype=np.int64)
            dow = complete_index.dayofweek.to_numpy(dtype=np.int64)
            t = np.arange(len(complete_index), dtype=np.int64)
            panels = []
            for zone in ZONES:
                panels.append(
                    pd.DataFrame(
                        {
                            "time_idx": time_idx,
                            "zone": zone,
                            "target": np.concatenate(
                                [
                                    observed_history[zone].to_numpy(float),
                                    np.zeros(decoder_length, dtype=float),
                                ]
                            ),
                            "forecast": known_forecast[zone].to_numpy(float),
                            "hour": hour,
                            "dow": dow,
                            "t": t,
                        }
                    )
                )
            day_data = pd.concat(panels, ignore_index=True)
            dataset = None
            day_token = day_start.strftime("%Y-%m-%d")
            data_sha256 = hashlib.sha256(
                pd.util.hash_pandas_object(day_data, index=True).values.tobytes()
            ).hexdigest()
            if prediction_cache_dir is not None:
                cache_paths = sorted(
                    prediction_cache_dir.glob(f"{day_token}-*.pt"),
                    key=lambda item: item.stat().st_mtime_ns,
                    reverse=True,
                )
                for cache_path in cache_paths:
                    try:
                        envelope = torch.load(
                            cache_path, map_location="cpu", weights_only=False
                        )
                        if not isinstance(envelope, dict):
                            raise ValueError("prediction dataset cache is not an envelope")
                        if envelope.get("config_hash") != self.config_hash:
                            raise ValueError("prediction dataset config hash mismatch")
                        if envelope.get("delivery_day") != day_token:
                            raise ValueError("prediction dataset day mismatch")
                        if envelope.get("data_sha256") != data_sha256:
                            raise ValueError("prediction dataset content hash mismatch")
                        candidate = envelope.get("dataset")
                        if not isinstance(candidate, TimeSeriesDataSet):
                            raise TypeError("prediction cache is not a TimeSeriesDataSet")
                        if len(candidate) != len(ZONES):
                            raise ValueError("prediction cache has unexpected series count")
                        dataset = candidate
                        prediction_cache_hits += 1
                        break
                    except Exception:  # noqa: BLE001
                        rejected_prediction_caches.append(cache_path.name)
            if dataset is None:
                dataset = TimeSeriesDataSet.from_dataset(
                    self._training_dataset,
                    day_data,
                    predict=True,
                    stop_randomization=True,
                )
                if len(dataset) != len(ZONES):
                    raise RuntimeError("TFT day dataset does not contain eleven zones")
                prediction_cache_builds += 1
                if prediction_cache_dir is not None:
                    cache_path = (
                        prediction_cache_dir / f"{day_token}-{os.getpid()}.pt"
                    )
                    temp_path = cache_path.with_suffix(".tmp")
                    torch.save(
                        {
                            "schema": "card10_tft_prediction_dataset_cache_v1",
                            "config_hash": self.config_hash,
                            "delivery_day": day_token,
                            "data_sha256": data_sha256,
                            "dataset": dataset,
                        },
                        temp_path,
                    )
                    os.replace(temp_path, cache_path)
            datasets.append(dataset)
            if (decoder_index.get_indexer(requested_day) < 0).any():
                raise RuntimeError(f"TFT decoder does not cover delivery day {day_start}")
            day_records.append((decoder_index, requested_day))

        prediction_batch_size = max(
            len(ZONES), int(self.candidate.get("batch_size", 64))
        )
        if len(datasets) == 1:
            loader = dataset.to_dataloader(
                train=False,
                batch_size=prediction_batch_size,
                num_workers=0,
            )
        else:
            class _IndexedConcatTimeSeriesDataSet(TimeSeriesDataSet):
                """Concatenate homogeneous prediction datasets for one Trainer.

                PyTorch Forecasting requires the loader dataset itself to be a
                TimeSeriesDataSet. Copying the first dataset's fitted metadata
                preserves its encoders/scalers; only item dispatch is combined.
                """

                def __init__(self, parts):
                    self.__dict__.update(parts[0].__dict__)
                    self._indexed_parts = tuple(parts)
                    self._indexed_part_length = len(parts[0])
                    if any(len(part) != self._indexed_part_length for part in parts):
                        raise ValueError("inhomogeneous TFT day dataset lengths")

                def __len__(self):
                    return len(self._indexed_parts) * self._indexed_part_length

                def __getitem__(self, index):
                    part, local = divmod(index, self._indexed_part_length)
                    return self._indexed_parts[part][local]

            combined = _IndexedConcatTimeSeriesDataSet(datasets)
            loader = combined.to_dataloader(
                train=False,
                batch_size=prediction_batch_size,
                num_workers=0,
            )

        prediction = self._tft.predict(
            loader,
            mode="quantiles",
            return_index=True,
            trainer_kwargs={
                "accelerator": "auto",
                "devices": 1,
                "logger": False,
                "enable_progress_bar": False,
            },
        )
        output = getattr(prediction, "output", prediction)
        index_frame = getattr(prediction, "index", None)
        arr = (
            output.detach().cpu().numpy()
            if hasattr(output, "detach")
            else np.asarray(output)
        )
        expected_shape = (
            len(scored_days) * len(ZONES),
            decoder_length,
            len(QUANTILES),
        )
        if arr.shape != expected_shape:
            raise RuntimeError(
                f"unexpected TFT prediction shape {arr.shape}; expected {expected_shape}"
            )
        if index_frame is None or "zone" not in index_frame:
            raise RuntimeError("TFT prediction did not return stable zone indices")
        all_predicted_zones = [str(value) for value in index_frame["zone"].tolist()]
        for day_i, (decoder_index, requested) in enumerate(day_records):
            start = day_i * len(ZONES)
            stop = start + len(ZONES)
            predicted_zones = all_predicted_zones[start:stop]
            if set(predicted_zones) != set(ZONES):
                raise RuntimeError(
                    f"unexpected TFT zone index for day {scored_days[day_i]}: "
                    f"{predicted_zones}"
                )
            positions = decoder_index.get_indexer(requested)
            for local_row, zone in enumerate(predicted_zones):
                # Frozen monotone rearrangement prevents finite-sample
                # quantile crossing without looking at outcomes.
                row = np.sort(arr[start + local_row, :, :], axis=1)
                q05.loc[requested, zone] = row[positions, 0]
                q50.loc[requested, zone] = row[positions, 1]
                q95.loc[requested, zone] = row[positions, 2]
                point.loc[requested, zone] = row[positions, 1]
            eligibility.loc[requested] = "eligible"

        scored = eligibility.eq("eligible")
        if point.loc[scored].isna().any().any():
            raise RuntimeError("TFT produced missing values for eligible timestamps")
        ordered = (q05 <= q50) & (q50 <= q95)
        if not ordered.loc[scored].all().all():
            raise RuntimeError("TFT quantiles are not ordered")
        self.fitted_state["prediction_dataset_cache_enabled"] = (
            prediction_cache_dir is not None
        )
        self.fitted_state["prediction_dataset_cache_hits"] = prediction_cache_hits
        self.fitted_state["prediction_dataset_cache_builds"] = prediction_cache_builds
        self.fitted_state["prediction_dataset_cache_rejected"] = len(
            rejected_prediction_caches
        )
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
                "feedback_lag_days": 2,
                "scored_delivery_day_via_timestamp_lookup": True,
                "quantile_crossing_rule": "rowwise_monotone_rearrangement",
                "dst_rule": "utc_contiguous_49h_decoder_scored_by_local_delivery_timestamp",
                "history_gap_rows_filled_from_known_p7": history_gap_rows,
                "known_p7_gap_rows_filled": known_forecast_gap_rows,
                "prediction_dataset_cache_enabled": prediction_cache_dir is not None,
                "prediction_dataset_cache_hits": prediction_cache_hits,
                "prediction_dataset_cache_builds": prediction_cache_builds,
                "prediction_dataset_cache_rejected": len(
                    rejected_prediction_caches
                ),
                "prediction_dataset_cache_rejected_examples": (
                    rejected_prediction_caches[:20]
                ),
                "gap_fill_scope": "connector_only_no_outcome_interpolation_one_step_endpoint_p7_padding",
                "model_hash": self.model_hash(),
                "config_hash": self.config_hash,
            },
        ), original_forecast)
