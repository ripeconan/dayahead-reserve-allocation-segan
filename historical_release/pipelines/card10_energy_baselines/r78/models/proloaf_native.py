"""M6: pinned public ProLoaF recurrent probabilistic baseline.

Only the NYISO data connector and frozen feature map are project code. Model,
loss, training loop, and quantile conversion use the public ProLoaF package.
This is therefore labelled a pinned-public-core adaptation, not paper-release
parity for a dataset/configuration the upstream project never shipped.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import QUANTILES, ZONES
from ..records import BlockerRow
from .base import ModelAdapter, PointForecast, empty_like


PROLOAF_REMOTE = "https://github.com/sogno-platform/proloaf.git"
PROLOAF_PIN = "e4a85c9dda725e7a5063899e805fa22a2f5e1129"


def _strict_time_series_data_class(base_class):
    """Return a one-dimensional ProLoaF dataset with fail-closed sample bounds.

    The pinned upstream ``TimeSeriesData.__getitem__`` maps indices through a
    cyclic modulo expression.  On the longer R86 rolling-origin frames this
    intermittently produced empty or near-full-frame slices during shuffled
    loading.  Card10 uses a simple one-dimensional DatetimeIndex, so direct
    bounded positional slices are the exact intended sample geometry.
    """

    class StrictTimeSeriesData(base_class):
        def __getitem__(self, idx):
            if not self.tensor_prepared:
                self.to_tensor()
            if isinstance(idx, np.generic):
                idx = idx.item()
            if not isinstance(idx, int):
                raise TypeError(f"ProLoaF sample index must be int, got {type(idx).__name__}")
            sample_count = len(self)
            if idx < 0 or idx >= sample_count:
                raise IndexError(
                    f"ProLoaF sample index {idx} outside [0, {sample_count})"
                )
            history = int(self.history_horizon)
            horizon = int(self.forecast_horizon)
            decoder_start = idx + history
            decoder_stop = decoder_start + horizon
            if decoder_stop > int(self.encoder_tensor.shape[0]):
                raise RuntimeError("ProLoaF sample exceeds prepared tensor boundary")
            encoder_slice = slice(idx, decoder_start)
            decoder_slice = slice(decoder_start, decoder_stop)
            last_value_slice = slice(decoder_start - 1, decoder_start)
            sample = (
                self.encoder_tensor[encoder_slice],
                self.aux_tensor[encoder_slice],
                self.decoder_tensor[decoder_slice],
                self.aux_tensor[decoder_slice],
                self.target_tensor[last_value_slice],
                self.target_tensor[decoder_slice],
            )
            expected_rows = (history, history, horizon, horizon, 1, horizon)
            observed_rows = tuple(int(value.shape[0]) for value in sample)
            if observed_rows != expected_rows:
                raise RuntimeError(
                    "ProLoaF strict sample shape mismatch: "
                    f"observed={observed_rows} expected={expected_rows} idx={idx}"
                )
            return sample

    return StrictTimeSeriesData


def probe_proloaf_clone(repo: Path) -> dict[str, Any]:
    if not repo.is_dir():
        return {"status": "MISSING", "repo": str(repo)}

    def git(*args: str) -> str:
        process = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip())
        return process.stdout.strip()

    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    dirty = git("status", "--porcelain")
    try:
        remote = git("remote", "get-url", "origin")
    except Exception:
        remote = ""
    return {
        "status": "PRESENT",
        "repo": str(repo),
        "commit": commit,
        "tree": tree,
        "remote": remote,
        "clean": dirty == "",
        "dirty_paths": dirty.splitlines(),
        "full_40hex": len(commit) == 40,
    }


def proloaf_blocker(reason: str, **extra: Any) -> BlockerRow:
    return BlockerRow(
        model="proloaf_native",
        code="BLOCKED_PROLOAF_ADAPTER_OR_ENV",
        reason=reason,
        urls=(PROLOAF_REMOTE, "https://doi.org/10.1016/j.softx.2023.101487"),
        access_date="2026-07-19",
        missing_elements=tuple(extra.get("missing_elements", ())),
    )


class ProLoaFNativeAdapter(ModelAdapter):
    """Pinned public recurrent core with a causal NYISO connector."""

    model_name = "proloaf_native"
    supports_quantiles = True
    implementation_label = (
        "pinned_public_proloaf_core_with_nyiso_connector_not_paper_config_parity"
    )

    def __init__(self, candidate: dict[str, Any] | None = None, repo: Path | None = None):
        candidate = candidate or {
            "candidate_id": "proloaf_recurrent_nllgauss_h168_f49_e8_b512",
            "seed": 20260718,
            "repo": None,
            "work_dir": "runs/proloaf/runtime",
            "history_horizon": 168,
            "forecast_horizon": 49,
            "max_epochs": 8,
            "batch_size": 512,
            "learning_rate": 1e-4,
            "early_stopping_patience": 2,
        }
        super().__init__(candidate)
        configured_repo = candidate.get("repo")
        self.repo = Path(repo or configured_repo) if (repo or configured_repo) else None
        self.identity: dict[str, Any] = {}
        self._mode = "unfitted"

    @staticmethod
    def _calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
        hour = index.hour.to_numpy(float)
        dow = index.dayofweek.to_numpy(float)
        return pd.DataFrame(
            {
                "hour_sin": np.sin(2 * np.pi * hour / 24.0),
                "hour_cos": np.cos(2 * np.pi * hour / 24.0),
                "dow_sin": np.sin(2 * np.pi * dow / 7.0),
                "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            },
            index=index,
        )

    @staticmethod
    def _known_forecast_on_grid(
        forecast: pd.DataFrame, index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        union = forecast.index.union(index).sort_values()
        filled = forecast.reindex(union).interpolate(
            method="time", limit_area="inside"
        ).ffill(limit=1).bfill(limit=1).reindex(index)
        if not np.isfinite(filled.to_numpy(float)).all():
            raise ValueError("known P7 forecast cannot fill the requested continuous grid")
        return filled

    def _frame(
        self, actual: pd.DataFrame, forecast: pd.DataFrame, index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        known = ProLoaFNativeAdapter._known_forecast_on_grid(forecast, index)
        observed = self._fill_history(actual.reindex(index), known)
        if not np.isfinite(observed.to_numpy(float)).all():
            raise ValueError("historical gaps cannot be filled from the known P7 forecast")
        frame = pd.DataFrame(index=index)
        for zone in ZONES:
            frame[f"target_{zone}"] = observed[zone].to_numpy(float)
            frame[f"p7_{zone}"] = known[zone].to_numpy(float)
        return frame.join(ProLoaFNativeAdapter._calendar(index))

    @staticmethod
    def _decoder_frame(
        forecast: pd.DataFrame, index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Build known-future inputs without accepting an outcome object."""
        known = ProLoaFNativeAdapter._known_forecast_on_grid(forecast, index)
        frame = pd.DataFrame(index=index)
        for zone in ZONES:
            frame[f"target_{zone}"] = 0.0
            frame[f"p7_{zone}"] = known[zone].to_numpy(float)
        return frame.join(ProLoaFNativeAdapter._calendar(index))

    @staticmethod
    def _lag2_indices(
        day_start: pd.Timestamp, history: int, horizon: int
    ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        prior_day_start = day_start - pd.DateOffset(days=1)
        history_index = pd.date_range(
            end=prior_day_start - pd.Timedelta(hours=1), periods=history, freq="h"
        )
        decoder_index = pd.date_range(
            start=history_index[-1] + pd.Timedelta(hours=1),
            periods=horizon,
            freq="h",
        )
        combined_utc = history_index.append(decoder_index).tz_convert("UTC")
        if not (np.diff(combined_utc.asi8) == pd.Timedelta(hours=1).value).all():
            raise RuntimeError("ProLoaF lag-2 history/decoder is not UTC-hour contiguous")
        return history_index, decoder_index

    def fit(
        self,
        actual: pd.DataFrame,
        forecast: pd.DataFrame,
        train_index: pd.DatetimeIndex,
        *,
        valid_index: pd.DatetimeIndex | None = None,
    ) -> None:
        actual, forecast = self._model_inputs(actual, forecast)
        if self.repo is None or not self.repo.is_dir():
            raise RuntimeError("ProLoaF repo missing; cannot fit pinned public core")
        self.identity = probe_proloaf_clone(self.repo)
        if self.identity.get("commit") != PROLOAF_PIN:
            raise RuntimeError(
                f"ProLoaF pin mismatch: {self.identity.get('commit')} != {PROLOAF_PIN}"
            )
        if not self.identity.get("clean"):
            raise RuntimeError(
                f"ProLoaF tree dirty; refuse pinned-public label: {self.identity['dirty_paths']}"
            )
        if valid_index is None or len(valid_index) < 48:
            raise RuntimeError("ProLoaF requires an explicit development-only validation split")

        import torch
        from proloaf import datahandler as dh
        from proloaf import modelhandler as mh
        from proloaf.tensorloader import TimeSeriesData

        StrictTimeSeriesData = _strict_time_series_data_class(TimeSeriesData)

        seed = int(self.candidate.get("seed", 20260718))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        targets = [f"target_{zone}" for zone in ZONES]
        p7 = [f"p7_{zone}" for zone in ZONES]
        aux = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        encoder_features = targets + p7
        decoder_features = p7
        feature_groups = [
            {"name": "loads", "scaler": ["minmax", 0.0, 1.0], "features": targets + p7},
            {"name": "aux", "scaler": None, "features": aux},
        ]
        scaler = dh.MultiScaler(feature_groups)
        train_frame = self._frame(actual, forecast, pd.DatetimeIndex(train_index))
        valid_frame = self._frame(actual, forecast, pd.DatetimeIndex(valid_index))
        if not np.isfinite(train_frame.to_numpy(float)).all():
            raise RuntimeError("non-finite ProLoaF training connector values")
        scaler.fit(train_frame)
        train_scaled = scaler.transform(train_frame)
        valid_scaled = scaler.transform(valid_frame)

        history = int(
            self.candidate.get(
                "history_horizon", self.candidate.get("history_hours", 168)
            )
        )
        horizon = int(self.candidate.get("forecast_horizon", 49))
        if horizon != 49:
            raise ValueError("ProLoaF DST-complete lag-2 connector requires forecast_horizon=49")
        common = {
            "history_horizon": history,
            "forecast_horizon": horizon,
            "encoder_features": encoder_features,
            "decoder_features": decoder_features,
            "aux_features": aux,
            "target_id": targets,
            "device": device,
        }
        train_data = StrictTimeSeriesData(train_scaled, **common)
        valid_data = StrictTimeSeriesData(valid_scaled, **common)
        config = {
            "model_name": self.candidate_id,
            "target_id": targets,
            "encoder_features": encoder_features,
            "decoder_features": decoder_features,
            "aux_features": aux,
            "optimizer_name": "adam",
            "early_stopping_patience": int(
                self.candidate.get("early_stopping_patience", 2)
            ),
            "early_stopping_margin": 0.0,
            "learning_rate": float(self.candidate.get("learning_rate", 1e-4)),
            "max_epochs": int(self.candidate.get("max_epochs", 8)),
            "forecast_horizon": horizon,
            "history_horizon": history,
            "model_class": "recurrent",
            "model_parameters": {
                "recurrent": {
                    "core_net": "torch.nn.LSTM",
                    "core_layers": 1,
                    "dropout_fc": 0.4,
                    "dropout_core": 0.3,
                    "rel_linear_hidden_size": 1.0,
                    "rel_core_hidden_size": 1.0,
                    "relu_leak": 0.1,
                }
            },
            "batch_size": int(self.candidate.get("batch_size", 512)),
            "exploration": False,
        }
        work_dir = Path(
            self.candidate.get("work_dir", "runs/proloaf/runtime")
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        handler = mh.ModelHandler(
            config=config,
            work_dir=str(work_dir),
            scalers=scaler,
            loss="nllgauss",
            device=device,
        )
        handler.fit(train_data, valid_data, exploration=False)
        self._handler = handler
        self._scaler = scaler
        self._features = {
            "targets": targets,
            "encoder": encoder_features,
            "decoder": decoder_features,
            "aux": aux,
        }
        self._history = history
        self._horizon = horizon
        self._device = device
        self._mode = "proloaf_public_recurrent_nllgauss"

        state_hash = hashlib.sha256()
        for key, value in sorted(handler.model_wrap.model.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            state_hash.update(key.encode("utf-8"))
            state_hash.update(str(tensor.dtype).encode("ascii"))
            state_hash.update(str(tuple(tensor.shape)).encode("ascii"))
            state_hash.update(tensor.numpy().tobytes())
        self.fitted_state = {
            "mode": self._mode,
            "implementation_label": self.implementation_label,
            "repo_identity": self.identity,
            "train_rows": int(len(train_index)),
            "valid_rows": int(len(valid_index)),
            "history_horizon": history,
            "forecast_horizon": horizon,
            "scored_delivery_day_via_timestamp_lookup": True,
            "feedback_lag_days": 2,
            "loss": "nllgauss",
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
        if self._mode != "proloaf_public_recurrent_nllgauss":
            raise RuntimeError("ProLoaF adapter not fitted")
        original_forecast = forecast
        actual, forecast = self._model_inputs(actual, forecast)
        import torch
        from proloaf.tensorloader import TimeSeriesData

        StrictTimeSeriesData = _strict_time_series_data_class(TimeSeriesData)

        predict_index = pd.DatetimeIndex(predict_index).sort_values().unique()
        point = empty_like(predict_index)
        q05 = empty_like(predict_index)
        q50 = empty_like(predict_index)
        q95 = empty_like(predict_index)
        eligibility = pd.Series("not_scored", index=predict_index, dtype=object)
        history_gap_rows = 0
        known_forecast_gap_rows = 0
        for day_start in pd.DatetimeIndex(predict_index.normalize().unique()).sort_values():
            history_index, decoder_index = self._lag2_indices(
                day_start, self._history, self._horizon
            )
            requested = predict_index[predict_index.normalize() == day_start]
            if not requested.isin(decoder_index).all():
                raise RuntimeError(f"49-hour decoder misses delivery day {day_start}")
            history_gap_rows += int((~history_index.isin(actual.index)).sum())
            known_forecast_gap_rows += int(
                (~history_index.append(decoder_index).isin(forecast.index)).sum()
            )
            history_frame = self._frame(actual, forecast, history_index)
            decoder_frame = self._decoder_frame(forecast, decoder_index)
            frame = pd.concat([history_frame, decoder_frame])
            scaled = self._scaler.transform(frame)
            dataset = StrictTimeSeriesData(
                scaled,
                history_horizon=self._history,
                forecast_horizon=self._horizon,
                encoder_features=self._features["encoder"],
                decoder_features=self._features["decoder"],
                aux_features=self._features["aux"],
                target_id=self._features["targets"],
                device=self._device,
            )
            # Public TimeSeriesData requires explicit tensor materialization
            # before direct indexing; ModelHandler only does this implicitly
            # for its training loaders.
            dataset.to_tensor()
            sample = dataset[0]
            inputs = [tensor.unsqueeze(0) for tensor in sample[:-1]]
            with torch.no_grad():
                prediction = self._handler.model_wrap.predict(*inputs)
                quantiles = self._handler.model_wrap.loss_metric.get_quantile_prediction(
                    predictions=prediction, quantiles=list(QUANTILES)
                ).values
            array = quantiles.detach().cpu().numpy()
            expected = (1, self._horizon, len(ZONES), len(QUANTILES))
            if array.shape != expected:
                raise RuntimeError(
                    f"unexpected ProLoaF quantile shape {array.shape}; expected {expected}"
                )
            positions = decoder_index.get_indexer(requested)
            if (positions < 0).any():
                raise RuntimeError(
                    f"ProLoaF decoder does not cover delivery day {day_start}"
                )
            for zone_i, zone in enumerate(ZONES):
                values = pd.DataFrame(
                    array[0, :, zone_i, :], columns=["q05", "q50", "q95"]
                )
                values = self._scaler.manual_inverse_transform(
                    values, scale_as=f"target_{zone}"
                )
                selected = values.to_numpy(float)[positions]
                q05.loc[requested, zone] = selected[:, 0]
                q50.loc[requested, zone] = selected[:, 1]
                q95.loc[requested, zone] = selected[:, 2]
                point.loc[requested, zone] = selected[:, 1]
            eligibility.loc[requested] = "eligible"
        scored = eligibility.eq("eligible")
        if point.loc[scored].isna().any().any():
            raise RuntimeError("ProLoaF produced missing eligible predictions")
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
                "repo_identity": self.identity,
                "dst_rule": "utc_contiguous_49h_decoder_scored_by_local_delivery_timestamp",
                "history_gap_rows_filled_from_known_p7": history_gap_rows,
                "known_p7_gap_rows_time_interpolated": known_forecast_gap_rows,
                "gap_fill_scope": "connector_only_no_outcome_interpolation_one_step_endpoint_p7_padding",
                "model_hash": self.model_hash(),
                "config_hash": self.config_hash,
            },
        ), original_forecast)

    def save_checkpoint(self, path: str) -> None:
        if self._mode != "proloaf_public_recurrent_nllgauss":
            raise RuntimeError("cannot save an unfitted ProLoaF checkpoint")
        self._handler.save_current_model(path)
