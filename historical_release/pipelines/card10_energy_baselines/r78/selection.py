"""Mechanical 2017-2018 candidate selection (no evaluation-year leakage)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import STRATUM_PRIMARY, ZONES
from .contracts_io import load_candidate_grid
from .features import build_primary_features
from .metrics.native_uq import native_uq_metrics
from .metrics.point import point_metrics
from .models import ADAPTERS
from .splits import (
    assert_no_test_year_in_selection,
    development_mask,
    early_stopping_validation_mask,
)


def _dev_frames(actual: pd.DataFrame, forecast: pd.DataFrame):
    dev = development_mask(actual.index, STRATUM_PRIMARY)
    idx = actual.index[dev.to_numpy()]
    assert_no_test_year_in_selection(idx)
    valid_mask = early_stopping_validation_mask(actual.index)
    valid_idx = actual.index[valid_mask.to_numpy()].intersection(idx)
    train_idx = idx.difference(valid_idx)
    return train_idx, valid_idx, idx


def select_candidates(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Select one candidate per model on 2017-2018 only."""
    grid = load_candidate_grid()
    train_idx, valid_idx, all_dev_idx = _dev_frames(actual, forecast)
    model_names = models or [
        "seasonal_naive_d7",
        "nyiso_p7_official",
        "ridge_lag2",
        "lightgbm_energy",
        "catboost_energy_quantile",
    ]
    selected: dict[str, Any] = {
        "selection_period": {"start": "2017-01-01", "end": "2018-12-31"},
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "models": {},
    }

    # Development absolute errors for upper-tail threshold from P-7 baseline.
    p7_err = np.abs(
        actual.loc[all_dev_idx, list(ZONES)].to_numpy(float)
        - forecast.loc[all_dev_idx, list(ZONES)].to_numpy(float)
    )
    dev_abs = np.mean(p7_err, axis=1)
    selected["upper_tail_threshold_dev"] = float(np.quantile(dev_abs[np.isfinite(dev_abs)], 0.99))

    for name in model_names:
        if name not in ADAPTERS:
            selected["models"][name] = {
                "status": "NO_ADAPTER",
                "selected_candidate_id": None,
            }
            continue
        spec = grid["models"][name]
        candidates = list(spec.get("candidates") or [])
        if not candidates:
            selected["models"][name] = {
                "status": "NO_CANDIDATES",
                "selected_candidate_id": None,
            }
            continue

        scored = []
        for cand in candidates:
            adapter = ADAPTERS[name](cand)
            adapter.fit(actual, forecast, train_idx, valid_index=valid_idx)
            pred = adapter.predict(actual, forecast, all_dev_idx)
            pm = point_metrics(
                actual.loc[all_dev_idx],
                pred.point,
                development_abs_errors=dev_abs,
            )
            entry = {
                "candidate_id": cand["candidate_id"],
                "mae": pm["mae"],
                "rmse": pm["rmse"],
                "wape": pm["wape"],
                "n_eligible": pm["n_eligible"],
                "config_hash": adapter.config_hash,
                "model_hash": adapter.model_hash(),
            }
            if pred.q05 is not None and pred.q95 is not None and pred.q50 is not None:
                nq = native_uq_metrics(
                    actual.loc[all_dev_idx], pred.q05, pred.q50, pred.q95
                )
                entry["mean_pinball"] = nq.get("mean_pinball")
            scored.append(entry)

        # Tie-break: lower MAE, then RMSE, then candidate_id.
        if name == "catboost_energy_quantile":
            scored.sort(
                key=lambda r: (
                    r.get("mean_pinball") if r.get("mean_pinball") is not None else 1e18,
                    r["mae"] if np.isfinite(r["mae"]) else 1e18,
                    r["rmse"] if np.isfinite(r["rmse"]) else 1e18,
                    r["candidate_id"],
                )
            )
        else:
            scored.sort(
                key=lambda r: (
                    r["mae"] if np.isfinite(r["mae"]) else 1e18,
                    r["rmse"] if np.isfinite(r["rmse"]) else 1e18,
                    r["candidate_id"],
                )
            )
        best = scored[0]
        selected["models"][name] = {
            "status": "SELECTED",
            "selected_candidate_id": best["candidate_id"],
            "selected": best,
            "all_candidates": scored,
            "selection_rule": spec.get("selection"),
        }
    return selected


def write_selected_candidates(path: Path, selected: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2, sort_keys=True)
        handle.write("\n")
