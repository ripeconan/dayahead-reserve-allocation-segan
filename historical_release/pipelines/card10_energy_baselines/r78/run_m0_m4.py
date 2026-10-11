#!/usr/bin/env python3
"""Run R78 M0-M4 selection + seven-origin evaluation (exposed-development only)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from card10_energy_baselines.r78.constants import (  # noqa: E402
    ACCESS_FLAGS,
    OUTPUT_REL,
    PRIMARY_RAW_SHA256,
    STRATUM_PRIMARY,
    ZONES,
)
from card10_energy_baselines.r78.contracts_io import (  # noqa: E402
    load_candidate_grid,
    load_model_registry,
    load_split_contract,
)
from card10_energy_baselines.r78.data_loader import (  # noqa: E402
    input_provenance,
    load_nyiso_raw,
)
from card10_energy_baselines.r78.hash_util import sha256_file, sha256_json  # noqa: E402
from card10_energy_baselines.r78.metrics.io_util import predictions_to_long  # noqa: E402
from card10_energy_baselines.r78.metrics.native_uq import native_uq_metrics  # noqa: E402
from card10_energy_baselines.r78.metrics.point import point_metrics  # noqa: E402
from card10_energy_baselines.r78.models import ADAPTERS  # noqa: E402
from card10_energy_baselines.r78.selection import (  # noqa: E402
    select_candidates,
    write_selected_candidates,
)
from card10_energy_baselines.r78.splits import (  # noqa: E402
    evaluation_year_mask,
    primary_origins,
    refit_mask_before_test_year,
    split_contract_summary,
)


def env_record() -> dict[str, Any]:
    rec: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cwd": str(Path.cwd()),
        "thread_env": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import lightgbm as lgb

        rec["lightgbm"] = lgb.__version__
    except Exception as exc:  # noqa: BLE001
        rec["lightgbm"] = f"unavailable:{exc}"
    try:
        import catboost

        rec["catboost"] = catboost.__version__
    except Exception as exc:  # noqa: BLE001
        rec["catboost"] = f"unavailable:{exc}"
    try:
        import sklearn

        rec["sklearn"] = sklearn.__version__
    except Exception as exc:  # noqa: BLE001
        rec["sklearn"] = f"unavailable:{exc}"
    try:
        import torch

        rec["torch"] = torch.__version__
        rec["cuda_available"] = bool(torch.cuda.is_available())
        rec["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except Exception as exc:  # noqa: BLE001
        rec["torch"] = f"unavailable:{exc}"
    try:
        import pytorch_forecasting

        rec["pytorch_forecasting"] = pytorch_forecasting.__version__
    except Exception as exc:  # noqa: BLE001
        rec["pytorch_forecasting"] = f"unavailable:{exc}"
    return rec


def candidate_for(name: str, selected: dict[str, Any], grid: dict[str, Any]) -> dict[str, Any]:
    model_sel = selected["models"][name]
    cid = model_sel["selected_candidate_id"]
    for cand in grid["models"][name]["candidates"]:
        if cand["candidate_id"] == cid:
            return cand
    raise KeyError(f"selected candidate not found for {name}: {cid}")


def run_model_origin(
    name: str,
    cand: dict[str, Any],
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    origin_label: str,
    test_year: int,
    dev_abs: np.ndarray,
    upper_tail_thr: float,
) -> dict[str, Any]:
    t0 = time.time()
    adapter = ADAPTERS[name](cand)
    train_mask = refit_mask_before_test_year(actual.index, test_year)
    train_idx = actual.index[train_mask.to_numpy()]
    # Validation: last 90 days before test year within train.
    if len(train_idx) > 24 * 100:
        valid_idx = train_idx[-24 * 90 :]
        fit_idx = train_idx[: -24 * 90]
    else:
        valid_idx = train_idx
        fit_idx = train_idx
    adapter.fit(actual, forecast, fit_idx, valid_index=valid_idx)
    test_mask = evaluation_year_mask(actual.index, test_year)
    test_idx = actual.index[test_mask.to_numpy()]
    pred = adapter.predict(actual, forecast, test_idx)
    pm = point_metrics(
        actual.loc[test_idx],
        pred.point,
        upper_tail_threshold=upper_tail_thr,
        development_abs_errors=dev_abs,
    )
    out: dict[str, Any] = {
        "model": name,
        "candidate_id": cand["candidate_id"],
        "origin": origin_label,
        "stratum": STRATUM_PRIMARY,
        "status": "PASS",
        "runtime_sec": time.time() - t0,
        "point_metrics": pm,
        "model_hash": adapter.model_hash(),
        "config_hash": adapter.config_hash,
        "prediction": pred,
        "adapter": adapter,
        "fitted_state": adapter.fitted_state,
    }
    if pred.q05 is not None and pred.q50 is not None and pred.q95 is not None:
        out["native_uq_metrics"] = native_uq_metrics(
            actual.loc[test_idx], pred.q05, pred.q50, pred.q95
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="R78 M0-M4 runner")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / OUTPUT_REL),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "seasonal_naive_d7",
            "nyiso_p7_official",
            "ridge_lag2",
            "lightgbm_energy",
            "catboost_energy_quantile",
        ],
    )
    parser.add_argument(
        "--origins",
        nargs="+",
        default=None,
        help="Optional subset like 2018->2019",
    )
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument(
        "--selected-json",
        type=str,
        default=None,
        help="Use an existing selected_candidates.json",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commands_log = []
    commands_log.append(
        {
            "argv": sys.argv,
            "start_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    panel = load_nyiso_raw(args.data)
    actual, forecast = panel.actual, panel.forecast
    provenance = input_provenance(panel)
    env = env_record()
    grid = load_candidate_grid()

    with (out_dir / "input_provenance.json").open("w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (out_dir / "environment.json").open("w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (out_dir / "split_and_information_contract.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {**load_split_contract(), **{"runtime_summary": split_contract_summary()}},
            fh,
            indent=2,
            sort_keys=True,
        )
        fh.write("\n")
    with (out_dir / "candidate_grid_v1.json").open("w", encoding="utf-8") as fh:
        json.dump(grid, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (out_dir / "model_registry.json").open("w", encoding="utf-8") as fh:
        json.dump(load_model_registry(), fh, indent=2, sort_keys=True)
        fh.write("\n")

    selected_path = Path(args.selected_json) if args.selected_json else out_dir / "selected_candidates.json"
    if args.skip_selection and selected_path.is_file():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
    else:
        t_sel = time.time()
        selected = select_candidates(actual, forecast, models=list(args.models))
        selected["selection_runtime_sec"] = time.time() - t_sel
        write_selected_candidates(selected_path, selected)

    # Dev abs errors for upper tail threshold (frozen).
    from card10_energy_baselines.r78.splits import development_mask

    dev_mask = development_mask(actual.index)
    dev_idx = actual.index[dev_mask.to_numpy()]
    dev_abs = np.mean(
        np.abs(
            actual.loc[dev_idx, list(ZONES)].to_numpy(float)
            - forecast.loc[dev_idx, list(ZONES)].to_numpy(float)
        ),
        axis=1,
    )
    upper_tail_thr = float(selected.get("upper_tail_threshold_dev", np.quantile(dev_abs, 0.99)))

    origins = primary_origins()
    if args.origins:
        # ``2018_to_2019`` is a shell-safe spelling for remote launchers;
        # normalize it to the frozen human-readable origin label.
        wanted = {value.replace("_to_", "->") for value in args.origins}
        origins = tuple(o for o in origins if o.label in wanted)

    point_rows = []
    native_rows = []
    pred_parts = []
    native_pred_parts = []
    runtime_rows = []
    blockers = []

    for origin in origins:
        for name in args.models:
            cell = f"{name}|{origin.label}"
            print(f"[R78] running {cell}", flush=True)
            try:
                if selected["models"].get(name, {}).get("status") != "SELECTED":
                    raise RuntimeError(f"no selected candidate for {name}")
                cand = candidate_for(name, selected, grid)
                result = run_model_origin(
                    name,
                    cand,
                    actual,
                    forecast,
                    origin.label,
                    origin.test_year,
                    dev_abs,
                    upper_tail_thr,
                )
                pm = result["point_metrics"]
                point_rows.append(
                    {
                        "model": name,
                        "candidate_id": result["candidate_id"],
                        "origin": origin.label,
                        "stratum": STRATUM_PRIMARY,
                        "status": "PASS",
                        **{k: pm[k] for k in pm},
                        "runtime_sec": result["runtime_sec"],
                        "model_hash": result["model_hash"],
                        "config_hash": result["config_hash"],
                    }
                )
                if "native_uq_metrics" in result:
                    nq = result["native_uq_metrics"]
                    native_rows.append(
                        {
                            "model": name,
                            "candidate_id": result["candidate_id"],
                            "origin": origin.label,
                            "stratum": STRATUM_PRIMARY,
                            "status": "PASS",
                            **nq,
                            "runtime_sec": result["runtime_sec"],
                        }
                    )
                pred = result["prediction"]
                checkpoint_record: dict[str, Any] = {}
                if hasattr(result["adapter"], "save_checkpoint"):
                    checkpoint_dir = out_dir / "checkpoints"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = checkpoint_dir / (
                        f"{name}_{origin.label.replace('->', '_to_')}.pt"
                    )
                    result["adapter"].save_checkpoint(str(checkpoint_path))
                    checkpoint_record = {
                        "artifact": str(checkpoint_path.name),
                        "artifact_sha256": sha256_file(checkpoint_path),
                        "cuda_peak_memory_bytes": result["fitted_state"].get(
                            "cuda_peak_memory_bytes"
                        ),
                        "state_dict_sha256": result["fitted_state"].get(
                            "state_dict_sha256"
                        ),
                    }
                long_df = predictions_to_long(
                    model=name,
                    candidate_id=result["candidate_id"],
                    origin=origin.label,
                    stratum=STRATUM_PRIMARY,
                    actual=actual,
                    result=pred,
                    model_hash=result["model_hash"],
                    config_hash=result["config_hash"],
                )
                if len(long_df):
                    pred_parts.append(long_df)
                    if "q05" in long_df.columns:
                        native_pred_parts.append(long_df)
                runtime_rows.append(
                    {
                        "model": name,
                        "origin": origin.label,
                        "runtime_sec": result["runtime_sec"],
                        "status": "PASS",
                        **checkpoint_record,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                print(f"[R78] FAIL {cell}: {exc}", flush=True)
                blockers.append(
                    {
                        "model": name,
                        "origin": origin.label,
                        "status": "FAIL",
                        "error": str(exc),
                        "traceback": tb,
                    }
                )
                point_rows.append(
                    {
                        "model": name,
                        "origin": origin.label,
                        "stratum": STRATUM_PRIMARY,
                        "status": "FAIL",
                        "error": str(exc),
                    }
                )
                runtime_rows.append(
                    {
                        "model": name,
                        "origin": origin.label,
                        "runtime_sec": None,
                        "status": "FAIL",
                        "error": str(exc),
                    }
                )

    point_df = pd.DataFrame(point_rows)
    point_df.to_csv(out_dir / "point_metrics_per_origin.csv", index=False)
    # Aggregate macro mean over successful origins.
    agg_rows = []
    for name, sub in point_df[point_df.get("status", "PASS") == "PASS"].groupby("model"):
        num_cols = [
            c
            for c in sub.columns
            if c
            not in {
                "model",
                "candidate_id",
                "origin",
                "stratum",
                "status",
                "model_hash",
                "config_hash",
                "error",
            }
            and pd.api.types.is_numeric_dtype(sub[c])
        ]
        row = {
            "model": name,
            "n_origins": int(sub["origin"].nunique()),
            "stratum": STRATUM_PRIMARY,
            "status": "PASS",
        }
        for c in num_cols:
            row[c] = float(sub[c].mean())
        if "candidate_id" in sub.columns:
            row["candidate_id"] = str(sub["candidate_id"].iloc[0])
        agg_rows.append(row)
    pd.DataFrame(agg_rows).to_csv(out_dir / "point_metrics_aggregate.csv", index=False)

    if native_rows:
        native_df = pd.DataFrame(native_rows)
        native_df.to_csv(out_dir / "native_uq_metrics_per_origin.csv", index=False)
        nagg = []
        for name, sub in native_df.groupby("model"):
            row = {
                "model": name,
                "n_origins": int(sub["origin"].nunique()),
                "stratum": STRATUM_PRIMARY,
                "estimand": "native_rectangular_quantile",
                "not_oas_ellipsoid": True,
            }
            for c in sub.columns:
                if c in {"model", "candidate_id", "origin", "stratum", "status", "estimand", "ranking_namespace", "crps_status"}:
                    continue
                if pd.api.types.is_numeric_dtype(sub[c]):
                    row[c] = float(sub[c].mean())
            nagg.append(row)
        pd.DataFrame(nagg).to_csv(out_dir / "native_uq_metrics_aggregate.csv", index=False)

    if pred_parts:
        all_pred = pd.concat(pred_parts, ignore_index=True)
        all_pred.to_csv(out_dir / "point_predictions.csv.gz", index=False, compression="gzip")
    if native_pred_parts:
        pd.concat(native_pred_parts, ignore_index=True).to_csv(
            out_dir / "native_quantile_predictions.csv.gz", index=False, compression="gzip"
        )

    pd.DataFrame(runtime_rows).to_csv(out_dir / "runtime_memory.csv", index=False)
    applicability = {
        "blockers": blockers,
        "access_flags": dict(ACCESS_FLAGS),
        "models_requested": list(args.models),
        "origins_requested": [o.label for o in origins],
        "primary_raw_sha256": provenance["primary_raw_sha256"],
        "expected_primary_raw_sha256": PRIMARY_RAW_SHA256,
        "confirmatory_2026_outcome_accessed": False,
        "confirmatory_2026_forecast_payload_accessed": False,
        "reserved_confirmatory_path_accessed": False,
        "r77_d2_values_accessed": False,
        "sota_claimed": False,
    }
    with (out_dir / "applicability_and_blockers.json").open("w", encoding="utf-8") as fh:
        json.dump(applicability, fh, indent=2, sort_keys=True)
        fh.write("\n")

    commands_log.append({"end_utc": datetime.now(timezone.utc).isoformat(), "n_blockers": len(blockers)})
    with (out_dir / "commands.log").open("w", encoding="utf-8") as fh:
        json.dump(commands_log, fh, indent=2)
        fh.write("\n")

    # Compact stage-2 binding for external artifacts (none large yet).
    binding = {
        "output_dir": str(out_dir),
        "files": {},
    }
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            binding["files"][path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    with (out_dir / "external_artifact_binding.json").open("w", encoding="utf-8") as fh:
        json.dump(binding, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(json.dumps({"status": "done", "blockers": len(blockers), "out": str(out_dir)}, indent=2))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
