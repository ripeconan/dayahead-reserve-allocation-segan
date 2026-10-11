"""Bind sealed R86 winner predictions to the exact R78 common-UQ schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE = "nyiso_p7_official"
MODELS = ("TFT", "NHiTS", "PatchTST", "ProLoaF")
EXPECTED_ORIGINS = {f"{year - 1}->{year}" for year in range(2019, 2026)}
KEY = ["origin", "_timestamp_utc", "zone"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(root: Path, model: str, selected_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selection = selected["models"][model]
    candidate_id = selection["selected_candidate_id"]
    target_mode = selection["selected"]["target_mode"]
    output_label = f"{model.lower()}_r86_fair_{target_mode}"
    frames = []
    sources = []
    for year in range(2019, 2026):
        origin_dir = root / model / f"{year - 1}_to_{year}"
        result_path = origin_dir / "result.json"
        prediction_path = origin_dir / "predictions.csv.gz"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS" or result.get("candidate_id") != candidate_id:
            raise RuntimeError(f"unsealed/mismatched R86 cell: {model} {year}")
        if result.get("selected_contract_sha256") != sha256_file(selected_path):
            raise RuntimeError(f"selected-contract binding mismatch: {model} {year}")
        if result.get("prediction_sha256") != sha256_file(prediction_path):
            raise RuntimeError(f"prediction hash mismatch: {model} {year}")
        frame = pd.read_csv(prediction_path)
        if set(frame["origin"].astype(str)) != {f"{year - 1}->{year}"}:
            raise RuntimeError(f"origin identity mismatch: {model} {year}")
        if set(frame["candidate_id"].astype(str)) != {candidate_id}:
            raise RuntimeError(f"candidate identity mismatch: {model} {year}")
        frame["model"] = output_label
        frame["model_hash"] = result["model_hash"]
        frame["config_hash"] = result["config_hash"]
        frames.append(frame)
        sources.append(
            {
                "origin": result["origin"],
                "result_sha256": sha256_file(result_path),
                "prediction_sha256": result["prediction_sha256"],
                "checkpoint_sha256": result["checkpoint_sha256"],
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["_timestamp_utc"] = pd.to_datetime(
        combined["timestamp_utc"], utc=True, errors="raise"
    )
    if set(combined["origin"].astype(str)) != EXPECTED_ORIGINS:
        raise RuntimeError(f"incomplete origin cohort for {model}")
    if combined.duplicated(KEY).any():
        raise RuntimeError(f"duplicate prediction keys for {model}")
    if not np.isfinite(
        combined[["actual", "point", "q05", "q50", "q95"]].to_numpy(float)
    ).all():
        raise RuntimeError(f"non-finite prediction values for {model}")
    return combined, {
        "source_model": model,
        "output_model": output_label,
        "candidate_id": candidate_id,
        "target_mode": target_mode,
        "selected_contract_sha256": sha256_file(selected_path),
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create R86 common-UQ input")
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("R86 common-input output directory must not exist")

    base_all = pd.read_csv(args.base)
    base = base_all[base_all["model"].astype(str) == BASE].copy()
    base["_timestamp_utc"] = pd.to_datetime(base["timestamp"], utc=True, errors="raise")
    if set(base["origin"].astype(str)) != EXPECTED_ORIGINS or base.duplicated(KEY).any():
        raise RuntimeError("R78 P7 base cohort/key mismatch")
    base_keys = pd.MultiIndex.from_frame(base[KEY])

    model_frames = []
    bindings = []
    native_parts = []
    for model in MODELS:
        selected_path = args.selection_root / model / "fair_deep_selected_v1.json"
        frame, binding = load_model(args.evaluation_root, model, selected_path)
        model_keys = pd.MultiIndex.from_frame(frame[KEY])
        if set(model_keys) != set(base_keys) or len(model_keys) != len(base_keys):
            raise RuntimeError(f"R86/P7 common support mismatch for {model}")
        aligned = frame.set_index(KEY).reindex(base_keys)
        if not np.allclose(
            aligned["actual"].to_numpy(float), base["actual"].to_numpy(float), rtol=0, atol=1e-9
        ):
            raise RuntimeError(f"actual mismatch after common alignment for {model}")
        point = base.drop(columns="_timestamp_utc").copy()
        point["model"] = binding["output_model"]
        point["candidate_id"] = binding["candidate_id"]
        point["point"] = aligned["point"].to_numpy(float)
        point["eligibility"] = "eligible"
        point["model_hash"] = aligned["model_hash"].astype(str).to_numpy()
        point["config_hash"] = aligned["config_hash"].astype(str).to_numpy()
        model_frames.append(point)
        native = point[
            ["model", "candidate_id", "origin", "timestamp", "delivery_date", "zone", "actual"]
        ].copy()
        for quantile in ("q05", "q50", "q95"):
            native[quantile] = aligned[quantile].to_numpy(float)
        native_parts.append(native)
        bindings.append(binding)

    combined_point = pd.concat(
        [base.drop(columns="_timestamp_utc"), *model_frames], ignore_index=True, sort=False
    )
    model_point = pd.concat(model_frames, ignore_index=True, sort=False)
    native_all = pd.concat(native_parts, ignore_index=True, sort=False)
    if combined_point.duplicated(["model", "origin", "timestamp", "zone"]).any():
        raise RuntimeError("duplicate combined common-UQ keys")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "combined_point_predictions.csv.gz": combined_point,
        "r86_point_predictions.csv.gz": model_point,
        "r86_native_quantile_predictions.csv.gz": native_all,
    }
    output_bindings = {}
    for name, frame in files.items():
        path = args.output_dir / name
        frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
        output_bindings[name] = {
            "rows": int(len(frame)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema": "card10_r86_fair_deep_common_input_v1",
        "status": "PASS_EXACT_P7_SUPPORT",
        "origins": sorted(EXPECTED_ORIGINS),
        "base": {"path": str(args.base.resolve()), "sha256": sha256_file(args.base)},
        "models": bindings,
        "outputs": output_bindings,
        "access_flags": {
            "confirmatory_2026_outcome_accessed": False,
            "confirmatory_2026_forecast_payload_accessed": False,
            "reserved_confirmatory_path_accessed": False,
            "r77_d2_values_accessed": False,
            "sota_claimed": False,
        },
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "models": bindings}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
