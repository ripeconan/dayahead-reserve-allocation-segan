"""Convert sealed R85 DeepAR predictions to the exact R78 common-UQ schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL = "deepar_fair_p7_residual"
BASE = "nyiso_p7_official"
LOCAL_TZ = "America/New_York"
EXPECTED_ORIGINS = {f"{year - 1}->{year}" for year in range(2019, 2026)}
KEY = ["origin", "_timestamp_utc", "zone"]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sealed_predictions(root: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames = []
    sources = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        prediction = directory / "predictions.csv.gz"
        seal_path = prediction.with_suffix(prediction.suffix + ".seal.json")
        if not prediction.is_file() or not seal_path.is_file():
            continue
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        observed = sha256_path(prediction)
        if observed != seal.get("sha256"):
            raise RuntimeError(f"prediction seal mismatch: {prediction}")
        frame = pd.read_csv(prediction)
        if len(frame) != int(seal["rows"]):
            raise RuntimeError("sealed prediction row count mismatch")
        frames.append(frame)
        sources.append(
            {
                "origin": str(frame["origin"].iloc[0]),
                "path": str(prediction.resolve()),
                "bytes": prediction.stat().st_size,
                "sha256": observed,
            }
        )
    if len(frames) != 7:
        raise RuntimeError(f"expected seven sealed prediction origins, got {len(frames)}")
    result = pd.concat(frames, ignore_index=True)
    if set(result["origin"].astype(str)) != EXPECTED_ORIGINS:
        raise RuntimeError("prediction origins differ from frozen cohort")
    result["_timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True, errors="raise")
    if result.duplicated(KEY).any():
        raise RuntimeError("duplicate DeepAR prediction keys")
    if not np.isfinite(result[["q05", "q50", "q95"]].to_numpy(float)).all():
        raise RuntimeError("non-finite DeepAR prediction")
    local_date = result["_timestamp_utc"].dt.tz_convert(LOCAL_TZ).dt.date.astype(str)
    if not local_date.equals(result["delivery_date_local"].astype(str)):
        raise RuntimeError("DeepAR stored delivery date differs from local timestamp")
    return result, sources


def convert(
    prediction_root: Path,
    base_path: Path,
    selected_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError("common-input output directory must not exist")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected.get("schema") != "card10_r85_deepar_fair_selected_v1":
        raise RuntimeError("selected contract schema mismatch")
    candidate = selected["selected_candidate"]
    if candidate.get("candidate_id") != "resid_c336_h40" or candidate.get("target_mode") != "p7_residual":
        raise RuntimeError("selected DeepAR identity drift")
    deep, deep_sources = load_sealed_predictions(prediction_root)
    base_all = pd.read_csv(base_path)
    required = {
        "model",
        "candidate_id",
        "origin",
        "timestamp",
        "delivery_date",
        "zone",
        "actual",
        "point",
        "eligibility",
        "model_hash",
        "config_hash",
    }
    if not required.issubset(base_all.columns):
        raise RuntimeError("base point file lacks common-UQ columns")
    base = base_all[base_all["model"].astype(str) == BASE].copy()
    if set(base["origin"].astype(str)) != EXPECTED_ORIGINS:
        raise RuntimeError("P7 base origin cohort mismatch")
    base["_timestamp_utc"] = pd.to_datetime(base["timestamp"], utc=True, errors="raise")
    if base.duplicated(KEY).any():
        duplicate = base.loc[
            base.duplicated(KEY, keep=False),
            ["origin", "timestamp", "zone", "candidate_id", "eligibility"],
        ].head(20)
        raise RuntimeError(
            "duplicate P7 base keys: " + json.dumps(duplicate.to_dict("records"))
        )
    deep_keys = pd.MultiIndex.from_frame(deep[KEY])
    base_keys = pd.MultiIndex.from_frame(base[KEY])
    missing_keys = set(base_keys) - set(deep_keys)
    extra_keys = set(deep_keys) - set(base_keys)
    if extra_keys:
        raise RuntimeError(f"DeepAR contains {len(extra_keys)} keys outside P7 support")
    support_note: dict[str, object]
    if missing_keys:
        # Require enough contiguous history to construct the fixed context window.
        # The known 2020 complete-case gap therefore makes the first 22 local
        # delivery days after that gap ineligible.  Permit only this exact,
        # outcome-free support restriction and apply it symmetrically to P7.
        missing_mask = pd.MultiIndex.from_frame(base[KEY]).isin(missing_keys)
        missing_rows = base.loc[missing_mask].copy()
        missing_rows["_local_date"] = missing_rows["_timestamp_utc"].dt.tz_convert(LOCAL_TZ).dt.date.astype(str)
        date_counts = missing_rows.groupby(["origin", "_local_date"]).size()
        if (
            len(missing_keys) != 5808
            or set(missing_rows["origin"].astype(str)) != {"2020->2021"}
            or len(date_counts) != 22
            or not date_counts.eq(24 * 11).all()
        ):
            raise RuntimeError(
                "unexpected DeepAR/P7 support mismatch "
                f"missing={len(missing_keys)} origins={sorted(set(missing_rows['origin'].astype(str)))} "
                f"dates={len(date_counts)}"
            )
        support_note = {
            "status": "COMMON_REDUCED_SUPPORT_KNOWN_2020_GAP",
            "dropped_p7_zone_hours": int(len(missing_keys)),
            "dropped_local_days": int(len(date_counts)),
            "origin": "2020->2021",
            "local_dates": sorted(missing_rows["_local_date"].unique().tolist()),
            "reason": "frozen DeepAR minimum 500-hour contiguous history after known segment gap",
        }
        keep = pd.MultiIndex.from_frame(base[KEY]).isin(set(deep_keys))
        base = base.loc[keep].copy()
        base_keys = pd.MultiIndex.from_frame(base[KEY])
    else:
        support_note = {"status": "EXACT_P7_SUPPORT", "dropped_p7_zone_hours": 0}
    if set(deep_keys) != set(base_keys) or len(deep_keys) != len(base_keys):
        raise RuntimeError("symmetric reduced-support construction failed")
    lookup = deep.set_index(KEY).sort_index()
    base = base.sort_values(KEY).reset_index(drop=True)
    aligned = lookup.reindex(pd.MultiIndex.from_frame(base[KEY]))
    if aligned[["q05", "q50", "q95"]].isna().any().any():
        raise RuntimeError("DeepAR alignment introduced missing values")
    selected_hash = sha256_path(selected_path)
    point = base.drop(columns="_timestamp_utc").copy()
    point["model"] = MODEL
    point["candidate_id"] = str(candidate["candidate_id"])
    point["point"] = aligned["q50"].to_numpy(float)
    point["eligibility"] = "eligible"
    point["model_hash"] = selected_hash
    point["config_hash"] = str(selected["candidate_grid_file_sha256"])
    output_columns = [
        "model",
        "candidate_id",
        "origin",
        "timestamp",
        "delivery_date",
        "zone",
        "actual",
        "point",
        "eligibility",
        "model_hash",
        "config_hash",
    ]
    point = point[output_columns].sort_values(["model", "origin", "timestamp", "zone"])
    native = point[["model", "candidate_id", "origin", "timestamp", "delivery_date", "zone", "actual"]].copy()
    native["q05"] = aligned["q05"].to_numpy(float)
    native["q50"] = aligned["q50"].to_numpy(float)
    native["q95"] = aligned["q95"].to_numpy(float)
    combined = pd.concat([base.drop(columns="_timestamp_utc"), point], ignore_index=True, sort=False)
    if combined.duplicated(["model", "origin", "timestamp", "zone"]).any():
        raise RuntimeError("combined common input has duplicate keys")
    output_dir.mkdir(parents=True, exist_ok=False)
    point_path = output_dir / "point_predictions.csv.gz"
    native_path = output_dir / "native_quantile_predictions.csv.gz"
    combined_path = output_dir / "combined_point_predictions.csv.gz"
    for frame, path in ((point, point_path), (native, native_path), (combined, combined_path)):
        frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    manifest: dict[str, object] = {
        "schema": "card10_r85_deepar_fair_common_input_v1",
        "model": MODEL,
        "candidate_id": candidate["candidate_id"],
        "origins": sorted(EXPECTED_ORIGINS),
        "rows_model": int(len(point)),
        "rows_combined": int(len(combined)),
        "selected_contract": {
            "path": str(selected_path.resolve()),
            "sha256": selected_hash,
        },
        "base": {
            "path": str(base_path.resolve()),
            "sha256": sha256_path(base_path),
        },
        "prediction_sources": deep_sources,
        "support": support_note,
        "outputs": {
            path.name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
            for path in (point_path, native_path, combined_path)
        },
        "access_flags": selected["access_flags"],
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = convert(
        args.prediction_root.resolve(),
        args.base.resolve(),
        args.selected.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps({"status": "PASS", "rows": result["rows_model"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
