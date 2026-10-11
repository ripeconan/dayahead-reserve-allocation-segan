"""R-A upper-direction audit of stored joint-region misses (NYISO, exposed).

Quantity semantics (verified from the producer source, sha256 pinned in the
protocol): each stored row holds, per hour and per direction d (11 zone unit
directions and the equal-weight aggregate direction),

  reserve_<d> = d . center + sqrt(threshold * d' Sigma d)   (upper directional bound)
  actual_<d>  = d . error                                   (realized projection)

where ``error`` is the bias-corrected day-ahead forecast error (actual minus
forecast, ridge-corrected) of the 11-zone load vector, in the NYISO zonal load
units (MW scale).  These are FORECAST-ERROR coordinates, not metered load
levels and not outage magnitudes.

For every hour whose joint region missed (``covered == 0``) we classify:
  * upward exceedance in a zone      : actual_z > reserve_z
  * upper-side miss without breach   : not possible (actual_z <= reserve_z never
    exceeds the bound); marginal misses with actual_z <= reserve_z lie below
    the lower directional bound (since |error_z - center_z| > radius_z), i.e.
    surplus-like for that zone, and are counted separately.
  * joint-geometry-only miss         : every zone marginal covered (|error_z -
    center_z| <= radius_z) while the joint ellipsoid missed (cross-correlation).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import r1_common as C  # noqa: E402

ZONES = ("CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL", "MHK VL",
         "MILLWD", "N.Y.C.", "NORTH", "WEST")


def run(root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    frame = C.read_gzip_csv(root / C.R39_HOURLY)
    frame = frame[(frame["method"] == "saocp") & (frame["layer"] == "base_no_fallback")].copy()
    for zone in ZONES:
        for prefix in ("covered_", "reserve_", "actual_"):
            if f"{prefix}{zone}" not in frame.columns:
                raise RuntimeError(f"missing column {prefix}{zone}")
    covered = frame["covered"].to_numpy(float)
    missed = covered < 0.5
    n_missed = int(missed.sum())
    up_counts = np.zeros(len(frame))
    lower_miss_counts = np.zeros(len(frame))
    max_exceed = np.full(len(frame), np.nan)
    for zone in ZONES:
        cov_z = frame[f"covered_{zone}"].to_numpy(float)
        res_z = frame[f"reserve_{zone}"].to_numpy(float)
        act_z = frame[f"actual_{zone}"].to_numpy(float)
        up = act_z > res_z
        up_counts += up.astype(float)
        lower = (cov_z < 0.5) & (~up)
        lower_miss_counts += lower.astype(float)
        exceed = np.where(up, act_z - res_z, np.nan)
        max_exceed = np.fmax(max_exceed, exceed)
    joint_only = missed & (up_counts == 0) & (lower_miss_counts == 0)
    agg_up = frame["actual_AGGREGATE_EQ"].to_numpy(float) > frame["reserve_AGGREGATE_EQ"].to_numpy(float)

    frame_out = pd.DataFrame({
        "origin": frame["origin"],
        "delivery_date": frame["delivery_date"],
        "hour": frame["hour"],
        "covered": covered,
        "n_zone_up": up_counts,
        "n_zone_lower_miss": lower_miss_counts,
        "max_zone_exceedance": max_exceed,
        "aggregate_direction_up": agg_up.astype(float),
        "joint_geometry_only_miss": joint_only.astype(float),
    })
    C.write_gzip_csv(frame_out, out / "r1_direction_audit_hourly.csv.gz")

    per_origin: list[dict] = []
    for origin, group in frame_out.groupby("origin", sort=True):
        m = group["covered"].to_numpy(float) < 0.5
        rows = {
            "origin": str(origin),
            "hours": int(len(group)),
            "missed_hours": int(m.sum()),
            "miss_rate": float(m.mean()),
            "missed_with_zone_up_share": float(np.mean(group["n_zone_up"].to_numpy(float)[m] >= 1)) if m.any() else None,
            "missed_agg_direction_up_share": float(np.mean(group["aggregate_direction_up"].to_numpy(float)[m] >= 0.5)) if m.any() else None,
            "missed_joint_geometry_only_share": float(np.mean(group["joint_geometry_only_miss"].to_numpy(float)[m] >= 0.5)) if m.any() else None,
            "missed_lower_side_share": float(np.mean((group["n_zone_lower_miss"].to_numpy(float)[m] >= 1)
                                                     & (group["n_zone_up"].to_numpy(float)[m] == 0))) if m.any() else None,
            "median_max_exceedance_among_up": float(np.nanmedian(group["max_zone_exceedance"].to_numpy(float)[m & (group["n_zone_up"].to_numpy(float) >= 1)])) if m.any() else None,
            "mean_zone_up_among_missed": float(np.mean(group["n_zone_up"].to_numpy(float)[m])) if m.any() else None,
            "all_hours_share_with_zone_up": float(np.mean(group["n_zone_up"].to_numpy(float) >= 1)),
        }
        per_origin.append(rows)
    macro = {k: float(np.mean([row[k] for row in per_origin if row[k] is not None]))
             for k in per_origin[0] if k != "origin" and all(row[k] is not None for row in per_origin)}
    report = {
        "stage": "R1-direction-audit",
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "quantity_semantics": {
            "reserve_<d>": "upper directional bound d.center + sqrt(threshold * d' Sigma d)",
            "actual_<d>": "realized projection d.error of the bias-corrected day-ahead forecast error (actual - forecast)",
            "units": "NYISO zonal load units (MW scale); forecast-error coordinates, not metered load levels",
            "positive_error_sign": "actual above day-ahead forecast (deficit/shortage-consumptive direction)",
            "lower_side_detection": "marginal miss with actual_z <= reserve_z implies error_z < center_z - radius_z",
            "producer_source_sha256": C.INPUT_HASHES.get("scripts/card10_killtest/nyiso_confirmatory_audit.py"),
        },
        "n_hours": int(len(frame_out)),
        "n_missed_hours": n_missed,
        "miss_rate": float(missed.mean()),
        "missed_with_zone_up_share": float(np.mean(up_counts[missed] >= 1)) if n_missed else None,
        "missed_agg_direction_up_share": float(np.mean(agg_up[missed])) if n_missed else None,
        "missed_joint_geometry_only_share": float(np.mean(joint_only[missed])) if n_missed else None,
        "missed_lower_side_only_share": float(np.mean((lower_miss_counts[missed] >= 1) & (up_counts[missed] == 0))) if n_missed else None,
        "median_max_zone_exceedance_among_up": float(np.nanmedian(max_exceed[missed & (up_counts >= 1)])) if n_missed else None,
        "per_origin": per_origin,
        "macro": macro,
    }
    C.write_json(out / "r1_direction_audit.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=C.EVIDENCE_ROOT)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "direction")
    args = parser.parse_args()
    report = run(args.root, args.out)
    keep = {k: v for k, v in report.items() if k not in ("per_origin", "quantity_semantics")}
    print(json.dumps(keep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
