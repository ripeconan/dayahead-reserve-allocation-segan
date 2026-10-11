"""R-A equal-cap comparator replay suite (per origin, resumable).

Runs, for every exposed origin of both cohort families:
  P  production DRB (must equal stored actions; Stage 0 already gates this)
  F1 pace7, F2 paceBank, F3 k2_count, F4 persist1, F5 bank1_causal
on the identical immutable q_t path and segment structure with the identical
horizon cap round(0.15 * n_days).  No outcome information beyond the cutoff-
eligible rule definitions is used by any policy; no parameter is tuned.

Per-origin outputs are resumable: a fingerprint sidecar records input/code
hashes and completion, and an identical fingerprint causes SKIP.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import r1_common as C  # noqa: E402
import r1_policies as P  # noqa: E402

POLICIES = ("P", "F1", "F2", "F3", "F4", "F5")


def _safe_name(cohort: str, origin: str) -> str:
    """Filesystem-safe output stem (origin labels contain '->')."""
    raw = f"{cohort}__{origin}"
    return "".join(ch if (ch.isalnum() or ch in "_-.") else "_" for ch in raw)


def _fingerprint(root: Path) -> str:
    return C.build_fingerprint(("r1_common.py", "r1_policies.py", "r1_allocation.py"))


def run_origin(origin: str, cohort: str, q: np.ndarray, segments: np.ndarray,
               dates: list, thresholds_hourly: pd.DataFrame | None, thresholds_daily: np.ndarray | None,
               cap: int, stored_action: np.ndarray) -> tuple[dict, pd.DataFrame, dict]:
    n = len(q)
    results: dict[str, dict] = {}
    frame = pd.DataFrame({
        "delivery_date": [pd.Timestamp(d).date().isoformat() for d in dates],
        "segment_id": segments,
        "q": q,
    })
    # P: production
    action_p, trace_p = P.production_drb(q, segments, dates, cap)
    if int(np.sum(action_p != stored_action)) != 0:
        raise RuntimeError(f"{cohort}/{origin}: production reverse replay failed inside runner")
    frame["action_P"] = action_p
    results["P"] = C.policy_metrics(q, action_p, segments)
    results["P"]["blocked_trigger_count"] = int(sum(1 for row in trace_p if row["blocked"]))

    # F1
    action_f1, trace_f1 = P.pace7(q, segments, dates, cap)
    frame["action_F1"] = action_f1
    results["F1"] = C.policy_metrics(q, action_f1, segments)
    results["F1"]["cap_skipped_scheduled_days"] = int(sum(1 for row in trace_f1 if row["cap_skipped"]))
    results["F1"]["all_windows_have_action"] = _all_7_windows_have_action(action_f1, segments)

    # F2
    action_f2, trace_f2 = P.pace_bank(q, segments, dates, cap)
    frame["action_F2"] = action_f2
    results["F2"] = C.policy_metrics(q, action_f2, segments)
    results["F2"]["credit_dropped_on_segment_resets"] = int(sum(row["dropped_on_reset"] for row in trace_f2))

    # F3
    action_f3, trace_f3 = P.k2_thresholded_count(q, segments, dates, cap)
    frame["action_F3"] = action_f3
    results["F3"] = C.policy_metrics(q, action_f3, segments)
    results["F3"]["blocked_trigger_count"] = int(sum(1 for row in trace_f3 if row["blocked"]))

    # F4
    action_f4, trace_f4 = P.persist1(q, segments, dates, cap)
    frame["action_F4"] = action_f4
    results["F4"] = C.policy_metrics(q, action_f4, segments)
    results["F4"]["blocked_trigger_count"] = int(sum(1 for row in trace_f4 if row["used_after"] == cap and row["triggered"]))

    # F5
    if thresholds_hourly is not None:
        tension = P.tension_from_thresholds(thresholds_hourly, dates, segments)
    else:
        tension = P.tension_from_daily_thresholds(thresholds_daily, segments)
    declared_end = C.declared_horizon_end(origin)
    # Explicit shadow recomputation: the F5 shadow mints must be the production
    # closed-loop actions on this exact q path (comparator-owned history is not
    # interchangeable with the shadow history).
    shadow_recomputed, _ = P.production_drb(q, segments, dates, cap)
    if not np.array_equal(shadow_recomputed, action_p):
        raise RuntimeError(f"{cohort}/{origin}: F5 shadow recomputation differs from production path")
    if not np.array_equal(shadow_recomputed, stored_action):
        raise RuntimeError(f"{cohort}/{origin}: F5 shadow differs from stored production actions")
    action_f5, trace_f5, flags = P.bank1_causal(shadow_recomputed, segments, tension, dates, cap,
                                                str(declared_end.date()))
    frame["action_F5"] = action_f5
    results["F5"] = C.policy_metrics(q, action_f5, segments)
    results["F5"].update({k: int(v) if isinstance(v, (int, np.integer)) else v for k, v in flags.items()})

    meta = {"cohort": cohort, "origin": origin, "n_days": n, "cap": cap,
            "declared_horizon_end": str(declared_end.date()),
            "production_action_total": int(action_p.sum())}
    return {"meta": meta, "policies": results}, frame, meta


def _all_7_windows_have_action(actions: np.ndarray, segments: np.ndarray) -> bool:
    for window in C.segment_windows(segments, 7):
        if not actions[window].any():
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=C.EVIDENCE_ROOT)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "allocation_value")
    parser.add_argument("--origins", type=str, default="", help="optional comma-separated origin filter")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(args.root)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    timing: list[dict] = []

    nyiso = C.load_nyiso_daily(args.root)
    h39 = C.read_gzip_csv(args.root / C.R39_HOURLY)
    h39 = h39[(h39["method"] == "saocp") & (h39["layer"] == "base_no_fallback")][
        ["origin", "delivery_date", "hour", "threshold"]]
    h39["delivery_date"] = pd.to_datetime(h39["delivery_date"], errors="raise")
    transfer = C.load_transfer_daily(args.root)

    filter_set = {s for s in args.origins.split(",") if s} if args.origins else None

    jobs: list[tuple[str, str, pd.DataFrame, pd.DataFrame | None, np.ndarray | None]] = []
    for origin, group in nyiso.groupby("origin", sort=True):
        jobs.append(("nyiso_exposed", str(origin), group, h39[h39["origin"] == origin], None))
    for origin, group in transfer.groupby("origin", sort=True):
        jobs.append((str(group["cohort"].iloc[0]), str(origin), group, None, group["threshold"].to_numpy(float)))

    for cohort, origin, group, thr_h, thr_d in jobs:
        if filter_set is not None and origin not in filter_set:
            continue
        group = group.sort_values("delivery_date", kind="mergesort").reset_index(drop=True)
        safe = _safe_name(cohort, origin)
        result_path = out / f"{safe}.json"
        trace_path = out / f"{safe}__daily.csv.gz"
        fp_path = out / f"{safe}__fingerprint.json"
        if result_path.is_file() and trace_path.is_file() and fp_path.is_file():
            stored = json.loads(fp_path.read_text(encoding="utf-8"))
            if stored.get("fingerprint") == fingerprint and stored.get("complete") is True:
                if (C.sha256_file(result_path) == stored.get("result_sha256")
                        and C.sha256_file(trace_path) == stored.get("trace_sha256")):
                    timing.append({"origin": origin, "status": "SKIP"})
                    continue
        t0 = time.time()
        q = group["q"].to_numpy(float)
        segments = P.normalize_segments(group["segment_id"].to_numpy(int))
        dates = list(group["delivery_date"])
        cap = P.origin_cap(len(group))
        stored_cap = int(group["action_cap"].iloc[0]) if "action_cap" in group else cap
        if cap != stored_cap:
            raise RuntimeError(f"{origin}: cap mismatch {cap} != {stored_cap}")
        payload, frame, meta = run_origin(
            origin, cohort, q, segments, dates,
            thr_h if thr_h is not None else None,
            thr_d, cap, group["production_action"].to_numpy(bool),
        )
        payload["fingerprint"] = fingerprint
        payload["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                                  "pandas": pd.__version__, "platform": platform.platform()}
        C.write_gzip_csv(frame, trace_path)
        C.write_json(result_path, payload)
        C.write_json(fp_path, {"fingerprint": fingerprint, "complete": True,
                               "result_sha256": C.sha256_file(result_path),
                               "trace_sha256": C.sha256_file(trace_path),
                               "seconds": round(time.time() - t0, 3)})
        timing.append({"origin": origin, "status": "RUN", "seconds": round(time.time() - t0, 3)})

    C.write_json(out / "allocation_run_meta.json", {
        "fingerprint": fingerprint, "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "timing": timing,
        "n_jobs": len(timing), "n_skip": sum(1 for row in timing if row["status"] == "SKIP"),
    })
    print(json.dumps({"jobs": len(timing), "skipped": sum(1 for row in timing if row["status"] == "SKIP"),
                      "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
