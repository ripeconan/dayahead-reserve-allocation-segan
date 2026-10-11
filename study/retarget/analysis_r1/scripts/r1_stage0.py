"""R-A Stage 0: identity gate (fail-closed, tombstone-first).

Checks, in order:
  1. SHA-256 of every bound input (r1_common.INPUT_HASHES).
  2. Per-origin cap convention: round(0.15 * n_days) == stored action_cap.
  3. Reverse replay: re-derived production DRB action vector == stored vector
     (R39 NYISO seven origins; R51 transfer 23 origins).
  4. Cross-check: merged NYISO q_t equals the reported no-fallback base path
     (recomputed from stored service when present); R02 daily q cross-check;
     R39 vs R02 hourly threshold cross-check (reported, non-blocking only when
     R39 is used as the canonical tension source).
  5. Identity/admissibility table for every comparator (recorded JSON).

A FAIL tombstone is written before any computation; PASS replaces it only
after all blocking checks succeed.  Zero unresolved identity mismatches are
permitted for blocking checks.
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


def _code_hashes() -> dict[str, str]:
    return {name: C.sha256_file(HERE / name) for name in
            ("r1_common.py", "r1_policies.py", "r1_stage0.py")}


def run(root: Path, out: Path) -> dict:
    report: dict = {"stage": "R1-stage0", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    fail_path = out / "r1_stage0_verification.json"
    C.write_json(fail_path, {"status": "FAIL", "reason": "tombstone-before-compute"})
    checks: dict = {}
    try:
        checks["input_hashes"] = C.verify_inputs(root)
        checks["environment"] = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        }
        checks["code_hashes"] = _code_hashes()

        nyiso = C.load_nyiso_daily(root)
        transfer = C.load_transfer_daily(root)

        # ---- caps and reverse replay -------------------------------------
        per_origin: list[dict] = []
        mismatch_total = 0
        for cohort_kind, frame in (("nyiso", nyiso), ("transfer", transfer)):
            for origin, group in frame.groupby("origin", sort=True):
                group = group.sort_values("delivery_date", kind="mergesort").reset_index(drop=True)
                q = group["q"].to_numpy(float)
                segments = P.normalize_segments(group["segment_id"].to_numpy(int))
                dates = list(group["delivery_date"])
                n_days = len(group)
                cap = P.origin_cap(n_days)
                stored_cap = int(group["action_cap"].iloc[0])
                n_distinct_caps = int(group["action_cap"].nunique())
                stored_action = group["production_action"].to_numpy(bool)
                action, _ = P.production_drb(q, segments, dates, cap)
                mismatches = int(np.sum(action != stored_action))
                mismatch_total += mismatches
                k2_mismatch = None
                if cohort_kind == "transfer":
                    k2, _ = P.k2_thresholded_count(q, segments, dates, cap)
                    k2_mismatch = int(np.sum(k2 != group["k2_action"].to_numpy(bool)))
                    mismatch_total += k2_mismatch
                per_origin.append({
                    "cohort_kind": cohort_kind,
                    "cohort": str(group["cohort"].iloc[0]) if "cohort" in group else "nyiso_exposed",
                    "origin": str(origin),
                    "n_days": n_days,
                    "n_segments": int(len(np.unique(segments))),
                    "cap_recomputed": cap,
                    "cap_stored": stored_cap,
                    "cap_stored_distinct_values": n_distinct_caps,
                    "cap_match": bool(cap == stored_cap),
                    "production_action_total": int(stored_action.sum()),
                    "reverse_replay_mismatches": mismatches,
                    "k2_reverse_replay_mismatches": k2_mismatch,
                    "binary_q": bool(np.all((q == 0) | (q == 1))) if cohort_kind == "transfer" else None,
                })
        checks["per_origin"] = per_origin
        checks["reverse_replay_mismatches_total"] = mismatch_total
        checks["reverse_replay_pass"] = bool(mismatch_total == 0)
        checks["cap_mismatches"] = [row["origin"] for row in per_origin if not row["cap_match"]]

        # ---- NYISO q cross-check vs r02 daily (historical earlier stream) ----
        r02 = C.read_gzip_csv(root / C.R02_DAILY)
        r02 = r02[(r02["policy"] == "no_fallback")][["origin", "delivery_date", "base_daily_coverage"]].copy()
        r02["delivery_date"] = pd.to_datetime(r02["delivery_date"], errors="raise")
        key = ["origin", "delivery_date"]
        r39_keys = C.key_set(nyiso, key)
        r02_keys = C.key_set(r02, key)
        merged = nyiso.merge(r02, on=key, how="inner", suffixes=("", "_r02"))
        q_diff = np.abs(merged["q"].to_numpy(float) - merged["base_daily_coverage"].to_numpy(float))
        checks["r02_q_crosscheck"] = {
            "label": ("r02 is a distinct, earlier NYISO UQ stream (nyiso_saocp_drb_current_r01 runner); "
                      "it is NOT combined with the canonical R39 base and is reported for identity "
                      "diagnosis only"),
            "canonical_source": "R39 (verified-SAOCP seven-origin admitted run)",
            "rows_compared": int(len(merged)),
            "missing_in_r02": int(len(r39_keys - r02_keys)),
            "extra_in_r02": int(len(r02_keys - r39_keys)),
            "max_abs_diff_on_common_support": float(q_diff.max()) if len(merged) else None,
            "matches_within_1e-9": bool(len(merged) and r39_keys == r02_keys and q_diff.max() <= 1e-9),
        }

        # ---- hourly threshold cross-check (R39 canonical vs r02) ---------
        h39 = C.read_gzip_csv(root / C.R39_HOURLY)
        h39 = h39[(h39["method"] == "saocp") & (h39["layer"] == "base_no_fallback")][
            ["origin", "delivery_date", "hour", "threshold", "segment_id", "covered"]].copy()
        h39["delivery_date"] = pd.to_datetime(h39["delivery_date"], errors="raise")
        h02 = C.load_nyiso_hourly(root)[["origin", "delivery_date", "hour", "threshold", "covered"]]
        hkey = ["origin", "delivery_date", "hour"]
        h39_keys = C.key_set(h39, hkey)
        h02_keys = C.key_set(h02, hkey)
        hh = h39.merge(h02, on=hkey, how="inner", suffixes=("_39", "_02"))
        thr_diff = np.abs(hh["threshold_39"].to_numpy(float) - hh["threshold_02"].to_numpy(float))
        checks["hourly_threshold_crosscheck"] = {
            "label": ("R39 vs r02 are distinct runs; the canonical F5 tension source is the R39 hourly "
                      "threshold stream, and the r02 stream is not combined with it"),
            "canonical_source": "R39 selected_hourly_results.csv.gz (method=saocp, layer=base_no_fallback)",
            "rows_compared": int(len(hh)),
            "missing_in_r02": int(len(h39_keys - h02_keys)),
            "extra_in_r02": int(len(h02_keys - h39_keys)),
            "max_abs_diff_on_common_support": float(thr_diff.max()) if len(hh) else None,
            "matches_within_1e-9": bool(len(hh) and h39_keys == h02_keys and thr_diff.max() <= 1e-9),
        }

        # ---- comparator admissibility table ------------------------------
        checks["comparator_admissibility"] = [
            {"id": "P", "name": "production DRB (continuous shortfall-sum, theta=1.0, memory 6, lag-2 calendar days)",
             "source": "re-derived; NYISO checked against R39 stored actions; transfer checked against the "
                       "p0_2 continuous production rows",
             "identity_note": ("R51's original column label 'drb_action' carries the thresholded-count K=2 rule "
                               "(verified equal to the p0_2 K2 rows on 23/23 origins); the continuous production "
                               "controller is the p0_2 'production_shortfall_sum_lag2' path. Naming differs; the "
                               "action vectors are pinned by hash and checked per origin."),
             "admissible": True, "contract": "causal_lag2", "horizon_flush": None},
            {"id": "F1", "name": "pace7 fixed 7-day schedule per segment", "source": "r1_policies.pace7",
             "admissible": True, "contract": "causal_outcome_blind; cap-skip recorded",
             "horizon_flush": None},
            {"id": "F2", "name": "paceBank credit bank (accrual 15/day, cost 100, segment reset)",
             "source": "r1_policies.pace_bank", "admissible": True,
             "contract": "causal_outcome_blind; residual credit dropped at segment reset",
             "horizon_flush": None},
            {"id": "F3", "name": "R51 thresholded-count K2 (count of eligible served<0.9 in last 6 >= 2)",
             "source": "r1_policies.k2_thresholded_count", "admissible": True,
             "contract": "causal_lag2"},
            {"id": "F4", "name": "persist1 (last eligible served <= 0.5)",
             "source": "r1_policies.persist1", "admissible": True, "contract": "causal_lag2"},
            {"id": "F5", "name": "bank1_causal (one-credit bank, flush at declared calendar end)",
             "source": "r1_policies.bank1_causal (mirrors confirm_runner A*)", "admissible": True,
             "contract": "causal_lag2_plus_declared_horizon_end",
             "horizon_flush": "declared calendar end per origin label; open credit at trace end dropped and flagged"},
            {"id": "F6", "name": "random placement (existing R02 draws, cap-matched)",
             "source": "R02 random_draw_endpoints.csv.gz", "admissible": "context_only",
             "contract": "outcome_independent_precommitted_random"},
            {"id": "F7", "name": "R45 COCD (budget/end-conditioned one-credit deferral)",
             "source": "R45 cocd_daily_ledger.csv.gz", "admissible": "context_only_not_deployable",
             "contract": "budget_and_segment_end_conditioned"},
            {"id": "F8", "name": "exact noncausal oracle (at count / at cap)",
             "source": "R43 package + re-solved MILP", "admissible": "noncausal_ceiling",
             "contract": "full_outcome_mip"},
        ]

        blocking_ok = bool(
            checks["reverse_replay_pass"]
            and not checks["cap_mismatches"]
        )
        report.update(checks)
        report["status"] = "PASS" if blocking_ok else "FAIL"
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        C.write_json(fail_path, report)
        return report
    except Exception as exc:  # fail closed, preserve partial evidence
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        C.write_json(fail_path, report)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=C.EVIDENCE_ROOT)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "stage0")
    args = parser.parse_args()
    report = run(args.root, args.out)
    print(json.dumps({k: v for k, v in report.items() if k not in ("input_hashes", "per_origin", "comparator_admissibility", "code_hashes")}, default=str))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
