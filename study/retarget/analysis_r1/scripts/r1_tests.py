"""R-A fail-closed guard suite.

T1 hand-computed bank1_causal (F5) toy paths (state-machine transcription check)
T2 eligibility invariance: perturbing all ineligible/current/future q values
   must not change any action at or before the perturbed decision day
T3 segment relabel invariance for every policy
T4 cap arithmetic, F1 cap-skip semantics, F2 credit-drop accounting
T5 determinism: repeat runs give identical action vectors on a real origin
T6 reverse replay: re-derived production equals stored actions (NYISO + transfer)

No project modules with import side effects are imported; the F5 mirror check
against confirm_runner.cocd_known_horizon_cap_registered is replaced by the
hand-computed cases plus a recorded source-level transcription note (the
admitted module imports a project-wide graph at module level).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import r1_common as C  # noqa: E402
import r1_policies as P  # noqa: E402
import r1_budget_oracle as ORACLE  # noqa: E402


def T1_handcomputed_bank1() -> dict:
    dates = pd.date_range("2026-01-01", periods=20, freq="D").tolist()
    segments = np.zeros(20, dtype=int)
    cap = P.origin_cap(20)  # round(0.15*20) = 3
    results = {}

    shadow = np.zeros(20, dtype=bool)
    shadow[[2, 10, 19]] = True
    tension = np.ones(20)
    action, trace, flags = P.bank1_causal(shadow, segments, tension, dates, cap, "2026-01-20")
    results["constant_tension_defer_then_flush"] = {
        "expected_action_days": [3, 11, 19],
        "got_action_days": np.flatnonzero(action).tolist(),
        "pass": np.flatnonzero(action).tolist() == [3, 11, 19],
        "flags": flags,
    }

    shadow2 = np.zeros(20, dtype=bool)
    action2, _, flags2 = P.bank1_causal(shadow2, segments, tension, dates, cap, "2026-01-20")
    results["no_shadow_no_actions"] = {
        "pass": bool(action2.sum() == 0 and not flags2["bank_open_at_trace_end"]),
    }

    shadow3 = np.zeros(20, dtype=bool)
    shadow3[[1, 5]] = True
    tension3 = np.arange(1.0, 21.0)
    action3, trace3, flags3 = P.bank1_causal(shadow3, segments, tension3, dates, cap, "2026-01-20")
    results["increasing_tension_immediate"] = {
        "expected_action_days": [1, 5],
        "got_action_days": np.flatnonzero(action3).tolist(),
        "pass": np.flatnonzero(action3).tolist() == [1, 5],
        "high_flags": [bool(row["high_tension"]) for row in trace3 if row["shadow_mint"] == 1],
    }

    # Declared end far beyond the trace end: a mint deferred on the final day
    # leaves the bank open; the credit is dropped with the flag (recorded
    # deviation), no synthetic action.
    shadow4 = np.zeros(20, dtype=bool)
    shadow4[[19]] = True
    action4, _, flags4 = P.bank1_causal(shadow4, segments, tension, dates, cap, "2026-02-19")
    results["open_bank_at_trace_end_flagged"] = {
        "got_action_days": np.flatnonzero(action4).tolist(),
        "pass": bool(action4.sum() == 0 and flags4["bank_open_at_trace_end"] is True),
    }
    results["all_pass"] = all(v.get("pass") for v in results.values() if isinstance(v, dict))
    return results


def T2_eligibility_invariance(root: Path) -> dict:
    """Perturb q at dates >= t-1 (including t) and require the action prefix
    through t to be unchanged.  Older (previous-segment) outcomes are NOT
    perturbed: they legitimately affect earlier spending and the global cap."""
    frame = C.load_nyiso_daily(root)
    out: dict = {"origins": {}, "checks": {}}
    origins = sorted(frame["origin"].unique())[:3]
    all_pass = True
    for origin in origins:
        group = frame[frame["origin"] == origin].sort_values("delivery_date", kind="mergesort").reset_index(drop=True)
        q = group["q"].to_numpy(float)
        segments = P.normalize_segments(group["segment_id"].to_numpy(int))
        dates = list(group["delivery_date"])
        cap = P.origin_cap(len(q))
        thr = C.read_gzip_csv(root / C.R39_HOURLY)
        thr = thr[(thr["method"] == "saocp") & (thr["layer"] == "base_no_fallback") & (thr["origin"] == origin)]
        thr = thr.sort_values(["delivery_date", "hour"], kind="mergesort")
        tension = P.tension_from_thresholds(thr, dates, segments)
        declared_end = str(C.declared_horizon_end(origin).date())
        origin_checks = {}
        for frac in (0.25, 0.5, 0.75):
            pivot = int(len(q) * frac)
            perturb = np.array([dates[i] >= dates[pivot] - pd.Timedelta(days=1) for i in range(len(q))])
            rng = np.random.default_rng(1000 + pivot)
            q_p = q.copy()
            q_p[perturb] = rng.random(int(perturb.sum()))
            for name, runner in (
                ("P", lambda data: P.production_drb(data, segments, dates, cap)[0]),
                ("F3", lambda data: P.k2_thresholded_count(data, segments, dates, cap)[0]),
                ("F4", lambda data: P.persist1(data, segments, dates, cap)[0]),
            ):
                base_actions = runner(q)
                pert_actions = runner(q_p)
                key = f"{name}_pivot{pivot}"
                origin_checks[key] = bool(np.array_equal(base_actions[:pivot + 1], pert_actions[:pivot + 1]))
            # F5: shadow recomputed under the perturbed path, tension fixed
            base5 = P.bank1_causal(P.production_drb(q, segments, dates, cap)[0], segments, tension,
                                   dates, cap, declared_end)[0]
            pert5 = P.bank1_causal(P.production_drb(q_p, segments, dates, cap)[0], segments, tension,
                                   dates, cap, declared_end)[0]
            origin_checks[f"F5_pivot{pivot}"] = bool(np.array_equal(base5[:pivot + 1], pert5[:pivot + 1]))
        out["origins"][origin] = origin_checks
        all_pass = all_pass and all(origin_checks.values())

    # Synthetic cases: binding budget and a gap.
    dates = pd.date_range("2026-01-01", periods=30, freq="D").tolist()
    segments = np.zeros(30, dtype=int)
    segments[15:] = 1  # observed gap at index 15
    q_syn = np.ones(30)
    q_syn[[3, 4, 9, 10, 11, 16, 17]] = 0.2
    cap_syn = 2  # binding
    pivot = 12
    perturb = np.array([i >= pivot - 1 for i in range(30)])
    q_syn_p = q_syn.copy()
    q_syn_p[perturb] = 0.9
    for name, runner in (
        ("P", lambda data: P.production_drb(data, segments, dates, cap_syn)[0]),
        ("F3", lambda data: P.k2_thresholded_count(data, segments, dates, cap_syn)[0]),
        ("F4", lambda data: P.persist1(data, segments, dates, cap_syn)[0]),
    ):
        out["checks"][f"synthetic_binding_{name}"] = bool(
            np.array_equal(runner(q_syn)[:pivot + 1], runner(q_syn_p)[:pivot + 1]))
    out["checks"]["synthetic_binding_actions_saturated"] = bool(
        int(np.sum(P.production_drb(q_syn, segments, dates, cap_syn)[0])) == cap_syn)
    all_pass = all_pass and all(out["checks"].values())
    out["all_pass"] = bool(all_pass)
    return out


def T7_oracle_toy(root: Path) -> dict:
    """True objective-conflict toy (reviewer note 1 regression).

    n=10, cap=1, q=1 except q0=0, q6=0.4, q9=0.2.  Max-rescue alone chooses
    day0 (rescue 1, W7 0.8); max-min W7 requires day6 (W7 6/7 = 0.8571,
    rescue 0.6).  The earlier toy (three deficient days, three actions) was
    vacuous and is replaced by this case.
    """
    import itertools
    n = 10
    segments = np.zeros(n, dtype=int)
    q = np.ones(n)
    q[0], q[6], q[9] = 0.0, 0.4, 0.2
    cap = 1
    windows = [list(range(s, s + 7)) for s in range(0, n - 6)]
    best_z, rescue_at_best_z, best_rescue_any = -1.0, -1.0, -1.0
    for combo in itertools.combinations(range(n), cap):
        a = np.zeros(n, dtype=bool)
        a[list(combo)] = True
        service = np.where(a, 1.0, q)
        z = min(float(np.mean(service[w])) for w in windows)
        rescue = float(np.sum((1.0 - q)[a]))
        best_rescue_any = max(best_rescue_any, rescue)
        if z > best_z + 1e-12:
            best_z, rescue_at_best_z = z, rescue
        elif abs(z - best_z) <= 1e-12:
            rescue_at_best_z = max(rescue_at_best_z, rescue)
    solved = ORACLE.solve_oracle(q, segments, "cap", cap)
    checks = {
        "conflict_z_matches_enumeration": bool(abs(solved["z1"] - best_z) <= 1e-9),
        "conflict_z_equals_6_over_7": bool(abs(solved["z1"] - 6.0 / 7.0) <= 1e-9),
        "conflict_rescue_matches_enumeration": bool(abs(solved["rescued_shortfall_sum"] - rescue_at_best_z) <= 1e-9),
        "rescue_optimum_alone_differs": bool(best_rescue_any > rescue_at_best_z + 1e-9),
        "rounded_actions_valid": bool(solved["validation"]["valid"]),
        "windows_retained_complete": bool(len(solved["windows"]) == 4),
        "canonical_hash_deterministic": bool(ORACLE.canonical_payload_sha256(solved)
                                             == ORACLE.canonical_payload_sha256(json.loads(json.dumps(solved)))),
    }
    return {"enumerated_best_z": best_z, "enumerated_rescue_at_best_z": rescue_at_best_z,
            "enumerated_best_rescue_any": best_rescue_any,
            "solver_z": solved["z1"], "solver_rescue": solved["rescued_shortfall_sum"],
            "checks": checks, "all_pass": all(checks.values()),
            "errata": "previous T7 toy (three actions covering three deficient days) was vacuous per reviewer note 1"}


def T8_fine_grid_crossings() -> dict:
    dec = C.crossing_lambda(-0.010, 0.05)   # more shortfall traded for more actions -> crossing at +0.2
    neg = C.crossing_lambda(+0.010, +0.05)  # more shortfall AND more actions -> crossing at -0.2
    equal = C.crossing_lambda(0.004, 0.0)
    checks = {
        "crossing_value": bool(abs(dec["crossing_lambda"] - 0.2) < 1e-12),
        "crossing_in_grid_flag": bool(dec["crossing_in_frozen_fine_grid"] is True),
        "negative_crossing_handled": bool(neg["equal_action_rate"] is False
                                          and neg["crossing_lambda"] < 0
                                          and neg["crossing_in_frozen_fine_grid"] is False),
        "equal_rate_reported_explicitly": bool(equal["equal_action_rate"] is True
                                               and equal["crossing_lambda"] is None
                                               and equal["sign"] == "positive"),
        "fine_grid_defined": bool(len(C.LAMBDA_FINE) == 11 and C.LAMBDA_FINE[0] == 0.0
                                  and abs(C.LAMBDA_FINE[-1] - 0.20) < 1e-12),
    }
    return {"checks": checks, "all_pass": all(checks.values())}


def T9_support_checks(root: Path) -> dict:
    a = pd.DataFrame({"origin": ["x", "y"], "d": [1, 2]})
    b = pd.DataFrame({"origin": ["x", "z"], "d": [1, 5]})
    missing_detected = False
    try:
        C.assert_same_support(a, b, ["origin", "d"], "a", "b")
    except RuntimeError as exc:
        missing_detected = "support mismatch" in str(exc)
    dup = pd.DataFrame({"origin": ["x", "x"], "d": [1, 1]})
    dup_detected = False
    try:
        C.assert_same_support(a, dup, ["origin", "d"], "a", "dup")
    except RuntimeError as exc:
        dup_detected = "duplicate" in str(exc)
    checks = {"missing_extra_detected": missing_detected, "duplicates_detected": dup_detected}
    return {"checks": checks, "all_pass": all(checks.values())}


def T3_segment_relabel(root: Path) -> dict:
    frame = C.load_nyiso_daily(root)
    origin = sorted(frame["origin"].unique())[0]
    group = frame[frame["origin"] == origin].sort_values("delivery_date", kind="mergesort").reset_index(drop=True)
    q = group["q"].to_numpy(float)
    seg = P.normalize_segments(group["segment_id"].to_numpy(int))
    seg_shift = seg + 1000
    dates = list(group["delivery_date"])
    cap = P.origin_cap(len(q))
    checks = {}
    for name, runner in (
        ("P", lambda segments: P.production_drb(q, segments, dates, cap)[0]),
        ("F1", lambda segments: P.pace7(q, segments, dates, cap)[0]),
        ("F2", lambda segments: P.pace_bank(q, segments, dates, cap)[0]),
        ("F3", lambda segments: P.k2_thresholded_count(q, segments, dates, cap)[0]),
        ("F4", lambda segments: P.persist1(q, segments, dates, cap)[0]),
    ):
        checks[name] = bool(np.array_equal(runner(seg), runner(seg_shift)))
    return {"checks": checks, "all_pass": all(checks.values())}


def T4_cap_and_accounting(root: Path) -> dict:
    frame = C.load_nyiso_daily(root)
    origin = sorted(frame["origin"].unique())[0]
    group = frame[frame["origin"] == origin].sort_values("delivery_date", kind="mergesort").reset_index(drop=True)
    q = group["q"].to_numpy(float)
    seg = P.normalize_segments(group["segment_id"].to_numpy(int))
    dates = list(group["delivery_date"])
    cap = P.origin_cap(len(q))
    out: dict = {"cap": cap, "checks": {}}
    zero = P.pace7(q, seg, dates, 0)[0]
    out["checks"]["pace7_zero_cap_no_actions"] = bool(zero.sum() == 0)
    actions, trace = P.pace7(q, seg, dates, cap)
    scheduled = sum(1 for row in trace if row["scheduled"])
    skipped = sum(1 for row in trace if row["cap_skipped"])
    out["checks"]["pace7_cap_semantics"] = bool(int(actions.sum()) + skipped == scheduled
                                                and int(actions.sum()) <= cap)
    out["pace7_scheduled"] = int(scheduled)
    out["pace7_skipped"] = int(skipped)
    actions2, trace2 = P.pace_bank(q, seg, dates, cap)
    dropped = sum(row["dropped_on_reset"] for row in trace2)
    out["checks"]["pacebank_accounting"] = bool(int(actions2.sum()) <= cap
                                                and all(row["bank"] >= 0 for row in trace2))
    out["pacebank_dropped_credit_units"] = int(dropped)
    out["checks"]["pacebank_uses_at_most_cap"] = bool(int(actions2.sum()) <= cap)
    out["all_pass"] = all(out["checks"].values())
    return out


def T5_determinism(root: Path) -> dict:
    frame = C.load_nyiso_daily(root)
    origin = sorted(frame["origin"].unique())[0]
    group = frame[frame["origin"] == origin].sort_values("delivery_date", kind="mergesort").reset_index(drop=True)
    q = group["q"].to_numpy(float)
    seg = P.normalize_segments(group["segment_id"].to_numpy(int))
    dates = list(group["delivery_date"])
    cap = P.origin_cap(len(q))
    a1 = P.production_drb(q, seg, dates, cap)[0]
    a2 = P.production_drb(q, seg, dates, cap)[0]
    b1 = P.pace_bank(q, seg, dates, cap)[0]
    b2 = P.pace_bank(q, seg, dates, cap)[0]
    return {"checks": {"production_deterministic": bool(np.array_equal(a1, a2)),
                       "pacebank_deterministic": bool(np.array_equal(b1, b2))},
            "all_pass": bool(np.array_equal(a1, a2) and np.array_equal(b1, b2))}


def T6_reverse_replay(root: Path) -> dict:
    nyiso = C.load_nyiso_daily(root)
    transfer = C.load_transfer_daily(root)
    bad_prod: list[str] = []
    bad_k2: list[str] = []
    for origin, group in nyiso.groupby("origin", sort=True):
        group = group.sort_values("delivery_date", kind="mergesort").reset_index(drop=True)
        q = group["q"].to_numpy(float)
        seg = P.normalize_segments(group["segment_id"].to_numpy(int))
        cap = P.origin_cap(len(group))
        action = P.production_drb(q, seg, list(group["delivery_date"]), cap)[0]
        if not np.array_equal(action, group["production_action"].to_numpy(bool)):
            bad_prod.append(f"nyiso/{origin}")
    for origin, group in transfer.groupby("origin", sort=True):
        group = group.sort_values("delivery_date", kind="mergesort").reset_index(drop=True)
        q = group["q"].to_numpy(float)
        seg = P.normalize_segments(group["segment_id"].to_numpy(int))
        cap = P.origin_cap(len(group))
        dates = list(group["delivery_date"])
        if not np.array_equal(P.production_drb(q, seg, dates, cap)[0],
                              group["production_action"].to_numpy(bool)):
            bad_prod.append(f"transfer/{origin}")
        if not np.array_equal(P.k2_thresholded_count(q, seg, dates, cap)[0],
                              group["k2_action"].to_numpy(bool)):
            bad_k2.append(f"transfer/{origin}")
    return {
        "production_mismatches": bad_prod,
        "k2_mismatches": bad_k2,
        "n_origins": int(nyiso["origin"].nunique() + transfer["origin"].nunique()),
        "identity_note": ("NYISO production checked against the R39 stored action column; "
                          "transfer production checked against the p0_2 continuous production rows, "
                          "and K2 against the p0_2/R51 thresholded-count rows (R51's original "
                          "'drb_action' label carries the K2 rule)."),
        "all_pass": len(bad_prod) == 0 and len(bad_k2) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=C.EVIDENCE_ROOT)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "tests" / "r1_tests.json")
    args = parser.parse_args()
    results = {
        "T1_bank1_handcomputed": T1_handcomputed_bank1(),
        "T2_eligibility_invariance": T2_eligibility_invariance(args.root),
        "T3_segment_relabel": T3_segment_relabel(args.root),
        "T4_cap_accounting": T4_cap_and_accounting(args.root),
        "T5_determinism": T5_determinism(args.root),
        "T6_reverse_replay": T6_reverse_replay(args.root),
        "T7_oracle_toy_enumeration": T7_oracle_toy(args.root),
        "T8_fine_grid_crossings": T8_fine_grid_crossings(),
        "T9_support_checks": T9_support_checks(args.root),
        "note_f5_mirror": ("strict runtime mirror against confirm_runner."
                           "cocd_known_horizon_cap_registered not executed: importing that module "
                           "loads a project-wide import graph at module level (side-effect risk); "
                           "transcription verified instead by T1 hand-computed cases and line-level "
                           "source comparison recorded in the R1 protocol."),
    }
    results["all_pass"] = all(v["all_pass"] for k, v in results.items() if isinstance(v, dict) and "all_pass" in v)
    C.write_json(args.out, results)
    print(json.dumps({k: (v.get("all_pass") if isinstance(v, dict) else v) for k, v in results.items()}, indent=1))
    return 0 if results["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
