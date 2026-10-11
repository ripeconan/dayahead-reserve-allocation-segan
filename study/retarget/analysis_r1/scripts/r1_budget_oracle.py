"""R-A budget-binding and oracle-headroom analyses (v3).

v3 changes (R02B additional review notes 2/3/4; amendment 2026-10-11C):
  * no self-referential hash: the canonical payload hash excludes every
    `*_sha256` field, and the unit's true file hash lives in an EXTERNAL
    sidecar; resume verifies the sidecar, the canonical hash, and re-derives
    every input-dependent objective value and constraint from the RETAINED
    action vector (never from stored PASS booleans).
  * units retain the full action vector, every within-segment window index and
    mean, the exact origin support (dates sha256), and solver diagnostics.
  * solver failures raise a structured `OracleSolveError` carrying stage
    artifacts and diagnostics; FAIL objects embed them; older FAIL objects are
    never unlinked - a supersession/attempt record is appended instead.
  * aggregate semantics: the expected origin set is enumerated explicitly;
    missing/unexpected units are reported; the seven-origin macro comparison
    runs only when all seven current-fingerprint units pass.

Three objects:
  A. Unlimited-cap production replay: mechanical closed-loop sensitivity.
  B. Oracle at count: exact noncausal max-min-7 allocation with the production
     realized count; verified per origin against the admitted R43 values.
  C. Oracle at cap: same objective family with sum(a) <= cap; C - B is an
     explicit quota-headroom calculation.  Both C - B and C - production are
     full-outcome ceiling comparisons, not a causal decomposition of the
     deployed policy.

Solver contract: scipy.optimize.milp (HiGHS); time_limit=300 s,
mip_rel_gap=0.0, presolve=True; process CPU affinity pinned to two logical
CPUs via psutil (applied and recorded) with OMP/MKL/OPENBLAS thread variables
set to 2 before the BLAS import.  Tie honesty: the tertiary early-action rule
is deterministic but the optimal action vector is generally NOT unique; only
objective values are claimed and verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import r1_common as C  # noqa: E402
import r1_policies as P  # noqa: E402

R43_HEADROOM = (0.154762, 0.178571, 0.184524, 0.267857, 0.178571, 0.190476, 0.113095)
R43_MACRO_ORACLE_W7 = 0.911565
R43_MACRO_DRB_W7 = 0.730442
VALIDATION_TOL = 1e-9
Z_TOL = 1e-9
AFFINITY_CPUS = [1, 2]
SCHEMA = "card10.segan_r1.oracle_unit.v3"


class OracleSolveError(RuntimeError):
    """Carries the stage artifacts and solver diagnostics of a failed solve."""

    def __init__(self, message: str, diagnostics: list[dict], artifacts: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.artifacts = artifacts or {}


def _apply_affinity() -> dict:
    try:
        import psutil
        proc = psutil.Process()
        before = proc.cpu_affinity()
        proc.cpu_affinity(AFFINITY_CPUS)
        return {"applied": True, "cpus_before": before, "cpus_after": proc.cpu_affinity(),
                "thread_env": {v: os.environ.get(v) for v in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    except Exception as exc:
        return {"applied": False, "error": f"{type(exc).__name__}: {exc}",
                "thread_env": {v: os.environ.get(v) for v in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def _fingerprint() -> str:
    return C.build_fingerprint(("r1_budget_oracle.py", "r1_common.py", "r1_policies.py"))


def canonical_payload_sha256(payload: dict) -> str:
    """Hash of the canonical payload EXCLUDING every ``*_sha256`` field.

    Excluding hash fields makes the value stable when written back into the
    file (the v2 self-reference bug wrote a hash of bytes that then changed).
    """
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in sorted(value.items()) if not k.endswith("_sha256")}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    blob = json.dumps(strip(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _windows_matrix(segments: np.ndarray, width: int = 7) -> tuple[sp.csr_matrix, int]:
    windows = C.segment_windows(segments, width)
    rows, cols = [], []
    for r, window in enumerate(windows):
        rows.extend([r] * len(window))
        cols.extend(window.tolist())
    data = np.ones(len(rows), dtype=float)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(windows), len(segments))), len(windows)


def _win_list(service: np.ndarray, segments: np.ndarray) -> list[dict]:
    return [{"w0": int(w[0]), "w_end": int(w[-1]), "mean": float(np.mean(service[w]))}
            for w in C.segment_windows(segments, 7)]


def validate_actions(q: np.ndarray, segments: np.ndarray, action: np.ndarray,
                     mode: str, level: int, z1: float, rescue_star: float) -> dict:
    n = len(q)
    service = np.where(action, 1.0, q)
    windows = _win_list(service, segments)
    worst = min((w["mean"] for w in windows), default=float("nan"))
    count = int(action.sum())
    rescue = float(np.sum((1.0 - q)[action]))
    level_ok = (count == level) if mode == "count" else (count <= level)
    return {
        "n_actions": count,
        "level_ok": bool(level_ok),
        "count_equals_level": bool(count == level),
        "within_cap": bool(count <= level) if mode == "cap" else None,
        "min_window_mean": worst,
        "worst7": worst,
        "mean_service": float(np.mean(service)),
        "z1": z1,
        "worst7_ge_z1_minus_tol": bool(worst >= z1 - VALIDATION_TOL),
        "rescued_shortfall_sum": rescue,
        "rescue_ge_stage2_minus_tol": bool(rescue >= rescue_star - VALIDATION_TOL),
        "all_windows_valid": bool(all(w["mean"] >= z1 - VALIDATION_TOL for w in windows)),
        "valid": bool(level_ok and worst >= z1 - VALIDATION_TOL
                      and rescue >= rescue_star - VALIDATION_TOL),
    }


def solve_oracle(q: np.ndarray, segments: np.ndarray, mode: str, level: int,
                 time_limit: float = 300.0) -> dict:
    n = len(q)
    W, n_windows = _windows_matrix(segments, 7)
    if n_windows == 0:
        raise OracleSolveError("no within-segment 7-windows", [])
    ones = np.ones(n)
    window_q = W @ q
    neg_w = np.asarray(W.todense())
    coef = 1.0 - q
    A_w = sp.hstack([sp.csr_matrix(neg_w * coef[None, :]),
                     sp.csr_matrix(-7.0 * np.ones((n_windows, 1)))]).tocsr()
    lb_w = -window_q
    ub_w = np.full(n_windows, np.inf)
    A_level = sp.csr_matrix(np.concatenate([ones, [0.0]])[None, :])
    if mode == "count":
        lb_e, ub_e = np.array([float(level)]), np.array([float(level)])
    elif mode == "cap":
        lb_e, ub_e = np.array([0.0]), np.array([float(level)])
    else:
        raise ValueError(mode)
    A0 = sp.vstack([A_w, A_level]).tocsr()
    lb0 = np.concatenate([lb_w, lb_e])
    ub0 = np.concatenate([ub_w, ub_e])
    integrality = np.concatenate([np.ones(n), np.zeros(1)])
    bounds = Bounds(lb=np.zeros(n + 1), ub=np.concatenate([np.ones(n), [1.0]]))

    diagnostics: list[dict] = []
    artifacts: dict = {"mode": mode, "level": int(level), "n_days": int(n)}

    def _solve(c, A_use, lb_use, ub_use, stage: str):
        t0 = time.time()
        res = milp(c=c, constraints=LinearConstraint(A_use, lb_use, ub_use),
                   integrality=integrality, bounds=bounds,
                   options={"time_limit": time_limit, "mip_rel_gap": 0.0, "presolve": True})
        diagnostics.append({
            "stage": stage, "status": int(res.status), "message": str(res.message),
            "mip_gap": float(getattr(res, "mip_gap", float("nan"))),
            "mip_node_count": int(getattr(res, "mip_node_count", -1)),
            "mip_dual_bound": float(getattr(res, "mip_dual_bound", float("nan"))),
            "fun": float(res.fun) if res.fun is not None else None,
            "seconds": round(time.time() - t0, 3),
        })
        if res.status != 0 or res.x is None:
            raise OracleSolveError(f"solver not optimal at {stage}: status={res.status} {res.message}",
                                   diagnostics, dict(artifacts))
        return res

    r1 = _solve(np.concatenate([np.zeros(n), [-1.0]]), A0, lb0, ub0, "max_min_w7")
    z1 = float(r1.x[-1])
    artifacts["stage1_z"] = z1

    # Preserved-z constraint: +1*z >= z1 - tol (the v1 sign error enforced the opposite).
    A_z = sp.csr_matrix(np.concatenate([np.zeros(n), [1.0]])[None, :])
    A2 = sp.vstack([A0, A_z]).tocsr()
    lb2 = np.concatenate([lb0, [z1 - Z_TOL]])
    ub2 = np.concatenate([ub0, [np.inf]])
    r2 = _solve(np.concatenate([-coef, [0.0]]), A2, lb2, ub2, "max_rescue_at_z1")
    rescue2 = float(coef @ r2.x[:n])
    artifacts["stage2_rescue"] = rescue2

    A_r = sp.csr_matrix(np.concatenate([coef, [0.0]])[None, :])
    A3 = sp.vstack([A2, A_r]).tocsr()
    lb3 = np.concatenate([lb2, [rescue2 - Z_TOL]])
    ub3 = np.concatenate([ub2, [np.inf]])
    c3 = np.concatenate([(np.arange(n) + 1) / n, [0.0]])
    r3 = _solve(c3, A3, lb3, ub3, "early_action_tiebreak")

    action = np.asarray(r3.x[:n]) > 0.5
    validation = validate_actions(q, segments, action, mode, level, z1, rescue2)
    return {
        "mode": mode, "level": int(level),
        "z1": z1,
        "worst7": validation["worst7"],
        "mean_service": validation["mean_service"],
        "action_total": int(action.sum()),
        "rescued_shortfall_sum": validation["rescued_shortfall_sum"],
        "stage2_rescue_optimum": rescue2,
        "validation": validation,
        "action_vector": [int(v) for v in action],
        "windows": _win_list(np.where(action, 1.0, q), segments),
        "diagnostics": diagnostics,
    }


def revalidate_unit(unit: dict, q: np.ndarray, segments: np.ndarray, dates: list,
                    cap: int, origin_index: int) -> dict:
    """Re-derive input-dependent objectives/constraints from retained artifacts."""
    problems: list[str] = []
    n = len(q)
    if unit.get("n_days") != n:
        problems.append(f"n_days {unit.get('n_days')} != {n}")
    if unit.get("dates_sha256") != hashlib.sha256(
            json.dumps([str(pd.Timestamp(d).date()) for d in dates]).encode()).hexdigest():
        problems.append("origin support dates hash mismatch")
    drb_w7 = unit.get("drb_worst7")
    for key in ("oracle_count", "oracle_cap"):
        block = unit.get(key)
        if not isinstance(block, dict):
            problems.append(f"{key} missing")
            continue
        action = np.asarray(block.get("action_vector", []), dtype=float) > 0.5
        if len(action) != n:
            problems.append(f"{key}: action_vector length {len(action)} != {n}")
            continue
        level = block.get("level")
        mode = block.get("mode")
        check = validate_actions(q, segments, action, mode, int(level),
                                 float(block.get("z1")), float(block.get("stage2_rescue_optimum")))
        if not check["valid"]:
            problems.append(f"{key}: retained actions violate locked constraints")
        for a, b in ((check["worst7"], block.get("worst7")),
                     (check["rescued_shortfall_sum"], block.get("rescued_shortfall_sum"))):
            if not (isinstance(b, (int, float)) and abs(a - b) <= 1e-9):
                problems.append(f"{key}: retained-vs-recomputed mismatch {a} vs {b}")
        windows = block.get("windows") or []
        if len(windows) != len(_win_list(np.where(action, 1.0, q), segments)):
            problems.append(f"{key}: retained window list length mismatch")
        for w in windows:
            idx = np.arange(int(w["w0"]), int(w["w_end"]) + 1)
            recomputed = float(np.mean(np.where(action, 1.0, q)[idx]))
            if abs(recomputed - float(w["mean"])) > 1e-9:
                problems.append(f"{key}: window {w['w0']} mean mismatch")
                break
    if unit.get("oracle_count", {}).get("mode") != "count":
        problems.append("oracle_count mode label wrong")
    if unit.get("oracle_cap", {}).get("mode") != "cap":
        problems.append("oracle_cap mode label wrong")
    headroom = (unit.get("oracle_count", {}).get("worst7", float("nan")) - drb_w7
                if isinstance(drb_w7, (int, float)) else float("nan"))
    r43_ok = abs(headroom - R43_HEADROOM[origin_index]) <= 1e-6
    if not r43_ok:
        problems.append(f"R43 headroom mismatch: {headroom} vs {R43_HEADROOM[origin_index]}")
    return {"revalidated": len(problems) == 0, "problems": problems,
            "recomputed_headroom": float(headroom) if np.isfinite(headroom) else None}


def _origin_inputs(frame: pd.DataFrame, origin: str):
    group = frame[frame["origin"] == origin].sort_values("delivery_date", kind="mergesort").reset_index(drop=True)
    q = group["q"].to_numpy(float)
    segments = P.normalize_segments(group["segment_id"].to_numpy(int))
    dates = list(group["delivery_date"])
    cap = P.origin_cap(len(group))
    stored = group["production_action"].to_numpy(bool)
    return q, segments, dates, cap, stored


def _origin_key(origin: str) -> str:
    return str(origin).replace("->", "_")


def _dates_sha(dates: list) -> str:
    return hashlib.sha256(json.dumps([str(pd.Timestamp(d).date()) for d in dates]).encode()).hexdigest()


def _attempt_record(out: Path, origin: str, entry: dict) -> None:
    path = out / f"nyiso__{_origin_key(origin)}.attempts.json"
    history = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    history.append(entry)
    C.write_json(path, history)


def run(root: Path, out: Path, only_origins: set[str] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    affinity = _apply_affinity()
    fingerprint = _fingerprint()
    nyiso = C.load_nyiso_daily(root)
    all_origins = sorted(nyiso["origin"].unique().tolist())
    expected = sorted(only_origins) if only_origins else all_origins
    summary: dict = {"schema": SCHEMA, "cohort": "nyiso_exposed", "origins": {},
                     "affinity": affinity, "fingerprint": fingerprint,
                     "expected_origins": expected, "failures": [], "missing_units": []}
    units: list[dict] = []
    for origin in all_origins:
        if origin not in expected:
            continue
        idx = all_origins.index(origin)
        safe = _origin_key(origin)
        path = out / f"nyiso__{safe}.json"
        side = out / f"nyiso__{safe}.sidecar.json"
        q, segments, dates, cap, stored_action = _origin_inputs(nyiso, origin)
        # ---- resume path: external sidecar + canonical hash + revalidation --
        if path.is_file() and side.is_file():
            sidecar = json.loads(side.read_text(encoding="utf-8"))
            unit = json.loads(path.read_text(encoding="utf-8"))
            if (sidecar.get("fingerprint") == fingerprint and sidecar.get("complete") is True
                    and sidecar.get("unit_file_sha256") == C.sha256_file(path)
                    and sidecar.get("canonical_payload_sha256") == canonical_payload_sha256(unit)
                    and unit.get("canonical_payload_sha256") == sidecar.get("canonical_payload_sha256")):
                check = revalidate_unit(unit, q, segments, dates, cap, idx)
                if check["revalidated"]:
                    units.append(unit)
                    summary["origins"][origin] = {k: v for k, v in unit.items()
                                                  if k not in ("oracle_count", "oracle_cap")}
                    _attempt_record(out, origin, {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                  "event": "resume_skip_verified",
                                                  "canonical_payload_sha256": sidecar.get("canonical_payload_sha256")})
                    continue
                summary["failures"].append(f"{origin}: resume revalidation failed: {check['problems'][:3]}")
        # ---- compute path ---------------------------------------------------
        t0 = time.time()
        attempt_id = f"{time.strftime('%Y%m%dT%H%M%S')}.{hashlib.sha256(f'{origin}{t0}'.encode()).hexdigest()[:8]}"
        try:
            drb, _ = P.production_drb(q, segments, dates, cap)
            if int(np.sum(drb != stored_action)) != 0:
                raise RuntimeError("production reverse replay failed inside budget runner")
            drb_w7 = C.worst_window(np.where(drb, 1.0, q), segments, 7)
            uncapped, uncapped_trace = P.production_drb(q, segments, dates, 10 ** 9)
            uncapped_w7 = C.worst_window(np.where(uncapped, 1.0, q), segments, 7)
            count = int(stored_action.sum())
            oracle_count = solve_oracle(q, segments, "count", count)
            oracle_cap = solve_oracle(q, segments, "cap", cap)
            headroom = oracle_count["worst7"] - drb_w7
            r43_ok = abs(headroom - R43_HEADROOM[idx]) <= 1e-6
            validations_ok = bool(oracle_count["validation"]["valid"] and oracle_cap["validation"]["valid"])
            unit = {
                "schema": SCHEMA, "fingerprint": fingerprint,
                "status": "PASS" if (r43_ok and validations_ok) else "FAIL",
                "origin": origin, "origin_index": idx,
                "n_days": len(q), "cap": cap, "production_count": count,
                "first_date": str(pd.Timestamp(dates[0]).date()),
                "last_date": str(pd.Timestamp(dates[-1]).date()),
                "dates_sha256": _dates_sha(dates),
                "drb_worst7": drb_w7,
                "unlimited_cap": {
                    "action_total": int(uncapped.sum()), "worst7": uncapped_w7,
                    "delta_worst7_vs_capped": float(uncapped_w7 - drb_w7),
                    "blocked_triggers_removed": int(sum(1 for row in uncapped_trace if row["blocked"])),
                    "interpretation": "mechanical closed-loop sensitivity; NOT a causal budget component",
                },
                "oracle_count": oracle_count,
                "oracle_cap": oracle_cap,
                "quota_headroom_cap_minus_count": float(oracle_cap["worst7"] - oracle_count["worst7"]),
                "mixed_headroom_cap_minus_drb": float(oracle_cap["worst7"] - drb_w7),
                "count_ceiling_gap_vs_drb": float(oracle_count["worst7"] - drb_w7),
                "r43_headroom_check": {"got": headroom, "admitted": R43_HEADROOM[idx],
                                       "within_1e-6": bool(r43_ok)},
                "seconds": round(time.time() - t0, 3), "affinity": affinity,
                "interpretation": ("oracle_at_cap and oracle_at_count are full-outcome ceiling "
                                   "comparisons; their difference is an explicit quota-headroom "
                                   "calculation, not a causal decomposition of the deployed policy"),
            }
            unit["canonical_payload_sha256"] = canonical_payload_sha256(unit)
            C.write_json(path, unit)
            C.write_json(side, {
                "fingerprint": fingerprint, "complete": True,
                "canonical_payload_sha256": unit["canonical_payload_sha256"],
                "unit_file_sha256": C.sha256_file(path),
                "unit_bytes": path.stat().st_size,
                "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            _attempt_record(out, origin, {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "attempt": attempt_id,
                                          "event": "computed", "status": unit["status"],
                                          "canonical_payload_sha256": unit["canonical_payload_sha256"]})
            if unit["status"] != "PASS":
                C.write_json(out / f"nyiso__{safe}.FAIL.{attempt_id}.json", unit)
                summary["failures"].append(f"{origin}: admission check failed (attempt {attempt_id})")
            units.append(unit)
            summary["origins"][origin] = {k: v for k, v in unit.items()
                                          if k not in ("oracle_count", "oracle_cap")}
        except OracleSolveError as exc:
            C.write_json(out / f"nyiso__{safe}.FAIL.{attempt_id}.json", {
                "schema": SCHEMA, "status": "FAIL", "origin": origin, "attempt": attempt_id,
                "error": str(exc), "diagnostics": exc.diagnostics, "artifacts": exc.artifacts,
                "fingerprint": fingerprint})
            _attempt_record(out, origin, {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "attempt": attempt_id,
                                          "event": "solve_failed", "error": str(exc)})
            summary["failures"].append(f"{origin}: {exc}")
        except Exception as exc:
            C.write_json(out / f"nyiso__{safe}.FAIL.{attempt_id}.json", {
                "schema": SCHEMA, "status": "FAIL", "origin": origin, "attempt": attempt_id,
                "error": f"{type(exc).__name__}: {exc}", "fingerprint": fingerprint})
            _attempt_record(out, origin, {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "attempt": attempt_id,
                                          "event": "error", "error": f"{type(exc).__name__}: {exc}"})
            summary["failures"].append(f"{origin}: {type(exc).__name__}: {exc}")

    # ---- aggregate semantics (reviewer note 4) ------------------------------
    valid = [u for u in units if u.get("status") == "PASS"]
    valid_origins = sorted(u["origin"] for u in valid)
    summary["valid_origins"] = valid_origins
    summary["missing_units"] = [o for o in expected if o not in valid_origins]
    complete = (not only_origins) and (valid_origins == all_origins) and not summary["failures"]
    per_unit_err = [abs(u["r43_headroom_check"]["got"] - u["r43_headroom_check"]["admitted"]) for u in valid]
    agg: dict = {"n_valid_units": len(valid), "n_expected_units": len(all_origins),
                 "complete_full_cohort": bool(complete),
                 "max_abs_headroom_error_vs_audit": max(per_unit_err) if per_unit_err else None,
                 "passes_1e-6": bool(per_unit_err and max(per_unit_err) <= 1e-6)}
    if complete:
        macro_oracle = float(np.mean([u["oracle_count"]["worst7"] for u in valid]))
        macro_drb = float(np.mean([u["drb_worst7"] for u in valid]))
        agg.update({"macro_oracle_worst7": macro_oracle,
                    "audited_macro_oracle_worst7": R43_MACRO_ORACLE_W7,
                    "macro_oracle_error": abs(macro_oracle - R43_MACRO_ORACLE_W7),
                    "macro_drb_worst7": macro_drb,
                    "audited_macro_drb_worst7": R43_MACRO_DRB_W7,
                    "macro_drb_error": abs(macro_drb - R43_MACRO_DRB_W7)})
        if not (agg["passes_1e-6"] and agg["macro_oracle_error"] <= 1e-6 and agg["macro_drb_error"] <= 1e-6):
            summary["failures"].append("aggregate verification failed vs admitted R43 macro values")
    elif only_origins:
        agg["note"] = "partial/smoke run: full-cohort macro comparison not applicable"
    else:
        agg["note"] = ("incomplete cohort: per-origin R43 checks apply; macro comparison deferred "
                       f"(missing {summary['missing_units']})")
    summary["aggregate_verification"] = agg
    if complete and agg["passes_1e-6"] and not summary["failures"]:
        summary["status"] = "PASS"
    elif only_origins or not complete:
        summary["status"] = "PARTIAL_INCOMPLETE"
    else:
        summary["status"] = "FAIL"
    C.write_json(out / "budget_oracle_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=C.EVIDENCE_ROOT)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "budget_oracle")
    parser.add_argument("--origins", type=str, default="")
    args = parser.parse_args()
    only = {s for s in args.origins.split(",") if s} if args.origins else None
    summary = run(args.root, args.out, only)
    print(json.dumps({k: v for k, v in summary.items() if k in
                      ("status", "affinity", "aggregate_verification", "failures",
                       "missing_units", "valid_origins")}, indent=1, default=str))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
