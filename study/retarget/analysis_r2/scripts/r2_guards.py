"""R2 guard suite — v4: tests call the REAL production pipeline functions.

Coverage (R03B item 10 requirements, extended by the R03E mailbox):
  P1  real end-to-end pipeline run (synthetic frames, pinned kernel) + artifact
      completeness (required columns, expected-hour grid, bridge grid cells)
  P2  UTC- vs New-York-indexed equivalence (REVIEWER_R2_TIMEZONE_RESULTS regression):
      both emitted `utc` columns are normalized through pd.to_datetime(utc=True)
      and must have EXACT timestamp-key support equality before any value compare
  P3  perturbation of actual VALUES and MISSINGNESS from t-1 onward: point
      forecasts, geometry, regions (thresholds, BOTH branches) and FULL action
      prefixes through t must be unchanged for every allocator; the prefix
      helper itself is exercised on future-support-differs / prefix-gap cases
  P4  year seam + whole-reference-year eligibility (boundary = Jan 1 00:00)
  P5  DST: provenance on both transitions; 23/25-hour expected days; one frozen
      threshold per (date, slot); the fall day's 25 ledger rows carry the exact
      source instants and the exact reveal date in the ORIGINAL emitted order
  P6  unknown missing outcome: commitment/budget retained, reset at the reveal
      date, deficit not scored, capacity still booked
  P7  manifest tamper rejection + real resume SKIP (via r2_run on a temp raw,
      through the SHARED admission validator)
  P8  support joins and date-aware windows (production utilities)
  P9  bridge arithmetic incl. worst-window = MAX, 1h per real hour, costs
  P10 determinism (identical walks + full numerical frames + payload with only
      the non-scientific runtime `seconds` field excluded)
  P11 theory propositions (support inclusion / binary-action degradation /
      blocked-trigger bound), exhaustive synthetic validation
  P12 missing day forced action (booking retained, never scored)
  P13 F5 tension independence: thresholds, F5 actions AND the PERSISTED NUMERIC
      pre-action tension series are compared through the pivot
  P14 aggregate admission + ASYMMETRIC sign fixture (no year fitting):
      policy-minus-none / P-minus-Fi labels verified on modeled and physical
      contrasts incl. a kappa sign flip, plus invalid-unit refusal controls
  P15 bootstrap draw equivalence: the vectorized (n_boot, n) draw matrix must be
      bit-identical to the pre-R03E per-draw loop at the production settings

Runner (R03E item 3): `--checkout` / `R2_CHECKOUT` resolution (no hardcoded
Windows path is required), per-test exception capture with traceback, partial
results + source fingerprints persisted after EVERY test, BEGIN/END timestamps
printed with flush, exit code nonzero on any fail/error/incomplete.  A
`--only`-selected subset is explicitly labelled and never claims to be the full
suite.  BLAS/OMP/MKL thread limits are set before numeric imports (r2_common).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import r2_env  # noqa: F401  (thread limits before numeric imports)

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import r2_common as C  # noqa: E402
import r2_pipeline as PL  # noqa: E402
import r2_saocp_batch as SB  # noqa: E402
import r2_theory as T  # noqa: E402

TEST_YEAR = 2021
STUB = SB.StubKernel
REAL_KERNEL = C.SAOCP_LOCAL_FALLBACK

# Portable checkout: --checkout wins, then R2_CHECKOUT, then the local default
# (development convenience only; native runs must pass --checkout or the env var).
LOCAL_CHECKOUT_DEFAULT = Path("D:/mypaper/literature-infra/card10-data/online_conformal_r85")
CHECKOUT: Path | None = None
CHECKOUT_SOURCE = "unset"

GUARD_SOURCE_FILES = ("r2_guards.py", "r2_run.py", "r2_common.py", "r2_pipeline.py",
                      "r2_pointcorr.py", "r2_saocp_batch.py", "r2_allocators.py",
                      "r2_bridge.py", "r2_theory.py", "r2_env.py",
                      "vendor/saocp_ast_kernel.py")


def resolve_checkout(cli_value: str | None) -> tuple[Path | None, str]:
    if cli_value:
        return Path(cli_value), "cli"
    env = os.environ.get("R2_CHECKOUT")
    if env:
        return Path(env), "env:R2_CHECKOUT"
    if LOCAL_CHECKOUT_DEFAULT.exists():
        return LOCAL_CHECKOUT_DEFAULT, "local_default"
    return None, "unset"


def require_checkout() -> Path:
    if CHECKOUT is None:
        raise RuntimeError("no pinned checkout configured: pass --checkout or set R2_CHECKOUT "
                           "(the hardcoded local default is absent on this host)")
    return CHECKOUT


def guard_source_fingerprint() -> dict:
    return {name: (C.sha256_file(HERE / name) if (HERE / name).is_file() else None)
            for name in GUARD_SOURCE_FILES}


def _run(frames_actual: pd.DataFrame, frames_forecast: pd.DataFrame, *, stub: bool = True,
         test_year: int = TEST_YEAR):
    if stub:
        factory = (lambda scale: STUB(scale))
    else:
        bundle, ast_module, _meta = SB.load_ast_kernel(require_checkout())
        factory = SB.ast_kernel_factory(bundle, ast_module)
    return PL.run_origin_from_frames(frames_actual, frames_forecast, test_year,
                                     kernel_factory=factory)


def P1_pipeline_end_to_end() -> dict:
    actual, forecast = PL.synthetic_frames(TEST_YEAR, include_reference_days=420)
    payload, frames = _run(actual, forecast)
    hourly = frames["hours"]
    required_cols = {"utc", "delivery_date", "slot", "duration_h", "actual_load_MW",
                     "issued_point_MW", "residual_MW", "threshold"}
    required_cols |= {f"{name}_{z}" for z in C.ZONES for name in
                      ("actual_load_MW", "issued_point_MW", "residual_MW",
                       "geometry_center_MW", "radius_MW", "upper_residual_support_MW")}
    checks = {
        "required_columns_present": bool(required_cols.issubset(set(hourly.columns))),
        "r0_nonnegative": bool((hourly[[f"r0_{z}" for z in C.ZONES]].to_numpy(float) >= 0).all()),
        "residual_identity": bool(np.allclose(
            (hourly["actual_load_MW"] - hourly["issued_point_MW"]).dropna(),
            hourly["residual_MW"].dropna(), equal_nan=True)),
        "expected_hours_match_grid": bool(sum(payload["expected_hours_per_day_counts"].values())
                                          == payload["n_expected_hours"]),
        "bridge_grid_complete": bool(len(payload["bridge_grid"])
                                     == 7 * len(C.ETA_GRID) * len(C.COST_RATIO_GRID)),
        "allocator_metrics_present": bool(set(payload["allocator_metrics"]) == set(PL.POLICIES)),
        "no_q_on_unobserved": bool((frames["daily"].loc[~frames["daily"]["observed"], "q"].isna()).all()),
    }
    return {"checks": checks, "pass": all(checks.values())}


def _issued_series(hours: pd.DataFrame) -> pd.Series:
    """issued_point_MW keyed by a NORMALIZED UTC timestamp.

    The emitted `utc` column is a string; an America/New_York-indexed frame and
    a UTC-indexed frame denote the SAME instants with DIFFERENT strings
    (`...-04:00` vs `...Z`).  Every cross-representation comparison therefore
    goes through pd.to_datetime(..., utc=True) - never through the raw strings.
    """
    frame = hours[["utc", "issued_point_MW"]].copy()
    frame["utc"] = pd.to_datetime(frame["utc"], utc=True)
    return frame.set_index("utc")["issued_point_MW"]


def _local_dates(index: pd.DatetimeIndex) -> pd.Index:
    return pd.Index(index.tz_convert(C.TZ).date.astype(str))


def _filter_prefix(series: pd.Series, pivot: str) -> pd.Series:
    local = _local_dates(pd.DatetimeIndex(series.index))
    return series[local <= pivot]


def _series_prefix_equal(left: pd.Series, right: pd.Series, pivot: str,
                         tolerance: float = 1e-9) -> tuple[bool, dict]:
    """Exact prefix equality for instants whose LOCAL delivery date <= pivot.

    Future support may legitimately differ (a future day loses its inputs when a
    lagged error disappears), so BOTH sides are filtered to the prefix FIRST;
    the prefix key sets must then be EXACTLY equal and NONEMPTY before any value
    comparison (R03E items 1/10).  An intersection that could hide a missing
    prefix key is never used, and an empty prefix is a failure, not a vacuous
    pass.
    """
    left_prefix, right_prefix = _filter_prefix(left, pivot), _filter_prefix(right, pivot)
    info = {"pivot": pivot, "left_rows": int(len(left_prefix)), "right_rows": int(len(right_prefix)),
            "future_left_only": int(len(left) - len(left_prefix)),
            "future_right_only": int(len(right) - len(right_prefix))}
    if len(left_prefix) == 0 or len(right_prefix) == 0:
        info["reason"] = "empty prefix (nothing was compared)"
        return False, info
    missing = set(left_prefix.index) - set(right_prefix.index)
    extra = set(right_prefix.index) - set(left_prefix.index)
    if missing or extra:
        info.update({"reason": "prefix key support differs", "missing_keys": len(missing),
                     "extra_keys": len(extra),
                     "first_missing": [str(x) for x in sorted(missing)[:3]],
                     "first_extra": [str(x) for x in sorted(extra)[:3]]})
        return False, info
    order = left_prefix.index.sort_values()
    a = left_prefix.reindex(order).to_numpy(float)
    b = right_prefix.reindex(order).to_numpy(float)
    both_nan = np.isnan(a) & np.isnan(b)
    ok = bool(np.all(both_nan | (np.abs(a - b) <= tolerance)))
    info["n_compared"] = int(len(order))
    info["n_both_nan"] = int(both_nan.sum())
    if not ok:
        info["reason"] = "value mismatch inside the prefix"
        bad = np.flatnonzero(~(both_nan | (np.abs(a - b) <= tolerance)))
        info["first_bad"] = str(order[int(bad[0])])
        info["first_bad_values"] = [float(a[bad[0]]), float(b[bad[0]])]
    return ok, info


def _threshold_prefix_equal(base_frames, other_frames, pivot: str) -> tuple[bool, dict]:
    """Frozen per-(date, slot) thresholds through the pivot: exact keys, then values."""
    def table(frames) -> pd.Series:
        thr = frames["adapter_thresholds"]
        sub = thr[thr["delivery_date"] <= pivot]
        return sub.set_index(["delivery_date", "slot"])["threshold"].sort_index()
    left, right = table(base_frames), table(other_frames)
    info = {"pivot": pivot, "left_rows": int(len(left)), "right_rows": int(len(right))}
    if len(left) == 0 or set(left.index) != set(right.index):
        info["reason"] = "empty or differing threshold key support"
        return False, info
    a = left.to_numpy(float)
    b = right.reindex(left.index).to_numpy(float)
    ok = bool(np.all(np.abs(a - b) <= 1e-12))
    if not ok:
        info["reason"] = "threshold value mismatch"
        bad = int(np.flatnonzero(np.abs(a - b) > 1e-12)[0])
        info["first_bad_key"] = [str(x) for x in left.index[bad]]
    return ok, info


def P2_utc_local_equivalence() -> dict:
    actual, forecast = PL.synthetic_frames(TEST_YEAR, include_reference_days=420)
    factory = lambda scale: STUB(scale)
    payload_ny, frames_ny = PL.run_origin_from_frames(actual, forecast, TEST_YEAR, kernel_factory=factory)
    payload_utc, frames_utc = PL.run_origin_from_frames(
        actual.tz_convert("UTC"), forecast.tz_convert("UTC"), TEST_YEAR, kernel_factory=factory)
    audit_ny = frames_ny["point_audit"]
    audit_utc = frames_utc["point_audit"]
    same_days = list(audit_ny.index) == list(audit_utc.index)
    same_counts = bool((audit_ny["issued_rows"].to_numpy() == audit_utc["issued_rows"].to_numpy()).all()
                       and (audit_ny["train_rows"].to_numpy() == audit_utc["train_rows"].to_numpy()).all())
    left = _issued_series(frames_ny["hours"])
    right = _issued_series(frames_utc["hours"])
    support_equal = set(left.index) == set(right.index)
    max_diff = None
    if support_equal:
        order = left.index.sort_values()
        a = left.reindex(order).to_numpy(float)
        b = right.reindex(order).to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.where(both_nan, 0.0, np.abs(a - b))
        max_diff = float(np.nanmax(diff)) if len(diff) and np.isfinite(diff).any() else 0.0
    checks = {"audit_days_equal": same_days, "audit_counts_equal": same_counts,
              "exact_utc_support_equality": bool(support_equal),
              "issued_point_identical": bool(max_diff is not None and max_diff <= 1e-9)}
    return {"checks": checks, "n_rows": int(len(left)), "max_issued_point_diff": max_diff,
            "pass": all(checks.values())}


def P3_perturbation_prefixes() -> dict:
    actual, forecast = PL.synthetic_frames(TEST_YEAR, include_reference_days=420)
    factory = lambda scale: STUB(scale)
    base_payload, base_frames = PL.run_origin_from_frames(actual, forecast, TEST_YEAR, kernel_factory=factory)
    pivot = "2021-06-15"
    pivot_ts = pd.Timestamp(pivot)
    local = actual.index.tz_convert(C.TZ).normalize().tz_localize(None)
    # (a) VALUES from t-1 onward
    rng = np.random.default_rng(7)
    mask_value = local >= (pivot_ts - pd.DateOffset(days=1))
    actual_v = actual.copy()
    actual_v.loc[mask_value, :] = rng.normal(0.0, 60.0, size=(int(mask_value.sum()), 11)) + 100.0
    payload_v, frames_v = PL.run_origin_from_frames(actual_v, forecast, TEST_YEAR, kernel_factory=factory)
    # (b) MISSINGNESS from t-1 onward
    actual_m = actual.copy()
    actual_m.loc[mask_value, :] = np.nan
    payload_m, frames_m = PL.run_origin_from_frames(actual_m, forecast, TEST_YEAR, kernel_factory=factory)
    checks = {"geometry_identical_v": bool(np.allclose(
        np.asarray(base_payload["geometry"]["center"]), np.asarray(payload_v["geometry"]["center"]))
        and np.allclose(np.asarray(base_payload["geometry"]["covariance"]),
                        np.asarray(payload_v["geometry"]["covariance"]))),
        "geometry_identical_m": bool(np.allclose(
            np.asarray(base_payload["geometry"]["center"]), np.asarray(payload_m["geometry"]["center"]))
            and np.allclose(np.asarray(base_payload["geometry"]["covariance"]),
                            np.asarray(payload_m["geometry"]["covariance"])))}
    base_issued = _issued_series(base_frames["hours"])
    issued_v, info_issued_v = _series_prefix_equal(base_issued, _issued_series(frames_v["hours"]), pivot)
    issued_m, info_issued_m = _series_prefix_equal(base_issued, _issued_series(frames_m["hours"]), pivot)
    checks["issued_through_pivot_identical_v"] = issued_v
    checks["issued_through_pivot_identical_m"] = issued_m
    thr_v, info_thr_v = _threshold_prefix_equal(base_frames, frames_v, pivot)
    thr_m, info_thr_m = _threshold_prefix_equal(base_frames, frames_m, pivot)
    checks["region_thresholds_through_pivot_identical_v"] = thr_v
    checks["region_thresholds_through_pivot_identical_m"] = thr_m
    for policy in PL.POLICIES:
        checks[f"action_prefix_identical_v_{policy}"] = _action_prefix_equal(
            base_frames["walks"][policy], frames_v["walks"][policy], pivot)
        checks[f"action_prefix_identical_m_{policy}"] = _action_prefix_equal(
            base_frames["walks"][policy], frames_m["walks"][policy], pivot)
    checks.update(_prefix_helper_checks(pivot))
    return {"checks": checks, "pass": all(checks.values()),
            "details": {"issued_v": info_issued_v, "issued_m": info_issued_m,
                        "thresholds_v": info_thr_v, "thresholds_m": info_thr_m}}


def _prefix_helper_checks(pivot: str) -> dict:
    """Exercise the prefix comparator itself (the guard must not be vacuous).

    A: identical prefixes with DIFFERENT future support -> equal (the real
       production case: future issuance legitimately disappears).
    B: a missing row INSIDE the prefix -> unequal (an intersection would hide it).
    C: a changed value inside the prefix -> unequal.
    D: empty prefix -> failure, never a vacuous pass.
    E: NaN patterns inside the prefix must match.
    """
    index = pd.DatetimeIndex(pd.date_range(pd.Timestamp(pivot) - pd.Timedelta(days=3),
                                           periods=6, freq="D", tz=C.TZ))
    base = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=index)  # rows 0-3 in prefix
    future_dropped = base.iloc[:5].copy()          # pivot+1 row absent (future only)
    prefix_gap = base.drop(index[2])               # a row through the pivot missing
    value_changed = base.copy()
    value_changed.iloc[2] = 3.5
    nan_pattern = base.copy()
    nan_pattern.iloc[1] = np.nan
    nan_mismatch = base.copy()
    nan_mismatch.iloc[1] = np.nan
    nan_mismatch.iloc[3] = np.nan
    empty = base.iloc[:0]
    a_ok, _ = _series_prefix_equal(base, future_dropped, pivot)
    b_ok, _ = _series_prefix_equal(base, prefix_gap, pivot)
    c_ok, _ = _series_prefix_equal(base, value_changed, pivot)
    d_ok, _ = _series_prefix_equal(base, empty, pivot)
    e_ok, _ = _series_prefix_equal(nan_pattern, nan_pattern, pivot)
    f_ok, _ = _series_prefix_equal(nan_pattern, nan_mismatch, pivot)
    return {"prefix_helper_future_support_differs_ok": bool(a_ok),
            "prefix_helper_prefix_gap_detected": bool(not b_ok),
            "prefix_helper_value_change_detected": bool(not c_ok),
            "prefix_helper_empty_prefix_fails": bool(not d_ok),
            "prefix_helper_nan_pattern_equal": bool(e_ok),
            "prefix_helper_nan_pattern_mismatch_detected": bool(not f_ok)}


def _action_prefix_equal(base_walk: pd.DataFrame, other_walk: pd.DataFrame, pivot: str) -> bool:
    a = base_walk[base_walk["delivery_date"] <= pivot]["action"].to_numpy()
    b = other_walk[other_walk["delivery_date"] <= pivot]["action"].to_numpy()
    return bool(len(a) == len(b) and (a == b).all())


def P4_year_seam() -> dict:
    cal = C.declared_calendar(TEST_YEAR)
    boundary = C.newest_label_boundary(pd.Timestamp(f"{TEST_YEAR}-01-02"))
    actual, forecast = PL.synthetic_frames(TEST_YEAR, include_reference_days=420)
    payload, frames = _run(actual, forecast)
    audit = frames["point_audit"]
    test_rows = audit[audit.index >= f"{TEST_YEAR}-01-02"]
    first = test_rows.iloc[0]
    checks = {
        "jan2_start": bool(cal[0] == pd.Timestamp(f"{TEST_YEAR}-01-02")),
        "boundary_jan1_midnight": bool(boundary == pd.Timestamp(f"{TEST_YEAR}-01-01")),
        "cap_55": bool(C.planned_cap(TEST_YEAR) == 55 and C.planned_cap(2024) == 55),
        "first_day_label_is_dec31": bool(first["newest_train_label_utc"] is not None
                                         and pd.Timestamp(first["newest_train_label_utc"])
                                         .tz_convert(C.TZ).date().isoformat()
                                         == f"{TEST_YEAR - 1}-12-31"),
    }
    return {"checks": checks, "pass": all(checks.values())}


def P5_dst_structure() -> dict:
    expected_2024 = PL.expected_utc_hours(2024)
    fall_day = expected_2024["2024-11-03"]
    spring_day = expected_2024["2024-03-10"]
    provenance_ok = all(C.lag_provenance_ok(t, t - pd.Timedelta(hours=lag), lag)
                        for t in pd.date_range("2024-03-09", "2024-11-04 23:00", freq="h", tz=C.TZ)
                        for lag in C.LAG_HOURS)
    actual, forecast = PL.synthetic_frames(2024, include_reference_days=420)
    payload, frames = _run(actual, forecast, test_year=2024)
    thr = frames["adapter_thresholds"]
    per_date_slot_unique = bool(not thr.duplicated(["delivery_date", "slot"]).any())
    ledger = frames["adapter_ledger"]
    fall_rows = ledger[ledger["source_local_date"] == "2024-11-03"]
    two_folds = int((fall_rows["slot"] == 1).sum())
    # chronology is asserted on the ORIGINAL emitted order (sorting first would
    # make the check tautological: R03E item 5)
    order_ok = bool(pd.to_datetime(fall_rows["source_utc"], utc=True).is_monotonic_increasing)
    applied_ok = bool((fall_rows["applied_decision_date"] == "2024-11-05").all()) if len(fall_rows) else False
    expected_fall_keys = set(pd.DatetimeIndex(PL.expected_utc_hours(2024)["2024-11-03"]).tz_convert("UTC"))
    emitted_fall_keys = set(pd.to_datetime(fall_rows["source_utc"], utc=True))
    reveal_keys = set(fall_rows["reveal_local_date"].astype(str))
    checks = {
        "fall_day_25h": bool(len(fall_day) == 25),
        "spring_day_23h": bool(len(spring_day) == 23),
        "provenance_all_ok": bool(provenance_ok),
        "one_frozen_threshold_per_date_slot": per_date_slot_unique,
        "fall_two_fold_scores": bool(two_folds == 2),
        "fall_ledger_rows_25": bool(len(fall_rows) == 25),
        "fall_source_keys_exact_25h": bool(emitted_fall_keys == expected_fall_keys
                                           and len(emitted_fall_keys) == 25),
        "fall_reveal_keys_exact": bool(reveal_keys == {"2024-11-05"}),
        "fall_release_chronological": order_ok,
        "fall_release_two_days_later": applied_ok,
    }
    return {"checks": checks, "pass": all(checks.values()),
            "fall_ledger_rows": int(len(fall_rows))}


def P6_missing_outcome_retained_costs() -> dict:
    actual, forecast = PL.synthetic_frames(TEST_YEAR, include_reference_days=420,
                                           missing_actual_days=("2021-07-10",))
    payload, frames = _run(actual, forecast)
    walk = frames["walks"]["P"]
    row = walk[walk["delivery_date"] == "2021-07-10"].iloc[0]
    reveal = "2021-07-12"
    event_rows = walk[walk["delivery_date"] == reveal]
    resets = payload["adapter_reset_events"]
    miss_events = [e for e in resets if e["kind"] == "missing_outcome" and e["at_date"] == reveal]
    hourly = frames["hours"]
    day_hours = hourly[hourly["delivery_date"] == "2021-07-10"]
    grid_row = [g for g in payload["bridge_grid"]
                if g["policy"] == "P" and g["eta"] == 1.0 and g["kappa"] == 0.0][0]
    checks = {
        "day_scorable": bool(row["scorable"] is True or bool(row["scorable"])),
        "day_not_observed": bool(not bool(row["observed"])),
        "budget_retained": bool(row["used_after"] is not None),
        "reset_at_reveal": bool(len(miss_events) == 1),
        "deficit_not_scored_for_missing_rows": bool(day_hours["residual_MW"].isna().all()),
        "capacity_still_booked": bool(grid_row["extra_committed_MWh"] > 0
                                      and grid_row["extra_committed_MWh"] > 0.0),
        "expected_hours_retained": bool(len(day_hours) in (23, 24, 25)),
    }
    return {"checks": checks, "pass": all(checks.values())}


def P7_manifest_tamper_and_skip() -> dict:
    import r2_run as R
    checkout = require_checkout()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw_frame = _write_synthetic_raw(tmp / "raw.csv.gz")
        out = tmp / "out"
        first = R.run_origin(TEST_YEAR, raw_frame, out, checkout, fixture_raw=True)
        second = R.run_origin(TEST_YEAR, raw_frame, out, checkout, fixture_raw=True)
        origin_dir = out / f"origin_{TEST_YEAR}"
        manifest = json.loads((origin_dir / "manifest.json").read_text(encoding="utf-8"))
        ok = C.verify_manifest(manifest, origin_dir, tuple(R.REQUIRED_ARTIFACTS))
        target = origin_dir / "hourly.csv.gz"
        tampered = bytearray(target.read_bytes())
        tampered[-1] ^= 0xFF
        target.write_bytes(bytes(tampered))
        after = C.verify_manifest(manifest, origin_dir, tuple(R.REQUIRED_ARTIFACTS))
        third = R.run_origin(TEST_YEAR, raw_frame, out, checkout, fixture_raw=True)
        checks = {
            "first_pass": bool(first["status"] == "PASS"),
            "second_skip": bool(second["status"] == "SKIP_VERIFIED"),
            "manifest_verifies": bool(ok["ok"]),
            "tamper_detected": bool(not after["ok"]),
            "tampered_run_not_skipped": bool(third["status"] != "SKIP_VERIFIED"),
        }
    return {"checks": checks, "pass": all(checks.values())}


def _write_synthetic_raw(path: Path) -> Path:
    actual, forecast = PL.synthetic_frames(TEST_YEAR, include_reference_days=420)
    frame = pd.DataFrame({"timestamp_utc": actual.index.tz_convert("UTC").astype(str)})
    for z in C.ZONES:
        frame[f"{z}_actual"] = actual[z].to_numpy()
        frame[f"{z}_forecast"] = forecast[z].to_numpy()
    C.write_gzip_csv(frame, path)
    return path


def P8_support_and_windows() -> dict:
    a = pd.DataFrame({"o": ["x", "y"], "d": [1, 2]})
    b = pd.DataFrame({"o": ["x", "z"], "d": [1, 5]})
    mismatch = False
    try:
        C.assert_same_support(a, b, ["o", "d"], "a", "b")
    except RuntimeError:
        mismatch = True
    dup = pd.DataFrame({"o": ["x", "x"], "d": [1, 1]})
    dupes = False
    try:
        C.assert_same_support(a, dup, ["o", "d"], "a", "dup")
    except RuntimeError:
        dupes = True
    # eight dates split by a gap (Jan-05 absent): expect NO 7-window
    gapped = ["2021-01-02", "2021-01-03", "2021-01-04", "2021-01-06", "2021-01-07",
              "2021-01-08", "2021-01-09", "2021-01-10"]
    gap_windows = C.consecutive_local_windows(gapped, np.zeros(len(gapped), dtype=int), 7)
    # a real consecutive block of exactly 9 days -> exactly 3 windows at starts 0,1,2
    block = [f"2021-02-{d:02d}" for d in range(1, 10)]
    block_windows = [sorted(w.tolist()) for w in
                     C.consecutive_local_windows(block, np.zeros(len(block), dtype=int), 7)]
    checks = {"mismatch_detected": mismatch, "duplicates_detected": dupes,
              "gapped_8_dates_have_no_7_window": bool(len(gap_windows) == 0),
              "consecutive_9_days_exact_windows": bool(block_windows ==
                                                       [list(range(0, 7)), list(range(1, 8)), list(range(2, 9))])}
    return {"checks": checks, "pass": all(checks.values())}


def P9_bridge_arithmetic() -> dict:
    """9 consecutive complete days; deficits [1,5,2,0,0,0,0,10,0] -> the three
    7-day sums are 8, 17, 12 so the WORST (max) is 17; full uncovered 18; full
    cost at kappa=0.1 with base 10 and no extra is 0.1*(10+0)+18 = 19."""
    import r2_bridge as BR
    dates = [f"2021-03-{d:02d}" for d in range(1, 10)]
    frame = pd.DataFrame({
        "delivery_date": dates,
        "deficit_MWh": [1.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
        "observed_adequate_fraction": [0.9] * 9,
        "extra_committed_MWh": [0.0] * 9,
        "action": [0] * 9,
        "expected_hours": [24.0] * 9,
        "observed_hours": [24.0] * 9,
        "complete_observed_day": [True] * 9,
    })
    segments = {d: 0 for d in dates}
    grid = BR.grid_summary({("P", 0.0): frame}, base_reserved_MWh=10.0, segments_by_date=segments)
    cell = [g for g in grid if g["kappa"] == 0.0][0]
    cell_k = [g for g in grid if abs(g["kappa"] - 0.1) < 1e-12][0]
    checks = {
        "worst7_is_max_17": bool(abs(cell["worst7_deficit_MWh_complete_days"] - 17.0) < 1e-9),
        "uncovered_finite_18": bool(abs(cell["uncovered_MWh_finite_hours"] - 18.0) < 1e-9),
        "full_cost_kappa01_19": bool(abs(cell_k["full_cost_MWh_equivalent"] - 19.0) < 1e-9),
        "segments_required": True,
    }
    try:
        BR.grid_summary({("P", 0.0): frame}, base_reserved_MWh=10.0, segments_by_date=None)
        checks["segments_required"] = False
    except Exception:
        checks["segments_required"] = True
    return {"checks": checks, "pass": all(checks.values())}


def P10_determinism() -> dict:
    actual, forecast = PL.synthetic_frames(TEST_YEAR, include_reference_days=420)
    factory = lambda scale: STUB(scale)
    p1, f1 = PL.run_origin_from_frames(actual, forecast, TEST_YEAR, kernel_factory=factory)
    p2, f2 = PL.run_origin_from_frames(actual, forecast, TEST_YEAR, kernel_factory=factory)
    same = all(f1["walks"][p].equals(f2["walks"][p]) for p in PL.POLICIES)
    frame_keys = ("hours", "daily", "adapter_ledger", "adapter_thresholds",
                  "reference_scores", "reference_residuals", "point_audit", "bridge_daily")
    frames_same = all(f1[k].equals(f2[k]) for k in frame_keys if k in f1 and k in f2)

    def _sci(obj):
        if isinstance(obj, dict):
            return {k: _sci(v) for k, v in obj.items() if k != "seconds"}
        if isinstance(obj, list):
            return [_sci(v) for v in obj]
        return obj
    checks = {"walks_deterministic": bool(same), "frames_deterministic": bool(frames_same),
              "payload_scientific_equal": bool(C.canonical_payload_sha256(_sci(p1))
                                               == C.canonical_payload_sha256(_sci(p2)))}
    return {"checks": checks, "pass": all(checks.values())}


def P11_theory() -> dict:
    rng = np.random.default_rng(20261011)
    inc = T.support_inclusion_checks(rng, trials=60)
    deg = T.degradation_bound_checks(rng, trials=60)
    blk = T.blocked_trigger_bound_check(days=11, budgets=(0, 1, 2))
    checks = {"support_inclusion": bool(inc["pass"]), "binary_action_degradation": bool(deg["pass"]),
              "blocked_trigger_bound": bool(blk["pass"])}
    return {"checks": checks, "pass": all(checks.values()),
            "details": {"inclusion": inc, "degradation": deg, "blocked": blk}}


def P12_missing_day_forced_action() -> dict:
    """F1 is scheduled on a real 24-h day (2021-03-12, segment-local offset 69):
    the committed action stays in the budget, the day is never scored, and the
    PERSISTED bridge daily record books exactly eta*H*24 for that day."""
    actual, forecast = PL.synthetic_frames(TEST_YEAR, include_reference_days=420,
                                           missing_actual_days=("2021-03-12",))
    payload, frames = _run(actual, forecast)
    f1 = frames["walks"]["F1"]
    row = f1[f1["delivery_date"] == "2021-03-12"].iloc[0]
    day = frames["bridge_daily"]
    day_rows = day[(day["policy"] == "F1") & (day["delivery_date"] == "2021-03-12")]
    headroom_sum = float(sum(payload["bridge_headroom_MW"].values()))
    booking_ok = {}
    for eta in (0.5, 1.0, 2.0):
        sub = day_rows[day_rows["eta"] == eta]
        if len(sub) != 1:
            booking_ok[f"eta{eta}"] = False
            continue
        booking_ok[f"eta{eta}"] = bool(abs(float(sub["extra_committed_MWh"].iloc[0])
                                           - eta * headroom_sum * 24.0) < 1e-6)
    checks = {"day_is_f1_scheduled": bool(row["action"] == 1),
              "budget_retained": bool(row["used_after"] is not None),
              "never_scored": bool(row["service"] is None or (isinstance(row["service"], float)
                                                              and row["service"] != row["service"])),
              "observed_hours_zero": bool(float(day_rows["observed_hours"].iloc[0]) == 0.0),
              "complete_flag_false": bool(not bool(day_rows["complete_observed_day"].iloc[0])),
              **booking_ok}
    return {"checks": checks, "pass": all(checks.values())}


def _tension_prefix_equal(base_frames, other_frames, pivot: str) -> tuple[bool, dict]:
    """The PERSISTED pre-action tension series through the pivot: exact dates,
    then numeric values (NaT-like None entries must be None on both sides).

    A series that is entirely non-numeric would make this comparison vacuous,
    so the numeric support inside the prefix is reported and required to be
    substantial (R03E item 1: compare the actual numbers, not a claim).
    """
    def series(frames) -> pd.Series:
        daily = frames["daily"]
        sub = daily[daily["delivery_date"] <= pivot]
        return sub.set_index("delivery_date")["tension"].sort_index()
    left, right = series(base_frames), series(other_frames)
    info = {"pivot": pivot, "left_rows": int(len(left)), "right_rows": int(len(right))}
    if len(left) == 0 or set(left.index) != set(right.index):
        info["reason"] = "empty or differing tension date support"
        return False, info
    a = pd.to_numeric(left, errors="coerce").to_numpy(float)
    b = pd.to_numeric(right.reindex(left.index), errors="coerce").to_numpy(float)
    both_nan = np.isnan(a) & np.isnan(b)
    ok = bool(np.all(both_nan | (np.abs(a - b) <= 1e-12)))
    info.update({"n_compared": int(len(a)), "n_finite_left": int(np.isfinite(a).sum()),
                 "n_finite_right": int(np.isfinite(b).sum()),
                 "tension_min": float(np.nanmin(a)) if np.isfinite(a).any() else None,
                 "tension_max": float(np.nanmax(a)) if np.isfinite(a).any() else None})
    if not ok:
        bad = int(np.flatnonzero(~(both_nan | (np.abs(a - b) <= 1e-12)))[0])
        info.update({"reason": "tension value mismatch", "first_bad_date": str(left.index[bad]),
                     "first_bad_values": [float(a[bad]), float(b[bad])]})
    return ok, info


def P13_f5_tension_independence() -> dict:
    """F5 tension/history must not depend on current/future actuals before the
    reveal: perturbing actuals from t-1 onward leaves issued thresholds, F5
    decisions AND the persisted numeric pre-action tension series through t
    unchanged."""
    actual, forecast = PL.synthetic_frames(TEST_YEAR, include_reference_days=420)
    payload_a, frames_a = _run(actual, forecast)
    pivot = "2021-06-15"
    local = actual.index.tz_convert(C.TZ).normalize().tz_localize(None)
    mask = local >= (pd.Timestamp(pivot) - pd.DateOffset(days=1))
    rng = np.random.default_rng(99)
    actual_b = actual.copy()
    actual_b.loc[mask, :] = rng.normal(0.0, 40.0, size=(int(mask.sum()), 11)) + 100.0
    payload_b, frames_b = _run(actual_b, forecast)
    thr_ok, thr_info = _threshold_prefix_equal(frames_a, frames_b, pivot)
    f5_ok = _action_prefix_equal(frames_a["walks"]["F5"], frames_b["walks"]["F5"], pivot)
    ten_ok, ten_info = _tension_prefix_equal(frames_a, frames_b, pivot)
    n_finite = int(ten_info.get("n_finite_left", 0))
    checks = {"thresholds_through_pivot_identical": bool(thr_ok),
              "f5_prefix_identical": bool(f5_ok),
              "tension_prefix_identical": bool(ten_ok),
              "tension_prefix_numeric_support": bool(n_finite >= 100)}
    return {"checks": checks, "pass": all(checks.values()),
            "details": {"thresholds": thr_info, "tension": ten_info}}


# --------------------------------------------------------------------------
# P14: aggregate admission + ASYMMETRIC sign fixture (no year is ever fitted)
# --------------------------------------------------------------------------

FIXTURE_POLICIES = ("none", "P", "F1", "F2", "F3", "F4", "F5")
# modeled offsets are ASYMMETRIC on purpose: service-style endpoints reward the
# production policy (positive policy-minus-none) while the loss-style endpoint
# L_lambda_1 punishes it (negative policy-minus-none) under the SAME label.
SERVICE_OFFSET = {"none": 0.0, "P": 0.10, "F1": 0.02, "F2": 0.03, "F3": 0.04, "F4": 0.05, "F5": 0.06}
LOSS_OFFSET = {"none": 0.0, "P": -0.10, "F1": -0.02, "F2": -0.03, "F3": -0.04, "F4": -0.05, "F5": -0.06}
ACTION_DAYS = {"none": 0, "P": 30, "F1": 40, "F2": 25, "F3": 20, "F4": 35, "F5": 28}
UNCOVERED = {"none": 50.0, "P": 40.0, "F1": 45.0, "F2": 44.0, "F3": 43.0, "F4": 42.0, "F5": 41.0}
WORST7_DEFICIT = {"none": 20.0, "P": 16.0, "F1": 19.0, "F2": 18.5, "F3": 18.0, "F4": 17.5, "F5": 17.0}
ADEQUACY = {"none": 0.80, "P": 0.95, "F1": 0.82, "F2": 0.84, "F3": 0.86, "F4": 0.88, "F5": 0.90}
BASE_RESERVED = 1000.0
HEADROOM_SUM = 10.0
COMMITTED_HOURS = 24.0
FIXTURE_CONTRAST_LABELS = tuple([f"{p}_minus_none" for p in FIXTURE_POLICIES if p != "none"]
                                + [f"P_minus_{p}" for p in FIXTURE_POLICIES
                                   if p not in ("none", "P")])
FIXTURE_MODELED_ENDPOINTS = ("service_worst3", "service_worst7", "service_worst14",
                             "mean_service", "L_lambda_1")
FIXTURE_PHYSICAL_ENDPOINTS = ("worst7_deficit_MWh_complete_days",
                              "worst7_adequacy_fraction_complete_days",
                              "uncovered_MWh_finite_hours", "extra_committed_MWh",
                              "full_cost_MWh_equivalent")


def _fixture_metrics(year: int) -> dict:
    drift = 0.002 * (year - C.TEST_YEARS[0])
    metrics = {}
    for rank, policy in enumerate(FIXTURE_POLICIES):
        service = 0.50 + drift + SERVICE_OFFSET[policy]
        metrics[policy] = {
            "mean_service": service,
            "service_worst3": service - 0.02,
            "service_worst7": service - 0.05,
            "service_worst14": service - 0.08,
            "L_lambda_1": 0.40 + LOSS_OFFSET[policy],
            "action_total_all_declared": ACTION_DAYS[policy],
            "action_rate_all_declared": ACTION_DAYS[policy] / 364.0,
            "action_total_observed": ACTION_DAYS[policy],
            "rescue_precision": 0.40 + 0.01 * rank,
            "blocked_trigger_days": rank,
            "n_missing_outcomes": 2,
        }
    return metrics


def _fixture_extra(eta: float, policy: str) -> float:
    return float(eta) * HEADROOM_SUM * COMMITTED_HOURS * ACTION_DAYS[policy]


def _fixture_full_cost(eta: float, kappa: float, policy: str) -> float:
    return float(kappa) * (BASE_RESERVED + _fixture_extra(eta, policy)) + UNCOVERED[policy]


def _fixture_grid() -> list[dict]:
    cells: list[dict] = []
    for policy in FIXTURE_POLICIES:
        for eta in C.ETA_GRID:
            booked = _fixture_extra(eta, policy)
            uncovered = UNCOVERED[policy]
            for kappa in C.COST_RATIO_GRID:
                cells.append({
                    "policy": policy, "eta": float(eta), "kappa": float(kappa),
                    "base_reserved_MWh": BASE_RESERVED,
                    "extra_committed_MWh": booked,
                    "uncovered_MWh_finite_hours": uncovered,
                    "uncovered_MWh_complete_days": uncovered,
                    "full_cost_MWh_equivalent": _fixture_full_cost(eta, kappa, policy),
                    "incremental_cost_MWh_equivalent": float(kappa) * booked + uncovered,
                    "mean_observed_adequate_fraction_complete_days": ADEQUACY[policy],
                    "worst7_deficit_MWh_complete_days": WORST7_DEFICIT[policy],
                    "worst7_adequacy_fraction_complete_days": ADEQUACY[policy],
                    "action_days": ACTION_DAYS[policy],
                    "observed_hours_total": 364 * 24.0,
                    "expected_hours_total": 364 * 24.0,
                    "complete_day_support": 364,
                    "declared_day_support": 364,
                })
    return cells


def _fixture_delta(kind: str, endpoint: str, left: str, right: str,
                   eta: float = 1.0, kappa: float = 0.0) -> float:
    """Independent left-minus-right value recomputed from the fixture constants."""
    if kind == "modeled":
        table = {"service_worst3": (-0.02, 0.50), "service_worst7": (-0.05, 0.50),
                 "service_worst14": (-0.08, 0.50), "mean_service": (0.0, 0.50),
                 "L_lambda_1": (None, 0.40)}
        if endpoint == "L_lambda_1":
            return LOSS_OFFSET[left] - LOSS_OFFSET[right]
        adj, base = table[endpoint]
        return (base + SERVICE_OFFSET[left] + adj) - (base + SERVICE_OFFSET[right] + adj)
    value = {"worst7_deficit_MWh_complete_days": WORST7_DEFICIT,
             "worst7_adequacy_fraction_complete_days": ADEQUACY,
             "uncovered_MWh_finite_hours": UNCOVERED,
             "extra_committed_MWh": {p: _fixture_extra(eta, p) for p in FIXTURE_POLICIES},
             "full_cost_MWh_equivalent": {p: _fixture_full_cost(eta, kappa, p)
                                          for p in FIXTURE_POLICIES}}[endpoint]
    return value[left] - value[right]


def _write_fixture_unit(out_root: Path, year: int, fingerprint: str, raw: Path) -> None:
    import r2_run as R
    origin_dir = out_root / f"origin_{year}"
    origin_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "card10.segan_r2.origin.synthetic_fixture",
               "fixture": True, "origin": year, "test_year": year,
               "allocator_metrics": _fixture_metrics(year),
               "bridge_grid": _fixture_grid(),
               "environment": {"identities": R._identity_values(), "fixture": True}}
    payload["canonical_payload_sha256"] = C.canonical_payload_sha256(payload)
    C.write_json(origin_dir / "origin.json", payload)
    stub = pd.DataFrame({"delivery_date": [f"{year}-01-02"], "value": [1.0], "segment": [0]})
    for name in R.REQUIRED_ARTIFACTS:
        if name == "origin.json":
            continue
        if name == "kernel_meta.json":
            # a synthetic fixture: the REAL wrapper hash, no fabricated git probe
            C.write_json(origin_dir / name, {"backend": "synthetic_fixture", "fixture": True,
                                             "wrapper_sha256": SB.AST_KERNEL_SHA, "git": None})
            continue
        C.write_gzip_csv(stub, origin_dir / name)
    environment = {"identities": R._identity_values(), "fixture": True,
                   "protocol": R._protocol_identity(require_sidecar=False),
                   "amendments": R._amendment_identities(require_sidecar=False)}
    manifest = C.manifest_for([origin_dir / n for n in R.REQUIRED_ARTIFACTS
                               if n != "manifest.json"],
                              origin_dir,
                              extra={"fingerprint": fingerprint, "raw_sha256": C.sha256_file(raw),
                                     "ast_wrapper_sha256": SB.AST_KERNEL_SHA,
                                     "checkout": "synthetic_fixture:none",
                                     "environment": environment, "attempt": "fixture"})
    C.write_json(origin_dir / "manifest.json", manifest)
    (origin_dir / "manifest.json.sha256").write_text(
        C.sha256_file(origin_dir / "manifest.json") + "\n", encoding="ascii")


def _mutate_fixture_unit(out_root: Path, year: int, mutate) -> None:
    path = out_root / f"origin_{year}" / "origin.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    payload["canonical_payload_sha256"] = C.canonical_payload_sha256(payload)
    C.write_json(path, payload)


def P14_aggregate_admission_and_signs() -> dict:
    import r2_run as R
    checks: dict[str, bool] = {}
    details: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out_root = tmp / "fixture_out"
        fingerprint = R._fingerprint()
        raw = tmp / "raw.csv.gz"
        C.write_gzip_csv(pd.DataFrame({"timestamp_utc": ["fixture"], "x": [1.0]}), raw)
        for year in C.TEST_YEARS:
            _write_fixture_unit(out_root, year, fingerprint, raw)
        summary = R.aggregate(out_root, raw=raw, checkout=None, fixture=True)
        checks["fixture_admitted_complete"] = bool(summary.get("status") == "COMPLETE")
        checks["fixture_all_units_admitted"] = bool(
            all(u["status"] == "ADMITTED" for u in summary.get("unit_status", [])))
        checks["fixture_grid_complete"] = bool(summary.get("grid_complete")
                                               and summary.get("grid_cells") == 7 * 4 * 9 * 7)
        checks["fixture_labels_recorded"] = bool(
            any("fixture mode" in label for label in summary.get("labels", [])))
        # ---- modeled label signs and values (independently recomputed)
        modeled = {(c["contrast"], c["endpoint"]): c for c in summary.get("contrasts", [])}
        checks["modeled_contrast_count"] = bool(len(modeled) == len(FIXTURE_CONTRAST_LABELS)
                                                * len(FIXTURE_MODELED_ENDPOINTS))
        sign_ok, value_ok = True, True
        for label in FIXTURE_CONTRAST_LABELS:
            left = label.split("_minus_")[0]
            right = label.split("_minus_")[1]
            for endpoint in FIXTURE_MODELED_ENDPOINTS:
                entry = modeled.get((label, endpoint))
                if entry is None:
                    sign_ok = value_ok = False
                    continue
                expect = _fixture_delta("modeled", endpoint, left, right)
                sign_ok &= bool(np.sign(entry["effect"]) == np.sign(expect))
                value_ok &= bool(abs(entry["effect"] - expect) <= 1e-12)
        checks["modeled_signs_match_independent_recompute"] = bool(sign_ok)
        checks["modeled_values_match_independent_recompute"] = bool(value_ok)
        checks["modeled_sign_is_asymmetric"] = bool(
            modeled.get(("P_minus_none", "service_worst7"), {}).get("effect", 0) > 0
            and modeled.get(("P_minus_none", "L_lambda_1"), {}).get("effect", 0) < 0)
        # ---- physical label signs and values across ALL eta/kappa cells
        physical = {(c["contrast"], c["endpoint"], c["eta"], c["kappa"]): c
                    for c in summary.get("physical_contrasts", [])}
        checks["physical_contrast_count"] = bool(
            len(physical) == len(FIXTURE_CONTRAST_LABELS) * len(C.ETA_GRID)
            * len(C.COST_RATIO_GRID) * len(FIXTURE_PHYSICAL_ENDPOINTS))
        phys_sign_ok, phys_value_ok = True, True
        for (label, endpoint, eta, kappa), entry in physical.items():
            left = label.split("_minus_")[0]
            right = label.split("_minus_")[1]
            expect = _fixture_delta("physical", endpoint, left, right, eta=eta, kappa=kappa)
            phys_sign_ok &= bool(np.sign(entry["effect"]) == np.sign(expect))
            phys_value_ok &= bool(abs(entry["effect"] - expect) <= 1e-9)
        checks["physical_signs_match_independent_recompute"] = bool(phys_sign_ok)
        checks["physical_values_match_independent_recompute"] = bool(phys_value_ok)
        flip_lo = physical.get(("P_minus_none", "full_cost_MWh_equivalent", 1.0, 0.0), {}).get("effect")
        flip_hi = physical.get(("P_minus_none", "full_cost_MWh_equivalent", 1.0, 0.5), {}).get("effect")
        checks["physical_sign_flips_with_kappa"] = bool(flip_lo is not None and flip_hi is not None
                                                       and flip_lo < 0 < flip_hi)
        checks["physical_p_minus_fi_sign_negative"] = bool(
            physical.get(("P_minus_F1", "worst7_deficit_MWh_complete_days", 1.0, 0.0),
                         {}).get("effect", 0) < 0)
        details["kappa_flip"] = {"kappa_0": flip_lo, "kappa_0.5": flip_hi}
        # ---- negative controls: refusal with reasons, never an exception
        def run_aggregate() -> dict:
            return R.aggregate(out_root, raw=raw, checkout=None, fixture=True)

        origin_json = out_root / f"origin_{C.TEST_YEARS[0]}" / "origin.json"
        good_bytes = origin_json.read_bytes()
        try:
            origin_json.write_text("{ this is not json", encoding="utf-8")
            bad = run_aggregate()
            reasons = json.dumps(bad.get("invalid_origins", []))
            checks["malformed_json_recorded_invalid"] = bool(
                bad.get("status") == "INCOMPLETE" and "malformed origin.json" in reasons)
        finally:
            origin_json.write_bytes(good_bytes)
        try:
            _mutate_fixture_unit(out_root, C.TEST_YEARS[0],
                                 lambda p: p.update(test_year=C.TEST_YEARS[0] + 1))
            bad = run_aggregate()
            reasons = json.dumps(bad.get("invalid_origins", []))
            checks["year_mismatch_recorded_invalid"] = bool(
                bad.get("status") == "INCOMPLETE" and "!= directory year" in reasons)
        finally:
            origin_json.write_bytes(good_bytes)
        try:
            def break_endpoint(payload):
                payload["bridge_grid"][0]["full_cost_MWh_equivalent"] = float("nan")
            _mutate_fixture_unit(out_root, C.TEST_YEARS[0], break_endpoint)
            bad = run_aggregate()
            reasons = json.dumps(bad.get("invalid_origins", []))
            checks["nonfinite_endpoint_recorded_invalid"] = bool(
                bad.get("status") == "INCOMPLETE" and "non-finite physical endpoint" in reasons)
        finally:
            origin_json.write_bytes(good_bytes)
        moved = out_root / f"origin_{C.TEST_YEARS[1]}.moved"
        try:
            (out_root / f"origin_{C.TEST_YEARS[1]}").rename(moved)
            bad = run_aggregate()
            checks["missing_unit_refuses_full_cohort"] = bool(
                bad.get("status") == "INCOMPLETE"
                and bad.get("missing_origins") == [C.TEST_YEARS[1]]
                and not bad.get("macro"))
        finally:
            moved.rename(out_root / f"origin_{C.TEST_YEARS[1]}")
        restored = run_aggregate()
        checks["fixture_complete_after_restore"] = bool(restored.get("status") == "COMPLETE")
    return {"checks": checks, "pass": all(checks.values()), "details": details}


def P15_bootstrap_draw_equivalence() -> dict:
    """The vectorized draw matrix must reproduce the pre-R03E per-draw loop
    EXACTLY at the production settings (10,000 draws, seed 20261011); this is
    the condition under which the vectorization was adopted at all."""
    diffs = np.array([0.10, -0.20, 0.35, 0.0, 0.42, -0.13, 0.27])
    loop = C.paired_origin_bootstrap_loop_reference(diffs, n_boot=C.N_BOOT, seed=C.BOOT_SEED)
    rng = np.random.default_rng(C.BOOT_SEED)
    vector = rng.choice(diffs, size=(C.N_BOOT, len(diffs)), replace=True).mean(axis=1)
    # the same equivalence the reviewer's independent summary audit asserts
    indices = np.random.default_rng(C.BOOT_SEED).integers(0, len(diffs),
                                                          size=(C.N_BOOT, len(diffs)))
    reviewer_style = diffs[indices].mean(axis=1)
    stat = C.paired_origin_bootstrap(diffs, n_boot=C.N_BOOT, seed=C.BOOT_SEED)
    checks = {
        "production_constants": bool(C.N_BOOT == 10_000 and C.BOOT_SEED == 20261011),
        "draws_bit_identical": bool(np.array_equal(loop, vector)),
        "draws_match_reviewer_index_design": bool(np.array_equal(loop, reviewer_style)
                                                  and np.array_equal(vector, reviewer_style)),
        "effect_exact": bool(stat["effect"] == float(np.mean(diffs))),
        "ci_matches_loop": bool(stat["ci_lo"] == float(np.quantile(loop, 0.025))
                                and stat["ci_hi"] == float(np.quantile(loop, 0.975))),
        "share_matches_loop": bool(stat["share_positive"] == float(np.mean(loop > 0.0))),
        "nonconstant_fixture": bool(np.unique(diffs).size == len(diffs)),
    }
    return {"checks": checks, "pass": all(checks.values()),
            "details": {"implementation": stat.get("draw_implementation"),
                        "effect": stat["effect"], "ci": [stat["ci_lo"], stat["ci_hi"]]}}


# --------------------------------------------------------------------------
# portable, persistent runner (R03E item 3)
# --------------------------------------------------------------------------

TESTS: tuple[tuple[str, object], ...] = (
    ("P1_pipeline_end_to_end", P1_pipeline_end_to_end),
    ("P2_utc_local_equivalence", P2_utc_local_equivalence),
    ("P3_perturbation_prefixes", P3_perturbation_prefixes),
    ("P4_year_seam", P4_year_seam),
    ("P5_dst_structure", P5_dst_structure),
    ("P6_missing_outcome_retained_costs", P6_missing_outcome_retained_costs),
    ("P7_manifest_tamper_and_skip", P7_manifest_tamper_and_skip),
    ("P8_support_and_windows", P8_support_and_windows),
    ("P9_bridge_arithmetic", P9_bridge_arithmetic),
    ("P10_determinism", P10_determinism),
    ("P11_theory", P11_theory),
    ("P12_missing_day_forced_action", P12_missing_day_forced_action),
    ("P13_f5_tension_independence", P13_f5_tension_independence),
    ("P14_aggregate_admission_and_signs", P14_aggregate_admission_and_signs),
    ("P15_bootstrap_draw_equivalence", P15_bootstrap_draw_equivalence),
)


def _stamp() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def run_suite(*, out_path: Path, only: tuple[str, ...] = (), label: str | None = None) -> dict:
    """Run the selected guards, persisting after EVERY test.

    Never marks an unexecuted, failed or interrupted test as passing; the
    record stays `complete: false` until the final test returns, so an
    interrupted run can never be mistaken for a full pass.
    """
    selected = [(name, fn) for name, fn in TESTS
                if not only or any(name == token or name.startswith(f"{token}_") for token in only)]
    record: dict = {
        "schema": "card10.segan_r2.guards.v4",
        "label": label, "complete": False, "full_suite": bool(not only),
        "selection": [name for name, _ in selected],
        "n_tests_defined": len(TESTS), "n_tests_selected": len(selected),
        "begin_local": _stamp(), "finished_local": None,
        "checkout": str(CHECKOUT) if CHECKOUT is not None else None,
        "checkout_source": CHECKOUT_SOURCE,
        "host": platform.node(), "python": sys.version.split()[0],
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "source_fingerprint": guard_source_fingerprint(),
        "source_fingerprint_after": None,
        "tests": {name: {"status": "NOT_RUN"} for name, _ in TESTS},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(out_path.suffix + ".log")

    def persist() -> None:
        record["updated_local"] = _stamp()
        C.write_json(out_path, record)

    def note(line: str) -> None:
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    note(f"[{_stamp()}] BEGIN guard suite label={label} "
         f"checkout={record['checkout']} ({CHECKOUT_SOURCE}) selection={record['selection']}")
    persist()
    interrupted = None
    try:
        for name, fn in selected:
            note(f"[{_stamp()}] BEGIN {name}")
            started = time.time()
            try:
                result = fn()
                status = "PASS" if result.get("pass") else "FAIL"
                entry = {"status": status, "seconds": round(time.time() - started, 2),
                         **result}
            except Exception as exc:
                entry = {"status": "ERROR", "seconds": round(time.time() - started, 2),
                         "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()[-4000:]}
            record["tests"][name] = entry
            persist()
            note(f"[{_stamp()}] END {name} status={entry['status']} "
                 f"seconds={entry['seconds']}")
    except KeyboardInterrupt:
        interrupted = "KeyboardInterrupt"
        note(f"[{_stamp()}] INTERRUPTED by user; partial results retained")
    record["interrupted"] = interrupted
    record["finished_local"] = _stamp()
    record["source_fingerprint_after"] = guard_source_fingerprint()
    record["sources_stable"] = bool(record["source_fingerprint_after"]
                                    == record["source_fingerprint"])
    record["complete"] = bool(interrupted is None and len(selected) == len(TESTS))
    record["all_selected_pass"] = bool(all(record["tests"][name]["status"] == "PASS"
                                           for name, _ in selected))
    record["all_pass"] = bool(record["complete"] and record["all_selected_pass"]
                              and record["sources_stable"])
    persist()
    note(f"[{_stamp()}] END guard suite complete={record['complete']} "
         f"all_selected_pass={record['all_selected_pass']} "
         f"sources_stable={record['sources_stable']}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="R2 guard suite (portable, persistent)")
    parser.add_argument("--checkout", default=None,
                        help="pinned SAOCP checkout (else R2_CHECKOUT, else the local default)")
    parser.add_argument("--out", default=None, help="results JSON path")
    parser.add_argument("--only", default="",
                        help="comma-separated subset (e.g. P8,P9); a subset run is labelled "
                             "and never claims to be the full suite")
    parser.add_argument("--label", default=None, help="free-text label recorded in the results")
    args = parser.parse_args()
    global CHECKOUT, CHECKOUT_SOURCE
    CHECKOUT, CHECKOUT_SOURCE = resolve_checkout(args.checkout)
    if CHECKOUT is not None and not Path(CHECKOUT).is_dir():
        print(f"[{_stamp()}] FATAL checkout not a directory: {CHECKOUT}", flush=True)
        return 2
    only = tuple(token.strip() for token in args.only.split(",") if token.strip())
    unknown = [t for t in only if not any(n == t or n.startswith(f"{t}_") for n, _ in TESTS)]
    if unknown:
        print(f"[{_stamp()}] FATAL unknown --only tokens: {unknown}", flush=True)
        return 2
    out_path = Path(args.out) if args.out else (C.R2 / "outputs" / "guards_v4" / "r2_guards_v4.json")
    record = run_suite(out_path=out_path, only=only, label=args.label)
    print(json.dumps({"complete": record["complete"], "full_suite": record["full_suite"],
                      "all_selected_pass": record["all_selected_pass"],
                      "sources_stable": record["sources_stable"],
                      "statuses": {k: v["status"] for k, v in record["tests"].items()}},
                     indent=1), flush=True)
    return 0 if record["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
