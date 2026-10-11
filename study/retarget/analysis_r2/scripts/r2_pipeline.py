"""R2 production pipeline — v3 (post R03B review; single source of truth).

Both the formal runner (`r2_run.py`) and the guard suite (`r2_guards.py`) call
these functions; no test-only stand-ins exist for production checks.

Guarantees implemented here:
  * full expected UTC calendar (23/24/25 real hours per local date) built before
    any data support is consulted; issuance absence is a KNOWN-at-cutoff event,
    outcome absence a reveal-date event, both handled by the shared date-causal
    event list;
  * issuance requires complete pre-action inputs (official forecast + 72/168 h
    lagged errors + calendar features) and does NOT depend on the current or
    future actual; scoring requires a complete day of actuals; q is MISSING
    (never a fabricated miss) when either is incomplete;
  * geometry is the official sklearn OAS (version recorded) fitted on the
    reference-year corrected residuals; per-hour radii use the producer's
    radicand clipping sqrt(max(R*Sigma_zz, 0)); the upper residual support is
    mu_z + radius_z and the reserve requirement clamps it at 0;
  * every artifact required for reviewer-side reconstruction is persisted and
    bound in an external manifest; resume rejects missing/modified artifacts.
"""
from __future__ import annotations

import platform
import sys
import time
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

import r2_allocators as A  # noqa: E402
import r2_bridge as B  # noqa: E402
import r2_common as C  # noqa: E402
import r2_pointcorr as PC  # noqa: E402
import r2_saocp_batch as SB  # noqa: E402

POLICIES = A.POLICIES


def sklearn_info() -> dict:
    import sklearn
    from sklearn.covariance import OAS
    return {"sklearn_version": sklearn.__version__, "estimator": "sklearn.covariance.OAS",
            "settings": {"assume_centered": False},
            "doc": "official OAS empirical covariance (no manual reimplementation)"}


def oas_geometry(reference_residual: pd.DataFrame) -> dict:
    from sklearn.covariance import OAS
    x = reference_residual.dropna().to_numpy(float)
    # the formal grid always provides the full prior reference year; synthetic
    # fixtures must still provide a declared minimum (14 days x 24 h)
    if x.shape[0] < 336 or not np.isfinite(x).all():
        raise RuntimeError(f"reference support invalid: rows={x.shape[0]} (min 336)")
    estimator = OAS(assume_centered=False).fit(x)
    center = x.mean(axis=0)
    covariance = np.asarray(estimator.covariance_, dtype=float)
    if not np.isfinite(covariance).all() or not np.isfinite(center).all():
        raise RuntimeError("non-finite OAS geometry")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if float(eigenvalues.min()) <= 0.0:
        raise RuntimeError(f"non-positive covariance eigenvalue {eigenvalues.min()}")
    return {"center": center, "covariance": covariance,
            "shrinkage": float(estimator.shrinkage_),
            "n_reference_rows": int(x.shape[0]),
            "eigen_min": float(eigenvalues.min()), **sklearn_info()}


def mahalanobis(residual: pd.DataFrame, geometry: dict) -> np.ndarray:
    x = residual.to_numpy(float) - geometry["center"]
    precision = np.linalg.inv(geometry["covariance"])
    return np.einsum("ij,jk,ik->i", x, precision, x)


def expected_utc_hours(test_year: int) -> dict[str, pd.DatetimeIndex]:
    """Full expected UTC hour grid per declared local date (23/24/25 hours)."""
    grid = pd.date_range(f"{test_year}-01-02 00:00", f"{test_year}-12-31 23:00",
                         freq="h", tz=C.TZ)
    per_day: dict[str, list[pd.Timestamp]] = {}
    for ts in grid:
        key = ts.tz_convert(C.TZ).normalize().tz_localize(None).date().isoformat()
        per_day.setdefault(key, []).append(ts)
    return {k: pd.DatetimeIndex(v) for k, v in per_day.items()}


def normalize_input_frames(actual: pd.DataFrame, forecast: pd.DataFrame, *,
                           tz: str = C.TZ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Instant-preserving input-index normalization (R03F repair of the native P2 defect).

    Index contracts inspected before this change:
      * production `r2_run.load_raw` builds a New_York index
        (``pd.to_datetime(..., utc=True).dt.tz_convert(C.TZ)``);
      * guard P2 / the reviewer capture legitimately hold the SAME instants in UTC;
      * `r2_pointcorr.build_features` already refuses a naive index.
    The grid construction needs one timezone, so a UTC-indexed input previously
    crashed in ``pd.date_range`` ("Start and end cannot both be tz-aware with
    different timezones") before any fit.  Normalization preserves instants, the
    full pre-reference training history and genuine missing rows; it never drops,
    fills or reorders data, and it never mutates the caller's frames (copies are
    returned).  Naive indexes are refused because their instants are ambiguous.
    """
    record: dict = {"target_tz": tz, "normalized": False}
    normalized_frames: list[pd.DataFrame] = []
    for name, frame in (("actual", actual), ("forecast", forecast)):
        index = frame.index
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError(f"{name}: index must be a pandas DatetimeIndex, "
                             f"got {type(index).__name__}")
        if index.tz is None:
            raise ValueError(f"{name}: naive DatetimeIndex is not admissible; pass a "
                             f"timezone-aware index (UTC or {tz}) so instants are unambiguous")
        converted = index.tz_convert(tz)
        record[f"{name}_input_tz"] = str(index.tz)
        record[f"{name}_rows"] = int(len(index))
        if converted.duplicated().any():
            raise ValueError(f"{name}: duplicate timestamps are not admissible in production")
        if not converted.is_monotonic_increasing:
            raise ValueError(f"{name}: index not sorted")
        normalized = frame.copy()
        normalized.index = converted
        normalized_frames.append(normalized)
    actual_n, forecast_n = normalized_frames
    if not actual_n.index.equals(forecast_n.index):
        raise ValueError("actual/forecast index mismatch after timezone normalization")
    record.update(normalized=bool(record["actual_input_tz"] != tz
                                  or record["forecast_input_tz"] != tz),
                  input_first=str(actual_n.index.min()),
                  input_last=str(actual_n.index.max()))
    return actual_n, forecast_n, record


def run_origin_from_frames(actual: pd.DataFrame, forecast: pd.DataFrame, test_year: int,
                           *, kernel_factory=None, kernel_meta: dict | None = None,
                           allow_neutral_scale: bool = False) -> dict:
    """Full pipeline over in-memory frames; returns the artifact dictionary."""
    t0 = time.time()
    reference_year = test_year - 1
    # ---- normalize BOTH input indexes to one timezone (instant-preserving)
    # BEFORE any grid construction; the caller's frames are never mutated
    actual, forecast, input_record = normalize_input_frames(actual, forecast)
    # ---- full UTC grid over ALL admitted history .. test end (pre-reference
    # history retained: early reference-year point fits need it), with NaN reindex
    input_index = actual.index
    grid = pd.date_range(input_index.min(), pd.Timestamp(f"{test_year}-12-31 23:59:59", tz=C.TZ),
                         freq="h")
    out_of_range_rows = int((~input_index.isin(grid)).sum())
    actual = actual.reindex(grid)
    forecast = forecast.reindex(grid)
    grid_rows_missing = int((~grid.isin(input_index)).sum())
    expected = expected_utc_hours(test_year)
    features, errors = PC.build_features(actual, forecast)
    corr, audit = PC.forward_correction_calendar(features, errors,
                                                 start=f"{reference_year}-01-01",
                                                 end=f"{test_year}-12-31")
    issued, residual = PC.issued_point_and_residual(forecast, corr, actual)
    # reference support = the EXACT prior calendar year (never narrowed to fit
    # a fixture; fixtures must supply enough earlier history instead)
    reference_start = pd.Timestamp(f"{reference_year}-01-01", tz=C.TZ)
    reference_end = pd.Timestamp(f"{reference_year}-12-31 23:59:59", tz=C.TZ)
    reference_mask = (errors.index >= reference_start) & (errors.index <= reference_end)
    geometry = oas_geometry(residual.loc[reference_mask])
    reference_residual_all = residual.loc[reference_mask].dropna()
    ref_scores_series = pd.Series(mahalanobis(reference_residual_all, geometry),
                                  index=reference_residual_all.index)
    # timestamped reference artifacts (score values AND corrected residuals/points)
    reference_scores = pd.DataFrame({
        "utc": ref_scores_series.index.astype(str),
        "local_date": ref_scores_series.index.tz_convert(C.TZ).date.astype(str),
        "slot": ref_scores_series.index.tz_convert(C.TZ).hour,
        "score": ref_scores_series.to_numpy(float),
    })
    reference_residuals = residual.loc[reference_residual_all.index].copy()
    reference_residuals.insert(0, "utc", reference_residuals.index.astype(str))
    reference_residuals.insert(1, "local_date",
                               reference_residuals.index.tz_convert(C.TZ).date.astype(str))
    reference_residuals.insert(2, "slot", reference_residuals.index.tz_convert(C.TZ).hour)
    issued_reference = issued.loc[reference_residual_all.index].copy()
    for z in C.ZONES:
        reference_residuals[f"issued_point_MW_{z}"] = issued_reference[z].to_numpy(float)
    first_test_decision = pd.Timestamp(f"{test_year}-01-02", tz=C.TZ)
    newest_reference_outcome = reference_residual_all.index.max()
    if newest_reference_outcome > first_test_decision - pd.Timedelta(days=C.FEEDBACK_LAG_DAYS) + pd.Timedelta(hours=23):
        raise RuntimeError("reference outcome newer than first decision minus two local days")
    prefix_by_slot = {h: ref_scores_series[ref_scores_series.index.tz_convert(C.TZ).hour == h]
                      .to_numpy(float).tolist() for h in range(24)}
    # ---- availability decisions (from data support, never from a future mask)
    scorable: dict[str, bool] = {}
    observed: dict[str, bool] = {}
    partial_actual_days: list[str] = []
    for date_str, hours in expected.items():
        issued_mask = np.isfinite(issued.loc[hours].to_numpy(float)).all(axis=1) \
            if len(hours) else np.array([], dtype=bool)
        actual_mask = np.isfinite(actual.loc[hours].to_numpy(float)).all(axis=1) \
            if len(hours) else np.array([], dtype=bool)
        scorable[date_str] = bool(len(hours) and issued_mask.all())
        observed[date_str] = bool(len(hours) and actual_mask.all())
        if len(hours) and actual_mask.any() and not actual_mask.all():
            partial_actual_days.append(date_str)
    events = A.build_events(scorable, observed, test_year)
    if kernel_factory is None:
        raise ValueError("kernel_factory required (production passes the pinned kernel factory)")
    adapter = SB.CalendarBatchSaocp(prefix_by_slot, kernel_factory, origin=str(test_year),
                                    allow_neutral_scale=allow_neutral_scale)
    # ---- daily loop: shared events, frozen thresholds, complete-day scores
    q_value: dict[str, float] = {}
    tension: dict[str, float] = {}
    threshold_history: dict[int, list] = {}
    hour_rows: list[dict] = []
    daily_rows: list[dict] = []
    events_by_effect: dict[str, list[dict]] = {}
    for event in events:
        events_by_effect.setdefault(event["effect_date"], []).append(event)
    all_hours = pd.DatetimeIndex(sorted(set().union(*[set(v) for v in expected.values()])))
    local_index = all_hours.tz_convert(C.TZ)
    from bisect import insort
    for date_str in [d.date().isoformat() for d in C.declared_calendar(test_year)]:
        hours = expected[date_str]
        for event in events_by_effect.get(date_str, []):
            adapter.reset(date_str, event["kind"], kind=event["kind"])
            threshold_history = {}                 # cleared at the same shared resets
        if scorable[date_str]:
            thresholds = adapter.decision(date_str)
            covered: list[float] = []
            scores_for_day: list[tuple] = []
            for ts in hours:
                slot = ts.tz_convert(C.TZ).hour
                row = residual.loc[ts]
                score = float(mahalanobis(pd.DataFrame([row]), geometry)[0]) if np.isfinite(row).all() else np.nan
                threshold = thresholds[slot]
                if np.isfinite(score):
                    covered.append(float(score <= threshold))
                    scores_for_day.append((ts, slot, score))
                hour_rows.append(_hour_row(ts, date_str, slot, actual, issued, residual, geometry,
                                           threshold))
            # tension + pre-action threshold history on EVERY issuable date,
            # independent of whether today's actual exists (R03B review item 4)
            p_values = []
            for slot in sorted(thresholds):
                hist = threshold_history.setdefault(slot, [])
                thr = thresholds[slot]
                p_values.append((1 + sum(1 for v in hist if v <= thr)) / (1 + len(hist)))
            tension[date_str] = float(np.mean(-np.log(np.asarray(p_values, dtype=float))))
            for slot in sorted(thresholds):
                insort(threshold_history.setdefault(slot, []), thresholds[slot])
            if observed[date_str]:
                q_value[date_str] = float(np.mean(covered)) if covered else np.nan
                for ts, slot, score in scores_for_day:
                    adapter.enqueue(ts, date_str, slot, score)
        daily_rows.append({"delivery_date": date_str, "scorable": scorable[date_str],
                           "observed": observed[date_str],
                           "q": q_value.get(date_str),
                           # PRE-ACTION tension (computed from thresholds known at
                           # this decision, before today's actions); persisted so
                           # P13 can compare the numeric series, not just a claim
                           "tension": tension.get(date_str)})
    walks = A.walk_all(test_year, scorable, observed,
                       {k: v for k, v in q_value.items() if np.isfinite(v)},
                       events, tension=tension)
    metrics = {p: A.metrics_for(w) for p, w in walks.items()}
    hours_frame = pd.DataFrame(hour_rows)
    segments_by_date = dict(zip(walks["P"]["delivery_date"], walks["P"]["segment"]))
    hours_frame["segment"] = hours_frame["delivery_date"].map(segments_by_date)
    daily_frame = pd.DataFrame(daily_rows)
    daily_frame["segment"] = daily_frame["delivery_date"].map(segments_by_date)
    headroom = B.headroom_from_reference(reference_residual_all)
    expected_hours = {k: float(len(v)) for k, v in expected.items()}
    actions_by_policy = {p: dict(zip(w["delivery_date"], w["action"])) for p, w in walks.items()}
    day_frames = {}
    for policy, actions in actions_by_policy.items():
        for eta in C.ETA_GRID:
            day_frames[(policy, eta)] = B.simulate_hours(hours_frame, actions, eta, headroom,
                                                         expected_hours=expected_hours)
    base_reserved = float(np.nansum(hours_frame[[f"r0_{z}" for z in C.ZONES]].to_numpy(float)
                                    .sum(axis=1) * hours_frame["duration_h"].to_numpy(float)))
    bridge_grid = B.grid_summary(day_frames, base_reserved_MWh=base_reserved,
                                 segments_by_date=segments_by_date)
    bridge_daily_parts = []
    for (policy, eta), frame in sorted(day_frames.items()):
        part = frame.copy()
        part.insert(0, "policy", policy)
        part.insert(1, "eta", float(eta))
        part["segment"] = part["delivery_date"].map(segments_by_date)
        bridge_daily_parts.append(part)
    bridge_daily = pd.concat(bridge_daily_parts, ignore_index=True)
    # common finite-hour observation support across policies on the same base
    support_keys = {(p, e): int(f["observed_hours"].sum()) for (p, e), f in day_frames.items()}
    support_values = set(support_keys.values())
    if len(support_values) != 1:
        raise RuntimeError(f"finite-hour support differs across policies: {sorted(support_values)[:3]}")
    issued_dates = sorted(d for d, ok in scorable.items() if ok)
    scored_dates = sorted(d for d, ok in observed.items() if ok and scorable.get(d))
    provenance = {"issued_rows": PC.provenance_report(pd.DatetimeIndex(issued.sort_index().dropna().index)),
                  "all_rows": PC.provenance_report(errors.index)}
    return {
        "schema": "card10.segan_r2.origin.v3",
        "origin": test_year, "test_year": test_year, "reference_year": reference_year,
        "expected_hours_per_day_counts": {k: int(len(v)) for k, v in expected.items()},
        "n_expected_hours": int(len(all_hours)),
        "reference_span": {"start": str(reference_start), "end": str(reference_end)},
        "n_scorable_days": int(sum(scorable.values())),
        "n_observed_days": int(sum(observed.values())),
        "geometry": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in geometry.items()},
        "prefix_info": {str(k): v for k, v in adapter.prefix_info.items()},
        "scales": {str(k): float(v) for k, v in adapter.scales.items()},
        "adapter_reset_events": adapter.reset_events,
        "adapter_decisions": adapter.decisions,
        "allocator_metrics": metrics,
        "bridge_headroom_MW": {z: float(headroom[z]) for z in C.ZONES},
        "bridge_base_reserved_MWh": base_reserved,
        "bridge_grid": bridge_grid,
        "grid_rows_missing": grid_rows_missing,
        "out_of_range_rows_payload": out_of_range_rows,
        "partial_actual_days": partial_actual_days,
        "n_issued_dates": len(issued_dates),
        "n_scored_dates": len(scored_dates),
        "issued_dates_sha256": C.sha256_text("|".join(issued_dates)),
        "scored_dates_sha256": C.sha256_text("|".join(scored_dates)),
        "point_provenance": provenance,
        "prefix_dates": {
            str(slot): {"first_utc": str(ref_scores_series[ref_scores_series.index.tz_convert(C.TZ).hour == slot].index.min()) if len(v) else None,
                        "last_utc": str(ref_scores_series[ref_scores_series.index.tz_convert(C.TZ).hour == slot].index.max()) if len(v) else None,
                        "n": len(v),
                        "values_sha256": C.sha256_text(",".join(f"{x:.12g}" for x in v))}
            for slot, v in prefix_by_slot.items()},
        "reference_scores_sha256": C.sha256_text(reference_scores.to_csv(index=False)),
        "reference_residuals_sha256": C.sha256_text(reference_residuals.to_csv(index=False)),
        "newest_reference_outcome_utc": str(newest_reference_outcome),
        "out_of_range_rows": out_of_range_rows,
        # executed input timezone / instant-preserving normalization record (R03F)
        "input_normalization": input_record,
        "kernel_meta": kernel_meta or {},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        **sklearn_info()},
        "seconds": round(time.time() - t0, 3),
    }, {"hours": hours_frame, "daily": daily_frame, "walks": walks,
        "reference_scores": reference_scores, "reference_residuals": reference_residuals,
        "bridge_daily": bridge_daily,
        "point_audit": audit, "adapter_ledger": pd.DataFrame(adapter.ledger),
        "adapter_thresholds": pd.DataFrame(
            [{"delivery_date": d, "slot": s, "threshold": t}
             for (d, s), t in sorted(adapter.thresholds.items())])}


def _hour_row(ts, date_str, slot, actual, issued, residual, geometry, threshold) -> dict:
    """Payload row (reviewer-proposed fast path; zone order and finite checks
    identical to the previous per-zone indexing construction)."""
    zones = list(C.ZONES)
    if list(actual.columns) != zones or list(issued.columns) != zones or list(residual.columns) != zones:
        raise ValueError("hourly payload zone order mismatch")
    a = actual.loc[ts].to_numpy(float)
    p = issued.loc[ts].to_numpy(float)
    e = residual.loc[ts].to_numpy(float)
    mu = np.asarray(geometry["center"], dtype=float)
    rad = np.sqrt(np.maximum(float(threshold) * np.diag(geometry["covariance"]), 0.0))
    support = mu + rad
    row = {"utc": str(ts), "delivery_date": date_str, "slot": slot, "duration_h": 1.0,
           "actual_load_MW": float(a.sum()) if np.isfinite(a).all() else np.nan,
           "issued_point_MW": float(p.sum()) if np.isfinite(p).all() else np.nan,
           "residual_MW": float(e.sum()) if np.isfinite(e).all() else np.nan,
           "threshold": float(threshold)}
    for i, z in enumerate(zones):
        row.update({f"actual_load_MW_{z}": float(a[i]) if np.isfinite(a[i]) else np.nan,
                    f"issued_point_MW_{z}": float(p[i]) if np.isfinite(p[i]) else np.nan,
                    f"residual_MW_{z}": float(e[i]) if np.isfinite(e[i]) else np.nan,
                    f"geometry_center_MW_{z}": float(mu[i]), f"radius_MW_{z}": float(rad[i]),
                    f"upper_residual_support_MW_{z}": float(support[i]),
                    f"r0_{z}": float(max(0.0, support[i]))})
    return row


def synthetic_frames(test_year: int, seed: int = 11, missing_days: tuple[str, ...] = (),
                     missing_actual_days: tuple[str, ...] = (),
                     include_reference_days: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic raw-like frames (reference span + test year) for guard runs.

    include_reference_days=None gives the whole prior reference year (formal
    shape); an integer gives that many days ending at the Jan-2 start (kept
    >= 1000 hours so the pipeline's training-row requirement is exercised, not
    bypassed).
    """
    if include_reference_days is None:
        start = pd.Timestamp(f"{test_year - 1}-01-01", tz=C.TZ)
    else:
        start = pd.Timestamp(f"{test_year}-01-02", tz=C.TZ) - pd.Timedelta(days=include_reference_days)
    end = pd.Timestamp(f"{test_year}-12-31 23:59:59", tz=C.TZ)
    idx = pd.date_range(start, end, freq="h")
    rng = np.random.default_rng(seed)
    forecast = pd.DataFrame(rng.normal(100.0, 20.0, size=(len(idx), 11)),
                            index=idx, columns=list(C.ZONES))
    actual = forecast + rng.normal(0.0, 8.0, size=(len(idx), 11))
    local = idx.tz_convert(C.TZ).normalize().tz_localize(None)
    for day in missing_days:
        mask = local == pd.Timestamp(day)
        forecast.loc[mask, :] = np.nan  # issuance inputs absent
    for day in missing_actual_days:
        mask = local == pd.Timestamp(day)
        actual.loc[mask, :] = np.nan
    return actual, forecast
