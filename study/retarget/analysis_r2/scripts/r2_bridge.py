"""R2 finite zonal balancing-reserve bridge — v3 (post R03B review).

Scope (stated once, reused everywhere): a ZONAL BALANCING SIMULATION.  For each
real UTC hour, residual_MW = actual_load_MW - issued_point_MW and the hour's
per-zone upper residual support is mu_z + sqrt(R * Sigma_zz) (radicand clipped
at 0 exactly as the producer does); the reserve requirement is
r0_z = max(0, mu_z + sqrt(R * Sigma_zz)).  A committed action day adds finite
headroom eta * H_z to every zone for the WHOLE committed day.  Uncovered
deficit max(residual_z - allowed_z, 0) is integrated with the hour's real
duration (1 h per actual UTC hour, both fall folds included).  No network
flows, no revenues, no delivered-grid reliability claim.

Units and costs (dimensionally coherent):
  * base_reserved_MWh   - Sigma_h Sigma_z r0_z(h) * duration(h); COMMON to every
                          policy and eta (the region-implied booking);
  * extra_committed_MWh - eta * Sigma_z H_z * (committed target hours); booked
                          for EVERY committed hour of an action day, including
                          hours whose actual is later missing;
  * uncovered_MWh       - scored ONLY on observed hours; missing support is
                          declared, never replaced by invented zeros;
  * full_cost(kappa)      = kappa * (base + extra) + uncovered;
  * incremental_cost(kappa) = kappa * extra + uncovered  (common base cancels).
  kappa is a transparent normalized capacity-to-deficit cost ratio.
  Worst-window DEFICIT is the MAXIMUM window deficit over date-aware windows.

H_z = 95th percentile (linear) of max(reference corrected residual_z, 0) over
the whole eligible reference year - reference-only, never test maxima.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import r2_env  # noqa: F401  (thread limits before numeric imports)

import numpy as np
import pandas as pd

import r2_common as C


def upper_support(center: pd.Series, radius: pd.DataFrame) -> pd.DataFrame:
    """mu_z + radius_z (unclipped); reserve requirement r0 = max(0, this)."""
    return center.reindex(radius.columns)[None, :] + radius


def headroom_from_reference(reference_residual: pd.DataFrame) -> pd.Series:
    positive = reference_residual.clip(lower=0.0)
    return positive.quantile(C.REFERENCE_QUANTILE, interpolation="linear")


def simulate_hours(hours: pd.DataFrame, actions: dict[str, int], eta: float,
                   headroom: pd.Series, expected_hours: dict[str, float] | None = None) -> pd.DataFrame:
    """hours columns (canonical pipeline names): delivery_date, duration_h,
    residual_MW_<z>, r0_<z>.  Returns per-day rows; deficits are scored only on
    rows with a finite residual (missing support declared, never zero-filled)."""
    zones = list(C.ZONES)
    residual = hours[[f"residual_MW_{z}" for z in zones]].to_numpy(float)
    support = hours[[f"r0_{z}" for z in zones]].to_numpy(float)
    duration = hours["duration_h"].to_numpy(float)
    day = hours["delivery_date"].to_numpy()
    obs = np.isfinite(residual).all(axis=1)
    extra = np.zeros_like(support)
    for i, d in enumerate(day):
        if actions.get(str(d), 0):
            extra[i, :] = eta * headroom.reindex(zones).to_numpy(float)
    allowed = support + extra
    deficit_mw = np.where(obs[:, None], np.maximum(residual - allowed, 0.0), 0.0)
    adequate = np.where(obs[:, None], residual <= allowed, False)
    rows: list[dict] = []
    for d in pd.unique(day):
        mask = day == d
        plannable = float(expected_hours.get(str(d), float(duration[mask].sum()))
                          if expected_hours else float(duration[mask].sum()))
        committed = plannable if int(actions.get(str(d), 0)) else 0.0
        obs_mask = mask & obs                      # finite-hours only (R03B review 7)
        observed_hours = float(duration[obs_mask].sum())
        complete = bool(abs(observed_hours - plannable) <= 1e-9 and plannable > 0)
        adequate_hours = float((adequate[obs_mask].all(axis=1) * duration[obs_mask]).sum())
        rows.append({
            "delivery_date": str(d),
            "expected_hours": plannable,
            "observed_hours": observed_hours,
            "complete_observed_day": complete,
            "action": int(actions.get(str(d), 0)),
            "deficit_MWh": float((deficit_mw[obs_mask].sum(axis=1) * duration[obs_mask]).sum()),
            "adequate_hours": adequate_hours,
            "observed_adequate_fraction": (float(adequate_hours / observed_hours)
                                           if observed_hours else np.nan),
            "extra_committed_MWh": float(eta * headroom.reindex(zones).sum() * committed),
        })
    return pd.DataFrame(rows).sort_values("delivery_date").reset_index(drop=True)


def grid_summary(day_frames: dict[tuple[str, float], pd.DataFrame],
                 base_reserved_MWh: float,
                 segments_by_date: dict[str, int] | None = None) -> list[dict]:
    """Every (policy, eta, kappa) cell is reported; no cell is selected.

    Physical windows use COMPLETE observed days only, require consecutive
    calendar dates AND the same decision-time segment (passed in by the
    caller).  Worst-window DEFICIT is the MAXIMUM window deficit.  Bookings on
    committed days are retained even when the actual is incomplete; deficits
    and adequacy are never scored on missing hours.
    """
    if segments_by_date is None:
        raise ValueError("segments_by_date is required (no silent all-zero default)")
    out: list[dict] = []
    for (policy, eta), frame in sorted(day_frames.items()):
        complete = frame["complete_observed_day"].to_numpy(bool)
        sub = frame[complete].reset_index(drop=True)
        deficit_all = frame["deficit_MWh"].to_numpy(float)
        booked = float(frame["extra_committed_MWh"].sum())
        uncovered_finite = float(deficit_all.sum())
        dates = sub["delivery_date"].tolist()
        segments = np.asarray([segments_by_date[d] for d in dates], dtype=int)
        windows = C.consecutive_local_windows(dates, segments, 7) if len(dates) else []
        worst7_deficit = (float(max(sub["deficit_MWh"].to_numpy(float)[w].sum() for w in windows))
                          if windows else None)
        adequacy = sub["observed_adequate_fraction"].to_numpy(float) if len(sub) else np.array([])
        worst7_adequacy = (float(min(np.nanmean(adequacy[w]) for w in windows)) if windows else None)
        for kappa in C.COST_RATIO_GRID:
            out.append({
                "policy": policy, "eta": float(eta), "kappa": float(kappa),
                "base_reserved_MWh": float(base_reserved_MWh),
                "extra_committed_MWh": booked,
                "uncovered_MWh_finite_hours": uncovered_finite,
                "uncovered_MWh_complete_days": float(sub["deficit_MWh"].sum()) if len(sub) else 0.0,
                "full_cost_MWh_equivalent": float(kappa * (base_reserved_MWh + booked) + uncovered_finite),
                "incremental_cost_MWh_equivalent": float(kappa * booked + uncovered_finite),
                "mean_observed_adequate_fraction_complete_days": (float(np.nanmean(adequacy))
                                                                  if len(adequacy) else None),
                "worst7_deficit_MWh_complete_days": worst7_deficit,
                "worst7_adequacy_fraction_complete_days": worst7_adequacy,
                "action_days": int(frame["action"].sum()),
                "observed_hours_total": float(frame["observed_hours"].sum()),
                "expected_hours_total": float(frame["expected_hours"].sum()),
                "complete_day_support": int(complete.sum()),
                "declared_day_support": int(len(frame)),
            })
    return out
