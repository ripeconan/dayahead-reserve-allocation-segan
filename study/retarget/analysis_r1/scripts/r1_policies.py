"""R-A deterministic allocator policies over one origin's daily trace.

All policies see only information eligible at the day-ahead cutoff:
same-segment records with delivery date u such that u + 2 calendar days <= t
(service values are post-action, i.e. the closed loop), plus the frozen region
quantities (thresholds) where a rule declares them.  No policy reads day-t or
later outcomes, nor future missingness/segment ends.

State machines for the production rule and the thresholded-count K2 variant
mirror the admitted implementations (hashes pinned in the R1 protocol):

  production  <-> run_production_transfer_mrp02._calendar_production
  R51 K2      <-> run_production_transfer_mrp02._r51_variant
  A*-bank     <-> confirm_runner.cocd_known_horizon_cap_registered

Pure module: no side effects on import.
"""
from __future__ import annotations

from bisect import bisect_right, insort
from typing import Sequence

import numpy as np
import pandas as pd

from r1_common import (
    BUDGET_FRAC,
    ELL,
    K2_COUNT,
    K2_SERVICE_THRESHOLD,
    MEMORY,
    PERSIST_THRESHOLD,
    THETA,
)


def origin_cap(n_days: int) -> int:
    """Frozen cap convention: round(0.15 * n_days) (Python round, as admitted)."""
    return int(round(BUDGET_FRAC * n_days))


def normalize_segments(segments: Sequence[int]) -> np.ndarray:
    """Relabel segment ids to 0..G-1 in chronological order (pure relabeling)."""
    out = np.zeros(len(segments), dtype=int)
    mapping: dict[int, int] = {}
    for i, value in enumerate(segments):
        key = int(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        out[i] = mapping[key]
    return out


def _eligible_indices(history_dates: Sequence[pd.Timestamp], history_segments: Sequence[int],
                      day: pd.Timestamp, segment: int) -> list[int]:
    return [
        i for i in range(len(history_dates))
        if history_segments[i] == segment and history_dates[i] + pd.Timedelta(days=ELL) <= day
    ]


def production_drb(q: np.ndarray, segments: np.ndarray, dates: Sequence[pd.Timestamp],
                   cap: int) -> tuple[np.ndarray, list[dict]]:
    """Frozen production shortfall-sum controller (exact replay of admitted code)."""
    actions = np.zeros(len(q), dtype=bool)
    served = np.zeros(len(q), dtype=float)
    history: list[tuple[pd.Timestamp, float, int]] = []
    used = 0
    trace: list[dict] = []
    for t in range(len(q)):
        date, base_value, segment = dates[t], float(q[t]), int(segments[t])
        eligible = [i for i in range(len(history))
                    if history[i][2] == segment and history[i][0] + pd.Timedelta(days=ELL) <= date]
        recent = eligible[-MEMORY:]
        state = float(sum(1.0 - history[i][1] for i in recent))
        before = used
        action = bool(state >= THETA and used < cap)
        blocked = bool(state >= THETA and used >= cap)
        if action:
            used += 1
        served_value = 1.0 if action else base_value
        actions[t] = action
        served[t] = served_value
        history.append((date, served_value, segment))
        trace.append({"h": state, "used_before": before, "used_after": used, "blocked": blocked})
    return actions, trace


def k2_thresholded_count(q: np.ndarray, segments: np.ndarray, dates: Sequence[pd.Timestamp],
                         cap: int) -> tuple[np.ndarray, list[dict]]:
    """R51 thresholded-count K2 variant (exact replay of admitted code)."""
    actions = np.zeros(len(q), dtype=bool)
    served = np.zeros(len(q), dtype=float)
    history: list[tuple[pd.Timestamp, float, int]] = []
    used = 0
    trace: list[dict] = []
    for t in range(len(q)):
        date, base_value, segment = dates[t], float(q[t]), int(segments[t])
        eligible = [i for i in range(len(history))
                    if history[i][2] == segment and history[i][0] + pd.Timedelta(days=ELL) <= date]
        recent = eligible[-MEMORY:]
        count = int(sum(history[i][1] < K2_SERVICE_THRESHOLD for i in recent))
        before = used
        action = bool(count >= K2_COUNT and used < cap)
        blocked = bool(count >= K2_COUNT and used >= cap)
        if action:
            used += 1
        served_value = 1.0 if action else base_value
        actions[t] = action
        served[t] = served_value
        history.append((date, served_value, segment))
        trace.append({"count": count, "used_before": before, "used_after": used, "blocked": blocked})
    return actions, trace


def pace7(q: np.ndarray, segments: np.ndarray, dates: Sequence[pd.Timestamp],
          cap: int) -> tuple[np.ndarray, list[dict]]:
    """F1: fixed 7-day pacing schedule within each segment.

    Segment-local day indices 6, 13, 20, ... (0-based) are scheduled action
    days, fixed in advance and independent of outcomes.  A scheduled day is
    skipped (no action, schedule unchanged) when the horizon cap is exhausted.
    Segments shorter than 7 days receive no scheduled action.  The schedule
    restarts at each segment.
    """
    actions = np.zeros(len(q), dtype=bool)
    used = 0
    trace: list[dict] = []
    local_index = -1
    previous_segment: int | None = None
    for t in range(len(q)):
        segment = int(segments[t])
        if previous_segment is None or segment != previous_segment:
            local_index = 0
        else:
            local_index += 1
        previous_segment = segment
        scheduled = (local_index >= 6) and ((local_index - 6) % 7 == 0)
        skipped = bool(scheduled and used >= cap)
        action = bool(scheduled and used < cap)
        if action:
            used += 1
        actions[t] = action
        trace.append({"local_index": local_index, "scheduled": bool(scheduled),
                      "cap_skipped": skipped, "used_after": used})
    return actions, trace


def pace_bank(q: np.ndarray, segments: np.ndarray, dates: Sequence[pd.Timestamp],
              cap: int) -> tuple[np.ndarray, list[dict]]:
    """F2: credit-bank pacing with exact integer arithmetic (accrual 15, cost 100).

    The bank accrues 15 credit units at every evaluated decision day; an action
    costs 100 units and requires used < cap.  A new segment resets the bank to
    zero (residual credit is dropped); the horizon-wide `used` counter never
    resets.  No outcome information is used.
    """
    actions = np.zeros(len(q), dtype=bool)
    used = 0
    bank = 0
    dropped_total = 0
    trace: list[dict] = []
    previous_segment: int | None = None
    for t in range(len(q)):
        segment = int(segments[t])
        dropped = 0
        if previous_segment is None or segment != previous_segment:
            if previous_segment is not None and bank > 0:
                dropped = bank
                dropped_total += bank
            bank = 0
        previous_segment = segment
        bank += 15
        cap_skipped = bool(bank >= 100 and used >= cap)
        action = bool(bank >= 100 and used < cap)
        if action:
            bank -= 100
            used += 1
        actions[t] = action
        trace.append({"bank": bank, "dropped_on_reset": dropped,
                      "cap_skipped": cap_skipped, "used_after": used})
    return actions, trace


def persist1(q: np.ndarray, segments: np.ndarray, dates: Sequence[pd.Timestamp],
             cap: int) -> tuple[np.ndarray, list[dict]]:
    """F4: act when the most recent eligible day's post-action service <= 0.5."""
    actions = np.zeros(len(q), dtype=bool)
    served = np.zeros(len(q), dtype=float)
    history: list[tuple[pd.Timestamp, float, int]] = []
    used = 0
    trace: list[dict] = []
    for t in range(len(q)):
        date, base_value, segment = dates[t], float(q[t]), int(segments[t])
        eligible = [i for i in range(len(history))
                    if history[i][2] == segment and history[i][0] + pd.Timedelta(days=ELL) <= date]
        last_service = history[eligible[-1]][1] if eligible else None
        triggered = bool(last_service is not None and last_service <= PERSIST_THRESHOLD)
        action = bool(triggered and used < cap)
        if action:
            used += 1
        served_value = 1.0 if action else base_value
        actions[t] = action
        served[t] = served_value
        history.append((date, served_value, segment))
        trace.append({"last_eligible_service": (-1.0 if last_service is None else float(last_service)),
                      "triggered": triggered, "used_after": used})
    return actions, trace


def tension_from_thresholds(threshold_frame: pd.DataFrame, dates: Sequence[pd.Timestamp],
                            segments: np.ndarray) -> np.ndarray:
    """Per-day mean -log(p) tension from frozen pre-action thresholds.

    Hourly variant (NYISO): per-hour p = (1 + #{history <= threshold})/(1+n),
    then day value = mean over hours of -log(p); history resets at segment
    changes and updates only after the day's decision.  Mirrors
    confirm_runner.tension_from_thresholds.
    """
    by_date: dict[pd.Timestamp, list[tuple[int, float]]] = {}
    for row in threshold_frame.itertuples(index=False):
        by_date.setdefault(pd.Timestamp(row.delivery_date), []).append((int(row.hour), float(row.threshold)))
    history: dict[int, list[float]] = {}
    result: list[float] = []
    previous_segment: int | None = None
    for date, segment in zip(dates, segments):
        segment = int(segment)
        if previous_segment is not None and segment != previous_segment:
            history = {}
        previous_segment = segment
        states = sorted(by_date[pd.Timestamp(date)], key=lambda item: item[0])
        p_values = []
        for hour, threshold in states:
            hist = history.setdefault(hour, [])
            p_values.append((1 + bisect_right(hist, threshold)) / (1 + len(hist)))
        result.append(float(np.mean(-np.log(np.asarray(p_values, dtype=float)))))
        for hour, threshold in states:
            insort(history.setdefault(hour, []), threshold)
    values = np.asarray(result, dtype=float)
    if len(values) != len(dates) or not np.isfinite(values).all() or np.any(values < 0):
        raise RuntimeError("invalid pre-action tension")
    return values


def tension_from_daily_thresholds(thresholds: Sequence[float], segments: np.ndarray) -> np.ndarray:
    """Daily variant (transfer cohorts): one threshold per day."""
    history: list[float] = []
    result: list[float] = []
    previous_segment: int | None = None
    for threshold, segment in zip(thresholds, segments):
        segment = int(segment)
        if previous_segment is not None and segment != previous_segment:
            history = []
        previous_segment = segment
        p = (1 + bisect_right(history, float(threshold))) / (1 + len(history))
        result.append(float(-np.log(p)))
        insort(history, float(threshold))
    values = np.asarray(result, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise RuntimeError("invalid daily pre-action tension")
    return values


def bank1_causal(shadow_actions: Sequence[bool], segments: Sequence[int], tension: Sequence[float],
                 dates: Sequence[pd.Timestamp], cap: int, registered_end: str) -> tuple[np.ndarray, list[dict]]:
    """F5: causal one-credit bank with a declared-calendar-end flush.

    Exact state machine of confirm_runner.cocd_known_horizon_cap_registered
    (recovery A*), with one deviation that is recorded, never silent: if the
    per-origin bank is still open at the final evaluated day (the trace ends
    before the declared end), the open credit is DROPPED at the trace end and
    the flag ``bank_open_at_trace_end`` is set instead of raising.  No synthetic
    action is inserted.
    """
    shadow = np.asarray(shadow_actions, dtype=bool)
    segment = np.asarray(segments, dtype=int)
    state = np.asarray(tension, dtype=float)
    day = pd.to_datetime(list(dates), errors="raise")
    if not (len(shadow) == len(segment) == len(state) == len(day)) or len(day) == 0:
        raise ValueError("bank1 inputs differ in length")
    if cap != round(BUDGET_FRAC * len(day)) or int(shadow.sum()) > cap:
        raise ValueError("bank1 shared cap mismatch")
    end_ts = pd.Timestamp(registered_end)
    action = np.zeros(len(day), dtype=bool)
    bank = 0
    history: list[float] = []
    trace: list[dict] = []
    dropped_on_gap_total = 0
    for index in range(len(day)):
        reset = index == 0 or segment[index] != segment[index - 1]
        dropped = 0
        if reset:
            if index and bank:
                dropped = 1
                dropped_on_gap_total += 1
            bank = 0
            history = []
        remaining_calendar_days = int((end_ts - day[index].normalize()).days + 1)
        flush_zone = remaining_calendar_days <= cap + 1
        tail_rank = (1 + sum(value >= state[index] for value in history)) / (1 + len(history))
        high = tail_rank <= 0.5
        mint = int(shadow[index])
        before = bank
        if bank:
            action[index] = True
            bank = mint
            reason = "spend_outstanding_and_relay" if mint else "spend_outstanding_and_close"
        elif not mint:
            reason = "no_mint"
        elif high or flush_zone:
            action[index] = True
            reason = "immediate_high" if high else "immediate_registered_horizon_flush"
        else:
            bank = 1
            reason = "defer_low"
        trace.append({
            "day_index": index,
            "delivery_date": day[index].date().isoformat(),
            "segment_id": int(segment[index]),
            "tension": float(state[index]),
            "tail_rank": float(tail_rank),
            "high_tension": bool(high),
            "remaining_registered_calendar_days": remaining_calendar_days,
            "flush_zone": bool(flush_zone),
            "shadow_mint": mint,
            "bank_before": before,
            "candidate_action": int(action[index]),
            "bank_after": bank,
            "dropped_credit_on_observed_gap": dropped,
            "decision_reason": reason,
        })
        history.append(float(state[index]))
    bank_open_at_trace_end = bool(bank != 0)
    if int(action.sum()) > cap:
        raise RuntimeError("bank1 exceeds shared cap")
    return action, trace, {
        "bank_open_at_trace_end": bank_open_at_trace_end,
        "dropped_credit_on_observed_gap_total": dropped_on_gap_total,
    }
