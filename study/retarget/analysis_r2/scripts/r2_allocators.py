"""R2 allocators over the DECLARED calendar — v3 (post R03B review).

ONE shared, date-causal availability/reset event list drives the UQ adapter and
every allocator (and the F5 rank history):

  * {"kind": "issuance_absent", "date": d} - known at the cutoff for d: reset
    IMMEDIATELY at d; d has no decision and no score ever (a day without a
    complete pre-action issuance cannot be scored);
  * {"kind": "missing_outcome", "date": u} - discovered at the reveal date
    u + 2: reset (and drop pending old-segment feedback) at that decision.

Every queue entry carries its SOURCE segment; a reset discards pending entries
from the old segment, so old feedback is never relabeled into a new segment.
Commitments on days that later prove unscorable are retained (budget spent).

Each policy walks in its own pass with a METHOD-OWNED served history; F5 takes
the production (P) action path as its shadow mints plus the per-day tension.
"""
from __future__ import annotations

from bisect import insort

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import r2_env  # noqa: F401  (thread limits before numeric imports)

import numpy as np
import pandas as pd

import r2_common as C

POLICIES = ("none", "P", "F1", "F2", "F3", "F4", "F5")


def build_events(scorable: dict, observed: dict, test_year: int) -> list[dict]:
    """The single shared event list (sorted by the date the event takes effect)."""
    events: list[dict] = []
    for date_str in [d.date().isoformat() for d in C.declared_calendar(test_year)]:
        if not scorable.get(date_str, False):
            events.append({"kind": "issuance_absent", "date": date_str, "effect_date": date_str})
        elif not observed.get(date_str, False):
            reveal = (pd.Timestamp(date_str) + pd.DateOffset(days=C.FEEDBACK_LAG_DAYS)).date().isoformat()
            events.append({"kind": "missing_outcome", "date": date_str, "effect_date": reveal})
    return sorted(events, key=lambda e: (e["effect_date"], e["date"]))


def walk_declared_calendar(test_year: int, scorable: dict, observed: dict,
                           q_base: dict, policy: str, cap: int,
                           events: list[dict], tension: dict | None = None,
                           shadow_actions: dict | None = None) -> pd.DataFrame:
    calendar = [d.date().isoformat() for d in C.declared_calendar(test_year)]
    by_effect: dict[str, list[dict]] = {}
    for event in events:
        if event["effect_date"] in calendar or True:
            by_effect.setdefault(event["effect_date"], []).append(event)
    records: list[dict] = []
    queue: list[dict] = []
    history: list[tuple[str, float, int]] = []
    segment = -1
    seg_started = False
    seg_index = -1
    used = 0
    bank = 0
    rank_history: list[float] = []

    def reset(reason: str, date_str: str) -> None:
        nonlocal segment, seg_started, seg_index, history, bank, rank_history
        queue[:] = [q for q in queue if q["segment"] != segment]
        segment += 1
        seg_started = True
        seg_index = -1
        history = []
        bank = 0
        rank_history = []

    for date_str in calendar:
        d = pd.Timestamp(date_str)
        event_here = [e for e in by_effect.get(date_str, []) if e["kind"] == "issuance_absent"]
        for _ in event_here:
            reset("issuance_absent_known_at_cutoff", date_str)
        reveal_resets = [e for e in by_effect.get(date_str, []) if e["kind"] == "missing_outcome"]
        # release feedback due at this decision, then apply reveal-date resets
        due = [q for q in queue if pd.Timestamp(q["source_local_date"])
               + pd.DateOffset(days=C.FEEDBACK_LAG_DAYS) <= d]
        for item in sorted(due, key=lambda q: q["source_utc"]):
            if item["segment"] == segment and item["service"] is not None:
                history.append((item["source_local_date"], float(item["service"]), item["segment"]))
        queue[:] = [q for q in queue if q not in due]
        for _ in reveal_resets:
            reset("missing_outcome_discovered_at_reveal", date_str)
        scorable_d = bool(scorable.get(date_str, False))
        action = False
        blocked = 0
        cap_skipped = 0
        state = None
        if scorable_d:
            if not seg_started:
                segment = 0
                seg_started = True
                seg_index = -1
            seg_index += 1
            eligible = [(u, s) for (u, s, g) in history
                        if g == segment and pd.Timestamp(u) + pd.DateOffset(days=C.FEEDBACK_LAG_DAYS) <= d]
            recent = eligible[-C.MEMORY:]
            if policy == "P":
                state = float(sum(1.0 - s for _, s in recent))
                if state >= C.THETA:
                    action = used < cap
                    blocked = int(not action)
            elif policy == "F3":
                state = int(sum(1 for _, s in recent if s < C.K2_SERVICE_THRESHOLD))
                if state >= C.K2_COUNT:
                    action = used < cap
                    blocked = int(not action)
            elif policy == "F4":
                state = recent[-1][1] if recent else None
                if state is not None and state <= C.PERSIST_THRESHOLD:
                    action = used < cap
            elif policy == "F1":
                scheduled = (seg_index >= 6) and ((seg_index - 6) % 7 == 0)
                state = int(scheduled)
                if scheduled:
                    action = used < cap
                    cap_skipped = int(not action)
            elif policy == "F2":
                bank += 15
                state = int(bank)
                if bank >= 100:
                    if used < cap:
                        action = True
                        bank -= 100
                    else:
                        cap_skipped = 1
            elif policy == "F5":
                if tension is None or shadow_actions is None:
                    raise ValueError("F5 requires tension and production shadow actions")
                mint = int(bool(shadow_actions.get(date_str, False)))
                remaining = (pd.Timestamp(f"{test_year}-12-31") - d).days + 1
                flush = remaining <= cap + 1
                current = float(tension.get(date_str, 0.0))
                tail_rank = (1 + sum(1 for v in rank_history if v >= current)) / (1 + len(rank_history))
                high = tail_rank <= 0.5
                state = {"mint": mint, "tail_rank": tail_rank, "flush": bool(flush)}
                if bank:
                    action = True
                    bank = mint
                elif not mint:
                    pass
                elif high or flush:
                    action = True
                else:
                    bank = 1
                insort(rank_history, current)
            else:
                raise ValueError(policy)
            if action:
                used += 1
        observed_d = bool(observed.get(date_str, False))
        service = None
        if scorable_d and observed_d:
            service = 1.0 if action else float(q_base[date_str])
        records.append({
            "delivery_date": date_str, "scorable": scorable_d, "observed": observed_d,
            "segment": segment, "action": int(action), "service": service,
            "q_base": float(q_base[date_str]) if (scorable_d and date_str in q_base) else None,
            "blocked": blocked, "cap_skipped": cap_skipped, "state": state,
            "used_after": used if scorable_d else None,
            "event_here": ";".join(sorted(e["kind"] for e in by_effect.get(date_str, []))),
        })
        if scorable_d:
            queue.append({"source_local_date": date_str, "service": service,
                          "segment": segment, "source_utc": date_str})
    return pd.DataFrame(records)


def walk_all(test_year: int, scorable: dict, observed: dict, q_base: dict,
             events: list[dict], tension: dict | None = None) -> dict[str, pd.DataFrame]:
    cap = C.planned_cap(test_year)
    walks = {policy: walk_declared_calendar(test_year, scorable, observed, q_base,
                                            policy, cap, events)
             for policy in ("P", "F1", "F2", "F3", "F4")}
    # modeled no-action baseline: an EXPLICIT zero-accounting walk with the
    # same event-derived segments/support (R03C late review item 3)
    walks["none"] = walk_none(test_year, scorable, observed, q_base, events)
    shadow = dict(zip(walks["P"]["delivery_date"], walks["P"]["action"].astype(bool)))
    walks["F5"] = walk_declared_calendar(test_year, scorable, observed, q_base, "F5", cap,
                                         events, tension=tension or {}, shadow_actions=shadow)
    return walks


def walk_none(test_year: int, scorable: dict, observed: dict, q_base: dict,
              events: list[dict]) -> pd.DataFrame:
    """Zero-action baseline: action/blocked/cap_skipped/used are exactly zero,
    state is None, and the segments follow the SAME shared event list as every
    other policy (no inherited production state)."""
    calendar = [d.date().isoformat() for d in C.declared_calendar(test_year)]
    by_effect: dict[str, list[dict]] = {}
    for event in events:
        by_effect.setdefault(event["effect_date"], []).append(event)
    records: list[dict] = []
    segment = -1
    seg_started = False
    for date_str in calendar:
        for event in [e for e in by_effect.get(date_str, []) if e["kind"] == "issuance_absent"]:
            segment += 1
            seg_started = True
        for event in [e for e in by_effect.get(date_str, []) if e["kind"] == "missing_outcome"]:
            segment += 1
            seg_started = True
        scorable_d = bool(scorable.get(date_str, False))
        if scorable_d and not seg_started:
            segment = 0
            seg_started = True
        observed_d = bool(observed.get(date_str, False))
        service = float(q_base[date_str]) if (scorable_d and observed_d and date_str in q_base) else None
        records.append({
            "delivery_date": date_str, "scorable": scorable_d, "observed": observed_d,
            "segment": segment, "action": 0, "service": service,
            "q_base": float(q_base[date_str]) if (scorable_d and date_str in q_base) else None,
            "blocked": 0, "cap_skipped": 0, "state": None,
            "used_after": 0 if scorable_d else None,
            "event_here": ";".join(sorted(e["kind"] for e in by_effect.get(date_str, []))),
        })
    return pd.DataFrame(records)


def metrics_for(walk: pd.DataFrame) -> dict:
    """Modeled-support metrics: a day counts only when scorable AND observed
    AND q is finite.  Undefined q is NEVER filled with 0 (R03B review item 5:
    a forecast-only gap with available actual must not become a fabricated
    miss).  All-declared action counts stay separate."""
    q = np.asarray([v if v is not None else np.nan for v in walk["q_base"]], dtype=float)
    actions = walk["action"].to_numpy(bool)
    scorable = walk["scorable"].to_numpy(bool)
    observed_raw = walk["observed"].to_numpy(bool)
    support = scorable & observed_raw & np.isfinite(q)
    dates = walk["delivery_date"].tolist()
    segments = walk["segment"].to_numpy(int)
    row = C.observable_metrics(np.where(np.isfinite(q), q, 0.0), actions, support, dates, segments)
    row.update({
        "n_modeled_support_days": int(support.sum()),
        "blocked_trigger_days": int(walk["blocked"].sum()),
        "cap_skipped_days": int(walk["cap_skipped"].sum()),
        "n_segments": int(pd.unique(segments).size),
        "n_missing_outcomes": int((scorable & ~observed_raw).sum()),
        "n_issuance_absent": int((~scorable).sum()),
        "service_defined_days": int(walk["service"].notna().sum()),
        "action_total_modeled_support": int(actions[support].sum()),
        "action_rate_modeled_support": (float(actions[support].sum() / support.sum())
                                        if support.sum() else float("nan")),
    })
    return row
