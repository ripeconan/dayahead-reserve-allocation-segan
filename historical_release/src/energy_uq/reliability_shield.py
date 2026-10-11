"""Strict-lag-2 reliability controller used in the operational replay.

The controller observes served shortfall only after the declared feedback lag.
At each decision time it sums the six most recent eligible shortfalls from the
current segment. When the sum reaches the fixed threshold and budget remains,
an action closes a predeclared fraction of the current base shortfall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence


def action_cap(n_days: int, budget_fraction: float = 0.15) -> int:
    """Return the precommitted horizon-level action cap."""
    if n_days < 0:
        raise ValueError("n_days must be nonnegative")
    if not 0.0 <= budget_fraction <= 1.0:
        raise ValueError("budget_fraction must lie in [0, 1]")
    return int(round(budget_fraction * n_days))


@dataclass(frozen=True)
class DayResult:
    """Controller output for one delivery day."""

    index: int
    segment: Hashable
    base_coverage: float
    action: bool
    service: float
    eligible_shortfall_sum: float
    revealed_indices: tuple[int, ...]
    actions_used: int
    action_cap: int


class StrictLag2ReliabilityShield:
    """Causal shortfall-sum controller with segment-local history."""

    def __init__(
        self,
        n_days: int,
        *,
        budget_fraction: float = 0.15,
        feedback_lag: int = 2,
        lookback: int = 6,
        shortfall_threshold: float = 1.0,
        action_effectiveness: float = 1.0,
    ) -> None:
        if feedback_lag < 1:
            raise ValueError("feedback_lag must be positive")
        if lookback < 1:
            raise ValueError("lookback must be positive")
        if shortfall_threshold < 0.0:
            raise ValueError("shortfall_threshold must be nonnegative")
        if not 0.0 <= action_effectiveness <= 1.0:
            raise ValueError("action_effectiveness must lie in [0, 1]")
        self.feedback_lag = feedback_lag
        self.lookback = lookback
        self.shortfall_threshold = shortfall_threshold
        self.action_effectiveness = action_effectiveness
        self.cap = action_cap(n_days, budget_fraction)
        self.actions_used = 0
        self._history: list[tuple[int, float, Hashable]] = []

    def step(self, index: int, base_coverage: float, segment: Hashable) -> DayResult:
        """Process one day without accessing same-day or lag-1 outcomes."""
        coverage = float(base_coverage)
        if not 0.0 <= coverage <= 1.0:
            raise ValueError("base_coverage must lie in [0, 1]")
        if self._history and index <= self._history[-1][0]:
            raise ValueError("indices must be strictly increasing")

        eligible = [
            (u, service)
            for u, service, source_segment in self._history
            if source_segment == segment and u + self.feedback_lag <= index
        ]
        recent = eligible[-self.lookback :]
        shortfall_sum = sum(max(0.0, 1.0 - service) for _, service in recent)

        should_act = (
            shortfall_sum >= self.shortfall_threshold
            and self.actions_used < self.cap
        )
        if should_act:
            self.actions_used += 1
        service = (
            coverage + self.action_effectiveness * (1.0 - coverage)
            if should_act
            else coverage
        )
        self._history.append((index, service, segment))

        return DayResult(
            index=index,
            segment=segment,
            base_coverage=coverage,
            action=should_act,
            service=service,
            eligible_shortfall_sum=float(shortfall_sum),
            revealed_indices=tuple(u for u, _ in recent),
            actions_used=self.actions_used,
            action_cap=self.cap,
        )

    def replay(
        self, base_coverage: Sequence[float], segments: Sequence[Hashable]
    ) -> list[DayResult]:
        """Replay a complete horizon in chronological order."""
        if len(base_coverage) != len(segments):
            raise ValueError("base_coverage and segments must have equal length")
        return [
            self.step(index, coverage, segment)
            for index, (coverage, segment) in enumerate(zip(base_coverage, segments))
        ]


def rolling_worst_service(
    service: Sequence[float], segments: Sequence[Hashable], window: int
) -> float:
    """Return the minimum complete-window mean without crossing segments."""
    if len(service) != len(segments):
        raise ValueError("service and segments must have equal length")
    if window < 1:
        raise ValueError("window must be positive")
    means: list[float] = []
    for end in range(window, len(service) + 1):
        segment_window = segments[end - window : end]
        if len(set(segment_window)) == 1:
            means.append(sum(service[end - window : end]) / window)
    if not means:
        raise ValueError("no complete within-segment window")
    return min(means)
