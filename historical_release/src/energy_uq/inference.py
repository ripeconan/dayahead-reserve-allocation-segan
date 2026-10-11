"""Small statistical utilities used by the reproducibility scripts."""

from __future__ import annotations

import math
from collections.abc import Sequence


def linear_quantile(values: Sequence[float], probability: float) -> float:
    """Return the linearly interpolated sample quantile.

    This matches the default quantile convention used to create the retained
    bootstrap summaries.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def empirical_two_sided_pvalue(null_draws: Sequence[float], observed: float) -> float:
    """Return the finite-sample corrected two-sided empirical p-value."""
    if not null_draws:
        raise ValueError("null_draws must not be empty")
    extreme = sum(abs(float(draw)) >= abs(float(observed)) for draw in null_draws)
    return (extreme + 1.0) / (len(null_draws) + 1.0)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""
    count = len(pvalues)
    indexed = sorted(enumerate(float(p) for p in pvalues), key=lambda item: item[1])
    adjusted = [1.0] * count
    running = 0.0
    for rank, (original_index, pvalue) in enumerate(indexed):
        running = max(running, (count - rank) * pvalue)
        adjusted[original_index] = min(1.0, running)
    return adjusted

