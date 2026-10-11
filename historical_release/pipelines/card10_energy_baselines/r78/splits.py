"""Chronological split helpers for R78 development and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .constants import (
    D1_GEFS_DEV_YEAR,
    D1_GEFS_EVAL_YEARS,
    EVAL_YEARS,
    HYPERPARAM_DEV_END,
    HYPERPARAM_DEV_START,
    ORIGINS,
    STRATUM_D1_GEFS,
    STRATUM_PRIMARY,
)
from .firewall import FirewallViolation


@dataclass(frozen=True)
class OriginSpec:
    reference_year: int
    test_year: int
    label: str
    stratum: str

    @property
    def refit_end_exclusive(self) -> str:
        """Timestamps strictly before this may be used to refit for the test year."""
        return f"{self.test_year}-01-01"


def primary_origins() -> tuple[OriginSpec, ...]:
    return tuple(
        OriginSpec(ref, test, f"{ref}->{test}", STRATUM_PRIMARY)
        for ref, test in ORIGINS
    )


def d1_gefs_origins() -> tuple[OriginSpec, ...]:
    # Development year 2019 only; evaluation 2020-2022 as separate sensitivity.
    return tuple(
        OriginSpec(year - 1, year, f"{year - 1}->{year}", STRATUM_D1_GEFS)
        for year in D1_GEFS_EVAL_YEARS
    )


def mask_period(index: pd.DatetimeIndex, start: str, end: str) -> pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if getattr(index, "tz", None) is not None:
        # Compare in local timezone calendar.
        local = index.tz_convert("America/New_York").tz_localize(None)
        start_ts = start_ts.tz_localize(None) if start_ts.tzinfo else start_ts
        end_ts = end_ts.tz_localize(None) if end_ts.tzinfo else end_ts
        # Inclusive end of calendar day.
        end_exclusive = end_ts + pd.Timedelta(days=1)
        return pd.Series((local >= start_ts) & (local < end_exclusive), index=index)
    end_exclusive = end_ts + pd.Timedelta(days=1)
    return pd.Series((index >= start_ts) & (index < end_exclusive), index=index)


def development_mask(index: pd.DatetimeIndex, stratum: str = STRATUM_PRIMARY) -> pd.Series:
    if stratum == STRATUM_PRIMARY:
        return mask_period(index, HYPERPARAM_DEV_START, HYPERPARAM_DEV_END)
    if stratum == STRATUM_D1_GEFS:
        return mask_period(index, f"{D1_GEFS_DEV_YEAR}-01-01", f"{D1_GEFS_DEV_YEAR}-12-31")
    raise ValueError(f"unknown stratum {stratum}")


def evaluation_year_mask(index: pd.DatetimeIndex, year: int) -> pd.Series:
    if year == 2026:
        raise FirewallViolation("evaluation year 2026 is forbidden")
    return mask_period(index, f"{year}-01-01", f"{year}-12-31")


def refit_mask_before_test_year(index: pd.DatetimeIndex, test_year: int) -> pd.Series:
    """Timestamps strictly before Y-01-01 for refitting selected hyperparameters."""
    if test_year == 2026:
        raise FirewallViolation("test year 2026 is forbidden")
    boundary = pd.Timestamp(f"{test_year}-01-01")
    if getattr(index, "tz", None) is not None:
        local = index.tz_convert("America/New_York").tz_localize(None)
        return pd.Series(local < boundary, index=index)
    return pd.Series(index < boundary, index=index)


def assert_no_test_year_in_selection(
    selection_index: pd.DatetimeIndex,
    test_years: tuple[int, ...] = EVAL_YEARS,
) -> None:
    years = set(int(y) for y in selection_index.year)
    leaked = years.intersection(set(test_years))
    if leaked:
        raise FirewallViolation(f"test-year leakage into selection: {sorted(leaked)}")


def early_stopping_validation_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Last 90 calendar days of 2018 only (primary stratum)."""
    return mask_period(index, "2018-10-03", "2018-12-31")


def split_contract_summary() -> dict:
    return {
        "primary_development": {
            "start": HYPERPARAM_DEV_START,
            "end": HYPERPARAM_DEV_END,
        },
        "primary_evaluation_years": list(EVAL_YEARS),
        "primary_origins": [o.label for o in primary_origins()],
        "d1_gefs_development_year": D1_GEFS_DEV_YEAR,
        "d1_gefs_evaluation_years": list(D1_GEFS_EVAL_YEARS),
        "d1_gefs_origins": [o.label for o in d1_gefs_origins()],
        "early_stopping_validation": "2018-10-03 through 2018-12-31",
        "refit_rule": "timestamps strictly before test_year-01-01",
    }
