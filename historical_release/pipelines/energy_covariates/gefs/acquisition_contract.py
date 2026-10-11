"""Route and identity helpers for the two-era GEFS acquisition."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from . import source_layout as R71

OLD_LAST_DELIVERY = dt.date(2020, 9, 23)
OLD_SOURCES = tuple(R71.ALL_MEMBER_TOKENS)
NEW_SOURCES = ("geavg", "gespr")
VARS = tuple(R71.OFFICIAL_VARS)


@dataclass(frozen=True)
class Route:
    stratum: str
    sources: tuple[str, ...]
    aggregation: str
    product: str


def route_for_delivery_day(day: dt.date) -> Route:
    if day <= OLD_LAST_DELIVERY:
        return Route(
            stratum="old_era",
            sources=OLD_SOURCES,
            aggregation="mean_and_population_std_from_all_21_members",
            product="NOAA/NCEP GEFS pgrb2a native 1p00",
        )
    return Route(
        stratum="new_era",
        sources=NEW_SOURCES,
        aggregation="official_geavg_and_gespr",
        product="NOAA/NCEP GEFS pgrb2a official geavg/gespr 0p50",
    )


def source_role(day: dt.date, source: str) -> str:
    route = route_for_delivery_day(day)
    if source not in route.sources:
        raise ValueError(f"source {source!r} is not admitted for {day} / {route.stratum}")
    if route.stratum == "old_era":
        return "ensemble_member"
    return "ensemble_mean" if source == "geavg" else "ensemble_spread"


def object_key(day: dt.date, source: str, lead: int) -> str:
    route = route_for_delivery_day(day)
    source_role(day, source)
    cycle_ds = (day - dt.timedelta(days=1)).strftime("%Y%m%d")
    return R71.object_key_for_lead(route.stratum, cycle_ds, source, lead)


def object_url(day: dt.date, source: str, lead: int) -> str:
    return f"{R71.BUCKET_URL}/{object_key(day, source, lead)}"


def expected_cell_count(day: dt.date) -> int:
    route = route_for_delivery_day(day)
    return len(route.sources) * len(R71.admitted_native_leads(day))
