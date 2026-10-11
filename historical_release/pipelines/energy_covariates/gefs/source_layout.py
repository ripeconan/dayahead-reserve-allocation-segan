"""Shared NOAA GEFS source-layout and delivery-time logic.

All time/Eastern-UTC/DST, version-stratum, lead-grid, nearest-grid and idx-parsing
logic lives here so it is identical across acquire, transform and the tests.  Network
access is confined to acquire.py.

The constants define the two historical product strata used by the experiments.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

# --- Official source identity -------------------------------------------------
BUCKET_HOST = "noaa-gefs-pds.s3.amazonaws.com"
BUCKET_URL = f"https://{BUCKET_HOST}"
OFFICIAL_PRODUCT_URL = (
    "https://www.ncei.noaa.gov/products/weather-climate-models/global-ensemble-forecast"
)

# --- Grid (frozen spatial lock v2: per-stratum native grids) -----------------
# old_era GEFS pgrb2a is natively 1.0-deg (360x181); the 0.5-deg old_era products
# (pgrb2ap5/pgrb2sp5/pgrb2bp5) are empty on the archive. new_era is 0.5-deg (720x361).
# Grids are never pooled without an explicit stratum field.
GRID_BY_STRATUM = {
    "old_era": dict(Nx=360, Ny=181, la1=90.0, lo1=0.0, la2=-90.0, lo2=359.0, di=1.0, dj=1.0),
    "new_era": dict(Nx=720, Ny=361, la1=90.0, lo1=0.0, la2=-90.0, lo2=359.5, di=0.5, dj=0.5),
}
# Back-compat single-grid constants (default = new_era) for callers/tests that
# address the 0.5-deg grid by name.  Use grid_geom(stratum) for per-stratum work.
GRID_NX = GRID_BY_STRATUM["new_era"]["Nx"]
GRID_NY = GRID_BY_STRATUM["new_era"]["Ny"]
GRID_LA1 = GRID_BY_STRATUM["new_era"]["la1"]
GRID_LO1 = GRID_BY_STRATUM["new_era"]["lo1"]
GRID_DI = GRID_BY_STRATUM["new_era"]["di"]
GRID_DJ = GRID_BY_STRATUM["new_era"]["dj"]
GRID_LA2 = GRID_BY_STRATUM["new_era"]["la2"]
GRID_LO2 = GRID_BY_STRATUM["new_era"]["lo2"]


def grid_geom(stratum: str) -> dict:
    g = GRID_BY_STRATUM[stratum]
    return dict(g)

# --- Members (frozen feature lock) -------------------------------------------
CONTROL_TOKEN = "gec00"
PERTURBED_TOKENS = [f"gep{i:02d}" for i in range(1, 21)]
ALL_MEMBER_TOKENS = [CONTROL_TOKEN] + PERTURBED_TOKENS  # 21
DERIVED_PRODUCTS = ["geavg", "gespr"]  # excluded from member count


def product_identity(stratum: str) -> str:
    """Exact native product identity used by a stratum.

    The old archive uses the native 1-degree pgrb2a product, whereas the newer
    archive uses the 0.5-degree product. Keeping this identity in one function
    prevents the two grids from being pooled silently.
    """
    if stratum == "old_era":
        return "NOAA/NCEP GEFS pgrb2a native 1p00"
    if stratum == "new_era":
        return "NOAA/NCEP GEFS pgrb2a 0p50"
    raise ValueError(f"unknown GEFS stratum: {stratum!r}")

# --- Variables (frozen feature lock) -----------------------------------------
# official raw GRIB fields, common to both strata
OFFICIAL_VARS = [
    ("v_tmp2m", "TMP",  "2 m above ground"),
    ("v_rh2m",  "RH",   "2 m above ground"),
    ("v_u10",   "UGRD", "10 m above ground"),
    ("v_v10",   "VGRD", "10 m above ground"),
    ("v_apcp",  "APCP", "surface"),
]
DERIVED_VARS = ["v_dpt2m", "v_wspd10"]
ALL_VAR_IDS = [v[0] for v in OFFICIAL_VARS] + DERIVED_VARS
STABLE_COMMON_OFFICIAL = [v[0] for v in OFFICIAL_VARS]

VERSION_BOUNDARY = dt.date(2020, 9, 23)


# --- Time / DST ---------------------------------------------------------------
def delivery_date_to_cycle_init_utc(delivery_day: dt.date) -> dt.datetime:
    """Frozen cycle = 00Z on calendar day t-1."""
    cycle_calendar = delivery_day - dt.timedelta(days=1)
    return dt.datetime(cycle_calendar.year, cycle_calendar.month, cycle_calendar.day, 0, 0, 0, tzinfo=UTC)


def dst_label(delivery_day: dt.date) -> str:
    """'EST' or 'EDT' for the delivery day clock, via IANA America/New_York."""
    noon_local = dt.datetime(delivery_day.year, delivery_day.month, delivery_day.day, 12, 0, tzinfo=EASTERN)
    off = noon_local.utcoffset()
    return "EST" if off == dt.timedelta(hours=-5) else "EDT"


def delivery_valid_utc_window(delivery_day: dt.date) -> Tuple[dt.datetime, dt.datetime, str]:
    """Return (start_utc, end_utc_exclusive, dst_label) for the 24 delivery hours of day t."""
    label = dst_label(delivery_day)
    start_local = dt.datetime(delivery_day.year, delivery_day.month, delivery_day.day, 0, 0, tzinfo=EASTERN)
    end_local = start_local + dt.timedelta(days=1)
    return (start_local.astimezone(UTC), end_local.astimezone(UTC), label)


def delivery_hours_he(delivery_day: dt.date) -> List[Dict]:
    """HE01..HE24 hour-ending delivery hours with their valid-at UTC (hour start) and lead.

    valid-at UTC = local clock start of the hour (hour-ending convention: HE_h covers
    [(h-1):00, h:00) Eastern).  lead_hours = (valid_at_utc - cycle_init_utc) in hours.
    """
    init = delivery_date_to_cycle_init_utc(delivery_day)
    start_local = dt.datetime(delivery_day.year, delivery_day.month, delivery_day.day, 0, 0, tzinfo=EASTERN)
    rows = []
    for h in range(1, 25):
        hour_start_local = start_local + dt.timedelta(hours=h - 1)
        valid_utc = hour_start_local.astimezone(UTC)
        lead = int((valid_utc - init).total_seconds() // 3600)
        rows.append(dict(he=h, valid_utc=valid_utc, lead_hours=lead))
    return rows


# --- Version stratum ----------------------------------------------------------
def expected_stratum(delivery_day: dt.date) -> str:
    return "new_era" if delivery_day > VERSION_BOUNDARY else "old_era"


@dataclass
class StratumLayout:
    stratum: str
    # function token(date_str, cycle) -> object key prefix (without lead)
    key_prefix: str  # template with {ds} = YYYYMMDD of cycle, {mem} = member token
    file_template: str  # template with {ds},{mem},{fNN}/{fNNN}


OLD_LAYOUT = StratumLayout(
    stratum="old_era",
    key_prefix="gefs.{ds}/00/pgrb2a/",
    file_template="{mem}.t00z.pgrb2af{f}",
)
NEW_LAYOUT = StratumLayout(
    stratum="new_era",
    key_prefix="gefs.{ds}/00/atmos/pgrb2ap5/",
    file_template="{mem}.t00z.pgrb2a.0p50.f{f}",
)


def stratum_layout(stratum: str) -> StratumLayout:
    return NEW_LAYOUT if stratum == "new_era" else OLD_LAYOUT


def object_key_for_lead(stratum: str, cycle_ds: str, member: str, lead: int) -> str:
    lay = stratum_layout(stratum)
    fw = 3 if stratum == "new_era" else _old_lead_width(lead)
    f = f"{lead:0{fw}d}"
    return lay.key_prefix.format(ds=cycle_ds) + lay.file_template.format(ds=cycle_ds, mem=member, f=f)


def _old_lead_width(lead: int) -> int:
    # old-era filenames observed: gec00.t00z.pgrb2af00, f06, f102, f384 -> variable width (no zero pad)
    return len(str(lead))


def object_url(stratum: str, cycle_ds: str, member: str, lead: int) -> str:
    return f"{BUCKET_URL}/{object_key_for_lead(stratum, cycle_ds, member, lead)}"


def idx_url(stratum: str, cycle_ds: str, member: str, lead: int) -> str:
    return object_url(stratum, cycle_ds, member, lead) + ".idx"


# --- Lead grids + admission ---------------------------------------------------
def native_lead_grid(stratum: str) -> List[int]:
    if stratum == "old_era":
        return list(range(0, 385, 6))
    # new_era: 0..240 step3, then 246..840 step6
    leads = list(range(0, 241, 3)) + list(range(246, 841, 6))
    return leads


def native_dt_hours(stratum: str, lead: int) -> int:
    """Accumulation/cadence window length at this lead (for APCP identity + delivery-hour block)."""
    if stratum == "old_era":
        return 6
    return 3 if lead <= 240 else 6


def admitted_native_leads(delivery_day: dt.date) -> List[Dict]:
    """Native leads whose valid-UTC (init+lead) falls inside the delivery-day UTC window.

    No interpolation.  No backfill.  Each carries native_dt and the delivery-hour block.
    """
    init = delivery_date_to_cycle_init_utc(delivery_day)
    win_start, win_end, label = delivery_valid_utc_window(delivery_day)
    strat = expected_stratum(delivery_day)
    grid = native_lead_grid(strat)
    out = []
    for L in grid:
        valid = init + dt.timedelta(hours=L)
        if win_start <= valid < win_end:
            ndt = native_dt_hours(strat, L)
            block_end = init + dt.timedelta(hours=L + ndt)
            # delivery HE hours whose valid_utc in [valid, block_end)
            block = [
                r["he"] for r in delivery_hours_he(delivery_day)
                if valid <= r["valid_utc"] < block_end
            ]
            out.append(dict(
                lead=L, valid_utc=valid, native_dt=ndt,
                delivery_hours=block, stratum=strat,
            ))
    return out


# --- Nearest grid -------------------------------------------------------------
def nearest_grid(lat: float, lon: float, stratum: str = "new_era") -> Tuple[int, int]:
    """Deterministic nearest grid point (row i, col j) on the stratum-native grid."""
    g = GRID_BY_STRATUM[stratum]
    lonE = ((lon + 360.0) % 360.0)
    j = int(round((lonE - g["lo1"]) / g["di"]))
    i = int(round((g["la1"] - lat) / g["dj"]))
    return i, j


def grid_index1(i: int, j: int, stratum: str = "new_era") -> int:
    return i * GRID_BY_STRATUM[stratum]["Nx"] + j + 1


# --- idx parsing --------------------------------------------------------------
IDX_LINE_RE = re.compile(r"^(\d+):(\d+):([^:]*):([^:]*):([^:]*):([^:]*)(?::(.*))?$")


@dataclass
class IdxEntry:
    n: int
    offset: int
    dtag: str
    var: str
    level: str
    ftag: str
    tail: str
    raw: str

    @property
    def is_accum(self) -> bool:
        return "acc" in self.ftag

    @property
    def lead_hour(self) -> Optional[int]:
        m = re.search(r"(\d+) hour fcst", self.ftag)
        return int(m.group(1)) if m else None


def parse_idx(raw_text: str) -> List[IdxEntry]:
    out = []
    for ln in raw_text.splitlines():
        if not ln.strip():
            continue
        parts = ln.split(":")
        if len(parts) < 6:
            continue
        try:
            n = int(parts[0]); off = int(parts[1])
        except ValueError:
            continue
        out.append(IdxEntry(
            n=n, offset=off, dtag=parts[2], var=parts[3],
            level=parts[4], ftag=parts[5],
            tail=":".join(parts[6:]) if len(parts) > 6 else "",
            raw=ln,
        ))
    return out


def find_message(idx_entries: List[IdxEntry], var: str, level_prefix: str) -> Optional[IdxEntry]:
    for e in idx_entries:
        if e.var == var and e.level.startswith(level_prefix):
            return e
    return None


def message_byte_range(idx_entries: List[IdxEntry], target: IdxEntry) -> Tuple[int, Optional[int]]:
    """Return [start, end_inclusive] byte range for the message. end=None => to EOF."""
    start = target.offset
    end = None
    for e in idx_entries:
        if e.offset > target.offset:
            end = e.offset - 1
            break
    return start, end


# --- Cutoff firewall (pure, unit-testable) -----------------------------------
def cutoff_check(dtag: str, expected_cycle_ds: str) -> Tuple[bool, str]:
    """Admit only the frozen 00Z cycle initialized on calendar day t-1.

    The wgrib2 idx dtag is d=YYYYMMDDHH.  Returns (admissible, reason).
    A later cycle (06/12/18Z) or a later date is a CUTOFF_VIOLATION even if that
    cycle's object is complete on the archive.
    """
    m = re.match(r"d=(\d{8})(\d{2})$", dtag)
    if not m:
        return False, "BAD_DTAG"
    if m.group(1) != expected_cycle_ds:
        return False, "INIT_DATE_NOT_T_MINUS_1"
    if m.group(2) != "00":
        return False, "NON_00Z_CYCLE"
    return True, "FROZEN_00Z_T_MINUS_1"


# --- Derived dewpoint ---------------------------------------------------------
def dewpoint_magnus(tmp_k: float, rh_percent: float) -> float:
    """Dewpoint (K) from temperature (K) and relative humidity (%) via Magnus-Tetens.

    Labeled DERIVED in the feature lock; RH:2m is the official companion field.
    Standard formula; rh clipped to (0.5, 100] to avoid log(0).
    """
    rh = min(max(rh_percent, 0.5), 100.0)
    tmp_c = tmp_k - 273.15
    import math
    a, b = 17.625, 243.04
    gamma = math.log(rh / 100.0) + (a * tmp_c) / (b + tmp_c)
    dp_c = (b * gamma) / (a - gamma)
    return dp_c + 273.15


def wind_speed(u: float, v: float) -> float:
    import math
    return math.hypot(u, v)
