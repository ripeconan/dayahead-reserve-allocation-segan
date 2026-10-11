"""Authorized NYISO load/forecast loader with SHA and firewall gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    ACCESS_FLAGS,
    D1_GEFS_PANEL_SHA256,
    LOCAL_PRIMARY_RAW,
    PRIMARY_RAW_SHA256,
    REMOTE_D1_GEFS_PANEL,
    REMOTE_PRIMARY_RAW,
    ZONES,
)
from .firewall import (
    FirewallViolation,
    assert_access_flags_clean,
    assert_frame_no_2026,
    assert_path_allowed,
    safe_resolve_existing,
)
from .hash_util import sha256_file


@dataclass(frozen=True)
class NyisoPanel:
    actual: pd.DataFrame
    forecast: pd.DataFrame
    source_path: str
    source_sha256: str
    n_rows: int
    time_start: str
    time_end: str


def resolve_primary_raw_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return safe_resolve_existing(path, context="primary_raw")
    # Prefer local development path when present; else remote path string if exists.
    if LOCAL_PRIMARY_RAW.exists():
        return safe_resolve_existing(LOCAL_PRIMARY_RAW, context="primary_raw_local")
    remote = Path(REMOTE_PRIMARY_RAW)
    if remote.exists():
        return safe_resolve_existing(remote, context="primary_raw_remote")
    raise FileNotFoundError(
        "Primary NYISO raw input not found at local or remote authorized paths"
    )


def verify_input_sha256(path: Path, expected: str) -> str:
    assert_path_allowed(path, context="verify_input_sha256")
    observed = sha256_file(path)
    if observed.lower() != expected.lower():
        raise FirewallViolation(
            f"INPUT_SHA256_MISMATCH path={path} expected={expected} observed={observed}"
        )
    return observed


def load_nyiso_raw(
    path: str | Path | None = None,
    *,
    expected_sha256: str = PRIMARY_RAW_SHA256,
) -> NyisoPanel:
    """Load eleven-zone actual/forecast frames after SHA and year firewall checks.

    Consumes values only after SHA verification succeeds.
    """
    assert_access_flags_clean(ACCESS_FLAGS)
    raw_path = resolve_primary_raw_path(path)
    observed_sha = verify_input_sha256(raw_path, expected_sha256)

    frame = pd.read_csv(raw_path, low_memory=False)
    if "timestamp_utc" not in frame.columns:
        raise ValueError("NYISO raw missing timestamp_utc column")

    index = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.tz_convert("America/New_York")
    actual = pd.DataFrame(
        frame[[f"{zone}_actual" for zone in ZONES]].to_numpy(float),
        index=pd.DatetimeIndex(index),
        columns=list(ZONES),
    )
    forecast = pd.DataFrame(
        frame[[f"{zone}_forecast" for zone in ZONES]].to_numpy(float),
        index=pd.DatetimeIndex(index),
        columns=list(ZONES),
    )
    if actual.index.duplicated().any() or not actual.index.is_monotonic_increasing:
        raise ValueError("NYISO index is duplicated or unsorted")
    if not actual.index.equals(forecast.index):
        raise ValueError("actual/forecast index mismatch")

    assert_frame_no_2026(actual)
    assert_frame_no_2026(forecast)

    return NyisoPanel(
        actual=actual,
        forecast=forecast,
        source_path=str(raw_path),
        source_sha256=observed_sha,
        n_rows=len(actual),
        time_start=str(actual.index.min()),
        time_end=str(actual.index.max()),
    )


def load_d1_gefs_panel(
    path: str | Path | None = None,
    *,
    expected_sha256: str = D1_GEFS_PANEL_SHA256,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the accepted D-1 GEFS feature panel for the sensitivity stratum only.

    Does not touch R77 D-2 revision values.
    """
    assert_access_flags_clean(ACCESS_FLAGS)
    if path is None:
        path = Path(REMOTE_D1_GEFS_PANEL)
        if not path.exists():
            raise FileNotFoundError(
                f"D1 GEFS panel not found at authorized remote path: {REMOTE_D1_GEFS_PANEL}"
            )
    panel_path = safe_resolve_existing(path, context="d1_gefs_panel")
    if "gefs_revision_r77" in str(panel_path).replace("\\", "/").lower():
        raise FirewallViolation("R77 path reached D1 GEFS loader")
    observed_sha = verify_input_sha256(panel_path, expected_sha256)
    frame = pd.read_csv(panel_path, low_memory=False)
    meta = {
        "source_path": str(panel_path),
        "source_sha256": observed_sha,
        "stratum": "D1_GEFS_SENSITIVITY",
        "r77_d2_values_accessed": False,
        "n_rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    return frame, meta


def input_provenance(panel: NyisoPanel) -> dict[str, Any]:
    return {
        "primary_raw_path": panel.source_path,
        "primary_raw_sha256": panel.source_sha256,
        "expected_primary_raw_sha256": PRIMARY_RAW_SHA256,
        "sha_match": panel.source_sha256.lower() == PRIMARY_RAW_SHA256.lower(),
        "n_rows": panel.n_rows,
        "time_start": panel.time_start,
        "time_end": panel.time_end,
        "zones": list(ZONES),
        "access_flags": dict(ACCESS_FLAGS),
        "confirmatory_2026_outcome_accessed": False,
        "confirmatory_2026_forecast_payload_accessed": False,
        "reserved_confirmatory_path_accessed": False,
        "r77_d2_values_accessed": False,
    }
