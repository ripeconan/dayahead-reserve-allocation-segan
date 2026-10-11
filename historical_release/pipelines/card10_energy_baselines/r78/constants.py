"""Frozen constants for the energy-baseline experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# Eleven NYISO zones in the order used by every model and metric table.
ZONES: Final[tuple[str, ...]] = (
    "CAPITL",
    "CENTRL",
    "DUNWOD",
    "GENESE",
    "HUD VL",
    "LONGIL",
    "MHK VL",
    "MILLWD",
    "N.Y.C.",
    "NORTH",
    "WEST",
)

# Seven chronological origins: reference_year -> test_year.
ORIGINS: Final[tuple[tuple[int, int], ...]] = tuple(
    (year, year + 1) for year in range(2018, 2025)
)
ORIGIN_LABELS: Final[tuple[str, ...]] = tuple(
    f"{ref}->{test}" for ref, test in ORIGINS
)

# Primary no-weather protocol.
HYPERPARAM_DEV_START: Final[str] = "2017-01-01"
HYPERPARAM_DEV_END: Final[str] = "2018-12-31"
EVAL_YEARS: Final[tuple[int, ...]] = (2019, 2020, 2021, 2022, 2023, 2024, 2025)

# D-1 GEFS weather-aware sensitivity (separately labelled).
D1_GEFS_DEV_YEAR: Final[int] = 2019
D1_GEFS_EVAL_YEARS: Final[tuple[int, ...]] = (2020, 2021, 2022)
STRATUM_PRIMARY: Final[str] = "PRIMARY_NO_WEATHER"
STRATUM_D1_GEFS: Final[str] = "D1_GEFS_SENSITIVITY"

# Information set.
FEEDBACK_LAG_DAYS: Final[int] = 2
ERROR_LAGS_HOURS: Final[tuple[int, ...]] = (48, 168)
MIN_OUTCOME_LAG_HOURS: Final[int] = 48

# Target joint coverage for common OAS+SAOCP stratum.
TARGET_COVERAGE: Final[float] = 0.90
SAOCP_COMMIT: Final[str] = "95790a9b9e4061166e6d7a8a13282e019b1a8663"
SAOCP_TREE: Final[str] = "77348e5fe7f2bb3ed5a031aaad9874b6a5ab2d7b"
SAOCP_LIFETIME: Final[int] = 8
SAOCP_BACKEND_KIND: Final[str] = "verified_wrapper_ast_kernel"

# Authorized input identities.
PRIMARY_RAW_SHA256: Final[str] = (
    "d9e492532e22bc3f41c6967acf99cc5b49a49ae2e5ba3ae3b7dbc8092d9e5d0d"
)
D1_GEFS_PANEL_SHA256: Final[str] = (
    "5d67f397de807aca653fed3e4fa221a9f8e35bd3939a067b089f61cd7b40a0cd"
)

LOCAL_PRIMARY_RAW: Final[Path] = Path(
    os.environ.get(
        "ENERGY_UQ_NYISO_DATA", "data/raw/nyiso/nyiso_zonal_2017_2025.csv.gz"
    )
)
REMOTE_PRIMARY_RAW: Final[str] = str(LOCAL_PRIMARY_RAW)
REMOTE_D1_GEFS_PANEL: Final[str] = os.environ.get(
    "ENERGY_UQ_GEFS_PANEL", "data/processed/gefs/feature_panel.csv.gz"
)
REMOTE_TASK_ROOT: Final[str] = os.environ.get("ENERGY_UQ_RUN_ROOT", "runs/energy_baselines")

# Forbidden confirmatory / R77 scientific paths (never open).
FORBIDDEN_PATH_SUBSTRINGS: Final[tuple[str, ...]] = (
    "confirmatory_locked/card10_nyiso_2026h1",
    "card10_confirmatory_unseen/card10_nyiso_2026h1",
    "gefs_revision_r77_d2_tmprh",
    "card10_nyiso_2026h1",
)

# Access flags that must remain false for the entire R78 task.
ACCESS_FLAGS: Final[dict[str, bool]] = {
    "confirmatory_2026_outcome_accessed": False,
    "confirmatory_2026_forecast_payload_accessed": False,
    "reserved_confirmatory_path_accessed": False,
    "r77_d2_values_accessed": False,
    "sota_claimed": False,
}

# Quantiles for native probabilistic variants.
QUANTILES: Final[tuple[float, ...]] = (0.05, 0.50, 0.95)

# Seeds (mechanical, pre-declared).
GLOBAL_SEED: Final[int] = 20260718
MODEL_SEEDS: Final[dict[str, int]] = {
    "seasonal_naive_d7": 0,
    "nyiso_p7_official": 0,
    "ridge_lag2": 0,
    "lightgbm_energy": 20260718,
    "catboost_energy_quantile": 20260718,
    "tft_quantile": 20260718,
    "proloaf_native": 20260718,
    "mamba_aci_energy": 20260718,
    "online_quantile_ensemble_energy": 20260718,
}

# Repo-relative paths.
PACKAGE_REL: Final[str] = "pipelines/card10_energy_baselines/r78"
OUTPUT_REL: Final[str] = "runs/energy_baselines"
REPORT_REL: Final[str] = "README.md"
