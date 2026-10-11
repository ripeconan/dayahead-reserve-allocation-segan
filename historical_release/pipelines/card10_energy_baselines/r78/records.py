"""Typed prediction and metric records for R78 adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass
class PredictionRecord:
    model: str
    candidate_id: str
    origin: str
    stratum: str
    timestamp: str
    delivery_date: str
    zone: str
    actual: float
    point: float
    q05: float | None = None
    q50: float | None = None
    q95: float | None = None
    information_cutoff: str | None = None
    feature_source_max_reveal: str | None = None
    model_hash: str | None = None
    config_hash: str | None = None
    checkpoint_hash: str | None = None
    eligibility: str = "eligible"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelRunResult:
    model: str
    candidate_id: str
    origin: str
    stratum: str
    status: str
    predictions: list[PredictionRecord] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    blocker_code: str | None = None
    notes: str | None = None

    def predictions_frame(self) -> pd.DataFrame:
        if not self.predictions:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in self.predictions])


@dataclass(frozen=True)
class BlockerRow:
    model: str
    code: str
    reason: str
    urls: tuple[str, ...] = ()
    access_date: str | None = None
    missing_elements: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "code": self.code,
            "reason": self.reason,
            "urls": list(self.urls),
            "access_date": self.access_date,
            "missing_elements": list(self.missing_elements),
        }
