"""Verified source-derived SAOCP stream with delayed feedback.

The upstream implementation is loaded from a clean checkout at the pinned Git
identity.  Its required method bodies are checked against frozen source and AST
hashes before execution.  Evaluation scores are revealed with an exact lag of
two stream steps and are never used to construct the calibration scale.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .saocp_ast_kernel import (
    PINNED_SAOCP_NODE_HASHES,
    PINNED_SAOCP_SOURCE_IDENTITY,
    REQUIRED_NODE_KEYS,
)
from .saocp_calibration_factory import freeze_calibration_prefix
from .saocp_delayed_adapter import (
    FEEDBACK_LAG,
    FROZEN_COVERAGE,
    FROZEN_LIFETIME,
    SaocpDelayedAdapter,
    build_saocp_delayed_adapter,
)

UPSTREAM_URL = "https://github.com/salesforce/online_conformal.git"
BACKEND_KIND = "verified_wrapper_ast_kernel"
IDENTITY_MODE = "production_git_verified"


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def prefix_sha256(prefix: Sequence[float]) -> str:
    """Return the byte-level identity used for a frozen calibration prefix."""
    return _sha256_array(np.asarray(tuple(prefix), dtype=np.float64))


def _require_lag_two(feedback_lag: Any) -> int:
    if type(feedback_lag) is not int or feedback_lag != FEEDBACK_LAG:
        raise ValueError("feedback_lag must be the exact integer 2")
    return feedback_lag


def _verified_source_files(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    files = source.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError("verified source-file identities are missing")
    verified: dict[str, dict[str, Any]] = {}
    for relpath, expected in PINNED_SAOCP_SOURCE_IDENTITY["files"].items():
        actual = files.get(relpath)
        if not isinstance(actual, Mapping):
            raise RuntimeError(f"source identity missing for {relpath}")
        for key in ("bytes", "sha256", "git_blob"):
            if actual.get(key) != expected[key]:
                raise RuntimeError(f"source identity mismatch for {relpath}: {key}")
        verified[relpath] = dict(actual)
    return verified


def _validate_adapter(
    adapter: SaocpDelayedAdapter,
    prefix: tuple[float, ...],
    scale: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    if type(adapter) is not SaocpDelayedAdapter:
        raise TypeError("unexpected delayed-adapter type")
    if adapter.production_verified is not True or adapter.identity_mode != IDENTITY_MODE:
        raise RuntimeError("SAOCP checkout was not verified for production use")

    source = adapter.source_identity
    pin = PINNED_SAOCP_SOURCE_IDENTITY
    required_flags = ("git_commit_verified", "git_tree_verified", "clean_verified")
    if source.get("test_only") is not False or any(source.get(key) is not True for key in required_flags):
        raise RuntimeError("SAOCP Git identity is incomplete")
    if source.get("commit") != pin["commit"] or source.get("tree") != pin["tree"]:
        raise RuntimeError("SAOCP commit or tree does not match the frozen pin")
    files = _verified_source_files(source)

    nodes = dict(adapter.calibrated.bundle.node_hashes)
    if set(nodes) != set(REQUIRED_NODE_KEYS):
        raise RuntimeError("SAOCP AST node set is incomplete")
    for key, expected in PINNED_SAOCP_NODE_HASHES.items():
        if nodes.get(key) != expected:
            raise RuntimeError(f"SAOCP AST node hash mismatch: {key}")

    if tuple(adapter.calibrated.calibration_prefix) != prefix:
        raise RuntimeError("adapter calibration prefix changed after freezing")
    if float(adapter.frozen_scale) != float(scale):
        raise RuntimeError("adapter calibration scale changed after freezing")
    return files, {key: str(nodes[key]) for key in REQUIRED_NODE_KEYS}


def run_verified_stream(
    *,
    repo: Path | str,
    calibration_prefix: Sequence[Any],
    scores: Sequence[Any],
    segment_ids: Sequence[Any] | None = None,
    feedback_lag: int = FEEDBACK_LAG,
) -> dict[str, Any]:
    """Run one local-hour SAOCP stream with strict lag-2 feedback.

    ``calibration_prefix`` must contain only prior-origin scores.  ``scores``
    are consumed in chronological order; a score at index ``t`` becomes
    eligible immediately before the prediction at ``t + 2``.
    """
    lag = _require_lag_two(feedback_lag)
    prefix, scale = freeze_calibration_prefix(calibration_prefix)
    if not prefix:
        raise ValueError("calibration_prefix must be non-empty")

    if not isinstance(scores, (list, tuple)):
        raise TypeError("scores must be a list or tuple")
    n_scores = len(scores)
    if segment_ids is None:
        segments: tuple[Any, ...] = (0,) * n_scores
    else:
        if not isinstance(segment_ids, (list, tuple)):
            raise TypeError("segment_ids must be a list or tuple")
        if len(segment_ids) != n_scores:
            raise ValueError("scores and segment_ids must have equal length")
        segments = tuple(segment_ids)

    adapter = build_saocp_delayed_adapter(Path(repo), prefix)
    source_files, node_hashes = _validate_adapter(adapter, prefix, scale)

    thresholds: list[float] = []
    for index, (score, segment) in enumerate(zip(scores, segments)):
        record = adapter.advance_and_predict(index, segment)
        threshold = float(record.snapshot.pre_reveal_threshold)
        if not math.isfinite(threshold):
            raise RuntimeError(f"non-finite SAOCP threshold at index {index}")
        thresholds.append(threshold)
        adapter.enqueue_score(index, segment, score)

    array = np.asarray(thresholds, dtype=float)
    identity = {
        "method_id": "saocp",
        "backend_kind": BACKEND_KIND,
        "official_direct": False,
        "source_derived_exact_method_bodies": True,
        "production_git_identity_verified": True,
        "public_oracle_parity": False,
        "coverage": float(FROZEN_COVERAGE),
        "lifetime": int(FROZEN_LIFETIME),
        "feedback_lag": lag,
        "calibration_source": "explicit_prior_origin_argument",
        "calibration_prefix_count": len(prefix),
        "calibration_prefix_sha256": prefix_sha256(prefix),
        "frozen_scale": float(scale),
        "official_repo": UPSTREAM_URL,
        "official_commit": PINNED_SAOCP_SOURCE_IDENTITY["commit"],
        "official_tree": PINNED_SAOCP_SOURCE_IDENTITY["tree"],
        "official_source_files": source_files,
        "node_hashes": node_hashes,
        "identity_mode": IDENTITY_MODE,
        "reveal_update_order": "predict after due updates, then enqueue current score",
    }
    return {
        "method_id": "saocp",
        "pre_action": array,
        "pre_hash": _sha256_array(array),
        "feedback_lag": lag,
        "pre_reveal_records": adapter.pre_reveal_records,
        "reveal_records": adapter.reveal_records,
        "pending_origin_indices": adapter.pending_origin_indices,
        "pending_count": adapter.pending_count,
        "update_count": adapter.update_count,
        "calibration_prefix": prefix,
        "frozen_scale": float(scale),
        "identity": identity,
    }

