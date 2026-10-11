"""Frozen calibration scale, one warm start, and segment reset factory.

Public production entry:

    build_calibrated_saocp_state(repo, calibration_prefix)

1. Pure-validate / freeze the calibration prefix and scale **before** any
   AST loader call or state allocation.
2. Load only via production ``load_saocp_ast_kernel(repo)`` (clean pinned Git).
3. Build empty kernel state at the frozen scale and, for nonempty prefix,
   call the official AST ``state.update`` **once** with the full residual
   series (no per-value second pass; no handwritten outer loop).

Test-only offline fixtures use ``build_calibrated_saocp_state_for_test`` with
an injected test-only bundle. That helper is not re-exported from
``load_official`` and never claims production git verification.

This round does **not** implement delayed queue / production backend /
registry wiring / public SAOCP class import / real matrix / outcome access.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - pandas required for official update
    raise ImportError(
        "pandas is required for SAOCP calibration warm-start "
        "(official update uses pd.Series)"
    ) from exc

from .saocp_ast_kernel import (
    SaocpAstKernelBundle,
    load_saocp_ast_kernel,
    make_minimal_saocp_kernel_state,
)

# ---------------------------------------------------------------------------
# Frozen contract constants (saocp_verified_wrapper_v1.json)
# ---------------------------------------------------------------------------

ERROR_SCALE_NONPOSITIVE = "SAOCP_CALIBRATION_SCALE_NONPOSITIVE"
ERROR_PREFIX_INVALID = "SAOCP_CALIBRATION_PREFIX_INVALID"
ERROR_IDENTITY_SPOOF = "SAOCP_CALIBRATION_IDENTITY_SPOOF"
ERROR_RESET_INVALID = "SAOCP_CALIBRATION_RESET_INVALID"

FROZEN_CALIBRATION_PREFIX: tuple[float, ...] = (
    0.5,
    1.2,
    0.8,
    1.5,
    0.9,
    2.0,
    1.1,
    0.7,
)
EXPECTED_FROZEN_SCALE: float = 3.4641016151377544

DEFAULT_COVERAGE: float = 0.9
DEFAULT_HORIZON: int = 1
DEFAULT_LIFETIME: int = 8

_IDENTITY_MODE_PRODUCTION = "production_git_verified"
_IDENTITY_MODE_TEST_ONLY = "test_only_file_hash"


# ---------------------------------------------------------------------------
# Narrow immutable return container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibratedSaocpState:
    """Immutable narrow container for calibrated / warm-started AST state.

    ``state`` is a live kernel object (mutable internals by design for later
    adapters). Container fields themselves cannot be reassigned. Test-only
    identity is never labeled production-verified.
    """

    bundle: SaocpAstKernelBundle
    state: Any
    frozen_scale: float
    calibration_prefix: tuple[float, ...]
    prefix_length: int
    source_identity: Mapping[str, Any]
    identity_mode: str
    production_verified: bool

    def snapshot_scale(self) -> float:
        """Read current state scale at horizon 1 (must match frozen_scale)."""
        return float(self.state.scale[DEFAULT_HORIZON])


# ---------------------------------------------------------------------------
# Prefix validation / scale freeze (pure; no loader / no state)
# ---------------------------------------------------------------------------


def freeze_calibration_prefix(
    calibration_prefix: Sequence[Any],
) -> tuple[tuple[float, ...], float]:
    """Validate prefix and return ``(abs residual scores, frozen scale)``.

    Fail-closed pure function: does not load AST, allocate state, or touch
    eval/future/running scores. Input sequence is not mutated.

    Raises
    ------
    TypeError
        When ``calibration_prefix`` is not a ``list`` or ``tuple``.
    ValueError
        When elements are invalid (Python bool / NumPy ``bool_`` / str /
        NaN/Inf/non-numeric) or when a nonempty prefix yields non-positive
        scale (``SAOCP_CALIBRATION_SCALE_NONPOSITIVE``).
    """
    if not isinstance(calibration_prefix, (list, tuple)):
        raise TypeError(
            f"{ERROR_PREFIX_INVALID}: calibration_prefix must be list or tuple; "
            f"got {type(calibration_prefix).__name__}"
        )

    scores: list[float] = []
    for i, x in enumerate(calibration_prefix):
        # Python bool is a subclass of int; NumPy np.bool_ is not, but both
        # convert via float() (True->1.0 / False->0.0) and must be rejected
        # as PREFIX_INVALID before any scale computation or loader/state call.
        if isinstance(x, bool) or type(x) is np.bool_ or isinstance(x, np.bool_):
            raise ValueError(
                f"{ERROR_PREFIX_INVALID}: element[{i}] is bool (not a finite real)"
            )
        if isinstance(x, str):
            raise ValueError(
                f"{ERROR_PREFIX_INVALID}: element[{i}] is str (not a finite real)"
            )
        if isinstance(x, (bytes, bytearray)):
            raise ValueError(
                f"{ERROR_PREFIX_INVALID}: element[{i}] is bytes-like "
                "(not a finite real)"
            )
        try:
            # Reject non-numeric early (complex, objects, …).
            if isinstance(x, complex) and not isinstance(x, (int, float)):
                raise ValueError(
                    f"{ERROR_PREFIX_INVALID}: element[{i}] is complex"
                )
            v = float(x)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{ERROR_PREFIX_INVALID}: element[{i}] not convertible to "
                f"finite float: {type(x).__name__}"
            ) from exc
        if not math.isfinite(v):
            raise ValueError(
                f"{ERROR_PREFIX_INVALID}: element[{i}] is non-finite "
                f"(NaN/Inf): {v!r}"
            )
        scores.append(abs(v))

    frozen = tuple(scores)
    if len(frozen) == 0:
        return frozen, 1.0

    scale = float(max(frozen) * math.sqrt(3.0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"{ERROR_SCALE_NONPOSITIVE}: nonempty calibration_prefix yields "
            f"non-positive scale={scale!r}; refuse loader/state construction "
            "(no epsilon, no silent scale=1.0, no eval/future/running recompute)"
        )
    return frozen, scale


def _freeze_identity_mapping(identity: Mapping[str, Any]) -> Mapping[str, Any]:
    """Shallow-freeze identity dict so callers cannot mutate recorded metadata."""
    return MappingProxyType(dict(identity))


def _is_production_git_identity(identity: Mapping[str, Any]) -> bool:
    return (
        identity.get("identity_mode") == "git_full"
        and identity.get("test_only") is False
        and identity.get("git_commit_verified") is True
        and identity.get("git_tree_verified") is True
        and identity.get("clean_verified") is True
        and isinstance(identity.get("commit"), str)
        and isinstance(identity.get("tree"), str)
        and "expected_commit" not in identity  # test-only uses expected_* only
    )


def _is_test_only_identity(identity: Mapping[str, Any]) -> bool:
    return (
        identity.get("test_only") is True
        or identity.get("identity_mode") == "test_only_file_hash"
        or identity.get("git_commit_verified") is False
    )


def _assert_scale_positive(scale: float, *, where: str) -> None:
    if not math.isfinite(scale) or float(scale) <= 0.0:
        raise ValueError(
            f"{ERROR_SCALE_NONPOSITIVE}: {where}: scale={scale!r} is non-positive; "
            "refuse state construction/reset"
        )


def _warm_start_once(
    bundle: SaocpAstKernelBundle,
    frozen_prefix: tuple[float, ...],
    frozen_scale: float,
) -> Any:
    """Create empty state at frozen scale; single official AST update if nonempty."""
    _assert_scale_positive(frozen_scale, where="warm_start")
    state = make_minimal_saocp_kernel_state(
        bundle,
        coverage=DEFAULT_COVERAGE,
        lifetime=DEFAULT_LIFETIME,
        horizon=DEFAULT_HORIZON,
        scale=float(frozen_scale),
        t=1,
    )
    if len(frozen_prefix) == 0:
        return state

    # Official SAOCP.__init__ warm-start: one update of the full residual series
    # with forecast zeros (score-only residual shell). No per-value second pass.
    n = len(frozen_prefix)
    ground_truth = pd.Series(list(frozen_prefix), dtype=float)
    forecast = pd.Series(np.zeros(n, dtype=float), dtype=float)
    state.update(ground_truth, forecast, horizon=DEFAULT_HORIZON)
    return state


def _build_calibrated_from_bundle(
    bundle: SaocpAstKernelBundle,
    frozen_prefix: tuple[float, ...],
    frozen_scale: float,
    *,
    require_production: bool,
) -> CalibratedSaocpState:
    identity = dict(bundle.identity)
    if require_production:
        if not _is_production_git_identity(identity):
            raise ValueError(
                f"{ERROR_IDENTITY_SPOOF}: production build requires "
                "git_full verified identity (commit/tree/clean); "
                f"got identity_mode={identity.get('identity_mode')!r} "
                f"test_only={identity.get('test_only')!r}"
            )
        identity_mode = _IDENTITY_MODE_PRODUCTION
        production_verified = True
    else:
        if not _is_test_only_identity(identity):
            # Explicit: test helper must not silently re-label a production
            # identity, and must not accept bundles missing test_only markers.
            if _is_production_git_identity(identity):
                # Allowed only if caller intentionally injects a production
                # bundle into the test helper — still label honestly as
                # production if identity is truly production. Prefer honesty.
                identity_mode = _IDENTITY_MODE_PRODUCTION
                production_verified = True
            else:
                raise ValueError(
                    f"{ERROR_IDENTITY_SPOOF}: test helper requires test_only "
                    "identity markers (or a true production identity); "
                    f"got keys={sorted(identity)}"
                )
        else:
            identity_mode = _IDENTITY_MODE_TEST_ONLY
            production_verified = False
            # Never promote test-only to production.
            if identity.get("git_commit_verified") is True:
                raise ValueError(
                    f"{ERROR_IDENTITY_SPOOF}: test-only identity must not claim "
                    "git_commit_verified=true"
                )

    state = _warm_start_once(bundle, frozen_prefix, frozen_scale)
    return CalibratedSaocpState(
        bundle=bundle,
        state=state,
        frozen_scale=float(frozen_scale),
        calibration_prefix=frozen_prefix,
        prefix_length=len(frozen_prefix),
        source_identity=_freeze_identity_mapping(identity),
        identity_mode=identity_mode,
        production_verified=production_verified,
    )


# ---------------------------------------------------------------------------
# Public production entry
# ---------------------------------------------------------------------------


def build_calibrated_saocp_state(
    repo: Path | str,
    calibration_prefix: Sequence[Any],
) -> CalibratedSaocpState:
    """Production: freeze prefix/scale, load clean pin kernel, warm-start once.

    Parameter order and names are intentional and strict. Unknown keywords
    (scale, max_scale, eval_scores, future_suffix, …) raise ``TypeError``.

    Nonempty all-zero (or non-positive scale) fails **before** any call to
    ``load_saocp_ast_kernel`` or state allocation.
    """
    # 1) Pure validate / freeze — no loader, no state.
    frozen_prefix, frozen_scale = freeze_calibration_prefix(calibration_prefix)

    # 2) Production AST loader only (clean pinned Git checkout).
    bundle = load_saocp_ast_kernel(Path(repo))

    # 3) Single warm-start via official AST update (or cold empty state).
    return _build_calibrated_from_bundle(
        bundle,
        frozen_prefix,
        frozen_scale,
        require_production=True,
    )


def build_calibrated_saocp_state_for_test(
    bundle: SaocpAstKernelBundle,
    calibration_prefix: Sequence[Any],
) -> CalibratedSaocpState:
    """Test-only helper: inject a pre-loaded (usually test-only) bundle.

    Name contains ``_for_test``. Not a production entry; does not call
    ``load_saocp_ast_kernel``. Identity is recorded honestly from the bundle
    and is never silently labeled production-verified for test-only fixtures.
    """
    if not isinstance(bundle, SaocpAstKernelBundle):
        raise TypeError(
            f"{ERROR_PREFIX_INVALID}: bundle must be SaocpAstKernelBundle; "
            f"got {type(bundle).__name__}"
        )
    frozen_prefix, frozen_scale = freeze_calibration_prefix(calibration_prefix)
    return _build_calibrated_from_bundle(
        bundle,
        frozen_prefix,
        frozen_scale,
        require_production=False,
    )


# ---------------------------------------------------------------------------
# Segment reset factory (state only; no pending queue this round)
# ---------------------------------------------------------------------------


def reset_segment_state(calibrated: CalibratedSaocpState) -> CalibratedSaocpState:
    """Construct a **new** empty state retaining frozen scale / source identity.

    - New kernel object: ``t=1``, experts empty, Residuals empty.
    - Does **not** replay the original calibration prefix.
    - Does **not** mutate the previous state object.
    - Non-positive scale and production/test identity spoofing fail closed.
    - Pending delayed feedback is owned by the delayed adapter.
    """
    if not isinstance(calibrated, CalibratedSaocpState):
        raise TypeError(
            f"{ERROR_RESET_INVALID}: calibrated must be CalibratedSaocpState; "
            f"got {type(calibrated).__name__}"
        )

    scale = float(calibrated.frozen_scale)
    _assert_scale_positive(scale, where="reset_segment_state")

    identity = calibrated.source_identity
    # Fail closed: production_verified flag must match real identity marks.
    actual_prod = _is_production_git_identity(identity)
    if calibrated.production_verified and not actual_prod:
        raise ValueError(
            f"{ERROR_IDENTITY_SPOOF}: production_verified=True but source "
            "identity is not a verified git_full pin "
            f"(test_only={identity.get('test_only')!r}, "
            f"identity_mode={identity.get('identity_mode')!r})"
        )
    if (
        calibrated.identity_mode == _IDENTITY_MODE_PRODUCTION
        and not actual_prod
    ):
        raise ValueError(
            f"{ERROR_IDENTITY_SPOOF}: identity_mode claims production but "
            "source identity is not git_full verified"
        )
    if (
        calibrated.identity_mode == _IDENTITY_MODE_TEST_ONLY
        and actual_prod
    ):
        raise ValueError(
            f"{ERROR_IDENTITY_SPOOF}: identity_mode claims test_only but "
            "source identity is production git_full"
        )
    if calibrated.production_verified and _is_test_only_identity(identity):
        raise ValueError(
            f"{ERROR_IDENTITY_SPOOF}: test-only identity must not be labeled "
            "production_verified"
        )

    # Fresh empty state — same frozen scale, no prefix replay.
    new_state = make_minimal_saocp_kernel_state(
        calibrated.bundle,
        coverage=DEFAULT_COVERAGE,
        lifetime=DEFAULT_LIFETIME,
        horizon=DEFAULT_HORIZON,
        scale=scale,
        t=1,
    )
    return CalibratedSaocpState(
        bundle=calibrated.bundle,
        state=new_state,
        frozen_scale=scale,
        calibration_prefix=calibrated.calibration_prefix,
        prefix_length=calibrated.prefix_length,
        source_identity=calibrated.source_identity,
        identity_mode=calibrated.identity_mode,
        production_verified=calibrated.production_verified,
    )


__all__ = [
    "CalibratedSaocpState",
    "DEFAULT_COVERAGE",
    "DEFAULT_HORIZON",
    "DEFAULT_LIFETIME",
    "ERROR_IDENTITY_SPOOF",
    "ERROR_PREFIX_INVALID",
    "ERROR_RESET_INVALID",
    "ERROR_SCALE_NONPOSITIVE",
    "EXPECTED_FROZEN_SCALE",
    "FROZEN_CALIBRATION_PREFIX",
    "build_calibrated_saocp_state",
    "build_calibrated_saocp_state_for_test",
    "freeze_calibration_prefix",
    "reset_segment_state",
]
