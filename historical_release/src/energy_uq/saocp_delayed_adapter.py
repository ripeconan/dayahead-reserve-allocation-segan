"""SAOCP delayed-reveal adapter with step-level observables.

Public production entry:

    build_saocp_delayed_adapter(repo, calibration_prefix)

1. Calls ``build_calibrated_saocp_state(repo, calibration_prefix)``
   (never accepts a pre-built test bundle as production).
2. Wraps the calibrated AST kernel state in a strict two-phase state machine:
   ``advance_and_predict`` then ``enqueue_score`` — never a single function that
   both takes day-t score and returns day-t action threshold.
3. Feedback lag is frozen at 2; coverage/horizon/lifetime/scale are not open
   parameters on the production API.

Test-only helper:

    build_saocp_delayed_adapter_for_test(calibrated)

accepts an honest ``CalibratedSaocpState`` (typically from
``build_calibrated_saocp_state_for_test``). Name contains ``_for_test``; not
re-exported from ``load_official``.

This round does **not** import public ``online_conformal.saocp.SAOCP``, does
not run public-oracle parity, does not wire production backends/registry, and
does not access real outcomes. P0-4 remains OPEN.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - pandas required for official update
    raise ImportError(
        "pandas is required for SAOCP delayed adapter "
        "(official update uses pd.Series)"
    ) from exc

from .saocp_calibration_factory import (
    DEFAULT_COVERAGE,
    DEFAULT_HORIZON,
    DEFAULT_LIFETIME,
    CalibratedSaocpState,
    build_calibrated_saocp_state,
    reset_segment_state,
)

# ---------------------------------------------------------------------------
# Frozen contract constants (saocp_verified_wrapper_v1.json)
# ---------------------------------------------------------------------------

FEEDBACK_LAG: int = 2
FROZEN_HORIZON: int = DEFAULT_HORIZON  # 1
FROZEN_COVERAGE: float = DEFAULT_COVERAGE  # 0.9
FROZEN_LIFETIME: int = DEFAULT_LIFETIME  # 8

ERROR_CAUSAL: str = "SAOCP_DELAYED_CAUSAL_VIOLATION"
ERROR_SCORE_INVALID: str = "SAOCP_DELAYED_SCORE_INVALID"
ERROR_OBSERVABLE: str = "SAOCP_DELAYED_OBSERVABLE_INVALID"
ERROR_UPDATE_CAPTURE: str = "SAOCP_DELAYED_UPDATE_CAPTURE_FAILED"
ERROR_SCALE_DRIFT: str = "SAOCP_DELAYED_SCALE_DRIFT"
ERROR_ADAPTER_INVALID: str = "SAOCP_DELAYED_ADAPTER_INVALID"
ERROR_IDENTITY: str = "SAOCP_DELAYED_IDENTITY_INVALID"

FROZEN_EVAL_SCORES: tuple[float, ...] = (
    1.0,
    0.4,
    1.6,
    0.9,
    2.1,
    0.7,
    1.3,
    0.5,
    1.8,
    1.1,
    0.6,
    1.4,
)
FUTURE_SUFFIX_ALT: tuple[float, ...] = (50.0, 50.0, 50.0, 50.0)
FUTURE_SUFFIX_CUT_INDEX: int = 8
SEGMENT_COUNTERFACTUAL_IDS: tuple[int, ...] = (
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
)
SEGMENT_PREGAP_PERTURB_INDEX: int = 2
SEGMENT_PREGAP_PERTURB_VALUE: float = 1000.0
SEGMENT_POSTGAP_FROM: int = 6


class SaocpDelayedAdapterError(RuntimeError):
    """Fail-closed error for delayed adapter / observables (never silent)."""


# ---------------------------------------------------------------------------
# Immutable observable records
# ---------------------------------------------------------------------------


def _freeze_identity(identity: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(identity))


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable SAOCP internal state at a single causal cut.

    The ten contract regular fields (threshold + scale + t + expert geometry +
    mixture weights) are always present. Floats must be finite; birth keys
    align expert_* lists and ``normalized_weights_p``.
    """

    pre_reveal_threshold: float
    scale: float
    t_counter: int
    expert_birth_indices: tuple[int, ...]
    expert_count: int
    expert_yhat: tuple[float, ...]
    expert_w: tuple[float, ...]
    expert_s_t: tuple[int, ...]
    expert_lifetime: tuple[int, ...]
    normalized_weights_p: Mapping[int, float]


@dataclass(frozen=True)
class PreRevealRecord:
    """One pre-reveal step after due residual updates and before day-t enqueue."""

    stream_index: int
    segment_id: Any
    source_identity: Mapping[str, Any]
    due_origin_indices: tuple[int, ...]
    pending_origin_indices: tuple[int, ...]
    snapshot: StateSnapshot


@dataclass(frozen=True)
class RevealUpdateRecord:
    """One actual residual reveal / official AST ``state.update`` call."""

    origin_index: int
    due_index: int
    current_index: int
    segment_id: Any
    abs_score: float
    before: StateSnapshot
    after: StateSnapshot
    meta_prediction: float
    meta_loss: float
    source_identity: Mapping[str, Any]


@dataclass(frozen=True)
class _PendingResidual:
    origin_index: int
    segment_id: Any
    abs_score: float
    due_time: int  # origin_index + FEEDBACK_LAG


# ---------------------------------------------------------------------------
# Score validation (mirror calibration factory bool / finite rules)
# ---------------------------------------------------------------------------


def _validate_finite_score(score: Any, *, where: str) -> float:
    """Copy ``abs(float(score))`` after rejecting bool / non-finite inputs."""
    if isinstance(score, bool) or type(score) is np.bool_ or isinstance(
        score, np.bool_
    ):
        raise ValueError(
            f"{ERROR_SCORE_INVALID}: {where}: score is bool (not a finite real)"
        )
    if isinstance(score, str):
        raise ValueError(
            f"{ERROR_SCORE_INVALID}: {where}: score is str (not a finite real)"
        )
    if isinstance(score, (bytes, bytearray)):
        raise ValueError(
            f"{ERROR_SCORE_INVALID}: {where}: score is bytes-like "
            "(not a finite real)"
        )
    if isinstance(score, complex) and not isinstance(score, (int, float)):
        raise ValueError(
            f"{ERROR_SCORE_INVALID}: {where}: score is complex"
        )
    try:
        v = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{ERROR_SCORE_INVALID}: {where}: score not convertible to "
            f"finite float: {type(score).__name__}"
        ) from exc
    if not math.isfinite(v):
        raise ValueError(
            f"{ERROR_SCORE_INVALID}: {where}: score is non-finite (NaN/Inf): {v!r}"
        )
    return abs(v)


def _is_bool_like(value: Any) -> bool:
    """True for Python ``bool`` and NumPy ``bool_`` (both coerce via float/int)."""
    return (
        isinstance(value, bool)
        or type(value) is np.bool_
        or isinstance(value, np.bool_)
    )


def _require_finite(value: Any, *, field: str) -> float:
    """Strict finite float guard for observables (threshold/scale/yhat/w/p/meta).

    Rejects Python/NumPy bool **before** any numeric conversion. Allows real
    finite int/float (and numpy floating scalars) via ``float(...)`` only after
    the bool gate. No silent coerce of str/bytes, no clip/default.
    """
    if _is_bool_like(value):
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: {field} is bool (not a finite real)"
        )
    if isinstance(value, str):
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: {field} is str (not a finite real)"
        )
    if isinstance(value, (bytes, bytearray)):
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: {field} is bytes-like (not a finite real)"
        )
    if isinstance(value, complex) and not isinstance(value, (int, float)):
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: {field} is complex (not a finite real)"
        )
    try:
        fv = float(value)
    except (TypeError, ValueError) as exc:
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: {field} not a finite float: "
            f"{type(value).__name__}={value!r}"
        ) from exc
    if not math.isfinite(fv):
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: {field} is non-finite: {fv!r}"
        )
    return fv


def _require_exact_python_int(
    value: Any,
    *,
    field: str,
    min_value: Optional[int] = None,
) -> int:
    """Exact Python ``int`` only — not bool, float, str, or numpy integer.

    Frozen AST normal paths emit Python ``int``. Observable layer must not
    normalize via ``int(9.75)`` / ``int("9")`` / ``int(True)``.
    """
    # ``bool`` is a subclass of ``int``; require exact type identity.
    if type(value) is not int:
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: {field} must be exact Python int "
            f"(not bool/float/str/numpy); got {type(value).__name__}={value!r}"
        )
    if min_value is not None and value < min_value:
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: {field}={value} out of range "
            f"(expected >= {min_value})"
        )
    return value


# ---------------------------------------------------------------------------
# Snapshot capture (fail-closed; no silent field deletion)
# ---------------------------------------------------------------------------


def _read_experts(state: Any) -> dict:
    try:
        experts_by_h = state.experts
        experts = experts_by_h[FROZEN_HORIZON]
    except Exception as exc:
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: cannot read state.experts[{FROZEN_HORIZON}]"
        ) from exc
    if not isinstance(experts, dict):
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: experts[{FROZEN_HORIZON}] is not a dict"
        )
    return experts


def _exact_birth_indices(experts: dict) -> tuple[int, ...]:
    """Validate expert birth keys as exact Python ints (birth >= 1); no coerce."""
    birth_list: list[int] = []
    for k in experts.keys():
        b = _require_exact_python_int(k, field="expert_birth_key", min_value=1)
        birth_list.append(b)
    birth_indices = tuple(sorted(birth_list))
    if len(birth_indices) != len(experts):
        # Duplicate keys after any hypothetical normalization would surface here;
        # with exact keys this is dict integrity / length mismatch.
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: birth key set size mismatch"
        )
    return birth_indices


def capture_state_snapshot(
    state: Any,
    *,
    frozen_scale: float,
    pre_reveal_threshold: Optional[float] = None,
) -> StateSnapshot:
    """Build an immutable StateSnapshot from a live AST kernel state.

    Raises ``SaocpDelayedAdapterError`` on unreadable / misaligned / non-finite
    / non-exact-int fields or scale drift (never bare KeyError/IndexError/
    ZeroDivisionError). No silent coerce, clip, drop, or defaults.
    """
    try:
        scale_raw = state.scale[FROZEN_HORIZON]
    except Exception as exc:
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: cannot read state.scale[{FROZEN_HORIZON}]"
        ) from exc
    scale = _require_finite(scale_raw, field="scale")
    if scale != float(frozen_scale):
        # Bitwise / exact float equality against frozen calibration scale.
        raise SaocpDelayedAdapterError(
            f"{ERROR_SCALE_DRIFT}: scale={scale!r} != frozen_scale={frozen_scale!r}"
        )

    try:
        t_raw = state.t
    except Exception as exc:
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: cannot read state.t"
        ) from exc
    t_counter = _require_exact_python_int(
        t_raw, field="t_counter", min_value=1
    )

    experts = _read_experts(state)
    birth_indices = _exact_birth_indices(experts)
    expert_count = len(birth_indices)

    yhat_list: list[float] = []
    w_list: list[float] = []
    s_t_list: list[int] = []
    life_list: list[int] = []
    for b in birth_indices:
        try:
            expert = experts[b]
        except Exception as exc:
            raise SaocpDelayedAdapterError(
                f"{ERROR_OBSERVABLE}: missing expert birth={b}"
            ) from exc
        try:
            yhat_list.append(
                _require_finite(expert.yhat, field=f"expert_yhat[{b}]")
            )
            w_list.append(_require_finite(expert.w, field=f"expert_w[{b}]"))
            s_t_list.append(
                _require_exact_python_int(
                    expert.s_t, field=f"expert_s_t[{b}]", min_value=0
                )
            )
            life_list.append(
                _require_exact_python_int(
                    expert.lifetime, field=f"expert_lifetime[{b}]", min_value=1
                )
            )
        except SaocpDelayedAdapterError:
            raise
        except Exception as exc:
            raise SaocpDelayedAdapterError(
                f"{ERROR_OBSERVABLE}: unreadable expert fields at birth={b}"
            ) from exc

    # Mixture weights via official AST get_p (not a handwritten prior).
    try:
        p_raw = state.get_p(FROZEN_HORIZON)
    except Exception as exc:
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: state.get_p failed"
        ) from exc
    if not isinstance(p_raw, dict):
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: get_p did not return a dict"
        )
    # Exact key types first — never int(k) which collides bool True with 1.
    p_keys: set[int] = set()
    for k in p_raw.keys():
        pk = _require_exact_python_int(k, field="get_p_key", min_value=1)
        p_keys.add(pk)
    birth_set = set(birth_indices)
    if p_keys != birth_set:
        raise SaocpDelayedAdapterError(
            f"{ERROR_OBSERVABLE}: normalized_weights_p keys {sorted(p_keys)} "
            f"!= expert_birth_indices {list(birth_indices)}"
        )
    p_frozen: dict[int, float] = {}
    for b in birth_indices:
        try:
            p_frozen[b] = _require_finite(
                p_raw[b], field=f"normalized_weights_p[{b}]"
            )
        except SaocpDelayedAdapterError:
            raise
        except Exception as exc:
            raise SaocpDelayedAdapterError(
                f"{ERROR_OBSERVABLE}: cannot read p[{b}]"
            ) from exc

    if pre_reveal_threshold is None:
        try:
            thr = state.predict(FROZEN_HORIZON)[1]
        except Exception as exc:
            raise SaocpDelayedAdapterError(
                f"{ERROR_OBSERVABLE}: state.predict failed for threshold"
            ) from exc
        threshold = _require_finite(thr, field="pre_reveal_threshold")
    else:
        threshold = _require_finite(
            pre_reveal_threshold, field="pre_reveal_threshold"
        )

    return StateSnapshot(
        pre_reveal_threshold=threshold,
        scale=scale,
        t_counter=t_counter,
        expert_birth_indices=birth_indices,
        expert_count=expert_count,
        expert_yhat=tuple(yhat_list),
        expert_w=tuple(w_list),
        expert_s_t=tuple(s_t_list),
        expert_lifetime=tuple(life_list),
        normalized_weights_p=MappingProxyType(p_frozen),
    )


def snapshots_equal(a: StateSnapshot, b: StateSnapshot) -> bool:
    """Exact field equality for counterfactual / isolation checks."""
    if a.t_counter != b.t_counter:
        return False
    if a.expert_birth_indices != b.expert_birth_indices:
        return False
    if a.expert_count != b.expert_count:
        return False
    if a.expert_s_t != b.expert_s_t:
        return False
    if a.expert_lifetime != b.expert_lifetime:
        return False
    if a.scale != b.scale:
        return False
    if a.pre_reveal_threshold != b.pre_reveal_threshold:
        return False
    if a.expert_yhat != b.expert_yhat:
        return False
    if a.expert_w != b.expert_w:
        return False
    if dict(a.normalized_weights_p) != dict(b.normalized_weights_p):
        return False
    return True


# ---------------------------------------------------------------------------
# Official update with predict capture (no handwritten update loop / meta)
# ---------------------------------------------------------------------------


def _apply_due_residual_update(
    state: Any,
    *,
    abs_score: float,
    pinball_loss: Any,
    frozen_scale: float,
) -> tuple[StateSnapshot, StateSnapshot, float, float]:
    """Call original AST ``state.update`` once; capture two official predicts.

    Temporarily wraps the instance ``predict`` to delegate to the original
    bound method and record return values. Restores the instance method after
    the call. Asserts exactly two predict calls; second is meta_prediction
    after create_expert. ``meta_loss`` uses bundle pinball_loss.
    """
    before = capture_state_snapshot(state, frozen_scale=frozen_scale)

    # Resolve original bound method without going through instance dict.
    original_predict = state.predict
    # If already wrapped / unbound oddly, still capture via callable.
    predict_returns: list[Any] = []

    def _wrapped_predict(horizon: int) -> Any:
        result = original_predict(horizon)
        predict_returns.append(result)
        return result

    # Instance override so official update body sees the wrapper.
    state.predict = _wrapped_predict  # type: ignore[method-assign]
    try:
        ground_truth = pd.Series([float(abs_score)], dtype=float)
        forecast = pd.Series([0.0], dtype=float)
        state.update(ground_truth, forecast, horizon=FROZEN_HORIZON)
    except SaocpDelayedAdapterError:
        raise
    except Exception as exc:
        raise SaocpDelayedAdapterError(
            f"{ERROR_UPDATE_CAPTURE}: official state.update failed: {exc!r}"
        ) from exc
    finally:
        # Restore class method by deleting instance attribute.
        try:
            del state.predict
        except Exception:
            try:
                state.predict = original_predict  # type: ignore[method-assign]
            except Exception as exc:
                raise SaocpDelayedAdapterError(
                    f"{ERROR_UPDATE_CAPTURE}: failed to restore state.predict"
                ) from exc

    if len(predict_returns) != 2:
        raise SaocpDelayedAdapterError(
            f"{ERROR_UPDATE_CAPTURE}: expected exactly 2 official predict "
            f"calls inside update; got {len(predict_returns)}"
        )

    try:
        second = predict_returns[1]
        meta_prediction = _require_finite(second[1], field="meta_prediction")
    except SaocpDelayedAdapterError:
        raise
    except Exception as exc:
        raise SaocpDelayedAdapterError(
            f"{ERROR_UPDATE_CAPTURE}: second predict return unreadable"
        ) from exc

    try:
        meta_loss_raw = pinball_loss(
            float(abs_score), float(meta_prediction), float(FROZEN_COVERAGE)
        )
        meta_loss = _require_finite(meta_loss_raw, field="meta_loss")
    except SaocpDelayedAdapterError:
        raise
    except Exception as exc:
        raise SaocpDelayedAdapterError(
            f"{ERROR_UPDATE_CAPTURE}: pinball_loss failed: {exc!r}"
        ) from exc

    after = capture_state_snapshot(state, frozen_scale=frozen_scale)
    return before, after, meta_prediction, meta_loss


# ---------------------------------------------------------------------------
# Adapter state machine
# ---------------------------------------------------------------------------


class SaocpDelayedAdapter:
    """Strict two-phase delayed reveal adapter over a calibrated AST state.

    Lifecycle per stream index ``t = 0, 1, 2, ...``:

    1. ``advance_and_predict(t, segment_id)`` — process due same-segment
       residuals (or segment reset + drop pending), then record pre-reveal
       snapshot. **Does not accept score/outcome.**
    2. ``enqueue_score(t, segment_id, score)`` — validate and queue
       ``abs(float(score))``; **does not update**.

    Exactly one matching enqueue must follow each advance. No end-of-stream
    flush of not-yet-due residuals.
    """

    __slots__ = (
        "_calibrated",
        "_state",
        "_bundle",
        "_frozen_scale",
        "_source_identity",
        "_production_verified",
        "_identity_mode",
        "_phase",
        "_next_stream_index",
        "_awaiting_enqueue_index",
        "_awaiting_enqueue_segment",
        "_current_segment_id",
        "_pending",
        "_pre_reveal_records",
        "_reveal_records",
        "_update_count",
    )

    def __init__(self, calibrated: CalibratedSaocpState) -> None:
        if not isinstance(calibrated, CalibratedSaocpState):
            raise TypeError(
                f"{ERROR_ADAPTER_INVALID}: calibrated must be CalibratedSaocpState; "
                f"got {type(calibrated).__name__}"
            )
        scale = float(calibrated.frozen_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"{ERROR_SCALE_DRIFT}: frozen_scale non-positive: {scale!r}"
            )
        # Fail closed if live state already drifted from container metadata.
        live = float(calibrated.state.scale[FROZEN_HORIZON])
        if live != scale:
            raise SaocpDelayedAdapterError(
                f"{ERROR_SCALE_DRIFT}: live scale {live!r} != frozen {scale!r}"
            )

        self._calibrated = calibrated
        self._state = calibrated.state
        self._bundle = calibrated.bundle
        self._frozen_scale = scale
        self._source_identity = _freeze_identity(calibrated.source_identity)
        self._production_verified = bool(calibrated.production_verified)
        self._identity_mode = str(calibrated.identity_mode)

        self._phase = "need_advance"  # or "need_enqueue"
        self._next_stream_index = 0
        self._awaiting_enqueue_index: Optional[int] = None
        self._awaiting_enqueue_segment: Any = None
        self._current_segment_id: Any = None
        self._pending: list[_PendingResidual] = []
        self._pre_reveal_records: list[PreRevealRecord] = []
        self._reveal_records: list[RevealUpdateRecord] = []
        self._update_count = 0

    # -- read-only properties ------------------------------------------------

    @property
    def frozen_scale(self) -> float:
        return self._frozen_scale

    @property
    def feedback_lag(self) -> int:
        return FEEDBACK_LAG

    @property
    def source_identity(self) -> Mapping[str, Any]:
        return self._source_identity

    @property
    def production_verified(self) -> bool:
        return self._production_verified

    @property
    def identity_mode(self) -> str:
        return self._identity_mode

    @property
    def state(self) -> Any:
        """Live AST kernel state (mutable internals; for tests / inspection)."""
        return self._state

    @property
    def calibrated(self) -> CalibratedSaocpState:
        """Current calibrated container (replaced on segment reset)."""
        return self._calibrated

    @property
    def pre_reveal_records(self) -> tuple[PreRevealRecord, ...]:
        return tuple(self._pre_reveal_records)

    @property
    def reveal_records(self) -> tuple[RevealUpdateRecord, ...]:
        return tuple(self._reveal_records)

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def pending_origin_indices(self) -> tuple[int, ...]:
        return tuple(p.origin_index for p in self._pending)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def next_stream_index(self) -> int:
        return self._next_stream_index

    @property
    def phase(self) -> str:
        return self._phase

    # -- internal helpers ----------------------------------------------------

    def _assert_scale(self) -> None:
        try:
            live = float(self._state.scale[FROZEN_HORIZON])
        except Exception as exc:
            raise SaocpDelayedAdapterError(
                f"{ERROR_OBSERVABLE}: cannot read live scale"
            ) from exc
        if live != self._frozen_scale:
            raise SaocpDelayedAdapterError(
                f"{ERROR_SCALE_DRIFT}: live scale {live!r} != "
                f"frozen_scale {self._frozen_scale!r}"
            )

    def _due_pending_sorted(self, stream_index: int) -> list[_PendingResidual]:
        """Pending residuals due at ``stream_index`` in (due_time, origin) order."""
        due = [
            p
            for p in self._pending
            if stream_index >= p.due_time
            and p.segment_id == self._current_segment_id
        ]
        due.sort(key=lambda p: (p.due_time, p.origin_index))
        return due

    def _drop_pending_all(self) -> None:
        self._pending.clear()

    def _remove_from_pending(self, origin_index: int) -> None:
        self._pending = [p for p in self._pending if p.origin_index != origin_index]

    def _segment_reset(self) -> None:
        """Reset to an empty state with the same frozen scale and identity."""
        old_state = self._state
        new_cal = reset_segment_state(self._calibrated)
        if new_cal.state is old_state:
            raise SaocpDelayedAdapterError(
                f"{ERROR_ADAPTER_INVALID}: reset_segment_state returned same state object"
            )
        # Old state must not be mutated by reset factory.
        self._calibrated = new_cal
        self._state = new_cal.state
        if float(new_cal.frozen_scale) != self._frozen_scale:
            raise SaocpDelayedAdapterError(
                f"{ERROR_SCALE_DRIFT}: reset changed frozen_scale"
            )
        self._source_identity = _freeze_identity(new_cal.source_identity)
        self._drop_pending_all()
        self._assert_scale()
        # Post-reset: t=1, experts empty (contract).
        t_after = _require_exact_python_int(
            self._state.t, field="t_counter_after_segment_reset", min_value=1
        )
        if t_after != 1:
            raise SaocpDelayedAdapterError(
                f"{ERROR_OBSERVABLE}: after segment reset t={t_after}, expected 1"
            )
        if len(_read_experts(self._state)) != 0:
            raise SaocpDelayedAdapterError(
                f"{ERROR_OBSERVABLE}: after segment reset experts not empty"
            )

    # -- public two-phase API ------------------------------------------------

    def advance_and_predict(self, stream_index: int, segment_id: Any) -> PreRevealRecord:
        """Process due reveals for ``stream_index``, then record pre-reveal.

        Does **not** accept score or outcome. Segment change at the start of
        this call resets state and drops old-segment pending without update.
        """
        if self._phase != "need_advance":
            raise ValueError(
                f"{ERROR_CAUSAL}: advance_and_predict called while phase="
                f"{self._phase!r}; expected need_advance (enqueue first)"
            )
        if not isinstance(stream_index, int) or isinstance(stream_index, bool):
            raise TypeError(
                f"{ERROR_CAUSAL}: stream_index must be int; got "
                f"{type(stream_index).__name__}"
            )
        if stream_index != self._next_stream_index:
            raise ValueError(
                f"{ERROR_CAUSAL}: stream_index={stream_index} but expected "
                f"contiguous next={self._next_stream_index} "
                "(skip/jump forbidden)"
            )
        if stream_index < 0:
            raise ValueError(
                f"{ERROR_CAUSAL}: stream_index must be >= 0; got {stream_index}"
            )

        self._assert_scale()

        # Segment isolation at the head of advance.
        if self._current_segment_id is None:
            self._current_segment_id = segment_id
        elif segment_id != self._current_segment_id:
            self._segment_reset()
            self._current_segment_id = segment_id

        # Due residuals: same segment, t >= origin + lag; ordered by due/origin.
        due_list = self._due_pending_sorted(stream_index)
        due_origins: list[int] = []
        for item in due_list:
            before, after, meta_pred, meta_loss = _apply_due_residual_update(
                self._state,
                abs_score=item.abs_score,
                pinball_loss=self._bundle.pinball_loss,
                frozen_scale=self._frozen_scale,
            )
            self._update_count += 1
            due_origins.append(item.origin_index)
            self._reveal_records.append(
                RevealUpdateRecord(
                    origin_index=item.origin_index,
                    due_index=stream_index,
                    current_index=stream_index,
                    segment_id=self._current_segment_id,
                    abs_score=float(item.abs_score),
                    before=before,
                    after=after,
                    meta_prediction=meta_pred,
                    meta_loss=meta_loss,
                    source_identity=self._source_identity,
                )
            )
            self._remove_from_pending(item.origin_index)

        # Pre-reveal snapshot AFTER due updates, BEFORE day-t score is known.
        snapshot = capture_state_snapshot(
            self._state, frozen_scale=self._frozen_scale
        )
        pending_origins = tuple(p.origin_index for p in self._pending)
        record = PreRevealRecord(
            stream_index=stream_index,
            segment_id=self._current_segment_id,
            source_identity=self._source_identity,
            due_origin_indices=tuple(due_origins),
            pending_origin_indices=pending_origins,
            snapshot=snapshot,
        )
        self._pre_reveal_records.append(record)

        self._phase = "need_enqueue"
        self._awaiting_enqueue_index = stream_index
        self._awaiting_enqueue_segment = segment_id
        return record

    def enqueue_score(
        self, stream_index: int, segment_id: Any, score: Any
    ) -> None:
        """Queue day-t abs residual after its pre-reveal prediction is recorded.

        Validates finite real (rejects Python/NumPy bool, NaN/Inf). Does **not**
        call ``state.update`` on the enqueue day.
        """
        if self._phase != "need_enqueue":
            raise ValueError(
                f"{ERROR_CAUSAL}: enqueue_score called while phase="
                f"{self._phase!r}; expected need_enqueue "
                "(enqueue-before-predict forbidden)"
            )
        if not isinstance(stream_index, int) or isinstance(stream_index, bool):
            raise TypeError(
                f"{ERROR_CAUSAL}: stream_index must be int; got "
                f"{type(stream_index).__name__}"
            )
        if stream_index != self._awaiting_enqueue_index:
            raise ValueError(
                f"{ERROR_CAUSAL}: enqueue stream_index={stream_index} but "
                f"awaiting {self._awaiting_enqueue_index}"
            )
        if segment_id != self._awaiting_enqueue_segment:
            raise ValueError(
                f"{ERROR_CAUSAL}: enqueue segment_id={segment_id!r} but "
                f"awaiting {self._awaiting_enqueue_segment!r}"
            )
        if segment_id != self._current_segment_id:
            raise ValueError(
                f"{ERROR_CAUSAL}: enqueue segment_id={segment_id!r} != "
                f"current segment {self._current_segment_id!r}"
            )

        abs_score = _validate_finite_score(
            score, where=f"enqueue stream_index={stream_index}"
        )
        # Defensive copy: store a fresh float; caller may mutate source list later.
        stored = float(abs_score)
        self._pending.append(
            _PendingResidual(
                origin_index=int(stream_index),
                segment_id=segment_id,
                abs_score=stored,
                due_time=int(stream_index) + FEEDBACK_LAG,
            )
        )

        self._phase = "need_advance"
        self._awaiting_enqueue_index = None
        self._awaiting_enqueue_segment = None
        self._next_stream_index = int(stream_index) + 1
        # Enqueue must never update.
        self._assert_scale()

    def run_score_stream(
        self,
        scores: Sequence[Any],
        segment_ids: Optional[Sequence[Any]] = None,
    ) -> tuple[tuple[PreRevealRecord, ...], tuple[RevealUpdateRecord, ...]]:
        """Convenience: alternate advance/enqueue over a score sequence.

        Does not flush trailing pending. ``segment_ids`` defaults to all 0.
        """
        if not isinstance(scores, (list, tuple)):
            raise TypeError(
                f"{ERROR_ADAPTER_INVALID}: scores must be list or tuple; "
                f"got {type(scores).__name__}"
            )
        n = len(scores)
        if segment_ids is None:
            segs: Sequence[Any] = (0,) * n
        else:
            if not isinstance(segment_ids, (list, tuple)):
                raise TypeError(
                    f"{ERROR_ADAPTER_INVALID}: segment_ids must be list or tuple"
                )
            if len(segment_ids) != n:
                raise ValueError(
                    f"{ERROR_ADAPTER_INVALID}: len(segment_ids)={len(segment_ids)} "
                    f"!= len(scores)={n}"
                )
            segs = segment_ids
        for i in range(n):
            self.advance_and_predict(i, segs[i])
            self.enqueue_score(i, segs[i], scores[i])
        return self.pre_reveal_records, self.reveal_records


# ---------------------------------------------------------------------------
# Construction entries
# ---------------------------------------------------------------------------


def build_saocp_delayed_adapter(
    repo: Path | str,
    calibration_prefix: Sequence[Any],
) -> SaocpDelayedAdapter:
    """Production: calibrated state from repo+prefix, then delayed adapter.

    Parameter order/names are strict. Unknown keywords (lag, coverage,
    lifetime, scale, eval_scores, future_suffix, bundle, …) raise ``TypeError``.
    Never accepts a pre-constructed test bundle as production input.
    """
    calibrated = build_calibrated_saocp_state(repo, calibration_prefix)
    if not calibrated.production_verified:
        raise ValueError(
            f"{ERROR_IDENTITY}: production delayed adapter requires "
            "production_verified calibrated state"
        )
    return SaocpDelayedAdapter(calibrated)


def build_saocp_delayed_adapter_for_test(
    calibrated: CalibratedSaocpState,
) -> SaocpDelayedAdapter:
    """Test-only helper: wrap an honest calibrated state (name contains ``_for_test``).

    Does not call production git loader. Not a production entry; not re-exported
    from ``load_official``.
    """
    if not isinstance(calibrated, CalibratedSaocpState):
        raise TypeError(
            f"{ERROR_ADAPTER_INVALID}: calibrated must be CalibratedSaocpState; "
            f"got {type(calibrated).__name__}"
        )
    return SaocpDelayedAdapter(calibrated)


__all__ = [
    "ERROR_ADAPTER_INVALID",
    "ERROR_CAUSAL",
    "ERROR_IDENTITY",
    "ERROR_OBSERVABLE",
    "ERROR_SCALE_DRIFT",
    "ERROR_SCORE_INVALID",
    "ERROR_UPDATE_CAPTURE",
    "FEEDBACK_LAG",
    "FROZEN_COVERAGE",
    "FROZEN_EVAL_SCORES",
    "FROZEN_HORIZON",
    "FROZEN_LIFETIME",
    "FUTURE_SUFFIX_ALT",
    "FUTURE_SUFFIX_CUT_INDEX",
    "PreRevealRecord",
    "RevealUpdateRecord",
    "SEGMENT_COUNTERFACTUAL_IDS",
    "SEGMENT_POSTGAP_FROM",
    "SEGMENT_PREGAP_PERTURB_INDEX",
    "SEGMENT_PREGAP_PERTURB_VALUE",
    "SaocpDelayedAdapter",
    "SaocpDelayedAdapterError",
    "StateSnapshot",
    "build_saocp_delayed_adapter",
    "build_saocp_delayed_adapter_for_test",
    "capture_state_snapshot",
    "snapshots_equal",
]
