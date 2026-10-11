"""R2 calendar-batched delayed SAOCP interface — v3 (post R03B review).

This is a NEW wrapper around the pinned public numerical kernel.  It is not the
historical observation-index adapter and no stepwise parity with it is claimed.

Scheduling contract:
  * one kernel instance per local hour slot; the frozen per-slot scale is
    max(|reference prefix score|)*sqrt(3) over a FINITE, NONEMPTY prefix - an
    empty or nonfinite prefix is refused (a declared neutral scale is available
    only to explicitly labeled synthetic tests);
  * the exact prefix used to fix the scale is the prefix fed to warm start;
  * at the decision for local date d: release queued scores whose source local
    date is <= d-2 in chronological UTC order, then freeze the 24 slot
    thresholds for d (both fall folds share the frozen slot threshold);
  * reveal date = source local date + 2 calendar days; absent hours produce no
    score (never a fabricated zero); partial days never enqueue;
  * resets come from ONE shared event list (date-causal): an issuance absence
    known at the cutoff resets immediately at that decision date; a missing
    outcome is discovered at its reveal date (u+2).  Every queue entry carries
    its SOURCE segment; a reset discards pending entries from the old segment
    and starts a new segment (no silent relabeling).
"""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Callable, Sequence

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import r2_env  # noqa: F401  (thread limits before numeric imports)

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import r2_common as C  # noqa: E402


def frozen_scale(prefix_scores: Sequence[float], *, allow_neutral: bool = False,
                 context: str = "") -> tuple[float, dict]:
    values = np.asarray(list(prefix_scores), dtype=float)
    finite = values[np.isfinite(values)]
    info = {"n_prefix": int(len(values)), "n_finite": int(len(finite)),
            "n_nonfinite": int(len(values) - len(finite)), "context": context}
    if len(finite) == 0 or len(finite) != len(values):
        if not allow_neutral:
            raise RuntimeError(f"reference prefix not finite/nonempty ({info})")
        info["neutral_scale_used"] = True
        return 1.0, info
    scale = float(np.max(np.abs(finite)) * np.sqrt(3.0))
    if not np.isfinite(scale) or scale <= 0.0:
        if not allow_neutral:
            raise RuntimeError(f"non-positive frozen scale ({info})")
        info["neutral_scale_used"] = True
        return 1.0, info
    return scale, info


AST_KERNEL_VENDOR = Path(__file__).resolve().parent / "vendor" / "saocp_ast_kernel.py"
AST_KERNEL_SHA = "3cede45f95ecad16ff9f77e8c830b5ac1d66fbc0b148bdce77f08d8409e47eef"


def load_ast_kernel(repo: Path | str) -> tuple[object, object, dict]:
    """Production numerical route: the admitted verified AST kernel.

    The vendored copy is byte-identical to
    scripts/card10_baselines/saocp_ast_kernel.py (hash-verified here); its
    public loader performs the pinned Git/freeze/AST verification and refuses
    dirty or wrong identities BEFORE any compile/exec.  Backend label:
    verified_wrapper_ast_kernel.  The calendar-batched layer above it is a NEW
    adapter; no public-runtime-parity claim is made.
    """
    vendored = Path(AST_KERNEL_VENDOR)
    if not vendored.is_file():
        raise FileNotFoundError(f"vendored AST kernel missing: {vendored}")
    sha = C.sha256_file(vendored)
    if sha != AST_KERNEL_SHA:
        raise RuntimeError(f"vendored AST kernel hash mismatch: {sha}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("r2_vendor_saocp_ast_kernel", vendored)
    module = importlib.util.module_from_spec(spec)
    sys.modules["r2_vendor_saocp_ast_kernel"] = module  # dataclasses need the module registered
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("r2_vendor_saocp_ast_kernel", None)
        raise
    bundle = module.load_saocp_ast_kernel(Path(repo))
    git_info = C.git_probe(Path(repo))
    C.assert_pinned_git_identity(git_info, C.SAOCP_COMMIT, C.SAOCP_TREE)
    meta = {"backend": "verified_wrapper_ast_kernel",
            "wrapper_sha256": sha,
            "wrapper_source": "scripts/card10_baselines/saocp_ast_kernel.py (vendored unchanged)",
            "repo": str(repo),
            "git": git_info,
            "parity_claim": "none: new calendar-batched wrapper; no public runtime parity"}
    return bundle, module, meta


def ast_kernel_factory(bundle, module):
    def make(scale: float):
        return module.make_minimal_saocp_kernel_state(bundle, coverage=C.SAOCP_COVERAGE,
                                                      lifetime=C.SAOCP_LIFETIME, horizon=1,
                                                      scale=float(scale))
    return make


def load_pinned_kernel(kernel_path: Path | str, *, require_pinned_identity: bool = True) -> tuple[type, dict]:
    """Public-kernel import route (kept for the record; the native smoke FAILED
    on this route because the public dependency graph needs Merlion - see the
    protocol chronology).  Formal execution uses ``load_ast_kernel`` instead.

    Guards against importlib cache reuse; git identity is enforced fail-closed
    when ``require_pinned_identity`` is true (fixtures may disable it and must
    label the result test-only).
    """
    path = Path(kernel_path)
    if not path.is_file():
        raise FileNotFoundError(f"pinned kernel not found: {path}")
    sha = C.sha256_file(path)
    if sha != C.SAOCP_SHA:
        raise RuntimeError(f"pinned kernel hash mismatch: {sha} != {C.SAOCP_SHA}")
    package_root = path.parents[1]
    for name in [m for m in list(sys.modules) if m == "online_conformal" or m.startswith("online_conformal.")]:
        del sys.modules[name]
    added = False
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
        added = True
    try:
        module = importlib.import_module("online_conformal.saocp")
    finally:
        if added:
            sys.path.remove(str(package_root))
    actual = Path(getattr(module, "__file__", "")).resolve()
    if actual != path.resolve():
        raise RuntimeError(f"imported module identity mismatch: {actual} != {path}")
    if C.sha256_file(actual) != C.SAOCP_SHA:
        raise RuntimeError("imported module hash mismatch after import")
    repo_root = None
    candidate = package_root
    for _ in range(4):
        if (candidate / ".git").exists():
            repo_root = candidate
            break
        candidate = candidate.parent
    git_info = C.git_probe(repo_root) if repo_root else {"ok": False, "note": "no .git found"}
    if require_pinned_identity:
        C.assert_pinned_git_identity(git_info, C.SAOCP_COMMIT, C.SAOCP_TREE)
    else:
        git_info = dict(git_info, test_only=True, identity_enforced=False)
    kernel_cls = getattr(module, "SAOCP")
    meta = {"path": str(actual), "sha256": sha, "package_root": str(package_root),
            "admitted_pin": {"commit": C.SAOCP_COMMIT, "tree": C.SAOCP_TREE},
            "git": git_info, "wrapper": "card10.segan_r2.calendar_batch.v3",
            "parity_claim": "none: new calendar-batched wrapper around the pinned kernel"}
    return kernel_cls, meta


def default_kernel_factory(kernel_cls: type) -> Callable[[float], object]:
    def make(scale: float):
        return kernel_cls(horizon=1, max_scale=float(scale), lifetime=C.SAOCP_LIFETIME,
                          coverage=C.SAOCP_COVERAGE)
    return make


class StubKernel:
    """Deterministic stub with the kernel call surface (explicitly synthetic)."""

    def __init__(self, scale: float):
        self.scale = float(scale)
        self.count = 0
        self.total = 0.0

    def update(self, ground_truth: pd.Series, forecast: pd.Series, horizon: int) -> None:
        values = np.abs(ground_truth.to_numpy(float) - forecast.to_numpy(float))
        self.count += len(values)
        self.total += float(values.sum())

    def predict(self, horizon: int) -> tuple[float, float]:
        mean = self.total / self.count if self.count else 0.0
        return -mean, mean


class CalendarBatchSaocp:
    def __init__(self, prefix_by_slot: dict[int, Sequence[float]],
                 kernel_factory: Callable[[float], object], origin: str, *,
                 allow_neutral_scale: bool = False):
        self.origin = origin
        self.prefix_by_slot = {int(k): list(v) for k, v in prefix_by_slot.items()}
        self.kernel_factory = kernel_factory
        self.scales: dict[int, float] = {}
        self.prefix_info: dict[int, dict] = {}
        for slot, vals in self.prefix_by_slot.items():
            scale, info = frozen_scale(vals, allow_neutral=allow_neutral_scale,
                                       context=f"{origin}:slot{slot}")
            self.scales[slot] = scale
            self.prefix_info[slot] = info
        self.kernels: dict[int, object] = {}
        self.segment = -1
        self._build()
        self.queue: list[dict] = []
        self.ledger: list[dict] = []
        self.thresholds: dict[tuple[str, int], float] = {}
        self.reset_events: list[dict] = []
        self.decisions: list[str] = []

    def _build(self) -> None:
        self.kernels = {}
        for slot, prefix in self.prefix_by_slot.items():
            kernel = self.kernel_factory(self.scales[slot])
            finite = [float(v) for v in prefix if np.isfinite(float(v))]
            if finite:
                kernel.update(pd.Series(finite), pd.Series(np.zeros(len(finite))), horizon=1)
            self.kernels[slot] = kernel

    def reset(self, at_date: str, reason: str, *, kind: str) -> None:
        """Shared date-causal reset: discards pending old-segment scores."""
        dropped = [q for q in self.queue if q["segment"] == self.segment]
        self.queue = [q for q in self.queue if q["segment"] != self.segment]
        self.segment += 1
        self._build()
        self.reset_events.append({"at_date": str(at_date), "reason": reason, "kind": kind,
                                  "dropped_pending_scores": len(dropped),
                                  "new_segment": self.segment})

    def enqueue(self, source_utc: pd.Timestamp, source_local_date: str, slot: int,
                abs_score: float) -> None:
        score = float(abs_score)
        if not np.isfinite(score):
            raise ValueError("non-finite score refused")
        reveal = (pd.Timestamp(source_local_date)
                  + pd.DateOffset(days=C.FEEDBACK_LAG_DAYS)).date().isoformat()
        self.queue.append({"source_utc": pd.Timestamp(source_utc),
                           "source_local_date": str(source_local_date),
                           "slot": int(slot), "abs_score": score,
                           "reveal_local_date": reveal, "segment": self.segment})

    def begin(self) -> None:
        """Align the UQ segment counter with the allocator walks (first = 0)."""
        if self.segment == -1:
            self.segment = 0

    def decision(self, local_date: str) -> dict[int, float]:
        self.begin()
        due = sorted([q for q in self.queue if q["reveal_local_date"] <= str(local_date)
                      and q["segment"] == self.segment], key=lambda q: q["source_utc"])
        for item in due:
            self.kernels[item["slot"]].update(pd.Series([item["abs_score"]]),
                                              pd.Series([0.0]), horizon=1)
            self.ledger.append({"origin": self.origin, "slot": item["slot"],
                                "segment": item["segment"],
                                "source_utc": str(item["source_utc"]),
                                "source_local_date": item["source_local_date"],
                                "reveal_local_date": item["reveal_local_date"],
                                "applied_decision_date": str(local_date),
                                "abs_score": item["abs_score"]})
        self.queue = [q for q in self.queue if q not in due]
        day_thresholds: dict[int, float] = {}
        for slot, kernel in self.kernels.items():
            threshold = float(kernel.predict(1)[1])
            if not np.isfinite(threshold):
                raise RuntimeError(f"non-finite threshold on {local_date} slot {slot}")
            day_thresholds[slot] = threshold
            self.thresholds[(str(local_date), slot)] = threshold
        self.decisions.append(str(local_date))
        return day_thresholds

    def pending_count(self) -> int:
        return len(self.queue)
