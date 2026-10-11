"""R2 (SEGAN R3) shared utilities — v3 (post R03B review).

Frozen end-to-end date contract:
  * seven exposed NYISO origins: reference years 2018-2024, evaluation calendar
    January 2 - December 31 of the following test year;
  * local hour slot = (local_date, local_hour); both folds of a repeated hour
    share that day's frozen slot threshold and each contributes one real hour;
  * T_planned = declared calendar length; B = round(0.15 * T_planned) is fixed
    BEFORE any evaluation support is read;
  * every W3/W7/W14 and physical window uses CONSECUTIVE LOCAL CALENDAR DATES
    within one decision-time segment (helper below).

Thread limits are set before the numeric imports (R03B item 10/43).
Pure module: no side effects other than the declared thread-limit env vars.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import r2_env  # noqa: F401  (thread limits before numeric imports)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TZ = "America/New_York"
ZONES = ("CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL", "MHK VL",
         "MILLWD", "N.Y.C.", "NORTH", "WEST")
LAG_HOURS = (72, 168)
RIDGE_PENALTY = 10.0
TRAIN_DAYS = 365
MIN_TRAIN_ROWS = 1000
FEEDBACK_LAG_DAYS = 2
SAOCP_LIFETIME = 8
SAOCP_COVERAGE = 0.90
BUDGET_FRAC = 0.15
ELL = 2
MEMORY = 6
THETA = 1.0
K2_SERVICE_THRESHOLD = 0.9
K2_COUNT = 2
PERSIST_THRESHOLD = 0.5
ETA_GRID = (0.0, 0.5, 1.0, 2.0)
COST_RATIO_GRID = (0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
REFERENCE_QUANTILE = 0.95
N_BOOT = 10_000
BOOT_SEED = 20261011
DELTA_MATERIAL = 0.01

EVIDENCE_ROOT = Path(__file__).resolve().parents[4] / "literature-infra" / "card10-workers" / "codex-r78fix-r80"
R2 = Path(__file__).resolve().parents[1]

RAW_NYISO_SERVER = "/home/icdm/card10_confirmatory/data/nyiso_zonal_2017_2025.csv.gz"
RAW_NYISO_SHA = "d9e492532e22bc3f41c6967acf99cc5b49a49ae2e5ba3ae3b7dbc8092d9e5d0d"
SAOCP_SERVER = "/home/icdm/card10_external/online_conformal/online_conformal/saocp.py"
SAOCP_SHA = "83f8845559786452aafabfafdf6c136ceaf9b9b3557fc5f8f9267fcec04d386f"
SAOCP_COMMIT = "95790a9b9e4061166e6d7a8a13282e019b1a8663"
SAOCP_TREE = "77348e5fe7f2bb3ed5a031aaad9874b6a5ab2d7b"
SAOCP_LOCAL_FALLBACK = Path("D:/mypaper/literature-infra/card10-data/online_conformal_r85/online_conformal/saocp.py")

REFERENCE_YEARS = tuple(range(2018, 2025))
TEST_YEARS = tuple(range(2019, 2026))


# --------------------------------------------------------------------------
# identity / io
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_fingerprint(code_files, extra: dict | None = None) -> str:
    """One definitive code+identity fingerprint (R03C late review item 9).

    Hashes {relative code file: actual file sha256} plus the supplied extra
    identities verbatim (fields ending in ``*_sha256`` are KEPT here - this is
    not the canonical-payload helper that strips self-hashes).
    """
    payload = {"code": {str(name): sha256_file(Path(__file__).resolve().parent / name)
                        for name in code_files},
               "extra": dict(extra or {})}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def canonical_payload_sha256(obj: Any) -> str:
    """Hash of a canonical payload EXCLUDING every ``*_sha256`` field.

    Integer and string keys are NORMALIZED to strings BEFORE sorting/encoding:
    point provenance carries integer lag keys (72, 168), and a JSON round trip
    turns them into strings, so a naive encoding would give two different
    canonical hashes for the same object.  A key that collides after
    normalization is an ERROR, never a silent overwrite (R03E item 13).
    """
    def strip(value):
        if isinstance(value, dict):
            items = sorted(value.items(), key=lambda kv: str(kv[0]))
            out: dict[str, Any] = {}
            for key, item in items:
                name = str(key)
                if name.endswith("_sha256"):
                    continue
                if name in out:
                    raise RuntimeError(f"canonical payload: duplicate key after string "
                                       f"normalization: {name!r}")
                out[name] = strip(item)
            return out
        if isinstance(value, (list, tuple)):
            return [strip(v) for v in value]
        return value
    blob = json.dumps(strip(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as gz:
        gz.write(buffer.getvalue().encode("utf-8"))


def git_probe(repo_root: Path) -> dict:
    """Clean-tree Git identity probe (local subprocess, no remote access).

    Return codes are checked; a failed command yields ok=False rather than an
    empty string that could masquerade as a clean tree (R03B second review 2).
    """
    def run(*args: str):
        proc = subprocess.run(["git", "-C", str(repo_root), *args],
                              capture_output=True, text=True, timeout=60)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()[:200]
    info = {"repo_root": str(repo_root)}
    try:
        rc_head, head, err_head = run("rev-parse", "HEAD")
        rc_tree, tree, err_tree = run("rev-parse", "HEAD^{tree}")
        rc_status, dirty, err_status = run("status", "--porcelain")
        info.update({
            "git_head": head if rc_head == 0 else None,
            "git_tree": tree if rc_tree == 0 else None,
            "clean": (dirty == "") if rc_status == 0 else None,
            "return_codes": {"head": rc_head, "tree": rc_tree, "status": rc_status},
            "dirty_entries": dirty.splitlines()[:5],
            "errors": [e for e in (err_head, err_tree, err_status) if e],
            "ok": bool(rc_head == 0 and rc_tree == 0 and rc_status == 0),
        })
    except Exception as exc:  # recorded, never hidden
        info.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return info


def assert_pinned_git_identity(git_info: dict, commit: str, tree: str) -> None:
    """Fail closed unless commit/tree match the pin and the worktree is clean."""
    problems = []
    if not git_info.get("ok"):
        problems.append("git probe failed")
    if git_info.get("git_head") != commit:
        problems.append(f"git HEAD {git_info.get('git_head')} != {commit}")
    if git_info.get("git_tree") != tree:
        problems.append(f"git tree {git_info.get('git_tree')} != {tree}")
    if git_info.get("clean") is not True:
        problems.append(f"worktree not clean: {git_info.get('dirty_entries')}")
    if problems:
        raise RuntimeError("pinned git identity failure: " + "; ".join(problems))


def manifest_for(paths: Sequence[Path], root: Path, extra: dict | None = None) -> dict:
    entries = {}
    for path in paths:
        path = Path(path)
        entries[str(path.relative_to(root))] = {"sha256": sha256_file(path),
                                                "bytes": path.stat().st_size}
    payload = {"root": str(root), "entries": entries, "extra": extra or {}}
    payload["manifest_canonical_sha256"] = canonical_payload_sha256(payload)
    return payload


def key_set(frame: pd.DataFrame, keys: Sequence[str]) -> set:
    return set(map(tuple, frame[list(keys)].to_numpy().tolist()))


def assert_same_support(left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str],
                        left_name: str, right_name: str) -> dict:
    """Exact support equality: no missing, extra or duplicate key rows."""
    for name, frame in ((left_name, left), (right_name, right)):
        dupes = int(frame.duplicated(list(keys)).sum())
        if dupes:
            raise RuntimeError(f"{name}: {dupes} duplicate key rows on {list(keys)}")
    lset, rset = key_set(left, keys), key_set(right, keys)
    missing, extra = sorted(rset - lset), sorted(lset - rset)
    if missing or extra:
        raise RuntimeError(f"support mismatch {left_name} vs {right_name}: "
                           f"missing={len(missing)} extra={len(extra)} "
                           f"first_missing={missing[:3]} first_extra={extra[:3]}")
    return {"rows_left": int(len(left)), "rows_right": int(len(right)), "matched": len(lset)}


def verify_manifest(manifest: dict, root: Path, required: Sequence[str]) -> dict:
    """Fail-closed manifest verification: missing/modified/extra entries reject."""
    problems: list[str] = []
    entries = manifest.get("entries", {})
    for rel in required:
        if rel not in entries:
            problems.append(f"missing required artifact: {rel}")
            continue
        path = Path(root) / rel
        if not path.is_file():
            problems.append(f"artifact absent on disk: {rel}")
            continue
        if sha256_file(path) != entries[rel]["sha256"]:
            problems.append(f"artifact hash mismatch: {rel}")
    if manifest.get("manifest_canonical_sha256") != canonical_payload_sha256(
            {k: v for k, v in manifest.items() if k != "manifest_canonical_sha256"}):
        problems.append("manifest canonical hash mismatch")
    return {"ok": not problems, "problems": problems}


# --------------------------------------------------------------------------
# calendar contract
# --------------------------------------------------------------------------

def declared_calendar(test_year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{test_year}-01-02", f"{test_year}-12-31", freq="D")


def planned_cap(test_year: int) -> int:
    return int(round(BUDGET_FRAC * len(declared_calendar(test_year))))


def midnight_before(local_date: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(local_date).normalize()


def newest_label_boundary(target_local_date: pd.Timestamp) -> pd.Timestamp:
    """Exactly local midnight STARTING Jan-1-style t-1: labels strictly before it
    (for t = Jan 2 this is Jan 1 00:00, so Dec 31 labels are eligible)."""
    return midnight_before(pd.Timestamp(target_local_date)) - pd.DateOffset(days=1)


def lag_provenance_ok(target_utc: pd.Timestamp, source_utc: pd.Timestamp, lag_hours: int) -> bool:
    if pd.Timestamp(source_utc) != pd.Timestamp(target_utc) - pd.Timedelta(hours=lag_hours):
        return False
    min_age = {72: 2, 168: 6}[lag_hours]
    t_loc = pd.Timestamp(target_utc).tz_convert(TZ).normalize().tz_localize(None)
    s_loc = pd.Timestamp(source_utc).tz_convert(TZ).normalize().tz_localize(None)
    return int((t_loc - s_loc).days) >= min_age


# --------------------------------------------------------------------------
# date-aware windows and metrics
# --------------------------------------------------------------------------

def consecutive_local_windows(dates: Sequence[str], segments: np.ndarray,
                              width: int) -> list[np.ndarray]:
    """Windows of `width` CONSECUTIVE LOCAL DATES inside one segment.

    A candidate window qualifies iff (a) every member shares the same
    decision-time segment id and (b) consecutive members are exactly one
    calendar day apart.  Index-consecutive but calendar-gapped windows are
    rejected.  `dates` must be sorted ISO date strings.
    """
    n = len(dates)
    days = [pd.Timestamp(d) for d in dates]
    windows: list[np.ndarray] = []
    if n < width:
        return windows
    same_segment_start = 0
    for start in range(0, n - width + 1):
        ok = True
        seg = segments[start]
        for k in range(width - 1):
            if segments[start + k + 1] != seg:
                ok = False
                break
            if (days[start + k + 1] - days[start + k]).days != 1:
                ok = False
                break
        if ok:
            windows.append(np.arange(start, start + width))
    return windows


def worst_window(values: np.ndarray, dates: Sequence[str], segments: np.ndarray,
                 width: int, mode: str = "min") -> float:
    windows = consecutive_local_windows(dates, segments, width)
    if not windows:
        return math.nan
    means = [float(np.mean(values[w])) for w in windows]
    return float(min(means)) if mode == "min" else float(max(means))


def observable_metrics(q_base: np.ndarray, actions: np.ndarray, observed: np.ndarray,
                       dates: Sequence[str], segments: np.ndarray) -> dict:
    """Metrics on OBSERVED days only; base q and served s kept separate."""
    n = len(q_base)
    served = np.where(actions, 1.0, q_base)
    mask = observed.astype(bool)
    row: dict[str, Any] = {
        "n_declared_days": int(n),
        "n_observed_days": int(mask.sum()),
        "action_total_all_declared": int(actions.sum()),
        "action_rate_all_declared": float(actions.sum() / n) if n else math.nan,
        "action_total_observed": int(actions[mask].sum()),
        "action_rate_observed": float(actions[mask].sum() / mask.sum()) if mask.sum() else math.nan,
    }
    if not mask.any():
        return row
    obs_dates = [d for d, m in zip(dates, mask) if m]
    obs_segments = segments[mask]
    q_obs, a_obs = q_base[mask], actions[mask]
    s_obs = served[mask]
    rescue = (q_obs < 1.0) & a_obs
    row.update({
        "mean_service": float(np.mean(s_obs)),
        "mean_base_coverage": float(np.mean(q_obs)),
        "mean_shortfall": float(np.mean(1.0 - s_obs)),
        "service_worst3": worst_window(s_obs, obs_dates, obs_segments, 3, "min"),
        "service_worst7": worst_window(s_obs, obs_dates, obs_segments, 7, "min"),
        "service_worst14": worst_window(s_obs, obs_dates, obs_segments, 14, "min"),
        "direct_rescue_count": int(rescue.sum()),
        "rescue_precision": float(rescue.sum() / a_obs.sum()) if a_obs.sum() else math.nan,
        "rescued_shortfall_sum": float(np.sum(1.0 - q_obs[rescue])),
    })
    for lam in (0.25, 0.5, 1.0, 2.0, 4.0):
        row[f"L_lambda_{lam:g}"] = float(row["mean_shortfall"]
                                         + lam * row["action_rate_observed"])
    return row


def paired_origin_bootstrap(diffs: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """IID-over-origin bootstrap of the paired mean difference.

    The draws are produced as ONE ``(n_boot, n)`` choice matrix instead of
    ``n_boot`` consecutive length-n calls.  This is adopted ONLY because the
    exact-draw equivalence for the production (n_boot=10,000, seed=20261011)
    setting is verified bit-for-bit by guard P15 (same generator stream, same
    draws, same effect/CI/share); if that check ever fails the loop form must be
    restored.  No estimand, seed or size change.
    """
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    draws = rng.choice(diffs, size=(int(n_boot), n), replace=True).mean(axis=1)
    return {"effect": float(np.mean(diffs)),
            "ci_lo": float(np.quantile(draws, 0.025)), "ci_hi": float(np.quantile(draws, 0.975)),
            "share_positive": float(np.mean(draws > 0.0)), "n_origins": int(n),
            "n_boot": int(n_boot), "seed": int(seed),
            "draw_implementation": "vectorized_single_choice_matrix_v1"}


def paired_origin_bootstrap_loop_reference(diffs: np.ndarray, n_boot: int = N_BOOT,
                                           seed: int = BOOT_SEED) -> np.ndarray:
    """Reference (pre-R03E) draw loop; kept ONLY as the P15 equivalence oracle."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    return np.asarray([np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(n_boot)])


def leave_one_out(diffs: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, label in enumerate(labels):
        keep = np.ones(len(diffs), dtype=bool)
        keep[i] = False
        out[str(label)] = float(np.mean(diffs[keep]))
    return out
