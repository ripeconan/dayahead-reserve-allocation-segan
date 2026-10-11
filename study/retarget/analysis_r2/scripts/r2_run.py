"""R2 formal runner — v4 (post R03E review).  Executed only after the gate.

Every origin writes a complete artifact set (payload, hourly/daily frames,
allocator walks, adapter ledger/thresholds, point audit, bridge grid, kernel
identity) inside its own directory, bound by an EXTERNAL manifest; resume
verifies the manifest and rejects missing/modified artifacts (retaining failed
attempts).  Aggregation refuses to compare a subset against full-cohort
references and enumerates missing units explicitly.

ADMISSION (R03E items 2/11): resume AND aggregate share ONE validator
(`admit_origin`) that requires the manifest sidecar, the full artifact
manifest, the CURRENT code fingerprint, the current frozen protocol AND every
dated amendment (with verified SHA sidecars), the admitted raw SHA, the
current clean pinned checkout, the stored kernel identity, the full numeric
environment identity, `payload.year == directory year`, the canonical payload
consistency and finite endpoint schema.  Malformed per-origin JSON is recorded
as invalid-with-reasons, never raised.  Formal aggregation requires --raw and
--checkout; the explicitly labelled `--fixture-aggregate` mode bypasses only
the formal raw SHA / checkout gate for tiny synthetic metadata fixtures.

Exact server commands (root only; do NOT run from this session):

  cd <analysis_r2>/scripts
  taskset -c 0-7,9-31 nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    /usr/bin/python3 r2_run.py \
      --raw /home/icdm/card10_confirmatory/data/nyiso_zonal_2017_2025.csv.gz \
      --checkout /home/icdm/card10_external/online_conformal \
      --out <analysis_r2>/outputs/formal_r2 --all
  /usr/bin/python3 r2_run.py --out <analysis_r2>/outputs/formal_r2 --aggregate \
      --raw /home/icdm/card10_confirmatory/data/nyiso_zonal_2017_2025.csv.gz \
      --checkout /home/icdm/card10_external/online_conformal
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

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
import r2_pipeline as PL  # noqa: E402
import r2_saocp_batch as SB  # noqa: E402
import traceback  # noqa: E402

PROTOCOL_PATH = C.R2 / "R2_PROTOCOL_FROZEN_2026-10-11.md"
AMENDMENT_GLOB = "R2_PROTOCOL_AMENDMENT_*.md"

CODE_FILES = ("r2_env.py", "r2_common.py", "r2_pointcorr.py", "r2_saocp_batch.py",
              "r2_allocators.py", "r2_bridge.py", "r2_pipeline.py", "r2_run.py",
              "vendor/saocp_ast_kernel.py")

# manifest.json cannot hash itself; it is verified separately via its sidecar.
REQUIRED_ARTIFACTS = (
    "origin.json", "hourly.csv.gz", "daily.csv.gz", "adapter_ledger.csv.gz",
    "adapter_thresholds.csv.gz", "point_audit.csv.gz", "bridge_grid.csv.gz",
    "bridge_daily.csv.gz", "reference_scores.csv.gz", "reference_residuals.csv.gz",
    "kernel_meta.json",
) + tuple(f"walk_{p}.csv.gz" for p in PL.POLICIES)  # none/P/F1-F5 (single source of truth)


def _sidecar_state(path: Path) -> tuple[str | None, bool]:
    """(recorded sha token, sidecar matches the CURRENT bytes)."""
    side = path.with_suffix(path.suffix + ".sha256")
    if not side.is_file():
        return None, False
    tokens = side.read_text(encoding="ascii").strip().split()
    recorded = tokens[0] if tokens else None
    return recorded, bool(recorded and recorded == C.sha256_file(path))


def _protocol_identity(*, require_sidecar: bool) -> dict:
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError("frozen protocol document missing")
    recorded, ok = _sidecar_state(PROTOCOL_PATH)
    if require_sidecar and not ok:
        raise RuntimeError("protocol SHA sidecar missing or mismatched")
    return {"sha256": C.sha256_file(PROTOCOL_PATH), "sidecar_ok": ok,
            "sidecar_sha256": recorded}


def _amendment_identities(*, require_sidecar: bool) -> dict:
    """Every dated amendment present, bound by bytes AND by its SHA sidecar."""
    out: dict[str, dict] = {}
    for path in sorted(C.R2.glob(AMENDMENT_GLOB)):
        recorded, ok = _sidecar_state(path)
        if require_sidecar and not ok:
            raise RuntimeError(f"amendment sidecar missing or mismatched: {path.name}")
        out[path.name] = {"sha256": C.sha256_file(path), "sidecar_ok": ok,
                          "sidecar_sha256": recorded}
    if not out:
        raise RuntimeError("no dated protocol amendment present")
    return out


def load_raw(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, low_memory=False)
    index = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.tz_convert(C.TZ)
    actual = pd.DataFrame(frame[[f"{z}_actual" for z in C.ZONES]].to_numpy(float),
                          index=pd.DatetimeIndex(index), columns=list(C.ZONES))
    forecast = pd.DataFrame(frame[[f"{z}_forecast" for z in C.ZONES]].to_numpy(float),
                            index=pd.DatetimeIndex(index), columns=list(C.ZONES))
    if actual.index.duplicated().any() or not actual.index.is_monotonic_increasing:
        raise ValueError("raw index duplicated or unsorted")
    return actual, forecast


def _fingerprint() -> str:
    """Current code + document identity (protocol AND every dated amendment).

    The amendment bytes describe the schema and are hashed here; this function
    never appears inside those documents, so there is no static-hash
    circularity (R03E item 4).
    """
    return C.build_fingerprint(CODE_FILES,
                               extra={"protocol": _protocol_identity(require_sidecar=False),
                                      "amendments": _amendment_identities(require_sidecar=False),
                                      "ast_wrapper_sha256": SB.AST_KERNEL_SHA,
                                      "raw_sha256": C.RAW_NYISO_SHA})


def _identity_values() -> dict:
    import numpy
    import pandas
    import sklearn
    import scipy
    import sys
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "pandas": pandas.__version__, "sklearn": sklearn.__version__,
            "scipy": scipy.__version__,
            "thread_env": {k: __import__("os").environ.get(k) for k in
                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def _environment_record() -> dict:
    import os
    import platform
    import sklearn
    return {"python": __import__("sys").version.split()[0], "platform": platform.platform(),
            "hostname": platform.node(), "numpy": __import__("numpy").__version__,
            "pandas": __import__("pandas").__version__, "sklearn": sklearn.__version__,
            "thread_env": {k: os.environ.get(k) for k in
                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "identities": _identity_values(),
            "affinity": (sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None),
            "protocol": _protocol_identity(require_sidecar=False),
            "amendments": _amendment_identities(require_sidecar=False)}


def _validate_endpoints(payload: dict) -> list[str]:
    """Required FINITE endpoints and exact policy/eta/kappa key identity."""
    import math as _math
    problems = []
    policies = tuple(PL.POLICIES)
    metrics_all = payload.get("allocator_metrics", {})
    if set(metrics_all) != set(policies):
        problems.append(f"allocator metric policies {sorted(metrics_all)} != {sorted(policies)}")
    for policy in policies:
        metrics = metrics_all.get(policy)
        if not metrics:
            problems.append(f"missing allocator metrics: {policy}")
            continue
        for key in ("mean_service", "service_worst3", "service_worst7", "service_worst14"):
            value = metrics.get(key)
            if value is None or not _math.isfinite(float(value)):
                problems.append(f"{policy}: non-finite endpoint {key}")
    grid = payload.get("bridge_grid")
    expected_cells = {(policy, eta, kappa) for policy in policies
                      for eta in C.ETA_GRID for kappa in C.COST_RATIO_GRID}
    if not grid:
        problems.append("bridge grid missing")
        return problems
    keys = [(g["policy"], g["eta"], g["kappa"]) for g in grid]
    if len(keys) != len(set(keys)):
        problems.append("bridge grid has duplicate cells")
    if set(keys) != expected_cells:
        problems.append(f"bridge grid keys incomplete: {len(set(keys))} != {len(expected_cells)}")
    for cell in grid:
        for field in ("worst7_deficit_MWh_complete_days",
                      "worst7_adequacy_fraction_complete_days",
                      "uncovered_MWh_finite_hours", "extra_committed_MWh",
                      "full_cost_MWh_equivalent"):
            value = cell.get(field)
            if value is None or not _math.isfinite(float(value)):
                problems.append(f"non-finite physical endpoint {field} in {cell.get('policy')}/{cell.get('eta')}/{cell.get('kappa')}")
                break
    return problems


ADMISSION_MODES = {
    # real unit, admitted raw + live pinned checkout + verified document sidecars
    "formal": {"raw_gate": "pinned", "checkout_gate": "pinned", "require_sidecars": True},
    # real pipeline unit computed on a synthetic raw file (guard P7): the raw SHA
    # pin is replaced by "stored must equal the supplied file", everything else stays strict
    "fixture_raw": {"raw_gate": "supplied", "checkout_gate": "pinned", "require_sidecars": True},
    # tiny SYNTHETIC METADATA unit (no fit, no kernel): must self-declare
    # fixture=true and is labelled as bypassing the raw/checkout identity gates
    "synthetic_fixture": {"raw_gate": "supplied", "checkout_gate": "none",
                          "require_sidecars": False},
}


def _checkout_problems(extra: dict, checkout: Path | None, checkout_meta: dict | None,
                       problems: list[str], notes: dict, checkout_gate: str) -> None:
    """Current clean pinned checkout identity (never a stale stored path only)."""
    if checkout_gate == "none":
        notes["checkout_gate"] = "none"
        notes["labels"].append("synthetic fixture: checkout/kernel git identity unchecked (labelled)")
        return
    if checkout is None:
        problems.append("checkout is required for this admission mode")
        return
    if extra.get("checkout") != str(checkout):
        problems.append(f"checkout path mismatch: stored {extra.get('checkout')} != {checkout}")
    meta = checkout_meta
    if meta is None:
        try:
            _bundle, _module, meta = SB.load_ast_kernel(Path(checkout))
        except Exception as exc:
            problems.append(f"current checkout identity check failed: {type(exc).__name__}: {exc}")
            notes["checkout_checked"] = False
            return
    if "error" in meta:
        problems.append(f"current checkout identity check failed: {meta['error']}")
        notes["checkout_checked"] = False
        return
    git = meta.get("git", {}) or {}
    if git.get("ok") is not True:
        problems.append(f"checkout git probe failed: {git.get('errors') or git}")
    if git.get("clean") is not True:
        problems.append(f"current checkout not clean: {git.get('dirty_entries')}")
    if git.get("git_head") != C.SAOCP_COMMIT or git.get("git_tree") != C.SAOCP_TREE:
        problems.append(f"current checkout not at the pinned commit/tree: "
                        f"{git.get('git_head')}/{git.get('git_tree')}")
    notes["checkout_checked"] = True
    notes["checkout_head"] = git.get("git_head")


def admit_origin(origin_dir: Path, *, year: int, fingerprint: str, mode: str = "formal",
                 raw: Path | None = None, checkout: Path | None = None,
                 checkout_meta: dict | None = None) -> dict:
    """THE shared admission validator used by run_origin resume AND aggregate.

    Requires (mode="formal"): verified manifest sidecar, complete artifact
    manifest, CURRENT code fingerprint, current protocol AND every dated
    amendment with verified sidecars, the admitted raw SHA, the current clean
    pinned checkout, the stored kernel identity, the full numeric environment
    identity, payload year == directory year, canonical payload consistency and
    the finite endpoint schema.  `fixture_raw` relaxes ONLY the raw SHA pin;
    `synthetic_fixture` additionally relaxes the checkout/kernel-git gates and
    requires the unit to self-declare ``fixture = true``.  Every relaxation is
    recorded in ``notes["labels"]``.

    Never raises: malformed JSON, missing files and identity drift all come
    back as explicit reasons.  Returns
    ``{"problems", "notes", "payload", "manifest", "admissible"}``.
    """
    if mode not in ADMISSION_MODES:
        raise ValueError(f"unknown admission mode {mode!r}")
    gates = ADMISSION_MODES[mode]
    raw_gate, checkout_gate = gates["raw_gate"], gates["checkout_gate"]
    if mode == "synthetic_fixture" and raw is None:
        raw_gate = "none"
    problems: list[str] = []
    notes: dict = {"mode": mode, "origin_dir": str(origin_dir), "origin": int(year),
                   "raw_gate": raw_gate, "checkout_gate": checkout_gate, "labels": []}
    manifest_path = origin_dir / "manifest.json"
    payload_path = origin_dir / "origin.json"
    for name, path in (("manifest.json", manifest_path), ("origin.json", payload_path),
                       ("kernel_meta.json", origin_dir / "kernel_meta.json")):
        if not path.is_file():
            problems.append(f"missing required file: {name}")
    if problems:
        notes["labels"].append("unit absent/incomplete: rejected before any content read")
        return {"problems": problems, "notes": notes, "payload": None, "manifest": None,
                "admissible": False}
    manifest = payload = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        problems.append(f"malformed manifest.json: {type(exc).__name__}: {exc}")
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception as exc:
        problems.append(f"malformed origin.json: {type(exc).__name__}: {exc}")
    if problems:
        notes["labels"].append("malformed unit metadata: rejected with reasons")
        return {"problems": problems, "notes": notes, "payload": None, "manifest": None,
                "admissible": False}
    if not isinstance(manifest, dict) or not isinstance(payload, dict):
        return {"problems": ["manifest.json/origin.json is not a JSON object"], "notes": notes,
                "payload": None, "manifest": None, "admissible": False}
    # ---- manifest sidecar, artifact manifest, identity binding
    recorded, sidecar_ok = _sidecar_state(manifest_path)
    notes["manifest_sidecar_sha256"] = recorded
    if not sidecar_ok:
        problems.append("manifest sidecar missing or mismatched")
    check = C.verify_manifest(manifest, origin_dir, REQUIRED_ARTIFACTS)
    if not check["ok"]:
        problems.extend(check["problems"][:5])
    extra = manifest.get("extra", {}) or {}
    if extra.get("fingerprint") != fingerprint:
        problems.append("fingerprint mismatch (code/protocol/amendment changed)")
    if extra.get("ast_wrapper_sha256") != SB.AST_KERNEL_SHA:
        problems.append("AST wrapper identity mismatch")
    try:
        current_protocol = _protocol_identity(require_sidecar=gates["require_sidecars"])
        current_amendments = _amendment_identities(require_sidecar=gates["require_sidecars"])
    except Exception as exc:
        problems.append(f"protocol/amendment identity unavailable: {type(exc).__name__}: {exc}")
        current_protocol = current_amendments = None
    stored_env = extra.get("environment") if isinstance(extra.get("environment"), dict) else {}
    if current_protocol is not None and stored_env.get("protocol") != current_protocol:
        problems.append("stored protocol identity != current protocol (bytes or sidecar)")
    if current_amendments is not None and stored_env.get("amendments") != current_amendments:
        problems.append("stored amendment identities != current amendments (bytes or sidecar)")
    stored_ident = stored_env.get("identities")
    if stored_ident != _identity_values():
        problems.append(f"numeric environment identity mismatch: {stored_ident} != {_identity_values()}")
    if mode == "synthetic_fixture":
        payload_env_check = payload.get("environment") if isinstance(payload.get("environment"), dict) else {}
        if stored_env.get("fixture") is not True or payload_env_check.get("fixture") is not True:
            problems.append("synthetic_fixture mode requires a unit that self-declares fixture=true")
        notes["labels"].append("synthetic metadata fixture: no fit and no kernel were used")
    # ---- raw input identity
    if raw_gate == "none":
        notes["raw_checked"] = False
        notes["labels"].append("synthetic fixture: raw SHA deliberately unchecked (labelled)")
    else:
        if raw is None or not Path(raw).is_file():
            problems.append(f"raw input (existing file) required for raw_gate={raw_gate}")
        else:
            sha = C.sha256_file(Path(raw))
            if raw_gate == "pinned" and sha != C.RAW_NYISO_SHA:
                problems.append(f"admitted raw hash mismatch {sha}")
            if extra.get("raw_sha256") != sha:
                problems.append("stored raw hash != supplied raw")
        notes["raw_checked"] = True
    # ---- current kernel checkout identity
    _checkout_problems(extra, checkout, checkout_meta, problems, notes, checkout_gate)
    # ---- stored kernel identity must itself be at the pin (real units only)
    try:
        kernel_meta = json.loads((origin_dir / "kernel_meta.json").read_text(encoding="utf-8"))
    except Exception as exc:
        kernel_meta = {}
        problems.append(f"malformed kernel_meta.json: {type(exc).__name__}: {exc}")
    if kernel_meta:
        if kernel_meta.get("wrapper_sha256") != SB.AST_KERNEL_SHA:
            problems.append("kernel_meta wrapper identity mismatch")
        kgit = kernel_meta.get("git", {}) or {}
        if checkout_gate == "pinned":
            if kgit.get("git_head") != C.SAOCP_COMMIT or kgit.get("git_tree") != C.SAOCP_TREE:
                problems.append("kernel_meta not at the pinned commit/tree")
            if kgit.get("clean") is not True:
                problems.append("kernel_meta checkout not clean")
        else:
            notes["labels"].append("synthetic fixture: kernel_meta git identity unchecked (labelled)")
    # ---- payload identity and schema
    if payload.get("test_year") != year or payload.get("origin") != year:
        problems.append(f"payload year {payload.get('test_year')}/{payload.get('origin')} "
                        f"!= directory year {year}")
    if payload.get("canonical_payload_sha256") != C.canonical_payload_sha256(payload):
        problems.append("canonical payload hash mismatch")
    endpoint_problems = _validate_endpoints(payload)
    if endpoint_problems:
        problems.extend(endpoint_problems[:5])
    payload_env = payload.get("environment") if isinstance(payload.get("environment"), dict) else {}
    if payload_env.get("identities") != _identity_values():
        problems.append("payload numeric environment identity mismatch")
    notes["n_problems"] = len(problems)
    return {"problems": problems, "notes": notes, "payload": payload, "manifest": manifest,
            "admissible": not problems}


def run_origin(test_year: int, raw: Path, out_root: Path, checkout: Path,
               *, fixture_raw: bool = False) -> dict:
    origin_dir = out_root / f"origin_{test_year}"
    origin_dir.mkdir(parents=True, exist_ok=True)
    if not fixture_raw:
        sha = C.sha256_file(raw)
        if sha != C.RAW_NYISO_SHA:
            C.write_json(origin_dir / "FAIL.raw_identity.json",
                         {"origin": test_year, "status": "FAIL", "stage": "raw_identity",
                          "error": f"raw hash mismatch {sha}"})
            return {"origin": test_year, "status": "FAIL", "stage": "raw_identity"}
        try:
            _protocol_identity(require_sidecar=True)
            _amendment_identities(require_sidecar=True)
        except Exception as exc:
            C.write_json(origin_dir / "FAIL.document_identity.json",
                         {"origin": test_year, "status": "FAIL", "stage": "document_identity",
                          "error": f"{type(exc).__name__}: {exc}"})
            return {"origin": test_year, "status": "FAIL", "stage": "document_identity"}
    try:
        fingerprint = _fingerprint()
    except Exception as exc:
        C.write_json(origin_dir / "FAIL.fingerprint.json",
                     {"origin": test_year, "status": "FAIL", "stage": "fingerprint",
                      "error": f"{type(exc).__name__}: {exc}"})
        return {"origin": test_year, "status": "FAIL", "stage": "fingerprint"}
    attempt = f"{time.strftime('%Y%m%dT%H%M%S')}"
    try:
        return _run_origin_body(test_year, raw, out_root, checkout, origin_dir, attempt,
                                fingerprint, formal=not fixture_raw)
    except Exception as exc:
        C.write_json(origin_dir / f"FAIL.{attempt}.json", {
            "origin": test_year, "status": "FAIL", "attempt": attempt,
            "stage": getattr(exc, "stage", "unknown"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
            "fingerprint": fingerprint})
        return {"origin": test_year, "status": "FAIL", "attempt": attempt,
                "error": f"{type(exc).__name__}: {exc}"}


def _run_origin_body(test_year: int, raw: Path, out_root: Path, checkout: Path,
                     origin_dir: Path, attempt: str, fingerprint: str,
                     *, formal: bool = True) -> dict:
    if (origin_dir / "manifest.json").is_file():
        admission = admit_origin(origin_dir, year=test_year, fingerprint=fingerprint,
                                 mode="formal" if formal else "fixture_raw",
                                 raw=raw, checkout=checkout)
        if admission["admissible"]:
            return {"origin": test_year, "status": "SKIP_VERIFIED"}
        resume_reasons = admission["problems"]
        # preserve the old bytes before any recomputation; never overwrite silently
        preserve = origin_dir / "attempts" / attempt
        preserve.mkdir(parents=True, exist_ok=True)
        for item in origin_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, preserve / item.name)
        C.write_json(preserve / "RESUME_REJECTED.json",
                     {"origin": test_year, "attempt": attempt, "reasons": resume_reasons,
                      "admission_notes": admission["notes"]})
    try:
        bundle, ast_module, kernel_meta = SB.load_ast_kernel(checkout)
        factory = SB.ast_kernel_factory(bundle, ast_module)
        actual, forecast = load_raw(raw)
        payload, frames = PL.run_origin_from_frames(actual, forecast, test_year,
                                                    kernel_factory=factory,
                                                    kernel_meta=kernel_meta)
        # ONE environment/document identity record, shared by the payload and the
        # manifest so admission can compare the same object in both places
        environment = _environment_record()
        payload["fingerprint"] = fingerprint
        payload["environment"] = environment
        endpoint_problems = _validate_endpoints(payload)
        if endpoint_problems:
            raise RuntimeError("endpoint validation failed: " + "; ".join(endpoint_problems[:5]))
        payload["canonical_payload_sha256"] = C.canonical_payload_sha256(payload)
        C.write_json(origin_dir / "origin.json", payload)
        C.write_json(origin_dir / "kernel_meta.json", kernel_meta)
        C.write_gzip_csv(frames["hours"], origin_dir / "hourly.csv.gz")
        C.write_gzip_csv(frames["daily"], origin_dir / "daily.csv.gz")
        C.write_gzip_csv(frames["point_audit"].reset_index(), origin_dir / "point_audit.csv.gz")
        C.write_gzip_csv(frames["adapter_ledger"], origin_dir / "adapter_ledger.csv.gz")
        C.write_gzip_csv(frames["adapter_thresholds"], origin_dir / "adapter_thresholds.csv.gz")
        C.write_gzip_csv(pd.DataFrame(payload["bridge_grid"]), origin_dir / "bridge_grid.csv.gz")
        C.write_gzip_csv(frames["bridge_daily"], origin_dir / "bridge_daily.csv.gz")
        C.write_gzip_csv(frames["reference_scores"], origin_dir / "reference_scores.csv.gz")
        C.write_gzip_csv(frames["reference_residuals"], origin_dir / "reference_residuals.csv.gz")
        for policy, walk in frames["walks"].items():
            C.write_gzip_csv(walk, origin_dir / f"walk_{policy}.csv.gz")
        manifest = C.manifest_for([origin_dir / name for name in REQUIRED_ARTIFACTS
                                   if name != "manifest.json"],
                                  origin_dir,
                                  extra={"fingerprint": fingerprint,
                                         "raw_sha256": C.sha256_file(raw),
                                         "ast_wrapper_sha256": SB.AST_KERNEL_SHA,
                                         "checkout": str(checkout),
                                         "environment": environment,
                                         "attempt": attempt})
        C.write_json(origin_dir / "manifest.json", manifest)
        (origin_dir / "manifest.json.sha256").write_text(
            C.sha256_file(origin_dir / "manifest.json") + "\n", encoding="ascii")
        return {"origin": test_year, "status": "PASS", "seconds": payload["seconds"]}
    except Exception as exc:
        C.write_json(origin_dir / f"FAIL.{attempt}.json", {
            "origin": test_year, "status": "FAIL", "attempt": attempt,
            "stage": getattr(exc, "stage", "unknown"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
            "fingerprint": fingerprint})
        return {"origin": test_year, "status": "FAIL", "attempt": attempt,
                "error": f"{type(exc).__name__}: {exc}"}


def aggregate(out_root: Path, *, raw: Path | None = None, checkout: Path | None = None,
              fixture: bool = False) -> dict:
    """Full-cohort aggregation behind the SHARED admission validator.

    Formal mode REQUIRES --raw and --checkout: every unit must pass
    `admit_origin` (current fingerprint, current protocol + every amendment with
    verified sidecars, admitted raw SHA, current clean pinned checkout, stored
    kernel identity, numeric environment identity, year match, canonical payload,
    finite endpoint schema).  A missing or invalid unit refuses the whole
    comparison — units are never silently dropped.  `fixture=True` is an
    explicitly labelled mode for tiny synthetic metadata fixtures only.
    """
    formal = not fixture
    expected_origins = list(C.TEST_YEARS)
    mode = "formal" if formal else "fixture"
    if formal and (raw is None or checkout is None):
        return {"status": "REFUSED", "mode": mode,
                "reason": "--raw and --checkout are required for formal aggregation; "
                          "--fixture-aggregate is only for labelled synthetic fixtures"}
    fingerprint = _fingerprint()
    checkout_meta: dict | None = None
    labels: list[str] = []
    if checkout is not None:
        try:
            _bundle, _module, checkout_meta = SB.load_ast_kernel(Path(checkout))
        except Exception as exc:
            checkout_meta = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        labels.append("fixture mode: checkout identity unchecked")
    payloads: dict[int, dict] = {}
    missing: list[int] = []
    invalid: list[dict] = []
    unit_status: list[dict] = []
    for year in expected_origins:
        origin_dir = out_root / f"origin_{year}"
        if not (origin_dir / "origin.json").is_file() \
                or not (origin_dir / "manifest.json").is_file():
            missing.append(year)
            unit_status.append({"origin": year, "status": "MISSING"})
            continue
        admission = admit_origin(origin_dir, year=year, fingerprint=fingerprint,
                                 mode="formal" if formal else "synthetic_fixture",
                                 raw=raw, checkout=checkout, checkout_meta=checkout_meta)
        if not admission["admissible"]:
            invalid.append({"origin": year, "reasons": admission["problems"][:8]})
            unit_status.append({"origin": year, "status": "INVALID",
                                "n_problems": len(admission["problems"])})
            continue
        payloads[year] = admission["payload"]
        unit_status.append({"origin": year, "status": "ADMITTED"})
    if missing or invalid:
        refused = {"status": "INCOMPLETE", "mode": mode, "missing_origins": missing,
                   "invalid_origins": invalid, "unit_status": unit_status,
                   "fingerprint": fingerprint, "labels": labels,
                   "note": "full-cohort macro/intervals not computed on a subset; "
                           "missing/invalid units are never dropped"}
        C.write_json(out_root / "aggregate_admission.json", refused)
        return refused
    # exact grid-key check per origin (duplicates cannot hide missing cells;
    # `_validate_endpoints` already covers this, the check stays as a second net)
    expected_cells = {(policy, eta, kappa)
                      for policy in PL.POLICIES
                      for eta in C.ETA_GRID for kappa in C.COST_RATIO_GRID}
    for year, payload in payloads.items():
        cells = {(g["policy"], g["eta"], g["kappa"]) for g in payload["bridge_grid"]}
        if cells != expected_cells or len(payload["bridge_grid"]) != len(expected_cells):
            refused = {"status": "INCOMPLETE", "mode": mode,
                       "missing_origins": missing,
                       "invalid_origins": [{"origin": year,
                                            "reasons": [f"grid keys incomplete: {len(cells)} "
                                                        f"!= {len(expected_cells)}"]}],
                       "unit_status": unit_status, "fingerprint": fingerprint, "labels": labels,
                       "note": "full-cohort macro/intervals not computed on a subset"}
            C.write_json(out_root / "aggregate_admission.json", refused)
            return refused
    rows: list[dict] = []
    for year, payload in payloads.items():
        for policy, metrics in payload["allocator_metrics"].items():
            rows.append({"origin": year, "policy": policy, **metrics})
    frame = pd.DataFrame(rows)
    macro = []
    for policy, group in frame.groupby("policy", sort=True):
        entry = {"policy": policy, "n_origins": int(len(group))}
        for key in ("service_worst7", "service_worst3", "mean_service", "action_total_all_declared",
                    "action_rate_all_declared", "action_total_observed", "rescue_precision",
                    "L_lambda_1", "blocked_trigger_days", "n_missing_outcomes"):
            if key in group:
                entry[f"macro_{key}"] = float(group[key].mean())
        macro.append(entry)
    contrasts = []
    # (label, left, right) with left-minus-right semantics
    contrast_pairs = [(f"{policy}_minus_none", policy, "none") for policy in PL.POLICIES if policy != "none"]
    contrast_pairs += [(f"P_minus_{policy}", "P", policy) for policy in PL.POLICIES if policy not in ("none", "P")]
    for label, left_policy, right_policy in contrast_pairs:
        sub_l = frame[frame["policy"] == left_policy].set_index("origin")
        sub_r = frame[frame["policy"] == right_policy].set_index("origin")
        common = sorted(set(sub_l.index) & set(sub_r.index))
        if len(common) != len(expected_origins):
            continue
        for endpoint in ("service_worst3", "service_worst7", "service_worst14",
                         "mean_service", "L_lambda_1"):
            diffs = (sub_l.loc[common, endpoint] - sub_r.loc[common, endpoint]).to_numpy(float)
            stat = C.paired_origin_bootstrap(diffs)
            contrasts.append({"contrast": label, "endpoint": endpoint,
                              **stat, "delta_material": C.DELTA_MATERIAL,
                              "loo": C.leave_one_out(diffs, [str(c) for c in common])})
    grid_cells = sum(len(p["bridge_grid"]) for p in payloads.values())
    expected_grid_cells = len(expected_origins) * len(PL.POLICIES) * len(C.ETA_GRID) * len(C.COST_RATIO_GRID)
    physical_rows: list[dict] = []
    for year, payload in payloads.items():
        for cell in payload["bridge_grid"]:
            physical_rows.append({"origin": year, **cell})
    physical_frame = pd.DataFrame(physical_rows)
    physical_contrasts = []
    phys_pairs = [(f"{policy}_minus_none", policy, "none") for policy in PL.POLICIES if policy != "none"]
    phys_pairs += [(f"P_minus_{policy}", "P", policy)
                   for policy in PL.POLICIES if policy not in ("none", "P")]
    for (eta, kappa) in [(eta, kappa) for eta in C.ETA_GRID for kappa in C.COST_RATIO_GRID]:
        for label, left_policy, right_policy in phys_pairs:
            sub_l = physical_frame[(physical_frame["policy"] == left_policy)
                                   & (physical_frame["eta"] == eta)
                                   & (physical_frame["kappa"] == kappa)].set_index("origin")
            sub_r = physical_frame[(physical_frame["policy"] == right_policy)
                                   & (physical_frame["eta"] == eta)
                                   & (physical_frame["kappa"] == kappa)].set_index("origin")
            common = sorted(set(sub_l.index) & set(sub_r.index))
            if len(common) != len(expected_origins):
                continue
            for endpoint in ("worst7_deficit_MWh_complete_days", "worst7_adequacy_fraction_complete_days",
                             "uncovered_MWh_finite_hours", "extra_committed_MWh",
                             "full_cost_MWh_equivalent"):
                diffs = (sub_l.loc[common, endpoint] - sub_r.loc[common, endpoint]).to_numpy(float)
                stat = C.paired_origin_bootstrap(diffs)
                physical_contrasts.append({"contrast": label, "endpoint": endpoint,
                                           "eta": float(eta), "kappa": float(kappa), **stat,
                                           "loo": C.leave_one_out(diffs, [str(c) for c in common])})
    physical_macro = []
    for (policy, eta, kappa), group in physical_frame.groupby(["policy", "eta", "kappa"], sort=True):
        physical_macro.append({
            "policy": policy, "eta": float(eta), "kappa": float(kappa),
            "n_origins": int(len(group)),
            "macro_worst7_deficit_MWh": float(group["worst7_deficit_MWh_complete_days"].mean()),
            "macro_worst7_adequacy_fraction": float(group["worst7_adequacy_fraction_complete_days"].mean()),
            "macro_uncovered_MWh_finite_hours": float(group["uncovered_MWh_finite_hours"].mean()),
            "macro_extra_committed_MWh": float(group["extra_committed_MWh"].mean()),
            "macro_full_cost_MWh_equivalent": float(group["full_cost_MWh_equivalent"].mean()),
        })
    summary = {"schema": "card10.segan_r2.aggregate.v4", "status": "COMPLETE",
               "mode": mode, "fingerprint": fingerprint,
               "raw_sha256": (C.sha256_file(Path(raw)) if raw is not None else None),
               "checkout": (str(checkout) if checkout is not None else None),
               "labels": labels, "unit_status": unit_status,
               "n_origins": len(payloads),
               "macro": macro, "contrasts": contrasts,
               "physical_macro": physical_macro,
               "physical_contrasts": physical_contrasts,
               "grid_cells": int(grid_cells), "expected_grid_cells": int(expected_grid_cells),
               "grid_complete": bool(grid_cells == expected_grid_cells),
               "missing_origins": missing, "invalid_origins": invalid}
    C.write_json(out_root / "aggregate_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--checkout", type=Path,
                        help="pinned SAOCP checkout containing online_conformal/saocp.py")
    parser.add_argument("--fixture-raw", action="store_true",
                        help="synthetic fixture mode (skips the formal raw-SHA gate)")
    parser.add_argument("--fixture-aggregate", action="store_true",
                        help="labelled synthetic fixture aggregation; bypasses ONLY the "
                             "formal raw-SHA / checkout gates")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--origin", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        summary = aggregate(args.out, raw=args.raw, checkout=args.checkout,
                            fixture=args.fixture_aggregate)
        bulky = ("macro", "contrasts", "physical_macro", "physical_contrasts")
        print(json.dumps({k: v for k, v in summary.items() if k not in bulky},
                         indent=1, default=str))
        if summary.get("status") == "INCOMPLETE":
            print(json.dumps(summary.get("invalid_origins", [])[:3], indent=1, default=str))
        return 0 if summary.get("status") == "COMPLETE" else 2
    if not args.raw or not args.checkout:
        print("--raw and --checkout are required for runs")
        return 2
    if not args.fixture_raw:
        sha = C.sha256_file(args.raw)
        if sha != C.RAW_NYISO_SHA:
            print(json.dumps({"status": "FAIL", "reason": f"raw hash mismatch {sha}"}))
            return 2
    years = list(C.TEST_YEARS) if args.all else ([args.origin] if args.origin else [])
    results = [run_origin(year, args.raw, args.out, args.checkout, fixture_raw=args.fixture_raw)
               for year in years]
    print(json.dumps(results, indent=1, default=str))
    return 0 if all(r["status"] in ("PASS", "SKIP_VERIFIED") for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
