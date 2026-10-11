# R2 protocol amendment B — pre-outcome repairs for aggregate admission, prefix guards, portable guards

- Applies to: `R2_PROTOCOL_FROZEN_2026-10-11.md`, whose bytes are **unchanged** at sha256
  `6a4238286ea66b1c4c0de05feb3239931658c42c9db23f24b284116481cdd982`
  (that original hash is restated here for continuity; this document does not alter it).
- Supersedes the code-level statements of `R2_PROTOCOL_AMENDMENT_2026-10-11A.md` only where listed below;
  amendment A remains a binding record of its own repairs.
- Date: 2026-10-11 · author: Claude Code (modifier role) · trigger: root mailbox items 10–13 plus the
  R03E handoff (four remaining implementation items).
- **Status: no formal outcome has been computed or admitted.** Every value in this amendment is a
  code/schema statement or a synthetic local check. Native execution remains root-owned.

## 1. Identity binding (items 2/11)

- `r2_run._fingerprint()` now hashes the code files **plus the frozen protocol document and every
  dated amendment** (`R2_PROTOCOL_AMENDMENT_*.md`) with their `.sha256` sidecar state.
- `admit_origin(...)` is the single admission validator shared by `run_origin` resume and
  `aggregate`. Three explicit modes are defined, and the mode is recorded in every result:
  - `formal` — raw SHA must equal the admitted raw pin; live checkout must be the clean pinned
    commit/tree; document sidecars must verify.
  - `fixture_raw` — real pipeline unit computed on a synthetic raw file (guard P7): only the raw
    SHA **pin** is relaxed to "stored hash == supplied file"; checkout and sidecars stay strict.
  - `synthetic_fixture` — tiny synthetic **metadata** unit (guard P14): raw/checkout/kernel-git
    gates are explicitly labelled as unchecked, and the unit must self-declare `fixture = true`
    in both `origin.json.environment` and `manifest.json.extra.environment`.
- Required in every mode: manifest sidecar verification, full artifact-manifest verification, the
  current code fingerprint, the stored vs current protocol/amendment identities, the full numeric
  environment identity (Python/numpy/pandas/sklearn/scipy + thread env), the stored kernel
  `wrapper_sha256`, `payload.test_year == payload.origin == <directory year>`, canonical payload
  consistency and `_validate_endpoints` (finite modeled endpoints; exact 7×4×9 physical key set,
  no duplicates, finite physical fields).
- Malformed per-origin JSON is recorded as an invalid unit **with reasons** (never an uncaught
  exception). Any missing or invalid unit refuses the whole-cohort comparison and writes
  `aggregate_admission.json`; units are never silently dropped.
- The fingerprint is computed **after** the documents exist (no static self-hash is written into
  any document); amendment bytes describe the schema, the run-time fingerprint binds them.

## 2. Prefix guards and numeric tension evidence (item 1)

- `_series_prefix_equal` (guards): both sides are normalized through `pd.to_datetime(..., utc=True)`,
  filtered to local delivery dates `<= pivot` **first**, and the prefix key sets must then be
  exactly equal and nonempty before values are compared. Future issuance support may legitimately
  differ (a future day loses its inputs when a lagged error disappears); the previous whole-year
  key-set equality was wrong and is removed.
- P3 compares **thresholds** and issued points and every allocator action prefix under BOTH the
  value perturbation and the missingness perturbation, and additionally exercises the comparator
  itself (future-support-differs → equal; prefix gap / value change / empty prefix / NaN-pattern
  mismatch → rejected).
- `daily.csv.gz` gains a `tension` column: the pre-action tension of the day (computed from the
  thresholds known at that decision, before that day's actions). P13 compares the persisted
  **numeric** tension series through the pivot, requires substantial numeric support, and still
  compares thresholds and F5 actions.

## 3. Portable, persistent guard runner (item 3)

- `r2_guards.py` accepts `--checkout`, else `R2_CHECKOUT`, else a labelled local default; no
  hardcoded Windows path is required to execute.
- Every test's exception is captured into an `ERROR` record with traceback; results and source
  fingerprints are persisted after **every** test; BEGIN/END timestamps are printed with flush to
  stdout and to `<results>.json.log`.
- Exit code is nonzero unless the run is complete, all selected tests passed, and the source
  fingerprints are unchanged before/after. A `--only` subset is labelled (`full_suite=false`,
  unselected tests stay `NOT_RUN`) and can never be reported as the full suite.

## 4. Inference implementation (unchanged estimand)

- `paired_origin_bootstrap` produces its draws as one `(n_boot, n)` matrix instead of `n_boot`
  consecutive length-`n` calls. This is adopted **only** because guard P15 verifies bit-identical
  draws against the retained pre-R03E loop at the production settings (10,000 draws, seed
  20261011, nonconstant 7-origin fixture). No estimand, seed, size or endpoint changed; if P15
  ever fails, the loop form must be restored.

## 5. Exact native commands (root-owned; do NOT run from this session)

```bash
# full guard suite (all P1-P15)
cd <analysis_r2>/scripts
taskset -c 0-7,9-31 nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /usr/bin/python3 r2_guards.py \
    --checkout /home/icdm/card10_external/online_conformal \
    --out <analysis_r2>/outputs/guards_v4/r2_guards_v4_native.json --label native_full_v4

# formal origins (unchanged semantics; resume/admission now fully identity-bound)
cd <analysis_r2>/scripts
taskset -c 0-7,9-31 nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /usr/bin/python3 r2_run.py \
    --raw /home/icdm/card10_confirmatory/data/nyiso_zonal_2017_2025.csv.gz \
    --checkout /home/icdm/card10_external/online_conformal \
    --out <analysis_r2>/outputs/formal_r2 --all

# formal aggregation now REQUIRES --raw and --checkout plus a verified amendment sidecar
/usr/bin/python3 r2_run.py --out <analysis_r2>/outputs/formal_r2 --aggregate \
    --raw /home/icdm/card10_confirmatory/data/nyiso_zonal_2017_2025.csv.gz \
    --checkout /home/icdm/card10_external/online_conformal
```

## 6. Superseded bytes

Pre-amendment-B scripts, protocol and amendment-A bytes are preserved under
`analysis_r2/superseded_r03d_2026-10-11/` with `HASH_MANIFEST.txt`/`.sha256`; the interrupted
pre-patch guard log (`outputs/guards_v3/r2_guards_v3_run.log`, empty: the run never produced a
result file) is retained there and must not be read as a pass.
