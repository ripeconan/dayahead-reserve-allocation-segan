# R2 frozen executable protocol — cutoff-safe D-1 point/UQ/controller pipeline and finite zonal reserve bridge

- Identity: `card10-segan-R2` · frozen 2026-10-11 (R03D) · author: Claude Code (modifier role)
- Status: **no formal outcome run has been executed**. Native execution is root-owned and runs this protocol only after root's byte review.
- Exposed-status chronology (truthful): the public-kernel route FAILED on native Python 3.10 because the public dependency graph requires Merlion (root's smoke, 2026-10-11); the admitted **verified AST kernel** route was then adopted and root independently ran the calendar-batched adapter with it on synthetic fall-fold dates (seven checks pass). All local guard work is synthetic.

## 1. Scope and non-goals

Exposed 2017–2025 NYISO raw data only. No 2026 outcomes, no network, no reserved objects, no deep retraining, no GPU. The bridge is a **zonal balancing simulation without network flows** — not delivered-grid reliability, no revenue or outage claims. Full-fallback `s=1` remains the abstract model of the modeled-service side.

## 2. Bound identities

| Object | Identity |
|---|---|
| Raw NYISO zonal | `/home/icdm/card10_confirmatory/data/nyiso_zonal_2017_2025.csv.gz` · sha256 `d9e492532e22bc3f41c6967acf99cc5b49a49ae2e5ba3ae3b7dbc8092d9e5d0d` · 78,144 rows · local delivery span 2017-01-01..2025-12-31 (whole-date gaps in 2017 and 2021 are REAL support, never bridged) |
| Pinned SAOCP checkout | commit `95790a9b9e4061166e6d7a8a13282e019b1a8663`, tree `77348e5fe7f2bb3ed5a031aaad9874b6a5ab2d7b`, worktree clean; `online_conformal/saocp.py` sha256 `83f8845559786452aafabfafdf6c136ceaf9b9b3557fc5f8f9267fcec04d386f` |
| Numerical backend | **verified AST kernel** `verified_wrapper_ast_kernel`; vendored unchanged copy `scripts/vendor/saocp_ast_kernel.py` sha256 `3cede45f95ecad16ff9f77e8c830b5ac1d66fbc0b148bdce77f08d8409e47eef` (source: `scripts/card10_baselines/saocp_ast_kernel.py`); its loader refuses dirty/wrong identities before compile/exec. Calendar batching is a NEW adapter; **no public-runtime-parity claim** |
| Code identity | `r2_run._fingerprint()` = sha256 over `{relative file: sha256}` of `r2_env, r2_common, r2_pointcorr, r2_saocp_batch, r2_allocators, r2_bridge, r2_pipeline, r2_run` + `vendor/saocp_ast_kernel.py`, plus extra identities (protocol sha, AST wrapper sha, raw sha); every manifest carries it |

## 3. Calendar, point correction, geometry (exact choices)

- Seven origins: reference year Y−1 (2018–2024), evaluation calendar **Jan 2 – Dec 31** of Y (2019–2025); T_planned = 364/365; **cap B = round(0.15·T_planned) = 55** per origin, fixed before any support is read.
- Grid: hourly UTC grid from the earliest admitted raw row through Dec 31 Y 23:59 local; missing cells stay NaN (whole-date gaps are real support). Duplicate raw timestamps are rejected in production. Out-of-range rows are recorded separately (`out_of_range_rows`).
- Point correction (one variant): realized-error lags **72 and 168 elapsed hours**; features = local-clock Fourier + official forecast + lagged errors; **finite** feature/label masks (inf/−inf = input absence, never numerically processed); rolling **365-day** training window ending strictly before local midnight of t−1 (latest completed label t−2; boundary for Jan 2 = Jan 1 00:00, so Dec 31 labels are eligible); ridge **penalty 10**, intercept unpenalized, z-score standardization on training columns; minimum 1000 eligible training rows; issuance requires finite pre-action inputs ONLY (never the current/future actual); per-row lag provenance (exact τ−72h/τ−168h instants AND local calendar age ≥ 2/≥ 6 days) is checked for every issued row and recorded.
- Reference window: the **exact prior calendar year** (never narrowed to a fixture). Geometry: official **sklearn.covariance.OAS** (`assume_centered=False`; version recorded) on reference-year corrected residuals; center = reference mean; radius per hour/zone = `sqrt(max(R·Σ_zz, 0))` with the producer's radicand clipping; upper residual support = `mu_z + radius`; reserve requirement `r0_z = max(0, upper support)`. Newest reference outcome ≤ first test decision − 2 local days (enforced).
- Reference artifacts persisted with timestamps: `reference_scores.csv.gz` (utc, local_date, slot, score), `reference_residuals.csv.gz` (+ per-zone issued point values) so covariance/H are reconstructible; per-slot prefix values sha256 + first/last UTC.

## 4. Calendar-batched SAOCP semantics

One kernel per local hour slot; scale = max|finite reference prefix|·√3, warm-started with that same finite prefix (empty/nonfinite prefix refused unless an explicitly labelled test passes a neutral scale); a score's reveal date = source local date + 2 calendar days and it is released at the first decision ≥ that date, chronological UTC order (both fall folds in order); thresholds are frozen once per (decision date, slot) BEFORE that day's scores exist; both folds share the slot threshold; absent hours produce no score; partial days are never enqueued. Resets come from ONE shared date-causal event list: an issuance absence known at the cutoff resets immediately at that decision; a missing outcome is discovered at its reveal date (u+2). Every queue entry carries its source segment; resets discard old-segment pending records. The UQ adapter's segment numbering is aligned with the allocator walks (first segment 0).

## 5. Allocators and the finite-reserve bridge

- Policies (single source of truth, 7): `none`, `P` (production shortfall-sum θ=1.0, memory 6), `F1` pace7, `F2` paceBank, `F3` K2, `F4` persist1, `F5` declared-Dec-31 bank (shadow = P's action path; tension/history updated on every issuable date, cleared at the shared resets, independent of current/future actuals). Each policy walks a method-owned history; the `none` walk has exactly zero action/blocked/cap_skipped/used accounting with the same event-derived segments.
- Modeled endpoints: base q and served s are separate; the modeled-support mask is `scorable & observed & finite(q_base)` (no fabricated misses, no zero-filling); windows use consecutive local dates within one decision-time segment; W3/W7/W14, mean service/base coverage, rescue (from base q), `L_λ` on {0.25,0.5,1,2,4}; all-declared and modeled-support action counts/rates separately.
- Bridge: `residual_MW = actual_load_MW − issued_point_MW` in MW; committed action days add `η·H_z` per zone to `r0_z` for the whole committed day; `H_z` = reference-only 95th percentile (linear) of `max(reference corrected residual_z, 0)` over the exact reference year; η grid {0, 0.5, 1, 2}; cost-ratio grid κ {0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5}; each real UTC hour contributes 1 h (both folds); deficits/adequacy are scored on finite hours only; bookings are retained on every committed day regardless of outcome completeness; `extra_committed_MWh = η·Σ_z H_z · (committed expected hours)`; `full_cost = κ·(base_reserved + extra) + uncovered_finite` and `incremental_cost = κ·extra + uncovered_finite` (base cancels); worst-window deficit = **maximum** window deficit over complete observed days in consecutive dates and the same segment; per-day physical frames are persisted (`bridge_daily.csv.gz`); finite-hour support is verified identical across policies on the same base.
- Every declared cell is reported: 7 policies × 4 η × 9 κ per origin, no selection.

## 6. Inference (descriptive)

Paired per-origin contrasts with IID-over-origin bootstrap (10,000 draws, seed 20261011) and leave-one-origin-out means: modeled `P − F_k`, `F_k − none` on W3/W7/W14/mean/L1; physical `policy − none` on worst-7 deficit, worst-7 adequacy, uncovered finite MWh, extra committed MWh, full cost at every (η, κ) cell. Descriptive exposed-data language only; no equivalence/independence/confirmation claims.

## 7. Resume, fail-closed rules, artifacts

Per origin: `origin.json`, `hourly.csv.gz`, daily, `walk_{none,P,F1..F5}.csv.gz`, adapter ledger/thresholds, `point_audit.csv.gz`, `bridge_grid.csv.gz`, `bridge_daily.csv.gz`, `reference_scores.csv.gz`, `reference_residuals.csv.gz`, `kernel_meta.json`, `manifest.json` (+ `.sha256` sidecar). Completion requires: all required artifacts present, manifest canonical hash valid, sidecar verified, endpoint validation (finite modeled endpoints; exact 7×4×9 physical keys, no duplicates) passed. Resume SKIP requires: sidecar match, manifest verification, fingerprint + protocol + raw + AST wrapper + checkout + thread-environment identities all matching. Any rejection preserves the old bytes under `origin_Y/attempts/<attempt>/` with a recorded reason; FAIL objects carry stage + traceback; failed attempts are never deleted.

## 8. Exact native commands (root-owned; do NOT run from the Claude session)

```bash
cd <analysis_r2>/scripts
taskset -c 0-7,9-31 nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /usr/bin/python3 r2_run.py \
    --raw /home/icdm/card10_confirmatory/data/nyiso_zonal_2017_2025.csv.gz \
    --checkout /home/icdm/card10_external/online_conformal \
    --out <analysis_r2>/outputs/formal_r2 --all
/usr/bin/python3 r2_run.py --out <analysis_r2>/outputs/formal_r2 --aggregate
```
(Fail-closed without the exact raw SHA, the clean pinned checkout, and this protocol file + sidecar; per-origin checkpoints; rerun is SKIP-verified.)

## 9. Amendment discipline

Any code change after this freeze invalidates the fingerprint and requires a dated amendment + preserved earlier bytes before a fresh run; previously produced exposed outputs are retained and marked superseded, never silently overwritten.
