# R-A protocol amendment B — implementation corrections before any outcome run

- Applies to: `R1_PROTOCOL_FROZEN_2026-10-11.md` (bytes unchanged, sha256 `3ebfa0958022908ff86d4197a37560cb5b0fc77b5b56ad5bee5bffa29a7f87a1`)
- Amendment date: 2026-10-11 · author: Claude Code (modifier role) · reviewer authorization: R02B (2026-10-11)
- Discipline: the frozen protocol document is NOT overwritten. This amendment supersedes only its §4 code-hash table and records the corrections below.

## 1. State at amendment time (what exists and what does not)

Produced so far, under the frozen protocol and this amendment: guard suite (`outputs/tests/r1_tests.json`, all 9 tests PASS) and Stage 0 (`outputs/stage0/r1_stage0_verification.json`, status PASS, reverse-replay mismatches 0/30 origins, cap mismatches 0). **No allocation, oracle, budget-sensitivity, direction-audit or aggregate outcome exists yet.** No test-selected parameter was changed; no protocol endpoint, cohort, policy definition or threshold was altered.

## 2. Why this amendment exists

(a) Reviewer static review `R02_PRELIMINARY_CODE_REVIEW.md` items 1–10 on in-progress bytes, and (b) three pre-run guard failures found while executing the guard suite, all before any outcome computation:

| # | Failure (pre-run) | Evidence | Correction |
|---|---|---|---|
| F1 | Oracle Stage-2 sign error (reviewer item 1): appended `-z >= -z1-eps` enforced `z <= z1+eps` | reviewer static review; corrected bytes | preserved-z row now carries coefficient `+1` on z with lower bound `z1 - 1e-9`; Stage 3 inherits the fixed row; every returned action vector re-validated (`validate_actions`) incl. all window inequalities, count/cap, rescue ≥ Stage-2 optimum |
| F2 | `EVIDENCE_ROOT` resolved to the portal, not `D:\mypaper` (item 3) | guard-suite FileNotFoundError, first run | `parents[4]` + explicit `--root`; no "similar name" fallback anywhere |
| F3 | `T2` boolean-mask length bug and a too-weak perturbation design (item 8) | guard-suite IndexError, then failure review | T2 rewritten: perturb q at dates ≥ t−1 (incl. t), recompute the F5 shadow under the perturbed path, compare prefixes through t; 3 origins × 3 pivots + synthetic binding-budget and gap cases; older previous-segment outcomes are not perturbed (they legitimately affect earlier spending and the global cap) |
| F4 | Transfer identity: R51 `drb_action` is the thresholded-count K2 rule, not the continuous production path (item 2) | guard-suite T6 failure on 23/23 transfer origins; then verified p0_2/K2 identity | loaders now take production from the authoritative MR-P0-2 daily rows, reverse-replay K2 separately, and assert R51-label ≡ p0_2-K2 equality; the Stage-0 mismatch is exposed, not weakened |
| F5 | R39 `base_no_fallback` layer stores `action_cap = 0`; the loader took the cap from that layer, so Stage 0's cap check silently skipped NYISO (its `> 0` escape) and the smoke runner aborted on `cap mismatch 55 != 0` | smoke run `--origins 2018->2019`, aborted before any output was written | loader takes `action_cap` from the production layer; Stage 0 checks every origin's cap strictly (no escape hatch) and records the distinct stored cap values; re-run: caps 55/55/53/55/55/55/55 (NYISO) and per-origin transfer caps all match `round(0.15·n_days)` |
| F6 | Allocation runner wrote origin labels containing `->` into Windows filenames (`OSError 22`), no output produced | smoke run 1 | output stems sanitized (`_safe_name`); the same smoke then completed |
| F7 | Oracle aggregate messaging conflated "incomplete smoke" with "failed vs R43 macro" | smoke run 2 (1/7 units) | aggregate now compares the R43 macro values only when all 7 units are valid and none failed; an incomplete aggregate is reported as `PARTIAL_INCOMPLETE` with the deferral reason, per-unit FAIL objects are retained, and the exit code is nonzero for partial as well as failed runs |
| F8 | Aggregate flag rows crashed on NaN from heterogeneous policy fields (`ValueError: cannot convert float NaN to integer`) | first full aggregate run | flag aggregation filters None/NaN, distinguishes boolean fields (all/list) from counts (sum); outcome files were already complete and are unchanged — reporting-stage fix only |

Also corrected from the same review: item 4 (explicit date normalization plus `assert_same_support` exact support equality on every pipeline join; cross-run diagnostics report missing/extra sets instead of inner-joining), item 5 (R39 is the canonical F5 tension source; the earlier r02 stream is a *distinct* run — 0 missing/extra days but max |Δq| 0.958 and max |Δthreshold| 372.6 on the common support — used for identity diagnosis only and never combined), item 6 (all runner fingerprints bind code hashes **and** all bound input hashes; allocation resume verifies result **and** daily-trace SHA-256; oracle resume verifies the stored output SHA), item 7 (process CPU affinity pinned to two logical CPUs via psutil — applied and recorded — plus OMP/MKL/OPENBLAS/NUMEXPR/VECLIB thread variables set to 2 before the BLAS import; tie allocations are documented as non-unique and only objective values are claimed/verified), item 9 (frozen fine grid {0, 0.02, …, 0.20} implemented per policy plus pairwise crossing λ* with explicit equal-action-rate handling), item 10 (unambiguous `action_total_sum` vs `mean_actions_per_origin` vs `action_rate_macro` fields; NYISO DRB reference total 280 noted).

## 3. Additional evidence recorded in this amendment

- R51 labelling: for all 23 transfer origins, `R51.drb_action ≡ p0_2.r51_thresholded_count_k2` (exact), while the continuous production path ≡ `p0_2.production_shortfall_sum_lag2` (exact). Naming differs from identity; identity wins.
- Guard suite: T1 hand-computed F5 state machine (incl. the open-bank-at-trace-end flag path), T2 perturbation invariance, T3 segment relabel invariance, T4 cap/F1/F2 accounting, T5 determinism, T6 reverse replay (30/30), T7 exhaustive oracle toy where max-min W7 and max-rescue compete (Stage 1 and Stage 2 both match enumeration; rounded actions re-validate), T8 fine-grid crossings incl. equal-rate and negative-crossing cases, T9 support-set missing/extra/duplicate detection — 9/9 PASS.
- Stage 0: PASS; per-origin caps match the stored caps on all 30 origins; reverse replay 0 mismatches (production on NYISO vs R39 stored; production and K2 on transfer vs p0_2 rows).

## 4. Superseding code-hash table (replaces protocol §4)

| File | SHA-256 | Δ vs protocol §4 |
|---|---|---|
| `scripts/r1_common.py` | `ba07e287492f3cc9b08b28ea689a9524495e9b2f70012bba40bf7420d8f0aa0a` | changed (fingerprint/support/crossing helpers, loaders, production-layer cap) |
| `scripts/r1_policies.py` | `a9643033b158da9859fb656b5c9f3b69fb6a073ab0d34fbc207a5222b7f296bc` | unchanged |
| `scripts/r1_stage0.py` | `c393eaf32db3355ff18c76046fd1f32c7d73d0dd9f050d89db9718b7383a1104` | changed (support-aware cross-checks, K2 reverse check, strict cap check) |
| `scripts/r1_allocation.py` | `366f4f94f9979ee17acacbacb218c9c1b05b0c96a4a4da475c9578655ccbdf52` | changed (fingerprint, trace-SHA resume, explicit F5 shadow recomputation, safe output stems) |
| `scripts/r1_budget_oracle.py` | `b4ec960e1b89a621b0cb1fb3d9e4dbb612aa6344f835c06766cc52ec16ccf5dc` | changed (sign fix, validation, per-origin fail-closed, affinity, aggregate verification over all valid units, accurate partial status) |
| `scripts/r1_direction_audit.py` | `78e3b50ea02c9c38bbb84b49c21d5d7c4ca513dfd16203262684cdf83ad9eaaf` | unchanged |
| `scripts/r1_aggregate.py` | `03f199bcc961dd6c91e7ccc4109a45ca73f722f608c79947c69f3f614554c0de` | changed (fine grid, crossings, unambiguous action fields, NaN-safe flag aggregation) |
| `scripts/r1_tests.py` | `861992dc0a9f4f177359558c0f43090d96c297acd022453c5a329b51b91f0cc6` | changed (T2 rewrite, T7/T8/T9 added) |

## 5. What runs next under this amendment

The admitted **exposed** suite only: allocation-value replays (NYISO seven origins + transfer 23 origins), budget/oracle objects, direction audit, aggregation. Oracle admission is fail-closed per origin (FAIL object retained with solver diagnostics, nonzero exit on any failure or failed aggregate verification against the admitted R43 objective values over **all valid units**). Any further code change invalidates this amendment and requires a new dated one before a fresh run; already-produced outcomes are then re-run and marked superseded.
