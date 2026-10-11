# R-A frozen executable protocol — allocation-value and budget-binding audit

- Identity: `card10-segan-R1-A` · frozen 2026-10-11 · author: Claude Code (modifier role)
- Authorization: reviewer decision R02 (2026-10-11), which authorizes these exposed-data runs under fixed definitions. Internal reviewer authorization; the reviewer will inspect code, raw outputs and PDFs.
- Status at freeze: no outcome computation for this protocol has been executed (only read-only source/identity inspection).

## 1. Scope and non-goals

Exposed-data, deterministic, allocation-only replays on **already admitted** traces. No new data access, no network, no SSH, no reserved objects, no retraining, no hyperparameter selection on outcomes.

Explicitly NOT done: no adoption of any comparator as the production rule; no count-conditioned replay of any new rule (equal-cap is the primary and only new operational comparison; the existing random/COCD/oracle rows serve as clearly labeled context); no migration of K2 (or any variant) results onto the production controller; no confirmatory language; no score or superiority claim.

## 2. Cohorts and origins

| Cohort | Origins | Source |
|---|---|---|
| `nyiso_exposed` (primary) | 7: `2018->2019` … `2024->2025` | R39 daily + R39 hourly |
| `gefcom_replay` | 5: `2010Q4, 2011Q1..2011Q4` | R51 daily |
| `caiso_dev_replay` | 12: `2019Q1..2021Q4` | R51 daily |
| `caiso_postsplit_replay` | 6: `2022Q1..2023Q2` | R51 daily |

Cohorts are never pooled. Each origin keeps its stored day set, segment labels (relabeled chronologically per origin only) and per-origin cap.

## 3. Bound inputs (SHA-256, fail closed)

| Path (relative to read-only evidence root) | SHA-256 |
|---|---|
| `reports/experiment_outputs/card10_p0_4_nyiso_verified_saocp_r39_2026-07-16/daily_policy_trace.csv.gz` | `c12f16f655a11d91ba26b1b942018c592d275a3cb941a4a2f2e35aedc4ed3ca5` |
| `reports/experiment_outputs/card10_p0_4_nyiso_verified_saocp_r39_2026-07-16/selected_hourly_results.csv.gz` | `f66f795a0f9aa779262d46f9c54c90bcdf7e661a853eeb19383a4fc0f699e9b4` |
| `reports/experiment_outputs/card10_nyiso_saocp_drb_current_r02_2026-07-15/policy_daily_trace.csv.gz` | `4ee214335ae62351b9e9875a91afcc30d5f930df2013bbc10b949db44276f958` |
| `…/card10_nyiso_saocp_drb_current_r02_2026-07-15/saocp_hourly_base.csv.gz` | `ce2ce87e57fb7d6e05b319022b35c67c736706172ecb6658eecc2e483744135c` |
| `…/card10_nyiso_saocp_drb_current_r02_2026-07-15/random_draw_endpoints.csv.gz` | `530dbe426aafc95416f965c7e45647accf68e1acdeca74ae21b26acd3599b723` |
| `…/card10_p0_3_p0_4_cross_dataset_transfer_r51_2026-07-17/daily_trace.csv.gz` | `3e84b66dd2fff4a57151d5f67b21516043642c3acb3221bdda175bcfe4f88e42` |
| `…/card10_mr_p0_2_production_transfer_2026-07-22/daily_trace.csv.gz` | `d50a106b2511da0747b5b3e7e96fd52761df7a91c6bc0124e7092c6251dc2b0a` |
| `…/card10_mr_p0_2_production_transfer_2026-07-22/aggregate_results.csv.gz` | `8498143f30f81a8fc814eebaf5570f3153abdfb2ccf078b02f1f4393eeaece9e` |
| `…/card10_p0_3_cocd_r45_2026-07-16/cocd_daily_ledger.csv.gz` | `edc3b6714c2961f71f15affd6efaa6e327c4037aa9c861b5c0c0aafcc8f86566` |
| `…/card10_p0_3_cocd_r45_2026-07-16/aggregate_results.csv.gz` | `40a8a6c30f8abb7b9303232703c93d110da6e43014fd21f0e66b07a1becd957d` |
| `…/card10_p0_3_exact_oracle_exchange_r43_2026-07-16/aggregate_results.csv.gz` | `36e04946bb4f2714351a32cf87dd4e893b8630e351939ea8b8eed09f47d92f89` |
| `…/card10_p0_3_exact_oracle_exchange_r43_2026-07-16/oracle_exchange_daily_trace.csv.gz` | `becc34f45ca32ecbfc5ebfd3f9a299162271376a2dadb0b3180464b82bc1ca61` |
| `…/card10_p0_3_exact_oracle_exchange_r43_2026-07-16/solver_diagnostics.csv.gz` | `9d92861ef8477941e6588fac50f1843542e375cb82811a2127e5b555f096b772` |

Reference source files used only to fix semantics (never imported at run time except where stated): `scripts/card10_killtest/nyiso_confirmatory_audit.py` `b9cfde3b27219a5c2fce07652d31ade0dc2ed7f4fef9eb9894e992fea0b2cddd` (residual/units semantics), `scripts/card10_confirmatory/confirm_runner.py` `13e9371c6c5f1091e0e62375d5d4e7119d8d24e9e35fb40f1fafa4d8079106a7` (A\* state machine), `scripts/card10_method/run_production_transfer_mrp02.py` `9dbb6da1c4badbebd5bc66c59c7c7a8dd766fb5980ea959a12596c0b67da3fef` (production + K2), `scripts/card10_baselines/run_nyiso_scientific_matrix.py` `fa131d1804f7ff34645e015a22ad483002f93f2d92e0069d566f5059694e4405` (directional columns), `scripts/card10_p03/run_exact_oracle_exchange_r43.py` `4ba9e02406bf82d2ba84ae0d80157cd16e3dacfd2f582be962d1f1486ddaaffb` (oracle contract), `scripts/card10_p03/cmsd_controller.py` `0cd764e0…` (production controller).

## 4. Code identity (SHA-256)

| File | SHA-256 |
|---|---|
| `scripts/r1_common.py` | `3cee3072ddb46024c7435ac6a0dd98eb6428651aef5839a1ddcb96c6b3d29b02` |
| `scripts/r1_policies.py` | `a9643033b158da9859fb656b5c9f3b69fb6a073ab0d34fbc207a5222b7f296bc` |
| `scripts/r1_stage0.py` | `a0cda9f48da6b6866fd9283c2e8f56273e9747adba5c5291bc071596dfefe6a3` |
| `scripts/r1_allocation.py` | `441c29e03684e2a8e61b338e9a21b8eaed1a42fbbaf65750b2930cd3a2f79ff1` |
| `scripts/r1_budget_oracle.py` | `20277812c85ecd7116c9a8fccf20049a08c43c89b22799073a4f69983739e3c1` |
| `scripts/r1_direction_audit.py` | `78e3b50ea02c9c38bbb84b49c21d5d7c4ca513dfd16203262684cdf83ad9eaaf` |
| `scripts/r1_aggregate.py` | `ef4b3474ca99513ae8889bbe881ed9ad294cc3995b615feb5f2d6fe7c2e52694` |
| `scripts/r1_tests.py` | `4f98166316eb1bfd96b348322c4786c91ea670665d0259fd26b43159b2269106` |

## 5. Frozen policy definitions (equal cap B = round(0.15 · n_days) per origin)

Common eligibility: record u is eligible for decision day t iff same segment and `date(u) + 2 calendar days ≤ date(t)`; at most the most recent 6 eligible records are used. All rules see post-action service in the closed loop. Day indices are the stored chronological evaluation days of the origin (missing days are absent, not imputed).

- **P (production DRB)** — state `h_t = Σ_{recent 6 eligible}(1 − s_u)`; action iff `h_t ≥ 1.0` and `used < B`. Must reproduce the stored action vector exactly (Stage 0 gate).
- **F1 `pace7`** — per segment, segment-local indices 6, 13, 20, … are scheduled action days, fixed in advance; a scheduled day is skipped (no action, schedule unchanged) iff `used = B`; segments with fewer than 7 days receive no scheduled action; the schedule restarts at every segment. Conditional guarantee (only when no scheduled day is cap-skipped and the segment has ≥ 7 days): every within-segment 7-window contains ≥ 1 action, hence `W7(paced) ≥ W7(base)` for that origin; the flag `all_windows_have_action` and `cap_skipped_scheduled_days` are reported per origin, and no floor is claimed otherwise.
- **F2 `paceBank`** — exact integer arithmetic: bank += 15 per evaluated decision day; action iff bank ≥ 100 and `used < B`, then bank −= 100; a new segment resets the bank to 0 (residual credit dropped, counted and reported); `used` never resets. No outcome information.
- **F3 `k2_count`** — R51 thresholded-count variant: count = #{recent 6 eligible with `s_u < 0.9`}; action iff count ≥ 2 and `used < B`.
- **F4 `persist1`** — action iff the most recent eligible record has `s_u ≤ 0.5` and `used < B` (no trigger when no eligible record exists).
- **F5 `bank1_causal`** — one-credit bank, transcription of the admitted recovery A\* state machine (`confirm_runner.cocd_known_horizon_cap_registered`): shadow mints `m_t = 1{h_t ≥ 1.0}` from the **production** (shadow) closed loop; `tail_rank = (1 + #{history ≥ tension_t}) / (1 + #history)`; `flush = remaining declared-calendar-days ≤ B + 1`; transitions exactly (bank=1 → action, bank←mint), (bank=0, mint=0 → none), (bank=0, mint=1, rank ≤ 0.5 or flush → action), else defer (bank←1); tension = per-day mean −log(p) of the frozen pre-action thresholds (hourly p per hour for NYISO, daily for transfer cohorts), history reset at segment changes and updated after the decision; a segment gap resets the bank and may drop one credit (counted). Declared horizon end = calendar end of the origin label (NYISO delivery year 12-31; transfer quarter end), which is data-independent; **no future missingness or segment end is queried**. Recorded deviation (never silent): if the bank is open at the final evaluated day because the trace ends before the declared end, the credit is dropped and `bank_open_at_trace_end = true` is recorded; no synthetic action is inserted. The original module is not imported at run time (project-wide import side effects); equivalence evidence = line-level transcription + hand-computed state-machine tests (T1).
- **Context rows (not new comparisons)**: F6 random (R02 existing draws, outcome-independent, cap-matched), F7 COCD (R45 ledger; **budget/end-conditioned, not deployable**), F8 oracle (below; **noncausal**).

## 6. Endpoints, materiality, inference

- Primary endpoint: origin-macro (equal-weight origins) within-segment **worst-seven-day modeled service** `W7`; secondary: `W3`, `W14`, mean service, action count/rate, cap utilization, blocked triggers, cap-skips, rescue counts/precision, rescued shortfall, `L_λ` on the frozen grid {0.25, 0.5, 1, 2, 4} plus the fine grid {0, 0.02, …, 0.20}, runtime.
- Primary contrast: paired per-origin `P − F_k`; **descriptive** IID-over-origin bootstrap (10,000 draws, seed 20261011) and leave-one-origin-out means; per-origin sign counts. No Holm family; no confirmatory, equivalence or "match" language: a CI covering zero is reported as "interval includes zero", never as equality.
- Materiality threshold δ = 0.01 on `W7` (inherited from the frozen η-sensitivity convention): |Δ| < δ ⇒ "not materially distinguished"; Δ ≥ δ ⇒ "materially higher on this exposed replay".
- Transfer cohorts: identical procedure per cohort, reported separately, descriptive only (5/12/6 origins).

## 7. Budget-binding and oracle headroom (interpretation fixed before running)

- **Unlimited-cap sensitivity**: replace the cap by 10⁹ in the production rule and replay. This is a *mechanical sensitivity of the closed loop* — removing the cap changes the actions themselves, so the difference **does not identify a causal "budget component"** separate from placement. Report with that wording.
- **Oracle at count** (verification of the admitted R43 object): max-min-7 MILP with `Σa = K_origin`; Stage 2 maximizes rescued shortfall at fixed optimum; Stage 3 deterministic sparsest tie-break. Verified against the admitted per-origin headroom list `(+.154762, +.178571, +.184524, +.267857, +.178571, +.190476, +.113095)` and macro `0.911565` / DRB `0.730442` (tolerance 1e-6).
- **Oracle at cap** (`Σa ≤ B`): same objective family, new object. `oracle_at_cap − oracle_at_count` = explicit **quota-headroom** calculation; `oracle_at_cap − P` = mixed information/policy/placement headroom, never a single causal quantity.
- Solver: `scipy.optimize.milp` (HiGHS); options `time_limit = 300 s`, `mip_rel_gap = 0.0`, `presolve = True`; `OMP_NUM_THREADS = 2` set by the caller and recorded; status, MIP gap, node count, dual bound and runtime recorded per solve. A non-optimal termination is reported as such; an LP relaxation would be labeled an upper bound and never substituted silently.

## 8. Upper-direction audit (units fixed before running)

Quantity semantics verified from `nyiso_confirmatory_audit.py` (hash above): `reserve_<d> = d·center + sqrt(th · d'Σd)` is the **upper directional bound**, and `actual_<d> = d·error` is the realized projection of the **bias-corrected day-ahead forecast error** (actual − forecast, ridge-corrected forwards; NYISO zonal load units, MW scale). These are forecast-error coordinates, **not metered load levels and not outage magnitudes**; positive error = actual above forecast.

Per stored hour with `covered == 0` (joint ellipsoid miss) and per origin: share with ≥ 1 zone upward exceedance (`actual_z > reserve_z`), magnitude of exceedance, zone attribution; share that is lower-side for at least one zone (marginal miss with `actual_z ≤ reserve_z` implies `error_z < center_z − radius_z`), and share that is joint-geometry-only (all marginals covered). Macro over origins; the equal-weight aggregate direction reported separately. No renaming to outages; no `s=1` physical-delivery assumption.

## 9. Guards, resumability, execution

- Guard suite (`r1_tests.py`): T1 hand-computed F5 state-machine cases; T2 eligibility invariance (all ineligible + current + future q perturbed, cutoff-available information fixed, actions at/before the pivot unchanged) for P/F3/F4/F5; T3 segment relabel invariance; T4 cap arithmetic and F1/F2 accounting; T5 determinism; T6 reverse replay on all 30 origins.
- Stage 0 tombstone-first fail-closed gate; **zero unresolved identity mismatches permitted**; a failing stage retains its failing object and stops that stage only.
- Resumability: per-origin outputs with fingerprint sidecars (input hashes + code hashes); identical fingerprint → SKIP; output SHA-256 verified on reuse; timings logged. Every manifest records access flags: `reserved_outcome_accessed=false`, `raw_outcome_accessed=false`, `network_used=false`, `ssh_used=false`.
- Execution order: T-suite → Stage 0 → one-origin smoke (`2018->2019`, all stages) → full suite. Local Windows Python 3.11.15, numpy 2.4.6, pandas 2.3.3, scipy 1.17.1 (matches the admitted p0_2 environment); these trace-scale replays are local per the reviewer's "local tiny trace replay is appropriate"; no GPU use; no GPU services touched.

## 10. Pre-committed interpretation rules (fixed before results)

| Outcome | Reading |
|---|---|
| P ≥ every deployable rule by ≥ δ on ≥ 5/7 NYISO origins and macro CI excludes zero | Report: the production rule's placement is materially higher than deployable pacing/threshold/risk rules on this exposed replay. Exposed; descriptive; no SOTA language. |
| P materially higher than some, not all | Name the families; narrow the contribution statement accordingly. |
| Some deployable rule materially higher than P | Report it; **no adoption**, no retuning; the production rule is described as not the best exposed-data allocator; the finding motivates (not constitutes) any prospective evaluation. |
| Differences within δ or CI covering zero | "Not materially distinguished on this exposed replay"; never "matches"/"equivalent". |
| F5 flagged `bank_open_at_trace_end` on any origin | Report macro with and without flagged origins. |
| Solver non-optimal on any origin | Retain the failing object; report the origin's oracle as unresolved (LP bound labeled); no substitution. |

## 11. Amendment discipline

Any change to code or definitions after the freeze invalidates this protocol; an amendment (dated, with the reason and a new hash table) must be written before the corrected run, and outcomes produced under the superseded version are re-run and marked superseded in the manifest.
