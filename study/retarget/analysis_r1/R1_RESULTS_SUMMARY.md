# R-A results summary (exposed replay, descriptive) — 2026-10-11

Protocol: `R1_PROTOCOL_FROZEN_2026-10-11.md` (sha256 `3ebfa095…`, bytes unchanged) + `R1_PROTOCOL_AMENDMENT_2026-10-11B.md` (sha256 `f9bd7e60…`).
Status: guards 9/9 PASS; Stage 0 PASS; allocation 30/30 origins; oracle 7/7 units PASS (admission vs admitted R43); direction audit complete.
All values are **exposed-data descriptive** results: no confirmatory language, no equivalence/"match" claims, no adoption.

## 1. Identity and guards

- Reverse replay: production DRB re-derived == stored actions on 7/7 NYISO origins (R39) and 23/23 transfer origins (p0_2 continuous production); K2 re-derived == p0_2/R51 thresholded-count rows 23/23.
- Caps: `round(0.15·n_days)` matches the stored production cap on all 30 origins (NYISO 55/55/53/55/55/55/55).
- Naming finding: R51's original `drb_action` column carries the K2 rule, not the continuous production rule (both identities pinned; the paper text should keep calling the transfer variant "the distinct thresholded-count variant").
- The earlier r02 NYISO stream is a *different* UQ run (0 missing/extra days, max |Δq| 0.958, max |Δthreshold| 372.6 vs R39): used for identity diagnosis only, never combined.

## 2. Equal-cap allocation suite (primary cohort: NYISO exposed, 7 origins)

Macro (equal-weight origins); actions are `mean_actions_per_origin` / `action_total_sum`:

| Policy | W3 | W7 | W14 | mean service | mean act | act sum | rate | L1 | rescue prec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P production DRB | 0.4623 | **0.7304** | 0.8091 | 0.9181 | 40.00 | 280 | 0.1101 | 0.1920 | 0.4942 |
| F1 pace7 | 0.4861 | 0.6905 | 0.8010 | 0.9222 | 51.57 | 361 | 0.1419 | 0.2198 | 0.5742 |
| F2 paceBank | 0.4464 | 0.6828 | 0.8095 | 0.9227 | 53.71 | 376 | 0.1479 | 0.2251 | 0.5617 |
| F3 k2_count | 0.3929 | 0.6395 | 0.7751 | 0.9206 | 54.71 | 383 | 0.1506 | 0.2300 | 0.5430 |
| F4 persist1 | 0.4345 | 0.6701 | 0.7870 | 0.9110 | 9.43 | 66 | 0.0259 | 0.1150 | 0.4481 |
| F5 bank1_causal | 0.4345 | 0.7024 | 0.8027 | 0.9168 | 40.00 | 280 | 0.1101 | 0.1933 | 0.4681 |

Note the honest wrinkle: on the 3-day endpoint, F1's 0.4861 exceeds P's 0.4623 (pacing spreads actions more evenly at short scale), while P leads on W7/W14 and on rescue precision at far fewer actions than F1–F3.

Paired `P − F_k` on W7 (descriptive IID-over-origin bootstrap, seed 20261011; LOO = leave-one-origin-out range):

| Contrast | effect | 95% interval | positive-share | LOO range |
|---|---:|---|---:|---|
| P − F1 | +0.03997 | [−0.01020, +0.09439] | 0.935 | [+0.0179, +0.0575] |
| P − F2 | +0.04762 | [+0.02636, +0.07398] | 1.000 | [+0.0357, +0.0526] |
| P − F3 | +0.09099 | [+0.04422, +0.15051] | 1.000 | [+0.0645, +0.1062] |
| P − F4 | +0.06037 | [+0.02041, +0.10714] | 1.000 | [+0.0407, +0.0704] |
| P − F5 | +0.02806 | [+0.00680, +0.05272] | 0.998 | [+0.0188, +0.0327] |

All effects exceed the pre-declared δ = 0.01; four of five intervals exclude zero (F1's includes zero, with all LOO means ≥ +0.018).

**F5 ≡ the admitted budget/end-conditioned COCD on this cohort**: F5's per-origin W7 equals the R45 COCD values to ≤ 5×10⁻¹⁵ on 7/7 origins, and the action path is identical on 6/7 origins (one origin differs in allocation but not endpoint). The deployable declared-calendar-end bank therefore reproduces the conditioned diagnostic's endpoint here; the R57 comparison (DRB 0.730442 vs COCD 0.702381) is unaffected.

Policy flags: F1 `all_windows_have_action = true` on all 7 origins, `cap_skipped_scheduled_days = 0` (the conditional 1-action-per-7-window property holds empirically here). P blocked triggers: 4 total, all in 2023→2024 (matches the admitted blocked-trigger audit). F3 has 75–155 blocked triggers per origin (it exhausts the cap). F5: `bank_open_at_trace_end = false`, zero credits dropped on gaps.

## 3. Transfer cohorts (secondary, descriptive, per cohort)

P is not the best rule on any transfer cohort; the frozen K2/persistence rules beat it while spending fewer actions:

| Cohort | P W7 | F1 | F2 | F3 | F4 | F5 | P−F3 (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---|
| GEFCom replay | 0.5714 | 0.5714 | 0.6000 | **0.6571** | **0.6571** | 0.6000 | −0.08571 [−0.14286, −0.02857] |
| CAISO development | 0.6667 | 0.6786 | 0.6667 | **0.6905** | 0.6905 | 0.6667 | −0.02381 [−0.07143, +0.02381] |
| CAISO exposed post | 0.5714 | 0.6190 | 0.5714 | **0.6429** | 0.5952 | 0.5714 | −0.07143 [−0.11905, −0.02381] |

P's action rate on transfer is ≈15.1–15.3 % (cap-exhausting); K2 uses 9.7–12.3 %.

## 4. Fine-grid trade-off and crossings

`ΔL(λ) = Δmean-shortfall + λ·Δrate` over the frozen fine grid {0, 0.02, …, 0.20}:

- NYISO: P−F1 crossing λ\* = 0.128; P−F3 λ\* = 0.0615; P−F2, P−F4 crossings below 0.05; P−F5 has **equal action rate** (Δrate = 0) with Δshortfall = −0.00127, i.e. P's mean-shortfall loss is lower at every λ.
- Transfer: all P−F crossings are **negative** (P has higher shortfall *and* equal-or-higher rate), i.e. P is dominated on L(λ) for every λ ≥ 0 there.

## 5. Budget-binding and oracle headroom (NYISO, per origin)

- **Unlimited-cap sensitivity (mechanical):** removing the cap leaves P's W7 unchanged on 7/7 origins (macro 0.730442 both; actions 23–59 vs 40/origin capped; 2023→2024 uses 59 > 55 with no endpoint change). On this cohort the cap is essentially non-binding for the production rule; the sensitivity does not identify a causal budget component (the closed loop itself changes).
- **Oracle at count (verification of the admitted R43 object):** macro W7 = 0.911565 (admitted 0.911565; max per-origin headroom error 4.3×10⁻⁷; DRB macro 0.7304422 vs 0.730442) — the sign-corrected solver reproduces the admitted objective values; all rounded-action validations pass; max MIP gap 2.1×10⁻¹⁶.
- **Oracle at cap (`Σa ≤ B`, new):** macro W7 = 0.928350 at 54.7 actions/origin. **Quota headroom** (cap-oracle − count-oracle) = **+0.016785** macro; **mixed headroom** (cap-oracle − DRB) = +0.197907 macro. Reading: at the same realized action count, an exact noncausal allocator would gain ≈0.181; allowing the full cap adds only ≈0.017 more, so the dominant gap is information/placement, not the quota.

## 6. Upper-direction audit (per-hour, all 7 NYISO origins)

Of 5,562 joint-region missed hours (miss rate 9.11 %):

- **5.5 %** involve ≥ 1 zone whose realized (bias-corrected) forecast-error projection exceeds the region's upper directional bound; the equal-weight aggregate direction exceeds its bound in 1.5 %.
- **85.2 %** are joint-geometry-only misses (every zone's marginal interval covered; the ellipsoid missed through cross-zone geometry).
- **9.2 %** are lower-side-only (below the lower directional bound in at least one zone).
- Median of the maximum per-hour upward exceedance among those rare upward misses: ≈ 23.4 (residual-coordinate units; NYISO zonal MW scale).

**Interpretation (fixed before running):** region misses on this benchmark are predominantly *not* directional load-above-forecast events; they are joint-shape events. Any reserve/deficit reading of the fallback proxy must therefore be stated as partial and directional-share-quantified, not assumed.

## 7. Output locations

- `outputs/stage0/r1_stage0_verification.json`, `outputs/tests/r1_tests.json`
- `outputs/allocation_value/*.json` + `*__daily.csv.gz` (+ fingerprints) and `allocation_run_meta.json`
- `outputs/budget_oracle/nyiso__*.json` + `budget_oracle_summary.json`
- `outputs/direction/r1_direction_audit.json` + `r1_direction_audit_hourly.csv.gz`
- `outputs/summary/r1_summary.json` + `r1_summary.md`
