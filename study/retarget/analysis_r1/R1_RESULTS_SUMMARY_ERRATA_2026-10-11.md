# Errata for `R1_RESULTS_SUMMARY.md` (dated 2026-10-11; original preserved)

Trigger: reviewer R03 result-interpretation corrections. The original summary file is kept byte-for-byte (review trail); the items below **supersede** the corresponding statements there. All numbers are recomputed from the same machine objects (`outputs/summary/r1_summary.json`, `outputs/budget_oracle/*`).

## E1. NYISO W14 and rescue precision (original overstated dominance)

- **W14:** F2 = 0.8095 **>** P = 0.8091. Production does **not** lead W14 (F1 = 0.8010, F5 = 0.8027, F3 = 0.7751, F4 = 0.7870).
- **Rescue precision:** P = 0.4942 is **below** F1 = 0.5742, F2 = 0.5617 and F3 = 0.5430 (and above F5 = 0.4681, F4 = 0.4481).
- Supported statement (unchanged): on the primary exposed cohort the production rule has the highest **W7** (0.7304) while using 280 total actions versus F1 361 / F2 376 / F3 383; no other dominance is claimed.
- Note (kept): F1's W3 = 0.4861 > P's 0.4623.

## E2. Transfer trade-off crossings (original "all negative" was false)

Exact frozen crossing table (`ΔL(λ) = Δshortfall + λ·Δrate`, macro; λ* = crossing):

| Cohort | P−F1 | P−F2 | P−F3 | P−F4 | P−F5 |
|---|---:|---:|---:|---:|---|
| GEFCom | −0.5009 | −0.4018 | −0.0695 | **+0.0391** | equal-rate; ΔL = +0.00657 |
| CAISO dev | −0.7726 | −1.0101 | −0.0338 | −0.0139 | equal-rate; ΔL = +0.00091 |
| CAISO post | −0.9852 | −0.3302 | −0.0480 | **+0.0776** | equal-rate; ΔL = +0.00362 |
| NYISO | +0.1281 | +0.1226 | +0.0615 | +0.0846 | equal-rate; ΔL = −0.00127 |

Correct reading: on the transfer cohorts, P's mean shortfall is **higher** than F1–F3 at a higher rate, so ΔL(λ) > 0 for every λ ≥ 0 for those comparators; **for F4 the crossing is positive** (λ* ≈ 0.039 GEFCom, ≈ 0.078 CAISO post — i.e. P is cheaper only for κ below those values), and on CAISO dev the F4 crossing is slightly negative (−0.014). On NYISO every crossing is positive (0.0615–0.1281), i.e. P's loss advantage holds only for λ below those values. The original blanket sentence "all P−F crossings are negative on transfer" is withdrawn.

## E3. Direction-audit wording (original conflated two events)

- The 5.5 % figure is the share of joint-region missed hours with **≥ 1 zone whose realized bias-corrected residual projection exceeds the region's upper directional support bound** (`actual_d > reserve_d = center_d + radius_d`). It is a *sufficient* marker of a positive residual beyond the fitted margin in that zone.
- It is **not** the share of hours with "actual load above the point forecast": a positive realized residual smaller than that zone's margin does not breach the bound, and such hours are included in the 85.2 % joint-geometry-only share. The equal-weight aggregate direction breaches its bound in 1.5 % of missed hours.
- Supported statements (unchanged): the audit is in residual coordinates (MW scale); per-zone upper bounds are `center_z + r_z`; joint-geometry-only misses dominate (85.2 %); lower-side-only misses are 9.2 %.

## E4. Oracle-gap interpretation (original causal phrasing withdrawn)

- Correct identities: **count-ceiling gap** = `oracle_at_count − production` = +0.181122 macro (admitted R43 semantics: same per-origin realized count, within-segment worst-7, full-outcome noncausal); **extra-cap gain** = `oracle_at_cap − oracle_at_count` = +0.016785 macro (cap = `round(0.15·n_days)`); **mixed ceiling gap** = `oracle_at_cap − production` = +0.197907 macro.
- These are full-outcome ceiling comparisons for the fixed replay problem. They quantify how much an exact allocator could add at the same count and with the full quota; they do **not** decompose the deployed policy's loss into causal components (information vs placement vs budget), and the unlimited-cap replay is a mechanical closed-loop sensitivity, not a budget-component estimate.

## E5. Record-keeping

- Oracle units were regenerated as v3 under `R1_PROTOCOL_AMENDMENT_2026-10-11C.md` (`32b09c86…`); v2 bytes are preserved under `outputs/budget_oracle/superseded_v2_2026-10-11B/` (including a reproduction of the v2 self-hash defect). Allocation outputs are unchanged (their bound files are byte-identical to amendment B).
