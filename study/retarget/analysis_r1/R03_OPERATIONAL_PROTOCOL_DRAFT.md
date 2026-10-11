# R03 operational protocol — DRAFT for reviewer approval (not executed)

Date: 2026-10-11 · author: Claude Code (modifier role) · status: **draft, awaiting reviewer approval**; no run has been performed; no new outcome has been accessed.

Scope: (A) repair the point-forecast stage so the seven-origin NYISO base is genuinely available at the day-ahead cutoff, then (B) run the finite-reserve decision bridge on that repaired base. The two stages get separate identities, protocols and audits.

---

## 0. Why the point stage must be repaired (independently confirmed chain)

| Step | Evidence (read-only) |
|---|---|
| The headline NYISO base is downstream of the `forward_correction` stream | `reports/experiment_outputs/card10_nyiso_saocp_drb_current_r03_2026-07-15/MANIFEST.json` binds corrected file SHA-256 `730580f0fdfa894a8475d24b14e8abb86041747ae4f6c5e28fa989878b3b9752`, produced by `forward_correction` at deployment commit `c003bc412d599110286e543b05d247180d5d613c`, runtime 8.53 s, `matched_historical=false` |
| The R39 runner consumes exactly that file | `scripts/card10_baselines/run_nyiso_scientific_matrix.py:54,1049` hard-fails unless the corrected SHA equals `730580f0…`; `run_nyiso_verified_saocp.py` passes `load_corrected(args.corrected)` into evaluation |
| At that commit the point correction is **not** cutoff-available | `git show c003bc41…:scripts/card10_killtest/nyiso_confirmatory_audit.py`: `build_features` builds error features at lags `(24, 168)` **elapsed hours**; `forward_correction` fits training labels at indices `< day` (labels through t−1). A D-1 full-day freeze cannot use day t−1 same-hour errors |
| An existing lag-2 helper exists but needs a lag-rule repair | `scripts/card10_confirmatory/lag2_pipeline.py`: `ERROR_LAGS_HOURS = (48, 168)`, forbids `lag24`, training window `< local midnight of t−1` via `DateOffset` (latest completed label t−2). However the feature lags are *elapsed hours*: verified with synthetic tz-aware timestamps, target `2024-11-04T23:00:00-05:00` minus 48 elapsed hours = `2024-11-03T00:00:00-04:00`, calendar age 1 day → ineligible under the completed-day contract; `DateOffset(days=2)` correctly lands on `2024-11-02` (calendar age 2) |

Conclusion: R-A remains a conditional allocation replay on the old fixed traces (its results are unchanged and it makes no end-to-end availability claim). The R03 base must be regenerated with a cutoff-safe feature-lag rule before any finite-reserve study.

## 1. Stage P: repaired point/UQ/controller base (seven origins)

**Identity:** `card10-segan-R3P-point-lag2-calendar` · frozen before execution · new outputs only; the old `730580f0` corrected stream is retained unchanged as historical/sensitivity evidence and never relabelled.

**Inputs (pinned):**
- Raw exposed NYISO zonal 2017–2025: `/home/icdm/card10_confirmatory/data/nyiso_zonal_2017_2025.csv.gz`, SHA-256 `d9e492532e22bc3f41c6967acf99cc5b49a49ae2e5ba3ae3b7dbc8092d9e5d0d` (present on `inner`).
- SAOCP pin: `salesforce/online_conformal` commit `95790a9b9e4061166e6d7a8a13282e019b1a8663`, tree `77348e5fe7f2bb3ed5a031aaad9874b6a5ab2d7b`; `online_conformal/saocp.py` SHA-256 `83f8845559786452aafabfafdf6c136ceaf9b9b3557fc5f8f9267fcec04d386f` (present on `inner`); full clean-clone verification required before execution.
- OAS/reference geometry, point-correction constants, origin calendar: inherited from the admitted pipeline (`train_days=365`, `penalty=10.0`, reference year per origin, seven origins `2018->2019 … 2024->2025`).
- Numeric stack: `/usr/bin/python3` (3.10.12; numpy 2.2.6, pandas 2.2.3, scipy 1.15.3) is the proposed runtime — recorded, not claimed bitwise-equal to the earlier stack; the 3.9 conda env is *not* assumed compatible.

**Feature-lag rule (frozen; the core repair).** For a target hour `h` on local delivery date `d` (timezone-aware `America/New_York`), the lagged-error features are the realized errors at the **same local hour on local dates `d−2` and `d−8`**, obtained by `index - pd.DateOffset(days=2 | 8)` (local-calendar mapping; `DateOffset` preserves local wall-clock across 23/25-hour DST days). Tie handling on ambiguous fall-back hours: `fold=0` (first occurrence) — fixed, not tuned. A conservative alternative contract (elapsed **72 h**, `Timedelta(hours=72)`) is pre-registered as a fallback if the calendar mapping fails its provenance checks; the choice is made by the provenance tests below, never by test scores.

**Newest-label checks (all must pass before any evaluation output).**
1. Every lagged feature used for target date `d` satisfies: feature local timestamp strictly `< local midnight of d−1` (i.e., calendar age ≥ 2 days) — checked on **every row**, not only the newest.
2. Ridge training labels for target date `d` use only indices `< local midnight of d−1` (latest completed label date `d−2`), implemented with `DateOffset`, asserted per day.
3. Spring/fall DST tests (synthetic timestamps, no outcomes): mapping of 2024-03-10/11 and 2024-11-03/04 targets resolves to the correct local dates; ambiguous times fold=0; nonexistent times shift deterministically.
4. Perturbation invariance: perturbing all target-`d`−1 / target-`d` / future realized errors leaves (i) the issued point forecast for `d`, (ii) the frozen region (threshold/geometry), and (iii) the controller action for `d` unchanged; prefix comparison through `d`.
5. Reference/test support identity and cohort guards fail closed; no silent row drops.

**Pipeline (per origin, resumable, one unit = one origin × stage):**
raw zonal (MW) → errors = actual − official P7 → calendar-lag features → daily-forward ridge correction (train `< d−1` midnight only) → corrected residuals (MW) → reference-year OAS ellipsoid (squared Mahalanobis, target joint coverage 0.90) → SAOCP thresholds with strict lag-2 eligibility and segment resets (pinned adapter, as admitted) → daily `q_t` → the **same declared causal comparator menu** as R-A (P production DRB, F1 pace7, F2 paceBank, F3 K2, F4 persist1, F5 declared-end bank) with identical caps `round(0.15·n_days)`, identical endpoints (worst-3/7/14, mean service, actions, rescue, `L_λ` grid + fine grid + crossings), identical descriptive inference (paired origin bootstrap, LOO). The regenerated base must also store, per hour: point forecast center (MW), per-zone radii/upper bounds (MW), realized load and error (MW) — the payload the finite-reserve bridge requires with verified units.

**Execution and gates:** isolated native-runtime smoke on one origin with retained stderr/no silent retries → provenance tests → reviewer-side check of the smoke manifest → full seven-origin CPU run on `inner` (logical CPU 8 excluded, two threads, per-origin checkpoints, retained failure logs). Heavy formal execution on the server; the smoke may be run locally only if the pinned pin/raw bytes are made available locally, otherwise on `inner` directly with the same guards.

**Expected cost:** point correction ≈ seconds per origin (8.5 s recorded for the whole 2018–2025 stream); geometry/SAOCP replay ≈ the admitted R39 scale (≈380 s for seven origins); no GPU, no deep retraining.

## 2. Stage F: finite-reserve decision bridge (runs only after Stage P passes)

**Question.** With a *finte, deterministic* reserve augmentation on action days — instead of the abstract `s_t = 1` — what do the same causal allocators achieve in verified energy units, and at what cost ratio?

**Decision semantics (fixed):** an action day commits a reserve block of `H` MW for that delivery day (all 24 hours, `H` is headroom). Response model: in each delivery hour, the committed headroom covers positive exceedance up to `H` where the realized (or, in the causal view, the region-based) deficit exceeds the support; excess beyond `H` remains uncovered. Full-fallback `s=1` stays the paper's abstract model, never presented as physical delivery.

**Quantities in verified MW/MWh (from the Stage-P hourly payload):**
- per-hour positive exceedance energy `E⁺ = Σ_z max(0, actual_z − upper_z) × 1 h` (MWh), with `upper_z` the per-zone support bound — units verified against the producer (MW load) before use;
- uncovered energy after the reserve; capacity commitment `H × (hours committed)` (MWh-equivalent);
- **cost/shortfall ratio grid** κ = (capacity cost per committed MWh) / (shortfall cost per uncovered MWh), predetermined: κ ∈ {0.01, 0.05, 0.1, 0.25, 0.5}, reported as the full curve; no κ selected after seeing results.

**Calibration rule (reference-only).** `H` is either (a) a fixed grid {10, 25, 50, 100, 200 MW} or (b) calibrated on the **reference period only** (e.g., reference-year 95th percentile of hourly `E⁺`), frozen per origin; test rows are never used for calibration. If a chosen `H` requires per-zone attribution that the regenerated payload does not carry, that component is declared "needs regeneration" rather than approximated.

**Allocators compared:** identical to R-A (P, F1–F5), same caps and eligibility; endpoints are the finite-response service, uncovered energy, committed capacity, and the κ-grid net-cost curve; inference descriptive, paired, LOO; exploratory labels.

**Data prerequisites and current status:**
| Requirement | Status |
|---|---|
| Per-hour point center + per-zone support in MW, repaired base | **needs regeneration** (Stage P stores it); the old R39 hourly payload carries center+radius already summed (`reserve_*`) in residual coordinates and cannot be split into MW center/radius |
| Realized load/error per zone-hour in MW | available in the raw zonal file (pinned); regenerated by Stage P |
| Reference-only calibration series | derivable from Stage P's reference window per origin |
| Fresh-period (2026+) acquisition | **preparation only**; no access in R03; separate protocol and reviewer authorization required |

## 3. Approval gates and prohibitions

- Gate P1: reviewer approves this protocol + the Stage-P feature-lag rule; Gate P2: smoke + provenance tests pass; Gate P3: reviewer inspects the Stage-P full-run manifest before Stage F starts.
- Prohibited: presenting the old lag-24-corrected region as an available day-ahead commitment anywhere in the paper; tuning the lag rule or `H` by test scores; touching reserved/sealed objects; touching 2026 outcomes without a new protocol and authorization; retraining deep models or using GPUs (not needed).
- The paper's timing description will be updated (concise, one paragraph) only after Stage P passes; the current R02B prose keeps the point-timing finding in this plan and the review response, not in the abstract.
