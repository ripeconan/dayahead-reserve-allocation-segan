# R-A protocol amendment C — oracle v3 (exposed outputs already exist)

- Applies to: `R1_PROTOCOL_FROZEN_2026-10-11.md` (sha256 `3ebfa095…`, bytes unchanged) and `R1_PROTOCOL_AMENDMENT_2026-10-11B.md` (sha256 `f9bd7e60…`).
- Date: 2026-10-11 · author: Claude Code (modifier role) · trigger: reviewer `R02B_ADDITIONAL_REVIEW_NOTES.md` items 1–4 and the R03 result-interpretation corrections.
- **State disclosure (required):** exposed outputs now exist. Under amendment B the following were produced before this amendment: guard suite; Stage 0; the full equal-cap allocation suite (30 units); the v2 oracle units (7 origins); the direction audit; the aggregate summary. This amendment does **not** pretend those changes preceded all outcomes. Only the **oracle units** are regenerated (v2 → v3) because only `r1_budget_oracle.py` (and the test file) changed; `r1_common.py`, `r1_policies.py` and `r1_allocation.py` bytes are unchanged, so the allocation fingerprint is unchanged and its outputs remain valid without re-running.

## 1. What v3 fixes

| Reviewer item | Fix in v3 |
|---|---|
| 1 — T7 toy vacuous | `r1_tests.py` T7 replaced by the true conflict case (n=10, cap=1, q=1 except q0=0, q6=0.4, q9=0.2): max-rescue alone picks day0 (rescue 1, W7 0.8); max-min W7 requires day6 (W7 6/7, rescue 0.6). Result: solver z = 6/7 and rescue = 0.6 match exhaustive enumeration; a regression note records the previous toy's defect. |
| 2 — self-referential resume hash | Unit files no longer contain a hash of their own bytes. A `canonical_payload_sha256` (hash over the canonical payload **excluding** every `*_sha256` field) is stored in the unit and in an **external sidecar** together with the unit's true file sha256; resume verifies sidecar ↔ canonical ↔ file hash **and re-derives** every objective value and constraint from the retained `action_vector` (window list, count/cap, rescue, W7, R43 headroom) instead of trusting stored PASS booleans. Each unit retains the full action vector, every within-segment window index/mean, and the exact origin support (dates sha256). |
| 3 — diagnostics lost on failure | Solver failures raise `OracleSolveError` carrying the stage diagnostics + artifacts; per-origin FAIL objects embed them under a unique attempt id; older FAIL objects are **never unlinked** — an `attempts.json` supersession record is appended. |
| 4 — aggregate semantics | The expected origin set is enumerated from the input trace; partial/smoke runs are `PARTIAL_INCOMPLETE` with a "macro comparison not applicable" note; the seven-origin macro check runs only for the complete cohort with zero failures, and missing/unexpected units are listed explicitly. |
| interpretation errata | `R1_RESULTS_SUMMARY_ERRATA_2026-10-11.md` (dated, preserves the original summary) corrects: W14 (F2 > P), rescue precision (P below F1/F2/F3), transfer crossings (P−F4 positive on two cohorts), the direction-audit wording (upper-support breach ≠ load above point forecast), and the oracle-gap wording (full-outcome ceiling comparison). |

## 2. Superseding code hashes

| File | SHA-256 | Note |
|---|---|---|
| `scripts/r1_budget_oracle.py` | `53b1ab05a679e756df1384aa61d47e341a263cde192b6f131ad8abbba8d7f798` | v3 |
| `scripts/r1_tests.py` | `e3100d4549c142905b30d3548092d119156c553aa57c72d5ff9d8088c6ea1bd8` | v3 tests |
| `scripts/r1_common.py` | `ba07e287492f3cc9b08b28ea689a9524495e9b2f70012bba40bf7420d8f0aa0a` | unchanged since B |
| `scripts/r1_allocation.py` | `366f4f94f9979ee17acacbacb218c9c1b05b0c96a4a4da475c9578655ccbdf52` | unchanged since B (allocation outputs stay valid) |
| `scripts/r1_policies.py` | `a9643033b158da9859fb656b5c9f3b69fb6a073ab0d34fbc207a5222b7f296bc` | unchanged |

## 3. Preservation

The v2 oracle units are preserved byte-for-byte under `outputs/budget_oracle/superseded_v2_2026-10-11B/` together with the reproduced self-hash defect (stored `output_sha256` ≠ file hash). No other exposed output is modified.

## 4. Regeneration scope

Full cohort (7 origins) v3 regeneration, followed by an immediate second invocation to demonstrate a verified resume: all units SKIP only when sidecar + canonical hash + file hash match **and** the retained-action revalidation passes; otherwise the unit is recomputed and an attempt record is written.
