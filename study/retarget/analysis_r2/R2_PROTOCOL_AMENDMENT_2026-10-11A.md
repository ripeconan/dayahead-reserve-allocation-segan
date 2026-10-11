# R2 protocol amendment A — post-freeze repairs (root mailbox patch-F review)

- Applies to: `R2_PROTOCOL_FROZEN_2026-10-11.md` (bytes unchanged, sha256 `6a4238286ea66b1c4c0de05feb3239931658c42c9db23f24b284116481cdd982`).
- Date: 2026-10-11 · author: Claude Code (modifier) · trigger: root mailbox review of patch F (nine concrete items).
- **No formal outcome exists.** All work remains synthetic/local; this amendment records corrections made after the freeze, per the protocol's own §9 discipline.

## Repairs

| Mailbox item | Defect | Repair |
|---|---|---|
| 9 (critical) | `r2_pointcorr` issuance still used `.notna()`; one +inf issuance input produced 24 rows and nonfinite predictions | issuance mask now `np.isfinite(features)` over the actual feature rows; reviewer counterexample rerun: 23 expected / 23 actual issued rows, non-finite input excluded, all emitted predictions finite (`REVIEWER_R2_FINITE_ISSUANCE_RESULT` regenerated PASS; the failing JSON is preserved in `superseded_r03c_2026-10-11/`) |
| 1 | modeled contrast tuples were `(label, none, policy)` while computing left−right → emitted `none − policy` under a `policy_minus_none` label | pairs are now `(label, left, right)` = `(policy, none)`; `P − F_i` pairs verified correct; physical contrasts extended to BOTH `policy − none` and `P − F_i` at every (η, κ) cell |
| 2 | resume compared only thread env; protocol sidecar not verified; malformed JSON outside the logger; aggregate did not enforce identities | resume now compares full numeric identities (Python/numpy/pandas/sklearn/scipy + thread env), verifies the protocol SHA-sidecar, re-checks the CURRENT checkout's clean pinned identity via `load_ast_kernel` before SKIP; JSON/parse failures are inside the guarded body (stage + traceback retained); aggregation verifies sidecar, canonical payload, manifest, and exact grid keys before any effect |
| 3 | — | reference-history capture passes (8568 == 8568 eligible prior-year labels); the production frame now spans the admitted history start (root's independent reconstruction is pending) |
| 4/7 | UTC string keys with mixed offsets; intersection could hide missing keys | all cross-representation comparisons normalize through `pd.to_datetime(..., utc=True)` and REQUIRE exact key-set equality before value comparison |
| 5 | `all(Series.is_monotonic_increasing)` TypeError and a tautological sorted-then-check | chronology asserted on the ORIGINAL emitted order via `pd.to_datetime(..., utc=True).is_monotonic_increasing` |
| 6 | P9 stale schema (3-day frame, `base` kwarg, old keys) | P9 rebuilt on the reviewer's exact 9-day arithmetic (deficits 1/5/2/0/0/0/0/10/0 → worst-7 MAX 17, uncovered 18, full cost at κ=0.1 = 19) and now also asserts that omitting `segments_by_date` fails |
| 8 | P10 compared `seconds` → always unequal | determinism compares full numerical frames plus the payload with only the runtime `seconds` field excluded |

Optional performance (not a scientific change): `paired_origin_bootstrap` remains the Python-draw loop in this amendment; the vectorized-draw equivalence was not yet verified locally, so it is NOT adopted.

## New code fingerprint

The frozen protocol's §2 code-identity statement is superseded by the fingerprint produced at run time by `r2_run._fingerprint()` (it hashes the actual files, including this amendment's changes: `r2_pointcorr.py`, `r2_run.py`, `r2_guards.py`). Run-time fingerprints are recorded in every manifest; this amendment does not claim a static hash table.
