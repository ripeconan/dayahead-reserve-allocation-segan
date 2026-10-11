# R2 protocol amendment C — pre-outcome repair of the input timezone normalization (native P2 defect)

- Applies to: `R2_PROTOCOL_FROZEN_2026-10-11.md`, whose bytes are **unchanged** at sha256
  `6a4238286ea66b1c4c0de05feb3239931658c42c9db23f24b284116481cdd982`.
  Amendments A (`R2_PROTOCOL_AMENDMENT_2026-10-11A.md`) and B
  (`R2_PROTOCOL_AMENDMENT_2026-10-11B.md`) are preserved unchanged.
- Date: 2026-10-11 · author: Claude Code (modifier role) · trigger: R03F — one native defect in the
  full 15-case guard suite.
- **Status: still no formal outcome.** This amendment records an input-handling repair; it contains
  no scientific result and no static fingerprint (the run-time fingerprint is computed over
  docs + code after they exist).

## 1. The defect

The full native suite (release `R2_candidate_20261011_1108`) passed **fourteen** cases;
`P2_utc_local_equivalence` ended in `ERROR`:

```
File ".../r2_pipeline.py", line 109, in run_origin_from_frames
  grid = pd.date_range(input_index.min(),
                       pd.Timestamp(f"{test_year}-12-31 23:59:59", tz=C.TZ), freq="h")
TypeError: Start and end cannot both be tz-aware with different timezones
```

`run_origin_from_frames` took the grid start from the **caller's** index while the end was fixed in
`America/New_York`. A New_York-indexed input worked by luck; a UTC-indexed input holding the *same
instants* (guard P2, and the reviewer capture with `--timezone UTC`) crashed **before any fit**.
The prefix/tension/admission/theory/bootstrap guards are unaffected: the failure is in the outer
production input path, not in the prefix comparator or the forecaster.

Independent corroboration: the same bytes (`r2_pipeline.py` sha256
`a1adf6fd934abc4eebc548982aacc24c00315ba1f5387839bc0b0a94f72a71ab`) failed identically in an
independent local guard run (P2 `ERROR`, 98.55 s) — evidence preserved in
`analysis_r2/superseded_r03e_2026-10-11/outputs/guards_v4/r03e_final_frozen_part2.json`; the native
capture is preserved as
`review_2026-10-11/REVIEWER_R2_REFERENCE_HISTORY_RESULT_NATIVE_OUTER_UTC_BEFORE_FIX.json`.

## 2. The repair (input handling only — no model, no protocol parameter)

New `r2_pipeline.normalize_input_frames(actual, forecast, tz=C.TZ)`, called at the very top of
`run_origin_from_frames` **before** any grid construction:

- both `actual` and `forecast` indexes are converted with `DatetimeIndex.tz_convert(C.TZ)`
  (**instant-preserving**); a New_York input is unchanged (`normalized = false`), a UTC input with
  the same instants now yields the identical index and the identical UTC grid;
- normalized **copies** are returned — the caller's frames are never mutated;
- a **naive** `DatetimeIndex` is refused with an explicit error (instants would be ambiguous);
- duplicate timestamps and unsorted indexes are refused with explicit errors (unchanged strictness);
- missing rows are preserved: nothing is dropped, filled or reordered here; absent hours still
  become NaN only through the existing grid reindex, so real support gaps stay real;
- `actual`/`forecast` instant equality is asserted after normalization.

Provenance is recorded per origin: the payload gains a top-level
`input_normalization` object (`target_tz`, `normalized`, `actual_input_tz`, `forecast_input_tz`,
row counts, first/last instant). `daily.csv.gz` and every scientific endpoint are unchanged.

Index contracts inspected before changing (no contract was altered): production
`r2_run.load_raw` builds a New_York index; `r2_pointcorr.build_features` already refuses naive
indexes; guard P2 and the reviewer capture may hold the same instants in UTC.

## 3. Superseded release and required re-verification

Any unit produced by release `R2_candidate_20261011_1108` carries the superseded fingerprint; no
formal unit exists yet, so nothing scientific is invalidated. The new fingerprint (code + frozen
protocol + amendments A/B/C and their sidecars) is recomputed at run time and bound into every
manifest and payload.

```bash
# full guard suite (all 15 cases) on the new immutable release — root-owned
cd <analysis_r2>/scripts
taskset -c 0-7,9-31 nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /usr/bin/python3 r2_guards.py \
    --checkout /home/icdm/card10_external/online_conformal \
    --out <analysis_r2>/outputs/guards_v4/r2_guards_v4_native.json --label native_full_v4_fixC
```

Formal origin and aggregate commands are unchanged from amendment B §5.
