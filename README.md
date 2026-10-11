# Budgeted Day-Ahead Reserve Allocation

Code and numerical evaluation artifacts for **Budgeted Day-Ahead Reserve Allocation from Conformal Forecasts under Delayed Feedback**, prepared by Ran Wang for submission to *Sustainable Energy, Grids and Networks*. Publication status: prepared for submission; not accepted or published.

Version: `segan-v1.0.0-20261011`. This repository is the current SEGAN reproduction release. The historical package in `historical_release/` keeps its original bytes, anonymous notices, experimental contracts and access records. Its old manuscript claims do not replace the current end-to-end study.

## Reproduce the primary results

Python 3.10 or newer is required. Install `requirements-r2.txt` in an isolated environment. Auditing supplied derived artifacts needs no GPU or original provider download:

```sh
python -m pip install -r requirements-r2.txt
python verify_publication_manifest.py
mkdir audit_replay
python study/retarget/review_2026-10-11/reviewer_r2_artifact_audit.py --out study/retarget/analysis_r2/outputs/formal_r2 --report audit_replay/artifact.json
python study/retarget/review_2026-10-11/reviewer_r2_summary_audit_v2.py --out study/retarget/analysis_r2/outputs/formal_r2 --report audit_replay/summary.json
python study/retarget/analysis_r1/scripts/r1_stage0.py --root legacy_inputs --out audit_replay/r1_stage0
```

The independent primary auditors reconstruct seven annual origins, all 1,764 reserve-response/price cells and 2,035 paired effects and interval summaries. Original outputs remain immutable; write new audit results to `audit_replay/`. See [REPRODUCTION.md](REPRODUCTION.md) for publication-asset generation, complete reruns, source-data acquisition, historical replay, native environment and external dependencies.

## What the evidence establishes

The primary study uses already exposed 2019-2025 NYISO data, a prior-year reference, day-ahead issuance and two-calendar-day feedback eligibility. It is an exposed historical reanalysis. No 2026 delivery outcomes are included. The frozen production fingerprint is `06d84da18dcd5355834f4844a104f2ee8936177e381d8f7894e75fa22c6ea322`.

At the declared finite-capacity multiplier, production allocation reduces the mean annual uncovered energy by 25.9% and the mean annual worst-seven-day deficit by 14.8% relative to no extra capacity. These percentages are ratios of macro means. About 79.9% of saved energy comes from 2025; excluding that origin gives reductions of 7.82% and 2.81%. The results describe protection in high-deficit events with additional capacity, rather than uniform annual improvements or observed market savings. The closest banking comparator has only a small modeled-service gap; bank pacing provides stronger physical tail protection with more capacity. Numerical concentration checks are in `publication_checks/`.

## Contents and data access

- `study/retarget/analysis_r2/`: immutable primary protocol, implementation and derived results.
- `study/retarget/analysis_r1/`: secondary fixed-trace allocation and oracle analyses.
- `study/retarget/review_2026-10-11/`: independent numerical auditors and historical provenance.
- `study/retarget/manuscript_segan/`: publication table and figure generators/assets.
- `historical_release/` and `legacy_inputs/`: preserved historical release and fixed replay inputs.
- `PUBLICATION_MANIFEST.json`: SHA256 identity of published files; verify with the supplied script.

Original provider archives, full legacy model weights and training dependencies are acquired separately. Included derived records contain aligned forecast, actual and residual values used in evaluation; they are not a copy of the original provider archive. A checksum identifies an external artifact and is not a download link. Provider acquisition details and terms are in `historical_release/data/acquisition_manifest.json` and `historical_release/THIRD_PARTY.md`.

## License and citation

New project code is BSD-3-Clause; preserved historical and third-party notices remain in their original locations. Upstream SAOCP is obtained separately at the pinned commit and retains its Apache-2.0 license. Source datasets remain subject to provider terms. Cite this fixed repository version using [CITATION.cff](CITATION.cff).
