# Scientific reproduction archive

The primary study is the seven-origin 2019–2025 end-to-end NYISO reanalysis, fingerprint 06d84da18dcd5355834f4844a104f2ee8936177e381d8f7894e75fa22c6ea322. It uses already exposed data. Protocol and amendments retain their dated, pre-execution status statements; the completed outputs and independent audit reports establish subsequent execution. No 2026 delivery outcome is used. No GPU is required. All files in study/retarget/analysis_r2 are byte-preserved from the admitted release/results.

From this archive root, using Python 3.10 and requirements-r2.txt:

```sh
mkdir audit_replay
python study/retarget/review_2026-10-11/reviewer_r2_artifact_audit.py --out study/retarget/analysis_r2/outputs/formal_r2 --report audit_replay/artifact.json
python study/retarget/review_2026-10-11/reviewer_r2_summary_audit_v2.py --out study/retarget/analysis_r2/outputs/formal_r2 --report audit_replay/summary.json
python study/retarget/manuscript_segan/r2_assets/make_r2_assets.py
python study/retarget/analysis_r1/scripts/r1_stage0.py --root legacy_inputs --out audit_replay/r1_stage0
```

The first two auditors independently reconstruct hourly geometry/physics, all 1,764 grid cells, modeled endpoints, and 2,035 paired effects/bootstrap intervals/leave-one-origin-out calculations. New audit reports belong outside the immutable original result directories. The asset generator regenerates publication tables and figures and records input, generator, and output identities. PDF container metadata may vary across Matplotlib environments; numerical values are authoritative.

Full primary-study rerun additionally requires the original NYISO prepared archive and a clean Git checkout of https://github.com/salesforce/online_conformal.git at commit 95790a9b9e4061166e6d7a8a13282e019b1a8663. The verified-AST wrapper identifies every required source blob and rejects a dirty or incorrect checkout. Native Python 3.10.12, NumPy 2.2.6, pandas 2.2.3, SciPy 1.15.3, sklearn 1.7.2 were used. The prepared NYISO CSV.GZ is not included: SHA256 d9e492532e22bc3f41c6967acf99cc5b49a49ae2e5ba3ae3b7dbc8092d9e5d0d, 7,093,132 bytes, 78,144 rows. Provider forecast/actual archives and preparation provenance are documented in historical_release/data/acquisition_manifest.json; historical_release/scripts/acquire_nyiso.py prepares the 2017–2025 archive. A SHA binding identifies the required prepared object; it is not a public download endpoint. Supplied derived evaluation rows include aligned forecast, actual, and residual records used for auditing. Original provider archives and model weights/training dependencies are acquired separately.

```sh
python study/retarget/analysis_r2/scripts/r2_guards.py --help
python study/retarget/analysis_r2/scripts/r2_run.py --raw /path/to/nyiso_zonal_2017_2025.csv.gz --checkout /path/to/online_conformal --out fresh_r2_run --all
python study/retarget/analysis_r2/scripts/r2_run.py --raw /path/to/nyiso_zonal_2017_2025.csv.gz --checkout /path/to/online_conformal --out fresh_r2_run --aggregate
python study/retarget/review_2026-10-11/reviewer_r2_point_audit.py --raw /path/to/nyiso_zonal_2017_2025.csv.gz --out study/retarget/analysis_r2/outputs/formal_r2 --report audit_replay/point.json
```

Always use a fresh output directory for full reruns: admission/resume deliberately binds code, data, source, environment, and artifacts. Differences in numeric stack invalidate native resume; they do not justify relabeling historical output as a new run. The server-specific physical-macro audit helper is retained as original provenance; its path allowlist is intentionally server-specific. Artifact/summary auditors above are portable and verify the same reported physical values.

historical_release is a separate 156-file frozen package for prior recovery/deep/transfer studies; its original README and scientific claims refer to those earlier pipelines, not the new end-to-end study. Its source-verified artifact-replay commands are standard-library based. Full legacy model retraining needs separately acquired inputs/dependencies. study/retarget/analysis_r1 supplies the later 30-origin policy replay, 14 exact offline-oracle solutions, and directional diagnostics; legacy_inputs contains its 14 SHA-bound replay inputs. The R-A and end-to-end studies retain separate fingerprints, calendars, and inference contracts. No universal dominance or causal decomposition of offline-oracle gaps is implied.

Retained incidents are included for chronology: the public SAOCP import failed because Merlion was absent, so the source-verified AST route was used; the earlier native guard candidate failed UTC/local outer-boundary equivalence and was never promoted. Protocol amendment C records its instant-preserving repair before formal execution. The admitted candidate passed all 15 native guards. The original summary auditor is retained alongside v2: v2 corrected only the treatment of undefined zero-action rescue precision; production outputs were unchanged.

Source verification wrappers are original project code; upstream SAOCP is Apache 2.0 and must be obtained separately. Historical project licensing and third-party notices remain in historical_release/LICENSE and THIRD_PARTY.md. This archive does not grant rights to upstream datasets or dependencies. PACKAGE_MANIFEST.json binds every delivered file except itself. No SSH/session credentials, Claude streams, or remote-control scripts are included.
