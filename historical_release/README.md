# Reproducibility package for delayed-feedback energy uncertainty quantification

This repository accompanies a double-anonymized journal submission on causal,
budgeted reliability control for day-ahead multivariate conformal forecasting.
It provides two complementary reproducibility routes:

1. **Artifact replay** verifies the frozen scientific objects, reconstructs the
   headline results, and checks the operational controller against the retained
   daily trace. This route uses only the Python standard library and finishes in
   seconds.
2. **Model retraining** starts from the public data sources and recreates the
   registered classical and deep point forecasts. See
   `docs/FULL_REPRODUCTION.md` for data and compute requirements. The common
   OAS-region, verified SAOCP, and operational-controller stages are included as
   executable code and are also checked by exact trace replay.

## Quick start

Python 3.10 or newer is required.

```bash
python scripts/build_manifest.py --verify
python scripts/reproduce_artifacts.py
python scripts/reproduce_confirmatory.py
python scripts/reproduce_robustness.py
python -m unittest discover -s tests -v
python scripts/check_internal_imports.py
python scripts/check_release.py
python scripts/build_release_manifest.py --verify
```

The headline replay writes a machine-readable summary to
`derived/reproduction_summary.json` and a compact table to
`derived/headline_results.csv`. The robustness replay verifies the frozen
theory, transfer, intervention-effectiveness, cost, inference, and sensitivity
objects. It also replays all 23 exposed transfer origins and the seven-origin
production hyperparameter reference directly from retained daily traces.

## Repository layout

```text
artifacts/       Frozen, compact scientific objects used in the paper
docs/            Data and full-rerun instructions
environment/     Dependency specifications for the two replay tiers
pipelines/       Frozen point-forecast training and evaluation pipelines
scripts/         Reproduction, integrity, and release checks
src/energy_uq/   Common conformal, verified SAOCP, and controller implementation
tests/           Unit and trace-replay tests
```

The repository does not redistribute third-party datasets. It contains no
author names, institutional identifiers, private infrastructure addresses, or
version-control history from the development workspace.

## Reproducibility scope

The artifact replay covers the unified baseline matrix, the same-base causal
controller comparison, the recovery-confirmatory inference, the fair deep
baseline evaluations, the analytical construction floor, cross-dataset
production transfer, partial action effectiveness, action-cost break-even
points, inference robustness, and controller hyperparameter sensitivity. The
retained objects include daily traces where they are needed to verify action
timing and paired inference, not just formatted paper tables.

The public controller exposes feedback lag, memory, threshold, budget, and
action effectiveness. The default action effectiveness is one, matching the
sure-cover production replay; values between zero and one reproduce the
closed-loop partial-effectiveness semantics.

The repository also contains the executable classical and deep-model pipelines
used to create the point forecasts. Directory suffixes `r78`, `r85`, and `r86`
identify frozen protocol revisions and are retained so that configurations and
artifact hashes remain traceable. They do not denote different paper versions.

The code is distributed under the BSD 3-Clause License. A permanent archive
identifier and non-anonymous citation metadata will be added after review.
