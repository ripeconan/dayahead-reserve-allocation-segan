# Model retraining and end-to-end workflow

The recommended first step is the artifact replay described in `README.md`.
It verifies the published numerical objects without requiring model retraining.
Run `scripts/reproduce_robustness.py` as part of that tier to check the frozen
robustness tables and replay the production controller on the retained transfer
and sensitivity traces.

The complete experimental workflow has four stages:

1. Acquire the public datasets and verify the acquisition manifest.
2. Recreate point forecasts and native uncertainty outputs for each registered
   model configuration.
3. Apply the common multivariate conformal shell and the strict-lag-2
   reliability controller.
4. Recompute paired, time-aware inference and compare the outputs with the
   frozen artifact manifest.

The anonymous package contains the registered training and evaluation code for
stage 2, the production common-conformal wrapper for stage 3, and exact artifact
replay for stages 3--4. Separate model environments and the size of the source
datasets mean that a raw-data-to-paper rerun is intentionally staged rather
than exposed as one command.

The deep-model stage requires separate environments because the supported
versions of NeuralForecast, PyTorch Forecasting, and ProLoaF do not share one
stable dependency stack. Install the relevant specification from `environment/`
and expose the package directory before running a pipeline:

```powershell
$env:PYTHONPATH = "pipelines"
```

### Classical baselines

```powershell
python -m pip install -r environment/requirements-classical.txt
python -m card10_energy_baselines.r78.run_m0_m4 `
  --data data/raw/nyiso/nyiso_zonal_2017_2025.csv.gz `
  --output-dir runs/classical
```

### DeepAR

```powershell
python -m pip install -r environment/requirements-deepar.txt
python -m card10_energy_baselines.r85.r85_deepar_fair self-test
python -m card10_energy_baselines.r85.r85_deepar_fair develop `
  --nyiso data/raw/nyiso/nyiso_zonal_2017_2025.csv.gz `
  --grid pipelines/card10_energy_baselines/r85/contracts/deepar_candidate_v1.json `
  --out-dir runs/deepar/development
python -m card10_energy_baselines.r85.r85_deepar_fair predict `
  --nyiso data/raw/nyiso/nyiso_zonal_2017_2025.csv.gz `
  --grid pipelines/card10_energy_baselines/r85/contracts/deepar_candidate_v1.json `
  --selected pipelines/card10_energy_baselines/r85/contracts/deepar_selected_v1.json `
  --out-dir runs/deepar/evaluation
```

### N-HiTS, PatchTST, and TFT

Use `requirements-neuralforecast.txt` for N-HiTS and PatchTST, and
`requirements-tft.txt` for TFT. Run one model and test year per process. For
example:

```powershell
python -m card10_energy_baselines.r86.r86_evaluate_selected `
  --data data/raw/nyiso/nyiso_zonal_2017_2025.csv.gz `
  --grid pipelines/card10_energy_baselines/r86/contracts/fair_deep_candidate_grid_v1.json `
  --selected pipelines/card10_energy_baselines/r86/selected/nhits.json `
  --model NHiTS --test-year 2019 `
  --output-dir runs/fair_deep/NHiTS/2019
```

For ProLoaF, check out the commit listed in `THIRD_PARTY.md` and set
`PROLOAF_REPO` to that checkout before running the corresponding selected
configuration.

The common conformal and controller outputs supplied with the package are
independently replayed from retained daily traces. Model retraining is not
required for artifact replay.

## Robustness analyses

The compact objects under `artifacts/robustness/` record the analytical
construction-floor checks, exposed cross-dataset transfer, partial action
effectiveness, action-cost break-even points, alternative inference contracts,
and the predeclared controller grid. The standard-library replay validates all
table dimensions and reported anchors, reconstructs every exposed transfer
action, and recomputes the production hyperparameter reference for all seven
origins. The configurable controller in `src/energy_uq/reliability_shield.py`
can be used to rerun other memory, threshold, budget, lag, and effectiveness
settings.

## Common conformal and operational stage

Create a clean checkout of the pinned upstream SAOCP repository. Line-ending
conversion must be disabled because the wrapper validates the upstream source
bytes before compiling the required AST nodes.

```powershell
git -c core.autocrlf=false clone https://github.com/salesforce/online_conformal.git external/online_conformal
git -C external/online_conformal -c core.autocrlf=false checkout --detach 95790a9b9e4061166e6d7a8a13282e019b1a8663
python -m pip install -r environment/requirements-common.txt
python scripts/run_common_conformal.py `
  --predictions runs/classical/point_predictions.csv.gz `
  --saocp-repo external/online_conformal `
  --output-dir runs/common
```

The wrapper verifies commit, tree, clean-worktree state, upstream source-file
hashes, and eight AST-node hashes. It fits the common eleven-dimensional OAS
geometry on the legal prior origin, runs one SAOCP stream per New York local
delivery hour, applies feedback at exact lag 2, preserves DST day boundaries,
and emits hourly, daily, policy, metric, and identity tables.

## Determinism

All stochastic stages must receive an explicit seed. Thread counts and numeric
library versions must be recorded with each run. A rerun is accepted when the
declared scientific tables match the frozen reference under their documented
comparison rule; runtime and memory measurements are reported separately and
are not expected to be byte-identical.

## External implementations

The common conformal stage loads a pinned public implementation of strongly
adaptive online conformal prediction. Exact upstream revisions are listed in
`THIRD_PARTY.md`; upstream source is not vendored. Model-specific repositories
retain their original licenses and attribution.
