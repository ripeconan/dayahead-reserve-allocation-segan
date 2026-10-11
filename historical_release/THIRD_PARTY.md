# Third-party software

The BSD 3-Clause License in `LICENSE` applies to the original code and
documentation in this package. It does not replace upstream software licenses
or grant redistribution rights for third-party datasets.

This package calls, but does not vendor, the following third-party software:

- GluonTS 0.16.3 for DeepAR.
- NeuralForecast 1.7.7 for N-HiTS and PatchTST.
- PyTorch Forecasting 1.8.0 for the Temporal Fusion Transformer.
- ProLoaF at commit `e4a85c9dda725e7a5063899e805fa22a2f5e1129`.
- LightGBM and CatBoost for the corresponding classical baselines.
- The public SAOCP implementation at commit
  `95790a9b9e4061166e6d7a8a13282e019b1a8663` for the common conformal stage
  (Apache License 2.0 upstream).

These projects remain subject to their upstream licenses. The repository
contains a local source-verification and delayed-feedback wrapper, but no copy
of the upstream SAOCP source files.

Third-party datasets are not redistributed. Provider terms apply to NYISO and
GEFCom2014; the CAISO Zenodo record is CC BY 4.0; NOAA data are public-domain
U.S. Government information with attribution and no-endorsement guidance. Exact
source records are in `data/acquisition_manifest.json`.
