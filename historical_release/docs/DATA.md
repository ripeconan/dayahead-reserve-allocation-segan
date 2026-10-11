# Data sources

The experiments use public load, forecast, and weather products from NYISO,
GEFCom2014, CAISO, and NOAA GEFS. Third-party files are not redistributed in
this repository.

Exact source URLs, archive templates, periods, byte counts, and verified hashes
are recorded in `data/acquisition_manifest.json`. Download scripts fail when a
static source does not match its frozen identity.

```powershell
python -m pip install -r environment/requirements-data.txt
python scripts/acquire_nyiso.py --start-year 2017 --end-year 2025
python scripts/acquire_static_data.py --dataset gefcom2014
python scripts/prepare_gefcom2014.py --root data/downloads/gefcom2014/Load
python scripts/acquire_static_data.py --dataset caiso
python scripts/verify_data.py --require-all
```

NOAA GEFS is a large object-store acquisition. Its public acquisition contract
and downloader are under `pipelines/energy_covariates/gefs/`; the manifest binds
59,712 retrieved cells, the eight structurally missing cells on 2020-09-24, the
cell ledger, and the raw-payload tree hash.

## Expected local layout

```text
data/
  raw/
    nyiso/
    gefcom2014/
    caiso/
    gefs/
  processed/
```

Raw and processed data directories are ignored by version control. The compact
files under `artifacts/` are derived scientific results rather than copies of
the source datasets.

## Split and timing rules

- A target-day region and any operational action are frozen at the day-ahead
  cutoff.
- Feedback from delivery day `u` is eligible on decision day `t` only when
  `u <= t - 2`.
- Controller history never crosses a declared segment boundary.
- Development periods are used for model selection; evaluation periods are not
  used to select configurations.
- The known NYISO complete-case gap is preserved rather than filled across
  calendar years.

The prepared NYISO and GEFCom hashes in the acquisition manifest are content
identities for the exact inputs used by the registered pipelines. The CAISO
record is a processed benchmark and is not claimed to be a native OASIS extract.
