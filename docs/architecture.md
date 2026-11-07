# Main structure with the Duong processing core

Current decisions and evidence: [merge report](merge_report.md).

```text
src/merit_covid/
  config.py, cli.py, utils.py        Main configuration, scripts and seed utilities
  data/
    manifest.py                     Main legacy/versioned manifest and tool helpers
    nifti_io.py                      Main Volume/export API using the shared reader
    preprocessing.py                Main PreparedVolume API using shared transforms
    slab.py, dataset.py              Main NumPy Dataset API using shared sampling
    splitting.py                     Compatibility view of the one canonical split
    pipeline/                       Authoritative Duong data/evaluation implementation
      config.py, cases.py, geometry.py, preprocessing.py
      slab.py, dataset.py, dataloader.py, splits.py, reporting.py, metrics.py
  engine/                           Main evaluation/reconstruction API on shared logic
  models/                           Preserved future model adapter; no model/trainer
configs/                            Main YAML, defaults LPS / native size / native intensity
scripts/                            Main check/split/synthetic-smoke entry points
tools/                              Preserved downloads, audits, viewer and primary report CLI
ct_pipeline/                        Import aliases only
splits/covid19_20_5fold.json          The canonical Duong split
analysis/                           Preserved dataset reports and integration verification
tests/                              Main suite, tests/duong suite, cross-API integration tests
```

## Read in this order

1. [README](../README.md): setup, native data and the first batch.
2. [Shared config](../src/merit_covid/data/pipeline/config.py).
3. [Cases](../src/merit_covid/data/pipeline/cases.py) and
   [splits](../src/merit_covid/data/pipeline/splits.py).
4. [Geometry](../src/merit_covid/data/pipeline/geometry.py),
   [preprocessing](../src/merit_covid/data/pipeline/preprocessing.py) and
   [sampler](../src/merit_covid/data/pipeline/slab.py).
5. [Tensor Dataset](../src/merit_covid/data/pipeline/dataset.py) and
   [fold loaders](../src/merit_covid/data/pipeline/dataloader.py).
6. [Main NumPy API](../src/merit_covid/data/dataset.py), if using the existing model adapter contract.
7. [Data contract](data_contract.md) and [verification](../analysis/merge/verification.json).

Arrays in NIfTI/PreparedVolume are XYZ. Samples use [1,D,H,W], with H=Y and W=X.
Both public APIs share native intensities, LPS orientation and physical offsets.
Default NumPy reading/sampling does not require torch; Tensor/DataLoader and
explicit in-plane resize do. Inference omits masks and never reads GT.
