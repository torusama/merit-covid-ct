# Architecture

There is one authoritative processing implementation:
`src/merit_covid/data/pipeline/`. Tensor loaders, older NumPy interfaces and the
baseline patch reader reuse it. Compatibility files are intentional, not duplicate
algorithms.

## Source layout

```text
src/merit_covid/
  config.py, cli.py, utils.py       Legacy/main config, package CLI, utilities
  data/
    manifest.py                    Legacy/versioned manifests and tool helpers
    nifti_io.py, preprocessing.py   Volume/PreparedVolume compatibility APIs
    slab.py, dataset.py            NumPy slab API over the shared core
    splitting.py                   Compatibility view of the canonical split
    pipeline/
      cases.py, config.py          Identity, paths, preprocessing options
      geometry.py                  Header validation, LPS transforms, restoration
      preprocessing.py             Scaling, explicit intensity/XY policies, QC
      slab.py                      Index/mm sampling and true targets/validity
      dataset.py, dataloader.py    Header index, one-volume cache, collate/loaders
      splits.py                    Immutable grouped split and fingerprint checks
      reporting.py, metrics.py    Descriptive audits and native-volume metrics
      preflight.py                Bounded-memory TRAIN-only intensity/resize QC
    unetrpp_input/
      config.py, tiling.py         Native patch windows, padding and coordinates
      dataset.py                   Shared-reader patch Dataset and loader
  models/
    base.py                        Torch-independent master-logit contract
    unetrpp/                       Standalone Synapse-derived baseline
    merit_unetrpp/
      modules.py                   TEP, MER, PCG, index/physical position handling
      model.py                     Z-preserving backbone and insertion points
      loss.py, adapter.py          Full-slab loss / binary master-logit adapter
  engine/
    training.py                    Frozen config, trainer, checkpoint, evaluation
    preflight.py                   Full-step resource profile and bounded CT pilot
    synthetic.py                   Temporary synthetic NIfTI smoke fixtures
    slab_inference.py              MERIT masters -> complete native volume
    patch_inference.py             Standalone baseline patch inference
    patch_reconstruction.py        Coordinate-based probability averaging
    prediction.py                  Shared VolumePrediction contract
    reconstruction.py, metrics.py Compatibility helpers over shared logic
ct_pipeline/                       Legacy import aliases only
configs/                           Explicit experiment/model configurations
scripts/                           Experiment, split, dataset and smoke CLIs
tools/                             Downloads, audit/report, viewer, source checks
scripts/model_smoke.py              Synthetic baseline/MERIT forward and backward
scripts/unetrpp_real_input.py       One real CT patch through the baseline
splits/                            Canonical saved five-fold partitions
analysis/                          Stored descriptive evidence; not training input
tests/                             Shared core, adapters and integrated regression
```

## Execution paths

```text
NIfTI CT (+ mask only for labeled operations)
  -> validate headers/units -> LPS array orientation
  -> NIfTI scaling -> explicit intensity and optional XY processing
  -> shared native volume
       MERIT: slab sampling -> TEP/MER/PCG
         train: full-slab GT + validity -> loss -> optimizer -> native validation
         infer: master fg-bg -> sigmoid -> native XY restoration
       Baseline: native 3D crop/padding -> UNETR++ -> sigmoid
         infer: discard padding -> mean overlapping probabilities
  -> complete native-LPS VolumePrediction
  -> original XYZ/orientation -> threshold -> shared full-volume metrics
```

NIfTI and `PreparedVolume` use XYZ. Network tensors use `[B,C,Z,H,W]`,
where H=Y and W=X. LPS describes array orientation; NiBabel's affine still maps
to RAS+ world coordinates. No implicit Z interpolation, deobliquing or GT-based
input selection is performed.

## Interfaces and dependency direction

- `data.pipeline` owns preprocessing, geometry and metric conventions.
- `CTSlabDataset` provides Torch tensors and true slab GT; the main `SlabDataset`
  preserves its older NumPy/master-target interface.
- `ct_pipeline` forwards historical imports. Existing tests/notebooks use it;
  deleting it would break compatibility without simplifying the core.
- Core NumPy reading/sampling remains usable without Torch. Tensor loaders,
  explicit XY resize and neural models require their corresponding dependencies.
- Model adapters return logits, not metrics. Training calls the MERIT model
  directly so auxiliary outputs are retained.
- Inference builds `VolumePrediction` on the complete native LPS grid.
  Evaluation restores original geometry and uses original GT.
- The baseline patch API deliberately rejects MERIT's four-tuple/two-class
  interface. Use `predict_slab_case` for MERIT.

No Python package was moved during documentation cleanup. This preserves import
paths, command-line entry points, saved configs and checkpoint loading.

## Config ownership

| Config | Consumer | Scope |
| --- | --- | --- |
| `experiment_baseline.yaml` | `merit_covid.config.load_config` | Legacy/main data API |
| `unetrpp_baseline.yaml` | Baseline examples | Model-only standalone UNETR++ |
| `merit_unetrpp.yaml` | MERIT model smoke example | Model/loss only; not a training experiment |
| `merit_index.yaml` | `scripts/merit_pipeline.py` | Full E1 index experiment |
| `merit_physical.yaml` | `scripts/merit_pipeline.py` | E2 physical-sampling template; spacing required |

These configs are not interchangeable. Model-only YAML files are retained because
examples and tests use them.

## Documentation ownership

Each topic has one maintained guide: data contracts in [data_pipeline.md](data_pipeline.md),
stored statistics in [dataset_reports.md](dataset_reports.md), model equations in
[merit_unetrpp.md](merit_unetrpp.md), execution in [merit_pipeline.md](merit_pipeline.md),
baseline architecture/input in [unetrpp_baseline.md](unetrpp_baseline.md) and
[unetrpp_input.md](unetrpp_input.md), and evidence/limitations in
[validation.md](validation.md). Historical implementation/merge/repair notes are
consolidated into validation and the root changelog rather than kept as competing
descriptions of the current system.
