# MERIT-UNETR++ for COVID-19 CT segmentation

Binary infection segmentation on chest CT: `0 = background`, `1 = lesion`.
The repository provides a shared geometry-preserving data pipeline, a standalone
UNETR++ baseline, and a paper-derived MERIT-UNETR++ implementation with training,
checkpoint/resume, full-volume inference and evaluation.

Software integration has been tested on synthetic data and selected real CT
inputs. There is no validation-selected COVID model, completed five-fold experiment or
external-test performance claim. See [validation and limitations](docs/validation.md).

## Quick start

Run commands from the repository root. Select a PyTorch build compatible with
your CPU/GPU before installing the extras.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[unetrpp,test,tools]"

# These checks do not need the real datasets.
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/merit_pipeline.py smoke --device cpu --output outputs/first_smoke.json
```

The `unetrpp` extra includes Torch, MONAI and einops; `tools` adds plotting,
download and viewer dependencies. Core NumPy I/O does not require Torch.
Use a new report filename when rerunning commands that refuse overwrites.

The shorter commands below and in the guides use `py` as interpreter shorthand.
Replace it with `.\.venv\Scripts\python.exe` when using the environment above.

## Data and protocol

```text
data/covid19_20/
  cases.json                         CT/mask paths and volume IDs
  Train/*_ct.nii.gz                   Original CT volumes
  Train/*_seg.nii.gz                  Original binary lesion masks
splits/covid19_20_5fold.json           Saved development partitions
analysis/covid19_20/                  Descriptive audit, CSV and figures
outputs/                             Local run outputs and diagnostic reports
```

COVID-19-20 contains 199 development volumes. The saved five-fold split has
train/validation/test counts `127/32/40` in folds 0-3 and `128/32/39` in fold 4.
Read the existing split; do not regenerate it before training.

Patient identity is not independently verified: current grouping falls back to
volume IDs. Three cases have unknown spatial units: A-0161, A-0585 and A-0599.
The default is to reject unknown units. Use `--assume-mm` only after accepting
an evidence-backed mm assumption; it does not verify or repair the source.

Zenodo20 is an external dataset, not development data. Do not use it to select
preprocessing, thresholds, hyperparameters or checkpoints.

## Start with MERIT

With the development data already available:

Before full training, run the train-only intensity/resize audit, full-optimizer
GPU profiles and bounded real-CT pilot in the
[preflight workflow](docs/merit_pipeline.md#pre-training-preflight). A diagnostic
pilot checkpoint is not a trained or validation-selected research model.

```powershell
py scripts/merit_pipeline.py check-split --config configs/merit_index.yaml
py scripts/merit_pipeline.py audit --config configs/merit_index.yaml --limit 1 --output outputs/first_audit.json
py scripts/merit_pipeline.py train --config configs/merit_index.yaml --fold 0 --output outputs/e1/fold0
```

Full-cohort training will reject the three unknown-unit cases until their units
are verified or the explicit mm assumption is accepted. See the
[MERIT pipeline guide](docs/merit_pipeline.md) for resume, validation, CV, physical
sampling, GPU checks and the explicitly gated external-test command.

The shared reader defaults to native XY size and NIfTI-scaled intensities.
The MERIT experiment explicitly resizes XY to `128x128`; it never interpolates Z.
Training supervises true full-slab GT. Inference predicts each master once,
restores probabilities to native geometry and then evaluates whole volumes.

## Repository map

| Location | Responsibility |
| --- | --- |
| `src/merit_covid/data/pipeline/` | Shared geometry, preprocessing, sampling, Dataset, splits and metrics |
| `src/merit_covid/models/merit_unetrpp/` | Z-preserving UNETR++ with TEP, MER, PCG, loss and master adapter |
| `src/merit_covid/models/unetrpp/` | Standalone upstream-derived 3D baseline and license/provenance |
| `src/merit_covid/data/unetrpp_input/` | Native 3D crops, tiling, padding and validity for the baseline |
| `src/merit_covid/engine/` | Training, inference, reconstruction and unified prediction output |
| `configs/` | Experiment configs and model-only smoke configs |
| `scripts/` | Experiment and package entry points |
| `tools/` | Acquisition, data audit/reporting, viewer and upstream verification |
| `examples/` | Small, explicit usage examples |
| `tests/` | Synthetic regression and integration checks |
| `ct_pipeline/` | Required legacy import aliases; not another processing implementation |
| `docs/` | Topic-oriented reference guides; no duplicate per-folder READMEs |

Data, archives, reference clones, caches and trained artifacts remain local.
The cleanup does not remove them. Do not treat stored CSV/JSON audits as model
input or as segmentation performance.

## Documentation

| Read when you need to... | Guide |
| --- | --- |
| Locate code and understand dependencies | [Architecture](docs/architecture.md) |
| Read batches, geometry, intensity, split and metric contracts | [Data pipeline](docs/data_pipeline.md) |
| Understand stored audits and all 71 primary CSV fields | [Dataset reports](docs/dataset_reports.md) |
| Inspect MERIT equations, insertion points and implementation choices | [MERIT model](docs/merit_unetrpp.md) |
| Train, resume, infer and evaluate MERIT | [MERIT pipeline](docs/merit_pipeline.md) |
| Inspect the upstream-derived standalone baseline | [UNETR++ baseline](docs/unetrpp_baseline.md) |
| Prepare baseline patches and reconstruct native volumes | [UNETR++ input/output](docs/unetrpp_input.md) |
| Check measured evidence and remaining research work | [Validation](docs/validation.md) |

[CHANGELOG.md](CHANGELOG.md) records major integration milestones.
Upstream UNETR++ attribution remains in its packaged `LICENSE`, `NOTICE` and
`upstream.json`.
