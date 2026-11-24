# Preprocessing V3 / B2 / B3A on the train branch

Read [REVIEW_PREPROCESSING.md](REVIEW_PREPROCESSING.md) first for the Vietnamese scope, evidence and review questions. This update contains data-processing changes only; compare with train base eacbfdc. MERIT architecture/loss, frozen BatchNorm support and Native512 baseline are preserved.

---

# MERIT-UNETR++ for COVID-19 CT segmentation

Binary infection segmentation on chest CT: `0 = background`, `1 = lesion`.
The repository provides a shared geometry-preserving data pipeline, a standalone
UNETR++ baseline, and a paper-derived MERIT-UNETR++ implementation with training,
checkpoint/resume, full-volume inference and evaluation.

Software integration has been tested on synthetic data and selected real CT
inputs. There is no validation-selected COVID model, completed five-fold experiment or
external-test performance claim. See [validation and limitations](docs/validation.md).

## Handoff: read this first

The primary training path now uses **native 512x512 with all-region activation
checkpointing in the production trainer**. Use `configs/merit_native512.yaml`;
it is also the default for the `train` and `cv` commands. `size_hw: null` forbids
XY resizing; a source/model grid mismatch fails before training.

Handoff status: the identified runtime repairs, bounded single-case GPU pilots
and trainer integration are implemented. The team member's full training and
validation results are still awaited. The older 128-grid configs remain explicit
research alternatives with the unresolved resize/auxiliary-label issue; they
are not used by the primary native path. No completed full-cohort run or
segmentation-quality result is claimed.

Read in this order:

1. This README: current status, runnable entry points and data requirements.
2. [Architecture](docs/architecture.md): source folders and how they connect.
3. [Data pipeline](docs/data_pipeline.md): what a batch contains and which
   transformations are explicitly configured. [Dataset reports](docs/data_pipeline.md#dataset-reports-and-csv-reference)
   explains the CSV fields and saved split.
4. [MERIT model](docs/architecture.md#merit-unetr-model): TEP, MER, PCG, supervision and paper correspondence.
5. [MERIT pipeline](docs/merit_pipeline.md): production training/resume/inference commands.
6. [Validation](docs/validation.md): measured results and limits.
7. [Kaggle factorized training](docs/merit_pipeline.md#kaggle-fold-0-native-512-with-factorized-mer): current Kaggle package and resume workflow.

### Completed work and current boundary

| Area | Implemented and checked | Where to look |
| --- | --- | --- |
| Shared data access | Lazy CT/GT loading, geometry checks, slab sampling, saved five-fold selection and native reconstruction | `src/merit_covid/data/pipeline/` |
| Models | Standalone UNETR++ and paper-derived MERIT with TEP/MER/PCG; core model files are unchanged by the latest runtime review | `src/merit_covid/models/` |
| Runtime and memory | CT/GT source binding on resume, strict thresholds/finite logits, volume-block shuffle, integrated all-region recomputation | `src/merit_covid/engine/`, `data/pipeline/dataloader.py` under the package |
| Diagnostics | Split/input/resize audits, isolated optimizer profiles, restored smoke and baseline input scripts | `scripts/merit_preflight.py`, `scripts/model_smoke.py`, `scripts/unetrpp_real_input.py` |
| Primary native-512 input | No XY resize, native intensity, same model/loss/split; production recomputation follows the two-step T4 pilot | `configs/merit_native512.yaml`, `src/merit_covid/engine/activation_checkpointing.py` |
| One measured improvement candidate | MER-only native-512 T4 pilot PASS; retain the earlier all-region overlay as reference because memory nearly doubles for an inconclusive timing gain | [MER attention execution evidence](docs/validation.md#mer-attention-execution-change) |

The latest full CPU suite reports **397 passed, 2 CUDA-only skipped**. The compact
production-checkpointing CUDA AMP test passed separately in the local GPU
environment. These checks
do not establish model accuracy or complete full-cohort training.

## Measured cloud results

These are reports supplied by the user after running the notebooks on Colab,
not cloud jobs launched from this checkout. The selected JSON reports, runtime
versions and original ZIP checksums are preserved in
[portable evidence](analysis/merit_native512/colab_evidence_20261113.json).

| Native-512 experiment on Tesla T4 | Result | Peak PyTorch allocation | Successful optimizer updates |
| --- | --- | --- | --- |
| Original execution, full allocator budget | CUDA out of memory in MER forward | 14627.58 MiB before failure | 0 |
| Checkpointing overlay, synthetic input | PASS | 4650.55 MiB | 2 |
| Same overlay, real A-0025 CT/GT | PASS | 4649.68 MiB | 2 |
| MER-only checkpointing, same real A-0025 | PASS | 9219.69 MiB | 2 |

The real steps took 13.958 s and 13.731 s. They preserved source bytes, native
pixels and `[1,1,11,512,512]` input, produced finite gradients and changed learned
parameters. One CT-only master prediction also passed; full-volume prediction
and segmentation quality were not evaluated in this pilot. Two steps are too
few to establish training throughput or a reliable speed comparison.

The MER-only report from another account used the same GPU model, Torch version,
input/config and initial parameters. Its mean step time was 13.625 s versus
13.845 s for the earlier overlay (observed 1.59% lower), with identical losses
and final parameter hashes. Two steps in different sessions cannot establish
a speedup. Keep the earlier all-region overlay as the recommended pilot mode:
4.54 GiB peak allocation versus 9.00 GiB for MER-only. Production training now
enables the all-region overlay in the native-512 config; the MER-only
candidate stays optional. See the completed experiment note for details.

## Quick start

An interactive local CT demo for the downloaded epoch-0 checkpoint is available
with `python -m streamlit run scripts/demo_merit.py`. See the
[demo setup and limits](docs/merit_pipeline.md#interactive-local-epoch-0-demo).

### Current cloud workflows

Use the [Colab attention pilot](notebooks/colab_mer_attention_pilot.ipynb) for
bounded tests on selected real cases. For the current Kaggle training and resume
procedure, follow the [factorized Kaggle guide](docs/merit_pipeline.md#kaggle-fold-0-native-512-with-factorized-mer).
The pilot does not establish segmentation quality or complete a training epoch.

### Train from the current source

Install the local environment below and make the development data available.
Then run from the repository root on a CUDA machine:

```powershell
py scripts/merit_pipeline.py train --config configs/merit_native512.yaml --fold 0 --output outputs/native512/fold0
# After an interruption, repeat with the same data/config and saved checkpoint.
py scripts/merit_pipeline.py train --config configs/merit_native512.yaml --fold 0 --output outputs/native512/fold0 --resume outputs/native512/fold0/last.pt
```

`train` and `cv` select this config when `--config` is omitted. It explicitly
assumes mm for the three source cases with unknown units; this does not verify
their original units. The pipeline does not silently drop these cases.

Checkpoints are saved after each completed epoch and validation, not midway
through an epoch. Use persistent storage for outputs and allow for long native
epochs. The production path has compact train/validation/resume regression
coverage; the complete 512-grid cohort run is for the receiving team to measure.

### Local setup

For other cloud modes and Kaggle setup, see the
[cloud guide](docs/merit_pipeline.md#colab-and-kaggle-quick-start).

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
The shared reader rejects unknown units by default. The MERIT index, native 512 and Colab
configs explicitly set `assume_mm: true`; this documents an assumption and
does not verify or repair the source.

Zenodo20 is an external dataset, not development data. Do not use it to select
preprocessing, thresholds, hyperparameters or checkpoints.

## Start with MERIT

With the development data already available, use the production CLI described
below. This is a separate path from the bounded notebook pilot.

Before full training, run the train-only intensity/resize audit, full-optimizer
GPU profiles and bounded real-CT pilot in the
[preflight workflow](docs/merit_pipeline.md#pre-training-preflight). A diagnostic
pilot checkpoint is not a trained or validation-selected research model.

```powershell
py scripts/merit_pipeline.py check-split --config configs/merit_native512.yaml
py scripts/merit_pipeline.py audit --config configs/merit_native512.yaml --limit 1 --output outputs/first_audit.json
```

Choose and review the input policy before issuing the guide's `train` command:

| Configuration | Input policy | Limitation |
| --- | --- | --- |
| `merit_index.yaml`, `merit_physical.yaml` | Explicit 128x128 XY resize; index or physical sampling | Small lesions may disappear and derived auxiliary labels may change; this design remains unresolved |
| `merit_native512.yaml` | Native XY (`size_hw: null`), 512x512 model grid, batch 1, AMP and all-region activation checkpointing | Primary path; other native grids fail explicitly; full-cohort results pending |

Both grids use feature size 16, depths 3/3/3/3 and TEP/MER/PCG. Learned projections
and positions depend on the input grid: parameter counts are 29,836,242 at 128
and 174,043,602 at 512. Their checkpoints are not directly shape-compatible.

The native 512 config explicitly assumes mm for the three unknown-unit cases.
Verify the source units when possible and document this assumption. See the
[MERIT pipeline guide](docs/merit_pipeline.md) for resume, validation, CV, physical
sampling, GPU checks and the explicitly gated external-test command.

The shared reader defaults to native XY size and NIfTI-scaled intensities.
The older MERIT index/physical configs explicitly resize XY to `128x128`; no
listed MERIT config interpolates Z. The native-512 pilot verifies that no XY
interpolation, intensity normalization or source-file modification occurs.
Training supervises true full-slab GT. Inference predicts each master once,
restores probabilities to native geometry and then evaluates whole volumes.

### Remaining limits and follow-up

- Retain the measured all-region overlay as reference. The MER-only comparison
  is complete for this two-step pilot; further timing measurements are optional.
- The production trainer now enables the selected overlay before optimizer
  construction and restores it from frozen checkpoint config. Native-512 T4
  evidence still comes from bounded pilots; complete training is not yet measured.
- Add persistent progress and mid-epoch resume for long cloud sessions. The
  existing production trainer resumes at epoch boundaries; trial notebooks do
  not export a resumable trained checkpoint.
- Check multiple TRAIN cases before a full fold. Full-cohort/five-fold training,
  validation-selected weights and external evaluation remain unrun.
- Resolve the three unknown-unit cases explicitly and verify patient identity
  if patient-level independence is required. No assumption is silently enabled.

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
| `tools/merit_main_trial.py` | Bounded integrated Trainer diagnostic with complete real source binding and temporary checkpoint verification |
| `scripts/model_smoke.py` | Synthetic model forward/backward without optimizer updates |
| `scripts/unetrpp_real_input.py` | One real native CT patch through the baseline, without training |
| `tests/` | Synthetic regression and integration checks |
| `ct_pipeline/` | Required legacy import aliases; not another processing implementation |
| `docs/` | Topic-oriented reference guides; no duplicate per-folder READMEs |

Data, archives, reference clones, caches and trained artifacts remain local.
The cleanup does not remove them. Do not treat stored CSV/JSON audits as model
input or as segmentation performance.

## Documentation

| Topic | Guide |
| --- | --- |
| Project architecture, MERIT and UNETR++ models | [Architecture and models](docs/architecture.md) |
| Data, saved reports, input and output contracts | [Data pipeline and reports](docs/data_pipeline.md) |
| Training, Kaggle, resume, inference and evaluation | [Training pipeline](docs/merit_pipeline.md) |
| Experiments, performance reviews and limitations | [Validation and research evidence](docs/validation.md) |

[CHANGELOG.md](CHANGELOG.md) records major integration milestones.
Upstream UNETR++ attribution remains in its packaged `LICENSE`, `NOTICE` and
`upstream.json`.
