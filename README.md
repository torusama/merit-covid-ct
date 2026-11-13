# MERIT-UNETR++ for COVID-19 CT segmentation

Binary infection segmentation on chest CT: `0 = background`, `1 = lesion`.
The repository provides a shared geometry-preserving data pipeline, a standalone
UNETR++ baseline, and a paper-derived MERIT-UNETR++ implementation with training,
checkpoint/resume, full-volume inference and evaluation.

Software integration has been tested on synthetic data and selected real CT
inputs. There is no validation-selected COVID model, completed five-fold experiment or
external-test performance claim. See [validation and limitations](docs/validation.md).

## Handoff: read this first

The next reproducible task is **one real native-resolution CT pilot**, not a
full training run. Start with the real-case notebook below. The measured
activation-checkpointing overlay is isolated in experimental helpers; selecting
`merit_native512.yaml` in the production trainer does **not** enable that overlay.

Handoff status: the identified runtime repairs and bounded single-case GPU
checks are complete. The team member's training/validation results are still
awaited. This closes the current repair-and-pilot phase, not every open research
or production task: full native training still needs the trainer integration
listed below, and the older 128-grid resize/auxiliary-label issue remains open.

Read in this order:

1. This README: current status, runnable entry points and data requirements.
2. [Architecture](docs/architecture.md): source folders and how they connect.
3. [Data pipeline](docs/data_pipeline.md): what a batch contains and which
   transformations are explicitly configured. [Dataset reports](docs/dataset_reports.md)
   explains the CSV fields and saved split.
4. [MERIT model](docs/merit_unetrpp.md): TEP, MER, PCG, supervision and paper correspondence.
5. [MERIT pipeline](docs/merit_pipeline.md): production training/resume/inference commands.
6. [Validation](docs/validation.md) and [improvement note 01](docs/improvement_01_mer_only_checkpointing.md):
   measured results, limits and the next single-case experiment.

### Completed work and current boundary

| Area | Implemented and checked | Where to look |
| --- | --- | --- |
| Shared data access | Lazy CT/GT loading, geometry checks, slab sampling, saved five-fold selection and native reconstruction | `src/merit_covid/data/pipeline/` |
| Models | Standalone UNETR++ and paper-derived MERIT with TEP/MER/PCG; core model files are unchanged by the latest runtime review | `src/merit_covid/models/` |
| Runtime repairs | CT/GT source binding on resume, strict thresholds and finite-logit checks, reproducible volume-block shuffle | `src/merit_covid/engine/`, `data/pipeline/dataloader.py` under the package |
| Diagnostics | Split/input/resize audits, isolated optimizer profiles, restored smoke and baseline input scripts | `scripts/merit_preflight.py`, `scripts/model_smoke.py`, `scripts/unetrpp_real_input.py` |
| Native-512 feasibility | Separate config and notebook helpers; two synthetic updates and two real A-0025 updates passed on a user-run Colab T4 | `configs/merit_native512.yaml`, `tools/merit_checkpoint_trial.py`, `tools/merit_real_case_trial.py` |
| One measured improvement candidate | MER-only native-512 T4 pilot PASS; retain the earlier all-region overlay as reference because memory nearly doubles for an inconclusive timing gain | `tools/merit_mer_only_trial.py`, improvement note 01 |

The latest full CPU suite reports **389 passed, 1 CUDA-only skipped**. The compact
CUDA AMP test also passed separately in the local GPU environment. These checks
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
4.54 GiB peak allocation versus 9.00 GiB for MER-only. Production defaults remain
unchanged; see the completed experiment note for details.

## Quick start

### Recommended handoff: one real case on Colab

Open [the real-case notebook in Colab](https://colab.research.google.com/github/torusama/merit-covid-ct/blob/main/notebooks/colab_real_case.ipynb),
select a GPU runtime, then **Run all**. When prompted, upload either the team's
`A0025_CT_GT.zip` or these two original files from `data/covid19_20/Train/`:

- `volume-covid19-A-0025_ct.nii.gz`
- `volume-covid19-A-0025_seg.nii.gz`

The prepared ZIP is local at `outputs/colab_inputs/A0025_CT_GT.zip`; raw images
and this ZIP are **not committed to GitHub**. Transfer the ZIP separately to the
next team member, or obtain the original pair from the dataset. The notebook
checks their exact sizes/hashes and frozen fold-0 TRAIN membership; it does not
download the whole cohort. The full-cohort downloader is
`py tools/download_covid19_20.py`, using `MedOtter/COVID-19-20` on Hugging Face.

At completion, download the report ZIP and keep it with the experiment record.
If browser download does not start, use the Colab Files pane, refresh, find the
ZIP under `/content/merit-results/`, and select **Download** from its menu.
If a GPU quota blocks execution, hand over the notebook and input pair; that
does not count as a failed model experiment or a completed measurement.

| Notebook | Input and purpose | Current evidence |
| --- | --- | --- |
| [Real case](notebooks/colab_real_case.ipynb) | Original A-0025 pair; bounded two-update pilot with checkpointing | Native-512 T4 PASS; recommended first run |
| [Checkpoint trial](notebooks/colab_checkpoint_trial.ipynb) | Synthetic input; numerical checks followed by memory profile | Native-512 T4 PASS |
| [Original execution](notebooks/colab_connect.ipynb) | Synthetic 128 sanity check followed by unoptimized 512 profile | 128 PASS; 512 T4 OOM recorded |
| [MER-only candidate](notebooks/colab_mer_only_case.ipynb) | Same A-0025 pair; retain more backbone activations to measure a possible speed improvement | Native-512 T4 PASS; retain earlier overlay as reference |

All four notebooks fetch model/source commit
`e3493dd9f2dac704d2def6225d76f19df3262ec4` and embed their experimental helpers.
This pin reproduces the measured experiments; it does not fetch the latest
production runtime repairs on `main`. Do not repin during a baseline comparison.
They reuse runtime Torch/CUDA and install only missing dependencies. Activation
checkpointing recomputes intermediate activations during backward; it is
different from saving a training checkpoint for later resume.

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
The default is to reject unknown units. Use `--assume-mm` only after accepting
an evidence-backed mm assumption; it does not verify or repair the source.

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
py scripts/merit_pipeline.py check-split --config configs/merit_index.yaml
py scripts/merit_pipeline.py audit --config configs/merit_index.yaml --limit 1 --output outputs/first_audit.json
```

Choose and review the input policy before issuing the guide's `train` command:

| Configuration | Input policy | Limitation |
| --- | --- | --- |
| `merit_index.yaml`, `merit_physical.yaml` | Explicit 128x128 XY resize; index or physical sampling | Small lesions may disappear and derived auxiliary labels may change; this design remains unresolved |
| `merit_native512.yaml` | Full 512x512 on the verified source grid, batch 1 and AMP | Feasibility config; production trainer lacks the experimental checkpointing overlay and may OOM on T4 |

Both grids use feature size 16, depths 3/3/3/3 and TEP/MER/PCG. Learned projections
and positions depend on the input grid: parameter counts are 29,836,242 at 128
and 174,043,602 at 512. Their checkpoints are not directly shape-compatible.

Full-cohort training will reject the three unknown-unit cases until their units
are verified or the explicit mm assumption is accepted. See the
[MERIT pipeline guide](docs/merit_pipeline.md) for resume, validation, CV, physical
sampling, GPU checks and the explicitly gated external-test command.

The shared reader defaults to native XY size and NIfTI-scaled intensities.
The older MERIT index/physical configs explicitly resize XY to `128x128`; no
listed MERIT config interpolates Z. The native-512 pilot verifies that no XY
interpolation, intensity normalization or source-file modification occurs.
Training supervises true full-slab GT. Inference predicts each master once,
restores probabilities to native geometry and then evaluates whole volumes.

### Work still required before full native training

- Retain the measured all-region overlay as reference. The MER-only comparison
  is complete for this two-step pilot; further timing measurements are optional.
- Integrate the selected overlay into the production trainer and verify training,
  validation and resume together. It is currently enabled only in trial helpers.
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
| `scripts/model_smoke.py` | Synthetic model forward/backward without optimizer updates |
| `scripts/unetrpp_real_input.py` | One real native CT patch through the baseline, without training |
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
| Continue one scoped speed/memory experiment | [Improvement note 01](docs/improvement_01_mer_only_checkpointing.md) |

[CHANGELOG.md](CHANGELOG.md) records major integration milestones.
Upstream UNETR++ attribution remains in its packaged `LICENSE`, `NOTICE` and
`upstream.json`.
