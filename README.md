# COVID-19 CT Lesion Segmentation with MERIT-UNETR++

This repository implements the **data and evaluation foundation** for binary COVID-19 lesion segmentation: background `0`, lesion `1`. There is no implemented research model, training loop, checkpoint or reported model performance.

The target remains original MERIT-UNETR++: UNETR++ backbone, TEP, MER and PCG. Its integration requires authorized original source/specifications. Spacing and PCG analysis are later work. No new gate or routing mechanism is introduced.

## Installation

Python 3.10 or newer. Verified in this session on Windows/Python 3.11.

```bash
git clone https://github.com/torusama/merit-covid-ct.git
cd merit-covid-ct
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/Colab shell: source .venv/bin/activate
python -m pip install -e ".[test,tools]"
```

`pip install -e .` installs NumPy, nibabel, SciPy and YAML. Existing `pip install -r requirements.txt` installs tool dependencies and the package. PyTorch is optional: install the appropriate build for your environment, then `pip install -e ".[training]"`. Core modules/tests run without it. The viewer also needs the platform's Tk installation.

## Implemented foundation

- Existing downloads, preparation, audits, plots and viewer.
- Unified versioned manifests plus legacy `cases.json` readers.
- Strict binary masks, paired geometry checks, RAS conversion and preserved original geometry.
- Explicit intensity policy, optional paired XY resize, inverse transformation and no Z resampling.
- Ordered 1/3/5/11-slice sampling with actual physical context offsets.
- Reproducible grouped folds, Dataset modes and optional PyTorch DataLoader.
- Future model adapter contract, strict volume reconstruction and patient metrics.
- Synthetic NIfTI smoke tests and regression tests without downloading data.

Defaults preserve native XY, raw intensities, CPU and batch size 1. These are pipeline defaults, not a validated training recipe. Larger batches need equal XY dimensions. Select normalization/resize using primary training/validation evidence. Nearest-neighbor downsampling can delete small lesions despite preserving label values.

## Datasets and protocol

| Dataset | Expected volumes | Role | Local manifest |
| --- | ---: | --- | --- |
| COVID-19-20 | 199 | Development and internal evaluation | `data/covid19_20/cases.json` |
| Zenodo20 | 20 | Final external test only | `data/zenodo20/cases.json` |

Never use Zenodo20 for training, validation, preprocessing tuning, hyperparameters, thresholds or checkpoint selection. Historical external descriptive analysis is QC, not model-selection evidence. Test ground truth never defines crops.

The committed split has five outer buckets. Fold `i` uses bucket `i` for internal test, `(i+1) mod 5` for validation and three buckets for training: approximately 60/20/20. Every volume is outer-tested once. Verified patient IDs group linked volumes. Legacy manifests do not verify patient identity, so the committed split explicitly uses volume proxies. Obtain linkage metadata before claiming verified patient independence.

```bash
# Creation refuses an existing destination; reuse committed folds_seed42.json.
python scripts/make_splits.py data/covid19_20/cases.json --output splits/covid19_20/new_folds_seed42.json
# Header-only mode explicitly skips voxel/label checks:
python scripts/check_dataset.py data/covid19_20/cases.json --headers-only --limit 1
# Default performs full voxel checks and can create a separate enriched manifest:
python scripts/check_dataset.py data/covid19_20/cases.json --write-manifest outputs/covid19_manifest_v1.json
```

Unknown spatial units fail. Use `--assumed-unit mm` only with evidence that unknown units represent millimetres. HU/windowed normalization requires explicit verified intensity provenance. Viewer heuristics do not establish HU. Z spacing means center-to-center distance, not slice thickness.

## Configuration and tensors

`configs/experiment_baseline.yaml` defines data/output paths, seed, folds, slab depth/master, XY size, normalization, batching, device and planned training parameters. YAML paths resolve relative to the file.

```python
from merit_covid.config import load_config
from merit_covid.data.manifest import load_manifest
from merit_covid.data.splitting import load_splits
from merit_covid.data.dataset import SlabDataset, make_dataloader

config = load_config("configs/experiment_baseline.yaml")
cases = load_manifest(config.dataset_root / "cases.json")
splits = load_splits("splits/covid19_20/folds_seed42.json", cases)
train_ids = set(splits["folds"][0]["train"])
dataset = SlabDataset([c for c in cases if c.case_id in train_ids], config, mode="train")
sample = dataset[0]  # Validates source geometry/units/labels on load.
# loader = make_dataloader(dataset)  # Requires optional PyTorch.
```

Each sample provides `image [1,D,H,W]`, `mask [1,H,W]`, `master_index`, `slice_index`, `case_id`, `dataset`, `context_indices [D]` and `z_offsets_mm [D]`. Default `D=11`, master position 5. Offsets reflect actual repeated/reflected source slices. Inference never reads masks and omits them. Batches add a leading batch dimension. Geometry stays in `dataset.prepared_volume()`.

## Verification without downloads or training

```bash
python -m pip install -e ".[test]"
python -m pytest tests -q
python -m compileall src scripts tools
python scripts/smoke_test.py
python tools/download_datasets.py --help
python scripts/check_dataset.py --help
python scripts/make_splits.py --help
```

The smoke test uses temporary synthetic NIfTI and identity reconstruction, not a model. Torch-specific tests skip when torch is absent. Exact results and unverified work are recorded in [the implementation report](docs/implementation_report.md).

## Preserved dataset tools

These manual commands download large datasets; they are not invoked by tests:

```bash
python tools/download_datasets.py --dataset all
python tools/download_datasets.py --dataset covid19_20
python tools/download_datasets.py --dataset zenodo20
```

Public sources are [MedOtter/COVID-19-20](https://huggingface.co/datasets/MedOtter/COVID-19-20) and [Zenodo record 3757476](https://zenodo.org/records/3757476). Data/archives are excluded from Git. The original storage estimate is at least 12 GB; retained derived data may require more. Download verification checks advertised size/count and selected geometry, not cryptographic source identity.

```bash
python tools/view_slices.py
python tools/view_slices.py --preview --dataset covid19_20
python tools/view_slices.py --preview --dataset zenodo20
python tools/download_zenodo20.py
python tools/prepare_zenodo20.py
python tools/download_covid19_20.py
# The following regenerate historical outputs; run deliberately:
python tools/audit_zenodo20.py
python tools/plot_zenodo20_audit.py
```

Preparation adds `lung_mask` while retaining legacy keys and source filenames. Audit/viewer use recorded mask paths; omitted legacy lung paths are resolved from actual files. Saved historical analysis remains unchanged.

## Layout and documentation

```text
configs/               Validated baseline configuration
src/merit_covid/        Config, data, evaluation, model contract, CLI and seeds
scripts/               Thin check/split/synthetic-smoke entry points
tools/                 Preserved acquisition, audit and viewer workflows
tests/                 Synthetic and mocked regression tests
splits/covid19_20/      Immutable grouped primary split
docs/                  Audit, architecture, contracts, roadmap, report
analysis/zenodo20/     Preserved historical external descriptive audit
data/, archives/       Local originals and downloads (ignored)
outputs/               Derived predictions and previews (ignored)
references/            Third-party references (ignored)
```

Read [source audit](docs/source_audit.md), [architecture](docs/architecture.md), [data contract](docs/data_contract.md), [roadmap](docs/implementation_roadmap.md) and [implementation report](docs/implementation_report.md).
