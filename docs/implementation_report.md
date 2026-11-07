> Historical main snapshot before MERGE. Current integration and verification: [merge_report.md](merge_report.md).

# Implementation report

Date: 2026-11-07 (Asia/Bangkok). All new documentation, code comments and messages
are in English, following the latest user instruction.

## A. Source audit

Ran `git pull --ff-only` first: main was already current at `aec40ed`. No reset,
stash, merge, commit or push was performed. Existing untracked comparison scripts
were inspected and preserved. The complete file-by-file audit, including purpose,
inputs, outputs, dependencies, reusable logic, issue classification and placement,
is in [source_audit.md](source_audit.md).

Confirmed issues repaired: case ID/source filename mismatch in mask lookup;
omitted lung-mask paths in preparation; bare downloader imports failing package
execution; duplicated GUI/preview loading without alignment checks; plot docstring
referencing a nonexistent script; README promoting speculative research extensions
instead of describing implemented code.

Missing features now supplied: installable package, config, manifests, paired I/O,
preprocessing, slabs, grouped folds, Dataset, optional DataLoader, model boundary,
reconstruction, metrics and synthetic tests. Potential risks still requiring data
evidence include patient linkage, HU/windowed provenance, unknown units, source
byte identity, small-lesion loss under resize and large-volume RAM use.

Existing manifests were read as metadata/path records: 199 primary and 20 external
cases. The primary manifest records Z spacing counts 188 at 5 mm, 6 at 2 mm, 4
at 1 mm and 1 at 0.8 mm. These are manifest observations, not newly measured
whole-volume validation. Historical external audit results were not regenerated.

## B. Architecture design

The final tree and per-file responsibilities are in
[architecture.md](architecture.md). The dependency direction is configuration and
manifest -> paired data processing -> sampling/Dataset -> future model adapter ->
independent reconstruction and evaluation. Core modules do not import torch or
model implementations. New scripts are thin entry points. Existing acquisition
and exploratory tools remain in place, with shared ID/path/geometry helpers in
the package. This avoids an unnecessary migration of functioning legacy code.

## C. Implementation changes

| Change | Files | Reason |
| --- | --- | --- |
| Added packaging | pyproject.toml, src/merit_covid/__init__.py | Editable installation; core/test/tools/training dependency separation |
| Added configuration | config.py, configs/experiment_baseline.yaml, utils.py | Validated portable paths, seed, slab, normalization, device and planned training settings |
| Added data layer | data/manifest.py, nifti_io.py, preprocessing.py, slab.py, splitting.py, dataset.py and package initializer | Explicit shared contracts, geometry retention, external isolation and reproducible sampling |
| Added evaluation | engine/metrics.py, reconstruction.py and initializer | Model-independent patient metrics and strict indexed assembly |
| Added model boundary | models/base.py, initializer, merit_unetrpp/README.md | Clear source-faithful integration location without invented modules |
| Added entry points | cli.py, scripts/check_dataset.py, make_splits.py, smoke_test.py | Usable checks, immutable splits and temporary synthetic smoke test |
| Added tests | tests/conftest.py and eleven test modules | Synthetic geometry/labels, sampling, leakage, metrics, reconstruction, CLI and legacy-tool regression evidence |
| Added primary split | splits/covid19_20/folds_seed42.json | Reusable 199-case seed-42 five-fold protocol |
| Updated legacy tools | download_datasets.py, prepare_zenodo20.py, audit_zenodo20.py, view_slices.py, plot_zenodo20_audit.py | Lazy/package-aware dispatch, explicit lung path, shared actual-file resolution, viewer alignment and corrected docstring |
| Updated setup | requirements.txt, .gitignore | Preserve old install workflow while adding package; ignore build/test caches |
| Added/rewrote documentation | README.md and docs/*.md | Accurate state, audit, architecture, tensor/data rules, verification and prioritized follow-up |

No existing file moved. No CT, mask, source archive, existing local manifest,
user comparison script or historical analysis artifact was edited. Legacy entry
points remain available. Generated package metadata/caches are ignored.

Saved fold sizes:

| Fold | Train | Validation | Internal test |
| --- | ---: | ---: | ---: |
| 0 | 119 | 40 | 40 |
| 1 | 119 | 40 | 40 |
| 2 | 119 | 40 | 40 |
| 3 | 120 | 39 | 40 |
| 4 | 120 | 40 | 39 |

All 199 patient IDs are unknown in the current manifests. The split therefore
guarantees disjoint volume IDs and recorded grouping, not independently verified
patient identity. The implementation supports verified patient grouping when
evidence becomes available. No Zenodo20 case enters these folds.

## D. Data flow

CT NIfTI -> legacy/versioned Manifest and validated split role -> paired shape,
affine, header spacing, units and labels -> shared canonical RAS orientation ->
explicit normalization and optional paired XY resize -> ordered master/context
slab -> NumPy Dataset / optional torch DataLoader -> future model adapter returning
master logits -> explicit fixed-threshold predictions -> indexed canonical volume
reconstruction -> inverse XY/orientation -> original-grid binary NIfTI -> patient
Dice/foreground IoU/sensitivity/precision -> equal-weight patient macro average.

Original/master/context indices and geometry are retained. Z is never resampled.
Inference drops label paths and cannot read GT. Original MERIT and model inference
remain unimplemented; the synthetic smoke test checks identity reconstruction,
not fabricated segmentation performance. See [data_contract.md](data_contract.md)
for shapes, padding, units, inverse transformation, empty-mask and leakage rules.

## E. Verification

Environment: Windows PowerShell, Python 3.11.9; NumPy 2.2.1, nibabel 5.4.2,
SciPy 1.17.1, PyYAML 6.0.3, pytest 9.2.0. PyTorch is absent.

| Exact command | Result |
| --- | --- |
| `git pull --ff-only` | PASS; already up to date on main |
| `python -m pip install -e ".[test]"` | PASS; editable build and installation |
| `python -m pip install -e .` | PASS; core-only editable installation |
| `python -m pytest tests -q` | PASS; final run: **56 passed, 1 skipped in 8.31 seconds** |
| `python -m compileall src scripts tools` | PASS; all package, new entry point and legacy tool sources compile |
| `python scripts/smoke_test.py` | PASS; temporary synthetic paired NIfTI -> Dataset/slabs -> shuffled reconstruction -> original-grid export -> Dice 1 |
| `python tools/download_datasets.py --help` | PASS; old file entry point |
| `python -m tools.download_datasets --help` | PASS; package/module execution |
| `python tools/view_slices.py --help` | PASS; viewer CLI import and parsing |
| `python scripts/check_dataset.py --help` | PASS |
| `python scripts/make_splits.py --help` | PASS |
| `python scripts/make_splits.py data/covid19_20/cases.json --output splits/covid19_20/folds_seed42.json` | PASS; 199 primary cases, five validated folds, exclusive output creation |
| `git diff --check` | PASS; Git emitted only configured LF/CRLF conversion notices |
| `git -c core.safecrlf=false diff --check` | PASS |
| `git diff --exit-code -- analysis/zenodo20` | PASS; no historical artifact changes |

The module import check was also run exactly as follows:

```bash
python -c "import pkgutil; import importlib; import sys; import merit_covid; modules=[m.name for m in pkgutil.walk_packages(merit_covid.__path__, merit_covid.__name__+'.')]; [importlib.import_module(name) for name in modules]; print('Imported',len(modules),'package modules'); print('Torch imported:', 'torch' in sys.modules)"
```

PASS: 15 package modules imported; torch was not imported. Existing local manifests,
the saved split and baseline config were loaded and the primary/external counts,
fingerprint, fold coverage/group disjointness and CPU setting were confirmed via
the following exact metadata-only check:

```bash
python -c "from merit_covid.config import load_config; from merit_covid.data.manifest import load_manifest; from merit_covid.data.splitting import load_splits; c=load_manifest('data/covid19_20/cases.json'); s=load_splits('splits/covid19_20/folds_seed42.json', c); print('primary cases:', len(c)); print('split role counts:', [{k:len(f[k]) for k in ['train','validation','internal_test']} for f in s['folds']]); print('unverified patients:',sum(x.patient_id is None for x in c)); print('external legacy records:',len(load_manifest('data/zenodo20/cases.json'))); print('configured CPU:',load_config('configs/experiment_baseline.yaml').device)"
```

Regression tests use synthetic files or mocks to cover explicit source filenames,
archive basename extraction, original downloader dispatch, external five-panel
viewer rendering, audit output generation, CLI full/header-only distinctions,
non-overwriting outputs, inference without accessible GT, paired flips/permutations,
physical units and padding, strict labels, linked-patient grouping, metrics and
missing/duplicate/out-of-order predictions.

Exactly one test was skipped: actual torch DataLoader tensor collation, because
torch is not installed. Its absent-dependency error path passed. Not run: live
downloads, full original-data voxel validation, real GUI interaction, GPU work,
training/model inference, Linux/Colab runtime or benchmark performance. No large
dataset was downloaded, and no GPU was used. Fresh download source identity and
original architecture fidelity are not certified by these synthetic tests.

## F. Next tasks

1. Data lead: verify patient linkage, units, labels and intensity provenance on
   primary data; fix preprocessing choices through internal validation and lesion
   retention QC. Create a new split if verified linkage changes grouping.
2. Supervisor/model lead: provide original authorized MERIT source/specification,
   license and commit; integrate original backbone/TEP/MER/PCG through the adapter.
3. Training lead: install suitable torch build, run CPU/multiprocess DataLoader
   tests across environments, then implement trainer/baselines with recorded
   config/split/source provenance.
4. Evaluation lead: add physical HD95, uncertainty intervals and study reporting;
   freeze settings internally before final external evaluation.
5. Research lead: consider MER spacing and PCG analyses only after original-model
   reproduction. No new architecture claim is part of this implementation.

See [implementation_roadmap.md](implementation_roadmap.md) for priorities, owners
and concrete integration requirements.
