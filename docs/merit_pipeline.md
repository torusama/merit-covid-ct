# MERIT training, inference and evaluation

This guide owns the runnable MERIT workflow. Model equations/provenance live in
[merit_unetrpp.md](merit_unetrpp.md); shared data/metric contracts live in
[data_pipeline.md](data_pipeline.md). The entry point is
`scripts/merit_pipeline.py`. No second reader, sampler or evaluator is introduced.

## Configuration and data flow

```text
CT (+ GT only for labeled operations)
  -> shared headers/units/LPS/scaling/intensity/explicit XY processing
  -> index or nearest-mm slab + true GT + duplicate validity
  -> MERITUNETRPP
       train: full-slab MERITLoss -> AdamW -> native-volume validation
       infer: master fg-bg -> sigmoid -> native XY probability restoration
  -> complete VolumePrediction -> original XYZ -> native-volume metrics
```

Experiment YAML requires exactly `data/preprocess/model/loss/training` sections.
Relative data paths resolve against the YAML location. Effective model/loss
defaults are materialized into the frozen config. The model's fixed
`img_size` must match preprocessing `slab_size/size_hw`. `size_hw: null` keeps
native XY; train/validation headers must then match the fixed model XY grid.

`configs/merit_unetrpp.yaml` is model/loss-only smoke configuration; it is not a
full experiment. The primary CLI experiment is `merit_native512.yaml`, also
the default for `train` and `cv`. The older `merit_index.yaml` and
`merit_physical.yaml` remain explicit 128-grid research configurations; resize
can remove small lesions and alter derived presence/salience targets there.

## Primary full native XY training (route 1)

`configs/merit_native512.yaml` is the primary full-slice experiment: input is
`[B,1,11,512,512]`, with the same feature size 16, depths 3/3/3/3 and all three
MERIT modules enabled. It preserves native intensity and the existing sampling,
loss, split and unknown-unit policies. `size_hw: null` disables XY interpolation
and source/model grid mismatch is rejected before training. The config enables
`training.activation_checkpointing: true`: the production trainer recomputes
MER query blocks and the validated stateless backbone regions during backward.
The model architecture, parameter/state shapes and native CT/GT are retained.

The `native-input` diagnostic checks one central slab from a selected TRAIN
case. It refuses a source/grid mismatch, forbids calls to the XY interpolation
function, compares CT and GT tensors with source pixels, and verifies source
file SHA-256 before/after. It does not run a model or an optimizer.

```powershell
py scripts/merit_preflight.py native-input --config configs/merit_native512.yaml --fold 0 --case-id volume-covid19-A-0025 --output outputs/native512/input.json
```

Profile the configured full architecture in a separate CUDA process before
training. `--memory-fraction` is an optional PyTorch allocator limit relative
to physical VRAM; 1.0 allows the full physical budget, not extra shared system
memory. Omit it for the allocator's default behavior. It does not change input
values, model parameters or attention rules. A failed profile records the
exception, peak allocated/reserved memory and successful optimizer-step count.

```powershell
py scripts/merit_preflight.py profile --config configs/merit_native512.yaml --size 512 --batch-size 1 --device cuda --warmup 1 --repeats 1 --memory-fraction 1.0 --output outputs/native512/profile.json
```

Larger grids also enlarge grid-dependent positional/projection parameters in
the original backbone. Rebuild the model for this config; a 128x128 checkpoint
is not interchangeable with a 512x512 checkpoint. Preserving input details does
not itself establish training feasibility or improved segmentation accuracy.
Recorded local feasibility results are in [validation.md](validation.md).

## Colab and Kaggle quick start

Use [notebooks/colab_connect.ipynb](../notebooks/colab_connect.ipynb).

1. Colab: upload the supplied notebook, select a GPU runtime, then **Run all**.
   A GitHub launch URL loads the published notebook, not uncommitted local edits.
2. Kaggle: import the same notebook, enable GPU and Internet, then **Run all**.
3. Save the generated report ZIP. Colab provides a final download cell; on
   Kaggle retain `/kaggle/working/merit-results/` with Save Version.

If Colab's browser download does not start, open the left Files panel, refresh,
expand `merit-results`, and choose Download from the ZIP file's three-dot menu.
The printed `/content/...` path identifies a runtime file, not a browser URL.

There is no Drive mount, hard-coded personal folder, token, CT download or
manual source checkout. The public source is fetched once and reused while
the runtime remains alive. Installed dependencies are checked before pip runs.
Torch and NumPy are constrained to the runtime's existing versions: setup will
fail on an incompatible runtime instead of downloading a replacement CUDA/Torch
stack. Colab/Kaggle still need a GPU assignment and network access; a new VM
may need to install the small missing project dependencies again.

Defaults run the full 128-grid sanity profile, then the full 512-grid profile,
both batch 1 with AMP and complete optimizer steps. A failed sanity check stops
the larger attempt. Each runs in a fresh child process; the notebook kernel
does not import Torch or reserve a CUDA context. The allocator budget is
explicitly recorded (default 90% physical VRAM). OOM/other failures remain
visible in JSON/logs and are included in the ZIP; OOM does not become PASS.

The notebook intentionally pins published source commit
`e3493dd9f2dac704d2def6225d76f19df3262ec4`. Its model is identical to the local
route-1 measurement. Unpublished local checkpoint/data-binding/inference repairs
are not fetched, and the notebook does not use those paths: it is synthetic
profiling only, with no dataset, checkpoint/resume, full training or evaluation.
Update the pinned commit deliberately when publishing a new experiment.

These cloud operations are described in the
[Colab FAQ](https://research.google.com/colaboratory/faq.html) and
[Kaggle notebook guide](https://www.kaggle.com/docs/notebooks).

### Experimental activation checkpointing

Upload [colab_checkpoint_trial.ipynb](../notebooks/colab_checkpoint_trial.ipynb),
enable GPU, and Run all. Defaults are 512x512, batch 1, AMP, and the full physical
allocator budget. No dataset is needed. The standalone helper
`tools/merit_checkpoint_trial.py` is embedded verbatim, with its SHA-256 recorded,
so it can be tested against the same pinned public model without publishing
changes to that model. Core model files remain untouched.

The helper checkpoints complete MER query-block calculations including the
distance bias; each query still attends all context keys. It also checkpoints
EPA, TEP, encoder1 and the final decoder. BatchNorm regions are excluded so
running statistics are not updated a second time during recomputation. RNG
state is preserved. Parameter objects, state keys and counts are unchanged.

Before profiling, the worker checks float64 MER output/all gradients, compact
full-model FP32 and AMP outputs/loss/input and parameter gradients, and exact
normalization-buffer equality. Failure stops profiling. Comparisons use
explicit recorded numerical tolerances; they are not a universal bitwise
equivalence claim. These checks also run on the assigned cloud GPU.

The report ZIP additionally contains `profile_512_equivalence.json` and the
exact trial helper. A compact validation PASS is not a native-512 training PASS:
read successful optimizer steps and memory results in `profile_512.json`.
The validated all-region strategy is now available in the production trainer
through `training.activation_checkpointing`; the primary native-512 config
enables it. This historical notebook stays pinned to its original experiment.

### One uploaded real TRAIN case

Use [colab_real_case.ipynb](../notebooks/colab_real_case.ipynb): enable GPU, Run all,
and upload the prepared `A0025_CT_GT.zip` when prompted. The ZIP contains only
the original A-0025 CT and mask; alternatively upload both original `.nii.gz`
files. No manual extraction or full-cohort download is required. Verified inputs
are reused within the same runtime. Wrong hashes, unexpected ZIP members and
non-TRAIN selections are rejected before training.

Base reader/model/loss and the canonical split come from the pinned public
commit. `tools/merit_real_case_trial.py` and the checkpoint helper are embedded
verbatim and hashed in the report, so a GitHub push is not required for the
experiment. The frozen split bytes and case/group membership are verified;
the other 198 source files are not replaced with placeholders or claimed as
audited. Complete-cohort manifest validation and Trainer.fit/resume are outside
this bounded pilot.

The pilot forbids XY interpolation, compares each selected CT/GT slab with
native source pixels, retains intensity/unit/sampling policies, and selects
central consecutive masters without lesion-based selection. It attempts two
AdamW/AMP updates with the shared MERIT loss and checkpointed model, records
losses/gradient checks/timing/memory and parameter changes, and runs one CT-only
master prediction under a GT-read guard. No complete volume or quality metric
is produced. The report ZIP excludes patient images and large model weights.

## Sampling and independent physical settings

`master_index` is local (5 of 11); `master_slice_index` is canonical-volume level.
Reorientation can change its relation to the original file index.

`index` sampling preserves consecutive native indices with declared boundary
handling. `physical_mm` requires `target_spacing_mm=d>0` and nearest native
centers at requested offsets `[-5d,...,5d]`. It performs no Z interpolation.
Actual offsets are retained separately from requested targets.
Full-affine center distances support oblique/sheared stacks. Nearest ties within
1e-7 voxel choose lower canonical indices.

Default `unique` validity keeps one loss representative per selected native
slice within each slab while retaining duplicate context input. Master has
priority; real clamped endpoints retain a representative. `all` is a repeated
supervision ablation. Validity does not mask attention input.

Three physical settings are independent:

| Setting | Baseline | Physical alternative |
| --- | --- | --- |
| Sampling | `preprocess.sampling_mode=index` | `physical_mm` with explicit target spacing |
| Position | `model.position_mode=index` | actual offsets / `position_unit_mm` |
| Loss weighting | `loss.weighting_mode=index`, `sigma_slices` | actual distances with explicit `sigma_mm` |

Missing physical offsets or required parameters fail fast, never fall back to
index semantics.

## Preprocessing and supervision

Shared-reader defaults preserve native XY and scaled intensities. Experiment
YAML explicitly chooses XY `128x128`; CT is bilinear, labels nearest-exact,
and Z unchanged. Full-slab targets come from the actual selected native GT,
not a replicated master mask. `valid_region` excludes invalid duplicates.

Native intensity does not imply verified HU. Explicit HU/normalized modes need
source provenance and use identical frozen settings for all partitions.
Lesion-retention reporting detects entire lost 2D components and positive
slices under XY resizing; it never excludes cases automatically. Final metrics
use native GT, not resized labels.

The Dataset's CT-only mode strips mask paths before header/voxel reading.
Recorded mask names can still participate in split fingerprint validation when
the files are missing. Each Windows worker caches at most one volume.

The experiment YAML files explicitly set `training.shuffle_mode: volume`:
shuffle the case order, then shuffle masters within each case. Every master is
visited once per epoch; no GT or lesion statistics control selection. With
workers=0 this needs one cache fill per case instead of reloading a whole CT
after almost every globally shuffled slice. Workers maintain separate bounded
caches, so multiple workers can each load the same case. This changes sample
order and is recorded in the frozen training protocol; it does not change the
split, CT values, or targets. `slice` retains global slice shuffling, the shared
loader API default. `sequential` disables shuffling for bounded diagnostics.
Validation/test always remain sequential. Loader RNG state includes volume
and within-volume permutations for exact epoch-boundary resume.

## Model and loss

```python
output = model(image, master_index, z_offsets_mm=batch["z_offsets_mm"])
losses = criterion(
    output, batch["mask_slab"], batch["master_index"],
    valid_region=batch["valid_region"], z_offsets_mm=batch["z_offsets_mm"],
)
losses["total"].backward()
```

Training retains the four-part model output. Per-slice Dice/CE and auxiliary
losses ignore invalid regions; master must have valid pixels. Gaussian weighting
is computed in FP64 to avoid complete supervision underflow at extreme sigma.
`gaussian_sigma` remains the legacy index-sigma alias.

Alignment defaults to maximum raw foreground probability plus MSE against
presence. Optional `top_k_mean` with `top_k_fraction` is a separate ablation.
A synthetic one-pixel probe tests sensitivity; it does not establish which
pooling improves clinical Dice. Presence/salience BCE runs FP32 with autocast
disabled, preserving the probability-BCE formula.

## Trainer and checkpoints

`training.activation_checkpointing` is a strict boolean, default false for
compatibility and explicitly true in `merit_native512.yaml`. The production
overlay is applied before optimizer creation. It recomputes MER distance bias
and attention per query block against all context keys, plus EPA, TEP, encoder1
and the final decoder. It preserves dropout RNG and excludes mutable BatchNorm
regions. Evaluation/no-grad executes without recomputation. Parameter/state
keys do not change. The setting is frozen in checkpoints; changing it on resume
is rejected. Historical schema-2 configs missing the field retain their disabled
behavior and fingerprint. The experimental helpers remain frozen for provenance.

`engine.training.Trainer` provides AdamW, CUDA FP16 AMP, gradient clipping,
seeding, deterministic settings, ReduceLROnPlateau and early stopping.
It logs individual loss components and skipped AMP optimizer steps.
PCG `set_epoch(epoch)` is called every epoch, default warmup 30.

Best checkpoint selection uses validation **group-macro full native-volume
Dice**, not paper LIDC master-ROI Dice. Default threshold 0.5 is declared in the
experiment; it is not selected from internal/external test labels.
The shared config/metric validator requires a finite `0 < threshold < 1`;
invalid endpoints fail before model construction or any optimizer step.

`best.pt` and `last.pt` save model, optimizer, scheduler, scaler, epoch,
best metric, early-stop counter, Python/NumPy/Torch/CUDA RNG, loader shuffle
state, PCG ramp, effective config, split identity/fingerprint/hash, resolved
CT and GT source paths and runtime. Schema 2 also binds the data configuration,
manifest/patient-map bytes (SHA-256), and each resolved CT/GT path, size and
nanosecond modification time. Checkpoints replace atomically.
Existing runs require explicit resume.

Resume verifies the frozen protocol. Only extension of the epoch budget is
permitted; model/loss/preprocessing/sampling/fold/path changes are rejected.
The trainer also rejects stale indexed source bindings and supplied CT/GT paths
that disagree with the current manifest. Schema-1 checkpoints lack this binding:
they remain loadable for CT-only inference, but cannot resume training. No
automatic legacy migration is performed.
Moving a dataset requires a new protocol or verified migration.
Exact interrupted/uninterrupted replay is tested at epoch boundaries, not
mid-batch. CPU/CUDA type switches are rejected; cross-hardware/version bitwise
reproducibility is not promised. Checkpoints do not hash every source voxel:
file stats catch ordinary in-place edits, not replacement with deliberately
preserved size and timestamp. Version and pin upstream data separately. Load
only trusted local checkpoints.

## Inference and metrics

`predict_slab_case` requires eval mode, predicts every canonical master once
without reading GT, converts two-class logits using `sigmoid(fg-bg)`, restores
probabilities to native XY, then constructs `VolumePrediction`.
Slice assembly rejects duplicate/missing slices or changed case/geometry.
Master logits must have the expected floating shape and be finite before
sigmoid; infinite logits cannot silently become apparently valid 0/1 scores.
Original orientation is restored before thresholding.

Standalone baseline patch inference remains separate and explicitly rejects
MERIT's incompatible output interface. The common evaluator computes whole
native-volume Dice/IoU/precision/recall, physical HD95 and lesion-volume errors.
It averages within group then across groups, reporting undefined metrics.
See [data contracts](data_pipeline.md#metrics) for empty-mask rules.

## Commands

Run from the repository root with installed model/test dependencies:

```powershell
# Validate the recorded split; no voxel reads or split regeneration.
py scripts/merit_pipeline.py check-split --config configs/merit_native512.yaml
# Real data inspection; start small.
py scripts/merit_pipeline.py audit --config configs/merit_native512.yaml --headers-only --limit 1 --output outputs/headers.json
py scripts/merit_pipeline.py audit --config configs/merit_native512.yaml --limit 1 --output outputs/voxel_audit.json

# Temporary synthetic train/validate/save/reload/resume.
py scripts/merit_pipeline.py smoke --device cpu --output outputs/cpu_smoke.json
py scripts/merit_pipeline.py smoke --device cuda --output outputs/cuda_smoke.json

# Development fold training and epoch-boundary resume.
py scripts/merit_pipeline.py train --config configs/merit_native512.yaml --fold 0 --output outputs/e1/fold0
py scripts/merit_pipeline.py train --config configs/merit_native512.yaml --fold 0 --output outputs/e1/fold0 --resume outputs/e1/fold0/last.pt
# Checkpoint-based validation and internal test.
py scripts/merit_pipeline.py evaluate --checkpoint outputs/e1/fold0/best.pt --partition validation --device cuda --output outputs/e1/fold0/validation.json
py scripts/merit_pipeline.py evaluate --checkpoint outputs/e1/fold0/best.pt --partition test --device cuda --output outputs/e1/fold0/test.json
# Long-running research experiment; not part of a software smoke check.
py scripts/merit_pipeline.py cv --config configs/merit_native512.yaml --output outputs/e1_cv
# Full reference forward/backward, without optimizer state.
py scripts/merit_pipeline.py benchmark --config configs/merit_native512.yaml --device cuda --batch-sizes 1 --output outputs/merit_memory.json
```

Reports use exclusive creation: use new output filenames on repeat runs.
Three primary cases lack spatial units; default full-cohort loading rejects
them. `--assume-mm` records an explicit evidence-backed assumption and must
match when resuming; it is not an automatic workaround for invalid metadata.

To create a genuinely new split version, do not overwrite the canonical file:

```powershell
py scripts/make_splits.py data/covid19_20/cases.json --output outputs/new_split.json
```

Physical template spacing is intentionally null. Supply a value selected using
training/validation. The following values demonstrate syntax, not recommendations:

```powershell
py scripts/merit_pipeline.py train --config configs/merit_physical.yaml --target-spacing-mm 3 --fold 0 --output outputs/e2/fold0
# Separate PE ablation; keep other axes fixed.
py scripts/merit_pipeline.py train --config configs/merit_physical.yaml --target-spacing-mm 3 --position-mode physical_mm --fold 0 --output outputs/e3_pe/fold0
# Separate physical-Gaussian ablation.
py scripts/merit_pipeline.py train --config configs/merit_physical.yaml --target-spacing-mm 3 --weighting-mode physical_mm --sigma-mm 6 --fold 0 --output outputs/e3_weight/fold0
```

## Pre-training preflight

`scripts/merit_preflight.py` provides bounded diagnostics. Run commands
sequentially on low-memory hosts. Unlike the historical `benchmark`, `profile`
includes AdamW state, one warmup and two measured complete optimizer steps.
Synthetic inputs measure resources only; `pilot` tests actual CT intensities.

```powershell
# One native TRAIN input; all-fold split checking uses metadata only.
py scripts/merit_preflight.py native-input --config configs/merit_native512.yaml --fold 0 --case-id volume-covid19-A-0025 --output outputs/preflight_new/native_input.json

# Separate processes isolate allocations/OOM; do not run concurrently.
py scripts/merit_preflight.py profile --config configs/merit_native512.yaml --size 512 --batch-size 1 --output outputs/preflight_new/gpu512_b1.json

# Actual TRAIN CT: four optimizer attempts, fresh Trainer resume, CT-only
# inference, complete reconstruction and original-shape/affine mask export.
# A-0025 is a fold-0 TRAIN case with explicit mm; no unit assumption needed.
py scripts/merit_preflight.py pilot --config configs/merit_native512.yaml --fold 0 --case-id volume-covid19-A-0025 --size 512 --steps 4 --device cuda --output outputs/preflight_new/pilot512
```

The production pilot needs the complete development manifest/sources for source
binding and saves a large diagnostic checkpoint; its all-master inference takes
longer than the two-update notebook. It is not a full-fold or selected checkpoint.
Optional resize-retention audits can still use `inspect --sizes 128 256` with an
explicit 128-grid research config; they do not change primary training input.

Use fresh outputs on reruns. Audit moments and histogram cover every native
scaled CT voxel in the selected TRAIN cases, before interpolation or
normalization. All-voxel percentiles use one-native-unit histogram bins: exact
for integer-valued data, less than one unit bin error for fractional data;
out-of-range counts are explicit. CSV includes storage dtype, NIfTI scaling
and spatial units. Neither ranges nor slope/intercept proves HU calibration.
Source documentation describing HU is not per-case DICOM verification.
No intensity window or normalization is applied by the preflight.

Resize uses production nearest-exact pixel-center mapping. Components are
**2D 4-connected regions**, not distinct clinical 3D lesions. Completely lost
components, their native voxel support, positive-to-empty slices and native
foreground missing after downsample/restore are separate metrics. Gained
background is also reported. Never subtract pixel counts on unequal grids.
Illustrations have display-only CT contrast and are selected by GT error only
for diagnostics, never for choosing training inputs. These geometric checks
are not model accuracy or a bound on attainable Dice.

All five folds are checked for case/group/path overlap using metadata only.
Volume-disjoint splits do not establish patient independence; duplicate content
and repeated-patient linkage need additional evidence. The pilot never reads
validation/test voxels or selects a best checkpoint, and guards against reading
GT during inference. Its checkpoint and mask are diagnostic, not clinical.
`PASS` means the requested software checks completed, not research readiness;
`OOM`, `FAIL` and unverified provenance must be reviewed separately.

## External testing and experiment design

Only after freezing the final development protocol and explicitly deciding to
evaluate external data:

```powershell
py scripts/merit_pipeline.py external --checkpoint outputs/e1/fold0/best.pt --manifest data/zenodo20/cases.json --confirm-frozen-protocol --device cuda --output outputs/external_final.json
```

The external manifest must contain exactly 20 Zenodo20 cases. Known ID/group/path
overlap is rejected. Cross-source clinical independence cannot be proved without
reliable patient linkage. The CLI does not adapt threshold/preprocessing to
external labels or silently ensemble/retrain five folds. Define a separate final
refit protocol if needed.

| Experiment | Change |
| --- | --- |
| E0 | Standalone UNETR++ with native patch preparation; same native evaluator |
| E1 | MERIT index sampling, index PE, index Gaussian, unique duplicate loss |
| E2 | Change only sampling to nearest-mm with declared spacing |
| E3 | Independently ablate physical PE, physical weighting, TEP/MER/PCG or pooling |

Keep split, seeds, preprocessing, threshold policy and training budget matched
when isolating one axis. No learned spatial alignment/agreement extension is
implemented. Full-CV and external benchmark results remain outstanding.
See [validation](validation.md) for actual software/real-data evidence.
