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
`img_size` must match explicit preprocessing `slab_size/size_hw`.

`configs/merit_unetrpp.yaml` is model/loss-only smoke configuration; it is not a
full experiment. Use `merit_index.yaml` or `merit_physical.yaml` for this CLI.

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

`engine.training.Trainer` provides AdamW, CUDA FP16 AMP, gradient clipping,
seeding, deterministic settings, ReduceLROnPlateau and early stopping.
It logs individual loss components and skipped AMP optimizer steps.
PCG `set_epoch(epoch)` is called every epoch, default warmup 30.

Best checkpoint selection uses validation **group-macro full native-volume
Dice**, not paper LIDC master-ROI Dice. Default threshold 0.5 is declared in the
experiment; it is not selected from internal/external test labels.

`best.pt` and `last.pt` save model, optimizer, scheduler, scaler, epoch,
best metric, early-stop counter, Python/NumPy/Torch/CUDA RNG, loader shuffle
state, PCG ramp, effective config, split identity/fingerprint/hash, resolved
source paths and runtime. Checkpoints replace atomically.
Existing runs require explicit resume.

Resume verifies the frozen protocol. Only extension of the epoch budget is
permitted; model/loss/preprocessing/sampling/fold/path changes are rejected.
Moving a dataset requires a new protocol or verified migration.
Exact interrupted/uninterrupted replay is tested at epoch boundaries, not
mid-batch. CPU/CUDA type switches are rejected; cross-hardware/version bitwise
reproducibility is not promised. Checkpoints do not hash every source voxel:
version and pin upstream data separately. Load only trusted local checkpoints.

## Inference and metrics

`predict_slab_case` requires eval mode, predicts every canonical master once
without reading GT, converts two-class logits using `sigmoid(fg-bg)`, restores
probabilities to native XY, then constructs `VolumePrediction`.
Slice assembly rejects duplicate/missing slices or changed case/geometry.
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
py scripts/merit_pipeline.py check-split --config configs/merit_index.yaml
# Real data inspection; start small.
py scripts/merit_pipeline.py audit --config configs/merit_index.yaml --headers-only --limit 1 --output outputs/headers.json
py scripts/merit_pipeline.py audit --config configs/merit_index.yaml --limit 1 --output outputs/voxel_audit.json

# Temporary synthetic train/validate/save/reload/resume.
py scripts/merit_pipeline.py smoke --device cpu --output outputs/cpu_smoke.json
py scripts/merit_pipeline.py smoke --device cuda --output outputs/cuda_smoke.json

# Development fold training and epoch-boundary resume.
py scripts/merit_pipeline.py train --config configs/merit_index.yaml --fold 0 --output outputs/e1/fold0
py scripts/merit_pipeline.py train --config configs/merit_index.yaml --fold 0 --output outputs/e1/fold0 --resume outputs/e1/fold0/last.pt
# Checkpoint-based validation and internal test.
py scripts/merit_pipeline.py evaluate --checkpoint outputs/e1/fold0/best.pt --partition validation --device cuda --output outputs/e1/fold0/validation.json
py scripts/merit_pipeline.py evaluate --checkpoint outputs/e1/fold0/best.pt --partition test --device cuda --output outputs/e1/fold0/test.json
# Long-running research experiment; not part of a software smoke check.
py scripts/merit_pipeline.py cv --config configs/merit_index.yaml --output outputs/e1_cv
# Full reference forward/backward, without optimizer state.
py scripts/merit_pipeline.py benchmark --config configs/merit_index.yaml --device cuda --batch-sizes 1 2 --output outputs/merit_memory.json
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
# Fold-0 TRAIN voxels only; all-fold split checking uses metadata only.
# The explicit spatial-mm assumption is recorded, not applied to source files.
py scripts/merit_preflight.py inspect --config configs/merit_index.yaml --fold 0 --sizes 128 256 --assume-mm --block-depth 8 --output outputs/preflight_new/data

# Separate processes isolate allocations/OOM; do not run concurrently.
py scripts/merit_preflight.py profile --config configs/merit_index.yaml --size 128 --batch-size 1 --output outputs/preflight_new/gpu128_b1.json
py scripts/merit_preflight.py profile --config configs/merit_index.yaml --size 128 --batch-size 2 --output outputs/preflight_new/gpu128_b2.json
py scripts/merit_preflight.py profile --config configs/merit_index.yaml --size 256 --batch-size 1 --output outputs/preflight_new/gpu256_b1.json
py scripts/merit_preflight.py profile --config configs/merit_index.yaml --size 256 --batch-size 2 --output outputs/preflight_new/gpu256_b2.json

# Actual TRAIN CT: four optimizer attempts, fresh Trainer resume, CT-only
# inference, complete reconstruction and original-shape/affine mask export.
# A-0025 is a fold-0 TRAIN case with explicit mm; no unit assumption needed.
py scripts/merit_preflight.py pilot --config configs/merit_index.yaml --fold 0 --case-id volume-covid19-A-0025 --size 128 --steps 4 --device cuda --output outputs/preflight_new/pilot128
```

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
