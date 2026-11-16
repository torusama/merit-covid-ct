# MERIT training, inference and evaluation

This guide owns the runnable MERIT workflow. Model equations/provenance live in
[the MERIT model section](architecture.md#merit-unetr-model); shared data/metric contracts live in
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

Use the [current Colab attention pilot](../notebooks/colab_mer_attention_pilot.ipynb)
for bounded tests on selected real cases. Its results are diagnostic and do not
establish segmentation quality. Follow the [factorized Kaggle guide](merit_pipeline.md#kaggle-fold-0-native-512-with-factorized-mer)
for the current training package and checkpoint-resume procedure; the
[original Kaggle guide](merit_pipeline.md#kaggle-training-native-512-fold-0) describes the earlier run.
Historical cloud measurements and their limitations remain in [validation](validation.md).

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
Three primary cases lack spatial units. The shared reader rejects unknown units
by default, while `merit_index.yaml`, `merit_native512.yaml` and `merit_colab_256.yaml` explicitly set
`assume_mm: true`. The CLI's `--assume-mm` can enable the same assumption for
other configs. This is not source verification or an intensity/HU conversion,
and the setting must match when resuming.

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

## Colab training script

`scripts/train_colab.py` provides a separate Colab training path using case-local
sampling. The older 256-grid configuration resizes inputs and needs its own
protocol assessment; it is not the current native-512 Kaggle workflow. Training
uses TRAIN and checkpoint selection uses validation; the test partition is not
used to choose settings.

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

---

## Kaggle fold 0: native 512 with factorized MER

Đây là run mới, dùng `configs/merit_native512_factorized.yaml`. Input vẫn gồm
11 lát native 512×512 và MER vẫn attention trên toàn bộ context. Lựa chọn
`factorized_mer: true` thay phép tính bias XY bằng biểu diễn tương đương về mặt
đại số; FP16 có thể làm khác kết quả do rounding. Config được lưu trong
checkpoint và fingerprint: không resume trực tiếp từ checkpoint native512 cũ.

Colab T4 với Torch 2.14.1, ba ca TRAIN A-0025/A-0295/A-0387: 12 kiểm tra
module CPU/CUDA pass; mỗi nhánh có 1 warmup + 6 measured optimizer updates,
khởi tạo giống nhau. Trung bình bản đối chứng **13,851 s/step**, factorized
**9,764 s/step**, giảm **29,5% thời gian/step**; peak allocated lần lượt
4693,60 và 4696,67 MiB. Không skip update; pilot checkpoint và CT-only master
inference pass. Đây là đo nhỏ trên Colab, không phải benchmark Kaggle, không
có validation Dice hoặc bằng chứng hội tụ.

Để bảo vệ khác biệt phiên bản GPU/Torch, mỗi phiên Kaggle chạy trước 4 kiểm tra
output/gradient trên chính GPU của phiên, bao gồm AMP, masked context và
activation recomputation. Kết quả nằm trong `factorized-gpu-guard.json`; nếu
FAIL, training không được bắt đầu. Run mới còn chạy một pilot 2 update bằng
production Trainer và một CT-only full-volume inference trước khi train.

### Files đã chuẩn bị

Thư mục `outputs/kaggle_factorized_fold0_verified/` chứa:

- `kaggle_train.py` và `kernel-metadata.json`: gói private Kaggle.
- `source_snapshot.zip`: snapshot source đã đóng băng; giữ lại cho resume.
- `package_provenance.json`: SHA-256 và config của snapshot.

Tạo mới một phiên sau khi xem lại gói, từ PowerShell tại repo root:

```powershell
.\tools\kaggle-cli\Scripts\kaggle.exe kernels push -p outputs/kaggle_factorized_fold0_verified --accelerator NvidiaTeslaT4
```

Theo dõi bằng `kernels status vtan2405/merit-factorized-fold0` và `kernels logs
vtan2405/merit-factorized-fold0 --follow`. Đây là một kernel riêng; không sửa
hoặc dừng `merit-native512-fold0-recovery` đang tồn tại.

### Thời gian và checkpoint

Config vẫn dùng toàn bộ **8.810 master** mỗi epoch, không downsample hay thay
sampling. Nếu tốc độ Colab 9,764 s/step giữ nguyên, riêng train cần khoảng
**23,9 giờ/epoch**; máy Kaggle có thể nhanh/chậm khác và validation còn tốn
thời gian. Đây không phải 100-epoch study đã hoàn tất. Phải đo trên Kaggle và
đánh giá native-volume validation trước khi kết luận chất lượng model.

Mỗi phiên dành tối đa 7 giờ cho train, ghi `in_progress.pt` mỗi 100 batch và
khi chạm deadline. Nếu dừng giữa epoch, `run-status.json` trỏ tới
`checkpoints/in_progress.pt` kể cả khi `last.pt` của epoch trước tồn tại.
Đây là file trong output của phiên; chỉ có thể dùng ở phiên tiếp theo sau khi
Kaggle lưu output. Bắt buộc bảo toàn source snapshot và dữ liệu.

### Resume sau khi Kaggle lưu output

Xem output của phiên, xác nhận có `MERIT/fold-00/checkpoints/in_progress.pt`,
rồi tạo phiên kế tiếp với chính snapshot đã lưu:

```powershell
python tools/prepare_kaggle_factorized.py --output outputs/kaggle_factorized_fold0_verified --resume-from vtan2405/merit-factorized-fold0
.\tools\kaggle-cli\Scripts\kaggle.exe kernels push -p outputs/kaggle_factorized_fold0_verified --accelerator NvidiaTeslaT4
```

Lần resume kế tiếp dùng `--resume-from` là slug của phiên vừa lưu, và đặt
`--kernel-id vtan2405/merit-factorized-fold0-resume2` hoặc slug mới khác với
nguồn. Công cụ đối chiếu SHA-256 snapshot với `source_provenance.json` của
phiên trước; runner kiểm tra hash nội dung CT/GT trước khi cập nhật timestamp
nguồn. Không thay config, dữ liệu, sampler hoặc code giữa các phiên resume.
Đường resume qua Kaggle của config mới chưa được kiểm chứng end-to-end; test
CPU production đã kiểm tra dừng/tiếp tục giữa epoch và load model inference.

Không tự giảm số lát/epoch rồi xem đây là cùng protocol. `samples_per_volume`
có thể dùng cho thí nghiệm riêng, nhưng làm thay đổi phân bố mẫu, số update,
PCG warmup và lịch validation; cần thiết kế và đánh giá trước khi chọn làm
phương án chính.

---

## Kaggle training: native 512, fold 0

The private run uses the existing `vtan2405/merit-covid19-20` dataset and a
snapshot of the local source. The three cases with missing spatial units are
explicitly assumed to be millimetres; this does not verify their source units.
It runs a two-step real-case pilot before attempting one training epoch.

### Output files

After a successful Kaggle saved run, open its Output tab and look under
`MERIT/fold-00/`:

- `source_provenance.json`: source archive hash, dataset and split identity.
- `pilot-report.json`: diagnostic pilot results, not a trained model.
- `run-status.json`: `epoch_complete`, `paused_at_checkpoint`, or `training_error`.
- `checkpoints/in_progress.pt`: latest periodic checkpoint for an incomplete epoch.
- `checkpoints/last.pt`: most recently completed epoch and validation.
- `checkpoints/best.pt`: best validation checkpoint, when an epoch completes.
- `checkpoints/history.jsonl`: completed epoch metrics.

The trainer saves `in_progress.pt` every 25 batches and after the final batch.
Each write uses a temporary file and atomic replacement. The state includes
model, optimizer, RNG, sampler replay position and partial training totals.
If the seven-hour training budget expires, the run saves at the next completed
batch and exits normally so Kaggle can save its output. Validation may take
additional time. An unexpected machine failure, forced termination, interrupted
checkpoint write, or Kaggle failing to save the version can still lose output;
periodic files inside the running machine are not independent cloud backups.

### Continue a paused run

After the run finishes and Kaggle has saved its output, confirm that its Output
contains `MERIT/fold-00/checkpoints/in_progress.pt`. Prepare the next private
kernel using that run as an input source, then push it:

```powershell
python tools/prepare_kaggle_run.py --resume-from vtan2405/merit-native512-fold0
kaggle kernels push -p outputs/kaggle_native512_fold0 --accelerator NvidiaTeslaT4
```

The resume job requires exactly one `in_progress.pt` in its attached inputs.
It verifies the checkpoint's frozen configuration and source identity; it
fails rather than silently continuing if paths or source metadata change.
This cross-session Kaggle resume path requires an actual saved-run validation.
Avoid starting two runs from the same checkpoint concurrently.

`scripts/kaggle/run_training.py` contains the cloud workflow;
`tools/prepare_kaggle_run.py` packages the current local source and builds the
Kaggle metadata. The generated upload file and metadata are under
`outputs/kaggle_native512_fold0/` and are not hand-edited.

## Fast execution continuation: epoch 1 only

The prepared continuation is `outputs/kaggle_fast_epoch1/`, kernel
`vtan2405/merit-fast-epoch1`. It uses the completed epoch-0 checkpoint from
`vtan2405/merit-native512-fold0-epoch0-finish` and the verified exact-byte
checkpoint SHA-256. Its immutable source snapshot is a minimal successor to
the original source; retain it with `package_provenance.json` for future resume.

The explicit epoch-boundary migration permits only total epochs=2,
deterministic=false, factorized_mer=true, mer_recomputation=false and query
chunk size=4096. Learning rate, loss, preprocessing, data identities, partitions,
sampling, scheduler, scaler and warmup policy are retained. Across Kaggle VMs,
source timestamps may be rebound only after verifying the full CT/mask content
hash manifest; other source identity changes fail. The new config, protocol
fingerprint and migration record are persisted in every new checkpoint.

Before training, the runner verifies exact CUDA-loaded optimizer moments,
scheduler and scaler. It takes two bounded native training updates at the
production epoch-1 PCG ramp, checks finite losses/gradients and effective
updates, then restores the original model, optimizer, scheduler, scaler, RNG
and loader state. Actual epoch 1 starts from the saved epoch-0 state, not the
diagnostic updates. Failure of a guard prevents full training.

The production Trainer then runs epoch index 1 with total budget 2 and no
application time deadline. It saves `in_progress.pt` every 100 batches,
evaluates the full saved 32-case validation partition, saves `last.pt`, updates
`best.pt` only if Dice improves, and stops before epoch 2. The prior best
checkpoint and original epoch-0 history are retained in the new outputs.
Validation logs show case starts/completions and slice progress. Kaggle's own
runtime limits still apply; periodic checkpoints remain necessary.

## Interactive local epoch-0 demo

Run `python -m streamlit run scripts/demo_merit.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false`
and open `http://127.0.0.1:8501`. Install the optional UI dependency with
`python -m pip install -e ".[unetrpp,demo]"` when preparing another machine.
The actual completed `last.pt` must remain at the downloaded epoch-0 output path.
Its SHA-256 is checked before loading; no model is initialized from random weights.

The interface lists all 32 validation cases and their previously recorded full
volume Dice. It starts at A-0319 as an explicitly identified high-Dice example;
the validation mean 0.2821 and remaining quality limits are shown in the UI.
Users can select a master slice, request CT-only native prediction and adjust
overlay opacity/threshold for display. Ground truth is supplied only to the
viewer and slice metrics, not to the prediction Dataset. A configurable display
threshold does not revise the frozen 0.5 validation protocol. Slice Dice must
not be confused with native-volume Dice or held-out cohort performance.

Local inference uses FP32 on the NVIDIA GPU, the exact trained PCG state,
full context and factorized MER. The demo keeps query chunk 128 to bound memory
on the local 4-GB GPU; this is independent of the frozen Kaggle training package.
Predictions are cached per case/master under `outputs/demo_epoch0/`. The tested
A-0319 master 26 produced finite 512x512 probabilities in 14.31 seconds after
20.6-second initial loading on the local GTX 1650. Initial cached-slice Dice
is 0.790; this selected slice is not representative of the cohort mean.
The UI's case selector, threshold slider and label visibility were checked
using Streamlit AppTest, and the running server health endpoint returned OK.
