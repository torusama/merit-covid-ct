# Validation evidence and limitations

This is the single maintained evidence summary. Historical source audits, merge
notes, implementation roadmaps and MERIT repair reports have been consolidated
here and into [CHANGELOG.md](../CHANGELOG.md). Stored verification JSON files
remain historical snapshots: old filenames/counts/environments inside them are
not claims about the latest checkout.

## 2026-11-13 production native-512 integration

The historical follow-up notebook `colab_main_integrated.ipynb` targeted published integration
commit `ac46169`. Its bounded orchestration uses real complete-manifest sources
and the actual Trainer optimizer/checkpoint/resume methods, then one CT-only
master prediction. The complete notebook worker/export path passed locally on
a native compact CPU fixture; the same three-update/checkpoint/resume helper
also passed on the local GPU with AMP. Focused notebook/bootstrap/worker checks
passed 19 tests with one CPU-environment CUDA skip, and the CUDA workflow passed
separately in 11.86 seconds. Native-512 T4 results for this integrated notebook
are pending. The earlier cloud reports used separate experiment helpers and
must not be relabeled as integrated-Trainer results.

The user selected native full slices and the validated all-region recomputation
strategy for the main trainer. `configs/merit_native512.yaml` now sets native
`size_hw: null` and `training.activation_checkpointing: true`; `train` and `cv`
default to it. Native/model XY mismatch fails from headers rather than resizing.
The older 128-grid configs remain explicit research alternatives with the
known lesion-retention issue; the primary path avoids that transformation.

`engine/activation_checkpointing.py` implements the same MER query-block and
stateless-region recomputation as the frozen T4 helper. BatchNorm regions stay
outside recomputation and dropout RNG is preserved. The trainer applies this
before optimizer construction; loading a checkpoint restores its execution
setting, and resume rejects changing it. Disabled legacy schema-2 configs keep
their fingerprint. Frozen cloud notebooks and helper hashes remain unchanged.

Production regressions compare original/frozen-helper outputs, losses, input
and all parameter gradients, and buffers on compact fixtures. They include
off-center masters, physical positions and partial query chunks. A native
512-grid single-pixel GT test forbids resizing and verifies unchanged source
bytes. Both shuffle modes exercise native compact training, native-volume
validation, exact epoch-boundary resume and checkpoint-driven CLI evaluation.
Full native-512 cohort training/validation remains unrun; no quality claim is
inferred from these tests or the earlier two-step pilots. Resume still saves
only completed epochs, and the three unknown-unit cases need an explicit policy.

Verification: the full CPU suite passed **397 tests**, with **two CUDA-only
skips**, five warnings and no failures in 140.21 seconds. The new compact
production-checkpointing CUDA AMP equivalence test passed separately on the
local GTX 1650 Max-Q in 11.53 seconds. The production `native-input` command
also passed on real A-0025 with the updated no-resize config, preserving CT/GT
source hashes and native pixel values. Local reports are
`outputs/production_checkpointing_20261113_pytest.xml` and
`outputs/production_native512_input_20261113.json` (ignored). A portable numeric
summary is committed in
[production integration evidence](../analysis/merit_native512/production_integration_20261113.json).

## Earlier 2026-11-13 review and runtime repairs

Handoff update: the user-supplied MER-only report
`20261113_090150_4f13abdd.zip` passes two native A-0025 updates on Tesla T4.
Source/config/initial-weight identities match the previous all-region trial;
both losses and final parameter hashes match exactly. Mean step time is
13.624661 s versus 13.844556 s, while peak allocation increases from 4649.68
to 9219.69 MiB. The observed 1.59% timing reduction across two separate-session
steps is inconclusive; retain the all-region overlay as the pilot reference.
See the [portable cloud evidence](../analysis/merit_native512/colab_evidence_20261113.json).
The identified runtime repairs and bounded pilots are complete; team-member
training results are awaited. At that snapshot production integration and the
older 128-grid issue were still open. The subsequent production integration is
described above; the 128-grid issue remains confined to those explicit configs.

Pre-integration handoff verification: **389 passed, one CUDA-only skipped**, zero failures
or errors in 108.048 seconds for the full CPU suite. The earlier counts below
describe intermediate snapshots. The compact CUDA AMP test passed separately
in the local GPU environment; the CPU suite itself does not validate CUDA.
Selected user-supplied Colab report JSON, environment versions, original ZIP
SHA-256 and the latest CPU test summary are committed as
[portable evidence](../analysis/merit_native512/colab_evidence_20261113.json).
Raw CT/GT, original report ZIPs and model weights remain outside Git.

The review started from main `e3493dd9f2dac704d2def6225d76f19df3262ec4`.
The initial CPU suite passed 340 checks with one CUDA-only skip; passing that
suite did not detect the failures reproduced below. The full user-supplied PDF
review source is identified in [the model guide](architecture.md#source-and-task-transfer).
TEP/MER/PCG equations, insertion points and additional parameter counts were
checked; no confirmed core-formula defect justified changing those modules.

After repair, `py -m pytest -q` passed **363 checks**, with **one CUDA-only
skip**, five warnings and no failures in 114.07 seconds on Python 3.13.16 /
Torch 2.15.0+cpu. Warnings comprise one upstream Torch deprecation and four
explicit intensity-provenance warnings in compatibility tests. No skip or
assertion was added to conceal a failure. The 21 new runtime regressions also
passed separately. `pip check` and `git diff --check` passed. This machine did
not validate CUDA behavior. JUnit output is local under
`outputs/code_fixes_20261113_pytest.xml` (ignored, not a committed artifact).

| Reproduced issue | Implemented correction and check |
| --- | --- |
| Redirected GT with unchanged basenames was accepted on resume | Schema 2 binds manifest bytes, data config, resolved CT/GT paths and source stats; changed GT paths/content and stale bindings are rejected |
| Infinite master logits silently became 0/1 probabilities | Validate floating logits, expected shape and finiteness before sigmoid; inject NaN and both infinities |
| Config accepted thresholds that metrics rejected after training | Reuse the strict metric validator before training; test endpoints, out-of-range and nonfinite values |
| Global slice shuffle defeated the one-volume cache | Explicit volume-block shuffle in experiment configs; verify exact coverage, bounded cache fills, Windows workers and saved RNG replay |
| Guide commands referenced deleted example scripts | Restore maintained script entry points, update commands and execute compact model-only backward checks; validate documented Python command paths |

Schema-1 models remain loadable for CT-only inference, but cannot safely resume
with their incomplete source identity. Both slice and volume shuffle modes have
an end-to-end uninterrupted-versus-resumed optimizer test with exact final
model-state comparison. Source stats are not a full voxel-content checksum;
changes that preserve file size and modification time are outside that binding.

Input/resize is explicitly deferred at the user's request. Whole-slice resize
can erase small lesions and flip derived presence/salience labels; that defect
remains unresolved. The COVID sampler also differs from the paper's inward
boundary-window shifts, and whole-slice resize differs from its ROI crops.
No sampling/input/auxiliary-label redesign was applied in this repair. Shared
preprocessing, model/loss equations, canonical split and stored cohort reports
are unchanged. No full real-CT training or new clinical evaluation was run.

## 2026-11-13 full native XY feasibility (route 1)

After the earlier resize discussion, the user authorized trying full native XY
first. The separate `configs/merit_native512.yaml` changes only the model/input
grid to 512x512 and explicitly selects CUDA; feature size, depths, TEP/MER/PCG,
sampling, losses, native intensity, units and split policies are retained.
This does not fix lesion disappearance in the existing 128x128 experiments.

| Check | Observed result |
| --- | --- |
| Header-only source XY check | All 199 CT/GT pairs have 512x512 XY; no cohort voxel decoding |
| Real A-0025, fold 0 TRAIN, central native slab | PASS; input `[1,11,512,512]` before batching; CT/GT exactly match selected native source pixels; XY interpolation forbidden; source SHA-256 unchanged |
| Compact CUDA AMP forward/loss/backward regression | 1 passed, no skips, 21.02 s; synthetic input, no optimizer |
| Full 512 grid, batch 1, AMP, 90% physical VRAM allocator budget | OOM during encoder EPA forward; 0 optimizer updates |
| Full 512 grid, batch 1, AMP, 100% physical VRAM allocator budget | Same OOM during encoder EPA forward, before TEP/MER/PCG, loss or backward; 0 optimizer updates |
| Model parameter count | 128 grid: 29,836,242; 512 grid: 174,043,602; grid-dependent positional/projection weights also grow |

After adding the native-input diagnostic, the full original CPU-environment
suite passed **365 tests**, with one CUDA-only skip, five warnings and no
failures in 110.77 seconds. That CUDA test was executed separately in the GPU
environment and passed as recorded above; a full GPU-environment suite was not
run. Two new regressions cover exact native input/tiny-lesion preservation and
rejection of resizing or validation/test case selection. The full-grid OOM is
a separately recorded feasibility failure, not a successful training test.

Hardware: GTX 1650 Max-Q, total 4095.69 MiB, driver 511.65. The separate ignored
`.venv-gpu` uses Python 3.13.16, Torch 2.7.1+cu118, CUDA runtime 11.8, MONAI
1.5.2 and einops 0.8.2; the original CPU environment is retained. This GPU setup
ran the previously skipped CUDA test. Its `pip check` passed. The
[official PyTorch version table](https://pytorch.org/get-started/previous-versions/#v271)
provides that CUDA wheel; no GPU driver was changed.

At the full-budget failure, peak PyTorch allocation was 2368.55 MiB and peak
reservation 2422.00 MiB; another 176 MiB allocation failed. These allocator
counters exclude some CUDA/library/display memory and are not total GPU usage
or a complete estimate of memory needed for training. The forward pass did not
finish, so no step timing or accuracy comparison can be reported. The planned
warmup/measured steps were not completed. This establishes infeasibility of the
tested setup, not a universal claim about all hardware or optimization methods.

Local raw reports are in `outputs/native512_20261113/`: `input.json`,
`cuda_test.xml`, `profile_512_attempt2.json` (90% budget),
`profile_512_full_budget.json`, and `summary.json`. They are ignored local
artifacts. `profile_512.json` is an earlier profiler device-index setup failure,
corrected before the two OOM measurements; it is not a model feasibility result.
No full real-CT training, learned diagnostic checkpoint, accuracy evaluation,
architecture reduction or route-2 crop/tiling was performed in this experiment.

To reproduce the isolated GPU environment and checks from the repository root:

```powershell
.venv\Scripts\python.exe -m venv --system-site-packages .venv-gpu
.venv-gpu\Scripts\python.exe -m pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
.venv-gpu\Scripts\python.exe -m pip install -e ".[unetrpp,test]"
.venv-gpu\Scripts\python.exe -m pytest tests/test_merit_pipeline.py::test_cuda_amp_real_forward_loss_backward -v -rs
.venv-gpu\Scripts\python.exe scripts/merit_preflight.py native-input --config configs/merit_native512.yaml --output outputs/native512_recheck/input.json
.venv-gpu\Scripts\python.exe scripts/merit_preflight.py profile --config configs/merit_native512.yaml --size 512 --batch-size 1 --device cuda --warmup 1 --repeats 1 --memory-fraction 1.0 --output outputs/native512_recheck/profile.json
```

The full profile is expected to exit with failure/OOM on the tested machine.
Use a new output path when repeating diagnostics; reports are not overwritten.

## Verification milestones

The real-case notebook's workflow was executed locally with native 32x32
synthetic NIfTI fixtures against the pinned public source: input ZIP upload was
simulated, an actual GPU optimizer update passed, the CT-only inference guard
passed and a small report ZIP was produced. The delivered notebook still pins
the real A-0025 bytes and native 512x512 grid. Four pilot regressions cover
native tiny-lesion preservation/effective updates/CT-only inference, wrong GT,
non-TRAIN membership and modified splits; upload tests cover member validation
and cached reuse. No original source images or canonical splits changed.

The user subsequently supplied `20261113_081310_7f8b19a4.zip`: the real A-0025
native-512 pilot PASSED on Colab T4 with two AdamW updates, no AMP skips, finite
gradient norms and changed parameter hashes. Native CT/GT pixels and source
SHA-256 matched; original/canonical LPS geometry and affine were identical, with
no XY interpolation or intensity normalization. One CT-only master prediction
passed the GT-read guard; full-volume reconstruction and accuracy were not run.
Peak PyTorch allocation was 4649.68 MiB, reservation 5072 MiB; update times were
13.96 and 13.73 seconds, total pilot time 54.96 seconds. Both helper byte hashes
matched the validated local sources. Compact equivalence tests again recorded
zero output/input-gradient/parameter-gradient error and identical norm buffers.

A header-only recount gives fold 0 TRAIN 127 volumes / 8810 masters. Applying
the two-step mean of 13.84 seconds to every batch-1 master gives about 33.9 hours
per epoch, excluding cohort I/O and validation. This is a preliminary projection
from one case, not measured epoch throughput. Longer multi-case testing and
mid-epoch progress persistence are needed before scheduling full cloud training.

An isolated execution-only checkpoint trial was added after the user supplied
Colab T4 reports confirming native 512 OOM in MER at both 90% and 100% allocator
budgets, with no optimizer updates. Production model files were not edited.
The experimental helper passed CPU float64 MER equivalence, CPU compact-model
equivalence, and GPU FP32/AMP compact-model equivalence. Measured maximum
output/input-gradient/parameter-gradient errors in the GPU fixture were zero;
normalization buffers matched exactly. Three regressions also verify numerical
equivalence and reduced unique saved-tensor storage for the MER fixture.

All cells of the checkpoint notebook were executed locally against its pinned
public source at grid 128: two optimizer updates passed, peak PyTorch allocation
648.78 MiB versus 1020.73 MiB in the prior uncheckpointed local run. These are
short synthetic probes, not an accuracy or statistically robust speed benchmark.
The 512 trial on the local 4 GiB GPU still OOMed in decoder EPA before MER.
The user subsequently supplied `20261113_061303_03f5c52a.zip` from Colab T4:
the optimized native-512 synthetic trial passed two optimizer updates with no
AMP skips, peak PyTorch allocation 4650.55 MiB, peak reservation 4958 MiB and one
measured post-warmup step of 13.70 s. Compact equivalence outputs/input gradients/
parameter gradients had zero recorded error in FP32 and AMP; normalization
buffers matched. The helper bytes matched the validated local SHA-256
`8c2bb3133e96ff2bb5a20e080c30eb446ffe0b62e05608ff5132acb6730b26f8`.
This establishes bounded synthetic feasibility, not real-data learning quality.

The revised Colab/Kaggle notebook was executed cell-by-cell locally using the
isolated GPU environment and a cached checkout of its pinned public commit.
It skipped pip because dependencies were already satisfied. Full 128x128
profiling completed two optimizer updates (peak allocation 1020.73 MiB), full
512x512 profiling reported encoder OOM and zero updates, and the ZIP export
completed. This verifies the notebook workflow on the local GPU, not a run in
an authenticated Colab or Kaggle session. Cold network clone/package download
and cloud hardware allocation were not timed. Three isolated regressions cover
syntax/generated worker compilation, rejection of missing CUDA before setup,
and dependency installation/cache behavior with Torch/NumPy protected.

| Stage | Recorded result | Scope |
| --- | --- | --- |
| Main foundation, 2026-11-07 | 56 passed, 1 skipped | Core synthetic pipeline; Torch unavailable then |
| Duong LPS/native-intensity data pipeline | 72 passed | Geometry, scaling, native size and data tools |
| Main/Duong integration | 135 passed, no skips | Shared core and both public APIs |
| Standalone UNETR++ integration | 188 passed, no skips | 53 new model checks plus prior regressions |
| Native baseline patch/output integration | 262 passed, no skips | 74 additional input/reconstruction checks |
| MERIT pipeline repair, 2026-11-08 | 325 passed, 0 failed/skipped, 4 warnings; 143.69 s | Full suite, including actual CUDA AMP check |
| Repository/documentation cleanup | 332 passed, 0 failed/skipped, 4 warnings; 133.32 s | Seven added hygiene/report checks plus all existing regressions |
| Main review/runtime repair, 2026-11-13 | 363 passed, 1 CUDA-only skip, 5 warnings; 114.07 s | Source binding, inference/config guards, volume-cache/RNG replay, repaired CLI commands and prior regressions |

The four MERIT-repair warnings explicitly report unverified nonnative intensity
provenance in compatibility tests. Earlier overlapping pytest/MONAI/indexer
processes exhausted Windows resources (WinError 1450/allocator errors).
No assertions were removed or skipped to hide this; the final sequential
325-test run passed. Avoid overlapping heavy test/index jobs on this 8 GB host.

## Confirmed repairs and tests

| Confirmed issue | Repair | Regression coverage |
| --- | --- | --- |
| CUDA autocast rejects probability BCE | Explicit FP32 BCE with autocast disabled | Full compact CUDA forward/loss/backward |
| CT-only inference could still open GT | Strip labels and return before mask header access | Missing-GT Dataset and native-volume inference |
| Dataset supplied only master GT to slab loss | True selected-index mask_slab plus validity | Slice/target correspondence and collation |
| MERIT incompatible with baseline patch API | Separate master-slab inference and explicit rejection | Patch rejection, two-class master conversion |
| New metadata destabilized CSV column order | Canonical metadata field ordering | Reversed audit-key insertion regression |
| Old mask paths assumed IDs equaled filenames | Shared manifest path resolution | Legacy preparation/audit/viewer tests |
| Downloader module imports failed | Lazy package-aware sibling imports | File/module --help invocation |

Physical sampling/PE/weighting are explicit extensions, not evidence that
index-only baselines were mathematically wrong. Alignment max pooling is a
documented choice, not a confirmed algorithm bug.

## Synthetic versus real evidence

Synthetic tests exercise nearest-mm reference selection at 1/1.5/3/5 mm,
boundaries, short stacks, duplicates, synchronized true masks, rotated/oblique/
sheared/metre affines, unknown units, inverse geometry, Windows workers,
all eight module ablations, extreme sigmas, small lesions, finite gradients,
actual optimizer updates and exact epoch-boundary checkpoint replay.
Clinical accuracy cannot be inferred from synthetic Dice values.

Stored real-data evidence includes:

- [Main/Duong verification](../analysis/merge/verification.json): six pre-merge
  real slabs from LPS A-0025/LPI A-0161, at first/middle/last masters, matched
  both APIs in values, masks, indices, world positions and metadata.
  All 199 primary metadata/range/count records matched the saved data audit.
- [UNETR++ source verification](../analysis/unetrpp/verification.json):
  reference-grid state-dict and all three outputs matched pinned upstream
  in the same dependency environment.
- [Baseline input verification](../analysis/unetrpp_input/verification.json):
  one real A-0025 patch forwarded without GT or parameter changes.
  Complete-volume reconstruction was tested synthetically, not on the real cohort.
- MERIT repair checked the existing 199-case split and audited one real CT,
  A-0003, shape 512x512x291. Explicit 512->128 resize lost 26 entire 2D
  components across 18 slices, but no completely positive slice.
  This is a preprocessing warning, not a reason to silently discard the case.

Earlier cohort audits reported 199/199 valid with the declared mm assumption,
and 196/199 valid under strict units. The newer MERIT repair did not rerun
voxel audits over the full cohort.

## MERIT execution and GPU checks

CPU and CUDA CLI smoke checks performed compact synthetic epoch-0 optimizer/
native-validation/save, checkpoint reload and epoch-1 resume. PCG ramp was
0 then 1/30. Temporary CT/checkpoints were removed by the smoke fixture;
local JSON evidence remains under outputs/merit_repair_*_smoke.json.
No real COVID training was performed at that milestone. The newer bounded
real-CT preflight below is not a full training experiment.

Reference GPU model: input [B,1,11,128,128], feature_size=16,
depths=(3,3,3,3), 29,836,242 parameters, Torch 2.5.1+cu118,
GeForce GTX 1650 with 4 GiB VRAM:

| Batch | Cold forward/backward | Peak allocated CUDA | Gradients |
| --- | ---: | ---: | --- |
| 1 | 16.736 s | 836,060,672 bytes (797.3 MiB) | finite |
| 2 | 1.490 s | 1,411,178,496 bytes (1345.8 MiB) | finite |

One iteration per shape, no warmup. Batch 1 includes first-use overhead; do not
infer that batch 2 is faster. These peaks exclude optimizer state, loader and
whole-CT inference. Compact optimizer smoke does not guarantee full-size
AdamW batch 2 always fits 4 GiB. Local evidence:
outputs/merit_repair_gpu_benchmark.json.

## Real-CT preflight, 2026-11-08

Local evidence is under `outputs/merit_preflight_fold0/`. Unlike historical
whole-cohort descriptive reports, this audit read CT/GT voxels from **only the
127 fold-0 TRAIN volumes**, using a bounded-memory proxy reader checked against
the shared reader. No validation/test CT or label was used to choose XY size.
The audit took 304.77 s and reported no modified source files, nonfinite values
or nonbinary labels. A-0599 required the explicitly recorded spatial-mm
assumption; this did not change production YAML or source headers.

Native scaled CT statistics over 2,309,488,640 voxels: min -9653, max 32737,
mean -883.404928, population std 735.307063, median -975, p1 -2048, p99 373,
p99.9 974. All scaled values were integer-valued, including the one float64
storage case A-0599, so the all-voxel histogram percentiles are exact.
The CT values at -2048 occupy about 20.5% of voxels; this may include acquisition
padding and must not be silently treated as calibrated air. Source literature
describes HU, but per-case DICOM calibration remains unverified. No HU window
or normalization was applied.

| TRAIN-only label geometry | 128 | 256 |
| --- | ---: | ---: |
| Entirely lost 2D 4-connected regions / 6295 | 325 | 186 |
| Native lesion voxels lost after nearest-exact down/up | 454378 (4.4185%) | 244979 (2.3822%) |
| Native voxels in entirely lost 2D regions | 996 | 274 |
| Positive slices becoming empty / 3278 | 34 | 10 |

Regions are not distinct 3D clinical lesions. The native-support roundtrip is
a geometric QC metric, not the lesion volume difference or model performance.
Per-case, per-slice and pooled CSV, histogram and three illustrations are saved.
All 15 within-fold partition-pair comparisons across the five saved folds have
zero case/group/path overlap. Group source is `volume_id_fallback`: patient
independence and content-duplicate absence are **not independently verified**.

Full reference resource tests on GTX 1650 4 GiB / Torch 2.5.1+cu118 used native
model defaults, CUDA AMP, AdamW, one warmup and two measured optimizer steps:

| XY / batch | Result | Peak PyTorch allocated / reserved MiB | Mean measured step s |
| --- | --- | ---: | ---: |
| 128 / 1 | PASS, 3 optimizer steps | 1020.73 / 1094 | 0.6969 |
| 128 / 2 | PASS, 3 optimizer steps | 1592.06 / 1688 | 1.1418 |
| 256 / 1 | CUDA OOM in forward; zero optimizer steps | 2349.36 / 2372 before failure | Not measured |
| 256 / 2 | CUDA OOM in forward; zero optimizer steps | 2357.37 / 2380 before failure | Not measured |

PyTorch peaks exclude non-allocator/driver memory and are not total device use
or the memory required to finish a failed step. Resolution-dependent backbone
parameters increase from 29,836,242 at 128 to 58,677,714 at 256. MER query
chunking bounds individual allocations, not all autograd history. No architecture
was silently shrunk to make 256 fit. These short profiles do not establish
full-run throughput or long-run stability.

Actual fold-0 TRAIN A-0025 pilot at 128, batch 1, 11 index slices, raw scaled
intensity: four effective optimizer steps, zero AMP skips, finite losses/final
gradients and changed weights. Fresh Trainer resume matched model bytes and
AdamW step counters, restoring start_epoch=1. CT-only inference (GT load guard)
assembled all 61 masters, restored 512x512x61 and exported/reloaded a binary
prediction with original affine. Probabilities were finite in [0.071109,
0.952300]. Training took 8.90 s; complete-case inference/restoration/export took
8.01 s; total pilot 24.22 s, peak allocated 1021.62 MiB. The diagnostic checkpoint
is not validation-selected, and prediction quality was not assessed.

Recommendation for the **current configuration/hardware**: 128 batch 1 is the
conservative executable baseline; batch 2 was also measured successfully.
256 has better TRAIN label retention but failed the actual memory gate. If
tiny-lesion retention is a priority, native-XY MERIT patches are a sensible next
engineering experiment, not an already-tested substitute: the existing baseline
patch inference explicitly rejects MERIT and needs a MERIT-specific tiled path.
Do not declare either resolution optimal from this audit; compare candidates
using TRAIN/validation only, with native metrics and a frozen final protocol.

For normalization, begin with an explicitly implemented/frozen, unit-agnostic
train-global z-score candidate `(x + 883.404928) / 735.307063` for fold 0; these
are TRAIN-only moments before XY interpolation. This is a proposal, not a
policy applied to the pilot, and must be refit for each fold. A robust candidate
clips at TRAIN empirical p0.1/p99.9 (-2048/974 here), then **refits** mean/std on
clipped TRAIN values. Do not reuse unclipped moments after clipping, relabel a
percentile clip as a verified HU window, fit on test data, or silently remove
the -2048 peak. Validate the handling of padding/outliers and persist the exact
normalization implementation, order and fitted parameters before full training.

The existing four intensity modes do not implement a fitted z-score: this
requires an explicit policy addition and tests, not setting `hu` or
`normalized_0_1` on these raw values. Native-intensity pilot success alone is
not evidence that raw intensity is the best learning policy.

### Reproduction and outstanding gates

Final suite: **341 passed, 0 failed/skipped**, four expected intensity
provenance warnings, 132.75 s. The initial run recorded 339 passed / 2 failed
in 127.48 s, reproducing a pre-existing batch-inspection call to the deleted
`examples/read_training_batch.py`. The maintained CLI now lives at
`scripts/read_training_batch.py`; the test caller and documentation use that
location. The example deletions remain intact and no test assertions were
weakened or skipped. The focused repaired-CLI/preflight/hygiene run passed
all 20 checks. All nine new preflight checks passed, including TRAIN-only access
guards, actual nearest-exact equivalence, streaming geometry/scaling, compact
checkpoint/native pilot and OOM reporting. Initial and final JUnit reports are
`outputs/merit_preflight_fold0/pytest.xml` and `pytest_final.xml` respectively.

See [preflight commands](merit_pipeline.md#pre-training-preflight) for actual
audit/profile/pilot invocations. No full epochs, five-fold CV, test evaluation,
normalization selection or native MERIT patch benchmark was run. Before formal
199-case development, resolve/document patient linkage limitations, spatial-unit
assumptions, upstream intensity provenance and the chosen normalization/grid;
then run short TRAIN/validation learning checks and freeze checkpoint selection.

## Reproduce software checks

Run heavy jobs sequentially:

~~~powershell
py -m pytest -q
py -m compileall -q src scripts tools examples ct_pipeline
git diff --check
py scripts/merit_pipeline.py smoke --device cpu --output outputs/new_cpu_smoke.json
py scripts/merit_pipeline.py smoke --device cuda --output outputs/new_cuda_smoke.json
~~~

Real-data inspection commands are in [merit_pipeline.md](merit_pipeline.md).
Documentation cleanup does not automatically train, download data, regenerate
splits, evaluate external cases or overwrite historical numeric evidence.

## Remaining issues by priority

| Priority | Work still needed |
| --- | --- |
| Critical software | No confirmed critical defect left open within the tested repair scope; not proof of clinical safety |
| Important data | Verify patient linkage, HU/normalization provenance, three unknown-unit cases and upstream source revision/identity |
| Important preprocessing | Fold-0 TRAIN retention is quantified; freeze an explicit normalization/grid and validate learning behavior |
| Important experiments | Choose physical spacing/other settings only from train/validation; run full matched five-fold training |
| Important evaluation | Freeze final protocol before internal/external reporting; establish cross-source clinical independence where possible |
| Optional research | Compare max/top-k and independent physical axes, add uncertainty intervals and multi-iteration memory/throughput measurements |
| Out of current implementation | Learned spatial correspondence/agreement extensions, additional model baselines and dedicated standalone-baseline trainer |

Checkpoint replay is epoch-boundary only, freezes resolved source paths, and is
not a voxel checksum or a cross-hardware/version bitwise guarantee.
No full COVID CV or Zenodo benchmark has been completed. Matching equations,
module parameter counts and synthetic tests does not prove exact paper
reproduction.

## Readiness

| Area | Current evidence / boundary |
| --- | --- |
| Data | Synthetic geometry/GT/CT-only contracts plus stored primary audits; unresolved identity/intensity/units provenance |
| Model | Baseline source checks; MERIT equations/counts and full-reference GPU gradients |
| Training | Compact CPU/CUDA tests and four actual CT optimizer steps/resume passed; full training outstanding |
| Inference | Complete real A-0025 native reconstruction passed; cohort runtime/clinical quality not established |
| Evaluation | Shared native metrics, group/fold summaries and checkpoint identity integrated; no clinical performance result |
| Reproducibility | Effective config/split/RNG/provenance saved; pin source/data and verify clinical linkage before research claims |

This is research software, not production-ready or clinically validated.

## Cleanup verification

- `py -m pytest -q`: **332 passed, 0 failed, 0 skipped**, 4 expected provenance
  warnings, 133.32 seconds. The seven added checks verify English project prose,
  local Markdown targets/anchors, one maintained README, topic-guide navigation,
  the complete saved CSV schema, pytest configuration, and report generation.
- The initial focused hygiene run passed all seven tests. Generated report text
  is checked against the stored primary summary without opening CT/mask files.
- Executable AST fingerprints of all 11 comment/docstring-only Python edits
  matched before/after after stripping docstrings. No algorithm/API was moved.
- `py -m compileall -q src scripts tools examples ct_pipeline`: passed.
- `py -m pip check`: environment warning/failure unrelated to the MERIT dependency
  set: installed `pymupdf 2.1` requires missing `pymupdf-layout` and `pymupdf4llm`.
  No environment packages were installed, removed or upgraded during cleanup.
- Canonical splits, numeric CSV/JSON audits, verification snapshots and upstream
  LICENSE/provenance files remain unchanged. Only the primary report's prose
  and generator were rewritten; measured summary values were not regenerated.
- Redundant docs/readmes and two unused hard-coded comparison scripts were
  removed after an external ZIP backup. Original data, checkpoints, reference
  clones and GitNexus storage were not deleted.

---

## MER attention execution change

The native-512 path now gives CUDA SDPA an explicit single-head dimension:
`[B,1,Q,C]` and `[B,1,K,C]`. The learned distance bias is cast to the query dtype;
per-sample masks receive the same head dimension. Both ordinary MER and the
production checkpointed MER call the same helper in `modules.py`.

This removes two known obstacles to memory-efficient CUDA attention: the old
3D Q/K/V layout and an FP32 bias with FP16 queries. PyTorch still selects the
backend according to device, dtype, mask and deterministic settings. No fused
kernel is forced during production training.

Every query still attends to every context token in one softmax. The change
does not pool context, change the input grid, remove checkpointing, or alter
parameter/state-dict shapes. Casting the bias retains its gradient, but CUDA
floating-point rounding and backend selection can change results; this is not
a bitwise reproduction of the previous CUDA execution. The CPU path retains
the previous layout.

Validation on 2026-11-15:

- 37 existing CPU model/equation/gradient/checkpoint tests passed.
- Eight new CUDA regression cases compare ordinary/checkpointed MER with the
  legacy math path in FP32 and AMP, with and without masked context. They also
  require the efficient backend, exercise a partial last query chunk, physical
  offsets, different masters, learned distance gradients and an all-masked
  context sample.
- The new CUDA regressions could not be completed locally: Windows reported
  insufficient paging memory, followed by CUDA OOM even when creating a
  one-element tensor. This is an unresolved validation limitation, not a pass.
- No native-512 T4 speedup or segmentation-quality result is claimed.

Run the CUDA regression on an available CUDA machine:

```text
python -m pytest tests/test_mer_attention_execution.py tests/test_production_checkpointing.py -q
```

After it passes, use the existing preflight profiler on T4 with the native-512
config, warmup and repeated optimizer steps. Compare against the saved source
snapshot, recording step times, peak VRAM and AMP retries/skips. A new source
bundle is needed for Kaggle; the already-running recovery job continues using
its existing snapshot. The normal archive-identity resume guard remains in place.

### First Colab run received, 2026-11-15

The saved notebook reports Tesla T4, Torch 2.14.1+cu130, and matching hashes for
all six source CT/GT files. CUDA regression results were 5 passed and 4 failed;
the native-512 benchmark was correctly skipped. The four failures are the
checkpointed comparison cases, with different attention tensor metadata during
recomputation. The compact full-model CUDA/AMP checkpoint comparison passed.

The comparison test selected an SDPA backend only around forward, then exited
that context before backward triggered recomputation. The corrected test keeps the
selected backend active through backward for both the legacy and optimized
paths. Autocast still ends after forward. No numerical assertion or checkpoint
determinism check was disabled. A new CPU regression checks the backend state
in both the original forward and checkpoint recomputation.

The original executed notebook is preserved under
`outputs/colab_attention_results/20261115_125837_098fe65c/`. The revised portable
notebook is updated in place at `notebooks/colab_mer_attention_pilot.ipynb`; it reuses the same
hash-verified dataset cache and emits the full short traceback if tests fail.
GPU results for this revision and native-512 performance remain pending.

Reference: [PyTorch checkpoint warns against changing execution state between
forward and recomputation](https://docs.pytorch.org/docs/2.14/checkpoint.html).

---

## Đánh giá thời gian train và thiết kế đầu vào — 15/11/2026

Kết luận: pipeline đã có nhiều sửa chữa để chạy được và tiếp tục sau gián đoạn,
nhưng cấu hình full-slice native 512 với MER attention toàn cục chưa phù hợp để
làm vòng lặp phát triển chính trên một T4. Chưa có validation để kết luận chất
lượng học. Cần cải thiện cả execution và ngân sách lấy mẫu; nếu chưa đạt tốc độ
cần thiết, thay đổi đầu vào hoặc thiết kế attention bằng thí nghiệm riêng.

### Bằng chứng và phạm vi

- Đọc source đang có trong working tree, bao gồm các thay đổi chưa commit.
  Không thay đổi trainer, model, preprocessing hay job Kaggle trong lần đánh giá.
- Log version 04: `outputs/kaggle_results/version-04/merit-native512-fold0.log`.
  Lần chạy này kết thúc với CUDA assertion ở probability BCE, sau log batch 1960.
- Lấy log trực tiếp từ `vtan2405/merit-native512-fold0-recovery` khi job còn RUNNING.
  Snapshot tại `outputs/kaggle_review_20261115/recovery-live-events.jsonl`;
  thống kê tại `recovery-summary.json` cùng thư mục. Snapshot cuối: epoch 0,
  batch 4010/8810. Đây không phải kết quả cuối cùng của job.
- Split/nhãn: `analysis/covid19_20/case_statistics.csv`; intensity train:
  `outputs/merit_preflight_fold0/intensity_cases.csv`.
- GitNexus: repository `merit-covid-ct`, đúng workspace này. Đã chạy lại analyzer;
  MCP vẫn báo metadata cũ, 8 commits behind. Kết luận được xác nhận bằng source
  hiện tại, không dựa vào việc graph không tìm được caller.

| Đại lượng | Kết quả |
| --- | ---: |
| TRAIN fold 0 | 127 volume, 8810 master slices |
| Master có lesion / empty | 3278 / 5532, tức 37,21% / 62,79% |
| Lesion voxel / toàn bộ voxel TRAIN | 0,4453% |
| VALIDATION | 32 volume, 2322 master slices |
| Version 04: thời gian trung bình giữa các log train, quy về batch | 5,342 s |
| Recovery: thời gian trung bình giữa các log cách nhau 10 batch | 5,378 s |
| Ngoại suy phần TRAIN của một epoch recovery | 13,16 giờ, chưa tính validation/setup |
| Version 04: độ trễ log checkpoint sau log train cùng batch, median 39 lần | 3,088 s |

Checkpoint mỗi 25 batch tương ứng khoảng 18 phút/8810 batch nếu thời gian ghi
giữ nguyên: khoảng 2,3% tổng thời gian train. Giảm tần suất ghi có ích nhưng
không thể giải quyết mức chậm hiện tại. 100 epoch tương ứng khoảng 1316 giờ
TRAIN/fold nếu không dừng sớm và tốc độ giữ nguyên; đây là ngoại suy, không phải
thời gian một thí nghiệm đã hoàn thành.

### Cách train hiện tại

1. Chọn partition theo split có sẵn; mỗi lát của mỗi CT đều là một master.
2. Lấy slab 11 lát quanh master, giữ XY 512, intensity native sau NIfTI scaling.
3. Shuffle volume, rồi shuffle master trong volume; cache một volume.
4. Backbone xử lý cả 11 lát, TEP ở bottleneck, MER ở feature H/4 × W/4,
   PCG trên feature cuối. MER chỉ sửa feature của master.
5. Dice + CE giám sát các lát hợp lệ trong slab với Gaussian trọng số quanh
   master; thêm presence, alignment và salience. Không phải chỉ loss master.
6. AdamW, AMP FP16, gradient clipping, activation recomputation. Mỗi 25 batch
   lưu model/optimizer/RNG và tiến độ. Validation ghép dự đoán từng master
   thành volume; chọn best bằng native-volume Dice.

Các slab chồng lặp làm cùng lát được xử lý nhiều lần với vai trò master/context
khác nhau. Việc này có ý nghĩa trong mô hình master-aware, nhưng tốn hơn nhiều
so với đọc một volume và dự đoán mọi lát trong một lượt.

### Nút thắt chính và mức độ chắc chắn

**1. MER toàn cục tăng theo lũy thừa bốn của chiều ảnh.**

Với N=(H/4)(W/4), một master query N token và 10N context keys tạo 10N² cặp
query/key. Đây là số cặp tính toán, không có nghĩa toàn ma trận luôn nằm trong VRAM.

| Input XY | MER XY | Cặp query/key | So với 512 |
| --- | --- | ---: | ---: |
| 128 | 32×32 | 10.485.760 | 1/256 |
| 256 | 64×64 | 167.772.160 | 1/16 |
| 512 | 128×128 | 2.684.354.560 | 1 |

Chunk 128 ở input 512 cần 128 attention calls cho một MER forward. Mỗi chunk
vẫn nhìn toàn bộ 163.840 context token. Recompute tiết kiệm bộ nhớ nhưng phải
tính lại các vùng được checkpoint khi backward. Tổng thời gian cần profiler
để phân bổ chính xác giữa MER, backbone, loss, data và optimizer.

512 còn làm tăng các position embedding và EPA projection phụ thuộc grid:
tài liệu repo ghi khoảng 174 triệu tham số tại 512 so với 29,8 triệu tại 128.
Do đó không nên quy toàn bộ chậm cho riêng MER.

**2. Có vấn đề cụ thể trong cách gọi SDPA, cần thử sửa trước khi đổi thuật toán.**

`MasterEvidenceRouting.forward` và production `checkpointed_mer` truyền Q/K/V
3 chiều `[B,tokens,C]` vào SDPA. Fused CUDA attention của PyTorch 2.11 yêu cầu
Q/K/V 4 chiều. Đây là trở ngại cụ thể đối với backend tối ưu, không chỉ là
giả thuyết về GPU utilization.

Probe nhỏ trên GTX 1650, Torch 2.5.1, lưu tại
`outputs/kaggle_review_20261115/local_sdpa_probe.json` cho thấy:

- Ép EFFICIENT_ATTENTION với input 3D bị từ chối vì số chiều.
- Input 4D nhưng bias FP32 với Q FP16 bị từ chối vì dtype bias.
- Input 4D và bias FP16 chạy forward/backward; gradient tới hệ số distance bias
  vẫn hữu hạn, cả deterministic bật và tắt.

Candidate: thêm head dimension kích thước 1, xử lý broadcast/dtype bias đúng,
giữ một softmax trên toàn context và giữ gradient của distance scale. Cần đối
chiếu output/gradient, mask, biên và checkpoint recomputation trước benchmark
T4 thực tế. Probe không chứng minh tương đương toàn mô hình hay tốc độ native512.
Một lần thử đối chiếu bổ sung bị OOM trên GPU local; chưa có kết quả đối chiếu đó.
Không cam kết speedup hay khẳng định chỉ reshape là đủ cho mọi backend.

Nguồn đối chiếu:
[PyTorch 2.11 shape checks](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/sdp_utils_cpp.h),
[SDPA 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html).

**3. Bản sửa số học đang giúp phục hồi, chưa giải quyết gốc rễ.**

Recovery có hai thông báo retry FP32 ở decoder4 và decoder3, rồi đi qua điểm
crash cũ tới ít nhất batch 4010. Wrapper chỉ in lần retry đầu mỗi layer; không
thể suy ra chỉ xảy ra đúng hai lần. Mỗi convolution có kiểm tra finite trong
AMP, tạo reduction và CPU/GPU synchronization khi Python đọc điều kiện.
Chưa có profiler để lượng hóa overhead này.

TRAIN intensity audit bao phủ 127 ca, min toàn bộ -9653, max 32737. Đây là lý do
để kiểm tra chuẩn hóa và activation range, không phải bằng chứng xác định
intensity gây toàn bộ overflow. `train_zscore` và `scripts/fit_train_zscore.py`
đã có trong source, nhưng native512 chưa dùng. Tài liệu cũ nói chưa triển khai
z-score không còn khớp source hiện tại.

Hướng bền vững: so sánh intensity policy trên TRAIN/validation, đếm retry mỗi
layer và AMP skipped updates, dùng FP32 có chủ đích ở vùng nhạy cảm nếu cần.
Không xóa toàn bộ guard khi chưa xử lý lỗi. Auxiliary heads có thể xuất thêm
raw logits để dùng BCEWithLogitsLoss, giữ probabilities cho TEP/PCG và metric.
Không đưa probabilities đã sigmoid vào BCEWithLogitsLoss.
[PyTorch giải thích tính ổn định](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BCEWithLogitsLoss.html).

**4. Vòng phát triển quá dài và chưa quan sát được chất lượng học.**

Recovery snapshot có 206 loss được log, 70 dưới 0,01 và 52 lớn hơn 1. Đó là
sampling mỗi 10 batch, không phải thống kê toàn bộ batch. Chưa có GT/prediction
foreground hay loss theo nhóm positive/negative nên không thể kết luận collapse.

PCG ramp theo epoch; epoch 0 có correction bằng 0 nhưng head vẫn nhận auxiliary
supervision. Script Kaggle hardcode `--epochs 1`, nên lần chạy một epoch hiện
tại chưa đánh giá PCG correction sau warmup. Đây là pilot dài, chưa phải protocol
hoàn chỉnh cho 100 epoch. Khi đổi số samples/epoch phải định nghĩa lại ramp,
scheduler/patience theo update hoặc coverage, không giữ số epoch một cách máy móc.

### Resize có còn đúng không?

Resize là lựa chọn mô hình hóa hợp lệ nếu train/inference nhất quán, mask được
xử lý đúng, phục hồi geometry và đánh giá trên native grid. Nhưng không bảo đảm
giữ tổn thương nhỏ; upsample probability không khôi phục được tín hiệu đã mất.
Paper tham chiếu dùng ROI 128, không tương đương thu nhỏ toàn lồng ngực 512→128.

QC đã có trong repo trên 127 TRAIN ca:

| Chỉ số hình học/nhãn | Full slice →128 | Full slice →256 |
| --- | ---: | ---: |
| Thành phần tổn thương 2D biến mất | 325 | 186 |
| Positive slice trở thành empty | 34 | 10 |
| Foreground native không được giữ qua down/up | 4,4185% | 2,3822% |

Đây không phải Dice model, số lesion 3D hoặc mức giảm accuracy đã đo. Resize
cũng có thể đổi nhãn presence/salience suy ra từ mask; cần thiết kế target
nhất quán. Không đơn thuần bật YAML 256 rồi coi mọi vấn đề đã được giải quyết.
Thay grid còn đổi shape tham số, checkpoint 512 không load trực tiếp sang 256.

### Lựa chọn thực tế, theo thứ tự ưu tiên

1. **Tối ưu execution giữ công thức hiện tại.** Sửa candidate SDPA nói trên;
   profile với warmup rồi nhiều optimizer updates, gồm ca lesion/empty và ca
   intensity cực trị. Đo data wait, forward/MER, backward, optimizer, checkpoint,
   peak VRAM, retries và skipped updates. Chỉ chọn chunk lớn hơn khi có đo VRAM.
   Chưa cần đổi loss weights hay tắt deterministic ngay từ đầu.
2. **Rút ngắn vòng thí nghiệm bằng train sampling.** Đã có
   `samples_per_volume`: 8→1016 samples/epoch; 16→2032. Với tốc độ recovery cũ,
   phần train tương ứng khoảng 1,52/3,04 giờ, chưa có validation. Sampler hiện
   tại uniform có replacement, không cân bằng lesion và không bảo đảm coverage
   một epoch. Nếu thêm positive/negative sampling thì giữ cả hai nhóm, xét cả
   lesion trong slab, log coverage và đánh giá đủ volume validation.
3. **Nếu muốn giữ pixel nhỏ: native patch 256 hoặc crop phổi theo CT.** Patch
   `[11,256,256]` giữ độ phân giải pixel, cắt bớt phạm vi XY; khác resize toàn ảnh.
   Train lấy patch positive/negative, val/inference phủ đủ ảnh bằng overlap và
   blend probabilities trước threshold. Crop phổi phải suy ra từ CT và có margin,
   có phương án khi crop lỗi, kiểm tra tổn thương sát màng phổi. Không dùng GT
   để chọn ROI tại validation/inference. Cần MERIT-specific tiling vì patch
   engine baseline hiện chưa hỗ trợ MERIT; targets presence/salience phải được
   định nghĩa theo patch. Bốn tile không overlap đã bù lại một phần lợi ích
   1/16 attention mỗi tile; overlap và số patch train quyết định lợi ích thực.
4. **Nếu cần full-field 512: giảm token trong MER thay vì giảm input.** Giữ
   master queries 128×128, pool context K/V xuống 32×32: số cặp còn 167,8 triệu,
   ít hơn 16 lần cho phần MER. Giữ positional/distance bias đúng grid. Hoặc
   chuyển MER xuống feature sâu hơn, dùng local windows/pooled global context.
   Đây là thay đổi thuật toán, phải train/ablate riêng; không phải tối ưu tương
   đương chính xác và không bảo đảm giữ accuracy.
5. **256 full-field là đối chứng hợp lý, 128 full-field là phương án cần thận
   trọng hơn.** Đo lesion recall theo kích thước, native Dice, false positives
   trên empty slices và thời gian; so với patch bằng cùng ngân sách updates/GPU.

Giảm decoder chỉ dự đoán master, reuse backbone giữa slab, hoặc model 2.5D nhỏ
hơn cũng là hướng nghiên cứu, nhưng cần thiết kế lại context features/loss;
không bỏ tùy tiện supervision các lát context khi chưa có ablation.

### Những nhận định cần sửa trong bản phân tích đính kèm

- Cache một volume không đồng nghĩa đọc lại liên tục: config đang shuffle theo
  volume, và Kaggle input lưu nhiều file `.nii` chưa nén. Cần đo data wait trước
  khi quy lỗi giải nén.
- `num_workers=0` đang phục vụ exact mid-epoch replay. Bật workers cần thiết kế
  lại sampler/progress để kiểm soát prefetch; không chỉ đổi một số trong YAML.
- Activation checkpointing là giải pháp VRAM hợp lý, không phải giải pháp giảm
  FLOPs; baseline từng OOM khi bỏ nó. Chỉ bỏ từng vùng sau khi có VRAM profile.
- PCG ramp=0 ở epoch 0 là đúng code; không chứng minh module hỏng, cũng không
  có nghĩa head presence hoàn toàn không được học.
- Log loss thấp chưa đủ khẳng định model collapse hoặc hội tụ.

Khuyến nghị: thử SDPA và quan sát số học trước. Nếu vẫn không đạt ngân sách,
ưu tiên native patches 256 có inference phủ đủ ảnh; nếu bắt buộc giữ full-field
512 thì thử pooled-context MER. Chọn bằng validation native và thời gian thực,
giữ snapshot recovery làm đối chứng thay vì thay protocol giữa checkpoint cũ.

---

## Đánh giá MERIT-UNETR++ cho phân đoạn COVID CT

Ngày 11/11/2026. Phạm vi: mã nguồn và cấu hình hiện tại ở commit `2638a79f3f63b9ad85391ae3b6e107ed3edb2aa8`, các báo cáo dữ liệu/preflight đã có, và kiểm tra split. Hai notebook đang có thay đổi cục bộ; báo cáo này không đánh giá chi tiết những thay đổi đó. Đây là đánh giá nghiên cứu và độ sẵn sàng của pipeline, không phải kết quả hiệu năng mô hình hoặc xác nhận lâm sàng.

### Kết luận

**Chưa nên chạy full 5-fold để lấy kết quả chính thức.** Mã có một baseline MERIT hợp lý về mặt cấu trúc: giữ trục lát cắt, có TEP/MER/PCG, train trên nhãn slab thực và suy luận toàn volume không dùng GT để chọn ROI. Tuy nhiên, chưa có bằng chứng chứng minh ba module này có ích cho tổn thương COVID, cũng chưa thể kết luận code tái hiện chính xác tác giả gốc. Các quyết định về intensity, nhóm bệnh nhân, mất tổn thương khi resize và khoảng cách lát cắt vẫn cần giải quyết hoặc công bố giới hạn trước khi khóa protocol.

Paper tham chiếu trong [tài liệu mô hình](architecture.md#merit-unetr-model) nghiên cứu nodule trên LIDC-IDRI với ROI dựa vào annotation. Bài toán ở đây là mask nhiễm COVID trên toàn CT: tổn thương có thể lan tỏa, nhiều ổ, nhỏ và không đều; tỷ lệ foreground trên toàn ảnh rất thấp. Vì vậy dùng kiến trúc paper làm baseline là có lý, nhưng các module và hyperparameter cần được kiểm nghiệm lại trên bài toán mới. Không có kết quả ablation ở đây để khẳng định module nào tốt hay hại.

### Những vấn đề cần xử lý trước full training

#### P0. Intensity chưa được xác minh và baseline đang dùng giá trị thô

[`merit_index.yaml`](../configs/merit_index.yaml) đặt `intensity_encoding: native`. [`normalize_ct`](../src/merit_covid/data/pipeline/preprocessing.py) giữ nguyên giá trị sau NIfTI scaling; không clipping hay chuẩn hóa. Điều này không tự xác nhận HU, và giá trị ngoại lai có thể ảnh hưởng học. Không thể chỉ từ min/max kết luận mỗi voxel có sai HU hay không.

[Preflight cũ](../outputs/merit_preflight_fold0/REPORT.md) đo đầy đủ **127 ca train fold 0**, không phải 199 ca: khoảng -9653 đến 32737, 39 ca có min dưới -2048 hoặc max trên 10000, slope/intercept 1/0 cho nhóm đó. [Báo cáo 199 volume](../analysis/covid19_20/REPORT.md) ghi khoảng -10380 đến 32737, nhưng tái sử dụng audit CT cũ và không xác nhận byte nguồn hoặc calibration từng ca. Hai khoảng không mâu thuẫn vì khác phạm vi. Nguồn challenge mô tả HU, nhưng lịch sử DICOM-to-NIfTI từng ca chưa được chứng minh.

**Việc cần làm:** kiểm tra nguồn và slope/intercept, datatype, histogram và ca ngoại lai cho toàn bộ cohort; phân biệt giá trị padding/ngoại lai với mô bệnh học. Chọn và đóng băng xử lý intensity bằng train/validation, fit riêng mỗi fold nếu dùng thống kê dữ liệu, áp dụng giống nhau khi inference. Không dùng Zenodo20 để chọn chính sách. Preflight đề xuất z-score train-fold nhưng code hiện chưa triển khai policy fitted đó; không nhầm đề xuất với tính năng đã chạy.

#### P0. Spatial unit chưa chắc chắn và split chưa xác nhận bệnh nhân

[`geometry.py`](../src/merit_covid/data/pipeline/geometry.py) mặc định từ chối unit không rõ. Tuy nhiên **baseline index hiện bật `assume_mm: true`**, khác checklist đã gửi; [`merit_physical.yaml`](../configs/merit_physical.yaml) đặt `false`. Ba ca `volume-covid19-A-0161`, `A-0585`, `A-0599` thiếu unit theo báo cáo cohort. Khi bật giả định, khoảng cách và HD95 tính theo mm của các ca đó phụ thuộc giả định, không phải bằng chứng nguồn. Cần xác minh từ metadata nguồn hoặc ghi rõ giả định/xử lý có provenance trước khi báo cáo số mm.

Chạy `python scripts/merit_pipeline.py check-split --config configs/merit_index.yaml` thành công: fold 0–3 có 127/32/40 train/val/test, fold 4 có 128/32/39. Preflight cũ xác nhận không trùng case/group/path trong 15 cặp partition. Nhưng [`cases.py`](../src/merit_covid/data/pipeline/cases.py) dùng `volume_id_fallback`, và split ghi `patient_ids_available: false`: **không chứng minh được độc lập theo bệnh nhân**, cũng không loại trừ dữ liệu giống nhau dưới tên khác. Cần nguồn patient ID đáng tin cậy; nếu không có, chỉ gọi volume-level split và nêu giới hạn. Không tự ý thay split sau khi có kết quả.

#### P0. 128×128 làm mất thông tin nhãn

[`resize_in_plane`](../src/merit_covid/data/pipeline/preprocessing.py) dùng bilinear cho CT, nearest-exact cho mask và không resize Z. Trên **127 ca train fold 0**, [preflight](../outputs/merit_preflight_fold0/REPORT.md) đo:

| Chỉ số | 128×128 | 256×256 |
| --- | ---: | ---: |
| Thành phần 2D biến mất hoàn toàn | 325 | 186 |
| Foreground native không được khôi phục sau down/up | 4,4185% | 2,3822% |
| Positive slice trở thành empty | 34 | 10 |

Đây là QC hình học, không phải Dice của model hay số lesion 3D. Có 6.295 thành phần 2D trong mẫu đo. 256 giữ nhãn tốt hơn theo tổng nhưng thử full optimizer trên GTX 1650 4 GB đã OOM, kể cả batch 1. Không tự nâng lên 256. Đánh giá thêm theo kích thước/vị trí tổn thương bằng train, cân nhắc crop/tiling dựa trên ảnh CT mà không dùng GT khi inference, và profile lại bộ nhớ. Hiện patch engine baseline chưa tích hợp MERIT.

#### P0. Xác nhận hình học và phục hồi trên dữ liệu thật

[`inspect_pair`](../src/merit_covid/data/pipeline/geometry.py) kiểm tra shape, affine và orientation trước đọc voxel; reorientation chỉ đổi trục/lật. Mask được kiểm tra 0/1 và resize nearest-exact. [`predict_slab_case`](../src/merit_covid/engine/slab_inference.py) tạo dataset CT-only, dự đoán từng master, nội suy probability về XY native rồi ghép lát; evaluator đọc GT sau dự đoán để tính metric native. Đây là thiết kế đúng hướng, không phải bằng chứng tất cả 199 ca đã qua end-to-end.

Pilot cũ dùng một ca train A-0025, bốn optimizer steps, kiểm tra inference không mở mask, khôi phục shape/affine gốc 512×512×61; chưa chứng minh convergence hoặc độ chính xác. Cần kiểm thêm alignment, nhãn và restoration ở các orientation/spacing/biên khác nhau, đối chiếu native GT, xác nhận không thiếu/trùng slice và nêu unit giả định. Không dùng GT để chọn crop hoặc master inference.

### Khoảng cách lát cắt: mã làm gì và có nghĩa gì?

Theo [thống kê cohort](../analysis/covid19_20/REPORT.md), 188/199 volume ghi spacing 5 mm, còn lại 0,799988/1/2 mm; ba unit chưa rõ phải coi là giả định. Số liệu là spacing header, không chứng minh slice thickness hoặc khoảng cách lát acquisition bất quy tắc. NIfTI affine chỉ mô tả lưới đều và không thể tự khôi phục vị trí từng lát gốc nếu conversion đã làm mất thông tin.

Baseline lấy 11 chỉ số liên tiếp quanh master: nếu spacing 5 mm thì tâm context danh nghĩa trải từ -25 đến +25 mm; nếu 1 mm thì chỉ -5 đến +5 mm. Hai volume vì vậy có receptive context vật lý khác, ngoài khác biệt về cấu trúc và intensity. Gần mép volume, `edge` lặp lát biên.

[`slab.py`](../src/merit_covid/data/pipeline/slab.py) có lựa chọn `physical_mm`: lấy lát native gần các offset mục tiêu, **không nội suy theo Z**, lưu cả requested offset và offset thực. Offset thực lấy từ displacement tâm lát theo cột thứ ba affine; với shear đây là khoảng cách giữa tâm tương ứng dọc cột affine, không nhất thiết là khoảng cách vuông góc giữa mặt phẳng. Không thể dựa vào nó để kết luận đã chuẩn hóa slice thickness. Target nhỏ hơn native spacing thường chọn trùng lát; target lớn hơn có thể bỏ qua lát, còn ranh giới vẫn lặp.

Ba lựa chọn độc lập cần ablation trên cùng fold, seed, preprocessing, kiến trúc và các đơn vị đã xử lý thống nhất:

| Thí nghiệm | Sampling | Position encoding | Trọng số loss |
| --- | --- | --- | --- |
| E1 | index | index | index |
| E2 | physical mm | index | index |
| E3 | physical mm | actual mm | index |
| E4 | physical mm | actual mm | actual mm |

E2 template cố ý để `target_spacing_mm: null`, phải chọn tham số từ train/validation. Khi so E1–E4 phải thống nhất policy unit: hai YAML hiện khác `assume_mm`, nếu giữ nguyên thì không còn là so sánh một yếu tố. Báo cáo native slices khác nhau, actual offsets, độ lặp theo volume và biên, cũng như Dice/recall theo spacing. `position_unit_mm` và `sigma_mm` cần được định nghĩa và cố định. Không mặc định 5 mm hay physical PE tốt nhất.

### Đánh giá các module

[`model.py`](../src/merit_covid/models/merit_unetrpp/model.py) đặt TEP ở bottleneck, MER sau decoder3 và PCG trước output; [tài liệu](architecture.md#merit-unetr-model) ghi công thức, kích thước, số tham số và các lựa chọn paper chưa đặc tả. Đây là reimplementation dựa trên paper chứ không phải source tác giả; khớp số tham số hoặc test công thức không chứng minh tái hiện kết quả.

- **TEP:** gộp thông tin các lát thành prototype theo salience và vị trí rồi attention trở lại từng lát. Với COVID đa ổ/lan tỏa, giả thuyết dùng context hợp lý nhưng một prototype chung có thể không phù hợp mọi tổn thương; đó là giả thuyết cần đo, không phải lỗi đã chứng minh. So UNETR++ baseline với TEP, theo lesion size và vị trí, kiểm tra salience có tương quan lesion hay anatomy/intensity.
- **MER:** master query attention tới context, cập nhật master trong module; các lát khác không được cập nhật bởi chính MER. Context xa có thể hữu ích hoặc nhiễu. So TEP-only với TEP+MER, kiểm tra 3/5/11 lát và phạm vi mm; đảm bảo cấu hình model và input khớp khi thay độ sâu.
- **PCG:** dự đoán presence và cộng log-odds vào foreground logits, warmup theo epoch. Có thể giảm false positive nhưng cũng có thể chặn tổn thương nhỏ; đo sensitivity, false positives trên slice rỗng, calibration ECE/Brier và so có/không PCG. `alignment_pool=max` nhạy với một pixel; `top_k_mean` đã là lựa chọn ablation trong loss, chưa có bằng chứng chọn phương án nào.
- **Duplicate context:** `duplicate_slice_policy: unique` chỉ loại bản trùng khỏi **loss** qua `valid_region`; [`modules.py`](../src/merit_covid/models/merit_unetrpp/modules.py) không nhận mask validity, vì vậy TEP/MER vẫn attention vào lát lặp. Ví dụ master đầu volume với edge padding có nhiều bản sao master trong context; kết quả attention có thể bị trọng số lệch. Đây là hành vi xác nhận từ code, chưa chứng minh gây giảm Dice. Thử tùy chọn mask và so baseline, bảo đảm softmax an toàn nếu không còn context hợp lệ; lưu ý mask attention không xóa ảnh hưởng duplicate qua các convolution 3D trước attention.

Master hiện luôn ở giữa slab trong data pipeline, dù model API nhận vị trí khác. Central master là protocol hợp lệ; shifted window là thí nghiệm riêng, không phải lỗi bắt buộc sửa ngay.

### Training và đánh giá

- [`CTSlabDataset`](../src/merit_covid/data/pipeline/dataset.py) tạo một master cho mỗi lát. Volume dài đóng góp nhiều mẫu hơn. [`CaseGroupedSampler`](../scripts/train_colab.py) chỉ xáo thứ tự case rồi lát và vẫn thăm mỗi master một lần; **không phải volume-balanced sampling**. Thử sampler cân bằng volume và positive/negative như ablation, tránh dùng nhãn để lọc inference.
- Đường train MERIT chính chưa thấy augmentation trong `CTSlabDataset`; augmentation ở `data/dataset.py` là dataset khác, không đồng nghĩa được dùng ở đây. Nếu bổ sung, transform hình học phải đồng bộ toàn slab/mask, chỉ trên train và có seed/provenance.
- [`metrics.py`](../src/merit_covid/data/pipeline/metrics.py) đã có Dice/IoU, precision/recall, HD95 và policy mask rỗng; aggregate theo group. Cần bổ sung hoặc báo cáo rõ lesion-size recall, false positives trên lát rỗng và ECE/Brier cho presence. HD95 undefined khi một trong hai mask rỗng và mm phụ thuộc unit đúng.
- Chốt protocol: U-Net 2D và UNETR++ cùng split/preprocessing/evaluation; baseline MERIT; ablation từng module và spacing; 5-fold báo cáo per-case/per-fold, mean ± SD; chỉ dùng Zenodo20 sau khi khóa preprocessing, model và threshold. Không so Dice trực tiếp với paper khác protocol.

### Kiểm thử và giới hạn bằng chứng

Kiểm tra split hiện tại thành công với hash `7b31f9e250e64826f1fff0d9fec4a4aba034252cec7d5123d5f0393ec665e54d`. Preflight lịch sử ngày 08/11 báo 341 tests pass, pilot CT và profile GPU như trên; đó là kết quả lịch sử, không đại diện tự động cho mọi thay đổi notebook hiện tại. Không có full training, 5-fold hay external evaluation được thực hiện trong lần review này.

Lần review này chạy `python -m pytest tests/test_merit_unetrpp.py tests/test_merit_pipeline.py tests/test_preprocessing.py tests/test_reconstruction.py tests/test_splitting.py -q`: **77 passed trong 77,84 giây** trên môi trường hiện tại. Đây là các test chọn lọc, không phải kiểm thử toàn bộ repo hay xác nhận hiệu năng COVID.

Index GitNexus hiện báo chậm hơn HEAD hai commit; lần refresh bị gián đoạn, nên các kết luận ở đây dựa trên đọc source trực tiếp và artifact đã lưu, không khẳng định graph review đầy đủ. Audit cũ không chứng minh dữ liệu hiện tại chưa đổi; cần pin phiên bản/hash dữ liệu trước nghiên cứu chính thức.

### Thứ tự thực hiện đề xuất

1. Xác minh nguồn intensity/units, patient identity; lưu quyết định và provenance, không thay split tùy tiện.
2. Đo retention và lựa chọn input khả thi trên train/val; kiểm hình học và CT-only restoration ở nhiều ca/biên.
3. Thêm và test tùy chọn duplicate attention mask, augmentation đồng bộ, sampler và metric thiếu khi cần; giữ baseline cũ cho đối chứng.
4. Chạy pilot dài hơn và xác nhận finite loss, resume, inference, VRAM trên cấu hình thực; khóa config/seed/fold và chọn checkpoint bằng validation native-volume.
5. Chạy baseline và ablation trước khi kết luận nên thay TEP/MER/PCG hoặc physical spacing; giữ external test độc lập.

## 2026-11-16 factorized MER query chunk pilot

Saved Colab notebook outputs report PASS on Tesla T4 with Torch 2.14.1+cu130.
The same three fold-0 TRAIN cases (A-0025, A-0295, A-0387), initial weights,
native 512 input, full context and sample order were used in isolated processes.
Each size had one warmup followed by six measured optimizer updates.

| Query chunk size | Mean seconds/update | Peak allocated MiB | Status |
| --- | --- | --- | --- |
| 128 | 9.744222 | 4696.67 | PASS |
| 256 | 8.376484 | 4696.67 | PASS |
| 512 | 7.642528 | 4696.67 | PASS |

Chunk 512 reduced observed step time by 21.57% versus chunk 128 in this pilot.
No optimizer updates were skipped; all three worker reports passed, with equal
initial-weight hashes and no FP32 convolution retries. Four compact chunking
output/input/parameter-gradient tests passed on the assigned CUDA runtime.
The worker reports also check native source pixels and unchanged CT/GT hashes.
This is a small throughput diagnostic, without validation Dice or convergence
measurements. The compact equivalence tests use small query chunks; they do not
establish bitwise equality across every native-512 update. The three sizes were
measured once in ascending order, so repeated or reordered trials are needed
before claiming stable production throughput. The existing Kaggle source package
and its training configuration were not changed by this experiment.

Local extracted evidence: `outputs/chunk_experiment/colab_20261116_013610_77226c6d/`.
The full worker JSON files remain in the Colab result ZIP; the local extraction
contains only the visible saved output, summary and notebook checksum.

## Follow-up execution sweep: Colab results received

The existing Colab notebook now includes `MER_EXECUTION_SWEEP_V1`: native query
chunks 512/1024/4096/16384, followed by a full-query variant that retains MER
activations instead of recomputing them. Each candidate must pass native-token-grid
output/input/parameter-gradient comparisons before the real-CT pilot. The compact
CUDA suite passed eight FP32/AMP, masked/unmasked and recomputation tests locally.
The local tests alone did not establish native-T4 performance; the later Colab measurements are recorded below.

The sweep repeats chunk 512 at the end, repeats the observed fastest setting,
and profiles that setting. Timing drift over 10% is labeled inconclusive. Inputs,
full context, weights and loss definitions are retained in these isolated
experiments; no production checkpoint or config is migrated by the notebook.

Local Dataset/collate checks on A-0025/A-0295/A-0387 took 1.13/1.98/1.30 seconds
for the first native volume read and 0.0266/0.0271/0.0255 seconds per warm batch
(mean of six). These are local CPU timings with `[1,1,11,512,512]` inputs, not
Colab or full-cohort throughput. Evidence is saved in
`outputs/chunk_experiment/local_io_profile.json`.

### Native execution sweep results, 2026-11-16

The six-run sweep and independent best-setting repeat passed on T4. All native
MER output/input/parameter-gradient guards passed their declared tolerances.
The native guard reports show zero output and input-gradient max differences;
parameter gradients were checked within tolerance, without a bitwise-equality
claim. The initial model hash is equal for every real-CT worker. No AMP update
was skipped or FP32 convolution retry reported in the visible output.

| Query chunk / MER recomputation | Mean seconds/update | Peak allocated MiB |
| --- | --- | --- |
| 512 / on, start | 7.539360 | 4696.67 |
| 1024 / on | 7.280379 | 4696.67 |
| 4096 / on | 7.201621 | 4696.67 |
| 16384 / on | 7.907465 | 4697.17 |
| 16384 / off | 7.779606 | 4699.85 |
| 512 / on, end | 7.674740 | 4696.67 |
| 4096 / on, repeat | 7.192366 | 4696.67 |

The repeated 4096 setting is stable in this small experiment. Its first run
reduced step time by 4.48% against the initial 512 run; the baseline drifted
1.80% between the start and end of the sweep. Larger chunks did not continue
to improve speed. These are two six-update observations on three TRAIN cases,
not segmentation-quality evidence or guaranteed full-epoch throughput.

Profiling the 4096 setting found about 5.51 seconds in efficient attention
backward out of a 7.28-second profiled step. The forward/loss CPU range was
about 0.54 seconds and optimizer CPU range 0.29 seconds. Nested CPU/GPU
profiler entries are not additive. The currently enabled deterministic
algorithm requirement is the next hypothesis to test: permitting parallel
reductions may improve backward speed, with reduced bitwise reproducibility.
No speed or quality claim for that unmeasured variant is made.

Visible evidence: `outputs/chunk_experiment/colab_execution_sweep_20261116_015345_d4b123dd/`.
The full worker files remain in the remote result ZIP.

### Parallel reduction sweep results, 2026-11-16

The next saved Colab T4 experiment passed its 14 compact numerical tests,
including FP32/AMP full-model loss and all parameter gradients. Native-grid
MER comparisons also passed at rtol 0.005 and atol 0.000002 before each real-CT
worker. These checks do not establish clinical accuracy or convergence.

| Execution at query chunk 4096 unless noted | Mean seconds/update | Peak allocated MiB |
| --- | --- | --- |
| Deterministic, MER recomputation on, start | 7.065987 | 4696.67 |
| Parallel reductions, MER recomputation on | 1.822295 | 4696.67 |
| Parallel reductions, query chunk 16384, recomputation on | 1.871375 | 4697.17 |
| Parallel reductions, MER recomputation off | 1.761345 | 4699.36 |
| Deterministic, MER recomputation on, end | 7.179553 | 4696.67 |
| Parallel reductions, MER recomputation off, repeat | 1.777857 | 4699.36 |

All runs start from the same initial-weight hash and use the same six measured
TRAIN samples, after one warmup. The fastest candidate reduced observed step
time by 75.07% (4.01x throughput) against the starting deterministic reference.
The baseline drift was 1.61%; the candidate repeat differed by 0.94%. No skipped
AMP updates or FP32 convolution retries were reported. The full 11-slice native
512 input, full context, canonical TRAIN cases and loss definitions remain in
use. Permitting nondeterministic kernels changes bitwise reproducibility, and
this execution policy must be explicit in any future training protocol and
checkpoint migration. No existing Kaggle checkpoint/config was changed.

The selected profile records about 0.2635 seconds in efficient-attention
backward, versus about 5.506 seconds in the previous deterministic profile.
The new profiled optimizer-step wall time is 1.837 seconds, and its two
unprofiled measured steps averaged 1.793 seconds. These profiles are separate
bounded measurements and are not added to inclusive CPU/GPU ranges.

Remaining evidence needed before using this as a production continuation:
longer multi-case native training including active PCG ramp, production
checkpoint/optimizer/scheduler/scaler continuation, and held-out native-volume
validation. No Dice or final model-quality improvement is claimed here.
Local visible evidence: `outputs/chunk_experiment/colab_parallel_20261116_021920_366ddbb2/`.
The full worker JSON files remain in the remote ZIP.

### Paired quality pilot: all-background predictions, 2026-11-16

Both the original execution and fast execution completed 24 diagnostic AdamW
updates from the same seeded random initialization. Three selected TRAIN cases
were reused and PCG ramp was explicitly forced to 1 as a stress diagnostic;
this differs from the production epoch-0 warmup policy and does not evaluate
the trained Kaggle checkpoint. The reference last-step loss was 1.078897 and
the candidate 1.081369, from an identical first-step loss of 2.406429.

| Held-out volume | GT foreground voxels | Reference predicted voxels | Fast predicted voxels | Dice, both |
| --- | --- | --- | --- | --- |
| A-0014 | 97552 | 0 | 0 | 0 |
| A-0072 | 10059 | 0 | 0 | 0 |

Both models predict only background at the fixed threshold 0.5. IoU, precision
and recall are also zero for both positive validation volumes. Execution PASS
means the worker finished its software checks; it is not a model-quality PASS.
Identical empty masks cannot establish preservation of useful segmentation.
Mean absolute probability differences between the branches are approximately
2.94e-9 and 2.31e-9, but close probabilities alone do not establish accuracy.
No threshold was selected or changed in response to these validation results.

The two full native CT volumes were predicted with GT access blocked until
inference completed. Original source hashes remain unchanged. Local saved
plots also show missing lesion predictions. All 24 train steps completed;
mean compute time excluding the first step was 13.9018 seconds for reference
and 1.7118 seconds for fast. These timings are bounded diagnostic observations.

This result does not identify the cause of all-background predictions.
Limited training, early fully-active PCG and other training/normalization issues
remain hypotheses, not demonstrated diagnoses. The next quality assessment
needs the actual trained checkpoint and its production warmup policy, evaluated
on the fixed validation partition. The test partition remains untouched.
Evidence extracted from saved notebook output:
`outputs/model_quality_review/colab_paired_20261116_032411_c7a50196/`.

## Kaggle epoch 0 completed: production checkpoint evaluation

Kernel `vtan2405/merit-native512-fold0-epoch0-finish` completed with
`run-status.json` reporting `epoch_complete` and `checkpoints/last.pt`.
This is the original production execution, not the fast Colab candidate.
The saved history contains one completed epoch, index 0, and full-native
validation on the 32 saved fold-0 validation volumes (2322 masters).

| Metric | Value |
| --- | --- |
| Validation group/volume macro Dice | 0.2820627 |
| Validation macro IoU | 0.1937204 |
| Validation macro precision | 0.5020440 |
| Validation macro recall | 0.3653914 |
| Mean HD95 on 31 defined groups | 80.1536 mm |
| Mean train total loss, 8810 examples | 0.5714969 |
| Successful optimizer steps | 8806 |
| AMP-skipped updates | 4 |
| PCG ramp | 0 |
| Learning rate | 0.0003 |

Three cases have zero Dice; one of these has an empty predicted mask. The best
case Dice is 0.7681618. Dice population standard deviation is 0.2598395, so the
mean conceals substantial case variability. The checkpoint shows nonzero
segmentation ability after one epoch, but low recall and poor average overlap;
it does not establish satisfactory final quality. Patient identity remains
unverified, so these are group/volume-based metrics, not verified patient means.

The final TRAIN checkpoint log time is 12343.75 seconds and the validation
completion line is 16906.36 seconds after session start: about 76 minutes for
the intervening validation and end-of-epoch processing. The job then reported
`epoch_complete` and saved its outputs. This actual measurement supersedes earlier
illustrative validation-time scenarios for this run.

Evidence downloaded from Kaggle is in
`outputs/kaggle_review_20261116/epoch0_complete/`; a flat per-case table is saved
as `validation-per-case.csv`. This result must be retained as the baseline for
any explicit migration to the fast execution; the fast execution has not yet
been assessed from these trained weights. No test-partition evaluation or
threshold tuning was performed.

## Trained epoch-0 checkpoint execution review: completed

The original completed `last.pt` was downloaded intact (2089373860 bytes),
SHA-256 `22596971dc0f3ff7b6796756a61cf61187ded97472a9bdf6d754766165cd6934`.
Its schema-2 protocol fingerprint verifies. Epoch index is 0; the saved model
PCG ramp is 0; optimizer step counters are all 8806. Scheduler, AMP scaler,
RNG and loader-generator state are present. The static model check confirms
all 661 state-tensor keys/shapes match and the fast overlay preserves all
492 parameter objects/order. Saved optimizer moment shapes match every
parameter. These checks do not establish successful cross-protocol resume.

A separate evaluation-only kernel was launched:
`vtan2405/merit-epoch0-fast-execution-validation`. It uses the exact original
source snapshot and checkpoint as read-only input, adds only the isolated
factorized MER forward overlay, and retains FP32 validation, threshold 0.5,
PCG ramp 0 and the original 32-case held-out partition. It verifies all 199
CT/mask pairs by uncompressed SHA-256 before running trained-weight probability
probes and full native-volume validation. There are no backward calls, optimizer
steps or trained-checkpoint writes in this kernel. Per-case metric drift above
0.001 causes `REVIEW_REQUIRED`; matching only the mean Dice is insufficient.
The inference comparison has completed as documented below; an epoch-1 training-state migration still requires its own bounded continuation test.

Local review artifacts:
`outputs/kaggle_review_20261116/epoch0_complete/local-checkpoint-audit.json`,
`optimizer-compatibility.json`, and `outputs/kaggle_trained_validation/`.

### Trained checkpoint validation outcome

The validation-only kernel completed with FINAL STATUS PASS. The checkpoint
SHA-256 matches the downloaded original epoch-0 checkpoint. All 32 cases have
exactly matching Dice, IoU, precision, recall, TP/FP/FN and foreground-voxel
counts in the saved reports. Macro Dice remains 0.2820627025084401, macro IoU
0.19372036713135504, precision 0.5020440146751306 and recall 0.3653914102282829.
Maximum absolute difference across the four per-case metrics is 0.0, below the
predeclared tolerance 0.001. PCG ramp remains 0; validation uses the original
FP32 policy and unchanged threshold 0.5. The model state hash is unchanged.

The direct original/fast probability probes on three fixed middle masters have
maximum absolute differences 2.9802322e-7, 2.1827873e-10 and 3.6379788e-11.
These are three direct probability comparisons; full-volume metric equality
is not a claim of bitwise probability equality on every voxel. The kernel
performs zero training updates. This establishes preservation of validation
metrics for the current trained weights, not identical future training or
improved model quality. Optimizer/scheduler/scaler migration and bounded
continuation remain separate checks before epoch-1 training.

Downloaded evidence is in `outputs/kaggle_trained_validation/results/`.
`trained-validation/per-case-comparison.csv` records the original/fast Dice
and recall for each volume. The original checkpoint remains unchanged.

## Fast epoch-1 continuation: migration checks

The immutable epoch-1 source snapshot SHA-256 is
`4cc2242171c4c28dd666fc182afb412b80fa9153f01350b55757fa52fbb36b40`.
Nine isolated tests executed against this packaged source passed. They cover
synthetic epoch-0 training, full-state migration, exactly one additional epoch,
checkpoint save/reload/inference execution restoration, original fingerprint
compatibility, timestamp-only rebinding, and rejection of changes to learning
rate, loss, threshold, sampling, epoch index and mid-epoch migration.

The actual 2.09-GB epoch-0 checkpoint also passed the metadata migration check:
model/optimizer/scheduler/scaler/RNG/loader state objects are preserved; only
the approved execution settings, total epoch budget and fingerprint change.
This local metadata check is supplemented by CUDA state verification and a
two-update/rollback guard in the launched kernel before epoch 1 may begin.
Kernel: `vtan2405/merit-fast-epoch1`. Full production epoch-1 results are pending.
