# Validation evidence and limitations

This is the single maintained evidence summary. Historical source audits, merge
notes, implementation roadmaps and MERIT repair reports have been consolidated
here and into [CHANGELOG.md](../CHANGELOG.md). Stored verification JSON files
remain historical snapshots: old filenames/counts/environments inside them are
not claims about the latest checkout.

## Verification milestones

| Stage | Recorded result | Scope |
| --- | --- | --- |
| Main foundation, 2026-11-07 | 56 passed, 1 skipped | Core synthetic pipeline; Torch unavailable then |
| Duong LPS/native-intensity data pipeline | 72 passed | Geometry, scaling, native size and data tools |
| Main/Duong integration | 135 passed, no skips | Shared core and both public APIs |
| Standalone UNETR++ integration | 188 passed, no skips | 53 new model checks plus prior regressions |
| Native baseline patch/output integration | 262 passed, no skips | 74 additional input/reconstruction checks |
| MERIT pipeline repair, 2026-11-08 | 325 passed, 0 failed/skipped, 4 warnings; 143.69 s | Full suite, including actual CUDA AMP check |
| Repository/documentation cleanup | 332 passed, 0 failed/skipped, 4 warnings; 133.32 s | Seven added hygiene/report checks plus all existing regressions |

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
