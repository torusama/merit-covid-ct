# Changelog

Major milestones are listed below. Current behavior belongs in the topic guides,
not in historical implementation notes. Numeric verification artifacts in
analysis/ retain their original snapshot content.

## 2026-11-13 - MERIT runtime and source-binding repair

- Record the completed MER-only native A-0025/T4 comparison from the user's
  second account. Two updates pass with identical losses/final parameter hash;
  retain the all-region pilot reference (4.54 versus 9.00 GiB peak allocation)
  because the observed 1.59% timing reduction is inconclusive. Mark the current
  runtime repair-and-pilot phase complete, awaiting team training results,
  while retaining explicit production integration and 128-grid issue limits.
- Rewrite the root README for handoff: reading order, source responsibilities,
  real-case Colab entry, input transfer, pinned-source behavior, measured native
  results and outstanding production work. Distinguish 128-grid research configs
  from native-512 pilots and experimental recomputation from saved checkpoints.
- Preserve selected user-supplied cloud JSON and archive checksums under
  `analysis/merit_native512/`. Latest full CPU verification: 389 passed and one
  CUDA-only skip, with no failures or errors.
- Add one scoped MER-only checkpointing notebook for A-0025. Reuse the previous
  helpers unchanged, gate the candidate on numerical/buffer checks, and export
  a comparison with the existing real-case baseline. Test the workflow with
  compact CPU/GPU fixtures; the subsequent native-512 T4 comparison is recorded
  in improvement note 01.
- Add the single-real-case Colab notebook, an input ZIP for local delivery,
  and a bounded pilot helper. Verify A-0025 source SHA-256 and frozen TRAIN
  membership, preserve native CT/GT pixels, attempt two optimizer updates and
  check one CT-only prediction. Keep images/large weights outside report ZIPs.
- Add an isolated activation-checkpointing trial and a self-contained cloud
  notebook. Preserve model parameter/state identities, attention equations and
  data/loss policy; recompute MER query blocks and stateless backbone regions,
  excluding BatchNorm updates. Gate profiling on output/gradient/buffer checks.
- Validate the execution overlay on CPU/GPU and complete two local 128-grid
  optimizer steps with lower peak memory. Leave production model source and
  trainer defaults unchanged. The user-supplied Colab T4 report subsequently
  confirms two optimized native-512 synthetic updates with peak PyTorch
  allocation 4650.55 MiB. The subsequent real A-0025 pilot also passes two native
  updates with unchanged input files, finite gradients and CT-only inference;
  this is not full-cohort training or a segmentation-quality result.
- Replace the machine-specific Drive-based connection notebook with a portable
  Colab/Kaggle Run-all preflight. Cache pinned public source, reuse installed
  Torch/CUDA, install only missing dependencies under Torch/NumPy constraints,
  isolate GPU jobs, and export reports even after OOM. No dataset is required.
- Check notebook syntax/worker generation, missing-GPU rejection, and cached
  dependency setup. Execute all notebook cells locally against the pinned
  model on the GPU: 128 profile passes, 512 reports OOM, ZIP export completes.
- Add the separate `merit_native512.yaml` route-1 feasibility config and a
  TRAIN-only `native-input` diagnostic verifying native CT/GT equality without
  XY interpolation or source-file changes. Keep the 128x128 configs available.
- Add an explicit CUDA allocator budget option to the isolated optimizer
  profiler, retaining normal profiling behavior when it is omitted.
- Native route-1 input checks pass. The full 512x512/batch-1/AMP configuration
  runs out of memory in encoder EPA forward on the tested GTX 1650 Max-Q 4 GiB,
  before any optimizer update. The previously skipped compact CUDA AMP test
  passes in the separate GPU environment; no quality comparison was run.

- Bind checkpoints to resolved CT and GT paths, file size/mtime, manifest and
  patient-map SHA-256, and the full data configuration. Reject stale sources,
  redirected GT, incompatible sample order and decreased epoch budgets on resume.
- Write schema-2 checkpoints. Historical schema-1 models remain readable for
  CT-only inference; their incomplete source identity cannot authorize resume.
- Reject invalid or nonfinite master logits before sigmoid, and use the shared
  strict threshold validator when loading an experiment, before training.
- Add reproducible volume-block shuffling to the MERIT experiment configs to
  avoid repeated whole-volume decompression with the one-volume cache. Retain
  global slice shuffling in the shared API and record sample order in configs.
- Restore model-only synthetic and real baseline input scripts under scripts/;
  repair guide commands and check documented Python entry points in tests.
- Add fault-injection/source-drift regressions and exact epoch-boundary resume
  coverage for both shuffle modes. The existing 128x128 resize/auxiliary-label
  policy remains open for design; the separate native-512 feasibility experiment
  does not alter it. Model equations, intensity handling, canonical splits and
  raw data are unchanged.

## 2026-11-08 - Train-only preflight diagnostics

- Add bounded-memory native CT moments, all-voxel percentiles/histogram and
  128/256 label-retention CSV/figures over the selected fold TRAIN cases only.
- Check all five folds for case/group/path overlap without claiming patient
  identity from volume IDs. Preserve canonical splits and all source images.
- Add isolated full-optimizer GPU profiling and a bounded actual-CT pilot using
  production Trainer, fresh checkpoint resume and CT-only native reconstruction.
- Keep HU calibration and normalization decisions explicit; no automatic HU
  window, full-epoch training or test-based image-size selection.
- Repair the broken batch-inspection entry point after example removal by
  retaining it under scripts/ and updating its caller; keep all test assertions.

## 2026-11-08 - Repository cleanup

- Keep one root README and eight topic-oriented guides under docs/.
- Consolidate data contracts into the data guide, the 71-field CSV walkthrough
  and external-report provenance into the dataset-report guide, and historical
  implementation/merge/repair evidence into validation.md.
- Translate maintained documentation, project comments/docstrings and generated
  primary report text to English; correct obsolete "no model/trainer" statements.
- Remove redundant per-folder READMEs, superseded status/roadmap reports and
  unused machine-specific comparison scripts.
- Move pytest configuration into pyproject.toml without changing test discovery
  or import-mode behavior.
- Preserve source APIs, canonical split, datasets, model artifacts, third-party
  source/license/provenance and historical numeric analysis.

## 2026-11-08 - MERIT integration and pipeline repair

- Add Z-preserving UNETR++ with paper-derived TEP/MER/PCG and full-slab loss.
- Add explicit index/nearest-mm sampling, true slab GT, duplicate validity,
  optional physical PE/Gaussian weighting and CT-only master reconstruction.
- Repair CUDA autocast probability BCE and baseline/MERIT inference mismatch.
- Add AdamW/AMP trainer, native validation, frozen config, checkpoint/resume,
  experiment CLI and CPU/CUDA synthetic integration checks.
- Verify 325 tests passing before the subsequent documentation cleanup.
  No full real COVID training or external evaluation.

## 2026-11-08 - Standalone baseline and native input/output

- Integrate the pinned Synapse UNETR++ architecture with Apache-2.0 attribution,
  model-only config, master adapter and upstream fidelity checks.
- Add native 3D crop/tiling/padding, optional real 3D targets, strict validity,
  probability-mean patch assembly and the common VolumePrediction contract.
- Verify one real A-0025 patch; complete reconstruction checked synthetically.
  Historical suite milestones: 188 then 262 passing tests.

## 2026-11-07 - Main/Duong integration

- Start MERGE from main 4fa8753 and integrate Duong 5127e52, retaining both histories.
- Place the authoritative data core in src/merit_covid/data/pipeline/.
  Retain main NumPy APIs and ct_pipeline import aliases.
- Keep LPS, native XY/intensities, strict unknown-unit behavior with explicit
  mm assumption, and Duong's canonical split and group-macro metrics.
- Remove the superseded main split to avoid competing protocols.
- Verify 135 tests, exact pre-merge slab equality and 199 primary audit records.

## 2026-11-07 - Data foundation and descriptive reports

- Add installable package/config, validated manifests, paired NIfTI I/O,
  geometry restoration, Dataset/loaders, grouped splits and model boundaries.
- Repair legacy downloader/module dispatch and actual mask filename resolution.
- Separate geometry, preprocessing, slab sampling, loading and reporting.
- Standardize LPS; disable default XY resize and intensity normalization.
- Save primary audit/CSV/figures and retain historical external descriptive
  statistics without reinterpreting them as clinical model results.
- Record unverified patient linkage, three unknown-unit cases and intensity
  provenance limits. Earlier implementation counts are historical milestones,
  not current test totals.
