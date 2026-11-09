# Changelog

Major milestones are listed below. Current behavior belongs in the topic guides,
not in historical implementation notes. Numeric verification artifacts in
analysis/ retain their original snapshot content.

## 2026-11-09 - Colab training preparation

- Set the MERIT index config's explicit unknown-spatial-unit mm assumption.
- Add a 256x256 Colab training config and a separate pilot-gated notebook with
  Drive checkpoint/resume, fixed run provenance and no automatic test evaluation.
- Add seeded case-grouped master shuffling to avoid repeated full-volume loads;
  preserve the shared Trainer and test epoch-boundary replay on synthetic data.
- Keep native intensity; HU calibration, patient linkage, normalization and
  clinical performance remain unverified.

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
