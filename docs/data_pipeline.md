# Shared data pipeline and contracts

Implementation: `src/merit_covid/data/pipeline/`.
Both public Dataset APIs and the standalone baseline patch adapter reuse this
core. `ct_pipeline` is a legacy import bridge. For model training commands see
[MERIT execution](merit_pipeline.md).

## Data, identity and the saved split

The legacy manifest is a list of records containing `id`, `image` and `mask`;
optional metadata includes shape, spacing and patient/group identity. Versioned
manifests are also supported. Paths resolve relative to the manifest, not the
current working directory. Duplicate IDs/paths, missing required files and
inconsistent partial patient mappings are rejected.

Grouping prefers recorded patient IDs, then group IDs, then volume IDs.
A complete verified patient-map JSON may supply `case_id -> patient_id`.
Filnames are never used to infer clinical identity. Current primary metadata
does not independently establish patient-level independence.

The canonical `splits/covid19_20_5fold.json` contains seed-42 grouped folds:
folds 0-3 have `127/32/40` train/validation/test volumes; fold 4 has `128/32/39`.
Each outer test group appears once. Validation takes approximately 20% of the
remaining development volumes. Groups remain disjoint in every fold.
The loader validates IDs, partitions and the manifest fingerprint before use.
The fingerprint identifies names/grouping, not the bytes of every CT voxel.
Do not regenerate the split before training.

Zenodo20 is external-only and must not enter development partitions or tuning.
A new verified patient mapping requires a separately named split and protocol.

## Geometry and coordinate systems

`inspect_pair` checks 3D shape, finite/invertible affine, spacing consistency,
units, orientation and paired CT/mask geometry before voxel preprocessing.
It never resizes a mismatched mask to hide misalignment.

Supported header units are converted to mm. Unknown units fail by default.
`assume_mm=True` is an explicit recorded assumption, not source verification.
Primary cases A-0161, A-0585 and A-0599 have unknown units.

Reorientation permutes/flips arrays to LPS without interpolation or deobliquing.
NiBabel world coordinates remain RAS+. NIfTI arrays/affines use XYZ; network
volumes use ZYX, and samples use `[1,D,H,W]` with H=Y, W=X.
Canonical slice indices can differ from original file indices after flips or
axis permutations. Original shape, affine, unit conversion and inverse
orientation are retained for export and evaluation.

Slice-center positions come from the full affine. The signed stack distance is
the projection along its Z-column direction, including oblique/sheared geometry;
it is not merely the world-Z coordinate difference or slice thickness.

## Intensity and resizing

NiBabel applies source slope/intercept before finite-value and binary-label
checks. All masks must contain exactly labels 0/1; no automatic relabeling.

| Encoding | Behavior |
| --- | --- |
| `native` (default) | Preserve scaled CT values; no window, normalization or HU inference |
| `hu` | Caller-declared HU with an explicit window |
| `auto_hu` | Legacy range sanity check plus explicit window; not HU provenance verification |
| `normalized_0_1` | Validate an already normalized range; do not normalize it again |

Nonnative encodings warn when `intensity_provenance` is absent. A native encoding
cannot silently activate windowing. No default per-volume min-max normalization
is used. Viewer/PNG grayscale scaling is display-only.

`size_hw=None` preserves native XY. Explicit resizing uses bilinear CT with
`align_corners=False` and nearest-exact binary masks. Z is not interpolated.
The older MERIT index/physical configs' `128x128` grid is an explicit experiment
choice, not the shared-reader default. Primary `merit_native512.yaml` uses
`size_hw: null`: no resize, and native/model XY mismatch is rejected from
headers before training. Derived auxiliary targets therefore retain the native
GT support on this path; this does not repair the older resize ablations.

`lesion_retention` reports completely lost positive slices and 2D 4-connected
components under the exact nearest-exact pixel-center mapping. Components can
disappear even if another lesion keeps the slice positive. This report never
automatically excludes a case. The stored historical native-resolution cohort
report did not run this newer resize audit.

## Sampling

Default `slab_size=11` places the master at local index 5; the sampler also
supports 1/3/5 slices.

- `index` selects consecutive indices around the master; edge replication is
  default and reflect is optional.
- `physical_mm` requires an explicit finite `target_spacing_mm=d>0`. It requests
  `[-5d,...,0,...,5d]` for 11 slices, finds nearest native centers using the affine
  and clips to endpoints. Ties within 1e-7 voxel select the lower canonical
  index. Only edge handling is supported; there is no Z interpolation.
- Requested offsets and actual selected offsets are both retained. Sparse CT
  stacks can map multiple requested targets to one actual slice.
- `duplicate_slice_policy=unique` supervises one representative per selected
  native slice within each slab. Prefer the master, then in-bounds requests,
  then proximity to the local master, then lower local index. A clamped real
  endpoint still retains one valid representative even if all its requests
  are out of bounds. `all` is an explicit repeated-supervision ablation.
- Validity affects loss only; duplicate context still enters the model.

## Tensor Dataset contract

One sample represents one master, including masters with empty masks.
After collation:

| Field | Batched value |
| --- | --- |
| `image_slab` | float32 `[B,1,D,H,W]` |
| `mask_slab` | float32 binary `[B,1,D,H,W]`; true GT at selected native indices |
| `master_mask` | float32 binary `[B,1,H,W]`; equals the slab target at the local master |
| `master_index` | int64 `[B]`; slab-local position |
| `master_slice_index` | int64 `[B]`; canonical volume index |
| `slice_indices` | int64 `[B,D]`; actual selected indices |
| `slice_positions_mm` | float64 `[B,D,3]`; RAS+ world centers |
| `z_offsets_mm` / `requested_offsets_mm` | float32 `[B,D]`; actual/requested distances |
| `slice_valid` | bool `[B,D]`; valid loss representatives |
| `valid_region` | bool `[B,1,D,H,W]`; loss-valid voxels |
| `case_id` / `group_id` | Lists of B identifiers |
| `metadata` | List of B geometry/config dictionaries |
| `sampling_mode` / `target_spacing_mm` | Sampling policy and configured nullable spacing |

`mask_slab` is never a repeated copy of `master_mask`.
`require_mask=False` removes label paths before all header/voxel reading and
omits GT outputs. Split inference retains recorded mask filenames for fingerprint
validation even if those mask files are absent.

Initialization reads selected headers only. Each worker caches at most one
processed volume, releasing it before loading another; cache is not pickled to
Windows workers. `len(dataset)` counts master slices, not volumes.
Random slice shuffle can repeatedly decompress CT files; no cohort preloading
is used to conceal that cost. MERIT experiment configs explicitly use volume
block shuffling with randomized masters inside each case; see
[training order and resume](merit_pipeline.md#preprocessing-and-supervision).
The shared loader keeps global slice shuffling as its compatibility default.

Collation checks compatible fields, stacks tensors, and keeps metadata/nullable
values separate. Use batch size 1 or shape buckets when native H/W differ.
Windows multiprocessing needs an `if __name__ == "__main__":` guard.

```python
from merit_covid import PreprocessConfig, build_fold_loaders

if __name__ == "__main__":
    loaders = build_fold_loaders(
        "data/covid19_20/cases.json", "splits/covid19_20_5fold.json",
        fold=0, preprocess=PreprocessConfig(), batch_size=1, num_workers=0,
    )
    batch = next(iter(loaders.train))
    image = batch["image_slab"]
    full_target = batch["mask_slab"]
    master_target = batch["master_mask"]
```

The default example rejects unknown-unit headers. Pass the explicit mm assumption
only when justified. The older NumPy `SlabDataset` retains `image`, master `mask`,
`context_indices`, `slice_index` and slab-local `master_index` keys. Both APIs
share values and geometry; only the newer tensor contract includes full-slab GT.

## Reconstruction and prediction output

`SliceReconstructor` requires each canonical probability slice exactly once;
it rejects missing/duplicate indices, invalid shapes, nonfinite values and
probabilities outside [0,1]. It accepts probabilities, not logits.

`VolumePrediction` contains `case_id`, float32 `probability_zyx [Z,H,W]` on the
complete native-size LPS grid, and restoration metadata. Incomplete/resized or
mismatched grids are rejected. `prediction_policy` records model-specific
preparation and output choices.

Baseline patches are averaged as probabilities over real overlapping voxels,
excluding padding. Master-slice outputs use the same existing reconstructor.
`to_original_xyz()` delegates to `restore_volume`. Restore probabilities to
native XY/orientation before thresholding; do not resize thresholded masks and
report them as equivalent native predictions.

Binary one-channel logits use sigmoid. MERIT two-channel logits use foreground
softmax, equivalently `sigmoid(fg-bg)`, not sigmoid of the foreground alone.

## Metrics

Foreground Dice/IoU, precision and recall use TP/FP/FN over the whole original
volume. Never average slice Dice and call it volume Dice.

| Masks | Dice / IoU / precision / recall | HD95 |
| --- | --- | --- |
| Both empty | 1 / 1 / 1 / 1 | null |
| Exactly one empty | 0 / 0 / 0 / 0 | null |
| Identical nonempty | 1 / 1 / 1 / 1 | 0 mm |

HD95 uses 6-connected voxel surfaces, full-affine world positions and the 95th
percentile of pooled bidirectional nearest-surface distances. Undefined values
are null, excluded from that metric's mean and counted explicitly.
Lesion volume uses `abs(det(affine[:3,:3]))/1000` mL per voxel; relative volume
error is null when GT is empty.

Average volume metrics within each recorded group, then compute equal-weight
group macro and population standard deviation. `sensitivity` is a compatibility
alias for recall. With unverified patient linkage, call results group/volume
macro, not independently verified patient macro.
Fix threshold using development policy; never select it from test labels.

## Data commands

```powershell
py tools/data_pipeline.py --help
py scripts/read_training_batch.py --fold 0 --partition train --batch-size 1
py scripts/merit_pipeline.py check-split --config configs/merit_index.yaml
py scripts/merit_pipeline.py audit --config configs/merit_index.yaml --limit 1 --output outputs/data_audit.json
# Only if a separately versioned split is genuinely needed:
py scripts/make_splits.py data/covid19_20/cases.json --output outputs/new_split.json
# Requires complete saved predictions matching the preprocessing policy:
py tools/data_pipeline.py evaluate --fold 0 --partition test --predictions outputs/predictions --threshold 0.5 --hd95 --output outputs/metrics.json
```

The generic data CLI is for data inspection or already exported predictions.
The MERIT CLI adds physical experiment settings, training and checkpoint-based
inference. Neither command should silently regenerate the canonical split.
