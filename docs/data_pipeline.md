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

---

## Dataset reports and CSV reference

Stored statistics describe CT and ground-truth masks, not model predictions.
They are not mandatory Dataset inputs. Original NIfTI files plus a manifest and
the saved split supply training tensors.

### Primary report

`analysis/covid19_20/` contains:

| Artifact | Meaning |
| --- | --- |
| [REPORT.md](../analysis/covid19_20/REPORT.md) | Measured cohort overview and limitations |
| [case_statistics.csv](../analysis/covid19_20/case_statistics.csv) | One row per volume, 199 rows and 71 saved columns |
| [summary.json](../analysis/covid19_20/summary.json) | Aggregates and per-slice GT areas |
| [audit.json](../analysis/covid19_20/audit.json) | Saved full CT/mask checks with explicit mm assumption |
| [audit_strict.json](../analysis/covid19_20/audit_strict.json) | Strict-unit checks; three unknown-unit cases rejected |
| [dataset_overview.png](../analysis/covid19_20/dataset_overview.png) | Spacing, length, burden and positive/empty slices |
| [lesion_trajectories.png](../analysis/covid19_20/lesion_trajectories.png) | GT area across physical stack distance |
| [ct_mask_examples.png](../analysis/covid19_20/ct_mask_examples.png) | Middle-slice CT, GT and overlay, not AI predictions |

The saved report uses LPS arrays, native XY and native NIfTI-scaled intensities.
There are 13,705 slices: 4,981 lesion-positive and 8,724 empty.
No primary lung mask is provided: lesion fraction uses **all CT voxels**, not
lung volume. Adjacent GT Dice measures continuity between labels on neighboring
slices, not segmentation accuracy.

Patient IDs remain unverified/blank. Physical measurements for A-0161, A-0585
and A-0599 depend on the declared mm assumption. Headers/masks were rechecked;
full CT intensity/finite checks were reused from the earlier saved audit.
No claim of source CT byte identity or HU calibration is made.

### Complete saved primary CSV schema

Lists/matrices/dictionaries in CSV cells are JSON. Empty cells can represent
unknown patient identity or null results; do not interpret them as zeros.
Field order is preserved for existing reports. New runtime sampling/retention
metadata is not retroactively inserted into the historical 71-column CSV.

| # | Field | Definition |
| ---: | --- | --- |
| 1 | `volume_order` | One-based row/plot order, not severity or patient identity |
| 2 | `case_id` | Volume identifier |
| 3 | `patient_id` | Verified identity if supplied; currently blank |
| 4 | `patient_ids_available` | Whether complete patient identity was supplied |
| 5 | `image` | CT path relative to manifest directory |
| 6 | `mask` | Binary infection-mask path relative to manifest |
| 7 | `group_id` | Group used by split validation; currently volume ID |
| 8 | `original_shape_x` | Original NIfTI X size |
| 9 | `original_shape_y` | Original NIfTI Y size |
| 10 | `original_shape_z` | Original NIfTI Z size |
| 11 | `original_orientation` | Original array axis codes |
| 12 | `canonical_orientation` | Standardized array axis codes, LPS |
| 13 | `spacing_x_mm` | Original affine X-vector norm in mm |
| 14 | `spacing_y_mm` | Original affine Y-vector norm in mm |
| 15 | `spacing_z_mm` | Original affine Z-vector norm in mm, not thickness |
| 16 | `spatial_unit_in_header` | Source spatial-unit declaration |
| 17 | `spatial_units_assumed_mm` | This specific case required the mm assumption |
| 18 | `ct_dtype` | Stored CT voxel dtype |
| 19 | `mask_dtype` | Stored mask voxel dtype |
| 20 | `ct_min` | Saved native scaled CT minimum |
| 21 | `ct_max` | Saved native scaled CT maximum |
| 22 | `ct_scaling_slope` | NIfTI CT scaling multiplier |
| 23 | `ct_scaling_intercept` | NIfTI CT scaling offset |
| 24 | `mask_label_values` | Observed native mask labels, [0,1] |
| 25 | `geometry_match` | Paired geometry validation passed |
| 26 | `mask_finite_binary` | Complete mask finite and exactly binary |
| 27 | `z_resampled` | Whether Z interpolation occurred; false here |
| 28 | `slices` | Canonical native slice count before padding |
| 29 | `positive_slices` | Slices containing any GT lesion voxel |
| 30 | `empty_slices` | Slices with empty GT, not missing/corrupt images |
| 31 | `positive_slice_fraction` | Positive-slice count / slice count |
| 32 | `first_positive_slice` | First positive canonical zero-based index, or null |
| 33 | `last_positive_slice` | Last positive canonical zero-based index, or null |
| 34 | `positive_run_count` | Number of contiguous positive-slice runs |
| 35 | `max_positive_run_length` | Length of longest positive run |
| 36 | `positive_run_lengths` | JSON list of run lengths |
| 37 | `lesion_voxels` | Total native foreground voxel count |
| 38 | `total_voxels` | Total CT-grid voxel count |
| 39 | `lesion_fraction_volume` | Lesion / all CT voxels; not fraction of lung |
| 40 | `voxel_volume_mm3` | Absolute determinant of affine spatial block |
| 41 | `lesion_volume_ml` | Lesion voxels * voxel volume / 1000 |
| 42 | `pixel_area_mm2` | Norm of cross product of canonical in-plane vectors |
| 43 | `slice_center_step_mm` | Norm of canonical affine Z vector |
| 44 | `adjacent_positive_pairs` | Neighboring slice pairs with both GTs positive |
| 45 | `adjacent_gt_dice_median` | Median GT-to-GT overlap over those pairs |
| 46 | `audit_original_shape_xyz` | Saved original XYZ shape |
| 47 | `audit_canonical_shape_xyz` | Saved native canonical XYZ shape |
| 48 | `audit_original_affine` | Original voxel-to-world transform in header units |
| 49 | `audit_original_affine_mm` | Original transform converted/assumed to mm |
| 50 | `audit_canonical_affine_mm` | LPS voxel indices to RAS+ world mm |
| 51 | `audit_canonical_to_original_orientation` | Inverse axis permutation/flips |
| 52 | `audit_original_orientation` | Original axis codes as JSON |
| 53 | `audit_canonical_orientation` | Canonical axis codes as JSON |
| 54 | `audit_native_spacing_mm` | Canonical affine-vector norms before XY resize |
| 55 | `audit_processed_size_hw` | Processed H/W used in the saved audit |
| 56 | `audit_native_size_hw` | Native canonical H/W |
| 57 | `audit_spatial_unit_in_header` | Saved unit declaration |
| 58 | `audit_assume_mm` | Run-level permission to assume unknown units are mm |
| 59 | `audit_scaled_intensity_range` | Native scaled CT min/max before windowing |
| 60 | `audit_intensity_encoding` | Declared processing policy, native here |
| 61 | `audit_label_values` | Saved observed GT labels |
| 62 | `audit_native_lesion_voxels` | Saved foreground count, rechecked by report |
| 63 | `audit_nifti_scaling` | CT slope/intercept dictionary |
| 64 | `audit_hu_window` | Explicit window; blank means null/disabled |
| 65 | `audit_padding` | Slab boundary policy; replicate_boundary |
| 66 | `audit_z_resampled` | Saved Z-interpolation flag |
| 67 | `fold_0` | Role in saved fold 0 |
| 68 | `fold_1` | Role in saved fold 1 |
| 69 | `fold_2` | Role in saved fold 2 |
| 70 | `fold_3` | Role in saved fold 3 |
| 71 | `fold_4` | Role in saved fold 4 |

`audit_assume_mm=True` is a run-wide permission; it does not mean every volume
used an assumption. A case with declared mm can still have
`spatial_units_assumed_mm=False`.

### Worked example: A-0025

Actual CSV row 6, `volume-covid19-A-0025`, rounded here for readability:

| Quantity | Value and interpretation |
| --- | --- |
| Shape/orientation | XYZ 512x512x61, already LPS; no reorientation needed |
| Spacing | Approximately 0.846 x 0.846 x 5 mm |
| Intensity | int16 stored, slope 1/intercept 0, scaled range [-2048,1672] |
| GT | 402,334 foreground voxels; labels [0,1] |
| Slice occupancy | 40 positive, 21 empty; positive indices 10 through 49 |
| Positive runs | One run of 40; 39 adjacent positive pairs |
| Slice fraction | 40/61 = 65.57%; this is not lesion volume fraction |
| Lesion CT fraction | 402,334 / 15,990,784 = approximately 2.516% |
| Physical lesion volume | Approximately 1,439.784 mL, from affine determinant |
| Adjacent GT Dice | Median approximately 0.833942; no model involved |
| Fold roles 0/1/2/3/4 | train / train / test / train / train |

Original and canonical mm affine are identical for this case (rounded):

```text
[-0.846,  0,      0,   216.373001]
[ 0,     -0.846,  0,   216.373398]
[ 0,      0,      5,  -252.5     ]
[ 0,      0,      0,     1       ]
```

LPS array orientation does not change NiBabel's RAS+ world convention.
With index sampling and master 30, the 11 selected slices are 25-35 and actual
offsets are -25,-20,...,0,...,20,25 mm. Native sample shape is
`[1,11,512,512]`; the explicit MERIT experiment later resizes XY.

### Historical Zenodo20 report and provenance

`analysis/zenodo20/case_statistics.csv` and `summary.json` describe 20 volumes
from Coronacases/Radiopaedia (10 each), totaling 3,520 slices: 1,844 positive and
1,676 empty. The two PNGs show case balance and example GT trajectories.

Important historical provenance: CT came from a Kaggle mirror; infection/lung
masks came from original Zenodo ZIPs. The mirror combined-mask label 3 matched
infection labels, and its nonzero region matched the lung/infection union for
the audited volumes. This does **not** prove CT byte identity with Zenodo's CT
archive. The old audit also recorded 114 infection voxels outside lung labels
in three Radiopaedia volumes.

Historical analysis definitions:

- Infection fraction of lung uses lung-mask voxels as denominator; unlike the
  primary whole-CT fraction, it is not directly comparable.
- Components are 2D **8-connected** foreground regions. Historical OpenCV nearest
  resize checks at 128/256/352 reported 38/19/6 lost components and a lost
  positive slice at 128/256.
- The newer shared retention checker uses **4-connectivity and nearest-exact**.
  Different component/interpolation definitions must not be merged into one result.
- Adjacent Dice compares neighboring GT masks only when both are positive.
- CT intensity percentiles used stride (8,8,4) samples, not every voxel.
- Geometry match means paired shape, affine and spacing agreement.

These existing external descriptive artifacts were not regenerated by the MERIT
repair or repository cleanup. A local MADGNet reference copy was used only to
inspect evaluation/split code, not to train a model. External labels must not
guide development choices.

### Regeneration and safe interpretation

With a complete primary audit and source data already available:

```powershell
py tools/summarize_covid19_20.py --assume-mm
```

The flag deliberately accepts the stated unit assumption; verify upstream evidence
before using it. The report script reads masks/headers, checks agreement with
the saved audit, and writes primary CSV/JSON/PNG/REPORT outputs. It does not
reread every CT intensity voxel or train a model. The English report generator
and stored report are maintained together.

Acquisition/legacy external audit tools are retained for explicit use, but are
not prerequisites to inspect stored reports. Do not rerun external acquisition
or analysis as part of a training setup or documentation check.

---

## Native patch input and unified volume output

The standalone baseline consumes native 3D crops rather than resized whole
slices. It reuses the shared LPS/intensity/geometry reader and does not alter the
slab Dataset or UNETR++ architecture. Model background:
[UNETR++ baseline](architecture.md#standalone-unetr-baseline).

### A real-volume example

A-0025 has canonical shape `[Z,Y,X]=[61,512,512]`.
Reference patch size is `(64,128,128)`, default stride the same.
Four Y starts and four X starts (0/128/256/384) produce 16 patches.
Each contains 61 real CT slices plus 3 trailing Z-padding slices.

Crop preserves native voxel values/spacing; it is not 512->128 resize.
All windows cover the volume independently of lesion masks.
One forwarded patch is not a complete CT prediction.

### Window and padding policy

```python
from merit_covid.data.unetrpp_input import PatchConfig

config = PatchConfig(
    patch_size=(64,128,128),  # Z,Y,X
    stride=None,
    overlap=None,
    padding_mode="edge",
    assume_mm=False,
)
```

Each axis uses positive strides no larger than the patch. The final window
ends at the volume boundary: for Z=130 and stride 64, starts are 0/64/66.
Thus even nominal zero overlap can have an overlapping final window.
All three axes follow the same coverage rule.

A short dimension is padded only at its trailing end; real voxel coordinates
do not shift. CT defaults to edge padding. Explicit constant padding is allowed,
but no value is assumed to be air HU. GT pads with zero while
`valid_region` marks padding false; padded zeros are not real negative labels.

Specify either `stride=(32,64,64)` or `overlap=(0.5,0.5,0.5)`.
Overlap must be in [0,1); stride is
`max(1,floor(patch_size*(1-overlap)))`.
Patch/model fixed grids must match. Other sizes can test geometry but do not
automatically reconfigure the architecture.

### Dataset, split and batch contract

```python
from merit_covid.data.unetrpp_input import (
    PatchConfig, UNETRPPPatchDataset, make_patch_dataloader,
)

if __name__ == "__main__":
    dataset = UNETRPPPatchDataset.from_split(
        manifest="data/covid19_20/cases.json",
        split_path="splits/covid19_20_5fold.json",
        fold=0, partition="train", config=PatchConfig(),
        require_mask=False,
        case_ids=["volume-covid19-A-0025"],
    )
    batch = next(iter(make_patch_dataloader(
        dataset, batch_size=1, num_workers=0,
    )))
```

Split identity/grouping/fingerprint is validated before filtering cases.
`case_ids` can only subset the requested partition. `len(dataset)` counts
patches; `len(dataset.cases)` counts volumes. No split is regenerated.

| Field | Batched value |
| --- | --- |
| `image_patch` | float32 `[B,1,D,H,W]`, native CT crop plus padding |
| `valid_region` | bool, same shape; true only for real voxels |
| `mask_patch` | Optional true binary 3D GT, same shape, when `require_mask=True` |
| `patch_index` | integer `[B]`, local to each volume |
| `case_id` / `group_id` | B identifiers |
| `metadata` | B dictionaries: geometry, coordinates, policy, split provenance |

Metadata records zero-based, end-exclusive `start_zyx/stop_zyx`, `padding_zyx`,
real valid shape, patch count, canonical volume shape, original XYZ shape,
affines and inverse orientation. Convert `(z,y,x)` to `(x,y,z)` before applying
an XYZ affine to patch coordinates.

CT-only inference strips GT paths before all header/voxel reads. Recorded
filenames remain available for split fingerprint checks; the actual GT files
need not exist. Labeled mode requires aligned real masks, never repeated master
targets. Unknown-unit policy remains strict unless explicitly overridden after
verification.

Only selected headers are read during initialization. Voxel access uses shared
`read_arrays` and a one-volume cache per worker; Windows pickling drops caches.
Validation/test cannot shuffle. Global random patch shuffle can repeatedly
decompress CTs; the implementation does not preload the cohort.

### Forward a real patch

```powershell
py scripts/unetrpp_real_input.py
py scripts/unetrpp_real_input.py --report outputs/real_input_check.json
py scripts/unetrpp_real_input.py --workers 1 --patch-index 15
```

The example defaults to one A-0025 train patch with the full 42,951,355-parameter
model in eval/inference mode. It checks finite output, source voxel agreement
and unchanged parameter hashes; no backward, optimizer, GT-based selection or
clinical quality evaluation occurs.

Historical observed input: `[1,1,64,128,128]`, 999,424 real voxels and 49,152
padding voxels. GT was not read and CT source size/mtime remained unchanged.
This proves a software connection, not model accuracy.

### Complete probability reconstruction

`predict_patch_batch` selects the main deep-supervision output and converts
one-channel logits by sigmoid outside the model.
`PatchReconstructor` inserts only the real valid extent at recorded coordinates
and averages **probabilities**, not logits, over overlap:

```text
probability[z,y,x] = sum(real patch probabilities at voxel) / coverage count
```

Padding contributes neither numerator nor denominator. Do not threshold each
patch, average patch metrics, or mix coarse auxiliary logits into the native grid.
Wrong case/coordinates/geometry/policy, duplicate/missing patches, uncovered
voxels, invalid shapes and invalid probability values are rejected.

```python
from merit_covid.models.unetrpp import UNETRPP
from merit_covid.engine.patch_inference import predict_patch_case

# Use the CT-only dataset created above.
model = UNETRPP(in_channels=1, out_channels=1, do_ds=True).eval()
prediction = predict_patch_case(model, dataset, case_index=0, batch_size=1)
native_lps_zyx = prediction.probability_zyx  # Complete real CT, without padding.
original_xyz = prediction.to_original_xyz()
```

This forwards all patches for one selected case; it does not train or compute
metrics. The sum/count buffers cover one full CT in RAM, alongside the reader's
one-volume cache; memory is not constant in volume size.
Full reconstruction is tested with synthetic volumes. The historical real-data
check forwarded one patch, not all 199 cases.

MERIT is explicitly rejected by this helper because its master-index/four-output
contract differs. Use `predict_slab_case` for MERIT.

### Shared output and evaluation

`VolumePrediction` contains case ID, finite float32 `[Z,H,W]` probabilities in
[0,1] on the complete native LPS grid, and original-geometry metadata plus
`prediction_policy`. Incomplete or resized grids cannot masquerade as native.

Master/2D models can use `prediction_from_slices`, which delegates to the existing
strict `SliceReconstructor` and returns the same output type.
Two-class heads must first compute foreground softmax (or `sigmoid(fg-bg)`);
the one-channel helper cannot do that conversion implicitly.

`prediction.slice_predictions()` bridges to the existing `VolumeEvaluator`.
Probabilities restore to original geometry before thresholding against original
GT. Compare models on the same saved cases/partitions, labels, geometry,
threshold-selection policy and group aggregation. Different input grids are
allowed but must be recorded, not hidden.

### Source and tests

Read in order: `data/unetrpp_input/config.py` -> `tiling.py` -> `dataset.py` ->
`engine/prediction.py` -> `patch_reconstruction.py` -> `patch_inference.py`.
No second CT reader, reorientation implementation or evaluator is introduced.

```powershell
py -m pytest tests/test_unetrpp_input.py tests/test_prediction_contract.py -q
```

Coverage includes native/scaled voxel equality, orientation flips/permutations,
affine shear, paired GT, all-axis padding, complete coverage, probability
averaging, inverse geometry, CT-only access, cache/Windows workers and split
integrity. Training choices and clinical results are separate research work;
see [validation](validation.md).
