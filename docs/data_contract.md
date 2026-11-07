# Data contract

## Manifest version 1

Paths resolve relative to the manifest, never the current working directory.
Writers create new files exclusively; legacy `cases.json` is not overwritten.

```json
{
  "schema_version": 1,
  "cases": [{
    "dataset": "covid19_20",
    "case_id": "example-volume",
    "patient_id": null,
    "source": "unknown",
    "image": "Train/example_ct.nii.gz",
    "infection_mask": "Train/example_seg.nii.gz",
    "lung_mask": null,
    "shape": [512, 512, 100],
    "spacing_mm": [0.7, 0.7, 5.0],
    "affine": [[0.7,0,0,0],[0,0.7,0,0],[0,0,5,0],[0,0,0,1]],
    "orientation": ["R", "A", "S"],
    "intensity": "unknown"
  }]
}
```

Required: supported dataset (`covid19_20` or `zenodo20`), case ID and image path.
Labeled Dataset modes also require infection masks. Shape, spacing, affine and
orientation may be absent until verified; provided values are checked against
source NIfTI. The checker can create a separate enriched manifest. Lung masks are
auxiliary and never define a ground-truth test crop.

Patient IDs are null unless independently verified. No filename parser invents
patient identities. Intensity is `unknown`, `HU` or `windowed_255`, never inferred
from ranges or filenames. Record identity/intensity/unit evidence in experiment
provenance; declaring an ID/unit/intensity does not itself establish that evidence.

Legacy lists remain readable: `id` maps to `case_id`, `mask` maps to
`infection_mask`, dataset comes from an argument or parent directory name. Source,
shape and spacing are retained. Existing combined-mask fields are not silently
converted to lesion masks. Generic legacy reading leaves omitted lung paths null;
the viewer/audit resolver indexes actual filenames only when a lung path was
omitted and requires one unique source-ID match. A recorded missing path fails.

Validation rejects duplicate IDs, repeated image/mask paths, unsupported datasets,
invalid metadata and missing listed files. It cannot prove that two differently
named files contain distinct patients or byte-distinct scans.

## Geometry, labels and intensity

CT and masks are finite 3D arrays with equal shape, affine (absolute tolerance
1e-4), header spacing and spatial-unit declarations. There is no silent
registration/geometry repair. Infection masks require integer 0/1 labels.
Multiclass files require an explicit `lesion_labels` argument at the I/O boundary,
e.g. `(3,)` for a verified combined encoding. Nonzero lung labels are never
automatically promoted to lesion. Dataset modes expect dedicated binary masks.

Paired axis permutations/flips produce RAS: right, anterior, superior. Native
axis 2 is not assumed axial. Obliquity is preserved; canonicalization is not
registration or oblique reslicing. Original shape, affine, header, qform/sform,
orientation and units are retained, together with orientation transform and
canonical/preprocessed affines.

`mm`, `meter`, `micron` convert explicitly to millimetres. Unknown units fail
unless `assumed_unit` is supplied with external evidence. The effective
`mm_per_unit` is retained, while exported headers preserve original units. Record
the justification separately. Slice-center positions come from the affine applied
to `[0,0,z,1]`; step is the norm of affine column 2 converted to mm. It means
center-to-center spacing, not slice thickness. Z is never resampled.

For a chosen policy, normalization is shared across modes: `none` copies values;
`zscore` uses the individual CT mean/std (constant image becomes zero);
`hu_window` clips/scales configured bounds only with verified HU;
`windowed_255` divides by 255 only for declared windowed data in [0,255]. No
external pooled statistics are fitted. Fix the policy using primary development
data. Viewer slice-specific windowing is display-only.

XY interpolation is linear for CT and nearest neighbor for masks, using identical
pixel-center coordinates. Aspect ratio can change with an explicit square size.
Affine metadata tracks the change. Default XY remains native. Inverse resizing
restores geometry, not deleted details. Entire positive-slice loss warns and is
recorded; partial component loss remains possible without a warning. Check
internal lesion/component retention before selecting a reduced size. No dilation
or ground-truth ROI crop is introduced.

## Dataset and model shapes

| Field | Sample | Batched | Meaning |
| --- | --- | --- | --- |
| image | `[1,D,H,W]` float32 | `[B,1,D,H,W]` | Canonical/preprocessed slab |
| mask | `[1,H,W]` float32 | `[B,1,H,W]` | Master target; absent in inference |
| master_index | integer | `[B]` | Slab position, usually 5 |
| slice_index | integer | `[B]` | Canonical volume master index |
| context_indices | `[D]` int64 | `[B,D]` | Actual sampled canonical indices |
| z_offsets_mm | `[D]` float32 | `[B,D]` | Signed physical distance of actual sampled slices |
| case_id, dataset | string | string lists | Identity and reconstruction routing |

`H,W` are canonical array axes 0/1 without viewer rotation. Supported depths are
1/3/5/11, requiring the centered master. Edge repeats nearest slices; reflection
folds repeatedly, including short volumes. Repeated slices have repeated offsets
(often zero). Reflection changes index order near edges and records actual
`context_indices`; interior slabs are ordered inferior to superior.

Modes: `train`, `validation`, `internal_test`, `external_test`, `inference`. Only
training accepts an augmentation callback, which must preserve paired slab/master
alignment and identity metadata. No augmentation recipe is implemented. Evaluation
rejects callbacks. Inference drops mask paths before checking or reading files.
Zenodo20 is allowed only in external-test/inference; external-test accepts only
Zenodo20.

The optional DataLoader shuffles training only and seeds torch/Python/NumPy
workers. Native dimensions require batch size 1 when XY varies. Geometry stays
outside batches in the prepared-volume cache. Model adapters return master logits
`[B,1,H,W]`; 2D models select the master, 3D models define a source-faithful master
mapping. No MERIT implementation or dummy prediction is provided.

## Split contract

Seeded group assignment uses verified patient IDs or explicitly documented volume
proxies. Stable ID ordering makes results independent of manifest order. Groups
stay intact and are assigned to size-balanced buckets. Fold `i` tests bucket `i`,
validates `(i+1) mod 5`, trains on the rest (about 60/20/20). Each fold covers all
primary cases with disjoint roles; every volume is outer-tested once. This is not
nested hyperparameter search. Expected primary count defaults to 199.

JSON records seed, count, fold roles, grouping policy and SHA-256 identity/group
fingerprint. Loading checks fingerprint, coverage, duplicate IDs and group leakage.
The fingerprint does not hash CT bytes; record source revisions separately.
Exclusive creation protects existing experiments. Zenodo20/mixed manifests fail
development split creation and validation. Unverified linkage prevents a stronger
claim of verified patient independence.

## Reconstruction and evaluation

Group predictions by case. `reconstruct_volume` consumes `(slice_index,[H,W])`
pairs in any arrival order and requires each canonical slice exactly once. Missing,
duplicate, non-integer/out-of-range indexes, wrong XY shape and non-finite values
fail. `restore_prediction` reverses XY resize and orientation using the matching
PreparedVolume. Binary inverse resize is nearest; probability inverse resize is
linear. Threshold using a fixed internal-validation-selected threshold before
binary evaluation/export. No inverse step recreates lost lesion details.

`save_prediction` requires original-shape binary arrays, preserves original header
geometry/forms and refuses to overwrite. Source images/masks are never output
targets. Metrics consume aligned binary original-grid arrays; continuous values
are rejected. Callers explicitly convert logits to probabilities and threshold.

| GT | Prediction | Dice / IoU | Sensitivity | Precision |
| --- | --- | --- | --- | --- |
| empty | empty | 1 / 1 | 1 | 1 |
| empty | positive | 0 / 0 | 1 | 0 |
| positive | empty | 0 / 0 | 0 | 1 |

These are explicit zero-denominator conventions. Report empty/nonempty supports
in study tables. `evaluate_patients` pools TP/predicted/GT counts over volumes
linked to one verified patient before metric calculation; absent IDs use volume
proxies. `aggregate_patients` gives each patient equal weight, never averages slice
scores or voxel-weights patients. HD95, confidence intervals and lesion-volume
error remain roadmap work.
