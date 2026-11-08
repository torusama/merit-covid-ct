# Native patch input and unified volume output

The standalone baseline consumes native 3D crops rather than resized whole
slices. It reuses the shared LPS/intensity/geometry reader and does not alter the
slab Dataset or UNETR++ architecture. Model background:
[unetrpp_baseline.md](unetrpp_baseline.md).

## A real-volume example

A-0025 has canonical shape `[Z,Y,X]=[61,512,512]`.
Reference patch size is `(64,128,128)`, default stride the same.
Four Y starts and four X starts (0/128/256/384) produce 16 patches.
Each contains 61 real CT slices plus 3 trailing Z-padding slices.

Crop preserves native voxel values/spacing; it is not 512->128 resize.
All windows cover the volume independently of lesion masks.
One forwarded patch is not a complete CT prediction.

## Window and padding policy

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

## Dataset, split and batch contract

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

## Forward a real patch

```powershell
py examples/unetrpp_real_input.py
py examples/unetrpp_real_input.py --report outputs/real_input_check.json
py examples/unetrpp_real_input.py --workers 1 --patch-index 15
```

The example defaults to one A-0025 train patch with the full 42,951,355-parameter
model in eval/inference mode. It checks finite output, source voxel agreement
and unchanged parameter hashes; no backward, optimizer, GT-based selection or
clinical quality evaluation occurs.

Historical observed input: `[1,1,64,128,128]`, 999,424 real voxels and 49,152
padding voxels. GT was not read and CT source size/mtime remained unchanged.
This proves a software connection, not model accuracy.

## Complete probability reconstruction

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

## Shared output and evaluation

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

## Source and tests

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
