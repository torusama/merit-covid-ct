# Integrated data contract

See [merge report](merge_report.md) for migration details and verification.

## Inputs and defaults

- CT/mask originals remain untouched on disk. NiBabel applies source slope/intercept.
- Strict paired 3D geometry, finite CT/mask and binary labels 0/1.
- Standardized array orientation LPS, without spatial interpolation/deobliquing.
- Native H,W and native intensity by default. No clipping, normalization or HU inference.
- No Z resampling. Slab API: D in 1/3/5/11, centered master; default D=11 and edge replication.
- Unknown spatial units fail unless the explicit agreed mm assumption is enabled.
- Optional window/resize requires explicit config; cannot silently activate on defaults.
- Primary-only development folds. External Zenodo20 remains external; no real external data was reprocessed.
- Patient identity is unverified for current cases; grouping guarantees apply to recorded IDs.

## Public APIs

Recommended Tensor API: `from merit_covid import PreprocessConfig, build_fold_loaders`.
The returned loaders expose `.train`, `.validation`, `.test`.

| Field | Tensor batch |
| --- | --- |
| image_slab | float32 [B,1,D,H,W] |
| master_mask | float32 binary [B,1,H,W] |
| master_slice_index | [B], zero-based LPS slice in the volume |
| slice_indices | [B,D], actual source slices after edge padding |
| slice_positions_mm | [B,D,3], centers in NiBabel RAS+ world mm |
| z_offsets_mm | [B,D], signed center offsets relative to master |
| metadata | list of B geometry/config dictionaries |

Main `SlabDataset` API returns NumPy `image [1,D,H,W]`, `mask [1,H,W]`,
`context_indices`, `z_offsets_mm`, `slice_index` and `master_index`.
`master_index` is the POSITION inside the slab (5 for D=11); `slice_index`
is the SLICE inside the volume. H=Y, W=X. The optional torch loader adds B.
Inference drops GT paths and omits mask. Default processing and sampling use the
same NumPy core underneath the Tensor API. Only explicit training augmentation
callbacks run augmentation; none is configured by default.

## Split and reconstruction

Canonical file: `splits/covid19_20_5fold.json`, unchanged from Duong.
Five deterministic outer test groups, validation 20% of the remainder.
Fold 0-3: 127/32/40 volumes; fold 4: 128/32/39.
Main `load_splits()` exposes list aliases including `internal_test`, validated
against the canonical data. No second split file is used.

Shared reconstructor expects indexed probabilities [H,W] in [0,1], rejects
missing/duplicate/invalid slices and returns the original XYZ layout after
inverse orientation. Default H,W preservation needs no prediction interpolation.
If resize is explicitly enabled, probability restoration precedes thresholding.

## Metrics

Whole-volume foreground Dice/IoU/precision/recall. Both empty yields 1; exactly
one empty yields 0 for these four metrics. HD95 is null if either mask is empty.
HD95 uses physical full-affine surface distances. Lesion volume uses voxel
volume in mm3. Mean volume metrics within each recorded group, then equal-weight
group macro; population std. Never average slice Dice and call it volume Dice.
Main `sensitivity` aliases Duong `recall`. Threshold default 0.5 remains a
configurable evaluator convention, not a validation-selected model threshold.

The data pipeline itself supplies inputs/targets and model-independent evaluation.
The separate [UNETR++ baseline](unetrpp_baseline.md) now supplies 3D logits and
an optional master-output adapter. Its fixed token grid does not accept the
native [B,1,11,512,512] batch directly. The separate [patch input adapter](unetrpp_input.md)
now crops the SAME reader's native CT to [B,1,64,128,128], with controlled padding
and a boolean valid_region. Optional mask_patch is actual aligned 3D GT, never
repeated master labels. Neither the slab loader nor model is modified. No
training/checkpoints/research performance are provided.

## Unified prediction output

`engine.prediction.VolumePrediction` contains case_id, probability_zyx float32
[Z,H,W] in [0,1] on the complete native-size LPS CT grid, and metadata for
original NIfTI restoration. It rejects resized/incomplete/mismatched grids.
`prediction_policy` records model-specific preparation and output policy.

PatchReconstructor removes padding by coordinates, requires every planned patch
exactly once, rejects wrong case/geometry, and computes a uniform arithmetic mean
of probabilities at overlap voxels. Sigmoid is applied outside the model, before
assembly; no threshold is applied. `prediction_from_slices()` uses the EXISTING
SliceReconstructor for master-slice/2D models and returns the same output type.

`to_original_xyz()` delegates to the existing restore_volume(). Alternatively,
`slice_predictions()` yields the interface for the existing VolumeEvaluator,
which restores probabilities before thresholding on the original CT grid.
No new evaluator or metric convention is introduced. Same case partitions,
original GT, evaluator, validation-selected threshold policy and volume/group
aggregation remain required for comparisons. Different model input shapes are
allowed and recorded; training choices are not made by this contract.

`models.base.master_slice_input()` accepts either batch convention; for a Duong slab it selects D//2, independently of the volume slice index.
