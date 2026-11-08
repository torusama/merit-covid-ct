# Dataset reports and CSV reference

Stored statistics describe CT and ground-truth masks, not model predictions.
They are not mandatory Dataset inputs. Original NIfTI files plus a manifest and
the saved split supply training tensors.

## Primary report

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

## Complete saved primary CSV schema

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

## Worked example: A-0025

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

## Historical Zenodo20 report and provenance

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

## Regeneration and safe interpretation

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
