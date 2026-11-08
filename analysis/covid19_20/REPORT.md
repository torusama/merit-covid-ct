# COVID-19-20 dataset report

Generated: 2026-11-07T16:27:47+07:00. Descriptive CT/ground-truth evidence,
not model predictions.

## Processing and provenance

- Arrays are standardized to **LPS** by axis permutation/flips, not spatial interpolation.
- GT statistics use native resolution; no Z resampling is performed.
- Saved audit intensity policy: `native`;
  HU window: `null`.
  Native mode preserves NIfTI-scaled values without clipping, normalization or HU inference.
- PNG grayscale is display-only; it does not change network input intensities.
- Canonical metadata refers to the LPS array grid. NiBabel world coordinates remain RAS+.
- Headers and full masks were rechecked against the saved audit. Full CT intensity/finite
  checks were reused from that audit; this report does not certify source byte identity.

## Artifacts

| File | Meaning |
| --- | --- |
| [case_statistics.csv](case_statistics.csv) | One row per CT volume; 199 data rows |
| [summary.json](summary.json) | Cohort aggregates and per-case GT area trajectories |
| [dataset_overview.png](dataset_overview.png) | Slice spacing, volume length, lesion fraction and positive/empty slices |
| [lesion_trajectories.png](lesion_trajectories.png) | Native GT lesion area across physical stack distance |
| [ct_mask_examples.png](ct_mask_examples.png) | Middle-slice CT, GT and overlay, not AI predictions |

The complete 71-field schema and worked A-0025 example are in
[the dataset-report guide](../../docs/dataset_reports.md).

## Measured cohort statistics

- 199 CT volumes; the number of distinct clinical patients is unverified.
- 13,705 canonical slices: 4,981 with GT lesion,
  8,724 with empty GT.
- Volume length: 39 to 361 slices;
  median 63.
- 14,818,707 lesion voxels / 3,592,683,520 total CT voxels.
- Pooled lesion / whole-CT fraction: 0.4125%.
- Mean per-volume lesion fraction: 0.4378%.
  Pooled and macro summaries have different weighting and need not agree.
- Per-case native lesion counts agree with the saved audit.

| Original Z spacing (mm; assumptions apply where units are missing) | Volumes |
| --- | ---: |
| 0.799988 | 1 |
| 1 | 4 |
| 2 | 6 |
| 5 | 188 |

## Interpretation and limitations

- Current groups fall back to volume IDs. No verified patient-level independence.
  The report neither changes splits nor selects training cases.
- Unknown-unit cases explicitly assumed to be mm: `volume-covid19-A-0161`, `volume-covid19-A-0585`, `volume-covid19-A-0599`.
  Their spacing, area and volume measurements depend on that assumption.
- No primary lung mask is available. Lesion fraction divides by the entire CT grid,
  including air outside the body; it is not percentage of infected lung.
- Adjacent GT Dice describes neighboring-label continuity, not model accuracy.
- This historical native-mask cohort report does not measure component loss under
  resizing. The newer runtime lesion-retention checker is documented separately.
- Saved scaled intensity range: [-10380.0, 32737.0].
  This is not proof of HU calibration.
- Examples use the first sorted case ID per spacing category and its middle slice,
  independently of GT; an example mask may be empty.
- Clinical linkage, intensity provenance and source byte identity still require verification.

## Regeneration

From the repository root, with the dataset and full audit already available:

~~~powershell
py tools/summarize_covid19_20.py --assume-mm
~~~

The flag deliberately accepts an unverified unit assumption. The command rechecks
headers/masks, reuses saved CT intensity checks and writes primary report artifacts.
CSV uses UTF-8 BOM for spreadsheet compatibility. Cases are processed one at a time.
No training, original voxel edits, external download or Zenodo20 analysis is performed.
