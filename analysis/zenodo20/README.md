# Zenodo20 dataset audit

Run from the workspace root:

```powershell
python tools/prepare_zenodo20.py
python tools/audit_zenodo20.py
python tools/plot_zenodo20_audit.py
```

`summary.json` contains aggregate statistics, `case_statistics.csv` contains one row per CT volume, and the two PNG files show case balance and example lesion trajectories. The viewer exports preview PNGs to `outputs/`.

The CT volumes were extracted from the Kaggle mirror of the 20-case Zenodo dataset. The infection and lung masks were extracted from the original Zenodo ZIP files. The combined Kaggle mask equals the original infection mask at label 3, and its nonzero region equals the union of original lung and infection masks in all 20 volumes. CT image values have **not** been proven byte-identical to Zenodo's large CT ZIP.

Definitions used by the analysis:

- Positive slice: at least one nonzero voxel in the original `Infection_Mask`.
- Infection fraction of lung: count of infection voxels divided by count of voxels in the original `Lung_Mask`. There are 114 infection voxels outside that lung mask in three Radiopaedia volumes.
- Adjacent Dice: Dice between infection masks of neighboring slices, computed only when both masks are positive. It measures ground-truth continuity, not model performance.
- Component: 8-connected foreground region on a single axial infection mask.
- Resize loss: a component whose label disappears after OpenCV nearest-neighbor resize of the entire original slice to 128, 256, or 352 square pixels. This is a preprocessing sensitivity check; other interpolation, crop, or threshold choices can differ.
- CT intensity percentiles: sampled on a regular grid with stride (8, 8, 4) for each volume, then pooled within Coronacases or Radiopaedia. These are not exact whole-volume percentiles.
- Geometry match: image, infection, lung, and combined-mask shape, affine, and voxel spacing agree for each volume.

The copied MADGNet source in `references/madgnet` was used only to inspect evaluation code and split CSV files. No model was trained in this audit.
