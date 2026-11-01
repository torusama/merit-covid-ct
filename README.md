# COVID-19 CT Lesion Segmentation with MERIT-UNETR++

## Project objective

This project segments COVID-19 lung lesions from chest CT volumes. The target is a binary lesion mask:

| Label | Meaning |
| --- | --- |
| `0` | Background / non-lesion |
| `1` | COVID-19 lung lesion |

The central model is **MERIT-UNETR++**, not a replacement architecture. It receives an ordered thin slab of 11 CT slices and predicts the lesion mask of one designated target slice, called the **master slice**.

## Research question

Can master-aware inter-slice evidence routing improve COVID-19 lesion segmentation and generalize to an independent CT source better than a 2D segmentation model such as MADGNet?

## Data

| Dataset | Role | Cases | Labels |
| --- | --- | ---: | --- |
| `data/covid19_20/` | Primary development and internal evaluation | 199 | Binary COVID lesion mask |
| `data/zenodo20/` | Independent external test set | 20 | Binary COVID lesion mask, left/right lung masks, combined mask |

The two datasets are never merged before a random image split. All splits must be at patient/volume level.

## Model

### Baseline: MERIT-UNETR++

The original MERIT-UNETR++ components remain the backbone of the method:

1. **UNETR++ backbone** processes a 3D slab of 11 axial CT slices.
2. **TEP** aggregates ordered evidence from the slab.
3. **MER** routes context-slice evidence toward the master slice.
4. **PCG** uses slice-level lesion presence to calibrate the foreground prediction.
5. The output is the binary lesion mask for the master slice.

### Proposed extension: spatially aligned, spacing-aware MER

The proposed method remains MERIT-UNETR++, with a targeted extension inside the MER module.

The original MER represents context position by slice index. CT studies have different axial spacing, so the same 11-slice input can cover very different physical distances. The proposed MER should:

1. Encode each context slice using its physical distance from the master slice in millimetres.
2. Learn spatial correspondence between context features and master-slice features before routing evidence.
3. Gate the routed evidence at each spatial location according to cross-slice agreement.

This preserves the purpose of MERIT-UNETR++: use neighbouring slices to improve the designated master-slice segmentation. The extension prevents evidence from non-corresponding anatomy or distant slices from being treated as equally reliable.

## Why this differs from MADGNet

MADGNet is a strong 2D medical segmentation baseline. Its multi-frequency and multi-scale attention improve boundary extraction, but each CT slice is segmented independently. The proposed MERIT-UNETR++ variant uses 3D thin-slab context to verify lesion evidence across neighbouring slices while respecting CT geometry.

The published MADGNet result of 83.7 Dice and 76.8 mIoU uses a different 2D image-level split. It is a literature reference until MADGNet is rerun under this project's patient-level protocol.

## Experimental protocol

### Internal evaluation

- Use 5-fold cross-validation on the 199 COVID-19-20 cases.
- Keep all slices from one volume in one fold.
- Select thresholds and checkpoints using only the training/validation portion of each fold.

### External evaluation

- Train the selected model on the COVID-19-20 training data.
- Evaluate once on all 20 Zenodo20 volumes.
- Do not fine-tune, select thresholds, or modify the model with Zenodo20 labels.

### Required comparisons

1. 2D U-Net or UNet++ baseline.
2. MADGNet rerun with the same patient-level split, if implementation resources allow.
3. UNETR++ slab baseline without TEP, MER, or PCG.
4. Original MERIT-UNETR++.
5. MERIT-UNETR++ with physical-distance encoding.
6. Full spatially aligned, spacing-aware MERIT-UNETR++.

### Required ablations

| Variant | Purpose |
| --- | --- |
| MERIT baseline | Establish the original inter-slice routing effect. |
| + physical spacing | Test whether millimetre distance is more useful than slice index. |
| + spatial alignment | Test whether correspondence-aware routing improves the master slice. |
| + agreement gate | Test whether local cross-slice agreement reduces false positives. |

## Metrics

The primary metrics are computed per patient volume and then macro-averaged across patients.

\[
\mathrm{Dice} = \frac{2 |P \cap G|}{|P| + |G|}
\]

\[
\mathrm{IoU} = \frac{|P \cap G|}{|P \cup G|}
\]

where `P` is the predicted lesion mask and `G` is the ground-truth lesion mask.

Report:

- Mean patient-level 3D Dice and lesion IoU, with confidence intervals.
- Sensitivity and precision.
- HD95 on volumes/slices containing a lesion.
- Lesion-volume error.
- Results stratified by lesion burden and axial slice spacing.
- External-test performance on Zenodo20.

## Implementation milestones

1. Create patient-level 5-fold split files for COVID-19-20.
2. Implement CT preprocessing: HU clipping, normalization, resize/resample policy, and 11-slice slab sampling.
3. Build a PyTorch dataset loader that returns CT slab, master-slice index, physical Z positions, and master mask.
4. Implement Dice, IoU, HD95, lesion-volume error, and per-patient aggregation.
5. Train and validate the UNETR++ slab baseline.
6. Implement and test the original MERIT-UNETR++ modules.
7. Add the spacing-aware and spatial-alignment extension to MER.
8. Run internal cross-validation, external testing, and ablations.
9. Produce quantitative tables and qualitative CT/mask comparisons for the paper.

## Workspace layout

```text
data/
  zenodo20/       20-volume external dataset
  covid19_20/     199-volume primary dataset
archives/         Downloaded source archives
tools/            Dataset preparation, audit, and viewer scripts
analysis/         Dataset audit outputs
outputs/          Viewer PNG exports
references/       MADGNet source consulted for comparison
```

## Useful commands

```powershell
# View CT slices from either dataset
python tools/view_slices.py

# Rebuild the local dataset layouts if needed
python tools/prepare_zenodo20.py
python tools/download_covid19_20.py

# Audit the 20-case external dataset
python tools/audit_zenodo20.py
python tools/plot_zenodo20_audit.py
```
