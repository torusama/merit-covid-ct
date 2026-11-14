# Spacing ablation: frozen shared protocol

E1 is `merit_index.yaml`. E2–E4 are `merit_spacing_e2.yaml`,
`merit_spacing_e3.yaml` and `merit_spacing_e4.yaml`; they start with the **same** preprocessing,
model, loss and training settings as E1. `merit_physical.yaml` is a syntax
template, but differs in `assume_mm`; do not compare it unchanged to E1.

Retain the same split, unit
policy, intensity and seed, and make only the changes listed below. Supply
values selected on training/validation data, never external or test data.
The baseline's `assume_mm: true` is an unverified assumption for three cases;
resolve or document it before using physical distances. The values below are
placeholders, not recommendations. Do not run until they are replaced.

| Experiment | `preprocess.sampling_mode` | `preprocess.target_spacing_mm` | `model.position_mode` | `loss.weighting_mode` |
| --- | --- | --- | --- | --- |
| E1 | `index` | null | `index` | `index` |
| E2 | `physical_mm` | `<d_mm>` | `index` | `index` |
| E3 | `physical_mm` | `<d_mm>` | `physical_mm` | `index` |
| E4 | `physical_mm` | `<d_mm>` | `physical_mm` | `physical_mm` |

For E3/E4 also explicitly freeze `model.position_unit_mm`; E4 requires
`loss.sigma_mm`. Choose these independently and record rationale. Log actual
offsets, duplicate identities and available context, not only target spacing.
No Z interpolation or acquisition slice-thickness correction is implied.
