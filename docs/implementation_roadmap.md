> Historical main snapshot before MERGE. Current integration and verification: [merge_report.md](merge_report.md).

# Implementation roadmap

## Completed

Audited source and historical analysis; packaged validated config, unified/legacy
manifests, strict geometry/binary I/O, paired RAS orientation, explicit intensity
policy, optional XY resize/inverse geometry, slab sampling, grouped immutable
folds, Dataset modes, optional torch DataLoader, model adapter protocol,
reconstruction, patient metrics, synthetic/mocked tests and English docs. Repaired
legacy mask paths while preserving source data and historical analysis.

No MERIT model, trainer, checkpoint, validated augmentation or performance claim
is included. Real torch execution needs the optional dependency.

## Next work by priority

| Priority | Suggested owner | Task and completion evidence |
| --- | --- | --- |
| P0 | Data lead | Verify primary patient linkage. Save a newly named split if linkage changes; verify its fingerprint. Do not claim patient independence from volume names. |
| P0 | Data lead | Establish primary HU/windowed provenance, spatial units, original-source revision and labels. Run full primary QC, including small-lesion retention and internal-only normalization/resize selection. |
| P0 | Supervisor / model lead | Supply authorized MERIT source, license, paper/specification, commit and checkpoint provenance. Confirm original UNETR++/TEP/MER/PCG definitions and losses. |
| P1 | Model lead | Integrate original modules at models/merit_unetrpp; test true master-output semantics through the common adapter without introducing new routing/gates. |
| P1 | Training lead | Install appropriate torch build; verify CPU DataLoader, Windows workers and Linux/Colab. Profile RAM/cache before full runs. Add source-faithful 2D U-Net/UNETR++ adapters. |
| P1 | Training lead | Add trainer, losses, optimizer/scheduler, validation/checkpoint selection, logs, resume and complete config/split/commit provenance. Training requires a separate user request. |
| P2 | Evaluation lead | Add geometry-aware HD95 with defined empty cases, lesion-volume error, bootstrap intervals and reporting supports. Verify real model export on original grids. |
| P2 | Baseline owners | Integrate MADGNet and nnU-Net with appropriate planning/export; rerun matched patient/volume protocols rather than treating incompatible literature splits as measurements. |
| P2 | Evaluation lead | Freeze model/preprocessing/threshold using primary data, then perform final Zenodo20 evaluation without tuning from external labels. |
| P3 | Research lead | After source-faithful reproduction, study MER spacing and PCG effects through controlled follow-up ablations with prespecified hypotheses. |

## Original-model integration requirements

Confirm backbone version, channels, input depth/patch divisibility, axis layout,
position encoding, TEP aggregation, MER routing, PCG calibration, auxiliary labels
and losses, deep supervision, padding/cropping and master mapping. Map source
outputs to `[B,1,H,W]` without changing the algorithm. Document adapter-only layout
changes. Establish fidelity with original tests/checkpoints before claiming
architecture reproduction.

Physical offsets currently support provenance and later analysis; they do not
implement spacing-aware MER. Reliability gates, spatial correspondence modules
and new routing are outside this baseline.
